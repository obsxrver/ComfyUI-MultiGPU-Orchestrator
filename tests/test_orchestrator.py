import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orchestrator import (  # noqa: E402
    PARENT_PID_ENV,
    MultiGpuOrchestrator,
    WorkerState,
    aggregate_assets_payload,
    aggregate_jobs_payload,
    aggregate_tags_payload,
    build_queue_info,
    build_worker_command,
    create_direct_prompt_routing_middleware,
    load_config,
    parse_device_list,
    read_log_chunk,
    reset_worker_logs,
    route_with_query,
    save_config,
    synthesize_jobs_payload,
    worker_log_path,
)


class OrchestratorPureTests(unittest.TestCase):
    def test_parse_device_list(self):
        self.assertEqual(parse_device_list("0, 2,3"), [0, 2, 3])
        self.assertIsNone(parse_device_list(""))
        self.assertIsNone(parse_device_list(None))

    def test_route_with_query_preserves_repeated_values(self):
        class FakeQuery:
            def keys(self):
                return ["include_tags", "wait"]

            def getall(self, key):
                return {
                    "include_tags": ["output", "video"],
                    "wait": ["true"],
                }[key]

        route = route_with_query("/api/assets/seed", FakeQuery())
        self.assertIn("include_tags=output", route)
        self.assertIn("include_tags=video", route)
        self.assertIn("wait=true", route)

    def test_build_worker_command_preserves_core_flags(self):
        args = SimpleNamespace(
            base_directory="/comfy/base",
            output_directory="/comfy/output",
            input_directory="/comfy/input",
            temp_directory="/comfy/temp",
            user_directory="/comfy/user",
            extra_model_paths_config=["/comfy/extra.yaml"],
            enable_assets=True,
        )

        command = build_worker_command(
            2,
            9123,
            python_executable="/python",
            comfy_root=Path("/comfy"),
            comfy_args=args,
        )

        self.assertEqual(command[:8], [
            "/python",
            str(Path("/comfy") / "main.py"),
            "--listen",
            "127.0.0.1",
            "--port",
            "9123",
            "--cuda-device",
            "2",
        ])
        self.assertIn("--disable-auto-launch", command)
        self.assertIn("--base-directory", command)
        self.assertIn("/comfy/base", command)
        self.assertIn("--extra-model-paths-config", command)
        self.assertIn("/comfy/extra.yaml", command)
        self.assertIn("--disable-assets", command)
        self.assertNotIn("--enable-assets", command)

    def test_worker_log_path_is_stable_per_gpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            previous = os.environ.get("COMFYUI_MGPU_LOG_DIR")
            os.environ["COMFYUI_MGPU_LOG_DIR"] = tmpdir
            try:
                path = worker_log_path(Path("/comfy"), gpu_index=2)
            finally:
                if previous is None:
                    os.environ.pop("COMFYUI_MGPU_LOG_DIR", None)
                else:
                    os.environ["COMFYUI_MGPU_LOG_DIR"] = previous

        self.assertEqual(path.name, "gpu-2.log")
        self.assertEqual(path.parent, Path(tmpdir))

    def test_reset_worker_logs_removes_old_logs_and_initializes_each_gpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            old_log = log_dir / "gpu-0-port-9000-20260708-153000-pid-100.log"
            old_log.write_text("old", encoding="utf-8")
            stable_log = log_dir / "gpu-0.log"
            stable_log.write_text("previous run", encoding="utf-8")
            unrelated = log_dir / "README.txt"
            unrelated.write_text("keep", encoding="utf-8")
            previous = os.environ.get("COMFYUI_MGPU_LOG_DIR")
            os.environ["COMFYUI_MGPU_LOG_DIR"] = tmpdir
            try:
                paths = reset_worker_logs(Path("/comfy"), [0, 2])
            finally:
                if previous is None:
                    os.environ.pop("COMFYUI_MGPU_LOG_DIR", None)
                else:
                    os.environ["COMFYUI_MGPU_LOG_DIR"] = previous

            self.assertEqual(paths, {0: log_dir / "gpu-0.log", 2: log_dir / "gpu-2.log"})
            self.assertFalse(old_log.exists())
            self.assertEqual((log_dir / "gpu-0.log").read_text(encoding="utf-8"), "")
            self.assertEqual((log_dir / "gpu-2.log").read_text(encoding="utf-8"), "")
            self.assertTrue(unrelated.exists())

    def test_read_log_chunk_tails_then_reads_incrementally_and_resets_after_truncation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gpu-0.log"
            path.write_bytes(b"old\nrecent")

            initial = read_log_chunk(path, None, max_bytes=6)
            self.assertEqual(initial, {"content": "recent", "cursor": 10, "reset": True})

            with path.open("ab") as log_file:
                log_file.write(b"\nnext")
            incremental = read_log_chunk(path, initial["cursor"], max_bytes=20)
            self.assertEqual(incremental, {"content": "\nnext", "cursor": 15, "reset": False})

            path.write_bytes(b"new")
            truncated = read_log_chunk(path, incremental["cursor"], max_bytes=20)
            self.assertEqual(truncated, {"content": "new", "cursor": 3, "reset": True})

    def test_select_worker_uses_least_busy_with_round_robin_tie(self):
        orchestrator = MultiGpuOrchestrator(prompt_server=SimpleNamespace())
        worker0 = WorkerState(gpu_index=0, port=9000, url="http://127.0.0.1:9000", status="healthy")
        worker1 = WorkerState(gpu_index=1, port=9001, url="http://127.0.0.1:9001", status="healthy")
        worker2 = WorkerState(gpu_index=2, port=9002, url="http://127.0.0.1:9002", status="healthy")
        worker0.running = 1
        worker1.pending = 0
        worker2.pending = 0
        orchestrator.workers = [worker0, worker1, worker2]

        self.assertIs(orchestrator.select_worker(), worker1)
        self.assertIs(orchestrator.select_worker(), worker2)

    def test_select_worker_returns_none_without_healthy_workers(self):
        orchestrator = MultiGpuOrchestrator(prompt_server=SimpleNamespace())
        orchestrator.workers = [
            WorkerState(gpu_index=0, port=9000, url="http://127.0.0.1:9000", status="failed")
        ]

        self.assertIsNone(orchestrator.select_worker())

    def test_aggregate_jobs_filters_sorts_and_paginates(self):
        payload = aggregate_jobs_payload(
            [
                {
                    "jobs": [
                        {"id": "old", "status": "completed", "create_time": 10},
                        {"id": "running", "status": "in_progress", "create_time": 30},
                    ]
                },
                {"jobs": [{"id": "new", "status": "completed", "create_time": 40}]},
            ],
            status_filter=["completed"],
            limit=1,
            offset=0,
        )

        self.assertEqual([job["id"] for job in payload["jobs"]], ["new"])
        self.assertEqual(payload["pagination"]["total"], 2)
        self.assertTrue(payload["pagination"]["has_more"])

    def test_aggregate_assets_deduplicates_and_paginates(self):
        payload = aggregate_assets_payload(
            [
                {
                    "assets": [
                        {"id": "a", "created_at": "2026-01-01T00:00:00Z"},
                        {"id": "b", "created_at": "2026-01-03T00:00:00Z"},
                    ]
                },
                {
                    "assets": [
                        {"id": "a", "created_at": "2026-01-01T00:00:00Z"},
                        {"id": "c", "created_at": "2026-01-02T00:00:00Z"},
                    ]
                },
            ],
            limit=2,
            offset=0,
        )

        self.assertEqual([asset["id"] for asset in payload["assets"]], ["b", "c"])
        self.assertEqual(payload["total"], 3)
        self.assertTrue(payload["has_more"])

    def test_aggregate_tags_merges_counts_and_paginates(self):
        payload = aggregate_tags_payload(
            [
                {
                    "tags": [
                        {"name": "output", "type": "system", "count": 2},
                        {"name": "video", "type": "media", "count": 1},
                    ]
                },
                {
                    "tags": [
                        {"name": "output", "type": "system", "count": 3},
                        {"name": "image", "type": "media", "count": 4},
                    ]
                },
            ],
            limit=2,
            offset=0,
        )

        self.assertEqual(payload["tags"][0], {"name": "output", "type": "system", "count": 5})
        self.assertEqual(payload["tags"][1], {"name": "image", "type": "media", "count": 4})
        self.assertEqual(payload["total"], 3)
        self.assertTrue(payload["has_more"])

    def test_synthesize_jobs_from_legacy_queue_and_history(self):
        payload = synthesize_jobs_payload(
            [
                {
                    "queue_running": [
                        [1, "running-id", {"1": {"class_type": "KSampler"}}, {"create_time": 30}, ["9"]]
                    ],
                    "queue_pending": [
                        [2, "pending-id", {"1": {"class_type": "KSampler"}}, {"create_time": 20}, ["9"]]
                    ],
                }
            ],
            [
                {
                    "done-id": {
                        "prompt": [
                            3,
                            "done-id",
                            {"9": {"class_type": "SaveImage"}},
                            {"create_time": 10, "extra_pnginfo": {"workflow": {"nodes": []}}},
                            ["9"],
                        ],
                        "outputs": {
                            "9": {
                                "images": [
                                    {"filename": "ComfyUI_00001_.png", "subfolder": "", "type": "output"}
                                ]
                            }
                        },
                        "status": {"status_str": "success"},
                    }
                }
            ],
            limit=10,
        )

        jobs = {job["id"]: job for job in payload["jobs"]}
        self.assertEqual(jobs["running-id"]["status"], "in_progress")
        self.assertEqual(jobs["pending-id"]["status"], "pending")
        self.assertEqual(jobs["done-id"]["status"], "completed")
        self.assertEqual(jobs["done-id"]["outputs_count"], 1)
        self.assertEqual(jobs["done-id"]["preview_output"]["mediaType"], "image")

    def test_build_queue_info_matches_comfy_status_shape_inner_payload(self):
        worker0 = WorkerState(gpu_index=0, port=9000, url="http://127.0.0.1:9000", status="healthy")
        worker1 = WorkerState(gpu_index=1, port=9001, url="http://127.0.0.1:9001", status="healthy")
        worker0.running = 1
        worker1.pending = 2

        payload = build_queue_info([worker0, worker1])

        self.assertEqual(payload["exec_info"]["queue_remaining"], 3)
        self.assertEqual(payload["mgpu"]["running"], 1)
        self.assertEqual(payload["mgpu"]["pending"], 2)

    def test_config_defaults_auto_start_enabled_and_persists_toggle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "mgpu_config.json"
            old_value = os.environ.get("COMFYUI_MGPU_CONFIG")
            os.environ["COMFYUI_MGPU_CONFIG"] = str(config_file)
            try:
                self.assertEqual(
                    load_config(),
                    {
                        "auto_start": True,
                        "auto_respawn": True,
                        "requeue_pending_on_respawn": True,
                    },
                )
                save_config(
                    {
                        "auto_start": False,
                        "auto_respawn": False,
                        "requeue_pending_on_respawn": False,
                    }
                )
                self.assertEqual(
                    load_config(),
                    {
                        "auto_start": False,
                        "auto_respawn": False,
                        "requeue_pending_on_respawn": False,
                    },
                )
            finally:
                if old_value is None:
                    os.environ.pop("COMFYUI_MGPU_CONFIG", None)
                else:
                    os.environ["COMFYUI_MGPU_CONFIG"] = old_value


class FakePromptServer:
    def __init__(self):
        self.sent = []

    async def send(self, message_type, data, client_id):
        self.sent.append((message_type, data, client_id))


class FakeSession:
    def __init__(self):
        self.calls = []

    async def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return SimpleNamespace(status=200)


class FakeProcess:
    def __init__(self, returncode=1, pid=1234):
        self.returncode = returncode
        self.pid = pid
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9


class FakeResponse:
    def __init__(self, body=b'{"prompt_id":"new-prompt"}', status=200):
        self.body = body
        self.status = status

    async def read(self):
        return self.body


class OrchestratorAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_standard_jobs_aggregate_primary_and_mixed_worker_apis(self):
        from contextlib import AsyncExitStack

        async with AsyncExitStack() as stack:
            orchestrator = MultiGpuOrchestrator(prompt_server=FakePromptServer())
            orchestrator._started = True
            orchestrator.sync_history = AsyncMock()
            orchestrator.start = AsyncMock()
            orchestrator._session = await stack.enter_async_context(aiohttp.ClientSession())
            primary_jobs = [
                {"id": "completed-primary", "status": "completed", "create_time": 10, "outputs_count": 2},
                {"id": "local", "status": "in_progress", "create_time": 40, "workflow_id": "wf"},
            ]
            worker_jobs = [
                {"id": "running", "status": "in_progress", "create_time": 30, "workflow_id": "wf"},
                {"id": "pending", "status": "pending", "create_time": 20, "workflow_id": "wf"},
                {"id": "completed-primary", "status": "in_progress", "create_time": 10},
                {"id": "worker-completed", "status": "completed", "create_time": 5},
            ]
            modern = web.Application()

            async def jobs(request):
                self.assertEqual(request.query.get("offset"), "0")
                self.assertNotIn("limit", request.query)
                return web.json_response({"jobs": worker_jobs})

            async def detail(request):
                job = next((job for job in worker_jobs if job["id"] == request.match_info["job_id"]), None)
                return web.json_response(job or {"error": "missing"}, status=200 if job else 404)

            modern.router.add_get("/api/jobs", jobs)
            modern.router.add_get("/api/jobs/{job_id}", detail)
            legacy = web.Application()

            async def queue(request):
                return web.json_response({
                    "queue_running": [[0, "legacy", {}, {"create_time": 50, "workflow_id": "wf"}, []]],
                    "queue_pending": [],
                })

            legacy.router.add_get("/queue", queue)
            for index, app in enumerate((modern, legacy)):
                server = await stack.enter_async_context(TestServer(app))
                orchestrator.workers.append(WorkerState(
                    gpu_index=index, port=server.port, url=str(server.make_url("/")).rstrip("/"), status="healthy",
                ))
            parent = web.Application(middlewares=[create_direct_prompt_routing_middleware(orchestrator)])

            async def native_jobs(request):
                query = request.query
                try:
                    limit = int(query["limit"]) if "limit" in query else None
                    offset = int(query.get("offset", 0))
                    if limit is not None and limit <= 0:
                        raise ValueError()
                except ValueError:
                    return web.json_response({"error": "invalid pagination"}, status=400)
                return web.json_response(aggregate_jobs_payload(
                    [{"jobs": primary_jobs}], limit=limit, offset=max(offset, 0),
                    status_filter=query["status"].split(",") if "status" in query else None,
                    workflow_id=query.get("workflow_id"), sort_order=query.get("sort_order", "desc"),
                ))

            async def native_detail(request):
                job = next((job for job in primary_jobs if job["id"] == request.match_info["job_id"]), None)
                return web.json_response(job or {"error": "missing"}, status=200 if job else 404)

            for prefix in ("", "/api", "/api/api"):
                parent.router.add_get(f"{prefix}/jobs", native_jobs)
                parent.router.add_get(f"{prefix}/jobs/{{job_id}}", native_detail)
            client = await stack.enter_async_context(TestClient(TestServer(parent)))
            for prefix in ("", "/api", "/api/api"):
                response = await client.get(f"{prefix}/jobs")
                self.assertEqual(response.status, 200)
                payload = await response.json()
                self.assertEqual([job["id"] for job in payload["jobs"]],
                                 ["legacy", "local", "running", "pending", "completed-primary"])
                self.assertEqual(payload["pagination"], {"total": 5, "offset": 0, "limit": None, "has_more": False})
                self.assertEqual(payload["jobs"][-1], primary_jobs[0])
                response = await client.get(f"{prefix}/jobs/running")
                self.assertEqual(await response.json(), worker_jobs[0])
                response = await client.get(f"{prefix}/jobs/completed-primary")
                self.assertEqual(await response.json(), primary_jobs[0])
            response = await client.get("/api/jobs?status=in_progress&workflow_id=wf&sort_order=asc&limit=1&offset=1")
            payload = await response.json()
            self.assertEqual([job["id"] for job in payload["jobs"]], ["local"])
            self.assertEqual(payload["pagination"], {"total": 3, "offset": 1, "limit": 1, "has_more": True})
            response = await client.get("/api/jobs/legacy")
            self.assertEqual((await response.json())["status"], "in_progress")
            response = await client.get("/api/jobs/unknown")
            self.assertEqual(response.status, 404)
            response = await client.get("/api/jobs?status=completed")
            self.assertEqual((await response.json())["jobs"], [primary_jobs[0]])
            for query in ("limit=0", "limit=bad", "offset=bad"):
                response = await client.get(f"/api/jobs?{query}")
                self.assertEqual(response.status, 400)
            orchestrator._fetch_worker_json_with_query = AsyncMock(side_effect=OSError("offline"))
            orchestrator._fetch_worker_json = AsyncMock(side_effect=OSError("offline"))
            response = await client.get("/api/jobs")
            self.assertEqual(response.status, 200)
            self.assertEqual(len((await response.json())["jobs"]), 2)
            # A failed/stopped pool preserves primary results.
            for worker in orchestrator.workers:
                worker.status = "stopped"
            response = await client.get("/api/jobs")
            self.assertEqual(len((await response.json())["jobs"]), 2)
            orchestrator.start.assert_not_awaited()

    async def test_standard_jobs_without_workers_or_startup_preserve_native_response(self):
        orchestrator = MultiGpuOrchestrator(prompt_server=FakePromptServer())
        orchestrator.sync_history = AsyncMock()
        app = web.Application(middlewares=[create_direct_prompt_routing_middleware(orchestrator)])

        async def native(request):
            return web.json_response({"native": request.path}, headers={"X-Native": "kept"})

        app.router.add_get("/api/jobs", native)
        app.router.add_get("/api/jobs/{job_id}", native)
        async with TestClient(TestServer(app)) as client:
            for started, workers in ((False, []), (True, []), (False, [WorkerState(0, 9000, "http://worker")])):
                orchestrator._started, orchestrator.workers = started, workers
                for route in ("/api/jobs", "/api/jobs/id"):
                    response = await client.get(route)
                    self.assertEqual(await response.json(), {"native": route})
                    self.assertEqual(response.headers["X-Native"], "kept")
        orchestrator.sync_history.assert_not_awaited()

    async def test_cancel_endpoints_reach_workers_without_browser_rewriting(self):
        from contextlib import AsyncExitStack

        async with AsyncExitStack() as stack:
            orchestrator = MultiGpuOrchestrator(prompt_server=FakePromptServer())
            orchestrator._session = await stack.enter_async_context(aiohttp.ClientSession())
            queues = []
            for index in range(2):
                running, pending, untouched = (str(uuid4()) for _ in range(3))
                active = {running, pending, untouched}
                queues.append((active, running, pending, untouched))
                app = web.Application()

                async def cancel(request, active=active):
                    job_id = request.match_info.get("job_id")
                    ids = [job_id] if job_id else (await request.json())["job_ids"]
                    cancelled = bool(active.intersection(ids))
                    active.difference_update(ids)
                    return web.json_response({"cancelled": cancelled})

                async def queue(request, active=active, running=running):
                    return web.json_response({
                        "queue_running": [[0, jid, {}, {}, []] for jid in active if jid == running],
                        "queue_pending": [[1, jid, {}, {}, []] for jid in active if jid != running],
                    })

                app.router.add_post("/api/jobs/cancel", cancel)
                app.router.add_post("/api/jobs/{job_id}/cancel", cancel)
                app.router.add_get("/queue", queue)
                server = await stack.enter_async_context(TestServer(app))
                worker = WorkerState(gpu_index=index, port=server.port, url=str(server.make_url("/")).rstrip("/"))
                worker.pending_prompt_payloads = {jid: {} for jid in active}
                worker.accepted_prompt_ids = set(active)
                orchestrator.workers.append(worker)

            parent = web.Application(middlewares=[create_direct_prompt_routing_middleware(orchestrator)])
            native = AsyncMock(return_value=web.json_response({"cancelled": False}))
            parent.router.add_post("/api/jobs/cancel", native)
            parent.router.add_post("/api/jobs/{job_id}/cancel", native)
            client = await stack.enter_async_context(TestClient(TestServer(parent)))

            # No ownership map: single cancel must still find worker 1.
            response = await client.post(f"/api/jobs/{queues[1][1]}/cancel")
            self.assertEqual(response.status, 200)
            self.assertEqual(await response.json(), {"cancelled": True})
            self.assertNotIn(queues[1][1], queues[1][0])
            response = await client.post("/api/jobs/cancel", json={
                "job_ids": [queues[0][1], queues[0][2], queues[1][2], str(uuid4())],
            })
            self.assertEqual(response.status, 200)
            self.assertEqual(await response.json(), {"cancelled": True})
            for worker, (active, running, pending, untouched) in zip(orchestrator.workers, queues):
                self.assertEqual(active, {untouched})
                self.assertEqual(set(worker.pending_prompt_payloads), {untouched})
                self.assertEqual(worker.accepted_prompt_ids, {untouched})
                self.assertEqual(worker.running, 0)
                self.assertEqual(worker.pending, 1)
            response = await client.post(f"/api/jobs/{queues[1][1]}/cancel")
            self.assertEqual(await response.json(), {"cancelled": False})
            native.assert_not_awaited()
            self.assertEqual(orchestrator.prompt_server.sent[-1][0], "status")

    async def test_cancel_validates_batch_before_contacting_workers(self):
        orchestrator = MultiGpuOrchestrator(prompt_server=FakePromptServer())
        orchestrator._post_worker = AsyncMock()
        for payload in (None, [], {}, {"job_ids": "all"}, {"job_ids": [str(uuid4()), {}]}, {"job_ids": [1]}):
            response = await orchestrator.proxy_job_cancel(SimpleNamespace(json=AsyncMock(return_value=payload)))
            self.assertEqual(response.status, 400)
        response = await orchestrator.proxy_job_cancel(SimpleNamespace(json=AsyncMock(side_effect=ValueError)))
        self.assertEqual(response.status, 400)
        response = await orchestrator.proxy_job_cancel(SimpleNamespace(json=AsyncMock(return_value={"job_ids": []})))
        self.assertEqual(json.loads(response.body), {"cancelled": False})
        orchestrator._post_worker.assert_not_awaited()

    async def test_cancel_reports_partial_and_total_worker_failures(self):
        orchestrator = MultiGpuOrchestrator(prompt_server=FakePromptServer())
        orchestrator.workers = [WorkerState(gpu_index=i, port=9000+i, url="http://worker") for i in range(2)]
        orchestrator._fetch_worker_json = AsyncMock(return_value={"queue_running": [], "queue_pending": []})
        orchestrator.start = AsyncMock()
        for first_response in (FakeResponse(b'{"cancelled":true}'), FakeResponse(b'{"cancelled":false}'), FakeResponse(b'{}', 404)):
            orchestrator._post_worker = AsyncMock(side_effect=[first_response, OSError("worker offline")])
            response = await orchestrator.proxy_job_cancel(SimpleNamespace(), prompt_id=str(uuid4()))
            self.assertEqual(response.status, 502)
            result = json.loads(response.body)
            self.assertEqual(result["cancelled"], first_response.body == b'{"cancelled":true}')
            self.assertTrue(result["workers"])
            self.assertEqual(orchestrator._post_worker.await_count, 2)
        orchestrator.start.assert_not_awaited()

    async def test_direct_cancel_middleware_handles_aliases_and_preserves_other_routes(self):
        orchestrator = SimpleNamespace(proxy_job_cancel=AsyncMock(return_value="cancelled"))
        middleware = create_direct_prompt_routing_middleware(orchestrator)
        native = AsyncMock(return_value="native")
        for prefix in ("", "/api", "/api/api"):
            for suffix, prompt_id in (("cancel", None), ("job-id/cancel", "job-id")):
                request = SimpleNamespace(method="POST", path=f"{prefix}/jobs/{suffix}")
                self.assertEqual(await middleware(request, native), "cancelled")
                orchestrator.proxy_job_cancel.assert_awaited_with(request, prompt_id=prompt_id, no_worker_fallback=native)
        for method, path in (("GET", "/api/jobs/id/retry"), ("POST", "/api/jobs/id/retry"), ("POST", "/mgpu/jobs/cancel")):
            self.assertEqual(await middleware(SimpleNamespace(method=method, path=path), native), "native")

    async def test_direct_cancel_without_workers_preserves_native_body(self):
        orchestrator = MultiGpuOrchestrator(prompt_server=FakePromptServer())
        app = web.Application(middlewares=[create_direct_prompt_routing_middleware(orchestrator)])

        async def native(request):
            return web.json_response(await request.json())

        app.router.add_post("/api/jobs/cancel", native)
        async with TestClient(TestServer(app)) as client:
            payload = {"job_ids": [str(uuid4())]}
            response = await client.post("/api/jobs/cancel", json=payload)
            self.assertEqual(response.status, 200)
            self.assertEqual(await response.json(), payload)

    async def test_connect_client_bridges_workers_and_replays_current_state(self):
        prompt_server = FakePromptServer()
        orchestrator = MultiGpuOrchestrator(prompt_server=prompt_server)
        worker0 = WorkerState(
            gpu_index=0,
            port=9000,
            url="http://127.0.0.1:9000",
            status="healthy",
        )
        worker1 = WorkerState(
            gpu_index=1,
            port=9001,
            url="http://127.0.0.1:9001",
            status="failed",
        )
        worker0.running = 1
        worker0.accepted_prompt_ids.add("prompt-a")
        orchestrator.workers = [worker0, worker1]
        orchestrator.prompt_progress["prompt-a"] = {
            "prompt_id": "prompt-a",
            "mgpu": {"total_percent": 42},
        }
        orchestrator.start = AsyncMock()
        orchestrator.refresh_queues = AsyncMock()
        orchestrator.ensure_bridge = AsyncMock(return_value=True)

        result = await orchestrator.connect_client("client-a")

        orchestrator.start.assert_awaited_once()
        orchestrator.refresh_queues.assert_awaited_once()
        orchestrator.ensure_bridge.assert_awaited_once_with(worker0, "client-a")
        self.assertEqual(orchestrator.connected_client_ids, {"client-a"})
        self.assertEqual(result["connected_workers"], [0])
        self.assertEqual(
            prompt_server.sent,
            [
                ("status", {"status": build_queue_info([worker0, worker1])}, "client-a"),
                (
                    "progress_state",
                    orchestrator.prompt_progress["prompt-a"],
                    "client-a",
                ),
            ],
        )

    async def test_connect_client_rejects_missing_client_id(self):
        orchestrator = MultiGpuOrchestrator(prompt_server=FakePromptServer())

        with self.assertRaisesRegex(ValueError, "client_id is required"):
            await orchestrator.connect_client("  ")

    async def test_worker_restart_reconnect_ignores_stale_browser_clients(self):
        prompt_server = FakePromptServer()
        prompt_server.sockets = {"client-live": object()}
        orchestrator = MultiGpuOrchestrator(prompt_server=prompt_server)
        worker = WorkerState(
            gpu_index=0,
            port=9000,
            url="http://127.0.0.1:9000",
            status="healthy",
        )
        orchestrator.connected_client_ids.update({"client-live", "client-stale"})
        orchestrator.ensure_bridge = AsyncMock(return_value=True)

        await orchestrator._connect_worker_to_registered_clients(worker)

        self.assertEqual(orchestrator.connected_client_ids, {"client-live"})
        orchestrator.ensure_bridge.assert_awaited_once_with(worker, "client-live")

    async def test_queue_snapshot_removes_completed_prompts_from_requeue_ledger(self):
        orchestrator = MultiGpuOrchestrator(prompt_server=FakePromptServer())
        worker = WorkerState(
            gpu_index=0,
            port=9000,
            url="http://127.0.0.1:9000",
            status="healthy",
        )
        payload = {"prompt": {"1": {"class_type": "SaveImage"}}}
        for prompt_id in ("running-prompt", "pending-prompt", "completed-prompt"):
            worker.accepted_prompt_ids.add(prompt_id)
            worker.pending_prompt_payloads[prompt_id] = payload
        orchestrator._fetch_worker_json = AsyncMock(
            return_value={
                "queue_running": [[1, "running-prompt", {}, {}, []]],
                "queue_pending": [[2, "pending-prompt", {}, {}, []]],
            }
        )

        await orchestrator.refresh_worker_queue(worker)

        self.assertEqual(worker.running, 1)
        self.assertEqual(worker.pending, 1)
        self.assertEqual(
            set(worker.pending_prompt_payloads),
            {"running-prompt", "pending-prompt"},
        )
        self.assertNotIn("completed-prompt", worker.accepted_prompt_ids)

    async def test_direct_prompt_middleware_routes_native_and_api_paths(self):
        orchestrator = SimpleNamespace(proxy_prompt=AsyncMock(return_value="worker-response"))
        middleware = create_direct_prompt_routing_middleware(orchestrator)
        native_handler = AsyncMock(return_value="native-response")

        for path in ("/prompt", "/api/prompt"):
            request = SimpleNamespace(method="POST", path=path)

            response = await middleware(request, native_handler)

            self.assertEqual(response, "worker-response")
            orchestrator.proxy_prompt.assert_awaited_with(request, no_worker_fallback=native_handler)

        native_handler.assert_not_awaited()

    async def test_direct_prompt_middleware_leaves_other_requests_unchanged(self):
        orchestrator = SimpleNamespace(proxy_prompt=AsyncMock(return_value="worker-response"))
        middleware = create_direct_prompt_routing_middleware(orchestrator)
        native_handler = AsyncMock(return_value="native-response")
        request = SimpleNamespace(method="GET", path="/prompt")

        response = await middleware(request, native_handler)

        self.assertEqual(response, "native-response")
        native_handler.assert_awaited_once_with(request)
        orchestrator.proxy_prompt.assert_not_awaited()

    async def test_prompt_proxy_uses_native_fallback_without_healthy_workers(self):
        orchestrator = MultiGpuOrchestrator(prompt_server=FakePromptServer())
        orchestrator._started = True
        orchestrator.workers = [
            WorkerState(gpu_index=0, port=9000, url="http://127.0.0.1:9000", status="failed")
        ]
        orchestrator.refresh_queues = AsyncMock()
        request = SimpleNamespace(json=AsyncMock(return_value={"prompt": {}}))
        native_handler = AsyncMock(return_value="native-response")

        response = await orchestrator.proxy_prompt(request, no_worker_fallback=native_handler)

        self.assertEqual(response, "native-response")
        native_handler.assert_awaited_once_with(request)

    async def test_native_fallback_can_reread_direct_prompt_body(self):
        class FallbackOrchestrator:
            routed_payload = None

            async def proxy_prompt(self, request, no_worker_fallback=None):
                self.routed_payload = await request.json()
                return await no_worker_fallback(request)

        orchestrator = FallbackOrchestrator()
        app = web.Application(middlewares=[create_direct_prompt_routing_middleware(orchestrator)])

        async def native_prompt(request):
            return web.json_response({"payload": await request.json()})

        app.router.add_post("/prompt", native_prompt)
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/prompt", json={"prompt": {"1": {"class_type": "SaveImage"}}})

            self.assertEqual(response.status, 200)
            self.assertEqual(await response.json(), {"payload": orchestrator.routed_payload})

    async def test_worker_status_message_sends_aggregate_status(self):
        prompt_server = FakePromptServer()
        orchestrator = MultiGpuOrchestrator(prompt_server=prompt_server)
        worker0 = WorkerState(gpu_index=0, port=9000, url="http://127.0.0.1:9000", status="healthy")
        worker1 = WorkerState(gpu_index=1, port=9001, url="http://127.0.0.1:9001", status="healthy")
        worker1.running = 1
        orchestrator.workers = [worker0, worker1]

        await orchestrator._forward_ws_text(
            worker0,
            "client-a",
            '{"type":"status","data":{"status":{"exec_info":{"queue_remaining":2}}}}',
        )

        self.assertEqual(worker0.pending, 2)
        self.assertEqual(
            prompt_server.sent[-1],
            (
                "status",
                {"status": {"exec_info": {"queue_remaining": 3}, "mgpu": {
                    "running": 1,
                    "pending": 2,
                    "workers": [worker0.public_dict(), worker1.public_dict()],
                }}},
                "client-a",
            ),
        )

    async def test_execution_success_refreshes_queue_and_sends_status(self):
        prompt_server = FakePromptServer()
        orchestrator = MultiGpuOrchestrator(prompt_server=prompt_server)
        worker = WorkerState(gpu_index=0, port=9000, url="http://127.0.0.1:9000", status="healthy")
        worker.running = 1
        worker.accepted_prompt_ids.add("abc")
        worker.pending_prompt_payloads["abc"] = {"prompt": {"1": {"class_type": "SaveImage"}}}
        orchestrator.workers = [worker]

        async def fake_refresh(refreshed_worker):
            refreshed_worker.running = 0
            refreshed_worker.pending = 0

        orchestrator.refresh_worker_queue = fake_refresh

        await orchestrator._forward_ws_text(
            worker,
            "client-a",
            '{"type":"execution_success","data":{"prompt_id":"abc"}}',
        )

        self.assertEqual(prompt_server.sent[0], ("execution_success", {"prompt_id": "abc"}, "client-a"))
        self.assertNotIn("abc", worker.accepted_prompt_ids)
        self.assertNotIn("abc", worker.pending_prompt_payloads)
        self.assertEqual(
            prompt_server.sent[-1],
            ("status", {"status": build_queue_info([worker])}, "client-a"),
        )

    async def test_progress_state_is_enriched_for_job_queue_rows(self):
        prompt_server = FakePromptServer()
        orchestrator = MultiGpuOrchestrator(prompt_server=prompt_server)
        worker = WorkerState(gpu_index=0, port=9000, url="http://127.0.0.1:9000", status="healthy")
        orchestrator.workers = [worker]
        orchestrator._remember_prompt_worker(
            worker,
            b'{"prompt_id":"prompt-a"}',
            {
                "1": {"class_type": "LoadCheckpoint", "_meta": {"title": "Load Model"}},
                "2": {"class_type": "KSampler", "_meta": {"title": "KSampler (Advanced)"}},
            },
        )

        await orchestrator._forward_ws_text(
            worker,
            "client-a",
            (
                '{"type":"progress_state","data":{"prompt_id":"prompt-a","nodes":{'
                '"1":{"state":"finished","value":1,"max":1,"node_id":"1","real_node_id":"1"},'
                '"2":{"state":"running","value":5,"max":10,"node_id":"2","real_node_id":"2"}'
                "}}}"
            ),
        )

        message_type, data, client_id = prompt_server.sent[-1]
        self.assertEqual(message_type, "progress_state")
        self.assertEqual(client_id, "client-a")
        self.assertEqual(data["mgpu"]["total_percent"], 75)
        self.assertEqual(data["mgpu"]["current_node_percent"], 50)
        self.assertEqual(data["mgpu"]["current_node_label"], "KSampler (Advanced)")
        self.assertEqual(data["nodes"]["2"]["node_label"], "KSampler (Advanced)")

    async def test_request_worker_response_preserves_method(self):
        orchestrator = MultiGpuOrchestrator(prompt_server=FakePromptServer())
        session = FakeSession()
        orchestrator._session = session
        worker = WorkerState(gpu_index=0, port=9000, url="http://127.0.0.1:9000", status="healthy")

        response = await orchestrator._request_worker_response(worker, "HEAD", "/api/assets/hash/blake3:test")

        self.assertEqual(response.status, 200)
        method, url, kwargs = session.calls[0]
        self.assertEqual((method, url), ("HEAD", "http://127.0.0.1:9000/api/assets/hash/blake3:test"))
        self.assertIsNotNone(kwargs["timeout"])

    async def test_worker_log_response_returns_requested_worker_chunk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "gpu-3.log"
            log_path.write_bytes(b"first\nsecond")
            orchestrator = MultiGpuOrchestrator(prompt_server=FakePromptServer())
            orchestrator.log_session_id = "test-session"
            orchestrator.workers = [
                WorkerState(gpu_index=3, port=9003, url="http://127.0.0.1:9003", log_path=str(log_path))
            ]

            response = await orchestrator.worker_log_response(3, 6)
            payload = json.loads(response.text)

            self.assertEqual(response.status, 200)
            self.assertEqual(
                payload,
                {
                    "content": "second",
                    "cursor": 12,
                    "reset": False,
                    "gpu_index": 3,
                    "session_id": "test-session",
                },
            )
            self.assertEqual((await orchestrator.worker_log_response(4, None)).status, 404)

    async def test_stop_worker_marks_worker_stopped_without_starting_others(self):
        orchestrator = MultiGpuOrchestrator(prompt_server=FakePromptServer())
        worker = WorkerState(gpu_index=0, port=9000, url="http://127.0.0.1:9000", status="healthy")
        worker.running = 1
        worker.pending = 2
        worker.accepted_prompt_ids.add("prompt-a")
        orchestrator.workers = [worker]
        orchestrator.prompt_to_worker["prompt-a"] = worker

        stopped = await orchestrator.stop_worker(0)

        self.assertIs(stopped, worker)
        self.assertEqual(worker.status, "stopped")
        self.assertEqual(worker.running, 0)
        self.assertEqual(worker.pending, 0)
        self.assertEqual(orchestrator.prompt_to_worker, {})

    async def test_unexpected_worker_exit_respawns_and_requeues_pending_prompts(self):
        orchestrator = MultiGpuOrchestrator(prompt_server=FakePromptServer())
        worker = WorkerState(
            gpu_index=0,
            port=9000,
            url="http://127.0.0.1:9000",
            process=FakeProcess(returncode=1),
            status="healthy",
        )
        old_payload = {"prompt": {"1": {"class_type": "SaveImage"}}, "client_id": "client-a"}
        worker.accepted_prompt_ids.add("old-prompt")
        worker.pending_prompt_payloads["old-prompt"] = old_payload
        orchestrator.workers = [worker]
        requeued_payloads = []

        async def fake_start(refreshed_worker, *, prefer_existing_port):
            self.assertTrue(prefer_existing_port)
            refreshed_worker.process = FakeProcess(returncode=None, pid=5678)
            refreshed_worker.status = "healthy"
            return True

        async def fake_post(refreshed_worker, route, payload):
            requeued_payloads.append((refreshed_worker, route, payload))
            return FakeResponse()

        async def fake_refresh(refreshed_worker):
            refreshed_worker.running = 0
            refreshed_worker.pending = len(refreshed_worker.pending_prompt_payloads)

        orchestrator._start_worker_with_retry = fake_start
        orchestrator._post_worker = fake_post
        orchestrator.refresh_worker_queue = fake_refresh

        handled = await orchestrator._handle_exited_worker(worker)

        self.assertTrue(handled)
        self.assertEqual(worker.status, "healthy")
        self.assertEqual(requeued_payloads, [(worker, "/prompt", old_payload)])
        self.assertNotIn("old-prompt", worker.pending_prompt_payloads)
        self.assertIn("new-prompt", worker.pending_prompt_payloads)
        self.assertIn("new-prompt", worker.accepted_prompt_ids)

    async def test_respawn_discards_pending_prompts_when_requeue_toggle_is_off(self):
        orchestrator = MultiGpuOrchestrator(prompt_server=FakePromptServer())
        orchestrator.requeue_pending_on_respawn = False
        worker = WorkerState(
            gpu_index=0,
            port=9000,
            url="http://127.0.0.1:9000",
            process=FakeProcess(returncode=1),
            status="healthy",
        )
        worker.accepted_prompt_ids.add("old-prompt")
        worker.pending_prompt_payloads["old-prompt"] = {
            "prompt": {"1": {"class_type": "SaveImage"}}
        }
        orchestrator.workers = [worker]
        orchestrator.prompt_to_worker["old-prompt"] = worker

        async def fake_start(refreshed_worker, *, prefer_existing_port):
            self.assertTrue(prefer_existing_port)
            refreshed_worker.process = FakeProcess(returncode=None, pid=5678)
            refreshed_worker.status = "healthy"
            return True

        orchestrator._start_worker_with_retry = fake_start
        orchestrator._post_worker = AsyncMock()
        orchestrator.refresh_worker_queue = AsyncMock()

        self.assertTrue(await orchestrator._handle_exited_worker(worker))

        orchestrator._post_worker.assert_not_awaited()
        self.assertEqual(worker.pending_prompt_payloads, {})
        self.assertEqual(worker.accepted_prompt_ids, set())
        self.assertNotIn("old-prompt", orchestrator.prompt_to_worker)

    async def test_close_stops_all_workers_and_closes_session(self):
        orchestrator = MultiGpuOrchestrator(prompt_server=FakePromptServer())
        processes = [
            FakeProcess(returncode=None, pid=1001),
            FakeProcess(returncode=None, pid=1002),
        ]
        orchestrator.workers = [
            WorkerState(
                gpu_index=index,
                port=9000 + index,
                url=f"http://127.0.0.1:{9000 + index}",
                process=process,
                status="healthy",
            )
            for index, process in enumerate(processes)
        ]
        session = SimpleNamespace(close=AsyncMock())
        orchestrator._session = session

        await orchestrator.close()

        self.assertTrue(orchestrator._closing)
        self.assertTrue(all(process.terminated for process in processes))
        self.assertTrue(all(worker.process is None for worker in orchestrator.workers))
        self.assertTrue(all(worker.status == "stopped" for worker in orchestrator.workers))
        session.close.assert_awaited_once()

    async def test_spawned_worker_receives_primary_process_id(self):
        orchestrator = MultiGpuOrchestrator(prompt_server=FakePromptServer())
        worker = WorkerState(gpu_index=0, port=9000, url="http://127.0.0.1:9000")
        process = FakeProcess(returncode=None)
        process.stdout = None

        with patch("orchestrator.subprocess.Popen", return_value=process) as popen:
            orchestrator._spawn_worker(worker, None)

        self.assertEqual(popen.call_args.kwargs["env"][PARENT_PID_ENV], str(os.getpid()))

    async def test_failed_auto_respawn_retries_with_backoff(self):
        orchestrator = MultiGpuOrchestrator(prompt_server=FakePromptServer())
        worker = WorkerState(
            gpu_index=0,
            port=9000,
            url="http://127.0.0.1:9000",
            process=FakeProcess(returncode=1),
            status="healthy",
        )
        attempts = 0

        async def fake_start(refreshed_worker, *, prefer_existing_port):
            nonlocal attempts
            attempts += 1
            refreshed_worker.process = FakeProcess(returncode=1)
            refreshed_worker.status = "failed"
            return False

        orchestrator._start_worker_with_retry = fake_start

        self.assertTrue(await orchestrator._handle_exited_worker(worker))
        self.assertFalse(orchestrator._respawn_due(worker))
        self.assertGreater(worker.next_respawn_at, 0.0)
        # Within the backoff window the exit is still handled but not retried.
        self.assertTrue(await orchestrator._handle_exited_worker(worker))
        self.assertEqual(attempts, 1)
        # Once the backoff expires, the next check respawns again.
        worker.next_respawn_at = 0.0
        self.assertTrue(await orchestrator._handle_exited_worker(worker))
        self.assertEqual(attempts, 2)
        self.assertEqual(worker.respawn_attempts, 2)

    async def test_failed_worker_without_process_is_retried_when_due(self):
        orchestrator = MultiGpuOrchestrator(prompt_server=FakePromptServer())
        worker = WorkerState(
            gpu_index=0,
            port=9000,
            url="http://127.0.0.1:9000",
            process=None,
            status="failed",
            error="startup timed out",
        )
        attempts = 0

        async def fake_start(refreshed_worker, *, prefer_existing_port):
            nonlocal attempts
            attempts += 1
            refreshed_worker.process = FakeProcess(returncode=None)
            refreshed_worker.status = "healthy"
            return True

        orchestrator._start_worker_with_retry = fake_start
        orchestrator._connect_worker_to_registered_clients = AsyncMock()

        self.assertTrue(await orchestrator._handle_exited_worker(worker))
        self.assertEqual(attempts, 1)
        self.assertEqual(worker.status, "healthy")
        self.assertEqual(worker.respawn_attempts, 0)
        self.assertEqual(worker.next_respawn_at, 0.0)

    async def test_healthcheck_marks_unresponsive_worker_and_restarts_it(self):
        orchestrator = MultiGpuOrchestrator(prompt_server=FakePromptServer())
        worker = WorkerState(
            gpu_index=0,
            port=9000,
            url="http://127.0.0.1:9000",
            process=FakeProcess(returncode=None),
            status="healthy",
        )
        orchestrator.workers = [worker]
        orchestrator._fetch_worker_json = AsyncMock(side_effect=asyncio.TimeoutError)
        restarted = []

        async def fake_restart(gpu_index):
            restarted.append(gpu_index)
            worker.healthcheck_failures = 0
            return worker

        orchestrator.restart_worker = fake_restart

        for expected_failures in (1, 2):
            await orchestrator._check_worker(worker)
            self.assertEqual(worker.healthcheck_failures, expected_failures)
            self.assertEqual(worker.status, "unhealthy")
            self.assertEqual(restarted, [])

        await orchestrator._check_worker(worker)
        self.assertEqual(restarted, [0])

    async def test_healthcheck_recovers_worker_and_reconnects_stale_bridges(self):
        prompt_server = FakePromptServer()
        prompt_server.sockets = {"client-live": object()}
        orchestrator = MultiGpuOrchestrator(prompt_server=prompt_server)
        worker = WorkerState(
            gpu_index=0,
            port=9000,
            url="http://127.0.0.1:9000",
            process=FakeProcess(returncode=None),
            status="unhealthy",
            error="healthcheck failed: boom",
        )
        orchestrator.workers = [worker]
        orchestrator.connected_client_ids.update({"client-live", "client-gone"})
        done_task = asyncio.ensure_future(asyncio.sleep(0))
        await done_task
        worker.client_bridge_tasks["client-gone"] = done_task
        worker.client_bridge_tasks["client-live"] = done_task
        orchestrator._fetch_worker_json = AsyncMock(return_value={})
        orchestrator.refresh_worker_queue = AsyncMock()
        bridged = []

        async def fake_bridge(worker, client_id):
            bridged.append(client_id)
            return True

        orchestrator.ensure_bridge = fake_bridge

        await orchestrator._check_worker(worker)

        self.assertEqual(worker.status, "healthy")
        self.assertIsNone(worker.error)
        self.assertEqual(worker.healthcheck_failures, 0)
        self.assertEqual(orchestrator.connected_client_ids, {"client-live"})
        self.assertEqual(bridged, ["client-live"])
        self.assertNotIn("client-live", worker.client_bridge_tasks)

    async def test_healthcheck_skips_starting_and_stopped_workers(self):
        orchestrator = MultiGpuOrchestrator(prompt_server=FakePromptServer())
        starting = WorkerState(gpu_index=0, port=9000, url="http://127.0.0.1:9000", status="starting")
        stopped = WorkerState(gpu_index=1, port=9001, url="http://127.0.0.1:9001", status="stopped")
        starting.process = FakeProcess(returncode=None)
        stopped.process = FakeProcess(returncode=None)
        orchestrator.workers = [starting, stopped]
        orchestrator._fetch_worker_json = AsyncMock()

        await orchestrator._check_worker(starting)
        await orchestrator._check_worker(stopped)

        orchestrator._fetch_worker_json.assert_not_awaited()
        self.assertEqual(starting.status, "starting")
        self.assertEqual(stopped.status, "stopped")


if __name__ == "__main__":
    unittest.main()
