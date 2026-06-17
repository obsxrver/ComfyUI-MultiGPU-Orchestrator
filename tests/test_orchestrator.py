import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orchestrator import (  # noqa: E402
    MultiGpuOrchestrator,
    WorkerState,
    aggregate_assets_payload,
    aggregate_jobs_payload,
    aggregate_tags_payload,
    build_queue_info,
    build_worker_command,
    load_config,
    parse_device_list,
    route_with_query,
    save_config,
    synthesize_jobs_payload,
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
        self.assertIn("--enable-assets", command)

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
                self.assertEqual(load_config(), {"auto_start": True})
                save_config({"auto_start": False})
                self.assertEqual(load_config(), {"auto_start": False})
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

    async def request(self, method, url):
        self.calls.append((method, url))
        return SimpleNamespace(status=200)


class OrchestratorAsyncTests(unittest.IsolatedAsyncioTestCase):
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
        self.assertEqual(session.calls, [("HEAD", "http://127.0.0.1:9000/api/assets/hash/blake3:test")])

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


if __name__ == "__main__":
    unittest.main()
