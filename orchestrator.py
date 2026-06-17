from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

try:
    import aiohttp
    from aiohttp import web
except Exception:  # pragma: no cover - ComfyUI provides aiohttp at runtime.
    aiohttp = None
    web = None

LOG_PREFIX = "[ComfyUI-MGPU]"
WORKER_ENV_FLAG = "COMFYUI_MGPU_WORKER"
DEFAULT_STARTUP_TIMEOUT_SECONDS = 120
DEFAULT_JOBS_FETCH_LIMIT = 1000
FORWARDED_WS_TYPES = {
    "execution_start",
    "execution_cached",
    "executing",
    "executed",
    "progress",
    "progress_text",
    "progress_state",
    "execution_success",
    "execution_error",
    "execution_interrupted",
    "notification",
}
@dataclass
class WorkerState:
    gpu_index: int
    port: int
    url: str
    process: subprocess.Popen | None = None
    status: str = "new"
    error: str | None = None
    running: int = 0
    pending: int = 0
    last_seen: float | None = None
    accepted_prompt_ids: set[str] = field(default_factory=set)
    client_bridge_tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    client_bridge_ready: dict[str, asyncio.Event] = field(default_factory=dict)

    @property
    def load(self) -> int:
        return self.running + self.pending

    @property
    def healthy(self) -> bool:
        return self.status == "healthy"

    def public_dict(self) -> dict[str, Any]:
        return {
            "gpu_index": self.gpu_index,
            "port": self.port,
            "url": self.url,
            "status": self.status,
            "error": self.error,
            "running": self.running,
            "pending": self.pending,
            "load": self.load,
            "last_seen": self.last_seen,
            "pid": self.process.pid if self.process else None,
        }


def parse_device_list(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    devices: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        devices.append(int(item))
    return devices


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def find_comfy_root() -> Path:
    candidates = [Path.cwd(), Path(__file__).resolve().parent.parent.parent]
    for candidate in candidates:
        if (candidate / "main.py").exists():
            return candidate
    return Path.cwd()


def _optional_path_flag(command: list[str], flag: str, value: Any) -> None:
    if value:
        command.extend([flag, str(value)])


def build_worker_command(
    gpu_index: int,
    port: int,
    *,
    python_executable: str | None = None,
    comfy_root: Path | None = None,
    comfy_args: Any = None,
) -> list[str]:
    root = comfy_root or find_comfy_root()
    command = [
        python_executable or sys.executable,
        str(root / "main.py"),
        "--listen",
        "127.0.0.1",
        "--port",
        str(port),
        "--cuda-device",
        str(gpu_index),
        "--disable-auto-launch",
    ]

    if comfy_args is not None:
        _optional_path_flag(command, "--base-directory", getattr(comfy_args, "base_directory", None))
        _optional_path_flag(command, "--output-directory", getattr(comfy_args, "output_directory", None))
        _optional_path_flag(command, "--input-directory", getattr(comfy_args, "input_directory", None))
        _optional_path_flag(command, "--temp-directory", getattr(comfy_args, "temp_directory", None))
        _optional_path_flag(command, "--user-directory", getattr(comfy_args, "user_directory", None))
        for config_path in getattr(comfy_args, "extra_model_paths_config", []) or []:
            _optional_path_flag(command, "--extra-model-paths-config", config_path)
        if getattr(comfy_args, "enable_assets", False):
            command.append("--enable-assets")

    extra_flags = os.environ.get("COMFYUI_MGPU_WORKER_FLAGS", "").strip()
    if extra_flags:
        import shlex

        command.extend(shlex.split(extra_flags))

    return command


def discover_cuda_devices() -> list[int]:
    configured = parse_device_list(os.environ.get("COMFYUI_MGPU_DEVICES"))
    if configured is not None:
        return configured

    try:
        import torch

        return list(range(torch.cuda.device_count()))
    except Exception:
        logging.exception("%s Failed to query torch.cuda.device_count()", LOG_PREFIX)
        return []


def parse_status_filter(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [status.strip().lower() for status in value.split(",") if status.strip()]


def parse_positive_int(value: str | None, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed <= 0:
        return default
    return parsed


def parse_offset(value: str | None) -> int:
    try:
        parsed = int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def query_to_dict(query: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in query.keys():
        values = query.getall(key) if hasattr(query, "getall") else [query.get(key)]
        result[key] = values if len(values) > 1 else values[0]
    return result


def route_with_query(route: str, query: Any) -> str:
    query_dict = query_to_dict(query)
    if not query_dict:
        return route
    return f"{route}?{urlencode(query_dict, doseq=True)}"


def job_matches_status(job: dict[str, Any], statuses: list[str] | None) -> bool:
    return statuses is None or job.get("status") in statuses


def job_matches_workflow(job: dict[str, Any], workflow_id: str | None) -> bool:
    return workflow_id is None or job.get("workflow_id") == workflow_id


def job_sort_value(job: dict[str, Any], sort_by: str) -> Any:
    if sort_by == "execution_duration":
        start = job.get("execution_start_time")
        end = job.get("execution_end_time")
        if start is None or end is None:
            return -1
        return end - start
    return job.get("create_time") or 0


def aggregate_jobs_payload(
    worker_payloads: list[dict[str, Any]],
    *,
    status_filter: list[str] | None = None,
    workflow_id: str | None = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    limit: int | None = 200,
    offset: int = 0,
) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    for payload in worker_payloads:
        for job in payload.get("jobs", []):
            if isinstance(job, dict) and job_matches_status(job, status_filter) and job_matches_workflow(job, workflow_id):
                jobs.append(job)

    reverse = sort_order != "asc"
    jobs.sort(key=lambda job: job_sort_value(job, sort_by), reverse=reverse)
    total = len(jobs)
    page_jobs = jobs[offset:] if limit is None else jobs[offset : offset + limit]
    return {
        "jobs": page_jobs,
        "pagination": {
            "offset": offset,
            "limit": limit,
            "total": total,
            "has_more": (offset + len(page_jobs)) < total,
        },
    }


def _queue_item_to_job(item: Any, status: str) -> dict[str, Any] | None:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None
    prompt_id = str(item[1])
    extra_data = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
    job = {
        "id": prompt_id,
        "status": status,
        "create_time": int(extra_data.get("create_time") or 0),
        "workflow_id": extra_data.get("workflow_id"),
        "priority": item[0] if item else None,
    }
    return {key: value for key, value in job.items() if value is not None}


def _history_status(entry: dict[str, Any]) -> str:
    status_info = entry.get("status")
    if isinstance(status_info, dict):
        status_text = str(status_info.get("status_str") or "").lower()
        messages = status_info.get("messages")
        if "error" in status_text or "failed" in status_text:
            return "failed"
        if "interrupt" in status_text or "cancel" in status_text:
            return "cancelled"
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, (list, tuple)) or not message:
                    continue
                event_name = str(message[0]).lower()
                if event_name == "execution_error":
                    return "failed"
                if event_name == "execution_interrupted":
                    return "cancelled"
    return "completed"


def _media_type_from_output_key(key: str) -> str:
    lowered = key.lower()
    if "video" in lowered or "gif" in lowered or "animation" in lowered:
        return "video"
    if "audio" in lowered:
        return "audio"
    return "image"


def _summarize_outputs(outputs: Any) -> tuple[dict[str, Any] | None, int]:
    if not isinstance(outputs, dict):
        return None, 0
    preview_output = None
    outputs_count = 0
    for node_id, node_output in outputs.items():
        if not isinstance(node_output, dict):
            continue
        for output_key, values in node_output.items():
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, dict):
                    continue
                if "filename" in value:
                    outputs_count += 1
                    if preview_output is None:
                        preview_output = {
                            **value,
                            "nodeId": str(node_id),
                            "mediaType": _media_type_from_output_key(str(output_key)),
                        }
    return preview_output, outputs_count


def _history_entry_to_job(prompt_id: str, entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    prompt_tuple = entry.get("prompt")
    extra_data = {}
    priority = None
    workflow = None
    if isinstance(prompt_tuple, (list, tuple)):
        if len(prompt_tuple) > 0:
            priority = prompt_tuple[0]
        if len(prompt_tuple) > 2:
            workflow = prompt_tuple[2]
        if len(prompt_tuple) > 3 and isinstance(prompt_tuple[3], dict):
            extra_data = prompt_tuple[3]

    outputs = entry.get("outputs")
    preview_output, outputs_count = _summarize_outputs(outputs)
    job = {
        "id": str(prompt_id),
        "status": _history_status(entry),
        "create_time": int(extra_data.get("create_time") or 0),
        "execution_start_time": None,
        "execution_end_time": None,
        "preview_output": preview_output,
        "outputs_count": outputs_count,
        "workflow_id": extra_data.get("workflow_id"),
        "priority": priority,
        "workflow": {"extra_data": extra_data} if extra_data else None,
        "outputs": outputs,
        "execution_status": entry.get("status"),
    }
    if workflow is not None and isinstance(job["workflow"], dict):
        job["workflow"]["prompt"] = workflow
    return {key: value for key, value in job.items() if value is not None}


def synthesize_jobs_payload(
    queue_payloads: list[dict[str, Any]],
    history_payloads: list[dict[str, Any]],
    *,
    status_filter: list[str] | None = None,
    workflow_id: str | None = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    limit: int | None = 200,
    offset: int = 0,
) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    for payload in queue_payloads:
        for item in payload.get("queue_running", []):
            job = _queue_item_to_job(item, "in_progress")
            if job is not None:
                jobs.append(job)
        for item in payload.get("queue_pending", []):
            job = _queue_item_to_job(item, "pending")
            if job is not None:
                jobs.append(job)

    for payload in history_payloads:
        for prompt_id, entry in payload.items():
            job = _history_entry_to_job(str(prompt_id), entry)
            if job is not None:
                jobs.append(job)

    return aggregate_jobs_payload(
        [{"jobs": jobs}],
        status_filter=status_filter,
        workflow_id=workflow_id,
        sort_by=sort_by,
        sort_order=sort_order,
        limit=limit,
        offset=offset,
    )


def aggregate_assets_payload(
    worker_payloads: list[dict[str, Any]],
    *,
    limit: int | None = 200,
    offset: int = 0,
    sort: str = "created_at",
    order: str = "desc",
) -> dict[str, Any]:
    assets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for payload in worker_payloads:
        for asset in payload.get("assets", []):
            if not isinstance(asset, dict):
                continue
            asset_id = asset.get("id")
            if asset_id and asset_id in seen_ids:
                continue
            if asset_id:
                seen_ids.add(asset_id)
            assets.append(asset)

    reverse = order != "asc"
    assets.sort(key=lambda asset: str(asset.get(sort) or asset.get("created_at") or ""), reverse=reverse)
    total = len(assets)
    page_assets = assets[offset:] if limit is None else assets[offset : offset + limit]
    return {
        "assets": page_assets,
        "total": total,
        "has_more": (offset + len(page_assets)) < total,
    }


def aggregate_tags_payload(
    worker_payloads: list[dict[str, Any]],
    *,
    limit: int | None = None,
    offset: int = 0,
    order: str = "desc",
) -> dict[str, Any]:
    counts: dict[tuple[str, str], int] = {}
    for payload in worker_payloads:
        for tag in payload.get("tags", []):
            if not isinstance(tag, dict) or not tag.get("name"):
                continue
            tag_type = str(tag.get("type") or "")
            key = (str(tag["name"]), tag_type)
            try:
                count = int(tag.get("count") or 0)
            except (TypeError, ValueError):
                count = 0
            counts[key] = counts.get(key, 0) + count

    tags = [{"name": name, "type": tag_type, "count": count} for (name, tag_type), count in counts.items()]
    reverse = order != "asc"
    tags.sort(key=lambda tag: (tag["count"], tag["name"]), reverse=reverse)
    total = len(tags)
    page_tags = tags[offset:] if limit is None else tags[offset : offset + limit]
    return {
        "tags": page_tags,
        "total": total,
        "has_more": (offset + len(page_tags)) < total,
    }


def build_queue_info(workers: list[WorkerState]) -> dict[str, Any]:
    running = sum(worker.running for worker in workers if worker.status != "failed")
    pending = sum(worker.pending for worker in workers if worker.status != "failed")
    return {
        "exec_info": {
            "queue_remaining": running + pending,
        },
        "mgpu": {
            "running": running,
            "pending": pending,
            "workers": [worker.public_dict() for worker in workers],
        },
    }


class MultiGpuOrchestrator:
    def __init__(self, prompt_server: Any):
        self.prompt_server = prompt_server
        self.workers: list[WorkerState] = []
        self.prompt_to_worker: dict[str, WorkerState] = {}
        self.asset_to_worker: dict[str, WorkerState] = {}
        self._session: aiohttp.ClientSession | None = None
        self._rr_index = 0
        self._started = False
        self._start_lock = asyncio.Lock()
        self.routing_policy = "least_busy"
        self.startup_timeout = float(
            os.environ.get("COMFYUI_MGPU_STARTUP_TIMEOUT", DEFAULT_STARTUP_TIMEOUT_SECONDS)
        )
        self.comfy_root = find_comfy_root()

    async def start(self) -> None:
        async with self._start_lock:
            if self._started:
                return
            self._started = True
            timeout = aiohttp.ClientTimeout(total=None)
            self._session = aiohttp.ClientSession(timeout=timeout)

            devices = discover_cuda_devices()
            if not devices:
                logging.warning("%s No CUDA devices found; UI will fall back to native /prompt", LOG_PREFIX)
                return

            try:
                from comfy.cli_args import args as comfy_args
            except Exception:
                comfy_args = None

            for gpu_index in devices:
                port = find_free_port()
                worker = WorkerState(
                    gpu_index=gpu_index,
                    port=port,
                    url=f"http://127.0.0.1:{port}",
                )
                self.workers.append(worker)
                self._spawn_worker(worker, comfy_args)

            await asyncio.gather(*(self._wait_for_worker(worker) for worker in self.workers))

    async def close(self) -> None:
        for worker in self.workers:
            for task in worker.client_bridge_tasks.values():
                task.cancel()
            if worker.process and worker.process.poll() is None:
                worker.process.terminate()
        if self._session:
            await self._session.close()

    def close_sync(self) -> None:
        for worker in self.workers:
            if worker.process and worker.process.poll() is None:
                worker.process.terminate()

    def _spawn_worker(self, worker: WorkerState, comfy_args: Any) -> None:
        command = build_worker_command(
            worker.gpu_index,
            worker.port,
            comfy_root=self.comfy_root,
            comfy_args=comfy_args,
        )
        env = os.environ.copy()
        env[WORKER_ENV_FLAG] = "1"
        try:
            worker.process = subprocess.Popen(
                command,
                cwd=str(self.comfy_root),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            worker.status = "starting"
            logging.info(
                "%s Started worker for GPU %s on %s (pid=%s)",
                LOG_PREFIX,
                worker.gpu_index,
                worker.url,
                worker.process.pid,
            )
        except Exception as exc:
            worker.status = "failed"
            worker.error = str(exc)
            logging.exception("%s Failed to start worker for GPU %s", LOG_PREFIX, worker.gpu_index)

    async def _wait_for_worker(self, worker: WorkerState) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if worker.process and worker.process.poll() is not None:
                worker.status = "failed"
                worker.error = f"worker exited with code {worker.process.returncode}"
                return
            try:
                await self._fetch_worker_json(worker, "/system_stats")
                worker.status = "healthy"
                worker.error = None
                worker.last_seen = time.time()
                await self.refresh_worker_queue(worker)
                return
            except Exception as exc:
                worker.error = str(exc)
                await asyncio.sleep(1)
        worker.status = "failed"
        worker.error = f"startup timed out after {self.startup_timeout:.0f}s"

    async def _fetch_worker_json(self, worker: WorkerState, route: str) -> Any:
        if self._session is None:
            raise RuntimeError("orchestrator session is not ready")
        async with self._session.get(worker.url + route) as response:
            response.raise_for_status()
            return await response.json()

    async def _fetch_worker_json_with_query(
        self,
        worker: WorkerState,
        route: str,
        query: dict[str, Any] | None = None,
    ) -> Any:
        if self._session is None:
            raise RuntimeError("orchestrator session is not ready")
        query_string = f"?{urlencode(query, doseq=True)}" if query else ""
        async with self._session.get(worker.url + route + query_string) as response:
            response.raise_for_status()
            return await response.json()

    async def refresh_worker_queue(self, worker: WorkerState) -> None:
        try:
            data = await self._fetch_worker_json(worker, "/queue")
            worker.running = len(data.get("queue_running", []))
            worker.pending = len(data.get("queue_pending", []))
            worker.status = "healthy"
            worker.error = None
            worker.last_seen = time.time()
        except Exception as exc:
            worker.running = 0
            worker.pending = 0
            worker.status = "unhealthy"
            worker.error = str(exc)

    async def refresh_queues(self) -> None:
        await asyncio.gather(*(self.refresh_worker_queue(worker) for worker in self.workers))

    def select_worker(self) -> WorkerState | None:
        healthy = [worker for worker in self.workers if worker.healthy]
        if not healthy:
            return None

        min_load = min(worker.load for worker in healthy)
        candidates = [worker for worker in healthy if worker.load == min_load]
        for _ in range(len(self.workers)):
            worker = self.workers[self._rr_index % len(self.workers)]
            self._rr_index += 1
            if worker in candidates:
                return worker
        return candidates[0]

    async def proxy_prompt(self, request: Any) -> Any:
        await self.start()
        await self.refresh_queues()
        json_data = await request.json()
        attempted: set[int] = set()
        last_error: str | None = None

        while len(attempted) < len(self.workers):
            worker = self.select_worker()
            if worker is None or worker.gpu_index in attempted:
                break
            attempted.add(worker.gpu_index)
            client_id = json_data.get("client_id")
            if client_id:
                await self.ensure_bridge(worker, str(client_id))
            try:
                response = await self._post_worker(worker, "/prompt", json_data)
                if response.status < 500:
                    body = await response.read()
                    if response.status == 200:
                        self._remember_prompt_worker(worker, body)
                        await self.refresh_worker_queue(worker)
                        if client_id:
                            await self._send_aggregate_status(str(client_id))
                    return web.Response(
                        body=body,
                        status=response.status,
                        headers=self._copy_response_headers(response),
                    )
                last_error = await response.text()
                worker.status = "unhealthy"
                worker.error = last_error
            except Exception as exc:
                last_error = str(exc)
                worker.status = "unhealthy"
                worker.error = last_error
                logging.warning(
                    "%s Failed to forward prompt to GPU %s: %s",
                    LOG_PREFIX,
                    worker.gpu_index,
                    exc,
                )

        return web.json_response(
            {
                "error": {
                    "type": "mgpu_no_worker",
                    "message": "No healthy multi-GPU workers are available",
                    "details": last_error or "Workers are still starting or failed to start",
                },
                "node_errors": {},
            },
            status=424,
        )

    async def prompt_status_response(self) -> Any:
        await self.start()
        await self.refresh_queues()
        return web.json_response(build_queue_info(self.workers))

    async def _post_worker(self, worker: WorkerState, route: str, payload: dict[str, Any]) -> aiohttp.ClientResponse:
        if self._session is None:
            raise RuntimeError("orchestrator session is not ready")
        return await self._session.post(
            worker.url + route,
            json=payload,
            headers={"Comfy-Usage-Source": "comfyui-mgpu-orchestrator"},
        )

    def _remember_prompt_worker(self, worker: WorkerState, body: bytes) -> None:
        try:
            data = json.loads(body.decode("utf-8"))
            prompt_id = data.get("prompt_id")
        except Exception:
            prompt_id = None
        if prompt_id:
            prompt_id = str(prompt_id)
            worker.accepted_prompt_ids.add(prompt_id)
            self.prompt_to_worker[prompt_id] = worker

    @staticmethod
    def _copy_response_headers(response: aiohttp.ClientResponse) -> dict[str, str]:
        allowed = {"Content-Type", "Content-Disposition", "Content-Length", "X-Content-Type-Options"}
        return {key: value for key, value in response.headers.items() if key in allowed}

    async def ensure_bridge(self, worker: WorkerState, client_id: str) -> bool:
        task = worker.client_bridge_tasks.get(client_id)
        ready = worker.client_bridge_ready.get(client_id)
        if task and not task.done():
            if ready is None:
                return True
            try:
                await asyncio.wait_for(ready.wait(), timeout=5)
                return True
            except asyncio.TimeoutError:
                logging.warning(
                    "%s Timed out waiting for worker websocket bridge GPU %s client %s",
                    LOG_PREFIX,
                    worker.gpu_index,
                    client_id,
                )
                return False

        ready = asyncio.Event()
        worker.client_bridge_ready[client_id] = ready
        task = asyncio.create_task(self._bridge_worker_socket(worker, client_id, ready))
        worker.client_bridge_tasks[client_id] = task
        try:
            await asyncio.wait_for(ready.wait(), timeout=5)
            return True
        except asyncio.TimeoutError:
            logging.warning(
                "%s Timed out waiting for worker websocket bridge GPU %s client %s",
                LOG_PREFIX,
                worker.gpu_index,
                client_id,
            )
            return False

    async def _bridge_worker_socket(
        self,
        worker: WorkerState,
        client_id: str,
        ready: asyncio.Event,
    ) -> None:
        if self._session is None:
            ready.set()
            return
        query = urlencode({"clientId": client_id})
        ws_url = f"{worker.url}/ws?{query}"
        try:
            async with self._session.ws_connect(ws_url, heartbeat=30) as ws:
                ready.set()
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.TEXT:
                        await self._forward_ws_text(worker, client_id, message.data)
                    elif message.type == aiohttp.WSMsgType.BINARY:
                        await self._forward_ws_binary(client_id, message.data)
                    elif message.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                        break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            ready.set()
            logging.warning(
                "%s Worker websocket bridge closed for GPU %s client %s: %s",
                LOG_PREFIX,
                worker.gpu_index,
                client_id,
                exc,
            )

    async def _forward_ws_text(self, worker: WorkerState, client_id: str, raw: str) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            return
        message_type = message.get("type")
        if message_type == "status":
            self._update_worker_counts_from_status(worker, message.get("data"))
            await self._send_aggregate_status(client_id)
            return
        if message_type not in FORWARDED_WS_TYPES:
            return
        data = message.get("data")
        if message_type == "execution_start":
            worker.running = max(worker.running, 1)
            worker.last_seen = time.time()
        await self.prompt_server.send(message_type, data, client_id)
        if message_type == "execution_success":
            await self.refresh_worker_queue(worker)
            await self._send_aggregate_status(client_id)
            self._start_primary_asset_seed()
        elif message_type in {"execution_error", "execution_interrupted"}:
            await self.refresh_worker_queue(worker)
            await self._send_aggregate_status(client_id)

    async def _forward_ws_binary(self, client_id: str, raw: bytes) -> None:
        socket_obj = getattr(self.prompt_server, "sockets", {}).get(client_id)
        if socket_obj is not None:
            await socket_obj.send_bytes(raw)

    async def proxy_interrupt(self, request: Any) -> Any:
        await self.start()
        payload = await _read_json_or_empty(request)
        prompt_id = payload.get("prompt_id")
        workers = [self.prompt_to_worker[prompt_id]] if prompt_id in self.prompt_to_worker else self.workers
        if not workers:
            return web.json_response({"workers": [], "error": "no workers available"}, status=424)
        result = await self._fanout_post(workers, "/interrupt", payload)
        status = 200 if any(item["ok"] for item in result) else 424
        return web.json_response({"workers": result}, status=status)

    async def proxy_free(self, request: Any) -> Any:
        await self.start()
        payload = await _read_json_or_empty(request)
        if not self.workers:
            return web.json_response({"workers": [], "error": "no workers available"}, status=424)
        result = await self._fanout_post(self.workers, "/free", payload)
        status = 200 if any(item["ok"] for item in result) else 424
        return web.json_response({"workers": result}, status=status)

    async def proxy_jobs(self, request: Any) -> Any:
        await self.start()
        query = request.rel_url.query
        requested_status = parse_status_filter(query.get("status"))
        workflow_id = query.get("workflow_id")
        sort_by = query.get("sort_by", "created_at").lower()
        sort_order = query.get("sort_order", "desc").lower()
        limit = parse_positive_int(query.get("limit"), 200)
        offset = parse_offset(query.get("offset"))
        worker_limit = max(DEFAULT_JOBS_FETCH_LIMIT, (limit or 0) + offset)

        worker_query = query_to_dict(query)
        worker_query["limit"] = str(worker_limit)
        worker_query["offset"] = "0"

        payloads = await self._fetch_worker_payloads("/api/jobs", worker_query)
        if not payloads and self.workers:
            queue_payloads = await self._fetch_worker_payloads("/queue")
            history_payloads = await self._fetch_worker_payloads("/history", {"max_items": str(worker_limit), "offset": "0"})
            if queue_payloads or history_payloads:
                return web.json_response(
                    synthesize_jobs_payload(
                        queue_payloads,
                        history_payloads,
                        status_filter=requested_status,
                        workflow_id=workflow_id,
                        sort_by=sort_by,
                        sort_order=sort_order,
                        limit=limit,
                        offset=offset,
                    )
                )
            return web.json_response({"error": "worker jobs API unavailable"}, status=424)
        return web.json_response(
            aggregate_jobs_payload(
                payloads,
                status_filter=requested_status,
                workflow_id=workflow_id,
                sort_by=sort_by,
                sort_order=sort_order,
                limit=limit,
                offset=offset,
            )
        )

    async def proxy_job_detail(self, request: Any) -> Any:
        await self.start()
        prompt_id = request.match_info.get("prompt_id", "")
        preferred = self.prompt_to_worker.get(prompt_id)
        workers = [preferred] if preferred else []
        workers.extend(worker for worker in self.workers if worker is not preferred)
        for worker in workers:
            if worker is None:
                continue
            try:
                response = await self._request_worker_response(worker, request.method, f"/api/jobs/{prompt_id}")
                body = await response.read()
                if response.status == 200:
                    return web.Response(
                        body=body,
                        status=response.status,
                        headers=self._copy_response_headers(response),
                    )
            except Exception:
                continue
        fallback = await self._synthesize_job_detail(prompt_id, workers)
        if fallback is not None:
            return web.json_response(fallback)
        return web.json_response({"error": "Job not found"}, status=404)

    async def proxy_assets(self, request: Any) -> Any:
        await self.start()
        query = request.rel_url.query
        limit = parse_positive_int(query.get("limit"), 200)
        offset = parse_offset(query.get("offset"))
        sort = query.get("sort", "created_at").lower()
        order = query.get("order", "desc").lower()
        worker_limit = max(DEFAULT_JOBS_FETCH_LIMIT, (limit or 0) + offset)

        worker_query = query_to_dict(query)
        worker_query["limit"] = str(worker_limit)
        worker_query["offset"] = "0"

        payloads: list[dict[str, Any]] = []
        self.asset_to_worker.clear()
        for worker in self.workers:
            if worker.status == "failed":
                continue
            try:
                payload = await self._fetch_worker_json_with_query(worker, "/api/assets", worker_query)
                if isinstance(payload, dict):
                    payloads.append(payload)
                    for asset in payload.get("assets", []):
                        if isinstance(asset, dict) and asset.get("id"):
                            self.asset_to_worker[str(asset["id"])] = worker
            except Exception as exc:
                worker.error = str(exc)

        if not payloads and self.workers:
            return web.json_response({"error": "worker assets API unavailable"}, status=424)
        return web.json_response(
            aggregate_assets_payload(
                payloads,
                limit=limit,
                offset=offset,
                sort=sort,
                order=order,
            )
        )

    async def proxy_asset_tail(self, request: Any) -> Any:
        await self.start()
        tail = request.match_info.get("tail", "")
        if not tail:
            return await self.proxy_assets(request)

        asset_id = tail.split("/", 1)[0]
        preferred = self.asset_to_worker.get(asset_id)
        workers = [preferred] if preferred else []
        workers.extend(worker for worker in self.workers if worker is not preferred)
        query = query_to_dict(request.rel_url.query)
        query_string = f"?{urlencode(query, doseq=True)}" if query else ""

        for worker in workers:
            if worker is None:
                continue
            try:
                response = await self._request_worker_response(
                    worker,
                    request.method,
                    f"/api/assets/{tail}{query_string}",
                )
                body = await response.read()
                if response.status == 200:
                    return web.Response(
                        body=body,
                        status=response.status,
                        headers=self._copy_response_headers(response),
                    )
            except Exception:
                continue
        return web.json_response({"error": {"code": "ASSET_NOT_FOUND", "message": "Asset not found"}}, status=404)

    async def proxy_asset_seed(self, request: Any) -> Any:
        await self.start()
        payload = await _read_json_or_empty(request)
        if not self.workers:
            return web.json_response({"workers": [], "error": "no workers available"}, status=424)
        route = route_with_query("/api/assets/seed", request.rel_url.query)
        result = await self._fanout_post(self.workers, route, payload or {"roots": ["output"]})
        status = 200 if any(item["ok"] for item in result) else 424
        self._start_primary_asset_seed()
        return web.json_response({"workers": result}, status=status)

    async def proxy_asset_seed_cancel(self, request: Any) -> Any:
        await self.start()
        payload = await _read_json_or_empty(request)
        if not self.workers:
            return web.json_response({"workers": [], "error": "no workers available"}, status=424)
        route = route_with_query("/api/assets/seed/cancel", request.rel_url.query)
        result = await self._fanout_post(self.workers, route, payload)
        status = 200 if any(item["ok"] for item in result) else 424
        return web.json_response({"workers": result}, status=status)

    async def proxy_tags(self, request: Any) -> Any:
        await self.start()
        query = request.rel_url.query
        limit = parse_positive_int(query.get("limit"), None)
        offset = parse_offset(query.get("offset"))
        order = query.get("order", "desc").lower()
        worker_query = query_to_dict(query)
        worker_query["offset"] = "0"
        if limit is not None:
            worker_query["limit"] = str(max(DEFAULT_JOBS_FETCH_LIMIT, limit + offset))

        payloads = await self._fetch_worker_payloads("/api/tags", worker_query)
        if not payloads and self.workers:
            return web.json_response({"error": "worker tags API unavailable"}, status=424)
        return web.json_response(
            aggregate_tags_payload(
                payloads,
                limit=limit,
                offset=offset,
                order=order,
            )
        )

    async def proxy_queue(self, _request: Any) -> Any:
        await self.start()
        payloads = await self._fetch_worker_payloads("/queue")
        running: list[Any] = []
        pending: list[Any] = []
        for payload in payloads:
            running.extend(payload.get("queue_running", []))
            pending.extend(payload.get("queue_pending", []))
        return web.json_response({"queue_running": running, "queue_pending": pending})

    async def _synthesize_job_detail(
        self,
        prompt_id: str,
        workers: list[WorkerState | None],
    ) -> dict[str, Any] | None:
        for worker in workers:
            if worker is None:
                continue
            try:
                history = await self._fetch_worker_json_with_query(worker, f"/history/{prompt_id}")
                if isinstance(history, dict):
                    if prompt_id in history:
                        job = _history_entry_to_job(prompt_id, history[prompt_id])
                    else:
                        job = _history_entry_to_job(prompt_id, history)
                    if job is not None:
                        return job
            except Exception:
                pass

            try:
                queue = await self._fetch_worker_json_with_query(worker, "/queue")
                for item in queue.get("queue_running", []):
                    job = _queue_item_to_job(item, "in_progress")
                    if job and job.get("id") == prompt_id:
                        return job
                for item in queue.get("queue_pending", []):
                    job = _queue_item_to_job(item, "pending")
                    if job and job.get("id") == prompt_id:
                        return job
            except Exception:
                pass
        return None

    async def proxy_queue_control(self, request: Any) -> Any:
        await self.start()
        payload = await _read_json_or_empty(request)
        if not self.workers:
            return web.json_response({"workers": [], "error": "no workers available"}, status=424)
        result = await self._fanout_post(self.workers, "/queue", payload)
        status = 200 if any(item["ok"] for item in result) else 424
        return web.json_response({"workers": result}, status=status)

    async def proxy_history(self, request: Any) -> Any:
        await self.start()
        prompt_id = request.match_info.get("prompt_id")
        route = f"/history/{prompt_id}" if prompt_id else "/history"
        payloads = await self._fetch_worker_payloads(route, query_to_dict(request.rel_url.query))
        history: dict[str, Any] = {}
        for payload in payloads:
            if isinstance(payload, dict):
                history.update(payload)
        return web.json_response(history)

    async def proxy_history_control(self, request: Any) -> Any:
        await self.start()
        payload = await _read_json_or_empty(request)
        if not self.workers:
            return web.json_response({"workers": [], "error": "no workers available"}, status=424)
        result = await self._fanout_post(self.workers, "/history", payload)
        status = 200 if any(item["ok"] for item in result) else 424
        return web.json_response({"workers": result}, status=status)

    async def _fanout_post(
        self,
        workers: list[WorkerState],
        route: str,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        results = []
        for worker in workers:
            try:
                response = await self._post_worker(worker, route, payload)
                text = await response.text()
                results.append(
                    {
                        "gpu_index": worker.gpu_index,
                        "status": response.status,
                        "ok": 200 <= response.status < 300,
                        "body": text,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "gpu_index": worker.gpu_index,
                        "status": 0,
                        "ok": False,
                        "body": str(exc),
                    }
                )
        return results

    async def _get_worker_response(self, worker: WorkerState, route: str) -> aiohttp.ClientResponse:
        return await self._request_worker_response(worker, "GET", route)

    async def _request_worker_response(
        self,
        worker: WorkerState,
        method: str,
        route: str,
    ) -> aiohttp.ClientResponse:
        if self._session is None:
            raise RuntimeError("orchestrator session is not ready")
        return await self._session.request(method, worker.url + route)

    async def _fetch_worker_payloads(
        self,
        route: str,
        query: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for worker in self.workers:
            if worker.status == "failed":
                continue
            try:
                payload = await self._fetch_worker_json_with_query(worker, route, query)
                if isinstance(payload, dict):
                    payloads.append(payload)
                    worker.status = "healthy"
                    worker.error = None
                    worker.last_seen = time.time()
            except Exception as exc:
                worker.status = "unhealthy"
                worker.error = str(exc)
        return payloads

    def _update_worker_counts_from_status(self, worker: WorkerState, data: Any) -> None:
        if not isinstance(data, dict):
            return
        status = data.get("status", data)
        if not isinstance(status, dict):
            return
        exec_info = status.get("exec_info")
        if isinstance(exec_info, dict) and "queue_remaining" in exec_info:
            worker.pending = max(int(exec_info.get("queue_remaining") or 0), 0)
            worker.last_seen = time.time()
            worker.status = "healthy"

    async def _send_aggregate_status(self, client_id: str) -> None:
        await self.prompt_server.send("status", {"status": build_queue_info(self.workers)}, client_id)

    def _start_primary_asset_seed(self) -> None:
        try:
            from app.assets.seeder import asset_seeder

            asset_seeder.start(roots=("output",))
        except Exception:
            logging.debug("%s Primary asset seed trigger skipped", LOG_PREFIX, exc_info=True)

    async def status_response(self) -> Any:
        await self.start()
        await self.refresh_queues()
        return web.json_response(
            {
                "enabled": True,
                "worker_mode": os.environ.get(WORKER_ENV_FLAG) == "1",
                "routing_policy": self.routing_policy,
                "workers": [worker.public_dict() for worker in self.workers],
            }
        )


async def _read_json_or_empty(request: Any) -> dict[str, Any]:
    try:
        data = await request.json()
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


_ORCHESTRATOR: MultiGpuOrchestrator | None = None


def get_orchestrator() -> MultiGpuOrchestrator | None:
    return _ORCHESTRATOR


def register_routes() -> MultiGpuOrchestrator | None:
    global _ORCHESTRATOR
    if web is None:
        logging.error("%s aiohttp is unavailable; routes not registered", LOG_PREFIX)
        return None
    if _ORCHESTRATOR is not None:
        return _ORCHESTRATOR

    from server import PromptServer

    prompt_server = PromptServer.instance
    orchestrator = MultiGpuOrchestrator(prompt_server)
    _ORCHESTRATOR = orchestrator
    routes = prompt_server.routes

    @routes.get("/mgpu/status")
    async def mgpu_status(_request):
        return await orchestrator.status_response()

    @routes.post("/mgpu/prompt")
    async def mgpu_prompt(request):
        return await orchestrator.proxy_prompt(request)

    @routes.get("/mgpu/prompt")
    async def mgpu_prompt_status(_request):
        return await orchestrator.prompt_status_response()

    @routes.post("/mgpu/interrupt")
    async def mgpu_interrupt(request):
        return await orchestrator.proxy_interrupt(request)

    @routes.post("/mgpu/free")
    async def mgpu_free(request):
        return await orchestrator.proxy_free(request)

    @routes.get("/mgpu/jobs")
    async def mgpu_jobs(request):
        return await orchestrator.proxy_jobs(request)

    @routes.get("/mgpu/jobs/{prompt_id}")
    async def mgpu_job_detail(request):
        return await orchestrator.proxy_job_detail(request)

    @routes.get("/mgpu/queue")
    async def mgpu_queue(request):
        return await orchestrator.proxy_queue(request)

    @routes.post("/mgpu/queue")
    async def mgpu_queue_control(request):
        return await orchestrator.proxy_queue_control(request)

    @routes.get("/mgpu/history")
    async def mgpu_history(request):
        return await orchestrator.proxy_history(request)

    @routes.post("/mgpu/history")
    async def mgpu_history_control(request):
        return await orchestrator.proxy_history_control(request)

    @routes.get("/mgpu/history/{prompt_id}")
    async def mgpu_history_prompt_id(request):
        return await orchestrator.proxy_history(request)

    @routes.get("/mgpu/assets")
    async def mgpu_assets(request):
        return await orchestrator.proxy_assets(request)

    @routes.post("/mgpu/assets/seed")
    async def mgpu_asset_seed(request):
        return await orchestrator.proxy_asset_seed(request)

    @routes.post("/mgpu/assets/seed/cancel")
    async def mgpu_asset_seed_cancel(request):
        return await orchestrator.proxy_asset_seed_cancel(request)

    @routes.get("/mgpu/assets/{tail:.*}")
    async def mgpu_asset_tail(request):
        return await orchestrator.proxy_asset_tail(request)

    @routes.get("/mgpu/tags")
    async def mgpu_tags(request):
        return await orchestrator.proxy_tags(request)

    prompt_server.loop.create_task(orchestrator.start())
    atexit.register(orchestrator.close_sync)
    logging.info("%s Registered routes and scheduled worker startup", LOG_PREFIX)
    return orchestrator
