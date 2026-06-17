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


class MultiGpuOrchestrator:
    def __init__(self, prompt_server: Any):
        self.prompt_server = prompt_server
        self.workers: list[WorkerState] = []
        self.prompt_to_worker: dict[str, WorkerState] = {}
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
        allowed = {"Content-Type"}
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
                        await self._forward_ws_text(client_id, message.data)
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

    async def _forward_ws_text(self, client_id: str, raw: str) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            return
        message_type = message.get("type")
        if message_type not in FORWARDED_WS_TYPES:
            return
        data = message.get("data")
        await self.prompt_server.send(message_type, data, client_id)

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

    @routes.post("/mgpu/interrupt")
    async def mgpu_interrupt(request):
        return await orchestrator.proxy_interrupt(request)

    @routes.post("/mgpu/free")
    async def mgpu_free(request):
        return await orchestrator.proxy_free(request)

    prompt_server.loop.create_task(orchestrator.start())
    atexit.register(orchestrator.close_sync)
    logging.info("%s Registered routes and scheduled worker startup", LOG_PREFIX)
    return orchestrator
