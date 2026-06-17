import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orchestrator import (  # noqa: E402
    MultiGpuOrchestrator,
    WorkerState,
    build_worker_command,
    parse_device_list,
)


class OrchestratorPureTests(unittest.TestCase):
    def test_parse_device_list(self):
        self.assertEqual(parse_device_list("0, 2,3"), [0, 2, 3])
        self.assertIsNone(parse_device_list(""))
        self.assertIsNone(parse_device_list(None))

    def test_build_worker_command_preserves_core_flags(self):
        args = SimpleNamespace(
            base_directory="/comfy/base",
            output_directory="/comfy/output",
            input_directory="/comfy/input",
            temp_directory="/comfy/temp",
            user_directory="/comfy/user",
            extra_model_paths_config=["/comfy/extra.yaml"],
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


if __name__ == "__main__":
    unittest.main()
