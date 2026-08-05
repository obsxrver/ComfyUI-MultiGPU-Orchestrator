import os
import sys
import tempfile
import types
import unittest
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import memory_accounting  # noqa: E402
from memory_accounting import (  # noqa: E402
    MemoryAccountingSource,
    detect_memory_accounting_source,
    parse_memory_size,
)


VirtualMemory = namedtuple(
    "VirtualMemory",
    ("total", "available", "percent", "used", "free", "active", "inactive", "buffers", "cached", "shared", "slab"),
)


class MemoryAccountingTests(unittest.TestCase):
    def test_parse_memory_size_supports_units_and_percentages(self):
        self.assertEqual(parse_memory_size("12GiB"), 12 * 1024**3)
        self.assertEqual(parse_memory_size("25%", 64 * 1024**3), 16 * 1024**3)

    def test_detects_read_only_outer_cgroup_limit_without_writing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            group = root / "vast-instance"
            group.mkdir()
            (group / "memory.current").write_text(str(6 * 1024**3), encoding="ascii")
            (group / "memory.high").write_text(str(16 * 1024**3), encoding="ascii")
            (group / "memory.max").write_text("max", encoding="ascii")
            proc_cgroup = root / "proc-cgroup"
            proc_cgroup.write_text("0::/vast-instance\n", encoding="ascii")

            source = detect_memory_accounting_source(
                64 * 1024**3,
                cgroup_root=root,
                proc_cgroup_path=proc_cgroup,
            )

            self.assertIsNotNone(source)
            self.assertEqual(source.limit_bytes, 16 * 1024**3)
            self.assertEqual(source.source, "memory.high")
            self.assertEqual(source.current_usage(), 6 * 1024**3)

    def test_configured_limit_is_combined_with_outer_cgroup_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            group = root / "container"
            group.mkdir()
            (group / "memory.current").write_text("1", encoding="ascii")
            (group / "memory.high").write_text(str(24 * 1024**3), encoding="ascii")
            (group / "memory.max").write_text(str(32 * 1024**3), encoding="ascii")
            proc_cgroup = root / "proc-cgroup"
            proc_cgroup.write_text("0::/container\n", encoding="ascii")

            source = detect_memory_accounting_source(
                64 * 1024**3,
                cgroup_root=root,
                proc_cgroup_path=proc_cgroup,
                configured_limit="20GiB",
            )

            self.assertEqual(source.limit_bytes, 20 * 1024**3)
            self.assertEqual(source.source, "COMFYUI_MGPU_SYSTEM_RAM_LIMIT")

    def test_virtual_memory_reports_aggregate_remaining_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            usage_path = Path(tmpdir) / "memory.current"
            usage_path.write_text(str(10 * 1024**3), encoding="ascii")
            host = VirtualMemory(
                total=64 * 1024**3,
                available=40 * 1024**3,
                percent=37.5,
                used=24 * 1024**3,
                free=20 * 1024**3,
                active=0,
                inactive=0,
                buffers=0,
                cached=0,
                shared=0,
                slab=0,
            )
            source = MemoryAccountingSource(
                limit_bytes=16 * 1024**3,
                usage_path=usage_path,
                source="memory.high",
                cgroup_path=Path(tmpdir),
                original_virtual_memory=lambda: host,
            )

            adjusted = source.virtual_memory()

            self.assertEqual(adjusted.total, 16 * 1024**3)
            self.assertEqual(adjusted.available, 6 * 1024**3)
            self.assertEqual(adjusted.used, 10 * 1024**3)
            self.assertEqual(adjusted.percent, 62.5)

    def test_install_updates_comfy_total_and_splits_pinned_budget(self):
        import psutil

        with tempfile.TemporaryDirectory() as tmpdir:
            usage_path = Path(tmpdir) / "memory.current"
            usage_path.write_text(str(2 * 1024**3), encoding="ascii")
            host = VirtualMemory(
                total=64 * 1024**3,
                available=40 * 1024**3,
                percent=37.5,
                used=24 * 1024**3,
                free=20 * 1024**3,
                active=0,
                inactive=0,
                buffers=0,
                cached=0,
                shared=0,
                slab=0,
            )
            source = MemoryAccountingSource(
                limit_bytes=16 * 1024**3,
                usage_path=usage_path,
                source="memory.high",
                cgroup_path=Path(tmpdir),
            )
            comfy_module = types.ModuleType("comfy")
            comfy_module.__path__ = []
            model_management = types.ModuleType("comfy.model_management")
            model_management.total_ram = host.total / 1024**2
            model_management.MAX_PINNED_MEMORY = 60 * 1024**3
            comfy_module.model_management = model_management
            previous_source = memory_accounting._INSTALLED_SOURCE
            memory_accounting._INSTALLED_SOURCE = None
            try:
                with (
                    patch.dict(os.environ, {"COMFYUI_MGPU_WORKER_COUNT": "4"}),
                    patch.dict(
                        sys.modules,
                        {
                            "comfy": comfy_module,
                            "comfy.model_management": model_management,
                        },
                    ),
                    patch.object(psutil, "virtual_memory", return_value=host),
                    patch.object(psutil, "swap_memory", return_value=SimpleNamespace(total=0)),
                    patch("memory_accounting.detect_memory_accounting_source", return_value=source),
                ):
                    result = memory_accounting.install_worker_memory_accounting()
            finally:
                memory_accounting._INSTALLED_SOURCE = previous_source

            self.assertTrue(result["enabled"])
            self.assertEqual(model_management.total_ram, 16 * 1024)
            self.assertLessEqual(model_management.MAX_PINNED_MEMORY * 4, source.limit_bytes)


if __name__ == "__main__":
    unittest.main()
