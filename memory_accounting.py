from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

LOG_PREFIX = "[ComfyUI-MGPU]"
SYSTEM_RAM_LIMIT_ENV = "COMFYUI_MGPU_SYSTEM_RAM_LIMIT"
WORKER_COUNT_ENV = "COMFYUI_MGPU_WORKER_COUNT"
CGROUP_ROOT_ENV = "COMFYUI_MGPU_CGROUP_ROOT"
MIN_MEMORY_LIMIT_BYTES = 256 * 1024**2


def parse_memory_size(value: str, reference_bytes: int | None = None) -> int:
    normalized = value.strip().lower()
    if normalized.endswith("%"):
        if reference_bytes is None:
            raise ValueError("percentage memory limits require a reference size")
        ratio = float(normalized[:-1]) / 100.0
        if not 0 < ratio <= 1:
            raise ValueError("memory limit percentage must be greater than 0 and at most 100")
        return int(reference_bytes * ratio)

    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([kmgtpe]?i?b?)?", normalized)
    if match is None:
        raise ValueError(f"invalid memory size: {value!r}")
    unit = (match.group(2) or "b").removesuffix("b").removesuffix("i")
    powers = {"": 0, "k": 1, "m": 2, "g": 3, "t": 4, "p": 5, "e": 6}
    parsed = int(float(match.group(1)) * 1024 ** powers[unit])
    if parsed < MIN_MEMORY_LIMIT_BYTES:
        raise ValueError(f"memory limit must be at least {MIN_MEMORY_LIMIT_BYTES} bytes")
    return parsed


def _current_cgroup_v2_path(
    root: Path = Path("/sys/fs/cgroup"),
    proc_cgroup_path: Path = Path("/proc/self/cgroup"),
) -> Path | None:
    try:
        lines = proc_cgroup_path.read_text(encoding="ascii").splitlines()
    except OSError:
        return None
    for line in lines:
        hierarchy, separator, remainder = line.partition(":")
        controllers, second_separator, relative = remainder.partition(":")
        if hierarchy == "0" and separator and second_separator and controllers == "":
            parts = [part for part in relative.strip().split("/") if part]
            return root.joinpath(*parts)
    return None


def _read_cgroup_number(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


@dataclass
class MemoryAccountingSource:
    limit_bytes: int
    usage_path: Path
    source: str
    cgroup_path: Path
    original_virtual_memory: Callable[[], Any] | None = None
    _read_warning_logged: bool = False

    def current_usage(self) -> int | None:
        return _read_cgroup_number(self.usage_path)

    def virtual_memory(self) -> Any:
        if self.original_virtual_memory is None:
            raise RuntimeError("memory accounting source is not installed")
        host = self.original_virtual_memory()
        usage = self.current_usage()
        if usage is None:
            if not self._read_warning_logged:
                logging.warning("%s Unable to read aggregate cgroup memory usage", LOG_PREFIX)
                self._read_warning_logged = True
            return host

        total = min(int(host.total), self.limit_bytes)
        usage = min(max(usage, 0), total)
        remaining = max(total - usage, 0)
        available = min(int(host.available), remaining)
        replacements = {
            "total": total,
            "available": available,
            "percent": ((total - available) / total * 100.0) if total else 100.0,
        }
        fields = set(getattr(host, "_fields", ()))
        if "used" in fields:
            replacements["used"] = usage
        if "free" in fields:
            replacements["free"] = min(int(host.free), remaining)
        return host._replace(**replacements)

    def public_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "limit_bytes": self.limit_bytes,
            "current_bytes": self.current_usage(),
            "source": self.source,
            "cgroup_path": str(self.cgroup_path),
        }


def detect_memory_accounting_source(
    host_total_bytes: int,
    *,
    cgroup_root: Path | None = None,
    proc_cgroup_path: Path = Path("/proc/self/cgroup"),
    configured_limit: str | None = None,
) -> MemoryAccountingSource | None:
    root = cgroup_root or Path(os.environ.get(CGROUP_ROOT_ENV, "/sys/fs/cgroup"))
    cgroup_path = _current_cgroup_v2_path(root, proc_cgroup_path)
    if cgroup_path is None or not (cgroup_path / "memory.current").exists():
        return None

    candidates: list[tuple[int, str]] = []
    configured = configured_limit if configured_limit is not None else os.environ.get(SYSTEM_RAM_LIMIT_ENV)
    if configured:
        candidates.append((parse_memory_size(configured, host_total_bytes), SYSTEM_RAM_LIMIT_ENV))
    for filename in ("memory.high", "memory.max"):
        value = _read_cgroup_number(cgroup_path / filename)
        if value is not None and value > 0:
            candidates.append((value, filename))
    if not candidates:
        return None

    limit_bytes, source = min(candidates, key=lambda item: item[0])
    limit_bytes = min(limit_bytes, host_total_bytes)
    return MemoryAccountingSource(
        limit_bytes=limit_bytes,
        usage_path=cgroup_path / "memory.current",
        source=source,
        cgroup_path=cgroup_path,
    )


def _worker_count() -> int:
    try:
        return max(int(os.environ.get(WORKER_COUNT_ENV, "1")), 1)
    except ValueError:
        return 1


def _aggregate_pin_budget(total_bytes: int, swap_bytes: int) -> int:
    return int(
        max(
            total_bytes * 0.40,
            min(
                total_bytes * 0.90,
                total_bytes - 4 * 1024**3,
                total_bytes + swap_bytes - 16 * 1024**3,
            ),
        )
    )


_INSTALLED_SOURCE: MemoryAccountingSource | None = None


def install_worker_memory_accounting() -> dict[str, Any]:
    global _INSTALLED_SOURCE
    if _INSTALLED_SOURCE is not None:
        return _INSTALLED_SOURCE.public_dict()

    try:
        import psutil
    except ImportError:
        return {"enabled": False, "error": "psutil is unavailable"}

    original_virtual_memory = psutil.virtual_memory
    host = original_virtual_memory()
    try:
        source = detect_memory_accounting_source(int(host.total))
    except (OSError, ValueError) as exc:
        return {"enabled": False, "error": str(exc)}
    if source is None:
        return {
            "enabled": False,
            "error": (
                "no finite cgroup memory.high/memory.max was found; set "
                f"{SYSTEM_RAM_LIMIT_ENV} to the instance RAM allocation"
            ),
        }

    source.original_virtual_memory = original_virtual_memory
    psutil.virtual_memory = source.virtual_memory
    _INSTALLED_SOURCE = source

    try:
        import comfy.model_management as model_management

        model_management.total_ram = source.limit_bytes / 1024**2
        current_pin_budget = int(getattr(model_management, "MAX_PINNED_MEMORY", -1))
        if current_pin_budget > 0:
            try:
                swap_bytes = int(psutil.swap_memory().total)
            except Exception:
                swap_bytes = 0
            per_worker_pin_budget = max(
                _aggregate_pin_budget(source.limit_bytes, swap_bytes) // _worker_count(),
                MIN_MEMORY_LIMIT_BYTES,
            )
            model_management.MAX_PINNED_MEMORY = min(current_pin_budget, per_worker_pin_budget)
    except Exception:
        logging.exception("%s Failed to update ComfyUI import-time RAM budgets", LOG_PREFIX)

    snapshot = source.public_dict()
    snapshot["worker_count"] = _worker_count()
    logging.info(
        "%s Installed aggregate worker RAM accounting: limit=%.2f GiB, source=%s, workers=%s",
        LOG_PREFIX,
        source.limit_bytes / 1024**3,
        source.source,
        snapshot["worker_count"],
    )
    return snapshot
