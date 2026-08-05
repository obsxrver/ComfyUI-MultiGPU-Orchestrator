import logging
import os

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"


def _register_orchestrator():
    if os.environ.get("COMFYUI_MGPU_DISABLED") == "1":
        logging.info("[ComfyUI-MGPU] Disabled by COMFYUI_MGPU_DISABLED=1")
        return

    if os.environ.get("COMFYUI_MGPU_WORKER") == "1":
        from .memory_accounting import install_worker_memory_accounting
        from .orchestrator import start_parent_watchdog

        memory_accounting = install_worker_memory_accounting()
        if not memory_accounting.get("enabled"):
            logging.warning(
                "[ComfyUI-MGPU] Aggregate worker RAM accounting is unavailable: %s",
                memory_accounting.get("error", "unknown error"),
            )
        watchdog_started = start_parent_watchdog()
        logging.info(
            "[ComfyUI-MGPU] Worker mode active; orchestration disabled%s%s",
            " and primary-process watchdog started" if watchdog_started else "",
            " with container-aware RAM accounting" if memory_accounting.get("enabled") else "",
        )
        return

    try:
        from .orchestrator import register_routes

        register_routes()
    except Exception:
        logging.exception("[ComfyUI-MGPU] Failed to register orchestrator")


_register_orchestrator()

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
