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
        logging.info("[ComfyUI-MGPU] Worker mode active; orchestration disabled")
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
