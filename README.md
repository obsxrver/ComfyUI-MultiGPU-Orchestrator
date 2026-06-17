# ComfyUI Multi-GPU Orchestrator

Custom ComfyUI extension that keeps the native Queue/Run UX while routing prompt
execution to one localhost-only ComfyUI worker per CUDA GPU.

The intended cloud setup is to expose only the primary ComfyUI port, for example
`18188:8188` on Vast.AI. Worker ports are chosen automatically and bind to
`127.0.0.1`.

## Install

Clone or copy this repository into `ComfyUI/custom_nodes/`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/obsxrver/ComfyUI-MultiGPU-Orchestrator
```

Then start ComfyUI normally. The primary process becomes the UI/orchestrator and
spawns one worker per visible CUDA device.

## Behavior

- Browser POST `/prompt` calls are rerouted to `/mgpu/prompt`.
- Workers are launched with `--cuda-device <gpu>` and
  `COMFYUI_MGPU_WORKER=1`.
- The orchestrator chooses the healthy worker with the smallest running/pending
  queue, tie-breaking round-robin.
- Worker WebSocket execution events are forwarded to the primary browser client.
- `/interrupt` and `/free` are fanned out to workers by the frontend wrapper.

## Configuration

Environment variables:

- `COMFYUI_MGPU_DISABLED=1`: disable orchestration.
- `COMFYUI_MGPU_WORKER=1`: worker mode; set automatically for child workers.
- `COMFYUI_MGPU_DEVICES=0,1`: override discovered CUDA device indexes.
- `COMFYUI_MGPU_STARTUP_TIMEOUT=120`: worker health-check timeout in seconds.
- `COMFYUI_MGPU_WORKER_FLAGS="..."`: append extra flags to each worker command.

Visit `/mgpu/status` on the primary ComfyUI server to inspect workers.

## Notes

This extension assumes a CUDA/NVIDIA runtime. If no workers are available, the
frontend wrapper falls back to native ComfyUI `/prompt` and logs a warning.
