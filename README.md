# ComfyUI Multi-GPU Orchestrator

Use every GPU on your remote ComfyUI box without changing how you work.

This custom node turns one ComfyUI session into a multi-GPU launcher. Keep using
the normal Queue button, job queue, history, previews, videos, images, and media
assets tab. Behind the scenes, the orchestrator starts one private worker per
CUDA GPU and sends each generation to the least-busy worker.

It is built for cloud GPU hosts like Vast.AI, where you usually want one public
ComfyUI port and zero fiddling with a separate browser tab for every GPU.

## Why Use It

- One ComfyUI UI controls all visible CUDA GPUs.
- No workflow changes: press Queue like usual.
- No public worker ports: backend workers bind to `127.0.0.1`.
- Jobs are spread across healthy GPUs automatically.
- Outputs, previews, save-image nodes, save-video nodes, history, and media
  assets stay available from the primary UI.
- Works well with port mappings like `18188:8188` on remote GPU providers.

## How It Works

When ComfyUI starts, the extension discovers the visible CUDA devices and starts
one local worker per GPU. The main ComfyUI process stays as your browser-facing
UI and orchestrator. Worker processes do the generation work.

The normal ComfyUI frontend is patched so queueing, status polling, job/history
reads, media asset reads, and refresh actions use the orchestrator automatically.
If workers are unavailable, the UI falls back to native ComfyUI behavior.

## Install

Clone this repository into `ComfyUI/custom_nodes/`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/obsxrver/ComfyUI-MultiGPU-Orchestrator
```

Then start ComfyUI normally.

Open:

```text
http://YOUR_HOST:18188
```

or whichever host/port you normally use for ComfyUI.

## Check Worker Status

After startup, visit:

```text
/mgpu/status
```

You should see one worker per visible CUDA GPU, each with a port, GPU index, and
health state.

Useful debug endpoints:

```text
/mgpu/jobs
/mgpu/queue
/mgpu/history
/mgpu/assets
/mgpu/tags
```

## Configuration

Most users do not need any configuration.

Optional environment variables:

- `COMFYUI_MGPU_DISABLED=1`: disable the orchestrator.
- `COMFYUI_MGPU_DEVICES=0,1`: choose which CUDA device indexes to use.
- `COMFYUI_MGPU_STARTUP_TIMEOUT=120`: worker startup timeout in seconds.
- `COMFYUI_MGPU_WORKER_FLAGS="..."`: append extra flags to every worker.

`COMFYUI_MGPU_WORKER=1` is set automatically for child workers. You normally
should not set it yourself.

## Notes

This extension targets NVIDIA/CUDA ComfyUI installs. Worker ports are local-only
by design, so you only need to expose your normal ComfyUI port to the outside
world.
