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
git clone <this-repo-url> ComfyUI-MultiGPU-Orchestrator
```

Then start ComfyUI normally. The primary process becomes the UI/orchestrator and
spawns one worker per visible CUDA device.

## Behavior

- Browser POST `/prompt` calls are rerouted to `/mgpu/prompt`.
- Browser GET `/prompt` queue-status polling is rerouted to `/mgpu/prompt`.
- Same-origin direct `fetch("/api/...")` calls for jobs, queue, history, assets,
  tags, and prompt status are also rewritten to their `/mgpu/*` equivalents.
- Asset detail/content/hash probe reads, including `HEAD /assets/...`, are
  proxied to workers when needed.
- Workers are launched with `--cuda-device <gpu>` and
  `COMFYUI_MGPU_WORKER=1`.
- The orchestrator chooses the healthy worker with the smallest running/pending
  queue, tie-breaking round-robin.
- Worker WebSocket execution events are forwarded to the primary browser client.
- `/interrupt` and `/free` are fanned out to workers by the frontend wrapper.
- The frontend Jobs API reads are routed through `/mgpu/jobs`, so running,
  pending, and completed worker jobs can appear in the primary job queue/history.
- If a worker does not expose ComfyUI's native Jobs API, `/mgpu/jobs` falls back
  to synthesized entries from that worker's `/queue` and `/history` endpoints.
- Queue/history clear and delete requests are routed to workers as well, so the
  primary UI controls operate on the visible worker-backed lists.
- Read-only asset list/detail/content requests are routed through `/mgpu/assets`
  when available, and the primary output asset seeder is kicked after worker
  completion to help the media assets tab discover shared-output files.
- Asset tag reads are routed through `/mgpu/tags` so media tab filters can see
  tags from worker asset indexes.
- Asset seed refresh requests are routed to workers for output rescans, and also
  trigger the primary output seeder.

## Configuration

Environment variables:

- `COMFYUI_MGPU_DISABLED=1`: disable orchestration.
- `COMFYUI_MGPU_WORKER=1`: worker mode; set automatically for child workers.
- `COMFYUI_MGPU_DEVICES=0,1`: override discovered CUDA device indexes.
- `COMFYUI_MGPU_STARTUP_TIMEOUT=120`: worker health-check timeout in seconds.
- `COMFYUI_MGPU_WORKER_FLAGS="..."`: append extra flags to each worker command.

Visit `/mgpu/status` on the primary ComfyUI server to inspect workers.
Useful read proxies for debugging are `/mgpu/jobs`, `/mgpu/queue`,
`/mgpu/history`, `/mgpu/assets`, and `/mgpu/tags`.

## Notes

This extension assumes a CUDA/NVIDIA runtime. If no workers are available, the
frontend wrapper falls back to native ComfyUI `/prompt` and logs a warning.
