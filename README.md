# ComfyUI Multi-GPU Orchestrator

## Use ALL the GPUs!

<p align="center">
  <img width="48%" alt="ComfyUI-MultiGPU-Example 1" src="https://github.com/user-attachments/assets/c8bb0720-e7d3-4014-85b4-8f46461fb582" />
  <img width="48%" alt="ComfyUI-MultiGPU-Example 2" src="https://github.com/user-attachments/assets/dad6aa42-3ce2-48a0-a2e2-c8a2b9a2d4d9" />
</p>

Run workflows across the CUDA GPUs on your system from one ComfyUI session.
Queue jobs with the usual "► Run" button, and the orchestrator sends each job
to the least-busy available GPU worker.

- One ComfyUI instance controls all GPUs.
- No workflow changes or added configuration required.
- Queue progress, history, and generated media are available from the main UI.
- Supports local and cloud-hosted systems with multiple CUDA GPUs.

## Install

Clone this repository into `ComfyUI/custom_nodes/`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/obsxrver/ComfyUI-MultiGPU-Orchestrator.git
```

Restart ComfyUI after installation. The extension automatically starts a worker
for each visible CUDA GPU.

## Usage

Open ComfyUI as usual, load a workflow, and click **► Run**. Queue multiple jobs
to keep multiple GPUs busy. Each job runs on one GPU; the extension does not
split a single workflow across GPUs or combine their VRAM.

Use the normal queue controls to cancel individual jobs or cancel all jobs.
Cancellation requires a ComfyUI version that supports the jobs cancellation API.
If no GPU worker is available, jobs fall back to the main ComfyUI process.

API clients can submit workflows to the main server's standard `POST /prompt`
endpoint and retrieve worker results through `/history` or `/history/{prompt_id}`.
Completed results are synchronized into the main process's existing prompt queue,
including requests without a `client_id` or WebSocket connection. History remains
available after a worker stops or restarts; clearing or deleting history applies
to both the main process and workers.

The standard `GET /api/jobs` endpoint includes running and pending worker jobs
alongside the main process's jobs, with combined filtering, sorting, and pagination.
`GET /api/jobs/{job_id}` also retrieves worker job details. Completed jobs use the
main process's synchronized history; job queries need no frontend URL rewriting.

## Worker Settings

Open the **MultiGPU** sidebar to check worker status and configure:

- **Auto start at startup** — start GPU workers automatically when ComfyUI starts.
- **Respawn failed workers** — automatically restart workers after a failure.
- **Re-queue pending jobs on respawn** — retry jobs last known to be running or
  waiting when a worker failed.

Workers stop when the main ComfyUI server shuts down or restarts.

## Logs

In the Console, use **Logs** for the main ComfyUI process and **GPU N** for an
individual worker. Worker logs are also saved to
`ComfyUI/logs/mgpu-workers/gpu-N.log` and cleared when the orchestrator starts.

## Intel XPU support (fork notes)

This fork adds Intel GPU support to the original ComfyUI-MultiGPU-Orchestrator.
If PyTorch has no CUDA device but an XPU device is available, each worker is pinned
to one GPU with `ZE_AFFINITY_MASK`. NVIDIA behaviour is unchanged, except that
`--disable-assets` is only passed on ComfyUI versions that support it, and workers
use a private in-memory database.

**Tested on:** 2x Intel Arc Pro B70, Windows, PyTorch 2.14.0+xpu, ComfyUI 0.37.4.
The NVIDIA path has not been tested on real hardware by the fork author.

**Limitations:** one backend per machine (CUDA first, then XPU); mixed NVIDIA + Intel
setups are not supported. Each worker loads its own copy of the model, so VRAM is not
pooled. Intel workers launch with `--disable-pinned-memory --cache-none
--enable-cors-header`; extend or override with `COMFYUI_MGPU_WORKER_FLAGS`. Set
`COMFYUI_MGPU_DEVICES=0,1` to choose which GPUs to use (for example, to skip an
integrated GPU).

No support is promised. Based on
[obsxrver/ComfyUI-MultiGPU-Orchestrator](https://github.com/obsxrver/ComfyUI-MultiGPU-Orchestrator),
licensed under MIT (see `LICENSE`). Intel XPU changes by dastj68-droid.
