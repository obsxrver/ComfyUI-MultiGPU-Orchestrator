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
