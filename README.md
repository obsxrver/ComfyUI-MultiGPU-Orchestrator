
# ComfyUI Multi-GPU Orchestrator

## Use ALL the GPUs!

<p align="center">
  <img width="48%" alt="ComfyUI-MultiGPU-Example 1" src="https://github.com/user-attachments/assets/c8bb0720-e7d3-4014-85b4-8f46461fb582" />
  <img width="48%" alt="ComfyUI-MultiGPU-Example 2" src="https://github.com/user-attachments/assets/dad6aa42-3ce2-48a0-a2e2-c8a2b9a2d4d9" />
</p>

Multi-GPU Orchestrator turns one ComfyUI session into a smart multi-GPU routing system. Perfect for cloud clusters (Vast.AI, Runpod) and CUDA-rich powerusers. Unlock maximum resource utilization by dispatching workflows across every GPU on your system. All you have to do is click the "► Run" button.  

- One ComfyUI instance controls all GPUs.
- No workflow changes or added configuration required.
- Queue progress, history, and media
  assets are all available from the primary UI.
- Works well with cloud providers out of the box.

## How It Works

When ComfyUI starts, the extension discovers visible CUDA devices and starts
a ComfyUI worker process on each GPU. The main ComfyUI process stays as the browser-facing
UI and orchestrator. Worker processes do the generation work.

The normal ComfyUI frontend is patched so queueing, status polling, job progress, and media assets are routed through the orchestrator automatically.

Direct API submissions work the same way: `POST /prompt` (and `/api/prompt`) on the main server is dispatched to the least-busy healthy worker. If no worker is available, the request falls back to the main ComfyUI process.

The Console keeps the standard `Logs` tab for the main process and adds a
`GPU N` tab for every worker. Worker output is written to
`ComfyUI/logs/mgpu-workers/gpu-N.log`; these files are cleared when the
orchestrator starts and reused for worker restarts during that session.

The MultiGPU sidebar settings can automatically respawn failed workers and,
independently, re-queue only the jobs that were still running or pending in the
worker's last queue snapshot. Completed jobs are removed from the replay ledger.
Workers also stop when the primary ComfyUI server shuts down, restarts, or exits
unexpectedly.

### Aggregate worker RAM accounting

Workers use the outer container's read-only cgroup-v2 counters to make ComfyUI
container-aware. ComfyUI normally asks `psutil` for host-wide RAM, which can be
much larger than a Vast.ai instance's allocation. In worker processes, the
orchestrator reports the smaller allocation and its aggregate remaining memory
to ComfyUI's existing RAM-pressure cache, model offload, and pinned-memory
logic. Caching stays enabled, but every worker reacts to the combined memory
used by the primary process and all workers.

When aggregate headroom falls below ComfyUI's normal RAM-pressure target, the
orchestrator also asks idle workers to release retained models and intermediate
outputs, and temporarily queues new work behind an already-active worker rather
than creating another model copy. This is pressure-triggered reclamation, not a
cache-disable flag; workers cache normally again as soon as pressure subsides.

On Vast.ai, the orchestrator reads the instance allocation from the official
instance API using the injected per-instance credentials. If that is
unavailable, it uses the existing `memory.high`/`memory.max` values. The
aggregate pinned-memory allowance is divided across workers so each process
cannot independently reserve a full-instance allowance.

If the host delegates a writable cgroup hierarchy, the orchestrator also adds a
kernel-enforced shared worker cgroup. Read-only container cgroups do not prevent
the accounting-based path from working. Memory accounting and cgroup details
are returned by `GET /mgpu/status`.

On a read-only hierarchy this is cooperative pressure control inside ComfyUI,
not a kernel hard cap; an exact hard ceiling still requires the container host
to delegate a memory controller. The fallback is designed to reclaim before the
outer runtime reaches that ceiling.

The detected aggregate allocation can be overridden with values such as `80%`,
`48GiB`, or a byte count:

```bash
export COMFYUI_MGPU_SYSTEM_RAM_LIMIT=56GiB
```

The writable-cgroup thresholds remain independently configurable through
`COMFYUI_MGPU_CGROUP_MEMORY_HIGH` and `COMFYUI_MGPU_CGROUP_MEMORY_MAX`. Set
`COMFYUI_MGPU_CGROUP=0` to skip the writable-cgroup attempt while retaining
aggregate accounting.

## Install

Clone this repository into `ComfyUI/custom_nodes/`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/obsxrver/ComfyUI-MultiGPU-Orchestrator.git
```



<details>
<summary>AI-Assisted Development Disclaimer</summary>
<b>AI-Assisted Development Disclaimer:</b> OpenAI Codex and GPT-5.5-High were utilized to assist in the development of this project. 
</details>
