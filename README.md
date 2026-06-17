# ComfyUI Multi-GPU Orchestrator

## Use ALL the GPUs!

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

## Install

Clone this repository into `ComfyUI/custom_nodes/`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/obsxrver/ComfyUI-MultiGPU-Orchestrator.git
```


<img width="2546" height="990" alt="Screenshot 2026-06-17 183235" src="https://github.com/user-attachments/assets/f746b1cd-efdd-4e13-90c8-a3ad24b58daf" />



<details>
<summary>AI-Assisted Development Disclaimer</summary>
<b>AI-Assisted Development Disclaimer:</b> OpenAI Codex and GPT-5.5-High were utilized to assist in the development of this project. 
</details>
