---
title: DOME Office Automation Backend
emoji: 🖥️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# DOME — Magma Office Automation Backend

FastAPI backend for screenshot-based UI automation on Word, Excel, and PowerPoint.

## Stack
- **Base VLM:** `microsoft/Magma-8B` (4-bit NF4 quantized)
- **LoRA adapter:** [`Wafair/DOME`](https://huggingface.co/Wafair/DOME) — 5 actions (CLICK, DOUBLE_CLICK, RIGHT_CLICK, TYPE, TERMINATE), exp 16, 60.8% on 273-sample benchmark
- **UI detector:** `microsoft/OmniParser-v2.0` (YOLO + EasyOCR)

## Endpoints
- `GET /health` — health check
- `POST /api/v1/tasks` — create a task
- `GET /api/v1/tasks/pending` — agent polls for the next pending task
- `POST /api/v1/tasks/process` — submit a screenshot, receive the next PyAutoGUI action
- `PATCH /api/v1/tasks/{id}/status` — cancel / update status

## Local agent

Actual mouse/keyboard execution runs on the client machine. Point the local agent at this Space:

```bash
python app/agent/executor.py --server-url https://wafair-dome.hf.space
```

## Hardware

- Runs on T4 small (16 GB VRAM) with 4-bit quantization.
- Persistent storage at `/data` recommended to avoid re-downloading Magma-8B (~16 GB) on cold starts.
