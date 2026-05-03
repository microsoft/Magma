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
- `GET /api/v1/tasks/pending` — claim the next pending task from the shared queue
- `POST /api/v1/tasks/{id}/claim` — claim one specific pending task
- `POST /api/v1/tasks/process` — submit a screenshot, receive the next PyAutoGUI action
- `PATCH /api/v1/tasks/{id}/status` — cancel / update status

## Local agent

Actual mouse/keyboard execution runs on the client machine. Start the local agent API and point it at this Space:

```bash
python app/agent/executor.py --server-url https://wafair-dome.hf.space --port 8010
```

The web client polls only the task it created, claims that task by ID, then
sends it to `http://localhost:8010/api/v1/agent/tasks` for local execution.

## Desktop app

The Electron app starts the built local agent automatically and opens the FE in
one desktop window.

Build the agent executable once from the repo root. Rebuild it only after
changing `agent/executor.py` or agent dependencies:

```bash
pyenv activate agent
poetry run pyinstaller --onedir --name dome-agent --hidden-import pyautogui agent/executor.py
```

Then start the Electron app:

```bash
cd client
npm run electron:dev
```

## Mock model mode

For local development without loading Magma/OmniParser, start the backend with:

```bash
bash run.sh server-mock
```

Or start the full local stack in mock mode:

```bash
bash run.sh all-mock
```

This sets `USE_MOCK_MODEL=1`, skips model warmup, and returns a mocked `CLICK`
at a random screenshot coordinate. After one click, it returns `TERMINATE` so
the task completes cleanly. Override the mock action and coordinate with
environment variables, for example:

```bash
USE_MOCK_MODEL=1 MOCK_MODEL_ACTION=CLICK MOCK_MODEL_X=100 MOCK_MODEL_Y=100 bash run.sh server
```

To keep returning clicks instead of terminating after the first click, set
`MOCK_MODEL_TERMINATE_AFTER_CLICK=0`.

### Electron with mock backend

To run the Electron desktop app against the mock backend, use two terminals.

Terminal 1: start the mock backend and keep it running:

```bash
pyenv activate agent
bash run.sh server-mock
```

Terminal 2: build the local agent executable if needed:

```bash
pyenv activate agent
poetry run pyinstaller --onedir --name dome-agent --hidden-import pyautogui agent/executor.py
```

Then start Electron pointed at the mock backend:

```bash
cd client
DOME_BACKEND_URL=http://127.0.0.1:8000 npm run electron:dev
```

## Hardware

- Runs on T4 small (16 GB VRAM) with 4-bit quantization.
- Persistent storage at `/data` recommended to avoid re-downloading Magma-8B (~16 GB) on cold starts.
