"""
UI Automation Agent - PyAutoGUI Executor

This agent runs on the local machine and:
1. Exposes a FastAPI endpoint for the web client to submit claimed tasks
2. Takes screenshots of the current screen
3. Sends screenshots to the backend API
4. Receives PyAutoGUI actions to execute
5. Executes actions
6. Loops until the backend returns a terminal task status

Usage:
    python executor.py --server-url http://localhost:8000 --port 8010
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import threading
import time

import httpx
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agent")

HTTP_TIMEOUT = httpx.Timeout(300.0, connect=30.0)
_pyautogui = None


def get_pyautogui():
    """Load PyAutoGUI only when a task actually needs desktop control."""
    global _pyautogui
    if _pyautogui is None:
        import pyautogui

        pyautogui.FAILSAFE = True  # Move mouse to corner to abort
        pyautogui.PAUSE = 0.5  # Pause between actions
        _pyautogui = pyautogui
    return _pyautogui


class AgentTask(BaseModel):
    id: str
    prompt: str
    status: str | None = None
    current_step: int = 0
    max_steps: int = 20
    actions_history: list[dict] = Field(default_factory=list)
    created_at: str | None = None


class SuccessResponse(BaseModel):
    data: dict
    api_version: str = "v1.0"
    errors: None = None


def take_screenshot_base64() -> str:
    """Capture the screen and return as base64 string."""
    pyautogui = get_pyautogui()
    screenshot = pyautogui.screenshot()
    buffer = io.BytesIO()
    screenshot.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def execute_action(action: dict) -> None:
    """Execute a PyAutoGUI action from the server response."""
    pyautogui = get_pyautogui()
    action_type = action["action_type"]
    params = action.get("parameters", {})
    description = action.get("description", "")

    logger.info(f"Executing: {action_type} - {description}")

    if action_type == "click":
        x, y = params.get("x", 0), params.get("y", 0)
        button = params.get("button", "left")
        clicks = params.get("clicks", 1)
        pyautogui.click(x=x, y=y, button=button, clicks=clicks)

    elif action_type == "double_click":
        x, y = params.get("x", 0), params.get("y", 0)
        pyautogui.doubleClick(x=x, y=y)

    elif action_type == "right_click":
        x, y = params.get("x", 0), params.get("y", 0)
        pyautogui.rightClick(x=x, y=y)

    elif action_type == "type":
        text = params.get("text", "")
        pyautogui.write(text, interval=0.05)

    elif action_type == "hotkey":
        keys = params.get("keys", [])
        if keys:
            pyautogui.hotkey(*keys)

    elif action_type == "scroll":
        dy = params.get("dy", 0)
        x = params.get("x")
        y = params.get("y")
        if dy != 0:
            pyautogui.scroll(dy, x=x, y=y)

    elif action_type == "move":
        x, y = params.get("x", 0), params.get("y", 0)
        pyautogui.moveTo(x=x, y=y, duration=0.3)

    elif action_type == "drag":
        x, y = params.get("x", 0), params.get("y", 0)
        pyautogui.dragTo(x=x, y=y, duration=0.5)

    elif action_type == "done":
        logger.info("Action: done - no execution needed")

    else:
        logger.warning(f"Unknown action type: {action_type}")


def update_task_status(client: httpx.Client, task_id: str, status: str) -> None:
    """Best-effort status update back to the backend."""
    try:
        resp = client.patch(f"/api/v1/tasks/{task_id}/status", json={"status": status})
        if resp.status_code != 200:
            logger.warning(
                f"Could not mark task {task_id} as {status}: "
                f"{resp.status_code} {resp.text[:500]}"
            )
    except httpx.HTTPError as e:
        logger.warning(f"Could not mark task {task_id} as {status}: {e}")


def run_task(client: httpx.Client, task: dict, delay: float) -> None:
    """Execute a single task: screenshot → process → execute → loop.

    Server decides when to stop:
    - Repeated action → status "done"
    - Max steps exceeded → status "failed"
    - User clicked Stop → status "cancelled"
    """
    task_id = task["id"]
    max_steps = task.get("max_steps", 20)

    logger.info(f"Picked up task {task_id}: {task['prompt']}")

    step = task.get("current_step", 0)

    while step < max_steps:
        logger.info(f"--- Step {step} ---")

        # Wait before taking screenshot (let previous action settle)
        time.sleep(delay)

        # Check if task was cancelled/done before starting expensive inference
        try:
            check = client.get(f"/api/v1/tasks/{task_id}")
            if check.status_code == 200:
                task_status = check.json()["data"]["status"]
                if task_status in ("done", "failed", "cancelled"):
                    logger.info(f"Task {task_id} is {task_status} — stopping before next step")
                    return
        except Exception:
            pass  # continue if check fails

        # Take screenshot
        logger.info("Taking screenshot...")
        screenshot_b64 = take_screenshot_base64()

        # Send to server for processing
        logger.info("Sending screenshot to server...")
        resp = client.post(
            "/api/v1/tasks/process",
            json={
                "task_id": task_id,
                "screenshot_base64": screenshot_b64,
                "step": step,
            },
        )
        if resp.status_code != 200:
            logger.error(f"Server error {resp.status_code}: {resp.text[:500]}")
            update_task_status(client, task_id, "failed")
            return
        result = resp.json()["data"]

        action = result["action"]
        status = result["status"]
        message = result.get("message", "")

        logger.info(f"Server response: status={status}, message={message}")
        logger.info(f"Action: {action['action_type']} - {action.get('description', '')}")

        # Server says stop? (done/failed/cancelled)
        if status in ("done", "failed", "cancelled"):
            logger.info(f"Task {task_id} finished with status: {status} — {message}")
            return

        # Execute the action
        execute_action(action)

        step += 1

    logger.warning(f"Task {task_id}: max steps ({max_steps}) reached")
    update_task_status(client, task_id, "failed")


def _run_task_background(app: FastAPI, task: dict) -> None:
    task_id = task["id"]
    try:
        with httpx.Client(base_url=app.state.server_url, timeout=HTTP_TIMEOUT) as client:
            run_task(client, task, app.state.delay)
    except Exception:
        logger.exception(f"Task {task_id} failed inside the local agent")
        try:
            with httpx.Client(base_url=app.state.server_url, timeout=HTTP_TIMEOUT) as client:
                update_task_status(client, task_id, "failed")
        except Exception:
            logger.exception(f"Could not report task {task_id} failure to backend")
    finally:
        with app.state.active_task_lock:
            if app.state.active_task_id == task_id:
                app.state.active_task_id = None


def create_app(server_url: str = "http://localhost:8000", delay: float = 1.0) -> FastAPI:
    app = FastAPI(
        title="UI Automation Agent",
        description="Local PyAutoGUI task executor",
        version="0.1.0",
    )
    app.state.server_url = server_url.rstrip("/")
    app.state.delay = delay
    app.state.active_task_id = None
    app.state.active_task_lock = threading.Lock()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {
            "service": "agent",
            "status": "ok",
            "backend": app.state.server_url,
            "active_task_id": app.state.active_task_id,
        }

    @app.post("/api/v1/agent/tasks", response_model=SuccessResponse)
    async def submit_task(task: AgentTask, background_tasks: BackgroundTasks):
        with app.state.active_task_lock:
            if app.state.active_task_id is not None:
                raise HTTPException(
                    status_code=409,
                    detail=f"Agent is already processing task {app.state.active_task_id}",
                )
            app.state.active_task_id = task.id

        logger.info(f"Accepted task {task.id}: {task.prompt}")
        background_tasks.add_task(_run_task_background, app, task.model_dump(mode="json"))
        return SuccessResponse(data={"task_id": task.id, "status": "accepted"})

    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser(description="UI Automation Agent")
    parser.add_argument(
        "--server-url",
        default="http://localhost:8000",
        help="Backend server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Agent API host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8010,
        help="Agent API port (default: 8010)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between action steps (default: 1.0)",
    )
    args = parser.parse_args()

    logger.info(
        f"Starting agent API on {args.host}:{args.port}; backend={args.server_url}; "
        f"delay={args.delay}s"
    )
    uvicorn.run(
        create_app(server_url=args.server_url, delay=args.delay), host=args.host, port=args.port
    )


if __name__ == "__main__":
    main()
