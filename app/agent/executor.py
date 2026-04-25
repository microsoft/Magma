"""
UI Automation Agent - PyAutoGUI Executor

This agent runs on the local machine and:
1. Polls the backend for tasks with status "in_progress"
2. Takes a screenshot of the current screen
3. Sends the screenshot to the backend API
4. Receives a PyAutoGUI action to execute
5. Executes the action
6. Loops until the backend returns "done" status
7. Returns to polling for the next task

Usage:
    python executor.py --server-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import time

import httpx
import pyautogui

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agent")

# PyAutoGUI safety settings
pyautogui.FAILSAFE = True  # Move mouse to corner to abort
pyautogui.PAUSE = 0.5  # Pause between actions


def take_screenshot_base64() -> str:
    """Capture the screen and return as base64 string."""
    screenshot = pyautogui.screenshot()
    buffer = io.BytesIO()
    screenshot.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def execute_action(action: dict) -> None:
    """Execute a PyAutoGUI action from the server response."""
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


def poll_for_task(client: httpx.Client) -> dict | None:
    """Poll the backend for a pending task to pick up."""
    try:
        resp = client.get("/api/v1/tasks/pending")
        if resp.status_code == 200:
            data = resp.json().get("data")
            if data:
                return data
        return None
    except httpx.HTTPError as e:
        logger.warning(f"Error polling for tasks: {e}")
        return None


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
            break
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


def run_agent(server_url: str, poll_interval: float = 2.0, delay: float = 1.0) -> None:
    """Main agent loop: poll for tasks, execute them, repeat."""
    # Generous timeout: HF cold starts can take several minutes (Magma-8B re-download + load).
    # Inference itself on a warm container is 3-15s per step.
    client = httpx.Client(base_url=server_url, timeout=httpx.Timeout(300.0, connect=30.0))

    logger.info(f"Agent started. Polling {server_url} every {poll_interval}s...")

    try:
        while True:
            task = poll_for_task(client)
            if task:
                run_task(client, task, delay)
            else:
                time.sleep(poll_interval)
    except KeyboardInterrupt:
        logger.info("Agent stopped by user (Ctrl+C).")
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description="UI Automation Agent")
    parser.add_argument(
        "--server-url",
        default="http://localhost:8000",
        help="Backend server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between polling for new tasks (default: 2.0)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between action steps (default: 1.0)",
    )
    args = parser.parse_args()

    run_agent(
        server_url=args.server_url,
        poll_interval=args.poll_interval,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
