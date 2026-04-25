from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path

from PIL import Image

from src.domains.task.dao import (
    ActionDAO,
    ActionParametersDAO,
    ActionType,
    TaskCreateDAO,
    TaskDAO,
    TaskProcessDAO,
    TaskProcessResponseDAO,
    TaskStatus,
)
from src.domains.task.model import infer

logger = logging.getLogger(__name__)

PROCESS_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "output"

# In-memory task store (replace with DB in production)
_tasks: dict[str, TaskDAO] = {}

# Map model ACTION strings → server ActionType. TERMINATE routes to DONE so the
# task is marked complete when the model signals the task is already finished.
_ACTION_MAP: dict[str, ActionType] = {
    "CLICK": ActionType.CLICK,
    "TYPE": ActionType.TYPE,
    "DOUBLE_CLICK": ActionType.DOUBLE_CLICK,
    "RIGHT_CLICK": ActionType.RIGHT_CLICK,
    "TERMINATE": ActionType.DONE,
}


def _save_process_data(task: TaskDAO, screenshot_b64: str, step: int) -> None:
    """Save the prompt and raw screenshot to disk for inspection."""
    task_dir = PROCESS_OUTPUT_DIR / task.id
    task_dir.mkdir(parents=True, exist_ok=True)

    prompt_file = task_dir / "prompt.txt"
    if not prompt_file.exists():
        prompt_file.write_text(task.prompt, encoding="utf-8")

    screenshot_file = task_dir / f"step_{step:03d}.png"
    screenshot_file.write_bytes(base64.b64decode(screenshot_b64))

    logger.info(f"Saved process data for task {task.id} step {step} → {task_dir}")


def _save_model_output_data(
    task: TaskDAO,
    step: int,
    previous_actions: str,
    result: dict,
    som_image_path: Path | None = None,
) -> None:
    """Save structured model output for the current step."""
    task_dir = PROCESS_OUTPUT_DIR / task.id
    task_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "task_id": task.id,
        "step": step,
        "prompt": task.prompt,
        "previous_actions": previous_actions,
        "model_output": {
            "action": result.get("action"),
            "x": result.get("x"),
            "y": result.get("y"),
            "value": result.get("value"),
            "mark_id": result.get("mark_id"),
            "raw_response": result.get("raw_response"),
        },
        "som_image_path": str(som_image_path) if som_image_path is not None else None,
    }

    output_file = task_dir / f"step_{step:03d}_model_output.json"
    output_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _decode_screenshot(screenshot_b64: str) -> Image.Image:
    """Decode a base64 screenshot string into a PIL Image."""
    raw = base64.b64decode(screenshot_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


class TaskService:
    @staticmethod
    def create_task(payload: TaskCreateDAO) -> TaskDAO:
        task = TaskDAO(prompt=payload.prompt, status=TaskStatus.PENDING)
        _tasks[task.id] = task
        logger.info(f"Task created: {task.id} - prompt: {task.prompt}")
        return task

    @staticmethod
    def get_task(task_id: str) -> TaskDAO | None:
        return _tasks.get(task_id)

    @staticmethod
    def list_tasks() -> list[TaskDAO]:
        return list(_tasks.values())

    @staticmethod
    def claim_pending_task() -> TaskDAO | None:
        """Find the oldest pending task and move it to in_progress for dispatch."""
        for task in _tasks.values():
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.IN_PROGRESS
                logger.info(f"Task claimed for dispatch: {task.id}")
                return task
        return None

    @staticmethod
    def claim_task(task_id: str) -> TaskDAO | None:
        """Move a specific pending task to in_progress for dispatch."""
        task = _tasks.get(task_id)
        if task is None:
            return None
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.IN_PROGRESS
            logger.info(f"Task claimed for dispatch: {task.id}")
        return task

    @staticmethod
    def process_screenshot(payload: TaskProcessDAO) -> TaskProcessResponseDAO:
        """
        Process a screenshot: run SoM detection → Magma inference → return next action.
        """
        task = _tasks.get(payload.task_id)
        if task is None:
            raise ValueError(f"Task not found: {payload.task_id}")

        # Save raw screenshot first (SoM image added after inference)
        _save_process_data(task, payload.screenshot_base64, payload.step)

        if task.status == TaskStatus.DONE:
            return TaskProcessResponseDAO(
                task_id=task.id,
                action=ActionDAO(
                    action_type=ActionType.DONE,
                    description="Task already completed",
                ),
                status=TaskStatus.DONE,
                step=task.current_step,
                message="Task already completed",
            )

        # Check max steps
        if payload.step >= task.max_steps:
            task.status = TaskStatus.FAILED
            return TaskProcessResponseDAO(
                task_id=task.id,
                action=ActionDAO(
                    action_type=ActionType.DONE,
                    description="Max steps exceeded, task marked as failed",
                ),
                status=TaskStatus.FAILED,
                step=payload.step,
                message="Max steps exceeded, task marked as failed",
            )

        # --- Real model inference ---
        image = _decode_screenshot(payload.screenshot_base64)

        # Build previous actions string for prompt
        if task.actions_history:
            prev = "\n".join(f"- {a.description}" for a in task.actions_history)
        else:
            prev = "None"

        result = infer(image, task.prompt, previous_actions=prev)

        # Save SoM-annotated image for debugging
        som_image = result.get("som_image")
        som_image_path = None
        if som_image is not None:
            task_dir = PROCESS_OUTPUT_DIR / task.id
            task_dir.mkdir(parents=True, exist_ok=True)
            som_image_path = task_dir / f"step_{payload.step:03d}_som.png"
            som_image.save(som_image_path)

        _save_model_output_data(
            task=task,
            step=payload.step,
            previous_actions=prev,
            result=result,
            som_image_path=som_image_path,
        )

        result.pop("som_image", None)

        predicted = result["action"]
        action_type = _ACTION_MAP.get(predicted, ActionType.DONE)

        params: dict = {}
        if result["x"] is not None and result["y"] is not None:
            params["x"] = result["x"]
            params["y"] = result["y"]
        if result["value"] is not None:
            params["text"] = result["value"]

        description = f"Model predicted {predicted}"
        if result["mark_id"] is not None:
            description += f" on mark {result['mark_id']}"
        description += f" (raw: {result['raw_response'][:120]})"

        action = ActionDAO(
            action_type=action_type,
            parameters=ActionParametersDAO(**params),
            description=description,
        )

        # Check if same as previous action → likely done or stuck
        is_done = action.action_type == ActionType.DONE
        repeated_action = False
        if not is_done and task.actions_history:
            prev = task.actions_history[-1]
            if prev.action_type == action.action_type and prev.parameters == action.parameters:
                is_done = True
                repeated_action = True
                logger.info(f"Task {task.id}: repeated action detected — marking as done")

        # Update task state
        task.current_step = payload.step + 1
        task.actions_history.append(action)
        task.status = TaskStatus.DONE if is_done else TaskStatus.IN_PROGRESS

        return TaskProcessResponseDAO(
            task_id=task.id,
            action=action,
            status=task.status,
            step=task.current_step,
            message=(
                "Task completed (repeated action)"
                if repeated_action
                else "Task completed" if is_done else f"Executing step {task.current_step}"
            ),
        )

    @staticmethod
    def update_task_status(task_id: str, status: str) -> TaskDAO | None:
        """Update task status (used by Stop button and agent)."""
        task = _tasks.get(task_id)
        if task is None:
            return None
        task.status = TaskStatus(status)
        logger.info(f"Task {task_id} status updated to: {status}")
        return task

    @staticmethod
    def delete_task(task_id: str) -> bool:
        if task_id in _tasks:
            del _tasks[task_id]
            return True
        return False
