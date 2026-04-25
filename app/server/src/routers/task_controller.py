from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from typing import Optional
from src.domains.task.dao import (
    SuccessResponseDAO,
    TaskDAO,
    TaskCreateDAO,
    TaskProcessDAO,
    TaskProcessResponseDAO,
    TaskUpdateStatusDAO,
)
from src.domains.task.service import TaskService

logger: logging.Logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])


@router.post("", response_model=SuccessResponseDAO)
async def create_task(payload: TaskCreateDAO):
    """Create a new automation task with a prompt."""
    task: TaskDAO = TaskService.create_task(payload)
    return SuccessResponseDAO(data=task.model_dump(mode="json"))


@router.get("", response_model=SuccessResponseDAO)
async def list_tasks():
    """List all tasks."""
    tasks: list[TaskDAO] = TaskService.list_tasks()
    return SuccessResponseDAO(data=[t.model_dump(mode="json") for t in tasks])


@router.get("/pending", response_model=SuccessResponseDAO)
async def get_pending_task():
    """
    Get the next pending task for the client to dispatch to the local agent.
    Moves the task from 'pending' to 'in_progress'.
    Returns null data if no pending tasks.
    """
    task: Optional[TaskDAO] = TaskService.claim_pending_task()
    if task is None:
        return SuccessResponseDAO(data=None)
    return SuccessResponseDAO(data=task.model_dump(mode="json"))


@router.get("/{task_id}", response_model=SuccessResponseDAO)
async def get_task(task_id: str):
    """Get a task by ID."""
    task: Optional[TaskDAO] = TaskService.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return SuccessResponseDAO(data=task.model_dump(mode="json"))


@router.post("/{task_id}/claim", response_model=SuccessResponseDAO)
async def claim_task(task_id: str):
    """
    Claim a specific pending task for local-agent dispatch.

    This lets the browser dispatch only the task it created instead of taking
    the oldest task from the shared pending queue.
    """
    task: Optional[TaskDAO] = TaskService.claim_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != "in_progress":
        raise HTTPException(status_code=409, detail=f"Task is already {task.status}")
    return SuccessResponseDAO(data=task.model_dump(mode="json"))


@router.post("/process", response_model=SuccessResponseDAO)
async def process_screenshot(payload: TaskProcessDAO):
    """
    Process a screenshot for a task and return the next PyAutoGUI action.

    The local agent sends a screenshot (base64) and the current step number.
    The server returns the next action to execute.
    """
    try:
        result: TaskProcessResponseDAO = TaskService.process_screenshot(payload)
        return SuccessResponseDAO(data=result.model_dump(mode="json"))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.patch("/{task_id}/status", response_model=SuccessResponseDAO)
async def update_task_status(task_id: str, payload: TaskUpdateStatusDAO):
    """Update task status (used by Stop button and agent)."""
    task = TaskService.update_task_status(task_id, payload.status)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return SuccessResponseDAO(data=task.model_dump(mode="json"))


@router.delete("/{task_id}", response_model=SuccessResponseDAO)
async def delete_task(task_id: str):
    """Delete a task."""
    deleted: bool = TaskService.delete_task(task_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Task not found")
    return SuccessResponseDAO(data={"deleted": True})
