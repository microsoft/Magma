import pytest

from src.domains.task.dao import TaskCreateDAO
from src.domains.task.service import TaskService


class TestTaskController:
    @pytest.mark.asyncio
    async def test_create_task(self, task_http_client):
        response = await task_http_client.post(
            "/api/v1/tasks",
            json=TaskCreateDAO(prompt="Create a task").model_dump(mode="json"),
        )

        assert response.status_code == 200
        payload = response.json()
        data = payload["data"]
        assert data["prompt"] == "Create a task"
        assert data["status"] == "pending"
        assert payload["api_version"] == "v1.0"
        assert payload["errors"] is None

    @pytest.mark.asyncio
    async def test_create_task_rejects_empty_prompt(self, task_http_client):
        response = await task_http_client.post(
            "/api/v1/tasks",
            json={"prompt": ""},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_list_tasks(self, task_http_client):
        TaskService.create_task(TaskCreateDAO(prompt="First"))
        TaskService.create_task(TaskCreateDAO(prompt="Second"))

        response = await task_http_client.get("/api/v1/tasks")

        assert response.status_code == 200
        payload = response.json()
        assert [item["prompt"] for item in payload["data"]] == ["First", "Second"]
        assert payload["api_version"] == "v1.0"

    @pytest.mark.asyncio
    async def test_list_tasks_returns_empty_array_when_no_tasks(self, task_http_client):
        response = await task_http_client.get("/api/v1/tasks")

        assert response.status_code == 200
        assert response.json()["data"] == []

    @pytest.mark.asyncio
    async def test_get_pending_task_returns_null_when_no_pending_tasks(self, task_http_client):
        response = await task_http_client.get("/api/v1/tasks/pending")

        assert response.status_code == 200
        assert response.json()["data"] is None

    @pytest.mark.asyncio
    async def test_get_pending_task_claims_pending_task(self, task_http_client):
        task = TaskService.create_task(TaskCreateDAO(prompt="Claim me"))

        response = await task_http_client.get("/api/v1/tasks/pending")

        assert response.status_code == 200
        payload = response.json()
        assert payload["data"]["id"] == task.id
        assert payload["data"]["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_claim_task_claims_specific_pending_task(self, task_http_client):
        first_task = TaskService.create_task(TaskCreateDAO(prompt="Do not claim me"))
        second_task = TaskService.create_task(TaskCreateDAO(prompt="Claim me"))

        response = await task_http_client.post(f"/api/v1/tasks/{second_task.id}/claim")

        assert response.status_code == 200
        payload = response.json()
        assert payload["data"]["id"] == second_task.id
        assert payload["data"]["status"] == "in_progress"
        assert first_task.status == "pending"

    @pytest.mark.asyncio
    async def test_claim_task_returns_404_for_missing_task(self, task_http_client):
        response = await task_http_client.post("/api/v1/tasks/missing-task/claim")

        assert response.status_code == 404
        assert response.json()["detail"] == "Task not found"

    @pytest.mark.asyncio
    async def test_get_task_returns_404_for_missing_task(self, task_http_client):
        response = await task_http_client.get("/api/v1/tasks/missing-task")

        assert response.status_code == 404
        assert response.json()["detail"] == "Task not found"

    @pytest.mark.asyncio
    async def test_get_task_returns_existing_task(self, task_http_client):
        task = TaskService.create_task(TaskCreateDAO(prompt="Fetch me"))

        response = await task_http_client.get(f"/api/v1/tasks/{task.id}")

        assert response.status_code == 200
        assert response.json()["data"]["id"] == task.id

    @pytest.mark.asyncio
    async def test_process_screenshot_returns_success_payload(
        self, task_http_client, sample_screenshot_base64
    ):
        task = TaskService.create_task(TaskCreateDAO(prompt="Finish this task"))

        response = await task_http_client.post(
            "/api/v1/tasks/process",
            json={
                "task_id": task.id,
                "screenshot_base64": sample_screenshot_base64,
                "step": 0,
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["api_version"] == "v1.0"
        assert payload["errors"] is None
        assert payload["data"]["task_id"] == task.id
        assert payload["data"]["status"] == "done"
        assert payload["data"]["action"]["action_type"] == "done"

    @pytest.mark.asyncio
    async def test_process_screenshot_returns_404_for_missing_task(self, task_http_client):
        response = await task_http_client.post(
            "/api/v1/tasks/process",
            json={
                "task_id": "missing-task",
                "screenshot_base64": "ZmFrZQ==",
                "step": 0,
            },
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "Task not found: missing-task"

    @pytest.mark.asyncio
    async def test_process_screenshot_rejects_invalid_payload(self, task_http_client):
        response = await task_http_client.post(
            "/api/v1/tasks/process",
            json={"task_id": "task-1", "step": 0},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_delete_task(self, task_http_client):
        task = TaskService.create_task(TaskCreateDAO(prompt="Delete me"))

        response = await task_http_client.delete(f"/api/v1/tasks/{task.id}")

        assert response.status_code == 200
        assert response.json()["data"] == {"deleted": True}

    @pytest.mark.asyncio
    async def test_delete_task_returns_404_when_missing(self, task_http_client):
        response = await task_http_client.delete("/api/v1/tasks/missing-task")

        assert response.status_code == 404
        assert response.json()["detail"] == "Task not found"
