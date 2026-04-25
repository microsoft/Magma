import pytest
from PIL import Image

from src.domains.task.dao import (
    ActionDAO,
    ActionType,
    TaskCreateDAO,
    TaskProcessDAO,
    TaskStatus,
)
from src.domains.task.service import TaskService, _decode_screenshot, _save_process_data


class TestTaskService:
    def test_create_get_list_and_delete_task(self):
        created_task = TaskService.create_task(TaskCreateDAO(prompt="Open the browser"))

        fetched_task = TaskService.get_task(created_task.id)
        listed_tasks = TaskService.list_tasks()

        assert fetched_task == created_task
        assert listed_tasks == [created_task]
        assert TaskService.delete_task(created_task.id) is True
        assert TaskService.get_task(created_task.id) is None
        assert TaskService.delete_task(created_task.id) is False

    def test_save_process_data_persists_prompt_once_and_each_step_image(
        self, sample_screenshot_base64, tmp_path
    ):
        task = TaskService.create_task(TaskCreateDAO(prompt="Original prompt"))
        task_dir = tmp_path / "output" / task.id

        _save_process_data(task, sample_screenshot_base64, 0)

        prompt_file = task_dir / "prompt.txt"
        first_prompt_contents = prompt_file.read_text(encoding="utf-8")
        prompt_file.write_text("Changed by test", encoding="utf-8")

        _save_process_data(task, sample_screenshot_base64, 1)

        assert first_prompt_contents == "Original prompt"
        assert prompt_file.read_text(encoding="utf-8") == "Changed by test"
        assert (task_dir / "step_000.png").exists()
        assert (task_dir / "step_001.png").exists()

    def test_decode_screenshot_returns_rgb_image(self, sample_screenshot_base64):
        image = _decode_screenshot(sample_screenshot_base64)

        assert image.mode == "RGB"
        assert image.size == (4, 4)

    def test_claim_pending_task_returns_oldest_pending_task(self):
        first_task = TaskService.create_task(TaskCreateDAO(prompt="First task"))
        second_task = TaskService.create_task(TaskCreateDAO(prompt="Second task"))

        claimed_first = TaskService.claim_pending_task()
        claimed_second = TaskService.claim_pending_task()

        assert claimed_first is first_task
        assert claimed_first.status == TaskStatus.IN_PROGRESS
        assert claimed_second is second_task
        assert claimed_second.status == TaskStatus.IN_PROGRESS
        assert TaskService.claim_pending_task() is None

    def test_claim_task_claims_only_requested_task(self):
        first_task = TaskService.create_task(TaskCreateDAO(prompt="First task"))
        second_task = TaskService.create_task(TaskCreateDAO(prompt="Second task"))

        claimed_second = TaskService.claim_task(second_task.id)

        assert claimed_second is second_task
        assert first_task.status == TaskStatus.PENDING
        assert second_task.status == TaskStatus.IN_PROGRESS

    def test_process_screenshot_raises_for_missing_task(self, sample_screenshot_base64):
        with pytest.raises(ValueError, match="Task not found: missing-task"):
            TaskService.process_screenshot(
                TaskProcessDAO(
                    task_id="missing-task",
                    screenshot_base64=sample_screenshot_base64,
                    step=0,
                )
            )

    def test_process_screenshot_returns_done_when_task_already_completed(
        self, monkeypatch, sample_screenshot_base64
    ):
        task = TaskService.create_task(TaskCreateDAO(prompt="Already done"))
        task.status = TaskStatus.DONE

        def fail_infer(*args, **kwargs):
            raise AssertionError("infer should not be called for completed tasks")

        monkeypatch.setattr("src.domains.task.service.infer", fail_infer)

        response = TaskService.process_screenshot(
            TaskProcessDAO(
                task_id=task.id,
                screenshot_base64=sample_screenshot_base64,
                step=0,
            )
        )

        assert response.status == TaskStatus.DONE
        assert response.action.action_type == ActionType.DONE
        assert response.message == "Task already completed"

    def test_process_screenshot_marks_task_failed_after_max_steps(
        self, monkeypatch, sample_screenshot_base64
    ):
        task = TaskService.create_task(TaskCreateDAO(prompt="Stop after max steps"))

        def fail_infer(*args, **kwargs):
            raise AssertionError("infer should not be called when max steps are exceeded")

        monkeypatch.setattr("src.domains.task.service.infer", fail_infer)

        response = TaskService.process_screenshot(
            TaskProcessDAO(
                task_id=task.id,
                screenshot_base64=sample_screenshot_base64,
                step=task.max_steps,
            )
        )

        assert response.status == TaskStatus.FAILED
        assert response.action.action_type == ActionType.DONE
        assert response.message == "Max steps exceeded, task marked as failed"
        assert task.status == TaskStatus.FAILED

    @pytest.mark.parametrize(
        ("predicted_action", "expected_action_type"),
        [
            ("CLICK", ActionType.CLICK),
            ("DOUBLE_CLICK", ActionType.DOUBLE_CLICK),
            ("RIGHT_CLICK", ActionType.RIGHT_CLICK),
        ],
    )
    def test_process_screenshot_maps_supported_pointer_actions(
        self,
        monkeypatch,
        sample_screenshot_base64,
        predicted_action,
        expected_action_type,
    ):
        task = TaskService.create_task(TaskCreateDAO(prompt=f"Run {predicted_action.lower()}"))

        def fake_infer(image, task_prompt, previous_actions):
            assert previous_actions == "None"
            return {
                "action": predicted_action,
                "x": 50,
                "y": 75,
                "value": None,
                "mark_id": 3,
                "raw_response": f"{predicted_action} raw output",
                "som_image": None,
            }

        monkeypatch.setattr("src.domains.task.service.infer", fake_infer)

        response = TaskService.process_screenshot(
            TaskProcessDAO(
                task_id=task.id,
                screenshot_base64=sample_screenshot_base64,
                step=0,
            )
        )

        assert response.action.action_type == expected_action_type
        assert response.action.parameters.x == 50
        assert response.action.parameters.y == 75
        assert response.status == TaskStatus.IN_PROGRESS
        assert response.message == "Executing step 1"

    def test_process_screenshot_records_model_action_and_debug_outputs(
        self, monkeypatch, sample_screenshot_base64, tmp_path
    ):
        task = TaskService.create_task(TaskCreateDAO(prompt="Click the submit button"))
        task.actions_history.append(
            ActionDAO(
                action_type=ActionType.CLICK,
                description="Clicked the email input",
            )
        )

        def fake_infer(image, task_prompt, previous_actions):
            assert image.size == (4, 4)
            assert task_prompt == "Click the submit button"
            assert previous_actions == "- Clicked the email input"
            return {
                "action": "CLICK",
                "x": 120,
                "y": 240,
                "value": None,
                "mark_id": 7,
                "raw_response": "CLICK mark 7",
                "som_image": Image.new("RGB", (2, 2), color="black"),
            }

        monkeypatch.setattr("src.domains.task.service.infer", fake_infer)

        response = TaskService.process_screenshot(
            TaskProcessDAO(
                task_id=task.id,
                screenshot_base64=sample_screenshot_base64,
                step=1,
            )
        )

        task_dir = tmp_path / "output" / task.id

        assert response.status == TaskStatus.IN_PROGRESS
        assert response.step == 2
        assert response.action.action_type == ActionType.CLICK
        assert response.action.parameters.x == 120
        assert response.action.parameters.y == 240
        assert "mark 7" in response.action.description
        assert task.current_step == 2
        assert task.status == TaskStatus.IN_PROGRESS
        assert len(task.actions_history) == 2
        assert (task_dir / "prompt.txt").read_text(encoding="utf-8") == task.prompt
        assert (task_dir / "step_001.png").exists()
        assert (task_dir / "step_001_som.png").exists()

    def test_process_screenshot_maps_type_action_and_truncates_raw_response(
        self, monkeypatch, sample_screenshot_base64
    ):
        task = TaskService.create_task(TaskCreateDAO(prompt="Type into the email field"))
        long_raw_response = "x" * 200

        def fake_infer(image, task_prompt, previous_actions):
            return {
                "action": "TYPE",
                "x": None,
                "y": None,
                "value": "user@example.com",
                "mark_id": 12,
                "raw_response": long_raw_response,
                "som_image": None,
            }

        monkeypatch.setattr("src.domains.task.service.infer", fake_infer)

        response = TaskService.process_screenshot(
            TaskProcessDAO(
                task_id=task.id,
                screenshot_base64=sample_screenshot_base64,
                step=0,
            )
        )

        assert response.action.action_type == ActionType.TYPE
        assert response.action.parameters.text == "user@example.com"
        assert response.action.parameters.x is None
        assert response.action.parameters.y is None
        assert "mark 12" in response.action.description
        assert long_raw_response[:120] in response.action.description
        assert long_raw_response[:121] not in response.action.description

    def test_process_screenshot_ignores_partial_coordinates(
        self, monkeypatch, sample_screenshot_base64
    ):
        task = TaskService.create_task(TaskCreateDAO(prompt="Handle partial coordinates"))

        def fake_infer(image, task_prompt, previous_actions):
            return {
                "action": "CLICK",
                "x": 10,
                "y": None,
                "value": None,
                "mark_id": None,
                "raw_response": "partial coordinates",
                "som_image": None,
            }

        monkeypatch.setattr("src.domains.task.service.infer", fake_infer)

        response = TaskService.process_screenshot(
            TaskProcessDAO(
                task_id=task.id,
                screenshot_base64=sample_screenshot_base64,
                step=0,
            )
        )

        assert response.action.action_type == ActionType.CLICK
        assert response.action.parameters.x is None
        assert response.action.parameters.y is None

    def test_process_screenshot_maps_unknown_actions_to_done(
        self, monkeypatch, sample_screenshot_base64
    ):
        task = TaskService.create_task(TaskCreateDAO(prompt="Finish when action is unknown"))

        def fake_infer(image, task_prompt, previous_actions):
            return {
                "action": "UNSUPPORTED",
                "x": None,
                "y": None,
                "value": None,
                "mark_id": None,
                "raw_response": "unsupported action",
                "som_image": None,
            }

        monkeypatch.setattr("src.domains.task.service.infer", fake_infer)

        response = TaskService.process_screenshot(
            TaskProcessDAO(
                task_id=task.id,
                screenshot_base64=sample_screenshot_base64,
                step=0,
            )
        )

        assert response.action.action_type == ActionType.DONE
        assert response.status == TaskStatus.DONE
        assert response.message == "Task completed"
        assert task.status == TaskStatus.DONE
        assert task.actions_history[-1] == response.action
