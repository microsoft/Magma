import os
from pathlib import Path

# Load environment variables from app/.env BEFORE any other imports that might
# read os.environ. Without this, FULL_APP_USE_REAL_MODEL_TESTS and other env
# vars defined in .env are invisible to pytest's collection phase, and the
# integration-test model stub is silently installed (causing every integration
# test to return "done" on the first step).
from dotenv import load_dotenv

APP_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(APP_ROOT / ".env")

import base64
import io
import sys
import types
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from PIL import Image

SERVER_ROOT = APP_ROOT / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

USE_REAL_TASK_MODEL_TESTS = os.environ.get("FULL_APP_USE_REAL_MODEL_TESTS") == "1"


def _default_infer(image, task_prompt, previous_actions="None"):
    return {
        "action": "DONE",
        "x": None,
        "y": None,
        "value": None,
        "mark_id": None,
        "raw_response": "stubbed model response",
        "som_image": None,
    }


task_model_stub = None
if not USE_REAL_TASK_MODEL_TESTS:
    task_model_stub = types.ModuleType("src.domains.task.model")
    task_model_stub.infer = _default_infer
    task_model_stub.warmup = MagicMock(name="warmup")
    sys.modules["src.domains.task.model"] = task_model_stub


@pytest.fixture
def task_model_module():
    if task_model_stub is None:
        import src.domains.task.model as task_model

        return task_model
    return task_model_stub


TESTS_OUTPUT_DIR = APP_ROOT / "tests" / "output"


@pytest.fixture(autouse=True)
def reset_task_service_state(monkeypatch, tmp_path, request):
    """Reset the in-memory task store and redirect PROCESS_OUTPUT_DIR.

    - Integration tests (pytest marker 'integration' AND real-model mode):
      save to app/tests/output/ so the user can inspect the SoM-annotated
      images, prompts, and model JSON outputs after the run. Each task gets
      its own <task_uuid> subfolder.
    - All other tests (unit, service, router, middleware, main): save to
      pytest's per-test `tmp_path` (auto-cleaned). This preserves the
      original behaviour that unit tests rely on — several of them
      hardcode `tmp_path / "output"` as the expected save location.
    """
    from src.domains.task import service

    service._tasks.clear()

    is_integration = request.node.get_closest_marker("integration") is not None
    if USE_REAL_TASK_MODEL_TESTS and is_integration:
        TESTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(service, "PROCESS_OUTPUT_DIR", TESTS_OUTPUT_DIR)
    else:
        monkeypatch.setattr(service, "PROCESS_OUTPUT_DIR", tmp_path / "output")

    yield
    service._tasks.clear()


@pytest.fixture
def sample_screenshot_base64():
    image = Image.new("RGB", (4, 4), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@pytest.fixture
def task_app():
    from src.routers.task_controller import router

    app = FastAPI(title="task-test-app")
    app.include_router(router)
    return app


@pytest_asyncio.fixture
async def task_http_client(task_app):
    transport = ASGITransport(app=task_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
