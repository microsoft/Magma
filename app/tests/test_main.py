from fastapi.testclient import TestClient

from src import main


class TestMain:
    def test_create_app_enables_docs_in_development(self, monkeypatch):
        monkeypatch.setattr(main, "is_dev", lambda: True)

        app = main.create_app()

        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"

    def test_create_app_runs_warmup_on_startup_and_exposes_health(self, task_model_module, monkeypatch):
        # Always replace warmup with a MagicMock for this test, regardless of
        # whether conftest installed the stub module or we're running in real-
        # model mode (FULL_APP_USE_REAL_MODEL_TESTS=1). The real warmup would
        # load the Magma checkpoint + OmniParser on app startup — slow, GPU-
        # heavy, and irrelevant for verifying that the startup hook fires.
        # monkeypatch auto-reverts after the test so other tests are unaffected.
        from unittest.mock import MagicMock

        mock_warmup = MagicMock(name="warmup")
        monkeypatch.setattr(task_model_module, "warmup", mock_warmup)

        with TestClient(main.create_app()) as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert "X-Request-Id" in response.headers
        mock_warmup.assert_called_once()

    def test_create_app_disables_docs_when_not_in_development(self, monkeypatch):
        monkeypatch.setattr(main, "is_dev", lambda: False)

        app = main.create_app()

        assert app.docs_url is None
        assert app.redoc_url is None

    def test_create_app_registers_expected_middleware_and_routes(self):
        app = main.create_app()
        route_paths = {route.path for route in app.routes}
        middleware_classes = {
            middleware.cls.__name__
            for middleware in app.user_middleware
        }

        assert "/health" in route_paths
        assert "/api/v1/tasks" in route_paths
        assert "/api/v1/tasks/{task_id}" in route_paths
        assert "LoggingMiddleware" in middleware_classes
        assert "CORSMiddleware" in middleware_classes
