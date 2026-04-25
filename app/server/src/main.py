from __future__ import annotations

import logging
import os
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add server root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.cfg import HOST, LOG_LEVEL, PORT, is_dev
from src.middleware.logging_middleware import LoggingMiddleware
from src.routers.task_controller import router as task_router

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="UI Automation API",
        description="Backend API for screenshot-based UI automation with PyAutoGUI",
        version="0.1.0",
        docs_url="/docs" if is_dev() else None,
        redoc_url="/redoc" if is_dev() else None,
    )

    # Middleware (LIFO order)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(task_router)

    @app.on_event("startup")
    async def _load_models():
        from src.domains.task.model import warmup

        logger.info("Loading Magma + OmniParser models on startup …")
        warmup()
        logger.info("Models ready.")

    @app.get("/")
    async def root():
        return {"service": "DOME", "status": "ok", "endpoints": ["/health", "/api/v1/tasks"]}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()

if __name__ == "__main__":
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run("src.main:app", host=HOST, port=PORT, reload=is_dev())
