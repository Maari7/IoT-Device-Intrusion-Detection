"""FastAPI application for hybrid IDS inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from api.metrics import APP_LATENCY, APP_REQUESTS
from api.middleware.error_handler import ErrorHandlerMiddleware
from api.middleware.logging import RequestLoggingMiddleware
from api.routes.health import router as health_router
from api.routes.monitoring import router as monitoring_router
from api.routes.predict import router as predict_router
from api.state import state
from src.inference.model_loader import InferenceArtifacts, InferenceModelLoader
from src.inference.predictor import HybridPredictor
from src.training.utils import load_yaml

LOGGER = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Creates and configures the FastAPI application."""
    inference_cfg = load_yaml("config/inference_config.yaml")
    server_cfg = inference_cfg.get("server", {})

    app = FastAPI(
        title=server_cfg.get("title", "HX-IDL Hybrid Inference API"),
        version=server_cfg.get("version", "1.0.0"),
        description="Hybrid IDS inference service for research and demo use.",
    )

    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)

    @app.on_event("startup")
    def startup_event() -> None:
        """Loads inference artifacts on startup."""
        loader = InferenceModelLoader(inference_config=inference_cfg)
        state.artifacts = loader.load()
        state.predictor = HybridPredictor(
            artifacts=state.artifacts,
            decision_threshold=float(inference_cfg.get("prediction", {}).get("decision_threshold", 0.5)),
        )
        state.model_ready = True
        LOGGER.info("Inference service ready. version=%s", state.artifacts.model_version if state.artifacts else "unknown")

    @app.get("/")
    def root() -> dict:
        """Returns a small service banner."""
        return {"service": "hx-idl-hybrid-inference", "ready": state.model_ready}

    @app.get("/metrics")
    def metrics() -> Response:
        """Exposes Prometheus metrics."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.include_router(health_router)
    app.include_router(predict_router)
    app.include_router(monitoring_router)

    return app


app = create_app()
