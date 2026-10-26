"""FastAPI application serving the current MLflow champion."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status

from mlops.api.model_loader import ModelState, ModelUnavailableError
from mlops.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PatientFeatures,
    PredictionResponse,
)
from mlops.config import get_settings
from mlops.utils import configure_logging

logger = logging.getLogger(__name__)


def get_model_state(request: Request) -> ModelState:
    """Dependency that exposes the application's model state (easy to override in tests)."""
    return request.app.state.model_state


def create_app(autoload: bool = True) -> FastAPI:
    """Application factory.

    ``autoload=False`` keeps tests fast and independent from a trained MLflow
    registry; production uses the module-level ``app`` (autoload enabled).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = ModelState()
        app.state.model_state = state
        if autoload:
            state.load(get_settings())
        yield

    app = FastAPI(
        title="Diabetes Progression API",
        description=(
            "Serves the champion model registered in the local MLflow Model Registry. "
            "The champion is selected by the automated MLOps pipeline."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(state: ModelState = Depends(get_model_state)) -> HealthResponse:
        """Liveness probe. Returns 200 even when the model is unavailable."""
        champion = state.champion
        return HealthResponse(
            status="ok",
            model_loaded=state.is_loaded,
            model_name=champion.info.model_name if champion else None,
            model_version=champion.info.version if champion else None,
        )

    @app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
    def model_info(state: ModelState = Depends(get_model_state)) -> ModelInfoResponse:
        """Metadata of the model currently behind the ``champion`` alias."""
        if not state.is_loaded or state.champion is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Champion model is not loaded. Run the training/registration pipeline first.",
            )
        info = state.champion.info
        return ModelInfoResponse(
            model_name=info.model_name,
            version=info.version,
            champion_alias=info.champion_alias,
            run_id=info.run_id,
            source=info.source,
            metrics=info.metrics,
            quality_gate=info.quality_gate,
            loaded_at=info.loaded_at,
        )

    @app.post("/predict", response_model=PredictionResponse, tags=["model"])
    def predict(
        payload: PatientFeatures,
        state: ModelState = Depends(get_model_state),
    ) -> PredictionResponse:
        """Predict disease progression for one patient record."""
        if not state.is_loaded or state.champion is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Champion model is not loaded.",
            )

        try:
            prediction = state.predict(payload.model_dump())
        except ModelUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Champion model is not loaded.",
            ) from exc
        except Exception as exc:
            logger.exception("Prediction failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction failed.",
            ) from exc

        info = state.champion.info
        return PredictionResponse(
            prediction=prediction,
            model_name=info.model_name,
            model_version=info.version,
            champion_alias=info.champion_alias,
        )

    return app


configure_logging()
app = create_app()
