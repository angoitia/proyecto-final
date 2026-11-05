"""Aplicación FastAPI que sirve al campeón actual de MLflow."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

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
    """Dependencia que expone el estado del modelo de la aplicación (fácil de sobrescribir en tests)."""
    return request.app.state.model_state


def create_app(autoload: bool = True) -> FastAPI:
    """Fábrica de la aplicación.

    El parámetro ``autoload=False`` mantiene los tests rápidos e independientes
    del registro de MLflow entrenado; la producción usa la ``app`` de módulo
    (autoload habilitado).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = ModelState()
        app.state.model_state = state
        if autoload:
            state.load(get_settings())
        yield

    app = FastAPI(
        title="API de Progresión de Diabetes",
        description=(
            "Sirve el modelo campeón registrado en el Model Registry local de MLflow. "
            "El campeón se selecciona mediante el pipeline MLOps automatizado."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(state: ModelState = Depends(get_model_state)) -> HealthResponse:
        """Prueba de vida. Devuelve 200 incluso cuando el modelo no está disponible."""
        champion = state.champion
        return HealthResponse(
            status="ok",
            model_loaded=state.is_loaded,
            model_name=champion.info.model_name if champion else None,
            model_version=champion.info.version if champion else None,
        )

    @app.get("/", tags=["system"])
    def root() -> dict[str, Any]:
        """Info básica del servicio."""
        return {"status": "ok", "service": "diabetes-progression-api", "docs": "/docs"}

    @app.get("/v1/models", tags=["model"])
    def list_models(state: ModelState = Depends(get_model_state)) -> dict[str, Any]:
        """Endpoint estandar de descubrimiento de modelos (estilo ``/v1/models``)."""
        if not state.is_loaded or state.champion is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Champion model is not loaded. Run the training/registration pipeline first.",
            )
        info = state.champion.info
        return {
            "object": "list",
            "data": [
                {
                    "id": f"{info.model_name}:{info.version}",
                    "object": "model",
                    "model_name": info.model_name,
                    "version": str(info.version),
                }
            ],
        }

    @app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
    def model_info(state: ModelState = Depends(get_model_state)) -> ModelInfoResponse:
        """Metadatos del modelo que está detrás del alias ``champion``."""
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
        """Predice la progresión de la enfermedad para un registro de paciente."""
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
