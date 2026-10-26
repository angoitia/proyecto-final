"""Load the champion model from the MLflow Model Registry."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from mlops.config import Settings, get_settings
from mlops.utils import utc_now

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when no champion model is loaded."""


@dataclass(frozen=True)
class ChampionInfo:
    model_name: str
    version: str
    champion_alias: str
    run_id: str
    source: str
    metrics: dict[str, float] = field(default_factory=dict)
    quality_gate: str | None = None
    loaded_at: str = field(default_factory=utc_now)


@dataclass
class LoadedChampion:
    model: Any
    info: ChampionInfo
    feature_columns: tuple[str, ...]

    def predict(self, features: dict[str, float]) -> float:
        frame = pd.DataFrame([features], columns=list(self.feature_columns))
        raw_predictions = np.asarray(self.model.predict(frame)).reshape(-1)
        if raw_predictions.size != 1:
            raise ValueError(f"Expected one prediction, received {raw_predictions.size}")
        prediction = float(raw_predictions[0])
        if not np.isfinite(prediction):
            raise ValueError("Model returned a non-finite prediction")
        return prediction


def load_champion(settings: Settings | None = None) -> LoadedChampion:
    """Resolve the ``champion`` alias and load its MLflow pyfunc model."""
    settings = settings or get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_tracking_uri)

    client = MlflowClient(
        tracking_uri=settings.mlflow_tracking_uri,
        registry_uri=settings.mlflow_tracking_uri,
    )
    version = client.get_model_version_by_alias(
        settings.registered_model_name, settings.champion_alias
    )
    if version.status != "READY":
        raise RuntimeError(
            f"Champion version {version.version} is not READY (status={version.status})"
        )

    model_uri = f"models:/{settings.registered_model_name}@{settings.champion_alias}"
    logger.info("Loading champion model from %s", model_uri)
    model = mlflow.pyfunc.load_model(model_uri)

    metrics: dict[str, float] = {}
    try:
        run = client.get_run(version.run_id)
        metrics = {
            key: float(value)
            for key, value in run.data.metrics.items()
            if key.startswith("test_")
        }
    except Exception:  # pragma: no cover - metadata is useful but not required to serve
        logger.warning("Could not load run metrics for champion version %s", version.version, exc_info=True)

    info = ChampionInfo(
        model_name=settings.registered_model_name,
        version=str(version.version),
        champion_alias=settings.champion_alias,
        run_id=version.run_id,
        source="mlflow-registry",
        metrics=metrics,
        quality_gate=(version.tags or {}).get("quality_gate"),
        loaded_at=utc_now(),
    )
    logger.info(
        "Champion loaded: %s version %s (test RMSE=%s)",
        info.model_name,
        info.version,
        metrics.get("test_rmse", "unknown"),
    )
    return LoadedChampion(
        model=model,
        info=info,
        feature_columns=settings.feature_columns,
    )


class ModelState:
    """Mutable state owned by the FastAPI application."""

    def __init__(self) -> None:
        self.champion: LoadedChampion | None = None
        self.error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.champion is not None

    def load(self, settings: Settings | None = None) -> bool:
        try:
            self.champion = load_champion(settings)
            self.error = None
            return True
        except Exception as exc:
            self.champion = None
            self.error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Champion model could not be loaded. The API is up but /predict and "
                "/model-info will return 503. Run `python -m mlops.pipeline` first. Error: %s",
                exc,
            )
            return False

    def predict(self, features: dict[str, float]) -> float:
        if self.champion is None:
            raise ModelUnavailableError("Champion model is not loaded")
        return self.champion.predict(features)
