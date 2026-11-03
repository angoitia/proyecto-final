"""Configuración de MLflow compartida por las etapas de entrenamiento y registro."""

from __future__ import annotations

import mlflow
from mlflow import MlflowClient

from mlops.config import Settings


def configure_mlflow(settings: Settings) -> None:
    """Apunta a MLflow al backend de seguimiento/registro configurado y al experimento.

    Para backends locales SQLite/archivo la ubicación de los artefactos se hace
    absoluta, de modo que los modelos puedan cargarse desde cualquier directorio
    de trabajo (pipeline, API, tests).
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_tracking_uri)

    client = MlflowClient(
        tracking_uri=settings.mlflow_tracking_uri,
        registry_uri=settings.mlflow_tracking_uri,
    )
    if client.get_experiment_by_name(settings.experiment_name) is None:
        if settings.mlflow_tracking_uri.startswith(("sqlite:", "file:")):
            mlflow.create_experiment(
                settings.experiment_name,
                artifact_location=settings.mlflow_artifacts_dir.resolve().as_uri(),
            )
        else:
            mlflow.create_experiment(settings.experiment_name)

    mlflow.set_experiment(settings.experiment_name)
