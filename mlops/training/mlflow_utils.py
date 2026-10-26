"""MLflow configuration shared by training and registry stages."""

from __future__ import annotations

import mlflow
from mlflow import MlflowClient

from mlops.config import Settings


def configure_mlflow(settings: Settings) -> None:
    """Point MLflow at the configured tracking/registry backend and experiment.

    For local SQLite/file backends the artifact location is made absolute, so
    models can be loaded from any working directory (pipeline, API, tests).
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
