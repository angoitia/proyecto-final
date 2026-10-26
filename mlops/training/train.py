"""Model training with MLflow experiment tracking.

This stage trains every candidate, logs parameters/metrics/artifacts in
individual MLflow runs and records which run produced the best test RMSE.
Registration happens in the next stage (``mlops.training.registry``).
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mlops.config import Settings, get_settings
from mlops.training.mlflow_utils import configure_mlflow
from mlops.training.models import build_candidates
from mlops.utils import git_sha, read_json, utc_now, write_json

logger = logging.getLogger(__name__)


def regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    """Compute the metrics used to evaluate every candidate."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def _load_split(path: Path, settings: Settings) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}. Run the 'split' stage first.")
    dataframe = pd.read_csv(path)
    required = list(settings.feature_columns) + [settings.target_column]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Split file {path} is missing columns: {missing}")
    return dataframe.loc[:, list(settings.feature_columns)], dataframe.loc[:, settings.target_column]


def _loggable_params(estimator: Any) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, value in estimator.get_params(deep=False).items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            params[key] = value
        else:
            params[key] = str(value)
    return params


def _feature_importance_frame(estimator: Any, feature_columns: tuple[str, ...]) -> pd.DataFrame | None:
    final_estimator = estimator.named_steps["model"]
    if hasattr(final_estimator, "feature_importances_"):
        values = final_estimator.feature_importances_
        kind = "importance"
    elif hasattr(final_estimator, "coef_"):
        values = np.ravel(final_estimator.coef_)
        kind = "coefficient"
    else:
        return None
    return pd.DataFrame({"feature": feature_columns, kind: values}).sort_values(
        kind, ascending=False, ignore_index=True
    )


def train_models(settings: Settings | None = None) -> dict[str, Any]:
    """Train all candidates and persist a training summary for registration."""
    settings = settings or get_settings()
    settings.ensure_directories()

    X_train, y_train = _load_split(settings.train_data_path, settings)
    X_test, y_test = _load_split(settings.test_data_path, settings)

    configure_mlflow(settings)
    commit = git_sha()
    candidates: list[dict[str, Any]] = []

    for model_name, estimator in build_candidates(settings.random_state).items():
        logger.info("Training candidate: %s", model_name)
        with mlflow.start_run(run_name=f"train_{model_name}") as run:
            mlflow.set_tags(
                {
                    "stage": "training",
                    "model_name": model_name,
                    "selection_metric": "test_rmse",
                    **({"git_sha": commit} if commit else {}),
                }
            )
            mlflow.log_params(_loggable_params(estimator))
            mlflow.log_params(
                {
                    "candidate": model_name,
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                    "feature_count": len(settings.feature_columns),
                }
            )

            estimator.fit(X_train, y_train)
            train_predictions = estimator.predict(X_train)
            test_predictions = estimator.predict(X_test)

            train_metrics = regression_metrics(y_train, train_predictions)
            test_metrics = regression_metrics(y_test, test_predictions)
            mlflow.log_metrics({f"train_{key}": value for key, value in train_metrics.items()})
            mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})

            signature = infer_signature(X_train, train_predictions)
            log_model_kwargs: dict[str, Any] = {
                "sk_model": estimator,
                "name": "model",
                "signature": signature,
                "input_example": X_train.head(3),
            }
            if "skops_trusted_types" in inspect.signature(mlflow.sklearn.log_model).parameters:
                # MLflow >= 3 serializes scikit-learn models with skops by
                # default. Tree-based candidates contain this internally
                # generated type; the artifact is produced by our own pipeline.
                log_model_kwargs["skops_trusted_types"] = ["sklearn.tree._tree.Tree"]
            model_info = mlflow.sklearn.log_model(**log_model_kwargs)
            mlflow.log_dict(
                {"train": train_metrics, "test": test_metrics},
                "metrics.json",
            )

            importance = _feature_importance_frame(estimator, settings.feature_columns)
            if importance is not None:
                mlflow.log_text(importance.to_csv(index=False), "feature_importance.csv")

            run_id = run.info.run_id
            logger.info(
                "Candidate %s finished (run_id=%s, test RMSE=%.4f)",
                model_name,
                run_id,
                test_metrics["rmse"],
            )

        candidates.append(
            {
                "model_name": model_name,
                "run_id": run_id,
                "model_uri": model_info.model_uri,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
            }
        )

    best = min(candidates, key=lambda candidate: candidate["test_metrics"]["rmse"])
    summary: dict[str, Any] = {
        "generated_at": utc_now(),
        "experiment_name": settings.experiment_name,
        "registered_model_name": settings.registered_model_name,
        "champion_alias": settings.champion_alias,
        "selection_metric": "test_rmse",
        "rmse_threshold": settings.rmse_threshold,
        "random_state": settings.random_state,
        "test_size": settings.test_size,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "candidates": candidates,
        "best": best,
    }
    write_json(settings.training_summary_path, summary)
    logger.info(
        "Best candidate: %s (test RMSE=%.4f, run_id=%s)",
        best["model_name"],
        best["test_metrics"]["rmse"],
        best["run_id"],
    )
    return summary


def load_training_summary(settings: Settings | None = None) -> dict[str, Any]:
    """Load the summary produced by :func:`train_models`."""
    settings = settings or get_settings()
    return read_json(settings.training_summary_path)
