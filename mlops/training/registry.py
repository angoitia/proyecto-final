"""Registro de modelos y selección del campeón.

Un candidato se promueve al alias ``champion`` solo cuando:

1. su RMSE de prueba pasa el quality gate (``RMSE_THRESHOLD``), y
2. no hay un campeón actual o su RMSE es mejor o igual al RMSE del campeón
   actual.

La decisión se guarda tanto en MLflow como en un informe JSON local.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from mlops.config import Settings, get_settings
from mlops.training.mlflow_utils import configure_mlflow
from mlops.utils import git_sha, read_json, utc_now, write_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    gate_passed: bool
    reason: str


def decide_promotion(
    new_rmse: float,
    threshold: float,
    current_champion_rmse: float | None = None,
) -> PromotionDecision:
    """Política pura de selección del campeón (fácil de probar unitariamente)."""
    if new_rmse > threshold:
        return PromotionDecision(
            promote=False,
            gate_passed=False,
            reason=(
                f"Quality gate failed: test RMSE {new_rmse:.4f} is above "
                f"the threshold {threshold:.4f}."
            ),
        )
    if current_champion_rmse is None:
        return PromotionDecision(
            promote=True,
            gate_passed=True,
            reason="No current champion; candidate passed the quality gate and was promoted.",
        )
    if new_rmse <= current_champion_rmse:
        return PromotionDecision(
            promote=True,
            gate_passed=True,
            reason=(
                f"Candidate test RMSE {new_rmse:.4f} is better than or equal to the "
                f"current champion's {current_champion_rmse:.4f}."
            ),
        )
    return PromotionDecision(
        promote=False,
        gate_passed=True,
        reason=(
            f"Candidate test RMSE {new_rmse:.4f} is worse than the current "
            f"champion's {current_champion_rmse:.4f}; the current champion is kept."
        ),
    )


def _version_rmse(client: MlflowClient, version: Any) -> float | None:
    """Lee el RMSE de prueba de la corrida, con fallback a la etiqueta de la versión del modelo."""
    try:
        run = client.get_run(version.run_id)
        value = run.data.metrics.get("test_rmse")
        if value is not None:
            return float(value)
    except MlflowException:
        logger.warning("Could not read run %s for champion version %s", version.run_id, version.version)

    tag_value = (version.tags or {}).get("test_rmse")
    return float(tag_value) if tag_value is not None else None


def _get_current_champion(
    client: MlflowClient, model_name: str, alias: str
) -> dict[str, Any] | None:
    try:
        version = client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None

    return {
        "version": str(version.version),
        "run_id": version.run_id,
        "test_rmse": _version_rmse(client, version),
    }


def _wait_until_ready(
    client: MlflowClient, model_name: str, version: str, timeout_seconds: float = 60.0
) -> Any:
    deadline = time.monotonic() + timeout_seconds
    latest = client.get_model_version(model_name, version)
    while latest.status != "READY":
        if latest.status == "FAILED_REGISTRATION":
            raise RuntimeError(f"MLflow failed to register model version {version}")
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Model version {version} did not become READY within {timeout_seconds}s"
            )
        time.sleep(0.5)
        latest = client.get_model_version(model_name, version)
    return latest


def register_and_select(
    summary: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Registra la mejor corrida y la promueve al campeón cuando califica."""
    settings = settings or get_settings()
    if summary is None:
        summary_path = settings.training_summary_path
        summary = read_json(summary_path)

    best = summary["best"]
    best_metrics = best["test_metrics"]
    new_rmse = float(best_metrics["rmse"])
    run_id = best["run_id"]

    configure_mlflow(settings)
    client = MlflowClient(
        tracking_uri=settings.mlflow_tracking_uri,
        registry_uri=settings.mlflow_tracking_uri,
    )

    selected_at = utc_now()
    commit = git_sha()
    tags = {
        "test_rmse": f"{new_rmse:.6f}",
        "test_mae": f"{float(best_metrics['mae']):.6f}",
        "test_r2": f"{float(best_metrics['r2']):.6f}",
        "training_run_id": run_id,
        "quality_gate_threshold": f"{settings.rmse_threshold:.6f}",
        "selected_at": selected_at,
        "candidate_model": str(best["model_name"]),
    }
    if commit:
        tags["git_sha"] = commit

    logger.info("Registering run %s as a new version of '%s'", run_id, settings.registered_model_name)
    model_uri = best.get("model_uri") or f"runs:/{run_id}/model"
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=settings.registered_model_name,
        tags=tags,
    )
    model_version = _wait_until_ready(
        client, settings.registered_model_name, str(model_version.version)
    )

    current_champion = _get_current_champion(
        client, settings.registered_model_name, settings.champion_alias
    )
    current_rmse = current_champion["test_rmse"] if current_champion else None
    decision = decide_promotion(new_rmse, settings.rmse_threshold, current_rmse)

    if decision.promote:
        client.set_registered_model_alias(
            name=settings.registered_model_name,
            alias=settings.champion_alias,
            version=str(model_version.version),
        )
        logger.info(
            "Promoted version %s to alias '%s' (test RMSE=%.4f)",
            model_version.version,
            settings.champion_alias,
            new_rmse,
        )
    else:
        logger.warning("Candidate not promoted: %s", decision.reason)

    promotion_status = "promoted" if decision.promote else "not_promoted"
    client.set_model_version_tag(
        settings.registered_model_name, str(model_version.version), "quality_gate", "passed" if decision.gate_passed else "failed"
    )
    client.set_model_version_tag(
        settings.registered_model_name, str(model_version.version), "promotion_status", promotion_status
    )
    client.update_model_version(
        name=settings.registered_model_name,
        version=str(model_version.version),
        description=(
            f"Candidate '{best['model_name']}' trained by the automated pipeline. "
            f"Test RMSE={new_rmse:.4f}; threshold={settings.rmse_threshold:.4f}."
        ),
    )

    if decision.promote:
        final_champion = {
            "version": str(model_version.version),
            "run_id": run_id,
            "test_rmse": new_rmse,
            "source": "promoted_this_run",
        }
    else:
        final_champion = {
            **(current_champion or {}),
            "source": "previous_champion" if current_champion else "none",
        }

    decision_payload: dict[str, Any] = {
        "decided_at": selected_at,
        "registered_model_name": settings.registered_model_name,
        "champion_alias": settings.champion_alias,
        "rmse_threshold": settings.rmse_threshold,
        "candidate": {
            "model_name": best["model_name"],
            "run_id": run_id,
            "version": str(model_version.version),
            "metrics": {"test_rmse": new_rmse, **{k: float(v) for k, v in best_metrics.items()}},
        },
        "current_champion": current_champion,
        "gate_passed": decision.gate_passed,
        "promoted": decision.promote,
        "reason": decision.reason,
        "final_champion": final_champion,
    }

    with mlflow.start_run(run_name="champion_selection") as selection_run:
        mlflow.set_tags(
            {
                "stage": "champion_selection",
                "registered_model_name": settings.registered_model_name,
                "candidate_version": str(model_version.version),
                "promoted": str(decision.promote).lower(),
                **({"git_sha": commit} if commit else {}),
            }
        )
        mlflow.log_params(
            {
                "candidate_model": best["model_name"],
                "candidate_run_id": run_id,
                "rmse_threshold": settings.rmse_threshold,
                "current_champion_version": (current_champion or {}).get("version", "none"),
            }
        )
        mlflow.log_metrics(
            {
                "candidate_test_rmse": new_rmse,
                "threshold": settings.rmse_threshold,
                "promoted": int(decision.promote),
            }
        )
        mlflow.log_dict(decision_payload, "champion_selection.json")
        decision_payload["selection_run_id"] = selection_run.info.run_id

    write_json(settings.champion_selection_path, decision_payload)
    return decision_payload


def load_selection(settings: Settings | None = None) -> dict[str, Any]:
    """Carga el último informe de selección del campeón."""
    settings = settings or get_settings()
    return read_json(settings.champion_selection_path)
