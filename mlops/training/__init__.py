"""Paquete de entrenamiento y registro."""

from mlops.training.registry import (
    PromotionDecision,
    decide_promotion,
    register_and_select,
)
from mlops.training.train import load_training_summary, regression_metrics, train_models

__all__ = [
    "PromotionDecision",
    "decide_promotion",
    "load_training_summary",
    "regression_metrics",
    "register_and_select",
    "train_models",
]
