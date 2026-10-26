"""Candidate model definitions."""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_candidates(random_state: int) -> dict[str, Pipeline]:
    """Return the models compared in every training run.

    The candidate definitions mirror the notebook's baseline comparison. Every
    candidate includes its own preprocessing inside a scikit-learn Pipeline, so
    the exact object logged to MLflow is also the object served by the API (no
    training/serving skew).
    """
    return {
        "linear_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=100,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    GradientBoostingRegressor(random_state=random_state),
                ),
            ]
        ),
    }
