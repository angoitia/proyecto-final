"""Definiciones de los modelos candidatos."""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_candidates(random_state: int) -> dict[str, Pipeline]:
    """Devuelve los modelos que se comparan en cada ejecución de entrenamiento.

    Las definiciones de candidatos reflejan la comparación de referencia del
    notebook. Cada candidato incluye su propio preprocesamiento dentro de un
    ``Pipeline`` de scikit-learn, de modo que el objeto exacto registrado en
    MLflow también es el objeto servido por la API (sin desviación entre el
    entrenamiento y el servicio).
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
