"""Definiciones de los modelos candidatos."""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def build_candidates(random_state: int) -> dict[str, Pipeline]:
    """Devuelve los modelos que se comparan en cada ejecución de entrenamiento.

    Las definiciones de candidatos reflejan la comparación de referencia del
    notebook. Cada candidato incluye su propio preprocesamiento dentro de un
    ``Pipeline`` de scikit-learn, de modo que el objeto exacto registrado en
    MLflow también es el objeto servido por la API (sin desviación entre el
    entrenamiento y el servicio).

    El ``Imputer`` con estrategia ``median`` se ajusta **sobre los datos de
    entrenamiento** dentro del pipeline y se aplica automáticamente a los
    conjuntos de validación y prueba, evitando así la fuga de información que
    supondría imputar con estadísticas calculadas sobre el dataset completo.
    """

    return {
        "linear_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
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
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    GradientBoostingRegressor(random_state=random_state),
                ),
            ]
        ),
    }
