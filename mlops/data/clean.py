"""Etapa de limpieza de datos.

El dataset de diabetes de scikit-learn ya está limpio, pero el flujo de
producción sigue aplicando una política de limpieza determinista para que el
mismo código pueda manejar datos nuevos con valores faltantes, duplicados o
etiquetas no válidas.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from mlops.config import FEATURE_COLUMNS, TARGET_COLUMN
from mlops.utils import utc_now

logger = logging.getLogger(__name__)


def clean_data(
    dataframe: pd.DataFrame,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aplica la política básica de limpieza del proyecto y devuelve un informe.

    Política:
    - Mantener solo las columnas del contrato y forzar tipos numéricos.
    - Tratar ``inf`` como un valor faltante.
    - Eliminar las filas sin un objetivo utilizable (finito y estrictamente
      positivo).
    - Eliminar las filas duplicadas exactas.

    Los valores faltantes de las características **no se imputan aquí**: la
    imputación con la mediana se realiza dentro del pipeline de entrenamiento,
    calculando la estadística **solo sobre los datos de entrenamiento**. Así se
    evita filtrar información del conjunto de validación/prueba al modelo.
    """
    required_columns = list(feature_columns) + [target_column]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Cannot clean data; missing required columns: {missing_columns}")

    cleaned = dataframe.loc[:, required_columns].copy()
    rows_in = int(len(cleaned))

    for column in required_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned[list(feature_columns)] = cleaned[list(feature_columns)].replace([np.inf, -np.inf], np.nan)
    cleaned[target_column] = cleaned[target_column].replace([np.inf, -np.inf], np.nan)

    missing_targets = int(cleaned[target_column].isna().sum())
    cleaned = cleaned.dropna(subset=[target_column])

    invalid_targets = int((cleaned[target_column] <= 0).sum())
    cleaned = cleaned.loc[cleaned[target_column] > 0]

    duplicates = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates()

    missing_by_column: dict[str, int] = {
        str(column): int(cleaned[column].isna().sum())
        for column in feature_columns
        if column in cleaned.columns and int(cleaned[column].isna().sum()) > 0
    }

    all_missing = [
        column
        for column in feature_columns
        if column in cleaned.columns and bool(cleaned[column].isna().all())
    ]
    if all_missing:
        raise ValueError(f"Cannot impute features that are entirely missing: {all_missing}")

    cleaned = cleaned.reset_index(drop=True)

    report: dict[str, Any] = {
        "status": "passed",
        "cleaned_at": utc_now(),
        "rows_in": rows_in,
        "rows_out": int(len(cleaned)),
        "rows_removed": rows_in - int(len(cleaned)),
        "removed_missing_target": missing_targets,
        "removed_invalid_target": invalid_targets,
        "duplicates_removed": duplicates,
        "missing_values_by_column": missing_by_column,
    }

    logger.info(
        "Cleaning finished: %s -> %s rows (%s duplicates, %s missing feature values)",
        rows_in,
        len(cleaned),
        duplicates,
        sum(report["missing_values_by_column"].values()),
    )
    return cleaned, report
