"""Etapa de validación de datos.

Comprueba que el dataset bruto cumpla el contrato esperado por el entrenamiento
y el servicio de predicción. Cuando se detectan fallos se levanta
:class:`DataValidationError`, lo que hace que el pipeline se detenga antes de que
un dataset incorrecto llegue al registro de modelos.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from mlops.config import FEATURE_COLUMNS, TARGET_COLUMN
from mlops.utils import utc_now

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Se levanta cuando el dataset viola el esquema requerido."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Data validation failed: " + " | ".join(errors))


def validate_data(
    dataframe: pd.DataFrame,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
) -> dict[str, Any]:
    """Valida el esquema, los tipos de datos y la integridad del objetivo.

    Los valores faltantes de las características se reportan como avisos porque
    la etapa de limpieza se encarga de la imputación. Los objetivos faltantes o
    no finitos son fatales: una fila sin etiqueta no puede utilizarse para el
    aprendizaje supervisado.
    """
    errors: list[str] = []
    warnings: list[str] = []

    required_columns = list(feature_columns) + [target_column]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    duplicate_columns = dataframe.columns[dataframe.columns.duplicated()].tolist()
    if duplicate_columns:
        errors.append(f"Duplicate column names: {duplicate_columns}")

    if dataframe.empty:
        errors.append("Dataset is empty")

    missing_values: dict[str, int] = {}
    if not missing_columns and not duplicate_columns:
        for column in required_columns:
            missing_count = int(dataframe[column].isna().sum())
            if missing_count:
                missing_values[column] = missing_count

            if not is_numeric_dtype(dataframe[column]):
                errors.append(f"Column '{column}' must be numeric, got {dataframe[column].dtype}")

        if is_numeric_dtype(dataframe[target_column]) and not dataframe.empty:
            target = dataframe[target_column]
            if target.isna().any():
                errors.append("Target column contains missing values")
            finite_target = target.dropna().to_numpy(dtype=float, copy=False)
            if finite_target.size and not np.isfinite(finite_target).all():
                errors.append("Target column contains infinite values")
            if finite_target.size and (finite_target <= 0).any():
                warnings.append("Target column contains non-positive values; cleaning will remove them")

        for column in feature_columns:
            if column not in dataframe.columns or not is_numeric_dtype(dataframe[column]):
                continue
            finite_features = dataframe[column].dropna().to_numpy(dtype=float, copy=False)
            if finite_features.size and not np.isfinite(finite_features).all():
                warnings.append(f"Feature '{column}' contains infinite values; cleaning will treat them as missing")
            if dataframe[column].isna().all():
                errors.append(f"Feature '{column}' is entirely missing")

        if missing_values:
            warnings.append(f"Missing feature values detected: {missing_values}")

    report: dict[str, Any] = {
        "status": "failed" if errors else "passed",
        "checked_at": utc_now(),
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "required_columns": required_columns,
        "missing_values": missing_values,
        "errors": errors,
        "warnings": warnings,
    }

    if errors:
        raise DataValidationError(errors)

    logger.info(
        "Validation passed: %s rows, %s columns, %s warning(s)",
        dataframe.shape[0],
        dataframe.shape[1],
        len(warnings),
    )
    return report
