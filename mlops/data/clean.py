"""Data cleaning stage.

The scikit-learn diabetes dataset is already clean, but the production flow
still applies a deterministic cleaning policy so that the same code can handle
new data with missing values, duplicates or invalid labels.
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
    """Apply the project's basic cleaning policy and return a report.

    Policy:
    - Keep only contract columns and force numeric dtypes.
    - Treat ``inf`` as missing.
    - Drop rows without a usable (finite, strictly positive) target.
    - Drop exact duplicate rows.
    - Impute missing feature values with the column median.
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

    imputed_by_column = {column: int(cleaned[column].isna().sum()) for column in feature_columns}
    medians = cleaned[list(feature_columns)].median(numeric_only=True)
    all_missing = [column for column in feature_columns if pd.isna(medians[column])]
    if all_missing:
        raise ValueError(f"Cannot impute features that are entirely missing: {all_missing}")

    cleaned[list(feature_columns)] = cleaned[list(feature_columns)].fillna(medians)
    cleaned[list(feature_columns)] = cleaned[list(feature_columns)].astype(float)
    cleaned[target_column] = cleaned[target_column].astype(float)
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
        "imputed_values": int(sum(imputed_by_column.values())),
        "imputed_by_column": imputed_by_column,
        "imputation_medians": {column: float(value) for column, value in medians.items()},
    }

    logger.info(
        "Cleaning finished: %s -> %s rows (%s duplicates, %s imputed values)",
        rows_in,
        len(cleaned),
        duplicates,
        report["imputed_values"],
    )
    return cleaned, report
