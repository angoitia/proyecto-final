"""División determinista en entrenamiento y prueba."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from mlops.config import FEATURE_COLUMNS, TARGET_COLUMN
from mlops.utils import utc_now

logger = logging.getLogger(__name__)


def split_data(
    dataframe: pd.DataFrame,
    test_size: float,
    random_state: int,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide un dataset limpio en los conjuntos de entrenamiento y prueba."""
    columns = list(feature_columns) + [target_column]
    missing_columns = [column for column in columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Cannot split data; missing required columns: {missing_columns}")

    train, test = train_test_split(
        dataframe.loc[:, columns],
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    logger.info(
        "Split dataset with test_size=%s, random_state=%s: train=%s rows, test=%s rows",
        test_size,
        random_state,
        len(train),
        len(test),
    )
    return train, test


def build_split_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    test_size: float,
    random_state: int,
    target_column: str = TARGET_COLUMN,
) -> dict[str, Any]:
    """Metadatos útiles para auditar la división."""
    return {
        "created_at": utc_now(),
        "test_size": test_size,
        "random_state": random_state,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_target_mean": float(train[target_column].mean()),
        "test_target_mean": float(test[target_column].mean()),
        "train_target_std": float(train[target_column].std()),
        "test_target_std": float(test[target_column].std()),
    }
