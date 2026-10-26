"""Data extraction stage.

The exploratory notebook loads scikit-learn's built-in diabetes dataset. In a
real project this module would fetch data from a database, object storage or an
API; the important part is that the rest of the pipeline consumes a raw file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_diabetes

from mlops.config import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


def extract_data(output_path: Path) -> pd.DataFrame:
    """Load the diabetes dataset and persist it as an immutable raw CSV."""
    dataset = load_diabetes(as_frame=True)
    dataframe = dataset.frame.copy()

    expected_columns = list(FEATURE_COLUMNS) + [TARGET_COLUMN]
    if list(dataframe.columns) != expected_columns:
        dataframe = dataframe.rename(columns={dataframe.columns[-1]: TARGET_COLUMN})
        dataframe = dataframe[expected_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)

    logger.info(
        "Extracted %s rows and %s columns from scikit-learn diabetes dataset -> %s",
        dataframe.shape[0],
        dataframe.shape[1],
        output_path,
    )
    return dataframe
