"""Etapa de extracción de datos.

El notebook exploratorio carga el dataset de diabetes integrado en
scikit-learn. En un proyecto real este módulo descargaría datos de una base de
datos, un almacén de objetos o una API; lo importante es que el resto del
pipeline consume un archivo sin procesar.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_diabetes

from mlops.config import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


def extract_data(output_path: Path) -> pd.DataFrame:
    """Carga el dataset de diabetes y lo persiste como un CSV sin procesar e inmutable."""
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
