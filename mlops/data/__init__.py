"""Paquete de ingeniería de datos: extracción, validación, limpieza y separación."""

from mlops.data.clean import clean_data
from mlops.data.extract import extract_data
from mlops.data.split import build_split_report, split_data
from mlops.data.validate import DataValidationError, validate_data

__all__ = [
    "DataValidationError",
    "build_split_report",
    "clean_data",
    "extract_data",
    "split_data",
    "validate_data",
]
