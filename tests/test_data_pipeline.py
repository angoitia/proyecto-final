"""Unit tests for extraction, validation, cleaning and splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes

from mlops.config import FEATURE_COLUMNS, TARGET_COLUMN
from mlops.data.clean import clean_data
from mlops.data.extract import extract_data
from mlops.data.split import split_data
from mlops.data.validate import DataValidationError, validate_data


@pytest.fixture()
def raw_dataframe() -> pd.DataFrame:
    return load_diabetes(as_frame=True).frame.copy()


def test_extract_data_writes_raw_csv(tmp_path, raw_dataframe):
    output_path = tmp_path / "raw" / "diabetes.csv"
    dataframe = extract_data(output_path)

    assert output_path.exists()
    assert dataframe.shape == (442, 11)
    assert list(dataframe.columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]
    assert pd.read_csv(output_path).shape == dataframe.shape


def test_validation_passes_on_original_dataset(raw_dataframe):
    report = validate_data(raw_dataframe)

    assert report["status"] == "passed"
    assert report["rows"] == 442
    assert report["errors"] == []


def test_validation_rejects_missing_column(raw_dataframe):
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_data(raw_dataframe.drop(columns=["bmi"]))


def test_validation_rejects_non_numeric_feature(raw_dataframe):
    dataframe = raw_dataframe.copy()
    dataframe["bmi"] = dataframe["bmi"].astype(str)

    with pytest.raises(DataValidationError, match="must be numeric"):
        validate_data(dataframe)


def test_validation_rejects_duplicate_columns(raw_dataframe):
    dataframe = pd.concat([raw_dataframe, raw_dataframe[["bmi"]]], axis=1)

    with pytest.raises(DataValidationError, match="Duplicate column names"):
        validate_data(dataframe)


def test_validation_rejects_missing_target(raw_dataframe):
    dataframe = raw_dataframe.copy()
    dataframe.loc[0, TARGET_COLUMN] = np.nan

    with pytest.raises(DataValidationError, match="Target column contains missing values"):
        validate_data(dataframe)


def test_cleaning_handles_duplicates_missing_values_and_invalid_targets(raw_dataframe):
    dataframe = raw_dataframe.head(20).copy()
    dataframe.loc[0, "bmi"] = np.nan
    dataframe.loc[1, "bmi"] = np.inf
    dataframe.loc[2, TARGET_COLUMN] = -5
    dataframe = pd.concat([dataframe, dataframe.iloc[[3]]], ignore_index=True)

    cleaned, report = clean_data(dataframe)

    assert report["rows_in"] == 21
    assert report["rows_out"] == 19
    assert report["duplicates_removed"] == 1
    assert report["removed_invalid_target"] == 1
    assert report["imputed_values"] >= 2
    assert not cleaned[list(FEATURE_COLUMNS)].isna().any().any()
    assert np.isfinite(cleaned[list(FEATURE_COLUMNS)].to_numpy()).all()
    assert (cleaned[TARGET_COLUMN] > 0).all()


def test_cleaning_rejects_entirely_missing_feature(raw_dataframe):
    dataframe = raw_dataframe.copy()
    dataframe["bmi"] = np.nan

    with pytest.raises(ValueError, match="entirely missing"):
        clean_data(dataframe)


def test_split_is_deterministic(raw_dataframe):
    train_a, test_a = split_data(raw_dataframe, test_size=0.2, random_state=42)
    train_b, test_b = split_data(raw_dataframe, test_size=0.2, random_state=42)

    assert len(train_a) == 353
    assert len(test_a) == 89
    assert train_a.equals(train_b)
    assert test_a.equals(test_b)
    combined = pd.concat([train_a, test_a], ignore_index=True)
    assert len(combined) == len(raw_dataframe)
    assert combined.duplicated().sum() == 0


def test_split_rejects_missing_column(raw_dataframe):
    with pytest.raises(ValueError, match="missing required columns"):
        split_data(raw_dataframe.drop(columns=["age"]), test_size=0.2, random_state=42)
