"""Pruebas de los pipelines de modelo candidato (preprocesamiento y predicción)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_diabetes

from mlops.config import FEATURE_COLUMNS, TARGET_COLUMN
from mlops.training.models import build_candidates


def _frame():
    return load_diabetes(as_frame=True).frame.copy()


def test_candidate_pipelines_include_median_imputer():
    candidates = build_candidates(random_state=42)

    for pipeline in candidates.values():
        assert "imputer" in pipeline.named_steps
        assert "model" in pipeline.named_steps


def test_candidate_pipeline_imputes_missing_values_from_training_data():
    frame = _frame()
    pipeline = build_candidates(random_state=42)["linear_regression"]
    features = frame.loc[:, list(FEATURE_COLUMNS)]

    X_fit = features.iloc[:60].copy()
    X_fit.loc[0, "bmi"] = np.nan
    pipeline.fit(X_fit, frame.loc[:, TARGET_COLUMN].iloc[:60])

    X_new = features.iloc[60:65].copy()
    X_new.loc[X_new.index[0], "bmi"] = np.nan
    predictions = pipeline.predict(X_new)

    assert len(predictions) == 5
    assert np.isfinite(np.asarray(predictions, dtype=float)).all()
