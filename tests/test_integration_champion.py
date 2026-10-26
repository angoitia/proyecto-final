"""Optional end-to-end test against the real MLflow registry.

Run it after the pipeline::

    RUN_INTEGRATION=1 pytest tests/test_integration_champion.py -q

CI enables it after training/registration, so the same test proves that the
actual champion artifact can be loaded and served.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mlops.api.main import create_app
from mlops.config import get_settings

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1",
    reason="Set RUN_INTEGRATION=1 after running the full pipeline.",
)


def test_real_champion_is_loaded_and_predicts():
    settings = get_settings()
    assert settings.test_data_path.exists(), "Run the full pipeline before this test"

    test_row = pd.read_csv(settings.test_data_path).iloc[0]
    payload = {
        column: float(test_row[column]) for column in settings.feature_columns
    }

    with TestClient(create_app()) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        model_info = client.get("/model-info")
        assert model_info.status_code == 200
        info = model_info.json()
        assert info["champion_alias"] == settings.champion_alias
        assert info["metrics"]["test_rmse"] <= settings.rmse_threshold

        prediction = client.post("/predict", json=payload)
        assert prediction.status_code == 200
        body = prediction.json()
        assert body["model_version"] == info["version"]
        assert body["champion_alias"] == settings.champion_alias
        assert isinstance(body["prediction"], float)
