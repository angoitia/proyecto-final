"""API tests use a fake in-memory champion so they do not need MLflow."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from mlops.api.main import create_app, get_model_state
from mlops.api.model_loader import ChampionInfo
from mlops.api.schemas import FEATURE_EXAMPLE


@dataclass
class FakeChampion:
    info: ChampionInfo
    fail_prediction: bool = False

    def predict(self, features: dict[str, float]) -> float:
        if self.fail_prediction:
            raise RuntimeError("simulated model failure")
        assert set(features) == set(FEATURE_EXAMPLE)
        return 152.75


class FakeState:
    def __init__(self, loaded: bool = True, fail_prediction: bool = False) -> None:
        info = ChampionInfo(
            model_name="diabetes-progression-model",
            version="7",
            champion_alias="champion",
            run_id="run-123",
            source="test",
            metrics={"test_rmse": 53.2, "test_mae": 42.1, "test_r2": 0.47},
            quality_gate="passed",
        )
        self.champion = FakeChampion(info=info, fail_prediction=fail_prediction) if loaded else None

    @property
    def is_loaded(self) -> bool:
        return self.champion is not None

    def predict(self, features: dict[str, float]) -> float:
        if self.champion is None:
            raise RuntimeError("model is not loaded")
        return self.champion.predict(features)


@pytest.fixture()
def make_client():
    def _make_client(state: FakeState) -> TestClient:
        app = create_app(autoload=False)
        app.dependency_overrides[get_model_state] = lambda: state
        return TestClient(app)

    return _make_client


def test_health_when_model_is_loaded(make_client):
    with make_client(FakeState(loaded=True)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_name": "diabetes-progression-model",
        "model_version": "7",
    }


def test_health_when_model_is_unavailable(make_client):
    with make_client(FakeState(loaded=False)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_model_info_returns_registry_metadata(make_client):
    with make_client(FakeState(loaded=True)) as client:
        response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "diabetes-progression-model"
    assert body["version"] == "7"
    assert body["champion_alias"] == "champion"
    assert body["metrics"]["test_rmse"] == 53.2
    assert body["quality_gate"] == "passed"


def test_model_info_returns_503_without_model(make_client):
    with make_client(FakeState(loaded=False)) as client:
        response = client.get("/model-info")

    assert response.status_code == 503


def test_predict_returns_prediction_and_version(make_client):
    with make_client(FakeState(loaded=True)) as client:
        response = client.post("/predict", json=FEATURE_EXAMPLE)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == pytest.approx(152.75)
    assert body["model_version"] == "7"
    assert body["champion_alias"] == "champion"


def test_predict_rejects_missing_feature(make_client):
    payload = dict(FEATURE_EXAMPLE)
    payload.pop("bmi")

    with make_client(FakeState(loaded=True)) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_extra_feature(make_client):
    payload = {**FEATURE_EXAMPLE, "smoker": 1}

    with make_client(FakeState(loaded=True)) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_wrong_type(make_client):
    payload = {**FEATURE_EXAMPLE, "bmi": "not-a-number"}

    with make_client(FakeState(loaded=True)) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_returns_503_without_model(make_client):
    with make_client(FakeState(loaded=False)) as client:
        response = client.post("/predict", json=FEATURE_EXAMPLE)

    assert response.status_code == 503


def test_predict_returns_500_on_model_failure(make_client):
    with make_client(FakeState(loaded=True, fail_prediction=True)) as client:
        response = client.post("/predict", json=FEATURE_EXAMPLE)

    assert response.status_code == 500
    assert response.json()["detail"] == "Prediction failed."
