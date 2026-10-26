"""Pydantic request/response schemas for the serving API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FEATURE_EXAMPLE = {
    "age": 0.0380759064,
    "sex": 0.0506801187,
    "bmi": 0.0616962065,
    "bp": 0.0218723549,
    "s1": -0.0442234984,
    "s2": -0.0348207628,
    "s3": -0.0434008456,
    "s4": -0.0025922620,
    "s5": 0.0199074862,
    "s6": -0.0176461252,
}


class PatientFeatures(BaseModel):
    """One patient record with the ten standardized diabetes features."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={"example": FEATURE_EXAMPLE},
    )

    age: float = Field(..., description="Age (standardized)")
    sex: float = Field(..., description="Sex (standardized)")
    bmi: float = Field(..., description="Body mass index (standardized)")
    bp: float = Field(..., description="Average blood pressure (standardized)")
    s1: float = Field(..., description="Total serum cholesterol (standardized)")
    s2: float = Field(..., description="Low-density lipoproteins (standardized)")
    s3: float = Field(..., description="High-density lipoproteins (standardized)")
    s4: float = Field(..., description="Total cholesterol / HDL ratio (standardized)")
    s5: float = Field(..., description="Log serum triglycerides (standardized)")
    s6: float = Field(..., description="Blood sugar level (standardized)")


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    champion_alias: str
    run_id: str
    source: str
    metrics: dict[str, float]
    quality_gate: str | None = None
    loaded_at: str


class PredictionResponse(BaseModel):
    prediction: float
    model_name: str
    model_version: str
    champion_alias: str
