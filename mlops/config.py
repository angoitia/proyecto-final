"""Configuración central del proyecto.

Todas las rutas y los valores ajustables viven aquí para que las capas de
datos, entrenamiento, registro e API no los definan directamente. Las variables
de entorno hacen que el proyecto sea fácil de ejecutar localmente, en CI o contra
un servidor MLflow remoto.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_COLUMNS: tuple[str, ...] = (
    "age",
    "sex",
    "bmi",
    "bp",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
)
TARGET_COLUMN = "target"


def _env_path(name: str, default: Path) -> Path:
    """Lee una ruta desde el entorno, resolviendo las rutas relativas al repositorio."""
    raw = os.getenv(name)
    path = Path(raw).expanduser() if raw else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None else default


@dataclass(frozen=True)
class Settings:
    """Ajustes de ejecución compartidos por todas las etapas del pipeline."""

    project_root: Path = PROJECT_ROOT
    data_raw_dir: Path = field(default_factory=lambda: _env_path("DATA_RAW_DIR", PROJECT_ROOT / "data" / "raw"))
    data_processed_dir: Path = field(
        default_factory=lambda: _env_path("DATA_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed")
    )
    reports_dir: Path = field(default_factory=lambda: _env_path("REPORTS_DIR", PROJECT_ROOT / "artifacts" / "reports"))
    mlflow_artifacts_dir: Path = field(
        default_factory=lambda: _env_path("MLFLOW_ARTIFACT_DIR", PROJECT_ROOT / "mlruns")
    )
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv(
            "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
        )
    )
    experiment_name: str = field(default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT", "diabetes-progression"))
    registered_model_name: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", "diabetes-progression-model")
    )
    champion_alias: str = field(default_factory=lambda: os.getenv("CHAMPION_ALIAS", "champion"))
    rmse_threshold: float = field(default_factory=lambda: _env_float("RMSE_THRESHOLD", 60.0))
    test_size: float = field(default_factory=lambda: _env_float("TEST_SIZE", 0.2))
    validation_size: float = field(default_factory=lambda: _env_float("VALIDATION_SIZE", 0.25))
    random_state: int = field(default_factory=lambda: int(os.getenv("RANDOM_STATE", "42")))
    target_column: str = TARGET_COLUMN
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS

    @property
    def raw_data_path(self) -> Path:
        return self.data_raw_dir / "diabetes.csv"

    def __post_init__(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError(f"TEST_SIZE must be between 0 and 1, got {self.test_size}")
        if self.rmse_threshold <= 0:
            raise ValueError(f"RMSE_THRESHOLD must be positive, got {self.rmse_threshold}")

    @property
    def clean_data_path(self) -> Path:
        return self.data_processed_dir / "diabetes_clean.csv"

    @property
    def train_data_path(self) -> Path:
        return self.data_processed_dir / "train.csv"

    @property
    def test_data_path(self) -> Path:
        return self.data_processed_dir / "test.csv"

    @property
    def validation_data_path(self) -> Path:
        return self.data_processed_dir / "validation.csv"

    @property
    def extraction_report_path(self) -> Path:
        return self.reports_dir / "extraction_report.json"

    @property
    def validation_report_path(self) -> Path:
        return self.reports_dir / "validation_report.json"

    @property
    def cleaning_report_path(self) -> Path:
        return self.reports_dir / "cleaning_report.json"

    @property
    def split_report_path(self) -> Path:
        return self.reports_dir / "split_report.json"

    @property
    def training_summary_path(self) -> Path:
        return self.reports_dir / "training_summary.json"

    @property
    def champion_selection_path(self) -> Path:
        return self.reports_dir / "champion_selection.json"

    def ensure_directories(self) -> None:
        for directory in (
            self.data_raw_dir,
            self.data_processed_dir,
            self.reports_dir,
            self.mlflow_artifacts_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve los ajustes en caché (limpiar la caché tras cambiar variables de entorno)."""
    return Settings()
