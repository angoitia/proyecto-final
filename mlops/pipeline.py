"""Pipeline orchestration.

Run every stage with::

    python -m mlops.pipeline

or a subset (always executed in the canonical order)::

    python -m mlops.pipeline --stages extract validate clean split
    python -m mlops.pipeline --stages train register
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Iterable

import pandas as pd

from mlops.config import Settings, get_settings
from mlops.data.clean import clean_data
from mlops.data.extract import extract_data
from mlops.data.split import build_split_report, split_data
from mlops.data.validate import DataValidationError, validate_data
from mlops.training.registry import register_and_select
from mlops.training.train import train_models
from mlops.utils import configure_logging, read_json, utc_now, write_json

logger = logging.getLogger(__name__)

STAGES = ("extract", "validate", "clean", "split", "train", "register")


def run_extract(settings: Settings) -> dict[str, Any]:
    dataframe = extract_data(settings.raw_data_path)
    report = {
        "status": "passed",
        "extracted_at": utc_now(),
        "source": "scikit-learn load_diabetes(as_frame=True)",
        "path": str(settings.raw_data_path),
        "rows": int(dataframe.shape[0]),
        "columns": list(dataframe.columns),
        "dtypes": {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
    }
    write_json(settings.extraction_report_path, report)
    return report


def run_validate(settings: Settings) -> dict[str, Any]:
    try:
        dataframe = pd.read_csv(settings.raw_data_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Raw data not found at {settings.raw_data_path}. Run the 'extract' stage first."
        ) from exc

    try:
        report = validate_data(dataframe)
    except DataValidationError as exc:
        write_json(
            settings.validation_report_path,
            {
                "status": "failed",
                "checked_at": utc_now(),
                "path": str(settings.raw_data_path),
                "errors": exc.errors,
            },
        )
        raise

    report["path"] = str(settings.raw_data_path)
    write_json(settings.validation_report_path, report)
    return report


def run_clean(settings: Settings) -> dict[str, Any]:
    if not settings.raw_data_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {settings.raw_data_path}. Run the 'extract' stage first."
        )
    dataframe = pd.read_csv(settings.raw_data_path)
    cleaned, report = clean_data(dataframe)
    cleaned.to_csv(settings.clean_data_path, index=False)
    report["path"] = str(settings.clean_data_path)
    write_json(settings.cleaning_report_path, report)
    return report


def run_split(settings: Settings) -> dict[str, Any]:
    if not settings.clean_data_path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {settings.clean_data_path}. Run the 'clean' stage first."
        )
    dataframe = pd.read_csv(settings.clean_data_path)
    train, test = split_data(dataframe, settings.test_size, settings.random_state)
    train.to_csv(settings.train_data_path, index=False)
    test.to_csv(settings.test_data_path, index=False)
    report = build_split_report(train, test, settings.test_size, settings.random_state)
    report["train_path"] = str(settings.train_data_path)
    report["test_path"] = str(settings.test_data_path)
    write_json(settings.split_report_path, report)
    return report


def run_train(settings: Settings) -> dict[str, Any]:
    return train_models(settings)


def run_register(settings: Settings) -> dict[str, Any]:
    if not settings.training_summary_path.exists():
        raise FileNotFoundError(
            f"Training summary not found at {settings.training_summary_path}. "
            "Run the 'train' stage first."
        )
    summary = read_json(settings.training_summary_path)
    result = register_and_select(summary, settings)
    final_champion = result.get("final_champion") or {}
    if not final_champion.get("version"):
        raise RuntimeError(
            "No champion model is available. The candidate did not pass the quality gate "
            f"(RMSE threshold={settings.rmse_threshold})."
        )
    return result


def run_pipeline(stages: Iterable[str] | None = None, settings: Settings | None = None) -> dict[str, Any]:
    """Execute selected stages in canonical order."""
    settings = settings or get_settings()
    settings.ensure_directories()

    selected = list(stages) if stages else list(STAGES)
    unknown = sorted(set(selected) - set(STAGES))
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}. Valid stages: {list(STAGES)}")
    selected = [stage for stage in STAGES if stage in selected]
    logger.info("Running stages: %s", selected)

    results: dict[str, Any] = {}
    stage_functions = {
        "extract": run_extract,
        "validate": run_validate,
        "clean": run_clean,
        "split": run_split,
        "train": run_train,
        "register": run_register,
    }

    for stage in selected:
        logger.info("=== Stage: %s ===", stage)
        results[stage] = stage_functions[stage](settings)

    logger.info("Pipeline finished successfully. Stages executed: %s", selected)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the diabetes MLOps pipeline.")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=[*STAGES, "all"],
        help="Subset of stages to run (default: all stages in canonical order).",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="Print the canonical stage list and exit.",
    )
    args = parser.parse_args(argv)

    configure_logging()
    if args.list_stages:
        print("\n".join(STAGES))
        return 0

    stages = args.stages
    if stages and "all" in stages:
        stages = None

    try:
        run_pipeline(stages)
    except Exception:
        logger.exception("Pipeline failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
