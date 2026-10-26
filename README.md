# Diabetes Progression Prediction — End-to-End MLOps

A small but complete MLOps project for a regression problem. It starts from an
exploratory notebook, moves the logic into reusable data/training modules, tracks
experiments and registers models with **MLflow**, promotes a **champion** behind
a quality gate, serves it with **FastAPI**, and automates the whole flow with
**GitHub Actions**.

The model predicts diabetes disease progression one year after baseline from ten
numerical patient features.

---

## 1. Project goal

The repository demonstrates the core MLOps lifecycle without pretending to be a
full production platform:

1. Extract raw data and persist it.
2. Validate the schema and target.
3. Clean the data and create a deterministic train/test split.
4. Train several regression candidates.
5. Log parameters, metrics and artifacts to MLflow.
6. Register the best candidate in the MLflow Model Registry.
7. Promote it to the `champion` alias only when it passes a quality gate.
8. Serve the champion through `/health`, `/model-info` and `/predict`.
9. Handle valid requests, invalid payloads, unavailable models and inference errors.
10. Run the pipeline and tests automatically on every push.

The exploratory notebook is
[`notebooks/01_exploratory_analysis.ipynb`](notebooks/01_exploratory_analysis.ipynb).
The original assignment notebook is kept untouched at the repository root.

---

## 2. Dataset

The project uses **scikit-learn's Diabetes dataset**:

| Property | Value |
|---|---|
| Rows | 442 |
| Features | 10 numeric features: `age`, `sex`, `bmi`, `bp`, `s1`–`s6` |
| Target | `target` — quantitative disease progression |
| Target range | 25–346 |
| Source | `sklearn.datasets.load_diabetes` |

The features are already **mean-centered and scaled** by scikit-learn. The API
therefore expects values in the same standardized scale used by the model, not
raw ages or BMI values.

The extraction stage writes the dataset to `data/raw/diabetes.csv`. This file is
generated at runtime and is intentionally ignored by git; the same is true for
the processed files.

---

## 3. Repository structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml                  # push/PR pipeline + tests
├── artifacts/
│   └── reports/                    # generated JSON audit reports (gitignored)
├── data/
│   ├── raw/                        # raw diabetes.csv (generated)
│   └── processed/                  # cleaned data + train/test splits (generated)
├── mlops/
│   ├── api/
│   │   ├── main.py                 # FastAPI app and endpoints
│   │   ├── model_loader.py         # loads models:/...@champion from MLflow
│   │   └── schemas.py              # Pydantic request/response schemas
│   ├── data/
│   │   ├── extract.py              # data extraction
│   │   ├── validate.py             # schema/quality validation
│   │   ├── clean.py                # missing values, duplicates, invalid labels
│   │   └── split.py                # deterministic train/test split
│   ├── training/
│   │   ├── models.py               # candidate model definitions
│   │   ├── train.py                # training + MLflow tracking
│   │   ├── registry.py             # registration + champion selection
│   │   └── mlflow_utils.py         # tracking/registry configuration
│   ├── config.py                   # paths, model names, thresholds, env vars
│   ├── pipeline.py                 # orchestration CLI
│   └── utils.py                    # logging, JSON and git helpers
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_champion_selection.py
│   ├── test_api.py
│   └── test_integration_champion.py
├── requirements.txt
├── requirements-notebook.txt
├── pytest.ini
└── README.md
```

Responsibilities are separated by layer:

- **Data logic** → `mlops/data/`
- **Training logic** → `mlops/training/train.py` and `models.py`
- **Registry/evaluation logic** → `mlops/training/registry.py`
- **API serving logic** → `mlops/api/`
- **Pipeline orchestration** → `mlops/pipeline.py`
- **CI/CD configuration** → `.github/workflows/ci.yml`

---

## 4. Pipeline overview

```text
extract → validate → clean → split → train → register
```

Run every stage:

```bash
python -m mlops.pipeline
```

Run only a subset (stages always execute in the canonical order):

```bash
python -m mlops.pipeline --stages extract validate clean split
python -m mlops.pipeline --stages train register
python -m mlops.pipeline --list-stages
```

### 4.1 Data extraction

`mlops/data/extract.py` loads `load_diabetes(as_frame=True)` and writes
`data/raw/diabetes.csv` together with `artifacts/reports/extraction_report.json`.
In a real project this module would be replaced by a database/object-storage/API
reader, while the rest of the pipeline would consume the same raw contract.

### 4.2 Data validation

`mlops/data/validate.py` checks:

- required feature columns and the target column exist;
- there are no duplicate column names;
- the dataset is not empty;
- feature and target dtypes are numeric;
- the target has no missing or infinite values;
- no feature is entirely missing.

Missing feature values and non-positive targets are reported as warnings because
the cleaning stage owns their treatment. Fatal problems raise
`DataValidationError`, write a failed report and stop the pipeline.

### 4.3 Data cleaning

`mlops/data/clean.py` applies a deterministic policy:

1. keep only contract columns;
2. coerce numeric types;
3. treat `inf`/`-inf` as missing;
4. drop rows with unusable targets (missing, non-finite or `<= 0`);
5. remove exact duplicate rows;
6. impute missing feature values with the column median;
7. cast features and target to `float`.

The scikit-learn dataset is already clean, so the first run reports zero changes.
The logic is exercised by unit tests with injected missing values, duplicates and
invalid targets.

### 4.4 Train/test split

`mlops/data/split.py` uses an 80/20 split with `random_state=42`:

- 353 training rows;
- 89 test rows.

The split is saved to `data/processed/train.csv` and `data/processed/test.csv`.
Reports are stored under `artifacts/reports/`.

---

## 5. Training and MLflow

### 5.1 Candidates

The production candidates mirror the notebook baseline and are wrapped in
scikit-learn `Pipeline` objects (scaler + estimator), so the exact preprocessing
used at training time is part of the logged model:

| Candidate | Definition |
|---|---|
| `linear_regression` | `StandardScaler` + `LinearRegression` |
| `random_forest` | `StandardScaler` + `RandomForestRegressor(n_estimators=100, random_state=42)` |
| `gradient_boosting` | `StandardScaler` + `GradientBoostingRegressor(random_state=42)` |

### 5.2 Metrics

For every candidate, the training stage logs:

- train and test **RMSE**, **MAE** and **R²**;
- candidate name, hyperparameters, row counts and feature count;
- the serialized model with an MLflow signature and input example;
- `metrics.json`;
- `feature_importance.csv` (tree candidates) or coefficients (linear candidate);
- the git commit SHA when available.

Current result on the deterministic test split:

| Model | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|
| **GradientBoosting (champion)** | **53.73** | **44.52** | **0.455** |
| LinearRegression | 53.85 | 42.79 | 0.453 |
| RandomForest | 54.40 | 44.17 | 0.441 |

The notebook's baseline comparison closely matches these values; the notebook
also explores interaction features, which are not used in the serving contract.

### 5.3 MLflow experiment

By default the project uses a local SQLite backend:

- tracking/registry database: `mlflow.db`;
- artifacts: `mlruns/`;
- experiment: `diabetes-progression`;
- runs: `train_<candidate>` and `champion_selection`.

Open the UI from the repository root:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then open <http://127.0.0.1:5000>. The **Models** tab shows the registered model,
its versions, tags, aliases and the `champion` alias.

A remote tracking server can be used via `MLFLOW_TRACKING_URI`; in that case the
SQLite-specific artifact location handling is skipped and the server's artifact
root is used.

---

## 6. Champion model and quality gate

The registered model name is `diabetes-progression-model` (configurable). The
alias is `champion`.

A candidate is promoted only when **both** conditions hold:

1. **Quality gate:** test RMSE ≤ `RMSE_THRESHOLD` (default **60.0**).
2. **Champion comparison:** there is no current champion, or the candidate's RMSE
   is better than or equal to the current champion's RMSE.

Why 60.0?

- The test target standard deviation is about 73.2.
- A constant-mean baseline has test RMSE ≈ 73.2.
- The quality gate (60.0) requires roughly an 18% improvement over that baseline.
- The current champion achieves 53.73, about a 27% improvement.

This is deliberately a simple, explainable gate—not an overfit threshold tuned to
the test set.

Each registered version is tagged with `test_rmse`, `test_mae`, `test_r2`,
`training_run_id`, `quality_gate_threshold`, `quality_gate`, `promotion_status`,
`candidate_model` and the git SHA. The full decision (including the previous
champion and the reason) is written to
`artifacts/reports/champion_selection.json` and logged as a separate MLflow run.

If the candidate fails the gate and there is no previous champion, the `register`
stage raises an error and the process exits non-zero, so CI fails. If a candidate
passes the gate but is worse than the current champion, the current champion is
kept and the pipeline succeeds.

---

## 7. API serving

The FastAPI application loads the model behind
`models:/diabetes-progression-model@champion` at startup. The API process can
start even when no champion exists; in that case `/health` still returns 200 and
the model endpoints return 503.

Start the API from the repository root:

```bash
uvicorn mlops.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive documentation: <http://127.0.0.1:8000/docs>

### 7.1 Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check; reports whether the model is loaded |
| `GET` | `/model-info` | Registered model name, version, alias, run, metrics and gate status |
| `POST` | `/predict` | Validates one patient record and returns a prediction |

### 7.2 Examples

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "diabetes-progression-model",
  "model_version": "1"
}
```

```bash
curl http://127.0.0.1:8000/model-info
```

```json
{
  "model_name": "diabetes-progression-model",
  "version": "1",
  "champion_alias": "champion",
  "run_id": "…",
  "source": "mlflow-registry",
  "metrics": {
    "test_rmse": 53.730732622787166,
    "test_mae": 44.51935924291035,
    "test_r2": 0.45509457538852915
  },
  "quality_gate": "passed",
  "loaded_at": "…"
}
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 0.0380759064,
    "sex": 0.0506801187,
    "bmi": 0.0616962065,
    "bp": 0.0218723549,
    "s1": -0.0442234984,
    "s2": -0.0348207628,
    "s3": -0.0434008456,
    "s4": -0.0025922620,
    "s5": 0.0199074862,
    "s6": -0.0176461252
  }'
```

```json
{
  "prediction": 237.35568496718219,
  "model_name": "diabetes-progression-model",
  "model_version": "1",
  "champion_alias": "champion"
}
```

The prediction depends on the current champion; the version number increases
each time the pipeline registers a new model version.

---

## 8. Error handling

| Scenario | HTTP status | Behavior |
|---|---|---|
| Valid request | `200` | Prediction returned with model name, version and alias |
| Missing feature, extra field, wrong type, invalid JSON | `422` | Pydantic validation error; invalid data never reaches the model |
| Champion alias/model not loaded | `503` | `/model-info` and `/predict` return a clear service-unavailable error; `/health` still returns `200` |
| Model loaded but inference fails | `500` | Error is logged and a generic `Prediction failed.` response is returned |

The distinction matters: `503` means the service cannot serve because the model is
unavailable, while `500` means the service is up but the model call failed.

Example invalid request (missing `bmi`):

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":0.03,"sex":0.05,"bp":0.02,"s1":-0.04,"s2":-0.03,"s3":-0.04,"s4":-0.002,"s5":0.01,"s6":-0.01}'
# 422
```

---

## 9. Installation

Use Python **3.12 or 3.13** (MLflow's current 3.x line is tested there).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

With `uv`:

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
```

Notebook dependencies are separate so the production/CI install stays small:

```bash
pip install -r requirements-notebook.txt
jupyter lab notebooks/01_exploratory_analysis.ipynb
```

### Quick start

```bash
# 1. Full pipeline: data + training + registry + champion
python -m mlops.pipeline

# 2. Unit tests
pytest -q

# 3. Real end-to-end test against the local MLflow registry
RUN_INTEGRATION=1 pytest -q

# 4. Serve the champion
uvicorn mlops.api.main:app --host 0.0.0.0 --port 8000
```

Then open <http://127.0.0.1:8000/docs> or call the endpoints above.

To reset all generated local state:

```bash
rm -rf mlflow.db mlruns data/raw/*.csv data/processed/*.csv artifacts/reports/*.json
python -m mlops.pipeline
```

---

## 10. Tests

| File | Type | What it covers |
|---|---|---|
| `tests/test_data_pipeline.py` | Unit | extraction, validation failures, cleaning, split determinism |
| `tests/test_champion_selection.py` | Unit | quality gate and champion comparison policy |
| `tests/test_api.py` | Unit | endpoints, 200/422/503/500 paths with a fake model |
| `tests/test_integration_champion.py` | Integration | loads the real MLflow champion and calls all endpoints |

Default run:

```bash
pytest -q
# 25 passed, 1 skipped (integration excluded)
```

End-to-end run after the pipeline:

```bash
RUN_INTEGRATION=1 pytest -q
# 26 passed
```

The integration test reads a real row from `data/processed/test.csv`, asks the
actual champion for a prediction and checks that the model version, alias and
metrics are consistent. CI sets `RUN_INTEGRATION=1` after training/registration.

---

## 11. CI/CD with GitHub Actions

Workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

Triggers:

- push to `main` or `master`;
- pull request targeting `main` or `master`;
- manual `workflow_dispatch`.

Job steps:

1. checkout;
2. set up Python 3.12 with pip cache;
3. install `requirements.txt`;
4. run `python -m mlops.pipeline` (extract → validate → clean → split → train → register);
5. run `python -m pytest -q` with `RUN_INTEGRATION=1`, so the real registry and API are exercised;
6. upload `artifacts/reports/`, `mlflow.db` and `mlruns/` as a workflow artifact.

The CI run uses an isolated SQLite database and starts with no champion, so it also
verifies that a fresh clone can train, register and serve a model. No secrets are
required for the default local setup. Switching `MLFLOW_TRACKING_URI` to a remote
MLflow server would require the corresponding credentials in GitHub Actions.

The quality gate is part of the automation: if no candidate passes and no previous
champion exists, the register stage exits non-zero and the workflow fails.

---

## 12. Configuration

All defaults live in `mlops/config.py`. Environment variables override them:

| Variable | Default | Purpose |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `sqlite:///<repo>/mlflow.db` | MLflow tracking and registry backend |
| `MLFLOW_ARTIFACT_DIR` | `<repo>/mlruns` | Artifact root for local SQLite/file backends |
| `MLFLOW_EXPERIMENT` | `diabetes-progression` | Experiment name |
| `MODEL_NAME` | `diabetes-progression-model` | Registered model name |
| `CHAMPION_ALIAS` | `champion` | Alias assigned to the selected model |
| `RMSE_THRESHOLD` | `60.0` | Quality gate for champion promotion |
| `TEST_SIZE` | `0.2` | Test split fraction |
| `RANDOM_STATE` | `42` | Split and model randomness |
| `DATA_RAW_DIR` | `<repo>/data/raw` | Raw data directory |
| `DATA_PROCESSED_DIR` | `<repo>/data/processed` | Cleaned/split data directory |
| `REPORTS_DIR` | `<repo>/artifacts/reports` | JSON audit reports |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `RUN_INTEGRATION` | unset | Test-only flag for the real MLflow integration test |

Example: require a stricter gate in CI:

```bash
RMSE_THRESHOLD=55.0 python -m mlops.pipeline
```

---

## 13. Design decisions and limitations

### Why aliases instead of stages?

MLflow stages (`Staging`, `Production`) are deprecated in MLflow 3.x. Aliases are
the current mechanism: `models:/diabetes-progression-model@champion` always points
to the selected version, and promoting a new version is a single registry update.

### Why a pipeline object?

The scaler and estimator are trained and serialized together. The API receives
the same object that was evaluated, avoiding training/serving skew.

### Why keep all candidates?

Every candidate gets its own run and model artifact. That preserves the experiment
history, even though only the best candidate is registered and potentially
promoted. The training summary is stored in
`artifacts/reports/training_summary.json`.

### Security note on serialization

MLflow 3 serializes scikit-learn models with `skops` by default. Tree-based
candidates require trusting `sklearn.tree._tree.Tree`, which is generated by our
own training code. The trusted type is declared explicitly in
`mlops/training/train.py`; no arbitrary pickle from an external source is loaded.

### Current limitations (intentional scope)

- Median imputation is computed on the full cleaned dataset before the split. The
  diabetes dataset has no missing values, so there is no practical leakage here.
  A stricter production design would fit the imputer on the training split only.
- SQLite is single-writer and intended for local/CI use. Use a remote MLflow
  backend for concurrent teams.
- There is no authentication, rate limiting, drift monitoring or automatic
  retraining schedule; these are outside the assignment's scope.
- The API serves one model and one prediction per request; batch prediction is not
  implemented.

---

## 14. Deliverables checklist

- [x] Exploratory notebook (executed copy under `notebooks/`).
- [x] Data extraction, validation and cleaning scripts.
- [x] Deterministic train/test split.
- [x] Training script with several candidates.
- [x] MLflow tracking (params, metrics, model, artifacts).
- [x] MLflow Model Registry and champion alias.
- [x] Quality gate and champion comparison logic.
- [x] FastAPI endpoints `/health`, `/model-info`, `/predict`.
- [x] Basic API error handling (422, 503, 500).
- [x] GitHub Actions workflow on push/PR.
- [x] README with setup, execution and API examples.
- [x] Requirements file and clear project structure.
- [x] Unit tests and a real-registry integration test.
