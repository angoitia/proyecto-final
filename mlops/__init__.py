"""End-to-end MLOps package for the diabetes progression project."""

import os

# Keep MLflow's optional agent hint out of CLI/CI logs.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

__version__ = "1.0.0"
