"""Paquete MLOps end-to-end para el proyecto de progresión de diabetes."""

import os

# Mantener fuera de los logs de CLI/CI la pista opcional de agente de MLflow.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

__version__ = "1.0.0"
