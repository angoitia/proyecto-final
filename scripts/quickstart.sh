#!/usr/bin/env bash
# Punto de entrada rapido: ejecuta el pipeline MLOps completo y las pruebas.
# Uso: ./scripts/quickstart.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate

echo "Ejecutando pipeline MLOps (extract -> validate -> clean -> split -> train -> register) ..."
python -m mlops.pipeline

echo "Ejecutando pruebas ..."
python -m pytest -q

echo "Quickstart completado: pipeline y pruebas finalizadas."
echo "Para regenerar el estado local: rm -f mlflow.db && rm -rf mlruns && rm -f data/raw/*.csv data/processed/*.csv artifacts/reports/*.json"
echo "Para servir el modelo campeón: uvicorn mlops.api.main:app --host 0.0.0.0 --port 8000 --reload"
