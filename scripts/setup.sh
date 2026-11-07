#!/usr/bin/env bash
# Configura el entorno de ejecución del proyecto MLOps.
# Uso: ./scripts/setup.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
    echo "Creando entorno virtual en .venv ..."
    python3 -m venv .venv
fi

if [ -n "${FISH_VERSION:-}" ]; then
    source .venv/bin/activate.fish
elif [ -n "${BASH_VERSION:-}" ]; then
    source .venv/bin/activate
fi

echo "Actualizando pip ..."
python -m pip install --upgrade pip

echo "Instalando dependencias de ejecución y pruebas ..."
python -m pip install -r requirements.txt

echo "Instalando dependencias para el notebook (EDA) ..."
python -m pip install -r requirements-notebook.txt

echo "Entorno listo. Activa el entorno con: source .venv/bin/activate"
