# Predicción de la progresión de la diabetes — MLOps de extremo a extremo

[Repositorio del proyecto](https://github.com/angoitia/proyecto-final)

Un proyecto MLOps pequeño pero completo para un problema de regresión. Parte de un
notebook exploratorio, traslada la lógica a módulos reutilizables de datos y
entrenamiento, hace seguimiento de experimentos y registra modelos con **MLflow**,
promueve un **modelo campeón** detrás de un criterio de calidad, lo sirve con
**FastAPI** y automatiza todo el flujo con **GitHub Actions**.

El modelo predice la progresión de la enfermedad de diabetes un año después de la
línea base a partir de diez características numéricas de pacientes.

---

## 1. Objetivo del proyecto

El repositorio demuestra el ciclo de vida MLOps fundamental sin pretender ser una
plataforma de producción completa:

1. Extraer los datos sin procesar y persistirlos.
2. Validar el esquema y la variable objetivo.
3. Limpiar los datos y crear una división determinista de train/test.
4. Entrenar varios candidatos de regresión.
5. Registrar parámetros, métricas y artefactos en MLflow.
6. Registrar el mejor candidato en el registro de modelos de MLflow.
7. Promoverlo al alias `champion` solo cuando supere un criterio de calidad.
8. Servir el modelo campeón a través de `/health`, `/model-info` y `/predict`.
9. Gestionar solicitudes válidas, cargas útiles no válidas, modelos no disponibles y errores de inferencia.
10. Ejecutar el pipeline y las pruebas automáticamente en cada push.

El notebook exploratorio es
[`notebooks/01_exploratory_analysis.ipynb`](notebooks/01_exploratory_analysis.ipynb).
El notebook original de la tarea se conserva intacto en la raíz del repositorio.

---

## 2. Conjunto de datos

El proyecto utiliza el **conjunto de datos Diabetes de scikit-learn**:

| Propiedad | Valor |
|---|---|
| Filas | 442 |
| Características | 10 características numéricas: `age`, `sex`, `bmi`, `bp`, `s1`–`s6` |
| Variable objetivo | `target` — progresión cuantitativa de la enfermedad |
| Rango de la variable objetivo | 25–346 |
| Origen | `sklearn.datasets.load_diabetes` |

scikit-learn ya **centra en la media y escala** las características. Por lo tanto, la
API espera valores en la misma escala estandarizada que utiliza el modelo, no edades
ni valores de IMC sin procesar.

La etapa de extracción escribe el conjunto de datos en `data/raw/diabetes.csv`. Este
archivo se genera en tiempo de ejecución y git lo ignora intencionadamente; lo mismo
ocurre con los archivos procesados.

---

## 3. Estructura del repositorio

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml                  # pipeline de push/PR + pruebas
├── artifacts/
│   └── reports/                    # informes de auditoría JSON generados (ignorados por git)
├── data/
│   ├── raw/                        # diabetes.csv sin procesar (generado)
│   └── processed/                  # datos limpios + divisiones train/test (generadas)
├── mlops/
│   ├── api/
│   │   ├── main.py                 # aplicación FastAPI y endpoints
│   │   ├── model_loader.py         # carga models:/...@champion desde MLflow
│   │   └── schemas.py              # esquemas de solicitud/respuesta de Pydantic
│   ├── data/
│   │   ├── extract.py              # extracción de datos
│   │   ├── validate.py             # validación de esquema/calidad
│   │   ├── clean.py                # valores faltantes, duplicados, etiquetas no válidas
│   │   └── split.py                # división determinista train/test
│   ├── training/
│   │   ├── models.py               # definiciones de modelos candidatos
│   │   ├── train.py                # entrenamiento + seguimiento en MLflow
│   │   ├── registry.py             # registro + selección del modelo campeón
│   │   └── mlflow_utils.py         # configuración de seguimiento/registro
│   ├── config.py                   # rutas, nombres de modelos, umbrales, variables de entorno
│   ├── pipeline.py                 # CLI de orquestación
│   └── utils.py                    # utilidades de logging, JSON y git
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
├── scripts/
│   ├── setup.sh                    # creación de .venv e instalación de dependencias
│   └── quickstart.sh               # pipeline MLOps + pruebas
└── README.md
```

Las responsabilidades se separan por capas:

- **Lógica de datos** → `mlops/data/`
- **Lógica de entrenamiento** → `mlops/training/train.py` y `models.py`
- **Lógica de registro/evaluación** → `mlops/training/registry.py`
- **Lógica de servicio de la API** → `mlops/api/`
- **Orquestación del pipeline** → `mlops/pipeline.py`
- **Configuración de CI/CD** → `.github/workflows/ci.yml`

---

## 4. Descripción general del pipeline

```text
extract → validate → clean → split → train → register
```

Ejecuta todas las etapas:

```bash
python -m mlops.pipeline
```

Ejecuta solo un subconjunto (las etapas siempre se ejecutan en el orden canónico):

```bash
python -m mlops.pipeline --stages extract validate clean split
python -m mlops.pipeline --stages train register
python -m mlops.pipeline --list-stages
```

### 4.1 Extracción de datos

`mlops/data/extract.py` carga `load_diabetes(as_frame=True)` y escribe
`data/raw/diabetes.csv` junto con `artifacts/reports/extraction_report.json`.
En un proyecto real, este módulo se sustituiría por un lector de base de
datos/almacenamiento de objetos/API, mientras que el resto del pipeline consumiría
el mismo contrato de datos sin procesar.

### 4.2 Validación de datos

`mlops/data/validate.py` comprueba:

- que existan las columnas de características requeridas y la columna de la variable objetivo;
- que no haya nombres de columnas duplicados;
- que el conjunto de datos no esté vacío;
- que los dtypes de las características y de la variable objetivo sean numéricos;
- que la variable objetivo no tenga valores faltantes ni infinitos;
- que ninguna característica esté completamente vacía.

Los valores de características faltantes y las variables objetivo no positivas se
notifican como advertencias porque la etapa de limpieza es responsable de tratarlos.
Los problemas fatales lanzan `DataValidationError`, escriben un informe de fallo y
detienen el pipeline.

### 4.3 Limpieza de datos

`mlops/data/clean.py` aplica una política determinista:

1. conservar solo las columnas del contrato;
2. convertir los tipos a numéricos;
3. tratar `inf`/`-inf` como valores faltantes;
4. eliminar las filas con variables objetivo inutilizables (faltantes, no finitas o `<= 0`);
5. eliminar las filas duplicadas exactas;
6. imputar los valores de características faltantes con la mediana de la columna;
7. convertir las características y la variable objetivo a `float`.

El conjunto de datos de scikit-learn ya está limpio, por lo que la primera ejecución
informa de cero cambios. La lógica se ejercita mediante pruebas unitarias con
valores faltantes, duplicados y variables objetivo no válidas inyectados.

### 4.4 División entrenamiento / validación / test

`mlops/data/split.py` divide el dataset limpio en:

- **entrenamiento**: 353 filas → se subdivide en ajuste de entrenamiento (264 filas) y validación (89 filas);
- **validación**: 89 filas — reservada para la selección del mejor candidato y para el quality gate;
- **test**: 89 filas — reservado para la evaluación final e independiente del campeón.

La división se realiza con una partición determinista (`random_state=42`, 80/20) y, a
partir del conjunto de entrenamiento, un `val_size=0.25` para la validación.

La división se guarda en `data/processed/train.csv`, `data/processed/validation.csv`
y `data/processed/test.csv`. Los informes se almacenan en `artifacts/reports/`.

---

## 5. Entrenamiento y MLflow

### 5.1 Candidatos

Los candidatos de producción reproducen la línea base del notebook y se envuelven en
objetos `Pipeline` de scikit-learn (imputador de mediana + escalador + estimador), de
modo que el preprocesamiento exacto utilizado durante el entrenamiento forma parte del
modelo que se registra: el imputador se ajusta sobre el train y se aplica a
validación/prueba automáticamente.

| Candidato | Definición |
|---|---|
| `linear_regression` | `StandardScaler` + `LinearRegression` |
| `random_forest` | `StandardScaler` + `RandomForestRegressor(n_estimators=100, random_state=42)` |
| `gradient_boosting` | `StandardScaler` + `GradientBoostingRegressor(random_state=42)` |

### 5.2 Métricas

Para cada candidato, la etapa de entrenamiento registra:

- **RMSE**, **MAE** y **R²** de entrenamiento, validación y test;
- nombre del candidato, hiperparámetros, recuentos de filas y número de características;
- el modelo serializado con una firma de MLflow y un ejemplo de entrada;
- `metrics.json`;
- `feature_importance.csv` (candidatos de árboles) o coeficientes (candidato lineal);
- el SHA del commit de git cuando esté disponible.

La selección del mejor candidato se realiza con el **RMSE estimado por validación
cruzada** (5 folds sobre el conjunto de entrenamiento); la evaluación final del
campeón se hace sobre el **test** determinista.

Resultado actual:

| Modelo | RMSE CV | RMSE test | MAE test | R² test |
|---|---:|---:|---:|---:|
| **LinearRegression (campeón)** | **57.02** | **54.16** | **42.92** | **0.446** |
| RandomForest | 62.09 | 55.14 | 44.57 | 0.426 |
| GradientBoosting | 62.03 | 54.78 | 44.49 | 0.434 |

La comparación con la línea base del notebook es coherente; el notebook también
explora características de interacción, que no se utilizan en el contrato de
servicio.

### 5.3 Experimento de MLflow

De forma predeterminada, el proyecto utiliza un backend SQLite local:

- base de datos de seguimiento/registro: `mlflow.db`;
- artefactos: `mlruns/`;
- experimento: `diabetes-progression`;
- ejecuciones: `train_<candidate>` y `champion_selection`.

Abre la interfaz de usuario desde la raíz del repositorio:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Luego abre <http://127.0.0.1:5000>. La pestaña **Models** muestra el modelo
registrado, sus versiones, etiquetas, alias y el alias `champion`.

Se puede utilizar un servidor de seguimiento remoto mediante `MLFLOW_TRACKING_URI`;
en ese caso se omite el tratamiento específico de SQLite para la ubicación de los
artefactos y se utiliza la raíz de artefactos del servidor.

---

## 6. Modelo campeón y criterio de calidad

El nombre del modelo registrado es `diabetes-progression-model` (configurable). El
alias es `champion`.

Un candidato se promueve solo cuando se cumplen **ambas** condiciones:

1. **Criterio de calidad:** RMSE estimado por validación cruzada ≤ `RMSE_THRESHOLD`
   (por defecto, **60.0**).
2. **Comparación con el modelo campeón:** no hay un modelo campeón actual, o el RMSE
   (CV) del candidato es mejor que el del modelo campeón actual. Ante empate, se
   conserva al campeón actual.

La decisión de promoción se basa en la métrica de validación cruzada (RMSE CV); la
evaluación final del campeón (RMSE de test, MAE y R²) se registra para reporte. Si el
candidato no supera el criterio y no hay un modelo campeón anterior, la etapa `register`
lanza un error y el proceso termina con un código distinto de cero, por lo que CI falla.
Si un candidato supera el criterio pero es peor que el modelo campeón actual, se
mantiene el modelo campeón actual y el pipeline finaliza correctamente.

¿Por qué 60.0?

- La desviación estándar de la variable objetivo en test es de aproximadamente 73.2.
- Una línea base de media constante tiene un RMSE de test ≈ 73.2.
- El criterio de calidad (60.0) exige aproximadamente una mejora del 18 % sobre esa línea base;
  el campeón supera ampliamente ese umbral con un RMSE estimado en validación cruzada de ≈ 57.
- El modelo campeón (LinearRegression) alcanza **57.02** en validación cruzada y
  **54.16** en test, con RMSE inferior a la línea base de media constante (≈ 73.2).

Se trata deliberadamente de un criterio simple y explicable, no de un umbral
sobreajustado al conjunto de validación.

Cada versión registrada se etiqueta con `validation_rmse`, `test_rmse`, `test_mae`,
`test_r2`, `training_run_id`, `quality_gate_threshold`, `quality_gate`,
`promotion_status`, `candidate_model` y el SHA de git. La decisión completa (incluido el
modelo campeón anterior y el motivo) se escribe en `artifacts/reports/champion_selection.json`
y se registra como una ejecución independiente de MLflow.

Si el candidato no supera el criterio y no hay un modelo campeón anterior, la etapa
`register` lanza un error y el proceso termina con un código distinto de cero, por
lo que CI falla. Si un candidato supera el criterio pero es peor que el modelo
campeón actual, se mantiene el modelo campeón actual y el pipeline finaliza
correctamente.

---

## 7. Servicio de la API

La aplicación FastAPI carga el modelo detrás de
`models:/diabetes-progression-model@champion` al iniciarse. El proceso de la API
puede arrancar incluso cuando no existe un modelo campeón; en ese caso, `/health`
sigue devolviendo 200 y los endpoints del modelo devuelven 503.

Arranca la API desde la raíz del repositorio:

```bash
uvicorn mlops.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Documentación interactiva: <http://127.0.0.1:8000/docs>

### 7.1 Endpoints

| Método | Ruta | Propósito |
|---|---|---|
| `GET` | `/health` | Comprobación de actividad; informa si el modelo está cargado |
| `GET` | `/model-info` | Nombre del modelo registrado, versión, alias, ejecución, métricas y estado del criterio de calidad |
| `GET` | `/v1/models` | Endpoint estándar de descubrimiento de modelos (estilo `/v1/models`) |
| `GET` | `/` | Info básica del servicio |
| `POST` | `/predict` | Valida un registro de paciente y devuelve una predicción |

### 7.2 Ejemplos

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

La predicción depende del modelo campeón actual; el número de versión aumenta cada
vez que el pipeline registra una nueva versión del modelo.

---

## 8. Gestión de errores

| Escenario | Estado HTTP | Comportamiento |
|---|---|---|
| Solicitud válida | `200` | Se devuelve la predicción con el nombre del modelo, la versión y el alias |
| Falta una característica, campo adicional, tipo incorrecto, JSON no válido | `422` | Error de validación de Pydantic; los datos no válidos nunca llegan al modelo |
| Alias/modelo campeón no cargado | `503` | `/model-info` y `/predict` devuelven un error claro de servicio no disponible; `/health` sigue devolviendo `200` |
| Modelo cargado pero la inferencia falla | `500` | Se registra el error y se devuelve una respuesta genérica `Prediction failed.` |

La distinción importa: `503` significa que el servicio no puede atender porque el
modelo no está disponible, mientras que `500` significa que el servicio está activo
pero la llamada al modelo falló.

Ejemplo de solicitud no válida (falta `bmi`):

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":0.03,"sex":0.05,"bp":0.02,"s1":-0.04,"s2":-0.03,"s3":-0.04,"s4":-0.002,"s5":0.01,"s6":-0.01}'
# 422
```

---

## 9. Instalación

Usa Python **3.12 o 3.13** (la línea 3.x actual de MLflow se prueba en esas
versiones).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Con `uv`:

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
```

Las dependencias del notebook son independientes para que la instalación de
producción/CI se mantenga pequeña:

```bash
pip install -r requirements-notebook.txt
jupyter lab notebooks/01_exploratory_analysis.ipynb
```

### Inicio rápido

```bash
# 1. Configurar el entorno (crea .venv e instala dependencias)
./scripts/setup.sh

# 2. Ejecución rápida: pipeline completo + pruebas
./scripts/quickstart.sh

# 3. Variante manual:
#    python -m mlops.pipeline
#    pytest -q
#    RUN_INTEGRATION=1 pytest -q

# 4. Servir el modelo campeón
uvicorn mlops.api.main:app --host 0.0.0.0 --port 8000
```

Luego abre <http://127.0.0.1:8000/docs> o llama a los endpoints anteriores.

Para restablecer todo el estado local generado:

```bash
rm -rf mlflow.db mlruns data/raw/*.csv data/processed/*.csv artifacts/reports/*.json
python -m mlops.pipeline
```

---

## 10. Pruebas

| Archivo | Tipo | Qué cubre |
|---|---|---|
| `tests/test_data_pipeline.py` | Unitaria | extracción, fallos de validación, limpieza, determinismo de la división |
| `tests/test_champion_selection.py` | Unitaria | criterio de calidad y política de comparación con el modelo campeón |
| `tests/test_api.py` | Unitaria | endpoints, rutas 200/422/503/500 con un modelo falso |
| `tests/test_integration_champion.py` | Integración | carga el modelo campeón real de MLflow y llama a todos los endpoints |

Ejecución predeterminada:

```bash
pytest -q
# 29 passed, 1 skipped (integración excluida)
```

Ejecución de extremo a extremo después del pipeline:

```bash
RUN_INTEGRATION=1 pytest -q
# 30 passed
```

La prueba de integración lee una fila real de `data/processed/test.csv`, solicita
una predicción al modelo campeón real y comprueba que la versión del modelo, el alias,
las métricas (CV) y el criterio de calidad sean coherentes. CI establece `RUN_INTEGRATION=1`
después del entrenamiento/registro.

---

## 11. CI/CD con GitHub Actions

Flujo de trabajo: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

Disparadores:

- push a `main` o `master`;
- pull request dirigida a `main` o `master`;
- `workflow_dispatch` manual.

Pasos del job:

1. checkout;
2. configurar Python 3.12 con caché de pip;
3. instalar `requirements.txt`;
4. ejecutar `python -m mlops.pipeline` (extract → validate → clean → split → train → register);
5. ejecutar `python -m pytest -q` con `RUN_INTEGRATION=1`, de modo que se ejerciten el registro y la API reales;
6. subir `artifacts/reports/`, `mlflow.db` y `mlruns/` como artefacto del flujo de trabajo.

La ejecución de CI utiliza una base de datos SQLite aislada y comienza sin modelo
campeón, por lo que también verifica que un clon nuevo puede entrenar, registrar y
servir un modelo. No se requieren secretos para la configuración local
predeterminada. Cambiar `MLFLOW_TRACKING_URI` a un servidor MLflow remoto requeriría
las credenciales correspondientes en GitHub Actions.

El criterio de calidad forma parte de la automatización: si ningún candidato lo
supera y no existe un modelo campeón anterior, la etapa de registro termina con un
código distinto de cero y el flujo de trabajo falla.

---

## 12. Configuración

Todos los valores predeterminados están en `mlops/config.py`. Las variables de
entorno los sobrescriben:

| Variable | Valor por defecto | Propósito |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `sqlite:///<repo>/mlflow.db` | Backend de seguimiento y registro de MLflow |
| `MLFLOW_ARTIFACT_DIR` | `<repo>/mlruns` | Raíz de artefactos para backends locales SQLite/archivo |
| `MLFLOW_EXPERIMENT` | `diabetes-progression` | Nombre del experimento |
| `MODEL_NAME` | `diabetes-progression-model` | Nombre del modelo registrado |
| `CHAMPION_ALIAS` | `champion` | Alias asignado al modelo seleccionado |
| `RMSE_THRESHOLD` | `60.0` | Criterio de calidad para la promoción del modelo campeón (RMSE de validación) |
| `TEST_SIZE` | `0.2` | Fracción del dataset reservada para el test final |
| `VALIDATION_SIZE` | `0.25` | Fracción del conjunto de entrenamiento reservada para validación |
| `RANDOM_STATE` | `42` | Aleatoriedad de la división y de los modelos |
| `DATA_RAW_DIR` | `<repo>/data/raw` | Directorio de datos sin procesar |
| `DATA_PROCESSED_DIR` | `<repo>/data/processed` | Directorio de datos limpios/divisiones |
| `REPORTS_DIR` | `<repo>/artifacts/reports` | Informes de auditoría JSON |
| `LOG_LEVEL` | `INFO` | Nivel de logging de Python |
| `RUN_INTEGRATION` | sin establecer | Indicador exclusivo de pruebas para la prueba de integración real con MLflow |

Ejemplo: exigir un criterio más estricto en CI:

```bash
RMSE_THRESHOLD=55.0 python -m mlops.pipeline
```

---

## 13. Decisiones de diseño y limitaciones

### ¿Por qué alias en lugar de etapas?

Las etapas de MLflow (`Staging`, `Production`) están obsoletas en MLflow 3.x. Los
alias son el mecanismo actual: `models:/diabetes-progression-model@champion` siempre
apunta a la versión seleccionada, y promover una nueva versión es una única
actualización del registro.

### ¿Por qué un objeto pipeline?

El escalador y el estimador se entrenan y serializan juntos. La API recibe el mismo
objeto que se evaluó, lo que evita el desajuste entre entrenamiento y servicio.

### ¿Por qué conservar todos los candidatos?

Cada candidato tiene su propia ejecución y artefacto de modelo. Eso preserva el
historial de experimentos, aunque solo el mejor candidato se registra y
potencialmente se promueve. El resumen del entrenamiento se almacena en
`artifacts/reports/training_summary.json`.

### Nota de seguridad sobre la serialización

MLflow 3 serializa los modelos de scikit-learn con `skops` de forma predeterminada.
Los candidatos basados en árboles requieren confiar en `sklearn.tree._tree.Tree`, que
genera nuestro propio código de entrenamiento. El tipo de confianza se declara
explícitamente en `mlops/training/train.py`; no se carga ningún pickle arbitrario de
una fuente externa.

### Limitaciones actuales (alcance intencionado)

- La imputación por mediana se realiza dentro del pipeline de entrenamiento
  (`SimpleImputer` dentro del `Pipeline`), calculada **solo sobre los datos de
  entrenamiento** y aplicada automáticamente a validación y prueba. La limpieza
  se encarga de la lógica estructural (tipos, valores inválidos, duplicados).
- SQLite admite un único escritor y está pensado para uso local/CI. Usa un backend
  de MLflow remoto para equipos con concurrencia.
- No hay autenticación, limitación de tasa, monitorización de deriva ni programación
  de reentrenamiento automático; quedan fuera del alcance de la tarea.
- La API sirve un modelo y una predicción por solicitud; la predicción por lotes no
  está implementada.

---

## 14. Lista de entregables

- [x] Notebook exploratorio (copia ejecutada en `notebooks/`).
- [x] Scripts de extracción, validación y limpieza de datos.
- [x] División determinista train/test.
- [x] Script de entrenamiento con varios candidatos.
- [x] Seguimiento de experimentos con MLflow (parámetros, métricas, modelo, artefactos).
- [x] Registro de modelos de MLflow y alias del modelo campeón.
- [x] Criterio de calidad y lógica de comparación con el modelo campeón.
- [x] Endpoints de FastAPI `/health`, `/model-info`, `/predict`.
- [x] Gestión básica de errores de la API (422, 503, 500).
- [x] Flujo de trabajo de GitHub Actions en push/PR.
- [x] README con configuración, ejecución y ejemplos de API.
- [x] Archivo de requisitos y estructura de proyecto clara.
- [x] Pruebas unitarias y una prueba de integración con el registro real.
