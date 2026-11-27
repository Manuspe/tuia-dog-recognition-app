# TP2 - Sistema de Detección y Clasificación de Razas de Perros

Plantilla base para desarrollar un pipeline completo de Computer Vision:

**Embeddings → Búsqueda por similitud → Clasificación → Detección → Pipeline completo**

> 📄 El enunciado completo del trabajo práctico está en [ENUNCIADO.md](ENUNCIADO.md) (versión PDF: [Trabajo Práctico N°2.pdf](<Trabajo Práctico N°2.pdf>)).

La cátedra provee la infraestructura completa (Docker, aplicación Gradio, base de datos vectorial, API, scripts y herramientas de evaluación y visualización). El grupo hace un **fork** de este repositorio, trabaja en una **feature branch** y completa **únicamente** las funciones indicadas en cada etapa. Cualquier modificación fuera de esas funciones debe justificarse en el Pull Request.

## Objetivo del backend

API asincrónica en Python que permite:

- Búsqueda por similitud (`/search`): imagen consultada, Top-K imágenes similares y raza predicha.
- Clasificación supervisada (`/classify`): raza y score con el modelo entrenado seleccionado.
- Detección + clasificación (`/detect`): bounding boxes, raza y scores de confianza.
- Consulta del estado del procesamiento asincrónico (`/status/{job_id}`).
- Selección dinámica del modelo de embeddings (`/models` + campo `model` en `/search`).

La API responde `HTTP 202` con un `job_id` y luego permite consultar el resultado con estado:

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none"
}
```

## Estructura

```text
tuia-dog-recognition-app/
├── 1-pre-processing.ipynb       # Parte 1: dataset y preprocesamiento
├── 2-train.ipynb                # Etapa 2: entrenamiento, evaluación (incl. NDCG@10 de la Etapa 1) y comparación
├── ENUNCIADO.md                 # Enunciado completo del TP
├── Trabajo Práctico N°2.pdf     # Enunciado en PDF
├── src/
│   ├── app/
│   │   └── main.py              # App FastAPI + /health
│   ├── frontend/                # UI Gradio provista por la cátedra (una pestaña por etapa)
│   │   ├── app.py
│   │   └── gradio_ui.py
│   └── lib/
│       ├── api.py               # Endpoints /upload, /search, /classify, /detect, /status, /models, /files
│       ├── bootstrap.py         # Construcción de los servicios (compartida por la API y los scripts)
│       ├── config.py            # Configuración vía .env
│       ├── files.py
│       ├── schemas.py
│       ├── evaluation/
│       │   └── metrics.py       # NDCG@k, precision/recall/F1, specificity
│       ├── visualization/
│       │   └── draw.py          # Dibujo de bounding boxes
│       ├── services/
│       │   ├── similarity_service.py   # ⚠️ Etapa 1: funciones a implementar
│       │   ├── classifier_service.py   # ⚠️ Etapa 2: funciones a implementar
│       │   ├── detection_service.py    # ⚠️ Etapa 3: funciones a implementar
│       │   └── task_manager.py
│       └── storage/
│           ├── base.py
│           ├── embedding_store.py      # Base vectorial en JSON
│           └── pgvector_store.py       # Base vectorial en PostgreSQL + pgvector
├── scripts/
│   ├── download_dataset.py      # Descarga el dataset de Kaggle
│   ├── build_index.py           # Indexa el dataset en la base vectorial
│   └── train_classifier.py      # Entrena/evalúa el clasificador (alternativa local a 2-train.ipynb)
├── data/
│   ├── dataset/                 # 70 Dog Breeds Image Dataset, no se versiona
│   └── embeddings.json          # Base vectorial JSON (si USE_PGVECTOR=false)
├── models/                      # Checkpoints entrenados (.pth / .onnx), no se versionan
├── output/                      # Imágenes subidas y resultados, no se versiona
├── requirements.txt
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.docker.example
└── .env.local.example
```

# Preparando el ambiente

## Requisitos

- Python 3.12
- Docker (Docker Desktop incluye Docker Compose)
- Git
- Cuenta de Kaggle (para descargar el dataset)

## Dataset

Dataset principal: [70 Dog Breeds Image Dataset (Kaggle)](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set).

```bash
python scripts/download_dataset.py
```

Requiere [credenciales de Kaggle](https://www.kaggle.com/docs/api) configuradas. Si no las tienen, descarguen el dataset manualmente desde Kaggle y descompriman `train/`, `valid/` y `test/` dentro de `data/dataset`.

## Configurá tus modelos

Entrená tus modelos (Etapa 2) con `2-train.ipynb` (en Google Colab, con GPU) y guardá los checkpoints dentro de la carpeta `models`. El backend admite modelos con extensión **.pth** (PyTorch, se carga con `torch.load` el modelo completo) o **.onnx** (se carga con `onnxruntime`). Si usan otro framework, expórtenlo a **.onnx**.

Recordá actualizar la configuración del `.env` correspondiente (`RESNET18_MODEL_NAME`, `CNN_CUSTOM_MODEL_NAME`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `SIMILARITY_THRESHOLD`) para que apunte a tus modelos.

> El backend arranca aunque falten los checkpoints: `/health` informa cuáles encuentra en `models/`. Las pestañas de las Etapas 2 y 3 fallan hasta que existan.

Con docker compose, el entorno reinicia la aplicación y actualiza el código automáticamente al detectar cambios. Lo mismo ocurre en local si usan `--reload`.

El reinicio automático puede no funcionar en algunas versiones de Docker Desktop para *Windows*. En ese caso, vuelvan a correr los comandos del siguiente apartado para actualizar el código dentro de Docker.

## Opción 1 - Corriendo dentro de Docker

Es la forma en que la cátedra evalúa el trabajo.

### 1. Construí y corré la aplicación

Ajustá el archivo **.env.docker.example** a tus necesidades. Luego corré desde la terminal:

```bash
docker compose build
docker compose up -d
```

Servicios disponibles: frontend en `http://localhost:8080`, backend en `http://localhost:8000` y PostgreSQL en `localhost:5432`.

### 2. Indexá la galería

Con la Etapa 1 implementada y el dataset en `data/dataset`, indexá el split de entrenamiento desde el contenedor del backend:

```bash
docker compose exec backend python /scripts/build_index.py --split train
```

## Opción 2 - Configurando el ambiente local

### 1. Cloná tu fork

```bash
git clone <url-de-tu-fork>
cd tuia-dog-recognition-app
git checkout -b feature/tp2-razas-perros
```

### 2. Instalá uv

Para usuarios de Linux o Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[Documentación de uv](https://docs.astral.sh/uv/getting-started/installation/). También pueden usar `python3.12 -m venv .venv` y `pip`.

### 3. Configurá un ambiente virtual con Python 3.12

```bash
uv venv --python 3.12 .venv
```

### 4. Activá el ambiente virtual

```bash
# Linux / Mac
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 5. Instalá las dependencias

```bash
uv pip install -r requirements.txt
```

### 6. Configurá las variables de entorno

Ajustá el archivo `.env.local.example` a tus necesidades y copialo a `src/.env`. El backend local y los scripts de `scripts/` leen la configuración desde ese archivo.

```bash
cp .env.local.example src/.env
```

### 7. Descargá el dataset

```bash
python scripts/download_dataset.py
```

### 8. Iniciá la base de datos

```bash
docker compose up postgres -d
```

Solo es necesaria con `USE_PGVECTOR=true` (por defecto). Con `USE_PGVECTOR=false` se usa la base vectorial JSON (`data/embeddings.json`) y no hace falta Docker para trabajar en local.

### 9. Iniciá el backend

```bash
cd src
uvicorn app.main:app --reload --port 8000
```

### 10. Iniciá el frontend (en otra terminal, con el ambiente virtual activado)

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 11. Verificá que todo esté corriendo

- Frontend: `http://localhost:8080`
- Backend: `http://localhost:8000/docs` (Swagger) y `http://localhost:8000/health` (muestra el modelo seleccionado y los checkpoints encontrados)

## Scripts provistos

Se ejecutan desde la raíz del repositorio con el ambiente local activo (y PostgreSQL corriendo si `USE_PGVECTOR=true`). Leen la configuración de `src/.env` si existe:

```bash
# Descargar el dataset de Kaggle a data/dataset
python scripts/download_dataset.py

# Indexar la galería en la base vectorial (requiere la Etapa 1 implementada)
python scripts/build_index.py --split train
# ... o lo mismo desde el contenedor del backend (con docker compose up -d corriendo)
docker compose exec backend python /scripts/build_index.py --split train

# Entrenar y evaluar el clasificador localmente (alternativa a 2-train.ipynb)
python scripts/train_classifier.py --model resnet18_finetuned
```

Con `USE_PGVECTOR=false`, la indexación del split completo es lenta: el JSON se reescribe en cada inserción. Para pruebas rápidas, usen `--limit` (máximo de imágenes por raza).

## Configuración

No hardcodeen parámetros: configúrenlos mediante `.env`.

1. Para la ejecución local, copien `.env.local.example` a `src/.env`.
2. Ajusten el modelo de embeddings (`EMBEDDING_MODEL`, `EMBEDDING_DIM`), la búsqueda (`SIMILARITY_METRIC`, `SIMILARITY_THRESHOLD`, `TOP_K`), los checkpoints (`RESNET18_MODEL_NAME`, `CNN_CUSTOM_MODEL_NAME`), YOLO (`YOLO_MODEL`, `YOLO_CONF_THRESHOLD`, `YOLO_DOG_CLASS_ID`) y los paths.
3. Opcional (habilitado por defecto): configuren la conexión a PostgreSQL + pgvector (`USE_PGVECTOR`). Con `USE_PGVECTOR=false`, los embeddings se guardan en un archivo JSON (`EMBEDDINGS_PATH`).

## Endpoints

- Backend: `http://localhost:8000` (documentación Swagger en `http://localhost:8000/docs`)
- PostgreSQL/pgvector: `localhost:5432`
- Frontend (provisto por la cátedra): `http://localhost:8080`

| Endpoint | Descripción |
|----------|-------------|
| `POST /upload` | Sube una imagen al servidor |
| `POST /search` | Etapa 1: búsqueda por similitud (acepta `model` y `top_k`) |
| `POST /classify` | Etapa 2: clasificación con el modelo entrenado (acepta `model`) |
| `POST /detect` | Etapa 3: detección + clasificación |
| `GET /status/{job_id}` | Estado del procesamiento asincrónico |
| `GET /models` | Modelos de embeddings disponibles |
| `GET /health` | Estado del backend y checkpoints encontrados |

## Pipeline

### Provisto por la cátedra

- API asincrónica (`/upload`, `/search`, `/classify`, `/detect`, `/status`, `/models`) y manejo de jobs.
- Orquestación de cada etapa: `search` e `index_image` (Etapa 1), `classify_image` (Etapa 2) y `predict` (Etapa 3: detección → recorte → clasificación → JSON).
- Selección dinámica del modelo de embeddings: `baseline`, `resnet18_finetuned` o `cnn_custom`.
- Carga de checkpoints (`.pth` / `.onnx`) desde `models/` y selección del modelo activo.
- Similitud configurable (`cosine` o `l2`) y manejo de desconocidos con `SIMILARITY_THRESHOLD`.
- Persistencia configurable en JSON o en PostgreSQL + pgvector (`USE_PGVECTOR`).
- Funciones auxiliares de evaluación (`lib/evaluation/metrics.py`) y de visualización (`lib/visualization/draw.py`).
- Frontend Gradio con una pestaña por etapa: Etapa 1 (imagen consultada, Top-K similares, raza predicha y selección de modelo), Etapa 2 (raza y score con el modelo entrenado) y Etapa 3 (bounding boxes, razas y scores).

### Implementado por el grupo

- **Preprocesamiento compartido (`src/lib/preprocessing.py` y `src/lib/splits.py`):** conversión explícita `BGR → RGB`, resize con **letterbox** manteniendo relación de aspecto (padding `(124, 116, 104)`), normalización ImageNet, pipeline de `albumentations` aplicado solo sobre `train` on-the-fly y carga reproducible de `data/splits.csv` (normalizando nombres como `American  Spaniel`).
- **Etapa 1 (`src/lib/services/similarity_service.py`):**
  - `extract_embedding`: carga lazy de backbone preentrenado (`BASELINE_MODEL_NAME=resnet18`, 512-d por defecto; configurable a `resnet50`, `efficientnet_b0`, `convnext_tiny`) + normalización L2.
  - `search_similar_images`: búsqueda en `pgvector` (`<=>` para `cosine`) o `EmbeddingStore` JSON, soportando `cosine` y `l2` ordenados descendentemente.
  - `predict_breed_from_neighbors`: voto mayoritario ponderado por score sobre vecinos `>= SIMILARITY_THRESHOLD`, devolviendo `"unknown"` si no hay vecinos o el mejor score queda debajo del umbral.
- **Etapa 2 (`src/lib/services/classifier_service.py` y `src/lib/models/cnn_custom.py`):**
  - `train_classifier`: entrenamiento reproducible con `AdamW`, `ReduceLROnPlateau`, `class_weights`, `label_smoothing`, *early stopping* sobre validación, guardado de modelo completo (`.pth`) con atributo `.classes` e historial JSON en `output/`.
  - `evaluate_classifier`: métricas macro (`accuracy`, `precision`, `recall`, `specificity`, `f1`) usando `src/lib/evaluation/metrics.py` y exportación de detalle a `output/*_eval.json`.
  - `extract_custom_embedding`: embeddings L2-normalizados de dimensión `EMBEDDING_DIM` (`512`) sin mutar capas del modelo cacheado.
- **Etapa 3 (`src/lib/services/detection_service.py`):**
  - `detect_dogs`: carga lazy de YOLO (`YOLO_MODEL` desde `.env`, comparando `yolov8n.pt` vs `yolo26n.pt` en `2-train.ipynb`) filtrando por `YOLO_DOG_CLASS_ID=16` y `YOLO_CONF_THRESHOLD`.
  - `classify_detected_dog`: inferencia sobre cada recorte con el modelo activo (`.pth` o `.onnx`) usando el mismo preprocesamiento compartido y lista de razas guardada en el modelo.

Ver el contrato de cada función, recomendaciones y errores frecuentes en [ENUNCIADO.md](ENUNCIADO.md#24-funciones-a-implementar).

### Orden de trabajo y ejecución

1. Ejecutar `1-pre-processing.ipynb` en Google Colab (EDA, filtrado, conjunto independiente, preprocesamiento, augmentation y guardado de `data/splits.csv`).
2. Ejecutar `2-train.ipynb` en Google Colab con GPU T4 (evaluación Etapa 1 + calibración de umbral, entrenamiento de `resnet18_finetuned` y `cnn_custom`, análisis comparativo y comparativa `yolov8n.pt` vs `yolo26n.pt`).
3. Descargar los checkpoints a `models/` y actualizar `.env.*.example` con el umbral calibrado y modelo ganador.
4. Indexar la galería con `python scripts/build_index.py --split train` y validar end-to-end con `docker compose build && docker compose up`.

## Notebooks

| Notebook | Contenido |
|----------|-----------|
| `1-pre-processing.ipynb` | Dataset, EDA, control de calidad, conjunto independiente, preprocesamiento, data augmentation y splits |
| `2-train.ipynb` | Evaluación de la búsqueda (NDCG@10), entrenamiento del Modelo A y del Modelo B, métricas, análisis de errores, comparación y exportación de checkpoints |

Se pueden entregar commiteadas en el repositorio o en Google Colab, con el link a la notebook de Colab en una celda Markdown al inicio de la notebook del repositorio. En ambos casos deben estar **ejecutadas y con las salidas visibles**: una notebook sin salidas se considera **no entregada** (ver [ENUNCIADO.md](ENUNCIADO.md#82-notebooks-repositorio-o-google-colab)).

## Dataset

- **Dataset principal:** *70 Dog Breeds Image Dataset* (70 razas; splits `train/`, `valid/`, `test/`). Se descarga con `python scripts/download_dataset.py` y se filtra en `1-pre-processing.ipynb` generando `data/splits.csv` (incluyendo normalización del nombre `valid/American  Spaniel`).
- **Conteos por split y descartes por regla:** de **9.346** imágenes originales se descartaron **234** (86 casi-duplicados `dHash <= 3`, 43 duplicados exactos intra-split `SHA256`, 1 exposición extrema, 28 fugas inter-split `dHash` y 76 fugas inter-split `SHA256`), resultando **0 intersecciones** entre splits finales: `train=7.729`, `valid=687`, `test=696` (70 razas).
- **Conjunto independiente de evaluación:** **157 imágenes** totales (`125` de perro único en **25 razas** —5 por raza—, `17` escenas complejas/multi-perro y `15` sin perros), con **0 solapamiento** (`SHA256` y `dHash <= 3`) contra el dataset principal. Fuentes documentadas en `data/independent_sources.csv` (91 Stanford Dogs / ImageNet y 66 Unsplash License) y descarga pública en [Google Drive (`TP2_Perros`)](https://drive.google.com/drive/folders/1zHdSnTbOOp2r2IEwy-BI2DZDG3H6cmvy?usp=sharing).
- **Preprocesamiento y augmentation:** *letterbox* a `224×224` preservando relación de aspecto (padding `(124, 116, 104)`), normalización ImageNet (`BGR→RGB`) y aumentaciones on-the-fly (`HorizontalFlip`, `Affine` leve, `RandomBrightnessContrast`, `HueSaturationValue`, `GaussianBlur`, `GaussNoise`) aplicadas únicamente sobre `train`.

## Modelos y entrenamiento

- **Baseline Etapa 1 (`EMBEDDING_MODEL=baseline`):** `ResNet18` preentrenado en ImageNet (`512-d`, `EMBEDDING_DIM=512`), compatible dimensionalmente sin re-esquematizar la base vectorial al cambiar entre `baseline`, `resnet18_finetuned` y `cnn_custom`.
- **Modelo A (`resnet18_finetuned`):** `ResNet18` preentrenado con reemplazo de `fc` a 70 clases, comparando estrategia `head_only` vs `partial` (`layer4 + fc` con LR diferenciado, `AdamW`, `weight_decay=1e-4`, `label_smoothing=0.1`, `class_weights`, *early stopping*).
- **Modelo B (`cnn_custom`):** `CNNCustom` residual en `src/lib/models/cnn_custom.py` (4 etapas `64→128→256→512`, GAP, embedding 512-d, Dropout `0.4`), comparada en tabla de experimentos contra `CNNCustomSimple` sin conexiones residuales.
- **Detector YOLO (Etapa 3):** comparativa `yolov8n.pt` (19 detecciones en 8 escenas, 0 FP en 8 imágenes sin perros, 6.2 MB, `369.92 ms/img` CPU) vs `yolo26n.pt` (20 detecciones, 0 FP, 5.3 MB, `375.90 ms/img` CPU); modelo elegido: **`models/yolo26n.pt`**, `YOLO_CONF_THRESHOLD=0.25`.
- **Umbral de similitud:** calibrado conjuntamente sobre `valid` e `independent_no_dogs` (`SIMILARITY_THRESHOLD=0.80`, con `EMBEDDING_MODEL=resnet18_finetuned`, manteniendo `95.34%` en `valid` con `0%` de `"unknown"` y rechazando el `86.67%` de imágenes sin perros).
- **Links públicos de checkpoints y datos (solo lectura):** carpeta compartida en [Google Drive (`TP2_Perros`)](https://drive.google.com/drive/folders/1zHdSnTbOOp2r2IEwy-BI2DZDG3H6cmvy?usp=sharing) (contiene `resnet18_finetuned.pth`, `cnn_custom.pth` e `independent_dataset.zip`).

## Resultados

- **Etapa 1 (`NDCG@10`, `Precision@10`, `Accuracy` en `test` e independiente):**
  - **Baseline (`ResNet18` ImageNet, `th=0.80`):** `test`: `NDCG@10 = 0.8739`, `P@10 = 0.8603`, `Acc = 93.10%` | `independent`: `NDCG@10 = 0.6953`, `P@10 = 0.6904`, `Acc = 74.40%`.
  - **`resnet18_finetuned` (`th=0.80`, elegido):** `test`: **`NDCG@10 = 0.9565`**, **`P@10 = 0.9543`**, **`Acc = 96.41%`** | `independent`: `NDCG@10 = 0.6968`, `Acc = 68.00%`.
  - **`cnn_custom` (`th=0.75`):** `test`: `NDCG@10 = 0.8584`, `P@10 = 0.8546`, `Acc = 88.22%` | `independent`: `NDCG@10 = 0.3174`, `Acc = 33.60%`.
- **Etapa 2 (Modelo A vs Modelo B en `test`, `independent` y costo CPU):**

| Modelo | Acc (`test`) | F1 (`test`) | Acc (`indep`) | F1 (`indep`) | Params | Tamaño | CPU (`ms/img`) |
|---|---|---|---|---|---|---|---|
| `resnet18_finetuned` (`partial`) | **95.40%** | **0.9524** | **67.20%** | **0.7301** | 11,212,422 | 42.86 MB | 20.62 ms |
| `cnn_custom` (residual desde cero) | 86.93% | 0.8696 | 30.40% | 0.3530 | 11,212,422 | 42.86 MB | 20.93 ms |

  *(Referencias previas en `test`: `ResNet18 head_only = 92.82%` / `F1 = 0.9269`; `CNNCustomSimple` 10 épocas = `39.37%`. En `resnet18_finetuned` `specificity = 0.9993` en `test` y `0.9963` en `independent`.)*
- **Comparación entre enfoques y análisis de errores (FP/FN):** la clasificación supervisada (`resnet18_finetuned`) obtiene fronteras discriminativas estables en latencia $O(1)$ (`95.40%`), mientras que la búsqueda por similitud aporta explicabilidad visual (Top-K vecinos) y extensibilidad sin reentrenamiento (`NDCG@10 = 0.9565` con embeddings fine-tuned). Las confusiones principales ocurren entre razas morfológicamente cercanas (`American Spaniel ↔ Irish Spaniel`, `Bulldog ↔ Boston Terrier`, `Malinois ↔ German Sheperd`, `Yorkie ↔ Cairn`), y la caída en el conjunto independiente (`95.40% → 67.20%` y `86.93% → 30.40%`) evidencia el cambio de dominio y la mayor capacidad de generalización del preentrenamiento en ImageNet frente al entrenamiento desde cero.

## Problemas encontrados y soluciones

- **Doble espacio en carpeta de Kaggle (`valid/American  Spaniel`):** el dataset original difiere en el nombre respecto a `train`/`test`, desalineando índices de `ImageFolder`. Se normalizó mediante `normalize_breed_name` en `src/lib/splits.py` y se renombra automáticamente en las notebooks.
- **Paridad de preprocesamiento entre entrenamiento e inferencia:** se encapsuló el *letterbox* y normalización ImageNet en `src/lib/preprocessing.py` para evitar inconsistencias `BGR/RGB` o recortes agresivos sobre crops de YOLO.
- **Portabilidad de checkpoints en Docker sin dataset:** los checkpoints `.pth` guardan el modelo completo (importable desde `lib.models.cnn_custom`) junto con la lista ordenada `model.classes`.
- **Solapamiento del conjunto independiente y calibración de umbral:** la auditoría `dHash` en `1-pre-processing.ipynb` detectó 34 casi-duplicados provenientes de Stanford Dogs dentro del dataset de Kaggle, los cuales fueron reemplazados por imágenes de Unsplash hasta verificar **0 duplicados**. Asimismo, calibrar `SIMILARITY_THRESHOLD` conjuntamente sobre `valid` e `independent_no_dogs` evitó un umbral trivial (`0.30`) y permitió fijarlo en `0.80` (notando que la métrica `l2` mantiene idéntico `NDCG@10` pero requiere su propia escala de umbral distinta a `cosine`).

## Notas importantes

- Para usar `pgvector`, levanten `postgres` y definan `USE_PGVECTOR=true` en `.env`.
- `EMBEDDING_DIM` debe coincidir con la dimensión del embedding del modelo elegido. Si cambian `EMBEDDING_DIM`, la tabla `embeddings` se recrea y se borran los registros previos.
- La base vectorial guarda los embeddings de un modelo a la vez: al cambiar de modelo de embeddings hay que reindexar (`scripts/build_index.py`).
- Las rutas de la galería se guardan relativas a `DATA_PATH` (por ejemplo, `dataset/train/<raza>/<imagen>.jpg`): una base indexada desde el entorno local también sirve para el backend en Docker, porque comparten PostgreSQL y la carpeta `data/`.
- Los modelos (`*.pth`, `*.pt`, `*.onnx`) y los datos no se versionan (ver `.gitignore`). Coloquen los checkpoints en `models/` y súbanlos a un link **público de solo lectura** para que los docentes puedan descargarlos.
- Los pesos preentrenados (torchvision/timm y YOLO) se descargan en el primer uso, por lo que se necesita conexión a internet.

## Entrega

- Pull Request desde su feature branch hacia `main` **de su propio repositorio**, **sin mergear**, con el link subido al campus. No abran el PR contra el repositorio de la cátedra.
- Si el repositorio es privado, los docentes deben tener acceso.
- Notebooks `1-pre-processing.ipynb` y `2-train.ipynb` ejecutadas, con sus salidas.
- Para aprobar, el sistema debe andar sin errores con:

```bash
docker compose build
docker compose up
```

**Fecha de entrega:** a definir por la cátedra. Solo se consideran los commits hasta esa fecha y hora; se recomienda trabajar de forma incremental.

El detalle del contenido del PR, la plantilla de descripción y el checklist están en [ENUNCIADO.md](ENUNCIADO.md#8-entrega-pull-request).
