# Paes Translator

Aplicación web de traducción multilingüe centrada en el par **español ⇄ piapoco/achagua**, construida sobre un modelo generativo **NLLB-200** ajustado con **LoRA (PEFT)** y respaldada por un diccionario persistente. Incluye interfaz web, diccionario editable, gestión de medios, reconocimiento de voz en el navegador y un panel de entrenamiento.

> Documentación interna adicional en [`docs/README.md`](docs/README.md).

---

## Tabla de contenidos

- [¿Qué es y qué resuelve?](#qué-es-y-qué-resuelve)
- [Características](#características)
- [Arquitectura](#arquitectura)
- [Tecnologías y por qué se usan](#tecnologías-y-por-qué-se-usan)
- [El modelo de traducción (NLLB + LoRA)](#el-modelo-de-traducción-nllb--lora)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Modelo de datos](#modelo-de-datos)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Cómo ejecutar](#cómo-ejecutar)
- [Entrenamiento del modelo](#entrenamiento-del-modelo)
- [API REST](#api-rest)
- [Flujo de una traducción](#flujo-de-una-traducción)
- [Limitaciones y notas](#limitaciones-y-notas)
- [Licencia](#licencia)

---

## ¿Qué es y qué resuelve?

Paes Translator es un traductor de lenguas indígenas del **Paisaje Lingüístico (Paes)** que combina dos estrategias complementarias:

1. **Traducción contextual con IA**: un adaptador LoRA sobre `facebook/nllb-200-distilled-600M` entrenado para las direcciones español ⇄ piapoco y español ⇄ achagua.
2. **Diccionario / memoria de traducción**: pares lingüísticos persistidos en base de datos, usados como respaldo rápido y como fuente de datos para reentrenar el modelo.

La aplicación es **auto-alimentada**: cada palabra o corrección que un usuario registra en el diccionario puede reconstruir el *dataset* de entrenamiento y, con ello, mejorar el modelo en la siguiente iteración de *fine-tuning*.

Idiomas soportados por el adaptador NLLB: **español (`spa_Latn`)**, **piapoco (`piapoco_Latn`)** y **achagua (`achagua_Latn`)**.

---

## Características

- **Traducción en 4 direcciones**: español → piapoco, piapoco → español, español → achagua, achagua → español.
- **Prioridad al modelo NLLB**: la estrategia generativa se intenta siempre primero; el diccionario solo actúa si el modelo no está disponible o no produce una traducción válida.
- **Diccionario colaborativo**: alta de pares, corrección de traducciones (*feedback*) y reconstrucción automática del dataset.
- **Gestión de medios**: imágenes y audios asociados a palabras clave.
- **Voz a traducción**: dictado por voz con la Web Speech API del navegador (Chrome/Edge).
- **Panel de modelo**: descarga de pesos base desde HuggingFace, inicio/detención de entrenamiento, log en tiempo real y recarga del modelo en caliente.
- **Tema claro/oscuro** y diseño responsivo.
- **CLI independiente** (`NLLBTest1`) para entrenar, evaluar (chrF++) y traducir desde terminal.

---

## Arquitectura

El backend sigue una **arquitectura en capas**, de modo que la lógica HTTP, la lógica de negocio y la persistencia estén desacopladas:

```
HTTP          controllers/api_controller.py     (Blueprint Flask, validación de entrada)
                  │
Negocio       services/                          (TranslationService, DictionaryService,
                  │                               MediaService, TrainingService)
Persistencia  repositories/  +  entities/models.py  (SQLAlchemy ORM)
                  │
Infra         infrastructure/db.py  +  database.py   (engine, sesiones, facade)
                  │
IA            translation_model.py  +  core/training/ + NLLBTest1/  (NLLB + LoRA)
```

Principios aplicados:

- **Los controladores no tocan el ORM**: solo extraen JSON válido y delegan a los servicios.
- **Fallo aislado del modelo**: si el adaptador NLLB no existe, falta `peft` o falla la descarga, la app **igualmente arranca** y usa el diccionario.
- **Inyección de dependencias sencilla**: `services/container.py` construye el grafo de objetos y Flask lo expone en `app.config["services"]`.
- **Frontend sin frameworks**: una sola página HTML + JavaScript "vanilla", alimentada por la API REST.

---

## Tecnologías y por qué se usan

| Área | Tecnología | Por qué se eligió |
|---|---|---|
| Framework web | **Flask** + **Werkzeug** | Micro-framework ligero y explícito, ideal para una API REST pequeña y una SPA servida como plantilla. No impone estructura y facilita la separación en capas. |
| ORM / persistencia | **SQLAlchemy 2.0** | Modelo declarativo tipado, portable entre motores (MySQL y SQLite) y con validación estricta del esquema. Evita SQL disperso y errores de integridad. |
| Base de datos (defecto) | **MySQL (XAMPP)** vía **PyMySQL** | El proyecto se despliega en entornos escolares sobre XAMPP. PyMySQL es un driver 100 % Python (sin compilación) que simplifica la instalación en Windows. |
| Base de datos (pruebas) | **SQLite** | Configuración de test en memoria (`sqlite+pysqlite:///:memory:`) y ejecución local sin servidor. También útil para demos rápidas. |
| Modelo de IA | **Transformers** + **PyTorch** | Ecosistema estándar para cargar NLLB-200 y ejecutar generación. PyTorch permite CPU y GPU (CUDA) con el mismo código; NLLB es un modelo seq2seq multilingüe ya entrenado (solo requiere *fine-tuning*). |
| Adaptación eficiente | **PEFT (LoRA)** | En lugar de reentrenar 600 M de parámetros, LoRA congela el modelo base e inyecta matrices de bajo rango. Reduce memoria/tiempo y produce adaptadores de pocos MB, fáciles de versionar. |
| Tokenización | **tokenizers** + **SentencePiece** | NLLB se tokeniza con SentencePiece; se añaden tokens de idioma nuevos (`piapoco_Latn`, `achagua_Latn`) como tokens especiales. |
| Distribución/descarga | **HuggingFace Hub** | `snapshot_download` obtiene los pesos base (~2.4 GB) y la arquitectura PEFT permite cargar el adaptador sin copiar el modelo completo. |
| Evaluación | **sacrebleu** (chrF++) | Métrica robusta para lenguas con mucha morfología y corpus pequeños, donde BLEU es poco fiable. chrF++ (con orden de palabra) refleja mejor la calidad en piapoco/achagua. |
| ML auxiliar | **scikit-learn**, **datasets**, **accelerate** | División *train/eval*, aceleración de dispositivos y manejo de datasets durante el entrenamiento. |
| Datos/notebooks | **pandas**, **numpy**, **matplotlib**, **Jupyter** | Preparación del corpus (`language_pair.csv`), análisis de resultados y experimentación en `NLLBTest1/notebooks/`. |
| Frontend | **HTML + CSS + JavaScript vanilla** | Cero dependencias de build; carga instantánea y fácil mantenimiento. El theming se hace con variables CSS. |
| Voz | **Web Speech API** (`SpeechRecognition`) | Reconocimiento de voz en el cliente, sin costes ni backend adicional; soportado por Chrome/Edge. |
| Configuración | **python-dotenv** + variables de entorno | Permite cambiar base de datos, secreto y rutas del modelo sin editar código. |

---

## El modelo de traducción (NLLB + LoRA)

El corazón del sistema es `translation_model.py` → clase `AdvancedTranslationModel`.

### Base y adaptador
- **Modelo base**: `facebook/nllb-200-distilled-600M`.
- **Adaptador LoRA**: `NLLBTest1/models/nllb-spa-piapoco-achagua/` (producido por `NLLBTest1`).
- **Tags de idioma**: `spa_Latn`, `piapoco_Latn`, `achagua_Latn`.

### Secuencia exacta de carga
Cargar el adaptador correctamente requiere replicar el proceso de entrenamiento:

1. **Tokenizer del adaptador** (incluye los tokens especiales `piapoco_Latn` y `achagua_Latn`).
2. **Modelo base NLLB** con los *embeddings* redimensionados al vocabulario del tokenizer del adaptador.
3. **Vectores de los tags nuevos** restaurados desde `tag_vectors.pt` (el adaptador LoRA no los guarda).
4. Envolver el modelo con **`PeftModel`** para aplicar los pesos LoRA.

### Inferencia
- Se fija `tokenizer.src_lang` con el tag de origen.
- Se fuerza el token de idioma destino con `forced_bos_token_id`.
- Generación con `num_beams=4`, `max_length=512`, `use_cache=True`.
- Media precisión (`bfloat16`/`float16`) en GPU y `float32` en CPU.

### Estrategia y respaldo
`translate()` aplica este orden **estrictamente**:

1. **NLLB** (siempre primero, cuando el modelo está cargado y la dirección es soportada).
2. **Diccionario** persistido (si NLLB no devuelve resultado válido).
3. **Texto original** (fallback final, para no romper la respuesta).

Las direcciones soportadas por el adaptador son solo español ⇄ piapoco y español ⇄ achagua; cualquier otro idioma cae al diccionario.

---

## Estructura del proyecto

```
Paes-Translator/
├── app.py                     # Punto de entrada Flask (factory create_app)
├── config.py                  # Configuración por ambiente (dev/test/prod)
├── database.py                # Fachada de acceso a datos (facade sobre repositorios)
├── translation_model.py       # Motor NLLB (LoRA) + respaldo de diccionario
├── requirements.txt           # Dependencias de la app
├── .env.example               # Ejemplo de variables de entorno
├── controllers/
│   └── api_controller.py      # Blueprint con todas las rutas HTTP
├── services/
│   ├── container.py           # Inyección de dependencias
│   ├── translation_service.py # Validación y orquestación de traducción
│   ├── dictionary_service.py  # Alta de palabras y feedback
│   ├── media_service.py       # Subida y listado de imágenes/audios
│   └── training_service.py    # Orquestación del entrenamiento/descarga
├── repositories/
│   ├── dictionary_repository.py  # Pares lingüísticos (tabla language_pair)
│   └── media_repository.py       # Inventario de medios en disco
├── entities/
│   └── models.py              # Entidades SQLAlchemy + init_db + seed_defaults
├── infrastructure/
│   └── db.py                  # Engine, sesiones y creación de base MySQL
├── core/training/
│   ├── training_config.py     # Rutas e hiperparámetros
│   ├── download_model.py      # Descarga de pesos base desde HuggingFace
│   ├── dataset_builder.py     # Construcción del dataset desde el diccionario
│   └── trainer.py             # Estado/log del entrenamiento (reconstruye dataset)
├── templates/index.html       # Interfaz de una sola página
├── static/
│   ├── app.js                 # Lógica de la interfaz
│   ├── style.css              # Estilos y temas
│   └── uploads/               # Imágenes y audios subidos
├── docs/
│   ├── README.md
│   └── LICENSE
└── NLLBTest1/                 # Subproyecto de entrenamiento y CLI
    ├── pyproject.toml         # Entry point: santoto-translator
    ├── src/
    │   ├── config.py          # Idiomas, direcciones, LoRA, hiperparámetros
    │   ├── modeling.py        # Carga del modelo, tags y traducción
    │   ├── train.py           # Fine-tuning LoRA + evaluación chrF++
    │   ├── cli.py             # Interfaz de línea de comandos
    │   ├── build_pairs.py     # Prepara CSVs desde language_pair.csv
    │   ├── tui.py             # Interfaz de terminal
    │   └── display.py, files.py, console.py
    ├── data/                  # Corpus crudo y procesado
    └── models/                # Adaptadores entrenados
```

---

## Modelo de datos

El esquema se declara con SQLAlchemy en `entities/models.py` y se **valida estrictamente** al arrancar (tablas, columnas, tipos, nulabilidad, claves primarias y foráneas). Tablas:

| Tabla | Descripción |
|---|---|
| `languages` | Catálogo de idiomas (`id`, `code`, `name`, `active`). |
| `dataset` | Conjunto de datos activo (`name`, `version`, `status`, ...). |
| `translation_direction` | Direcciones permitidas origen → destino. |
| `language_pair` | **El diccionario**: pares de texto y su estado de validación. |
| `model_version` | Versiones del modelo entrenado. |
| `evaluation` | Métricas automáticas (BLEU, chrF, COMET). |
| `human_evaluation` | Evaluación humana (adecuación, fluidez, global). |

**Inicialización**: `init_db()` crea la base (MySQL) y las tablas si faltan, y luego valida el esquema. `seed_defaults()` siembra, cuando están vacías, tres idiomas (`es`/Español, `piapoco`, `achagua`) y un dataset activo, de modo que el desplegable de idiomas **siempre** tenga opciones.

Los identificadores usan `BIGINT` sin `autoincrement` (se calcula el siguiente id disponible).

---

## Instalación

Requisitos: **Python 3.10+**, y opcionalmente **MySQL (XAMPP)**.

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd Paes-Translator

# 2. (Recomendado) Crear un entorno virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Instalar dependencias
pip install -r requirements.txt
```

> El modelo base NLLB y el adaptador LoRA requieren `torch`, `transformers` y `peft` (ya incluidos en `requirements.txt`). En la primera ejecución, si no existen los pesos locales, se descargarán desde HuggingFace (~2.4 GB) o se podrá descargar desde el panel **Modelo IA**.

---

## Configuración

Copia `.env.example` a `.env` y ajusta los valores, o define las variables en el entorno.

| Variable | Descripción | Valor por defecto |
|---|---|---|
| `APP_ENV` | Ambiente (`dev`, `test`, `prod`) | `dev` |
| `SECRET_KEY` | Clave secreta de Flask | `dev-secret-key-change-me` |
| `DATABASE_URL` | Conexión SQLAlchemy | `mysql+pymysql://admin:admin123@localhost:3306/nasa_yuwe_translator` |
| `TEST_DATABASE_URL` | Base para pruebas | `sqlite+pysqlite:///:memory:` |
| `NLLB_ADAPTER_DIR` | Ruta del adaptador LoRA | `NLLBTest1/models/nllb-spa-piapoco-achagua` |
| `NLLB_BASE_MODEL_PATH` | Ruta local del modelo base | `models/nllb-200-distilled-600M` |
| `NLLB_BASE_MODEL_ID` | Id del modelo base en HuggingFace | `facebook/nllb-200-distilled-600M` |

Si no quieres levantar MySQL, puedes usar SQLite para probar:

```powershell
$env:DATABASE_URL="sqlite:///C:/ruta/al/proyecto/data/app.db"; $env:SECRET_KEY="dev"; python app.py
```

---

## Cómo ejecutar

### Aplicación web

```bash
python app.py
```

Abre `http://127.0.0.1:5000` en el navegador. Al arrancar, la app:

1. Inicializa/valida la base de datos.
2. Siembra idiomas por defecto si el catálogo está vacío.
3. Construye los servicios y registra las rutas.
4. Carga el modelo NLLB **de forma perezosa** en la primera traducción.

### CLI del subproyecto (entrenar/ traducir)

```bash
cd NLLBTest1
pip install -r requirements.txt

# Traducir desde la terminal
python -m src.cli --src spanish --tgt piapoco --text "Buenos dias"

# Evaluar con chrF++ contra un CSV de referencia
python -m src.cli --src spanish --tgt achagua --csv-column source --csv-reference reference

# Interfaz de terminal (TUI)
python -m src.cli --ui tui

# Ver direcciones disponibles
python -m src.cli --list-directions
```

Si el paquete está instalado, el entry point es `santoto-translator` (definido en `NLLBTest1/pyproject.toml`).

---

## Entrenamiento del modelo

Hay **dos rutas** de entrenamiento:

### 1. Subproyecto `NLLBTest1` (entrenamiento real)

`NLLBTest1/src/train.py` realiza el *fine-tuning* LoRA real:

- Divide el corpus por idioma (opcionalmente con *holdout*).
- Construye pares en ambas direcciones.
- Aplica `LoraConfig` (`r`, `lora_alpha`, `lora_dropout`, `target_modules=q_proj,k_proj,v_proj,out_proj,fc1,fc2`).
- Usa `Seq2SeqTrainingArguments` (bf16/fp16, `save_total_limit=1`).
- Evalúa cada dirección con **chrF++** (`sacrebleu`) y escribe `training_report.json`.
- Guarda el adaptador, el tokenizer y `tag_vectors.pt` en `models/nllb-spa-piapoco-achagua/`.

Hiperparámetros típicos en `src/config.py`: `EPOCHS=30`, `LR=5e-4`, `LORA_R=16` (CPU) / `32` (GPU), `LORA_ALPHA=32`/`64`, `WARMUP_RATIO=0.05`.

### 2. Panel **Modelo IA** en la app web

Desde la interfaz se puede:

- **Descargar** los pesos base (~2.4 GB) desde HuggingFace.
- **Iniciar/Detener** el proceso y ver el **log en tiempo real**.
- **Aplicar modelo** (recargar el adaptador en memoria sin reiniciar).

> Importante: en la app web, `/api/training/*` reconstruye el **dataset** (`data/training_pairs.json`) a partir del diccionario y gestiona la descarga de pesos; el *fine-tuning* real con LoRA se ejecuta en `NLLBTest1/src/train.py`.

---

## API REST

Todas las rutas viven en `controllers/api_controller.py`.

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Interfaz principal. |
| `GET` | `/api/languages` | Catálogo de idiomas activos. |
| `POST` | `/api/translate-text` | Traduce `{text, source_lang, target_lang}`. |
| `GET` | `/api/model-info` | Estado del modelo (tipo, dispositivo, idiomas, ...). |
| `POST` | `/api/upload-media` | Sube imagen/audio (`file`, `type`, `tag`). |
| `GET` | `/api/media-list` | Lista medios por `type` y `tag`. |
| `POST` | `/add_word` | Añade un par al diccionario. |
| `POST` | `/api/feedback` | Guarda una corrección (upsert). |
| `POST` | `/api/training/start` | Inicia el proceso de entrenamiento. |
| `POST` | `/api/training/stop` | Solicita detener el entrenamiento. |
| `GET` | `/api/training/status` | Estado del entrenamiento. |
| `GET` | `/api/training/log` | Últimas líneas del log (`lines`). |
| `POST` | `/api/training/download-model` | Inicia la descarga de pesos base. |
| `GET` | `/api/training/download-status` | Estado de la descarga. |
| `POST` | `/api/training/reload-model` | Recarga el adaptador en memoria. |
| `POST` | `/api/training/rebuild-dataset` | Reconstruye el dataset desde el diccionario. |

Ejemplo de traducción:

```bash
curl -X POST http://127.0.0.1:5000/api/translate-text \
  -H "Content-Type: application/json" \
  -d '{"text": "Buenos dias", "source_lang": "es", "target_lang": "piapoco"}'
```

---

## Flujo de una traducción

```
Usuario escribe texto
      │
      ▼
doTranslate()  ──►  POST /api/translate-text
      │                    │
      │                    ▼
      │            TranslationService.translate_text()   (validación)
      │                    │
      │                    ▼
      │            AdvancedTranslationModel.translate()
      │                    │
      │         ┌──────────┴──────────┐
      │         ▼                     ▼
      │   NLLB (LoRA)            Diccionario
      │   ¿resultado válido?     ¿coincidencia?
      │         │                     │
      │         └──────────┬──────────┘
      │                    ▼
      │            texto traducido / original (fallback)
      ▼
Respuesta con translation + method + confidence
```

La interfaz muestra un **estado de carga** en el botón (*spinner* + "Traduciendo...") mientras se espera la respuesta y una barra de progreso indeterminada en el panel.

---

## Limitaciones y notas

- El adaptador NLLB solo cubre **español ⇄ piapoco/achagua**. Otros idiomas del catálogo usan únicamente el diccionario.
- El diccionario funciona **palabra por palabra** (traducción literal por token), por lo que es un respaldo, no un traductor de frases.
- La primera carga del modelo puede tardar (descarga de ~2.4 GB y carga en CPU).
- Para desarrollo local del backend se usa el servidor de Flask; en producción debe usarse un WSGI serio (p. ej. Waitress/Gunicorn).
- Actualmente no hay pruebas automatizadas en el repositorio, aunque `pytest` y `pytest-flask` están contemplados en `requirements.txt` y existe una configuración de test con SQLite en memoria.

---

## Licencia

Copyright (c) 2025 — **Todos los derechos reservados**. Ver [`docs/LICENSE`](docs/LICENSE).
