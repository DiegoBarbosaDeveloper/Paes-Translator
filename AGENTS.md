# AGENTS.md — Paes Translator

Guía operativa para agentes de código que trabajen en este repositorio.

## 1. Resumen del proyecto
Paes Translator es una aplicación web Flask para traducir entre español, piapoco y achagua. Combina:
- Diccionario persistente en PostgreSQL (tabla `language_pair`).
- Modelo generativo NLLB-200 + adaptador LoRA (opcional, requiere deps de IA).

La app arranca y funciona aunque no existan las dependencias de IA: en ese caso usa solo el diccionario y fallback al texto original.

## 2. Stack
- Python 3.10+
- Flask + Werkzeug + Gunicorn
- SQLAlchemy 2.0
- PostgreSQL (Supabase) vía psycopg v3
- SQLite para pruebas
- (Opcional) PyTorch, Transformers, PEFT, HuggingFace Hub

## 3. Estructura
- `app.py`: factory `create_app()`.
- `config.py`: configuración por ambiente (`dev/test/prod`).
- `controllers/api_controller.py`: blueprint HTTP.
- `services/`: lógica de negocio (`translation`, `dictionary`, `media`, `training`).
- `repositories/`: acceso ORM.
- `entities/models.py`: entidades SQLAlchemy + `init_db()` + `seed_defaults()`.
- `infrastructure/db.py`: engine, sesiones, `get_database_url()`, `reset_engine()`.
- `translation_model.py`: motor NLLB + respaldo de diccionario.
- `core/training/`: descarga de pesos, dataset builder, trainer.
- `scripts/`: `import_data.py`, `export_corpus.py`.
- `db/schema.postgres.sql`: esquema oficial PostgreSQL.
- `NLLBTest1/`: subproyecto de entrenamiento y CLI.

## 4. Variables de entorno
| Variable | Descripción | Default |
|---|---|---|
| `APP_ENV` | `dev`, `test`, `prod` | `dev` |
| `SECRET_KEY` | Clave Flask | `dev-secret-key-change-me` |
| `DATABASE_URL` | Conexión SQLAlchemy | `postgresql+psycopg://admin:admin123@localhost:5432/postgres` |
| `TEST_DATABASE_URL` | Base para tests | `sqlite+pysqlite:///:memory:` |
| `NLLB_ADAPTER_DIR` | Adaptador LoRA | `NLLBTest1/models/nllb-spa-piapoco-achagua` |
| `NLLB_BASE_MODEL_PATH` | Modelo base local | `models/nllb-200-distilled-600M` |
| `NLLB_BASE_MODEL_ID` | Id HuggingFace | `facebook/nllb-200-distilled-600M` |

Copiar `.env.example` a `.env`. Nunca commitear `.env`.

## 5. Instalación
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```
Para IA/entrenamiento:
```bash
pip install -r requirements-training.txt  # alias: requirements-ml.txt
```
Para desarrollo/tests:
```bash
pip install -r requirements-dev.txt
```

## 6. Base de datos
Aplicar esquema y cargar corpus (requiere los CSV en `data/`, que NO se versionan):
```bash
psql -U admin -d postgres -f db/schema.postgres.sql
python scripts/import_data.py --reset
```
Exportar corpus para entrenamiento:
```bash
python scripts/export_corpus.py
```
La app también crea/valida tablas al arrancar (`init_db()`). Supabase: usar el pooler URL en `DATABASE_URL`.

## 7. Ejecución
Desarrollo:
```bash
python app.py
```
Producción:
```bash
gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
```

## 8. Dependencias
- Producción: `requirements.txt` (solo web + BD).
- Entrenamiento/IA: `requirements-training.txt` / `requirements-ml.txt`.
- Desarrollo/tests: `requirements-dev.txt`.

No instalar dependencias de entrenamiento en el deploy de producción.

## 9. Convenciones
- Controladores no acceden al ORM; delegan a servicios.
- Validaciones funcionales en servicios.
- Fallo aislado del modelo: la app arranca aunque falte IA.
- Cambios de esquema: actualizar `entities/models.py` y `db/schema.postgres.sql`, y validar con `init_db()`.
- No commitear secretos ni `data/`.

## 10. Endpoints principales
Ver README.md. Rutas clave: `/api/languages`, `/api/translate-text`, `/api/model-info`, `/add_word`, `/api/feedback`, `/api/training/*`.

## 11. Pruebas
Configuración de test con SQLite en memoria (`TEST_DATABASE_URL`). Ejecutar:
```bash
pytest
```
Actualmente no hay tests automatizados, pero la infraestructura está lista.

## 12. Notas de despliegue
- Usar Gunicorn como servidor WSGI.
- `runtime.txt` define Python 3.12.x.
- `data/` está ignorado: los CSV de corpus no viajan con el repo; importar manualmente o montarlos.
- `static/uploads/` es efímero en PaaS; para persistencia usar almacenamiento externo.
- Supabase transaction pooler: la URL ya funciona con psycopg v3; mantener `pool_pre_ping=True`.

## 13. Troubleshooting
- `SchemaValidationError`: aplicar `db/schema.postgres.sql` y verificar columnas/tipos.
- DB vacía: ejecutar `scripts/import_data.py --reset`.
- Modelo no disponible: instalar `requirements-training.txt` y descargar pesos; la app sigue funcionando con diccionario.
- Import `reset_engine` faltante: ya implementado en `infrastructure/db.py`.
