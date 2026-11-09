# CLAUDE.md — Paes Translator

Lee primero `AGENTS.md`: contiene la guía completa del proyecto. Este archivo resume lo esencial para trabajar con Claude Code.

## Reglas rápidas
- No instalar dependencias de entrenamiento en producción; usar `requirements.txt`.
- No commitear `.env` ni `data/` (información sensible).
- Controladores no tocan ORM; delegar a servicios.
- El modelo NLLB es opcional: la app debe arrancar sin torch/transformers.
- Cambios de esquema: actualizar `entities/models.py` + `db/schema.postgres.sql`.

## Comandos
```bash
# Dev
python app.py

# Prod
gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120

# Tests
pytest

# BD
psql -U admin -d postgres -f db/schema.postgres.sql
python scripts/import_data.py --reset
python scripts/export_corpus.py
```

## Dependencias
- Web/BD: `requirements.txt`
- IA/entrenamiento: `requirements-training.txt` (alias `requirements-ml.txt`)
- Dev/tests: `requirements-dev.txt`

## Arquitectura
Flask (`controllers`) -> servicios (`services`) -> repositorios (`repositories`) -> ORM (`entities`) -> PostgreSQL.

## Archivos clave
- `app.py`, `config.py`, `wsgi.py`
- `controllers/api_controller.py`
- `services/container.py`, `services/translation_service.py`
- `translation_model.py`
- `entities/models.py`, `infrastructure/db.py`
- `scripts/import_data.py`, `scripts/export_corpus.py`

## Despliegue
- Gunicorn + `runtime.txt` (Python 3.12.x).
- `Procfile` incluido.
- `data/` no se versiona; importar corpus manualmente.
- Uploads efímeros en PaaS.
