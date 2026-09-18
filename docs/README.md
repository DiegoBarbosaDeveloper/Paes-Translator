# Paes Translator

Aplicación web de traducción multilingüe basada en Flask, SQLAlchemy y un modelo generativo opcional.

## Modelo de datos

La aplicación usa directamente las tablas definidas en `schema.sql`:

- `languages` contiene el catálogo de idiomas y sus códigos.
- `language_pair` contiene los textos y sus direcciones.
- `dataset` identifica el conjunto de datos activo.
- `model_version`, `evaluation` y `human_evaluation` contienen el ciclo de evaluación y entrenamiento.

El ORM valida al iniciar que la base de datos coincida exactamente con las tablas, columnas, tipos, nulabilidad, claves primarias y restricciones foráneas del esquema. No crea ni modifica tablas automáticamente.

## Configuración

1. Instala las dependencias con `pip install -r requirements.txt`.
2. Define `DATABASE_URL` y `SECRET_KEY` en el entorno.
3. Ejecuta el contenido de `schema.sql` en la base de datos.
4. Carga al menos dos idiomas activos y un dataset activo.
5. Inicia la aplicación con `python app.py`.

El frontend obtiene los idiomas mediante `/api/languages`; no contiene códigos ni nombres de idiomas fijos.

## API principal

- `GET /api/languages`
- `POST /api/translate-text` con `text`, `source_lang` y `target_lang`
- `POST /add_word` con `source_text`, `target_text`, `source_lang` y `target_lang`
- `POST /api/feedback` con la dirección seleccionada

Los pares lingüísticos se consultan desde `language_pair`, y los archivos multimedia se mantienen en `static/uploads` porque el esquema no define una tabla para ellos.
