"""
Controlador principal de la API REST.

Este módulo es una frontera de confianza: todo lo que llega por HTTP es dato
externo y puede tener tipos, campos o valores inesperados. Los endpoints solo
extraen objetos JSON válidos, validan los campos mínimos y delegan la decisión
de negocio en los servicios. No acceden directamente al ORM, de modo que los
errores de persistencia y las reglas multilingües permanecen centralizados.
"""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template, request


api_bp = Blueprint("api", __name__)


def _service(name: str):
    """Resolver servicio desde el contenedor registrado en Flask app config."""
    return current_app.config["services"][name]


def _json_object():
    """Obtener un objeto JSON o devolver ``None`` para responder con 400."""
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else None


def _text_field(data: dict, name: str) -> str:
    """Leer un campo textual sin permitir que tipos inesperados rompan la API."""
    value = data.get(name, "")
    return value.strip() if isinstance(value, str) else ""


@api_bp.route("/")
def index():
    """Renderizar interfaz principal."""
    return render_template("index.html")


@api_bp.route("/api/languages", methods=["GET"])
def get_languages():
    """
    Exponer el catálogo activo que gobierna toda la interfaz.

    El frontend no debe conocer códigos ni nombres de idiomas por adelantado:
    los recibe de esta ruta y los reenvía en cada operación de traducción,
    alta o retroalimentación. Así la configuración de la base controla el
    comportamiento disponible sin editar plantillas ni JavaScript.
    """
    try:
        return jsonify(_service("dictionary").get_languages())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api_bp.route("/api/translate-text", methods=["POST"])
def translate_text_endpoint():
    """Endpoint de traducción textual entre dos idiomas configurados."""
    try:
        data = _json_object()
        if data is None:
            return jsonify({"error": "El cuerpo debe ser un objeto JSON válido"}), 400
        text = _text_field(data, "text")
        source_lang = _text_field(data, "source_lang")
        target_lang = _text_field(data, "target_lang")

        if not text or not source_lang or not target_lang:
            return jsonify({"error": "Texto e idiomas son obligatorios"}), 400

        result = _service("translation").translate_text(text, source_lang, target_lang)
        return jsonify(result["payload"]), result["status_code"]
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api_bp.route("/api/model-info", methods=["GET"])
def get_model_info():
    """Obtener metadata operativa del modelo de traducción."""
    try:
        info = _service("translation").get_model_info()
        return jsonify(info)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api_bp.route("/api/upload-media", methods=["POST"])
def upload_media():
    """Subir un archivo multimedia y registrar metadatos."""
    try:
        media_type = request.form.get("type", "").strip()
        tag = request.form.get("tag", "").strip()

        if "file" not in request.files:
            return jsonify({"error": "No se envió ningún archivo"}), 400

        uploaded_file = request.files["file"]
        if uploaded_file.filename == "":
            return jsonify({"error": "Nombre de archivo vacío"}), 400

        result = _service("media").upload_media(uploaded_file, media_type, tag)
        return jsonify(result["payload"]), result["status_code"]
    except Exception as exc:
        return jsonify({"error": f"Error al subir el archivo: {str(exc)}"}), 500


@api_bp.route("/api/media-list", methods=["GET"])
def media_list():
    """Listar archivos multimedia registrados."""
    try:
        media_type = request.args.get("type", "").strip()
        tag = request.args.get("tag", "").strip()

        result = _service("media").list_media(media_type, tag if tag else None)
        return jsonify(result["payload"]), result["status_code"]
    except Exception as exc:
        return jsonify({"error": f"Error al listar medios: {str(exc)}"}), 500


@api_bp.route("/add_word", methods=["POST"])
def add_word():
    """Agregar una nueva palabra al diccionario."""
    try:
        data = _json_object()
        if data is None:
            return jsonify({"error": "El cuerpo debe ser un objeto JSON válido"}), 400
        source_text = _text_field(data, "source_text")
        target_text = _text_field(data, "target_text")
        source_lang = _text_field(data, "source_lang")
        target_lang = _text_field(data, "target_lang")

        if not source_text or not target_text or not source_lang or not target_lang:
            return jsonify({"error": "Todos los campos son obligatorios"}), 400

        result = _service("dictionary").add_word(source_text, target_text, source_lang, target_lang)
        return jsonify(result["payload"]), result["status_code"]
    except Exception as exc:
        return jsonify({"error": f"Error al agregar la palabra: {str(exc)}"}), 500


@api_bp.route("/api/feedback", methods=["POST"])
def receive_feedback():
    """Recibir retroalimentación de traducciones para mejora continua."""
    try:
        data = _json_object()
        if data is None:
            return jsonify({"error": "El cuerpo debe ser un objeto JSON válido"}), 400
        original_text = _text_field(data, "original_text")
        corrected_translation = _text_field(data, "corrected_translation")
        source_lang = _text_field(data, "source_lang")
        target_lang = _text_field(data, "target_lang")

        if not original_text or not corrected_translation or not source_lang or not target_lang:
            return jsonify({"error": "Texto, corrección e idiomas son obligatorios"}), 400

        persisted = _service("dictionary").process_feedback(
            original_text,
            corrected_translation,
            source_lang,
            target_lang,
        )
        if not persisted:
            return jsonify({"error": "No se pudo guardar la retroalimentación"}), 503
        return jsonify({"status": "success", "message": "Retroalimentación guardada exitosamente"})
    except Exception as exc:
        return jsonify({"error": f"Error al procesar la retroalimentación: {str(exc)}"}), 500


@api_bp.route("/api/training/start", methods=["POST"])
def training_start():
    """Iniciar entrenamiento del modelo en segundo plano."""
    try:
        return jsonify(_service("training").start())
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500


@api_bp.route("/api/training/stop", methods=["POST"])
def training_stop():
    """Solicitar detención del entrenamiento en curso."""
    try:
        return jsonify(_service("training").stop())
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500


@api_bp.route("/api/training/status", methods=["GET"])
def training_status():
    """Consultar estado del entrenamiento."""
    try:
        return jsonify(_service("training").status())
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500


@api_bp.route("/api/training/log", methods=["GET"])
def training_log():
    """Obtener últimas líneas del log de entrenamiento."""
    try:
        lines = request.args.get("lines", 50, type=int)
        return jsonify(_service("training").log(lines))
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500


@api_bp.route("/api/training/download-model", methods=["POST"])
def training_download_model():
    """Iniciar descarga de pesos base del modelo."""
    try:
        return jsonify(_service("training").start_model_download())
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500


@api_bp.route("/api/training/download-status", methods=["GET"])
def training_download_status():
    """Consultar estado de descarga de pesos del modelo."""
    try:
        return jsonify(_service("training").model_download_status())
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500


@api_bp.route("/api/training/reload-model", methods=["POST"])
def training_reload_model():
    """Recargar modelo en memoria para usar último artefacto disponible."""
    try:
        return jsonify(_service("training").reload_model())
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500


@api_bp.route("/api/training/rebuild-dataset", methods=["POST"])
def training_rebuild_dataset():
    """Reconstruir dataset de entrenamiento desde el diccionario actual."""
    try:
        payload, status = _service("training").rebuild_dataset()
        return jsonify(payload), status
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500
