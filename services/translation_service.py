"""
Servicio de traducción independiente de idiomas concretos.

La capa HTTP solo proporciona texto y códigos. Este servicio rechaza entradas
malformadas, evita direcciones idénticas y deja que el modelo consulte el
catálogo persistido. No mantiene una lista de idiomas permitidos porque esa
lista pertenece a ``languages`` y puede cambiar sin desplegar código nuevo.
"""

from __future__ import annotations

from translation_model import AdvancedTranslationModel


class TranslationService:
    """Gestiona validación funcional y uso del modelo de traducción."""

    def __init__(self):
        self._translation_model = None

    def _get_model(self) -> AdvancedTranslationModel:
        """Inicializar el modelo bajo demanda para reducir tiempo de arranque."""
        if self._translation_model is None:
            self._translation_model = AdvancedTranslationModel()
        return self._translation_model

    def reset_model(self) -> None:
        """Forzar recarga del modelo en la siguiente traducción."""
        self._translation_model = None

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> dict:
        """Aplicar reglas de validación y ejecutar traducción."""
        if not isinstance(text, str) or not text.strip():
            return {"ok": False, "status_code": 400, "payload": {"error": "No se proporcionó texto para traducir"}}

        if not isinstance(source_lang, str) or not isinstance(target_lang, str):
            return {
                "ok": False,
                "status_code": 400,
                "payload": {"error": "Los códigos de idioma deben ser texto"},
            }

        source_lang = source_lang.strip()
        target_lang = target_lang.strip()
        if source_lang.casefold() == target_lang.casefold():
            return {
                "ok": False,
                "status_code": 400,
                "payload": {"error": "El idioma de origen y destino no pueden ser iguales"},
            }

        if not source_lang or not target_lang:
            return {
                "ok": False,
                "status_code": 400,
                "payload": {"error": "Debe indicar los idiomas de origen y destino"},
            }

        result = self._get_model().translate(text, source_lang, target_lang)
        return {
            "ok": True,
            "status_code": 200,
            "payload": {
                "translation": result["translation"],
                "status": "success",
                "method": result["method"],
                "confidence": result["confidence"],
                "methods_tried": result.get("methods_tried", []),
            },
        }

    def get_model_info(self) -> dict:
        """Retornar metadatos operativos del modelo de traducción."""
        info = self._get_model().get_model_info()
        return {"status": "success", "model_info": info}
