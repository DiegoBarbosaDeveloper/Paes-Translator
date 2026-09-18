"""
Servicio de negocio para gestionar pares lingüísticos.

Esta capa valida reglas funcionales antes de delegar en el repositorio. Una
solicitud puede ser válida y aun así no poder persistirse, por ejemplo cuando
no existe un dataset activo o falla una transacción; por eso nunca informa
éxito sin comprobar explícitamente el resultado de la escritura.
"""

from __future__ import annotations


class DictionaryService:
    """Orquesta reglas del diccionario desacoplando controlador y repositorio."""

    def __init__(self, dictionary_repository):
        self.dictionary_repository = dictionary_repository

    def get_languages(self) -> list[dict]:
        """Obtener idiomas configurados para los selectores de la aplicación."""
        return self.dictionary_repository.get_languages()

    def add_word(self, source_text: str, target_text: str, source_language: str, target_language: str):
        """Agregar un par lingüístico validando datos, dirección y duplicados."""
        values = (source_text, target_text, source_language, target_language)
        if any(not isinstance(value, str) or not value.strip() for value in values):
            return {
                "ok": False,
                "status_code": 400,
                "payload": {"error": "Los textos y códigos de idioma son obligatorios"},
            }
        if source_language.strip().casefold() == target_language.strip().casefold():
            return {
                "ok": False,
                "status_code": 400,
                "payload": {"error": "Los idiomas de origen y destino deben ser distintos"},
            }

        existing = self.dictionary_repository.get_word(source_text, source_language, target_language)
        if existing:
            return {
                "ok": False,
                "status_code": 409,
                "payload": {"error": f'El texto "{existing["source_text"]}" ya existe en el diccionario'},
            }

        persisted = self.dictionary_repository.add_word(
            source_text, target_text, source_language, target_language
        )
        if not persisted:
            return {
                "ok": False,
                "status_code": 503,
                "payload": {"error": "No se pudo guardar el par lingüístico"},
            }
        return {
            "ok": True,
            "status_code": 200,
            "payload": {
                "status": "success",
                "message": "Par lingüístico agregado exitosamente al diccionario",
            },
        }

    def process_feedback(
        self,
        original_text: str,
        corrected_translation: str,
        source_lang: str,
        target_lang: str,
    ) -> bool:
        """
        Persistir una corrección en la dirección seleccionada.

        Las correcciones se comportan como un upsert: actualizan el par cuando
        ya existe y crean uno nuevo cuando no existe. El booleano de salida
        permite que el controlador diferencie una operación persistida de una
        solicitud inválida o de un fallo de base de datos.
        """
        values = (original_text, corrected_translation, source_lang, target_lang)
        if any(not isinstance(value, str) or not value.strip() for value in values):
            return False
        if source_lang.strip().casefold() == target_lang.strip().casefold():
            return False

        existing = self.dictionary_repository.get_word(original_text, source_lang, target_lang)
        if existing:
            return self.dictionary_repository.update_word(
                original_text,
                source_lang,
                target_lang,
                target_text=corrected_translation,
            )
        return self.dictionary_repository.add_word(
            original_text,
            corrected_translation,
            source_lang,
            target_lang,
        )
