"""
Fachada de base de datos compatible con la API anterior.

Aunque internamente ahora se usa SQLAlchemy ORM, este módulo mantiene
los mismos métodos públicos (`get_db`, `add_word`, etc.) para no romper
componentes existentes durante la transición arquitectónica.
"""

from __future__ import annotations

import logging

from entities.models import init_db
from infrastructure.db import get_session_factory
from repositories.dictionary_repository import DictionaryRepository
from repositories.media_repository import MediaRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_db_instance = None


def get_db():
    """Obtener instancia singleton de la fachada de base de datos."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


class Database:
    """Fachada orientada a compatibilidad que delega en repositorios ORM."""

    def __init__(self):
        init_db()
        self.session = get_session_factory()
        self.dictionary_repository = DictionaryRepository(self.session)
        self.media_repository = MediaRepository(self.session)
        logger.info("Conexión ORM inicializada")

    def get_languages(self) -> list[dict]:
        return self.dictionary_repository.get_languages()

    def get_dictionary(self, source_language: str, target_language: str) -> dict:
        return self.dictionary_repository.get_dictionary(source_language, target_language)

    def get_training_pairs(self) -> list[dict]:
        return self.dictionary_repository.get_training_pairs()

    def get_word(self, source_text: str, source_language: str, target_language: str):
        return self.dictionary_repository.get_word(source_text, source_language, target_language)

    def count_entries(self, source_language: str, target_language: str) -> int:
        return self.dictionary_repository.count_entries(source_language, target_language)

    def add_word(self, source_text: str, target_text: str, source_language: str, target_language: str, validation_status: str | None = None):
        return self.dictionary_repository.add_word(source_text, target_text, source_language, target_language, validation_status)

    def update_word(
        self,
        source_text: str,
        source_language: str,
        target_language: str,
        target_text: str | None = None,
        validation_status: str | None = None,
    ):
        return self.dictionary_repository.update_word(
            source_text,
            source_language,
            target_language,
            target_text,
            validation_status,
        )

    def upsert_word(
        self,
        source_text: str,
        target_text: str,
        source_language: str,
        target_language: str,
        validation_status: str | None = None,
    ):
        self.dictionary_repository.upsert_word(
            source_text,
            target_text,
            source_language,
            target_language,
            validation_status,
        )

    def delete_word(self, source_text: str, source_language: str, target_language: str):
        return self.dictionary_repository.delete_word(source_text, source_language, target_language)

    def get_media(self, media_type: str, tag: str | None = None):
        return self.media_repository.get_media(media_type, tag)

    def add_media(self, media_type: str, filename: str, url: str, tag: str = ""):
        return self.media_repository.add_media(media_type, filename, url, tag)
