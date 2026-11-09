"""
Repositorio de pares lingüísticos respaldado por ``language_pair``.

Este módulo traduce las operaciones de alto nivel de la aplicación a
consultas SQLAlchemy. La tabla guarda identificadores de idiomas, no nombres,
por lo que cada operación resuelve los códigos contra ``languages``. Así se
pueden añadir idiomas desde la base de datos sin cambiar el código de negocio.

El esquema usa identificadores obligatorios sin autoincremento. Las escrituras
validan los datos, usan el siguiente identificador disponible y convierten los
errores de integridad en resultados controlados para que una solicitud no deje
la sesión ORM en estado inválido.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from entities.models import Dataset, Language, LanguagePair


class DictionaryRepository:
    """Consulta y modifica pares para cualquier idioma registrado."""

    def __init__(self, session):
        self.session = session

    @staticmethod
    def _normalize(value: str) -> str:
        """Normalizar valores externos antes de usarlos en consultas."""
        if not isinstance(value, str):
            return ""
        return value.strip().casefold()

    def _language_id(self, language: str) -> int | None:
        normalized = self._normalize(language)
        if not normalized:
            return None
        return self.session.execute(
            select(Language.id).where(
                (func.lower(Language.code) == normalized)
                | (func.lower(Language.name) == normalized),
                func.lower(Language.active).notin_(["0", "false", "inactive", "inactivo"]),
            )
        ).scalar_one_or_none()

    def _dataset_id(self, target_language_id: int | None = None) -> int | None:
        """
        Resolver el dataset asociado al idioma destino.

        La importación crea un dataset por idioma (``piapoco`` y ``achagua``)
        para que las nuevas palabras queden en el corpus correcto. Si el
        dataset específico no existe se usa el primer dataset activo como
        compatibilidad con instalaciones anteriores.
        """
        query = select(Dataset.id).where(
            func.lower(Dataset.status).in_(["active", "activo"])
        )
        if target_language_id is not None:
            language = self.session.get(Language, target_language_id)
            code = (language.code or "").strip().casefold() if language else ""
            if code:
                dataset_id = self.session.execute(
                    query.where(func.lower(Dataset.name) == code)
                    .order_by(Dataset.id)
                    .limit(1)
                ).scalar_one_or_none()
                if dataset_id is not None:
                    return dataset_id
        return self.session.execute(
            query.order_by(Dataset.id).limit(1)
        ).scalar_one_or_none()

    def _query_pairs(self, source_language: str, target_language: str):
        source_id = self._language_id(source_language)
        target_id = self._language_id(target_language)
        if source_id is None or target_id is None:
            return None
        return select(LanguagePair).where(
            LanguagePair.source_language_id == source_id,
            LanguagePair.target_language_id == target_id,
        )

    def _find_pair(self, source_text: str, source_language: str, target_language: str):
        """Resolver un par concreto después de validar su dirección."""
        normalized_text = self._normalize(source_text)
        query = self._query_pairs(source_language, target_language)
        if not normalized_text or query is None:
            return None
        return self.session.execute(
            query.where(func.lower(LanguagePair.source_text) == normalized_text)
        ).scalar_one_or_none()

    def get_languages(self) -> list[dict]:
        """Retornar el catálogo de idiomas disponible para la interfaz."""
        rows = self.session.execute(
            select(Language)
            .where(func.lower(Language.active).notin_(
                ["0", "false", "inactive", "inactivo"]
            ))
            .order_by(Language.name)
        ).scalars().all()
        return [{"id": row.id, "code": row.code, "name": row.name, "active": row.active} for row in rows]

    def get_dictionary(self, source_language: str, target_language: str) -> dict:
        """Retornar pares de texto para cualquier dirección registrada."""
        query = self._query_pairs(source_language, target_language)
        if query is None:
            return {}
        rows = self.session.execute(query.order_by(LanguagePair.id)).scalars().all()
        return {
            row.source_text: {
                "target_text": row.target_text,
                "validation_status": row.validation_status,
            }
            for row in rows
        }

    def get_training_pairs(self) -> list[dict]:
        """Retornar todos los pares con sus códigos de idioma."""
        source = Language.__table__.alias("source_language")
        target = Language.__table__.alias("target_language")
        rows = self.session.execute(
            select(
                LanguagePair.source_text,
                LanguagePair.target_text,
                source.c.code.label("source_lang"),
                target.c.code.label("target_lang"),
            )
            .join(source, source.c.id == LanguagePair.source_language_id)
            .join(target, target.c.id == LanguagePair.target_language_id)
            .order_by(LanguagePair.id)
        ).all()
        return [
            {
                "source": row.source_text,
                "target": row.target_text,
                "source_lang": row.source_lang,
                "target_lang": row.target_lang,
            }
            for row in rows
        ]

    def get_word(self, source_text: str, source_language: str, target_language: str):
        """Buscar texto exacto ignorando mayúsculas/minúsculas."""
        row = self._find_pair(source_text, source_language, target_language)
        if row is None:
            return None
        return {
            "source_text": row.source_text,
            "target_text": row.target_text,
            "validation_status": row.validation_status,
        }

    def count_entries(self, source_language: str, target_language: str) -> int:
        """Contar pares de una dirección."""
        query = self._query_pairs(source_language, target_language)
        if query is None:
            return 0
        return self.session.execute(select(func.count()).select_from(query.subquery())).scalar_one()

    def add_word(self, source_text: str, target_text: str, source_language: str, target_language: str, validation_status: str | None = None) -> bool:
        """Insertar un par usando un dataset y dos idiomas ya existentes."""
        source_text = source_text.strip() if isinstance(source_text, str) else ""
        target_text = target_text.strip() if isinstance(target_text, str) else ""
        if not source_text or not target_text:
            return False
        source_id = self._language_id(source_language)
        target_id = self._language_id(target_language)
        dataset_id = self._dataset_id(target_id)
        if source_id is None or target_id is None or dataset_id is None:
            return False

        entity = LanguagePair(
            dataset_id=dataset_id,
            source_language_id=source_id,
            target_language_id=target_id,
            source_text=source_text,
            target_text=target_text,
            validation_status=validation_status,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        self.session.add(entity)
        try:
            self.session.commit()
            return True
        except (IntegrityError, SQLAlchemyError):
            self.session.rollback()
            return False

    def update_word(self, source_text: str, source_language: str, target_language: str, target_text: str | None = None, validation_status: str | None = None) -> bool:
        """Actualizar un par existente."""
        row = self._find_pair(source_text, source_language, target_language)
        if row is None:
            return False
        if target_text is not None:
            row.target_text = target_text
        if validation_status is not None:
            row.validation_status = validation_status
        try:
            self.session.commit()
            return True
        except SQLAlchemyError:
            self.session.rollback()
            return False

    def upsert_word(self, source_text: str, target_text: str, source_language: str, target_language: str, validation_status: str | None = None) -> None:
        """Insertar o actualizar un par."""
        if self.get_word(source_text, source_language, target_language) is None:
            self.add_word(source_text, target_text, source_language, target_language, validation_status)
            return
        self.update_word(source_text, source_language, target_language, target_text, validation_status)

    def delete_word(self, source_text: str, source_language: str, target_language: str) -> bool:
        """Eliminar un par."""
        row = self._find_pair(source_text, source_language, target_language)
        if row is None:
            return False
        self.session.delete(row)
        try:
            self.session.commit()
            return True
        except SQLAlchemyError:
            self.session.rollback()
            return False
