"""
Definición de entidades del dominio con SQLAlchemy ORM.

Las entidades representan tablas de forma portable, evitando SQL específico
del motor para facilitar migraciones de base de datos.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BIGINT, DATETIME, FLOAT, VARCHAR, ForeignKey, inspect
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from infrastructure.db import ensure_database_exists, get_engine


class Base(DeclarativeBase):
    """Clase base para todas las entidades ORM."""


class Dataset(Base):
    __tablename__ = "dataset"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    version: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    description: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    status: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DATETIME, nullable=False)


class Language(Base):
    __tablename__ = "languages"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=False)
    code: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    name: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    active: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)


class TranslationDirection(Base):
    __tablename__ = "translation_direction"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=False)
    source_language_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("languages.id", name="fk_translation_direction_source_language_id"),
        nullable=False,
    )
    target_language_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("languages.id", name="fk_translation_direction_target_language_id"),
        nullable=False,
    )
    active: Mapped[int] = mapped_column(BIGINT, nullable=False)


class LanguagePair(Base):
    __tablename__ = "language_pair"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=False)
    dataset_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("dataset.id", name="fk_language_pair_dataset_id"),
        nullable=False,
    )
    source_language_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("languages.id", name="fk_language_pair_source_language_id"),
        nullable=False,
    )
    target_language_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("languages.id", name="fk_language_pair_target_language_id"),
        nullable=False,
    )
    source_text: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    target_text: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    validation_status: Mapped[str | None] = mapped_column(VARCHAR(255), nullable=True)
    validated_at: Mapped[datetime | None] = mapped_column(DATETIME, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DATETIME, nullable=False)


class ModelVersion(Base):
    __tablename__ = "model_version"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    version: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    base_model: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    dataset_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("dataset.id", name="fk_model_version_dataset_id"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DATETIME, nullable=False)


class Evaluation(Base):
    __tablename__ = "evaluation"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=False)
    model_version_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("model_version.id", name="fk_evaluation_model_version_id"),
        nullable=False,
    )
    direction_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("translation_direction.id", name="fk_evaluation_direction_id"),
        nullable=False,
    )
    dataset_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("dataset.id", name="fk_evaluation_dataset_id"),
        nullable=False,
    )
    bleu: Mapped[float] = mapped_column(FLOAT, nullable=False)
    chr_f: Mapped[float] = mapped_column(FLOAT, nullable=False)
    comet: Mapped[float] = mapped_column(FLOAT, nullable=False)
    evaluation_date: Mapped[datetime] = mapped_column(DATETIME, nullable=False)


class HumanEvaluation(Base):
    __tablename__ = "human_evaluation"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=False)
    model_version_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("model_version.id", name="fk_human_evaluation_model_version_id"),
        nullable=False,
    )
    direction_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("translation_direction.id", name="fk_human_evaluation_direction_id"),
        nullable=False,
    )
    sample_direction: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    adequacy_score: Mapped[float] = mapped_column(FLOAT, nullable=False)
    fluency_score: Mapped[float] = mapped_column(FLOAT, nullable=False)
    overall_score: Mapped[float] = mapped_column(FLOAT, nullable=False)
    evaluation_date: Mapped[datetime] = mapped_column(DATETIME, nullable=False)


class SchemaValidationError(RuntimeError):
    """La base de datos no coincide exactamente con el modelo ORM."""


def _type_signature(column_type):
    """Reducir tipos SQLAlchemy a sus atributos estructurales comparables."""
    return (
        type(column_type).__name__.lower(),
        getattr(column_type, "length", None),
        getattr(column_type, "precision", None),
        getattr(column_type, "scale", None),
    )


def validate_schema() -> None:
    """
    Comparar la base real con el contrato ORM sin modificarla.

    Esta validación es deliberadamente estricta: detecta tablas sobrantes o
    ausentes, columnas distintas, tipos, nulabilidad, claves primarias y
    restricciones foráneas. El resultado evita que una versión de la
    aplicación opere sobre una base parcialmente migrada y, al no llamar a
    ``create_all``, deja la administración del esquema exclusivamente en el
    proceso de despliegue que ejecuta ``schema.sql``.
    """
    engine = get_engine()
    inspector = inspect(engine)
    actual_tables = set(inspector.get_table_names())
    expected_tables = set(Base.metadata.tables)
    errors = []

    if actual_tables != expected_tables:
        errors.append(
            f"tablas esperadas={sorted(expected_tables)}, existentes={sorted(actual_tables)}"
        )

    for table_name, table in Base.metadata.tables.items():
        if table_name not in actual_tables:
            continue

        actual_columns = {column["name"]: column for column in inspector.get_columns(table_name)}
        expected_columns = {column.name: column for column in table.columns}
        if set(actual_columns) != set(expected_columns):
            errors.append(f"{table_name}: columnas no coinciden")
            continue

        for column_name, expected in expected_columns.items():
            actual = actual_columns[column_name]
            if (
                _type_signature(actual["type"]) != _type_signature(expected.type)
                or actual["nullable"] != expected.nullable
            ):
                errors.append(
                    f"{table_name}.{column_name}: tipo/nulabilidad no coincide "
                    f"(ORM={_type_signature(expected.type)}/{expected.nullable}, "
                    f"DB={_type_signature(actual['type'])}/{actual['nullable']})"
                )

        actual_pk = set(inspector.get_pk_constraint(table_name).get("constrained_columns", []))
        expected_pk = {column.name for column in table.primary_key.columns}
        if actual_pk != expected_pk:
            errors.append(f"{table_name}: clave primaria no coincide")

        actual_fks = {
            (
                fk["name"],
                fk["constrained_columns"][0],
                fk["referred_table"],
                fk["referred_columns"][0],
            )
            for fk in inspector.get_foreign_keys(table_name)
        }
        expected_fks = {
            (fk.constraint.name, fk.parent.name, fk.column.table.name, fk.column.name)
            for fk in table.foreign_keys
        }
        if actual_fks != expected_fks:
            errors.append(f"{table_name}: claves foráneas no coinciden")

    if errors:
        if not actual_tables:
            database_name = engine.url.database or "<base_de_datos>"
            raise SchemaValidationError(
                f"La base de datos '{database_name}' no contiene tablas. "
                "Ejecuta el esquema oficial antes de iniciar la aplicación: "
                f"mysql -u <usuario> -p {database_name} < schema.sql"
            )
        raise SchemaValidationError(
            "La base de datos no coincide con schema.sql: " + "; ".join(errors)
        )


def init_db() -> None:
    """
    Provisionar y validar el esquema administrado por el ORM.

    El ORM crea la base MySQL si falta y crea las tablas declaradas en
    ``Base.metadata``. Después se ejecuta la comparación estricta para detectar
    diferencias de tipos, nulabilidad o relaciones. No se hacen migraciones
    destructivas: una tabla o columna incompatible provoca un error explícito
    para que pueda corregirse sin perder datos.
    """
    ensure_database_exists()
    try:
        Base.metadata.create_all(bind=get_engine())
    except OperationalError as error:
        original = str(error.orig or error)
        if "REFERENCES command denied" in original or "1142" in original:
            database_name = get_engine().url.database or "<base_de_datos>"
            raise SchemaValidationError(
                "El usuario de la base no tiene el privilegio REFERENCES para "
                f"crear las claves foráneas de '{database_name}'. "
                "Ejecuta con un usuario administrador: "
                f"GRANT REFERENCES ON `{database_name}`.* TO 'admin'@'%'; "
                "FLUSH PRIVILEGES;"
            ) from error
        raise
    validate_schema()
