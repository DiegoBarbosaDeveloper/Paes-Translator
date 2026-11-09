"""
Definición de entidades del dominio con SQLAlchemy ORM.

Las entidades representan tablas de forma portable, evitando SQL específico
del motor para facilitar migraciones de base de datos.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BIGINT,
    VARCHAR,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Integer,
    inspect,
    text,
)
from sqlalchemy import types as sa_types
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from infrastructure.db import ensure_database_exists, get_engine


class Base(DeclarativeBase):
    """Clase base para todas las entidades ORM."""


def _pk_type():
    """Tipo de PK portable: BIGINT en PostgreSQL y INTEGER en SQLite."""
    return BIGINT().with_variant(Integer(), "sqlite")


class Dataset(Base):
    __tablename__ = "dataset"

    id: Mapped[int] = mapped_column(_pk_type(), Identity(), primary_key=True)
    name: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    version: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    description: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    status: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class Language(Base):
    __tablename__ = "languages"

    id: Mapped[int] = mapped_column(_pk_type(), Identity(), primary_key=True)
    code: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    name: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    active: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)


class TranslationDirection(Base):
    __tablename__ = "translation_direction"

    id: Mapped[int] = mapped_column(_pk_type(), Identity(), primary_key=True)
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

    id: Mapped[int] = mapped_column(_pk_type(), Identity(), primary_key=True)
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
    validated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ModelVersion(Base):
    __tablename__ = "model_version"

    id: Mapped[int] = mapped_column(_pk_type(), Identity(), primary_key=True)
    name: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    version: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    base_model: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    dataset_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("dataset.id", name="fk_model_version_dataset_id"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class Evaluation(Base):
    __tablename__ = "evaluation"

    id: Mapped[int] = mapped_column(_pk_type(), Identity(), primary_key=True)
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
    bleu: Mapped[float] = mapped_column(Float, nullable=False)
    chr_f: Mapped[float] = mapped_column(Float, nullable=False)
    comet: Mapped[float] = mapped_column(Float, nullable=False)
    evaluation_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class HumanEvaluation(Base):
    __tablename__ = "human_evaluation"

    id: Mapped[int] = mapped_column(_pk_type(), Identity(), primary_key=True)
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
    adequacy_score: Mapped[float] = mapped_column(Float, nullable=False)
    fluency_score: Mapped[float] = mapped_column(Float, nullable=False)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    evaluation_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class SchemaValidationError(RuntimeError):
    """La base de datos no coincide exactamente con el modelo ORM."""


def _type_signature(column_type):
    """
    Reducir tipos SQLAlchemy a una firma comparable entre dialectos.

    PostgreSQL refleja ``DateTime`` como ``TIMESTAMP`` y ``Float`` como
    ``DOUBLE PRECISION``. Comparar nombres de clase haría fallar la validación
    aunque el contrato ORM sea correcto, por lo que se compara la familia de
    tipo y, cuando aplica, la longitud/precisión.
    """
    if isinstance(column_type, sa_types.Boolean):
        return ("boolean",)
    if isinstance(column_type, sa_types.Float):
        return ("float",)
    if isinstance(column_type, sa_types.Numeric):
        return (
            "numeric",
            getattr(column_type, "precision", None),
            getattr(column_type, "scale", None),
        )
    if isinstance(column_type, sa_types.Integer):
        return ("integer",)
    if isinstance(column_type, sa_types.DateTime):
        return ("datetime",)
    if isinstance(column_type, sa_types.String):
        return ("string", getattr(column_type, "length", None))
    return (type(column_type).__name__.lower(),)


def validate_schema() -> None:
    """
    Comparar la base real con el contrato ORM sin modificarla.

    Esta validación es deliberadamente estricta: detecta tablas sobrantes o
    ausentes, columnas distintas, tipos, nulabilidad, claves primarias y
    restricciones foráneas. El resultado evita que una versión de la
    aplicación opere sobre una base parcialmente migrada y deja la
    administración del esquema alineada con ``db/schema.postgres.sql``.
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
                f"psql -U <usuario> -d {database_name} -f db/schema.postgres.sql"
            )
        raise SchemaValidationError(
            "La base de datos no coincide con el esquema ORM: " + "; ".join(errors)
        )


def _ensure_postgres_indexes() -> None:
    """
    Crear índices que ``create_all`` no añade a tablas ya existentes.

    El índice único es parte del contrato de integridad del diccionario: evita
    dos traducciones para la misma fuente dentro de una dirección, ignorando
    mayúsculas y espacios exteriores. Se aplica solo en PostgreSQL porque usa
    una expresión portable entre las instalaciones soportadas.
    """
    if get_engine().dialect.name != "postgresql":
        return
    with get_engine().begin() as connection:
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "uq_language_pair_direction_source_ci "
                "ON language_pair ("
                "source_language_id, target_language_id, "
                "lower(btrim(source_text))"
                ")"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_language_pair_direction "
                "ON language_pair (source_language_id, target_language_id)"
            )
        )


def init_db() -> None:
    """
    Provisionar y validar el esquema administrado por el ORM.

    El ORM crea la base (PostgreSQL, MySQL/MariaDB o SQLite) si falta y crea las
    tablas declaradas en ``Base.metadata``. Después se ejecuta la comparación
    estricta para detectar diferencias de tipos, nulabilidad o relaciones. No se
    hacen migraciones destructivas: una tabla o columna incompatible provoca un
    error explícito para que pueda corregirse sin perder datos.
    """
    ensure_database_exists()
    try:
        Base.metadata.create_all(bind=get_engine())
        _ensure_postgres_indexes()
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


def seed_defaults() -> None:
    """
    Insertar idiomas, datasets y direcciones base cuando faltan.

    Se usan identificadores estables para que la importación de datos y el
    repositorio puedan asociar cada par con el corpus correcto:
    ``dataset 1 = piapoco`` y ``dataset 2 = achagua``.
    """
    from infrastructure.db import get_session

    session = get_session()
    try:
        for language_id, code, name in (
            (1, "es", "Español"),
            (2, "piapoco", "Piapoco"),
            (3, "achagua", "Achagua"),
        ):
            if session.get(Language, language_id) is None:
                session.add(
                    Language(id=language_id, code=code, name=name, active="true")
                )

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        for dataset_id, name, description in (
            (1, "piapoco", "Corpus piapoco"),
            (2, "achagua", "Corpus achagua"),
        ):
            existing = session.get(Dataset, dataset_id)
            if existing is None:
                session.add(
                    Dataset(
                        id=dataset_id,
                        name=name,
                        version="1",
                        description=description,
                        status="active",
                        created_at=now,
                    )
                )
            elif existing.name == "default":
                existing.name = name
                existing.description = description

        for direction_id, source_id, target_id in (
            (1, 1, 2),
            (2, 2, 1),
            (3, 1, 3),
            (4, 3, 1),
        ):
            if session.get(TranslationDirection, direction_id) is None:
                session.add(
                    TranslationDirection(
                        id=direction_id,
                        source_language_id=source_id,
                        target_language_id=target_id,
                        active=1,
                    )
                )

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
