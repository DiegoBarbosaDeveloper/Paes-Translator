"""
Configuración central del ORM y sesión de base de datos.

Este módulo encapsula la inicialización de SQLAlchemy para desacoplar
la aplicación del motor de base de datos concreto.
"""

from __future__ import annotations

import os
import re

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import scoped_session, sessionmaker


DEFAULT_DATABASE_URL = "postgresql+psycopg://admin:admin123@localhost:5432/postgres"

_engine = None
_session_factory = None


def get_database_url() -> str:
    """Obtener URL de conexión desde variables de entorno o valor por defecto."""
    url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    # Normalizar alias comunes al driver psycopg v3 usado por el proyecto.
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://") and "+psycopg" not in url:
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def get_engine():
    """Retornar la instancia singleton del engine SQLAlchemy."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            get_database_url(),
            pool_pre_ping=True,
            future=True,
        )
    return _engine


def ensure_database_exists() -> None:
    """
    Crear la base configurada cuando el servidor la conoce pero aún no existe.

    ``Base.metadata.create_all`` solo crea tablas dentro de una base ya
    existente; no puede solucionar un error de MySQL del tipo ``Unknown
    database``. Para ese caso se abre una conexión temporal al servidor sin
    seleccionar base y se ejecuta ``CREATE DATABASE IF NOT EXISTS``. La
    operación solo se aplica a dialectos MySQL/MariaDB y nunca intenta crear
    usuarios ni conceder permisos, porque esas tareas pertenecen al
    administrador de la instancia.
    """
    url = make_url(get_database_url())
    if url.get_backend_name() not in {"mysql", "mariadb"} or not url.database:
        return

    if not re.fullmatch(r"[A-Za-z0-9_$-]+", url.database):
        raise ValueError("El nombre de la base de datos contiene caracteres no permitidos")

    server_engine = create_engine(url.set(database=None), pool_pre_ping=True, future=True)
    try:
        with server_engine.begin() as connection:
            connection.execute(text(f"CREATE DATABASE IF NOT EXISTS `{url.database}`"))
    finally:
        server_engine.dispose()


def reset_engine() -> None:
    """Descartar el engine/sesión singleton (útil en scripts y tests)."""
    global _engine, _session_factory
    if _engine is not None:
        try:
            _engine.dispose()
        except Exception:
            pass
    _engine = None
    _session_factory = None


def get_session_factory():
    """Crear y retornar un factory de sesiones thread-safe."""
    global _session_factory
    if _session_factory is None:
        _session_factory = scoped_session(
            sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, future=True)
        )
    return _session_factory


def get_session():
    """Crear una nueva sesión de base de datos."""
    return get_session_factory()()


def remove_session() -> None:
    """Limpiar la sesión del contexto actual (por ejemplo, al finalizar request)."""
    session_factory = get_session_factory()
    session_factory.remove()
