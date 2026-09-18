"""
Configuración central de la aplicación por ambientes.

Este módulo permite ejecutar la app en desarrollo, pruebas o producción
sin cambiar código, únicamente mediante variables de entorno.
"""

from __future__ import annotations

import os


class BaseConfig:
    """Configuración base compartida por todos los ambientes."""

    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-me")
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "mysql+pymysql://admin:admin123@localhost:3306/nasa_yuwe_translator",
    )
    JSON_SORT_KEYS = False


class DevelopmentConfig(BaseConfig):
    """Configuración para desarrollo local."""

    DEBUG = True
    TESTING = False


class TestingConfig(BaseConfig):
    """Configuración para pruebas automatizadas."""

    DEBUG = False
    TESTING = True
    DATABASE_URL = os.getenv("TEST_DATABASE_URL", "sqlite+pysqlite:///:memory:")


class ProductionConfig(BaseConfig):
    """Configuración para despliegues productivos."""

    DEBUG = False
    TESTING = False


CONFIG_BY_ENV = {
    "dev": DevelopmentConfig,
    "development": DevelopmentConfig,
    "test": TestingConfig,
    "testing": TestingConfig,
    "prod": ProductionConfig,
    "production": ProductionConfig,
}


def get_config(environment: str | None = None):
    """Resolver clase de configuración según el ambiente solicitado."""
    env = (environment or os.getenv("APP_ENV", "dev")).lower()
    return CONFIG_BY_ENV.get(env, DevelopmentConfig)
