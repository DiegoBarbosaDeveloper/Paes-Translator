"""
Punto de entrada Flask con arquitectura en capas.

Capas aplicadas:
1. Controladores: endpoints HTTP (controllers)
2. Servicios: lógica de negocio (services)
3. Entidades y repositorios: persistencia con ORM (entities/repositories)
"""

from __future__ import annotations

import os

from flask import Flask

from config import get_config
from controllers.api_controller import api_bp
from entities.models import init_db, seed_defaults
from infrastructure.db import get_session_factory, remove_session
from services.container import build_services


def create_app(environment: str | None = None) -> Flask:
    """Crear y configurar instancia de Flask app."""
    app = Flask(__name__)

    config_class = get_config(environment)
    app.config.from_object(config_class)

    # Sincronizar variable para el subsistema ORM desacoplado.
    os.environ["DATABASE_URL"] = app.config["DATABASE_URL"]

    # El ORM provisiona la base/tablas ausentes y valida el resultado.
    init_db()

    # Sembrar idiomas por defecto cuando el catálogo está vacío.
    seed_defaults()

    # Construir contenedor de servicios con una sesión compartida.
    session = get_session_factory()
    services = build_services(session)
    services["media"].ensure_upload_dirs()

    # Registrar contenedor para resolución en controladores.
    app.config["services"] = services

    # Registrar blueprint principal de la API y vistas.
    app.register_blueprint(api_bp)

    @app.teardown_appcontext
    def cleanup_session(_exception=None):
        """Liberar sesión ORM al finalizar cada ciclo de request."""
        remove_session()

    return app


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    flask_app = create_app()
    flask_app.run(debug=flask_app.config.get("DEBUG", False))
