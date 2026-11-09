"""Punto de entrada WSGI para servidores de producción (Gunicorn)."""

from app import create_app

app = create_app()
