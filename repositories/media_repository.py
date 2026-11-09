"""Repositorio de archivos multimedia almacenados en el filesystem."""

from __future__ import annotations

import os


class MediaRepository:
    """Mantiene media fuera de la BD porque el esquema no define una tabla para ella."""

    def __init__(self, session):
        self.root = os.path.join("static", "uploads")

    def get_media(self, media_type: str, tag: str | None = None) -> list:
        """Listar archivos existentes; ``tag`` se conserva por compatibilidad."""
        folder_name = "images" if media_type == "image" else "audio"
        folder = os.path.join(self.root, folder_name)
        if not os.path.isdir(folder):
            return []
        return [
            {
                "filename": filename,
                "url": f"/static/uploads/{folder_name}/{filename}",
            }
            for filename in sorted(os.listdir(folder))
            if os.path.isfile(os.path.join(folder, filename))
        ]

    def add_media(self, media_type: str, filename: str, url: str, tag: str = "") -> str:
        """Confirmar el nombre del archivo ya guardado por el servicio."""
        return filename
