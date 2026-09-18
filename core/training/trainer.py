"""Superficie de entrenamiento genérica para pares multilingües."""

from __future__ import annotations

import json
import os
import threading

from core.training.dataset_builder import build_and_save
from core.training.training_config import (
    TRAINING_LOG_PATH,
    TRAINING_STATUS_PATH,
    TRAINING_STOP_FLAG,
)

_training_thread = None


def _write_status(status):
    os.makedirs(os.path.dirname(TRAINING_STATUS_PATH), exist_ok=True)
    with open(TRAINING_STATUS_PATH, "w", encoding="utf-8") as file:
        json.dump(status, file, ensure_ascii=False, indent=2)


def _append_log(message):
    os.makedirs(os.path.dirname(TRAINING_LOG_PATH), exist_ok=True)
    with open(TRAINING_LOG_PATH, "a", encoding="utf-8") as file:
        file.write(message + "\n")


def read_training_status():
    try:
        with open(TRAINING_STATUS_PATH, encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"state": "idle", "message": "Sin entrenamiento activo"}


def is_training_running():
    return _training_thread is not None and _training_thread.is_alive()


def stop_training():
    os.makedirs(os.path.dirname(TRAINING_STOP_FLAG), exist_ok=True)
    open(TRAINING_STOP_FLAG, "w", encoding="utf-8").close()
    return {"success": True, "message": "Solicitud de detención recibida."}


def _run_training():
    _write_status({"state": "building_dataset", "message": "Construyendo dataset multilingüe."})
    try:
        result = build_and_save()
        _write_status({"state": "ready", "message": "Dataset multilingüe listo.", "metadata": result["metadata"]})
        _append_log("Dataset multilingüe reconstruido.")
    except Exception as error:
        _write_status({"state": "error", "message": str(error)})


def start_training():
    global _training_thread
    if is_training_running():
        return {"success": False, "message": "Ya existe un proceso en ejecución."}
    _training_thread = threading.Thread(target=_run_training, daemon=True)
    _training_thread.start()
    return {"success": True, "message": "Entrenamiento iniciado."}


def get_training_log(max_lines=50):
    try:
        with open(TRAINING_LOG_PATH, encoding="utf-8") as file:
            return file.readlines()[-max_lines:]
    except FileNotFoundError:
        return []