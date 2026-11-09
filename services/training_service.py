"""Servicio para operaciones de entrenamiento y descarga del modelo NLLB."""

from __future__ import annotations

from core.training.download_model import (
    model_weights_exist,
    read_download_status,
    start_download_thread,
)
from core.training.trainer import (
    get_training_log,
    is_training_running,
    read_training_status,
    start_training,
    stop_training,
)


class TrainingService:
    """Encapsula la lógica de negocio para entrenamiento y monitoreo."""

    def __init__(self, translation_service):
        self.translation_service = translation_service

    def start(self):
        return start_training()

    def stop(self):
        return stop_training()

    def status(self):
        status = read_training_status()
        status["is_running"] = is_training_running()
        return {"success": True, "status": status}

    def log(self, lines: int):
        log_lines = get_training_log(max_lines=lines)
        return {"success": True, "log": log_lines}

    def start_model_download(self):
        if model_weights_exist():
            return {
                "success": True,
                "message": "Los pesos del modelo ya están disponibles.",
                "already_downloaded": True,
            }
        start_download_thread()
        return {
            "success": True,
            "message": (
                "Descarga iniciada en background. "
                "Use /api/training/download-status para verificar el progreso."
            ),
            "already_downloaded": False,
        }

    def model_download_status(self):
        status = read_download_status()
        status["model_ready"] = model_weights_exist()
        return {"success": True, "status": status}

    def reload_model(self):
        self.translation_service.reset_model()
        info = self.translation_service.get_model_info()["model_info"]
        return {
            "success": True,
            "message": "Modelo recargado exitosamente.",
            "model_info": info,
        }

    def rebuild_dataset(self):
        from core.training.dataset_builder import build_and_save

        result = build_and_save()
        if not result:
            return {"success": False, "message": "No se pudo reconstruir el dataset."}, 500

        metadata = result["metadata"]
        return {
            "success": True,
            "message": "Dataset reconstruido exitosamente.",
            "total_pairs": metadata["total_pairs"],
            "train_pairs": metadata["train_pairs"],
            "val_pairs": metadata["val_pairs"],
            "dict_entries": metadata["dictionary_entries"],
            "piapoco_pairs": metadata.get("piapoco_pairs", 0),
            "achagua_pairs": metadata.get("achagua_pairs", 0),
        }, 200
