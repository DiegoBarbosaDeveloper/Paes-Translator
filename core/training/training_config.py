"""
Configuración del pipeline de fine-tuning del modelo NLLB-200.
Ajustada para ser viable en CPU con pocos datos.
"""

import os

# ─── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_BASE_PATH      = os.path.join(BASE_DIR, "models", "nllb-200-distilled-600M")
MODEL_FINETUNED_PATH = os.path.join(BASE_DIR, "models", "nllb-finetuned")
TRAINING_PAIRS_PATH  = os.path.join(BASE_DIR, "data", "training_pairs.json")
TRAINING_STATUS_PATH = os.path.join(BASE_DIR, "data", "training_status.json")
DOWNLOAD_STATUS_PATH = os.path.join(BASE_DIR, "data", "download_status.json")
TRAINING_STOP_FLAG   = os.path.join(BASE_DIR, "data", "training_stop.flag")
TRAINING_LOG_PATH    = os.path.join(BASE_DIR, "data", "training_log.txt")

# ─── Identificadores de idioma NLLB ──────────────────────────────────────────
# ─── Hiperparámetros de entrenamiento ────────────────────────────────────────
TRAINING_CONFIG = {
    # Épocas — con pocos datos, 5-10 épocas son suficientes
    "num_epochs": 5,

    # Batch pequeño para funcionar en CPU con poca RAM
    "batch_size": 2,

    # Acumulación de gradientes → batch efectivo = batch_size * grad_accumulation
    "gradient_accumulation_steps": 4,

    # Tasa de aprendizaje baja para fine-tuning (no pre-entrenamiento)
    "learning_rate": 3e-5,

    # Pasos de warmup lineal antes de alcanzar el LR máximo
    "warmup_steps": 20,

    # Longitud máxima de secuencias (tokens)
    "max_source_length": 128,
    "max_target_length": 128,

    # Guardar checkpoint cada N pasos
    "save_steps": 100,

    # Semilla para reproducibilidad
    "seed": 42,

    # Máxima norma del gradiente (clip) para estabilidad
    "max_grad_norm": 1.0,

    # Peso de decay del optimizador
    "weight_decay": 0.01,
}

# ─── Nombre del modelo en HuggingFace Hub ────────────────────────────────────
HUGGINGFACE_MODEL_ID = "facebook/nllb-200-distilled-600M"
