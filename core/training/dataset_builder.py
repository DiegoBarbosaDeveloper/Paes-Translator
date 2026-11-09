"""Construcción genérica de datasets desde ``language_pair``."""

from __future__ import annotations

import json
import os
import random

from database import get_db
from core.training.training_config import TRAINING_PAIRS_PATH

KNOWN_LANGUAGE_CODES = {"es", "piapoco", "achagua"}


def load_pairs() -> list[dict]:
    """Cargar pares lingüísticos desde la base de datos."""
    return get_db().get_training_pairs()


def build_training_pairs(pairs: list[dict]) -> list[dict]:
    """Normalizar pares y agregar su dirección inversa."""
    result = []
    for pair in pairs:
        source = pair.get("source", "").strip()
        target = pair.get("target", "").strip()
        source_lang = pair.get("source_lang", "").strip().casefold()
        target_lang = pair.get("target_lang", "").strip().casefold()
        if not source or not target or not source_lang or not target_lang:
            continue
        if source_lang not in KNOWN_LANGUAGE_CODES or target_lang not in KNOWN_LANGUAGE_CODES:
            continue
        result.append({"source": source, "target": target, "source_lang": source_lang, "target_lang": target_lang, "type": "direct"})
        result.append({"source": target, "target": source, "source_lang": target_lang, "target_lang": source_lang, "type": "inverse"})
    return result


def split_dataset(pairs: list[dict], train_ratio: float = 0.85):
    random.seed(42)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    split_index = max(1, int(len(shuffled) * train_ratio))
    return shuffled[:split_index], shuffled[split_index:]


def _count_by_target(pairs: list[dict]) -> dict[str, int]:
    """Contar pares originales por idioma destino (sin direcciones inversas)."""
    counts = {"piapoco": 0, "achagua": 0}
    for pair in pairs:
        target_lang = pair.get("target_lang", "").strip().casefold()
        if target_lang in counts:
            counts[target_lang] += 1
    return counts


def build_and_save():
    raw_pairs = load_pairs()
    counts = _count_by_target(raw_pairs)
    pairs = build_training_pairs(raw_pairs)
    train, validation = split_dataset(pairs)
    output = {
        "metadata": {
            "total_pairs": len(pairs),
            "train_pairs": len(train),
            "val_pairs": len(validation),
            "dictionary_entries": len(raw_pairs),
            "piapoco_pairs": counts["piapoco"],
            "achagua_pairs": counts["achagua"],
            "version": "2.1",
        },
        "train": train,
        "validation": validation,
    }
    os.makedirs(os.path.dirname(TRAINING_PAIRS_PATH), exist_ok=True)
    with open(TRAINING_PAIRS_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    return output