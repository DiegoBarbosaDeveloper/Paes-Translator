"""
Motor de traducción basado en el adaptador NLLB fine-tuneado (LoRA).

El modelo vive en ``NLLBTest1/models/nllb-spa-piapoco-achagua``: es un
adaptador PEFT sobre ``facebook/nllb-200-distilled-600M`` entrenado para
español <-> piapoco/achagua. Cargarlo requiere cuatro pasos que deben
respetar exactamente el entrenamiento:

1. Tokenizer propio del adaptador (incluye los tokens ``piapoco_Latn`` y
   ``achagua_Latn`` añadidos como tokens especiales).
2. Modelo base NLLB con los embeddings redimensionados al vocabulario del
   tokenizer del adaptador.
3. Vectores de los tags nuevos (``tag_vectors.pt``) restaurados en los
   embeddings, porque el adaptador no los guarda.
4. ``PeftModel`` con los pesos LoRA.

El adaptador NLLB es la estrategia principal y prioritaria; el diccionario
persistido actúa como respaldo rápido. Si el adaptador no está disponible o
falla, la aplicación arranca igual y delega en el diccionario, por lo que cada
etapa falla de forma aislada.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

try:
    import torch
except ImportError:  # Dependencia opcional: solo necesaria para el modelo NLLB
    torch = None

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:  # Dependencia opcional: solo necesaria para el modelo NLLB
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None

from database import get_db


class AdvancedTranslationModel:
    """
    Modelo de traducción que combina:
    1. Diccionario personalizado para términos específicos.
    2. Adaptador NLLB fine-tuneado para traducción contextual.
    """

    BASE_DIR = Path(__file__).resolve().parent

    # Rutas configurables por variables de entorno.
    DEFAULT_ADAPTER_DIR = BASE_DIR / "NLLBTest1" / "models" / "nllb-spa-piapoco-achagua"
    DEFAULT_BASE_PATH = BASE_DIR / "models" / "nllb-200-distilled-600M"
    DEFAULT_BASE_MODEL_ID = "facebook/nllb-200-distilled-600M"

    LANGUAGE_CONFIG = "language_config.json"
    TAG_VECTORS = "tag_vectors.pt"
    ADAPTER_CONFIG = "adapter_config.json"

    # El código de idioma que llega desde la base puede ser el código corto, el
    # nombre o el nombre completo del idioma. Se normaliza a un nombre canónico.
    ALIASES = {
        "es": "spanish",
        "spa": "spanish",
        "spanish": "spanish",
        "castellano": "spanish",
        "espanol": "spanish",
        "español": "spanish",
        "pia": "piapoco",
        "piapoco": "piapoco",
        "aca": "achagua",
        "achagua": "achagua",
    }

    def __init__(self, db=None):
        self.db = db or get_db()
        self.model = None
        self.tokenizer = None
        self.dictionary = {}
        self.model_loaded = False
        self.device = torch.device("cpu") if torch is not None else None
        self.tags: dict[str, str] = {}
        self._lock = threading.Lock()

        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Resolver rutas del adaptador y del modelo base.
        self.adapter_dir = Path(
            os.getenv("NLLB_ADAPTER_DIR", str(self.DEFAULT_ADAPTER_DIR))
        )
        self.base_model_path = Path(
            os.getenv("NLLB_BASE_MODEL_PATH", str(self.DEFAULT_BASE_PATH))
        )
        self.base_model_id = os.getenv("NLLB_BASE_MODEL_ID", self.DEFAULT_BASE_MODEL_ID)

        # Inicializar componentes
        self._load_nllb_model()

    def _load_dictionary(self, source_lang, target_lang):
        """Cargar los pares de texto de la dirección solicitada.

        La base solo almacena pares español -> piapoco/achagua. Para las
        direcciones inversas se construye el diccionario a partir de esos
        pares, invirtiendo origen y destino. Las claves se normalizan para que
        la búsqueda no dependa de mayúsculas ni de espacios exteriores.
        """
        try:
            source_lang = source_lang.strip()
            target_lang = target_lang.strip()
            raw = self.db.get_dictionary(source_lang, target_lang)

            # Si no hay pares directos, intentar construirlos a partir de la
            # dirección opuesta ya almacenada (piapoco/achagua -> español).
            if not raw:
                opposite = self.db.get_dictionary(target_lang, source_lang)
                if opposite:
                    raw = {
                        item["target_text"]: {
                            "target_text": key,
                            "validation_status": item.get("validation_status"),
                        }
                        for key, item in opposite.items()
                    }

            self.dictionary = {}
            for source_text, info in (raw or {}).items():
                if not isinstance(info, dict):
                    info = {"target_text": info}
                target_text = info.get("target_text", "")
                if not str(source_text).strip() or not str(target_text).strip():
                    continue
                self.dictionary[str(source_text).strip().casefold()] = {
                    "target_text": str(target_text).strip(),
                    "validation_status": info.get("validation_status"),
                }
            self.logger.info(f"Pares cargados: {len(self.dictionary)}")
        except Exception as e:
            self.logger.error(f"Error cargando pares lingüísticos: {e}")
            self.dictionary = {}

    @classmethod
    def _resolve_language(cls, language):
        """Normalizar el código o nombre de idioma a un nombre canónico."""
        if not isinstance(language, str):
            return None
        key = language.strip().lower()
        if not key:
            return None
        if key in cls.ALIASES:
            return cls.ALIASES[key]

        # Códigos con región (p. ej. "es-ES" o "es_CO").
        base = key.replace("_", "-").split("-")[0]
        if base in cls.ALIASES:
            return cls.ALIASES[base]

        # Coincidencias parciales sobre nombres completos.
        if "piapo" in key:
            return "piapoco"
        if "achagu" in key:
            return "achagua"
        if "espa" in key or "castell" in key or "span" in key:
            return "spanish"
        return None

    @staticmethod
    def _pick_dtype(device):
        """Usar media precisión en GPU y precisión completa en CPU."""
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    def _load_language_tags(self):
        """Leer los tags de idioma almacenados junto al adaptador."""
        path = self.adapter_dir / self.LANGUAGE_CONFIG
        if path.exists():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                tags = saved.get("tags")
                if isinstance(tags, dict) and tags:
                    return dict(tags)
            except (json.JSONDecodeError, OSError) as e:
                self.logger.warning(f"No se pudo leer {path}: {e}")
        return {"spanish": "spa_Latn", "piapoco": "piapoco_Latn", "achagua": "achagua_Latn"}

    def _base_model_source(self):
        """Preferir el modelo base local; si no existe, usar el id de HuggingFace."""
        if self.base_model_path.exists() and any(self.base_model_path.iterdir()):
            return str(self.base_model_path)
        return self.base_model_id

    def _load_tag_vectors(self, base):
        """Restaurar en los embeddings los vectores de los tags nuevos."""
        path = self.adapter_dir / self.TAG_VECTORS
        if not path.exists():
            self.logger.warning(
                "No se encontró '%s'; los tags nuevos conservarán los pesos aleatorios.",
                path,
            )
            return
        vectors = torch.load(path, map_location="cpu", weights_only=True)
        weights = base.get_input_embeddings().weight
        vocab = self.tokenizer.get_vocab()
        with torch.no_grad():
            for tag, vector in vectors.items():
                if tag not in vocab:
                    self.logger.warning("El tag '%s' no está en el tokenizer.", tag)
                    continue
                weights[self.tokenizer.convert_tokens_to_ids(tag)] = vector.to(weights.dtype)

    def _load_nllb_model(self):
        """
        Cargar el adaptador sin convertirlo en requisito de arranque.

        La aplicación puede operar con el diccionario aunque el adaptador no
        exista, falte ``peft`` o falle la descarga del modelo base. Cada error
        se registra y deja el motor en estado ``none`` para que el fallback
        continúe.
        """
        if torch is None or AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            self.logger = logging.getLogger(__name__)
            self.logger.warning(
                "Dependencias de IA no instaladas (torch/transformers); "
                "el modelo NLLB no estará disponible y se usará el diccionario."
            )
            self.model_type = "none"
            return

        if not (self.adapter_dir / self.ADAPTER_CONFIG).exists():
            self.logger.warning(
                "Adaptador NLLB no encontrado en '%s'; usando solo el diccionario.",
                self.adapter_dir,
            )
            self.model_type = "none"
            return

        try:
            from peft import PeftModel
        except ImportError as e:
            self.logger.error(f"Falta la dependencia 'peft' para cargar el adaptador NLLB: {e}")
            self.model_type = "none"
            return

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = self._pick_dtype(device)
            source = self._base_model_source()
            self.tags = self._load_language_tags()

            self.logger.info(f"Cargando tokenizer del adaptador desde '{self.adapter_dir}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.adapter_dir))

            self.logger.info(f"Cargando modelo base NLLB desde '{source}'...")
            try:
                base = AutoModelForSeq2SeqLM.from_pretrained(source, dtype=dtype)
            except TypeError:
                # Compatibilidad con versiones que aún usan torch_dtype.
                base = AutoModelForSeq2SeqLM.from_pretrained(source, torch_dtype=dtype)

            base.resize_token_embeddings(max(len(self.tokenizer), base.config.vocab_size))
            self._load_tag_vectors(base)

            self.model = PeftModel.from_pretrained(base, str(self.adapter_dir)).to(device).eval()
            self.model.config.use_cache = True
            self.device = device
            self.model_loaded = True
            self.model_type = "fine-tuned"
            self.logger.info(
                f"Adaptador NLLB cargado en {device} (tags: {sorted(self.tags)})"
            )
        except Exception as e:
            self.logger.error(f"Error cargando el adaptador NLLB: {e}")
            self.model_loaded = False
            self.model_type = "none"

    def _translate_with_dictionary(self, text, source_lang, target_lang):
        """Traducción usando el diccionario personalizado."""
        if not self.dictionary:
            return None

        normalized_text = text.strip().casefold()

        # Primero buscar la frase completa, porque el corpus contiene frases
        # enteras y no solo palabras sueltas.
        if normalized_text in self.dictionary:
            return {
                "translation": self.dictionary[normalized_text]["target_text"],
                "method": "dictionary",
                "confidence": 0.80,
                "tried_methods": ["nllb", "dictionary"],
            }

        # Mantener como respaldo la traducción palabra por palabra.
        words = normalized_text.split()
        translated_words = []
        found_translations = False
        for word in words:
            if word in self.dictionary:
                translated_words.append(self.dictionary[word]["target_text"])
                found_translations = True
            else:
                translated_words.append(word)

        if found_translations:
            return {
                "translation": " ".join(translated_words),
                "method": "dictionary",
                "confidence": 0.80,
                "tried_methods": ["nllb", "dictionary"],
            }

        return None

    def _translate_with_nllb(self, text, source_lang, target_lang):
        """
        Ejecutar la estrategia generativa de forma defensiva.

        El catálogo de idiomas es configurable y no todo código corresponde a un
        tag del adaptador. Se resuelve el idioma y se comprueba que exista antes
        de llamar a ``generate``, devolviendo ``None`` ante cualquier
        incompatibilidad para permitir que el fallback continúe.
        """
        if not self.model_loaded:
            return None

        source = self._resolve_language(source_lang)
        target = self._resolve_language(target_lang)
        if source is None or target is None:
            self.logger.warning(
                "Idioma no reconocido para el adaptador: %s -> %s", source_lang, target_lang
            )
            return None
        if source not in self.tags or target not in self.tags:
            self.logger.warning(
                "El adaptador no soporta la dirección %s -> %s", source, target
            )
            return None

        try:
            with self._lock:
                self.tokenizer.src_lang = self.tags[source]
                target_id = self.tokenizer.convert_tokens_to_ids(self.tags[target])
                if target_id is None or target_id == self.tokenizer.unk_token_id:
                    self.logger.warning("Tag de destino no registrado: %s", self.tags[target])
                    return None

                inputs = self.tokenizer(
                    text, return_tensors="pt", padding=True, truncation=True, max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    generated_tokens = self.model.generate(
                        **inputs,
                        forced_bos_token_id=target_id,
                        max_length=512,
                        num_beams=4,
                        use_cache=True,
                    )

                return self.tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )[0].strip()
        except Exception as e:
            self.logger.error(f"Error en traducción NLLB: {e}")
            return None

    def translate(self, text, source_lang, target_lang):
        """
        Ejecutar estrategias en orden de coste y precisión.

        Primero se consulta el adaptador NLLB porque es la estrategia
        contextual principal y debe tener prioridad siempre. Solo si el
        adaptador no está disponible o no produce una traducción válida se
        consulta el catálogo persistido; finalmente se devuelve el texto de
        entrada, manteniendo una respuesta válida incluso sin modelo local.
        """
        if not text or not text.strip():
            return {'translation': '', 'method': 'empty', 'confidence': 0}

        text = text.strip()
        self._load_dictionary(source_lang, target_lang)

        # Intentar diferentes métodos de traducción en orden de prioridad

        # 1. Intentar con el adaptador NLLB (estrategia principal contextual)
        if self.model_loaded:
            nllb_translation = self._translate_with_nllb(text, source_lang, target_lang)
            if nllb_translation and nllb_translation.lower() != text.lower():
                return {
                    'translation': nllb_translation,
                    'method': 'nllb',
                    'confidence': 0.85,
                    'tried_methods': ['nllb', 'dictionary']
                }

        # 2. Intentar con diccionario personalizado como respaldo
        dict_result = self._translate_with_dictionary(text, source_lang, target_lang)
        if dict_result:
            return dict_result

        # Fallback: devolver texto original
        return {
            'translation': text,
            'method': 'fallback',
            'confidence': 0.1,
            'tried_methods': ['nllb', 'dictionary']
        }

    def get_model_info(self):
        """Obtener información sobre el estado del modelo."""
        return {
            'nllb_loaded': self.model_loaded,
            'model_type': getattr(self, 'model_type', 'unknown'),
            'dictionary_entries': len(self.dictionary),
            'device': str(self.device) if self.model_loaded else 'N/A',
            'adapter_dir': str(self.adapter_dir),
            'finetuned_available': (self.adapter_dir / self.ADAPTER_CONFIG).exists(),
            'languages': sorted(self.tags) if self.tags else [],
        }
