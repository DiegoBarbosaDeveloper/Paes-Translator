"""
Motor de traducción desacoplado del catálogo de idiomas.

El diccionario rápido se resuelve por ``source_lang`` y ``target_lang`` en la
base de datos. El modelo generativo es una estrategia opcional: puede no
existir localmente, no tener un token para un código concreto o fallar al
cargar sus pesos. Ninguna de esas condiciones debe impedir que la aplicación
arranque ni que el diccionario siga funcionando, por lo que cada etapa falla
de forma aislada y termina en un fallback válido.
"""

from transformers import AutoModelForSeq2SeqLM
import torch
import os
from database import get_db
import logging

class AdvancedTranslationModel:
    """
    Modelo de traducción avanzado que combina:
    1. Modelo NLLB-200 para traducción contextual
    2. Diccionario personalizado para términos específicos
    3. Modelo generativo opcional con códigos de idioma configurables
    """
    
    def __init__(self, db=None):
        self.db = db or get_db()
        self.model = None
        self.tokenizer = None
        self.dictionary = {}
        self.model_loaded = False
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Inicializar componentes
        self._load_nllb_model()
    
    def _load_dictionary(self, source_lang, target_lang):
        """Cargar los pares de texto de la dirección solicitada."""
        try:
            source_lang = source_lang.strip()
            target_lang = target_lang.strip()
            self.dictionary = self.db.get_dictionary(source_lang, target_lang)
            self.logger.info(f"Pares cargados: {len(self.dictionary)}")
        except Exception as e:
            self.logger.error(f"Error cargando pares lingüísticos: {e}")
            self.dictionary = {}
    
    def _load_nllb_model(self):
        """
        Cargar pesos opcionales sin convertirlos en requisito de arranque.

        La aplicación puede operar con el repositorio léxico aunque el modelo
        sea demasiado grande, no esté descargado o tenga una configuración
        incompatible. Por ello los errores de importación, tokenización y
        carga se registran y dejan el motor en estado ``none``.
        """
        finetuned_path = "models/nllb-finetuned"
        base_path      = "models/nllb-200-distilled-600M"

        # Preferir modelo fine-tuneado si está disponible
        if os.path.exists(finetuned_path) and os.listdir(finetuned_path):
            model_path   = finetuned_path
            self.model_type = "fine-tuned"
            self.logger.info("Usando modelo NLLB fine-tuneado")
        elif os.path.exists(base_path):
            model_path   = base_path
            self.model_type = "base"
            self.logger.info("Usando modelo NLLB base (sin fine-tuning)")
        else:
            self.logger.warning("Ningún modelo NLLB disponible, usando pares lingüísticos")
            self.model_type = "none"
            return

        # Verificar que los pesos existen
        weight_files = ["model.safetensors", "pytorch_model.bin"]
        has_weights  = any(os.path.exists(os.path.join(model_path, w)) for w in weight_files)
        if not has_weights and os.path.exists(model_path):
            has_weights = any(
                f.startswith("model-") and f.endswith(".safetensors")
                for f in os.listdir(model_path)
            )

        if not has_weights:
            self.logger.warning(
                f"Pesos del modelo no encontrados en '{model_path}'. "
                "Descarga el modelo desde la pestaña de Entrenamiento."
            )
            self.model_type = "none"
            return

        try:
            self.logger.info(f"Cargando modelo NLLB desde '{model_path}'...")
            from transformers import NllbTokenizer
            self.tokenizer = NllbTokenizer.from_pretrained(model_path)
            self.model     = AutoModelForSeq2SeqLM.from_pretrained(model_path)

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)

            self.model_loaded = True
            self.logger.info(f"Modelo NLLB cargado en {self.device} (tipo: {self.model_type})")
        except Exception as e:
            self.logger.error(f"Error cargando modelo NLLB: {e}")
            self.model_loaded = False
            self.model_type   = "none"
    
    def _get_language_code(self, lang):
        """Usar el código configurado por el catálogo de idiomas."""
        return lang
    
    def _translate_with_dictionary(self, text, source_lang, target_lang):
        """Traducción usando el diccionario personalizado"""
        if not self.dictionary:
            return None
            
        words = text.lower().split()
        translated_words = []
        found_translations = False
        
        for word in words:
            if word in self.dictionary:
                translated_words.append(self.dictionary[word]['target_text'])
                found_translations = True
            else:
                translated_words.append(word)
        
        if found_translations:
            result = ' '.join(translated_words)
            return {
                'translation': result,
                'method': 'dictionary',
                'confidence': 0.80,
                'tried_methods': ['dictionary']
            }
        
        return None
    
    def _translate_with_nllb(self, text, source_lang, target_lang):
        """
        Ejecutar la estrategia generativa de forma defensiva.

        Los códigos de ``languages`` son datos configurables y no todos son
        necesariamente tokens válidos para NLLB. Se comprueba el token de
        destino antes de llamar a ``generate`` y se devuelve ``None`` ante
        cualquier incompatibilidad, permitiendo que el fallback continúe.
        """
        if not self.model_loaded:
            return None
        
        try:
            # Preparar códigos de idioma para NLLB
            src_lang_code = self._get_language_code(source_lang)
            tgt_lang_code = self._get_language_code(target_lang)
            
            # Configurar idioma fuente en el tokenizador (requerido por NLLB)
            self.tokenizer.src_lang = src_lang_code
            
            # Tokenizar
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Obtener el ID del token de idioma destino
            tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang_code)
            if tgt_lang_id is None or tgt_lang_id == self.tokenizer.unk_token_id:
                self.logger.warning(
                    "Código de idioma no soportado por el tokenizer: %s",
                    tgt_lang_code,
                )
                return None
            if tgt_lang_id is None or tgt_lang_id == self.tokenizer.unk_token_id:
                self.logger.warning("Código de idioma no soportado por el tokenizer: %s", tgt_lang_code)
                return None
            
            # Generar traducción
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_length=512,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decodificar resultado
            translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return translation.strip()
            
        except Exception as e:
            self.logger.error(f"Error en traducción NLLB: {e}")
            return None
    
    def translate(self, text, source_lang, target_lang):
        """
        Ejecutar estrategias en orden de coste y precisión.

        Primero se consulta el catálogo persistente porque es rápido y está
        adaptado al corpus de la aplicación. Solo si no hay coincidencia se
        intenta el modelo generativo; finalmente se devuelve el texto de
        entrada, manteniendo una respuesta válida incluso sin modelo local.
        """
        if not text or not text.strip():
            return {'translation': '', 'method': 'empty', 'confidence': 0}
        
        text = text.strip()
        self._load_dictionary(source_lang, target_lang)
        
        # Intentar diferentes métodos de traducción en orden de prioridad
        
        # 1. Intentar con diccionario personalizado (mayor precisión)
        dict_result = self._translate_with_dictionary(text, source_lang, target_lang)
        if dict_result:
            return dict_result
        
        # 2. Intentar con NLLB para traducción contextual
        if self.model_loaded:
            nllb_translation = self._translate_with_nllb(text, source_lang, target_lang)
            if nllb_translation and nllb_translation.lower() != text.lower():
                return {
                    'translation': nllb_translation,
                    'method': 'nllb',
                    'confidence': 0.7 if self.model_type == 'base' else 0.85,
                    'tried_methods': ['dictionary', 'nllb']
                }
        
        # Fallback: devolver texto original
        return {
            'translation': text,
            'method': 'fallback',
            'confidence': 0.1,
            'tried_methods': ['dictionary', 'nllb']
        }
    
    def get_model_info(self):
        """Obtener información sobre el estado del modelo"""
        return {
            'nllb_loaded':            self.model_loaded,
            'model_type':             getattr(self, 'model_type', 'unknown'),
            'dictionary_entries':     len(self.dictionary),
            'device':                 str(self.device) if self.model_loaded else 'N/A',
            'finetuned_available':    os.path.exists('models/nllb-finetuned') and
                                      bool(os.listdir('models/nllb-finetuned')),
        }