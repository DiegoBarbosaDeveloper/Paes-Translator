"""Central settings for the NLLB custom-language project.
Edit this file, never the notebook, when you want to change a name or a hyperparameter."""
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "language_pair.csv"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
ADAPTER_DIR = MODELS / "nllb-spa-piapoco-achagua"
TAG_VECTORS = ADAPTER_DIR / "tag_vectors.pt"

# ---------- Raw CSV schema ----------
SPANISH_ID = 1
LANGUAGE_IDS = {1: "spanish", 2: "piapoco", 3: "achagua"}
ID_COLUMNS = ("source_language_id", "target_language_id")
TEXT_COLUMNS = ("source_text", "target_text")

# ---------- Model ----------
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# ---------- Languages ----------
SPA_TAG = "spa_Latn"      # Spanish tag already known by NLLB
TAG_MODE = "new"          # "new" adds piapoco_Latn/achagua_Latn, "alias" reuses existing NLLB tags
TAG_INIT = "language"     # "language" seeds each tag from a different NLLB tag. "mean" (the average of every
                          # language tag) leaves piapoco/achagua near-identical and frozen, so the decoder
                          # cannot tell the two target languages apart and both directions collapse into one.
LANGUAGES = {
    "piapoco": {"id": 2, "tag": "piapoco_Latn", "alias": "ita_Latn", "init_from": "ita_Latn"},
    "achagua": {"id": 3, "tag": "achagua_Latn", "alias": "ayr_Latn", "init_from": "ayr_Latn"},
}
FALLBACK_INIT_TAG = "afr_Latn"   # any Latin-script tag, only used if init_from is missing
ALIASES = {
    "es": "spanish", "spa": "spanish", "spanish": "spanish", "castellano": "spanish",
    "pia": "piapoco", "piapoco": "piapoco",
    "aca": "achagua", "achagua": "achagua",
}
LANG_NAMES = ("spanish",) + tuple(LANGUAGES)
SHORT_NAMES = {"spanish": "es", "piapoco": "pia", "achagua": "aca"}
DIRECTIONS = (("spanish", "piapoco"), ("piapoco", "spanish"),
              ("spanish", "achagua"), ("achagua", "spanish"))

# ---------- Output CSVs ----------
ES_COL = "spanish"
PAIR_FILES = {name: PROCESSED / f"spanish_{name}.csv" for name in LANGUAGES}

# ---------- Translation ----------
MAX_LEN = 64                # max tokens per sentence; raise if your sentences are long
NUM_BEAMS = 4
BEAM_CHOICES = (1, 4, 8)
EVAL_BATCH_SIZE = 8
GPU_EVAL_BATCH_SIZE = 32

# ---------- Training (auto-scales: CPU stays small, a 6 GB GPU takes bigger batches) ----------
MAX_TOKENS_PER_BATCH = 2048
SEED = 42
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
CPU_BATCH_SIZE = 4
CPU_GRAD_ACCUM = 2           # effective batch = BATCH_SIZE * GRAD_ACCUM
GPU_BATCH_SIZE = 8           # 8 x 1 keeps the same effective batch as the CPU run, so lr=5e-4 still fits
GPU_GRAD_ACCUM = 1
GPU_LORA_R = 32              # a 6 GB card has room for a wider adapter
GPU_LORA_ALPHA = 64
BATCH_SIZE = None            # None picks CPU_/GPU_ defaults from the detected device
GRAD_ACCUM = None
LR = 5e-4
EPOCHS = 30
WARMUP_RATIO = 0.05
HOLDOUT = 0.0               # 0.0 trains on every row (best demo quality), 0.2 gives an honest test set
GRAD_CHECKPOINTING = False   # only useful when VRAM is tight