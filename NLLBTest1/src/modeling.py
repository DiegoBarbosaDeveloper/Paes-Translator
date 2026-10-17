"""Language tags, adapter loading and the translation call used by the CLI."""
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import (ADAPTER_DIR, ALIASES, CPU_BATCH_SIZE, EVAL_BATCH_SIZE, FALLBACK_INIT_TAG,
                         GPU_BATCH_SIZE, GPU_EVAL_BATCH_SIZE, LANGUAGES, LANG_NAMES, MAX_LEN,
                         MODEL_NAME, NUM_BEAMS, SEED, SPA_TAG, TAG_INIT, TAG_MODE, TAG_VECTORS)

LANGUAGE_TAG_PATTERN = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")
LANGUAGE_CONFIG = "language_config.json"


def resolve_language(name):
    """Map 'es', 'piapoco', ... to a canonical language name."""
    key = str(name).strip().lower()
    if key not in ALIASES:
        raise ValueError(f"Unknown language {name!r}. Known: {', '.join(LANG_NAMES)}")
    return ALIASES[key]


class LanguageTags:
    """Which NLLB token represents each language for this adapter."""

    def __init__(self, mapping):
        self._mapping = dict(mapping)

    @classmethod
    def build(cls, mode=TAG_MODE):
        if mode not in {"new", "alias"}:
            raise ValueError(f"Unknown tag mode {mode!r}, use 'new' or 'alias'")
        key = "alias" if mode == "alias" else "tag"
        mapping = {"spanish": SPA_TAG}
        mapping.update({name: LANGUAGES[name][key] for name in LANGUAGES})
        return cls(mapping)

    @classmethod
    def load(cls, model_dir=ADAPTER_DIR):
        """Read the tags stored next to an adapter, so reloads match training exactly."""
        path = Path(model_dir) / LANGUAGE_CONFIG
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            return cls(saved["tags"]), saved.get("mode", "new")
        return cls.build(), TAG_MODE

    def save(self, model_dir, mode):
        path = Path(model_dir) / LANGUAGE_CONFIG
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"mode": mode, "tags": self._mapping}, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    def __getitem__(self, language):
        language = resolve_language(language)
        if language not in self._mapping:
            raise ValueError(f"{language} is not part of this model")
        return self._mapping[language]

    def __contains__(self, language):
        return resolve_language(language) in self._mapping

    def as_dict(self):
        return dict(self._mapping)

    @property
    def uses_new_tags(self):
        return any(LANGUAGE_TAG_PATTERN.match(tag) is None for tag in self._mapping.values())


def language_tag(language, tags=None):
    return (tags or LanguageTags.build())[language]


def pick_device(preferred=None):
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device):
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def enable_fast_matmul(device):
    """TF32 matmuls on Ada/Hopper/Blackwell: big speed-up on Ampere+ without changing weights."""
    if device.type != "cuda":
        return False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass
    return True


def default_batch_size(device):
    """Larger batches on a GPU; the CPU path stays small so a laptop can still train."""
    return GPU_BATCH_SIZE if device.type == "cuda" else CPU_BATCH_SIZE


def default_eval_batch_size(device):
    return GPU_EVAL_BATCH_SIZE if device.type == "cuda" else EVAL_BATCH_SIZE


def add_language_tags(tokenizer, languages=LANGUAGES):
    """Register the custom tags as special tokens (Spanish keeps NLLB's own tag)."""
    vocab = tokenizer.get_vocab()
    missing = [name for name in languages if languages[name]["tag"] not in vocab]
    if missing:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [languages[name]["tag"] for name in missing]},
            replace_extra_special_tokens=False,
        )
    return {name: tokenizer.convert_tokens_to_ids(languages[name]["tag"]) for name in languages}


def resolve_init_tag(tokenizer, language):
    """The tag whose embedding seeds a new one: a related language, never Spanish."""
    wanted = LANGUAGES[language]["init_from"]
    known = tokenizer.convert_tokens_to_ids(wanted) != tokenizer.unk_token_id
    if known:
        return wanted
    print(f"WARNING: {wanted} is not an NLLB tag, seeding {language} with {FALLBACK_INIT_TAG} instead")
    return FALLBACK_INIT_TAG


def nllb_language_tag_ids(tokenizer):
    return [tokenizer.convert_tokens_to_ids(token) for token in tokenizer.get_vocab()
            if LANGUAGE_TAG_PATTERN.match(token)]


def init_tag_vectors(model, tokenizer, seed=SEED, languages=LANGUAGES, strategy=TAG_INIT, noise=0.02):
    """Seed each new tag's embedding so the model has no strong prior about what it means.

    "mean" averages NLLB's own language tags (the new tag then plays the same role without
    inheriting one language's vocabulary), "language" copies a related language's embedding.
    """
    weights = model.get_input_embeddings().weight
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        with torch.no_grad():
            if strategy == "mean":
                tag_ids = nllb_language_tag_ids(tokenizer)
                base = weights[tag_ids].float().mean(dim=0)
            elif strategy == "language":
                base = None
            else:
                raise ValueError(f"Unknown tag init {strategy!r}, use 'mean' or 'language'")
            for language in languages:
                new_id = tokenizer.convert_tokens_to_ids(LANGUAGES[language]["tag"])
                if new_id == tokenizer.unk_token_id:
                    raise ValueError(f"The tag for {language} was not registered on the tokenizer")
                if strategy == "language":
                    source = weights[tokenizer.convert_tokens_to_ids(resolve_init_tag(tokenizer, language))].clone()
                    base = source + 0.5 * source.std() * torch.randn_like(source)
                else:
                    base = base.to(weights.dtype)
                weights[new_id] = base + noise * base.std() * torch.randn_like(base)
    finally:
        torch.random.set_rng_state(generator_state)
    return {language: tokenizer.convert_tokens_to_ids(LANGUAGES[language]["tag"]) for language in languages}


def save_tag_vectors(model, tokenizer, path=TAG_VECTORS, tags=None):
    """Save the embeddings of the tags this run introduced, so a reload is exact."""
    tags = tags or LanguageTags.build()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = model.get_input_embeddings().weight.detach().to(torch.float32).cpu()
    vectors = {tag: weights[tokenizer.convert_tokens_to_ids(tag)].clone()
               for tag in tags.as_dict().values()
               if LANGUAGE_TAG_PATTERN.match(tag) is None}
    torch.save(vectors, path)
    return path


def load_tag_vectors(model, tokenizer, path=TAG_VECTORS):
    """Put the saved tag embeddings back after loading a fresh copy of the base model."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the fine-tune first (notebooks/NLLB_piapoco.ipynb)")
    vectors = torch.load(path, map_location="cpu", weights_only=True)
    weights = model.get_input_embeddings().weight
    vocab = tokenizer.get_vocab()
    with torch.no_grad():
        for tag, vector in vectors.items():
            if tag not in vocab:
                raise ValueError(f"{tag} is missing from the saved tokenizer")
            weights[tokenizer.convert_tokens_to_ids(tag)] = vector.to(weights.dtype).to(weights.device)
    return {tag: tokenizer.convert_tokens_to_ids(tag) for tag in vectors}


class Translator:
    """Loads the fine-tuned adapter once and translates many texts with it."""

    def __init__(self, tokenizer, model, device, tags, num_beams=NUM_BEAMS, batch_size=EVAL_BATCH_SIZE):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.tags = tags
        self.num_beams = num_beams
        self.batch_size = batch_size

    @classmethod
    def load(cls, model_dir=ADAPTER_DIR, device=None, num_beams=NUM_BEAMS, dtype=None,
             batch_size=None):
        model_dir = Path(model_dir)
        if not (model_dir / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"No fine-tuned adapter in {model_dir}. Run src.train first "
                f"(or notebooks/NLLB_piapoco.ipynb).")
        device = pick_device(device)
        dtype = dtype or pick_dtype(device)
        enable_fast_matmul(device)
        tags, _ = LanguageTags.load(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, dtype=dtype)
        base.resize_token_embeddings(max(len(tokenizer), base.config.vocab_size))
        load_tag_vectors(base, tokenizer, model_dir / "tag_vectors.pt")
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, str(model_dir)).to(device).eval()
        # KV cache plus eager attention is the fast path for batched generation.
        model.config.use_cache = True
        if device.type == "cuda":
            model.config.attn_implementation = "sdpa"
        return cls(tokenizer, model, device, tags, num_beams=num_beams,
                   batch_size=batch_size or default_eval_batch_size(device))

    def tag_id(self, language):
        return self.tokenizer.convert_tokens_to_ids(self.tags[language])

    @torch.inference_mode()
    def translate(self, texts, source, target, num_beams=None, batch_size=None,
                  max_length=MAX_LEN):
        """Translate a list of strings from `source` to `target` language names."""
        source, target = resolve_language(source), resolve_language(target)
        if source == target:
            raise ValueError("Source and target language must differ")
        num_beams = self.num_beams if num_beams is None else num_beams
        batch_size = batch_size or self.batch_size
        target_id = self.tag_id(target)
        self.tokenizer.src_lang = self.tags[source]
        self.model.eval()

        clean = [str(text).strip() for text in texts]
        results = ["" for _ in clean]
        pending = [index for index, text in enumerate(clean) if text]
        for start in range(0, len(pending), batch_size):
            chunk = pending[start:start + batch_size]
            batch = self.tokenizer([clean[index] for index in chunk], return_tensors="pt",
                                   padding=True, truncation=True, max_length=MAX_LEN).to(self.device)
            generated = self.model.generate(
                **batch,
                forced_bos_token_id=target_id,
                max_length=max_length,
                num_beams=num_beams,
                use_cache=True,
            )
            for index, decoded in zip(chunk, self.tokenizer.batch_decode(generated, skip_special_tokens=True)):
                results[index] = decoded.strip()
        return results

    def translate_one(self, text, source, target, **kwargs):
        return self.translate([text], source, target, **kwargs)[0]