"""Fine-tune NLLB with LoRA on Spanish <-> Piapoco and Spanish <-> Achagua."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import sacrebleu
import torch
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

from src.build_pairs import build_pairs
from src.console import force_utf8
from src.config import (ADAPTER_DIR, CPU_BATCH_SIZE, CPU_GRAD_ACCUM, EPOCHS, ES_COL,
                         GRAD_CHECKPOINTING, GPU_BATCH_SIZE, GPU_GRAD_ACCUM, GPU_LORA_ALPHA,
                         GPU_LORA_R, HOLDOUT, LORA_ALPHA, LORA_DROPOUT, LORA_R, LORA_TARGETS, LR,
                         LANGUAGES, MAX_LEN, MODEL_NAME, SEED, SPA_TAG, TAG_INIT, TAG_MODE,
                         WARMUP_RATIO)
from src.modeling import (LanguageTags, Translator, add_language_tags, default_batch_size,
                          default_eval_batch_size, enable_fast_matmul, init_tag_vectors,
                          pick_device, pick_dtype, save_tag_vectors)

MIN_ROWS_FOR_HOLDOUT = 4
TRAIN_ARTEFACTS = ("adapter_config.json", "adapter_model.safetensors", "tag_vectors.pt")


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_pairs(pairs, holdout=HOLDOUT, seed=SEED):
    """Split each pair into train/test; with holdout == 0 everything is training data."""
    split = {}
    for language, frame in pairs.items():
        if holdout > 0 and len(frame) >= MIN_ROWS_FOR_HOLDOUT:
            train, test = train_test_split(frame, test_size=holdout, random_state=seed)
        else:
            train, test = frame, frame.iloc[0:0]
        split[language] = (train.reset_index(drop=True), test.reset_index(drop=True))
    return split


def encode_direction(tokenizer, frame, src_col, tgt_col, src_language, tgt_language, tags):
    tokenizer.src_lang, tokenizer.tgt_lang = tags[src_language], tags[tgt_language]
    encoded = tokenizer(frame[src_col].tolist(), text_target=frame[tgt_col].tolist(),
                        truncation=True, max_length=MAX_LEN)
    return Dataset.from_dict({key: encoded[key] for key in ("input_ids", "attention_mask", "labels")})


def build_datasets(tokenizer, split, tags):
    """Both directions of every pair, tagged with the right language tokens."""
    datasets = {}
    for language, (train, test) in split.items():
        forward = encode_direction(tokenizer, train, ES_COL, language, "spanish", language, tags)
        backward = encode_direction(tokenizer, train, language, ES_COL, language, "spanish", tags)
        datasets[language] = {
            "train": concatenate_datasets([forward, backward]),
            "test": concatenate_datasets([
                encode_direction(tokenizer, test, ES_COL, language, "spanish", language, tags),
                encode_direction(tokenizer, test, language, ES_COL, language, "spanish", tags),
            ]) if len(test) else Dataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []}),
        }
    return datasets


def check_tagging(tokenizer, tags):
    """The sentence must start with the source language tag and end with </s>."""
    language = next(name for name in LANGUAGES)
    tokenizer.src_lang, tokenizer.tgt_lang = tags[language], tags["spanish"]
    ids = tokenizer("Buenos dias para todos")["input_ids"]
    print("tag check:", [tokenizer.convert_ids_to_tokens(index) for index in ids])
    if ids[0] != tokenizer.convert_tokens_to_ids(tags[language]) or ids[-1] != tokenizer.eos_token_id:
        raise ValueError("The source language tag is not being prepended to the input")


def warmup_steps(num_examples, epochs, batch_size, grad_accum):
    steps_per_epoch = math.ceil(num_examples / max(1, batch_size * grad_accum))
    return max(1, round(WARMUP_RATIO * steps_per_epoch * epochs))


def training_arguments(output_dir, epochs, batch_size, grad_accum, device, with_eval, lr, num_examples=1,
                       eval_batch_size=None):
    cuda = device.type == "cuda"
    # A 6 GB card has no use for activation checkpointing at this batch size, and it costs ~30% speed.
    checkpointing = cuda and (GRAD_CHECKPOINTING or batch_size > 32)
    return Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size or batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if checkpointing else None,
        bf16=cuda and torch.cuda.is_bf16_supported(),
        fp16=cuda and not torch.cuda.is_bf16_supported(),
        learning_rate=lr,
        num_train_epochs=epochs,
        warmup_steps=warmup_steps(num_examples, epochs, batch_size, grad_accum),
        lr_scheduler_type="linear",
        eval_strategy="epoch" if with_eval else "no",
        save_strategy="epoch" if with_eval else "no",
        save_total_limit=1,
        load_best_model_at_end=with_eval,
        metric_for_best_model="eval_loss" if with_eval else None,
        logging_steps=5,
        predict_with_generate=False,
        remove_unused_columns=False,
        report_to="none",
        seed=SEED,
    )


def score_directions(translator, split, batch_size=None):
    """chrF++ per pair and direction, plus a few example translations."""
    report = {}
    batch_size = batch_size or translator.batch_size
    for language, (train_frame, test_frame) in split.items():
        held_out = len(test_frame) > 0
        frame = test_frame if held_out else train_frame
        origin = "held-out test" if held_out else "training rows (seen by the model)"
        report[language] = {"evaluated_on": origin, "directions": {}}
        for source, target in ((ES_COL, language), (language, ES_COL)):
            source_texts = frame[source].tolist()
            expected = frame[target].tolist()
            hypothesis = translator.translate(source_texts, source, target, batch_size=batch_size)
            chrf = sacrebleu.corpus_chrf(hypothesis, [expected], word_order=2).score
            report[language]["directions"][f"{source}->{target}"] = {
                "chrf_pp": round(chrf, 1),
                "examples": [{"input": source_texts[i], "expected": expected[i], "model": hypothesis[i]}
                             for i in range(min(3, len(source_texts)))],
            }
    return report


def print_report(report):
    for language, info in report.items():
        print(f"\n{language} (chrF++ on {info['evaluated_on']})")
        for direction, score in info["directions"].items():
            print(f"  {direction:26s} {score['chrf_pp']:5.1f}")
            for example in score["examples"]:
                print(f"    {example['input']!r}\n      -> {example['model']!r}   (expected {example['expected']!r})")


def adapter_is_trained(model_dir=ADAPTER_DIR):
    return all((Path(model_dir) / name).exists() for name in TRAIN_ARTEFACTS)


def train(epochs=EPOCHS, batch_size=None, grad_accum=None, lr=LR, holdout=HOLDOUT,
          model_dir=ADAPTER_DIR, device=None, force=False, verbose=True, tag_mode=TAG_MODE,
          tag_init=TAG_INIT, lora_r=None, lora_dropout=None):
    """Build the data, fine-tune LoRA and save the adapter, tokenizer and tag vectors.

    `batch_size`/`grad_accum` default to the values that suit the detected device;
    `lora_r`/`lora_dropout` default to the config values (alpha follows the 2:1 alpha:r ratio).
    """
    model_dir = Path(model_dir)
    if adapter_is_trained(model_dir) and not force:
        print(f"Adapter already trained in {model_dir} (use force=True to retrain)")
        return model_dir

    seed_everything()
    force_utf8()
    device = pick_device(device)
    dtype = pick_dtype(device)
    tf32 = enable_fast_matmul(device)
    batch_size = batch_size or default_batch_size(device)
    grad_accum = grad_accum if grad_accum is not None else (
        GPU_GRAD_ACCUM if device.type == "cuda" else CPU_GRAD_ACCUM)
    eval_batch_size = default_eval_batch_size(device)
    print(f"Device: {device} ({dtype}, tf32={tf32})")
    print(f"batch {batch_size} x accum {grad_accum} = effective {batch_size * grad_accum}, "
          f"eval batch {eval_batch_size}")
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    pairs = build_pairs()
    split = split_pairs(pairs, holdout=holdout)
    for language, (train_frame, test_frame) in split.items():
        print(f"{language:8s} train {len(train_frame):3d} | test {len(test_frame):3d}")

    tags = LanguageTags.build(tag_mode)
    print("tags:", tags.as_dict())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tags.uses_new_tags:
        print("new tag ids:", add_language_tags(tokenizer))

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, dtype=dtype)
    model.resize_token_embeddings(max(len(tokenizer), model.config.vocab_size))
    if tags.uses_new_tags:
        init_tag_vectors(model, tokenizer, strategy=tag_init)
    check_tagging(tokenizer, tags)

    datasets = build_datasets(tokenizer, split, tags)
    train_dataset = concatenate_datasets([datasets[language]["train"] for language in split])
    eval_dataset = concatenate_datasets([datasets[language]["test"] for language in split])
    with_eval = len(eval_dataset) > 0
    print(f"{len(train_dataset)} training examples, {len(eval_dataset)} evaluation examples")

    cuda = device.type == "cuda"
    custom_rank = lora_r is not None
    lora_r = lora_r or (GPU_LORA_R if cuda else LORA_R)
    lora_alpha = 2 * lora_r if custom_rank else (GPU_LORA_ALPHA if cuda else LORA_ALPHA)
    lora_dropout = LORA_DROPOUT if lora_dropout is None else lora_dropout
    model.config.use_cache = not GRAD_CHECKPOINTING
    model = get_peft_model(model, LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=LORA_TARGETS, task_type="SEQ_2_SEQ_LM"))
    model.print_trainable_parameters()

    checkpoints = model_dir.parent / "checkpoints"
    args = training_arguments(checkpoints, epochs, batch_size, grad_accum, device, with_eval, lr,
                              num_examples=len(train_dataset), eval_batch_size=eval_batch_size)
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if with_eval else None,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
    )
    trainer.train()

    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    save_tag_vectors(trainer.model, tokenizer, model_dir / "tag_vectors.pt", tags)
    tags.save(model_dir, tag_mode)
    print("Saved adapter to", model_dir)

    translator = Translator.load(model_dir, device=device, batch_size=eval_batch_size)
    report = score_directions(translator, split, eval_batch_size)
    if verbose:
        print_report(report)
    payload = {"model_name": MODEL_NAME, "tag_mode": tag_mode, "tag_init": tag_init,
               "tags": tags.as_dict(), "spanish_tag": SPA_TAG, "holdout": holdout, "epochs": epochs,
               "learning_rate": lr, "train_loss": trainer.state.log_history[-1].get("train_loss"),
               "device": str(device), "dtype": str(dtype), "lora_r": lora_r,
               "lora_dropout": lora_dropout, "batch_size": batch_size,
               "grad_accum": grad_accum, "runtime_seconds": round(time.time() - started, 1),
               "peak_vram_gb": round(torch.cuda.max_memory_allocated(device) / 1024**3, 2)
               if device.type == "cuda" else None,
               "rows": {name: len(split[name][0]) for name in split}, "report": report}
    (model_dir / "training_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    return model_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fine-tune NLLB on the extracted language pairs.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"defaults to {GPU_BATCH_SIZE} on a GPU, {CPU_BATCH_SIZE} on CPU")
    parser.add_argument("--grad-accum", type=int, default=None,
                        help=f"defaults to {GPU_GRAD_ACCUM} on a GPU, {CPU_GRAD_ACCUM} on CPU")
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--holdout", type=float, default=HOLDOUT,
                        help="0 trains on every row, 0.2 keeps a test split")
    parser.add_argument("--model-dir", default=str(ADAPTER_DIR))
    parser.add_argument("--device", default=None)
    parser.add_argument("--tag-mode", choices=("new", "alias"), default=TAG_MODE,
                        help="'new' registers piapoco_Latn/achagua_Latn, 'alias' reuses NLLB tags")
    parser.add_argument("--tag-init", choices=("mean", "language"), default=TAG_INIT)
    parser.add_argument("--lora-r", type=int, default=None,
                        help=f"LoRA rank, defaults to {GPU_LORA_R} on a GPU, {LORA_R} on CPU")
    parser.add_argument("--lora-dropout", type=float, default=None, help=f"defaults to {LORA_DROPOUT}")
    parser.add_argument("--force", action="store_true", help="retrain even if the adapter exists")
    args = parser.parse_args(argv)
    train(epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum, lr=args.lr,
          holdout=args.holdout, model_dir=args.model_dir, device=args.device, force=args.force,
          tag_mode=args.tag_mode, tag_init=args.tag_init, lora_r=args.lora_r,
          lora_dropout=args.lora_dropout)


if __name__ == "__main__":
    main()