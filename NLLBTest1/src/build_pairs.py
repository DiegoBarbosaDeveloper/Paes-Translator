"""Turn the raw export into one two-column CSV per language pair."""
import argparse
from pathlib import Path

import pandas as pd

from src.config import (ALIASES, ES_COL, ID_COLUMNS, LANGUAGE_IDS, LANGUAGES, PAIR_FILES,
                        PROCESSED, RAW_CSV, SPANISH_ID, TEXT_COLUMNS)

REQUIRED_COLUMNS = ID_COLUMNS + TEXT_COLUMNS
WHITESPACE = r"\s+"


def load_raw(path=RAW_CSV):
    """Read the raw export as text, keeping empty strings instead of NaN."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing the column(s): {', '.join(missing)}")
    for column in ID_COLUMNS:
        frame[column] = frame[column].str.strip().astype(int)
    return frame


def load_pair_csv(path):
    """Read a (spanish, language) CSV and rewrite it in the raw schema.

    Returns None when the columns match neither the raw export nor a language pair,
    so an unrelated file in data/raw is skipped instead of crashing the build.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    columns = {ALIASES.get(str(name).strip().lower(), str(name).strip().lower()): name
               for name in frame.columns}
    language = next((name for name in columns if name in LANGUAGES and name != ES_COL), None)
    if ES_COL not in columns or language is None or len(columns) != 2:
        return None
    rows = pd.DataFrame({
        "source_language_id": SPANISH_ID,
        "target_language_id": LANGUAGES[language]["id"],
        "source_text": frame[columns[ES_COL]],
        "target_text": frame[columns[language]],
    })
    print(f"{path.name:28s} {len(rows):3d} {language} rows (spanish -> {language})")
    return rows


def load_all_raw(raw_csv=RAW_CSV):
    """The raw export plus every pair-shaped CSV sitting next to it."""
    raw = load_raw(raw_csv)
    raw_path = Path(raw_csv)
    for path in sorted(raw_path.parent.glob("*.csv")):
        if path.resolve() == raw_path.resolve():
            continue
        rows = load_pair_csv(path)
        if rows is None:
            print(f"{path.name:28s} skipped (not a raw export nor a (spanish, language) pair file)")
            continue
        raw = pd.concat([raw, rows], ignore_index=True)
    return raw


def _tidy(series):
    return series.astype(str).str.strip().str.replace(WHITESPACE, " ", regex=True)


def extract_pair(raw, language):
    """Rows of one language pair, always ordered as (spanish, language) and deduplicated."""
    language_id = LANGUAGES[language]["id"]
    foreign_to_spanish = (raw["source_language_id"] == language_id) & (raw["target_language_id"] == SPANISH_ID)
    spanish_to_foreign = (raw["source_language_id"] == SPANISH_ID) & (raw["target_language_id"] == language_id)

    forward = raw.loc[spanish_to_foreign, ["source_text", "target_text"]].rename(
        columns={"source_text": ES_COL, "target_text": language})
    reverse = raw.loc[foreign_to_spanish, ["source_text", "target_text"]].rename(
        columns={"source_text": language, "target_text": ES_COL})

    pair = pd.concat([forward, reverse], ignore_index=True)[[ES_COL, language]]
    pair[ES_COL] = _tidy(pair[ES_COL])
    pair[language] = _tidy(pair[language])
    pair = pair[(pair[ES_COL] != "") & (pair[language] != "")]
    return pair.drop_duplicates(ignore_index=True)


def build_pairs(out_dir=PROCESSED, raw_csv=RAW_CSV, verbose=True):
    """Write one CSV per language pair and return {language: DataFrame}."""
    raw = load_all_raw(raw_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = {}
    for language in LANGUAGES:
        pair = extract_pair(raw, language)
        destination = out_dir / PAIR_FILES[language].name
        pair.to_csv(destination, index=False, encoding="utf-8")
        pairs[language] = pair
        if verbose:
            print(f"{language:8s} {len(pair):3d} unique pairs -> {destination}")

    if verbose:
        if "validation_status" in raw.columns:
            counts = raw["validation_status"].value_counts().to_dict()
            print("raw validation_status counts:", counts)
        unknown = sorted({value for value in raw["source_language_id"]} - set(LANGUAGE_IDS))
        if unknown:
            print("WARNING: rows with unmapped source_language_id were ignored:", unknown)
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build one two-column CSV per language pair.")
    parser.add_argument("--raw", default=str(RAW_CSV))
    parser.add_argument("--out-dir", default=str(PROCESSED))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    build_pairs(out_dir=args.out_dir, raw_csv=args.raw, verbose=not args.quiet)


if __name__ == "__main__":
    main()