"""Reading and writing translation files (plain text or CSV)."""
from pathlib import Path

CSV_SUFFIXES = {".csv", ".tsv"}


def is_csv(path):
    return Path(path).suffix.lower() in CSV_SUFFIXES


def read_lines(path, column=None, reference=None):
    """Read a text file (one sentence per line) or a column of a CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not is_csv(path):
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()], None

    import pandas as pd

    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    frame = pd.read_csv(path, encoding="utf-8", sep=separator)
    if column not in frame.columns:
        raise ValueError(f"{path} has no column {column!r}; available: {', '.join(frame.columns)}")
    texts = [str(value).strip() for value in frame[column]]
    if reference is None:
        return texts, None
    if reference not in frame.columns:
        raise ValueError(f"{path} has no column {reference!r}; available: {', '.join(frame.columns)}")
    return texts, [str(value).strip() for value in frame[reference]]


def write_output(texts, translations, path, reference=None):
    """Write input/translation pairs as CSV when the path ends in .csv/.tsv, else as tab separated lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_csv(path):
        import pandas as pd

        data = {"input": texts, "translation": translations}
        if reference is not None:
            data["reference"] = reference
        pd.DataFrame(data).to_csv(path, index=False, encoding="utf-8")
    else:
        lines = [f"{text}\t{translation}" for text, translation in zip(texts, translations)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path