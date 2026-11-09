#!/usr/bin/env python3
"""
Exporta el corpus de PostgreSQL al CSV que consume NLLBTest1.

El archivo generado usa el mismo esquema que ``data/language_pair.csv``:

    id, dataset_id, source_language_id, target_language_id,
    source_text, target_text, validation_status, validated_at, created_at

Así ``python -m src.build_pairs`` puede separar piapoco y achagua sin que el
subproyecto de entrenamiento se conecte directamente a la base.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from sqlalchemy import select

from entities.models import Language, LanguagePair
from infrastructure.db import get_database_url, get_session, reset_engine

DEFAULT_OUTPUT = PROJECT_ROOT / "NLLBTest1" / "data" / "raw" / "language_pair.csv"
FIELDNAMES = [
    "id",
    "dataset_id",
    "source_language_id",
    "target_language_id",
    "source_text",
    "target_text",
    "validation_status",
    "validated_at",
    "created_at",
]


def format_datetime(value: datetime | None) -> str:
    """Serializar timestamps con el formato histórico de los CSV."""
    if value is None:
        return ""
    return value.strftime("%Y-%m-%d %H:%M:%S")


def export_rows(session) -> list[dict]:
    """Consultar los pares español -> piapoco/achagua ordenados por ID."""
    source = Language.__table__.alias("source_language")
    target = Language.__table__.alias("target_language")
    rows = session.execute(
        select(
            LanguagePair.id,
            LanguagePair.dataset_id,
            LanguagePair.source_language_id,
            LanguagePair.target_language_id,
            LanguagePair.source_text,
            LanguagePair.target_text,
            LanguagePair.validation_status,
            LanguagePair.validated_at,
            LanguagePair.created_at,
        )
        .join(source, source.c.id == LanguagePair.source_language_id)
        .join(target, target.c.id == LanguagePair.target_language_id)
        .where(
            source.c.code == "es",
            target.c.code.in_(["piapoco", "achagua"]),
        )
        .order_by(LanguagePair.id)
    ).all()
    return [dict(row._mapping) for row in rows]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT),
        help=f"Archivo de salida (por defecto: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--url", help="Sobrescribir DATABASE_URL")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.url:
        os.environ["DATABASE_URL"] = args.url
        reset_engine()

    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    session = get_session()
    try:
        rows = export_rows(session)
    finally:
        session.close()

    counts = {"piapoco": 0, "achagua": 0}
    code_by_id = {2: "piapoco", 3: "achagua"}
    for row in rows:
        code = code_by_id.get(row["target_language_id"])
        if code:
            counts[code] += 1

    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "validated_at": format_datetime(row["validated_at"]),
                    "created_at": format_datetime(row["created_at"]),
                }
            )

    print(f"Base: {get_database_url()}")
    print(f"Exportado: {output}")
    print(f"  Piapoco: {counts['piapoco']}")
    print(f"  Achagua: {counts['achagua']}")
    print(f"  Total:   {len(rows)}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
