#!/usr/bin/env python3
"""
Importa los CSV históricos de ``data/`` a PostgreSQL.

Reglas principales:

* Cada fila debe ser español -> piapoco (1 -> 2) o español -> achagua (1 -> 3).
* La separación no depende del nombre del archivo: se valida el
  ``target_language_id`` de cada fila y se contrasta con el archivo cuando este
  declara un idioma (``piapoco.csv``/``achagua.csv``).
* Piapoco y achagua se guardan en datasets distintos (1 y 2).
* Las fuentes duplicadas se resuelven a favor del archivo específico más
  reciente; las variantes descartadas se escriben en un reporte.
* La importación es idempotente: se puede ejecutar varias veces sin duplicar.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from sqlalchemy import delete, select, text
from sqlalchemy.exc import SQLAlchemyError

from entities.models import (
    Dataset,
    Language,
    LanguagePair,
    TranslationDirection,
    init_db,
    seed_defaults,
)
from infrastructure.db import (
    get_database_url,
    get_engine,
    get_session,
    remove_session,
    reset_engine,
)

LANGUAGES = {1: "es", 2: "piapoco", 3: "achagua"}
DATASET_BY_TARGET = {2: 1, 3: 2}
DATASET_NAMES = {1: "piapoco", 2: "achagua"}
EXPECTED_COUNTS = {"piapoco": 75, "achagua": 24}
FILE_LANGUAGE = {"piapoco.csv": 2, "achagua.csv": 3}
SKIP_FILES = {"import_conflicts.csv"}


@dataclass
class ImportRow:
    """Fila normalizada lista para insertarse en ``language_pair``."""

    source_id: int
    target_id: int
    source_text: str
    target_text: str
    validation_status: str | None
    validated_at: datetime | None
    created_at: datetime | None
    record_id: int | None
    source_file: str
    priority: int


def clean_text(value: object) -> str:
    """Quitar espacios exteriores y colapsar espacios internos."""
    return " ".join(str(value or "").split())


def parse_int(value: object) -> int | None:
    """Convertir un valor CSV a entero, ignorando vacíos e inválidos."""
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def parse_datetime(value: object) -> datetime | None:
    """Interpretar las fechas ISO de los CSV como timestamps sin zona."""
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        result = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if result.tzinfo is not None:
        result = result.replace(tzinfo=None)
    return result


def load_rows(data_dir: Path) -> list[ImportRow]:
    """Leer todos los CSV relevantes y normalizarlos."""
    rows: list[ImportRow] = []
    for path in sorted(data_dir.glob("*.csv")):
        name = path.name.casefold()
        if name in SKIP_FILES or path.name.startswith(".~lock"):
            continue

        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                print(f"{path.name}: ignorado (sin cabecera)")
                continue
            if "source_text" not in reader.fieldnames or "target_text" not in reader.fieldnames:
                print(f"{path.name}: ignorado (no es un CSV de pares)")
                continue

            expected_target = FILE_LANGUAGE.get(path.name)
            for line_number, raw in enumerate(reader, start=2):
                source_text = clean_text(raw.get("source_text"))
                target_text = clean_text(raw.get("target_text"))
                if not source_text or not target_text:
                    print(f"{path.name}:{line_number}: omitido (texto vacío)")
                    continue

                source_id = parse_int(raw.get("source_language_id")) or 1
                target_id = parse_int(raw.get("target_language_id"))
                if target_id is None:
                    target_id = expected_target

                if source_id != 1 or target_id not in DATASET_BY_TARGET:
                    raise ValueError(
                        f"{path.name}:{line_number}: dirección inválida "
                        f"{source_id} -> {target_id}. Solo se admite 1 -> 2/3."
                    )
                if expected_target is not None and target_id != expected_target:
                    raise ValueError(
                        f"{path.name}:{line_number}: el archivo declara "
                        f"{LANGUAGES[expected_target]} pero la fila apunta a "
                        f"{LANGUAGES[target_id]}."
                    )

                rows.append(
                    ImportRow(
                        source_id=source_id,
                        target_id=target_id,
                        source_text=source_text,
                        target_text=target_text,
                        validation_status=(raw.get("validation_status") or "").strip() or None,
                        validated_at=parse_datetime(raw.get("validated_at")),
                        created_at=parse_datetime(raw.get("created_at")),
                        record_id=parse_int(raw.get("id")),
                        source_file=path.name,
                        priority=2 if path.name in FILE_LANGUAGE else 1,
                    )
                )

    return rows


def deduplicate(rows: list[ImportRow]) -> tuple[list[ImportRow], list[tuple[ImportRow, ImportRow]]]:
    """
    Elegir una sola traducción por (idioma destino, texto origen).

    Gana primero el archivo específico del idioma y, en empate, el ID más alto
    (fila más reciente). Las variantes descartadas se devuelven para reporte.
    """
    groups: dict[tuple[int, str], list[ImportRow]] = defaultdict(list)
    for row in rows:
        groups[(row.target_id, row.source_text.casefold())].append(row)

    selected: list[ImportRow] = []
    conflicts: list[tuple[ImportRow, ImportRow]] = []
    for group in groups.values():
        chosen = max(group, key=lambda row: (row.priority, row.record_id or 0))
        selected.append(chosen)
        for other in group:
            if other is chosen:
                continue
            if (
                other.source_text.casefold() != chosen.source_text.casefold()
                or other.target_text != chosen.target_text
            ):
                conflicts.append((chosen, other))

    selected.sort(key=lambda row: (row.target_id, row.record_id or 0, row.source_text))
    return selected, conflicts


def write_conflicts(path: Path, conflicts: list[tuple[ImportRow, ImportRow]]) -> None:
    """Escribir un CSV con las variantes descartadas para revisión manual."""
    if not conflicts:
        if path.exists():
            path.unlink()
        return

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "target_language_id",
                "source_text",
                "kept_id",
                "kept_target_text",
                "kept_file",
                "discarded_id",
                "discarded_target_text",
                "discarded_file",
            ]
        )
        for kept, discarded in conflicts:
            writer.writerow(
                [
                    kept.target_id,
                    kept.source_text,
                    kept.record_id,
                    kept.target_text,
                    kept.source_file,
                    discarded.record_id,
                    discarded.target_text,
                    discarded.source_file,
                ]
            )


def counts_by_language(rows: list[ImportRow]) -> dict[str, int]:
    """Contar filas seleccionadas por nombre de idioma destino."""
    counts = {"piapoco": 0, "achagua": 0}
    for row in rows:
        counts[LANGUAGES[row.target_id]] += 1
    return counts


def print_summary(all_rows: list[ImportRow], selected: list[ImportRow], conflicts, conflict_path: Path) -> None:
    """Mostrar un resumen legible del proceso."""
    print("\nResumen de importación")
    print("----------------------")
    for name in sorted({row.source_file for row in all_rows}):
        total = sum(1 for row in all_rows if row.source_file == name)
        print(f"  {name:24s} {total:3d} filas")
    counts = counts_by_language(selected)
    print(f"  Piapoco seleccionados:  {counts['piapoco']:3d}")
    print(f"  Achagua seleccionados:  {counts['achagua']:3d}")
    print(f"  Total seleccionados:    {len(selected):3d}")
    print(f"  Variantes en conflicto: {len(conflicts):3d}")
    if conflicts:
        print(f"  Reporte de conflictos:  {conflict_path}")


def ensure_reference_data(session) -> None:
    """Garantizar idiomas, datasets por idioma y direcciones base."""
    for language_id, code, name in (
        (1, "es", "Español"),
        (2, "piapoco", "Piapoco"),
        (3, "achagua", "Achagua"),
    ):
        if session.get(Language, language_id) is None:
            session.add(
                Language(id=language_id, code=code, name=name, active="true")
            )

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for dataset_id, name in DATASET_NAMES.items():
        dataset = session.get(Dataset, dataset_id)
        if dataset is None:
            session.add(
                Dataset(
                    id=dataset_id,
                    name=name,
                    version="1",
                    description=f"Corpus {name}",
                    status="active",
                    created_at=now,
                )
            )
        elif dataset.name in {"default", name}:
            dataset.name = name
            dataset.description = f"Corpus {name}"
            dataset.status = "active"

    for direction_id, source_id, target_id in (
        (1, 1, 2),
        (2, 2, 1),
        (3, 1, 3),
        (4, 3, 1),
    ):
        if session.get(TranslationDirection, direction_id) is None:
            session.add(
                TranslationDirection(
                    id=direction_id,
                    source_language_id=source_id,
                    target_language_id=target_id,
                    active=1,
                )
            )

    session.commit()


def upsert_rows(session, rows: list[ImportRow], reset: bool) -> None:
    """Insertar/actualizar pares de forma idempotente."""
    if reset:
        session.execute(delete(LanguagePair))
        session.commit()

    existing = {
        (row.target_language_id, row.source_text.strip().casefold()): row
        for row in session.execute(select(LanguagePair)).scalars().all()
    }
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    for row in rows:
        key = (row.target_id, row.source_text.casefold())
        entity = existing.get(key)
        dataset_id = DATASET_BY_TARGET[row.target_id]

        if entity is not None:
            entity.dataset_id = dataset_id
            entity.source_text = row.source_text
            entity.target_text = row.target_text
            entity.validation_status = row.validation_status
            entity.validated_at = row.validated_at
            continue

        entity = LanguagePair(
            dataset_id=dataset_id,
            source_language_id=row.source_id,
            target_language_id=row.target_id,
            source_text=row.source_text,
            target_text=row.target_text,
            validation_status=row.validation_status,
            validated_at=row.validated_at,
            created_at=row.created_at or now,
        )
        if row.record_id is not None and session.get(LanguagePair, row.record_id) is None:
            entity.id = row.record_id
        session.add(entity)
        existing[key] = entity

    session.commit()


def sync_sequences(session) -> None:
    """Alinear las secuencias IDENTITY con los IDs históricos importados."""
    if get_engine().dialect.name != "postgresql":
        return
    for table in (
        "languages",
        "dataset",
        "translation_direction",
        "language_pair",
        "model_version",
        "evaluation",
        "human_evaluation",
    ):
        session.execute(
            text(
                f"SELECT setval("
                f"pg_get_serial_sequence('{table}', 'id'), "
                f"GREATEST(COALESCE((SELECT MAX(id) FROM {table}), 1), 1), "
                f"(SELECT MAX(id) FROM {table}) IS NOT NULL"
                f")"
            )
        )
    session.commit()


def verify_database(session) -> dict[str, int]:
    """Comprobar que la base quedó separada por idioma y sin filas inválidas."""
    rows = session.execute(
        select(
            Language.code,
            LanguagePair.dataset_id,
            LanguagePair.source_language_id,
            LanguagePair.target_language_id,
        )
        .join(Language, Language.id == LanguagePair.target_language_id)
    ).all()

    counts = {"piapoco": 0, "achagua": 0}
    for code, dataset_id, source_id, target_id in rows:
        if source_id != 1 or target_id not in DATASET_BY_TARGET:
            raise RuntimeError(
                f"Dirección inválida en la base: {source_id} -> {target_id}"
            )
        if code not in counts:
            raise RuntimeError(f"Idioma destino inesperado en la base: {code}")
        if dataset_id != DATASET_BY_TARGET[target_id]:
            raise RuntimeError(
                f"Mezcla de dataset para {code}: dataset_id={dataset_id}, "
                f"esperado={DATASET_BY_TARGET[target_id]}"
            )
        counts[code] += 1
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data"),
        help="Directorio con los CSV (por defecto: data/)",
    )
    parser.add_argument("--url", help="Sobrescribir DATABASE_URL")
    parser.add_argument("--dry-run", action="store_true", help="No escribir en la base")
    parser.add_argument("--reset", action="store_true", help="Vaciar language_pair antes de importar")
    parser.add_argument(
        "--allow-count-change",
        action="store_true",
        help="Permitir conteos distintos de 75 piapoco / 24 achagua",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.url:
        os.environ["DATABASE_URL"] = args.url
        reset_engine()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        print(f"No existe el directorio de datos: {data_dir}", file=sys.stderr)
        return 2

    try:
        all_rows = load_rows(data_dir)
    except ValueError as error:
        print(f"Error de validación: {error}", file=sys.stderr)
        return 1

    selected, conflicts = deduplicate(all_rows)
    conflict_path = data_dir / "import_conflicts.csv"
    print_summary(all_rows, selected, conflicts, conflict_path)
    write_conflicts(conflict_path, conflicts)

    counts = counts_by_language(selected)
    if not args.allow_count_change and counts != EXPECTED_COUNTS:
        print(
            f"\nConteos inesperados: {counts}. Se esperaban {EXPECTED_COUNTS}. "
            "Revisa los CSV o usa --allow-count-change.",
            file=sys.stderr,
        )
        return 3

    if args.dry_run:
        print("\nDry-run: no se modificó la base de datos.")
        return 0

    print(f"\nBase de datos: {get_database_url()}")
    init_db()
    seed_defaults()
    remove_session()

    session = get_session()
    try:
        ensure_reference_data(session)
        upsert_rows(session, selected, reset=args.reset)
        sync_sequences(session)
        db_counts = verify_database(session)
        print("\nImportación completada")
        print(f"  Piapoco en base: {db_counts['piapoco']}")
        print(f"  Achagua en base: {db_counts['achagua']}")
        if not args.allow_count_change and db_counts != EXPECTED_COUNTS:
            print(f"Conteos finales inesperados: {db_counts}", file=sys.stderr)
            return 4
        return 0
    except (SQLAlchemyError, RuntimeError, ValueError) as error:
        session.rollback()
        print(f"\nError durante la importación: {error}", file=sys.stderr)
        return 1
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
