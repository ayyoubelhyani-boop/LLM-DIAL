from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SentenceRecord:
    sentence_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_sentences(
    path: str | Path,
    text_col: str = "text",
    id_col: str | None = None,
    dedupe: bool = True,
) -> list[SentenceRecord]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input file does not exist: {source}")

    suffix = source.suffix.lower()
    if suffix == ".csv":
        rows = _load_csv(source)
    elif suffix == ".jsonl":
        rows = _load_jsonl(source)
    else:
        raise ValueError("Unsupported input format. Use .csv or .jsonl")

    seen_texts: set[str] = set()
    records: list[SentenceRecord] = []
    for idx, row in enumerate(rows):
        if text_col not in row:
            raise ValueError(f"Missing text column '{text_col}' in row {idx}")
        text = str(row[text_col]).strip()
        if not text:
            continue
        dedupe_key = text.casefold()
        if dedupe and dedupe_key in seen_texts:
            continue
        seen_texts.add(dedupe_key)
        sentence_id = str(row[id_col]).strip() if id_col and row.get(id_col) else f"row-{idx}"
        metadata = {key: value for key, value in row.items() if key not in {text_col, id_col}}
        records.append(SentenceRecord(sentence_id=sentence_id, text=text, metadata=metadata))
    return records


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
    return rows
