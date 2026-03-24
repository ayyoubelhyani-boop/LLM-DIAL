from __future__ import annotations

import ast
import csv
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class BenchmarkSpec:
    dataset_path: str
    default_config: str
    text_field: str
    label_field: str


HF_BENCHMARKS: dict[str, BenchmarkSpec] = {
    "banking77": BenchmarkSpec(
        dataset_path="PolyAI/banking77",
        default_config="default",
        text_field="text",
        label_field="label",
    ),
    "clinc150": BenchmarkSpec(
        dataset_path="DeepPavlov/clinc150",
        default_config="default",
        text_field="utterance",
        label_field="label",
    ),
    "mtop": BenchmarkSpec(
        dataset_path="mteb/MTOPIntentClassification",
        default_config="en",
        text_field="text",
        label_field="label",
    ),
    "massive": BenchmarkSpec(
        dataset_path="AmazonScience/massive",
        default_config="en-US",
        text_field="utt",
        label_field="intent",
    ),
}

OFFICIAL_SAMPLE_URLS = {
    "dialin-labels": "https://raw.githubusercontent.com/mengze-hong/Dial-in-LLM/main/data/data_labels.csv",
    "dialin-goodness": "https://raw.githubusercontent.com/mengze-hong/Dial-in-LLM/main/data/sample_dialogue_intent_goodness.json",
    "dialin-label": "https://raw.githubusercontent.com/mengze-hong/Dial-in-LLM/main/data/sample_dialogue_intent_label.json",
}

_INPUT_RE = re.compile(r"input\s*[:：]\s*(\[[\s\S]*\])", re.IGNORECASE)


def export_hf_benchmark(
    benchmark: str,
    out_path: str | Path,
    *,
    split: str = "train",
    config: str | None = None,
    output_format: str = "csv",
    intent_only: bool = False,
) -> Path:
    spec = HF_BENCHMARKS.get(benchmark)
    if spec is None:
        raise ValueError(f"Unknown benchmark dataset: {benchmark}")

    dataset = _load_hf_dataset(
        spec.dataset_path,
        config or spec.default_config,
        split=split,
    )
    label_names = _resolve_label_names(dataset, benchmark, spec)

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(dataset):
        text = str(item[spec.text_field]).strip()
        if not text:
            continue
        label_value = item[spec.label_field]
        row: dict[str, Any] = {
            "sentence_id": f"{benchmark}-{split}-{idx}",
            "text": text,
            "label": label_value,
            "split": split,
            "source_dataset": benchmark,
            "source_config": config or spec.default_config,
        }
        label_text = _label_text_from_value(label_value, label_names)
        if label_text is not None:
            row["label_text"] = label_text

        if intent_only and benchmark == "clinc150" and not row.get("label_text"):
            continue

        if benchmark == "massive":
            row["locale"] = item.get("locale")
            row["partition"] = item.get("partition")
            row["scenario"] = item.get("scenario")
        rows.append(row)

    return _write_rows(rows, out_path, output_format)


def export_official_sample(
    source: str,
    out_path: str | Path,
    *,
    output_format: str = "jsonl",
    layout: str = "clusters",
) -> Path:
    if source == "dialin-labels":
        if output_format != "csv":
            raise ValueError("dialin-labels can only be exported as CSV")
        return _download_to_path(OFFICIAL_SAMPLE_URLS[source], out_path)

    if source not in {"dialin-goodness", "dialin-label"}:
        raise ValueError(f"Unknown official sample source: {source}")

    with urllib.request.urlopen(OFFICIAL_SAMPLE_URLS[source], timeout=30) as response:
        payload = json.load(response)

    if layout == "clusters":
        rows = _sample_clusters_to_rows(payload, source)
    elif layout == "utterances":
        rows = _sample_utterances_to_rows(payload, source)
    else:
        raise ValueError(f"Unknown layout: {layout}")
    return _write_rows(rows, out_path, output_format)


def _sample_clusters_to_rows(payload: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task = source.removeprefix("dialin-")
    for idx, item in enumerate(payload):
        sentences = parse_sample_input(item.get("input", ""))
        rows.append(
            {
                "sample_id": f"{task}-{idx}",
                "db_id": item.get("db_id"),
                "task": task,
                "reference_output": item.get("output"),
                "num_sentences": len(sentences),
                "sentences": sentences,
            }
        )
    return rows


def _sample_utterances_to_rows(payload: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task = source.removeprefix("dialin-")
    for idx, item in enumerate(payload):
        sentences = parse_sample_input(item.get("input", ""))
        for sentence_idx, sentence in enumerate(sentences):
            rows.append(
                {
                    "sentence_id": f"{task}-{idx}-{sentence_idx}",
                    "sample_id": f"{task}-{idx}",
                    "db_id": item.get("db_id"),
                    "task": task,
                    "reference_output": item.get("output"),
                    "text": sentence,
                }
            )
    return rows


def parse_sample_input(raw: str) -> list[str]:
    match = _INPUT_RE.search(raw)
    if not match:
        raise ValueError("Could not extract sentence list from sample input")

    parsed = ast.literal_eval(match.group(1))
    if not isinstance(parsed, list):
        raise ValueError("Sample input did not decode to a list")
    return [str(item).strip() for item in parsed if str(item).strip()]


def _download_to_path(url: str, out_path: str | Path) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read()
    target.write_bytes(data)
    return target


def _write_rows(rows: Iterable[dict[str, Any]], out_path: str | Path, output_format: str) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if output_format == "csv":
        fieldnames = _collect_fieldnames(rows)
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
        return target
    if output_format == "jsonl":
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return target
    raise ValueError("Unsupported output format. Use csv or jsonl")


def _collect_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _load_hf_dataset(dataset_path: str, config: str, *, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is not installed. Install the 'benchmarks' extra to export Hugging Face benchmarks."
        ) from exc
    try:
        return load_dataset(dataset_path, name=config, split=split)
    except RuntimeError as exc:
        # MASSIVE currently resolves through dataset scripts on some local
        # environments. Fall back to the public parquet export exposed by HF.
        if "Dataset scripts are no longer supported" in str(exc):
            if dataset_path == "AmazonScience/massive":
                return _load_massive_from_parquet(config=config, split=split)
            if dataset_path == "PolyAI/banking77":
                return _load_banking77_from_parquet(split=split)
        raise


def _load_massive_from_parquet(*, config: str, split: str):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required to load MASSIVE from parquet. Install the 'benchmarks' extra."
        ) from exc

    url = (
        "https://huggingface.co/datasets/AmazonScience/massive/resolve/"
        f"refs%2Fconvert%2Fparquet/{config}/{split}/0000.parquet"
    )
    frame = pd.read_parquet(url)
    return frame.to_dict(orient="records")


def _load_banking77_from_parquet(*, split: str):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required to load BANKING77 from parquet. Install the 'benchmarks' extra."
        ) from exc

    url = (
        "https://huggingface.co/datasets/PolyAI/banking77/resolve/"
        f"refs%2Fconvert%2Fparquet/default/{split}/0000.parquet"
    )
    frame = pd.read_parquet(url)
    return frame.to_dict(orient="records")


def _resolve_label_names(dataset, benchmark: str, spec: BenchmarkSpec) -> list[str] | None:
    feature = getattr(dataset, "features", {}).get(spec.label_field)
    names = getattr(feature, "names", None)
    if names:
        return list(names)

    if benchmark != "clinc150":
        return None

    intents_dataset = _load_hf_dataset(spec.dataset_path, "intents", split="intents")
    mapping: dict[int, str] = {}
    for item in intents_dataset:
        intent_id = item.get("id")
        name = item.get("name")
        if isinstance(intent_id, int) and isinstance(name, str):
            mapping[intent_id] = name
    if not mapping:
        return None
    max_index = max(mapping)
    return [mapping.get(idx, str(idx)) for idx in range(max_index + 1)]


def _label_text_from_value(label_value: Any, label_names: list[str] | None) -> str | None:
    if label_names is None:
        return None
    if not isinstance(label_value, int):
        return None
    if 0 <= label_value < len(label_names):
        return label_names[label_value]
    return None
