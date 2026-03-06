from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Protocol, Sequence

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class CoherenceEvaluator(Protocol):
    def coherence_eval(self, sentences: Sequence[str]) -> bool:
        ...


class ClusterNamer(Protocol):
    def name_cluster(self, sentences: Sequence[str]) -> str:
        ...


class JsonCache:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._cache: dict[str, str] = {}
        if self.path and self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                self._cache = json.load(handle)

    def get(self, key: str) -> str | None:
        return self._cache.get(key)

    def set(self, key: str, value: str) -> None:
        self._cache[key] = value
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(self._cache, handle, indent=2, sort_keys=True)
        tmp_path.replace(self.path)


class DummyCoherenceEvaluator:
    def coherence_eval(self, sentences: Sequence[str]) -> bool:
        token_counts: dict[str, int] = {}
        for sentence in sentences:
            for token in _tokenize(sentence):
                if token not in ENGLISH_STOP_WORDS and len(token) > 2:
                    token_counts[token] = token_counts.get(token, 0) + 1
        if not token_counts:
            return len(sentences) <= 2
        support = max(token_counts.values())
        threshold = max(2, int(len(sentences) * 0.4 + 0.999))
        return support >= threshold


class DummyClusterNamer:
    def name_cluster(self, sentences: Sequence[str]) -> str:
        counts: dict[str, int] = {}
        for sentence in sentences:
            for token in _tokenize(sentence):
                if token not in ENGLISH_STOP_WORDS and token.isalpha() and len(token) > 2:
                    counts[token] = counts.get(token, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if len(ranked) >= 2:
            return f"{ranked[0][0]}-{ranked[1][0]}"
        if len(ranked) == 1:
            return f"general-{ranked[0][0]}"
        return "general-request"


class OpenAICoherenceEvaluator:
    def __init__(
        self,
        *,
        model: str,
        cache: JsonCache | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.client = _build_openai_client(timeout=timeout)
        self.model = model
        self.cache = cache or JsonCache()
        self.timeout = timeout
        self.max_retries = max_retries

    def coherence_eval(self, sentences: Sequence[str]) -> bool:
        payload = "\n".join(f"- {sentence}" for sentence in sentences)
        key = _hash_payload("coherence", self.model, sentences)
        cached = self.cache.get(key)
        if cached:
            return _parse_good_bad(cached) == "good"

        prompt = (
            "You are evaluating whether a sampled cluster of customer service utterances is coherent.\n"
            "Return only one token: Good or Bad.\n"
            "Good means the utterances express one intent. Bad means the intent is mixed or unclear.\n"
            f"Sentences:\n{payload}"
        )
        response = self._call_with_retries(prompt)
        verdict = _parse_good_bad(response)
        self.cache.set(key, verdict)
        return verdict == "good"

    def _call_with_retries(self, prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    timeout=self.timeout,
                    messages=[
                        {"role": "system", "content": "Return only the requested output."},
                        {"role": "user", "content": prompt},
                    ],
                )
                return completion.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                time.sleep(min(2**attempt, 5))
        raise RuntimeError(f"OpenAI coherence evaluation failed after retries: {last_error}") from last_error


class OpenAIClusterNamer:
    LABEL_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)+$")

    def __init__(
        self,
        *,
        model: str,
        cache: JsonCache | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.client = _build_openai_client(timeout=timeout)
        self.model = model
        self.cache = cache or JsonCache()
        self.timeout = timeout
        self.max_retries = max_retries

    def name_cluster(self, sentences: Sequence[str]) -> str:
        payload = "\n".join(f"- {sentence}" for sentence in sentences)
        key = _hash_payload("name", self.model, sentences)
        cached = self.cache.get(key)
        if cached:
            return self._validate_label(cached)

        prompt = (
            "Name this customer service intent cluster with a strict lowercase action-objective label.\n"
            "Use only letters, numbers, and hyphens.\n"
            "Return only the label, for example inquire-insurance.\n"
            f"Sentences:\n{payload}"
        )
        response = self._call_with_retries(prompt)
        label = self._validate_label(response)
        self.cache.set(key, label)
        return label

    def _call_with_retries(self, prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    timeout=self.timeout,
                    messages=[
                        {"role": "system", "content": "Return only the requested output."},
                        {"role": "user", "content": prompt},
                    ],
                )
                return completion.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                time.sleep(min(2**attempt, 5))
        raise RuntimeError(f"OpenAI cluster naming failed after retries: {last_error}") from last_error

    def _validate_label(self, raw: str) -> str:
        label = raw.strip().lower()
        if not self.LABEL_RE.fullmatch(label):
            raise ValueError(f"LLM returned an invalid cluster label: {raw!r}")
        return label


def _build_openai_client(*, timeout: float) -> Any:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai is not installed. Install the 'llm' extra to enable --use-llm") from exc
    return OpenAI(api_key=api_key, timeout=timeout)


def _hash_payload(task: str, model: str, sentences: Sequence[str]) -> str:
    payload = json.dumps({"task": task, "model": model, "sentences": list(sentences)}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_good_bad(raw: str) -> str:
    normalized = raw.strip().lower()
    if normalized not in {"good", "bad"}:
        raise ValueError(f"LLM returned an invalid coherence verdict: {raw!r}")
    return normalized


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
