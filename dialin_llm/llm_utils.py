from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Protocol, Sequence

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

LABEL_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)+")


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


class LocalTransformersTextGenerator:
    def __init__(
        self,
        *,
        model: str,
        cache_dir: str | Path | None = ".hf-cache",
        device_map: str = "auto",
        quantization: str = "none",
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for the local LLM backend. "
                "Install the 'local-llm' extra or provide them in the environment."
            ) from exc

        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        quantization_config = _build_quantization_config(quantization)
        accelerate_available = _module_available("accelerate")
        sharded_device_map = _uses_sharded_device_map(device_map)
        fallback_device = _resolve_single_device(device_map=device_map, cuda_available=torch.cuda.is_available())

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if self.cache_dir:
            model_kwargs["cache_dir"] = str(self.cache_dir)

        if quantization_config is not None:
            if not accelerate_available:
                raise RuntimeError(
                    "Quantized local LLM loading requires accelerate in the current environment. "
                    "Use --local-llm-quantization none for a plain single-GPU load."
                )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = device_map if sharded_device_map else {"": fallback_device}
        else:
            if torch.cuda.is_available():
                model_kwargs["dtype"] = torch.float16
            if accelerate_available and sharded_device_map:
                model_kwargs["device_map"] = device_map

        tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if self.cache_dir:
            tokenizer_kwargs["cache_dir"] = str(self.cache_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(model, **tokenizer_kwargs)
        self.generator = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
        if quantization_config is None and (not accelerate_available or not sharded_device_map):
            self.generator = self.generator.to(fallback_device)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        import torch

        prompt = self._render_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.generator.parameters()).device
        inputs = {name: value.to(device) for name, value in inputs.items()}
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature

        with torch.inference_mode():
            output = self.generator.generate(**inputs, **generation_kwargs)
        generated = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _render_messages(self, messages: Sequence[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    list(messages),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        parts = []
        for message in messages:
            role = message.get("role", "user").strip().upper()
            content = message.get("content", "").strip()
            parts.append(f"{role}: {content}")
        parts.append("ASSISTANT:")
        return "\n\n".join(parts)


class LocalTransformersCoherenceEvaluator:
    def __init__(
        self,
        *,
        generator: LocalTransformersTextGenerator,
        cache: JsonCache | None = None,
        prompt_style: str = "simple",
    ) -> None:
        self.generator = generator
        self.cache = cache or JsonCache()
        self.prompt_style = prompt_style

    def coherence_eval(self, sentences: Sequence[str]) -> bool:
        key = _hash_payload("local-coherence", f"{self.generator.model}:{self.prompt_style}", sentences)
        cached = self.cache.get(key)
        if cached:
            return cached == "good"

        response = self.generator.generate(_build_coherence_messages(sentences, prompt_style=self.prompt_style))
        try:
            verdict = _parse_good_bad_loose(response)
        except ValueError:
            retry_response = self.generator.generate(
                _build_coherence_retry_messages(sentences, prompt_style=self.prompt_style)
            )
            verdict = _parse_good_bad_loose_or_default_bad(retry_response)
        self.cache.set(key, verdict)
        return verdict == "good"


class LocalTransformersClusterNamer:
    def __init__(
        self,
        *,
        generator: LocalTransformersTextGenerator,
        cache: JsonCache | None = None,
        prompt_style: str = "simple",
    ) -> None:
        self.generator = generator
        self.cache = cache or JsonCache()
        self.prompt_style = prompt_style

    def name_cluster(self, sentences: Sequence[str]) -> str:
        key = _hash_payload("local-name", f"{self.generator.model}:{self.prompt_style}", sentences)
        cached = self.cache.get(key)
        if cached:
            return _extract_label(cached)

        response = self.generator.generate(_build_naming_messages(sentences, prompt_style=self.prompt_style))
        label = _extract_label(response)
        self.cache.set(key, label)
        return label


class OpenAICoherenceEvaluator:
    def __init__(
        self,
        *,
        model: str,
        cache: JsonCache | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        prompt_style: str = "simple",
    ) -> None:
        self.client = _build_openai_client(timeout=timeout)
        self.model = model
        self.cache = cache or JsonCache()
        self.timeout = timeout
        self.max_retries = max_retries
        self.prompt_style = prompt_style

    def coherence_eval(self, sentences: Sequence[str]) -> bool:
        key = _hash_payload("coherence", f"{self.model}:{self.prompt_style}", sentences)
        cached = self.cache.get(key)
        if cached:
            return _parse_good_bad(cached) == "good"

        response = self._call_with_retries(_build_coherence_messages(sentences, prompt_style=self.prompt_style))
        verdict = _parse_good_bad(response)
        self.cache.set(key, verdict)
        return verdict == "good"

    def _call_with_retries(self, messages: Sequence[dict[str, str]]) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    timeout=self.timeout,
                    messages=list(messages),
                )
                return completion.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                time.sleep(min(2**attempt, 5))
        raise RuntimeError(f"OpenAI coherence evaluation failed after retries: {last_error}") from last_error


class OpenAIClusterNamer:
    def __init__(
        self,
        *,
        model: str,
        cache: JsonCache | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        prompt_style: str = "simple",
    ) -> None:
        self.client = _build_openai_client(timeout=timeout)
        self.model = model
        self.cache = cache or JsonCache()
        self.timeout = timeout
        self.max_retries = max_retries
        self.prompt_style = prompt_style

    def name_cluster(self, sentences: Sequence[str]) -> str:
        key = _hash_payload("name", f"{self.model}:{self.prompt_style}", sentences)
        cached = self.cache.get(key)
        if cached:
            return _validate_label(cached)

        response = self._call_with_retries(_build_naming_messages(sentences, prompt_style=self.prompt_style))
        label = _validate_label(response)
        self.cache.set(key, label)
        return label

    def _call_with_retries(self, messages: Sequence[dict[str, str]]) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    timeout=self.timeout,
                    messages=list(messages),
                )
                return completion.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                time.sleep(min(2**attempt, 5))
        raise RuntimeError(f"OpenAI cluster naming failed after retries: {last_error}") from last_error


def _build_openai_client(*, timeout: float) -> Any:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai is not installed. Install the 'llm' extra to enable --use-llm") from exc
    return OpenAI(api_key=api_key, timeout=timeout)


def _module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def _uses_sharded_device_map(device_map: str) -> bool:
    return device_map.strip().lower() in {"auto", "balanced", "balanced_low_0", "sequential"}


def _resolve_single_device(*, device_map: str, cuda_available: bool) -> str:
    normalized = device_map.strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized.startswith("cuda"):
        return device_map
    if normalized.isdigit():
        return f"cuda:{normalized}" if cuda_available else "cpu"
    return "cuda" if cuda_available else "cpu"


def _build_quantization_config(mode: str) -> Any | None:
    normalized = mode.strip().lower()
    if normalized == "none":
        return None
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "bitsandbytes quantization requires transformers, torch, accelerate, and bitsandbytes."
        ) from exc
    if normalized == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    if normalized == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unsupported local LLM quantization mode: {mode}")


def _hash_payload(task: str, model: str, sentences: Sequence[str]) -> str:
    payload = json.dumps({"task": task, "model": model, "sentences": list(sentences)}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_good_bad(raw: str) -> str:
    normalized = raw.strip().lower()
    if normalized not in {"good", "bad"}:
        raise ValueError(f"LLM returned an invalid coherence verdict: {raw!r}")
    return normalized


def _parse_good_bad_loose(raw: str) -> str:
    normalized = raw.strip().lower()
    if normalized in {"good", "bad"}:
        return normalized
    match = re.search(r"\b(good|bad)\b", normalized)
    if match:
        return match.group(1)
    raise ValueError(f"Local LLM returned an invalid coherence verdict: {raw!r}")


def _parse_good_bad_loose_or_default_bad(raw: str) -> str:
    try:
        return _parse_good_bad_loose(raw)
    except ValueError:
        normalized = raw.strip().lower()
        if any(token in normalized for token in {"maybe", "unclear", "unsure", "not sure", "cannot determine"}):
            return "bad"
        return "bad"


def _validate_label(raw: str) -> str:
    label = _coerce_label(raw)
    if not LABEL_RE.fullmatch(label):
        raise ValueError(f"LLM returned an invalid cluster label: {raw!r}")
    return label


def _extract_label(raw: str) -> str:
    direct_match = LABEL_RE.search(raw.strip().lower().replace("_", "-"))
    if direct_match:
        return direct_match.group(0)

    normalized = _coerce_label(raw)
    if LABEL_RE.fullmatch(normalized):
        return normalized
    raise ValueError(f"Local LLM returned an invalid cluster label: {raw!r}")


def _coerce_label(raw: str) -> str:
    normalized = raw.strip().lower().replace("_", "-")
    if LABEL_RE.fullmatch(normalized):
        return normalized

    tokens = re.findall(r"[a-z0-9]+", normalized)
    if len(tokens) >= 2:
        return f"{tokens[0]}-{tokens[1]}"
    if len(tokens) == 1:
        return f"general-{tokens[0]}"
    return normalized


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_coherence_messages(sentences: Sequence[str], *, prompt_style: str) -> list[dict[str, str]]:
    payload = "\n".join(f"- {sentence}" for sentence in sentences)
    normalized = prompt_style.strip().lower()
    if normalized == "benchmark":
        return [
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator for intent clustering quality. "
                    "Follow the rubric exactly and answer with one token only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Task: decide whether all utterances express the same customer intent.\n"
                    "Return exactly one token: Good or Bad.\n"
                    "Good: the utterances ask for the same underlying action or objective, even with wording variation.\n"
                    "Bad: the utterances mix different actions, different objectives, or only share a broad topic.\n"
                    "Example A:\n"
                    "- refund my order\n"
                    "- get a refund for the purchase\n"
                    "- request order refund\n"
                    "Answer: Good\n"
                    "Example B:\n"
                    "- refund my order\n"
                    "- cancel my subscription\n"
                    "- where is my package\n"
                    "Answer: Bad\n"
                    "Now evaluate this cluster.\n"
                    f"Sentences:\n{payload}\n"
                    "Answer:"
                ),
            },
        ]
    return [
        {"role": "system", "content": "Return only the requested output."},
        {
            "role": "user",
            "content": (
                "You are evaluating whether a sampled cluster of customer service utterances is coherent.\n"
                "Return only one token: Good or Bad.\n"
                "Good means the utterances express one intent. Bad means the intent is mixed or unclear.\n"
                f"Sentences:\n{payload}"
            ),
        },
    ]


def _build_coherence_retry_messages(sentences: Sequence[str], *, prompt_style: str) -> list[dict[str, str]]:
    payload = "\n".join(f"- {sentence}" for sentence in sentences)
    normalized = prompt_style.strip().lower()
    if normalized == "benchmark":
        return [
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator for intent clustering quality. "
                    "Reply with exactly one token."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Your previous answer was invalid.\n"
                    "Return exactly one token: Good or Bad.\n"
                    "If there is any uncertainty, return Bad.\n"
                    f"Sentences:\n{payload}\n"
                    "Answer:"
                ),
            },
        ]
    return [
        {"role": "system", "content": "Return exactly one token."},
        {
            "role": "user",
            "content": (
                "Your previous answer was invalid.\n"
                "Return exactly one token: Good or Bad.\n"
                "If you are unsure, return Bad.\n"
                f"Sentences:\n{payload}\n"
                "Answer:"
            ),
        },
    ]


def _build_naming_messages(sentences: Sequence[str], *, prompt_style: str) -> list[dict[str, str]]:
    payload = "\n".join(f"- {sentence}" for sentence in sentences)
    normalized = prompt_style.strip().lower()
    if normalized == "benchmark":
        return [
            {
                "role": "system",
                "content": (
                    "You name customer service intent clusters. "
                    "Return exactly one lowercase hyphenated label and nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Task: produce a concise action-objective label for this intent cluster.\n"
                    "Rules:\n"
                    "- use only lowercase letters, numbers, and hyphens\n"
                    "- prefer verb-object labels such as refund-order or track-package\n"
                    "- avoid generic labels such as issue-problem or customer-service\n"
                    "- if the intent is about information seeking, use a verb like check, ask, verify, or track\n"
                    "Example A:\n"
                    "- refund my order\n"
                    "- get a refund for the purchase\n"
                    "Answer: refund-order\n"
                    "Example B:\n"
                    "- where is my package\n"
                    "- track package status\n"
                    "Answer: track-package\n"
                    "Now label this cluster.\n"
                    f"Sentences:\n{payload}\n"
                    "Answer:"
                ),
            },
        ]
    return [
        {"role": "system", "content": "Return only the requested output."},
        {
            "role": "user",
            "content": (
                "Name this customer service intent cluster with a strict lowercase action-objective label.\n"
                "Use only letters, numbers, and hyphens.\n"
                "Return only the label, for example inquire-insurance.\n"
                f"Sentences:\n{payload}"
            ),
        },
    ]
