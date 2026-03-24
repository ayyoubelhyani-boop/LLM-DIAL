from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingBackend(Protocol):
    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        ...

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        ...


class TfidfEmbeddingBackend:
    def __init__(
        self,
        *,
        analyzer: str = "word",
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
        )

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.vectorizer.fit_transform(texts).toarray().astype(np.float64, copy=False)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray().astype(np.float64, copy=False)


class SentenceTransformerEmbeddingBackend:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install the "
                "'sentence-transformers' extra to use this backend."
            ) from exc
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        model_kwargs = {}
        if self.cache_dir:
            model_kwargs["cache_folder"] = str(self.cache_dir)
        self.model = SentenceTransformer(model_name, **model_kwargs)

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=False), dtype=np.float64)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=False), dtype=np.float64)


def build_embedding_backend(
    name: str,
    *,
    sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    sentence_transformer_cache_dir: str | Path | None = None,
) -> EmbeddingBackend:
    normalized = name.strip().lower()
    if normalized == "tfidf":
        return TfidfEmbeddingBackend()
    if normalized in {"sentence-transformers", "sentence_transformer", "st"}:
        return SentenceTransformerEmbeddingBackend(
            sentence_transformer_model,
            cache_dir=sentence_transformer_cache_dir,
        )
    raise ValueError(f"Unknown embedding backend: {name}")


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms
