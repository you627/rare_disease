from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    text = re.sub(r"\s+", " ", text.strip().lower())
    # Keep simple for multilingual text: token by chars + words.
    words = re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text)
    return words


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    field: str
    text: str
    source_url: str


class HybridRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        tokenized = [_tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf.fit_transform(texts)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q_tokens = _tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(q_tokens), dtype=float)
        q_vec = self.tfidf.transform([query])
        tfidf_scores = (self.tfidf_matrix @ q_vec.T).toarray().reshape(-1)

        def _norm(x: np.ndarray) -> np.ndarray:
            if x.size == 0:
                return x
            mx, mn = float(x.max()), float(x.min())
            if mx == mn:
                return np.zeros_like(x)
            return (x - mn) / (mx - mn)

        fused = 0.55 * _norm(bm25_scores) + 0.45 * _norm(tfidf_scores)
        idx = np.argsort(-fused)[:top_k]
        return [
            {
                "score": float(fused[i]),
                "chunk_id": self.chunks[i].chunk_id,
                "doc_id": self.chunks[i].doc_id,
                "title": self.chunks[i].title,
                "field": self.chunks[i].field,
                "text": self.chunks[i].text,
                "source_url": self.chunks[i].source_url,
            }
            for i in idx
        ]

    def save(self, path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "HybridRetriever":
        with path.open("rb") as f:
            return pickle.load(f)


def save_chunks_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
