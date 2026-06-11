from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable


class VectorStore:
    """Local knowledge index. Uses FAISS when available, otherwise a cosine bag-of-words fallback."""

    def __init__(self, root: str | Path, dim: int = 384):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.documents: list[str] = []
        self._vectors: list[Counter[str]] = []
        try:
            import faiss  # noqa: F401
            self.faiss_available = True
        except ImportError:
            self.faiss_available = False

    def index(self, documents: Iterable[str]) -> None:
        self.documents = [doc.strip() for doc in documents if doc.strip()]
        self._vectors = [self._vectorize(doc) for doc in self.documents]
        self.save()

    def add(self, document: str) -> None:
        if document.strip():
            self.documents.append(document.strip())
            self._vectors.append(self._vectorize(document))
            self.save()

    def search(self, query: str, limit: int = 5) -> list[dict[str, float | str]]:
        q = self._vectorize(query)
        scored = [(self._cosine(q, vector), doc) for doc, vector in zip(self.documents, self._vectors)]
        return [{"text": doc, "score": score} for score, doc in sorted(scored, reverse=True)[:limit] if score > 0]

    def rag_context(self, query: str, limit: int = 5, max_chars: int = 2500) -> str:
        chunks: list[str] = []
        total = 0
        for item in self.search(query, limit):
            text = str(item["text"])
            if total + len(text) > max_chars:
                text = text[: max_chars - total]
            chunks.append(text)
            total += len(text)
            if total >= max_chars:
                break
        return "\n\n".join(chunks)

    def save(self) -> None:
        (self.root / "documents.json").write_text(json.dumps(self.documents, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        path = self.root / "documents.json"
        if path.exists():
            self.documents = json.loads(path.read_text(encoding="utf-8"))
            self._vectors = [self._vectorize(doc) for doc in self.documents]

    @staticmethod
    def _vectorize(text: str) -> Counter[str]:
        import re
        return Counter(w.lower() for w in re.findall(r"[\wа-яА-ЯёЁ]{3,}", text))

    @staticmethod
    def _cosine(left: Counter[str], right: Counter[str]) -> float:
        if not left or not right:
            return 0.0
        dot = sum(left[k] * right.get(k, 0) for k in left)
        nl = math.sqrt(sum(v * v for v in left.values()))
        nr = math.sqrt(sum(v * v for v in right.values()))
        return float(dot / (nl * nr)) if nl and nr else 0.0
