from __future__ import annotations

import html
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class RawDocument:
    source: str
    text: str
    kind: str


class DataCollector:
    USER_AGENT = "AutonomousAIAssistant/1.0 local-data-collector"

    def fetch_url(self, url: str, timeout: int = 20) -> RawDocument:
        req = urllib.request.Request(url, headers={"User-Agent": self.USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content_type = response.headers.get("content-type", "")
            raw = response.read(3_000_000)
        text = raw.decode(self._charset(content_type), errors="replace")
        kind = "html" if "html" in content_type or url.lower().endswith((".html", "/")) else "txt"
        return RawDocument(url, text, kind)

    def load_text_file(self, path: str | Path) -> RawDocument:
        path = Path(path)
        return RawDocument(str(path), path.read_text(encoding="utf-8", errors="replace"), path.suffix.lower().lstrip(".") or "txt")

    def import_local_documents(self, paths: Iterable[str | Path]) -> list[RawDocument]:
        docs: list[RawDocument] = []
        for item in paths:
            path = Path(item)
            if path.is_dir():
                for child in path.rglob("*"):
                    if child.suffix.lower() in {".txt", ".md", ".html", ".htm", ".pdf"}:
                        docs.append(self._load_any(child))
            else:
                docs.append(self._load_any(path))
        return docs

    def _load_any(self, path: Path) -> RawDocument:
        if path.suffix.lower() == ".pdf":
            return RawDocument(str(path), self._read_pdf(path), "pdf")
        return self.load_text_file(path)

    def _read_pdf(self, path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install pypdf to import PDF files.") from exc
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)

    @staticmethod
    def _charset(content_type: str) -> str:
        match = re.search(r"charset=([^;]+)", content_type)
        return match.group(1) if match else "utf-8"


class DataCleaner:
    def clean_documents(self, documents: Iterable[RawDocument | str], min_chars: int = 80) -> list[str]:
        seen: set[str] = set()
        cleaned: list[str] = []
        for doc in documents:
            text = doc.text if isinstance(doc, RawDocument) else str(doc)
            if isinstance(doc, RawDocument) and doc.kind in {"html", "htm"}:
                text = self.remove_html_noise(text)
            text = self.normalize_text(text)
            fingerprint = text.lower()[:1000]
            if len(text) >= min_chars and fingerprint not in seen:
                seen.add(fingerprint)
                cleaned.append(text)
        return cleaned

    @staticmethod
    def remove_html_noise(text: str) -> str:
        text = re.sub(r"(?is)<(script|style|noscript|svg).*?>.*?</\1>", " ", text)
        text = re.sub(r"(?is)<br\s*/?>", "\n", text)
        text = re.sub(r"(?is)</(p|div|li|h[1-6]|section|article)>", "\n", text)
        text = re.sub(r"<[^>]+>", " ", text)
        return html.unescape(text)

    @staticmethod
    def normalize_text(text: str) -> str:
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.replace("\r", "\n").split("\n")]
        lines = [line for line in lines if line]
        return "\n".join(lines).strip()
