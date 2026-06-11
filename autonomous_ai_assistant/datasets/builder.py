from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class DatasetBuilder:
    """Builds JSONL language-model datasets with records shaped as {""" + '"text": "..."' + """}."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(self, documents: Iterable[str], name: str = "dataset.jsonl", min_chars: int = 20) -> Path:
        path = self.output_dir / name
        count = 0
        with path.open("w", encoding="utf-8") as fh:
            for text in documents:
                clean = text.strip()
                if len(clean) < min_chars:
                    continue
                fh.write(json.dumps({"text": clean}, ensure_ascii=False) + "\n")
                count += 1
        if count == 0:
            raise ValueError("Dataset is empty after filtering.")
        return path

    @staticmethod
    def read_jsonl(path: str | Path) -> list[str]:
        texts: list[str] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                texts.append(json.loads(line)["text"])
        return texts
