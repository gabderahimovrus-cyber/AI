from __future__ import annotations

import csv
import json
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".html", ".py", ".js", ".css", ".cpp", ".java"}


class WorkspaceFiles:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def safe_path(self, name: str) -> Path:
        candidate = (self.root / name.strip().lstrip("/\\")).resolve()
        if self.root.resolve() not in candidate.parents and candidate != self.root.resolve():
            raise ValueError("Файл должен находиться внутри рабочей папки приложения")
        if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Неподдерживаемый формат: {candidate.suffix}. Поддерживаются: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def write(self, name: str, content: str) -> Path:
        path = self.safe_path(name)
        path.write_text(content, encoding="utf-8", newline="")
        return path

    def read(self, name: str) -> str:
        return self.safe_path(name).read_text(encoding="utf-8", errors="replace")

    def list(self) -> list[Path]:
        return sorted([p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS])

    def create_json(self, name: str, data: object) -> Path:
        return self.write(name, json.dumps(data, ensure_ascii=False, indent=2))

    def create_csv(self, name: str, rows: list[dict[str, object]]) -> Path:
        path = self.safe_path(name)
        if not rows:
            path.write_text("", encoding="utf-8")
            return path
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path
