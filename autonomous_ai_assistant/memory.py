from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import time
from typing import Any, Iterable


@dataclass(slots=True)
class MemoryItem:
    id: int
    kind: str
    title: str
    content: str
    metadata: dict[str, Any]
    created_at: float


class MemoryStore:
    """SQLite-backed durable memory for conversations, knowledge, tasks, files and reports."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._db:
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                )
                """
            )
            self._db.execute("CREATE INDEX IF NOT EXISTS idx_memory_kind ON memory(kind)")
            self._db.execute("CREATE INDEX IF NOT EXISTS idx_memory_created ON memory(created_at)")
            self._db.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                USING fts5(title, content, content='memory', content_rowid='id')
                """
            )
            self._db.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory BEGIN
                    INSERT INTO memory_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
                END;
                CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory BEGIN
                    INSERT INTO memory_fts(memory_fts, rowid, title, content) VALUES('delete', old.id, old.title, old.content);
                END;
                CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memory BEGIN
                    INSERT INTO memory_fts(memory_fts, rowid, title, content) VALUES('delete', old.id, old.title, old.content);
                    INSERT INTO memory_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
                END;
                """
            )

    def add(self, kind: str, title: str, content: str, metadata: dict[str, Any] | None = None) -> int:
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO memory(kind, title, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                (kind, title.strip() or kind, content, metadata_json, time()),
            )
            return int(cur.lastrowid)

    def recent(self, limit: int = 20, kind: str | None = None) -> list[MemoryItem]:
        sql = "SELECT * FROM memory"
        args: list[Any] = []
        if kind:
            sql += " WHERE kind = ?"
            args.append(kind)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            return [self._row_to_item(row) for row in self._db.execute(sql, args)]

    def search(self, query: str, limit: int = 8) -> list[MemoryItem]:
        clean = " ".join(part for part in query.replace('"', ' ').split() if len(part) > 2)
        with self._lock:
            if clean:
                try:
                    rows = self._db.execute(
                        """
                        SELECT m.* FROM memory_fts f
                        JOIN memory m ON m.id = f.rowid
                        WHERE memory_fts MATCH ?
                        ORDER BY rank LIMIT ?
                        """,
                        (clean, limit),
                    ).fetchall()
                    if rows:
                        return [self._row_to_item(row) for row in rows]
                except sqlite3.OperationalError:
                    pass
            like = f"%{query[:80]}%"
            rows = self._db.execute(
                "SELECT * FROM memory WHERE title LIKE ? OR content LIKE ? ORDER BY created_at DESC LIMIT ?",
                (like, like, limit),
            ).fetchall()
            return [self._row_to_item(row) for row in rows]

    def all(self) -> list[MemoryItem]:
        with self._lock:
            return [self._row_to_item(row) for row in self._db.execute("SELECT * FROM memory ORDER BY created_at DESC")]

    def stats(self) -> dict[str, int]:
        with self._lock:
            rows = self._db.execute("SELECT kind, COUNT(*) c FROM memory GROUP BY kind").fetchall()
            total = self._db.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
        data = {row["kind"]: int(row["c"]) for row in rows}
        data["total"] = int(total)
        return data

    def export_json(self, path: str | Path) -> None:
        items = [item.__dict__ for item in self.all()]
        Path(path).write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=int(row["id"]),
            kind=row["kind"],
            title=row["title"],
            content=row["content"],
            metadata=json.loads(row["metadata"] or "{}"),
            created_at=float(row["created_at"]),
        )
