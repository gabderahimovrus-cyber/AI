from __future__ import annotations

from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import RLock


class ActionLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self.events: Queue[str] = Queue()

    def log(self, message: str) -> str:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        self.events.put(line)
        return line

    def tail(self, limit: int = 400) -> list[str]:
        if not self.path.exists():
            return []
        return self.path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
