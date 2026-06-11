from __future__ import annotations

from pathlib import Path


class SelfImprovementPolicy:
    """Allows only knowledge, dataset and model updates inside the project data root."""

    allowed_dirs = {"memory", "datasets", "models", "vector_store", "logs"}
    forbidden_markers = {"Windows", "System32", "Program Files", "/etc", "/usr/bin", "/bin"}

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()

    def is_allowed(self, path: str | Path) -> bool:
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(self.project_root)
        except ValueError:
            return False
        if any(marker in str(resolved) for marker in self.forbidden_markers):
            return False
        return relative.parts[:1] and relative.parts[0] in self.allowed_dirs

    def ensure_allowed(self, path: str | Path) -> Path:
        if not self.is_allowed(path):
            raise PermissionError(f"Self-improvement cannot modify {path}; only local knowledge, datasets and models are allowed.")
        return Path(path)
