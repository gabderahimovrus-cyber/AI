from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .transformer import DecoderOnlyTransformer, TransformerConfig, torch

MODEL_SLOTS = ("base", "candidate", "production", "archive")


@dataclass(slots=True)
class ModelVersionInfo:
    version: str
    created: str
    loss: float | None = None
    perplexity: float | None = None
    slot: str = "candidate"
    parameters: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelManager:
    """Stores, loads and promotes local model versions."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        for slot in MODEL_SLOTS:
            (self.root / slot).mkdir(parents=True, exist_ok=True)

    def slot_dir(self, slot: str) -> Path:
        self._validate_slot(slot)
        return self.root / slot

    def save_model(self, model: DecoderOnlyTransformer, slot: str, info: ModelVersionInfo | None = None) -> Path:
        if torch is None:
            raise ImportError("PyTorch is required to save model weights.")
        self._validate_slot(slot)
        target = self.slot_dir(slot)
        target.mkdir(parents=True, exist_ok=True)
        config = model.config.to_dict()
        (target / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        torch.save(model.state_dict(), target / "model.pt")
        info = info or ModelVersionInfo(version=datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"), created=datetime.now(timezone.utc).isoformat(), slot=slot)
        info.slot = slot
        info.parameters = getattr(model, "parameter_count", None)
        (target / "metadata.json").write_text(json.dumps(info.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def load_model(self, slot: str, map_location: str | None = None) -> DecoderOnlyTransformer:
        if torch is None:
            raise ImportError("PyTorch is required to load model weights.")
        self._validate_slot(slot)
        target = self.slot_dir(slot)
        config_path = target / "config.json"
        weights_path = target / "model.pt"
        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError(f"No model checkpoint found in {target}")
        config = TransformerConfig(**json.loads(config_path.read_text(encoding="utf-8")))
        model = DecoderOnlyTransformer(config)
        model.load_state_dict(torch.load(weights_path, map_location=map_location or self.device()))
        return model

    def save_checkpoint(self, state: dict[str, Any], slot: str = "candidate", name: str = "checkpoint.pt") -> Path:
        if torch is None:
            raise ImportError("PyTorch is required to save checkpoints.")
        path = self.slot_dir(slot) / name
        torch.save(state, path)
        return path

    def load_checkpoint(self, slot: str = "candidate", name: str = "checkpoint.pt", map_location: str | None = None) -> dict[str, Any]:
        if torch is None:
            raise ImportError("PyTorch is required to load checkpoints.")
        return torch.load(self.slot_dir(slot) / name, map_location=map_location or self.device())

    def promote_candidate(self) -> None:
        production = self.slot_dir("production")
        candidate = self.slot_dir("candidate")
        if any(production.iterdir()):
            archive = self.slot_dir("archive") / datetime.now(timezone.utc).strftime("production_%Y%m%d_%H%M%S")
            shutil.copytree(production, archive)
        self._replace_dir(candidate, production)
        candidate.mkdir(parents=True, exist_ok=True)

    def rollback_candidate(self) -> None:
        shutil.rmtree(self.slot_dir("candidate"), ignore_errors=True)
        self.slot_dir("candidate").mkdir(parents=True, exist_ok=True)

    def metadata(self, slot: str = "production") -> dict[str, Any]:
        path = self.slot_dir(slot) / "metadata.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"slot": slot, "version": "not-created"}

    @staticmethod
    def device() -> str:
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    @staticmethod
    def _replace_dir(src: Path, dst: Path) -> None:
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)

    @staticmethod
    def _validate_slot(slot: str) -> None:
        if slot not in MODEL_SLOTS:
            raise ValueError(f"Unsupported model slot {slot!r}. Use one of {MODEL_SLOTS}.")
