from __future__ import annotations

from pathlib import Path
from typing import Iterable


class SentencePieceTokenizer:
    """Thin wrapper around SentencePiece BPE with a deterministic fallback error."""

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else None
        self.processor = None
        if model_path:
            self.load(model_path)

    def train(self, input_files: Iterable[str | Path], model_prefix: str | Path, vocab_size: int = 32_000, character_coverage: float = 1.0) -> Path:
        try:
            import sentencepiece as spm
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install sentencepiece to train the tokenizer.") from exc
        files = [str(Path(path)) for path in input_files]
        if not files:
            raise ValueError("At least one text file is required to train tokenizer.")
        prefix = str(model_prefix)
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)
        spm.SentencePieceTrainer.train(
            input=",".join(files),
            model_prefix=prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=character_coverage,
            bos_id=1,
            eos_id=2,
            unk_id=0,
            pad_id=3,
        )
        self.load(f"{prefix}.model")
        return Path(f"{prefix}.model")

    def load(self, model_path: str | Path) -> None:
        try:
            import sentencepiece as spm
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install sentencepiece to load the tokenizer.") from exc
        self.model_path = Path(model_path)
        self.processor = spm.SentencePieceProcessor(model_file=str(self.model_path))

    def save(self, target: str | Path) -> Path:
        if self.model_path is None or not self.model_path.exists():
            raise FileNotFoundError("Tokenizer model is not loaded.")
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.model_path.read_bytes())
        return target

    def encode(self, text: str) -> list[int]:
        if self.processor is None:
            raise RuntimeError("Tokenizer is not loaded.")
        return list(self.processor.encode(text, out_type=int))

    def decode(self, ids: Iterable[int]) -> str:
        if self.processor is None:
            raise RuntimeError("Tokenizer is not loaded.")
        return str(self.processor.decode(list(ids)))
