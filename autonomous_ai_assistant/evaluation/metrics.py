from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Iterable

from ..models.transformer import torch

DEFAULT_TEST_QUESTIONS = ["Кто такой Ньютон?", "Что такое Python?", "Объясни цикл for"]


class Evaluator:
    def __init__(self, tokenizer, device: str | None = None):
        if torch is None:
            raise ImportError("PyTorch is required for evaluation.")
        self.tokenizer = tokenizer
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    @torch.no_grad()
    def validation_loss(self, model, texts: Iterable[str], context_length: int = 512) -> float:
        model.eval().to(self.device)
        losses: list[float] = []
        for text in texts:
            ids = self.tokenizer.encode(text)[: context_length + 1]
            if len(ids) < 3:
                continue
            x = torch.tensor([ids[:-1]], dtype=torch.long, device=self.device)
            y = torch.tensor([ids[1:]], dtype=torch.long, device=self.device)
            loss = model(x, labels=y)["loss"]
            losses.append(float(loss.cpu()))
        return sum(losses) / max(1, len(losses))

    @staticmethod
    def perplexity(loss: float) -> float:
        return float(math.exp(min(loss, 20)))

    @torch.no_grad()
    def generation_speed(self, model, prompt: str = "Привет", tokens: int = 32) -> float:
        model.eval().to(self.device)
        ids = torch.tensor([self.tokenizer.encode(prompt) or [1]], dtype=torch.long, device=self.device)
        started = time.time()
        model.generate(ids, max_new_tokens=tokens)
        elapsed = max(time.time() - started, 1e-6)
        return tokens / elapsed

    def gpu_memory_mb(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    @torch.no_grad()
    def run_test_questions(self, model, questions: Iterable[str] = DEFAULT_TEST_QUESTIONS, max_new_tokens: int = 64, output_path: str | Path | None = None) -> list[dict[str, str]]:
        model.eval().to(self.device)
        answers: list[dict[str, str]] = []
        for question in questions:
            ids = torch.tensor([self.tokenizer.encode(question) or [1]], dtype=torch.long, device=self.device)
            out = model.generate(ids, max_new_tokens=max_new_tokens)[0].detach().cpu().tolist()
            answers.append({"question": question, "answer": self.tokenizer.decode(out)})
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
        return answers

    def full_report(self, model, validation_texts: Iterable[str], output_path: str | Path | None = None) -> dict[str, float]:
        loss = self.validation_loss(model, validation_texts, getattr(model.config, "context_length", 512))
        report = {"validation_loss": loss, "perplexity": self.perplexity(loss), "generation_tokens_per_sec": self.generation_speed(model), "gpu_memory_mb": self.gpu_memory_mb()}
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
