from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable

from ..models.manager import ModelManager, ModelVersionInfo
from ..models.transformer import DecoderOnlyTransformer, TransformerConfig, torch


class TextTokenDataset:
    def __init__(self, texts: Iterable[str], tokenizer, context_length: int):
        if torch is None:
            raise ImportError("PyTorch is required for training.")
        self.samples: list[torch.Tensor] = []
        for text in texts:
            ids = tokenizer.encode(text)
            for start in range(0, max(0, len(ids) - 1), context_length):
                chunk = ids[start : start + context_length + 1]
                if len(chunk) > 2:
                    self.samples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        return chunk[:-1], chunk[1:]


def collate_batch(batch):
    max_len = max(x.size(0) for x, _ in batch)
    inputs = torch.full((len(batch), max_len), 3, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        inputs[i, : x.size(0)] = x
        labels[i, : y.size(0)] = y
    return inputs, labels


class Trainer:
    def __init__(
        self,
        model_dir: str | Path,
        tokenizer,
        config: TransformerConfig | None = None,
        log_path: str | Path = "logs/training.log",
        learning_rate: float = 3e-4,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 8,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for Trainer.")
        self.manager = ModelManager(model_dir)
        self.tokenizer = tokenizer
        self.config = config or TransformerConfig()
        self.device = torch.device(ModelManager.device())
        self.model = DecoderOnlyTransformer(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.global_step = 0
        self.epoch = 0
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=self.log_path, level=logging.INFO, format="%(asctime)s %(message)s", force=True)

    def train(self, texts: Iterable[str], epochs: int = 1) -> dict[str, float | int | str]:
        from torch.utils.data import DataLoader

        dataset = TextTokenDataset(texts, self.tokenizer, self.config.context_length)
        if len(dataset) == 0:
            raise ValueError("No tokenized samples available for training.")
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_batch)
        started = time.time()
        last_loss = 0.0
        self.model.train()
        for _ in range(epochs):
            self.epoch += 1
            self.optimizer.zero_grad(set_to_none=True)
            for micro_step, (x, y) in enumerate(loader, start=1):
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x, labels=y)
                loss = output["loss"] / self.gradient_accumulation_steps
                loss.backward()
                last_loss = float(loss.detach().cpu() * self.gradient_accumulation_steps)
                if micro_step % self.gradient_accumulation_steps == 0 or micro_step == len(loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    logging.info(json.dumps({"loss": last_loss, "learning_rate": self.learning_rate, "step": self.global_step, "epoch": self.epoch, "elapsed_sec": round(time.time() - started, 2)}, ensure_ascii=False))
        return {"loss": last_loss, "step": self.global_step, "epoch": self.epoch, "device": str(self.device), "elapsed_sec": round(time.time() - started, 2)}

    def resume(self, checkpoint: str | Path | None = None) -> None:
        self.load_checkpoint(checkpoint)

    def save_checkpoint(self, checkpoint: str | Path | None = None) -> Path:
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "global_step": self.global_step, "epoch": self.epoch, "config": self.config.to_dict()}
        if checkpoint:
            path = Path(checkpoint)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, path)
            return path
        return self.manager.save_checkpoint(state)

    def load_checkpoint(self, checkpoint: str | Path | None = None) -> None:
        state = torch.load(Path(checkpoint), map_location=self.device) if checkpoint else self.manager.load_checkpoint(map_location=str(self.device))
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.global_step = int(state.get("global_step", 0))
        self.epoch = int(state.get("epoch", 0))

    def save_model(self, slot: str = "candidate", loss: float | None = None, perplexity: float | None = None) -> Path:
        info = ModelVersionInfo(version=f"step-{self.global_step}", created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), loss=loss, perplexity=perplexity, slot=slot)
        return self.manager.save_model(self.model, slot, info)
