from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..datasets.builder import DatasetBuilder
from ..internet.collector import DataCleaner, DataCollector
from ..models.manager import ModelManager
from ..self_improvement.policy import SelfImprovementPolicy
from ..vector_store.faiss_store import VectorStore


@dataclass(slots=True)
class LearningCycleResult:
    accepted: bool
    previous_perplexity: float | None
    candidate_perplexity: float | None
    dataset_path: str
    report_path: str


class LearningAgent:
    """Autonomous collect → clean → dataset → train → evaluate → promote/rollback loop."""

    def __init__(self, root: str | Path, trainer=None, evaluator=None):
        self.root = Path(root)
        self.collector = DataCollector()
        self.cleaner = DataCleaner()
        self.datasets = DatasetBuilder(self.root / "datasets")
        self.models = ModelManager(self.root / "models")
        self.vector_store = VectorStore(self.root / "vector_store")
        self.policy = SelfImprovementPolicy(self.root)
        self.trainer = trainer
        self.evaluator = evaluator
        (self.root / "logs").mkdir(parents=True, exist_ok=True)

    def run_cycle(self, urls: Iterable[str] = (), local_paths: Iterable[str | Path] = (), max_train_epochs: int = 1) -> LearningCycleResult:
        raw_docs = []
        for url in urls:
            raw_docs.append(self.collector.fetch_url(url))
        raw_docs.extend(self.collector.import_local_documents(local_paths))
        documents = self.cleaner.clean_documents(raw_docs)
        dataset_path = self.policy.ensure_allowed(self.datasets.build(documents))
        self.vector_store.index(documents)

        previous = self.models.metadata("production")
        previous_ppl = previous.get("perplexity")
        candidate_ppl = None
        if self.trainer and self.evaluator:
            result = self.trainer.train(documents, epochs=max_train_epochs)
            self.trainer.save_model("candidate", loss=float(result["loss"]))
            report = self.evaluator.full_report(self.trainer.model, documents, self.root / "logs" / "evaluation.json")
            candidate_ppl = float(report["perplexity"])
            if previous_ppl is None or candidate_ppl < float(previous_ppl):
                self.models.promote_candidate()
                accepted = True
            else:
                self.models.rollback_candidate()
                accepted = False
        else:
            accepted = False

        result = LearningCycleResult(accepted, float(previous_ppl) if previous_ppl is not None else None, candidate_ppl, str(dataset_path), str(self.root / "logs" / "learning_cycle.json"))
        Path(result.report_path).write_text(json.dumps(result.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
