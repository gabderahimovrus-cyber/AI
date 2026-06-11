import json

from autonomous_ai_assistant.datasets import DatasetBuilder
from autonomous_ai_assistant.internet import DataCleaner, RawDocument
from autonomous_ai_assistant.memory import MemoryStore
from autonomous_ai_assistant.models import TransformerConfig
from autonomous_ai_assistant.models.manager import ModelManager
from autonomous_ai_assistant.self_improvement.policy import SelfImprovementPolicy
from autonomous_ai_assistant.vector_store import VectorStore


def test_data_cleaner_and_dataset_builder(tmp_path):
    docs = [
        RawDocument("inline", "<html><script>x()</script><p>Python is useful for automation and data analysis.</p></html>", "html"),
        RawDocument("dup", "Python is useful for automation and data analysis.", "txt"),
        RawDocument("short", "tiny", "txt"),
    ]
    cleaned = DataCleaner().clean_documents(docs, min_chars=20)
    assert cleaned == ["Python is useful for automation and data analysis."]

    dataset = DatasetBuilder(tmp_path).build(cleaned)
    record = json.loads(dataset.read_text(encoding="utf-8").strip())
    assert record == {"text": cleaned[0]}


def test_memory_types_and_vector_search(tmp_path):
    memory = MemoryStore(tmp_path / "memory.sqlite3")
    memory.add_episodic("user", "Привет", {"role": "user"})
    memory.add_semantic("Python", "Python is a programming language")
    memory.add_learning("run-1", "loss=2.0", {"perplexity": 7.4})

    assert memory.by_memory_type("episodic")[0].kind == "dialog"
    assert memory.by_memory_type("semantic")[0].kind == "knowledge"
    assert memory.by_memory_type("learning")[0].metadata["perplexity"] == 7.4

    store = VectorStore(tmp_path / "vector_store")
    store.index(["Python automates routine tasks", "Newton studied gravity"])
    assert "Python" in store.rag_context("automation with Python")


def test_model_manager_metadata_and_policy(tmp_path):
    manager = ModelManager(tmp_path / "models")
    assert manager.metadata("production")["version"] == "not-created"
    assert manager.device() in {"cpu", "cuda"}

    config = TransformerConfig()
    assert config.vocab_size == 32_000
    assert config.hidden_size == 256
    assert config.num_layers == 6
    assert config.num_heads == 8
    assert config.context_length == 512

    policy = SelfImprovementPolicy(tmp_path)
    assert policy.is_allowed(tmp_path / "datasets" / "dataset.jsonl")
    assert not policy.is_allowed(tmp_path.parent / "outside.txt")
