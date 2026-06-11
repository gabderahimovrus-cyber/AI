from autonomous_ai_assistant.engine import AssistantEngine
from autonomous_ai_assistant.web_learning import Source


class FakeLearner:
    def search(self, topic, limit=6):
        return [Source("Python guide", "https://example.com/python", "Python is a programming language.")]

    def read_sources(self, sources, max_chars_per_source=4500):
        for source in sources:
            source.text = "Python is a programming language. It is used for automation, data analysis and web services. Practice projects help learning."
        return sources

    def summarize(self, topic, sources):
        return "Резюме по теме «Python»: Python используется для автоматизации и анализа данных.", ["Python используется для автоматизации"]


def test_chat_persists_dialog_and_knowledge(tmp_path):
    engine = AssistantEngine(tmp_path)
    engine.learner = FakeLearner()

    answer = engine.chat("Изучи Python")

    assert "Резюме" in answer
    assert engine.memory.stats()["dialog"] == 2
    assert engine.memory.stats()["knowledge"] == 1
    assert engine.files.list()

    restarted = AssistantEngine(tmp_path)
    assert restarted.memory.stats()["knowledge"] == 1
    assert restarted.memory.search("Python")


def test_code_and_file_creation(tmp_path):
    engine = AssistantEngine(tmp_path)

    code_answer = engine.chat("Напиши программу на Python")
    file_answer = engine.chat("Создай файл plan.md")

    assert "main.py" in code_answer
    assert "plan.md" in file_answer
    names = {path.name for path in engine.files.list()}
    assert {"main.py", "plan.md"}.issubset(names)


def test_settings_are_saved(tmp_path):
    engine = AssistantEngine(tmp_path)
    engine.settings["resource_level"] = "высокая нагрузка"
    engine.save_settings()

    restarted = AssistantEngine(tmp_path)
    assert restarted.settings["resource_level"] == "высокая нагрузка"


def test_chat_uses_lively_fallback_and_remembers_name(tmp_path):
    engine = AssistantEngine(tmp_path)
    engine.settings["llm_backend"] = "off"

    first = engine.chat("Привет, меня зовут Алекс")
    second = engine.chat("Как ты?")

    assert "Алекс" in first
    assert "теплее" in second or "готов" in second
    user_items = [item for item in engine.memory.recent(10, "dialog") if item.metadata.get("role") == "user"]
    assert any(item.metadata.get("user_name") == "Алекс" for item in user_items)


def test_llm_settings_are_saved(tmp_path):
    engine = AssistantEngine(tmp_path)
    engine.settings["llm_backend"] = "ollama"
    engine.settings["llm_model"] = "llama3.2"
    engine.save_settings()

    restarted = AssistantEngine(tmp_path)
    assert restarted.settings["llm_backend"] == "ollama"
    assert restarted.settings["llm_model"] == "llama3.2"


def test_learning_progress_and_llm_coach_are_recorded(tmp_path):
    engine = AssistantEngine(tmp_path)
    engine.learner = FakeLearner()
    engine.settings["llm_backend"] = "off"

    answer = engine.learn("Python")
    progress = engine.learning_progress()
    knowledge = engine.memory.recent(1, "knowledge")[0]

    assert "Подсказки языковой модели-наставника" in answer
    assert progress
    assert progress[0]["percent"] == 100
    assert progress[0]["status"] == "готово"
    assert knowledge.metadata["llm_learning_coach"] == "fallback"


def test_self_modification_creates_workspace_plan(tmp_path):
    engine = AssistantEngine(tmp_path)
    engine.settings["llm_backend"] = "off"

    answer = engine.chat("Самомодифицируй себя и предложи улучшение")

    assert "план самомодификации" in answer.lower()
    names = {str(path.relative_to(engine.files.root)) for path in engine.files.list()}
    assert any(name.startswith("self_mods/self_improvement_") and name.endswith(".md") for name in names)
    assert engine.memory.stats()["self_modification"] == 1
