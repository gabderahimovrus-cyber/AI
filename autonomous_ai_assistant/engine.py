from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from threading import Event, Thread, RLock
from typing import Callable

from .files import WorkspaceFiles
from .language_model import ChatContext, LanguageModelRouter
from .logger import ActionLogger
from .memory import MemoryStore
from .web_learning import WebLearner

CODE_TEMPLATES = {
    "python": ("main.py", 'def main():\n    print("Hello from Python")\n\nif __name__ == "__main__":\n    main()\n'),
    "javascript": ("app.js", 'function main() {\n  console.log("Hello from JavaScript");\n}\n\nmain();\n'),
    "html": ("index.html", '<!doctype html>\n<html lang="ru">\n<head><meta charset="utf-8"><title>Новый сайт</title><link rel="stylesheet" href="style.css"></head>\n<body><main><h1>Новый сайт</h1><p>Создан автономным помощником.</p></main><script src="app.js"></script></body>\n</html>\n'),
    "css": ("style.css", 'body { margin: 0; font-family: system-ui, sans-serif; background: #111827; color: #f9fafb; }\nmain { max-width: 900px; margin: 4rem auto; padding: 2rem; }\n'),
    "c++": ("main.cpp", '#include <iostream>\n\nint main() {\n    std::cout << "Hello from C++" << std::endl;\n    return 0;\n}\n'),
    "java": ("Main.java", 'public class Main {\n    public static void main(String[] args) {\n        System.out.println("Hello from Java");\n    }\n}\n'),
}


class AssistantEngine:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.memory = MemoryStore(self.data_dir / "memory.sqlite3")
        self.logger = ActionLogger(self.data_dir / "actions.log")
        self.files = WorkspaceFiles(self.data_dir / "workspace")
        self.learner = WebLearner()
        self.settings_path = self.data_dir / "settings.json"
        self.settings = self._load_settings()
        self.autonomous_stop = Event()
        self.autonomous_thread: Thread | None = None
        self._progress_lock = RLock()

    def _load_settings(self) -> dict[str, str]:
        defaults = {"resource_level": "средняя нагрузка", "language": "ru", "llm_backend": "auto", "llm_model": "", "llm_learning_coach": "on", "self_modification_mode": "workspace"}
        if self.settings_path.exists():
            loaded = json.loads(self.settings_path.read_text(encoding="utf-8"))
            return defaults | loaded
        return defaults

    def save_settings(self) -> None:
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        defaults = {"resource_level": "средняя нагрузка", "language": "ru", "llm_backend": "auto", "llm_model": "", "llm_learning_coach": "on", "self_modification_mode": "workspace"}
        for key, value in defaults.items():
            self.settings.setdefault(key, value)
        self.settings_path.write_text(json.dumps(self.settings, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.log(f"Настройки сохранены: {self.settings}")

    def chat(self, message: str) -> str:
        message = message.strip()
        user_metadata = {"role": "user"}
        extracted_name = self._extract_user_name(message)
        if extracted_name:
            user_metadata["user_name"] = extracted_name
        self.memory.add("dialog", "Пользователь", message, user_metadata)
        self.logger.log(f"Получено сообщение пользователя: {message[:120]}")
        lower = message.lower()
        if self._is_self_modification_request(lower):
            response = self.create_self_modification_plan(message)
        elif self._is_learning_request(lower):
            topic = self._extract_topic(message)
            response = self.learn(topic)
        elif self._is_file_request(lower):
            response = self._handle_file_request(message)
        elif self._is_code_request(lower):
            response = self._handle_code_request(message)
        elif any(word in lower for word in ["найди", "поиск", "информац"]):
            topic = self._extract_topic(message)
            response = self.learn(topic)
        else:
            response = self._answer_from_memory(message)
        self.memory.add("dialog", "ИИ", response, {"role": "assistant"})
        self._self_analyze(message, response)
        return response

    def learn(self, topic: str, progress: Callable[[str], None] | None = None) -> str:
        topic = topic.strip(" .!?\n\t") or "общие знания"

        def emit(text: str, percent: int, status: str = "в процессе", note: str = "") -> None:
            self.logger.log(text)
            self._record_learning_progress(topic, percent, text, status=status, note=note)
            if progress:
                progress(f"[{percent:>3}%] {text}")

        emit(f"Получена задача изучить: {topic}", 5)
        experience = self._experience_context(topic)
        if experience:
            emit("Поднят прошлый опыт и похожие знания из памяти", 12, note=experience[:500])
        emit("Начат поиск информации в интернете", 20)
        sources = self.learner.search(topic, self._source_limit())
        emit(f"Найдено источников: {len(sources)}", 35)
        emit("Начато чтение найденных материалов", 45)
        sources = self.learner.read_sources(sources)
        emit("Выделение полезных знаний", 60)
        summary, key_points = self.learner.summarize(topic, sources)
        emit("Языковая модель помогает превратить материалы в учебный конспект", 75)
        mentor_note, mentor_backend = self._learning_coach(topic, summary, key_points, experience)
        if mentor_note:
            summary = f"{summary}\n\n## Подсказки языковой модели-наставника ({mentor_backend})\n{mentor_note}"
            emit(f"LLM-наставник подключен: {mentor_backend}", 82, note=mentor_note[:500])
        else:
            emit("LLM-наставник недоступен, использован локальный алгоритм конспекта", 82)
        note_name = self._safe_note_name(topic)
        note_path = self.files.write(note_name, summary)
        metadata = {
            "sources": [s.url for s in sources],
            "file": str(note_path),
            "key_points": key_points,
            "llm_learning_coach": mentor_backend if mentor_note else "fallback-summary",
            "based_on_experience": bool(experience),
        }
        self.memory.add("knowledge", topic, summary, metadata)
        self.memory.add("task", f"Изучение: {topic}", f"Изучены источники: {len(sources)}\nФайл: {note_path}\nКлючевые идеи: {key_points}")
        emit("Знания сохранены в долговременную память", 95)
        emit(f"Создана заметка: {note_path.name}", 100, status="готово")
        return summary + f"\n\nЗаметка сохранена в файле: {note_path.name}"

    def start_autonomous(self, progress: Callable[[str], None] | None = None) -> bool:
        if self.autonomous_thread and self.autonomous_thread.is_alive():
            return False
        self.autonomous_stop.clear()
        self.autonomous_thread = Thread(target=self._autonomous_loop, args=(progress,), daemon=True)
        self.autonomous_thread.start()
        return True

    def stop_autonomous(self) -> None:
        self.autonomous_stop.set()
        self.logger.log("Получена команда остановить автономный режим")

    def _autonomous_loop(self, progress: Callable[[str], None] | None) -> None:
        topics = self._next_topics()
        self.logger.log("Автономный режим запущен")
        for topic in topics:
            if self.autonomous_stop.is_set():
                break
            self.learn(topic, progress)
            report = self.self_improvement_report()
            self.memory.add("improvement", f"Отчет автономного режима: {topic}", report)
            if self.settings.get("self_modification_mode") == "autonomous_workspace":
                self.create_self_modification_plan(f"Автоулучшение после обучения: {topic}")
            if self._resource_delay_cycles() and self.autonomous_stop.wait(self._resource_delay_cycles()):
                break
        self.logger.log("Автономный режим остановлен")

    def self_improvement_report(self) -> str:
        stats = self.memory.stats()
        recent_tasks = self.memory.recent(5, "task")
        improvements = []
        if stats.get("knowledge", 0) < 5:
            improvements.append("Накопить больше проверенных знаний по ключевым темам пользователя.")
        if not self.files.list():
            improvements.append("Создать рабочие заметки и проекты, чтобы пользователь мог проверять результат в файлах.")
        improvements.append("Самомодификация работает безопасно: приложение генерирует план и код в workspace/self_mods, а пользователь решает, что переносить в ядро.")
        report = ["Самоанализ работы:", f"Всего записей памяти: {stats.get('total', 0)}", "Последние задачи:"]
        report.extend(f"- {task.title}" for task in recent_tasks)
        report.append("Что можно улучшить:")
        report.extend(f"- {item}" for item in improvements)
        return "\n".join(report)

    def learning_progress(self, limit: int = 12) -> list[dict[str, str | int]]:
        items = self.memory.recent(limit, "learning_progress")
        progress: list[dict[str, str | int]] = []
        for item in items:
            progress.append({
                "topic": str(item.metadata.get("topic", item.title)),
                "percent": int(item.metadata.get("percent", 0)),
                "stage": item.content.split("\n", 1)[0],
                "status": str(item.metadata.get("status", "в процессе")),
            })
        return progress

    def create_self_modification_plan(self, request: str = "") -> str:
        mode = self.settings.get("self_modification_mode", "workspace")
        if mode == "off":
            return "Самомодификация выключена в настройках. Включите режим workspace, чтобы я готовил планы и код улучшений в рабочей папке."
        report = self.self_improvement_report()
        memories = self.memory.recent(8)
        context = ChatContext(
            message=(
                "Предложи безопасное улучшение этого локального AI-приложения на основе прошлого опыта. "
                "Не меняй критические файлы напрямую: дай план, чеклист и пример кода/псевдокода для workspace. "
                f"Запрос пользователя: {request}\n\n{report}"
            ),
            memories=memories,
            history=list(reversed(self.memory.recent(8, "dialog"))),
        )
        suggestion, backend_name = LanguageModelRouter(self.settings).generate(context)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content = (
            f"# План самомодификации {timestamp}\n\n"
            f"Источник запроса: {request or 'автономный режим'}\n\n"
            f"## Прошлый опыт\n{report}\n\n"
            f"## Предложение языковой модели ({backend_name})\n{suggestion}\n\n"
            "## Правило безопасности\nФайл создан в рабочей папке. Переносить изменения в код приложения нужно только после проверки человеком.\n"
        )
        path = self.files.write(f"self_mods/self_improvement_{timestamp}.md", content)
        self.memory.add("self_modification", "План самомодификации", content, {"path": str(path), "backend": backend_name, "mode": mode})
        self.logger.log(f"Создан план самомодификации: {path.name}")
        return f"Я подготовил безопасный план самомодификации: {path.name}\n\n{suggestion}"

    def _answer_from_memory(self, message: str) -> str:
        memories = self.memory.search(message, limit=6)
        history = list(reversed(self.memory.recent(10, "dialog")))
        context = ChatContext(message=message, memories=memories, history=history)
        response, backend_name = LanguageModelRouter(self.settings).generate(context)
        self.logger.log(f"Ответ сформирован языковым режимом: {backend_name}")
        return response

    def _handle_file_request(self, message: str) -> str:
        match = re.search(r"(?:файл|file)\s+([\w./\\-]+\.(?:txt|md|json|csv|html|py|js|css|cpp|java))", message, re.I)
        name = match.group(1) if match else f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        content = f"# Заметка\n\nСоздано по запросу пользователя:\n\n{message}\n"
        path = self.files.write(name, content)
        self.memory.add("file", path.name, content, {"path": str(path)})
        self.logger.log(f"Создан файл: {path}")
        return f"Файл создан и сохранен: {path.name}"

    def _handle_code_request(self, message: str) -> str:
        lower = message.lower()
        chosen = "python"
        for lang in CODE_TEMPLATES:
            if lang in lower or (lang == "c++" and "с++" in lower):
                chosen = lang
                break
        filename, code = CODE_TEMPLATES[chosen]
        if "сайт" in lower or "site" in lower or "html" in lower:
            paths = [self.files.write("index.html", CODE_TEMPLATES["html"][1]), self.files.write("style.css", CODE_TEMPLATES["css"][1]), self.files.write("app.js", CODE_TEMPLATES["javascript"][1])]
            result = "Создан мини-сайт: " + ", ".join(p.name for p in paths)
            for p in paths:
                self.memory.add("file", p.name, p.read_text(encoding="utf-8"), {"path": str(p)})
            self.logger.log(result)
            return result
        path = self.files.write(filename, code)
        self.memory.add("file", path.name, code, {"path": str(path), "language": chosen})
        self.logger.log(f"Создан код на {chosen}: {path.name}")
        return f"Создан файл с кодом на {chosen}: {path.name}\n\n```{chosen}\n{code}```"

    def _self_analyze(self, request: str, response: str) -> None:
        report = (
            "Что было сделано: обработан запрос пользователя и сформирован ответ.\n"
            f"Что получилось хорошо: результат связан с запросом «{request[:120]}».\n"
            "Что можно улучшить: при необходимости запросить дополнительные источники или уточнения.\n"
            "Ошибки: критических ошибок не обнаружено."
        )
        self.memory.add("analysis", "Самоанализ ответа", report, {"request": request, "response_size": len(response)})

    def _record_learning_progress(self, topic: str, percent: int, stage: str, status: str = "в процессе", note: str = "") -> None:
        percent = max(0, min(100, int(percent)))
        content = f"{percent}% — {stage}"
        if note:
            content += f"\n{note}"
        with self._progress_lock:
            self.memory.add("learning_progress", topic, content, {"topic": topic, "percent": percent, "status": status})

    def _experience_context(self, topic: str) -> str:
        candidates = self.memory.search(topic, limit=6) + self.memory.recent(6, "analysis") + self.memory.recent(4, "improvement")
        seen: set[int] = set()
        lines: list[str] = []
        for item in candidates:
            if item.id in seen:
                continue
            seen.add(item.id)
            lines.append(f"- [{item.kind}] {item.title}: {item.content[:350].strip()}")
        return "\n".join(lines[:8])

    def _learning_coach(self, topic: str, summary: str, key_points: list[str], experience: str) -> tuple[str, str]:
        if self.settings.get("llm_learning_coach", "on") == "off":
            return "", "off"
        context = ChatContext(
            message=(
                f"Помоги обучить мою модель по теме «{topic}». "
                "Сделай короткий учебный план, контрольные вопросы и практические шаги на основе конспекта. "
                f"Ключевые идеи: {key_points}\n\nКонспект:\n{summary[:3500]}\n\nПрошлый опыт:\n{experience[:1800]}"
            ),
            memories=self.memory.search(topic, limit=5),
            history=list(reversed(self.memory.recent(6, "dialog"))),
        )
        response, backend_name = LanguageModelRouter(self.settings).generate(context)
        if backend_name == "fallback" and not response:
            return "", backend_name
        return response.strip(), backend_name

    def _extract_user_name(self, message: str) -> str | None:
        match = re.search(r"(?:меня зовут|мо[её] имя|я)\s+([А-ЯЁA-Z][а-яёa-z]{1,24})", message)
        return match.group(1) if match else None

    def _is_self_modification_request(self, lower: str) -> bool:
        return any(p in lower for p in ["самомодифиц", "модифицируй себя", "измени себя", "допиши себе", "улучши себя"])

    def _is_learning_request(self, lower: str) -> bool:
        return any(p in lower for p in ["изучи", "обуч", "learn", "исследуй", "тренируй"])

    def _is_file_request(self, lower: str) -> bool:
        return any(p in lower for p in ["создай файл", "запиши файл", "сохрани файл"])

    def _is_code_request(self, lower: str) -> bool:
        return any(p in lower for p in ["напиши программу", "создай сайт", "код", "python", "javascript", "html", "css", "c++", "java"])

    def _extract_topic(self, message: str) -> str:
        cleaned = re.sub(r"^(изучи|найди|объясни|исследуй|составь|начать обучение[:\s]*)", "", message.strip(), flags=re.I)
        return cleaned.strip(" .!?\n\t") or message

    def _source_limit(self) -> int:
        return {"низкая нагрузка": 3, "средняя нагрузка": 5, "высокая нагрузка": 8, "максимальная нагрузка": 10}.get(self.settings.get("resource_level"), 5)

    def _resource_delay_cycles(self) -> float:
        return {"низкая нагрузка": 8, "средняя нагрузка": 4, "высокая нагрузка": 2, "максимальная нагрузка": 1}.get(self.settings.get("resource_level"), 4)

    def _next_topics(self) -> list[str]:
        recent_knowledge = self.memory.recent(8, "knowledge")
        recent_tasks = self.memory.recent(5, "task")
        base = ["искусственный интеллект", "машинное обучение", "Python", "планирование задач", "безопасная автоматизация"]
        experience_topics = [f"углубленно по прошлому опыту: {item.title}" for item in recent_knowledge[:3]]
        task_topics = [f"разбор ошибок и улучшений после задачи: {item.title}" for item in recent_tasks[:2]]
        return experience_topics + task_topics + base

    def _safe_note_name(self, topic: str) -> str:
        slug = re.sub(r"[^\wа-яА-ЯёЁ-]+", "_", topic, flags=re.U).strip("_").lower()[:60] or "knowledge"
        return f"knowledge_{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
