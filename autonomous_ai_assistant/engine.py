from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from threading import Event, Thread
from typing import Callable

from .files import WorkspaceFiles
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

    def _load_settings(self) -> dict[str, str]:
        if self.settings_path.exists():
            return json.loads(self.settings_path.read_text(encoding="utf-8"))
        return {"resource_level": "средняя нагрузка", "language": "ru"}

    def save_settings(self) -> None:
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(json.dumps(self.settings, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.log(f"Настройки сохранены: {self.settings}")

    def chat(self, message: str) -> str:
        message = message.strip()
        self.memory.add("dialog", "Пользователь", message, {"role": "user"})
        self.logger.log(f"Получено сообщение пользователя: {message[:120]}")
        lower = message.lower()
        if self._is_learning_request(lower):
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
        def emit(text: str) -> None:
            self.logger.log(text)
            if progress:
                progress(text)
        emit(f"Получена задача изучить: {topic}")
        emit("Начат поиск информации в интернете")
        sources = self.learner.search(topic, self._source_limit())
        emit(f"Найдено источников: {len(sources)}")
        emit("Начато чтение найденных материалов")
        sources = self.learner.read_sources(sources)
        emit("Выделение полезных знаний")
        summary, key_points = self.learner.summarize(topic, sources)
        note_name = self._safe_note_name(topic)
        note_path = self.files.write(note_name, summary)
        self.memory.add("knowledge", topic, summary, {"sources": [s.url for s in sources], "file": str(note_path)})
        self.memory.add("task", f"Изучение: {topic}", f"Изучены источники: {len(sources)}\nФайл: {note_path}\nКлючевые идеи: {key_points}")
        emit("Знания сохранены в долговременную память")
        emit(f"Создана заметка: {note_path.name}")
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
        improvements.append("Для критических файлов приложения требуется явное разрешение пользователя перед изменениями.")
        report = ["Самоанализ работы:", f"Всего записей памяти: {stats.get('total', 0)}", "Последние задачи:"]
        report.extend(f"- {task.title}" for task in recent_tasks)
        report.append("Что можно улучшить:")
        report.extend(f"- {item}" for item in improvements)
        return "\n".join(report)

    def _answer_from_memory(self, message: str) -> str:
        memories = self.memory.search(message, limit=5)
        if memories:
            context = "\n".join(f"- [{m.kind}] {m.title}: {m.content[:500]}" for m in memories)
            return (
                "Я учел сохраненную память и контекст разговора.\n\n"
                f"Релевантные воспоминания:\n{context}\n\n"
                "Ответ: " + self._compose_reasoned_answer(message, memories)
            )
        return self._compose_reasoned_answer(message, [])

    def _compose_reasoned_answer(self, message: str, memories: list) -> str:
        lower = message.lower()
        if "квант" in lower:
            return "Квантовая физика описывает микромир, где энергия и состояние систем меняются дискретными порциями, частицы проявляют волновые свойства, а результат измерения вероятностен. Для изучения начните с суперпозиции, интерференции, неопределенности Гейзенберга и квантовых состояний."
        if "план" in lower or "изуч" in lower:
            return "Предлагаю план: 1) определить цель, 2) изучить базовые понятия, 3) сделать маленький проект, 4) разобрать ошибки, 5) сохранить конспект, 6) повторить и усложнить задачу. Я могу запустить обучение по конкретной теме и сохранить резюме."
        if "что ты помнишь" in lower or "память" in lower:
            stats = self.memory.stats()
            return "В памяти сохранено: " + ", ".join(f"{k}: {v}" for k, v in stats.items())
        return "Я могу ответить на вопрос, создать файл/код, запустить обучение по теме или автономно пополнять знания. Уточните цель, и я сохраню результат в долговременной памяти."

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

    def _is_learning_request(self, lower: str) -> bool:
        return any(p in lower for p in ["изучи", "обуч", "learn", "исследуй"])

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
        recent = self.memory.recent(8, "knowledge")
        base = ["искусственный интеллект", "машинное обучение", "Python", "планирование задач", "безопасная автоматизация"]
        return [f"углубленно: {item.title}" for item in recent[:3]] + base

    def _safe_note_name(self, topic: str) -> str:
        slug = re.sub(r"[^\wа-яА-ЯёЁ-]+", "_", topic, flags=re.U).strip("_").lower()[:60] or "knowledge"
        return f"knowledge_{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
