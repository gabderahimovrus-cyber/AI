from __future__ import annotations

import importlib
import importlib.util
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Protocol

from .memory import MemoryItem


SYSTEM_PERSONA = (
    "Ты дружелюбный русскоязычный AI-помощник в стиле ChatGPT: живой, "
    "внимательный, честный о своих ограничениях, задаешь уточняющие вопросы, "
    "помнишь контекст и даешь структурированные полезные ответы."
)


@dataclass(slots=True)
class ChatContext:
    message: str
    memories: list[MemoryItem]
    history: list[MemoryItem]


class LanguageBackend(Protocol):
    name: str

    def generate(self, context: ChatContext) -> str | None:
        ...


class OllamaBackend:
    """Uses a locally running Ollama model when available."""

    name = "ollama"

    def __init__(self, model: str = "llama3.2", url: str = "http://127.0.0.1:11434/api/generate"):
        self.model = model or "llama3.2"
        self.url = url

    def generate(self, context: ChatContext) -> str | None:
        prompt = build_prompt(context)
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data = json.loads(response.read().decode("utf-8", errors="replace"))
        except (TimeoutError, OSError, urllib.error.URLError, json.JSONDecodeError):
            return None
        text = str(data.get("response", "")).strip()
        return text or None


class TransformersBackend:
    """Uses transformers.pipeline('text-generation') if the optional package is installed."""

    name = "transformers"

    def __init__(self, model: str = "distilgpt2"):
        self.model = model or "distilgpt2"
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            if importlib.util.find_spec("transformers") is None:
                return None
            transformers = importlib.import_module("transformers")
            self._pipeline = transformers.pipeline("text-generation", model=self.model)
        return self._pipeline

    def generate(self, context: ChatContext) -> str | None:
        pipe = self._load_pipeline()
        if pipe is None:
            return None
        prompt = build_prompt(context)
        try:
            result = pipe(prompt, max_new_tokens=180, do_sample=True, temperature=0.8, top_p=0.9)
        except Exception:
            return None
        if not result:
            return None
        generated = str(result[0].get("generated_text", ""))
        text = generated[len(prompt):].strip() if generated.startswith(prompt) else generated.strip()
        return text or None


class FallbackChatBackend:
    """A deterministic local conversational layer with personality and memory use."""

    name = "fallback"

    def generate(self, context: ChatContext) -> str:
        message = context.message.strip()
        lower = message.lower()
        name = self._known_user_name(context) or self._extract_user_name(message)
        if name:
            greeting_name = f", {name}"
        else:
            greeting_name = ""

        if self._is_greeting(lower):
            return (
                f"Привет{greeting_name}! Я рядом 😊\n\n"
                "Можешь писать как в обычный ChatGPT: вопрос, идею, задачу, текст для правки, просьбу написать код или тему для изучения. "
                "Я буду отвечать живее, помнить важные детали и при необходимости сам предложу следующий шаг."
            )
        if self._extract_user_name(message):
            extracted = self._extract_user_name(message)
            return f"Очень приятно, {extracted}! Запомню, как к тебе обращаться. Чем займемся дальше?"
        if any(phrase in lower for phrase in ["как дела", "как ты", "ты жив", "живой"]):
            return (
                "Я в порядке и готов помогать — не человек, конечно, но могу вести диалог теплее и внимательнее. "
                "Расскажи, что хочешь сделать, а я подхвачу: объясню, спланирую, напишу текст или код."
            )
        if "помоги обучить мою модель" in lower or "llm-наставник" in lower:
            return self._learning_coach_answer(context)
        if any(phrase in lower for phrase in ["что ты умеешь", "помоги", "возможности"]):
            return self._capabilities_answer(context)
        if "что ты помнишь" in lower or "память" in lower:
            return self._memory_answer(context)
        if self._looks_like_explanation_request(lower):
            return self._explanation_answer(context)
        if self._looks_like_advice_request(lower):
            return self._advice_answer(context)
        return self._general_answer(context)

    def _general_answer(self, context: ChatContext) -> str:
        memory_block = self._memory_block(context.memories)
        opener = "Понял. Давай разберем это спокойно и по делу."
        if context.memories:
            opener += " Я также учел то, что уже есть в памяти."
        return (
            f"{opener}\n\n"
            f"**Короткий ответ:** {self._short_response(context.message, context.memories)}\n\n"
            "**Что можно сделать дальше:**\n"
            "1. Уточнить цель и желаемый результат.\n"
            "2. Разбить задачу на 2–4 конкретных шага.\n"
            "3. Если нужны факты — запустить обучение/поиск и сохранить конспект.\n"
            "4. Если нужен результат в файле или коде — я могу создать его в рабочей папке."
            f"{memory_block}"
        )

    def _explanation_answer(self, context: ChatContext) -> str:
        topic = re.sub(r"^(объясни|расскажи|что такое|как работает)\s+", "", context.message, flags=re.I).strip(" ?.!")
        topic = topic or context.message
        return (
            f"Конечно. Объясню простыми словами: **{topic}** — это тема, которую лучше понимать через цель, основные понятия и пример.\n\n"
            "**Как подойти:**\n"
            "- сначала выделить ключевые термины;\n"
            "- затем посмотреть, где это применяется на практике;\n"
            "- после этого разобрать маленький пример;\n"
            "- в конце зафиксировать выводы в заметке.\n\n"
            "Если хочешь, напиши: «изучи " + topic + "», и я найду источники, сделаю резюме и сохраню его в памяти."
        )

    def _advice_answer(self, context: ChatContext) -> str:
        return (
            "Я бы предложил такой практичный путь:\n\n"
            "1. **Определи результат** — что должно получиться в конце.\n"
            "2. **Сделай первый маленький шаг** за 10–20 минут, без попытки сразу идеально.\n"
            "3. **Проверь обратную связь**: что получилось, что мешает, чего не хватает.\n"
            "4. **Улучши одну вещь**, а не всё сразу.\n\n"
            "Если расскажешь контекст чуть подробнее, я составлю конкретный план под твою ситуацию."
        )

    def _learning_coach_answer(self, context: ChatContext) -> str:
        topic_match = re.search(r"теме «([^»]+)»", context.message)
        topic = topic_match.group(1) if topic_match else "текущей теме"
        memory_hint = "Опирайся на уже сохраненные знания и отмечай, что изменилось после нового обучения." if context.memories else "Сначала накопи базовые факты, затем закрепи их вопросами и практикой."
        return (
            f"### Учебный план для модели по теме «{topic}»\n"
            "1. Сжать найденные материалы до 3–5 устойчивых тезисов.\n"
            "2. Сравнить новые тезисы с прошлой памятью и отдельно сохранить расхождения.\n"
            "3. Сформировать контрольные вопросы: определение, пример, типичная ошибка, практическое применение.\n"
            "4. Закрепить результат маленькой задачей в рабочей папке.\n\n"
            "### Контрольные вопросы\n"
            "- Какие 3 идеи являются главными?\n"
            "- Какой пример доказывает, что тема понята?\n"
            "- Что стоит повторить на следующем автономном цикле?\n\n"
            f"### Связь с прошлым опытом\n{memory_hint}"
        )

    def _capabilities_answer(self, context: ChatContext) -> str:
        return (
            "Я могу работать ближе к формату ChatGPT:\n"
            "- вести живой диалог и помнить важные детали;\n"
            "- объяснять темы простым языком;\n"
            "- искать и конспектировать информацию через обучение;\n"
            "- создавать файлы, заметки и стартовый код;\n"
            "- подключаться к локальной языковой модели Ollama или transformers, если они установлены;\n"
            "- без внешней модели использовать встроенный разговорный режим."
        )

    def _memory_answer(self, context: ChatContext) -> str:
        if not context.memories:
            return "Пока по этому запросу в памяти ничего близкого не нашел. Но все важные диалоги и результаты обучения я сохраняю и смогу использовать позже."
        lines = ["Вот что нашлось в памяти по теме:"]
        for item in context.memories[:5]:
            lines.append(f"- **{item.title}** ({item.kind}): {item.content[:220].strip()}")
        return "\n".join(lines)

    def _short_response(self, message: str, memories: list[MemoryItem]) -> str:
        if memories:
            return f"по запросу «{message}» уже есть связанный контекст, поэтому лучше опереться на него и уточнить конечную цель."
        return f"по запросу «{message}» я могу помочь сформулировать план, объяснение или сразу подготовить результат."

    def _memory_block(self, memories: list[MemoryItem]) -> str:
        if not memories:
            return ""
        lines = ["\n\n**Релевантная память:**"]
        for item in memories[:3]:
            lines.append(f"- {item.title}: {item.content[:180].strip()}")
        return "\n".join(lines)

    def _known_user_name(self, context: ChatContext) -> str | None:
        for item in context.history + context.memories:
            if item.metadata.get("user_name"):
                return str(item.metadata["user_name"])
            found = self._extract_user_name(item.content)
            if found:
                return found
        return None

    def _extract_user_name(self, message: str) -> str | None:
        match = re.search(r"(?:меня зовут|мо[её] имя|я)\s+([А-ЯЁA-Z][а-яёa-z]{1,24})", message)
        if match:
            return match.group(1)
        return None

    def _is_greeting(self, lower: str) -> bool:
        words = {"привет", "здравствуй", "здравствуйте", "hello", "hi", "хай"}
        return lower.strip(" !?.") in words or any(lower.startswith(word + " ") for word in words)

    def _looks_like_explanation_request(self, lower: str) -> bool:
        return lower.startswith(("объясни", "расскажи", "что такое", "как работает"))

    def _looks_like_advice_request(self, lower: str) -> bool:
        return any(word in lower for word in ["посоветуй", "как лучше", "что делать", "план действий"])


def build_prompt(context: ChatContext) -> str:
    history = _format_items(context.history[-8:])
    memories = _format_items(context.memories[:6])
    return (
        f"Система: {SYSTEM_PERSONA}\n\n"
        f"История диалога:\n{history or 'Пока нет.'}\n\n"
        f"Релевантная память:\n{memories or 'Пока нет.'}\n\n"
        f"Пользователь: {context.message}\n"
        "Ассистент:"
    )


def _format_items(items: Iterable[MemoryItem]) -> str:
    lines = []
    for item in items:
        role = item.metadata.get("role", item.kind)
        lines.append(f"- {role}: {item.title} — {item.content[:600]}")
    return "\n".join(lines)


class LanguageModelRouter:
    def __init__(self, settings: dict[str, str]):
        self.settings = settings
        self.fallback = FallbackChatBackend()

    def generate(self, context: ChatContext) -> tuple[str, str]:
        for backend in self._enabled_backends():
            response = backend.generate(context)
            if response:
                return response, backend.name
        return self.fallback.generate(context), self.fallback.name

    def _enabled_backends(self) -> list[LanguageBackend]:
        backend = self.settings.get("llm_backend", "auto")
        model = self.settings.get("llm_model", "").strip()
        if backend == "off":
            return []
        if backend == "ollama":
            return [OllamaBackend(model or "llama3.2")]
        if backend == "transformers":
            return [TransformersBackend(model or "distilgpt2")]
        return [OllamaBackend(model or "llama3.2")]
