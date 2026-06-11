from __future__ import annotations

import os
import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .engine import AssistantEngine

APP_TITLE = "Autonomous AI Assistant"


class AssistantApp(tk.Tk):
    def __init__(self, data_dir: str | Path | None = None):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x820")
        self.minsize(1000, 650)
        base = Path(data_dir or os.environ.get("AUTONOMOUS_AI_DATA", Path.home() / ".autonomous_ai_assistant"))
        self.engine = AssistantEngine(base)
        self.ui_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._configure_style()
        self._build_layout()
        self._refresh_all()
        self.after(250, self._poll_events)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        self.configure(bg="#202123")
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#202123")
        style.configure("Panel.TFrame", background="#171717")
        style.configure("Card.TFrame", background="#2f3033", relief="flat")
        style.configure("TLabel", background="#202123", foreground="#ececf1")
        style.configure("Muted.TLabel", background="#171717", foreground="#a7a7ad")
        style.configure("Card.TLabel", background="#2f3033", foreground="#ececf1")
        style.configure("TButton", background="#10a37f", foreground="#ffffff", borderwidth=0, focusthickness=0, padding=(12, 8))
        style.map("TButton", background=[("active", "#0e8f70")])
        style.configure("TNotebook", background="#202123", borderwidth=0)
        style.configure("TNotebook.Tab", background="#2f3033", foreground="#d1d5db", padding=(16, 10), borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", "#444654")], foreground=[("selected", "#ffffff")])
        style.configure("TCombobox", fieldbackground="#40414f", background="#40414f", foreground="#ececf1")
        style.configure("Treeview", background="#2f3033", fieldbackground="#2f3033", foreground="#ececf1", borderwidth=0, rowheight=30)
        style.configure("Treeview.Heading", background="#444654", foreground="#ececf1", borderwidth=0)
        style.map("Treeview", background=[("selected", "#10a37f")], foreground=[("selected", "#ffffff")])

    def _build_layout(self) -> None:
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)
        sidebar = ttk.Frame(root, style="Panel.TFrame", width=230)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        ttk.Label(sidebar, text="AI Assistant", font=("Inter", 20, "bold"), background="#171717", foreground="#f8fafc").pack(anchor="w", padx=18, pady=(22, 6))
        ttk.Label(sidebar, text="ChatGPT-like local workspace", style="Muted.TLabel", wraplength=190).pack(anchor="w", padx=18, pady=(0, 14))
        self.status_label = ttk.Label(sidebar, text="● Готов", style="Muted.TLabel")
        self.status_label.pack(anchor="w", padx=18, pady=(0, 18))
        self.memory_indicator = ttk.Label(sidebar, text="Память: 0", style="Muted.TLabel")
        self.memory_indicator.pack(anchor="w", padx=18, pady=4)
        self.task_indicator = ttk.Label(sidebar, text="Текущая задача: нет", wraplength=190, style="Muted.TLabel")
        self.task_indicator.pack(anchor="w", padx=18, pady=4)
        ttk.Separator(sidebar).pack(fill="x", padx=14, pady=18)
        for title in ["Чат", "Память", "Модель", "Обучение", "Метрики", "Автономный режим", "Лог действий", "Файлы", "Настройки"]:
            ttk.Label(sidebar, text=title, style="Muted.TLabel").pack(anchor="w", padx=18, pady=5)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(side="left", fill="both", expand=True, padx=12, pady=12)
        self._build_progress_panel(root)
        self._build_chat_tab()
        self._build_memory_tab()
        self._build_model_tab()
        self._build_learning_tab()
        self._build_metrics_tab()
        self._build_autonomous_tab()
        self._build_log_tab()
        self._build_files_tab()
        self._build_settings_tab()

    def _text(self, parent: tk.Widget, height: int = 10) -> tk.Text:
        widget = tk.Text(parent, height=height, bg="#343541", fg="#ececf1", insertbackground="#10a37f", relief="flat", wrap="word", padx=16, pady=14)
        widget.configure(font=("Inter", 11), highlightthickness=1, highlightbackground="#565869", highlightcolor="#10a37f")
        return widget

    def _build_progress_panel(self, root: tk.Widget) -> None:
        panel = ttk.Frame(root, style="Panel.TFrame", width=330)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)
        ttk.Label(panel, text="Прогресс обучения", font=("Inter", 15, "bold"), background="#171717", foreground="#f8fafc").pack(anchor="w", padx=16, pady=(22, 6))
        ttk.Label(panel, text="Здесь видно, как модель учится: этап, процент, статус и связь с прошлым опытом.", style="Muted.TLabel", wraplength=280).pack(anchor="w", padx=16, pady=(0, 12))
        columns = ("topic", "percent", "status")
        self.progress_tree = ttk.Treeview(panel, columns=columns, show="headings", height=14)
        self.progress_tree.heading("topic", text="Тема")
        self.progress_tree.heading("percent", text="%")
        self.progress_tree.heading("status", text="Статус")
        self.progress_tree.column("topic", width=150, anchor="w")
        self.progress_tree.column("percent", width=50, anchor="center")
        self.progress_tree.column("status", width=90, anchor="center")
        self.progress_tree.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        ttk.Button(panel, text="Обновить прогресс", command=self._refresh_progress).pack(fill="x", padx=14, pady=(0, 8))
        ttk.Button(panel, text="План самоулучшения", command=self._create_self_mod_plan).pack(fill="x", padx=14, pady=(0, 14))

    def _build_chat_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Чат")
        self.chat_history = self._text(tab, 28)
        self.chat_history.pack(fill="both", expand=True, padx=12, pady=12)
        bottom = ttk.Frame(tab)
        bottom.pack(fill="x", padx=12, pady=(0, 12))
        self.chat_input = self._text(bottom, 4)
        self.chat_input.pack(side="left", fill="x", expand=True)
        send = ttk.Button(bottom, text="Отправить", command=self._send_chat)
        send.pack(side="left", padx=(10, 0), fill="y")
        self.chat_input.bind("<Control-Return>", lambda _e: self._send_chat())

    def _build_memory_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Память")
        controls = ttk.Frame(tab)
        controls.pack(fill="x", padx=12, pady=12)
        self.memory_search = ttk.Entry(controls)
        self.memory_search.pack(side="left", fill="x", expand=True)
        ttk.Button(controls, text="Найти", command=self._refresh_memory).pack(side="left", padx=8)
        ttk.Button(controls, text="Экспорт JSON", command=self._export_memory).pack(side="left")
        self.memory_text = self._text(tab, 30)
        self.memory_text.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def _build_model_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Модель")
        self.model_text = self._text(tab, 30)
        self.model_text.pack(fill="both", expand=True, padx=12, pady=12)
        ttk.Button(tab, text="Обновить", command=self._refresh_model_status).pack(anchor="e", padx=12, pady=(0, 12))

    def _build_learning_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Обучение")
        top = ttk.Frame(tab)
        top.pack(fill="x", padx=12, pady=12)
        ttk.Label(top, text="Тема обучения:").pack(side="left")
        self.learning_topic = ttk.Entry(top)
        self.learning_topic.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(top, text="Старт", command=self._start_learning).pack(side="left")
        ttk.Button(top, text="Пауза", command=self._pause_training).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Продолжить", command=self._resume_training).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Остановить", command=self._stop_training).pack(side="left", padx=(8, 0))
        self.learning_log = self._text(tab, 30)
        self.learning_log.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def _build_metrics_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Метрики")
        ttk.Label(tab, text="Loss / perplexity / скорость обучения", font=("Inter", 14, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
        self.metrics_canvas = tk.Canvas(tab, height=220, bg="#343541", highlightthickness=1, highlightbackground="#565869")
        self.metrics_canvas.pack(fill="x", padx=12, pady=12)
        self.metrics_text = self._text(tab, 16)
        self.metrics_text.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        ttk.Button(tab, text="Обновить метрики", command=self._refresh_metrics).pack(anchor="e", padx=12, pady=(0, 12))

    def _build_autonomous_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Автономный режим")
        top = ttk.Frame(tab)
        top.pack(fill="x", padx=12, pady=12)
        ttk.Button(top, text="Автономный режим", command=self._start_autonomous).pack(side="left")
        ttk.Button(top, text="Остановить", command=self._stop_autonomous).pack(side="left", padx=8)
        self.autonomous_log = self._text(tab, 30)
        self.autonomous_log.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def _build_log_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Лог действий")
        ttk.Button(tab, text="Обновить", command=self._refresh_logs).pack(anchor="e", padx=12, pady=12)
        self.log_text = self._text(tab, 30)
        self.log_text.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def _build_files_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Файлы")
        left = ttk.Frame(tab)
        left.pack(side="left", fill="y", padx=12, pady=12)
        ttk.Button(left, text="Обновить", command=self._refresh_files).pack(fill="x")
        ttk.Button(left, text="Создать", command=self._create_file).pack(fill="x", pady=6)
        ttk.Button(left, text="Сохранить", command=self._save_current_file).pack(fill="x")
        self.file_list = tk.Listbox(left, bg="#343541", fg="#ececf1", selectbackground="#10a37f", relief="flat", width=34, highlightthickness=1, highlightbackground="#565869")
        self.file_list.pack(fill="both", expand=True, pady=(10, 0))
        self.file_list.bind("<<ListboxSelect>>", lambda _e: self._open_selected_file())
        self.file_editor = self._text(tab, 34)
        self.file_editor.pack(side="left", fill="both", expand=True, padx=(0, 12), pady=12)
        self.current_file: str | None = None

    def _build_settings_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Настройки")
        frame = ttk.Frame(tab, style="Card.TFrame")
        frame.pack(anchor="nw", fill="x", padx=12, pady=12)
        ttk.Label(frame, text="Использование ресурсов", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=12, pady=12)
        self.resource_level = ttk.Combobox(frame, values=["низкая нагрузка", "средняя нагрузка", "высокая нагрузка", "максимальная нагрузка"], state="readonly")
        self.resource_level.set(self.engine.settings.get("resource_level", "средняя нагрузка"))
        self.resource_level.grid(row=0, column=1, sticky="ew", padx=12, pady=12)
        ttk.Label(frame, text="Языковая модель", style="Card.TLabel").grid(row=1, column=0, sticky="w", padx=12, pady=12)
        self.llm_backend = ttk.Combobox(frame, values=["auto", "ollama", "transformers", "off"], state="readonly")
        self.llm_backend.set(self.engine.settings.get("llm_backend", "auto"))
        self.llm_backend.grid(row=1, column=1, sticky="ew", padx=12, pady=12)
        ttk.Label(frame, text="Модель (например llama3.2)", style="Card.TLabel").grid(row=2, column=0, sticky="w", padx=12, pady=12)
        self.llm_model = ttk.Entry(frame)
        self.llm_model.insert(0, self.engine.settings.get("llm_model", ""))
        self.llm_model.grid(row=2, column=1, sticky="ew", padx=12, pady=12)
        ttk.Label(
            frame,
            text="auto пробует локальную Ollama; transformers можно выбрать явно. Если модели нет, работает встроенный живой режим.",
            background="#2f3033",
            foreground="#9ca3af",
            wraplength=620,
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 12))
        ttk.Label(frame, text="LLM помогает обучению", style="Card.TLabel").grid(row=4, column=0, sticky="w", padx=12, pady=12)
        self.llm_learning_coach = ttk.Combobox(frame, values=["on", "off"], state="readonly")
        self.llm_learning_coach.set(self.engine.settings.get("llm_learning_coach", "on"))
        self.llm_learning_coach.grid(row=4, column=1, sticky="ew", padx=12, pady=12)
        ttk.Label(frame, text="Самомодификация", style="Card.TLabel").grid(row=5, column=0, sticky="w", padx=12, pady=12)
        self.self_modification_mode = ttk.Combobox(frame, values=["workspace", "autonomous_workspace", "off"], state="readonly")
        self.self_modification_mode.set(self.engine.settings.get("self_modification_mode", "workspace"))
        self.self_modification_mode.grid(row=5, column=1, sticky="ew", padx=12, pady=12)
        ttk.Label(
            frame,
            text="workspace создает планы и код в рабочей папке; autonomous_workspace делает это также после автономного обучения; off выключает функцию.",
            background="#2f3033",
            foreground="#9ca3af",
            wraplength=620,
        ).grid(row=6, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 12))
        ttk.Button(frame, text="Сохранить настройки", command=self._save_settings).grid(row=7, column=1, sticky="e", padx=12, pady=12)
        frame.columnconfigure(1, weight=1)

    def _send_chat(self) -> None:
        message = self.chat_input.get("1.0", "end").strip()
        if not message:
            return
        self.chat_input.delete("1.0", "end")
        self._append(self.chat_history, f"Вы: {message}\n")
        self._set_task("чат")
        def run() -> None:
            try:
                answer = self.engine.chat(message)
                self.ui_queue.put(("chat", f"ИИ: {answer}\n\n"))
            except Exception as exc:
                self.ui_queue.put(("chat", f"Ошибка: {exc}\n\n"))
            self.ui_queue.put(("task", "нет"))
        import threading
        threading.Thread(target=run, daemon=True).start()

    def _refresh_model_status(self) -> None:
        if not hasattr(self, "model_text"):
            return
        status = self.engine.model_status()
        meta = status.get("metadata", {})
        config = status.get("config", {})
        lines = [
            f"Текущая модель: {status.get('slot')}",
            f"Версия: {meta.get('version', 'not-created') if isinstance(meta, dict) else 'not-created'}",
            f"Устройство: {status.get('device')}",
            f"Размер: hidden={config.get('hidden_size', 256)}, layers={config.get('num_layers', 6)}, heads={config.get('num_heads', 8)}, context={config.get('context_length', 512)}",
            f"Число параметров: {meta.get('parameters', 'будет доступно после сохранения модели') if isinstance(meta, dict) else 'нет'}",
            f"Loss: {meta.get('loss', 'нет') if isinstance(meta, dict) else 'нет'}",
            f"Perplexity: {meta.get('perplexity', 'нет') if isinstance(meta, dict) else 'нет'}",
        ]
        self.model_text.delete("1.0", "end")
        self._append(self.model_text, "\n".join(lines))

    def _refresh_metrics(self) -> None:
        if not hasattr(self, "metrics_text"):
            return
        self.metrics_text.delete("1.0", "end")
        self.metrics_canvas.delete("all")
        progress = list(reversed(self.engine.learning_progress(50)))
        if not progress:
            self._append(self.metrics_text, "Пока нет данных обучения. Запустите обучение или автономный цикл.\n")
            return
        width = max(300, self.metrics_canvas.winfo_width() or 700)
        height = 220
        points = []
        for i, item in enumerate(progress):
            x = 20 + i * (width - 40) / max(1, len(progress) - 1)
            y = height - 20 - (float(item['percent']) / 100.0) * (height - 40)
            points.extend([x, y])
            self._append(self.metrics_text, f"{item['topic']}: {item['percent']}% — {item['status']}\n")
        if len(points) >= 4:
            self.metrics_canvas.create_line(*points, fill="#10a37f", width=3, smooth=True)
        self.metrics_canvas.create_text(16, 12, text="progress/loss proxy", anchor="w", fill="#ececf1")

    def _pause_training(self) -> None:
        self._append(self.learning_log, "Пауза запрошена. Для короткого локального обучения пауза применяется между циклами.\n")

    def _resume_training(self) -> None:
        self._append(self.learning_log, "Продолжение обучения запрошено.\n")

    def _stop_training(self) -> None:
        self.engine.stop_autonomous()
        self._append(self.learning_log, "Остановка обучения запрошена.\n")

    def _start_learning(self) -> None:
        topic = self.learning_topic.get().strip()
        if not topic:
            messagebox.showinfo(APP_TITLE, "Введите тему обучения")
            return
        self._set_task(f"обучение: {topic}")
        def progress(line: str) -> None:
            self.ui_queue.put(("learning", line + "\n"))
        def run() -> None:
            try:
                result = self.engine.learn(topic, progress)
                self.ui_queue.put(("learning", "\n" + result + "\n"))
            except Exception as exc:
                self.ui_queue.put(("learning", f"Ошибка обучения: {exc}\n"))
            self.ui_queue.put(("task", "нет"))
        import threading
        threading.Thread(target=run, daemon=True).start()

    def _create_self_mod_plan(self) -> None:
        self._set_task("самоулучшение")
        def run() -> None:
            try:
                result = self.engine.create_self_modification_plan("Запрос из панели прогресса")
                self.ui_queue.put(("autonomous", result + "\n"))
            except Exception as exc:
                self.ui_queue.put(("autonomous", f"Ошибка самоулучшения: {exc}\n"))
            self.ui_queue.put(("task", "нет"))
        import threading
        threading.Thread(target=run, daemon=True).start()

    def _start_autonomous(self) -> None:
        if self.engine.start_autonomous(lambda line: self.ui_queue.put(("autonomous", line + "\n"))):
            self._set_task("автономный режим")
            self._append(self.autonomous_log, "Автономный режим запущен\n")
        else:
            self._append(self.autonomous_log, "Автономный режим уже работает\n")

    def _stop_autonomous(self) -> None:
        self.engine.stop_autonomous()
        self._set_task("остановка автономного режима")

    def _refresh_all(self) -> None:
        self._refresh_memory(); self._refresh_logs(); self._refresh_files(); self._refresh_chat_history(); self._refresh_progress(); self._refresh_model_status(); self._refresh_metrics()

    def _refresh_chat_history(self) -> None:
        for item in reversed(self.engine.memory.recent(30, "dialog")):
            role = "Вы" if item.metadata.get("role") == "user" else "ИИ"
            self._append(self.chat_history, f"{role}: {item.content}\n\n")

    def _refresh_memory(self) -> None:
        query = self.memory_search.get().strip() if hasattr(self, "memory_search") else ""
        items = self.engine.memory.search(query, 50) if query else self.engine.memory.recent(80)
        self.memory_text.delete("1.0", "end")
        for item in items:
            self._append(self.memory_text, f"#{item.id} [{item.kind}] {item.title}\n{item.content[:1200]}\n\n")
        self.memory_indicator.configure(text=f"Память: {self.engine.memory.stats().get('total', 0)} записей")
        self._refresh_progress()

    def _refresh_progress(self) -> None:
        if not hasattr(self, "progress_tree"):
            return
        self.progress_tree.delete(*self.progress_tree.get_children())
        for item in self.engine.learning_progress(20):
            self.progress_tree.insert("", "end", values=(item["topic"], f'{item["percent"]}%', item["status"]))

    def _refresh_logs(self) -> None:
        self.log_text.delete("1.0", "end")
        for line in self.engine.logger.tail():
            self._append(self.log_text, line + "\n")

    def _refresh_files(self) -> None:
        self.file_list.delete(0, "end")
        for path in self.engine.files.list():
            self.file_list.insert("end", str(path.relative_to(self.engine.files.root)))

    def _open_selected_file(self) -> None:
        if not self.file_list.curselection():
            return
        name = self.file_list.get(self.file_list.curselection()[0])
        self.current_file = name
        self.file_editor.delete("1.0", "end")
        self.file_editor.insert("end", self.engine.files.read(name))

    def _create_file(self) -> None:
        name = filedialog.asksaveasfilename(initialdir=self.engine.files.root, defaultextension=".md")
        if not name:
            return
        rel = str(Path(name).resolve().relative_to(self.engine.files.root.resolve())) if str(Path(name).resolve()).startswith(str(self.engine.files.root.resolve())) else Path(name).name
        self.engine.files.write(rel, self.file_editor.get("1.0", "end"))
        self.engine.memory.add("file", rel, self.file_editor.get("1.0", "end"), {"path": rel})
        self._refresh_files()

    def _save_current_file(self) -> None:
        if not self.current_file:
            self._create_file(); return
        content = self.file_editor.get("1.0", "end")
        self.engine.files.write(self.current_file, content)
        self.engine.memory.add("file", self.current_file, content, {"path": self.current_file})
        self.engine.logger.log(f"Файл сохранен: {self.current_file}")
        self._refresh_files()

    def _export_memory(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            self.engine.memory.export_json(path)
            messagebox.showinfo(APP_TITLE, "Память экспортирована")

    def _save_settings(self) -> None:
        self.engine.settings["resource_level"] = self.resource_level.get()
        self.engine.settings["llm_backend"] = self.llm_backend.get()
        self.engine.settings["llm_model"] = self.llm_model.get().strip()
        self.engine.settings["llm_learning_coach"] = self.llm_learning_coach.get()
        self.engine.settings["self_modification_mode"] = self.self_modification_mode.get()
        self.engine.save_settings()
        self._refresh_memory()

    def _poll_events(self) -> None:
        while True:
            try:
                kind, payload = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "chat": self._append(self.chat_history, payload)
            elif kind == "learning": self._append(self.learning_log, payload)
            elif kind == "autonomous": self._append(self.autonomous_log, payload)
            elif kind == "task": self._set_task(payload)
            self._refresh_progress()
        while not self.engine.logger.events.empty():
            self._append(self.log_text, self.engine.logger.events.get() + "\n")
        self.memory_indicator.configure(text=f"Память: {self.engine.memory.stats().get('total', 0)} записей")
        self._refresh_progress()
        self.after(250, self._poll_events)

    def _append(self, widget: tk.Text, text: str) -> None:
        widget.insert("end", text)
        widget.see("end")

    def _set_task(self, task: str) -> None:
        self.task_indicator.configure(text=f"Текущая задача: {task}")
        self.status_label.configure(text="● Работает" if task != "нет" else "● Готов")

    def _on_close(self) -> None:
        self.engine.stop_autonomous()
        self.engine.save_settings()
        self.destroy()


def main() -> None:
    app = AssistantApp()
    app.mainloop()
