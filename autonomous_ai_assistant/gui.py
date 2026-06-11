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
        self.configure(bg="#0f172a")
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#0f172a")
        style.configure("Panel.TFrame", background="#111827")
        style.configure("TLabel", background="#0f172a", foreground="#e5e7eb")
        style.configure("Muted.TLabel", background="#111827", foreground="#9ca3af")
        style.configure("TButton", background="#2563eb", foreground="#ffffff", borderwidth=0, focusthickness=0, padding=8)
        style.map("TButton", background=[("active", "#1d4ed8")])
        style.configure("TNotebook", background="#0f172a", borderwidth=0)
        style.configure("TNotebook.Tab", background="#1f2937", foreground="#d1d5db", padding=(14, 8))
        style.map("TNotebook.Tab", background=[("selected", "#2563eb")], foreground=[("selected", "#ffffff")])
        style.configure("TCombobox", fieldbackground="#1f2937", background="#1f2937", foreground="#e5e7eb")

    def _build_layout(self) -> None:
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)
        sidebar = ttk.Frame(root, style="Panel.TFrame", width=230)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        ttk.Label(sidebar, text="AI Assistant", font=("Inter", 20, "bold"), background="#111827", foreground="#f8fafc").pack(anchor="w", padx=18, pady=(22, 6))
        self.status_label = ttk.Label(sidebar, text="● Готов", style="Muted.TLabel")
        self.status_label.pack(anchor="w", padx=18, pady=(0, 18))
        self.memory_indicator = ttk.Label(sidebar, text="Память: 0", style="Muted.TLabel")
        self.memory_indicator.pack(anchor="w", padx=18, pady=4)
        self.task_indicator = ttk.Label(sidebar, text="Текущая задача: нет", wraplength=190, style="Muted.TLabel")
        self.task_indicator.pack(anchor="w", padx=18, pady=4)
        ttk.Separator(sidebar).pack(fill="x", padx=14, pady=18)
        for title in ["Чат", "Память", "Обучение", "Автономный режим", "Лог действий", "Файлы", "Настройки"]:
            ttk.Label(sidebar, text=title, style="Muted.TLabel").pack(anchor="w", padx=18, pady=5)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        self._build_chat_tab()
        self._build_memory_tab()
        self._build_learning_tab()
        self._build_autonomous_tab()
        self._build_log_tab()
        self._build_files_tab()
        self._build_settings_tab()

    def _text(self, parent: tk.Widget, height: int = 10) -> tk.Text:
        widget = tk.Text(parent, height=height, bg="#020617", fg="#e5e7eb", insertbackground="#93c5fd", relief="flat", wrap="word", padx=12, pady=12)
        widget.configure(font=("Inter", 11))
        return widget

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

    def _build_learning_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Обучение")
        top = ttk.Frame(tab)
        top.pack(fill="x", padx=12, pady=12)
        ttk.Label(top, text="Тема обучения:").pack(side="left")
        self.learning_topic = ttk.Entry(top)
        self.learning_topic.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(top, text="Начать обучение", command=self._start_learning).pack(side="left")
        self.learning_log = self._text(tab, 30)
        self.learning_log.pack(fill="both", expand=True, padx=12, pady=(0, 12))

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
        self.file_list = tk.Listbox(left, bg="#020617", fg="#e5e7eb", selectbackground="#2563eb", relief="flat", width=34)
        self.file_list.pack(fill="both", expand=True, pady=(10, 0))
        self.file_list.bind("<<ListboxSelect>>", lambda _e: self._open_selected_file())
        self.file_editor = self._text(tab, 34)
        self.file_editor.pack(side="left", fill="both", expand=True, padx=(0, 12), pady=12)
        self.current_file: str | None = None

    def _build_settings_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Настройки")
        frame = ttk.Frame(tab, style="Panel.TFrame")
        frame.pack(anchor="nw", fill="x", padx=12, pady=12)
        ttk.Label(frame, text="Использование ресурсов", background="#111827").grid(row=0, column=0, sticky="w", padx=12, pady=12)
        self.resource_level = ttk.Combobox(frame, values=["низкая нагрузка", "средняя нагрузка", "высокая нагрузка", "максимальная нагрузка"], state="readonly")
        self.resource_level.set(self.engine.settings.get("resource_level", "средняя нагрузка"))
        self.resource_level.grid(row=0, column=1, sticky="ew", padx=12, pady=12)
        ttk.Button(frame, text="Сохранить настройки", command=self._save_settings).grid(row=1, column=1, sticky="e", padx=12, pady=12)
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
        self._refresh_memory(); self._refresh_logs(); self._refresh_files(); self._refresh_chat_history()

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
        while not self.engine.logger.events.empty():
            self._append(self.log_text, self.engine.logger.events.get() + "\n")
        self.memory_indicator.configure(text=f"Память: {self.engine.memory.stats().get('total', 0)} записей")
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
