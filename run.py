"""Запуск Autonomous AI Assistant из корня проекта.

Двойной клик по этому файлу или команда `python run.py` запускают GUI-приложение.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """Start the desktop application with a clear error for missing Tkinter."""
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from autonomous_ai_assistant.gui import main as gui_main
    except ModuleNotFoundError as exc:
        if exc.name == "tkinter":
            print(
                "Ошибка: не найден Tkinter. Установите Python с Tkinter "
                "или пакет python3-tk, затем запустите файл снова.",
                file=sys.stderr,
            )
            return 1
        raise

    gui_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
