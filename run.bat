@echo off
setlocal
cd /d "%~dp0"
py -3 run.py
if errorlevel 1 (
    echo.
    echo Запуск через py не удался, пробую python...
    python run.py
)
pause
