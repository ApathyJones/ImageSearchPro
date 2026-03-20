@echo off
if not exist "%~dp0venv" (
    echo Venv not found! Please run Install.bat first.
    pause
    exit
)
call "%~dp0venv\Scripts\activate.bat"
python PhotoSearchPro.py
pause