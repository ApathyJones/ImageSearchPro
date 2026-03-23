@echo off

REM Check if venv exists AND is functional (survives folder renames/moves).
REM venvs embed absolute paths, so a renamed folder breaks them silently.
set VENV_OK=0
if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" -c "" >nul 2>&1
    if not errorlevel 1 set VENV_OK=1
)

if "%VENV_OK%"=="0" (
    if exist "%~dp0venv" (
        echo Existing venv is broken (folder was likely renamed or moved). Recreating...
        rmdir /s /q "%~dp0venv"
    )
    py -3.12 -m venv venv
    if errorlevel 1 py -3.11 -m venv venv
    if errorlevel 1 py -3.10 -m venv venv
    if errorlevel 1 python -m venv venv
    if errorlevel 1 (
        echo No compatible Python found! Please install Python 3.10, 3.11, or 3.12 from python.org
        pause
        exit
    )
)

call "%~dp0venv\Scripts\activate.bat"
pip install -r requirements.txt
python PhotoSearchPro.py
pause
