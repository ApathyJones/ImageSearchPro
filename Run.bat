@echo off

REM Verify venv is functional — a renamed/moved folder breaks embedded paths.
set VENV_OK=0
if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" -c "" >nul 2>&1
    if not errorlevel 1 set VENV_OK=1
)

if "%VENV_OK%"=="0" (
    echo Venv not found or broken (was the folder renamed?). Run Install.bat to fix it.
    pause
    exit
)

call "%~dp0venv\Scripts\activate.bat"
python PhotoSearchPro.py
pause
