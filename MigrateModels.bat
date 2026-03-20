@echo off
:: MigrateModels.bat
:: Moves cached model files from %USERPROFILE%\.cache into the app's
:: models\ folder so nothing needs to be re-downloaded.
:: Just double-click this file — no other setup required.

cd /d "%~dp0"
python migrate_models.py
