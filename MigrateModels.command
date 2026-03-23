#!/bin/bash
# MigrateModels.command — PhotoSearchPro macOS model cache migration
# Moves previously-downloaded model files from ~/.cache into the app's
# models/ folder so nothing needs to be re-downloaded.
# Double-click this in Finder to run.

cd "$(dirname "$0")"

if [[ ! -d "venv" ]]; then
    echo "Virtual environment not found — please run Install.command first."
    echo ""
    echo "Press any key to exit."
    read -r -n 1
    exit 1
fi

source venv/bin/activate
python migrate_models.py

echo ""
echo "Press any key to exit."
read -r -n 1
