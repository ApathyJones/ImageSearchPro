#!/bin/bash
# Run.command — PhotoSearchPro macOS launcher
# Double-click this in Finder to start the app after installation.

cd "$(dirname "$0")"

if [[ ! -d "venv" ]]; then
    echo "Virtual environment not found."
    echo "Please run Install.command first."
    echo ""
    echo "Press any key to exit."
    read -r -n 1
    exit 1
fi

source venv/bin/activate
python PhotoSearchPro.py

echo ""
echo "PhotoSearchPro closed. Press any key to exit."
read -r -n 1
