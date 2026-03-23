#!/bin/bash
# Run.command — PhotoSearchPro macOS launcher
# Double-click this in Finder to start the app after installation.

cd "$(dirname "$0")"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "ERROR: Run.command is for macOS only. On Linux, run: python PhotoSearchPro.py"
    exit 1
fi

if [[ ! -f "venv/bin/python" ]] || ! venv/bin/python -c "" &>/dev/null; then
    echo "Venv not found or broken (was the folder renamed?). Run Install.command to fix it."
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
