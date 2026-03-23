#!/bin/bash
# Update.command — PhotoSearchPro macOS updater
# Double-click this in Finder to pull the latest version and update dependencies.

cd "$(dirname "$0")"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "ERROR: Update.command is for macOS only. On Linux, run: git pull && pip install -r requirements.txt"
    exit 1
fi

BOLD='\033[1m'
GREEN='\033[0;32m'
RESET='\033[0m'

echo ""
echo -e "${BOLD}================================================${RESET}"
echo -e "${BOLD}   PhotoSearchPro — macOS Updater${RESET}"
echo -e "${BOLD}================================================${RESET}"
echo ""

echo -e "${BOLD}Pulling latest changes from git...${RESET}"
git pull
echo ""

if [[ ! -d "venv" ]]; then
    echo "Virtual environment not found — please run Install.command first."
    echo ""
    echo "Press any key to exit."
    read -r -n 1
    exit 1
fi

source venv/bin/activate

echo -e "${BOLD}Updating Python dependencies...${RESET}"
pip install --upgrade pip --quiet
pip install -r requirements-mac.txt
echo ""

echo -e "${GREEN}${BOLD}Update complete!${RESET}"
echo "Run Run.command to launch PhotoSearchPro."
echo ""
echo "Press any key to exit."
read -r -n 1
