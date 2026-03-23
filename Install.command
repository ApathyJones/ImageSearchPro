#!/bin/bash
# Install.command — PhotoSearchPro macOS one-click installer
# Double-click this file in Finder to set up and launch the app.
# On first run macOS may ask you to allow it: right-click → Open.

set -euo pipefail

# ── 1. Change to script's own directory ─────────────────────────────────────
cd "$(dirname "$0")"

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

echo ""
echo -e "${BOLD}================================================${RESET}"
echo -e "${BOLD}   PhotoSearchPro — macOS Installer${RESET}"
echo -e "${BOLD}================================================${RESET}"
echo ""

# ── 2. Check / install Homebrew ──────────────────────────────────────────────
if ! command -v brew &>/dev/null; then
    echo -e "${YELLOW}Homebrew not found. Installing Homebrew...${RESET}"
    echo "(This is required to install system libraries. It may take a few minutes.)"
    echo ""
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Add brew to PATH for Apple Silicon (arm64)
    if [[ "$(uname -m)" == "arm64" ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    else
        eval "$(/usr/local/bin/brew shellenv)"
    fi
    echo ""
fi

# ── 3. Install libraw (required by rawpy for RAW photo support) ──────────────
if ! brew list libraw &>/dev/null 2>&1; then
    echo -e "${YELLOW}Installing libraw (needed for RAW photo support)...${RESET}"
    brew install libraw
    echo ""
fi

# ── 4. Find a compatible Python (3.12, 3.11, or 3.10) ───────────────────────
PYTHON=""

for ver in 3.12 3.11 3.10; do
    # Check Homebrew-installed pythons first (most reliable on macOS)
    if command -v "python${ver}" &>/dev/null; then
        PYTHON="python${ver}"
        break
    fi
done

# Fall back to python3 if a brew-managed version wasn't found
if [[ -z "$PYTHON" ]] && command -v python3 &>/dev/null; then
    py3_ver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ "$py3_ver" == "3.10" || "$py3_ver" == "3.11" || "$py3_ver" == "3.12" ]]; then
        PYTHON="python3"
    fi
fi

if [[ -z "$PYTHON" ]]; then
    echo -e "${YELLOW}Python 3.10–3.12 not found. Installing Python 3.12 via Homebrew...${RESET}"
    brew install python@3.12
    PYTHON="python3.12"
    echo ""
fi

echo -e "${GREEN}Using Python: $($PYTHON --version)${RESET}"
echo ""

# ── 5. Create virtual environment ────────────────────────────────────────────
if [[ ! -d "venv" ]]; then
    echo -e "${BOLD}Creating virtual environment...${RESET}"
    "$PYTHON" -m venv venv
    echo ""
fi

# ── 6. Activate venv ─────────────────────────────────────────────────────────
source venv/bin/activate

# ── 7. Upgrade pip silently ───────────────────────────────────────────────────
echo -e "${BOLD}Upgrading pip...${RESET}"
pip install --upgrade pip --quiet
echo ""

# ── 8. Install Python dependencies ───────────────────────────────────────────
echo -e "${BOLD}Installing Python dependencies (this may take several minutes on first run)...${RESET}"
echo ""
pip install -r requirements-mac.txt
echo ""

# ── 9. Make the other helper scripts executable ───────────────────────────────
chmod +x Run.command Update.command MigrateModels.command 2>/dev/null || true

# ── 10. Launch the app ────────────────────────────────────────────────────────
echo -e "${GREEN}${BOLD}Installation complete! Launching PhotoSearchPro...${RESET}"
echo ""
python PhotoSearchPro.py

echo ""
echo -e "${BOLD}PhotoSearchPro closed. Press any key to exit.${RESET}"
read -r -n 1
