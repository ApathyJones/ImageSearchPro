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
# Always add Homebrew to PATH — .command files launch with a minimal bash
# environment that does not source ~/.zshrc, so /opt/homebrew/bin may be absent.
if [[ "$(uname -m)" == "arm64" ]] && [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
elif [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
fi

if ! command -v brew &>/dev/null; then
    echo -e "${YELLOW}Homebrew not found. Installing Homebrew...${RESET}"
    echo "(This is required to install system libraries. It may take a few minutes.)"
    echo ""
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Re-initialise after fresh install
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

# Try versioned names first, then plain python3
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c 'import sys; v=sys.version_info; print(f"{v.major}.{v.minor}")')
        if [[ "$ver" == "3.10" || "$ver" == "3.11" || "$ver" == "3.12" ]]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo -e "${YELLOW}Python 3.10–3.12 not found. Installing Python 3.12 via Homebrew...${RESET}"
    brew install python@3.12
    PYTHON="python3.12"
    echo ""
fi

PY_ARCH=$("$PYTHON" -c "import platform; print(platform.machine())")
echo -e "${GREEN}Using Python: $($PYTHON --version)  [$PY_ARCH]${RESET}"
if [[ "$(uname -m)" == "arm64" ]] && [[ "$PY_ARCH" != "arm64" ]]; then
    echo -e "${YELLOW}NOTE: Using a Rosetta/x86_64 Python on Apple Silicon.${RESET}"
    echo -e "${YELLOW}      The app will work, but DINOv2/v3 models will be unavailable${RESET}"
    echo -e "${YELLOW}      (dinotool requires torch>=2.6, which needs native ARM64 Python).${RESET}"
    echo ""
fi
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
