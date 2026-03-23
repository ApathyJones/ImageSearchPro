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
# On Apple Silicon, prefer ARM64 Pythons and skip any x86_64 ones.
PYTHON=""
NEED_ARM64=$([[ "$(uname -m)" == "arm64" ]] && echo "yes" || echo "no")

_is_good_python() {
    local candidate="$1"
    command -v "$candidate" &>/dev/null || return 1
    local full_path
    full_path=$(command -v "$candidate")
    local ver
    ver=$("$full_path" -c 'import sys; v=sys.version_info; print(f"{v.major}.{v.minor}")')
    [[ "$ver" == "3.10" || "$ver" == "3.11" || "$ver" == "3.12" ]] || return 1
    if [[ "$NEED_ARM64" == "yes" ]]; then
        local arch
        arch=$("$full_path" -c "import platform; print(platform.machine())")
        [[ "$arch" == "arm64" ]] || return 1
    fi
    echo "$full_path"
}

# Build candidate list: Miniconda/Mambaforge ARM paths + versioned names + plain python3
CONDA_BASE="${CONDA_PREFIX:-${HOME}/miniconda3}"
for candidate in \
    "${CONDA_BASE}/bin/python3.12" \
    "${CONDA_BASE}/bin/python3.11" \
    "${CONDA_BASE}/bin/python3.10" \
    "${CONDA_BASE}/bin/python3" \
    "${HOME}/mambaforge/bin/python3" \
    /opt/homebrew/bin/python3.12 \
    /opt/homebrew/bin/python3.11 \
    /opt/homebrew/bin/python3.10 \
    python3.12 python3.11 python3.10 python3
do
    result=$(_is_good_python "$candidate" 2>/dev/null) && PYTHON="$result" && break
done

if [[ -z "$PYTHON" ]]; then
    echo -e "${YELLOW}No compatible ARM64 Python 3.10–3.12 found. Installing Python 3.12 via Homebrew...${RESET}"
    brew install python@3.12
    PYTHON=/opt/homebrew/bin/python3.12
    echo ""
fi

PY_ARCH=$("$PYTHON" -c "import platform; print(platform.machine())")
echo -e "${GREEN}Using Python: $("$PYTHON" --version)  [$PY_ARCH]  ($PYTHON)${RESET}"
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
