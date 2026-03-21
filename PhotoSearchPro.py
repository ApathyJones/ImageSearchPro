import os
import sys
import io
import pickle
import json
import shutil
import threading
from datetime import datetime
from threading import Thread
from pathlib import Path
import time
import queue
import ctypes
import subprocess
import re
import gc
import warnings
from PIL import Image
import numpy as np

# On Windows, torch MUST be imported before PyQt6.  PyQt6 loads Qt DLLs that
# corrupt the DLL resolution environment, causing c10.dll's DllMain to fail
# (WinError 1114) when torch is imported afterward.  Importing torch first
# (after registering its lib dir) avoids this ordering conflict.
# The Qt platform plugin env vars get corrected later in main() before
# QApplication() is created, so the reverse Qt-vs-torch conflict is also fixed.
if os.name == 'nt':
    try:
        import importlib.util as _ilu
        _torch_spec = _ilu.find_spec('torch')
        if _torch_spec and _torch_spec.origin:
            _torch_lib = os.path.join(os.path.dirname(_torch_spec.origin), 'lib')
            if os.path.isdir(_torch_lib):
                os.add_dll_directory(_torch_lib)  # Python 3.8+ Windows API
        import torch as _torch_preload  # noqa: F401 — side-effect: loads CUDA DLLs first
    except Exception:
        pass

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton,
    QLineEdit, QCheckBox, QSlider, QSpinBox, QScrollArea, QListWidget,
    QListWidgetItem, QMenu, QDialog, QTabWidget, QProgressBar,
    QFileDialog, QMessageBox, QHBoxLayout, QVBoxLayout, QGridLayout,
    QSizePolicy, QAbstractItemView, QSplitter, QRubberBand,
    QInputDialog, QComboBox)
from PyQt6.QtGui import (
    QPixmap, QImage, QFont, QCursor, QAction, QKeySequence, QIcon,
    QPalette, QColor)
from PyQt6.QtCore import (
    Qt, QTimer, QPoint, QRect, QSize, QByteArray, QMimeData, QUrl, QEvent, pyqtSignal,
    QFileSystemWatcher)

# Prevent PIL from crashing on legitimately large images (scanned maps, panoramas, etc.)
# Files that truly cannot be decoded still get caught by the try/except in open_image()
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# ---------------------------------------------------------------------------
# App directory & model cache redirection
# Models are stored in a "models" sub-folder next to the script/executable
# instead of the default ~/.cache location on the system drive.
# ---------------------------------------------------------------------------
_SCRIPT = Path(sys.executable if getattr(sys, "frozen", False) else __file__)
APP_DIR   = _SCRIPT.resolve().parent
MODELS_DIR = APP_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Redirect HuggingFace hub (open_clip + transformers) before any HF import.
# HF_HOME sets the root; HUGGINGFACE_HUB_CACHE is the exact hub download dir.
_HF_CACHE = str(MODELS_DIR / "huggingface" / "hub")
os.makedirs(_HF_CACHE, exist_ok=True)
os.environ.setdefault("HF_HOME",                str(MODELS_DIR / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE",  _HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE",     _HF_CACHE)
# Redirect PyTorch hub downloads (e.g. open_clip weights fetched via torch.hub)
os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))

# --- Cross-Platform Configuration & Auto-Tuning ---
def get_system_vram():
    """
    Cross-platform VRAM detection.
    Returns VRAM in bytes, or None if detection fails.
    """
    # Method 1: PyTorch (Best for NVIDIA CUDA and macOS MPS)
    try:
        import torch
        
        # NVIDIA CUDA (Windows/Linux)
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory
            return vram
        
        # Apple Silicon MPS (macOS)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            import psutil
            return int(psutil.virtual_memory().total * 0.8)
    except Exception:
        pass

    # Method 2: nvidia-smi (most accurate for NVIDIA GPUs — avoids WMI 32-bit overflow)
    if os.name == 'nt' or sys.platform.startswith('linux'):
        try:
            cmd = ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits']
            output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL,
                                             creationflags=0x08000000 if os.name == 'nt' else 0
                                             ).decode('utf-8', errors='ignore').strip()
            # output is VRAM in MiB, one line per GPU — take the largest
            values = [int(s) for s in re.findall(r'\d+', output) if int(s) > 0]
            if values:
                return max(values) * 1024 * 1024  # MiB → bytes
        except Exception:
            pass

    # Method 3: Windows PowerShell CimInstance fallback (32-bit field — overflows for >4 GB GPUs)
    if os.name == 'nt':
        try:
            cmd = ['powershell', '-NoProfile', '-NonInteractive', '-Command',
                   'Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty AdapterRAM']
            output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL,
                                             creationflags=0x08000000).decode('utf-8', errors='ignore')
            values = [int(s) for s in re.findall(r'\d+', output) if int(s) > 1_000_000]
            if values:
                return max(values)
        except Exception:
            pass
    
    # Method 3: Linux sysfs (AMD GPUs)
    elif sys.platform.startswith('linux'):
        try:
            import glob
            vram_paths = glob.glob('/sys/class/drm/card*/device/mem_info_vram_total')
            if vram_paths:
                with open(vram_paths[0], 'r') as f:
                    return int(f.read().strip())
        except Exception:
            pass
    
    # Method 4: macOS system_profiler
    elif sys.platform == 'darwin':
        try:
            cmd = 'system_profiler SPDisplaysDataType'
            output = subprocess.check_output(cmd, shell=True).decode('utf-8')
            match = re.search(r'VRAM.*?(\d+)\s*(MB|GB)', output, re.IGNORECASE)
            if match:
                size = int(match.group(1))
                unit = match.group(2).upper()
                return size * (1024**3 if unit == 'GB' else 1024**2)
        except Exception:
            pass
    
    return None

def determine_batch_size(vram_bytes=None):
    if vram_bytes is None:
        vram_bytes = get_system_vram()
    if vram_bytes is None:
        print("[CONFIG] Could not detect VRAM. Defaulting to Batch Size: 16")
        return 16
    
    vram_gb = vram_bytes / (1024**3)
    print(f"[CONFIG] Detected VRAM: {vram_gb:.2f} GB")
    
    if vram_gb >= 31:
        return 384
    elif vram_gb >= 23:
        return 256
    elif vram_gb >= 19:
        return 192
    elif vram_gb >= 15:
        return 160
    elif vram_gb >= 11:
        return 128
    elif vram_gb >= 7:
        return 64
    elif vram_gb >= 4:
        return 32
    else:
        return 16

def determine_video_batch_size(vram_bytes=None):
    if vram_bytes is None:
        vram_bytes = get_system_vram()
    if vram_bytes is None:
        return 8
    vram_gb = vram_bytes / (1024**3)
    if vram_gb >= 31:
        return 64
    elif vram_gb >= 23:
        return 48
    elif vram_gb >= 19:
        return 32
    elif vram_gb >= 15:
        return 24
    elif vram_gb >= 11:
        return 16
    elif vram_gb >= 7:
        return 12
    elif vram_gb >= 4:
        return 8
    else:
        return 4

# Query VRAM once — both batch sizes share the same detection result
_VRAM_BYTES = get_system_vram()
BATCH_SIZE = determine_batch_size(_VRAM_BYTES)
print(f"[CONFIG] Selected Batch Size: {BATCH_SIZE}")
VIDEO_BATCH_SIZE = determine_video_batch_size(_VRAM_BYTES)
print(f"[CONFIG] Selected Video Batch Size: {VIDEO_BATCH_SIZE}")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif",
              ".cr2", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf", ".pef", ".sr2")
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v",
              ".wmv", ".flv", ".ts", ".mpg", ".mpeg", ".3gp", ".vob")
VIDEO_FRAME_INTERVAL = 5  # minimum seconds between sampled frames
MAX_FRAMES_PER_VIDEO = 50  # cap frames per video — interval scales up for long videos
TOP_RESULTS = 30
THUMBNAIL_SIZE = (180, 180)
MAX_THUMBNAIL_CACHE = 2000  # Limit RAM usage - clear old thumbnails after this
CELL_WIDTH = 220
CELL_HEIGHT = 260
CACHE_PREFIX = ".clip_cache_"
CACHE_SUFFIX = ".pkl"

# ── Model registry ─────────────────────────────────────────────────────────────
# Each entry exposes the full configuration needed to load and run one backend.
# has_text=True  → supports both text queries and image queries (full search)
# has_text=False → image similarity only (DINOv2/v3 have no text encoder)
MODEL_REGISTRY = {
    "clip-vit-l-14-laion": {
        "label":      "CLIP ViT-L-14",
        "subtitle":   "LAION-2B  •  Text + Image search  •  Default",
        "type":       "openclip",
        "model_name": "ViT-L-14",
        "pretrained": "laion2b_s32b_b82k",
        "has_text":   True,
        "cache_key":  "ViT-L-14_LAION2B",   # preserves old cache filenames
        "input_size": 224,
    },
    "siglip2-so400m-384": {
        "label":        "SigLIP2 SO/400M",
        "subtitle":     "384 px  •  Text + Image search  •  Higher quality than CLIP",
        "type":         "siglip2",
        "hf_model_id":  "google/siglip2-so400m-patch14-384",
        "has_text":     True,
        "cache_key":    "SigLIP2-SO400M-384",
        "input_size":   384,
    },
    "dinov2-l": {
        "label":     "DINOv2 ViT-L/14",
        "subtitle":  "Image similarity only  •  Open access  •  Strong visual features",
        "type":      "dinotool",
        "dino_name": "vit-l",
        "has_text":  False,
        "cache_key": "DINOv2-L",
        "input_size": 518,
    },
    "dinov2-g": {
        "label":     "DINOv2 ViT-G/14",
        "subtitle":  "Best DINOv2  •  Image similarity only  •  Open access",
        "type":      "dinotool",
        "dino_name": "vit-g",
        "has_text":  False,
        "cache_key": "DINOv2-G",
        "input_size": 518,
    },
    "dinov3-s": {
        "label":            "DINOv3 ViT-S/16",
        "subtitle":         "Meta 2025  •  Fastest DINOv3  •  Image only  •  Requires HuggingFace login",
        "type":             "dinotool",
        "dino_name":        "dinov3-s",
        "has_text":         False,
        "cache_key":        "DINOv3-S",
        "input_size":       518,
        "requires_hf_auth": True,
    },
    "dinov3-b": {
        "label":            "DINOv3 ViT-B/16",
        "subtitle":         "Meta 2025  •  Image only  •  Requires HuggingFace login",
        "type":             "dinotool",
        "dino_name":        "dinov3-b",
        "has_text":         False,
        "cache_key":        "DINOv3-B",
        "input_size":       518,
        "requires_hf_auth": True,
    },
    "dinov3-l": {
        "label":            "DINOv3 ViT-L/16",
        "subtitle":         "Meta 2025  •  Image only  •  Requires HuggingFace login",
        "type":             "dinotool",
        "dino_name":        "dinov3-l",
        "has_text":         False,
        "cache_key":        "DINOv3-L",
        "input_size":       518,
        "requires_hf_auth": True,
    },
    "dinov3-hplus": {
        "label":            "DINOv3 ViT-H+/16",
        "subtitle":         "Meta 2025  •  High quality  •  Image only  •  Requires HuggingFace login",
        "type":             "dinotool",
        "dino_name":        "dinov3-hplus",
        "has_text":         False,
        "cache_key":        "DINOv3-Hplus",
        "input_size":       518,
        "requires_hf_auth": True,
    },
    "dinov3-7b": {
        "label":            "DINOv3 ViT-7B/16",
        "subtitle":         "Meta 2025  •  Best DINOv3  •  ~14GB VRAM  •  Image only  •  Requires HuggingFace login",
        "type":             "dinotool",
        "dino_name":        "dinov3-7b",
        "has_text":         False,
        "cache_key":        "DINOv3-7B",
        "input_size":       518,
        "requires_hf_auth": True,
    },
}

DEFAULT_MODEL_KEY = "clip-vit-l-14-laion"

# Persisted user preferences (model choice, etc.)
SETTINGS_FILE = Path.home() / ".photosearchpro_config.json"

def _load_app_settings():
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_app_settings(data):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        safe_print(f"[SETTINGS] Save failed: {e}")

def _load_saved_indexes():
    """Return the list of saved index entries from settings, newest first."""
    return _load_app_settings().get("saved_indexes", [])

def _save_indexes_list(entries):
    """Persist the saved index entries list to settings."""
    settings = _load_app_settings()
    settings["saved_indexes"] = entries
    _save_app_settings(settings)

# --- ONNX Toggle ---
# Set USE_ONNX = True ONLY if PyTorch CUDA doesn't work on your GPU
# (e.g. early RTX 50-series Blackwell cards on PyTorch < 2.7)
# For most users PyTorch native CUDA is faster and uses less VRAM.
USE_ONNX = False

BG               = "#0d1117"   # Deep background
PANEL_BG         = "#161b22"   # Toolbar / panel areas
CARD_BG          = "#21262d"   # Cards, inputs
CARD_HOVER       = "#2d333b"   # Card hover state
FG               = "#e6edf3"   # Primary text
FG_MUTED         = "#8b949e"   # Secondary / hint text
ACCENT           = "#3fb950"   # Green — success, status
ACCENT_SECONDARY = "#388bfd"   # Blue — primary interactive (Search, Move …)
DANGER           = "#f85149"   # Red — destructive actions
ORANGE           = "#e3b341"   # Amber — warnings
BORDER           = "#30363d"   # Subtle border
BORDER_ACTIVE    = "#58a6ff"   # Focus / hover border highlight

RAW_EXTS = (".cr2", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf", ".pef", ".sr2")

# NudeNet label groups (ordered explicit → suggestive → body → face)
NUDENET_LABEL_GROUPS = {
    "Explicit": [
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "ANUS_EXPOSED",
    ],
    "Nudity": [
        "FEMALE_BREAST_EXPOSED",
        "MALE_BREAST_EXPOSED",
        "BUTTOCKS_EXPOSED",
    ],
    "Suggestive / Covered": [
        "FEMALE_GENITALIA_COVERED",
        "FEMALE_BREAST_COVERED",
        "BUTTOCKS_COVERED",
        "ANUS_COVERED",
    ],
    "Body Parts": [
        "BELLY_EXPOSED",
        "BELLY_COVERED",
        "ARMPITS_EXPOSED",
        "ARMPITS_COVERED",
        "FEET_EXPOSED",
        "FEET_COVERED",
    ],
    "Faces": [
        "FACE_FEMALE",
        "FACE_MALE",
    ],
}

# Serializes disk reads so HDD head moves sequentially instead of thrashing.
_DISK_LOCK = threading.Lock()

DARK_QSS = f"""
    /* ── Base ─────────────────────────────────────────────────────────── */
    QMainWindow, QWidget {{ background-color: {BG}; color: {FG}; }}

    /* ── Buttons — default: subtle gradient with visible border ─────────── */
    QPushButton {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #2d333b, stop:1 {CARD_BG});
        color: {FG};
        border: 1px solid #3a4050;
        padding: 5px 14px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 9pt;
        min-height: 22px;
    }}
    QPushButton:hover {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #373e47, stop:1 #2d333b);
        border-color: {FG_MUTED};
    }}
    QPushButton:pressed {{ background: {PANEL_BG}; border-color: {BORDER_ACTIVE}; }}

    /* accent — vivid blue (Search, Move, pagination …) */
    QPushButton[class="accent"] {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4fa3ff, stop:1 {ACCENT_SECONDARY});
        color: white;
        border: 1px solid {ACCENT_SECONDARY};
    }}
    QPushButton[class="accent"]:hover {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #70b8ff, stop:1 #4fa3ff);
        border-color: {BORDER_ACTIVE};
    }}
    QPushButton[class="accent"]:pressed {{ background: {ACCENT_SECONDARY}; }}

    /* danger — vivid red (Delete, STOP, EXIT) */
    QPushButton[class="danger"] {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 {DANGER}, stop:1 #bf3a38);
        color: white;
        border: 1px solid #bf3a38;
    }}
    QPushButton[class="danger"]:hover {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #ff7b78, stop:1 {DANGER});
        border-color: #ff7b78;
    }}
    QPushButton[class="danger"]:pressed {{ background: #bf3a38; }}

    /* ── Inputs ────────────────────────────────────────────────────────── */
    QLineEdit {{
        background-color: {BG};
        color: {FG};
        border: 1px solid {BORDER};
        padding: 5px 10px 7px 10px;
        border-radius: 6px;
        min-height: 26px;
        selection-background-color: {ACCENT_SECONDARY};
    }}
    QLineEdit:focus {{ border: 1px solid {BORDER_ACTIVE}; }}

    QSpinBox {{
        background-color: {CARD_BG};
        color: {FG};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 4px 8px;
    }}
    QSpinBox:focus {{ border-color: {BORDER_ACTIVE}; }}

    QComboBox {{
        background-color: {CARD_BG};
        color: {FG};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 4px 10px;
        min-height: 22px;
    }}
    QComboBox:hover {{ border-color: {FG_MUTED}; }}
    QComboBox:focus {{ border-color: {BORDER_ACTIVE}; }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
    QComboBox QAbstractItemView {{
        background-color: {PANEL_BG};
        color: {FG};
        border: 1px solid {BORDER};
        border-radius: 0px;
        selection-background-color: {ACCENT_SECONDARY};
        outline: none;
    }}
    QComboBox QAbstractItemView::item {{
        background-color: {PANEL_BG};
        color: {FG};
        padding: 4px 8px;
    }}
    QComboBox QAbstractItemView::item:selected,
    QComboBox QAbstractItemView::item:hover {{
        background-color: {ACCENT_SECONDARY};
        color: {FG};
    }}
    QComboBox QAbstractScrollArea {{
        background-color: {PANEL_BG};
        border: none;
    }}
    QComboBox QAbstractScrollArea > QWidget {{
        background-color: {PANEL_BG};
    }}

    /* ── Lists ─────────────────────────────────────────────────────────── */
    QListWidget {{
        background-color: {PANEL_BG};
        color: {FG};
        border: 1px solid {BORDER};
        border-radius: 6px;
        outline: none;
        padding: 2px;
    }}
    QListWidget::item {{ padding: 4px 8px; border-radius: 4px; }}
    QListWidget::item:selected {{ background-color: {ACCENT_SECONDARY}; color: white; }}
    QListWidget::item:hover:!selected {{ background-color: {CARD_BG}; }}

    /* ── Scrollbars ─────────────────────────────────────────────────────── */
    QScrollArea {{ background-color: {BG}; border: none; }}
    QScrollBar:vertical {{
        background: {BG};
        width: 8px;
        margin: 0;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER};
        border-radius: 4px;
        min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {FG_MUTED}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; border: none; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
    QScrollBar:horizontal {{
        background: {BG};
        height: 8px;
        margin: 0;
        border-radius: 4px;
    }}
    QScrollBar::handle:horizontal {{
        background: {BORDER};
        border-radius: 4px;
        min-width: 24px;
    }}
    QScrollBar::handle:horizontal:hover {{ background: {FG_MUTED}; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; border: none; }}
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: none; }}

    /* ── Checkboxes ─────────────────────────────────────────────────────── */
    QCheckBox {{ color: {FG}; spacing: 6px; background: transparent; }}
    QCheckBox::indicator {{
        width: 14px;
        height: 14px;
        border: 1px solid {BORDER};
        border-radius: 3px;
        background: {CARD_BG};
    }}
    QCheckBox::indicator:hover {{ border-color: {BORDER_ACTIVE}; }}
    QCheckBox::indicator:checked {{
        background: {ACCENT_SECONDARY};
        border-color: {ACCENT_SECONDARY};
    }}

    /* ── Sliders ────────────────────────────────────────────────────────── */
    QSlider::groove:horizontal {{
        background: {CARD_BG};
        height: 6px;
        border-radius: 3px;
        border: 1px solid {BORDER};
    }}
    QSlider::sub-page:horizontal {{
        background: {ACCENT_SECONDARY};
        height: 6px;
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 {BORDER_ACTIVE}, stop:1 {ACCENT_SECONDARY});
        width: 16px;
        height: 16px;
        border-radius: 8px;
        margin: -6px 0;
        border: 2px solid {BG};
    }}
    QSlider::handle:horizontal:hover {{ background: {BORDER_ACTIVE}; }}

    /* ── Progress bar ───────────────────────────────────────────────────── */
    QProgressBar {{
        background-color: {CARD_BG};
        border: none;
        border-radius: 4px;
    }}
    QProgressBar::chunk {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {ACCENT_SECONDARY}, stop:1 {ACCENT});
        border-radius: 4px;
    }}

    /* ── Tabs ───────────────────────────────────────────────────────────── */
    QTabWidget::pane {{
        border: 1px solid {BORDER};
        border-radius: 6px;
        background: {PANEL_BG};
        top: -1px;
    }}
    QTabBar::tab {{
        background: {CARD_BG};
        color: {FG_MUTED};
        padding: 6px 18px;
        border: 1px solid {BORDER};
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        margin-right: 2px;
        font-weight: 600;
    }}
    QTabBar::tab:selected {{
        background: {PANEL_BG};
        color: {FG};
        border-bottom: 2px solid {ACCENT_SECONDARY};
    }}
    QTabBar::tab:hover:!selected {{ color: {FG}; background: {CARD_HOVER}; }}

    /* ── Menus ──────────────────────────────────────────────────────────── */
    QMenu {{
        background-color: {PANEL_BG};
        color: {FG};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 4px 0;
    }}
    QMenu::item {{ padding: 7px 24px; border-radius: 4px; }}
    QMenu::item:selected {{ background-color: {ACCENT_SECONDARY}; color: white; }}
    QMenu::separator {{ background: {BORDER}; height: 1px; margin: 4px 8px; }}

    /* ── Dialogs & tips ─────────────────────────────────────────────────── */
    QDialog {{ background-color: {BG}; color: {FG}; }}
    QToolTip {{
        background-color: {PANEL_BG};
        color: {FG};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 9pt;
    }}
    QLabel {{ background: transparent; color: {FG}; }}
"""

def get_safe_path(path):
    """Prepend Windows extended path prefix to handle paths longer than 260 chars."""
    if os.name == 'nt':
        path = os.path.normpath(path)
        if not path.startswith('\\\\?\\'):
            path = '\\\\?\\' + path
    return path

def open_image(path):
    """Open any image including RAW formats. Falls back gracefully if rawpy not installed."""
    safe_path = get_safe_path(path)
    # Check extension on original path (without prefix)
    if path.lower().endswith(RAW_EXTS):
        try:
            import rawpy
            # Lock disk read — sequential HDD access, then decode in RAM without lock
            with _DISK_LOCK:
                with open(safe_path, 'rb') as f:
                    raw_bytes = f.read()
            with rawpy.imread(io.BytesIO(raw_bytes)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
            img = Image.fromarray(rgb)
        except ImportError:
            safe_print(f"[RAW] rawpy not installed, skipping {os.path.basename(path)}. Install with: pip install rawpy")
            return None
        except Exception as e:
            safe_print(f"[RAW] Failed to open {os.path.basename(path)}: {e}")
            return None
    else:
        try:
            # Lock disk read — sequential HDD access, then decode in RAM without lock
            with _DISK_LOCK:
                with open(safe_path, 'rb') as fh:
                    file_bytes = fh.read()
            # Use BytesIO with explicit format derived from extension so PIL
            # doesn't have to guess — works for WEBP, JPG, PNG etc.
            ext = os.path.splitext(path)[1].lower().lstrip('.')
            fmt_map = {'jpg': 'JPEG', 'jpeg': 'JPEG', 'png': 'PNG',
                       'webp': 'WEBP', 'bmp': 'BMP', 'gif': 'GIF'}
            fmt = fmt_map.get(ext)
            img = Image.open(io.BytesIO(file_bytes), formats=[fmt] if fmt else None)
            img.load()   # force full decode in RAM, no file handle needed
        except MemoryError:
            safe_print(f"[IMAGE] Skipping {os.path.basename(path)}: image too large for available RAM")
            return None
        except Exception as e:
            safe_print(f"[IMAGE] Failed to open {os.path.basename(path)}: {e}")
            return None

    if img.mode == 'P' and 'transparency' in img.info:
        img = img.convert("RGBA")

    if img.mode == 'RGBA':
        # Composite onto white background — better for CLIP than black (matches training data)
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # use alpha channel as mask
        img = background
    elif img.mode != 'RGB':
        img = img.convert("RGB")
    return img

def safe_print(text, end='\n'):
    try:
        print(text, end=end)
    except:
        pass

def pil_to_pixmap(pil_img):
    """Convert PIL Image to QPixmap via numpy bridge."""
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    arr = np.array(pil_img)
    h, w, c = arr.shape
    qimg = QImage(arr.data, w, h, w * c, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _detect_device():
    """Return (device, device_name, amp_dtype) for the best available accelerator."""
    import torch
    device_name = "CPU"
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = f"CUDA (GPU {torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            device_name = "Metal (Apple GPU)"
        elif os.name == 'nt':
            try:
                import torch_directml
                device = torch_directml.device()
                device_name = "DirectML (Windows GPU)"
            except ImportError:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
    except Exception:
        device = torch.device("cpu")

    # Detect best AMP dtype
    amp_dtype = None
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            if major >= 8:
                if major >= 9 or (major == 8 and minor >= 9):
                    amp_dtype = torch.bfloat16
                else:
                    amp_dtype = torch.float16
            elif major >= 7:
                amp_dtype = torch.float16
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            amp_dtype = torch.float16
    except Exception:
        amp_dtype = None

    return device, device_name, amp_dtype


class HybridCLIPModel:
    """
    Cross-Platform Hybrid Model Wrapper (open_clip backend).
    Accepts a model_cfg dict from MODEL_REGISTRY.
    """
    def __init__(self, model_cfg):
        import torch
        import open_clip
        if USE_ONNX:
            import onnxruntime as ort

        self.model_cfg = model_cfg
        self.has_text = model_cfg["has_text"]
        model_name = model_cfg["model_name"]
        pretrained  = model_cfg.get("pretrained", "")

        # 1. Determine Device
        self.device, self.device_name, self.amp_dtype = _detect_device()
        safe_print(f"[MODEL] Using Device: {self.device_name}")
        if self.amp_dtype is not None:
            safe_print(f"[MODEL] Mixed precision enabled: {self.amp_dtype}")

        # Enable TF32 on Ampere+ for free matmul speedup (ignored on non-CUDA)
        try:
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        # RTX 50-series (Blackwell) check — warn if PyTorch version is too old
        try:
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                if major >= 12:
                    pt_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
                    if pt_version < (2, 7):
                        safe_print(f"\n{'='*60}")
                        safe_print(f"[WARNING] RTX 50-series (Blackwell) detected!")
                        safe_print(f"[WARNING] Your PyTorch version ({torch.__version__}) may not fully support this GPU.")
                        safe_print(f"[WARNING] If you see CUDA errors, upgrade PyTorch:")
                        safe_print(f"[WARNING] pip install torch --index-url https://download.pytorch.org/whl/cu128")
                        safe_print(f"[WARNING] Or enable ONNX fallback: set USE_ONNX = True at top of script")
                        safe_print(f"{'='*60}\n")
                    else:
                        safe_print(f"[MODEL] RTX 50-series detected — PyTorch {torch.__version__} has native support")
        except Exception:
            pass

        safe_print(f"[MODEL] Loading: {model_name}")

        # 2. Load PyTorch Model
        model_loaded = False

        try:
            import huggingface_hub
            huggingface_hub.constants.HF_HUB_OFFLINE = True
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            kw = dict(pretrained=pretrained) if pretrained else {}
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, cache_dir=_HF_CACHE, **kw
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            safe_print(f"[MODEL] Loaded from local cache")
            model_loaded = True
        except Exception:
            safe_print(f"[MODEL] Cache not available, connecting to download...")

        if not model_loaded:
            try:
                import huggingface_hub
                huggingface_hub.constants.HF_HUB_OFFLINE = False
                os.environ["HF_HUB_OFFLINE"] = "0"
                os.environ["TRANSFORMERS_OFFLINE"] = "0"
                safe_print(f"[MODEL] Downloading {model_name} (this may take a while)...")
                kw = dict(pretrained=pretrained) if pretrained else {}
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name, cache_dir=_HF_CACHE, **kw
                )
                self.tokenizer = open_clip.get_tokenizer(model_name)
                safe_print(f"[MODEL] Download complete!")
            except Exception as e:
                safe_print(f"[MODEL] Download failed: {e}")
                raise
        
        # Move model to device — fall back to CPU if GPU transfer fails (e.g. TDR reset,
        # exclusive GPU lock, driver crash, or cudaErrorDevicesUnavailable)
        try:
            self.model = self.model.to(self.device).eval()
        except Exception as gpu_err:
            safe_print(f"[MODEL] GPU transfer failed ({gpu_err}), falling back to CPU")
            safe_print(f"[MODEL] Tip: close other GPU-heavy apps (games, other ML tools) and restart")
            self.device = torch.device("cpu")
            self.device_name = "CPU (GPU fallback)"
            self.amp_dtype = None  # disable mixed precision on CPU
            self.model = self.model.to(self.device).eval()

        # 3. Setup ONNX Visual Encoder (only if USE_ONNX = True)
        if USE_ONNX:
            self.setup_onnx_encoder()
        else:
            # ONNX disabled — use pure PyTorch (faster, less VRAM)
            self.onnx_visual_path = None
            self.use_onnx_visual = False
            self.visual_session = None
            self.onnx_disabled = True
            safe_print(f"[MODEL] Using PyTorch native inference (ONNX disabled)")
            safe_print(f"[MODEL] Ready!\n")

    def setup_onnx_encoder(self):
        """Setup ONNX Visual Encoder with graceful fallback"""
        import torch
        import onnxruntime as ort
        
        # Initialize fallback state first
        self.onnx_visual_path = None
        self.use_onnx_visual = False
        self.visual_session = None
        self.onnx_disabled = False
        
        # Test if ONNX export is supported before attempting
        if not self._test_onnx_support():
            safe_print(f"[ONNX] Not supported on this system")
            safe_print(f"[ONNX] Using PyTorch (works perfectly)")
            self.onnx_disabled = True
            safe_print(f"[MODEL] Ready!\n")
            return
        
        # Setup ONNX Visual Encoder
        cache_dir = MODELS_DIR / "onnx_clip"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_visual_path = cache_dir / f"{self.model_cfg['cache_key'].replace('-', '_')}_visual.onnx"
        
        if not self.onnx_visual_path.exists():
            safe_print(f"[ONNX] Attempting visual encoder export...")
            
            try:
                dummy_image = torch.randn(1, 3, 224, 224).to(self.device)
                with torch.no_grad():
                    torch.onnx.export(
                        self.model.visual,
                        dummy_image,
                        self.onnx_visual_path,
                        input_names=['pixel_values'],
                        output_names=['image_embeds'],
                        dynamic_axes={'pixel_values': {0: 'batch'}},
                        opset_version=14,
                        do_constant_folding=True,
                        verbose=False
                    )
                safe_print(f"[ONNX] Export successful")
                
            except (Exception, SystemError, RuntimeError, KeyboardInterrupt) as e:
                # Catch all possible exceptions including C++ errors
                if self.onnx_visual_path and self.onnx_visual_path.exists():
                    try:
                        self.onnx_visual_path.unlink()
                    except:
                        pass
                
                safe_print(f"[ONNX] Export failed, using PyTorch")
                self.onnx_visual_path = None
                self.use_onnx_visual = False
                self.visual_session = None
                self.onnx_disabled = True
                safe_print(f"[MODEL] Ready!\n")
                return
        
        self._create_onnx_session()
        safe_print(f"[MODEL] Ready!\n")
    
    def _test_onnx_support(self):
        """Quick test if ONNX export is supported"""
        import torch
        
        # Disable ONNX on Linux by default to prevent segfaults
        # PyTorch still runs on GPU normally without ONNX
        if sys.platform.startswith('linux'):
            safe_print("[ONNX] Disabled on Linux to prevent segfaults (GPU still active via PyTorch)")
            return False
        
        return True
    
    def _create_onnx_session(self):
        """Create or recreate ONNX inference session"""
        import torch
        import onnxruntime as ort
        
        # Initialize to False first
        self.use_onnx_visual = False
        self.visual_session = None
        
        # If ONNX is disabled or path doesn't exist, skip
        if getattr(self, 'onnx_disabled', False) or not hasattr(self, 'onnx_visual_path') or self.onnx_visual_path is None:
            return
        
        if self.onnx_visual_path and self.onnx_visual_path.exists():
            # Check for corrupted/zero-byte ONNX file before attempting to load
            try:
                file_size = self.onnx_visual_path.stat().st_size
                if file_size < 1024:  # anything under 1KB is certainly corrupt
                    safe_print(f"[ONNX] Corrupted ONNX file detected ({file_size} bytes), deleting...")
                    self.onnx_visual_path.unlink()
                    self.use_onnx_visual = False
                    self.onnx_disabled = True
                    return
            except Exception:
                pass

            try:
                providers = []
                
                if torch.cuda.is_available():
                    providers.append('CUDAExecutionProvider')
                
                if sys.platform == 'darwin':
                    providers.append('CoreMLExecutionProvider')
                
                if os.name == 'nt':
                    providers.append('DmlExecutionProvider')
                
                providers.append('CPUExecutionProvider')
                
                self.visual_session = ort.InferenceSession(str(self.onnx_visual_path), providers=providers)
                self.use_onnx_visual = True
                active_provider = self.visual_session.get_providers()[0]
                safe_print(f"[ONNX] Visual encoder ready on {active_provider}")
            except Exception as e:
                safe_print(f"[ONNX] Failed to load, using PyTorch: {e}")
                self.use_onnx_visual = False
                self.visual_session = None
    
    def _destroy_onnx_session(self):
        """Destroy ONNX session to free VRAM (only if ONNX was used)"""
        if hasattr(self, 'visual_session') and self.visual_session is not None:
            try:
                # Delete the session object
                del self.visual_session
                self.visual_session = None
                self.use_onnx_visual = False
                
                # Force CUDA cleanup
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                safe_print("[ONNX] Session destroyed, VRAM freed")
            except Exception as e:
                safe_print(f"[ONNX] Cleanup warning: {e}")

    def preprocess_image_onnx(self, image):
        target_size = 224
        w, h = image.size
        scale = target_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = image.resize((new_w, new_h), Image.BICUBIC)
        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        img = img.crop((left, top, left + target_size, top + target_size))
        img_np = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        img_np = (img_np - mean) / std
        img_np = img_np.transpose(2, 0, 1)
        return np.expand_dims(img_np, axis=0)
    
    def preprocess_image_pytorch(self, image):
        return self.preprocess(image).unsqueeze(0)
    
    def encode_image_batch(self, images):
        import torch
        
        # Only try ONNX if it's enabled and session exists
        if getattr(self, 'use_onnx_visual', False) and self.visual_session is not None:
            try:
                batch_inputs = [self.preprocess_image_onnx(img) for img in images]
                input_tensor = np.concatenate(batch_inputs, axis=0)
                outputs = self.visual_session.run(None, {"pixel_values": input_tensor})
                features = outputs[0]
                norms = np.linalg.norm(features, axis=1, keepdims=True)
                return features / norms
            except Exception as e:
                safe_print(f"[ONNX] Inference failed, falling back to PyTorch: {e}")
                # Disable ONNX for future calls
                self.use_onnx_visual = False
        
        # PyTorch path (always works, with optional mixed precision)
        try:
            batch_tensors = [self.preprocess_image_pytorch(img) for img in images]
            input_tensor = torch.cat(batch_tensors).to(self.device)
            amp_dtype = getattr(self, 'amp_dtype', None)
            with torch.no_grad():
                if amp_dtype is not None and torch.cuda.is_available():
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        features = self.model.encode_image(input_tensor)
                elif amp_dtype is not None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    with torch.autocast(device_type='mps', dtype=amp_dtype):
                        features = self.model.encode_image(input_tensor)
                else:
                    features = self.model.encode_image(input_tensor)
                features = features.float()  # cast back to float32 for normalization
                features = features / features.norm(dim=-1, keepdim=True)
            
            result = features.cpu().numpy()
            del features, input_tensor, batch_tensors
            return result
        except Exception as e:
            safe_print(f"[ERROR] Image encoding failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def encode_tensor_batch(self, tensors):
        """Encode a batch of already-preprocessed tensors (torch.Tensor, shape [N,3,224,224]).

        Called by _process_batch when preprocessing was done in prefetch workers.
        Skips the serial preprocess_image_pytorch loop that blocked the GPU in encode_image_batch.
        encode_image_batch (PIL path) is unchanged — still used by video indexing and image search.
        """
        import torch
        try:
            stacked = torch.stack(tensors)
            # pin_memory allows async non-blocking CPU->GPU transfer on CUDA — GPU doesn't
            # stall waiting for the copy to finish, improving utilization significantly.
            if torch.cuda.is_available():
                stacked = stacked.pin_memory()
                input_tensor = stacked.to(self.device, non_blocking=True)
            else:
                input_tensor = stacked.to(self.device)
            amp_dtype = getattr(self, 'amp_dtype', None)
            with torch.no_grad():
                if amp_dtype is not None and torch.cuda.is_available():
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        features = self.model.encode_image(input_tensor)
                elif amp_dtype is not None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    with torch.autocast(device_type='mps', dtype=amp_dtype):
                        features = self.model.encode_image(input_tensor)
                else:
                    features = self.model.encode_image(input_tensor)
                features = features.float()
                features = features / features.norm(dim=-1, keepdim=True)
            result = features.cpu().numpy()
            del features, input_tensor, tensors
            return result
        except Exception as e:
            safe_print(f"[ERROR] Tensor batch encoding failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def encode_text(self, texts):
        import torch
        
        try:
            tokens = self.tokenizer(texts)
            if isinstance(tokens, dict):
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
            else:
                tokens = tokens.to(self.device)
            amp_dtype = getattr(self, 'amp_dtype', None)
            with torch.no_grad():
                if amp_dtype is not None and torch.cuda.is_available():
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        if isinstance(tokens, dict):
                            features = self.model.encode_text(**tokens)
                        else:
                            features = self.model.encode_text(tokens)
                else:
                    if isinstance(tokens, dict):
                        features = self.model.encode_text(**tokens)
                    else:
                        features = self.model.encode_text(tokens)
                features = features.float()  # cast back to float32 for normalization
                features = features / features.norm(dim=-1, keepdim=True)
            
            result = features.cpu().numpy()
            del features, tokens
            safe_print(f"[ENCODE] Text encoded successfully, shape: {result.shape}")
            return result
        except Exception as e:
            safe_print(f"[ERROR] Text encoding failed: {e}")
            import traceback
            traceback.print_exc()
            raise


# ── SigLIP2 backend (google/siglip2 via HuggingFace transformers) ──────────────
class SigLIP2BackendModel:
    """
    Vision-language backend using Google SigLIP2 via the transformers library.
    Supports both text and image queries; higher quality than CLIP ViT-L-14.
    """
    def __init__(self, model_cfg):
        import torch
        self.model_cfg = model_cfg
        self.has_text  = True
        self.use_onnx_visual = False
        self.onnx_disabled   = True

        self.device, self.device_name, self.amp_dtype = _detect_device()
        safe_print(f"[MODEL] Using Device: {self.device_name}")
        if self.amp_dtype is not None:
            safe_print(f"[MODEL] Mixed precision enabled: {self.amp_dtype}")

        try:
            from transformers import AutoProcessor, AutoModel
        except ImportError as e:
            raise RuntimeError(
                "SigLIP2 requires the 'transformers' library (>=4.56).\n"
                "Install with: pip install transformers>=4.56"
            ) from e

        hf_id = model_cfg["hf_model_id"]
        input_size = model_cfg.get("input_size", 384)
        safe_print(f"[MODEL] Loading SigLIP2: {hf_id}")

        # Try offline first (uses local HuggingFace cache if the model was previously
        # downloaded), then fall back to a live download on first use.
        _prev_hf  = os.environ.get("HF_HUB_OFFLINE", "")
        _prev_tr  = os.environ.get("TRANSFORMERS_OFFLINE", "")
        try:
            os.environ["HF_HUB_OFFLINE"]     = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            self.processor = AutoProcessor.from_pretrained(hf_id, cache_dir=_HF_CACHE)
            self.hf_model  = AutoModel.from_pretrained(hf_id, cache_dir=_HF_CACHE)
            safe_print(f"[MODEL] Loaded from local cache")
        except Exception:
            safe_print(f"[MODEL] Not cached — downloading {hf_id} (first run only)...")
            os.environ["HF_HUB_OFFLINE"]     = "0"
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            try:
                self.processor = AutoProcessor.from_pretrained(hf_id, cache_dir=_HF_CACHE)
                self.hf_model  = AutoModel.from_pretrained(hf_id, cache_dir=_HF_CACHE)
                safe_print(f"[MODEL] Download complete!")
            except Exception as dl_err:
                raise RuntimeError(
                    f"Could not load {hf_id}.\n\n"
                    "SigLIP2 must be downloaded once before it can be used offline.\n"
                    "Please ensure internet access and try again, or switch to\n"
                    "CLIP ViT-L-14 (always available) via the Model button.\n\n"
                    f"Original error: {dl_err}"
                ) from dl_err
        finally:
            # Restore whatever offline setting was in effect before we changed it
            if _prev_hf:
                os.environ["HF_HUB_OFFLINE"] = _prev_hf
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)
            if _prev_tr:
                os.environ["TRANSFORMERS_OFFLINE"] = _prev_tr
            else:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)

        try:
            self.hf_model = self.hf_model.to(self.device).eval()
        except Exception as gpu_err:
            safe_print(f"[MODEL] GPU transfer failed ({gpu_err}), falling back to CPU")
            self.device      = torch.device("cpu")
            self.device_name = "CPU (GPU fallback)"
            self.amp_dtype   = None
            self.hf_model    = self.hf_model.to(self.device).eval()

        # Torchvision preprocess compatible with the indexing worker threads.
        # SigLIP2 uses 0.5/0.5 normalisation (maps [0,1] → [-1,1]).
        import torchvision.transforms as T
        self.preprocess = T.Compose([
            T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        safe_print(f"[MODEL] Ready!\n")

    def encode_image_batch(self, images):
        import torch
        tensors = [self.preprocess(img.convert("RGB")) for img in images]
        return self.encode_tensor_batch(tensors)

    def encode_tensor_batch(self, tensors):
        import torch
        stacked = torch.stack(tensors)
        if torch.cuda.is_available():
            stacked = stacked.pin_memory()
            pixel_values = stacked.to(self.device, non_blocking=True)
        else:
            pixel_values = stacked.to(self.device)

        with torch.no_grad():
            if self.amp_dtype and torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    features = self.hf_model.get_image_features(pixel_values=pixel_values)
            elif self.amp_dtype and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                with torch.autocast(device_type="mps", dtype=self.amp_dtype):
                    features = self.hf_model.get_image_features(pixel_values=pixel_values)
            else:
                features = self.hf_model.get_image_features(pixel_values=pixel_values)

        # Some transformers versions return a BaseModelOutputWithPooling dataclass
        # instead of a raw tensor.  Unwrap to the pooled embedding.
        if not isinstance(features, torch.Tensor):
            if hasattr(features, 'pooler_output') and features.pooler_output is not None:
                features = features.pooler_output
            elif hasattr(features, 'last_hidden_state'):
                features = features.last_hidden_state[:, 0]  # CLS token
            else:
                features = features[0]

        features = features.float()
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    def encode_text(self, texts):
        import torch
        inputs = self.processor(text=texts, return_tensors="pt",
                                padding="max_length", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()
                  if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            if self.amp_dtype and torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    features = self.hf_model.get_text_features(**inputs)
            else:
                features = self.hf_model.get_text_features(**inputs)

        if not isinstance(features, torch.Tensor):
            if hasattr(features, 'pooler_output') and features.pooler_output is not None:
                features = features.pooler_output
            elif hasattr(features, 'last_hidden_state'):
                features = features.last_hidden_state[:, 0]
            else:
                features = features[0]

        features = features.float()
        features = features / features.norm(dim=-1, keepdim=True)
        result = features.cpu().numpy()
        safe_print(f"[ENCODE] Text encoded successfully, shape: {result.shape}")
        return result


# ── DINOv2 / DINOv3 backend (via dinotool) ─────────────────────────────────────
class DinoBackendModel:
    """
    Vision-only backend using Meta DINOv2 or DINOv3 via the dinotool library.
    Produces rich spatial visual features; does NOT support text queries.

    Install:  pip install dinotool
    DINOv3 access: requires HuggingFace account + model access grant.
                   Run: hf auth login
    """
    def __init__(self, model_cfg):
        import torch
        self.model_cfg = model_cfg
        self.has_text  = False
        self.use_onnx_visual = False
        self.onnx_disabled   = True

        self.device, self.device_name, self.amp_dtype = _detect_device()
        safe_print(f"[MODEL] Using Device: {self.device_name}")
        if self.amp_dtype is not None:
            safe_print(f"[MODEL] Mixed precision enabled: {self.amp_dtype}")

        try:
            from dinotool import DinoToolModel
        except ImportError as e:
            raise RuntimeError(
                "DINOv2/v3 backends require the 'dinotool' library.\n"
                "Install with: pip install dinotool"
            ) from e

        dino_name  = model_cfg["dino_name"]
        input_size = model_cfg.get("input_size", 518)
        safe_print(f"[MODEL] Loading dinotool: {dino_name}  (input {input_size}px)")

        if model_cfg.get("requires_hf_auth"):
            safe_print(f"[MODEL] NOTE: {dino_name} is a gated HuggingFace model.")
            _hf_tok = os.environ.get("HF_TOKEN", "")
            if _hf_tok:
                try:
                    import huggingface_hub as _hfh
                    _hfh.login(token=_hf_tok, add_to_git_credential=False)
                    safe_print(f"[MODEL] HuggingFace authentication OK")
                except Exception as _auth_err:
                    safe_print(f"[MODEL] HF auth warning: {_auth_err}")
            else:
                safe_print(f"[MODEL] WARNING: No HF token found — use the 'HF Token' button to add one.")

        # dinotool calls AutoModel.from_pretrained() with no torch_dtype, so weights
        # default to float32 (~28 GB for a 7B model).  We patch from_pretrained at
        # the class level to inject torch_dtype=fp16/bf16 so the model loads at half
        # precision (~14 GB) directly onto the target device — no post-load moves or
        # device-routing mismatches.
        _restore_fp = []
        if self.device.type != "cpu":
            cast_dtype = self.amp_dtype if self.amp_dtype is not None else torch.float16
            try:
                import transformers as _tf
                _orig_func = _tf.AutoModel.from_pretrained.__func__
                _cast = cast_dtype  # capture for closure
                _tok  = os.environ.get("HF_TOKEN") or None
                @classmethod
                def _fp_patched(cls, *args, **kwargs):
                    kwargs.setdefault('torch_dtype', _cast)
                    if _tok:
                        kwargs.setdefault('token', _tok)
                    return _orig_func(cls, *args, **kwargs)
                _tf.AutoModel.from_pretrained = _fp_patched
                _restore_fp.append((_tf, _orig_func))
                safe_print(f"[MODEL] Loading DINOv3 in {cast_dtype} on {self.device}...")
            except Exception as patch_err:
                safe_print(f"[MODEL] Could not enable {cast_dtype} load ({patch_err}), loading in fp32")
        else:
            safe_print(f"[MODEL] Loading DINOv3 on CPU...")

        try:
            self.dino_model = DinoToolModel(dino_name, device=str(self.device), verbose=True)
        except RuntimeError as oom_err:
            if "out of memory" in str(oom_err).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                safe_print(f"[MODEL] OOM loading DINOv3 to GPU — falling back to CPU")
                self.dino_model = DinoToolModel(dino_name, device="cpu", verbose=True)
                self.device      = torch.device("cpu")
                self.device_name = "CPU (insufficient VRAM for DINOv3)"
                self.amp_dtype   = None
            else:
                raise
        finally:
            for (_tf_ref, _orig_f) in _restore_fp:
                try:
                    _tf_ref.AutoModel.from_pretrained = classmethod(_orig_f)
                except Exception:
                    pass

        try:
            self.dino_model = self.dino_model.eval()
        except Exception:
            pass  # DinoToolModel may not be an nn.Module directly

        # Standard DINOv2/v3 preprocessing (ImageNet statistics).
        import torchvision.transforms as T
        self.input_size = input_size
        self.preprocess = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),  # handles RGBA / palette PNGs
            T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        safe_print(f"[MODEL] Ready!\n")

    def encode_image_batch(self, images):
        import torch
        tensors = [self.preprocess(img.convert("RGB")) for img in images]
        return self.encode_tensor_batch(tensors)

    def encode_tensor_batch(self, tensors):
        import torch
        stacked = torch.stack(tensors)
        if torch.cuda.is_available():
            stacked = stacked.pin_memory()
            input_tensor = stacked.to(self.device, non_blocking=True)
        else:
            input_tensor = stacked.to(self.device)

        with torch.no_grad():
            if self.amp_dtype and torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    raw = self.dino_model(input_tensor, features="frame")
            elif self.amp_dtype and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                with torch.autocast(device_type="mps", dtype=self.amp_dtype):
                    raw = self.dino_model(input_tensor, features="frame")
            else:
                raw = self.dino_model(input_tensor, features="frame")

        # Normalise to a (B, D) float32 numpy array regardless of what dinotool returns.
        feat = self._extract_feature_tensor(raw)
        feat_np = feat.float().cpu().numpy()
        norms = np.linalg.norm(feat_np, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return feat_np / norms

    @staticmethod
    def _extract_feature_tensor(raw):
        """Collapse whatever dinotool returns to a 2-D (B, D) tensor."""
        import torch
        if isinstance(raw, torch.Tensor):
            feat = raw
        elif hasattr(raw, "tensor"):           # dinotool wrapper object
            feat = raw.tensor
        elif hasattr(raw, "features"):        # LocalFeatures .features attribute
            feat = raw.features
        elif hasattr(raw, "full"):             # LocalFeatures .full() spatial grid
            feat = raw.full()
        elif hasattr(raw, "flat"):             # LocalFeatures .flat() patch sequence
            feat = raw.flat()
        else:
            feat = torch.as_tensor(raw)

        # Squeeze any spatial dims: (B,1,1,D) → (B,D), (B,1,D) → (B,D), etc.
        if feat.dim() == 4:
            B, H, W, D = feat.shape
            if H == 1 and W == 1:
                feat = feat.reshape(B, D)
            else:
                feat = feat.reshape(B, H * W * D)
        elif feat.dim() == 3:
            B, N, D = feat.shape
            if N == 1:
                feat = feat.squeeze(1)
            else:
                feat = feat.mean(dim=1)    # mean-pool patches as global descriptor
        return feat

    def encode_text(self, texts):
        raise RuntimeError(
            f"The active model ({self.model_cfg['label']}) has no text encoder.\n"
            "Switch to CLIP or SigLIP2 to use text search."
        )


# ── Model factory ───────────────────────────────────────────────────────────────
def create_model(model_key: str):
    """Instantiate and return the model backend for *model_key*."""
    cfg = MODEL_REGISTRY.get(model_key)
    if cfg is None:
        raise ValueError(f"Unknown model key: {model_key!r}")
    mtype = cfg["type"]
    if mtype == "openclip":
        return HybridCLIPModel(cfg)
    elif mtype == "siglip2":
        return SigLIP2BackendModel(cfg)
    elif mtype == "dinotool":
        return DinoBackendModel(cfg)
    else:
        raise ValueError(f"Unknown model type: {mtype!r}")


class ResultCard(QFrame):
    """A card widget displaying a single search result (image or video frame)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_path = None
        self._on_single_click = None
        self._on_double_click = None
        self._on_context_menu = None
        self.setFixedSize(CELL_WIDTH, CELL_HEIGHT)
        self.setStyleSheet(
            f"ResultCard {{ background-color: {CARD_BG}; border: 1px solid {BORDER};"
            f"              border-radius: 8px; }}"
            f"ResultCard:hover {{ border-color: {ACCENT_SECONDARY};"
            f"                   background-color: {CARD_HOVER}; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 4)
        layout.setSpacing(3)
        
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setStyleSheet("background: transparent;")
        layout.addWidget(self.img_label)
        
        self.select_cb = QCheckBox("Select")
        self.select_cb.setStyleSheet(f"color: {FG_MUTED}; font-size: 8pt; background: transparent;")
        layout.addWidget(self.select_cb, alignment=Qt.AlignmentFlag.AlignCenter)

        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet(
            f"color: {FG_MUTED}; font-size: 11px; border: none; background: transparent;"
        )
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._on_single_click:
                self._on_single_click(self._image_path, self)
        elif event.button() == Qt.MouseButton.RightButton:
            if self._on_context_menu:
                self._on_context_menu(event.globalPosition().toPoint(), self._image_path)
        super().mousePressEvent(event)
    
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._on_double_click:
                self._on_double_click(self._image_path)
        super().mouseDoubleClickEvent(event)


class ClickableImageLabel(QLabel):
    """QLabel that calls a callback when clicked, used for inline thumbnails in dialogs."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._on_click = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click to open in viewer")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._on_click:
            self._on_click()
        super().mousePressEvent(event)


class ResultsScrollArea(QScrollArea):
    """Scrollable area for result cards with rubber-band selection and DnD support."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet(f"background-color: {BG}; border: none;")

        self._container = QWidget()
        self._container.setStyleSheet(f"background-color: {BG};")
        self._grid = QGridLayout(self._container)
        self._grid.setSpacing(10)
        self._grid.setContentsMargins(10, 10, 22, 10)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setWidget(self._container)
        
        # Rubber-band setup
        self._rb_band = QRubberBand(QRubberBand.Shape.Rectangle, self.viewport())
        self._rb_start = QPoint()
        self._rb_active = False
        self._rb_pending = False
        self.viewport().installEventFilter(self)
        
        # DnD
        self.setAcceptDrops(True)
        self._on_drop = None  # callback
        
        self._on_press_background = None  # callback for right-click background
        self._get_cards_fn = None  # reference to app._get_all_cards
        self._on_rubber_band_select = None
        
        # Resize callback
        self._on_resize = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._on_resize:
            self._on_resize()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls and self._on_drop:
            path = urls[0].toLocalFile()
            self._on_drop(path)

    def contextMenuEvent(self, event):
        # Only show background context menu if not clicking on a card
        if self._on_press_background:
            self._on_press_background(event.globalPos())

    def eventFilter(self, obj, event):
        if obj is self.viewport():
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                pos = event.pos()
                hit_card = False
                if self._get_cards_fn:
                    for card in self._get_cards_fn():
                        card_pos = card.mapTo(self.viewport(), QPoint(0, 0))
                        if QRect(card_pos, card.size()).contains(pos):
                            hit_card = True
                            break
                if not hit_card:
                    self._rb_pending = True
                    self._rb_active = False
                    self._rb_start = pos
                    return False
            
            elif event.type() == QEvent.Type.MouseMove and self._rb_pending:
                pos = event.pos()
                dx = abs(pos.x() - self._rb_start.x())
                dy = abs(pos.y() - self._rb_start.y())
                if dx > 5 or dy > 5:
                    self._rb_active = True
                    rect = QRect(self._rb_start, pos).normalized()
                    self._rb_band.setGeometry(rect)
                    self._rb_band.show()
                return False
            
            elif event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                if self._rb_pending:
                    was_active = self._rb_active
                    self._rb_pending = False
                    self._rb_active = False
                    self._rb_band.hide()
                    if was_active and self._on_rubber_band_select:
                        pos = event.pos()
                        sel_rect = QRect(self._rb_start, pos).normalized()
                        deselect = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                        self._on_rubber_band_select(sel_rect, deselect)
                    return False
        return super().eventFilter(obj, event)


class ModelSelectorDialog(QDialog):
    """
    Dialog for choosing the active embedding model.
    Shows all models in MODEL_REGISTRY with labels, subtitles, and capability badges.
    """
    def __init__(self, parent, current_key: str):
        super().__init__(parent)
        self.setWindowTitle("Select Embedding Model")
        self.resize(620, 420)
        self.selected_key = current_key

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        hdr = QLabel(
            "Choose the model used to index and search your media.\n"
            "Models with a text encoder support both text queries and image queries.\n"
            "Vision-only models support image-similarity search only.\n\n"
            "Note: switching models requires re-indexing your folder."
        )
        hdr.setWordWrap(True)
        hdr.setStyleSheet(f"color: {FG_MUTED}; font-size: 9pt;")
        layout.addWidget(hdr)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            f"QListWidget::item {{ padding: 8px 10px; border-bottom: 1px solid {BORDER}; border-radius: 0; }}"
            f"QListWidget::item:selected {{ background: {ACCENT_SECONDARY}; color: white; }}"
            f"QListWidget::item:hover:!selected {{ background: {CARD_BG}; }}"
        )
        self.list_widget.setFont(QFont("Segoe UI", 10))

        self._keys = list(MODEL_REGISTRY.keys())
        for key in self._keys:
            cfg = MODEL_REGISTRY[key]
            badge = "✓ Text + Image" if cfg["has_text"] else "⬛ Image only"
            hf_note = "  [HF login required]" if cfg.get("requires_hf_auth") else ""
            text = f"{cfg['label']}  —  {badge}{hf_note}\n  {cfg['subtitle']}"
            item = QListWidgetItem(text)
            if key == current_key:
                item.setForeground(__import__('PyQt6.QtGui', fromlist=['QColor']).QColor(ACCENT))
            self.list_widget.addItem(item)
            if key == current_key:
                self.list_widget.setCurrentRow(len(self._keys) - 1 -
                    (len(self._keys) - 1 - self._keys.index(key)))

        # Select current model row
        try:
            self.list_widget.setCurrentRow(self._keys.index(current_key))
        except ValueError:
            pass

        self.list_widget.itemDoubleClicked.connect(self._accept_selection)
        layout.addWidget(self.list_widget, stretch=1)

        # Install-hint label
        self.hint_label = QLabel("")
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(f"color: {ORANGE}; font-size: 9pt;")
        layout.addWidget(self.hint_label)
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        self._on_row_changed(self.list_widget.currentRow())

        btn_row = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_select = QPushButton("Use This Model")
        btn_select.setProperty("class", "accent")
        btn_select.clicked.connect(self._accept_selection)
        btn_row.addWidget(btn_cancel)
        btn_row.addStretch()
        btn_row.addWidget(btn_select)
        layout.addLayout(btn_row)

    def _on_row_changed(self, row):
        if row < 0 or row >= len(self._keys):
            self.hint_label.setText("")
            return
        key = self._keys[row]
        cfg = MODEL_REGISTRY[key]
        hints = []
        mtype = cfg["type"]
        if mtype == "dinotool":
            hints.append("Requires: pip install dinotool")
        if mtype == "siglip2":
            hints.append("Requires: pip install transformers>=4.56")
        if cfg.get("requires_hf_auth"):
            hints.append("Requires HuggingFace access: run  hf auth login")
        self.hint_label.setText("  ".join(hints) if hints else "")

    def _accept_selection(self):
        row = self.list_widget.currentRow()
        if 0 <= row < len(self._keys):
            self.selected_key = self._keys[row]
        self.accept()


class FolderDropListWidget(QListWidget):
    """QListWidget that accepts folder drag-and-drop from the OS file manager."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setToolTip("Drag folders here to add them")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            # Accept only if at least one URL is a local directory
            if any(url.isLocalFile() and os.path.isdir(url.toLocalFile())
                   for url in event.mimeData().urls()):
                event.acceptProposedAction()
                return
        event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        existing = [self.item(i).text() for i in range(self.count())]
        added = 0
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if os.path.isdir(path) and path not in existing:
                    self.addItem(path)
                    existing.append(path)
                    added += 1
        if added:
            event.acceptProposedAction()
        else:
            event.ignore()


class ImageSearchApp(QMainWindow):
    # Used by _safe_after to marshal arbitrary callables to the main thread.
    # Emitting from a background thread automatically uses a queued connection
    # because self lives in the main thread.
    _dispatch_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._dispatch_signal.connect(self._dispatch_invoke)
        self.setWindowTitle("PhotoSearchPro - AI Media Search")
        self.resize(1400, 900)

        if os.name == 'nt':
            self.apply_dark_title_bar()

        # ── Active model ──────────────────────────────────────────────────────
        _settings = _load_app_settings()

        # Apply saved HuggingFace token so downloads work without any manual setup
        _hf_token = _settings.get("hf_token", "")
        if _hf_token and not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = _hf_token
            try:
                import huggingface_hub as _hfh
                _hfh.login(token=_hf_token, add_to_git_credential=False)
            except Exception:
                pass

        self.active_model_key = _settings.get("model_key", DEFAULT_MODEL_KEY)
        if self.active_model_key not in MODEL_REGISTRY:
            self.active_model_key = DEFAULT_MODEL_KEY
        self.model_has_text = MODEL_REGISTRY[self.active_model_key]["has_text"]

        self.folder = None      # primary folder (for cache / exclusions / history)
        self.folders = []       # all selected folders (including primary)
        self.cache_file = None
        self.image_paths = []  # NOW STORES ABSOLUTE PATHS
        self.image_embeddings = None
        self.thumbnail_images = {}
        self.selected_images = set()
        self.excluded_folders = set()

        # Video index — parallel to image index
        self.video_paths = []         # list of (rel_video_path_str, timestamp_float)
        self.video_embeddings = None  # numpy array (M, 512)
        self.video_cache_file = None
        self._pending_video_refresh = False

        # Face recognition index and presets
        self._face_app = None         # insightface FaceAnalysis (lazy-loaded)
        self.face_index = {}          # rel_path -> list of 512d ArcFace embeddings
        self.face_presets = {}        # name -> {"embedding": np.array, "references": [rel_paths]}

        # Pending batch accumulators — filled during indexing, flushed before save/search
        # Avoids O(N^2) np.concatenate per batch for large collections
        self._pending_image_batches = []
        self._pending_video_batches = []
        self._cache_lock = threading.Lock()  # guards flush+save vs live search race

        # Failed file tracking — populated during indexing, written to log at end of run
        self._failed_images = []   # list of (abs_path, reason)
        self._failed_videos = []   # list of (abs_path, reason)
        self._last_failed_images = []   # snapshot of failures from last completed run
        self._last_failed_videos = []   # snapshot of failures from last completed run

        # Hybrid search anchor image embedding
        self._anchor_embed = None   # numpy array (D,) if anchor set
        self._anchor_path = None    # str path for display

        # Last image used for image search (enables Re-run)
        self._last_image_search_path = None
        self._last_search_type = None   # 'text' or 'image'

        # File-system watcher for auto-incremental index
        self._fs_watcher = QFileSystemWatcher()
        self._fs_debounce_timer = QTimer()
        self._fs_debounce_timer.setSingleShot(True)
        self._fs_debounce_timer.timeout.connect(self._trigger_auto_refresh)
        self._fs_watcher.directoryChanged.connect(self._on_folder_changed)

        self.clip_model = None
        self.model_loading = False
        
        self.is_indexing = False
        self.stop_indexing = False
        self.is_stopping = False
        
        self.is_searching = False
        self.stop_search = False
        self.search_thread = None
        self.index_thread = None
        
        # Click timer for single vs double click disambiguation
        self.click_timer = QTimer()
        self.click_timer.setSingleShot(True)
        
        self.total_found = 0
        self.search_generation = 0
        self.render_cols = 1
        self.thumbnail_count = 0  # reliable counter for grid placement
        self.thumbnail_queue = queue.Queue()
        self.all_search_results = []   # stores ALL sorted results in memory
        self.show_more_offset = 0      # how many results currently displayed
        self._stored_all_results = []  # backup for pagination
        self._thumbnail_worker_thread = None  # tracks active thumbnail loader thread
        
        # Queue for pending actions after stop
        self.pending_action = None
        
        # Search history & saved presets
        self.search_history = []   # list of {"query": str, "timestamp": str}
        self.search_presets = []   # list of {"name": str, "query": str}

        self.build_ui()
        Thread(target=self.load_model, daemon=True).start()

    def apply_dark_title_bar(self):
        try:
            import ctypes
            hwnd = int(self.winId())
            value = ctypes.c_int(1)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(value), ctypes.sizeof(value))
        except:
            pass

    def closeEvent(self, event):
        """Handle window close — warn if indexing is running."""
        if not self.is_indexing and not self.is_stopping:
            event.accept()
            os._exit(0)
        else:
            event.ignore()
            self._show_close_dialog()

    def get_cache_filename(self):
        # Each model gets its own cache file keyed by model.cache_key.
        # CLIP ViT-L-14 keeps the old filename for backward-compat with existing indices.
        key = MODEL_REGISTRY[self.active_model_key]["cache_key"]
        return [f".clip_cache_{key}.pkl"]

    def get_video_cache_filename(self):
        key = MODEL_REGISTRY[self.active_model_key]["cache_key"]
        return f".clip_cache_videos_{key}.pkl"

    def get_exclusions_path(self):
        if not self.folder:
            return None
        return os.path.join(self.folder, ".clip_exclusions.json")

    def _is_excluded(self, path):
        if not self.excluded_folders:
            return False
        # Derive relative path from whichever base folder the path belongs to.
        # If path is already relative (old callers) or doesn't match any folder,
        # use it as-is so pattern matching still works.
        rel_path = path
        for base in (self.folders if self.folders else ([self.folder] if self.folder else [])):
            try:
                candidate = os.path.relpath(path, base).replace('\\', '/')
                if not candidate.startswith('..'):
                    rel_path = candidate
                    break
            except ValueError:
                pass
        normalized = rel_path.replace(os.sep, "/")
        return any(pattern in normalized for pattern in self.excluded_folders)

    def load_exclusions(self):
        path = self.get_exclusions_path()
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.excluded_folders = set(data.get("excluded_patterns", []))
                safe_print(f"[EXCLUSIONS] Loaded {len(self.excluded_folders)} pattern(s)")
            except Exception as e:
                safe_print(f"[EXCLUSIONS] Load error: {e}")
                self.excluded_folders = set()
        else:
            self.excluded_folders = set()

    def save_exclusions(self):
        path = self.get_exclusions_path()
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"excluded_patterns": sorted(self.excluded_folders)}, f, indent=2)
            safe_print(f"[EXCLUSIONS] Saved {len(self.excluded_folders)} pattern(s)")
        except Exception as e:
            safe_print(f"[EXCLUSIONS] Save error: {e}")

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 6, 8, 6)
        main_layout.setSpacing(4)

        # --- Toolbar ---
        toolbar_widget = QWidget()
        toolbar_widget.setObjectName("toolbarPanel")
        toolbar_widget.setStyleSheet(
            f"QWidget#toolbarPanel {{ background-color: {PANEL_BG};"
            f"  border-bottom: 1px solid {BORDER}; }}"
        )
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(8, 6, 8, 6)
        toolbar_layout.setSpacing(5)

        self.btn_folder = QPushButton("Folders")
        self.btn_folder.clicked.connect(self.on_select_folder)
        toolbar_layout.addWidget(self.btn_folder)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.on_force_reindex)
        toolbar_layout.addWidget(self.btn_refresh)

        self.btn_index_videos = QPushButton("Index Videos")
        self.btn_index_videos.clicked.connect(self.on_index_videos_click)
        toolbar_layout.addWidget(self.btn_index_videos)

        self.btn_model = QPushButton(f"Model: {MODEL_REGISTRY[self.active_model_key]['label']}")
        self.btn_model.setProperty("class", "accent")
        self.btn_model.setToolTip("Click to change the active embedding model")
        self.btn_model.clicked.connect(self.open_model_selector)
        toolbar_layout.addWidget(self.btn_model)

        btn_unload = QPushButton("Unload Model")
        btn_unload.setToolTip("Move the active model off the GPU to free VRAM")
        btn_unload.clicked.connect(self.on_unload_model)
        toolbar_layout.addWidget(btn_unload)
        toolbar_layout.addSpacing(8)

        self.btn_stop = QPushButton("Stop Index")
        self.btn_stop.setProperty("class", "danger")
        self.btn_stop.clicked.connect(self.stop_indexing_process)
        toolbar_layout.addWidget(self.btn_stop)

        btn_exit = QPushButton("Exit")
        btn_exit.setProperty("class", "danger")
        btn_exit.clicked.connect(self.force_quit)
        toolbar_layout.addWidget(btn_exit)

        btn_exclusions = QPushButton("Exclusions")
        btn_exclusions.clicked.connect(self.open_exclusions_dialog)
        toolbar_layout.addWidget(btn_exclusions)

        btn_duplicates = QPushButton("Duplicates")
        btn_duplicates.clicked.connect(self.on_find_duplicates)
        toolbar_layout.addWidget(btn_duplicates)

        btn_smart = QPushButton("Smart Albums")
        btn_smart.clicked.connect(self.on_smart_albums)
        toolbar_layout.addWidget(btn_smart)

        btn_nsfw = QPushButton("NSFW Scan")
        btn_nsfw.setToolTip("Scan indexed images with NudeNet (pip install nudenet)")
        btn_nsfw.clicked.connect(self.on_nsfw_scan)
        toolbar_layout.addWidget(btn_nsfw)

        btn_faces = QPushButton("Face Presets")
        btn_faces.setToolTip(
            "Create named presets for people and search your library by face.\n"
            "Uses InsightFace ArcFace (buffalo_l) — pip install insightface onnxruntime"
        )
        btn_faces.clicked.connect(self.on_face_presets)
        toolbar_layout.addWidget(btn_faces)

        self.status_label = QLabel("Starting...")
        self.status_label.setMinimumWidth(240)
        self.status_label.setStyleSheet(f"color: {FG_MUTED}; font-size: 9pt;")
        toolbar_layout.addWidget(self.status_label)

        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet(f"color: {FG_MUTED}; font-size: 8pt;")
        toolbar_layout.addWidget(self.stats_label)

        toolbar_layout.addStretch()

        self.device_label = QLabel("...")
        self.device_label.setStyleSheet(
            f"color: {ACCENT_SECONDARY}; font-size: 8pt; font-weight: 600;"
        )
        toolbar_layout.addWidget(self.device_label)

        btn_hf_token = QPushButton("HF Token")
        btn_hf_token.setToolTip("Set your HuggingFace token (needed for SigLIP2 / DINOv3 downloads)")
        btn_hf_token.clicked.connect(self.set_hf_token)
        toolbar_layout.addWidget(btn_hf_token)

        self.auto_update_cb = QCheckBox("Auto-Update")
        self.auto_update_cb.setToolTip(
            "Watch the indexed folder for new files and automatically refresh the index when changes are detected."
        )
        self.auto_update_cb.setChecked(False)
        self.auto_update_cb.toggled.connect(self._on_auto_update_toggled)
        toolbar_layout.addWidget(self.auto_update_cb)

        btn_info = QPushButton("?")
        btn_info.setFixedSize(28, 28)
        btn_info.setStyleSheet(
            "QPushButton { padding: 0px; font-weight: 700; font-size: 13pt;"
            f"  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"    stop:0 #2d333b, stop:1 {CARD_BG});"
            f"  border: 1px solid #3a4050; border-radius: 6px; color: {FG_MUTED}; }}"
            f"QPushButton:hover {{ border-color: {FG_MUTED}; color: {FG}; }}"
        )
        btn_info.clicked.connect(self.show_index_info)
        toolbar_layout.addWidget(btn_info)

        main_layout.addWidget(toolbar_widget)

        # --- Search bar ---
        search_widget = QWidget()
        search_widget.setObjectName("searchPanel")
        search_widget.setStyleSheet(
            f"QWidget#searchPanel {{ background-color: {PANEL_BG};"
            f"  border-bottom: 1px solid {BORDER}; }}"
        )
        search_layout = QHBoxLayout(search_widget)
        search_layout.setContentsMargins(10, 7, 10, 7)
        search_layout.setSpacing(7)

        lbl_search = QLabel("Search:")
        lbl_search.setStyleSheet(f"color: {FG_MUTED}; font-weight: 600; font-size: 9pt;")
        search_layout.addWidget(lbl_search)

        self.query_entry = QLineEdit()
        self.query_entry.setFont(QFont("Segoe UI", 12))
        self.query_entry.setMinimumHeight(34)
        self.query_entry.setMaxLength(500)
        self.query_entry.returnPressed.connect(self.on_search_click)
        self.query_entry.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.query_entry.customContextMenuRequested.connect(self._show_search_context_menu)
        search_layout.addWidget(self.query_entry, stretch=1)

        self.btn_search = QPushButton("Search")
        self.btn_search.setProperty("class", "accent")
        self.btn_search.clicked.connect(self.on_search_click)
        search_layout.addWidget(self.btn_search)

        btn_image = QPushButton("Image")
        btn_image.clicked.connect(self.on_image_click)
        search_layout.addWidget(btn_image)

        self.last_image_thumb = QLabel()
        self.last_image_thumb.setFixedSize(32, 32)
        self.last_image_thumb.setScaledContents(True)
        self.last_image_thumb.setStyleSheet(
            f"border: 1px solid {BORDER}; border-radius: 3px;"
        )
        self.last_image_thumb.setVisible(False)
        search_layout.addWidget(self.last_image_thumb)

        self.btn_rerun_image = QPushButton("Re-run")
        self.btn_rerun_image.setToolTip("Re-run image search with the same image")
        self.btn_rerun_image.clicked.connect(self.on_rerun_image_click)
        self.btn_rerun_image.setVisible(False)
        search_layout.addWidget(self.btn_rerun_image)

        self.btn_clear_image_search = QPushButton("X")
        self.btn_clear_image_search.setFixedWidth(26)
        self.btn_clear_image_search.setStyleSheet("QPushButton { padding: 0px; }")
        self.btn_clear_image_search.setToolTip("Clear loaded image")
        self.btn_clear_image_search.clicked.connect(self._clear_image_search)
        self.btn_clear_image_search.setVisible(False)
        search_layout.addWidget(self.btn_clear_image_search)

        btn_paste = QPushButton("Paste")
        btn_paste.setToolTip("Search by image from clipboard (Ctrl+V)")
        btn_paste.clicked.connect(self._paste_clipboard_search)
        search_layout.addWidget(btn_paste)

        self.btn_anchor = QPushButton("Anchor")
        self.btn_anchor.setToolTip(
            "Set a reference image to blend with text search.\n"
            "Use the weight slider to control text vs. image balance."
        )
        self.btn_anchor.clicked.connect(self._set_anchor_image)
        search_layout.addWidget(self.btn_anchor)

        self.anchor_label = QLabel("")
        self.anchor_label.setStyleSheet(f"color: {FG_MUTED}; font-size: 9pt;")
        self.anchor_label.setVisible(False)
        search_layout.addWidget(self.anchor_label)

        self.btn_clear_anchor = QPushButton("✕")
        self.btn_clear_anchor.setFixedWidth(26)
        self.btn_clear_anchor.setToolTip("Clear anchor image")
        self.btn_clear_anchor.clicked.connect(self._clear_anchor)
        self.btn_clear_anchor.setVisible(False)
        search_layout.addWidget(self.btn_clear_anchor)

        search_layout.addWidget(QLabel("Blend:"))
        self.hybrid_slider = QSlider(Qt.Orientation.Horizontal)
        self.hybrid_slider.setRange(0, 100)
        self.hybrid_slider.setValue(0)
        self.hybrid_slider.setFixedWidth(90)
        self.hybrid_slider.setToolTip("0 = text only, 100 = anchor image only")
        self.hybrid_slider.setVisible(False)
        search_layout.addWidget(self.hybrid_slider)

        self.hybrid_val_label = QLabel("0%")
        self.hybrid_val_label.setStyleSheet(f"color: {FG_MUTED}; font-size: 9pt;")
        self.hybrid_val_label.setVisible(False)
        self.hybrid_slider.valueChanged.connect(
            lambda v: self.hybrid_val_label.setText(f"{v}%"))
        search_layout.addWidget(self.hybrid_val_label)

        btn_history = QPushButton("History")
        btn_history.clicked.connect(self.on_history_click)
        search_layout.addWidget(btn_history)

        main_layout.addWidget(search_widget)

        # --- Controls bar ---
        controls_widget = QWidget()
        controls_widget.setObjectName("controlsPanel")
        controls_widget.setStyleSheet(
            f"QWidget#controlsPanel {{ background-color: {PANEL_BG};"
            f"  border-bottom: 1px solid {BORDER}; }}"
        )
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(10, 5, 10, 5)
        controls_layout.setSpacing(5)

        lbl_sim = QLabel("Similarity:")
        lbl_sim.setStyleSheet(f"color: {FG_MUTED}; font-size: 9pt;")
        controls_layout.addWidget(lbl_sim)
        self.score_slider = QSlider(Qt.Orientation.Horizontal)
        self.score_slider.setRange(0, 100)
        self.score_slider.setValue(15)  # 0.15 default
        self.score_slider.setFixedWidth(140)
        self.score_val_label = QLabel("0.15")
        self.score_slider.valueChanged.connect(
            lambda v: self.score_val_label.setText(f"{v/100.0:.2f}"))
        controls_layout.addWidget(self.score_slider)
        controls_layout.addWidget(self.score_val_label)

        controls_layout.addSpacing(6)
        lbl_rpp = QLabel("Per Page:")
        lbl_rpp.setStyleSheet(f"color: {FG_MUTED}; font-size: 9pt;")
        controls_layout.addWidget(lbl_rpp)
        self.top_n_slider = QSlider(Qt.Orientation.Horizontal)
        self.top_n_slider.setRange(1, 50)
        self.top_n_slider.setValue(3)  # 3*10=30 default
        self.top_n_slider.setFixedWidth(170)
        self.top_n_val_label = QLabel("30")
        self.top_n_slider.valueChanged.connect(
            lambda v: self.top_n_val_label.setText(str(v * 10)))
        controls_layout.addWidget(self.top_n_slider)
        controls_layout.addWidget(self.top_n_val_label)

        btn_clear = QPushButton("Clear Results")
        btn_clear.clicked.connect(self.on_clear_click)
        controls_layout.addWidget(btn_clear)

        btn_copy = QPushButton("Copy")
        btn_copy.clicked.connect(self.on_copy_click)
        controls_layout.addWidget(btn_copy)

        btn_move = QPushButton("Move")
        btn_move.setProperty("class", "accent")
        btn_move.clicked.connect(self.on_move_click)
        controls_layout.addWidget(btn_move)

        btn_delete = QPushButton("Delete")
        btn_delete.setProperty("class", "danger")
        btn_delete.clicked.connect(self.on_delete_click)
        controls_layout.addWidget(btn_delete)

        btn_select_all = QPushButton("Select All")
        btn_select_all.clicked.connect(self._select_all_cards)
        controls_layout.addWidget(btn_select_all)

        btn_deselect_all = QPushButton("Deselect All")
        btn_deselect_all.clicked.connect(self._deselect_all_cards)
        controls_layout.addWidget(btn_deselect_all)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setFixedWidth(1)
        sep1.setFixedHeight(20)
        sep1.setStyleSheet(f"background-color: {BORDER}; border: none;")
        controls_layout.addWidget(sep1)

        self.show_images_cb = QCheckBox("Images")
        self.show_images_cb.setChecked(True)
        controls_layout.addWidget(self.show_images_cb)

        self.show_videos_cb = QCheckBox("Videos")
        self.show_videos_cb.setChecked(True)
        controls_layout.addWidget(self.show_videos_cb)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setFixedWidth(1)
        sep2.setFixedHeight(20)
        sep2.setStyleSheet(f"background-color: {BORDER}; border: none;")
        controls_layout.addWidget(sep2)

        self.dedup_video_cb = QCheckBox("Best Frame/Video")
        self.dedup_video_cb.setChecked(False)
        controls_layout.addWidget(self.dedup_video_cb)

        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.VLine)
        sep3.setFixedWidth(1)
        sep3.setFixedHeight(20)
        sep3.setStyleSheet(f"background-color: {BORDER}; border: none;")
        controls_layout.addWidget(sep3)

        lbl_sort = QLabel("Sort:")
        lbl_sort.setStyleSheet(f"color: {FG_MUTED}; font-size: 9pt;")
        controls_layout.addWidget(lbl_sort)
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Score ↓", "Date (Newest)", "Date (Oldest)", "Name A→Z", "Name Z→A", "Size ↓"])
        self.sort_combo.setFixedWidth(130)
        self.sort_combo.setToolTip("Re-sort the current search results")
        self.sort_combo.currentIndexChanged.connect(self._resort_and_redisplay)
        controls_layout.addWidget(self.sort_combo)

        sep4 = QFrame()
        sep4.setFrameShape(QFrame.Shape.VLine)
        sep4.setFixedWidth(1)
        sep4.setFixedHeight(20)
        sep4.setStyleSheet(f"background-color: {BORDER}; border: none;")
        controls_layout.addWidget(sep4)

        self.auto_find_cb = QCheckBox("Auto-find")
        self.auto_find_cb.setChecked(False)
        self.auto_find_cb.setToolTip(
            "When enabled, automatically lowers the similarity score\n"
            "and retries the search until at least one result is found."
        )
        controls_layout.addWidget(self.auto_find_cb)

        main_layout.addWidget(controls_widget)

        # --- Progress bar ---
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(8)
        main_layout.addWidget(self.progress)

        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.progress_label)

        # --- Inline info bar (replaces chatty popups) ---
        self.info_bar = QLabel("")
        self.info_bar.setWordWrap(True)
        self.info_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_bar.setStyleSheet(
            f"color: {FG};"
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            f"  stop:0 #1a2235, stop:0.6 #161b22, stop:1 {PANEL_BG});"
            f"border: 1px solid {BORDER};"
            f"border-left: 3px solid {ACCENT_SECONDARY};"
            f"border-radius: 6px; padding: 6px 12px; font-size: 9pt;"
        )
        self.info_bar.setVisible(False)
        main_layout.addWidget(self.info_bar)

        # --- Page navigation ---
        self.page_nav_widget = QWidget()
        page_nav_layout = QHBoxLayout(self.page_nav_widget)
        page_nav_layout.setContentsMargins(0, 0, 0, 0)
        page_nav_layout.addStretch()

        self.prev_page_btn = QPushButton("← Prev Page")
        self.prev_page_btn.setProperty("class", "accent")
        self.prev_page_btn.clicked.connect(self.prev_page_results)
        page_nav_layout.addWidget(self.prev_page_btn)

        self.page_label = QLabel("")
        self.page_label.setStyleSheet(
            f"color: {FG_MUTED}; font-size: 9pt; font-weight: 600;"
            f"padding: 0 12px;"
        )
        page_nav_layout.addWidget(self.page_label)

        self.show_more_btn = QPushButton("Next Page →")
        self.show_more_btn.setProperty("class", "accent")
        self.show_more_btn.clicked.connect(self.show_more_results)
        page_nav_layout.addWidget(self.show_more_btn)

        page_nav_layout.addStretch()
        self.page_nav_widget.setVisible(False)
        main_layout.addWidget(self.page_nav_widget)

        # --- Results scroll area ---
        self.scroll_area = ResultsScrollArea()
        self.scroll_area._on_drop = self._on_drop_image
        self.scroll_area._get_cards_fn = self._get_all_cards
        self.scroll_area._on_rubber_band_select = self._on_rubber_band_select
        self.scroll_area._on_press_background = self._show_canvas_context_menu
        self.scroll_area._on_resize = self.on_scroll_area_resize
        main_layout.addWidget(self.scroll_area, stretch=1)

        # Apply initial text-search state (may be disabled for vision-only models)
        self._update_text_search_state()

    # ---- UI helpers ----

    def update_status(self, text, color="blue"):
        color_map = {
            "green":  ACCENT,
            "orange": ORANGE,
            "red":    DANGER,
            "blue":   ACCENT_SECONDARY,
        }
        c = color_map.get(color, color)
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {c}; font-size: 9pt; font-weight: 600;")

    def show_info_bar(self, msg: str):
        """Display a non-blocking informational message in the inline info bar."""
        self.info_bar.setText(msg)
        self.info_bar.setVisible(True)

    def hide_info_bar(self):
        self.info_bar.setVisible(False)

    def update_stats(self):
        has_images = self.image_embeddings is not None and len(self.image_paths) > 0
        has_videos = self.video_embeddings is not None and len(self.video_paths) > 0

        if has_images and has_videos:
            n_imgs = len(self.image_paths)
            n_frames = len(self.video_paths)
            n_vids = len(set(vp for vp, _ in self.video_paths))
            self.stats_label.setText(f"{n_imgs:,} images | {n_vids:,} videos ({n_frames:,} frames)")
        elif has_images:
            self.stats_label.setText(f"{len(self.image_paths):,} images indexed")
        elif has_videos:
            n_frames = len(self.video_paths)
            n_vids = len(set(vp for vp, _ in self.video_paths))
            self.stats_label.setText(f"{n_vids:,} videos ({n_frames:,} frames)")
        else:
            self.stats_label.setText("")

    def update_progress(self, value, text):
        self.progress.setRange(0, 100)
        self.progress.setValue(int(value))
        self.progress_label.setText(text)

    def _dispatch_invoke(self, fn):
        """Slot that receives callables from background threads via _dispatch_signal."""
        try:
            fn()
        except Exception as e:
            import traceback
            safe_print(f"[DISPATCH ERROR] {e}\n{traceback.format_exc()}")

    def _safe_after(self, ms, func):
        """Schedule func on the main-thread event loop from any thread.

        Uses _dispatch_signal (pyqtSignal) which Qt automatically delivers via
        a queued connection when emitted from a non-main thread, marshalling
        the call safely to the main thread's event loop.

        For ms=0: emits directly.
        For ms>0: emits a wrapper that schedules a QTimer on the main thread
                  (2-arg QTimer.singleShot is safe when called from the main thread).
        """
        import threading
        safe_print(f"[SAFE_AFTER] scheduling {getattr(func, '__name__', repr(func)[:60])} "
                   f"in {ms}ms from thread={threading.current_thread().name}")
        try:
            if ms <= 0:
                self._dispatch_signal.emit(func)
            else:
                def _schedule():
                    QTimer.singleShot(ms, func)
                self._dispatch_signal.emit(_schedule)
        except Exception as e:
            safe_print(f"[SAFE_AFTER ERROR] {e}")

    def on_scroll_area_resize(self):
        vp_width = self.scroll_area.viewport().width()
        new_cols = max(1, vp_width // CELL_WIDTH)
        if new_cols != self.render_cols:
            self.render_cols = new_cols
            self._reflow_grid()

    def _reflow_grid(self, cards=None):
        if cards is None:
            cards = self._get_all_cards()
        cols = max(1, getattr(self, 'render_cols', 1))
        for idx, card in enumerate(cards):
            r, c = divmod(idx, cols)
            self.scroll_area._grid.addWidget(card, r, c)
        self.thumbnail_count = len(cards)

    def _get_all_cards(self):
        """Return all ResultCard widgets currently in the grid."""
        cards = []
        layout = self.scroll_area._grid
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item and isinstance(item.widget(), ResultCard):
                cards.append(item.widget())
        return cards

    def _on_rubber_band_select(self, sel_rect, deselect):
        """Handle rubber-band selection completion."""
        for card in self._get_all_cards():
            card_pos = card.mapTo(self.scroll_area.viewport(), QPoint(0, 0))
            card_rect = QRect(card_pos, card.size())
            if sel_rect.intersects(card_rect):
                self._set_card_selection_by_path(card._image_path, not deselect)

    # ---- Model loading ----

    def load_model(self):
        self.model_loading = True
        cfg = MODEL_REGISTRY[self.active_model_key]
        self._safe_after(0, lambda: self.update_status(f"Loading {cfg['label']}...", "orange"))
        try:
            self.clip_model = create_model(self.active_model_key)
            self._safe_after(0, lambda: self.update_status("Ready", "green"))
            # Update model button label now that we know the real device
            self._safe_after(0, lambda: self.btn_model.setText(
                f"Model: {MODEL_REGISTRY[self.active_model_key]['label']}"))
            # Update text-search state in case model was changed at runtime
            self._safe_after(0, self._update_text_search_state)
            device = self.clip_model.device_name
            batch = BATCH_SIZE
            if "CUDA" in device:
                short = f"GPU  •  Batch {batch}"
            elif "Metal" in device:
                short = f"MPS  •  Batch {batch}"
            elif "DirectML" in device:
                short = f"DirectML  •  Batch {batch}"
            else:
                short = f"CPU  •  Batch {batch}"
            self._safe_after(0, lambda s=short: self.device_label.setText(s))
            safe_print(f"[LOAD] Success!\n")
        except Exception as e:
            safe_print(f"[ERROR] {e}")
            err_msg = str(e)
            # Detect CUDA DLL init failure (WinError 1114) and give actionable guidance
            if "1114" in err_msg or "DLL" in err_msg or "c10" in err_msg:
                hint = (
                    "PyTorch CUDA DLLs failed to load.\n\n"
                    "Common fixes:\n"
                    "1. Install Visual C++ Redistributable 2022:\n"
                    "   https://aka.ms/vs/17/release/vc_redist.x64.exe\n\n"
                    "2. Reinstall PyTorch matching your CUDA version, e.g.:\n"
                    "   pip install torch --index-url https://download.pytorch.org/whl/cu124\n\n"
                    "3. Or install CPU-only PyTorch:\n"
                    "   pip install torch --index-url https://download.pytorch.org/whl/cpu\n\n"
                    f"Original error:\n{err_msg}"
                )
                display_msg = hint
            else:
                display_msg = f"Failed to load model\n{err_msg}"
            self._safe_after(0, lambda: self.update_status("Load Failed", "red"))
            self._safe_after(0, lambda: self.device_label.setText("Load Failed"))
            self._safe_after(0, lambda m=display_msg: QMessageBox.critical(self, "Error", m))
        self.model_loading = False

    def on_unload_model(self):
        """Move the active model off the GPU and free all VRAM it holds."""
        if self.model_loading:
            QMessageBox.information(self, "Model Loading",
                                    "Please wait for the model to finish loading.")
            return
        if not self.is_safe_to_act(action_name="unload model"):
            return
        if self.clip_model is None:
            QMessageBox.information(self, "No Model Loaded", "No model is currently loaded.")
            return
        try:
            import torch, gc
            old_model = self.clip_model
            self.clip_model = None
            inner = getattr(old_model, 'model', None)
            if inner is not None and hasattr(inner, 'cpu'):
                inner.cpu()
            for attr in ('visual_model', 'text_model', '_model', 'dino_model'):
                sub = getattr(old_model, attr, None)
                if sub is not None:
                    sub_inner = getattr(sub, 'model', None)
                    if sub_inner is not None and hasattr(sub_inner, 'cpu'):
                        sub_inner.cpu()
                    elif hasattr(sub, 'cpu'):
                        sub.cpu()
            if hasattr(old_model, '_destroy_onnx_session'):
                old_model._destroy_onnx_session()
            del old_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            self.update_status("Model unloaded", "orange")
            self.device_label.setText("Unloaded")
            self._update_text_search_state()
        except Exception as e:
            QMessageBox.warning(self, "Unload Error", f"Error while unloading model:\n{e}")

    # ---- Model selection ----

    def _update_text_search_state(self):
        """Enable / disable text-search controls based on whether the active model
        has a text encoder.  Called after every model change."""
        has_text = MODEL_REGISTRY[self.active_model_key]["has_text"]
        self.model_has_text = has_text
        self.query_entry.setEnabled(has_text)
        self.btn_search.setEnabled(has_text)
        if has_text:
            self.query_entry.setPlaceholderText("")
        else:
            label = MODEL_REGISTRY[self.active_model_key]["label"]
            self.query_entry.setPlaceholderText(
                f"{label} is vision-only — use Image search")

    # ---- Feature: Clipboard paste image search ----

    def _paste_clipboard_search(self):
        """Search by image currently on the clipboard (Ctrl+V shortcut)."""
        if self.clip_model is None:
            QMessageBox.warning(self, "Wait", "Model is still loading, please wait.")
            return
        if not self.is_safe_to_act(action_name="image search"):
            return
        if not self.folder:
            QMessageBox.warning(self, "No Folder", "Please select a folder first.")
            return
        if self.image_embeddings is None and self.video_embeddings is None:
            QMessageBox.warning(self, "Not Indexed", "Please index a folder first.")
            return
        clipboard = QApplication.clipboard()
        mime = clipboard.mimeData()
        if mime.hasImage():
            qimg = clipboard.image()
            if qimg.isNull():
                QMessageBox.warning(self, "Empty Clipboard", "No image found on clipboard.")
                return
            # Convert QImage → PIL Image via RGB888 bytes
            qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
            w, h = qimg.width(), qimg.height()
            ptr = qimg.bits()
            ptr.setsize(w * h * 3)
            pil_img = Image.frombytes("RGB", (w, h), bytes(ptr))
            self.cancel_search(clear_ui=True)
            gen = self.search_generation + 1
            self.search_thread = Thread(
                target=lambda: self._image_search_pil(pil_img, gen, label="clipboard"),
                daemon=True,
            )
            self.search_thread.start()
        elif mime.hasUrls():
            # Clipboard has a file path — treat as image file
            for url in mime.urls():
                path = url.toLocalFile()
                if path and os.path.isfile(path):
                    self._on_drop_image(path)
                    return
            QMessageBox.warning(self, "No Image", "No image or image file found on clipboard.")
        else:
            QMessageBox.warning(self, "No Image",
                "No image found on clipboard.\n\nCopy an image first, then click Paste.")

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Paste):
            self._paste_clipboard_search()
        else:
            super().keyPressEvent(event)

    # ---- Feature: Hybrid text + image search ----

    def _set_anchor_image(self):
        """Open a file to use as the anchor (reference) image for hybrid search."""
        if self.clip_model is None:
            QMessageBox.warning(self, "Wait", "Model is still loading.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Anchor Image", "",
            "Images (*.jpg *.jpeg *.png *.webp *.bmp *.gif)"
        )
        if not path:
            return
        try:
            img = open_image(path)
            if img is None:
                raise ValueError("Could not open image")
            feats = self.clip_model.encode_image_batch([img])
            self._anchor_embed = feats[0]
            self._anchor_path = path
            name = os.path.basename(path)
            self.anchor_label.setText(name)
            self.anchor_label.setVisible(True)
            self.btn_clear_anchor.setVisible(True)
            self.hybrid_slider.setVisible(True)
            self.hybrid_val_label.setVisible(True)
            if self.hybrid_slider.value() == 0:
                self.hybrid_slider.setValue(50)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not encode anchor image: {e}")

    def _clear_anchor(self):
        """Remove the hybrid search anchor."""
        self._anchor_embed = None
        self._anchor_path = None
        self.anchor_label.setText("")
        self.anchor_label.setVisible(False)
        self.btn_clear_anchor.setVisible(False)
        self.hybrid_slider.setVisible(False)
        self.hybrid_val_label.setVisible(False)
        self.hybrid_slider.setValue(0)

    def _set_last_image_search(self, path):
        """Store the last image search path and reveal the Re-run controls."""
        self._last_image_search_path = path
        self._last_search_type = 'image'
        # Build and display a small thumbnail
        try:
            pil_img = open_image(path)
            if pil_img is not None:
                pil_img.thumbnail((64, 64))
                pixmap = pil_to_pixmap(pil_img)
                self.last_image_thumb.setPixmap(pixmap)
        except Exception:
            self.last_image_thumb.clear()
        self.last_image_thumb.setToolTip(os.path.basename(path))
        self.last_image_thumb.setVisible(True)
        self.btn_rerun_image.setVisible(True)
        self.btn_clear_image_search.setVisible(True)

    def on_rerun_image_click(self):
        """Re-run image search using the last loaded image."""
        if not self._last_image_search_path:
            return
        if not self.is_safe_to_act(action_name="image search"):
            return
        self.cancel_search(clear_ui=True)
        path = self._last_image_search_path
        gen = self.search_generation + 1
        self.search_thread = Thread(target=lambda: self._image_search(path, gen), daemon=True)
        self.search_thread.start()

    def _clear_image_search(self):
        """Clear the loaded image and hide Re-run controls."""
        self._last_image_search_path = None
        self.last_image_thumb.clear()
        self.last_image_thumb.setVisible(False)
        self.btn_rerun_image.setVisible(False)
        self.btn_clear_image_search.setVisible(False)

    # ---- Feature: Post-search sort ----

    def _resort_and_redisplay(self):
        """Re-sort all_search_results by the selected sort criterion and show first page."""
        if not self.all_search_results or self.is_searching:
            return
        idx = self.sort_combo.currentIndex()
        results = list(self.all_search_results)
        if idx == 0:  # Score ↓
            results.sort(key=lambda x: x[0], reverse=True)
        elif idx == 1:  # Date newest
            def _mtime(item):
                try:
                    return os.path.getmtime(item[1])
                except Exception:
                    return 0.0
            results.sort(key=_mtime, reverse=True)
        elif idx == 2:  # Date oldest
            def _mtime_asc(item):
                try:
                    return os.path.getmtime(item[1])
                except Exception:
                    return 0.0
            results.sort(key=_mtime_asc)
        elif idx == 3:  # Name A→Z
            results.sort(key=lambda x: os.path.basename(x[1]).lower())
        elif idx == 4:  # Name Z→A
            results.sort(key=lambda x: os.path.basename(x[1]).lower(), reverse=True)
        elif idx == 5:  # Size ↓
            def _size(item):
                try:
                    return os.path.getsize(item[1])
                except Exception:
                    return 0
            results.sort(key=_size, reverse=True)

        saved_total = self.total_found
        self.selected_images.clear()
        self.clear_results(keep_results=True)
        self.all_search_results = results
        self.total_found = saved_total
        page_size = max(10, self.top_n_slider.value() * 10)
        first_batch = results[:page_size]
        self.show_more_offset = len(first_batch)
        if not first_batch:
            return
        self.stop_search = False
        gen = self.search_generation
        t = Thread(target=self.load_thumbnails_worker, args=(first_batch, gen), daemon=True)
        self._thumbnail_worker_thread = t
        t.start()
        QTimer.singleShot(10, lambda: self.check_thumbnail_queue(gen))

    # ---- Feature: Auto-incremental index via file-system watcher ----

    def _on_auto_update_toggled(self, checked):
        """Enable or disable the file-system watcher for all selected folders."""
        if checked and self.folders:
            for f in self.folders:
                self._fs_watcher.addPath(f)
            safe_print(f"[WATCHER] Watching: {self.folders}")
        else:
            paths = self._fs_watcher.directories()
            if paths:
                self._fs_watcher.removePaths(paths)
            self._fs_debounce_timer.stop()
            safe_print("[WATCHER] Stopped")

    def _on_folder_changed(self, path):
        """Called by QFileSystemWatcher when the watched folder emits directoryChanged."""
        if not self.auto_update_cb.isChecked():
            return
        safe_print(f"[WATCHER] Change detected in: {path} — debouncing 5 s")
        self._fs_debounce_timer.start(5000)  # 5-second debounce

    def _trigger_auto_refresh(self):
        """Debounced slot — start a refresh index if idle."""
        if not self.folder or self.is_indexing or self.is_searching:
            safe_print("[WATCHER] Auto-refresh skipped (busy or no folder)")
            return
        if self.clip_model is None:
            safe_print("[WATCHER] Auto-refresh skipped (model not loaded)")
            return
        safe_print("[WATCHER] Auto-refresh triggered")
        self._safe_after(0, lambda: self.start_indexing(mode="refresh"))

    # ---- Feature: Failed / skipped files report ----

    def _show_failed_files_dialog(self):
        """Show a dialog listing files that failed to index in the last run."""
        combined = (
            [("image", p, r) for p, r in self._last_failed_images] +
            [("video", p, r) for p, r in self._last_failed_videos]
        )
        if not combined:
            QMessageBox.information(self, "No Failures", "No files were skipped in the last index run.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Skipped Files — {len(combined)} total")
        dlg.resize(680, 420)
        layout = QVBoxLayout(dlg)
        lbl = QLabel(
            f"<b>{len(combined)} file(s)</b> could not be indexed in the last run. "
            "These have also been logged to <i>photosearchpro_skipped_images.txt</i> / "
            "<i>photosearchpro_skipped_videos.txt</i> in your folder."
        )
        lbl.setWordWrap(True)
        layout.addWidget(lbl)
        lst = QListWidget()
        lst.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        for kind, path, reason in combined:
            icon = "🖼" if kind == "image" else "🎬"
            lst.addItem(f"{icon}  {os.path.basename(path)}  —  {reason}\n    {path}")
        layout.addWidget(lst)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        layout.addWidget(btn_close)
        dlg.exec()

    def set_hf_token(self):
        """Let the user enter their HuggingFace token and save it to the app config."""
        current = _load_app_settings().get("hf_token", os.environ.get("HF_TOKEN", ""))
        token, ok = QInputDialog.getText(
            self, "HuggingFace Token",
            "Paste your HuggingFace token (hf_...).\n"
            "Required for first-time download of SigLIP2 and DINOv3 models.\n"
            "Get a free read token at huggingface.co/settings/tokens\n\n"
            "Leave blank to clear the saved token.",
            QLineEdit.EchoMode.Password,
            current,
        )
        if not ok:
            return
        token = token.strip()
        settings = _load_app_settings()
        if token:
            settings["hf_token"] = token
            os.environ["HF_TOKEN"] = token
            try:
                import huggingface_hub as _hfh
                _hfh.login(token=token, add_to_git_credential=False)
            except Exception:
                pass
            QMessageBox.information(self, "Token Saved",
                "HuggingFace token saved.\n"
                "It will be applied automatically on every launch.")
        else:
            settings.pop("hf_token", None)
            os.environ.pop("HF_TOKEN", None)
            QMessageBox.information(self, "Token Cleared", "HuggingFace token removed.")
        _save_app_settings(settings)

    def open_model_selector(self):
        """Open the model-selection dialog.  Only allowed when not indexing."""
        if self.model_loading:
            QMessageBox.information(self, "Model Loading",
                                    "Please wait for the model to finish loading.")
            return
        if not self.is_safe_to_act(action_name="change model"):
            return
        dlg = ModelSelectorDialog(self, self.active_model_key)
        if dlg.exec() and dlg.selected_key != self.active_model_key:
            new_key = dlg.selected_key
            cfg = MODEL_REGISTRY[new_key]
            self.active_model_key = new_key
            _save_app_settings({**_load_app_settings(), "model_key": new_key})
            self.btn_model.setText(f"Model: {cfg['label']}")
            self._update_text_search_state()
            # Clear the existing index — different model → incompatible embeddings
            self.image_embeddings  = None
            self.image_paths       = []
            self.video_embeddings  = None
            self.video_paths       = []
            self.cache_file        = None
            self.video_cache_file  = None
            self._pending_image_batches = []
            self._pending_video_batches = []
            # Explicitly evict the old model from VRAM before loading the new one.
            # Simply setting clip_model=None is not enough — Python's GC may not
            # run before the new model's allocations begin, leaving both models in
            # VRAM simultaneously and causing an OOM error.
            old_model = self.clip_model
            self.clip_model = None
            if old_model is not None:
                try:
                    import torch, gc
                    # Move weights to CPU so CUDA allocator can reclaim the pages.
                    inner = getattr(old_model, 'model', None)
                    if inner is not None and hasattr(inner, 'cpu'):
                        inner.cpu()
                    # Also handle SigLIP2 / DINOv3 which store the model differently.
                    for attr in ('visual_model', 'text_model', '_model', 'dino_model'):
                        sub = getattr(old_model, attr, None)
                        if sub is not None:
                            sub_inner = getattr(sub, 'model', None)
                            if sub_inner is not None and hasattr(sub_inner, 'cpu'):
                                sub_inner.cpu()
                            elif hasattr(sub, 'cpu'):
                                sub.cpu()
                    del old_model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception:
                    del old_model
            self.update_status(f"Loading {cfg['label']}...", "orange")
            QMessageBox.information(
                self, "Model Changed",
                f"Switched to: {cfg['label']}\n\n"
                f"{cfg['subtitle']}\n\n"
                "The model is loading now.  Once ready, select your folder and re-index "
                "(each model needs its own index file)."
            )
            Thread(target=self.load_model, daemon=True).start()

    # ---- Action handlers ----

    def is_safe_to_act(self, action_callback=None, action_name="action"):
        """Returns True if no indexing is happening."""
        if self.is_indexing or self.is_stopping:
            QMessageBox.information(
                self,
                "Indexing in Progress",
                "Indexing is currently running.\n\n"
                "Please wait for it to finish, or press STOP INDEX to save your partial cache and stop.\n\n"
                "(Your partial index is safe - stopping saves everything indexed so far.)"
            )
            return False
        return True

    def stop_all_processes(self):
        self.cancel_search(clear_ui=True)
        if self.is_indexing:
            self.stop_indexing_process()

    def on_select_folder(self):
        if not self.is_safe_to_act(action_callback=self.select_folder, action_name="select folder"):
            return
        self.cancel_search(clear_ui=True)
        self.select_folder()

    def on_force_reindex(self):
        if not self.is_safe_to_act(action_callback=self.force_reindex, action_name="refresh index"):
            return
        self.cancel_search(clear_ui=True)
        self.force_reindex()

    def on_clear_click(self):
        self.cancel_search(clear_ui=True)
        self.clear_results()
        self.update_status("Results cleared", "green")

    def on_copy_click(self):
        self.export_selected()

    def on_move_click(self):
        self.move_selected()

    def on_delete_click(self):
        self.delete_selected()

    def on_search_click(self):
        self._last_search_type = 'text'
        self.hide_info_bar()
        self.cancel_search(clear_ui=True)
        self.do_search()

    def on_image_click(self):
        if not self.is_safe_to_act(action_name="image search"):
            return
        self.cancel_search(clear_ui=True)
        self.image_search()

    def _on_drop_image(self, path):
        """Handle image file dropped onto the scroll area."""
        if self.clip_model is None:
            QMessageBox.warning(self, "Wait", "Model is still loading, please wait.")
            return
        if not self.is_safe_to_act(action_name="image search"):
            return
        if not self.folder:
            QMessageBox.warning(self, "No Folder", "Please select a folder first before searching by image.")
            return
        if self.image_embeddings is None and self.video_embeddings is None:
            QMessageBox.warning(self, "Not Indexed", "Please index a folder first before searching by image.")
            return
        valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif',
                      '.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.raf', '.pef', '.sr2')
        if not path.lower().endswith(valid_exts):
            QMessageBox.warning(self, "Unsupported File",
                "Only image files can be used for image search.\nDrop a JPG, PNG, WEBP or RAW file.")
            return
        if not os.path.isfile(path):
            return
        self._set_last_image_search(path)
        self.cancel_search(clear_ui=True)
        gen = self.search_generation + 1
        self.search_thread = Thread(target=lambda: self._image_search(path, gen), daemon=True)
        self.search_thread.start()

    def force_quit(self):
        if QMessageBox.question(self, "Force Quit", "Force quit application?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            os._exit(0)

    def _show_close_dialog(self):
        """Show dialog when user closes window while indexing is running."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Indexing in Progress")
        dlg.setFixedSize(440, 170)
        layout = QVBoxLayout(dlg)

        lbl1 = QLabel("Indexing is currently running.")
        lbl1.setStyleSheet(f"font-size: 11pt; font-weight: bold; color: {ORANGE};")
        layout.addWidget(lbl1)
        layout.addWidget(QLabel("What would you like to do?"))

        btn_row = QHBoxLayout()

        def stop_and_close():
            dlg.accept()
            self.stop_indexing = True
            self.is_stopping = True
            self._safe_after(0, lambda: self.update_status("Stopping & saving before exit...", ORANGE))
            safe_print("\n[CLOSE] Stop & save requested.")
            def _wait():
                if self.is_indexing:
                    self._safe_after(200, _wait)
                else:
                    os._exit(0)
            self._safe_after(200, _wait)

        b1 = QPushButton("Stop & Save")
        b1.setProperty("class", "accent")
        b1.clicked.connect(stop_and_close)
        b2 = QPushButton("Quit Anyway")
        b2.setProperty("class", "danger")
        b2.clicked.connect(lambda: os._exit(0))
        b3 = QPushButton("Cancel")
        b3.clicked.connect(dlg.reject)

        btn_row.addWidget(b1)
        btn_row.addWidget(b2)
        btn_row.addWidget(b3)
        layout.addLayout(btn_row)
        dlg.exec()

    def stop_indexing_process(self):
        if self.is_indexing and not self.is_stopping:
            self.stop_indexing = True
            self.is_stopping = True
            self.update_status("Stopping... Please wait for file save.", DANGER)
            safe_print("\n[STOP] Stop signal sent. Waiting for batch to finish...")
        elif self.is_stopping:
            if self.pending_action:
                safe_print("[STOP] Clearing pending action...")
                self.pending_action = None
                self.update_status("Stopping... (Pending action cancelled)", DANGER)
                self.btn_stop.setText("STOP INDEX")

    def cancel_search(self, clear_ui=False):
        """Cancel ongoing search and optionally clear UI"""
        self.search_generation += 1
        with self.thumbnail_queue.mutex:
            self.thumbnail_queue.queue.clear()
        
        self.stop_search = True
        self.total_found = 0
        
        if clear_ui:
            self.clear_results()  # This will free thumbnail RAM
        
        if not self.is_indexing:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress_label.setText("")
        
        if self.is_searching:
            self.update_status("Search Cancelled", "orange")
        self.is_searching = False

    def _show_folder_picker_dialog(self):
        """Show a dialog that lets the user manage a list of folders to index.

        Left panel  – saved indexes (load/delete).
        Right panel – current folder set editor with drag-and-drop support.

        Returns a non-empty list of folder paths, or an empty list if cancelled.
        Pre-populates the folder list with any currently selected folders.
        """
        from PyQt6.QtWidgets import QSplitter, QListWidget
        from datetime import datetime as _dt

        dlg = QDialog(self)
        dlg.setWindowTitle("Select Folders to Index")
        dlg.resize(960, 480)
        outer = QVBoxLayout(dlg)
        outer.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, stretch=1)

        # ── Left panel: saved indexes ──────────────────────────────────────
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 4, 0)

        saved_lbl = QLabel("Saved Indexes")
        saved_lbl.setStyleSheet("font-weight: bold; font-size: 9pt;")
        left_layout.addWidget(saved_lbl)

        saved_list = QListWidget()
        saved_list.setToolTip("Double-click or click Load to open a saved index")
        left_layout.addWidget(saved_list, stretch=1)

        saved_btn_row = QHBoxLayout()
        load_btn  = QPushButton("Load")
        del_btn   = QPushButton("Delete")
        load_btn.setToolTip("Load the selected saved index into the editor")
        del_btn.setToolTip("Remove this entry from the saved list")
        saved_btn_row.addWidget(load_btn)
        saved_btn_row.addWidget(del_btn)
        saved_btn_row.addStretch()
        left_layout.addLayout(saved_btn_row)

        splitter.addWidget(left_widget)

        # ── Right panel: folder editor ─────────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 0, 0, 0)

        right_layout.addWidget(QLabel(
            "Add one or more folders — all are indexed together.\n"
            "The first folder stores the cache and settings files.\n"
            "Drag folders from your file manager and drop them into the list."
        ))

        list_widget = FolderDropListWidget()
        right_layout.addWidget(list_widget, stretch=1)

        edit_btn_row = QHBoxLayout()
        add_btn       = QPushButton("Add Folder…")
        remove_btn    = QPushButton("Remove Selected")
        select_all_btn = QPushButton("Select All")
        select_all_btn.setToolTip("Select all folders in the list")
        select_all_btn.clicked.connect(list_widget.selectAll)
        edit_btn_row.addWidget(add_btn)
        edit_btn_row.addWidget(remove_btn)
        edit_btn_row.addWidget(select_all_btn)
        edit_btn_row.addStretch()
        right_layout.addLayout(edit_btn_row)

        splitter.addWidget(right_widget)
        splitter.setSizes([300, 660])

        # ── Populate saved list ────────────────────────────────────────────
        saved_entries = _load_saved_indexes()

        def _rebuild_saved_list():
            saved_list.clear()
            for entry in saved_entries:
                folders = entry.get("folders", [])
                name    = entry.get("name", folders[0] if folders else "?")
                when    = entry.get("last_used", "")[:10]
                extra   = f"  (+{len(folders)-1} more)" if len(folders) > 1 else ""
                item_txt = f"{name}{extra}"
                if when:
                    item_txt += f"   [{when}]"
                item = QListWidgetItem(item_txt)
                item.setToolTip("\n".join(folders))
                saved_list.addItem(item)

        _rebuild_saved_list()

        # ── Pre-populate folder editor ─────────────────────────────────────
        for f in self.folders:
            list_widget.addItem(f)

        # ── Helpers ────────────────────────────────────────────────────────
        def _current_folders():
            return [list_widget.item(i).text() for i in range(list_widget.count())]

        def add_folder():
            start_dir = list_widget.item(list_widget.count()-1).text() \
                        if list_widget.count() > 0 else ""
            folder = QFileDialog.getExistingDirectory(dlg, "Add Folder", start_dir)
            if folder and folder not in _current_folders():
                list_widget.addItem(folder)

        def remove_folder():
            for item in list_widget.selectedItems():
                list_widget.takeItem(list_widget.row(item))

        def load_saved():
            row = saved_list.currentRow()
            if row < 0:
                return
            folders = saved_entries[row].get("folders", [])
            list_widget.clear()
            for f in folders:
                list_widget.addItem(f)

        def delete_saved():
            row = saved_list.currentRow()
            if row < 0:
                return
            entry = saved_entries[row]
            name = entry.get("name", "this entry")
            if QMessageBox.question(
                dlg, "Remove Saved Index",
                f'Remove "{name}" from the saved list?\n(Your actual files are not affected.)',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) != QMessageBox.StandardButton.Yes:
                return
            saved_entries.pop(row)
            _save_indexes_list(saved_entries)
            _rebuild_saved_list()

        add_btn.clicked.connect(add_folder)
        remove_btn.clicked.connect(remove_folder)
        load_btn.clicked.connect(load_saved)
        del_btn.clicked.connect(delete_saved)
        saved_list.itemDoubleClicked.connect(lambda _: load_saved())

        # ── Bottom OK / Cancel ─────────────────────────────────────────────
        ok_btn     = QPushButton("OK")
        ok_btn.setProperty("class", "accent")
        cancel_btn = QPushButton("Cancel")

        def on_ok():
            if list_widget.count() == 0:
                QMessageBox.warning(dlg, "No Folder", "Please add at least one folder.")
                return
            dlg.accept()

        ok_btn.clicked.connect(on_ok)
        cancel_btn.clicked.connect(dlg.reject)

        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        bottom_row.addWidget(ok_btn)
        bottom_row.addWidget(cancel_btn)
        outer.addLayout(bottom_row)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return []

        return _current_folders()

    def _persist_index_entry(self, folders):
        """Save (or refresh) a folder-set entry in the saved indexes list."""
        from datetime import datetime as _dt
        entries = _load_saved_indexes()
        # Remove any existing entry with the same folder set
        entries = [e for e in entries if e.get("folders") != folders]
        name = os.path.basename(folders[0]) if folders else "Index"
        new_entry = {
            "name": name,
            "folders": folders,
            "last_used": _dt.now().strftime("%Y-%m-%d %H:%M"),
        }
        entries.insert(0, new_entry)
        # Keep at most 20 entries
        _save_indexes_list(entries[:20])

    def select_folder(self):
        if self.clip_model is None:
            QMessageBox.warning(self, "Wait", "Model is still loading...")
            return

        # Show multi-folder picker — don't wipe anything until user confirms
        new_folders = self._show_folder_picker_dialog()
        if not new_folders:
            self.update_status("No folder selected", "orange")
            return
        folder = new_folders[0]  # primary folder

        # User confirmed — now safe to wipe old data
        self.image_paths = []
        self.image_embeddings = None
        self.folder = None
        self.folders = []
        self.cache_file = None
        self.video_paths = []
        self.video_embeddings = None
        self.video_cache_file = None
        self.clear_results()
        self.update_stats()

        self.folder = folder
        self.folders = new_folders
        self.excluded_folders = set()
        self.load_exclusions()
        self.load_search_history()
        self._load_face_data()
        safe_print(f"\n{'='*60}\n[FOLDERS] {new_folders}")

        # Persist this folder set in the saved indexes list (most-recent first,
        # deduplicated by folder list, capped at 20 entries).
        self._persist_index_entry(new_folders)

        # Update file-system watcher to track all selected folders
        existing_watched = self._fs_watcher.directories()
        if existing_watched:
            self._fs_watcher.removePaths(existing_watched)
        if self.auto_update_cb.isChecked():
            for f in new_folders:
                self._fs_watcher.addPath(f)
            safe_print(f"[WATCHER] Now watching: {new_folders}")
        
        cache_files = self.get_cache_filename()
        found_cache = None
        for cache_name in cache_files:
            cache_path = os.path.join(folder, cache_name)
            if os.path.exists(cache_path):
                found_cache = cache_path
                safe_print(f"[CACHE] Found existing: {cache_name}")
                break
        
        # Check for video cache first so we know before showing any popup
        video_cache_name = self.get_video_cache_filename()
        video_cache_path = os.path.join(folder, video_cache_name)
        found_video_cache = os.path.exists(video_cache_path)

        if found_cache:
            self.cache_file = found_cache
            self.load_cache_data(found_cache)
            self.query_entry.clear()
        else:
            safe_print("[CACHE] Image cache not found")
            if not found_video_cache:
                # Check if a *different* model's cache exists — helps the user understand why
                active_label = MODEL_REGISTRY[self.active_model_key]["label"]
                other_labels = [
                    cfg["label"]
                    for key, cfg in MODEL_REGISTRY.items()
                    if key != self.active_model_key
                    and os.path.exists(os.path.join(folder, f".clip_cache_{cfg['cache_key']}.pkl"))
                ]
                if other_labels:
                    detail = (
                        f"No index found for the active model ({active_label}).\n\n"
                        f"Found an existing index for: {', '.join(other_labels)}\n"
                        f"These are incompatible with {active_label} — each model stores\n"
                        f"embeddings in a different format.\n\n"
                        f"Re-index this folder now to create a {active_label} index?"
                    )
                else:
                    detail = f"No index found for this folder.\n\nIndex images now with {active_label}?"
                if QMessageBox.question(self, "Index Folder?", detail,
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                    self.cache_file = os.path.join(folder, cache_files[0])
                    self.start_indexing(mode="full")
                else:
                    self.update_status("Folder loaded (Not indexed)", "orange")
            # If video cache exists, skip the popup — user has something useful already

        if found_video_cache:
            self.load_video_cache_data(video_cache_path)
            safe_print(f"[VCACHE] Auto-loaded: {video_cache_name}")
            self.query_entry.clear()
            # Update status to reflect what's actually loaded
            has_images = self.image_embeddings is not None and len(self.image_paths) > 0
            has_videos = self.video_embeddings is not None and len(self.video_paths) > 0
            if has_images and has_videos:
                n_imgs = len(self.image_paths)
                n_vids = len(set(vp for vp, _ in self.video_paths))
                self.update_status(f"Loaded {n_imgs:,} images + {n_vids:,} videos", "green")
            elif has_videos:
                n_vids = len(set(vp for vp, _ in self.video_paths))
                self.update_status(f"Loaded video cache: {n_vids:,} videos", "green")
            elif has_images:
                self.update_status(f"Loaded {len(self.image_paths):,} images", "green")

    def load_cache_data(self, cache_path):
        try:
            safe_print(f"[CACHE] Loading: {cache_path}")
            self.update_status("Loading cache from disk...", "orange")

            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                # Format: (paths_list, embeddings)
                self.image_paths, self.image_embeddings = data

            # Normalize separators
            self.image_paths = [p.replace('\\', '/') for p in self.image_paths]

            # Backward compat: old caches stored RELATIVE paths; convert to absolute
            if self.image_paths and not os.path.isabs(self.image_paths[0]):
                base = os.path.dirname(cache_path)
                self.image_paths = [
                    os.path.join(base, p).replace('\\', '/') for p in self.image_paths
                ]
                safe_print("[CACHE] Converted relative paths → absolute (backward compat)")

            if hasattr(self.image_embeddings, 'cpu'):
                self.image_embeddings = self.image_embeddings.cpu().numpy()

            self.cache_file = cache_path
            self.folder = os.path.dirname(cache_path)
            if self.folder not in self.folders:
                self.folders = [self.folder] + [f for f in self.folders if f != self.folder]

            self.load_exclusions()
            self.update_stats()
            n_imgs = len(self.image_paths)
            n_vids = len(set(vp for vp, _ in self.video_paths)) if self.video_paths else 0
            if n_vids > 0:
                self.update_status(f"Loaded {n_imgs:,} images, {n_vids:,} videos", "green")
            else:
                self.update_status(f"Loaded {n_imgs:,} images", "green")
            safe_print(f"[CACHE] Success. {n_imgs:,} images (absolute paths).")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Load failed: {e}")
            self.update_status("Cache load failed", "red")

    def load_video_cache_data(self, cache_path):
        try:
            safe_print(f"[VCACHE] Loading: {cache_path}")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                self.video_paths, self.video_embeddings = data

            # Normalize separators
            self.video_paths = [(vp.replace('\\', '/'), ts) for vp, ts in self.video_paths]

            # Backward compat: old caches stored RELATIVE paths; convert to absolute
            if self.video_paths and not os.path.isabs(self.video_paths[0][0]):
                base = os.path.dirname(cache_path)
                self.video_paths = [
                    (os.path.join(base, vp).replace('\\', '/'), ts)
                    for vp, ts in self.video_paths
                ]
                safe_print("[VCACHE] Converted relative paths → absolute (backward compat)")

            if hasattr(self.video_embeddings, 'cpu'):
                self.video_embeddings = self.video_embeddings.cpu().numpy()

            self.video_cache_file = cache_path
            self.update_stats()
            n_videos = len(set(vp for vp, _ in self.video_paths))
            n_frames = len(self.video_paths)
            safe_print(f"[VCACHE] Loaded {n_frames:,} frames from {n_videos:,} videos.")
        except Exception as e:
            safe_print(f"[VCACHE] Load error: {e}")
            self.video_paths = []
            self.video_embeddings = None

    def start_indexing(self, mode="full"):
        # Guard against double-start
        if self.is_indexing:
            safe_print(f"[INDEX] start_indexing called while already indexing (mode={mode}), ignoring")
            return

        self.stop_indexing = False
        self.is_stopping = False
        self.pending_action = None
        self.update_status("Indexing...", "orange")
        self.btn_stop.setText("STOP INDEX")
        
        if mode == "full":
            self.index_thread = Thread(target=self.index_all_images, daemon=True)
        elif mode == "refresh":
            self.index_thread = Thread(target=self.refresh_index, daemon=True)
        elif mode == "video_full":
            self.index_thread = Thread(target=self.index_all_videos, daemon=True)
        elif mode == "video_refresh":
            self.index_thread = Thread(target=self.refresh_video_index, daemon=True)
        else:
            safe_print(f"[INDEX] Unknown mode: {mode}")
            return

        self.index_thread.start()


    def refresh_index(self):
        if not self.folder or self.clip_model is None: return
        self.is_indexing = True

        if not self.cache_file:
            cache_files = self.get_cache_filename()
            self.cache_file = os.path.join(self.folder, cache_files[0])
        
        if self.clip_model and not getattr(self.clip_model, 'use_onnx_visual', False):
            if not getattr(self.clip_model, 'onnx_disabled', False):
                try:
                    self.clip_model._create_onnx_session()
                except:
                    pass
        
        safe_print("\n[SCAN] Scanning folder(s) for changes...")
        self._safe_after(0, lambda: self.update_status("Scanning folder...", "orange"))

        current_disk_files = set()  # absolute paths currently on disk
        new_files_to_add = []

        existing_paths_set = set(self.image_paths)  # already absolute

        for scan_folder in (self.folders if self.folders else [self.folder]):
            if self.stop_indexing:
                break
            for root, _, files in os.walk(scan_folder):
                if self.stop_indexing:
                    break
                for f in files:
                    if f.lower().endswith(IMAGE_EXTS):
                        abs_path = os.path.join(root, f)
                        if self._is_excluded(abs_path):
                            continue
                        current_disk_files.add(abs_path)
                        if abs_path not in existing_paths_set:
                            new_files_to_add.append(abs_path)
        
        if self.stop_indexing:
            self._handle_stop()
            return

        safe_print("[SCAN] Pruning deleted/renamed files...")
        valid_indices = []
        pruned_paths = []

        for i, abs_path in enumerate(self.image_paths):
            if abs_path in current_disk_files:
                valid_indices.append(i)
                pruned_paths.append(abs_path)
        
        removed_count = len(self.image_paths) - len(valid_indices)
        
        if removed_count > 0:
            if self.image_embeddings is not None:
                self.image_embeddings = self.image_embeddings[valid_indices]
            self.image_paths = pruned_paths
            safe_print(f"[SCAN] Pruned {removed_count} stale entries.")
        
        if new_files_to_add:
            safe_print(f"[SCAN] Found {len(new_files_to_add)} new files.")
            self._process_batch(new_files_to_add, is_update=True)
        else:
            if removed_count > 0:
                self._save_cache(allow_shrink=True)
            
            self.is_indexing = False
            self.is_stopping = False
            safe_print("[SCAN] Index is up to date.")

            if getattr(self, '_pending_video_refresh', False):
                self._pending_video_refresh = False
                if not self.video_cache_file:
                    self.video_cache_file = os.path.join(self.folder, self.get_video_cache_filename())
                self._safe_after(200, lambda: self.start_indexing(mode="video_refresh"))
            else:
                self._safe_after(0, lambda: self.update_status("Up to date", "green"))
            self._safe_after(0, self.update_stats)

    def index_all_images(self):
        if not self.folder or self.clip_model is None: return
        self.is_indexing = True
        
        if self.clip_model and not getattr(self.clip_model, 'use_onnx_visual', False):
            if not getattr(self.clip_model, 'onnx_disabled', False):
                try:
                    self.clip_model._create_onnx_session()
                except:
                    pass
        
        old_paths = self.image_paths[:]
        old_embeddings = self.image_embeddings.copy() if self.image_embeddings is not None else None
        self.image_paths = []
        self.image_embeddings = None
        
        all_images = []
        for scan_folder in (self.folders if self.folders else [self.folder]):
            if self.stop_indexing:
                break
            for root, _, files in os.walk(scan_folder):
                if self.stop_indexing:
                    break
                for f in files:
                    if f.lower().endswith(IMAGE_EXTS):
                        abs_path = os.path.join(root, f)
                        if not self._is_excluded(abs_path):
                            all_images.append(abs_path)
        
        if self.stop_indexing:
            self.image_paths = old_paths
            self.image_embeddings = old_embeddings
            self._handle_stop()
            return

        if not all_images:
            self.image_paths = old_paths
            self.image_embeddings = old_embeddings
            self.is_indexing = False
            self._safe_after(0, lambda: self.update_status("No images found", "orange"))
            return

        safe_print(f"[INDEX] Found {len(all_images)} images.")
        self._process_batch(all_images, is_update=False)

    def _process_batch(self, file_list, is_update=False):
        """Process images and store RELATIVE paths.

        Pipeline:
        - 8 worker threads each do: open_image() + CLIP preprocess() -> tensor
          (preprocessing is stateless transforms — fully thread-safe)
        - 2 batches are always prefetched so the GPU never waits for CPU
        - GPU receives pre-built tensors via encode_tensor_batch() — no serial
          preprocess loop blocking before encode
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
        try:
            total = len(file_list)
            processed = 0
            existing_paths_set = set(self.image_paths)
            self._pending_image_batches = []
            import torch

            use_onnx = getattr(self.clip_model, 'use_onnx_visual', False) and \
                       getattr(self.clip_model, 'visual_session', None) is not None

            if use_onnx:
                def load_worker(abs_path):
                    try:
                        torch.set_num_threads(1)
                        img = open_image(abs_path)
                        return (abs_path, img, None)
                    except Exception:
                        return (abs_path, None, None)
            else:
                clip_preprocess = self.clip_model.preprocess
                def load_worker(abs_path):
                    try:
                        torch.set_num_threads(1)
                        img = open_image(abs_path)
                        if img is None:
                            return (abs_path, None, None)
                        tensor = clip_preprocess(img)
                        return (abs_path, img, tensor)
                    except Exception:
                        return (abs_path, None, None)

            batches = [file_list[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
            executor = ThreadPoolExecutor(max_workers=4)

            def submit_batch(idx):
                if idx < len(batches):
                    return [executor.submit(load_worker, p) for p in batches[idx]]
                return []

            prefetch_queue = [submit_batch(0), submit_batch(1)]

            STREAM_CHUNK = 64

            for batch_idx, batch_paths in enumerate(batches):
                if self.stop_indexing:
                    safe_print("\n[INDEX] Stopping batch loop.")
                    break

                current_futures = prefetch_queue.pop(0)
                prefetch_queue.append(submit_batch(batch_idx + 2))

                buf_tensors   = []
                buf_pils      = []
                buf_paths     = []

                def _flush_buf():
                    if not buf_paths or self.stop_indexing:
                        buf_tensors.clear(); buf_pils.clear(); buf_paths.clear()
                        return
                    try:
                        if buf_tensors and not use_onnx:
                            feats = self.clip_model.encode_tensor_batch(buf_tensors)
                        else:
                            feats = self.clip_model.encode_image_batch(buf_pils)
                        if feats is not None and feats.size > 0:
                            nf, np_ = [], []
                            for i2, ap in enumerate(buf_paths):
                                # Store absolute path directly
                                if ap not in existing_paths_set:
                                    np_.append(ap); nf.append(feats[i2])
                                    existing_paths_set.add(ap)
                            if np_:
                                self.image_paths.extend(np_)
                                self._pending_image_batches.append(np.array(nf))
                                _proc[0] += len(np_)
                    except Exception as enc_e:
                        safe_print(f"[ERROR] Stream encode chunk: {enc_e}")
                    buf_tensors.clear(); buf_pils.clear(); buf_paths.clear()

                _proc = [0]

                for fut in (_as_completed(current_futures) if current_futures else []):
                    if self.stop_indexing:
                        break
                    try:
                        abs_path, img, tensor = fut.result()
                    except Exception:
                        continue
                    if img is None:
                        self._failed_images.append((abs_path, "Failed to open or decode"))
                        continue
                    buf_paths.append(abs_path)
                    if tensor is not None:
                        buf_tensors.append(tensor)
                    else:
                        buf_pils.append(img)
                    if len(buf_paths) >= STREAM_CHUNK:
                        _flush_buf()

                if buf_paths and not self.stop_indexing:
                    _flush_buf()

                processed += _proc[0]

                if self.stop_indexing:
                    break

                if batch_idx % 30 == 0 and batch_idx > 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()

                if processed > 0 and processed % 5000 < BATCH_SIZE:
                    with self._cache_lock:
                        self._flush_pending_batches()
                        self._save_cache()
                    safe_print(f"[INDEX] Auto-saved at {processed:,} images")

                pct = (processed / total) * 100 if total > 0 else 0
                msg = f"{'Updating' if is_update else 'Indexing'}: {processed:,}/{total:,}"
                self._safe_after(0, lambda v=pct, m=msg: self.update_progress(v, m))
                safe_print(f"\r[INDEX] {msg}", end='')

            executor.shutdown(wait=False)
            safe_print("")

        finally:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()
            except Exception as cleanup_err:
                safe_print(f"[INDEX] VRAM cleanup warning (non-fatal): {cleanup_err}")

        with self._cache_lock:
            self._flush_pending_batches()
            self._save_cache()
        self._handle_stop()

    def _process_video_batch(self, file_list, is_update=False):
        """Extract frames from videos in parallel, encode with CLIP, store (rel_path, timestamp) tuples."""
        try:
            import cv2
        except ImportError:
            safe_print("[VINDEX ERROR] OpenCV not installed. Run: pip install opencv-python")
            self._safe_after(0, lambda: QMessageBox.critical(
                self,
                "Missing Dependency",
                "OpenCV is required for video indexing.\n\nInstall it with:\n  pip install opencv-python"
            ))
            self._handle_video_stop()
            return

        try:
            import torch
        except ImportError:
            torch = None

        try:
            from concurrent.futures import ThreadPoolExecutor
            import threading as _threading

            import os as _os
            _os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
            try:
                import cv2 as _cv2pre
                _cv2pre.setLogLevel(0)
            except Exception:
                pass

            total = len(file_list)
            existing_set = set(self.video_paths) if is_update else set()
            self._pending_video_batches = []
            CHUNK_SIZE = max(VIDEO_BATCH_SIZE * 2, 32)
            VIDEO_PARALLEL = 1

            def encode_chunk(frames, timestamps, _rel_path, _existing):
                if not frames or self.stop_indexing:
                    return
                try:
                    features = self.clip_model.encode_image_batch(frames)
                    if features is None or features.size == 0:
                        return
                    new_tuples = []
                    new_features = []
                    for j, ts in enumerate(timestamps):
                        tup = (_rel_path, ts)
                        if tup not in _existing:
                            new_tuples.append(tup)
                            new_features.append(features[j])
                            _existing.add(tup)
                    if new_tuples:
                        self.video_paths.extend(new_tuples)
                        self._pending_video_batches.append(np.array(new_features))
                except Exception as enc_err:
                    safe_print(f"[VINDEX ERROR] Encoding failed: {enc_err}")
                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()

            def extract_frames(abs_video_path):
                rel_path = abs_video_path  # Store absolute path in video_paths tuples
                safe_print(f"[VINDEX] Analyzing: {os.path.basename(abs_video_path)}")
                frames = []
                cap = None
                _devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
                _old_stderr_fd = _os.dup(2)
                _os.dup2(_devnull_fd, 2)
                try:
                    cap = cv2.VideoCapture(abs_video_path)
                    if not cap.isOpened():
                        safe_print(f"[VINDEX] Cannot open: {abs_video_path}")
                        return (rel_path, frames)

                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

                    if fps <= 0:
                        fps = 25.0
                    if total_frames_count <= 0:
                        duration_seconds = 60.0
                    else:
                        duration_seconds = total_frames_count / fps

                    if duration_seconds <= 0:
                        cap.release()
                        return (rel_path, frames)

                    if duration_seconds < VIDEO_FRAME_INTERVAL:
                        frames_to_sample = {duration_seconds / 2.0}
                    else:
                        interval = max(VIDEO_FRAME_INTERVAL, duration_seconds / MAX_FRAMES_PER_VIDEO)
                        t = interval
                        frames_to_sample = set()
                        while t < duration_seconds:
                            frames_to_sample.add(round(t, 3))
                            t += interval
                        if not frames_to_sample:
                            frames_to_sample = {duration_seconds / 2.0}

                    best_frame = None
                    best_brightness = -1.0
                    all_skipped = True

                    for target_t in sorted(frames_to_sample):
                        if self.stop_indexing:
                            break
                        if is_update and (rel_path, target_t) in existing_set:
                            continue
                        cap.set(cv2.CAP_PROP_POS_MSEC, max(0, (target_t - 2.0)) * 1000.0)
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            continue
                        try:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        except Exception:
                            del frame
                            continue
                        mean_brightness = float(frame_rgb[::8, ::8].mean())
                        if mean_brightness > best_brightness:
                            best_brightness = mean_brightness
                            best_frame = (Image.fromarray(frame_rgb.copy()), target_t)
                        if mean_brightness >= 10:
                            all_skipped = False
                            frames.append((Image.fromarray(frame_rgb), target_t))
                        del frame, frame_rgb

                    cap.release()

                    if all_skipped and best_frame is not None:
                        return (rel_path, [best_frame])

                    return (rel_path, frames)

                except (MemoryError, Exception) as video_err:
                    safe_print(f"[VINDEX] Error extracting frames from {os.path.basename(abs_video_path)}: {video_err}")
                    try:
                        if cap:
                            cap.release()
                    except Exception:
                        pass
                    return (rel_path, [])
                finally:
                    _os.dup2(_old_stderr_fd, 2)
                    _os.close(_old_stderr_fd)
                    _os.close(_devnull_fd)

            SUBMIT_BATCH = VIDEO_PARALLEL * 2
            file_idx = 0

            with ThreadPoolExecutor(max_workers=VIDEO_PARALLEL) as executor:
                for batch_start in range(0, total, SUBMIT_BATCH):
                    if self.stop_indexing:
                        safe_print("\n[VINDEX] Stopping video batch loop.")
                        break
                    batch_paths = file_list[batch_start:batch_start + SUBMIT_BATCH]
                    batch_futures = [executor.submit(extract_frames, p) for p in batch_paths]

                    for future in batch_futures:
                        if self.stop_indexing:
                            safe_print("\n[VINDEX] Stopping video batch loop.")
                            break

                        try:
                            rel_video_path, frame_list = future.result()
                        except Exception as fe:
                            safe_print(f"[VINDEX] Future error for file {file_idx}: {fe}")
                            file_idx += 1
                            continue

                        if not frame_list:
                            abs_video_path = file_list[file_idx] if file_idx < len(file_list) else rel_video_path
                            self._failed_videos.append((abs_video_path, "No frames extracted"))

                        chunk_frames = []
                        chunk_timestamps = []
                        for pil_img, ts in frame_list:
                            if self.stop_indexing:
                                break
                            chunk_frames.append(pil_img)
                            chunk_timestamps.append(ts)
                            if len(chunk_frames) >= CHUNK_SIZE:
                                encode_chunk(chunk_frames, chunk_timestamps, rel_video_path, existing_set)
                                chunk_frames = []
                                chunk_timestamps = []
                        if chunk_frames and not self.stop_indexing:
                            encode_chunk(chunk_frames, chunk_timestamps, rel_video_path, existing_set)

                        del frame_list

                        if file_idx % 5 == 0:
                            if torch and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            elif torch and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                                torch.mps.empty_cache()

                        if file_idx % 10 == 0:
                            gc.collect()

                        file_idx += 1
                        pct = (file_idx / total) * 100
                        n_frames = len(self.video_paths)
                        msg = f"{'Updating' if is_update else 'Indexing'} Videos: {file_idx:,}/{total:,} files ({n_frames:,} frames)"
                        self._safe_after(0, lambda v=pct, m=msg: self.update_progress(v, m))
                        safe_print(f"\r[VINDEX] {msg}", end='')

                        if file_idx > 0 and file_idx % 20 == 0:
                            with self._cache_lock:
                                self._flush_pending_batches()
                                self._save_video_cache()
                            safe_print(f"[VINDEX] Auto-saved at {file_idx:,} videos")

            safe_print("")

        finally:
            try:
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                elif torch and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()
            except Exception as cleanup_err:
                safe_print(f"[VINDEX] VRAM cleanup warning (non-fatal): {cleanup_err}")

        with self._cache_lock:
            self._flush_pending_batches()
            self._save_video_cache()
        self._handle_video_stop()

    def index_all_videos(self):
        if not self.folder or self.clip_model is None:
            return
        self.is_indexing = True

        if self.clip_model and not getattr(self.clip_model, 'use_onnx_visual', False):
            if not getattr(self.clip_model, 'onnx_disabled', False):
                try:
                    self.clip_model._create_onnx_session()
                except:
                    pass

        old_video_paths = self.video_paths[:]
        old_video_embeddings = self.video_embeddings.copy() if self.video_embeddings is not None else None
        self.video_paths = []
        self.video_embeddings = None

        all_videos = []
        for scan_folder in (self.folders if self.folders else [self.folder]):
            if self.stop_indexing:
                break
            for root_dir, _, files in os.walk(scan_folder):
                if self.stop_indexing:
                    break
                for f in files:
                    if f.lower().endswith(VIDEO_EXTS):
                        abs_path = os.path.join(root_dir, f)
                        if not self._is_excluded(abs_path):
                            all_videos.append(abs_path)

        if self.stop_indexing:
            self.video_paths = old_video_paths
            self.video_embeddings = old_video_embeddings
            self._handle_video_stop()
            return

        if not all_videos:
            self.video_paths = old_video_paths
            self.video_embeddings = old_video_embeddings
            self.is_indexing = False
            self._safe_after(0, lambda: self.update_status("No videos found", "orange"))
            return

        safe_print(f"[VINDEX] Found {len(all_videos)} video files.")
        self._process_video_batch(all_videos, is_update=False)

    def refresh_video_index(self):
        if not self.folder or self.clip_model is None:
            return
        self.is_indexing = True

        if self.clip_model and not getattr(self.clip_model, 'use_onnx_visual', False):
            if not getattr(self.clip_model, 'onnx_disabled', False):
                try:
                    self.clip_model._create_onnx_session()
                except:
                    pass

        safe_print("\n[VSCAN] Scanning folder(s) for video changes...")
        self._safe_after(0, lambda: self.update_status("Scanning for video changes...", "orange"))

        current_disk_videos = set()  # absolute paths
        new_videos_to_add = []
        existing_video_set = set(vp for vp, _ in self.video_paths)

        for scan_folder in (self.folders if self.folders else [self.folder]):
            if self.stop_indexing:
                break
            for root_dir, _, files in os.walk(scan_folder):
                if self.stop_indexing:
                    break
                for f in files:
                    if f.lower().endswith(VIDEO_EXTS):
                        abs_path = os.path.join(root_dir, f)
                        if self._is_excluded(abs_path):
                            continue
                        current_disk_videos.add(abs_path)
                        if abs_path not in existing_video_set:
                            new_videos_to_add.append(abs_path)

        if self.stop_indexing:
            self._handle_video_stop()
            return

        keep_indices = [i for i, (vp, _) in enumerate(self.video_paths) if vp in current_disk_videos]
        removed_count = len(self.video_paths) - len(keep_indices)

        if removed_count > 0:
            if self.video_embeddings is not None:
                self.video_embeddings = self.video_embeddings[keep_indices]
            self.video_paths = [self.video_paths[i] for i in keep_indices]
            safe_print(f"[VSCAN] Pruned {removed_count} stale frame entries.")

        if new_videos_to_add:
            safe_print(f"[VSCAN] Found {len(new_videos_to_add)} new video files.")
            self._process_video_batch(new_videos_to_add, is_update=True)
        else:
            if removed_count > 0:
                self._save_video_cache(allow_shrink=True)
            self.is_indexing = False
            self.is_stopping = False
            safe_print("[VSCAN] Video index is up to date.")
            self._safe_after(0, lambda: self.update_status("Up to date", "green"))
            self._safe_after(0, self.update_stats)

    def _flush_pending_batches(self):
        """Consolidate accumulated batch lists into single numpy arrays."""
        if getattr(self, '_pending_image_batches', None):
            batches = self._pending_image_batches
            self._pending_image_batches = []
            try:
                stacked = np.concatenate(batches, axis=0)
                del batches
                if self.image_embeddings is None:
                    self.image_embeddings = stacked
                else:
                    combined = np.concatenate([self.image_embeddings, stacked], axis=0)
                    del stacked
                    self.image_embeddings = combined
            except MemoryError:
                safe_print("[ERROR] Out of memory consolidating image embeddings.")
                self._pending_image_batches = []

        if getattr(self, '_pending_video_batches', None):
            batches = self._pending_video_batches
            self._pending_video_batches = []
            try:
                stacked = np.concatenate(batches, axis=0)
                del batches
                if self.video_embeddings is None:
                    self.video_embeddings = stacked
                else:
                    combined = np.concatenate([self.video_embeddings, stacked], axis=0)
                    del stacked
                    self.video_embeddings = combined
            except MemoryError:
                safe_print("[ERROR] Out of memory consolidating video embeddings.")

    def _write_failed_log(self, failed_list, log_filename):
        """Append failed files from this index run to a log file in the folder."""
        if not failed_list or not self.folder:
            return
        seen = set()
        unique = []
        for path, reason in failed_list:
            if path not in seen:
                seen.add(path)
                unique.append((path, reason))
        if not unique:
            return
        log_path = os.path.join(self.folder, log_filename)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n=== Index run: {timestamp} ===\n")
                for path, reason in unique:
                    f.write(f"{os.path.basename(path)} - {reason}\n")
                f.write(f"=== {len(unique)} failed ===\n")
            safe_print(f"[LOG] Wrote {len(unique)} failed entries to {log_filename}")
        except Exception as e:
            safe_print(f"[LOG] Could not write failed log: {e}")

    def _save_cache(self, allow_shrink=False):
        """Save cache with ABSOLUTE paths — never overwrites a larger existing cache."""
        if self.image_embeddings is not None and len(self.image_paths) > 0:
            try:
                if not allow_shrink and os.path.exists(self.cache_file):
                    try:
                        with open(self.cache_file, "rb") as f:
                            up = pickle.Unpickler(f)
                            existing_paths = up.load()
                        if len(existing_paths) > len(self.image_paths):
                            safe_print(f"[CACHE] Skipping save — existing cache has {len(existing_paths):,} images, current has only {len(self.image_paths):,}")
                            return
                    except Exception:
                        pass

                temp_file = self.cache_file + ".tmp"
                with open(temp_file, "wb") as f:
                    universal_paths = [p.replace('\\', '/') for p in self.image_paths]
                    pickle.dump((universal_paths, self.image_embeddings), f, protocol=pickle.HIGHEST_PROTOCOL)
                if os.path.exists(self.cache_file):
                    os.remove(self.cache_file)
                os.rename(temp_file, self.cache_file)
                safe_print(f"[CACHE] Saved {len(self.image_paths)} absolute paths to {self.cache_file}")
            except Exception as e:
                safe_print(f"[CACHE] Save Error: {e}")

    def _save_video_cache(self, allow_shrink=False):
        """Save video cache — never overwrites a larger existing cache."""
        if self.video_embeddings is not None and len(self.video_paths) > 0:
            try:
                if not allow_shrink and os.path.exists(self.video_cache_file):
                    try:
                        with open(self.video_cache_file, "rb") as f:
                            up = pickle.Unpickler(f)
                            existing_paths = up.load()
                        if len(existing_paths) > len(self.video_paths):
                            safe_print(f"[VCACHE] Skipping save — existing cache has {len(existing_paths):,} frames, current has only {len(self.video_paths):,}")
                            return
                    except Exception:
                        pass

                temp_file = self.video_cache_file + ".tmp"
                with open(temp_file, "wb") as f:
                    universal_video_paths = [(vp.replace('\\', '/'), ts) for vp, ts in self.video_paths]
                    pickle.dump((universal_video_paths, self.video_embeddings), f, protocol=pickle.HIGHEST_PROTOCOL)
                if os.path.exists(self.video_cache_file):
                    os.remove(self.video_cache_file)
                os.rename(temp_file, self.video_cache_file)
                safe_print(f"[VCACHE] Saved {len(self.video_paths):,} frame entries to {self.video_cache_file}")
            except Exception as e:
                safe_print(f"[VCACHE] Save Error: {e}")

    def _handle_stop(self):
        was_stopped = self.stop_indexing
        count = len(self.image_paths)

        self._write_failed_log(self._failed_images, "photosearchpro_skipped_images.txt")
        self._last_failed_images = list(self._failed_images)
        self._failed_images = []

        self.is_indexing = False
        self.stop_indexing = False
        self.is_stopping = False
        
        if self.clip_model and hasattr(self.clip_model, 'model'):
            import torch
            try:
                safe_print("[VRAM] Forcing memory release...")
                if not getattr(self.clip_model, 'onnx_disabled', False):
                    self.clip_model._destroy_onnx_session()
                original_device = self.clip_model.device
                self.clip_model.model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()
                self.clip_model.model.to(original_device)
                safe_print("[VRAM] Memory released, model back on GPU")
            except Exception as e:
                safe_print(f"[VRAM] Cleanup warning: {e}")
        
        self._safe_after(0, lambda: self.btn_stop.setText("STOP INDEX"))
        self._safe_after(0, lambda: self.progress.setValue(0))
        self._safe_after(0, lambda: self.progress_label.setText(""))
        self._safe_after(0, self.update_stats)

        if was_stopped:
            msg = f"Stopped. Saved {count:,} images."
            safe_print(f"[INDEX] {msg}")
            self._safe_after(0, lambda: self.update_status(msg, DANGER))

            if self.pending_action:
                safe_print("[ACTION] Executing pending action...")
                action = self.pending_action
                self.pending_action = None
                self._safe_after(100, action)
            elif count > 0:
                try:
                    query = self.query_entry.text().strip()
                    if query:
                        self._safe_after(500, self.do_search)
                except Exception:
                    pass
        else:
            if getattr(self, '_pending_video_refresh', False):
                self._pending_video_refresh = False
                if not self.video_cache_file:
                    self.video_cache_file = os.path.join(self.folder, self.get_video_cache_filename())
                self._safe_after(200, lambda: self.start_indexing(mode="video_refresh"))
                return
            self._safe_after(0, lambda: self.update_status("Indexing Complete", "green"))
            def _show_complete_bar(c=count):
                n_failed = len(self._last_failed_images)
                if n_failed:
                    msg = (f"Index complete — {c:,} images indexed. "
                           f"<a href='show_failed' style='color:#f5a623;'>"
                           f"{n_failed} file(s) skipped — click to view</a>")
                    self.info_bar.setOpenExternalLinks(False)
                    self.info_bar.linkActivated.connect(
                        lambda _: self._show_failed_files_dialog())
                    self.show_info_bar(msg)
                else:
                    self.show_info_bar(f"Index complete — {c:,} images indexed.")
            self._safe_after(0, _show_complete_bar)
            try:
                query = self.query_entry.text().strip()
                if query:
                    self._safe_after(500, self.do_search)
            except Exception:
                pass

    def _handle_video_stop(self):
        was_stopped = self.stop_indexing
        n_frames = len(self.video_paths)
        n_videos = len(set(vp for vp, _ in self.video_paths)) if self.video_paths else 0

        self._write_failed_log(self._failed_videos, "photosearchpro_skipped_videos.txt")
        self._last_failed_videos = list(self._failed_videos)
        self._failed_videos = []

        self.is_indexing = False
        self.stop_indexing = False
        self.is_stopping = False

        if self.clip_model and hasattr(self.clip_model, 'model'):
            import torch
            try:
                if not getattr(self.clip_model, 'onnx_disabled', False):
                    self.clip_model._destroy_onnx_session()
                original_device = self.clip_model.device
                self.clip_model.model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()
                self.clip_model.model.to(original_device)
            except Exception as e:
                safe_print(f"[VRAM] Video cleanup warning: {e}")

        self._safe_after(0, lambda: self.btn_stop.setText("STOP INDEX"))
        self._safe_after(0, lambda: self.progress.setValue(0))
        self._safe_after(0, lambda: self.progress_label.setText(""))
        self._safe_after(0, self.update_stats)

        if was_stopped:
            msg = f"Stopped. Saved {n_frames:,} frames from {n_videos:,} videos."
            safe_print(f"[VINDEX] {msg}")
            self._safe_after(0, lambda: self.update_status(msg, DANGER))
            if self.pending_action:
                action = self.pending_action
                self.pending_action = None
                self._safe_after(100, action)
        else:
            self._safe_after(0, lambda: self.update_status("Video indexing complete", "green"))
            def _show_video_complete_bar(v=n_videos, f=n_frames):
                n_failed = len(self._last_failed_videos)
                if n_failed:
                    msg = (f"Video index complete — {v:,} videos | {f:,} frames indexed. "
                           f"<a href='show_failed' style='color:#f5a623;'>"
                           f"{n_failed} video(s) skipped — click to view</a>")
                    self.info_bar.setOpenExternalLinks(False)
                    self.info_bar.linkActivated.connect(
                        lambda _: self._show_failed_files_dialog())
                    self.show_info_bar(msg)
                else:
                    self.show_info_bar(f"Video index complete — {v:,} videos | {f:,} frames indexed.")
            self._safe_after(0, _show_video_complete_bar)
            try:
                query = self.query_entry.text().strip()
                if query:
                    self._safe_after(500, self.do_search)
            except Exception:
                pass

    def delete_cache(self):
        if not self.folder: return

        img_count = len(self.image_paths)
        vid_count = len(set(vp for vp, _ in self.video_paths)) if self.video_paths else 0
        frame_count = len(self.video_paths)

        msg = "This will permanently DELETE cache files and re-index from scratch.\n\n"
        if img_count:
            msg += f"  - {img_count:,} images will be re-indexed\n"
        if vid_count:
            msg += f"  - {vid_count:,} videos ({frame_count:,} frames) will be re-indexed\n"
        msg += "\nThis cannot be undone. Continue?"

        if QMessageBox.question(self, "Delete Cache & Re-Index?", msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return

        try:
            if self.cache_file and os.path.exists(self.cache_file):
                os.remove(self.cache_file)
                safe_print("[CACHE] Deleted.")
        except: pass

        try:
            if self.video_cache_file and os.path.exists(self.video_cache_file):
                os.remove(self.video_cache_file)
                safe_print("[VCACHE] Deleted.")
        except: pass

        self.image_paths = []
        self.image_embeddings = None
        self.video_paths = []
        self.video_embeddings = None
        self.video_cache_file = None
        self.clear_results()
        self.update_stats()
        self.start_indexing(mode="full")

    def force_reindex(self):
        if not self.folder:
            QMessageBox.warning(self, "Warning", "Select a folder first.")
            return
        if self.video_paths or self.video_embeddings is not None:
            self._pending_video_refresh = True
        self.start_indexing(mode="refresh")

    def on_index_videos_click(self):
        if not self.is_safe_to_act(action_callback=self.index_videos, action_name="index videos"):
            return
        self.cancel_search(clear_ui=True)
        self.index_videos()

    def index_videos(self):
        if not self.folder:
            QMessageBox.warning(self, "Warning", "Select a folder first.")
            return
        if self.clip_model is None:
            QMessageBox.warning(self, "Wait", "Model is still loading...")
            return
        if not self.video_cache_file:
            self.video_cache_file = os.path.join(self.folder, self.get_video_cache_filename())
        if self.video_paths:
            vid_count = len(set(vp for vp, _ in self.video_paths))
            frame_count = len(self.video_paths)
            answer = QMessageBox.question(
                self,
                "Video Index",
                f"Video index has {vid_count:,} videos ({frame_count:,} frames).\n\n"
                f"Yes = Refresh (add new videos only, keeps existing)\n"
                f"No = Cancel (do nothing)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            mode = "video_refresh"
        else:
            mode = "video_full"
        self.start_indexing(mode=mode)


    def _deduplicate_video_results(self, all_results):
        """For video results, keep only the best scoring frame per video file."""
        seen_videos = {}
        deduped = []

        for item in all_results:
            score, path, result_type, metadata = item
            if result_type == "video":
                if path not in seen_videos or score > seen_videos[path][0]:
                    seen_videos[path] = (score, metadata.get("timestamp", 0.0))
            else:
                deduped.append(item)

        for path, (score, timestamp) in seen_videos.items():
            deduped.append((score, path, "video", {"timestamp": timestamp}))

        return deduped

    def parse_query(self, query):
        """Parse a query string into positive and negative terms."""
        positive_terms = []
        negative_terms = []

        pattern = r'(-?"[^"]+"|[-\w]+)'
        tokens = re.findall(pattern, query)

        for token in tokens:
            if token.startswith('-'):
                term = token[1:].strip('"')
                if term:
                    negative_terms.append(term)
            else:
                term = token.strip('"')
                if term:
                    positive_terms.append(term)

        return positive_terms, negative_terms

    def do_search(self):
        if self.is_searching or self.clip_model is None:
            safe_print("[SEARCH] Already searching or model not loaded")
            return

        # Vision-only models (DINOv2/v3) cannot process text queries
        if not self.model_has_text:
            label = MODEL_REGISTRY[self.active_model_key]["label"]
            QMessageBox.warning(
                self, "Text Search Not Available",
                f"The active model ({label}) does not have a text encoder.\n\n"
                "Switch to CLIP or SigLIP2 via the Model button to enable text search,\n"
                "or use the Image button to search by visual similarity."
            )
            return

        has_image_data = (self.image_embeddings is not None and len(self.image_paths) > 0) or \
                         bool(getattr(self, '_pending_image_batches', None))
        has_video_data = (self.video_embeddings is not None and len(self.video_paths) > 0) or \
                         bool(getattr(self, '_pending_video_batches', None))
        if not has_image_data and not has_video_data:
            QMessageBox.warning(self, "No Data", "Index is empty. Please select a folder.")
            return

        query = self.query_entry.text().strip()
        if not query:
            safe_print("[SEARCH] Empty query")
            self.update_status("Enter a search term", "orange")
            QMessageBox.information(
                self,
                "Empty Search",
                "Please type something in the search box to search.\n\n"
                "To search by image similarity, use the Image button next to the search box."
            )
            return

        safe_print(f"\n[SEARCH] Starting search for: '{query}'")
        self._save_to_history(query)
        self.search_thread = Thread(target=lambda: self.search(query, self.search_generation + 1), daemon=True)
        self.search_thread.start()

    def search(self, query, generation):
        if self.clip_model and not getattr(self.clip_model, 'use_onnx_visual', False):
            if not getattr(self.clip_model, 'onnx_disabled', False):
                try:
                    self.clip_model._create_onnx_session()
                except:
                    pass
        
        self.search_generation = generation
        self.is_searching = True
        self.stop_search = False
        self.thumbnail_count = 0
        
        safe_print(f"[SEARCH] Generation: {generation}, Query: '{query}'")
        
        self._safe_after(0, self.clear_results)
        self.total_found = 0

        if not self.is_indexing:
            self._safe_after(0, lambda: self.update_status("Searching...", "orange"))
            self._safe_after(0, lambda: self.progress.setRange(0, 0))

        try:
            positive_terms, negative_terms = self.parse_query(query)

            if not positive_terms:
                safe_print("[SEARCH] No positive search terms found")
                self._safe_after(0, lambda: self.update_status("No positive search terms", "orange"))
                self.is_searching = False
                return

            safe_print(f"[SEARCH] Positive: {positive_terms}, Negative: {negative_terms}")
            safe_print(f"[SEARCH] Encoding query...")

            pos_query = " ".join(positive_terms)
            text_embed = self.clip_model.encode_text([pos_query])

            if text_embed is None or text_embed.size == 0:
                safe_print("[SEARCH ERROR] Text encoding returned empty array")
                self._safe_after(0, lambda: self.update_status("Search failed - text encoding error", "red"))
                self.is_searching = False
                return

            # Hybrid search: blend text embedding with anchor image embedding
            anchor = getattr(self, '_anchor_embed', None)
            if anchor is not None:
                w = self.hybrid_slider.value() / 100.0
                blended = (1.0 - w) * text_embed.flatten() + w * anchor.flatten()
                norm = np.linalg.norm(blended)
                if norm > 1e-8:
                    blended = blended / norm
                text_embed = blended.reshape(text_embed.shape)
                safe_print(f"[SEARCH] Hybrid blend: text={1-w:.0%}, anchor={w:.0%}")

            if self.stop_search or generation != self.search_generation:
                safe_print("[SEARCH] Cancelled after text encoding")
                return

            safe_print(f"[SEARCH] Computing similarities...")

            neg_embed = None
            if negative_terms:
                neg_query = " ".join(negative_terms)
                safe_print(f"[SEARCH] Encoding negative terms: '{neg_query}'")
                neg_embed = self.clip_model.encode_text([neg_query])
                if neg_embed is not None and neg_embed.size > 0:
                    safe_print(f"[SEARCH] Negative terms applied")
                else:
                    neg_embed = None

            if self.stop_search:
                safe_print("[SEARCH] Cancelled after similarity computation")
                return

            if not self.is_indexing:
                self._safe_after(0, lambda: self.progress.setRange(0, 100))

            show_images = self.show_images_cb.isChecked()
            show_videos = self.show_videos_cb.isChecked()
            all_results = []

            with self._cache_lock:
                self._flush_pending_batches()

            min_score = self.score_slider.value() / 100.0

            if show_images and self.image_embeddings is not None and len(self.image_paths) > 0:
                sims_img = (self.image_embeddings @ text_embed.T).flatten()
                if neg_embed is not None:
                    sims_img = sims_img - (self.image_embeddings @ neg_embed.T).flatten()
                above = np.where(sims_img >= min_score)[0]
                for i in above:
                    abs_path = self.image_paths[i]  # already absolute
                    if not self._is_excluded(abs_path):
                        all_results.append((float(sims_img[i]), abs_path, "image", {}))

            if show_videos and self.video_embeddings is not None and len(self.video_paths) > 0:
                sims_vid = (self.video_embeddings @ text_embed.T).flatten()
                if neg_embed is not None:
                    sims_vid = sims_vid - (self.video_embeddings @ neg_embed.T).flatten()
                above_v = np.where(sims_vid >= min_score)[0]
                for i in above_v:
                    abs_vid_path, timestamp = self.video_paths[i]  # already absolute
                    if not self._is_excluded(abs_vid_path):
                        all_results.append((float(sims_vid[i]), abs_vid_path, "video", {"timestamp": timestamp}))

            safe_print(f"[SEARCH] Found {len(all_results)} total results")

            if self.dedup_video_cb.isChecked():
                all_results = self._deduplicate_video_results(all_results)

            all_results.sort(key=lambda x: x[0], reverse=True)
            self.all_search_results = all_results
            self.total_found = len(all_results)
            self.show_more_offset = 0

            if all_results:
                initial_n = max(10, self.top_n_slider.value() * 10)
                first_batch = all_results[:initial_n]
                self.show_more_offset = len(first_batch)

                safe_print(f"[SEARCH] Displaying first {len(first_batch)} of {self.total_found} results")

                if self.total_found < 6:
                    self._safe_after(500, self._maybe_suggest_lower_score)

                vp_width = self.scroll_area.viewport().width()
                cw = max(vp_width, CELL_WIDTH)
                self.render_cols = max(1, cw // CELL_WIDTH)

                self.start_thumbnail_loader(first_batch, generation)
            else:
                safe_print("[SEARCH] No results found")
                self._safe_after(0, lambda: self.update_status("No results found", "green"))
                self._safe_after(100, self._maybe_suggest_lower_score)
                self._safe_after(150, self._auto_find_retry)
                self.is_searching = False

        except Exception as e:
            if not self.stop_search:
                safe_print(f"[SEARCH ERROR] {e}")
                import traceback
                traceback.print_exc()
                self._safe_after(0, lambda: self.update_status("Search error - check console", "red"))
            self.is_searching = False

    def image_search(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Images (*.jpg *.jpeg *.png *.webp)")
        if not path: return
        self._set_last_image_search(path)
        gen = self.search_generation + 1
        self.search_thread = Thread(target=lambda: self._image_search(path, gen), daemon=True)
        self.search_thread.start()

    def _image_search_pil(self, pil_img, generation, label="image"):
        """Like _image_search but works directly from a PIL Image (e.g. from clipboard)."""
        if self.clip_model and not getattr(self.clip_model, 'use_onnx_visual', False):
            if not getattr(self.clip_model, 'onnx_disabled', False):
                try:
                    self.clip_model._create_onnx_session()
                except Exception:
                    pass
        self.search_generation = generation
        self.is_searching = True
        self.stop_search = False
        self.thumbnail_count = 0
        self._safe_after(0, self.clear_results)
        vp_width = self.scroll_area.viewport().width()
        cw = max(vp_width, CELL_WIDTH)
        self.render_cols = max(1, cw // CELL_WIDTH)
        self._safe_after(0, lambda: self.update_status(f"Searching by {label}...", "orange"))
        try:
            features = self.clip_model.encode_image_batch([pil_img])
            emb = features[0]
            show_images = self.show_images_cb.isChecked()
            show_videos = self.show_videos_cb.isChecked()
            all_results = []
            min_score = self.score_slider.value() / 100.0
            if show_images and self.image_embeddings is not None:
                sims_img = (self.image_embeddings @ emb).flatten()
                above = np.where(sims_img >= min_score)[0]
                for i in above:
                    abs_path = self.image_paths[i]  # already absolute
                    if not self._is_excluded(abs_path):
                        all_results.append((float(sims_img[i]), abs_path, "image", {}))
            if show_videos and self.video_embeddings is not None:
                sims_vid = (self.video_embeddings @ emb).flatten()
                above_v = np.where(sims_vid >= min_score)[0]
                for i in above_v:
                    abs_vid_path, timestamp = self.video_paths[i]  # already absolute
                    if not self._is_excluded(abs_vid_path):
                        all_results.append((float(sims_vid[i]), abs_vid_path, "video", {"timestamp": timestamp}))
            if all_results:
                if self.dedup_video_cb.isChecked():
                    all_results = self._deduplicate_video_results(all_results)
                all_results.sort(key=lambda x: x[0], reverse=True)
                self.all_search_results = all_results
                self.total_found = len(all_results)
                self.show_more_offset = 0
                initial_n = max(10, self.top_n_slider.value() * 10)
                first_batch = all_results[:initial_n]
                self.show_more_offset = len(first_batch)
                self.start_thumbnail_loader(first_batch, generation)
            else:
                self._safe_after(0, lambda: self.update_status("No matches", "green"))
                self._safe_after(150, self._auto_find_retry)
                self.is_searching = False
        except Exception as e:
            safe_print(f"[IMAGE SEARCH PIL ERROR] {e}")
            self.is_searching = False

    def _image_search(self, path, generation):
        if self.clip_model and not getattr(self.clip_model, 'use_onnx_visual', False):
            if not getattr(self.clip_model, 'onnx_disabled', False):
                try:
                    self.clip_model._create_onnx_session()
                except:
                    pass
        
        self.search_generation = generation
        self.is_searching = True
        self.stop_search = False
        self.thumbnail_count = 0
        
        self._safe_after(0, self.clear_results)

        vp_width = self.scroll_area.viewport().width()
        cw = max(vp_width, CELL_WIDTH)
        self.render_cols = max(1, cw // CELL_WIDTH)

        self._safe_after(0, lambda: self.update_status("Searching by image...", "orange"))
        
        try:
            img = open_image(path)
            if img is None:
                self.is_searching = False
                return
            features = self.clip_model.encode_image_batch([img])
            emb = features[0]
            
            show_images = self.show_images_cb.isChecked()
            show_videos = self.show_videos_cb.isChecked()
            all_results = []
            min_score = self.score_slider.value() / 100.0

            if show_images and self.image_embeddings is not None:
                sims_img = (self.image_embeddings @ emb).flatten()
                above = np.where(sims_img >= min_score)[0]
                for i in above:
                    abs_path = self.image_paths[i]  # already absolute
                    if not self._is_excluded(abs_path):
                        all_results.append((float(sims_img[i]), abs_path, "image", {}))

            if show_videos and self.video_embeddings is not None:
                sims_vid = (self.video_embeddings @ emb).flatten()
                above_v = np.where(sims_vid >= min_score)[0]
                for i in above_v:
                    abs_vid_path, timestamp = self.video_paths[i]  # already absolute
                    if not self._is_excluded(abs_vid_path):
                        all_results.append((float(sims_vid[i]), abs_vid_path, "video", {"timestamp": timestamp}))

            if all_results:
                if self.dedup_video_cb.isChecked():
                    all_results = self._deduplicate_video_results(all_results)
                all_results.sort(key=lambda x: x[0], reverse=True)
                self.all_search_results = all_results
                self.total_found = len(all_results)
                self.show_more_offset = 0

                initial_n = max(10, self.top_n_slider.value() * 10)
                first_batch = all_results[:initial_n]
                self.show_more_offset = len(first_batch)

                if self.total_found < 6:
                    self._safe_after(500, self._maybe_suggest_lower_score)

                self.start_thumbnail_loader(first_batch, generation)
            else:
                self._safe_after(0, lambda: self.update_status("No matches", "green"))
                self._safe_after(100, self._maybe_suggest_lower_score)
                self._safe_after(150, self._auto_find_retry)
                self.is_searching = False
        except Exception as e:
            safe_print(f"[IMAGE SEARCH ERROR] {e}")
            self.is_searching = False

    def start_thumbnail_loader(self, results, generation):
        import threading
        safe_print(f"[THUMBNAILS] start_thumbnail_loader called: {len(results)} results, "
                   f"gen={generation}, thread={threading.current_thread().name}")
        with self.thumbnail_queue.mutex:
            self.thumbnail_queue.queue.clear()
        t = Thread(target=self.load_thumbnails_worker, args=(results, generation), daemon=True)
        self._thumbnail_worker_thread = t
        t.start()
        safe_print(f"[THUMBNAILS] worker thread started: {t.name}, scheduling check_thumbnail_queue")
        self._safe_after(10, lambda: self.check_thumbnail_queue(generation))

    def load_thumbnails_worker(self, results, generation):
        import threading
        safe_print(f"[THUMBNAILS] load_thumbnails_worker started: {len(results)} items, "
                   f"gen={generation}, thread={threading.current_thread().name}")
        loaded = 0
        failed = 0
        for item in results:
            score, path, result_type, metadata = item
            if self.stop_search or generation != self.search_generation:
                safe_print(f"[THUMBNAILS] Stopped early: stop_search={self.stop_search}, "
                           f"gen={generation} vs current={self.search_generation}")
                break
            try:
                if result_type == "image":
                    if path.lower().endswith(RAW_EXTS):
                        try:
                            import rawpy
                            with rawpy.imread(path) as raw:
                                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
                            img = Image.fromarray(rgb)
                        except ImportError:
                            safe_print(f"[THUMBNAILS] rawpy not installed, skipping RAW: {path}")
                            failed += 1
                            continue
                        except Exception as e:
                            safe_print(f"[THUMBNAILS] RAW load failed: {path}: {e}")
                            failed += 1
                            continue
                    else:
                        safe_path = get_safe_path(path)
                        with open(safe_path, 'rb') as fh:
                            img = Image.open(fh)
                            if img.mode == 'P' and 'transparency' in img.info:
                                img = img.convert("RGBA")
                            img.load()
                    img.thumbnail(THUMBNAIL_SIZE)
                elif result_type == "video":
                    try:
                        import cv2
                    except ImportError:
                        if not getattr(self, '_cv2_missing_warned', False):
                            self._cv2_missing_warned = True
                            self._safe_after(0, lambda: QMessageBox.warning(
                                self,
                                "Missing Dependency",
                                "OpenCV is not installed - video thumbnails cannot be displayed.\n\n"
                                "Install it with:\n    pip install opencv-python"
                            ))
                        failed += 1
                        continue
                    timestamp = metadata.get("timestamp", 0.0)
                    cap = cv2.VideoCapture(path)
                    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                    ret, frame = cap.read()
                    cap.release()
                    if not ret or frame is None:
                        safe_print(f"[THUMBNAILS] Video frame read failed: {path}")
                        failed += 1
                        continue
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img.thumbnail(THUMBNAIL_SIZE)
                else:
                    safe_print(f"[THUMBNAILS] Unknown result_type={result_type}: {path}")
                    failed += 1
                    continue
                self.thumbnail_queue.put((score, path, img, result_type, metadata))
                loaded += 1
                safe_print(f"[THUMBNAILS] Queued {loaded}: {os.path.basename(path)}")
            except Exception as e:
                import traceback
                safe_print(f"[THUMBNAILS] Exception loading {path}: {e}")
                safe_print(traceback.format_exc())
                failed += 1
        safe_print(f"[THUMBNAILS] Worker done: {loaded} loaded, {failed} failed, "
                   f"queue size now={self.thumbnail_queue.qsize()}")

    def check_thumbnail_queue(self, generation):
        import threading
        safe_print(f"[QUEUE] check_thumbnail_queue called: gen={generation}, "
                   f"current_gen={self.search_generation}, stop={self.stop_search}, "
                   f"qsize={self.thumbnail_queue.qsize()}, thread={threading.current_thread().name}")
        if self.stop_search or generation != self.search_generation:
            safe_print(f"[QUEUE] Bailing: stop_search={self.stop_search}, "
                       f"gen mismatch={generation}!={self.search_generation}")
            return

        start_time = time.time()
        processed_this_cycle = 0
        while not self.thumbnail_queue.empty():
            try:
                item = self.thumbnail_queue.get_nowait()
                score, path, img, result_type, metadata = item
                safe_print(f"[QUEUE] Displaying: {os.path.basename(path)}")
                self.add_result_thumbnail(score, path, img, result_type, metadata)
                processed_this_cycle += 1
            except queue.Empty:
                break
            except Exception as e:
                import traceback
                safe_print(f"[QUEUE] Error displaying thumbnail: {e}")
                safe_print(traceback.format_exc())
                break

            if time.time() - start_time > 0.1: break

        done = self.thumbnail_count

        if processed_this_cycle > 0:
            safe_print(f"[QUEUE] Displayed {done} results this session")
        
        if not self.is_indexing:
            page_size = max(10, self.top_n_slider.value() * 10)
            total_pages = max(1, -(-self.total_found // page_size)) if self.total_found > 0 else 1
            current_page = max(1, (self.show_more_offset - 1) // page_size + 1) if self.show_more_offset > 0 else 1
            self.progress_label.setText(f"Page {current_page} of {total_pages}  -  {self.total_found:,} total results")

        if not self.thumbnail_queue.empty():
            QTimer.singleShot(10, lambda: self.check_thumbnail_queue(generation))
            return

        active_thread = getattr(self, '_thumbnail_worker_thread', None)
        if active_thread and active_thread.is_alive():
            QTimer.singleShot(20, lambda: self.check_thumbnail_queue(generation))
            return

        # All done
        self.is_searching = False
        if not self.is_indexing:
            safe_print(f"\n[THUMBNAILS] Display complete: {done} shown of {self.total_found:,} total")
            self.update_status(f"Found {self.total_found:,} results", "green")
        QTimer.singleShot(0, self._update_show_more_button)

    def _update_show_more_button(self):
        """Update page navigation buttons and page label"""
        page_size = max(10, self.top_n_slider.value() * 10)
        total = len(self.all_search_results)
        if total == 0:
            self.page_nav_widget.setVisible(False)
            return

        current_page = max(1, (self.show_more_offset - 1) // page_size + 1) if self.show_more_offset > 0 else 1
        total_pages = max(1, -(-total // page_size))

        self.page_label.setText(f"Page {current_page} of {total_pages}")
        self.prev_page_btn.setEnabled(self.show_more_offset > page_size)
        self.show_more_btn.setEnabled(self.show_more_offset < total)
        self.page_nav_widget.setVisible(True)

    def add_result_thumbnail(self, score, path, pil_img, result_type="image", metadata=None):
        import threading
        safe_print(f"[ADD_THUMB] called for {os.path.basename(path)}, thread={threading.current_thread().name}")
        if self.stop_search:
            safe_print(f"[ADD_THUMB] Skipped (stop_search=True)")
            return
        if metadata is None:
            metadata = {}

        cols = max(1, getattr(self, "render_cols", 1))
        idx = self.thumbnail_count
        row, col = divmod(idx, cols)
        self.thumbnail_count += 1
        safe_print(f"[ADD_THUMB] Adding card at row={row}, col={col}, cols={cols}")

        card = ResultCard()
        card._image_path = path
        card._on_single_click = self.handle_single_click
        card._on_double_click = self.handle_double_click
        card._on_context_menu = self._show_card_context_menu
        
        # Prune cache
        if len(self.thumbnail_images) >= MAX_THUMBNAIL_CACHE:
            oldest = list(self.thumbnail_images.keys())[:len(self.thumbnail_images) - MAX_THUMBNAIL_CACHE + 1]
            for k in oldest:
                del self.thumbnail_images[k]
        
        pixmap = pil_to_pixmap(pil_img)
        pixmap = pixmap.scaled(THUMBNAIL_SIZE[0], THUMBNAIL_SIZE[1],
                               Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
        
        cache_key = f"{path}@{metadata.get('timestamp',0)}" if result_type == "video" else path
        self.thumbnail_images[cache_key] = pixmap
        
        if result_type == "video":
            badge = QLabel("VIDEO", card)
            badge.setStyleSheet(f"background-color: {ORANGE}; color: #000000; font-size: 7px; font-weight: bold; padding: 1px 3px;")
            badge.move(4, 4)
            badge.raise_()
        
        card.img_label.setPixmap(pixmap)
        card.select_cb.setChecked(path in self.selected_images)
        card.select_cb.stateChanged.connect(
            lambda state, p=path: self._set_card_selection_by_path(p, state == Qt.CheckState.Checked.value))
        
        if result_type == "video":
            ts = metadata.get("timestamp", 0.0)
            m, s = int(ts)//60, int(ts)%60
            name = os.path.basename(path)[:40]
            text = f"{score:.3f}\n{name}\nt={m}:{s:02d}"
        else:
            name = os.path.basename(path)[:40]
            text = f"{score:.3f}\n{name}"
        card.info_label.setText(text)

        self.scroll_area._grid.addWidget(card, row, col)
        safe_print(f"[ADD_THUMB] Card added to grid OK")

    def clear_results(self, keep_results=False):
        """Clear search results and free RAM from thumbnails"""
        for card in self._get_all_cards():
            self.scroll_area._grid.removeWidget(card)
            card.deleteLater()
        
        self.thumbnail_count = 0
        self.page_nav_widget.setVisible(False)
        
        if not keep_results:
            self.thumbnail_images.clear()
            self.all_search_results = []
            self.show_more_offset = 0
            self.total_found = 0
            self.selected_images.clear()
        
        gc.collect()
        self.scroll_area.verticalScrollBar().setValue(0)

    def _maybe_suggest_lower_score(self):
        """Show an inline hint when search returns very few or no results."""
        current_score = self.score_slider.value() / 100.0
        if current_score > 0.05:
            self.show_info_bar(
                f"Only {self.total_found} result(s) at score {current_score:.2f}. "
                "Try lowering the Similarity Score slider "
                "(text: 0.15–0.30 | image: 0.60–0.85) "
                "or check that the Image/Video filter buttons are enabled."
            )

    def _auto_find_retry(self):
        """If Auto-find is enabled and last search had no results, lower score and retry."""
        if not getattr(self, 'auto_find_cb', None) or not self.auto_find_cb.isChecked():
            return
        if self.total_found > 0:
            return
        current = self.score_slider.value()
        if current <= 0:
            self.update_status("Auto-find: no matches found at any score", "orange")
            return
        step = 5
        new_val = max(0, current - step)
        self.score_slider.setValue(new_val)
        safe_print(f"[AUTO-FIND] No results at {current/100:.2f}, retrying at {new_val/100:.2f}")
        if self._last_search_type == 'image' and self._last_image_search_path:
            path = self._last_image_search_path
            gen = self.search_generation + 1
            self.search_thread = Thread(target=lambda: self._image_search(path, gen), daemon=True)
            self.search_thread.start()
        elif self._last_search_type == 'text':
            query = self.query_entry.text().strip()
            if query:
                gen = self.search_generation + 1
                self.search_thread = Thread(target=lambda: self.search(query, gen), daemon=True)
                self.search_thread.start()

    def prev_page_results(self):
        """Go back one page"""
        if self.is_searching or not self.all_search_results:
            return
        page_size = max(10, self.top_n_slider.value() * 10)
        new_offset = max(0, self.show_more_offset - (page_size * 2))
        prev_batch = self.all_search_results[new_offset:new_offset + page_size]
        if not prev_batch:
            return

        saved_results = self.all_search_results
        saved_total = self.total_found

        self.selected_images.clear()
        self.clear_results(keep_results=True)

        self.all_search_results = saved_results
        self.total_found = saved_total
        self.show_more_offset = new_offset + len(prev_batch)

        self.stop_search = False
        gen = self.search_generation
        t = Thread(target=self.load_thumbnails_worker, args=(prev_batch, gen), daemon=True)
        self._thumbnail_worker_thread = t
        t.start()
        QTimer.singleShot(10, lambda: self.check_thumbnail_queue(gen))

    def show_more_results(self):
        """Clear current page widgets and load next results"""
        if self.is_searching or not self.all_search_results:
            return
        page_size = max(10, self.top_n_slider.value() * 10)
        next_batch = self.all_search_results[self.show_more_offset:self.show_more_offset + page_size]
        if not next_batch:
            return

        saved_results = self.all_search_results
        saved_total = self.total_found
        new_offset = self.show_more_offset + len(next_batch)

        self.selected_images.clear()
        self.clear_results(keep_results=True)

        self.all_search_results = saved_results
        self.total_found = saved_total
        self.show_more_offset = new_offset

        self.stop_search = False
        gen = self.search_generation
        t = Thread(target=self.load_thumbnails_worker, args=(next_batch, gen), daemon=True)
        self._thumbnail_worker_thread = t
        t.start()
        QTimer.singleShot(10, lambda: self.check_thumbnail_queue(gen))


    def handle_single_click(self, path, widget=None):
        """Single click toggles card selection only. Use double-click to open viewer."""
        if widget is not None and hasattr(widget, "select_cb"):
            new_state = not widget.select_cb.isChecked()
            widget.select_cb.setChecked(new_state)
            self.toggle_selection(path, new_state)

    def handle_double_click(self, path):
        self.click_timer.stop()
        try:
            self.click_timer.timeout.disconnect()
        except:
            pass
        self.open_image_viewer(path)

    def open_in_explorer(self, path):
        """Open file location - path is already absolute from search results"""
        if isinstance(path, tuple):
            path = path[0]
        if os.path.exists(path):
            path = os.path.normpath(path)
            if os.name == 'nt':
                explorer_path = path.replace('\\\\?\\', '').replace('\\??\\', '')
                subprocess.Popen(f'explorer /select,"{explorer_path}"')
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', '-R', path])
            else:
                import shutil as _shutil
                folder = os.path.dirname(path)
                _fm_select = [
                    ('dolphin',  ['dolphin',  '--select', path]),
                    ('nautilus', ['nautilus', '--select', path]),
                    ('nemo',     ['nemo',     path]),
                    ('caja',     ['caja',     '--select', path]),
                    ('thunar',   ['thunar',   folder]),
                    ('pcmanfm',  ['pcmanfm',  folder]),
                ]
                launched = False
                for _name, _cmd in _fm_select:
                    if _shutil.which(_name):
                        subprocess.Popen(_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        launched = True
                        break
                if not launched:
                    subprocess.Popen(['xdg-open', folder], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def open_image_viewer(self, path):
        """Open image - path is already absolute from search results"""
        if os.path.exists(path):
            try:
                if os.name == 'nt':
                    os.startfile(path)
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', path])
                else:
                    subprocess.Popen(['xdg-open', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                safe_print(f"[OPEN] Failed to open viewer: {e}")

    def _remove_cards_from_ui(self, abs_paths):
        paths_set = set(abs_paths)
        to_remove = [c for c in self._get_all_cards() if c._image_path in paths_set]
        for card in to_remove:
            self.scroll_area._grid.removeWidget(card)
            card.deleteLater()
            if card._image_path in self.thumbnail_images:
                del self.thumbnail_images[card._image_path]
        
        # Re-grid remaining after a brief delay to allow deleteLater to complete
        QTimer.singleShot(50, lambda: self._reflow_grid())

    def _remove_paths_from_index(self, abs_paths):
        if not self.folder and not self.folders:
            return
        paths_to_remove = set(abs_paths)
        if not paths_to_remove:
            return

        # image_paths now stores absolute paths directly
        keep_indices = [i for i, p in enumerate(self.image_paths) if p not in paths_to_remove]
        self.image_paths = [self.image_paths[i] for i in keep_indices]
        if self.image_embeddings is not None:
            if keep_indices:
                self.image_embeddings = self.image_embeddings[keep_indices]
            else:
                self.image_embeddings = None
        self._save_cache(allow_shrink=True)

        if self.video_paths and paths_to_remove:
            keep_video = [i for i, (vp, _) in enumerate(self.video_paths) if vp not in paths_to_remove]
            if len(keep_video) < len(self.video_paths):
                if self.video_embeddings is not None:
                    self.video_embeddings = self.video_embeddings[keep_video] if keep_video else None
                self.video_paths = [self.video_paths[i] for i in keep_video]
                self._save_video_cache(allow_shrink=True)

        self.update_stats()

    def _select_card(self, card, select=True):
        """Programmatically select or deselect a card and update its checkbox"""
        path = getattr(card, '_image_path', None)
        if path is None:
            return
        if select:
            self.selected_images.add(path)
        else:
            self.selected_images.discard(path)
        card.select_cb.setChecked(select)

    def _clear_all_selections(self):
        """Clear selected_images AND uncheck all card checkboxes visually."""
        self.selected_images.clear()
        for card in self._get_all_cards():
            card.select_cb.setChecked(False)

    def _set_card_selection_by_path(self, path, select):
        """Select or deselect all cards matching path."""
        for card in self._get_all_cards():
            if getattr(card, '_image_path', None) == path:
                self._select_card(card, select=select)

    def _select_all_cards(self):
        for card in self._get_all_cards():
            self._select_card(card, select=True)

    def _deselect_all_cards(self):
        for card in self._get_all_cards():
            self._select_card(card, select=False)

    def toggle_selection(self, path, selected):
        if selected: 
            self.selected_images.add(path)
        else: 
            self.selected_images.discard(path)

    def _show_search_context_menu(self, pos):
        """Right-click context menu for the search bar."""
        menu = self.query_entry.createStandardContextMenu()
        menu.exec(self.query_entry.mapToGlobal(pos))

    def _show_canvas_context_menu(self, global_pos):
        """Right-click on scroll area background — general context menu"""
        menu = QMenu(self)
        menu.addAction("Select All", self._select_all_cards)
        menu.addAction("Deselect All", self._deselect_all_cards)
        menu.addSeparator()
        menu.addAction("Copy Selected", self.export_selected)
        menu.addAction("Move Selected", self.move_selected)
        menu.addAction("Delete Selected", self.delete_selected)
        menu.exec(global_pos)

    def _show_card_context_menu(self, global_pos, path):
        """Right-click on a specific card"""
        menu = QMenu(self)
        is_selected = path in self.selected_images
        if is_selected:
            menu.addAction("Deselect", lambda: self._set_card_selection_by_path(path, False))
        else:
            menu.addAction("Select", lambda: self._set_card_selection_by_path(path, True))
        menu.addSeparator()
        menu.addAction("Open in Viewer", lambda: self.open_image_viewer(path))
        menu.addAction("Show in File Manager", lambda: self.open_in_explorer(path))
        menu.addSeparator()
        menu.addAction("Copy", self.export_selected)
        menu.addAction("Move", self.move_selected)
        menu.addAction("Delete", self.delete_selected)
        menu.exec(global_pos)

    def export_selected(self):
        if not self.selected_images:
            QMessageBox.information(self, "Info", "No images selected")
            return
        export_dir = QFileDialog.getExistingDirectory(self, "Export to")
        if not export_dir: return
        copied = 0
        skipped = 0
        errors = []
        for path in self.selected_images:
            try:
                dest = os.path.join(export_dir, os.path.basename(path))
                if os.path.exists(dest):
                    skipped += 1
                    continue
                shutil.copy2(path, export_dir)
                copied += 1
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")
        lines = []
        if copied:
            lines.append(f"  {copied} file(s) copied successfully")
        if skipped:
            lines.append(f"  {skipped} file(s) skipped - already exist at destination")
        if errors:
            lines.append(f"  {len(errors)} file(s) failed:")
            lines.extend(f"    {e}" for e in errors[:5])
            if len(errors) > 5:
                lines.append(f"    ... and {len(errors) - 5} more")
        QMessageBox.information(self, "Copy Complete", "\n".join(lines) if lines else "Nothing to copy.")

    def move_selected(self):
        if not self.selected_images:
            QMessageBox.information(self, "Info", "No images selected")
            return
        dest_dir = QFileDialog.getExistingDirectory(self, "Move selected images to...")
        if not dest_dir:
            return
        moved = []
        skipped = 0
        errors = []
        for path in list(self.selected_images):
            try:
                clean_path = path.replace('\\\\?\\', '').replace('\\??\\', '')
                clean_path = os.path.normpath(os.path.abspath(clean_path))
                dest = os.path.join(dest_dir, os.path.basename(clean_path))
                if os.path.exists(dest):
                    skipped += 1
                    continue
                shutil.move(clean_path, dest_dir)
                moved.append(path)
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")
        if moved:
            self._remove_cards_from_ui(moved)
            self._remove_paths_from_index(moved)
            for p in moved:
                self.selected_images.discard(p)
        lines = []
        if moved:
            lines.append(f"  {len(moved)} file(s) moved successfully")
        if skipped:
            lines.append(f"  {skipped} file(s) skipped - already exist at destination")
        if errors:
            lines.append(f"  {len(errors)} file(s) failed:")
            lines.extend(f"    {e}" for e in errors[:5])
            if len(errors) > 5:
                lines.append(f"    ... and {len(errors) - 5} more")
        QMessageBox.information(self, "Move Complete", "\n".join(lines) if lines else "Nothing to move.")
        self.update_status(f"Moved {len(moved)} images", "green")

    def delete_selected(self):
        if not self.selected_images:
            QMessageBox.information(self, "Info", "No images selected")
            return

        try:
            from send2trash import send2trash
        except ImportError:
            QMessageBox.critical(
                self,
                "Missing Dependency",
                "Cannot delete safely - 'send2trash' is not installed.\n\n"
                "Files will NOT be deleted. Install it with:\n"
                "    pip install send2trash\n\n"
                "This ensures files go to the Recycle Bin and can be recovered."
            )
            return

        count = len(self.selected_images)
        if QMessageBox.question(self, "Confirm Delete",
                f"Move {count} selected file(s) to the Recycle Bin?\nFiles can be restored from the Recycle Bin.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return

        deleted = []
        errors = []
        if os.name != 'nt' and sys.platform != 'darwin':
            self.update_status("Moving to Trash... please wait", "orange")
        for path in list(self.selected_images):
            clean_path = path.replace('\\\\?\\', '').replace('\\??\\', '')
            clean_path = os.path.normpath(os.path.abspath(clean_path))
            safe_print(f"[DELETE] Attempting: {repr(clean_path)}, exists={os.path.exists(clean_path)}")
            try:
                send2trash(clean_path)
                deleted.append(path)
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")

        if deleted:
            self._remove_cards_from_ui(deleted)
            self._remove_paths_from_index(deleted)
            for p in deleted:
                self.selected_images.discard(p)

        if deleted:
            QMessageBox.information(self, "Moved to Recycle Bin",
                f"Successfully moved {len(deleted)} file(s) to the Recycle Bin.")
            self.update_status(f"Moved {len(deleted)} files to Recycle Bin", "green")
        if errors:
            QMessageBox.critical(self, "Delete Errors",
                f"{len(errors)} file(s) could not be deleted:\n\n" + "\n".join(errors[:8]))

    def show_index_info(self):
        if self.folders and len(self.folders) > 1:
            folder_str = f"{self.folders[0]}  (+{len(self.folders)-1} more)"
        else:
            folder_str = self.folder if self.folder else "No folder selected"
        cache_str = self.cache_file if self.cache_file else "No cache loaded"

        cache_size_str = "N/A"
        cache_mtime_str = "N/A"
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                size_bytes = os.path.getsize(self.cache_file)
                if size_bytes >= 1024 ** 3:
                    cache_size_str = f"{size_bytes / 1024**3:.2f} GB"
                elif size_bytes >= 1024 ** 2:
                    cache_size_str = f"{size_bytes / 1024**2:.2f} MB"
                else:
                    cache_size_str = f"{size_bytes / 1024:.1f} KB"
                mtime = os.path.getmtime(self.cache_file)
                cache_mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

        video_cache_str = self.video_cache_file if self.video_cache_file else "No video cache"
        video_cache_size_str = "N/A"
        if self.video_cache_file and os.path.exists(self.video_cache_file):
            try:
                size_bytes = os.path.getsize(self.video_cache_file)
                if size_bytes >= 1024 ** 3:
                    video_cache_size_str = f"{size_bytes / 1024**3:.2f} GB"
                elif size_bytes >= 1024 ** 2:
                    video_cache_size_str = f"{size_bytes / 1024**2:.2f} MB"
                else:
                    video_cache_size_str = f"{size_bytes / 1024:.1f} KB"
            except Exception:
                pass

        exclusions_str = ", ".join(sorted(self.excluded_folders)) if self.excluded_folders else "None"
        total_images = len(self.image_paths) if self.image_paths else 0
        total_frames = len(self.video_paths) if self.video_paths else 0
        total_videos = len(set(vp for vp, _ in self.video_paths)) if self.video_paths else 0

        info = (
            f"Folder:\n  {folder_str}\n\n"
            f"Image Cache:\n  {cache_str}\n"
            f"Cache Size:  {cache_size_str}\n"
            f"Cache Modified:  {cache_mtime_str}\n\n"
            f"Images Indexed:  {total_images:,}\n\n"
            f"Video Cache:\n  {video_cache_str}\n"
            f"Video Cache Size:  {video_cache_size_str}\n\n"
            f"Videos Indexed:  {total_videos:,} ({total_frames:,} frames)\n\n"
            f"Model:  {MODEL_REGISTRY[self.active_model_key]['label']}\n"
            f"  ({MODEL_REGISTRY[self.active_model_key]['subtitle']})\n\n"
            f"Exclusion Patterns:  {exclusions_str}"
        )
        QMessageBox.information(self, "Index Info", info)


    # --- Feature 1: Search History & Saved Presets ---

    def get_history_filename(self):
        return ".clip_search_history.json"

    def load_search_history(self):
        if not self.folder:
            self.search_history = []
            self.search_presets = []
            return
        path = os.path.join(self.folder, self.get_history_filename())
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.search_history = data.get("history", [])
                self.search_presets = data.get("presets", [])
                safe_print(f"[HISTORY] Loaded {len(self.search_history)} history entries, {len(self.search_presets)} presets")
            except Exception as e:
                safe_print(f"[HISTORY] Load error: {e}")
                self.search_history = []
                self.search_presets = []
        else:
            self.search_history = []
            self.search_presets = []

    def save_search_history(self):
        if not self.folder:
            return
        path = os.path.join(self.folder, self.get_history_filename())
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"history": self.search_history, "presets": self.search_presets}, f, indent=2)
        except Exception as e:
            safe_print(f"[HISTORY] Save error: {e}")

    def _save_to_history(self, query):
        """Record a search query in history (most recent first, no duplicates)."""
        if not query:
            return
        self.search_history = [h for h in self.search_history if h["query"] != query]
        self.search_history.insert(0, {
            "query": query,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        self.search_history = self.search_history[:50]
        self.save_search_history()

    def on_history_click(self):
        self.open_history_dialog()

    def open_history_dialog(self):
        if not self.folder:
            QMessageBox.warning(self, "No Folder", "Please select a folder first.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Search History & Saved Presets")
        dlg.resize(520, 500)
        layout = QVBoxLayout(dlg)

        tabs = QTabWidget()
        layout.addWidget(tabs, stretch=1)

        # --- Recent Searches tab ---
        history_widget = QWidget()
        h_layout = QVBoxLayout(history_widget)
        h_layout.addWidget(QLabel("Recent searches - double-click to run, or select and click Use:"))
        h_list = QListWidget()
        for item in self.search_history:
            h_list.addItem(f"{item['timestamp']}   {item['query']}")
        h_layout.addWidget(h_list, stretch=1)

        def use_history():
            row = h_list.currentRow()
            if row < 0: return
            query = self.search_history[row]["query"]
            self.query_entry.setText(query)
            dlg.accept()
            self.on_search_click()

        def star_history():
            row = h_list.currentRow()
            if row < 0: return
            query = self.search_history[row]["query"]
            if any(p["query"] == query for p in self.search_presets):
                QMessageBox.information(dlg, "Already Saved", "This query is already in your presets.")
                return
            name, ok = QInputDialog.getText(dlg, "Save Preset", "Preset name:", text=query)
            if ok and name.strip():
                self.search_presets.append({"name": name.strip(), "query": query})
                self.save_search_history()
                QMessageBox.information(dlg, "Saved", f"Saved preset: {name.strip()}")

        def clear_history():
            if QMessageBox.question(dlg, "Clear History", "Clear all recent search history?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                self.search_history = []
                self.save_search_history()
                h_list.clear()

        h_list.itemDoubleClicked.connect(lambda item: use_history())
        h_btn_row = QHBoxLayout()
        use_btn = QPushButton("Use")
        use_btn.setProperty("class", "accent")
        use_btn.clicked.connect(use_history)
        star_btn = QPushButton("Save as Preset")
        star_btn.clicked.connect(star_history)
        clear_btn = QPushButton("Clear History")
        clear_btn.setProperty("class", "danger")
        clear_btn.clicked.connect(clear_history)
        h_btn_row.addWidget(use_btn)
        h_btn_row.addWidget(star_btn)
        h_btn_row.addStretch()
        h_btn_row.addWidget(clear_btn)
        h_layout.addLayout(h_btn_row)
        tabs.addTab(history_widget, "Recent Searches")

        # --- Saved Presets tab ---
        presets_widget = QWidget()
        p_layout = QVBoxLayout(presets_widget)
        p_layout.addWidget(QLabel("Saved preset searches - double-click to run:"))
        p_list = QListWidget()

        def refresh_presets():
            p_list.clear()
            for p in self.search_presets:
                label = f"  {p['name']}" if p['name'] == p['query'] else f"  {p['name']}   ({p['query']})"
                p_list.addItem(label)

        refresh_presets()
        p_layout.addWidget(p_list, stretch=1)

        def use_preset():
            row = p_list.currentRow()
            if row < 0: return
            query = self.search_presets[row]["query"]
            self.query_entry.setText(query)
            dlg.accept()
            self.on_search_click()

        def delete_preset():
            row = p_list.currentRow()
            if row < 0: return
            del self.search_presets[row]
            self.save_search_history()
            refresh_presets()

        p_list.itemDoubleClicked.connect(lambda item: use_preset())
        p_btn_row = QHBoxLayout()
        p_use_btn = QPushButton("Use")
        p_use_btn.setProperty("class", "accent")
        p_use_btn.clicked.connect(use_preset)
        p_del_btn = QPushButton("Delete")
        p_del_btn.setProperty("class", "danger")
        p_del_btn.clicked.connect(delete_preset)
        p_btn_row.addWidget(p_use_btn)
        p_btn_row.addWidget(p_del_btn)
        p_btn_row.addStretch()
        p_layout.addLayout(p_btn_row)
        tabs.addTab(presets_widget, "Saved Presets")

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)
        dlg.exec()

    def open_exclusions_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Folder Exclusions")
        dlg.resize(480, 380)
        layout = QVBoxLayout(dlg)

        info = QLabel("Exclude images whose path contains any pattern below.\n"
                      "Case-sensitive substring match (e.g. 'nsfw', 'temp', 'backup').\n"
                      "Use forward slashes for folder separators (e.g. 'raw/originals').")
        info.setWordWrap(True)
        layout.addWidget(info)

        listbox = QListWidget()
        for pattern in sorted(self.excluded_folders):
            listbox.addItem(pattern)
        layout.addWidget(listbox, stretch=1)

        entry_row = QHBoxLayout()
        entry = QLineEdit()
        entry_row.addWidget(entry)

        def add_pattern():
            pat = entry.text().strip()
            if not pat or pat in self.excluded_folders:
                return
            self.excluded_folders.add(pat)
            listbox.addItem(pat)
            self.save_exclusions()
            entry.clear()

        def remove_pattern():
            row = listbox.currentRow()
            if row < 0: return
            pat = listbox.item(row).text()
            self.excluded_folders.discard(pat)
            listbox.takeItem(row)
            self.save_exclusions()

        entry.returnPressed.connect(add_pattern)
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(add_pattern)
        entry_row.addWidget(add_btn)
        layout.addLayout(entry_row)

        btn_row = QHBoxLayout()
        rem_btn = QPushButton("Remove Selected")
        rem_btn.setProperty("class", "danger")
        rem_btn.clicked.connect(remove_pattern)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(rem_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        note = QLabel("Note: Run Refresh after changing exclusions to apply them to the index.")
        note.setStyleSheet(f"color: {ORANGE}; font-style: italic; font-size: 8pt;")
        layout.addWidget(note)

        dlg.exec()

    # --- Feature 2: Duplicate & Near-Duplicate Finder ---

    def on_find_duplicates(self):
        if self.image_embeddings is None or len(self.image_paths) < 2:
            QMessageBox.warning(self, "Not Ready", "Please index images first (at least 2 images required).")
            return
        if self.is_indexing:
            QMessageBox.warning(self, "Busy", "Please wait for indexing to complete.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Find Duplicates")
        dlg.setMinimumWidth(460)
        layout = QVBoxLayout(dlg)

        layout.addWidget(QLabel("Similarity threshold for duplicate detection:"))
        layout.addWidget(QLabel("0.97 = near-identical   •   0.90 = very similar   •   0.80 = similar"))

        thresh_slider = QSlider(Qt.Orientation.Horizontal)
        thresh_slider.setRange(700, 999)
        thresh_slider.setValue(970)
        thresh_label = QLabel("0.970")
        thresh_slider.valueChanged.connect(lambda v: thresh_label.setText(f"{v/1000.0:.3f}"))
        row = QHBoxLayout()
        row.addWidget(thresh_slider)
        row.addWidget(thresh_label)
        layout.addLayout(row)

        auto_cb = QCheckBox("Auto-adjust threshold if no matches found")
        auto_cb.setToolTip(
            "If the scan finds 0 duplicate groups, automatically retry\n"
            "with the threshold lowered by 0.01 each pass until a match\n"
            "is found or the floor of 0.70 is reached."
        )
        auto_cb.setChecked(False)
        layout.addWidget(auto_cb)

        n_images = len(self.image_paths)
        layout.addWidget(QLabel(f"{n_images:,} images in index"))

        def start_scan():
            threshold = thresh_slider.value() / 1000.0
            auto_adjust = auto_cb.isChecked()
            dlg.accept()
            self.update_status(f"Scanning for duplicates (threshold={threshold:.3f})...", "orange")
            self.progress.setRange(0, 0)
            Thread(target=lambda: self._find_duplicates_worker(threshold, auto_adjust), daemon=True).start()

        btn_frame = QHBoxLayout()
        scan_btn = QPushButton("Find Duplicates")
        scan_btn.setProperty("class", "accent")
        scan_btn.clicked.connect(start_scan)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dlg.reject)
        btn_frame.addWidget(scan_btn)
        btn_frame.addWidget(cancel_btn)
        layout.addLayout(btn_frame)
        dlg.exec()

    def _find_duplicates_worker(self, threshold, auto_adjust=False):
        """Find near-duplicate image groups using embedding cosine similarity."""
        try:
            embeddings = self.image_embeddings
            n = len(self.image_paths)
            MIN_THRESHOLD = 0.70
            current_threshold = threshold

            while True:
                safe_print(f"[DUPES] Scanning {n:,} images, threshold={current_threshold:.3f}...")

                parent = list(range(n))

                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x

                def union(x, y):
                    px, py = find(x), find(y)
                    if px != py:
                        parent[px] = py

                chunk_size = 500
                total_chunks = (n + chunk_size - 1) // chunk_size
                pair_count = 0

                for chunk_idx in range(total_chunks):
                    start = chunk_idx * chunk_size
                    end = min(start + chunk_size, n)
                    chunk = embeddings[start:end]
                    sims = chunk @ embeddings.T

                    rows, cols = np.where(sims >= current_threshold)
                    for r, c in zip(rows.tolist(), cols.tolist()):
                        actual_r = start + r
                        if actual_r < c:
                            union(actual_r, c)
                            pair_count += 1

                    pct = int((chunk_idx + 1) / total_chunks * 100)
                    t = current_threshold
                    self._safe_after(0, lambda p=pct, th=t: self.progress_label.setText(
                        f"Scanning for duplicates (threshold={th:.3f})... {p}%"))

                # Rebuild parent after loop (find() mutates parent in-place via path compression)
                groups = {}
                for i in range(n):
                    root = find(i)
                    groups.setdefault(root, []).append(i)

                dup_groups = [sorted(members) for members in groups.values() if len(members) >= 2]
                dup_groups.sort(key=lambda g: len(g), reverse=True)

                safe_print(f"[DUPES] Found {len(dup_groups)} groups ({pair_count} pairs) "
                           f"at threshold={current_threshold:.3f}")

                if dup_groups:
                    break  # success

                # No matches — optionally retry at a lower threshold
                next_threshold = round(current_threshold - 0.01, 3)
                if auto_adjust and next_threshold >= MIN_THRESHOLD:
                    safe_print(f"[DUPES] No matches; auto-adjusting to {next_threshold:.3f}...")
                    current_threshold = next_threshold
                    parent = list(range(n))  # reset union-find for next pass
                    continue
                else:
                    break  # give up

            self._safe_after(0, lambda: self.progress.setRange(0, 100))
            self._safe_after(0, lambda: self.progress.setValue(0))
            self._safe_after(0, lambda: self.progress_label.setText(""))

            if not dup_groups:
                min_tried = current_threshold
                self._safe_after(0, lambda: self.update_status("No duplicates found", "green"))
                msg = (f"No duplicate images found at threshold {threshold:.3f}.")
                if auto_adjust:
                    msg += f"\n\nAuto-adjust searched down to {min_tried:.3f} with no results."
                else:
                    msg += "\n\nTry lowering the threshold or enabling Auto-adjust."
                self._safe_after(0, lambda m=msg: QMessageBox.information(
                    self, "No Duplicates", m))
                return

            # Pre-load PIL images in this background thread so the main thread
            # doesn't block on disk I/O when building the dialog.
            safe_print(f"[DUPES] Pre-loading thumbnails for dialog...")
            group_data = []
            for group in dup_groups:
                members = []
                for img_idx in group:
                    abs_path = self.image_paths[img_idx]  # already absolute
                    pil_img = None
                    try:
                        if abs_path.lower().endswith(RAW_EXTS):
                            import rawpy
                            with rawpy.imread(abs_path) as raw:
                                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
                            pil_img = Image.fromarray(rgb)
                        else:
                            safe_p = get_safe_path(abs_path)
                            pil_img = Image.open(safe_p)
                            pil_img.load()
                        pil_img.thumbnail((150, 150))
                    except Exception:
                        pil_img = None
                    members.append((abs_path, abs_path, pil_img))
                group_data.append(members)

            total_redundant = sum(len(g) - 1 for g in dup_groups)
            self._safe_after(0, lambda: self.update_status(
                f"Found {len(dup_groups)} duplicate groups ({total_redundant} redundant files)", ORANGE))
            self._safe_after(0, lambda gd=group_data, th=current_threshold:
                self.open_duplicates_dialog(gd, th))

        except Exception as e:
            safe_print(f"[DUPES] Error: {e}")
            import traceback; traceback.print_exc()
            self._safe_after(0, lambda: self.progress.setRange(0, 100))
            self._safe_after(0, lambda: self.progress.setValue(0))
            self._safe_after(0, lambda: self.update_status("Duplicate scan failed", "red"))

    def open_duplicates_dialog(self, group_data, threshold=None):
        """Display duplicate groups with checkboxes and multiple action options.

        group_data: list of groups, each group is a list of
            (abs_path, rel_path, pil_img_or_None) pre-loaded by the worker thread.
        threshold: the similarity threshold that was used (for display).
        """
        dlg = QDialog(self)
        total_redundant = sum(len(g) - 1 for g in group_data)
        thresh_str = f"  (threshold: {threshold:.3f})" if threshold is not None else ""
        dlg.setWindowTitle(
            f"Duplicate Finder — {len(group_data)} groups, {total_redundant} redundant files{thresh_str}")
        dlg.resize(960, 680)
        layout = QVBoxLayout(dlg)

        # Header
        hdr_lbl = QLabel(
            f"{len(group_data)} duplicate groups   —   {total_redundant} potentially redundant files{thresh_str}")
        hdr_lbl.setStyleSheet("font-size: 10pt; font-weight: bold;")
        layout.addWidget(hdr_lbl)
        layout.addWidget(QLabel(
            "Tick images you want to act on. 'Keep First, Check Rest' is a quick shortcut per group."))

        # Scrollable area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner_widget = QWidget()
        inner_layout = QVBoxLayout(inner_widget)
        scroll.setWidget(inner_widget)
        layout.addWidget(scroll, stretch=1)

        # path → (QCheckBox, group_index) so we can look up group later
        action_vars = {}   # abs_path -> QCheckBox
        path_to_group = {}  # abs_path -> grp_idx

        for grp_idx, group in enumerate(group_data):
            grp_frame = QFrame()
            grp_frame.setStyleSheet(f"background-color: {CARD_BG}; border: 1px solid {BORDER};")
            grp_layout = QVBoxLayout(grp_frame)

            grp_hdr = QHBoxLayout()
            grp_hdr_lbl = QLabel(f"Group {grp_idx + 1}  —  {len(group)} similar images")
            grp_hdr_lbl.setStyleSheet("font-weight: bold;")
            grp_hdr.addWidget(grp_hdr_lbl)

            def _check_rest(g=group):
                for k, (ap, _rp, _img) in enumerate(g):
                    if ap in action_vars:
                        action_vars[ap].setChecked(k > 0)

            def _check_all_in_group(g=group):
                for ap, _rp, _img in g:
                    if ap in action_vars:
                        action_vars[ap].setChecked(True)

            all_btn = QPushButton("Select All in Group")
            all_btn.clicked.connect(lambda _checked=False, g=group: _check_all_in_group(g))
            grp_hdr.addWidget(all_btn)

            grp_hdr.addStretch()

            keep_btn = QPushButton("Keep First, Check Rest")
            keep_btn.clicked.connect(lambda _checked=False, g=group: _check_rest(g))
            grp_hdr.addWidget(keep_btn)
            grp_layout.addLayout(grp_hdr)

            thumbs_row_widget = QWidget()
            thumbs_row_layout = QHBoxLayout(thumbs_row_widget)
            thumbs_row_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

            for k, (abs_path, rel_path, pil_img) in enumerate(group):
                cell = QWidget()
                cell_layout = QVBoxLayout(cell)
                cell_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

                if pil_img is not None:
                    pixmap = pil_to_pixmap(pil_img)
                    img_lbl = ClickableImageLabel()
                    img_lbl.setPixmap(pixmap)
                    img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    img_lbl._on_click = (lambda p=abs_path: self.open_image_viewer(p))
                    cell_layout.addWidget(img_lbl)
                else:
                    no_prev = QLabel("[no preview]")
                    no_prev.setFixedSize(150, 100)
                    cell_layout.addWidget(no_prev)

                name = os.path.basename(rel_path)
                if len(name) > 22:
                    name = name[:19] + "..."
                name_lbl = QLabel(name)
                name_lbl.setWordWrap(True)
                name_lbl.setStyleSheet("font-size: 8pt;")
                cell_layout.addWidget(name_lbl)

                try:
                    fsize = os.path.getsize(abs_path)
                    size_str = (f"{fsize/1024:.0f} KB" if fsize < 1024*1024
                                else f"{fsize/1024/1024:.1f} MB")
                    size_lbl = QLabel(size_str)
                    size_lbl.setStyleSheet("color: #888888; font-size: 8pt;")
                    cell_layout.addWidget(size_lbl)
                except Exception:
                    pass

                act_cb = QCheckBox("Act on")
                act_cb.setChecked(k > 0)
                action_vars[abs_path] = act_cb
                path_to_group[abs_path] = grp_idx
                cell_layout.addWidget(act_cb)

                thumbs_row_layout.addWidget(cell)

            grp_layout.addWidget(thumbs_row_widget)
            inner_layout.addWidget(grp_frame)

        inner_layout.addStretch()

        # ── Bottom action bar ─────────────────────────────────────────────────
        count_label = QLabel("")
        layout.addWidget(count_label)

        def _update_count():
            n = sum(1 for cb in action_vars.values() if cb.isChecked())
            action = action_combo.currentText()
            count_label.setText(f"{n} file(s) checked  —  action: {action}")

        action_row = QHBoxLayout()
        action_row.addWidget(QLabel("Action:"))
        action_combo = QComboBox()
        action_combo.addItems([
            "Delete (Recycle Bin)",
            "Move to Folder…",
            "Move into Group Subfolders…",
        ])
        action_combo.setMinimumWidth(240)
        action_combo.currentIndexChanged.connect(lambda _: _update_count())
        action_row.addWidget(action_combo)

        apply_btn = QPushButton("Apply to Checked")
        apply_btn.setProperty("class", "accent")
        action_row.addWidget(apply_btn)
        action_row.addStretch()

        def _select_all():
            for cb in action_vars.values():
                cb.setChecked(True)

        def _select_none():
            for cb in action_vars.values():
                cb.setChecked(False)

        sel_all_btn = QPushButton("Select All")
        sel_all_btn.clicked.connect(_select_all)
        action_row.addWidget(sel_all_btn)

        sel_none_btn = QPushButton("Select None")
        sel_none_btn.clicked.connect(_select_none)
        action_row.addWidget(sel_none_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        action_row.addWidget(close_btn)
        layout.addLayout(action_row)

        for cb in action_vars.values():
            cb.stateChanged.connect(lambda _: _update_count())
        _update_count()

        # ── Action implementations ────────────────────────────────────────────

        def _checked_paths():
            return [p for p, cb in action_vars.items() if cb.isChecked() and os.path.exists(p)]

        def _do_delete():
            to_del = _checked_paths()
            if not to_del:
                QMessageBox.information(dlg, "Nothing Checked", "No files are checked.")
                return
            if QMessageBox.question(
                    dlg, "Confirm Delete",
                    f"Move {len(to_del)} file(s) to the Recycle Bin?\n"
                    "Files can be restored from the Recycle Bin.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) != QMessageBox.StandardButton.Yes:
                return
            try:
                from send2trash import send2trash
            except ImportError:
                QMessageBox.critical(dlg, "Missing Dependency",
                    "send2trash is not installed.\npip install send2trash")
                return
            deleted, errors = [], []
            for p in to_del:
                try:
                    send2trash(os.path.normpath(p))
                    deleted.append(p)
                except Exception as e:
                    errors.append(f"{os.path.basename(p)}: {e}")
            if deleted:
                self._remove_paths_from_index(deleted)
                self._remove_cards_from_ui(deleted)
                for p in deleted:
                    action_vars.pop(p, None)
                _update_count()
            if errors:
                QMessageBox.critical(dlg, "Errors", "\n".join(errors[:5]))
            if deleted:
                QMessageBox.information(dlg, "Done",
                    f"Moved {len(deleted)} file(s) to Recycle Bin.")

        def _do_move_to_folder():
            to_move = _checked_paths()
            if not to_move:
                QMessageBox.information(dlg, "Nothing Checked", "No files are checked.")
                return
            dest = QFileDialog.getExistingDirectory(
                dlg, "Select Destination Folder", self.folder or "")
            if not dest:
                return
            moved, errors = [], []
            for p in to_move:
                target = os.path.join(dest, os.path.basename(p))
                # Avoid collision
                if os.path.exists(target):
                    base, ext = os.path.splitext(os.path.basename(p))
                    target = os.path.join(dest, f"{base}_dup{ext}")
                try:
                    import shutil as _shutil
                    _shutil.move(p, target)
                    moved.append(p)
                except Exception as e:
                    errors.append(f"{os.path.basename(p)}: {e}")
            if moved:
                self._remove_paths_from_index(moved)
                self._remove_cards_from_ui(moved)
                for p in moved:
                    action_vars.pop(p, None)
                _update_count()
            if errors:
                QMessageBox.critical(dlg, "Errors", "\n".join(errors[:5]))
            if moved:
                QMessageBox.information(dlg, "Done",
                    f"Moved {len(moved)} file(s) to:\n{dest}")

        def _do_move_to_group_subfolders():
            to_move = _checked_paths()
            if not to_move:
                QMessageBox.information(dlg, "Nothing Checked", "No files are checked.")
                return
            dest_root = QFileDialog.getExistingDirectory(
                dlg, "Select Root Folder for Group Subfolders", self.folder or "")
            if not dest_root:
                return
            import shutil as _shutil
            moved, errors = [], []
            for p in to_move:
                grp_idx = path_to_group.get(p, 0)
                grp_folder = os.path.join(dest_root, f"Group_{grp_idx + 1:03d}")
                os.makedirs(grp_folder, exist_ok=True)
                target = os.path.join(grp_folder, os.path.basename(p))
                if os.path.exists(target):
                    base, ext = os.path.splitext(os.path.basename(p))
                    target = os.path.join(grp_folder, f"{base}_dup{ext}")
                try:
                    _shutil.move(p, target)
                    moved.append(p)
                except Exception as e:
                    errors.append(f"{os.path.basename(p)}: {e}")
            if moved:
                self._remove_paths_from_index(moved)
                self._remove_cards_from_ui(moved)
                for p in moved:
                    action_vars.pop(p, None)
                _update_count()
            if errors:
                QMessageBox.critical(dlg, "Errors", "\n".join(errors[:5]))
            if moved:
                QMessageBox.information(dlg, "Done",
                    f"Moved {len(moved)} file(s) into group subfolders under:\n{dest_root}")

        def _apply():
            idx = action_combo.currentIndex()
            if idx == 0:
                _do_delete()
            elif idx == 1:
                _do_move_to_folder()
            elif idx == 2:
                _do_move_to_group_subfolders()

        apply_btn.clicked.connect(_apply)
        dlg.exec()


    # --- Feature 3: Smart Albums (Auto-Collections via Clustering) ---

    def _kmeans_numpy(self, embeddings, n_clusters, max_iter=30):
        """K-Means clustering on L2-normalized embeddings (cosine similarity)."""
        n = len(embeddings)
        if n <= n_clusters:
            return list(range(n)), embeddings.copy()

        rng = np.random.default_rng(42)
        idx = rng.choice(n, n_clusters, replace=False)
        centroids = embeddings[idx].copy()
        labels = np.zeros(n, dtype=np.int32)

        for iteration in range(max_iter):
            sims = embeddings @ centroids.T
            new_labels = np.argmax(sims, axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels) and iteration > 0:
                break
            labels = new_labels
            new_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                mask = labels == k
                if mask.any():
                    c = embeddings[mask].mean(axis=0)
                    norm = np.linalg.norm(c)
                    new_centroids[k] = c / norm if norm > 0 else c
                else:
                    new_centroids[k] = embeddings[rng.integers(n)]
            centroids = new_centroids

        return labels.tolist(), centroids

    def on_smart_albums(self):
        if self.image_embeddings is None or len(self.image_paths) < 2:
            QMessageBox.warning(self, "Not Ready", "Please index images first.")
            return
        if self.is_indexing:
            QMessageBox.warning(self, "Busy", "Please wait for indexing to complete.")
            return

        n_images = len(self.image_paths)
        default_clusters = min(15, max(3, n_images // 100))

        dlg = QDialog(self)
        dlg.setWindowTitle("Smart Albums - Auto-Collections")
        dlg.resize(400, 230)
        dlg.setFixedSize(400, 230)
        layout = QVBoxLayout(dlg)

        lbl = QLabel("Automatically group your images into thematic albums\nusing AI clustering on their visual embeddings.")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)

        spin_row = QHBoxLayout()
        spin_row.addWidget(QLabel("Number of albums:"))
        n_clusters_spin = QSpinBox()
        n_clusters_spin.setRange(2, 50)
        n_clusters_spin.setValue(default_clusters)
        spin_row.addWidget(n_clusters_spin)
        spin_row.addStretch()
        layout.addLayout(spin_row)

        layout.addWidget(QLabel(f"{n_images:,} images will be clustered"))

        def start_clustering():
            n_clusters = n_clusters_spin.value()
            dlg.accept()
            self.update_status(f"Building {n_clusters} smart albums...", "orange")
            self.progress.setRange(0, 0)
            Thread(target=lambda: self._smart_albums_worker(n_clusters), daemon=True).start()

        btn_row = QHBoxLayout()
        build_btn = QPushButton("Build Smart Albums")
        build_btn.setProperty("class", "accent")
        build_btn.clicked.connect(start_clustering)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(build_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        dlg.exec()

    def _smart_albums_worker(self, n_clusters):
        """Cluster images into smart albums using K-Means."""
        try:
            embeddings = self.image_embeddings
            n = len(self.image_paths)
            safe_print(f"[ALBUMS] Clustering {n:,} images into {n_clusters} albums...")
            self._safe_after(0, lambda: self.progress_label.setText("Running K-Means clustering..."))

            try:
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=n_clusters, random_state=42, n_init=5, max_iter=100)
                labels = km.fit_predict(embeddings).tolist()
                centroids = km.cluster_centers_
                norms = np.linalg.norm(centroids, axis=1, keepdims=True)
                centroids = centroids / np.maximum(norms, 1e-8)
                safe_print("[ALBUMS] Used sklearn KMeans")
            except ImportError:
                safe_print("[ALBUMS] sklearn not found, using numpy K-Means")
                labels, centroids = self._kmeans_numpy(embeddings, n_clusters)

            cluster_info = []
            for k in range(n_clusters):
                members = [i for i, lbl in enumerate(labels) if lbl == k]
                if not members:
                    continue
                centroid = centroids[k]
                member_embs = embeddings[members]
                sims = member_embs @ centroid
                best_idx = members[int(np.argmax(sims))]
                cluster_info.append({
                    "cluster_id": k,
                    "members": members,
                    "representative": best_idx,
                    "size": len(members)
                })

            cluster_info.sort(key=lambda x: x["size"], reverse=True)
            safe_print(f"[ALBUMS] Built {len(cluster_info)} albums")

            self._safe_after(0, lambda: self.progress.setRange(0, 100))
            self._safe_after(0, lambda: self.progress.setValue(0))
            self._safe_after(0, lambda: self.progress_label.setText(""))
            self._safe_after(0, lambda: self.update_status(
                f"Smart Albums ready: {len(cluster_info)} albums", "green"))
            self._safe_after(0, lambda: self.open_smart_albums_dialog(cluster_info))

        except Exception as e:
            safe_print(f"[ALBUMS] Error: {e}")
            import traceback; traceback.print_exc()
            self._safe_after(0, lambda: self.progress.setRange(0, 100))
            self._safe_after(0, lambda: self.progress.setValue(0))
            self._safe_after(0, lambda: self.update_status("Smart Albums failed", "red"))

    def open_smart_albums_dialog(self, cluster_info):
        """Display smart albums as a grid of representative thumbnails."""
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Smart Albums - {len(cluster_info)} Auto-Collections")
        dlg.resize(920, 660)
        layout = QVBoxLayout(dlg)

        info_lbl = QLabel("AI-generated collections based on visual similarity. Click 'View Album' to browse a collection.")
        info_lbl.setStyleSheet(f"background-color: {PANEL_BG}; padding: 5px;")
        layout.addWidget(info_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(8)
        scroll.setWidget(grid_widget)
        layout.addWidget(scroll, stretch=1)

        ALBUM_THUMB = (160, 160)
        COLS = 4

        for idx, info in enumerate(cluster_info):
            rep_idx = info["representative"]
            abs_path = self.image_paths[rep_idx]  # already absolute
            row, col = divmod(idx, COLS)

            card = QFrame()
            card.setStyleSheet(f"background-color: {CARD_BG}; border: 1px solid {BORDER};")
            card_layout = QVBoxLayout(card)
            card_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

            try:
                if abs_path.lower().endswith(RAW_EXTS):
                    import rawpy
                    with rawpy.imread(abs_path) as raw:
                        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
                    img = Image.fromarray(rgb)
                else:
                    safe_p = get_safe_path(abs_path)
                    img = Image.open(safe_p)
                    img.load()
                img.thumbnail(ALBUM_THUMB)
                pixmap = pil_to_pixmap(img)
                img_lbl = QLabel()
                img_lbl.setPixmap(pixmap)
                img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                card_layout.addWidget(img_lbl)
            except Exception:
                no_prev = QLabel("[preview unavailable]")
                no_prev.setFixedSize(160, 80)
                card_layout.addWidget(no_prev)

            alb_title = QLabel(f"Album {idx + 1}")
            alb_title.setStyleSheet(f"color: {ACCENT_SECONDARY}; font-weight: bold; font-size: 9pt; border: none;")
            card_layout.addWidget(alb_title)

            size_lbl = QLabel(f"{info['size']:,} images")
            size_lbl.setStyleSheet("border: none;")
            card_layout.addWidget(size_lbl)

            def view_album(members=info["members"], album_num=idx + 1):
                self.cancel_search(clear_ui=True)
                album_results = [
                    (1.0, self.image_paths[i], "image", {})  # already absolute
                    for i in members
                ]
                self.all_search_results = album_results
                self.total_found = len(album_results)
                self.show_more_offset = 0
                self.update_status(f"Smart Album {album_num}: {len(album_results)} images", "green")
                vp_width = self.scroll_area.viewport().width()
                cw = max(vp_width, CELL_WIDTH)
                self.render_cols = max(1, cw // CELL_WIDTH)
                initial_n = max(10, self.top_n_slider.value() * 10)
                first_batch = album_results[:initial_n]
                self.show_more_offset = len(first_batch)
                self.stop_search = False
                self.is_searching = True
                self.start_thumbnail_loader(first_batch, self.search_generation)

            view_btn = QPushButton("View Album")
            view_btn.clicked.connect(lambda _, fn=view_album: fn())
            card_layout.addWidget(view_btn)

            grid_layout.addWidget(card, row, col)

        bottom_row = QHBoxLayout()
        tip_lbl = QLabel("Tip: Rebuild with more or fewer albums to change granularity.")
        tip_lbl.setStyleSheet("color: #888888; font-size: 8pt;")
        bottom_row.addWidget(tip_lbl)
        bottom_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        bottom_row.addWidget(close_btn)
        layout.addLayout(bottom_row)
        dlg.exec()


    # ── NudeNet NSFW scanning ──────────────────────────────────────────────────

    def on_nsfw_scan(self):
        """Entry point: guard-check then open label selector."""
        if not self.folder:
            QMessageBox.warning(self, "No Folder", "Please select a folder first.")
            return
        if not self.image_paths:
            QMessageBox.warning(self, "Not Indexed", "Please index a folder first.")
            return
        if self.is_indexing:
            QMessageBox.warning(self, "Busy", "Please wait for indexing to complete.")
            return
        try:
            from nudenet import NudeDetector  # noqa: F401
        except ImportError:
            QMessageBox.critical(self, "Missing Dependency",
                "NudeNet is not installed.\n\nInstall with:\n    pip install nudenet")
            return
        self._open_nsfw_label_selector()

    def _open_nsfw_label_selector(self):
        """Dialog: choose which NudeNet labels to scan for."""
        dlg = QDialog(self)
        dlg.setWindowTitle("NSFW Scan — Select Labels")
        dlg.resize(460, 580)
        layout = QVBoxLayout(dlg)

        header = QLabel(
            "Select which NudeNet labels to detect.\n"
            "Images will be grouped and sorted by the labels found."
        )
        header.setStyleSheet(f"background-color: {PANEL_BG}; padding: 6px;")
        header.setWordWrap(True)
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(2)
        scroll.setWidget(content)
        layout.addWidget(scroll, stretch=1)

        checkboxes = {}
        for group_name, labels in NUDENET_LABEL_GROUPS.items():
            grp_lbl = QLabel(group_name)
            grp_lbl.setStyleSheet(
                f"color: {ACCENT}; font-weight: bold; font-size: 9pt; padding-top: 6px;")
            content_layout.addWidget(grp_lbl)
            # Default: check Explicit + Nudity; uncheck the rest
            default_checked = group_name in ("Explicit", "Nudity")
            for label in labels:
                cb = QCheckBox("  " + label.replace("_", " ").title())
                cb.setChecked(default_checked)
                content_layout.addWidget(cb)
                checkboxes[label] = cb
        content_layout.addStretch()

        # Select All / None row
        sel_row = QHBoxLayout()
        btn_all  = QPushButton("Select All")
        btn_none = QPushButton("Select None")
        btn_all.clicked.connect(lambda: [cb.setChecked(True)  for cb in checkboxes.values()])
        btn_none.clicked.connect(lambda: [cb.setChecked(False) for cb in checkboxes.values()])
        sel_row.addWidget(btn_all)
        sel_row.addWidget(btn_none)
        sel_row.addStretch()
        layout.addLayout(sel_row)

        btn_row = QHBoxLayout()
        btn_run    = QPushButton("Run Scan")
        btn_run.setProperty("class", "accent")
        btn_cancel = QPushButton("Cancel")
        btn_run.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        btn_row.addStretch()
        btn_row.addWidget(btn_run)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        selected = [lbl for lbl, cb in checkboxes.items() if cb.isChecked()]
        if not selected:
            QMessageBox.information(self, "No Labels Selected",
                "Please select at least one label to scan for.")
            return

        import threading
        threading.Thread(
            target=self._nsfw_scan_worker, args=(selected,), daemon=True
        ).start()

    def _nsfw_scan_worker(self, selected_labels):
        """Background thread: run NudeNet over all indexed images."""
        try:
            from nudenet import NudeDetector
            detector = NudeDetector()

            selected_set = set(selected_labels)
            total = len(self.image_paths)
            safe_print(f"[NSFW] Scanning {total:,} images for {len(selected_labels)} label(s)...")

            self._safe_after(0, lambda: self.progress.setRange(0, total))
            self._safe_after(0, lambda: self.progress.setValue(0))
            self._safe_after(0, lambda: self.update_status("Running NSFW scan...", "orange"))

            # results_by_label: {label: [(abs_path, score), ...]} sorted by score desc
            results_by_label = {lbl: [] for lbl in selected_labels}
            # all_detections: {abs_path: [{"class": ..., "score": ...}, ...]}
            all_detections = {}

            for i, abs_path in enumerate(self.image_paths):  # already absolute
                try:
                    detections = detector.detect(get_safe_path(abs_path))
                    matching = [d for d in detections if d["class"] in selected_set]
                    if matching:
                        all_detections[abs_path] = matching
                        for d in matching:
                            results_by_label[d["class"]].append((abs_path, float(d["score"])))
                except Exception as det_err:
                    safe_print(f"[NSFW] Skip {abs_path}: {det_err}")

                if (i + 1) % 5 == 0 or i == total - 1:
                    pct = i + 1
                    self._safe_after(0, lambda p=pct: self.progress.setValue(p))
                    self._safe_after(0, lambda p=pct, t=total:
                        self.progress_label.setText(f"NSFW scan: {p:,} / {t:,}"))

            # Sort each bucket highest confidence first
            for lbl in results_by_label:
                results_by_label[lbl].sort(key=lambda x: x[1], reverse=True)

            # Drop empty buckets
            results_by_label = {k: v for k, v in results_by_label.items() if v}

            n_flagged = len(all_detections)
            safe_print(f"[NSFW] Done. {n_flagged:,} image(s) flagged, "
                       f"{len(results_by_label)} label bucket(s).")

            self._safe_after(0, lambda: self.progress.setValue(0))
            self._safe_after(0, lambda: self.progress.setRange(0, 100))
            self._safe_after(0, lambda: self.progress_label.setText(""))
            self._safe_after(0, lambda: self.update_status(
                f"NSFW scan done: {n_flagged:,} image(s) flagged",
                "green" if n_flagged == 0 else "orange"))
            self._safe_after(0, lambda r=results_by_label, a=all_detections:
                self._open_nsfw_results_dialog(r, a))

        except Exception as e:
            safe_print(f"[NSFW] Error: {e}")
            import traceback; traceback.print_exc()
            self._safe_after(0, lambda: self.progress.setValue(0))
            self._safe_after(0, lambda: self.progress.setRange(0, 100))
            self._safe_after(0, lambda: self.progress_label.setText(""))
            self._safe_after(0, lambda: self.update_status("NSFW scan failed", "red"))

    def _open_nsfw_results_dialog(self, results_by_label, all_detections):
        """Show bucketed scan results grouped by label, sorted by confidence."""
        if not results_by_label:
            QMessageBox.information(self, "NSFW Scan",
                "No flagged images found for the selected labels.")
            return

        total_flagged = len(all_detections)

        dlg = QDialog(self)
        dlg.setWindowTitle(f"NSFW Scan Results — {total_flagged:,} image(s) flagged")
        dlg.resize(980, 720)
        layout = QVBoxLayout(dlg)

        header = QLabel(
            f"{total_flagged:,} image(s) flagged across {len(results_by_label)} label bucket(s). "
            "Buckets are ordered by severity. An image may appear in multiple buckets. "
            "Click a bucket's button to browse those images sorted by confidence score."
        )
        header.setWordWrap(True)
        header.setStyleSheet(f"background-color: {PANEL_BG}; padding: 6px;")
        layout.addWidget(header)

        # ── View All Flagged ──────────────────────────────────────────────────
        top_row = QHBoxLayout()
        btn_view_all = QPushButton(f"View All Flagged  ({total_flagged:,} images) →")
        btn_view_all.setProperty("class", "accent")

        def _view_all():
            scored = sorted(
                ((max(d["score"] for d in dets), path)
                 for path, dets in all_detections.items()),
                reverse=True,
            )
            self._nsfw_load_results(
                [(score, path, "image", {}) for score, path in scored],
                "All Flagged Images"
            )

        btn_view_all.clicked.connect(lambda _: _view_all())
        top_row.addStretch()
        top_row.addWidget(btn_view_all)
        layout.addLayout(top_row)

        # ── Label bucket grid ─────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(10)
        scroll.setWidget(grid_widget)
        layout.addWidget(scroll, stretch=1)

        THUMB_SIZE = (160, 160)
        COLS = 4

        # Build display order: iterate label groups so explicit appears first
        ordered_labels = [
            lbl
            for group_labels in NUDENET_LABEL_GROUPS.values()
            for lbl in group_labels
            if lbl in results_by_label
        ]

        for card_idx, label in enumerate(ordered_labels):
            entries = results_by_label[label]   # [(abs_path, score), ...]
            rep_path, rep_score = entries[0]     # highest-confidence image

            row, col = divmod(card_idx, COLS)

            card = QFrame()
            card.setStyleSheet(
                f"background-color: {CARD_BG}; border: 1px solid {BORDER};")
            card_layout = QVBoxLayout(card)
            card_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

            # Representative thumbnail
            try:
                img = Image.open(get_safe_path(rep_path))
                img.load()
                img.thumbnail(THUMB_SIZE)
                px = pil_to_pixmap(img)
                thumb = QLabel()
                thumb.setPixmap(px)
                thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
                card_layout.addWidget(thumb)
            except Exception:
                ph = QLabel("[preview unavailable]")
                ph.setFixedSize(160, 80)
                card_layout.addWidget(ph)

            title_lbl = QLabel(label.replace("_", " ").title())
            title_lbl.setStyleSheet(
                f"color: {ACCENT_SECONDARY}; font-weight: bold;"
                " font-size: 9pt; border: none;")
            title_lbl.setWordWrap(True)
            card_layout.addWidget(title_lbl)

            info_lbl = QLabel(
                f"{len(entries):,} image(s)  •  top score: {rep_score:.2f}")
            info_lbl.setStyleSheet("border: none; font-size: 8pt;")
            card_layout.addWidget(info_lbl)

            def _view_bucket(lbl=label, ents=entries):
                self._nsfw_load_results(
                    [(score, path, "image", {}) for path, score in ents],
                    lbl.replace("_", " ").title()
                )

            view_btn = QPushButton("View Images")
            view_btn.clicked.connect(lambda _, fn=_view_bucket: fn())
            card_layout.addWidget(view_btn)

            grid_layout.addWidget(card, row, col)

        # ── Close ─────────────────────────────────────────────────────────────
        close_row = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        close_row.addStretch()
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        dlg.exec()

    def _nsfw_load_results(self, album_results, title):
        """Load a scored result list into the main results grid."""
        self.cancel_search(clear_ui=True)
        self.all_search_results = album_results
        self.total_found = len(album_results)
        self.show_more_offset = 0
        self.update_status(f"{title}: {len(album_results):,} images", "green")
        vp_width = self.scroll_area.viewport().width()
        self.render_cols = max(1, max(vp_width, CELL_WIDTH) // CELL_WIDTH)
        initial_n = max(10, self.top_n_slider.value() * 10)
        first_batch = album_results[:initial_n]
        self.show_more_offset = len(first_batch)
        self.stop_search = False
        self.is_searching = True
        self.start_thumbnail_loader(first_batch, self.search_generation)


    # ── Feature: Face Recognition Presets ─────────────────────────────────────

    def _get_face_app(self):
        """Lazy-load InsightFace FaceAnalysis with ArcFace buffalo_l model."""
        if self._face_app is not None:
            return self._face_app
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise RuntimeError(
                "insightface is not installed.\n"
                "Install with:  pip install insightface onnxruntime\n"
                "(Use onnxruntime-gpu for faster processing on NVIDIA GPUs)"
            )
        app = FaceAnalysis(name="buffalo_l",
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        self._face_app = app
        return app

    def _face_index_path(self):
        return os.path.join(self.folder, ".face_index.pkl") if self.folder else None

    def _face_presets_path(self):
        return os.path.join(self.folder, ".face_presets.json") if self.folder else None

    def _load_face_data(self):
        """Load face index and presets from the current folder."""
        self.face_index = {}
        self.face_presets = {}
        fp = self._face_index_path()
        if fp and os.path.exists(fp):
            try:
                with open(fp, "rb") as f:
                    self.face_index = pickle.load(f)
                safe_print(f"[FACE] Loaded face index: {len(self.face_index)} images")
            except Exception as e:
                safe_print(f"[FACE] Failed to load face index: {e}")
                self.face_index = {}
        pp = self._face_presets_path()
        if pp and os.path.exists(pp):
            try:
                with open(pp, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.face_presets = {
                    name: {
                        "embedding": np.array(data["embedding"], dtype=np.float32),
                        "references": data.get("references", []),
                    }
                    for name, data in raw.items()
                }
                safe_print(f"[FACE] Loaded {len(self.face_presets)} face presets")
            except Exception as e:
                safe_print(f"[FACE] Failed to load presets: {e}")
                self.face_presets = {}

    def _save_face_index(self):
        fp = self._face_index_path()
        if not fp:
            return
        try:
            tmp = fp + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(self.face_index, f, protocol=pickle.HIGHEST_PROTOCOL)
            if os.path.exists(fp):
                os.remove(fp)
            os.rename(tmp, fp)
            safe_print(f"[FACE] Saved face index: {len(self.face_index)} images")
        except Exception as e:
            safe_print(f"[FACE] Failed to save face index: {e}")

    def _save_face_presets(self):
        pp = self._face_presets_path()
        if not pp:
            return
        try:
            serializable = {
                name: {
                    "embedding": data["embedding"].tolist(),
                    "references": data["references"],
                }
                for name, data in self.face_presets.items()
            }
            with open(pp, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
        except Exception as e:
            safe_print(f"[FACE] Failed to save presets: {e}")

    def on_face_presets(self):
        if not self.folder:
            QMessageBox.warning(self, "No Folder", "Please select and index a folder first.")
            return
        self.open_face_presets_dialog()

    def open_face_presets_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Face Presets — Person Recognition")
        dlg.resize(920, 620)
        main_layout = QVBoxLayout(dlg)

        # ── Face index status bar ──────────────────────────────────────────
        idx_bar = QHBoxLayout()
        n_indexed = len(self.face_index)
        n_total = len(self.image_paths)
        idx_status_lbl = QLabel(
            f"Face index: {n_indexed:,} / {n_total:,} images scanned"
            if n_indexed else "Face index not built yet — click 'Build Face Index' to start."
        )
        idx_bar.addWidget(idx_status_lbl)
        idx_bar.addStretch()
        build_btn = QPushButton("Build / Rebuild Face Index")
        build_btn.setToolTip(
            "Scans every indexed image with InsightFace ArcFace (buffalo_l).\n"
            "Downloads ~300 MB model on first run.\n"
            "Required before searching by person."
        )
        idx_bar.addWidget(build_btn)
        main_layout.addLayout(idx_bar)

        # ── Split: preset list (left) + reference photos (right) ──────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        # Left panel — preset list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(QLabel("Person Presets:"))
        preset_list = QListWidget()
        for name in sorted(self.face_presets.keys()):
            preset_list.addItem(name)
        left_layout.addWidget(preset_list, stretch=1)
        preset_btn_row = QHBoxLayout()
        add_preset_btn = QPushButton("New Preset")
        rename_preset_btn = QPushButton("Rename")
        del_preset_btn = QPushButton("Delete")
        del_preset_btn.setProperty("class", "danger")
        preset_btn_row.addWidget(add_preset_btn)
        preset_btn_row.addWidget(rename_preset_btn)
        preset_btn_row.addWidget(del_preset_btn)
        left_layout.addLayout(preset_btn_row)
        splitter.addWidget(left_widget)

        # Right panel — reference photos for selected preset
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_header = QHBoxLayout()
        right_header.addWidget(QLabel("Reference photos (select a preset to manage):"))
        right_header.addStretch()
        add_ref_btn = QPushButton("Add Reference Photo…")
        add_ref_btn.setEnabled(False)
        right_header.addWidget(add_ref_btn)
        right_layout.addLayout(right_header)
        ref_scroll = QScrollArea()
        ref_scroll.setWidgetResizable(True)
        ref_inner = QWidget()
        ref_grid = QGridLayout(ref_inner)
        ref_grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        ref_scroll.setWidget(ref_inner)
        right_layout.addWidget(ref_scroll, stretch=1)
        splitter.addWidget(right_widget)
        splitter.setSizes([280, 640])

        # ── Bottom search controls ─────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        main_layout.addWidget(sep)
        search_bar = QHBoxLayout()
        search_bar.addWidget(QLabel("Match threshold:"))
        thresh_slider = QSlider(Qt.Orientation.Horizontal)
        thresh_slider.setRange(20, 80)
        thresh_slider.setValue(45)
        thresh_lbl = QLabel("0.45")
        thresh_slider.valueChanged.connect(lambda v: thresh_lbl.setText(f"{v/100:.2f}"))
        thresh_slider.setToolTip(
            "Cosine similarity required to count as a match.\n"
            "0.45 is a good starting point — lower to catch more, raise to reduce false positives."
        )
        search_bar.addWidget(thresh_slider)
        search_bar.addWidget(thresh_lbl)
        search_bar.addStretch()
        search_btn = QPushButton("Search by Selected Person")
        search_btn.setProperty("class", "accent")
        search_btn.setEnabled(False)
        search_bar.addWidget(search_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        search_bar.addWidget(close_btn)
        main_layout.addLayout(search_bar)

        # ── Helpers ────────────────────────────────────────────────────────

        def _recompute_preset_embedding(name):
            """Re-extract face embeddings from all reference images and average them."""
            preset = self.face_presets.get(name)
            if preset is None:
                return
            embs = []
            try:
                app = self._get_face_app()
            except RuntimeError:
                return
            for ref_path in preset["references"]:
                abs_path = (ref_path if os.path.isabs(ref_path)
                            else os.path.join(self.folder, ref_path))
                if not os.path.exists(abs_path):
                    continue
                try:
                    pil_img = open_image(abs_path)
                    if pil_img is None:
                        continue
                    img_bgr = np.array(pil_img.convert("RGB"))[:, :, ::-1]
                    faces = app.get(img_bgr)
                    if faces:
                        largest = max(faces,
                                      key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                        embs.append(largest.embedding)
                except Exception as e:
                    safe_print(f"[FACE] Error re-extracting {ref_path}: {e}")
            if embs:
                avg = np.mean(embs, axis=0).astype(np.float32)
                norm = np.linalg.norm(avg)
                preset["embedding"] = avg / norm if norm > 0 else avg
            else:
                preset["embedding"] = np.zeros(512, dtype=np.float32)

        def _refresh_ref_panel(name):
            while ref_grid.count():
                item = ref_grid.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            if name not in self.face_presets:
                return
            refs = self.face_presets[name]["references"]
            if not refs:
                ref_grid.addWidget(
                    QLabel("No reference photos yet.\nClick 'Add Reference Photo…' to add some."), 0, 0)
                return
            for idx, ref_path in enumerate(refs):
                abs_path = (ref_path if os.path.isabs(ref_path)
                            else os.path.join(self.folder, ref_path))
                cell = QWidget()
                cell_layout = QVBoxLayout(cell)
                cell_layout.setContentsMargins(4, 4, 4, 4)
                pil_img = open_image(abs_path) if os.path.exists(abs_path) else None
                if pil_img:
                    pil_img.thumbnail((100, 100))
                    lbl = QLabel()
                    lbl.setPixmap(pil_to_pixmap(pil_img))
                    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    cell_layout.addWidget(lbl)
                else:
                    cell_layout.addWidget(QLabel("[missing]"))
                fname = os.path.basename(ref_path)
                if len(fname) > 16:
                    fname = fname[:13] + "..."
                name_lbl = QLabel(fname)
                name_lbl.setStyleSheet("font-size: 8pt;")
                name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell_layout.addWidget(name_lbl)
                rm_btn = QPushButton("Remove")
                rm_btn.setProperty("class", "danger")

                def _remove_ref(checked=False, i=idx, n=name):
                    self.face_presets[n]["references"].pop(i)
                    _recompute_preset_embedding(n)
                    self._save_face_presets()
                    _refresh_ref_panel(n)

                rm_btn.clicked.connect(_remove_ref)
                cell_layout.addWidget(rm_btn)
                ref_grid.addWidget(cell, idx // 4, idx % 4)

        def _on_preset_selected():
            item = preset_list.currentItem()
            has = item is not None
            add_ref_btn.setEnabled(has)
            search_btn.setEnabled(has and len(self.face_index) > 0)
            if has:
                _refresh_ref_panel(item.text())
            else:
                while ref_grid.count():
                    w = ref_grid.takeAt(0)
                    if w.widget():
                        w.widget().deleteLater()

        preset_list.currentItemChanged.connect(lambda *_: _on_preset_selected())

        # ── Preset management ──────────────────────────────────────────────

        def _add_preset():
            text, ok = QInputDialog.getText(dlg, "New Preset", "Person name:")
            text = text.strip()
            if not ok or not text:
                return
            if text in self.face_presets:
                QMessageBox.warning(dlg, "Already Exists", f"A preset named '{text}' already exists.")
                return
            self.face_presets[text] = {
                "embedding": np.zeros(512, dtype=np.float32),
                "references": [],
            }
            self._save_face_presets()
            preset_list.addItem(text)
            preset_list.setCurrentRow(preset_list.count() - 1)

        def _rename_preset():
            item = preset_list.currentItem()
            if not item:
                return
            old = item.text()
            text, ok = QInputDialog.getText(dlg, "Rename Preset", "New name:", text=old)
            text = text.strip()
            if not ok or not text or text == old:
                return
            if text in self.face_presets:
                QMessageBox.warning(dlg, "Already Exists", f"'{text}' already exists.")
                return
            self.face_presets[text] = self.face_presets.pop(old)
            self._save_face_presets()
            item.setText(text)

        def _delete_preset():
            item = preset_list.currentItem()
            if not item:
                return
            name = item.text()
            if QMessageBox.question(
                dlg, "Delete Preset", f"Delete preset '{name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) != QMessageBox.StandardButton.Yes:
                return
            del self.face_presets[name]
            self._save_face_presets()
            preset_list.takeItem(preset_list.row(item))
            _on_preset_selected()

        add_preset_btn.clicked.connect(_add_preset)
        rename_preset_btn.clicked.connect(_rename_preset)
        del_preset_btn.clicked.connect(_delete_preset)

        # ── Add reference photo ────────────────────────────────────────────

        def _pick_face_dialog(pil_img, faces, img_path):
            """Show cropped face thumbnails so the user can pick the right person."""
            pick_dlg = QDialog(dlg)
            pick_dlg.setWindowTitle("Multiple faces — select the correct person")
            pl = QVBoxLayout(pick_dlg)
            pl.addWidget(QLabel(
                f"{len(faces)} faces found in {os.path.basename(img_path)}.\n"
                "Click the correct person:"))
            face_row = QHBoxLayout()
            chosen = [None]
            img_arr = np.array(pil_img.convert("RGB"))
            for i, face in enumerate(faces):
                x1, y1, x2, y2 = [int(c) for c in face.bbox]
                pad = 20
                h, w = img_arr.shape[:2]
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                crop = Image.fromarray(img_arr[y1:y2, x1:x2])
                crop.thumbnail((120, 120))
                pixmap = pil_to_pixmap(crop)
                btn = QPushButton()
                btn.setIcon(QIcon(pixmap))
                btn.setIconSize(pixmap.size())
                btn.setFixedSize(pixmap.width() + 16, pixmap.height() + 16)
                btn.setToolTip(f"Face {i + 1}")

                def _pick(checked=False, idx=i):
                    chosen[0] = idx
                    pick_dlg.accept()

                btn.clicked.connect(_pick)
                face_row.addWidget(btn)
            pl.addLayout(face_row)
            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(pick_dlg.reject)
            pl.addWidget(cancel_btn)
            pick_dlg.exec()
            return chosen[0]

        def _add_reference():
            item = preset_list.currentItem()
            if not item:
                return
            preset_name = item.text()
            paths, _ = QFileDialog.getOpenFileNames(
                dlg, "Select Reference Photo(s)", self.folder or "",
                "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tiff *.tif)")
            if not paths:
                return
            added, errors = 0, []
            for path in paths:
                try:
                    app = self._get_face_app()
                    pil_img = open_image(path)
                    if pil_img is None:
                        errors.append(f"{os.path.basename(path)}: could not open")
                        continue
                    img_bgr = np.array(pil_img.convert("RGB"))[:, :, ::-1]
                    faces = app.get(img_bgr)
                    if not faces:
                        errors.append(f"{os.path.basename(path)}: no face detected")
                        continue
                    if len(faces) > 1:
                        face_idx = _pick_face_dialog(pil_img, faces, path)
                        if face_idx is None:
                            continue
                    else:
                        face_idx = 0
                    # Store absolute path for multi-folder compatibility
                    self.face_presets[preset_name]["references"].append(path)
                    added += 1
                except RuntimeError as e:
                    QMessageBox.critical(dlg, "Missing Dependency", str(e))
                    return
                except Exception as e:
                    errors.append(f"{os.path.basename(path)}: {e}")
            if added:
                _recompute_preset_embedding(preset_name)
                self._save_face_presets()
                _refresh_ref_panel(preset_name)
            if errors:
                QMessageBox.warning(dlg, "Some photos skipped",
                    "\n".join(errors[:10]) + ("\n…" if len(errors) > 10 else ""))

        add_ref_btn.clicked.connect(_add_reference)

        # ── Build face index ───────────────────────────────────────────────

        def _start_build():
            if not self.image_paths:
                QMessageBox.warning(dlg, "Not Ready", "No images indexed yet.")
                return
            build_btn.setEnabled(False)
            build_btn.setText("Building…")
            idx_status_lbl.setText("Initialising InsightFace model…")
            Thread(target=_build_worker, daemon=True).start()

        def _build_worker():
            try:
                app = self._get_face_app()
            except RuntimeError as e:
                self._safe_after(0, lambda: QMessageBox.critical(dlg, "Missing Dependency", str(e)))
                self._safe_after(0, lambda: build_btn.setEnabled(True))
                self._safe_after(0, lambda: build_btn.setText("Build / Rebuild Face Index"))
                return
            n = len(self.image_paths)
            face_index = {}
            self._safe_after(0, lambda: self.progress.setRange(0, n))
            for i, abs_path in enumerate(self.image_paths):  # already absolute
                if self.stop_search:
                    break
                try:
                    pil_img = open_image(abs_path)
                    if pil_img is None:
                        continue
                    img_bgr = np.array(pil_img.convert("RGB"))[:, :, ::-1]
                    faces = app.get(img_bgr)
                    if faces:
                        face_index[abs_path] = [f.embedding for f in faces]
                except Exception as e:
                    safe_print(f"[FACE] Error on {abs_path}: {e}")
                if i % 20 == 0:
                    pct = i + 1
                    self._safe_after(0, lambda v=pct: self.progress.setValue(v))
                    self._safe_after(0, lambda v=pct, tot=n: idx_status_lbl.setText(
                        f"Scanning… {v:,} / {tot:,}"))
            self.face_index = face_index
            self._save_face_index()
            n_faces = sum(len(v) for v in face_index.values())
            msg = f"Face index: {len(face_index):,} images, {n_faces:,} faces detected"
            self._safe_after(0, lambda: self.progress.setRange(0, 100))
            self._safe_after(0, lambda: self.progress.setValue(100))
            self._safe_after(0, lambda: idx_status_lbl.setText(msg))
            self._safe_after(0, lambda: build_btn.setEnabled(True))
            self._safe_after(0, lambda: build_btn.setText("Build / Rebuild Face Index"))
            self._safe_after(0, _on_preset_selected)
            safe_print(f"[FACE] {msg}")

        build_btn.clicked.connect(_start_build)

        # ── Search by person ───────────────────────────────────────────────

        def _do_search():
            item = preset_list.currentItem()
            if not item:
                return
            preset_name = item.text()
            threshold = thresh_slider.value() / 100.0
            if not self.face_index:
                QMessageBox.warning(dlg, "No Face Index",
                    "Build the face index first.")
                return
            preset_emb = self.face_presets[preset_name]["embedding"]
            if np.all(preset_emb == 0):
                QMessageBox.warning(dlg, "No References",
                    "Add at least one reference photo to this preset first.")
                return
            dlg.accept()
            self.update_status(f"Searching for '{preset_name}'…", "orange")
            Thread(
                target=lambda: self._face_search_worker(preset_name, preset_emb, threshold),
                daemon=True,
            ).start()

        search_btn.clicked.connect(_do_search)
        _on_preset_selected()
        dlg.exec()

    def _face_search_worker(self, preset_name, preset_emb, threshold):
        try:
            results = []
            norm_preset = preset_emb / (np.linalg.norm(preset_emb) + 1e-8)
            for stored_path, face_embs in self.face_index.items():
                best_sim = max(
                    float(np.dot(norm_preset, fe / (np.linalg.norm(fe) + 1e-8)))
                    for fe in face_embs
                )
                if best_sim >= threshold:
                    # stored_path may be absolute (new) or relative (old face index)
                    if os.path.isabs(stored_path):
                        abs_path = stored_path
                    else:
                        abs_path = os.path.join(self.folder, stored_path)
                    results.append((best_sim, abs_path, "image", {}))
            results.sort(key=lambda x: x[0], reverse=True)
            if not results:
                self._safe_after(0, lambda: self.update_status(
                    f"No matches for '{preset_name}'", "orange"))
                self._safe_after(0, lambda: QMessageBox.information(
                    self, "No Matches",
                    f"No images matched '{preset_name}' above threshold {threshold:.2f}.\n"
                    "Try lowering the threshold or adding more reference photos."))
                return
            title = f"Face: {preset_name}"
            self._safe_after(0, lambda r=results, t=title: self._nsfw_load_results(r, t))
        except Exception as e:
            safe_print(f"[FACE] Search error: {e}")
            import traceback; traceback.print_exc()
            self._safe_after(0, lambda: self.update_status("Face search failed", "red"))


if __name__ == "__main__":
    print("=" * 60)
    print("PhotoSearchPro - AI Media Search (Cross-Platform GPU Accelerated)")
    print("=" * 60)

    # Ensure Qt uses its own platform plugins.
    # opencv-python and other packages can corrupt QT_PLUGIN_PATH or load
    # conflicting Qt DLLs, causing qwindows.dll to fail even though it's found.
    # Pinning the path to PyQt6's own plugins directory fixes this.
    import PyQt6 as _pyqt6_pkg
    _qt_plugin_dir = os.path.join(os.path.dirname(_pyqt6_pkg.__file__), 'Qt6', 'plugins')
    if os.path.isdir(_qt_plugin_dir):
        os.environ['QT_PLUGIN_PATH'] = _qt_plugin_dir
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(_qt_plugin_dir, 'platforms')

    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # consistent cross-platform look

    # Force dark palette so Fusion-style popup windows (e.g. combobox
    # dropdowns) use dark colors instead of the system default white.
    _pal = QPalette()
    _pal.setColor(QPalette.ColorRole.Window,          QColor(BG))
    _pal.setColor(QPalette.ColorRole.WindowText,      QColor(FG))
    _pal.setColor(QPalette.ColorRole.Base,            QColor(PANEL_BG))
    _pal.setColor(QPalette.ColorRole.AlternateBase,   QColor(CARD_BG))
    _pal.setColor(QPalette.ColorRole.Text,            QColor(FG))
    _pal.setColor(QPalette.ColorRole.Button,          QColor(CARD_BG))
    _pal.setColor(QPalette.ColorRole.ButtonText,      QColor(FG))
    _pal.setColor(QPalette.ColorRole.Highlight,       QColor(ACCENT_SECONDARY))
    _pal.setColor(QPalette.ColorRole.HighlightedText, QColor(FG))
    _pal.setColor(QPalette.ColorRole.ToolTipBase,     QColor(PANEL_BG))
    _pal.setColor(QPalette.ColorRole.ToolTipText,     QColor(FG))
    app.setPalette(_pal)

    # Apply dark QSS theme
    app.setStyleSheet(DARK_QSS)

    window = ImageSearchApp()
    window.show()
    sys.exit(app.exec())

