# PhotoSearchPro

AI-powered desktop app to search your image and video library using text descriptions or a reference image. No cloud, no filenames needed — just describe what you're looking for.

![PhotoSearchPro Demo](demo.gif)

![PhotoSearchPro Screenshot](Demo-pic.png)

---

## Requirements

- Python 3.10, 3.11, or 3.12
- Windows, macOS, or Linux
- NVIDIA GPU recommended (AMD and CPU-only also supported)

---

## Installation

### Windows (Easy)

1. Clone the repo: `git clone https://github.com/Ubaida-M-Yusuf/Makimus-AI.git`
2. Double-click `Install.bat` — sets up everything and launches the app
3. Use `Run.bat` to launch in the future

> `Install.bat` installs CUDA 12.8 PyTorch. AMD or CPU-only users should follow the manual steps below.

---

### Manual Installation

**1. Clone the repo:**
```bash
git clone https://github.com/Ubaida-M-Yusuf/Makimus-AI.git
cd Makimus-AI
```

**2. Linux only — install system dependencies:**
```bash
sudo apt install python3-tk python3-pip libgl1 libglib2.0-0 ffmpeg
```

**3. Create and activate a virtual environment:**
```bash
python -m venv venv

# Windows (Command Prompt)
venv\Scripts\activate

# Windows (Git Bash) / macOS / Linux
source venv/bin/activate
```

**4. Install dependencies for your GPU:**

| GPU | Command |
|-----|---------|
| NVIDIA (RTX 20xx/30xx/40xx/50xx) | `pip install -r requirements.txt` |
| NVIDIA (GTX 900 series or older) | See note below |
| Apple Silicon (M1/M2/M3) | `pip install Pillow==12.0.0 numpy==2.2.6 open_clip_torch==3.2.0 rawpy==0.26.1 opencv-python==4.13.0.92 Send2Trash==2.1.0 tkinterdnd2 && pip install torch==2.10.0 torchvision==0.25.0` |
| AMD GPU — Windows | Same as Apple Silicon, then `pip install torch-directml` |
| CPU only | Same as Apple Silicon |

> **Older NVIDIA GPUs (GTX 900 series and earlier):**
> ```bash
> pip install Pillow==12.0.0 numpy==2.2.6 open_clip_torch==3.2.0 rawpy==0.26.1 opencv-python==4.13.0.92 Send2Trash==2.1.0 tkinterdnd2
> pip install torch==2.9.1+cu126 torchvision==0.24.1+cu126 --index-url https://download.pytorch.org/whl/cu126
> ```

> **AMD GPU — Linux (ROCm):** Visit [pytorch.org](https://pytorch.org/get-started/locally) and select ROCm. Only RX 6000/7000 series are well supported. Falls back to CPU if setup fails.

**5. Run:**
```bash
python PhotoSearchPro.py
```

---

## First Launch

1. The app downloads the AI model (~1.7 GB) automatically — this happens once and is cached.
2. Click **Folders** and select a folder containing your images or videos.
3. A prompt will ask if you want to index it — click **Yes**.
4. Indexing runs in the background. You can start searching before it finishes.

---

## Features

### Text Search

Type a description in the search bar and press Enter or click **Search**.

- Describe the content, not the filename — *"sunset over ocean"*, *"person in blue jacket"*, *"fight scene"*
- **Negative terms:** Add a minus sign before a word to exclude it — `cat -dog` finds cats without dogs. Set the Similarity Score to 0.10 or lower when using negative terms.

---

### Image Search

Find images/videos that look similar to a reference image.

- Click the **Image** button and pick a file
- Or **drag and drop** any image directly onto the results area

> Drag and drop requires `tkinterdnd2`. If it's not working: `pip install tkinterdnd2==0.4.3`

---

### Video Search

Works the same as image search — text or image queries return matching video frames with timestamps (e.g. `t=1:23`). Double-click a result to open the video in your default player.

**Best Frame mode:** When enabled, returns the single best-matching frame from each video instead of the nearest sampled frame. Toggle it in the interface. Useful when you want one clean result per video rather than multiple frames.

---

### Hybrid Search

Blend a text query with a reference image using the anchor slider. Use the **Image** button to set the anchor image, type a text query, then adjust the slider to control how much weight each gets.

---

### Similarity Score Slider

Controls how strict the matching is. Lower = more results, higher = only close matches.

---

### Images / Videos Toggle

Filter results to show only images or only videos.

---

### Selecting Results

- **Single click** — opens the file's folder in Explorer / Finder / Dolphin
- **Double click** — opens the file in your default viewer or player
- **Right-click** — context menu: open, show in folder, copy, move, delete, select/deselect
- **Click and drag on empty space** — rubber-band select multiple files at once

---

### File Operations

Select one or more files, then:

| Action | How |
|--------|-----|
| Export (copy) | Click **Export** and choose a destination |
| Move | Click **Move** and choose a destination |
| Delete | Right-click → Delete (goes to Recycle Bin, recoverable) |

Your selection stays active after an operation so you can copy and move without reselecting.

---

### Managing the Index

- **Refresh** — scans for new or removed files and only processes what changed. No full re-index needed.
- The cache file is stored inside your media folder and loads automatically next time.
- Moving or renaming the indexed folder does not break the cache.
- If any files fail to process, a log is written to `photosearchpro_skipped_images.txt` or `photosearchpro_skipped_videos.txt` inside the indexed folder.

---

### Multiple Folders

Click **Folders** to add multiple folders. All are searched simultaneously.

---

### Folder Exclusions

Exclude specific subfolders from indexing and search using pattern-based rules. Accessible from the interface.

---

### Model Selection

Choose from four AI models with different trade-offs between speed and accuracy:

| Model | Notes |
|-------|-------|
| CLIP ViT-L-14 | Default. Best general-purpose performance. |
| SigLIP2 | Strong text-image alignment. |
| DINOv2 | Visual similarity focused, no text search. |
| DINOv3 | Updated DINO variant. |

Change models via the model selector in the UI. Each model uses a separate cache — switching models requires re-indexing.

---

### Duplicates Detection

Find perceptually similar images in your collection. Access it from the tools menu.

---

### Smart Albums

Automatically groups your images into semantic categories. Access it from the tools menu.

---

### NSFW Scan *(optional)*

Scans your indexed images for explicit content using NudeNet.

**Install NudeNet:**
```bash
pip install nudenet
```

Then use the NSFW Scan option in the tools menu.

---

### Face Presets *(optional)*

Create named presets for people in your collection. The app uses InsightFace to find photos of that person across your library.

**Install:**
```bash
pip install insightface onnxruntime
```

Then access Face Presets from the tools menu. Add a preset by providing a name and a sample photo of that person.

---

## Supported File Formats

| Type | Formats |
|------|---------|
| Images | JPG, JPEG, PNG, WEBP, BMP, GIF |
| RAW photos | CR2, NEF, ARW, DNG, ORF, RW2, RAF, PEF, SR2 |
| Videos | MP4, MKV, MOV, AVI, WEBM, M4V, WMV, FLV, TS, MPG, MPEG, 3GP, VOB |

---

## Updating

```bash
git pull
pip uninstall torch torchvision -y
pip install -r requirements.txt
```

> Existing cache files are fully compatible — no re-indexing needed after an update.

---

## GPU Support

| Platform | Backend |
|----------|---------|
| NVIDIA | CUDA (12.8 or 12.6) |
| Apple Silicon | MPS (built-in) |
| AMD — Windows | DirectML |
| AMD — Linux | ROCm (manual setup required) |
| No GPU | CPU (works, but significantly slower) |

The app detects your hardware on startup and auto-tunes batch sizes based on available VRAM.

---

## HuggingFace Token

Some models require a HuggingFace account token. Enter it via the settings in the UI. The token is saved locally and used automatically on future launches.

---

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — free for personal, non-commercial use.

---

## Support

A pre-packaged Windows installer (no Python required) is available on [Payhip](https://payhip.com/b/ZUEPu).

Donations: [PayPal](https://www.paypal.me/UbaidahYusuf) | BTC: `32Ztk8P3zMavx51usno2adnDiZzLmZL42y` | USDT TRC-20: `TYwaZ2xNqHUYDXH83teMSNtWfYETiBAEDk`
