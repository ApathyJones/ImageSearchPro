"""
migrate_models.py
-----------------
Moves previously-downloaded model files from the old default cache locations
(~/.cache/...) into the app's own  models/  folder so nothing has to be
re-downloaded after the cache-redirect update.

Run from any working directory — the script always writes into the  models/
folder that lives next to itself.
"""

import os
import sys
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DIR   = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"

HOME_CACHE = Path.home() / ".cache"

MIGRATIONS = [
    # (source in ~/.cache,            destination in models/)
    ("huggingface/hub",               "huggingface/hub"),
    ("huggingface/accelerate",        "huggingface/accelerate"),
    ("open_clip",                     "huggingface/hub"),   # some open_clip builds use this
    ("onnx_clip",                     "onnx_clip"),
    ("torch/hub",                     "torch/hub"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_size(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


def merge_move(src: Path, dst: Path):
    """Move src into dst, merging if dst already exists."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                merge_move(item, target)       # recurse to merge
                item.rmdir() if not any(item.iterdir()) else None
            else:
                shutil.move(str(item), str(target))
        else:
            if not target.exists():
                shutil.move(str(item), str(target))
            else:
                print(f"  [SKIP]  {item.name}  (already exists in destination)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  ImageSearchPro — Model Cache Migration")
    print("=" * 60)
    print(f"  App folder : {APP_DIR}")
    print(f"  Models dir : {MODELS_DIR}")
    print(f"  Old cache  : {HOME_CACHE}")
    print()

    found = []
    for rel_src, rel_dst in MIGRATIONS:
        src = HOME_CACHE / rel_src
        if src.exists() and any(src.iterdir()):
            dst = MODELS_DIR / rel_dst
            found.append((src, dst, rel_src))

    if not found:
        print("  Nothing to migrate — no model files found in ~/.cache.")
        print("  (They may already be in the models/ folder, or not yet downloaded.)")
        _pause()
        return

    print("  The following folders will be MOVED:\n")
    for src, dst, _ in found:
        print(f"    FROM  {src}")
        print(f"    TO    {dst}")
        print(f"    SIZE  {fmt_size(src)}")
        print()

    ans = input("  Proceed? [y/N] ").strip().lower()
    if ans != "y":
        print("  Cancelled.")
        _pause()
        return

    print()
    ok = 0
    for src, dst, label in found:
        try:
            print(f"  Moving {label} ...", end=" ", flush=True)
            merge_move(src, dst)
            # Remove the (now-empty) source directory
            try:
                src.rmdir()
            except OSError:
                pass  # not empty — some files were skipped
            print("done")
            ok += 1
        except Exception as e:
            print(f"FAILED\n    {e}")

    print()
    print(f"  Migration complete ({ok}/{len(found)} folders moved).")
    print()
    print("  You can now delete any leftover empty folders under ~/.cache manually.")
    _pause()


def _pause():
    if os.name == "nt":
        input("\n  Press Enter to close...")


if __name__ == "__main__":
    main()
