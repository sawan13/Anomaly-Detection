"""
Organize image files: delete unwanted files and move images into `images/` folder.
Run: python organize_images.py
This script will:
 - delete `violinplot_features_distribution.png` (if present)
 - delete `TEAM_IMAGE_ANALYSIS.txt` (if present)
 - create `images/` in the project root (if missing)
 - move all image files (*.png, *.jpg, *.jpeg, *.pdf) into `images/` (flattened)
It will avoid moving itself and will resolve name conflicts by appending a counter.
"""
import os
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
IMAGES_DIR = BASE / 'images'
UNWANTED = [
    BASE / 'violinplot_features_distribution.png',
    BASE / 'TEAM_IMAGE_ANALYSIS.txt',
]
PATTERNS = ['*.png', '*.jpg', '*.jpeg', '*.pdf']

def delete_unwanted():
    for p in UNWANTED:
        try:
            if p.exists():
                if p.is_file():
                    p.unlink()
                    print(f"Deleted unwanted: {p}")
        except Exception as e:
            print(f"Failed to delete {p}: {e}")

def ensure_images_dir():
    try:
        IMAGES_DIR.mkdir(exist_ok=True)
        print(f"Images folder: {IMAGES_DIR}")
    except Exception as e:
        print(f"Failed to create images dir: {e}")

def move_images():
    moved = []
    for root, dirs, files in os.walk(BASE):
        rootp = Path(root)
        # skip the images folder itself
        if rootp == IMAGES_DIR:
            continue
        for pattern in PATTERNS:
            for p in rootp.glob(pattern):
                # skip directories
                if not p.is_file():
                    continue
                # skip the script itself
                if p.resolve() == (BASE / 'organize_images.py').resolve():
                    continue
                # skip files already targeted for deletion
                if p in UNWANTED:
                    continue
                # destination
                dest = IMAGES_DIR / p.name
                if dest.exists():
                    # find a non-conflicting name
                    stem = p.stem
                    suffix = p.suffix
                    i = 1
                    while True:
                        candidate = IMAGES_DIR / f"{stem}_{i}{suffix}"
                        if not candidate.exists():
                            dest = candidate
                            break
                        i += 1
                try:
                    shutil.move(str(p), str(dest))
                    moved.append((p, dest))
                except Exception as e:
                    print(f"Failed to move {p} -> {dest}: {e}")
    for s,d in moved:
        print(f"Moved: {s} -> {d}")

if __name__ == '__main__':
    print(f"Base: {BASE}")
    delete_unwanted()
    ensure_images_dir()
    move_images()
    print('Done.')
