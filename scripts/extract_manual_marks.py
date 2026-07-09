#!/usr/bin/env python3
"""
Extract 64x64 training crops from manual marks saved by qa_gui.py.

Reads *_qa_marks.json files, opens the original map, applies the same
upscale + red-suppression pipeline used during detection, and saves
64x64 crops at each manual mark coordinate.

Crops go to training_data/positive/ (manual marks = confirmed triangles).

Usage:
    python extract_manual_marks.py            # all JSONs in Map_Scans sheet folders
    python extract_manual_marks.py ../Map_Scans/JPG_from_TIFF/14-15-Lydda-1942_qa_marks.json
    python extract_manual_marks.py --dry-run               # just count, don't save
"""

import sys
import json
import numpy as np
import cv2
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from image_loader import load_image, suppress_red


def extract_crops_from_marks(json_path, output_dir, dry_run=False):
    """Extract 64x64 crops from manual marks in a QA marks JSON file."""
    json_path = Path(json_path)
    data = json.loads(json_path.read_text())

    map_name = data['map']
    manual_marks = data.get('manual_marks', [])

    if not manual_marks:
        print(f"  {map_name}: no manual marks, skipping")
        return 0

    # Find the map image
    map_file = Path(data['file'])
    if not map_file.exists():
        # Try relative to json location
        map_file = json_path.parent / f"{map_name}.jpg"
    if not map_file.exists():
        # Try common extensions
        for ext in ('.jpg', '.jpeg', '.tif', '.tiff', '.png'):
            candidate = json_path.parent / f"{map_name}{ext}"
            if candidate.exists():
                map_file = candidate
                break
    if not map_file.exists():
        print(f"  {map_name}: map image not found at {data['file']}")
        return 0

    print(f"\n  {map_name}: {len(manual_marks)} manual marks")
    print(f"  Image: {map_file}")

    if dry_run:
        return len(manual_marks)

    # Load and upscale (same pipeline as qa_detections.py)
    img = load_image(str(map_file))
    h, w = img.shape[:2]

    scale = max(1.0, 14000.0 / w)
    scale = round(scale * 2) / 2
    if scale > 1.0:
        img_up = cv2.resize(img, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_LINEAR)
        print(f"  Upscaled {w}x{h} -> {img_up.shape[1]}x{img_up.shape[0]} ({scale}x)")
    else:
        img_up = img
        scale = 1.0
        print(f"  {w}x{h} (no upscale)")

    prep = suppress_red(img_up)
    h_up, w_up = prep.shape[:2]
    half = 32

    saved = 0
    skipped = 0
    for mark in manual_marks:
        # Manual marks are in original-image coordinates; convert to upscaled
        mx = int(mark['x'] * scale)
        my = int(mark['y'] * scale)

        # Bounds check
        if mx - half < 0 or my - half < 0 or mx + half >= w_up or my + half >= h_up:
            skipped += 1
            continue

        crop = prep[my - half:my + half, mx - half:mx + half]
        if crop.shape != (64, 64):
            skipped += 1
            continue

        # Save as BGR (consistent with other training data)
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        fname = f"{map_name}_x{mark['x']}_y{mark['y']}_manual.png"
        dest = output_dir / fname
        if not dest.exists():
            cv2.imwrite(str(dest), crop_bgr)
            saved += 1
        else:
            skipped += 1

    print(f"  Saved: {saved}, skipped: {skipped}")
    return saved


def main():
    dry_run = '--dry-run' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--dry-run']

    # Collect JSON files
    json_files = []
    if args:
        for arg in args:
            p = Path(arg)
            if p.is_file() and p.suffix == '.json':
                json_files.append(p)
            elif p.is_dir():
                json_files.extend(sorted(p.glob('*_qa_marks.json')))
    else:
        # Default: search the flat sheet-scan folders
        from data_paths import sheet_image_dirs
        seen = set()
        for maps_dir in sheet_image_dirs():
                for jf in sorted(maps_dir.glob('*_qa_marks.json')):
                    real = jf.resolve()
                    if real not in seen:
                        seen.add(real)
                        json_files.append(jf)

    if not json_files:
        print("No *_qa_marks.json files found.")
        print("Usage: python extract_manual_marks.py [json_file ...] [directory]")
        sys.exit(1)

    output_dir = BASE_DIR / 'training_data' / 'positive'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'[DRY RUN] ' if dry_run else ''}Processing {len(json_files)} QA marks file(s)")
    print(f"Output: {output_dir}")

    total = 0
    for jf in json_files:
        n = extract_crops_from_marks(jf, output_dir, dry_run=dry_run)
        total += n

    print(f"\n{'='*50}")
    print(f"Total crops {'(would be) ' if dry_run else ''}saved: {total}")
    if not dry_run and total > 0:
        print(f"\nNext steps:")
        print(f"  python apply_curate_labels.py --source original   # rebuild train/val")
        print(f"  python train_classifier.py")


if __name__ == '__main__':
    main()
