#!/usr/bin/env python3
"""
Extract 64x64 crops from QA detections for curation and CNN retraining.

Reuses the same pipeline as qa_detections.py (upscale, template match, CNN)
but saves crops to disk instead of just drawing circles.

Usage:
    python extract_qa_crops.py ../all_maps/
    python extract_qa_crops.py --cnn-threshold 0.70 ../all_maps/
    python extract_qa_crops.py --save-negatives ../all_maps/

Output:
    training_data/qa_candidates/       — crops with CNN >= threshold
    training_data/qa_hard_negatives/   — crops with 0.40 <= CNN < threshold
                                         (if --save-negatives)

Crop naming: <mapname>_x<col>_y<row>_c<cnn_prob>.png
"""
import sys
import time
import numpy as np
import cv2
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from image_loader import load_image, suppress_red
from db_matcher import load_grayscale_templates
from train_classifier import TriangleCNN
from grid_label_ocr import find_neatline
from qa_detections import (
    load_cnn_model,
    find_template_candidates,
    classify_crops_chunked,
    grid_nms,
)


def extract_and_classify(prep_gray, candidates, model, max_candidates=None):
    """Like classify_candidates but returns crops alongside results.

    max_candidates=None classifies every candidate (see classify_candidates
    in qa_detections.py — a top-N cap silently drops real triangles on
    dense maps).
    """
    h, w = prep_gray.shape[:2]
    half = 32

    candidates = sorted(candidates, key=lambda c: c[2], reverse=True)
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    crops = []
    valid_candidates = []
    for x, y, tconf, tname in candidates:
        if x - half < 0 or y - half < 0 or x + half >= w or y + half >= h:
            continue
        crop = prep_gray[y - half:y + half, x - half:x + half]
        crops.append(crop)
        valid_candidates.append((x, y, tconf, tname))

    if not crops:
        return [], []

    probs = classify_crops_chunked(model, crops)

    results = []
    for i, (x, y, tconf, tname) in enumerate(valid_candidates):
        results.append((x, y, tconf, float(probs[i])))

    return results, crops


def process_map_crops(image_path, templates, model, output_dir, neg_dir=None,
                      cnn_threshold=0.70, tmpl_threshold=0.55):
    """Extract and save crops from a single map."""
    image_path = Path(image_path)
    map_name = image_path.stem
    print(f"\n{'='*60}")
    print(f"  {map_name}")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()
    img = load_image(str(image_path))
    h_img, w_img = img.shape[:2]

    # Upscale to match template training resolution
    scale = max(1.0, 14000.0 / w_img)
    scale = round(scale * 2) / 2
    if scale > 1.0:
        img_up = cv2.resize(img, (int(w_img * scale), int(h_img * scale)),
                            interpolation=cv2.INTER_LINEAR)
        print(f"  {w_img}x{h_img} -> {img_up.shape[1]}x{img_up.shape[0]} ({scale}x)")
    else:
        img_up = img
        print(f"  {w_img}x{h_img} (no upscale)")
    sys.stdout.flush()

    prep = suppress_red(img_up)

    # Find neatline on original image
    neatline = find_neatline(img)

    # Template matching
    t1 = time.time()
    candidates = find_template_candidates(prep, templates, threshold=tmpl_threshold)
    print(f"  Template candidates: {len(candidates)} ({time.time() - t1:.1f}s)")
    sys.stdout.flush()

    # CNN classification — get crops back
    t1 = time.time()
    results, crops = extract_and_classify(prep, candidates, model)
    print(f"  CNN classified: {len(results)} ({time.time() - t1:.1f}s)")
    sys.stdout.flush()

    # Filter by neatline (in upscaled coords)
    if neatline:
        margin = int(30 * scale)
        nl_left = int(neatline['left'] * scale)
        nl_right = int(neatline['right'] * scale)
        nl_top = int(neatline['top'] * scale)
        nl_bottom = int(neatline['bottom'] * scale)

        filtered_results = []
        filtered_crops = []
        for i, (x, y, tconf, cnn_prob) in enumerate(results):
            if (nl_left - margin <= x <= nl_right + margin and
                    nl_top - margin <= y <= nl_bottom + margin):
                filtered_results.append(results[i])
                filtered_crops.append(crops[i])
        before = len(results)
        results = filtered_results
        crops = filtered_crops
        if before > len(results):
            print(f"  Neatline filter: {before} -> {len(results)}")

    # Save crops
    n_pos = 0
    n_neg = 0
    for i, (x, y, tconf, cnn_prob) in enumerate(results):
        crop_bgr = cv2.cvtColor(crops[i], cv2.COLOR_GRAY2BGR)
        fname = f"{map_name}_x{int(x)}_y{int(y)}_c{cnn_prob:.2f}.png"

        if cnn_prob >= cnn_threshold:
            cv2.imwrite(str(output_dir / fname), crop_bgr)
            n_pos += 1
        elif neg_dir is not None and cnn_prob >= 0.40:
            cv2.imwrite(str(neg_dir / fname), crop_bgr)
            n_neg += 1

    elapsed = time.time() - t0
    print(f"  Saved: {n_pos} candidates (>={cnn_threshold}), "
          f"{n_neg} hard negatives (0.40-{cnn_threshold})")
    print(f"  {elapsed:.1f}s")

    return n_pos, n_neg


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_qa_crops.py [--cnn-threshold 0.70] "
              "[--save-negatives] <image ...> | <directory>")
        sys.exit(1)

    cnn_threshold = 0.70
    save_negatives = False
    args = sys.argv[1:]

    if '--cnn-threshold' in args:
        idx = args.index('--cnn-threshold')
        cnn_threshold = float(args[idx + 1])
        args = args[:idx] + args[idx + 2:]
    if '--save-negatives' in args:
        save_negatives = True
        args.remove('--save-negatives')

    # Collect image paths
    image_paths = []
    for arg in args:
        p = Path(arg)
        if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.tif', '.tiff', '.png'):
            image_paths.append(p)
        elif p.is_dir():
            for ext in ('*.jpg', '*.jpeg', '*.tif', '*.tiff'):
                image_paths.extend(sorted(p.glob(ext)))

    if not image_paths:
        print("No image files found.")
        sys.exit(1)

    # Setup output directories
    output_dir = BASE_DIR / 'training_data' / 'qa_candidates'
    output_dir.mkdir(parents=True, exist_ok=True)
    neg_dir = None
    if save_negatives:
        neg_dir = BASE_DIR / 'training_data' / 'qa_hard_negatives'
        neg_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(image_paths)} maps, cnn_threshold={cnn_threshold}")
    print(f"  Output: {output_dir}")
    if neg_dir:
        print(f"  Hard negatives: {neg_dir}")

    # Load shared resources
    templates = load_grayscale_templates(SCRIPT_DIR / 'templates')
    print(f"  {len(templates)} templates")
    model = load_cnn_model()

    total_pos = 0
    total_neg = 0
    for img_path in image_paths:
        n_pos, n_neg = process_map_crops(
            img_path, templates, model, output_dir, neg_dir,
            cnn_threshold=cnn_threshold,
        )
        total_pos += n_pos
        total_neg += n_neg

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")
    print(f"  Total candidates saved: {total_pos}")
    if neg_dir:
        print(f"  Total hard negatives saved: {total_neg}")
    print(f"\nNext steps:")
    print(f"  1. Curate:  python curate.py --dir {output_dir}")
    print(f"  2. Apply:   python apply_curate_labels.py --source qa")
    print(f"  3. Retrain: python train_classifier.py")


if __name__ == '__main__':
    main()
