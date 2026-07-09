#!/usr/bin/env python3
"""
QA tool: find ALL triangles on a map by scanning the full image.

No DB, no OCR, no coordinates needed. Pipeline:
1. Upscale image 2x (to match template training resolution)
2. Template matching to find candidate positions
3. CNN classifier to filter true triangles from false positives

Usage:
    python qa_detections.py <image1.jpg> [image2.jpg ...]
    python qa_detections.py <directory_with_maps>
    python qa_detections.py --cnn-threshold 0.7 <image.jpg>

Output: <input_stem>_qa.jpg saved next to the input image.
Colors: green (CNN >= 0.9), yellow (0.7-0.9), red (0.5-0.7)
"""
import sys
import time
import numpy as np
import cv2
import torch
from pathlib import Path
from scipy.ndimage import maximum_filter

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from image_loader import load_image, suppress_red
from db_matcher import load_grayscale_templates
from train_classifier import TriangleCNN
from grid_label_ocr import find_neatline, find_grid_bounds


def load_cnn_model():
    """Load the trained triangle CNN classifier."""
    model = TriangleCNN()
    model_path = SCRIPT_DIR / 'triangle_classifier.pth'
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  CNN model: F1={ckpt['f1']:.3f} prec={ckpt['precision']:.3f} rec={ckpt['recall']:.3f}")
    return model


def grid_nms(detections, min_distance=20):
    """Fast NMS using a grid for O(n) average performance instead of O(n²)."""
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d[2], reverse=True)
    cell_size = min_distance
    grid = {}
    keep = []

    for det in detections:
        x, y = det[0], det[1]
        gx, gy = int(x // cell_size), int(y // cell_size)

        suppressed = False
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for kx, ky in grid.get((gx + dx, gy + dy), []):
                    if (x - kx)**2 + (y - ky)**2 < min_distance**2:
                        suppressed = True
                        break
                if suppressed:
                    break
            if suppressed:
                break

        if not suppressed:
            keep.append(det)
            grid.setdefault((gx, gy), []).append((x, y))

    return keep


def find_template_candidates(prep_gray, templates, threshold=0.55, min_distance=20):
    """Run template matching on full image, return peak positions."""
    all_peaks = []
    for tmpl_name, tmpl_gray, _, _ in templates:
        th, tw = tmpl_gray.shape[:2]
        h_img, w_img = prep_gray.shape[:2]
        if th >= h_img or tw >= w_img:
            continue

        result = cv2.matchTemplate(prep_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
        size = 2 * min_distance + 1
        local_max = maximum_filter(result, size=size)
        peaks_mask = (result == local_max) & (result >= threshold)
        ys, xs = np.where(peaks_mask)
        confs = result[ys, xs]

        half_w, half_h = tw // 2, th // 2
        for i in range(len(xs)):
            all_peaks.append((int(xs[i]) + half_w, int(ys[i]) + half_h,
                              float(confs[i]), tmpl_name))

    # Fast grid-based NMS across templates
    return grid_nms(all_peaks, min_distance)


def classify_crops_chunked(model, crops, chunk_size=2048):
    """Run the CNN over a list of 64x64 crops in memory-safe chunks.

    Crops are numpy views into the preprocessed image; only one chunk is
    materialized as a float batch at a time, so 100k+ candidates are fine.
    Returns an array of probabilities.
    """
    probs = []
    with torch.no_grad():
        for i in range(0, len(crops), chunk_size):
            batch = np.stack(crops[i:i + chunk_size]).astype(np.float32) / 255.0
            batch_tensor = torch.from_numpy(batch).unsqueeze(1)
            logits = model(batch_tensor).squeeze(1)
            probs.append(torch.sigmoid(logits).numpy())
    return np.concatenate(probs) if probs else np.array([])


def classify_candidates(prep_gray, candidates, model, max_candidates=None):
    """Extract 64x64 crops at candidate positions and run CNN.

    max_candidates=None classifies EVERY candidate. Capping by template
    score is a recall killer: on dense maps with ~100k candidates, real
    triangles rank far below the top few thousand (on 14-15-Lydda, 126 of
    127 manually-marked missed triangles were cut by a top-1500 cap before
    the CNN ever saw them).
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
        return []

    probs = classify_crops_chunked(model, crops)

    results = []
    for i, (x, y, tconf, tname) in enumerate(valid_candidates):
        results.append((x, y, tconf, float(probs[i])))

    return results


def process_map_qa(image_path, templates, model, cnn_threshold=0.5,
                   tmpl_threshold=0.55):
    """Scan a map for triangles using template matching + CNN."""
    image_path = Path(image_path)
    map_name = image_path.stem
    print(f"\n{'='*60}")
    print(f"QA: {map_name}")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()
    img = load_image(str(image_path))
    h_img, w_img = img.shape[:2]

    # Determine upscale factor — templates trained on ~14000px images
    scale = max(1.0, 14000.0 / w_img)
    scale = round(scale * 2) / 2  # round to nearest 0.5
    if scale > 1.0:
        img_up = cv2.resize(img, (int(w_img * scale), int(h_img * scale)),
                            interpolation=cv2.INTER_LINEAR)
        print(f"  {w_img}x{h_img} -> {img_up.shape[1]}x{img_up.shape[0]} (upscale {scale}x)")
    else:
        img_up = img
        print(f"  {w_img}x{h_img} (no upscale needed)")
    sys.stdout.flush()

    prep = suppress_red(img_up)

    # Find neatline on original image
    neatline = find_neatline(img)
    if neatline:
        print(f"  Neatline: T={neatline['top']} B={neatline['bottom']} "
              f"L={neatline['left']} R={neatline['right']}")

    # Step 1: template matching for candidates
    t1 = time.time()
    candidates = find_template_candidates(prep, templates, threshold=tmpl_threshold)
    print(f"  Template candidates: {len(candidates)} ({time.time() - t1:.1f}s)")
    sys.stdout.flush()

    # Step 2: CNN classification
    t1 = time.time()
    results = classify_candidates(prep, candidates, model)
    print(f"  CNN classified: {len(results)} ({time.time() - t1:.1f}s)")
    sys.stdout.flush()

    # Filter by CNN threshold
    detections = [(x, y, tconf, cnn_prob) for x, y, tconf, cnn_prob in results
                  if cnn_prob >= cnn_threshold]
    print(f"  CNN positives (>={cnn_threshold}): {len(detections)}")

    # Map back to original image coordinates
    detections_orig = [(x / scale, y / scale, tconf, cnn_prob)
                       for x, y, tconf, cnn_prob in detections]

    # Filter by grid bounds (tighter than neatline) or fall back to neatline
    grid_bounds = find_grid_bounds(img, neatline) if neatline else None

    if grid_bounds:
        margin = 15
        before = len(detections_orig)
        detections_orig = [
            (x, y, tc, cp) for x, y, tc, cp in detections_orig
            if (grid_bounds['left'] - margin <= x <= grid_bounds['right'] + margin and
                grid_bounds['top'] - margin <= y <= grid_bounds['bottom'] + margin)
        ]
        if before > len(detections_orig):
            print(f"  Grid bounds filter: {before} -> {len(detections_orig)}")
    elif neatline:
        margin = 30
        before = len(detections_orig)
        detections_orig = [
            (x, y, tc, cp) for x, y, tc, cp in detections_orig
            if (neatline['left'] - margin <= x <= neatline['right'] + margin and
                neatline['top'] - margin <= y <= neatline['bottom'] + margin)
        ]
        if before > len(detections_orig):
            print(f"  Neatline filter: {before} -> {len(detections_orig)}")

    high = sum(1 for _, _, _, cp in detections_orig if cp >= 0.9)
    med = sum(1 for _, _, _, cp in detections_orig if 0.7 <= cp < 0.9)
    low = sum(1 for _, _, _, cp in detections_orig if cp < 0.7)
    print(f"  Final: {high} high (>=0.9), {med} med (0.7-0.9), {low} low (<0.7)")

    # Draw on full-res original image
    vis = img.copy()
    if neatline:
        cv2.rectangle(vis, (neatline['left'], neatline['top']),
                      (neatline['right'], neatline['bottom']), (255, 100, 0), 2)
    if grid_bounds:
        # Draw grid bounds in cyan
        cv2.rectangle(vis, (grid_bounds['left'], grid_bounds['top']),
                      (grid_bounds['right'], grid_bounds['bottom']), (255, 255, 0), 2)

    for x, y, tconf, cnn_prob in sorted(detections_orig, key=lambda d: d[3]):
        ix, iy = int(round(x)), int(round(y))
        if cnn_prob >= 0.9:
            color, radius, thick = (0, 200, 0), 30, 3
        elif cnn_prob >= 0.7:
            color, radius, thick = (0, 220, 220), 25, 2
        else:
            color, radius, thick = (0, 0, 220), 20, 2

        cv2.circle(vis, (ix, iy), radius, color, thick)
        label = f"{cnn_prob:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th_t), bl = cv2.getTextSize(label, font, 0.5, 1)
        lx, ly = ix + radius + 3, iy + th_t // 2
        cv2.rectangle(vis, (lx - 1, ly - th_t - 1),
                      (lx + tw + 1, ly + bl + 1), (0, 0, 0), cv2.FILLED)
        cv2.putText(vis, label, (lx, ly), font, 0.5, color, 1)

    elapsed = time.time() - t0
    for i, line in enumerate([
        f"{map_name}  ({w_img}x{h_img}, upscale={scale}x)",
        f"Detections: {high} high, {med} med, {low} low ({len(detections_orig)} total)",
        f"CNN>={cnn_threshold}  tmpl>={tmpl_threshold}  |  {elapsed:.0f}s",
    ]):
        (tw, th_t), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        y_off = 40 + i * (th_t + 16)
        cv2.rectangle(vis, (10, y_off - th_t - 6), (20 + tw, y_off + 6), (0, 0, 0), cv2.FILLED)
        cv2.putText(vis, line, (14, y_off), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    output_path = image_path.parent / f"{map_name}_qa.jpg"
    cv2.imwrite(str(output_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  Output: {output_path}")
    print(f"  Done in {elapsed:.1f}s")

    return {'map_name': map_name, 'n_detections': len(detections_orig),
            'n_high': high, 'n_med': med, 'n_low': low, 'output': str(output_path)}


def main():
    if len(sys.argv) < 2:
        print("Usage: python qa_detections.py [--cnn-threshold 0.5] [--tmpl-threshold 0.55] <image ...> | <directory>")
        sys.exit(1)

    cnn_threshold = 0.5
    tmpl_threshold = 0.55
    args = sys.argv[1:]
    if '--cnn-threshold' in args:
        idx = args.index('--cnn-threshold')
        cnn_threshold = float(args[idx + 1])
        args = args[:idx] + args[idx + 2:]
    if '--tmpl-threshold' in args:
        idx = args.index('--tmpl-threshold')
        tmpl_threshold = float(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

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

    print(f"Processing {len(image_paths)} map(s), cnn_threshold={cnn_threshold}, tmpl_threshold={tmpl_threshold}")

    # Load shared resources
    templates = load_grayscale_templates(SCRIPT_DIR / 'templates')
    print(f"  {len(templates)} templates")
    model = load_cnn_model()

    results = []
    for img_path in image_paths:
        result = process_map_qa(img_path, templates, model,
                                cnn_threshold=cnn_threshold,
                                tmpl_threshold=tmpl_threshold)
        results.append((img_path.stem, result))

    print(f"\n{'='*70}")
    print(f"  QA SUMMARY  (cnn>={cnn_threshold})")
    print(f"{'='*70}")
    print(f"{'Map':>30} | {'High':>4} | {'Med':>4} | {'Low':>4} | {'Total':>5}")
    print(f"{'-'*70}")
    for name, res in results:
        if res:
            print(f"{name:>30} | {res['n_high']:>4} | {res['n_med']:>4} | "
                  f"{res['n_low']:>4} | {res['n_detections']:>5}")
        else:
            print(f"{name:>30} | {'':>4} | {'':>4} | {'':>4} | {'FAIL':>5}")

    ok = sum(1 for _, r in results if r)
    print(f"\n{ok}/{len(results)} maps processed")


if __name__ == '__main__':
    main()
