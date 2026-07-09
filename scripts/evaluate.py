"""
Evaluate triangle detection against known control points.

For each map that has a _controlpoints.txt ground truth file:
1. Run DB-guided detection (db_matcher.py)
2. Convert control point map coords to pixel positions
3. Check if detections match known points within a radius
4. Report precision, recall, and positional error

This tells us: of the high-confidence DB candidates, how many
correspond to actual control points on the map?
"""

import sys
import time
import csv
import numpy as np
from pathlib import Path

from coord_converter import load_tfwx, map_to_pixel, load_control_points, get_map_extent
from image_loader import load_image, suppress_red
from db_matcher import (
    load_geodetic_db, filter_points_to_extent,
    load_grayscale_templates, verify_candidates,
)


def run_detection(map_dir, geo_db, template_dir, use_tif=False):
    """
    Run DB-guided detection on a single map and load its ground truth.

    This is the expensive part (template matching at every DB candidate);
    run it once per map and score against multiple thresholds with
    score_detections().

    Returns:
        dict with 'map_name', 'detections', 'gt_pixels', 'n_db_candidates',
        or None if the map can't be processed.
    """
    map_dir = Path(map_dir)
    map_name = map_dir.name

    # Find required files
    ext = '*.tif' if use_tif else '*.jpg'
    img_files = list(map_dir.glob(ext))
    tfwx_files = list(map_dir.glob('*.tfwx'))
    cp_files = list(map_dir.glob('*controlpoints.txt'))

    if not img_files or not tfwx_files or not cp_files:
        return None

    img_path = img_files[0]
    tfwx_path = tfwx_files[0]
    cp_path = cp_files[0]

    # Load affine transform
    affine = load_tfwx(tfwx_path)

    # Get image size
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(img_path) as pil_img:
        w, h = pil_img.size

    # Get map extent and filter DB
    extent = get_map_extent(affine, w, h)
    candidates = filter_points_to_extent(geo_db, extent)

    if not candidates:
        return None

    # Load image and run detection
    img = load_image(img_path)
    templates = load_grayscale_templates(template_dir)
    detections = verify_candidates(img, candidates, affine, templates)

    # Load ground truth control points
    ground_truth = load_control_points(cp_path)
    gt_pixels = []
    for cp in ground_truth:
        if cp.get('enable', 1) == 0:
            continue
        px, py = map_to_pixel(cp['map_x'], cp['map_y'], affine)
        gt_pixels.append((px, py, cp['map_x'], cp['map_y']))

    return {
        'map_name': map_name,
        'detections': detections,
        'gt_pixels': gt_pixels,
        'n_db_candidates': len(candidates),
    }


def score_detections(detection_result, match_radius=30, conf_threshold=0.5):
    """
    Score precomputed detections against ground truth at one threshold.

    Args:
        detection_result: dict from run_detection()
        match_radius: max distance (px) to count as match
        conf_threshold: minimum confidence to count a detection

    Returns:
        dict with evaluation metrics
    """
    map_name = detection_result['map_name']
    detections = detection_result['detections']
    gt_pixels = detection_result['gt_pixels']

    # Filter detections by confidence threshold
    high_conf = [d for d in detections if d.confidence >= conf_threshold]

    # Match detections to ground truth
    matched_gt = set()     # indices of matched ground truth points
    matched_det = set()    # indices of matched detections
    match_distances = []   # pixel distances for matched pairs

    for di, det in enumerate(high_conf):
        best_dist = float('inf')
        best_gi = -1
        for gi, (gx, gy, *_) in enumerate(gt_pixels):
            dist = np.sqrt((det.pixel_x - gx)**2 + (det.pixel_y - gy)**2)
            if dist < best_dist:
                best_dist = dist
                best_gi = gi

        if best_dist < match_radius and best_gi not in matched_gt:
            matched_gt.add(best_gi)
            matched_det.add(di)
            match_distances.append(best_dist)

    n_gt = len(gt_pixels)
    n_det = len(high_conf)
    n_matched = len(matched_gt)

    recall = n_matched / max(n_gt, 1)
    precision = n_matched / max(n_det, 1)
    mean_error = np.mean(match_distances) if match_distances else 0.0

    return {
        'map_name': map_name,
        'n_ground_truth': n_gt,
        'n_db_candidates': detection_result['n_db_candidates'],
        'n_detections': n_det,
        'n_matched': n_matched,
        'recall': recall,
        'precision': precision,
        'mean_error_px': mean_error,
        'match_distances': match_distances,
        'conf_threshold': conf_threshold,
        'high_conf_detections': high_conf,
        'gt_pixels': gt_pixels,
    }


def evaluate_map(map_dir, geo_db, template_dir, match_radius=30,
                 conf_threshold=0.5, use_tif=False):
    """
    Evaluate detection on a single map against its ground truth.

    Convenience wrapper: run_detection() + score_detections() in one call.
    When evaluating multiple thresholds, call run_detection() once and
    score_detections() per threshold instead.
    """
    detection_result = run_detection(map_dir, geo_db, template_dir, use_tif=use_tif)
    if detection_result is None:
        return None
    return score_detections(detection_result, match_radius=match_radius,
                            conf_threshold=conf_threshold)


def print_results(results):
    """Print formatted evaluation results."""
    print(f"\n{'Map':>15s} | {'GT':>3s} | {'DB Cand':>7s} | {'Dets':>5s} | "
          f"{'Match':>5s} | {'Recall':>7s} | {'Prec':>6s} | {'Err(px)':>8s}")
    print('-' * 80)

    totals = {'gt': 0, 'det': 0, 'matched': 0}

    for r in results:
        print(f"{r['map_name']:>15s} | {r['n_ground_truth']:3d} | "
              f"{r['n_db_candidates']:7d} | {r['n_detections']:5d} | "
              f"{r['n_matched']:5d} | {r['recall']:6.1%} | "
              f"{r['precision']:5.1%} | {r['mean_error_px']:7.1f}")
        totals['gt'] += r['n_ground_truth']
        totals['det'] += r['n_detections']
        totals['matched'] += r['n_matched']

    print('-' * 80)
    overall_recall = totals['matched'] / max(totals['gt'], 1)
    overall_precision = totals['matched'] / max(totals['det'], 1)
    print(f"{'TOTAL':>15s} | {totals['gt']:3d} | {'':>7s} | {totals['det']:5d} | "
          f"{totals['matched']:5d} | {overall_recall:6.1%} | "
          f"{overall_precision:5.1%} |")


if __name__ == '__main__':
    base = Path(__file__).resolve().parent.parent
    template_dir = base / 'scripts' / 'templates'

    # Load geodetic database
    print("Loading geodetic database...")
    geo_db = load_geodetic_db(base / 'Control_Points' / 'nikudot_bakara_slim.csv')
    print(f"  Loaded {len(geo_db)} points")

    # Discover all map folders across ground-truth series
    from data_paths import discover_map_dirs
    target = sys.argv[1] if len(sys.argv) > 1 else None
    map_dirs = discover_map_dirs(name_filter=target)

    # Confidence thresholds to evaluate
    thresholds = [0.5, 0.6, 0.7]

    # Detection is the expensive part — run it once per map,
    # then score against every threshold
    print(f"\nRunning detection on {len(map_dirs)} maps...")
    detection_results = []
    for map_dir in map_dirs:
        t0 = time.time()
        dr = run_detection(map_dir, geo_db, template_dir)
        elapsed = time.time() - t0
        if dr:
            detection_results.append(dr)
            print(f"  {dr['map_name']:>15s}: {len(dr['detections'])} detections, "
                  f"{len(dr['gt_pixels'])} ground truth points ({elapsed:.1f}s)")

    for threshold in thresholds:
        print(f"\n{'='*80}")
        print(f"  EVALUATION: confidence threshold = {threshold}")
        print(f"{'='*80}")

        results = [score_detections(dr, conf_threshold=threshold)
                   for dr in detection_results]
        if results:
            print_results(results)
