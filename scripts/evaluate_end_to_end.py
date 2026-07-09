#!/usr/bin/env python3
"""
End-to-end georeferencing benchmark on the frozen held-out maps.

Crop-level F1 (evaluate_holdout.py) is a proxy: the product is the
TRANSFORM. This benchmark runs the full bootstrap pipeline on each
held-out map AS IF it had no georeferencing (OCR margins -> Old Grid
affine -> project DB -> template match + CNN -> EPSG:6991 affine) and
grades the recovered transform against the human ground truth:

  - control-point error: recovered affine applied at the ground-truth
    pixel positions of the human control points vs their known
    EPSG:6991 coordinates (meters) — the headline number;
  - corner drift: recovered vs ground-truth affine at the image
    corners (meters) — shows extrapolation behavior.

Nothing is written into the map directories; all outputs go to
output/end_to_end/<map_id>/.

Usage:
    python evaluate_end_to_end.py              # all held-out maps
    python evaluate_end_to_end.py M9_4149      # one map
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_paths import BASE_DIR, GEODETIC_DB, TEMPLATE_DIR, ground_truth_series
from holdout import HELD_OUT_MAP_IDS
from coord_converter import load_tfwx, map_to_pixel, pixel_to_map
from image_loader import load_image, suppress_red
from ingest_job_maps import (find_assets, load_points_file, parse_aux_gcps,
                             _score_positions, TARGET_DPI)
from grid_label_ocr import read_grid_labels, labels_to_affine
from db_matcher import load_geodetic_db, load_grayscale_templates
from bootstrap_from_grid import (
    bootstrap_georeference, labels_to_grid_points, labels_to_old_grid_extent,
    sheet_label_ranges,
)

OUT_ROOT = BASE_DIR / 'output' / 'end_to_end'


def locate_assets(map_ids):
    """Find image / ground-truth tfwx / control points for each map id."""
    found = {}
    for series in ground_truth_series():
        for map_id, a in find_assets(series).items():
            if map_id in map_ids and map_id not in found:
                found[map_id] = a
    return found


def load_rescaled(image_path):
    """Load an image rescaled to TARGET_DPI, same convention as ingest.

    Returns (img, scale). The ground-truth world file refers to the
    ORIGINAL resolution; divide its linear coefficients by `scale` to
    move it into this frame.
    """
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(image_path) as im:
        dpi = tuple(float(v) for v in im.info.get('dpi', (TARGET_DPI,) * 2))
    img = load_image(str(image_path))
    scale = round((TARGET_DPI / dpi[0]) * 2) / 2 if dpi[0] else 1.0
    if abs(scale - 1.0) < 0.01:
        scale = 1.0
    if scale != 1.0:
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                         interpolation=cv2.INTER_LINEAR)
    return img, scale


def rescale_affine(affine, scale):
    """Convert a world-file affine from original-resolution pixels to a
    frame upscaled by `scale` (col_new = col * scale)."""
    if scale == 1.0:
        return affine
    out = dict(affine)
    for k in ('a', 'b', 'c', 'd'):
        out[k] = affine[k] / scale
    M = np.array([[out['a'], out['c']], [out['b'], out['d']]])
    out['M'], out['M_inv'] = M, np.linalg.inv(M)
    out['offset'] = np.array([out['e'], out['f']])
    out['forward'] = np.array([[out['a'], out['c'], out['e']],
                               [out['b'], out['d'], out['f']]])
    return out


def run_one(map_id, assets, geo_db):
    """Bootstrap one map blind; grade against ground truth.

    Returns a result dict; 'status' != 'ok' explains any failure.
    """
    res = {'map_id': map_id, 'status': None}
    a = assets.get(map_id)
    if not a or a['image'] is None or a['tfwx'] is None:
        res['status'] = 'missing image or ground-truth tfwx'
        return res
    targets = load_points_file(a['points']) if a['points'] else None
    if (targets is None or len(targets) == 0) and a['aux']:
        gcps = parse_aux_gcps(a['aux'])
        if gcps is not None:
            targets = gcps[1]          # map-coordinate side of the GCPs
    if targets is None or len(targets) == 0:
        res['status'] = 'no control points to grade against'
        return res

    print(f"\n{'=' * 60}\n  {map_id}  ({a['image'].name})\n{'=' * 60}")
    img, scale = load_rescaled(a['image'])
    h, w = img.shape[:2]
    gt = rescale_affine(load_tfwx(a['tfwx']), scale)
    if scale != 1.0:
        print(f"  rescaled {scale:g}x to {TARGET_DPI:.0f} dpi frame")

    # Trust-but-verify the ground truth itself: control points ARE triangle
    # symbols, so the GT transform must land them on triangles in THIS image
    # (catches a world file written for a different-resolution copy).
    gt_px_all = np.array([map_to_pixel(tx, ty, gt) for tx, ty in targets])
    score, n_valid = _score_positions(suppress_red(img), gt_px_all,
                                      load_grayscale_templates(TEMPLATE_DIR))
    if n_valid < 3 or score < 0.30:
        res['status'] = (f'ground-truth tfwx does not fit this image '
                         f'(match {score:.2f} at {n_valid} pts)')
        return res
    print(f"  GT transform verified (template match {score:.2f} "
          f"at {n_valid} control points)")

    # --- OCR the margins (blind: no use of the ground truth) ---
    t0 = time.time()
    sheet_ranges = sheet_label_ranges(a['image'].name)  # None for M-names
    if sheet_ranges:
        ocr = read_grid_labels(img, expected_easting_range=sheet_ranges[0],
                               expected_northing_range=sheet_ranges[1])
    else:
        ocr = read_grid_labels(img)
    n_e, n_n = len(ocr['easting_labels']), len(ocr['northing_labels'])
    res['ocr_labels'] = f"{n_e}E/{n_n}N"
    print(f"  OCR: {n_e} easting + {n_n} northing labels ({time.time()-t0:.0f}s)")
    if n_e < 2 or n_n < 2:
        res['status'] = 'OCR: not enough grid labels'
        return res
    label_affine = labels_to_affine(ocr)
    if label_affine is None:
        res['status'] = 'OCR: label fit rejected (labels_to_affine -> None)'
        return res

    grid_points = labels_to_grid_points(ocr, label_affine)
    old_e_range, old_n_range = labels_to_old_grid_extent(ocr, label_affine)

    out_dir = OUT_ROOT / map_id
    out_dir.mkdir(parents=True, exist_ok=True)
    boot = bootstrap_georeference(a['image'], geo_db, TEMPLATE_DIR,
                                  grid_points, old_e_range, old_n_range,
                                  output_dir=out_dir, image=img)
    if boot is None:
        res['status'] = 'bootstrap failed (too few verified points)'
        return res
    rec = boot['affine']
    res.update(n_points=boot['n_points'], n_inliers=boot['n_inliers'],
               fit_rmse_m=boot['fit_rmse_m'])

    # --- grade: control-point error in meters ---
    gt_px = np.array([map_to_pixel(tx, ty, gt) for tx, ty in targets])
    inside = ((gt_px[:, 0] >= 0) & (gt_px[:, 0] < w) &
              (gt_px[:, 1] >= 0) & (gt_px[:, 1] < h))
    gt_px, tgt = gt_px[inside], np.asarray(targets)[inside]
    pred = np.array([pixel_to_map(px, py, rec) for px, py in gt_px])
    err = np.sqrt(((pred - tgt) ** 2).sum(axis=1))
    res['cp_n'] = len(err)
    res['cp_mean_m'] = float(err.mean())
    res['cp_median_m'] = float(np.median(err))
    res['cp_max_m'] = float(err.max())

    # --- corner drift between the two affines ---
    corners = [(0, 0), (w, 0), (0, h), (w, h)]
    drift = [np.hypot(*(np.subtract(pixel_to_map(px, py, rec),
                                    pixel_to_map(px, py, gt))))
             for px, py in corners]
    res['corner_max_m'] = float(max(drift))
    res['status'] = 'ok'
    print(f"  GRADE: {len(err)} control points, "
          f"err mean={res['cp_mean_m']:.1f}m median={res['cp_median_m']:.1f}m "
          f"max={res['cp_max_m']:.1f}m | corner drift max={res['corner_max_m']:.1f}m")
    return res


def main():
    map_ids = [m for m in sys.argv[1:] if not m.startswith('-')] or HELD_OUT_MAP_IDS
    assets = locate_assets(set(map_ids))
    print(f"Benchmarking {len(map_ids)} held-out maps: {', '.join(map_ids)}")
    print("Loading geodetic database...")
    geo_db = load_geodetic_db(GEODETIC_DB)

    results = [run_one(m, assets, geo_db) for m in map_ids]

    print(f"\n{'=' * 78}\n  END-TO-END SUMMARY (meters, vs human ground truth)\n{'=' * 78}")
    print(f"{'map':<10} {'ocr':>7} {'pts':>4} {'inl':>4} {'fit':>6} "
          f"{'cp_mean':>8} {'cp_med':>7} {'cp_max':>7} {'corner':>7}  status")
    print('-' * 78)
    ok_meds = []
    for r in results:
        if r['status'] == 'ok':
            ok_meds.append(r['cp_median_m'])
            print(f"{r['map_id']:<10} {r.get('ocr_labels','-'):>7} "
                  f"{r['n_points']:>4} {r['n_inliers']:>4} {r['fit_rmse_m']:>5.1f}m "
                  f"{r['cp_mean_m']:>7.1f}m {r['cp_median_m']:>6.1f}m "
                  f"{r['cp_max_m']:>6.1f}m {r['corner_max_m']:>6.1f}m  ok")
        else:
            print(f"{r['map_id']:<10} {r.get('ocr_labels','-'):>7} "
                  f"{'':>4} {'':>4} {'':>6} {'':>8} {'':>7} {'':>7} {'':>7}  "
                  f"FAIL: {r['status']}")
    print('-' * 78)
    n_ok = len(ok_meds)
    print(f"\nGeoreferenced {n_ok}/{len(results)} maps blind"
          + (f"; median control-point error {np.median(ok_meds):.1f} m"
             if ok_meds else ""))
    print("This measures the actual product. Crop F1 is only a proxy.")


if __name__ == '__main__':
    main()
