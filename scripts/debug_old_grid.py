#!/usr/bin/env python3
"""Diagnose the old-grid affine for a held-out map (no template matching).

Builds the OCR label affine and the old-grid affine exactly as the
bootstrap does, then checks the old-grid affine's TRUE pixel size (M-matrix
singular values) and compares its DB-point projections against where the
ground-truth transform + geodetic DB say those points actually sit.
"""
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_paths import GEODETIC_DB, ground_truth_series
from coord_converter import load_tfwx, map_to_pixel
from ingest_job_maps import find_assets, TARGET_DPI
from grid_label_ocr import read_grid_labels, labels_to_affine
from db_matcher import load_geodetic_db
from bootstrap_from_grid import (
    build_old_grid_affine, labels_to_grid_points, labels_to_old_grid_extent,
    filter_points_by_old_grid,
)
from evaluate_end_to_end import load_rescaled, rescale_affine, locate_assets


def main():
    map_id = sys.argv[1] if len(sys.argv) > 1 else "M7_4138"
    a = locate_assets({map_id})[map_id]
    img, scale = load_rescaled(a['image'])
    h, w = img.shape[:2]
    gt = rescale_affine(load_tfwx(a['tfwx']), scale)
    print(f"{map_id}: image {w}x{h}, scale {scale:g}")
    print(f"GT tfwx (this frame): a={gt['a']:.4f} b={gt['b']:.4f} "
          f"c={gt['c']:.4f} d={gt['d']:.4f}")
    gtM = np.array([[gt['a'], gt['c']], [gt['b'], gt['d']]])
    print(f"GT M singular values (m/px): {np.linalg.svd(gtM, compute_uv=False)}")

    ocr = read_grid_labels(img)
    label_affine = labels_to_affine(ocr)
    if label_affine is None:
        print("labels_to_affine -> None; cannot continue")
        return
    print(f"\nLabel affine: a(e/col)={label_affine['a']:.4f} "
          f"d(n/row)={label_affine['d']:.4f} "
          f"e={label_affine['e']:.1f} f={label_affine['f']:.1f}")

    grid_points = labels_to_grid_points(ocr, label_affine)
    gp = np.array(grid_points)
    print(f"\ngrid_points: {len(gp)}  "
          f"px col[{gp[:,0].min():.0f},{gp[:,0].max():.0f}] "
          f"row[{gp[:,1].min():.0f},{gp[:,1].max():.0f}]")
    print(f"  old_e span [{gp[:,2].min():.0f},{gp[:,2].max():.0f}] "
          f"old_n span [{gp[:,3].min():.0f},{gp[:,3].max():.0f}]")

    old_affine = build_old_grid_affine(grid_points)
    oM = old_affine['M']
    print(f"\nold-grid affine M:\n{oM}")
    print(f"  reported 'pixel size' a,d = {abs(old_affine['a']):.4f}, "
          f"{abs(old_affine['d']):.4f}  (misleading print)")
    print(f"  TRUE M singular values (m/px): "
          f"{np.linalg.svd(oM, compute_uv=False)}")

    # Residual of the old-grid affine on its own grid points
    pred = np.array([map_to_pixel(e, n, old_affine) for _, _, e, n in grid_points])
    # map_to_pixel inverts; instead check forward fit residual in meters
    fwd = old_affine['forward']
    homog = np.column_stack([gp[:, 0], gp[:, 1], np.ones(len(gp))])
    pred_map = (fwd @ homog.T).T          # (old_e, old_n)
    resid = np.hypot(pred_map[:, 0] - gp[:, 2], pred_map[:, 1] - gp[:, 3])
    print(f"  old-grid forward residual on grid points: "
          f"mean={resid.mean():.1f}m max={resid.max():.1f}m")

    # The real test: project DB points two ways and compare.
    #   way A: DB.old_e/old_n through the bootstrap old-grid affine -> pixel
    #   way B: DB.easting_6991/northing_6991 through the GT tfwx     -> pixel
    # For a correct old-grid affine these agree (both put the triangle in the
    # same place). Disagreement in km/px = the bug.
    geo_db = load_geodetic_db(GEODETIC_DB)
    old_e_range, old_n_range = labels_to_old_grid_extent(ocr)
    cands = filter_points_by_old_grid(geo_db, old_e_range[0], old_e_range[1],
                                      old_n_range[0], old_n_range[1])
    print(f"\n{len(cands)} DB points in extent; comparing projections:")
    diffs = []
    for p in cands:
        ax, ay = map_to_pixel(p.old_e, p.old_n, old_affine)
        bx, by = map_to_pixel(p.easting_6991, p.northing_6991, gt)
        if 0 <= bx < w and 0 <= by < h:
            diffs.append(np.hypot(ax - bx, ay - by))
    if diffs:
        diffs = np.array(diffs)
        print(f"  old-grid vs GT pixel disagreement over {len(diffs)} in-frame "
              f"points: median={np.median(diffs):.0f}px mean={diffs.mean():.0f}px "
              f"min={diffs.min():.0f}px")
        print(f"  (median {np.median(diffs)*abs(gt['a']):.0f} m). If this is "
              f">~1 crop (40px) the projection points at the wrong triangles.")


if __name__ == '__main__':
    main()
