"""
Automatic georeferencing pipeline.

Given a scanned map image and the geodetic database:
1. Detect triangle symbols using DB-guided template matching
2. Select the best spatially-distributed high-confidence detections
3. Match detections to geodetic database points
4. Compute an affine (or polynomial) georeferencing transform
5. Output a world file (.tfwx) for the map

For validation: compare the computed transform against the existing
manually-created transform and report accuracy.
"""

import sys
import time
import csv
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares

from coord_converter import (
    load_tfwx, map_to_pixel, pixel_to_map, load_control_points, get_map_extent,
)
from image_loader import load_image, suppress_red, suppress_colors
from db_matcher import (
    load_geodetic_db, filter_points_to_extent,
    load_grayscale_templates, verify_candidates,
)


def select_best_points(detections, n_points=15, min_conf=0.5, min_distance=200):
    """
    Select the best spatially-distributed detections for georeferencing.

    Uses a greedy algorithm: start with the highest confidence detection,
    then add the next highest confidence that is at least min_distance pixels
    away from all already-selected points.

    Args:
        detections: list of Detection objects (sorted by confidence desc)
        n_points: maximum number of points to select
        min_conf: minimum confidence threshold
        min_distance: minimum pixel distance between selected points

    Returns:
        list of selected Detection objects
    """
    # Filter by confidence
    candidates = [d for d in detections if d.confidence >= min_conf]

    selected = []
    for det in candidates:
        if len(selected) >= n_points:
            break

        # Check distance to already selected points
        too_close = False
        for s in selected:
            dist = np.sqrt((det.pixel_x - s.pixel_x)**2 +
                          (det.pixel_y - s.pixel_y)**2)
            if dist < min_distance:
                too_close = True
                break

        if not too_close:
            selected.append(det)

    return selected


def compute_affine_transform(pixel_points, map_points):
    """
    Compute the best-fit affine transform from pixel to map coordinates.

    pixel_to_map: [map_x, map_y]^T = A @ [px, py, 1]^T

    Uses least squares to fit:
        map_x = a * px + c * py + e
        map_y = b * px + d * py + f

    Args:
        pixel_points: Nx2 array of (px, py)
        map_points: Nx2 array of (map_x, map_y)

    Returns:
        dict with affine coefficients compatible with coord_converter
    """
    n = len(pixel_points)
    assert n >= 3, "Need at least 3 points for affine transform"

    px = np.array(pixel_points)
    mp = np.array(map_points)

    # Build design matrix: [px, py, 1]
    A = np.column_stack([px, np.ones(n)])  # Nx3

    # Solve for map_x = a*px + c*py + e
    coeffs_x, residuals_x, _, _ = np.linalg.lstsq(A, mp[:, 0], rcond=None)
    # Solve for map_y = b*px + d*py + f
    coeffs_y, residuals_y, _, _ = np.linalg.lstsq(A, mp[:, 1], rcond=None)

    a, c, e = coeffs_x  # px_coeff, py_coeff, const for map_x
    b, d, f = coeffs_y  # px_coeff, py_coeff, const for map_y

    # Build forward matrix
    forward = np.array([[a, c, e], [b, d, f]])
    M = np.array([[a, c], [b, d]])
    M_inv = np.linalg.inv(M)
    offset = np.array([e, f])

    # Compute residuals
    predicted = (forward @ np.column_stack([px, np.ones(n)]).T).T
    errors = mp - predicted
    rmse = np.sqrt(np.mean(errors**2))

    return {
        'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f,
        'forward': forward,
        'M': M,
        'M_inv': M_inv,
        'offset': offset,
        'rmse_meters': rmse,
        'n_points': n,
        'residuals': errors,
    }


def compute_polynomial_transform(pixel_points, map_points, order=2):
    """
    Compute a polynomial transform (order 2) from pixel to map coordinates.

    For order=2:
        map_x = a0 + a1*px + a2*py + a3*px*py + a4*px^2 + a5*py^2
        map_y = b0 + b1*px + b2*py + b3*px*py + b4*px^2 + b5*py^2

    Args:
        pixel_points: Nx2 array
        map_points: Nx2 array
        order: 1 (affine) or 2 (polynomial)

    Returns:
        dict with transform coefficients and residuals
    """
    n = len(pixel_points)
    px = np.array(pixel_points)
    mp = np.array(map_points)

    if order == 1:
        # Same as affine
        return compute_affine_transform(pixel_points, map_points)

    min_points = 6 if order == 2 else 3
    assert n >= min_points, f"Need at least {min_points} points for order-{order} transform"

    # Build design matrix for order 2
    x, y = px[:, 0], px[:, 1]
    A = np.column_stack([
        np.ones(n), x, y, x * y, x**2, y**2
    ])

    # Solve for both map_x and map_y
    coeffs_x, _, _, _ = np.linalg.lstsq(A, mp[:, 0], rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(A, mp[:, 1], rcond=None)

    # Compute residuals
    pred_x = A @ coeffs_x
    pred_y = A @ coeffs_y
    errors = mp - np.column_stack([pred_x, pred_y])
    rmse = np.sqrt(np.mean(errors**2))

    return {
        'order': order,
        'coeffs_x': coeffs_x,
        'coeffs_y': coeffs_y,
        'rmse_meters': rmse,
        'n_points': n,
        'residuals': errors,
    }


def apply_polynomial_transform(px, py, transform):
    """Apply a polynomial transform to pixel coordinates."""
    if 'a' in transform:
        # Affine transform
        return pixel_to_map(px, py, transform)

    cx = transform['coeffs_x']
    cy = transform['coeffs_y']

    map_x = cx[0] + cx[1]*px + cx[2]*py + cx[3]*px*py + cx[4]*px**2 + cx[5]*py**2
    map_y = cy[0] + cy[1]*px + cy[2]*py + cy[3]*px*py + cy[4]*px**2 + cy[5]*py**2
    return map_x, map_y


def write_tfwx(transform, output_path):
    """Write a TFWX world file from an affine transform."""
    with open(output_path, 'w') as f:
        f.write(f"{transform['a']:.10f}\n")
        f.write(f"{transform['b']:.10f}\n")
        f.write(f"{transform['c']:.10f}\n")
        f.write(f"{transform['d']:.10f}\n")
        f.write(f"{transform['e']:.6f}\n")
        f.write(f"{transform['f']:.6f}\n")


def evaluate_transform(computed_transform, reference_tfwx_path,
                        controlpoints_path, image_size):
    """
    Compare a computed transform against the existing manual transform.

    Args:
        computed_transform: dict from compute_affine_transform
        reference_tfwx_path: path to existing .tfwx
        controlpoints_path: path to _controlpoints.txt
        image_size: (width, height) of image

    Returns:
        dict with accuracy metrics
    """
    ref_affine = load_tfwx(reference_tfwx_path)
    cps = load_control_points(controlpoints_path)

    errors = []
    for cp in cps:
        if cp.get('enable', 1) == 0:
            continue

        # Get reference pixel position
        ref_px, ref_py = map_to_pixel(cp['map_x'], cp['map_y'], ref_affine)

        # Compute map coords from computed transform
        if 'a' in computed_transform:
            comp_mx, comp_my = pixel_to_map(ref_px, ref_py, computed_transform)
        else:
            comp_mx, comp_my = apply_polynomial_transform(
                ref_px, ref_py, computed_transform
            )

        # Error in meters
        err = np.sqrt((comp_mx - cp['map_x'])**2 + (comp_my - cp['map_y'])**2)
        errors.append(err)

    errors = np.array(errors)
    if len(errors) == 0:
        return {'n_checkpoints': 0, 'mean_error_m': float('nan'),
                'max_error_m': float('nan'), 'rmse_m': float('nan'),
                'median_error_m': float('nan'), 'errors': errors}
    return {
        'n_checkpoints': len(errors),
        'mean_error_m': float(errors.mean()),
        'max_error_m': float(errors.max()),
        'rmse_m': float(np.sqrt(np.mean(errors**2))),
        'median_error_m': float(np.median(errors)),
        'errors': errors,
    }


def ransac_affine(pixel_pts, map_pts, n_iter=500, inlier_thresh=15.0, min_inliers=5):
    """
    RANSAC-based robust affine estimation.

    Randomly samples 3 points, fits affine, counts inliers,
    and returns the best model.

    Args:
        pixel_pts: Nx2 array
        map_pts: Nx2 array
        n_iter: number of RANSAC iterations
        inlier_thresh: max residual (meters) to count as inlier
        min_inliers: minimum inliers for a valid model

    Returns:
        (best_affine, inlier_mask) or (None, None)
    """
    n = len(pixel_pts)
    if n < 3:
        return None, None

    best_inliers = None
    best_n_inliers = 0

    for _ in range(n_iter):
        # Random sample of 3 points
        idx = np.random.choice(n, 3, replace=False)
        try:
            affine = compute_affine_transform(pixel_pts[idx], map_pts[idx])
        except (np.linalg.LinAlgError, AssertionError):
            continue

        # Count inliers
        predicted = (affine['forward'] @ np.column_stack(
            [pixel_pts, np.ones(n)]).T).T
        residuals = np.sqrt(np.sum((map_pts - predicted)**2, axis=1))
        inlier_mask = residuals < inlier_thresh

        n_inliers = inlier_mask.sum()
        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_inliers = inlier_mask

    if best_inliers is None or best_n_inliers < min_inliers:
        return None, None

    # Refit with all inliers
    final_affine = compute_affine_transform(
        pixel_pts[best_inliers], map_pts[best_inliers]
    )

    return final_affine, best_inliers


def auto_georeference_map(map_dir, geo_db, template_dir,
                            min_conf=0.5, n_points=50, use_tif=False):
    """
    Full automatic georeferencing pipeline for a single map.

    Args:
        map_dir: path to map folder
        geo_db: geodetic database
        template_dir: templates folder
        min_conf: minimum detection confidence
        n_points: number of control points to use

    Returns:
        dict with transform, selected points, and evaluation metrics
    """
    map_dir = Path(map_dir)
    map_name = map_dir.name

    # Find files
    ext = '*.tif' if use_tif else '*.jpg'
    img_files = list(map_dir.glob(ext))
    tfwx_files = list(map_dir.glob('*.tfwx'))

    if not img_files or not tfwx_files:
        print(f"  Missing files in {map_dir}")
        return None

    img_path = img_files[0]
    tfwx_path = tfwx_files[0]

    # Load reference transform (for evaluation only — in production we won't have this)
    ref_affine = load_tfwx(tfwx_path)

    # Get image dimensions
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(img_path) as pil_img:
        w, h = pil_img.size

    # Get map extent and filter DB
    extent = get_map_extent(ref_affine, w, h)
    candidates = filter_points_to_extent(geo_db, extent)

    if not candidates:
        print(f"  No DB candidates for {map_name}")
        return None

    # Load image and run detection
    img = load_image(img_path)
    templates = load_grayscale_templates(template_dir)
    detections = verify_candidates(img, candidates, ref_affine, templates)

    # Select best points
    selected = select_best_points(detections, n_points=n_points, min_conf=min_conf)

    if len(selected) < 3:
        print(f"  Only {len(selected)} points selected — not enough for transform")
        return None

    # Build point pairs: (pixel_x, pixel_y) -> (map_easting, map_northing)
    pixel_pts = np.array([(d.pixel_x, d.pixel_y) for d in selected])
    map_pts = np.array([(d.geo_point.easting_6991, d.geo_point.northing_6991)
                        for d in selected])

    # Compute affine transform
    affine = compute_affine_transform(pixel_pts, map_pts)

    # RANSAC-robust affine (reject outliers)
    ransac_result, inlier_mask = ransac_affine(pixel_pts, map_pts)
    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0

    # Evaluate against reference
    cp_files = list(map_dir.glob('*controlpoints.txt'))
    eval_result = None
    eval_ransac = None
    if cp_files:
        eval_result = evaluate_transform(affine, tfwx_path, cp_files[0], (w, h))
        if ransac_result:
            eval_ransac = evaluate_transform(
                ransac_result, tfwx_path, cp_files[0], (w, h)
            )

    return {
        'map_name': map_name,
        'n_candidates': len(candidates),
        'n_detections': len(detections),
        'n_selected': len(selected),
        'n_inliers': n_inliers,
        'selected': selected,
        'affine': affine,
        'ransac_affine': ransac_result,
        'eval_affine': eval_result,
        'eval_ransac': eval_ransac,
    }


if __name__ == '__main__':
    base = Path(__file__).resolve().parent.parent
    template_dir = base / 'scripts' / 'templates'
    output_dir = base / 'output'
    output_dir.mkdir(exist_ok=True)

    # Load geodetic DB
    print("Loading geodetic database...")
    geo_db = load_geodetic_db(base / 'Control_Points' / 'nikudot_bakara_slim.csv')
    print(f"  Loaded {len(geo_db)} points")

    # Discover maps across ground-truth series
    from data_paths import discover_map_dirs
    target = sys.argv[1] if len(sys.argv) > 1 else None
    map_dirs = discover_map_dirs(name_filter=target)

    # Process all maps
    print(f"\nAuto-georeferencing {len(map_dirs)} maps...\n")

    results = []
    for map_dir in map_dirs:
        t0 = time.time()
        result = auto_georeference_map(map_dir, geo_db, template_dir)
        elapsed = time.time() - t0

        if result:
            ea = result['eval_affine']
            er = result['eval_ransac']

            affine_rmse = f"{ea['rmse_m']:.1f}m" if ea else "N/A"
            ransac_rmse = f"{er['rmse_m']:.1f}m" if er else "N/A"

            print(f"  {result['map_name']:>15s}: "
                  f"{result['n_selected']:2d} pts ({result['n_inliers']} inliers) | "
                  f"Affine: {affine_rmse:>7s} | "
                  f"RANSAC: {ransac_rmse:>7s} | "
                  f"({elapsed:.1f}s)")
            results.append(result)

    # Summary
    if results:
        print(f"\n{'='*80}")
        print(f"  AUTOMATIC GEOREFERENCING SUMMARY")
        print(f"{'='*80}")
        print(f"\n{'Map':>15s} | {'Pts':>3s} | {'In':>2s} | "
              f"{'Affine RMSE':>12s} | {'RANSAC RMSE':>12s} | {'Max Err':>8s}")
        print('-' * 75)

        affine_rmses = []
        ransac_rmses = []

        for r in results:
            ea = r['eval_affine']
            er = r['eval_ransac']

            a_rmse = ea['rmse_m'] if ea else float('nan')
            r_rmse = er['rmse_m'] if er else float('nan')
            max_err = ea['max_error_m'] if ea else float('nan')

            print(f"{r['map_name']:>15s} | {r['n_selected']:3d} | "
                  f"{r['n_inliers']:2d} | "
                  f"{a_rmse:10.1f} m | {r_rmse:10.1f} m | "
                  f"{max_err:6.1f} m")

            if ea:
                affine_rmses.append(a_rmse)
            if er:
                ransac_rmses.append(r_rmse)

        print('-' * 75)
        if affine_rmses:
            print(f"{'MEDIAN':>15s} | {'':>3s} | {'':>2s} | "
                  f"{np.median(affine_rmses):10.1f} m | "
                  f"{np.median(ransac_rmses):10.1f} m |")
            print(f"{'MEAN':>15s} | {'':>3s} | {'':>2s} | "
                  f"{np.mean(affine_rmses):10.1f} m | "
                  f"{np.mean(ransac_rmses):10.1f} m |")

        # Save summary CSV
        summary_path = output_dir / 'auto_georef_summary.csv'
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['map_name', 'n_points', 'n_inliers',
                           'affine_rmse_m', 'ransac_rmse_m', 'max_error_m',
                           'fit_rmse_m'])
            for r in results:
                ea = r['eval_affine']
                er = r['eval_ransac']
                writer.writerow([
                    r['map_name'], r['n_selected'], r['n_inliers'],
                    f"{ea['rmse_m']:.2f}" if ea else '',
                    f"{er['rmse_m']:.2f}" if er else '',
                    f"{ea['max_error_m']:.2f}" if ea else '',
                    f"{r['affine']['rmse_meters']:.2f}",
                ])
        print(f"\nSummary saved to {summary_path}")
