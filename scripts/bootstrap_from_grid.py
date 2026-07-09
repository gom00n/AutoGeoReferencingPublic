"""
Bootstrap georeferencing from grid coordinate labels.

For maps WITHOUT existing TFWX world files:
1. Read grid line positions and labels from map margins (manually or via OCR)
2. Build a pixel → Old Palestine Grid affine transform
3. Filter geodetic DB points by Old Grid extent
4. Project DB points to pixels using Old Grid affine
5. Run template matching at each projected position
6. Compute final pixel → EPSG:6991 affine from matched points
7. Output a TFWX world file

This bridges the gap between "no world file" and "full auto-georeferencing".
"""

import re
import sys
import time
import numpy as np
import cv2
from pathlib import Path

from coord_converter import pixel_to_map, map_to_pixel, get_map_extent
from image_loader import load_image, suppress_red, suppress_colors, to_black_white
from db_matcher import (
    GeoPoint, Detection, load_geodetic_db,
    load_grayscale_templates, match_crop_grayscale, match_crop_edges,
    verify_triangle_shape, find_triangle_centroid, find_dot_center,
    _match_single_preproc,
)
from auto_georeference import (
    compute_affine_transform, select_best_points, evaluate_transform,
    write_tfwx,
)


def build_old_grid_affine(grid_points):
    """
    Build a pixel → Old Palestine Grid affine from grid control points.

    Args:
        grid_points: list of (pixel_x, pixel_y, old_e_m, old_n_m) tuples
                     where old_e_m and old_n_m are in meters

    Returns:
        affine dict compatible with coord_converter functions,
        mapping pixels to Old Palestine Grid (meters)
    """
    pixel_pts = np.array([(p[0], p[1]) for p in grid_points])
    map_pts = np.array([(p[2], p[3]) for p in grid_points])

    n = len(pixel_pts)
    # Build least-squares system for affine: map = A @ [px, py, 1]
    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    for i in range(n):
        px, py = pixel_pts[i]
        mx, my = map_pts[i]
        A[2*i]   = [px, 0, py, 0, 1, 0]
        A[2*i+1] = [0, px, 0, py, 0, 1]
        b[2*i]   = mx
        b[2*i+1] = my

    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, bv, c, d, e, f = params

    M = np.array([[a, c], [bv, d]])
    M_inv = np.linalg.inv(M)
    offset = np.array([e, f])

    forward = np.array([
        [a, c, e],
        [bv, d, f]
    ])

    return {
        'a': a, 'b': bv, 'c': c, 'd': d, 'e': e, 'f': f,
        'forward': forward,
        'M': M,
        'M_inv': M_inv,
        'offset': offset,
    }


def parse_sheet_number(filename):
    """
    Parse Palestine 1:20,000 sheet number from filename.

    Filenames follow the pattern:  CC-RR-Name-Year.jpg
    where CC = sheet column (easting), RR = sheet row (northing).
    Special cases like '1112-14' or '13-1415' combine two columns/rows.

    Each sheet covers a 10 km x 10 km area in Old Palestine Grid:
        old_n (easting on map)  = CC * 10,000  to  (CC+1) * 10,000  metres
        old_e (northing on map) = RR * 10,000 + 1,000,000  to  (RR+1) * 10,000 + 1,000,000

    Returns:
        (old_e_range, old_n_range) in metres, or None if unparseable
    """
    stem = Path(filename).stem  # e.g. '12-11-ElFaluje-1948'

    # Try normal pattern: CC-RR-...
    m = re.match(r'^(\d{2})-(\d{2})-', stem)
    if m:
        col, row = int(m.group(1)), int(m.group(2))
        old_n_min = col * 10_000
        old_n_max = (col + 1) * 10_000
        old_e_min = row * 10_000 + 1_000_000
        old_e_max = (row + 1) * 10_000 + 1_000_000
        return (old_e_min, old_e_max), (old_n_min, old_n_max)

    # Try combined-column pattern: CCCC-RR-...  (e.g. '1112-14')
    m = re.match(r'^(\d{2})(\d{2})-(\d{2})-', stem)
    if m:
        col1, col2, row = int(m.group(1)), int(m.group(2)), int(m.group(3))
        old_n_min = min(col1, col2) * 10_000
        old_n_max = (max(col1, col2) + 1) * 10_000
        old_e_min = row * 10_000 + 1_000_000
        old_e_max = (row + 1) * 10_000 + 1_000_000
        return (old_e_min, old_e_max), (old_n_min, old_n_max)

    # Try combined-row pattern: CC-RRRR-...  (e.g. '10-1112')
    m = re.match(r'^(\d{2})-(\d{2})(\d{2})-', stem)
    if m:
        col, row1, row2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        old_n_min = col * 10_000
        old_n_max = (col + 1) * 10_000
        old_e_min = min(row1, row2) * 10_000 + 1_000_000
        old_e_max = (max(row1, row2) + 1) * 10_000 + 1_000_000
        return (old_e_min, old_e_max), (old_n_min, old_n_max)

    return None


def sheet_label_ranges(filename, slack_km=1):
    """
    Expected grid-label value ranges (km) for a sheet, from its filename.

    Used to constrain OCR: a label printed on sheet CC-RR can only carry
    easting values CC*10..CC*10+10 and northing values RR*10..RR*10+10.
    Feeding these to read_grid_labels() rejects decade misreads (e.g. all
    "14X" read as "19X") at the source — without the constraint a
    systematic misread can win the majority vote and produce a
    consistent-but-wrong affine that passes every downstream sanity check.

    Returns:
        (easting_range_km, northing_range_km) with slack applied,
        or None if the filename has no parseable sheet number.
    """
    parsed = parse_sheet_number(filename)
    if parsed is None:
        return None
    old_e_range, old_n_range = parsed
    easting_range = (old_n_range[0] // 1000 - slack_km,
                     old_n_range[1] // 1000 + slack_km)
    northing_range = ((old_e_range[0] - 1_000_000) // 1000 - slack_km,
                      (old_e_range[1] - 1_000_000) // 1000 + slack_km)
    return easting_range, northing_range


def labels_to_grid_points(ocr_result, label_affine):
    """
    Convert OCR grid labels + fitted label affine into grid control points.

    Old Palestine Grid convention in the geodetic DB:
        old_n = map easting label (km) × 1000
        old_e = map northing label (km) × 1000 + 1,000,000

    Each label fixes one axis exactly; the cross-axis coordinate is placed
    on the neatline and estimated from the label affine (self-consistent
    with the fitted row/col → meters mapping).

    Args:
        ocr_result: dict from grid_label_ocr.read_grid_labels()
        label_affine: dict from grid_label_ocr.labels_to_affine()

    Returns:
        list of (pixel_x, pixel_y, old_e_m, old_n_m) tuples
    """
    neatline = ocr_result['neatline']
    # Use the labels the affine fit actually kept, not the raw OCR output —
    # otherwise a misread the fit already rejected re-poisons the old-grid
    # affine (see labels_to_affine). Fall back to raw labels only if an older
    # affine dict without the inlier lists is passed.
    easting_labels = label_affine.get('easting_labels_used',
                                      ocr_result['easting_labels'])
    northing_labels = label_affine.get('northing_labels_used',
                                       ocr_result['northing_labels'])
    grid_points = []
    for col, e_km, conf, edge in easting_labels:
        old_n = e_km * 1000
        row = neatline['top'] if edge == 'top' else neatline['bottom']
        old_e = label_affine['d'] * row + label_affine['f'] + 1_000_000
        grid_points.append((col, row, old_e, old_n))
    for row, n_km, conf, edge in northing_labels:
        old_e = n_km * 1000 + 1_000_000
        col = neatline['left'] if edge == 'left' else neatline['right']
        old_n = label_affine['a'] * col + label_affine['e']
        grid_points.append((col, row, old_e, old_n))
    return grid_points


def labels_to_old_grid_extent(ocr_result, label_affine=None, margin_m=1000):
    """
    Compute the Old Palestine Grid extent covered by OCR'd grid labels.

    Pass label_affine to use the labels the affine fit kept (recommended): a
    single km-scale misread left in the raw labels would otherwise inflate the
    extent by tens of km, dragging in the wrong slab of the geodetic DB.

    Returns:
        (old_e_range, old_n_range) as ((min, max), (min, max)) in meters
    """
    easting_labels = ocr_result['easting_labels']
    northing_labels = ocr_result['northing_labels']
    if label_affine is not None:
        easting_labels = label_affine.get('easting_labels_used', easting_labels)
        northing_labels = label_affine.get('northing_labels_used', northing_labels)
    e_values = [e_km * 1000 for _, e_km, _, _ in easting_labels]
    n_values = [n_km * 1000 + 1_000_000 for _, n_km, _, _ in northing_labels]
    old_n_range = (min(e_values) - margin_m, max(e_values) + margin_m)
    old_e_range = (min(n_values) - margin_m, max(n_values) + margin_m)
    return old_e_range, old_n_range


def filter_points_by_old_grid(points, old_e_min, old_e_max, old_n_min, old_n_max, margin=1000):
    """Filter geodetic DB points by Old Palestine Grid extent (in meters)."""
    filtered = []
    for p in points:
        if (old_e_min - margin <= p.old_e <= old_e_max + margin and
            old_n_min - margin <= p.old_n <= old_n_max + margin):
            filtered.append(p)
    return filtered


def verify_candidates_old_grid(image_bgr, candidates, old_grid_affine, templates,
                                crop_size=80, gray_threshold=0.45, edge_threshold=0.25,
                                color_mode='multi', cnn_model=None, cnn_threshold=0.5):
    """
    Like verify_candidates but projects DB points using Old Grid coordinates.

    Instead of point.easting_6991/northing_6991 with EPSG:6991 affine,
    uses point.old_e/old_n with Old Grid affine.

    Args:
        color_mode: 'multi' (default, best of all methods), 'suppress_red',
                    'suppress_colors', or 'black_white'
        cnn_model: optional trained TriangleCNN for filtering false positives
        cnn_threshold: minimum CNN probability to accept a detection
    """
    import torch

    # Prepare all preprocessing variants
    preprocs = {}
    if color_mode == 'multi':
        preprocs['sr'] = suppress_red(image_bgr)
        preprocs['sc'] = suppress_colors(image_bgr)
        preprocs['bw'] = to_black_white(image_bgr)
    elif color_mode == 'suppress_colors':
        preprocs['sc'] = suppress_colors(image_bgr)
    elif color_mode == 'black_white':
        preprocs['bw'] = to_black_white(image_bgr)
    else:
        preprocs['sr'] = suppress_red(image_bgr)

    # For CNN: use suppress_red grayscale (same as QA pipeline training data)
    if cnn_model is not None:
        cnn_gray = preprocs.get('sr')
        if cnn_gray is None:
            cnn_gray = suppress_red(image_bgr)

    h_img, w_img = list(preprocs.values())[0].shape[:2]
    half = crop_size // 2
    cnn_half = 32  # CNN expects 64x64 crops

    detections = []
    for point in candidates:
        # Project using OLD GRID coordinates
        px, py = map_to_pixel(point.old_e, point.old_n, old_grid_affine)
        px_i, py_i = int(round(px)), int(round(py))

        if (px_i - half < 0 or py_i - half < 0 or
            px_i + half >= w_img or py_i + half >= h_img):
            continue

        # Try all preprocessing methods, keep best
        best_conf = -1
        best_method = ''
        best_ox, best_oy = 0, 0

        for prep_name, preprocessed in preprocs.items():
            crop = preprocessed[py_i - half:py_i + half, px_i - half:px_i + half]
            conf, method, ox, oy = _match_single_preproc(
                crop, templates, gray_threshold, edge_threshold)
            if conf > best_conf:
                best_conf = conf
                best_method = f"{method}_{prep_name}" if color_mode == 'multi' else method
                best_ox, best_oy = ox, oy

        # Refine position: try to find the actual dot below the template center
        if best_conf >= gray_threshold:
            best_prep = list(preprocs.values())[0]
            for pn, pp in preprocs.items():
                if pn in best_method:
                    best_prep = pp
                    break
            dot_crop = best_prep[py_i - half:py_i + half, px_i - half:px_i + half]
            dot_cx, dot_cy, dot_found = find_dot_center(dot_crop, best_ox, best_oy)
            if dot_found:
                best_ox, best_oy = dot_cx, dot_cy

        refined_px = px_i - half + best_ox
        refined_py = py_i - half + best_oy

        # CNN verification: extract 64x64 crop at refined position
        if cnn_model is not None and best_conf >= gray_threshold:
            cx, cy = int(round(refined_px)), int(round(refined_py))
            if (cx - cnn_half >= 0 and cy - cnn_half >= 0 and
                    cx + cnn_half < w_img and cy + cnn_half < h_img):
                cnn_crop = cnn_gray[cy - cnn_half:cy + cnn_half,
                                    cx - cnn_half:cx + cnn_half]
                cnn_input = torch.from_numpy(
                    cnn_crop.astype(np.float32) / 255.0
                ).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    cnn_prob = torch.sigmoid(cnn_model(cnn_input).squeeze()).item()
                if cnn_prob < cnn_threshold:
                    best_conf = cnn_prob * 0.1  # demote, don't discard entirely
                    best_method = f"cnn_rejected({cnn_prob:.2f})"

        detections.append(Detection(
            geo_point=point,
            pixel_x=refined_px,
            pixel_y=refined_py,
            confidence=best_conf,
            method=best_method,
        ))

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def bootstrap_georeference(image_path, geo_db, template_dir, grid_points,
                            old_e_range, old_n_range, output_dir=None,
                            color_mode='suppress_red', image=None):
    """
    Full bootstrap georeferencing pipeline.

    Args:
        image_path: path to map image
        geo_db: full geodetic database
        template_dir: path to templates
        grid_points: list of (pixel_x, pixel_y, old_e_m, old_n_m) from grid labels
        old_e_range: (min, max) Old Grid easting in meters
        old_n_range: (min, max) Old Grid northing in meters
        output_dir: where to save results (default: same as image)
        image: optional preloaded BGR array (e.g. dpi-rescaled); when given,
            image_path is used only for naming and is not read

    Returns:
        dict with results
    """
    t0 = time.time()
    image_path = Path(image_path)
    map_name = image_path.stem
    if output_dir is None:
        output_dir = image_path.parent
    output_dir = Path(output_dir)

    print(f"\n{'='*60}")
    print(f"Bootstrap georeferencing: {map_name}")
    print(f"{'='*60}")

    # Step 1: Build Old Grid affine from grid labels
    print(f"\nStep 1: Building Old Grid affine from {len(grid_points)} grid points...")
    old_affine = build_old_grid_affine(grid_points)
    print(f"  Pixel size: ~{abs(old_affine['a']):.4f} x {abs(old_affine['d']):.4f} m/px")
    print(f"  Origin: ({old_affine['e']:.1f}, {old_affine['f']:.1f})")

    # Step 2: Filter DB points by Old Grid extent
    print(f"\nStep 2: Filtering DB points to Old Grid extent...")
    candidates = filter_points_by_old_grid(
        geo_db, old_e_range[0], old_e_range[1],
        old_n_range[0], old_n_range[1]
    )
    print(f"  {len(candidates)} DB points in Old Grid extent "
          f"E=[{old_e_range[0]/1000:.0f},{old_e_range[1]/1000:.0f}]km "
          f"N=[{old_n_range[0]/1000:.0f},{old_n_range[1]/1000:.0f}]km")

    # Step 3: Load image, templates, and CNN model
    print(f"\nStep 3: Loading image and templates...")
    img = image if image is not None else load_image(str(image_path))
    h_img, w_img = img.shape[:2]
    print(f"  Image: {w_img}x{h_img}")

    templates = load_grayscale_templates(template_dir)
    print(f"  Templates: {len(templates)}")

    # Load CNN for filtering false positives
    import torch
    from train_classifier import TriangleCNN
    cnn_model = TriangleCNN()
    cnn_path = Path(__file__).resolve().parent / 'triangle_classifier.pth'
    ckpt = torch.load(cnn_path, map_location='cpu', weights_only=False)
    cnn_model.load_state_dict(ckpt['model_state_dict'])
    cnn_model.eval()
    print(f"  CNN model: F1={ckpt['f1']:.3f}")

    # Step 4: Verify candidates using Old Grid projection + CNN
    print(f"\nStep 4: Template matching + CNN at {len(candidates)} projected positions...")
    print(f"  Color mode: {color_mode}")
    detections = verify_candidates_old_grid(img, candidates, old_affine, templates,
                                             color_mode=color_mode,
                                             cnn_model=cnn_model)

    # Step 5: Select best points
    print(f"\nStep 5: Selecting best points...")
    selected = select_best_points(detections, n_points=50, min_conf=0.75, min_distance=200)
    print(f"  {len(selected)} high-confidence, well-distributed detections")

    if len(selected) < 3:
        print(f"  ERROR: Need at least 3 points, got {len(selected)}")
        return None

    # Show top detections
    for i, det in enumerate(selected[:10]):
        print(f"    {i+1}. {det.geo_point.name:>8s} conf={det.confidence:.3f} "
              f"px=({det.pixel_x:.0f},{det.pixel_y:.0f}) "
              f"old=({det.geo_point.old_e:.0f},{det.geo_point.old_n:.0f}) "
              f"6991=({det.geo_point.easting_6991:.0f},{det.geo_point.northing_6991:.0f})")

    # Step 6: Compute EPSG:6991 affine from matched points
    print(f"\nStep 6: Computing EPSG:6991 affine transform...")
    pixel_pts = np.array([(d.pixel_x, d.pixel_y) for d in selected])
    map_pts = np.array([(d.geo_point.easting_6991, d.geo_point.northing_6991)
                        for d in selected])

    affine_6991 = compute_affine_transform(pixel_pts, map_pts)
    print(f"  a={affine_6991['a']:.6f}, d={affine_6991['d']:.6f}")
    print(f"  Pixel size: ~{abs(affine_6991['a']):.3f} x {abs(affine_6991['d']):.3f} m")
    print(f"  Origin: ({affine_6991['e']:.1f}, {affine_6991['f']:.1f})")

    # Step 7: Evaluate fit quality (internal RMSE)
    predicted = np.array([pixel_to_map(px, py, affine_6991)
                          for px, py in pixel_pts])
    errors = np.sqrt(np.sum((predicted - map_pts)**2, axis=1))
    fit_rmse = np.sqrt(np.mean(errors**2))
    max_err = np.max(errors)
    print(f"  Internal fit: RMSE={fit_rmse:.1f}m, max={max_err:.1f}m")

    # Step 8: RANSAC — reject outliers and recompute
    print(f"\nStep 7: RANSAC outlier rejection...")
    threshold = max(15.0, fit_rmse * 2)
    inlier_mask = errors < threshold
    n_inliers = np.sum(inlier_mask)
    print(f"  Threshold: {threshold:.1f}m, inliers: {n_inliers}/{len(selected)}")

    if n_inliers >= 3 and n_inliers < len(selected):
        affine_6991 = compute_affine_transform(
            pixel_pts[inlier_mask], map_pts[inlier_mask])

        predicted2 = np.array([pixel_to_map(px, py, affine_6991)
                               for px, py in pixel_pts[inlier_mask]])
        errors2 = np.sqrt(np.sum((predicted2 - map_pts[inlier_mask])**2, axis=1))
        fit_rmse = np.sqrt(np.mean(errors2**2))
        max_err = np.max(errors2)
        print(f"  After RANSAC: RMSE={fit_rmse:.1f}m, max={max_err:.1f}m")

    # Step 8: Write TFWX
    tfwx_path = output_dir / f"{map_name}.tfwx"
    write_tfwx(affine_6991, tfwx_path)
    print(f"\nOutput: {tfwx_path}")

    # Step 9: Visualization
    vis_path = output_dir / f"{map_name}_bootstrap.jpg"
    vis = cv2.resize(img, (img.shape[1]//8, img.shape[0]//8))
    scale = 1.0 / 8
    for det in selected:
        x, y = int(det.pixel_x * scale), int(det.pixel_y * scale)
        color = (0, 255, 0) if det.confidence >= 0.6 else (0, 255, 255)
        cv2.circle(vis, (x, y), 6, color, 2)
        cv2.putText(vis, f"{det.geo_point.name}", (x+8, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    cv2.imwrite(str(vis_path), vis)
    print(f"Visualization: {vis_path}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Points: {len(selected)} ({n_inliers} inliers)")
    print(f"  Fit RMSE: {fit_rmse:.1f}m")

    return {
        'affine': affine_6991,
        'n_points': len(selected),
        'n_inliers': int(n_inliers),
        'fit_rmse_m': fit_rmse,
        'max_error_m': max_err,
        'tfwx_path': str(tfwx_path),
        'detections': selected,
    }


# ============================================================
# Test on M5_4598 (BUREIJ, sheet 14/12)
# ============================================================
if __name__ == '__main__':
    base = Path(__file__).resolve().parent.parent
    template_dir = base / 'scripts' / 'templates'

    print("Loading geodetic database...")
    geo_db = load_geodetic_db(base / 'Control_Points' / 'nikudot_bakara_slim.csv')
    print(f"  Loaded {len(geo_db)} points")

    # ---- M5_4598 grid labels (read from margin inspection) ----
    # Left neatline: col ~891, easting = 140,000 (Old Grid meters)
    # Top neatline:  row ~934, northing = 130,000
    # Grid spacing: ~1183 px/km (easting), ~1187 px/km (northing)
    #
    # Verified labels:
    #   Row 934,  left margin: N=130  -> (891, 934) -> (140000, 130000)
    #   Row 2115, left margin: N=129  -> (891, 2115) -> (140000, 129000)
    #   Row 3308, left margin: N=128  -> (891, 3308) -> (140000, 128000)
    #   Col 10354, top margin: E=148  -> (10354, 934) -> (148000, 130000)
    #   Col 11537, top margin: E=149  -> (11537, 934) -> (149000, 130000)
    #   Col 12721, top margin: E=150  -> (12721, 934) -> (150000, 130000)
    #
    # Cross-check: col 891 = E=140, col 12721 = E=150
    #   (12721 - 891) / (150 - 140) = 11830 / 10 = 1183 px/km ✓

    # Old Palestine Grid convention in the DB:
    #   old_n = map easting label × 1000     (140 → 140,000)
    #   old_e = map northing label × 1000 + 1,000,000  (130 → 1,130,000)
    #
    # Verified: Old_N ≈ EPSG:6991_Easting - 50,000
    #           Old_E ≈ EPSG:6991_Northing + 500,000
    #
    # Grid labels on the map:
    #   Easting (horizontal): 140, 141, ..., 150  → old_n
    #   Northing (vertical):  120, 121, ..., 130   → old_e (with 1M prefix)

    grid_points = [
        # (pixel_x, pixel_y, old_e_m, old_n_m)
        # old_e = northing_label * 1000 + 1_000_000
        # old_n = easting_label * 1000
        (891,   934,  1130000, 140000),  # top-left: N=130, E=140
        (891,   2115, 1129000, 140000),  # left edge, N=129
        (891,   3308, 1128000, 140000),  # left edge, N=128
        (10354, 934,  1130000, 148000),  # top edge, E=148
        (11537, 934,  1130000, 149000),  # top edge, E=149
        (12721, 934,  1130000, 150000),  # top edge, E=150
        # Estimated corners for better affine:
        (12721, 12804, 1120000, 150000),  # bottom-right: N≈120, E=150
        (891,   12804, 1120000, 140000),  # bottom-left: N≈120, E=140
    ]

    old_e_range = (1120000, 1130000)  # Old_E: northing 120-130 + 1M
    old_n_range = (140000, 150000)    # Old_N: easting 140-150

    output_dir = base / 'Control_Maps' / 'M5_4598'
    output_dir.mkdir(parents=True, exist_ok=True)

    result = bootstrap_georeference(
        base / 'Control_Maps' / 'M5_4598' / 'M5_4598.jpg',
        geo_db, template_dir,
        grid_points, old_e_range, old_n_range,
        output_dir=output_dir,
    )

    if result:
        print(f"\n{'='*60}")
        print(f"SUCCESS: {result['n_points']} points, "
              f"fit RMSE = {result['fit_rmse_m']:.1f}m")
        print(f"TFWX: {result['tfwx_path']}")
        print(f"{'='*60}")
