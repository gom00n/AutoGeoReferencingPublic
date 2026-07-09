"""
DB-guided triangle detection and matching.

Instead of scanning the entire image, this module:
1. Loads the geodetic database and filters to points within the map extent
2. Projects each DB point to pixel coordinates using the TFWX transform
3. Checks each candidate location for triangle presence using local crop matching
4. Uses multiple templates and matching methods for robust detection

This eliminates false positives entirely (every detection is a known geodetic point)
and focuses the matching on small local crops for speed and accuracy.
"""

import cv2
import numpy as np
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

from coord_converter import load_tfwx, map_to_pixel, pixel_to_map, get_map_extent
from image_loader import load_image, suppress_red, suppress_colors, to_black_white


@dataclass
class GeoPoint:
    """A geodetic control point from the database."""
    name: str
    easting_6991: float  # EPSG:6991 Easting = CSV 'New_N' (Israeli convention)
    northing_6991: float  # EPSG:6991 Northing = CSV 'New_E' (Israeli convention)
    old_e: float
    old_n: float
    height: str

    @property
    def map_xy(self):
        """Map coordinates as used in TFWX (easting, northing)."""
        return (self.easting_6991, self.northing_6991)


@dataclass
class Detection:
    """A detected triangle at a known geodetic point location."""
    geo_point: GeoPoint
    pixel_x: float
    pixel_y: float
    confidence: float
    method: str  # 'grayscale', 'edge', 'combined'


def load_geodetic_db(csv_path):
    """
    Load geodetic points from CSV.

    CSV columns: Name, New_E, New_N, Old_E, Old_N, Height
    Israeli convention: New_E = EPSG:6991 Northing, New_N = EPSG:6991 Easting
    """
    points = []
    n_skipped = 0
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                points.append(GeoPoint(
                    name=row['Name'].strip(),
                    easting_6991=float(row['New_N']),   # SWAPPED
                    northing_6991=float(row['New_E']),   # SWAPPED
                    old_e=float(row['Old_E']) if row.get('Old_E') else 0.0,
                    old_n=float(row['Old_N']) if row.get('Old_N') else 0.0,
                    height=row.get('Height', '').strip(),
                ))
            except (ValueError, KeyError):
                n_skipped += 1
                continue
    if n_skipped:
        print(f"  WARNING: skipped {n_skipped} malformed rows in {Path(csv_path).name}")
    return points


def filter_points_to_extent(points, extent, margin=200):
    """
    Filter geodetic points to those within the map extent.

    Args:
        points: list of GeoPoint
        extent: dict from get_map_extent() with min_x, max_x, min_y, max_y
        margin: extra margin in meters

    Returns:
        list of GeoPoint within extent
    """
    filtered = []
    for p in points:
        if (extent['min_x'] - margin <= p.easting_6991 <= extent['max_x'] + margin and
            extent['min_y'] - margin <= p.northing_6991 <= extent['max_y'] + margin):
            filtered.append(p)
    return filtered


def _compute_dot_offset_y(gray_template):
    """Compute Y offset from template center to the estimated dot position.

    For upward-pointing triangles, the dot sits at approximately 2/3 of
    the triangle height from the apex (centroid of an equilateral triangle).
    Returns the offset in pixels (positive = downward).
    """
    _, binary = cv2.threshold(gray_template, 128, 255, cv2.THRESH_BINARY_INV)
    ys = np.where(binary > 0)[0]
    if len(ys) == 0:
        return 0.0
    apex_y = ys.min()
    base_y = ys.max()
    tri_height = base_y - apex_y
    if tri_height < 4:
        return 0.0
    estimated_dot_y = apex_y + tri_height * 2 / 3
    geo_center_y = gray_template.shape[0] / 2
    return estimated_dot_y - geo_center_y


def load_grayscale_templates(template_dir, series=None):
    """
    Load grayscale (red-suppressed) triangle templates.

    Args:
        template_dir: Path to templates folder
        series: 'T1', 'T2', or None (load all)

    Returns list of (name, gray_array, dot_dx, dot_dy) tuples.
    dot_dx/dot_dy is the offset from template center to the dot.
    All templates use a uniform dot offset (0, +3) to shift from
    the template geometric center down to the dot, since triangles
    point upward with the dot below center.
    """
    template_dir = Path(template_dir)
    templates = []

    # Dot offset: triangles point up, dot is slightly below geometric center.
    # Testing shows any offset > 0 worsens affine RMSE on training maps,
    # meaning template matching already finds positions close to the dot.
    DOT_DY = 0.0

    # Load real grayscale templates (red-suppressed)
    for f in sorted(template_dir.glob('final_*_color.png')):
        # Filter by series if specified
        if series == 'T1' and 'T2' in f.stem:
            continue
        if series == 'T2' and 'T2' not in f.stem:
            continue

        bgr = cv2.imread(str(f))
        if bgr is not None:
            # Apply same red suppression as image preprocessing
            b, g = bgr[:, :, 0].astype(np.int16), bgr[:, :, 1].astype(np.int16)
            gray = np.minimum(b, g).astype(np.uint8)
            templates.append((f.stem, gray, 0.0, DOT_DY))

    if not templates:
        # Fallback: use any available template
        for f in sorted(template_dir.glob('*.png')):
            if 'zoom' in f.stem or 'sheet' in f.stem or 'debug' in f.stem:
                continue
            gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if gray is not None and gray.shape[0] < 40:
                templates.append((f.stem, gray, 0.0, DOT_DY))
                if len(templates) >= 3:
                    break

    return templates


def match_crop_grayscale(crop_gray, templates):
    """
    Match templates against a small grayscale crop.
    Returns (best_confidence, best_template_name, best_offset_x, best_offset_y).
    Offset points to the dot center (corrected for template asymmetry).
    """
    best_conf = -1.0
    best_name = ''
    best_ox, best_oy = 0, 0

    for tmpl_entry in templates:
        tmpl_name, tmpl = tmpl_entry[0], tmpl_entry[1]
        dot_dx = tmpl_entry[2] if len(tmpl_entry) > 2 else 0.0
        dot_dy = tmpl_entry[3] if len(tmpl_entry) > 3 else 0.0

        th, tw = tmpl.shape[:2]
        ch, cw = crop_gray.shape[:2]

        if th >= ch or tw >= cw:
            continue

        result = cv2.matchTemplate(crop_gray, tmpl, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > best_conf:
            best_conf = max_val
            best_name = tmpl_name
            # Template center + dot offset correction
            best_ox = int(round(max_loc[0] + tw / 2 + dot_dx))
            best_oy = int(round(max_loc[1] + th / 2 + dot_dy))

    return best_conf, best_name, best_ox, best_oy


def match_crop_edges(crop_gray, templates):
    """
    Match using Canny edges — invariant to background brightness.
    """
    crop_edges = cv2.Canny(crop_gray, 40, 120)

    best_conf = -1.0
    best_name = ''
    best_ox, best_oy = 0, 0

    for tmpl_entry in templates:
        tmpl_name, tmpl = tmpl_entry[0], tmpl_entry[1]
        dot_dx = tmpl_entry[2] if len(tmpl_entry) > 2 else 0.0
        dot_dy = tmpl_entry[3] if len(tmpl_entry) > 3 else 0.0

        th, tw = tmpl.shape[:2]
        ch, cw = crop_edges.shape[:2]

        if th >= ch or tw >= cw:
            continue

        tmpl_edges = cv2.Canny(tmpl, 40, 120)
        if tmpl_edges.sum() == 0:
            continue

        result = cv2.matchTemplate(crop_edges, tmpl_edges, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > best_conf:
            best_conf = max_val
            best_name = tmpl_name
            best_ox = int(round(max_loc[0] + tw / 2 + dot_dx))
            best_oy = int(round(max_loc[1] + th / 2 + dot_dy))

    return best_conf, best_name, best_ox, best_oy


def find_triangle_centroid(crop_gray, search_cx, search_cy, search_radius=18):
    """Find the centroid of a triangle contour near (search_cx, search_cy).

    After template matching locates the triangle, this finds the actual
    contour and computes its centroid — which corresponds to the geodetic
    dot position inside the triangle.

    Returns (cx, cy, found) where cx/cy are in crop coordinates.
    """
    h, w = crop_gray.shape[:2]
    y1 = max(0, search_cy - search_radius)
    y2 = min(h, search_cy + search_radius)
    x1 = max(0, search_cx - search_radius)
    x2 = min(w, search_cx + search_radius)
    region = crop_gray[y1:y2, x1:x2]

    if region.size == 0:
        return search_cx, search_cy, False

    binary = cv2.adaptiveThreshold(region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 21, 8)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score = 0
    best_cx, best_cy = search_cx, search_cy
    found = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50 or area > 800:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue

        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        n_verts = len(approx)

        if n_verts < 3 or n_verts > 6:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 8 or bh < 8 or bw > 35 or bh > 35:
            continue

        score = 0
        if n_verts == 3:
            score = 1.0
        elif n_verts == 4:
            score = 0.7
        elif n_verts == 5:
            score = 0.4

        if score > best_score:
            best_score = score
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                best_cx = x1 + M['m10'] / M['m00']
                best_cy = y1 + M['m01'] / M['m00']
                found = True
            else:
                best_cx = x1 + x + bw / 2
                best_cy = y1 + y + bh / 2
                found = True

    return best_cx, best_cy, found


def find_dot_center(crop_gray, match_cx, match_cy, search_radius=6, max_shift=4.0):
    """Find the geodetic dot near the template match position.

    Searches a small window around (and slightly below) the match center
    for the darkest cluster of pixels, which is likely the geodetic dot.
    The shift is capped to avoid jumping to unrelated dark features.

    Args:
        crop_gray: grayscale crop image
        match_cx, match_cy: template match center in crop coordinates
        search_radius: half-size of search window
        max_shift: maximum allowed shift in pixels from match position

    Returns:
        (dot_cx, dot_cy, found) in crop coordinates
    """
    h, w = crop_gray.shape[:2]
    mcx, mcy = int(round(match_cx)), int(round(match_cy))

    # Search window biased 2px downward (dot is below triangle center)
    y1 = max(0, mcy - search_radius + 2)
    y2 = min(h, mcy + search_radius + 2)
    x1 = max(0, mcx - search_radius)
    x2 = min(w, mcx + search_radius)
    region = crop_gray[y1:y2, x1:x2]

    if region.size == 0:
        return match_cx, match_cy, False

    # Weight = inverted intensity (darker = heavier)
    weights = (255.0 - region.astype(np.float64)) ** 2
    total = weights.sum()
    if total == 0:
        return match_cx, match_cy, False

    ys, xs = np.mgrid[0:region.shape[0], 0:region.shape[1]]
    wcx = (xs * weights).sum() / total + x1
    wcy = (ys * weights).sum() / total + y1

    # Cap the shift
    dx = wcx - match_cx
    dy = wcy - match_cy
    dist = (dx**2 + dy**2) ** 0.5
    if dist > max_shift:
        scale = max_shift / dist
        wcx = match_cx + dx * scale
        wcy = match_cy + dy * scale

    return wcx, wcy, True


def verify_triangle_shape(crop_gray, center_x, center_y, search_radius=18):
    """
    Verify that a triangular shape exists near the center of the crop.

    Uses contour analysis on a small region around the match location.
    Returns a shape_score (0.0 to 1.0) and refined center (cx, cy).

    A real triangle symbol has:
    - A contour with ~3 vertices (approxPolyDP)
    - Area in range 60-500 sq px
    - Roughly equilateral aspect ratio
    - Apex pointing upward
    """
    ch, cw = crop_gray.shape[:2]

    # Extract small region around match center
    y1 = max(0, center_y - search_radius)
    y2 = min(ch, center_y + search_radius)
    x1 = max(0, center_x - search_radius)
    x2 = min(cw, center_x + search_radius)
    region = crop_gray[y1:y2, x1:x2]

    if region.size == 0:
        return 0.0, center_x, center_y

    # Adaptive threshold to find dark features
    binary = cv2.adaptiveThreshold(
        region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 8
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score = 0.0
    best_cx, best_cy = center_x, center_y

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40 or area > 600:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue

        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        n_verts = len(approx)

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 6 or h < 6 or w > 35 or h > 30:
            continue

        aspect = w / max(h, 1)

        # Score based on shape properties
        score = 0.0

        # Vertex count: 3 is best, 4-5 acceptable
        if n_verts == 3:
            score += 0.4
        elif n_verts == 4:
            score += 0.2
        elif n_verts == 5:
            score += 0.1

        # Aspect ratio: triangle should be roughly 0.7-1.8
        if 0.6 < aspect < 2.0:
            score += 0.2

        # Area in sweet spot (80-350 typical for these maps)
        if 60 < area < 400:
            score += 0.2

        # Circularity: triangles have low circularity (~0.6)
        circularity = 4 * np.pi * area / (peri * peri)
        if 0.3 < circularity < 0.8:
            score += 0.1

        # Apex check: topmost point should be near horizontal center
        topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        center_offset = abs(topmost[0] - (x + w // 2)) / max(w, 1)
        if center_offset < 0.3:
            score += 0.1

        if score > best_score:
            best_score = score
            # Refined center in crop coordinates
            best_cx = x1 + x + w // 2
            best_cy = y1 + y + h // 2

    return best_score, best_cx, best_cy


def _match_single_preproc(crop, templates, gray_threshold=0.45, edge_threshold=0.25):
    """Run grayscale + edge matching on a single preprocessed crop.
    Returns (best_conf, method, offset_x, offset_y)."""
    gray_conf, gray_name, gox, goy = match_crop_grayscale(crop, templates)
    edge_conf, edge_name, eox, eoy = match_crop_edges(crop, templates)

    if gray_conf >= edge_conf:
        best_conf = gray_conf
        method = 'grayscale'
        offset_x, offset_y = gox, goy
    else:
        best_conf = edge_conf
        method = 'edge'
        offset_x, offset_y = eox, eoy

    if gray_conf >= gray_threshold and edge_conf >= edge_threshold:
        best_conf = max(gray_conf, edge_conf) * 1.05
        method = 'combined'

    return best_conf, method, offset_x, offset_y


def verify_candidates(image_bgr, candidates, affine, templates,
                       crop_size=80, gray_threshold=0.45, edge_threshold=0.25,
                       color_mode='multi'):
    """
    For each candidate geodetic point, check if a triangle symbol exists
    at its projected pixel location.

    Uses multiple preprocessing methods and takes the best match.

    Args:
        image_bgr: full map image (BGR)
        candidates: list of GeoPoint within the map extent
        affine: TFWX affine transform dict
        templates: list of (name, gray_template) tuples
        crop_size: size of local search window (pixels)
        gray_threshold: minimum grayscale match confidence
        edge_threshold: minimum edge match confidence
        color_mode: 'multi' (best of all methods), 'suppress_red',
                    'suppress_colors', or 'black_white'

    Returns:
        list of Detection objects, sorted by confidence (descending)
    """
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

    h_img, w_img = list(preprocs.values())[0].shape[:2]
    half = crop_size // 2

    detections = []

    for point in candidates:
        px, py = map_to_pixel(point.easting_6991, point.northing_6991, affine)
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

        detections.append(Detection(
            geo_point=point,
            pixel_x=refined_px,
            pixel_y=refined_py,
            confidence=best_conf,
            method=best_method,
        ))

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def process_map(map_dir, geo_db, template_dir, use_tif=False):
    """
    Full pipeline: detect triangles on a single map.

    Args:
        map_dir: Path to map folder (contains .jpg, .tfwx, etc.)
        geo_db: list of GeoPoint (full database)
        template_dir: Path to templates folder
        use_tif: if True, use .tif instead of .jpg

    Returns:
        list of Detection objects
    """
    map_dir = Path(map_dir)
    map_name = map_dir.name

    # Find image and TFWX files
    if use_tif:
        img_files = list(map_dir.glob('*.tif'))
    else:
        img_files = list(map_dir.glob('*.jpg'))

    if not img_files:
        print(f"  No image found in {map_dir}")
        return []

    img_path = img_files[0]
    tfwx_files = list(map_dir.glob('*.tfwx'))
    if not tfwx_files:
        print(f"  No TFWX found in {map_dir}")
        return []

    tfwx_path = tfwx_files[0]

    # Load affine transform
    affine = load_tfwx(tfwx_path)

    # Get image dimensions (without loading full image)
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(img_path) as pil_img:
        w, h = pil_img.size

    # Get map extent and filter DB points
    extent = get_map_extent(affine, w, h)
    candidates = filter_points_to_extent(geo_db, extent)
    print(f"  {map_name}: {len(candidates)} DB points in extent")

    if not candidates:
        return []

    # Load image
    img = load_image(img_path)

    # Load templates — use all templates (both series) for best coverage
    templates = load_grayscale_templates(template_dir)
    if not templates:
        print(f"  WARNING: No templates found in {template_dir}")
        return []

    # Run verification
    detections = verify_candidates(img, candidates, affine, templates)

    return detections


def save_results_csv(detections, output_path):
    """Save detection results to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'point_name', 'confidence', 'method',
            'pixel_x', 'pixel_y',
            'easting_6991', 'northing_6991',
            'old_e', 'old_n', 'height'
        ])
        for d in detections:
            writer.writerow([
                d.geo_point.name, f'{d.confidence:.4f}', d.method,
                f'{d.pixel_x:.1f}', f'{d.pixel_y:.1f}',
                f'{d.geo_point.easting_6991:.3f}', f'{d.geo_point.northing_6991:.3f}',
                f'{d.geo_point.old_e:.2f}', f'{d.geo_point.old_n:.2f}',
                d.geo_point.height,
            ])


def visualize_results(image_bgr, detections, output_path,
                       conf_threshold=0.5, thumbnail_width=2000):
    """
    Draw detections on a map thumbnail.
    Green = high confidence, Yellow = medium, Red = low.
    """
    h, w = image_bgr.shape[:2]
    scale = thumbnail_width / w
    thumb = cv2.resize(image_bgr, (thumbnail_width, int(h * scale)))

    for d in detections:
        tx = int(d.pixel_x * scale)
        ty = int(d.pixel_y * scale)

        if d.confidence >= 0.7:
            color = (0, 255, 0)   # green
        elif d.confidence >= 0.5:
            color = (0, 255, 255) # yellow
        else:
            color = (0, 100, 255) # orange

        if d.confidence >= conf_threshold:
            cv2.circle(thumb, (tx, ty), 6, color, 2)
            label = f"{d.geo_point.name} ({d.confidence:.2f})"
            cv2.putText(thumb, label, (tx + 8, ty - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    cv2.imwrite(str(output_path), thumb)


# --- CLI: process a single map or all maps ---
if __name__ == '__main__':
    import sys
    import time

    base = Path(__file__).resolve().parent.parent
    template_dir = base / 'scripts' / 'templates'
    output_dir = base / 'output'
    output_dir.mkdir(exist_ok=True)

    # Load geodetic database (once)
    print("Loading geodetic database...")
    t0 = time.time()
    geo_db = load_geodetic_db(base / 'Control_Points' / 'nikudot_bakara_slim.csv')
    print(f"  Loaded {len(geo_db)} points in {time.time()-t0:.1f}s")

    # Discover all map folders across ground-truth series
    from data_paths import discover_map_dirs
    target = sys.argv[1] if len(sys.argv) > 1 else None
    map_dirs = discover_map_dirs(name_filter=target)

    print(f"\nProcessing {len(map_dirs)} maps...")

    all_results = {}
    for map_dir in map_dirs:
        map_name = map_dir.name
        print(f"\n{'='*60}")
        print(f"Processing: {map_name}")
        t0 = time.time()

        detections = process_map(map_dir, geo_db, template_dir)
        elapsed = time.time() - t0

        if detections:
            # Count by confidence level
            high = sum(1 for d in detections if d.confidence >= 0.7)
            med = sum(1 for d in detections if 0.5 <= d.confidence < 0.7)
            low = sum(1 for d in detections if d.confidence < 0.5)

            print(f"  Found {len(detections)} candidates: "
                  f"{high} high, {med} medium, {low} low confidence "
                  f"({elapsed:.1f}s)")

            # Save results to per-map subfolder
            map_output = output_dir / map_name
            map_output.mkdir(parents=True, exist_ok=True)
            save_results_csv(detections, map_output / f'{map_name}_detections.csv')

            # Save visualization
            img = load_image(map_dir / f'{map_name}.jpg')
            visualize_results(img, detections,
                             map_output / f'{map_name}_detected.png',
                             conf_threshold=0.4)

            # Print top detections
            print(f"  Top 10:")
            for d in detections[:10]:
                print(f"    {d.geo_point.name:>10s} h={d.geo_point.height:>7s} "
                      f"conf={d.confidence:.3f} ({d.method})")

            all_results[map_name] = detections

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for map_name, dets in all_results.items():
        high = sum(1 for d in dets if d.confidence >= 0.7)
        med = sum(1 for d in dets if 0.5 <= d.confidence < 0.7)
        total = len(dets)
        print(f"  {map_name:>15s}: {total:3d} candidates, {high:2d} high + {med:2d} medium confidence")
