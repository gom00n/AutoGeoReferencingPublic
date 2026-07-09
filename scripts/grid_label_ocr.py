"""
OCR reading of grid coordinate labels from scanned Palestine 1:20,000 maps.

Grid labels are 2-3 digit numbers printed at the map's neatline (inner border)
where grid lines meet the edge. They represent old Palestine grid values in km.

Layout:
- Easting labels along top and bottom edges (e.g., 140, 141, 142, ...)
- Northing labels along left and right edges (e.g., 130, 129, 128, ...)
- Labels are black text ~30-60px tall on cream/white background

The module:
1. Finds the neatline (map border rectangle)
2. Scans margin strips around the neatline for numbers via OCR
3. Validates labels form a consistent grid sequence
4. Returns (pixel_position, grid_coordinate_km) pairs
5. These pairs are sufficient to compute an initial affine transform

This enables georeferencing from ONLY the scan image — no filename or metadata.
"""

import cv2
import numpy as np
from pathlib import Path

from image_loader import load_image, suppress_red


# ---------------------------------------------------------------------------
# Lazy-initialized easyocr reader (shared across all OCR calls)
# ---------------------------------------------------------------------------
_ocr_reader = None


def _get_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _ocr_reader


# ---------------------------------------------------------------------------
# OCR digit confusions: digits that look similar in the map label font
# (each maps to all plausible misreads — a digit can be confused with
# more than one other)
# ---------------------------------------------------------------------------

DIGIT_CONFUSIONS = {
    '3': ['8'],
    '8': ['3', '0'],
    '5': ['6'],
    '6': ['5', '0'],
    '1': ['7'],
    '7': ['1'],
    '0': ['6', '8'],
    '9': ['4'],
    '4': ['9'],
}


def digit_confusion_variants(value):
    """All single-digit-substitution variants of a number (excluding itself)."""
    s = str(value)
    variants = []
    for i, ch in enumerate(s):
        for sub in DIGIT_CONFUSIONS.get(ch, []):
            variants.append(int(s[:i] + sub + s[i+1:]))
    return variants


def rescue_out_of_range(value, expected_range):
    """
    Try to recover an out-of-range OCR value via digit confusion.

    With a tight expected_range (from the sheet number), a systematic
    misread like "131" read as "181" lands outside the range. If exactly
    ONE digit-confusion variant falls inside, that's the real value.

    Returns the rescued value, or None if no variant (or more than one)
    is in range.
    """
    in_range = [v for v in set(digit_confusion_variants(value))
                if expected_range[0] <= v <= expected_range[1]]
    return in_range[0] if len(in_range) == 1 else None


# ---------------------------------------------------------------------------
# Step 1: Neatline detection
# ---------------------------------------------------------------------------

def find_neatline(image_bgr, margin=50):
    """
    Find the map's neatline (inner border rectangle).

    The neatline is a prominent dark rectangle enclosing the map content.
    It's typically the strongest rectangular feature in the image.

    Returns:
        dict with 'top', 'bottom', 'left', 'right' pixel positions
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Binary threshold for dark lines
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    # Project to find strong horizontal and vertical lines
    row_sums = binary.sum(axis=1).astype(np.float64) / 255.0
    col_sums = binary.sum(axis=0).astype(np.float64) / 255.0

    # The neatline should span nearly the full width/height
    h_threshold = w * 0.65
    v_threshold = h * 0.65

    strong_rows = np.where(row_sums > h_threshold)[0]
    strong_cols = np.where(col_sums > v_threshold)[0]

    def cluster_lines(positions):
        if len(positions) == 0:
            return []
        clusters = []
        current = [positions[0]]
        for p in positions[1:]:
            if p - current[-1] <= 3:
                current.append(p)
            else:
                clusters.append(int(np.mean(current)))
                current = [p]
        clusters.append(int(np.mean(current)))
        return clusters

    h_clusters = [c for c in cluster_lines(strong_rows)
                  if h * 0.03 < c < h * 0.97]
    v_clusters = [c for c in cluster_lines(strong_cols)
                  if w * 0.03 < c < w * 0.97]

    top = min((c for c in h_clusters if c < h * 0.15), default=int(h * 0.05))
    bottom = max((c for c in h_clusters if h * 0.70 < c < h * 0.92),
                 default=int(h * 0.85))
    left = min((c for c in v_clusters if c < w * 0.15), default=int(w * 0.05))
    right = max((c for c in v_clusters if c > w * 0.85), default=int(w * 0.95))

    return {'top': top, 'bottom': bottom, 'left': left, 'right': right}


# ---------------------------------------------------------------------------
# Step 1b: Grid line detection — find outermost grid lines inside neatline
# ---------------------------------------------------------------------------

def find_grid_bounds(image_bgr, neatline):
    """
    Detect grid lines inside the neatline and return the outermost ones.

    Grid lines are strong dark lines that span most of the neatline width/height.
    Triangulation markers live inside the grid; elements in the margin between
    the neatline and the first grid line are false positives.

    Returns:
        dict with 'top', 'bottom', 'left', 'right' pixel positions of the
        outermost grid lines, or None if detection fails.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    t, b, l, r = neatline['top'], neatline['bottom'], neatline['left'], neatline['right']
    nl_w = r - l
    nl_h = b - t

    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    def cluster_lines(positions, tolerance=5):
        if len(positions) == 0:
            return []
        clusters = []
        current = [positions[0]]
        for p in positions[1:]:
            if p - current[-1] <= tolerance:
                current.append(p)
            else:
                clusters.append(int(np.mean(current)))
                current = [p]
        clusters.append(int(np.mean(current)))
        return clusters

    def find_regular_grid(lines, min_lines=3):
        """From candidate lines, find the largest subset with regular spacing."""
        if len(lines) < min_lines:
            return lines
        spacings = np.diff(lines)
        if len(spacings) == 0:
            return lines
        # The dominant spacing is the median of all spacings
        median_sp = np.median(spacings)
        if median_sp < 50:
            return lines
        # Keep lines consistent with this spacing
        good = [lines[0]]
        for i in range(1, len(lines)):
            sp = lines[i] - good[-1]
            if 0.5 * median_sp <= sp <= 1.5 * median_sp:
                good.append(lines[i])
            elif sp > 1.5 * median_sp:
                n_steps = round(sp / median_sp)
                if n_steps >= 1 and abs(sp / n_steps - median_sp) / median_sp < 0.3:
                    good.append(lines[i])
        return good if len(good) >= min_lines else lines

    def detect_lines(sums, offset, span_length, neatline_lo, neatline_hi):
        """Detect grid lines from projection sums using adaptive threshold.

        Tries thresholds from 25% down to 15% of span_length, picking
        the highest threshold that yields >= 3 regular lines.
        """
        for pct in [0.25, 0.22, 0.20, 0.18, 0.15]:
            thresh = span_length * pct
            strong = np.where(sums > thresh)[0] + offset
            lines = cluster_lines(strong)
            lines = [p for p in lines if p > neatline_lo + 30 and p < neatline_hi - 30]
            regular = find_regular_grid(lines)
            if len(regular) >= 3:
                return regular
        return []

    # --- Horizontal grid lines ---
    row_sums = binary[t:b, l:r].sum(axis=1).astype(np.float64) / 255.0
    h_lines = detect_lines(row_sums, t, nl_w, t, b)

    # --- Vertical grid lines ---
    col_sums = binary[t:b, l:r].sum(axis=0).astype(np.float64) / 255.0
    v_lines = detect_lines(col_sums, l, nl_h, l, r)

    if len(h_lines) < 2 or len(v_lines) < 2:
        print(f"  Grid bounds: insufficient lines (h={len(h_lines)}, v={len(v_lines)})")
        return None

    grid_top, grid_bottom = min(h_lines), max(h_lines)
    grid_left, grid_right = min(v_lines), max(v_lines)

    # Sanity check: grid bounds should cover at least 50% of neatline in each axis
    grid_h = grid_bottom - grid_top
    grid_w = grid_right - grid_left
    if grid_h < nl_h * 0.5 or grid_w < nl_w * 0.5:
        print(f"  Grid bounds: too small ({grid_w}x{grid_h} vs neatline {nl_w}x{nl_h})")
        return None

    grid_bounds = {
        'top': grid_top,
        'bottom': grid_bottom,
        'left': grid_left,
        'right': grid_right,
        'h_lines': h_lines,
        'v_lines': v_lines,
    }
    print(f"  Grid bounds: T={grid_bounds['top']} B={grid_bounds['bottom']} "
          f"L={grid_bounds['left']} R={grid_bounds['right']} "
          f"({len(h_lines)}h x {len(v_lines)}v lines)")
    return grid_bounds


# ---------------------------------------------------------------------------
# Step 2: Margin strip extraction
# ---------------------------------------------------------------------------

def extract_margin_strips(image_bgr, neatline, margin_outer=180, margin_inner=500):
    """
    Extract the four margin strips around the neatline where grid labels appear.

    Labels sit at the neatline boundary — often 50-400px INSIDE the map,
    on or near grid lines. We take wide strips that span both outside and
    well inside the neatline. 500px inner margin needed for 1940s maps
    where labels sit at the first grid line, ~300px inside the neatline.

    Args:
        image_bgr: full map image
        neatline: dict with top/bottom/left/right
        margin_outer: pixels to extend outward from neatline
        margin_inner: pixels to extend inward from neatline (needs to be large
                      because labels sit on the first grid line inside the map)

    Returns:
        dict with 'top', 'bottom', 'left', 'right', each containing
        (crop_bgr, origin_x, origin_y)
    """
    h, w = image_bgr.shape[:2]
    t, b, l, r = neatline['top'], neatline['bottom'], neatline['left'], neatline['right']

    strips = {}

    # Top strip: labels are 50-150px BELOW the neatline, on the first grid line
    y1 = max(0, t - margin_outer)
    y2 = min(h, t + margin_inner)
    strips['top'] = (image_bgr[y1:y2, l:r].copy(), l, y1)

    # Bottom strip: labels are 50-150px ABOVE the bottom neatline
    # Don't extend far below — that's the legend
    y1 = max(0, b - margin_inner)
    y2 = min(h, b + 40)
    strips['bottom'] = (image_bgr[y1:y2, l:r].copy(), l, y1)

    # Left strip: labels are just inside the left neatline
    x1 = max(0, l - margin_outer)
    x2 = min(w, l + margin_inner)
    strips['left'] = (image_bgr[t:b, x1:x2].copy(), x1, t)

    # Right strip: labels are just inside the right neatline
    x1 = max(0, r - margin_inner)
    x2 = min(w, r + margin_outer)
    strips['right'] = (image_bgr[t:b, x1:x2].copy(), x1, t)

    return strips


# ---------------------------------------------------------------------------
# Step 3: OCR on margin strips
# ---------------------------------------------------------------------------

def ocr_margin_strip(strip_bgr, origin_x, origin_y, orientation,
                     expected_range=(50, 300), min_conf=0.2):
    """
    Run easyocr on an entire margin strip and return all valid grid numbers.

    Args:
        strip_bgr: cropped margin image
        origin_x, origin_y: top-left offset in full image coordinates
        orientation: 'horizontal' (top/bottom strips) or 'vertical' (left/right)
        expected_range: valid Palestine grid km range
        min_conf: minimum OCR confidence to keep

    Returns:
        list of (pixel_position, grid_value_km, confidence) sorted by position
    """
    reader = _get_reader()

    gray = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2GRAY)

    # Enhance contrast for better OCR
    # CLAHE helps with faded/low-contrast labels
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Run OCR on both original and enhanced, merge results
    results = reader.readtext(gray, allowlist='0123456789',
                              paragraph=False, min_size=10)
    results += reader.readtext(enhanced, allowlist='0123456789',
                               paragraph=False, min_size=10)

    detections = []
    for (bbox, text, conf) in results:
        text = text.strip()
        if not text or conf < min_conf:
            continue

        try:
            value = int(text)
        except ValueError:
            continue

        # Handle zone prefix: some maps print "1130" meaning 130km
        if value > expected_range[1] and (value % 1000) >= expected_range[0]:
            value = value % 1000

        if not (expected_range[0] <= value <= expected_range[1]):
            # Systematic misreads land out of range when expected_range is
            # tight (ErRamle: all tens-digit 3s read as 8s with conf 1.0)
            rescued = rescue_out_of_range(value, expected_range)
            if rescued is None:
                continue
            value = rescued
            conf = conf * 0.9

        # Compute center position of this detection in full-image coordinates
        # bbox is a list of 4 corner points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0

        if orientation == 'horizontal':
            pixel_pos = cx + origin_x  # x-center → column in full image
        else:
            pixel_pos = cy + origin_y  # y-center → row in full image

        detections.append((pixel_pos, value, conf))

    # Sort by position
    detections.sort(key=lambda d: d[0])

    # Deduplicate: if two detections at nearly the same position, keep best
    deduped = []
    for det in detections:
        if deduped and abs(det[0] - deduped[-1][0]) < 50:
            # Keep the one with higher confidence
            if det[2] > deduped[-1][2]:
                deduped[-1] = det
        else:
            deduped.append(det)

    return deduped


# ---------------------------------------------------------------------------
# Step 4: Grid sequence validation
# ---------------------------------------------------------------------------

def validate_grid_sequence(detections, expected_spacing_px=(500, 2500)):
    """
    Validate that detections form a consistent arithmetic grid sequence.

    Grid labels should be integers with consistent pixel spacing per km step.
    Values can increase or decrease with position (northing goes down on
    north-up maps).

    Args:
        detections: list of (pixel_position, grid_value_km, confidence)
        expected_spacing_px: (min, max) expected pixels per 1km grid step

    Returns:
        filtered list of (pixel_position, grid_value_km, confidence)
    """
    if len(detections) < 2:
        return detections

    # Compute implied pixels-per-km for all pairs where values differ
    # Use signed rate to detect direction (easting increases left→right,
    # northing decreases top→bottom for north-up maps)
    pair_rates = []
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            pos_i, val_i, _ = detections[i]
            pos_j, val_j, _ = detections[j]
            delta_val = val_j - val_i
            delta_pos = pos_j - pos_i

            if delta_val == 0:
                continue

            rate = delta_pos / delta_val  # signed pixels per km
            if expected_spacing_px[0] <= abs(rate) <= expected_spacing_px[1]:
                pair_rates.append(rate)

    if not pair_rates:
        return []

    # Use median signed rate — this captures the direction
    median_rate = np.median(pair_rates)

    # Score each detection: check consistency with all other detections
    scores = np.zeros(len(detections))
    for i in range(len(detections)):
        for j in range(len(detections)):
            if i == j:
                continue
            pos_i, val_i, _ = detections[i]
            pos_j, val_j, _ = detections[j]
            delta_val = val_j - val_i
            if delta_val == 0:
                continue
            delta_pos = pos_j - pos_i
            expected_pos = delta_val * median_rate
            error = abs(delta_pos - expected_pos)
            if abs(expected_pos) > 0:
                rel_error = error / abs(expected_pos)
                if rel_error < 0.30:  # within 30%
                    scores[i] += 1

    # Keep detections that are consistent with at least one other
    result = [det for det, score in zip(detections, scores) if score >= 1]

    return result


def fix_ocr_digit_confusion(detections, expected_spacing_px=(500, 2500)):
    """
    Fix common OCR digit confusions (3↔8, 5↔6, 1↔7, etc.).

    If a detection's value doesn't fit the grid sequence but a digit-swapped
    variant does, replace it. Example: [130, 181, 132, 133, 134] →
    the "8" should be a "3" → [130, 131, 132, 133, 134].

    When misreads outnumber correct reads (e.g., three 18X vs two 13X),
    the majority wins — the direction is ambiguous from spacing alone.

    Works by trying single-digit substitutions and picking the variant that
    produces the most consistent grid sequence.
    """
    if len(detections) < 3:
        return detections

    def gen_variants(value):
        """The value itself plus all single-digit-substitution variants."""
        return [value] + digit_confusion_variants(value)

    def score_sequence(vals, positions):
        """Score how well values form a consistent grid.

        Prefers sequences where:
        1. Adjacent pairs have consistent px/km rate
        2. Delta values are small integers (consecutive labels)
        3. Values are monotonic
        """
        if len(vals) < 2:
            return 0
        score = 0.0
        rates = []
        for i in range(len(vals) - 1):
            delta_val = vals[i+1] - vals[i]
            delta_pos = positions[i+1] - positions[i]
            if delta_val == 0:
                continue
            rate = abs(delta_pos / delta_val)
            if expected_spacing_px[0] <= rate <= expected_spacing_px[1]:
                rates.append(rate)
                score += 1.0
                # Bonus: prefer consecutive values (delta=1)
                if abs(delta_val) == 1:
                    score += 0.5

        # Bonus for low variance in rate (consistent spacing)
        if len(rates) >= 2:
            cv = np.std(rates) / np.mean(rates)  # coefficient of variation
            if cv < 0.1:
                score += 2.0
            elif cv < 0.2:
                score += 1.0

        return score

    positions = [d[0] for d in detections]
    original_vals = [d[1] for d in detections]
    confs = [d[2] for d in detections]

    # Two passes:
    # 1. Merge the two largest tens-groups (handles systematic misreads
    #    like all tens-digit 8s read as 3s — the minority group is folded
    #    into the majority).
    # 2. Greedy single-digit fixes on whatever still doesn't fit.
    best_vals = list(original_vals)
    best_score = score_sequence(best_vals, positions)
    if all(50 <= v <= 300 for v in original_vals):
        # Group by tens (e.g., 130-139 → 13, 180-189 → 18)
        tens_groups = {}
        for v in original_vals:
            tens = v // 10
            tens_groups[tens] = tens_groups.get(tens, 0) + 1

        # Find the two largest groups — they're likely the real values
        # and their confused variants
        groups = sorted(tens_groups.items(), key=lambda x: -x[1])
        if len(groups) >= 2:
            # Try merging top-2 groups in both directions
            for g_from, g_to in [(groups[1][0], groups[0][0]),
                                  (groups[0][0], groups[1][0])]:
                s_from = str(g_from)
                s_to = str(g_to)
                if len(s_from) != len(s_to):
                    continue
                diffs = sum(1 for a, b in zip(s_from, s_to) if a != b)
                if diffs != 1:
                    continue
                # Merge g_from toward g_to
                fixed_vals = []
                for v in original_vals:
                    if v // 10 == g_from:
                        units = v % 10
                        fixed_vals.append(g_to * 10 + units)
                    else:
                        fixed_vals.append(v)
                fixed_score = score_sequence(fixed_vals, positions)
                if fixed_score > best_score:
                    best_vals = fixed_vals
                    best_score = fixed_score

    # Then try individual fixes on remaining mismatches
    improved = True
    while improved:
        improved = False
        for i in range(len(best_vals)):
            current_score = score_sequence(best_vals, positions)
            for variant in gen_variants(best_vals[i]):
                if variant == best_vals[i]:
                    continue
                trial = list(best_vals)
                trial[i] = variant
                new_score = score_sequence(trial, positions)
                if new_score > current_score:
                    best_vals[i] = variant
                    improved = True
                    break

    # Rebuild detections with corrected values
    result = []
    for i in range(len(detections)):
        pos, old_val, conf = detections[i]
        new_val = best_vals[i]
        if new_val != old_val:
            conf = conf * 0.9
        result.append((pos, new_val, conf))

    return result


def merge_edge_detections(dets_a, edge_a, dets_b, edge_b,
                          max_position_spread=150):
    """
    Merge labels from two opposite edges (e.g., top + bottom for eastings).

    Labels for the same grid value should appear at similar pixel positions
    (within max_position_spread px — allows for slight scan rotation).
    Merging provides redundancy — if OCR misses one edge, the other
    compensates. If the two edges disagree on where a value sits, one of
    them is a misread: keep the higher-confidence one instead of averaging
    a position that's wrong for both.

    Returns:
        list of (pixel_position, grid_value_km, confidence, edge)
    """
    # Index by grid value
    by_value = {}
    for pos, val, conf in dets_a:
        by_value.setdefault(val, []).append((pos, conf, edge_a))
    for pos, val, conf in dets_b:
        by_value.setdefault(val, []).append((pos, conf, edge_b))

    merged = []
    for val, entries in by_value.items():
        positions = [e[0] for e in entries]
        if max(positions) - min(positions) > max_position_spread:
            pos, conf, edge = max(entries, key=lambda e: e[1])
            merged.append((pos, val, conf, edge))
            continue
        # Average position, max confidence
        avg_pos = np.mean(positions)
        max_conf = max(e[1] for e in entries)
        edge = entries[0][2] if len(entries) == 1 else 'both'
        merged.append((avg_pos, val, max_conf, edge))

    merged.sort(key=lambda x: x[0])
    return merged


# ---------------------------------------------------------------------------
# Step 5: Main pipeline
# ---------------------------------------------------------------------------

def read_grid_labels(image_bgr, expected_easting_range=(50, 300),
                     expected_northing_range=(50, 300)):
    """
    Full pipeline: find neatline, scan margins, OCR grid labels.

    Works from ONLY the scan image — no filename or external metadata needed.

    Args:
        image_bgr: full map image
        expected_easting_range: (min, max) km for easting labels
        expected_northing_range: (min, max) km for northing labels

    Returns:
        dict with:
            'easting_labels': list of (pixel_col, easting_km, confidence, edge)
            'northing_labels': list of (pixel_row, northing_km, confidence, edge)
            'neatline': neatline positions
            'h_grid_lines': [] (kept for backward compatibility)
            'v_grid_lines': [] (kept for backward compatibility)
    """
    h, w = image_bgr.shape[:2]
    print(f"  Image: {w}x{h}")

    # Step 1: Find neatline
    neatline = find_neatline(image_bgr)
    print(f"  Neatline: top={neatline['top']}, bottom={neatline['bottom']}, "
          f"left={neatline['left']}, right={neatline['right']}")

    # Step 2: Extract margin strips
    strips = extract_margin_strips(image_bgr, neatline)
    for name, (crop, ox, oy) in strips.items():
        print(f"  Strip '{name}': {crop.shape[1]}x{crop.shape[0]} at ({ox},{oy})")

    # Step 3: OCR each strip
    print("  OCR on margin strips...")

    top_dets = ocr_margin_strip(*strips['top'], 'horizontal', expected_easting_range)
    bottom_dets = ocr_margin_strip(*strips['bottom'], 'horizontal', expected_easting_range)
    left_dets = ocr_margin_strip(*strips['left'], 'vertical', expected_northing_range)
    right_dets = ocr_margin_strip(*strips['right'], 'vertical', expected_northing_range)

    print(f"    Raw detections: top={len(top_dets)} bottom={len(bottom_dets)} "
          f"left={len(left_dets)} right={len(right_dets)}")

    # Step 4: Fix OCR digit confusions, then validate each edge
    top_dets = fix_ocr_digit_confusion(top_dets)
    bottom_dets = fix_ocr_digit_confusion(bottom_dets)
    left_dets = fix_ocr_digit_confusion(left_dets)
    right_dets = fix_ocr_digit_confusion(right_dets)

    top_valid = validate_grid_sequence(top_dets)
    bottom_valid = validate_grid_sequence(bottom_dets)
    left_valid = validate_grid_sequence(left_dets)
    right_valid = validate_grid_sequence(right_dets)

    print(f"    After validation: top={len(top_valid)} bottom={len(bottom_valid)} "
          f"left={len(left_valid)} right={len(right_valid)}")

    # Step 5: Merge opposite edges
    easting_labels = merge_edge_detections(top_valid, 'top', bottom_valid, 'bottom')
    northing_labels = merge_edge_detections(left_valid, 'left', right_valid, 'right')

    # Log results
    for pos, val, conf, edge in easting_labels:
        print(f"    E: col={pos:7.0f}  {val}km  conf={conf:.2f}  ({edge})")
    for pos, val, conf, edge in northing_labels:
        print(f"    N: row={pos:7.0f}  {val}km  conf={conf:.2f}  ({edge})")

    print(f"  Final: {len(easting_labels)} easting, {len(northing_labels)} northing")

    return {
        'easting_labels': easting_labels,
        'northing_labels': northing_labels,
        'neatline': neatline,
        'h_grid_lines': [],
        'v_grid_lines': [],
    }


# ---------------------------------------------------------------------------
# Affine computation from labels
# ---------------------------------------------------------------------------

def remove_non_monotonic(labels):
    """Iteratively remove labels that violate monotonic ordering.

    Grid labels should be monotonically increasing or decreasing
    with position. Iteratively remove the worst violator."""
    if len(labels) < 3:
        return labels

    result = list(labels)
    # Determine expected direction by majority vote of adjacent steps.
    # Don't compare endpoints (vals[-1] > vals[0]) — a junk detection at
    # either end flips the direction and the real labels get removed.
    vals = [l[1] for l in result]
    ascending = np.sign(np.diff(vals)).sum() > 0

    for _ in range(len(labels) // 2):  # max iterations
        vals = [l[1] for l in result]
        # Count ordering violations against ALL other labels, not just
        # neighbors — one junk label is out of order with every real
        # label, while each real label only conflicts with the junk one.
        # (Adjacent-only counting ties 1-1 and can remove the real label.)
        violations = [0] * len(result)
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                out_of_order = (vals[j] <= vals[i]) if ascending \
                    else (vals[j] >= vals[i])
                if out_of_order:
                    violations[i] += 1
                    violations[j] += 1

        # Most violations first; break ties by dropping the label with
        # the LOWER OCR confidence (a misread next to a real label ties
        # 1-1 — confidence tells them apart)
        worst = max(range(len(violations)),
                    key=lambda i: (violations[i], -result[i][2]))
        if violations[worst] == 0:
            break  # all monotonic
        result.pop(worst)

    return result


def consensus_labels(labels, px_per_km=(500, 2500), tol_km=0.25, min_inliers=3):
    """Keep the largest subset of labels collinear in (position, value).

    Grid labels lie on a line  position = m*value + b  whose slope |m| is
    the map's pixel density (px per km). OCR junk — spurious low-value
    misreads picked up in the margins — does not lie on that line. A RANSAC
    consensus over candidate slopes from every label pair finds the real
    grid and discards the junk regardless of its sign or magnitude.

    This is more robust than remove_non_monotonic, whose direction vote is
    flipped by a cluster of junk (so it deletes the real labels instead) —
    the dominant bootstrap failure mode on M-named archival sheets, which
    have no sheet-number range to pre-filter the OCR. Falls back to
    remove_non_monotonic when no slope gathers min_inliers support.

    px_per_km bounds match validate_grid_sequence's expected_spacing_px
    (0.4–2.0 m/px). tol_km is generous: real labels fit to <0.05 km, junk
    that happens to land near a real grid line carries a different value so
    its predicted position is far away.
    """
    if len(labels) < max(min_inliers, 3):
        return remove_non_monotonic(labels)

    pos = np.array([l[0] for l in labels], dtype=np.float64)
    val = np.array([l[1] for l in labels], dtype=np.float64)
    best_mask, best_key = None, None
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if val[j] == val[i]:
                continue
            m = (pos[j] - pos[i]) / (val[j] - val[i])   # px per km, signed
            if not (px_per_km[0] <= abs(m) <= px_per_km[1]):
                continue
            b = pos[i] - m * val[i]
            resid = np.abs(pos - (m * val + b))
            inliers = resid <= tol_km * abs(m)
            cnt = int(inliers.sum())
            # most inliers wins; break ties by tighter total residual
            key = (cnt, -float(resid[inliers].sum()))
            if best_key is None or key > best_key:
                best_key, best_mask = key, inliers

    if best_mask is None or int(best_mask.sum()) < min_inliers:
        return remove_non_monotonic(labels)
    return [lab for lab, keep in zip(labels, best_mask) if keep]


def labels_to_affine(result):
    """
    Compute an approximate affine transform from grid labels.

    Accepts either the full result dict from read_grid_labels(), or
    (easting_labels, northing_labels) as two separate arguments for
    backward compatibility.

    Each easting label gives: pixel_col → easting_km * 1000 (meters)
    Each northing label gives: pixel_row → northing_km * 1000 (meters)

    Returns:
        dict with affine coefficients, or None
    """
    # Handle both call conventions
    if isinstance(result, dict):
        easting_labels = result['easting_labels']
        northing_labels = result['northing_labels']
    else:
        # Legacy: labels_to_affine(easting_labels, northing_labels)
        raise TypeError("labels_to_affine now takes the result dict from read_grid_labels()")

    if len(easting_labels) < 2 or len(northing_labels) < 2:
        return None

    # Drop OCR junk: keep the largest set of labels collinear in
    # (position, value) at a plausible pixel density. Robust to spurious
    # low-value misreads that flip remove_non_monotonic's direction vote.
    easting_labels = consensus_labels(easting_labels)
    northing_labels = consensus_labels(northing_labels)

    if len(easting_labels) < 2 or len(northing_labels) < 2:
        return None

    def fit_with_outlier_rejection(positions, values_km):
        """Iteratively fit and reject outliers using MAD-based threshold."""
        pos = np.array(positions, dtype=np.float64)
        vals = np.array(values_km, dtype=np.float64)
        mask = np.ones(len(pos), dtype=bool)

        for iteration in range(5):
            A = np.column_stack([pos[mask], np.ones(mask.sum())])
            coeffs, _, _, _ = np.linalg.lstsq(A, vals[mask], rcond=None)
            predicted = coeffs[0] * pos + coeffs[1]
            residuals = np.abs(vals - predicted)

            # Use MAD-based threshold: median + 3*MAD
            inlier_residuals = residuals[mask]
            med_res = np.median(inlier_residuals)
            mad = np.median(np.abs(inlier_residuals - med_res))
            threshold = max(med_res + 3 * max(mad, 0.5), 2.0)  # at least 2km

            new_mask = residuals < threshold
            if new_mask.sum() < 2:
                break
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask

        # Refit on the final mask so coeffs always match the returned inliers
        A = np.column_stack([pos[mask], np.ones(mask.sum())])
        coeffs, _, _, _ = np.linalg.lstsq(A, vals[mask], rcond=None)
        n_rejected = len(pos) - mask.sum()
        return coeffs, mask, n_rejected

    # Easting: fit col → easting_km, then convert to meters
    e_cols = [lab[0] for lab in easting_labels]
    e_vals_km = [lab[1] for lab in easting_labels]
    (a_e_km, offset_e_km), e_mask, e_rejected = fit_with_outlier_rejection(e_cols, e_vals_km)
    a_e = a_e_km * 1000.0
    offset_e = offset_e_km * 1000.0

    # Northing: fit row → northing_km
    n_rows = [lab[0] for lab in northing_labels]
    n_vals_km = [lab[1] for lab in northing_labels]
    (d_n_km, offset_n_km), n_mask, n_rejected = fit_with_outlier_rejection(n_rows, n_vals_km)
    d_n = d_n_km * 1000.0
    offset_n = offset_n_km * 1000.0

    if e_rejected > 0 or n_rejected > 0:
        print(f"    Outliers rejected: {e_rejected} easting, {n_rejected} northing")

    # Validate pixel size — should be ~0.5-3.0 m/px for these maps
    pixel_size_x = abs(a_e)
    pixel_size_y = abs(d_n)
    if not (0.3 <= pixel_size_x <= 10.0) or not (0.3 <= pixel_size_y <= 10.0):
        print(f"    WARNING: unusual pixel size {pixel_size_x:.3f} x {pixel_size_y:.3f} m/px")
        return None

    # Check pixel size ratio
    ratio = max(pixel_size_x, pixel_size_y) / min(pixel_size_x, pixel_size_y)
    if ratio > 3.0:
        print(f"    WARNING: pixel size ratio {ratio:.1f} too skewed")
        return None

    # Build affine: pixel (col, row) → old grid (easting, northing)
    a = a_e
    c = 0.0
    e = offset_e
    b = 0.0
    d = d_n
    f = offset_n

    forward = np.array([[a, c, e], [b, d, f]])
    M = np.array([[a, c], [b, d]])
    M_inv = np.linalg.inv(M)
    offset = np.array([e, f])

    # Compute residuals on inliers
    e_cols_arr = np.array(e_cols)
    e_vals_arr = np.array(e_vals_km) * 1000.0
    e_pred = a * e_cols_arr[e_mask] + offset_e
    e_err = np.sqrt(np.mean((e_vals_arr[e_mask] - e_pred)**2))

    n_rows_arr = np.array(n_rows)
    n_vals_arr = np.array(n_vals_km) * 1000.0
    n_pred = d * n_rows_arr[n_mask] + offset_n
    n_err = np.sqrt(np.mean((n_vals_arr[n_mask] - n_pred)**2))

    # A sound fit on km-spaced labels has RMSE well under 100m; anything
    # near a full grid step means the labels are garbage — don't hand a
    # useless affine to template matching downstream
    if e_err > 1000 or n_err > 1000:
        print(f"    WARNING: label fit RMSE too high "
              f"({e_err:.0f}, {n_err:.0f} m) — rejecting affine")
        return None

    print(f"    Affine: pixel_size={pixel_size_x:.3f}x{pixel_size_y:.3f} m/px, "
          f"RMSE=({e_err:.1f}, {n_err:.1f}) m")

    # The labels that actually survived remove_non_monotonic + MAD outlier
    # rejection. Callers (labels_to_grid_points, labels_to_old_grid_extent)
    # MUST use these, not the raw ocr labels — a single km-scale misread that
    # the fit already discarded would otherwise throw the old-grid affine off
    # by tens of km (it contradicts every real label).
    easting_labels_used = [lab for lab, keep in zip(easting_labels, e_mask) if keep]
    northing_labels_used = [lab for lab, keep in zip(northing_labels, n_mask) if keep]

    return {
        'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f,
        'forward': forward,
        'M': M,
        'M_inv': M_inv,
        'offset': offset,
        'pixel_size_x': pixel_size_x,
        'pixel_size_y': pixel_size_y,
        'easting_rmse_m': e_err,
        'northing_rmse_m': n_err,
        'coordinate_system': 'old_palestine_grid',
        'n_easting_labels': len(easting_labels),
        'n_northing_labels': len(northing_labels),
        'easting_labels_used': easting_labels_used,
        'northing_labels_used': northing_labels_used,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_against_tfwx(computed_affine, tfwx_path, image_size):
    """
    Compare computed old-grid affine against a known TFWX (which is in EPSG:6991).
    """
    from coord_converter import load_tfwx

    ref = load_tfwx(tfwx_path)
    w, h = image_size

    return {
        'computed_pixel_size_x': computed_affine['pixel_size_x'],
        'computed_pixel_size_y': computed_affine['pixel_size_y'],
        'reference_pixel_size_x': abs(ref['a']),
        'reference_pixel_size_y': abs(ref['d']),
        'pixel_size_error_x': abs(computed_affine['pixel_size_x'] - abs(ref['a'])),
        'pixel_size_error_y': abs(computed_affine['pixel_size_y'] - abs(ref['d'])),
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import time

    base = Path(__file__).resolve().parent.parent

    if len(sys.argv) > 1:
        # Accept a path to any image
        target = Path(sys.argv[1])
        if target.is_file():
            img_files = [target]
        elif target.is_dir():
            img_files = sorted(target.glob('*.jpg'))
        else:
            # Try as map name in T1/T2
            from data_paths import ground_truth_series, sheet_image_dirs
            for series_dir in ground_truth_series() + sheet_image_dirs():
                p = series_dir / target
                if p.exists():
                    img_files = sorted(p.glob('*.jpg')) if p.is_dir() else [p]
                    break
            else:
                print(f"Not found: {target}")
                sys.exit(1)
    else:
        # Default test
        from data_paths import WIKI_MAPS
        img_files = [WIKI_MAPS / '13-14-ErRamle-1948.jpg']

    for img_path in img_files:
        print(f"\n{'='*60}")
        print(f"  Grid Label OCR: {img_path.name}")
        print(f"{'='*60}")

        t0 = time.time()
        img = load_image(str(img_path))
        t_load = time.time() - t0

        t0 = time.time()
        result = read_grid_labels(img)
        t_ocr = time.time() - t0

        print(f"\n  Summary:")
        print(f"    Easting labels:  {len(result['easting_labels'])}")
        print(f"    Northing labels: {len(result['northing_labels'])}")
        print(f"    Time: {t_load:.1f}s load + {t_ocr:.1f}s OCR")

        affine = labels_to_affine(result)
        if affine:
            print(f"\n  Computed affine (old Palestine grid):")
            print(f"    Pixel size: {affine['pixel_size_x']:.4f} x "
                  f"{affine['pixel_size_y']:.4f} m/px")
            print(f"    RMSE: E={affine['easting_rmse_m']:.1f}m, "
                  f"N={affine['northing_rmse_m']:.1f}m")

            # Compare with TFWX if available
            tfwx_files = list(img_path.parent.glob('*.tfwx'))
            if tfwx_files:
                from PIL import Image
                Image.MAX_IMAGE_PIXELS = None
                with Image.open(str(img_path)) as pi:
                    w, h = pi.size
                comp = validate_against_tfwx(affine, tfwx_files[0], (w, h))
                print(f"\n  vs TFWX reference:")
                print(f"    Reference: {comp['reference_pixel_size_x']:.4f} x "
                      f"{comp['reference_pixel_size_y']:.4f} m/px")
                print(f"    Error: {comp['pixel_size_error_x']:.4f} x "
                      f"{comp['pixel_size_error_y']:.4f} m")
        else:
            print(f"\n  FAILED: could not compute affine")
