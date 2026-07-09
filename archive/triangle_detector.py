"""
Full-map triangle detection for Palestine Survey maps.

Detects triangulation point symbols (small upward-pointing triangles with
a dot inside) by scanning the entire map image, without requiring grid
coordinates or DB-guided projection.

Pipeline:
1. Preprocess image (suppress colours, enhance contrast)
2. Find neatline (map border) → restrict search area
3. Multi-scale template matching across the masked image
4. Non-maximum suppression
5. Verify each candidate (shape, dot, contrast)
6. OCR nearby text to read the point name and height
7. Match names to geodetic DB → (pixel, coordinate) pairs

Name format on maps:   12D / △ / 67.6
Name format in DB:     12/D  (Height: 67.57)
"""

import cv2
import numpy as np
import pickle
from pathlib import Path

# ---------------------------------------------------------------------------
# Trained classifier (HOG + SVM)
# ---------------------------------------------------------------------------

_classifier = None

def _load_classifier():
    global _classifier
    if _classifier is not None:
        return _classifier
    model_path = Path(__file__).parent / 'triangle_classifier.pkl'
    if not model_path.exists():
        return None
    with open(model_path, 'rb') as f:
        _classifier = pickle.load(f)
    return _classifier


def _classify_crops(gray_crops):
    """
    Classify a batch of grayscale crops using the trained HOG+SVM model.
    Returns array of probabilities (0-1) for each crop being a triangle.
    """
    from skimage.feature import hog

    clf_data = _load_classifier()
    if clf_data is None:
        return np.ones(len(gray_crops)) * 0.5  # fallback: neutral

    svm = clf_data['svm']
    scaler = clf_data['scaler']
    img_size = clf_data['img_size']
    hog_params = clf_data.get('hog_params', {
        'orientations': 12, 'pixels_per_cell': (8, 8),
        'cells_per_block': (2, 2),
    })

    features = []
    for crop in gray_crops:
        resized = cv2.resize(crop, (img_size, img_size)).astype(np.float32) / 255.0
        h = hog(resized, feature_vector=True,
                orientations=hog_params.get('orientations', 12),
                pixels_per_cell=hog_params.get('pixels_per_cell', (8, 8)),
                cells_per_block=hog_params.get('cells_per_block', (2, 2)))
        features.append(h)

    features = scaler.transform(np.array(features))
    probs = svm.predict_proba(features)[:, 1]
    return probs

# ---------------------------------------------------------------------------
# Synthetic template generation
# ---------------------------------------------------------------------------

def _make_triangle_template(size, line_width=1, with_dot=True):
    """Create a synthetic upward-pointing triangle template (white on black)."""
    img = np.zeros((size, size), dtype=np.uint8)
    half = size // 2
    margin = max(2, size // 8)

    # Triangle vertices: top-centre, bottom-left, bottom-right
    top = (half, margin)
    bl = (margin, size - margin)
    br = (size - margin, size - margin)

    pts = np.array([top, bl, br], dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=255, thickness=line_width)

    if with_dot:
        cx = half
        cy = int(margin + (size - 2 * margin) * 0.65)  # slightly below centre
        cv2.circle(img, (cx, cy), max(1, size // 12), 255, -1)

    return img


def make_templates(sizes=(18, 22, 26, 30), with_dot=True):
    """Generate a bank of synthetic triangle templates at various sizes."""
    templates = []
    for s in sizes:
        for lw in (1, 2):
            t = _make_triangle_template(s, line_width=lw, with_dot=with_dot)
            templates.append((t, s, lw))
    return templates


# ---------------------------------------------------------------------------
# Neatline masking
# ---------------------------------------------------------------------------

def _build_neatline_mask(shape, neatline, shrink=10):
    """Binary mask: 255 inside neatline, 0 outside."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    t = neatline['top'] + shrink
    b = neatline['bottom'] - shrink
    l = neatline['left'] + shrink
    r = neatline['right'] - shrink
    mask[t:b, l:r] = 255
    return mask


# ---------------------------------------------------------------------------
# Full-map template matching
# ---------------------------------------------------------------------------

def _match_template_masked(gray, template, mask, threshold=0.40):
    """Run matchTemplate and return peaks above threshold inside the mask."""
    th, tw = template.shape[:2]
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

    # Crop mask to match result dimensions (result is smaller by template size)
    half_h, half_w = th // 2, tw // 2
    mask_crop = mask[half_h:half_h + result.shape[0],
                     half_w:half_w + result.shape[1]]
    result[mask_crop == 0] = -1  # suppress outside neatline

    locs = np.where(result >= threshold)
    detections = []
    for y, x in zip(*locs):
        cx = x + half_w
        cy = y + half_h
        detections.append((cx, cy, float(result[y, x])))
    return detections


def _nms(detections, radius=12):
    """
    Greedy non-maximum suppression using a grid for O(n) performance.
    """
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d[2], reverse=True)

    # Grid-based NMS: bucket detections into cells of size `radius`
    occupied = {}  # (grid_x, grid_y) -> True
    keep = []
    r2 = radius * radius

    for x, y, c in dets:
        gx, gy = int(x // radius), int(y // radius)
        # Check nearby cells
        is_suppressed = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = (gx + dx, gy + dy)
                if key in occupied:
                    for ox, oy in occupied[key]:
                        if (x - ox) ** 2 + (y - oy) ** 2 < r2:
                            is_suppressed = True
                            break
                if is_suppressed:
                    break
            if is_suppressed:
                break

        if not is_suppressed:
            keep.append((x, y, c))
            key = (gx, gy)
            if key not in occupied:
                occupied[key] = []
            occupied[key].append((x, y))

    return keep


# ---------------------------------------------------------------------------
# Candidate verification
# ---------------------------------------------------------------------------

def _verify_triangle(gray_crop):
    """
    Quick sanity check on a small crop centred on a candidate detection.
    Returns a quality score 0-1 (higher = more likely a real triangle).

    Checks:
    - Sufficient dark pixels (triangle outline)
    - Contrast between centre and border region
    - Rough bilateral symmetry
    """
    if gray_crop is None or gray_crop.size == 0:
        return 0.0

    h, w = gray_crop.shape[:2]
    if h < 10 or w < 10:
        return 0.0

    # Adaptive threshold
    _, bw = cv2.threshold(gray_crop, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dark pixel fraction should be moderate (5-40%)
    dark_frac = np.sum(bw > 0) / bw.size
    if dark_frac < 0.03 or dark_frac > 0.50:
        return 0.0

    # Centre region should have a dot (darker than surroundings)
    cy, cx = h // 2, w // 2
    r = max(2, min(h, w) // 6)
    centre_mean = float(gray_crop[max(0, cy-r):cy+r, max(0, cx-r):cx+r].mean())
    border_mean = float(gray_crop.mean())
    contrast = (border_mean - centre_mean) / max(border_mean, 1)

    # Bilateral symmetry: compare left half to flipped right half
    left = bw[:, :w // 2]
    right = bw[:, (w + 1) // 2:]
    right_flip = cv2.flip(right, 1)
    min_w = min(left.shape[1], right_flip.shape[1])
    if min_w > 0:
        sym_score = 1.0 - np.sum(np.abs(left[:, :min_w].astype(float) -
                                         right_flip[:, :min_w].astype(float))) / (255 * left[:, :min_w].size)
    else:
        sym_score = 0.0

    # Combined score
    score = 0.0
    score += 0.3 * min(dark_frac / 0.15, 1.0)   # dark pixels present
    score += 0.3 * max(0, contrast)                # centre darker than average
    score += 0.4 * sym_score                       # symmetric
    return score


# ---------------------------------------------------------------------------
# OCR of point names and heights
# ---------------------------------------------------------------------------

_ocr_reader = None

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _ocr_reader


def _ocr_name(crop_bgr):
    """
    Read point name from a crop ABOVE the triangle.
    Expected format: digits + letter, e.g. '12D', '48C', '1357K'.
    Returns name string in DB format ('12/D') or None.
    """
    import re
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    reader = _get_ocr_reader()
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr

    results = reader.readtext(gray, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                               paragraph=False, min_size=8)

    for _, text, conf in sorted(results, key=lambda r: r[2], reverse=True):
        text = text.strip().upper()
        # Match pattern: digits followed by a single letter
        m = re.match(r'^(\d{1,4})([A-Z])$', text)
        if m and conf > 0.2:
            return f"{m.group(1)}/{m.group(2)}"

    return None


def _ocr_height(crop_bgr):
    """
    Read height value from a crop BELOW the triangle.
    Expected format: number with optional decimal, e.g. '67.6', '108.0'.
    Returns float or None.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    reader = _get_ocr_reader()
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr

    results = reader.readtext(gray, allowlist='0123456789.',
                               paragraph=False, min_size=8)

    for _, text, conf in sorted(results, key=lambda r: r[2], reverse=True):
        text = text.strip()
        try:
            val = float(text)
            if 0 < val < 2000 and conf > 0.2:  # reasonable height range
                return val
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Sliding window detection (primary — most robust)
# ---------------------------------------------------------------------------

def detect_triangles(image_bgr, neatline=None, min_classifier_prob=0.7,
                     stride=16, crop_size=120, nms_radius=20):
    """
    Detect triangle symbols using sliding window + HOG/SVM classifier.

    Slides a window across the image (inside neatline), classifies each
    position, and applies NMS. Slower than contour-based but much more
    robust because the classifier sees the same crop format as training.

    Args:
        image_bgr:      full-resolution map image
        neatline:        dict with top/bottom/left/right (auto-detected if None)
        min_classifier_prob: minimum SVM probability to accept
        stride:          sliding window step in pixels
        crop_size:       window size (must match training: 120)
        nms_radius:      NMS suppression radius

    Returns:
        list of dicts with pixel_x, pixel_y, classifier_prob
    """
    from skimage.feature import hog
    from grid_label_ocr import find_neatline

    h_img, w_img = image_bgr.shape[:2]

    # Load classifier
    clf_data = _load_classifier()
    if clf_data is None:
        raise RuntimeError("No trained classifier found")

    svm = clf_data['svm']
    scaler = clf_data['scaler']
    img_size = clf_data['img_size']
    hog_params = clf_data.get('hog_params', {
        'orientations': 12, 'pixels_per_cell': (8, 8),
        'cells_per_block': (2, 2),
    })

    # 1. Neatline
    if neatline is None:
        neatline = find_neatline(image_bgr)

    # 2. Preprocess — green channel
    green = image_bgr[:, :, 1]

    # 3. Pre-filter: only scan positions with sufficient dark content
    #    (avoids wasting time on blank areas)
    half = crop_size // 2

    # Build a quick dark-pixel density map at reduced resolution
    _, dark_mask = cv2.threshold(green, 160, 255, cv2.THRESH_BINARY_INV)
    # Downsample the mask to stride resolution for fast scanning
    ds = stride
    small_mask = cv2.resize(dark_mask, (w_img // ds, h_img // ds),
                             interpolation=cv2.INTER_AREA)
    # A window needs some dark pixels (triangle outline) — at least 5% of area
    min_dark_pixels = (crop_size // ds) ** 2 * 0.03

    # 4. Slide window
    raw_detections = []
    batch_crops = []
    batch_positions = []
    batch_size = 512  # process in batches for efficiency

    y_start = max(neatline['top'], half)
    y_end = min(neatline['bottom'], h_img - half)
    x_start = max(neatline['left'], half)
    x_end = min(neatline['right'], w_img - half)

    for y in range(y_start, y_end, stride):
        for x in range(x_start, x_end, stride):
            # Quick dark-pixel check on downsampled mask
            sy, sx = y // ds, x // ds
            sh, sw = crop_size // ds, crop_size // ds
            sy1 = max(0, sy - sh // 2)
            sy2 = min(small_mask.shape[0], sy + sh // 2)
            sx1 = max(0, sx - sw // 2)
            sx2 = min(small_mask.shape[1], sx + sw // 2)
            region = small_mask[sy1:sy2, sx1:sx2]
            if region.size > 0 and region.mean() < min_dark_pixels:
                continue

            crop = green[y - half:y + half, x - half:x + half]
            if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
                continue

            batch_crops.append(crop)
            batch_positions.append((x, y))

            if len(batch_crops) >= batch_size:
                probs = _classify_crops(batch_crops)
                for (bx, by), prob in zip(batch_positions, probs):
                    if prob >= min_classifier_prob:
                        raw_detections.append((bx, by, float(prob)))
                batch_crops = []
                batch_positions = []

    # Process remaining batch
    if batch_crops:
        probs = _classify_crops(batch_crops)
        for (bx, by), prob in zip(batch_positions, probs):
            if prob >= min_classifier_prob:
                raw_detections.append((bx, by, float(prob)))

    # 5. NMS
    nms_dets = _nms(raw_detections, radius=nms_radius)

    # 6. Format output
    verified = []
    for x, y, prob in nms_dets:
        verified.append({
            'pixel_x': x, 'pixel_y': y,
            'classifier_prob': prob,
        })

    verified.sort(key=lambda d: d['classifier_prob'], reverse=True)
    return verified


# ---------------------------------------------------------------------------
# Contour-based triangle detection (alternative)
# ---------------------------------------------------------------------------

def _is_upward_triangle(contour, approx):
    """Check if an approximated contour is an upward-pointing triangle."""
    if len(approx) != 3:
        return False

    pts = approx.reshape(3, 2)
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    top = sorted_by_y[0]
    bottom_left = sorted_by_y[1] if sorted_by_y[1][0] < sorted_by_y[2][0] else sorted_by_y[2]
    bottom_right = sorted_by_y[2] if sorted_by_y[1][0] < sorted_by_y[2][0] else sorted_by_y[1]

    base_centre_x = (bottom_left[0] + bottom_right[0]) / 2
    if abs(top[0] - base_centre_x) > (bottom_right[0] - bottom_left[0]) * 0.4:
        return False

    if top[1] >= min(bottom_left[1], bottom_right[1]):
        return False

    base_dy = abs(bottom_left[1] - bottom_right[1])
    base_dx = abs(bottom_right[0] - bottom_left[0])
    if base_dx > 0 and base_dy / base_dx > 0.35:
        return False

    height = min(bottom_left[1], bottom_right[1]) - top[1]
    if base_dx > 0:
        aspect = height / base_dx
        if aspect < 0.5 or aspect > 1.5:
            return False

    return True


def _has_dot_inside(gray, cx, cy, triangle_size):
    """
    Check if there's a small dark dot inside the triangle.
    The dot is typically in the lower-centre of the triangle.

    Returns True if a dot-like feature is found.
    """
    # The dot should be in the lower 60% of the triangle, centred
    r = max(2, triangle_size // 6)
    dot_y = cy + triangle_size // 6  # slightly below centre

    # Extract small region around expected dot position
    y1 = max(0, dot_y - r)
    y2 = min(gray.shape[0], dot_y + r)
    x1 = max(0, cx - r)
    x2 = min(gray.shape[1], cx + r)

    if y2 <= y1 or x2 <= x1:
        return False

    dot_region = gray[y1:y2, x1:x2]
    surround_r = r * 3

    sy1 = max(0, dot_y - surround_r)
    sy2 = min(gray.shape[0], dot_y + surround_r)
    sx1 = max(0, cx - surround_r)
    sx2 = min(gray.shape[1], cx + surround_r)
    surround = gray[sy1:sy2, sx1:sx2]

    if surround.size == 0 or dot_region.size == 0:
        return False

    # The dot should be darker than surrounding interior
    dot_mean = float(dot_region.mean())
    surround_mean = float(surround.mean())

    # Dot should be noticeably darker
    return dot_mean < surround_mean - 15


def detect_triangles_contour(image_bgr, neatline=None, min_classifier_prob=0.5,
                             min_area=15, max_area=500, nms_radius=20):
    """
    Detect triangle symbols using contour analysis + classifier verification.

    Much more precise than template matching because it directly checks
    for 3-vertex polygonal shapes of the right size and orientation.

    Args:
        image_bgr:      full-resolution map image
        neatline:        dict with top/bottom/left/right (auto-detected if None)
        min_classifier_prob: minimum SVM probability to accept
        min_area:        minimum contour area in pixels
        max_area:        maximum contour area in pixels
        nms_radius:      suppress duplicates within this radius

    Returns:
        list of dicts with pixel_x, pixel_y, classifier_prob, contour_area
    """
    from grid_label_ocr import find_neatline

    h_img, w_img = image_bgr.shape[:2]

    # Scale area thresholds by image resolution (defaults tuned at ~14k px
    # wide). Contour area grows with the SQUARE of resolution, so thresholds
    # scale proportionally: bigger image → bigger expected triangle area.
    # The 0.5x / 2.0x factors leave slack for size variation between sheets.
    res_scale = (w_img / 14000.0) ** 2
    scaled_min_area = max(10, int(min_area * res_scale * 0.5))
    scaled_max_area = int(max_area * res_scale * 2.0)

    # 1. Neatline mask
    if neatline is None:
        neatline = find_neatline(image_bgr)

    # 2. Preprocess: green channel (triangles are dark, roads are red)
    green = image_bgr[:, :, 1]

    # 3. Find upward-pointing triangle contours
    candidates = []

    for block_size in (21, 31, 51):
        binary = cv2.adaptiveThreshold(
            green, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, 8
        )

        contours, _ = cv2.findContours(binary, cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < scaled_min_area or area > scaled_max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw == 0 or bh == 0:
                continue
            if bw / bh < 0.4 or bw / bh > 2.5:
                continue

            cx = x + bw // 2
            cy = y + bh // 2
            if (cy < neatline['top'] or cy > neatline['bottom'] or
                    cx < neatline['left'] or cx > neatline['right']):
                continue

            # Try multiple epsilon values to get 3 vertices
            peri = cv2.arcLength(cnt, True)
            is_triangle = False
            for eps in (0.03, 0.04, 0.05, 0.06, 0.08):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 3:
                    if _is_upward_triangle(cnt, approx):
                        is_triangle = True
                    break

            if not is_triangle:
                continue

            # Check for dot inside the triangle
            tri_size = max(bw, bh)
            if _has_dot_inside(green, cx, cy, tri_size):
                candidates.append((cx, cy, area))

    # 4. NMS on candidates
    if not candidates:
        return []

    nms_cands = _nms([(c[0], c[1], c[2]) for c in candidates],
                      radius=nms_radius)

    # 5. Extract crops and classify with HOG+SVM
    crop_half = 60
    crops = []
    valid_indices = []
    for i, (x, y, area) in enumerate(nms_cands):
        y1, y2 = max(0, y - crop_half), min(h_img, y + crop_half)
        x1, x2 = max(0, x - crop_half), min(w_img, x + crop_half)
        crop = green[y1:y2, x1:x2]
        if crop.shape[0] >= 40 and crop.shape[1] >= 40:
            crops.append(crop)
            valid_indices.append(i)

    if not crops:
        return []

    probs = _classify_crops(crops)

    # 6. Filter
    verified = []
    for idx, prob in zip(valid_indices, probs):
        if prob >= min_classifier_prob:
            x, y, area = nms_cands[idx]
            verified.append({
                'pixel_x': x, 'pixel_y': y,
                'classifier_prob': float(prob),
                'contour_area': area,
            })

    verified.sort(key=lambda d: d['classifier_prob'], reverse=True)
    return verified


# ---------------------------------------------------------------------------
# Template-based detection (fallback)
# ---------------------------------------------------------------------------

def detect_triangles_template(image_bgr, neatline=None, threshold=0.45,
                              nms_radius=15, min_classifier_prob=0.5,
                              template_sizes=(20, 26), downscale=None):
    """
    Detect triangle symbols across the full map image.

    Uses template matching for candidate generation, then a trained
    HOG+SVM classifier to filter false positives.

    Args:
        image_bgr:  full-resolution colour map image
        neatline:   dict with top/bottom/left/right (auto-detected if None)
        threshold:  template matching confidence threshold (low = more candidates)
        nms_radius: NMS suppression radius in pixels
        min_classifier_prob: minimum classifier probability to keep a detection
        template_sizes: triangle template sizes to try (fewer = faster)
        downscale:  downscale factor for template matching (None = auto)

    Returns:
        list of dicts with keys:
            pixel_x, pixel_y, match_score, classifier_prob
    """
    from image_loader import suppress_red
    from grid_label_ocr import find_neatline

    h_full, w_full = image_bgr.shape[:2]

    # Auto-determine downscale: target ~4000px wide for speed
    if downscale is None:
        downscale = max(1.0, w_full / 4000.0)

    # 1. Preprocess — green channel suppresses red roads
    gray_full = suppress_red(image_bgr)

    # Downscale for template matching (much faster)
    if downscale > 1.0:
        new_w = int(w_full / downscale)
        new_h = int(h_full / downscale)
        gray = cv2.resize(gray_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Scale template sizes and nms_radius accordingly
        scaled_sizes = tuple(max(10, int(s / downscale)) for s in template_sizes)
        scaled_nms = max(5, int(nms_radius / downscale))
    else:
        gray = gray_full
        scaled_sizes = template_sizes
        scaled_nms = nms_radius

    # 2. Neatline mask (on downscaled image)
    if neatline is None:
        neatline = find_neatline(image_bgr)
    scaled_neatline = {
        'top': int(neatline['top'] / downscale),
        'bottom': int(neatline['bottom'] / downscale),
        'left': int(neatline['left'] / downscale),
        'right': int(neatline['right'] / downscale),
    }
    mask = _build_neatline_mask(gray.shape, scaled_neatline)

    # 3. Generate templates — just 2 sizes × 1 line width = fast
    templates = make_templates(sizes=scaled_sizes, with_dot=True)

    # 4. Template matching
    raw_detections = []
    for tmpl, sz, lw in templates:
        dets = _match_template_masked(gray, tmpl, mask, threshold=threshold)
        raw_detections.extend(dets)

    # 5. NMS on downscaled coordinates
    nms_dets = _nms(raw_detections, radius=scaled_nms)

    if not nms_dets:
        return []

    # 6. Map back to full resolution and extract crops for classifier
    crop_half = 60  # 120x120 crop (matches training data)
    crops = []
    valid_indices = []
    for i, (x, y, conf) in enumerate(nms_dets):
        # Scale back to full resolution
        fx = int(x * downscale)
        fy = int(y * downscale)
        y1, y2 = max(0, fy - crop_half), min(h_full, fy + crop_half)
        x1, x2 = max(0, fx - crop_half), min(w_full, fx + crop_half)
        crop = gray_full[y1:y2, x1:x2]
        if crop.shape[0] >= 40 and crop.shape[1] >= 40:
            crops.append(crop)
            valid_indices.append(i)

    # 7. Classify with trained HOG+SVM
    if crops:
        probs = _classify_crops(crops)
    else:
        return []

    # 8. Filter by classifier probability
    verified = []
    for idx, prob in zip(valid_indices, probs):
        if prob >= min_classifier_prob:
            x, y, conf = nms_dets[idx]
            # Full-resolution coordinates
            fx = int(x * downscale)
            fy = int(y * downscale)
            verified.append({
                'pixel_x': fx, 'pixel_y': fy,
                'match_score': conf, 'classifier_prob': float(prob),
            })

    # Sort by classifier probability
    verified.sort(key=lambda d: d['classifier_prob'], reverse=True)
    return verified


def identify_triangles(image_bgr, detections, ocr_above=60, ocr_below=50,
                        ocr_width=80):
    """
    For each detected triangle, OCR the name above and height below.

    Args:
        image_bgr:  full map image
        detections: list of dicts from detect_triangles()
        ocr_above:  pixels above triangle centre to search for name
        ocr_below:  pixels below triangle centre to search for height
        ocr_width:  half-width of OCR crop

    Returns:
        list of dicts adding 'name' and 'height' keys to each detection
    """
    h_img, w_img = image_bgr.shape[:2]
    results = []
    for det in detections:
        x, y = det['pixel_x'], det['pixel_y']

        # Name crop: region above the triangle
        ny1 = max(0, y - ocr_above)
        ny2 = max(0, y - 8)
        nx1 = max(0, x - ocr_width // 2)
        nx2 = min(w_img, x + ocr_width // 2)
        name_crop = image_bgr[ny1:ny2, nx1:nx2] if ny2 > ny1 else None

        # Height crop: region below the triangle
        hy1 = min(h_img, y + 8)
        hy2 = min(h_img, y + ocr_below)
        hx1 = max(0, x - ocr_width // 2)
        hx2 = min(w_img, x + ocr_width // 2)
        height_crop = image_bgr[hy1:hy2, hx1:hx2] if hy2 > hy1 else None

        name = _ocr_name(name_crop)
        height = _ocr_height(height_crop)

        det_copy = dict(det)
        det_copy['name'] = name
        det_copy['height'] = height
        results.append(det_copy)

    return results


def match_to_db(identified_detections, geo_db, height_tolerance=1.0):
    """
    Match identified triangles to geodetic DB entries by name (primary)
    or height (fallback).

    Args:
        identified_detections: list of dicts with 'name' and 'height'
        geo_db: list of GeoPoint objects
        height_tolerance: max height difference for fallback matching (metres)

    Returns:
        list of (detection_dict, GeoPoint) pairs for successful matches
    """
    # Build lookup indices
    name_index = {}
    for pt in geo_db:
        name_index[pt.name] = pt

    # Also index by rounded height for fallback
    height_index = {}
    for pt in geo_db:
        try:
            h = float(pt.height)
            key = round(h, 1)
            height_index.setdefault(key, []).append(pt)
        except (ValueError, TypeError):
            pass

    matched = []
    for det in identified_detections:
        geo_pt = None

        # Primary: match by name
        if det.get('name'):
            geo_pt = name_index.get(det['name'])

        # Fallback: match by height (only if unique within tolerance)
        if geo_pt is None and det.get('height') is not None:
            target_h = det['height']
            candidates = []
            for dh in np.arange(-height_tolerance, height_tolerance + 0.1, 0.1):
                key = round(target_h + dh, 1)
                candidates.extend(height_index.get(key, []))
            # Deduplicate
            seen = set()
            unique = []
            for c in candidates:
                if c.name not in seen:
                    seen.add(c.name)
                    unique.append(c)
            if len(unique) == 1:
                geo_pt = unique[0]

        if geo_pt is not None:
            matched.append((det, geo_pt))

    return matched


def visualize_detections(image_bgr, detections, output_path,
                          matched_names=None, thumbnail_width=2000):
    """
    Draw detected triangles on a thumbnail for review.

    Colour code:
        green  = matched to DB by name
        cyan   = identified (has name) but not matched
        yellow = detected but not identified
    """
    h, w = image_bgr.shape[:2]
    scale = thumbnail_width / w
    thumb = cv2.resize(image_bgr, (thumbnail_width, int(h * scale)))

    matched_names = matched_names or set()

    for det in detections:
        tx = int(det['pixel_x'] * scale)
        ty = int(det['pixel_y'] * scale)
        name = det.get('name')
        height = det.get('height')

        if name and name in matched_names:
            color = (0, 255, 0)    # green: matched
        elif name:
            color = (255, 255, 0)  # cyan: identified
        else:
            color = (0, 255, 255)  # yellow: detected only

        cv2.circle(thumb, (tx, ty), 5, color, 2)
        label_parts = []
        if name:
            label_parts.append(name)
        if height is not None:
            label_parts.append(f"h={height}")
        label = ' '.join(label_parts) if label_parts else f"{det.get('classifier_prob', det.get('confidence', 0)):.2f}"
        cv2.putText(thumb, label, (tx + 7, ty - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1)

    cv2.imwrite(str(output_path), thumb)
    return str(output_path)


# ---------------------------------------------------------------------------
# Convenience: full pipeline in one call
# ---------------------------------------------------------------------------

def detect_and_match(image_bgr, geo_db, neatline=None,
                      min_classifier_prob=0.7, nms_radius=20,
                      ocr_all=False, max_ocr=100):
    """
    Full pipeline: detect → identify (OCR) → match to DB.

    To save time, OCR is only run on the top *max_ocr* candidates
    (sorted by confidence).

    Args:
        image_bgr:  full map image
        geo_db:     list of GeoPoint objects
        neatline:   optional precomputed neatline
        min_classifier_prob: minimum classifier probability to keep
        ocr_all:    if True, OCR every detection (slow)
        max_ocr:    max detections to OCR (ignored if ocr_all)

    Returns:
        dict with:
            detections: all verified detections
            identified: detections with OCR results
            matched:    list of (detection, GeoPoint) pairs
            neatline:   neatline used
    """
    # Detect
    detections = detect_triangles(
        image_bgr, neatline=neatline,
        min_classifier_prob=min_classifier_prob, nms_radius=nms_radius,
    )

    # OCR top candidates
    to_ocr = detections if ocr_all else detections[:max_ocr]
    identified = identify_triangles(image_bgr, to_ocr)

    # Match
    matched = match_to_db(identified, geo_db)

    return {
        'detections': detections,
        'identified': identified,
        'matched': matched,
        'neatline': neatline,
    }
