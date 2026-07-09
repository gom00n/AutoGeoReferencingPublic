"""
Triangle detection via multi-scale template matching.

Detects △ triangulation point symbols on scanned maps using:
1. Multi-scale normalized cross-correlation (OpenCV matchTemplate)
2. Non-maximum suppression to eliminate duplicate detections
3. Optional contour-based detection as supplementary method
"""

import cv2
import numpy as np
from pathlib import Path


def load_templates(template_dir):
    """
    Load all template images from the templates directory.

    Returns:
        list of (name, gray_template) tuples
    """
    template_dir = Path(template_dir)
    templates = []

    # Load synthetic templates (binary, white triangle on black)
    for f in sorted(template_dir.glob('synthetic_*_nodot.png')):
        tmpl = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if tmpl is not None:
            templates.append((f.stem, tmpl))

    for f in sorted(template_dir.glob('synthetic_*_s?.png')):
        tmpl = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if tmpl is not None:
            templates.append((f.stem, tmpl))

    # Load real templates (binary versions)
    for f in sorted(template_dir.glob('final_*_binary.png')):
        tmpl = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if tmpl is not None:
            templates.append((f.stem, tmpl))

    if not templates:
        raise FileNotFoundError(f"No templates found in {template_dir}")

    print(f"Loaded {len(templates)} templates: {[t[0] for t in templates]}")
    return templates


def match_template_multiscale(image_gray, template, scales, threshold=0.5,
                               method=cv2.TM_CCOEFF_NORMED):
    """
    Run template matching at multiple scales.

    Args:
        image_gray: grayscale or binary image to search
        template: grayscale or binary template
        scales: list of scale factors (e.g. [0.8, 1.0, 1.2])
        threshold: minimum match confidence
        method: OpenCV template matching method

    Returns:
        list of (x, y, confidence, scale, template_w, template_h) detections
    """
    detections = []
    th, tw = template.shape[:2]

    for scale in scales:
        if scale != 1.0:
            new_w = max(int(tw * scale), 5)
            new_h = max(int(th * scale), 5)
            tmpl_scaled = cv2.resize(template, (new_w, new_h),
                                      interpolation=cv2.INTER_LINEAR)
        else:
            tmpl_scaled = template
            new_w, new_h = tw, th

        # Skip if template is larger than image
        if new_h >= image_gray.shape[0] or new_w >= image_gray.shape[1]:
            continue

        result = cv2.matchTemplate(image_gray, tmpl_scaled, method)

        # Find locations above threshold
        locations = np.where(result >= threshold)
        for py, px in zip(*locations):
            confidence = result[py, px]
            # Report center of matched region
            cx = px + new_w // 2
            cy = py + new_h // 2
            detections.append((cx, cy, float(confidence), scale, new_w, new_h))

    return detections


def non_max_suppression(detections, min_distance=15):
    """
    Remove duplicate/overlapping detections, keeping the highest confidence.

    Args:
        detections: list of (x, y, confidence, ...) tuples
        min_distance: minimum pixel distance between kept detections

    Returns:
        filtered list of detections
    """
    if not detections:
        return []

    # Sort by confidence (descending)
    sorted_dets = sorted(detections, key=lambda d: d[2], reverse=True)
    kept = []

    for det in sorted_dets:
        x, y = det[0], det[1]
        # Check if too close to an already-kept detection
        is_duplicate = False
        for kx, ky, *_ in kept:
            dist = np.sqrt((x - kx) ** 2 + (y - ky) ** 2)
            if dist < min_distance:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(det)

    return kept


def detect_triangles(image_gray, binary_image, templates,
                     scales=(0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
                     threshold=0.45, nms_distance=20, content_mask=None):
    """
    Main triangle detection function.

    Runs multi-scale template matching with all templates,
    then applies NMS to merge results.

    Args:
        image_gray: red-suppressed grayscale image
        binary_image: binarized version (dark features = white)
        templates: list of (name, template_array) from load_templates()
        scales: scale factors for multi-scale matching
        threshold: minimum match confidence
        nms_distance: minimum distance between detections (pixels)
        content_mask: optional mask (255=search area, 0=skip)

    Returns:
        list of (x, y, confidence, scale, template_name) detections
    """
    all_detections = []

    for tmpl_name, tmpl in templates:
        print(f"  Matching template: {tmpl_name} ({tmpl.shape[1]}x{tmpl.shape[0]})")

        # Match on binary image (more robust to background variation)
        dets = match_template_multiscale(binary_image, tmpl, scales, threshold)

        for x, y, conf, scale, tw, th in dets:
            # Apply content mask if provided
            if content_mask is not None:
                if y < 0 or y >= content_mask.shape[0] or x < 0 or x >= content_mask.shape[1]:
                    continue
                if content_mask[y, x] == 0:
                    continue
            all_detections.append((x, y, conf, scale, tmpl_name))

    print(f"  Raw detections (before NMS): {len(all_detections)}")

    # NMS across all templates
    filtered = non_max_suppression(all_detections, nms_distance)
    print(f"  After NMS: {len(filtered)} detections")

    return filtered


def detect_triangles_contour(binary_image, content_mask=None,
                              min_area=50, max_area=500):
    """
    Supplementary triangle detection using contour analysis.

    Finds contours that approximate to triangles (3 vertices).

    Returns:
        list of (x, y, confidence, 1.0, 'contour') detections
    """
    contours, hierarchy = cv2.findContours(
        binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue

        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        n_verts = len(approx)

        if n_verts < 3 or n_verts > 6:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)

        # Triangle shape criteria
        if not (0.5 < aspect < 2.0 and 8 < w < 35 and 8 < h < 30):
            continue

        # Check apex orientation (should point up: topmost point near center-x)
        topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        if abs(topmost[0] - (x + w // 2)) > w * 0.4:
            continue  # apex too far from center = probably not pointing up

        cx = x + w // 2
        cy = y + h // 2

        # Apply content mask
        if content_mask is not None:
            if cy >= content_mask.shape[0] or cx >= content_mask.shape[1]:
                continue
            if content_mask[cy, cx] == 0:
                continue

        # Confidence based on how triangular the contour is
        # (closer to 3 vertices = higher confidence)
        confidence = 0.5 if n_verts == 3 else 0.4

        detections.append((cx, cy, confidence, 1.0, 'contour'))

    return detections


def visualize_detections(image_bgr, detections, output_path,
                          known_points=None, thumbnail_width=2000):
    """
    Draw detections on a thumbnail of the map.

    Args:
        image_bgr: original color image
        detections: list of (x, y, confidence, scale, name) tuples
        output_path: where to save the annotated image
        known_points: optional list of (px, py) known control point positions
        thumbnail_width: width of output thumbnail
    """
    h, w = image_bgr.shape[:2]
    scale = thumbnail_width / w
    thumb = cv2.resize(image_bgr, (thumbnail_width, int(h * scale)))

    # Draw known points (blue circles)
    if known_points:
        for kx, ky in known_points:
            tx, ty = int(kx * scale), int(ky * scale)
            cv2.circle(thumb, (tx, ty), 8, (255, 150, 0), 2)  # blue

    # Draw detections (green circles for high confidence, yellow for low)
    for x, y, conf, sc, name in detections:
        tx, ty = int(x * scale), int(y * scale)
        if conf >= 0.6:
            color = (0, 255, 0)  # green
        elif conf >= 0.5:
            color = (0, 255, 255)  # yellow
        else:
            color = (0, 100, 255)  # orange

        cv2.circle(thumb, (tx, ty), 6, color, 2)
        cv2.putText(thumb, f'{conf:.2f}', (tx + 8, ty - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    cv2.imwrite(str(output_path), thumb)
    print(f"Visualization saved to {output_path}")


# --- CLI test: run detection on M5_4048 ---
if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from image_loader import load_image, preprocess, get_map_content_mask
    from coord_converter import load_tfwx, map_to_pixel, load_control_points

    base = Path(__file__).resolve().parent.parent
    map_dir = base / 'T1' / 'M5_4048'
    template_dir = base / 'scripts' / 'templates'
    output_dir = base / 'output'
    output_dir.mkdir(exist_ok=True)

    # Load image and preprocess
    print("Loading image...")
    img = load_image(map_dir / 'M5_4048.jpg')

    print("Preprocessing...")
    black_ink, binary = preprocess(img)
    content_mask = get_map_content_mask(img)

    # Load templates
    print("\nLoading templates...")
    templates = load_templates(template_dir)

    # Run detection
    print("\nRunning triangle detection...")
    detections = detect_triangles(
        black_ink, binary, templates,
        scales=(0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        threshold=0.45,
        nms_distance=20,
        content_mask=content_mask
    )

    # Also run contour-based detection
    print("\nRunning contour-based detection...")
    contour_dets = detect_triangles_contour(binary, content_mask)
    print(f"  Contour detections: {len(contour_dets)}")

    # Merge and deduplicate
    all_dets = detections + contour_dets
    final = non_max_suppression(all_dets, min_distance=20)
    print(f"\nFinal detections (template + contour, after NMS): {len(final)}")

    # Load known control points for comparison
    affine = load_tfwx(map_dir / 'M5_4048.tfwx')
    cps = load_control_points(map_dir / 'M5_4048_controlpoints.txt')
    known_pixels = []
    for cp in cps:
        px, py = map_to_pixel(cp['map_x'], cp['map_y'], affine)
        known_pixels.append((px, py))

    # Visualize
    print("\nGenerating visualization...")
    visualize_detections(
        img, final,
        output_dir / 'M5_4048_detections.png',
        known_points=known_pixels,
        thumbnail_width=2000
    )

    # Report matches vs known points
    print("\n--- Matching Results ---")
    match_radius = 30  # pixels
    matched_known = set()
    matched_detected = set()

    for di, (dx, dy, conf, sc, name) in enumerate(final):
        best_dist = float('inf')
        best_ki = -1
        for ki, (kx, ky) in enumerate(known_pixels):
            dist = np.sqrt((dx - kx) ** 2 + (dy - ky) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_ki = ki

        if best_dist < match_radius:
            matched_known.add(best_ki)
            matched_detected.add(di)
            print(f"  Detection {di} ({dx:.0f},{dy:.0f}) conf={conf:.2f} "
                  f"-> Known point {best_ki} (dist={best_dist:.1f}px)")

    print(f"\nRecall: {len(matched_known)}/{len(known_pixels)} "
          f"({len(matched_known)/len(known_pixels)*100:.1f}%) known points detected")
    print(f"Precision: {len(matched_detected)}/{len(final)} "
          f"({len(matched_detected)/max(len(final),1)*100:.1f}%) detections match known points")
    print(f"False positives: {len(final) - len(matched_detected)}")
    print(f"Missed known points: {len(known_pixels) - len(matched_known)}")
