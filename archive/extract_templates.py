"""
Extract triangle templates from known control point locations.

Uses existing control points (from ArcGIS Pro georeferencing) to locate
triangle symbols, then crops them as templates for template matching.
"""

import cv2
import numpy as np
from pathlib import Path
from coord_converter import load_tfwx, map_to_pixel, load_control_points


def extract_candidates(image_path, control_points, affine, crop_size=40):
    """
    Crop regions around each known control point.

    Args:
        image_path: path to the map image (JPG or TIF)
        control_points: list of dicts with 'map_x', 'map_y'
        affine: dict from load_tfwx()
        crop_size: size of crop in pixels (square)

    Returns:
        list of (crop_bgr, crop_gray, pixel_x, pixel_y, idx)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    half = crop_size // 2
    candidates = []

    for idx, cp in enumerate(control_points):
        if not cp.get('enable', 1):
            continue

        px, py = map_to_pixel(cp['map_x'], cp['map_y'], affine)
        px, py = int(round(px)), int(round(py))

        # Skip if too close to edge
        if px - half < 0 or py - half < 0 or px + half >= w or py + half >= h:
            print(f"  Skipping point {idx}: pixel ({px}, {py}) too close to edge")
            continue

        crop = img[py - half:py + half, px - half:px + half]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        candidates.append((crop, gray, px, py, idx))

    return candidates


def score_candidate(gray_crop):
    """
    Score a candidate crop for template quality.
    Higher score = cleaner, more distinct triangle on uniform background.

    Criteria:
    - High contrast (large std dev in center region)
    - Clean background (low std dev in outer ring)
    - Dark triangle pixels present (low min value in center)
    """
    h, w = gray_crop.shape
    ch, cw = h // 2, w // 2

    # Center region (where triangle should be)
    center = gray_crop[ch - 8:ch + 8, cw - 8:cw + 8]
    # Outer ring (background)
    outer_mask = np.ones_like(gray_crop, dtype=bool)
    outer_mask[ch - 10:ch + 10, cw - 10:cw + 10] = False
    outer = gray_crop[outer_mask]

    # Score components
    center_contrast = center.std()  # Higher = more variation = triangle present
    bg_uniformity = 255 - outer.std()  # Higher = more uniform background
    dark_pixels = (center < 120).sum()  # Count of dark (ink) pixels

    score = center_contrast * 0.5 + bg_uniformity * 0.3 + dark_pixels * 2.0
    return score


def create_template(gray_crop, method='adaptive'):
    """
    Create a clean binary template from a grayscale crop.

    Args:
        gray_crop: grayscale crop centered on triangle
        method: 'adaptive' or 'otsu'

    Returns:
        binary template (dark features = white, background = black)
    """
    if method == 'adaptive':
        binary = cv2.adaptiveThreshold(
            gray_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 10
        )
    else:
        _, binary = cv2.threshold(
            gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    return binary


def suppress_red_crop(bgr_crop):
    """
    Suppress red/brown features (contour lines) and keep black ink.
    Returns a grayscale image where only black ink features remain dark.
    """
    b, g, r = cv2.split(bgr_crop)
    # Black ink has low values in ALL channels
    # Red features have high R but low B,G
    # min(B, G) keeps black ink dark but makes red features bright
    black_ink = np.minimum(b, g)
    return black_ink


def extract_and_save_templates(base_dir, map_name='M5_4048', map_type='T1',
                                n_templates=5, crop_size=40):
    """
    Extract the best triangle templates from a map.

    Args:
        base_dir: base directory (Map_Scans)
        map_name: map folder name
        map_type: 'T1' or 'T2'
        n_templates: number of best templates to save
        crop_size: crop size in pixels
    """
    base = Path(base_dir)
    map_dir = base / map_type / map_name
    template_dir = base / 'scripts' / 'templates'
    template_dir.mkdir(exist_ok=True)

    # Load affine transform
    tfwx_path = map_dir / f'{map_name}.tfwx'
    affine = load_tfwx(tfwx_path)
    print(f"Loaded affine from {tfwx_path}")

    # Load control points
    cp_path = map_dir / f'{map_name}_controlpoints.txt'
    cps = load_control_points(cp_path)
    print(f"Loaded {len(cps)} control points")

    # Extract candidate crops
    img_path = map_dir / f'{map_name}.jpg'
    print(f"Loading image: {img_path}")
    candidates = extract_candidates(img_path, cps, affine, crop_size)
    print(f"Extracted {len(candidates)} candidate crops")

    # Score and rank candidates
    scored = []
    for crop_bgr, crop_gray, px, py, idx in candidates:
        # Also create red-suppressed version
        black_ink = suppress_red_crop(crop_bgr)
        score = score_candidate(black_ink)
        scored.append((score, crop_bgr, crop_gray, black_ink, px, py, idx))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Save top N templates
    print(f"\nTop {n_templates} templates (by quality score):")
    for rank, (score, crop_bgr, crop_gray, black_ink, px, py, idx) in enumerate(scored[:n_templates]):
        print(f"  #{rank+1}: score={score:.1f}, pixel=({px}, {py}), cp_idx={idx}")

        # Save color crop for inspection
        cv2.imwrite(str(template_dir / f'{map_type}_{map_name}_cp{idx}_color.png'), crop_bgr)
        # Save grayscale
        cv2.imwrite(str(template_dir / f'{map_type}_{map_name}_cp{idx}_gray.png'), crop_gray)
        # Save red-suppressed
        cv2.imwrite(str(template_dir / f'{map_type}_{map_name}_cp{idx}_blackink.png'), black_ink)
        # Save binary template
        binary = create_template(black_ink, method='adaptive')
        cv2.imwrite(str(template_dir / f'{map_type}_{map_name}_cp{idx}_binary.png'), binary)

    # Also save ALL candidates as a contact sheet for visual review
    print(f"\nSaving contact sheet of all {len(candidates)} candidates...")
    cols = 6
    rows = (len(scored) + cols - 1) // cols
    cell_size = crop_size + 4  # 2px border
    sheet = np.ones((rows * cell_size, cols * cell_size, 3), dtype=np.uint8) * 200

    for i, (score, crop_bgr, _, _, px, py, idx) in enumerate(scored):
        r, c = divmod(i, cols)
        y0, x0 = r * cell_size + 2, c * cell_size + 2
        h, w = crop_bgr.shape[:2]
        sheet[y0:y0+h, x0:x0+w] = crop_bgr

    cv2.imwrite(str(template_dir / f'{map_type}_{map_name}_contact_sheet.png'), sheet)
    print(f"Contact sheet saved to {template_dir / f'{map_type}_{map_name}_contact_sheet.png'}")

    return scored[:n_templates]


if __name__ == '__main__':
    base = str(Path(__file__).resolve().parent.parent)

    print("=" * 60)
    print("Extracting templates from T1/M5_4048")
    print("=" * 60)
    templates = extract_and_save_templates(base, 'M5_4048', 'T1', n_templates=5, crop_size=40)
