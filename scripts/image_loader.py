"""
Image loading and preprocessing for map triangle detection.

Key preprocessing: red/brown contour line suppression to isolate black ink.
"""

import cv2
import numpy as np


def load_image(image_path):
    """
    Load a map image (JPG or TIF) using OpenCV.
    Returns BGR numpy array.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    print(f"Loaded image: {img.shape[1]}x{img.shape[0]} ({image_path})")
    return img


def suppress_red(image_bgr):
    """
    Suppress red/brown features (contour lines, cadastral boundaries)
    while preserving black ink (triangles, text, grid lines).

    Method: min(Blue, Green) channel. Red ink has high R but low B,G
    so it becomes bright. Black ink has low values in all channels so
    it remains dark.

    Returns: grayscale image (uint8)
    """
    b = image_bgr[:, :, 0].astype(np.int16)
    g = image_bgr[:, :, 1].astype(np.int16)
    black_ink = np.minimum(b, g).astype(np.uint8)
    return black_ink


def suppress_colors(image_bgr):
    """
    Suppress ALL colored backgrounds/ink (yellow, red, brown, blue, etc.)
    while preserving black ink (triangles, text, grid lines).

    Method: Use the HSV Value channel (= max(B,G,R)) as the base grayscale.
    This makes any colored area bright (since at least one channel is high),
    while black ink stays dark (all channels are low).

    Then reduce brightness of high-saturation pixels further: areas with
    strong color get pushed brighter, ensuring colored ink doesn't masquerade
    as dark features.

    Key insight: black ink has low values in ALL channels. Any colored
    area (yellow, red, brown, blue) has at least one high channel, so
    the max channel naturally separates black ink from colored backgrounds.

    Returns: grayscale image (uint8) where black ink is dark, everything else bright
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)  # Value = max(B,G,R)
    s = hsv[:, :, 1].astype(np.float32)  # Saturation

    # Push colored (high-saturation) areas brighter:
    # saturated pixels get a brightness boost proportional to saturation
    result = v + s * 0.5
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def to_black_white(image_bgr):
    """
    Convert a map image to high-contrast binary black and white.
    Designed to handle colored backgrounds (yellow, brown, etc.)
    that interfere with triangle detection.

    Pipeline:
      1. Convert to grayscale using standard luminance weighting
      2. Apply adaptive thresholding for clean binarization
      3. Return binary image: dark features (triangles, text, grid lines) = 0 (black),
         everything else = 255 (white)

    Returns: binary image (uint8, values 0 or 255)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold: THRESH_BINARY (not BINARY_INV) so that
    # dark features become 0 (black) and background becomes 255 (white)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=25,
        C=10
    )

    return binary


def preprocess(image_bgr, method='adaptive', block_size=15, c_offset=10):
    """
    Full preprocessing pipeline: red suppression + binarization.

    Args:
        image_bgr: BGR input image
        method: 'adaptive' or 'otsu'
        block_size: adaptive threshold block size
        c_offset: adaptive threshold constant

    Returns:
        (black_ink_gray, binary) where binary has dark features = white (255)
    """
    black_ink = suppress_red(image_bgr)

    if method == 'adaptive':
        binary = cv2.adaptiveThreshold(
            black_ink, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, c_offset
        )
    else:
        _, binary = cv2.threshold(
            black_ink, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    return black_ink, binary


def get_map_content_mask(image_bgr, border_margin=100):
    """
    Create a mask of the actual map content area, excluding borders,
    legend, and margins.

    Simple approach: the map is bounded by thick black neatlines.
    We detect these and create a mask of the interior.

    Args:
        image_bgr: BGR input image
        border_margin: pixels to exclude from each edge

    Returns:
        binary mask (255 = map content, 0 = margin/legend)
    """
    h, w = image_bgr.shape[:2]

    # Simple approach: exclude outer border_margin pixels
    # and bottom 15% (legend area)
    mask = np.zeros((h, w), dtype=np.uint8)
    legend_cutoff = int(h * 0.87)  # bottom ~13% is typically legend
    mask[border_margin:legend_cutoff, border_margin:w - border_margin] = 255

    return mask


# --- CLI test ---
if __name__ == '__main__':
    from pathlib import Path

    base = Path(__file__).resolve().parent.parent
    img = load_image(base / 'Map_Scans' / 'sample-series' / 'M5_4048' / 'M5_4048.jpg')

    print("Running preprocessing...")
    black_ink, binary = preprocess(img)

    print(f"Black ink: {black_ink.shape}, dtype={black_ink.dtype}")
    print(f"Binary: {binary.shape}, white pixels: {(binary > 0).sum()}/{binary.size} "
          f"({(binary > 0).sum()/binary.size*100:.1f}%)")

    # Save small preview
    preview_scale = 800.0 / black_ink.shape[1]
    preview_bk = cv2.resize(black_ink, None, fx=preview_scale, fy=preview_scale)
    preview_bn = cv2.resize(binary, None, fx=preview_scale, fy=preview_scale)
    cv2.imwrite(str(base / 'output' / 'preview_blackink.png'), preview_bk)
    cv2.imwrite(str(base / 'output' / 'preview_binary.png'), preview_bn)
    print("Preview images saved to output/")

    # Test content mask
    mask = get_map_content_mask(img)
    print(f"Content mask: {(mask > 0).sum()/mask.size*100:.1f}% of image")
