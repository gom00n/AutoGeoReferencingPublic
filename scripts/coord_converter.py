"""
Coordinate conversion utilities for map georeferencing.

Parses TFWX world files (affine transform) to convert between
pixel coordinates and EPSG:6991 (Israeli Grid 05/12) map coordinates.
"""

import numpy as np
from pathlib import Path


def load_tfwx(tfwx_path):
    """
    Parse a TFWX world file and return the affine transform coefficients.

    TFWX format (6 lines):
        line 1: a  (x-component of pixel width)
        line 2: b  (rotation term)
        line 3: c  (rotation term)
        line 4: d  (y-component of pixel height, typically negative)
        line 5: e  (x-coordinate of upper-left pixel center)
        line 6: f  (y-coordinate of upper-left pixel center)

    The affine transform maps pixel (col, row) to map coordinates:
        map_x = a * col + c * row + e
        map_y = b * col + d * row + f

    Returns:
        dict with keys 'a', 'b', 'c', 'd', 'e', 'f' and 'forward' (2x3 matrix)
    """
    with open(tfwx_path, 'r') as fp:
        lines = [line.strip() for line in fp.readlines() if line.strip()]

    if len(lines) < 6:
        raise ValueError(f"TFWX file must have 6 lines, got {len(lines)}: {tfwx_path}")

    a = float(lines[0])  # pixel width in x
    b = float(lines[1])  # rotation
    c = float(lines[2])  # rotation
    d = float(lines[3])  # pixel height in y (negative = north-up)
    e = float(lines[4])  # origin x (easting of upper-left pixel center)
    f = float(lines[5])  # origin y (northing of upper-left pixel center)

    # Forward transform matrix: [map_x, map_y]^T = M @ [col, row, 1]^T
    forward = np.array([
        [a, c, e],
        [b, d, f]
    ])

    # Inverse transform: [col, row]^T = M_inv @ [map_x, map_y, 1]^T
    M = np.array([[a, c], [b, d]])
    M_inv = np.linalg.inv(M)
    offset = np.array([e, f])

    return {
        'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f,
        'forward': forward,
        'M': M,
        'M_inv': M_inv,
        'offset': offset,
    }


def pixel_to_map(pixel_x, pixel_y, affine):
    """
    Convert pixel coordinates (col, row) to map coordinates (easting, northing)
    in EPSG:6991.

    Args:
        pixel_x: column (0 = left edge)
        pixel_y: row (0 = top edge)
        affine: dict from load_tfwx()

    Returns:
        (map_x, map_y) = (easting, northing) in EPSG:6991
    """
    a, c, e = affine['a'], affine['c'], affine['e']
    b, d, f = affine['b'], affine['d'], affine['f']

    map_x = a * pixel_x + c * pixel_y + e
    map_y = b * pixel_x + d * pixel_y + f

    return map_x, map_y


def map_to_pixel(map_x, map_y, affine):
    """
    Convert map coordinates (easting, northing in EPSG:6991) to pixel coordinates.

    Args:
        map_x: easting in EPSG:6991
        map_y: northing in EPSG:6991
        affine: dict from load_tfwx()

    Returns:
        (pixel_x, pixel_y) = (column, row) as floats
    """
    M_inv = affine['M_inv']
    offset = affine['offset']

    delta = np.array([map_x - offset[0], map_y - offset[1]])
    pixel = M_inv @ delta

    return pixel[0], pixel[1]


def pixel_to_map_batch(pixel_coords, affine):
    """
    Convert an array of pixel coordinates to map coordinates.

    Args:
        pixel_coords: Nx2 array of (col, row) pairs
        affine: dict from load_tfwx()

    Returns:
        Nx2 array of (map_x, map_y) pairs
    """
    coords = np.asarray(pixel_coords, dtype=np.float64)
    ones = np.ones((coords.shape[0], 1))
    aug = np.hstack([coords, ones])  # Nx3
    map_coords = (affine['forward'] @ aug.T).T  # Nx2
    return map_coords


def map_to_pixel_batch(map_coords, affine):
    """
    Convert an array of map coordinates to pixel coordinates.

    Args:
        map_coords: Nx2 array of (map_x, map_y) pairs
        affine: dict from load_tfwx()

    Returns:
        Nx2 array of (pixel_x, pixel_y) pairs
    """
    coords = np.asarray(map_coords, dtype=np.float64)
    M_inv = affine['M_inv']
    offset = affine['offset']

    delta = coords - offset[np.newaxis, :]
    pixels = (M_inv @ delta.T).T
    return pixels


def get_map_extent(affine, image_width, image_height):
    """
    Compute the bounding box of the map in EPSG:6991 coordinates.

    Returns:
        dict with 'min_x', 'max_x', 'min_y', 'max_y' (easting/northing)
    """
    corners = np.array([
        [0, 0],
        [image_width, 0],
        [0, image_height],
        [image_width, image_height]
    ])
    map_corners = pixel_to_map_batch(corners, affine)

    return {
        'min_x': map_corners[:, 0].min(),
        'max_x': map_corners[:, 0].max(),
        'min_y': map_corners[:, 1].min(),
        'max_y': map_corners[:, 1].max(),
    }


def load_control_points(txt_path):
    """
    Load existing control points from ArcGIS Pro export file.

    Format: mapX,mapY,pixelX,pixelY,enable
    Note: pixelX/pixelY are in ArcGIS layout units, not actual pixels.
    We use mapX/mapY (EPSG:6991) and convert to pixels via inverse affine.

    Returns:
        list of dicts with 'map_x', 'map_y', 'enable'
    """
    points = []
    with open(txt_path, 'r') as fp:
        header = fp.readline().strip()  # skip header
        for line in fp:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 5:
                points.append({
                    'map_x': float(parts[0]),
                    'map_y': float(parts[1]),
                    'enable': int(parts[4]) if parts[4] else 1,
                })
    return points


# --- CLI test ---
if __name__ == '__main__':
    import sys

    base = Path(__file__).resolve().parent.parent
    test_map = base / 'Map_Scans' / 'sample-series' / 'M5_4048'

    # Load affine
    affine = load_tfwx(test_map / 'M5_4048.tfwx')
    print("Affine coefficients:")
    print(f"  a={affine['a']:.6f}, b={affine['b']:.6f}")
    print(f"  c={affine['c']:.6f}, d={affine['d']:.6f}")
    print(f"  e={affine['e']:.2f}, f={affine['f']:.2f}")
    print(f"  Pixel size: ~{abs(affine['a']):.3f} x {abs(affine['d']):.3f} m")

    # Load control points and convert to pixel positions
    cps = load_control_points(test_map / 'M5_4048_controlpoints.txt')
    print(f"\nLoaded {len(cps)} control points")
    print(f"Map X range: {min(p['map_x'] for p in cps):.1f} - {max(p['map_x'] for p in cps):.1f}")
    print(f"Map Y range: {min(p['map_y'] for p in cps):.1f} - {max(p['map_y'] for p in cps):.1f}")

    # Convert to pixels
    print("\nFirst 5 control points (map coords -> pixel coords):")
    for cp in cps[:5]:
        px, py = map_to_pixel(cp['map_x'], cp['map_y'], affine)
        print(f"  Map ({cp['map_x']:.1f}, {cp['map_y']:.1f}) -> Pixel ({px:.1f}, {py:.1f})")

    # Round-trip test
    print("\nRound-trip test (pixel -> map -> pixel):")
    for cp in cps[:3]:
        px, py = map_to_pixel(cp['map_x'], cp['map_y'], affine)
        mx, my = pixel_to_map(px, py, affine)
        print(f"  Original map: ({cp['map_x']:.4f}, {cp['map_y']:.4f})")
        print(f"  Round-trip:   ({mx:.4f}, {my:.4f})")
        print(f"  Error: ({abs(mx - cp['map_x']):.6f}, {abs(my - cp['map_y']):.6f}) m")

    # Map extent
    # Get image size without loading (use PIL for just the header)
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(test_map / 'M5_4048.jpg') as img:
        w, h = img.size

    extent = get_map_extent(affine, w, h)
    print(f"\nImage size: {w} x {h}")
    print(f"Map extent (EPSG:6991):")
    print(f"  Easting:  {extent['min_x']:.1f} - {extent['max_x']:.1f}")
    print(f"  Northing: {extent['min_y']:.1f} - {extent['max_y']:.1f}")
