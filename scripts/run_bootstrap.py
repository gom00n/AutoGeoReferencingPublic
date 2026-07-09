#!/usr/bin/env python3
"""
End-to-end bootstrap georeferencing for unreferenced maps.

Pipeline:
1. OCR grid labels from map margins
2. Build Old Palestine Grid affine from labels
3. Filter geodetic DB to map extent
4. Detect triangles via template matching
5. Compute EPSG:6991 affine transform
6. Output TFWX world file + visualization
"""
import sys
import time
from pathlib import Path

from image_loader import load_image
from grid_label_ocr import read_grid_labels, labels_to_affine
from db_matcher import load_geodetic_db, load_grayscale_templates
from bootstrap_from_grid import (
    bootstrap_georeference, labels_to_grid_points, labels_to_old_grid_extent,
    sheet_label_ranges,
)


def run_bootstrap(map_dir, geo_db, template_dir):
    """Run full bootstrap on a single map directory."""
    map_dir = Path(map_dir)
    map_name = map_dir.name

    # Find the JPG
    img_files = list(map_dir.glob('*.jpg'))
    if not img_files:
        print(f"  ERROR: No JPG found in {map_dir}")
        return None
    image_path = img_files[0]

    print(f"\n{'='*60}")
    print(f"  Bootstrap: {map_name}")
    print(f"{'='*60}")

    # Step 1: Load image and OCR grid labels
    t0 = time.time()
    print(f"\nStep 0: Loading image and reading grid labels...")
    img = load_image(str(image_path))
    h_img, w_img = img.shape[:2]

    # Sheet-number cross-check: the filename encodes the 10x10 km sheet
    # (CC-RR-Name-Year). When parseable, constrain OCR to that range so a
    # systematic decade misread can't produce a consistent-but-wrong affine.
    sheet_ranges = sheet_label_ranges(image_path.name)
    if sheet_ranges:
        e_range, n_range = sheet_ranges
        print(f"  Sheet number: eastings {e_range[0]}-{e_range[1]} km, "
              f"northings {n_range[0]}-{n_range[1]} km")
        result = read_grid_labels(img, expected_easting_range=e_range,
                                  expected_northing_range=n_range)
    else:
        result = read_grid_labels(img)
    e_labels = result['easting_labels']
    n_labels = result['northing_labels']
    print(f"  Easting labels: {len(e_labels)}, Northing labels: {len(n_labels)}")

    if len(e_labels) < 2 or len(n_labels) < 2:
        print(f"  ERROR: Not enough grid labels for {map_name}")
        return None

    # Step 2: Build Old Grid affine from labels
    affine = labels_to_affine(result)
    if affine is None:
        print(f"  ERROR: Could not compute affine from labels for {map_name}")
        return None

    print(f"  Old Grid affine: pixel_size={affine['pixel_size_x']:.4f}x{affine['pixel_size_y']:.4f} m/px")
    print(f"  RMSE: easting={affine['easting_rmse_m']:.1f}m, northing={affine['northing_rmse_m']:.1f}m")

    # Convert labels to grid_points format for bootstrap_georeference
    grid_points = labels_to_grid_points(result, affine)
    print(f"  Grid points for bootstrap: {len(grid_points)}")

    # Determine map extent from labels
    old_e_range, old_n_range = labels_to_old_grid_extent(result, affine)

    print(f"  Old Grid extent: E=[{old_e_range[0]/1000:.0f},{old_e_range[1]/1000:.0f}]km "
          f"N=[{old_n_range[0]/1000:.0f},{old_n_range[1]/1000:.0f}]km")

    t_ocr = time.time() - t0
    print(f"  OCR time: {t_ocr:.1f}s")

    # Step 3: Run bootstrap georeferencing
    result = bootstrap_georeference(
        image_path, geo_db, template_dir,
        grid_points, old_e_range, old_n_range,
        output_dir=map_dir,
    )

    return result


if __name__ == '__main__':
    base = Path(__file__).resolve().parent.parent
    template_dir = base / 'scripts' / 'templates'

    print("Loading geodetic database...")
    geo_db = load_geodetic_db(base / 'Control_Points' / 'nikudot_bakara_slim.csv')
    print(f"  Loaded {len(geo_db)} points")

    # Find maps to process
    control_maps = base / 'Control_Maps'
    if len(sys.argv) > 1:
        # Process specific maps
        map_dirs = [control_maps / name for name in sys.argv[1:]]
    else:
        # Process all maps without a TFWX
        map_dirs = []
        for d in sorted(control_maps.iterdir()):
            if d.is_dir() and d.name.startswith('M'):
                tfwx_files = list(d.glob('*.tfwx'))
                if not tfwx_files:
                    map_dirs.append(d)

    print(f"\nProcessing {len(map_dirs)} maps: {[d.name for d in map_dirs]}")

    results = []
    for map_dir in map_dirs:
        result = run_bootstrap(map_dir, geo_db, template_dir)
        if result:
            results.append((map_dir.name, result))
        else:
            results.append((map_dir.name, None))

    # Summary
    print(f"\n{'='*60}")
    print(f"  BOOTSTRAP SUMMARY")
    print(f"{'='*60}")
    print(f"{'Map':>15} | {'Pts':>4} | {'Inliers':>7} | {'RMSE':>8} | {'Status'}")
    print(f"{'-'*60}")
    for name, res in results:
        if res:
            print(f"{name:>15} | {res['n_points']:>4} | {res['n_inliers']:>7} | "
                  f"{res['fit_rmse_m']:>6.1f}m | OK")
        else:
            print(f"{name:>15} | {'':>4} | {'':>7} | {'':>8} | FAILED")
