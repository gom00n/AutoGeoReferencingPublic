"""
Generate YOLO-format training data for triangle detection.

Uses existing ground truth (controlpoints.txt files) to create labeled crops:
- Positive examples: crops centered on known control points (triangles)
- Negative examples: crops from DB-candidate locations that don't match any control point

Two output formats:
1. Classification crops (triangle / not-triangle) for binary CNN
2. YOLO detection format (bounding boxes) for object detection

The geodetic DB provides a natural source of "hard negatives" — locations
that have geodetic significance but don't contain visible triangles.
"""

import sys
import csv
import shutil
import random
import numpy as np
import cv2
from pathlib import Path

from coord_converter import load_tfwx, map_to_pixel, load_control_points, get_map_extent
from image_loader import load_image, suppress_red
from db_matcher import load_geodetic_db, filter_points_to_extent


def generate_crops_for_map(map_dir, geo_db, output_dir,
                            crop_size=64, match_radius=25):
    """
    Generate labeled crops from a single map.

    Args:
        map_dir: path to map folder
        geo_db: full geodetic database
        output_dir: root output directory (will create pos/neg subdirs)
        crop_size: size of square crops
        match_radius: max distance to count DB point as matching a control point

    Returns:
        dict with counts of positive and negative crops
    """
    map_dir = Path(map_dir)
    map_name = map_dir.name

    # Find files
    img_files = list(map_dir.glob('*.jpg'))
    tfwx_files = list(map_dir.glob('*.tfwx'))
    cp_files = list(map_dir.glob('*controlpoints.txt'))

    if not img_files or not tfwx_files or not cp_files:
        return None

    # Load transform
    affine = load_tfwx(tfwx_files[0])

    # Get image dimensions
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(img_files[0]) as pil_img:
        w, h = pil_img.size

    # Load image
    img = load_image(img_files[0])
    h_img, w_img = img.shape[:2]
    half = crop_size // 2

    # Load ground truth control points → pixel positions
    ground_truth = load_control_points(cp_files[0])
    gt_pixels = []
    for cp in ground_truth:
        if cp.get('enable', 1) == 0:
            continue
        px, py = map_to_pixel(cp['map_x'], cp['map_y'], affine)
        px_i, py_i = int(round(px)), int(round(py))
        if (half <= px_i < w_img - half and half <= py_i < h_img - half):
            gt_pixels.append((px_i, py_i))

    # Get DB candidates in extent
    extent = get_map_extent(affine, w, h)
    candidates = filter_points_to_extent(geo_db, extent)

    # Project all candidates to pixel positions
    candidate_pixels = []
    for point in candidates:
        px, py = map_to_pixel(point.easting_6991, point.northing_6991, affine)
        px_i, py_i = int(round(px)), int(round(py))
        if (half <= px_i < w_img - half and half <= py_i < h_img - half):
            candidate_pixels.append((px_i, py_i, point.name))

    # Classify each candidate as positive or negative
    pos_dir = output_dir / 'positive'
    neg_dir = output_dir / 'negative'
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    n_pos = 0
    n_neg = 0

    for cx, cy, name in candidate_pixels:
        # Check if this candidate matches any ground truth point
        is_positive = False
        for gx, gy in gt_pixels:
            dist = np.sqrt((cx - gx)**2 + (cy - gy)**2)
            if dist < match_radius:
                is_positive = True
                break

        # Extract crop (full color)
        crop = img[cy - half:cy + half, cx - half:cx + half]
        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            continue

        # Save
        safe_name = name.replace('/', '_')
        fname = f"{map_name}_{safe_name}_{cx}_{cy}.png"

        if is_positive:
            cv2.imwrite(str(pos_dir / fname), crop)
            n_pos += 1
        else:
            cv2.imwrite(str(neg_dir / fname), crop)
            n_neg += 1

    # Also generate YOLO-format annotations
    yolo_img_dir = output_dir / 'yolo' / 'images'
    yolo_lbl_dir = output_dir / 'yolo' / 'labels'
    yolo_img_dir.mkdir(parents=True, exist_ok=True)
    yolo_lbl_dir.mkdir(parents=True, exist_ok=True)

    # For YOLO: save full-resolution crops (larger context) with bbox annotations
    yolo_crop_size = 256
    yolo_half = yolo_crop_size // 2

    for gx, gy in gt_pixels:
        if (yolo_half <= gx < w_img - yolo_half and
            yolo_half <= gy < h_img - yolo_half):

            crop = img[gy - yolo_half:gy + yolo_half,
                       gx - yolo_half:gx + yolo_half]

            if crop.shape[0] != yolo_crop_size or crop.shape[1] != yolo_crop_size:
                continue

            fname = f"{map_name}_{gx}_{gy}"
            cv2.imwrite(str(yolo_img_dir / f"{fname}.png"), crop)

            # YOLO label: class_id center_x center_y width height (normalized)
            # Triangle centered at (0.5, 0.5) with ~24px size = 24/256 ≈ 0.094
            tri_size = 24.0 / yolo_crop_size
            with open(yolo_lbl_dir / f"{fname}.txt", 'w') as f:
                f.write(f"0 0.5 0.5 {tri_size:.4f} {tri_size:.4f}\n")

    return {
        'map_name': map_name,
        'n_positive': n_pos,
        'n_negative': n_neg,
        'n_ground_truth': len(gt_pixels),
        'n_candidates': len(candidate_pixels),
    }


def create_train_val_split(output_dir, val_ratio=0.2, seed=42):
    """
    Split positive/negative crops into train/val sets.
    Also creates YOLO dataset.yaml.
    """
    random.seed(seed)

    # Classification split
    for label in ['positive', 'negative']:
        src_dir = output_dir / label
        if not src_dir.exists():
            continue

        files = list(src_dir.glob('*.png'))
        random.shuffle(files)
        n_val = int(len(files) * val_ratio)

        train_dir = output_dir / 'train' / label
        val_dir = output_dir / 'val' / label
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for i, f in enumerate(files):
            dst = val_dir if i < n_val else train_dir
            shutil.copy2(f, dst / f.name)

        print(f"  {label}: {len(files) - n_val} train, {n_val} val")

    # YOLO split
    yolo_dir = output_dir / 'yolo'
    img_dir = yolo_dir / 'images'
    if img_dir.exists():
        images = list(img_dir.glob('*.png'))
        random.shuffle(images)
        n_val = int(len(images) * val_ratio)

        for split, split_images in [('val', images[:n_val]),
                                      ('train', images[n_val:])]:
            split_img = yolo_dir / split / 'images'
            split_lbl = yolo_dir / split / 'labels'
            split_img.mkdir(parents=True, exist_ok=True)
            split_lbl.mkdir(parents=True, exist_ok=True)

            for img_path in split_images:
                shutil.copy2(img_path, split_img / img_path.name)
                lbl_path = yolo_dir / 'labels' / img_path.with_suffix('.txt').name
                if lbl_path.exists():
                    shutil.copy2(lbl_path, split_lbl / lbl_path.name)

        # Write dataset.yaml
        yaml_content = f"""# Triangle Detection Dataset
# Auto-generated from {len(images)} ground truth control points

path: {yolo_dir.resolve()}
train: train/images
val: val/images

nc: 1
names:
  0: triangle
"""
        with open(yolo_dir / 'dataset.yaml', 'w') as f:
            f.write(yaml_content)

        print(f"  YOLO: {len(images) - n_val} train, {n_val} val")
        print(f"  Dataset config: {yolo_dir / 'dataset.yaml'}")


if __name__ == '__main__':
    base = Path(__file__).resolve().parent.parent
    output_dir = base / 'training_data'
    output_dir.mkdir(exist_ok=True)

    # Load geodetic DB
    print("Loading geodetic database...")
    geo_db = load_geodetic_db(base / 'Control_Points' / 'nikudot_bakara_slim.csv')
    print(f"  Loaded {len(geo_db)} points")

    # Discover map folders
    map_dirs = []
    for series in ['T1', 'T2']:
        series_dir = base / series
        if series_dir.exists():
            for d in sorted(series_dir.iterdir()):
                if d.is_dir() and d.name.startswith('M'):
                    map_dirs.append(d)

    if len(sys.argv) > 1:
        target = sys.argv[1]
        map_dirs = [d for d in map_dirs if target in d.name]

    # Generate crops
    print(f"\nGenerating training data from {len(map_dirs)} maps...")
    total_pos = 0
    total_neg = 0

    for map_dir in map_dirs:
        result = generate_crops_for_map(map_dir, geo_db, output_dir)
        if result:
            print(f"  {result['map_name']}: {result['n_positive']} pos, "
                  f"{result['n_negative']} neg "
                  f"(from {result['n_ground_truth']} GT, "
                  f"{result['n_candidates']} candidates)")
            total_pos += result['n_positive']
            total_neg += result['n_negative']

    print(f"\nTotal: {total_pos} positive, {total_neg} negative crops")

    # Create train/val split
    print("\nCreating train/val split...")
    create_train_val_split(output_dir)

    print(f"\nTraining data saved to {output_dir}")
