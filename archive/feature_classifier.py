"""
Feature-based triangle classifier using gradient boosting.

Instead of raw pixels (CNN), extract meaningful features from each crop:
- Template matching scores (grayscale + edge)
- Local contrast statistics
- Shape/contour properties
- Dark pixel density

Then train a lightweight sklearn gradient boosting classifier.
Works better than CNN with noisy labels and small positive sets.
"""

import sys
import pickle
import numpy as np
import cv2
from pathlib import Path

from coord_converter import load_tfwx, map_to_pixel, load_control_points, get_map_extent
from image_loader import load_image, suppress_red
from db_matcher import (
    load_geodetic_db, filter_points_to_extent,
    load_grayscale_templates, match_crop_grayscale, match_crop_edges,
    verify_triangle_shape,
)

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report, precision_recall_curve
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def extract_features(crop_gray, templates):
    """
    Extract a feature vector from a grayscale crop.

    Returns: dict of feature_name -> value
    """
    features = {}
    h, w = crop_gray.shape

    # 1. Template matching scores
    gray_conf, gray_name, gox, goy = match_crop_grayscale(crop_gray, templates)
    edge_conf, edge_name, eox, eoy = match_crop_edges(crop_gray, templates)
    features['gray_conf'] = gray_conf
    features['edge_conf'] = edge_conf
    features['conf_max'] = max(gray_conf, edge_conf)
    features['conf_min'] = min(gray_conf, edge_conf)
    features['conf_mean'] = (gray_conf + edge_conf) / 2

    # 2. Local pixel statistics at match center
    cx, cy = gox if gray_conf >= edge_conf else eox, goy if gray_conf >= edge_conf else eoy
    r = 14
    local = crop_gray[max(0,cy-r):min(h,cy+r), max(0,cx-r):min(w,cx+r)]
    if local.size > 0:
        features['local_mean'] = float(local.mean())
        features['local_std'] = float(local.std())
        features['local_min'] = float(local.min())
        features['local_p10'] = float(np.percentile(local, 10))
        features['local_p90'] = float(np.percentile(local, 90))
    else:
        features['local_mean'] = 200.0
        features['local_std'] = 0.0
        features['local_min'] = 200.0
        features['local_p10'] = 200.0
        features['local_p90'] = 200.0

    # 3. Background vs center contrast
    bg = crop_gray[max(0,cy-30):min(h,cy+30), max(0,cx-30):min(w,cx+30)]
    bg_mean = float(bg.mean()) if bg.size > 0 else 200.0
    features['contrast'] = (bg_mean - features['local_mean']) / max(bg_mean, 1)

    # Dark pixel ratio
    dark_thresh = bg_mean * 0.45
    if local.size > 0:
        features['dark_ratio'] = float((local < dark_thresh).sum()) / local.size
    else:
        features['dark_ratio'] = 0.0

    # 4. Shape verification score
    shape_score, _, _ = verify_triangle_shape(crop_gray, cx, cy)
    features['shape_score'] = shape_score

    # 5. Edge density around match center
    edges = cv2.Canny(crop_gray, 40, 120)
    local_edges = edges[max(0,cy-r):min(h,cy+r), max(0,cx-r):min(w,cx+r)]
    features['edge_density'] = float(local_edges.mean()) / 255.0 if local_edges.size > 0 else 0.0

    # 6. Symmetry (flip and compare)
    if local.size > 0 and local.shape[0] > 4 and local.shape[1] > 4:
        flipped = cv2.flip(local, 1)  # horizontal flip
        diff = np.abs(local.astype(np.float32) - flipped.astype(np.float32))
        features['h_symmetry'] = 1.0 - float(diff.mean()) / 128.0
    else:
        features['h_symmetry'] = 0.5

    return features


FEATURE_NAMES = [
    'gray_conf', 'edge_conf', 'conf_max', 'conf_min', 'conf_mean',
    'local_mean', 'local_std', 'local_min', 'local_p10', 'local_p90',
    'contrast', 'dark_ratio', 'shape_score', 'edge_density', 'h_symmetry',
]


def features_to_vector(feat_dict):
    """Convert feature dict to numpy array."""
    return np.array([feat_dict.get(name, 0.0) for name in FEATURE_NAMES])


def build_training_data(map_dirs, geo_db, template_dir, match_radius=25):
    """
    Build feature matrix and labels from all maps.

    Uses controlpoints.txt as ground truth. For each DB candidate,
    extracts features and labels it positive/negative.
    """
    templates = load_grayscale_templates(template_dir)
    X_list = []
    y_list = []
    meta = []  # (map_name, point_name, px, py) for debugging

    for map_dir in map_dirs:
        map_dir = Path(map_dir)
        map_name = map_dir.name

        img_files = list(map_dir.glob('*.jpg'))
        tfwx_files = list(map_dir.glob('*.tfwx'))
        cp_files = list(map_dir.glob('*controlpoints.txt'))

        if not img_files or not tfwx_files or not cp_files:
            continue

        affine = load_tfwx(tfwx_files[0])

        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(img_files[0]) as pil_img:
            w, h = pil_img.size

        extent = get_map_extent(affine, w, h)
        candidates = filter_points_to_extent(geo_db, extent)

        if not candidates:
            continue

        img = load_image(img_files[0])
        black_ink = suppress_red(img)
        h_img, w_img = black_ink.shape[:2]

        # Load ground truth
        ground_truth = load_control_points(cp_files[0])
        gt_pixels = []
        for cp in ground_truth:
            if cp.get('enable', 1) == 0:
                continue
            px, py = map_to_pixel(cp['map_x'], cp['map_y'], affine)
            gt_pixels.append((int(round(px)), int(round(py))))

        # Process each candidate
        half = 40  # crop half-size for features
        map_start = len(y_list)
        for point in candidates:
            px, py = map_to_pixel(point.easting_6991, point.northing_6991, affine)
            px_i, py_i = int(round(px)), int(round(py))

            if (px_i - half < 0 or py_i - half < 0 or
                px_i + half >= w_img or py_i + half >= h_img):
                continue

            crop = black_ink[py_i - half:py_i + half, px_i - half:px_i + half]

            # Label
            is_positive = False
            for gx, gy in gt_pixels:
                if np.sqrt((px_i - gx)**2 + (py_i - gy)**2) < match_radius:
                    is_positive = True
                    break

            # Extract features
            feat = extract_features(crop, templates)
            X_list.append(features_to_vector(feat))
            y_list.append(1 if is_positive else 0)
            meta.append((map_name, point.name, px_i, py_i))

        map_labels = y_list[map_start:]
        print(f"  {map_name}: {sum(1 for x in map_labels if x == 1)} pos, "
              f"{sum(1 for x in map_labels if x == 0)} neg")

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y, meta


def train_classifier(X, y, output_path):
    """Train gradient boosting classifier and save."""
    pos = y.sum()
    neg = len(y) - pos
    scale = neg / max(pos, 1)
    print(f"\nTraining: {int(pos)} positive, {int(neg)} negative (ratio 1:{scale:.0f})")

    # Sample weights to handle imbalance
    weights = np.where(y == 1, scale, 1.0)

    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )

    # Cross-validation
    print("Cross-validation (5-fold)...")
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_preds = np.zeros(len(y))
    all_probs = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        clf_fold = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=5, random_state=42,
        )
        clf_fold.fit(X[train_idx], y[train_idx],
                     sample_weight=weights[train_idx])

        probs = clf_fold.predict_proba(X[val_idx])[:, 1]
        preds = (probs > 0.5).astype(int)
        all_preds[val_idx] = preds
        all_probs[val_idx] = probs

        tp = ((preds == 1) & (y[val_idx] == 1)).sum()
        fp = ((preds == 1) & (y[val_idx] == 0)).sum()
        fn = ((preds == 0) & (y[val_idx] == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        print(f"  Fold {fold+1}: prec={prec:.3f} rec={rec:.3f} F1={f1:.3f}")

    # Overall metrics
    print("\nOverall cross-validation:")
    print(classification_report(y, all_preds, target_names=['negative', 'positive']))

    # Find optimal threshold for high recall.
    # recalls is DECREASING (recalls[0]=1.0 at the lowest threshold), so scan
    # from the high-threshold end to find the highest threshold (= best
    # precision) that still achieves recall >= 0.85.
    precisions, recalls, thresholds = precision_recall_curve(y, all_probs)
    optimal_threshold, optimal_prec, optimal_rec = 0.3, 0.0, 0.0
    for i in range(len(thresholds) - 1, -1, -1):
        if recalls[i] >= 0.85:
            optimal_threshold = thresholds[i]
            optimal_prec = precisions[i]
            optimal_rec = recalls[i]
            break

    print(f"\nOptimal threshold for recall≥85%: {optimal_threshold:.3f} "
          f"(prec={optimal_prec:.3f}, rec={optimal_rec:.3f})")

    # Train final model on all data
    clf.fit(X, y, sample_weight=weights)

    # Feature importance
    print("\nFeature importance:")
    for name, imp in sorted(zip(FEATURE_NAMES, clf.feature_importances_),
                             key=lambda x: -x[1]):
        print(f"  {name:>15s}: {imp:.3f}")

    # Save
    with open(output_path, 'wb') as f:
        pickle.dump({
            'classifier': clf,
            'feature_names': FEATURE_NAMES,
            'optimal_threshold': optimal_threshold,
        }, f)
    print(f"\nModel saved to {output_path}")

    return clf, optimal_threshold


if __name__ == '__main__':
    if not HAS_SKLEARN:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                              'scikit-learn', '--quiet'])
        print("Installed scikit-learn. Please re-run.")
        sys.exit(0)

    base = Path(__file__).resolve().parent.parent
    template_dir = base / 'scripts' / 'templates'
    model_path = base / 'scripts' / 'triangle_gbm.pkl'

    # Load geodetic DB
    print("Loading geodetic database...")
    geo_db = load_geodetic_db(base / 'Control_Points' / 'nikudot_bakara_slim.csv')
    print(f"  Loaded {len(geo_db)} points")

    # Discover maps
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

    # Build training data
    print(f"\nExtracting features from {len(map_dirs)} maps...")
    X, y, meta = build_training_data(map_dirs, geo_db, template_dir)
    print(f"\nTotal: {X.shape[0]} samples, {int(y.sum())} positive, {int(len(y) - y.sum())} negative")

    # Train
    clf, threshold = train_classifier(X, y, model_path)
