#!/usr/bin/env python3
"""
Evaluate the triangle CNN on the frozen held-out test set.

The maps in holdout.py are never in train/ or val/, so this metric is
STABLE across training runs and comparable session-to-session — unlike
val F1, whose composition drifts as data is added.

Reports precision / recall / F1 at the decision threshold, an overall
line, and a per-map breakdown. Maps with no negative crops contribute
recall only (e.g. the job TIFFs, which so far have only ground-truth
positives) — those are flagged.

Usage:
    python evaluate_holdout.py                 # threshold 0.5
    python evaluate_holdout.py --threshold 0.85
"""
import sys
from pathlib import Path

import numpy as np
import cv2
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_paths import TRAINING_DATA
from holdout import HELD_OUT_MAP_IDS, held_out_map_of
from train_classifier import TriangleCNN
from qa_detections import classify_crops_chunked


def load_model(path=None):
    model = TriangleCNN()
    ckpt = torch.load(path or (SCRIPT_DIR / 'triangle_classifier.pth'),
                      map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


def gather(map_id, label):
    """All crops for one held-out map with a given label.

    label=1 -> positive/ + partial/   label=0 -> negative/ + hard_negatives/
    Returns list of (path, gray_crop).
    """
    if label == 1:
        dirs = ['positive', 'partial']
    else:
        dirs = ['negative', 'hard_negatives']
    out = []
    seen = set()
    for d in dirs:
        for f in (TRAINING_DATA / d).glob(f"{map_id}_*.png"):
            # The test set must contain only HUMAN-verified labels and must
            # never change. _dbh crops are machine-harvested (template-matched,
            # auto-labeled) and post-date the freeze — exclude them so the
            # test set stays fixed and trustworthy.
            if "_dbh" in f.name:
                continue
            if f.name in seen:
                continue
            seen.add(f.name)
            bgr = cv2.imread(str(f))
            if bgr is None:
                continue
            b = bgr[:, :, 0].astype(np.int16)
            g = bgr[:, :, 1].astype(np.int16)
            gray = np.minimum(b, g).astype(np.uint8)
            if gray.shape != (64, 64):
                gray = cv2.resize(gray, (64, 64))
            out.append((f, gray))
    return out


def main():
    threshold = 0.5
    if '--threshold' in sys.argv:
        threshold = float(sys.argv[sys.argv.index('--threshold') + 1])
    model_path = None
    if '--model' in sys.argv:
        model_path = sys.argv[sys.argv.index('--model') + 1]

    model, ckpt = load_model(model_path)
    print(f"Model: triangle_classifier.pth (trained F1={ckpt.get('f1', float('nan')):.3f})")
    print(f"Held-out test set: {len(HELD_OUT_MAP_IDS)} maps, threshold={threshold}\n")

    print(f"{'map':<12} {'pos':>4} {'neg':>5} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'prec':>6} {'rec':>6} {'F1':>6}")
    print("-" * 60)

    tot_tp = tot_fp = tot_fn = tot_tn = 0
    for mid in HELD_OUT_MAP_IDS:
        pos = gather(mid, 1)
        neg = gather(mid, 0)
        if not pos and not neg:
            print(f"{mid:<12}  (no crops found — check the id in holdout.py)")
            continue

        crops = [c for _, c in pos] + [c for _, c in neg]
        labels = np.array([1] * len(pos) + [0] * len(neg))
        probs = classify_crops_chunked(model, crops)
        preds = (probs >= threshold).astype(int)

        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        tot_tp += tp; tot_fp += fp; tot_fn += fn; tot_tn += tn

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        flag = "  (recall only — no negatives)" if not neg else ""
        print(f"{mid:<12} {len(pos):>4} {len(neg):>5} {tp:>4} {fp:>4} {fn:>4} "
              f"{prec:>6.3f} {rec:>6.3f} {f1:>6.3f}{flag}")

    prec = tot_tp / max(tot_tp + tot_fp, 1)
    rec = tot_tp / max(tot_tp + tot_fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print("-" * 60)
    print(f"{'OVERALL':<12} {tot_tp + tot_fn:>4} {tot_fp + tot_tn:>5} "
          f"{tot_tp:>4} {tot_fp:>4} {tot_fn:>4} "
          f"{prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")
    print(f"\nFrozen-test F1 = {f1:.3f}  "
          f"(precision {prec:.3f}, recall {rec:.3f})")
    print("This number is comparable across training runs. Track it.")


if __name__ == '__main__':
    main()
