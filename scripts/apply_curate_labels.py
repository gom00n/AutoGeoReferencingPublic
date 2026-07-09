"""
Apply curated labels to reorganize training data.

Actions:
1. From positives curation:
   - pos: keep in positive/
   - partial: move to partial/ (separate folder)
   - neg: move from positive/ to negative/

2. From hard_negatives curation:
   - pos: copy to positive/ (rescued triangles!)
   - partial: copy to partial/
   - neg: keep in hard_negatives/ (confirmed)

3. From qa_candidates curation (--source qa):
   - pos: copy to positive/
   - partial: copy to partial/
   - neg: copy to hard_negatives/

4. Rebuild train/val split with clean data:
   - train: 80% positive + negatives + confirmed hard_negatives
   - val: 20% positive + negatives + confirmed hard_negatives

Usage:
    python apply_curate_labels.py [--dry-run]
    python apply_curate_labels.py --source qa [--dry-run]
    python apply_curate_labels.py --source all [--dry-run]
"""

import json
import shutil
import random
from pathlib import Path
from collections import Counter


def load_labels_file(path):
    """Load a curate_labels JSON, return dict or empty dict if missing."""
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def process_original_labels(pos_dir, neg_dir, hn_dir, partial_dir, labels_dir):
    """Process the original positives + hard_negatives labels."""
    moves = []

    pos_labels_path = labels_dir / "curate_labels_positives.json"
    hn_labels_path = labels_dir / "curate_labels_hard_negatives.json"

    pos_labels = load_labels_file(pos_labels_path)
    hn_labels = load_labels_file(hn_labels_path)

    if not pos_labels and not hn_labels:
        print("  No original curation labels found, skipping.")
        return moves

    if pos_labels:
        print(f"Positive labels: {Counter(pos_labels.values())}")
    if hn_labels:
        print(f"Hard negative labels: {Counter(hn_labels.values())}")

    # 1. Process positive labels
    for fname, label in pos_labels.items():
        src = pos_dir / fname
        if not src.exists():
            continue
        if label == "neg":
            moves.append((src, neg_dir / fname, "pos→neg"))
        elif label == "partial":
            moves.append((src, partial_dir / fname, "pos→partial"))

    # 2. Process hard_negative labels (copy, don't move originals)
    for fname, label in hn_labels.items():
        src = hn_dir / fname
        if not src.exists():
            continue
        if label == "pos":
            moves.append((src, pos_dir / fname, "hn→pos (RESCUED)"))
        elif label == "partial":
            moves.append((src, partial_dir / fname, "hn→partial"))

    return moves


def process_qa_labels(qa_dir, pos_dir, hn_dir, partial_dir, labels_dir):
    """Process QA-sourced candidate labels."""
    moves = []

    # Look for QA labels in labels_dir first, then in qa_dir
    qa_labels = load_labels_file(labels_dir / "curate_labels_qa_candidates.json")
    if not qa_labels:
        qa_labels = load_labels_file(qa_dir / "curate_labels.json")
    if not qa_labels:
        # Try the default download location
        qa_labels = load_labels_file(
            Path(__file__).parent.parent / "curate_labels.json")

    if not qa_labels:
        print("  No QA curation labels found.")
        print(f"  Expected at: {labels_dir / 'curate_labels_qa_candidates.json'}")
        print(f"           or: {qa_dir / 'curate_labels.json'}")
        return moves

    print(f"QA candidate labels: {Counter(qa_labels.values())}")

    for fname, label in qa_labels.items():
        src = qa_dir / fname
        if not src.exists():
            # Try without extension variations
            stem = Path(fname).stem
            candidates = list(qa_dir.glob(f"{stem}.*"))
            if candidates:
                src = candidates[0]
            else:
                continue

        if label == "pos":
            moves.append((src, pos_dir / src.name, "qa→pos"))
        elif label == "neg":
            moves.append((src, hn_dir / src.name, "qa→hn"))
        elif label == "partial":
            moves.append((src, partial_dir / src.name, "qa→partial"))

    return moves


def rebuild_train_val(base, pos_dir, neg_dir, hn_dir, partial_dir):
    """Rebuild train/val split from current data directories."""
    train_dir = base / "train"
    val_dir = base / "val"

    print("\n=== Rebuilding train/val split ===")

    from holdout import is_held_out, HELD_OUT_MAP_IDS

    def without_holdout(files):
        return [f for f in files if not is_held_out(f)]

    all_pos = without_holdout(sorted(pos_dir.glob("*.png")))
    all_neg = without_holdout(sorted(neg_dir.glob("*.png")))
    all_hn = without_holdout(sorted(hn_dir.glob("*.png")))
    partials_all = without_holdout(
        sorted(partial_dir.glob("*.png")) if partial_dir.exists() else [])

    n_held = (len(list(pos_dir.glob("*.png"))) - len(all_pos))
    print(f"  Held out {n_held} crops from {len(HELD_OUT_MAP_IDS)} frozen "
          f"test maps (see holdout.py) — excluded from train/val")

    # Rescued crops (hn→pos, hn→partial) are COPIED, so the original stays
    # in hard_negatives/. Exclude anything that is now a positive or partial
    # from the negatives pool — otherwise the same image trains as both
    # a positive and a negative.
    pos_names = {f.name for f in all_pos} | {f.name for f in partials_all}
    n_conflicts = sum(1 for f in all_neg + all_hn if f.name in pos_names)
    if n_conflicts:
        print(f"  Excluding {n_conflicts} negatives that were rescued as pos/partial")
    all_neg = [f for f in all_neg if f.name not in pos_names]
    all_hn = [f for f in all_hn if f.name not in pos_names]

    # Deduplicate by filename — prefer negative/ over hard_negatives/ if same name
    seen_names = {f.name for f in all_neg}
    all_hn_deduped = [f for f in all_hn if f.name not in seen_names]
    all_neg_combined = all_neg + all_hn_deduped

    partials = partials_all
    eff_pos = len(all_pos) + len(partials)
    eff_neg = min(len(all_neg_combined), eff_pos * 3)
    print(f"  Positives: {len(all_pos)} + {len(partials)} partials = {eff_pos} effective")
    print(f"  Negatives: {len(all_neg)} base + {len(all_hn)} hard neg = {len(all_neg_combined)} "
          f"(capped to ~{eff_neg} at training)")
    if partial_dir.exists():
        print(f"  Partials (used as positives in training): {len(list(partial_dir.glob('*.png')))}")

    # Shuffle and split 80/20
    random.seed(42)
    random.shuffle(all_pos)
    random.shuffle(all_neg_combined)

    split = 0.8
    train_pos = all_pos[:int(len(all_pos) * split)]
    val_pos = all_pos[int(len(all_pos) * split):]
    train_neg = all_neg_combined[:int(len(all_neg_combined) * split)]
    val_neg = all_neg_combined[int(len(all_neg_combined) * split):]

    print(f"  Train: {len(train_pos)} pos + {len(train_neg)} neg")
    print(f"  Val:   {len(val_pos)} pos + {len(val_neg)} neg")

    # Clear old split dirs
    for subdir in ["positive", "negative"]:
        for parent in [train_dir, val_dir]:
            d = parent / subdir
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    # Symlink files into split dirs
    for f in train_pos:
        (train_dir / "positive" / f.name).symlink_to(f.resolve())
    for f in val_pos:
        (val_dir / "positive" / f.name).symlink_to(f.resolve())
    for f in train_neg:
        (train_dir / "negative" / f.name).symlink_to(f.resolve())
    for f in val_neg:
        (val_dir / "negative" / f.name).symlink_to(f.resolve())

    print(f"\n  train/positive: {len(list((train_dir / 'positive').iterdir()))}")
    print(f"  train/negative: {len(list((train_dir / 'negative').iterdir()))}")
    print(f"  val/positive:   {len(list((val_dir / 'positive').iterdir()))}")
    print(f"  val/negative:   {len(list((val_dir / 'negative').iterdir()))}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without moving files")
    parser.add_argument("--source", default="original",
                        choices=["original", "qa", "all"],
                        help="Which label source to apply: original, qa, or all")
    args = parser.parse_args()

    base = Path(__file__).parent.parent / "training_data"
    labels_dir = base / "Data_labeling" / "Triangle_detection"

    pos_dir = base / "positive"
    neg_dir = base / "negative"
    hn_dir = base / "hard_negatives"
    partial_dir = base / "partial"
    qa_dir = base / "qa_candidates"

    # Create dirs
    if not args.dry_run:
        for d in [pos_dir, neg_dir, hn_dir, partial_dir]:
            d.mkdir(exist_ok=True)

    moves = []

    if args.source in ("original", "all"):
        print("--- Original labels (positives + hard_negatives) ---")
        moves.extend(process_original_labels(
            pos_dir, neg_dir, hn_dir, partial_dir, labels_dir))

    if args.source in ("qa", "all"):
        print("--- QA candidate labels ---")
        moves.extend(process_qa_labels(
            qa_dir, pos_dir, hn_dir, partial_dir, labels_dir))

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Actions ({len(moves)} file moves):")
    action_counts = Counter(m[2] for m in moves)
    for action, count in action_counts.most_common():
        print(f"  {action}: {count}")

    if args.dry_run:
        for src, dst, action in moves:
            print(f"  {action}: {src.name} → {dst.parent.name}/")
        return

    # Execute moves
    for src, dst, action in moves:
        if dst.exists():
            print(f"  SKIP (exists): {dst.name}")
            continue
        if "qa→" in action or "RESCUED" in action or "hn→partial" in action:
            # Copy (keep originals)
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))
        print(f"  {action}: {src.name}")

    # Rebuild train/val split
    rebuild_train_val(base, pos_dir, neg_dir, hn_dir, partial_dir)
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
