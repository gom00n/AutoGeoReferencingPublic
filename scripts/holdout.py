"""
Frozen held-out test set.

A fixed list of WHOLE maps whose crops are never placed in train/ or val/.
They exist only to give a STABLE metric across training runs: because the
train/val split changes every time data is added (and val composition
drifts), val F1 is not comparable session-to-session. The held-out maps
never change, so evaluate_holdout.py on them IS comparable over time.

Rules for picking held-out maps (don't change the list casually — that
breaks comparability):
  - Each map is excluded ENTIRELY from train/val (every crop, both classes).
  - Prefer maps that have BOTH positive and negative crops so precision
    and recall are both meaningful.
  - Span the distributions we care about (archival 1:10k series + the
    600-dpi job TIFFs that are the project's end goal).

To add a held-out map later, append its id and rebuild the split
(apply_curate_labels.py) so its crops leave train/val.
"""

# Map-id prefixes. A crop belongs to a held-out map if its filename starts
# with "<id>_" (crop names are "<map_id>_x<col>_y<row>_<tag>.png").
HELD_OUT_MAP_IDS = [
    "M7_4138",    # sample-series   archival 1:10k — 33 pos / ~507 neg (balanced)
    "M9_4149",    # sample-series   archival 1:10k — 35 pos / ~223 neg (balanced)
    "M53_8930",   # sample-series  archival 1:10k — 21 pos / ~720 neg (balanced)
    "M9_0156",    # sample-series job 600-dpi TIFF — 14 pos / 0 neg (target recall)
]


def is_held_out(filename):
    """True if a crop filename belongs to a held-out test map."""
    name = filename.name if hasattr(filename, "name") else str(filename)
    return any(name.startswith(mid + "_") for mid in HELD_OUT_MAP_IDS)


def held_out_map_of(filename):
    """The held-out map id a crop belongs to, or None."""
    name = filename.name if hasattr(filename, "name") else str(filename)
    for mid in HELD_OUT_MAP_IDS:
        if name.startswith(mid + "_"):
            return mid
    return None
