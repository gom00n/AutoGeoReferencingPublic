#!/usr/bin/env python3
"""
Ingest job-georeferenced maps: inventory assets, recover transforms,
and extract training crops at the human-placed control points.

An archival project delivers 600-dpi TIFF scans (1:10,000 and 1:2,500 sheets)
plus georeferencing done in ArcGIS. Ground truth arrives in three shapes:

  1. <map>.tfwx / <map>.jgw      world file (final transform)
  2. <map>.txt / *controlpoints* control point export
                                 (old: header + mapX,mapY,pixelX,pixelY,enable
                                  new: 4 whitespace columns srcX srcY mapX mapY)
  3. <image>.aux.xml             ESRI GeodataXform with SourceGCPs/TargetGCPs

Every control point IS a triangle the colleagues clicked — perfect training
positives. Pixel positions are recovered as:
  - transform exists  -> pixel = inverse_transform(target_point)
  - no transform, aux source GCPs in inches (fresh 600-dpi scan) ->
    pixel = (sx*dpi, row_flip(sy*dpi)); the y-flip is resolved empirically
    by template-matching at both hypotheses (the winning side must contain
    triangles). The resulting pixel->map affine is then WRITTEN as <map>.tfwx
    so the rest of the pipeline can use the map.

Usage:
    python ingest_job_maps.py                      # inventory all <series>-* series
    python ingest_job_maps.py sample-series            # inventory one series
    python ingest_job_maps.py sample-series --extract  # extract crops + write tfwx
    python ingest_job_maps.py sample-series --extract --dry-run

Crops go to training_data/positive/ as <map_id>_x<px>_y<py>_gt.png and a
contact sheet (all crops side by side) is written next to the image for a
quick visual sanity check. Re-running skips crops that already exist.
After extraction: apply_curate_labels.py --source original, then retrain.
"""

import sys
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_paths import ground_truth_series, TRAINING_DATA, GEODETIC_DB
from coord_converter import load_tfwx, map_to_pixel, get_map_extent
from image_loader import load_image, suppress_red
from auto_georeference import compute_affine_transform, write_tfwx
from db_matcher import (
    load_grayscale_templates, match_crop_grayscale,
    load_geodetic_db, filter_points_to_extent, verify_candidates,
)

CROP_HALF = 32          # 64x64 crops, same as the rest of the training data
TARGET_DPI = 600.0      # training-crop resolution (600 dpi scans, scale 1.0)


# ---------------------------------------------------------------------------
# Ground-truth parsing
# ---------------------------------------------------------------------------

def parse_aux_gcps(aux_path):
    """Extract (source, target) GCP arrays from an ESRI .aux.xml.

    Returns (src Nx2, tgt Nx2) or None if the file has no GCPs.
    """
    try:
        root = ET.parse(aux_path).getroot()
    except ET.ParseError:
        return None
    arrs = {}
    for tag in ('SourceGCPs', 'TargetGCPs'):
        for e in root.iter():
            if e.tag.endswith(tag):
                vals = [float(k.text) for k in e if k.text]
                if vals and len(vals) % 2 == 0:
                    arrs[tag] = np.array(vals).reshape(-1, 2)
                break
    if 'SourceGCPs' in arrs and 'TargetGCPs' in arrs and \
            len(arrs['SourceGCPs']) == len(arrs['TargetGCPs']) >= 3:
        return arrs['SourceGCPs'], arrs['TargetGCPs']
    return None


def load_points_file(txt_path):
    """Load target map coordinates from a control point text file.

    Handles both formats:
      - old ArcGIS export: header line, comma-separated mapX,mapY,...
      - new job format: no header, whitespace columns srcX srcY mapX mapY
        (source columns are in ArcGIS layout units — ignored; only the
        map coordinates are reliable across files)

    Returns Nx2 array of (mapX, mapY).
    """
    points = []
    for line in Path(txt_path).read_text().splitlines():
        line = line.strip()
        if not line or line.lower().startswith('mapx'):
            continue
        parts = line.replace(',', ' ').split()
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        if len(vals) >= 4 and abs(vals[2]) > 10_000 and abs(vals[3]) > 10_000:
            points.append((vals[2], vals[3]))     # new: srcX srcY mapX mapY
        elif len(vals) >= 2 and abs(vals[0]) > 10_000:
            points.append((vals[0], vals[1]))     # old: mapX,mapY,...
    return np.array(points) if points else None


# ---------------------------------------------------------------------------
# Asset discovery
# ---------------------------------------------------------------------------

def find_assets(series_dir):
    """Map each map id in a series to its image / transform / points files.

    Looks both in per-map dirs (M<id>/) and flat folders like Scaned/.
    """
    series_dir = Path(series_dir)
    assets = {}

    def entry(map_id):
        return assets.setdefault(map_id, {
            'image': None, 'tfwx': None, 'points': None, 'aux': None})

    # Collect candidate files from map dirs and any flat subfolder
    search_dirs = [series_dir] + [d for d in series_dir.iterdir() if d.is_dir()]
    for d in search_dirs:
        for f in d.iterdir():
            if not f.is_file():
                continue
            name = f.name
            stem = name.split('.')[0]
            # *_controlpoints.txt belongs to the map, not a separate entry
            for suffix in ('_controlpoints', '_control_points'):
                if stem.endswith(suffix):
                    stem = stem[:-len(suffix)]
            if not stem.startswith('M'):
                continue
            low = name.lower()
            if low.endswith(('.tif', '.tiff', '.jpg', '.jpeg')):
                entry(stem)['image'] = f
            elif low.endswith(('.tfwx', '.jgw', '.tifw', '.wld')):
                # Prefer .tfwx over other world files (a .jgw may have been
                # written for a different-resolution JPG of the same map)
                e = entry(stem)
                if e['tfwx'] is None or low.endswith('.tfwx'):
                    e['tfwx'] = f
            elif low.endswith('.aux.xml'):
                entry(stem)['aux'] = f
            elif low.endswith('.txt') and ('controlpoint' in low or
                                           low == f'{stem.lower()}.txt'):
                entry(stem)['points'] = f
            # other sidecars (*.png, *.xml, Thumbs.db) are ignored
    return assets


# ---------------------------------------------------------------------------
# Transform recovery from inch-style aux GCPs
# ---------------------------------------------------------------------------

def _score_positions(gray, positions, templates, crop_half=50):
    """Mean best template-match confidence over crops at given positions."""
    h, w = gray.shape[:2]
    scores = []
    for px, py in positions:
        x, y = int(round(px)), int(round(py))
        if x - crop_half < 0 or y - crop_half < 0 or \
                x + crop_half >= w or y + crop_half >= h:
            continue
        crop = gray[y - crop_half:y + crop_half, x - crop_half:x + crop_half]
        conf, _, _, _ = match_crop_grayscale(crop, templates)
        scores.append(conf)
    return float(np.mean(scores)) if scores else -1.0, len(scores)


def transform_from_inch_gcps(img_gray, src, tgt, dpi, templates):
    """Recover a pixel->map affine from inch-frame source GCPs.

    The y axis direction of the source frame is ambiguous (an affine fit
    absorbs a flip, so residuals can't tell). Resolve it empirically: the
    GCPs are exactly where triangles sit, so template-match both hypotheses
    and keep the one whose crops actually contain triangles.

    Returns (affine_dict, pixel_positions, label) or (None, None, reason).
    """
    h, w = img_gray.shape[:2]
    hyps = {
        'y-up':   np.column_stack([src[:, 0] * dpi[0], h - src[:, 1] * dpi[1]]),
        'y-down': np.column_stack([src[:, 0] * dpi[0], src[:, 1] * dpi[1]]),
    }
    scored = {}
    for label, pixels in hyps.items():
        score, n_valid = _score_positions(img_gray, pixels, templates)
        scored[label] = (score, n_valid, pixels)
    best = max(scored, key=lambda k: scored[k][0])
    score, n_valid, pixels = scored[best]
    other = min(scored.values(), key=lambda v: v[0])[0]

    if n_valid < 3:
        return None, None, "GCP positions fall outside the image"
    if score < 0.3 or score - other < 0.05:
        return None, None, (f"orientation ambiguous (best {best} "
                            f"score={score:.2f} vs {other:.2f})")

    affine = compute_affine_transform(pixels, tgt)
    return affine, pixels, f"{best}, match={score:.2f}, fit RMSE={affine['rmse_meters']:.1f}m"


# ---------------------------------------------------------------------------
# Crop extraction
# ---------------------------------------------------------------------------

def _existing_positions(out_dir, map_id):
    """Pixel positions of ALL crops already extracted for this map (any tag).

    Cross-tag so harvest (_dbh) doesn't re-capture a control point (_gt)
    or vice versa — the same triangle must not enter training twice.
    """
    import re
    positions = []
    for f in out_dir.glob(f"{map_id}_x*_y*.png"):
        m = re.search(r'_x(\d+)_y(\d+)_', f.name)
        if m:
            positions.append((int(m.group(1)), int(m.group(2))))
    return positions


def extract_crops(img_gray, pixel_positions, map_id, out_dir, dry_run=False,
                  tag='gt', confs=None):
    """Save 64x64 red-suppressed crops at pixel positions. Returns count.

    Idempotent by PROXIMITY across all tags: different transform paths or
    the harvest pass place the same triangle a few px apart, so an existing
    crop within 10px counts as already extracted.

    If confs is given (one per position), it is embedded in the filename as
    _c<conf> so review_positives.py --sort confidence shows worst-first.
    Crops are returned sorted by confidence ascending for the contact sheet.
    """
    h, w = img_gray.shape[:2]
    existing = _existing_positions(out_dir, map_id)
    if confs is None:
        confs = [None] * len(pixel_positions)
    order = range(len(pixel_positions))
    if confs[0] is not None:
        order = sorted(order, key=lambda i: confs[i])  # worst first
    saved, skipped = 0, 0
    crops_for_sheet = []
    for i in order:
        px, py = pixel_positions[i]
        x, y = int(round(px)), int(round(py))
        if x - CROP_HALF < 0 or y - CROP_HALF < 0 or \
                x + CROP_HALF >= w or y + CROP_HALF >= h:
            skipped += 1
            continue
        crop = img_gray[y - CROP_HALF:y + CROP_HALF, x - CROP_HALF:x + CROP_HALF]
        crops_for_sheet.append(crop)
        if any((x - ex) ** 2 + (y - ey) ** 2 < 10 ** 2 for ex, ey in existing):
            skipped += 1
            continue
        cstr = f"_c{confs[i]:.2f}" if confs[i] is not None else ""
        if not dry_run:
            cv2.imwrite(str(out_dir / f"{map_id}_x{x}_y{y}{cstr}_{tag}.png"),
                        cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR))
        existing.append((x, y))
        saved += 1
    return saved, skipped, crops_for_sheet


def write_contact_sheet(crops, dest_path, cols=8, cell=68):
    """Side-by-side sheet of all extracted crops for a quick eyeball check."""
    if not crops:
        return
    rows = (len(crops) + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell), 220, dtype=np.uint8)
    for i, c in enumerate(crops):
        r, col = divmod(i, cols)
        y0, x0 = r * cell + 2, col * cell + 2
        sheet[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] = c
    cv2.imwrite(str(dest_path), sheet)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def harvest_series(series_dir, geo_db, templates, harvest_conf=0.60,
                   dry_run=False):
    """Harvest extra triangle positives from now-georeferenced maps.

    Control points are a sparse subset (~20/map); a map has 100s of
    geodetic triangles in extent. For each map with a triangle-scale
    .tfwx, project the FULL geodetic DB and template-match (CNN-INDEPENDENT
    — projecting known DB points and confirming with template matching
    avoids the classifier reinforcing its own errors). Keep high-confidence
    matches not already captured as control points.

    Saves crops as <map_id>_x_y_dbh.png (db-harvested) + a contact sheet.
    """
    print(f"\n{'=' * 70}\n  HARVEST  {series_dir.name}\n{'=' * 70}")
    pos_dir = TRAINING_DATA / 'positive'
    if not dry_run:
        pos_dir.mkdir(parents=True, exist_ok=True)
    assets = find_assets(series_dir)
    total = 0

    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    from holdout import HELD_OUT_MAP_IDS

    for map_id in sorted(assets):
        a = assets[map_id]
        if a['image'] is None or a['tfwx'] is None:
            continue
        if map_id in HELD_OUT_MAP_IDS:    # never harvest a frozen test map
            print(f"  {map_id:<10} held-out test map — skipped")
            continue
        affine = load_tfwx(a['tfwx'])
        if abs(affine['a']) < 0.2:        # 1:2,500 circle sheet — skip
            continue

        with Image.open(a['image']) as im:
            w, h = im.size
        extent = get_map_extent(affine, w, h)
        candidates = filter_points_to_extent(geo_db, extent)
        if not candidates:
            continue

        img = load_image(str(a['image']))
        # suppress_red color mode keeps it fast; templates already match
        # this resolution (~0.85 m/px)
        detections = verify_candidates(img, candidates, affine, templates,
                                       color_mode='suppress_red')
        kept = [d for d in detections if d.confidence >= harvest_conf]
        gray = suppress_red(img)
        pixels = np.array([(d.pixel_x, d.pixel_y) for d in kept], dtype=np.float64)
        confs = [float(d.confidence) for d in kept]

        saved, skipped, crops = extract_crops(
            gray, pixels, map_id, pos_dir, dry_run=dry_run, tag='dbh', confs=confs)
        if not dry_run and crops:
            write_contact_sheet(
                crops[:64], a['image'].parent / f"{map_id}_dbh_crops.png")
        total += saved
        print(f"  {map_id:<10} {len(candidates):>4} DB pts in extent, "
              f"{len(kept):>3} matched >= {harvest_conf}, "
              f"{saved} new harvested, {skipped} already had a crop")

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Harvested {total} crops from "
          f"{series_dir.name}")
    return total


def process_series(series_dir, extract=False, dry_run=False):
    print(f"\n{'=' * 70}\n  {series_dir.name}\n{'=' * 70}")
    assets = find_assets(series_dir)
    templates = load_grayscale_templates(SCRIPT_DIR / 'templates') if extract else None
    pos_dir = TRAINING_DATA / 'positive'
    if extract and not dry_run:
        pos_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    for map_id in sorted(assets):
        a = assets[map_id]
        have = lambda k: '+' if a[k] else '-'
        gcps = parse_aux_gcps(a['aux']) if a['aux'] else None
        points = load_points_file(a['points']) if a['points'] else None
        n_pts = len(points) if points is not None else (len(gcps[1]) if gcps else 0)
        status = (f"img:{have('image')} tfwx:{have('tfwx')} "
                  f"points:{n_pts:>2}")

        if a['image'] is None:
            print(f"  {map_id:<10} {status}  -> waiting for image")
            continue
        if a['tfwx'] is None and gcps is None:
            print(f"  {map_id:<10} {status}  -> waiting for transform/GCPs")
            continue
        targets = points if points is not None else (gcps[1] if gcps else None)
        if targets is None or len(targets) == 0:
            print(f"  {map_id:<10} {status}  -> waiting for control points")
            continue

        # Skip maps that already contributed crops to positive/ (the old
        # series were ingested historically under the same map-id prefix);
        # re-extracting would put the same triangle in train AND val
        existing = list((TRAINING_DATA / 'positive').glob(f'{map_id}_*'))
        if existing and not any(f.name.endswith('_gt.png') for f in existing):
            print(f"  {map_id:<10} {status}  -> already in training data "
                  f"({len(existing)} crops), skipping")
            continue

        if not extract:
            print(f"  {map_id:<10} {status}  -> READY ({len(targets)} points)")
            continue

        # --- extraction path ---
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(a['image']) as im:
            dpi = tuple(float(v) for v in im.info.get('dpi', (TARGET_DPI,) * 2))
        img = load_image(str(a['image']))
        scale = round((TARGET_DPI / dpi[0]) * 2) / 2 if dpi[0] else 1.0
        if abs(scale - 1.0) < 0.01:
            scale = 1.0
        if scale != 1.0:
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                             interpolation=cv2.INTER_LINEAR)
            dpi = (dpi[0] * scale, dpi[1] * scale)
            print(f"  {map_id:<10} rescaled {scale:g}x to match {TARGET_DPI:.0f} dpi")
        gray = suppress_red(img)

        m_per_px = [None]  # set by whichever transform path succeeds

        def pixels_via_world_file(path):
            affine = load_tfwx(path)
            m_per_px[0] = abs(affine['a'])
            if scale != 1.0:
                # World file refers to original resolution: col_new = col*scale
                # so all linear coefficients shrink by scale; rebuild matrices
                for k in ('a', 'b', 'c', 'd'):
                    affine[k] /= scale
                M = np.array([[affine['a'], affine['c']],
                              [affine['b'], affine['d']]])
                affine['M'], affine['M_inv'] = M, np.linalg.inv(M)
                affine['offset'] = np.array([affine['e'], affine['f']])
                affine['forward'] = np.array(
                    [[affine['a'], affine['c'], affine['e']],
                     [affine['b'], affine['d'], affine['f']]])
            return np.array([map_to_pixel(tx, ty, affine) for tx, ty in targets],
                            dtype=np.float64)

        def pixels_via_inch_gcps():
            """Recover transform from inch-frame aux GCPs; writes a tfwx."""
            nonlocal targets
            if gcps is None:
                return None, "no aux GCPs to fall back to"
            src, tgt = gcps
            if np.abs(src).max() > 1000:
                return None, "aux GCPs are in a prior georef frame"
            affine, px, why = transform_from_inch_gcps(gray, src, tgt, dpi, templates)
            if affine is None:
                return None, why
            m_per_px[0] = abs(affine['a'])
            targets = tgt
            tfwx_out = a['image'].with_suffix('.tfwx')
            if not dry_run:
                write_tfwx(affine, tfwx_out)
            print(f"  {map_id:<10} {'[dry-run] would write' if dry_run else 'wrote'} "
                  f"{tfwx_out.name} ({why})")
            return px, f"aux GCPs ({why})"

        # Validate every transform before trusting it: each control point IS
        # a triangle, so crops at the projected positions must look like
        # triangles. Catches wrong world files (e.g. a .jgw made for a
        # different-resolution JPG applied to the TIFF), bad GCP frames, etc.
        pixels, how = None, None
        if a['tfwx'] is not None:
            candidate = pixels_via_world_file(a['tfwx'])
            score, n_valid = _score_positions(gray, candidate, templates)
            if n_valid >= 3 and score >= 0.30:
                pixels, how = candidate, f"tfwx ({a['tfwx'].name}, match={score:.2f})"
            else:
                print(f"  {map_id:<10} {a['tfwx'].name} SUSPECT for this image "
                      f"(template match {score:.2f} at {n_valid} points) — "
                      f"trying aux GCPs")
        if pixels is None:
            pixels, how = pixels_via_inch_gcps()
            if pixels is None:
                print(f"  {map_id:<10} {status}  -> no usable transform ({how})")
                continue

        # Symbol gate: 1:2,500 sheets (~0.1 m/px) mark geodetic points with
        # CIRCLE-dot symbols, not triangles (verified on M26/M27 contact
        # sheets). Feeding them to the triangle CNN as positives would
        # poison it — recover the transform (done above) but skip crops.
        if m_per_px[0] is not None and m_per_px[0] < 0.2:
            print(f"  {map_id:<10} {status}  -> 1:2,500 sheet "
                  f"({m_per_px[0]:.2f} m/px): symbols are circles, not "
                  f"triangles — transform recovered, crops NOT extracted")
            continue

        saved, skipped, crops = extract_crops(gray, pixels, map_id, pos_dir,
                                              dry_run=dry_run)
        sheet = a['image'].parent / f"{map_id}_gt_crops.png"
        if not dry_run:
            write_contact_sheet(crops, sheet)
        total_saved += saved
        print(f"  {map_id:<10} {status}  -> {saved} crops saved, "
              f"{skipped} skipped via {how}")
        if not dry_run and crops:
            print(f"  {'':<10} contact sheet: {sheet}")

    return total_saved


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    extract = '--extract' in sys.argv
    harvest = '--harvest' in sys.argv
    dry_run = '--dry-run' in sys.argv

    series = ground_truth_series()
    if args:
        series = [s for s in series if any(a in s.name for a in args)]
    if not series:
        print("No matching series under Map_Scans/")
        sys.exit(1)

    if harvest:
        print("Loading geodetic DB and templates for harvest...")
        geo_db = load_geodetic_db(GEODETIC_DB)
        templates = load_grayscale_templates(SCRIPT_DIR / 'templates')
        total = 0
        for s in series:
            total += harvest_series(s, geo_db, templates, dry_run=dry_run)
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Total harvested: {total}")
        if total and not dry_run:
            print("Review the *_dbh_crops.png contact sheets, then:")
            print("  apply_curate_labels.py --source original && train_classifier.py")
        return

    total = 0
    for s in series:
        total += process_series(s, extract=extract, dry_run=dry_run)

    if extract:
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Total crops saved: {total}")
        if total and not dry_run:
            print("\nNext steps:")
            print("  /opt/anaconda3/bin/python apply_curate_labels.py --source original")
            print("  /opt/anaconda3/bin/python train_classifier.py")
    else:
        print("\nInventory only. Run with --extract to extract crops.")


if __name__ == '__main__':
    main()
