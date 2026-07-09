"""
Regression tests for the pure-logic parts of the georeferencing pipeline.

Covers OCR label post-processing (grid_label_ocr), affine fitting
(bootstrap_from_grid, auto_georeference), and coordinate conversion
(coord_converter). No images, OCR engine, or model files needed.

Run directly:   /opt/anaconda3/bin/python test_pipeline.py
Or via pytest:  /opt/anaconda3/bin/python -m pytest test_pipeline.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grid_label_ocr import (
    fix_ocr_digit_confusion, validate_grid_sequence,
    merge_edge_detections, labels_to_affine,
    digit_confusion_variants, rescue_out_of_range,
    consensus_labels,
)
from coord_converter import (
    load_tfwx, pixel_to_map, map_to_pixel,
    pixel_to_map_batch, map_to_pixel_batch,
)
from bootstrap_from_grid import (
    build_old_grid_affine, write_tfwx,
    labels_to_grid_points, labels_to_old_grid_extent,
    parse_sheet_number, sheet_label_ranges,
)
from auto_georeference import compute_affine_transform


# ---------------------------------------------------------------------------
# fix_ocr_digit_confusion
# ---------------------------------------------------------------------------

def test_digit_fix_units_8_to_3():
    # 133 misread as 138 — same tens group, only the substitution pass can fix it
    dets = [(1000, 130, 0.9), (2000, 131, 0.9), (3000, 132, 0.9), (4000, 138, 0.9)]
    fixed = fix_ocr_digit_confusion(dets)
    assert [v for _, v, _ in fixed] == [130, 131, 132, 133]


def test_digit_fix_tens_group_merge():
    # 131 misread as 181 — minority 18X group folded into majority 13X
    dets = [(1000, 130, 0.9), (2000, 181, 0.9), (3000, 132, 0.9), (4000, 133, 0.9)]
    fixed = fix_ocr_digit_confusion(dets)
    assert [v for _, v, _ in fixed] == [130, 131, 132, 133]


def test_digit_fix_clean_sequence_unchanged():
    dets = [(1000, 140, 0.9), (2000, 141, 0.9), (3000, 142, 0.9)]
    fixed = fix_ocr_digit_confusion(dets)
    assert [v for _, v, _ in fixed] == [140, 141, 142]
    assert all(abs(c - 0.9) < 1e-9 for _, _, c in fixed)  # confidence untouched


def test_digit_fix_short_input_passthrough():
    dets = [(1000, 130, 0.9), (2000, 138, 0.9)]
    assert fix_ocr_digit_confusion(dets) == dets


def test_digit_confusion_variants():
    assert 131 in digit_confusion_variants(181)  # 8->3
    assert 181 in digit_confusion_variants(131)  # 3->8
    assert 131 not in digit_confusion_variants(131)  # excludes self


def test_rescue_out_of_range():
    # ErRamle regression: easting "131" read as "181" with conf 1.0;
    # with sheet range 129-141, the unique in-range variant is 131
    assert rescue_out_of_range(181, (129, 141)) == 131
    # Junk with no in-range variant is dropped
    assert rescue_out_of_range(61, (129, 141)) is None
    # Ambiguous (two variants in range) is dropped, not guessed:
    # 86 -> {36, 6, 85, 80}; both 36 and 80 are in (30, 80)
    assert rescue_out_of_range(86, (30, 80)) is None


# ---------------------------------------------------------------------------
# validate_grid_sequence
# ---------------------------------------------------------------------------

def test_validate_keeps_consistent_sequence():
    dets = [(1000, 130, 0.9), (2200, 131, 0.9), (3400, 132, 0.9)]
    assert validate_grid_sequence(dets) == dets


def test_validate_drops_spacing_outlier():
    # 999 at a position implying an absurd px/km rate
    dets = [(1000, 130, 0.9), (2200, 131, 0.9), (3400, 132, 0.9), (3500, 250, 0.9)]
    valid = validate_grid_sequence(dets)
    assert [v for _, v, _ in valid] == [130, 131, 132]


def test_validate_rejects_all_when_no_plausible_rate():
    # Two values 1km apart but only 10px apart — no valid rate
    dets = [(1000, 130, 0.9), (1010, 131, 0.9)]
    assert validate_grid_sequence(dets) == []


# ---------------------------------------------------------------------------
# merge_edge_detections
# ---------------------------------------------------------------------------

def test_merge_averages_agreeing_edges():
    top = [(5000, 145, 0.8)]
    bottom = [(5040, 145, 0.6)]
    merged = merge_edge_detections(top, 'top', bottom, 'bottom')
    assert len(merged) == 1
    pos, val, conf, edge = merged[0]
    assert val == 145 and edge == 'both'
    assert abs(pos - 5020) < 1e-9
    assert abs(conf - 0.8) < 1e-9


def test_merge_keeps_single_edge_label():
    merged = merge_edge_detections([(5000, 145, 0.8)], 'top', [], 'bottom')
    assert merged == [(5000, 145, 0.8, 'top')]


def test_merge_conflict_keeps_higher_confidence():
    # Same value at columns 4000px apart — one edge misread.
    # Averaging would put it at 7000, wrong for both. Keep the confident one.
    top = [(5000, 145, 0.9)]
    bottom = [(9000, 145, 0.4)]
    merged = merge_edge_detections(top, 'top', bottom, 'bottom')
    assert merged == [(5000, 145, 0.9, 'top')]


# ---------------------------------------------------------------------------
# labels_to_affine
# ---------------------------------------------------------------------------

def _synthetic_label_result(px_per_km=1183.0, col0=891, row0=934):
    """Labels mimicking a real sheet: eastings 140-150 km left→right,
    northings 130-120 km top→bottom."""
    eastings = [(col0 + (e - 140) * px_per_km, e, 0.9, 'both')
                for e in range(140, 151)]
    northings = [(row0 + (130 - n) * px_per_km, n, 0.9, 'both')
                 for n in range(130, 119, -1)]
    return {'easting_labels': eastings, 'northing_labels': northings,
            'neatline': {'top': row0, 'bottom': row0 + 10 * px_per_km,
                         'left': col0, 'right': col0 + 10 * px_per_km},
            'h_grid_lines': [], 'v_grid_lines': []}


def test_affine_from_clean_labels():
    px_per_km = 1183.0
    affine = labels_to_affine(_synthetic_label_result(px_per_km))
    assert affine is not None
    expected_m_per_px = 1000.0 / px_per_km
    assert abs(affine['pixel_size_x'] - expected_m_per_px) < 0.01
    assert abs(affine['pixel_size_y'] - expected_m_per_px) < 0.01
    assert affine['a'] > 0          # easting increases left→right
    assert affine['d'] < 0          # northing decreases top→bottom
    assert affine['easting_rmse_m'] < 1.0
    assert affine['northing_rmse_m'] < 1.0
    # Round-trip a known label: col of easting 145 must map to 145,000 m
    col_145 = 891 + 5 * px_per_km
    assert abs((affine['a'] * col_145 + affine['e']) - 145000) < 50


def test_affine_survives_one_bad_label():
    result = _synthetic_label_result()
    # Corrupt one easting label value (e.g., unfixed OCR misread)
    pos, _, conf, edge = result['easting_labels'][5]
    result['easting_labels'][5] = (pos, 195, conf, edge)
    affine = labels_to_affine(result)
    assert affine is not None
    assert abs(affine['pixel_size_x'] - 1000.0 / 1183.0) < 0.01
    assert affine['easting_rmse_m'] < 5.0


def test_affine_survives_junk_label_at_sequence_end():
    # Regression: a junk low-confidence detection at the END of an ascending
    # sequence used to flip the inferred direction (endpoints comparison),
    # causing the real labels to be removed instead of the junk one.
    # Seen on 13-14-ErRamle-1948: "68km" misread after labels 181-189.
    px_per_km = 593.0
    eastings = [(1106 + (e - 181) * px_per_km, e, 0.9, 'top')
                for e in range(181, 190)]
    eastings.append((6045, 68, 0.36, 'bottom'))  # junk, position-sorted last
    result = _synthetic_label_result()
    result['easting_labels'] = sorted(eastings)
    # northing labels at matching scale
    result['northing_labels'] = [(1246 + (149 - n) * px_per_km, n, 0.9, 'right')
                                 for n in range(149, 139, -1)]
    affine = labels_to_affine(result)
    assert affine is not None
    expected = 1000.0 / px_per_km
    assert abs(affine['pixel_size_x'] - expected) < 0.01
    assert abs(affine['pixel_size_y'] - expected) < 0.01
    assert affine['easting_rmse_m'] < 5.0


def test_affine_drops_low_confidence_misread_on_tie():
    # Regression from 14-15-Lydda-1942: "141" misread as "138" (conf 0.59)
    # ties 1-1 in monotonic violations with the REAL "140" (conf 1.00) next
    # to it. The tie must be broken by confidence, or the real label gets
    # removed and the misread tilts the fit (easting axis was off by 16%).
    px_per_km = 590.0
    eastings = [(782, 140, 1.0, 'top'), (1216, 138, 0.59, 'bottom')]
    eastings += [(782 + (e - 140) * px_per_km, e, 1.0, 'both')
                 for e in range(142, 151)]
    result = _synthetic_label_result()
    result['easting_labels'] = sorted(eastings)
    result['northing_labels'] = [(690 + (160 - n) * px_per_km, n, 1.0, 'both')
                                 for n in range(160, 149, -1)]
    affine = labels_to_affine(result)
    assert affine is not None
    expected = 1000.0 / px_per_km
    assert abs(affine['pixel_size_x'] - expected) < 0.02, affine['pixel_size_x']
    assert affine['easting_rmse_m'] < 30.0, affine['easting_rmse_m']


def test_affine_survives_junk_cluster_flipping_direction():
    # Regression (M5_4048 etc., sample-series series): the real northing series is
    # DESCENDING (130→120 top→bottom), but a cluster of spurious low-value
    # OCR misreads (50-100 km) scattered through the margin made the
    # sign-of-diffs direction vote come out ASCENDING — so the old
    # remove_non_monotonic deleted the real labels and the fit was rejected
    # as "pixel size ratio too skewed". The consensus filter keeps the
    # collinear real grid regardless of how much junk is mixed in.
    px_per_km = 1176.0
    real = [(910 + (130 - n) * px_per_km, n, 1.0, 'left')
            for n in range(130, 119, -1)]            # 11 real labels
    junk = [(1082, 63, 0.29, 'right'), (1266, 69, 0.99, 'right'),
            (1928, 96, 0.23, 'left'), (4354, 59, 1.00, 'right'),
            (4982, 93, 0.64, 'left'), (11400, 46, 0.20, 'right')]
    result = _synthetic_label_result(px_per_km=px_per_km)
    result['northing_labels'] = sorted(real + junk)
    affine = labels_to_affine(result)
    assert affine is not None
    expected = 1000.0 / px_per_km
    assert abs(affine['pixel_size_y'] - expected) < 0.02, affine['pixel_size_y']
    assert affine['d'] < 0, affine['d']          # northing still descends
    assert affine['northing_rmse_m'] < 5.0, affine['northing_rmse_m']
    # the junk must not survive into the labels handed downstream
    used_vals = {v for _, v, *_ in affine['northing_labels_used']}
    assert used_vals <= set(range(120, 131)), used_vals


def test_consensus_labels_picks_larger_collinear_set():
    # Two competing self-consistent series (left margin 130-138, right margin
    # systematically misread 40 km high as 170-178): the consensus keeps the
    # larger set. Downstream template matching validates the geometry, so the
    # larger consensus is the safe default when there is no sheet number.
    px = 1170.0
    left = [(1026 + (140 - n) * px, n, 1.0, 'left') for n in range(140, 129, -1)]
    right = [(940 + (180 - n) * px, n, 0.9, 'right') for n in range(180, 173, -1)]
    kept = consensus_labels(sorted(left + right))
    kept_vals = sorted(v for _, v, *_ in kept)
    assert kept_vals == list(range(130, 141)), kept_vals


def test_consensus_labels_falls_back_when_no_consensus():
    # No plausible-slope line gathers >=3 inliers -> fall back to the old
    # monotonic path rather than returning nothing.
    labels = [(100, 50, 0.5, 'a'), (200, 90, 0.5, 'b'), (300, 51, 0.5, 'c')]
    kept = consensus_labels(labels)
    assert kept is not None and len(kept) >= 1


def test_affine_requires_two_labels_per_axis():
    result = _synthetic_label_result()
    result['northing_labels'] = result['northing_labels'][:1]
    assert labels_to_affine(result) is None


# ---------------------------------------------------------------------------
# coord_converter + TFWX round trip
# ---------------------------------------------------------------------------

def test_tfwx_write_load_roundtrip():
    affine_in = {'a': 0.8453, 'b': 0.0012, 'c': -0.0009, 'd': -0.8421,
                 'e': 139250.5, 'f': 1130420.25}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'test.tfwx'
        write_tfwx(affine_in, path)
        loaded = load_tfwx(path)
    for k in 'abcdef':
        assert abs(loaded[k] - affine_in[k]) < 1e-6, k

    # pixel → map → pixel round trip
    px, py = 4321.0, 8765.0
    mx, my = pixel_to_map(px, py, loaded)
    px2, py2 = map_to_pixel(mx, my, loaded)
    assert abs(px2 - px) < 1e-6 and abs(py2 - py) < 1e-6

    # batch functions agree with scalar ones
    batch_map = pixel_to_map_batch([[px, py]], loaded)
    assert abs(batch_map[0, 0] - mx) < 1e-9 and abs(batch_map[0, 1] - my) < 1e-9
    batch_px = map_to_pixel_batch([[mx, my]], loaded)
    assert abs(batch_px[0, 0] - px) < 1e-6 and abs(batch_px[0, 1] - py) < 1e-6


# ---------------------------------------------------------------------------
# Old Palestine Grid conversion (bootstrap_from_grid)
# ---------------------------------------------------------------------------

def test_labels_to_grid_points_old_grid_convention():
    # Pins down the DB convention: old_n = easting_km*1000,
    # old_e = northing_km*1000 + 1,000,000 (the historical "+1M offset bug")
    result = _synthetic_label_result()
    affine = labels_to_affine(result)
    grid_points = labels_to_grid_points(result, affine)
    assert len(grid_points) == 22  # 11 easting + 11 northing labels

    # Easting label 140 at col 891 → old_n = 140,000
    px, py, old_e, old_n = grid_points[0]
    assert px == 891 and old_n == 140000
    assert 1_100_000 < old_e < 1_140_000  # northing axis carries the +1M prefix

    # Northing label 130 at row 934 → old_e = 1,130,000 exactly
    px, py, old_e, old_n = grid_points[11]
    assert py == 934 and old_e == 1_130_000
    assert 130_000 < old_n < 160_000  # estimated from affine, no 1M prefix

    # Self-consistency: every grid point must satisfy the label affine
    for px, py, old_e, old_n in grid_points:
        assert abs((affine['a'] * px + affine['e']) - old_n) < 100
        assert abs((affine['d'] * py + affine['f'] + 1_000_000) - old_e) < 100


def test_grid_points_reject_junk_label():
    # Regression (M7_4138): labels_to_affine rejects a km-scale misread, but
    # labels_to_grid_points/labels_to_old_grid_extent used to re-read the RAW
    # ocr labels and drag the junk back in — a single 70-vs-160 km misread
    # threw the old-grid affine off by ~31 km (3 m/px instead of ~0.85).
    px_per_km = 1183.0
    result = _synthetic_label_result(px_per_km)
    n_clean = len(result['easting_labels']) + len(result['northing_labels'])
    # Inject a bottom-edge easting misread (decade error: 145 -> 245 km)
    result['easting_labels'].append((891 + 5 * px_per_km, 245, 0.3, 'bottom'))

    affine = labels_to_affine(result)
    assert affine is not None
    grid_points = labels_to_grid_points(result, affine)
    assert len(grid_points) == n_clean        # the misread was dropped

    # The old-grid affine must keep the true pixel size, not blow up.
    M = build_old_grid_affine(grid_points)['M']
    sv = np.linalg.svd(M, compute_uv=False)
    expected = 1000.0 / px_per_km
    assert abs(sv[0] - expected) < 0.05 and abs(sv[1] - expected) < 0.05, sv

    # Extent must not be inflated by the junk (245 km would balloon old_n).
    _, old_n_range = labels_to_old_grid_extent(result, affine)
    assert old_n_range[1] < 152_000, old_n_range


def test_parse_sheet_number():
    # Normal: 14-15-Lydda-1942 → eastings 140-150 km, northings 150-160 km
    old_e_range, old_n_range = parse_sheet_number('14-15-Lydda-1942.jpg')
    assert old_n_range == (140_000, 150_000)
    assert old_e_range == (1_150_000, 1_160_000)

    # Combined column: 1415-24-Haifa-1942 spans two sheet columns
    old_e_range, old_n_range = parse_sheet_number('1415-24-Haifa-1942.jpg')
    assert old_n_range == (140_000, 160_000)
    assert old_e_range == (1_240_000, 1_250_000)

    # Combined row: 10-1112-Foo
    old_e_range, old_n_range = parse_sheet_number('10-1112-Foo-1948.jpg')
    assert old_n_range == (100_000, 110_000)
    assert old_e_range == (1_110_000, 1_130_000)

    # Unparseable (Control_Maps style name)
    assert parse_sheet_number('M5_4598.jpg') is None


def test_sheet_label_ranges():
    # Sheet 14-15: easting labels can only be 140-150 km (±1 slack),
    # northing labels 150-160 km. A "19X" decade misread falls outside.
    e_range, n_range = sheet_label_ranges('14-15-Lydda-1942.jpg')
    assert e_range == (139, 151)
    assert n_range == (149, 161)
    assert not (e_range[0] <= 195 <= e_range[1])  # misread decade rejected

    assert sheet_label_ranges('M5_4598.jpg') is None


def test_labels_to_old_grid_extent():
    result = _synthetic_label_result()  # eastings 140-150, northings 120-130
    old_e_range, old_n_range = labels_to_old_grid_extent(result)
    assert old_n_range == (139000, 151000)            # 140-150 km ± 1 km
    assert old_e_range == (1_119_000, 1_131_000)      # 120-130 km + 1M ± 1 km


# ---------------------------------------------------------------------------
# Affine fitting (bootstrap_from_grid, auto_georeference)
# ---------------------------------------------------------------------------

def _project(points, a, b, c, d, e, f):
    return [(a * px + c * py + e, b * px + d * py + f) for px, py in points]


def test_build_old_grid_affine_exact_recovery():
    true = dict(a=0.85, b=0.002, c=-0.001, d=-0.84, e=140000.0, f=1130000.0)
    pixels = [(891, 934), (12721, 934), (891, 12804), (12721, 12804), (6800, 6900)]
    maps = _project(pixels, **true)
    grid_points = [(px, py, mx, my) for (px, py), (mx, my) in zip(pixels, maps)]
    affine = build_old_grid_affine(grid_points)
    for k, v in true.items():
        assert abs(affine[k] - v) < 1e-6, k


def test_compute_affine_transform_exact_recovery():
    true = dict(a=0.85, b=0.002, c=-0.001, d=-0.84, e=189000.0, f=630000.0)
    pixels = [(100, 200), (9000, 150), (300, 11000), (8000, 9000)]
    maps = _project(pixels, **true)
    affine = compute_affine_transform(np.array(pixels), np.array(maps))
    for k, v in true.items():
        assert abs(affine[k] - v) < 1e-6, k
    assert affine['rmse_meters'] < 1e-6


# ---------------------------------------------------------------------------
# Slow end-to-end integration check (opt-in: --integration)
# ---------------------------------------------------------------------------

def integration_ocr_real_map():
    """OCR a real scan end-to-end and verify the affine (~40s, needs easyocr
    and the ErRamle scan in Maps_from_wiki). Not run by default or by pytest.

    This map is the worst case we know: ALL easting labels (131-139) are
    misread as 18X with conf 1.0 — only the sheet-number constraint plus
    digit-confusion rescue recovers the true values.
    """
    from data_paths import WIKI_MAPS
    map_path = WIKI_MAPS / '13-14-ErRamle-1948.jpg'
    if not map_path.exists():
        print(f"  SKIP  integration_ocr_real_map (missing {map_path.name})")
        return True

    from image_loader import load_image
    from grid_label_ocr import read_grid_labels
    from bootstrap_from_grid import sheet_label_ranges

    e_range, n_range = sheet_label_ranges(map_path.name)
    img = load_image(str(map_path))
    result = read_grid_labels(img, expected_easting_range=e_range,
                              expected_northing_range=n_range)
    vals_e = [v for _, v, _, _ in result['easting_labels']]
    vals_n = [v for _, v, _, _ in result['northing_labels']]
    assert len(vals_e) >= 4, f"too few easting labels: {vals_e}"
    assert len(vals_n) >= 4, f"too few northing labels: {vals_n}"
    assert all(129 <= v <= 141 for v in vals_e), f"eastings off-sheet: {vals_e}"
    assert all(139 <= v <= 151 for v in vals_n), f"northings off-sheet: {vals_n}"

    affine = labels_to_affine(result)
    assert affine is not None, "affine rejected"
    assert abs(affine['pixel_size_x'] - affine['pixel_size_y']) < 0.05, \
        f"pixel sizes inconsistent: {affine['pixel_size_x']:.3f} vs {affine['pixel_size_y']:.3f}"
    assert affine['easting_rmse_m'] < 100 and affine['northing_rmse_m'] < 100, \
        f"RMSE too high: {affine['easting_rmse_m']:.1f}/{affine['northing_rmse_m']:.1f} m"
    print(f"  eastings {min(vals_e)}-{max(vals_e)} km, "
          f"northings {min(vals_n)}-{max(vals_n)} km, "
          f"RMSE E={affine['easting_rmse_m']:.1f}m N={affine['northing_rmse_m']:.1f}m")
    return True


# ---------------------------------------------------------------------------
# Runner (no pytest needed)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    if '--integration' in sys.argv:
        tests.append(('integration_ocr_real_map', integration_ocr_real_map))
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
