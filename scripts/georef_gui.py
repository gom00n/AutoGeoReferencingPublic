#!/usr/bin/env python3
"""
GUI for automated map georeferencing.

Provides a tkinter interface for:
- Selecting map images (JPG/TIF) or map directories
- Auto-detecting processing mode (Auto vs Bootstrap)
- Configuring parameters
- Running the georeferencing pipeline with real-time logs
- Exporting logs as TXT or JSON
"""

import json
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
from pathlib import Path
from io import StringIO

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAS_DND = True
except ImportError:
    HAS_DND = False

# Ensure scripts/ is on the path
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ---------------------------------------------------------------------------
# Logger – captures structured log entries and feeds them to the GUI
# ---------------------------------------------------------------------------

class PipelineLogger:
    """Thread-safe logger that stores structured entries and notifies a callback."""

    def __init__(self, callback=None):
        self._entries = []
        self._lock = threading.Lock()
        self._callback = callback
        self._start_time = time.time()

    def set_callback(self, cb):
        self._callback = cb

    def log(self, message, level="INFO", **extra):
        entry = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_s": round(time.time() - self._start_time, 2),
            "level": level,
            "message": message,
        }
        entry.update(extra)
        with self._lock:
            self._entries.append(entry)
        if self._callback:
            self._callback(entry)

    def info(self, msg, **kw):
        self.log(msg, "INFO", **kw)

    def warn(self, msg, **kw):
        self.log(msg, "WARN", **kw)

    def error(self, msg, **kw):
        self.log(msg, "ERROR", **kw)

    def result(self, msg, **kw):
        self.log(msg, "RESULT", **kw)

    def reset(self):
        with self._lock:
            self._entries.clear()
        self._start_time = time.time()

    @property
    def entries(self):
        with self._lock:
            return list(self._entries)

    def as_text(self):
        lines = []
        for e in self.entries:
            ts = e["timestamp"]
            el = e["elapsed_s"]
            lv = e["level"]
            msg = e["message"]
            extra = {k: v for k, v in e.items()
                     if k not in ("timestamp", "elapsed_s", "level", "message")}
            extra_str = f"  {extra}" if extra else ""
            lines.append(f"[{ts}] [{el:>7.2f}s] [{lv:>6s}] {msg}{extra_str}")
        return "\n".join(lines)

    def as_json(self):
        return json.dumps(self.entries, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Processing wrapper – runs the pipeline in a background thread
# ---------------------------------------------------------------------------

def _detect_mode(map_dir):
    """Return 'auto' if the directory has a .tfwx, else 'bootstrap'."""
    map_dir = Path(map_dir)
    if list(map_dir.glob("*.tfwx")):
        return "auto"
    return "bootstrap"


def _find_image(map_dir):
    """Find the map image inside a directory. Prefers JPG, falls back to TIF."""
    map_dir = Path(map_dir)
    for ext in ("*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        files = list(map_dir.glob(ext))
        if files:
            return files[0]
    return None


def _resolve_inputs(paths):
    """
    Given a list of user-selected paths (files or dirs), return a list of
    (map_dir, image_path) tuples ready for processing.

    Rules:
    - If a path is a directory containing a map image, use it directly.
    - If a path is an image file, use its parent as the map_dir.
    - If a path is a directory containing sub-directories with images,
      treat each sub-directory as a map.
    """
    results = []
    seen = set()
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff"):
            map_dir = p.parent
            if map_dir not in seen:
                seen.add(map_dir)
                results.append((map_dir, p))
        elif p.is_dir():
            img = _find_image(p)
            if img:
                if p not in seen:
                    seen.add(p)
                    results.append((p, img))
            else:
                # Check sub-directories
                for child in sorted(p.iterdir()):
                    if child.is_dir():
                        img = _find_image(child)
                        if img and child not in seen:
                            seen.add(child)
                            results.append((child, img))
    return results


def run_pipeline(map_items, params, logger, progress_cb=None, cancel_event=None):
    """
    Run the georeferencing pipeline on a list of (map_dir, image_path) tuples.

    Args:
        map_items: list of (map_dir, image_path)
        params: dict of user-configurable parameters
        logger: PipelineLogger instance
        progress_cb: callable(current_index, total) for progress updates
        cancel_event: threading.Event to signal cancellation
    """
    # Lazy imports so the GUI itself starts instantly
    import numpy as np
    from db_matcher import load_geodetic_db, load_grayscale_templates, verify_candidates
    from coord_converter import load_tfwx, get_map_extent, load_control_points
    from image_loader import load_image
    from auto_georeference import (
        select_best_points, compute_affine_transform,
        ransac_affine, evaluate_transform, write_tfwx,
    )

    db_path = Path(params.get("db_path", BASE_DIR / "Control_Points" / "nikudot_bakara_slim.csv"))
    template_dir = Path(params.get("template_dir", SCRIPT_DIR / "templates"))
    output_dir_raw = params.get("output_dir", "").strip()
    # Empty string → save outputs next to the input image file
    output_root = Path(output_dir_raw) if output_dir_raw else None
    if output_root:
        output_root.mkdir(parents=True, exist_ok=True)

    min_conf = params.get("min_conf", 0.5)
    n_points = params.get("n_points", 50)
    color_mode = params.get("color_mode", "multi")

    total = len(map_items)
    logger.info(f"Starting pipeline for {total} map(s)")

    # Load geodetic DB
    t0 = time.time()
    logger.info("Loading geodetic database...", path=str(db_path))
    geo_db = load_geodetic_db(db_path)
    logger.info(f"Geodetic DB loaded: {len(geo_db)} points", duration_s=round(time.time() - t0, 2))

    # Load templates
    t0 = time.time()
    templates = load_grayscale_templates(template_dir)
    logger.info(f"Loaded {len(templates)} templates", duration_s=round(time.time() - t0, 2))

    summary_results = []

    for idx, (map_dir, image_path) in enumerate(map_items):
        if cancel_event and cancel_event.is_set():
            logger.warn("Pipeline cancelled by user")
            break

        map_name = Path(image_path).stem   # always use image filename
        mode = _detect_mode(map_dir)
        logger.info(f"[{idx+1}/{total}] Processing {map_name}", mode=mode)

        if progress_cb:
            progress_cb(idx, total)

        t_map = time.time()

        try:
            if mode == "auto":
                result = _process_auto(
                    map_dir, image_path, geo_db, templates,
                    min_conf, n_points, color_mode, output_root, logger,
                )
            else:
                result = _process_bootstrap(
                    map_dir, image_path, geo_db, templates,
                    min_conf, n_points, color_mode, output_root, logger,
                )
        except Exception as exc:
            logger.error(f"Failed to process {map_name}: {exc}")
            result = None

        elapsed_map = round(time.time() - t_map, 2)

        if result:
            succeeded = bool(result.get("output_tfwx")) and result.get("n_inliers", 0) >= 3
            if succeeded:
                summary_results.append(result)
            logger.result(
                f"Finished {map_name}",
                duration_s=elapsed_map,
                n_candidates=result.get("n_candidates", 0),
                n_detections=result.get("n_detections", 0),
                n_selected=result.get("n_selected", 0),
                n_good=result.get("n_good", 0),
                n_inliers=result.get("n_inliers", 0),
                fit_rmse_m=result.get("fit_rmse_m"),
                eval_rmse_m=result.get("eval_rmse_m"),
                output_tfwx=result.get("output_tfwx"),
                status="OK" if succeeded else "FAILED",
            )
        else:
            logger.warn(f"No result for {map_name}", duration_s=elapsed_map)

    if progress_cb:
        progress_cb(total, total)

    logger.info(f"Pipeline complete: {len(summary_results)}/{total} maps succeeded")
    return summary_results


def _process_auto(map_dir, image_path, geo_db, templates,
                  min_conf, n_points, color_mode, output_root, logger):
    """Process a map that already has a TFWX (auto-georeferencing mode)."""
    import numpy as np
    from PIL import Image as PILImage
    from db_matcher import (
        filter_points_to_extent, verify_candidates,
    )
    from coord_converter import load_tfwx, get_map_extent, load_control_points
    from image_loader import load_image
    from auto_georeference import (
        select_best_points, compute_affine_transform,
        ransac_affine, evaluate_transform, write_tfwx,
    )
    from db_matcher import save_results_csv, visualize_results

    map_dir = Path(map_dir)
    image_path = Path(image_path)
    map_name = image_path.stem          # use image filename, not dir name

    # Load reference TFWX
    tfwx_files = list(map_dir.glob("*.tfwx"))
    if not tfwx_files:
        logger.error(f"  No TFWX found in {map_dir}")
        return None
    ref_affine = load_tfwx(tfwx_files[0])

    # Image dimensions
    PILImage.MAX_IMAGE_PIXELS = None
    t0 = time.time()
    with PILImage.open(image_path) as pil_img:
        w, h = pil_img.size
    logger.info(f"  Image dimensions: {w}x{h}", duration_s=round(time.time() - t0, 2))

    # Filter DB
    extent = get_map_extent(ref_affine, w, h)
    candidates = filter_points_to_extent(geo_db, extent)
    logger.info(f"  DB candidates in extent: {len(candidates)}")

    if not candidates:
        return None

    # Load image & detect
    t0 = time.time()
    logger.info("  Loading image...")
    img = load_image(str(image_path))
    logger.info(f"  Image loaded", duration_s=round(time.time() - t0, 2))

    t0 = time.time()
    logger.info(f"  Running template matching on {len(candidates)} candidates (color_mode={color_mode})...")
    detections = verify_candidates(img, candidates, ref_affine, templates, color_mode=color_mode)
    logger.info(f"  Template matching done: {len(detections)} detections", duration_s=round(time.time() - t0, 2))

    # Confidence breakdown
    high = sum(1 for d in detections if d.confidence >= 0.7)
    med = sum(1 for d in detections if 0.5 <= d.confidence < 0.7)
    low = sum(1 for d in detections if d.confidence < 0.5)
    logger.info(f"  Confidence breakdown: {high} high, {med} medium, {low} low")

    # Select best
    selected = select_best_points(detections, n_points=n_points, min_conf=min_conf)
    logger.info(f"  Selected {len(selected)} points (min_conf={min_conf}, n_points={n_points})")

    if len(selected) < 3:
        logger.warn(f"  Only {len(selected)} points — not enough for transform")
        return {"map_name": map_name, "n_candidates": len(candidates),
                "n_detections": len(detections), "n_selected": len(selected),
                "n_good": high + med, "n_inliers": 0}

    # Compute affine
    pixel_pts = np.array([(d.pixel_x, d.pixel_y) for d in selected])
    map_pts = np.array([(d.geo_point.easting_6991, d.geo_point.northing_6991) for d in selected])

    t0 = time.time()
    affine = compute_affine_transform(pixel_pts, map_pts)
    logger.info(f"  Affine fit RMSE: {affine['rmse_meters']:.2f} m", duration_s=round(time.time() - t0, 2))

    # RANSAC
    ransac_result, inlier_mask = ransac_affine(pixel_pts, map_pts)
    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    ransac_rmse = ransac_result['rmse_meters'] if ransac_result else None
    logger.info(f"  RANSAC: {n_inliers}/{len(selected)} inliers"
                + (f", RMSE={ransac_rmse:.2f} m" if ransac_rmse else ""))

    # Use the better transform
    best_transform = ransac_result if (ransac_result and ransac_rmse < affine['rmse_meters']) else affine
    best_rmse = best_transform['rmse_meters']

    # Evaluate against control points if available
    eval_rmse = None
    cp_files = list(map_dir.glob("*controlpoints.txt"))
    if cp_files:
        eval_result = evaluate_transform(best_transform, tfwx_files[0], cp_files[0], (w, h))
        eval_rmse = eval_result['rmse_m']
        logger.info(f"  Evaluation vs control points: RMSE={eval_rmse:.2f} m, "
                    f"max={eval_result['max_error_m']:.2f} m, "
                    f"n_checkpoints={eval_result['n_checkpoints']}")

    # Save outputs — next to input image if no output_root specified
    if output_root is None:
        map_output = image_path.parent
    else:
        map_output = output_root / map_name
        map_output.mkdir(parents=True, exist_ok=True)

    tfwx_out = map_output / f"{map_name}.tfwx"
    write_tfwx(best_transform, tfwx_out)
    logger.info(f"  TFWX written: {tfwx_out}")

    csv_out = map_output / f"{map_name}_detections.csv"
    save_results_csv(detections, csv_out)

    vis_out = map_output / f"{map_name}_detected.png"
    visualize_results(img, detections, vis_out, conf_threshold=0.4)
    logger.info(f"  Visualization: {vis_out}")

    return {
        "map_name": map_name,
        "mode": "auto",
        "n_candidates": len(candidates),
        "n_detections": len(detections),
        "n_selected": len(selected),
        "n_good": high + med,
        "n_inliers": n_inliers,
        "fit_rmse_m": round(best_rmse, 2),
        "eval_rmse_m": round(eval_rmse, 2) if eval_rmse else None,
        "output_tfwx": str(tfwx_out),
    }


def _process_bootstrap(map_dir, image_path, geo_db, templates,
                       min_conf, n_points, color_mode, output_root, logger):
    """Process a map without TFWX (bootstrap georeferencing)."""
    import numpy as np
    from image_loader import load_image
    from grid_label_ocr import read_grid_labels, labels_to_affine, find_neatline
    from bootstrap_from_grid import (
        bootstrap_georeference, build_old_grid_affine,
        filter_points_by_old_grid, verify_candidates_old_grid, write_tfwx,
        labels_to_grid_points, labels_to_old_grid_extent,
        parse_sheet_number, sheet_label_ranges,
    )
    from auto_georeference import select_best_points, compute_affine_transform
    from coord_converter import pixel_to_map

    map_dir = Path(map_dir)
    image_path = Path(image_path)
    map_name = image_path.stem          # use image filename, not dir name

    # Output dir — next to input image if no output_root specified
    if output_root is None:
        map_output = image_path.parent
    else:
        map_output = output_root / map_name
        map_output.mkdir(parents=True, exist_ok=True)

    # Step 1: Load image and OCR grid labels
    t0 = time.time()
    logger.info("  Loading image...")
    img = load_image(str(image_path))
    h_img, w_img = img.shape[:2]
    logger.info(f"  Image loaded: {w_img}x{h_img}", duration_s=round(time.time() - t0, 2))

    t0 = time.time()
    logger.info("  Running OCR on grid labels...")
    # Sheet-number cross-check: constrain OCR to the sheet's km range so a
    # systematic decade misread can't produce a consistent-but-wrong affine
    sheet_ranges = sheet_label_ranges(image_path.name)
    if sheet_ranges:
        sheet_e_range, sheet_n_range = sheet_ranges
        logger.info(f"  Sheet number: eastings {sheet_e_range[0]}-{sheet_e_range[1]} km, "
                    f"northings {sheet_n_range[0]}-{sheet_n_range[1]} km")
        ocr_result = read_grid_labels(img, expected_easting_range=sheet_e_range,
                                      expected_northing_range=sheet_n_range)
    else:
        ocr_result = read_grid_labels(img)
    e_labels = ocr_result['easting_labels']
    n_labels = ocr_result['northing_labels']
    logger.info(f"  OCR result: {len(e_labels)} easting labels, {len(n_labels)} northing labels",
                duration_s=round(time.time() - t0, 2))

    # ---------- Determine affine source: OCR labels or sheet-number fallback ----------
    ocr_ok = len(e_labels) >= 2 and len(n_labels) >= 2
    label_affine = None
    if ocr_ok:
        label_affine = labels_to_affine(ocr_result)
        # Sanity-check: pixel sizes should be roughly equal and ~0.5-5 m/px
        if label_affine is not None:
            psx, psy = label_affine['pixel_size_x'], label_affine['pixel_size_y']
            ratio = max(psx, psy) / max(min(psx, psy), 1e-9)
            if ratio > 3.0 or psx < 0.3 or psx > 10.0 or psy < 0.3 or psy > 10.0:
                logger.warn(f"  OCR affine looks bad (pixel_size={psx:.4f}x{psy:.4f} m/px, ratio={ratio:.1f}) — discarding")
                label_affine = None

    use_sheet_fallback = label_affine is None

    if use_sheet_fallback:
        # ---- Fallback: derive extent from sheet number in filename ----
        sheet_extents = parse_sheet_number(image_path.name)
        if sheet_extents is None:
            logger.error("  OCR failed and could not parse sheet number from filename")
            return None
        old_e_range, old_n_range = sheet_extents
        logger.info(f"  Using sheet-number fallback  "
                    f"old_e=[{old_e_range[0]/1000:.0f},{old_e_range[1]/1000:.0f}]km  "
                    f"old_n=[{old_n_range[0]/1000:.0f},{old_n_range[1]/1000:.0f}]km")

        # Build a rough affine from neatline corners + sheet extent
        neatline = ocr_result.get('neatline') or find_neatline(img)
        grid_points = [
            (neatline['left'],  neatline['top'],    old_e_range[1], old_n_range[0]),
            (neatline['right'], neatline['top'],    old_e_range[1], old_n_range[1]),
            (neatline['left'],  neatline['bottom'], old_e_range[0], old_n_range[0]),
            (neatline['right'], neatline['bottom'], old_e_range[0], old_n_range[1]),
        ]
        logger.info(f"  Grid points from neatline corners: {len(grid_points)}")
    else:
        # ---- OCR path ----
        logger.info(f"  Old Grid affine: pixel_size={label_affine['pixel_size_x']:.4f}x{label_affine['pixel_size_y']:.4f} m/px")

        grid_points = labels_to_grid_points(ocr_result, label_affine)
        logger.info(f"  Grid points for bootstrap: {len(grid_points)}")

        # Determine extent from labels
        old_e_range, old_n_range = labels_to_old_grid_extent(ocr_result, label_affine)

    logger.info(f"  Old Grid extent: E=[{old_e_range[0]/1000:.0f},{old_e_range[1]/1000:.0f}]km "
                f"N=[{old_n_range[0]/1000:.0f},{old_n_range[1]/1000:.0f}]km")

    # Step 3: Build old grid affine and filter DB
    t0 = time.time()
    old_affine = build_old_grid_affine(grid_points)
    candidates = filter_points_by_old_grid(geo_db, old_e_range[0], old_e_range[1],
                                           old_n_range[0], old_n_range[1])
    logger.info(f"  DB candidates in Old Grid extent: {len(candidates)}", duration_s=round(time.time() - t0, 2))

    if not candidates:
        logger.error("  No DB candidates in extent")
        return None

    # Step 4: Template matching
    t0 = time.time()
    logger.info(f"  Running template matching on {len(candidates)} candidates...")
    detections = verify_candidates_old_grid(img, candidates, old_affine, templates,
                                            color_mode=color_mode)
    logger.info(f"  Detections: {len(detections)}", duration_s=round(time.time() - t0, 2))

    high = sum(1 for d in detections if d.confidence >= 0.7)
    med = sum(1 for d in detections if 0.5 <= d.confidence < 0.7)
    low = sum(1 for d in detections if d.confidence < 0.5)
    logger.info(f"  Confidence breakdown: {high} high, {med} medium, {low} low")

    # Step 5: Select best points (use higher threshold for bootstrap)
    bootstrap_conf = max(min_conf, 0.75)
    selected = select_best_points(detections, n_points=n_points, min_conf=bootstrap_conf)
    logger.info(f"  Selected {len(selected)} points (min_conf={bootstrap_conf})")

    if len(selected) < 3:
        logger.warn(f"  Only {len(selected)} points — not enough")
        return {"map_name": map_name, "n_candidates": len(candidates),
                "n_detections": len(detections), "n_selected": len(selected),
                "n_good": high + med, "n_inliers": 0}

    # Step 6: Compute EPSG:6991 affine
    pixel_pts = np.array([(d.pixel_x, d.pixel_y) for d in selected])
    map_pts = np.array([(d.geo_point.easting_6991, d.geo_point.northing_6991) for d in selected])

    affine_6991 = compute_affine_transform(pixel_pts, map_pts)
    fit_rmse = affine_6991['rmse_meters']
    logger.info(f"  EPSG:6991 affine fit RMSE: {fit_rmse:.2f} m")

    # Step 7: RANSAC outlier rejection
    predicted = np.array([pixel_to_map(px, py, affine_6991) for px, py in pixel_pts])
    errors = np.sqrt(np.sum((predicted - map_pts)**2, axis=1))
    threshold = max(15.0, fit_rmse * 2)
    inlier_mask = errors < threshold
    n_inliers = int(np.sum(inlier_mask))
    logger.info(f"  RANSAC: {n_inliers}/{len(selected)} inliers (threshold={threshold:.1f} m)")

    if n_inliers >= 3 and n_inliers < len(selected):
        affine_6991 = compute_affine_transform(pixel_pts[inlier_mask], map_pts[inlier_mask])
        fit_rmse = affine_6991['rmse_meters']
        logger.info(f"  After outlier rejection: RMSE={fit_rmse:.2f} m")

    # Step 8: Write TFWX
    tfwx_out = map_output / f"{map_name}.tfwx"
    write_tfwx(affine_6991, tfwx_out)
    logger.info(f"  TFWX written: {tfwx_out}")

    # Visualization
    import cv2
    vis_path = map_output / f"{map_name}_bootstrap.jpg"
    vis = cv2.resize(img, (img.shape[1] // 8, img.shape[0] // 8))
    scale = 1.0 / 8
    for det in selected:
        x, y = int(det.pixel_x * scale), int(det.pixel_y * scale)
        color = (0, 255, 0) if det.confidence >= 0.6 else (0, 255, 255)
        cv2.circle(vis, (x, y), 6, color, 2)
    cv2.imwrite(str(vis_path), vis)
    logger.info(f"  Visualization: {vis_path}")

    return {
        "map_name": map_name,
        "mode": "bootstrap",
        "n_candidates": len(candidates),
        "n_detections": len(detections),
        "n_selected": len(selected),
        "n_good": high + med,
        "n_inliers": n_inliers,
        "fit_rmse_m": round(fit_rmse, 2),
        "eval_rmse_m": None,
        "output_tfwx": str(tfwx_out),
    }


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class GeorefGUI:
    """Main application window."""

    def __init__(self):
        if HAS_DND:
            self.root = TkinterDnD.Tk()
        else:
            self.root = tk.Tk()
        self.root.title("Map Georeferencer")
        self.root.geometry("960x720")
        self.root.minsize(800, 600)

        self.logger = PipelineLogger()
        self.cancel_event = threading.Event()
        self._worker_thread = None
        self._selected_paths = []

        self._build_ui()
        self.logger.set_callback(self._on_log_entry)

    # ---- UI construction ----

    def _build_ui(self):
        # Use a style for consistent look
        style = ttk.Style()
        style.theme_use("default")

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        # === Top: file selection ===
        file_frame = ttk.LabelFrame(main, text="Input Maps", padding=6)
        file_frame.pack(fill=tk.X, pady=(0, 6))

        btn_row = ttk.Frame(file_frame)
        btn_row.pack(fill=tk.X)

        ttk.Button(btn_row, text="Add Files…", command=self._add_files).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row, text="Add Folder…", command=self._add_folder).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row, text="Clear", command=self._clear_files).pack(side=tk.LEFT, padx=(0, 4))

        dnd_hint = " (drag & drop supported)" if HAS_DND else " (install tkinterdnd2 for drag & drop)"
        ttk.Label(btn_row, text=dnd_hint, foreground="#888888").pack(side=tk.LEFT, padx=(8, 0))

        self.file_listvar = tk.StringVar()
        self.file_listbox = tk.Listbox(file_frame, listvariable=self.file_listvar,
                                       height=4, selectmode=tk.EXTENDED,
                                       font=("Menlo", 11))
        self.file_listbox.pack(fill=tk.X, pady=(4, 0))

        if HAS_DND:
            self.file_listbox.drop_target_register(DND_FILES)
            self.file_listbox.dnd_bind('<<Drop>>', self._on_drop)

        # === Middle: parameters ===
        param_frame = ttk.LabelFrame(main, text="Parameters", padding=6)
        param_frame.pack(fill=tk.X, pady=(0, 6))

        grid = ttk.Frame(param_frame)
        grid.pack(fill=tk.X)

        # Min confidence
        ttk.Label(grid, text="Min Confidence:").grid(row=0, column=0, sticky=tk.W, padx=(0, 4))
        self.var_min_conf = tk.DoubleVar(value=0.5)
        ttk.Spinbox(grid, from_=0.1, to=0.95, increment=0.05,
                    textvariable=self.var_min_conf, width=6).grid(row=0, column=1, sticky=tk.W)

        # N points
        ttk.Label(grid, text="Max Points:").grid(row=0, column=2, sticky=tk.W, padx=(16, 4))
        self.var_n_points = tk.IntVar(value=50)
        ttk.Spinbox(grid, from_=5, to=200, increment=5,
                    textvariable=self.var_n_points, width=6).grid(row=0, column=3, sticky=tk.W)

        # Color mode
        ttk.Label(grid, text="Color Mode:").grid(row=0, column=4, sticky=tk.W, padx=(16, 4))
        self.var_color_mode = tk.StringVar(value="multi")
        color_combo = ttk.Combobox(grid, textvariable=self.var_color_mode, width=14, state="readonly",
                                   values=["multi", "suppress_red", "suppress_colors", "black_white"])
        color_combo.grid(row=0, column=5, sticky=tk.W)

        # DB path
        ttk.Label(grid, text="Geodetic DB:").grid(row=1, column=0, sticky=tk.W, padx=(0, 4), pady=(4, 0))
        self.var_db_path = tk.StringVar(
            value=str(BASE_DIR / "Control_Points" / "nikudot_bakara_slim.csv"))
        db_entry = ttk.Entry(grid, textvariable=self.var_db_path, width=50)
        db_entry.grid(row=1, column=1, columnspan=4, sticky=tk.EW, pady=(4, 0))
        ttk.Button(grid, text="…", width=3,
                   command=self._browse_db).grid(row=1, column=5, pady=(4, 0))

        # Output dir
        ttk.Label(grid, text="Output Dir\n(blank=input):").grid(row=2, column=0, sticky=tk.W, padx=(0, 4), pady=(4, 0))
        self.var_output_dir = tk.StringVar(value="")   # empty = save next to each input file
        out_entry = ttk.Entry(grid, textvariable=self.var_output_dir, width=50)
        out_entry.grid(row=2, column=1, columnspan=4, sticky=tk.EW, pady=(4, 0))
        ttk.Button(grid, text="…", width=3,
                   command=self._browse_output).grid(row=2, column=5, pady=(4, 0))

        grid.columnconfigure(1, weight=1)

        # === Controls row ===
        ctrl_frame = ttk.Frame(main)
        ctrl_frame.pack(fill=tk.X, pady=(0, 6))

        self.btn_run = ttk.Button(ctrl_frame, text="▶  Run", command=self._run)
        self.btn_run.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_cancel = ttk.Button(ctrl_frame, text="■  Cancel", command=self._cancel, state=tk.DISABLED)
        self.btn_cancel.pack(side=tk.LEFT, padx=(0, 4))

        # Progress bar
        self.progress = ttk.Progressbar(ctrl_frame, mode="determinate", length=200)
        self.progress.pack(side=tk.LEFT, padx=(8, 4), fill=tk.X, expand=True)

        self.var_status = tk.StringVar(value="Ready")
        ttk.Label(ctrl_frame, textvariable=self.var_status).pack(side=tk.LEFT, padx=(4, 0))

        # === Log area ===
        log_frame = ttk.LabelFrame(main, text="Logs", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True)

        # Log toolbar
        log_toolbar = ttk.Frame(log_frame)
        log_toolbar.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(log_toolbar, text="Export TXT…", command=self._export_txt).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(log_toolbar, text="Export JSON…", command=self._export_json).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(log_toolbar, text="Clear Logs", command=self._clear_logs).pack(side=tk.LEFT, padx=(0, 4))

        # Scrollable text area
        log_container = ttk.Frame(log_frame)
        log_container.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_container, wrap=tk.WORD, font=("Menlo", 11),
                                state=tk.DISABLED, background="#1e1e1e",
                                foreground="#d4d4d4", insertbackground="#d4d4d4",
                                selectbackground="#264f78")
        scrollbar = ttk.Scrollbar(log_container, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Tag colors for log levels
        self.log_text.tag_configure("INFO", foreground="#d4d4d4")
        self.log_text.tag_configure("WARN", foreground="#e5c07b")
        self.log_text.tag_configure("ERROR", foreground="#e06c75")
        self.log_text.tag_configure("RESULT", foreground="#98c379")
        self.log_text.tag_configure("timestamp", foreground="#6b7280")

    # ---- File management callbacks ----

    def _add_files(self):
        files = filedialog.askopenfilenames(
            title="Select Map Images",
            filetypes=[
                ("Map images", "*.jpg *.jpeg *.tif *.tiff"),
                ("JPEG", "*.jpg *.jpeg"),
                ("TIFF", "*.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        for f in files:
            if f not in self._selected_paths:
                self._selected_paths.append(f)
        self._update_file_list()

    def _add_folder(self):
        folder = filedialog.askdirectory(title="Select Map Folder")
        if folder and folder not in self._selected_paths:
            self._selected_paths.append(folder)
            self._update_file_list()

    def _clear_files(self):
        self._selected_paths.clear()
        self._update_file_list()

    def _on_drop(self, event):
        """Handle files dropped onto the file list (requires tkinterdnd2)."""
        # tk.splitlist correctly handles Tcl-formatted lists: paths with spaces
        # are wrapped in braces, e.g. {/path/with spaces/file.jpg}
        try:
            dropped = self.root.tk.splitlist(event.data)
        except Exception:
            dropped = event.data.split()
        for p in dropped:
            p = p.strip()
            if p and p not in self._selected_paths:
                self._selected_paths.append(p)
        self._update_file_list()

    def _update_file_list(self):
        display = []
        for p in self._selected_paths:
            p = Path(p)
            if p.is_file():
                display.append(f"  [file] {p.parent.name}/{p.name}")
            else:
                display.append(f"  [dir]  {p.name}/")
        self.file_listbox.delete(0, tk.END)
        for item in display:
            self.file_listbox.insert(tk.END, item)

    def _browse_db(self):
        f = filedialog.askopenfilename(
            title="Select Geodetic DB CSV",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if f:
            self.var_db_path.set(f)

    def _browse_output(self):
        d = filedialog.askdirectory(title="Select Output Directory")
        if d:
            self.var_output_dir.set(d)

    # ---- Run / Cancel ----

    def _run(self):
        if not self._selected_paths:
            messagebox.showwarning("No Input", "Please add at least one map image or folder.")
            return

        map_items = _resolve_inputs(self._selected_paths)
        if not map_items:
            messagebox.showwarning("No Maps Found",
                                   "No valid map images found in the selected paths.")
            return

        # Gather params
        params = {
            "db_path": self.var_db_path.get(),
            "template_dir": str(SCRIPT_DIR / "templates"),
            "output_dir": self.var_output_dir.get(),
            "min_conf": self.var_min_conf.get(),
            "n_points": self.var_n_points.get(),
            "color_mode": self.var_color_mode.get(),
        }

        # Reset state
        self.logger.reset()
        self._clear_logs()
        self.cancel_event.clear()
        self.progress["value"] = 0
        self.progress["maximum"] = len(map_items)

        self.btn_run.configure(state=tk.DISABLED)
        self.btn_cancel.configure(state=tk.NORMAL)
        self.var_status.set(f"Processing 0/{len(map_items)}…")

        self.logger.info(f"Resolved {len(map_items)} map(s) from selection")
        for md, ip in map_items:
            mode = _detect_mode(md)
            self.logger.info(f"  {md.name}: {ip.name} [{mode}]")

        def worker():
            try:
                run_pipeline(
                    map_items, params, self.logger,
                    progress_cb=self._on_progress,
                    cancel_event=self.cancel_event,
                )
            except Exception as exc:
                self.logger.error(f"Pipeline error: {exc}")
            finally:
                self.root.after(0, self._on_done)

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _cancel(self):
        self.cancel_event.set()
        self.var_status.set("Cancelling…")

    def _on_done(self):
        self.btn_run.configure(state=tk.NORMAL)
        self.btn_cancel.configure(state=tk.DISABLED)
        if self.cancel_event.is_set():
            self.var_status.set("Cancelled")
        else:
            self.var_status.set("Done")

    def _on_progress(self, current, total):
        self.root.after(0, lambda: self._update_progress(current, total))

    def _update_progress(self, current, total):
        self.progress["value"] = current
        self.progress["maximum"] = total
        self.var_status.set(f"Processing {current}/{total}…")

    # ---- Log display ----

    def _on_log_entry(self, entry):
        """Called from worker thread — schedule GUI update on main thread."""
        self.root.after(0, lambda e=entry: self._append_log(e))

    def _append_log(self, entry):
        self.log_text.configure(state=tk.NORMAL)

        ts = entry["timestamp"].split("T")[1]
        elapsed = entry["elapsed_s"]
        level = entry["level"]
        msg = entry["message"]
        extra = {k: v for k, v in entry.items()
                 if k not in ("timestamp", "elapsed_s", "level", "message")}

        prefix = f"[{ts}] [{elapsed:>7.2f}s] "
        self.log_text.insert(tk.END, prefix, "timestamp")
        self.log_text.insert(tk.END, f"[{level:>6s}] ", level)
        self.log_text.insert(tk.END, msg, level)
        if extra:
            extra_str = "  " + " | ".join(f"{k}={v}" for k, v in extra.items())
            self.log_text.insert(tk.END, extra_str, "timestamp")
        self.log_text.insert(tk.END, "\n")

        self.log_text.configure(state=tk.DISABLED)
        self.log_text.see(tk.END)

    def _clear_logs(self):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    # ---- Export ----

    def _export_txt(self):
        text = self.logger.as_text()
        if not text.strip():
            messagebox.showinfo("Empty", "No log entries to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Export Logs as TXT",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt")],
            initialfile=f"georef_log_{datetime.now():%Y%m%d_%H%M%S}.txt",
        )
        if path:
            Path(path).write_text(text, encoding="utf-8")
            self.logger.info(f"Logs exported to {path}")

    def _export_json(self):
        entries = self.logger.entries
        if not entries:
            messagebox.showinfo("Empty", "No log entries to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Export Logs as JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile=f"georef_log_{datetime.now():%Y%m%d_%H%M%S}.json",
        )
        if path:
            Path(path).write_text(self.logger.as_json(), encoding="utf-8")
            self.logger.info(f"Logs exported to {path}")

    # ---- Main loop ----

    def run(self):
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = GeorefGUI()
    app.run()
