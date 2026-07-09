#!/usr/bin/env python3
"""
Interactive GUI for QA of triangle detections on map scans.

Browse a map → adjust thresholds → Run → zoom/pan results → click any
detection to inspect its 64×64 crop. Right-click (or press M to toggle
mark mode) to mark missed triangles for future retraining.

Controls:
    Arrow keys     Pan the map
    Scroll wheel   Zoom in/out
    +/-            Zoom in/out
    F              Fit to window
    M              Toggle mark mode (add missed triangles)
    Ctrl+Z         Undo last manual mark
    S              Save — writes missed triangles + false positive crops to disk
    Click          Inspect detection (normal mode) / add missed mark (mark mode)
    X              Mark selected detection as FALSE POSITIVE (after clicking it)

Usage:
    python qa_gui.py
    python qa_gui.py <map.jpg>
"""

import sys
import json
import time
import threading
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

try:
    from PIL import Image, ImageTk, ImageDraw
except ImportError:
    print("Pillow not installed. Run: pip install Pillow")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ── Detection ─────────────────────────────────────────────────────────────────

def run_full_detection(image_path, tmpl_threshold, log_cb):
    """
    Run template matching + CNN on a map. Returns all candidates with crops.
    CNN threshold is NOT applied here — caller filters by cnn conf.

    Returns: list of dicts with keys x, y, cnn, tmpl, crop (64x64 BGR)
    """
    import torch
    from image_loader import load_image, suppress_red
    from db_matcher import load_grayscale_templates
    from grid_label_ocr import find_neatline, find_grid_bounds
    from qa_detections import load_cnn_model, find_template_candidates

    log_cb("Loading image…")
    img = load_image(str(image_path))
    h, w = img.shape[:2]

    scale = max(1.0, 14000.0 / w)
    scale = round(scale * 2) / 2
    if scale > 1.0:
        img_up = cv2.resize(img, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_LINEAR)
        log_cb(f"Upscaled {w}×{h} → {img_up.shape[1]}×{img_up.shape[0]} ({scale}×)")
    else:
        img_up = img
        log_cb(f"Image: {w}×{h} (no upscale)")

    prep = suppress_red(img_up)

    log_cb("Finding neatline…")
    neatline = find_neatline(img)
    if neatline:
        log_cb(f"Neatline: L={neatline['left']} R={neatline['right']} "
               f"T={neatline['top']} B={neatline['bottom']}")
    else:
        log_cb("No neatline found")

    log_cb("Loading templates…")
    templates = load_grayscale_templates(SCRIPT_DIR / 'templates')
    log_cb(f"{len(templates)} templates")

    log_cb(f"Template matching (threshold={tmpl_threshold:.2f})…")
    t0 = time.time()
    candidates = find_template_candidates(prep, templates, threshold=tmpl_threshold)
    log_cb(f"Candidates: {len(candidates)} ({time.time() - t0:.1f}s)")

    log_cb("Loading CNN…")
    model = load_cnn_model()

    log_cb("Running CNN…")
    t0 = time.time()
    from qa_detections import classify_crops_chunked
    half = 32
    h_up, w_up = prep.shape[:2]
    # Classify ALL candidates — a top-N cap silently drops real triangles
    # on dense maps (Lydda: 126/127 marked misses never reached the CNN)
    cands_sorted = sorted(candidates, key=lambda c: c[2], reverse=True)

    crops_gray = []
    valid = []
    for x, y, tconf, tname in cands_sorted:
        if x - half < 0 or y - half < 0 or x + half >= w_up or y + half >= h_up:
            continue
        crops_gray.append(prep[y - half:y + half, x - half:x + half])
        valid.append((x, y, tconf))

    detections = []
    if crops_gray:
        probs = classify_crops_chunked(model, crops_gray)

        # Keep only candidates the GUI could plausibly display — storing a
        # BGR crop for every one of ~100k candidates would eat ~1 GB
        keep_floor = 0.20
        for i, (x, y, tconf) in enumerate(valid):
            cnn = float(probs[i])
            if cnn < keep_floor:
                continue
            # Convert to original-image coordinates
            x0 = int(x / scale)
            y0 = int(y / scale)
            crop_bgr = cv2.cvtColor(crops_gray[i], cv2.COLOR_GRAY2BGR)
            detections.append({'x': x0, 'y': y0, 'cnn': cnn,
                               'tmpl': tconf, 'crop': crop_bgr})

    log_cb(f"CNN done: {len(detections)} candidates ({time.time() - t0:.1f}s)")

    # Grid bounds filter (tighter than neatline), fall back to neatline
    grid_bounds = find_grid_bounds(img, neatline) if neatline else None
    if grid_bounds:
        margin = 15
        before = len(detections)
        detections = [d for d in detections if (
            grid_bounds['left'] - margin <= d['x'] <= grid_bounds['right'] + margin and
            grid_bounds['top'] - margin <= d['y'] <= grid_bounds['bottom'] + margin
        )]
        if len(detections) < before:
            log_cb(f"Grid bounds filtered: {before} → {len(detections)}")
    elif neatline:
        margin = 30
        before = len(detections)
        detections = [d for d in detections if (
            neatline['left'] - margin <= d['x'] <= neatline['right'] + margin and
            neatline['top'] - margin <= d['y'] <= neatline['bottom'] + margin
        )]
        if len(detections) < before:
            log_cb(f"Neatline filtered: {before} → {len(detections)}")

    log_cb(f"Done. {len(detections)} candidates total.")
    return detections, img, scale, grid_bounds


def annotate_image(img_bgr, detections, cnn_threshold,
                   manual_marks=None, false_positives=None, grid_bounds=None):
    """Draw circles on a copy of img_bgr for detections above threshold."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    r = max(12, int(min(w, h) / 400))
    fp_set = false_positives or set()

    # Draw grid lines as thin cyan lines
    if grid_bounds:
        grid_color = (200, 180, 0)  # cyan-ish in BGR
        for y in grid_bounds.get('h_lines', []):
            cv2.line(out, (0, y), (w, y), grid_color, 1)
        for x in grid_bounds.get('v_lines', []):
            cv2.line(out, (x, 0), (x, h), grid_color, 1)

    for i, d in enumerate(detections):
        if d['cnn'] < cnn_threshold:
            continue
        x, y, cnn = d['x'], d['y'], d['cnn']

        if i in fp_set:
            # False positive: grey circle + red X
            cv2.circle(out, (x, y), r, (100, 100, 100), 2)
            cv2.line(out, (x - r + 4, y - r + 4), (x + r - 4, y + r - 4),
                     (0, 0, 220), 2)
            cv2.line(out, (x + r - 4, y - r + 4), (x - r + 4, y + r - 4),
                     (0, 0, 220), 2)
        else:
            if cnn >= 0.85:
                color = (0, 220, 0)    # green
            elif cnn >= 0.70:
                color = (0, 190, 255)  # yellow
            else:
                color = (0, 0, 220)    # red
            cv2.circle(out, (x, y), r, color, 2)
            cv2.circle(out, (x, y), max(2, r // 5), color, -1)

    # Draw manual marks as cyan diamonds
    if manual_marks:
        for mx, my in manual_marks:
            pts = np.array([
                [mx, my - r], [mx + r, my], [mx, my + r], [mx - r, my]
            ], dtype=np.int32)
            cv2.polylines(out, [pts], True, (255, 255, 0), 2)
            cv2.circle(out, (mx, my), max(2, r // 5), (255, 255, 0), -1)

    return out


# ── App ───────────────────────────────────────────────────────────────────────

DARK_BG = '#1a1a2e'
PANEL_BG = '#16213e'
ACCENT = '#0f3460'
GREEN = '#4ade80'
YELLOW = '#fbbf24'
RED = '#f87171'
CYAN = '#22d3ee'
FG = '#eee'
MUTED = '#888'
FONT = ('SF Mono', 12)
FONT_SM = ('SF Mono', 11)
FONT_BOLD = ('SF Mono', 13, 'bold')

PAN_STEP = 80  # pixels to pan per arrow key press


class QAApp(tk.Tk):
    def __init__(self, initial_path=None):
        super().__init__()
        self.title("Triangle QA")
        self.geometry("1280x820")
        self.configure(bg=DARK_BG)

        # State
        self.map_path = None
        self.detections = []       # full list from last detection run
        self.manual_marks = []     # list of (x, y) in original-image coords
        self.false_positives = set()  # set of detection indices marked as FP
        self.orig_img = None       # original map image (BGR numpy)
        self.grid_bounds = None    # grid line positions from detection
        self.img_scale = 1.0       # upscale factor used during detection
        self.pil_orig = None       # PIL version of original (no annotations)
        self.pil_annotated = None  # current annotated PIL image
        self.photo = None          # tk PhotoImage for canvas
        self.zoom_level = 1.0
        self.selected = None       # selected detection dict (or None)
        self.selected_idx = None   # index into self.detections
        self.mark_mode = False     # whether clicks add manual marks

        self._build_ui()
        self._bind_events()

        if initial_path and Path(initial_path).exists():
            self._load_path(Path(initial_path))

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Toolbar
        bar = tk.Frame(self, bg=ACCENT, height=50)
        bar.pack(fill='x')
        bar.pack_propagate(False)

        tk.Button(bar, text='Browse…', command=self.browse,
                  bg='#1a5276', fg=FG, relief='flat', padx=10,
                  font=FONT).pack(side='left', padx=8, pady=8)

        self.path_var = tk.StringVar(value='No map selected')
        tk.Label(bar, textvariable=self.path_var, bg=ACCENT, fg=MUTED,
                 font=FONT_SM).pack(side='left', padx=4)

        self.run_btn = tk.Button(bar, text='▶  Run', command=self.run,
                                 bg='#166534', fg=FG, relief='flat', padx=14,
                                 font=FONT_BOLD, state='disabled')
        self.run_btn.pack(side='right', padx=8, pady=8)

        self.mark_btn = tk.Button(bar, text='✦ Mark (M)', command=self.toggle_mark,
                                   bg=ACCENT, fg=MUTED, relief='flat', padx=10,
                                   font=FONT)
        self.mark_btn.pack(side='right', padx=4, pady=8)

        self.save_btn = tk.Button(bar, text='Save (S)', command=self.save_marks,
                                   bg=ACCENT, fg=MUTED, relief='flat', padx=10,
                                   font=FONT)
        self.save_btn.pack(side='right', padx=4, pady=8)

        self.status_var = tk.StringVar()
        tk.Label(bar, textvariable=self.status_var, bg=ACCENT, fg=GREEN,
                 font=FONT_SM).pack(side='right', padx=10)

        # Main area
        main = tk.Frame(self, bg=DARK_BG)
        main.pack(fill='both', expand=True)

        self._build_controls(main)
        self._build_canvas(main)
        self._build_inspector(main)

    def _build_controls(self, parent):
        ctrl = tk.Frame(parent, bg=PANEL_BG, width=220)
        ctrl.pack(side='left', fill='y', padx=(8, 4), pady=8)
        ctrl.pack_propagate(False)

        def section(text):
            tk.Frame(ctrl, bg=ACCENT, height=1).pack(fill='x', padx=8, pady=(14, 6))
            tk.Label(ctrl, text=text, bg=PANEL_BG, fg=MUTED,
                     font=FONT_SM).pack(anchor='w', padx=10)

        def slider(parent_frame, label, var, from_, to_, step, color, cmd):
            tk.Label(parent_frame, text=label, bg=PANEL_BG, fg=MUTED,
                     font=FONT_SM).pack(anchor='w', padx=10, pady=(10, 0))
            val_lbl = tk.Label(parent_frame, bg=PANEL_BG, fg=color, font=FONT_BOLD)
            val_lbl.pack(anchor='w', padx=10)
            sc = tk.Scale(parent_frame, from_=from_, to=to_, resolution=step,
                          orient='horizontal', variable=var, bg=PANEL_BG, fg=FG,
                          highlightthickness=0, troughcolor=ACCENT,
                          activebackground=color, command=cmd, showvalue=False)
            sc.pack(fill='x', padx=10)
            return val_lbl

        # CNN threshold — default to 0.85 (only green)
        section('DISPLAY THRESHOLD')
        self.cnn_var = tk.DoubleVar(value=0.85)
        self.cnn_lbl = slider(ctrl, 'CNN threshold (instant)', self.cnn_var,
                              0.0, 1.0, 0.05, GREEN, self._on_cnn_change)
        self.cnn_lbl.config(text='0.85')

        # Template threshold (requires re-run)
        section('RUN PARAMETERS')
        self.tmpl_var = tk.DoubleVar(value=0.55)
        self.tmpl_lbl = slider(ctrl, 'Template threshold (re-run)', self.tmpl_var,
                               0.20, 0.80, 0.05, YELLOW, self._on_tmpl_change)
        self.tmpl_lbl.config(text='0.55')

        # Stats
        section('RESULTS')
        self.stat_high = tk.Label(ctrl, text='● High  ≥0.85:  —',
                                   bg=PANEL_BG, fg=GREEN, font=FONT_SM)
        self.stat_high.pack(anchor='w', padx=12, pady=1)
        self.stat_fp = tk.Label(ctrl, text='✕ False pos:  —',
                                 bg=PANEL_BG, fg=RED, font=FONT_SM)
        self.stat_fp.pack(anchor='w', padx=12, pady=1)
        self.stat_manual = tk.Label(ctrl, text='◆ Missed:  —',
                                     bg=PANEL_BG, fg=CYAN, font=FONT_SM)
        self.stat_manual.pack(anchor='w', padx=12, pady=1)
        self.stat_total = tk.Label(ctrl, text='Total: —',
                                    bg=PANEL_BG, fg=FG, font=FONT_BOLD)
        self.stat_total.pack(anchor='w', padx=12, pady=(4, 0))

        # Zoom
        section('ZOOM / PAN')
        zf = tk.Frame(ctrl, bg=PANEL_BG)
        zf.pack(fill='x', padx=10, pady=4)
        tk.Button(zf, text='−', command=self.zoom_out,
                  bg=ACCENT, fg=FG, relief='flat', width=3,
                  font=FONT_BOLD).pack(side='left')
        self.zoom_lbl = tk.Label(zf, text='fit', bg=PANEL_BG, fg=FG,
                                  font=FONT_SM, width=6)
        self.zoom_lbl.pack(side='left', padx=4)
        tk.Button(zf, text='+', command=self.zoom_in,
                  bg=ACCENT, fg=FG, relief='flat', width=3,
                  font=FONT_BOLD).pack(side='left')
        tk.Button(ctrl, text='Fit to window  (F)', command=self.zoom_fit,
                  bg=ACCENT, fg=FG, relief='flat', font=FONT_SM
                  ).pack(fill='x', padx=10, pady=2)
        tk.Label(ctrl, text='Arrow keys to pan', bg=PANEL_BG, fg='#555',
                 font=('SF Mono', 10)).pack(anchor='w', padx=12)

        # Log
        section('LOG')
        self.log = tk.Text(ctrl, bg='#0a0a1a', fg=MUTED, font=('SF Mono', 10),
                            wrap='word', relief='flat', state='disabled')
        self.log.pack(fill='both', expand=True, padx=8, pady=(4, 10))

    def _build_canvas(self, parent):
        cf = tk.Frame(parent, bg=DARK_BG)
        cf.pack(side='left', fill='both', expand=True, pady=8)

        self.canvas = tk.Canvas(cf, bg='#111', highlightthickness=0,
                                 cursor='crosshair')
        hbar = tk.Scrollbar(cf, orient='horizontal', command=self.canvas.xview)
        vbar = tk.Scrollbar(cf, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        hbar.pack(side='bottom', fill='x')
        vbar.pack(side='right', fill='y')
        self.canvas.pack(fill='both', expand=True)

        # xscrollincrement=1 makes xview_scroll(N, 'units') move exactly N pixels
        self.canvas.configure(xscrollincrement=1, yscrollincrement=1)

        self.canvas.create_text(500, 300,
                                text='Browse a map and click  ▶ Run',
                                fill='#444', font=('SF Mono', 18), tags='hint')

    def _build_inspector(self, parent):
        insp = tk.Frame(parent, bg=PANEL_BG, width=200)
        insp.pack(side='right', fill='y', padx=(4, 8), pady=8)
        insp.pack_propagate(False)

        tk.Label(insp, text='Inspector', bg=PANEL_BG, fg=MUTED,
                 font=FONT_SM).pack(pady=(14, 8))

        self.crop_canvas = tk.Canvas(insp, width=168, height=168,
                                      bg='#000', highlightthickness=1,
                                      highlightbackground=ACCENT)
        self.crop_canvas.pack(padx=16)
        self.crop_photo = None

        self.insp_cnn = tk.Label(insp, text='CNN: —', bg=PANEL_BG, fg=GREEN,
                                  font=FONT_BOLD)
        self.insp_cnn.pack(pady=(8, 2))
        self.insp_tmpl = tk.Label(insp, text='Tmpl: —', bg=PANEL_BG, fg=YELLOW,
                                   font=FONT_SM)
        self.insp_tmpl.pack(pady=2)
        self.insp_pos = tk.Label(insp, text='', bg=PANEL_BG, fg=MUTED,
                                  font=FONT_SM)
        self.insp_pos.pack(pady=2)

        tk.Frame(insp, bg=ACCENT, height=1).pack(fill='x', padx=10, pady=14)

        self.insp_hint = tk.Label(insp,
                                  text='Click circle → inspect\nX → false positive\n\n'
                                       'M → mark mode\n(add missed triangles)\n\n'
                                       'Right-click → add mark\nCtrl+Z → undo mark\n\n'
                                       'S → save',
                                  bg=PANEL_BG, fg='#555', font=FONT_SM,
                                  justify='center')
        self.insp_hint.pack()

        # Mark mode indicator
        self.mode_label = tk.Label(insp, text='', bg=PANEL_BG, fg=CYAN,
                                    font=FONT_BOLD)
        self.mode_label.pack(pady=(10, 0))

    # ── Events ────────────────────────────────────────────────────────────────

    def _bind_events(self):
        self.canvas.bind('<Button-1>', self.on_click)
        # Right-click always adds a mark regardless of mode
        self.canvas.bind('<Button-2>', self.on_right_click)  # Mac middle
        self.canvas.bind('<Button-3>', self.on_right_click)  # Right-click
        self.canvas.bind('<MouseWheel>', self.on_scroll)     # Mac
        self.canvas.bind('<Button-4>', self.on_scroll)       # Linux up
        self.canvas.bind('<Button-5>', self.on_scroll)       # Linux down

        # Focus canvas to receive key events
        self.canvas.bind('<Enter>', lambda _: self.canvas.focus_set())

        # Arrow keys for panning (bind to canvas so they work when focused)
        self.canvas.bind('<Left>',  lambda _: self.pan(-PAN_STEP, 0))
        self.canvas.bind('<Right>', lambda _: self.pan(PAN_STEP, 0))
        self.canvas.bind('<Up>',    lambda _: self.pan(0, -PAN_STEP))
        self.canvas.bind('<Down>',  lambda _: self.pan(0, PAN_STEP))

        # Global keys
        self.bind('f', lambda _: self.zoom_fit())
        self.bind('F', lambda _: self.zoom_fit())
        self.bind('+', lambda _: self.zoom_in())
        self.bind('=', lambda _: self.zoom_in())
        self.bind('-', lambda _: self.zoom_out())
        self.bind('m', lambda _: self.toggle_mark())
        self.bind('M', lambda _: self.toggle_mark())
        self.bind('s', lambda _: self.save_marks())
        self.bind('S', lambda _: self.save_marks())
        self.bind('x', lambda _: self.mark_fp())
        self.bind('X', lambda _: self.mark_fp())
        # Ctrl+Z to undo last mark
        self.bind('<Control-z>', lambda _: self.undo_mark())
        self.bind('<Command-z>', lambda _: self.undo_mark())  # Mac

    # ── Pan ───────────────────────────────────────────────────────────────────

    def pan(self, dx, dy):
        """Scroll the canvas by dx, dy pixels."""
        if dx:
            self.canvas.xview_scroll(dx, 'units')
        if dy:
            self.canvas.yview_scroll(dy, 'units')

    # ── File / Run ────────────────────────────────────────────────────────────

    def browse(self):
        path = filedialog.askopenfilename(
            title='Select Map Image',
            filetypes=[('Images', '*.jpg *.jpeg *.tif *.tiff *.png'), ('All', '*')])
        if path:
            self._load_path(Path(path))

    def _load_path(self, path):
        # Marks belong to a map — clear them when a different map is loaded
        if self.map_path != path:
            self.manual_marks = []
            self.false_positives = set()
        self.map_path = path
        self.path_var.set(path.name)
        self.run_btn.config(state='normal')

    def run(self):
        if not self.map_path:
            return
        self.run_btn.config(state='disabled', text='Running…')
        self.status_var.set('')
        self._log_clear()
        threading.Thread(target=self._run_thread, daemon=True).start()

    def _run_thread(self):
        try:
            dets, orig, scale, grid_bounds = run_full_detection(
                self.map_path, self.tmpl_var.get(), self._log)
            self.after(0, lambda: self._on_done(dets, orig, scale, grid_bounds))
        except Exception as e:
            import traceback
            self._log(f'ERROR: {e}')
            self._log(traceback.format_exc())
            self.after(0, lambda: self.run_btn.config(
                state='normal', text='▶  Run'))

    def _on_done(self, dets, orig, scale, grid_bounds=None):
        self.detections = dets
        self.orig_img = orig
        self.grid_bounds = grid_bounds
        self.img_scale = scale
        # Keep manual_marks: they are map locations in original-image
        # coordinates, still valid after a re-run with new thresholds.
        # FP indices point into the OLD detections list — must reset.
        self.false_positives = set()
        self.selected = None
        self.selected_idx = None
        if self.manual_marks:
            self._log(f"Kept {len(self.manual_marks)} manual marks from previous run")
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        self.pil_orig = Image.fromarray(orig_rgb)
        self._refresh_annotations()
        self.zoom_fit()
        self.run_btn.config(state='normal', text='▶  Run')
        self._update_stats()

    # ── Mark mode ─────────────────────────────────────────────────────────────

    def toggle_mark(self):
        self.mark_mode = not self.mark_mode
        if self.mark_mode:
            self.mark_btn.config(bg='#164e63', fg=CYAN)
            self.mode_label.config(text='◆ MARK MODE\nClick map to add')
            self.canvas.config(cursor='tcross')
            self.status_var.set('Mark mode ON — click to add missed triangles')
        else:
            self.mark_btn.config(bg=ACCENT, fg=MUTED)
            self.mode_label.config(text='')
            self.canvas.config(cursor='crosshair')
            self.status_var.set('')

    def add_mark(self, img_x, img_y):
        """Add or remove a manual mark. Removes if clicking near an existing one."""
        # Check if clicking near an existing mark (within 30 image pixels)
        for i, (mx, my) in enumerate(self.manual_marks):
            dist = ((mx - img_x) ** 2 + (my - img_y) ** 2) ** 0.5
            if dist < 30:
                self.manual_marks.pop(i)
                self._log(f"Removed mark at ({mx}, {my})")
                self._refresh_annotations()
                return
        # No nearby mark — add new one
        self.manual_marks.append((int(img_x), int(img_y)))
        self._log(f"Marked triangle at ({int(img_x)}, {int(img_y)})")
        self._refresh_annotations()

    def undo_mark(self):
        if self.manual_marks:
            removed = self.manual_marks.pop()
            self._log(f"Undid mark at ({removed[0]}, {removed[1]})")
            self._refresh_annotations()

    def mark_fp(self):
        """Mark the currently selected detection as a false positive."""
        if self.selected_idx is None:
            return
        if self.selected_idx in self.false_positives:
            # Toggle off
            self.false_positives.discard(self.selected_idx)
            self._log(f"Unmarked FP at ({self.selected['x']}, {self.selected['y']})")
        else:
            self.false_positives.add(self.selected_idx)
            self._log(f"Marked FP at ({self.selected['x']}, {self.selected['y']}) "
                      f"CNN={self.selected['cnn']:.2f}")
        self._refresh_annotations()
        self._update_stats()

    def save_marks(self):
        """Save session results:
        - JSON with auto detections + missed marks next to the map
        - FP crops written directly to training_data/hard_negatives/
        """
        if not self.map_path:
            return

        map_name = self.map_path.stem
        thresh = self.cnn_var.get()

        # Auto-detected above threshold (excluding marked FPs)
        auto = [{'x': d['x'], 'y': d['y'], 'cnn': round(d['cnn'], 3),
                 'source': 'auto'}
                for i, d in enumerate(self.detections)
                if d['cnn'] >= thresh and i not in self.false_positives]

        # Manual missed marks
        manual = [{'x': mx, 'y': my, 'cnn': None, 'source': 'manual'}
                  for mx, my in self.manual_marks]

        # False positives listed in JSON too
        fp_list = [{'x': self.detections[i]['x'], 'y': self.detections[i]['y'],
                    'cnn': round(self.detections[i]['cnn'], 3), 'source': 'false_positive'}
                   for i in self.false_positives if i < len(self.detections)]

        out = {'map': map_name, 'file': str(self.map_path),
               'cnn_threshold': thresh,
               'detections': auto,
               'manual_marks': manual,
               'false_positives': fp_list}

        # Save JSON next to the map
        out_path = self.map_path.parent / f'{map_name}_qa_marks.json'
        out_path.write_text(json.dumps(out, indent=2))

        # Write FP crops to hard_negatives/ (immediately usable for retraining)
        hn_dir = BASE_DIR / 'training_data' / 'hard_negatives'
        hn_dir.mkdir(parents=True, exist_ok=True)
        saved_fp = 0
        for i in self.false_positives:
            if i >= len(self.detections):
                continue
            d = self.detections[i]
            fname = f"{map_name}_x{d['x']}_y{d['y']}_c{d['cnn']:.2f}_fp.png"
            dest = hn_dir / fname
            if not dest.exists():
                cv2.imwrite(str(dest), d['crop'])
                saved_fp += 1

        msg = (f"Saved: {len(auto)} auto + {len(manual)} missed + "
               f"{saved_fp} FP crops → hard_negatives/")
        self._log(msg)
        self.status_var.set(f'Saved to {out_path.name}')

    # ── Annotations ───────────────────────────────────────────────────────────

    def _refresh_annotations(self):
        """Redraw circles at current CNN threshold (no re-run needed)."""
        if self.pil_orig is None:
            return
        thresh = self.cnn_var.get()
        annotated = annotate_image(self.orig_img, self.detections, thresh,
                                   self.manual_marks, self.false_positives,
                                   self.grid_bounds)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        self.pil_annotated = Image.fromarray(rgb)
        self._redraw()
        self._update_stats()
        n = sum(1 for d in self.detections if d['cnn'] >= thresh)
        self.status_var.set(f'{n} auto + {len(self.manual_marks)} manual')

    def _redraw(self):
        if self.pil_annotated is None:
            return
        w = int(self.pil_annotated.width * self.zoom_level)
        h = int(self.pil_annotated.height * self.zoom_level)
        img = self.pil_annotated.resize((w, h), Image.BILINEAR)

        # Highlight selected detection
        if self.selected:
            img = img.copy()
            draw = ImageDraw.Draw(img)
            sx = int(self.selected['x'] * self.zoom_level)
            sy = int(self.selected['y'] * self.zoom_level)
            r = max(16, int(22 * self.zoom_level))
            draw.ellipse([sx - r, sy - r, sx + r, sy + r], outline='white', width=3)

        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, w, h))

    # ── Zoom ──────────────────────────────────────────────────────────────────

    def zoom_fit(self):
        if self.pil_annotated is None:
            return
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        self.zoom_level = min(cw / self.pil_annotated.width,
                              ch / self.pil_annotated.height) * 0.97
        self.zoom_lbl.config(text='fit')
        self._redraw()

    def zoom_in(self):
        self.zoom_level = min(self.zoom_level * 1.35, 8.0)
        self.zoom_lbl.config(text=f'{self.zoom_level:.2f}×')
        self._redraw()

    def zoom_out(self):
        self.zoom_level = max(self.zoom_level / 1.35, 0.03)
        self.zoom_lbl.config(text=f'{self.zoom_level:.2f}×')
        self._redraw()

    def on_scroll(self, event):
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            self.zoom_in()
        else:
            self.zoom_out()

    # ── Click ─────────────────────────────────────────────────────────────────

    def on_click(self, event):
        if self.orig_img is None:
            return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        ix = cx / self.zoom_level
        iy = cy / self.zoom_level

        if self.mark_mode:
            self.add_mark(ix, iy)
            return

        # Normal mode: inspect nearest detection
        if not self.detections:
            return
        thresh = self.cnn_var.get()
        best, best_idx, best_dist = None, None, float('inf')
        for i, d in enumerate(self.detections):
            if d['cnn'] < thresh:
                continue
            dist = ((d['x'] - ix) ** 2 + (d['y'] - iy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best, best_idx = dist, d, i

        if best and best_dist * self.zoom_level < 40:
            self.selected = best
            self.selected_idx = best_idx
            self._show_inspector(best)
            self._redraw()

    def on_right_click(self, event):
        """Right-click always adds a mark, regardless of mode."""
        if self.orig_img is None:
            return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        ix = cx / self.zoom_level
        iy = cy / self.zoom_level
        self.add_mark(ix, iy)

    def _show_inspector(self, d):
        crop_rgb = cv2.cvtColor(d['crop'], cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(crop_rgb).resize((168, 168), Image.NEAREST)
        self.crop_photo = ImageTk.PhotoImage(pil)
        self.crop_canvas.delete('all')
        self.crop_canvas.create_image(0, 0, anchor='nw', image=self.crop_photo)

        cnn = d['cnn']
        is_fp = self.selected_idx in self.false_positives
        color = RED if is_fp else (GREEN if cnn >= 0.85 else YELLOW if cnn >= 0.70 else RED)
        label = f'CNN: {cnn:.3f}' + ('  [FP]' if is_fp else '')
        self.insp_cnn.config(text=label, fg=color)
        self.insp_tmpl.config(text=f'Tmpl: {d["tmpl"]:.3f}')
        self.insp_pos.config(text=f'x={d["x"]}  y={d["y"]}\nPress X to mark FP')

    # ── Stats / Log ───────────────────────────────────────────────────────────

    def _update_stats(self):
        thresh = self.cnn_var.get()
        # false_positives holds indices into self.detections — enumerate the
        # full list, not a filtered copy, or FP exclusion hits the wrong items
        n_h = sum(1 for i, d in enumerate(self.detections)
                  if d['cnn'] >= thresh and d['cnn'] >= 0.85
                  and i not in self.false_positives)
        n_fp = len(self.false_positives)
        n_m = len(self.manual_marks)
        self.stat_high.config(text=f'● High  ≥0.85:  {n_h}')
        self.stat_fp.config(text=f'✕ False pos:  {n_fp}')
        self.stat_manual.config(text=f'◆ Missed:  {n_m}')
        self.stat_total.config(text=f'True positives: {n_h + n_m}')

    def _on_cnn_change(self, val):
        self.cnn_lbl.config(text=f'{float(val):.2f}')
        if self.detections or self.manual_marks:
            self._refresh_annotations()

    def _on_tmpl_change(self, val):
        self.tmpl_lbl.config(text=f'{float(val):.2f}')
        self.run_btn.config(bg='#92400e')  # orange = needs re-run

    def _log(self, msg):
        self.after(0, lambda: self._append_log(msg))

    def _append_log(self, msg):
        self.log.config(state='normal')
        self.log.insert('end', msg + '\n')
        self.log.see('end')
        self.log.config(state='disabled')
        self.run_btn.config(bg='#166534')

    def _log_clear(self):
        self.log.config(state='normal')
        self.log.delete('1.0', 'end')
        self.log.config(state='disabled')


if __name__ == '__main__':
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    app = QAApp(initial_path=initial)
    app.mainloop()
