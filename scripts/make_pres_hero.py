#!/usr/bin/env python3
"""Presentation asset: real triangle detections on a map, with zoom insets.

Blindly-recovered transform (output/end_to_end/M13_4037) is used ONLY to
project the geodetic DB; each projected point is then verified by the actual
template-match + detection engine. Circles = symbols the system found on its
own. Russian captions for the slide deck.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from data_paths import BASE_DIR, GEODETIC_DB, TEMPLATE_DIR
from coord_converter import load_tfwx, get_map_extent, map_to_pixel
from image_loader import load_image
from db_matcher import (load_geodetic_db, load_grayscale_templates,
                        filter_points_to_extent, verify_candidates)

MAP_ID = 'M13_4037'
ACC_M = 5.9
IMG = BASE_DIR / 'Map_Scans' / 'sample-series' / MAP_ID / f'{MAP_ID}.tif'
TFWX = BASE_DIR / 'output' / 'end_to_end' / MAP_ID / f'{MAP_ID}.tfwx'
OUT = BASE_DIR / 'presentation'
OUT.mkdir(exist_ok=True)
FONT = '/opt/anaconda3/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'
FONTB = '/opt/anaconda3/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf'
f = lambda s, b=False: ImageFont.truetype(FONTB if b else FONT, s)

INK = (30, 33, 38)
MUT = (110, 116, 125)
GRN = (22, 158, 90)
ACC = (37, 99, 175)

print("loading DB + image...")
db = load_geodetic_db(GEODETIC_DB)
aff = load_tfwx(TFWX)
img = load_image(str(IMG))
h, w = img.shape[:2]
ext = get_map_extent(aff, w, h)
cand = filter_points_to_extent(db, ext)
templates = load_grayscale_templates(TEMPLATE_DIR)
print(f"  {w}x{h}, {len(cand)} DB points in extent, {len(templates)} templates")

print("running detection engine...")
dets = verify_candidates(img, cand, aff, templates, color_mode='suppress_red')
hits = [d for d in dets if d.confidence >= 0.6]
print(f"  {len(hits)} detections >= 0.6")

# --- overview (map + all detections) ---
ov_h = 1250
sc = ov_h / h
ov_w = int(round(w * sc))
ov = cv2.cvtColor(cv2.resize(img, (ov_w, ov_h), interpolation=cv2.INTER_AREA),
                  cv2.COLOR_BGR2RGB)
ov = Image.fromarray(ov)
od = ImageDraw.Draw(ov)
for d in hits:
    x, y = d.pixel_x * sc, d.pixel_y * sc
    od.ellipse([x - 6, y - 6, x + 6, y + 6], outline=GRN, width=2)

# --- pick 3 high-conf, well-separated detections for insets ---
picks = []
for d in sorted(hits, key=lambda d: -d.confidence):
    if all((d.pixel_x - p.pixel_x) ** 2 + (d.pixel_y - p.pixel_y) ** 2 > 3200 ** 2
           for p in picks):
        picks.append(d)
    if len(picks) == 3:
        break

S = 400
CROP = 200
insets = []
for i, d in enumerate(picks, 1):
    cx, cy = int(round(d.pixel_x)), int(round(d.pixel_y))
    x1, y1 = max(0, cx - CROP // 2), max(0, cy - CROP // 2)
    crop = img[y1:y1 + CROP, x1:x1 + CROP]
    crop = cv2.cvtColor(cv2.resize(crop, (S, S), interpolation=cv2.INTER_CUBIC),
                        cv2.COLOR_BGR2RGB)
    im = Image.fromarray(crop)
    dd = ImageDraw.Draw(im)
    # no marker circle inside insets — clean zoom crops (circles added by hand)
    dd.rectangle([0, 0, S - 1, S - 1], outline=(210, 214, 220), width=2)
    insets.append((i, d, im))
    # marker on overview
    ox, oy = d.pixel_x * sc, d.pixel_y * sc
    od.rectangle([ox - 11, oy - 11, ox + 11, oy + 11], outline=ACC, width=3)
    od.text((ox + 13, oy - 22), str(i), font=f(26, True), fill=ACC)

# --- compose ---
M = 50
HEAD = 150
FOOT = 96
cw = M + ov_w + 50 + S + M
ch = M + HEAD + ov_h + FOOT + M
canvas = Image.new('RGB', (cw, ch), 'white')
dr = ImageDraw.Draw(canvas)
dr.text((M, M), 'Automatic detection of triangulation control points',
        font=f(38, True), fill=INK)
dr.text((M, M + 56),
        f'Sheet {MAP_ID} · blind georeferencing from detected symbols, '
        f'no prior world file', font=f(23), fill=MUT)
top = M + HEAD
canvas.paste(ov, (M, top))
ix = M + ov_w + 50
gap = (ov_h - 3 * S) // 2
for k, (i, d, im) in enumerate(insets):
    iy = top + k * (S + gap)
    canvas.paste(im, (ix, iy))
    dr.ellipse([ix + 8, iy + 8, ix + 42, iy + 42], fill=ACC)
    dr.text((ix + 18, iy + 12), str(i), font=f(24, True), fill='white')
    dr.text((ix + 52, iy + 13),
            f'confidence {d.confidence:.2f}', font=f(21, True), fill='white')
fy = top + ov_h + 22
dr.text((M, fy),
        f'Detected {len(hits)} triangulation symbols', font=f(28, True), fill=GRN)
dr.text((M, fy + 40),
        f'Final georeferencing accuracy: {ACC_M:.1f} m at control points '
        f'(about one map line width at 1:20,000 scale)',
        font=f(21), fill=MUT)
p = OUT / 'detection_map.png'
canvas.save(p, dpi=(200, 200))
print("saved", p, canvas.size)
