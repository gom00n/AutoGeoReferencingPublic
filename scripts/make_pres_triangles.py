#!/usr/bin/env python3
"""Presentation asset: a gallery of confirmed triangulation symbols from
several sheets. Each tile is a human-verified geodetic control point (i.e. a
real triangulation symbol) projected via the ground-truth world file and
cropped in colour from the original scan."""
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_end_to_end import locate_assets
from ingest_job_maps import load_points_file, parse_aux_gcps
from coord_converter import load_tfwx, map_to_pixel
from image_loader import load_image

MAPS = ['M13_4037', 'M5_4048', 'M7_4138', 'M4_4059', 'M12_4116']
PER_MAP = 4
N_TILES = 15
WIN = 150            # native crop half-window*2
OUT = Path(__file__).resolve().parent.parent / 'presentation'
OUT.mkdir(exist_ok=True)
FD = '/opt/anaconda3/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'
FB = '/opt/anaconda3/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf'
f = lambda s, b=False: ImageFont.truetype(FB if b else FD, s)
INK, MUT = (28, 31, 36), (120, 126, 135)

assets = locate_assets(set(MAPS))
per_map = {}
for mid in MAPS:
    a = assets.get(mid)
    if not a or a['tfwx'] is None or a['image'] is None:
        print('skip', mid, '(no assets)'); continue
    targets = load_points_file(a['points']) if a['points'] else None
    if (targets is None or len(targets) == 0) and a['aux']:
        g = parse_aux_gcps(a['aux'])
        targets = g[1] if g else None
    if targets is None or len(targets) == 0:
        print('skip', mid, '(no control points)'); continue
    img = load_image(str(a['image']))
    h, w = img.shape[:2]
    gt = load_tfwx(a['tfwx'])
    px = [map_to_pixel(tx, ty, gt) for tx, ty in targets]
    # keep in-frame with room for the window, spread out
    chosen = []
    for (x, y) in px:
        if not (WIN < x < w - WIN and WIN < y < h - WIN):
            continue
        if all((x - cx) ** 2 + (y - cy) ** 2 > 1800 ** 2 for cx, cy in chosen):
            chosen.append((x, y))
        if len(chosen) >= PER_MAP:
            break
    crops = []
    for (x, y) in chosen:
        xi, yi = int(round(x)), int(round(y))
        c = img[yi - WIN:yi + WIN, xi - WIN:xi + WIN]
        crops.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
    per_map[mid] = crops
    print(f'{mid}: {len(crops)} symbols')

# round-robin so the grid mixes sheets
tiles = []
i = 0
while len(tiles) < N_TILES and any(per_map.values()):
    mid = MAPS[i % len(MAPS)]
    i += 1
    if per_map.get(mid):
        tiles.append((mid, per_map[mid].pop(0)))
    if i > 200:
        break

# ---- compose grid ----
COLS, TILE, GAP, M, LBL = 5, 240, 18, 44, 30
ROWS = (len(tiles) + COLS - 1) // COLS
W = M * 2 + COLS * TILE + (COLS - 1) * GAP
H = 132 + ROWS * (TILE + LBL) + (ROWS - 1) * GAP + M
im = Image.new('RGB', (W, H), 'white')
dr = ImageDraw.Draw(im)
dr.text((M, 44), 'Confirmed triangulation symbols from multiple sheets', font=f(34, True), fill=INK)
dr.text((M, 88), 'human-verified geodetic control points, cropped in colour from the original scans',
        font=f(19), fill=MUT)
for k, (mid, arr) in enumerate(tiles):
    col, row = k % COLS, k // COLS
    x = M + col * (TILE + GAP)
    y = 132 + row * (TILE + LBL + GAP)
    tile = Image.fromarray(arr).resize((TILE, TILE), Image.LANCZOS)
    im.paste(tile, (x, y))
    dr.rectangle([x, y, x + TILE - 1, y + TILE - 1], outline=(205, 209, 216), width=2)
    wlab = dr.textlength(mid, font=f(17))
    dr.text((x + (TILE - wlab) / 2, y + TILE + 6), mid, font=f(17), fill=MUT)
p = OUT / 'triangle_examples.png'
im.save(p, dpi=(200, 200))
print('saved', p, im.size, f'({len(tiles)} tiles)')
