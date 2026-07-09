#!/usr/bin/env python3
"""Presentation assets: results table, accuracy bar chart, headline KPI card.
Real numbers from the final sample-series end-to-end benchmark (tol=0.25). PIL only."""
from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_paths import BASE_DIR

OUT = BASE_DIR / 'presentation'
OUT.mkdir(exist_ok=True)
FD = '/opt/anaconda3/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'
FB = '/opt/anaconda3/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf'
f = lambda s, b=False: ImageFont.truetype(FB if b else FD, s)

INK = (28, 31, 36)
MUT = (120, 126, 135)
LINE = (222, 226, 232)
GRN = (22, 158, 90)
GRN_BG = (231, 246, 238)
AMB = (196, 130, 20)
AMB_BG = (250, 240, 219)
RED = (196, 60, 55)
RED_BG = (250, 232, 231)
BLU = (37, 99, 175)
BAND = (245, 247, 250)

# map, ocr, pts, inl, fit, median, maxerr, status(0 ok / 1 fail), note
ROWS = [
    ('M13_4037', '15E/18N', 50, 50, 9.5, 5.9, 11.5, 0, ''),
    ('M5_4048', '10E/12N', 37, 37, 7.6, 9.5, 18.8, 0, ''),
    ('M7_4138', '14E/14N', 40, 38, 10.1, 11.8, 14.3, 0, ''),
    ('M4_4059', '13E/15N', 31, 30, 8.2, 13.2, 24.2, 0, ''),
    ('M12_4116', '14E/12N', 50, 49, 10.9, 14.3, 21.5, 0, ''),
    ('M9_4149', '12E/17N', 14, 14, 9.0, 21.3, 21.4, 0, ''),
    ('M14_4150', '19E/16N', 38, 38, 9.1, 44.0, 111.7, 0, ''),
    ('M8_4082', '25E/14N', 6, 6, 20.2, 64.1, 120.8, 0, ''),
    ('M11_4071', '16E/14N', 8, 8, 6.1, 70.4, 70.5, 0, ''),
]

# status buckets by median control-point error (m): text, fg, bg, font size
CATS = [('georeferenced', GRN, GRN_BG, 17),
        ('requires manual refinement', AMB, AMB_BG, 15),
        ('failure · discrepancies too large', RED, RED_BG, 14)]


def bucket(med):
    return 0 if med <= 7 else (1 if med <= 15 else 2)


def err_color(v):
    if v is None:
        return MUT
    return GRN if v < 25 else (AMB if v < 70 else RED)


def center(dr, box, txt, font, fill):
    x0, y0, x1, y1 = box
    w = dr.textlength(txt, font=font)
    a, d = font.getmetrics()
    dr.text(((x0 + x1 - w) / 2, (y0 + y1 - a - d) / 2), txt, font=font, fill=fill)


def rrect(dr, box, r, fill):
    dr.rounded_rectangle(box, radius=r, fill=fill)


def pill(dr, box, text, font, fg, bg):
    x0, y0, x1, y1 = box
    pw = dr.textlength(text, font=font) + 40
    cx = (x0 + x1) / 2
    rrect(dr, (cx - pw / 2, y0 + 9, cx + pw / 2, y1 - 9), 16, bg)
    center(dr, box, text, font, fg)


# ============================ TABLE ============================
def build_table():
    cols = [('Sheet', 134, 'l'), ('OCR labels', 132, 'c'), ('Points', 94, 'c'),
            ('Inliers', 128, 'c'), ('RMSE, m', 116, 'c'),
            ('Error,\nmedian, m', 152, 'c'), ('max, m', 104, 'c'),
            ('Status', 316, 'c')]
    M, rowh, hh = 44, 50, 70
    tw = sum(c[1] for c in cols)
    W = M * 2 + tw
    H = 140 + hh + len(ROWS) * rowh + 84
    im = Image.new('RGB', (W, H), 'white')
    dr = ImageDraw.Draw(im)
    dr.text((M, 40), 'End-to-end benchmark results — series sample-series', font=f(34, True), fill=INK)
    dr.text((M, 86), 'Each sheet processed automatically "blind"; error in meters '
            'vs. manual ground truth', font=f(19), fill=MUT)
    y = 140
    # header
    x = M
    for head, cwid, _ in cols:
        lines = head.split('\n')
        for k, ln in enumerate(lines):
            center(dr, (x, y + 8 + k * 22, x + cwid, y + 30 + k * 22), ln, f(19, True), INK)
        x += cwid
    dr.line([M, y + hh, M + tw, y + hh], fill=INK, width=2)
    y += hh
    for ri, r in enumerate(ROWS):
        mp, ocr, pts, inl, fit, med, mx, st, note = r
        if ri % 2 == 1:
            dr.rectangle([M, y, M + tw, y + rowh], fill=BAND)
        x = cols[0][1]
        center(dr, (M, y, M + cols[0][1], y + rowh), '', f(18), INK)
        # left-align map id
        a, d = f(20, True).getmetrics()
        dr.text((M + 16, y + (rowh - a - d) / 2), mp, font=f(20, True), fill=INK)
        boxes = [ocr,
                 '—' if pts is None else str(pts),
                 '—' if inl is None else str(inl),
                 '—' if fit is None else f'{fit:.1f}']
        xx = M + cols[0][1]
        for ci, val in enumerate(boxes, 1):
            center(dr, (xx, y, xx + cols[ci][1], y + rowh), val, f(19), INK if val != '—' else MUT)
            xx += cols[ci][1]
        cat = bucket(med)
        col = CATS[cat][1]
        center(dr, (xx, y, xx + cols[5][1], y + rowh), f'{med:.1f}', f(22, True), col)
        xx += cols[5][1]
        center(dr, (xx, y, xx + cols[6][1], y + rowh), f'{mx:.0f}', f(19), col)
        xx += cols[6][1]
        txt, fg, bg, fs = CATS[cat]
        pill(dr, (xx, y, xx + cols[7][1], y + rowh), txt, f(fs, True), fg, bg)
        dr.line([M, y + rowh, M + tw, y + rowh], fill=LINE, width=1)
        y += rowh
    dr.text((M, y + 24), '9 sheets processed · 1 automatic · 4 need manual refinement · '
            '4 excessive error', font=f(20, True), fill=BLU)
    p = OUT / 'results_table.png'
    im.save(p, dpi=(200, 200))
    print('saved', p, im.size)


# ======================= ACCURACY BARS =======================
def build_bars():
    data = sorted([(r[0], r[5]) for r in ROWS if r[7] == 0], key=lambda t: t[1])
    W, H = 1240, 640
    im = Image.new('RGB', (W, H), 'white')
    dr = ImageDraw.Draw(im)
    dr.text((48, 40), 'Automatic georeferencing accuracy by sheet', font=f(32, True), fill=INK)
    dr.text((48, 82), 'median error at control points, meters (lower is better)',
            font=f(20), fill=MUT)
    px0, py0, px1, py1 = 96, 150, W - 60, H - 96
    vmax = 80
    for gv in range(0, vmax + 1, 20):
        gy = py1 - (gv / vmax) * (py1 - py0)
        dr.line([px0, gy, px1, gy], fill=LINE, width=1)
        dr.text((px0 - 46, gy - 11), f'{gv}', font=f(17), fill=MUT)
    # reference band: cartographic line width ~ up to 20 m
    gy = py1 - (20 / vmax) * (py1 - py0)
    dr.line([px0, gy, px1, gy], fill=(150, 190, 160), width=2)
    dr.text((px0 + 14, gy - 26), 'about one map line width (1:20,000)', font=f(16, True), fill=GRN)
    n = len(data)
    slot = (px1 - px0) / n
    bw = slot * 0.56
    for i, (mp, v) in enumerate(data):
        cx = px0 + slot * (i + 0.5)
        bh = (v / vmax) * (py1 - py0)
        col = GRN if v < 25 else (AMB if v < 70 else RED)
        dr.rectangle([cx - bw / 2, py1 - bh, cx + bw / 2, py1], fill=col)
        lbl = f'{v:.1f}'
        dr.text((cx - dr.textlength(lbl, font=f(19, True)) / 2, py1 - bh - 28), lbl, font=f(19, True), fill=col)
        dr.text((cx - dr.textlength(mp, font=f(16)) / 2, py1 + 10), mp, font=f(16), fill=INK)
    dr.line([px0, py1, px1, py1], fill=INK, width=2)
    dr.text((px0 - 76, py0 - 40), 'meters', font=f(17, True), fill=MUT)
    p = OUT / 'accuracy_bars.png'
    im.save(p, dpi=(200, 200))
    print('saved', p, im.size)


# ========================= KPI CARD =========================
def build_kpi():
    W, H = 1240, 500
    im = Image.new('RGB', (W, H), 'white')
    dr = ImageDraw.Draw(im)
    dr.text((60, 46), 'Algorithm improvement result', font=f(34, True), fill=INK)
    dr.text((60, 92), 'after fixing the grid-label reading bug in the map margins',
            font=f(21), fill=MUT)
    def card(x0, x1, big, line1, line2, col):
        rrect(dr, (x0, 158, x1, 430), 20, (245, 247, 250))
        center(dr, (x0, 186, x1, 302), big, f(84, True), col)
        center(dr, (x0, 326, x1, 366), line1, f(22, True), INK)
        center(dr, (x0, 366, x1, 406), line2, f(22, True), INK)
    card(60, 470, '5 → 9', 'of 12 sheets', 'georeferenced automatically', BLU)
    card(500, 830, '14.3 m', 'median', 'accuracy', GRN)
    card(860, 1180, '5.9 m', 'best', 'result', GRN)
    p = OUT / 'summary_kpi.png'
    im.save(p, dpi=(200, 200))
    print('saved', p, im.size)


build_table()
build_bars()
build_kpi()
