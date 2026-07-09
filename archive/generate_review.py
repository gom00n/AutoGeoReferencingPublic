"""
Generate an HTML review page for triangle detections.

For each detection, shows:
- Cropped image around the detected location
- Point name, confidence, method
- Buttons to mark as: True Positive, False Positive, Uncertain

Results are saved to a JSON file that can be fed back to improve detection.

Usage:
    python generate_review.py <map_dir> [--bootstrap] [--grid-points FILE]

For maps with TFWX:
    python generate_review.py ../T1/M5_4048

For bootstrap maps (Control_Maps):
    python generate_review.py ../Control_Maps/M5_4598 --bootstrap
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from base64 import b64encode

from image_loader import load_image, suppress_red
from db_matcher import (
    load_geodetic_db, load_grayscale_templates, filter_points_to_extent,
    verify_candidates,
)
from coord_converter import load_tfwx, get_map_extent


def generate_crops(img, detections, crop_size=120):
    """Generate crop images for each detection. Returns list of (detection, crop_bytes)."""
    half = crop_size // 2
    h_img, w_img = img.shape[:2]
    results = []

    for det in detections:
        px_i, py_i = int(round(det.pixel_x)), int(round(det.pixel_y))
        y1, y2 = max(0, py_i - half), min(h_img, py_i + half)
        x1, x2 = max(0, px_i - half), min(w_img, px_i + half)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_big = cv2.resize(crop, (240, 240), interpolation=cv2.INTER_LINEAR)
        # Draw crosshair
        ch, cw = crop_big.shape[:2]
        cv2.line(crop_big, (cw//2-12, ch//2), (cw//2+12, ch//2), (0, 255, 0), 1)
        cv2.line(crop_big, (cw//2, ch//2-12), (cw//2, ch//2+12), (0, 255, 0), 1)

        _, buf = cv2.imencode('.jpg', crop_big, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = b64encode(buf).decode('ascii')
        results.append((det, b64))

    return results


def generate_html(map_name, crops_data, output_path):
    """Generate a self-contained HTML review page with keyboard-driven grading.

    Grading scale (keys 1-5):
      1 = ? (uncertain / can't tell)
      2 = FP (no triangle here)
      3 = TP-far (triangle visible but cross is way off center)
      4 = TP-close (triangle found, cross slightly off center)
      5 = TP (perfect or near-perfect centering)

    Navigation: Arrow keys or J/K to move between cards. Space to skip.
    """
    cards_html = []
    for i, (det, b64_img) in enumerate(crops_data):
        conf = det.confidence
        name = det.geo_point.name
        method = det.method
        height = det.geo_point.height
        px, py = det.pixel_x, det.pixel_y

        if conf >= 0.7:
            badge_class = 'badge-green'
        elif conf >= 0.5:
            badge_class = 'badge-yellow'
        else:
            badge_class = 'badge-red'

        card = f'''
        <div class="card" data-idx="{i}" data-name="{name}" data-conf="{conf:.4f}">
            <img src="data:image/jpeg;base64,{b64_img}" />
            <div class="info">
                <div class="point-name">{name}</div>
                <div><span class="badge {badge_class}">{conf:.3f}</span> {method}</div>
                <div class="detail">h={height} px=({px:.0f},{py:.0f})</div>
            </div>
            <div class="buttons">
                <button class="btn-1" onclick="grade(this,1)">1:?</button>
                <button class="btn-2" onclick="grade(this,2)">2:FP</button>
                <button class="btn-3" onclick="grade(this,3)">3:far</button>
                <button class="btn-4" onclick="grade(this,4)">4:close</button>
                <button class="btn-5" onclick="grade(this,5)">5:TP</button>
            </div>
        </div>'''
        cards_html.append(card)

    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Detection Review: {map_name}</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; margin: 20px; }}
h1 {{ color: #e94560; margin-bottom: 4px; }}
.help {{ color: #888; font-size: 13px; margin-bottom: 12px; }}
.help kbd {{ background: #333; padding: 1px 6px; border-radius: 3px; border: 1px solid #555; font-size: 12px; }}
.controls {{ margin: 10px 0 16px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
.controls button {{ padding: 5px 12px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; }}
.controls select {{ padding: 5px; border-radius: 4px; font-size: 13px; }}
.stats {{ color: #aaa; font-size: 13px; }}
.grid {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.card {{ background: #16213e; border-radius: 8px; padding: 8px; width: 252px;
         border: 3px solid transparent; transition: all 0.15s; }}
.card.active {{ box-shadow: 0 0 0 3px #e94560; }}
.card.g5 {{ border-color: #4caf50; }}
.card.g4 {{ border-color: #2196f3; }}
.card.g3 {{ border-color: #ff9800; }}
.card.g2 {{ border-color: #f44336; }}
.card.g1 {{ border-color: #666; }}
.card.hidden {{ display: none; }}
.card img {{ width: 240px; height: 240px; border-radius: 4px; }}
.info {{ font-size: 12px; margin-top: 4px; }}
.point-name {{ font-weight: bold; font-size: 14px; color: #e94560; }}
.detail {{ color: #888; font-size: 11px; }}
.badge {{ padding: 2px 6px; border-radius: 3px; font-size: 11px; font-weight: bold; }}
.badge-green {{ background: #4caf50; color: white; }}
.badge-yellow {{ background: #ff9800; color: white; }}
.badge-red {{ background: #f44336; color: white; }}
.buttons {{ display: flex; gap: 3px; margin-top: 6px; }}
.buttons button {{ flex: 1; padding: 5px 2px; border: none; border-radius: 4px; cursor: pointer;
                   font-weight: bold; font-size: 12px; opacity: 0.5; transition: opacity 0.15s; }}
.buttons button:hover {{ opacity: 0.9; }}
.buttons button.active {{ opacity: 1; box-shadow: 0 0 6px rgba(255,255,255,0.4); }}
.btn-1 {{ background: #666; color: white; }}
.btn-2 {{ background: #f44336; color: white; }}
.btn-3 {{ background: #ff9800; color: white; }}
.btn-4 {{ background: #2196f3; color: white; }}
.btn-5 {{ background: #4caf50; color: white; }}
#export-area {{ width: 100%; height: 200px; margin-top: 10px; font-family: monospace;
                font-size: 12px; background: #0f3460; color: #eee; border: 1px solid #444;
                border-radius: 4px; padding: 8px; }}
</style>
</head>
<body>
<h1>Detection Review: {map_name}</h1>
<div class="help">
  <kbd>1</kbd>-<kbd>5</kbd> grade current card &amp; advance |
  <kbd>J</kbd>/<kbd>K</kbd> or <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> navigate |
  <kbd>E</kbd> export |
  Grades: 1=? 2=FP 3=TP-far 4=TP-close 5=TP
</div>
<div class="controls">
    <button onclick="showAll()">All</button>
    <button onclick="showUnreviewed()">Unreviewed</button>
    <button onclick="showByGrade(5)" class="btn-5">5:TP</button>
    <button onclick="showByGrade(4)" class="btn-4">4:close</button>
    <button onclick="showByGrade(3)" class="btn-3">3:far</button>
    <button onclick="showByGrade(2)" class="btn-2">2:FP</button>
    <button onclick="showByGrade(1)" class="btn-1">1:?</button>
    <select onchange="filterConf(this.value)">
        <option value="0">All confidences</option>
        <option value="0.5">conf &ge; 0.5</option>
        <option value="0.6">conf &ge; 0.6</option>
        <option value="0.7">conf &ge; 0.7</option>
    </select>
    <button onclick="exportResults()" style="background:#7c4dff;color:white">Export JSON</button>
    <span class="stats" id="stats"></span>
</div>
<div class="grid" id="grid">
{"".join(cards_html)}
</div>
<textarea id="export-area" style="display:none"></textarea>
<script>
const grades = {{}};
let confFilter = 0;
let currentIdx = 0;
const gradeLabels = {{1:'unk', 2:'fp', 3:'tp-far', 4:'tp-close', 5:'tp'}};
const gradeClasses = {{1:'g1', 2:'g2', 3:'g3', 4:'g4', 5:'g5'}};

function allCards() {{ return [...document.querySelectorAll('.card')]; }}
function visibleCards() {{ return allCards().filter(c => !c.classList.contains('hidden')); }}

function setActive(idx) {{
    allCards().forEach(c => c.classList.remove('active'));
    const vis = visibleCards();
    if (idx < 0) idx = 0;
    if (idx >= vis.length) idx = vis.length - 1;
    currentIdx = idx;
    const card = vis[currentIdx];
    if (card) {{
        card.classList.add('active');
        card.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
    }}
}}

function grade(btn, g) {{
    const card = btn ? btn.closest('.card') : visibleCards()[currentIdx];
    if (!card) return;
    const name = card.dataset.name;

    // Toggle off if same grade
    if (grades[name] === g) {{
        delete grades[name];
        Object.values(gradeClasses).forEach(cls => card.classList.remove(cls));
        card.querySelectorAll('.buttons button').forEach(b => b.classList.remove('active'));
    }} else {{
        grades[name] = g;
        Object.values(gradeClasses).forEach(cls => card.classList.remove(cls));
        card.classList.add(gradeClasses[g]);
        card.querySelectorAll('.buttons button').forEach(b => b.classList.remove('active'));
        btn = btn || card.querySelectorAll('.buttons button')[g - 1];
        btn.classList.add('active');
    }}
    updateStats();
}}

function gradeCurrentAndAdvance(g) {{
    const vis = visibleCards();
    const card = vis[currentIdx];
    if (!card) return;
    const name = card.dataset.name;
    grades[name] = g;
    Object.values(gradeClasses).forEach(cls => card.classList.remove(cls));
    card.classList.add(gradeClasses[g]);
    card.querySelectorAll('.buttons button').forEach(b => b.classList.remove('active'));
    card.querySelectorAll('.buttons button')[g - 1].classList.add('active');
    updateStats();
    // Advance to next
    if (currentIdx < vis.length - 1) setActive(currentIdx + 1);
}}

function updateStats() {{
    const total = allCards().length;
    const counts = {{}};
    for (const g of Object.values(grades)) counts[g] = (counts[g] || 0) + 1;
    const reviewed = Object.keys(grades).length;
    document.getElementById('stats').textContent =
        `${{reviewed}}/${{total}} | ` +
        `5:TP=${{counts[5]||0}} 4:close=${{counts[4]||0}} 3:far=${{counts[3]||0}} 2:FP=${{counts[2]||0}} 1:?=${{counts[1]||0}} | ` +
        `${{total - reviewed}} left`;
}}

function showAll() {{
    allCards().forEach(c => {{
        c.classList.remove('hidden');
        if (confFilter > 0 && parseFloat(c.dataset.conf) < confFilter) c.classList.add('hidden');
    }});
    setActive(0);
}}

function showUnreviewed() {{
    allCards().forEach(c => {{
        const name = c.dataset.name;
        if (grades[name] || (confFilter > 0 && parseFloat(c.dataset.conf) < confFilter)) {{
            c.classList.add('hidden');
        }} else {{
            c.classList.remove('hidden');
        }}
    }});
    setActive(0);
}}

function showByGrade(g) {{
    allCards().forEach(c => {{
        if (grades[c.dataset.name] === g) c.classList.remove('hidden');
        else c.classList.add('hidden');
    }});
    setActive(0);
}}

function filterConf(val) {{
    confFilter = parseFloat(val);
    showAll();
}}

function exportResults() {{
    // Convert grades to labeled format
    const result = {{}};
    for (const [name, g] of Object.entries(grades)) {{
        result[name] = gradeLabels[g];
    }}
    const area = document.getElementById('export-area');
    area.style.display = 'block';
    area.value = JSON.stringify(result, null, 2);
    area.select();
}}

// Keyboard handler
document.addEventListener('keydown', (e) => {{
    // Ignore if typing in textarea
    if (e.target.tagName === 'TEXTAREA') return;

    const key = e.key;
    if (key >= '1' && key <= '5') {{
        e.preventDefault();
        gradeCurrentAndAdvance(parseInt(key));
    }} else if (key === 'j' || key === 'ArrowRight' || key === 'ArrowDown') {{
        e.preventDefault();
        setActive(currentIdx + 1);
    }} else if (key === 'k' || key === 'ArrowLeft' || key === 'ArrowUp') {{
        e.preventDefault();
        setActive(currentIdx - 1);
    }} else if (key === ' ') {{
        e.preventDefault();
        setActive(currentIdx + 1);
    }} else if (key === 'e' || key === 'E') {{
        e.preventDefault();
        exportResults();
    }}
}});

updateStats();
setActive(0);
</script>
</body>
</html>'''

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Review page: {output_path}")
    print(f"  {len(crops_data)} detections")


def review_with_tfwx(map_dir, geo_db, template_dir):
    """Generate review for a map that has a TFWX file."""
    map_dir = Path(map_dir)
    map_name = map_dir.name

    img_path = next(map_dir.glob('*.jpg'))
    tfwx_path = next(map_dir.glob('*.tfwx'))

    affine = load_tfwx(tfwx_path)
    img = load_image(str(img_path))
    h, w = img.shape[:2]

    extent = get_map_extent(affine, w, h)
    candidates = filter_points_to_extent(geo_db, extent)
    print(f"  {len(candidates)} DB candidates")

    templates = load_grayscale_templates(template_dir)
    detections = verify_candidates(img, candidates, affine, templates, color_mode='multi')

    crops = generate_crops(img, detections)
    output_path = map_dir / f'{map_name}_review.html'
    generate_html(map_name, crops, output_path)
    return output_path


def review_bootstrap(map_dir, geo_db, template_dir, grid_points, old_e_range, old_n_range):
    """Generate review for a bootstrap-georeferenced map."""
    from bootstrap_from_grid import (
        build_old_grid_affine, filter_points_by_old_grid,
        verify_candidates_old_grid,
    )

    map_dir = Path(map_dir)
    map_name = map_dir.name
    img_path = next(map_dir.glob('*.jpg'))

    old_affine = build_old_grid_affine(grid_points)
    candidates = filter_points_by_old_grid(
        geo_db, old_e_range[0], old_e_range[1], old_n_range[0], old_n_range[1])
    print(f"  {len(candidates)} DB candidates")

    img = load_image(str(img_path))
    templates = load_grayscale_templates(template_dir)
    detections = verify_candidates_old_grid(
        img, candidates, old_affine, templates, color_mode='multi')

    crops = generate_crops(img, detections)
    output_path = map_dir / f'{map_name}_review.html'
    generate_html(map_name, crops, output_path)
    return output_path


if __name__ == '__main__':
    base = Path(__file__).resolve().parent.parent
    template_dir = base / 'scripts' / 'templates'

    print("Loading geodetic database...")
    geo_db = load_geodetic_db(base / 'Control_Points' / 'nikudot_bakara_slim.csv')
    print(f"  {len(geo_db)} points")

    if len(sys.argv) < 2:
        print("Usage: python generate_review.py <map_dir> [--bootstrap]")
        sys.exit(1)

    map_dir = Path(sys.argv[1])
    is_bootstrap = '--bootstrap' in sys.argv

    if is_bootstrap:
        # M5_4598 hardcoded for now — extend later
        grid_points = [
            (891, 934, 1130000, 140000), (891, 2115, 1129000, 140000),
            (891, 3308, 1128000, 140000), (10354, 934, 1130000, 148000),
            (11537, 934, 1130000, 149000), (12721, 934, 1130000, 150000),
            (12721, 12804, 1120000, 150000), (891, 12804, 1120000, 140000),
        ]
        path = review_bootstrap(map_dir, geo_db, template_dir,
                                grid_points, (1120000, 1130000), (140000, 150000))
    else:
        path = review_with_tfwx(map_dir, geo_db, template_dir)

    print(f"\nOpen in browser: file://{path}")
