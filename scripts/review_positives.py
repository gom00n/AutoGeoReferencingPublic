#!/usr/bin/env python3
"""
Review all positive training crops to find and remove poisoned/mislabeled data.

Optimized for speed: everything starts as KEEP. Only press X on bad crops.
Space / → = keep and advance (hold to fly through)
X / ↓     = reject (not a triangle)
←         = go back
E         = export rejects

Usage:
    python review_positives.py                     # review positive/ only
    python review_positives.py --include-partial   # also review partial/
    python review_positives.py --sort confidence   # worst-confidence first

After export, run:
    python review_positives.py --apply review_rejects.json
"""

import sys
import json
import argparse
import re
from pathlib import Path
from base64 import b64encode


def extract_confidence(filename):
    """Extract CNN confidence from filename like mapname_x123_y456_c0.85.png"""
    m = re.search(r'_c([\d.]+)\.png$', filename)
    return float(m.group(1)) if m else 1.0


def load_crops(dirs, sort_by='name', name_filter=None):
    crops = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.png")):
            if name_filter and name_filter not in f.name:
                continue
            b64 = b64encode(f.read_bytes()).decode("ascii")
            crops.append({
                "name": f.stem,
                "file": f.name,
                "dir": str(f.parent.name),
                "b64": b64,
                "conf": extract_confidence(f.name),
            })

    if sort_by == 'confidence':
        crops.sort(key=lambda c: c['conf'])  # lowest confidence first
    elif sort_by == 'map':
        crops.sort(key=lambda c: c['file'])

    return crops


def generate_html(crops, output_path):
    crops_meta = json.dumps([
        {"name": c["name"], "file": c["file"], "dir": c["dir"], "conf": c["conf"]}
        for c in crops
    ])
    img_entries = ",\n".join(
        f'  "{c["name"]}": "data:image/png;base64,{c["b64"]}"'
        for c in crops
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Review Positives</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: #1a1a2e; color: #eee;
    font-family: 'SF Mono', 'Consolas', monospace;
    display: flex; flex-direction: column; align-items: center; height: 100vh;
    user-select: none;
}}
.header {{
    padding: 10px 20px; width: 100%; display: flex; justify-content: space-between;
    align-items: center; background: #16213e; border-bottom: 2px solid #0f3460;
    flex-shrink: 0;
}}
.header h1 {{ font-size: 15px; font-weight: 500; }}
.stats {{ font-size: 13px; }}
.stats .keep {{ color: #4ade80; }}
.stats .reject {{ color: #f87171; }}
.stats .pct {{ color: #aaa; }}

.main {{
    flex: 1; display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 12px; padding: 16px; min-height: 0;
}}
.crop-wrap {{
    position: relative; border: 4px solid #333; border-radius: 8px;
    overflow: hidden; background: #000;
    transition: border-color 0.1s;
}}
.crop-wrap.keep {{ border-color: #4ade80; }}
.crop-wrap.reject {{ border-color: #f87171; box-shadow: 0 0 20px rgba(248,113,113,0.4); }}
.crop-wrap img {{
    display: block; image-rendering: pixelated;
    width: 320px; height: 320px;
}}
.badge {{
    position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%);
    padding: 3px 12px; border-radius: 4px; font-size: 13px; font-weight: 700;
    pointer-events: none;
}}
.keep .badge {{ background: #166534; color: #4ade80; }}
.reject .badge {{ background: #7f1d1d; color: #f87171; }}

.meta {{
    font-size: 12px; color: #888; text-align: center; max-width: 360px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}
.conf-bar {{
    width: 320px; height: 4px; background: #333; border-radius: 2px; overflow: hidden;
}}
.conf-fill {{
    height: 100%; border-radius: 2px;
    transition: width 0.15s, background 0.15s;
}}

.counter {{ font-size: 20px; font-weight: 700; }}

.keys {{
    display: flex; gap: 16px; font-size: 12px; color: #888;
    flex-wrap: wrap; justify-content: center;
}}
.keys .key {{ display: inline-flex; align-items: center; gap: 4px; }}
.keys .key kbd {{
    background: #333; color: #fff; padding: 2px 7px;
    border-radius: 4px; font-size: 11px; border: 1px solid #555;
}}
.keys .keep-key kbd {{ background: #166534; }}
.keys .reject-key kbd {{ background: #991b1b; }}

.progress-bar {{
    width: 320px; height: 5px; background: #333;
    border-radius: 3px; overflow: hidden;
}}
.progress-fill {{
    height: 100%; background: linear-gradient(90deg, #4ade80, #22d3ee);
    transition: width 0.15s ease;
}}

.filmstrip {{
    width: 100%; padding: 6px 12px; background: #16213e;
    border-top: 2px solid #0f3460; display: flex; gap: 3px;
    overflow-x: auto; align-items: center; justify-content: center;
    min-height: 52px; flex-shrink: 0;
}}
.filmstrip .thumb {{
    width: 36px; height: 36px; border-radius: 3px; cursor: pointer;
    border: 2px solid transparent; opacity: 0.4; transition: all 0.12s;
    flex-shrink: 0; image-rendering: pixelated;
}}
.filmstrip .thumb.active {{ border-color: #fff; opacity: 1; transform: scale(1.2); }}
.filmstrip .thumb.t-keep {{ border-color: #4ade80; opacity: 0.8; }}
.filmstrip .thumb.t-reject {{ border-color: #f87171; opacity: 0.9; }}

.export-btn {{
    position: fixed; top: 10px; right: 12px; background: #0f3460;
    color: #fff; border: 1px solid #1a5276; padding: 5px 12px;
    border-radius: 6px; cursor: pointer; font-size: 12px; font-family: inherit;
}}
.export-btn:hover {{ background: #1a5276; }}
.toast {{
    position: fixed; bottom: 72px; left: 50%; transform: translateX(-50%);
    background: #111; color: #fff; padding: 8px 20px; border-radius: 6px;
    font-size: 13px; border: 1px solid #333;
    opacity: 0; transition: opacity 0.25s; pointer-events: none;
}}
.toast.show {{ opacity: 1; }}
</style>
</head>
<body>

<div class="header">
    <h1>Review Positives — press X to reject, Space to keep</h1>
    <div class="stats">
        <span class="keep" id="stat-keep">0</span> keep &nbsp;
        <span class="reject" id="stat-reject">0</span> reject &nbsp;
        <span class="pct" id="stat-pct"></span>
    </div>
</div>

<div class="main">
    <div class="counter" id="counter">1 / N</div>
    <div class="crop-wrap keep" id="crop-wrap">
        <img id="crop-img" src="" />
        <div class="badge" id="badge">KEEP</div>
    </div>
    <div class="conf-bar"><div class="conf-fill" id="conf-fill"></div></div>
    <div class="meta" id="meta"></div>
    <div class="progress-bar"><div class="progress-fill" id="prog-fill"></div></div>
    <div class="keys">
        <span class="key keep-key"><kbd>Space</kbd><kbd>→</kbd> Keep</span>
        <span class="key reject-key"><kbd>X</kbd><kbd>↓</kbd> Reject</span>
        <span class="key"><kbd>←</kbd> Back</span>
        <span class="key"><kbd>E</kbd> Export rejects</span>
    </div>
</div>

<div class="filmstrip" id="filmstrip"></div>
<button class="export-btn" onclick="exportRejects()">Export (E)</button>
<div class="toast" id="toast"></div>

<script>
const crops = {crops_meta};
const imageData = {{{img_entries}}};
const N = crops.length;
// true = keep, false = reject, null = not yet seen (treated as keep)
const decisions = new Array(N).fill(null);
let idx = 0;

// Build filmstrip
const filmstrip = document.getElementById('filmstrip');
crops.forEach((c, i) => {{
    const img = document.createElement('img');
    img.className = 'thumb';
    img.src = imageData[c.name];
    img.onclick = () => {{ idx = i; render(); }};
    img.id = 'thumb-' + i;
    filmstrip.appendChild(img);
}});

function confColor(conf) {{
    if (conf >= 0.95) return '#4ade80';
    if (conf >= 0.80) return '#86efac';
    if (conf >= 0.70) return '#fbbf24';
    return '#f87171';
}}

function render() {{
    const c = crops[idx];
    const d = decisions[idx];
    const isReject = d === false;

    document.getElementById('crop-img').src = imageData[c.name];
    document.getElementById('counter').textContent = (idx + 1) + ' / ' + N;

    const wrap = document.getElementById('crop-wrap');
    wrap.className = 'crop-wrap ' + (isReject ? 'reject' : 'keep');
    document.getElementById('badge').textContent = isReject ? 'REJECT' : 'KEEP';

    const conf = c.conf;
    const fill = document.getElementById('conf-fill');
    fill.style.width = (conf * 100) + '%';
    fill.style.background = confColor(conf);

    document.getElementById('meta').textContent =
        c.dir + '/' + c.file + (conf < 1.0 ? '  •  CNN ' + conf.toFixed(2) : '');

    // Stats
    let keeps = 0, rejects = 0;
    decisions.forEach(d => {{
        if (d === true) keeps++;
        else if (d === false) rejects++;
    }});
    document.getElementById('stat-keep').textContent = keeps;
    document.getElementById('stat-reject').textContent = rejects;
    const total = keeps + rejects;
    document.getElementById('stat-pct').textContent =
        total > 0 ? '(' + (rejects / total * 100).toFixed(1) + '% bad)' : '';

    // Progress
    const seen = decisions.filter(d => d !== null).length;
    document.getElementById('prog-fill').style.width = (seen / N * 100) + '%';

    // Filmstrip
    document.querySelectorAll('.filmstrip .thumb').forEach((t, i) => {{
        t.className = 'thumb';
        if (decisions[i] === true) t.classList.add('t-keep');
        else if (decisions[i] === false) t.classList.add('t-reject');
        if (i === idx) t.classList.add('active');
    }});
    document.getElementById('thumb-' + idx)
        ?.scrollIntoView({{ behavior: 'smooth', inline: 'center', block: 'nearest' }});
}}

function decide(keep) {{
    decisions[idx] = keep;
    if (idx < N - 1) idx++;
    render();
}}

function toast(msg) {{
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 2500);
}}

function exportRejects() {{
    const rejects = [];
    crops.forEach((c, i) => {{
        if (decisions[i] === false) rejects.push(c.file);
    }});
    const blob = new Blob([JSON.stringify(rejects, null, 2)], {{type: 'application/json'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'review_rejects.json';
    a.click();
    URL.revokeObjectURL(url);
    toast('Exported ' + rejects.length + ' rejects');
}}

document.addEventListener('keydown', e => {{
    if (e.key === 'ArrowRight' || e.key === ' ') {{
        decide(true); e.preventDefault();
    }} else if (e.key === 'ArrowDown' || e.key === 'x' || e.key === 'X') {{
        decide(false); e.preventDefault();
    }} else if (e.key === 'ArrowLeft') {{
        if (idx > 0) idx--;
        render(); e.preventDefault();
    }} else if (e.key === 'e' || e.key === 'E') {{
        exportRejects(); e.preventDefault();
    }}
}});

render();
</script>
</body>
</html>"""

    Path(output_path).write_text(html, encoding='utf-8')
    mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"Wrote {output_path} ({len(crops)} crops, {mb:.1f} MB)")


def apply_rejects(rejects_path, base_dir):
    """Move rejected files from positive/ (or partial/) to negative/."""
    rejects_path = Path(rejects_path)
    if not rejects_path.exists():
        print(f"Not found: {rejects_path}")
        return

    rejects = json.loads(rejects_path.read_text())
    base = Path(base_dir)
    pos_dir = base / 'positive'
    partial_dir = base / 'partial'
    neg_dir = base / 'negative'
    neg_dir.mkdir(exist_ok=True)

    moved = 0
    for fname in rejects:
        src = pos_dir / fname
        if not src.exists():
            src = partial_dir / fname
        if src.exists():
            dst = neg_dir / fname
            src.rename(dst)
            print(f"  moved: {fname}")
            moved += 1
        else:
            print(f"  NOT FOUND: {fname}")

    print(f"\n{moved}/{len(rejects)} files moved to negative/")
    if moved > 0:
        print("\nRun to rebuild train/val split:")
        print("  python apply_curate_labels.py --source original")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-partial', action='store_true',
                        help='Also review partial/ crops')
    parser.add_argument('--sort', default='name',
                        choices=['name', 'confidence', 'map'],
                        help='Sort order (confidence = worst first)')
    parser.add_argument('--apply', metavar='REJECTS_JSON',
                        help='Apply a review_rejects.json (move bad files to negative/)')
    parser.add_argument('--filter', default=None, metavar='SUBSTR',
                        help='Only review crops whose filename contains SUBSTR '
                             '(e.g. --filter _dbh to review just the DB-harvest batch)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output HTML path')
    args = parser.parse_args()

    base = Path(__file__).parent.parent / 'training_data'

    if args.apply:
        apply_rejects(args.apply, base)
        return

    dirs = [base / 'positive']
    if args.include_partial:
        dirs.append(base / 'partial')

    print(f"Loading crops from: {', '.join(str(d) for d in dirs)}")
    print(f"Sort order: {args.sort}" + (f", filter: {args.filter}" if args.filter else ""))

    crops = load_crops(dirs, sort_by=args.sort, name_filter=args.filter)
    if not crops:
        print("No crops found.")
        sys.exit(1)

    print(f"Loaded {len(crops)} crops")

    output = args.output or str(Path(__file__).parent.parent / 'review_positives.html')
    generate_html(crops, output)

    print(f"\nOpen in browser:  open {output}")
    print("\nControls:")
    print("  Space / →  keep (hold to fly through)")
    print("  X / ↓      reject (bad label)")
    print("  ←          go back")
    print("  E          export review_rejects.json")
    print("\nAfter export, run:")
    print(f"  python {Path(__file__).name} --apply review_rejects.json")


if __name__ == '__main__':
    main()
