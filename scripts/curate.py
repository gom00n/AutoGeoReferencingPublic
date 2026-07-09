"""
Generate a keyboard-driven HTML curation page for training data crops.

Arrow keys for everything:
  Left/Right  — navigate between crops
  Up          — mark as POSITIVE (real triangle, centered)
  Down        — mark as NEGATIVE (not a triangle)
  P           — mark as PARTIAL (triangle but cut off or off-center)
  Space       — skip (leave unlabeled)

Press E to export results as JSON.

Usage:
    python curate.py                          # curate positives
    python curate.py --dir ../training_data/hard_negatives  # curate hard negatives
    python curate.py --dir ../training_data/negative --sample 100  # random sample
    python curate.py --resume labels.json     # resume from previous session
"""

import sys
import json
import random
import argparse
from pathlib import Path
from base64 import b64encode


def load_crops(crop_dir, sample=None, seed=42):
    """Load crop images from directory, optionally sampling."""
    crop_dir = Path(crop_dir)
    files = sorted(crop_dir.glob("*.png"))
    if not files:
        files = sorted(crop_dir.glob("*.jpg"))
    if not files:
        print(f"No images found in {crop_dir}")
        sys.exit(1)

    if sample and sample < len(files):
        random.seed(seed)
        files = random.sample(files, sample)
        files.sort()

    crops = []
    for f in files:
        b64 = b64encode(f.read_bytes()).decode("ascii")
        ext = f.suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        crops.append({"name": f.stem, "file": f.name, "b64": b64, "mime": mime})

    return crops


def generate_html(crops, title, output_path, resume_data=None):
    """Generate self-contained HTML curation page."""
    crops_json = json.dumps(
        [{"name": c["name"], "file": c["file"]} for c in crops]
    )
    resume_json = json.dumps(resume_data or {})

    # Build image data array separately to avoid huge JSON
    img_entries = ",\n".join(
        f'  "{c["name"]}": "data:{c["mime"]};base64,{c["b64"]}"' for c in crops
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Curate: {title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: #1a1a2e; color: #eee; font-family: 'SF Mono', 'Consolas', monospace;
    display: flex; flex-direction: column; align-items: center; height: 100vh;
    user-select: none;
}}
.header {{
    padding: 12px 20px; width: 100%; display: flex; justify-content: space-between;
    align-items: center; background: #16213e; border-bottom: 2px solid #0f3460;
}}
.header h1 {{ font-size: 16px; font-weight: 500; }}
.stats {{ font-size: 13px; color: #aaa; }}
.stats .pos {{ color: #4ade80; }}
.stats .neg {{ color: #f87171; }}
.stats .partial {{ color: #fb923c; }}
.stats .skip {{ color: #a78bfa; }}

.main {{
    flex: 1; display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 16px; padding: 20px;
}}
.crop-container {{
    position: relative; border: 3px solid #333; border-radius: 8px;
    overflow: hidden; background: #000;
}}
.crop-container img {{
    display: block; image-rendering: pixelated;
    width: 256px; height: 256px;
}}
.crop-container.labeled-pos {{ border-color: #4ade80; }}
.crop-container.labeled-neg {{ border-color: #f87171; }}
.crop-container.labeled-skip {{ border-color: #a78bfa; }}
.crop-container.labeled-partial {{ border-color: #fb923c; }}

.label-badge {{
    position: absolute; top: 8px; right: 8px; padding: 3px 10px;
    border-radius: 4px; font-size: 13px; font-weight: 700;
    display: none;
}}
.labeled-pos .label-badge {{
    display: block; background: #4ade80; color: #000;
}}
.labeled-neg .label-badge {{
    display: block; background: #f87171; color: #000;
}}
.labeled-skip .label-badge {{
    display: block; background: #a78bfa; color: #000;
}}
.labeled-partial .label-badge {{
    display: block; background: #fb923c; color: #000;
}}

.crop-name {{
    font-size: 13px; color: #888; margin-top: 4px; text-align: center;
}}
.counter {{
    font-size: 22px; font-weight: 700; color: #e2e8f0;
}}

.keys {{
    display: flex; gap: 20px; margin-top: 8px; font-size: 13px; color: #888;
}}
.keys .key {{
    display: inline-flex; align-items: center; gap: 5px;
}}
.keys .key kbd {{
    background: #333; color: #fff; padding: 2px 8px; border-radius: 4px;
    font-size: 12px; border: 1px solid #555;
}}
.keys .key.pos kbd {{ background: #166534; }}
.keys .key.neg kbd {{ background: #991b1b; }}

/* Progress bar */
.progress-bar {{
    width: 100%; max-width: 600px; height: 6px; background: #333;
    border-radius: 3px; overflow: hidden; margin-top: 4px;
}}
.progress-fill {{
    height: 100%; background: linear-gradient(90deg, #4ade80, #22d3ee);
    transition: width 0.2s ease;
}}

/* Filmstrip at bottom */
.filmstrip {{
    width: 100%; padding: 8px 16px; background: #16213e;
    border-top: 2px solid #0f3460; display: flex; gap: 4px;
    overflow-x: auto; align-items: center; justify-content: center;
    min-height: 56px;
}}
.filmstrip .thumb {{
    width: 40px; height: 40px; border-radius: 4px; cursor: pointer;
    border: 2px solid transparent; opacity: 0.5; transition: all 0.15s;
    flex-shrink: 0; image-rendering: pixelated;
}}
.filmstrip .thumb.active {{ border-color: #fff; opacity: 1; transform: scale(1.15); }}
.filmstrip .thumb.t-pos {{ border-color: #4ade80; opacity: 0.9; }}
.filmstrip .thumb.t-neg {{ border-color: #f87171; opacity: 0.9; }}
.filmstrip .thumb.t-partial {{ border-color: #fb923c; opacity: 0.9; }}
.filmstrip .thumb.t-skip {{ border-color: #a78bfa; opacity: 0.7; }}

/* Export button */
.export-btn {{
    position: fixed; top: 12px; right: 16px; background: #0f3460;
    color: #fff; border: 1px solid #1a5276; padding: 6px 14px;
    border-radius: 6px; cursor: pointer; font-size: 13px;
    font-family: inherit;
}}
.export-btn:hover {{ background: #1a5276; }}

/* Toast */
.toast {{
    position: fixed; bottom: 80px; left: 50%; transform: translateX(-50%);
    background: #333; color: #fff; padding: 8px 20px; border-radius: 6px;
    font-size: 14px; opacity: 0; transition: opacity 0.3s;
    pointer-events: none;
}}
.toast.show {{ opacity: 1; }}
</style>
</head>
<body>

<div class="header">
    <h1>Curate: {title}</h1>
    <div class="stats">
        <span class="pos" id="stat-pos">0</span> pos &nbsp;
        <span class="neg" id="stat-neg">0</span> neg &nbsp;
        <span class="partial" id="stat-partial">0</span> partial &nbsp;
        <span class="skip" id="stat-skip">0</span> skip &nbsp;
        | <span id="stat-remaining">0</span> remaining
    </div>
</div>

<div class="main">
    <div class="counter" id="counter">1 / N</div>
    <div class="crop-container" id="crop-container">
        <img id="crop-img" src="" />
        <div class="label-badge" id="label-badge"></div>
    </div>
    <div class="crop-name" id="crop-name"></div>
    <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
    <div class="keys">
        <span class="key pos"><kbd>&uarr;</kbd> Positive</span>
        <span class="key neg"><kbd>&darr;</kbd> Negative</span>
        <span class="key" style="color:#fb923c"><kbd>P</kbd> Partial</span>
        <span class="key"><kbd>&larr;</kbd><kbd>&rarr;</kbd> Navigate</span>
        <span class="key"><kbd>Space</kbd> Skip</span>
        <span class="key"><kbd>E</kbd> Export</span>
    </div>
</div>

<div class="filmstrip" id="filmstrip"></div>
<button class="export-btn" onclick="exportResults()">Export (E)</button>
<div class="toast" id="toast"></div>

<script>
const crops = {crops_json};
const imageData = {{{img_entries}}};
const labels = {resume_json};  // name -> "pos"|"neg"|"skip"
let idx = 0;
const N = crops.length;

// Build filmstrip
const filmstrip = document.getElementById('filmstrip');
crops.forEach((c, i) => {{
    const thumb = document.createElement('img');
    thumb.className = 'thumb';
    thumb.src = imageData[c.name];
    thumb.onclick = () => {{ idx = i; render(); }};
    thumb.id = 'thumb-' + i;
    filmstrip.appendChild(thumb);
}});

function render() {{
    const c = crops[idx];
    document.getElementById('crop-img').src = imageData[c.name];
    document.getElementById('crop-name').textContent = c.file;
    document.getElementById('counter').textContent = (idx + 1) + ' / ' + N;

    const container = document.getElementById('crop-container');
    const badge = document.getElementById('label-badge');
    container.className = 'crop-container';
    if (labels[c.name] === 'pos') {{
        container.classList.add('labeled-pos');
        badge.textContent = 'POSITIVE';
    }} else if (labels[c.name] === 'neg') {{
        container.classList.add('labeled-neg');
        badge.textContent = 'NEGATIVE';
    }} else if (labels[c.name] === 'partial') {{
        container.classList.add('labeled-partial');
        badge.textContent = 'PARTIAL';
    }} else if (labels[c.name] === 'skip') {{
        container.classList.add('labeled-skip');
        badge.textContent = 'SKIP';
    }}

    // Stats
    let pos = 0, neg = 0, partial = 0, skip = 0;
    for (const v of Object.values(labels)) {{
        if (v === 'pos') pos++;
        else if (v === 'neg') neg++;
        else if (v === 'partial') partial++;
        else if (v === 'skip') skip++;
    }}
    document.getElementById('stat-pos').textContent = pos;
    document.getElementById('stat-neg').textContent = neg;
    document.getElementById('stat-partial').textContent = partial;
    document.getElementById('stat-skip').textContent = skip;
    document.getElementById('stat-remaining').textContent = N - pos - neg - partial - skip;

    // Progress
    const pct = ((pos + neg + partial + skip) / N * 100).toFixed(1);
    document.getElementById('progress-fill').style.width = pct + '%';

    // Filmstrip highlighting
    document.querySelectorAll('.filmstrip .thumb').forEach((t, i) => {{
        t.className = 'thumb';
        const label = labels[crops[i].name];
        if (label === 'pos') t.classList.add('t-pos');
        else if (label === 'neg') t.classList.add('t-neg');
        else if (label === 'partial') t.classList.add('t-partial');
        else if (label === 'skip') t.classList.add('t-skip');
        if (i === idx) t.classList.add('active');
    }});

    // Scroll filmstrip to active
    const activeThumb = document.getElementById('thumb-' + idx);
    if (activeThumb) activeThumb.scrollIntoView({{ behavior: 'smooth', inline: 'center', block: 'nearest' }});
}}

function label(value) {{
    const c = crops[idx];
    labels[c.name] = value;
    // Auto-advance
    if (idx < N - 1) idx++;
    render();
}}

function toast(msg) {{
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 2000);
}}

function exportResults() {{
    // Build export with original filenames
    const out = {{}};
    for (const c of crops) {{
        if (labels[c.name]) {{
            out[c.file] = labels[c.name];
        }}
    }}
    const blob = new Blob([JSON.stringify(out, null, 2)], {{type: 'application/json'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'curate_labels.json';
    a.click();
    URL.revokeObjectURL(url);

    let pos = 0, neg = 0, partial = 0;
    for (const v of Object.values(out)) {{
        if (v === 'pos') pos++;
        else if (v === 'neg') neg++;
        else if (v === 'partial') partial++;
    }}
    toast('Exported ' + Object.keys(out).length + ' labels (' + pos + ' pos, ' + neg + ' neg, ' + partial + ' partial)');
}}

document.addEventListener('keydown', (e) => {{
    if (e.key === 'ArrowRight') {{ if (idx < N - 1) idx++; render(); e.preventDefault(); }}
    else if (e.key === 'ArrowLeft') {{ if (idx > 0) idx--; render(); e.preventDefault(); }}
    else if (e.key === 'ArrowUp') {{ label('pos'); e.preventDefault(); }}
    else if (e.key === 'ArrowDown') {{ label('neg'); e.preventDefault(); }}
    else if (e.key === 'p' || e.key === 'P') {{ label('partial'); e.preventDefault(); }}
    else if (e.key === ' ') {{ label('skip'); e.preventDefault(); }}
    else if (e.key === 'e' || e.key === 'E') {{ exportResults(); e.preventDefault(); }}
}});

// Find first unlabeled if resuming
if (Object.keys(labels).length > 0) {{
    const firstUnlabeled = crops.findIndex(c => !labels[c.name]);
    if (firstUnlabeled >= 0) idx = firstUnlabeled;
}}

render();
</script>
</body>
</html>"""

    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote {output_path} ({len(crops)} crops, {output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate keyboard-driven curation page")
    parser.add_argument("--dir", default=None,
                        help="Directory with crop images (default: training_data/positive)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Random sample N images from the directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--resume", default=None,
                        help="Resume from a previous curate_labels.json")
    parser.add_argument("-o", "--output", default=None,
                        help="Output HTML path (default: curate_<dirname>.html)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    if args.dir:
        crop_dir = Path(args.dir)
        if not crop_dir.is_absolute():
            crop_dir = Path.cwd() / crop_dir
    else:
        crop_dir = script_dir.parent / "training_data" / "positive"

    dirname = crop_dir.name
    output = args.output or str(script_dir.parent / f"curate_{dirname}.html")

    resume_data = {}
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            raw = json.loads(resume_path.read_text())
            # Convert file keys to name keys
            for k, v in raw.items():
                name = Path(k).stem
                resume_data[name] = v
            print(f"Resuming with {len(resume_data)} existing labels")

    print(f"Loading crops from {crop_dir}...")
    crops = load_crops(crop_dir, sample=args.sample, seed=args.seed)
    print(f"Loaded {len(crops)} crops")

    generate_html(crops, dirname, output, resume_data)
    print(f"\nOpen in browser:  open {output}")


if __name__ == "__main__":
    main()
