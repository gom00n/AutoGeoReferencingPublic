"""
Central registry of data locations.

All map scans live under Map_Scans/. Folder names changed once already
(T1/T2/All_maps/New_maps -> Map_Scans/...) and broke every hardcoded
path — scripts must import locations from here instead of hardcoding.

Layout:
    Map_Scans/<series>-*/         ground-truth series from an archival georeferencing project:
                             per-map dirs (M<id>/) with image, world file
                             (.tfwx/.jgw) and *controlpoints*.txt / *.txt
                             ground truth. New series will be added over
                             time — anything matching <series>-* is picked up
                             automatically.
    Map_Scans/Maps_from_wiki/  downloaded sheet scans (CC-RR-Name-Year.jpg)
    Map_Scans/JPG_from_TIFF/   sheet scans converted from TIFFs
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MAP_SCANS = BASE_DIR / 'Map_Scans'
WIKI_MAPS = MAP_SCANS / 'Maps_from_wiki'
JPG_MAPS = MAP_SCANS / 'JPG_from_TIFF'
CONTROL_MAPS = BASE_DIR / 'Control_Maps'

GEODETIC_DB = BASE_DIR / 'Control_Points' / 'nikudot_bakara_slim.csv'
TEMPLATE_DIR = BASE_DIR / 'scripts' / 'templates'
TRAINING_DATA = BASE_DIR / 'training_data'
OUTPUT_DIR = BASE_DIR / 'output'

# Resolution all templates/training crops were extracted at (m per pixel,
# ~600 dpi at 1:20,000). The QA pipeline upscales toward this.
TARGET_M_PER_PX = 0.847

# Folder-name prefix identifying the ground-truth map series that ship with
# per-map world files + control-point exports. Set to whatever prefix your
# series folders use under Map_Scans/ (e.g. "301").
SERIES_PREFIX = '301'


def ground_truth_series():
    """Series folders with per-map ground truth (currently <series>-*)."""
    if not MAP_SCANS.exists():
        return []
    return sorted(d for d in MAP_SCANS.iterdir()
                  if d.is_dir() and d.name.startswith(SERIES_PREFIX))


def discover_map_dirs(name_filter=None):
    """All per-map directories (M<id>/) across the ground-truth series.

    Args:
        name_filter: optional substring to match against the dir name
    """
    dirs = []
    for series in ground_truth_series():
        for d in sorted(series.iterdir()):
            if not d.is_dir() or not d.name.startswith('M'):
                continue
            if name_filter and name_filter not in d.name:
                continue
            dirs.append(d)
    return dirs


def sheet_image_dirs():
    """Folders holding flat CC-RR-Name-Year sheet scans."""
    return [d for d in (WIKI_MAPS, JPG_MAPS) if d.exists()]
