"""Small version of every pool for looking at examples, in its own data root (data/preview/):

    python -m datagen [--data PATH] preview                 # -> PATH/preview (default data/preview)
    python -m datagen --data PATH/preview view              # or open PATH/preview/reports/index.html

Uses data/sources if present. Without the Open Images download it renders flat/noise backgrounds
only.
"""

import json
import os
import shutil
import subprocess
import sys

from .common import DATA, DEFAULT_PARAMS, PARAMS_PATH

ROOT = DATA / "preview"


def prepare():
    (DATA / "sources").mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    link = ROOT / "sources"
    if not link.exists():
        link.symlink_to((DATA / "sources").resolve(), target_is_directory=True)
    params = json.loads(PARAMS_PATH.read_text()) if PARAMS_PATH.exists() else dict(DEFAULT_PARAMS)
    params["shared_block_max"] = params.get("shared_block_max") or 51  # VLMBias max count; `sources` measures it
    params["e2_matched_n"] = 8
    (ROOT / "params.json").write_text(json.dumps(params, indent=2))
    for rel in ("exclusions", "reports/length_check.json"):
        src, dst = DATA / rel, ROOT / rel
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            (shutil.copytree if src.is_dir() else shutil.copy)(src, dst)
    if not (ROOT / "exclusions").exists():
        from .sources import EXCLUSIONS, VLMBIAS_EXCLUDED_TERMS  # noqa: F401
        (ROOT / "exclusions").mkdir(parents=True)
        (ROOT / "exclusions" / "vlmbias_categories.json").write_text(
            json.dumps({"topics": [], "excluded_terms": VLMBIAS_EXCLUDED_TERMS}))


def run_inside():
    """Runs in a subprocess whose data root is the preview directory."""
    from . import build as b
    b.E1_PAIRS, b.E1_VAL_IMAGES, b.SHARED_N = 72, 36, 36
    b.T0_COUNT_PAIRS = {"traffic_light": 8, "snowflake": 8}
    b.T2_COLOR_PAIRS = 8
    from . import photo_color
    photo_color.MAX_ANALYZE_PER_OBJECT = 4  # L2 sample only if the photo sources were fetched
    b.build()
    from .export import export
    export()
    from .checks import run_checks
    run_checks()
    from .report import build_report
    build_report()


def preview():
    if os.environ.get("COUNTERPOINT_PREVIEW") == "1":
        return run_inside()
    prepare()
    env = dict(os.environ, COUNTERPOINT_DATA=str(ROOT), COUNTERPOINT_PREVIEW="1")
    subprocess.run([sys.executable, "-m", "datagen", "preview"], env=env, check=True)
    print(f"\nPreview data in {ROOT}\n  browse:  uv run python -m datagen --data {ROOT} view"
          f"\n  report:  {ROOT / 'reports' / 'index.html'}  (bundle: {ROOT / 'reports.tar.gz'})")
