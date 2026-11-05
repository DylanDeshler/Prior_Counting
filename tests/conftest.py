import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # answer_parsing.py lives at the repo root

# Point the package at a throwaway data dir with a few fake background photos, before import.
_tmp = Path(tempfile.mkdtemp(prefix="counterpoint-test-"))
os.environ["COUNTERPOINT_DATA"] = str(_tmp)
(_tmp / "sources" / "openimages_bg").mkdir(parents=True)
_r = np.random.default_rng(0)
for i in range(3):
    Image.fromarray((_r.random((240, 320, 3)) * 255).astype("uint8")).save(_tmp / "sources" / "openimages_bg" / f"bg{i}.jpg")
