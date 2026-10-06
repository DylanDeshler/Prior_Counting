"""Shared utilities: paths, seeding, compression (§3.1), masks (§3.3), point order (§3.2), record I/O."""

import hashlib
import io
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

IMG = 448  # every image is 448x448 RGB (§3.1)
MIN_UNIT_PX = 12  # minimum counted-unit size (§3.1)

DATA = Path(os.environ.get("COUNTERPOINT_DATA", Path(__file__).resolve().parent.parent / "data"))
SOURCES = DATA / "sources"
REPORTS = DATA / "reports"


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def item_seed(*parts) -> int:
    """Stable 63-bit seed from any parts, e.g. item_seed("e1_conflict", "die", 12, attempt)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") >> 1


def rng_for(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generator_version() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).parent, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "datagen"], capture_output=True, text=True,
                               cwd=Path(__file__).parent).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Compression randomization (§3.1): resize-and-restore, then JPEG, then PNG
# ---------------------------------------------------------------------------

def sample_compression(rng: np.random.Generator) -> dict:
    return {"downscale": round(float(rng.uniform(0.6, 1.0)), 4), "jpeg_quality": int(rng.integers(60, 96))}


def compress(img: Image.Image, params: dict) -> Image.Image:
    side = max(1, round(IMG * params["downscale"]))
    if side != IMG:
        img = img.resize((side, side), Image.BICUBIC).resize((IMG, IMG), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=params["jpeg_quality"], subsampling=2, optimize=False)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def save_png(img: Image.Image, path: Path) -> str:
    """Write deterministically; return the sha256 of the bytes."""
    data = png_bytes(img)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Masks and points
# ---------------------------------------------------------------------------

def rle_encode(mask: np.ndarray) -> dict:
    r = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": list(r["size"]), "counts": r["counts"].decode()}


def rle_decode(rle: dict) -> np.ndarray:
    return mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode()}).astype(bool)


def reading_order(points: list, unit_size: float) -> list[int]:
    """Indices of points in reading order (§3.2): rows grouped when y differs by < unit_size / 2."""
    if not points:
        return []
    idx = sorted(range(len(points)), key=lambda i: (points[i][1], points[i][0]))
    rows, current, row_y = [], [idx[0]], points[idx[0]][1]
    for i in idx[1:]:
        if abs(points[i][1] - row_y) < unit_size / 2:
            current.append(i)
        else:
            rows.append(current)
            current, row_y = [i], points[i][1]
    rows.append(current)
    return [i for row in rows for i in sorted(row, key=lambda j: points[j][0])]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def to_jsonable(x):
    """Convert numpy scalars/arrays and tuples inside params to plain JSON types."""
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


# ---------------------------------------------------------------------------
# Parameters set by measurements (§12), persisted in data/params.json
# ---------------------------------------------------------------------------

PARAMS_PATH = DATA / "params.json"
DEFAULT_PARAMS = {
    "shared_block_max": None,      # U: max(40, largest VLMBias count), then the §3.5 length rule
    "shared_block_max_source": None,
    "max_length": 1536,            # §3.5 token budget
    "qwen35_point_format": "rel1000",  # §3.2, confirmed by tools/point_format_probe.py
    "e2_matched_n": 1024,          # §5.8 N_count = N_color; 1024 until the L3 yield is known
}


def load_params() -> dict:
    p = dict(DEFAULT_PARAMS)
    if PARAMS_PATH.exists():
        p.update(json.loads(PARAMS_PATH.read_text()))
    return p


def update_params(**kw):
    p = load_params()
    p.update(kw)
    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_PATH.write_text(json.dumps(p, indent=2) + "\n")
    return p
