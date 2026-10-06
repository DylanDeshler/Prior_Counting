"""Noto Emoji assets (Apache-2.0 images, googlefonts/noto-emoji 2D/png/512) and CIELAB recoloring.

Asset keys (registered with render.register_asset_loader):
    emoji:<cps>                                   original RGBA
    emoji:<cps>|<term>|<a>,<b>|<k>|<feather>      recolor pixels named <term> to ab=(a, b), L fixed,
                                                  keeping k * (pixel ab - mean ab) texture; feathered edge
    emoji:<cps>|<term>|id|<k>|<feather>           identity recolor (null edit, §5.7): same mask and
                                                  feathering, ab unchanged, so only LAB round-trip traces
    emoji:<cps>|<term>|mask                       hard recolor mask as alpha (for verification)
<cps> is the lowercase hex codepoint sequence joined by '_' with FE0F removed (Noto file naming).
"""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import requests
from PIL import Image
from scipy.ndimage import gaussian_filter

from .colornames import lab_to_srgb, name_indices, srgb_to_lab, TERMS
from .common import SOURCES
from .render import register_asset_loader

NOTO_URL = "https://raw.githubusercontent.com/googlefonts/noto-emoji/main/2D/png/512/emoji_u{cps}.png"
EMOJI_TEST_URL = "https://unicode.org/Public/emoji/16.0/emoji-test.txt"
EMOJI_DIR = SOURCES / "noto_emoji"
EMOJI_TEST = SOURCES / "emoji-test.txt"


def cps_of(codepoints):
    return "_".join(c.lower() for c in codepoints if c.lower() != "fe0f")


def fetch_index():
    if not EMOJI_TEST.exists():
        EMOJI_TEST.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(EMOJI_TEST_URL, timeout=60)
        r.raise_for_status()
        EMOJI_TEST.write_text(r.text)


@lru_cache(maxsize=1)
def emoji_index():
    """name (lowercase CLDR short name) -> {"cps", "group", "subgroup"} for fully-qualified emoji."""
    out, group, sub = {}, None, None
    for line in EMOJI_TEST.read_text().splitlines():
        if line.startswith("# group:"):
            group = line.split(":", 1)[1].strip()
        elif line.startswith("# subgroup:"):
            sub = line.split(":", 1)[1].strip()
        elif "; fully-qualified" in line and not line.startswith("#"):
            cps, rest = line.split(";", 1)
            name = rest.split("#", 1)[1].strip().split(" ", 2)[2].lower()
            out.setdefault(name, {"cps": cps_of(cps.split()), "group": group, "subgroup": sub})
    return out


def fetch_assets(cps_list):
    """Download the needed Noto PNGs; returns the set of cps that exist."""
    EMOJI_DIR.mkdir(parents=True, exist_ok=True)

    def get(cps):
        path = EMOJI_DIR / f"emoji_u{cps}.png"
        if path.exists():
            return cps
        r = requests.get(NOTO_URL.format(cps=cps), timeout=60)
        if r.status_code != 200:
            return None
        path.write_bytes(r.content)
        return cps

    from tqdm import tqdm
    todo = sorted(set(cps_list))
    with ThreadPoolExecutor(16) as ex:
        return {c for c in tqdm(ex.map(get, todo), total=len(todo), desc="Noto Emoji assets", unit="img") if c}


@lru_cache(maxsize=512)
def base_rgba(cps) -> np.ndarray:
    return np.asarray(Image.open(EMOJI_DIR / f"emoji_u{cps}.png").convert("RGBA"))


def term_mask(rgba, term):
    return (np.array(TERMS)[name_indices(rgba[..., :3])] == term) & (rgba[..., 3] > 127)


def recolor(rgba, term, ab, k, feather):
    """ab=None -> identity recolor (null edit)."""
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    mask = term_mask(rgba, term)
    soft = gaussian_filter(mask.astype(np.float64), feather) * (alpha > 0)
    lab = srgb_to_lab(rgb)
    if ab is None:
        new_ab = lab[..., 1:]
    else:
        mean = lab[mask][:, 1:].mean(axis=0)
        new_ab = np.asarray(ab, float) + k * (lab[..., 1:] - mean)
    out_ab = lab[..., 1:] * (1 - soft[..., None]) + new_ab * soft[..., None]
    out = np.round(lab_to_srgb(np.concatenate([lab[..., :1], out_ab], axis=-1))).astype(np.uint8)
    out = np.where((alpha > 0)[..., None], out, rgb)
    return np.dstack([out, alpha])


def recolor_key(cps, term, ab, k=0.5, feather=2.0):
    a = "id" if ab is None else f"{ab[0]:.1f},{ab[1]:.1f}"
    return f"emoji:{cps}|{term}|{a}|{k}|{feather}"


def mask_key(cps, term):
    return f"emoji:{cps}|{term}|mask"


@lru_cache(maxsize=1024)
def load(key) -> Image.Image:
    parts = key.split(":", 1)[1].split("|")
    rgba = base_rgba(parts[0])
    if len(parts) == 1:
        return Image.fromarray(rgba, "RGBA")
    term = parts[1]
    if parts[2] == "mask":
        m = term_mask(rgba, term)
        return Image.fromarray(np.dstack([np.full(m.shape + (3,), 255, np.uint8), (m * 255).astype(np.uint8)]), "RGBA")
    ab = None if parts[2] == "id" else tuple(float(v) for v in parts[2].split(","))
    return Image.fromarray(recolor(rgba, term, ab, float(parts[3]), float(parts[4])), "RGBA")


register_asset_loader("emoji", load)
