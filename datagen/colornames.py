"""11-term color naming (van de Weijer et al. 2009, "Learning Color Names for Real-World Applications").

The w2c table maps a 32x32x32 sRGB grid to probabilities over the 11 basic terms, in alphabetical
order (the same order as CoDa). Index = r//8 + 32*(g//8) + 1024*(b//8).
"""

import io
import tarfile
from functools import lru_cache

import numpy as np
import requests

from .common import SOURCES

TERMS = ["black", "blue", "brown", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow"]
CHROMATIC = ["blue", "brown", "green", "orange", "pink", "purple", "red", "yellow"]
W2C_URL = "http://lear.inrialpes.fr/people/vandeweijer/code/ColorNaming.tar"
W2C_PATH = SOURCES / "w2c.npy"


def fetch():
    if W2C_PATH.exists():
        return
    data = requests.get(W2C_URL, timeout=120).content
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        txt = tar.extractfile("ColorNaming/w2c.txt").read().decode()
    table = np.loadtxt(io.StringIO(txt), dtype=np.float64)
    assert table.shape == (32768, 14), table.shape
    rgb = table[:, :3]
    # Verify the bin order (R fastest) before trusting the index formula.
    idx = (rgb[:, 0] // 8 + 32 * (rgb[:, 1] // 8) + 1024 * (rgb[:, 2] // 8)).astype(int)
    assert (idx == np.arange(32768)).all()
    W2C_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(W2C_PATH, table[:, 3:].astype(np.float32))


@lru_cache(maxsize=1)
def w2c():
    return np.load(W2C_PATH)


def name_indices(rgb):
    """rgb: (..., 3) uint8/int array -> (...) term indices."""
    rgb = np.asarray(rgb).astype(np.int64)
    idx = rgb[..., 0] // 8 + 32 * (rgb[..., 1] // 8) + 1024 * (rgb[..., 2] // 8)
    return np.argmax(w2c()[idx], axis=-1)


def name_pixels(rgb):
    """Per-pixel term names as an array of strings."""
    return np.array(TERMS)[name_indices(rgb)]


def majority_name(rgb, mask=None):
    """Most common term over the pixels in mask (or all pixels)."""
    px = np.asarray(rgb)[mask] if mask is not None else np.asarray(rgb).reshape(-1, 3)
    if len(px) == 0:
        return None
    counts = np.bincount(name_indices(px), minlength=len(TERMS))
    return TERMS[int(np.argmax(counts))]


def name_fractions(rgb, mask=None):
    px = np.asarray(rgb)[mask] if mask is not None else np.asarray(rgb).reshape(-1, 3)
    counts = np.bincount(name_indices(px), minlength=len(TERMS))
    return {t: float(c) / max(1, len(px)) for t, c in zip(TERMS, counts)}


# ---------------------------------------------------------------------------
# sRGB (D65) <-> CIELAB
# ---------------------------------------------------------------------------

_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
_MINV = np.linalg.inv(_M)
_WHITE = np.array([0.95047, 1.0, 1.08883])


def srgb_to_lab(rgb):
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.T / _WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def lab_to_srgb(lab, clip=True):
    """Returns float sRGB in [0, 255] (clipped to gamut if clip)."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], axis=-1)
    xyz = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * _WHITE
    lin = xyz @ _MINV.T
    c = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.abs(lin) ** (1 / 2.4) * np.sign(lin) - 0.055)
    out = c * 255
    return np.clip(out, 0, 255) if clip else out
