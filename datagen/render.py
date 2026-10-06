"""Scene rendering (§4.1).

Families describe an object as primitives in object-local coordinates (origin at the object
center, y down). The renderer applies one affine transform (scale, rotation, translation),
draws at 4x supersampling and downsamples, so canonical and counterfactual images that share
render parameters differ only where the object differs.

Primitives (tuples, so families stay terse):
    ("poly", pts, fill, outline, width)          closed polygon; width in local units
    ("circle", (x, y), r, fill, outline, width)
    ("line", pts, color, width)                   open polyline, round joins
    ("text", s, (x, y), size, color)              centered text, size = cap height in local units
    ("image", key, (x, y), size)                  RGBA asset (emoji), size = longest side, local units

A unit (the thing being counted) is a dict:
    {"point": (x, y), "mask": [primitives...], "size": local size, "mask_px_width": optional}
Mask primitives are rasterized at 448 px; "line" masks with "px_width" use a fixed pixel width
(e.g. a polygon side "dilated by 3 px", §4.2).
"""

import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .common import IMG, MIN_UNIT_PX, SOURCES

SS = 4  # supersampling factor


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class Affine:
    def __init__(self, cx=IMG / 2, cy=IMG / 2, scale=1.0, angle_deg=0.0):
        self.cx, self.cy, self.scale, self.angle = cx, cy, scale, angle_deg
        a = math.radians(angle_deg)
        self.c, self.s = math.cos(a), math.sin(a)

    @classmethod
    def from_params(cls, p):
        return cls(p["cx"], p["cy"], p["scale"], p["rotation_deg"])

    def apply(self, pts):
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        x, y = pts[:, 0] * self.scale, pts[:, 1] * self.scale
        return np.stack([self.cx + self.c * x - self.s * y, self.cy + self.s * x + self.c * y], axis=1)


def regular_polygon(n, r, center=(0.0, 0.0), start_deg=-90.0):
    a = np.radians(start_deg) + 2 * np.pi * np.arange(n) / n
    return np.stack([center[0] + r * np.cos(a), center[1] + r * np.sin(a)], axis=1)


def rounded_rect(x0, y0, x1, y1, r, steps=8):
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    pts = []
    for cx, cy, a0 in [(x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)]:
        for k in range(steps + 1):
            a = math.radians(a0 + 90 * k / steps)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts)


def rect(x0, y0, x1, y1):
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], dtype=float)


def thick_segment(p, q, width):
    """Rectangle polygon around segment p-q."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-12) * width / 2
    return np.array([p + n, q + n, q - n, p - n])


def prim_points(prim):
    """Points (and radius padding) bounding a primitive, for extents."""
    kind = prim[0]
    if kind == "poly":
        return np.asarray(prim[1], float), prim[4] / 2 if prim[3] else 0.0
    if kind == "circle":
        (x, y), r = prim[1], prim[2]
        return np.array([(x - r, y - r), (x + r, y + r)]), prim[5] / 2 if prim[4] else 0.0
    if kind == "line":
        return np.asarray(prim[1], float), prim[3] / 2
    if kind == "text":
        s, (x, y), size = prim[1], prim[2], prim[3]
        w = 0.62 * size * len(s)
        return np.array([(x - w / 2, y - size * 0.7), (x + w / 2, y + size * 0.7)]), 0.0
    if kind == "image":
        (x, y), size = prim[2], prim[3]
        return np.array([(x - size / 2, y - size / 2), (x + size / 2, y + size / 2)]), 0.0
    raise ValueError(kind)


def local_extent(prims):
    """(x0, y0, x1, y1) of primitives in local units."""
    xs0, ys0, xs1, ys1 = [], [], [], []
    for p in prims:
        pts, pad = prim_points(p)
        xs0.append(pts[:, 0].min() - pad), ys0.append(pts[:, 1].min() - pad)
        xs1.append(pts[:, 0].max() + pad), ys1.append(pts[:, 1].max() + pad)
    return min(xs0), min(ys0), max(xs1), max(ys1)


def corners(ext):
    x0, y0, x1, y1 = ext
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def image_bbox(ext, aff):
    pts = aff.apply(corners(ext))
    return [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]


# ---------------------------------------------------------------------------
# Placement (§4.1): scale, rotation within ±15°, position anywhere in the frame
# ---------------------------------------------------------------------------

def sample_placement(rng, extents, min_unit_local, size_range=(130, 380), margin=6, max_rotation=15.0,
                     rotation=None, min_unit_px=MIN_UNIT_PX):
    """Pick one placement that fits every layout in `extents` (e.g. both images of a pair).

    Scale is chosen so the largest object dimension lands in size_range px and the smallest
    unit is at least min_unit_px (>= MIN_UNIT_PX, §3.1).
    """
    ext = (min(e[0] for e in extents), min(e[1] for e in extents),
           max(e[2] for e in extents), max(e[3] for e in extents))
    span = max(ext[2] - ext[0], ext[3] - ext[1])
    lo = max(size_range[0] / span, max(min_unit_px, MIN_UNIT_PX) * 1.05 / min_unit_local)
    for _ in range(100):
        angle = float(rng.uniform(-max_rotation, max_rotation)) if rotation is None else rotation
        hi = size_range[1] / span
        # Shrink until the rotated union fits in the frame.
        rot = Affine(0, 0, 1.0, angle).apply(corners(ext))
        rw, rh = np.ptp(rot[:, 0]), np.ptp(rot[:, 1])
        hi = min(hi, (IMG - 2 * margin) / max(rw, rh))
        if hi < lo:
            continue
        scale = float(rng.uniform(lo, hi))
        bx0, by0 = rot[:, 0].min() * scale, rot[:, 1].min() * scale
        bx1, by1 = rot[:, 0].max() * scale, rot[:, 1].max() * scale
        cx = float(rng.uniform(margin - bx0, IMG - margin - bx1))
        cy = float(rng.uniform(margin - by0, IMG - margin - by1))
        return {"cx": round(cx, 3), "cy": round(cy, 3), "scale": round(scale, 4), "rotation_deg": round(angle, 3)}
    raise ValueError(f"object cannot satisfy min unit size: extent {ext}, min unit {min_unit_local}")


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

def luminance(c):
    r, g, b = [v / 255 for v in c[:3]]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def color_dist(a, b):
    return float(np.linalg.norm(np.subtract(a[:3], b[:3], dtype=float)))


def hsv_color(h, s, v):
    """h in [0, 360), s, v in [0, 1] -> RGB int tuple."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360, s, v)
    return (round(r * 255), round(g * 255), round(b * 255))


def random_color(rng, s=(0.5, 1.0), v=(0.45, 1.0)):
    return hsv_color(rng.uniform(0, 360), rng.uniform(*s), rng.uniform(*v))


def distinct_color(rng, avoid, min_dist=110, **kw):
    """Random color at least min_dist (RGB) from every color in avoid."""
    for _ in range(500):
        c = random_color(rng, **kw)
        if all(color_dist(c, a) >= min_dist for a in avoid):
            return c
    raise ValueError("no distinct color found")


# ---------------------------------------------------------------------------
# Backgrounds (§4.1): flat color, noise texture, or blurred Open Images crop
# ---------------------------------------------------------------------------

BG_POOL_DIR = SOURCES / "openimages_bg"


@lru_cache(maxsize=1)
def bg_pool():
    files = sorted(p.name for p in BG_POOL_DIR.glob("*.jpg")) if BG_POOL_DIR.exists() else []
    if not files:
        raise FileNotFoundError(f"No background photos in {BG_POOL_DIR}; run `python -m datagen sources` first.")
    return files


def sample_background(rng, main_color, kinds=("flat", "noise", "photo")):
    if os.environ.get("COUNTERPOINT_PREVIEW") == "1" and not BG_POOL_DIR.exists():
        kinds = tuple(k for k in kinds if k != "photo")  # previews may run before the photo download
    kind = kinds[int(rng.integers(len(kinds)))]
    target = luminance(main_color)
    if kind == "flat":
        for _ in range(500):
            c = random_color(rng, s=(0.0, 0.8), v=(0.1, 1.0))
            if abs(luminance(c) - target) >= 0.25:
                return {"kind": "flat", "color": list(c)}
    if kind == "noise":
        for _ in range(500):
            c1, c2 = random_color(rng, s=(0, .8), v=(.1, 1)), random_color(rng, s=(0, .8), v=(.1, 1))
            if min(abs(luminance(c1) - target), abs(luminance(c2) - target)) >= 0.2:
                return {"kind": "noise", "colors": [list(c1), list(c2)], "grid": int(rng.choice([4, 8, 16, 32])),
                        "seed": int(rng.integers(2**31))}
    pool = bg_pool()
    name = pool[int(rng.integers(len(pool)))]
    return {"kind": "photo", "file": name, "crop_frac": round(float(rng.uniform(0.35, 1.0)), 4),
            "crop_xy": [round(float(rng.uniform()), 4), round(float(rng.uniform()), 4)],
            "blur": round(float(rng.uniform(3, 8)), 3), "target_lum": round(target, 4)}


def make_background(spec) -> Image.Image:
    kind = spec["kind"]
    if kind == "flat":
        return Image.new("RGB", (IMG, IMG), tuple(spec["color"]))
    if kind == "noise":
        r = np.random.default_rng(spec["seed"])
        g = spec["grid"]
        field = Image.fromarray((r.random((g, g)) * 255).astype(np.uint8)).resize((IMG, IMG), Image.BICUBIC)
        t = np.asarray(field.filter(ImageFilter.GaussianBlur(IMG / g / 3)), dtype=float)[..., None] / 255
        c1, c2 = np.array(spec["colors"][0], float), np.array(spec["colors"][1], float)
        return Image.fromarray(np.clip(c1 * (1 - t) + c2 * t, 0, 255).astype(np.uint8))
    if kind == "photo":
        im = Image.open(BG_POOL_DIR / spec["file"]).convert("RGB")
        w, h = im.size
        side = max(16, int(min(w, h) * spec["crop_frac"]))
        x0 = int((w - side) * spec["crop_xy"][0])
        y0 = int((h - side) * spec["crop_xy"][1])
        im = im.crop((x0, y0, x0 + side, y0 + side)).resize((IMG, IMG), Image.BICUBIC)
        im = im.filter(ImageFilter.GaussianBlur(spec["blur"]))
        # Keep the object readable: push mean luminance away from the object's main color.
        arr = np.asarray(im, dtype=float)
        lum = (arr @ [0.2126, 0.7152, 0.0722]).mean() / 255
        if abs(lum - spec["target_lum"]) < 0.25:
            shift = 0.3 if spec["target_lum"] < 0.5 else -0.3
            arr = np.clip(arr + shift * 255, 0, 255)
            im = Image.fromarray(arr.astype(np.uint8))
        return im
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Distractors (§4.1): 0-3 shapes of a different class, outside the object bbox
# ---------------------------------------------------------------------------

DISTRACTOR_SHAPES = ["triangle", "square", "cross", "ring", "diamond", "crescent", "heart"]


def distractor_prims(d):
    """Primitives (image coords) for one distractor spec."""
    (x, y), r, shape, color, ang = d["center"], d["radius"], d["shape"], tuple(d["color"]), d["angle"]
    aff = Affine(x, y, r, ang)
    if shape == "triangle":
        return [("poly", aff.apply(regular_polygon(3, 1.0)), color, None, 0)]
    if shape == "square":
        return [("poly", aff.apply(regular_polygon(4, 1.0, start_deg=45)), color, None, 0)]
    if shape == "diamond":
        return [("poly", aff.apply([(0, -1), (0.6, 0), (0, 1), (-0.6, 0)]), color, None, 0)]
    if shape == "cross":
        return [("poly", aff.apply(rect(-1, -0.3, 1, 0.3)), color, None, 0),
                ("poly", aff.apply(rect(-0.3, -1, 0.3, 1)), color, None, 0)]
    if shape == "ring":
        return [("circle", (x, y), r * 0.8, None, color, r * 0.35)]
    if shape == "crescent":
        t = np.linspace(-np.pi / 2, np.pi / 2, 24)
        outer = np.stack([np.cos(t), np.sin(t)], 1)
        inner = np.stack([0.45 * np.cos(t[::-1]), np.sin(t[::-1])], 1)
        return [("poly", aff.apply(np.vstack([outer, inner])), color, None, 0)]
    if shape == "heart":
        t = np.linspace(0, 2 * np.pi, 48)
        pts = np.stack([16 * np.sin(t) ** 3, -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))], 1) / 17
        return [("poly", aff.apply(pts), color, None, 0)]
    raise ValueError(shape)


def sample_distractors(rng, avoid_bbox, exclude=(), avoid_colors=(), max_n=3):
    n = int(rng.integers(0, max_n + 1))
    shapes = [s for s in DISTRACTOR_SHAPES if s not in exclude]
    x0, y0, x1, y1 = avoid_bbox
    out = []
    for _ in range(n):
        for _ in range(200):
            r = float(rng.uniform(10, 26))
            x, y = float(rng.uniform(r + 2, IMG - r - 2)), float(rng.uniform(r + 2, IMG - r - 2))
            if x + r > x0 - 4 and x - r < x1 + 4 and y + r > y0 - 4 and y - r < y1 + 4:
                continue
            if any(math.hypot(x - d["center"][0], y - d["center"][1]) < r + d["radius"] + 4 for d in out):
                continue
            out.append({"shape": shapes[int(rng.integers(len(shapes)))], "center": [round(x, 2), round(y, 2)],
                        "radius": round(r, 2), "angle": round(float(rng.uniform(0, 360)), 2),
                        "color": list(distinct_color(rng, avoid_colors, min_dist=60))})
            break
    return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def font(px):
    return ImageFont.load_default(size=max(4, int(px)))


_ASSET_LOADERS = {}


def register_asset_loader(prefix, fn):
    """fn(key) -> RGBA PIL image. Used for emoji assets (prefix 'emoji:')."""
    _ASSET_LOADERS[prefix] = fn


def load_asset(key):
    prefix = key.split(":", 1)[0]
    return _ASSET_LOADERS[prefix](key)


def text_box(s, center, size, pad=0.0):
    """Local-coordinate rectangle of rendered text (same metrics as drawing), e.g. for unit masks."""
    ref = 200.0
    x0, y0, x1, y1 = font(ref / 0.72).getbbox(s)
    k = size / ref
    w, h = (x1 - x0 + 4) * k / 2 + pad, (y1 - y0 + 4) * k / 2 + pad
    return rect(center[0] - w, center[1] - h, center[0] + w, center[1] + h)


def _text_layer(s, px, color):
    f = font(px)
    x0, y0, x1, y1 = f.getbbox(s)
    layer = Image.new("RGBA", (x1 - x0 + 4, y1 - y0 + 4), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((2 - x0, 2 - y0), s, font=f, fill=tuple(color) + (255,))
    return layer


def _paste_centered(base, layer, cx, cy, angle):
    if angle:
        layer = layer.rotate(-angle, resample=Image.BICUBIC, expand=True)
    base.alpha_composite(layer, (round(cx - layer.width / 2), round(cy - layer.height / 2)))


def draw_prims(base: Image.Image, prims, aff: Affine, k: float):
    """Draw primitives on an RGBA image whose pixel = k * image px."""
    draw = ImageDraw.Draw(base)
    sc = aff.scale * k
    for p in prims:
        kind = p[0]
        if kind == "poly":
            pts = [tuple(v) for v in aff.apply(p[1]) * k]
            fill = tuple(p[2]) if p[2] is not None else None
            if p[3] is not None and p[4] > 0:
                draw.polygon(pts, fill=fill)
                draw.line(pts + [pts[0]], fill=tuple(p[3]), width=max(1, round(p[4] * sc)), joint="curve")
            else:
                draw.polygon(pts, fill=fill)
        elif kind == "circle":
            (cx, cy), = aff.apply([p[1]]) * k
            r = p[2] * sc
            w = max(1, round(p[5] * sc)) if p[4] is not None and p[5] > 0 else 0
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(p[3]) if p[3] is not None else None,
                         outline=tuple(p[4]) if w else None, width=w)
        elif kind == "line":
            pts = [tuple(v) for v in aff.apply(p[1]) * k]
            w = max(1, round(p[3] * sc))
            draw.line(pts, fill=tuple(p[2]), width=w, joint="curve")
            for x, y in (pts[0], pts[-1]):  # round caps
                draw.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=tuple(p[2]))
        elif kind == "text":
            (cx, cy), = aff.apply([p[2]]) * k
            _paste_centered(base, _text_layer(p[1], p[3] * sc / 0.72, p[4]), cx, cy, aff.angle)
            draw = ImageDraw.Draw(base)
        elif kind == "image":
            (cx, cy), = aff.apply([p[2]]) * k
            asset = load_asset(p[1])
            s = p[3] * sc / max(asset.size)
            layer = asset.resize((max(1, round(asset.width * s)), max(1, round(asset.height * s))), Image.LANCZOS)
            _paste_centered(base, layer, cx, cy, aff.angle)
            draw = ImageDraw.Draw(base)
        else:
            raise ValueError(kind)


def render_scene(background: Image.Image, layers) -> Image.Image:
    """layers: list of (prims, Affine). Drawn in order at SS and downsampled to 448."""
    big = background.convert("RGBA").resize((IMG * SS, IMG * SS), Image.BICUBIC)
    for prims, aff in layers:
        draw_prims(big, prims, aff, SS)
    # Box filter = area averaging: anti-aliased, and a change never spreads beyond its own pixels.
    return big.resize((IMG, IMG), Image.BOX).convert("RGB")


def rasterize_mask(prims, aff: Affine, px_width=None) -> np.ndarray:
    """Binary 448x448 mask of primitives (filled), at 2x then downsampled with a 50% threshold."""
    k = 2
    m = Image.new("L", (IMG * k, IMG * k), 0)
    d = ImageDraw.Draw(m)
    sc = aff.scale * k
    for p in prims:
        kind = p[0]
        if kind == "poly":
            d.polygon([tuple(v) for v in aff.apply(p[1]) * k], fill=255)
        elif kind == "circle":
            (cx, cy), = aff.apply([p[1]]) * k
            r = p[2] * sc + (p[5] * sc / 2 if p[4] is not None and p[5] else 0)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        elif kind == "line":
            pts = [tuple(v) for v in aff.apply(p[1]) * k]
            w = round(px_width * k) if px_width else max(1, round(p[3] * sc))
            d.line(pts, fill=255, width=w, joint="curve")
            for x, y in (pts[0], pts[-1]):
                d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=255)
        elif kind in ("text", "image"):
            layer = Image.new("RGBA", m.size, (0, 0, 0, 0))
            draw_prims(layer, [p if kind == "image" else p[:4] + ((255, 255, 255),)], aff, k)
            alpha = layer.getchannel("A").point(lambda v: 255 if v > 127 else 0)
            m.paste(255, mask=alpha)
        else:
            raise ValueError(kind)
    arr = np.asarray(m, dtype=np.uint16).reshape(IMG, k, IMG, k).mean(axis=(1, 3))
    return arr >= 128
