"""Family interface and recount helpers.

A generator ("family") is a set of pure functions over a JSON-serializable params dict:
    sample(rng, count=None)          -> params (canonical layout if count is None and the family has
                                        a familiar answer; otherwise a layout with `count` units)
    feasible_deltas(params, deltas)  -> deltas reachable from these params (§4.2: never clip)
    counterfactual(params, delta, rng) -> (params', edit dict)
    scene(params)                    -> (primitives, units) in object-local coordinates
    question(params, template)       -> question text (§3.4)
    recount(img, params, aff)        -> count measured from pixels of an analysis render (§10.2)
params always holds "count".
"""

import cv2
import numpy as np

from ..render import Affine, color_dist, local_extent

COUNT_SUFFIX = " Answer with a number in curly brackets, e.g., {9}."
COLOR_SUFFIX = " Answer with one color word in curly brackets, e.g., {red}."
DELTAS = (-3, -2, -1, 1, 2, 3)


class Family:
    name = ""
    unit = ""
    label = ""                 # list label for point JSON / list targets
    familiar = None            # familiar count (None for neutral generators)
    count_range = (1, 99)      # allowed counts (inclusive)
    distractor_exclude = ()
    templates = {"how_many": "", "count_the": ""}
    placement_kw = {}
    fixed_frame = False        # primitives already in image coordinates (no placement)

    # --- layout ---
    def sample(self, rng, count=None, **kw):
        raise NotImplementedError

    def feasible_deltas(self, params, deltas=DELTAS):
        lo, hi = self.count_range
        return [d for d in deltas if lo <= params["count"] + d <= hi and params["count"] + d >= 1]

    def counterfactual(self, params, delta, rng):
        raise NotImplementedError

    def scene(self, params):
        raise NotImplementedError

    # --- text ---
    def label_for(self, params):
        return self.label

    def question(self, params, template):
        return self.templates[template].format(**self.question_fields(params)) + COUNT_SUFFIX

    def question_fields(self, params):
        return {}

    # --- colors ---
    def main_color(self, params):
        return tuple(params["colors"]["main"])

    def colors(self, params):
        return [tuple(c) for c in params["colors"].values()]

    # --- verification ---
    def recount(self, img, params, aff):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Recount helpers: pixel analysis of a clean render (flat background, no distractors,
# rotation 0), independent of the generator's count variable.
# ---------------------------------------------------------------------------

def color_mask(img, color, tol=40):
    return np.linalg.norm(img.astype(float) - np.array(color[:3], float), axis=2) < tol


def components(mask, min_area=6):
    """[(cx, cy, x0, y0, x1, y1, area)] of connected components with area >= min_area."""
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_area:
            out.append((cents[i][0], cents[i][1], x, y, x + w, y + h, a))
    return out


def count_components(mask, min_area=6):
    return len(components(mask, min_area))


def cluster_count(values, gap):
    """Number of clusters in 1-D values where consecutive sorted values differ by > gap."""
    v = sorted(values)
    return 0 if not v else 1 + sum(1 for a, b in zip(v, v[1:]) if b - a > gap)


def merge_boxes_count(comps, gap):
    """Count groups of components whose boxes are within `gap` px of each other (e.g. digits of '12')."""
    boxes = [list(c[2:6]) for c in comps]
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
            dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
            if max(dx, dy) <= gap:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(len(boxes))})


def polygon_vertices(mask, eps_frac=0.012):
    """Vertices of the largest outer contour after polygon simplification."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    return cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True)[:, 0, :]


def convex_vertex_count(poly):
    """Number of convex vertices of a simple polygon (tips of a star / burst)."""
    p = poly.astype(float)
    area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    sign = np.sign(area)
    n = len(p)
    cnt = 0
    for i in range(n):
        a, b, c = p[i - 1], p[i], p[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if np.sign(cross) == sign:
            cnt += 1
    return cnt


def ring_crossings(mask, center, radius, samples=2048):
    """Number of separate runs of True along a circle (e.g. snowflake arms)."""
    t = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    xs = np.clip(np.round(center[0] + radius * np.cos(t)).astype(int), 0, mask.shape[1] - 1)
    ys = np.clip(np.round(center[1] + radius * np.sin(t)).astype(int), 0, mask.shape[0] - 1)
    v = mask[ys, xs].astype(int)
    if v.all():
        return 1
    return int(np.sum((v == 1) & (np.roll(v, 1) == 0)))


def object_region(img_shape, prims, aff: Affine, pad=3):
    """Slice of the image covering the object (for analysis renders, rotation 0)."""
    x0, y0, x1, y1 = local_extent(prims)
    pts = aff.apply([(x0, y0), (x1, y1)])
    xa, ya = int(max(0, pts[:, 0].min() - pad)), int(max(0, pts[:, 1].min() - pad))
    xb, yb = int(min(img_shape[1], pts[:, 0].max() + pad)), int(min(img_shape[0], pts[:, 1].max() + pad))
    return slice(ya, yb), slice(xa, xb)


def not_colors_mask(img, colors, tol=40):
    m = np.ones(img.shape[:2], bool)
    for c in colors:
        m &= ~color_mask(img, c, tol)
    return m


def pick_distinct(rng, palette, avoid, min_dist=110):
    """Choose a color from palette at least min_dist from all colors in avoid."""
    opts = [c for c in palette if all(color_dist(c, a) >= min_dist for a in avoid)]
    return tuple(opts[int(rng.integers(len(opts)))])
