"""Pip families (§4.2): die face, playing card, domino; and their neutral twins."""

import itertools

import numpy as np

from ..render import rect, regular_polygon, rounded_rect, thick_segment
from .base import (Family, color_mask, count_components, object_region, pick_distinct)

WHITES = [(250, 250, 248), (245, 240, 225), (235, 235, 240), (255, 252, 240)]
DARKS = [(20, 20, 20), (30, 30, 60), (60, 20, 20), (15, 50, 30)]


def _pick_pair(rng, faces, pips, min_dist=130):
    face = tuple(faces[int(rng.integers(len(faces)))])
    pip = pick_distinct(rng, pips, [face], min_dist)
    return face, pip


def _random_face_colors(rng):
    """Mostly classic (white face, dark pips), sometimes colored."""
    from ..render import distinct_color, random_color
    if rng.random() < 0.6:
        face, pip = _pick_pair(rng, WHITES, DARKS)
    else:
        face = random_color(rng, s=(0.3, 0.9), v=(0.5, 1.0))
        pip = distinct_color(rng, [face], 140, s=(0.0, 1.0), v=(0.0, 1.0))
    outline = pick_distinct(rng, [(90, 90, 90), (40, 40, 40), (150, 150, 150), (200, 200, 200)], [pip, face], 70)
    return {"main": list(face), "pip": list(pip), "outline": list(outline)}


def scatter(rng, n, box, r, gap, tries=4000):
    """n non-overlapping disc centers of radius r inside box (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    pts = []
    for _ in range(tries):
        if len(pts) == n:
            return pts
        p = (float(rng.uniform(x0 + r, x1 - r)), float(rng.uniform(y0 + r, y1 - r)))
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= (2 * r + gap) ** 2 for q in pts):
            pts.append(p)
    return pts if len(pts) == n else None


def grid_edit(layout, positions, delta, rng, standard, lo, hi):
    """Add/remove `delta` grid positions; result must not be a standard layout (§4.2)."""
    cur = set(map(tuple, layout)) if layout and isinstance(layout[0], (list, tuple)) else set(layout)
    free = [p for p in positions if p not in cur]
    if delta > 0:
        options = [cur | set(c) for c in itertools.combinations(free, delta)]
    else:
        options = [cur - set(c) for c in itertools.combinations(sorted(cur), -delta)]
    options = [o for o in options if lo <= len(o) <= hi and frozenset(o) not in standard]
    if not options:
        return None
    return sorted(options[int(rng.integers(len(options)))])


# ---------------------------------------------------------------------------
# Die face
# ---------------------------------------------------------------------------

DIE_STD = {1: [[4]], 2: [[0, 8], [2, 6]], 3: [[0, 4, 8], [2, 4, 6]], 4: [[0, 2, 6, 8]], 5: [[0, 2, 4, 6, 8]],
           6: [[0, 3, 6, 2, 5, 8], [0, 1, 2, 6, 7, 8]]}
DIE_STD_SETS = {frozenset(v) for vs in DIE_STD.values() for v in vs}
DIE_STD_SETS_WITH_BLANK = DIE_STD_SETS | {frozenset()}


def pip_grid_pos(i, spacing, center=(0.0, 0.0)):
    r, c = divmod(i, 3)
    return (center[0] + (c - 1) * spacing, center[1] + (r - 1) * spacing)


def die_grid_deltas(fam, params, deltas):
    return [d for d in Family.feasible_deltas(fam, params, deltas)
            if grid_edit(params["layout"], range(9), d, np.random.default_rng(0), DIE_STD_SETS, *fam.count_range)]


class Die(Family):
    name, unit, label, familiar = "die", "pip", "pip", None  # familiar = canonical face value, per item
    count_range = (1, 9)
    distractor_exclude = ("ring",)
    templates = {"how_many": "How many pips are on this die?", "count_the": "Count the pips on this die."}
    PIP_R, SPACING = 0.085, 0.28

    def sample(self, rng, count=None):
        k = int(rng.integers(1, 7))
        variants = DIE_STD[k]
        return {"count": k, "layout": sorted(variants[int(rng.integers(len(variants)))]),
                "corner": round(float(rng.uniform(0.08, 0.2)), 3), "colors": _random_face_colors(rng)}

    def feasible_deltas(self, params, deltas=(-3, -2, -1, 1, 2, 3)):
        return die_grid_deltas(self, params, deltas)

    def counterfactual(self, params, delta, rng):
        new = grid_edit(params["layout"], range(9), delta, rng, DIE_STD_SETS, *self.count_range)
        changed = sorted(set(new) ^ set(params["layout"]))
        return ({**params, "count": len(new), "layout": new},
                {"op": "add_pips" if delta > 0 else "remove_pips", "params": {"grid_positions": changed}})

    def scene(self, p):
        col = p["colors"]
        prims = [("poly", rounded_rect(-0.5, -0.5, 0.5, 0.5, p["corner"]), col["main"], col["outline"], 0.03)]
        units = []
        for i in p["layout"]:
            c = pip_grid_pos(i, self.SPACING)
            prims.append(("circle", c, self.PIP_R, col["pip"], None, 0))
            units.append({"point": c, "mask": [("circle", c, self.PIP_R, 1, None, 0)], "size": 2 * self.PIP_R})
        return prims, units

    def recount(self, img, p, aff):
        prims, _ = self.scene(p)
        ys, xs = object_region(img.shape, prims, aff)
        return count_components(color_mask(img[ys, xs], p["colors"]["pip"]))


class DotsInSquare(Family):
    """Neutral twin of the die: dots at random positions inside a plain rounded square."""
    name, unit, label = "twin_dots_square", "dot", "dot"
    count_range = (1, 9)
    distractor_exclude = ("ring",)
    templates = {"how_many": "How many dots are in the square?", "count_the": "Count the dots in the square."}

    def sample(self, rng, count=None):
        for _ in range(100):
            r = float(rng.uniform(0.05, 0.085))
            pts = scatter(rng, count, (-0.4, -0.4, 0.4, 0.4), r, 0.035)
            if pts:
                return {"count": count, "dots": [[round(x, 4), round(y, 4)] for x, y in pts], "r": round(r, 4),
                        "corner": round(float(rng.uniform(0.03, 0.2)), 3), "colors": _random_face_colors(rng)}
        raise ValueError("cannot place dots")

    def scene(self, p):
        col = p["colors"]
        prims = [("poly", rounded_rect(-0.5, -0.5, 0.5, 0.5, p["corner"]), col["main"], col["outline"], 0.03)]
        units = []
        for c in p["dots"]:
            prims.append(("circle", tuple(c), p["r"], col["pip"], None, 0))
            units.append({"point": tuple(c), "mask": [("circle", tuple(c), p["r"], 1, None, 0)], "size": 2 * p["r"]})
        return prims, units

    recount = Die.recount


# ---------------------------------------------------------------------------
# Playing card: 3 columns x 7 rows pip grid, corner indices removed
# ---------------------------------------------------------------------------

L, C, R = 0, 1, 2
CARD_STD = {
    1: [(C, 3)],
    2: [(C, 0), (C, 6)],
    3: [(C, 0), (C, 3), (C, 6)],
    4: [(L, 0), (R, 0), (L, 6), (R, 6)],
    5: [(L, 0), (R, 0), (L, 6), (R, 6), (C, 3)],
    6: [(L, 0), (R, 0), (L, 3), (R, 3), (L, 6), (R, 6)],
    7: [(L, 0), (R, 0), (L, 3), (R, 3), (L, 6), (R, 6), (C, 1)],
    8: [(L, 0), (R, 0), (L, 3), (R, 3), (L, 6), (R, 6), (C, 1), (C, 5)],
    9: [(L, 0), (L, 2), (L, 4), (L, 6), (R, 0), (R, 2), (R, 4), (R, 6), (C, 3)],
    10: [(L, 0), (L, 2), (L, 4), (L, 6), (R, 0), (R, 2), (R, 4), (R, 6), (C, 1), (C, 5)],
}
CARD_STD_SETS = {frozenset(v) for v in CARD_STD.values()}
CARD_GRID = [(c, r) for r in range(7) for c in range(3)]
CARD_W, CARD_H = 0.71, 1.0
SUIT_COLORS = {"hearts": (200, 25, 40), "diamonds": (200, 25, 40), "spades": (25, 25, 25), "clubs": (25, 25, 25)}


def heart_pts(n=40):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([16 * np.sin(t) ** 3, -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))], 1)
    return (pts - [0, 1.5]) / 16.5


def suit_prims(suit, center, size, color, flip):
    """Primitives for one pip; size = pip height (local units)."""
    s = size / 2
    sign = -1 if flip else 1

    def tf(pts):
        pts = np.asarray(pts, float)
        return np.stack([center[0] + pts[:, 0] * s, center[1] + sign * pts[:, 1] * s], 1)

    if suit == "diamonds":
        return [("poly", tf([(0, -1), (0.72, 0), (0, 1), (-0.72, 0)]), color, None, 0)]
    if suit == "hearts":
        return [("poly", tf(heart_pts()), color, None, 0)]
    stem = ("poly", tf([(0, 0.2), (0.3, 1.0), (-0.3, 1.0)]), color, None, 0)
    if suit == "spades":
        h = heart_pts() * [1, -1] + [0, -0.1]
        return [("poly", tf(h * 0.95), color, None, 0), stem]
    # clubs: three overlapping lobes + stem
    lobes = [(0, -0.45), (-0.47, 0.18), (0.47, 0.18)]
    out = [("circle", tuple(tf([c])[0]), 0.44 * s, color, None, 0) for c in lobes]
    out.append(("poly", tf([(-0.2, -0.1), (0.2, -0.1), (0.2, 0.3), (-0.2, 0.3)]), color, None, 0))
    return out + [stem]


class PlayingCard(Family):
    name, unit, label = "playing_card", "pip", "pip"
    count_range = (1, 13)
    distractor_exclude = ("heart", "diamond")
    templates = {"how_many": "How many pips are on this playing card?",
                 "count_the": "Count the pips on this playing card."}
    PIP_H = 0.12  # ~12% of card height (real cards ~15%); rows 0.135 apart so pips never touch
    # Dense layouts: keep cells/pips countable (the model failed to count small, dense instances).
    placement_kw = {"size_range": (220, 380), "min_unit_px": 20}

    def sample(self, rng, count=None):
        rank = int(rng.integers(2, 11))
        suit = ["hearts", "diamonds", "spades", "clubs"][int(rng.integers(4))]
        face = tuple(WHITES[int(rng.integers(len(WHITES)))])
        return {"count": rank, "layout": sorted(map(list, CARD_STD[rank])), "suit": suit,
                "corner": round(float(rng.uniform(0.03, 0.07)),3),
                "colors": {"main": list(face), "pip": list(SUIT_COLORS[suit]), "outline": [120, 120, 120]}}

    def feasible_deltas(self, params, deltas=(-3, -2, -1, 1, 2, 3)):
        return [d for d in super().feasible_deltas(params, deltas)
                if grid_edit(params["layout"], CARD_GRID, d, np.random.default_rng(0), CARD_STD_SETS, 1, 13)]

    def counterfactual(self, params, delta, rng):
        new = grid_edit(params["layout"], CARD_GRID, delta, rng, CARD_STD_SETS, *self.count_range)
        old = set(map(tuple, params["layout"]))
        changed = sorted(set(new) ^ old)
        return ({**params, "count": len(new), "layout": [list(p) for p in new]},
                {"op": "add_pips" if delta > 0 else "remove_pips", "params": {"grid_positions": [list(p) for p in changed]}})

    @staticmethod
    def grid_xy(c, r):
        return ((c - 1) * 0.205, -0.405 + 0.135 * r)

    def scene(self, p):
        col = p["colors"]
        prims = [("poly", rounded_rect(-CARD_W / 2, -CARD_H / 2, CARD_W / 2, CARD_H / 2, p["corner"]),
                  col["main"], col["outline"], 0.012)]
        units = []
        for c, r in p["layout"]:
            xy = self.grid_xy(c, r)
            pp = suit_prims(p["suit"], xy, self.PIP_H, col["pip"], flip=r > 3)
            prims += pp
            units.append({"point": xy, "mask": pp, "size": self.PIP_H * 0.8})
        return prims, units

    def recount(self, img, p, aff):
        prims, _ = self.scene(p)
        ys, xs = object_region(img.shape, prims, aff)
        return count_components(color_mask(img[ys, xs], p["colors"]["pip"]))


GLYPHS = ["circle", "triangle", "square", "plus"]


def glyph_prims(kind, c, s, color):
    x, y = c
    if kind == "circle":
        return [("circle", (x, y), s / 2, color, None, 0)]
    if kind == "triangle":
        return [("poly", regular_polygon(3, s * 0.58, (x, y + s * 0.08)), color, None, 0)]
    if kind == "square":
        return [("poly", rect(x - s * 0.42, y - s * 0.42, x + s * 0.42, y + s * 0.42), color, None, 0)]
    return [("poly", rect(x - s / 2, y - s * 0.14, x + s / 2, y + s * 0.14), color, None, 0),
            ("poly", rect(x - s * 0.14, y - s / 2, x + s * 0.14, y + s / 2), color, None, 0)]


class GlyphsOnCard(Family):
    """Neutral twin of the playing card: suit-free glyphs scattered on a white card."""
    name, unit, label = "twin_glyph_card", "mark", "mark"
    count_range = (1, 13)
    distractor_exclude = ("heart", "diamond")
    templates = {"how_many": "How many marks are on the card?", "count_the": "Count the marks on the card."}
    SIZE = 0.12  # same glyph size as the card's pips, so the twin matches its conflict family
    # Dense layouts: keep cells/pips countable (the model failed to count small, dense instances).
    placement_kw = {"size_range": (220, 380), "min_unit_px": 20}

    def sample(self, rng, count=None):
        pts = None
        while pts is None:
            pts = scatter(rng, count, (-0.31, -0.45, 0.31, 0.45), self.SIZE / 2 * 1.05, 0.025)
        from ..render import random_color
        face = tuple(WHITES[int(rng.integers(len(WHITES)))])
        glyph_color = random_color(rng, s=(0.4, 1.0), v=(0.1, 0.6))
        return {"count": count, "glyph": GLYPHS[int(rng.integers(len(GLYPHS)))],
                "marks": [[round(x, 4), round(y, 4)] for x, y in pts], "corner": round(float(rng.uniform(0.03, 0.07)), 3),
                "colors": {"main": list(face), "pip": list(glyph_color), "outline": [120, 120, 120]}}

    def scene(self, p):
        col = p["colors"]
        prims = [("poly", rounded_rect(-CARD_W / 2, -CARD_H / 2, CARD_W / 2, CARD_H / 2, p["corner"]),
                  col["main"], col["outline"], 0.012)]
        units = []
        for c in p["marks"]:
            g = glyph_prims(p["glyph"], tuple(c), self.SIZE, col["pip"])
            prims += g
            units.append({"point": tuple(c), "mask": g, "size": self.SIZE * 0.84})
        return prims, units

    recount = PlayingCard.recount


# ---------------------------------------------------------------------------
# Domino: count the pips on the named half
# ---------------------------------------------------------------------------

class Domino(Family):
    name, unit, label = "domino", "pip", "pip"
    count_range = (1, 9)
    distractor_exclude = ("ring",)
    templates = {"how_many": "How many pips are on the {side} half of this domino?",
                 "count_the": "Count the pips on the {side} half of this domino."}
    PIP_R, SPACING = 0.042, 0.13

    def question_fields(self, p):
        return {"side": "left" if p["orientation"] == "h" else "top"}

    def sample(self, rng, count=None):
        k = int(rng.integers(1, 7))
        other_k = int(rng.integers(0, 7))
        pick = lambda vs: sorted(vs[int(rng.integers(len(vs)))])
        return {"count": k, "orientation": "h" if rng.random() < 0.5 else "v",
                "layout": pick(DIE_STD[k]), "other": pick(DIE_STD[other_k]) if other_k else [],
                "corner": round(float(rng.uniform(0.03, 0.08)), 3), "colors": _random_face_colors(rng)}

    def feasible_deltas(self, params, deltas=(-3, -2, -1, 1, 2, 3)):
        return die_grid_deltas(self, params, deltas)

    def counterfactual(self, params, delta, rng):
        new = grid_edit(params["layout"], range(9), delta, rng, DIE_STD_SETS, *self.count_range)
        changed = sorted(set(new) ^ set(params["layout"]))
        return ({**params, "count": len(new), "layout": new},
                {"op": "add_pips" if delta > 0 else "remove_pips", "params": {"grid_positions": changed, "half": "named"}})

    @staticmethod
    def geometry(p):
        if p["orientation"] == "h":
            return (-0.5, -0.25, 0.5, 0.25), (-0.25, 0.0), (0.25, 0.0), ((0, -0.22), (0, 0.22))
        return (-0.25, -0.5, 0.25, 0.5), (0.0, -0.25), (0.0, 0.25), ((-0.22, 0), (0.22, 0))

    def scene(self, p):
        col = p["colors"]
        tile, named, other, divider = self.geometry(p)
        prims = [("poly", rounded_rect(*tile, p["corner"]), col["main"], col["outline"], 0.015),
                 ("poly", thick_segment(*divider, 0.012), col["outline"], None, 0)]
        units = []
        for i in p["other"]:
            prims.append(("circle", pip_grid_pos(i, self.SPACING, other), self.PIP_R, col["pip"], None, 0))
        for i in p["layout"]:
            c = pip_grid_pos(i, self.SPACING, named)
            prims.append(("circle", c, self.PIP_R, col["pip"], None, 0))
            units.append({"point": c, "mask": [("circle", c, self.PIP_R, 1, None, 0)], "size": 2 * self.PIP_R})
        return prims, units

    def recount(self, img, p, aff):
        _, named, _, _ = self.geometry(p)
        half = [("poly", rect(named[0] - 0.21, named[1] - 0.21, named[0] + 0.21, named[1] + 0.21), 1, None, 0)]
        ys, xs = object_region(img.shape, half, aff, pad=0)
        return count_components(color_mask(img[ys, xs], p["colors"]["pip"]))


class DotsInSplitRect(Family):
    """Neutral twin of the domino: dots in a rectangle split into halves; the question names one."""
    name, unit, label = "twin_split_rect", "dot", "dot"
    count_range = (1, 9)
    distractor_exclude = ("ring",)
    templates = {"how_many": "How many dots are in the {side} half of the rectangle?",
                 "count_the": "Count the dots in the {side} half of the rectangle."}

    question_fields = Domino.question_fields
    geometry = staticmethod(Domino.geometry)

    def sample(self, rng, count=None):
        orientation = "h" if rng.random() < 0.5 else "v"
        for _ in range(100):
            r = float(rng.uniform(0.032, 0.045))
            box = lambda c: (c[0] - 0.21, c[1] - 0.21, c[0] + 0.21, c[1] + 0.21)
            _, named, other, _ = Domino.geometry({"orientation": orientation})
            a = scatter(rng, count, box(named), r, 0.025)
            b = scatter(rng, int(rng.integers(0, 10)), box(other), r, 0.025)
            if a is not None and b is not None:
                return {"count": count, "orientation": orientation, "r": round(r, 4),
                        "dots": [[round(x, 4), round(y, 4)] for x, y in a],
                        "other_dots": [[round(x, 4), round(y, 4)] for x, y in b],
                        "corner": round(float(rng.uniform(0.0, 0.05)), 3), "colors": _random_face_colors(rng)}
        raise ValueError("cannot place dots")

    def scene(self, p):
        col = p["colors"]
        tile, _, _, divider = self.geometry(p)
        prims = [("poly", rounded_rect(*tile, p["corner"]), col["main"], col["outline"], 0.015),
                 ("poly", thick_segment(*divider, 0.012), col["outline"], None, 0)]
        for c in p["other_dots"]:
            prims.append(("circle", tuple(c), p["r"], col["pip"], None, 0))
        units = []
        for c in p["dots"]:
            prims.append(("circle", tuple(c), p["r"], col["pip"], None, 0))
            units.append({"point": tuple(c), "mask": [("circle", tuple(c), p["r"], 1, None, 0)], "size": 2 * p["r"]})
        return prims, units

    recount = Domino.recount
