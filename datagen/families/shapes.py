"""Shared range block (§4.3): one shape type per image, all instances counted, no distractors."""

import math

from ..render import random_color, regular_polygon, rect
from .base import Family, color_mask, count_components

SHAPES = {"circle": "circles", "square": "squares", "triangle": "triangles", "diamond": "diamonds", "cross": "crosses"}


def shape_prims(shape, c, r, angle, color):
    x, y = c
    if shape == "circle":
        return [("circle", (x, y), r, color, None, 0)]
    if shape == "square":
        return [("poly", regular_polygon(4, r, c, 45 + angle), color, None, 0)]
    if shape == "triangle":
        return [("poly", regular_polygon(3, r, c, -90 + angle), color, None, 0)]
    a = math.radians(angle)
    rot = lambda pts: [(x + px * math.cos(a) - py * math.sin(a), y + px * math.sin(a) + py * math.cos(a)) for px, py in pts]
    if shape == "diamond":
        return [("poly", rot([(0, -r), (0.6 * r, 0), (0, r), (-0.6 * r, 0)]), color, None, 0)]
    w = 0.32 * r
    return [("poly", rot(rect(-r, -w, r, w)), color, None, 0), ("poly", rot(rect(-w, -r, w, r)), color, None, 0)]


class SharedShapes(Family):
    name, unit = "shared_shapes", "shape"
    count_range = (1, 200)
    fixed_frame = True
    templates = {"how_many": "How many {plural} are in the image?", "count_the": "Count the {plural} in the image."}
    GAP = 4.0

    def label_for(self, p):
        return p["shape"]

    def question_fields(self, p):
        return {"plural": SHAPES[p["shape"]]}

    def sample(self, rng, count=None):
        shape = list(SHAPES)[int(rng.integers(len(SHAPES)))]
        r_max = min(30.0, (math.sqrt(0.4 * 440 ** 2 / count) - self.GAP) / 2)
        for _ in range(200):
            r = float(rng.uniform(9.0, max(9.0, r_max)))
            pts = []
            for _ in range(20000):
                if len(pts) == count:
                    break
                c = (float(rng.uniform(r + 4, 448 - r - 4)), float(rng.uniform(r + 4, 448 - r - 4)))
                if all((c[0] - q[0]) ** 2 + (c[1] - q[1]) ** 2 >= (2 * r + self.GAP) ** 2 for q in pts):
                    pts.append(c)
            if len(pts) == count:
                return {"count": count, "shape": shape, "r": round(r, 3),
                        "centers": [[round(x, 2), round(y, 2)] for x, y in pts],
                        "angles": [round(float(rng.uniform(0, 360)), 2) for _ in pts],
                        "colors": {"main": list(random_color(rng))}}
        raise ValueError("cannot place shapes")

    def scene(self, p):
        prims, units = [], []
        for c, a in zip(p["centers"], p["angles"]):
            sp = shape_prims(p["shape"], tuple(c), p["r"], a, p["colors"]["main"])
            prims += sp
            units.append({"point": tuple(c), "mask": sp, "size": 1.5 * p["r"] if p["shape"] != "cross" else 2 * p["r"]})
        return prims, units

    def recount(self, img, p, aff):
        return count_components(color_mask(img, p["colors"]["main"], 40), min_area=20)
