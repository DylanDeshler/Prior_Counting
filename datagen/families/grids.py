"""Grid / bar families (§4.2): calendar week (columns), piano octave (black keys), Rubik's face
(rows or columns); and their neutral twins."""

import numpy as np

from ..render import distinct_color, random_color, rect, rounded_rect
from .base import (Family, cluster_count, color_mask, components, count_components, not_colors_mask,
                   object_region, pick_distinct)


def _cells_cluster(img, mask, axis, gap_px, min_area=20):
    comps = components(mask, min_area)
    return cluster_count([c[axis] for c in comps], gap_px)


# ---------------------------------------------------------------------------
# Calendar week and grid twin
# ---------------------------------------------------------------------------

class Calendar(Family):
    name, unit, label, familiar = "calendar", "day column", "column", 7
    count_range = (4, 10)
    distractor_exclude = ("square",)
    templates = {"how_many": "How many day columns does this calendar have?",
                 "count_the": "Count the day columns in this calendar."}
    W, H, GAP, MARGIN, HEADER = 0.12, 0.1, 0.022, 0.04, 0.13

    def sample(self, rng, count=None):
        page = random_color(rng, s=(0, 0.1), v=(0.93, 1))
        cell = distinct_color(rng, [page], 100, s=(0.0, 0.4), v=(0.45, 0.78))
        header = distinct_color(rng, [page, cell], 90, s=(0.5, 1), v=(0.4, 0.9))
        rings = pick_distinct(rng, [(60, 60, 60), (150, 150, 155), (30, 30, 30)], [page, cell, header], 60)
        scribble = distinct_color(rng, [cell, page], 70, s=(0, 0.3), v=(0.3, 0.6))
        return {"count": 7, "rows": int(rng.integers(4, 7)), "n_rings": int(rng.integers(2, 5)),
                "scribbles": rng.random(60).round(3).tolist(),
                "colors": {"main": list(page), "cell": list(cell), "header": list(header), "rings": list(rings),
                           "scribble": list(scribble)}}

    def counterfactual(self, params, delta, rng):
        n = params["count"] + delta
        return {**params, "count": n}, {"op": "set_columns", "params": {"n": n}}

    def geometry(self, p):
        c, r = p["count"], p["rows"]
        gw = c * (self.W + self.GAP) - self.GAP
        gh = r * (self.H + self.GAP) - self.GAP
        pw, ph = gw + 2 * self.MARGIN, gh + self.HEADER + 2 * self.MARGIN
        x0, y0 = -pw / 2, -ph / 2
        return x0, y0, pw, ph, x0 + self.MARGIN, y0 + self.MARGIN + self.HEADER

    def scene(self, p):
        col = p["colors"]
        x0, y0, pw, ph, gx, gy = self.geometry(p)
        prims = [("poly", rect(x0, y0, x0 + pw, y0 + ph), col["main"], None, 0),
                 ("poly", rect(x0, y0, x0 + pw, y0 + self.HEADER), col["header"], None, 0)]
        for k in range(p["n_rings"]):
            rx = x0 + pw * (k + 1) / (p["n_rings"] + 1)
            prims.append(("poly", rounded_rect(rx - 0.012, y0 - 0.035, rx + 0.012, y0 + 0.035, 0.01), col["rings"], None, 0))
        units = []
        for j in range(p["count"]):
            cx0 = gx + j * (self.W + self.GAP)
            for i in range(p["rows"]):
                cy0 = gy + i * (self.H + self.GAP)
                prims.append(("poly", rect(cx0, cy0, cx0 + self.W, cy0 + self.H), col["cell"], None, 0))
                if p["scribbles"][(i * 10 + j) % 60] < 0.8:  # placeholder strokes, no legible text
                    prims.append(("line", [(cx0 + 0.02, cy0 + 0.03), (cx0 + 0.055, cy0 + 0.03)], col["scribble"], 0.012))
            top = (cx0 + self.W / 2, gy + self.H / 2)
            colrect = rect(cx0, gy, cx0 + self.W, gy + p["rows"] * (self.H + self.GAP) - self.GAP)
            units.append({"point": top, "mask": [("poly", colrect, 1, None, 0)], "size": self.W})
        return prims, units

    def recount(self, img, p, aff):
        prims, _ = self.scene(p)
        ys, xs = object_region(img.shape, prims, aff)
        return _cells_cluster(img, color_mask(img[ys, xs], p["colors"]["cell"], 30), 0, 0.5 * self.W * aff.scale)


class Grid(Family):
    """Neutral twin of the calendar: random r x c grid of cells, no text; the question asks columns."""
    name, unit, label = "twin_grid", "column", "column"
    count_range = (1, 14)
    distractor_exclude = ("square",)
    templates = {"how_many": "How many columns does this grid have?", "count_the": "Count the columns in this grid."}

    def sample(self, rng, count=None):
        page = random_color(rng, s=(0, 1), v=(0.1, 1))
        cell = distinct_color(rng, [page], 120, s=(0, 1), v=(0, 1))
        return {"count": count, "rows": int(rng.integers(2, 8)), "w": round(float(rng.uniform(0.09, 0.14)), 4),
                "h": round(float(rng.uniform(0.07, 0.14)), 4), "gap": round(float(rng.uniform(0.022, 0.04)), 4),
                "colors": {"main": list(page), "cell": list(cell)}}

    def scene(self, p):
        w, h, g = p["w"], p["h"], p["gap"]
        gw, gh = p["count"] * (w + g) + g, p["rows"] * (h + g) + g
        x0, y0 = -gw / 2, -gh / 2
        prims = [("poly", rect(x0, y0, x0 + gw, y0 + gh), p["colors"]["main"], None, 0)]
        units = []
        for j in range(p["count"]):
            cx0 = x0 + g + j * (w + g)
            for i in range(p["rows"]):
                cy0 = y0 + g + i * (h + g)
                prims.append(("poly", rect(cx0, cy0, cx0 + w, cy0 + h), p["colors"]["cell"], None, 0))
            units.append({"point": (cx0 + w / 2, y0 + g + h / 2),
                          "mask": [("poly", rect(cx0, y0 + g, cx0 + w, y0 + gh - g), 1, None, 0)], "size": w})
        return prims, units

    def recount(self, img, p, aff):
        prims, _ = self.scene(p)
        ys, xs = object_region(img.shape, prims, aff)
        return _cells_cluster(img, color_mask(img[ys, xs], p["colors"]["cell"], 30), 0, 0.5 * p["w"] * aff.scale)


# ---------------------------------------------------------------------------
# Piano octave and bars twin
# ---------------------------------------------------------------------------

PIANO_CANONICAL_GAPS = [0, 1, 3, 4, 5]  # C#, D#, F#, G#, A#


class Piano(Family):
    name, unit, label, familiar = "piano", "black key", "key", 5
    count_range = (2, 6)
    distractor_exclude = ("square",)
    templates = {"how_many": "How many black keys are in this octave?", "count_the": "Count the black keys in this octave."}
    WW, WH, BW, BH, TOP = 0.14, 0.62, 0.085, 0.4, -0.31

    def sample(self, rng, count=None):
        ivory = random_color(rng, s=(0, 0.12), v=(0.9, 1))
        black = random_color(rng, s=(0, 0.3), v=(0.03, 0.15))
        casing = pick_distinct(rng, [(120, 20, 30), (100, 65, 35), (40, 60, 120), (150, 150, 150), (60, 100, 60)],
                               [black, ivory], 80)
        return {"count": 5, "gaps": list(PIANO_CANONICAL_GAPS),
                "colors": {"main": list(ivory), "black": list(black), "outline": [125, 125, 125], "casing": list(casing)}}

    def feasible_deltas(self, params, deltas=None):
        return [n - params["count"] for n in (2, 3, 4, 6)] if params["count"] == 5 else []

    def counterfactual(self, params, delta, rng):
        n = params["count"] + delta
        gaps = sorted(rng.choice(6, size=n, replace=False).tolist())
        return {**params, "count": n, "gaps": gaps}, {"op": "set_black_keys", "params": {"gaps": gaps}}

    def scene(self, p):
        col = p["colors"]
        x0 = -3.5 * self.WW
        prims = [("poly", rect(x0 - 0.02, self.TOP - 0.09, -x0 + 0.02, self.TOP), col["casing"], None, 0)]
        for i in range(7):
            prims.append(("poly", rect(x0 + i * self.WW, self.TOP, x0 + (i + 1) * self.WW, self.TOP + self.WH),
                          col["main"], col["outline"], 0.008))
        units = []
        for g in p["gaps"]:
            cx = x0 + (g + 1) * self.WW
            r = rect(cx - self.BW / 2, self.TOP, cx + self.BW / 2, self.TOP + self.BH)
            prims.append(("poly", r, col["black"], None, 0))
            units.append({"point": (cx, self.TOP + self.BH / 2), "mask": [("poly", r, 1, None, 0)], "size": self.BW})
        return prims, units

    def recount(self, img, p, aff):
        prims, _ = self.scene(p)
        ys, xs = object_region(img.shape, prims, aff)
        return count_components(color_mask(img[ys, xs], p["colors"]["black"], 35), min_area=20)


class Bars(Family):
    """Neutral twin of the piano octave: alternating dark and light bars of random width."""
    name, unit, label = "twin_bars", "dark bar", "bar"
    count_range = (1, 12)
    distractor_exclude = ("square",)
    templates = {"how_many": "How many dark bars are there?", "count_the": "Count the dark bars."}

    def sample(self, rng, count=None):
        n = count
        light = [round(float(rng.uniform(0.04, 0.1)), 4) for _ in range(n + 1)]
        dark = [round(float(rng.uniform(0.045, 0.11)), 4) for _ in range(n)]
        return {"count": n, "light_w": light, "dark_w": dark, "h": round(float(rng.uniform(0.35, 0.7)), 4),
                "colors": {"main": list(random_color(rng, s=(0, 0.4), v=(0.8, 1))),
                           "dark": list(random_color(rng, s=(0, 1), v=(0.05, 0.35)))}}

    def scene(self, p):
        total = sum(p["light_w"]) + sum(p["dark_w"])
        x, h = -total / 2, p["h"]
        prims, units = [], []
        for i in range(2 * p["count"] + 1):
            if i % 2 == 0:
                w = p["light_w"][i // 2]
                prims.append(("poly", rect(x, -h / 2, x + w, h / 2), p["colors"]["main"], None, 0))
            else:
                w = p["dark_w"][i // 2]
                r = rect(x, -h / 2, x + w, h / 2)
                prims.append(("poly", r, p["colors"]["dark"], None, 0))
                units.append({"point": (x + w / 2, 0.0), "mask": [("poly", r, 1, None, 0)], "size": w})
            x += w
        return prims, units

    def recount(self, img, p, aff):
        prims, _ = self.scene(p)
        ys, xs = object_region(img.shape, prims, aff)
        return count_components(color_mask(img[ys, xs], p["colors"]["dark"], 35), min_area=20)


# ---------------------------------------------------------------------------
# Rubik's face and colored-grid twin
# ---------------------------------------------------------------------------

STICKERS = [(245, 245, 245), (250, 210, 0), (200, 20, 30), (255, 110, 0), (0, 70, 175), (0, 155, 70)]


class Rubik(Family):
    name, unit, familiar = "rubik", "row", 3
    count_range = (1, 6)
    distractor_exclude = ("square",)
    templates = {"how_many": "How many {dim} of squares are on this face of the cube?",
                 "count_the": "Count the {dim} of squares on this face of the cube."}
    CELL, GAP, MARGIN = 0.3, 0.03, 0.04

    def label_for(self, p):
        return "row" if p["dim"] == "rows" else "column"

    def question_fields(self, p):
        return {"dim": p["dim"]}

    def sample(self, rng, count=None):
        body = [(20, 20, 20), (35, 35, 35)][int(rng.integers(2))]
        return {"count": 3, "rows": 3, "cols": 3, "dim": "rows" if rng.random() < 0.5 else "columns",
                "stickers": rng.integers(0, 6, size=36).tolist(), "colors": {"main": list(body)}}

    def colors(self, p):
        return [tuple(p["colors"]["main"])] + STICKERS

    def feasible_deltas(self, params, deltas=None):
        return [n - 3 for n in (1, 2, 4, 5, 6)] if params["count"] == 3 else []

    def counterfactual(self, params, delta, rng):
        n = params["count"] + delta
        key = "rows" if params["dim"] == "rows" else "cols"
        return {**params, "count": n, key: n}, {"op": f"set_{params['dim']}", "params": {"n": n}}

    def cell_rects(self, p, cell, gap, margin):
        w = p["cols"] * (cell + gap) - gap + 2 * margin
        h = p["rows"] * (cell + gap) - gap + 2 * margin
        x0, y0 = -w / 2, -h / 2
        return (x0, y0, w, h), [[(x0 + margin + j * (cell + gap), y0 + margin + i * (cell + gap))
                                 for j in range(p["cols"])] for i in range(p["rows"])]

    def scene(self, p):
        (x0, y0, w, h), cells = self.cell_rects(p, self.CELL, self.GAP, self.MARGIN)
        prims = [("poly", rounded_rect(x0, y0, x0 + w, y0 + h, 0.05), p["colors"]["main"], None, 0)]
        for i, row in enumerate(cells):
            for j, (cx, cy) in enumerate(row):
                color = STICKERS[p["stickers"][i * 6 + j]]
                prims.append(("poly", rounded_rect(cx, cy, cx + self.CELL, cy + self.CELL, 0.03), color, None, 0))
        return prims, self.dim_units(p, cells, self.CELL)

    @staticmethod
    def dim_units(p, cells, cell):
        units = []
        if p["dim"] == "rows":
            for row in cells:
                (ax, ay), (bx, _) = row[0], row[-1]
                units.append({"point": (ax + cell / 2, ay + cell / 2),
                              "mask": [("poly", rect(ax, ay, bx + cell, ay + cell), 1, None, 0)], "size": cell})
        else:
            for j in range(len(cells[0])):
                (ax, ay), (_, by) = cells[0][j], cells[-1][j]
                units.append({"point": (ax + cell / 2, ay + cell / 2),
                              "mask": [("poly", rect(ax, ay, ax + cell, by + cell), 1, None, 0)], "size": cell})
        return units

    def recount(self, img, p, aff):
        prims, _ = self.scene(p)
        ys, xs = object_region(img.shape, prims, aff, pad=-2)
        mask = not_colors_mask(img[ys, xs], [p["colors"]["main"], tuple(img[0, 0])], 60)
        return _cells_cluster(img, mask, 1 if p["dim"] == "rows" else 0, 0.5 * self.CELL * aff.scale)


class ColoredGrid(Family):
    """Neutral twin of the Rubik's face: random r x c grid of colored squares."""
    name, unit = "twin_colored_grid", "row"
    count_range = (1, 8)
    distractor_exclude = ("square",)
    templates = {"how_many": "How many {dim} does this grid have?", "count_the": "Count the {dim} in this grid."}

    label_for = Rubik.label_for
    question_fields = Rubik.question_fields

    def sample(self, rng, count=None):
        dim = "rows" if rng.random() < 0.5 else "columns"
        other = int(rng.integers(2, 7))
        body = random_color(rng, s=(0, 1), v=(0, 0.3))
        palette = [list(distinct_color(rng, [body], 120)) for _ in range(int(rng.integers(2, 7)))]
        return {"count": count, "dim": dim, "rows": count if dim == "rows" else other,
                "cols": count if dim == "columns" else other, "cell": round(float(rng.uniform(0.2, 0.3)), 4),
                "palette": palette, "cells": rng.integers(0, len(palette), size=64).tolist(),
                "colors": {"main": list(body)}}

    def colors(self, p):
        return [tuple(p["colors"]["main"])] + [tuple(c) for c in p["palette"]]

    def scene(self, p):
        cell = p["cell"]
        (x0, y0, w, h), cells = Rubik.cell_rects(None, p, cell, cell * 0.1, cell * 0.13)
        prims = [("poly", rect(x0, y0, x0 + w, y0 + h), p["colors"]["main"], None, 0)]
        for i, row in enumerate(cells):
            for j, (cx, cy) in enumerate(row):
                prims.append(("poly", rect(cx, cy, cx + cell, cy + cell), p["palette"][p["cells"][i * 8 + j]], None, 0))
        return prims, Rubik.dim_units(p, cells, cell)

    def recount(self, img, p, aff):
        prims, _ = self.scene(p)
        ys, xs = object_region(img.shape, prims, aff, pad=-2)
        mask = not_colors_mask(img[ys, xs], [p["colors"]["main"], tuple(img[0, 0])], 60)
        return _cells_cluster(img, mask, 1 if p["dim"] == "rows" else 0, 0.5 * p["cell"] * aff.scale)
