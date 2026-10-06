"""Radial / polygon families (§4.2): analog clock (hour marks; numerals for Exp 2), stop sign,
five-pointed star, plus held-out traffic light and snowflake; and their neutral twins."""

import math

import numpy as np

from ..render import (distinct_color, hsv_color, random_color, regular_polygon, rounded_rect, text_box,
                      thick_segment)
from .base import (Family, color_mask, components, convex_vertex_count, count_components,
                   not_colors_mask, object_region, pick_distinct, polygon_vertices,
                   ring_crossings)


def polar(r, deg, c=(0.0, 0.0)):
    a = math.radians(deg)
    return (c[0] + r * math.cos(a), c[1] + r * math.sin(a))


def _clock_colors(rng):
    dial = random_color(rng, s=(0.0, 0.25), v=(0.85, 1.0)) if rng.random() < 0.7 else random_color(rng)
    marks = distinct_color(rng, [dial], 140, s=(0.0, 1.0), v=(0.0, 0.5))
    hands = distinct_color(rng, [dial, marks], 110, s=(0.5, 1.0), v=(0.3, 1.0))
    rim = distinct_color(rng, [dial, marks, hands], 90, s=(0.0, 1.0), v=(0.1, 0.9))
    return {"main": list(dial), "marks": list(marks), "hands": list(hands), "rim": list(rim)}


def _hands(p, length_h, length_m, width):
    col = p["colors"]["hands"]
    return [("line", [(0, 0), polar(length_h, p["hour_deg"])], col, width),
            ("line", [(0, 0), polar(length_m, p["minute_deg"])], col, width * 0.7),
            ("circle", (0, 0), width * 0.9, col, None, 0)]


class Clock(Family):
    """Exp 1 clock: hour marks, hands, no numerals."""
    name, unit, label, familiar = "clock", "hour mark", "mark", 12
    count_range = (9, 15)
    distractor_exclude = ("ring",)
    templates = {"how_many": "How many hour marks are on this clock?",
                 "count_the": "Count the hour marks on this clock."}
    R_IN, R_OUT = 0.375, 0.455

    def sample(self, rng, count=None):
        return {"count": 12 if count is None else count, "mark_w": round(float(rng.uniform(0.025, 0.04)), 4),
                "hour_deg": round(float(rng.uniform(0, 360)), 2), "minute_deg": round(float(rng.uniform(0, 360)), 2),
                "colors": _clock_colors(rng)}

    def counterfactual(self, params, delta, rng):
        return {**params, "count": params["count"] + delta}, {"op": "set_mark_count", "params": {"n": params["count"] + delta}}

    def scene(self, p):
        col = p["colors"]
        prims = [("circle", (0, 0), 0.485, col["main"], col["rim"], 0.03)]
        units = []
        for k in range(p["count"]):
            a = -90 + 360 * k / p["count"]
            seg = thick_segment(polar(self.R_IN, a), polar(self.R_OUT, a), p["mark_w"])
            prims.append(("poly", seg, col["marks"], None, 0))
            units.append({"point": polar((self.R_IN + self.R_OUT) / 2, a), "mask": [("poly", seg, 1, None, 0)],
                          "size": self.R_OUT - self.R_IN})
        return prims + _hands(p, 0.2, 0.31, 0.035), units

    def recount(self, img, p, aff):
        prims, _ = self.scene(p)
        ys, xs = object_region(img.shape, prims, aff)
        return count_components(color_mask(img[ys, xs], p["colors"]["marks"]))


class ClockNumerals(Family):
    """Exp 2 clock: 12 Arabic numerals; counterfactuals remove numerals (§5.2)."""
    name, unit, label, familiar = "clock_numerals", "numeral", "numeral", 12
    count_range = (1, 12)
    distractor_exclude = ("ring",)
    templates = {"how_many": "How many numerals are on this clock?", "count_the": "Count the numerals on this clock."}
    R_NUM, CAP = 0.355, 0.085

    def sample(self, rng, count=None):
        return {"count": 12, "numerals": list(range(1, 13)),
                "hour_deg": round(float(rng.uniform(0, 360)), 2), "minute_deg": round(float(rng.uniform(0, 360)), 2),
                "colors": _clock_colors(rng)}

    def counterfactual(self, params, delta, rng):
        assert delta < 0, "clock numerals: removal only"
        removed = sorted(rng.choice(params["numerals"], size=-delta, replace=False).tolist())
        keep = [n for n in params["numerals"] if n not in removed]
        return {**params, "count": len(keep), "numerals": keep}, {"op": "remove_numerals", "params": {"numerals": removed}}

    def scene(self, p):
        col = p["colors"]
        prims = [("circle", (0, 0), 0.485, col["main"], col["rim"], 0.03)]
        for k in range(60):  # minute dots in the rim color
            prims.append(("circle", polar(0.445, 6 * k), 0.008 if k % 5 else 0.014, col["rim"], None, 0))
        units = []
        for n in p["numerals"]:
            c = polar(self.R_NUM, -90 + 30 * n)
            prims.append(("text", str(n), c, self.CAP, col["marks"]))
            units.append({"point": c, "mask": [("poly", text_box(str(n), c, self.CAP), 1, None, 0)], "size": self.CAP})
        return prims + _hands(p, 0.17, 0.25, 0.03), units

    def recount(self, img, p, aff):
        prims, _ = self.scene(p)
        ys, xs = object_region(img.shape, prims, aff)
        cap_px = self.CAP * aff.scale
        comps = components(color_mask(img, p["colors"]["marks"]), min_area=max(4, int(0.05 * cap_px ** 2)))
        # Digits of one numeral sit within a few degrees of each other around the dial; neighbors are 30 deg apart.
        angles = sorted(math.degrees(math.atan2(c[1] - aff.cy, c[0] - aff.cx)) % 360 for c in comps
                        if 0.25 * aff.scale < math.hypot(c[0] - aff.cx, c[1] - aff.cy) < 0.44 * aff.scale)
        if not angles:
            return 0
        gaps = [b - a for a, b in zip(angles, angles[1:])] + [angles[0] + 360 - angles[-1]]
        return max(1, sum(g > 12 for g in gaps))


class TickRing(Family):
    """Neutral twin of the clock: radial ticks on a circle, no hands."""
    name, unit, label = "twin_ticks", "tick", "tick"
    count_range = (1, 30)
    distractor_exclude = ("ring",)
    templates = {"how_many": "How many tick marks are on the circle?", "count_the": "Count the tick marks on the circle."}

    def sample(self, rng, count=None):
        ring = random_color(rng, s=(0, 1), v=(0.2, 0.9))
        tick = distinct_color(rng, [ring], 120, s=(0, 1), v=(0, 1))
        return {"count": count, "phase": round(float(rng.uniform(0, 360)), 2),
                "len": round(float(rng.uniform(0.07, 0.11)), 4), "w": round(float(rng.uniform(0.022, 0.04)), 4),
                "colors": {"main": list(ring), "ticks": list(tick)}}

    def scene(self, p):
        col = p["colors"]
        prims = [("circle", (0, 0), 0.43, None, col["main"], 0.012)]
        units = []
        for k in range(p["count"]):
            a = p["phase"] + 360 * k / p["count"]
            seg = thick_segment(polar(0.43 - p["len"] / 2, a), polar(0.43 + p["len"] / 2, a), p["w"])
            prims.append(("poly", seg, col["ticks"], None, 0))
            units.append({"point": polar(0.43, a), "mask": [("poly", seg, 1, None, 0)], "size": p["len"]})
        return prims, units

    def recount(self, img, p, aff):
        prims, _ = self.scene(p)
        ys, xs = object_region(img.shape, prims, aff)
        return count_components(color_mask(img[ys, xs], p["colors"]["ticks"]))


# ---------------------------------------------------------------------------
# Stop sign and n-gon twin
# ---------------------------------------------------------------------------

def ngon(n, r, rot=None):
    return regular_polygon(n, r, start_deg=-90 + 180 / n if rot is None else rot)


def side_units(poly):
    units = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        units.append({"point": tuple((a + b) / 2), "mask": [("line", [tuple(a), tuple(b)], 1, 0.0)],
                      "size": float(np.linalg.norm(b - a)), "px_width": 7})
    return units


class StopSign(Family):
    name, unit, label, familiar = "stop_sign", "side", "side", 8
    count_range = (5, 11)
    distractor_exclude = ("triangle", "square", "diamond")
    templates = {"how_many": "How many sides does this sign have?", "count_the": "Count the sides of this sign."}

    def sample(self, rng, count=None):
        red = hsv_color(rng.uniform(-6, 6), rng.uniform(0.8, 0.95), rng.uniform(0.7, 0.9))
        return {"count": 8, "colors": {"main": list(red), "border": [250, 250, 250], "text": [250, 250, 250]},
                "text_size": round(float(rng.uniform(0.15, 0.18)), 4)}

    def counterfactual(self, params, delta, rng):
        n = params["count"] + delta
        return {**params, "count": n}, {"op": "set_sides", "params": {"n": n}}

    def scene(self, p):
        col = p["colors"]
        outer = ngon(p["count"], 0.5)
        prims = [("poly", outer, col["border"], None, 0), ("poly", ngon(p["count"], 0.45), col["main"], None, 0),
                 ("text", "STOP", (0, 0.005), p["text_size"], col["text"])]
        return prims, side_units(outer)

    def recount(self, img, p, aff):
        prims, _ = self.scene(p)
        ys, xs = object_region(img.shape, prims, aff, pad=6)
        sub = img[ys, xs]
        bg = sub[0, 0]
        return len(polygon_vertices(~color_mask(sub, bg, 30)))


class NGon(Family):
    """Neutral twin of the stop sign: random-colored regular n-gon, no text."""
    name, unit, label = "twin_ngon", "side", "side"
    count_range = (3, 14)
    distractor_exclude = ("triangle", "square", "diamond")
    templates = {"how_many": "How many sides does this shape have?", "count_the": "Count the sides of this shape."}

    def sample(self, rng, count=None):
        main = hsv_color(rng.uniform(25, 335), rng.uniform(0.3, 1), rng.uniform(0.3, 1))  # never stop-sign red
        outline = distinct_color(rng, [main], 80, s=(0, 1), v=(0, 1))
        return {"count": count, "rot": round(float(rng.uniform(0, 360)), 2), "outline_w": round(float(rng.uniform(0, 0.04)), 4),
                "colors": {"main": list(main), "outline": list(outline)}}

    def scene(self, p):
        poly = ngon(p["count"], 0.5, p["rot"])
        col = p["colors"]
        w = p["outline_w"]
        prims = [("poly", poly, col["main"], col["outline"] if w > 0.005 else None, w)]
        return prims, side_units(poly)

    recount = StopSign.recount


# ---------------------------------------------------------------------------
# Star and burst twin
# ---------------------------------------------------------------------------

STAR_INNER = {3: 0.26, 4: 0.33, 5: 0.38, 6: 0.45, 7: 0.48, 8: 0.52}


def star_poly(outer, inner, tip_angles, inner_angles):
    pts = []
    for ro, ao, ri, ai in zip(outer, tip_angles, inner, inner_angles):
        pts += [polar(ro, ao), polar(ri, ai)]
    return np.array(pts)


def tip_units(poly):
    units = []
    n = len(poly) // 2
    for i in range(n):
        tip, left, right = poly[2 * i], poly[2 * i - 1], poly[2 * i + 1]
        tri = np.array([left, tip, right])
        length = float(np.linalg.norm(tip - (left + right) / 2))
        units.append({"point": tuple(tip), "mask": [("poly", tri, 1, None, 0),
                                                    ("line", [tuple(left), tuple(tip), tuple(right)], 1, 0.0)],
                      "size": length, "px_width": 4})
    return units


class Star(Family):
    name, unit, label, familiar = "star", "point", "point", 5
    count_range = (3, 8)
    distractor_exclude = ()
    templates = {"how_many": "How many points does this star have?", "count_the": "Count the points of this star."}

    def sample(self, rng, count=None):
        main = hsv_color(rng.uniform(40, 55), rng.uniform(0.7, 1), rng.uniform(0.85, 1)) if rng.random() < 0.5 \
            else random_color(rng)
        outline = distinct_color(rng, [main], 80, s=(0, 1), v=(0, 0.7))
        return {"count": 5, "inner_jitter": round(float(rng.uniform(-0.03, 0.03)), 4),
                "outline_w": round(float(rng.uniform(0, 0.03)), 4), "colors": {"main": list(main), "outline": list(outline)}}

    def feasible_deltas(self, params, deltas=None):
        return [n - params["count"] for n in (3, 4, 6, 7, 8)] if params["count"] == 5 else []

    def counterfactual(self, params, delta, rng):
        n = params["count"] + delta
        return {**params, "count": n}, {"op": "set_points", "params": {"n": n}}

    def scene(self, p):
        n = p["count"]
        inner = STAR_INNER[n] + p["inner_jitter"]
        tips = [-90 + 360 * k / n for k in range(n)]
        poly = star_poly([0.5] * n, [0.5 * inner] * n, tips, [a + 180 / n for a in tips])
        col = p["colors"]
        w = p["outline_w"]
        return [("poly", poly, col["main"], col["outline"] if w > 0.005 else None, w)], tip_units(poly)

    def recount(self, img, p, aff):
        prims, _ = self.scene(p)
        ys, xs = object_region(img.shape, prims, aff, pad=6)
        sub = img[ys, xs]
        return convex_vertex_count(polygon_vertices(~color_mask(sub, sub[0, 0], 30), eps_frac=0.006))


class Burst(Family):
    """Neutral twin of the star: irregular n-pointed burst."""
    name, unit, label = "twin_burst", "point", "point"
    count_range = (3, 12)
    templates = {"how_many": "How many points does this shape have?", "count_the": "Count the points of this shape."}

    def sample(self, rng, count=None):
        n = count
        step = 360 / n
        tips = [k * step + float(rng.uniform(-0.2, 0.2)) * step for k in range(n)]
        outer = [float(rng.uniform(0.36, 0.5)) for _ in range(n)]
        inner, inner_ang = [], []
        for k in range(n):
            a0, a1 = tips[k], tips[(k + 1) % n] + (360 if k == n - 1 else 0)
            ai = (a0 + a1) / 2 + float(rng.uniform(-0.1, 0.1)) * step
            # Keep inner vertices clearly concave: well inside the chord between neighboring tips.
            chord = min(outer[k], outer[(k + 1) % n]) * math.cos(math.radians((a1 - a0) / 2))
            inner.append(float(rng.uniform(0.35, 0.7)) * min(0.25, chord))
            inner_ang.append(ai)
        main = random_color(rng)
        return {"count": n, "outer": [round(v, 4) for v in outer], "inner": [round(v, 4) for v in inner],
                "tips": [round(v, 3) for v in tips], "inner_ang": [round(v, 3) for v in inner_ang],
                "colors": {"main": list(main)}}

    def scene(self, p):
        poly = star_poly(p["outer"], p["inner"], p["tips"], p["inner_ang"])
        return [("poly", poly, p["colors"]["main"], None, 0)], tip_units(poly)

    recount = Star.recount


# ---------------------------------------------------------------------------
# Held out: snowflake (arms) and traffic light (lamps)
# ---------------------------------------------------------------------------

class Snowflake(Family):
    name, unit, label, familiar = "snowflake", "arm", "arm", 6
    count_range = (3, 9)
    templates = {"how_many": "How many arms does this snowflake have?", "count_the": "Count the arms of this snowflake."}

    def sample(self, rng, count=None):
        main = hsv_color(rng.uniform(180, 220), rng.uniform(0, 0.5), rng.uniform(0.8, 1)) if rng.random() < 0.6 \
            else random_color(rng)
        return {"count": 6, "phase": round(float(rng.uniform(0, 360)), 2), "w": round(float(rng.uniform(0.035, 0.05)), 4),
                "branch_len": round(float(rng.uniform(0.08, 0.12)), 4), "colors": {"main": list(main)}}

    def feasible_deltas(self, params, deltas=None):
        return [n - 6 for n in (3, 4, 5, 7, 8, 9)] if params["count"] == 6 else []

    def counterfactual(self, params, delta, rng):
        n = params["count"] + delta
        return {**params, "count": n}, {"op": "set_arms", "params": {"n": n}}

    def scene(self, p):
        col, w, bl = p["colors"]["main"], p["w"], p["branch_len"]
        prims = [("circle", (0, 0), 0.07, col, None, 0)]
        units = []
        for k in range(p["count"]):
            a = p["phase"] + 360 * k / p["count"]
            arm = [("line", [(0, 0), polar(0.47, a)], col, w)]
            for t, ln in ((0.25, bl), (0.36, bl * 0.75)):
                base = polar(t, a)
                arm += [("line", [base, polar(ln, a + 45, base)], col, w * 0.8),
                        ("line", [base, polar(ln, a - 45, base)], col, w * 0.8)]
            prims += arm
            units.append({"point": polar(0.47, a), "mask": arm, "size": 0.47})
        return prims, units

    def recount(self, img, p, aff):
        mask = color_mask(img, p["colors"]["main"], 60)
        return ring_crossings(mask, (aff.cx, aff.cy), 0.16 * aff.scale)


LAMP_COLORS = [(230, 30, 30), (245, 190, 20), (30, 200, 70)]


class TrafficLight(Family):
    name, unit, label, familiar = "traffic_light", "lamp", "lamp", 3
    count_range = (1, 6)
    distractor_exclude = ("ring",)
    templates = {"how_many": "How many lamps does this traffic light have?",
                 "count_the": "Count the lamps on this traffic light."}
    SPACING, LAMP_R = 0.27, 0.1

    def sample(self, rng, count=None):
        housing = [(35, 35, 35), (20, 20, 20), (60, 60, 55), (210, 170, 20), (25, 60, 35)][int(rng.integers(5))]
        return {"count": 3, "orientation": "v" if rng.random() < 0.7 else "h", "lit": int(rng.integers(0, 6)),
                "colors": {"main": list(housing)}}

    def feasible_deltas(self, params, deltas=None):
        return [n - 3 for n in (1, 2, 4, 5, 6)] if params["count"] == 3 else []

    def counterfactual(self, params, delta, rng):
        n = params["count"] + delta
        return {**params, "count": n}, {"op": "set_lamps", "params": {"n": n}}

    def lamp_centers(self, p):
        n = p["count"]
        offs = [(k - (n - 1) / 2) * self.SPACING for k in range(n)]
        return [(0.0, o) if p["orientation"] == "v" else (o, 0.0) for o in offs]

    def lamp_color(self, p, k):
        n = p["count"]
        base = LAMP_COLORS[1] if n == 1 else LAMP_COLORS[min(2, round(k * 2 / (n - 1)))] if n <= 3 \
            else LAMP_COLORS[k % 3]
        return base if k == p["lit"] % n else tuple(round(v * 0.55) for v in base)

    def scene(self, p):
        n = p["count"]
        half_len = n * self.SPACING / 2 + 0.04
        box = (-0.16, -half_len, 0.16, half_len) if p["orientation"] == "v" else (-half_len, -0.16, half_len, 0.16)
        prims = [("poly", rounded_rect(*box, 0.06), p["colors"]["main"], None, 0)]
        units = []
        for k, c in enumerate(self.lamp_centers(p)):
            prims.append(("circle", c, self.LAMP_R, self.lamp_color(p, k), None, 0))
            units.append({"point": c, "mask": [("circle", c, self.LAMP_R, 1, None, 0)], "size": 2 * self.LAMP_R})
        return prims, units

    def recount(self, img, p, aff):
        prims, _ = self.scene(p)
        ys, xs = object_region(img.shape, prims, aff, pad=-int(0.07 * aff.scale))
        sub = img[ys, xs]
        return count_components(not_colors_mask(sub, [p["colors"]["main"]], 35), min_area=20)
