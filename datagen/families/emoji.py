"""Emoji generators for Exp 2 L0 (§5.2, §5.3, §5.7): recolored objects (color prior) and crowds
(neutral count)."""

import json
from functools import lru_cache

import cv2
import numpy as np

from .. import emoji as E
from ..colornames import majority_name
from ..common import SOURCES
from ..render import rasterize_mask
from .base import COLOR_SUFFIX, Family, components

TERM_RGB = {"black": (20, 20, 20), "blue": (40, 80, 220), "brown": (120, 80, 40), "gray": (128, 128, 128),
            "green": (40, 170, 60), "orange": (250, 150, 20), "pink": (250, 150, 200), "purple": (140, 50, 170),
            "red": (220, 30, 30), "white": (245, 245, 245), "yellow": (250, 220, 30)}


@lru_cache(maxsize=1)
def color_objects():
    return {o["name"]: o for o in json.loads((SOURCES / "color_objects.json").read_text())["objects"]}


@lru_cache(maxsize=512)
def asset_stats(cps):
    """Alpha centroid and bbox (fractions of the asset side), mean color, and a few dominant colors."""
    rgba = E.base_rgba(cps)
    a = rgba[..., 3] > 127
    ys, xs = np.nonzero(a)
    side = rgba.shape[0]
    rgb = rgba[..., :3][a].astype(float)
    q = (rgb // 64).astype(int)
    keys, inv, counts = np.unique(q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2], return_inverse=True, return_counts=True)
    top = np.argsort(-counts)[:4]
    palette = [tuple(int(v) for v in rgb[inv.ravel() == k].mean(axis=0)) for k in top]
    # Point: the centroid, unless it isn't well inside the shape (e.g. a doughnut's hole); then the
    # nearest pixel that is at least half as deep inside the silhouette as the deepest one.
    import cv2
    depth = cv2.distanceTransform(a.astype(np.uint8), cv2.DIST_L2, 5)
    mx, my = xs.mean(), ys.mean()
    if depth[int(my), int(mx)] < 0.5 * depth.max():
        dy, dx = np.nonzero(depth >= 0.5 * depth.max())
        k = np.argmin((dx - mx) ** 2 + (dy - my) ** 2)
        mx, my = float(dx[k]), float(dy[k])
    return {"cx": mx / side - 0.5, "cy": my / side - 0.5,
            "w": (xs.max() - xs.min() + 1) / side, "h": (ys.max() - ys.min() + 1) / side,
            "mean": tuple(int(v) for v in rgb.mean(axis=0)), "palette": palette}


class EmojiColor(Family):
    """One recolorable emoji object. The first image of a pair is an identity recolor (null edit);
    the second recolors the pixels named `term` to the target color with L fixed."""
    name, unit = "emoji_color", "object"
    templates = {"how_many": "What color is the {object}?", "count_the": "Name the color of the {object}."}
    placement_kw = {"size_range": (150, 380)}
    K, FEATHER = 0.5, 2.0

    def label_for(self, p):
        return None

    def question(self, p, template):
        return self.templates[template].format(object=p["object"]) + COLOR_SUFFIX

    def sample(self, rng, count=None, object=None, mode="conflict"):
        o = color_objects()[object]
        term = o["modal"] if o["group"] == "single" else o["current"]
        st = asset_stats(o["cps"])
        return {"object": o["name"], "cps": o["cps"], "mode": mode, "term": term, "ab": None, "answer": term,
                "colors": {"main": list(st["mean"])}}

    def counterfactual(self, p, delta, rng):
        o = color_objects()[p["object"]]
        if p["mode"] == "conflict":
            options = sorted(o["targets"])
            target = options[int(rng.integers(len(options)))]
        else:  # neutral: sample from the object's own CoDa distribution, excluding the current color
            terms = sorted(o["target_probs"])
            probs = np.array([o["target_probs"][t] for t in terms])
            target = terms[int(rng.choice(len(terms), p=probs / probs.sum()))]
        ab = o["targets"][target]
        return ({**p, "ab": ab, "answer": target},
                {"op": "recolor", "params": {"from": p["term"], "to": target, "ab": ab, "k": self.K,
                                             "feather": self.FEATHER}})

    def null_edit(self, p):
        return {"op": "identity_recolor", "params": {"term": p["term"], "k": self.K, "feather": self.FEATHER},
                "null_edit": True}

    def scene(self, p):
        key = E.recolor_key(p["cps"], p["term"], p["ab"], self.K, self.FEATHER)
        st = asset_stats(p["cps"])
        unit = {"point": (st["cx"], st["cy"]), "mask": [("image", f"emoji:{p['cps']}", (0.0, 0.0), 1.0)],
                "size": min(st["w"], st["h"])}
        return [("image", key, (0.0, 0.0), 1.0)], [unit]

    def colors(self, p):
        return [tuple(p["colors"]["main"]), TERM_RGB[p["answer"]], TERM_RGB[p["term"]]]

    def verify_color(self, img, p, aff):
        """Majority color name of the recolor region in the rendered (pre-compression) image."""
        m = rasterize_mask([("image", E.mask_key(p["cps"], p["term"]), (0.0, 0.0), 1.0)], aff)
        eroded = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
        return majority_name(img, eroded if eroded.sum() >= 20 else m)


# ---------------------------------------------------------------------------
# Neutral crowds: scattered instances of one category with no familiar count
# ---------------------------------------------------------------------------

CROWD_CATEGORIES = {  # category -> (emoji name, plural)
    "apple": ("red apple", "apples"), "orange": ("tangerine", "oranges"), "lemon": ("lemon", "lemons"),
    "pear": ("pear", "pears"), "peach": ("peach", "peaches"), "strawberry": ("strawberry", "strawberries"),
    "tomato": ("tomato", "tomatoes"), "bottle": ("baby bottle", "bottles"), "cup": ("cup with straw", "cups"),
    "teacup": ("teacup without handle", "teacups"), "car": ("automobile", "cars"), "taxi": ("taxi", "taxis"),
    "bus": ("bus", "buses"), "boat": ("speedboat", "boats"), "balloon": ("balloon", "balloons"),
    "cupcake": ("cupcake", "cupcakes"), "doughnut": ("doughnut", "doughnuts"), "basketball": ("basketball", "basketballs"),
}


def crowd_cps(category):
    return E.emoji_index()[CROWD_CATEGORIES[category][0]]["cps"]


def fetch_crowd_assets():
    E.fetch_index()
    have = E.fetch_assets([crowd_cps(c) for c in CROWD_CATEGORIES])
    missing = {c for c in CROWD_CATEGORIES if crowd_cps(c) not in have}
    assert not missing, f"missing Noto assets for crowd categories: {missing}"


class EmojiCrowd(Family):
    name, unit = "emoji_crowd", "instance"
    count_range = (1, 40)
    fixed_frame = True
    templates = {"how_many": "How many {plural} are in the image?", "count_the": "Count the {plural} in the image."}
    GAP = 0.25  # min gap between instance boxes, as a fraction of instance size

    def label_for(self, p):
        return p["category"]

    def question_fields(self, p):
        return {"plural": CROWD_CATEGORIES[p["category"]][1]}

    def _free_spot(self, rng, centers, size, tries=3000):
        lo, hi = size / 2 + 4, 448 - size / 2 - 4
        for _ in range(tries):
            c = (float(rng.uniform(lo, hi)), float(rng.uniform(lo, hi)))
            if all(max(abs(c[0] - x), abs(c[1] - y)) >= size * (1 + self.GAP) for x, y in centers):
                return [round(c[0], 2), round(c[1], 2)]
        return None

    def sample(self, rng, count=None, category=None):
        cats = sorted(CROWD_CATEGORIES)
        category = category or cats[int(rng.integers(len(cats)))]
        fit = (0.35 * 440 ** 2 / max(1, count + 3)) ** 0.5 / (1 + self.GAP)  # leave room for edits
        for _ in range(50):
            size = float(rng.uniform(max(30.0, 0.5 * min(fit, 80)), max(30.0, min(fit, 80.0))))
            centers = []
            for _ in range(count):
                c = self._free_spot(rng, centers, size)
                if c is None:
                    break
                centers.append(c)
            if len(centers) == count:
                return {"count": count, "category": category, "cps": crowd_cps(category), "size": round(size, 2),
                        "centers": centers, "colors": {"main": list(asset_stats(crowd_cps(category))["mean"])}}
        raise ValueError("cannot place crowd")

    def counterfactual(self, p, delta, rng):
        centers = [list(c) for c in p["centers"]]
        if delta > 0:
            added = []
            for _ in range(delta):
                c = self._free_spot(rng, centers, p["size"])
                if c is None:
                    raise ValueError("no room to add instance")
                centers.append(c)
                added.append(c)
            edit = {"op": "add_instances", "params": {"centers": added}}
        else:
            idx = sorted(rng.choice(len(centers), size=-delta, replace=False).tolist())
            edit = {"op": "remove_instances", "params": {"centers": [centers[i] for i in idx]}}
            centers = [c for i, c in enumerate(centers) if i not in idx]
        return {**p, "count": len(centers), "centers": centers}, edit

    def scene(self, p):
        st = asset_stats(p["cps"])
        s = p["size"]
        prims, units = [], []
        for x, y in p["centers"]:
            prim = ("image", f"emoji:{p['cps']}", (x, y), s)
            prims.append(prim)
            units.append({"point": (x + st["cx"] * s, y + st["cy"] * s), "mask": [prim], "size": s * min(st["w"], st["h"])})
        return prims, units

    def colors(self, p):
        return [tuple(c) for c in asset_stats(p["cps"])["palette"]]

    def recount(self, img, p, aff):
        bg = img[0, 0].astype(float)
        fg = np.linalg.norm(img.astype(float) - bg, axis=2) > 30
        k = max(3, int(0.12 * p["size"]) | 1)
        fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        return len(components(fg, min_area=int(0.04 * p["size"] ** 2)))
