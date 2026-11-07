"""Real-photo counting with points (neutral, unedited): LVIS crowds (spec §5.5 crowd rules).

Each item is a square crop around all instances of one LVIS category in a COCO photo:
  - the category is exhaustively annotated in that image (not in not_exhaustive_category_ids) and is not
    marked crowd in COCO; 3-15 instances, every one fully inside the crop, >= 20 px on its short side at
    448 px, >= 300 px of mask filling >= 25% of its box, not split into fragments, no two overlapping by
    IoU > 0.5;
  - no instance touches the photo border (it may continue outside the frame); a few categories that
    looked wrong in a visual check are denied (DENY);
  - only manufactured objects and whole fruit (WordNet), never prepared food / dishes / vegetables, which
    are often annotated as slices or portions;
  - no traffic light in the crop (held-out class), no POPE/ORIC image (§9.2), no near-duplicate of an
    eval image (§9.4, checked per image);
  - animals, body parts, people and the §4.2 excluded categories are never the counted category.
Points: one per instance, the deepest point of its mask (always inside it); masks: one per instance.
The count comes from LVIS's exhaustive annotation (SAM 3 recount, as in the spec, is not run), which can
still miss instances in busy scenes: audit tests/R/count with the viewer (Gate 4) before reporting it.
Split by image id: 20% of images -> tests/R/count (held out), the rest -> real/count (training).
"""

import hashlib
import re
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from . import coco
from .photo_color import crop_mask, overlap_frac

# Tightened after a visual check (2026-11-07): LVIS's "exhaustive" flag misses instances in crowded
# scenes, and small / overlapping / sliced instances make the count ambiguous.
MIN_N, MAX_N = 3, 15
MIN_SHORT_PX, MIN_MASK_PX = 20, 300
MIN_FILL = 0.25       # mask area / box area: drops thin, fragmented or barely visible instances
MAX_PAIR_IOU = 0.5    # instances overlapping more than this are ambiguous to count
EDGE_PX = 2           # an instance touching the photo border may continue outside it: drop the image
# Categories that looked wrong in the visual check: sliced fruit, parts whose extent is unclear, frequent
# mislabels (ties as chokers), and drawers (cabinet fronts, hard to count and over-represented).
DENY = {"peach", "choker", "cornet", "hook", "crate", "drinking straw", "straw", "drawer"}
# Usually annotated as slices / pieces, so "how many X" has no clear answer.
PIECE_WORDS = {"ham", "bacon", "pepperoni", "salami", "sausage", "meatball", "cheese", "onion", "lettuce",
               "pickle", "mushroom", "olive", "potato", "kiwi", "avocado", "cucumber", "carrot", "chip", "fries",
               "candy", "cookie", "cracker", "crouton", "noodle", "rice", "bread", "toast"}
MAX_PER_CATEGORY = {"train": 120, "test": 25}
TEST_FRAC = 0.2


def clean(name):
    return re.sub(r"\s*\(.*?\)", "", name.replace("_", " ")).strip().lower()


def plural(word):
    if re.search(r"(s|x|z|ch|sh)$", word):
        return word + "es"
    if re.search(r"[^aeiou]y$", word):
        return word[:-1] + "ies"
    return word + "s"


def discrete_object(synset_name, wn):
    """Manufactured objects and whole fruit: things counted as whole units. Prepared food, dishes and
    vegetables (often annotated as slices or portions) are excluded."""
    try:
        syn = wn.synset(synset_name)
    except Exception:
        return False
    closure = {x.name() for x in syn.closure(lambda x: x.hypernyms())} | {syn.name()}
    return "artifact.n.01" in closure or ("edible_fruit.n.01" in closure and "vegetable.n.01" not in closure)


def allowed_categories(cats):
    from .colorsets import EXCLUDED_WORDS, _wordnet, wordnet_excluded
    wn = _wordnet()
    ok = {}
    for cid, c in cats.items():
        name = clean(c["name"])
        if any(w in name for w in EXCLUDED_WORDS) or wordnet_excluded(c.get("synset"), wn):
            continue
        if any(re.search(rf"\b{w}", name) for w in PIECE_WORDS) or not discrete_object(c.get("synset"), wn):
            continue
        ok[cid] = name
    return ok


def polygon_area(poly):
    x, y = np.asarray(poly[0::2], float), np.asarray(poly[1::2], float)
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def fragmented(seg):
    """Annotation split into pieces (heavy occlusion): the largest polygon is < 80% of the total."""
    areas = [polygon_area(p) for p in seg if len(p) >= 6]
    return not areas or max(areas) < 0.8 * sum(areas)


def box_iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0.0, min(ax1, bx1) - max(a[0], b[0])), max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(1e-6, a[2] * a[3] + b[2] * b[3] - inter)


def square_around(box, w, h, margin=1.1):
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * margin
    if side > min(w, h):
        side = max(x1 - x0, y1 - y0)
        if side > min(w, h):
            return None
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return [round(min(max(cx - side / 2, 0), w - side), 2), round(min(max(cy - side / 2, 0), h - side), 2), round(side, 2)]


def candidates():
    """One crop per image at most (the category with the most valid instances), with split labels."""
    from .exclusions import excluded_coco_ids
    excluded, extra = excluded_coco_ids(), coco.coco_extra()
    out, seen = [], set()
    for split in [s for s in ("train", "val") if (coco.DIR / f"lvis_{s}.pkl").exists()]:
        L = coco.lvis(split)
        names = {k: v for k, v in allowed_categories(L["cats"]).items() if v not in DENY}
        tl = {cid for cid, c in L["cats"].items() if c["name"] == "traffic_light"}
        for image_id, anns in tqdm(L["anns"].items(), desc=f"LVIS {split} crowds", unit="img"):
            if image_id in excluded or image_id in seen:
                continue
            seen.add(image_id)
            im = L["images"][image_id]
            ex = extra.get(image_id, {"crowd": [], "traffic_light": []})
            crowd = {clean(c) for c in ex["crowd"]}
            lights = [a["bbox"] for a in anns if a["cat"] in tl] + list(ex["traffic_light"])
            by_cat = defaultdict(list)
            for a in anns:
                if a["cat"] in names and a["cat"] not in im["not_exhaustive"]:
                    by_cat[a["cat"]].append(a)
            best = None
            for cid, group in by_cat.items():
                if not MIN_N <= len(group) <= MAX_N or names[cid] in crowd:
                    continue
                if any(a["bbox"][0] < EDGE_PX or a["bbox"][1] < EDGE_PX or a["bbox"][0] + a["bbox"][2] > im["w"] - EDGE_PX
                       or a["bbox"][1] + a["bbox"][3] > im["h"] - EDGE_PX for a in group):
                    continue
                xs0 = min(a["bbox"][0] for a in group); ys0 = min(a["bbox"][1] for a in group)
                xs1 = max(a["bbox"][0] + a["bbox"][2] for a in group); ys1 = max(a["bbox"][1] + a["bbox"][3] for a in group)
                crop = square_around((xs0, ys0, xs1, ys1), im["w"], im["h"])
                if crop is None or any(overlap_frac(b, crop) > 0 for b in lights):
                    continue
                scale = 448 / crop[2]
                if any(min(a["bbox"][2], a["bbox"][3]) * scale < MIN_SHORT_PX or a["area"] * scale ** 2 < MIN_MASK_PX
                       or a["area"] < MIN_FILL * a["bbox"][2] * a["bbox"][3] or fragmented(a["seg"]) for a in group):
                    continue
                if any(box_iou(a["bbox"], b["bbox"]) > MAX_PAIR_IOU for i, a in enumerate(group) for b in group[i + 1:]):
                    continue
                if best is None or len(group) > best["count"]:
                    best = {"image_id": image_id, "url": im["url"], "category": names[cid], "count": len(group),
                            "crop": crop, "segs": [a["seg"] for a in group], "ann_ids": [a["id"] for a in group]}
            if best:
                h = int(hashlib.sha256(str(image_id).encode()).hexdigest(), 16) % 100
                best["split"] = "test" if h < TEST_FRAC * 100 else "train"
                out.append(best)
    return out


def balanced(cands, split):
    """At most MAX_PER_CATEGORY[split] crops per category, chosen by a stable hash."""
    by = defaultdict(list)
    for c in cands:
        if c["split"] == split:
            by[c["category"]].append(c)
    picks = []
    for cat in sorted(by):
        cs = sorted(by[cat], key=lambda c: hashlib.sha256(f"{split}|{c['image_id']}".encode()).hexdigest())
        picks += cs[:MAX_PER_CATEGORY[split]]
    return sorted(picks, key=lambda c: hashlib.sha256(f"order|{c['image_id']}".encode()).hexdigest())


def count_jobs(cands, split):
    dataset, out = ("real_count", "real/count") if split == "train" else ("real_count_test", "tests/R/count")
    jobs = []
    for i, c in enumerate(balanced(cands, split)):
        jobs.append({"kind": "single", "generator": "photo_count", "family": c["category"].replace(" ", "_"),
                     "dataset": dataset, "id_prefix": dataset.replace("_", "-"), "idx": i, "pool_index": i,
                     "count": c["count"], "template": ("how_many", "count_the")[i % 2], "data_type": "neutral",
                     "out": out, "level": "L2", "optional": True, "photo": True, "distractors": False,
                     "sample_kw": {"cand": c},
                     "source": {"dataset": "coco2017+lvis_v1", "image_id": c["image_id"], "ann_ids": c["ann_ids"],
                                "license": "COCO / Flickr terms: release ids + crop params, not images"}})
    return jobs
