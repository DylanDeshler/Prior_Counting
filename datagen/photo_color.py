"""L2 color (§5.5, §5.7, §6 T2): real COCO/LVIS photos, rule-based recolor.

Pipeline:
  1. candidates()  LVIS instances of color objects that pass the photo-selection rules (§5.5):
                   longest bbox side >= 96 px; square crop 1.6x the longest side, clamped in the image,
                   resized to 448; exactly one instance of the class in the crop (category exhaustively
                   annotated); no traffic light in the crop; no crowd annotation of the class; image not
                   used by an eval set (§9). Images holding held-out color objects or traffic lights are
                   reserved for the test sets (§9.3) and never used in training.
  2. analyze()     per instance, on the actual 448 crop: the object's color name must cover >= 70% of
                   its mask with a confident name; conflict objects must read as their CoDa modal color
                   (canonical filter); lists the target colors reachable by a CIELAB recolor at the
                   object's lightness with a confident name.
  3. PhotoColor    generator (families/photo.py): canonical/original = identity LAB recolor through the same mask, feathering
                   and compression (null edit, §5.7); counterfactual/edited = recolor of the whole object
                   mask with L fixed. Verified per image: the answer color covers >= 65% of the object.
Records keep source ids + crop + polygon + recolor params, so images regenerate from the COCO photo.
"""

import hashlib
import json
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from . import coco
from .colornames import CHROMATIC, TERMS, lab_to_srgb, name_indices, srgb_to_lab, w2c
from .colorsets import load as load_color_objects
from .common import IMG, SOURCES, item_seed, rng_for
from .render import register_asset_loader

MIN_BBOX, CROP_SCALE = 96, 1.6
DOMINANT_MIN, VERIFY_MIN = 0.70, 0.65
REACH_MIN, CONF_MIN = 0.9, 0.6
L_RANGE = {"yellow": (70, 101), "pink": (60, 101), "orange": (50, 101), "brown": (0, 55)}
K, FEATHER = 0.5, 2.0
MAX_ANALYZE_PER_OBJECT = 300
CANDIDATES = SOURCES / "coco" / "l2_color_candidates.jsonl"


# ---------------------------------------------------------------------------
# Crops and masks
# ---------------------------------------------------------------------------

def crop_box(bbox, w, h):
    x, y, bw, bh = bbox
    side = min(CROP_SCALE * max(bw, bh), w, h)
    cx, cy = x + bw / 2, y + bh / 2
    x0 = min(max(cx - side / 2, 0), w - side)
    y0 = min(max(cy - side / 2, 0), h - side)
    return [round(x0, 2), round(y0, 2), round(side, 2)]


def overlap_frac(bbox, crop):
    x, y, bw, bh = bbox
    x0, y0, s = crop
    ix = max(0.0, min(x + bw, x0 + s) - max(x, x0))
    iy = max(0.0, min(y + bh, y0 + s) - max(y, y0))
    return ix * iy / max(1e-6, bw * bh)


def inside(bbox, crop):
    x, y, bw, bh = bbox
    x0, y0, s = crop
    return x >= x0 - 1 and y >= y0 - 1 and x + bw <= x0 + s + 1 and y + bh <= y0 + s + 1


def crop_image(image_id, crop):
    x0, y0, s = crop
    return coco.load_image(image_id).crop((x0, y0, x0 + s, y0 + s)).resize((IMG, IMG), Image.LANCZOS)


def crop_mask(seg, crop, k=2):
    """Polygon mask in 448 crop coordinates (rasterized at 2x, 50% threshold)."""
    x0, y0, s = crop
    sc = IMG * k / s
    m = Image.new("L", (IMG * k, IMG * k), 0)
    d = ImageDraw.Draw(m)
    for poly in seg:
        pts = [((poly[i] - x0) * sc, (poly[i + 1] - y0) * sc) for i in range(0, len(poly) - 1, 2)]
        if len(pts) >= 3:
            d.polygon(pts, fill=255)
    return np.asarray(m, dtype=np.uint16).reshape(IMG, k, IMG, k).mean(axis=(1, 3)) >= 128


def erode(mask, px=3):
    e = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=px).astype(bool)
    return e if e.sum() >= 200 else mask


def color_region(rgb, mask):
    """Pixels that show the object's surface color: the eroded mask minus deep shadows and specular
    highlights, which people discount when naming an object's color (and the color namer would call
    black / white)."""
    core = erode(mask)
    lab = srgb_to_lab(rgb[core])
    keep = ~((lab[:, 0] < 12) | ((lab[:, 0] > 95) & (np.hypot(lab[:, 1], lab[:, 2]) < 6)))
    region = core.copy()
    region[core] = keep
    return region if region.sum() >= 200 else core


# ---------------------------------------------------------------------------
# Recolor (CIELAB, L fixed, feathered) and color naming
# ---------------------------------------------------------------------------

def recolor(rgb, mask, ab, k=K, feather=FEATHER):
    """ab=None -> identity recolor: the same LAB round trip and feathering, no color change (null edit)."""
    lab = srgb_to_lab(rgb)
    soft = gaussian_filter(mask.astype(np.float64), feather)
    if ab is None:
        new_ab = lab[..., 1:]
    else:
        mean = lab[mask][:, 1:].mean(axis=0)
        new_ab = np.asarray(ab, float) + k * (lab[..., 1:] - mean)
    out_ab = lab[..., 1:] * (1 - soft[..., None]) + new_ab * soft[..., None]
    return np.round(lab_to_srgb(np.concatenate([lab[..., :1], out_ab], axis=-1))).astype(np.uint8)


def name_stats(rgb, mask):
    px = rgb[mask].astype(np.int64)
    probs = w2c()[px[:, 0] // 8 + 32 * (px[:, 1] // 8) + 1024 * (px[:, 2] // 8)]
    names = probs.argmax(-1)
    share = np.bincount(names, minlength=len(TERMS)) / max(1, len(names))
    top = int(np.argmax(share))
    return TERMS[top], float(share[top]), float(probs[names == top, top].mean()) if (names == top).any() else 0.0


def best_ab(lab_px, target, k=K, step=6):
    """ab maximizing the share of pixels named `target` after a recolor with L fixed (§5.3)."""
    mean = lab_px[:, 1:].mean(axis=0)
    dev = k * (lab_px[:, 1:] - mean)
    grid = np.arange(-96, 97, step, dtype=float)
    ti = TERMS.index(target)
    table = w2c()
    best = (-1.0, None, 0.0, 0.0)
    for a in grid:
        cand = np.stack(np.meshgrid([a], grid, indexing="ij"), -1).reshape(-1, 2)
        ab = cand[:, None, :] + dev[None]
        lab = np.concatenate([np.broadcast_to(lab_px[None, :, :1], ab.shape[:2] + (1,)), ab], -1)
        raw = lab_to_srgb(lab, clip=False)
        clip = np.abs(raw - np.clip(raw, 0, 255)).mean(axis=(1, 2))
        px = np.round(np.clip(raw, 0, 255)).astype(np.int64)
        probs = table[px[..., 0] // 8 + 32 * (px[..., 1] // 8) + 1024 * (px[..., 2] // 8)]
        frac = (probs.argmax(-1) == ti).mean(axis=1)
        conf = probs[..., ti].mean(axis=1)
        score = frac + conf - 0.002 * clip
        i = int(np.argmax(score))
        if score[i] > best[0]:
            best = (float(score[i]), (round(float(cand[i, 0]), 1), round(float(cand[i, 1]), 1)), float(frac[i]), float(conf[i]))
    return best[1], best[2], best[3]


def l_ok(term, mean_l):
    lo, hi = L_RANGE.get(term, (0, 101))
    return lo <= mean_l <= hi


# ---------------------------------------------------------------------------
# 1. Candidate instances
# ---------------------------------------------------------------------------

def candidates():
    from .colorsets import norm
    from .exclusions import excluded_coco_ids
    objects = load_color_objects()
    cat_obj = {c: o["name"] for o in objects.values() for c in o["lvis_cats"]}
    held_cats = {c for o in objects.values() if o["split"] == "heldout" for c in o["lvis_cats"]}
    excluded = excluded_coco_ids()
    extra = coco.coco_extra()
    out, seen = [], set()
    for split in [s for s in ("train", "val") if (coco.DIR / f"lvis_{s}.pkl").exists()]:
        L = coco.lvis(split)
        tl_cats = {cid for cid, c in L["cats"].items() if c["name"] == "traffic_light"}
        for image_id, anns in tqdm(L["anns"].items(), desc=f"LVIS {split} candidates", unit="img"):
            if image_id in excluded or image_id in seen:
                continue
            seen.add(image_id)
            im = L["images"][image_id]
            ex = extra.get(image_id, {"crowd": [], "traffic_light": []})
            # Reserve images (§9.3): held-out color objects or traffic lights anywhere -> test only.
            reserve = bool(ex["traffic_light"]) or any(a["cat"] in held_cats or a["cat"] in tl_cats for a in anns)
            for a in anns:
                obj = cat_obj.get(a["cat"])
                if obj is None or max(a["bbox"][2:]) < MIN_BBOX or a["cat"] in im["not_exhaustive"]:
                    continue
                o = objects[obj]
                held = o["split"] == "heldout"
                if reserve != held:  # train objects only from train-pool images; held-out only from reserve
                    continue
                crop = crop_box(a["bbox"], im["w"], im["h"])
                if not inside(a["bbox"], crop):
                    continue
                same = [b for b in anns if b["cat"] in o["lvis_cats"] and b["id"] != a["id"] and overlap_frac(b["bbox"], crop) > 0.1]
                if same or norm(obj) in {norm(c) for c in ex["crowd"]}:
                    continue
                tl = [b for b in anns if b["cat"] in tl_cats] + [{"bbox": bb} for bb in ex["traffic_light"]]
                if any(overlap_frac(b["bbox"], crop) > 0.0 for b in tl) and not held:
                    continue
                out.append({"image_id": image_id, "url": im["url"], "ann_id": a["id"], "object": obj,
                            "group": o["group"], "split": o["split"], "crop": crop, "seg": a["seg"]})
    return out


# ---------------------------------------------------------------------------
# 2. Per-instance color analysis
# ---------------------------------------------------------------------------

def analyze(c):
    objects = load_color_objects()
    o = objects[c["object"]]
    rgb = np.asarray(crop_image(c["image_id"], c["crop"]))
    mask = crop_mask(c["seg"], c["crop"])
    if mask.sum() < 1500:
        return {**c, "ok": False, "why": "mask too small"}
    core = color_region(rgb, mask)
    current, share, conf = name_stats(rgb, core)
    lab_px = srgb_to_lab(rgb[core][:: max(1, core.sum() // 1500)])
    mean_l = float(lab_px[:, 0].mean())
    res = {**c, "current": current, "share": round(share, 3), "conf": round(conf, 3), "mean_l": round(mean_l, 1)}
    if share < DOMINANT_MIN:
        return {**res, "ok": False, "why": f"{current} covers {share:.0%} of the object"}
    if not l_ok(current, mean_l):  # e.g. a dark "orange" that people call brown
        return {**res, "ok": False, "why": f"{current} at L*={mean_l:.0f} is ambiguous"}
    if o["group"] == "single" and current != o["modal"]:
        return {**res, "ok": False, "why": f"reads {current}, CoDa modal {o['modal']}"}
    options = o["cf_colors"] if o["group"] == "single" else [t for t in CHROMATIC if t != current and o["coda"][t] > 0]
    targets = {}
    for t in options:
        if l_ok(t, mean_l):
            ab, frac, tconf = best_ab(lab_px, t)
            if frac >= REACH_MIN and tconf >= CONF_MIN:
                targets[t] = ab
    if not targets:
        return {**res, "ok": False, "why": "no reachable target colors"}
    return {**res, "ok": True, "targets": targets}


def analyze_all(cands):
    """Analyze up to MAX_ANALYZE_PER_OBJECT random candidates per object (cached in CANDIDATES)."""
    from concurrent.futures import ProcessPoolExecutor
    from .build import workers
    done = {}
    if CANDIDATES.exists():
        for line in CANDIDATES.open():
            r = json.loads(line)
            done[r["ann_id"]] = r
    by_obj = {}
    for c in cands:
        by_obj.setdefault(c["object"], []).append(c)
    todo = []
    for obj, cs in sorted(by_obj.items()):
        cs = sorted(cs, key=lambda c: hashlib.sha256(f"{obj}|{c['ann_id']}".encode()).hexdigest())
        todo += [c for c in cs[:MAX_ANALYZE_PER_OBJECT] if c["ann_id"] not in done]
    if todo:
        have = coco.download_images([(c["image_id"], c["url"]) for c in todo])
        todo = [c for c in todo if c["image_id"] in have]
        with ProcessPoolExecutor(workers()) as ex, CANDIDATES.open("a") as f:
            for r in tqdm(ex.map(analyze, todo, chunksize=4), total=len(todo), desc="analyzing instances", unit="inst"):
                f.write(json.dumps(r) + "\n")
                done[r["ann_id"]] = r
    return [r for r in done.values() if r["ok"]]


# ---------------------------------------------------------------------------
# 3. Generator
# ---------------------------------------------------------------------------

def _key(p, mask_only=False):
    d = {"i": p["image_id"], "c": p["crop"], "s": p["seg"]}
    if not mask_only:
        d.update(ab=p["ab"], k=K, f=FEATHER)
    return "photo:" + json.dumps(d, separators=(",", ":"))


@lru_cache(maxsize=256)
def _load(key):
    d = json.loads(key.split(":", 1)[1])
    mask = crop_mask(d["s"], d["c"])
    if "ab" not in d:  # unit mask: alpha = object mask
        return Image.fromarray(np.dstack([np.full(mask.shape + (3,), 255, np.uint8), (mask * 255).astype(np.uint8)]), "RGBA")
    rgb = np.asarray(crop_image(d["i"], d["c"]))
    out = recolor(rgb, mask, None if d["ab"] is None else tuple(d["ab"]), d["k"], d["f"])
    return Image.fromarray(np.dstack([out, np.full(mask.shape, 255, np.uint8)]), "RGBA")


register_asset_loader("photo", _load)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def _round_robin(by_obj, n_pairs, max_per_instance=4):
    """Pairs balanced across objects. Each photo is used up to max_per_instance times, each time with
    a different target color (variant v), and only after every photo of that object was used once."""
    order = sorted(by_obj)
    queues = {o: [(c, v) for v in range(max_per_instance) for c in by_obj[o] if v < len(c["targets"])] for o in order}
    picks, i = [], 0
    while len(picks) < n_pairs and any(queues.values()):
        o = order[i % len(order)]
        if queues[o]:
            picks.append(queues[o].pop(0))
        i += 1
    return picks


def color_jobs(analyzed, dataset, out, mode, n_pairs, split):
    group = "single" if mode == "conflict" else "any"
    by_obj = {}
    for c in analyzed:
        if c["group"] == group and c["split"] == split:
            by_obj.setdefault(c["object"], []).append(c)
    for o in by_obj:
        by_obj[o].sort(key=lambda c: hashlib.sha256(f"{dataset}|{c['ann_id']}".encode()).hexdigest())
    jobs = []
    for i, (c, variant) in enumerate(_round_robin(by_obj, n_pairs)):
        jobs.append({"kind": "pair", "generator": "photo_color", "family": c["object"].replace(" ", "_"),
                     "dataset": dataset, "id_prefix": dataset.replace("_", "-").lower(), "idx": i, "pool_index": i,
                     "template": ("how_many", "count_the")[i % 2], "data_type": "conflict" if mode == "conflict" else "neutral",
                     "roles": ("canonical", "counterfactual") if mode == "conflict" else ("original", "edited"),
                     "out": out, "prior": "color", "level": "L2", "delta": 0, "optional": True, "photo": True,
                     "sample_kw": {"cand": c, "mode": mode, "variant": variant},
                     "source": {"dataset": "coco2017+lvis_v1", "image_id": c["image_id"], "ann_ids": [c["ann_id"]],
                                "license": "COCO / Flickr terms: release ids + edit params, not images"}})
    return jobs, {o: len(v) for o, v in by_obj.items()}
