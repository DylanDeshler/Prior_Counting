"""Real-photo PART counting from Pascal-Part (see pascal_part.py): legs of one animal.

Pools (records in the usual format: points + per-unit masks, so every export format works):
  parts/count       unedited crops; answer = visible legs                      (neutral, single images)
  parts/conflict    edit pairs on fully visible animals                        (conflict, canonical/counterfactual)
  tests/P/count, tests/P/conflict   the same from held-out images (20% by image-id hash)

Edit pairs. The counterfactual gets a real edit, its canonical partner the same KIND of edit on the
background (a null edit), so both images are edited and "edited" never predicts "count changed":
  add     copy one of the animal's own legs (feathered alpha) beside the others, its top touching the
          body: count = canonical + 1
          null: a background patch pasted, in the same leg shape, onto background
  remove  inpaint one leg (OpenCV Telea on a slightly dilated mask): count = canonical - 1
          null: the same inpainting on a leg-shaped background region
The count is known by construction; checks verify the edit landed inside the crop, didn't touch the
other legs, and changed the pixels. Every parameter is stored in the record, so `render(rec)` regenerates
the image (OpenCV inpainting is deterministic).
"""

import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from . import pascal_part as pp
from .common import (DATA, IMG, REPORTS, compress, generator_version, item_seed, reading_order, rle_encode, rng_for,
                     sample_compression, save_png, write_jsonl)

POOLS = {("count", "train"): ("parts/count", "parts_count"), ("count", "test"): ("tests/P/count", "parts_count_test"),
         ("conflict", "train"): ("parts/conflict", "parts_conflict"),
         ("conflict", "test"): ("tests/P/conflict", "parts_conflict_test")}
QUESTIONS = {"how_many": "How many legs of the {name} can you see? Answer with a number in curly brackets, e.g., {{9}}.",
             "count_the": "Count the visible legs of the {name}. Answer with a number in curly brackets, e.g., {{9}}."}
NAMES = {"cat": "cat", "dog": "dog", "cow": "cow", "sheep": "sheep", "horse": "horse", "bird": "bird"}
MARGIN = 1.15             # crop side = 1.15 x the animal's longer bbox side
MIN_LEG_PX = 150          # every visible leg >= 150 px of mask at 448 px (count pool; unit size >= 12 px rule)
MIN_EDIT_LEG_PX = 250     # legs used in edit pairs: larger, so the edit is clearly visible
MIN_OBJ_FRAC = 0.12       # the animal covers >= 12% of the crop
MAX_OTHER_OVERLAP = 0.10  # another animal of the same class may have <= 10% of its box inside the crop
INPAINT = ("cv2_telea", 5)
GENERATOR_VERSION = generator_version()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def square_crop(bbox, w, h, margin=MARGIN):
    x0, y0, x1, y1 = bbox
    side = min(max(x1 - x0, y1 - y0) * margin, w, h)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return [int(round(min(max(cx - side / 2, 0), w - side))), int(round(min(max(cy - side / 2, 0), h - side))), int(side)]


def in_crop_frac(mask, crop):
    x, y, s = crop
    tot = mask.sum()
    return mask[y:y + s, x:x + s].sum() / tot if tot else 0.0


def box_overlap_frac(b, crop):
    x, y, s = crop
    ix = max(0, min(b[2], x + s) - max(b[0], x)); iy = max(0, min(b[3], y + s) - max(b[1], y))
    area = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / area


def shift(mask, dx, dy):
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    ys2, xs2 = ys + dy, xs + dx
    ok = (ys2 >= 0) & (ys2 < h) & (xs2 >= 0) & (xs2 < w)
    out[ys2[ok], xs2[ok]] = True
    return out, ok.mean() if len(ok) else 0.0


def dilate(mask, r):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def deepest_point(mask):
    d = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return float(x), float(y)


def foot_point(mask):
    """Point near the bottom of a leg (v2 targets): the deepest point of the lowest quarter of the mask.
    The base model points at the feet, where legs separate; mid-leg points (v1) land where legs overlap."""
    ys = np.nonzero(mask)[0]
    if not len(ys):
        return deepest_point(mask)
    band = mask.copy()
    band[: ys.min() + int(0.75 * (ys.max() - ys.min()))] = False
    return deepest_point(band if band.any() else mask)


def to_crop(mask, crop):
    x, y, s = crop
    m = Image.fromarray(mask[y:y + s, x:x + s].astype(np.uint8) * 255).resize((IMG, IMG), Image.NEAREST)
    return np.asarray(m) > 127


# ---------------------------------------------------------------------------
# Edits (full resolution, before the crop)
# ---------------------------------------------------------------------------

def paste(img, src_mask, dx, dy, dst_shift=(0, 0)):
    """Copy the pixels under src_mask, moved by (dx, dy), with a feathered edge."""
    dst, _ = shift(src_mask, dx, dy)
    alpha = cv2.GaussianBlur(dst.astype(np.float32), (0, 0), 1.2)[..., None]
    moved = np.roll(np.roll(img, dy, axis=0), dx, axis=1).astype(np.float32)
    return (alpha * moved + (1 - alpha) * img.astype(np.float32)).round().clip(0, 255).astype(np.uint8), dst


def inpaint(img, region):
    return cv2.inpaint(img, region.astype(np.uint8) * 255, INPAINT[1], cv2.INPAINT_TELEA)


def apply_edit(img, legs, e):
    """Apply edit spec e to img (H x W x 3 uint8). Returns (image, list of leg masks after the edit)."""
    op = e["op"]
    if op == "add":
        out, new = paste(img, legs[e["leg"]], e["dx"], e["dy"])
        return out, legs + [new]
    if op == "remove":
        region = dilate(legs[e["leg"]], e["dilate"])
        return inpaint(img, region), [m for i, m in enumerate(legs) if i != e["leg"]]
    if op == "null_add":  # background patch pasted in leg shape onto background
        shape, _ = shift(legs[e["leg"]], e["tx"], e["ty"])          # source location (background)
        out, _ = paste(img, shape, e["dx"], e["dy"])                # moved to the target location (background)
        return out, legs
    if op == "null_remove":
        shape, _ = shift(legs[e["leg"]], e["tx"], e["ty"])
        return inpaint(img, dilate(shape, e["dilate"])), legs
    raise ValueError(op)


def plan_add(legs, obj_mask, crop, rng, others_mask, wide=False):
    """A leg and a horizontal offset where the copy sits inside the crop, clear of the other legs, mostly
    below the body (not pasted over the torso/head). wide (v2): more offsets, small vertical jitter and a
    looser body-overlap limit, so fewer add edits are infeasible."""
    body = obj_mask & ~np.logical_or.reduce(legs)
    near = dilate(np.logical_or.reduce(legs), 3)
    factors = [0.8, -0.8, 1.1, -1.1, 1.4, -1.4] + ([0.6, -0.6, 1.7, -1.7, 2.0, -2.0] if wide else [])
    for leg in rng.permutation(len(legs)):
        m = legs[leg]
        ys, xs = np.nonzero(m)
        wd, ht = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        for f in rng.permutation(factors):
            for dy in ([0, int(0.03 * ht), -int(0.03 * ht)] if wide else [0]):
                dx = int(round(f * wd))
                new, kept = shift(m, dx, dy)
                if kept < 0.999 or in_crop_frac(new, crop) < 0.999:
                    continue
                area = new.sum()
                if (new & near).sum() > 0.02 * area or (new & body).sum() > (0.40 if wide else 0.30) * area \
                        or (new & others_mask).sum() > 0.02 * area:
                    continue
                # attached: the top of the copied leg must touch the body, or it reads as a separate object
                nys = np.nonzero(new)[0]
                top = new.copy(); top[nys.min() + max(2, int(0.15 * (nys.max() - nys.min() + 1))):] = False
                if not (dilate(top, 4) & body).any():
                    continue
                return {"op": "add", "leg": int(leg), "dx": dx, "dy": dy}
    return None


def plan_remove(legs, rng):
    """A leg whose dilated mask barely touches the other legs (so removing it leaves them intact)."""
    for leg in rng.permutation(len(legs)):
        r = max(3, int(0.03 * np.sqrt(legs[leg].sum())))
        region = dilate(legs[leg], r)
        if all((region & m).sum() <= 0.03 * m.sum() for i, m in enumerate(legs) if i != leg):
            return {"op": "remove", "leg": int(leg), "dilate": r}
    return None


def plan_null(kind, legs, obj_mask, crop, rng, edit):
    """Same leg shape, moved onto background (clear of the animal by 10 px), inside the crop."""
    leg = edit["leg"]
    m = legs[leg]
    keep_out = dilate(obj_mask, 10)
    x, y, s = crop
    ys, xs = np.nonzero(m)
    lx0, lx1, ly0, ly1 = xs.min(), xs.max(), ys.min(), ys.max()
    if lx1 - lx0 >= s or ly1 - ly0 >= s:
        return None
    for _ in range(60):  # offsets relative to the leg's own position, landing inside the crop
        tx, ty = int(rng.integers(x - lx0, x + s - lx1)), int(rng.integers(y - ly0, y + s - ly1))
        tgt, kept = shift(m, tx, ty)
        if kept < 0.999 or in_crop_frac(tgt, crop) < 0.999 or (tgt & keep_out).any():
            continue
        if kind == "remove":
            return {"op": "null_remove", "leg": leg, "tx": tx, "ty": ty, "dilate": edit["dilate"]}
        # null add: the patch comes from another background spot of the same shape
        for _ in range(30):
            dx, dy = int(rng.integers(-s // 3, s // 3)), int(rng.integers(-s // 3, s // 3))
            src, _ = shift(m, tx - dx, ty - dy)
            if src.sum() >= 0.999 * m.sum() and in_crop_frac(src, crop) > 0.999 and not (src & keep_out).any() and (dx or dy):
                return {"op": "null_add", "leg": leg, "tx": tx - dx, "ty": ty - dy, "dx": dx, "dy": dy}
    return None


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def candidates():
    """Animals usable for leg counting, from part-annotated objects. Light: no masks kept."""
    out = []
    for iid in tqdm(pp.image_ids(), desc="Pascal-Part candidates", unit="img"):
        if not (pp.IMAGES / f"{iid}.jpg").exists():
            continue
        objs = pp.load(iid)
        h, w = objs[0]["mask"].shape if objs else (0, 0)
        for k, o in enumerate(objs):
            if o["class"] not in pp.ANIMALS or not o["has_parts"]:
                continue
            legs = list(o["units"]["leg"].values())
            if not legs:
                continue
            crop = square_crop(o["bbox"], w, h)
            scale = IMG / crop[2]
            if o["mask"].sum() * scale ** 2 < MIN_OBJ_FRAC * IMG * IMG or min(in_crop_frac(m, crop) for m in legs) < 0.98:
                continue
            if any(j != k and p["class"] == o["class"] and box_overlap_frac(p["bbox"], crop) > MAX_OTHER_OVERLAP
                   for j, p in enumerate(objs)):
                continue
            areas = [m.sum() * scale ** 2 for m in legs]
            if min(areas) < MIN_LEG_PX:
                continue
            canon = pp.UNITS[o["class"]]["leg"][0]
            out.append({"image_id": iid, "obj": k, "class": o["class"], "count": len(legs), "canonical": canon,
                        "crop": crop, "edit_ok": bool(len(legs) == canon and min(areas) >= MIN_EDIT_LEG_PX),
                        "split": pp.split_of(iid)})
    return out


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def _load_img(iid):
    return np.asarray(Image.open(pp.IMAGES / f"{iid}.jpg").convert("RGB"))


def render(rec):
    """Regenerate a record's image from its stored parameters (also used by the builder)."""
    p = rec["render"]["params"]
    o = pp.load(p["image_id"])[p["obj"]]
    legs = [o["units"]["leg"][k] for k in sorted(o["units"]["leg"])]
    img = _load_img(p["image_id"])
    if p.get("edit"):
        img, legs = apply_edit(img, legs, p["edit"])
    x, y, s = p["crop"]
    crop = Image.fromarray(img[y:y + s, x:x + s]).resize((IMG, IMG), Image.LANCZOS)
    return compress(crop, rec["compression"]), legs, o


def make_record(c, dataset, out, idx, role, edit, compression, pair_id, familiar, delta, template):
    name = NAMES[c["class"]]
    suffix = {"neutral": "neu", "canonical": "can", "counterfactual": "cf"}[role]
    rid = f"{dataset.replace('_', '-')}-{c['class']}-{idx:06d}-{suffix}"
    return {
        "id": rid, "pair_id": pair_id, "dataset": dataset, "pool_index": idx, "level": "L2", "prior": "count",
        "data_type": "neutral" if role == "neutral" else "conflict", "role": role, "family": c["class"], "unit": "leg",
        "label": "leg", "image": f"{out}/images/{rid}.png", "question_template": template,
        "question": QUESTIONS[template].format(name=name), "answer": None, "familiar_answer": familiar, "delta": delta,
        "points": None, "masks_rle": None, "unit_size_px": None, "object_bbox": None,
        "edit": {"op": (edit or {}).get("op", "none"), "params": edit or {}, "null_edit": bool(edit and edit["op"].startswith("null"))},
        "source": {"dataset": "pascal_part_voc2010", "image_id": c["image_id"], "ann_ids": [c["obj"]],
                   "license": "PASCAL VOC (Flickr terms); Pascal-Part annotations (Chen et al. 2014)"},
        "render": {"generator": "pascal_part", "params": {"image_id": c["image_id"], "obj": c["obj"], "crop": c["crop"],
                                                           "edit": edit, "inpaint": list(INPAINT)}},
        "compression": compression, "seed": item_seed(dataset, c["class"], idx), "attempt": 0,
        "generator_version": GENERATOR_VERSION,
        "checks": {"independent_recount": None, "sam3_count": None, "ocr_count": None, "polygon_sides": None,
                   "color_name": None, "audit": None, "by_construction": True, "phash_min_dist": None},
    }


def finalize(rec, write=True):
    from .families.photo import phash_check
    img, legs, o = render(rec)
    crop = rec["render"]["params"]["crop"]
    masks = [to_crop(m, crop) for m in legs]
    masks = [m for m in masks if m.any()]
    anchor = foot_point if rec["render"]["params"].get("anchor") == "foot" else deepest_point
    pts = [anchor(m) for m in masks]
    sizes = [float(np.sqrt(m.sum())) for m in masks]
    order = reading_order(pts, float(np.median(sizes)) if sizes else 1.0)
    rec["answer"] = len(masks)
    rec["points"] = [[round(pts[i][0], 2), round(pts[i][1], 2)] for i in order]
    rec["masks_rle"] = [rle_encode(masks[i]) for i in order]
    rec["unit_size_px"] = round(float(np.median(sizes)), 2) if sizes else 0.0
    ob = to_crop(o["mask"], crop)
    ys, xs = np.nonzero(ob)
    rec["object_bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else [0, 0, IMG, IMG]
    rec["checks"]["independent_recount"] = len(masks)  # by construction: annotated legs +/- the edited one
    phash_check(img, rec)
    if write:
        rec["image_sha256"] = save_png(img, DATA / rec["image"])
    return rec


def count_job(args):
    c, dataset, out, idx = args
    rng = rng_for(item_seed(dataset, c["class"], idx))
    rec = make_record(c, dataset, out, idx, "neutral", None, sample_compression(rng), None, None, None,
                      ("how_many", "count_the")[idx % 2])
    try:
        return [finalize(rec)], None
    except Exception as e:
        return [], str(e)[:80]


def pair_job(args):
    """v1: even idx adds a leg, odd idx removes one (falling back to the other kind if infeasible).
    v2 (opts): the job's kind only (no fallback, so the add share stays high), wide add search, foot points."""
    c, dataset, out, idx, *rest = args
    opts = rest[0] if rest else {}
    rng = rng_for(item_seed(dataset, c["class"], idx))
    o = pp.load(c["image_id"])[c["obj"]]
    legs = [o["units"]["leg"][k] for k in sorted(o["units"]["leg"])]
    others = np.zeros_like(o["mask"])
    for j, p in enumerate(pp.load(c["image_id"])):
        if j != c["obj"]:
            others |= p["mask"]
    kinds = opts.get("kinds") or (("add", "remove") if idx % 2 == 0 else ("remove", "add"))
    for kind in kinds:
        edit = plan_add(legs, o["mask"], c["crop"], rng, others, wide=opts.get("wide", False)) if kind == "add" \
            else plan_remove(legs, rng)
        null = plan_null(kind, legs, o["mask"] | others, c["crop"], rng, edit) if edit else None
        if edit and null:
            break
    else:
        return [], "no feasible edit"
    comp = sample_compression(rng)
    template = ("how_many", "count_the")[idx % 2]
    pid = f"{dataset.replace('_', '-')}-{c['class']}-{idx:06d}"
    delta = 1 if edit["op"] == "add" else -1
    recs = [make_record(c, dataset, out, idx, "canonical", null, comp, pid, c["canonical"], 0, template),
            make_record(c, dataset, out, idx, "counterfactual", edit, comp, pid, c["canonical"], delta, template)]
    if opts.get("anchor"):
        for r in recs:
            r["render"]["params"]["anchor"] = opts["anchor"]
    try:
        recs = [finalize(r) for r in recs]
    except Exception as e:
        return [], str(e)[:80]
    can, cf = recs
    if can["answer"] != c["canonical"] or cf["answer"] != c["canonical"] + delta:
        return [], f"count check failed ({can['answer']}, {cf['answer']})"
    # the edit must actually change pixels where it says it does
    a, b = (np.asarray(Image.open(DATA / r["image"]).convert("L"), np.int16) for r in recs)
    if (np.abs(a - b) > 12).sum() < 150:  # at least ~150 px visibly changed (a mean over the image misses small legs)
        return [], "edit not visible"
    return recs, None


def run(fn, jobs, desc):
    from .build import workers
    recs, dropped = [], Counter()
    with ProcessPoolExecutor(workers()) as ex:
        for r, err in tqdm(ex.map(fn, jobs, chunksize=4), total=len(jobs), desc=desc, unit="item"):
            recs += r
            if err:
                dropped[err] += 1
    return recs, dropped


def build(limit=None):
    """parts/count, parts/conflict and their held-out tests. limit: items per pool (quick local checks)."""
    t = time.time()
    cands = candidates()
    report = {"candidates": len(cands), "by_class_count": Counter(f"{c['class']}:{c['count']}" for c in cands),
              "edit_ok": int(sum(bool(c["edit_ok"]) for c in cands))}
    for (kind, split), (out, dataset) in POOLS.items():
        pool = [c for c in cands if c["split"] == split and (kind == "count" or c["edit_ok"])]
        if kind == "conflict":
            pool = [c for c in pool for _ in range(2)]  # one add pair and one remove pair per animal
        jobs = [(c, dataset, out, i) for i, c in enumerate(pool)][:limit]
        recs, dropped = run(count_job if kind == "count" else pair_job, jobs, dataset)
        write_jsonl(DATA / out / "records.jsonl", recs)
        report[dataset] = {"jobs": len(jobs), "records": len(recs), "dropped": dict(dropped),
                           "answers": dict(sorted(Counter(r["answer"] for r in recs).items())),
                           "edits": dict(Counter(r["edit"]["op"] for r in recs))}
        print(f"{out}: {len(recs)} images from {len(jobs)} jobs; dropped {dict(dropped)}")
    report["seconds"] = round(time.time() - t, 1)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "build_parts.json").write_text(json.dumps(report, indent=1, default=dict))
    return report


V2_POOL = ("parts/conflict_v2", "parts_conflict_v2")
V2_SCHEDULE = ("add", "add", "remove")  # per fully visible animal: two add pairs, one remove pair


def build_v2(limit=None):
    """Training-only v2 edit pairs (2026-11-10), evaluated on the unchanged v1 tests (tests/P/*):
      - add-heavy: per animal two add jobs and one remove job, no fallback between kinds (v1 had 14% five-legged
        images vs 75% in VLMBias animals), with a wider add search so fewer adds are infeasible;
      - foot-anchored points (v1 targets sat mid-leg, where legs overlap; the base model points at the feet).
    Same edits otherwise (copy-paste add, Telea remove, background null edits): v2 tests the data mix and
    point placement, not edit realism."""
    t = time.time()
    cands = [c for c in candidates() if c["split"] == "train" and c["edit_ok"]]
    out, dataset = V2_POOL
    jobs = [(c, dataset, out, i, {"kinds": (V2_SCHEDULE[i % 3],), "wide": True, "anchor": "foot"})
            for i, c in enumerate(c for c in cands for _ in V2_SCHEDULE)][:limit]
    recs, dropped = run(pair_job, jobs, dataset)
    write_jsonl(DATA / out / "records.jsonl", recs)
    report = {"animals": len(cands), "jobs": len(jobs), "records": len(recs), "dropped": dict(dropped),
              "answers": dict(sorted(Counter(r["answer"] for r in recs).items())),
              "edits": dict(Counter(r["edit"]["op"] for r in recs)), "seconds": round(time.time() - t, 1)}
    (REPORTS / "build_parts_v2.json").write_text(json.dumps(report, indent=1, default=dict))
    print(f"{out}: {len(recs)} images from {len(jobs)} jobs; answers {report['answers']}; dropped {dict(dropped)}")
    return report
