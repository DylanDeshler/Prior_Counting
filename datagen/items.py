"""Build records (§3.3) for pairs and single images, render them from their own metadata, and
verify each count by an independent pixel recount (§10.2).

Every image is rendered *from the record* (after a JSON round trip), so a record always holds
everything needed to regenerate its image byte-for-byte (§10.1 determinism).
"""

import json
import statistics

import numpy as np
from PIL import Image

from .common import (DATA, IMG, MIN_UNIT_PX, compress, generator_version, item_seed, reading_order,
                     rle_encode, rng_for, sample_compression, save_png, to_jsonable)
from .families import GENERATORS
from .families.base import DELTAS
from .render import (Affine, color_dist, distractor_prims, image_bbox, local_extent, make_background,
                     rasterize_mask, render_scene, sample_background, sample_distractors, sample_placement)

ROLE_SUFFIX = {"canonical": "can", "counterfactual": "cf", "original": "orig", "edited": "edit", "neutral": "neu"}
IDENTITY = {"cx": 0.0, "cy": 0.0, "scale": 1.0, "rotation_deg": 0.0}
ANALYSIS_BGS = [(255, 0, 255), (0, 255, 255), (255, 255, 0), (128, 0, 255), (0, 128, 128), (255, 128, 192),
                (64, 255, 64), (128, 64, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0)]
MAX_ATTEMPTS = 25
GENERATOR_VERSION = generator_version()


class Rejected(Exception):
    pass


# ---------------------------------------------------------------------------
# Rendering from a record
# ---------------------------------------------------------------------------

def scene_for(rec):
    r = rec["render"]
    fam = GENERATORS[r["generator"]]
    prims, units = fam.scene(r["params"])
    return fam, prims, units


def render_clean(rec):
    """Final image before compression, plus scene info."""
    fam, prims, units = scene_for(rec)
    r = rec["render"]
    aff = Affine.from_params(r["placement"])
    layers = [(prims, aff)] + [(distractor_prims(d), Affine.from_params(IDENTITY)) for d in r["distractors"]]
    return render_scene(make_background(r["background"]), layers), prims, units, aff


def render_record(rec) -> Image.Image:
    img, *_ = render_clean(rec)
    return compress(img, rec["compression"])


def render_analysis(rec):
    """Clean render for recounting: flat background far from every palette color, no distractors,
    rotation 0, centered, no compression."""
    fam, prims, _ = scene_for(rec)
    r = rec["render"]
    params = r["params"]
    colors = fam.colors(params)
    bg = max(ANALYSIS_BGS, key=lambda c: min(color_dist(c, x) for x in colors))
    if fam.fixed_frame:
        aff = Affine.from_params(IDENTITY)
    else:
        aff = Affine(IMG / 2, IMG / 2, r["placement"]["scale"], 0.0)
    img = render_scene(Image.new("RGB", (IMG, IMG), bg), [(prims, aff)])
    return np.asarray(img), aff


def unit_geometry(units, aff):
    pts = aff.apply([u["point"] for u in units]) if units else np.zeros((0, 2))
    sizes = [u["size"] * aff.scale for u in units]
    med = statistics.median(sizes) if sizes else 0.0
    order = reading_order(pts.tolist(), med)
    points = [[round(float(pts[i, 0]), 2), round(float(pts[i, 1]), 2)] for i in order]
    masks = [rle_encode(rasterize_mask(units[i]["mask"], aff, units[i].get("px_width"))) for i in order]
    return points, masks, (min(sizes) if sizes else 0.0), med


def finalize(rec, out_root=DATA, write=True):
    """Render, fill geometry fields, verify the count, save the PNG. Raises Rejected on failure."""
    rec = json.loads(json.dumps(to_jsonable(rec)))  # render from exactly what is stored
    fam = GENERATORS[rec["render"]["generator"]]
    clean, prims, units, aff = render_clean(rec)
    points, masks, min_size, med_size = unit_geometry(units, aff)
    if units and min_size < MIN_UNIT_PX:
        raise Rejected(f"unit size {min_size:.1f}px < {MIN_UNIT_PX}")
    if rec["prior"] == "count" and len(points) != rec["answer"]:
        raise Rejected(f"{len(points)} units but answer {rec['answer']}")
    x0, y0, x1, y1 = image_bbox(local_extent(prims), aff)
    rec["points"], rec["masks_rle"] = points, masks
    rec["unit_size_px"] = round(med_size, 2)
    rec["object_bbox"] = [max(0, int(x0)), max(0, int(y0)), min(IMG, int(np.ceil(x1))), min(IMG, int(np.ceil(y1)))]
    if rec["prior"] == "count":
        analysis, aff_a = render_analysis(rec)
        recount = int(fam.recount(analysis, rec["render"]["params"], aff_a))
        rec["checks"]["independent_recount"] = recount
        if recount != rec["answer"]:
            raise Rejected(f"recount {recount} != answer {rec['answer']}")
    else:  # color: name the edited pixels (§5.3 / §10.2), reject if they don't read as the answer
        name = fam.verify_color(np.asarray(clean), rec["render"]["params"], aff)
        rec["checks"]["color_name"] = name
        if name != rec["answer"]:
            raise Rejected(f"color name {name} != answer {rec['answer']}")
    img = compress(clean, rec["compression"])
    if write:
        rec["image_sha256"] = save_png(img, out_root / rec["image"])
    return rec


# ---------------------------------------------------------------------------
# Job execution: one job = one pair or one single image
# ---------------------------------------------------------------------------

def base_record(job, role, params, edit, render, compression, seed, attempt, familiar, delta):
    fam = GENERATORS[job["generator"]]
    fam_part = "" if job["family"] == job["dataset"] else f"-{job['family'].replace('_', '-')}"
    stem = f"{job['id_prefix']}{fam_part}-{job['idx']:06d}"
    rid = f"{stem}-{ROLE_SUFFIX[role]}"
    template = job["template"]
    bg = render["background"]
    return {
        "id": rid,
        "pair_id": stem if job["kind"] == "pair" else None,
        "dataset": job["dataset"],
        "pool_index": job.get("pool_index", job["idx"]),
        "level": "L0",
        "prior": job.get("prior", "count"),
        "data_type": job["data_type"],
        "role": role,
        "family": job["family"],
        "unit": fam.unit,
        "label": fam.label_for(params),
        "image": f"{job['out']}/images/{rid}.png",
        "question_template": template,
        "question": fam.question(params, template),
        "answer": params.get("answer", params.get("count")),
        "familiar_answer": familiar,
        "delta": delta,
        "points": None,
        "masks_rle": None,
        "unit_size_px": None,
        "object_bbox": None,
        "edit": {"op": edit.get("op", "none"), "params": edit.get("params", {}),
                 "null_edit": bool(edit.get("null_edit", False))},
        "source": {"dataset": None, "image_id": None, "ann_ids": [], "license": "generated",
                   "background": ({"dataset": "openimages_v7_train", "image_id": bg["file"].rsplit(".", 1)[0],
                                   "license": "CC-BY-2.0"} if bg["kind"] == "photo" else None)},
        "render": render,
        "compression": compression,
        "seed": seed,
        "attempt": attempt,
        "generator_version": GENERATOR_VERSION,
        "checks": {"independent_recount": None, "sam3_count": None, "ocr_count": None, "polygon_sides": None,
                   "color_name": None, "audit": None},
    }


def common_render(fam, rng, layouts, distractors=True):
    """Shared render parameters for all images of one item (pairs share everything, §4.1)."""
    scenes = [fam.scene(p) for p in layouts]
    if fam.fixed_frame:
        placement = dict(IDENTITY)
        union = [0, 0, IMG, IMG]
    else:
        exts = [local_extent(s[0]) for s in scenes]
        min_unit = min(u["size"] for s in scenes for u in s[1]) if any(s[1] for s in scenes) else 1.0
        placement = sample_placement(rng, exts, min_unit, **fam.placement_kw)
        aff = Affine.from_params(placement)
        boxes = [image_bbox(e, aff) for e in exts]
        union = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
    background = sample_background(rng, fam.main_color(layouts[0]))
    ds = [] if (fam.fixed_frame or not distractors) else \
        sample_distractors(rng, union, fam.distractor_exclude, avoid_colors=fam.colors(layouts[0]))
    return {"placement": placement, "background": background, "distractors": ds}


def attempt_job(job, attempt):
    fam = GENERATORS[job["generator"]]
    seed = item_seed(job["dataset"], job["family"], job["idx"], attempt)
    rng = rng_for(seed)
    if job["kind"] == "single":
        params = fam.sample(rng, count=job["count"], **job.get("sample_kw", {}))
        render = {"generator": job["generator"], "params": params, **common_render(fam, rng, [params], job.get("distractors", True))}
        compression = sample_compression(rng)
        return [base_record(job, "neutral", params, {}, render, compression, seed, attempt, None, None)]

    # Pair: canonical/counterfactual (conflict) or original/edited (neutral, Exp 2).
    # Color generators ignore count/delta and choose the recolor themselves.
    first = fam.sample(rng, count=job.get("count"), **job.get("sample_kw", {}))
    if "delta" in job:
        delta = job["delta"]
    else:
        feasible = fam.feasible_deltas(first, tuple(job.get("deltas", DELTAS)))
        if not feasible:
            raise Rejected("no feasible delta")
        delta = int(feasible[int(rng.integers(len(feasible)))])
    second, edit = fam.counterfactual(first, delta, rng)
    shared = common_render(fam, rng, [first, second], job.get("distractors", True))
    compression = sample_compression(rng)
    roles = job.get("roles", ("canonical", "counterfactual"))
    familiar = first.get("answer", first.get("count")) if job["data_type"] == "conflict" else None
    first_edit = fam.null_edit(first) if hasattr(fam, "null_edit") else {}
    color = job.get("prior") == "color"
    d2 = None if color else second["count"] - first["count"]
    recs = []
    for role, params, e, d in ((roles[0], first, first_edit, None if color else 0), (roles[1], second, edit, d2)):
        render = {"generator": job["generator"], "params": params, **shared}
        recs.append(base_record(job, role, params, e, render, compression, seed, attempt, familiar, d))
    return recs


def run_job(job):
    """Returns (records, n_rejected, last_error)."""
    errors = []
    for attempt in range(MAX_ATTEMPTS):
        try:
            recs = attempt_job(job, attempt)
            return [finalize(r) for r in recs], attempt, None
        except (Rejected, ValueError) as e:
            errors.append(str(e))
            if job.get("debug"):
                print(f"reject {job['family']}/{job['idx']} attempt {attempt}: {e}")
    raise RuntimeError(f"job {job['dataset']}/{job['family']}/{job['idx']} failed {MAX_ATTEMPTS} attempts: {errors[-3:]}")
