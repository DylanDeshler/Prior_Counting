import hashlib
import json

import numpy as np
import pytest

from answer_parsing import extract_answer, matches
from datagen.common import DATA, png_bytes, reading_order, rle_decode
from datagen.families import E1_FAMILIES, GENERATORS, HELD_OUT, TWIN_OF
from datagen.items import render_clean, render_record, run_job

PAIR_GENERATORS = E1_FAMILIES + HELD_OUT + ["clock_numerals"]


def pair_job(gen, i):
    job = {"kind": "pair", "generator": gen, "family": gen, "dataset": "test", "id_prefix": "test", "idx": i,
           "template": ("how_many", "count_the")[i % 2], "data_type": "conflict", "out": "test"}
    if gen == "clock_numerals":
        job["deltas"] = (-1, -2, -3)
    return job


@pytest.fixture(scope="module")
def records():
    recs = []
    for gen in PAIR_GENERATORS:
        for i in range(3):
            recs += run_job(pair_job(gen, i))[0]
    for fam, twin in TWIN_OF.items():
        lo, hi = GENERATORS[fam].count_range
        for i, c in enumerate((lo, (lo + hi) // 2, hi)):
            recs += run_job({"kind": "single", "generator": twin, "family": fam, "dataset": "test_n", "id_prefix": "t-n",
                             "idx": i, "count": c, "template": "how_many", "data_type": "neutral", "out": "test"})[0]
    for i, c in enumerate((12, 51)):
        recs += run_job({"kind": "single", "generator": "shared_shapes", "family": "shared_range", "dataset": "test_s",
                         "id_prefix": "t-s", "idx": i, "count": c, "template": "count_the", "data_type": "shared_range",
                         "out": "test", "distractors": False})[0]
    return recs


def test_counts_verified(records):
    for r in records:
        assert r["checks"]["independent_recount"] == r["answer"] == len(r["points"]) == len(r["masks_rle"]), r["id"]
        assert r["unit_size_px"] >= 12


def test_conflict_pairs(records):
    for r in records:
        if r["role"] == "counterfactual":
            assert r["answer"] != r["familiar_answer"] and r["delta"] == r["answer"] - r["familiar_answer"]


def test_points_inside_masks(records):
    for r in records:
        for (x, y), m in zip(r["points"], r["masks_rle"]):
            assert rle_decode(m)[int(y), int(x)], r["id"]


def test_determinism(records):
    for r in records[::5]:
        r = json.loads(json.dumps(r))
        assert hashlib.sha256(png_bytes(render_record(r))).hexdigest() == r["image_sha256"]
        assert hashlib.sha256((DATA / r["image"]).read_bytes()).hexdigest() == r["image_sha256"]


def test_pair_edit_confined_to_object(records):
    by = {}
    for r in records:
        if r["pair_id"]:
            by.setdefault(r["pair_id"], []).append(r)
    for a, b in by.values():
        assert a["compression"] == b["compression"]
        ia, ib = np.asarray(render_clean(a)[0], int), np.asarray(render_clean(b)[0], int)
        ys, xs = np.nonzero(np.abs(ia - ib).max(axis=2))
        if len(xs):
            box = [min(a["object_bbox"][k], b["object_bbox"][k]) for k in (0, 1)] + \
                  [max(a["object_bbox"][k], b["object_bbox"][k]) for k in (2, 3)]
            assert xs.min() >= box[0] - 2 and ys.min() >= box[1] - 2 and xs.max() <= box[2] + 2 and ys.max() <= box[3] + 2


def test_reading_order():
    pts = [[50, 12], [10, 10], [30, 31], [10, 30]]
    assert reading_order(pts, 10) == [1, 0, 3, 2]


def test_export_targets_parse(records):
    from datagen.export import target_text
    for r in records:
        for fmt in ("number", "points", "list", "randpoints"):
            t = target_text(r, fmt, "qwen3.5", "rel1000")
            assert matches(extract_answer(t), r["answer"])
            if fmt in ("points", "randpoints"):
                pts = json.loads(t.split("\nTotal:")[0])
                assert len(pts) == r["answer"] and all(0 <= v <= 1000 for p in pts for v in p["point_2d"])
