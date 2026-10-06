"""Build the L0 datasets (§2, §4, §5.3, §6) in parallel on CPU.

Steps (each writes <dir>/images/*.png + <dir>/records.jsonl and reports/build_<step>.json):
    shared       shared range block (1,024)                       shared_range/
    e1           E1 conflict pool (3,584 pairs), neutral pool (7,168), dice pool (3,584 pairs),
                 synthetic validation (1,024 per data type)       e1/{conflict,neutral,dice,val_*}/
    t0           T0 count: traffic lights + snowflakes (500 + 500 pairs); T0 color (500 pairs)
    e2           Exp 2 L0: count conflict (stop signs + numeral clocks) and neutral crowds, color conflict
                 and neutral; 4N pairs each (the N-pair arms are prefixes, §5.8)   e2/L0/{count,color}/
"""

import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

from .common import DATA, REPORTS, item_seed, load_params, read_jsonl, write_jsonl
from .families import E1_FAMILIES, GENERATORS, TWIN_OF
from .items import run_job

TEMPLATES = ("how_many", "count_the")
E1_PAIRS, E1_VAL_IMAGES, SHARED_N = 3584, 1024, 1024
T0_COUNT_PAIRS = {"traffic_light": 500, "snowflake": 500}
T0_COLOR_PAIRS = 500
E2_VOLUME = 4  # C-L0x4 / C-L2x4 volume arms (§5.8)
E2_COUNT_FAMILIES = {"stop_sign": (-2, -1, 1, 2), "clock_numerals": (-1, -2, -3)}  # §5.2 Δ per family
STEPS = ["shared", "e1", "t0", "e2"]


def workers():
    # Leave headroom for eval/training jobs sharing the machine.
    return max(1, (os.cpu_count() or 2) - 2)


def split_evenly(total, k):
    return [total // k + (1 if i < total % k else 0) for i in range(k)]


def run_jobs(jobs, out_dir, step):
    """Run jobs in parallel; write records.jsonl (appending other datasets already in out_dir is the caller's job)."""
    t = time.time()
    records, retries = [], Counter()
    with ProcessPoolExecutor(workers()) as ex:
        for recs, attempts, _ in tqdm(ex.map(run_job, jobs, chunksize=8), total=len(jobs), desc=step):
            records += recs
            retries[recs[0]["render"]["generator"]] += attempts
    records.sort(key=lambda r: (r["dataset"], r["pool_index"], r["id"]))
    return records, {"jobs": len(jobs), "images": len(records), "seconds": round(time.time() - t, 1),
                     "retries_by_generator": dict(retries)}


def write_pool(out_dir, records, step, stats):
    write_jsonl(DATA / out_dir / "records.jsonl", records)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"build_{step}.json").write_text(json.dumps(stats, indent=1))
    rej = sum(stats["retries_by_generator"].values())
    print(f"{out_dir}: {len(records)} images in {stats['seconds']}s, {rej} rejected attempts")


def pair_job(generator, family, dataset, out, idx, data_type="conflict", **kw):
    prefix = dataset.replace("_", "-").lower()
    return {"kind": "pair", "generator": generator, "family": family, "dataset": dataset, "id_prefix": prefix,
            "idx": idx, "template": TEMPLATES[idx % 2], "data_type": data_type, "out": out, **kw}


def single_job(generator, family, dataset, out, idx, count, data_type="neutral", **kw):
    prefix = dataset.replace("_", "-").lower()
    return {"kind": "single", "generator": generator, "family": family, "dataset": dataset, "id_prefix": prefix,
            "idx": idx, "count": count, "template": TEMPLATES[idx % 2], "data_type": data_type, "out": out, **kw}


# ---------------------------------------------------------------------------
# Job lists
# ---------------------------------------------------------------------------

def shared_jobs(dataset, out, n=None):
    n = n or SHARED_N
    U = load_params()["shared_block_max"]
    if U is None:
        raise SystemExit("shared_block_max unknown: run `python -m datagen sources` (and tools/check_length.py) first.")
    counts = [12 + i % (U - 11) for i in range(n)]
    np.random.default_rng(item_seed(dataset, "counts")).shuffle(counts)
    return [single_job("shared_shapes", "shared_range", dataset, out, i, c, data_type="shared_range",
                       distractors=False) for i, c in enumerate(counts)]


def conflict_jobs(dataset, out, total, families=E1_FAMILIES):
    jobs = []
    for fam, q in zip(families, split_evenly(total, len(families))):
        jobs += [pair_job(fam, fam, dataset, out, i) for i in range(q)]
    return jobs


def twin_jobs(conflict_records, dataset, out):
    """Neutral twins with the conflict pool's count histogram, family by family (§2, §10.1)."""
    by_family = defaultdict(list)
    for r in conflict_records:
        by_family[r["family"]].append(r["answer"])
    jobs = []
    for fam in E1_FAMILIES:
        counts = sorted(by_family[fam])
        np.random.default_rng(item_seed(dataset, fam, "counts")).shuffle(counts)
        jobs += [single_job(TWIN_OF[fam], fam, dataset, out, i, c) for i, c in enumerate(counts)]
    return jobs


def e2_count_conflict_jobs(n_pairs):
    """Equal family quotas; interleaved so every prefix of 2k pairs has k per family (§5.8 subsets)."""
    fams = list(E2_COUNT_FAMILIES)
    jobs = []
    for i in range(n_pairs):
        fam = fams[i % len(fams)]
        j = pair_job(fam, fam, "e2_L0_count_conflict", "e2/L0/count", i // len(fams), deltas=E2_COUNT_FAMILIES[fam])
        j["pool_index"] = i
        j["template"] = TEMPLATES[(i // len(fams)) % 2]
        jobs.append(j)
    return jobs


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def build_shared():
    recs, stats = run_jobs(shared_jobs("shared_range", "shared_range"), "shared_range", "shared")
    write_pool("shared_range", recs, "shared", stats)


def build_e1():
    conflict, s = run_jobs(conflict_jobs("e1_conflict", "e1/conflict", E1_PAIRS), "e1/conflict", "e1_conflict")
    write_pool("e1/conflict", conflict, "e1_conflict", s)
    neutral, s = run_jobs(twin_jobs(conflict, "e1_neutral", "e1/neutral"), "e1/neutral", "e1_neutral")
    write_pool("e1/neutral", neutral, "e1_neutral", s)
    dice, s = run_jobs([pair_job("die", "die", "e1_dice", "e1/dice", i) for i in range(E1_PAIRS)], "e1/dice", "e1_dice")
    write_pool("e1/dice", dice, "e1_dice", s)
    # Synthetic validation: same generators, disjoint seeds (dataset name is part of every seed).
    vc, s = run_jobs(conflict_jobs("e1_val_conflict", "e1/val_conflict", E1_VAL_IMAGES // 2), "e1/val_conflict", "e1_val_conflict")
    write_pool("e1/val_conflict", vc, "e1_val_conflict", s)
    vn, s = run_jobs(twin_jobs(vc, "e1_val_neutral", "e1/val_neutral"), "e1/val_neutral", "e1_val_neutral")
    write_pool("e1/val_neutral", vn, "e1_val_neutral", s)
    vs, s = run_jobs(shared_jobs("e1_val_shared", "e1/val_shared", E1_VAL_IMAGES), "e1/val_shared", "e1_val_shared")
    write_pool("e1/val_shared", vs, "e1_val_shared", s)


def build_t0():
    jobs = []
    for fam, n in T0_COUNT_PAIRS.items():
        jobs += [pair_job(fam, fam, "t0_count", "tests/T0/count", i) for i in range(n)]
    recs, s = run_jobs(jobs, "tests/T0/count", "t0_count")
    write_pool("tests/T0/count", recs, "t0_count", s)
    from .build_color import t0_color_jobs
    recs, s = run_jobs(t0_color_jobs(T0_COLOR_PAIRS), "tests/T0/color", "t0_color")
    write_pool("tests/T0/color", recs, "t0_color", s)


def build_e2():
    from .build_color import e2_color_jobs, e2_crowd_jobs
    n = load_params()["e2_matched_n"] * E2_VOLUME
    conflict, s1 = run_jobs(e2_count_conflict_jobs(n), "e2/L0/count", "e2_count_conflict")
    neutral, s2 = run_jobs(e2_crowd_jobs(conflict, n), "e2/L0/count", "e2_count_neutral")
    write_pool("e2/L0/count", conflict + neutral, "e2_count", {"conflict": s1, "neutral": s2,
               "retries_by_generator": {**s1["retries_by_generator"], **s2["retries_by_generator"]},
               "seconds": s1["seconds"] + s2["seconds"]})
    cc, s1 = run_jobs(e2_color_jobs(n, "conflict"), "e2/L0/color", "e2_color_conflict")
    cn, s2 = run_jobs(e2_color_jobs(n, "neutral"), "e2/L0/color", "e2_color_neutral")
    write_pool("e2/L0/color", cc + cn, "e2_color", {"conflict": s1, "neutral": s2,
               "retries_by_generator": {**s1["retries_by_generator"], **s2["retries_by_generator"]},
               "seconds": s1["seconds"] + s2["seconds"]})


def build(steps=None):
    for step in steps or STEPS:
        {"shared": build_shared, "e1": build_e1, "t0": build_t0, "e2": build_e2}[step]()
