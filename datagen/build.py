"""Build the L0 datasets (§2, §4, §5.3, §6) in parallel on CPU.

Steps (each writes <dir>/images/*.png + <dir>/records.jsonl and reports/build_<step>.json):
    shared       shared range block (1,024)                       shared_range/
    e1           E1 conflict pool (3,584 pairs), neutral pool (7,168), dice pool (3,584 pairs),
                 synthetic validation (1,024 per data type)       e1/{conflict,neutral,dice,val_*}/
    t0           T0 count: traffic lights + snowflakes (500 + 500 pairs); T0 color (500 pairs)
    e2           Exp 2 L0 count conflict (stop signs + numeral clocks), 4N pairs (N-pair arms are prefixes,
                 §5.8)                                                              e2/L0/count/
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
E2_VOLUME = 4  # C-L0x4 / C-L2x4 volume arms (§5.8)
E2_COUNT_FAMILIES = {"stop_sign": (-2, -1, 1, 2), "clock_numerals": (-1, -2, -3)}  # §5.2 Δ per family
T2_COLOR_PAIRS = 500
STEPS = ["shared", "e1", "t0", "e2", "l2"]


def workers():
    # Leave headroom for eval/training jobs sharing the machine.
    return max(1, (os.cpu_count() or 2) - 2)


def split_evenly(total, k):
    return [total // k + (1 if i < total % k else 0) for i in range(k)]


def run_jobs(jobs, out_dir, step):
    """Run jobs in parallel; write records.jsonl (appending other datasets already in out_dir is the caller's job)."""
    t = time.time()
    records, retries, dropped = [], Counter(), Counter()
    with ProcessPoolExecutor(workers()) as ex:
        for job, (recs, attempts, err) in zip(jobs, tqdm(ex.map(run_job, jobs, chunksize=8), total=len(jobs), desc=step,
                                                          unit="item", smoothing=0.05)):
            records += recs
            retries[job["generator"]] += attempts
            if not recs:
                dropped[err.split(" (")[0][:60]] += 1
    records.sort(key=lambda r: (r["dataset"], r["pool_index"], r["id"]))
    return records, {"jobs": len(jobs), "images": len(records), "seconds": round(time.time() - t, 1),
                     "retries_by_generator": dict(retries), "dropped_items": dict(dropped)}


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


def build_e2():
    # Emoji were dropped as a source (2026-10-06): no L0 color or neutral crowds. Exp 2 color comes
    # from real photos at L2 (`l2` step); L0 neutral count waits for real-photo crowds.
    n = load_params()["e2_matched_n"] * E2_VOLUME
    conflict, s = run_jobs(e2_count_conflict_jobs(n), "e2/L0/count", "e2_count_conflict")
    write_pool("e2/L0/count", conflict, "e2_count", s)


def renumber(records):
    """Contiguous pool_index per dataset after dropped photo items, so N-pair prefixes stay balanced."""
    for ds in {r["dataset"] for r in records}:
        old = sorted({r["pool_index"] for r in records if r["dataset"] == ds})
        new = {o: i for i, o in enumerate(old)}
        for r in records:
            if r["dataset"] == ds:
                r["pool_index"] = new[r["pool_index"]]
    return records


def build_l2(n_pairs=None, t2_pairs=None):
    """L2 color from real photos (§5.5): conflict + neutral 4N pairs each, and T2 color."""
    from . import coco
    from . import photo_color as pc
    from .colorsets import OUT as color_objects
    if not (coco.DIR / "lvis_val.pkl").exists() or not color_objects.exists():
        print("l2: skipped (photo sources missing: run `python -m datagen sources`)")
        return
    analyzed = pc.analyze_all(pc.candidates())
    n = n_pairs or load_params()["e2_matched_n"] * E2_VOLUME
    recs, stats = [], {}
    for mode in ("conflict", "neutral"):
        jobs, per_obj = pc.color_jobs(analyzed, f"e2_L2_color_{mode}", "e2/L2/color", mode, n, "train")
        r, s = run_jobs(jobs, "e2/L2/color", f"e2_L2_color_{mode}")
        recs += renumber(r)
        stats[mode] = {**s, "usable_instances_per_object": per_obj,
                       "unique_source_photos": len({x["source"]["image_id"] for x in r})}
    write_pool("e2/L2/color", recs, "e2_L2_color", {**stats, "seconds": sum(v["seconds"] for v in stats.values()),
               "retries_by_generator": {"photo_color": sum(v["retries_by_generator"].get("photo_color", 0) for v in stats.values())}})
    jobs, per_obj = pc.color_jobs(analyzed, "t2_color", "tests/T2/color", "conflict", t2_pairs or T2_COLOR_PAIRS, "heldout")
    r, s = run_jobs(jobs, "tests/T2/color", "t2_color")
    write_pool("tests/T2/color", renumber(r), "t2_color", {**s, "usable_instances_per_object": per_obj,
               "unique_source_photos": len({x["source"]["image_id"] for x in r})})


def preflight(steps):
    """Fail fast, in the main process, with the data dir in the message (not deep inside a worker)."""
    import os
    from .common import DATA, SOURCES
    print(f"Data directory: {DATA}")
    rendered = [s for s in steps if s in ("shared", "e1", "t0", "e2")]
    bg = SOURCES / "openimages_bg"
    if rendered and os.environ.get("COUNTERPOINT_PREVIEW") != "1" and not any(bg.glob("*.jpg")):
        raise SystemExit(f"No background photos in {bg}.\n"
                         f"  If you built before with --data <dir>, pass the same --data again.\n"
                         f"  Otherwise run `python -m datagen{' --data ' + str(DATA) if 'COUNTERPOINT_DATA' in os.environ else ''} sources` first.")
    if rendered and load_params()["shared_block_max"] is None:
        raise SystemExit(f"{DATA / 'params.json'} has no shared_block_max: run `python -m datagen sources` (same --data).")


STEP_POOLS = {"shared": ["shared_range"], "e1": ["e1/conflict", "e1/neutral", "e1/dice", "e1/val_conflict",
               "e1/val_neutral", "e1/val_shared"], "t0": ["tests/T0/count"], "e2": ["e2/L0/count"],
               "l2": ["e2/L2/color", "tests/T2/color"]}


def fingerprint(step):
    """Hash of the generator code + the parameters a step depends on. If it is unchanged and the
    step's pools exist, the step's output would be byte-identical, so it can be skipped."""
    import hashlib
    from pathlib import Path
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.rglob("*.py")):
        h.update(f.name.encode() + f.read_bytes())
    p = load_params()
    h.update(json.dumps([step, p["shared_block_max"], p["e2_matched_n"], E1_PAIRS, E1_VAL_IMAGES, SHARED_N,
                         T0_COUNT_PAIRS, T2_COLOR_PAIRS]).encode())
    return h.hexdigest()[:16]


def up_to_date(step):
    stamp = REPORTS / f"step_{step}.json"
    return (stamp.exists() and json.loads(stamp.read_text()).get("fingerprint") == fingerprint(step)
            and all((DATA / pool / "records.jsonl").exists() for pool in STEP_POOLS[step]))


def build(steps=None, skip_up_to_date=False):
    preflight(steps or STEPS)
    for step in steps or STEPS:
        if skip_up_to_date and up_to_date(step):
            print(f"{step}: up to date, skipped")
            continue
        {"shared": build_shared, "e1": build_e1, "t0": build_t0, "e2": build_e2, "l2": build_l2}[step]()
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / f"step_{step}.json").write_text(json.dumps({"fingerprint": fingerprint(step)}))
