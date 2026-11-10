"""Training exports (§3.5, §3.6): exports/<arm>/<model>/<format>/{train,val,train_rl}.jsonl

SFT rows use the messages + images format read by both ms-swift and LLaMA-Factory. RL (GRPO)
rows drop the assistant turn and add the ground truth (answer, points) as extra columns for the
reward function. Output is byte-identical for identical records: rows are ordered by a hash of
(arm, id), JSON is written with fixed key order, and coordinates are integers.
"""

import hashlib
import json

from PIL import Image
from tqdm import tqdm

from .common import DATA, IMG, item_seed, load_params, read_jsonl, rng_for, write_jsonl

MODELS = ("qwen3.5", "qwen2.5-vl")
EXPORTS = DATA / "exports"
GRAY = EXPORTS / "_gray.png"

# Exp 1 arms (§4.4): (train pools, val pools, format, blind, family filter)
E1_ARMS = {
    "conflict-count": (["e1/conflict"], ["e1/val_conflict"], "number", False, None),
    "conflict-point": (["e1/conflict"], ["e1/val_conflict"], "points", False, None),
    "neutral-count": (["e1/neutral"], ["e1/val_neutral"], "number", False, None),
    "neutral-point": (["e1/neutral"], ["e1/val_neutral"], "points", False, None),
    "neutral-list": (["e1/neutral"], ["e1/val_neutral"], "list", False, None),
    "conflict-blind": (["e1/conflict"], ["e1/val_conflict"], "number", True, None),
    "dice-only": (["e1/dice"], ["e1/val_conflict"], "number", False, "die"),
    "neutral-randpoint": (["e1/neutral"], ["e1/val_neutral"], "randpoints", False, None),
    # Distillation arms (H3c): the same images with both targets, half each, chosen per item:
    # pointing prompt -> point list + "Total: {N}", or the plain question -> "{N}".
    "conflict-mixed": (["e1/conflict"], ["e1/val_conflict"], "mixed", False, None),
    "neutral-mixed": (["e1/neutral"], ["e1/val_neutral"], "mixed", False, None),
    # Real-photo counting (unedited LVIS crowds) with ground-truth points: grounding on photos. Real photos
    # only: no synthetic shared block (REAL_ONLY), so they test pointing without synthetic layouts.
    "real-count": (["real/count"], [], "number", False, None),
    "real-point": (["real/count"], [], "points", False, None),
    "real-mixed": (["real/count"], [], "mixed", False, None),
    # Real-photo PART counting (Pascal-Part animal legs; parts.py): natural visible-leg counts, and edit pairs
    # (a leg added or removed, partner with a background null edit). Real photos only, like the real- arms.
    "part-neutral-point": (["parts/count"], [], "points", False, None),
    "part-conflict-point": (["parts/conflict"], [], "points", False, None),
    "part-conflict-count": (["parts/conflict"], [], "number", False, None),
}
SHARED, VAL_SHARED = "shared_range", "e1/val_shared"
REAL_ONLY = ("real-", "part-")
# Out-of-distribution validation for every arm: real-photo crops held out of real/count (never trained on,
# disjoint from tests/R/count). Its loss is logged every 50 steps during training, so overfitting to the
# synthetic layouts shows up as a rising real-photo loss; the checkpoint sweep scores its accuracy.
REAL_POOL, REAL_VAL, REAL_VAL_N = "real/count", "real/val", 300


def real_val_ids():
    """Write real/val/records.jsonl (REAL_VAL_N records of real/count chosen by a stable hash of the id) and
    return their ids. None if the real pool is not built."""
    path = DATA / REAL_POOL / "records.jsonl"
    if not path.exists():
        return None
    recs = read_jsonl(path)
    pick = sorted(recs, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[:REAL_VAL_N]
    out = DATA / REAL_VAL / "records.jsonl"
    if not out.exists() or [r["id"] for r in read_jsonl(out)] != [r["id"] for r in pick]:
        out.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(out, pick)
    return {r["id"] for r in pick}


def model_point(model, x, y, fmt):
    if model == "qwen3.5" and fmt == "rel1000":
        return [round(x / IMG * 1000), round(y / IMG * 1000)]
    return [round(x), round(y)]


def target_text(rec, fmt, model, point_fmt):
    ans = rec["answer"]
    if rec["prior"] == "color":
        return f"{{{ans}}}"
    if fmt == "number":
        return f"{{{ans}}}"
    label = rec["label"]
    if fmt == "list":
        return ", ".join(f"{label} {i}" for i in range(1, ans + 1)) + f". Total: {{{ans}}}"
    if fmt == "points":
        pts = rec["points"]
    elif fmt == "randpoints":
        rng = rng_for(item_seed("randpoint", rec["id"]))
        pts = rng.uniform(0, IMG, size=(ans, 2)).tolist()
    else:
        raise ValueError(fmt)
    items = [{"point_2d": model_point(model, x, y, point_fmt), "label": label} for x, y in pts]
    return json.dumps(items) + f"\nTotal: {{{ans}}}"


def rows_for(records, fmt, model, point_fmt, blind=False):
    sft, rl = [], []
    for r in records:
        image = str(GRAY if (blind and r["data_type"] != "shared_range") else (DATA / r["image"]).resolve())
        question, item_fmt = r["question"], fmt
        if fmt == "mixed":
            from answer_parsing import rewrite_prompt  # repo root: same wording as the eval's point-total prompt
            if r["prior"] == "count" and item_seed("mixed", r["id"]) % 2:
                question, item_fmt = rewrite_prompt(r["question"], "point-total"), "points"
            else:
                item_fmt = "number"
        user = {"role": "user", "content": "<image>" + question}
        sft.append({"id": r["id"], "messages": [user, {"role": "assistant", "content": target_text(r, item_fmt, model, point_fmt)}],
                    "images": [image]})
        rl_row = {"id": r["id"], "messages": [user], "images": [image], "answer": r["answer"],
                  "familiar_answer": r["familiar_answer"], "prior": r["prior"], "label": r.get("label")}
        if r["prior"] == "count":
            rl_row["points"] = [model_point(model, x, y, point_fmt) for x, y in r["points"]]
        rl.append(rl_row)
    return sft, rl


def write_rows(path, rows, arm):
    rows = sorted(rows, key=lambda row: (item_seed(arm, row["id"]), row["id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load(pool, family=None, n_pairs=None, data_type=None):
    recs = read_jsonl(DATA / pool / "records.jsonl")
    if family:
        recs = [r for r in recs if r["family"] == family]
    if data_type:
        recs = [r for r in recs if r["data_type"] == data_type]
    if n_pairs is not None:
        recs = [r for r in recs if r["pool_index"] < n_pairs]
    return recs


def export_arm(arm, train, val, fmt, blind, point_fmt):
    for model in tqdm(MODELS, desc=f"export {arm}/{fmt}", unit="model", leave=False):
        out = EXPORTS / arm / model / fmt
        sft, rl = rows_for(train, fmt, model, point_fmt, blind)
        write_rows(out / "train.jsonl", sft, arm)
        write_rows(out / "train_rl.jsonl", rl, arm)
        if val:
            write_rows(out / "val.jsonl", rows_for(val, fmt, model, point_fmt, blind)[0], arm)
    return len(train)


def export_e1(point_fmt, only=None):
    shared, val_shared = load(SHARED), load(VAL_SHARED)
    held = real_val_ids()
    real_val = load(REAL_VAL) if held else None
    arms = {a: v for a, v in E1_ARMS.items() if only is None or a.startswith(only)}
    for arm, (pools, val_pools, fmt, blind, fam) in tqdm(arms.items(), desc="Exp 1 arms", unit="arm"):
        if not all((DATA / p / "records.jsonl").exists() for p in pools):
            print(f"{arm:<20} skipped: pool not built ({', '.join(pools)})")
            continue
        train = [r for p in pools for r in load(p, fam) if not (held and r["id"] in held)]
        if not arm.startswith(REAL_ONLY):
            train += shared
        # Validation loss on held-out real photos when available (out of distribution for the synthetic
        # arms); in-distribution synthetic accuracy is measured after training by the eval suite.
        val = real_val if real_val else [r for p in val_pools for r in load(p, fam)] + val_shared
        n = export_arm(arm, train, val, fmt, blind, point_fmt)
        print(f"{arm:<20} {fmt:<11} {n} train rows")


def export_e2(point_fmt):
    """Exp 2 arms per level (§5.8) in both count formats; color targets are always {color}."""
    n = load_params()["e2_matched_n"]
    shared = load(SHARED)
    for level in ("L0", "L2"):
        arms = {f"C-{level}": ("conflict", n), f"N-{level}": ("neutral", n),
                f"C-{level}x4": ("conflict", None), f"N-{level}x4": ("neutral", None)}
        for arm, (dtype, n_pairs) in tqdm(arms.items(), desc=f"Exp 2 {level} arms", unit="arm"):
            export_e2_arm(arm, level, dtype, n_pairs, shared, point_fmt)


def export_e2_arm(arm, level, dtype, n_pairs, shared, point_fmt):
    varying = [r for pool in (f"e2/{level}/count", f"e2/{level}/color") if (DATA / pool / "records.jsonl").exists()
               for r in load(pool, data_type=dtype, n_pairs=n_pairs)]
    if not varying:
        print(f"{arm:<20} skipped: no {dtype} items at {level}")
        return
    train = varying + shared
    for fmt in ("number", "points"):
        export_arm(arm, train, None, fmt, False, point_fmt)
    print(f"{arm:<20} {len(train)} train rows (number + points)")


def export(which=("e1", "e2")):
    EXPORTS.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (IMG, IMG), (128, 128, 128)).save(GRAY, format="PNG", compress_level=6)
    point_fmt = load_params()["qwen35_point_format"]
    if "e1" in which:
        export_e1(point_fmt)
    elif "real" in which:  # only the real-photo arms (safe while other arms' files are being trained on)
        export_e1(point_fmt, only=REAL_ONLY)
    if "e2" in which:
        export_e2(point_fmt)
