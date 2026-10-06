"""Training exports (§3.5, §3.6): exports/<arm>/<model>/<format>/{train,val,train_rl}.jsonl

SFT rows use the messages + images format read by both ms-swift and LLaMA-Factory. RL (GRPO)
rows drop the assistant turn and add the ground truth (answer, points) as extra columns for the
reward function. Output is byte-identical for identical records: rows are ordered by a hash of
(arm, id), JSON is written with fixed key order, and coordinates are integers.
"""

import json

from PIL import Image

from .common import DATA, IMG, item_seed, load_params, read_jsonl, rng_for

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
}
SHARED, VAL_SHARED = "shared_range", "e1/val_shared"


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
        user = {"role": "user", "content": "<image>" + r["question"]}
        sft.append({"id": r["id"], "messages": [user, {"role": "assistant", "content": target_text(r, fmt, model, point_fmt)}],
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
    for model in MODELS:
        out = EXPORTS / arm / model / fmt
        sft, rl = rows_for(train, fmt, model, point_fmt, blind)
        write_rows(out / "train.jsonl", sft, arm)
        write_rows(out / "train_rl.jsonl", rl, arm)
        if val:
            write_rows(out / "val.jsonl", rows_for(val, fmt, model, point_fmt, blind)[0], arm)
    return len(train)


def export_e1(point_fmt):
    shared, val_shared = load(SHARED), load(VAL_SHARED)
    for arm, (pools, val_pools, fmt, blind, fam) in E1_ARMS.items():
        train = [r for p in pools for r in load(p, fam)] + shared
        val = [r for p in val_pools for r in load(p, fam)] + val_shared
        n = export_arm(arm, train, val, fmt, blind, point_fmt)
        print(f"{arm:<20} {fmt:<11} {n} train rows")


def export_e2(point_fmt):
    """Exp 2 L0 arms (§5.8) in both count formats; color targets are always {color}."""
    n = load_params()["e2_matched_n"]
    shared = load(SHARED)
    arms = {"C-L0": ("conflict", n), "N-L0": ("neutral", n), "C-L0x4": ("conflict", None), "N-L0x4": ("neutral", None)}
    for arm, (dtype, n_pairs) in arms.items():
        train = load("e2/L0/count", data_type=dtype, n_pairs=n_pairs) + \
            load("e2/L0/color", data_type=dtype, n_pairs=n_pairs) + shared
        for fmt in ("number", "points"):
            export_arm(arm, train, None, fmt, False, point_fmt)
        print(f"{arm:<20} {len(train)} train rows (number + points)")


def export(which=("e1", "e2")):
    EXPORTS.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (IMG, IMG), (128, 128, 128)).save(GRAY, format="PNG", compress_level=6)
    point_fmt = load_params()["qwen35_point_format"]
    if "e1" in which:
        export_e1(point_fmt)
    if "e2" in which:
        export_e2(point_fmt)
