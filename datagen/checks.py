"""Automated checks (§10.1) and generator validation (§10.2) for the L0 data.

`python -m datagen check` runs everything below and writes reports/ci_report.json; it exits
non-zero if any check fails. `python -m datagen audit` writes contact sheets for the render audit.
"""

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from .common import DATA, IMG, MIN_UNIT_PX, REPORTS, png_bytes, read_jsonl, rle_decode
from .families import E1_FAMILIES, HELD_OUT
from .items import render_clean, render_record

POOLS = ["shared_range", "e1/conflict", "e1/neutral", "e1/dice", "e1/val_conflict", "e1/val_neutral", "e1/val_shared",
         "tests/T0/count", "tests/T0/color", "e2/L0/count", "e2/L0/color"]
TRAIN_POOLS = ["shared_range", "e1/conflict", "e1/neutral", "e1/dice", "e2/L0/count", "e2/L0/color"]

# Allowed Δ per family (§4.2, §5.2). Fixed-canonical families use exactly these values, uniformly.
FIXED_DELTAS = {"clock": {-3, -2, -1, 1, 2, 3}, "stop_sign": {-3, -2, -1, 1, 2, 3}, "calendar": {-3, -2, -1, 1, 2, 3},
                "star": {-2, -1, 1, 2, 3}, "piano": {-3, -2, -1, 1}, "rubik": {-2, -1, 1, 2, 3},
                "traffic_light": {-2, -1, 1, 2, 3}, "snowflake": {-3, -2, -1, 1, 2, 3}}
E2_DELTAS = {"stop_sign": {-2, -1, 1, 2}, "clock_numerals": {-1, -2, -3}}
VARIABLE_DELTAS = {-3, -2, -1, 1, 2, 3}  # die, card, domino: feasible subset per item


class Report:
    def __init__(self):
        self.results = {}

    def add(self, name, ok, **details):
        self.results[name] = {"ok": bool(ok), **details}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {details.get('summary', '')}" if details.get("summary") else ""))

    @property
    def ok(self):
        return all(r["ok"] for r in self.results.values())


def load_pools(pools=POOLS):
    out = {}
    for p in tqdm([p for p in pools if (DATA / p / "records.jsonl").exists()], desc="loading records", unit="pool",
                  leave=False):
        out[p] = read_jsonl(DATA / p / "records.jsonl")
    return out


def pairs_of(records):
    by = defaultdict(list)
    for r in records:
        if r["pair_id"]:
            by[r["pair_id"]].append(r)
    return by


# ---------------------------------------------------------------------------
# §10.1 checks
# ---------------------------------------------------------------------------

def check_records(rep, pools):
    bad = Counter()
    examples = defaultdict(list)
    for pool, recs in pools.items():
        for r in tqdm(recs, desc=f"records {pool}", unit="rec", leave=False):
            problems = []
            if r["unit_size_px"] is not None and r["points"] and r["prior"] == "count":
                if r["unit_size_px"] < MIN_UNIT_PX:
                    problems.append("unit_size")
            if r["prior"] == "count":
                if len(r["points"]) != r["answer"] or len(r["masks_rle"]) != r["answer"]:
                    problems.append("n_points")
                if r["checks"]["independent_recount"] != r["answer"]:
                    problems.append("recount")
            elif r["checks"]["color_name"] != r["answer"]:
                problems.append("color_name")
            if not (DATA / r["image"]).exists():
                problems.append("missing_image")
            for p in problems:
                bad[p] += 1
                if len(examples[p]) < 5:
                    examples[p].append(r["id"])
    n = sum(len(v) for v in pools.values())
    rep.add("records: schema, unit size >= 12px, recount/color check, images exist", not bad,
            summary=f"{n} records, problems {dict(bad)}", problems=dict(bad), examples=dict(examples))


def check_points_in_masks(rep, pools, tol_px=0):
    bad, total = [], 0
    for pool, recs in pools.items():
        for r in tqdm(recs, desc=f"points in masks {pool}", unit="rec", leave=False):
            if r["prior"] != "count":
                continue
            for (x, y), m in zip(r["points"], r["masks_rle"]):
                total += 1
                mask = rle_decode(m)
                xi, yi = min(IMG - 1, int(x)), min(IMG - 1, int(y))
                if not mask[yi, xi]:
                    bad.append(r["id"])
                    break
    rep.add("points inside their unit masks", not bad, summary=f"{total} points, {len(bad)} records failing",
            examples=bad[:10])


def check_determinism(rep, pools, n=100, seed=0):
    recs = [r for p in pools.values() for r in p if r["level"] == "L0"]
    rng = np.random.default_rng(seed)
    sample = [recs[i] for i in rng.choice(len(recs), size=min(n, len(recs)), replace=False)]
    bad = []
    for r in tqdm(sample, desc="determinism (re-render)", unit="img", leave=False):
        h = hashlib.sha256(png_bytes(render_record(r))).hexdigest()
        on_disk = hashlib.sha256((DATA / r["image"]).read_bytes()).hexdigest()
        if not (h == r["image_sha256"] == on_disk):
            bad.append(r["id"])
    rep.add("determinism: regenerate random records from metadata", not bad,
            summary=f"{len(sample)} regenerated, {len(bad)} mismatched", examples=bad[:10])


def check_pair_integrity(rep, pools, max_pairs_per_pool=300):
    bad, n = [], 0
    for pool, recs in pools.items():
        pairs = [v for v in pairs_of(recs).values() if len(v) == 2]
        step = max(1, len(pairs) // max_pairs_per_pool)
        for a, b in tqdm(pairs[::step], desc=f"pair integrity {pool}", unit="pair", leave=False):
            n += 1
            if a["compression"] != b["compression"]:
                bad.append((a["pair_id"], "compression differs"))
                continue
            ia, *_ = render_clean(a)
            ib, *_ = render_clean(b)
            diff = np.abs(np.asarray(ia, int) - np.asarray(ib, int)).max(axis=2) > 0
            x0 = min(a["object_bbox"][0], b["object_bbox"][0]) - 2
            y0 = min(a["object_bbox"][1], b["object_bbox"][1]) - 2
            x1 = max(a["object_bbox"][2], b["object_bbox"][2]) + 2
            y1 = max(a["object_bbox"][3], b["object_bbox"][3]) + 2
            inside = np.zeros_like(diff)
            inside[max(0, y0):y1, max(0, x0):x1] = True
            if (diff & ~inside).any():
                bad.append((a["pair_id"], f"{int((diff & ~inside).sum())} px changed outside object bbox"))
    rep.add("pair integrity: edits confined to object bbox (pre-compression), shared compression", not bad,
            summary=f"{n} pairs checked, {len(bad)} failing", examples=bad[:10])


def check_distributions(rep, pools):
    problems = []
    hist = {}
    # Δ per family
    for pool in ("e1/conflict", "e1/dice", "e1/val_conflict", "tests/T0/count", "e2/L0/count"):
        recs = [r for r in pools.get(pool, []) if r["data_type"] == "conflict" and r["role"] == "counterfactual"]
        by = defaultdict(Counter)
        for r in recs:
            by[r["family"]][r["delta"]] += 1
        for fam, c in by.items():
            hist[f"{pool}:{fam}"] = dict(sorted(c.items()))
            allowed = E2_DELTAS[fam] if pool.startswith("e2") else FIXED_DELTAS.get(fam, VARIABLE_DELTAS)
            if set(c) - allowed:
                problems.append(f"{pool}:{fam} has Δ {sorted(set(c) - allowed)}")
            if fam in FIXED_DELTAS or pool.startswith("e2"):
                expected = sum(c.values()) / len(allowed)
                if sum(c.values()) >= 100 and any(abs(c.get(d, 0) - expected) > 0.35 * expected + 3 for d in allowed):
                    problems.append(f"{pool}:{fam} Δ not uniform: {dict(c)}")
    # Twin histograms equal conflict histograms, family by family
    for conf, neu in (("e1/conflict", "e1/neutral"), ("e1/val_conflict", "e1/val_neutral")):
        if conf in pools and neu in pools:
            for fam in E1_FAMILIES:
                a = Counter(r["answer"] for r in pools[conf] if r["family"] == fam)
                b = Counter(r["answer"] for r in pools[neu] if r["family"] == fam)
                if a != b:
                    problems.append(f"{neu}:{fam} count histogram != {conf}")
    # Exp 2: crowd histogram equals pooled conflict count histogram; quotas equal
    if "e2/L0/count" in pools:
        recs = pools["e2/L0/count"]
        a = Counter(r["answer"] for r in recs if r["data_type"] == "conflict")
        b = Counter(r["answer"] for r in recs if r["data_type"] == "neutral")
        if a != b:
            problems.append("e2/L0/count neutral crowd histogram != pooled conflict histogram")
        q = Counter(r["family"] for r in recs if r["data_type"] == "conflict")
        if len(set(q.values())) > 1:
            problems.append(f"e2 count family quotas unequal: {dict(q)}")
    # E1 family quotas: 398 or 399 pairs per family
    if "e1/conflict" in pools:
        q = Counter(r["family"] for r in pools["e1/conflict"] if r["role"] == "canonical")
        if max(q.values()) - min(q.values()) > 1:
            problems.append(f"e1 family quotas uneven: {dict(q)}")
    # Templates 50/50 within every dataset
    for pool, recs in pools.items():
        t = Counter(r["question_template"] for r in recs)
        if t and abs(t["how_many"] - t["count_the"]) > max(2, 0.02 * len(recs)):
            problems.append(f"{pool}: templates {dict(t)}")
    rep.add("distributions: Δ per family, twin/crowd histograms, quotas, 50/50 templates", not problems,
            summary="; ".join(problems[:5]), problems=problems, delta_histograms=hist)


def check_leakage(rep, pools):
    problems = []
    excl_file = DATA / "exclusions" / "vlmbias_categories.json"
    terms = json.loads(excl_file.read_text())["excluded_terms"] if excl_file.exists() else []
    color_file = DATA / "sources" / "color_objects.json"
    held_color, vcf = set(), set()
    if color_file.exists():
        co = json.loads(color_file.read_text())
        objs = co.get("objects", co if isinstance(co, list) else [])
        held_color = {o["name"] for o in objs if o.get("split") == "heldout"}
        vcf = set(co.get("vcf_objects", []))
    for pool in TRAIN_POOLS:
        for r in pools.get(pool, []):
            if r["family"] in HELD_OUT or r["render"]["generator"] in HELD_OUT:
                problems.append(f"{r['id']}: held-out family in training")
            q = r["question"].lower()
            if any(re.search(rf"\b{re.escape(t)}s?\b", q) for t in terms):
                problems.append(f"{r['id']}: question mentions an excluded VLMBias category")
            obj = r["render"]["params"].get("object")
            if obj and (obj in held_color or obj in vcf):
                problems.append(f"{r['id']}: held-out / Visual CounterFact color object {obj}")
    if not excl_file.exists():
        problems.append("exclusions/vlmbias_categories.json missing (run `python -m datagen sources`)")
    rep.add("leakage: no held-out families/objects or excluded categories in training", not problems,
            summary=f"{len(problems)} problems", examples=problems[:10])


def check_length_rule(rep):
    f = REPORTS / "length_check.json"
    if not f.exists():
        rep.add("length rule (§3.5) with both tokenizers", False, summary="run tools/check_length.py")
        return
    res = json.loads(f.read_text())
    rep.add("length rule (§3.5) with both tokenizers", res.get("pass"), summary=json.dumps(res.get("tokens", {})))


def check_exports(rep):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from answer_parsing import extract_answer, matches
    by_id = {r["id"]: r for recs in load_pools().values() for r in recs}
    problems, n = [], 0
    for path in tqdm(sorted((DATA / "exports").glob("*/*/*/train.jsonl")), desc="exports parse", unit="file", leave=False):
        fmt = path.parent.name
        for line in path.open():
            row = json.loads(line)
            n += 1
            rec = by_id.get(row["id"])
            target = row["messages"][1]["content"]
            if rec is None:
                problems.append(f"{path}: unknown id {row['id']}")
                continue
            if not matches(extract_answer(target), rec["answer"]):
                problems.append(f"{path}: {row['id']} parses to {extract_answer(target)!r}, answer {rec['answer']}")
            if fmt in ("points", "randpoints") and rec["prior"] == "count":
                pts = json.loads(target.split("\nTotal:")[0])
                if len(pts) != rec["answer"] or not all("point_2d" in p and len(p["point_2d"]) == 2 for p in pts):
                    problems.append(f"{path}: {row['id']} bad point JSON")
            if len(problems) > 50:
                break
    rep.add("exports parse with the eval harness answer parser + point JSON", n > 0 and not problems,
            summary=f"{n} rows, {len(problems)} problems", examples=problems[:10])


def run_checks():
    rep = Report()
    pools = load_pools()
    if not pools:
        raise SystemExit("No pools built yet.")
    check_records(rep, pools)
    check_points_in_masks(rep, pools)
    check_determinism(rep, pools)
    check_pair_integrity(rep, pools)
    check_distributions(rep, pools)
    check_leakage(rep, pools)
    check_length_rule(rep)
    check_exports(rep)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "ci_report.json").write_text(json.dumps(rep.results, indent=1, default=str))
    print(f"\n{'ALL CHECKS PASSED' if rep.ok else 'SOME CHECKS FAILED'} -> {REPORTS / 'ci_report.json'}")
    return rep.ok


# ---------------------------------------------------------------------------
# §10.2 render audit: contact sheets with ground-truth points drawn on top
# ---------------------------------------------------------------------------

def audit_sheets(per_generator=50, tile=224, cols=10):
    out = REPORTS / "audit"
    out.mkdir(parents=True, exist_ok=True)
    by_gen = defaultdict(list)
    for recs in load_pools().values():
        for r in recs:
            by_gen[r["render"]["generator"]].append(r)
    for gen, recs in tqdm(sorted(by_gen.items()), desc="audit contact sheets", unit="sheet"):
        rng = np.random.default_rng(0)
        pick = [recs[i] for i in sorted(rng.choice(len(recs), size=min(per_generator, len(recs)), replace=False))]
        rows = (len(pick) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tile, rows * (tile + 28)), "white")
        for k, r in enumerate(pick):
            im = Image.open(DATA / r["image"]).convert("RGB")
            d = ImageDraw.Draw(im)
            for x, y in r["points"] or []:
                d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(255, 0, 0), width=2)
            x, y = (k % cols) * tile, (k // cols) * (tile + 28)
            sheet.paste(im.resize((tile, tile), Image.LANCZOS), (x, y + 28))
            dd = ImageDraw.Draw(sheet)
            dd.text((x + 2, y + 1), f"{r['role'][:4]} ans={r['answer']} fam={r['familiar_answer']}", fill="black")
            dd.text((x + 2, y + 14), r["id"][-26:], fill=(90, 90, 90))
        sheet.save(out / f"{gen}.jpg", quality=88)
    print(f"Audit sheets for {len(by_gen)} generators in {out}")
