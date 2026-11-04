"""Plots + result-driven samples for eyeballing that the edits work and the model's answers make sense.

    uv run python tools/results_report.py --data /data/vlm [--results results] [--out results/report]
    scp -r <host>:<repo>/results/report.tar.gz .    # then untar and open report/index.html

Reads every results/<model>_<set>_<mode>.jsonl (predictions) and joins generated items to their data
records (image, ground-truth points, pair partner) by ID. VLMBias images come from the HF dataset.

Per model it writes:
  plots    VLMBias accuracy/bias by topic; canonical vs counterfactual accuracy and bias per family;
           where counterfactual answers land (truth / familiar / other) per family; answer offset
           (pred - truth) on canonical vs counterfactual; shared-block accuracy vs count;
           T2 color confusion matrix; and a cross-model comparison when several models exist.
  samples  pairs side by side (canonical | counterfactual | pixel diff of the edit) with ground-truth
           points and the model's answers, picked by outcome: prior override (counterfactual answered
           with the familiar value), correct counterfactuals (edit is visible and read), canonical
           failures (perception, not prior), off-by-other errors, unparsed/truncated replies,
           shared-block misses, T2 color misses, and VLMBias bias-aligned failures per topic.
"""

import argparse
import html
import json
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt  # noqa: E402

from datagen.stats import BLUE, GRID, INK, INK2, ORANGE, SEQ, SURFACE, _bars, _grid  # noqa: E402  (sets the plot style)

GREY = "#b4b2ab"
FILE = re.compile(r"(.+?)_(main|original|gen-.+?)_(think|nothink)\.jsonl$")
THUMB = 240


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

PRED_MTIME = {}


def load_predictions(results):
    """{model: {set_tag: [prediction records]}} (thinking-off and -on runs as separate tags)."""
    out = defaultdict(dict)
    for f in sorted(Path(results).glob("*.jsonl")):
        m = FILE.match(f.name)
        if not m or f.stat().st_size == 0:
            continue
        model, tag, mode = m.groups()
        key = tag + ("" if mode == "nothink" else "+think")
        out[model][key] = [json.loads(l) for l in f.open() if l.strip()]
        PRED_MTIME[(model, key)] = f.stat().st_mtime
    return out


def load_records(data, ids):
    """Data records for the given item ids (image path, points, familiar answer, ...)."""
    want, recs = set(ids), {}
    for f in Path(data).rglob("records.jsonl"):
        if {"preview", "exports", "sources"} & set(f.relative_to(data).parts[:-1]):
            continue
        for line in f.open():
            i = line.find('"id": "')
            rid = line[i + 7:line.find('"', i + 7)]
            if rid in want:
                r = json.loads(line)
                recs[rid] = {k: r.get(k) for k in ("id", "pair_id", "image", "points", "object_bbox", "answer",
                                                   "familiar_answer", "role", "family", "edit", "label")}
                recs[rid]["_built"] = f.stat().st_mtime
    return recs


def vlmbias_images(ids, split):
    try:
        from datasets import load_dataset
        ds = load_dataset("anvo25/vlms-are-biased", split=split)
    except Exception as e:  # no network / not cached: samples without images
        print(f"VLMBias images unavailable ({e}); VLMBias samples will be text only")
        return {}
    want = set(ids)
    idx = [i for i, x in enumerate(ds["ID"]) if x in want]
    return {ds[i]["ID"]: ds[i]["image"] for i in idx}


def family_role(p):
    parts = p["sub_topic"].split(" · ")
    return (parts[1], parts[2]) if len(parts) == 3 else (p["sub_topic"], "counterfactual")


def outcome(p):
    if p["pred"] == "":
        return "unparsed"
    if p["correct"]:
        return "correct"
    if p.get("bias_aligned"):
        return "familiar"
    return "other"


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save(fig, out, name, title):
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out / name, dpi=110)
    plt.close(fig)
    return name


def plot_vlmbias_topics(preds, out, model):
    rows = [p for p in preds if p["topic"] != "Optical Illusion"]
    if not rows:
        return None
    topics = sorted({p["topic"] for p in rows})
    acc = [np.mean([p["correct"] for p in rows if p["topic"] == t]) * 100 for t in topics]
    bias = [np.mean([p["bias_aligned"] for p in rows if p["topic"] == t]) * 100 for t in topics]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    w = 0.4
    _bars(ax, topics, acc, BLUE, -w / 2, w, "accuracy")
    _bars(ax, topics, bias, ORANGE, w / 2, w, "bias rate (familiar answer)")
    ax.set_xticks(range(len(topics)), topics)
    ax.set_ylim(0, 100), ax.set_ylabel("%"), ax.legend(loc="upper right")
    return save(fig, out, f"{model}_vlmbias_topics.png", f"{model}: VLMBias counting, accuracy vs bias by topic")


def plot_families(preds, out, model, tag):
    fams = defaultdict(lambda: defaultdict(list))
    for p in preds:
        f, role = family_role(p)
        fams[f][role].append(p)
    fams = {f: d for f, d in fams.items() if "canonical" in d and "counterfactual" in d}
    if not fams:
        return None
    names = sorted(fams, key=lambda f: -np.mean([p["bias_aligned"] for p in fams[f]["counterfactual"]]))
    can = [np.mean([p["correct"] for p in fams[f]["canonical"]]) * 100 for f in names]
    cf = [np.mean([p["correct"] for p in fams[f]["counterfactual"]]) * 100 for f in names]
    bias = [np.mean([p["bias_aligned"] for p in fams[f]["counterfactual"]]) * 100 for f in names]
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(names) + 2), 3.6))
    w = 0.27
    _bars(ax, names, can, BLUE, -w, w, "canonical acc")
    _bars(ax, names, cf, ORANGE, 0, w, "counterfactual acc")
    _bars(ax, names, bias, GREY, w, w, "counterfactual bias")
    ax.set_xticks(range(len(names)), names, rotation=20 if len(names) > 6 else 0)
    ax.set_ylim(0, 105), ax.set_ylabel("%"), ax.legend(loc="upper right", ncol=3, fontsize=8)
    return save(fig, out, f"{model}_{tag}_families.png",
                f"{model} · {tag}: per family (sorted by bias). Gap canonical→counterfactual = prior override")


def plot_where_answers_land(preds, out, model, tag):
    """Stacked share of counterfactual answers: correct / familiar / other / unparsed, per family."""
    fams = defaultdict(list)
    for p in preds:
        f, role = family_role(p)
        if role == "counterfactual":
            fams[f].append(outcome(p))
    if not fams:
        return None
    names = sorted(fams, key=lambda f: -fams[f].count("familiar") / len(fams[f]))
    kinds = [("correct", BLUE), ("familiar", ORANGE), ("other", GREY), ("unparsed", INK2)]
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(names) + 2), 3.4))
    bottom = np.zeros(len(names))
    for k, c in kinds:
        v = np.array([100 * fams[f].count(k) / len(fams[f]) for f in names])
        ax.bar(range(len(names)), v, 0.7, bottom=bottom, color=c, edgecolor=SURFACE, linewidth=1.5, label=k)
        bottom += v
    ax.set_xticks(range(len(names)), names, rotation=20 if len(names) > 6 else 0)
    ax.set_ylabel("% of counterfactual answers"), ax.legend(loc="upper right", ncol=4, fontsize=8)
    return save(fig, out, f"{model}_{tag}_landing.png", f"{model} · {tag}: where counterfactual answers land")


def plot_offsets(preds, out, model, tag):
    off = {"canonical": [], "counterfactual": []}
    for p in preds:
        _, role = family_role(p)
        if role in off and p["pred"].lstrip("-").isdigit() and p["ground_truth"].isdigit():
            off[role].append(int(p["pred"]) - int(p["ground_truth"]))
    if not off["counterfactual"]:
        return None
    lim = 6
    bins = np.arange(-lim - 0.5, lim + 1.5)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for role, c in (("canonical", BLUE), ("counterfactual", ORANGE)):
        v = np.clip(off[role], -lim, lim)
        if len(v):
            ax.hist(v, bins=bins, color=c, alpha=0.75, edgecolor=SURFACE, label=f"{role} (n={len(v)})",
                    weights=np.full(len(v), 100 / len(v)))
    ax.set_xlabel(f"prediction − truth (clipped at ±{lim})"), ax.set_ylabel("% of items"), ax.legend()
    return save(fig, out, f"{model}_{tag}_offsets.png", f"{model} · {tag}: answer offset")


def plot_shared(preds, out, model, tag):
    rows = [p for p in preds if "shared" in p["topic"] and p["ground_truth"].isdigit()]
    if not rows:
        return None
    by = defaultdict(list)
    for p in rows:
        by[int(p["ground_truth"])].append(p)
    ks = sorted(by)
    acc = [100 * np.mean([p["correct"] for p in by[k]]) for k in ks]
    mae = [np.mean([abs(int(p["pred"]) - k) if p["pred"].isdigit() else k for p in by[k]]) for k in ks]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    axes[0].plot(ks, acc, color=BLUE, linewidth=2), axes[0].set_ylim(0, 100)
    axes[0].set_xlabel("true count"), axes[0].set_ylabel("accuracy %"), axes[0].set_title("Exact accuracy", loc="left")
    axes[1].plot(ks, mae, color=ORANGE, linewidth=2)
    axes[1].set_xlabel("true count"), axes[1].set_ylabel("mean |error|"), axes[1].set_title("Mean absolute error", loc="left")
    return save(fig, out, f"{model}_{tag}_shared.png", f"{model} · {tag}: shared large-count block")


def plot_color_confusion(preds, out, model, tag):
    rows = [p for p in preds if not p["ground_truth"].isdigit()]
    if not rows:
        return None
    from matplotlib.colors import LinearSegmentedColormap
    truths = sorted({p["ground_truth"] for p in rows})
    predv = sorted({p["pred"] or "(none)" for p in rows})
    M = np.zeros((len(truths), len(predv)))
    for p in rows:
        M[truths.index(p["ground_truth"]), predv.index(p["pred"] or "(none)")] += 1
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(predv), 1 + 0.5 * len(truths)))
    ax.imshow(M, cmap=LinearSegmentedColormap.from_list("seq", SEQ), aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(predv)), predv, rotation=45, ha="right"), ax.set_yticks(range(len(truths)), truths)
    for i in range(len(truths)):
        for j in range(len(predv)):
            if M[i, j]:
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=8,
                        color=SURFACE if M[i, j] > M.max() / 2 else INK)
    ax.set_xlabel("model answer"), ax.set_ylabel("true color")
    return save(fig, out, f"{model}_{tag}_color_confusion.png", f"{model} · {tag}: color confusion")


def plot_models(summaries, out):
    if len(summaries) < 2:
        return None
    metrics = [("VLMBias cnt acc", "vlm_acc"), ("VLMBias bias", "vlm_bias"), ("Orig (canonical)", "orig"),
               ("T0 cf acc", "t0_cf"), ("Val conflict", "val_c")]
    models = list(summaries)
    fig, axes = _grid(len(metrics), cols=len(metrics), w=3.0, h=3.4)
    for ax, (title, key) in zip(axes, metrics):
        v = [100 * (summaries[m].get(key) or 0) for m in models]
        ax.barh(range(len(models)), v, color=[BLUE if i == 0 else ORANGE for i in range(len(models))], edgecolor=SURFACE)
        ax.set_yticks(range(len(models)), [m[:28] for m in models] if ax is axes[0] else [""] * len(models))
        ax.set_title(title, loc="left"), ax.set_xlim(0, 100), ax.grid(axis="x"), ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    return save(fig, out, "models.png", "Models compared (blue = first / base)")


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------

def thumb(img, points=None, box=None, name=None, out=None):
    im = img.convert("RGB")
    d = ImageDraw.Draw(im)
    if box:
        d.rectangle(box, outline=(255, 255, 255))
    for x, y in points or []:
        d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(255, 0, 0), width=2)
    im = im.resize((THUMB, THUMB), Image.LANCZOS)
    im.save(out / name, quality=85)
    return f"img/{name}"


def diff_thumb(a, b, name, out):
    x, y = np.asarray(a.convert("RGB"), int), np.asarray(b.convert("RGB"), int)
    d = np.abs(x - y).max(axis=2) > 24  # past JPEG noise: the edit itself
    base = (y * 0.35).astype(np.uint8)
    base[d] = (255, 64, 64)
    Image.fromarray(base).resize((THUMB, THUMB), Image.NEAREST).save(out / name, quality=85)
    return f"img/{name}"


def card(cells, caption):
    figs = "".join(f"<figure><img src='{src}'><figcaption>{html.escape(lbl)}</figcaption></figure>" for src, lbl in cells)
    return f"<div class=card><div class=row>{figs}</div><div class=cap>{caption}</div></div>"


def pred_line(p):
    if p is None:
        return ""
    o = outcome(p)
    tag = {"correct": "✓", "familiar": "✗ familiar", "other": "✗", "unparsed": "∅ unparsed"}[o]
    resp = html.escape(p["response"][-220:].replace("\n", " "))
    return (f"<b>{html.escape(p['ID'].rsplit('-', 1)[-1])}</b>: truth {html.escape(p['ground_truth'])} · "
            f"model <b>{html.escape(p['pred'] or '—')}</b> {tag}<br><span class=resp>…{resp}</span>")


def pair_samples(preds, recs, data, out, n_per):
    """Cards for generated pairs, grouped by outcome on the counterfactual."""
    by_id = {p["ID"]: p for p in preds}
    groups = defaultdict(lambda: defaultdict(list))
    for p in preds:
        f, role = family_role(p)
        if role in ("counterfactual", "edited"):
            groups[outcome(p)][f].append(p)
        elif role in ("canonical", "original") and not p["correct"]:
            groups["canonical_wrong"][f].append(p)
    sections = []
    titles = {"familiar": "Prior override: counterfactual answered with the familiar value",
              "correct": "Correct counterfactuals: the edit is visible and read",
              "other": "Wrong in another way (neither truth nor familiar)",
              "unparsed": "Unparsed / truncated replies",
              "canonical_wrong": "Canonical (unedited) failures: perception, not prior"}
    for key in ("familiar", "correct", "other", "canonical_wrong", "unparsed"):
        fams = groups.get(key, {})
        if not fams:
            continue
        cards = []
        for f in sorted(fams):
            for p in sorted(fams[f], key=lambda p: p["ID"])[:n_per]:
                stem = p["ID"].rsplit("-", 1)[0]
                can_id = next((stem + s for s in ("-can", "-orig") if stem + s in by_id or stem + s in recs), None)
                cf_id = next((stem + s for s in ("-cf", "-edit") if stem + s in by_id or stem + s in recs), None)
                cells = []
                imgs = {}
                for rid in (can_id, cf_id):
                    r = recs.get(rid)
                    if r and (Path(data) / r["image"]).exists():
                        im = Image.open(Path(data) / r["image"])
                        imgs[rid] = im
                        role = "canonical" if rid == can_id else "counterfactual"
                        cells.append((thumb(im, r["points"] if r["points"] and len(r["points"]) <= 60 else None,
                                            None, f"{rid}.jpg", out), f"{role}: answer {r['answer']}"))
                if len(imgs) == 2:
                    cells.append((diff_thumb(imgs[can_id], imgs[cf_id], f"{stem}_diff.jpg", out), "edit (pixel diff)"))
                fam_ans = recs.get(cf_id, {}).get("familiar_answer") if cf_id in recs else None
                cap = (f"<b>{html.escape(f)}</b> · {html.escape(p['prompt'][:110])}"
                       + (f" · familiar answer {html.escape(str(fam_ans))}" if fam_ans is not None else "")
                       + f"<br>{pred_line(by_id.get(can_id))}<br>{pred_line(by_id.get(cf_id))}")
                cards.append(card(cells, cap) if cells else f"<div class=card><div class=cap>{cap}<br><i>image not found under --data</i></div></div>")
        sections.append((titles[key], cards))
    return sections


def single_samples(preds, recs, data, out, n, title, filt):
    rows = sorted([p for p in preds if filt(p)], key=lambda p: p["ID"])[:n]
    cards = []
    for p in rows:
        r = recs.get(p["ID"])
        cells = []
        if r and (Path(data) / r["image"]).exists():
            cells.append((thumb(Image.open(Path(data) / r["image"]),
                                r["points"] if r["points"] and len(r["points"]) <= 60 else None, None,
                                f"{p['ID']}.jpg", out), f"answer {r['answer']}"))
        cards.append(card(cells, f"{html.escape(p['prompt'][:110])}<br>{pred_line(p)}"))
    return [(title, cards)] if cards else []


def vlmbias_samples(preds, out, n_per, split):
    rows = [p for p in preds if p["topic"] != "Optical Illusion"]
    pick = []
    for t in sorted({p["topic"] for p in rows}):
        pool = [p for p in rows if p["topic"] == t]
        pick += [p for p in pool if p["bias_aligned"]][:n_per] + [p for p in pool if p["correct"]][: max(1, n_per // 2)]
    imgs = vlmbias_images([p["ID"] for p in pick], split)
    cards = []
    for p in pick:
        cells = [(thumb(imgs[p["ID"]], name=f"vlmbias_{p['ID']}.jpg", out=out), p["topic"])] if p["ID"] in imgs else []
        cards.append(card(cells, f"<b>{html.escape(p['topic'])}</b> · {html.escape(p['prompt'][:110])}"
                                 f"<br>familiar {html.escape(str(p['expected_bias']))} · {pred_line(p)}"))
    return [(f"VLMBias ({split}): bias-aligned failures and some correct answers, per topic", cards)] if cards else []


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

CSS = (f"body{{font-family:system-ui,sans-serif;background:{SURFACE};color:{INK};margin:24px;max-width:1400px}}"
       f"img.plot{{max-width:100%;margin:8px 0 20px}}.card{{border-bottom:1px solid {GRID};padding:8px 0}}"
       f".row{{display:flex;gap:8px;flex-wrap:wrap}}figure{{margin:0}}figcaption{{font-size:11px;color:{INK2}}}"
       f".cap{{font-size:13px;margin-top:4px}}.resp{{color:{INK2};font-size:11px}}details{{margin:6px 0 14px}}"
       f"summary{{cursor:pointer;font-weight:600}}table{{border-collapse:collapse}}td,th{{border-bottom:1px solid {GRID};"
       f"padding:2px 8px;font-size:13px}}")


def summary_numbers(sets):
    def acc(rows):
        return float(np.mean([p["correct"] for p in rows])) if rows else None
    s = {}
    main = [p for p in sets.get("main", []) if p["topic"] != "Optical Illusion"]
    if main:
        s["vlm_acc"], s["vlm_bias"] = acc(main), float(np.mean([p["bias_aligned"] for p in main]))
    orig = [p for p in sets.get("original", []) if p["topic"] != "Optical Illusion"]
    s["orig"] = acc(orig)
    t0 = next((v for k, v in sets.items() if "T0-count" in k and "+think" not in k), [])
    s["t0_cf"] = acc([p for p in t0 if family_role(p)[1] == "counterfactual"])
    val = next((v for k, v in sets.items() if "val_conflict" in k and "+think" not in k), [])
    s["val_c"] = acc([p for p in val if p["topic"] == "e1_val_conflict"])
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="data dir used for generation (records + images)")
    ap.add_argument("--results", default=Path("results"), type=Path)
    ap.add_argument("--out", default=None, type=Path, help="default: <results>/report")
    ap.add_argument("--models", nargs="*", help="only these models (default: all found)")
    ap.add_argument("--per-family", type=int, default=3, help="sample cards per family and outcome")
    args = ap.parse_args()
    out = args.out or args.results / "report"
    (out / "img").mkdir(parents=True, exist_ok=True)

    preds = load_predictions(args.results)
    if args.models:
        preds = {m: v for m, v in preds.items() if m in args.models}
    if not preds:
        raise SystemExit(f"no prediction files in {args.results}")
    models = sorted(preds, key=lambda m: (m != "Qwen3.5-4B", m))
    gen_ids = [p["ID"] for m in models for k, rows in preds[m].items() if k.startswith("gen-") for p in rows]
    recs = load_records(args.data, gen_ids)
    print(f"{len(models)} model(s); joined {len(recs)} generated items to data records")

    img = out / "img"
    body = ["<h1>Results report</h1>"]
    summaries = {m: summary_numbers(preds[m]) for m in models}
    rm = args.results / "RESULTS.md"
    if rm.exists():
        lines = [l for l in rm.read_text().splitlines() if l.startswith("|")]
        if lines:
            head = [c.strip() for c in lines[0].strip("|").split("|")]
            rows = "".join("<tr>" + "".join(f"<td>{html.escape(c.strip())}</td>" for c in l.strip("|").split("|")) + "</tr>"
                           for l in lines[2:])
            body.append("<table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in head) + f"</tr>{rows}</table>")
    p = plot_models(summaries, img)
    if p:
        body.append(f"<img class=plot src='img/{p}'>")

    for m in models:
        sets = preds[m]
        body.append(f"<h2>{html.escape(m)}</h2>")
        plots, sections = [], []
        if "main" in sets:
            plots.append(plot_vlmbias_topics(sets["main"], img, m))
        for tag, rows in sorted(sets.items()):
            if not tag.startswith("gen-"):
                continue
            short = tag.replace("gen-", "").replace("+", "_")
            built = [recs[p["ID"]]["_built"] for p in rows if p["ID"] in recs]
            if built and max(built) > PRED_MTIME.get((m, tag), 0):
                body.append(f"<p style='color:#b42318'><b>Warning:</b> {html.escape(short)} predictions are older than "
                            f"the data they are matched to (data rebuilt after this eval). Sample images may differ "
                            f"from what the model saw; rerun the eval to refresh.</p>")
                print(f"WARNING: {m} {short}: predictions predate the data build; samples may show other images")
            plots += [plot_families(rows, img, m, short), plot_where_answers_land(rows, img, m, short),
                      plot_offsets(rows, img, m, short), plot_shared(rows, img, m, short),
                      plot_color_confusion(rows, img, m, short)]
            sections += [(f"{short}: {t}", c) for t, c in pair_samples(rows, recs, args.data, img, args.per_family)]
            sections += [(f"{short}: {t}", c) for t, c in single_samples(
                rows, recs, args.data, img, 12, "Shared block misses (large counts)",
                lambda p: "shared" in p["topic"] and not p["correct"])]
        for split in ("main", "original"):
            if split in sets:
                sections += vlmbias_samples(sets[split], img, args.per_family, split)
        body += [f"<img class=plot src='img/{p}'>" for p in plots if p]
        for title, cards in sections:
            body.append(f"<details><summary>{html.escape(title)} ({len(cards)})</summary>{''.join(cards)}</details>")
        print(f"{m}: {sum(1 for p in plots if p)} plots, {sum(len(c) for _, c in sections)} sample cards")

    (out / "index.html").write_text(f"<html><head><meta charset='utf-8'><title>Results report</title>"
                                    f"<style>{CSS}</style></head><body>{''.join(body)}</body></html>")
    tar = out.parent / f"{out.name}.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(out, arcname=out.name)
    print(f"report: {out / 'index.html'}\nbundle: {tar} ({tar.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
