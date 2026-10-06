"""Statistics and plots of what was generated: data/reports/stats/{index.html,summary.json,*.png}

    python -m datagen stats
"""

import html
import json
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .checks import load_pools  # noqa: E402
from .common import IMG, MIN_UNIT_PX, REPORTS  # noqa: E402

OUT = REPORTS / "stats"
# Validated categorical slots 1-2 (blue, orange), sequential blue ramp, recessive ink.
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.titleweight": "bold", "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "legend.frameon": False,
})


def _bars(ax, labels, values, color=BLUE, offset=0.0, width=0.8, label=None):
    x = np.arange(len(labels)) + offset
    ax.bar(x, values, width=width, color=color, edgecolor=SURFACE, linewidth=1.5, label=label)
    return x


def _grid(n, cols=3, w=3.6, h=2.6):
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * w, rows * h), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:n])


def _save(fig, name, title):
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=110)
    plt.close(fig)
    return name


def _paired_counts(ax, a, b, la, lb, title):
    keys = sorted(set(a) | set(b), key=lambda k: (isinstance(k, str), k))
    w = 0.4
    _bars(ax, keys, [a.get(k, 0) for k in keys], BLUE, -w / 2, w, la)
    _bars(ax, keys, [b.get(k, 0) for k in keys], ORANGE, w / 2, w, lb)
    ax.set_xticks(range(len(keys)), [str(k) for k in keys])
    ax.set_title(title, loc="left")


# ---------------------------------------------------------------------------

def plot_pool_sizes(pools):
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(pools) + 1))
    names = list(pools)[::-1]
    ax.barh(names, [len(pools[n]) for n in names], color=BLUE, edgecolor=SURFACE, linewidth=1.5)
    for i, n in enumerate(names):
        ax.text(len(pools[n]), i, f" {len(pools[n]):,}", va="center", color=INK2)
    ax.grid(axis="x"), ax.grid(axis="y", visible=False)
    ax.set_xlabel("images")
    return _save(fig, "pool_sizes.png", "Images per pool")


def plot_e1_counts(pools):
    """Conflict answers vs neutral-twin answers per family: the twin histogram must match."""
    conf, neu = pools.get("e1/conflict", []), pools.get("e1/neutral", [])
    fams = sorted({r["family"] for r in conf})
    if not fams:
        return None
    fig, axes = _grid(len(fams))
    for ax, fam in zip(axes, fams):
        _paired_counts(ax, Counter(r["answer"] for r in conf if r["family"] == fam),
                       Counter(r["answer"] for r in neu if r["family"] == fam), "conflict", "neutral twin", fam)
    axes[0].legend(loc="upper right")
    return _save(fig, "e1_answer_hist.png", "E1 answers per family: conflict pool vs neutral twins (should match)")


def plot_roles(pools, pool, title, name, roles=("canonical", "counterfactual")):
    recs = pools.get(pool, [])
    fams = sorted({r["family"] for r in recs if r["role"] in roles})
    if not fams:
        return None
    fig, axes = _grid(len(fams), cols=min(3, len(fams)))
    for ax, fam in zip(axes, fams):
        _paired_counts(ax, Counter(r["answer"] for r in recs if r["family"] == fam and r["role"] == roles[0]),
                       Counter(r["answer"] for r in recs if r["family"] == fam and r["role"] == roles[1]),
                       roles[0], roles[1], fam)
    axes[0].legend(loc="upper right")
    return _save(fig, name, title)


def plot_deltas(pools):
    groups = defaultdict(Counter)
    for pool, recs in pools.items():
        for r in recs:
            if r["prior"] == "count" and r["role"] in ("counterfactual", "edited"):
                groups[f"{pool} · {r['family']}"][r["delta"]] += 1
    if not groups:
        return None
    keys = sorted(groups)
    fig, axes = _grid(len(keys), cols=4, w=3.0, h=2.2)
    for ax, k in zip(axes, keys):
        ds = sorted(groups[k])
        _bars(ax, ds, [groups[k][d] for d in ds])
        ax.set_xticks(range(len(ds)), [f"{d:+d}" for d in ds])
        ax.set_title(k, loc="left", fontsize=8)
    return _save(fig, "delta_hist.png", "Δ (edited answer − original answer) per pool and family")


def plot_unit_sizes(pools):
    by = defaultdict(list)
    for recs in pools.values():
        for r in recs:
            if r["prior"] == "count" and r["unit_size_px"]:
                by[r["render"]["generator"]].append(r["unit_size_px"])
    keys = sorted(by)
    fig, axes = _grid(len(keys), cols=5, w=2.6, h=2.0)
    for ax, k in zip(axes, keys):
        ax.hist(by[k], bins=20, color=BLUE, edgecolor=SURFACE, linewidth=1)
        ax.axvline(MIN_UNIT_PX, color=INK2, linestyle="--", linewidth=1)
        ax.set_title(k, loc="left", fontsize=8)
    return _save(fig, "unit_size_hist.png", f"Median unit size per image (px); dashed line = {MIN_UNIT_PX} px minimum")


def plot_placement(pools):
    recs = [r for p in pools.values() for r in p if r["render"]["placement"]["scale"] != 1.0 or r["render"]["placement"]["cx"]]
    if not recs:
        return None
    cx = [(r["object_bbox"][0] + r["object_bbox"][2]) / 2 for r in recs]
    cy = [(r["object_bbox"][1] + r["object_bbox"][3]) / 2 for r in recs]
    size = [max(r["object_bbox"][2] - r["object_bbox"][0], r["object_bbox"][3] - r["object_bbox"][1]) for r in recs]
    rot = [r["render"]["placement"]["rotation_deg"] for r in recs]
    px = [p for r in recs for p in (r["points"] or [])]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    for ax, (xs, ys, title, unit) in zip(axes[:2], ((cx, cy, "Object center (image px)", "images"),
                                                   ([p[0] for p in px], [p[1] for p in px], "Ground-truth unit points (image px)", "points"))):
        h = ax.hist2d(xs, ys, bins=16, range=[[0, IMG], [0, IMG]], cmap=cmap)
        ax.invert_yaxis(), ax.set_aspect("equal"), ax.grid(False)
        ax.set_title(title, loc="left")
        fig.colorbar(h[3], ax=ax, label=unit)
    axes[2].hist(size, bins=30, color=BLUE, edgecolor=SURFACE, linewidth=1)
    axes[2].set_title("Object bbox longest side (px)", loc="left")
    axes[3].hist(rot, bins=30, color=BLUE, edgecolor=SURFACE, linewidth=1)
    axes[3].set_title("Rotation (degrees)", loc="left")
    return _save(fig, "placement.png", "Object placement (all placed objects)")


def plot_render_params(pools):
    recs = [r for p in pools.values() for r in p]
    panels = [
        ("Background", Counter(r["render"]["background"]["kind"] for r in recs)),
        ("Distractors per image", Counter(len(r["render"]["distractors"]) for r in recs)),
        ("Question template", Counter(r["question_template"] for r in recs)),
        ("Distractor shapes", Counter(d["shape"] for r in recs for d in r["render"]["distractors"])),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (title, c) in zip(axes.flat[:4], panels):
        keys = sorted(c, key=str)
        _bars(ax, keys, [c[k] for k in keys])
        ax.set_xticks(range(len(keys)), [str(k) for k in keys], rotation=30 if len(keys) > 4 else 0)
        ax.set_title(title, loc="left")
    axes.flat[4].hist([r["compression"]["downscale"] for r in recs], bins=20, color=BLUE, edgecolor=SURFACE)
    axes.flat[4].set_title("Compression: downscale factor", loc="left")
    axes.flat[5].hist([r["compression"]["jpeg_quality"] for r in recs], bins=36, color=BLUE, edgecolor=SURFACE)
    axes.flat[5].set_title("Compression: JPEG quality", loc="left")
    return _save(fig, "render_params.png", "Render and compression parameters (all images)")


def plot_color(pools):
    recs = [r for p in ("e2/L0/color", "tests/T0/color") for r in pools.get(p, [])]
    if not recs:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (dtype, roles) in zip(axes[:2], (("conflict", ("canonical", "counterfactual")), ("neutral", ("original", "edited")))):
        sub = [r for r in recs if r["data_type"] == dtype]
        _paired_counts(ax, Counter(r["answer"] for r in sub if r["role"] == roles[0]),
                       Counter(r["answer"] for r in sub if r["role"] == roles[1]), roles[0], roles[1],
                       f"{dtype}: answer color")
        ax.tick_params(axis="x", rotation=40)
        ax.legend(loc="upper right")
    objs = Counter(r["render"]["params"]["object"] for r in recs if r["role"] in ("canonical", "original"))
    keys = [k for k, _ in objs.most_common()]
    axes[2].barh(keys[::-1], [objs[k] for k in keys[::-1]], color=BLUE, edgecolor=SURFACE)
    axes[2].grid(axis="x"), axes[2].grid(axis="y", visible=False)
    axes[2].set_title("Pairs per object (all color pools)", loc="left")
    return _save(fig, "color.png", "Color items (Exp 2 L0 + T0)")


def plot_crowds_and_shared(pools):
    crowd = [r for r in pools.get("e2/L0/count", []) if r["data_type"] == "neutral"]
    shared = pools.get("shared_range", [])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    c = Counter(r["render"]["params"].get("category") for r in crowd if r["role"] == "original")
    keys = sorted(c, key=str)
    _bars(axes[0], keys, [c[k] for k in keys])
    axes[0].set_xticks(range(len(keys)), [str(k) for k in keys], rotation=60, fontsize=7)
    axes[0].set_title("Exp 2 crowds: pairs per category", loc="left")
    s = Counter(r["answer"] for r in shared)
    ks = sorted(s)
    axes[1].bar(ks, [s[k] for k in ks], color=BLUE, edgecolor=SURFACE, linewidth=1)
    axes[1].set_title("Shared range block: count", loc="left")
    sh = Counter(r["render"]["params"]["shape"] for r in shared)
    keys = sorted(sh)
    _bars(axes[2], keys, [sh[k] for k in keys])
    axes[2].set_xticks(range(len(keys)), keys)
    axes[2].set_title("Shared range block: shape", loc="left")
    return _save(fig, "crowds_shared.png", "Neutral crowds and shared range block")


def plot_rejections():
    retries = Counter()
    for f in sorted(REPORTS.glob("build_*.json")):
        d = json.loads(f.read_text())
        for g, n in d.get("retries_by_generator", {}).items():
            retries[g] += n
    if not retries:
        return None
    keys = sorted(retries)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    _bars(ax, keys, [retries[k] for k in keys])
    ax.set_xticks(range(len(keys)), keys, rotation=50, ha="right", fontsize=8)
    ax.set_ylabel("rejected attempts")
    return _save(fig, "rejections.png", "Rejected generation attempts (failed recount / color check / placement), resampled")


# ---------------------------------------------------------------------------

def summary_tables(pools):
    t = {}
    for pool, recs in pools.items():
        rows = Counter((r["family"], r["data_type"], r["role"]) for r in recs)
        t[pool] = {
            "images": len(recs),
            "by_family_role": {f"{f} / {d} / {ro}": n for (f, d, ro), n in sorted(rows.items())},
            "templates": dict(Counter(r["question_template"] for r in recs)),
            "answer_range": [min(r["answer"] for r in recs), max(r["answer"] for r in recs)]
            if recs and recs[0]["prior"] == "count" else None,
            "unique_source_backgrounds": len({r["render"]["background"].get("file") for r in recs} - {None}),
        }
    return t


def run_stats():
    OUT.mkdir(parents=True, exist_ok=True)
    pools = load_pools()
    if not pools:
        raise SystemExit("No pools built yet.")
    plots = [plot_pool_sizes(pools), plot_e1_counts(pools),
             plot_roles(pools, "e1/conflict", "E1 conflict pool: canonical vs counterfactual answers", "e1_roles.png"),
             plot_roles(pools, "tests/T0/count", "T0 count: canonical vs counterfactual answers", "t0_roles.png"),
             plot_roles(pools, "e2/L0/count", "Exp 2 L0 count conflict: canonical vs counterfactual", "e2_roles.png"),
             plot_deltas(pools), plot_unit_sizes(pools), plot_placement(pools), plot_render_params(pools),
             plot_color(pools), plot_crowds_and_shared(pools), plot_rejections()]
    plots = [p for p in plots if p]
    tables = summary_tables(pools)
    (OUT / "summary.json").write_text(json.dumps(tables, indent=1))

    parts = ["<html><head><meta charset='utf-8'><title>Generated data statistics</title><style>"
             f"body{{font-family:system-ui,sans-serif;background:{SURFACE};color:{INK};margin:24px;max-width:1300px}}"
             f"table{{border-collapse:collapse;margin:8px 0 24px}}td,th{{border-bottom:1px solid {GRID};padding:3px 10px;"
             f"text-align:left;font-size:13px}}th{{color:{INK2}}}img{{max-width:100%;margin:12px 0 28px}}</style></head><body>",
             "<h1>Generated data statistics</h1>"]
    for p in plots:
        parts.append(f"<img src='{p}' alt='{p}'>")
    parts.append("<h2>Tables</h2>")
    for pool, t in tables.items():
        parts.append(f"<h3>{html.escape(pool)} — {t['images']:,} images</h3><table><tr><th>family / data type / role</th>"
                     "<th>images</th></tr>")
        parts += [f"<tr><td>{html.escape(k)}</td><td>{v:,}</td></tr>" for k, v in t["by_family_role"].items()]
        parts.append(f"</table><p>templates: {t['templates']} · answer range: {t['answer_range']} · "
                     f"distinct background photos: {t['unique_source_backgrounds']}</p>")
    parts.append("</body></html>")
    (OUT / "index.html").write_text("\n".join(parts))
    print(f"{len(plots)} plots + tables -> {OUT / 'index.html'}")
    return plots
