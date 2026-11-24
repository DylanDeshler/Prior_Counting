"""Charts for the 2026-11-24 deck update (steering round 2, mixed-format training, synthetic neutral vs conflict).
Values are copied from results/ and reports/ (see PROGRESS.md §3.0). Writes ~/Documents/research/deck_assets/."""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(os.path.expanduser("~/Documents/research/deck_assets")); OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, NAVY, GRAY = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#104281", "#a8a7a2")
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False})


def bars(ax, groups, series, colors, labels, refs=None, ymax=100):
    n = len(series); w = 0.8 / n; x = np.arange(len(groups))
    for i, (vals, col, lab) in enumerate(zip(series, colors, labels)):
        xs = x - 0.4 + w * (i + 0.5)
        ax.bar(xs, [np.nan if v is None else v for v in vals], w - 0.03, color=col, label=lab, zorder=3)
        for xi, v in zip(xs, vals):
            if v is not None:
                ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    for lab, (vals, col) in (refs or {}).items():
        for xi, v in zip(x, vals):
            ax.hlines(v, xi - 0.42, xi + 0.42, colors=col, linestyles="--", lw=1.6, zorder=4)
        ax.plot([], [], "--", color=col, label=lab)
    ax.set_xticks(x, groups); ax.set_ylim(0, ymax * 1.08)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.tick_params(length=0)


REFS = {"No training, direct-answer prompt": ([0.0, 12.4, 95.1], GRAY), "No training, pointing prompt": ([78.8, 24.0, 79.1], ORANGE)}

# Strategy: mixed-format training (distillation) — data fixed per panel
fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.6), sharey=True)
groups = ["VLMBias animals", "VLMBias, other topics", "Unedited originals"]
for ax, title, num, mix, pts in (
        (axs[0], "Trained on synthetic conflict pairs", [12.6, 23.6, 92.0], [11.0, 23.1, 85.3], [40.5, 23.9, 72.4]),
        (axs[1], "Trained on real-photo objects (LVIS)", [14.8, 16.5, 82.2], [22.2, 16.5, 79.8], [51.1, 22.5, 71.8])):
    bars(ax, groups, [num, mix, pts], [AQUA, YELLOW, GREEN],
         ["numbers", "half point lists, half numbers; asked for a direct answer", "point lists"], refs=REFS)
    ax.set_title(title, loc="left", fontsize=13, color=INK)
axs[0].set_ylabel("Accuracy (%)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.1), fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / "s_distill.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# Dataset: synthetic conflict pairs vs synthetic neutral images (both point lists)
fig, ax = plt.subplots(figsize=(11, 4.4))
bars(ax, ["Held-out synthetic\nfamilies (edited)", "VLMBias\nanimals", "VLMBias,\nother topics", "Unedited\noriginals"],
     [[88.7, 40.5, 23.9, 72.4], [80.0, 49.5, 25.6, 65.0]], [GREEN, MAGENTA],
     ["Synthetic conflict pairs (canonical vs edited)", "Synthetic neutral images (no familiar count)"],
     refs={"No training, pointing prompt": ([61.1, 78.8, 24.0, 79.1], ORANGE)})
ax.set_ylabel("Accuracy (%)"); ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / "d_neutral.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# Steering round 2 (untrained model)
steer = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.path.expanduser("~/Downloads/reports/steer/Qwen3.5-4B/text-a-10_1_3_10_30_100/steer.json"))
d = json.load(open(steer))
alphas, layers, R = d["alphas"], d["layers"], d["results"]
xs = [a * 0.1 for a in alphas]
fig, axs = plt.subplots(1, 4, figsize=(17, 4.0), sharex=True)
cols = [BLUE, AQUA, VIOLET, NAVY]
for ax, key, title in zip(axs, ("animals_acc", "other_acc", "canonical_acc", "digit_mass"),
                          ("VLMBias animals: true count", "VLMBias other topics: true count", "Unedited originals: correct",
                           "Still answering with a number")):
    base = R["baseline"][key] * (100 if key == "digit_mass" else 1)
    ax.axhline(base, color=GRAY, lw=1.5, ls="--", label="no steering")
    for l, c in zip(layers, cols):
        ys = [R[f"L{l}_a{a}"][key] * (100 if key == "digit_mass" else 1) for a in alphas]
        if key != "digit_mass":  # hide accuracy where the model no longer answers with a number (digit probability < 50%)
            ys = [y if (R[f"L{l}_a{a}"]["digit_mass"] or 0) >= 0.5 else np.nan for y, a in zip(ys, alphas)]
        ax.plot(xs, ys, "-o", color=c, ms=4, label=f"layer {l}")
    ax.set_xscale("symlog", linthresh=0.1); ax.set_xticks([-1, 0.1, 0.3, 1, 3, 10], ["−1", "0.1", "0.3", "1", "3", "10"])
    ax.set_ylim(-3, 103); ax.set_title(title, loc="left", fontsize=12, color=INK); ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.axvspan(2.5, 12, color="#eb6834", alpha=0.06)
axs[0].set_ylabel("%"); axs[3].text(4.5, 50, "model\nbreaks", color="#c4511f", fontsize=9, ha="center")
fig.text(0.01, -0.04, "Accuracy points are hidden where the model no longer answers with a number (digit probability < 50%).", fontsize=9, color=INK2)
for ax in axs:
    ax.set_xlabel("strength (× hidden-state size; −1 = reversed)")
axs[0].legend(frameon=False, fontsize=8.5, loc="center left")
fig.tight_layout(); fig.savefig(OUT / "s_steer2.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("wrote", OUT)
