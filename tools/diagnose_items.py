# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.4", "torchvision", "transformers>=5.2.0", "accelerate", "pillow", "numpy", "scikit-learn",
#     "datasets>=3.0", "matplotlib", "pycocotools", "qwen-vl-utils",
# ]
# ///
"""Why does the model give the familiar count on specific items? Per-item mechanistic diagnosis with figures.

    uv run tools/diagnose_items.py --data /data/vlm --clean runs/<arm>/v*/checkpoint-*-merged [--base Qwen/Qwen3.5-4B]

Items: VLMBias counting items where the untrained model gives the familiar count and the trained model (same tokens)
gives the true count, forced direct answer ("{" teacher-forced). Animals and other topics are kept separate. Metric:
logit(true count) - logit(familiar count) at the answer position ("preference"; > 0 favors the true count).

Architecture: Qwen3.5 is single-stream (vision-encoder tokens and text share one sequence and one decoder). Every 4th
decoder layer is standard self-attention (with per-head attention weights); the rest are linear-attention layers with
no attention matrix. Attention measures cover the standard layers; attribution and patching cover all layers.

Analyses (three contrasts + a causal test):
  A. Untrained vs trained model, same input.
     1. Attention from the answer position to image tokens / subject words (the counted part and the object noun,
        e.g. "legs", "animal") / other question words / chat template, per standard layer (hypothesis: early layers
        look at the image, the late layers that overwrite the count look at the subject words).
     2. Knockout: block attention to the image or the subject words at one standard layer, from the answer position
        or from every text position after the image.
     3. Direct logit attribution per layer (attention/linear-attention block and MLP), and its running total
        ("tug of war": where the preference turns toward the familiar count).
     4. Activation patching trained -> untrained per layer at answer / image / text positions (gap restored).
  B. Edited image vs the unedited original (animals with a matching original): running preference for both images,
     and how different the answer-position states are by layer (does the model register the edit, then erase it?).
  C. Subject-word swap: "this animal" -> "this object" (the object noun only): does the familiar answer weaken?
  D. Ablation (causal test): zero the components that push hardest toward the familiar count (chosen on half the
     items), on the other half and on the unedited originals: does the answer flip to the true count without breaking
     the originals?
Writes <data>/reports/diagnose/<trained model>/: index.html (figures with plain-language captions), item cards,
diagnose.json.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from answer_parsing import counted_part, rewrite_prompt  # noqa: E402
from probe_vision import DIRECT, IMG, Runner, num  # noqa: E402

GROUPS = ("image", "subject", "question", "template")
GCOL = {"image": "#2a78d6", "subject": "#eb6834", "question": "#1baf7a", "template": "#b8b7b2"}
GLAB = {"image": "image tokens", "subject": "subject words (e.g. 'legs', 'animal')", "question": "other question words",
        "template": "chat template"}
UNTRAINED, TRAINED, ORIGINAL = "#6f6e6a", "#104281", "#2a78d6"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--base", default="Qwen/Qwen3.5-4B")
    p.add_argument("--clean", required=True, help="model that answers correctly on the same inputs (e.g. a trained arm)")
    p.add_argument("--per-group", type=int, default=30, help="items per group (animals, other topics)")
    p.add_argument("--max-scan", type=int, default=900)
    p.add_argument("--cards", type=int, default=6, help="item cards per group")
    p.add_argument("--any-clean", action="store_true", help="don't require the clean model to be right (smoke tests)")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def model_tag(path):
    mp = Path(str(path).rstrip("/"))
    return (f"{mp.parent.parent.name}-{mp.name.replace('checkpoint-', 'ckpt')}" if mp.exists() and mp.name.startswith("checkpoint")
            else str(path).split("/")[-1])


def species(item_id, tag):
    m = re.search(rf"animals?[-_](?:legs[-_])?\d+[-_]([a-z_]+?)[-_]{tag}", item_id.lower())
    return m.group(1) if m else None


class Ctx:
    """One item: inputs, token groups, true/familiar digit ids. `question` overrides the prompt (subject swap)."""
    def __init__(self, R, image, prompt, truth, familiar, question=None):
        self.prompt = prompt
        self.q = question or rewrite_prompt(prompt, DIRECT)
        self.image = image
        self.enc = R.inputs(image, self.q, prefix="{")
        ids = self.enc["input_ids"][0].tolist()
        img, side = R.image_positions(self.enc)
        img = img.tolist()
        self.side, self.n, self.ans = side, len(ids), len(ids) - 1
        self.tokens = [t.replace("Ġ", " ").replace("▁", " ") for t in R.tok.convert_ids_to_tokens(ids)]
        low = [t.strip().lower() for t in self.tokens]
        words = set(re.findall(r"[a-z]+", counted_part(prompt).lower()))
        m = re.search(r"\b(?:this|the|these)\s+([a-z]+)", self.q.lower())
        if m:
            words.add(m.group(1))
        words |= {w[:-1] for w in words if w.endswith("s")}
        after = img[-1] + 1
        qtext = R.tok(self.q, add_special_tokens=False)["input_ids"]
        q_start = next((i for i in range(after, self.n - len(qtext) + 1) if ids[i:i + 3] == qtext[:3]), after)
        self.q_span = (q_start, min(q_start + len(qtext), self.n - 1))
        self.groups = {g: [] for g in GROUPS}
        self.groups["image"] = img
        for i in range(self.n - 1):
            if i in img:
                continue
            if self.q_span[0] <= i < self.q_span[1]:
                self.groups["subject" if (low[i] and low[i] in words) else "question"].append(i)
            else:
                self.groups["template"].append(i)
        self.text_after_image = list(range(after, self.n))
        self.true, self.fam = truth, familiar
        self.d_true, self.d_fam = R.digit_ids[truth], R.digit_ids[familiar]


def lm_of(R):
    return R.model.model.language_model if hasattr(R.model.model, "language_model") else R.model.model


def pref(R, ctx):
    with torch.no_grad():
        lg = R.model(**ctx.enc).logits[0, -1].float()
    return float(lg[ctx.d_true] - lg[ctx.d_fam])


def run_capture(R, ctx, attentions=False):
    """One forward: per-layer mixer/MLP outputs at the answer position, layer outputs (answer position), pre-norm
    residual, attention weights of the standard layers, logits."""
    lm = lm_of(R)
    st, hooks = {}, [R.norm.register_forward_pre_hook(lambda m, a: st.__setitem__("prenorm", a[0][0, -1].float()))]
    for i, layer in enumerate(lm.layers):
        mixer = layer.self_attn if hasattr(layer, "self_attn") else layer.linear_attn
        hooks.append(mixer.register_forward_hook(lambda m, a, o, i=i: st.__setitem__(("mix", i), (o[0] if isinstance(o, tuple) else o)[0, -1].float())))
        hooks.append(layer.mlp.register_forward_hook(lambda m, a, o, i=i: st.__setitem__(("mlp", i), o[0, -1].float())))
        hooks.append(layer.register_forward_hook(lambda m, a, o, i=i: st.__setitem__(("out", i), (o[0] if isinstance(o, tuple) else o)[0, -1].float())))
    try:
        with torch.no_grad():
            out = R.model(**ctx.enc, output_attentions=attentions, output_hidden_states=True)
    finally:
        for h in hooks:
            h.remove()
    L = len(lm.layers)
    hfin = st["prenorm"]
    with torch.no_grad():
        normed = R.norm(hfin[None, None].to(R.model.dtype))[0, 0].float()
    g = torch.where(hfin.abs() > 1e-6, normed / hfin, torch.zeros_like(hfin))  # linearized final norm
    w = (R.head.weight[ctx.d_true] - R.head.weight[ctx.d_fam]).float()
    proj = lambda v: float(((v * g) * w).sum())
    res = {"mix": [proj(st[("mix", i)]) for i in range(L)], "mlp": [proj(st[("mlp", i)]) for i in range(L)],
           "emb": proj(out.hidden_states[0][0, -1].float()),
           "states": torch.stack([st[("out", i)] for i in range(L)]).cpu().numpy(),
           "pref": float(out.logits[0, -1, ctx.d_true] - out.logits[0, -1, ctx.d_fam])}
    res["running"] = (res["emb"] + np.cumsum(np.array(res["mix"]) + np.array(res["mlp"]))).tolist()
    if attentions:
        res["att"] = [a[0, :, ctx.ans].float().cpu().numpy() for a in out.attentions if a is not None]  # (heads, keys)
    return res


def knockout(R, ctx, layer, target, source):
    attn = lm_of(R).layers[layer].self_attn
    tgt = ctx.groups[target]
    src = [ctx.ans] if source == "answer" else ctx.text_after_image
    def pre(module, args, kwargs):
        am = kwargs.get("attention_mask")
        if am is None:
            n = ctx.n
            am = torch.zeros(1, 1, n, n, device=R.dev, dtype=R.model.dtype)
            am = am.masked_fill(torch.ones(n, n, device=R.dev, dtype=torch.bool).triu(1), torch.finfo(am.dtype).min)
        am = am.clone()
        am[:, :, torch.tensor(src, device=am.device)[:, None], torch.tensor(tgt, device=am.device)[None, :]] = torch.finfo(am.dtype).min
        kwargs["attention_mask"] = am
        return args, kwargs
    h = attn.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        return pref(R, ctx)
    finally:
        h.remove()


def patch_curve(Rb, Rc, ctx):
    lmc, lmb = lm_of(Rc), lm_of(Rb)
    cache, hooks = {}, []
    for l, layer in enumerate(lmc.layers):
        hooks.append(layer.register_forward_hook(lambda m, a, o, l=l: cache.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0].detach())))
    try:
        with torch.no_grad():
            oc = Rc.model(**ctx.enc)
    finally:
        for h in hooks:
            h.remove()
    clean = float(oc.logits[0, -1, ctx.d_true] - oc.logits[0, -1, ctx.d_fam])
    corrupt = pref(Rb, ctx)
    gap = clean - corrupt
    img = set(ctx.groups["image"])
    pos_sets = {"answer": [ctx.ans], "image": sorted(img), "text": [i for i in range(ctx.n) if i not in img]}
    res = {k: [] for k in pos_sets}
    for l, layer in enumerate(lmb.layers):
        src = cache[l].to(Rb.model.dtype)
        for name, pos in pos_sets.items():
            idx = torch.tensor(pos, device=Rb.dev)
            def hook(m, a, o, idx=idx, src=src):
                hs = o[0] if isinstance(o, tuple) else o
                hs[0, idx] = src[idx].to(hs.device)
                return o
            h = layer.register_forward_hook(hook)
            try:
                d = pref(Rb, ctx)
            finally:
                h.remove()
            res[name].append((d - corrupt) / gap if abs(gap) > 1e-3 else float("nan"))
    return res


def ablate(R, ctx, comps):
    """Zero the listed components' outputs at the answer position; return the preference."""
    lm, hooks = lm_of(R), []
    for kind, i in comps:
        layer = lm.layers[i]
        mod = (layer.self_attn if hasattr(layer, "self_attn") else layer.linear_attn) if kind == "mix" else layer.mlp
        def hook(m, a, o):
            hs = o[0] if isinstance(o, tuple) else o
            hs[0, -1] = 0
            return o
        hooks.append(mod.register_forward_hook(hook))
    try:
        return pref(R, ctx)
    finally:
        for h in hooks:
            h.remove()


def swap_subject(q):
    """'this animal' -> 'this object' (object noun only; the counted part stays)."""
    return re.sub(r"\b(this|the)\s+([a-z]+)", lambda m: f"{m.group(1)} object", q, count=1, flags=re.I)


# ---------------------------------------------------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = args.out or args.data / "reports" / "diagnose" / model_tag(args.clean)
    out_dir.mkdir(parents=True, exist_ok=True)
    Rb = Runner(args.base, attn_implementation="eager")
    Rc = Runner(args.clean, attn_implementation="eager")
    lmb = lm_of(Rb)
    L = len(lmb.layers)
    full = [i for i, l in enumerate(lmb.layers) if hasattr(l, "self_attn")]
    from datasets import load_dataset
    from tqdm import tqdm
    ds = load_dataset("anvo25/vlms-are-biased", split="main")
    orig = load_dataset("anvo25/vlms-are-biased", split="original")
    orig_by_species = {}
    for k, oid in enumerate(orig["ID"]):
        sp = species(oid, "original")
        if sp and "count" in oid.lower() and num(orig[k]["ground_truth"]) is not None:
            orig_by_species.setdefault(sp, k)
    rng = np.random.default_rng(0)
    cand = [i for i, (gt, b, t) in enumerate(zip(ds["ground_truth"], ds["expected_bias"], ds["topic"]))
            if t != "Optical Illusion" and num(gt) is not None and num(b) is not None and num(gt) < 10 and num(b) < 10 and num(gt) != num(b)]
    rng.shuffle(cand)
    picked = {"animals": [], "other": []}
    for i in tqdm(cand[: args.max_scan], desc="select items (untrained wrong, trained right)", unit="item"):
        if all(len(v) >= args.per_group for v in picked.values()):
            break
        ex = ds[i]
        grp = "animals" if ex["topic"] == "Animals" else "other"
        if len(picked[grp]) >= args.per_group:
            continue
        ctx = Ctx(Rb, ex["image"], ex["prompt"], num(ex["ground_truth"]), num(ex["expected_bias"]))
        if pref(Rb, ctx) < 0 and (args.any_clean or pref(Rc, ctx) > 0):
            picked[grp].append((i, ctx))
    res = {"base": args.base, "clean": args.clean, "standard_attention_layers": full, "n_layers": L,
           "n": {g: len(v) for g, v in picked.items()}}
    print("items:", res["n"])
    cards = []

    for grp, items in picked.items():
        if not items:
            continue
        acc = {"att_b": [], "att_c": [], "running_b": [], "running_c": [], "running_o": [], "mix": [], "mlp": [], "emb": [],
               "ko": {f"{t}<-{s}": [] for t in ("image", "subject") for s in ("answer", "text")}, "patch": {"answer": [], "image": [], "text": []},
               "div_b": [], "div_c": [], "swap": [], "pref_b": [], "orig_pref": []}
        orig_ctx = {}
        for n_item, (i, ctx) in enumerate(tqdm(items, desc=f"diagnose {grp}", unit="item")):
            ex = ds[i]
            cb, cc = run_capture(Rb, ctx, attentions=True), run_capture(Rc, ctx, attentions=True)
            def masses(att):
                return np.array([[a[:, ctx.groups[g]].sum(1).mean() if ctx.groups[g] else 0.0 for g in GROUPS] for a in att])
            acc["att_b"].append(masses(cb["att"])); acc["att_c"].append(masses(cc["att"]))
            acc["running_b"].append(cb["running"]); acc["running_c"].append(cc["running"])
            acc["mix"].append(cb["mix"]); acc["mlp"].append(cb["mlp"]); acc["emb"].append(cb["emb"]); acc["pref_b"].append(cb["pref"])
            for key in acc["ko"]:
                t, s = key.split("<-")
                acc["ko"][key].append([knockout(Rb, ctx, l, t, s) - cb["pref"] if ctx.groups[t] else np.nan for l in full])
            pc = patch_curve(Rb, Rc, ctx)
            for k in acc["patch"]:
                acc["patch"][k].append(pc[k])
            # B. edited vs original (animals)
            co = None
            sp = species(ex["ID"], "notitle")
            if grp == "animals" and sp in orig_by_species:
                ox = orig[orig_by_species[sp]]
                octx = Ctx(Rb, ox["image"], ox["prompt"], ctx.true, ctx.fam)  # same digits: true vs familiar of the edit
                co = run_capture(Rb, octx)
                cco = run_capture(Rc, octx)
                acc["running_o"].append(co["running"]); acc["orig_pref"].append(co["pref"])
                cosd = lambda a, b: (1 - (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)).tolist()
                acc["div_b"].append(cosd(cb["states"], co["states"])); acc["div_c"].append(cosd(cc["states"], cco["states"]))
                orig_ctx[i] = octx
            # C. subject swap
            sctx = Ctx(Rb, ex["image"], ex["prompt"], ctx.true, ctx.fam, question=swap_subject(ctx.q))
            acc["swap"].append(pref(Rb, sctx) - cb["pref"])
            if n_item < args.cards:
                cards.append({"grp": grp, "id": ex["ID"], "ctx": ctx, "cb": cb, "cc": cc, "co": co, "image": ex["image"]})
        # D. ablation: components chosen on the first half, tested on the second half and on the originals
        half = len(items) // 2
        mean_dla = np.array(acc["mix"][:half]).mean(0), np.array(acc["mlp"][:half]).mean(0)
        comps = sorted([(mean_dla[0][l], "mix", l) for l in range(L)] + [(mean_dla[1][l], "mlp", l) for l in range(L)])
        abl = {}
        for k in (1, 3, 5, 10):
            top = [(kind, l) for _, kind, l in comps[:k]]
            held = [ablate(Rb, ctx, top) for _, ctx in items[half:]]
            origs = [ablate(Rb, orig_ctx[i], top) for i, _ in items[half:] if i in orig_ctx]
            abl[k] = {"components": [f"{'attn' if (kind == 'mix' and l in full) else ('linear-attn' if kind == 'mix' else 'MLP')} L{l}" for kind, l in top],
                      "heldout_true_wins": float(np.mean([d > 0 for d in held])) if held else None,
                      "heldout_pref": float(np.mean(held)) if held else None,
                      "original_still_4": float(np.mean([d < 0 for d in origs])) if origs else None}
        base_held = [pref(Rb, ctx) for _, ctx in items[half:]]
        abl[0] = {"components": [], "heldout_true_wins": float(np.mean([d > 0 for d in base_held])), "heldout_pref": float(np.mean(base_held)),
                  "original_still_4": (lambda o: float(np.mean([d < 0 for d in o])) if o else None)(
                      [pref(Rb, orig_ctx[i]) for i, _ in items[half:] if i in orig_ctx])}
        r = {"items": [ds[i]["ID"] for i, _ in items],
             "attention_untrained": np.mean(acc["att_b"], 0).tolist(), "attention_trained": np.mean(acc["att_c"], 0).tolist(),
             "running_untrained": np.mean(acc["running_b"], 0).tolist(), "running_trained": np.mean(acc["running_c"], 0).tolist(),
             "running_original": np.mean(acc["running_o"], 0).tolist() if acc["running_o"] else None,
             "dla_mix": np.mean(acc["mix"], 0).tolist(), "dla_mlp": np.mean(acc["mlp"], 0).tolist(), "dla_emb": float(np.mean(acc["emb"])),
             "knockout": {k: np.nanmean(np.array(v), 0).tolist() for k, v in acc["ko"].items()},
             "patch": {k: np.nanmean(np.array(v), 0).tolist() for k, v in acc["patch"].items()},
             "divergence_untrained": np.mean(acc["div_b"], 0).tolist() if acc["div_b"] else None,
             "divergence_trained": np.mean(acc["div_c"], 0).tolist() if acc["div_c"] else None,
             "n_with_original": len(acc["div_b"]),
             "subject_swap_delta": {"mean": float(np.mean(acc["swap"])), "flipped_to_true": float(np.mean([s > -p for s, p in zip(acc["swap"], acc["pref_b"])]))},
             "ablation": abl}
        res[grp] = r
        print(f"[{grp}] attention image/subject by standard layer (untrained):",
              {l: (round(a[0], 3), round(a[1], 3)) for l, a in zip(full, r["attention_untrained"])})
        print(f"[{grp}] ablation:", {k: (v["heldout_true_wins"], v["original_still_4"]) for k, v in abl.items()})
    (out_dir / "diagnose.json").write_text(json.dumps(res, indent=1))
    make_figures(res, cards, full, L, out_dir)
    print(f"wrote {out_dir}/index.html")


# ---------------------------------------------------------------------------------------------------------------------
# Figures

def make_figures(res, cards, full, L, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#d9d8d4"})
    figs = []  # (file, title, caption)
    for grp in ("animals", "other"):
        if grp not in res:
            continue
        r = res[grp]
        name = "VLMBias animals" if grp == "animals" else "VLMBias, other topics"
        # 1. where the answer looks
        fig, axs = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
        for ax, key, title in ((axs[0], "attention_untrained", "untrained model (answers the familiar count)"),
                               (axs[1], "attention_trained", "trained model (answers the true count)")):
            a = np.array(r[key]) * 100
            ax.stackplot(full, a.T, colors=[GCOL[g] for g in GROUPS], labels=[GLAB[g] for g in GROUPS], alpha=0.9)
            ax.set_title(title, loc="left"); ax.set_xlabel("decoder layer (standard-attention layers only)"); ax.set_xticks(full)
        axs[0].set_ylabel("% of the answer position's attention"); axs[0].legend(loc="upper left", fontsize=9, frameon=False)
        fig.tight_layout(); f = f"flow_{grp}.png"; fig.savefig(out / f, dpi=130); plt.close(fig)
        figs.append((f, f"{name}: where the answer position looks, layer by layer",
                     "Each band is the share of attention from the token where the answer is produced. Your hypothesis predicts blue "
                     "(image) early and orange (subject words) late in the untrained model."))
        # 2. tug of war
        fig, ax = plt.subplots(figsize=(11, 4))
        xs = range(1, L + 1)
        ax.plot(xs, r["running_untrained"], color=UNTRAINED, lw=2.2, label="untrained, edited image")
        ax.plot(xs, r["running_trained"], color=TRAINED, lw=2.2, label="trained, edited image")
        if r["running_original"]:
            ax.plot(xs, r["running_original"], color=ORIGINAL, lw=1.8, ls="--", label="untrained, unedited original")
        ax.axhline(0, color="#999", lw=1)
        ax.fill_between(xs, 0, -max(1, abs(min(r["running_untrained"]))) * 1.1, color="#eb6834", alpha=0.06)
        ax.text(1.2, -0.3, "below 0: favors the familiar count", color="#c4511f", fontsize=9, va="top")
        for l in full:
            ax.axvline(l + 1, color="#ddd", lw=0.8, zorder=0)
        ax.set_xlabel("after decoder layer"); ax.set_ylabel("logit(true) − logit(familiar)"); ax.legend(frameon=False, fontsize=9)
        ax.set_title("running preference: true vs familiar count, built up layer by layer", loc="left")
        fig.tight_layout(); f = f"tug_{grp}.png"; fig.savefig(out / f, dpi=130); plt.close(fig)
        figs.append((f, f"{name}: the tug of war between the true and the familiar count",
                     "The running total of every layer's direct push on the answer (it ends at the model's actual preference). Where a "
                     "line drops below 0 is where the familiar count takes over. Grey lines mark the standard-attention layers."))
        # 3. which layers push
        fig, ax = plt.subplots(figsize=(11, 3.8))
        x = np.arange(L)
        ax.bar(x - 0.2, r["dla_mix"], 0.4, color="#2a78d6", label="attention / linear-attention block")
        ax.bar(x + 0.2, r["dla_mlp"], 0.4, color="#eb6834", label="MLP block")
        ax.axhline(0, color="#999", lw=1)
        for l in full:
            ax.axvline(l, color="#ddd", lw=0.8, zorder=0)
        ax.set_xlabel("decoder layer"); ax.set_ylabel("push toward true (+) / familiar (−)"); ax.legend(frameon=False, fontsize=9)
        ax.set_title("which blocks push toward the familiar count (untrained model)", loc="left")
        fig.tight_layout(); f = f"blocks_{grp}.png"; fig.savefig(out / f, dpi=130); plt.close(fig)
        figs.append((f, f"{name}: which blocks push the answer", "Each bar is one block's direct contribution to the answer. Large "
                     "negative bars are where the familiar count is written in; the ablation test removes the biggest ones."))
        # 4. knockout
        fig, ax = plt.subplots(figsize=(11, 3.8))
        keys = list(r["knockout"]); w_ = 0.8 / len(keys)
        cols = {"image<-answer": "#2a78d6", "image<-text": "#8cb8ee", "subject<-answer": "#eb6834", "subject<-text": "#f4ae90"}
        labs = {"image<-answer": "block image, from the answer", "image<-text": "block image, from all text",
                "subject<-answer": "block subject words, from the answer", "subject<-text": "block subject words, from all text"}
        for j, k in enumerate(keys):
            ax.bar(np.arange(len(full)) - 0.4 + w_ * (j + 0.5), r["knockout"][k], w_, color=cols[k], label=labs[k])
        ax.axhline(0, color="#999", lw=1); ax.set_xticks(range(len(full)), [f"L{l}" for l in full])
        ax.set_ylabel("change in preference"); ax.legend(frameon=False, fontsize=8, ncol=2)
        ax.set_title("knockout: what happens if a layer cannot see the image / the subject words", loc="left")
        fig.tight_layout(); f = f"knockout_{grp}.png"; fig.savefig(out / f, dpi=130); plt.close(fig)
        figs.append((f, f"{name}: knockouts (causal test of the attention routes)",
                     "Positive = removing that route moves the answer toward the true count (the route carried the prior). Negative = the "
                     "route carried evidence for the true count."))
        # 5. patching heatmap
        fig, ax = plt.subplots(figsize=(11, 2.6))
        M = np.array([r["patch"][k] for k in ("answer", "image", "text")])
        im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_yticks(range(3), ["answer position", "image tokens", "text tokens"]); ax.set_xlabel("decoder layer patched")
        fig.colorbar(im, ax=ax, label="share of the gap closed")
        ax.set_title("copying one layer from the trained model into the untrained one", loc="left")
        fig.tight_layout(); f = f"patch_{grp}.png"; fig.savefig(out / f, dpi=130); plt.close(fig)
        figs.append((f, f"{name}: where the trained model's fix lives",
                     "Dark = copying that layer's state (at those positions) from the trained model makes the untrained model answer the "
                     "true count. Shows which layers and positions carry the fix."))
        # 6. divergence edited vs original
        if r.get("divergence_untrained"):
            fig, ax = plt.subplots(figsize=(11, 3.6))
            ax.plot(range(L), r["divergence_untrained"], color=UNTRAINED, lw=2.2, label="untrained")
            ax.plot(range(L), r["divergence_trained"], color=TRAINED, lw=2.2, label="trained")
            ax.set_xlabel("decoder layer"); ax.set_ylabel("difference of answer states\n(1 − cosine)"); ax.legend(frameon=False)
            ax.set_title(f"edited vs unedited image: does the answer position register the edit? ({r['n_with_original']} animals)", loc="left")
            fig.tight_layout(); f = f"divergence_{grp}.png"; fig.savefig(out / f, dpi=130); plt.close(fig)
            figs.append((f, f"{name}: does the model register the edit, then erase it?",
                         "How different the answer position is between the edited image and its unedited original. A rise then fall means "
                         "the edit is registered and then erased before the answer."))
    # 7. subject swap + ablation summary
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.8))
    gs = [g for g in ("animals", "other") if g in res]
    axs[0].bar(range(len(gs)), [res[g]["subject_swap_delta"]["mean"] for g in gs], color="#eb6834")
    axs[0].axhline(0, color="#999", lw=1); axs[0].set_xticks(range(len(gs)), gs)
    axs[0].set_ylabel("change in preference"); axs[0].set_title("'this animal' → 'this object': effect on the answer", loc="left")
    for g, c in zip(gs, ("#104281", "#1baf7a")):
        ks = sorted(int(k) for k in res[g]["ablation"])
        ab = res[g]["ablation"]
        pts = [(k, ab[k]["heldout_true_wins"]) for k in ks if ab[k]["heldout_true_wins"] is not None]
        axs[1].plot([k for k, _ in pts], [100 * v for _, v in pts], "-o", color=c, label=f"{g}: edited items now true")
        pts = [(k, ab[k]["original_still_4"]) for k in ks if ab[k]["original_still_4"] is not None]
        if pts:
            axs[1].plot([k for k, _ in pts], [100 * v for _, v in pts], "--s", color=c, label=f"{g}: originals still correct")
    axs[1].set_xlabel("blocks removed (biggest familiar-count pushers)"); axs[1].set_ylabel("%"); axs[1].legend(frameon=False, fontsize=8)
    axs[1].set_title("ablation on held-out items", loc="left"); axs[1].set_ylim(-3, 103)
    fig.tight_layout(); fig.savefig(out / "swap_ablation.png", dpi=130); plt.close(fig)
    figs.append(("swap_ablation.png", "Subject-word swap and ablation (causal tests)",
                 "Left: does removing the word 'animal' (or 'flag', ...) weaken the familiar answer? Right: removing the blocks that push "
                 "hardest toward the familiar count (chosen on other items): do edited items flip to the true count while unedited "
                 "originals stay correct?"))
    # item cards
    card_files = []
    for k, c in enumerate(cards):
        f = f"card_{k:02d}.png"
        item_card(c, full, L, out / f)
        card_files.append((f, c["id"]))
    html = ["<html><head><meta charset='utf-8'><style>body{font-family:-apple-system,Helvetica,sans-serif;max-width:1400px;margin:24px auto;color:#222}"
            "h2{margin-top:36px} p.cap{color:#555;max-width:1100px} img{max-width:100%}</style></head><body>",
            f"<h1>Why the model answers the familiar count</h1><p>Untrained {res['base']} vs trained {res['clean']}. "
            f"Items where the untrained model gives the familiar count and the trained model the true one: {res['n']}. Forced direct answer. "
            "Qwen3.5 is single-stream (image and text tokens in one sequence); attention figures cover its standard-attention layers "
            f"{res['standard_attention_layers']}.</p>"]
    for f, t, cap in figs:
        html += [f"<h2>{t}</h2>", f"<img src='{f}'>", f"<p class='cap'>{cap}</p>"]
    html += ["<h2>Item cards</h2><p class='cap'>One item per card: where the answer position looks in the image at an early and a "
             "late standard-attention layer, how much it attends to each question word at the late layer, and the running "
             "preference for this item.</p>"] + [f"<h3>{i}</h3><img src='{f}'>" for f, i in card_files]
    html += ["<h2>Numbers</h2><pre>" + json.dumps({g: {"ablation": res[g]["ablation"], "subject_swap": res[g]["subject_swap_delta"]}
                                                   for g in ("animals", "other") if g in res}, indent=1) + "</pre></body></html>"]
    (out / "index.html").write_text("\n".join(html))


def item_card(c, full, L, path):
    import matplotlib.pyplot as plt
    ctx, cb = c["ctx"], c["cb"]
    img = c["image"].convert("RGB").resize((IMG, IMG))
    early = 0
    # late overlay: the second-half standard-attention layer that attends to the image most (layer 31 attends ~8%)
    half = len(full) // 2
    late = max(range(half, len(full)), key=lambda li: cb["att"][li].mean(0)[ctx.groups["image"]].sum())
    fig = plt.figure(figsize=(16, 4.2))
    gsp = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.5, 1.6])
    for j, (li, title) in enumerate(((early, "early"), (late, "late"))):
        ax = fig.add_subplot(gsp[0, j])
        a = cb["att"][li].mean(0)[ctx.groups["image"]]
        hm = a.reshape(ctx.side, ctx.side) / (a.max() + 1e-9)
        ax.imshow(img); ax.imshow(np.kron(hm, np.ones((IMG // ctx.side, IMG // ctx.side))), cmap="magma", alpha=0.55)
        share = 100 * cb["att"][li].mean(0)[ctx.groups["image"]].sum()
        ax.set_title(f"layer {full[li]} ({title}): {share:.0f}% of attention on the image", fontsize=9); ax.axis("off")
    ax = fig.add_subplot(gsp[0, 2]); ax.axis("off")
    a = cb["att"][late].mean(0)
    qpos = list(range(ctx.q_span[0], ctx.q_span[1]))
    vals = np.array([a[i] for i in qpos]); vals = vals / (vals.max() + 1e-9)
    ax.set_title(f"question words, attention at layer {full[late]} (darker = more)", fontsize=9, loc="left")
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    x, y = 0.0, 0.92
    for i, v in zip(qpos, vals):
        t = ctx.tokens[i].replace("Ċ", " ").strip() or "·"
        lead = 0.012 if ctx.tokens[i].startswith(" ") and x > 0 else 0.0  # word boundary: small gap
        txt = ax.text(x + lead, y, t, fontsize=10, color="black", va="top",
                      bbox=dict(boxstyle="round,pad=0.12", fc=(0.92, 0.41, 0.2, 0.06 + 0.85 * v), ec="none"))
        bb = txt.get_window_extent(renderer=rend)
        (x0, _), (x1, _) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        if x1 > 1.0 and x > 0:  # wrap
            x, y = 0.0, y - 0.13
            txt.set_position((x, y))
            bb = txt.get_window_extent(renderer=rend)
            (x0, _), (x1, _) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        x = x1 + 0.004
    ax = fig.add_subplot(gsp[0, 3])
    xs = range(1, L + 1)
    ax.plot(xs, cb["running"], color=UNTRAINED, lw=2, label="untrained")
    ax.plot(xs, c["cc"]["running"], color=TRAINED, lw=2, label="trained")
    if c["co"]:
        ax.plot(xs, c["co"]["running"], color=ORIGINAL, lw=1.5, ls="--", label="untrained, original image")
    ax.axhline(0, color="#999", lw=1); ax.set_xlabel("after layer"); ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"true {ctx.true} vs familiar {ctx.fam}: running preference", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
