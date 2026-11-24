# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.4", "torchvision", "transformers>=5.2.0", "accelerate", "pillow", "numpy", "scikit-learn",
#     "datasets>=3.0", "matplotlib", "pycocotools", "qwen-vl-utils",
# ]
# ///
"""Inference-only fixes that target the diagnosed mechanism (the true count is formed mid-depth, then late MLPs write
the familiar count). Untrained model by default; forced direct answer ("{" teacher-forced) throughout.

    uv run tools/interventions.py --data /data/vlm [--model Qwen/Qwen3.5-4B] [--only readout,erase,mlp]

1. READOUT: answer the question from a mid-layer count probe instead of the final logits.
   The probe is a 10-way linear classifier on the ANSWER-POSITION state (one vector per item, not image patches; it
   is not the per-patch "is this a leg?" probe). Validity checks, all reported:
     - training data: class-balanced across counts 0-9 from non-VLMBias sources (leg edits v2 with 1-5 legs, natural
       leg crops, synthetic conflict + neutral families); the count histogram per source is written out;
     - held-out accuracy per true count (held-out leg edits + held-out synthetic families) vs a shuffled-label probe;
     - readout layer chosen on held-out non-VLMBias data only;
     - shortcut checks on VLMBias: accuracy separately on 3-legged and 5-legged animals, and on the UNEDITED originals
       (must answer 4 there; reading "not 4" for every animal would pass edited items and fail originals).
2. ERASE: LEACE concept erasure of the FAMILIAR count. Fit on counterfactual items (familiar != true) from synthetic
   families and leg edits, so the familiar count varies across families independently of the true count; erase that
   subspace at every text position after the image, at one layer (12/16/20/24) or at all four. Checks: after erasure the
   familiar count is not linearly decodable on held-out items, while the true count still is.
3. MLP: scale the MLP outputs of layers 27-30 (the diagnosed familiar-count writers) by lambda in {0.75, 0.5, 0.25, 0}
   at every text position after the image.
Every condition is scored on VLMBias (animals, other topics), the unedited originals, and controls that are NOT bias
questions: held-out synthetic canonicals, real-photo counting, visible-leg counting, and real-photo color questions.
Writes <data>/reports/interventions/<model>/ (interventions.json, figures, index.html).
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from answer_parsing import rewrite_prompt  # noqa: E402
from probe_vision import DIRECT, Runner, TorchLogReg, num, read_jsonl  # noqa: E402

COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray", "grey"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--only", default="readout,erase,mlp",
                   help="parts: readout, erase, mlp, readout_iid (in-distribution VLMBias-split readout test)")
    p.add_argument("--per-class", type=int, default=150, help="max training items per count class (probe)")
    p.add_argument("--heldout", type=int, default=600)
    p.add_argument("--vlmbias", type=int, default=1200, help="VLMBias counting items evaluated (true and familiar < 10)")
    p.add_argument("--controls", type=int, default=200, help="items per control set")
    p.add_argument("--erase-fit", type=int, default=1200)
    p.add_argument("--iid-subjects", type=int, default=0, help="cap subjects for readout_iid (smoke tests); 0 = all")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def model_tag(path):
    mp = Path(str(path).rstrip("/"))
    return (f"{mp.parent.parent.name}-{mp.name.replace('checkpoint-', 'ckpt')}" if mp.exists() and mp.name.startswith("checkpoint")
            else str(path).split("/")[-1])


class Item:
    def __init__(self, image, question, truth, familiar=None, kind="count", group="", src=""):
        self.image, self.question, self.truth, self.familiar, self.kind, self.group, self.src = image, question, truth, familiar, kind, group, src


def local_items(data, pool, rng, n=None, roles=None, src=None):
    out = []
    recs = read_jsonl(data / pool / "records.jsonl")
    rng.shuffle(recs)
    for r in recs:
        if roles and r.get("role") not in roles:
            continue
        if r.get("prior") == "count" and isinstance(r.get("answer"), int) and r["answer"] < 10:
            out.append(Item(data / r["image"], r["question"], r["answer"], r.get("familiar_answer"), "count", pool, src or pool))
        elif r.get("prior") == "color":
            out.append(Item(data / r["image"], r["question"], r["answer"], r.get("familiar_answer"), "color", pool, src or pool))
        if n and len(out) >= n:
            break
    return out


class Engine:
    """Forced direct answers with optional interventions (hooks), answer-position states, digit/color readout."""
    def __init__(self, R):
        self.R = R
        self.lm = R.model.model.language_model if hasattr(R.model.model, "language_model") else R.model.model
        self.color_ids = {c: R.tok(c, add_special_tokens=False)["input_ids"][0] for c in COLORS}
        self.hooks = []

    def enc(self, it):
        img = Image.open(it.image) if isinstance(it.image, (str, Path)) else it.image
        q = rewrite_prompt(it.question, DIRECT) if it.kind == "count" else it.question
        e = self.R.inputs(img, q, prefix="{")
        pos, _ = self.R.image_positions(e)
        return e, int(pos[-1]) + 1

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def add_erasure(self, layer, eraser):
        """`layer` indexes hidden_states (hidden_states[l] = output of decoder layer l-1), matching how the eraser was fit."""
        def hook(m, a, o):
            hs = o[0] if isinstance(o, tuple) else o
            s = self._start
            hs[0, s:] = eraser(hs[0, s:].float()).to(hs.dtype)
            return o
        self.hooks.append(self.lm.layers[layer - 1].register_forward_hook(hook))

    def add_mlp_scale(self, layers, lam):
        for l in [l for l in layers if l < len(self.lm.layers)]:
            def hook(m, a, o):
                o[0, self._start:] = o[0, self._start:] * lam
                return o
            self.hooks.append(self.lm.layers[l].mlp.register_forward_hook(hook))

    @torch.no_grad()
    def answer(self, it, states=False):
        e, self._start = self.enc(it)
        out = self.R.model(**e, output_hidden_states=states)
        lg = out.logits[0, -1].float()
        if it.kind == "count":
            pr = torch.softmax(lg, -1)[self.R.digit_ids].cpu().numpy()
            res = {"pred": int(pr.argmax()), "mass": float(pr.sum())}
        else:
            ids = list(self.color_ids.values())
            pr = torch.softmax(lg, -1)[ids].cpu().numpy()
            pred = list(self.color_ids)[int(pr.argmax())]
            res = {"pred": "gray" if pred == "grey" else pred, "mass": float(pr.sum())}
        if states:
            res["ans_states"] = torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy()
            res["text_states"] = [h[0, self._start:] for h in out.hidden_states]  # kept on GPU briefly by caller
        return res


def correct(it, pred):
    return (str(pred).lower() == str(it.truth).lower()) if it.kind == "color" else (pred == it.truth)


def score(eng, sets, desc="score"):
    """{set: {acc, familiar_rate, mass, n}} for the current hooks."""
    res = {}
    bar = tqdm(total=sum(len(v) for v in sets.values()), desc=desc, unit="item")
    for name, items in sets.items():
        ok, fam, mass = [], [], []
        for it in items:
            bar.update()
            a = eng.answer(it)
            ok.append(correct(it, a["pred"])); mass.append(a["mass"])
            if it.familiar is not None and it.kind == "count":
                fam.append(a["pred"] == it.familiar)
        res[name] = {"acc": 100 * float(np.mean(ok)) if ok else None, "familiar": 100 * float(np.mean(fam)) if fam else None,
                     "digit_mass": float(np.mean(mass)) if mass else None, "n": len(items)}
    bar.close()
    return res


class Leace:
    """LEACE (Belrose et al. 2023): least-squares concept erasure. x' = x - W+ P W (x - mu), W = Sigma_xx^-1/2,
    P = projection onto the column space of W Sigma_xz."""
    def __init__(self, X, Z, dev):
        # on the GPU in float64: the CPU version (numpy eigh + 2560^3 products) took many minutes per layer on a 4-core box
        fdev = dev if str(dev).startswith("cuda") else "cpu"  # MPS has no float64
        X = torch.as_tensor(X, dtype=torch.float64, device=fdev); Z = torch.as_tensor(Z, dtype=torch.float64, device=fdev)
        mu = X.mean(0); Xc = X - mu; Zc = Z - Z.mean(0)
        S = Xc.T @ Xc / len(X)
        ev, V = torch.linalg.eigh(S)
        ev = ev.clamp(min=float(ev.max()) * 1e-6)
        W = (V * ev ** -0.5) @ V.T; Winv = (V * ev ** 0.5) @ V.T
        A = W @ (Xc.T @ Zc / len(X))
        U, s_, _ = torch.linalg.svd(A, full_matrices=False)
        U = U[:, s_ > float(s_.max()) * 1e-6]
        M = Winv @ (U @ U.T) @ W
        self.mu = mu.float().to(dev)
        self.M = M.float().to(dev)
        self.rank = U.shape[1]

    def __call__(self, x):
        return x - (x - self.mu) @ self.M.T


def main():
    args = parse_args()
    data = args.data
    out = args.out or data / "reports" / "interventions" / model_tag(args.model)
    out.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(","))
    rng = np.random.default_rng(0)
    R = Runner(args.model)
    eng = Engine(R)
    L = R.n_layers
    from datasets import load_dataset
    from tqdm import tqdm
    res = {"model": args.model}

    # ---- evaluation sets -----------------------------------------------------------------------------------------
    ds = load_dataset("anvo25/vlms-are-biased", split="main")
    orig = load_dataset("anvo25/vlms-are-biased", split="original")
    vb = [i for i, (gt, b, t) in enumerate(zip(ds["ground_truth"], ds["expected_bias"], ds["topic"]))
          if t != "Optical Illusion" and num(gt) is not None and num(b) is not None and num(gt) < 10 and num(b) < 10]
    rng.shuffle(vb)
    vb = vb[: args.vlmbias]
    sets = {"vlmbias_animals": [], "vlmbias_other": []}
    for i in vb:
        ex = ds[i]
        sets["vlmbias_animals" if ex["topic"] == "Animals" else "vlmbias_other"].append(
            Item(ex["image"], ex["prompt"], num(ex["ground_truth"]), num(ex["expected_bias"]), "count", ex["topic"], "vlmbias"))
    sets["originals"] = [Item(orig[k]["image"], orig[k]["prompt"], num(orig[k]["ground_truth"]), None, "count", orig[k]["topic"], "original")
                         for k in range(len(orig)) if orig[k]["topic"] != "Optical Illusion"
                         and num(orig[k]["ground_truth"]) is not None and num(orig[k]["ground_truth"]) < 10]
    # controls (not bias questions)
    sets["ctrl_synthetic_canonical"] = local_items(data, "tests/T0/count", rng, args.controls, roles={"canonical"})
    sets["ctrl_real_photo_count"] = local_items(data, "tests/R/count", rng, args.controls)
    sets["ctrl_visible_legs"] = local_items(data, "tests/P/count", rng, args.controls)
    sets["ctrl_color"] = [it for it in local_items(data, "tests/T2/color", rng, args.controls * 2) if str(it.truth).lower() in COLORS][: args.controls]
    res["eval_sizes"] = {k: len(v) for k, v in sets.items()}
    print("eval sets:", res["eval_sizes"])
    res["baseline"] = score(eng, sets, "baseline: score")
    print("baseline:", {k: v["acc"] for k, v in res["baseline"].items()})

    # ---- 1. READOUT ------------------------------------------------------------------------------------------------
    if "readout" in only:
        pools = [("parts/conflict_v2", None), ("parts/count", None), ("e1/conflict", None), ("e1/neutral", None)]
        by_class, hist = defaultdict(list), defaultdict(Counter)
        for pool, roles in pools:
            for it in local_items(data, pool, rng, None, roles):
                if it.kind == "count":
                    by_class[it.truth].append(it); hist[pool][it.truth] += 1
        train = []
        for c in range(10):
            items = by_class.get(c, [])
            rng.shuffle(items)
            train += items[: args.per_class]
        res["readout_train_histogram"] = {p: dict(sorted(h.items())) for p, h in hist.items()}
        res["readout_train_per_class"] = dict(sorted(Counter(it.truth for it in train).items()))
        held = (local_items(data, "tests/P/conflict", rng, args.heldout // 2) + local_items(data, "tests/T0/count", rng, args.heldout // 2))
        held = [it for it in held if it.kind == "count"]
        print("probe training items per count:", res["readout_train_per_class"])

        def states(items, desc):
            return np.stack([eng.answer(it, states=True)["ans_states"] for it in tqdm(items, desc=desc, unit="item")])
        Htr, Hhe = states(train, "readout: train states"), states(held, "readout: held-out states")
        ytr, yhe = np.array([it.truth for it in train]), np.array([it.truth for it in held])
        Hvb = {k: states(sets[k], f"readout: {k}") for k in ("vlmbias_animals", "vlmbias_other", "originals")}
        layers = list(range(4, L + 1, 2))
        rows = {}
        for l in tqdm(layers, desc="readout probes", unit="layer"):
            mu, sd = Htr[:, l].mean(0), Htr[:, l].std(0) + 1e-6
            clf = TorchLogReg(R.dev).fit((Htr[:, l] - mu) / sd, ytr)
            shuf = TorchLogReg(R.dev).fit((Htr[:, l] - mu) / sd, rng.permutation(ytr))
            ph = clf.predict((Hhe[:, l] - mu) / sd)
            row = {"heldout_acc": float((ph == yhe).mean()), "heldout_acc_shuffled_labels": float((shuf.predict((Hhe[:, l] - mu) / sd) == yhe).mean()),
                   "heldout_per_count": {int(c): float((ph[yhe == c] == c).mean()) for c in np.unique(yhe)}}
            for k, H in Hvb.items():
                p = clf.predict((H[:, l] - mu) / sd)
                truth = np.array([it.truth for it in sets[k]])
                row[k] = {"acc": float((p == truth).mean()),
                          "per_true_count": {int(c): float((p[truth == c] == c).mean()) for c in np.unique(truth)},
                          "pred_hist": dict(Counter(int(x) for x in p))}
                if k != "originals":
                    fam = np.array([it.familiar for it in sets[k]])
                    row[k]["familiar_rate"] = float((p == fam).mean())
            rows[l] = row
        best = max(layers, key=lambda l: rows[l]["heldout_acc"])  # chosen on held-out NON-VLMBias data
        res["readout"] = {"by_layer": rows, "chosen_layer": best,
                          "note": "layer chosen by held-out accuracy on non-VLMBias items (leg edits + synthetic families)"}
        r = rows[best]
        print(f"readout layer {best}: held-out {r['heldout_acc']:.2f} (shuffled {r['heldout_acc_shuffled_labels']:.2f}) | "
              f"animals {r['vlmbias_animals']['acc']:.2f} {r['vlmbias_animals']['per_true_count']} | other {r['vlmbias_other']['acc']:.2f} | "
              f"originals {r['originals']['acc']:.2f}")

    # ---- 1b. IN-DISTRIBUTION READOUT TEST (VLMBias-style images on both sides) ------------------------------------
    # Train and test the answer-position count probe on VLMBias itself, split by SUBJECT (species / flag / logo /
    # board: every resolution and question variant of one image stays on one side). Guards against the label
    # distribution: within a topic the labels are few (animals: 3/5 edited, 4 original), so the probe could guess from
    # topic + "looks edited". Baselines: majority label per (topic, edited?) on the training half; a probe trained on
    # labels shuffled WITHIN (topic, edited?) groups (same category/label distribution, count information removed).
    # Key measure: held-out edited animals, 3- vs 5-legged balanced accuracy (both edited, so only the count separates
    # them); unedited originals must read their true count.
    if "readout_iid" in only:
        import re
        def subject(item_id):
            # "animals-legs-001-horse-notitle-q1-384", "animal_001_horse_notitle_Q1_px384", "animals-legs-001-horse-original-count-na"
            m = re.match(r"([a-z]+?)s?[-_](?:[a-z]+[-_])?\d+[-_]([a-z0-9_]+?)[-_](?:notitle|original|withtitle)", item_id.lower())
            if m:
                return f"{m.group(1)}-{m.group(2)}"
            return re.split(r"[-_](?:notitle|original|withtitle)", item_id.lower())[0]  # e.g. flag_stars_001
        allv = [i for i, (gt, t) in enumerate(zip(ds["ground_truth"], ds["topic"])) if t != "Optical Illusion" and num(gt) is not None and num(gt) < 10]
        items = [(subject(ds[i]["ID"]), Item(ds[i]["image"], ds[i]["prompt"], num(ds[i]["ground_truth"]),
                  num(ds[i]["expected_bias"]), "count", ds[i]["topic"], "edited")) for i in allv]
        items += [(subject(orig[k]["ID"]), Item(orig[k]["image"], orig[k]["prompt"], num(orig[k]["ground_truth"]), None, "count",
                   orig[k]["topic"], "original")) for k in range(len(orig))
                  if orig[k]["topic"] != "Optical Illusion" and num(orig[k]["ground_truth"]) is not None and num(orig[k]["ground_truth"]) < 10]
        subs = sorted({s_ for s_, _ in items}); rng.shuffle(subs)
        if args.iid_subjects:
            subs = subs[: args.iid_subjects]; items = [(s_, it) for s_, it in items if s_ in set(subs)]
        test_subs = set(subs[: len(subs) // 2])
        tr = [it for s_, it in items if s_ not in test_subs]; te = [it for s_, it in items if s_ in test_subs]
        print(f"in-distribution split: {len(subs)} subjects; train {len(tr)} items, test {len(te)} items")
        Htr = np.stack([eng.answer(it, states=True)["ans_states"] for it in tqdm(tr, desc="iid: train states", unit="item")])
        Hte = np.stack([eng.answer(it, states=True)["ans_states"] for it in tqdm(te, desc="iid: test states", unit="item")])
        ytr, yte = np.array([it.truth for it in tr]), np.array([it.truth for it in te])
        gtr = [(it.group, it.src) for it in tr]; gte = [(it.group, it.src) for it in te]
        # baseline 1: majority label per (topic, edited/original) on the training half
        maj = {g: Counter(y for gg, y in zip(gtr, ytr) if gg == g).most_common(1)[0][0] for g in set(gtr)}
        base_pred = np.array([maj.get(g, Counter(ytr).most_common(1)[0][0]) for g in gte])
        # baseline 2 labels: shuffled within (topic, edited/original)
        y_shuf = ytr.copy()
        for g in set(gtr):
            idx = np.array([k for k, gg in enumerate(gtr) if gg == g]); y_shuf[idx] = rng.permutation(ytr[idx])
        sel = {"edited animals": np.array([g == ("Animals", "edited") for g in gte]),
               "unedited animals": np.array([g == ("Animals", "original") for g in gte]),
               "edited other topics": np.array([g[1] == "edited" and g[0] != "Animals" for g in gte]),
               "unedited other topics": np.array([g[1] == "original" and g[0] != "Animals" for g in gte])}
        def summarize(pred):
            out_ = {"all": float((pred == yte).mean())}
            for k, m in sel.items():
                if m.any():
                    out_[k] = float((pred[m] == yte[m]).mean())
            m = sel["edited animals"]
            if m.any():  # 3 vs 5 balanced accuracy on edited animals
                # needs BOTH classes: always answering 5 (75% of edited animals) must score 50%, not 100%
                has_both = all((m & (yte == c)).any() for c in (3, 5))
                out_["edited animals 3-vs-5 balanced"] = (float(np.mean([(pred[m & (yte == c)] == c).mean() for c in (3, 5)]))
                                                         if has_both else None)
                out_["edited animals n by truth"] = {c: int((m & (yte == c)).sum()) for c in (3, 5)}
                out_["edited animals pred_hist"] = dict(Counter(int(x) for x in pred[m]))
            return out_
        rows = {}
        for l in tqdm(list(range(4, L + 1, 2)), desc="iid probes", unit="layer"):
            mu, sd = Htr[:, l].mean(0), Htr[:, l].std(0) + 1e-6
            rows[l] = {"probe": summarize(TorchLogReg(R.dev).fit((Htr[:, l] - mu) / sd, ytr).predict((Hte[:, l] - mu) / sd)),
                       "shuffled_within_topic": summarize(TorchLogReg(R.dev).fit((Htr[:, l] - mu) / sd, y_shuf).predict((Hte[:, l] - mu) / sd))}
        res["readout_iid"] = {"n_subjects": len(subs), "n_train": len(tr), "n_test": len(te),
                              "label_hist_train": {f"{g[0]}|{g[1]}": dict(Counter(int(y) for gg, y in zip(gtr, ytr) if gg == g)) for g in sorted(set(gtr))},
                              "majority_baseline": summarize(base_pred), "model_answer": None, "by_layer": rows}
        mod = np.array([eng.answer(it)["pred"] for it in tqdm(te, desc="iid: model answers", unit="item")])
        res["readout_iid"]["model_answer"] = summarize(mod)
        # layer chosen WITHOUT the test half: 2-fold cross-validation over training-half subjects
        tr_subs = sorted({s_ for s_, it in items if s_ not in test_subs}); rng.shuffle(tr_subs)
        fold = {s_: k % 2 for k, s_ in enumerate(tr_subs)}
        tr_fold = np.array([fold[s_] for s_, it in items if s_ not in test_subs])
        cv = {}
        for l in rows:
            scores = []
            for f in (0, 1):
                a_, b_ = tr_fold != f, tr_fold == f
                mu, sd = Htr[a_, l].mean(0), Htr[a_, l].std(0) + 1e-6
                pb = TorchLogReg(R.dev).fit((Htr[a_, l] - mu) / sd, ytr[a_]).predict((Htr[b_, l] - mu) / sd)
                scores.append(float((pb == ytr[b_]).mean()))
            cv[l] = float(np.mean(scores))
        best = max(cv, key=cv.get)
        res["readout_iid"]["cv_accuracy_by_layer"] = cv
        res["readout_iid"]["chosen_layer"] = best
        res["readout_iid"]["chosen_layer_note"] = "2-fold CV accuracy on the training-half subjects; the test half is not used"
        print("iid majority baseline:", res["readout_iid"]["majority_baseline"])
        print("iid model's own answer:", res["readout_iid"]["model_answer"])
        print(f"iid layer {best} (chosen by CV on the training half): probe", rows[best]["probe"],
              "| shuffled-within-topic", rows[best]["shuffled_within_topic"])

    # ---- 2. ERASE --------------------------------------------------------------------------------------------------
    if "erase" in only:
        fit = [it for it in local_items(data, "e1/conflict", rng, args.erase_fit // 2, roles={"counterfactual"})
               + local_items(data, "parts/conflict_v2", rng, args.erase_fit // 2, roles={"counterfactual"})
               if it.kind == "count" and it.familiar is not None]
        fams = sorted({it.familiar for it in fit})
        res["erase_fit"] = {"n": len(fit), "familiar_classes": dict(Counter(it.familiar for it in fit))}
        print("erasure fit:", res["erase_fit"])
        target_layers = [l for l in (12, 16, 20, 24) if l < L]
        Xs = {l: [] for l in target_layers}; Zs, Ts = [], []
        for it in tqdm(fit, desc="erase: collect text states", unit="item"):
            a = eng.answer(it, states=True)
            idx = np.linspace(0, a["text_states"][0].shape[0] - 1, num=min(8, a["text_states"][0].shape[0])).astype(int)
            for l in target_layers:
                Xs[l].append(a["text_states"][l][idx].float().cpu().numpy())
            onehot = np.eye(len(fams))[fams.index(it.familiar)]
            Zs.append(np.repeat(onehot[None], len(idx), 0)); Ts.append(np.repeat([it.truth], len(idx)))
        Z = np.concatenate(Zs); T = np.concatenate(Ts)
        erasers, checks = {}, {}
        n_fit = int(0.8 * len(Z))
        for l in tqdm(target_layers, desc="erase: fit erasers + checks", unit="layer"):
            X = np.concatenate(Xs[l])
            e = Leace(X[:n_fit], Z[:n_fit], R.dev)
            erasers[l] = e
            Xe = e(torch.tensor(X[n_fit:], dtype=torch.float32, device=R.dev)).cpu().numpy()
            zf = Z.argmax(1)
            def acc(Xtr_, ytr_, Xte_, yte_):
                mu, sd = Xtr_.mean(0), Xtr_.std(0) + 1e-6
                return float((TorchLogReg(R.dev).fit((Xtr_ - mu) / sd, ytr_).predict((Xte_ - mu) / sd) == yte_).mean())
            Xtr_e = e(torch.tensor(X[:n_fit], dtype=torch.float32, device=R.dev)).cpu().numpy()
            checks[l] = {"rank_erased": e.rank,
                         "familiar_decodable_before": acc(X[:n_fit], zf[:n_fit], X[n_fit:], zf[n_fit:]),
                         "familiar_decodable_after": acc(Xtr_e, zf[:n_fit], Xe, zf[n_fit:]),
                         "true_decodable_before": acc(X[:n_fit], T[:n_fit], X[n_fit:], T[n_fit:]),
                         "true_decodable_after": acc(Xtr_e, T[:n_fit], Xe, T[n_fit:])}
            print(f"erase layer {l}:", checks[l])
        res["erase_checks"] = checks
        res["erase"] = {}
        for name, ls in [(f"L{l}", [l]) for l in target_layers] + [("+".join(f"L{l}" for l in target_layers), target_layers)]:
            for l in ls:
                eng.add_erasure(l, erasers[l])
            try:
                res["erase"][name] = score(eng, sets, f"erase {name}: score")
            finally:
                eng.clear()
            print(f"erase {name}:", {k: v["acc"] for k, v in res["erase"][name].items()})

    # ---- 3. MLP SCALING --------------------------------------------------------------------------------------------
    if "mlp" in only:
        res["mlp"] = {}
        for lam in (0.75, 0.5, 0.25, 0.0):
            eng.add_mlp_scale([27, 28, 29, 30], lam)
            try:
                res["mlp"][str(lam)] = score(eng, sets, f"MLP 27-30 x{lam}: score")
            finally:
                eng.clear()
            print(f"MLP 27-30 x{lam}:", {k: v["acc"] for k, v in res["mlp"][str(lam)].items()})

    name = "interventions.json" if only == {"readout", "erase", "mlp"} else f"interventions_{'_'.join(sorted(only))}.json"
    (out / name).write_text(json.dumps(res, indent=1, default=float))
    if name == "interventions.json":
        figures(res, out)
    else:
        iid_figure(res, out)
    print(f"wrote {out}/index.html")


def iid_figure(res, out):
    if "readout_iid" not in res:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = res["readout_iid"]; ls = sorted(int(l) for l in r["by_layer"])
    g = lambda l: r["by_layer"][l] if l in r["by_layer"] else r["by_layer"][str(l)]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
    for ax, key, title in ((axs[0], "edited animals 3-vs-5 balanced", "edited animals: 3 vs 5 legs (balanced accuracy)"),
                           (axs[1], "unedited animals", "unedited animals: reads 4?")):
        ax.plot(ls, [100 * (g(l)["probe"].get(key) or 0) for l in ls], "-o", color="#104281", label="count probe (VLMBias train half)")
        ax.plot(ls, [100 * (g(l)["shuffled_within_topic"].get(key) or 0) for l in ls], ":", color="#a8a7a2", label="labels shuffled within topic")
        for lab, v, c in (("majority label per topic", r["majority_baseline"].get(key), "#eb6834"), ("model's own answer", r["model_answer"].get(key), "#1baf7a")):
            if v is not None:
                ax.axhline(100 * v, color=c, ls="--", lw=1.2, label=lab)
        ax.set_title(title, loc="left"); ax.set_xlabel("layer of the answer-position state"); ax.set_ylim(0, 105)
    axs[0].set_ylabel("%"); axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "readout_iid.png", dpi=130); plt.close(fig)


def figures(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#d9d8d4"})
    figs = []
    SETS = [("vlmbias_animals", "VLMBias animals", "#104281"), ("vlmbias_other", "VLMBias other topics", "#2a78d6"),
            ("originals", "Unedited originals", "#1baf7a"), ("ctrl_synthetic_canonical", "Control: synthetic canonical", "#a8a7a2"),
            ("ctrl_real_photo_count", "Control: real-photo counting", "#6250d6"), ("ctrl_visible_legs", "Control: visible legs", "#e87ba4"),
            ("ctrl_color", "Control: color questions", "#eda100")]
    base = res["baseline"]
    if "readout" in res:
        rows = res["readout"]["by_layer"]; ls = sorted(int(l) for l in rows)
        g = lambda l: rows[l] if l in rows else rows[str(l)]
        fig, axs = plt.subplots(1, 2, figsize=(14, 4.2))
        axs[0].plot(ls, [100 * g(l)["heldout_acc"] for l in ls], "-o", color="#52514e", label="held-out (non-VLMBias): probe")
        axs[0].plot(ls, [100 * g(l)["heldout_acc_shuffled_labels"] for l in ls], ":", color="#a8a7a2", label="held-out: shuffled-label control")
        for k, lab, c in SETS[:3]:
            axs[0].plot(ls, [100 * g(l)[k]["acc"] for l in ls], "-", color=c, label=f"{lab}: probe readout")
            axs[0].axhline(base[k]["acc"], color=c, ls="--", lw=1)
        axs[0].axvline(res["readout"]["chosen_layer"], color="#eb6834", lw=1)
        axs[0].set_xlabel("layer of the answer-position state"); axs[0].set_ylabel("accuracy (%)"); axs[0].legend(fontsize=8, frameon=False)
        axs[0].set_title("reading the count from a mid layer (dashed: the model's own answer)", loc="left")
        b = res["readout"]["chosen_layer"]; r = g(b)
        labels, vals = [], []
        for c, v in sorted(r["vlmbias_animals"]["per_true_count"].items(), key=lambda x: int(x[0])):
            labels.append(f"VLMBias animals,\n{c} legs"); vals.append(100 * v)
        for c, v in sorted(r["originals"]["per_true_count"].items(), key=lambda x: int(x[0])):
            labels.append(f"originals,\ntrue {c}"); vals.append(100 * v)
        axs[1].bar(range(len(vals)), vals, color="#104281")
        axs[1].set_xticks(range(len(vals)), labels, fontsize=8); axs[1].set_ylim(0, 105)
        axs[1].set_title(f"shortcut check at layer {b}: right on 3- AND 5-legged animals, and on originals?", loc="left", fontsize=10)
        fig.tight_layout(); fig.savefig(out / "readout.png", dpi=130); plt.close(fig)
        figs.append(("readout.png", "1. Reading the count from a mid layer",
                     "Left: accuracy if the answer is read from a count probe at each layer instead of the model's final answer (dashed lines). "
                     "The probe never saw VLMBias; the orange line is the layer chosen on held-out non-VLMBias data. Right: the shortcut check."))
    for key, title, xlab, order in (("erase", "2. Erasing the familiar count (LEACE)", "erasure layer(s)", None),
                                    ("mlp", "3. Turning down the late MLPs (layers 27-30)", "MLP output scale", ["0.75", "0.5", "0.25", "0.0"])):
        if key not in res:
            continue
        conds = order or list(res[key])
        fig, ax = plt.subplots(figsize=(13, 4.2))
        x = np.arange(len(conds) + 1); w = 0.8 / len(SETS)
        for j, (k, lab, c) in enumerate(SETS):
            ys = [base[k]["acc"]] + [res[key][cd][k]["acc"] for cd in conds]
            ax.bar(x - 0.4 + w * (j + 0.5), [np.nan if y is None else y for y in ys], w, color=c, label=lab)
        ax.set_xticks(x, ["none"] + conds); ax.set_xlabel(xlab); ax.set_ylabel("accuracy (%)")
        ax.legend(fontsize=8, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
        fig.tight_layout(); f = f"{key}.png"; fig.savefig(out / f, dpi=130, bbox_inches="tight"); plt.close(fig)
        figs.append((f, title, "Accuracy on the bias questions (blue/green) and on control questions that are not bias questions (grey, "
                     "violet, pink, yellow). A useful intervention raises the blues without lowering the rest."))
    html = ["<html><head><meta charset='utf-8'><style>body{font-family:-apple-system,Helvetica,sans-serif;max-width:1400px;margin:24px auto}"
            "p{color:#555;max-width:1100px} img{max-width:100%}</style></head><body><h1>Inference-only interventions</h1>",
            f"<p>{res['model']}; forced direct answers. Eval sizes: {res['eval_sizes']}</p>"]
    for f, t, cap in figs:
        html += [f"<h2>{t}</h2><img src='{f}'><p>{cap}</p>"]
    html += ["<h2>Numbers</h2><pre>" + json.dumps({k: res[k] for k in res if k not in ("readout",)}, indent=1, default=float)[:20000] + "</pre></body></html>"]
    (out / "index.html").write_text("\n".join(html))


if __name__ == "__main__":
    main()
