# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.4", "torchvision", "transformers>=5.2.0", "accelerate", "pillow", "numpy", "scikit-learn",
#     "datasets>=3.0", "matplotlib", "pycocotools", "qwen-vl-utils",
# ]
# ///
"""Where does the count prior take over? Interpretability on Qwen3.5 (HF transformers, one GPU, ~10 GB).

    uv run tools/probe_vision.py --data /data/vlm [--model Qwen/Qwen3.5-4B | runs/<arm>/v*/checkpoint-*-merged]

1. Leg probe: is "this image token shows a leg" linearly decodable from the image-token states, layer by layer?
   Logistic probes on the hidden states at the image positions (layer 0 = the vision encoder's output as the
   language model receives it), trained on Pascal-Part leg masks (parts/count), scored by AUC on held-out
   photos (tests/P/count).
2. Added legs: on the held-out leg edits (tests/P/conflict, "add" counterfactuals) the probe score of the
   pasted leg vs the original legs vs background, split by whether the model answered correctly. A pasted leg
   that scores like a real leg on items the model got wrong = the encoder saw it, the decision dropped it.
3. Logit lens on VLMBias counting items: with a forced direct answer ("{" teacher-forced), the probability of
   the true count vs the familiar count read out from every layer (final norm + LM head). Where the familiar
   answer overtakes the true one is where the prior takes over.
4. Saliency gallery: input-x-gradient of logit(true) - logit(familiar) w.r.t. the image tokens, next to the
   probe heatmap, for VLMBias animals and leg edits.
Writes <data>/reports/probe/<model>/ (probe.json, plots, index.html). Images are resized to 448 x 448, so every
image is a 14 x 14 grid of image tokens (patch 16, 2 x 2 merge).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from answer_parsing import extract_answer, rewrite_prompt  # noqa: E402

IMG = 448
DIRECT = "direct"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--probe-train", type=int, default=800, help="parts/count images for the leg probe")
    p.add_argument("--probe-test", type=int, default=200)
    p.add_argument("--edits", type=int, default=200, help="held-out add-leg edit pairs to score")
    p.add_argument("--lens", type=int, default=600, help="VLMBias counting items for the logit lens")
    p.add_argument("--gallery", type=int, default=12)
    p.add_argument("--layers", default="auto", help="comma list of layers to probe (auto: 0, L/4, L/2, 3L/4, L)")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def rle_decode(m):
    from pycocotools import mask as mu
    return mu.decode({"size": m["size"], "counts": m["counts"].encode()}).astype(bool)


def read_jsonl(path):
    return [json.loads(line) for line in open(path)] if Path(path).exists() else []


class Runner:
    def __init__(self, model_id):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.dev = dev
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16 if dev != "cpu" else torch.float32, device_map=dev).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.proc = AutoProcessor.from_pretrained(model_id)
        ip = self.proc.image_processor
        for k, v in (("min_pixels", IMG * IMG), ("max_pixels", IMG * IMG)):
            if hasattr(ip, k):
                setattr(ip, k, v)
        if hasattr(ip, "size") and isinstance(ip.size, dict):
            ip.size = {**ip.size, "shortest_edge": IMG * IMG, "longest_edge": IMG * IMG}
        cfg = self.model.config
        self.image_token_id = getattr(cfg, "image_token_id", None) or getattr(cfg, "image_token_index", None)
        self.tok = self.proc.tokenizer
        self.digit_ids = [self.tok.convert_tokens_to_ids(str(d)) for d in range(10)]
        lm = self.model.model.language_model if hasattr(self.model.model, "language_model") else self.model.model
        self.norm, self.head = lm.norm, self.model.lm_head
        self.n_layers = cfg.text_config.num_hidden_layers if hasattr(cfg, "text_config") else cfg.num_hidden_layers

    def inputs(self, img, question, prefix=""):
        img = img.convert("RGB").resize((IMG, IMG), Image.BICUBIC)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)
        enc = self.proc(text=[text + prefix], images=[img], return_tensors="pt")
        return {k: v.to(self.dev) for k, v in enc.items()}

    def image_positions(self, enc):
        pos = (enc["input_ids"][0] == self.image_token_id).nonzero().flatten()
        side = int(round(len(pos) ** 0.5))
        return pos, side

    @torch.no_grad()
    def hidden(self, img, question, layers):
        enc = self.inputs(img, question)
        out = self.model(**enc, output_hidden_states=True)
        pos, side = self.image_positions(enc)
        return {l: out.hidden_states[l][0, pos].float().cpu().numpy() for l in layers}, side

    @torch.no_grad()
    def lens(self, img, question):
        """Digit distribution after a teacher-forced "{" at every layer: (n_layers + 1, 10)."""
        enc = self.inputs(img, question, prefix="{")
        out = self.model(**enc, output_hidden_states=True)
        rows = []
        for h in out.hidden_states:
            logits = self.head(self.norm(h[:, -1]))[0].float()
            rows.append(torch.softmax(logits, -1)[self.digit_ids].cpu().numpy())
        return np.stack(rows)

    def saliency(self, img, question, good, bad):
        """input x gradient of logit(good) - logit(bad) on the image tokens -> (side, side) map."""
        enc = self.inputs(img, question, prefix="{")
        emb = self.model.get_input_embeddings()(enc["input_ids"]).detach()
        # let the model place the image features, then take gradients on the merged embeddings
        store = {}
        def hook(module, args, kwargs):
            if kwargs.get("inputs_embeds") is not None:
                e = kwargs["inputs_embeds"].detach().requires_grad_(True)
                kwargs["inputs_embeds"] = e
                store["e"] = e
            return args, kwargs
        lm = self.model.model.language_model if hasattr(self.model.model, "language_model") else self.model.model
        h = lm.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            with torch.enable_grad():
                out = self.model(**enc)
                logits = out.logits[0, -1].float()
                (logits[self.digit_ids[good]] - logits[self.digit_ids[bad]]).backward()
        finally:
            h.remove()
        if "e" not in store or store["e"].grad is None:
            return None
        pos, side = self.image_positions(enc)
        e = store["e"]
        s = (e.grad[0, pos] * e[0, pos]).sum(-1).float().detach().cpu().numpy()
        return s.reshape(side, side)


def token_labels(masks, side, pos_frac=0.3):
    """Per-token leg labels from unit masks: 1 if >= 30% of the cell is leg, 0 if none, -1 otherwise."""
    leg = np.zeros((IMG, IMG), bool)
    for m in masks:
        leg |= m
    cell = IMG // side
    frac = leg[: cell * side, : cell * side].reshape(side, cell, side, cell).mean(axis=(1, 3))
    lab = np.full((side, side), -1)
    lab[frac >= pos_frac] = 1
    lab[frac == 0] = 0
    return lab.ravel(), frac


def num(x):
    try:
        return int(str(x).strip())
    except (TypeError, ValueError):
        return None


def main():
    args = parse_args()
    data = args.data
    mp = Path(args.model.rstrip("/"))  # result-file naming as in run_vlms_are_biased.model_name
    name = (f"{mp.parent.parent.name}-{mp.name.replace('checkpoint-', 'ckpt')}" if mp.exists() and mp.name.startswith("checkpoint")
            else args.model.split("/")[-1])
    out = args.out or data / "reports" / "probe" / name
    out.mkdir(parents=True, exist_ok=True)
    R = Runner(args.model)
    L = R.n_layers
    layers = sorted({0, L // 4, L // 2, 3 * L // 4, L}) if args.layers == "auto" else [int(x) for x in args.layers.split(",")]
    rng = np.random.default_rng(0)
    res = {"model": args.model, "layers": layers}

    # ---- 1. leg probe ------------------------------------------------------------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from tqdm import tqdm
    def collect(recs, n, balance=True):
        X = {l: [] for l in layers}; Y = []
        for r in tqdm(recs[:n], desc="probe features", unit="img"):
            hs, side = R.hidden(Image.open(data / r["image"]), r["question"], layers)
            lab, _ = token_labels([rle_decode(m) for m in r["masks_rle"]], side)
            keep = np.nonzero(lab >= 0)[0]
            if balance:  # all leg tokens + as many background tokens
                posi, nega = keep[lab[keep] == 1], keep[lab[keep] == 0]
                keep = np.concatenate([posi, rng.choice(nega, min(len(nega), max(len(posi), 4)), replace=False)])
            for l in layers:
                X[l].append(hs[l][keep])
            Y.append(lab[keep])
        return {l: np.concatenate(v) for l, v in X.items()}, np.concatenate(Y)
    train = read_jsonl(data / "parts/count/records.jsonl"); rng.shuffle(train)
    test = read_jsonl(data / "tests/P/count/records.jsonl"); rng.shuffle(test)
    Xtr, ytr = collect(train, args.probe_train)
    Xte, yte = collect(test, args.probe_test, balance=False)
    probes, auc = {}, {}
    for l in layers:
        mu, sd = Xtr[l].mean(0), Xtr[l].std(0) + 1e-6
        clf = LogisticRegression(max_iter=2000, C=0.1).fit((Xtr[l] - mu) / sd, ytr)
        probes[l] = (clf, mu, sd)
        auc[l] = float(roc_auc_score(yte, clf.predict_proba((Xte[l] - mu) / sd)[:, 1]))
        print(f"layer {l:3d}: leg-token probe AUC {auc[l]:.3f} (held-out photos)")
    res["probe_auc"] = auc
    l0 = layers[0]

    def heat(img, q, layer=l0):
        hs, side = R.hidden(img, q, [layer])
        clf, mu, sd = probes[layer]
        return clf.predict_proba((hs[layer] - mu) / sd)[:, 1].reshape(side, side)

    # ---- 2. added legs on held-out edits --------------------------------------------------------------------
    edits = [r for r in read_jsonl(data / "tests/P/conflict/records.jsonl")
             if r["role"] == "counterfactual" and r["edit"]["op"] == "add"][: args.edits]
    pf = Path("results") / f"{name}_gen-P-conflict_nothink.jsonl"  # the model's own answers on these items, if evaluated
    preds = {r["ID"]: num(extract_answer(r["response"])) for r in map(json.loads, open(pf))} if pf.exists() else {}
    rows = []
    for r in tqdm(edits, desc="added legs", unit="img"):
        img = Image.open(data / r["image"])
        layer_scores = {}
        masks = [rle_decode(m) for m in r["masks_rle"]]
        # which mask is the pasted leg: the one not present in the canonical partner
        can = next((c for c in read_jsonl(data / "tests/P/conflict/records.jsonl") if c["id"] == r["pair_id"] + "-can"), None)
        if can is None:
            continue
        cmasks = [rle_decode(m) for m in can["masks_rle"]]
        new = [m for m in masks if all((m & c).sum() < 0.5 * m.sum() for c in cmasks)]
        if len(new) != 1:
            continue
        for layer in layers:
            hm = heat(img, r["question"], layer)
            side = hm.shape[0]
            _, f_new = token_labels(new, side)
            _, f_old = token_labels([m for m in masks if m is not new[0]], side)
            layer_scores[layer] = {"added": float(hm[f_new >= 0.3].mean()) if (f_new >= 0.3).any() else None,
                                   "original": float(hm[f_old >= 0.3].mean()) if (f_old >= 0.3).any() else None,
                                   "background": float(hm[(f_new == 0) & (f_old == 0)].mean())}
        rows.append({"id": r["id"], "answer": r["answer"], "pred": preds.get(r["id"]), "scores": layer_scores})
    def avg(sel, layer, k):
        v = [x["scores"][layer][k] for x in sel if x["scores"][layer][k] is not None]
        return float(np.mean(v)) if v else None
    res["added_legs"] = {str(l): {grp: {k: avg(sel, l, k) for k in ("added", "original", "background")}
                                  for grp, sel in (("all", rows), ("model_right", [x for x in rows if x["pred"] == x["answer"]]),
                                                   ("model_wrong", [x for x in rows if x["pred"] is not None and x["pred"] != x["answer"]]))}
                         for l in layers}
    res["added_legs_n"] = len(rows)
    print("added-leg probe scores (layer 0):", json.dumps(res["added_legs"][str(l0)], indent=None))

    # ---- 3. logit lens on VLMBias -------------------------------------------------------------------------
    from datasets import load_dataset
    ds = load_dataset("anvo25/vlms-are-biased", split="main")
    idx = [i for i, (gt, b) in enumerate(zip(ds["ground_truth"], ds["expected_bias"])) if num(gt) is not None and num(b) is not None
           and num(gt) < 10 and num(b) < 10 and num(gt) != num(b)]
    rng.shuffle(idx)
    animals = [i for i in idx if ds[i]["topic"] == "Animals"][: args.lens // 2]
    others = [i for i in idx if ds[i]["topic"] not in ("Animals", "Optical Illusion")][: args.lens // 2]
    lens = {}
    for grp, ids in (("animals", animals), ("other", others)):
        pt, pb = [], []
        for i in tqdm(ids, desc=f"logit lens {grp}", unit="item"):
            ex = ds[i]
            d = R.lens(ex["image"], rewrite_prompt(ex["prompt"], DIRECT))
            pt.append(d[:, num(ex["ground_truth"])]); pb.append(d[:, num(ex["expected_bias"])])
        if pt:
            logodds = np.mean(np.log(np.array(pt) + 1e-9) - np.log(np.array(pb) + 1e-9), 0)
            lens[grp] = {"p_true": np.mean(pt, 0).tolist(), "p_familiar": np.mean(pb, 0).tolist(), "n": len(pt),
                         "log_odds_true_vs_familiar": logodds.tolist(),
                         "final_true_wins": float(np.mean([a[-1] > b[-1] for a, b in zip(pt, pb)]))}
    res["logit_lens"] = lens

    # ---- 4. saliency gallery ------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gal = []
    items = [("vlmbias", ds[i]["image"], rewrite_prompt(ds[i]["prompt"], DIRECT), num(ds[i]["ground_truth"]), num(ds[i]["expected_bias"]))
             for i in animals[: args.gallery]]
    items += [("edit", Image.open(data / r["image"]), r["question"], r["answer"], r["familiar_answer"]) for r in edits[: args.gallery]]
    for k, (kind, img, q, good, bad) in enumerate(items):
        img448 = img.convert("RGB").resize((IMG, IMG))
        hm = heat(img, q)
        sal = None
        try:
            sal = R.saliency(img, q, good, bad)
        except Exception as e:  # saliency is optional
            print("saliency failed:", str(e)[:120])
        fig, ax = plt.subplots(1, 3 if sal is not None else 2, figsize=(9 if sal is not None else 6, 3))
        ax[0].imshow(img448); ax[0].set_title(f"{kind}: true {good}, prior {bad}", fontsize=8)
        ax[1].imshow(img448); ax[1].imshow(np.kron(hm, np.ones((32, 32))), alpha=0.5, cmap="viridis", vmin=0, vmax=1)
        ax[1].set_title("leg probe (vision output)", fontsize=8)
        if sal is not None:
            v = np.abs(sal).max() or 1
            ax[2].imshow(img448); ax[2].imshow(np.kron(sal, np.ones((32, 32))), alpha=0.5, cmap="coolwarm", vmin=-v, vmax=v)
            ax[2].set_title(f"saliency: logit {good} - logit {bad}", fontsize=8)
        for a in ax:
            a.axis("off")
        fig.tight_layout(); fig.savefig(out / f"gallery_{k:02d}.png", dpi=110); plt.close(fig)
        gal.append(f"gallery_{k:02d}.png")

    # plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    ax[0].plot(layers, [auc[l] for l in layers], marker="o"); ax[0].set_ylim(0.5, 1); ax[0].set_xlabel("layer")
    ax[0].set_title("leg-token probe AUC (held-out photos)")
    for grp, st in lens.items():
        ax[1].plot(st["p_true"], label=f"{grp}: true count"); ax[1].plot(st["p_familiar"], "--", label=f"{grp}: familiar count")
    ax[1].set_xlabel("layer (logit lens)"); ax[1].set_ylabel("probability"); ax[1].legend(fontsize=8)
    ax[1].set_title("VLMBias: true vs familiar answer by layer")
    for grp, st in lens.items():
        ax[2].plot(st["log_odds_true_vs_familiar"], label=grp)
    ax[2].axhline(0, color="gray", lw=0.8); ax[2].set_xlabel("layer (logit lens)")
    ax[2].set_ylabel("mean log P(true) - log P(familiar)"); ax[2].legend(fontsize=8)
    ax[2].set_title("> 0: layer prefers the true count")
    fig.tight_layout(); fig.savefig(out / "summary.png", dpi=130); plt.close(fig)
    (out / "probe.json").write_text(json.dumps(res, indent=1))
    html = ["<html><body style='font-family:sans-serif'>", f"<h2>Probe: {args.model}</h2>", "<img src='summary.png' width=900>",
            "<pre>" + json.dumps({"probe_auc": auc, "added_legs_layer0": res["added_legs"][str(l0)],
                                  "logit_lens_final_true_wins": {g: s["final_true_wins"] for g, s in lens.items()}}, indent=1) + "</pre>"]
    html += [f"<img src='{g}' width=700><br>" for g in gal] + ["</body></html>"]
    (out / "index.html").write_text("\n".join(html))
    print(f"wrote {out}/index.html")


if __name__ == "__main__":
    main()
