# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.4", "torchvision", "transformers>=5.2.0", "accelerate", "pillow", "numpy", "scikit-learn",
#     "datasets>=3.0", "matplotlib", "pycocotools", "qwen-vl-utils",
# ]
# ///
"""Activation steering: is there a general "count, don't recall" direction in the language model?

    uv run tools/steer_count.py --data /data/vlm [--model Qwen/Qwen3.5-4B] [--layers 12,16,20,24] [--alphas 0.5,1,2]

1. Direction (no VLMBias data): forced direct answers on non-VLMBias counterfactuals with a familiar answer
   (synthetic validation pairs e1/val_conflict + real leg edits parts/conflict). At each layer, the answer-position
   direction d_L = mean(state | model answered the true count) - mean(state | model answered the familiar count).
2. Steering: add alpha x (mean state norm at that layer) x unit(d_L) to the answer-position state at layer L,
   one layer at a time, and read the direct answer (teacher-forced "{", argmax over digits).
3. Evaluation on VLMBias counting items (true and familiar count < 10): accuracy and familiar-answer rate,
   animals vs everything else (the direction saw leg data, so "excl. animals" is the transfer test), and the
   VLMBias original split (unedited images: steering must not break canonical answers).
Writes <data>/reports/steer/<model>/ (steer.json, steer.png).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from answer_parsing import rewrite_prompt  # noqa: E402
from probe_vision import DIRECT, Runner, num, read_jsonl  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--layers", default="auto", help="comma list (auto: 3L/8, L/2, 5L/8, 3L/4)")
    p.add_argument("--alphas", default="1,3,6,10", help="steering strengths, in units of the mean state norm x 0.1 "
                   "(first run used 0.5-2, i.e. at most 0.2 x norm, and had no effect)")
    p.add_argument("--positions", choices=["last", "text"], default="text",
                   help="add the vector at the answer position only, or at every text position after the image")
    p.add_argument("--direction-items", type=int, default=800)
    p.add_argument("--max-items", type=int, default=1000, help="VLMBias main items evaluated per configuration")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


class Steer:
    """Forward hook adding a vector to one decoder layer's output: at the last position, or at every position from
    `start` on (the text after the image)."""
    def __init__(self, layer_module):
        self.vec, self.start = None, None
        self.h = layer_module.register_forward_hook(self.hook)

    def hook(self, module, inputs, output):
        if self.vec is None:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        if self.start is None:
            hs[:, -1, :] += self.vec.to(hs.dtype)
        else:
            hs[:, self.start:, :] += self.vec.to(hs.dtype)
        return output

    def remove(self):
        self.h.remove()


def main():
    args = parse_args()
    data = args.data
    mp = Path(args.model.rstrip("/"))
    name = (f"{mp.parent.parent.name}-{mp.name.replace('checkpoint-', 'ckpt')}" if mp.exists() and mp.name.startswith("checkpoint")
            else args.model.split("/")[-1])
    out = args.out or data / "reports" / "steer" / name / f"{args.positions}-a{args.alphas.replace(',', '_')}"
    out.mkdir(parents=True, exist_ok=True)
    R = Runner(args.model)
    L = R.n_layers
    layers = [3 * L // 8, L // 2, 5 * L // 8, 3 * L // 4] if args.layers == "auto" else [int(x) for x in args.layers.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]
    rng = np.random.default_rng(0)
    from tqdm import tqdm

    # ---- 1. direction from non-VLMBias counterfactuals --------------------------------------------------------
    pool = []
    for p in ("e1/val_conflict", "parts/conflict"):
        recs = [r for r in read_jsonl(data / p / "records.jsonl") if r.get("role") == "counterfactual"
                and r.get("familiar_answer") is not None and r["answer"] < 10 and r["familiar_answer"] < 10]
        rng.shuffle(recs)
        pool += recs[: args.direction_items // 2]
    good, bad, norms = {l: [] for l in layers}, {l: [] for l in layers}, {l: [] for l in layers}
    for r in tqdm(pool, desc="direction states", unit="item"):
        probs, h = R.lens(Image.open(data / r["image"]), rewrite_prompt(r["question"], DIRECT), keep_states=True)
        pred = int(np.argmax(probs[-1]))
        for l in layers:
            norms[l].append(np.linalg.norm(h[l]))
            if pred == r["answer"]:
                good[l].append(h[l])
            elif pred == r["familiar_answer"]:
                bad[l].append(h[l])
    dirs = {}
    for l in layers:
        if good[l] and bad[l]:
            d = np.mean(good[l], 0) - np.mean(bad[l], 0)
            dirs[l] = (torch.tensor(d / np.linalg.norm(d)), float(np.mean(norms[l])))
    res = {"model": args.model, "layers": layers, "alphas": alphas, "positions": args.positions,
           "direction_n": {"true": len(good[layers[0]]), "familiar": len(bad[layers[0]])}}
    print(f"direction from {res['direction_n']['true']} true-count vs {res['direction_n']['familiar']} familiar-count answers")

    # ---- 2/3. steer and evaluate ------------------------------------------------------------------------------
    from datasets import load_dataset
    main_ds = load_dataset("anvo25/vlms-are-biased", split="main")
    orig_ds = load_dataset("anvo25/vlms-are-biased", split="original")
    def items(ds, need_familiar):
        keep = [i for i, (gt, b, t) in enumerate(zip(ds["ground_truth"], ds["expected_bias"], ds["topic"]))
                if num(gt) is not None and num(gt) < 10 and t != "Optical Illusion"
                and (not need_familiar or (num(b) is not None and num(b) < 10))]
        rng.shuffle(keep)
        return keep
    main_ids = items(main_ds, True)[: args.max_items]
    orig_ids = items(orig_ds, False)
    lm = R.model.model.language_model if hasattr(R.model.model, "language_model") else R.model.model
    hook_ref = {}
    def answer_probs(img, q):
        h = hook_ref.get("h")
        if h is not None and args.positions == "text":
            enc = R.inputs(img, q, prefix="{")
            pos, _ = R.image_positions(enc)
            h.start = int(pos[-1]) + 1
        return R.lens(img, q)[-1]
    def evaluate(tag):
        rows = {"animals": [0, 0, 0], "other": [0, 0, 0], "canonical": [0, 0]}
        for i in tqdm(main_ids, desc=f"VLMBias {tag}", unit="item", leave=False):
            ex = main_ds[i]
            pred = int(np.argmax(answer_probs(ex["image"], rewrite_prompt(ex["prompt"], DIRECT))))
            g = rows["animals" if ex["topic"] == "Animals" else "other"]
            g[0] += 1; g[1] += pred == num(ex["ground_truth"]); g[2] += pred == num(ex["expected_bias"])
        for i in tqdm(orig_ids, desc=f"original {tag}", unit="item", leave=False):
            ex = orig_ds[i]
            pred = int(np.argmax(answer_probs(ex["image"], rewrite_prompt(ex["prompt"], DIRECT))))
            rows["canonical"][0] += 1; rows["canonical"][1] += pred == num(ex["ground_truth"])
        f = lambda a, b: round(100 * a / b, 1) if b else None
        return {"animals_acc": f(rows["animals"][1], rows["animals"][0]), "animals_familiar": f(rows["animals"][2], rows["animals"][0]),
                "other_acc": f(rows["other"][1], rows["other"][0]), "other_familiar": f(rows["other"][2], rows["other"][0]),
                "canonical_acc": f(rows["canonical"][1], rows["canonical"][0]),
                "n": {"animals": rows["animals"][0], "other": rows["other"][0], "canonical": rows["canonical"][0]}}
    results = {"baseline": evaluate("baseline")}
    print("baseline:", results["baseline"])
    for l, (d, norm) in dirs.items():
        hook = Steer(lm.layers[l]); hook_ref["h"] = hook
        try:
            for a in alphas:
                hook.vec = (a * 0.1 * norm * d).to(R.dev)
                results[f"L{l}_a{a}"] = evaluate(f"layer {l}, alpha {a}")
                print(f"layer {l} alpha {a}:", results[f"L{l}_a{a}"])
        finally:
            hook.remove(); hook_ref.pop("h", None)
    res["results"] = results
    (out / "steer.json").write_text(json.dumps(res, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, key, title in zip(axs, ("animals_acc", "other_acc", "canonical_acc"),
                              ("VLMBias animals", "VLMBias excl. animals (transfer)", "VLMBias originals (canonical)")):
        ax.axhline(results["baseline"][key], color="gray", lw=1, label="no steering")
        for l in dirs:
            ax.plot(alphas, [results[f"L{l}_a{a}"][key] for a in alphas], marker="o", label=f"layer {l}")
        ax.set_title(title); ax.set_xlabel("steering strength (x 0.1 mean norm)")
    axs[0].set_ylabel("direct-answer accuracy (%)"); axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "steer.png", dpi=130); plt.close(fig)
    print(f"wrote {out}/steer.json")


if __name__ == "__main__":
    main()
