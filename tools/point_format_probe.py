# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.4", "torchvision", "transformers>=5.2.0", "pillow", "numpy", "scipy", "pycocotools",
#     "vllm>=0.30; sys_platform == 'linux'",
# ]
# ///
"""§12: zero-shot test of the coordinate convention Qwen3.5-4B emits for `point_2d`.

Asks the model to point at every counted unit on L0 images, then scores its points under two
hypotheses -- 0-1000 relative (Qwen3-VL convention) and absolute pixels -- by Hungarian matching
against the ground-truth unit masks. Writes the better convention to data/params.json
(qwen35_point_format) and the details to data/reports/point_format_probe.json.

    uv run tools/point_format_probe.py [--data PATH]
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment


def _data_dir():
    """--data PATH (or $COUNTERPOINT_DATA, default ./data); removes the option from sys.argv."""
    for i, a in enumerate(sys.argv):
        if a == "--data" and i + 1 < len(sys.argv):
            path = sys.argv[i + 1]
            del sys.argv[i:i + 2]
            return Path(path).expanduser().resolve()
        if a.startswith("--data="):
            del sys.argv[i]
            return Path(a.split("=", 1)[1]).expanduser().resolve()
    return Path(os.environ.get("COUNTERPOINT_DATA", Path(__file__).resolve().parent.parent / "data"))


DATA = _data_dir()
PROMPT = 'Point to every {label} in the image. Output JSON: [{{"point_2d": [x, y], "label": "{label}"}}, ...]'
POINT = re.compile(r'"point_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]')


def f1(pred, masks):
    if not pred or not masks:
        return 0.0
    hit = np.array([[m[min(447, max(0, int(y))), min(447, max(0, int(x)))] for m in masks] for x, y in pred], float)
    rows, cols = linear_sum_assignment(-hit)
    tp = hit[rows, cols].sum()
    return 2 * tp / (len(pred) + len(masks))


def main(model="Qwen/Qwen3.5-4B", n=240):
    from vllm import LLM, SamplingParams
    recs = [json.loads(l) for l in (DATA / "e1/conflict/records.jsonl").open()]
    rng = np.random.default_rng(0)
    recs = [recs[i] for i in sorted(rng.choice(len(recs), size=min(n, len(recs)), replace=False))]
    llm = LLM(model=model, gpu_memory_utilization=0.3, max_model_len=8192, limit_mm_per_prompt={"image": 1})
    convs = [[{"role": "user", "content": [
        {"type": "image_pil", "image_pil": Image.open(DATA / r["image"]).convert("RGB")},
        {"type": "text", "text": PROMPT.format(label=r["label"])}]}] for r in recs]
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=2048), chat_template_kwargs={"enable_thinking": False})
    scores = {"rel1000": [], "abs": []}
    emitted = 0
    examples = []
    for r, o in zip(recs, outs):
        text = o.outputs[0].text
        pts = [(float(x), float(y)) for x, y in POINT.findall(text)]
        emitted += bool(pts)
        masks = [mask_utils.decode({"size": m["size"], "counts": m["counts"].encode()}).astype(bool) for m in r["masks_rle"]]
        scores["rel1000"].append(f1([(x / 1000 * 448, y / 1000 * 448) for x, y in pts], masks))
        scores["abs"].append(f1(pts, masks))
        if len(examples) < 20:
            examples.append({"id": r["id"], "answer": r["answer"], "output": text[:400]})
    mean = {k: round(float(np.mean(v)), 4) for k, v in scores.items()}
    best = max(mean, key=mean.get)
    report = {"model": model, "n": len(recs), "emitted_point_json": emitted, "mean_point_f1": mean, "chosen": best,
              "examples": examples}
    (DATA / "reports").mkdir(parents=True, exist_ok=True)
    (DATA / "reports" / "point_format_probe.json").write_text(json.dumps(report, indent=1))
    params_path = DATA / "params.json"
    params = json.loads(params_path.read_text()) if params_path.exists() else {}
    params["qwen35_point_format"] = best
    params_path.write_text(json.dumps(params, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=1))
    print("Re-run exports if the convention changed: python -m datagen export")


if __name__ == "__main__":
    main(*sys.argv[1:])
