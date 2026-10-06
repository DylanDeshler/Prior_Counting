# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.4", "torchvision", "transformers>=5.2.0", "pillow"]
# ///
"""§3.5 length rule: the longest target (largest shared-block count, points format) plus prompt
plus image tokens must fit in max_length (1,536) with BOTH tokenizers.

Runs on CPU (tokenizers/processors only). If U doesn't fit, lowers it to the largest count that
fits and records that decision in data/params.json; then rebuild the shared block.

    uv run tools/check_length.py [--data PATH]
"""

import json
import os
import sys
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor


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
MODELS = {"qwen3.5": "Qwen/Qwen3.5-4B", "qwen2.5-vl": "Qwen/Qwen2.5-VL-3B-Instruct"}
QUESTION = "How many triangles are in the image? Answer with a number in curly brackets, e.g., {9}."


def worst_target(n, model):
    # Widest coordinates (3 digits) and the longest shared-block label.
    xy = [999, 999] if model == "qwen3.5" else [447, 447]
    return json.dumps([{"point_2d": xy, "label": "triangle"}] * n) + f"\nTotal: {{{n}}}"


def n_tokens(proc, n, model):
    messages = [{"role": "user", "content": [{"type": "image", "image": Image.new("RGB", (448, 448), (128, 128, 128))},
                                             {"type": "text", "text": QUESTION}]},
                {"role": "assistant", "content": [{"type": "text", "text": worst_target(n, model)}]}]
    out = proc.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")
    return int(out["input_ids"].shape[1])


def main():
    params_path = DATA / "params.json"
    params = json.loads(params_path.read_text())
    U, limit = params["shared_block_max"], params.get("max_length", 1536)
    procs = {m: AutoProcessor.from_pretrained(mid) for m, mid in MODELS.items()}
    tokens = {m: n_tokens(p, U, m) for m, p in procs.items()}
    ok = all(t <= limit for t in tokens.values())
    report = {"U": U, "max_length": limit, "tokens": tokens, "pass": ok}
    if not ok:
        fit = U
        while fit > 12 and any(n_tokens(p, fit, m) > limit for m, p in procs.items()):
            fit -= 1
        params.update(shared_block_max=fit, shared_block_max_source=f"lowered from {U} by the §3.5 length rule")
        params_path.write_text(json.dumps(params, indent=2) + "\n")
        report.update(lowered_U_to=fit, tokens_after={m: n_tokens(p, fit, m) for m, p in procs.items()}, **{"pass": True})
        print(f"U={U} needs {tokens} tokens > {limit}; lowered U to {fit}. Rebuild: python -m datagen build shared e1")
    (DATA / "reports").mkdir(parents=True, exist_ok=True)
    (DATA / "reports" / "length_check.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
