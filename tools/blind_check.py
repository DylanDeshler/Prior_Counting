# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.4", "transformers>=5.2.0",
#     "vllm>=0.30; sys_platform == 'linux'",
# ]
# ///
"""Blind solvability / text-only leak check (§10.2, §10.3): answer every question with the
language model only (no image).

Pass conditions:
  fixed-answer conflict families (clock, stop sign, star, calendar, piano, Rubik's, numeral clocks;
  color conflict pairs): 45-55% accuracy (familiar answer = all canonicals right, all counterfactuals wrong)
  variable-count families (die, card, domino) and all neutral items: <= 55%

Shares the GPU with eval/training: uses ~30% of GPU memory and skips the vision encoder.

    uv run tools/blind_check.py [--data PATH]
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from answer_parsing import extract_answer, matches  # noqa: E402


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
POOLS = ["e1/conflict", "e1/neutral", "e2/L0/count", "e2/L0/color"]
VARIABLE = {"die", "playing_card", "domino"}
SYSTEM = "The image is not available to you. Answer the question with your best guess."


def require_free_gpu(fraction):
    """vLLM reserves `fraction` of total GPU memory; fail fast with a clear message if it isn't free."""
    import torch
    free, total = torch.cuda.mem_get_info()
    if free < fraction * total:
        raise SystemExit(f"GPU busy: {free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB, this tool needs "
                         f"~{fraction * total / 2**30:.1f} GiB. Rerun when other jobs finish (check with nvidia-smi).")


def main(model="Qwen/Qwen3.5-4B"):
    from vllm import LLM, SamplingParams
    require_free_gpu(0.3)
    recs = []
    for pool in POOLS:
        f = DATA / pool / "records.jsonl"
        if f.exists():
            recs += [dict(json.loads(l), pool=pool) for l in f.open()]
    questions = sorted({r["question"] for r in recs})
    llm = LLM(model=model, language_model_only=True, gpu_memory_utilization=0.3, max_model_len=4096)
    convs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}] for q in questions]
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=512),
                    chat_template_kwargs={"enable_thinking": False})
    answer = {q: extract_answer(o.outputs[0].text) for q, o in zip(questions, outs)}

    groups = defaultdict(list)
    for r in recs:
        groups[(r["pool"], r["family"], r["data_type"])].append(matches(answer[r["question"]], r["answer"]))
    results, ok = [], True
    for (pool, fam, dtype), hits in sorted(groups.items()):
        acc = 100 * sum(hits) / len(hits)
        if dtype == "conflict" and fam not in VARIABLE:
            passed, rule = 45 <= acc <= 55, "45-55%"
        else:
            passed, rule = acc <= 55, "<=55%"
        ok &= passed
        results.append({"pool": pool, "family": fam, "data_type": dtype, "n": len(hits), "accuracy": round(acc, 2),
                        "rule": rule, "pass": passed})
        print(f"{'PASS' if passed else 'FAIL'}  {pool:<14} {fam:<16} {dtype:<9} n={len(hits):<5} acc={acc:5.1f}%  ({rule})")
    (DATA / "reports").mkdir(parents=True, exist_ok=True)
    (DATA / "reports" / "blind_check.json").write_text(json.dumps(
        {"model": model, "system_prompt": SYSTEM, "pass": ok, "groups": results,
         "answers": answer}, indent=1))
    print("blind check", "PASSED" if ok else "FAILED")


if __name__ == "__main__":
    main(*sys.argv[1:])
