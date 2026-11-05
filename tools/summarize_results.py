# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""One markdown table of every evaluated model (base and arms) from results/*_summary.json.

    uv run tools/summarize_results.py [results_dir]      -> prints and writes results/RESULTS.md

Columns (thinking off):
  VLMBias cnt acc / bias   1,992 counting items: accuracy and bias rate (primary Study 1 outcome)
  Orig acc                 VLMBias `original` split: unedited images (guardrail; must stay within 5 pts of base)
  Balanced                 mean of the two above (cannot be gamed by pushing away from the prior)
  T0 can / cf / bias       held-out rendered families: canonical acc, counterfactual acc, bias rate
  T2 can / cf / bias       held-out color objects on real photos (same three numbers)
  Val conflict / shared    in-distribution synthetic validation: conflict pairs (Gate 2: >= 95%) and the shared
                           large-count block (12-51 shapes; hard, reported separately)
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def main(results="results"):
    rows = defaultdict(dict)
    for f in sorted(Path(results).glob("*_summary.json")):
        m = re.match(r"(.+?)_(main|original|gen-.+?)_(think|nothink)_summary$", f.stem)
        if not m or m.group(3) != "nothink":
            continue
        model, tag = m.group(1), m.group(2)
        s = json.loads(f.read_text())
        r = rows[model]
        if tag == "main":
            c = s.get("counting", s["overall"])
            r["vlm_acc"], r["vlm_bias"] = c["accuracy"], c["bias_ratio"]
            if c.get("pointing_rate"):
                r["pt_rate"], r["vlm_list"], r["vlm_list_bias"] = c["pointing_rate"], c["list_accuracy"], c.get("list_bias")
        elif tag == "original":
            r["orig"] = s.get("counting", s["overall"])["accuracy"]  # 326 counting items, illusions excluded
        elif "T0-count" in tag or "T2-color" in tag:
            k = "t0" if "T0" in tag else "t2"
            roles = s.get("by_role", {})
            r[k + "_can"] = roles.get("canonical", {}).get("accuracy")
            r[k + "_cf"] = roles.get("counterfactual", {}).get("accuracy")
            r[k + "_bias"] = roles.get("counterfactual", {}).get("bias_ratio")
            if s["overall"].get("point_f1") is not None:
                r[k + "_f1"] = s["overall"]["point_f1"]
        elif "val_conflict" in tag:
            topics = s.get("by_topic", {})
            r["val_c"] = topics.get("e1_val_conflict", s["overall"])["accuracy"]
            r["val_s"] = topics.get("e1_val_shared", {}).get("accuracy")
    if not rows:
        print("no results yet")
        return
    head = ["model", "VLMBias cnt acc", "bias", "Orig acc", "Balanced", "T0 can", "T0 cf", "T0 bias",
            "T2 can", "T2 cf", "T2 bias", "Val conflict", "Val shared"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for model, r in sorted(rows.items(), key=lambda kv: (kv[0] != "Qwen3.5-4B", kv[0])):
        bal = (r["vlm_acc"] + r["orig"]) / 2 if "vlm_acc" in r and "orig" in r else None
        cells = [r.get("vlm_acc"), r.get("vlm_bias"), r.get("orig"), bal, r.get("t0_can"), r.get("t0_cf"),
                 r.get("t0_bias"), r.get("t2_can"), r.get("t2_cf"), r.get("t2_bias"), r.get("val_c"), r.get("val_s")]
        lines.append("| " + " | ".join([model] + [pct(c) for c in cells]) + " |")
    if any("pt_rate" in r for r in rows.values()):  # pointing readouts (pointing prompt or point-trained arms)
        ph = ["model", "pointing rate", "VLMBias list-length acc", "list bias", "stated-total acc", "T0 point F1"]
        lines += ["", "Pointing readouts (counting items):", "", "| " + " | ".join(ph) + " |", "|" + "---|" * len(ph)]
        for model, r in sorted(rows.items()):
            if "pt_rate" in r:
                lines.append("| " + " | ".join([model] + [pct(r.get(k)) for k in
                                                           ("pt_rate", "vlm_list", "vlm_list_bias", "vlm_acc", "t0_f1")]) + " |")
    table = "\n".join(lines)
    print(table)
    (Path(results) / "RESULTS.md").write_text("# Results (thinking off, greedy)\n\n" + table + "\n\n" + __doc__.split("Columns")[1])


if __name__ == "__main__":
    main(*sys.argv[1:])
