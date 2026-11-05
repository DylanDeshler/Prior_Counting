# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "datasets", "tqdm", "numpy", "pycocotools", "scipy"]
# ///
"""Re-score saved predictions with the current parser and metrics (no GPU, no regeneration).

    uv run tools/rescore_results.py [results_dir]

Rewrites every <results>/*_summary.json from its predictions file (point F1 is kept as stored), then
rebuilds RESULTS.md. Use after a parser/metric fix.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_vlms_are_biased as ev  # noqa: E402
from answer_parsing import extract_answer, matches  # noqa: E402


def rescore(path):
    recs = [json.loads(line) for line in path.open() if line.strip()]
    for r in recs:
        r["pred"] = extract_answer(r["response"])
        r["correct"] = matches(r["pred"], r["ground_truth"])
        r["bias_aligned"] = r["expected_bias"] not in (None, "") and matches(r["pred"], r["expected_bias"])
        pts = ev.parse_points(r["response"])
        r["n_points"] = len(pts)
        r["list_correct"] = bool(pts) and matches(str(len(pts)), r["ground_truth"])
        r["list_familiar"] = bool(pts) and r["expected_bias"] not in (None, "") and matches(str(len(pts)), r["expected_bias"])
        r["agree"] = bool(pts) and r["pred"].isdigit() and int(r["pred"]) == len(pts)
    return ev.summarize(recs)


def main(results="results"):
    for f in sorted(Path(results).glob("*.jsonl")):
        if f.stat().st_size == 0:
            continue
        summ_path = f.with_name(f.name[:-len(".jsonl")] + "_summary.json")
        old = json.loads(summ_path.read_text()) if summ_path.exists() else {}
        new = rescore(f)
        if "config" in old:
            new["config"] = old["config"]
        summ_path.write_text(json.dumps(new, indent=2))
        print(f"rescored {f.name}")
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).parent / "summarize_results.py"), results])


if __name__ == "__main__":
    main(*sys.argv[1:])
