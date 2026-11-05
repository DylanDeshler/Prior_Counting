#!/usr/bin/env bash
# Stage 2 of the plan (PROGRESS.md §5): the training pilot (Gate 2), then the pointing-prompt baseline (Gate 1).
#
#   tools/stage2_pilot_and_pointing.sh [data_dir]           # default /data/vlm
#
#   1. train the pilot arm (default C-count, seed 0), merge the adapter
#   2. evaluate it on the core suite and check Gate 2 (in-distribution conflict accuracy >= 95%)
#   3. evaluate the BASE model on the core suite with the pointing prompt (results as Qwen3.5-4B+point)
#   4. summary table + results report (plots and samples)
#
# Safe to rerun: training resumes from its latest checkpoint (every 50 steps) or is skipped once merged;
# evaluations resume per question.
# GPU steps wait for free memory (the GPU is shared). The pointing baseline runs even if Gate 2 fails.
# Env: ARM / FORMAT / SEED (pilot arm, default conflict-count / number / 0), BATCH (per-device train batch, 4),
#      TRAIN_GPU_GIB (free memory to wait for before training, 24), GPU_WAIT_MAX_H (12).
set -euo pipefail
DATA=${1:-/data/vlm}
ARM=${ARM:-conflict-count}; FORMAT=${FORMAT:-number}; SEED=${SEED:-0}
BASE=Qwen/Qwen3.5-4B
cd "$(dirname "$0")/.."
source tools/common.sh
start_log "$DATA" stage2

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: run this on the GPU box"; exit 1; }
[[ -f "$DATA/exports/$ARM/qwen3.5/$FORMAT/train.jsonl" ]] || { echo "no exports for $ARM/$FORMAT in $DATA (run stage 1)"; exit 1; }
RUN="runs/${ARM}-${FORMAT}-qwen3.5-s${SEED}"
echo "data dir: $DATA   pilot: $RUN   log: $LOG"
uv sync

step "1/4  train pilot: $ARM ($FORMAT), seed $SEED"
if ls "$RUN"/v*/checkpoint-*-merged/.complete >/dev/null 2>&1; then
  echo "merged checkpoint exists, skipping training"
else
  wait_for_gpu "${TRAIN_GPU_GIB:-24}"
  SKIP_EVAL=1 tools/train_arm.sh "$DATA" "$ARM" "$FORMAT" "$SEED"   # resumes from its latest checkpoint
fi
MERGED=$(dirname "$(ls -d "$RUN"/v*/checkpoint-*-merged/.complete | sort -V | tail -1)")
echo "merged model: $MERGED"

step "2/4  evaluate pilot on the core suite"
wait_for_gpu 16
EVAL_GPU_UTIL=$(eval_gpu_util) tools/eval_suite.sh "$DATA" "$MERGED"

# Gate 2: in-distribution synthetic conflict accuracy >= 95% (the shared large-count block is reported apart).
GATE2=$(python3 - "$RUN" <<'EOF'
import glob, json, sys
run = sys.argv[1].rstrip("/").split("/")[-1]
files = sorted(glob.glob(f"results/{run}-ckpt*_gen-e1-val_conflict*_nothink_summary.json"))
if not files:
    print("UNKNOWN (no validation summary found)"); sys.exit()
t = json.load(open(files[-1]))["by_topic"]
acc = t["e1_val_conflict"]["accuracy"]
shared = t.get("e1_val_shared", {}).get("accuracy")
verdict = "PASS" if acc >= 0.95 else "FAIL"
print(f"{verdict}: in-distribution conflict accuracy {100 * acc:.1f}% (need >= 95%; base 61.4%)"
      + (f", shared block {100 * shared:.1f}% (base 8.9%)" if shared is not None else ""))
EOF
)
echo "Gate 2: $GATE2"
[[ "$GATE2" == FAIL* ]] && echo "Gate 2 failed: fix training (lr / epochs / data loading) before reading VLMBias numbers."

step "3/4  pointing-prompt baseline (base model, Gate 1)"
wait_for_gpu 16
PROMPT_STYLE=point EVAL_GPU_UTIL=$(eval_gpu_util) tools/eval_suite.sh "$DATA" "$BASE"

step "4/4  summary + report"
uv run tools/summarize_results.py results
uv run python tools/results_report.py --data "$DATA" || echo "WARNING: results report failed (results are still in results/)"
echo
echo "Gate 2:        $GATE2"
echo "summary:       results/RESULTS.md   (rows: Qwen3.5-4B, Qwen3.5-4B+point, $(basename "$RUN")-...)"
echo "report:        results/report.tar.gz"
echo "full log:      $LOG"
