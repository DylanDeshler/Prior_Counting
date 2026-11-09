#!/usr/bin/env bash
# Run the experiment queue unattended, in order, until it is done or the machine goes down.
#
#   tools/queue_experiments.sh [data_dir]          # default /data/vlm
#
# Rerun the same command any time (e.g. the next day): finished jobs are skipped, an interrupted arm
# resumes from its last checkpoint, and an interrupted eval resumes per question. A failing job is logged
# and the queue moves on (rerunning retries it). Every GPU step waits for free memory (shared GPU).
#
# Job types:
#   train <arm> <format> <seed>      train a LoRA arm, merge, evaluate on the core suite (plain prompts)
#   eval  <model> <prompt-style>     core suite with a prompt variant; <model> is "base" or a run name
#                                    (e.g. conflict-point-points-qwen3.5-s0); styles: point, point-total, direct
#   real-sweep                       real-photo counting test (tests/R/count) for the base model (plain and
#                                    pointing prompts) and every trained arm that doesn't have it yet
# The real-photo counting pool is built first, on its own (it needs a lot of RAM; running it next to a
# vLLM/training job froze the machine), then skipped on later runs.
# Log: <data_dir>/reports/logs/queue_<timestamp>.log   Results: results/RESULTS.md, results/report.tar.gz
set -uo pipefail
DATA=${1:-/data/vlm}
BASE=Qwen/Qwen3.5-4B
cd "$(dirname "$0")/.."
source tools/common.sh
start_log "$DATA" queue

# Ordered by the pointing hypotheses (PROGRESS.md §5.0):
#   mechanism  - base with a pointing prompt that must end in a stated total: does the list say 5 and the total 4?
#   H3b        - point-then-count training (C-point, N-point): pointing as the default, canonicals recovered?
#   H3c        - distillation: direct answers of point / mixed-format arms vs C-count (forced direct prompt)
#   then the rest of the 2x2 and controls, then seeds.
JOBS=(
  "eval  base                                   point-total"
  "eval  base                                   direct"
  "train conflict-point  points 0"
  "eval  conflict-point-points-qwen3.5-s0       direct"
  "train neutral-point   points 0"
  "eval  neutral-point-points-qwen3.5-s0        direct"
  "train conflict-mixed  mixed  0"
  "eval  conflict-mixed-mixed-qwen3.5-s0        direct"
  "eval  conflict-mixed-mixed-qwen3.5-s0        point-total"
  "train neutral-mixed   mixed  0"
  "eval  neutral-mixed-mixed-qwen3.5-s0         direct"
  "eval  conflict-count-number-qwen3.5-s0       direct"
  "real-sweep"
  "train neutral-count   number 0"
  "train neutral-list    list   0"
  "train dice-only       number 0"
  "train C-L0            number 0"
  "train real-point      points 0"
  "train real-mixed      mixed  0"
  "real-sweep"
  "train conflict-point  points 1"
  "train conflict-mixed  mixed  1"
  "train conflict-count  number 1"
  "train neutral-point   points 1"
  "train neutral-mixed   mixed  1"
  "train neutral-count   number 1"
  "train conflict-point  points 2"
  "train conflict-mixed  mixed  2"
  "train conflict-count  number 2"
  "train neutral-point   points 2"
  "train neutral-mixed   mixed  2"
  "train neutral-count   number 2"
)

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: run this on the GPU box"; exit 1; }
echo "data dir: $DATA   jobs: ${#JOBS[@]}   log: $LOG"
uv sync
# New export arms (e.g. the mixed-format distillation arms) appear after a re-export; it is fast and
# byte-identical for existing arms.
[[ -f "$DATA/exports/conflict-mixed/qwen3.5/mixed/train.jsonl" ]] || uv run python -m datagen --data "$DATA" export

# Real-photo counting pool: build it before any GPU job, never alongside one (see above). A failure is
# logged and the queue continues without the real-photo jobs; rerunning retries the build.
if [[ ! -f "$DATA/tests/R/count/records.jsonl" || ! -f "$DATA/exports/real-point/qwen3.5/points/train.jsonl" ]]; then
  step "build the real-photo counting pool (CPU only, before the GPU jobs)"
  { uv run python -m datagen --data "$DATA" build real && uv run python -m datagen --data "$DATA" export real; } \
    || echo "WARNING: real-photo pool build failed; real-photo jobs will be skipped"
fi

merged_of() { local m; m=$(ls -d runs/"$1"/v*/checkpoint-*-merged/.complete 2>/dev/null | sort -V | tail -1); [[ -n "$m" ]] && dirname "$m"; }
summaries() {  # number of core-suite summaries for a result-file model name
  ls results/"$1"_{main,original,gen-T0-count,gen-e1-val_conflict+e1-val_shared}_nothink_summary.json 2>/dev/null | wc -l
}

declare -a STATUS
status() {  # progress at a glance, including the job that is running right now
  { printf '%s\n' "${STATUS[@]}"; [[ -n "${1:-}" ]] && echo "$1"; } > "$DATA/reports/queue_status.txt"
}
for job in "${JOBS[@]}"; do
  read -r KIND A B C <<< "$job"
  t0=$(date +%s)
  status "RUNNING since $(date '+%F %T'): $job"
  if [[ "$KIND" == real-sweep ]]; then
    [[ -f "$DATA/tests/R/count/records.jsonl" ]] || { STATUS+=("real-sweep: skipped (real-photo pool not built; see the queue log)"); continue; }
    step "real-photo counting test for every model that lacks it"
    rc=0; n=0
    for spec in "Qwen/Qwen3.5-4B|default|Qwen3.5-4B" "Qwen/Qwen3.5-4B|point|Qwen3.5-4B+point" \
                $(for m in runs/*/v*/checkpoint-*-merged/.complete; do [[ -f "$m" ]] || continue; d=$(dirname "$m");
                    r=$(basename "$(dirname "$(dirname "$d")")"); echo "$d|default|$r-$(basename "$d" | sed 's/checkpoint-/ckpt/')"; done); do
      IFS='|' read -r MODEL STYLE NAME <<< "$spec"
      [[ -f "results/${NAME}_gen-R-count_nothink_summary.json" ]] && continue
      wait_for_gpu 16 || { rc=1; continue; }
      uv run run_vlms_are_biased.py --model "$MODEL" --records "$DATA/tests/R/count" --prompt-style "$STYLE" \
        --max-new-tokens "${EVAL_MAX_TOKENS:-4096}" --output-dir results --gpu-memory-utilization "$(eval_gpu_util)" || rc=1
      n=$(( n + 1 ))
    done
    tag="real-sweep ($n models)"
  elif [[ "$KIND" == train ]]; then
    ARM=$A FMT=$B SEED=$C RUN="${A}-${B}-qwen3.5-s${C}" tag="train $A/$B/s$C"
    [[ -f "$DATA/exports/$ARM/qwen3.5/$FMT/train.jsonl" ]] || { STATUS+=("$tag: skipped (no export)"); continue; }
    m=$(merged_of "$RUN")
    if [[ -n "$m" ]] && (( $(summaries "$RUN-$(basename "$m" | sed 's/checkpoint-/ckpt/')") >= 4 )); then
      STATUS+=("$tag: already done"); continue
    fi
    step "$tag"
    if wait_for_gpu "${TRAIN_GPU_GIB:-24}"; then tools/train_arm.sh "$DATA" "$ARM" "$FMT" "$SEED"; rc=$?
    else rc=1; fi
  else
    STYLE=$B tag="eval $A +$B"
    case $STYLE in point) T=+point;; point-total) T=+pointT;; direct) T=+direct;; *) STATUS+=("$tag: unknown style"); continue;; esac
    if [[ "$A" == base ]]; then MODEL=$BASE NAME="Qwen3.5-4B$T"
    else
      MODEL=$(merged_of "$A")
      [[ -n "$MODEL" ]] || { STATUS+=("$tag: skipped ($A not trained yet)"); continue; }
      NAME="$A-$(basename "$MODEL" | sed 's/checkpoint-/ckpt/')$T"
    fi
    (( $(summaries "$NAME") >= 4 )) && { STATUS+=("$tag: already done"); continue; }
    step "$tag"
    if wait_for_gpu 16; then PROMPT_STYLE=$STYLE EVAL_GPU_UTIL=$(eval_gpu_util) tools/eval_suite.sh "$DATA" "$MODEL"; rc=$?
    else rc=1; fi
  fi
  mins=$(( ($(date +%s) - t0) / 60 ))
  if (( rc == 0 )); then STATUS+=("$tag: done in $mins min")
  else STATUS+=("$tag: FAILED after $mins min (see log; rerun the queue to retry)"); fi
  # Re-score every saved answer with the current parser (picks up parser fixes after a git pull), then
  # rebuild RESULTS.md. No GPU, ~1 min.
  uv run tools/rescore_results.py results >/dev/null 2>&1 || true
  status
done

step "queue finished"
printf '  %s\n' "${STATUS[@]}"
uv run tools/rescore_results.py results >/dev/null 2>&1 || true
cat results/RESULTS.md
uv run python tools/results_report.py --data "$DATA" || echo "WARNING: results report failed (results are still in results/)"
echo
echo "summary: results/RESULTS.md   report: results/report.tar.gz   log: $LOG"
echo "download: tar czf queue_results.tgz results runs/*/v*/logging.jsonl runs/*/v*/args.json $LOG"
