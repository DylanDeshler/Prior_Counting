#!/usr/bin/env bash
# Stage 1 of the plan (PROGRESS.md §5, day 1): generate / finish all data, then evaluate the base model.
#
#   tools/stage1_data_and_baseline.sh [data_dir]          # default /data/vlm
#
# Safe to rerun: data generation resumes (`datagen all` skips finished work) and evaluation resumes
# per question. GPU steps wait until enough GPU memory is free instead of failing on a busy GPU.
# Env: MODEL (default Qwen/Qwen3.5-4B), GPU_WAIT_MAX_H (give up waiting after N hours, default 12).
# Full log: <data_dir>/reports/logs/stage1_<timestamp>.log   Results: results/RESULTS.md
set -euo pipefail
DATA=${1:-/data/vlm}
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
cd "$(dirname "$0")/.."

mkdir -p "$DATA/reports/logs"
LOG="$DATA/reports/logs/stage1_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
step() { echo -e "\n==================== $(date '+%F %T')  $*"; }

gpu_free_mib()  { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_total_mib() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1; }

# Block until at least $1 GiB of GPU memory is free (other jobs share this GPU).
wait_for_gpu() {
  local need=$(( $1 * 1024 )) waited=0 max=$(( ${GPU_WAIT_MAX_H:-12} * 60 ))
  while (( $(gpu_free_mib) < need )); do
    (( waited >= max )) && { echo "GPU never had $1 GiB free within ${GPU_WAIT_MAX_H:-12} h; stopping."; exit 1; }
    (( waited % 10 == 0 )) && echo "$(date +%T) waiting for GPU: $(gpu_free_mib) MiB free, need $1 GiB"
    sleep 60; waited=$(( waited + 1 ))
  done
}

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: run this on the GPU box"; exit 1; }
echo "data dir: $DATA   model: $MODEL   log: $LOG"

step "1/4  environment"
uv sync

step "2/4  data: generate / resume everything (sources, build, GPU checks, export, CI checks, report)"
# The GPU checks inside `all` need ~30% of GPU memory; wait for it so they don't get skipped.
wait_for_gpu $(( $(gpu_total_mib) * 30 / 100 / 1024 + 1 ))
uv run python -m datagen --data "$DATA" all || echo "WARNING: some data checks failed; see $DATA/reports/ci_report.json"
for r in point_format_probe blind_check; do
  if [[ ! -f "$DATA/reports/$r.json" ]]; then
    echo "$r did not finish (GPU got busy?); waiting for the GPU and retrying"
    wait_for_gpu $(( $(gpu_total_mib) * 30 / 100 / 1024 + 1 ))
    uv run python -m datagen --data "$DATA" all || true      # resumes: only the missing GPU tools + export/check/report
    break
  fi
done

step "3/4  baseline: core eval suite for $MODEL"
# vLLM takes a fixed share of GPU memory; use what is free now (at most 85%), and at least ~16 GiB.
wait_for_gpu 16
free=$(gpu_free_mib); total=$(gpu_total_mib)
util=$(python3 -c "print(f'{min(0.85, $free / $total - 0.05):.2f}')")
echo "GPU: $free / $total MiB free -> EVAL_GPU_UTIL=${EVAL_GPU_UTIL:-$util}"
EVAL_GPU_UTIL=${EVAL_GPU_UTIL:-$util} tools/eval_suite.sh "$DATA" "$MODEL"

step "4/4  done"
echo "data report:   $DATA/reports.tar.gz   (CI: $DATA/reports/ci_report.json)"
echo "GPU checks:    $DATA/reports/point_format_probe.json  $DATA/reports/blind_check.json"
echo "baseline:      results/RESULTS.md"
echo "full log:      $LOG"
cat results/RESULTS.md 2>/dev/null || true
