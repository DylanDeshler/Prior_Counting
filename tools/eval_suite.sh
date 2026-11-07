#!/usr/bin/env bash
# Core evaluation suite for one model (base or merged arm). Results -> results/<model>_<set>_nothink*.json
#
#   tools/eval_suite.sh <data_dir> [model]        # default model: Qwen/Qwen3.5-4B (the base)
#
# Same generation cap for every model so arms are comparable (EVAL_MAX_TOKENS, default 4096).
# EVAL_GPU_UTIL: share of GPU memory vLLM may take (default 0.85; lower it when the GPU is shared).
# PROMPT_STYLE=point: pointing prompt (point at every item, then the total); saved as model '<name>+point'.
set -euo pipefail
DATA=${1:?data dir}; MODEL=${2:-Qwen/Qwen3.5-4B}; CAP=${EVAL_MAX_TOKENS:-4096}
cd "$(dirname "$0")/.."
EVAL=(uv run run_vlms_are_biased.py --model "$MODEL" --max-new-tokens "$CAP" --output-dir results
      --gpu-memory-utilization "${EVAL_GPU_UTIL:-0.85}" --prompt-style "${PROMPT_STYLE:-default}")

"${EVAL[@]}"                                     # VLMBias main: 1,992 counting items (primary) + 792 illusions
"${EVAL[@]}" --split original                    # guardrail: unedited originals (canonical accuracy)
"${EVAL[@]}" --records "$DATA/tests/T0/count"    # held-out families (traffic lights, snowflakes), rendered
[[ -d "$DATA/tests/T2/color" ]] && "${EVAL[@]}" --records "$DATA/tests/T2/color"   # held-out color objects, real photos
[[ -d "$DATA/tests/R/count" ]] && "${EVAL[@]}" --records "$DATA/tests/R/count"     # real-photo counting with points (held out)
"${EVAL[@]}" --records "$DATA/e1/val_conflict" "$DATA/e1/val_shared"              # in-distribution (Gate 2)
uv run tools/summarize_results.py results
