#!/usr/bin/env bash
# GRPO (RL) on top of a trained SFT arm, with verifiable rewards: correct total, points on the real units,
# list length = stated total (tools/rl_rewards.py). Then merge and run the core eval suite.
#
#   tools/train_rl.sh <data_dir> <start> [rl_arm ...]
#   tools/train_rl.sh /data/vlm base                                   # base model + point-then-total prompt (default plan)
#   tools/train_rl.sh /data/vlm part-conflict-point-points-qwen3.5-s0  # on top of a trained arm (plain prompts)
#
# start: "base" = Qwen3.5-4B with the point-then-total prompt on every RL question and in the eval (the
# strongest untrained condition, so a gain is measured against the bar that matters); or a run under runs/
# with a merged checkpoint. A new LoRA is trained on top. rl_arm: export arms whose points/train_rl.jsonl rows
# are the RL prompts (counting rows only; default part2-conflict-point). Output: runs/<start>-grpo (base:
# runs/base-pointT-grpo); results <run>-ckpt<N>-merged (+pointT for base).
# Env: RL_STEPS (200), RL_GENERATIONS (8 completions per prompt), RL_PROMPTS (8 prompts per step),
#      RL_LR (1e-5), RL_BETA (0.04, KL to the start policy), RL_MAX_TOKENS (768), RL_VLLM (0: HF generate;
#      1: colocated vLLM rollouts, faster but needs vllm in the ms-swift env), SKIP_EVAL=1.
# Resumable like train_arm.sh: checkpoints every 25 steps; rerun to continue.
set -euo pipefail
DATA=${1:?data dir}; START=${2:?start run, e.g. part-conflict-point-points-qwen3.5-s0}; shift 2
ARMS=("$@"); (( ${#ARMS[@]} )) || ARMS=(part2-conflict-point)
cd "$(dirname "$0")/.."
source tools/common.sh
if [[ "$START" == base ]]; then RUN="runs/base-pointT-grpo"; STYLE=point-total; else RUN="runs/${START}-grpo"; STYLE=default; fi
SWIFT_ENV=${SWIFT_ENV:-.venv-swift}
[[ -x "$SWIFT_ENV/bin/swift" ]] || { echo "missing $SWIFT_ENV (run tools/train_arm.sh once to create it)"; exit 1; }
export PYTHONNOUSERSITE=1 TRITON_DEFAULT_FP_FUSION=1
SWIFT=("$SWIFT_ENV/bin/swift")

if [[ "$START" == base ]]; then MODEL=Qwen/Qwen3.5-4B
else
  MODEL=$(ls -d runs/"$START"/v*/checkpoint-*-merged/.complete 2>/dev/null | sort -V | tail -1)
  [[ -n "$MODEL" ]] || { echo "no merged checkpoint for $START (train it first)"; exit 1; }
  MODEL=$(dirname "$MODEL")
fi
mkdir -p "$RUN/rl_data"
DATASETS=()
for a in "${ARMS[@]}"; do
  f="$DATA/exports/$a/qwen3.5/points/train_rl.jsonl"
  [[ -f "$f" ]] || { echo "missing $f (run: python -m datagen --data $DATA export)"; exit 1; }
  grep -q '"unit_size"' "$f" || { echo "$f predates the RL point reward (re-export)"; exit 1; }
  # counting rows only (color rows have no verifiable count/points); questions rewritten to the eval prompt style
  python3 - "$f" "$RUN/rl_data/$a.jsonl" "$STYLE" <<'PY'
import json, sys
sys.path.insert(0, ".")
from answer_parsing import rewrite_prompt
src, dst, style = sys.argv[1:]
n = 0
with open(src) as fi, open(dst, "w") as fo:
    for line in fi:
        r = json.loads(line)
        if r.get("prior") != "count":
            continue
        u = r["messages"][0]
        u["content"] = "<image>" + rewrite_prompt(u["content"].replace("<image>", "", 1), style)
        fo.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
print(f"{dst}: {n} counting prompts ({style} prompt)")
PY
  DATASETS+=("$RUN/rl_data/$a.jsonl")
done
STEPS=${RL_STEPS:-200}; GEN=${RL_GENERATIONS:-8}; PROMPTS=${RL_PROMPTS:-8}
VLLM=(--use_vllm false)
[[ "${RL_VLLM:-0}" == 1 ]] && VLLM=(--use_vllm true --vllm_mode colocate --vllm_gpu_memory_utilization 0.35 --sleep_level 1)
echo "GRPO from $MODEL on ${ARMS[*]}: $STEPS steps x $PROMPTS prompts x $GEN completions"

ckpt_state() {  # "<path> <step> <max>" of the latest complete checkpoint, or "none 0 0"
  python3 - "$RUN" <<'PY'
import glob, json, sys
best = ("none", 0, 0)
for d in glob.glob(f"{sys.argv[1]}/v*/checkpoint-*"):
    if d.endswith("-merged"):
        continue
    try:
        s = json.load(open(f"{d}/trainer_state.json"))
    except Exception:
        continue
    if s["global_step"] > best[1]:
        best = (d, s["global_step"], s.get("max_steps", 0))
print(*best)
PY
}

if ! ls "$RUN"/v*/checkpoint-*-merged/.complete >/dev/null 2>&1; then
  read -r CKPT STEP MAX <<< "$(ckpt_state)"
  if [[ "$CKPT" == none ]] || (( STEP < MAX )); then
    RESUME=(); [[ "$CKPT" != none ]] && { echo "resuming from $CKPT (step $STEP of $MAX)"; RESUME=(--resume_from_checkpoint "$CKPT"); }
    wait_for_gpu "${TRAIN_GPU_GIB:-30}"
    mkdir -p "$RUN"
    # Completions per step = per_device_train_batch_size x gradient_accumulation_steps = PROMPTS x GEN.
    USE_HF=1 "${SWIFT[@]}" rlhf --rlhf_type grpo \
      --model "$MODEL" --tuner_type lora --target_modules all-linear --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
      --dataset "${DATASETS[@]}" --add_non_thinking_prefix true \
      --external_plugins tools/rl_rewards.py --reward_funcs count_acc point_f1 points_consistent \
      --reward_weights 1.0 1.0 0.2 \
      --num_generations "$GEN" --temperature 1.0 --max_completion_length "${RL_MAX_TOKENS:-768}" \
      --per_device_train_batch_size "$GEN" --gradient_accumulation_steps "$PROMPTS" \
      --learning_rate "${RL_LR:-1e-5}" --beta "${RL_BETA:-0.04}" --max_steps "$STEPS" --warmup_ratio 0.03 \
      --torch_dtype bfloat16 --gradient_checkpointing true --seed 0 \
      --save_strategy steps --save_steps 25 --save_total_limit 20 --logging_steps 5 --log_completions true \
      --dataloader_num_workers 4 --dataset_num_proc 4 --output_dir "$RUN" \
      "${VLLM[@]}" ${RESUME[@]+"${RESUME[@]}"} 2>&1 | tee -a "$RUN/train_raw.log" | grep --line-buffered -E "reward|Train|loss|Error|error" || true
    read -r CKPT STEP MAX <<< "$(ckpt_state)"
  fi
  (( MAX > 0 && STEP >= MAX )) || { echo "GRPO did not finish: latest checkpoint at step $STEP of $MAX (see $RUN/train_raw.log)"; exit 1; }
  rm -rf "${CKPT}-merged"
  USE_HF=1 "${SWIFT[@]}" export --adapters "$CKPT" --merge_lora true
  touch "${CKPT}-merged/.complete"
fi
MERGED=$(dirname "$(ls -d "$RUN"/v*/checkpoint-*-merged/.complete | sort -V | tail -1)")
echo "merged model: $MERGED"
[[ "${SKIP_EVAL:-0}" == 1 ]] && exit 0
wait_for_gpu 16
PROMPT_STYLE=$STYLE EVAL_GPU_UTIL=$(eval_gpu_util) tools/eval_suite.sh "$DATA" "$MERGED"
