#!/usr/bin/env bash
# GRPO (RL) on top of a trained SFT arm, with verifiable rewards (tools/rl_rewards.py): correct total (weight 1.0,
# the objective), points on the real units (0.2) and list length = stated total (0.1) as small shaping terms. Equal
# count/point weights (first run, 2026-11-24) raised point F1 3x faster than count accuracy. Then merge and eval.
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
#      RL_LR (1e-5), RL_BETA (0.04, KL to the start policy), RL_MAX_TOKENS (768), SKIP_EVAL=1,
#      RL_HF_GENERATE=1 (skip vLLM; roll out with HF generate, ~40 s/step for 64 completions).
# Rollouts use colocated vLLM from a separate env (RL_ENV, default .venv-swift-rl: the SFT env's requirements plus
# vllm), so the SFT env train_arm.sh uses is never modified. If that env can't be built, fails its check, or the
# vLLM run dies, training falls back to HF generate (resuming from the latest checkpoint).
# Resumable like train_arm.sh: checkpoints every 25 steps; rerun to continue.
set -euo pipefail
DATA=${1:?data dir}; START=${2:?start run, e.g. part-conflict-point-points-qwen3.5-s0}; shift 2
ARMS=("$@"); (( ${#ARMS[@]} )) || ARMS=(part2-conflict-point)
cd "$(dirname "$0")/.."
source tools/common.sh
if [[ "$START" == base ]]; then RUN="runs/base-pointT-grpo"; STYLE=point-total; else RUN="runs/${START}-grpo"; STYLE=default; fi
# a non-default RL data set gets its own run (e.g. runs/base-pointT-grpo-part3-nonanimal-point: legs held out)
[[ "${ARMS[*]}" != "part2-conflict-point" ]] && RUN="$RUN-$(IFS=+; echo "${ARMS[*]}")"
SWIFT_ENV=${SWIFT_ENV:-.venv-swift}
[[ -x "$SWIFT_ENV/bin/swift" ]] || { echo "missing $SWIFT_ENV (run tools/train_arm.sh once to create it)"; exit 1; }
# GRPO needs packages the SFT install doesn't pull in (first failure: ModuleNotFoundError: msgspec).
for pkg in msgspec; do
  "$SWIFT_ENV/bin/python" -c "import $pkg" 2>/dev/null || uv pip install --python "$SWIFT_ENV/bin/python" "$pkg"
done
export PYTHONNOUSERSITE=1 TRITON_DEFAULT_FP_FUSION=1
SWIFT=("$SWIFT_ENV/bin/swift")
HF_GEN=(--use_vllm false)
# Context cap for vLLM's KV cache: Qwen3.5's 262k default would not fit next to training (448 px image = 196 tokens).
VLLM_GEN=(--use_vllm true --vllm_mode colocate --vllm_gpu_memory_utilization 0.35 --sleep_level 1
          --vllm_max_model_len 4096 --vllm_limit_mm_per_prompt '{"image": 1}')
MODES=(hf)
if [[ "${RL_HF_GENERATE:-0}" != 1 ]]; then
  RL_ENV=${RL_ENV:-.venv-swift-rl}
  RL_REQS="$(cat "$SWIFT_ENV/.reqs") vllm>=0.30 msgspec"
  if [[ ! -x "$RL_ENV/bin/swift" || "$(cat "$RL_ENV/.reqs" 2>/dev/null)" != "$RL_REQS" ]]; then
    echo "setting up $RL_ENV for vLLM rollouts (first run only)"
    rm -rf "$RL_ENV"
    # shellcheck disable=SC2086  # RL_REQS is a space-separated requirement list
    { uv venv "$RL_ENV" --python 3.12 --managed-python \
        && uv pip install --python "$RL_ENV/bin/python" --torch-backend auto $RL_REQS \
        && echo "$RL_REQS" > "$RL_ENV/.reqs"; } || echo "WARNING: could not build $RL_ENV"
  fi
  if [[ -f "$RL_ENV/.reqs" ]] && "$RL_ENV/bin/python" - <<'PY'
import torch, transformers, vllm, swift, msgspec
from packaging.version import Version
assert Version(transformers.__version__) >= Version("5.9"), f"transformers {transformers.__version__} < 5.9"
assert torch.cuda.is_available(), "no CUDA"
print(f"vLLM rollouts: vllm {vllm.__version__}, transformers {transformers.__version__}, torch {torch.__version__}")
PY
  then MODES=(vllm hf)
  else echo "WARNING: vLLM env not usable; rolling out with HF generate"; fi
fi

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
    for MODE in "${MODES[@]}"; do
    RESUME=(); [[ "$CKPT" != none ]] && { echo "resuming from $CKPT (step $STEP of $MAX)"; RESUME=(--resume_from_checkpoint "$CKPT"); }
    if [[ "$MODE" == vllm ]]; then SWIFT=("$RL_ENV/bin/swift"); ROLLOUT=("${VLLM_GEN[@]}")
    else SWIFT=("$SWIFT_ENV/bin/swift"); ROLLOUT=("${HF_GEN[@]}"); fi
    echo "rollouts: $MODE"
    wait_for_gpu "${TRAIN_GPU_GIB:-30}"
    mkdir -p "$RUN"
    # Completions per step = per_device_train_batch_size x gradient_accumulation_steps = PROMPTS x GEN.
    USE_HF=1 "${SWIFT[@]}" rlhf --rlhf_type grpo \
      --model "$MODEL" --tuner_type lora --target_modules all-linear --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
      --dataset "${DATASETS[@]}" --add_non_thinking_prefix true \
      --external_plugins tools/rl_rewards.py --reward_funcs count_acc point_f1 points_consistent \
      --reward_weights 1.0 0.2 0.1 \
      --num_generations "$GEN" --temperature 1.0 --max_completion_length "${RL_MAX_TOKENS:-768}" \
      --per_device_train_batch_size "$GEN" --gradient_accumulation_steps "$PROMPTS" \
      --learning_rate "${RL_LR:-1e-5}" --beta "${RL_BETA:-0.04}" --max_steps "$STEPS" --warmup_ratio 0.03 \
      --torch_dtype bfloat16 --gradient_checkpointing true --seed 0 \
      --save_strategy steps --save_steps 25 --save_total_limit 20 --logging_steps 5 --log_completions true \
      --dataloader_num_workers 4 --dataset_num_proc 4 --output_dir "$RUN" \
      "${ROLLOUT[@]}" ${RESUME[@]+"${RESUME[@]}"} 2>&1 | tee -a "$RUN/train_raw.log" || true  # unfiltered, like train_arm.sh: a grep here hid the progress bar
    read -r CKPT STEP MAX <<< "$(ckpt_state)"
    (( MAX > 0 && STEP >= MAX )) && break
    echo "WARNING: GRPO with $MODE rollouts stopped at step $STEP of $MAX; trying the next rollout mode"
    done
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
