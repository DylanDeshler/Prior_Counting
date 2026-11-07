#!/usr/bin/env bash
# Train one LoRA arm with the design-doc settings (§9.2), merge the adapter, and run the core eval suite.
#
#   tools/train_arm.sh <data_dir> <arm> <format> <seed> [model]
#   tools/train_arm.sh /data/vlm conflict-count number 0
#   tools/train_arm.sh /data/vlm neutral-point points 1
#   tools/train_arm.sh /data/vlm C-L2 number 0            # Exp 2 arm (color + shared block)
#
# arm/format must exist under <data_dir>/exports/<arm>/<model_tag>/<format>/ (python -m datagen export).
# Env: BATCH (micro-batch; default auto = largest that fits free GPU memory; effective batch is always 32),
#      SKIP_EVAL=1, EVAL_MAX_TOKENS (4096).
# Resumable: rerun the same command after an interruption; training continues from its latest checkpoint
# (saved every 50 steps), and an interrupted merge is redone.
# Runs share the GPU with other jobs: lower BATCH if training runs out of memory.
set -euo pipefail
DATA=${1:?data dir}; ARM=${2:?arm}; FMT=${3:?format}; SEED=${4:-0}; MODEL=${5:-Qwen/Qwen3.5-4B}
cd "$(dirname "$0")/.."

if [[ "$MODEL" == *"2.5"* ]]; then TAG=qwen2.5-vl; THINK=(); else TAG=qwen3.5; THINK=(--add_non_thinking_prefix true); fi
EXP="$DATA/exports/$ARM/$TAG/$FMT"
RUN="runs/${ARM}-${FMT}-${TAG}-s${SEED}"
BATCH=${BATCH:-auto}
source tools/common.sh
# Effective batch is fixed at 32 (design doc; 256 updates/epoch on 8,192 examples). Only the micro-batch
# adapts to the GPU: the largest divisor of 32 whose estimated peak fits in 85% of the free memory, with
# gradient accumulation making up the rest. Per-example memory grows with answer length (points >> {N}).
pick_batch() {
  python3 - "$FMT" "$(gpu_free_mib)" <<'PY'
import sys
fmt, free_gib = sys.argv[1], int(sys.argv[2]) / 1024
per_example = {"number": 0.6, "list": 1.0}.get(fmt, 2.0)   # GiB, with gradient checkpointing, max_length 1536
budget = 0.85 * free_gib - 10                               # ~10 GiB: bf16 weights, LoRA + optimizer, CUDA context
print(next((b for b in (32, 16, 8, 4, 2) if b * per_example <= budget), 1))
PY
}
[[ "$BATCH" == auto ]] && BATCH=$(pick_batch)
[[ -f "$EXP/train.jsonl" ]] || { echo "missing $EXP/train.jsonl (run: python -m datagen --data $DATA export)"; exit 1; }
VAL=(--eval_strategy no); [[ -f "$EXP/val.jsonl" ]] && VAL=(--val_dataset "$EXP/val.jsonl" --eval_strategy epoch)

# ms-swift in its own isolated venv on a uv-managed Python 3.12 (Qwen3.5 needs transformers>=5.9; fla kernels
# need 3.12). Not `uv run --with` on the system Python: that layers onto the interpreter's own site-packages,
# and Ubuntu's system pyOpenSSL/boto3 then break the transformers import. Rebuilt when the requirements change.
SWIFT_ENV=${SWIFT_ENV:-.venv-swift}
SWIFT_REQS=("ms-swift>=4.6" "transformers>=5.9" "qwen_vl_utils>=0.0.14" "flash-linear-attention" "torchvision" "peft>=0.17")
if [[ ! -x "$SWIFT_ENV/bin/swift" || "$(cat "$SWIFT_ENV/.reqs" 2>/dev/null)" != "${SWIFT_REQS[*]}" ]]; then
  echo "setting up the ms-swift environment in $SWIFT_ENV (first run only)"
  rm -rf "$SWIFT_ENV"
  uv venv "$SWIFT_ENV" --python 3.12 --managed-python
  uv pip install --python "$SWIFT_ENV/bin/python" --torch-backend auto "${SWIFT_REQS[@]}"
  echo "${SWIFT_REQS[*]}" > "$SWIFT_ENV/.reqs"
fi
export PYTHONNOUSERSITE=1          # no ~/.local packages either
export TRITON_DEFAULT_FP_FUSION=1  # keep Triton's current default and its deprecation notice quiet
SWIFT=("$SWIFT_ENV/bin/swift")

# The most advanced complete checkpoint across run versions (a resumed run writes into a new v* dir):
# prints "<path> <global_step> <max_steps>", or "none 0 0" if there is none.
ckpt_state() {
  python3 - "$RUN" <<'PY'
import glob, json, sys
best = ("none", 0, 0)
for d in glob.glob(f"{sys.argv[1]}/v*/checkpoint-*"):
    if d.endswith("-merged"):
        continue
    try:
        s = json.load(open(f"{d}/trainer_state.json"))  # written last: a half-saved checkpoint has none
    except Exception:
        continue
    if s["global_step"] > best[1]:
        best = (d, s["global_step"], s.get("max_steps", 0))
print(*best)
PY
}
merged_done() { ls "$RUN"/v*/checkpoint-*-merged/.complete >/dev/null 2>&1; }

if ! merged_done; then
  read -r CKPT STEP MAX <<< "$(ckpt_state)"
  # Resumable: checkpoints every 50 steps (~6-8 min); rerunning continues from the latest one with its
  # optimizer state and step. On out-of-memory (shared GPU), halve the micro-batch, double the accumulation
  # (same effective batch, same samples per step) and resume.
  while [[ "$CKPT" == none ]] || (( STEP < MAX )); do
    ACCUM=$(( 32 / BATCH ))
    echo "micro-batch $BATCH x accumulation $ACCUM = effective batch 32  (GPU free: $(gpu_free_mib) MiB)"
    RESUME=()
    [[ "$CKPT" != none ]] && { echo "resuming from $CKPT (step $STEP of $MAX)"; RESUME=(--resume_from_checkpoint "$CKPT"); }
    mkdir -p "$RUN"
    RAW_START=0; [[ -f "$RUN/train_raw.log" ]] && RAW_START=$(wc -l < "$RUN/train_raw.log")
    if USE_HF=1 "${SWIFT[@]}" sft \
      --model "$MODEL" --tuner_type lora --target_modules all-linear \
      --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
      --dataset "$EXP/train.jsonl" "${VAL[@]}" ${THINK[@]+"${THINK[@]}"} \
      --torch_dtype bfloat16 --num_train_epochs 2 --learning_rate 1e-4 \
      --lr_scheduler_type cosine --warmup_ratio 0.03 \
      --per_device_train_batch_size "$BATCH" --gradient_accumulation_steps "$ACCUM" \
      --gradient_checkpointing true --max_length 1536 --truncation_strategy delete \
      --seed "$SEED" --data_seed "$SEED" --output_dir "$RUN" \
      --save_strategy steps --save_steps 50 --save_total_limit 2 --logging_steps 10 \
      --dataloader_num_workers 4 --dataset_num_proc 4 ${RESUME[@]+"${RESUME[@]}"} 2>&1 \
      | "$SWIFT_ENV/bin/python" tools/train_progress.py "$RUN/train_raw.log"   # one live status line; full output in the raw log
    then
      read -r CKPT STEP MAX <<< "$(ckpt_state)"
      break
    fi
    read -r CKPT STEP MAX <<< "$(ckpt_state)"
    if tail -n +"$((RAW_START + 1))" "$RUN/train_raw.log" | grep -qi "out of memory" && (( BATCH > 1 )); then
      BATCH=$(( BATCH / 2 ))
      echo "out of GPU memory: retrying with micro-batch $BATCH (from step $STEP)"
      continue
    fi
    echo "training failed (see $RUN/train_raw.log)"; exit 1
  done
  (( MAX > 0 && STEP >= MAX )) || { echo "training did not finish: latest checkpoint at step $STEP of $MAX"; exit 1; }
  rm -rf "${CKPT}-merged"  # leftovers of an interrupted merge
  # Always merge: vLLM's dynamic LoRA gives NaNs past ~400 tokens on Qwen3.5 (vLLM #52568), and point outputs are ~900.
  USE_HF=1 "${SWIFT[@]}" export --adapters "$CKPT" --merge_lora true
  touch "${CKPT}-merged/.complete"  # only a fully written merge counts as done
fi
MERGED=$(dirname "$(ls -d "$RUN"/v*/checkpoint-*-merged/.complete | sort -V | tail -1)")
echo "merged model: $MERGED"
[[ "${SKIP_EVAL:-0}" == 1 ]] && exit 0

# The GPU is shared: wait for room, and give vLLM what is free now (unless EVAL_GPU_UTIL is set).
command -v nvidia-smi >/dev/null && wait_for_gpu 16
EVAL_GPU_UTIL=$(eval_gpu_util) tools/eval_suite.sh "$DATA" "$MERGED"
