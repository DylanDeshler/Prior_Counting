#!/usr/bin/env bash
# Train one LoRA arm with the design-doc settings (§9.2), merge the adapter, and run the core eval suite.
#
#   tools/train_arm.sh <data_dir> <arm> <format> <seed> [model]
#   tools/train_arm.sh /data/vlm conflict-count number 0
#   tools/train_arm.sh /data/vlm neutral-point points 1
#   tools/train_arm.sh /data/vlm C-L2 number 0            # Exp 2 arm (color + shared block)
#
# arm/format must exist under <data_dir>/exports/<arm>/<model_tag>/<format>/ (python -m datagen export).
# Env: BATCH (per-device batch, default 4; effective batch is always 32), SKIP_EVAL=1, EVAL_MAX_TOKENS (4096).
# Resumable: rerun the same command after an interruption; training continues from its latest checkpoint
# (saved every 50 steps), and an interrupted merge is redone.
# Runs share the GPU with other jobs: lower BATCH if training runs out of memory.
set -euo pipefail
DATA=${1:?data dir}; ARM=${2:?arm}; FMT=${3:?format}; SEED=${4:-0}; MODEL=${5:-Qwen/Qwen3.5-4B}
cd "$(dirname "$0")/.."

if [[ "$MODEL" == *"2.5"* ]]; then TAG=qwen2.5-vl; THINK=(); else TAG=qwen3.5; THINK=(--add_non_thinking_prefix true); fi
EXP="$DATA/exports/$ARM/$TAG/$FMT"
RUN="runs/${ARM}-${FMT}-${TAG}-s${SEED}"
BATCH=${BATCH:-4}; ACCUM=$(( (32 + BATCH - 1) / BATCH ))
[[ -f "$EXP/train.jsonl" ]] || { echo "missing $EXP/train.jsonl (run: python -m datagen --data $DATA export)"; exit 1; }
VAL=(--eval_strategy no); [[ -f "$EXP/val.jsonl" ]] && VAL=(--val_dataset "$EXP/val.jsonl" --eval_strategy epoch)

# ms-swift in its own environment (Qwen3.5 needs transformers>=5.9; fla kernels need Python 3.12).
SWIFT=(uv run --no-project --python 3.12 --with "ms-swift>=4.6" --with "transformers>=5.9" --with "qwen_vl_utils>=0.0.14"
       --with flash-linear-attention --with torchvision --with "peft>=0.17" swift)

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
  if [[ "$CKPT" == none ]] || (( STEP < MAX )); then
    # Resumable: checkpoints every 50 steps (~6-8 min); after an interruption, rerunning continues from the
    # latest one with its optimizer state and step (same arguments, as ms-swift requires).
    RESUME=()
    [[ "$CKPT" != none ]] && { echo "resuming from $CKPT (step $STEP of $MAX)"; RESUME=(--resume_from_checkpoint "$CKPT"); }
    USE_HF=1 "${SWIFT[@]}" sft \
      --model "$MODEL" --tuner_type lora --target_modules all-linear \
      --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
      --dataset "$EXP/train.jsonl" "${VAL[@]}" "${THINK[@]}" \
      --torch_dtype bfloat16 --num_train_epochs 2 --learning_rate 1e-4 \
      --lr_scheduler_type cosine --warmup_ratio 0.03 \
      --per_device_train_batch_size "$BATCH" --gradient_accumulation_steps "$ACCUM" \
      --gradient_checkpointing true --max_length 1536 --truncation_strategy delete \
      --seed "$SEED" --data_seed "$SEED" --output_dir "$RUN" \
      --save_strategy steps --save_steps 50 --save_total_limit 2 --logging_steps 10 \
      --dataloader_num_workers 4 --dataset_num_proc 4 "${RESUME[@]}"
    read -r CKPT STEP MAX <<< "$(ckpt_state)"
  fi
  (( MAX > 0 && STEP >= MAX )) || { echo "training did not finish: latest checkpoint at step $STEP of $MAX"; exit 1; }
  rm -rf "${CKPT}-merged"  # leftovers of an interrupted merge
  # Always merge: vLLM's dynamic LoRA gives NaNs past ~400 tokens on Qwen3.5 (vLLM #52568), and point outputs are ~900.
  USE_HF=1 "${SWIFT[@]}" export --adapters "$CKPT" --merge_lora true
  touch "${CKPT}-merged/.complete"  # only a fully written merge counts as done
fi
MERGED=$(dirname "$(ls -d "$RUN"/v*/checkpoint-*-merged/.complete | sort -V | tail -1)")
echo "merged model: $MERGED"
[[ "${SKIP_EVAL:-0}" == 1 ]] && exit 0

tools/eval_suite.sh "$DATA" "$MERGED"
