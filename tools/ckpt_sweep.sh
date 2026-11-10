#!/usr/bin/env bash
# Overfitting check for one trained arm: accuracy at 25/50/75/100% of training, on data the arm never
# trained on, next to the validation loss logged during training.
#
#   tools/ckpt_sweep.sh <data_dir> <run>        # e.g. tools/ckpt_sweep.sh /data/vlm real-point-points-qwen3.5-s0
#
# For each intermediate checkpoint: merge it, evaluate on
#   real/val       300 held-out real photos (out of distribution for the synthetic arms; never trained on)
#   tests/T0/count held-out synthetic families
# then delete the merged weights (~9 GB each; the results stay). The final checkpoint reuses its merged model.
# Accuracy that peaks early and then falls on real/val while T0 keeps rising = overfitting to the synthetic
# layouts. VLMBias is deliberately not used here: picking a checkpoint on it would be tuning on the test set.
# Needs the arm's intermediate checkpoints (kept since save_total_limit 20; arms trained earlier only have
# their last two). Resumable: finished checkpoints are skipped. Prints a table; results in results/.
set -euo pipefail
DATA=${1:?data dir}; RUN_NAME=${2:?run name, e.g. real-point-points-qwen3.5-s0}
cd "$(dirname "$0")/.."
source tools/common.sh
RUN="runs/$RUN_NAME"
SWIFT_ENV=${SWIFT_ENV:-.venv-swift}
export PYTHONNOUSERSITE=1
[[ -d "$DATA/real/val" ]] || { echo "missing $DATA/real/val (run: python -m datagen --data $DATA export)"; exit 1; }

# Checkpoints nearest to 25/50/75% of training, plus the final one: "<path> <step>" per line.
mapfile -t PICKS < <(python3 - "$RUN" <<'PY'
import glob, json, re, sys
ck = {}
for d in glob.glob(f"{sys.argv[1]}/v*/checkpoint-*"):
    m = re.search(r"checkpoint-(\d+)$", d)
    if m and not d.endswith("-merged"):
        try:
            s = json.load(open(f"{d}/trainer_state.json"))
        except Exception:
            continue
        ck[int(m.group(1))] = (d, s.get("max_steps", 0))
if not ck:
    sys.exit(0)
final = max(ck)
total = ck[final][1] or final
picks = sorted({min(ck, key=lambda s: abs(s - f * total)) for f in (0.25, 0.5, 0.75)} | {final})
for s in picks:
    print(ck[s][0], s)
PY
)
(( ${#PICKS[@]} )) || { echo "no checkpoints under $RUN"; exit 1; }
(( ${#PICKS[@]} >= 3 )) || echo "note: only ${#PICKS[@]} checkpoint(s) kept for this run (trained before checkpoints were kept)"

done_for() { [[ -f "results/$1_gen-real-val_nothink_summary.json" && -f "results/$1_gen-T0-count_nothink_summary.json" ]]; }
EVAL=(uv run run_vlms_are_biased.py --max-new-tokens "${EVAL_MAX_TOKENS:-4096}" --output-dir results)

for pick in "${PICKS[@]}"; do
  read -r CKPT STEP <<< "$pick"
  NAME="$RUN_NAME-ckpt$STEP-merged"
  done_for "$NAME" && { echo "step $STEP: done"; continue; }
  step "sweep $RUN_NAME: checkpoint $STEP"
  MERGED="${CKPT}-merged"; TEMP=0
  if [[ ! -f "$MERGED/.complete" ]]; then
    rm -rf "$MERGED"
    wait_for_gpu 16
    USE_HF=1 "$SWIFT_ENV/bin/swift" export --adapters "$CKPT" --merge_lora true
    TEMP=1
  fi
  wait_for_gpu 16
  UTIL=$(eval_gpu_util)
  "${EVAL[@]}" --model "$MERGED" --records "$DATA/real/val" --gpu-memory-utilization "$UTIL"
  "${EVAL[@]}" --model "$MERGED" --records "$DATA/tests/T0/count" --gpu-memory-utilization "$UTIL"
  (( TEMP )) && rm -rf "$MERGED"   # intermediate merges are only needed for this sweep
done

# Table: accuracy per checkpoint + the logged real-photo validation loss at that step.
python3 - "$RUN" "$RUN_NAME" <<'PY'
import glob, json, sys
run, name = sys.argv[1], sys.argv[2]
loss = {}
for f in sorted(glob.glob(f"{run}/v*/logging.jsonl")):
    for line in open(f):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        step = r.get("global_step/max_steps", "").split("/")[0] if isinstance(r.get("global_step/max_steps"), str) else r.get("global_step")
        if "eval_loss" in r and step not in (None, ""):
            loss[int(step)] = r["eval_loss"]
def summ(tag, step):
    try:
        return json.load(open(f"results/{name}-ckpt{step}-merged_{tag}_nothink_summary.json"))
    except OSError:
        return None
steps = sorted({int(p.split("-ckpt")[1].split("-")[0]) for p in glob.glob(f"results/{name}-ckpt*-merged_gen-real-val_nothink_summary.json")})
print(f"\n{name}: accuracy on data never trained on, by checkpoint")
print(f"{'step':>6} {'real-val loss':>13} {'real-val acc':>12} {'real-val F1':>11} {'T0 can':>7} {'T0 cf':>6} {'T0 bias':>7}")
for s in steps:
    rv, t0 = summ("gen-real-val", s), summ("gen-T0-count", s)
    o = rv["overall"] if rv else {}
    br = (t0 or {}).get("by_role", {})
    pct = lambda x: "-" if x is None else f"{100 * x:.1f}"
    f1 = o.get("point_f1")
    print(f"{s:>6} {loss.get(s, float('nan')):>13.4f} {pct(o.get('accuracy')):>12} {('-' if f1 is None else f'{f1:.2f}'):>11} "
          f"{pct(br.get('canonical', {}).get('accuracy')):>7} {pct(br.get('counterfactual', {}).get('accuracy')):>6} "
          f"{pct(br.get('counterfactual', {}).get('bias_ratio')):>7}")
PY
