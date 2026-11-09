# Shared helpers for the stage scripts (source this file).
step() { echo -e "\n==================== $(date '+%F %T')  $*"; }
gpu_free_mib()  { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_total_mib() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1; }

# Processes holding GPU memory (pid, name, MiB): shows who blocks a wait, e.g. a leftover vLLM engine.
gpu_holders() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/    /'
}

# Block until at least $1 GiB of GPU memory is free (other jobs share this GPU). Returns 1 after
# GPU_WAIT_MAX_H hours (default 12): scripts with `set -e` stop, the queue fails just that job.
wait_for_gpu() {
  local need=$(( $1 * 1024 )) waited=0 max=$(( ${GPU_WAIT_MAX_H:-12} * 60 ))
  while (( $(gpu_free_mib) < need )); do
    if (( waited >= max )); then
      echo "GPU never had $1 GiB free within ${GPU_WAIT_MAX_H:-12} h. Holders:"; gpu_holders; return 1
    fi
    if (( waited % 10 == 0 )); then
      echo "$(date +%T) waiting for GPU: $(gpu_free_mib) MiB free, need $1 GiB. Holders:"; gpu_holders
    fi
    sleep 60; waited=$(( waited + 1 ))
  done
}

# vLLM takes a fixed share of GPU memory: use what is free now (at most 85%), unless EVAL_GPU_UTIL is set.
eval_gpu_util() {
  [[ -n "${EVAL_GPU_UTIL:-}" ]] && { echo "$EVAL_GPU_UTIL"; return; }
  python3 -c "print(f'{min(0.85, $(gpu_free_mib) / $(gpu_total_mib) - 0.05):.2f}')"
}

# Log everything to <data>/reports/logs/<name>_<timestamp>.log as well as the terminal.
start_log() {
  mkdir -p "$1/reports/logs"
  LOG="$1/reports/logs/$2_$(date +%Y%m%d_%H%M%S).log"
  exec > >(tee -a "$LOG") 2>&1
}
