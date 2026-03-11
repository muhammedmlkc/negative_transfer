#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUNLOG_DIR="$ROOT_DIR/runlogs"
mkdir -p "$RUNLOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
STATUS_FILE="${STATUS_FILE:-$RUNLOG_DIR/final_sdwpf_${STAMP}.status}"
LOG_FILE="${LOG_FILE:-$RUNLOG_DIR/final_sdwpf_${STAMP}.log}"

SEEDS=(${SEEDS:-42 43 44})
LOW_DATA_DAYS=(${LOW_DATA_DAYS:-14 30 60})
METHODS=(
  persistence ridge lightgbm
  fedformer safe_fedformer
  dlinear itransformer
  patchtst safe_patchtst
  tcn safe_tcn
)

COMMON_ARGS=(
  --dataset sdwpf
  --device cuda
  --max_sources 1
  --batch_size 64
  --num_workers 8
  --prefetch_factor 4
  --matmul_precision high
  --pretrain_epochs 16
  --finetune_epochs 14
  --safe_epochs 14
  --nf_max_steps 180
  --nf_val_check_steps 10
  --nf_early_stop_patience_steps 30
  --nf_batch_size 32
  --nf_windows_batch_size 256
  --nf_inference_windows_batch_size 512
)

echo "RUNNING" > "$STATUS_FILE"
echo "STARTED_AT=$(date --iso-8601=seconds)" >> "$STATUS_FILE"
echo "LOG_FILE=$LOG_FILE" >> "$STATUS_FILE"

run_step() {
  local name="$1"
  shift
  echo "STEP=$name" >> "$STATUS_FILE"
  echo "" | tee -a "$LOG_FILE"
  echo "===== $name =====" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
}

run_step "SDWPF_BENCHMARK" \
  python safe_tcn_lab/run_benchmark.py \
    "${COMMON_ARGS[@]}" \
    --methods "${METHODS[@]}" \
    --seeds "${SEEDS[@]}" \
    --baseline_method fedformer \
    --primary_method safe_fedformer \
    --benchmark_root safe_tcn_lab/final_runs/benchmark_sdwpf

run_step "SDWPF_SWEEP" \
  python safe_tcn_lab/run_transfer_sweep.py \
    "${COMMON_ARGS[@]}" \
    --methods "${METHODS[@]}" \
    --seeds "${SEEDS[@]}" \
    --baseline_method fedformer \
    --primary_method safe_fedformer \
    --target_train_days_list "${LOW_DATA_DAYS[@]}" \
    --sweep_root safe_tcn_lab/final_runs/sweep_sdwpf

echo "SUCCESS" > "$STATUS_FILE"
echo "FINISHED_AT=$(date --iso-8601=seconds)" >> "$STATUS_FILE"
echo "LOG_FILE=$LOG_FILE" >> "$STATUS_FILE"
