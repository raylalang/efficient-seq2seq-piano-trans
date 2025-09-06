#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob globstar

# ===== controls =====
VARIANTS="${VARIANTS:-Beat,Bar,BeatBar,BeatAttn,BarAttn,BeatBarAttn,Vanilla}"
RUNS_ROOT="${RUNS_ROOT:-runs/Transformer-T5}"
ACCEL="${ACCEL:-gpu}"
STEPS="${STEPS:-100000}"               # new target total steps
EXTRA="${EXTRA:-}"                     # extra Hydra overrides
LOG_ROOT="${LOG_ROOT:-logs}"
BACKGROUND="${BACKGROUND:-false}"
VARIANT_GPUS="${VARIANT_GPUS:-}"       # e.g. 'Beat=0;Bar=1;BeatBar=2;BeatBarAttn=3;Vanilla=4,5'
ALLOW_LEGACY="${ALLOW_LEGACY:-true}"

# variant -> tag (folder suffix)
declare -A TAG
TAG[Beat]="BeatContext"
TAG[Bar]="BarContext"
TAG[BeatBar]="BeatBarContext"
TAG[BeatAttn]="BeatContextAttn"
TAG[BarAttn]="BarContextAttn"
TAG[BeatBarAttn]="BeatBarContextAttn"
TAG[Vanilla]="NoContext"

# parse GPU map
declare -A GPU_FOR
if [[ -n "$VARIANT_GPUS" ]]; then
  IFS=';' read -ra PAIRS <<< "$VARIANT_GPUS"
  for p in "${PAIRS[@]}"; do
    [[ -z "$p" ]] && continue
    k="${p%%=*}"; v="${p#*=}"
    GPU_FOR["$k"]="$v"
  done
fi

mkdir -p "${LOG_ROOT}"

find_latest_run () {
  local tag="$1"
  ls -d "${RUNS_ROOT}"/**/*_"${tag}" 2>/dev/null | sort -r | head -n1 || true
}
find_ckpt () {
  local run_dir="$1"
  if [[ -f "${run_dir}/cpt/latest.ckpt" ]]; then
    echo "${run_dir}/cpt/latest.ckpt"
  else
    ls -t "${run_dir}/cpt"/steps_*.ckpt 2>/dev/null | head -n1 || true
  fi
}

# extract numeric step from ckpt path; fallback to reading latest_epoch.txt; else "latest"
ckpt_step () {
  local ckpt="$1"
  local base
  base="$(basename "$ckpt")"
  if [[ "$base" =~ ^steps_([0-9]+)\.ckpt$ ]]; then
    echo "${BASH_REMATCH[1]}"; return
  fi
  local dir
  dir="$(dirname "$ckpt")"
  if [[ -f "${dir}/latest_epoch.txt" ]]; then
    local s
    s="$(awk 'match($0,/global step=([0-9]+)/,m){print m[1]}' "${dir}/latest_epoch.txt" | tail -n1)"
    [[ -n "$s" ]] && { echo "$s"; return; }
  fi
  echo "latest"
}

DEFAULT_CVD="${CUDA_VISIBLE_DEVICES-}"
IFS=',' read -r -a LIST <<< "$VARIANTS"

for V in "${LIST[@]}"; do
  [[ -z "${TAG[$V]:-}" ]] && { echo "Unknown variant: $V"; exit 1; }

  RUN_DIR="$(find_latest_run "${TAG[$V]}")"
  [[ -z "$RUN_DIR" ]] && { echo "No run folder matching *_${TAG[$V]} under ${RUNS_ROOT}"; exit 1; }
  [[ ! -f "${RUN_DIR}/experiment_config.yaml" ]] && { echo "Missing ${RUN_DIR}/experiment_config.yaml"; exit 1; }

  CKPT="$(find_ckpt "${RUN_DIR}")"
  [[ -z "$CKPT" ]] && { echo "No checkpoint found under ${RUN_DIR}/cpt"; exit 1; }

  # Decide resume mode
  if strings "$CKPT" 2>/dev/null | grep -q "pytorch-lightning_version"; then
    RESUME_MODE="lightning"
  else
    if [[ "$ALLOW_LEGACY" == "true" ]]; then
      echo "WARN: $CKPT is legacy (torch.save). Falling back to warm-start."
      RESUME_MODE="legacy"
    else
      echo "ERROR: $CKPT is not a Lightning checkpoint and ALLOW_LEGACY=false."; exit 1
    fi
  fi

  # GPUs
  export CUDA_VISIBLE_DEVICES="${DEFAULT_CVD}"
  DEVCOUNT="${DEVICES-1}"
  if [[ -n "${GPU_FOR[$V]:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_FOR[$V]}"
    IFS=',' read -ra G <<< "${GPU_FOR[$V]}"; DEVCOUNT="${#G[@]}"
  elif [[ -n "${CUDA_VISIBLE_DEVICES-}" ]]; then
    IFS=',' read -ra G <<< "${CUDA_VISIBLE_DEVICES}"; DEVCOUNT="${#G[@]}"
  fi

  TS="$(date +%y%m%d-%H%M%S)"
  GPU_TAG="${CUDA_VISIBLE_DEVICES//,/}"
  STEP="$(ckpt_step "$CKPT")"
  LOG_FILE="${LOG_ROOT}/train_ckpt_${TAG[$V]}_${STEP}_${TS}_g${GPU_TAG}.log"

  echo ">>> Resuming $V  GPUs=[${CUDA_VISIBLE_DEVICES-}] devices=${DEVCOUNT}"
  echo "    run=${RUN_DIR}"
  echo "    ckpt=${CKPT}"
  echo "    step=${STEP}"
  echo "    log=${LOG_FILE}"

  # Base command
  CMD=(python train.py
    --config-dir "${RUN_DIR}"
    --config-name "experiment_config"
    hydra.job.chdir=false
    accelerator="${ACCEL}" devices="${DEVCOUNT}"
    training.log_dir="${RUN_DIR}"
    training.training_steps="${STEPS}"
  )

  # Append resume mode
  if [[ "$RESUME_MODE" == "lightning" ]]; then
    CMD+=( +training.resume_from_ckpt="${CKPT}" )
  else
    CMD+=( model.checkpoint_path="${CKPT}" +training.resume_from_ckpt=null )
  fi

  # Optional extra overrides
  [[ -n "$EXTRA" ]] && CMD+=( ${EXTRA} )

  if [[ "${BACKGROUND}" == "true" ]]; then
    nohup bash -lc "stdbuf -oL -eL ${CMD[*]}" >> "${LOG_FILE}" 2>&1 &
    echo "    PID $!"
  else
    stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
  fi
done
