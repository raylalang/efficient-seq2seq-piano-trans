#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME="${CONFIG_NAME:-experiment_T5_V5}"
VARIANTS="${VARIANTS:-Beat,Bar,BeatBar,BeatAttn,BarAttn,BeatBarAttn,Vanilla}"
ACCEL="${ACCEL:-gpu}"
STEPS="${STEPS:-100000}"
NOTES_SUFFIX="${NOTES_SUFFIX:-}"
EXTRA="${EXTRA:-}"                       # extra Hydra overrides
LOG_ROOT="${LOG_ROOT:-logs}"             # where stdout/stderr logs go
BACKGROUND="${BACKGROUND:-false}"        # true -> run each variant in background
VARIANT_GPUS="${VARIANT_GPUS:-}"         # e.g. 'Beat=0;Bar=1;BeatBarConcat=2;BeatBarAttn=3;Vanilla=4,5'

declare -A TAG
TAG[Beat]="BeatContext"
TAG[Bar]="BarContext"
TAG[BeatBar]="BeatBarContext"
TAG[BeatAttn]="BeatContextAttn"
TAG[BarAttn]="BarContextAttn"
TAG[BeatBarAttn]="BeatBarContextAttn"
TAG[Vanilla]="NoContext"

# parse VARIANT_GPUS into an associative array
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

DEFAULT_CVD="${CUDA_VISIBLE_DEVICES-}"
IFS=',' read -r -a LIST <<< "$VARIANTS"
for V in "${LIST[@]}"; do
  [[ -z "${TAG[$V]:-}" ]] && { echo "Unknown variant: $V"; exit 1; }

  # set CUDA_VISIBLE_DEVICES and devices count per variant
  export CUDA_VISIBLE_DEVICES="${DEFAULT_CVD}"
  DEVCOUNT="${DEVICES-1}"
  if [[ -n "${GPU_FOR[$V]:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_FOR[$V]}"
    IFS=',' read -ra G <<< "${GPU_FOR[$V]}"; DEVCOUNT="${#G[@]}"
  elif [[ -n "${CUDA_VISIBLE_DEVICES-}" ]]; then
    IFS=',' read -ra G <<< "${CUDA_VISIBLE_DEVICES}"; DEVCOUNT="${#G[@]}"
  fi

  TS="$(date +%y%m%d-%H%M%S)"
  GPU_TAG="${CUDA_VISIBLE_DEVICES//,/}"          # e.g., "013" for 0,1,3
  NOTE="${TAG[$V]}"; [[ -n "$NOTES_SUFFIX" ]] && NOTE="${NOTE}_${NOTES_SUFFIX}"
  LOG_FILE="${LOG_ROOT}/train_${V}_${TS}_g${GPU_TAG}.log"

  echo ">>> Training $V  GPUs=[${CUDA_VISIBLE_DEVICES-}] devices=${DEVCOUNT}  log=${LOG_FILE}"

  CMD=(python train.py
    --config-name="${CONFIG_NAME}"
    hydra.job.chdir=false
    accelerator="${ACCEL}" devices="${DEVCOUNT}"
    variant="${V}"
    training.training_steps="${STEPS}"
    training.notes="${TAG[$V]}${NOTES_SUFFIX:+_${NOTES_SUFFIX}}"
  )
  [[ -n "$EXTRA" ]] && CMD+=( ${EXTRA} )

  if [[ "${BACKGROUND}" == "true" ]]; then
    nohup bash -lc "stdbuf -oL -eL ${CMD[*]}" >> "${LOG_FILE}" 2>&1 &
  else
    stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
  fi
done
