#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob globstar

CONFIG_NAME="${CONFIG_NAME:-experiment_T5_V5}"
VARIANTS="${VARIANTS:-Beat,Bar,BeatBar,BeatAttn,BarAttn,BeatBarAttn,Vanilla}"
RUNS_ROOT="${RUNS_ROOT:-runs/Transformer-T5}"
ACCEL="${ACCEL:-gpu}"
LOG_ROOT="${LOG_ROOT:-logs}"
OUT_SUFFIX="${OUT_SUFFIX:-$(date +%y%m%d-%H%M%S)}"
EXTRA="${EXTRA:-}"                       # extra Hydra overrides
BACKGROUND="${BACKGROUND:-false}"
VARIANT_GPUS="${VARIANT_GPUS:-}"   # e.g., 'Beat=5;Vanilla=6,7'

declare -A TAG
TAG[Beat]="BeatContext"
TAG[Bar]="BarContext"
TAG[BeatBar]="BeatBarContext"
TAG[BeatAttn]="BeatContextAttn"
TAG[BarAttn]="BarContextAttn"
TAG[BeatBarAttn]="BeatBarContextAttn"
TAG[Vanilla]="NoContext"

declare -A GPU_FOR
if [[ -n "$VARIANT_GPUS" ]]; then
  IFS=';' read -ra PAIRS <<< "$VARIANT_GPUS"
  for p in "${PAIRS[@]}"; do
    [[ -z "$p" ]] && continue
    k="${p%%=*}"; v="${p#*=}"
    GPU_FOR["$k"]="$v"
  done
fi

find_latest_ckpt () {
  local tag="$1"
  # newest run dir whose name ends with _${tag} (avoid matching ...${tag}Attn, etc.)
  local run_dir
  run_dir=$(ls -dt "${RUNS_ROOT}"/**/*_"${tag}" 2>/dev/null | head -n1 || true)
  [[ -z "$run_dir" ]] && return 1
  if [[ -f "${run_dir}/cpt/latest.ckpt" ]]; then
    echo "${run_dir}/cpt/latest.ckpt"
  else
    ls -t "${run_dir}/cpt"/steps_*.ckpt 2>/dev/null | head -n1 || true
  fi
}

mkdir -p "${LOG_ROOT}"

DEFAULT_CVD="${CUDA_VISIBLE_DEVICES-}"
IFS=',' read -r -a LIST <<< "$VARIANTS"
for V in "${LIST[@]}"; do
  [[ -z "${TAG[$V]:-}" ]] && { echo "Unknown variant: $V"; exit 1; }
  CKPT=$(find_latest_ckpt "${TAG[$V]}" || true)
  [[ -z "$CKPT" ]] && { echo "No checkpoint for ${TAG[$V]} under ${RUNS_ROOT}"; exit 1; }

  export CUDA_VISIBLE_DEVICES="${DEFAULT_CVD}"
  DEVCOUNT="${DEVICES-1}"
  if [[ -n "${GPU_FOR[$V]:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_FOR[$V]}"
    IFS=',' read -ra G <<< "${GPU_FOR[$V]}"; DEVCOUNT="${#G[@]}"
  elif [[ -n "${CUDA_VISIBLE_DEVICES-}" ]] ; then
    IFS=',' read -ra G <<< "${CUDA_VISIBLE_DEVICES}"; DEVCOUNT="${#G[@]}"
  fi

  TS="$(date +%y%m%d-%H%M%S)"
  GPU_TAG="${CUDA_VISIBLE_DEVICES//,/}"
  LOG_FILE="${LOG_ROOT}/eval_${V}_${TS}_g${GPU_TAG}.log"
  OUTDIR="$(dirname "$(dirname "$CKPT")")/eval/${TAG[$V]}_${OUT_SUFFIX}"

  echo ">>> Evaluating $V  GPUs=[${CUDA_VISIBLE_DEVICES-}] devices=${DEVCOUNT}  ckpt=${CKPT}  log=${LOG_FILE}"

  CMD=(python evaluate.py
    --config-name="${CONFIG_NAME}"
    hydra.job.chdir=false
    accelerator="${ACCEL}" devices="${DEVCOUNT}"
    model.checkpoint_path="$CKPT"
    training.log_dir="$OUTDIR"
  )
  [[ -n "$EXTRA" ]] && CMD+=( ${EXTRA} )


  if [[ "${BACKGROUND}" == "true" ]]; then
    nohup bash -lc "stdbuf -oL -eL ${CMD[*]}" >> "${LOG_FILE}" 2>&1 &
  else
    stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
  fi
done
