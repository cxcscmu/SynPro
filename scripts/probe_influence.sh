#!/bin/bash
#SBATCH --job-name=probe_influence
#SBATCH --output=runs/probe_influence_%j.out
#SBATCH --error=runs/probe_influence_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00

# print commands
set -x

# Probe data influence via validation-updated model (YAML-based, FSDP-compatible)
# This script uses the full train.py infrastructure with a custom config
# Usage: bash scripts/probe_influence.sh [CHECKPOINT_PATH]
set -euo pipefail

source .env

GCS_ROOT="gs://your-bucket/synpro"

# Get checkpoint path from argument or use default
if [ $# -eq 1 ]; then
  CHECKPOINT_PATH="$1"
  echo "Using checkpoint path from argument: ${CHECKPOINT_PATH}"
else
  CHECKPOINT_PATH="out/OLMo-400M/dclm_1.4B_all_repetition/step8624-unsharded"
  echo "No checkpoint path provided, using default: ${CHECKPOINT_PATH}"
fi

# Download checkpoint if needed
# export CHECKPOINT_DIR="${LOCAL_ROOT}/${CHECKPOINT_PATH}"
export CHECKPOINT_DIR="/tmp/${CHECKPOINT_PATH}"
mkdir -p "$(dirname "${CHECKPOINT_DIR}")"
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "Checkpoint directory missing, downloading..."
    gcloud storage cp -r "${GCS_ROOT}/${CHECKPOINT_PATH}" "$(dirname "${CHECKPOINT_DIR}")"
else
    echo "Checkpoint directory exists, skipping download"
fi

# Configuration
# CONFIG_PATH="configs/dclm/probe-influence-train_2x.yaml"
CONFIG_PATH="configs/dclm/probe-influence-train_1B.yaml"
export CHECKPOINT_NAME=$(basename $(dirname "${CHECKPOINT_DIR}"))/$(basename "${CHECKPOINT_DIR}")
export OUTPUT_DIR="${CHECKPOINT_DIR}/data_influence"

# Count visible GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "Detected ${NUM_GPUS} GPUs"

if [ "$NUM_GPUS" -eq 0 ]; then
  echo "ERROR: No GPUs detected"
  exit 1
fi

# Track files to clean up on exit
CLEANUP_FILES=()
cleanup() {
    for f in "${CLEANUP_FILES[@]}"; do
        rm -f "$f"
    done
}
trap cleanup EXIT

# Create a processed config with environment variables substituted
CONFIG_PROCESSED="${CONFIG_PATH}_$$.yaml"
envsubst < "${CONFIG_PATH}" > "${CONFIG_PROCESSED}"
CLEANUP_FILES+=("${CONFIG_PROCESSED}")

echo "=================================================="
echo "Data Influence Computation (FSDP-compatible)"
echo "=================================================="
echo "Config: ${CONFIG_PROCESSED}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "=================================================="

# Run training to get updated model
# if [ ! -f "${OUTPUT_DIR}/influence.npy" ]; then
torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=$((RANDOM + 20000)) \
    scripts/train.py "${CONFIG_PROCESSED}"
# fi

# CONFIG_PATH="configs/dclm/probe-influence-eval_2x.yaml"
CONFIG_PATH="configs/dclm/probe-influence-eval_1B.yaml"

CONFIG_PROCESSED="${CONFIG_PATH}_$$.yaml"
envsubst < "${CONFIG_PATH}" > "${CONFIG_PROCESSED}"
CLEANUP_FILES+=("${CONFIG_PROCESSED}")

# Run influence computation
# if [ ! -f "${OUTPUT_DIR}/influence.npy" ]; then
torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=$((RANDOM + 20001)) \
    scripts/probe_influence.py \
    "${CONFIG_PROCESSED}"
# fi

python scripts/select_data.py \
    ${CONFIG_PROCESSED} \
    --metrics-file ${OUTPUT_DIR}/influence.npy \
    --sample-ratio -1 \
    --count-positive-by-path
    # --output ${LOCAL_ROOT}/data/preprocessed/dclm/${CHECKPOINT_NAME}/selection/train_ids_olmo_gumbel.npy \
    # --gumbel
    # ${DATA_PATH:+--replay-data-path "$DATA_PATH"}

CONFIG_PATH="configs/dclm/probe-influence-eval_1B_2.yaml"

CONFIG_PROCESSED="${CONFIG_PATH}_$$.yaml"
envsubst < "${CONFIG_PATH}" > "${CONFIG_PROCESSED}"
CLEANUP_FILES+=("${CONFIG_PROCESSED}")

# Run influence computation
# if [ ! -f "${OUTPUT_DIR}/influence.npy" ]; then
torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=$((RANDOM + 20001)) \
    scripts/probe_influence.py \
    "${CONFIG_PROCESSED}"
# fi

python scripts/select_data.py \
    ${CONFIG_PROCESSED} \
    --metrics-file ${OUTPUT_DIR}/influence.npy \
    --sample-ratio -1 \
    --count-positive-by-path

echo "=================================================="
echo "Selected data indices saved to: ${LOCAL_ROOT}/data/preprocessed/dclm/${CHECKPOINT_NAME}/selection/train_ids_olmo_gumbel.npy"
echo "=================================================="