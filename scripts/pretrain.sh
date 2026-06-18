#!/bin/bash
#SBATCH --job-name=dclm_pretrain
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --output=runs/dclm_pretrain_%j.out
#SBATCH --error=runs/dclm_pretrain_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=208
#SBATCH --mem=1792G
#SBATCH --time=2-00:00:00

# print commands
set -x

source .env

GCS_ROOT="gs://your-bucket/synpro"
CONFIG_PATH="${1:-configs/dclm/OLMo-400M.yaml}"

# Setup checkpoint sync function
sync_checkpoints() {
  echo "Syncing checkpoints to GCS..."
  gsutil -m rsync -r "/tmp/out" "${GCS_ROOT}/out"
  echo "Checkpoint sync complete"
}

# Background sync loop - runs every 2 hours
start_background_sync() {
  while true; do
    sleep 7200  # 2 hours
    sync_checkpoints
  done &
  SYNC_PID=$!
  echo "Started background sync process (PID: $SYNC_PID)"
}

# Preemption handler - SIGTERM is sent ~30s before preemption
handle_preemption() {
  echo "SIGTERM received - job being preempted!"
  echo "Forcing final checkpoint sync..."
  kill $SYNC_PID 2>/dev/null || true
  sync_checkpoints
  echo "Emergency sync complete, exiting..."
  exit 143  # 128 + 15 (SIGTERM)
}

trap handle_preemption SIGTERM

mkdir -p "${LOCAL_ROOT}/data/preprocessed/"
echo "LOCAL_ROOT=${LOCAL_ROOT}"
if [[ ! -d "${LOCAL_ROOT}/data/preprocessed/dclm" ]]; then
    echo "Directory missing, downloading..."
    gcloud storage cp -r "${GCS_ROOT}/data/preprocessed/dclm" "${LOCAL_ROOT}/data/preprocessed/"
else
    echo "Directory exists, skipping download"
fi

# Start background sync
# start_background_sync

# Count visible GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "Detected ${NUM_GPUS} GPUs"

if [ "$NUM_GPUS" -eq 0 ]; then
  echo "ERROR: No GPUs detected"
  exit 1
fi

# Create a processed config with environment variables substituted
CONFIG_PROCESSED="${CONFIG_PATH}_$$.yaml"
envsubst < "${CONFIG_PATH}" > "${CONFIG_PROCESSED}"

# Ensure cleanup on exit (success or failure) and final sync
# trap "sync_checkpoints; kill $SYNC_PID 2>/dev/null || true; rm -f '${CONFIG_PROCESSED}'" EXIT

# mkdir -p /tmp/out/OLMo-1.1B/dclm_46B_unique
# gcloud storage cp -r ${GCS_ROOT}/out/OLMo-1.1B/dclm_46B_unique/step17472-unsharded /tmp/out/OLMo-1.1B/dclm_46B_unique
# gcloud storage cp -r ${GCS_ROOT}/out/OLMo-1.1B/dclm_2.3B_organic+recycled+newrecycled_repetition/step37128-unsharded  /tmp/out/OLMo-1.1B/dclm_2.3B_organic+recycled+newrecycled_repetition

torchrun \
  --nproc_per_node="$NUM_GPUS" \
  scripts/train.py "${CONFIG_PROCESSED}"

# Final sync on successful completion
# sync_checkpoints