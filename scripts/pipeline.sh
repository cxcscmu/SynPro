#!/usr/bin/env bash
# Reformat Influence Pipeline
# Usage: bash scripts/dclm/pipeline_reformat_influence.sh [CHECKPOINT_GCS_PATH]
# Example: bash scripts/dclm/pipeline_reformat_influence.sh \
#   out/OLMo-400M/dclm_0.8B_organic+recycled+reformat_repetition/step10920-unsharded
#
# Steps:
#   1. Probe influence training (all GPUs) → saves after-checkpoint
#   2. Launch influence HTTP server (GPU 0) + reformat faithfulness server (GPU 1)
#   3. GRPO reformat generator training (GPUs 2-7, 240 steps, resume from checkpoint-240)
#   4. reformat inference data-parallel vLLM (all GPUs)
#   5. Tokenize reformat JSONL → .npy
#   6. Continue pretraining from CHECKPOINT_PATH
#
# Each step is skipped if its output already exists (safe to resume after failure).

set -euo pipefail
set -x

# ============================================================
# Configuration
# ============================================================
CHECKPOINT_PATH="${1:-out/OLMo-400M/dclm_0.8B_organic+recycled+reformat_repetition/step21840-unsharded}"
ONLY_STEP="${2:-}"  # Optional: run only this step number (1-6). Empty = run all.
GCS_ROOT="gs://your-bucket/synpro"

OLMO_ROOT="$SYNPRO_ROOT"
OPEN_R1_ROOT="$SYNPRO_ROOT"

BEFORE_CKPT="/tmp/${CHECKPOINT_PATH}"
export CHECKPOINT_DIR="${BEFORE_CKPT}"
export CHECKPOINT_NAME="$(basename "$(dirname "${BEFORE_CKPT}")")/$(basename "${BEFORE_CKPT}")"
export OUTPUT_DIR="${BEFORE_CKPT}/data_influence"

GRPO_STEPS=240
# GRPO_RESUME_CKPT="/tmp/synthetic_data_generator_OLMo2-1.1B-reformat-faithfulness_step2000/checkpoint-240"
CKPT_STEP_TAG="$(basename "${CHECKPOINT_PATH}")"
CKPT_STEP_TAG="${CKPT_STEP_TAG%-unsharded}"
GRPO_OUTPUT_DIR="/tmp/${CKPT_STEP_TAG}_synthetic_data_generator_OLMo2-1.1B-reformat-influence+faithfulness"
# GRPO_OUTPUT_DIR="/tmp/${CKPT_STEP_TAG}_synthetic_data_generator_OLMo2-1.1B-reformat-influence+faithfulness_from240"
INFER_CHECKPOINT="${GRPO_OUTPUT_DIR}/checkpoint-${GRPO_STEPS}"
export OUTPUT_DIR_NAME="${CKPT_STEP_TAG}_OLMo2-1.1B-reformat-faithful-grpo-${GRPO_STEPS}"
# export OUTPUT_DIR_NAME="${CKPT_STEP_TAG}_OLMo2-1.1B-reformat-faithful-grpo-${GRPO_STEPS}_from240"
INFER_CONFIG="${OLMO_ROOT}/configs/dclm/OLMo-400M_2x_0.8B.yaml"
PRETRAIN_CONFIG="${OLMO_ROOT}/configs/pretrain/OLMo-400M.yaml"

REFORMAT_FAITHFULNESS_MODEL="/tmp/out/Qwen3-1.7B-judge-sft/checkpoint-180"

INFLUENCE_SERVER_PID=""
FAITHFULNESS_SERVER_PID=""
DATAMAN_SERVER_PID=""

cleanup() {
    [[ -n "${INFLUENCE_SERVER_PID}" ]] && kill "${INFLUENCE_SERVER_PID}" 2>/dev/null || true
    [[ -n "${FAITHFULNESS_SERVER_PID}" ]] && kill "${FAITHFULNESS_SERVER_PID}" 2>/dev/null || true
    [[ -n "${DATAMAN_SERVER_PID}" ]] && kill "${DATAMAN_SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# ============================================================
# 0. Setup  (OLMo env)
# ============================================================
mkdir -p /tmp/logs
cd "${OLMO_ROOT}"
source .env

if [[ ! -d "${BEFORE_CKPT}" ]]; then
    echo "[Pipeline] Downloading checkpoint from GCS..."
    mkdir -p "$(dirname "${BEFORE_CKPT}")"
    gcloud storage cp -r "${GCS_ROOT}/${CHECKPOINT_PATH}" "$(dirname "${BEFORE_CKPT}")"
fi

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "[Pipeline] Detected ${NUM_GPUS} GPUs"

# ============================================================
# Step 1: Probe Influence Training  (OLMo env, all GPUs)
# ============================================================
AFTER_CKPT=""; [[ -d "${OUTPUT_DIR}" ]] && AFTER_CKPT=$(find "${OUTPUT_DIR}" -maxdepth 1 -name 'step*-unsharded' -type d 2>/dev/null | sort -V | tail -1)
if [[ -n "${ONLY_STEP}" && "${ONLY_STEP}" != "1" ]]; then
    echo "[Pipeline] Step 1: SKIPPED (ONLY_STEP=${ONLY_STEP})"
elif [[ -n "${AFTER_CKPT}" ]]; then
    echo "[Pipeline] Step 1: SKIPPED — after-checkpoint exists: ${AFTER_CKPT}"
else
    echo "[Pipeline] Step 1: Probe influence training (all ${NUM_GPUS} GPUs)..."

    PROBE_CONFIG="configs/probe/probe-influence-train.yaml"
    if [[ "${CHECKPOINT_PATH}" =~ OLMo-1.1B ]]; then
        PROBE_CONFIG="configs/probe/probe-influence-train.yaml"
    fi
    echo "[Pipeline] Using probe config: ${PROBE_CONFIG}"

    PROBE_CONFIG_TMP="${PROBE_CONFIG}_$$.yaml"
    envsubst < "${PROBE_CONFIG}" > "${PROBE_CONFIG_TMP}"

    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port=$((RANDOM + 20000)) \
        scripts/train.py "${PROBE_CONFIG_TMP}" \
        --eval_on_load=false

    rm -f "${PROBE_CONFIG_TMP}"

    AFTER_CKPT=""; [[ -d "${OUTPUT_DIR}" ]] && AFTER_CKPT=$(find "${OUTPUT_DIR}" -maxdepth 1 -name 'step*-unsharded' -type d 2>/dev/null | sort -V | tail -1)
    if [[ -z "${AFTER_CKPT}" ]]; then
        echo "ERROR: No unsharded checkpoint found in ${OUTPUT_DIR}"
        exit 1
    fi
fi
echo "[Pipeline] After-checkpoint: ${AFTER_CKPT}"

# ============================================================
# Step 2: Launch Influence Server (GPU 0) + Faithfulness Server (GPU 1)
# ============================================================
if [[ -n "${ONLY_STEP}" && "${ONLY_STEP}" != "2" && "${ONLY_STEP}" != "3" ]]; then
    echo "[Pipeline] Steps 2-3: SKIPPED (ONLY_STEP=${ONLY_STEP})"
elif [[ -d "${INFER_CHECKPOINT}" ]]; then
    echo "[Pipeline] Steps 2-3: SKIPPED — GRPO checkpoint exists: ${INFER_CHECKPOINT}"
else
    echo "[Pipeline] Step 2: Launching influence server on GPU 0..."
    cd "${OLMO_ROOT}"
    source .env
    export PYTHONPATH="${OPEN_R1_ROOT}/src:${OLMO_ROOT}:${PYTHONPATH:-}"

    CUDA_VISIBLE_DEVICES=0 python -m generator.infer.get_influence \
        --before-checkpoint "${BEFORE_CKPT}" \
        --after-checkpoint  "${AFTER_CKPT}" \
        --host 0.0.0.0 \
        --port 24775 \
        --batch-size 8 \
        --max-seq-len 2048 \
        > "/tmp/logs/influence_server.log" 2>&1 &
    INFLUENCE_SERVER_PID=$!

    echo "[Pipeline] Launching reformat faithfulness server on GPU 1..."
    cd "${OPEN_R1_ROOT}"
    source .env
    export PYTHONPATH="${OPEN_R1_ROOT}/src:${OLMO_ROOT}:${PYTHONPATH:-}"

    CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
        --model "${REFORMAT_FAITHFULNESS_MODEL}" \
        --port 8002 \
        --gpu-memory-utilization 0.8 \
        --max-model-len 4096 \
        --dtype bfloat16 \
        > "/tmp/logs/reformat_faithfulness_server.log" 2>&1 &
    FAITHFULNESS_SERVER_PID=$!

    echo "[Pipeline] Waiting for influence server (up to 120s)..."
    for i in $(seq 120); do
        if curl -sf http://127.0.0.1:24775/health > /dev/null 2>&1; then
            echo "[Pipeline] Influence server ready after ${i}s"
            break
        fi
        sleep 1
        if [[ $i -eq 120 ]]; then
            echo "ERROR: Influence server did not start within 120s"
            cat /tmp/logs/influence_server.log
            exit 1
        fi
    done

    echo "[Pipeline] Waiting for reformat faithfulness server (up to 300s)..."
    for i in $(seq 60); do
        if curl -sf http://127.0.0.1:8002/health > /dev/null 2>&1; then
            echo "[Pipeline] Faithfulness server ready after $((i*5))s"
            break
        fi
        sleep 5
        if [[ $i -eq 60 ]]; then
            echo "ERROR: reformat faithfulness server did not start within 300s"
            cat /tmp/logs/reformat_faithfulness_server.log
            exit 1
        fi
    done

    echo "[Pipeline] Launching DataMan vLLM server on GPU 2, port 8000..."
    cd "${OPEN_R1_ROOT}"
    CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
        --model "RuPeng/DataMan-1.5B-EN" \
        --port 8000 \
        --gpu-memory-utilization 0.8 \
        --max-model-len 4096 \
        --dtype bfloat16 \
        > "/tmp/logs/dataman_server.log" 2>&1 &
    DATAMAN_SERVER_PID=$!

    echo "[Pipeline] Waiting for DataMan server (up to 300s)..."
    for i in $(seq 60); do
        if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
            echo "[Pipeline] DataMan server ready after $((i*5))s"
            break
        fi
        sleep 5
        if [[ $i -eq 60 ]]; then
            echo "ERROR: DataMan server did not start within 300s"
            cat /tmp/logs/dataman_server.log
            exit 1
        fi
    done

    # ============================================================
    # Step 3: GRPO reformat Training  (open-r1 env, GPUs 3-7)
    # ============================================================
    echo "[Pipeline] Step 3: GRPO reformat training for ${GRPO_STEPS} steps (GPUs 3-7)..."
    cd "${OPEN_R1_ROOT}"
    source .env

    CUDA_VISIBLE_DEVICES=3,4,5,6,7 \
    DATA_INFLUENCE_ENDPOINT="http://127.0.0.1:24775/get_influence" \
    REFORMAT_FAITHFULNESS_MODEL="${REFORMAT_FAITHFULNESS_MODEL}" \
    REFORMAT_FAITHFULNESS_SERVER_URL="http://127.0.0.1:8002/v1" \
    PYTHONPATH="${OPEN_R1_ROOT}/src:${OLMO_ROOT}:${PYTHONPATH:-}" \
    ACCELERATE_LOG_LEVEL=info \
        accelerate launch \
            --config_file recipes/accelerate_configs/zero3_5gpu.yaml \
            generator/grpo_reformat.py \
            --config recipes/OLMo2/grpo/configs/generator/reformat.yaml \
            --max_steps "${GRPO_STEPS}" \
            --output_dir "${GRPO_OUTPUT_DIR}"
            # --model_name_or_path "${GRPO_RESUME_CKPT}"

    echo "[Pipeline] Stopping servers..."
    kill "${INFLUENCE_SERVER_PID}" || true
    kill "${FAITHFULNESS_SERVER_PID}" || true
    kill "${DATAMAN_SERVER_PID}" || true
    wait "${INFLUENCE_SERVER_PID}" 2>/dev/null || true
    wait "${FAITHFULNESS_SERVER_PID}" 2>/dev/null || true
    wait "${DATAMAN_SERVER_PID}" 2>/dev/null || true
    INFLUENCE_SERVER_PID=""
    FAITHFULNESS_SERVER_PID=""
    DATAMAN_SERVER_PID=""
fi

# ============================================================
# Step 4: Reformat Inference  (open-r1 env, all GPUs)
# ============================================================
FIRST_NPY=$(grep 'OUTPUT_DIR_NAME' "${PRETRAIN_CONFIG}" | grep '\.npy' | head -1 | sed 's/.*- //')
FIRST_NPY="${FIRST_NPY/\$\{LOCAL_ROOT\}/$LOCAL_ROOT}"
FIRST_NPY="${FIRST_NPY/\$\{OUTPUT_DIR_NAME\}/$OUTPUT_DIR_NAME}"
if [[ -n "${ONLY_STEP}" && "${ONLY_STEP}" != "4" && "${ONLY_STEP}" != "5" ]]; then
    echo "[Pipeline] Steps 4-5: SKIPPED (ONLY_STEP=${ONLY_STEP})"
elif [[ -f "${FIRST_NPY}" ]]; then
    echo "[Pipeline] Step 4-5: SKIPPED — tokenized output exists: ${FIRST_NPY}"
else
    echo "[Pipeline] Step 4: reformat inference with ${INFER_CHECKPOINT}..."
    cd "${OPEN_R1_ROOT}"
    source .env
    cd "${OLMO_ROOT}"

    if [[ ! -d "${INFER_CHECKPOINT}" ]]; then
        echo "ERROR: Inference checkpoint not found: ${INFER_CHECKPOINT}"
        exit 1
    fi

    VLLM_USE_V1=1 \
    TOTAL_GPUS="${NUM_GPUS}" \
    GPUS_PER_DP_RANK=1 \
    MODEL_PATH="${INFER_CHECKPOINT}" \
    OUTPUT_DIR_NAME="${OUTPUT_DIR_NAME}" \
        python scripts/run_infer_dp.py "${INFER_CONFIG}"

    # ============================================================
    # Step 5: Tokenize reformat JSONL -> .npy  (OLMo env, CPU)
    # ============================================================
    echo "[Pipeline] Step 5: Tokenizing reformat JSONL to .npy..."
    cd "${OLMO_ROOT}"
    source .env

    grep 'OUTPUT_DIR_NAME' "${PRETRAIN_CONFIG}" | grep '\.npy' | sed 's/.*- //' | while read -r raw_path; do
        npy="${raw_path/\$\{LOCAL_ROOT\}/$LOCAL_ROOT}"
        npy="${npy/\$\{OUTPUT_DIR_NAME\}/$OUTPUT_DIR_NAME}"
        jsonl="${npy%.npy}.jsonl"
        echo "[Tokenize] ${jsonl##*/} -> ${npy##*/}"
        python scripts/convert_text_jsonl.py \
            --reverse \
            --config "${INFER_CONFIG}" \
            --input-jsonl "${jsonl}" \
            --output-npy  "${npy}"
    done
fi

# ============================================================
# Step 6: Continue Pretraining from CHECKPOINT_PATH
# ============================================================
if [[ -n "${ONLY_STEP}" && "${ONLY_STEP}" != "6" ]]; then
    echo "[Pipeline] Step 6: SKIPPED (ONLY_STEP=${ONLY_STEP})"
else
    echo "[Pipeline] Step 6: Pretraining (load from ${BEFORE_CKPT})..."
    cd "${OLMO_ROOT}"
    source .env

    export LOAD_PATH="${BEFORE_CKPT}"
    export OUTPUT_DIR_NAME="${OUTPUT_DIR_NAME}"

    bash scripts/dclm/pretrain_new.sh "${PRETRAIN_CONFIG}"
fi

echo "=================================================="
echo "[Pipeline] Complete!"
echo "  GRPO model   : ${GRPO_OUTPUT_DIR}/checkpoint-${GRPO_STEPS}"
echo "  reformat JSONL/NPY : .../${OUTPUT_DIR_NAME}/..."
echo "  Pretrain     : ${PRETRAIN_CONFIG} (loaded from ${BEFORE_CKPT})"
echo "=================================================="
