# SynPro: Model-Aware Synthetic Data Generation for Data-Bound LLM Pretraining

[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/pdf/2605.17849)

> LLM pretraining is shifting from a compute-bound to a data-bound regime, as the supply of high-quality human (organic) text can no longer match scaling demands. SynPro is a model-aware synthetic data generation framework that continuously supplies effective data in the data-bound scaling regime.

SynPro generates synthetic data optimized via reinforcement learning with **quality**, **faithfulness**, and **data influence** rewards. The first two ensure coherent, source-grounded outputs, while the last steers synthesis toward content the pretraining model has yet to absorb, and is refreshed when current pretraining plateaus.

## Architecture

| Component | Model | Details |
|-|-|-|
| Downstream (400M) | OLMo-400M | 16L, d=1024, 16H, SwiGLU, RoPE, RMSNorm |
| Downstream (1.1B) | OLMo-1.1B | 16L, d=2048, 16H, SwiGLU, RoPE, RMSNorm |
| Generator | OLMo-2-1B-Instruct | GRPO fine-tuned with multiple rewards |
| Quality Scorer | DataMan-1.5B-EN | Text quality evaluation |
| Faithfulness Verifier | Qwen3-1.7B (SFT) | Source-grounded verification |

## Pipeline Overview

```
[Organic Data] --> [1. Probe Influence] --> [Influence Scores]
                                                |
[Organic Data] --> [2. Generator Training] <----+
                        |        |
                   [Rephrase] [Reformat]
                        |        |
                   [3. Inference (vLLM)] --> [Synthetic JSONL]
                                                |
                                           [4. Tokenize] --> [Synthetic NPY]
                                                |
[Organic NPY] + [Synthetic NPY] --> [5. Continue Pretraining] --> [Improved Model]
                                                |
                                    (iterate: re-probe when plateaus)
```

## Requirements

See `requirements.txt` for the full list. Key dependencies:

- PyTorch 2.1+
- vLLM (inference and reward servers)
- HuggingFace transformers, trl, accelerate

## Setup

```bash
# Clone and install
git clone <repo_url> SynPro
cd SynPro
pip install -e .

# Set environment variables
export SYNPRO_ROOT=$(pwd)
export DATA_ROOT=/path/to/your/data
export LOCAL_ROOT=$DATA_ROOT  # Used in config files
```

## Step 1: Probe Influence Training

Fine-tunes a lightweight probe on the reference task (data available [here](https://drive.google.com/drive/folders/1_hxBogH7oPJMycIMMddoKG2r7yefmXeH?usp=sharing)) from a frozen checkpoint, then computes per-instance influence scores (loss_before - loss_after) on the pretraining data.

```bash
# Set checkpoint to probe
export CHECKPOINT_DIR=/path/to/checkpoint-unsharded
export CHECKPOINT_NAME=$(basename $CHECKPOINT_DIR)
export OUTPUT_DIR=${CHECKPOINT_DIR}/data_influence

# Train probe (all GPUs, ~3 min)
envsubst < configs/probe/probe-influence-train.yaml > /tmp/probe_train.yaml
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) \
    scripts/train.py /tmp/probe_train.yaml

# Compute influence scores on pretraining data (all GPUs)
envsubst < configs/probe/probe-influence-eval.yaml > /tmp/probe_eval.yaml
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) \
    scripts/probe_influence.py /tmp/probe_eval.yaml

# Output: ${OUTPUT_DIR}/influence.npy
```

### Data Selection (Optional)

Select top-K data by influence score for selective pretraining:

```bash
python scripts/select_data.py /tmp/probe_eval.yaml \
    --metrics-file ${OUTPUT_DIR}/influence.npy \
    --sample-ratio 0.5 \
    --output /path/to/selected_data.npy
```

## Step 2a: Rephrase Generator Training

Trains a generator to rephrase text, optimized with influence + quality + similarity rewards.

```bash
# Launch reward servers
# GPU 0: Influence server
CUDA_VISIBLE_DEVICES=0 python -m generator.infer.get_influence \
    --before-checkpoint $CHECKPOINT_DIR \
    --after-checkpoint ${CHECKPOINT_DIR}/data_influence/latest-unsharded \
    --port 24775 &

# GPU 1: DataMan quality server
CUDA_VISIBLE_DEVICES=1 vllm serve RuPeng/DataMan-1.5B-EN --port 8000 &

# GPU 2: Structure server
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-4B --port 8001 &

# Wait for servers to be ready
sleep 120

# Train generator (GPUs 3-6)
CUDA_VISIBLE_DEVICES=3,4,5,6 \
DATA_INFLUENCE_ENDPOINT=http://127.0.0.1:24775/get_influence \
PYTHONPATH=$SYNPRO_ROOT \
accelerate launch --num_processes 4 \
    --config_file configs/accelerate/zero2.yaml \
    generator/grpo_rephrase.py \
    --config configs/generator/rephrase.yaml
```

## Step 2b: Reformat Generator Training

Trains a generator to transform text into structured format with faithfulness verification.

```bash
# Launch reward servers (same as rephrase, plus faithfulness server)
# GPU 0: Influence server (same as above)
# GPU 1: DataMan quality server (same as above)
# GPU 2: Faithfulness server
CUDA_VISIBLE_DEVICES=2 vllm serve /path/to/faithfulness-verifier --port 8002 &

# Train generator (GPUs 3-6)
CUDA_VISIBLE_DEVICES=3,4,5,6 \
DATA_INFLUENCE_ENDPOINT=http://127.0.0.1:24775/get_influence \
REFORMAT_FAITHFULNESS_SERVER_URL=http://127.0.0.1:8002/v1 \
PYTHONPATH=$SYNPRO_ROOT \
accelerate launch --num_processes 4 \
    --config_file configs/accelerate/zero2.yaml \
    generator/grpo_reformat.py \
    --config configs/generator/reformat.yaml
```

## Step 3: Inference

Generate synthetic data at scale using data-parallel vLLM.

```bash
VLLM_USE_V1=1 TOTAL_GPUS=8 GPUS_PER_DP_RANK=1 \
MODEL_PATH=/path/to/trained-generator/checkpoint-N \
OUTPUT_DIR_NAME=synthetic_rephrase \
python scripts/run_infer_dp.py configs/pretrain/OLMo-400M.yaml
```

## Step 4: Tokenize

Convert generated JSONL files to tokenized NPY format.

```bash
python scripts/convert_text_jsonl.py --reverse \
    --input /path/to/synthetic.jsonl \
    --output /path/to/synthetic.npy
```

## Step 5: Continue Pretraining

Train the downstream model on a mix of organic + synthetic data.

```bash
export LOAD_PATH=/path/to/current-checkpoint
export OUTPUT_DIR_NAME=synthetic_rephrase  # matches Step 3

envsubst < configs/pretrain/OLMo-400M.yaml > /tmp/pretrain.yaml
torchrun --nproc_per_node=8 scripts/train.py /tmp/pretrain.yaml
```

The pretrain config uses `${LOAD_PATH}` and `${OUTPUT_DIR_NAME}` as environment variables to specify the checkpoint and synthetic data directory.

## Full Pipeline

For automated execution of all steps:

```bash
bash scripts/pipeline.sh /path/to/checkpoint
```

See `scripts/pipeline.sh` for the full orchestration script.

## Reward Functions

| Reward | Weight | Description | Server |
|-|-|-|-|
| `data_influence` | 3.0 | Steers toward content the model needs | `get_influence.py` (port 24775) |
| `quality` | 1.0 | DataMan text quality score | vLLM (port 8000) |
| `faithfulness` | 1.0 | Source-grounded factual consistency | vLLM (port 8002) |
| `similarity` | 1.0 | BERTScore semantic similarity | Local (no server) |
| `structure` | 1.0 | Structural quality preservation | vLLM (port 8001) |
| `length` | 1.0 | Length appropriateness | Local (no server) |
| `format` | 1.0 | Output format compliance | Local (no server) |

## Configuration

All configs use environment variables substituted via `envsubst` at runtime. Required variables:

| Variable | Description | Example |
|-|-|-|
| `LOCAL_ROOT` | Root directory for preprocessed data | `/data/synpro` |
| `CHECKPOINT_DIR` | Path to model checkpoint for probing | `/tmp/out/OLMo-400M/step10920-unsharded` |
| `CHECKPOINT_NAME` | Checkpoint identifier for logging | `step10920-unsharded` |
| `OUTPUT_DIR` | Output directory for probe influence results | `${CHECKPOINT_DIR}/data_influence` |
| `LOAD_PATH` | Checkpoint to load for continue pretraining | same as `CHECKPOINT_DIR` |
| `SAVE_FOLDER` | Output directory for pretrain checkpoints | `/tmp/out/OLMo-400M/run_name` |
| `OUTPUT_DIR_NAME` | Name of generated data directory | `OLMo2-1.1B-reformat-grpo-240` |
| `SYNPRO_DATA` | Path to training JSONL for GRPO generator | `data/training_data.jsonl` |

See `configs/` for templates.

## Citation

If you find this work useful, please consider citing:

```
@article{yu2026synpro,
  title={Generating Pretraining Tokens from Organic Data for Data-Bound Scaling},
  author={Yu, Zichun and Xiong, Chenyan},
  journal={ArXiv preprint},
  year={2026}
}
```