# Data-parallel version of run_infer_olmo_reformat.py
# Usage:
#   VLLM_USE_V1=1 python scripts/dclm/run_infer_olmo_reformat_dp.py <config_path>
#
# Env vars: TOTAL_GPUS (default 8), GPUS_PER_DP_RANK (default 2),
#           MODEL_PATH, OUTPUT_DIR_NAME, LOCAL_ROOT, VLLM_SPECULATIVE_MODEL

import os
import sys
import json
import re
import math
from multiprocessing import Process
from pathlib import Path

import datasets
import yaml
from tqdm import tqdm

from vllm import LLM, SamplingParams

TOTAL_GPUS = int(os.environ.get("TOTAL_GPUS", 8))
GPUS_PER_DP_RANK = int(os.environ.get("GPUS_PER_DP_RANK", 1))
DP_SIZE = TOTAL_GPUS // GPUS_PER_DP_RANK
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/tmp/synthetic_data_generator_OLMo2-1.1B-reformat_step2000/checkpoint-60",
)
OUTPUT_DIR_NAME = os.environ.get("OUTPUT_DIR_NAME", "step37128_OLMo2-1.1B-reformat-grpo-60")

reformat_prompt = """Task: Read the text and convert it into a different format.
Follow these instructions:
1. Extract diverse key information that covers different aspects of the text.
2. Produce various types of structured entries such as:
    - Factual statements that capture concrete details from the text.
    - Analytical entries that explain relationships or reasoning in the text.
    - Conceptual entries that define or clarify key terms and ideas.
    - Comparative entries that highlight similarities or differences between concepts.
    - Problem-solving entries that capture any logical, mathematical, or procedural knowledge.
3. Focus on factual information, important knowledge, and concrete details.
4. Write entries using clear and concise language.
5. Use plain text. Do not use Markdown.
6. Each entry should be on a separate line with a "Topic:" and "Content:" tag.

Text:
{TEXT}
Task:
After reading the above text, produce up to 8 structured entries following the instructions. Give your response in this format:
Here are the structured entries based on the provided text:
- Topic: [first topic] Content: [first content]
- Topic: [second topic] Content: [second content]
...."""

sampling_params = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=2048)


def make_conversation(example, tokenizer=None, max_prompt_tokens=4096):
    sys_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite responses to the user."
    text_content = example["text"]

    if tokenizer is not None:
        base_tokens = tokenizer.encode(reformat_prompt.replace("{TEXT}", ""))
        sys_tokens = tokenizer.encode(sys_prompt)
        available = max_prompt_tokens - len(base_tokens) - len(sys_tokens) - 50
        text_tokens = tokenizer.encode(text_content)
        if len(text_tokens) > available:
            text_content = (
                tokenizer.decode(text_tokens[: available // 2])
                + "... "
                + tokenizer.decode(text_tokens[-(available // 2) :])
            )

    return {
        "prompt": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": reformat_prompt.format(TEXT=text_content)},
        ]
    }


REFORMAT_HEADING_PATTERN = re.compile(
    r"Here are the structured entries based on the provided text:",
    re.IGNORECASE,
)


def transform(llm, conversations):
    outputs = llm.chat(conversations, sampling_params, use_tqdm=False)
    results = []
    for output in outputs:
        content = output.outputs[0].text
        match = REFORMAT_HEADING_PATTERN.search(content)
        results.append(content[match.end() :].strip() if match else "")
    return results


def vllm_inference(llm, dataset):
    batch_size = 8192
    all_texts = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Inference..."):
        batch = dataset[i : i + batch_size]["prompt"]
        all_texts.extend(transform(llm, batch))
    return all_texts


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def paths_from_config(config_path: Path, local_root: str | None) -> list[Path]:
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    paths = config.get("data", {}).get("paths", [])
    out = []
    for p in paths:
        p = str(p)
        if local_root is not None:
            p = p.replace("${LOCAL_ROOT}", local_root)
        p = p.replace("dolma2-tokenizer", "text").replace(".npy", ".jsonl")
        out.append(Path(p))
    return out


def load_and_chunk(read_file_path):
    """Load jsonl and split each document into 7000-char chunks. Returns (data_list, tids)."""
    data_list, tids = [], []
    tid = 0
    for item in load_jsonl(read_file_path):
        text = item["text"]
        for j in range(0, len(text), 7000):
            data_list.append({"text": text[j : j + 7000]})
            tids.append(tid)
        tid += 1
    return data_list, tids


def aggregate_and_write(f, tids, transformed_texts):
    """Aggregate chunks by tid, write one JSONL line per document. Returns count of empty docs."""
    cnt = 0
    prev_tid = tids[0] if tids else 0
    texts = []
    for tid, text in zip(tids, transformed_texts):
        if tid == prev_tid:
            texts.append(text)
        else:
            cur = " ".join(texts).strip()
            if not cur:
                cnt += 1
            f.write(json.dumps({"text": cur}) + "\n")
            prev_tid = tid
            texts = [text]
    if texts:
        cur = " ".join(texts).strip()
        if not cur:
            cnt += 1
        f.write(json.dumps({"text": cur}) + "\n")
    return cnt


def dp_worker(dp_rank, gpu_ids, path_assignments, config_path_str):
    # Use script-level data parallelism only; do not enable vLLM internal DP.
    os.environ.pop("VLLM_DP_RANK", None)
    os.environ.pop("VLLM_DP_SIZE", None)
    os.environ.pop("VLLM_DP_MASTER_IP", None)
    os.environ.pop("VLLM_DP_MASTER_PORT", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    print(f"[DP rank {dp_rank}] GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}, paths: {len(path_assignments)}")

    llm_kwargs = {"model": MODEL_PATH, "tensor_parallel_size": len(gpu_ids)}
    spec = os.environ.get("VLLM_SPECULATIVE_MODEL")
    if spec:
        llm_kwargs["speculative_config"] = {
            "method": "suffix",
            "speculative_model": spec,
            "num_speculative_tokens": 32,
        }

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    all_read_paths = paths_from_config(Path(config_path_str), os.environ.get("LOCAL_ROOT"))

    for path_idx, shard_rank, num_shards in path_assignments:
        read_file_path = all_read_paths[path_idx]
        base_write = str(read_file_path).replace("text", OUTPUT_DIR_NAME)
        write_file_path = Path(base_write + (f".shard{shard_rank}_of_{num_shards}" if num_shards > 1 else ""))

        if write_file_path.exists():
            print(f"[DP rank {dp_rank}] Exists, skipping: {write_file_path}")
            continue

        print(f"[DP rank {dp_rank}] Processing: {read_file_path} (shard {shard_rank}/{num_shards})")

        data_list, tids = load_and_chunk(read_file_path)

        # Shard by document boundary: find unique tids and split them
        if num_shards > 1:
            unique_tids = sorted(set(tids))
            n = len(unique_tids)
            shard_size = math.ceil(n / num_shards)
            tid_start = shard_rank * shard_size
            tid_end = min(tid_start + shard_size, n)
            keep_tids = set(unique_tids[tid_start:tid_end])
            indices = [i for i, t in enumerate(tids) if t in keep_tids]
            data_list = [data_list[i] for i in indices]
            tids = [tids[i] for i in indices]

        if not data_list:
            print(f"[DP rank {dp_rank}] No data in shard, skipping.")
            continue

        dataset = datasets.Dataset.from_list(data_list).map(lambda x: make_conversation(x, tokenizer=tokenizer))
        transformed_texts = vllm_inference(llm, dataset)

        write_file_path.parent.mkdir(parents=True, exist_ok=True)
        with write_file_path.open("w") as f:
            cnt = aggregate_and_write(f, tids, transformed_texts)
        print(f"[DP rank {dp_rank}] Wrote {len(transformed_texts)} items, {cnt} invalid.")

    # Kill vLLM GPU worker sub-processes before exiting to free GPU memory.
    import subprocess
    subprocess.run(['pkill', '-KILL', '-P', str(os.getpid())], capture_output=True)
    # Bypass Python/vLLM atexit cleanup which hangs on executor shutdown.
    os._exit(0)


def merge_shards(all_read_paths, assignments):
    """Concatenate shard files into one merged file per path, then delete shards."""
    sharded_paths = {}
    for dp_assignments in assignments:
        for path_idx, _, num_shards in dp_assignments:
            if num_shards > 1:
                sharded_paths[path_idx] = num_shards

    for path_idx, num_shards in sharded_paths.items():
        base = str(all_read_paths[path_idx]).replace("text", OUTPUT_DIR_NAME)
        merged_path = Path(base)
        shard_paths = [Path(base + f".shard{s}_of_{num_shards}") for s in range(num_shards)]

        missing = [p for p in shard_paths if not p.exists()]
        if missing:
            print(f"Warning: missing shards for {merged_path}: {missing}, skipping.")
            continue
        if merged_path.exists():
            print(f"Merged file exists, skipping: {merged_path}")
            continue

        print(f"Merging {num_shards} shards -> {merged_path}")
        with merged_path.open("w") as out_f:
            for sp in shard_paths:
                with sp.open("r") as in_f:
                    for line in in_f:
                        out_f.write(line)

        for sp in shard_paths:
            sp.unlink()


def main():
    config_path = sys.argv[1]
    all_read_paths = paths_from_config(Path(config_path), os.environ.get("LOCAL_ROOT"))
    num_paths = len(all_read_paths)

    print(f"GPUs: {TOTAL_GPUS}, TP: {GPUS_PER_DP_RANK}, DP: {DP_SIZE}, Paths: {num_paths}")

    # Assign DP ranks to paths. When DP > paths, multiple ranks shard one path.
    assignments = [[] for _ in range(DP_SIZE)]

    if num_paths >= DP_SIZE:
        for path_idx in range(num_paths):
            assignments[path_idx % DP_SIZE].append((path_idx, 0, 1))
    else:
        base = DP_SIZE // num_paths
        extra = DP_SIZE % num_paths
        dp_rank = 0
        for path_idx in range(num_paths):
            n_ranks = base + (1 if path_idx < extra else 0)
            for shard_rank in range(n_ranks):
                assignments[dp_rank].append((path_idx, shard_rank, n_ranks))
                dp_rank += 1

    for r, a in enumerate(assignments):
        print(f"  DP rank {r}: {a}")

    if TOTAL_GPUS % GPUS_PER_DP_RANK != 0:
        raise ValueError(f"TOTAL_GPUS ({TOTAL_GPUS}) must be divisible by GPUS_PER_DP_RANK ({GPUS_PER_DP_RANK}).")

    env_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_visible:
        visible_gpu_ids = [x.strip() for x in env_visible.split(",") if x.strip()]
    else:
        visible_gpu_ids = [str(i) for i in range(TOTAL_GPUS)]

    visible_gpu_ids = visible_gpu_ids[:TOTAL_GPUS]
    procs = []
    for dp_rank in range(DP_SIZE):
        start = dp_rank * GPUS_PER_DP_RANK
        end = start + GPUS_PER_DP_RANK
        rank_gpu_ids = visible_gpu_ids[start:end]
        proc = Process(
            target=dp_worker,
            args=(
                dp_rank,
                rank_gpu_ids,
                assignments[dp_rank],
                config_path,
            ),
        )
        proc.start()
        procs.append(proc)

    exit_code = 0
    worker_timeout = 4 * 3600  # 4 hours max per worker
    for proc in procs:
        proc.join(timeout=worker_timeout)
        if proc.is_alive():
            print(f"Worker PID {proc.pid} still alive after {worker_timeout}s timeout — killing.")
            proc.kill()
            proc.join()
            exit_code = 1
        elif proc.exitcode and proc.exitcode not in (0, -9):
            exit_code = proc.exitcode

    if exit_code != 0:
        print(f"Workers failed with exit code {exit_code}")
        exit(exit_code)

    merge_shards(all_read_paths, assignments)
    print("Done.")


if __name__ == "__main__":
    main()
