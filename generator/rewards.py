# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward functions for GRPO training."""
import re
import os
from typing import Callable
from bert_score import BERTScorer
from transformers import AutoTokenizer
from generator.infer.simple_dataman import DataManInference
from generator.infer.simple_structure import StructureInference
from generator.infer.simple_faithfulness import ReformatFaithfulnessInference
from generator.data_influence_reward import DataInfluenceClient

# fasttext_path = "$HOME/code/dclm/baselines/mappers/enrichers/quality_prediction_enrichment_models/fasttext_oh_eli5.bin"
# fasttext_model = fasttext.load_model(fasttext_path)
bert_scorer = BERTScorer(
    model_type="microsoft/deberta-large-mnli",
    device=f"cuda:{os.environ.get('LOCAL_RANK', 0)}",
)
# tokenizer = AutoTokenizer.from_pretrained(
#     "bert-base-uncased",
#     max_length=2048,
#     padding="max_length",
# )
dataman_llm = DataManInference(use_server=True)
structure_llm = StructureInference(use_server=True)
reformat_faithfulness_llm = ReformatFaithfulnessInference(
    model_name=os.environ.get("REFORMAT_FAITHFULNESS_MODEL", "Qwen/Qwen3-1.7B"),
    server_url=os.environ.get("REFORMAT_FAITHFULNESS_SERVER_URL", "http://localhost:8002/v1"),
    use_server=True,
)

PARAPHRASE_PREFIX_PATTERN = re.compile(
    re.escape("Here is a paraphrased version:"),
    re.IGNORECASE,
)
REFORMAT_HEADING_PATTERN = re.compile(
    r"Here are the structured entries based on the provided text:",
    re.IGNORECASE,
)
AUTO_EXTRACT_PATTERNS = (
    PARAPHRASE_PREFIX_PATTERN,
    REFORMAT_HEADING_PATTERN,
)


def dataman_reward(
    completions: list[list[dict[str, str]]],
    text: list[str],
    dataman_score: list[float],
    **kwargs,
) -> list[float]:
    """Reward function that uses a data quality model to score completions."""
    contents = []
    for completion in completions:
        content = completion[0]["content"]
        match = re.search(r"Here is a paraphrased version:(.*)", content, re.DOTALL)
        if match:
            # Extract the response within the tags
            response = match.group(1).strip()
            contents.append(response)
        else:
            contents.append("")

    results = dataman_llm.score_texts(contents)
    dataman_rewards = [r.get("overall_score", 0) for r in results]
    # return [dataman_rewards[i] - dataman_rewards[len(contents)] for i in range(len(contents))]
    return [dataman_rewards[i] - dataman_score[i] for i in range(len(contents))]


def dataman_reformat_reward(
    completions: list[list[dict[str, str]]],
    text: list[str],
    dataman_score: list[float],
    **kwargs,
) -> list[float]:
    """Dataman reward for reformat completions. Uses _extract_completion_auto to handle both formats."""
    contents = [_extract_completion_auto(c) for c in completions]
    results = dataman_llm.score_texts(contents)
    dataman_rewards = [r.get("overall_score", 0) for r in results]
    return [dataman_rewards[i] - dataman_score[i] for i in range(len(contents))]


def bert_score_reward(
    completions: list[list[dict[str, str]]], text: list[str], **kwargs
) -> list[float]:
    """Reward function that uses BERTScore to compare generated content with original text."""
    contents = []
    for completion in completions:
        content = completion[0]["content"]
        # match = re.search(r"<response>\n(.*?)\n</response>", content, re.DOTALL)
        match = re.search(r"Here is a paraphrased version:(.*)", content, re.DOTALL)
        if match:
            # Extract the response within the tags
            response = match.group(1).strip()
            contents.append(response)
        else:
            contents.append("")
    # print(len(completions), "completions")
    # print(f"MODEL GENERATION:\n{completions[0][0]['content']}")
    # print("-" * 80)
    # print(f"TEXT:\n{text[0]}")
    # print("-" * 80)
    P, R, F1 = bert_scorer.score(contents, text, batch_size=1)

    # Return F1 scores as rewards
    # return F1.tolist()
    return [int(float(f1) > 0.65) for f1 in F1.tolist()]
    # return [int(float(f1) > 0.75) for f1 in F1.tolist()]


def structure_reward(
    completions: list[list[dict[str, str]]],
    text: list[str],
    dataman_score: list[float],
    **kwargs,
) -> list[float]:
    """Reward function that uses a structure comparison model to score completions."""
    contents = []
    for completion in completions:
        content = completion[0]["content"]
        match = re.search(r"Here is a paraphrased version:(.*)", content, re.DOTALL)
        if match:
            # Extract the response within the tags
            response = match.group(1).strip()
            contents.append(response)
        else:
            contents.append("")

    structure_rewards = structure_llm.score_texts(text, contents)
    return structure_rewards


def length_reward(
    completions: list[list[dict[str, str]]], text: list[str], **kwargs
) -> list[float]:
    """Reward function that uses the length of the generated content."""
    contents = []
    for completion in completions:
        content = completion[0]["content"]
        # match = re.search(r"<response>\n(.*?)\n</response>", content, re.DOTALL)
        match = re.search(r"Here is a paraphrased version:(.*)", content, re.DOTALL)
        if match:
            # Extract the response within the tags
            response = match.group(1).strip()
            contents.append(response)
        else:
            contents.append("")

    return [len(c) <= 1.25 * len(t) for c, t in zip(contents, text)]


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion contains the reformat heading exactly once.
    Returns 1.0 if exactly one heading found, 0.0 if none, -1.0 if multiple (penalize repetition).
    """
    completion_contents = [completion[0]["content"] for completion in completions]
    return [
        (
            1.0
            if sum(len(list(p.finditer(c))) for p in AUTO_EXTRACT_PATTERNS) == 1
            else 0.0
        )
        for c in completion_contents
    ]


data_influence_client = DataInfluenceClient()


def _extract_completion_after_prefix(
    completion: list[dict[str, str]],
    prefix: str,
) -> str:
    """Extract generated text after a literal prefix using simple regex logic."""
    content = completion[0]["content"]
    pattern = re.escape(prefix) + r"(.*)"
    match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(1).strip()
    return ""


def _extract_completion_auto(completion: list[dict[str, str]]) -> str:
    """Extract generated text after the latest known heading/prefix."""
    content = completion[0]["content"]
    for pattern in AUTO_EXTRACT_PATTERNS:
        match = pattern.search(content)
        if match:
            return content[match.end() :].strip()
    return ""


def data_influence_reward(
    completions: list[list[dict[str, str]]],
    text: list[str],
    **kwargs,
) -> list[float]:
    """Reward with data influence computed from two checkpoints (loss_before - loss_after)."""
    contents = []
    for completion, t in zip(completions, text):
        extracted = _extract_completion_auto(completion)
        contents.append(extracted if len(extracted) >= 500 else t)

    cur_values = data_influence_client.score_texts(contents)
    baseline_values = data_influence_client.score_texts(text)

    return [float(cur) - float(base) for cur, base in zip(cur_values, baseline_values)]


def data_influence_pair_reward(
    completions: list[list[dict[str, str]]],
    text_a: list[str],
    text_b: list[str],
    **kwargs,
) -> list[float]:
    """Reward high-influence synthesized generations relative to both inputs."""
    gen_texts = []
    zero_reward_flags = []
    expected_prefix = "Here is a synthesized text capturing their shared essence:"
    for completion, a, b in zip(completions, text_a, text_b):
        gen_text = _extract_completion_after_prefix(completion, expected_prefix)
        # wrong_format = not completion[0]["content"].startswith(expected_prefix)
        wrong_format = False
        if not gen_text:
            gen_text = completion[0]["content"].strip()
            wrong_format = True

        # Pair analogue of data_influence_reward's short-output guard:
        # require at least 70% of the mean input length, otherwise fall back.
        mean_input_len = 0.5 * (len(a) + len(b))
        too_short = len(gen_text) < 0.25 * mean_input_len
        # print(len(gen_text), mean_input_len, too_short)
        # too_short = False
        if too_short:
            gen_texts.append(a if len(a) >= len(b) else b)
        else:
            gen_texts.append(gen_text)
        zero_reward_flags.append(wrong_format or too_short)

    gen_values = data_influence_client.score_texts(gen_texts)
    a_values = data_influence_client.score_texts(text_a)
    b_values = data_influence_client.score_texts(text_b)

    return [
        0.0 if zero_flag else 1.0 + 3 * (float(gen) - 0.5 * (float(a) + float(b)))
        for gen, a, b, zero_flag in zip(
            gen_values, a_values, b_values, zero_reward_flags
        )
    ]


def reformat_faithfulness_reward(
    completions: list[list[dict[str, str]]],
    text: list[str],
    **kwargs,
) -> list[float]:
    """Reward based on how faithfully the generated reformat pairs reflect the source text.

    Extracts the reformat block from each completion (after the standard heading),
    then scores each reformat pair as Faithful (1.0) or Unfaithful (0.0) using the
    SFT-trained verifier served at REFORMAT_FAITHFULNESS_SERVER_URL.
    Returns the mean faithful fraction per completion.
    """
    reformat_blocks = []
    for completion in completions:
        content = completion[0]["content"]
        match = REFORMAT_HEADING_PATTERN.search(content)
        reformat_blocks.append(content[match.end() :].strip() if match else "")

    return reformat_faithfulness_llm.score_texts(text, reformat_blocks)


def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "format": format_reward,
        "dataman": dataman_reward,
        "dataman_reformat": dataman_reformat_reward,
        "bert_score": bert_score_reward,
        "structure": structure_reward,
        "length": length_reward,
        "data_influence": data_influence_reward,
        "data_influence_pair": data_influence_pair_reward,
        "reformat_faithfulness": reformat_faithfulness_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs
