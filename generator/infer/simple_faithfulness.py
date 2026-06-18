"""reformat faithfulness verifier inference via a vLLM server (SFT-trained judge)."""

import re
from transformers import AutoTokenizer
from openai import OpenAI

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite responses to the user. /no_think"
)

VERIFIER_PROMPT_TEMPLATE = """You are given an original text passage and a list of topic-content pairs generated from it.
For each pair, determine whether it is faithful to the original text.

A faithful reformat pair satisfies ALL of the following:
1. The topic is about a subject covered or clearly implied by the original text.
2. The content is correct and supported by the original text.
3. The content does not contradict the text.

Labels:
- "Faithful": the topic is relevant and the content is correct.
- "Unfaithful_Topic": the topic covers something not covered or implied by the text.
- "Unfaithful_Content": the topic is valid but the content is wrong or unsupported.

Original Text:
{text}

Reformat Pairs:
{reformat_pairs}

Respond with exactly one label per line in this format (no extra text):
1. <label>
2. <label>
..."""

LABEL_PATTERN = re.compile(r"(Faithful|Unfaithful_Topic|Unfaithful_Content)")
# TODO: Current pattern requires optional "-" prefix before "Topic:". Models sometimes
# generate other prefixes (e.g. "**", "### Topic: [1]") that won't be matched, resulting
# in 0 faithfulness score. Consider relaxing to just anchor on "Topic:" / "Content:" with
# no prefix restriction. Risk: numbering artifacts (e.g. "[1]") in captured topic text,
# and rare false positives if source text contains "Topic:"/"Content:".
REFORMAT_PATTERN = re.compile(
    r"-?\s*Topic:\s*(.*?)\s+Content:\s*(.*?)(?=\s*-?\s*Topic:|\Z)",
    re.DOTALL | re.IGNORECASE,
)
LABEL_TO_REWARD = {
    "Faithful": 1.0,
    "Unfaithful_Topic": 0.0,
    "Unfaithful_Content": 0.0,
}


class ReformatFaithfulnessInference:
    def __init__(
        self,
        model_name="Qwen/Qwen3-1.7B",
        max_tokens=256,
        temperature=0.0,
        seed=1024,
        use_server=False,
        server_url="http://localhost:8002/v1",
        api_key="EMPTY",
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.use_server = use_server

        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left", use_fast=True
        )

        if use_server:
            print(f"Initializing OpenAI client for server: {server_url}")
            self.client = OpenAI(api_key=api_key, base_url=server_url)
        else:
            from vllm import LLM, SamplingParams

            print(f"Loading local vLLM model: {self.model_name}")
            self.llm = LLM(model=self.model_name, tensor_parallel_size=1)
            self.sampling_params = SamplingParams(
                temperature=temperature, seed=seed, max_tokens=max_tokens
            )
        print("ReformatFaithfulnessInference ready.")

    def _parse_reformat_pairs(self, reformat_block):
        pairs = []
        for m in REFORMAT_PATTERN.finditer(reformat_block):
            q = m.group(1).strip()
            a = m.group(2).strip()
            if q and a:
                pairs.append((q, a))
        return pairs

    def _make_prompt(self, orig_text, pairs):
        reformat_lines = "\n".join(
            f"{i+1}. Topic: {q} Content: {a}" for i, (q, a) in enumerate(pairs)
        )
        return VERIFIER_PROMPT_TEMPLATE.format(text=orig_text[:3000], reformat_pairs=reformat_lines)

    def _create_chat_text(self, orig_text, pairs):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._make_prompt(orig_text, pairs)},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _parse_labels(self, response, n_pairs):
        # Strip thinking blocks (Qwen3 with /no_think usually omits them, but be safe)
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        labels = []
        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue
            m = LABEL_PATTERN.search(line)
            if m:
                labels.append(m.group(1))
        return (labels + [""] * n_pairs)[:n_pairs]

    def score_texts(self, original_texts, reformat_blocks):
        """
        Args:
            original_texts: list of source document texts
            reformat_blocks: list of reformat completion blocks (text after the reformat heading)
        Returns:
            list of scalar rewards in [0.0, 1.0] (mean Faithful fraction per doc)
        """
        pairs_per_doc = [self._parse_reformat_pairs(reformat) for reformat in reformat_blocks]
        has_pairs = [i for i, p in enumerate(pairs_per_doc) if p]

        rewards = [0.0] * len(original_texts)
        if not has_pairs:
            return rewards

        chat_texts = [
            self._create_chat_text(original_texts[i], pairs_per_doc[i])
            for i in has_pairs
        ]

        if self.use_server:
            completions = self.client.completions.create(
                model=self.model_name,
                prompt=chat_texts,
                temperature=self.temperature,
                seed=self.seed,
                max_tokens=self.max_tokens,
            )
            responses = [choice.text.strip() for choice in completions.choices]
        else:
            outputs = self.llm.generate(chat_texts, self.sampling_params)
            responses = [o.outputs[0].text.strip() for o in outputs]

        for doc_idx, response in zip(has_pairs, responses):
            pairs = pairs_per_doc[doc_idx]
            labels = self._parse_labels(response, len(pairs))
            scores = [LABEL_TO_REWARD.get(l, 0.0) for l in labels]
            rewards[doc_idx] = sum(scores) / len(scores) if scores else 0.0

        return rewards
