from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from openai import OpenAI
import re

structure_prompt = """[Instruction]
You are given two pieces of text: an original pretraining data sample and a rephrased version.
Your task is to judge if the rephrased version preserves the **structure** of the original sample.
- By "structure", we mean formatting, style, and presentation (e.g., paragraphing, JSON, list format, code blocks, markdown usage, plain text style).
- Do NOT consider semantic meaning. Ignore whether the words are the same or the content is equivalent.
- Focus only on whether the rephrased sample follows the same textual structure as the original (e.g., if the original is plain text paragraphs, the rephrased should also be plain text; if the original has bullet lists, the rephrased should also have bullet lists).

[Output]
Output **only** `1` if the structure is preserved.
Output **only** `0` if the structure is not preserved.

[Examples]

Example 1:
Original:
This is a paragraph.
This is another line.

Rephrased:
Here is a rewritten paragraph.
Here is another line of text.

Explanation: Both are plain text paragraphs, no special formatting. Structure preserved. Output: 1

---

Example 2:
Original:
- Item one
- Item two

Rephrased:
First item. Second item.

Explanation: The original uses a bullet list, while the rephrased is plain sentences. Structure not preserved. Output: 0

---

Example 3:
Original:
{{"name": "Alice", "age": 30}}

Rephrased:
{{"person": "A.", "years": 30}}

Explanation: Both are JSON objects with the same structured format. Structure preserved. Output: 1

---

Example 4:
Original:
def add(a, b):
    return a + b

print(add(1, 2))

Rephrased:
```python
def add(x, y):
    return x + y

print(add(2, 3))
```

Explanation: The original is plain code with no markdown fences, while the rephrased introduces code fences. Structure not preserved. Output: 0

[Original]
{original_text}

[Rephrased]
{rephrased_text}
"""

class StructureInference:
    def __init__(
        self,
        model_name="Qwen/Qwen3-4B",
        max_tokens=8,
        temperature=0.0,
        seed=1024,
        use_server=True,
        server_url="http://localhost:8001/v1",
        api_key="EMPTY",
    ):
        """
        Initialize the Structure inference model.

        Args:
            model_name (str): The model name or path. Default is "Qwen3-1.7B"
            max_tokens (int): Maximum number of output tokens
            temperature (float): Generation temperature
            seed (int): Random seed for reproducibility
            use_server (bool): Whether to use an existing vLLM server instead of loading the model locally
            server_url (str): The base URL of the vLLM server (e.g., "http://localhost:8001/v1")
            api_key (str): API key for the server (use "EMPTY" for vLLM servers)
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.use_server = use_server
        self.server_url = server_url
        self.api_key = api_key

        # Set the request template based on model type (using English)
        self.request_template = structure_prompt

        # Regular expressions for filtering
        self.ellipsis_pattern = r'^["“”]…+["“”]|^["“”]\.\.\.+["“”]'
        self.link_pattern = r"!\[\]\([\/]?[^\)]+\)\n"

        # Text truncation limits (optimized for English)
        self.truncate_max_length = 1894 // 2  # Token limit for English
        self.char_truncate_max_length = 20000 // 2  # Character limit for English

        # Initialize tokenizer and model
        self._load_model()

    def _load_model(self):
        """Load the tokenizer and LLM model or initialize OpenAI client."""
        # Always load tokenizer for text preprocessing
        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left", use_fast=True
        )

        if self.use_server:
            # Initialize OpenAI client for server mode
            print(f"Initializing OpenAI client for server: {self.server_url}")
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.server_url,
            )
            print("OpenAI client initialized successfully!")
        else:
            # Initialize local vLLM model
            print(f"Loading local model: {self.model_name}")
            self.llm = LLM(model=self.model_name, tokenizer=self.model_name)

        # Set up sampling parameters
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            seed=self.seed,
            max_tokens=self.max_tokens,
        )

        print("Model loaded successfully!")

    def _preprocess_text(self, text):
        """
        Preprocess the input text by applying regular filtering and truncation.

        Args:
            text (str): Raw input text

        Returns:
            str: Preprocessed text
        """
        # Apply regular expression filtering
        text = re.sub(self.ellipsis_pattern, "", text)
        text = re.sub(self.link_pattern, "", text)

        # Truncate text if too long
        if len(text) > self.char_truncate_max_length:
            # Take first and last halves if character length exceeds limit
            mid_point = self.char_truncate_max_length // 2
            text = text[:mid_point] + text[-mid_point:]

        # Tokenize and check token length
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.truncate_max_length:
            # Truncate tokens and add ellipsis in the middle
            mid_point = self.truncate_max_length // 2
            tokens = tokens[:mid_point] + ["..."] + tokens[-mid_point:]

        # Convert tokens back to string
        text = self.tokenizer.convert_tokens_to_string(tokens).strip()

        return text

    def _create_chat_text(self, original_text, rephrased_text):
        """
        Create a chat-formatted text for the model.

        Args:
            text (str): Preprocessed text

        Returns:
            str: Chat-formatted text ready for inference
        """
        request = self.request_template.format(
            original_text=original_text, rephrased_text=rephrased_text
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant. /no_think"},
            {"role": "user", "content": request},
        ]

        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return chat_text

    def score_texts(self, original_texts, rephrased_texts):
        """
        Score multiple texts in batch.

        Args:
            texts (list): List of raw input texts to be scored

        Returns:
            list: List of parsed results based on model_type
        """
        # Preprocess all texts
        preprocessed_original = [self._preprocess_text(text) for text in original_texts]
        preprocessed_rephrased = [self._preprocess_text(text) for text in rephrased_texts]

        # Create chat-formatted texts
        chat_texts = [
            self._create_chat_text(original_text, rephrased_text)
            for original_text, rephrased_text in zip(
                preprocessed_original, preprocessed_rephrased
            )
        ]

        completions = self.client.completions.create(
            model=self.model_name,
            prompt=chat_texts,
            temperature=self.temperature,
            seed=self.seed,
            max_tokens=self.max_tokens,
        )
        results = [output.text.strip()[-1] for output in completions.choices]
        results = [int(r) if r in ["0", "1"] else 0 for r in results]

        return results


def main():
    # Create instance (here use_server=True assumes a vLLM server is running at server_url)
    scorer = StructureInference()

    # Define simple test cases
    originals = [
        "This is a paragraph.\n\nThis is another paragraph.",
        "- item one\n- item two\n- item three",
        '{"name": "Alice", "age": 30}',
    ]

    rephrased = [
        "Here is a rewritten paragraph.\n\nHere is another rewritten one.",  # should be 1
        "item one\nitem two\nitem three",  # should be 0
        '{"person": "A.", "years": 30}',  # should be 1
    ]

    # Run scoring
    results = scorer.score_texts(originals, rephrased)

    # Print results
    print("Results:", results)


if __name__ == "__main__":
    main()
