import os
import re
from typing import Optional

import requests


class DataInfluenceClient:
    """Client for querying the /get_influence service."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        batch_size: Optional[int] = None,
    ):
        endpoint = endpoint or os.getenv("DATA_INFLUENCE_ENDPOINT")
        if endpoint is None:
            host = os.getenv("DATA_INFLUENCE_HOST", "127.0.0.1")
            port = os.getenv("DATA_INFLUENCE_PORT", "24775")
            route = os.getenv("DATA_INFLUENCE_ROUTE", "/get_influence")
            endpoint = f"http://{host}:{port}{route}"

        self.endpoint = endpoint
        self.timeout_seconds = float(
            timeout_seconds or os.getenv("DATA_INFLUENCE_TIMEOUT", "120")
        )
        self.batch_size = int(
            batch_size or os.getenv("DATA_INFLUENCE_BATCH_SIZE", "16")
        )
        self.session = requests.Session()

    def _query_batch(self, texts: list[str]) -> list[float]:
        response = self.session.post(
            self.endpoint,
            json={"texts": texts},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return [float(x) for x in response.json()["influence_scores"]]

    def score_texts(self, texts: list[str]) -> list[float]:
        if not texts:
            return []

        outputs: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            outputs.extend(self._query_batch(texts[start : start + self.batch_size]))
        return outputs


_client: Optional[DataInfluenceClient] = None


def _get_client() -> DataInfluenceClient:
    global _client
    if _client is None:
        _client = DataInfluenceClient()
    return _client


def data_influence_reward(
    completions: list[list[dict[str, str]]],
    text: list[str],
    **kwargs,
) -> list[float]:
    """Reward with data influence computed from two checkpoints (loss_before - loss_after)."""
    contents = []
    for completion, t in zip(completions, text):
        content = completion[0]["content"]
        match = re.search(r"Here is a paraphrased version:(.*)", content, re.DOTALL)
        if match and (c := match.group(1).strip()):
            if len(c) >= 0.7 * len(t):
                # If the paraphrased version is not too short, use it for scoring.
                contents.append(c)
            else:
                # Otherwise, use the original text.
                contents.append(t)
        else:
            contents.append(t)

    cur_values = _get_client().score_texts(contents)
    baseline_values = _get_client().score_texts(text)

    return [float(cur) - float(base) for cur, base in zip(cur_values, baseline_values)]
