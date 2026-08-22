"""Optional LLM-as-judge scoring.

The results reported in the paper use rule-based scoring (`--judge_mode rule`),
which is the default and requires no API access. This module provides a thin,
provider-agnostic wrapper for the optional `--judge_mode api|both` paths.

To use it, set the following environment variables:

    export VIG_JUDGE_API_KEY=<your key>
    export VIG_JUDGE_BASE_URL=<your OpenAI-compatible endpoint>
    export VIG_JUDGE_MODEL=<model name>          # optional

No credentials are bundled with this repository.
"""

import os
from concurrent.futures import ThreadPoolExecutor


JUDGE_PROMPT = (
    "You are grading a model's answer to a multimodal reasoning question.\n"
    "Question: {question}\n"
    "Reference answer: {gold}\n"
    "Model answer: {pred}\n\n"
    "Reply with exactly one word, CORRECT or INCORRECT, judging whether the "
    "model answer is semantically equivalent to the reference answer."
)


class ApiJudge:
    """Judge answers with an OpenAI-compatible chat endpoint.

    Credentials are read from the environment; nothing is hard-coded.
    """

    def __init__(self, workers: int = 32, model: str | None = None):
        self.workers = workers
        self.model = model or os.environ.get("VIG_JUDGE_MODEL", "gpt-4o-mini")
        self.api_key = os.environ.get("VIG_JUDGE_API_KEY")
        self.base_url = os.environ.get("VIG_JUDGE_BASE_URL")
        if not self.api_key:
            raise RuntimeError(
                "LLM-as-judge scoring requires VIG_JUDGE_API_KEY (and usually "
                "VIG_JUDGE_BASE_URL). Use --judge_mode rule to reproduce the "
                "results reported in the paper without any API access."
            )
        from openai import OpenAI  # imported lazily so `rule` mode needs no SDK

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _check_one(self, item: dict) -> bool:
        prompt = JUDGE_PROMPT.format(
            question=item.get("question", ""),
            gold=item.get("gold", ""),
            pred=item.get("pred", ""),
        )
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=8,
            )
            return "CORRECT" in resp.choices[0].message.content.upper()
        except Exception:
            # Fall back to the rule-based verdict already attached to the item.
            return bool(item.get("correct", False))

    def check_batch(self, items: list) -> list:
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(self._check_one, items))
