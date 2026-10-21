"""
llm_client.py - Minimal JSON client for an OpenAI-compatible endpoint (LiteLLM proxy).

Configuration (.env or environment):
  LITELLM_BASE_URL     default http://localhost:4000/v1
  LITELLM_API_KEY
  TAPIR_LLM_MODEL      default groq/openai/gpt-oss-120b
  TAPIR_LLM_MAX_CALLS  default 5 (hard cap on API requests per client)
  TAPIR_LLM_REASONING_EFFORT  default low (less reasoning output before the JSON answer)

Output format: JSON schema (structured output, strict) when the endpoint accepts it, otherwise
json_object (the schema is then only described in the prompt). The answer is validated by the
caller's `validate`; an invalid answer is retried once. Groq's json_validate_failed (a 400 raised
when the generation does not match the format; seen intermittently, with an empty generation,
on gpt-oss-120b) counts as invalid.
Every request counts towards the cap, including one rejected because json_schema is unsupported.
"""
import json, os
from collections import Counter
from dotenv import load_dotenv
import openai

load_dotenv()

BASE_URL = os.getenv("LITELLM_BASE_URL") or "http://localhost:4000/v1"
MODEL = os.getenv("TAPIR_LLM_MODEL") or "groq/openai/gpt-oss-120b"
MAX_CALLS = int(os.getenv("TAPIR_LLM_MAX_CALLS") or 5)
REASONING_EFFORT = os.getenv("TAPIR_LLM_REASONING_EFFORT") or "low"


class BudgetExceeded(RuntimeError):
    pass


class FormatError(ValueError):
    pass


class LLMClient:
    def __init__(self, model=MODEL, max_calls=MAX_CALLS, reasoning_effort=REASONING_EFFORT, sdk=None):
        """sdk: an openai.OpenAI-like object (tests); by default the LiteLLM proxy."""
        self.model, self.max_calls, self.reasoning_effort = model, max_calls, reasoning_effort
        self.sdk = sdk or openai.OpenAI(base_url=BASE_URL, api_key=os.getenv("LITELLM_API_KEY"))
        self.schema_ok = True                 # False once the endpoint rejects json_schema
        self.stats = Counter(prompt_tokens=0, completion_tokens=0, total_tokens=0,
                             calls=0, format_failures=0)

    def _request(self, prompt, schema):
        if self.stats["calls"] >= self.max_calls:
            raise BudgetExceeded(f"TAPIR_LLM_MAX_CALLS={self.max_calls} reached")
        self.stats["calls"] += 1
        fmt = (dict(type="json_schema", json_schema=dict(name="answer", schema=schema, strict=True))
               if self.schema_ok else dict(type="json_object"))
        resp = self.sdk.chat.completions.create(
            model=self.model, messages=[dict(role="user", content=prompt)], response_format=fmt,
            reasoning_effort=self.reasoning_effort)
        u = resp.usage
        if u is not None:
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                self.stats[k] += getattr(u, k, 0) or 0
        return resp.choices[0].message.content

    def ask(self, prompt, schema, validate):
        """validate(parsed JSON) -> value, raising ValueError if the format is wrong."""
        for attempt in range(2):
            try:
                try:
                    text = self._request(prompt, schema)
                except openai.BadRequestError as e:
                    if "json_validate_failed" in str(e):
                        raise ValueError("json_validate_failed") from e
                    if not self.schema_ok:
                        raise
                    self.schema_ok = False    # structured output not supported: json_object
                    text = self._request(prompt, schema)
                return validate(json.loads(text or ""))
            except ValueError as e:           # json.JSONDecodeError is a ValueError
                self.stats["format_failures"] += 1
                err = e
        raise FormatError(f"invalid answer after one retry: {err}")
