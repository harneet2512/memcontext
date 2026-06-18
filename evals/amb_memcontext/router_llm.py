"""OpenAI-compatible router LLMs for AMB reader and judge roles."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

try:  # pragma: no cover - imported only when AMB is available.
    from memory_bench.llm.base import LLM, Schema
except Exception:  # noqa: BLE001

    class Schema:  # type: ignore[no-redef]
        properties: dict[str, Any]
        required: list[str]

    class LLM:  # type: ignore[no-redef]
        @property
        def model_id(self) -> str:
            return self.__class__.__name__


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TOKENROUTER_BASE_URL = "https://api.tokenrouter.com/v1"
OPENROUTER_READER_MODEL = "openai/gpt-oss-120b:free"
TOKENROUTER_GEMINI_MODEL = "google/gemini-3-flash-preview"
TOKENROUTER_JUDGE_MODEL = TOKENROUTER_GEMINI_MODEL
TOKENROUTER_EXTRACTOR_MODEL = TOKENROUTER_GEMINI_MODEL
_MAX_RETRIES = 6
_RETRY_BASE_DELAY = 5


def _first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise RuntimeError("Router returned non-object JSON content.")
    return parsed


def _default_for_schema_property(spec: Any) -> Any:
    if not isinstance(spec, dict):
        return ""
    typ = spec.get("type")
    if isinstance(typ, list):
        typ = next((item for item in typ if item != "null"), typ[0] if typ else "string")
    if typ == "array":
        return []
    if typ == "object":
        return {}
    if typ == "boolean":
        return False
    if typ in {"integer", "number"}:
        return 0
    return ""


def _coerce_to_schema(data: dict[str, Any], schema: Schema) -> dict[str, Any]:
    """AMB indexes required keys directly; keep router oddities from crashing it."""
    out = dict(data)
    properties = getattr(schema, "properties", {}) or {}
    for key in getattr(schema, "required", []) or []:
        if key in out:
            continue
        if key == "answer":
            for alias in ("response", "final_answer", "content", "text"):
                value = out.get(alias)
                if isinstance(value, str) and value.strip():
                    out[key] = value
                    break
        if key == "reasoning" and key not in out:
            for alias in ("rationale", "explanation", "thought"):
                value = out.get(alias)
                if isinstance(value, str) and value.strip():
                    out[key] = value
                    break
        if key not in out:
            out[key] = _default_for_schema_property(properties.get(key))
    return out


class _RouterLLM(LLM):
    role = "router"
    default_base_url = OPENROUTER_BASE_URL
    default_model = OPENROUTER_READER_MODEL
    model_env = "ROUTER_MODEL"
    key_envs = ("ROUTER_API_KEY",)
    reasoning_effort = "low"
    reasoning_effort_env = "ROUTER_REASONING_EFFORT"
    reasoning_exclude_env = "ROUTER_REASONING_EXCLUDE"

    def __init__(self, model: str | None = None):
        self._model = model or os.environ.get(self.model_env, self.default_model)
        self._api_key = _first_env(*self.key_envs)
        self._base_url = os.environ.get(
            self.base_url_env, self.default_base_url
        ).rstrip("/")

    @property
    def base_url_env(self) -> str:
        return f"{self.role.upper().replace('-', '_')}_BASE_URL"

    @property
    def model_id(self) -> str:
        return f"{self.role}:{self._model}"

    def generate(self, prompt: str, schema: Schema) -> dict:
        if not self._api_key:
            raise RuntimeError(
                f"Missing API key for {self.role}. Set one of: {', '.join(self.key_envs)}."
            )
        payload = self._payload(prompt, schema)
        delay = _RETRY_BASE_DELAY
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._post_json(payload, schema)
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code in {429, 500, 502, 503, 504} and attempt < _MAX_RETRIES - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise RuntimeError(f"{self.role} request failed after {_MAX_RETRIES} retries: {last_error}")

    def _payload(self, prompt: str, schema: Schema) -> dict[str, Any]:
        schema_json = {
            "type": "object",
            "properties": schema.properties,
            "required": schema.required,
            "additionalProperties": False,
        }
        json_prompt = (
            f"{prompt}\n\n"
            "Return only a valid JSON object matching this schema:\n"
            f"{json.dumps(schema_json, ensure_ascii=False)}"
        )
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": json_prompt}],
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema_json,
                    "strict": True,
                },
            },
        }
        effort = os.environ.get(self.reasoning_effort_env, self.reasoning_effort).strip()
        exclude = os.environ.get(self.reasoning_exclude_env, "1").strip() != "0"
        if not effort:
            return payload
        if effort != "none":
            payload["reasoning"] = {
                "effort": effort,
                "exclude": exclude,
            }
        else:
            payload["reasoning"] = {
                "effort": "none",
                "exclude": exclude,
            }
        return payload

    def _post_json(self, payload: dict[str, Any], schema: Schema | None = None) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "X-Title": "memcontext-amb",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=180.0) as response:
            data = json.loads(response.read().decode("utf-8"))
        message = data.get("choices", [{}])[0].get("message", {})
        content = _message_content(message)
        try:
            parsed = _parse_json_object(content)
        except (json.JSONDecodeError, RuntimeError):
            parsed = {"response": content} if content.strip() else {}
        return _coerce_to_schema(parsed, schema) if schema is not None else parsed


def _message_content(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


class OpenRouterReaderLLM(_RouterLLM):
    role = "openrouter-reader"
    default_base_url = OPENROUTER_BASE_URL
    default_model = OPENROUTER_READER_MODEL
    model_env = "OMB_ANSWER_MODEL"
    key_envs = ("OPENROUTER_AMB_READER_KEY", "OPENROUTER_API_KEY")
    reasoning_effort = "high"
    reasoning_effort_env = "OMB_ANSWER_REASONING_EFFORT"
    reasoning_exclude_env = "OMB_ANSWER_REASONING_EXCLUDE"

    @property
    def base_url_env(self) -> str:
        return "OPENROUTER_BASE_URL"


class TokenRouterJudgeLLM(_RouterLLM):
    role = "tokenrouter-judge"
    default_base_url = TOKENROUTER_BASE_URL
    default_model = TOKENROUTER_JUDGE_MODEL
    model_env = "OMB_JUDGE_MODEL"
    key_envs = ("TOKENROUTER_AMB_GEMINI_KEY", "TOKENROUTER_AMB_JUDGE_KEY", "TOKENROUTER_API_KEY")
    reasoning_effort = ""
    reasoning_effort_env = "OMB_JUDGE_REASONING_EFFORT"
    reasoning_exclude_env = "OMB_JUDGE_REASONING_EXCLUDE"

    @property
    def base_url_env(self) -> str:
        return "TOKENROUTER_BASE_URL"
