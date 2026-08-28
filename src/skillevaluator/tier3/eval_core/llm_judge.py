# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LLM judge prompt builders and public-provider caller.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

from skillevaluator.inference.types import EmptyLLMResponseError
from skillevaluator.provider_config import CHAT_DEFAULT_OPENAI, _model_leaf, _supports_custom_temperature

logger = logging.getLogger(__name__)

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
NVIDIA_BUILD_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_JUDGE_MODEL = CHAT_DEFAULT_OPENAI

_ERROR_REDACTION_MARKER = "[REDACTED]"
_JUDGE_ERROR_REASON_LIMIT = 512
_JUDGE_TEXT_LIMIT = 512
# Match verifier log redaction; shorter placeholders can corrupt ordinary diagnostic text.
_MIN_EXACT_SECRET_LENGTH = 8
_CREDENTIAL_ENV_VARS = (
    "OPENAI_API_KEY",
    "NVIDIA_API_KEY",
    "ANTHROPIC_API_KEY",
    "SKILL_EVAL_LLM_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SECURITY_TOKEN",
    "AWS_SESSION_TOKEN",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN",
)


# ---------------------------------------------------------------------------
# Public provider HTTP caller
# ---------------------------------------------------------------------------


def _dedupe_models(models: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for model in models:
        model = str(model or "").strip()
        if model and model not in seen:
            seen.add(model)
            result.append(model)
    return result


def _fallback_models(primary_model: str) -> list[str]:
    env_fallbacks = [
        item.strip() for item in os.environ.get("LLM_JUDGE_FALLBACK_MODELS", "").split(",") if item.strip()
    ]
    return _dedupe_models([primary_model, *env_fallbacks])


def _provider() -> str:
    configured = os.environ.get("SKILL_EVAL_LLM_PROVIDER", "").strip().lower()
    if configured:
        return configured
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("NVIDIA_API_KEY"):
        return "nv_build"
    return ""


def _resolve_url(provider: str) -> str:
    if provider == "nv_build":
        return os.environ.get("SKILL_EVAL_LLM_BASE_URL") or NVIDIA_BUILD_CHAT_URL
    base_url = os.environ.get("SKILL_EVAL_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    return base_url.rstrip("/") + "/chat/completions" if base_url else OPENAI_CHAT_URL


def _is_native_openai_chat_url(provider: str, request_url: str) -> bool:
    if str(provider or "").strip().casefold() != "openai":
        return False

    raw_url = str(request_url or "")
    if raw_url != raw_url.strip() or any(ord(character) < 0x20 or ord(character) == 0x7F for character in raw_url):
        return False
    try:
        parsed = urlparse(raw_url)
        port = parsed.port
    except ValueError:
        return False

    return (
        parsed.scheme.casefold() == "https"
        and parsed.hostname is not None
        and parsed.hostname.casefold() == "api.openai.com"
        and parsed.netloc.casefold() in {"api.openai.com", "api.openai.com:443"}
        and port in {None, 443}
        and parsed.path in {"/v1/chat/completions", "/v1/chat/completions/"}
        and parsed.username is None
        and parsed.password is None
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
        and ";" not in raw_url
        and "?" not in raw_url
        and "#" not in raw_url
    )


def _configured_secret_values(extra_secret_values: tuple[str | None, ...] = ()) -> list[str]:
    values = {
        value
        for name in _CREDENTIAL_ENV_VARS
        if (value := os.environ.get(name, "")) and len(value) >= _MIN_EXACT_SECRET_LENGTH
    }
    for value in extra_secret_values:
        text = str(value) if value else ""
        if len(text) >= _MIN_EXACT_SECRET_LENGTH:
            values.add(text)
    return sorted(values, key=len, reverse=True)


def _redact_configured_credentials(text: str, extra_secret_values: tuple[str | None, ...] = ()) -> str:
    redacted = str(text)
    for secret in _configured_secret_values(extra_secret_values):
        redacted = redacted.replace(secret, _ERROR_REDACTION_MARKER)
    return redacted


def _judge_error(error_reason: str, **metadata: Any) -> dict[str, Any]:
    """Return a bounded, redacted result that cannot be mistaken for a judged zero."""
    safe_reason = _redact_configured_credentials(error_reason).strip() or "LLM judge failed"
    if len(safe_reason) > _JUDGE_ERROR_REASON_LIMIT:
        safe_reason = safe_reason[: _JUDGE_ERROR_REASON_LIMIT - 3] + "..."
    return {**metadata, "score": None, "status": "error", "reason": safe_reason}


def _bounded_judge_text(value: Any) -> str:
    """Normalize trusted-shape model text before it reaches artifacts and reports."""
    text = _redact_configured_credentials(value).strip() if isinstance(value, str) else ""
    if len(text) > _JUDGE_TEXT_LIMIT:
        text = text[: _JUDGE_TEXT_LIMIT - 3] + "..."
    return text


def _finite_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, int):
        if value <= 0:
            return 0.0
        return 1.0
    score = float(value)
    if not math.isfinite(score):
        return None
    return max(0.0, min(1.0, score))


def _format_http_error_with_fallback(error: urllib.error.HTTPError) -> tuple[str, bool]:
    try:
        body = error.read().decode("utf-8", "replace").strip()
    except Exception:
        body = ""
    raw_detail = f"HTTP {error.code}: {error.reason}"
    safe_detail = raw_detail
    if body:
        raw_detail = f"{raw_detail} - {body}"
        safe_detail = f"{safe_detail} - {_redact_configured_credentials(body)[:500]}"
    return _redact_configured_credentials(safe_detail), _should_try_fallback(raw_detail)


def _format_http_error(error: urllib.error.HTTPError) -> str:
    return _format_http_error_with_fallback(error)[0]


def _should_try_fallback(error: str) -> bool:
    text = error.lower()
    return (
        "key_model_access_denied" in text
        or "not allowed to access model" in text
        or "invalid model" in text
        or "model not found" in text
    )


def _chat_completion_payload(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    provider: str | None = None,
    request_url: str | None = None,
) -> dict[str, Any]:
    resolved_provider = _provider() if provider is None else provider
    resolved_request_url = _resolve_url(resolved_provider) if request_url is None else request_url
    token_key = (
        "max_completion_tokens"
        if _model_leaf(model).startswith("gpt-5")
        and _is_native_openai_chat_url(resolved_provider, resolved_request_url)
        else "max_tokens"
    )
    payload: dict[str, Any] = {
        "model": model,
        token_key: max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None and _supports_custom_temperature(model):
        payload["temperature"] = temperature
    return payload


def call_public_llm(
    prompt: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout: int = 60,
    allow_model_fallback: bool = True,
) -> tuple[str | None, str | None]:
    """Call the configured public provider through the shared client.

    The shared client is responsible for OpenAI-compatible endpoints, Anthropic,
    and Bedrock. Keeping this judge on that path prevents provider behavior from
    drifting between dataset generation, Tier 1, and Tier 3.
    """
    _ = timeout, allow_model_fallback
    try:
        from skillevaluator.inference.client import LLMClient

        client = LLMClient(
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return client.completions("You are a precise evaluation judge.", prompt), None
    except EmptyLLMResponseError:
        return "", None
    except Exception as exc:
        detail = f"Public provider call failed: {exc}"
        return None, _redact_configured_credentials(detail, (api_key,))


_JSON_WHITESPACE = " \t\r\n"
_MAX_JSON_TEXT_CHARS = 100_000
_MAX_JSON_NESTING = 128
_JSON_NUMBER_PREFIX_RE = re.compile(r"-?(?:(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]*)?|(?:0|[1-9][0-9]*)\.)?")


def _balanced_json_container_end(text: str, start: int) -> int | None:
    """Return the exclusive end of one bounded structural container."""
    if start >= len(text) or text[start] not in "{[":
        return None
    stack: list[str] = []
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
            if len(stack) > _MAX_JSON_NESTING:
                return None
        elif ch in "}]":
            expected = "{" if ch == "}" else "["
            if not stack or stack[-1] != expected:
                return None
            stack.pop()
            if not stack:
                return i + 1
    return None


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build an object while rejecting ambiguous duplicate members."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object member")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(_value: str) -> None:
    raise ValueError("Non-standard JSON constant")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON number overflowed to a non-finite value")
    return parsed


def _json_nesting_within_limit(text: str) -> bool:
    """Bound structural nesting without recursively parsing partial JSON."""
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "{[":
            depth += 1
            if depth > _MAX_JSON_NESTING:
                return False
        elif character in "}]" and depth:
            depth -= 1
    return True


def _extract_json(text: str) -> dict[str, Any] | list[Any] | None:
    """Extract a JSON payload from LLM response text.

    Tolerates markdown fences and prose around exactly one valid bounded JSON
    container. Multiple complete documents are ambiguous, and an unfinished
    earlier structural segment blocks promotion of a nested object. Top-level
    arrays parse through unchanged (``harbor.report`` relies on that); judge
    callers must dict-check the result themselves.
    """
    text = (text or "").strip()
    if not text or len(text) > _MAX_JSON_TEXT_CHARS:
        return None

    documents: list[dict[str, Any] | list[Any]] = []
    index = 0
    while index < len(text):
        if text[index] not in "{[":
            index += 1
            continue
        end = _balanced_json_container_end(text, index)
        if end is None:
            return documents[0] if documents else None
        candidate = text[index:end]
        try:
            parsed = json.loads(
                candidate,
                object_pairs_hook=_reject_duplicate_object_pairs,
                parse_constant=_reject_nonstandard_json_constant,
                parse_float=_parse_finite_json_float,
            )
        except (json.JSONDecodeError, RecursionError, ValueError):
            index = end
            continue
        if isinstance(parsed, (dict, list)):
            documents.append(parsed)
            if len(documents) > 1:
                return None
        index = end
    return documents[0] if documents else None


def _is_json_string_prefix(text: str) -> bool:
    """Return whether an unfinished bounded string can be completed as JSON."""
    if not text.startswith('"'):
        return False
    index = 1
    while index < len(text):
        character = text[index]
        if ord(character) < 0x20 or character == '"':
            return False
        if character != "\\":
            index += 1
            continue
        index += 1
        if index >= len(text):
            return True
        escape = text[index]
        if escape == "u":
            for offset in range(1, 5):
                if index + offset >= len(text):
                    return True
                if text[index + offset] not in "0123456789abcdefABCDEF":
                    return False
            index += 5
        elif escape in '"\\/bfnrt':
            index += 1
        else:
            return False
    return True


def _is_json_scalar_prefix(text: str) -> bool:
    if not text:
        return True
    if text.startswith('"'):
        return _is_json_string_prefix(text)
    literals = {"t": "true", "f": "false", "n": "null"}
    if text[0] in literals:
        return literals[text[0]].startswith(text)
    if text[0] == "-" or text[0] in "0123456789":
        return _JSON_NUMBER_PREFIX_RE.fullmatch(text) is not None
    return False


def _is_append_only_json_object_prefix(fragment: str) -> bool:
    """Validate an unfinished flat result entry using bounded decoder steps."""
    if not fragment or len(fragment) > _MAX_JSON_TEXT_CHARS:
        return False

    decoder = json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_object_pairs,
        parse_constant=_reject_nonstandard_json_constant,
        parse_float=_parse_finite_json_float,
    )

    def _skip_whitespace(index: int) -> int:
        while index < len(fragment) and fragment[index] in _JSON_WHITESPACE:
            index += 1
        return index

    index = _skip_whitespace(0)
    if index >= len(fragment) or fragment[index] != "{":
        return False
    index += 1
    keys: set[str] = set()
    while True:
        index = _skip_whitespace(index)
        if index >= len(fragment):
            return True
        if fragment[index] == "}":
            return False
        try:
            key, next_index = decoder.raw_decode(fragment, index)
        except (json.JSONDecodeError, RecursionError, ValueError):
            return _is_json_string_prefix(fragment[index:])
        if not isinstance(key, str) or key in keys:
            return False
        keys.add(key)
        index = _skip_whitespace(next_index)
        if index >= len(fragment):
            return True
        if fragment[index] != ":":
            return False
        index = _skip_whitespace(index + 1)
        if index >= len(fragment):
            return True
        value_start = index
        try:
            value, next_index = decoder.raw_decode(fragment, index)
        except (json.JSONDecodeError, RecursionError, ValueError):
            return _is_json_scalar_prefix(fragment[value_start:])
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and next_index < len(fragment)
            and fragment[next_index] in ".eE"
        ):
            return _JSON_NUMBER_PREFIX_RE.fullmatch(fragment[value_start:]) is not None
        index = _skip_whitespace(next_index)
        if index >= len(fragment):
            return True
        if fragment[index] == "}":
            return False
        if fragment[index] != ",":
            return False
        index += 1


def _salvage_behavior_results(text: str) -> list[dict[str, Any]]:
    """Recover complete per-behavior entries from a truncated ``results`` array.

    Reasoning judges that hit the output-token cap emit ``{"results": [...`` and
    stop mid-entry (``finish_reason="length"``); every fully-formed ``{...}``
    entry before the cut is still valid JSON and can be scored.
    """
    text = text or ""
    if len(text) > _MAX_JSON_TEXT_CHARS or not _json_nesting_within_limit(text):
        return []
    object_start = text.find("{")
    if object_start == -1:
        return []
    if any(character in "[]{}" for character in text[:object_start]):
        return []

    decoder = json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_object_pairs,
        parse_constant=_reject_nonstandard_json_constant,
        parse_float=_parse_finite_json_float,
    )

    def _skip_whitespace(index: int) -> int:
        while index < len(text) and text[index] in " \t\r\n":
            index += 1
        return index

    # Parse only complete top-level fields preceding ``results``. This rejects
    # nested/unrelated arrays and lets us validate a score emitted before the
    # array without requiring the outer object itself to be complete.
    i = object_start + 1
    array_start = None
    seen_keys: set[str] = set()
    while i < len(text):
        i = _skip_whitespace(i)
        try:
            key, i = decoder.raw_decode(text, i)
        except (json.JSONDecodeError, RecursionError, ValueError):
            return []
        if not isinstance(key, str):
            return []
        if key in seen_keys:
            return []
        seen_keys.add(key)
        i = _skip_whitespace(i)
        if i >= len(text) or text[i] != ":":
            return []
        i = _skip_whitespace(i + 1)
        if key == "results":
            if i >= len(text) or text[i] != "[":
                return []
            array_start = i
            break
        try:
            value, i = decoder.raw_decode(text, i)
        except (json.JSONDecodeError, RecursionError, ValueError):
            return []
        if key == "score" and _finite_score(value) is None:
            return []
        i = _skip_whitespace(i)
        if i >= len(text) or text[i] != ",":
            return []
        i += 1

    if array_start is None:
        return []

    results: list[dict[str, Any]] = []
    i = array_start + 1
    while True:
        i = _skip_whitespace(i)
        if i >= len(text):
            return results
        if text[i] != "{":
            return []
        try:
            entry, i = decoder.raw_decode(text, i)
        except (json.JSONDecodeError, RecursionError, ValueError):
            return results if _is_append_only_json_object_prefix(text[i:]) else []
        if not isinstance(entry, dict):
            return []
        results.append(entry)
        i = _skip_whitespace(i)
        if i >= len(text):
            return results
        # Salvage is only for an array truncated before its closing bracket.
        # A closed results array with a malformed outer object is not partial
        # per-entry output and must take the structured-error path.
        if text[i] == "]":
            return []
        if text[i] != ",":
            return []
        i = _skip_whitespace(i + 1)
        if i >= len(text):
            return results
        if text[i] == "]":
            return []


STRUCTURED_JUDGE_MAX_TOKENS = 4096

_JUDGE_RETRY_REMINDER = (
    "\n\nIMPORTANT: Your previous reply could not be parsed or validated. Respond with ONLY the "
    "minified JSON object on a single line -- no markdown fences, no prose, and keep explanations brief."
)


def _call_validated_json_judge(
    prompt: str,
    validate: Any,
    call: Any,
    extract: Any,
    **call_kwargs: Any,
) -> tuple[Any, str | None, dict[str, Any]]:
    call_kwargs.setdefault("max_tokens", STRUCTURED_JUDGE_MAX_TOKENS)

    def invoke(call_prompt: str) -> tuple[Any, str | None, dict[str, Any], str | None]:
        content, error, *metadata = call(call_prompt, **call_kwargs)
        provenance = metadata[0] if metadata and isinstance(metadata[0], dict) else {}
        parsed = extract(content) if content else None
        validation_error = validate(parsed) if not error else None
        return parsed, error, provenance, validation_error

    parsed, error, provenance, validation_error = invoke(prompt)
    if error:
        return None, f"LLM judge error: {error}", provenance
    if validation_error is None:
        return parsed, None, provenance

    parsed, error, provenance, validation_error = invoke(prompt + _JUDGE_RETRY_REMINDER)
    if error:
        return None, f"LLM judge retry error: {error}", provenance
    if validation_error is not None:
        return None, f"{validation_error} after retry", provenance
    return parsed, None, provenance


# ---------------------------------------------------------------------------
# Accuracy judge (5-criterion)
# ---------------------------------------------------------------------------

ACCURACY_PROMPT = """You are an expert evaluator for AI agent responses. Evaluate by checking \
each criterion below against the expected answer. For each, answer YES or NO.

1. SKILL_IDENTIFIED: Does the response reference or use the correct skill for the task?
2. ACTION_CORRECT: Does the response describe or execute the correct actions/scripts?
3. FACTUALLY_ACCURATE: Are the factual claims consistent with the expected answer?
4. TASK_ADDRESSED: Does the response directly address the user's request?
5. ACTIONABLE: Does the response provide actionable information (not just acknowledgment)?

For each criterion write: YES or NO with a brief reason.
Then compute score = count(YES) / 5.
Be lenient on exact wording but strict on factual correctness.

Respond with ONLY a JSON object:
{{"criteria": {{"SKILL_IDENTIFIED": true/false, "ACTION_CORRECT": true/false, \
"FACTUALLY_ACCURATE": true/false, "TASK_ADDRESSED": true/false, "ACTIONABLE": true/false}}, \
"score": <float>, "reason": "<brief summary>"}}

USER QUESTION:
{question}

EXPECTED ANSWER:
{ground_truth}

SELECTED EVIDENCE (final response + produced artifacts; low-relevance steps may be omitted):
{agent_text}"""

_ACCURACY_CRITERIA_KEYS = frozenset(
    {
        "SKILL_IDENTIFIED",
        "ACTION_CORRECT",
        "FACTUALLY_ACCURATE",
        "TASK_ADDRESSED",
        "ACTIONABLE",
    }
)


def _valid_accuracy_criteria(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.keys() == _ACCURACY_CRITERIA_KEYS
        and all(isinstance(item, bool) for item in value.values())
    )


def _accuracy_payload_error(parsed: Any) -> str | None:
    if not isinstance(parsed, dict):
        return "Judge response was not a valid JSON object"
    if "reason" in parsed and not isinstance(parsed["reason"], str):
        return "Judge response contained an invalid accuracy reason"
    criteria = parsed.get("criteria")
    criteria_valid = _valid_accuracy_criteria(criteria)
    if "criteria" in parsed and not criteria_valid:
        return "Judge response contained invalid accuracy criteria"
    if _finite_score(parsed.get("score")) is None and not criteria_valid:
        return "Judge response contained no valid accuracy score or complete criteria"
    return None


def judge_accuracy(
    question: str,
    ground_truth: str,
    agent_text: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the 5-criterion accuracy judge. Returns ``{"score": float, "reason": str, ...}``."""
    if not ground_truth:
        return {"score": 1.0, "reason": "No ground_truth -- skipped"}

    prompt = ACCURACY_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        agent_text=agent_text,
    )

    parsed, error, _provenance = _call_validated_json_judge(
        prompt,
        _accuracy_payload_error,
        call_public_llm,
        _extract_json,
        **kwargs,
    )
    if error:
        return _judge_error(error)

    assert isinstance(parsed, dict)
    criteria = parsed.get("criteria")
    criteria_valid = _valid_accuracy_criteria(criteria)

    score = _finite_score(parsed.get("score"))
    if score is None:
        assert criteria_valid
        yes_count = sum(1 for v in criteria.values() if v is True)
        score = yes_count / 5.0

    return {
        "score": round(score, 4),
        "reason": _bounded_judge_text(parsed.get("reason", "")),
        "criteria": criteria if criteria_valid else {},
    }


# ---------------------------------------------------------------------------
# Goal accuracy judge
# ---------------------------------------------------------------------------

GOAL_ACCURACY_PROMPT = """You are an evaluation judge. Determine whether an AI agent achieved \
the expected goal by analyzing the full conversation.

Step 1: What was the user's goal?
Step 2: What end state did the agent reach?
Step 3: Compare the end state to the expected outcome.

USER REQUEST:
{question}

EXPECTED OUTCOME (ground truth):
{ground_truth}

AGENT'S TOOL CALLS:
{tool_summary}

END-STATE EVIDENCE:
{agent_text}

Did the agent achieve the expected goal?

Respond with ONLY a JSON object:
{{"user_goal": "<inferred goal>", "end_state": "<what agent achieved>", \
"achieved": true/false, "score": 1.0 or 0.0, "reason": "<brief explanation>"}}"""


def _goal_payload_error(parsed: Any) -> str | None:
    if not isinstance(parsed, dict):
        return "Judge response was not a valid JSON object"
    for field in ("reason", "user_goal", "end_state"):
        if field in parsed and not isinstance(parsed[field], str):
            return f"Judge response contained an invalid {field} value"
    if not isinstance(parsed.get("achieved"), bool):
        return "Judge response contained an invalid achieved value"
    if "score" in parsed and _finite_score(parsed["score"]) is None:
        return "Judge response contained an invalid goal score"
    return None


def judge_goal_accuracy(
    question: str,
    ground_truth: str,
    agent_text: str,
    tool_summary: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the goal accuracy judge (two-step: infer goal, compare outcome)."""
    if not ground_truth:
        return {"score": 1.0, "reason": "No ground_truth -- skipped"}

    prompt = GOAL_ACCURACY_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        tool_summary=tool_summary,
        agent_text=agent_text,
    )

    parsed, error, _provenance = _call_validated_json_judge(
        prompt,
        _goal_payload_error,
        call_public_llm,
        _extract_json,
        **kwargs,
    )
    if error:
        return _judge_error(error)

    assert isinstance(parsed, dict)
    achieved = parsed.get("achieved")
    assert isinstance(achieved, bool)

    score = 1.0 if achieved else 0.0
    if "score" in parsed:
        score = _finite_score(parsed["score"])
        assert score is not None

    return {
        "score": score,
        "reason": _bounded_judge_text(parsed.get("reason", "")),
        "user_goal": _bounded_judge_text(parsed.get("user_goal", "")),
        "end_state": _bounded_judge_text(parsed.get("end_state", "")),
    }


# ---------------------------------------------------------------------------
# Behavior check judge
# ---------------------------------------------------------------------------

BEHAVIOR_CHECK_PROMPT = """You are evaluating whether an AI agent followed expected behaviors \
during a task. Analyze the full conversation and determine if each expected behavior was \
observed.

CONVERSATION:
{conversation}

EXPECTED BEHAVIORS:
{behaviors}

For each behavior, respond YES (observed) or NO (not observed) with a brief reason.

Respond with ONLY a JSON object:
{{"results": [{{"step": 1, "passed": true/false, "reason": "..."}}, ...], \
"score": <float between 0.0 and 1.0>, "summary": "<brief summary>"}}"""


def _compact_behavior_conversation(conversation_text: str, limit: int = 8000) -> str:
    """Keep both setup context and late outcome evidence in behavior prompts."""
    if len(conversation_text) <= limit:
        return conversation_text

    marker = "\n...[middle truncated for behavior check]...\n"
    if limit <= len(marker):
        return conversation_text[:limit]

    head = max(1, (limit - len(marker)) * 2 // 3)
    tail = max(1, limit - len(marker) - head)
    return f"{conversation_text[:head]}{marker}{conversation_text[-tail:]}"


# Reasoning judges (e.g. openai/openai/gpt-5*) spend completion budget on hidden
# reasoning tokens before emitting the per-behavior results array; the old 1024
# cap was observed live to truncate behavior_check output to EMPTY content
# (finish_reason="length", reasoning_tokens=1024).
BEHAVIOR_JUDGE_MAX_TOKENS = STRUCTURED_JUDGE_MAX_TOKENS

_BEHAVIOR_RETRY_REMINDER = (
    "\n\nIMPORTANT: Your previous reply could not be parsed. Respond with ONLY the "
    "minified JSON object on a single line -- no markdown fences, no prose, and "
    'keep every "reason" under 15 words.'
)


def judge_behavior_check(
    conversation_text: str,
    expected_behaviors: list[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the behavior check LLM judge."""
    if not expected_behaviors:
        return {"score": 1.0, "reason": "No expected_behavior defined", "results": []}

    behaviors_text = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(expected_behaviors))

    prompt = BEHAVIOR_CHECK_PROMPT.format(
        conversation=_compact_behavior_conversation(conversation_text),
        behaviors=behaviors_text,
    )
    kwargs.setdefault("max_tokens", BEHAVIOR_JUDGE_MAX_TOKENS)

    content, error = call_public_llm(prompt, **kwargs)
    if error:
        return _judge_error(f"LLM judge error: {error}", results=[])

    def _parse_judge_object(text: str) -> dict[str, Any] | list[Any] | None:
        return _extract_json(text) if text else None

    parsed = _parse_judge_object(content)
    score = _behavior_payload_score(parsed, len(expected_behaviors))
    attempts = [(content or "", parsed)]
    retry_error = None
    if score is None:
        # One retry max, with an explicit machine-readable-output reminder.
        retry_content, retry_error = call_public_llm(prompt + _BEHAVIOR_RETRY_REMINDER, **kwargs)
        if not retry_error:
            parsed = _parse_judge_object(retry_content)
            attempts.append((retry_content or "", parsed))
            score = _behavior_payload_score(parsed, len(expected_behaviors))

    if score is None:
        # Salvage complete entries from a truncated results array (newest first).
        for text, extracted in reversed(attempts):
            if extracted is not None:
                continue
            salvaged = _salvage_behavior_results(text)
            if salvaged:
                candidate = {
                    "results": salvaged,
                    "summary": (
                        f"Salvaged {len(salvaged)}/{len(expected_behaviors)} behavior "
                        "results from truncated judge response"
                    ),
                }
                candidate_score = _behavior_payload_score(
                    candidate,
                    len(expected_behaviors),
                    allow_partial=True,
                )
                if candidate_score is not None:
                    parsed = candidate
                    score = candidate_score
                    break

    if score is None:
        if retry_error:
            return _judge_error(f"LLM judge retry error: {retry_error}", results=[])
        return _judge_error("Judge response was unparseable or invalid after retry", results=[])

    assert isinstance(parsed, dict)
    results = parsed["results"]

    return {
        "score": round(score, 4),
        "reason": parsed.get("summary", ""),
        "results": results,
    }


def _behavior_payload_score(
    parsed: dict[str, Any] | list[Any] | None,
    expected_count: int,
    *,
    allow_partial: bool = False,
) -> float | None:
    if not isinstance(parsed, dict):
        return None
    results = parsed.get("results")
    if not isinstance(results, list):
        return None
    if any(not isinstance(result, dict) or not isinstance(result.get("passed"), bool) for result in results):
        return None
    if allow_partial:
        if not results or len(results) > expected_count:
            return None
    elif len(results) != expected_count:
        return None
    if "score" in parsed and _finite_score(parsed["score"]) is None:
        return None
    denominator = expected_count if allow_partial else len(results)
    if denominator <= 0:
        return None
    return sum(1 for result in results if result["passed"]) / denominator
