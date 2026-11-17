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

from skillevaluator.inference.json_span import (
    MAX_JSON_NESTING,
    STRICT_JSON_DECODE_HOOKS,
    balanced_json_container_end,
    json_nesting_within_limit,
)
from skillevaluator.inference.types import EmptyLLMResponseError, LLMClientTruncatedError
from skillevaluator.provider_config import (
    MAX_COMPLETION_TOKENS,
    ProviderConfig,
    effective_reasoning_effort,
)
from skillevaluator.tier3.eval_core.atif_helpers import (
    _BEHAVIOR_EVIDENCE_GAP_HINT,
    _SECTION_COMPACT_TOOL_HISTORY,
    _SECTION_FINAL_RESPONSE,
    _SECTION_USER_REQUEST,
    _behavior_check_budget,
    _behavior_final_response_limit,
    _truncate_for_behavior,
    behavior_evidence_gap,
)
from skillevaluator.tier3.eval_core.character_facts import append_character_facts
from skillevaluator.tier3.eval_core.optional_reference import normalize_expected_behaviors

logger = logging.getLogger(__name__)

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
NVIDIA_BUILD_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
# Sent only where effective_reasoning_effort allows it (GPT-6, Claude 5.5+ on the anthropic provider).
JUDGE_REASONING_EFFORT = "medium"

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


NOT_APPLICABLE_STATUS = "not_applicable"
_NO_GROUND_TRUTH_REASON = "N/A: no ground_truth defined for this eval case"
_NO_EXPECTED_BEHAVIOR_REASON = "N/A: no expected_behavior defined for this eval case"


def _judge_not_applicable(reason: str, **metadata: Any) -> dict[str, Any]:
    """Return a scoreless result for a judge that has nothing to judge against."""
    return {**metadata, "score": None, "status": NOT_APPLICABLE_STATUS, "reason": reason}


def _has_judge_reference(value: Any) -> bool:
    """Return whether a ground_truth / expected_behavior value gives a judge something to check."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(_has_judge_reference(item) for item in value)
    if isinstance(value, (bool, int, float)):
        # 0, 0.0 and False are real reference answers, not a missing one.
        return True
    return bool(value)


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
    response_schema: dict[str, Any] | None = None,
    schema_name: str = "judge_response",
) -> dict[str, Any]:
    """Return the OpenAI-compatible judge request body ``LLMClient`` would send.

    Built from the shared client's request helpers, so the drift test holds the
    Harbor template's in-container copy to what the client sends.
    """
    from skillevaluator.inference.client import (
        _build_openai_response_format,
        _temperature_kwargs,
        _token_limit_kwargs,
    )

    resolved_provider = _provider() if provider is None else provider
    resolved_request_url = _resolve_url(resolved_provider) if request_url is None else request_url
    config = ProviderConfig(
        provider=resolved_provider,
        model=model,
        api_key=None,
        base_url=resolved_request_url.rstrip("/").removesuffix("/chat/completions"),
        litellm_model=f"openai/{model}",
    )
    payload: dict[str, Any] = {
        "model": model,
        **_token_limit_kwargs(config, max_tokens),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        **_temperature_kwargs(model, temperature),
    }
    if effort := effective_reasoning_effort(resolved_provider, model, JUDGE_REASONING_EFFORT):
        payload["reasoning_effort"] = effort
    if response_schema is not None:
        payload["response_format"] = _build_openai_response_format(response_schema, schema_name)
    return payload


def call_public_llm(
    prompt: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout: int = 60,
    allow_model_fallback: bool = True,
    response_schema: dict[str, Any] | None = None,
    schema_name: str = "judge_response",
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
            reasoning_effort=JUDGE_REASONING_EFFORT,
        )
        return (
            client.completions(
                "You are a precise evaluation judge.",
                prompt,
                response_schema=response_schema,
                schema_name=schema_name,
            ),
            None,
        )
    except EmptyLLMResponseError:
        return "", None
    except LLMClientTruncatedError as exc:
        # Hand back the partial reply so judges can salvage its complete entries.
        return exc.content, None
    except Exception as exc:
        detail = f"Public provider call failed: {exc}"
        return None, _redact_configured_credentials(detail, (api_key,))


_JSON_WHITESPACE = " \t\r\n"
_MAX_JSON_TEXT_CHARS = 100_000
_MAX_JSON_NESTING = MAX_JSON_NESTING
_JSON_NUMBER_PREFIX_RE = re.compile(r"-?(?:(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]*)?|(?:0|[1-9][0-9]*)\.)?")
_json_nesting_within_limit = json_nesting_within_limit


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
        end = balanced_json_container_end(text, index)
        if end is None:
            return documents[0] if documents else None
        candidate = text[index:end]
        try:
            parsed = json.loads(
                candidate,
                **STRICT_JSON_DECODE_HOOKS,
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
        **STRICT_JSON_DECODE_HOOKS,
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
        **STRICT_JSON_DECODE_HOOKS,
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


STRUCTURED_JUDGE_MAX_TOKENS = MAX_COMPLETION_TOKENS
_JUDGE_PROBLEM_CHARS = 300


def _judge_retry_feedback(problem: str) -> str:
    """Retry instruction that names what was wrong with the previous reply."""
    return (
        f"\n\nIMPORTANT: Your previous reply could not be parsed or validated: {problem[:_JUDGE_PROBLEM_CHARS]}. "
        "Respond with ONLY the minified JSON object on a single line in the shape described above -- "
        "no markdown fences, no prose, no score field, and keep explanations brief."
    )


def _call_validated_json_judge(
    prompt: str,
    validate: Any,
    call: Any,
    extract: Any,
    *,
    salvage: Any = None,
    **call_kwargs: Any,
) -> tuple[Any, str | None, dict[str, Any]]:
    """Invoke a JSON judge; on an invalid reply, retry once and name the problem.

    *validate* returns ``None`` for a usable reply or a short description of the
    problem, which the retry sends back to the judge. Provider errors are not
    retried here: the transport already retried them within its own budget.
    *salvage* may rebuild a usable reply from text that never parsed, such as a
    reply cut off by the output-token cap; it is tried only after both attempts.
    """
    call_kwargs.setdefault("max_tokens", STRUCTURED_JUDGE_MAX_TOKENS)
    unparsed_replies: list[str] = []

    def invoke(call_prompt: str) -> tuple[Any, str | None, dict[str, Any], str | None]:
        content, error, *metadata = call(call_prompt, **call_kwargs)
        provenance = metadata[0] if metadata and isinstance(metadata[0], dict) else {}
        parsed = extract(content) if content else None
        problem = None
        if not error:
            problem = "the reply was empty" if not content else validate(parsed)
            if content and parsed is None:
                unparsed_replies.append(content)
        return parsed, error, provenance, problem

    parsed, error, provenance, problem = invoke(prompt)
    if error:
        return None, f"LLM judge error: {error}", provenance
    if problem is None:
        return parsed, None, provenance

    parsed, error, provenance, problem = invoke(prompt + _judge_retry_feedback(problem))
    if not error and problem is None:
        return parsed, None, provenance
    if salvage is not None:
        # Newest first: only text that never parsed can be a reply cut off mid-object.
        for text in reversed(unparsed_replies):
            candidate = salvage(text)
            if candidate is not None:
                return candidate, None, provenance
    if error:
        return None, f"LLM judge retry error: {error}", provenance
    return None, f"Judge response was unparseable or invalid after retry: {problem}", provenance


# ---------------------------------------------------------------------------
# Accuracy judge (5-criterion)
# ---------------------------------------------------------------------------

# Accuracy grades the outcome only. Whether the agent found and used the skill is
# scored by skill_execution, so a correct answer from the no-skill baseline can
# reach 1.0 here.
_ACCURACY_CRITERIA = (
    ("ACTION_CORRECT", "Does the response describe or carry out the correct actions or commands for the task?"),
    ("FACTUALLY_ACCURATE", "Are the factual claims consistent with the expected answer?"),
    ("TASK_ADDRESSED", "Does the response directly address the user's request?"),
    ("ACTIONABLE", "Does the response give usable results or next steps, not just an acknowledgment?"),
    (
        "FORMAT_FOLLOWED",
        "If the USER QUESTION explicitly requires an output format (a template, prefix, fields, ordering, "
        "or units), does the response follow it? When the question requires no specific format, answer true "
        "if the response gives an answer and false if it gives none.",
    ),
)
_ACCURACY_DEDUCTION_SENTENCES = {
    "ACTION_CORRECT": "The agent did not take the action the expected answer calls for",
    "FACTUALLY_ACCURATE": "The answer's facts do not match the expected answer",
    "TASK_ADDRESSED": "The answer does not address what the user asked",
    "ACTIONABLE": "The answer does not give the user a result they can act on",
    "FORMAT_FOLLOWED": "The answer does not follow the output format the question requires",
}
_ACCURACY_CRITERIA_KEYS = frozenset(key for key, _question in _ACCURACY_CRITERIA)

ACCURACY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "criteria": {
            "type": "object",
            "properties": {key: {"type": "boolean"} for key, _question in _ACCURACY_CRITERIA},
            "required": [key for key, _question in _ACCURACY_CRITERIA],
            "additionalProperties": False,
        },
        "reason": {"type": "string"},
    },
    "required": ["criteria", "reason"],
    "additionalProperties": False,
}

GOAL_ACCURACY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "user_goal": {"type": "string"},
        "end_state": {"type": "string"},
        "achieved": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["user_goal", "end_state", "achieved", "reason"],
    "additionalProperties": False,
}

BEHAVIOR_CHECK_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer"},
                    "passed": {"type": ["boolean", "null"]},
                    "reason": {"type": "string"},
                },
                "required": ["step", "passed", "reason"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["results", "summary"],
    "additionalProperties": False,
}

ACCURACY_PROMPT = (
    """You are an expert evaluator for AI agent responses. Check the response against \
the expected answer on each criterion below and decide true (satisfied) or false (not satisfied) for each.

"""
    + "\n".join(f"{index}. {key}: {criterion}" for index, (key, criterion) in enumerate(_ACCURACY_CRITERIA, 1))
    + """

Judge the outcome only. Do not reward or penalize the response for naming, or not naming, \
a tool or skill: that is scored separately.
If the evidence holds no answer to the question (for example, the agent stopped, timed out or \
refused), every criterion is false, unless the expected answer is itself a refusal.
Be lenient on wording, but strict on factual correctness and on any output format the user \
question explicitly requires.

For FORMAT_FOLLOWED, judge the entire final response literally against only the format explicitly \
required by the USER QUESTION, independently of factual correctness.
When a FINAL RESPONSE section is supplied, its contents are the final answer; other section \
headings, tool results, history, and artifacts are context, not part of that answer.
Inspect every line of the final response, including text after an initially valid answer. \
Line breaks do not end the final response. Do not mentally remove labels or Markdown fence characters \
when checking the stated format.
Do not extract a compliant answer from surrounding noncompliant text or substitute a correct \
value from tool output or the expected answer.
If only a number is required, extra prose, a label, or a code fence violates that requirement.
Required prefixes, units, or prose are valid when they match the stated format.
Wording leniency does not relax explicit format constraints.

Respond with ONLY one JSON object with a "criteria" object holding exactly these five keys, each \
true or false, and a one-sentence "reason". Do not include a score: it is computed from the criteria.
Example of the shape (the values are only an illustration):
{{"criteria": {{"ACTION_CORRECT": true, "FACTUALLY_ACCURATE": false, "TASK_ADDRESSED": true, "ACTIONABLE": true, \
"FORMAT_FOLLOWED": true}}, "reason": "Ran the right command but reported the wrong version number."}}

USER QUESTION:
{question}

EXPECTED ANSWER:
{ground_truth}

SELECTED EVIDENCE (final response + produced artifacts; low-relevance steps may be omitted):
{agent_text}"""
)


def _accuracy_payload_error(parsed: Any) -> str | None:
    """Describe what is wrong with an accuracy reply, or return ``None`` when it is usable.

    The score is computed from the five verdicts, so a ``score`` field in the
    reply is ignored rather than trusted.
    """
    if not isinstance(parsed, dict):
        return "the reply did not contain one JSON object"
    criteria = parsed.get("criteria")
    if not isinstance(criteria, dict):
        return 'the reply has no "criteria" object'
    missing = sorted(_ACCURACY_CRITERIA_KEYS - criteria.keys())
    if missing:
        return f"criteria is missing {', '.join(missing)}"
    unexpected = sorted(str(key)[:40] for key in criteria.keys() - _ACCURACY_CRITERIA_KEYS)
    if unexpected:
        return f"criteria has unexpected keys {', '.join(unexpected[:5])}"
    not_boolean = sorted(key for key, value in criteria.items() if not isinstance(value, bool))
    if not_boolean:
        return f"criteria values must be true or false, not {', '.join(not_boolean)}"
    if "reason" in parsed and not isinstance(parsed["reason"], str):
        return '"reason" must be a string'
    return None


def judge_accuracy(
    question: str,
    ground_truth: str,
    agent_text: str,
    *,
    skill_identification_applicable: bool | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Judge accuracy from five outcome verdicts; the score is computed here.

    *skill_identification_applicable* is deprecated and ignored. Accuracy no
    longer has a skill criterion, so every case and both arms use the same five
    criteria. The keyword is still accepted so existing callers keep working.
    """
    del skill_identification_applicable
    if not _has_judge_reference(ground_truth):
        return _judge_not_applicable(_NO_GROUND_TRUTH_REASON)

    prompt = ACCURACY_PROMPT.format(question=question, ground_truth=ground_truth, agent_text=agent_text)
    prompt = append_character_facts(prompt, question=question, ground_truth=ground_truth, agent_text=agent_text)
    kwargs.setdefault("response_schema", ACCURACY_JSON_SCHEMA)
    kwargs.setdefault("schema_name", "accuracy_judgment")

    parsed, error, _provenance = _call_validated_json_judge(
        prompt,
        _accuracy_payload_error,
        call_public_llm,
        _extract_json,
        **kwargs,
    )
    if error:
        return _judge_error(error)

    criteria = {key: parsed["criteria"][key] for key, _question in _ACCURACY_CRITERIA}
    return {
        "score": round(sum(criteria.values()) / len(criteria), 4),
        "reason": _bounded_judge_text(parsed.get("reason", "")),
        "criteria": criteria,
    }


# ---------------------------------------------------------------------------
# Goal accuracy judge
# ---------------------------------------------------------------------------

GOAL_ACCURACY_PROMPT = """You are an evaluation judge. Decide whether an AI agent achieved the expected goal.

Step 1: State the user's goal.
Step 2: State the end state the agent reached.
Step 3: Decide whether that end state achieves the expected outcome.

USER REQUEST:
{question}

EXPECTED OUTCOME:
{ground_truth}
{tool_section}
END-STATE EVIDENCE:
{agent_text}

Respond with ONLY one JSON object with the keys "user_goal", "end_state", "achieved" (true or \
false) and a one-sentence "reason". Do not include a score: it is 1.0 when achieved and 0.0 otherwise.
Example of the shape (the values are only an illustration):
{{"user_goal": "List the open release blockers", "end_state": "Listed two of the three blockers", \
"achieved": false, "reason": "One blocker from the expected outcome is missing."}}"""


def _goal_payload_error(parsed: Any) -> str | None:
    """Describe what is wrong with a goal reply, or return ``None`` when it is usable.

    Goal accuracy is binary, like RAGAS AgentGoalAccuracyWithReference: the
    score is 1.0 when ``achieved`` is true and 0.0 otherwise, so a ``score``
    field in the reply is ignored.
    """
    if not isinstance(parsed, dict):
        return "the reply did not contain one JSON object"
    if not isinstance(parsed.get("achieved"), bool):
        return '"achieved" must be true or false'
    for field in ("reason", "user_goal", "end_state"):
        if field in parsed and not isinstance(parsed[field], str):
            return f'"{field}" must be a string'
    return None


def judge_goal_accuracy(
    question: str,
    ground_truth: str,
    agent_text: str,
    tool_summary: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the goal accuracy judge (two-step: infer goal, compare outcome); the score is binary."""
    if not _has_judge_reference(ground_truth):
        return _judge_not_applicable(_NO_GROUND_TRUTH_REASON)

    tool_section = f"\nAGENT'S TOOL CALLS:\n{tool_summary}\n" if str(tool_summary or "").strip() else ""
    prompt = GOAL_ACCURACY_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        tool_section=tool_section,
        agent_text=agent_text,
    )
    prompt = append_character_facts(
        prompt, question=question, ground_truth=ground_truth, tool_summary=tool_summary, agent_text=agent_text
    )
    kwargs.setdefault("response_schema", GOAL_ACCURACY_JSON_SCHEMA)
    kwargs.setdefault("schema_name", "goal_accuracy_judgment")

    parsed, error, _provenance = _call_validated_json_judge(
        prompt,
        _goal_payload_error,
        call_public_llm,
        _extract_json,
        **kwargs,
    )
    if error:
        return _judge_error(error)

    achieved = parsed["achieved"]
    return {
        "score": 1.0 if achieved else 0.0,
        "reason": _bounded_judge_text(parsed.get("reason", "")),
        "achieved": achieved,
        "user_goal": _bounded_judge_text(parsed.get("user_goal", "")),
        "end_state": _bounded_judge_text(parsed.get("end_state", "")),
    }


# ---------------------------------------------------------------------------
# Behavior check judge
# ---------------------------------------------------------------------------

BEHAVIOR_CHECK_PROMPT = """You are evaluating whether an AI agent showed each expected behavior during a task. \
Read the conversation and decide, for each numbered behavior, whether it was observed.

CONVERSATION:
{conversation}

EXPECTED BEHAVIORS:
{behaviors}

For each numbered behavior, check only what that behavior requires.
A format-only behavior does not require the answer's value to be factually correct: a wrong value \
can still satisfy its format. If a behavior also requires factual correctness, evaluate that requirement too.
For a behavior about final-response format, judge the entire final response literally against only \
the stated format requirement, independently of factual correctness.
When a FINAL RESPONSE section is supplied, its contents are the final answer; other section \
headings, tool results, history, and artifacts are context, not part of that answer.
Inspect every line of the final response, including text after an initially valid answer. \
Line breaks do not end the final response. Do not mentally remove labels or Markdown fence characters \
when checking the stated format.
Do not extract a compliant answer from surrounding noncompliant text or substitute a correct \
value from tool output or the expected answer.
If only a number is required, extra prose, a label, or a code fence violates that requirement.
Required prefixes, units, or prose are valid when they match the stated format.
When the complete final response establishes a format violation, mark that behavior false; \
keep null when missing or incomplete evidence prevents checking it.
Before returning, ensure each passed value agrees with its reason: a reason establishing a violation \
requires false, not true; an evidence gap that prevents checking requires null.

Respond with ONLY one JSON object. "results" must hold exactly {expected_count} entries, one per \
expected behavior: "step" is the behavior's number (each number from 1 to {expected_count} exactly \
once), "passed" is true (observed), false (evidence supports a violation or absence), or null \
when missing or incomplete evidence prevents checking the behavior. Missing artifact contents, \
unavailable tool output, or a truncated trace require null, NOT false, for checks that depend on \
that evidence. A missing observation alone is not proof the behavior failed. Explain that evidence gap \
in "reason" (under 15 words). An unrelated task \
can be a valid negative control: evaluate its stated behavior without requiring skill use. Add a \
one-sentence "summary". Do not include a score: it is computed from the results.
Example of the shape for this task's {expected_count} behaviors (the values are only an illustration):
{example}"""


def _slice_with_middle_marker(
    text: str,
    budget: int,
    marker: str,
    *,
    max_head: int | None = None,
    fallback_tail: bool = False,
) -> str:
    """Compact *text* to fit *budget* by replacing the middle with *marker*."""
    if budget <= 0 or not text:
        return ""
    if len(text) <= budget:
        return text
    if budget <= len(marker) + 2:
        return text[-budget:] if fallback_tail else text[:budget]
    avail = budget - len(marker)
    head = max(1, avail // 2 if max_head is None else min(max_head, avail // 2))
    tail = max(1, avail - head)
    return f"{text[:head]}{marker}{text[-tail:]}"


def _compact_behavior_conversation(conversation_text: str, limit: int | None = None) -> str:
    """Keep both setup context and late outcome evidence in behavior prompts."""
    if limit is None:
        limit = _behavior_check_budget()
    if len(conversation_text) <= limit:
        return conversation_text

    marker = "\n...[middle truncated for behavior check]...\n"
    if limit <= len(marker) + 1:
        return conversation_text[:limit]

    final_limit = _behavior_final_response_limit()
    final_header = f"{_SECTION_FINAL_RESPONSE}\n"
    final_idx = -1
    if conversation_text.startswith(final_header):
        final_idx = 0
    else:
        pos = conversation_text.find(f"\n\n{final_header}")
        if pos != -1:
            final_idx = pos + 2

    if final_idx != -1:
        final_end = len(conversation_text)
        for next_hdr in (f"\n\n{_SECTION_USER_REQUEST}\n", f"\n\n{_SECTION_COMPACT_TOOL_HISTORY}\n"):
            pos = conversation_text.find(next_hdr, final_idx)
            if pos != -1 and pos < final_end:
                final_end = pos
        prefix = conversation_text[:final_idx]
        final_sec = conversation_text[final_idx:final_end]
        suffix = conversation_text[final_end:]

        reserved_other = min(1600, max(0, limit - final_limit), limit // 2)
        max_final = min(final_limit, max(1, limit - reserved_other))
        if len(final_sec) > max_final:
            final_body = final_sec[len(final_header) :]
            body_limit = max(1, max_final - len(final_header))
            final_sec = f"{final_header}{_truncate_for_behavior(final_body, body_limit)}"[:max_final]

        rem = limit - len(final_sec)
        if rem <= 0:
            return final_sec[:limit]
        if len(prefix) + len(suffix) <= rem:
            return f"{prefix}{final_sec}{suffix}"
        if not suffix:
            pre_comp = _slice_with_middle_marker(prefix, rem, marker)
            return f"{pre_comp}{final_sec}"[:limit]
        if len(prefix) <= rem // 2:
            suf_comp = _slice_with_middle_marker(suffix, rem - len(prefix), marker, max_head=800, fallback_tail=True)
            return f"{prefix}{final_sec}{suf_comp}"[:limit]
        pre_budget = max(1, min(len(prefix), rem // 2))
        suf_budget = max(0, rem - pre_budget)
        pre_comp = _slice_with_middle_marker(prefix, pre_budget, marker)
        suf_comp = _slice_with_middle_marker(suffix, suf_budget, marker, max_head=800, fallback_tail=True)
        return f"{pre_comp}{final_sec}{suf_comp}"[:limit]

    available = limit - len(marker)
    reserved_head = min(1600, available // 2)
    tail = max(1, available // 3, min(final_limit, max(1, available - max(1, reserved_head))))
    head = max(1, available - tail)
    return f"{conversation_text[:head]}{marker}{conversation_text[-tail:]}"


# Reasoning judges (e.g. openai/openai/gpt-5*) spend completion budget on hidden
# reasoning tokens before emitting the per-behavior results array; the old 1024
# cap was observed live to truncate behavior_check output to EMPTY content
# (finish_reason="length", reasoning_tokens=1024).
BEHAVIOR_JUDGE_MAX_TOKENS = STRUCTURED_JUDGE_MAX_TOKENS


def _behavior_prompt_example(expected_count: int) -> str:
    """One entry per expected behavior, so the example is a valid reply if a judge copies it."""
    return json.dumps(
        {
            "results": [
                {
                    "step": step,
                    "passed": None if step % 3 == 0 else step % 3 == 1,
                    "reason": (
                        "Required tool output was not captured."
                        if step % 3 == 0
                        else "Seen in the transcript."
                        if step % 3 == 1
                        else "Observed output violates the requirement."
                    ),
                }
                for step in range(1, expected_count + 1)
            ],
            "summary": "A short summary of what was and was not observed.",
        }
    )


def judge_behavior_check(
    conversation_text: str,
    expected_behaviors: list[str],
    *,
    evidence_omitted: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the behavior check LLM judge; the score is the share of behaviors observed.

    *evidence_omitted* is the behavior_check bundle's omission accounting. When
    it reports that required evidence did not reach the judge, no model is
    called and an explicit evaluation error is returned with no score: grading
    a shortened final answer scores the evidence loss, not the agent.
    """
    expected_behaviors = normalize_expected_behaviors(expected_behaviors)
    if not _has_judge_reference(expected_behaviors):
        return _judge_not_applicable(_NO_EXPECTED_BEHAVIOR_REASON, results=[])

    expected_count = len(expected_behaviors)
    behaviors_text = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(expected_behaviors))

    gap = behavior_evidence_gap(evidence_omitted)
    if gap:
        return _judge_error(gap, results=[], expectations=_behavior_expectations(expected_behaviors, [], gap))

    conversation = _compact_behavior_conversation(conversation_text)
    if evidence_omitted is not None and conversation != conversation_text:
        # The bundle is built to fit this budget, so a second cut here would
        # drop evidence nothing has accounted for. Refuse rather than score it.
        second_cut = (
            "behavior_check evidence was shortened again inside the judge, past the omission accounting. "
            f"{_BEHAVIOR_EVIDENCE_GAP_HINT}"
        )
        return _judge_error(
            second_cut, results=[], expectations=_behavior_expectations(expected_behaviors, [], second_cut)
        )

    prompt = BEHAVIOR_CHECK_PROMPT.format(
        conversation=conversation,
        behaviors=behaviors_text,
        expected_count=expected_count,
        example=_behavior_prompt_example(expected_count),
    )
    prompt = append_character_facts(prompt, conversation=conversation, expected_behaviors=behaviors_text)
    kwargs.setdefault("max_tokens", BEHAVIOR_JUDGE_MAX_TOKENS)
    kwargs.setdefault("response_schema", BEHAVIOR_CHECK_JSON_SCHEMA)
    kwargs.setdefault("schema_name", "behavior_check_judgment")

    def salvage(text: str) -> dict[str, Any] | None:
        # Complete entries from a reply cut off before its results array closed.
        salvaged = _salvage_behavior_results(text)
        candidate = {
            "results": salvaged,
            "summary": f"Salvaged {len(salvaged)}/{expected_count} behavior results from truncated judge response",
        }
        if not salvaged or _behavior_payload_error(candidate, expected_count, allow_partial=True):
            return None
        return candidate

    try:
        parsed, error, _provenance = _call_validated_json_judge(
            prompt,
            lambda payload: _behavior_payload_error(payload, expected_count),
            call_public_llm,
            _extract_json,
            salvage=salvage,
            **kwargs,
        )
    except Exception as exc:
        error = f"Required behavior_check judge raised {type(exc).__name__}: {exc}"
    if error:
        return _judge_error(error, results=[], expectations=_behavior_expectations(expected_behaviors, [], error))

    # Keep the conservative score denominator without calling missing verdicts failures.
    results = sorted(parsed["results"], key=lambda result: result["step"])
    score = sum(1 for result in results if result["passed"]) / expected_count
    return {
        "score": round(score, 4),
        "reason": parsed.get("summary", ""),
        "results": results,
        "expectations": _behavior_expectations(expected_behaviors, results),
    }


def _behavior_expectations(
    expected_behaviors: list[str], results: list[dict[str, Any]], error: str = ""
) -> list[dict[str, Any]]:
    by_step = {result["step"]: result for result in results}
    return [
        {
            "step": step,
            "requirement": requirement,
            "status": ("met" if by_step[step]["passed"] else "unmet")
            if step in by_step and type(by_step[step]["passed"]) is bool
            else "not_checked",
            "reason": by_step[step].get("reason", "")
            if step in by_step
            else _redact_configured_credentials(error) or "No judgment returned for this expectation",
        }
        for step, requirement in enumerate(expected_behaviors, 1)
    ]


def _behavior_payload_error(parsed: Any, expected_count: int, *, allow_partial: bool = False) -> str | None:
    """Describe what is wrong with a behavior reply, or return ``None`` when it is usable.

    Each expected behavior is judged exactly once, identified by its 1-based
    ``step`` number, with a boolean ``passed`` or an explicit null abstention. *allow_partial* accepts a
    salvaged prefix that judges each of its steps once. The score is computed
    from the verdicts, so a ``score`` field in the reply is ignored.
    """
    if not isinstance(parsed, dict):
        return "the reply did not contain one JSON object"
    results = parsed.get("results")
    if not isinstance(results, list):
        return 'the reply has no "results" list'
    if allow_partial:
        if not results or len(results) > expected_count:
            return f'"results" has {len(results)} entries; it needs 1 to {expected_count}'
    elif len(results) != expected_count:
        return f'"results" has {len(results)} entries; it needs exactly {expected_count}, one per behavior'
    if any(not isinstance(result, dict) for result in results):
        return 'every "results" entry must be an object'
    if any(
        "passed" not in result or (result["passed"] is not None and type(result["passed"]) is not bool)
        for result in results
    ):
        return 'every "results" entry needs "passed" set to true, false, or null'
    if any(
        result["passed"] is None and (not isinstance(result.get("reason"), str) or not result["reason"].strip())
        for result in results
    ):
        return 'unchecked entries need a nonempty "reason" explaining the missing evidence'
    steps = [result.get("step") for result in results]
    if any(isinstance(step, bool) or not isinstance(step, int) for step in steps):
        return 'every "results" entry needs an integer "step" number'
    expected_steps = set(range(1, expected_count + 1))
    duplicated = sorted({step for step in steps if steps.count(step) > 1})
    unexpected = sorted(set(steps) - expected_steps)[:10]
    missing = [] if allow_partial else sorted(expected_steps - set(steps))[:10]
    if duplicated or missing or unexpected:
        parts = []
        if duplicated:
            parts.append(f"steps judged more than once: {duplicated[:10]}")
        if missing:
            parts.append(f"steps not judged: {missing}")
        if unexpected:
            parts.append(f"steps outside 1..{expected_count}: {unexpected}")
        return "; ".join(parts)
    if "summary" in parsed and not isinstance(parsed["summary"], str):
        return '"summary" must be a string'
    return None
