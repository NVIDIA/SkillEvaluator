# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LLM judge prompt builders and public-provider caller.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
NVIDIA_BUILD_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_JUDGE_MODEL = "gpt-5.4-mini"

_ERROR_REDACTION_MARKER = "[REDACTED]"
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


def _supports_custom_temperature(model: str) -> bool:
    """Return false for chat models that only accept their default temperature."""
    lowered = str(model or "").lower()
    return not lowered.startswith("openai/openai/gpt-5")


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
        if str(model or "").casefold().startswith("gpt-5")
        and _is_native_openai_chat_url(resolved_provider, resolved_request_url)
        else "max_tokens"
    )
    payload: dict[str, Any] = {
        "model": model,
        token_key: max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if _supports_custom_temperature(model):
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
    except Exception as exc:
        detail = f"Public provider call failed: {exc}"
        return None, _redact_configured_credentials(detail, (api_key,))


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def _find_balanced_json(text: str) -> str | None:
    """Return the first balanced ``{...}`` block, honoring strings and escapes."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
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
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_json(text: str) -> dict[str, Any] | list[Any] | None:
    """Extract a JSON payload from LLM response text.

    Tolerates markdown fences (```json anywhere in the text), leading or
    trailing prose -- including prose that itself contains braces -- via
    first-balanced-brace extraction.  Top-level JSON arrays parse through
    unchanged (the suggestion generator in ``harbor.report`` relies on that);
    judge callers must dict-check the result themselves.
    """
    text = (text or "").strip()
    if not text:
        return None
    candidates = [text]
    if text.startswith("```"):
        candidates.append(text.split("\n", 1)[-1].rsplit("```", 1)[0].strip())
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    balanced = _find_balanced_json(text)
    if balanced:
        candidates.append(balanced)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    return None


def _salvage_behavior_results(text: str) -> list[dict[str, Any]]:
    """Recover complete per-behavior entries from a truncated ``results`` array.

    Reasoning judges that hit the output-token cap emit ``{"results": [...`` and
    stop mid-entry (``finish_reason="length"``); every fully-formed ``{...}``
    entry before the cut is still valid JSON and can be scored.
    """
    text = text or ""
    marker = text.find('"results"')
    if marker == -1:
        return []
    array_start = text.find("[", marker)
    if array_start == -1:
        return []
    results: list[dict[str, Any]] = []
    i = array_start + 1
    while i < len(text):
        ch = text[i]
        if ch == "{":
            block = _find_balanced_json(text[i:])
            if not block:
                break
            try:
                entry = json.loads(block)
            except (json.JSONDecodeError, ValueError):
                break
            if isinstance(entry, dict):
                results.append(entry)
            i += len(block)
        elif ch == "]":
            break
        else:
            i += 1
    return results


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

    content, error = call_public_llm(prompt, **kwargs)
    if error:
        return {"score": 0.0, "reason": f"LLM judge error: {error}"}

    parsed = _extract_json(content) if content else None
    if not parsed:
        yes_count = (content or "").upper().count("YES")
        score = min(yes_count / 5.0, 1.0)
        return {"score": round(score, 2), "reason": f"Parsed {yes_count}/5 YES from text"}

    score = parsed.get("score", 0.0)
    if isinstance(score, (int, float)):
        score = max(0.0, min(1.0, float(score)))
    else:
        criteria = parsed.get("criteria", {})
        yes_count = sum(1 for v in criteria.values() if v is True)
        score = yes_count / 5.0

    return {
        "score": round(score, 4),
        "reason": parsed.get("reason", ""),
        "criteria": parsed.get("criteria", {}),
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

    content, error = call_public_llm(prompt, **kwargs)
    if error:
        return {"score": 0.0, "reason": f"LLM judge error: {error}"}

    parsed = _extract_json(content) if content else None
    if not parsed:
        return {"score": 0.0, "reason": "Could not parse judge response"}

    score = 1.0 if parsed.get("achieved", False) else 0.0
    if "score" in parsed and isinstance(parsed["score"], (int, float)):
        score = max(0.0, min(1.0, float(parsed["score"])))

    return {
        "score": score,
        "reason": parsed.get("reason", ""),
        "user_goal": parsed.get("user_goal", ""),
        "end_state": parsed.get("end_state", ""),
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
BEHAVIOR_JUDGE_MAX_TOKENS = 4096

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
        return {"score": 0.0, "reason": f"LLM judge error: {error}", "results": []}

    def _parse_judge_object(text: str) -> dict[str, Any] | None:
        parsed = _extract_json(text) if text else None
        return parsed if isinstance(parsed, dict) else None

    parsed = _parse_judge_object(content)
    attempts = [content or ""]
    if not parsed:
        # One retry max, with an explicit machine-readable-output reminder.
        retry_content, retry_error = call_public_llm(prompt + _BEHAVIOR_RETRY_REMINDER, **kwargs)
        if not retry_error:
            attempts.append(retry_content or "")
            parsed = _parse_judge_object(retry_content)

    salvaged_from_truncation = False
    if not parsed:
        # Salvage complete entries from a truncated results array (newest first).
        for text in reversed(attempts):
            salvaged = _salvage_behavior_results(text)
            if salvaged:
                salvaged_from_truncation = True
                parsed = {
                    "results": salvaged,
                    "summary": (
                        f"Salvaged {len(salvaged)}/{len(expected_behaviors)} behavior "
                        "results from truncated judge response"
                    ),
                }
                break

    if not parsed:
        last = attempts[-1]
        head = last[:80].replace("\n", " ")
        return {
            "score": 0.0,
            "reason": f"Judge response unparseable after retry (len={len(last)}, head={head!r})",
            "results": [],
        }

    results = parsed.get("results", [])
    if results:
        passed_count = sum(1 for r in results if r.get("passed"))
        # A salvaged array is incomplete: behaviors the truncation cut off were
        # never judged and count as not-passed, so keep the denominator at the
        # number of expected behaviors instead of inflating against the few
        # recovered entries.
        denominator = len(expected_behaviors) if salvaged_from_truncation else len(results)
        score = passed_count / denominator
    else:
        score = parsed.get("score", 0.0)
        if not isinstance(score, (int, float)):
            score = 0.0

    return {
        "score": round(max(0.0, min(1.0, float(score))), 4),
        "reason": parsed.get("summary", ""),
        "results": results,
    }
