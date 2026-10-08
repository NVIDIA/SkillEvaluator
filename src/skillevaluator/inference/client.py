# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified public-provider LLM client for SkillEvaluator.

Provides a single ``LLMClient`` class for public provider-backed checks. The
class supports two usage patterns:

1. **Direct** -- call ``completions()`` or ``extract_json_from_response()``
   with explicit system/user prompts.
2. **Template-method** -- subclass and override ``get_system_prompt``,
   ``create_user_prompt``, ``parse_response``, and
   ``get_fallback_response``, then call ``process(**kwargs)``.

The provider is resolved lazily from ``SKILL_EVAL_LLM_PROVIDER`` and its
provider-native credential. Importing this module never requires a key.
"""

from __future__ import annotations

import io
import json
import os
import re
import urllib.error
from collections.abc import Callable
from dataclasses import replace
from typing import Any, NamedTuple
from urllib.parse import urlsplit

from skillevaluator.constants import LLM_VERIFY_TEMPERATURE
from skillevaluator.inference.diagnostics import llm_failure_diagnostic
from skillevaluator.inference.retry import RetryConfig, resolve_retry_config, retry_call_with_backoff
from skillevaluator.inference.types import (
    EmptyLLMResponseError,
    LLMClientError,
    LLMClientRefusedError,
    LLMClientTruncatedError,
)
from skillevaluator.logging_config import get_logger
from skillevaluator.provider_config import (
    ANTHROPIC_REFUSAL_FALLBACK_MODEL,
    ANTHROPIC_SERVER_SIDE_FALLBACK_BETA,
    CHAT_DEFAULT_MODELS,
    MAX_COMPLETION_TOKENS,
    OPENAI_BASE_URL,
    ProviderConfig,
    ProviderConfigurationError,
    _supports_custom_temperature,
    completion_token_limit,
    effective_reasoning_effort,
    is_claude_5_5_or_later,
    is_openai_reasoning_model,
    resolve_llm_provider,
)

logger = get_logger(__name__)

_NATIVE_OPENAI_AUTHORITIES = frozenset({"api.openai.com", "api.openai.com:443"})
_NATIVE_OPENAI_PATHS = frozenset({"/v1", "/v1/"})


def _effective_openai_base_url(explicit_base_url: str | None) -> str:
    if explicit_base_url is not None:
        return explicit_base_url
    return os.environ.get("OPENAI_BASE_URL", OPENAI_BASE_URL)


def _is_canonical_openai_base_url(base_url: str | None) -> bool:
    if (
        not isinstance(base_url, str)
        or base_url != base_url.strip()
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in base_url)
        or any(delimiter in base_url for delimiter in ("?", "#", ";", "\\"))
    ):
        return False

    try:
        endpoint = urlsplit(base_url)
        endpoint_port = endpoint.port
    except (TypeError, ValueError):
        return False

    return (
        endpoint.scheme.casefold() == "https"
        and endpoint.netloc.casefold() in _NATIVE_OPENAI_AUTHORITIES
        and endpoint.hostname is not None
        and endpoint.hostname.casefold() == "api.openai.com"
        and endpoint_port in {None, 443}
        and endpoint.path in _NATIVE_OPENAI_PATHS
        and endpoint.username is None
        and endpoint.password is None
        and not endpoint.query
        and not endpoint.fragment
    )


def _is_native_openai_endpoint(config: ProviderConfig) -> bool:
    return config.provider.casefold() == "openai" and _is_canonical_openai_base_url(config.base_url)


def _sdk_targets_native_openai(client: Any) -> bool:
    try:
        request_url = str(client.base_url.join("chat/completions"))
    except (AttributeError, TypeError, ValueError):
        return False
    suffix = "/chat/completions"
    return request_url.endswith(suffix) and _is_canonical_openai_base_url(request_url[: -len(suffix)])


def _is_native_anthropic_endpoint(config: ProviderConfig) -> bool:
    return config.provider == "anthropic" and (
        not config.base_url or urlsplit(config.base_url).hostname == "api.anthropic.com"
    )


def _token_limit_kwargs(config: ProviderConfig, max_tokens: int | None) -> dict[str, int]:
    if max_tokens is None:
        return {}
    max_tokens = completion_token_limit(config.model, max_tokens)
    if _is_native_openai_endpoint(config) and is_openai_reasoning_model(config.model):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


def _anthropic_max_tokens(model: str, max_tokens: int) -> int:
    """Clamp ``max_tokens`` to the Anthropic SDK's per-model non-streaming limit.

    The SDK refuses a non-streaming request above that limit (8,192 for
    Claude Opus 4 and 4.1) before sending it.
    """
    try:
        from anthropic._constants import MODEL_NONSTREAMING_TOKENS
    except ImportError:
        return max_tokens
    limit = MODEL_NONSTREAMING_TOKENS.get(model)
    return min(max_tokens, limit) if isinstance(limit, int) else max_tokens


def _temperature_kwargs(model: str, temperature: float | None) -> dict[str, float]:
    if temperature is None or not _supports_custom_temperature(model):
        return {}
    return {"temperature": temperature}


class SchemaTargetKey(NamedTuple):
    """Identify a provider, base URL, and model target for schema capability memoization."""

    provider: str
    base_url: str
    model: str


_SCHEMA_UNSUPPORTED_TARGETS: set[SchemaTargetKey | tuple[str, str, str]] = set()


def _build_openai_response_format(schema: dict[str, Any], schema_name: str = "judge_response") -> dict[str, Any]:
    """Build OpenAI-compatible JSON schema response_format payload."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": schema,
        },
    }


def _build_anthropic_output_config(schema: dict[str, Any]) -> dict[str, Any]:
    """Build Anthropic Messages API output_config payload."""
    return {
        "format": {
            "type": "json_schema",
            "schema": schema,
        }
    }


_UNSUPPORTED_REASON_INDICATORS: tuple[str, ...] = (
    "unsupported",
    "not supported",
    "extra input",
    "extra inputs",
    "unknown parameter",
    "unknown field",
    "unknown argument",
    "unrecognized request argument",
    "unrecognized parameter",
    "unexpected keyword argument",
    "unexpected argument",
    "invalid parameter",
    "invalid argument",
    "not permitted",
    "not allowed",
    "disallowed",
)

# Gateways can turn a schema into a forced tool call, so a tool_choice rejection is a schema rejection too.
_SCHEMA_OPTION_PATTERN = (
    r"(?:response_format|response format|output_config|json_schema|structured[_ ]outputs?|tool_choice)"
)
_SCHEMA_REJECTION_REASON = (
    r"(?:unsupported|not supported|not permitted|not allowed|disallowed|"
    r"unknown (?:parameter|field|argument)|unrecognized (?:request argument|parameter)|"
    r"unexpected (?:keyword )?argument|extra inputs?(?: are not permitted)?)"
)
_SCHEMA_REJECTION_AFTER_OPTION = re.compile(
    rf"\b{_SCHEMA_OPTION_PATTERN}\b(?:\.[a-z0-9_]+)*"
    rf"(?:\s+of\s+type\s+['\"]?[a-z0-9_]+['\"]?)?"
    rf"\s*(?:(?:is|are|was|were)\s+(?:an?\s+)?|:\s*)?"
    rf"{_SCHEMA_REJECTION_REASON}\b",
    re.IGNORECASE,
)
_SCHEMA_REJECTION_BEFORE_OPTION = re.compile(
    rf"\b(?:unsupported|not supported|extra inputs?(?: are not permitted)?|unknown (?:parameter|field|argument)|"
    rf"unrecognized (?:request argument|parameter)|unexpected (?:keyword argument|argument)|"
    rf"invalid (?:parameter|argument)|not permitted|not allowed|disallowed)\b"
    rf"(?:\s+supplied)?[\s:'\"\[\]{{}}(),-]{{0,32}}\b{_SCHEMA_OPTION_PATTERN}\b",
    re.IGNORECASE,
)


def _message_rejects_schema_option(text: str, param: str | None = None) -> bool:
    """Match a rejection of the schema option itself, not unrelated error text."""
    if param:
        if not re.search(rf"\b{_SCHEMA_OPTION_PATTERN}\b", param, re.IGNORECASE):
            return False
        return any(indicator in text.lower() for indicator in _UNSUPPORTED_REASON_INDICATORS)
    return bool(_SCHEMA_REJECTION_AFTER_OPTION.search(text) or _SCHEMA_REJECTION_BEFORE_OPTION.search(text))


def _is_schema_unsupported_error(exc: Exception) -> bool:
    """Determine whether an exception indicates structured output schema is unsupported."""
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code is None and isinstance(exc, urllib.error.HTTPError):
        status_code = exc.code

    is_type_error = isinstance(exc, TypeError)
    if status_code not in {400, 422} and not is_type_error:
        return False

    parts: list[str] = [str(exc), getattr(exc, "message", "")]
    error_param: str | None = None
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        parts.append(str(body))
        error_dict = body.get("error")
        if isinstance(error_dict, dict):
            parts.append(str(error_dict.get("message", "")))
            param = error_dict.get("param")
            if isinstance(param, str):
                error_param = param
    elif isinstance(body, str):
        parts.append(body)

    response = getattr(exc, "response", None)
    if response is not None:
        text = getattr(response, "text", None)
        if isinstance(text, str):
            parts.append(text)

    if isinstance(exc, urllib.error.HTTPError):
        try:
            body_bytes = exc.read()
            exc.fp = io.BytesIO(body_bytes)
            parts.append(body_bytes.decode("utf-8", "replace"))
        except Exception:
            pass

    full_text = " ".join(part for part in parts if part)
    return _message_rejects_schema_option(full_text, error_param)


def _call_with_schema_fallback(
    call_fn: Any,
    build_kwargs: Callable[[bool], dict[str, Any]],
    *,
    target_key: SchemaTargetKey,
    use_schema: bool,
) -> Any:
    """Invoke call_fn and downgrade to prompt-only on confirmed HTTP 400/422 schema errors.

    ``build_kwargs(include_schema)`` returns the provider request with or
    without its structured-output option, so each provider keeps its other
    request fields on the downgrade.
    """
    try:
        return call_fn(**build_kwargs(use_schema))
    except Exception as exc:
        if use_schema and _is_schema_unsupported_error(exc):
            logger.warning(
                "Structured output schema unsupported by provider=%s model=%s; "
                "downgrading to prompt-only JSON and memoizing target.",
                target_key.provider,
                target_key.model,
            )
            result = call_fn(**build_kwargs(False))
            _SCHEMA_UNSUPPORTED_TARGETS.add(target_key)
            return result
        raise


# OpenAI-compatible finish_reason and Anthropic stop_reason values.
_REFUSAL_STOP_REASONS = frozenset({"content_filter", "refusal"})
_TRUNCATION_STOP_REASONS = frozenset({"length", "max_tokens"})


def _checked_content(content: str, stop_reason: object, refusal: object = None) -> str:
    """Return model text, raising typed errors for refusals, truncation and empty replies."""
    if stop_reason in _REFUSAL_STOP_REASONS or (isinstance(refusal, str) and refusal):
        raise LLMClientRefusedError(f"LLM provider declined the request (stop reason: {stop_reason})")
    if stop_reason in _TRUNCATION_STOP_REASONS:
        raise LLMClientTruncatedError("LLM response stopped at the output-token limit", content)
    if not content:
        raise EmptyLLMResponseError("LLM returned empty response content")
    return content


def _extract_choice_content(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    first_choice = choices[0] if choices else None
    message = getattr(first_choice, "message", None) if first_choice is not None else None
    content = getattr(message, "content", None) if message is not None else ""
    return _checked_content(
        content.strip() if content else "",
        getattr(first_choice, "finish_reason", None),
        getattr(message, "refusal", None),
    )


class LLMClient:
    """Public-provider client for chat completions.

    Supports two usage modes:

    1. **Direct** -- call :meth:`completions` or
       :meth:`extract_json_from_response` with explicit prompts.
    2. **Template-method** -- subclass and override
       :meth:`get_system_prompt`, :meth:`create_user_prompt`,
       :meth:`parse_response`, and :meth:`get_fallback_response`, then
       call :meth:`process`.

    Subclasses may override the ``default_*`` class attributes to change
    model, token limit, temperature, or reasoning effort without touching
    ``__init__``, and set ``response_schema`` to request structured output
    from :meth:`process`. A subclass whose :meth:`parse_response` keeps the
    complete entries of a cut-off reply sets ``salvage_truncated_response``.
    """

    default_model: str | None = None
    default_max_tokens: int | None = None
    default_temperature: float | None = LLM_VERIFY_TEMPERATURE
    default_reasoning_effort: str | None = None
    response_schema: dict[str, Any] | None = None
    schema_name: str = "judge_response"
    salvage_truncated_response: bool = False

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        max_retries: int | None = None,
        retry_base_delay: float | None = None,
        retry_max_delay: float | None = None,
        http_client: Any = None,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        self._temperature = temperature if temperature is not None else self.default_temperature
        self._reasoning_effort = reasoning_effort if reasoning_effort is not None else self.default_reasoning_effort
        retry_cfg = resolve_retry_config()
        self._retry_config = RetryConfig(
            max_retries=max_retries if max_retries is not None else retry_cfg.max_retries,
            base_delay=retry_base_delay if retry_base_delay is not None else retry_cfg.base_delay,
            max_delay=retry_max_delay if retry_max_delay is not None else retry_cfg.max_delay,
        )
        self._http_client = http_client
        self._client: Any = None
        self._provider_config: ProviderConfig | None = None
        self.last_failure: str | None = None

    # -- public read-only properties for introspection --------------------

    @property
    def model(self) -> str:
        return self._model or self._resolved_config().model

    @property
    def base_url(self) -> str | None:
        return self._base_url or self._resolved_config().base_url

    @property
    def api_key(self) -> str | None:
        return self._api_key or self._resolved_config().api_key

    @property
    def temperature(self) -> float | None:
        return self._temperature

    @property
    def reasoning_effort(self) -> str | None:
        """Return the reasoning effort sent to the resolved model, or ``None`` for its default."""
        config = self._resolved_config()
        return effective_reasoning_effort(config.provider, config.model, self._reasoning_effort)

    @property
    def max_retries(self) -> int:
        """Return the maximum number of retry attempts for transient errors."""
        return self._retry_config.max_retries

    @property
    def retry_base_delay(self) -> float:
        """Return the initial base backoff delay in seconds."""
        return self._retry_config.base_delay

    @property
    def retry_max_delay(self) -> float:
        """Return the maximum delay ceiling in seconds for a retry backoff."""
        return self._retry_config.max_delay

    # -- client management ------------------------------------------------

    def _resolved_config(self) -> ProviderConfig:
        """Resolve and cache the selected public provider configuration."""
        if self._provider_config is not None:
            return self._provider_config

        if self._api_key or self._base_url:
            base_url = _effective_openai_base_url(self._base_url)
            provider = "openai" if _is_canonical_openai_base_url(base_url) else "openai-compatible"
            model = self._model or self.default_model or CHAT_DEFAULT_MODELS[provider]
            self._provider_config = ProviderConfig(
                provider=provider,
                model=model,
                api_key=self._api_key,
                base_url=base_url,
                litellm_model=f"openai/{model}",
            )
            return self._provider_config

        try:
            config = resolve_llm_provider()
        except ProviderConfigurationError as exc:
            raise LLMClientError(str(exc)) from exc
        if self._model:
            config = replace(config, model=self._model, litellm_model=_litellm_model(config.provider, self._model))
        self._provider_config = config
        return config

    def _get_client(self) -> Any:
        """Lazily construct the SDK client for the selected provider."""
        if self._client is not None:
            return self._client

        config = self._resolved_config()

        if config.provider == "bedrock":
            self._client = config
            return self._client

        if not config.api_key:
            raise LLMClientError(f"No API key resolved for {config.provider}.")

        if config.provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise LLMClientError(
                    "The 'anthropic' package is required for Anthropic LLM operations. Install with: pip install 'skillevaluator[llm]'"
                ) from exc
            client_kwargs: dict[str, Any] = {"api_key": config.api_key, "max_retries": 0}
            if config.base_url:
                client_kwargs["base_url"] = config.base_url
            if self._http_client is not None:
                client_kwargs["http_client"] = self._http_client
            self._client = Anthropic(**client_kwargs)
            return self._client

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMClientError(
                "The 'openai' package is required for LLM operations. Install it with: pip install openai"
            ) from exc

        client_kwargs: dict[str, Any] = {
            "api_key": config.api_key,
            "base_url": config.base_url,
            "max_retries": 0,
        }
        if self._http_client is not None:
            client_kwargs["http_client"] = self._http_client
        client = OpenAI(**client_kwargs)
        if (
            config.base_url is not None
            and not _is_canonical_openai_base_url(config.base_url)
            and _sdk_targets_native_openai(client)
        ):
            client.close()
            raise LLMClientError(
                f"OpenAI base URL is a noncanonical alias for the native OpenAI endpoint. Use {OPENAI_BASE_URL}."
            )
        self._client = client
        return self._client

    # -- direct-use methods -----------------------------------------------

    def completions(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "judge_response",
    ) -> str:
        """Send a chat completion request and return the response text.

        Raises :class:`LLMClientError` when the response is empty.
        """
        config = self._resolved_config()
        client = self._get_client()
        target_key = SchemaTargetKey(config.provider, config.base_url or "", config.model)
        reasoning_effort = self.reasoning_effort

        def _invoke_provider() -> str:
            use_schema = response_schema is not None and target_key not in _SCHEMA_UNSUPPORTED_TARGETS
            if config.provider == "anthropic":

                def _anthropic_kwargs(include_schema: bool) -> dict[str, Any]:
                    call_kwargs: dict[str, Any] = {
                        "model": config.model,
                        "max_tokens": _anthropic_max_tokens(config.model, self._max_tokens or MAX_COMPLETION_TOKENS),
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": user_prompt}],
                        **_temperature_kwargs(config.model, self._temperature),
                    }
                    output_config: dict[str, Any] = {}
                    if include_schema and response_schema is not None:
                        output_config.update(_build_anthropic_output_config(response_schema))
                    if reasoning_effort is not None:
                        output_config["effort"] = reasoning_effort
                    if output_config:
                        call_kwargs["output_config"] = output_config
                    if is_claude_5_5_or_later(config.model) and _is_native_anthropic_endpoint(config):
                        # Server-side fallback (beta): a safety-classifier refusal is retried on
                        # the fallback model within the same call. Older SDKs lack typed fields.
                        call_kwargs["extra_headers"] = {"anthropic-beta": ANTHROPIC_SERVER_SIDE_FALLBACK_BETA}
                        call_kwargs["extra_body"] = {"fallbacks": [{"model": ANTHROPIC_REFUSAL_FALLBACK_MODEL}]}
                    return call_kwargs

                response = _call_with_schema_fallback(
                    client.messages.create,
                    _anthropic_kwargs,
                    target_key=target_key,
                    use_schema=use_schema,
                )
                content = "".join(
                    str(block.text) for block in response.content if getattr(block, "type", None) == "text"
                )
                return _checked_content(content.strip(), getattr(response, "stop_reason", None))
            if config.provider == "bedrock":
                try:
                    from litellm import completion
                except ImportError as exc:
                    raise LLMClientError(
                        "The 'litellm' package is required for Bedrock LLM operations. Install with: pip install 'skillevaluator[llm]'"
                    ) from exc
                response = completion(
                    model=config.litellm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    aws_region_name=config.region,
                    **_temperature_kwargs(config.model, self._temperature),
                    **({"max_tokens": self._max_tokens} if self._max_tokens is not None else {}),
                )
                return _extract_choice_content(response)

            def _openai_kwargs(include_schema: bool) -> dict[str, Any]:
                call_kwargs: dict[str, Any] = {
                    "model": config.model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    **_temperature_kwargs(config.model, self._temperature),
                    **_token_limit_kwargs(config, self._max_tokens),
                    **({"reasoning_effort": reasoning_effort} if reasoning_effort is not None else {}),
                }
                if include_schema and response_schema is not None:
                    call_kwargs["response_format"] = _build_openai_response_format(response_schema, schema_name)
                return call_kwargs

            response = _call_with_schema_fallback(
                client.chat.completions.create,
                _openai_kwargs,
                target_key=target_key,
                use_schema=use_schema,
            )
            return _extract_choice_content(response)

        return retry_call_with_backoff(_invoke_provider, config=self._retry_config)

    def extract_json_from_response(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "judge_response",
    ) -> dict:
        """Send a completion and parse JSON from the response."""
        raw = self.completions(system_prompt, user_prompt, response_schema=response_schema, schema_name=schema_name)

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0].strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise LLMClientError(f"LLM returned invalid JSON: {e}\nRaw: {raw[:500]}") from e

    # -- template-method hooks (override in subclasses) -------------------

    def get_system_prompt(self) -> str:
        """Return the system-role prompt.  Override in subclasses."""
        raise NotImplementedError("Subclasses must implement get_system_prompt for template-method usage")

    def create_user_prompt(self, **kwargs: Any) -> str:
        """Build the user-role prompt.  Override in subclasses."""
        raise NotImplementedError("Subclasses must implement create_user_prompt for template-method usage")

    def parse_response(self, response_text: str, **kwargs: Any) -> Any:
        """Parse the raw LLM response text.  Override in subclasses."""
        raise NotImplementedError("Subclasses must implement parse_response for template-method usage")

    def get_fallback_response(self, **kwargs: Any) -> Any:
        """Return a safe fallback result.  Override in subclasses."""
        raise NotImplementedError("Subclasses must implement get_fallback_response for template-method usage")

    # -- template-method orchestrator -------------------------------------

    def process(self, **kwargs: Any) -> Any:
        """Orchestrate a full LLM interaction: prompt -> call -> parse.

        On failure the fallback response is returned so callers always
        receive a usable result, and ``last_failure`` records a bounded
        diagnostic of the cause. When ``salvage_truncated_response`` is set,
        a reply cut off at the token limit is parsed for its complete entries
        instead, and ``last_failure`` still records the truncation.
        """
        self.last_failure = None
        try:
            system_prompt = self.get_system_prompt()
            user_prompt = self.create_user_prompt(**kwargs)
            raw = self.completions(
                system_prompt,
                user_prompt,
                response_schema=self.response_schema,
                schema_name=self.schema_name,
            )
            return self.parse_response(raw, **kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:
            self.last_failure = llm_failure_diagnostic(exc)
            if isinstance(exc, LLMClientTruncatedError) and exc.content and self.salvage_truncated_response:
                logger.warning("LLM reply was truncated (%s) - keeping its complete entries", self.last_failure)
                try:
                    return self.parse_response(exc.content, **kwargs)
                except Exception:
                    pass
            logger.warning("LLM call failed (%s) - using fallback response", self.last_failure)
            return self.get_fallback_response(**kwargs)


def _litellm_model(provider: str, model: str) -> str:
    if provider == "bedrock":
        return f"bedrock/{model}"
    if provider == "anthropic":
        return f"anthropic/{model}"
    return f"openai/{model}"
