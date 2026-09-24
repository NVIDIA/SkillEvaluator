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
import urllib.error
from dataclasses import replace
from typing import Any
from urllib.parse import urlsplit

from skillevaluator.constants import LLM_VERIFY_MODEL, LLM_VERIFY_TEMPERATURE
from skillevaluator.inference.retry import resolve_retry_config, retry_call_with_backoff
from skillevaluator.inference.types import EmptyLLMResponseError, LLMClientError
from skillevaluator.logging_config import get_logger
from skillevaluator.provider_config import (
    OPENAI_BASE_URL,
    ProviderConfig,
    ProviderConfigurationError,
    _model_leaf,
    _supports_custom_temperature,
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


def _token_limit_kwargs(config: ProviderConfig, max_tokens: int | None) -> dict[str, int]:
    if max_tokens is None:
        return {}
    if _is_native_openai_endpoint(config) and _model_leaf(config.model).startswith("gpt-5"):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


def _temperature_kwargs(model: str, temperature: float | None) -> dict[str, float]:
    if temperature is None or not _supports_custom_temperature(model):
        return {}
    return {"temperature": temperature}


_SCHEMA_UNSUPPORTED_TARGETS: set[tuple[str, str, str]] = set()


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


_SCHEMA_OPTION_INDICATORS: frozenset[str] = frozenset(
    {
        "response_format",
        "output_config",
        "json_schema",
        "structured output",
        "structured outputs",
        "structured_output",
        "structured_outputs",
    }
)

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
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        parts.append(str(body))
        error_dict = body.get("error")
        if isinstance(error_dict, dict):
            parts.append(str(error_dict.get("message", "")))
            param = error_dict.get("param")
            if param:
                parts.append(str(param))
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

    full_text = " ".join(part for part in parts if part).lower()
    has_option = any(indicator in full_text for indicator in _SCHEMA_OPTION_INDICATORS)
    if not has_option and "schema" in full_text and ("unsupported" in full_text or "not supported" in full_text):
        has_option = True
    has_reason = any(indicator in full_text for indicator in _UNSUPPORTED_REASON_INDICATORS)
    return has_option and has_reason


def _call_with_schema_fallback(
    call_fn: Any,
    call_kwargs: dict[str, Any],
    *,
    schema_key: str,
    target_key: tuple[str, str, str],
    use_schema: bool,
) -> Any:
    """Invoke call_fn and downgrade to prompt-only on confirmed HTTP 400/422 schema errors."""
    try:
        return call_fn(**call_kwargs)
    except Exception as exc:
        if use_schema and _is_schema_unsupported_error(exc):
            logger.warning(
                "Structured output schema unsupported by provider=%s model=%s; "
                "downgrading to prompt-only JSON and memoizing target.",
                target_key[0],
                target_key[2],
            )
            fallback_kwargs = dict(call_kwargs)
            fallback_kwargs.pop(schema_key, None)
            result = call_fn(**fallback_kwargs)
            _SCHEMA_UNSUPPORTED_TARGETS.add(target_key)
            return result
        raise


def _extract_choice_content(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    first_choice = choices[0] if choices else None
    message = getattr(first_choice, "message", None) if first_choice is not None else None
    content = getattr(message, "content", None) if message is not None else ""
    if not content:
        return ""
    return content.strip()


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
    model, token limit, or temperature without touching ``__init__``.
    """

    default_model: str = LLM_VERIFY_MODEL
    default_max_tokens: int | None = None
    default_temperature: float | None = LLM_VERIFY_TEMPERATURE

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
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
        retry_cfg = resolve_retry_config()
        self._max_retries = max_retries if max_retries is not None else retry_cfg.max_retries
        self._retry_base_delay = retry_base_delay if retry_base_delay is not None else retry_cfg.base_delay
        self._retry_max_delay = retry_max_delay if retry_max_delay is not None else retry_cfg.max_delay
        self._http_client = http_client
        self._client: Any = None
        self._provider_config: ProviderConfig | None = None

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
    def max_retries(self) -> int:
        """Return the maximum number of retry attempts for transient errors."""
        return self._max_retries

    @property
    def retry_base_delay(self) -> float:
        """Return the initial base backoff delay in seconds."""
        return self._retry_base_delay

    @property
    def retry_max_delay(self) -> float:
        """Return the maximum delay ceiling in seconds for a retry backoff."""
        return self._retry_max_delay

    # -- client management ------------------------------------------------

    def _resolved_config(self) -> ProviderConfig:
        """Resolve and cache the selected public provider configuration."""
        if self._provider_config is not None:
            return self._provider_config

        if self._api_key or self._base_url:
            model = self._model or self.default_model
            base_url = _effective_openai_base_url(self._base_url)
            self._provider_config = ProviderConfig(
                provider="openai" if _is_canonical_openai_base_url(base_url) else "openai-compatible",
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
        target_key = (config.provider, config.base_url or "", config.model)

        def _invoke_provider() -> str:
            use_schema = response_schema is not None and target_key not in _SCHEMA_UNSUPPORTED_TARGETS
            if config.provider == "anthropic":
                call_kwargs: dict[str, Any] = {
                    "model": config.model,
                    "max_tokens": self._max_tokens or 4096,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                    **_temperature_kwargs(config.model, self._temperature),
                }
                if use_schema and response_schema is not None:
                    call_kwargs["output_config"] = _build_anthropic_output_config(response_schema)
                response = _call_with_schema_fallback(
                    client.messages.create,
                    call_kwargs,
                    schema_key="output_config",
                    target_key=target_key,
                    use_schema=use_schema,
                )
                content = "".join(
                    str(block.text) for block in response.content if getattr(block, "type", None) == "text"
                )
                if not content:
                    raise EmptyLLMResponseError("LLM returned empty response content")
                return content.strip()
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
                content = _extract_choice_content(response)
                if not content:
                    raise EmptyLLMResponseError("LLM returned empty response content")
                return content
            call_kwargs: dict[str, Any] = {
                "model": config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **_temperature_kwargs(config.model, self._temperature),
                **_token_limit_kwargs(config, self._max_tokens),
            }
            if use_schema and response_schema is not None:
                call_kwargs["response_format"] = _build_openai_response_format(response_schema, schema_name)

            response = _call_with_schema_fallback(
                client.chat.completions.create,
                call_kwargs,
                schema_key="response_format",
                target_key=target_key,
                use_schema=use_schema,
            )
            content = _extract_choice_content(response)
            if not content:
                raise EmptyLLMResponseError("LLM returned empty response content")
            return content

        return retry_call_with_backoff(
            _invoke_provider,
            max_retries=self._max_retries,
            base_delay=self._retry_base_delay,
            max_delay=self._retry_max_delay,
        )

    def extract_json_from_response(self, system_prompt: str, user_prompt: str) -> dict:
        """Send a completion and parse JSON from the response."""
        raw = self.completions(system_prompt, user_prompt)

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
        receive a usable result.
        """
        try:
            system_prompt = self.get_system_prompt()
            user_prompt = self.create_user_prompt(**kwargs)
            raw = self.completions(system_prompt, user_prompt)
            return self.parse_response(raw, **kwargs)
        except LLMClientError:
            logger.warning("LLM not configured - using fallback response")
            return self.get_fallback_response(**kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:
            logger.warning(f"LLM call failed ({exc}) - using fallback response")
            return self.get_fallback_response(**kwargs)


def _litellm_model(provider: str, model: str) -> str:
    if provider == "bedrock":
        return f"bedrock/{model}"
    if provider == "anthropic":
        return f"anthropic/{model}"
    return f"openai/{model}"
