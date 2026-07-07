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

import json
from dataclasses import replace
from typing import Any

from skillevaluator.constants import LLM_VERIFY_MODEL, LLM_VERIFY_TEMPERATURE
from skillevaluator.inference.types import LLMClientError
from skillevaluator.logging_config import get_logger
from skillevaluator.provider_config import ProviderConfig, ProviderConfigurationError, resolve_llm_provider

logger = get_logger(__name__)


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
    default_temperature: float = LLM_VERIFY_TEMPERATURE

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        self._temperature = temperature if temperature is not None else self.default_temperature
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
    def temperature(self) -> float:
        return self._temperature

    # -- client management ------------------------------------------------

    def _resolved_config(self) -> ProviderConfig:
        """Resolve and cache the selected public provider configuration."""
        if self._provider_config is not None:
            return self._provider_config

        if self._api_key or self._base_url:
            model = self._model or self.default_model
            self._provider_config = ProviderConfig(
                provider="openai-compatible",
                model=model,
                api_key=self._api_key,
                base_url=self._base_url,
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
            client_kwargs: dict[str, Any] = {"api_key": config.api_key}
            if config.base_url:
                client_kwargs["base_url"] = config.base_url
            self._client = Anthropic(**client_kwargs)
            return self._client

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMClientError(
                "The 'openai' package is required for LLM operations. Install it with: pip install openai"
            ) from exc

        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        return self._client

    # -- direct-use methods -----------------------------------------------

    def completions(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request and return the response text.

        Raises :class:`LLMClientError` when the response is empty.
        """
        config = self._resolved_config()
        client = self._get_client()
        if config.provider == "anthropic":
            response = client.messages.create(
                model=config.model,
                max_tokens=self._max_tokens or 4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=self._temperature,
            )
            content = "".join(str(block.text) for block in response.content if getattr(block, "type", None) == "text")
            if not content:
                raise LLMClientError("LLM returned empty response content")
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
                temperature=self._temperature,
                aws_region_name=config.region,
                **({"max_tokens": self._max_tokens} if self._max_tokens is not None else {}),
            )
            content = response.choices[0].message.content
            if not content:
                raise LLMClientError("LLM returned empty response content")
            return str(content).strip()
        call_kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._temperature,
        }
        if self._max_tokens is not None:
            call_kwargs["max_tokens"] = self._max_tokens

        response = client.chat.completions.create(**call_kwargs)
        content = response.choices[0].message.content
        if not content:
            raise LLMClientError("LLM returned empty response content")
        return content.strip()

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
