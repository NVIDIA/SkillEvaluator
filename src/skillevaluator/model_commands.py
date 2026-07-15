# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-facing authenticated model-catalog command."""

from __future__ import annotations

import json
import unicodedata
from urllib.parse import urlsplit

import click

from skillevaluator.model_catalog import ModelCatalogError, fetch_model_records, select_catalog_models
from skillevaluator.provider_config import ProviderConfigurationError, resolve_llm_provider


def run_models_command(*, limit: int, as_json: bool) -> int:
    """List models visible to the canonically selected public provider."""
    try:
        config = resolve_llm_provider()
        records = fetch_model_records(config)
        models = select_catalog_models(config, records, limit=limit)
    except (ProviderConfigurationError, ModelCatalogError) as exc:
        click.echo(f"Error: {exc}", err=True)
        return 1

    endpoint = _safe_endpoint_origin(config.base_url)
    if as_json:
        payload = {
            "provider": config.provider,
            "endpoint": endpoint,
            "configured_model": config.model,
            "models": [
                {
                    "id": model.id,
                    "created": model.created,
                    "is_configured": model.is_configured,
                }
                for model in models
            ],
        }
        click.echo(json.dumps(payload, sort_keys=True, ensure_ascii=True))
        return 0

    click.echo(f"Provider: {config.provider}")
    click.echo(f"Endpoint: {endpoint}")
    click.echo(f"Configured model: {_safe_terminal_text(config.model)}")
    click.echo("Catalog models:")
    if not models:
        click.echo("  (no model candidates matched the filter)")
    for model in models:
        marker = "*" if model.is_configured else " "
        click.echo(f"{marker} {_safe_terminal_text(model.id)}")
    click.echo("* configured model")
    return 0


def _safe_endpoint_origin(base_url: str | None) -> str:
    """Show only endpoint origin; custom paths can contain secret routing tokens."""
    if not base_url:
        return "provider default"
    try:
        parsed = urlsplit(base_url)
        if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
            return "custom endpoint (details hidden)"
        port = parsed.port
    except ValueError:
        return "custom endpoint (details hidden)"
    hostname = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    authority = f"{hostname}:{port}" if port is not None else hostname
    return f"{parsed.scheme.casefold()}://{authority}"


def _safe_terminal_text(value: str) -> str:
    return "".join(
        character
        if unicodedata.category(character) not in {"Cc", "Cf", "Cs"}
        else "\N{REPLACEMENT CHARACTER}"
        for character in value
    )
