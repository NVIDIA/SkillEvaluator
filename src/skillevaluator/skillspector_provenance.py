# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and project SkillSpector LLM provenance at report boundaries.

SkillSpector v2.12.0 and later report ``metadata.llm_provenance``: the provider
adapter, the model and package revision behind each LLM analyzer, and the
sampling controls of the scan. The record is copied into reports only when it
matches that fixed, versioned contract. Unknown fields are dropped, and a
malformed, inconsistent or secret-shaped record is omitted as a whole rather
than redacted into a misleading identity.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

from skillevaluator.utils.redaction import contains_credential_value

__all__ = [
    "SKILLSPECTOR_LLM_ANALYZERS",
    "SKILLSPECTOR_LLM_PROVENANCE_CONTROL_ADAPTERS",
    "SKILLSPECTOR_LLM_PROVENANCE_SCHEMA_VERSION",
    "SKILLSPECTOR_PROVIDER_IDS",
    "sanitize_skillspector_llm_provenance",
    "sanitize_skillspector_provenance_metadata",
    "skillspector_provenance_records",
]

SKILLSPECTOR_LLM_PROVENANCE_SCHEMA_VERSION = 1
SKILLSPECTOR_LLM_ANALYZERS = (
    "mcp_tool_poisoning",
    "semantic_developer_intent",
    "semantic_quality_policy",
    "semantic_security_discovery",
    "meta_analyzer",
)
SKILLSPECTOR_LLM_PROVENANCE_SOURCES = frozenset(
    {"environment", "provider_default", "unset", "invalid_environment", "out_of_range", "unknown"}
)
SKILLSPECTOR_LLM_PROVENANCE_DEPLOYMENT_SOURCES = frozenset(
    {"environment", "resolved_model", "not_applicable", "unknown"}
)
SKILLSPECTOR_LLM_PROVENANCE_API_VERSION_SOURCES = frozenset(
    {"environment", "provider_default", "not_applicable", "unknown"}
)
SKILLSPECTOR_LLM_PROVENANCE_SOURCE_REVISION_SOURCES = frozenset(
    {"build_environment", "package_vcs_metadata", "unknown"}
)
# SkillSpector's adapter support for each optional sampling control. A record
# that claims support an adapter does not have is inconsistent and is omitted.
SKILLSPECTOR_LLM_PROVENANCE_CONTROL_ADAPTERS = {
    "temperature": frozenset(
        {
            "anthropic",
            "anthropic_proxy",
            "azure_openai",
            "bedrock",
            "nv_build",
            "ollama",
            "openai",
            "openai_compatible",
        }
    ),
    "seed": frozenset({"azure_openai", "nv_build", "ollama", "openai", "openai_compatible"}),
    "reasoning_effort": frozenset(
        {
            "anthropic",
            "anthropic_proxy",
            "nv_build",
            "ollama",
            "openai",
            "openai_compatible",
        }
    ),
}
# SkillEvaluator providers the SkillSpector bridge resolves (``--llm``). They
# name the SkillEvaluator configuration, which SkillSpector's own adapter label
# cannot: ``nv_build`` and ``openai-compatible`` both run as its ``openai`` adapter.
SKILLSPECTOR_PROVIDER_IDS = frozenset({"anthropic", "bedrock", "nv_build", "openai", "openai-compatible"})
SKILLSPECTOR_LLM_PROVENANCE_MIN_SEED = -(1 << 63)
SKILLSPECTOR_LLM_PROVENANCE_MAX_SEED = (1 << 63) - 1

_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,255}")
_SETTING_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._:/+\-]{0,255}")
_SOURCE_REVISION_RE = re.compile(r"[0-9a-f]{7,64}")
_CREDENTIAL_PREFIXES = ("sk-", "nvapi-", "ghp_", "glpat-", "bearer-")


def _bounded_text(value: object, *, max_length: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text if text and len(text) <= max_length else None


def _text_is_safe(text: str) -> bool:
    return not (
        "://" in text or "@" in text or text.lower().startswith(_CREDENTIAL_PREFIXES) or contains_credential_value(text)
    )


def _label(value: object) -> str | None:
    text = _bounded_text(value, max_length=256)
    if text is None or not _LABEL_RE.fullmatch(text) or not _text_is_safe(text):
        return None
    return text


def _setting(value: object) -> str | None:
    text = _bounded_text(value, max_length=256)
    if text is None or not _SETTING_RE.fullmatch(text) or not _text_is_safe(text):
        return None
    return text


def _sanitize_control(name: str, value: object, *, expected_adapter_support: bool) -> dict | None:
    if not isinstance(value, Mapping):
        return None
    source = value.get("source")
    adapter_support = value.get("adapter_support")
    if (
        not isinstance(source, str)
        or source not in SKILLSPECTOR_LLM_PROVENANCE_SOURCES
        or not isinstance(adapter_support, bool)
        or adapter_support is not expected_adapter_support
        or value.get("provider_support") != "unknown"
    ):
        return None

    def _scalar(raw: object) -> int | float | str | None:
        if raw is None:
            return None
        if isinstance(raw, bool):
            raise ValueError
        if name == "temperature":
            if isinstance(raw, (int, float)) and math.isfinite(raw) and 0 <= raw <= 1:
                return float(raw)
            raise ValueError
        if name == "seed":
            if (
                isinstance(raw, int)
                and SKILLSPECTOR_LLM_PROVENANCE_MIN_SEED <= raw <= SKILLSPECTOR_LLM_PROVENANCE_MAX_SEED
            ):
                return raw
            raise ValueError
        label = _setting(raw)
        if label is not None:
            return label
        raise ValueError

    try:
        requested = _scalar(value.get("requested"))
        forwarded = _scalar(value.get("forwarded_to_client"))
    except ValueError:
        return None
    if source != "environment" and (requested is not None or forwarded is not None):
        return None
    if source == "environment" and requested is None:
        return None
    if forwarded is not None and (not adapter_support or forwarded != requested):
        return None
    return {
        "requested": requested,
        "source": source,
        "forwarded_to_client": forwarded,
        "adapter_support": adapter_support,
        "provider_support": "unknown",
    }


def _sanitize_routing(raw_routing: Mapping, *, resolved_adapter: str) -> dict | None:
    deployment_override_raw = raw_routing.get("deployment_override")
    deployment_override = None if deployment_override_raw is None else _label(deployment_override_raw)
    api_version_raw = raw_routing.get("api_version")
    api_version = None if api_version_raw is None else _label(api_version_raw)
    deployment_source = raw_routing.get("deployment_source")
    api_version_source = raw_routing.get("api_version_source")
    if (
        (deployment_override_raw is not None and deployment_override is None)
        or (api_version_raw is not None and api_version is None)
        or not isinstance(deployment_source, str)
        or deployment_source not in SKILLSPECTOR_LLM_PROVENANCE_DEPLOYMENT_SOURCES
        or not isinstance(api_version_source, str)
        or api_version_source not in SKILLSPECTOR_LLM_PROVENANCE_API_VERSION_SOURCES
    ):
        return None

    if resolved_adapter == "azure_openai":
        if deployment_override is None:
            if deployment_source not in {"resolved_model", "unknown"}:
                return None
        elif deployment_source != "environment":
            return None
        if api_version is None:
            if api_version_source != "unknown":
                return None
        elif api_version_source not in {"environment", "provider_default"}:
            return None
    elif resolved_adapter == "unknown":
        if (
            deployment_override is not None
            or deployment_source != "unknown"
            or api_version is not None
            or api_version_source != "unknown"
        ):
            return None
    elif (
        deployment_override is not None
        or deployment_source != "not_applicable"
        or api_version is not None
        or api_version_source != "not_applicable"
    ):
        return None
    return {
        "deployment_override": deployment_override,
        "deployment_source": deployment_source,
        "api_version": api_version,
        "api_version_source": api_version_source,
    }


def _sanitize_analyzer_revision(revision: object) -> dict | None:
    if not isinstance(revision, Mapping) or revision.get("source") != "skillspector_package":
        return None
    value = _label(revision.get("value"))
    if value is None:
        return None
    canonical: dict[str, object] = {"value": value, "source": "skillspector_package"}
    if "source_revision" not in revision:
        return canonical
    raw_source_revision = revision.get("source_revision")
    if not isinstance(raw_source_revision, Mapping):
        return None
    source_revision = raw_source_revision.get("value")
    source_revision_source = raw_source_revision.get("source")
    if (
        not isinstance(source_revision_source, str)
        or source_revision_source not in SKILLSPECTOR_LLM_PROVENANCE_SOURCE_REVISION_SOURCES
        or not isinstance(source_revision, str)
        or (source_revision == "unknown") != (source_revision_source == "unknown")
        or (source_revision != "unknown" and not _SOURCE_REVISION_RE.fullmatch(source_revision))
    ):
        return None
    canonical["source_revision"] = {"value": source_revision, "source": source_revision_source}
    return canonical


def sanitize_skillspector_llm_provenance(value: object) -> dict | None:
    """Return the canonical safe SkillSpector provenance record, or ``None`` if invalid."""
    if not isinstance(value, Mapping):
        return None
    schema_version = value.get("schema_version")
    if type(schema_version) is not int or schema_version != SKILLSPECTOR_LLM_PROVENANCE_SCHEMA_VERSION:
        return None

    raw_provider = value.get("provider")
    raw_analyzers = value.get("analyzers")
    raw_sampling = value.get("sampling")
    raw_determinism = value.get("determinism")
    if not all(
        isinstance(item, expected)
        for item, expected in (
            (raw_provider, Mapping),
            (raw_analyzers, list),
            (raw_sampling, Mapping),
            (raw_determinism, Mapping),
        )
    ):
        return None

    configured_adapter = _label(raw_provider.get("configured_adapter"))
    resolved_adapter = _label(raw_provider.get("resolved_adapter"))
    effective_adapter = _label(raw_provider.get("effective_adapter"))
    raw_effective_adapters = raw_provider.get("effective_adapters")
    raw_routing = raw_provider.get("routing")
    if (
        configured_adapter is None
        or resolved_adapter is None
        or effective_adapter is None
        or raw_provider.get("service") != "unknown"
        or not isinstance(raw_effective_adapters, list)
        or not isinstance(raw_routing, Mapping)
    ):
        return None

    effective_adapters: list[str] = []
    for raw_adapter in raw_effective_adapters:
        adapter = _label(raw_adapter)
        if adapter is None or adapter in effective_adapters:
            return None
        effective_adapters.append(adapter)
    if effective_adapters != sorted(effective_adapters):
        return None

    routing = _sanitize_routing(raw_routing, resolved_adapter=resolved_adapter)
    if routing is None:
        return None

    analyzer_by_id: dict[str, Mapping] = {}
    for item in raw_analyzers:
        if not isinstance(item, Mapping):
            return None
        analyzer_id = item.get("analyzer_id")
        if (
            not isinstance(analyzer_id, str)
            or analyzer_id not in SKILLSPECTOR_LLM_ANALYZERS
            or analyzer_id in analyzer_by_id
        ):
            return None
        analyzer_by_id[analyzer_id] = item
    if set(analyzer_by_id) != set(SKILLSPECTOR_LLM_ANALYZERS):
        return None

    analyzers = []
    for analyzer_id in SKILLSPECTOR_LLM_ANALYZERS:
        item = analyzer_by_id[analyzer_id]
        model = _label(item.get("model"))
        revision = _sanitize_analyzer_revision(item.get("analyzer_revision"))
        if model is None or item.get("model_source") != "resolved_configuration" or revision is None:
            return None
        analyzers.append(
            {
                "analyzer_id": analyzer_id,
                "model": model,
                "model_source": "resolved_configuration",
                "analyzer_revision": revision,
            }
        )

    # Adapter support follows the adapters that answered, or the resolved
    # adapter when no provider response was observed.
    support_adapters = effective_adapters or [resolved_adapter]
    sampling = {}
    for name in ("temperature", "seed", "reasoning_effort"):
        expected_adapter_support = all(
            adapter in SKILLSPECTOR_LLM_PROVENANCE_CONTROL_ADAPTERS[name] for adapter in support_adapters
        )
        control = _sanitize_control(
            name,
            raw_sampling.get(name),
            expected_adapter_support=expected_adapter_support,
        )
        if control is None:
            return None
        sampling[name] = control

    classification = raw_determinism.get("classification")
    control_status = raw_determinism.get("control_status")
    if (
        not isinstance(classification, str)
        or not isinstance(control_status, str)
        or classification not in {"nondeterministic", "not_applicable"}
        or raw_determinism.get("provider_guarantee") is not False
    ):
        return None
    # Effective adapters and forwarded controls come only from provider
    # responses, which a scan without LLM execution never has.
    if classification == "not_applicable":
        expected_effective_adapter = "not_applicable"
        if effective_adapters:
            return None
    elif effective_adapters:
        expected_effective_adapter = effective_adapters[0] if len(effective_adapters) == 1 else "mixed"
    else:
        expected_effective_adapter = "unknown"
    if effective_adapter != expected_effective_adapter or (
        not effective_adapters and any(control["forwarded_to_client"] is not None for control in sampling.values())
    ):
        return None

    requested_controls = [control for control in sampling.values() if control["source"] == "environment"]
    if classification == "not_applicable":
        expected_control_statuses = {"not_applied"}
    elif any(control["source"] in {"invalid_environment", "out_of_range"} for control in sampling.values()):
        expected_control_statuses = {"invalid_configuration"}
    elif any(control["source"] == "unknown" for control in sampling.values()):
        expected_control_statuses = {"configuration_unknown"}
    elif not effective_adapters:
        expected_control_statuses = {"controls_not_observed"}
    elif not requested_controls:
        # A response that did not record every control is "not observed".
        expected_control_statuses = {"provider_defaults", "controls_not_observed"}
    elif all(control["forwarded_to_client"] is not None for control in requested_controls):
        expected_control_statuses = {"best_effort_controls_forwarded", "controls_not_observed"}
    else:
        expected_control_statuses = {"controls_partially_forwarded", "controls_not_observed"}
    if control_status not in expected_control_statuses:
        return None
    reason = (
        "Optional controls do not guarantee identical provider output."
        if classification == "nondeterministic"
        else "LLM analysis was not executed for this scan."
    )
    return {
        "schema_version": SKILLSPECTOR_LLM_PROVENANCE_SCHEMA_VERSION,
        "provider": {
            "configured_adapter": configured_adapter,
            "resolved_adapter": resolved_adapter,
            "effective_adapter": effective_adapter,
            "effective_adapters": effective_adapters,
            "service": "unknown",
            "routing": routing,
        },
        "analyzers": analyzers,
        "sampling": sampling,
        "determinism": {
            "classification": classification,
            "control_status": control_status,
            "provider_guarantee": False,
            "reason": reason,
        },
    }


def _provider_id(value: object) -> str | None:
    return value if isinstance(value, str) and value in SKILLSPECTOR_PROVIDER_IDS else None


def sanitize_skillspector_provenance_metadata(metadata: object) -> dict[str, object]:
    """Project only the canonical, report-safe SkillSpector provenance fields of result metadata."""
    if not isinstance(metadata, Mapping):
        return {}
    projected: dict[str, object] = {}
    provenance = sanitize_skillspector_llm_provenance(metadata.get("skillspector_llm_provenance"))
    if provenance is not None:
        projected["skillspector_llm_provenance"] = provenance
    provider_id = _provider_id(metadata.get("skillspector_provider_id"))
    if provider_id is not None:
        projected["skillspector_provider_id"] = provider_id
    return projected


def skillspector_provenance_records(metadata: object) -> list[dict[str, object]]:
    """Return the provenance record of one result for human-readable reports, if any."""
    projected = sanitize_skillspector_provenance_metadata(metadata)
    if not projected:
        return []
    return [
        {
            "provider_id": projected.get("skillspector_provider_id"),
            "provenance": projected.get("skillspector_llm_provenance"),
        }
    ]
