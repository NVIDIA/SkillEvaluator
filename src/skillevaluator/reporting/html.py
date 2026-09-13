# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTML reporter for standalone web reports.

This reporter produces self-contained HTML files suitable for:
- Email attachments
- CI/CD artifacts
- Compliance audit trails
- Standalone report viewing

Features include:
- Professional MARSFlow-inspired styling
- Dark mode toggle
- JSON export functionality
- Navigation tabs for future Tier 2/3/4 support
- Filtering and collapsible sections
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import pkgutil
import re
import sys
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlparse

from jinja2 import BaseLoader, Environment

from skillevaluator import __version__
from skillevaluator.constants import TIER3_LIFT_FAIL_THRESHOLD, TIER3_LIFT_PASS_THRESHOLD
from skillevaluator.publication_evidence import result_publication_evidence_dict
from skillevaluator.reporting.base import (
    ReporterBase,
    agent_eval_publication_evidence_complete,
    agent_eval_report_serialization_issue,
    agent_eval_report_serialization_limits,
    agent_eval_report_text_limit,
    assess_publication,
    assess_tier3_evidence,
    get_skip_reason,
    is_advisory_agent_eval_skip,
    is_cleanly_skipped,
    is_tier2_result,
    is_tier3_result,
    passes_required_gate,
    publication_identity_present,
    publication_semantic_text,
    publication_target_conflict_marker,
    publication_target_dict,
    result_publication_target_conflict_marker,
    result_publication_target_dict,
    select_agent_eval_candidate,
)
from skillevaluator.reporting.base import (
    is_tier2_validator_name as _is_tier2_validator_name,
)
from skillevaluator.reporting.harbor_viewer import normalize_agent_eval_harbor_links

if TYPE_CHECKING:
    from skillevaluator.models import ValidationResult


# Tier 3 already enforces a 2 MiB canonical payload limit. HTML needs a
# separate bound because script-safe escaping (``<`` -> ``\u003c``), pretty
# diagnostics, and visible dataset fields can otherwise multiply that payload
# many times over. Large canonical payloads are embedded once as base64 while a
# bounded projection feeds the human-readable panels.
_TIER3_JSON_EMBED_MAX_BYTES = 512 * 1024
_TIER3_HTML_PREVIEW_TRIGGER_BYTES = 256 * 1024
_TIER3_HTML_PREVIEW_CHARS = 128 * 1024
_TIER3_HTML_PREVIEW_STRING_CHARS = 4 * 1024
_TIER3_HTML_PREVIEW_COLLECTION_ITEMS = 64
_TIER3_PREVIEW_MARKER = "... [HTML preview truncated; download the full Tier 3 payload]"
_TIER3_JSON_SAFE_PRIORITY_KEYS = (
    "schema_version",
    "verdict",
    "execution_status",
    "summary",
    "skill_name",
    "publication_target",
    "run_id",
    "evaluated_at",
    "evaluator_version",
    "dataset_digest",
    "dataset_digest_algorithm",
    "benchmark_policy",
    "attempt_policy",
    "dataset_summary",
    "agents",
    "model",
    "model_name",
    "llm_model",
    "with_skill",
    "overall_score",
    "scored_attempts",
    "dimensions",
    "id",
    "score",
    "evaluators",
)
_TIER3_PROBABILITY_TEXT = re.compile(r"(?:\d{1,16}(?:\.\d{0,16})?|\.\d{1,16})(?:[eE][+-]?\d{1,9})?")


def is_tier2_validator_name(validator_name: str | None) -> bool:
    """Compatibility wrapper for callers that imported this helper here."""
    return _is_tier2_validator_name(validator_name)


def _finite_number(
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    """Return a bounded finite number, rejecting booleans and shaped values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if minimum is not None and number < minimum:
        return None
    if maximum is not None and number > maximum:
        return None
    return number


def _nonnegative_count(value: object) -> int:
    return count if (count := _strict_nonnegative_count(value)) is not None else 0


def _strict_nonnegative_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value.bit_length() > 63:
        return None
    return value


def _json_safe_tier3_text(value: str) -> str:
    """Normalize template-facing text and replace invalid UTF-8 surrogates."""
    safe_value = value.encode("utf-8", errors="replace").decode("utf-8")
    return publication_semantic_text(safe_value)


def _json_safe_tier3_mapping_keys(value: dict[object, object]) -> Iterator[object]:
    """Yield proof keys first without materializing an untrusted wide mapping."""
    priority = frozenset(_TIER3_JSON_SAFE_PRIORITY_KEYS)
    for key in _TIER3_JSON_SAFE_PRIORITY_KEYS:
        if key in value:
            yield key
    for key in value:
        if key not in priority:
            yield key


def _json_safe_tier3_value(
    value: object,
    *,
    _depth: int = 0,
    _active_containers: set[int] | None = None,
    _remaining_nodes: list[int] | None = None,
    _remaining_text_bytes: list[int] | None = None,
    _truncated: list[bool] | None = None,
    _max_depth: int | None = None,
    _preserve_exact_text: bool = False,
) -> Any:
    """Copy untrusted Tier 3 metadata into a finite JSON-compatible shape."""
    max_depth, max_nodes = agent_eval_report_serialization_limits()
    if _max_depth is not None:
        max_depth = _max_depth
    active_containers = _active_containers if _active_containers is not None else set()
    remaining_nodes = _remaining_nodes if _remaining_nodes is not None else [max_nodes]
    remaining_text_bytes = (
        _remaining_text_bytes if _remaining_text_bytes is not None else [agent_eval_report_text_limit()]
    )
    truncated = _truncated if _truncated is not None else [False]
    if remaining_nodes[0] <= 0:
        truncated[0] = True
        return None
    remaining_nodes[0] -= 1
    if _depth > max_depth:
        truncated[0] = True
        return None
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        if len(value) > remaining_text_bytes[0]:
            truncated[0] = True
            return None
        utf8_safe_value = value.encode("utf-8", errors="replace").decode("utf-8")
        safe_value = _json_safe_tier3_text(value)
        if utf8_safe_value != value or safe_value != unicodedata.normalize("NFKC", utf8_safe_value):
            truncated[0] = True
        elif _preserve_exact_text and unicodedata.normalize("NFC", utf8_safe_value) == utf8_safe_value:
            # Publication target names are filesystem identities. Preserve
            # their exact NFC spelling instead of collapsing compatibility-
            # distinct ASCII and fullwidth spellings.
            safe_value = utf8_safe_value
        encoded_size = len(safe_value.encode("utf-8"))
        if encoded_size > remaining_text_bytes[0]:
            truncated[0] = True
            return None
        remaining_text_bytes[0] -= encoded_size
        return safe_value
    if isinstance(value, int):
        if value.bit_length() <= 256:
            return value
        truncated[0] = True
        return None
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        truncated[0] = True
        return None
    if isinstance(value, dict):
        container_id = id(value)
        if container_id in active_containers:
            truncated[0] = True
            return None
        active_containers.add(container_id)
        safe: dict[str, Any] = {}
        collided_keys: set[str] = set()
        try:
            total_keys = len(value)
            processed_keys = 0
            if total_keys and remaining_nodes[0] <= 0:
                truncated[0] = True
                return safe
            for key in _json_safe_tier3_mapping_keys(value):
                processed_keys += 1
                if isinstance(key, str):
                    if len(key) > remaining_text_bytes[0]:
                        truncated[0] = True
                        remaining_nodes[0] -= 1
                        if remaining_nodes[0] <= 0 and processed_keys < total_keys:
                            break
                        continue
                    utf8_safe_key = key.encode("utf-8", errors="replace").decode("utf-8")
                    safe_key = _json_safe_tier3_text(key)
                    if utf8_safe_key != key or safe_key != unicodedata.normalize("NFKC", utf8_safe_key):
                        truncated[0] = True
                elif (
                    isinstance(key, bool)
                    or (isinstance(key, int) and key.bit_length() <= 256)
                    or (isinstance(key, float) and math.isfinite(key))
                ):
                    safe_key = str(key)
                    truncated[0] = True
                else:
                    truncated[0] = True
                    remaining_nodes[0] -= 1
                    if remaining_nodes[0] <= 0 and processed_keys < total_keys:
                        break
                    continue
                if len(safe_key) > remaining_text_bytes[0]:
                    truncated[0] = True
                    remaining_nodes[0] -= 1
                    if remaining_nodes[0] <= 0 and processed_keys < total_keys:
                        break
                    continue
                encoded_key_size = len(safe_key.encode("utf-8"))
                if encoded_key_size > remaining_text_bytes[0]:
                    truncated[0] = True
                    remaining_nodes[0] -= 1
                    if remaining_nodes[0] <= 0 and processed_keys < total_keys:
                        break
                    continue
                remaining_text_bytes[0] -= encoded_key_size
                if not safe_key or safe_key in collided_keys:
                    truncated[0] = True
                    remaining_nodes[0] -= 1
                    if remaining_nodes[0] <= 0 and processed_keys < total_keys:
                        break
                    continue
                if safe_key in safe:
                    # Drop every alias in a normalized-key collision rather
                    # than retaining an order-dependent first or last value.
                    safe.pop(safe_key)
                    collided_keys.add(safe_key)
                    truncated[0] = True
                    remaining_nodes[0] -= 1
                    if remaining_nodes[0] <= 0 and processed_keys < total_keys:
                        break
                    continue
                item = value[key]
                if safe_key == "publication_target":
                    item = publication_target_dict(item)
                elif safe_key == "publication_target_conflict":
                    item = publication_target_conflict_marker(item)
                safe[safe_key] = _json_safe_tier3_value(
                    item,
                    _depth=_depth + 1,
                    _active_containers=active_containers,
                    _remaining_nodes=remaining_nodes,
                    _remaining_text_bytes=remaining_text_bytes,
                    _truncated=truncated,
                    _max_depth=max_depth,
                    _preserve_exact_text=_preserve_exact_text or safe_key in {"publication_target", "skill_name"},
                )
                if remaining_nodes[0] <= 0 and processed_keys < total_keys:
                    truncated[0] = True
                    break
        finally:
            active_containers.remove(container_id)
        return safe
    if isinstance(value, (list, tuple)):
        if isinstance(value, tuple):
            truncated[0] = True
        container_id = id(value)
        if container_id in active_containers:
            truncated[0] = True
            return None
        active_containers.add(container_id)
        safe_items: list[Any] = []
        try:
            for item in value:
                if remaining_nodes[0] <= 0:
                    truncated[0] = True
                    break
                safe_items.append(
                    _json_safe_tier3_value(
                        item,
                        _depth=_depth + 1,
                        _active_containers=active_containers,
                        _remaining_nodes=remaining_nodes,
                        _remaining_text_bytes=remaining_text_bytes,
                        _truncated=truncated,
                        _max_depth=max_depth,
                        _preserve_exact_text=_preserve_exact_text,
                    )
                )
        finally:
            active_containers.remove(container_id)
        return safe_items
    truncated[0] = True
    return None


def _json_safe_tier3_payload(value: object) -> dict[str, Any]:
    """Return a bounded payload while disclosing any serialization truncation."""
    truncated = [False]
    normalized = _json_safe_tier3_value(value, _truncated=truncated)
    payload = normalized if isinstance(normalized, dict) else {}
    if not isinstance(value, dict):
        truncated[0] = True
    if truncated[0]:
        payload["_serialization_truncated"] = True
    return payload


def _string_list(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str) and item] if isinstance(value, list) else []


def _dict_list(value: object) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _tier3_display_identifier(value: object) -> str:
    if not publication_identity_present(value):
        return ""
    identifier = publication_semantic_text(value).strip()
    return identifier if publication_identity_present(identifier) else ""


def _tier3_identifier_list(value: object) -> list[str]:
    """Return visible, normalized, collision-free identifiers in source order."""
    if not isinstance(value, list):
        return []
    identifiers: list[str] = []
    seen: set[str] = set()
    for raw_identifier in value:
        identifier = _tier3_display_identifier(raw_identifier)
        identity_key = identifier.casefold()
        if not identifier or identity_key in seen:
            continue
        seen.add(identity_key)
        identifiers.append(identifier)
    return identifiers


def _tier3_agent_entries(value: object) -> list[tuple[str, dict[str, Any]]]:
    """Normalize agent keys and fail closed on malformed or colliding peers."""
    if not isinstance(value, dict):
        return []
    entries: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for raw_name, raw_agent in value.items():
        if not isinstance(raw_name, str):
            return []
        name = _tier3_display_identifier(raw_name)
        identity_key = name.casefold()
        if not name or not isinstance(raw_agent, dict) or identity_key in seen:
            return []
        seen.add(identity_key)
        entries.append((name, raw_agent))
    return entries


def _score_mapping(
    value: object,
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    scores: dict[str, float] = {}
    seen: set[str] = set()
    for key, item in value.items():
        identifier = _tier3_display_identifier(key)
        identity_key = identifier.casefold()
        if not identifier:
            continue
        if identity_key in seen:
            return {}
        seen.add(identity_key)
        score = _finite_number(item, minimum=minimum, maximum=maximum)
        if score is not None:
            scores[identifier] = score
    return scores


def _tier3_text_mapping(value: object) -> dict[str, str]:
    """Normalize public label maps without silently overwriting aliases."""
    if not isinstance(value, dict):
        return {}
    labels: dict[str, str] = {}
    seen: set[str] = set()
    for raw_name, raw_label in value.items():
        name = _tier3_display_identifier(raw_name)
        identity_key = name.casefold()
        if not publication_identity_present(raw_label):
            continue
        label = publication_semantic_text(raw_label).strip()
        if not name or not publication_identity_present(label):
            continue
        if identity_key in seen:
            return {}
        seen.add(identity_key)
        labels[name] = label
    return labels


def _sanitize_tier3_dimension(value: dict[str, Any]) -> dict[str, Any]:
    dimension = dict(value)
    for key in ("baseline", "with_skill", "score"):
        dimension[key] = _finite_number(dimension.get(key), minimum=0.0, maximum=1.0)
    dimension["lift"] = _finite_number(dimension.get("lift"), minimum=-1.0, maximum=1.0)
    for key in ("id", "dimension", "label"):
        if key in dimension:
            dimension[key] = _tier3_display_identifier(dimension[key])
    if "explanation" in dimension and not isinstance(dimension["explanation"], str):
        dimension["explanation"] = ""
    if "verdict" in dimension:
        raw_verdict = dimension["verdict"]
        dimension["verdict"] = None if raw_verdict is None else _tier3_display_identifier(raw_verdict)
    dimension["evaluators"] = _tier3_identifier_list(dimension.get("evaluators"))
    dimension["reasoning_bullets"] = _string_list(dimension.get("reasoning_bullets"))
    return dimension


def _sanitize_tier3_pass_condition(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    condition = dict(value)
    if "rate" in condition:
        rate = _finite_number(condition.get("rate"), minimum=0.0, maximum=1.0)
        if rate is None:
            condition.pop("rate", None)
        else:
            condition["rate"] = rate
    for key in (
        "passed_cases",
        "failed_cases",
        "total_cases",
        "attempts_used",
        "max_attempts_possible",
    ):
        if key in condition:
            condition[key] = _nonnegative_count(condition.get(key))

    cases: dict[str, dict[str, Any]] = {}
    raw_cases = condition.get("cases")
    if isinstance(raw_cases, dict):
        case_ids: set[str] = set()
        for raw_case_id, raw_case in raw_cases.items():
            case_id = _tier3_display_identifier(raw_case_id)
            identity_key = case_id.casefold()
            if not isinstance(raw_case, dict):
                continue
            if not case_id:
                continue
            if identity_key in case_ids:
                cases = {}
                break
            case_ids.add(identity_key)
            case = dict(raw_case)
            for key in ("attempts_used", "attempts_missing", "attempts_skipped"):
                case[key] = _nonnegative_count(case.get(key))
            case["best_score"] = _finite_number(case.get("best_score"), minimum=0.0, maximum=1.0)
            first_pass = case.get("first_pass_attempt")
            case["first_pass_attempt"] = (
                _nonnegative_count(first_pass) if isinstance(first_pass, int) and first_pass > 0 else None
            )
            case["passed"] = case.get("passed") if isinstance(case.get("passed"), bool) else False
            attempts: list[dict[str, Any]] = []
            for raw_attempt in _dict_list(case.get("attempts")):
                attempt = dict(raw_attempt)
                attempt["score"] = _finite_number(attempt.get("score"), minimum=0.0, maximum=1.0)
                attempt_number = attempt.get("attempt")
                attempt["attempt"] = (
                    _nonnegative_count(attempt_number)
                    if isinstance(attempt_number, int) and attempt_number > 0
                    else None
                )
                attempts.append(attempt)
            case["attempts"] = attempts
            cases[case_id] = case
    condition["cases"] = cases
    return condition


def _sanitize_tier3_probability_text(value: object) -> tuple[str, Decimal] | None:
    if not isinstance(value, str):
        return None
    text = publication_semantic_text(value).strip()
    if not _TIER3_PROBABILITY_TEXT.fullmatch(text):
        return None
    try:
        probability = Decimal(text)
    except InvalidOperation:
        return None
    if not probability.is_finite() or not Decimal(0) <= probability <= Decimal(1):
        return None
    return text, probability


def _probability_text_matches_number(probability: Decimal, number: float) -> bool:
    numeric_probability = Decimal(str(number))
    if numeric_probability == 0:
        return probability == 0
    relative_match = abs(probability - numeric_probability) <= abs(numeric_probability) * Decimal("1e-9")
    if abs(number) < sys.float_info.min:
        return abs(float(probability) - number) <= math.ulp(number) or relative_match
    return relative_match


def _sanitize_tier3_probability_fields(
    value: dict[str, Any],
    *,
    numeric_key: str,
    text_key: str,
    underflow_key: str,
) -> dict[str, Any] | None:
    if numeric_key not in value:
        return None
    raw_number = value.get(numeric_key)
    raw_text = value.get(text_key)
    parsed_text = _sanitize_tier3_probability_text(raw_text)
    if text_key in value and raw_text is not None and parsed_text is None:
        return None
    raw_underflow = value.get(underflow_key)
    if underflow_key in value and not isinstance(raw_underflow, bool):
        return None

    if raw_number is None:
        if not (
            parsed_text is not None and parsed_text[1] > 0 and float(parsed_text[1]) == 0.0 and raw_underflow is True
        ):
            return None
        return {
            numeric_key: None,
            text_key: parsed_text[0],
            underflow_key: True,
        }

    number = _finite_number(raw_number, minimum=0.0, maximum=1.0)
    if number is None or number <= 0.0 or raw_underflow is True:
        return None
    if parsed_text is not None and not _probability_text_matches_number(parsed_text[1], number):
        return None
    if number < sys.float_info.min and parsed_text is None:
        return None

    fields: dict[str, Any] = {numeric_key: number}
    if parsed_text is not None:
        fields[text_key] = parsed_text[0]
    if underflow_key in value:
        fields[underflow_key] = False
    return fields


def _sanitize_tier3_mcnemar_exact(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    p_value_fields = _sanitize_tier3_probability_fields(
        value,
        numeric_key="p_value",
        text_key="p_value_text",
        underflow_key="p_value_numeric_underflow",
    )
    if p_value_fields is None:
        return {}

    exact = dict(p_value_fields)
    minimum_keys = {
        "minimum_attainable_p_value",
        "minimum_attainable_p_value_text",
        "minimum_attainable_p_value_numeric_underflow",
    }
    if minimum_keys.intersection(value):
        minimum_fields = _sanitize_tier3_probability_fields(
            value,
            numeric_key="minimum_attainable_p_value",
            text_key="minimum_attainable_p_value_text",
            underflow_key="minimum_attainable_p_value_numeric_underflow",
        )
        if minimum_fields is not None:
            exact.update(minimum_fields)

    for key in ("method", "null_hypothesis"):
        text = _tier3_display_identifier(value.get(key))
        if text:
            exact[key] = text
    for key, omitted_key in (
        ("p_value_exact", "p_value_exact_omitted"),
        ("minimum_attainable_p_value_exact", "minimum_attainable_p_value_exact_omitted"),
    ):
        raw_text = value.get(key)
        if isinstance(raw_text, str):
            text = publication_semantic_text(raw_text).strip()
            if text:
                exact[key] = text
        elif key in value and raw_text is None and value.get(omitted_key) is True:
            exact[key] = None
    for key in (
        "p_value_exact_omitted",
        "p_value_numeric_underflow",
        "minimum_attainable_p_value_exact_omitted",
        "minimum_attainable_p_value_numeric_underflow",
        "resolution_limited_at_alpha_0_05",
    ):
        if isinstance(value.get(key), bool):
            exact[key] = value[key]
    for key in (
        "p_value_exact_omitted_reason",
        "minimum_attainable_p_value_exact_omitted_reason",
    ):
        text = _tier3_display_identifier(value.get(key))
        if text:
            exact[key] = text
    return exact


def _sanitize_tier3_paired_comparison(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    paired: dict[str, Any] = {}
    if "pairing_status" in value:
        status = _tier3_display_identifier(value.get("pairing_status")).casefold()
        paired["pairing_status"] = status if status in {"complete", "partial", "unavailable"} else "unavailable"
    count_keys = (
        "paired_cases",
        "with_skill_unpaired_case_count",
        "without_skill_unpaired_case_count",
        "with_skill_unidentified_cases",
        "without_skill_unidentified_cases",
        "both_pass",
        "with_skill_only_pass",
        "without_skill_only_pass",
        "neither_pass",
        "discordant_cases",
    )
    valid_counts: dict[str, int] = {}
    for key in count_keys:
        if key in value:
            count = _strict_nonnegative_count(value.get(key))
            paired[key] = count if count is not None else 0
            if count is not None:
                valid_counts[key] = count
    for key in ("with_skill_unpaired_case_ids", "without_skill_unpaired_case_ids"):
        if key in value:
            paired[key] = _tier3_identifier_list(value.get(key))
    for key in (
        "with_skill_unpaired_case_ids_truncated",
        "without_skill_unpaired_case_ids_truncated",
    ):
        if key in value:
            paired[key] = value.get(key) is True
    outcome_keys = (
        "with_skill_only_pass",
        "without_skill_only_pass",
        "both_pass",
        "neither_pass",
    )
    counts_are_consistent = (
        valid_counts.get("paired_cases", 0) > 0
        and all(key in valid_counts for key in (*outcome_keys, "discordant_cases"))
        and valid_counts["paired_cases"] == sum(valid_counts[key] for key in outcome_keys)
        and valid_counts["discordant_cases"]
        == valid_counts["with_skill_only_pass"] + valid_counts["without_skill_only_pass"]
    )
    if counts_are_consistent:
        paired["paired_rate_delta"] = (
            valid_counts["with_skill_only_pass"] - valid_counts["without_skill_only_pass"]
        ) / valid_counts["paired_cases"]
    if paired.get("pairing_status") == "complete" and counts_are_consistent:
        mcnemar_exact = _sanitize_tier3_mcnemar_exact(value.get("mcnemar_exact"))
        if mcnemar_exact:
            paired["mcnemar_exact"] = mcnemar_exact
    return paired


def _sanitize_tier3_pass_at_k(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    pass_at_k = dict(value)
    pass_at_k["with_skill"] = _sanitize_tier3_pass_condition(pass_at_k.get("with_skill"))
    pass_at_k["without_skill"] = _sanitize_tier3_pass_condition(pass_at_k.get("without_skill"))
    lift = pass_at_k.get("lift") if isinstance(pass_at_k.get("lift"), dict) else {}
    delta = _finite_number(lift.get("delta"), minimum=-1.0, maximum=1.0)
    if delta is None:
        lift.pop("delta", None)
    else:
        lift["delta"] = delta
    paired_comparison = _sanitize_tier3_paired_comparison(lift.get("paired_comparison"))
    if paired_comparison:
        lift["paired_comparison"] = paired_comparison
    else:
        lift.pop("paired_comparison", None)
    pass_at_k["lift"] = lift
    return pass_at_k


def _sanitize_tier3_evaluator_card(value: dict[str, Any]) -> dict[str, Any] | None:
    card = dict(value)
    card["with_skill"] = _finite_number(card.get("with_skill"), minimum=0.0, maximum=1.0)
    if card["with_skill"] is None:
        return None
    card["baseline"] = _finite_number(card.get("baseline"), minimum=0.0, maximum=1.0)
    card["lift"] = _finite_number(card.get("lift"), minimum=-1.0, maximum=1.0)
    for key in ("id", "label", "evaluator", "status"):
        if key in card:
            card[key] = _tier3_display_identifier(card[key])
    evidence: list[dict[str, Any]] = []
    for raw_entry in _dict_list(card.get("evidence")):
        entry = dict(raw_entry)
        entry["entry_id"] = _tier3_display_identifier(entry.get("entry_id"))
        occurrences = _nonnegative_count(entry.get("occurrences"))
        entry["occurrences"] = occurrences if occurrences > 0 else 1
        entry["score"] = _finite_number(entry.get("score"), minimum=0.0, maximum=1.0)
        for key in ("notes", "failures", "checks", "evidence_refs"):
            entry[key] = _string_list(entry.get(key))
        evidence.append(entry)
    card["evidence"] = evidence
    sampling = card.get("evidence_sampling") if isinstance(card.get("evidence_sampling"), dict) else {}
    for key in ("represented_cases", "represented_trials", "total_trials", "scanned_trials"):
        if key in sampling:
            sampling[key] = _nonnegative_count(sampling.get(key))
    sampling["truncated"] = sampling.get("truncated") is True
    card["evidence_sampling"] = sampling
    return card


def _sanitize_tier3_agent(value: dict[str, Any]) -> dict[str, Any]:
    agent = dict(value)
    for key in ("model", "model_name", "llm_model", "display_name", "label"):
        if key not in agent:
            continue
        if not publication_identity_present(agent[key]):
            agent.pop(key)
            continue
        normalized = publication_semantic_text(agent[key])
        if publication_identity_present(normalized):
            agent[key] = normalized
        else:
            agent.pop(key)
    agent["baseline"] = _finite_number(agent.get("baseline"), minimum=0.0, maximum=1.0)
    agent["with_skill"] = _finite_number(agent.get("with_skill"), minimum=0.0, maximum=1.0)
    agent["overall_score"] = _finite_number(agent.get("overall_score"), minimum=0.0, maximum=1.0)
    agent["lift"] = _finite_number(agent.get("lift"), minimum=-1.0, maximum=1.0)
    agent["num_trials"] = _nonnegative_count(agent.get("num_trials"))
    agent["num_trials_baseline"] = _nonnegative_count(agent.get("num_trials_baseline"))
    agent["expected_attempts"] = _nonnegative_count(agent.get("expected_attempts"))
    agent["scored_attempts"] = _nonnegative_count(agent.get("scored_attempts"))
    agent["dimensions"] = [_sanitize_tier3_dimension(item) for item in _dict_list(agent.get("dimensions"))]
    agent["evaluator_cards"] = [
        card
        for item in _dict_list(agent.get("evaluator_cards"))
        if (card := _sanitize_tier3_evaluator_card(item)) is not None
    ]
    agent["findings"] = [_sanitize_tier3_dimension(item) for item in _dict_list(agent.get("findings"))]
    agent["pass_at_k"] = _sanitize_tier3_pass_at_k(agent.get("pass_at_k"))
    return agent


def _scrub_untrusted_tier3_agents(value: object) -> dict[str, dict[str, Any]]:
    """Retain agent identity/coverage while removing untrusted score evidence."""
    agents: dict[str, dict[str, Any]] = {}
    for name, raw_agent in _tier3_agent_entries(value):
        agent: dict[str, Any] = {
            "baseline": None,
            "with_skill": None,
            "overall_score": None,
            "lift": None,
        }
        for key in ("model", "model_name", "llm_model", "display_name", "label", "execution_status"):
            if isinstance(raw_agent.get(key), str):
                agent[key] = raw_agent[key]
        for key in (
            "num_trials",
            "num_trials_baseline",
            "expected_attempts",
            "scored_attempts",
        ):
            agent[key] = _nonnegative_count(raw_agent.get(key))
        for key in ("execution_errors", "warnings"):
            agent[key] = _string_list(raw_agent.get(key))
        agent["dimensions"] = [
            {
                **{
                    key: raw_dimension[key]
                    for key in ("id", "dimension", "label")
                    if isinstance(raw_dimension.get(key), str)
                },
                "baseline": None,
                "with_skill": None,
                "score": None,
                "lift": None,
                "verdict": None,
            }
            for raw_dimension in _dict_list(raw_agent.get("dimensions"))
        ]
        agents[name] = _sanitize_tier3_agent(agent)
    return agents


def _sanitize_tier3_trial(value: dict[str, Any]) -> dict[str, Any]:
    trial = dict(value)
    for key in ("agent", "entry_id", "trial_id"):
        trial[key] = _tier3_display_identifier(trial.get(key))
    trial["overall"] = _finite_number(trial.get("overall"), minimum=0.0, maximum=1.0)
    trial["scores"] = _score_mapping(trial.get("scores"))
    trial["baseline_scores"] = _score_mapping(trial.get("baseline_scores"))
    trial["lift_scores"] = _score_mapping(trial.get("lift_scores"), minimum=-1.0, maximum=1.0)
    tokens = trial.get("tokens") if isinstance(trial.get("tokens"), dict) else {}
    trial["tokens"] = {
        "prompt": _nonnegative_count(tokens.get("prompt")),
        "completion": _nonnegative_count(tokens.get("completion")),
    }
    trial["steps"] = _nonnegative_count(trial.get("steps"))
    trial["warnings"] = _string_list(trial.get("warnings"))
    recovery = trial.get("error_recovery") if isinstance(trial.get("error_recovery"), dict) else {}
    recovery["details"] = _string_list(recovery.get("details"))
    trial["error_recovery"] = recovery
    return trial


def _sanitize_tier3_harbor_viewer(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    viewer = dict(value)
    viewer["jobs"] = _dict_list(viewer.get("jobs"))
    viewer["evidence_links"] = _dict_list(viewer.get("evidence_links"))
    return viewer


def _sanitize_tier3_insight_items(value: object) -> list[dict[str, Any]]:
    items = _dict_list(value)
    for item in items:
        if not isinstance(item.get("evidence"), dict):
            item.pop("evidence", None)
        if not isinstance(item.get("harbor_evidence"), dict):
            item.pop("harbor_evidence", None)
    return items


def _sanitize_tier3_dataset_case(value: dict[str, Any]) -> dict[str, Any]:
    case = dict(value)
    case["id"] = _tier3_display_identifier(case.get("id"))
    case["assertions"] = _string_list(case.get("assertions"))
    case["expected_behavior"] = _string_list(case.get("expected_behavior"))
    return case


def _sanitize_tier3_display_payload(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize template-facing types after merging canonical and fallback data."""
    payload = _json_safe_tier3_payload(value)
    payload["summary"] = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    payload["agents"] = {
        identifier: _sanitize_tier3_agent(agent) for identifier, agent in _tier3_agent_entries(payload.get("agents"))
    }
    for container in (payload, payload["summary"]):
        for key in ("skill_name", "evaluator_version", "environment"):
            if key not in container:
                continue
            if key == "skill_name":
                exact_name = container[key]
                if (
                    isinstance(exact_name, str)
                    and unicodedata.normalize("NFC", exact_name) == exact_name
                    and publication_identity_present(exact_name)
                ):
                    continue
                container.pop(key)
                continue
            normalized = publication_semantic_text(container[key])
            if publication_identity_present(normalized):
                container[key] = normalized
            else:
                container.pop(key)
    evaluators: dict[str, dict[str, float | None]] = {}
    raw_evaluators = payload.get("evaluators")
    if isinstance(raw_evaluators, dict):
        evaluator_names: set[str] = set()
        for raw_name, scores in raw_evaluators.items():
            name = _tier3_display_identifier(raw_name)
            identity_key = name.casefold()
            if not name or not isinstance(scores, dict) or identity_key in evaluator_names:
                evaluators = {}
                break
            evaluator_names.add(identity_key)
            evaluators[name] = {
                "with_skill": _finite_number(scores.get("with_skill"), minimum=0.0, maximum=1.0),
                "baseline": _finite_number(scores.get("baseline"), minimum=0.0, maximum=1.0),
                "lift": _finite_number(scores.get("lift"), minimum=-1.0, maximum=1.0),
            }
    payload["evaluators"] = evaluators
    for key in (
        "insights",
        "dataset_summary",
        "verdict_policy",
        "provenance",
    ):
        if not isinstance(payload.get(key), dict):
            payload[key] = {}
    payload["metric_labels"] = _tier3_text_mapping(payload.get("metric_labels"))
    payload["dimension_hints"] = _tier3_text_mapping(payload.get("dimension_hints"))
    evaluator_paths = payload["provenance"].get("evaluator_paths")
    payload["provenance"]["evaluator_paths"] = evaluator_paths if isinstance(evaluator_paths, dict) else {}
    harbor_viewer = _sanitize_tier3_harbor_viewer(payload.get("harbor_viewer"))
    if harbor_viewer:
        payload["harbor_viewer"] = harbor_viewer
    else:
        payload.pop("harbor_viewer", None)
    summary_harbor_viewer = _sanitize_tier3_harbor_viewer(payload["summary"].get("harbor_viewer"))
    if summary_harbor_viewer:
        payload["summary"]["harbor_viewer"] = summary_harbor_viewer
    else:
        payload["summary"].pop("harbor_viewer", None)
    attempt_policy = payload.get("attempt_policy") if isinstance(payload.get("attempt_policy"), dict) else {}
    if "max_attempts" in attempt_policy:
        max_attempts = _nonnegative_count(attempt_policy.get("max_attempts"))
        if max_attempts < 1:
            attempt_policy.pop("max_attempts", None)
        else:
            attempt_policy["max_attempts"] = max_attempts
    if "pass_threshold" in attempt_policy:
        threshold = _finite_number(attempt_policy.get("pass_threshold"), minimum=0.0, maximum=1.0)
        if threshold is None:
            attempt_policy["pass_threshold"] = None
        else:
            attempt_policy["pass_threshold"] = threshold
    else:
        attempt_policy["pass_threshold"] = None
    if "stop_on_pass" in attempt_policy:
        attempt_policy["stop_on_pass"] = attempt_policy.get("stop_on_pass") is True
    payload["attempt_policy"] = attempt_policy
    payload["pass_at_k"] = _sanitize_tier3_pass_at_k(payload.get("pass_at_k"))
    payload["evaluator_cards"] = [
        card
        for item in _dict_list(payload.get("evaluator_cards"))
        if (card := _sanitize_tier3_evaluator_card(item)) is not None
    ]
    payload["cases"] = _dict_list(payload.get("cases"))
    payload["dimensions"] = [
        dimension
        for item in _dict_list(payload.get("dimensions"))
        if (
            (dimension := _sanitize_tier3_dimension(item)).get("score") is not None
            or dimension.get("with_skill") is not None
        )
    ]
    for key in ("conclusions", "recommendations", "suggestions_v2"):
        payload[key] = _sanitize_tier3_insight_items(payload.get(key))
    payload["dataset"] = [_sanitize_tier3_dataset_case(item) for item in _dict_list(payload.get("dataset"))]
    payload["trials"] = [_sanitize_tier3_trial(item) for item in _dict_list(payload.get("trials"))]
    timing: list[dict[str, Any]] = []
    for item in _dict_list(payload.get("timing")):
        entry = dict(item)
        entry["seconds"] = _finite_number(entry.get("seconds"), minimum=0.0)
        if entry["seconds"] is not None:
            timing.append(entry)
    payload["timing"] = timing
    for key in ("agents_run", "metric_ids"):
        payload[key] = _tier3_identifier_list(payload.get(key))
    for key in ("execution_errors", "suggestions"):
        payload[key] = _string_list(payload.get(key))
    # ``skill_name`` was already validated above as an exact NFC filesystem
    # identity. Do not pass it through compatibility-normalizing display text.
    for key in ("best_agent", "verdict", "execution_status"):
        for container in (payload, payload["summary"]):
            if key not in container:
                continue
            normalized = _tier3_display_identifier(container.get(key))
            if normalized:
                container[key] = normalized
            else:
                container.pop(key, None)
    for key in ("overall_score",):
        payload[key] = _finite_number(payload.get(key), minimum=0.0, maximum=1.0)
        payload["summary"][key] = _finite_number(payload["summary"].get(key), minimum=0.0, maximum=1.0)
    for key in ("overall_lift", "composite_lift"):
        payload[key] = _finite_number(payload.get(key), minimum=-1.0, maximum=1.0)
        if key in payload["summary"]:
            payload["summary"][key] = _finite_number(
                payload["summary"].get(key),
                minimum=-1.0,
                maximum=1.0,
            )
    payload["runtime_seconds"] = _finite_number(payload.get("runtime_seconds"), minimum=0.0) or 0.0
    payload["summary"]["runtime_seconds"] = (
        _finite_number(payload["summary"].get("runtime_seconds"), minimum=0.0) or 0.0
    )
    for key in ("expected_attempts", "scored_attempts"):
        for container in (payload, payload["summary"]):
            if key in container:
                container[key] = _nonnegative_count(container.get(key))
    truncation = payload.get("report_truncation") if isinstance(payload.get("report_truncation"), dict) else {}
    omitted = truncation.get("omitted") if isinstance(truncation.get("omitted"), dict) else {}
    sanitized_omitted = {
        str(section): count for section, value in omitted.items() if (count := _nonnegative_count(value)) > 0
    }
    if sanitized_omitted:
        truncation["omitted"] = sanitized_omitted
    else:
        truncation.pop("omitted", None)
    if truncation:
        payload["report_truncation"] = truncation
    else:
        payload.pop("report_truncation", None)
    return payload


@dataclass
class _Tier3PreviewBudget:
    chars_remaining: int = _TIER3_HTML_PREVIEW_CHARS
    omitted_characters: int = 0
    omitted_items: int = 0


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _script_safe_json(value: object) -> str:
    """Serialize JSON for an HTML raw-text script element without expansion attacks."""
    return (
        _compact_json(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _canonical_tier3_embed(payload: dict[str, Any] | None) -> tuple[str, str]:
    """Return one safe canonical Tier 3 copy and its browser decoding mode."""
    if not payload:
        return "", "json"
    raw = _compact_json(payload)
    safe = _script_safe_json(payload)
    if len(safe.encode("utf-8")) <= _TIER3_JSON_EMBED_MAX_BYTES:
        return safe, "json"
    return base64.b64encode(raw.encode("utf-8")).decode("ascii"), "base64"


def _bounded_tier3_preview_value(value: Any, budget: _Tier3PreviewBudget) -> Any:
    if isinstance(value, str):
        allowed = min(_TIER3_HTML_PREVIEW_STRING_CHARS, max(0, budget.chars_remaining))
        if len(value) <= allowed:
            budget.chars_remaining -= len(value)
            return value
        budget.omitted_characters += len(value) - allowed
        budget.chars_remaining -= allowed
        if allowed <= len(_TIER3_PREVIEW_MARKER):
            return _TIER3_PREVIEW_MARKER[:allowed]
        return value[: allowed - len(_TIER3_PREVIEW_MARKER)] + _TIER3_PREVIEW_MARKER

    if isinstance(value, list):
        kept = min(len(value), _TIER3_HTML_PREVIEW_COLLECTION_ITEMS)
        budget.omitted_items += len(value) - kept
        bounded: list[Any] = []
        for item in value[:kept]:
            bounded.append(_bounded_tier3_preview_value(item, budget))
        return bounded

    if isinstance(value, dict):
        items = list(value.items())
        kept = min(len(items), _TIER3_HTML_PREVIEW_COLLECTION_ITEMS)
        budget.omitted_items += len(items) - kept
        bounded_dict: dict[Any, Any] = {}
        for key, item in items[:kept]:
            budget.chars_remaining = max(0, budget.chars_remaining - len(str(key)))
            bounded_dict[key] = _bounded_tier3_preview_value(item, budget)
        return bounded_dict

    return value


def _bounded_tier3_preview(payload: dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, int]]:
    """Return a presentation-only projection plus visible omission counts."""
    if not payload or len(_compact_json(payload).encode("utf-8")) <= _TIER3_HTML_PREVIEW_TRIGGER_BYTES:
        return payload, {}

    budget = _Tier3PreviewBudget()
    preview = _bounded_tier3_preview_value(payload, budget)
    notice = {
        key: value
        for key, value in {
            "characters": budget.omitted_characters,
            "items": budget.omitted_items,
        }.items()
        if value > 0
    }
    return preview, notice


def _related_paths(finding: object) -> list[str]:
    """Return distinct path-like string values carried in finding metadata."""
    metadata = finding.get("metadata", {}) if isinstance(finding, dict) else getattr(finding, "metadata", {})
    if not isinstance(metadata, dict):
        return []

    paths: list[str] = []
    for key, value in metadata.items():
        normalized_key = str(key).casefold()
        if not (normalized_key == "path" or normalized_key.startswith("path_") or normalized_key.endswith("_path")):
            continue
        if isinstance(value, str) and value and value not in paths:
            paths.append(value)
    return paths


def _adaptive_unsigned_percent_decimals(percentage: float) -> int:
    """Return the existing bounded precision policy for one unsigned percentage."""
    if percentage in {0.0, 100.0}:
        return 0
    if 0 < percentage < 1 or 99 < percentage < 100:
        distance_from_endpoint = min(percentage, 100 - percentage)
        return max(2, min(6, math.ceil(-math.log10(distance_from_endpoint))))
    return 0


def _adaptive_percent(value: object, signed: bool = False) -> str:
    """Format percentages without rounding narrow nonzero effects to zero or endpoints."""
    try:
        percentage = float(value) * 100
    except (TypeError, ValueError, OverflowError):
        return "—"
    if not math.isfinite(percentage):
        return "—"

    if signed:
        magnitude = abs(percentage)
        if magnitude == 0 or magnitude >= 0.1:
            decimals = 1
        else:
            decimals = max(2, min(6, math.ceil(-math.log10(magnitude))))
        return f"{percentage:+.{decimals}f}%"

    decimals = _adaptive_unsigned_percent_decimals(percentage)
    return f"{percentage:.{decimals}f}%"


def _adaptive_interval_percent(interval: object) -> str:
    """Format a percentage interval with shared precision and visible bounds."""
    if not isinstance(interval, dict):
        return "—"
    try:
        lower = float(interval["lower"]) * 100
        upper = float(interval["upper"]) * 100
    except (KeyError, TypeError, ValueError, OverflowError):
        return "—"
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        return "—"

    decimals = max(
        _adaptive_unsigned_percent_decimals(lower),
        _adaptive_unsigned_percent_decimals(upper),
    )
    if lower != upper:
        for candidate in range(decimals, 7):
            if f"{lower:.{candidate}f}" != f"{upper:.{candidate}f}":
                decimals = candidate
                break
        else:
            # Preserve a distinct interval narrower than the six-decimal
            # display budget by rounding its bounds outward at that budget.
            decimals = 6
            scale = 10**decimals
            lower = math.floor(lower * scale) / scale
            upper = math.ceil(upper * scale) / scale

    def _bound(value: float) -> str:
        # Exact mathematical endpoints remain compact even when the other
        # bound needs additional precision.
        if value in {0.0, 100.0}:
            return f"{value:.0f}%"
        return f"{value:.{decimals}f}%"

    return f"{_bound(lower)}\N{EN DASH}{_bound(upper)}"


class PackageLoader(BaseLoader):
    """Custom Jinja2 loader that loads templates from package resources."""

    def __init__(self, package: str, path: str) -> None:
        self.package = package
        self.path = path

    def get_source(self, _environment: Environment, template: str) -> tuple[str, str, callable]:
        """Load template source from package resources."""
        template_path = f"{self.path}/{template}"
        try:
            source = resources.files(self.package).joinpath(template_path).read_text(encoding="utf-8")
        except AttributeError as exc:
            encoded = pkgutil.get_data(self.package, template_path)
            if encoded is None:
                raise FileNotFoundError(template_path) from exc
            source = encoded.decode("utf-8")
        return source, template, lambda: True


class HTMLReporter(ReporterBase):
    """HTML report generator using Jinja2 templates."""

    ICONS: ClassVar[dict[str, str]] = {
        "checkmark": '<svg class="success-icon" viewBox="0 0 16 16" fill="currentColor"><path d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"/></svg>',
        "file": '<svg class="meta-icon" viewBox="0 0 16 16" fill="currentColor"><path d="M3.75 1.5a.25.25 0 00-.25.25v11.5c0 .138.112.25.25.25h8.5a.25.25 0 00.25-.25V4.664a.25.25 0 00-.073-.177l-2.914-2.914a.25.25 0 00-.177-.073H3.75z"/><path fill-rule="evenodd" d="M2 1.75C2 .784 2.784 0 3.75 0h5.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v8.586A1.75 1.75 0 0112.25 15h-8.5A1.75 1.75 0 012 13.25V1.75z"/></svg>',
        "search": '<svg class="meta-icon" viewBox="0 0 16 16" fill="currentColor"><path fill-rule="evenodd" d="M11.5 7a4.499 4.499 0 11-8.998 0A4.499 4.499 0 0111.5 7zm-.82 4.74a6 6 0 111.06-1.06l3.04 3.04a.75.75 0 11-1.06 1.06l-3.04-3.04z"/></svg>',
        "arrow_up": '<svg viewBox="0 0 16 16" fill="currentColor" width="20" height="20"><path fill-rule="evenodd" d="M8 12a.75.75 0 01-.75-.75V5.56L5.03 7.78a.75.75 0 01-1.06-1.06l3.5-3.5a.75.75 0 011.06 0l3.5 3.5a.75.75 0 01-1.06 1.06L8.75 5.56v5.69A.75.75 0 018 12z"/></svg>',
    }

    DEFAULT_TABS: ClassVar[list[dict[str, str]]] = [
        {"id": "tier1", "label": "Tier 1: Security and Static Validation"},
    ]

    def __init__(
        self,
        *,
        include_timestamp: bool = True,
        title: str | None = None,
        tabs: list[dict[str, str]] | None = None,
        target_path: str | None = None,
        content_label: str = "Skill",
        profile: str | None = None,
        expected_skill_name: str | None = None,
    ) -> None:
        self.include_timestamp = include_timestamp
        self.title = title or "SkillEvaluator Validation Report"
        self.tabs = list(tabs) if tabs is not None else list(self.DEFAULT_TABS)
        self._tabs_explicit = tabs is not None
        self.target_path = target_path
        self.content_label = content_label
        # Active validation profile (e.g. "internal", "external"). Surfaced
        # in the report header so reviewers can tell which audience the
        # validation gate was applied for.
        self.profile = profile
        self.expected_skill_name = expected_skill_name
        self._env = self._create_environment()

    def _create_environment(self) -> Environment:
        loader = PackageLoader("skillevaluator.reporting", "templates")
        environment = Environment(loader=loader, autoescape=True)
        environment.filters["cleanly_skipped"] = is_cleanly_skipped
        environment.filters["related_paths"] = _related_paths
        environment.filters["adaptive_percent"] = _adaptive_percent
        environment.filters["adaptive_interval_percent"] = _adaptive_interval_percent
        environment.filters["skip_reason"] = get_skip_reason
        return environment

    @staticmethod
    def _infer_profile_from_results(results: list[ValidationResult]) -> str | None:
        """Read the active profile name out of result metadata.

        ``commands.validate._stamp_policy`` writes ``result.metadata['policy']``
        for every result; we read it back here so reporters constructed
        without an explicit ``profile=`` argument still surface the profile.
        Returns ``None`` if no result carries policy metadata.
        """
        for r in results:
            policy_meta = (r.metadata or {}).get("policy") if isinstance(r.metadata, dict) else None
            if isinstance(policy_meta, dict) and policy_meta.get("profile"):
                return str(policy_meta["profile"])
        return None

    @property
    def name(self) -> str:
        return "html"

    @property
    def description(self) -> str:
        return "Standalone HTML reports for archiving/sharing"

    def _is_single_skill_mode(self, results: list[ValidationResult]) -> str | None:
        """Detect if results are from a single-skill run (not folder-of-skills).

        In folder mode, findings have ``[skill-name] path`` prefixes from
        ``merge_with_prefix`` and success_details use the skill name as
        ``check_name``.  In single-skill mode neither of these holds; instead
        success_detail check_names are validator-internal IDs like
        ``manifest_exists``.

        Returns the inferred skill name if single-skill mode, else None.
        """
        # Check for [prefix] pattern in any finding — indicates folder mode
        for r in results:
            for f in r.findings:
                if f.file_path.startswith("[") and "]" in f.file_path:
                    return None

        # Try to infer skill name from the first absolute file_path
        for r in results:
            for f in r.findings:
                fp = f.file_path
                if "/" in fp:
                    p = Path(fp)
                    while p.parent != p:
                        if (p / "SKILL.md").exists() or (p / "skill.md").exists():
                            return p.name
                        p = p.parent

        # Fall back: look for quality_scores metadata which always has skill_name
        for r in results:
            qs = r.metadata.get("quality_scores") if r.metadata else None
            if qs and qs.get("skill_name"):
                return qs["skill_name"]

        # Fall back: target_path
        if self.target_path:
            tp = Path(self.target_path)
            if (tp / "SKILL.md").exists() or (tp / "skill.md").exists():
                return tp.name

        return None

    def _reorganize_single_skill(
        self,
        results: list[ValidationResult],
        skill_name: str,
    ) -> dict[str, dict[str, Any]]:
        """Reorganize results for a single-skill run into one entry."""
        skill_data: dict[str, Any] = {"passed": True, "validators": {}, "issue_count": 0}

        for result in results:
            vn = result.validator_name
            vdata: dict[str, Any] = {
                "passed": result.passed,
                "description": result.validator_description,
                "details": [],
                "findings": [],
            }

            for detail in result.success_details:
                vdata["details"].append(
                    {
                        "check_name": detail.check_name,
                        "message": detail.message,
                        "metadata": detail.metadata,
                    }
                )

            for finding in result.findings:
                clean_path = finding.file_path
                if "/" + skill_name + "/" in clean_path:
                    clean_path = clean_path[clean_path.index("/" + skill_name + "/") + len(skill_name) + 2 :]

                confidence = "high"
                if hasattr(finding, "metadata") and isinstance(finding.metadata, dict):
                    confidence = finding.metadata.get("confidence", "high")

                vdata["findings"].append(
                    {
                        "category": finding.category,
                        "severity": finding.severity.value
                        if hasattr(finding.severity, "value")
                        else str(finding.severity),
                        "check_name": finding.check_name,
                        "message": self._normalize_message(finding.message),
                        "file_path": clean_path,
                        "line_number": finding.line_number,
                        "line_content": finding.line_content,
                        "suggestion": finding.suggestion,
                        "location": finding.location,
                        "confidence": confidence,
                        "metadata": finding.metadata if hasattr(finding, "metadata") else {},
                    }
                )

            if not result.passed:
                skill_data["passed"] = False
                vdata["passed"] = False

            skill_data["validators"][vn] = vdata

        # Deduplicate findings per validator
        for vdata in skill_data["validators"].values():
            if vdata["findings"]:
                vdata["findings"] = self._deduplicate_findings(vdata["findings"])

        skill_data["issue_count"] = sum(
            sum(f.get("occurrences", 1) for f in vd["findings"]) for vd in skill_data["validators"].values()
        )

        return {skill_name: skill_data}

    def _reorganize_by_skill(self, results: list[ValidationResult]) -> dict[str, dict[str, Any]]:
        """Reorganize validation results by skill instead of by validator.

        Returns a dict like:
        {
            "skill-name": {
                "passed": True/False,
                "validators": {
                    "SCHEMA": {"passed": True, "details": [...], "findings": [...]},
                    "SECURITY": {"passed": True, "details": [...], "findings": [...]},
                }
            }
        }
        """
        # Detect single-skill mode and use the dedicated path
        single_skill = self._is_single_skill_mode(results)
        if single_skill:
            return self._reorganize_single_skill(results, single_skill)

        skills: dict[str, dict[str, Any]] = {}

        for result in results:
            validator_name = result.validator_name

            # Extract skill names from success_details
            for detail in result.success_details:
                skill_name = detail.check_name
                # Skip discovery/folder-level checks
                if skill_name in (
                    "skill_discovery",
                    "folder_structure",
                    "skills_directory",
                    "team_skills_directory",
                    "pii_scan_start",
                    "pii_detection",
                ):
                    continue

                if skill_name not in skills:
                    skills[skill_name] = {"passed": True, "validators": {}}

                if validator_name not in skills[skill_name]["validators"]:
                    skills[skill_name]["validators"][validator_name] = {
                        "passed": True,
                        "description": result.validator_description,
                        "details": [],
                        "findings": [],
                    }

                skills[skill_name]["validators"][validator_name]["details"].append(
                    {
                        "check_name": detail.check_name,
                        "message": detail.message,
                        "metadata": detail.metadata,
                    }
                )

            # Extract skill names from findings (failures)
            for finding in result.findings:
                # Extract skill name from file_path (e.g., "[skill-name] file.md")
                file_path = finding.file_path
                skill_name = None
                if file_path.startswith("[") and "]" in file_path:
                    skill_name = file_path[1 : file_path.index("]")]
                else:
                    # Try to extract from path
                    parts = file_path.split("/")
                    if len(parts) > 0:
                        skill_name = parts[0]

                if skill_name:
                    if skill_name not in skills:
                        skills[skill_name] = {"passed": False, "validators": {}, "issue_count": 0}

                    if validator_name not in skills[skill_name]["validators"]:
                        skills[skill_name]["validators"][validator_name] = {
                            "passed": False,
                            "description": result.validator_description,
                            "details": [],
                            "findings": [],
                        }

                    skills[skill_name]["validators"][validator_name]["passed"] = False
                    skills[skill_name]["passed"] = False

                    # Clean file_path: strip redundant [skill-name] prefix
                    clean_path = file_path
                    if file_path.startswith("[") and "] " in file_path:
                        clean_path = file_path[file_path.index("] ") + 2 :]

                    # Strip absolute paths -- keep only path relative to skill dir
                    if "/" + skill_name + "/" in clean_path:
                        clean_path = clean_path[clean_path.index("/" + skill_name + "/") + len(skill_name) + 2 :]

                    confidence = "high"
                    if hasattr(finding, "metadata") and isinstance(finding.metadata, dict):
                        confidence = finding.metadata.get("confidence", "high")
                    skills[skill_name]["validators"][validator_name]["findings"].append(
                        {
                            "category": finding.category,
                            "severity": finding.severity.value
                            if hasattr(finding.severity, "value")
                            else str(finding.severity),
                            "check_name": finding.check_name,
                            "message": finding.message,
                            "file_path": clean_path,
                            "line_number": finding.line_number,
                            "line_content": finding.line_content,
                            "suggestion": finding.suggestion,
                            "location": finding.location,
                            "confidence": confidence,
                            "metadata": finding.metadata if hasattr(finding, "metadata") else {},
                        }
                    )
                    skills[skill_name]["issue_count"] = skills[skill_name].get("issue_count", 0) + 1

        # Deduplicate findings within each skill/validator
        for skill_data in skills.values():
            # Recompute issue_count after deduplication (use occurrences)
            deduped_count = 0
            for validator_data in skill_data["validators"].values():
                if validator_data["findings"]:
                    validator_data["findings"] = self._deduplicate_findings(validator_data["findings"])
                    deduped_count += sum(f.get("occurrences", 1) for f in validator_data["findings"])
            if deduped_count > 0:
                skill_data["issue_count"] = deduped_count

        # Ensure all skills have issue_count (including passing ones)
        for skill_data in skills.values():
            if "issue_count" not in skill_data:
                skill_data["issue_count"] = 0

        return skills

    @staticmethod
    def _normalize_message(msg: str) -> str:
        """Normalize a finding message for deduplication.

        Strips trailing whitespace/newlines and collapses internal whitespace
        so that near-identical messages (e.g., "Privilege Escalation: .env\\n"
        vs "Privilege Escalation: .env ") are grouped together.
        """
        # Strip and collapse whitespace
        return " ".join((msg or "").split())

    @staticmethod
    def _deduplicate_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Aggregate duplicate findings that share the same message, file, and suggestion.

        Groups findings by (message, file_path, suggestion) and merges duplicates
        into a single entry with an 'occurrences' count and a 'lines' list.
        This dramatically reduces visual noise when the same PII pattern
        (e.g., "Personal macOS user path") fires on many lines in the same file.
        """
        groups: dict[tuple, dict[str, Any]] = {}

        for finding in findings:
            # Normalize message for deduplication (strip trailing whitespace/newlines)
            norm_msg = HTMLReporter._normalize_message(finding.get("message", ""))
            key = (
                norm_msg,
                finding.get("file_path", ""),
                finding.get("suggestion", ""),
            )

            if key in groups:
                groups[key]["occurrences"] += 1
                line_num = finding.get("line_number")
                if line_num and line_num not in groups[key]["lines"]:
                    groups[key]["lines"].append(line_num)
                # Keep highest severity
                sev_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
                existing_sev = sev_order.get(groups[key]["severity"], 0)
                new_sev = sev_order.get(finding.get("severity", "medium"), 0)
                if new_sev > existing_sev:
                    groups[key]["severity"] = finding["severity"]
                # Preserve the first line_content as representative
                if not groups[key].get("line_content") and finding.get("line_content"):
                    groups[key]["line_content"] = finding["line_content"]
                # Track confidence (keep lowest = most suspicious)
                conf_order = {"low": 0, "medium": 1, "high": 2}
                existing_conf = groups[key].get("confidence", "high")
                new_conf = finding.get("confidence", "high")
                if conf_order.get(new_conf, 2) < conf_order.get(existing_conf, 2):
                    groups[key]["confidence"] = new_conf
            else:
                entry = dict(finding)
                entry["message"] = norm_msg  # Store normalized message
                entry["occurrences"] = 1
                line_num = finding.get("line_number")
                entry["lines"] = [line_num] if line_num else []
                # Extract confidence from metadata if present
                metadata = finding.get("metadata", {})
                if isinstance(metadata, dict) and "confidence" in metadata:
                    entry["confidence"] = metadata["confidence"]
                elif "confidence" not in entry:
                    entry["confidence"] = "high"
                groups[key] = entry

        # Sort lines for display
        for group in groups.values():
            group["lines"].sort()

        return list(groups.values())

    @staticmethod
    def _extract_issue_group_key(message: str, category: str) -> str:
        """Extract a grouping key from a finding message.

        For SECURITY findings from skillspector, messages often include the matched
        code snippet after a colon (e.g., "Privilege Escalation: ~/.ssh/id_ed25519").
        We group these by the prefix so all "Privilege Escalation" variants are
        aggregated into one top-issue row in the executive summary.

        For PII and SCHEMA findings, the full normalized message is used as-is.
        """
        norm = HTMLReporter._normalize_message(message)
        if category == "SECURITY" and ": " in norm:
            # Use the category prefix (e.g., "Privilege Escalation", "Data Exfiltration")
            return norm.split(": ", 1)[0]
        return norm

    @staticmethod
    def _compute_issue_key(category: str, group_key: str) -> str:
        """Stable, JS/CSS-safe identifier shared between top-issue rows and findings.

        The HTML report cross-links the executive-summary row, the per-skill
        affected pill, and the finding card itself. We need an identifier we
        can safely embed in ``data-issue-key`` attributes and inline ``onclick``
        JS strings without escaping concerns. Truncated SHA-1 hex meets that
        bar: hex-only, deterministic, and short enough to read in DevTools.
        16 hex chars (~64 bits) is well below collision risk for the few
        dozen unique issue groups a single report ever shows.
        """
        payload = f"{category}::{group_key}".encode()
        return hashlib.sha1(payload, usedforsecurity=False).hexdigest()[:16]

    @staticmethod
    def _compute_top_issues(
        skills: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compute cross-skill aggregation of top issues for the executive summary.

        For SECURITY findings, groups by category prefix (e.g., all
        "Privilege Escalation: ..." variants are merged into one row).
        For PII/SCHEMA findings, groups by exact (normalized) message.

        As a side effect, each underlying finding is tagged with
        ``issue_key`` (matching the row it rolls up to) so the template can
        deep-link from the executive summary directly to the finding card.

        Returns a list of dicts sorted by total_count descending.
        """
        sev_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
        issue_map: dict[tuple, dict[str, Any]] = {}

        for skill_name, skill_data in skills.items():
            for validator_data in skill_data["validators"].values():
                for finding in validator_data.get("findings", []):
                    msg = finding.get("message", "")
                    category = finding.get("category", "")
                    group_key = HTMLReporter._extract_issue_group_key(msg, category)
                    issue_key = HTMLReporter._compute_issue_key(category, group_key)
                    # Tag the finding so the template can attach a matching
                    # ``data-issue-key`` to its card and JS can locate it.
                    finding["issue_key"] = issue_key
                    key = (group_key, category)

                    occurrences = finding.get("occurrences", 1)

                    if key in issue_map:
                        issue_map[key]["total_count"] += occurrences
                        if skill_name not in issue_map[key]["skills_affected"]:
                            issue_map[key]["skills_affected"].append(skill_name)
                        # Keep highest severity
                        existing = sev_order.get(issue_map[key]["severity"], 0)
                        new = sev_order.get(finding.get("severity", "medium"), 0)
                        if new > existing:
                            issue_map[key]["severity"] = finding["severity"]
                    else:
                        issue_map[key] = {
                            "message": group_key,
                            "severity": finding.get("severity", "medium"),
                            "total_count": occurrences,
                            "skills_affected": [skill_name],
                            "category": category,
                            "suggestion": finding.get("suggestion", ""),
                            "issue_key": issue_key,
                        }

        return sorted(issue_map.values(), key=lambda x: x["total_count"], reverse=True)

    @staticmethod
    def _extract_contributors(
        skills: dict[str, dict[str, Any]],
        results: list[ValidationResult],
    ) -> list[dict[str, Any]]:
        """Extract contributor summary by mapping authors to their content items.

        Searches for author information in:
        1. Success details metadata (for passed skills — author_format check)
        2. Finding metadata (for failed skills — current_author key)

        Returns a list of contributor dicts sorted by issue count descending:
        [
            {
                "author": "John Doe <john@example.com>",
                "items": [{"name": "skill-x", "passed": True, "issue_count": 0}, ...],
                "total_items": 3,
                "passed_count": 2,
                "failed_count": 1,
                "total_issues": 5,
            },
        ]
        """
        # Build skill_name -> author mapping from validation results
        skill_authors: dict[str, str] = {}

        # Track author from unprefixed author_format (single-skill runs)
        single_skill_author: str | None = None

        for result in results:
            # Check success_details for author info (passed skills)
            for detail in result.success_details:
                skill_name = detail.check_name
                if skill_name in skills and skill_name not in skill_authors:
                    # Look in nested checks metadata for author_format
                    checks = detail.metadata.get("checks", [])
                    for check in checks:
                        if check.get("name") == "author_format":
                            msg = check.get("description", "")
                            # Message format: "Valid author format: Name <email>"
                            if ": " in msg:
                                author = msg.split(": ", 1)[1]
                                skill_authors[skill_name] = author

                # Handle prefixed success details from failed skills:
                # check_name format: "[skill-name] author_format"
                if detail.check_name.startswith("[") and "] author_format" in detail.check_name:
                    sname = detail.check_name[1 : detail.check_name.index("]")]
                    if sname in skills and sname not in skill_authors:
                        msg = detail.message
                        if ": " in msg:
                            author = msg.split(": ", 1)[1]
                            skill_authors[sname] = author

                # Handle unprefixed author_format (single-skill runs):
                # check_name is just "author_format" with author in message
                if (
                    detail.check_name == "author_format"
                    and not detail.check_name.startswith("[")
                    and single_skill_author is None
                ):
                    msg = detail.message
                    if ": " in msg:
                        single_skill_author = msg.split(": ", 1)[1]

            # Check findings for author info (failed skills with author metadata)
            for finding in result.findings:
                if finding.check_name == "author_format" and finding.metadata:
                    current_author = finding.metadata.get("current_author")
                    if current_author:
                        # Extract skill name from prefixed file_path "[skill-name] path"
                        fp = finding.file_path
                        if fp.startswith("[") and "]" in fp:
                            sname = fp[1 : fp.index("]")]
                            if sname in skills:
                                skill_authors[sname] = current_author

        # For single-skill runs, apply the unprefixed author to all skills
        # that don't already have an author assigned
        if single_skill_author:
            for skill_name in skills:
                if skill_name not in skill_authors:
                    skill_authors[skill_name] = single_skill_author

        # Group skills by author
        author_map: dict[str, list[dict[str, Any]]] = {}
        for skill_name, skill_data in skills.items():
            author = skill_authors.get(skill_name, "Unknown")
            if author not in author_map:
                author_map[author] = []
            author_map[author].append(
                {
                    "name": skill_name,
                    "passed": skill_data["passed"],
                    "issue_count": skill_data.get("issue_count", 0),
                }
            )

        # Build contributor list
        contributors = []
        for author, items in author_map.items():
            passed_count = sum(1 for i in items if i["passed"])
            total_issues = sum(i["issue_count"] for i in items)
            contributors.append(
                {
                    "author": author,
                    "items": sorted(items, key=lambda x: (x["passed"], x["name"])),
                    "total_items": len(items),
                    "passed_count": passed_count,
                    "failed_count": len(items) - passed_count,
                    "total_issues": total_issues,
                }
            )

        # Sort: most issues first, then by name
        contributors.sort(key=lambda c: (-c["total_issues"], c["author"]))
        return contributors

    @staticmethod
    def _compute_target_display(url: str) -> str:
        """Compute a short display label for a target URL.

        For repository URLs like
        ``https://github.com/example/project/tree/HEAD/skills/my-skill``
        returns ``ai_tools/ai_rules / skills/my-skill``.

        Falls back to the path portion of the URL, or the raw string for
        non-URL values.
        """
        if not url or not url.startswith("https://"):
            return url or ""
        parsed = urlparse(url)
        path = parsed.path.lstrip("/")
        if "/-/tree/" in path:
            repo_part, _, rest = path.partition("/-/tree/")
            branch_and_path = rest.split("/", 1)
            if len(branch_and_path) == 2:
                return f"{repo_part} / {branch_and_path[1]}"
            return repo_part
        return path or url

    @staticmethod
    def _compute_friendly_skill_label(target: str | None) -> str:
        """Strip repo / filesystem prefixes so the hero card shows ``skills/<name>``.

        The header keeps the full clickable path/URL for traceability; the hero
        card only gets a short, audience-friendly label so reviewers don't have
        to skim ``/workspaces/example-project/.../skills/log-analyzer`` to find
        the skill they care about.

        Resolution order — first match wins:

        1. ``team-skills/<team>/<name>`` is preserved verbatim. The team prefix
           carries useful provenance ("which team owns this skill") so we keep
           it instead of collapsing to the bare name.
        2. ``skills/<name>`` is preserved verbatim. The single ``skills/``
           segment makes it obvious the artifact is a skill (not a rule or
           workflow) without leaking the surrounding repo / worktree path.
        3. Falls back to the basename — best effort when no canonical
           ``skills/`` or ``team-skills/`` segment is present (e.g. validating
           an ad-hoc directory outside the standard SkillEvaluator layout).
        4. Empty string when ``target`` is ``None`` / empty so callers can
           defensibly chain ``label or fallback``.
        """
        if not target:
            return ""
        path = target
        if path.startswith("https://"):
            parsed = urlparse(path)
            path = parsed.path.lstrip("/")
            if "/-/tree/" in path:
                _repo, _, rest = path.partition("/-/tree/")
                branch_and_path = rest.split("/", 1)
                if len(branch_and_path) == 2:
                    path = branch_and_path[1]
        for marker in ("team-skills/", "skills/"):
            idx = path.rfind(marker)
            if idx >= 0:
                return path[idx:]
        try:
            return Path(path).name or path
        except (ValueError, OSError):
            return path

    # Validator names produced by ``commands.validate`` for each tier.  Used
    # to bucket combined run results back into per-tier summaries for the
    # hero card so the chip for "Tier 1" reflects only Tier 1 validators
    # instead of leaking Tier 2 / Tier 3 stats from the global totals.
    @classmethod
    def _split_results_by_tier(
        cls, results: list[ValidationResult]
    ) -> tuple[list[ValidationResult], list[ValidationResult], list[ValidationResult]]:
        """Bucket a flat results list into ``(tier1, tier2, tier3)`` slices.

        ``commands.validate`` concatenates results in tier order before handing
        them to the reporter, so we restore tier identity here by validator
        name rather than relying on list slicing (which would silently break
        if a tier ever produced zero results).
        """
        tier1: list[ValidationResult] = []
        tier2: list[ValidationResult] = []
        tier3: list[ValidationResult] = []
        for r in results:
            if is_tier3_result(r):
                tier3.append(r)
            elif is_tier2_result(r):
                tier2.append(r)
            else:
                tier1.append(r)
        return tier1, tier2, tier3

    @staticmethod
    def _compute_tier_summary(results: list[ValidationResult]) -> dict[str, Any]:
        """Compact stats for a single tier — what the hero chip displays.

        Returns ``passed`` (boolean), executed/pass/skip counts,
        ``issue_count`` (sum of findings), and per-severity totals.
        ``total == 0`` lets the template hide the chip entirely without an
        extra "did this tier run?" predicate.
        """
        total = len(results)
        skipped_count = sum(1 for r in results if is_cleanly_skipped(r))
        executed_count = total - skipped_count
        passed_count = sum(1 for r in results if r.passed and not r.is_incomplete and not is_cleanly_skipped(r))
        advisory_skipped_count = sum(1 for r in results if is_advisory_agent_eval_skip(r))
        failed_count = sum(1 for r in results if not r.passed and not is_cleanly_skipped(r))
        incomplete_count = sum(1 for r in results if r.is_incomplete)
        issue_count = 0
        critical = high = medium = low = 0
        for r in results:
            issue_count += len(r.findings or [])
            for f in r.findings or []:
                sev = f.severity.value if hasattr(f.severity, "value") else str(f.severity).lower()
                if sev == "critical":
                    critical += 1
                elif sev == "high":
                    high += 1
                elif sev == "medium":
                    medium += 1
                elif sev == "low":
                    low += 1
        return {
            "total": total,
            "executed_count": executed_count,
            "passed_count": passed_count,
            "skipped_count": skipped_count,
            "advisory_skipped_count": advisory_skipped_count,
            "failed_count": failed_count,
            "issue_count": issue_count,
            "all_passed": executed_count > 0 and failed_count == 0 and incomplete_count == 0,
            "incomplete_count": incomplete_count,
            "critical": critical,
            "high": high,
            "medium": medium,
            "low": low,
        }

    def _results_to_dict(self, results: list[ValidationResult]) -> list[dict[str, Any]]:
        output = []
        for result in results:
            skipped = is_cleanly_skipped(result)
            result_dict = {
                "validator_name": result.validator_name,
                "validator_description": result.validator_description,
                "passed": result.passed,
                "status": "skipped" if skipped else result.status,
                "skipped": skipped,
                "skip_reason": get_skip_reason(result) if skipped else None,
                "incomplete_scans": result.incomplete_scans,
                "summary": {
                    "files_scanned": result.summary.files_scanned,
                    "checks_performed": result.summary.checks_performed,
                    "errors": result.summary.errors,
                    "warnings": result.summary.warnings,
                    "critical_count": result.summary.critical_count,
                    "high_count": result.summary.high_count,
                    "medium_count": result.summary.medium_count,
                    "low_count": result.summary.low_count,
                },
                "findings": [],
                "success_details": [],
                "messages": result.messages,
                "errors": result.errors,
                "warnings": result.warnings,
            }
            publication_target = result_publication_target_dict(result)
            if publication_target is not None:
                result_dict["publication_target"] = publication_target
            publication_target_conflict = result_publication_target_conflict_marker(result)
            if publication_target_conflict is not None:
                result_dict["publication_target_conflict"] = publication_target_conflict
            publication_evidence = result_publication_evidence_dict(result)
            if publication_evidence is not None:
                result_dict["publication_evidence"] = publication_evidence
            for finding in result.findings:
                result_dict["findings"].append(
                    {
                        "category": finding.category,
                        "severity": finding.severity.value,
                        "check_name": finding.check_name,
                        "message": finding.message,
                        "file_path": finding.file_path,
                        "line_number": finding.line_number,
                        "line_content": finding.line_content,
                        "suggestion": finding.suggestion,
                        "location": finding.location,
                        "metadata": finding.metadata,
                    }
                )
            for detail in result.success_details:
                result_dict["success_details"].append(
                    {
                        "check_name": detail.check_name,
                        "message": detail.message,
                        "metadata": detail.metadata,
                    }
                )
            output.append(result_dict)
        return output

    @staticmethod
    def _tier3_report_data(results: list[ValidationResult]) -> dict[str, Any] | None:
        """Return safe Tier 3 display data without certifying partial evidence."""
        if not results:
            return None

        def fallback(verdict: str, execution_status: str) -> dict[str, Any]:
            messages = [
                message for result in results for message in (*result.errors, *result.warnings, *result.messages)
            ]
            if execution_status == "skipped":
                skipped_result = next(result for result in results if is_cleanly_skipped(result))
                message = get_skip_reason(skipped_result)
            else:
                incomplete_scans = list(dict.fromkeys(tool for result in results for tool in result.incomplete_scans))
                if incomplete_scans:
                    message = f"Missing trustworthy evidence from {', '.join(incomplete_scans)}."
                else:
                    message = messages[0] if messages else "Tier 3 did not provide canonical evaluation details."
            provenance = {"source": "validation_result", "message": message}
            if execution_status == "skipped":
                provenance.update({"reason": "skipped", "advisory": True})
            execution_errors = [error for result in results for error in result.errors]
            return {
                "schema_version": "2.0",
                "summary": {
                    "schema_version": "2.0",
                    "verdict": verdict,
                    "skill_name": "",
                    "best_agent": "",
                    "agents_run": [],
                    "overall_score": None,
                    "overall_lift": None,
                    "environment": None,
                    "runtime_seconds": 0.0,
                    "execution_status": execution_status,
                    "execution_errors": execution_errors,
                    "expected_attempts": 0,
                    "scored_attempts": 0,
                },
                "skill_name": "",
                "verdict": verdict,
                "best_agent": "",
                "agents_run": [],
                "environment": None,
                "overall_score": None,
                "overall_lift": None,
                "composite_lift": None,
                "execution_status": execution_status,
                "execution_errors": execution_errors,
                "expected_attempts": 0,
                "scored_attempts": 0,
                "runtime_seconds": 0.0,
                "agents": {},
                "dimensions": [],
                "evaluators": {},
                "evaluator_cards": [],
                "cases": [],
                "trials": [],
                "insights": {},
                "conclusions": [],
                "recommendations": [],
                "suggestions": messages,
                "suggestions_v2": [],
                "metric_ids": [],
                "metric_labels": {},
                "attempt_policy": {},
                "dataset": [],
                "dataset_summary": {
                    "total_tasks": 0,
                    "positive_tasks": 0,
                    "negative_tasks": 0,
                    "unclassified_tasks": 0,
                    "source": "unavailable",
                },
                "verdict_policy": {},
                "provenance": provenance,
            }

        def payload_summary(candidate: dict[str, Any]) -> dict[str, Any]:
            summary = candidate.get("summary")
            return summary if isinstance(summary, dict) else {}

        def payload_verdict(candidate: dict[str, Any]) -> str:
            summary = payload_summary(candidate)
            value = candidate.get("verdict") or summary.get("verdict") or ""
            return value.lower() if isinstance(value, str) else ""

        def payload_execution_status(candidate: dict[str, Any]) -> str:
            summary = payload_summary(candidate)
            value = candidate.get("execution_status") or summary.get("execution_status") or ""
            return value.lower() if isinstance(value, str) else ""

        def result_has_execution_evidence(result: ValidationResult) -> bool:
            return bool(result.success_details or result.findings or result.summary.checks_performed > 0)

        def payload_has_agent_evidence(candidate: dict[str, Any]) -> bool:
            """Require a succeeded, scored agent with a positive attempt count."""
            summary = payload_summary(candidate)
            agents = candidate.get("agents")
            if not isinstance(agents, dict):
                return False
            payload_attempts = max(
                _nonnegative_count(candidate.get("scored_attempts")),
                _nonnegative_count(summary.get("scored_attempts")),
            )
            for agent in agents.values():
                if not isinstance(agent, dict):
                    continue
                status = agent.get("execution_status")
                if not isinstance(status, str) or status.lower() != "succeeded":
                    continue
                score = _finite_number(agent.get("with_skill"), minimum=0.0, maximum=1.0)
                if score is None:
                    score = _finite_number(agent.get("overall_score"), minimum=0.0, maximum=1.0)
                attempts = max(payload_attempts, _nonnegative_count(agent.get("scored_attempts")))
                if score is not None and attempts > 0:
                    return True
            return False

        def payload_has_execution_evidence(
            result: ValidationResult,
            candidate: dict[str, Any],
        ) -> bool:
            if agent_eval_publication_evidence_complete(candidate):
                return True
            if payload_verdict(candidate) not in {"pass", "neutral", "fail"}:
                return False
            execution_status = payload_execution_status(candidate)
            if execution_status == "succeeded":
                return payload_has_agent_evidence(candidate)
            return not execution_status and result_has_execution_evidence(result)

        selected = select_agent_eval_candidate(results)
        payload_result, payload = selected if selected is not None else (None, None)
        serialization_issue = agent_eval_report_serialization_issue(payload) if payload is not None else None
        rejected_assessment = (
            assess_tier3_evidence(results, payload) if payload is not None and serialization_issue is not None else None
        )
        bounded_rejected_has_execution_evidence = False
        if payload is not None and serialization_issue is not None:
            bounded_rejected_payload = _json_safe_tier3_payload(payload)
            bounded_evidence_candidate = dict(bounded_rejected_payload)
            bounded_evidence_candidate.pop("_serialization_truncated", None)
            bounded_rejected_has_execution_evidence = payload_has_execution_evidence(
                payload_result,
                bounded_evidence_candidate,
            )
        selected_verdict = (
            rejected_assessment.verdict
            if rejected_assessment is not None
            else payload_verdict(payload)
            if payload is not None
            else ""
        )
        selected_execution_status = (
            rejected_assessment.execution_status
            if rejected_assessment is not None
            else payload_execution_status(payload)
            if payload is not None
            else ""
        )
        has_trustworthy_payload = bool(
            payload_result is not None
            and payload is not None
            and serialization_issue is None
            and payload_has_execution_evidence(payload_result, payload)
        )
        has_explicit_failure_payload = bool(
            payload_result is not None
            and payload is not None
            and (
                rejected_assessment.status == "fail"
                if rejected_assessment is not None
                else selected_execution_status in {"failed", "incomplete"}
            )
        )

        def normalized_payload(
            *,
            forced_verdict: str | None = None,
            forced_execution_status: str | None = None,
            scrub_untrusted_evidence: bool = False,
        ) -> dict[str, Any]:
            assert payload is not None
            verdict = selected_verdict or "incomplete"
            execution_status = selected_execution_status or "incomplete"
            display = fallback(forced_verdict or verdict, forced_execution_status or execution_status)
            normalized = normalize_agent_eval_harbor_links(_sanitize_tier3_display_payload(payload))
            normalized_summary = normalized.get("summary") if isinstance(normalized.get("summary"), dict) else {}
            display.update(normalized)
            display["summary"] = {**display["summary"], **normalized_summary}

            def merge_strings(value: object, additions: list[str]) -> list[str]:
                existing = value if isinstance(value, list) else []
                return list(dict.fromkeys(item for item in (*existing, *additions) if isinstance(item, str) and item))

            if forced_verdict is not None:
                display["verdict"] = forced_verdict
                display["summary"]["verdict"] = forced_verdict
            if forced_execution_status is not None:
                display["execution_status"] = forced_execution_status
                display["summary"]["execution_status"] = forced_execution_status
            display = _sanitize_tier3_display_payload(display)
            if scrub_untrusted_evidence:
                preserved = display
                display = fallback(
                    forced_verdict or verdict,
                    forced_execution_status or execution_status,
                )
                display["agents"] = _scrub_untrusted_tier3_agents(preserved.get("agents"))
                display["agents_run"] = list(display["agents"])
                for key in (
                    "schema_version",
                    "skill_name",
                    "publication_target",
                    "publication_target_conflict",
                    "run_id",
                    "environment",
                    "evaluated_at",
                    "evaluator_version",
                    "benchmark_policy",
                    "attempt_policy",
                    "provenance",
                    "harbor_viewer",
                ):
                    if key in preserved:
                        display[key] = preserved[key]
                for key in (
                    "schema_version",
                    "skill_name",
                    "publication_target",
                    "publication_target_conflict",
                    "run_id",
                    "environment",
                    "evaluated_at",
                    "evaluator_version",
                    "benchmark_policy",
                ):
                    if key in preserved["summary"]:
                        display["summary"][key] = preserved["summary"][key]
                if "harbor_viewer" in preserved["summary"]:
                    display["summary"]["harbor_viewer"] = preserved["summary"]["harbor_viewer"]
                if preserved.get("_serialization_truncated") is True:
                    display["_serialization_truncated"] = True

            result_errors = [error for result in results for error in result.errors]
            result_diagnostics = [
                message for result in results for message in (*result.errors, *result.warnings, *result.messages)
            ]
            display["execution_errors"] = merge_strings(display.get("execution_errors"), result_errors)
            display["summary"]["execution_errors"] = merge_strings(
                display["summary"].get("execution_errors"),
                result_errors,
            )
            display["suggestions"] = merge_strings(display.get("suggestions"), result_diagnostics)
            return _sanitize_tier3_display_payload(display)

        # Incomplete scanner evidence takes precedence over failures, matching
        # ValidationResult.status and the publication benchmark.
        if any(result.is_incomplete for result in results):
            if payload is not None:
                return normalized_payload(
                    forced_verdict="incomplete",
                    forced_execution_status="incomplete",
                    scrub_untrusted_evidence=not has_trustworthy_payload,
                )
            return fallback("incomplete", "incomplete")

        # Aggregate all Tier 3 results before falling back so the display does
        # not depend on result ordering.
        if any(not result.passed and not is_cleanly_skipped(result) for result in results):
            if payload is not None:
                return normalized_payload(
                    forced_verdict="fail",
                    forced_execution_status="failed",
                    scrub_untrusted_evidence=not (has_trustworthy_payload or has_explicit_failure_payload),
                )
            return fallback("fail", "failed")

        if all(is_cleanly_skipped(result) for result in results):
            if payload is not None:
                return normalized_payload(
                    forced_verdict="skipped",
                    forced_execution_status="skipped",
                    scrub_untrusted_evidence=True,
                )
            return fallback("skipped", "skipped")

        if payload is not None:
            verdict = selected_verdict
            execution_status = selected_execution_status
            if serialization_issue is not None:
                if rejected_assessment is not None and rejected_assessment.status == "fail":
                    return normalized_payload(
                        forced_verdict="fail",
                        forced_execution_status="failed",
                        scrub_untrusted_evidence=True,
                    )
                return normalized_payload(
                    forced_verdict=None if bounded_rejected_has_execution_evidence else "incomplete",
                    forced_execution_status=None if bounded_rejected_has_execution_evidence else "incomplete",
                    scrub_untrusted_evidence=True,
                )
            if verdict == "fail" or execution_status == "failed":
                return normalized_payload(
                    forced_verdict="fail",
                    forced_execution_status="failed",
                    scrub_untrusted_evidence=not (has_trustworthy_payload or has_explicit_failure_payload),
                )
            if execution_status == "incomplete":
                return normalized_payload(
                    forced_verdict="incomplete",
                    forced_execution_status="incomplete",
                    scrub_untrusted_evidence=not has_trustworthy_payload,
                )
            if has_trustworthy_payload:
                return normalized_payload(
                    forced_execution_status="incomplete" if not execution_status else None,
                )
            return normalized_payload(
                forced_verdict="incomplete",
                forced_execution_status="incomplete",
                scrub_untrusted_evidence=True,
            )

        # A default-passed bare result or partial payload is not proof that
        # Tier 3 completed.
        return fallback("incomplete", "incomplete")

    def render(self, result: ValidationResult) -> str:
        return self.render_all([result])

    def render_all(self, results: list[ValidationResult]) -> str:
        tier1_results, tier2_results, tier3_results = self._split_results_by_tier(results)
        tier3_data = self._tier3_report_data(tier3_results)
        # Preserve the raw policy/provenance assessment, then let the emitted
        # payload make the decision only more conservative. Display shaping
        # must never invent a successful run or erase a persisted waiver.
        raw_publication = assess_publication(results, expected_skill_name=self.expected_skill_name)
        emitted_publication = assess_publication(
            results,
            tier3_data,
            expected_skill_name=self.expected_skill_name,
        )
        publication = max(
            (raw_publication, emitted_publication),
            key=lambda assessment: {"pass": 0, "neutral": 1, "incomplete": 2, "fail": 3}.get(
                assessment.status,
                3,
            ),
        )

        all_passed = all(passes_required_gate(r) for r in results)
        has_incomplete = any(r.is_incomplete for r in results)
        overall_status = "incomplete" if has_incomplete else "passed" if all_passed else "failed"
        total_errors = sum(r.summary.errors for r in results)
        total_warnings = sum(r.summary.warnings for r in results)
        skipped_count = sum(1 for r in results if is_cleanly_skipped(r))
        passed_count = sum(1 for r in results if r.passed and not r.is_incomplete and not is_cleanly_skipped(r))
        advisory_skipped_count = sum(1 for r in results if is_advisory_agent_eval_skip(r))
        failed_count = sum(1 for r in results if not passes_required_gate(r))
        total_issues = total_errors + total_warnings
        total_validators = len(results)
        executed_count = total_validators - skipped_count
        pass_percentage = round((passed_count / executed_count * 100) if executed_count > 0 else 0, 1)

        timestamp = ""
        if self.include_timestamp:
            timestamp = datetime.now(tz=UTC).strftime("%B %d, %Y at %I:%M %p UTC")

        # Reorganize results by skill for the new view
        skills_by_name = self._reorganize_by_skill(results)

        # Count total skills
        total_skills = len(skills_by_name)
        passed_skills = sum(1 for s in skills_by_name.values() if s["passed"])
        failed_skills = total_skills - passed_skills

        # Compute cross-skill top issues for executive summary
        top_issues = self._compute_top_issues(skills_by_name)

        # Extract contributor summary (author -> skills mapping)
        contributors = self._extract_contributors(skills_by_name, results)

        # Compute severity breakdown from actual findings (summary counts may be
        # incomplete when merge_with_prefix is used, so count from findings directly)
        total_critical = 0
        total_high = 0
        total_medium = 0
        total_low = 0
        for r in results:
            for f in r.findings:
                sev = f.severity.value if hasattr(f.severity, "value") else str(f.severity).lower()
                if sev == "critical":
                    total_critical += 1
                elif sev == "high":
                    total_high += 1
                elif sev == "medium":
                    total_medium += 1
                elif sev == "low":
                    total_low += 1

        target_display = self._compute_target_display(self.target_path or "")
        # Friendly hero label (``skills/<name>`` or ``team-skills/<team>/<name>``)
        # — strips repo / filesystem prefixes so the hero stays readable when
        # ``target_path`` is something like
        # ``/workspaces/example-project/skills/log-analyzer``. The full path
        # remains in the top-right header (``Target: ...``) for traceability.
        friendly_label = self._compute_friendly_skill_label(self.target_path or "")
        # Per-tier summaries so the hero card chips can show "Tier 1: 6/7
        # passed (0 critical)" etc. without leaking Tier 2 / Tier 3 stats
        # from the global aggregate counters.
        tier1_summary = self._compute_tier_summary(tier1_results)
        tier2_summary = self._compute_tier_summary(tier2_results)
        tier3_summary = self._compute_tier_summary(tier3_results)
        tier3_preview, tier3_preview_notice = _bounded_tier3_preview(tier3_data)
        tier3_canonical_data, tier3_canonical_encoding = _canonical_tier3_embed(tier3_data)
        tier3_truncation = tier3_data.get("report_truncation", {}) if isinstance(tier3_data, dict) else {}

        # Keep the Tier 1 dashboard scoped to Tier 1. Tier 2 and Tier 3 have
        # dedicated tabs; including an advisory Tier 3 skip here would make
        # the dashboard report a failure even though it does not gate Tier 1.
        tier1_display_results = tier1_results
        tier1_skills = self._reorganize_by_skill(tier1_display_results)
        tier1_top_issues = self._compute_top_issues(tier1_skills)
        tier1_contributors = self._extract_contributors(tier1_skills, tier1_display_results)
        tier1_display_skipped = sum(1 for result in tier1_display_results if is_cleanly_skipped(result))
        tier1_display_total = len(tier1_display_results) - tier1_display_skipped
        tier1_display_passed = sum(
            1
            for result in tier1_display_results
            if result.passed and not result.is_incomplete and not is_cleanly_skipped(result)
        )
        tier1_display_failed = sum(
            1 for result in tier1_display_results if not result.passed and not is_cleanly_skipped(result)
        )
        tier1_display_total_skills = len(tier1_skills)
        tier1_display_passed_skills = sum(1 for skill in tier1_skills.values() if skill["passed"])
        tier1_display_summary = {
            "total_validators": tier1_display_total,
            "passed_count": tier1_display_passed,
            "skipped_count": tier1_display_skipped,
            "failed_count": tier1_display_failed,
            "total_issues": sum(result.summary.errors + result.summary.warnings for result in tier1_display_results),
            "pass_percentage": round(
                (tier1_display_passed / tier1_display_total * 100) if tier1_display_total else 0,
                1,
            ),
            "total_skills": tier1_display_total_skills,
            "passed_skills": tier1_display_passed_skills,
            "failed_skills": tier1_display_total_skills - tier1_display_passed_skills,
        }

        # Keep report gating aligned with the exact policy used for the CLI
        # exit code. Results produced outside ``validate`` retain the public
        # defaults: Tier 1/Tier 2 block and Tier 3 is advisory.
        blocking_results: list[ValidationResult] = []
        advisory_results: list[ValidationResult] = []
        blocking_tiers: list[str] = []
        advisory_tiers: list[str] = []
        for tier_name, tier_results, default_blocking in (
            ("tier1", tier1_results, True),
            ("tier2", tier2_results, True),
            ("tier3", tier3_results, False),
        ):
            if not tier_results:
                (blocking_tiers if default_blocking else advisory_tiers).append(tier_name)
                continue
            tier_blocks = False
            tier_advises = False
            for result in tier_results:
                result_gating = result.metadata.get("gating") if isinstance(result.metadata, dict) else None
                is_blocking = (
                    bool(result_gating.get("blocking", default_blocking))
                    if isinstance(result_gating, dict)
                    else default_blocking
                )
                if is_blocking:
                    blocking_results.append(result)
                    tier_blocks = True
                else:
                    advisory_results.append(result)
                    tier_advises = True
            if tier_blocks:
                blocking_tiers.append(tier_name)
            if tier_advises:
                advisory_tiers.append(tier_name)

        def _severity_totals(tier_results: list[ValidationResult]) -> dict[str, int]:
            totals = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for result in tier_results:
                for finding in result.findings:
                    severity = (
                        finding.severity.value if hasattr(finding.severity, "value") else str(finding.severity).lower()
                    )
                    if severity in totals:
                        totals[severity] += 1
            return totals

        blocking_totals = _severity_totals(blocking_results)
        advisory_totals = _severity_totals(advisory_results)
        blocking_critical = blocking_totals["critical"]
        blocking_high = blocking_totals["high"]
        blocking_medium = blocking_totals["medium"]
        blocking_low = blocking_totals["low"]
        advisory_critical = advisory_totals["critical"]
        advisory_high = advisory_totals["high"]
        advisory_medium = advisory_totals["medium"]
        advisory_low = advisory_totals["low"]
        gating = {
            "blocking_tiers": blocking_tiers,
            "advisory_tiers": advisory_tiers,
            "blocking": blocking_totals,
            "advisory": advisory_totals,
            "blocking_findings": blocking_totals["critical"] + blocking_totals["high"],
            "would_block": any(not passes_required_gate(result) for result in blocking_results),
        }

        # Extract quality scores from results for per-skill quality display
        quality_scores_by_skill: dict[str, dict] = {}
        for r in results:
            qs = r.metadata.get("quality_scores") if r.metadata else None
            if qs and qs.get("dimensions"):
                sname = qs.get("skill_name", "")
                if sname:
                    quality_scores_by_skill[sname] = qs
            # Also check folder-level aggregation
            qs_all = r.metadata.get("quality_scores_all") if r.metadata else None
            if qs_all:
                for q in qs_all:
                    sname = q.get("skill_name", "")
                    if sname and sname not in quality_scores_by_skill:
                        quality_scores_by_skill[sname] = q

        def _attach_quality_scores(skills: dict[str, dict[str, Any]]) -> None:
            # Attach quality scores to per-skill data for the template.
            # Two strategies: (1) exact name match for folder-of-skills mode,
            # (2) attach to the FIRST entry only for single-skill mode (where
            # _reorganize_by_skill keys are check names, not skill names).
            matched = set()
            for sname, sdata in skills.items():
                if sname in quality_scores_by_skill:
                    sdata["quality"] = quality_scores_by_skill[sname]
                    matched.add(sname)

            # If no exact matches found, the report is for a single skill whose
            # name doesn't appear as a key. Attach to the first entry only to
            # avoid duplicating the quality panel across every check entry.
            if not matched and quality_scores_by_skill and skills:
                single_qs = next(iter(quality_scores_by_skill.values()))
                first_key = next(iter(skills))
                skills[first_key]["quality"] = single_qs

        _attach_quality_scores(skills_by_name)
        _attach_quality_scores(tier1_skills)

        report_data = {
            "title": self.title,
            "timestamp": timestamp,
            "version": __version__,
            "target_path": self.target_path,
            "target_display": target_display,
            "summary": {
                "all_passed": all_passed,
                "status": overall_status,
                "incomplete_scans": list(dict.fromkeys(tool for result in results for tool in result.incomplete_scans)),
                "total_validators": total_validators,
                "executed_count": executed_count,
                "passed_count": passed_count,
                "skipped_count": skipped_count,
                "advisory_skipped_count": advisory_skipped_count,
                "failed_count": failed_count,
                "total_issues": total_issues,
                "pass_percentage": pass_percentage,
                "total_skills": total_skills,
                "passed_skills": passed_skills,
                "failed_skills": failed_skills,
                "publication_status": publication.status,
            },
            "results": self._results_to_dict(results),
            "skills": skills_by_name,
            "top_issues": top_issues,
            "contributors": contributors,
            "quality_scores": quality_scores_by_skill,
            # Tier 3 is embedded once in ``#tier3-full``. The export helper
            # resolves this reference at download time, avoiding a second full
            # copy inside ``#report-data``.
            "tier3": {"$ref": "#tier3-full"} if tier3_data else None,
            "gating": gating,
            "publication": {
                "status": publication.status,
                "eligible": publication.status == "pass",
                "reasons": list(publication.reasons),
                "tier3": {
                    "status": publication.tier3.status,
                    "evidence_complete": publication.tier3.evidence_complete,
                    "reason": publication.tier3.reason,
                },
            },
        }
        report_json = (
            json.dumps(report_data, indent=2, allow_nan=False)
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("&", "\\u0026")
        )

        # Dynamically add Tier 3 tab when agent eval data is present
        active_tabs = list(self.tabs)
        if not self._tabs_explicit and tier2_results and not any(tab["id"] == "tier2" for tab in active_tabs):
            active_tabs.append({"id": "tier2", "label": "Tier 2: Deduplication"})
        if not self._tabs_explicit and tier2_results and not tier1_display_results:
            active_tabs = [tab for tab in active_tabs if tab["id"] != "tier1"]
        if tier3_data and not any(t["id"] == "tier3" for t in active_tabs):
            active_tabs.append({"id": "tier3", "label": "Tier 3: Live Agent Evaluation"})

        # When the run only produced agent-eval results (no real Tier 1 validators
        # ran), drop the Tier 1 tab entirely so the report opens on the tier
        # that actually has data. This mirrors the user experience for
        # ``skill-evaluator agent-eval`` and ``skill-evaluator agent-eval-report``: there is
        # no "Tier 1: Security and Static Validation" content to show.
        agent_eval_only = bool(results) and all(getattr(r, "validator_name", None) == "AGENT_EVAL" for r in results)
        if agent_eval_only and tier3_data:
            active_tabs = [t for t in active_tabs if t["id"] != "tier1"]

        template = self._env.get_template("report.html.j2")
        cl = self.content_label
        return template.render(
            title=self.title,
            timestamp=timestamp,
            version=__version__,
            target_path=self.target_path,
            target_display=target_display,
            friendly_label=friendly_label,
            profile=self.profile or self._infer_profile_from_results(results),
            all_passed=all_passed,
            has_incomplete=has_incomplete,
            overall_status=overall_status,
            publication_status=publication.status,
            publication_reasons=publication.reasons,
            tier3_effective_status=publication.tier3.status,
            total_validators=total_validators,
            executed_count=executed_count,
            passed_count=passed_count,
            skipped_count=skipped_count,
            advisory_skipped_count=advisory_skipped_count,
            failed_count=failed_count,
            total_issues=total_issues,
            pass_percentage=pass_percentage,
            total_skills=total_skills,
            passed_skills=passed_skills,
            failed_skills=failed_skills,
            results=results,
            skills=skills_by_name,
            top_issues=top_issues,
            total_critical=total_critical,
            total_high=total_high,
            total_medium=total_medium,
            total_low=total_low,
            blocking_critical=blocking_critical,
            blocking_high=blocking_high,
            blocking_medium=blocking_medium,
            blocking_low=blocking_low,
            advisory_critical=advisory_critical,
            advisory_high=advisory_high,
            advisory_medium=advisory_medium,
            advisory_low=advisory_low,
            gating=gating,
            icons=self.ICONS,
            tabs=active_tabs,
            contributors=contributors,
            report_json=report_json,
            content_label=cl,
            content_label_plural=cl + "s",
            quality_scores=quality_scores_by_skill,
            tier3=tier3_preview,
            tier3_canonical_data=tier3_canonical_data,
            tier3_canonical_encoding=tier3_canonical_encoding,
            tier3_truncation=tier3_truncation,
            tier3_preview_notice=tier3_preview_notice,
            tier1_summary=tier1_summary,
            tier2_summary=tier2_summary,
            tier3_summary=tier3_summary,
            tier1_display_results=tier1_display_results,
            tier1_skills=tier1_skills,
            tier1_top_issues=tier1_top_issues,
            tier1_contributors=tier1_contributors,
            tier1_display_summary=tier1_display_summary,
            tier2_results=tier2_results,
            tier3_lift_pass_threshold=TIER3_LIFT_PASS_THRESHOLD,
            tier3_lift_fail_threshold=TIER3_LIFT_FAIL_THRESHOLD,
        )

    def get_file_extension(self) -> str:
        return ".html"
