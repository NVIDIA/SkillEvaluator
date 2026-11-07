# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import precomputed custom rewards without claiming a local evaluation ran."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from skillevaluator.tier3.case_ids import validate_case_id
from skillevaluator.tier3.harbor.metrics import (
    DIMENSION_DEFINITIONS,
    MAX_CUSTOM_METRICS,
    RESERVED_METRIC_NAMES,
    average_custom_metrics,
    custom_metric_name_is_publishable,
    score_value,
)
from skillevaluator.utils.redaction import contains_credential_value
from skillevaluator.utils.structured_data import preflight_json_structure

MAX_INPUT_BYTES = 4 * 1024 * 1024
MAX_OBSERVATIONS = 1000
ARMS = ("with_skill", "without_skill")


def _identity(value: str) -> str:
    if not value.strip() or value != value.strip() or not value.isprintable() or contains_credential_value(value):
        raise ValueError("identities must be nonempty printable text without credentials or surrounding whitespace")
    return value


Identity = Annotated[str, Field(min_length=1, max_length=256), AfterValidator(_identity)]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class MetricDefinition(_Contract):
    """All imported scores use the existing custom grader's normalized 0..1 scale."""

    version: Identity
    unit: Identity
    direction: Literal["higher_is_better", "lower_is_better"]
    aggregation: Literal["mean"]


class EvaluationIdentity(_Contract):
    """One compatible dataset, skill, agent/model and environment per import."""

    dataset_id: Identity
    dataset_version: Identity
    skill_id: Identity
    skill_version: Identity
    agent: Identity
    agent_version: Identity
    model: Identity
    environment: Identity


class ExternalProvenance(_Contract):
    producer: Identity
    run_id: Identity
    created_at: Identity

    @field_validator("created_at")
    @classmethod
    def _timestamp(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("created_at must be an ISO 8601 timestamp with a timezone") from exc
        if parsed.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        return value


class ExternalObservation(_Contract):
    case_id: Identity
    attempt: Annotated[int, Field(ge=1, le=MAX_OBSERVATIONS)]
    arm: Literal["with_skill", "without_skill"]
    custom_metrics: dict[str, float | None] = Field(min_length=1, max_length=MAX_CUSTOM_METRICS)

    @field_validator("case_id")
    @classmethod
    def _case_id(cls, value: str) -> str:
        return validate_case_id(value)

    @field_validator("custom_metrics", mode="before")
    @classmethod
    def _scores(cls, values: Any) -> Any:
        if not isinstance(values, dict) or any(v is not None and score_value(v) is None for v in values.values()):
            raise ValueError("custom metrics must be finite numbers in [0, 1] or null, never booleans or strings")
        return values


class ExternalMetrics(_Contract):
    schema_version: Literal["1"]
    evaluation: EvaluationIdentity
    provenance: ExternalProvenance
    metric_definitions: dict[str, MetricDefinition] = Field(min_length=1, max_length=MAX_CUSTOM_METRICS)
    observations: list[ExternalObservation] = Field(min_length=1, max_length=MAX_OBSERVATIONS)

    @field_validator("metric_definitions")
    @classmethod
    def _names(cls, definitions: dict[str, MetricDefinition]) -> dict[str, MetricDefinition]:
        reserved = RESERVED_METRIC_NAMES | DIMENSION_DEFINITIONS.keys()
        if any(name in reserved or not custom_metric_name_is_publishable(name) for name in definitions):
            raise ValueError("custom metric names must be publishable and cannot use reserved metrics or dimensions")
        return definitions

    @model_validator(mode="after")
    def _observations(self) -> ExternalMetrics:
        seen = set()
        case_names: dict[str, str] = {}
        for row in self.observations:
            identity = (row.case_id, row.attempt, row.arm)
            if identity in seen:
                raise ValueError("duplicate case/attempt/arm observation")
            seen.add(identity)
            if case_names.setdefault(row.case_id.casefold(), row.case_id) != row.case_id:
                raise ValueError("case IDs must not differ only in letter case")
            if set(row.custom_metrics) - self.metric_definitions.keys():
                raise ValueError("observations reference undefined custom metrics")
        return self


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON number")


def load_external_metrics(path: Path) -> tuple[ExternalMetrics, str]:
    """Validate a bounded, duplicate-free JSON document and hash its exact bytes."""
    with path.open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("external metric input exceeds 4 MiB")
    try:
        text = raw.decode("utf-8")
        preflight_json_structure(text, max_depth=12, max_tokens=500_000, max_collection_items=MAX_OBSERVATIONS)
        data = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:
        raise ValueError("invalid external metrics JSON (check syntax, duplicate keys and size limits)") from exc
    try:
        contract = ExternalMetrics.model_validate(data)
    except ValidationError as exc:
        reason = exc.errors(include_input=False, include_context=False)[0]["msg"]
        raise ValueError(f"invalid external metrics contract: {reason}") from exc
    return contract, hashlib.sha256(raw).hexdigest()


def _paired_summary(contract: ExternalMetrics, name: str) -> dict[str, Any]:
    scores = {
        (row.case_id, row.attempt, row.arm): score
        for row in contract.observations
        if (score := row.custom_metrics.get(name)) is not None
    }
    by_case: dict[str, list[float]] = defaultdict(list)
    for (case_id, attempt, arm), value in scores.items():
        baseline = scores.get((case_id, attempt, "without_skill"))
        if arm == "with_skill" and baseline is not None:
            by_case[case_id].append(value - baseline)
    delta = None
    if by_case:
        delta = round(math.fsum(math.fsum(values) / len(values) for values in by_case.values()) / len(by_case), 4)
        if delta == 0:
            delta = 0.0
    direction = contract.metric_definitions[name].direction
    return {
        "delta": delta,
        "improvement": delta if delta is None or delta == 0 or direction == "higher_is_better" else -delta,
        "paired_cases": len(by_case),
        "paired_attempts": sum(len(values) for values in by_case.values()),
    }


def summarize_external_metrics(contract: ExternalMetrics, source_sha256: str) -> dict[str, Any]:
    """Reuse custom reward aggregation, keeping imported evidence out of canonical scores."""
    arms: dict[str, Any] = {}
    for arm in ARMS:
        rows = [row for row in contract.observations if row.arm == arm]
        arms[arm] = {
            "observations": len(rows),
            "custom_metrics": average_custom_metrics([{"custom_metrics": row.custom_metrics} for row in rows]),
            "counts": {
                name: sum(row.custom_metrics.get(name) is not None for row in rows)
                for name in contract.metric_definitions
            },
        }
    return {
        "report_kind": "external_custom_metrics",
        "schema_version": "1",
        "observation_source": "external",
        "executed_by_skillevaluator": False,
        "imported_at": datetime.now(UTC).isoformat(),
        "source_sha256": source_sha256,
        **contract.model_dump(),
        "arms": arms,
        "custom_lift": {name: _paired_summary(contract, name) for name in sorted(contract.metric_definitions)},
    }
