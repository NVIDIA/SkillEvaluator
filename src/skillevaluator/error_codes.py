# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable, vendor-neutral error codes for evaluator execution failures."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class ErrorDomain(StrEnum):
    """Factual subsystem that produced an evaluator execution failure."""

    AUTH = "AUTH"
    CONFIG = "CONFIG"
    DEPENDENCY = "DEPENDENCY"
    UNKNOWN = "UNKNOWN"
    RUNTIME = "RUNTIME"
    CONTRACT = "CONTRACT"


class EvaluatorErrorCode(StrEnum):
    """Stable support-facing identifiers emitted by SkillEvaluator.

    Allocations are append-only. Existing values must never be renamed, reused,
    or assigned a different meaning.
    """

    AUTHENTICATION = "SKILLEVALUATOR-AUTH-002"
    AUTHORIZATION = "SKILLEVALUATOR-AUTH-003"
    INVALID_CONFIGURATION = "SKILLEVALUATOR-CONFIG-001"
    DEPENDENCY_UNAVAILABLE = "SKILLEVALUATOR-DEPENDENCY-001"
    MODEL_NOT_FOUND = "SKILLEVALUATOR-DEPENDENCY-003"
    DEPENDENCY_TIMEOUT = "SKILLEVALUATOR-DEPENDENCY-005"
    RATE_LIMITED = "SKILLEVALUATOR-DEPENDENCY-006"
    UNSUPPORTED = "SKILLEVALUATOR-DEPENDENCY-007"
    INVALID_RESPONSE = "SKILLEVALUATOR-DEPENDENCY-008"
    OTHER_HTTP = "SKILLEVALUATOR-DEPENDENCY-009"
    UNKNOWN = "SKILLEVALUATOR-UNKNOWN-001"
    PROCESS_SPAWN_FAILED = "SKILLEVALUATOR-RUNTIME-005"
    EXECUTION_TIMEOUT = "SKILLEVALUATOR-RUNTIME-007"
    PROCESS_EXITED = "SKILLEVALUATOR-RUNTIME-008"
    JOB_RESULT_INVALID = "SKILLEVALUATOR-CONTRACT-007"


@dataclass(frozen=True, slots=True)
class ErrorCodeDefinition:
    """Immutable public metadata for one registered error code."""

    domain: ErrorDomain
    summary: str


# Registry insertion order is allocation order. Add new entries at the end.
ERROR_CODE_REGISTRY: Mapping[str, ErrorCodeDefinition] = MappingProxyType(
    {
        EvaluatorErrorCode.AUTHENTICATION.value: ErrorCodeDefinition(
            ErrorDomain.AUTH,
            "Provider authentication failed.",
        ),
        EvaluatorErrorCode.AUTHORIZATION.value: ErrorCodeDefinition(
            ErrorDomain.AUTH,
            "Provider authorization failed.",
        ),
        EvaluatorErrorCode.INVALID_CONFIGURATION.value: ErrorCodeDefinition(
            ErrorDomain.CONFIG,
            "Evaluator configuration was invalid.",
        ),
        EvaluatorErrorCode.DEPENDENCY_UNAVAILABLE.value: ErrorCodeDefinition(
            ErrorDomain.DEPENDENCY,
            "A required dependency was unavailable.",
        ),
        EvaluatorErrorCode.MODEL_NOT_FOUND.value: ErrorCodeDefinition(
            ErrorDomain.DEPENDENCY,
            "The configured model was not found.",
        ),
        EvaluatorErrorCode.DEPENDENCY_TIMEOUT.value: ErrorCodeDefinition(
            ErrorDomain.DEPENDENCY,
            "A required dependency timed out.",
        ),
        EvaluatorErrorCode.RATE_LIMITED.value: ErrorCodeDefinition(
            ErrorDomain.DEPENDENCY,
            "A required dependency rate-limited the evaluator.",
        ),
        EvaluatorErrorCode.UNSUPPORTED.value: ErrorCodeDefinition(
            ErrorDomain.DEPENDENCY,
            "The selected dependency operation is unsupported.",
        ),
        EvaluatorErrorCode.INVALID_RESPONSE.value: ErrorCodeDefinition(
            ErrorDomain.DEPENDENCY,
            "A required dependency returned an invalid response.",
        ),
        EvaluatorErrorCode.OTHER_HTTP.value: ErrorCodeDefinition(
            ErrorDomain.DEPENDENCY,
            "A required dependency returned another HTTP failure.",
        ),
        EvaluatorErrorCode.UNKNOWN.value: ErrorCodeDefinition(
            ErrorDomain.UNKNOWN,
            "The evaluator could not classify the failure.",
        ),
        EvaluatorErrorCode.PROCESS_SPAWN_FAILED.value: ErrorCodeDefinition(
            ErrorDomain.RUNTIME,
            "The evaluator could not start a runtime process.",
        ),
        EvaluatorErrorCode.EXECUTION_TIMEOUT.value: ErrorCodeDefinition(
            ErrorDomain.RUNTIME,
            "Evaluator runtime execution timed out.",
        ),
        EvaluatorErrorCode.PROCESS_EXITED.value: ErrorCodeDefinition(
            ErrorDomain.RUNTIME,
            "An evaluator runtime process exited unsuccessfully.",
        ),
        EvaluatorErrorCode.JOB_RESULT_INVALID.value: ErrorCodeDefinition(
            ErrorDomain.CONTRACT,
            "The evaluator runtime produced an invalid job result.",
        ),
    }
)

ERROR_CODE_PATTERN_TEXT = r"^SKILLEVALUATOR-[A-Z][A-Z0-9]*-[0-9]{3}$"
ERROR_CODE_PATTERN = re.compile(ERROR_CODE_PATTERN_TEXT, flags=re.ASCII)


def error_code_schema() -> dict[str, object]:
    """Return a serialization-ready JSON Schema for registered error codes."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SkillEvaluator error code",
        "type": "string",
        "pattern": ERROR_CODE_PATTERN_TEXT,
        "enum": list(ERROR_CODE_REGISTRY),
    }


def _validate_registry() -> None:
    enum_values = {member.value for member in EvaluatorErrorCode}
    if set(ERROR_CODE_REGISTRY) != enum_values:
        raise RuntimeError("error code registry and enum are inconsistent")
    for code, definition in ERROR_CODE_REGISTRY.items():
        if not code.isascii() or ERROR_CODE_PATTERN.fullmatch(code) is None:
            raise RuntimeError(f"invalid registered error code: {code!r}")
        if f"-{definition.domain.value}-" not in code:
            raise RuntimeError(f"error code domain does not match its registry entry: {code}")


_validate_registry()


def is_registered_error_code(value: object) -> bool:
    """Return whether *value* is an exact registered error code."""
    return isinstance(value, str) and value.isascii() and value in ERROR_CODE_REGISTRY


def validate_error_code(value: object) -> str:
    """Return a valid registered code or raise ``ValueError``."""
    if not is_registered_error_code(value):
        raise ValueError(f"unregistered SkillEvaluator error code: {value!r}")
    return str(value)


def provider_failure_error_code(
    failure_kind: object,
    http_status: object = None,
    *,
    timed_out: bool = False,
) -> EvaluatorErrorCode:
    """Map structured provider failure metadata without inspecting message text."""
    kind = str(failure_kind) if failure_kind is not None else "unknown"
    status = http_status if isinstance(http_status, int) and not isinstance(http_status, bool) else None
    if kind == "local_process":
        return EvaluatorErrorCode.EXECUTION_TIMEOUT if timed_out else EvaluatorErrorCode.PROCESS_SPAWN_FAILED
    if kind == "unavailable":
        if status == 429:
            return EvaluatorErrorCode.RATE_LIMITED
        if timed_out or status == 408:
            return EvaluatorErrorCode.DEPENDENCY_TIMEOUT
        return EvaluatorErrorCode.DEPENDENCY_UNAVAILABLE
    return {
        "authentication": EvaluatorErrorCode.AUTHENTICATION,
        "authorization": EvaluatorErrorCode.AUTHORIZATION,
        "invalid_configuration": EvaluatorErrorCode.INVALID_CONFIGURATION,
        "model_not_found": EvaluatorErrorCode.MODEL_NOT_FOUND,
        "unsupported": EvaluatorErrorCode.UNSUPPORTED,
        "invalid_response": EvaluatorErrorCode.INVALID_RESPONSE,
        "other_http": EvaluatorErrorCode.OTHER_HTTP,
        "unknown": EvaluatorErrorCode.UNKNOWN,
    }.get(kind, EvaluatorErrorCode.UNKNOWN)


def primary_error_code(codes: Iterable[object]) -> str:
    """Return a consensus code, or UNKNOWN when failures conflict or are absent."""
    registered = {str(code) for code in codes if is_registered_error_code(code)}
    return registered.pop() if len(registered) == 1 else EvaluatorErrorCode.UNKNOWN.value


__all__ = (
    "ERROR_CODE_PATTERN",
    "ERROR_CODE_PATTERN_TEXT",
    "ERROR_CODE_REGISTRY",
    "ErrorCodeDefinition",
    "ErrorDomain",
    "EvaluatorErrorCode",
    "error_code_schema",
    "is_registered_error_code",
    "primary_error_code",
    "provider_failure_error_code",
    "validate_error_code",
)
