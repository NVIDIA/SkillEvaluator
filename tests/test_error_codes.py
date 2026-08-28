# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden contract tests for stable evaluator error codes."""

from __future__ import annotations

import json
from types import MappingProxyType

import pytest

from skillevaluator.error_codes import (
    ERROR_CODE_PATTERN_TEXT,
    ERROR_CODE_REGISTRY,
    ErrorDomain,
    EvaluatorErrorCode,
    error_code_schema,
    is_registered_error_code,
    primary_error_code,
    provider_failure_error_code,
    validate_error_code,
)


def test_error_code_registry_matches_append_only_public_golden() -> None:
    golden = [
        ("SKILLEVALUATOR-AUTH-002", "AUTH", "Provider authentication failed."),
        ("SKILLEVALUATOR-AUTH-003", "AUTH", "Provider authorization failed."),
        ("SKILLEVALUATOR-CONFIG-001", "CONFIG", "Evaluator configuration was invalid."),
        ("SKILLEVALUATOR-DEPENDENCY-001", "DEPENDENCY", "A required dependency was unavailable."),
        ("SKILLEVALUATOR-DEPENDENCY-003", "DEPENDENCY", "The configured model was not found."),
        ("SKILLEVALUATOR-DEPENDENCY-005", "DEPENDENCY", "A required dependency timed out."),
        (
            "SKILLEVALUATOR-DEPENDENCY-006",
            "DEPENDENCY",
            "A required dependency rate-limited the evaluator.",
        ),
        (
            "SKILLEVALUATOR-DEPENDENCY-007",
            "DEPENDENCY",
            "The selected dependency operation is unsupported.",
        ),
        (
            "SKILLEVALUATOR-DEPENDENCY-008",
            "DEPENDENCY",
            "A required dependency returned an invalid response.",
        ),
        (
            "SKILLEVALUATOR-DEPENDENCY-009",
            "DEPENDENCY",
            "A required dependency returned another HTTP failure.",
        ),
        ("SKILLEVALUATOR-UNKNOWN-001", "UNKNOWN", "The evaluator could not classify the failure."),
        (
            "SKILLEVALUATOR-RUNTIME-005",
            "RUNTIME",
            "The evaluator could not start a runtime process.",
        ),
        ("SKILLEVALUATOR-RUNTIME-007", "RUNTIME", "Evaluator runtime execution timed out."),
        (
            "SKILLEVALUATOR-RUNTIME-008",
            "RUNTIME",
            "An evaluator runtime process exited unsuccessfully.",
        ),
        (
            "SKILLEVALUATOR-CONTRACT-007",
            "CONTRACT",
            "The evaluator runtime produced an invalid job result.",
        ),
    ]

    assert [
        (code, definition.domain.value, definition.summary) for code, definition in ERROR_CODE_REGISTRY.items()
    ] == golden


def test_error_code_registry_is_immutable_and_schema_is_serializable() -> None:
    assert isinstance(ERROR_CODE_REGISTRY, MappingProxyType)
    assert ERROR_CODE_PATTERN_TEXT == r"^SKILLEVALUATOR-[A-Z][A-Z0-9]*-[0-9]{3}$"
    schema = error_code_schema()
    assert schema["pattern"] == ERROR_CODE_PATTERN_TEXT
    assert schema["enum"] == list(ERROR_CODE_REGISTRY)
    assert json.loads(json.dumps(schema)) == schema
    schema["enum"] = []
    assert error_code_schema()["enum"] == list(ERROR_CODE_REGISTRY)
    assert {definition.domain for definition in ERROR_CODE_REGISTRY.values()} == set(ErrorDomain)
    with pytest.raises(TypeError):
        ERROR_CODE_REGISTRY["SKILLEVALUATOR-UNKNOWN-999"] = ERROR_CODE_REGISTRY[  # type: ignore[index]
            EvaluatorErrorCode.UNKNOWN.value
        ]


@pytest.mark.parametrize(
    ("failure_kind", "http_status", "timed_out", "expected"),
    [
        ("authentication", 401, False, "SKILLEVALUATOR-AUTH-002"),
        ("authorization", 403, False, "SKILLEVALUATOR-AUTH-003"),
        ("invalid_configuration", None, False, "SKILLEVALUATOR-CONFIG-001"),
        ("model_not_found", 404, False, "SKILLEVALUATOR-DEPENDENCY-003"),
        ("unsupported", 405, False, "SKILLEVALUATOR-DEPENDENCY-007"),
        ("invalid_response", None, False, "SKILLEVALUATOR-DEPENDENCY-008"),
        ("other_http", 418, False, "SKILLEVALUATOR-DEPENDENCY-009"),
        ("unknown", None, False, "SKILLEVALUATOR-UNKNOWN-001"),
        ("unavailable", 429, False, "SKILLEVALUATOR-DEPENDENCY-006"),
        ("unavailable", 408, False, "SKILLEVALUATOR-DEPENDENCY-005"),
        ("unavailable", None, True, "SKILLEVALUATOR-DEPENDENCY-005"),
        ("unavailable", 503, False, "SKILLEVALUATOR-DEPENDENCY-001"),
        ("unavailable", None, False, "SKILLEVALUATOR-DEPENDENCY-001"),
    ],
)
def test_provider_failure_mapping_is_structured_and_stable(
    failure_kind: str,
    http_status: int | None,
    timed_out: bool,
    expected: str,
) -> None:
    assert provider_failure_error_code(failure_kind, http_status, timed_out=timed_out).value == expected


@pytest.mark.parametrize(
    "value",
    [
        "skillevaluator-auth-002",
        "SKILLEVALUATOR-AUTH-2",
        "SKILLEVALUATOR-SERVICE-999",
        "SKILLEVALUATOR-AUTH-002\n",
        "SKILL\u212aEVALUATOR-AUTH-002",
        None,
    ],
)
def test_error_code_validation_rejects_noncanonical_values(value: object) -> None:
    assert is_registered_error_code(value) is False
    with pytest.raises(ValueError, match="unregistered"):
        validate_error_code(value)


def test_primary_error_code_requires_consensus_and_falls_back_to_unknown() -> None:
    assert primary_error_code(["invalid", EvaluatorErrorCode.AUTHORIZATION]) == "SKILLEVALUATOR-AUTH-003"
    assert primary_error_code([]) == "SKILLEVALUATOR-UNKNOWN-001"
    assert (
        primary_error_code([EvaluatorErrorCode.AUTHENTICATION, EvaluatorErrorCode.INVALID_CONFIGURATION])
        == "SKILLEVALUATOR-UNKNOWN-001"
    )
