# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safety boundaries that must remain after optional telemetry is removed."""

from __future__ import annotations

import json

import pytest

from skillevaluator.utils.process_environment import child_process_env
from skillevaluator.utils.redaction import contains_credential_value, redact_sensitive_data, redact_sensitive_text


def _pem_block(
    label: str,
    body: str = "synthetic-key-material-for-redaction-only",
    *,
    end_label: str | None = None,
) -> str:
    return "\n".join((f"-----BEGIN {label}-----", body, f"-----END {end_label or label}-----"))


def test_redact_sensitive_text_masks_common_credentials() -> None:
    source = (
        "Authorization: Bearer bearer-secret-value\n"
        "api_key='plain-secret-value'\n"
        "token=another-secret-value\n"
        "NVIDIA_API_KEY=x"
    )

    redacted = redact_sensitive_text(source)

    assert "bearer-secret-value" not in redacted
    assert "plain-secret-value" not in redacted
    assert "another-secret-value" not in redacted
    assert "NVIDIA_API_KEY=x" not in redacted
    assert redacted.count("<redacted>") >= 4


@pytest.mark.parametrize(
    "value",
    [
        "case-sk-abcdefghijk",
        "case_nvapi-abcdefghijk",
        "case_ghp_" + ("a" * 36),
        "case_gho_" + ("a" * 36),
        "case_ghu_" + ("a" * 36),
        "case_github_pat_" + ("a" * 30),
        "case_ghr_" + ("a" * 36),
        "caseghp_" + ("a" * 36),
        "casegho_" + ("a" * 36) + "suffix",
        "case_ghs_123456789_" + ("a" * 32) + "." + ("b" * 32) + "." + ("c" * 32),
        "case_" + "".join(("xoxb-", "1234567890-abcdefghijklmnopqrstuvwx")),  # noqa: FLY002
        "case_" + "AIza" + ("A" * 35),
        "case_glpat-" + ("a" * 20),
    ],
)
def test_contains_credential_value_detects_embedded_identity_tokens(value: str) -> None:
    assert contains_credential_value(value) is True


def test_contains_credential_value_preserves_case_sensitive_noncanonical_prefix() -> None:
    assert contains_credential_value("case-SK-ABCDEFGHIJK") is False


def test_contains_credential_value_does_not_match_sk_inside_an_ordinary_word() -> None:
    assert contains_credential_value("task-legacy-model") is False


@pytest.mark.parametrize("prefix", ["ghp", "gho", "ghu", "ghs", "ghr"])
def test_contains_credential_value_does_not_match_short_github_shaped_identifiers(prefix: str) -> None:
    assert contains_credential_value(f"case_{prefix}_{'a' * 35}") is False


@pytest.mark.parametrize(
    "source",
    (
        "proxy https://alice:correct-horse-battery@example.test/path failed",
        "proxy https://alice:correct@horse@example.test/path failed",
        "proxy socks5://bearer-token@example.test failed",
        "proxy http://user%3Apassword@example.test failed",
    ),
)
def test_redact_sensitive_text_masks_uri_userinfo(source: str) -> None:
    redacted = redact_sensitive_text(source)

    assert "alice" not in redacted
    assert "correct-horse-battery" not in redacted
    assert "correct@horse" not in redacted
    assert "bearer-token" not in redacted
    assert "user%3Apassword" not in redacted
    assert "<redacted>@example.test" in redacted


def test_redact_sensitive_text_masks_unlabelled_secret_shapes() -> None:
    aws_access_key = "".join(("AKIA", "IOSFODNN7", "EXAMPLE"))  # noqa: FLY002 - scanner-safe fixture
    jwt = ".".join(
        (
            "".join(("eyJ", "hbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")),  # noqa: FLY002
            "".join(("eyJ", "zdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ")),  # noqa: FLY002
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        )
    )
    private_key = "\n".join(
        (
            "-----BEGIN " + "RSA PRIVATE KEY-----",
            "synthetic-key-material-for-redaction-only",
            "-----END " + "RSA PRIVATE KEY-----",
        )
    )
    source = f"aws={aws_access_key}\njwt={jwt}\n{private_key}"

    redacted = redact_sensitive_text(source)

    assert aws_access_key not in redacted
    assert jwt not in redacted
    assert private_key not in redacted
    assert redacted.count("<redacted>") >= 3


@pytest.mark.parametrize(
    "label",
    (
        "PRIVATE KEY",
        "RSA PRIVATE KEY",
        "OPENSSH PRIVATE KEY",
        "EC PRIVATE KEY",
        "DSA PRIVATE KEY",
        "ENCRYPTED PRIVATE KEY",
        "PGP PRIVATE KEY BLOCK",
    ),
)
def test_redact_sensitive_text_masks_standard_private_key_pem_labels(label: str) -> None:
    assert redact_sensitive_text(_pem_block(label)) == "private-key-<redacted>"


def test_redact_sensitive_text_masks_mismatched_private_key_delimiters() -> None:
    source = _pem_block("RSA PRIVATE KEY", end_label="EC PRIVATE KEY")

    assert redact_sensitive_text(source) == "private-key-<redacted>"


@pytest.mark.parametrize("label", ("PUBLIC KEY", "RSA PUBLIC KEY", "CERTIFICATE", "CERTIFICATE REQUEST"))
def test_redact_sensitive_text_preserves_non_private_pem_blocks(label: str) -> None:
    source = _pem_block(label, "synthetic-public-material")

    assert redact_sensitive_text(source) == source


def test_redact_sensitive_text_redacts_multiple_private_key_blocks_independently() -> None:
    source = "\n".join(
        (
            _pem_block("RSA PRIVATE KEY", "first-synthetic-key-material"),
            "diagnostic separator",
            _pem_block("ENCRYPTED PRIVATE KEY", "second-synthetic-key-material"),
        )
    )

    assert redact_sensitive_text(source) == ("private-key-<redacted>\ndiagnostic separator\nprivate-key-<redacted>")


def test_redact_sensitive_text_does_not_span_unrelated_pem_blocks() -> None:
    source = _pem_block(
        "RSA PRIVATE KEY",
        "\n".join(
            (
                "synthetic-key-material-for-redaction-only",
                _pem_block("CERTIFICATE", "synthetic-certificate-material"),
            )
        ),
    )

    assert redact_sensitive_text(source) == source


def test_redact_sensitive_text_preserves_multiline_diagnostics_around_private_key() -> None:
    source = "\n".join(
        (
            "provider request failed",
            "diagnostic details follow",
            _pem_block("DSA PRIVATE KEY", "line-one\nline-two\nline-three"),
            "retry disabled",
        )
    )

    assert redact_sensitive_text(source) == (
        "provider request failed\ndiagnostic details follow\nprivate-key-<redacted>\nretry disabled"
    )


@pytest.mark.parametrize("label", ("PRIVATE KEY", "DSA PRIVATE KEY", "ENCRYPTED PRIVATE KEY"))
def test_redact_sensitive_text_masks_truncated_private_key_through_eof(label: str) -> None:
    source = f"provider failed\n-----BEGIN {label}-----\nSENSITIVE-KEY-MATERIAL"

    assert redact_sensitive_text(source) == "provider failed\nprivate-key-<redacted>"


@pytest.mark.parametrize("separator", ("=", ":"))
def test_redact_sensitive_text_masks_private_key_after_sensitive_assignment(separator: str) -> None:
    private_key = _pem_block("ENCRYPTED PRIVATE KEY", "SENSITIVE-KEY-MATERIAL")
    source = f"private_key{separator}{private_key}"

    redacted = redact_sensitive_text(source)

    assert "SENSITIVE-KEY-MATERIAL" not in redacted
    assert "BEGIN" not in redacted
    assert "END" not in redacted


def test_redact_sensitive_text_masks_private_key_after_authorization_header() -> None:
    private_key = _pem_block("PRIVATE KEY", "SENSITIVE-KEY-MATERIAL")

    redacted = redact_sensitive_text(f"Authorization: Bearer {private_key}")

    assert "SENSITIVE-KEY-MATERIAL" not in redacted
    assert "BEGIN" not in redacted
    assert "END" not in redacted


def test_redact_sensitive_data_masks_secret_keys_and_nested_text() -> None:
    source = {
        "provider": {
            "apiKey": "provider-secret",
            "message": "Authorization: Bearer nested-secret-value",
        },
        "token_count": 42,
        "last_token_usage": {"prompt_tokens": 10, "completion_tokens": 2},
        "tokens": ["opaque-secret"],
        "passwords": ["another-secret"],
    }

    assert redact_sensitive_data(source) == {
        "provider": {
            "apiKey": "<redacted>",
            "message": "Authorization:<redacted>",
        },
        "token_count": 42,
        "last_token_usage": {"prompt_tokens": 10, "completion_tokens": 2},
        "tokens": "<redacted>",
        "passwords": "<redacted>",
    }


def test_redact_sensitive_data_drops_credential_values_used_as_mapping_keys() -> None:
    credential = "sk-abcdefghijk"

    redacted = redact_sensitive_data(
        {
            "custom_metrics": {credential: 0.5, "quality": 0.7},
            "api_key": "provider-secret",
        }
    )

    assert redacted == {
        "custom_metrics": {"quality": 0.7},
        "api_key": "<redacted>",
    }
    assert credential not in json.dumps(redacted)


def test_child_process_env_strips_observability_configuration_without_injecting_flags() -> None:
    source = {
        "PATH": "/usr/bin",
        "APP_MODE": "test",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "https://collector.example.test",
        "DD_TRACE_ENABLED": "true",
        "SKILLEVALUATOR_TELEMETRY_ENABLED": "true",
    }

    assert child_process_env(source) == {
        "PATH": "/usr/bin",
        "APP_MODE": "test",
    }
