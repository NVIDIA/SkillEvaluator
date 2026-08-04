# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safety boundaries that must remain after optional telemetry is removed."""

from __future__ import annotations

from skillevaluator.utils.process_environment import child_process_env
from skillevaluator.utils.redaction import redact_sensitive_data, redact_sensitive_text


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


def test_redact_sensitive_data_masks_secret_keys_and_nested_text() -> None:
    source = {
        "provider": {
            "apiKey": "provider-secret",
            "message": "Authorization: Bearer nested-secret-value",
        },
        "token_count": 42,
    }

    assert redact_sensitive_data(source) == {
        "provider": {
            "apiKey": "<redacted>",
            "message": "Authorization:<redacted>",
        },
        "token_count": 42,
    }


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
