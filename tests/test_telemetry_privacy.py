# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public telemetry privacy defaults."""

from __future__ import annotations

from skillevaluator import telemetry


def test_telemetry_does_not_enable_from_an_ambient_otlp_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("SKILLEVALUATOR_TELEMETRY_ENABLED", raising=False)
    monkeypatch.delenv("SKILLEVALUATOR_TELEMETRY_DISABLED", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector.example.test")

    assert telemetry._should_enable() is False


def test_telemetry_requires_explicit_enablement(monkeypatch) -> None:
    monkeypatch.setenv("SKILLEVALUATOR_TELEMETRY_ENABLED", "true")
    monkeypatch.delenv("SKILLEVALUATOR_TELEMETRY_DISABLED", raising=False)

    assert telemetry._should_enable() is True


def test_telemetry_identity_defaults_to_team_only(monkeypatch) -> None:
    monkeypatch.delenv("SKILLEVALUATOR_TELEMETRY_IDENTITY_MODE", raising=False)
    monkeypatch.setenv("GITHUB_ACTOR", "external-user")
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/skill")
    monkeypatch.setenv("GITHUB_REPOSITORY_OWNER", "example")

    attributes = telemetry.user_identity_attributes()

    assert attributes == {
        "skillevaluator.project.namespace": "example",
        "skillevaluator.project.path": "example/skill",
    }
