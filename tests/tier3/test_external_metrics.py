# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precomputed scores share custom grader bounds, never canonical score semantics."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from skillevaluator.cli import cli
from skillevaluator.reporting.dashboard_data import parse_dashboard_report
from skillevaluator.reporting.external_metrics import import_external_metrics, render_external_metrics_markdown
from skillevaluator.tier3.external_metrics import (
    MAX_INPUT_BYTES,
    ExternalMetrics,
    load_external_metrics,
    summarize_external_metrics,
)
from skillevaluator.tier3.harbor.metrics import RESERVED_METRIC_NAMES, average_custom_metrics


@pytest.fixture
def payload() -> dict:
    return {
        "schema_version": "1",
        "evaluation": {
            "dataset_id": "circuit-simulation",
            "dataset_version": "sha256:example",
            "skill_id": "timing-analysis",
            "skill_version": "v1",
            "agent": "sim-driver",
            "agent_version": "v2",
            "model": "example-model-v1",
            "environment": "simulator-v3",
        },
        "provenance": {"producer": "sim-exporter-v1", "run_id": "sim-123", "created_at": "2026-10-01T12:00:00Z"},
        "metric_definitions": {
            "timing_error_fraction": {
                "version": "1",
                "unit": "fraction_of_budget",
                "direction": "lower_is_better",
                "aggregation": "mean",
            }
        },
        "observations": [
            {"case_id": "adder", "attempt": 1, "arm": "with_skill", "custom_metrics": {"timing_error_fraction": 0.1}},
            {
                "case_id": "adder",
                "attempt": 1,
                "arm": "without_skill",
                "custom_metrics": {"timing_error_fraction": 0.4},
            },
        ],
    }


def summarize(payload: dict) -> dict:
    return summarize_external_metrics(ExternalMetrics.model_validate(payload), "0" * 64)


def test_reuses_custom_reward_averages_without_canonical_scores(payload):
    report = summarize(payload)
    expected = average_custom_metrics([{"custom_metrics": {"timing_error_fraction": 0.1}}])
    assert report["arms"]["with_skill"]["custom_metrics"] == expected
    assert report["custom_lift"]["timing_error_fraction"] == {
        "delta": -0.3,
        "improvement": 0.3,
        "paired_cases": 1,
        "paired_attempts": 1,
    }
    assert report["observation_source"] == "external"
    assert report["executed_by_skillevaluator"] is False
    assert not {"overall", "overall_score", "dimensions", "passed", "score_policy"} & report.keys()
    assert report["provenance"] == payload["provenance"]
    assert report["observations"] == payload["observations"]


@pytest.mark.parametrize(
    "value", [True, False, "0.5", -0.01, 1.01, float("nan"), float("inf"), -float("inf"), 10**400, {}]
)
def test_rejects_invalid_scores(payload, value):
    payload["observations"][0]["custom_metrics"]["timing_error_fraction"] = value
    with pytest.raises(ValidationError):
        ExternalMetrics.model_validate(payload)


@pytest.mark.parametrize(
    "name",
    [
        *sorted(RESERVED_METRIC_NAMES),
        "correctness",
        "efficiency",
        "discoverability",
        "effectiveness",
        "bad\nname",
        " ",
        "a" * 257,
    ],
)
def test_rejects_reserved_and_unpublishable_names(payload, name):
    definition = payload["metric_definitions"].pop("timing_error_fraction")
    payload["metric_definitions"][name] = definition
    with pytest.raises(ValidationError):
        ExternalMetrics.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [("direction", "up"), ("aggregation", "sum"), ("version", ""), ("unit", ""), ("unit", "line\nbreak")],
)
def test_rejects_invalid_metric_metadata(payload, field, value):
    payload["metric_definitions"]["timing_error_fraction"][field] = value
    with pytest.raises(ValidationError):
        ExternalMetrics.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    ["dataset_id", "dataset_version", "skill_id", "skill_version", "agent", "agent_version", "model", "environment"],
)
def test_requires_complete_evaluation_identity(payload, field):
    del payload["evaluation"][field]
    with pytest.raises(ValidationError):
        ExternalMetrics.model_validate(payload)


@pytest.mark.parametrize("timestamp", ["yesterday", "2026-10-01", "2026-10-01T12:00:00"])
def test_requires_timezone_provenance(payload, timestamp):
    payload["provenance"]["created_at"] = timestamp
    with pytest.raises(ValidationError):
        ExternalMetrics.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [("case_id", "../bad"), ("case_id", ""), ("attempt", True), ("attempt", 0), ("attempt", "1"), ("arm", "baseline")],
)
def test_requires_valid_observation_identity(payload, field, value):
    payload["observations"][0][field] = value
    with pytest.raises(ValidationError):
        ExternalMetrics.model_validate(payload)


def test_rejects_duplicate_or_case_colliding_observations(payload):
    payload["observations"].append(copy.deepcopy(payload["observations"][0]))
    with pytest.raises(ValidationError, match="duplicate"):
        ExternalMetrics.model_validate(payload)
    payload["observations"][-1]["case_id"] = "Adder"
    with pytest.raises(ValidationError, match="letter case"):
        ExternalMetrics.model_validate(payload)


def test_rejects_undefined_metrics_and_per_row_context_overrides(payload):
    payload["observations"][0]["custom_metrics"]["undefined"] = 0.2
    with pytest.raises(ValidationError, match="undefined"):
        ExternalMetrics.model_validate(payload)
    del payload["observations"][0]["custom_metrics"]["undefined"]
    payload["observations"][0]["evaluation"] = {"dataset_version": "different"}
    with pytest.raises(ValidationError, match="Extra inputs"):
        ExternalMetrics.model_validate(payload)


@pytest.mark.parametrize("missing", [None, "absent"])
def test_missing_scores_are_counted_not_zeroed_or_paired(payload, missing):
    if missing is None:
        payload["observations"][1]["custom_metrics"]["timing_error_fraction"] = None
    else:
        payload["observations"] = payload["observations"][:1]
    report = summarize(payload)
    assert report["arms"]["without_skill"]["custom_metrics"] == {}
    assert report["arms"]["without_skill"]["counts"] == {"timing_error_fraction": 0}
    assert report["custom_lift"]["timing_error_fraction"] == {
        "delta": None,
        "improvement": None,
        "paired_cases": 0,
        "paired_attempts": 0,
    }


def test_absent_metric_is_allowed_and_reported_unscored(payload):
    payload["metric_definitions"]["other_metric"] = copy.deepcopy(
        payload["metric_definitions"]["timing_error_fraction"]
    )
    report = summarize(payload)
    assert report["arms"]["with_skill"]["counts"]["other_metric"] == 0
    assert report["custom_lift"]["other_metric"]["delta"] is None


def test_pairing_requires_same_attempt_not_just_same_case(payload):
    payload["observations"][1]["attempt"] = 2
    report = summarize(payload)
    assert report["custom_lift"]["timing_error_fraction"]["delta"] is None


def test_lift_is_case_weighted_on_matched_attempts_only(payload):
    rows = payload["observations"]
    for case_id, attempt, with_score, without_score in [("adder", 2, 0.3, 0.4), ("multiplier", 1, 0.6, 0.5)]:
        for arm, score in [("with_skill", with_score), ("without_skill", without_score)]:
            rows.append(
                {"case_id": case_id, "attempt": attempt, "arm": arm, "custom_metrics": {"timing_error_fraction": score}}
            )
    rows.append({"case_id": "adder", "attempt": 3, "arm": "with_skill", "custom_metrics": {"timing_error_fraction": 1}})
    lift = summarize(payload)["custom_lift"]["timing_error_fraction"]
    assert lift == {"delta": -0.05, "improvement": 0.05, "paired_cases": 2, "paired_attempts": 3}
    payload["metric_definitions"]["timing_error_fraction"]["direction"] = "higher_is_better"
    assert summarize(payload)["custom_lift"]["timing_error_fraction"]["improvement"] == -0.05


def test_loader_preserves_hash_and_rejects_duplicate_keys(payload, tmp_path):
    path = tmp_path / "metrics.json"
    raw = json.dumps(payload).encode()
    path.write_bytes(raw)
    _, digest = load_external_metrics(path)
    assert digest == hashlib.sha256(raw).hexdigest()
    path.write_text('{"schema_version":"1","schema_version":"1"}')
    with pytest.raises(ValueError, match="duplicate keys"):
        load_external_metrics(path)


@pytest.mark.parametrize("raw", ['{"x":NaN}', '{"x":Infinity}', "[" * 14 + "0" + "]" * 14, "\ud800", "not JSON"])
def test_loader_rejects_malformed_or_unbounded_input(tmp_path, raw):
    path = tmp_path / "metrics.json"
    path.write_bytes(raw.encode("utf-8", errors="surrogatepass"))
    with pytest.raises(ValueError):
        load_external_metrics(path)


def test_input_size_limit(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_bytes(b" " * (MAX_INPUT_BYTES + 1))
    with pytest.raises(ValueError, match="4 MiB"):
        load_external_metrics(path)


@pytest.mark.parametrize("container", ["metric_definitions", "observations"])
def test_empty_containers_rejected(payload, container):
    payload[container] = {} if container == "metric_definitions" else []
    with pytest.raises(ValidationError):
        ExternalMetrics.model_validate(payload)


def test_documented_simulator_import_end_to_end(tmp_path):
    example = Path(__file__).resolve().parents[2] / "docs" / "assets" / "simulator-metrics.json"
    output = tmp_path / "external"
    result = CliRunner().invoke(cli, ["tier3", "import-metrics", str(example), "-o", str(output)])
    assert result.exit_code == 0, result.output
    assert "did not execute these tests" in result.output
    assert {path.name for path in output.iterdir()} == {"external_metrics.json", "external_metrics.md"}
    report = json.loads((output / "external_metrics.json").read_text())
    assert parse_dashboard_report(report, "external_metrics.json").rows == []
    assert report["source_sha256"] == hashlib.sha256(example.read_bytes()).hexdigest()
    assert report["provenance"]["run_id"] == "synthetic-example-1"
    assert report["arms"]["with_skill"]["custom_metrics"] == {"convergence_rate": 0.85, "timing_error_fraction": 0.15}
    assert report["arms"]["without_skill"]["custom_metrics"] == {"convergence_rate": 0.65, "timing_error_fraction": 0.3}
    assert report["custom_lift"]["timing_error_fraction"] == {
        "delta": -0.15,
        "improvement": 0.15,
        "paired_cases": 2,
        "paired_attempts": 2,
    }
    markdown = (output / "external_metrics.md").read_text()
    assert "did not execute these tests" in markdown
    assert "No canonical dimensions, Overall" in markdown
    assert "0.8500 (2/2) | 0.6500 (2/2) | 0.2000 | 0.2000 | 2/2" in markdown
    assert "0.1500 (2/2) | 0.3000 (2/2) | -0.1500 | 0.1500 | 2/2" in markdown


def test_invalid_import_writes_nothing(payload, tmp_path):
    payload["observations"][0]["custom_metrics"]["timing_error_fraction"] = "not a score"
    source = tmp_path / "source.json"
    source.write_text(json.dumps(payload))
    output = tmp_path / "not-created"
    result = CliRunner().invoke(cli, ["tier3", "import-metrics", str(source), "-o", str(output)])
    assert result.exit_code == 1
    assert "invalid external metrics contract" in result.output
    assert not output.exists()


def test_existing_output_and_source_are_not_overwritten(payload, tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(payload))
    output = tmp_path / "live-run"
    output.mkdir()
    saved = output / "result.json"
    saved.write_text("untouched")
    with pytest.raises(FileExistsError):
        import_external_metrics(source, output)
    assert saved.read_text() == "untouched"
    assert list(output.iterdir()) == [saved]
    assert json.loads(source.read_text()) == payload


def test_markdown_escapes_untrusted_metadata(payload):
    payload["provenance"]["producer"] = '<img src=x onerror="alert(1)">[link](https://example.org)|*text*'
    markdown = render_external_metrics_markdown(summarize(payload))
    assert "<img" not in markdown
    assert "[link](https://example.org)" not in markdown
    assert "\\|\\*text\\*" in markdown


def test_unscored_report_shows_no_score_not_zero(payload):
    payload["observations"][1]["custom_metrics"]["timing_error_fraction"] = None
    markdown = render_external_metrics_markdown(summarize(payload))
    assert "NO SCORE (0/1) | NO SCORE | NO SCORE | 0/0" in markdown


def test_unknown_schema_version_rejected(payload):
    payload["schema_version"] = "2"
    with pytest.raises(ValidationError):
        ExternalMetrics.model_validate(payload)


def test_row_and_metric_limits(payload):
    payload["observations"] *= 501
    with pytest.raises(ValidationError):
        ExternalMetrics.model_validate(payload)
    payload["observations"] = payload["observations"][:1]
    template = payload["metric_definitions"]["timing_error_fraction"]
    payload["metric_definitions"] = {f"metric_{index}": template for index in range(129)}
    with pytest.raises(ValidationError):
        ExternalMetrics.model_validate(payload)


def test_loader_diagnostics_do_not_echo_secret_values_or_keys(payload, tmp_path):
    secret_text = "sensitive-unexpected-field"
    payload[secret_text] = "sensitive-value"
    source = tmp_path / "source.json"
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError) as caught:
        load_external_metrics(source)
    assert secret_text not in str(caught.value)
    assert "sensitive-value" not in str(caught.value)


@pytest.mark.parametrize("value", [0, 1])
def test_boundary_scores_and_zero_lift_are_not_missing(payload, value):
    for row in payload["observations"]:
        row["custom_metrics"]["timing_error_fraction"] = value
    report = summarize(payload)
    assert report["arms"]["with_skill"]["counts"] == {"timing_error_fraction": 1}
    assert report["custom_lift"]["timing_error_fraction"] == {
        "delta": 0,
        "improvement": 0,
        "paired_cases": 1,
        "paired_attempts": 1,
    }
    markdown = render_external_metrics_markdown(report)
    assert "NO SCORE (" not in markdown
    assert "-0.0000" not in markdown
