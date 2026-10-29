# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every report names the quality config that produced the Tier 1 quality score."""

import json

import pytest

from skillevaluator.models import ValidationResult
from skillevaluator.reporting import BenchmarkReporter, JSONReporter, MarkdownReporter

_DIGEST = "sha256:" + "a" * 64


def _quality_result(quality_config: dict | None) -> ValidationResult:
    result = ValidationResult(validator_name="QUALITY", validator_description="Quality score")
    if quality_config is not None:
        result.metadata["quality_config"] = quality_config
    return result


def _benchmark_line(results: list[ValidationResult], field: str = "Quality config") -> str | None:
    card = BenchmarkReporter().render_all(results)
    return next((line for line in card.splitlines() if line.startswith(f"- {field}:")), None)


def test_json_report_records_quality_config() -> None:
    config = {"source": "custom", "file": "team-quality.yaml", "digest": _DIGEST, "min_score": 80}

    data = json.loads(JSONReporter().render_all([_quality_result(config)]))

    assert data["quality_config"] == config


def test_markdown_report_records_quality_config() -> None:
    report = MarkdownReporter().render_all(
        [_quality_result({"source": "custom", "file": "team-quality.yaml", "digest": _DIGEST, "min_score": 80})]
    )

    assert f"**Quality config:** custom (`team-quality.yaml`), `{_DIGEST}`, min score 80" in report


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(
            {"source": "custom", "file": "team-quality.yaml", "digest": _DIGEST},
            f"- Quality config: custom (team-quality.yaml), `{_DIGEST}`",
            id="custom",
        ),
        pytest.param(
            {"source": "packaged", "file": None, "digest": _DIGEST},
            f"- Quality config: packaged, `{_DIGEST}`",
            id="packaged",
        ),
        pytest.param(
            {"source": "custom", "file": "team_*rules*.yaml", "digest": _DIGEST},
            f"- Quality config: custom (team\\_\\*rules\\*.yaml), `{_DIGEST}`",
            id="markdown-escaped-file-name",
        ),
    ],
)
def test_benchmark_records_quality_config(config: dict, expected: str) -> None:
    assert _benchmark_line([_quality_result(config)]) == expected


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(None, id="not-recorded"),
        pytest.param({"source": "custom", "file": "x.yaml", "digest": "sha256:`injected`"}, id="malformed-digest"),
        pytest.param({"source": "other", "file": "x.yaml", "digest": _DIGEST}, id="unknown-source"),
    ],
)
def test_benchmark_omits_missing_or_malformed_quality_config(config: dict | None) -> None:
    assert _benchmark_line([_quality_result(config)]) is None


def test_benchmark_records_quality_min_score() -> None:
    config = {"source": "packaged", "file": None, "digest": _DIGEST, "min_score": 80}

    assert _benchmark_line([_quality_result(config)], "Quality min score") == "- Quality min score: 80"


@pytest.mark.parametrize("min_score", [None, True, "80", float("nan")], ids=["missing", "bool", "string", "nan"])
def test_benchmark_omits_missing_or_malformed_quality_min_score(min_score: object) -> None:
    config = {"source": "packaged", "file": None, "digest": _DIGEST, "min_score": min_score}

    assert _benchmark_line([_quality_result(config)], "Quality min score") is None
