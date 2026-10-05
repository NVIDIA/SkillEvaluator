# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual optional Streamlit app and its dataframes without mocks."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from skillevaluator.reporting.dashboard_app import rows_csv


def _write_report(
    path: Path,
    *,
    skill: str = "calculator",
    model: str = "model-a",
    harness: str = "opencode",
    features: list[str] | None = None,
    baseline_status: str = "succeeded",
    measured: bool = True,
) -> Path:
    def trial(score: float, tokens: int) -> dict:
        value = {"entry_id": "case-1", "trial_id": "case-1__attempt1", "overall": score}
        if measured:
            value.update({"tokens": {"prompt": tokens, "completion": 20, "cached": 0}})
        return value

    report = {
        "schema_version": "2.0",
        "skill_name": skill,
        "run_id": path.stem,
        "agents": {
            harness: {
                "name": harness,
                "model": model,
                "dashboard_metadata": {"harness_features": features if features is not None else ["tools"]},
                "execution_status": "succeeded",
                "conditions": {
                    "with_skill": {"execution_status": "succeeded", "expected_attempts": 1, "scored_attempts": 1},
                    "without_skill": {
                        "execution_status": baseline_status,
                        "expected_attempts": 1,
                        "scored_attempts": 1,
                    },
                },
                "with_skill": 0.9,
                "baseline": 0.6,
                "num_trials": 1,
                "num_trials_baseline": 1,
                "trials": [trial(0.9, 120)],
                "trials_baseline": [trial(0.6, 100)],
                "pass_at_k": {"with_skill": {"rate": 1.0}, "without_skill": {"rate": 0.0}},
                "raw_transcript": "private execution text must stay out of the UI",
            },
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _app(paths: list[Path]) -> AppTest:
    script = f"from skillevaluator.reporting.dashboard_app import main\nmain({[str(path) for path in paths]!r})\n"
    return AppTest.from_string(script, default_timeout=15).run()


def test_empty_dashboard_has_upload_direction() -> None:
    app = _app([])
    assert not app.exception
    assert [message.value for message in app.info] == ["Upload JSON evaluation reports to start comparing results."]
    assert len(app.dataframe) == 0


def test_dashboard_renders_numeric_performance_and_paired_comparison(tmp_path: Path) -> None:
    path = _write_report(tmp_path / "run-1.json")
    app = _app([path, path])
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Performance", "With vs without skill"]
    performance = app.dataframe[0].value
    comparison = app.dataframe[1].value
    assert len(performance) == 2
    assert performance["score"].tolist() == [0.9, 0.6]
    assert performance["total_tokens"].tolist() == [140.0, 120.0]
    assert performance["cost_usd"].isna().all()
    assert performance["cached_tokens"].tolist() == [0.0, 0.0]
    assert comparison["score_delta"].iloc[0] == pytest.approx(0.3)
    assert comparison["total_tokens_delta"].iloc[0] == 20.0
    assert comparison["total_tokens_percent"].iloc[0] == pytest.approx(100 / 6)
    assert comparison["comparison_status"].iloc[0] == "comparable"
    assert "raw_transcript" not in performance.columns
    assert all(not column.startswith("_") for column in performance.columns)
    assert [metric.value for metric in app.metric] == ["1", "1", "2", "2"]


def test_dimension_filters_and_condition_retain_baseline_for_comparison(tmp_path: Path) -> None:
    paths = [
        _write_report(tmp_path / "a.json"),
        _write_report(tmp_path / "b.json", skill="summarizer", model="model-b", harness="claude-code", features=[]),
        _write_report(tmp_path / "c.json", skill="calculator", model="model-c", features=["tools", "memory"]),
    ]
    app = _app(paths)
    app.multiselect(key="filter_skill").set_value(["calculator"])
    app.multiselect(key="filter_model").set_value(["model-a"])
    app.multiselect(key="filter_harness").set_value(["opencode"])
    app.multiselect(key="filter_features").set_value(["tools"])
    app.multiselect(key="filter_status").set_value(["succeeded"])
    app.multiselect(key="filter_condition").set_value(["with_skill"])
    app.run()
    assert not app.exception
    assert app.dataframe[0].value["condition"].tolist() == ["with_skill"]
    assert app.dataframe[1].value["score_delta"].iloc[0] == pytest.approx(0.3)
    assert [metric.value for metric in app.metric] == ["1", "1", "1", "1"]

    app.multiselect(key="filter_model").set_value(["model-b"]).run()
    assert not app.exception
    assert len(app.dataframe) == 0
    assert "No rows match these filters" in app.info[0].value
    assert "No comparisons are available" in app.info[1].value

    for widget in app.multiselect:
        widget.set_value([])
    app.run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 6
    assert len(app.dataframe[1].value) == 3


def test_unmeasured_and_failed_baseline_are_not_zero_or_comparable(tmp_path: Path) -> None:
    path = _write_report(tmp_path / "failed.json", baseline_status="failed", measured=False)
    app = _app([path])
    assert not app.exception
    performance = app.dataframe[0].value
    assert performance["total_tokens"].isna().all()
    assert performance["duration_seconds"].isna().all()
    assert performance.loc[performance["condition"] == "without_skill", "score"].isna().all()
    comparison = app.dataframe[1].value
    assert comparison["score_delta"].isna().all()
    assert comparison["comparison_status"].iloc[0] == "incomplete or mismatched coverage"
    app.multiselect(key="filter_status").set_value(["succeeded"]).run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 1
    assert app.dataframe[1].value["comparison_status"].iloc[0] == "missing condition"


def test_bad_report_does_not_hide_other_results(tmp_path: Path) -> None:
    valid = _write_report(tmp_path / "valid.json")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{broken JSON", encoding="utf-8")
    app = _app([invalid, valid, tmp_path / "missing.json"])
    assert not app.exception
    assert len(app.warning) == 2
    assert len(app.dataframe[0].value) == 2


def test_status_filter_does_not_pair_arms_from_separate_embedded_reports(tmp_path: Path) -> None:
    path = _write_report(tmp_path / "combined.json", baseline_status="failed")
    first = json.loads(path.read_text())
    second = json.loads(path.read_text())
    second["agents"]["opencode"]["conditions"]["with_skill"]["execution_status"] = "failed"
    second["agents"]["opencode"]["conditions"]["without_skill"]["execution_status"] = "succeeded"
    path.write_text(json.dumps([first, second]))
    app = _app([path])
    app.multiselect(key="filter_status").set_value(["succeeded"]).run()
    assert not app.exception
    comparisons = app.dataframe[1].value
    assert len(comparisons) == 2
    assert comparisons["comparison_status"].tolist() == ["missing condition", "missing condition"]
    assert comparisons["score_delta"].isna().all()


def test_cyclic_input_path_does_not_hide_valid_report(tmp_path: Path) -> None:
    valid = _write_report(tmp_path / "valid.json")
    loop = tmp_path / "loop.json"
    try:
        loop.symlink_to(loop)
    except OSError:
        pytest.skip("Symlinks are unavailable on this host")
    app = _app([loop, valid])
    assert not app.exception
    assert len(app.warning) == 1
    assert str(loop) in app.warning[0].value
    assert len(app.dataframe[0].value) == 2


@pytest.mark.parametrize(
    ("view", "measured_column"),
    [
        ("Overview", "total_tokens_delta"),
        ("Quality", "with_pass_rate"),
        ("Tokens", "with_total_tokens"),
        ("Duration", "with_duration_seconds"),
        ("Cost", "with_cost_usd"),
        ("All metrics", "without_prompt_tokens"),
    ],
)
def test_comparison_views_keep_measurements_numeric(tmp_path: Path, view: str, measured_column: str) -> None:
    app = _app([_write_report(tmp_path / "run.json")])
    app.selectbox(key="comparison_metrics").set_value(view).run()
    assert not app.exception
    comparison = app.dataframe[1]
    assert measured_column in comparison.value.columns
    config = json.loads(comparison.proto.columns)
    assert config[measured_column]["type_config"]["type"] == "number"
    assert config["comparison_status"]["type_config"]["type"] == "text"
    assert comparison.value["comparison_status"].iloc[0] == "comparable"
    assert len(comparison.value.columns) == len(set(comparison.value.columns))
    if view == "All metrics":
        for field in comparison.value.columns:
            if any(metric in field for metric in ("tokens", "duration_seconds", "cost_usd", "score", "pass_rate")):
                assert config[field]["type_config"]["type"] == "number", field


def test_csv_preserves_numeric_deltas_and_escapes_formula_strings() -> None:
    exported = rows_csv(
        [
            {"skill": ' =HYPERLINK("https://example.com")', "score_delta": -0.25, "cost_usd": None},
            {"skill": "\tprivate-name", "score_delta": 0.0, "cost_usd": 0.0},
            {"skill": "@SUM(A1)", "score_delta": 1.25, "cost_usd": 0.1, "_secret": "must not be exported"},
        ],
        ("skill", "score_delta", "cost_usd"),
    )
    cells = list(csv.DictReader(io.StringIO(exported)))
    assert cells[0] == {"skill": '\' =HYPERLINK("https://example.com")', "score_delta": "-0.25", "cost_usd": ""}
    assert cells[1] == {"skill": "'\tprivate-name", "score_delta": "0.0", "cost_usd": "0.0"}
    assert cells[2]["skill"] == "'@SUM(A1)"
    assert "must not be exported" not in exported
