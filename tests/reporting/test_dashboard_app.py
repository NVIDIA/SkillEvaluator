# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual optional Streamlit app and its dataframes without mocks."""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
# st.dataframe otherwise imports these inside AppTest's timed script thread.
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
import streamlit as st
from streamlit.testing.v1 import AppTest

from skillevaluator.reporting import dashboard_app
from skillevaluator.reporting.dashboard_app import rows_csv

# A ceiling for cold imports under xdist and coverage; passing runs finish far sooner.
_APP_TIMEOUT_SECONDS = 60


def _write_report(
    path: Path,
    *,
    skill: str = "calculator",
    model: str = "model-a",
    harness: str = "opencode",
    features: list[str] | None = None,
    baseline_status: str = "succeeded",
    measured: bool = True,
    record_features: bool = True,
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
    if not record_features:
        del report["agents"][harness]["dashboard_metadata"]
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _app(paths: list[Path]) -> AppTest:
    script = f"from skillevaluator.reporting.dashboard_app import main\nmain({[str(path) for path in paths]!r})\n"
    return AppTest.from_string(script, default_timeout=_APP_TIMEOUT_SECONDS).run()


def _upload_app() -> AppTest:
    app = _app([])
    if not hasattr(app, "file_uploader"):
        pytest.skip("This Streamlit version's AppTest cannot drive st.file_uploader")
    return app


def _rewrite_score(path: Path, score: float) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    report["agents"]["opencode"]["with_skill"] = score
    report["agents"]["opencode"]["padding"] = "x" * 10  # change the size too, for coarse mtimes
    path.write_text(json.dumps(report), encoding="utf-8")


@pytest.fixture(autouse=True)
def _clear_dashboard_cache():
    # Cached loads are process-wide, so keep each test's inputs isolated.
    st.cache_data.clear()
    yield
    st.cache_data.clear()


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
    assert json.loads(app.dataframe[0].proto.columns)["duration_seconds"]["label"] == "Agent time (s)"
    assert [metric.value for metric in app.metric] == ["1", "1", "2", "2"]
    downloads = app.get("download_button")
    assert len(downloads) == 2
    assert all(button.proto.ignore_rerun for button in downloads)


def test_performance_table_shows_per_trial_usage(tmp_path: Path) -> None:
    path = _write_report(tmp_path / "run.json")
    app = _app([path])
    assert not app.exception
    performance = app.dataframe[0]
    assert performance.value["total_tokens_per_trial"].tolist() == [140.0, 120.0]
    config = json.loads(performance.proto.columns)
    assert config["total_tokens_per_trial"]["label"] == "Tokens / trial"
    for field in ("total_tokens_per_trial", "duration_seconds_per_trial", "cost_usd_per_trial"):
        assert config[field]["type_config"]["type"] == "number", field


def test_never_activated_dashboard_retains_scores_without_comparison_deltas(tmp_path: Path) -> None:
    path = _write_report(tmp_path / "never-activated.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    report["agents"]["opencode"]["skill_activation"] = {"status": "never_activated"}
    path.write_text(json.dumps(report), encoding="utf-8")
    app = _app([path])
    app.selectbox(key="comparison_metrics").set_value("Quality").run()
    assert not app.exception
    performance = app.dataframe[0].value
    comparison = app.dataframe[1].value
    assert performance["score"].tolist() == [0.9, 0.6]
    assert comparison["with_score"].iloc[0] == 0.9
    assert comparison["without_score"].iloc[0] == 0.6
    assert comparison["comparison_status"].iloc[0] == "inconclusive"
    assert "skill never activated" in comparison["comparison_reason"].iloc[0]
    assert comparison["score_delta"].isna().all()
    assert comparison["pass_rate_delta"].isna().all()
    rows = dashboard_app.load_dashboard_path(path).rows
    header = next(csv.reader(io.StringIO(rows_csv(rows, dashboard_app._PERFORMANCE_COLUMNS))))
    assert header.index("total_tokens_per_trial") == header.index("score") + 1


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
    assert "No rows match these filters" in app.info[1].value

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
    # Status narrows performance only, so the comparison still names the failed arm.
    comparison = app.dataframe[1].value
    assert comparison["comparison_status"].iloc[0] == "incomplete or mismatched coverage"
    assert comparison["comparison_reason"].iloc[0] == "without-skill arm status failed"
    assert comparison["coverage"].iloc[0] == "1 with / 1 without"
    assert comparison["score_delta"].isna().all()
    app.multiselect(key="filter_status").set_value(["failed"]).run()
    assert not app.exception
    assert app.dataframe[0].value["condition"].tolist() == ["without_skill"]
    assert app.dataframe[1].value["comparison_reason"].tolist() == ["without-skill arm status failed"]


def test_no_comparable_pair_hint(tmp_path: Path) -> None:
    app = _app([_write_report(tmp_path / "failed.json", baseline_status="failed")])
    assert not app.exception
    assert any("No pair is comparable" in message.value for message in app.info)
    app = _app([_write_report(tmp_path / "paired.json")])
    assert not app.exception
    assert not any("No pair is comparable" in message.value for message in app.info)


def test_unrecorded_features_caption(tmp_path: Path) -> None:
    app = _app([_write_report(tmp_path / "unrecorded.json", record_features=False)])
    assert not app.exception
    assert app.dataframe[0].value["features"].tolist() == ["unknown", "unknown"]
    assert any("No harness features are recorded" in caption.value for caption in app.caption)
    app = _app([_write_report(tmp_path / "recorded.json")])
    assert not app.exception
    assert not any("No harness features are recorded" in caption.value for caption in app.caption)


def test_bad_report_does_not_hide_other_results(tmp_path: Path) -> None:
    valid = _write_report(tmp_path / "valid.json")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{broken JSON", encoding="utf-8")
    app = _app([invalid, valid, tmp_path / "missing.json"])
    assert not app.exception
    assert len(app.warning) == 1
    assert len(app.text) == 2
    assert len(app.dataframe[0].value) == 2


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="Needs POSIX permissions enforced for this user")
def test_unreadable_subdirectory_does_not_hide_other_results(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_report(runs / "run.json")
    locked = runs / "locked"
    locked.mkdir()
    other = _write_report(tmp_path / "other.json", model="model-b")
    locked.chmod(0)
    try:
        app = _app([runs, other])
    finally:
        locked.chmod(0o755)
    assert not app.exception
    assert len(app.dataframe[0].value) == 4
    assert any(text.value.startswith(f"{locked.resolve()}: cannot discover reports") for text in app.text)


def test_input_that_fails_to_load_does_not_hide_other_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = dashboard_app.load_dashboard_path
    unreadable = _write_report(tmp_path / "unreadable.json")
    valid = _write_report(tmp_path / "valid.json", model="model-b")

    def failing(path: Path):
        # Stands in for any error the loader does not anticipate.
        if path == unreadable.resolve():
            raise PermissionError(13, "Permission denied", str(path))
        return real(path)

    monkeypatch.setattr(dashboard_app, "load_dashboard_path", failing)
    app = _app([unreadable, valid])
    assert not app.exception
    assert app.dataframe[0].value["model"].tolist() == ["model-b", "model-b"]
    assert any(text.value.startswith(f"{unreadable.resolve()}: cannot load dashboard input:") for text in app.text)


def test_oversized_trial_count_keeps_other_reports(tmp_path: Path) -> None:
    valid = _write_report(tmp_path / "valid.json")
    oversized = _write_report(tmp_path / "oversized.json")
    report = json.loads(oversized.read_text(encoding="utf-8"))
    agent = report["agents"]["opencode"]
    agent["num_trials"] = 10**20
    for trial in agent["trials"]:
        trial.pop("entry_id")
    oversized.write_text(json.dumps(report), encoding="utf-8")
    app = _app([valid, oversized])
    assert not app.exception
    assert len(app.dataframe[0].value) == 4
    assert any("invalid trial coverage counts" in text.value for text in app.text)


def test_report_notices_are_plain_text(tmp_path: Path) -> None:
    path = _write_report(tmp_path / "beacon.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    agent = report["agents"].pop("opencode")
    agent["num_trials"] = -1
    report["agents"]["![x](https://beacon.example.test/x.png)"] = agent
    path.write_text(json.dumps(report), encoding="utf-8")
    app = _app([path])
    assert not app.exception
    assert not any("](" in warning.value for warning in app.warning)
    assert any("beacon.example.test" in text.value for text in app.text)


def test_notice_count_matches_unique_notices(tmp_path: Path) -> None:
    report = _write_report(tmp_path / "run.json")
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["agents"]["opencode"]["num_trials"] = -1
    report.write_text(json.dumps(payload), encoding="utf-8")
    # The directory walk and the direct file path report the same notice for one file.
    app = _app([tmp_path, report])
    assert not app.exception
    assert app.expander[0].label == "Report notices (2)"
    assert len(app.text) == 2
    assert app.text[1].value == "Overlapping inputs: 2 duplicate condition rows loaded once."


def test_directory_and_file_inside_it_load_once(tmp_path: Path) -> None:
    report = _write_report(tmp_path / "run.json")
    app = _app([tmp_path, report])
    assert not app.exception
    assert len(app.dataframe[0].value) == 2
    comparison = app.dataframe[1].value
    assert comparison["comparison_status"].tolist() == ["comparable"]
    assert comparison["score_delta"].iloc[0] == pytest.approx(0.3)
    assert any("duplicate" in text.value for text in app.text)


def test_rewritten_file_inside_directory_input_is_reloaded(tmp_path: Path) -> None:
    report = _write_report(tmp_path / "run.json")
    # The directory comes first, and its own fingerprint misses an in-place rewrite of the file.
    app = _app([tmp_path, report])
    _rewrite_score(report, 0.5)
    app.run()
    assert not app.exception
    assert app.dataframe[0].value["score"].tolist() == [0.5, 0.6]


def test_results_dir_and_latest_symlink_load_once(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run = results / "run-1"
    run.mkdir(parents=True)
    _write_report(run / "report.json")
    try:
        (results / "latest").symlink_to(run, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks are unavailable on this host")
    app = _app([results, results / "latest"])
    assert not app.exception
    assert len(app.dataframe[0].value) == 2
    comparison = app.dataframe[1].value
    assert comparison["comparison_status"].tolist() == ["comparable"]
    assert comparison["score_delta"].iloc[0] == pytest.approx(0.3)
    assert any("duplicate" in text.value for text in app.text)


def test_status_filter_does_not_pair_arms_from_separate_embedded_reports(tmp_path: Path) -> None:
    path = _write_report(tmp_path / "combined.json", baseline_status="failed")
    first = json.loads(path.read_text(encoding="utf-8"))
    second = json.loads(path.read_text(encoding="utf-8"))
    second["agents"]["opencode"]["conditions"]["with_skill"]["execution_status"] = "failed"
    second["agents"]["opencode"]["conditions"]["without_skill"]["execution_status"] = "succeeded"
    path.write_text(json.dumps([first, second]), encoding="utf-8")
    app = _app([path])
    app.multiselect(key="filter_status").set_value(["succeeded"]).run()
    assert not app.exception
    comparisons = app.dataframe[1].value
    assert len(comparisons) == 2
    assert comparisons["comparison_status"].tolist() == ["incomplete or mismatched coverage"] * 2
    assert comparisons["comparison_reason"].tolist() == [
        "without-skill arm status failed",
        "with-skill arm status failed",
    ]
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
    assert len(app.text) == 1
    assert str(loop) in app.text[0].value
    assert len(app.dataframe[0].value) == 2


def test_filter_reruns_do_not_reload_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []
    real = dashboard_app.load_dashboard_path

    def counting(path: Path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(dashboard_app, "load_dashboard_path", counting)
    path = _write_report(tmp_path / "run.json")
    app = _app([path])
    for value in (["model-a"], [], ["model-a"]):
        app.multiselect(key="filter_model").set_value(value).run()
        assert not app.exception
    assert calls == [path.resolve()]


def test_rewritten_report_file_is_reloaded_on_next_rerun(tmp_path: Path) -> None:
    path = _write_report(tmp_path / "run.json")
    app = _app([path])
    assert app.dataframe[0].value["score"].tolist() == [0.9, 0.6]
    _rewrite_score(path, 0.5)
    app.run()
    assert not app.exception
    assert app.dataframe[0].value["score"].tolist() == [0.5, 0.6]


def test_reload_button_rereads_nested_directory_edits(tmp_path: Path) -> None:
    nested = tmp_path / "runs" / "a"
    nested.mkdir(parents=True)
    path = _write_report(nested / "run.json")
    app = _app([tmp_path / "runs"])
    assert app.dataframe[0].value["score"].tolist() == [0.9, 0.6]
    _rewrite_score(path, 0.5)
    app.button(key="reload_reports").click().run()
    assert not app.exception
    assert app.dataframe[0].value["score"].tolist() == [0.5, 0.6]


def test_uploads_are_parsed_once_per_uploaded_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real = dashboard_app.parse_dashboard_report

    def counting(payload, source):
        calls.append(source)
        return real(payload, source)

    monkeypatch.setattr(dashboard_app, "parse_dashboard_report", counting)
    content = _write_report(tmp_path / "upload.json").read_bytes()
    app = _upload_app()
    app.file_uploader(key="report_uploads").set_value([("upload.json", content, "application/json")]).run()
    assert len(app.dataframe[0].value) == 2
    app.multiselect(key="filter_condition").set_value(["with_skill"]).run()
    assert not app.exception
    assert calls == ["upload:1:upload.json"]


def test_uploaded_report_renders_with_upload_source(tmp_path: Path) -> None:
    content = _write_report(tmp_path / "run.json").read_bytes()
    app = _upload_app()
    app.file_uploader(key="report_uploads").set_value(("run.json", content, "application/json")).run()
    assert not app.exception
    performance = app.dataframe[0].value
    assert performance["source"].tolist() == ["upload:1:run.json", "upload:1:run.json"]
    assert performance["total_tokens"].tolist() == [140.0, 120.0]


def test_upload_batch_reports_oversize_and_invalid_and_keeps_valid(tmp_path: Path) -> None:
    content = _write_report(tmp_path / "run.json").read_bytes()
    oversized = content + b" " * (16_000_001 - len(content))
    app = _upload_app()
    app.file_uploader(key="report_uploads").set_value(
        [
            ("big.json", oversized, "application/json"),
            ("broken.json", b"{", "application/json"),
            ("deep.json", b"[" * 100_000, "application/json"),
            ("run.json", content, "application/json"),
        ]
    ).run()
    assert not app.exception
    notices = [text.value for text in app.text]
    assert any(notice.startswith("big.json: report exceeds 16 MB") for notice in notices), notices
    assert sum("invalid JSON" in notice for notice in notices) == 2, notices
    assert len(app.dataframe[0].value) == 2


def test_identical_uploads_are_loaded_once(tmp_path: Path) -> None:
    blob = _write_report(tmp_path / "run.json").read_bytes()
    app = _upload_app()
    app.file_uploader(key="report_uploads").set_value([("run.json", blob, "application/json")] * 2).run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 2
    assert [metric.value for metric in app.metric] == ["1", "1", "2", "2"]
    assert any("identical to an earlier upload; skipped." in text.value for text in app.text)

    other = _write_report(tmp_path / "other.json", model="model-b").read_bytes()
    app.file_uploader(key="report_uploads").set_value(
        [("run.json", blob, "application/json"), ("run.json", other, "application/json")]
    ).run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 4
    assert not any("identical to an earlier upload" in text.value for text in app.text)


def test_upload_at_exact_limit_is_accepted(tmp_path: Path) -> None:
    content = _write_report(tmp_path / "run.json").read_bytes()
    exact = content + b" " * (16_000_000 - len(content))
    assert len(exact) == 16_000_000
    app = _upload_app()
    app.file_uploader(key="report_uploads").set_value(("exact.json", exact, "application/json")).run()
    assert not app.exception
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
    assert config["comparison_reason"]["type_config"]["type"] == "text"
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


@pytest.mark.parametrize(
    "value",
    ["\r=1", "\n=1", "\xa0=1", "\u3000=1", "\uff1d1", "\uff0b1", "\uff0d1", "\uff20A1"],
)
def test_csv_escapes_formula_prefixes_after_whitespace_and_width_folding(value: str) -> None:
    exported = rows_csv([{"skill": value, "score_delta": -0.25, "cost_usd": 7}], ("skill", "score_delta", "cost_usd"))
    cells = list(csv.DictReader(io.StringIO(exported)))
    assert cells == [{"skill": "'" + value, "score_delta": "-0.25", "cost_usd": "7"}]
