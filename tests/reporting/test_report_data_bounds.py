# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from skillevaluator.tier3.harbor import report_data


def _write_summary(agent_dir: Path) -> None:
    summary = agent_dir / "with-skill" / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(
            {
                "scores": {"accuracy": 1.0},
                "metrics": ["accuracy"],
                "num_trials": 1,
                "execution_status": "succeeded",
                "expected_attempts": 1,
                "scored_attempts": 1,
            }
        ),
        encoding="utf-8",
    )


def _write_trial(agent_dir: Path, trial_name: str, reward: dict, trajectory: dict | None = None) -> None:
    trial_dir = agent_dir / "with-skill" / "trials" / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "reward.json").write_text(json.dumps(reward), encoding="utf-8")
    if trajectory is not None:
        (trial_dir / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")


def _reasons(agent: dict) -> list[dict]:
    marker = agent.get("_report_truncation", {})
    return marker.get("reasons", []) if isinstance(marker, dict) else []


def test_normal_agent_artifacts_are_loaded_without_truncation_marker(tmp_path: Path) -> None:
    agent_dir = tmp_path / "codex"
    _write_summary(agent_dir)
    _write_trial(
        agent_dir,
        "case-001__1",
        {"entry_id": "case-001", "accuracy": 1.0},
        {
            "steps": [{"action": "answer"}],
            "final_metrics": {
                "total_prompt_tokens": 10,
                "total_completion_tokens": 4,
                "total_cached_tokens": 2,
            },
        },
    )

    agents = report_data.load_agent_data(tmp_path)

    assert agents["codex"]["with_skill"] == {"accuracy": 1.0}
    assert agents["codex"]["rewards"] == [
        {
            "entry_id": "case-001",
            "accuracy": 1.0,
            "_traj": {"steps": 1, "prompt_tokens": 10, "completion_tokens": 4, "cached_tokens": 2},
        }
    ]
    assert "_report_truncation" not in agents["codex"]


@pytest.mark.parametrize(
    ("metrics", "expected_tokens", "expected_total", "native_total"),
    [
        ({}, {"prompt": None, "completion": None, "cached": None}, None, 22),
        ({"total_prompt_tokens": 12}, {"prompt": 12, "completion": None, "cached": None}, None, 14),
        (
            {"total_prompt_tokens": 0, "total_completion_tokens": 0, "total_cached_tokens": 0},
            {"prompt": 0, "completion": 0, "cached": 0},
            0,
            0,
        ),
        (
            {"total_prompt_tokens": 10, "total_completion_tokens": 4, "total_cached_tokens": 2},
            {"prompt": 10, "completion": 4, "cached": 2},
            14,
            14,
        ),
    ],
)
def test_missing_token_measurements_stay_unknown_in_canonical_exports(
    tmp_path: Path, metrics: dict, expected_tokens: dict, expected_total: int | None, native_total: int
) -> None:
    from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory
    from skillevaluator.reporting import HTMLReporter, JSONReporter
    from skillevaluator.reporting.dashboard_data import load_dashboard_path, parse_dashboard_report

    agent_dir = tmp_path / "codex"
    _write_summary(agent_dir)
    _write_trial(
        agent_dir,
        "case-001__1",
        {"entry_id": "case-001", "trial_id": "case-001__1", "accuracy": 1.0},
        {"steps": [], "final_metrics": metrics},
    )
    (tmp_path / "result.json").write_text(json.dumps({"skill_name": "token-measurements"}), encoding="utf-8")
    (tmp_path / "run_config.json").write_text(
        json.dumps({"agents": {"codex": {"agent": "codex", "model": "test-model"}}}), encoding="utf-8"
    )
    # Retained engine counters can recover usage even when an exported
    # trajectory has no counters. The canonical export must not invent zeros.
    (agent_dir / "with-skill" / "trials" / "case-001__1" / "result.json").write_text(
        json.dumps({"agent_result": {"n_input_tokens": 20, "n_output_tokens": 2}}), encoding="utf-8"
    )

    result = agent_eval_result_from_directory(Path("token-measurements"), tmp_path, use_llm_judge=False)
    assert result is not None
    canonical = result.metadata["agent_eval"]
    assert canonical["agents"]["codex"]["trials"][0]["tokens"] == expected_tokens
    exported = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    assert exported["results"][0]["tier3"]["agents"]["codex"]["trials"][0]["tokens"] == expected_tokens
    assert parse_dashboard_report(exported, "report.json").rows[0]["total_tokens"] == expected_total
    assert load_dashboard_path(tmp_path).rows[0]["total_tokens"] == native_total
    # Existing HTML token consumers accept null counters without an exception.
    assert "tier3-full" in HTMLReporter(include_timestamp=False).render_all([result])


def test_agent_directory_symlink_is_not_discovered(tmp_path: Path) -> None:
    outside_agent = tmp_path / "outside" / "codex"
    _write_summary(outside_agent)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "codex").symlink_to(outside_agent, target_is_directory=True)

    assert report_data.load_agent_data(results_dir) == {}


@pytest.mark.parametrize("escape", ["condition", "trials", "trial"])
def test_intermediate_directory_symlinks_are_not_followed(tmp_path: Path, escape: str) -> None:
    results_dir = tmp_path / "results"
    agent_dir = results_dir / "codex"
    outside = tmp_path / "outside"
    if escape == "condition":
        _write_summary(outside)
        agent_dir.mkdir(parents=True)
        (agent_dir / "with-skill").symlink_to(outside / "with-skill", target_is_directory=True)
    else:
        _write_summary(agent_dir)
        outside_trial = outside / "case-001__1"
        outside_trial.mkdir(parents=True)
        (outside_trial / "reward.json").write_text(
            json.dumps({"entry_id": "case-001", "accuracy": 1.0}),
            encoding="utf-8",
        )
        trials_dir = agent_dir / "with-skill" / "trials"
        if escape == "trials":
            trials_dir.symlink_to(outside, target_is_directory=True)
        else:
            trials_dir.mkdir()
            (trials_dir / "case-001__1").symlink_to(outside_trial, target_is_directory=True)

    agents = report_data.load_agent_data(results_dir)

    if escape == "condition":
        assert agents == {}
    else:
        assert agents["codex"]["rewards"] == []


def test_oversized_summary_reward_and_trajectory_are_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(report_data, "_MAX_JSON_BYTES", 256)

    oversized_summary = tmp_path / "oversized-summary" / "with-skill" / "summary.json"
    oversized_summary.parent.mkdir(parents=True)
    oversized_summary.write_text(json.dumps({"scores": {}, "padding": "x" * 300}), encoding="utf-8")

    reward_agent = tmp_path / "oversized-reward"
    _write_summary(reward_agent)
    _write_trial(reward_agent, "case-001__1", {"entry_id": "case-001", "padding": "x" * 300})

    trajectory_agent = tmp_path / "oversized-trajectory"
    _write_summary(trajectory_agent)
    _write_trial(
        trajectory_agent,
        "case-001__1",
        {"entry_id": "case-001", "accuracy": 1.0},
        {"steps": [], "padding": "x" * 300},
    )

    agents = report_data.load_agent_data(tmp_path)

    assert "oversized-summary" not in agents
    assert agents["oversized-reward"]["rewards"] == []
    assert any(reason["artifact"] == "reward" for reason in _reasons(agents["oversized-reward"]))
    assert "_traj" not in agents["oversized-trajectory"]["rewards"][0]
    assert any(reason["artifact"] == "trajectory" for reason in _reasons(agents["oversized-trajectory"]))


@pytest.mark.parametrize(
    ("limit_name", "limit", "reward", "expected_code"),
    [
        (
            "_MAX_JSON_DEPTH",
            3,
            {"entry_id": "case-001", "details": {"a": {"b": {"c": 1}}}},
            "json_depth",
        ),
        (
            "_MAX_JSON_NODES",
            20,
            {"entry_id": "case-001", "values": list(range(30))},
            "json_nodes",
        ),
    ],
)
def test_pathological_json_is_skipped_before_report_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit: int,
    reward: dict,
    expected_code: str,
) -> None:
    monkeypatch.setattr(report_data, limit_name, limit)
    agent_dir = tmp_path / "codex"
    _write_summary(agent_dir)
    _write_trial(agent_dir, "case-001__1", reward)

    agents = report_data.load_agent_data(tmp_path)

    assert agents["codex"]["rewards"] == []
    assert any(reason["code"] == expected_code for reason in _reasons(agents["codex"]))


def test_huge_json_integer_is_rejected_without_crashing() -> None:
    diagnostics: list[dict] = []
    raw = ('{"scores": {"accuracy": ' + "9" * 10_000 + "}}").encode()

    loaded = report_data._decode_bounded_json(raw, diagnostics, artifact="summary")

    assert loaded is report_data._INVALID_JSON
    assert any(reason["code"] == "json_number" for reason in diagnostics)


def test_excess_trials_are_capped_in_name_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(report_data, "_MAX_TRIALS_PER_CONDITION", 2)
    agent_dir = tmp_path / "codex"
    _write_summary(agent_dir)
    for name in ("case-c", "case-a", "case-b"):
        _write_trial(agent_dir, name, {"entry_id": name, "accuracy": 1.0})

    agents = report_data.load_agent_data(tmp_path)

    assert [reward["entry_id"] for reward in agents["codex"]["rewards"]] == ["case-a", "case-b"]
    assert any(reason["code"] == "trial_limit" and reason["limit"] == 2 for reason in _reasons(agents["codex"]))


def test_staged_tasks_and_dataset_records_are_capped_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(report_data, "_MAX_STAGED_TASKS", 2)
    monkeypatch.setattr(report_data, "_MAX_DATASET_RECORDS", 2)
    for name in ("task-c", "task-a", "task-b"):
        tests_dir = tmp_path / "run" / "_harbor-tasks" / name / "tests"
        tests_dir.mkdir(parents=True)
        (tests_dir / "entry.json").write_text(json.dumps({"id": name}), encoding="utf-8")

    evals_dir = tmp_path / "skill" / "evals"
    evals_dir.mkdir(parents=True)
    (evals_dir / "evals.json").write_text(
        json.dumps([{"id": "case-c"}, {"id": "case-a"}, {"id": "case-b"}]),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger=report_data.__name__):
        staged = report_data.load_staged_harbor_dataset(tmp_path / "run")
        dataset = report_data.load_dataset(tmp_path / "skill")

    assert [entry["id"] for entry in staged] == ["task-a", "task-b"]
    assert [entry["id"] for entry in dataset] == ["case-c", "case-a"]
    assert "staged_task_limit" in caplog.text
    assert "dataset_record_limit" in caplog.text


def test_json_reader_does_not_use_unbounded_path_read_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"ok": true}', encoding="utf-8")
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _self: pytest.fail("bounded JSON loading must not call Path.read_bytes"),
    )

    diagnostics: list[dict] = []
    loaded = report_data._load_bounded_json(artifact, diagnostics, artifact="test")

    assert loaded == {"ok": True}
    assert diagnostics == []


def test_path_selection_stops_at_the_visit_budget() -> None:
    visited: list[int] = []

    def paths():
        for index in range(100):
            visited.append(index)
            yield Path(f"item-{index:03d}")

    selected, selection_truncated, scan_truncated = report_data._bounded_smallest(paths(), 2, scan_limit=3)

    assert [path.name for path in selected] == ["item-000", "item-001"]
    assert selection_truncated is True
    assert scan_truncated is True
    assert visited == [0, 1, 2, 3]


def test_malformed_jsonl_rejects_the_entire_candidate(tmp_path: Path) -> None:
    evals_dir = tmp_path / "skill" / "evals"
    evals_dir.mkdir(parents=True)
    (evals_dir / "evals.jsonl").write_text(
        '{"id": "case-a"}\n{"id": broken}\n{"id": "case-b"}\n',
        encoding="utf-8",
    )

    assert report_data.load_dataset(tmp_path / "skill") == []


def test_yaml_json_shape_limit_emits_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(report_data, "_MAX_JSON_DEPTH", 2)
    evals_dir = tmp_path / "skill" / "evals"
    evals_dir.mkdir(parents=True)
    (evals_dir / "evals.yaml").write_text(
        "evals:\n  - id: case-a\n    nested:\n      deeper: value\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger=report_data.__name__):
        dataset = report_data.load_dataset(tmp_path / "skill")

    assert dataset == []
    assert "json_depth" in caplog.text


def test_loader_truncation_reaches_the_canonical_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.evaluation.tier3_report import build_agent_eval_payload

    monkeypatch.setattr(report_data, "_MAX_TRIALS_PER_CONDITION", 1)
    monkeypatch.setattr(report_data, "_MAX_DATASET_RECORDS", 1)
    agent_dir = tmp_path / "run" / "codex"
    _write_summary(agent_dir)
    _write_trial(agent_dir, "case-a", {"entry_id": "case-a", "accuracy": 1.0})
    _write_trial(agent_dir, "case-b", {"entry_id": "case-b", "accuracy": 1.0})
    evals_dir = tmp_path / "skill" / "evals"
    evals_dir.mkdir(parents=True)
    (evals_dir / "evals.json").write_text(
        json.dumps([{"id": "case-a"}, {"id": "case-b"}]),
        encoding="utf-8",
    )

    agents = report_data.load_agent_data(tmp_path / "run")
    dataset = report_data.load_dataset(tmp_path / "skill")
    payload = build_agent_eval_payload("skill", agents, dataset=dataset, use_llm_judge=False)

    assert payload is not None
    reasons = payload["report_truncation"]["artifact_loading"]
    assert {reason["code"] for reason in reasons} == {"dataset_record_limit", "trial_limit"}
