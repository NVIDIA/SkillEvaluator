# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical, evidence, orchestration, and CLI contracts for skill interactions."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.evaluation import EvaluationOptions, EvaluationService
from skillevaluator.tier3.harbor.metrics import CUSTOM_SCORE_POLICY, DEFAULT_METRIC_SET, DEFAULT_SCORE_POLICY
from skillevaluator.tier3.interactions import compare_configurations, interaction_scores, run_interaction


def _skill(root: Path, name: str) -> Path:
    skill = root / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Help with {name}.\n---\n# {name}\n", encoding="utf-8"
    )
    return skill


def _source(root: Path, config: str = "") -> Path:
    # Keep the source checkout separate from the test operator's key, even with an in-repo TMPDIR.
    skill = _skill(root / "suite-repo", "reviewed-suite")
    subprocess.run(["git", "init", "-q", str(skill.parent)], check=True)
    (skill / "evals").mkdir()
    (skill / "evals" / "evals.json").write_text(
        json.dumps(
            {
                "skill_name": skill.name,
                "evals": [{"id": "case-1", "prompt": "Solve the task", "expected_output": "A correct answer"}],
            }
        ),
        encoding="utf-8",
    )
    if config:
        (skill / "evals" / "config.yml").write_text("schema_version: 1\n" + config, encoding="utf-8")
    return skill


def _evidence(
    root: Path,
    values: dict[str, list[float]],
    *,
    n_attempts: int = 1,
    condition: str = "with-skill",
    custom_only: bool = False,
) -> dict[str, Any]:
    root.mkdir(parents=True)
    entries = [
        {
            "id": case_id,
            "question": f"Solve {case_id}",
            "ground_truth": "answer",
            "has_skill": condition == "with-skill",
        }
        for case_id in values
    ]
    (root / "dataset_snapshot.json").write_text(json.dumps({"dataset": entries}), encoding="utf-8")
    for case_id, scores in values.items():
        for ordinal, value in enumerate(scores):
            trial = f"{case_id}-{ordinal}"
            directory = root / "codex" / condition / "trials" / trial
            directory.mkdir(parents=True)
            reward = {
                "entry_id": case_id,
                "trial_id": trial,
                "goal_accuracy": value,
                "accuracy": value,
                "metric_set": "custom-only" if custom_only else DEFAULT_METRIC_SET,
                "score_policy": CUSTOM_SCORE_POLICY if custom_only else DEFAULT_SCORE_POLICY,
                "details": {
                    "goal_accuracy": {"judge_model_used": "same-judge", "method": "custom"},
                    "accuracy": {"judge_model_used": "same-judge", "method": "custom"},
                },
                **({"overall": value} if custom_only else {}),
            }
            (directory / "reward.json").write_text(json.dumps(reward), encoding="utf-8")
    return {
        "execution_status": "succeeded",
        "run_dir": str(root),
        "evaluator_version": "test-version",
        "agents": {"codex": {"model": "same-agent-model"}},
        "run_config": {
            "harbor": {
                "n_attempts": n_attempts,
                "stop_on_pass": False,
                "environment": {"value": "docker", "source": "CLI"},
            },
            "grading": {"mode": "custom_only" if custom_only else "default"},
            "judge": {"enabled": not custom_only, "model": "same-judge"},
        },
    }


def _matrix(tmp_path: Path, scores=(0.50, 0.75, 0.70, 0.65), *, custom_only=False) -> list[dict[str, Any]]:
    configurations = []
    for index, score in enumerate(scores):
        condition = "without-skill" if index == 0 else "with-skill"
        configurations.append(
            {
                "id": f"config-{index}",
                "condition": condition,
                "result": _evidence(
                    tmp_path / f"config-{index}", {"case-1": [score]}, condition=condition, custom_only=custom_only
                ),
            }
        )
    return configurations


@pytest.mark.parametrize(
    ("baseline", "individuals", "combined", "interaction", "classification"),
    [
        (0.50, [0.75, 0.70], 0.65, -0.30, "negative"),
        (0.20, [0.30, 0.35], 0.60, 0.15, "positive"),
        (0.20, [0.30, 0.35], 0.45, 0.0, "zero"),
        (0.20, [0.30, 0.35, 0.40], 0.80, 0.15, "positive"),
        (0.50, [0.20, 0.30], 0.10, 0.10, "positive"),
    ],
)
def test_signed_interaction_equation(baseline, individuals, combined, interaction, classification):
    result = interaction_scores(baseline, individuals, combined)
    assert result["interaction"] == pytest.approx(interaction)
    assert result["classification"] == classification
    assert result["combined_lift"] == pytest.approx(combined - baseline)


@pytest.mark.parametrize("invalid", [True, float("nan"), float("inf"), -0.1, 1.1, "0.5"])
def test_rejects_invalid_outcome_scores(invalid):
    with pytest.raises(ValueError, match="finite numbers"):
        interaction_scores(0.1, [0.2, invalid], 0.3)


def test_requires_two_individuals():
    with pytest.raises(ValueError, match="at least two"):
        interaction_scores(0.1, [0.2], 0.3)


def test_compares_paired_outcomes_and_keeps_trial_evidence(tmp_path):
    report = compare_configurations(_matrix(tmp_path), metric="goal_accuracy", n_attempts=1)
    assert report["status"] == "complete"
    score = report["agents"]["codex"]
    assert score["combined_lift"] == pytest.approx(0.15)
    assert score["interaction"] == pytest.approx(-0.30)
    assert score["classification"] == "negative"
    assert score["cases"]["case-1"]["interaction"] == pytest.approx(-0.30)
    assert Path(score["evidence"]["config-0"]["case-1"][0]["reward_path"]).is_file()


def test_catalog_probe_status_does_not_change_judge_identity(tmp_path):
    matrix = _matrix(tmp_path)
    for arm in matrix:
        arm["result"]["run_config"]["judge"]["catalog_verification"] = "verified"
    judge = matrix[-1]["result"]["run_config"]["judge"]
    judge["catalog_verification"] = "degraded"

    report = compare_configurations(matrix, metric="goal_accuracy", n_attempts=1)
    assert report["status"] == "complete"
    assert judge["catalog_verification"] == "degraded"

    judge["override_applied"] = True
    report = compare_configurations(matrix, metric="goal_accuracy", n_attempts=1)
    assert report["status"] == "incomparable"


def _edit_reward(config, change):
    run_dir = Path(config["result"]["run_dir"])
    path = next(run_dir.glob("codex/*/trials/*/reward.json"))
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize(
    "change",
    [
        "budget",
        "early_stop",
        "model",
        "judge",
        "missing_judge",
        "case_contents",
        "missing_trial",
        "extra_trial",
        "duplicate_trial",
        "unknown_case",
        "nan",
        "not_applicable",
        "fallback",
        "method",
        "missing_identity",
        "partial",
        "metric_set",
        "multi_step",
    ],
)
def test_refuses_incomparable_evidence(tmp_path, change):
    matrix = _matrix(tmp_path)
    config = matrix[-1]
    result = config["result"]
    run_dir = Path(result["run_dir"])
    if change == "budget":
        result["run_config"]["harbor"]["n_attempts"] = 2
    elif change == "early_stop":
        result["run_config"]["harbor"]["stop_on_pass"] = True
    elif change == "model":
        result["agents"]["codex"]["model"] = "different-model"
    elif change == "judge":
        result["run_config"]["judge"]["model"] = "different-judge"
    elif change == "missing_judge":
        _edit_reward(config, lambda reward: reward.pop("details"))
    elif change == "case_contents":
        path = run_dir / "dataset_snapshot.json"
        path.write_text(json.dumps({"dataset": [{"id": "case-1", "question": "Changed task"}]}), encoding="utf-8")
    elif change == "missing_trial":
        next(run_dir.glob("codex/*/trials/*/reward.json")).unlink()
    elif change in {"extra_trial", "duplicate_trial", "multi_step"}:
        source = next(run_dir.glob("codex/*/trials/*/reward.json"))
        dest = source.parent.parent / "another-trial" / "reward.json"
        dest.parent.mkdir()
        data = json.loads(source.read_text())
        if change == "extra_trial":
            data["trial_id"] = "another-trial"
        dest.write_text(json.dumps(data), encoding="utf-8")
    elif change == "unknown_case":
        _edit_reward(config, lambda reward: reward.update(entry_id="unknown"))
    elif change == "nan":
        _edit_reward(config, lambda reward: reward.update(goal_accuracy=float("nan")))
    elif change == "not_applicable":
        _edit_reward(config, lambda reward: reward.update(goal_accuracy=None))
        _edit_reward(config, lambda reward: reward["details"]["goal_accuracy"].update(status="not_applicable"))
    elif change == "fallback":
        _edit_reward(
            config, lambda reward: reward["details"]["goal_accuracy"].update(judge_model_used="fallback-judge")
        )
    elif change == "method":
        _edit_reward(config, lambda reward: reward["details"]["goal_accuracy"].update(method="ragas"))
    elif change == "missing_identity":
        _edit_reward(config, lambda reward: reward.pop("trial_id"))
    elif change == "partial":
        result["run_config"]["case_selection"] = {"skipped": 1}
    elif change == "metric_set":
        _edit_reward(config, lambda reward: reward.update(metric_set="different-contract"))
    report = compare_configurations(matrix, metric="goal_accuracy", n_attempts=1)
    assert report["status"] == "incomparable"
    assert report["agents"]["codex"]["reasons"]
    assert "interaction" not in report["agents"]["codex"]


def test_requires_identical_case_coverage_not_equal_total_counts(tmp_path):
    matrix = _matrix(tmp_path)
    _edit_reward(matrix[-1], lambda reward: reward.update(entry_id="different-case"))
    report = compare_configurations(matrix, metric="goal_accuracy", n_attempts=1)
    assert report["status"] == "incomparable"


def test_equal_attempt_budgets_average_attempts_within_each_case(tmp_path):
    matrix = []
    for index, values in enumerate(
        [
            {"one": [0.2, 0.8], "two": [0.0, 0.0]},
            {"one": [0.5, 0.5], "two": [0.5, 0.5]},
            {"one": [0.5, 0.5], "two": [0.5, 0.5]},
            {"one": [1.0, 0.0], "two": [1.0, 1.0]},
        ]
    ):
        condition = "without-skill" if index == 0 else "with-skill"
        result = _evidence(tmp_path / str(index), values, n_attempts=2, condition=condition)
        matrix.append({"id": str(index), "condition": condition, "result": result})
    score = compare_configurations(matrix, metric="goal_accuracy", n_attempts=2)["agents"]["codex"]
    assert score["baseline"] == pytest.approx(0.25)
    assert score["combined"] == pytest.approx(0.75)
    assert score["interaction"] == pytest.approx(0.0)


def test_custom_only_compares_shared_grader_overall(tmp_path):
    matrix = _matrix(tmp_path, custom_only=True)
    report = compare_configurations(matrix, metric="overall", n_attempts=1)
    assert report["status"] == "complete"
    assert report["agents"]["codex"]["interaction"] == pytest.approx(-0.3)


def test_standard_overall_is_not_a_comparable_task_outcome(tmp_path):
    report = compare_configurations(_matrix(tmp_path), metric="overall", n_attempts=1)
    assert report["status"] == "incomparable"


@pytest.mark.parametrize(
    "failure",
    [
        {"error": "launch failed"},
        {"execution_status": "failed"},
        {},
        {"execution_status": "succeeded", "execution_errors": ["unscored trial"]},
    ],
)
def test_failed_runs_never_produce_zero_or_numeric_interaction(tmp_path, failure):
    matrix = _matrix(tmp_path)
    matrix[-1]["result"] = failure
    report = compare_configurations(matrix, metric="goal_accuracy", n_attempts=1)
    assert report["status"] != "complete"
    assert all("interaction" not in agent for agent in report["agents"].values())


@pytest.mark.parametrize("failure", ["missing_result", "empty_error", "empty_error_with_result"])
def test_missing_results_and_empty_errors_are_incomplete(tmp_path, failure):
    matrix = _matrix(tmp_path)
    if failure != "empty_error_with_result":
        del matrix[-1]["result"]
    if failure != "missing_result":
        matrix[-1]["error"] = ""
    report = compare_configurations(matrix, metric="goal_accuracy", n_attempts=1)
    assert report["status"] == "incomplete"
    assert report["reasons"] and all(report["reasons"])
    assert report["agents"] == {}
    assert "interaction" not in report


def test_failed_configuration_reasons_identify_each_arm(tmp_path):
    matrix = _matrix(tmp_path)
    for config in matrix[1:]:
        config["error"] = "same setup failure"
    report = compare_configurations(matrix, metric="goal_accuracy", n_attempts=1)
    assert report["reasons"] == [f"config-{index}: same setup failure" for index in (1, 2, 3)]


def test_terminal_deduplicates_failure_reasons(capsys):
    from skillevaluator.tier3.interaction_commands import _print_report

    _print_report(
        {
            "status": "incomplete",
            "metric": "goal_accuracy",
            "skills": [],
            "run_dir": "/results/experiment",
            "reasons": ["baseline: setup failed"] * 2,
            "agents": {"codex": {"status": "incomparable", "reasons": ["missing evidence"] * 2}},
        }
    )
    output = capsys.readouterr().out
    assert output.count("baseline: setup failed") == 1
    assert output.count("missing evidence") == 1


def _fake_engine(monkeypatch, scores=(0.50, 0.75, 0.70, 0.65), *, on_run=None):
    calls = []

    def evaluate(self, options, **kwargs):
        index = len(calls)
        calls.append(copy.deepcopy(options))
        assert (options.skill_path / "evals" / "evals.json").is_file()
        staged_entry = on_run(index, options) if on_run else None
        condition = "without-skill" if options.skip_with_skill else "with-skill"
        result = _evidence(
            options.results_dir / "run",
            {"case-1": [scores[index]] * options.n_attempts},
            condition=condition,
            n_attempts=options.n_attempts,
        )
        if staged_entry:
            (Path(result["run_dir"]) / "dataset_snapshot.json").write_text(
                json.dumps({"dataset": [staged_entry]}), encoding="utf-8"
            )
        return result

    monkeypatch.setattr(EvaluationService, "evaluate", evaluate)
    return calls


def test_orchestrates_frozen_inputs_and_only_selected_skills(monkeypatch, tmp_path):
    source = _source(tmp_path, "harbor:\n  n_attempts: 7\n  stop_on_pass: true\n")
    skills = [_skill(tmp_path, name) for name in ("a-skill", "b-skill")]
    original_evals = (source / "evals" / "evals.json").read_bytes()
    original_skill = (skills[1] / "SKILL.md").read_text()
    source_copies = []

    def inspect(index, options):
        source_copies.append((options.skill_path / "evals" / "evals.json").read_bytes())
        if index == 0:
            (skills[1] / "SKILL.md").write_text("changed after freezing", encoding="utf-8")
        if index == 2:
            assert (options.skill_path / "SKILL.md").read_text() == original_skill
        assert not (options.skill_path / "evals" / "results").exists()

    calls = _fake_engine(monkeypatch, on_run=inspect)
    report = run_interaction(skills, eval_source=source, options=EvaluationOptions(source, n_attempts=2))
    assert report["status"] == "complete"
    assert len(calls) == 4
    assert calls[0].skip_with_skill and not calls[0].skip_baseline
    assert all(call.skip_baseline and not call.skip_with_skill for call in calls[1:])
    assert [call.skill_path.name for call in calls] == ["a-skill", "a-skill", "b-skill", "a-skill"]
    assert [len(call.include_skills) for call in calls] == [0, 0, 0, 1]
    assert calls[-1].include_skills[0].name == "b-skill"
    assert [call.skill_workspace_mode for call in calls] == ["isolated", "isolated", "isolated", "group"]
    assert all(call.n_attempts == 2 and call.stop_on_pass is False for call in calls)
    assert source_copies == [original_evals] * 4
    assert len({str(call.results_dir) for call in calls}) == 4
    assert (source / "evals" / "evals.json").read_bytes() == original_evals
    saved = json.loads((Path(report["run_dir"]) / "interaction.json").read_text())
    assert saved == report
    assert len(saved["skills"]) == 2
    assert saved["source"]["fingerprint"]
    assert (Path(report["run_dir"]) / "source" / "evals" / "evals.json").read_bytes() == original_evals


def test_group_runs_n_plus_two_configurations(monkeypatch, tmp_path):
    source = _source(tmp_path)
    skills = [_skill(tmp_path, name) for name in ("a", "b", "c")]
    calls = _fake_engine(monkeypatch, (0.2, 0.3, 0.35, 0.4, 0.8))
    report = run_interaction(skills, eval_source=source, options=EvaluationOptions(source))
    assert len(calls) == 5
    assert len(calls[-1].include_skills) == 2
    assert report["agents"]["codex"]["interaction"] == pytest.approx(0.15)


@pytest.mark.parametrize("use_cli_root", [False, True])
def test_results_root_honors_environment_and_explicit_precedence(monkeypatch, tmp_path, use_cli_root):
    source = _source(tmp_path)
    skills = [_skill(tmp_path, name) for name in ("a", "b")]
    env_root = tmp_path / "env-results"
    cli_root = tmp_path / "cli-results" if use_cli_root else None
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(env_root))
    _fake_engine(monkeypatch)
    report = run_interaction(skills, eval_source=source, options=EvaluationOptions(source, results_dir=cli_root))
    assert Path(report["run_dir"]).parent == (cli_root or env_root).resolve()


def test_results_root_resolves_symlinked_parent(monkeypatch, tmp_path):
    source = _source(tmp_path)
    skills = [_skill(tmp_path, name) for name in ("a", "b")]
    real_root = tmp_path / "real-results"
    real_root.mkdir()
    linked_root = tmp_path / "linked-results"
    linked_root.symlink_to(real_root, target_is_directory=True)
    _fake_engine(monkeypatch)
    report = run_interaction(
        skills, eval_source=source, options=EvaluationOptions(source, results_dir=linked_root / "runs")
    )
    assert Path(report["run_dir"]).parent == real_root / "runs"


def test_results_root_symlink_into_selected_skill_is_rejected(monkeypatch, tmp_path):
    source = _source(tmp_path)
    skills = [_skill(tmp_path, name) for name in ("a", "b")]
    linked_root = tmp_path / "linked-results"
    linked_root.symlink_to(skills[1], target_is_directory=True)
    calls = _fake_engine(monkeypatch)
    with pytest.raises(ValueError, match="Save interaction results outside"):
        run_interaction(skills, eval_source=source, options=EvaluationOptions(source, results_dir=linked_root / "runs"))
    assert calls == []


def test_existing_harbor_staging_installs_the_configuration_and_keeps_tasks_fixed(monkeypatch, tmp_path):
    from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks

    source = _source(tmp_path)
    skills = [_skill(tmp_path, name) for name in ("a", "b")]
    staged = []

    def inspect(index, options):
        # Bound the fixture's repository context even when TMPDIR is inside a checkout.
        subprocess.run(["git", "init", "-q", str(options.skill_path.parent)], check=True)
        task = generate_harbor_tasks(
            options.skill_path,
            tmp_path / f"staged-{index}",
            with_skill=not options.skip_with_skill,
            workspace_skill_paths=list(options.include_skills),
            workspace_mode=options.skill_workspace_mode,
        )[0]
        installed = {path.parent.name for path in (task / "environment" / "skills").glob("*/SKILL.md")}
        entry = json.loads((task / "tests" / "entry.json").read_text())
        staged.append((installed, entry, (task / "instruction.md").read_text()))
        return entry

    _fake_engine(monkeypatch, on_run=inspect)
    report = run_interaction(skills, eval_source=source, options=EvaluationOptions(source))
    assert report["status"] == "complete", report.get("reasons")
    assert [row[0] for row in staged] == [set(), {"a"}, {"b"}, {"a", "b"}]
    assert len({row[2] for row in staged}) == 1
    assert len({row[1]["ground_truth"] for row in staged}) == 1
    assert [row[1]["has_skill"] for row in staged] == [False, True, True, True]


def test_native_harbor_source_reuses_the_reviewed_tasks(monkeypatch, tmp_path):
    from skillevaluator.tier3.commands import init_harbor_task
    from skillevaluator.tier3.harbor.adapter import stage_native_harbor_tasks

    source = _skill(tmp_path, "reviewed-native")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    init_harbor_task(source, force=False, case_id="case-1", mode="default", language="python", with_config=True)
    skills = [_skill(tmp_path, name) for name in ("a", "b")]
    staged = []

    def inspect(index, options):
        subprocess.run(["git", "init", "-q", str(options.skill_path.parent)], check=True)
        task = stage_native_harbor_tasks(
            options.skill_path,
            tmp_path / f"native-staged-{index}",
            with_skill=not options.skip_with_skill,
            workspace_skill_paths=list(options.include_skills),
            workspace_mode=options.skill_workspace_mode,
        )[0]
        staged.append((task / "instruction.md").read_bytes())
        return json.loads((task / "tests" / "entry.json").read_text())

    _fake_engine(monkeypatch, on_run=inspect)
    report = run_interaction(skills, eval_source=source, options=EvaluationOptions(source))
    assert report["status"] == "complete", report.get("reasons")
    assert len(staged) == 4 and all(instruction == staged[0] for instruction in staged)


def test_interrupted_experiment_retains_explicit_incomplete_report(monkeypatch, tmp_path):
    source = _source(tmp_path)
    skills = [_skill(tmp_path, name) for name in ("a", "b")]

    def interrupt(self, options, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(EvaluationService, "evaluate", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_interaction(skills, eval_source=source, options=EvaluationOptions(source))
    saved = json.loads(next((source.parent / "interaction-results").glob("*/interaction.json")).read_text())
    assert saved["status"] == "incomplete"
    assert "interrupted" in saved["reasons"][0]
    assert "interaction" not in saved


@pytest.mark.parametrize("location", ["source", "repo", "second_skill", "results"])
def test_checks_original_provenance_key_location_before_snapshot(monkeypatch, tmp_path, location):
    source = _source(tmp_path)
    skills = [_skill(tmp_path, name) for name in ("a", "b")]
    results = tmp_path / "results"
    roots = {"source": source, "repo": source.parent, "second_skill": skills[1], "results": results}
    monkeypatch.setenv("SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE", str(roots[location] / "marker.key"))

    def unexpected_snapshot(*args, **kwargs):
        pytest.fail("Input validation must finish before snapshotting")

    monkeypatch.setattr("skillevaluator.tier3.harbor.adapter.private_evaluator_skill_snapshot", unexpected_snapshot)
    with pytest.raises(ValueError, match="Output provenance key must be outside"):
        run_interaction(skills, eval_source=source, options=EvaluationOptions(source, results_dir=results))
    assert not results.exists()


@pytest.mark.parametrize(
    "invalid",
    [
        "one_skill",
        "duplicate",
        "same_name",
        "overlap",
        "inside_output",
        "source_selection",
        "partial",
        "standard_overall",
        "missing_source",
        "missing_evals_dir",
        "malformed_environment_kwarg",
    ],
)
@pytest.mark.parametrize("via_cli", [False, True])
def test_rejects_invalid_experiments_before_execution(monkeypatch, tmp_path, invalid, via_cli):
    source = _source(tmp_path)
    skills = [_skill(tmp_path, name) for name in ("a", "b")]
    options = EvaluationOptions(source)
    metric = "goal_accuracy"
    if invalid == "one_skill":
        skills.pop()
    elif invalid == "duplicate":
        skills[1] = skills[0]
    elif invalid == "same_name":
        (skills[1] / "SKILL.md").write_text("---\nname: a\n---\n", encoding="utf-8")
    elif invalid == "overlap":
        skills[1] = _skill(skills[0], "nested")
    elif invalid == "inside_output":
        options.results_dir = skills[0] / "evals" / "results"
    elif invalid == "source_selection":
        (source / "evals" / "config.yml").write_text(
            "schema_version: 1\nskill_workspace:\n  mode: group\n  include: [a]\n", encoding="utf-8"
        )
    elif invalid == "partial":
        (source / "evals" / "config.yml").write_text("schema_version: 1\nharbor:\n  max_cases: 1\n", encoding="utf-8")
    elif invalid == "standard_overall":
        metric = "overall"
    elif invalid == "missing_source":
        (source / "evals" / "evals.json").unlink()
    elif invalid == "missing_evals_dir":
        (source / "evals").rename(source / "unused")
    elif invalid == "malformed_environment_kwarg":
        options.environment_kwarg = ("malformed",)
    calls = _fake_engine(monkeypatch)
    if via_cli:
        args = ["tier3", "interact", *(str(path) for path in skills), "--eval-source", str(source), "--metric", metric]
        if options.results_dir:
            args.extend(["--results-dir", str(options.results_dir)])
        for value in options.environment_kwarg:
            args.extend(["--ek", value])
        result = CliRunner().invoke(cli, args)
        assert result.exit_code == 2, result.output
        assert "Usage:" in result.stderr
    else:
        with pytest.raises((OSError, ValueError)):
            run_interaction(skills, eval_source=source, options=options, metric=metric)
    assert calls == []
    assert not (source.parent / "interaction-results").exists()


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        (RuntimeError("a configuration failed"), "a configuration failed"),
        (TimeoutError(), "TimeoutError"),
    ],
)
def test_preserves_other_configuration_evidence_after_failure(monkeypatch, tmp_path, failure, expected_error):
    source = _source(tmp_path)
    skills = [_skill(tmp_path, name) for name in ("a", "b")]
    counter = []

    def failing(self, options, **kwargs):
        counter.append(options)
        if len(counter) == 2:
            raise failure
        return _evidence(
            options.results_dir / "run",
            {"case-1": [0.5]},
            condition="without-skill" if options.skip_with_skill else "with-skill",
        )

    monkeypatch.setattr(EvaluationService, "evaluate", failing)
    report = run_interaction(skills, eval_source=source, options=EvaluationOptions(source))
    assert report["status"] == "incomplete"
    assert len(counter) == 4
    assert report["configurations"][1]["error"] == expected_error
    assert all("result" in report["configurations"][index] for index in (0, 2, 3))
    assert "interaction" not in report
    assert json.loads((Path(report["run_dir"]) / "interaction.json").read_text()) == report


@pytest.mark.parametrize("as_json", [False, True])
def test_cli_reports_signed_interaction_and_keeps_json_clean(monkeypatch, tmp_path, as_json):
    source = _source(tmp_path)
    skills = [_skill(tmp_path, name) for name in ("a", "b")]
    calls = _fake_engine(monkeypatch)
    args = [
        "tier3",
        "interact",
        *(str(path) for path in skills),
        "--eval-source",
        str(source),
        "--n-attempts",
        "2",
        "--agents",
        "codex",
        "--model",
        "test-model",
    ]
    if as_json:
        args.append("--json")
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert len(calls) == 4 and calls[0].agents == "codex" and calls[0].model == "test-model"
    if as_json:
        report = json.loads(result.stdout)
        assert report["agents"]["codex"]["classification"] == "negative"
        assert "running baseline" in result.stderr
    else:
        assert "Interaction: -0.3000 (negative)" in result.stdout
        assert "combined lift: +0.1500" in result.stdout


def test_cli_sends_each_arms_progress_to_stderr(monkeypatch, tmp_path):
    from skillevaluator.tier3.harbor.progress import ProgressEvent, Tier3RunPlan

    source = _source(tmp_path)
    skills = [_skill(tmp_path, name) for name in ("a", "b")]

    def evaluate(self, options, *, progress_reporter=None):
        if progress_reporter is not None:
            progress_reporter.start(
                Tier3RunPlan(skill_name=options.skill_path.name, environment=options.env_mode, agents=("codex",))
            )
            progress_reporter.emit(ProgressEvent(stage="fixture-trial", state="running"))
            progress_reporter.emit(ProgressEvent(stage="fixture-trial", state="complete"))
            progress_reporter.close()
        return _evidence(
            options.results_dir / "run",
            {"case-1": [0.5]},
            condition="without-skill" if options.skip_with_skill else "with-skill",
        )

    monkeypatch.setattr(EvaluationService, "evaluate", evaluate)
    result = CliRunner().invoke(
        cli, ["tier3", "interact", *(str(path) for path in skills), "--eval-source", str(source), "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "complete"
    assert result.stderr.count("fixture-trial: running") == 4


@pytest.mark.parametrize("failure", [None, TimeoutError(), ValueError("evaluation failed")])
def test_cli_incomplete_is_nonzero_and_machine_readable(monkeypatch, tmp_path, failure):
    source = _source(tmp_path)
    skills = [_skill(tmp_path, name) for name in ("a", "b")]

    def failing(self, options, **kwargs):
        if failure is not None:
            raise failure
        return {"error": ["launch failed"]}

    monkeypatch.setattr(EvaluationService, "evaluate", failing)
    result = CliRunner().invoke(
        cli, ["tier3", "interact", *(str(path) for path in skills), "--eval-source", str(source), "--json"]
    )
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["status"] == "incomplete"
    assert "interaction" not in report
    assert all(config["error"] for config in report["configurations"])


def test_cli_output_io_failure_uses_exit_one(monkeypatch, tmp_path):
    source = _source(tmp_path)
    skills = [_skill(tmp_path, name) for name in ("a", "b")]

    def fail_to_write(*args, **kwargs):
        raise OSError("Cannot write interaction results")

    monkeypatch.setattr("skillevaluator.tier3.interactions.run_interaction", fail_to_write)
    result = CliRunner().invoke(
        cli, ["tier3", "interact", *(str(path) for path in skills), "--eval-source", str(source)]
    )
    assert result.exit_code == 1
    assert "Cannot write interaction results" in result.stderr


def test_cli_requires_two_skills_and_never_generates_a_source(monkeypatch, tmp_path):
    skill = _skill(tmp_path, "a")
    calls = _fake_engine(monkeypatch)
    result = CliRunner().invoke(cli, ["tier3", "interact", str(skill), "--eval-source", str(skill)])
    assert result.exit_code == 2
    assert "at least two" in result.output
    assert calls == []
