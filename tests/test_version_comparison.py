# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from rich.console import Console

from skillevaluator.cli import cli
from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory
from skillevaluator.provider_config import ProviderConfig
from skillevaluator.reporting.benchmark import BenchmarkReporter
from skillevaluator.reporting.cli import CLIReporter
from skillevaluator.reporting.html import HTMLReporter
from skillevaluator.reporting.markdown import MarkdownReporter
from skillevaluator.tier3.baseline_store import condition_eligibility
from skillevaluator.tier3.harbor import runner, runtime_preflight
from skillevaluator.tier3.result_display import render_evaluation_result
from skillevaluator.tier3.version_comparison import compare_versions, select_prior_version, version_comparison_lines
from test_baseline_store import _job, _write


@pytest.fixture
def evaluation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    skill = tmp_path / "demo"
    _write(skill / "evals/evals.json", [])
    provider = ProviderConfig(
        provider="openai",
        model="test-model",
        api_key="test-key",
        base_url="https://provider.example/v1",
        litellm_model="openai/test-model",
    )
    monkeypatch.setattr(runner, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(runner, "load_evals_config", lambda _path: ({"harbor": {"task_source": "evals_json"}}, None))
    monkeypatch.setattr(runner, "find_evals_file", lambda _path: skill / "evals/evals.json")
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setattr(
        runtime_preflight,
        "probe_model",
        lambda selected: runtime_preflight.ModelProbeResult(True, selected.provider, selected.model, "available"),
    )
    monkeypatch.setattr(runner, "render_agent_eval_html_report", lambda *_args, **_kwargs: tmp_path / "report.html")

    def emit(_skill, target, **_kwargs):
        task = target / "case-001"
        task.mkdir(parents=True)
        (task / "task.toml").write_text(
            'schema_version = "1.3"\n[task]\nname = "case-001"\n[metadata]\nentry_id = "case-001"\n'
        )
        return [task]

    monkeypatch.setattr(runner, "generate_harbor_tasks", emit)
    active = {}
    launches = []

    def launch(**kwargs):
        launches.append(kwargs)
        for condition, suffix, score in (("with_skill", "with", active["score"]), ("baseline", "without", 0.5)):
            if kwargs[condition] is not None:
                _job(kwargs["jobs_dir"] / f"demo-opencode-{suffix}", score, grading_mode=active["grading"])
        return []

    monkeypatch.setattr(runner, "_run_agent_pair", launch)

    def evaluate(version="a", score=0.8, **options):
        active.update(score=score, grading=options.get("grading_mode", "default"))
        (skill / "SKILL.md").write_text(f"---\nname: demo\ndescription: Arithmetic note\n---\nVersion {version}\n")
        return runner.run_harbor_eval(
            skill,
            ["opencode"],
            output_dir=tmp_path / "results/demo",
            env_mode="docker",
            agent_runtime_preflight=False,
            agent_models={"opencode": "openai/test-model"},
            evaluated_source={"repository": "example/skill", "commit": version * 40},
            **options,
        )

    evaluate.skill = skill
    evaluate.launches = launches
    return evaluate


@pytest.mark.parametrize("grading_mode", ["default", "custom_only"])
def test_live_and_readback_share_regression_math(evaluation, grading_mode):
    prior = evaluation(grading_mode=grading_mode)
    assert "error" not in prior, prior
    current = evaluation("b", 0.2, grading_mode=grading_mode, compare_to=prior["run_id"], reuse_baseline="latest")
    assert "error" not in current, current
    comparison = current["prior_version_comparison"]
    assert comparison["status"] == "available", comparison
    assert comparison["regression"] is True
    assert comparison["agents"]["opencode"]["delta"] == -0.6
    assert comparison["agents"]["opencode"]["current_overall"] == 0.2
    assert comparison["agents"]["opencode"]["prior_overall"] == 0.8
    if grading_mode == "default":
        assert current["agents"]["opencode"]["lift"]["overall"]["delta"] == -0.3
        assert comparison["agents"]["opencode"]["metrics"]["accuracy"]["delta"] == -0.6
    assert current["baseline"]["mode"] == "reused"
    assert evaluation.launches[-1]["baseline"] is None
    readback = compare_versions(Path(current["run_dir"]), Path(prior["run_dir"]))
    assert readback == {key: value for key, value in comparison.items() if key != "rejected_candidates"}


def test_explicit_incompatible_refuses_before_execution(evaluation):
    prior = evaluation()
    current = evaluation("b", 0.2, compare_to=prior["run_id"], n_attempts=2)
    assert "attempt policy changed" in str(current["error"])
    assert len(evaluation.launches) == 1


def test_latest_requires_different_skill_and_can_be_unavailable(evaluation):
    prior = evaluation()
    same = evaluation(compare_to="latest")
    unavailable = same["prior_version_comparison"]
    assert unavailable["status"] == "unavailable"
    assert "unchanged" in str(unavailable["rejected_candidates"])
    assert "regression" not in unavailable
    different = evaluation("b", 0.2, compare_to="latest")
    assert different["prior_version_comparison"]["status"] == "available"
    assert different["prior_version_comparison"]["source_run_id"] in {same["run_id"], prior["run_id"]}


def test_identical_explicit_source_is_flagged(evaluation):
    prior = evaluation()
    same = evaluation(compare_to=prior["run_id"])
    comparison = same["prior_version_comparison"]
    assert comparison["status"] == "available"
    assert comparison["identical_skill"] is True
    assert comparison["regression"] is False
    assert "not a version change" in " ".join(version_comparison_lines(comparison))


def test_revision_fallback_is_used_when_publication_digest_is_missing(evaluation):
    prior = evaluation("a", 0.8)
    current = evaluation("b", 0.2)
    for run in (prior, current):
        result_path = Path(run["run_dir"]) / "result.json"
        result = json.loads(result_path.read_text())
        result["publication_target"]["skill_digest"] = None
        result["comparison_identity"]["skill"]["publication_target"]["skill_digest"] = None
        _write(result_path, result)

    comparison = compare_versions(Path(current["run_dir"]), Path(prior["run_dir"]))

    assert comparison["status"] == "available"
    assert comparison["identical_skill"] is False
    assert f"rev {'a' * 40}" in version_comparison_lines(comparison)[0]


@pytest.mark.parametrize("arm", ["prior", "current"])
@pytest.mark.parametrize("damage", ["missing", "corrupt", "unscored", "old", "partial", "failed"])
def test_missing_or_unscored_is_not_a_regression(evaluation, damage, arm):
    prior, current = evaluation(), evaluation("b", 0.2)
    damaged = prior if arm == "prior" else current
    reward = next((Path(damaged["run_dir"]) / "opencode/with-skill/trials").glob("*/reward.json"))
    row = json.loads(reward.read_text())
    if damage == "missing":
        reward.unlink()
    elif damage == "corrupt":
        reward.write_text("{")
    elif damage == "failed":
        damaged["agents"]["opencode"]["conditions"]["with_skill"]["execution_status"] = "failed"
        _write(Path(damaged["run_dir"]) / "result.json", damaged)
    else:
        if damage == "unscored":
            row["evaluation_status"] = "failed"
        elif damage == "old":
            row.pop("baseline_evidence")
        else:
            row["entry_id"] = "another-case"
        _write(reward, row)
    comparison = compare_versions(Path(current["run_dir"]), Path(prior["run_dir"]))
    assert comparison["status"] == "unavailable"
    assert comparison["reasons"]
    assert comparison["agents"] == {}
    assert "regression" not in comparison


def test_reused_no_skill_run_can_supply_prior_with_skill(evaluation):
    original = evaluation()
    prior = evaluation("b", 0.7, reuse_baseline=original["run_id"])
    current = evaluation("c", 0.2, compare_to=prior["run_id"])
    assert current["prior_version_comparison"]["status"] == "available"


def test_posthoc_is_read_only_and_prints_table_or_json(evaluation):
    prior, current = evaluation(), evaluation("b", 0.2)
    root = Path(current["run_dir"]).parent

    def snapshot():
        return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}

    before = snapshot()
    args = [
        "tier3",
        "compare-versions",
        str(evaluation.skill),
        "--current",
        current["run_id"],
        "--prior",
        prior["run_id"],
        "--results-dir",
        str(root.parent),
    ]
    result = CliRunner().invoke(cli, [*args, "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["regression"] is True
    table = CliRunner().invoke(cli, args)
    assert table.exit_code == 0, table.output
    assert "Regression" in table.output and "-0.6000" in table.output
    invalid_args = list(args)
    invalid_args[6] = "../outside"
    invalid = CliRunner().invoke(cli, [*invalid_args, "--json"])
    assert invalid.exit_code != 0
    assert snapshot() == before
    assert len(evaluation.launches) == 2


def test_all_report_formats_keep_version_delta_separate(evaluation):
    prior = evaluation()
    current = evaluation("b", 0.2, compare_to=prior["run_id"])
    result = agent_eval_result_from_directory(
        evaluation.skill, Path(current["run_dir"]), engine_result=current, use_llm_judge=False
    )
    assert result is not None
    for reporter in (
        BenchmarkReporter(skill_name="demo", report_scope="tier3"),
        MarkdownReporter(),
        HTMLReporter(),
        CLIReporter(),
    ):
        rendered = reporter.render_all([result])
        assert "Versus prior version" in rendered
        assert "regression" in rendered
    stream = io.StringIO()
    render_evaluation_result(current, console=Console(file=stream, width=200), verbose=False)
    assert "Versus prior version" in stream.getvalue() and "regression" in stream.getvalue()


@pytest.mark.parametrize("changed", ["model", "environment", "judge", "cases", "agents"])
def test_version_compatibility_uses_shared_identity(evaluation, changed):
    prior = evaluation()
    identity = copy.deepcopy(prior["comparison_identity"])
    if changed == "model":
        identity["config"]["agents"]["opencode"]["model"] = "another-model"
    elif changed == "environment":
        identity["config"]["harbor"]["environment"] = "local"
    elif changed == "judge":
        identity["config"]["judge"]["model"] = "another-judge"
    elif changed == "cases":
        identity["expected_case_ids"] = ["another-case"]
    else:
        identity["config"]["agents"] = {}
    with pytest.raises(ValueError, match="Cannot compare to"):
        select_prior_version(
            Path(prior["run_dir"]).parent,
            prior["run_id"],
            identity,
            current_run_dir=Path(prior["run_dir"]).parent / "new",
        )


def test_latest_rejects_incompatible_candidates(evaluation):
    prior = evaluation()
    identity = dict(prior["comparison_identity"])
    identity["evaluator_version"] = "another-version"
    selection = select_prior_version(
        Path(prior["run_dir"]).parent, "latest", identity, current_run_dir=Path(prior["run_dir"]).parent / "new"
    )
    assert selection["status"] == "unavailable"
    assert "evaluator version changed" in str(selection["rejected_candidates"])


def test_latest_never_selects_a_run_newer_than_the_current_run(evaluation):
    current = evaluation("a", 0.8)
    evaluation("b", 0.2)

    selection = select_prior_version(
        Path(current["run_dir"]).parent,
        "latest",
        current["comparison_identity"],
        current_run_dir=Path(current["run_dir"]),
    )

    assert selection["status"] == "unavailable"
    assert selection["reasons"] == ["No compatible completed run of a different skill revision"]


def test_with_skill_eligibility_reasons_name_the_with_skill_arm(evaluation):
    run = evaluation()
    summary_path = Path(run["run_dir"]) / "opencode/with-skill/summary.json"
    summary = json.loads(summary_path.read_text())
    summary["execution_status"] = "failed"
    _write(summary_path, summary)

    reasons, _ = condition_eligibility(Path(run["run_dir"]), ["opencode"], run["comparison_identity"], "with_skill")

    assert "opencode: with-skill summary is incomplete or failed" in reasons
    assert not any("baseline" in reason for reason in reasons)


def test_malformed_display_scores_are_unavailable():
    assert "unavailable" in version_comparison_lines({"status": "available", "agents": {"agent": {}}})[0]
