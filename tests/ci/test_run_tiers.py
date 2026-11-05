# SPDX-License-Identifier: Apache-2.0
"""Consumer CI execution must retain honest, independent tier outcomes."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from ci import run_tiers


@pytest.fixture
def consumer(tmp_path):
    repo = tmp_path / "consumer"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    for name in ("alpha", "beta"):
        skill = repo / "skills" / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Use to test {name}.\n---\nInstructions.\n")
    git = ["git", "-C", str(repo), "-c", "user.name=CI tests", "-c", "user.email=ci@example.com"]
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-m", "fixtures"], check=True, capture_output=True)
    return repo


def env_for(consumer, **updates):
    sha = subprocess.check_output(["git", "-C", str(consumer), "rev-parse", "HEAD"], text=True).strip()
    return {
        "PATH": os.environ["PATH"],
        "SKILL_EVAL_PATHS": "skills",
        "REPORT_COMMIT": sha,
        "REPORT_REPOSITORY": "example/consumer",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_ACTIONS": "true",
        "SKILL_EVAL_PROVIDER": "nv_build",
        **updates,
    }


def test_missing_credentials_retains_incomplete_required_tier(consumer, tmp_path):
    output = tmp_path / "reports"
    code = run_tiers.evaluate(2, consumer, output, env_for(consumer))
    result = json.loads((output / "tier2" / "ci-tier.json").read_text())
    assert code == 1
    assert result["tiers"][0]["status"] == "incomplete"
    assert result["tiers"][0]["required"] is True
    assert "credential" in result["tiers"][0]["reason"].lower()


def test_missing_tier3_dataset_retains_advisory_failure_without_paid_execution(consumer, tmp_path, monkeypatch):
    def no_execution(*args, **kwargs):
        raise AssertionError("An invalid dataset must fail before running an agent or provider")

    monkeypatch.setattr(run_tiers, "run_command", no_execution)
    output = tmp_path / "reports"
    code = run_tiers.evaluate(3, consumer, output, env_for(consumer, NVIDIA_API_KEY="test-key-value-123456"))
    result = json.loads((output / "tier3" / "ci-tier.json").read_text())
    assert code == 0
    tier = result["tiers"][0]
    assert tier["status"] == "incomplete"
    assert tier["required"] is False
    assert len(tier["skills"]) == 2
    assert all(row["status"] == "incomplete" for row in tier["skills"])


@pytest.mark.parametrize("mode", ["gke", "cloud", "docker;echo secret"])
def test_unsupported_environment_fails_before_execution(consumer, tmp_path, mode):
    with pytest.raises(run_tiers.ConfigurationError, match="docker or local"):
        run_tiers.evaluate(3, consumer, tmp_path / "reports", env_for(consumer, SKILL_EVAL_ENV_MODE=mode))


def test_tier3_command_demands_all_cases_both_arms_and_explicit_mode(consumer, tmp_path):
    args = run_tiers.skill_command(
        3, consumer / "skills/alpha", tmp_path / "raw", env_for(consumer, SKILL_EVAL_ENV_MODE="local")
    )
    assert "native-tier3" in args
    assert "validate" not in args
    assert args[args.index("--env-mode") + 1] == "local"
    assert not set(args) & {"--skip-baseline", "--skip-with-skill", "--max-cases"}


def test_native_tier3_options_preserve_complete_paired_coverage(consumer, tmp_path):
    options = run_tiers.tier3_options(consumer / "skills/alpha", tmp_path, env_for(consumer))
    assert options.skip_baseline is False
    assert options.skip_with_skill is False
    assert options.max_cases is None
    assert options.stop_on_pass is False
    assert options.n_attempts == 1
    assert options.evaluated_source["commit"] == env_for(consumer)["REPORT_COMMIT"]


def test_ci_chat_model_uses_native_agent_provider_resolution(consumer, tmp_path):
    # OpenCode needs its adapter namespace as well as the model's publisher.
    # Passing the chat model as a native CLI override bypasses that resolution.
    options = run_tiers.tier3_options(
        consumer / "skills/alpha", tmp_path, env_for(consumer, SKILL_EVAL_MODEL="nvidia/test-model")
    )
    assert options.model is None


def test_native_tier3_engine_failure_still_emits_incomplete_evidence(consumer, tmp_path, monkeypatch):
    from skillevaluator.evaluation.service import EvaluationService

    monkeypatch.setattr(run_tiers, "dataset_preflight", lambda _skill: {})
    monkeypatch.setattr(EvaluationService, "evaluate", lambda _self, _options: {"error": ["Model routing unavailable"]})
    monkeypatch.setenv("NVIDIA_API_KEY", "private-provider-key-1234")
    assert run_tiers.native_tier3(consumer / "skills/alpha", tmp_path, env_for(consumer)) == 0
    report = json.loads((tmp_path / "skillevaluator-tier3.json").read_text())
    assert report["results"][0]["status"] == "incomplete"
    assert "Model routing unavailable" in json.dumps(report)


def test_child_environment_never_passes_feedback_or_repository_tokens():
    env = run_tiers.child_environment(
        {
            "PATH": "/bin",
            "GITHUB_TOKEN": "repository-token",
            "CI_JOB_TOKEN": "job-token",
            "SKILLEVALUATOR_GITLAB_TOKEN": "bot-token",
            "NVIDIA_API_KEY": "provider-token",
            "SKILL_EVAL_PROVIDER": "nv_build",
        }
    )
    assert env["NVIDIA_API_KEY"] == "provider-token"
    assert not set(env) & {"GITHUB_TOKEN", "GH_TOKEN", "CI_JOB_TOKEN", "SKILLEVALUATOR_GITLAB_TOKEN"}


def test_setup_workflow_only_change_selects_full_collection(consumer):
    base = subprocess.check_output(["git", "-C", str(consumer), "rev-parse", "HEAD"], text=True).strip()
    file = consumer / ".github/workflows/skill-checks.yml"
    file.parent.mkdir(parents=True)
    file.write_text("name: Skill checks\n")
    git = ["git", "-C", str(consumer), "-c", "user.name=CI tests", "-c", "user.email=ci@example.com"]
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-m", "setup"], check=True, capture_output=True)
    selected, selection = run_tiers.select_skills(
        consumer, env_for(consumer, DIFF_BASE_SHA=base, SKILL_EVAL_SCAN_MODE="auto")
    )
    assert [p.name for p in selected] == ["alpha", "beta"]
    assert selection["reason"] == "ci_configuration_changed"


def test_changed_selection_without_changes_is_explicitly_skipped(consumer, tmp_path):
    env = env_for(consumer)
    env["DIFF_BASE_SHA"] = env["REPORT_COMMIT"]
    code = run_tiers.evaluate(2, consumer, tmp_path / "reports", env)
    result = json.loads((tmp_path / "reports/tier2/ci-tier.json").read_text())
    assert code == 0
    assert result["tiers"][0]["status"] == "skipped"
    assert result["tiers"][0]["skills"] == []


def test_report_directory_cannot_modify_consumer_checkout(consumer):
    with pytest.raises(run_tiers.ConfigurationError, match="outside"):
        run_tiers.evaluate(2, consumer, consumer / "reports", env_for(consumer))


def test_head_identity_is_checked_before_full_scan(consumer, tmp_path):
    with pytest.raises(run_tiers.ConfigurationError, match="commit"):
        run_tiers.evaluate(2, consumer, tmp_path / "reports", env_for(consumer, REPORT_COMMIT="a" * 40))


def test_interrupted_second_skill_cannot_publish_a_passing_partial_gate(consumer, tmp_path, monkeypatch):
    from skillevaluator.models import ValidationResult
    from skillevaluator.reporting.json_reporter import JSONReporter

    attempts = []

    def execute(args, **_kwargs):
        if "catalog" in args:
            return 0
        attempts.append(args)
        if len(attempts) == 2:
            raise KeyboardInterrupt
        result = ValidationResult(
            validator_name="Context Deduplication", metadata={"gating": {"tier": 2, "blocking": True}}
        )
        output = Path(args[args.index("-o") + 1])
        (output / "skillevaluator-tier2.json").write_text(JSONReporter().render_all([result]))
        return 0

    monkeypatch.setattr(run_tiers, "run_command", execute)
    reports = tmp_path / "reports"
    env = env_for(consumer, NVIDIA_API_KEY="test-provider-value-1234")
    with pytest.raises(KeyboardInterrupt):
        run_tiers.evaluate(2, consumer, reports, env)
    retained = json.loads((reports / "tier2/ci-tier.json").read_text())
    tier = retained["tiers"][0]
    assert tier["status"] == "incomplete"
    assert len(tier["skills"]) == 2
    assert [row["status"] for row in tier["skills"]] == ["passed", "incomplete"]
    assert retained["overall_passed"] is False
    assert run_tiers.aggregate(reports, (2,), env) == 1


@pytest.mark.parametrize("phase", ["doctor", "catalog", "evaluation"])
def test_worker_failure_keeps_redacted_diagnostics_after_scratch_cleanup(consumer, tmp_path, monkeypatch, phase):
    secret = "test-provider-secret-never-retain-12345"
    scratch_paths = []

    def execute(args, *, log, **_kwargs):
        log.write_text(f"{phase} startup failure NVIDIA_API_KEY={secret}\n")
        scratch_paths.append(log.parent)
        if phase == "evaluation":
            if "doctor" in args:
                return 0
            raw = Path(args[args.index("--output") + 1])
            trial = raw / "harbor/trials/case"
            trial.mkdir(parents=True)
            (trial / "exception.txt").write_text(f"Compose startup failed; provider={secret}\n")
        return 1

    monkeypatch.setattr(run_tiers, "dataset_preflight", lambda _skill: {})
    monkeypatch.setattr(run_tiers, "run_command", execute)
    tier = 2 if phase == "catalog" else 3
    output = tmp_path / "reports"
    code = run_tiers.evaluate(tier, consumer, output, env_for(consumer, NVIDIA_API_KEY=secret))
    assert code == (1 if tier == 2 else 0)
    assert all(not path.exists() for path in scratch_paths)
    diagnostics = list((output / f"tier{tier}").rglob("diagnostics/index.json"))
    assert diagnostics, "Temporary worker errors must survive even without a canonical report"
    retained = "\n".join(p.read_text() for p in output.rglob("*") if p.is_file())
    assert secret not in retained
    assert f"{phase} startup failure" in retained
    if phase == "evaluation":
        assert "Compose startup failed" in retained
    document = json.loads((output / f"tier{tier}/ci-tier.json").read_text())
    assert document["tiers"][0]["status"] == "incomplete"


def test_consumer_python_module_cannot_shadow_installed_evaluator(consumer, tmp_path):
    shadow = consumer / "skillevaluator"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("")
    (shadow / "cli.py").write_text("print('CANDIDATE_CODE_EXECUTED')\n")
    args = run_tiers.skill_command(2, consumer / "skills/alpha", tmp_path / "raw", env_for(consumer))
    args = [*args[: args.index("skillevaluator.cli") + 1], "--version"]
    log = tmp_path / "command.log"
    code = run_tiers.run_command(
        args, env=run_tiers.child_environment(env_for(consumer)), cwd=consumer, timeout=30, log=log
    )
    assert code == 0
    assert "CANDIDATE_CODE_EXECUTED" not in log.read_text()
    assert "skillevaluator" in log.read_text().lower()


@pytest.mark.parametrize("legacy_status,expected_code", [("skipped", 0), ("error", 1)])
def test_legacy_tier1_no_work_and_runtime_error_still_produce_aggregate(
    consumer, tmp_path, legacy_status, expected_code
):
    from ci.run_tier1 import _summary

    env = env_for(consumer)
    reports = tmp_path / "reports"
    (reports / "tier1").mkdir(parents=True)
    legacy = _summary([], "Runtime unavailable" if legacy_status == "error" else None)
    legacy.update(overall_status=legacy_status, execution={"commit": env["REPORT_COMMIT"]})
    if legacy_status == "skipped":
        _, legacy["selection"] = run_tiers.select_changed_skills(
            consumer, "skills", env["REPORT_COMMIT"], env["REPORT_COMMIT"]
        )
    (reports / "tier1/summary.json").write_text(json.dumps(legacy))
    assert run_tiers.aggregate(reports, (1,), env) == expected_code
    document = json.loads((reports / "ci-results.json").read_text())
    assert document["tiers"][0]["status"] == ("skipped" if legacy_status == "skipped" else "incomplete")
    assert document["tiers"][0]["reason"]
    assert (reports / "index.html").is_file()
