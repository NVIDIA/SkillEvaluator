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


@pytest.mark.parametrize("tier", [2, 3])
@pytest.mark.parametrize("legacy_required", ["false", "true", "unused-legacy-value"])
def test_missing_credentials_retains_incomplete_advisory_tier(consumer, tmp_path, tier, legacy_required):
    output = tmp_path / "reports"
    code = run_tiers.evaluate(tier, consumer, output, env_for(consumer, SKILL_EVAL_TIER3_REQUIRED=legacy_required))
    result = json.loads((output / f"tier{tier}" / "ci-tier.json").read_text())
    assert code == 0
    assert result["tiers"][0]["status"] == "incomplete"
    assert result["tiers"][0]["required"] is False
    assert result["overall_passed"] is True
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


def test_native_agent_models_survive_the_serialized_execution_boundary(consumer, tmp_path, monkeypatch):
    requested = "codex=openai/example-gpt\nclaude-code=anthropic/example-claude"
    observed = []
    doctors = []
    monkeypatch.setattr(run_tiers, "dataset_preflight", lambda _skill: {})

    def execute(args, **_kwargs):
        if "doctor" in args:
            doctors.append(args)
            return 0
        config_path = Path(args[args.index("--config") + 1])
        saved = json.loads(config_path.read_text())
        options = run_tiers.tier3_options(consumer / "skills/alpha", config_path.parent, saved)
        observed.append(options.agent_model)
        return 1  # No model calls; missing evidence must remain advisory.

    monkeypatch.setattr(run_tiers, "run_command", execute)
    environment = env_for(
        consumer,
        NVIDIA_API_KEY="fixture-build-key",
        SKILL_EVAL_AGENTS="codex,claude-code",
        SKILL_EVAL_AGENT_MODELS=requested,
    )
    assert run_tiers.evaluate(3, consumer, tmp_path / "reports", environment) == 0
    expected = tuple(requested.splitlines())
    assert observed == [expected, expected]
    checked = tuple(
        args[index + 1] for args in doctors for index, value in enumerate(args[:-1]) if value == "--agent-model"
    )
    assert checked == expected


def test_native_tier3_engine_failure_still_emits_incomplete_evidence(consumer, tmp_path, monkeypatch):
    from skillevaluator.evaluation.service import EvaluationService

    monkeypatch.setattr(run_tiers, "dataset_preflight", lambda _skill: {})
    monkeypatch.setattr(EvaluationService, "evaluate", lambda _self, _options: {"error": ["Model routing unavailable"]})
    monkeypatch.setenv("NVIDIA_API_KEY", "private-provider-key-1234")
    assert (
        run_tiers.native_tier3(consumer / "skills/alpha", tmp_path, env_for(consumer, SKILL_EVAL_TIER3_REQUIRED="true"))
        == 0
    )
    report = json.loads((tmp_path / "skillevaluator-tier3.json").read_text())
    assert [entry["name"] for entry in report["skills"]] == ["alpha"]
    assert report["results"][0]["status"] == "incomplete"
    assert report["gating"]["tiers"]["3"]["blocking"] is False
    assert report["results"][0]["gating"] == {"tier": 3, "blocking": False}
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


def test_hidden_setup_gitlink_still_selects_full_collection(consumer):
    """Setup detection reads the same list as the paid-job skip, so .gitmodules cannot hide it."""
    base = subprocess.check_output(["git", "-C", str(consumer), "rev-parse", "HEAD"], text=True).strip()
    (consumer / ".gitmodules").write_text(
        '[submodule "actions"]\n\tpath = .github/actions\n\turl = ./actions\n\tignore = all\n'
    )
    git = ["git", "-C", str(consumer), "-c", "user.name=CI tests", "-c", "user.email=ci@example.com"]
    subprocess.run([*git, "add", ".gitmodules"], check=True)
    subprocess.run([*git, "update-index", "--add", "--cacheinfo", f"160000,{'b' * 40},.github/actions"], check=True)
    subprocess.run([*git, "commit", "-m", "setup gitlink"], check=True, capture_output=True)
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


def test_interrupted_advisory_tier_retains_honest_partial_results(consumer, tmp_path, monkeypatch):
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
    assert tier["required"] is False
    assert retained["overall_passed"] is True
    assert run_tiers.aggregate(reports, (2,), env) == 0


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
    assert code == 0
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


@pytest.mark.parametrize("tier", [2, 3])
def test_advisory_timeout_retains_incomplete_outcome_and_diagnostics(consumer, tmp_path, monkeypatch, tier):
    def execute(args, *, log, **_kwargs):
        log.write_text("Worker exceeded its time budget.\n")
        return 0 if "doctor" in args or "catalog" in args else 124

    monkeypatch.setattr(run_tiers, "dataset_preflight", lambda _skill: {})
    monkeypatch.setattr(run_tiers, "run_command", execute)
    reports = tmp_path / "reports"
    assert (
        run_tiers.evaluate(
            tier,
            consumer,
            reports,
            env_for(consumer, NVIDIA_API_KEY="test-provider-value-1234", SKILL_EVAL_TIER3_REQUIRED="true"),
        )
        == 0
    )
    retained = json.loads((reports / f"tier{tier}/ci-tier.json").read_text())
    assert retained["tiers"][0]["status"] == "incomplete"
    assert all(skill["status"] == "incomplete" and skill["error"] for skill in retained["tiers"][0]["skills"])
    assert retained["overall_passed"] is True
    assert list((reports / f"tier{tier}").rglob("diagnostics/index.json"))


def write_tier(reports, tier, env, status, *, required=True):
    document = run_tiers.context(env)
    document.update(
        expected_tiers=[tier],
        tiers=[
            {
                "tier": tier,
                "required": required,
                "status": status,
                "skills": [
                    {
                        "path": "skills/alpha",
                        "status": status,
                        "report_json": f"tier{tier}/skill-001/report.json",
                        "report_html": f"tier{tier}/skill-001/report.html",
                        "metrics": {"findings": 7},
                    }
                ],
            }
        ],
    )
    run_tiers.write_json(reports / f"tier{tier}/ci-tier.json", document)


@pytest.mark.parametrize("status", ["failed", "incomplete", "needs_action", "not_run", "neutral", "skipped", None])
def test_aggregate_only_tier1_blocks_despite_legacy_paid_requirements(consumer, tmp_path, status):
    reports = tmp_path / "reports"
    env = env_for(consumer, SKILL_EVAL_TIER3_REQUIRED="true")
    write_tier(reports, 1, env, "passed")
    if status is not None:
        for tier in (2, 3):
            write_tier(reports, tier, env, status)
    assert run_tiers.aggregate(reports, (1, 2, 3), env) == 0
    document = json.loads((reports / "ci-results.json").read_text())
    assert document["overall_passed"] is True
    assert document["overall_status"] == "passed"
    assert [row["required"] for row in document["tiers"]] == [True, False, False]
    assert [row["status"] for row in document["tiers"]] == ["passed", status or "incomplete", status or "incomplete"]
    if status is not None:
        assert document["tiers"][1]["skills"][0]["report_html"] == "tier2/skill-001/report.html"
        assert document["tiers"][2]["skills"][0]["metrics"] == {"findings": 7}


@pytest.mark.parametrize("status", ["failed", "incomplete", None])
def test_aggregate_tier1_nonpass_still_blocks(consumer, tmp_path, status):
    reports = tmp_path / "reports"
    env = env_for(consumer)
    if status is not None:
        write_tier(reports, 1, env, status)
    for tier in (2, 3):
        write_tier(reports, tier, env, "passed")
    assert run_tiers.aggregate(reports, (1, 2, 3), env) == 1
    document = json.loads((reports / "ci-results.json").read_text())
    assert document["overall_passed"] is False
    assert document["tiers"][0]["status"] == (status or "incomplete")


@pytest.mark.parametrize("damage", ["identity", "required", "path", "status"])
def test_advisory_artifact_validation_remains_strict(consumer, tmp_path, damage):
    reports = tmp_path / "reports"
    env = env_for(consumer)
    write_tier(reports, 1, env, "passed")
    write_tier(reports, 2, env, "failed")
    path = reports / "tier2/ci-tier.json"
    document = json.loads(path.read_text())
    if damage == "identity":
        document["commit"] = "f" * 40
    elif damage == "required":
        document["tiers"][0]["required"] = "true"
    elif damage == "path":
        document["tiers"][0]["skills"][0]["report_html"] = "../private.html"
    else:
        document["tiers"][0]["status"] = "unknown"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        run_tiers.aggregate(reports, (1, 2, 3), env)
    assert not (reports / "ci-results.json").exists()


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


def test_child_environment_forwards_docker_tls_settings():
    env = run_tiers.child_environment(
        {
            "PATH": "/bin",
            "DOCKER_HOST": "tcp://docker:2376",
            "DOCKER_TLS_VERIFY": "1",
            "DOCKER_CERT_PATH": "/certs/client",
            "NVIDIA_API_KEY": "provider-token",
            "SKILL_EVAL_PROVIDER": "nv_build",
        }
    )
    assert env["DOCKER_TLS_VERIFY"] == "1"
    assert env["DOCKER_CERT_PATH"] == "/certs/client"


def test_anthropic_chat_with_unset_embedding_provider_uses_openai_embeddings():
    env = run_tiers.child_environment(
        {
            "SKILL_EVAL_PROVIDER": "anthropic",
            "SKILL_EVAL_EMBEDDING_PROVIDER": "",
            "ANTHROPIC_API_KEY": "anthropic-token",
            "OPENAI_API_KEY": "openai-token",
        }
    )
    assert env["SKILL_EVAL_EMBEDDING_PROVIDER"] == "openai"
    assert env["OPENAI_API_KEY"] == "openai-token"


def test_post_run_dataset_recheck_failure_downgrades_the_skill(consumer, tmp_path, monkeypatch):
    calls = []

    def preflight(_skill):
        calls.append(_skill)
        if len(calls) % 2 == 0:
            raise run_tiers.ConfigurationError("Tier 3 authored inputs cannot contain symlinks.")
        return {}

    monkeypatch.setattr(run_tiers, "dataset_preflight", preflight)
    monkeypatch.setattr(run_tiers, "run_command", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(run_tiers, "_retain", lambda *_args: {"status": "passed", "metrics": {}, "complete": True})
    reports = tmp_path / "reports"
    env = env_for(consumer, NVIDIA_API_KEY="test-provider-value-1234")
    assert run_tiers.evaluate(3, consumer, reports, env) == 0
    tier = json.loads((reports / "tier3/ci-tier.json").read_text())["tiers"][0]
    assert tier["status"] == "incomplete"
    assert {(s["status"], s["error"]) for s in tier["skills"]} == {
        ("incomplete", "Tier 3 authored inputs cannot contain symlinks.")
    }
    assert run_tiers.aggregate(reports, (3,), env) == 0
    assert (reports / "ci-results.json").is_file()
    assert (reports / "summary.md").is_file()


def test_timeout_keeps_the_time_budget_diagnostic_for_every_skill(consumer, tmp_path, monkeypatch):
    import sys

    real_run = run_tiers.run_command

    def execute(args, **kwargs):
        if "catalog" in args:
            return 0
        return real_run([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)

    monkeypatch.setattr(run_tiers, "run_command", execute)
    reports = tmp_path / "reports"
    env = env_for(consumer, NVIDIA_API_KEY="test-provider-value-1234", SKILL_EVAL_TIMEOUT_SECONDS="1")
    assert run_tiers.evaluate(2, consumer, reports, env) == 0
    skills = json.loads((reports / "tier2/ci-tier.json").read_text())["tiers"][0]["skills"]
    # The second skill starts after the budget is spent and must say so too.
    assert [(s["status"], s["error"]) for s in skills] == [
        ("incomplete", "Evaluation exceeded the configured time budget.")
    ] * 2


@pytest.mark.parametrize("status", ["passed", "neutral", "skipped"])
def test_aggregate_rejects_a_completed_claim_missing_selected_skills(consumer, tmp_path, status):
    reports = tmp_path / "reports"
    env = env_for(consumer)
    write_tier(reports, 1, env, "passed")
    write_tier(reports, 3, env, status)
    path = reports / "tier3/ci-tier.json"
    document = json.loads(path.read_text())
    document["selection"] = {"reason": "full_scan", "selected": ["skills/alpha", "skills/beta"]}
    path.write_text(json.dumps(document))
    assert run_tiers.aggregate(reports, (1, 3), env) == 0
    tier3 = json.loads((reports / "ci-results.json").read_text())["tiers"][1]
    assert tier3["status"] == "incomplete"
    assert "missing for selected skills" in tier3["reason"]


def test_skill_directories_with_url_delimiters_still_aggregate(tmp_path, monkeypatch):
    from skillevaluator.models import ValidationResult
    from skillevaluator.reporting.json_reporter import JSONReporter

    repo = tmp_path / "consumer"
    names = ("c#-helper", "100%-coverage", "a\\b")
    for name in names:
        skill = repo / "skills" / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: helper\ndescription: Use to test paths.\n---\nInstructions.\n")
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    git = ["git", "-C", str(repo), "-c", "user.name=CI tests", "-c", "user.email=ci@example.com"]
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-m", "fixtures"], check=True, capture_output=True)

    def execute(args, **_kwargs):
        if "catalog" in args:
            return 0
        result = ValidationResult(
            validator_name="Context Deduplication", metadata={"gating": {"tier": 2, "blocking": False}}
        )
        (Path(args[args.index("-o") + 1]) / "skillevaluator-tier2.json").write_text(JSONReporter().render_all([result]))
        return 0

    monkeypatch.setattr(run_tiers, "run_command", execute)
    reports = tmp_path / "reports"
    env = env_for(repo, NVIDIA_API_KEY="test-provider-value-1234")
    assert run_tiers.evaluate(2, repo, reports, env) == 0
    assert run_tiers.aggregate(reports, (2,), env) == 0
    tier2 = json.loads((reports / "ci-results.json").read_text())["tiers"][0]
    assert tier2["status"] == "passed"
    assert sorted(s["path"] for s in tier2["skills"]) == sorted(f"skills/{name}" for name in names)


def test_retained_reports_record_the_repository_relative_target_not_the_runner_path(consumer, tmp_path, monkeypatch):
    from skillevaluator.models import ValidationResult
    from skillevaluator.reporting.json_reporter import JSONReporter

    def execute(args, **_kwargs):
        if "catalog" in args:
            return 0
        result = ValidationResult(
            validator_name="Context Deduplication", metadata={"gating": {"tier": 2, "blocking": False}}
        )
        # The evaluator records the target as the CI passed it, an absolute runner path.
        document = JSONReporter(target_path=args[args.index("tier2") + 1]).render_all([result])
        (Path(args[args.index("-o") + 1]) / "skillevaluator-tier2.json").write_text(document)
        return 0

    monkeypatch.setattr(run_tiers, "run_command", execute)
    reports = tmp_path / "reports"
    env = env_for(consumer, NVIDIA_API_KEY="test-provider-value-1234")
    assert run_tiers.evaluate(2, consumer, reports, env) == 0

    targets = {
        child.name: json.loads((child / "skillevaluator-tier2.json").read_text())["target_path"]
        for child in sorted((reports / "tier2").glob("skill-*"))
    }
    assert targets == {"skill-001": "skills/alpha", "skill-002": "skills/beta"}


def write_attempt(staging, tier, attempt, env, status):
    """Lay out one downloaded tier artifact as actions/download-artifact does for a pattern."""
    artifact = staging / f"skill-evaluator-tier{tier}-attempt-{attempt}"
    write_tier(artifact, tier, {**env, "GITHUB_RUN_ATTEMPT": str(attempt)}, status)
    return artifact


def test_rerun_failed_jobs_reuses_earlier_attempt_tier_evidence(consumer, tmp_path):
    from ci.run_tier1 import _summary

    env = env_for(consumer, GITHUB_RUN_ATTEMPT="2")
    staging, reports = tmp_path / "artifacts", tmp_path / "reports"
    # Attempt 1: Tier 1 had no work and Tier 3 failed; only the failed Tier 2 job was rerun.
    tier1 = staging / "skill-evaluator-tier1-attempt-1"
    tier1.mkdir(parents=True)
    legacy = _summary([], None)
    legacy.update(overall_status="skipped", execution={"commit": env["REPORT_COMMIT"]})
    _, legacy["selection"] = run_tiers.select_changed_skills(
        consumer, "skills", env["REPORT_COMMIT"], env["REPORT_COMMIT"]
    )
    (tier1 / "summary.json").write_text(json.dumps(legacy))
    write_attempt(staging, 2, 1, env, "incomplete")
    write_attempt(staging, 2, 2, env, "passed")
    write_attempt(staging, 3, 1, env, "failed")
    # Neither a future attempt nor another artifact family can be selected.
    write_attempt(staging, 3, 3, env, "passed")
    (staging / "skill-evaluator-results-attempt-1").mkdir()
    (staging / "skill-evaluator-tier1-provenance-attempt-2").mkdir()

    assert run_tiers.select_attempts(staging, reports, 2) == {"1": "1", "2": "2", "3": "1"}
    assert json.loads((reports / "source-attempts.json").read_text()) == {"1": "1", "2": "2", "3": "1"}
    assert run_tiers.aggregate(reports, (1, 2, 3), env) == 0
    document = json.loads((reports / "ci-results.json").read_text())
    assert document["run_attempt"] == "2"
    assert [row["status"] for row in document["tiers"]] == ["skipped", "passed", "failed"]
    assert [row["source_attempt"] for row in document["tiers"]] == ["1", "2", "1"]
    assert "evidence from run attempt 1" in (reports / "summary.md").read_text()


def test_select_attempts_cli_reads_the_current_attempt(consumer, tmp_path, monkeypatch):
    env = env_for(consumer)
    staging, reports = tmp_path / "artifacts", tmp_path / "reports"
    write_attempt(staging, 2, 1, env, "passed")
    write_attempt(staging, 2, 2, env, "failed")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    assert run_tiers.main(["select-attempts", "--artifacts", str(staging), "--reports", str(reports)]) == 0
    assert json.loads((reports / "tier2/ci-tier.json").read_text())["run_attempt"] == "1"


@pytest.mark.parametrize("damage", ["future", "run_id", "recorded", "recorded_future"])
def test_aggregate_rejects_foreign_or_future_tier_attempts(consumer, tmp_path, damage):
    env = env_for(consumer, GITHUB_RUN_ATTEMPT="2")
    reports = tmp_path / "reports"
    write_tier(reports, 1, env, "passed")
    write_tier(reports, 2, {**env, "GITHUB_RUN_ATTEMPT": "1"}, "passed")
    path = reports / "tier2/ci-tier.json"
    document = json.loads(path.read_text())
    if damage == "future":
        document["run_attempt"] = "3"
    elif damage == "run_id":
        document["run_id"] = "124"
    path.write_text(json.dumps(document))
    if damage.startswith("recorded"):
        recorded = {"2": "2"} if damage == "recorded" else {"2": "3"}
        (reports / "source-attempts.json").write_text(json.dumps(recorded))
    with pytest.raises(run_tiers.ConfigurationError):
        run_tiers.aggregate(reports, (1, 2), env)
    assert not (reports / "ci-results.json").exists()


def _commit_change(consumer, *, write=(), rename=()):
    git = ["git", "-C", str(consumer), "-c", "user.name=CI tests", "-c", "user.email=ci@example.com"]
    for relative, text in write:
        path = consumer / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for source, target in rename:
        (consumer / target).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([*git, "mv", source, target], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-m", "change"], check=True, capture_output=True)
    return subprocess.check_output(["git", "-C", str(consumer), "rev-parse", "HEAD"], text=True).strip()


def test_unrelated_ci_directory_change_keeps_the_tier1_changed_selection(consumer):
    base = subprocess.check_output(["git", "-C", str(consumer), "rev-parse", "HEAD"], text=True).strip()
    # A consumer's own ci/ scripts are not SkillEvaluator setup; Tier 1 keeps changed mode here too.
    _commit_change(
        consumer, write=[("ci/build.sh", "echo build\n"), ("skills/alpha/SKILL.md", "---\nname: alpha\n---\n")]
    )
    selected, selection = run_tiers.select_skills(consumer, env_for(consumer, DIFF_BASE_SHA=base))
    assert [p.name for p in selected] == ["alpha"]
    assert selection["mode"] == "changed" and "reason" not in selection


def test_setup_file_renamed_away_still_selects_full_collection(consumer):
    _commit_change(consumer, write=[(".gitlab/skill-feedback.yml", "include: []\n" * 20)])
    base = subprocess.check_output(["git", "-C", str(consumer), "rev-parse", "HEAD"], text=True).strip()
    # GitHub eligibility checks a rename's previous path; native selection must see it too.
    _commit_change(consumer, rename=[(".gitlab/skill-feedback.yml", "docs/skill-feedback.yml")])
    selected, selection = run_tiers.select_skills(consumer, env_for(consumer, DIFF_BASE_SHA=base))
    assert [p.name for p in selected] == ["alpha", "beta"]
    assert selection["reason"] == "ci_configuration_changed"


def test_aggregate_keeps_each_tier_selection_so_unselected_skills_are_not_incomplete(consumer, tmp_path):
    from ci.comment_summary import render_comment_summary

    reports = tmp_path / "reports"
    env = env_for(consumer)
    for tier, selected in ((1, ["skills/alpha"]), (2, ["skills/alpha", "skills/beta"])):
        write_tier(reports, tier, env, "passed")
        path = reports / f"tier{tier}/ci-tier.json"
        document = json.loads(path.read_text())
        document["selection"] = {"mode": "changed", "selected": selected}
        document["tiers"][0]["selected"] = ["forged"]
        skill = document["tiers"][0]["skills"][0]
        document["tiers"][0]["skills"] = [
            {
                **skill,
                "path": p,
                "report_json": f"tier{tier}/skill-00{i}/r.json",
                "report_html": f"tier{tier}/skill-00{i}/r.html",
            }
            for i, p in enumerate(selected, 1)
        ]
        path.write_text(json.dumps(document))
    assert run_tiers.aggregate(reports, (1, 2), env) == 0
    manifest = json.loads((reports / "ci-results.json").read_text())
    assert [row["selected"] for row in manifest["tiers"]] == [["skills/alpha"], ["skills/alpha", "skills/beta"]]
    body = render_comment_summary(manifest, "https://github.com/example/consumer/actions/runs/123/artifacts/9")
    beta = next(line for line in body.splitlines() if line.startswith("| skills/beta |"))
    assert "Required gate: **Passed**" in body
    assert beta.split(" | ")[1] == "SKIPPED · issues unknown · Not selected for this tier"
    assert "INCOMPLETE" not in beta


def _failed_tier2_report(output: Path, *, incomplete: bool = False) -> None:
    from skillevaluator.models import Finding, Severity, ValidationResult
    from skillevaluator.reporting.json_reporter import JSONReporter

    result = ValidationResult(
        validator_name="Context Deduplication",
        passed=False,
        errors=["Duplicate deployment instructions"],
        metadata={"gating": {"tier": 2, "blocking": True}},
    )
    if incomplete:
        # A blocking finding outranks the incomplete scan, so the status stays failed.
        result.add_finding(
            Finding(
                category="DEDUP",
                severity=Severity.HIGH,
                check_name="duplicate-instructions",
                message="Duplicate deployment instructions",
                file_path="SKILL.md",
            )
        )
        result.mark_scan_incomplete("LLM review")
    (output / "skillevaluator-tier2.json").write_text(JSONReporter().render_all([result]))


@pytest.mark.parametrize(
    ("failed", "incomplete", "status"), [(True, False, "failed"), (False, True, "incomplete"), (True, True, "failed")]
)
def test_retained_tier2_reports_apply_ci_advisory_policy(consumer, tmp_path, failed, incomplete, status):
    from skillevaluator.models import Finding, Severity, ValidationResult
    from skillevaluator.publication_evidence import stamp_publication_evidence
    from skillevaluator.publication_identity import finalize_publication_target, publication_target_from_path
    from skillevaluator.reporting.json_reporter import JSONReporter
    from skillevaluator.reporting.retained import retained_results
    from skillevaluator.reporting.run_status import resolve_run_status
    from skillevaluator.tier_workflows import _record_scope

    skill = consumer / "skills/alpha"
    result = ValidationResult(validator_name="Context Deduplication")
    if failed:
        result.add_finding(
            Finding(
                category="DEDUP",
                severity=Severity.HIGH,
                check_name="duplicate-instructions",
                message="Duplicate deployment instructions",
                file_path="SKILL.md",
            )
        )
    if incomplete:
        result.mark_scan_incomplete("LLM review")
    stamp_publication_evidence([result], tier=2, check_id="context-optimization")
    finalize_publication_target([result], skill, publication_target_from_path(skill))
    _record_scope([result], 2, {"intra_skill": {"status": status}})
    env = env_for(consumer)
    source = {"repository": "example/consumer", "commit": env["REPORT_COMMIT"]}
    result.metadata["evaluated_source"] = source
    document = JSONReporter(target_path=str(skill)).render_all([result])
    original = json.loads(document)
    assert original["target_path"] == str(skill)
    assert original["overall_passed"] is False
    assert original["results"][0]["gating"] == {"tier": 2, "blocking": True}
    assert original["gating"]["tiers"]["2"]["validator_policies"] == [
        {"validator": "Context Deduplication", "blocking": True}
    ]
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "skillevaluator-tier2.json").write_text(document)
    reports = tmp_path / "reports"
    destination = reports / "tier2/skill-001"

    outcome = run_tiers._retain(raw, destination, 2, set(), skill, 1, "skills/alpha")

    assert outcome == {"status": status, "metrics": {"findings": int(failed)}, "complete": not incomplete}
    retained = json.loads((destination / "skillevaluator-tier2.json").read_text())
    assert retained["gating"]["tiers"]["2"]["blocking"] is False
    assert retained["gating"]["tiers"]["2"]["validator_policies"] == [
        {"validator": "Context Deduplication", "blocking": False}
    ]
    assert retained["overall_passed"] is True
    assert retained["overall_status"] == "passed"
    assert retained["results"][0]["gating"] == {"tier": 2, "blocking": False}
    assert retained["results"][0]["status"] == status
    assert retained["results"][0]["passed"] is False
    assert retained["evaluated_source"] == source
    # The evaluator saw the runner's absolute path; the retained report keeps the repository path.
    assert retained["target_path"] == "skills/alpha"
    for key in set(original) - {"gating", "overall_passed", "overall_status", "results", "target_path"}:
        assert retained[key] == original[key], key
    for key in set(original["results"][0]) - {"gating"}:
        assert retained["results"][0][key] == original["results"][0][key], key
    restored = retained_results(retained, allowed_tiers={2})
    assert resolve_run_status(restored).tiers[2].status.value == status
    html = (destination / "skillevaluator-tier2.html").read_text()
    assert "Advisory (Tier 2)" in html
    assert "Blocking (Tier 2)" not in html
    if failed:
        assert "Failed (advisory)" in html
    markdown = (destination / "skillevaluator-tier2.md").read_text()
    assert "**Status:** ✅ PASSED" in markdown
    assert f"| ❌ Failed | {int(failed)} |" in markdown
    assert f"| ⚠️ Incomplete | {int(incomplete)} |" in markdown
    assert (raw / "skillevaluator-tier2.json").read_text() == document
    assert result.metadata["gating"] == {"tier": 2, "blocking": True}

    tier2 = run_tiers.context(env)
    tier2.update(
        expected_tiers=[2],
        tiers=[
            {
                "tier": 2,
                "required": False,
                "status": status,
                "skills": [
                    {
                        "path": "skills/alpha",
                        "status": status,
                        "report_json": "tier2/skill-001/skillevaluator-tier2.json",
                        "report_html": "tier2/skill-001/skillevaluator-tier2.html",
                    }
                ],
            }
        ],
    )
    run_tiers.write_json(reports / "tier2/ci-tier.json", tier2)
    write_tier(reports, 1, env, "passed")
    assert run_tiers.aggregate(reports, (1, 2), env) == 0
    aggregate = json.loads((reports / "ci-results.json").read_text())
    assert aggregate["overall_passed"] is True
    assert aggregate["tiers"][1]["status"] == status


@pytest.mark.parametrize(
    ("code", "incomplete", "retained"), [(1, False, False), (1, True, True), (2, False, True), (-9, False, True)]
)
def test_worker_diagnostics_are_kept_for_execution_problems_not_complete_verdicts(
    consumer, tmp_path, monkeypatch, code, incomplete, retained
):
    def execute(args, *, log, **_kwargs):
        if "catalog" in args:
            return 0
        log.write_text("worker output\n")
        _failed_tier2_report(Path(args[args.index("-o") + 1]), incomplete=incomplete)
        return code

    monkeypatch.setattr(run_tiers, "run_command", execute)
    reports = tmp_path / "reports"
    assert run_tiers.evaluate(2, consumer, reports, env_for(consumer, NVIDIA_API_KEY="test-provider-value-1234")) == 0
    skills = json.loads((reports / "tier2/ci-tier.json").read_text())["tiers"][0]["skills"]
    assert [s["status"] for s in skills] == ["failed", "failed"]
    assert bool(list((reports / "tier2").rglob("diagnostics"))) is retained


def test_neutral_tier3_verdict_keeps_no_worker_diagnostics(consumer, tmp_path, monkeypatch):
    def execute(args, *, log, **_kwargs):
        log.write_text("worker output\n")
        if "doctor" in args:
            return 0
        trial = Path(args[args.index("--output") + 1]) / "harbor/run/_harbor-jobs/job/trial"
        trial.mkdir(parents=True)
        (trial / "trial.log").write_text("trial ran\n")
        return 1  # native-tier3 exits 1 for any headline other than passed

    monkeypatch.setattr(run_tiers, "dataset_preflight", lambda _skill: {})
    monkeypatch.setattr(run_tiers, "run_command", execute)
    monkeypatch.setattr(run_tiers, "_retain", lambda *_args: {"status": "neutral", "metrics": {}, "complete": True})
    reports = tmp_path / "reports"
    assert run_tiers.evaluate(3, consumer, reports, env_for(consumer, NVIDIA_API_KEY="test-provider-value-1234")) == 0
    tier = json.loads((reports / "tier3/ci-tier.json").read_text())["tiers"][0]
    assert {s["status"] for s in tier["skills"]} == {"neutral"}
    assert not list((reports / "tier3").rglob("diagnostics"))


@pytest.mark.parametrize("code", [0, 1])
def test_failed_tier3_verdict_with_incomplete_execution_keeps_worker_diagnostics(consumer, tmp_path, monkeypatch, code):
    from skillevaluator.evaluation.tier3_report import _validation_result_from_payload
    from skillevaluator.reporting.json_reporter import JSONReporter

    def execute(args, *, log, **_kwargs):
        log.write_text("worker output: codex installation failed\n")
        if "doctor" in args:
            return 0
        output = Path(args[args.index("--output") + 1])
        trial = output / "harbor/run/_harbor-jobs/skill-codex-with/trial"
        trial.mkdir(parents=True)
        (trial / "trial.log").write_text("trial ran\n")
        (trial / "exception.txt").write_text("AgentInstallError\n")
        # One agent failed to execute while another finished with a failing score.
        result = _validation_result_from_payload(
            {
                "execution_status": "failed",
                "execution_errors": ["codex: agent installation failed"],
                "overall_score": 0.41,
                "verdict": "fail",
                "summary": {"verdict": "fail"},
            }
        )
        result.metadata["gating"] = {"tier": 3, "blocking": False}
        result.mark_scan_incomplete("Tier 3 execution")  # as native_tier3 records a failure reason
        (output / "skillevaluator-tier3.json").write_text(JSONReporter().render_all([result]))
        return code

    monkeypatch.setattr(run_tiers, "dataset_preflight", lambda _skill: {})
    monkeypatch.setattr(run_tiers, "run_command", execute)
    reports = tmp_path / "reports"
    assert run_tiers.evaluate(3, consumer, reports, env_for(consumer, NVIDIA_API_KEY="test-provider-value-1234")) == 0
    skills = json.loads((reports / "tier3/ci-tier.json").read_text())["tiers"][0]["skills"]
    assert [(s["status"], s["metrics"]["execution_complete"]) for s in skills] == [("failed", False)] * 2
    assert "complete" not in skills[0]
    for number in (1, 2):
        diagnostics = reports / f"tier3/skill-00{number}/diagnostics"
        assert (diagnostics / "command.log").is_file()
        assert (diagnostics / "harbor/run/_harbor-jobs/skill-codex-with/trial/exception.txt").is_file()


def test_tier_diagnostics_budget_is_shared_by_every_skill(consumer, tmp_path, monkeypatch):
    from ci import diagnostics

    def execute(args, *, log, **_kwargs):
        log.write_text("worker startup failed\n")
        return 0 if "catalog" in args else 1

    monkeypatch.setattr(diagnostics, "MAX_TIER_FILES", 1)
    monkeypatch.setattr(run_tiers, "run_command", execute)
    reports = tmp_path / "reports"
    assert run_tiers.evaluate(2, consumer, reports, env_for(consumer, NVIDIA_API_KEY="test-provider-value-1234")) == 0
    first, second = (reports / "tier2/skill-001/diagnostics", reports / "tier2/skill-002/diagnostics")
    assert (first / "command.log").is_file()
    assert not (second / "command.log").exists()
    index = json.loads((second / "index.json").read_text())
    assert index["omitted"] == [{"path": "command.log", "reason": "tier_limit"}]


@pytest.mark.parametrize(
    ("skills_changed", "advisory"), [("false", "skipped"), ("true", "incomplete"), (None, "incomplete")]
)
def test_paid_tiers_skipped_for_no_skill_change_are_reported_as_skipped(consumer, tmp_path, skills_changed, advisory):
    """Only the trusted "false" turns missing Tier 2/3 evidence into a skip; Tier 1 stays required."""
    from ci.run_tier1 import _summary

    env = env_for(consumer)
    if skills_changed is not None:
        env["SKILL_EVAL_SKILLS_CHANGED"] = skills_changed
    reports = tmp_path / "reports"
    (reports / "tier1").mkdir(parents=True)
    legacy = _summary([], None)
    legacy.update(overall_status="skipped", execution={"commit": env["REPORT_COMMIT"]})
    _, legacy["selection"] = run_tiers.select_changed_skills(
        consumer, "skills", env["REPORT_COMMIT"], env["REPORT_COMMIT"]
    )
    (reports / "tier1/summary.json").write_text(json.dumps(legacy))
    assert run_tiers.aggregate(reports, (1, 2, 3), env) == 0
    rows = {row["tier"]: row for row in json.loads((reports / "ci-results.json").read_text())["tiers"]}
    assert rows[2]["status"] == rows[3]["status"] == advisory
    if advisory == "skipped":
        assert "No file under a configured skill root changed" in rows[3]["reason"]
    assert rows[1]["required"] is True


def test_no_skill_change_never_excuses_missing_tier1_evidence(consumer, tmp_path):
    env = env_for(consumer, SKILL_EVAL_SKILLS_CHANGED="false")
    reports = tmp_path / "reports"
    reports.mkdir()
    assert run_tiers.aggregate(reports, (1, 2), env) == 1
    rows = {row["tier"]: row for row in json.loads((reports / "ci-results.json").read_text())["tiers"]}
    assert rows[1]["status"] == "incomplete"
    assert rows[2]["status"] == "skipped"


def _evaluate_tier2_with_unselected_game(consumer, tmp_path, monkeypatch, frontmatter):
    """Run Tier 2 with only alpha selected and an unselected ``skills/game`` carrying *frontmatter*."""
    import contextlib

    from skillevaluator.embedding.registry import EmbeddingRegistry

    game = consumer / "skills" / "game"
    game.mkdir()
    (game / "SKILL.md").write_bytes(f"---\n{frontmatter}\n---\nInstructions.\n".encode())
    git = ["git", "-C", str(consumer), "-c", "user.name=CI tests", "-c", "user.email=ci@example.com"]
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-m", "game"], check=True, capture_output=True)
    # Only alpha changed; the catalog still covers every discovered skill.
    monkeypatch.setattr(run_tiers, "select_skills", lambda workspace, _env: ([workspace / "skills/alpha"], {}))

    def no_embedding(*_args, **_kwargs):
        raise AssertionError("A catalog skill with this frontmatter must be rejected before any embedding request")

    monkeypatch.setattr(EmbeddingRegistry, "_build_entries", no_embedding)
    commands = []

    def execute(args, *, log, **_kwargs):
        # Run the catalog helper in this process, with its stderr in the command log.
        commands.append(args[3])
        with log.open("w", encoding="utf-8") as stream, contextlib.redirect_stderr(stream):
            return run_tiers.main(args[3:])

    monkeypatch.setattr(run_tiers, "run_command", execute)
    reports = tmp_path / "reports"
    env = env_for(consumer, NVIDIA_API_KEY="test-provider-value-1234")
    assert run_tiers.evaluate(2, consumer, reports, env) == 0
    assert commands == ["catalog"]
    tier = json.loads((reports / "tier2/ci-tier.json").read_text(encoding="utf-8"))["tiers"][0]
    assert tier["status"] == "incomplete"
    log = (reports / "tier2/catalog/diagnostics/command.log").read_text(encoding="utf-8")
    return reports, env, tier, log


@pytest.mark.parametrize("value", ["name: 2048", "name: yes", "name: [Draft]"])
def test_skill_without_valid_frontmatter_stops_the_catalog_with_a_frontmatter_reason(
    consumer, tmp_path, monkeypatch, value
):
    from ci.comment_summary import render_comment_summary

    reports, env, tier, log = _evaluate_tier2_with_unselected_game(
        consumer, tmp_path, monkeypatch, f"{value}\ndescription: Play the game."
    )
    reason = "A skill's invalid SKILL.md frontmatter stopped the comparison catalog; its diagnostics name the skill."
    assert tier["reason"] == reason
    assert "non-empty string name and description: skills/game" in log
    assert str(consumer) not in log
    # The feedback comment cuts each cell to a fixed length; the whole reason,
    # with its pointer to the catalog diagnostics, must survive that cut.
    assert run_tiers.aggregate(reports, (2,), env) == 0
    manifest = json.loads((reports / "ci-results.json").read_text(encoding="utf-8"))
    body = render_comment_summary(manifest, "https://github.com/example/consumer/actions/runs/123/artifacts/9")
    alpha = next(line for line in body.splitlines() if line.startswith("| skills/alpha |"))
    assert alpha.split(" | ")[2] == f"INCOMPLETE · issues unknown · {reason}"


@pytest.mark.parametrize(
    "frontmatter",
    [
        pytest.param("name: " + "x" * 16_385 + "\ndescription: Play the game.", id="name-over-length-limit"),
        pytest.param(
            "name: game\ndescription: Play the game.\nextra: " + "[" * 1_500 + "safe" + "]" * 1_500,
            id="frontmatter-over-structure-limit",
        ),
    ],
)
def test_frontmatter_over_a_limit_stops_the_catalog_with_the_provider_reason(
    consumer, tmp_path, monkeypatch, frontmatter
):
    # Only the frontmatter rejection gets its own exit status and reason. Any
    # other catalog failure, including a limit error, keeps the generic ones.
    _reports, _env, tier, log = _evaluate_tier2_with_unselected_game(consumer, tmp_path, monkeypatch, frontmatter)
    assert tier["reason"] == (
        "The complete comparison catalog could not be built; inspect provider and embedding configuration."
    )
    assert log.strip() == "CI execution could not complete safely."
