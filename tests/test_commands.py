# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.tier3.commands import parse_agent_model_overrides, parse_agents

FIXTURE = Path(__file__).parent / "fixtures" / "skills" / "simple"


def test_version_check_is_in_the_default_tier1_lineup() -> None:
    from skillevaluator.tier1.commands import DEFAULT_CHECKS, OPTIONAL_CHECKS, enabled_check_lineup

    assert "version" in DEFAULT_CHECKS
    assert "version" not in OPTIONAL_CHECKS
    assert enabled_check_lineup(None) == list(DEFAULT_CHECKS)
    assert enabled_check_lineup("security,version") == ["version", "security"]


def test_validate_passes_explicit_previous_version(monkeypatch) -> None:
    from skillevaluator import cli as cli_module

    captured: list[str | None] = []

    def _run_validation(_target: Path, **kwargs):
        captured.append(kwargs["previous_version"])
        return []

    monkeypatch.setattr(cli_module, "run_validation", _run_validation)
    monkeypatch.setattr(cli_module, "emit_reports", lambda *_args, **_kwargs: True)

    result = CliRunner().invoke(
        cli,
        [
            "validate",
            str(FIXTURE),
            "--no-dedup",
            "--checks",
            "version",
            "--previous-version",
            "1.2.0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == ["1.2.0"]


def test_claude_alias_is_canonicalized_and_deduplicated() -> None:
    assert parse_agents("claude, claude-code, opencode, claude") == ["claude-code", "opencode"]


def test_claude_alias_model_override_uses_canonical_agent_name() -> None:
    assert parse_agent_model_overrides(("claude=anthropic/claude-sonnet",)) == {
        "claude-code": ["anthropic/claude-sonnet"]
    }


def test_claude_alias_and_canonical_model_overrides_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"claude.*claude-code.*same agent"):
        parse_agent_model_overrides(
            (
                "claude=anthropic/claude-sonnet",
                "claude-code=anthropic/claude-opus",
            )
        )


def test_repeated_canonical_model_overrides_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"claude-code.*claude-code.*one model"):
        parse_agent_model_overrides(
            (
                "claude-code=anthropic/claude-sonnet",
                "claude-code=anthropic/claude-opus",
            )
        )


def test_validate_fixture_no_llm() -> None:
    result = CliRunner().invoke(
        cli, ["validate", str(FIXTURE), "--verbose", "--no-llm", "--no-dedup", "--checks", "schema,quality,lint"]
    )

    assert result.exit_code == 0, result.output
    assert "All validations passed" in result.output


def test_validate_prints_tier1_section_banner() -> None:
    # The Tier 1 section is announced as it runs so it is visibly reported in
    # CI logs (SkillEvaluator parity), not only inside the final combined report.
    result = CliRunner().invoke(
        cli, ["validate", str(FIXTURE), "--verbose", "--no-llm", "--no-dedup", "--checks", "schema,quality,lint"]
    )

    assert result.exit_code == 0, result.output
    assert "Tier 1: Security and Static Validation" in result.output


def test_validate_tier1_banner_is_stable_in_narrow_terminal(monkeypatch) -> None:
    monkeypatch.setenv("COLUMNS", "20")

    result = CliRunner().invoke(
        cli, ["validate", str(FIXTURE), "--verbose", "--no-llm", "--no-dedup", "--checks", "schema,quality,lint"]
    )

    assert result.exit_code == 0, result.output
    assert "Tier 1: Security and Static Validation" in result.output


def test_validate_flushes_tier1_results_before_tier3(monkeypatch) -> None:
    # Tier 1 (and Tier 2) results must reach the terminal BEFORE the
    # long-running Tier 3 agent evaluation, so they remain visible in CI even
    # when Tier 3 is slow or interrupted. Tier 3 is stubbed to keep the test
    # fast and offline.
    from skillevaluator import cli as cli_module
    from skillevaluator.models.result import ValidationResult

    def _stub_agent_eval(*_args, **_kwargs) -> ValidationResult:
        result = ValidationResult(
            validator_name="AGENT_EVAL",
            validator_description="Tier 3: Live Agent Evaluation",
        )
        result.add_warning("stubbed Tier 3")
        # Tier 3 truth remains failed in the report, while validate's exit code
        # is gated only by the Tier 1/Tier 2 snapshot.
        result.passed = False
        return result

    monkeypatch.setattr(cli_module, "_run_agent_eval_or_skip", _stub_agent_eval)

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["validate", str(FIXTURE), "--verbose", "--no-llm", "--no-dedup", "--agent-eval", "--checks", "schema"],
        )

    assert result.exit_code == 0, result.output
    out = result.output
    assert "Tier 1: Security and Static Validation" in out
    assert "Tier 3: Live Agent Evaluation" in out
    # The interim Tier 1 summary table is flushed ahead of the Tier 3 section.
    assert out.index("Validation Results") < out.index("Tier 3: Live Agent Evaluation")


def test_tier1_lint_scripts_fixture() -> None:
    result = CliRunner().invoke(cli, ["tier1", "lint-scripts", str(FIXTURE)])

    assert result.exit_code == 0, result.output
    assert "SCRIPT_LINT" in result.output


def test_create_dataset_dry_run_no_llm() -> None:
    result = CliRunner().invoke(cli, ["create-eval-dataset", str(FIXTURE), "--no-llm", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert '"prompt"' in result.output


def test_init_custom_grader_creates_valid_starter(tmp_path: Path) -> None:
    skill_path = tmp_path / "simple"
    shutil.copytree(FIXTURE, skill_path)

    result = CliRunner().invoke(cli, ["init-custom-grader", str(skill_path)])

    assert result.exit_code == 0, result.output
    assert (skill_path / "evals" / "grader.py").exists()
    assert (skill_path / "evals" / "config.yml").exists()
    assert "mode: default_plus_custom" in (skill_path / "evals" / "config.yml").read_text(encoding="utf-8")


def test_init_harbor_task_creates_valid_contract(tmp_path: Path) -> None:
    skill_path = tmp_path / "simple"
    shutil.copytree(FIXTURE, skill_path)

    result = CliRunner().invoke(cli, ["init-harbor-task", str(skill_path), "--with-config"])

    assert result.exit_code == 0, result.output
    assert (skill_path / "evals" / "harbor" / "case-001" / "task.toml").exists()

    validate = CliRunner().invoke(cli, ["tier3", "validate", str(skill_path), "--harbor-contract"])

    assert validate.exit_code == 0, validate.output
    assert "all checks passed" in validate.output


def _plain_text(text: str) -> str:
    """Strip ANSI escape codes (CI may set FORCE_COLOR, making rich emit them)."""
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_validate_quiet_default_renders_pipeline_view() -> None:
    # Without --verbose the compact pipeline view replaces the banner/table
    # stream: tier sections, a verdict panel, and report pointers.
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli, ["validate", str(FIXTURE), "--no-llm", "--no-tier2", "--tier1-checks", "schema,quality"]
        )

    out = _plain_text(result.output)
    assert result.exit_code == 0, result.output
    assert "Tier 1 · Static & Security" in out
    assert "skill: simple" in out
    assert "✓ schema" in out  # the per-check ticker persists with final states
    assert "PASS" in out
    assert ".html" in out and ".json" in out
    # Verbose-stream furniture must not leak into the quiet view.
    assert "Tier 1: Security and Static Validation" not in out
    assert "Validation Results" not in out


def test_validate_quiet_always_writes_html_and_json_reports() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["validate", str(FIXTURE), "--no-llm", "--no-dedup", "--checks", "schema"])
        assert result.exit_code == 0, result.output
        reports = list(Path("reports").iterdir())
        assert any(p.suffix == ".html" for p in reports), reports
        assert any(p.suffix == ".json" for p in reports), reports


def test_validate_tier_aliases_and_selector(monkeypatch) -> None:
    from skillevaluator import cli as cli_module
    from skillevaluator.models.result import ValidationResult

    calls: dict[str, bool] = {}

    def _tier1(*_args, **_kwargs) -> list[ValidationResult]:
        result = ValidationResult(validator_name="SCHEMA")
        result.add_success("schema", "ok")
        return [result]

    def _tier2(*_args, **_kwargs) -> list[ValidationResult]:
        calls["tier2"] = True
        result = ValidationResult(validator_name="Tier 2 Deduplication")
        result.add_success("dedup", "ok")
        return [result]

    def _tier3(*_args, **_kwargs) -> ValidationResult:
        calls["tier3"] = True
        result = ValidationResult(validator_name="AGENT_EVAL")
        result.add_success("agent_eval", "ok")
        return result

    monkeypatch.setattr(cli_module, "run_validation", _tier1)
    monkeypatch.setattr(cli_module, "_run_dedup_or_skip", _tier2)
    monkeypatch.setattr(cli_module, "_run_agent_eval_or_skip", _tier3)

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["validate", str(FIXTURE), "--no-llm", "--tiers", "1,3", "--checks", "schema"])
    assert result.exit_code == 0, result.output
    assert calls == {"tier3": True}

    calls.clear()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli, ["validate", str(FIXTURE), "--no-llm", "--no-tier2", "--tier3", "--checks", "schema"]
        )
    assert result.exit_code == 0, result.output
    assert calls == {"tier3": True}

    result = CliRunner().invoke(cli, ["validate", str(FIXTURE), "--tiers", "2,3"])
    assert result.exit_code != 0
    assert "Tier 1 always runs" in result.output
    result = CliRunner().invoke(cli, ["validate", str(FIXTURE), "--tiers", "1,9"])
    assert result.exit_code != 0


def test_validate_full_runs_autopilot_dataset(monkeypatch) -> None:
    from skillevaluator import cli as cli_module
    from skillevaluator.models.result import ValidationResult

    generated: list[Path] = []
    monkeypatch.setattr(
        cli_module, "_ensure_autopilot_dataset", lambda path, **_kwargs: generated.append(path) or "auto-generated"
    )

    def _tier3(*_args, **_kwargs) -> ValidationResult:
        result = ValidationResult(validator_name="AGENT_EVAL")
        result.add_success("agent_eval", "ok")
        return result

    monkeypatch.setattr(cli_module, "_run_agent_eval_or_skip", _tier3)

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli, ["validate", str(FIXTURE), "--no-llm", "--no-tier2", "--full", "--checks", "schema"]
        )
    assert result.exit_code == 0, result.output
    assert generated, "--full must run the autopilot dataset flow"


def test_validate_catalog_runs_each_skill_as_separate_job() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        catalog = Path("catalog")
        for name in ("simple", "simple2"):
            shutil.copytree(FIXTURE, catalog / name)
        # keep frontmatter name matching the directory for the copy
        second = catalog / "simple2" / "SKILL.md"
        second.write_text(second.read_text(encoding="utf-8").replace("name: simple", "name: simple2"), encoding="utf-8")
        result = runner.invoke(
            cli,
            ["validate", str(catalog.resolve()), "--no-llm", "--no-dedup", "--checks", "quality", "-o", "out"],
        )

        out = _plain_text(result.output)
        assert "skill 1/2 · simple" in out
        assert "skill 2/2 · simple2" in out
        assert "Catalog Result" in out
        assert "all 2 skills passed" in out
        assert result.exit_code == 0, result.output
        assert any(Path("out/simple").glob("*.html"))
        assert any(Path("out/simple2").glob("*.html"))


def test_validate_catalog_rejects_one_previous_version_for_every_skill() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        catalog = Path("catalog")
        for name in ("simple", "simple2"):
            shutil.copytree(FIXTURE, catalog / name)
        second = catalog / "simple2" / "SKILL.md"
        second.write_text(second.read_text(encoding="utf-8").replace("name: simple", "name: simple2"), encoding="utf-8")

        result = runner.invoke(
            cli,
            [
                "validate",
                str(catalog.resolve()),
                "--no-llm",
                "--no-dedup",
                "--checks",
                "version",
                "--previous-version",
                "1.2.0",
            ],
        )

        assert result.exit_code != 0
        assert "cannot be reused for a catalog" in result.output
        assert not Path("skillevaluator-results").exists()


def test_validate_quiet_failing_run_renders_verdict_and_fails_cleanly(monkeypatch) -> None:
    # Regression: a failing DEFAULT (quiet) run must render the verdict panel
    # and exit via the machine-readable ClickException — not an AttributeError.
    from skillevaluator import cli as cli_module
    from skillevaluator.models.result import Finding, Severity, ValidationResult

    failing = ValidationResult(validator_name="Schema & Repository Governance")
    failing.add_structured_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.HIGH,
            check_name="author_missing",
            message="Author not specified in metadata",
            file_path="SKILL.md",
            suggestion="Add 'metadata.author' with format 'Name <email@example.com>'",
        ),
        is_error=True,
    )
    monkeypatch.setattr(cli_module, "run_validation", lambda *_args, **_kwargs: [failing])

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["validate", str(FIXTURE), "--no-llm", "--no-tier2", "--checks", "schema"])

    out = _plain_text(result.output)
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit), result.output
    assert "FAIL" in out
    assert "Schema & Repository Governance failed" in out
    assert "validation failed" in out  # machine-readable line preserved


def test_validate_quiet_tier3_execution_errors_do_not_render_green(monkeypatch) -> None:
    # Regression: a Tier 3 result that ran but reported execution errors must
    # not render as a green pass or claim "all tiers passed" (it stays
    # advisory: the exit code is unaffected).
    from skillevaluator import cli as cli_module
    from skillevaluator.models.result import ValidationResult

    def _tier3(*_args, **_kwargs) -> ValidationResult:
        result = ValidationResult(validator_name="AGENT_EVAL")
        result.add_error("2 of 2 trials crashed")
        result.metadata["agent_eval"] = {
            "execution_status": "failed",
            "execution_errors": ["2 of 2 trials crashed"],
            "summary": {
                "schema_version": "2.0",
                "execution_status": "failed",
                "agents_run": ["codex"],
                "verdict": "fail",
                "skill_name": "simple",
                "best_agent": "codex",
                "overall_score": 0.0,
                "overall_lift": 0.0,
                "environment": "docker",
                "runtime_seconds": 1.0,
            },
            "agents_run": ["codex"],
            "verdict": "fail",
            "overall_score": 0.0,
            "overall_lift": 0.0,
            "agents": {},
            "cases": [],
            "recommendations": [],
            "conclusions": [],
            "insights": [],
            "dimensions": [],
            "evaluators": {},
        }
        return result

    monkeypatch.setattr(cli_module, "_run_agent_eval_or_skip", _tier3)

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli, ["validate", str(FIXTURE), "--no-llm", "--no-tier2", "--tier3", "--checks", "schema"]
        )

    out = _plain_text(result.output)
    assert result.exit_code == 0, result.output  # advisory: gate unaffected
    assert "✗ fail" in out
    assert "2 of 2 trials crashed" in out
    assert "all 2 tiers passed" not in out
    assert "advisory" in out


def test_validate_autopilot_generation_failure_degrades_to_skip(monkeypatch) -> None:
    # Regression: a dataset-generation failure must not abort validate after
    # Tier 1/2 already ran — Tier 3 skips with the reason and the reports and
    # verdict still render (Tier 3 is advisory).
    import click

    from skillevaluator import cli as cli_module
    from skillevaluator.models.result import ValidationResult

    def _boom(*_args, **_kwargs) -> str:
        raise click.ClickException("no provider key")

    def _tier3(*_args, **_kwargs) -> ValidationResult:
        result = ValidationResult(validator_name="AGENT_EVAL")
        result.add_warning("Skipped: evals.json missing")
        result.metadata.update({"execution_status": "skipped", "skip_reason": "evals.json missing"})
        return result

    monkeypatch.setattr(cli_module, "_ensure_autopilot_dataset", _boom)
    monkeypatch.setattr(cli_module, "_run_agent_eval_or_skip", _tier3)

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["validate", str(FIXTURE), "--no-llm", "--no-tier2", "--autopilot", "--checks", "schema", "-o", "out"],
        )

        out = _plain_text(result.output)
        assert result.exit_code == 0, result.output
        assert "autopilot dataset generation failed: no provider key" in out
        # The whole point: Tier 1 reports survive the Tier 3 dataset failure.
        assert any(Path("out").glob("*.html")), "reports must still be written"


def test_partial_agent_eval_result_carries_engine_errors(monkeypatch, tmp_path) -> None:
    # A run that produced usable per-agent results alongside recorded errors
    # is a ran-with-errors advisory, not a skip: errors carried (deduplicated),
    # passed False, never skipped.
    from skillevaluator import cli as cli_module
    from skillevaluator.evaluation import tier3_report
    from skillevaluator.models.result import ValidationResult
    from skillevaluator.tier3 import results_location

    run_dir = tmp_path / "results" / "20260709_000000"
    run_dir.mkdir(parents=True)

    def _normalized(*_args, **_kwargs) -> ValidationResult:
        result = ValidationResult(validator_name="AGENT_EVAL")
        result.add_success("agent_eval", "ok")
        return result

    monkeypatch.setattr(tier3_report, "agent_eval_result_from_run", _normalized)
    monkeypatch.setattr(results_location, "resolve_latest_results", lambda *_a, **_k: run_dir)

    engine_result = {
        "agents": {"codex": {}},
        "run_dir": str(run_dir),
        "execution_errors": ["agent codex crashed", "agent codex crashed"],
    }
    result = cli_module._partial_agent_eval_result(
        tmp_path,
        engine_result=engine_result,
        failure="agent codex crashed",
        results_dir=None,
        env_mode="docker",
    )
    assert result is not None
    assert result.passed is False
    assert result.errors == ["agent codex crashed"]


def test_partial_agent_eval_result_rejects_stale_results(monkeypatch, tmp_path) -> None:
    # The normalizer reads the skill's ``latest`` results; a mismatch with the
    # engine's run_dir means an OLDER run would be reported as fresh output —
    # degrade to the skip path instead.
    from skillevaluator import cli as cli_module
    from skillevaluator.tier3 import results_location

    run_dir = tmp_path / "results" / "fresh"
    run_dir.mkdir(parents=True)
    stale = tmp_path / "results" / "stale"
    stale.mkdir()
    monkeypatch.setattr(results_location, "resolve_latest_results", lambda *_a, **_k: stale)

    engine_result = {"agents": {"codex": {}}, "run_dir": str(run_dir), "execution_errors": ["boom"]}
    partial = cli_module._partial_agent_eval_result(
        tmp_path, engine_result=engine_result, failure="boom", results_dir=None, env_mode="docker"
    )
    assert partial is None


def test_run_agent_eval_partial_run_returns_ran_with_errors(monkeypatch, tmp_path) -> None:
    # End-to-end through _run_agent_eval_or_skip: an engine result with usable
    # agents data AND execution errors no longer collapses into a skip (which
    # discarded real results) — it returns the ran-with-errors advisory shape
    # the pipeline view renders red.
    from skillevaluator import cli as cli_module
    from skillevaluator.evaluation import tier3_report
    from skillevaluator.evaluation.service import EvaluationService
    from skillevaluator.models.result import ValidationResult
    from skillevaluator.tier3 import results_location

    run_dir = tmp_path / "results" / "20260709_010101"
    run_dir.mkdir(parents=True)
    engine = {
        "agents": {"codex": {"with_skill": {}}},
        "run_dir": str(run_dir),
        "execution_status": "failed",
        "execution_errors": ["1 of 2 agents crashed"],
        "error": ["1 of 2 agents crashed"],
    }

    def _normalized(*_args, **_kwargs) -> ValidationResult:
        result = ValidationResult(validator_name="AGENT_EVAL")
        result.add_success("agent_eval", "ok")
        return result

    monkeypatch.setattr(EvaluationService, "evaluate", lambda _self, _options, **_k: engine)
    monkeypatch.setattr(tier3_report, "agent_eval_result_from_run", _normalized)
    monkeypatch.setattr(results_location, "resolve_latest_results", lambda *_a, **_k: run_dir)

    result = cli_module._run_agent_eval_or_skip(
        tmp_path,
        agents="codex",
        env_mode="docker",
        skip_baseline=False,
        n_concurrent=None,
        max_agents=None,
        validate_source=False,
    )
    assert result.passed is False
    assert "1 of 2 agents crashed" in result.errors
    assert (result.metadata or {}).get("execution_status") != "skipped"


def test_validate_tier2_default_is_blocking_and_can_be_advisory(monkeypatch) -> None:
    from skillevaluator import cli as cli_module
    from skillevaluator.models.result import ValidationResult

    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "ok")

    def _tier2(*_args, **_kwargs) -> list[ValidationResult]:
        result = ValidationResult(validator_name="Context Deduplication")
        result.add_error("dedup failed")
        return [result]

    monkeypatch.setattr(cli_module, "run_validation", lambda *_args, **_kwargs: [tier1])
    monkeypatch.setattr(cli_module, "_run_dedup_or_skip", _tier2)

    runner = CliRunner()
    with runner.isolated_filesystem():
        blocking = runner.invoke(cli, ["validate", str(FIXTURE), "--checks", "schema"])
        advisory = runner.invoke(
            cli,
            ["validate", str(FIXTURE), "--checks", "schema", "--no-block-on-dedup"],
        )

    assert blocking.exit_code != 0
    assert "failed in Tier 2" in _plain_text(blocking.output)
    assert advisory.exit_code == 0, advisory.output


def test_validate_missing_tier3_source_respects_blocking_flag(monkeypatch, tmp_path: Path) -> None:
    from skillevaluator import cli as cli_module
    from skillevaluator.evaluation import EvaluationService
    from skillevaluator.models.result import ValidationResult

    skill = tmp_path / "source-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: source-skill\ndescription: test\n---\n")
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "ok")
    monkeypatch.setattr(cli_module, "run_validation", lambda *_args, **_kwargs: [tier1])
    monkeypatch.setattr(
        EvaluationService,
        "evaluate",
        lambda *_args, **_kwargs: pytest.fail("source preflight must run before live evaluation"),
    )

    runner = CliRunner()
    advisory = runner.invoke(
        cli,
        [
            "validate",
            str(skill),
            "--no-tier2",
            "--tier3",
            "--verbose",
            "--report",
            "json",
            "--output-dir",
            str(tmp_path / "advisory"),
        ],
    )
    blocking = runner.invoke(
        cli,
        [
            "validate",
            str(skill),
            "--no-tier2",
            "--tier3",
            "--block-on-agent-eval",
            "--verbose",
            "--report",
            "json",
            "--output-dir",
            str(tmp_path / "blocking"),
        ],
    )

    assert advisory.exit_code == 0, advisory.output
    assert blocking.exit_code != 0

    advisory_report = json.loads(next((tmp_path / "advisory").glob("*.json")).read_text())
    blocking_report = json.loads(next((tmp_path / "blocking").glob("*.json")).read_text())
    assert advisory_report["tier3_applicability"]["applicability"] == "not_required"
    assert advisory_report["tier3"]["verdict"] == "neutral"
    assert advisory_report["gating"]["tiers"]["3"]["blocking"] is False
    assert advisory_report["overall_status"] == "passed"
    assert blocking_report["tier3_applicability"]["applicability"] == "required"
    assert blocking_report["tier3"]["verdict"] == "fail"
    assert blocking_report["gating"]["tiers"]["3"]["blocking"] is True
    assert blocking_report["overall_status"] == "failed"


def test_validate_quiet_honors_explicit_report_formats() -> None:
    # An explicit -r is a contract: "-r cli" renders the full Rich report
    # below the pipeline view (and writes no unrequested files); "-r json"
    # writes exactly json. Only the DEFAULT quiet run swaps in html+json.
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["validate", str(FIXTURE), "--no-llm", "--no-dedup", "--checks", "schema", "-r", "cli", "-o", "out"],
        )
        assert result.exit_code == 0, result.output
        assert "Validation Results" in _plain_text(result.output)
        assert not list(Path("out").glob("*.html"))
        assert not list(Path("out").glob("*.json"))

    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["validate", str(FIXTURE), "--no-llm", "--no-dedup", "--checks", "schema", "-r", "json", "-o", "out"],
        )
        assert result.exit_code == 0, result.output
        assert list(Path("out").glob("*.json"))
        assert not list(Path("out").glob("*.html"))


def test_validate_catalog_survives_failing_skills(monkeypatch) -> None:
    # Regression: one failing skill must not abort the catalog — every skill
    # runs and the scoreboard prints.
    from skillevaluator import cli as cli_module
    from skillevaluator.models.result import Finding, Severity, ValidationResult

    def _failing_tier1(*_args, **_kwargs) -> list[ValidationResult]:
        failing = ValidationResult(validator_name="SCHEMA")
        failing.add_structured_finding(
            Finding(
                category="SCHEMA",
                severity=Severity.HIGH,
                check_name="x",
                message="boom",
                file_path="SKILL.md",
            ),
            is_error=True,
        )
        return [failing]

    monkeypatch.setattr(cli_module, "run_validation", _failing_tier1)

    runner = CliRunner()
    with runner.isolated_filesystem():
        catalog = Path("catalog")
        for name in ("simple", "simple2"):
            shutil.copytree(FIXTURE, catalog / name)
        result = runner.invoke(
            cli, ["validate", str(catalog.resolve()), "--no-llm", "--no-dedup", "--checks", "schema", "-o", "out"]
        )

    out = _plain_text(result.output)
    assert "skill 1/2" in out and "skill 2/2" in out
    assert "Catalog Result" in out
    assert "0/2 skills passed" in out
    assert result.exit_code != 0


def test_render_evaluation_result_invokes_findings_report(monkeypatch) -> None:
    # Regression: the per-evaluator findings panel (the feedback surface) must
    # be invoked after the score tables — the renderer existed but had no
    # caller. Its content is loaded from reward files in the run dir, so the
    # wiring (arguments included) is what this test pins.
    import io

    from rich.console import Console

    from skillevaluator.tier3.harbor import report as harbor_report
    from skillevaluator.tier3.result_display import render_evaluation_result

    calls: list[tuple] = []
    monkeypatch.setattr(harbor_report, "display_findings_report", lambda *args, **_kwargs: calls.append(args))

    result = {
        "skill_name": "simple",
        "run_dir": "/tmp/run",
        "execution_status": "succeeded",
        "agents": {
            "codex": {
                "execution_status": "succeeded",
                "num_trials_with": 1,
                "with_skill": {"security": 1.0},
                "without_skill": {"security": 1.0},
                "lift": {"overall": {"with_skill": 1.0, "without_skill": 0.5, "delta": 0.5}},
            }
        },
    }
    render_evaluation_result(result, console=Console(file=io.StringIO(), width=120))
    assert calls, "findings report must be invoked after the score tables"
    _harbor_result, skill_name, agent_list, results_dir = calls[0]
    assert skill_name == "simple"
    assert agent_list == ["codex"]
    assert str(results_dir) == "/tmp/run"


def test_summarize_tier2_lists_duplicate_details() -> None:
    # "duplicates found" alone is useless — the section must say WHAT
    # duplicated, up to three findings plus an overflow pointer.
    from skillevaluator.models.result import Finding, Severity, ValidationResult
    from skillevaluator.reporting.console_ui import summarize_tier2

    failing = ValidationResult(validator_name="Context Deduplication", passed=False)
    for index in range(5):
        failing.add_finding(
            Finding(
                category="CONTENT_DEDUP",
                severity=Severity.HIGH,
                check_name="context_dedup",
                message=f"Sections 'Setup {index}' and 'Install {index}' repeat the same guidance",
                file_path="SKILL.md",
            )
        )

    ran, ok, rows, _skip = summarize_tier2([failing])
    assert ran and not ok
    rendered = ["".join(chunk for chunk, _style in row.segments) for row in rows]
    assert any("duplicates found" in line for line in rendered)
    assert any("Sections 'Setup 0'" in line for line in rendered)
    duplicate_rows = [row for row in rows if row.label == "duplicate"]
    assert len(duplicate_rows) == 3
    assert any("2 more" in line for line in rendered)


def test_rerun_hint_preserves_run_shaping_flags(monkeypatch) -> None:
    # The FAIL panel's rerun line must echo the user's actual invocation:
    # dropping --min-score (etc.) means following the hint can silently
    # "pass" the failure away.
    from skillevaluator import cli as cli_module

    monkeypatch.setattr("sys.argv", ["/x/bin/skillevaluator", "validate", "./skill", "--min-score", "95"])
    assert cli_module._rerun_hint(Path("./skill"), agent_eval=False) == (
        "skillevaluator validate ./skill --min-score 95"
    )

    # Outside a CLI launch (tests, API embedding) fall back to the bare form.
    monkeypatch.setattr("sys.argv", ["pytest"])
    assert cli_module._rerun_hint(Path("./skill"), agent_eval=True).endswith("--tier3")


def test_validate_tiers_selector_is_authoritative_over_full(monkeypatch) -> None:
    # --full turns Tier 3 (and autopilot) on, but an explicit --tiers 1,2 must
    # turn both back off — the selector wins in both directions.
    from skillevaluator import cli as cli_module
    from skillevaluator.models.result import ValidationResult

    calls: dict[str, bool] = {}

    def _tier2(*_args, **_kwargs) -> list[ValidationResult]:
        result = ValidationResult(validator_name="Tier 2 Deduplication")
        result.add_success("dedup", "ok")
        return [result]

    def _tier3(*_args, **_kwargs) -> ValidationResult:
        calls["tier3"] = True
        result = ValidationResult(validator_name="AGENT_EVAL")
        result.add_success("agent_eval", "ok")
        return result

    def _generate(*_args, **_kwargs) -> str:
        calls["autopilot"] = True
        return "generated"

    monkeypatch.setattr(cli_module, "_run_dedup_or_skip", _tier2)
    monkeypatch.setattr(cli_module, "_run_agent_eval_or_skip", _tier3)
    monkeypatch.setattr(cli_module, "_ensure_autopilot_dataset", _generate)

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli, ["validate", str(FIXTURE), "--no-llm", "--full", "--tiers", "1,2", "--checks", "schema"]
        )

    assert result.exit_code == 0, result.output
    assert calls == {}, f"Tier 3/autopilot must not run under --tiers 1,2: {calls}"


def test_validate_catalog_reports_unexpected_errors(monkeypatch) -> None:
    # A crash inside one skill's pipeline must not abort the catalog silently:
    # the loop finishes, the scoreboard prints, and the reason is named.
    from skillevaluator import cli as cli_module

    def _boom(*_args, **_kwargs):
        raise RuntimeError("validator exploded")

    monkeypatch.setattr(cli_module, "run_validation", _boom)

    runner = CliRunner()
    with runner.isolated_filesystem():
        catalog = Path("catalog")
        for name in ("simple", "simple2"):
            shutil.copytree(FIXTURE, catalog / name)
        result = runner.invoke(
            cli, ["validate", str(catalog.resolve()), "--no-llm", "--no-dedup", "--checks", "schema", "-o", "out"]
        )

    out = _plain_text(result.output)
    assert "Catalog Result" in out
    assert "unexpected error: validator exploded" in out
    assert result.exit_code != 0


def test_summarize_tier2_empty_results_name_a_reason() -> None:
    from skillevaluator.reporting.console_ui import summarize_tier2

    ran, ok, _rows, skip = summarize_tier2([])
    assert not ran and ok
    assert skip, "an empty Tier 2 result set must still explain the skip"


def test_tier3_evaluate_visible_copy_owns_its_params() -> None:
    # The visible tier3 registration is a copy of the hidden top-level
    # command; sharing the mutable params list would let registration on one
    # silently mutate the other.
    from skillevaluator import cli as cli_module

    assert cli_module._tier3_evaluate_visible.params is not cli_module.evaluate.params
    assert [p.name for p in cli_module._tier3_evaluate_visible.params] == [p.name for p in cli_module.evaluate.params]


def test_compact_feedback_panel_filters_messages_rendered_in_detailed_panel(monkeypatch) -> None:
    # Regression (live duplicate-feedback bug): the detailed per-evaluator
    # findings report and the compact payload panel both rendered the same
    # suggestions after the score tables. The compact panel now filters items
    # that the detailed report says it rendered.
    import io

    from rich.console import Console

    from skillevaluator.tier3.harbor import report as harbor_report
    from skillevaluator.tier3.result_display import render_evaluation_result

    result = {
        "skill_name": "simple",
        "run_dir": "/tmp/run",
        "execution_status": "succeeded",
        "agents": {
            "codex": {
                "execution_status": "succeeded",
                "num_trials_with": 1,
                "with_skill": {"security": 1.0},
                "without_skill": {"security": 1.0},
                "lift": {"overall": {"with_skill": 1.0, "without_skill": 0.5, "delta": 0.5}},
            }
        },
        "tier3_feedback": {"recommendations": [{"suggestion": "Add more eval cases"}]},
    }

    monkeypatch.setattr(harbor_report, "display_findings_report", lambda *_a, **_k: {"Add more eval cases"})
    stream = io.StringIO()
    render_evaluation_result(result, console=Console(file=stream, width=200))
    assert "Feedback & Suggestions" not in stream.getvalue()

    monkeypatch.setattr(harbor_report, "display_findings_report", lambda *_a, **_k: set())
    stream = io.StringIO()
    render_evaluation_result(result, console=Console(file=stream, width=200))
    assert "Feedback & Suggestions" in stream.getvalue()
    assert "Add more eval cases" in stream.getvalue()


def test_display_findings_report_reports_whether_it_rendered(tmp_path) -> None:
    # The dedup gating relies on the return value: no reward files on disk
    # means nothing rendered and the caller must fall back.
    from skillevaluator.tier3.harbor.report import display_findings_report

    rendered = display_findings_report(
        {"agents": {"codex": {}}},
        "simple",
        ["codex"],
        tmp_path,
    )
    assert rendered == set()


def test_summarize_tier2_labels_scan_errors_as_errors_not_duplicates() -> None:
    """A failed scan with no findings must not claim duplicates were found."""
    from skillevaluator.models.result import ValidationResult
    from skillevaluator.reporting.console_ui import summarize_tier2

    crashed = ValidationResult(validator_name="Tier 2 Deduplication")
    crashed.add_error("Skill tree contains more than 4096 paths.")

    ran, ok, rows, _skip = summarize_tier2([crashed])

    assert ran and not ok
    rendered = [(row.label, "".join(chunk for chunk, _style in row.segments)) for row in rows]
    labels = [label for label, _text in rendered]
    texts = " | ".join(text for _label, text in rendered)
    assert "scan failed" in texts
    assert "duplicates found" not in texts
    assert "error" in labels and "duplicate" not in labels
    assert "more than 4096 paths" in texts


@pytest.mark.parametrize("check_name", ["path_count_limit", "chunk_count_limit", "embedding_error"])
def test_summarize_tier2_labels_structured_scan_failures_as_errors(check_name: str) -> None:
    from skillevaluator.models.result import Finding, Severity, ValidationResult
    from skillevaluator.reporting.console_ui import summarize_tier2

    failed_scan = ValidationResult(validator_name="Tier 2 Deduplication")
    failed_scan.add_finding(
        Finding(
            category="CONTENT_DEDUP",
            severity=Severity.CRITICAL,
            check_name=check_name,
            message="Tier 2 stopped before duplicate analysis completed.",
            file_path="SKILL.md",
        )
    )

    ran, ok, rows, _skip = summarize_tier2([failed_scan])

    assert ran and not ok
    rendered = [(row.label, "".join(chunk for chunk, _style in row.segments)) for row in rows]
    labels = [label for label, _text in rendered]
    texts = " | ".join(text for _label, text in rendered)
    assert "scan failed" in texts
    assert "duplicates found" not in texts
    assert "error" in labels and "duplicate" not in labels
