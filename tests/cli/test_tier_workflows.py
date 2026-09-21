# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public behavior of direct tier workflows and their expert commands."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.cli_help import render_help
from skillevaluator.models.result import ValidationResult


@pytest.fixture(autouse=True)
def clean_provider_environment(monkeypatch):
    import os

    for name in os.environ:
        if name.startswith(("SKILL_EVAL_", "SKILLSPECTOR_")) or name in {
            "OPENAI_API_KEY",
            "NVIDIA_API_KEY",
            "ANTHROPIC_API_KEY",
            "LLM_JUDGE_MODEL",
        }:
            monkeypatch.delenv(name)


@pytest.fixture
def skill(tmp_path):
    path = tmp_path / "sample"
    path.mkdir()
    (path / "SKILL.md").write_text(
        "---\nname: sample\ndescription: Use when asked to summarize a local text file.\n"
        "metadata:\n  author: Example Author <author@example.com>\n---\n"
        "# Summarize text\n\nRead the supplied file and return a concise summary.\n",
    )
    return path


def _result(name):
    result = ValidationResult(validator_name=name)
    result.add_success("checked", "Completed this check")
    return [result]


def _report(output_dir):
    files = list(output_dir.glob("*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text())


@pytest.mark.parametrize("tier", ["tier1", "tier2", "tier3"])
def test_bare_tier_displays_help_without_running(tier):
    result = CliRunner().invoke(cli, [tier])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output
    assert "PATH" in result.output
    assert "Commands:" in result.output


@pytest.mark.parametrize("tier", ["tier1", "tier2", "tier3"])
def test_unknown_subcommand_and_removed_run_are_errors(tier):
    for name in ("misspelled-command", "run"):
        result = CliRunner().invoke(cli, [tier, name])
        assert result.exit_code == 2
        assert "No such command" in result.output


def test_direct_tier1_runs_only_requested_static_checks(skill, tmp_path):
    output = tmp_path / "reports"
    result = CliRunner().invoke(
        cli,
        ["tier1", str(skill), "--checks", "schema", "--no-llm", "-r", "json", "-o", str(output)],
    )
    assert result.exit_code == 0, result.output
    payload = _report(output)
    assert len(payload["results"]) == 1
    assert "Schema" in payload["results"][0]["validator"]
    assert payload["results"][0]["workflow"]["tier"] == 1


def test_direct_tier1_options_before_path_and_failed_findings(skill, tmp_path):
    (skill / "SKILL.md").write_text("Missing required frontmatter\n")
    result = CliRunner().invoke(
        cli,
        ["tier1", "--checks", "schema", "--no-llm", str(skill), "-r", "json", "-o", str(tmp_path / "reports")],
    )
    assert result.exit_code == 1, result.output
    assert _report(tmp_path / "reports")["results"][0]["passed"] is False


def test_default_tier1_includes_dependency_and_configured_llm_stages(skill, monkeypatch):
    from skillevaluator.tier1 import commands

    calls = []

    def validation(path, **kwargs):
        calls.append(("static", path, kwargs))
        return _result("Schema Validation")

    def rubric(path, **kwargs):
        calls.append(("rubric", path, kwargs))
        return _result("Rubric Evaluation")

    monkeypatch.setattr(commands, "run_validation", validation)
    monkeypatch.setattr(commands, "run_rubric_eval", rubric)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-credential")
    result = CliRunner().invoke(cli, ["tier1", str(skill)])
    assert result.exit_code == 0, result.output
    assert [call[0] for call in calls] == ["static", "rubric"]
    assert "dependency" in calls[0][2]["checks"].split(",")
    assert calls[0][2]["use_llm"] is True
    assert calls[0][2]["llm_verify"] is True


def test_keyless_tier1_reports_unrun_llm_scope(skill, tmp_path):
    result = CliRunner().invoke(
        cli,
        ["tier1", str(skill), "--checks", "schema", "-r", "json", "-o", str(tmp_path / "reports")],
    )
    assert result.exit_code == 0, result.output
    workflow = _report(tmp_path / "reports")["results"][0]["workflow"]
    assert workflow["stages"]["llm"]["status"] == "skipped"
    assert "LLM" in result.output and "skipped" in result.output


@pytest.mark.parametrize("tier,options", [("tier1", ["--llm"]), ("tier2", [])])
def test_required_provider_failure_is_a_configuration_error(tier, options, skill, tmp_path):
    result = CliRunner().invoke(cli, [tier, str(skill), *options, "-o", str(tmp_path / "reports")])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "provider" in result.output.lower()
    assert "\n\nFor NVIDIA Build, set:\n  export " in result.stderr
    assert ("--no-llm" in result.stderr) is (tier == "tier1")
    assert "No such command" not in result.output
    assert not (tmp_path / "reports").exists()


def test_tier2_combines_intra_and_catalog_comparison_once(skill, tmp_path, monkeypatch):
    from skillevaluator.embedding.client import EmbeddingClient
    from skillevaluator.embedding.registry import EmbeddingRegistry
    from skillevaluator.tier2 import commands

    calls = []
    catalog = tmp_path / "catalog.json"

    def intra(path, **kwargs):
        calls.append(("intra", kwargs))
        return _result("Context Deduplication")

    def inter(path, **kwargs):
        calls.append(("inter", kwargs))
        return _result("Similarity Check")

    monkeypatch.setattr(commands, "run_context_optimization_check", intra)
    monkeypatch.setattr(commands, "run_similarity_check", inter)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-credential")
    client = EmbeddingClient()
    monkeypatch.setattr(client, "embed", lambda _texts: [[1.0, 0.0]])
    registry = EmbeddingRegistry(client)
    registry.build_from_directory(skill.parent, "skill")
    registry.save_catalog(catalog)
    result = CliRunner().invoke(cli, ["tier2", str(skill), "--catalog", str(catalog)])
    assert result.exit_code == 0, result.output
    assert [call[0] for call in calls] == ["intra", "inter"]
    assert calls[1][1]["catalog"] == catalog


def test_tier2_invalid_catalog_fails_before_model_work(skill, tmp_path, monkeypatch):
    from skillevaluator.tier2 import commands

    calls = []
    monkeypatch.setattr(commands, "run_context_optimization_check", lambda *_a, **_kw: calls.append("intra") or [])
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-credential")
    catalog = tmp_path / "bad.json"
    catalog.write_text("[]")
    result = CliRunner().invoke(cli, ["tier2", str(skill), "--catalog", str(catalog)])
    assert result.exit_code != 0
    assert not calls
    assert "Catalog" in result.output


def test_tier1_static_failure_does_not_become_a_rubric_failure(skill, tmp_path, monkeypatch):
    from skillevaluator.tier1 import commands

    failure = ValidationResult(validator_name="Schema Validation")
    failure.add_error("Invalid schema")
    monkeypatch.setattr(commands, "run_validation", lambda *_a, **_kw: [failure])
    monkeypatch.setattr(commands, "run_rubric_eval", lambda *_a, **_kw: _result("Rubric Evaluation"))
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-credential")
    result = CliRunner().invoke(cli, ["tier1", str(skill), "-r", "json", "-o", str(tmp_path / "reports")])
    assert result.exit_code == 1
    stages = _report(tmp_path / "reports")["workflow"]["stages"]
    assert stages["validation"]["status"] == "failed"
    assert stages["rubric"]["status"] == "passed"
    assert stages["llm"]["status"] == "enabled"


def test_tier2_without_catalog_does_not_manufacture_a_similarity_pass(skill, tmp_path, monkeypatch):
    from skillevaluator.tier2 import commands

    monkeypatch.setattr(commands, "run_context_optimization_check", lambda *_a, **_kw: _result("Context Deduplication"))
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-credential")
    result = CliRunner().invoke(
        cli,
        ["tier2", str(skill), "-r", "json", "-o", str(tmp_path / "reports")],
    )
    assert result.exit_code == 0, result.output
    payload = _report(tmp_path / "reports")
    assert len(payload["results"]) == 1
    assert payload["results"][0]["workflow"]["stages"]["catalog_comparison"]["status"] == "skipped"
    assert "--catalog" in result.output


def test_tier_name_collision_requires_explicit_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "quality-check"
    path.mkdir()
    (path / "SKILL.md").write_text(
        "---\nname: quality-check\ndescription: Use when asked to check quality.\n"
        "metadata:\n  author: Example Author <author@example.com>\n---\n# Quality check\n",
    )
    result = CliRunner().invoke(cli, ["tier1", "./quality-check", "--checks", "schema", "--no-llm"])
    assert result.exit_code == 0, result.output
    expert = CliRunner().invoke(cli, ["tier1", "quality-check", "--help"])
    assert expert.exit_code == 0
    assert "--min-score" in expert.output


def test_tier1_validate_alias_stays_inside_tier1(skill, monkeypatch):
    from skillevaluator.tier1 import commands

    monkeypatch.setattr(commands, "run_validation", lambda *_a, **_kw: _result("Schema Validation"))
    result = CliRunner().invoke(cli, ["tier1", "validate", str(skill), "--no-llm"])
    assert result.exit_code == 0, result.output
    assert "Tier 2" not in result.output


@pytest.mark.parametrize("tier,flag", [("tier1", "--no-llm"), ("tier2", "--catalog"), ("tier3", "--agents")])
def test_tier_help_shows_default_workflow_options(tier, flag):
    result = CliRunner().invoke(cli, [tier, "--help"])
    assert result.exit_code == 0
    assert flag in result.output


@pytest.mark.parametrize("width", [80, 120])
def test_tier3_help_keeps_option_descriptions_visible(width):
    import click

    command = cli.commands["tier3"]
    output = render_help(command.workflow, click.Context(command.workflow), width=width)
    normalized = " ".join(output.split())
    assert "Run an extra agent-only execution of" in normalized
    assert "first staged task before the full" in normalized
    assert "matrix [default: disabled]" in normalized
    assert "Per-agent model override" in normalized


def test_tier2_rejects_linked_target_before_provider_resolution(skill, tmp_path):
    link = tmp_path / "linked-skill"
    link.symlink_to(skill, target_is_directory=True)
    result = CliRunner().invoke(cli, ["tier2", str(link)])
    assert result.exit_code == 2
    assert "symlink" in result.output


def test_tier2_rejects_catalog_report_collision_before_reading_or_overwriting(skill, tmp_path):
    catalog = tmp_path / "skillevaluator-tier2.json"
    catalog.write_text("keep this source unchanged")
    result = CliRunner().invoke(cli, ["tier2", str(skill), "--catalog", str(catalog), "-o", str(tmp_path)])
    assert result.exit_code == 2
    assert "conflicts" in result.output
    assert catalog.read_text() == "keep this source unchanged"


def test_full_body_without_catalog_is_rejected(skill):
    result = CliRunner().invoke(cli, ["tier2", str(skill), "--full-body"])
    assert result.exit_code == 2
    assert "--full-body requires --catalog" in result.output


def test_tier1_invalid_provider_is_not_silently_treated_as_keyless(skill, monkeypatch):
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "misspelled-provider")
    result = CliRunner().invoke(cli, ["tier1", str(skill), "--checks", "schema"])
    assert result.exit_code != 0
    assert "provider" in result.output.lower()
    static = CliRunner().invoke(cli, ["tier1", str(skill), "--checks", "schema", "--no-llm", "-r", "cli"])
    assert static.exit_code == 0, static.output


def test_missing_target_and_unknown_workflow_flag_are_errors(tmp_path):
    for tier in ("tier1", "tier2", "tier3"):
        missing = CliRunner().invoke(cli, [tier, str(tmp_path / "missing")])
        assert missing.exit_code == 2
        assert "does not exist" in missing.output
        bad_flag = CliRunner().invoke(cli, [tier, "--unknown-option"])
        assert bad_flag.exit_code == 2
        assert "No such option" in bad_flag.output
