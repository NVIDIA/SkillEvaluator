# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ordinary validation path prepares and runs all three skill tiers."""

import json

import pytest
from click.testing import CliRunner

from skillevaluator import cli as cli_module
from skillevaluator.models.result import ValidationResult


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], ["tier1", "tier2", "dataset", "tier3"]),
        (["--full"], ["tier1", "tier2", "dataset", "tier3"]),
        (["--tier3"], ["tier1", "tier2", "dataset", "tier3"]),
        (["--autopilot"], ["tier1", "tier2", "dataset", "tier3"]),
        (["--tiers", "1,3"], ["tier1", "dataset", "tier3"]),
        (["--tiers", "1,2"], ["tier1", "tier2"]),
        (["--full", "--tiers", "1"], ["tier1"]),
        (["--no-tier2"], ["tier1", "dataset", "tier3"]),
        (["--no-tier3"], ["tier1", "tier2"]),
        (["--no-agent-eval"], ["tier1", "tier2"]),
        (["--full", "--no-tier3"], ["tier1", "tier2"]),
        (["--autopilot", "--no-tier3"], ["tier1", "tier2"]),
        (["--no-autopilot"], ["tier1", "tier2", "tier3"]),
        (["--full", "--no-autopilot"], ["tier1", "tier2", "tier3"]),
        (["--no-tier3", "--tiers", "1,3"], ["tier1", "dataset", "tier3"]),
    ],
)
def test_skill_validate_selects_tiers_and_prepares_dataset(monkeypatch, tmp_path, flags, expected):
    skill = tmp_path / "sample"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: sample\ndescription: Test skill\n---\n")
    calls = []

    def run(name, path, **kwargs):
        assert path == skill
        calls.append(name)
        result = ValidationResult(validator_name="AGENT_EVAL" if name == "tier3" else name)
        result.add_success(name, "ok")
        return result

    monkeypatch.setattr(cli_module, "run_validation", lambda path, **kw: [run("tier1", path, **kw)])
    monkeypatch.setattr(cli_module, "_run_dedup_or_skip", lambda path: [run("tier2", path)])
    monkeypatch.setattr(cli_module, "_ensure_autopilot_dataset", lambda path, **_kw: run("dataset", path) and None)
    monkeypatch.setattr(cli_module, "_run_agent_eval_or_skip", lambda path, **kw: run("tier3", path, **kw))
    result = CliRunner().invoke(cli_module.cli, ["validate", str(skill), *flags, "-o", str(tmp_path / "reports")])
    assert result.exit_code == 0, result.output
    assert calls == expected


def test_default_autopilot_receives_skill_directory_for_file_input(monkeypatch, tmp_path):
    from skillevaluator.tier3 import evals_spec

    skill = tmp_path / "file-input"
    skill.mkdir()
    marker = skill / "SKILL.md"
    marker.write_text("---\nname: file-input\ndescription: Test skill\n---\n")
    calls = []
    monkeypatch.setattr(cli_module, "run_validation", lambda *_a, **_kw: [])
    monkeypatch.setattr(cli_module, "_run_dedup_or_skip", lambda *_a, **_kw: [])
    monkeypatch.setattr(cli_module, "_ensure_autopilot_dataset", lambda path, **_kw: calls.append(path))

    def evaluate(path, **kwargs):
        calls.append(path)
        result = ValidationResult(validator_name="AGENT_EVAL")
        result.metadata["agent_eval"] = {"agents": {}}
        return result

    monkeypatch.setattr(cli_module, "_run_agent_eval_or_skip", evaluate)

    def validate_source(path):
        assert path == skill
        return "native_harbor", []

    monkeypatch.setattr(evals_spec, "validate_tier3_source", validate_source)
    result = CliRunner().invoke(
        cli_module.cli, ["validate", str(marker), "-r", "json", "-o", str(tmp_path / "reports")]
    )
    assert result.exit_code == 0, result.output
    assert calls == [skill, skill]
    report = json.loads(next((tmp_path / "reports").glob("*.json")).read_text())
    assert report["tier3_applicability"]["source_kind"] == "native_harbor"


@pytest.mark.parametrize("content_type", ["rules", "workflows", "plugin"])
def test_non_skill_default_does_not_generate_or_run_skill_evals(monkeypatch, tmp_path, content_type):
    monkeypatch.setattr(cli_module, "run_validation", lambda *_a, **_kw: [])
    monkeypatch.setattr(cli_module, "_run_dedup_or_skip", lambda *_a, **_kw: [])
    monkeypatch.setattr(
        cli_module, "_ensure_autopilot_dataset", lambda *_a, **_kw: pytest.fail("unexpected generation")
    )
    monkeypatch.setattr(cli_module, "_run_agent_eval_or_skip", lambda *_a, **_kw: pytest.fail("unexpected evaluation"))
    result = CliRunner().invoke(
        cli_module.cli, ["validate", str(tmp_path), "--type", content_type, "-o", str(tmp_path / "reports")]
    )
    assert result.exit_code == 0, result.output


def test_default_tier3_runs_after_tier1_failure(monkeypatch, tmp_path):
    skill = tmp_path / "failing-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: failing-skill\ndescription: Test skill\n---\n")
    failed = ValidationResult(validator_name="SCHEMA")
    failed.add_error("metadata.author missing")
    calls = []
    monkeypatch.setattr(cli_module, "run_validation", lambda *_a, **_kw: [failed])
    monkeypatch.setattr(cli_module, "_run_dedup_or_skip", lambda *_a, **_kw: [])
    monkeypatch.setattr(cli_module, "_ensure_autopilot_dataset", lambda *_a, **_kw: calls.append("dataset"))

    def evaluate(*args, **kwargs):
        calls.append("tier3")
        return ValidationResult(validator_name="AGENT_EVAL")

    monkeypatch.setattr(cli_module, "_run_agent_eval_or_skip", evaluate)
    result = CliRunner().invoke(cli_module.cli, ["validate", str(skill), "-o", str(tmp_path / "reports")])
    assert result.exit_code == 1
    assert calls == ["dataset", "tier3"]
    assert list((tmp_path / "reports").glob("*.json"))
