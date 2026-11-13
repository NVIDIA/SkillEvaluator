# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the advisory harness check, panel and harness-check command."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.harness import HARNESS_STATUSES, evaluate_harnesses, load_profiles, skill_facts
from skillevaluator.harness.overlay import PROFILES_DIR
from skillevaluator.models.result import Severity
from skillevaluator.tier1.commands import DEFAULT_CHECKS
from skillevaluator.validators.harness import HarnessValidator

HARNESSES = ["claude-code", "codex", "copilot-cli", "hermes", "openclaw", "opencode"]
BODY = "\n# Demo\n\n## Instructions\n\n1. Step\n\n## Examples\n\nExample.\n"


def _skill(
    root: Path, extra: str = "", desc: str = "Use when checking harness compatibility.", body: str = BODY
) -> Path:
    skill = root / "skills" / "demo-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: demo-skill\ndescription: {desc}\nmetadata:\n  author: Demo User <demo@example.com>\n{extra}---\n{body}",
        encoding="utf-8",
    )
    return skill


def _status(skill: Path) -> dict[str, str]:
    return {r.harness: r.status for r in evaluate_harnesses(skill)}


def _rooted(root: Path) -> Path:
    """A clean skill in .agents/skills, which every harness but Claude Code scans."""
    skill = root / ".agents" / "skills" / "demo-skill"
    skill.parent.mkdir(parents=True)
    _skill(root).rename(skill)
    return skill


def test_packaged_profiles_load_and_pin_versions():
    profiles = load_profiles("all")

    assert sorted(p["harness"] for p in profiles) == HARNESSES
    assert all(p["verified"]["version"] and p["verified"]["date"] for p in profiles)
    assert len(list(PROFILES_DIR.glob("*.yaml"))) == 6


def test_unknown_harness_is_rejected():
    with pytest.raises(ValueError, match="Unknown harness: nope"):
        load_profiles("codex,nope")


def test_clean_skill_loads_everywhere(tmp_path: Path):
    assert set(_status(_skill(tmp_path)).values()) == {"loads"}


def test_install_root_only_applies_inside_known_harness_roots(tmp_path: Path):
    skill = tmp_path / ".claude" / "skills" / "demo-skill"
    skill.parent.mkdir(parents=True)
    _skill(tmp_path).rename(skill)

    facts = skill_facts(skill)
    status = _status(skill)

    assert facts["install_root"] == ".claude/skills"
    assert status["claude-code"] == "loads"
    assert status["codex"] == "not_loaded"


def test_user_root_is_told_apart_from_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    skill = tmp_path / ".copilot" / "skills" / "demo-skill"
    skill.parent.mkdir(parents=True)
    _skill(tmp_path / "src").rename(skill)

    assert skill_facts(skill)["install_root"] == "~/.copilot/skills"
    assert _status(skill)["copilot-cli"] == "loads"


def test_symlinked_skill_folder_counts_where_the_link_is(tmp_path: Path):
    link = tmp_path / ".claude" / "skills" / "demo-skill"
    link.parent.mkdir(parents=True)
    link.symlink_to(_rooted(tmp_path / "real"), target_is_directory=True)

    assert skill_facts(link)["install_root"] == ".claude/skills"


def test_parent_segments_do_not_make_a_false_root(tmp_path: Path):
    _skill(tmp_path)
    (tmp_path / ".claude" / "skills").mkdir(parents=True)

    assert skill_facts(tmp_path / ".claude" / "skills" / ".." / ".." / "skills" / "demo-skill")["install_root"] is None


def test_missing_home_folder_does_not_stop_the_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def no_home() -> Path:
        raise RuntimeError("no home")

    monkeypatch.setattr(Path, "home", no_home)

    assert skill_facts(_rooted(tmp_path))["install_root"] == ".agents/skills"


def test_manifest_file_target_checks_its_skill(tmp_path: Path):
    result = HarnessValidator("claude-code").validate(_rooted(tmp_path) / "SKILL.md")

    assert next(iter(result.metadata["harness_compatibility"].values()))[0]["status"] == "not_loaded"


def test_every_status_is_known(tmp_path: Path):
    assert {r.status for r in evaluate_harnesses(_skill(tmp_path, extra="owner: x\n"))} <= set(HARNESS_STATUSES)


def test_cli_adds_json_block_and_keeps_exit_code(tmp_path: Path):
    skill = _rooted(tmp_path / "src")
    args = ["validate", str(skill), "--tiers", "1", "--checks", "schema,harness", "-r", "json"]

    plain = CliRunner().invoke(cli, [*args, "--output-dir", str(tmp_path / "a")])
    overlay = CliRunner().invoke(cli, [*args, "--output-dir", str(tmp_path / "b"), "--harness", "codex,claude-code"])

    assert plain.exit_code == overlay.exit_code == 0, overlay.output
    report = json.loads(next((tmp_path / "b").rglob("skillevaluator-output-*.json")).read_text())
    rows = {r["harness"]: r for r in next(iter(report["harness_compatibility"].values()))}
    assert rows["claude-code"]["status"] == "not_loaded"
    assert rows["codex"]["status"] == "loads"
    assert "CC-ROOT" in {f["id"] for f in rows["claude-code"]["findings"]}


def test_cli_rejects_unknown_harness(tmp_path: Path):
    result = CliRunner().invoke(cli, ["validate", str(_skill(tmp_path)), "--tiers", "1", "--harness", "nope"])

    assert result.exit_code == 2
    assert "Unknown harness: nope" in result.output


def test_harness_check_runs_by_default_and_never_fails(tmp_path: Path):
    assert "harness" in DEFAULT_CHECKS
    result = HarnessValidator("claude-code,codex").validate(_rooted(tmp_path))

    assert result.passed
    finding = next(f for f in result.findings if f.check_name == "CC-ROOT")
    assert finding.severity is Severity.MEDIUM
    assert finding.metadata["harness"] == "claude-code"
    assert all(f.severity not in (Severity.HIGH, Severity.CRITICAL) for f in result.findings)
    rows = next(iter(result.metadata["harness_compatibility"].values()))
    assert {r["harness"]: r["status"] for r in rows} == {"claude-code": "not_loaded", "codex": "loads"}


def test_native_root_skips_root_rules(tmp_path):
    skill = tmp_path / ".claude" / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Demo skill for root checks.\n---\nBody.\n")

    def ids(cfg):
        return {f["id"] for r in evaluate_harnesses(skill, "all", config=cfg) for f in r.findings}

    assert "CX-ROOT" in ids({})
    assert not {i for i in ids({"native_root": True}) if i.endswith("-ROOT")}
