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
from skillevaluator.harness.report import harness_block, harness_note, harness_panel
from skillevaluator.models.result import Severity
from skillevaluator.reporting.console_ui import final_check_ticker_row
from skillevaluator.tier1.commands import DEFAULT_CHECKS, run_validation
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


def test_duplicate_key_is_not_loaded_by_strict_parsers(tmp_path: Path):
    status = _status(_skill(tmp_path, extra="description: Use when a second value is given.\n"))

    assert status["codex"] == status["copilot-cli"] == status["openclaw"] == status["opencode"] == "not_loaded"
    assert status["claude-code"] == "loads"


def test_bom_is_not_loaded_by_codex_only(tmp_path: Path):
    skill = _skill(tmp_path)
    manifest = skill / "SKILL.md"
    manifest.write_bytes(b"\xef\xbb\xbf" + manifest.read_bytes())

    status = _status(skill)

    assert status["codex"] == "not_loaded"
    assert {h for h, s in status.items() if s != "loads"} == {"codex"}


def test_utf16_overflow_is_not_loaded_by_copilot(tmp_path: Path):
    status = _status(_skill(tmp_path, desc="Use when testing emoji. " + "\U0001f600" * 600))

    assert status["copilot-cli"] == "not_loaded"
    assert status["codex"] == "loads"


def test_symlinked_manifest(tmp_path: Path):
    skill = _skill(tmp_path)
    target = tmp_path / "real.md"
    (skill / "SKILL.md").rename(target)
    (skill / "SKILL.md").symlink_to(target)

    status = _status(skill)

    assert {h for h, s in status.items() if s == "not_loaded"} == {"codex", "hermes", "openclaw"}


def test_install_root_only_applies_inside_known_harness_roots(tmp_path: Path):
    skill = tmp_path / ".claude" / "skills" / "demo-skill"
    skill.parent.mkdir(parents=True)
    _skill(tmp_path).rename(skill)

    facts = skill_facts(skill)
    status = _status(skill)

    assert facts["install_root"] == ".claude/skills"
    assert status["claude-code"] == "loads"
    assert status["codex"] == "not_loaded"


def test_every_status_is_known(tmp_path: Path):
    assert {r.status for r in evaluate_harnesses(_skill(tmp_path, extra="owner: x\n"))} <= set(HARNESS_STATUSES)


def test_cli_adds_json_block_and_keeps_exit_code(tmp_path: Path):
    skill = _skill(tmp_path, desc="Use when testing emoji. " + "\U0001f600" * 600)
    args = ["validate", str(skill), "--tiers", "1", "--checks", "schema,harness", "-r", "json"]

    plain = CliRunner().invoke(cli, [*args, "--output-dir", str(tmp_path / "a")])
    overlay = CliRunner().invoke(cli, [*args, "--output-dir", str(tmp_path / "b"), "--harness", "codex,copilot-cli"])

    assert plain.exit_code == overlay.exit_code == 0, overlay.output
    report = json.loads(next((tmp_path / "b").rglob("skillevaluator-output-*.json")).read_text())
    rows = {r["harness"]: r for r in next(iter(report["harness_compatibility"].values()))}
    assert rows["copilot-cli"]["status"] == "not_loaded"
    assert rows["codex"]["status"] == "loads"
    assert "CP-DESC-LIMIT" in {f["id"] for f in rows["copilot-cli"]["findings"]}


def test_cli_rejects_unknown_harness(tmp_path: Path):
    result = CliRunner().invoke(cli, ["validate", str(_skill(tmp_path)), "--tiers", "1", "--harness", "nope"])

    assert result.exit_code == 2
    assert "Unknown harness: nope" in result.output


EMOJI_DESC = "Use when testing emoji. " + "\U0001f600" * 600


def _render(panel) -> str:
    from rich.console import Console

    console = Console(width=120, record=True, color_system=None)
    console.print(panel)
    return console.export_text()


def test_harness_check_runs_by_default_and_never_fails(tmp_path: Path):
    assert "harness" in DEFAULT_CHECKS
    result = HarnessValidator("copilot-cli,codex").validate(_skill(tmp_path, desc=EMOJI_DESC))

    assert result.passed
    finding = next(f for f in result.findings if f.check_name == "CP-DESC-LIMIT")
    assert finding.severity is Severity.MEDIUM
    assert finding.metadata["harness"] == "copilot-cli"
    assert all(f.severity not in (Severity.HIGH, Severity.CRITICAL) for f in result.findings)
    rows = next(iter(result.metadata["harness_compatibility"].values()))
    assert {r["harness"]: r["status"] for r in rows} == {"copilot-cli": "not_loaded", "codex": "loads"}


def test_harness_check_alone_marks_ticker_and_panel(tmp_path: Path):
    results = run_validation(_skill(tmp_path, desc=EMOJI_DESC), checks="harness", harness="all")
    block = harness_block(results)

    assert [r.validator_name for r in results] == ["Harness Compatibility"]
    assert any(text == "! harness" for text, _ in final_check_ticker_row(["harness"], results).segments)
    assert harness_note(block) == "won't load in copilot-cli"
    text = _render(harness_panel(block))
    assert "Harness compatibility · advisory" in text
    assert "NOT LOADED" in text and "[CP-DESC-LIMIT]" in text
    assert "1 of 6 not loaded" in text
    assert "harness-check <skill>" in text


def test_clean_skill_panel_is_one_line(tmp_path: Path):
    block = harness_block(run_validation(_skill(tmp_path), checks="harness"))

    assert harness_note(block) is None
    text = _render(harness_panel(block))
    assert "6 of 6 load" in text and "harness-check" not in text


def test_validate_shows_panel_and_verdict_note(tmp_path: Path):
    skill = _skill(tmp_path, desc=EMOJI_DESC)
    result = CliRunner().invoke(
        cli, ["validate", str(skill), "--tiers", "1", "--checks", "schema,harness", "-o", str(tmp_path / "r")]
    )

    assert result.exit_code == 0, result.output
    assert "Harness compatibility · advisory" in result.output
    assert "! harness: won't load in copilot-cli" in result.output


def test_harness_check_command_exits_zero_and_writes_json(tmp_path: Path):
    skill = _skill(tmp_path, desc=EMOJI_DESC)
    out = tmp_path / "h.json"
    result = CliRunner().invoke(
        cli, ["harness-check", str(skill), "--harness", "codex,copilot-cli", "--json", str(out)]
    )

    assert result.exit_code == 0, result.output
    assert "Harness compatibility · demo-skill" in result.output
    assert "1 of 2 not loaded" in result.output
    rows = next(iter(json.loads(out.read_text())["harness_compatibility"].values()))
    assert [r["status"] for r in rows] == ["loads", "not_loaded"]


def test_harness_check_command_rejects_unknown_harness(tmp_path: Path):
    result = CliRunner().invoke(cli, ["harness-check", str(_skill(tmp_path)), "--harness", "nope"])

    assert result.exit_code == 2
    assert "Unknown harness: nope" in result.output
