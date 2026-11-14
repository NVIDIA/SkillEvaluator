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
from skillevaluator.reporting.console_ui import final_check_ticker_row, summarize_tier1
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


def test_plain_parent_segment_keeps_the_root(tmp_path: Path):
    skill = tmp_path / ".claude" / "skills" / "demo-skill"
    (skill.parent / "group").mkdir(parents=True)
    _skill(tmp_path / "src").rename(skill)

    facts = skill_facts(skill.parent / "group" / ".." / "demo-skill")

    assert facts["install_root"] == ".claude/skills"
    assert facts["grouping_depth"] == 0


def test_parent_segment_after_a_symlink_goes_to_the_real_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    (project / "sub").mkdir(parents=True)
    skill = project / ".claude" / "skills" / "demo-skill"
    skill.parent.mkdir(parents=True)
    _skill(tmp_path / "src").rename(skill)
    (home / "link").symlink_to(project / "sub", target_is_directory=True)
    monkeypatch.setattr(Path, "home", lambda: home)

    # home/link/.. is the project folder, even though home/.claude/skills would be a user root.
    assert skill_facts(home / "link" / ".." / ".claude" / "skills" / "demo-skill")["install_root"] == ".claude/skills"


def test_parent_segment_after_a_symlink_does_not_lose_the_root(tmp_path: Path):
    store = tmp_path / "store"
    skill = _rooted(store)
    (tmp_path / "project").mkdir()
    (tmp_path / "project" / "link").symlink_to(skill.parent, target_is_directory=True)

    # project/link/.. is store/.agents, so the skill sits in .agents/skills.
    assert (
        skill_facts(tmp_path / "project" / "link" / ".." / "skills" / "demo-skill")["install_root"] == ".agents/skills"
    )


def test_symlinked_skills_folder_counts_where_the_link_is(tmp_path: Path):
    link = tmp_path / "project" / ".claude" / "skills"
    link.parent.mkdir(parents=True)
    link.symlink_to(_rooted(tmp_path / "real").parent, target_is_directory=True)
    (link / "group").mkdir()

    assert skill_facts(link / "demo-skill")["install_root"] == ".claude/skills"
    assert skill_facts(link / "group" / ".." / "demo-skill")["install_root"] == ".claude/skills"


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


def _render(panel) -> str:
    from rich.console import Console

    console = Console(width=120, record=True, color_system=None)
    console.print(panel)
    return console.export_text()


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


def test_harness_check_alone_marks_ticker_and_panel(tmp_path: Path):
    results = run_validation(_rooted(tmp_path), checks="harness", harness="all")
    block = harness_block(results)

    assert [r.validator_name for r in results] == ["Harness Compatibility"]
    assert any(text == "! harness" for text, _ in final_check_ticker_row(["harness"], results).segments)
    assert harness_note(block) == "skill won't load in claude-code"
    text = _render(harness_panel(block))
    assert "Harness compatibility" in text and "advisory" not in text
    assert "not found" in text and "[CC-ROOT]" not in text
    assert "1 of 6 not loaded" in text and "Details and fixes: -v" in text
    full = _render(harness_panel(block, verbose=True))
    assert "NOT LOADED" in full and "[CC-ROOT]" in full


def test_clean_skill_panel_is_one_line(tmp_path: Path):
    block = harness_block(run_validation(_skill(tmp_path), checks="harness"))

    assert harness_note(block) is None
    text = _render(harness_panel(block))
    assert "6 of 6 load" in text and "harness-check" not in text


def test_validate_shows_panel_and_verdict_note(tmp_path: Path):
    skill = _rooted(tmp_path)
    result = CliRunner().invoke(
        cli, ["validate", str(skill), "--tiers", "1", "--checks", "schema,harness", "-o", str(tmp_path / "r")]
    )

    assert result.exit_code == 0, result.output
    assert "Harness compatibility" in result.output and "· advisory" not in result.output
    assert "! harness: skill won't load in claude-code" in result.output


def test_harness_check_command_exits_zero_and_writes_json(tmp_path: Path):
    skill = _rooted(tmp_path)
    out = tmp_path / "h.json"
    result = CliRunner().invoke(
        cli, ["harness-check", str(skill), "--harness", "codex,claude-code", "--json", str(out)]
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


def test_tier1_summary_gives_harness_its_own_line(tmp_path: Path):
    results = run_validation(_rooted(tmp_path), checks="schema,harness", harness="all")
    _, rows = summarize_tier1(results)
    labels = {row.label: "".join(text for text, _ in row.segments) for row in rows}

    assert labels["harness"].startswith("1 not loaded")
    assert "harness" not in labels.get("advisories", "")


def _rule_ids(skill: Path, harness: str, config: dict | None = None) -> set[str]:
    return {f["id"] for r in evaluate_harnesses(skill, harness, config=config) for f in r.findings}


def test_native_root_skips_root_rules(tmp_path):
    skill = tmp_path / ".claude" / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Demo skill for root checks.\n---\nBody.\n")

    def ids(cfg):
        return {f["id"] for r in evaluate_harnesses(skill, "all", config=cfg) for f in r.findings}

    assert "CX-ROOT" in ids({})
    assert not {i for i in ids({"native_root": True}) if i.endswith("-ROOT")}


def test_copilot_drops_skill_without_front_matter(tmp_path: Path):
    skill = tmp_path / ".github" / "skills" / "bare"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Bare\n\nNo front matter here.\n")

    assert "CP-NO-FRONTMATTER" in _rule_ids(skill, "copilot-cli")
    assert "CP-NO-FRONTMATTER" not in _rule_ids(_skill(tmp_path), "copilot-cli")


def test_harness_note_names_the_skill_not_the_harness():
    block = {
        "a": [{"harness": "codex", "status": "not_loaded"}, {"harness": "hermes", "status": "degraded"}],
        "b": [{"harness": "opencode", "status": "not_loaded"}, {"harness": "hermes", "status": "loads"}],
    }

    assert harness_note(block) == "2 skills won't load in codex, opencode · skill loads degraded in hermes"


def test_tab_inside_a_block_scalar_is_not_tab_indentation(tmp_path: Path):
    skill = _rooted(tmp_path)
    (skill / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: |\n  Use when checking.\n  \tindented\n---\nBody\n"
    )

    assert skill_facts(skill)["frontmatter_tab_indent"] is False


def test_tab_after_spaces_that_breaks_yaml_is_flagged(tmp_path: Path):
    skill = _rooted(tmp_path)
    (skill / "SKILL.md").write_text("---\nname: demo-skill\nmetadata:\n  \tauthor: x\n---\nBody\n")

    assert skill_facts(skill)["frontmatter_tab_indent"] is True
