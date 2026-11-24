# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the advisory harness check, panel and harness-check command."""

import json
import os
import re
from pathlib import Path

import pytest
from click.testing import CliRunner
from tests.yaml_alias_fixtures import alias_bomb_skill_md, alias_graph_lines, deeply_nested_flow, merge_bomb_yaml

from skillevaluator.cli import cli
from skillevaluator.harness import HARNESS_STATUSES, evaluate_harnesses, facts_for, load_profiles, overlay, skill_facts
from skillevaluator.harness.overlay import PROFILES_DIR, _description_words, listed_description
from skillevaluator.harness.report import _LISTING, harness_block, harness_note, harness_panel
from skillevaluator.models.result import Severity
from skillevaluator.reporting.console_ui import final_check_ticker_row, summarize_tier1
from skillevaluator.tier1.commands import DEFAULT_CHECKS, run_validation
from skillevaluator.validators.harness import HarnessValidator
from skillevaluator.validators.schema import SchemaValidator

HARNESSES = ["claude-code", "codex", "copilot-cli", "hermes", "openclaw", "opencode", "pi"]
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
    assert len(list(PROFILES_DIR.glob("*.yaml"))) == 7


def test_unknown_harness_is_rejected():
    with pytest.raises(ValueError, match="Unknown harness: nope"):
        load_profiles("codex,nope")


def test_clean_skill_loads_everywhere(tmp_path: Path):
    assert set(_status(_skill(tmp_path)).values()) == {"loads"}


def test_duplicate_key_is_not_loaded_by_strict_parsers(tmp_path: Path):
    status = _status(_skill(tmp_path, extra="description: Use when a second value is given.\n"))

    assert (
        status["codex"]
        == status["copilot-cli"]
        == status["openclaw"]
        == status["opencode"]
        == status["pi"]
        == "not_loaded"
    )
    assert status["claude-code"] == "loads"


def test_bom_is_not_loaded_by_codex_only(tmp_path: Path):
    skill = _skill(tmp_path)
    manifest = skill / "SKILL.md"
    manifest.write_bytes(b"\xef\xbb\xbf" + manifest.read_bytes())

    status = _status(skill)

    assert status["codex"] == "not_loaded"
    assert status["pi"] == "loads"
    assert {h for h, s in status.items() if s != "loads"} == {"codex"}


def test_utf16_overflow_is_not_loaded_by_copilot(tmp_path: Path):
    status = _status(_skill(tmp_path, desc="Use when testing emoji. " + "\U0001f600" * 600))

    assert status["copilot-cli"] == "not_loaded"
    assert status["codex"] == "loads"


def test_copilot_shows_body_for_missing_description_but_drops_non_string(tmp_path: Path):
    skill = tmp_path / "skills" / "no-desc"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: no-desc\n---\nCustom skill\n", encoding="utf-8")
    rows = _rows(skill)
    assert rows["copilot-cli"]["status"] == "degraded"
    assert "CP-DESC-MISSING" in _ids(rows, "copilot-cli") and "CP-STRICT-YAML" not in _ids(rows, "copilot-cli")
    listed = _skill(tmp_path / "list", desc="[a, b]")
    assert "CP-STRICT-YAML" in _ids(_rows(listed), "copilot-cli")


def test_copilot_lists_empty_description_by_name_only(tmp_path: Path):
    for n, desc in enumerate(['""', '"   "']):
        rows = _rows(_skill(tmp_path / str(n), desc=desc))
        assert rows["copilot-cli"]["status"] == "degraded"
        assert "CP-DESC-EMPTY" in _ids(rows, "copilot-cli") and "CP-DESC-MISSING" not in _ids(rows, "copilot-cli")


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
    assert "1 of 7 not loaded" in text and "Details and fixes: -v" in text
    full = _render(harness_panel(block, verbose=True))
    assert "NOT LOADED" in full and "[CC-ROOT]" in full


def test_clean_skill_panel_is_one_line(tmp_path: Path):
    block = harness_block(run_validation(_skill(tmp_path), checks="harness"))

    assert harness_note(block) is None
    text = _render(harness_panel(block))
    assert "7 of 7 load" in text and "harness-check" not in text


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


def _catalog(root: Path, count: int, desc_len: int = 300) -> Path:
    skills = root / ".claude" / "skills"
    for index in range(count):
        skill = skills / f"filler-{index:03d}"
        skill.mkdir(parents=True)
        desc = ("Use when testing catalog budgets. " + "x" * desc_len)[:desc_len]
        (skill / "SKILL.md").write_text(f"---\nname: filler-{index:03d}\ndescription: {desc}\n---\n{BODY}")
    return skills


def _rows(skill: Path) -> dict[str, dict]:
    return {r.harness: r.to_dict() for r in evaluate_harnesses(skill)}


def _ids(rows: dict[str, dict], harness: str) -> set[str]:
    return {f["id"] for f in rows[harness]["findings"]}


def test_single_skill_is_measured_against_its_installed_folder(tmp_path: Path):
    _catalog(tmp_path, 40)
    rows = _rows(tmp_path / ".claude" / "skills" / "filler-000")

    assert rows["claude-code"]["status"] == "degraded"
    assert "CC-LIST-BUDGET" in _ids(rows, "claude-code")
    assert "CP-LIST-BUDGET" in _ids(rows, "copilot-cli")
    message = next(f["message"] for f in rows["claude-code"]["findings"] if f["id"] == "CC-LIST-BUDGET")
    assert message.startswith("The 40 skills in this folder have 12,000 description chars")
    assert rows["claude-code"]["listed"] == rows["codex"]["listed"] == {"mode": "shortened", "text": ""}
    assert rows["claude-code"]["catalog"]["size"] == 40


def test_listing_budget_scales_with_context_window(tmp_path: Path):
    skill = _catalog(tmp_path, 34) / "filler-000"

    def ids(tokens: int) -> set[str]:
        rows = {r.harness: r.to_dict() for r in evaluate_harnesses(skill, config={"context_tokens": tokens})}
        return _ids(rows, "claude-code") | _ids(rows, "codex")

    assert {"CC-LIST-BUDGET", "CX-LIST-BUDGET"} <= ids(0)
    assert not {"CC-LIST-BUDGET", "CX-LIST-BUDGET"} & ids(272000)
    assert {"CC-LIST-BUDGET", "CX-LIST-BUDGET"} & ids(200000) == {"CC-LIST-BUDGET"}


def test_skill_outside_a_skills_folder_has_no_catalog_rules(tmp_path: Path):
    rows = _rows(_skill(tmp_path))

    assert not any(
        f["id"].endswith(("LIST-BUDGET", "DUP-NAME", "PRECEDENCE")) for r in rows.values() for f in r["findings"]
    )


def test_folder_of_skills_is_its_own_catalog(tmp_path: Path):
    skills = _catalog(tmp_path, 3)
    facts = facts_for(sorted(skills.iterdir()))

    assert {f["catalog_size"] for f in facts.values()} == {3}
    assert {f["catalog_desc_chars"] for f in facts.values()} == {900}


def test_duplicate_names_report_each_harness_winner(tmp_path: Path):
    skills = _catalog(tmp_path, 2)
    copy = skills / "filler-copy"
    copy.mkdir()
    (copy / "SKILL.md").write_text(f"---\nname: filler-000\ndescription: Use when testing duplicates.\n---\n{BODY}")
    rows = _rows(skills / "filler-000")

    assert rows["opencode"]["status"] == "degraded"
    assert {
        "CC-PRECEDENCE",
        "CX-DUP-NAME",
        "CP-DUP-NAME",
        "HM-DUP-NAME",
        "OW-DUP-NAME",
        "OC-DUP-NAME",
        "PI-NAME-COLLISION",
    } <= {f["id"] for r in rows.values() for f in r["findings"]}
    assert "(filler-copy/SKILL.md)" in next(
        f["message"] for f in rows["opencode"]["findings"] if f["id"] == "OC-DUP-NAME"
    )
    assert "lists both" in next(f["message"] for f in rows["hermes"]["findings"] if f["id"] == "HM-DUP-NAME")


def _install(folder: Path, dirname: str, name: str | None = None) -> Path:
    skill = folder / dirname
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name or dirname}\ndescription: Use when testing catalogs.\n---\n{BODY}", encoding="utf-8"
    )
    return skill


def test_separate_skills_folders_are_separate_catalogs(tmp_path: Path):
    claude = [_install(tmp_path / ".claude" / "skills", n) for n in ("alpha", "beta")]
    codex = [_install(tmp_path / ".codex" / "skills", n) for n in ("alpha", "gamma", "delta")]

    facts = facts_for([*claude, *codex])

    assert [facts[s]["catalog_size"] for s in claude] == [2, 2]
    assert [facts[s]["catalog_size"] for s in codex] == [3, 3, 3]
    assert not any(f["duplicate_name"] for f in facts.values())
    assert facts[claude[0]]["catalog_base"].endswith(".claude/skills")


def test_unrooted_skills_given_together_stay_one_catalog(tmp_path: Path):
    pack = [_install(tmp_path / "pack", n, name="same") for n in ("one", "two")]

    facts = facts_for(pack)

    assert {f["catalog_size"] for f in facts.values()} == {2}
    assert all(f["duplicate_name"] for f in facts.values())


def test_workspace_skills_folder_counts_its_siblings(tmp_path: Path):
    first = _install(tmp_path / "skills", "first", name="same")
    _install(tmp_path / "skills", "second", name="same")

    facts = facts_for([first])[first]

    assert facts["install_root"] is None
    assert facts["catalog_size"] == 2 and facts["catalog_base"].endswith("skills")
    assert "OW-DUP-NAME" in _ids(_rows(first), "openclaw")


def test_symlinked_skill_folders_count_toward_the_catalog(tmp_path: Path):
    skills = tmp_path / ".claude" / "skills"
    real = _install(skills, "real", name="same")
    store = _install(tmp_path / "store", "linked", name="same")
    (skills / "linked").symlink_to(store, target_is_directory=True)
    (skills / "loop").symlink_to(skills, target_is_directory=True)  # a link back to its own folder

    facts = facts_for([real])[real]

    assert facts["catalog_size"] == 2
    assert facts["duplicate_name"]


def test_reload_guidance_reaches_findings(tmp_path: Path):
    rows = _rows(_skill(tmp_path))

    reload = {f["id"]: f for r in rows.values() for f in r["findings"] if f["id"].endswith("RELOAD")}
    assert set(reload) == {"CP-RELOAD", "OC-RELOAD"}
    assert {f["effect"] for f in reload.values()} == {"advisory"}
    assert {r["status"] for r in rows.values()} == {"loads"}


def test_user_level_copy_of_a_project_skill_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home, project = tmp_path / "home", tmp_path / "project"
    monkeypatch.setattr(Path, "home", lambda: home)
    for folder in (project / ".claude/skills", home / ".claude/skills", home / ".agents/skills"):
        (folder / "report").mkdir(parents=True)
        (folder / "report" / "SKILL.md").write_text(
            f"---\nname: report\ndescription: Use when testing copies.\n---\n{BODY}"
        )
    rows = _rows(project / ".claude/skills/report")

    copies = {
        h: next((f["message"] for f in r["findings"] if f["id"].endswith("-USER-COPY")), "") for h, r in rows.items()
    }
    assert "~/.claude/skills/report; Claude Code uses that user-level copy" in copies["claude-code"]
    assert "~/.agents/skills/report; Copilot CLI uses this project copy" in copies["copilot-cli"]
    assert "~/.agents/skills/report, ~/.claude/skills/report" in copies["opencode"]
    assert copies["codex"] == copies["hermes"] == copies["openclaw"] == ""
    assert not _rows(home / ".claude/skills/report")["claude-code"]["findings"]


def test_only_the_user_roots_a_harness_reads_are_searched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    searched = []
    find_skills = overlay._find_skills
    monkeypatch.setattr(overlay, "_find_skills", lambda folder: searched.append(folder) or find_skills(folder))
    for folder in (home / ".claude" / "skills", home / ".codex" / "skills", home / ".agents" / "skills"):
        _install(folder, "report")
    skill = _install(tmp_path / "project" / ".claude" / "skills", "report")

    evaluate_harnesses(skill, "claude-code")

    assert [f for f in searched if home in f.parents] == [home / ".claude" / "skills"]


def test_unreadable_user_skill_does_not_abort_the_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    _install(home / ".claude" / "skills", "broken")
    _install(home / ".claude" / "skills", "report")
    read_manifest = overlay._read_manifest

    def read(path: Path, explicit=None):
        if path.name == "broken":
            raise PermissionError(13, "Permission denied")
        return read_manifest(path, explicit)

    monkeypatch.setattr(overlay, "_read_manifest", read)
    skill = _install(tmp_path / "project" / ".claude" / "skills", "report")

    rows = _rows(skill)

    assert "CC-USER-COPY" in _ids(rows, "claude-code")


def test_user_copy_in_a_folder_the_harness_does_not_scan_is_not_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    home, project = tmp_path / "home", tmp_path / "project"
    monkeypatch.setattr(Path, "home", lambda: home)
    _install(home / ".claude" / "skills" / "group", "report")
    _install(home / ".codex" / "skills" / "a" / "b" / "c" / "d" / "e" / "f", "report")  # six folders down: past Codex
    claude = _install(project / ".claude" / "skills", "report")
    codex = _install(project / ".codex" / "skills", "report")

    claude_rows, codex_rows = _rows(claude), _rows(codex)

    assert "CC-USER-COPY" not in _ids(claude_rows, "claude-code")
    opencode = next(f["message"] for f in claude_rows["opencode"]["findings"] if f["id"] == "OC-USER-COPY")
    assert "~/.claude/skills/group/report" in opencode
    assert "CX-USER-COPY" not in _ids(codex_rows, "codex")


def _custom_grouping_profile(tmp_path: Path, grouping: str) -> Path:
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "acme.yaml").write_text(
        "harness: acme\n"
        "roots: {project: [.claude/skills], user: [~/.claude/skills]}\n"
        "rules:\n"
        f'  - {{id: AC-GROUPING, when: "{grouping}", effect: not_loaded, message: "Grouping is not searched."}}\n'
        '  - {id: AC-USER-COPY, when: "user_copy", effect: advisory, message: "Also at {user_copy_paths}."}\n',
        encoding="utf-8",
    )
    return profiles


def _custom_user_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, grouping: str, config: dict | None = None
) -> set[str]:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    _install(home / ".claude" / "skills", "report")
    _install(home / ".claude" / "skills" / "group", "report")
    skill = _install(tmp_path / "project" / ".claude" / "skills", "report")
    profiles = _custom_grouping_profile(tmp_path, grouping)

    (result,) = evaluate_harnesses(skill, "acme", profiles, config)

    message = next((f["message"] for f in result.findings if f["id"] == "AC-USER-COPY"), "")
    return set(message.removeprefix("Also at ").removesuffix(".").split(", ")) - {""}


def test_custom_grouping_rule_can_name_roots_when_looking_for_a_user_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    both = {"~/.claude/skills/report", "~/.claude/skills/group/report"}

    # The rule is read for the user copy's root: it hides the grouped copy and keeps the direct one.
    user_rule = "install_root in roots.user and grouping_depth > 0"
    assert _custom_user_copies(tmp_path / "user", monkeypatch, user_rule) == {"~/.claude/skills/report"}
    # A project-only rule says nothing about a copy in a user root, even though the project skill sits in a project root.
    project_rule = "install_root in roots.project and grouping_depth > 0"
    assert _custom_user_copies(tmp_path / "project", monkeypatch, project_rule) == both


def test_custom_grouping_rule_can_name_the_configuration_when_looking_for_a_user_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    rule = "custom_provider and grouping_depth > 0"

    hidden = _custom_user_copies(tmp_path / "hidden", monkeypatch, rule, {"custom_provider": True})
    shown = _custom_user_copies(tmp_path / "shown", monkeypatch, rule)

    assert hidden == {"~/.claude/skills/report"}
    assert shown == {"~/.claude/skills/report", "~/.claude/skills/group/report"}


def test_unevaluable_grouping_rule_does_not_stop_the_user_copy_lookup():
    rule = {"id": "X", "when": "grouping_depth > 0 and nope", "effect": "not_loaded"}
    profile = {"harness": "acme", "roots": {}, "rules": [rule]}
    names = overlay._rule_names(profile, {"install_root": None, "catalog_size": 0}, {"native_root": False})

    assert overlay._finds_at_depth(profile, names, "~/.claude/skills", 1)


def test_openclaw_workspace_skill_is_checked_against_both_user_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home, project = tmp_path / "home", tmp_path / "project"
    monkeypatch.setattr(Path, "home", lambda: home)
    for folder in (home / ".agents" / "skills", home / ".openclaw" / "skills", home / ".claude" / "skills"):
        _install(folder, "report")
    workspace = _install(project / "skills", "report")
    loose = _install(project / "loose", "report")

    rows = _rows(workspace)

    message = next(f["message"] for f in rows["openclaw"]["findings"] if f["id"] == "OW-USER-COPY")
    assert "~/.agents/skills/report, ~/.openclaw/skills/report;" in message
    assert not any(f["id"].endswith("-USER-COPY") for h, r in rows.items() if h != "openclaw" for f in r["findings"])
    assert facts_for([loose])[loose]["user_copies"] == []


def test_claude_listing_cut_counts_when_to_use(tmp_path: Path):
    desc = "Use when building reports."
    over = _rows(_skill(tmp_path / "over", extra=f"when_to_use: {'w' * 1520}\n", desc=desc))["claude-code"]
    under = _rows(_skill(tmp_path / "under", extra=f"when_to_use: {'w' * 1500}\n", desc=desc))["claude-code"]

    cut = next(f for f in over["findings"] if f["id"] == "CC-DESC-CUT")
    assert "1,549 with when_to_use" in cut["message"]
    assert over["listed"]["mode"] == "cut"
    assert over["listed"]["text"].startswith(f"{desc} - www")
    assert "CC-DESC-CUT" not in {f["id"] for f in under["findings"]}
    assert under["listed"]["mode"] == "full"


def test_listed_description_projection(tmp_path: Path):
    desc = "Use when " + "a" * 1600
    rows = _rows(_skill(tmp_path, desc=desc))

    assert rows["hermes"]["listed"] == {"mode": "cut", "text": desc[:60], "limit": 60}
    assert rows["claude-code"]["listed"]["limit"] == 1536
    assert rows["codex"]["listed"]["limit"] == 1024
    assert rows["copilot-cli"]["listed"]["mode"] == "dropped"
    assert rows["opencode"]["listed"]["mode"] == rows["openclaw"]["listed"]["mode"] == "full"


def test_bundle_files_symlinks_and_large_references(tmp_path: Path):
    skill = _skill(tmp_path)
    (skill / "references").mkdir()
    (skill / "references" / "big.md").write_text("line\n" * 20000)
    outside = tmp_path / "shared.md"
    outside.write_text("shared")
    (skill / "references" / "shared.md").symlink_to(outside)
    facts = skill_facts(skill)
    rows = _rows(skill)

    assert facts["resource_symlinks"] == facts["resource_symlink_escapes"] == 1
    assert facts["largest_resource"] == "references/big.md" and facts["largest_resource_lines"] == 20000
    assert "HM-RESOURCE-SYMLINK" in _ids(rows, "hermes")
    assert "OW-REF-BUDGET" in _ids(rows, "openclaw")
    assert "OC-REF-CUT" in _ids(rows, "opencode")


def test_disable_model_invocation_is_ignored_by_three_harnesses(tmp_path: Path):
    rows = _rows(_skill(tmp_path, extra="disable-model-invocation: true\n"))

    flagged = {h for h, r in rows.items() if any(f["id"].endswith("DMI-IGNORED") for f in r["findings"])}
    assert flagged == {"codex", "hermes", "opencode"}


@pytest.mark.parametrize(
    ("value", "hidden"),
    [("true", True), ("True", True), ('"true"', False), ("yes", False), ("false", False)],
)
def test_pi_hidden_matches_pis_boolean_semantics(tmp_path: Path, value: str, hidden: bool):
    skill = _skill(tmp_path, extra=f"disable-model-invocation: {value}\n")
    rows = _rows(skill)

    assert ("PI-HIDDEN" in _ids(rows, "pi")) is hidden
    # A value that looks hiding to a YAML 1.1 reader or _is_true but is a plain
    # string for pi gets an advisory instead of silence.
    invalid = "PI-HIDDEN-INVALID" in _ids(rows, "pi")
    assert invalid is (not hidden and value.strip('"') in {"true", "yes"})


def test_pi_hidden_matches_pis_semantics_for_quoted_keys_and_anchors(tmp_path: Path):
    """A quoted key and an anchored/aliased scalar map the same as the plain spelling."""
    for extra in (
        '"disable-model-invocation": true\n',
        "disable-model-invocation: &flag true\n",
        "flag: &flag true\ndisable-model-invocation: *flag\n",
    ):
        skill = _skill(tmp_path / f"q{abs(hash(extra)) % 1000}", extra=extra)
        rows = _rows(skill)

        assert "PI-HIDDEN" in _ids(rows, "pi"), extra
        assert "PI-HIDDEN-INVALID" not in _ids(rows, "pi"), extra


def test_pi_hidden_resolves_pi_bool_tag_on_quoted_scalar(tmp_path: Path):
    """js-yaml's core schema resolves an explicit !!bool tag, so !!bool "true" hides the skill for pi."""
    skill = _skill(tmp_path, extra='disable-model-invocation: !!bool "true"\n')
    rows = _rows(skill)

    assert "PI-HIDDEN" in _ids(rows, "pi")
    assert "PI-HIDDEN-INVALID" not in _ids(rows, "pi")


def test_pi_hidden_reads_the_top_level_value_over_a_metadata_copy(tmp_path: Path):
    """pi resolves the top-level key; a metadata mapping holding the same key doesn't override it."""
    skill = tmp_path / "skills" / "demo-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\n"
        "name: demo-skill\n"
        "description: Use when checking harness compatibility.\n"
        "metadata:\n"
        "  author: Demo User <demo@example.com>\n"
        "  disable-model-invocation: false\n"
        "disable-model-invocation: true\n"
        f"---\n{BODY}",
        encoding="utf-8",
    )
    rows = _rows(skill)

    assert "PI-HIDDEN" in _ids(rows, "pi")
    assert "PI-HIDDEN-INVALID" not in _ids(rows, "pi")


def test_direct_root_duplicate_scan_stays_inside_the_skills_root(tmp_path: Path):
    """A loose .pi/skills/SKILL.md must not collide with same-named skills outside the root."""
    root = tmp_path / "proj" / ".pi" / "skills"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Use when testing duplicates.\n---\nBody\n", encoding="utf-8"
    )
    example = tmp_path / "proj" / ".pi" / "notes" / "example"
    example.mkdir(parents=True)
    (example / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Use when testing duplicates.\n---\nBody\n", encoding="utf-8"
    )

    rows = _rows(root)
    assert rows["pi"]["status"] == "loads"
    assert "PI-NAME-COLLISION" not in _ids(rows, "pi")


def test_pi_trust_covers_a_manifest_directly_at_the_root(tmp_path: Path):
    """A .pi/skills/SKILL.md with no child folder still counts as installed at the root."""
    root = tmp_path / "proj" / ".pi" / "skills"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Use when testing trust.\n---\nBody\n", encoding="utf-8"
    )

    rows = _rows(root)
    assert rows["pi"]["status"] == "loads"
    assert "PI-TRUST" in _ids(rows, "pi")  # project roots require trust, equal-to-root too
    assert "PI-ROOT" not in _ids(rows, "pi")


def test_harness_check_accepts_the_documented_mixed_case_manifest(tmp_path: Path):
    """harness-check on an explicit Skill.md (pi loads it nested under .agents/skills)
    reports instead of exiting 1 with No skills found."""
    skill = tmp_path / "proj" / ".agents" / "skills" / "demo" / "Skill.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: demo\ndescription: Use when testing casing.\n---\nBody\n", encoding="utf-8")

    result = CliRunner().invoke(
        cli, ["harness-check", str(skill), "--harness", "pi", "--json", str(tmp_path / "h.json")]
    )

    assert result.exit_code == 0, result.output
    assert "No skills found" not in result.output
    # The schema summary gives the canonical-name diagnostic, not the generic rerun.
    assert "no canonical SKILL.md" in result.output
    assert "SKILL.md not found in skill directory" not in result.output
    rows = next(iter(json.loads((tmp_path / "h.json").read_text())["harness_compatibility"].values()))
    assert rows[0]["harness"] == "pi" and rows[0]["status"] == "loads"


def test_explicit_manifest_argument_evaluates_that_exact_file(tmp_path: Path):
    """With Skill.md and skill.md siblings (case-sensitive filesystem), an explicit
    Skill.md path evaluates that file's front matter, not a rediscovered sibling."""
    skill = tmp_path / "proj" / ".agents" / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "Skill.md").write_text(
        "---\nname: demo\ndescription: Use when testing casing.\ndisable-model-invocation: true\n---\nBody\n",
        encoding="utf-8",
    )
    (skill / "skill.md").write_text(
        "---\nname: demo\ndescription: Use when testing casing.\n---\nBody\n", encoding="utf-8"
    )
    if (skill / "Skill.md").samefile(skill / "skill.md"):
        pytest.skip("case-insensitive filesystem: distinct-cased siblings are one file")

    result = CliRunner().invoke(
        cli, ["harness-check", str(skill / "Skill.md"), "--harness", "pi", "--json", str(tmp_path / "h.json")]
    )

    assert result.exit_code == 0, result.output
    rows = next(iter(json.loads((tmp_path / "h.json").read_text())["harness_compatibility"].values()))
    ids = [f["id"] for f in rows[0]["findings"]]
    assert "PI-HIDDEN" in ids  # the selected Skill.md hides the skill; the skill.md sibling would not
    assert "PI-HIDDEN-INVALID" not in ids


def test_pi_hidden_ignores_an_implicit_yaml_merge(tmp_path: Path):
    """pi's YAML parser (eemeli/yaml 2.x) keeps ``<<`` an ordinary key, so a merged
    disable-model-invocation never applies (pi 1.6.0 keeps the skill visible)."""
    skill = tmp_path / "skills" / "demo-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\n"
        "name: demo-skill\n"
        "description: Use when checking harness compatibility.\n"
        "metadata: &defaults\n"
        "  disable-model-invocation: true\n"
        "<<: *defaults\n"
        f"---\n{BODY}",
        encoding="utf-8",
    )
    rows = _rows(skill)

    assert "PI-HIDDEN" not in _ids(rows, "pi")
    assert "PI-HIDDEN-INVALID" not in _ids(rows, "pi")  # the value never reached the top level

    # An explicit top-level key still resolves over the merged copy.
    (skill / "SKILL.md").write_text(
        "---\n"
        "name: demo-skill\n"
        "description: Use when checking harness compatibility.\n"
        "metadata: &defaults\n"
        "  disable-model-invocation: false\n"
        "<<: *defaults\n"
        "disable-model-invocation: true\n"
        f"---\n{BODY}",
        encoding="utf-8",
    )
    assert "PI-HIDDEN" in _ids(_rows(skill), "pi")


def test_duplicate_keys_drop_the_skill_without_an_explicit_invocation_suggestion(tmp_path: Path):
    """Duplicate top-level keys are a parse error for pi's YAML parser: the skill is
    dropped and /skill:name stays unavailable, so PI-HIDDEN must not suggest it."""
    skill = _skill(tmp_path, extra="disable-model-invocation: false\ndisable-model-invocation: true\n")
    rows = _rows(skill)

    assert "PI-FRONTMATTER-DUPKEYS" in _ids(rows, "pi")
    assert rows["pi"]["status"] == "not_loaded"
    assert "PI-HIDDEN" not in _ids(rows, "pi")
    assert "/skill:" not in " ".join(f["message"] for f in rows["pi"]["findings"])


def test_explicit_manifest_next_to_a_canonical_sibling_is_not_discovered(tmp_path: Path):
    """With Skill.md and SKILL.md siblings (case-sensitive filesystem), pi discovers
    only SKILL.md, so an explicit Skill.md check reports the skill as not loaded."""
    skill = tmp_path / "proj" / ".pi" / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Use when testing casing.\n---\nBody\n", encoding="utf-8"
    )
    (skill / "Skill.md").write_text(
        "---\nname: demo\ndescription: Use when testing casing.\ndisable-model-invocation: true\n---\nBody\n",
        encoding="utf-8",
    )
    if (skill / "Skill.md").samefile(skill / "SKILL.md"):
        pytest.skip("case-insensitive filesystem: distinct-cased siblings are one file")

    result = CliRunner().invoke(
        cli, ["harness-check", str(skill / "Skill.md"), "--harness", "pi", "--json", str(tmp_path / "h.json")]
    )
    assert result.exit_code == 0, result.output
    rows = next(iter(json.loads((tmp_path / "h.json").read_text())["harness_compatibility"].values()))
    ids = [f["id"] for f in rows[0]["findings"]]
    assert rows[0]["status"] == "not_loaded"
    assert "PI-CASING" in ids
    assert "PI-HIDDEN" not in ids  # a file pi never discovers cannot be hidden


def test_harness_check_schema_diagnostics_follow_the_selected_manifest(tmp_path: Path):
    """With Skill.md and SKILL.md siblings, the Tier 1 line reports the selected
    file's schema state, not the canonical sibling's."""
    skill = tmp_path / "proj" / ".pi" / "skills" / "demo"
    skill.mkdir(parents=True)
    broken = "---\nname: demo\ndescription: Use when testing [unclosed\n---\nBody\n"
    good = "---\nname: demo\ndescription: Use when testing casing.\n---\n# Demo\nBody\n"
    (skill / "SKILL.md").write_text(broken, encoding="utf-8")
    (skill / "Skill.md").write_text(good, encoding="utf-8")
    if (skill / "Skill.md").samefile(skill / "SKILL.md"):
        pytest.skip("case-insensitive filesystem: distinct-cased siblings are one file")

    result = CliRunner().invoke(cli, ["harness-check", str(skill / "Skill.md"), "--harness", "pi"])
    # The selected Skill.md is valid; the broken canonical sibling must not be reported.
    assert "Tier 1 schema also fails" not in result.output

    (skill / "Skill.md").write_text(broken, encoding="utf-8")
    (skill / "SKILL.md").write_text(good, encoding="utf-8")
    result = CliRunner().invoke(cli, ["harness-check", str(skill / "Skill.md"), "--harness", "pi"])
    assert "Tier 1 schema also fails for Skill.md" in result.output


def test_pi_never_loads_an_uppercase_md_extension(tmp_path: Path):
    """A loose SKILL.MD under .pi/skills casefolds to skill.md, but pi compares the
    extension exactly and never discovers it."""
    root = tmp_path / "proj" / ".pi" / "skills"
    root.mkdir(parents=True)
    (root / "SKILL.MD").write_text(
        "---\nname: demo\ndescription: Use when testing extensions.\n---\nBody\n", encoding="utf-8"
    )

    result = CliRunner().invoke(
        cli, ["harness-check", str(root / "SKILL.MD"), "--harness", "pi", "--json", str(tmp_path / "h.json")]
    )
    assert result.exit_code == 0, result.output
    rows = next(iter(json.loads((tmp_path / "h.json").read_text())["harness_compatibility"].values()))
    assert rows[0]["status"] == "not_loaded"
    assert "PI-CASING" in [f["id"] for f in rows[0]["findings"]]


def test_pi_collision_check_enumerates_loose_root_manifests(tmp_path: Path):
    """pi discovers aaa.md and skill.md as two loose skills in .pi/skills; checking
    the exact skill.md path reports the name collision between them."""
    root = tmp_path / "proj" / ".pi" / "skills"
    root.mkdir(parents=True)
    for name in ("aaa.md", "skill.md"):
        (root / name).write_text(
            f"---\nname: demo\ndescription: Use when testing duplicates ({name}).\n---\nBody\n",
            encoding="utf-8",
        )

    result = CliRunner().invoke(
        cli, ["harness-check", str(root / "skill.md"), "--harness", "pi", "--json", str(tmp_path / "h.json")]
    )
    assert result.exit_code == 0, result.output
    rows = next(iter(json.loads((tmp_path / "h.json").read_text())["harness_compatibility"].values()))
    collision = next(f for f in rows[0]["findings"] if f["id"] == "PI-NAME-COLLISION")
    assert "aaa.md" in collision["message"]

    # A named child SKILL.md next to the loose manifest collides the same way.
    nested = root / "aaa"
    nested.mkdir()
    (nested / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Use when testing duplicates (nested).\n---\nBody\n", encoding="utf-8"
    )
    result = CliRunner().invoke(
        cli, ["harness-check", str(root / "skill.md"), "--harness", "pi", "--json", str(tmp_path / "h2.json")]
    )
    rows = next(iter(json.loads((tmp_path / "h2.json").read_text())["harness_compatibility"].values()))
    collision = next(f for f in rows[0]["findings"] if f["id"] == "PI-NAME-COLLISION")
    assert "aaa.md" in collision["message"] and "aaa/SKILL.md" in collision["message"]


def test_hidden_pi_skills_leave_the_model_list_empty(tmp_path: Path):
    """pi omits a disable-model-invocation skill from the model list: the listed
    field is empty even though the description is present."""
    skill = _skill(tmp_path, extra="disable-model-invocation: true\n")
    rows = _rows(skill)

    assert rows["pi"]["listed"] == {"mode": "hidden", "text": ""}
    assert "PI-HIDDEN" in _ids(rows, "pi")  # explicit /skill:name invocation is still reported


def test_pi_hidden_keeps_explicit_tag_semantics(tmp_path: Path):
    """An explicit !!str tag keeps the value a string (pi stays visible), while an
    explicitly tagged !!merge key resolves its mapping and hides the skill."""
    skill = tmp_path / "skills" / "demo-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\n"
        "name: demo-skill\n"
        "description: Use when checking harness compatibility.\n"
        "disable-model-invocation: !!str true\n"
        f"---\n{BODY}",
        encoding="utf-8",
    )
    rows = _rows(skill)

    assert "PI-HIDDEN" not in _ids(rows, "pi")  # pi keeps the string value visible
    assert "PI-HIDDEN-INVALID" in _ids(rows, "pi")  # other parsers read a truthy string

    (skill / "SKILL.md").write_text(
        "---\n"
        "name: demo-skill\n"
        "description: Use when checking harness compatibility.\n"
        "metadata: &defaults\n"
        "  disable-model-invocation: true\n"
        "!!merge <<: *defaults\n"
        f"---\n{BODY}",
        encoding="utf-8",
    )
    rows = _rows(skill)

    assert "PI-HIDDEN" in _ids(rows, "pi")  # an explicit !!merge resolves in pi's parser
    assert "PI-HIDDEN-INVALID" not in _ids(rows, "pi")


def test_hidden_listing_mode_renders_in_verbose_reports(tmp_path: Path):
    """The verbose entry points must render the hidden listing mode instead of crashing (KeyError)."""
    skill = _skill(tmp_path, extra="disable-model-invocation: true\n")

    result = CliRunner().invoke(
        cli, ["harness-check", str(skill), "--harness", "pi", "--verbose", "--json", str(tmp_path / "h.json")]
    )
    assert result.exit_code == 0, result.output
    assert "hidden (not in the model list)" in result.output

    result = CliRunner().invoke(
        cli, ["validate", str(skill), "--tiers", "1", "--checks", "harness", "--harness", "pi", "--verbose"]
    )
    assert result.exit_code == 0, result.output
    assert "hidden (not in the model list)" in result.output


_LISTING_MODES = ("cut", "name_only", "dropped", "hidden", "shortened")  # "full" is skipped by the grid
_LISTING_INPUTS = {
    "cut": ({"listing": {"cut": 10}}, {}),
    "dropped": ({"listing": {"drop_over_utf16": 10}}, {}),
    "hidden": ({"listing": {"hidden_when": "pi_disable_model_invocation"}}, {"pi_disable_model_invocation": True}),
    "name_only": ({"listing": {}}, {"description_text": ""}),
    "shortened": ({"listing": {"budget": {"desc_chars": 10}}}, {}),
}


@pytest.mark.parametrize("mode", _LISTING_MODES)
def test_every_listing_mode_renders_in_verbose_panel(tmp_path: Path, mode: str):
    """Both verbose entry points share harness_panel(verbose=True): every mode the facts
    layer can emit must have a _LISTING entry there (r9 shipped `hidden` and crashed both)."""
    profile, fact_overrides = _LISTING_INPUTS[mode]
    facts = {
        "description_text": "Use when checking harness compatibility.",
        "description_utf16_units": 44,
        "catalog_size": 1,
        "catalog_desc_chars": 44,
        **fact_overrides,
    }
    listed = listed_description(profile, facts)
    assert listed["mode"] == mode

    block = harness_block(run_validation(_skill(tmp_path), checks="harness"))
    rows = next(rows for rows in block.values() if any(r["harness"] == "pi" for r in rows))
    next(r for r in rows if r["harness"] == "pi")["listed"] = listed
    assert _LISTING[mode].format(limit=listed.get("limit", "")) in _render(harness_panel(block, verbose=True))


def test_harness_check_ignores_an_explicit_non_manifest_file(tmp_path: Path):
    """Passing a README.md sibling to harness-check keeps the canonical skill's row
    and its schema diagnostics; the README is not a manifest any harness selects."""
    skill = tmp_path / "proj" / ".pi" / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Use when testing casing.\n---\n# Demo\nBody\n", encoding="utf-8"
    )
    (skill / "README.md").write_text("---\nname: broken: [unclosed\n---\nBody\n", encoding="utf-8")

    result = CliRunner().invoke(
        cli, ["harness-check", str(skill / "README.md"), "--harness", "pi", "--json", str(tmp_path / "h.json")]
    )
    assert result.exit_code == 0, result.output
    assert "Tier 1 schema also fails" not in result.output
    assert "README" not in result.output
    rows = next(iter(json.loads((tmp_path / "h.json").read_text())["harness_compatibility"].values()))
    assert rows[0]["status"] == "loads"  # the canonical SKILL.md row


def test_pi_census_counts_only_files_pi_loads(tmp_path: Path):
    """Files the root's ignore rules exclude, or loose files without a description,
    never load in pi, so they cannot collide with the valid skill."""
    root = tmp_path / "proj" / ".pi" / "skills"
    root.mkdir(parents=True)
    (root / ".gitignore").write_text("ignored.md\n", encoding="utf-8")
    (root / "ignored.md").write_text(
        "---\nname: demo\ndescription: Use when testing collisions.\n---\nB\n", encoding="utf-8"
    )
    (root / "skill.md").write_text(
        "---\nname: demo\ndescription: Use when testing collisions.\n---\nB\n", encoding="utf-8"
    )
    result = CliRunner().invoke(
        cli, ["harness-check", str(root / "skill.md"), "--harness", "pi", "--json", str(tmp_path / "h.json")]
    )
    assert result.exit_code == 0, result.output
    rows = next(iter(json.loads((tmp_path / "h.json").read_text())["harness_compatibility"].values()))
    assert "PI-NAME-COLLISION" not in [f["id"] for f in rows[0]["findings"]]

    (root / ".gitignore").unlink()
    (root / "ignored.md").unlink()
    (root / "nodesc.md").write_text("---\nname: demo\n---\nB\n", encoding="utf-8")
    result = CliRunner().invoke(
        cli, ["harness-check", str(root / "skill.md"), "--harness", "pi", "--json", str(tmp_path / "h2.json")]
    )
    assert result.exit_code == 0, result.output
    rows = next(iter(json.loads((tmp_path / "h2.json").read_text())["harness_compatibility"].values()))
    assert "PI-NAME-COLLISION" not in [f["id"] for f in rows[0]["findings"]]


def test_pi_census_includes_nested_agents_root_manifests(tmp_path: Path):
    """pi loads loose .md files nested under .agents/skills; two same-named ones collide."""
    folder = tmp_path / "proj" / ".agents" / "skills" / "demo"
    folder.mkdir(parents=True)
    for name in ("aaa.md", "skill.md"):
        (folder / name).write_text(
            f"---\nname: demo\ndescription: Use when testing collisions ({name}).\n---\nB\n", encoding="utf-8"
        )

    result = CliRunner().invoke(
        cli,
        ["harness-check", str(folder / "skill.md"), "--harness", "pi", "--json", str(tmp_path / "h.json")],
    )
    assert result.exit_code == 0, result.output
    rows = next(iter(json.loads((tmp_path / "h.json").read_text())["harness_compatibility"].values()))
    collision = next(f for f in rows[0]["findings"] if f["id"] == "PI-NAME-COLLISION")
    assert "aaa.md" in collision["message"]


def test_codex_policy_makes_disable_model_invocation_effective_in_codex(tmp_path: Path):
    skill = _skill(tmp_path, extra="disable-model-invocation: true\n")
    (skill / "agents").mkdir()
    (skill / "agents" / "openai.yaml").write_text("policy:\n  allow_implicit_invocation: false\n")
    rows = _rows(skill)

    flagged = {h for h, r in rows.items() if any(f["id"].endswith("DMI-IGNORED") for f in r["findings"])}
    assert flagged == {"hermes", "opencode"}
    assert not any(f["id"].endswith("CODEX-POLICY") for r in rows.values() for f in r["findings"])
    assert rows["codex"]["status"] == "loads"


def test_codex_implicit_policy_is_codex_only(tmp_path: Path):
    skill = _skill(tmp_path)
    (skill / "agents").mkdir()
    (skill / "agents" / "openai.yaml").write_text("policy:\n  allow_implicit_invocation: false\n")
    rows = _rows(skill)

    flagged = {h for h, r in rows.items() if any(f["id"].endswith("CODEX-POLICY") for f in r["findings"])}
    assert flagged == set(HARNESSES) - {"codex"}  # pi has no implicit-invocation policy mechanism either
    # pi honors disable-model-invocation, so the fix names that setting.
    finding = next(f for f in rows["pi"]["findings"] if f["id"] == "PI-CODEX-POLICY")
    assert finding["fix"] == (
        "Also set disable-model-invocation: true (pi hides the skill from the model; the user can still run /skill:name)."
    )


def test_claude_placeholders_reach_other_harnesses_literally(tmp_path: Path):
    rows = _rows(_skill(tmp_path, body=BODY + "Run ${CLAUDE_SKILL_DIR}/run.sh $ARGUMENTS\n"))

    for harness in set(HARNESSES) - {"claude-code"}:  # pi substitutes neither placeholder
        assert any(f["id"].endswith("-ARGUMENTS") for f in rows[harness]["findings"]), harness
        assert any(f["id"].endswith("-SKILL-DIR") for f in rows[harness]["findings"]), harness
    assert not rows["claude-code"]["findings"]


def test_gating_and_claude_only_fields(tmp_path: Path):
    extra = 'argument-hint: "<file>"\nplatforms: [macos]\n'
    skill = _skill(tmp_path, extra=extra)
    manifest = skill / "SKILL.md"
    manifest.write_text(
        manifest.read_text().replace("  author:", '  openclaw: {"requires": {"bins": ["pdftotext"]}}\n  author:')
    )
    rows = _rows(skill)

    assert "HM-PLATFORMS" in _ids(rows, "hermes")
    assert "Left out of the skill list unless the host has: bins pdftotext." in {
        f["message"] for f in rows["openclaw"]["findings"]
    }
    assert "OC-CLAUDE-ONLY" in _ids(rows, "opencode")
    assert not any(f["id"].endswith("CLAUDE-ONLY") for f in rows["claude-code"]["findings"])


def test_every_rule_has_a_fix_and_placeholders_are_filled(tmp_path: Path):
    for profile in load_profiles("all"):
        for rule in profile["rules"]:
            assert rule["when"] == "always" or rule.get("fix"), rule["id"]
    skills = _catalog(tmp_path, 200, desc_len=400)
    rows = _rows(skills / "filler-000")
    messages = [f["message"] + f.get("fix", "") for r in rows.values() for f in r["findings"]]
    assert messages and not any(re.search(r"\{[a-z0-9_]+\}", m) for m in messages)
    assert {"OW-COMPACT", "OW-PROMPT-CAP"} <= _ids(rows, "openclaw")


def test_rule_ids_named_in_messages_exist():
    profiles = load_profiles("all")
    known = {rule["id"] for profile in profiles for rule in profile["rules"]}
    prefixes = "|".join(sorted({rule_id.split("-")[0] for rule_id in known}))
    rule_id_pattern = rf"\b(?:{prefixes})-[A-Z0-9]+(?:-[A-Z0-9]+)*"
    for profile in profiles:
        for rule in profile["rules"]:
            text = f"{rule['message']} {rule.get('fix', '')}"
            for mentioned in re.findall(rule_id_pattern, text):
                assert mentioned in known, f"{rule['id']} refers to {mentioned}, which no profile defines"


def test_fix_reaches_findings_and_panels(tmp_path: Path):
    skill = _skill(tmp_path, desc="Use when testing emoji. " + "\U0001f600" * 600)
    result = HarnessValidator().validate(skill)
    finding = next(f for f in result.findings if f.check_name == "CP-DESC-LIMIT")

    assert finding.suggestion == "Cut the description by at least 200 UTF-16 units."
    out = CliRunner().invoke(cli, ["harness-check", str(skill), "-v"], env={"COLUMNS": "120"}).output
    assert "→ Cut the description by at least 200 UTF-16 units." in out
    assert "Description in the model's skill list" in out


def test_hermes_platforms_listing_every_os_is_not_flagged(tmp_path: Path):
    everywhere = _rows(_skill(tmp_path / "a", extra="platforms: [linux, macos, windows]\n"))
    mac_only = _rows(_skill(tmp_path / "b", extra="platforms: macos\n"))

    assert "HM-PLATFORMS" not in _ids(everywhere, "hermes")
    assert "Hidden from the skill list on linux, windows (platforms: macos)." in {
        f["message"] for f in mac_only["hermes"]["findings"]
    }


def test_pi_drops_tab_indented_frontmatter(tmp_path: Path):
    skill = _skill(tmp_path)
    (skill / "SKILL.md").write_text("---\nname: demo-skill\nmetadata:\n  \tauthor: x\n---\nBody\n")

    rows = _rows(skill)

    assert rows["pi"]["status"] == "not_loaded"
    assert "PI-FRONTMATTER-TABS" in _ids(rows, "pi")


def test_pi_drops_skills_without_frontmatter_or_description(tmp_path: Path):
    bare = tmp_path / "skills" / "bare"
    bare.mkdir(parents=True)
    (bare / "SKILL.md").write_text("No frontmatter, just a body.\n")
    blank = _skill(tmp_path / "blank", desc='"   "')

    assert _rows(bare)["pi"]["status"] == "not_loaded"
    assert "PI-UNPARSEABLE" in _ids(_rows(bare), "pi")
    assert _rows(blank)["pi"]["status"] == "not_loaded"
    assert "PI-DESC-MISSING" in _ids(_rows(blank), "pi")


@pytest.mark.parametrize("target, flagged", [(1024, False), (1025, True)])
def test_pi_description_length_boundary(tmp_path: Path, target: int, flagged: bool):
    rows = _rows(_skill(tmp_path / str(target), desc="a" * target))

    assert ("PI-DESC-LONG" in _ids(rows, "pi")) is flagged
    assert rows["pi"]["status"] == "loads"


def test_pi_hidden_disable_model_invocation(tmp_path: Path):
    rows = _rows(_skill(tmp_path, extra="disable-model-invocation: true\n"))

    assert rows["pi"]["status"] == "degraded"
    assert "PI-HIDDEN" in _ids(rows, "pi")


def test_pi_project_root_loads_for_pi_and_gates_the_others(tmp_path: Path):
    rows = _rows(_skill(tmp_path / "proj" / ".pi" / "skills"))

    assert rows["pi"]["status"] == "loads"
    assert "PI-TRUST" in _ids(rows, "pi")  # project roots load only once trusted
    assert rows["claude-code"]["status"] == "not_loaded"
    assert "CC-ROOT" in _ids(rows, "claude-code")


def _loose_skill(root: Path) -> Path:
    """A non-exact manifest sitting directly in a skills root (pi mode "pi")."""
    root.mkdir(parents=True)
    (root / "skill.md").write_text(
        "---\nname: demo\ndescription: Use when testing casing.\n---\nBody\n", encoding="utf-8"
    )
    return root


def test_pi_casing_follows_the_root_family_and_depth(tmp_path: Path):
    # Nested (collectSkillEntries mode "agents"): any .md manifest name loads.
    agents_skill = _skill(tmp_path / "proj" / ".agents")
    (agents_skill / "SKILL.md").rename(agents_skill / "skill.md")  # rename, not copy: case-insensitive FS
    rows = _rows(agents_skill)
    assert rows["pi"]["status"] == "loads"
    assert "PI-CASING" not in _ids(rows, "pi")

    # Nested under a .pi root: only the exact entry name loads.
    dotpi_skill = _skill(tmp_path / "proj2" / ".pi" / "skills")
    (dotpi_skill / "SKILL.md").rename(dotpi_skill / "skill.md")
    rows = _rows(dotpi_skill)
    assert rows["pi"]["status"] == "not_loaded"
    assert "PI-CASING" in _ids(rows, "pi")

    # Loose at .agents/skills (collectSkillEntries mode "agents", dir === root): a loose
    # .md file directly in the root is not discovered (verified against the pinned
    # DefaultResourceLoader); nested one folder down any .md name loads, and the
    # census harness-check counts follows the same rule.
    # Sibling control: a working named child skill next to the loose manifest
    # keeps loading.
    agents_root = tmp_path / "proj3" / ".agents" / "skills"
    loose = _loose_skill(agents_root)
    sky = agents_root / "sky"
    sky.mkdir(parents=True)
    (sky / "SKILL.md").write_text(
        "---\nname: sky\ndescription: Use when testing siblings.\n---\nBody\n", encoding="utf-8"
    )
    nested = agents_root / "nested"
    nested.mkdir()
    (nested / "alpha.md").write_text(
        "---\nname: alpha\ndescription: Use when testing nesting.\n---\nBody\n", encoding="utf-8"
    )
    rows = _rows(loose)
    assert rows["pi"]["status"] == "not_loaded"
    assert "PI-CASING" in _ids(rows, "pi")
    rows = _rows(sky)
    assert rows["pi"]["status"] == "loads"
    assert "PI-CASING" not in _ids(rows, "pi")
    assert sorted(p.relative_to(agents_root).as_posix() for p in overlay._pi_manifests(agents_root)) == [
        "nested/alpha.md",
        "sky/SKILL.md",
    ]

    # Loose at a .pi root: mode "pi" accepts any .md name directly in the root.
    rows = _rows(_loose_skill(tmp_path / "proj4" / ".pi" / "skills"))
    assert rows["pi"]["status"] == "loads"
    assert "PI-CASING" not in _ids(rows, "pi")


def test_claude_root_is_not_a_pi_root(tmp_path: Path):
    rows = _rows(_skill(tmp_path / "proj" / ".claude" / "skills"))

    assert rows["pi"]["status"] == "not_loaded"
    assert "PI-ROOT" in _ids(rows, "pi")


def test_pi_drops_non_string_descriptions(tmp_path: Path):
    rows = _rows(_skill(tmp_path / "numeric", desc="42"))

    assert rows["pi"]["status"] == "not_loaded"
    assert "PI-DESC-MISSING" in _ids(rows, "pi")


def test_pi_name_collision_is_a_warning_not_a_drop(tmp_path: Path):
    skills = tmp_path / "skills"
    for folder in ("first", "second"):
        skill = skills / folder
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            f"---\nname: same\ndescription: Use when testing collisions.\n---\n{BODY}", encoding="utf-8"
        )
    first = _rows(skills / "first")
    second = _rows(skills / "second")

    # The loader keeps the first skill it scans; the check cannot know the scan
    # order, so both sides report the collision as degraded, never as loaded.
    assert "PI-NAME-COLLISION" in _ids(first, "pi")
    assert "PI-NAME-COLLISION" in _ids(second, "pi")
    assert first["pi"]["status"] == second["pi"]["status"] == "degraded"


def test_pi_name_falls_back_to_the_folder(tmp_path: Path):
    skill = tmp_path / "skills" / "folder-name"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\ndescription: Use when testing name fallback.\n---\nBody\n")

    rows = _rows(skill)

    assert "PI-NAME-FALLBACK" in _ids(rows, "pi")
    assert rows["pi"]["status"] == "loads"


def _sized(root: Path, fact: str, target: int) -> Path:
    """A skill whose ``fact`` (a body-length measure) is exactly ``target``."""
    probe = _skill(root / "probe", body="a")
    padding = target - skill_facts(probe)[fact] + 1
    skill = _skill(root / str(target), body="a" * padding)
    assert skill_facts(skill)[fact] == target
    return skill


@pytest.mark.parametrize(
    ("harness", "rule", "fact", "limit"),
    [
        ("hermes", "HM-VIEW-BUDGET", "hermes_view_chars", 100_000),
        ("openclaw", "OW-READ-BUDGET", "openclaw_read_chars", 16_000),
        ("openclaw", "OW-READ-REFUSED", "openclaw_read_chars", 64_000),
        ("codex", "CX-SHELL-READ", "skill_md_bytes", 40_000),
        ("opencode", "OC-BODY-CUT", "opencode_body_output_bytes", 51_200),
    ],
)
def test_body_limits_fire_just_over_the_limit(tmp_path: Path, harness: str, rule: str, fact: str, limit: int):
    assert rule not in _ids(_rows(_sized(tmp_path / "at", fact, limit)), harness)
    assert rule in _ids(_rows(_sized(tmp_path / "over", fact, limit + 1)), harness)


@pytest.mark.parametrize(("harness", "rule"), [("codex", "CX-DESC-CUT"), ("copilot-cli", "CP-DESC-LIMIT")])
def test_description_limit_is_1024(tmp_path: Path, harness: str, rule: str):
    at = _rows(_skill(tmp_path / "at", desc="Use when " + "a" * 1015))
    over = _rows(_skill(tmp_path / "over", desc="Use when " + "a" * 1016))

    assert rule not in _ids(at, harness)
    assert any(f["id"] == rule and "1 over" in f["message"] for f in over[harness]["findings"])


def test_price_like_dollar_numbers_are_flagged_with_lines(tmp_path: Path):
    prices = _rows(_skill(tmp_path / "a", body="# Prices\n\nInput $4.00, cache $0.25.\nBatch $4 / $20 per MTok.\n"))
    safe = _rows(_skill(tmp_path / "b", body="Input USD 4.00, \\$0.25, $10k, $ARGUMENTS.\nRun on $1.\n"))

    finding = next(f for f in prices["claude-code"]["findings"] if f["id"] == "CC-DOLLAR-N")
    assert finding["message"].startswith("4 price-like $N tokens (lines 9, 10)")
    assert "OC-DOLLAR-N" in _ids(prices, "opencode")
    assert "CC-DOLLAR-N" not in _ids(safe, "claude-code") and "OC-DOLLAR-N" not in _ids(safe, "opencode")
    assert skill_facts(tmp_path / "b" / "skills" / "demo-skill")["dollar_placeholder_count"] == 1


@pytest.mark.parametrize(
    ("line", "prices", "placeholders"),
    [
        ("Budget is $5.", 1, 0),
        ("The plan costs $20", 1, 0),
        ("A $20 plan for the team (about $50).", 2, 0),
        ("Batch $3 / $15 per MTok, or $5/month.", 3, 0),
        ("Run $1 /path/to/project", 0, 1),
        ("cp $1/$2 ~/notes; cd /tmp/$3", 0, 3),
        ("Copy $5 /tmp/out and $6 ~/notes", 0, 2),
        ("awk '{print $5}' and \"$9\"", 0, 2),
        ("Use $0, $1 and $2 as the inputs.", 0, 3),
    ],
)
def test_dollar_n_is_a_price_or_an_argument_not_both(tmp_path: Path, line: str, prices: int, placeholders: int):
    facts = skill_facts(_skill(tmp_path, body=f"{BODY}{line}\n"))

    assert (facts["dollar_literal_count"], facts["dollar_placeholder_count"]) == (prices, placeholders), line


def test_standalone_prices_are_flagged_and_path_arguments_are_not(tmp_path: Path):
    prices = _rows(_skill(tmp_path / "a", body=BODY + "The plan costs $20, with a $5 credit.\n"))
    args = _rows(_skill(tmp_path / "b", body=BODY + "Run $1 /path/to/project, then cp $1/$2 out.\n"))

    for harness, rule in (("claude-code", "CC-DOLLAR-N"), ("opencode", "OC-DOLLAR-N")):
        assert any(f["id"] == rule and "2 price-like" in f["message"] for f in prices[harness]["findings"]), harness
        assert rule not in _ids(args, harness), harness


def test_every_oversized_file_long_line_and_unlisted_folder_is_counted(tmp_path: Path):
    skill = _skill(tmp_path)
    for rel, size in {"references/a.md": 51_201, "shared/b.md": 64_001, "shared/c.md": 51_200}.items():
        (skill / rel).parent.mkdir(parents=True, exist_ok=True)
        (skill / rel).write_text("x" * size, encoding="utf-8")
    (skill / "python").mkdir()
    (skill / "python" / "wide.md").write_text("y" * 2001 + "\n" + "z" * 2000 + "\n", encoding="utf-8")
    facts = skill_facts(skill)
    rows = _rows(skill)

    assert (facts["files_over_51200"], facts["files_over_64000"]) == (2, 1)
    assert (facts["long_line_count"], facts["hermes_unlisted_files"]) == (4, 3)
    assert facts["hermes_unlisted_dirs"] == "python, shared"
    assert {"OC-REF-CUT", "OC-LONG-LINES"} <= _ids(rows, "opencode")
    assert {"HM-LINKED-LAYOUT"} <= _ids(rows, "hermes") and "OW-REF-BUDGET" in _ids(rows, "openclaw")


def test_unreadable_resources_are_skipped_not_raised(tmp_path: Path):
    skill = _skill(tmp_path)
    refs = skill / "references"
    refs.mkdir()
    (refs / "ok.md").write_text("x" * 100)
    (refs / "locked.md").write_text("secret")
    (refs / "locked.md").chmod(0)
    (refs / "dangling.md").symlink_to(tmp_path / "missing.md")
    (refs / "loop").symlink_to(refs / "loop")
    os.mkfifo(refs / "pipe")  # reading a pipe would block forever

    facts = skill_facts(skill)
    result = HarnessValidator().validate(skill)

    assert facts["largest_resource"] == "references/ok.md"
    assert facts["resource_symlinks"] == 2
    assert result.metadata["harness_compatibility"][str(skill)]


def test_symlinked_reference_outside_the_skill_still_needs_paging(tmp_path: Path):
    skill = _skill(tmp_path)
    (skill / "references").mkdir()
    shared = tmp_path / "shared.md"
    shared.write_text("line\n" * 20_000 + "w" * 2_001 + "\n")
    (skill / "references" / "shared.md").symlink_to(shared)
    (skill / "real").mkdir()
    (skill / "real" / "big.md").write_text("line\n" * 20_000)
    (skill / "references" / "big.md").symlink_to(skill / "real" / "big.md")
    facts = skill_facts(skill)
    rows = _rows(skill)

    assert facts["resource_symlinks"] == 2 and facts["resource_symlink_escapes"] == 1
    # The link into the skill is counted once, under its real path; the one leaving it is read through the link.
    assert facts["files_over_51200_paths"] == "real/big.md, references/shared.md"
    assert facts["long_line_files"] == "references/shared.md"
    assert {"OC-REF-CUT", "OC-LONG-LINES"} <= _ids(rows, "opencode")
    # OpenClaw is not known to read through links, so its multi-read count stays on files inside the skill.
    assert facts["files_over_64000_paths"] == "real/big.md" and "OW-REF-BUDGET" in _ids(rows, "openclaw")


def test_binary_resources_are_not_read_as_text(tmp_path: Path):
    skill = _skill(tmp_path)
    (skill / "assets").mkdir()
    (skill / "assets" / "padded.bin").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 5_000 + bytes(range(256)))
    (skill / "assets" / "photo.png").write_bytes(bytes(range(256)) * 400)
    facts = skill_facts(skill)
    rows = _rows(skill)

    assert (facts["long_line_count"], facts["files_over_51200"], facts["files_over_64000"]) == (0, 0, 0)
    assert facts["largest_resource"] == ""
    assert not {"OC-REF-CUT", "OC-LONG-LINES"} & _ids(rows, "opencode")
    assert "OW-REF-BUDGET" not in _ids(rows, "openclaw")


def test_compact_view_shows_worst_finding_per_harness_and_catalog(tmp_path: Path):
    _catalog(tmp_path, 40)
    block = harness_block(run_validation(tmp_path / ".claude" / "skills" / "filler-000", checks="harness"))
    compact, full = _render(harness_panel(block)), _render(harness_panel(block, verbose=True))

    assert "catalog  40 skills in" in compact and "(12,000 desc" in compact
    assert "not found" in compact and "listing shortened" in compact
    assert "[CX-ROOT]" not in compact and "[CX-ROOT]" in full and "[CX-LIST-BUDGET]" in full
    assert len(compact.splitlines()) < len(full.splitlines())


def test_tier1_summary_gives_harness_its_own_line(tmp_path: Path):
    results = run_validation(_rooted(tmp_path), checks="schema,harness", harness="all")
    _, rows = summarize_tier1(results)
    labels = {row.label: "".join(text for text, _ in row.segments) for row in rows}

    assert labels["harness"].startswith("1 not loaded")
    assert "harness" not in labels.get("advisories", "")


def test_quotes_in_description_are_escaped_in_copilot_and_openclaw_lists(tmp_path: Path):
    rows = _rows(_skill(tmp_path, desc='Use when asked to "summarize" a file.'))
    assert "CP-DESC-ESCAPED" in _ids(rows, "copilot-cli") and "OW-DESC-ESCAPED" in _ids(rows, "openclaw")
    assert "CC-DESC-ESCAPED" not in _ids(rows, "claude-code") and rows["copilot-cli"]["status"] == "loads"


def _rule_ids(skill: Path, harness: str, config: dict | None = None) -> set[str]:
    return {f["id"] for r in evaluate_harnesses(skill, harness, config=config) for f in r.findings}


def test_openclaw_read_budget_depends_on_model_context(tmp_path: Path):
    mid = _skill(tmp_path / "mid", body="x" * 20_000)
    big = _skill(tmp_path / "big", body="x" * 40_000)

    assert "OW-READ-BUDGET" in _rule_ids(mid, "openclaw")
    assert "OW-READ-BUDGET" not in _rule_ids(mid, "openclaw", {"context_tokens": 150_000})
    assert "OW-READ-BUDGET" not in _rule_ids(mid, "openclaw", {"custom_provider": True})
    assert "OW-READ-BUDGET" in _rule_ids(big, "openclaw", {"custom_provider": True})
    assert "OW-READ-BUDGET" not in _rule_ids(big, "openclaw", {"context_tokens": 200_000})


def test_codex_shell_cut_is_lower_for_custom_providers(tmp_path: Path):
    skill = _skill(tmp_path, body="x" * 12_000)

    assert "CX-SHELL-READ" not in _rule_ids(skill, "codex")
    assert "CX-SHELL-READ" in _rule_ids(skill, "codex", {"custom_provider": True})


def test_codex_explicit_invocation_delivers_whole(tmp_path: Path):
    skill = _skill(tmp_path, body="x" * 50_000)

    assert "CX-SHELL-READ" in _rule_ids(skill, "codex", {"custom_provider": True})
    assert "CX-SHELL-READ" not in _rule_ids(skill, "codex", {"custom_provider": True, "explicit_invocation": True})


def test_harness_check_accepts_model_config(tmp_path: Path):
    skill = _skill(tmp_path, body="x" * 20_000)
    out = tmp_path / "out.json"

    result = CliRunner().invoke(
        cli, ["harness-check", str(skill), "--harness", "openclaw", "--context-tokens", "150000", "--json", str(out)]
    )

    assert result.exit_code == 0, result.output
    assert "OW-READ-BUDGET" not in out.read_text()


def test_openclaw_read_budget_weights_cjk_not_utf8_bytes(tmp_path: Path):
    kana = _skill(tmp_path / "kana", body="\u3042" * 8_000)
    cyrillic = _skill(tmp_path / "cyr", body="\u0430" * 20_000)

    assert skill_facts(kana)["skill_md_bytes"] < 32_000 < skill_facts(cyrillic)["skill_md_bytes"]
    assert "OW-READ-BUDGET" in _rule_ids(kana, "openclaw", {"custom_provider": True})
    assert "OW-READ-BUDGET" not in _rule_ids(cyrillic, "openclaw", {"custom_provider": True})


def test_openclaw_read_chars_counts_code_units(tmp_path: Path):
    flat = skill_facts(_skill(tmp_path / "flat", body="a" * 1_000))["openclaw_read_chars"]
    lines = skill_facts(_skill(tmp_path / "lines", body="a\n" * 500))["openclaw_read_chars"]
    emoji = skill_facts(_skill(tmp_path / "emoji", body="\U0001f600" * 500))["openclaw_read_chars"]

    assert lines == flat == emoji


def test_hermes_view_budget_counts_code_points(tmp_path: Path):
    skill = _skill(tmp_path, body="\u65c5" * 60_000)

    assert "HM-VIEW-BUDGET" not in _rule_ids(skill, "hermes")


def test_claude_allowed_tools_blocks_headless_load(tmp_path: Path):
    skill = _skill(tmp_path, extra="allowed-tools: Read\n")

    assert "CC-ALLOWED-TOOLS-HEADLESS" not in _rule_ids(skill, "claude-code")
    assert "CC-ALLOWED-TOOLS-HEADLESS" in _rule_ids(skill, "claude-code", {"headless": True})


def test_lookalike_descriptions_are_flagged(tmp_path: Path):
    def write(name: str, description: str) -> Path:
        skill = tmp_path / ".agents" / "skills" / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n# {name}\n")
        return skill

    same = (
        "Generate PDF reports from spreadsheet data. Use when the user asks for a report, summary PDF, or chart export."
    )
    builder = write("report-builder", same)
    write("report-generator", same.lower().replace(", or", " or"))
    hr = write("hr-headcount", "Count employees per department from the HR roster. Use for headcount questions.")
    write("sales-pipeline", "Summarize open deals by stage from the CRM export. Use for sales pipeline questions.")

    facts = facts_for([builder, hr])
    assert facts[builder]["lookalike_names"] == "report-generator"
    assert facts[hr]["lookalike_names"] == ""
    finding = next(f for r in evaluate_harnesses(builder, "codex") for f in r.findings if f["id"] == "CX-LOOKALIKE")
    assert finding["effect"] == "advisory"
    assert "report-generator" in finding["message"]


def _lookalikes(tmp_path: Path, first: str, second: str) -> str:
    skills = []
    for name, description in (("skill-one", first), ("skill-two", second)):
        skill = tmp_path / ".agents" / "skills" / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n# {name}\n", "utf-8")
        skills.append(skill)
    return facts_for(skills)[skills[0]]["lookalike_names"]


@pytest.mark.parametrize(
    "description",
    [
        "Создаёт отчёты сотрудников по отделам из таблиц компании",
        "従業員の部署別レポートを会社の表データから作成します。人事の質問に使います",
        "为公司表格中的员工生成按部门划分的报告\uff0c用于人事问题",
    ],
)
def test_lookalike_non_latin_descriptions_are_flagged(tmp_path: Path, description: str):
    assert _lookalikes(tmp_path, description, description) == "skill-two"


def test_lookalike_different_non_latin_descriptions_are_not_flagged(tmp_path: Path):
    russian = _lookalikes(
        tmp_path / "ru",
        "Создаёт отчёты сотрудников по отделам из таблиц компании",
        "Считает остатки товаров на складе по артикулам из выгрузки учёта",
    )
    japanese = _lookalikes(
        tmp_path / "ja",
        "従業員の部署別レポートを会社の表データから作成します。人事の質問に使います",
        "商品の在庫数を倉庫ごとに集計します。物流の質問に使います。出荷の遅れも確認します",
    )

    assert russian == japanese == ""


def test_lookalike_mixed_language_descriptions_keep_distinguishing_words(tmp_path: Path):
    hr = "Generate PDF reports from spreadsheet data for 人事部門."
    sales = "Generate PDF reports from spreadsheet data for 営業部門."

    assert _lookalikes(tmp_path / "different", hr, sales) == ""
    assert _lookalikes(tmp_path / "same", hr, hr) == "skill-two"


def test_description_words_keep_unicode_letters_and_ascii_behavior():
    assert _description_words("Use when PDF_reports meet v2, don't") == {"pdf", "reports", "meet", "v2", "don", "t"}
    assert _description_words("Straße CAFÉ") == _description_words("strasse café") == {"strasse", "café"}
    assert _description_words("кра\u0438\u0306") == _description_words("кра\u0439") == {"край"}
    assert _description_words("हिन्दी भाषा") == {"हिन्दी", "भाषा"}


def test_claude_ignored_field_values_are_noted(tmp_path: Path):
    bad = _skill(tmp_path / "bad", extra="effort: extreme\nshell: zsh\nbackground: true\nmaxTurns: 5\n")
    good = _skill(tmp_path / "good", extra="effort: XHigh\nshell: powershell\ncontext: fork\nbackground: true\n")
    numeric = _skill(tmp_path / "numeric", extra="effort: 3\n")

    finding = next(f for r in evaluate_harnesses(bad, "claude-code") for f in r.findings if f["id"] == "CC-FIELD-VALUE")
    assert finding["effect"] == "advisory"
    for part in ("effort 'extreme'", "shell 'zsh'", "background without context: fork", "maxTurns"):
        assert part in finding["message"]
    assert _status(bad)["claude-code"] == "loads"
    assert "CC-FIELD-VALUE" not in _rule_ids(good, "claude-code")
    assert "CC-FIELD-VALUE" not in _rule_ids(numeric, "claude-code")


def test_claude_effort_accepts_whole_number_floats_only(tmp_path: Path):
    whole = _skill(tmp_path / "whole", extra="effort: 3.0\n")
    fraction = _skill(tmp_path / "fraction", extra="effort: 3.5\n")
    flag = _skill(tmp_path / "flag", extra="effort: true\n")

    assert "CC-FIELD-VALUE" not in _rule_ids(whole, "claude-code")
    for skill, shown in ((fraction, "effort '3.5'"), (flag, "effort 'True'")):
        finding = next(
            f for r in evaluate_harnesses(skill, "claude-code") for f in r.findings if f["id"] == "CC-FIELD-VALUE"
        )
        assert shown in finding["message"]


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


def test_version_outside_profile_adds_advisory(tmp_path: Path):
    skill = _skill(tmp_path)

    assert "VERSION-UNVERIFIED" not in _rule_ids(skill, "codex", {"harness_versions": {"codex": "0.166.0"}})
    assert "VERSION-UNVERIFIED" in _rule_ids(skill, "codex", {"harness_versions": {"codex": "0.170.0"}})
    assert "VERSION-UNVERIFIED" not in _rule_ids(skill, "codex", {"harness_versions": {"claude-code": "9.9.9"}})


def test_cli_harness_version_flag(tmp_path: Path):
    skill = _skill(tmp_path)
    result = CliRunner().invoke(
        cli, ["harness-check", str(skill), "--harness", "codex", "--harness-version", "codex=0.170.0", "-v"]
    )

    assert result.exit_code == 0, result.output
    assert "0.170.0" in result.output


def test_harness_check_ignores_a_malformed_profile_for_an_unselected_harness(tmp_path: Path):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "codex.yaml").write_text((PROFILES_DIR / "codex.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    (profiles / "custom.yaml").write_text("harness: [unclosed\n", encoding="utf-8")
    skill = _skill(tmp_path)

    result = CliRunner().invoke(
        cli,
        ["harness-check", str(skill), "--harness", "codex", "--harness-profiles", str(profiles)],
    )
    assert result.exit_code == 0, result.output

    # A version for the unselected-but-present harness is still accepted; a made-up name is still a usage error.
    ok = CliRunner().invoke(
        cli,
        [
            "harness-check",
            str(skill),
            "--harness",
            "codex",
            "--harness-profiles",
            str(profiles),
            "--harness-version",
            "custom=1.0",
        ],
    )
    assert ok.exit_code == 0, ok.output
    bad = CliRunner().invoke(
        cli,
        ["harness-check", str(skill), "--harness", "nope", "--harness-profiles", str(profiles)],
    )
    assert bad.exit_code == 2 and "Unknown harness: nope" in bad.output


def test_harness_version_uses_the_profiles_harness_name_not_its_file_name(tmp_path: Path):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    text = (PROFILES_DIR / "codex.yaml").read_text(encoding="utf-8")
    (profiles / "mine.yaml").write_text(text.replace("harness: codex", "harness: acme", 1), encoding="utf-8")
    skill = _skill(tmp_path)

    def run(version: str):
        return CliRunner().invoke(
            cli,
            [
                "harness-check",
                str(skill),
                "--harness",
                "mine",
                "--harness-profiles",
                str(profiles),
                "--harness-version",
                version,
                "-v",
            ],
        )

    ok = run("acme=9.9.9")
    assert ok.exit_code == 0, ok.output
    assert "you run 9.9.9" in ok.output
    # The file name is not what versions are keyed by, so it is a usage error instead of a version that is ignored.
    for name in ("mine", "nope"):
        bad = run(f"{name}=9.9.9")
        assert bad.exit_code == 2 and f"unknown harness {name}" in bad.output and "acme" in bad.output


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


def test_lone_surrogate_description_is_counted_not_fatal(tmp_path: Path):
    skill = _skill(tmp_path, desc='"Use when checking \\ud800 harness compatibility."')

    assert skill_facts(skill)["description_utf16_units"] > 0


def test_trigger_cut_at_the_listing_window_counts_as_missing(tmp_path: Path):
    facts = skill_facts(_skill(tmp_path, desc="x" * 55 + " Use when checking harness compatibility."))

    assert facts["trigger_end"] > 60


def test_grouped_skill_linked_to_a_sibling_group_stays_inside_the_root(tmp_path: Path):
    root = tmp_path / ".agents" / "skills"
    real = root / "group-b" / "demo-skill"
    real.parent.mkdir(parents=True)
    _skill(tmp_path / "src").rename(real)
    link = root / "group-a" / "demo-skill"
    link.parent.mkdir()
    link.symlink_to(real, target_is_directory=True)

    assert skill_facts(link)["skill_dir_symlink_target"] == "inside_root"


def test_nested_manifest_under_evals_is_reported(tmp_path: Path):
    skill = _rooted(tmp_path)
    (skill / "evals" / "fixture").mkdir(parents=True)
    (skill / "evals" / "fixture" / "SKILL.md").write_text("---\nname: fixture\n---\n", encoding="utf-8")

    assert skill_facts(skill)["nested_skill_md"] is True


@pytest.mark.parametrize(
    ("folder", "reported"),
    [("evals", True), ("versions", True), ("references", True), ("node_modules", False), (".git", False)],
)
def test_schema_and_harness_agree_on_where_a_nested_manifest_counts(tmp_path: Path, folder: str, reported: bool):
    skill = _rooted(tmp_path)
    (skill / folder / "fixture").mkdir(parents=True)
    (skill / folder / "fixture" / "SKILL.md").write_text("---\nname: fixture\n---\n", encoding="utf-8")

    schema = [f for f in SchemaValidator().validate(skill).findings if f.check_name == "nested_manifest"]

    assert skill_facts(skill)["nested_skill_md"] is reported
    assert bool(schema) is reported


def test_lowercase_nested_manifest_is_not_a_second_skill_in_codex(tmp_path: Path):
    skill = _rooted(tmp_path)
    (skill / "sub").mkdir()
    (skill / "sub" / "skill.md").write_text("---\nname: sub\n---\n", encoding="utf-8")

    facts = skill_facts(skill)
    assert facts["nested_skill_md"] is True
    assert facts["nested_skill_md_exact"] is False
    assert _status(skill)["codex"] == "loads"


def test_nested_alias_frontmatter_reads_as_empty_manifest(tmp_path):

    skill = tmp_path / "demo-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(alias_bomb_skill_md(), encoding="utf-8")

    from skillevaluator.harness import overlay

    _manifest, _raw, _text, fm_yaml, data = overlay._read_manifest(skill)

    assert data == {}
    assert overlay._tab_indent_breaks_yaml(fm_yaml) is False


@pytest.mark.parametrize(
    "policy",
    [merge_bomb_yaml(), deeply_nested_flow(), "policy:\n  allow_implicit_invocation: false\nreleased: 2026-13-45\n"],
    ids=["merge-key-bomb", "deep-nesting", "invalid-date"],
)
def test_unreadable_codex_policy_file_is_not_expanded_and_does_not_abort_the_check(tmp_path: Path, policy: str):
    skill = _rooted(tmp_path)
    (skill / "agents").mkdir()
    (skill / "agents" / "openai.yaml").write_text(policy, encoding="utf-8")

    assert overlay._codex_implicit_disabled(skill) is False
    assert skill_facts(skill)["codex_implicit_disabled"] is False


def test_codex_policy_file_still_disables_implicit_invocation(tmp_path: Path):
    skill = _rooted(tmp_path)
    (skill / "agents").mkdir()
    (skill / "agents" / "openai.yaml").write_text("policy:\n  allow_implicit_invocation: false\n", encoding="utf-8")

    assert overlay._codex_implicit_disabled(skill) is True


def test_frontmatter_nested_past_the_composer_recursion_limit_does_not_abort_the_check(tmp_path: Path):
    skill = _rooted(tmp_path)
    (skill / "SKILL.md").write_text(f"---\nname: demo-skill\n{deeply_nested_flow()}\n---\nBody\n", encoding="utf-8")

    assert skill_facts(skill)["frontmatter_duplicate_keys"] is False


def test_over_limit_frontmatter_is_not_reported_as_tab_indentation():
    # A tab inside a block scalar is valid YAML; the graph around it is over the parser limits.
    frontmatter = "name: demo\n" + "\n".join(alias_graph_lines(5)) + "\ntext: |\n  one\n  \ttwo\n"

    assert overlay._tab_indent_breaks_yaml(frontmatter) is False


def test_pi_name_collision_needs_both_skills_to_load(tmp_path: Path):
    """Only skills pi loads can collide (verified against the pinned loader): with `sky/` in the
    root's .gitignore, same-named sky/ and other/ load just other/, with no collision diagnostic."""
    root = tmp_path / "proj" / ".pi" / "skills"
    root.mkdir(parents=True)
    (root / ".gitignore").write_text("sky/\n", encoding="utf-8")
    for folder in ("sky", "other"):
        _install(root, folder, name="sky")

    rows = _rows(root / "sky")
    assert rows["pi"]["status"] == "not_loaded"
    assert "PI-IGNORED" in _ids(rows, "pi")
    assert "PI-NAME-COLLISION" not in _ids(rows, "pi")
    assert "PI-NAME-COLLISION" not in _ids(_rows(root / "other"), "pi")

    (root / ".gitignore").unlink()
    assert "PI-NAME-COLLISION" in _ids(_rows(root / "sky"), "pi")
    assert "PI-NAME-COLLISION" in _ids(_rows(root / "other"), "pi")


@pytest.mark.parametrize(
    "frontmatter",
    ["name: x\n", "name: x\ndescription: a\ndescription: b\n", "name: x\n\tdescription: b\n"],
    ids=["no-description", "duplicate-keys", "tab-indent"],
)
def test_pi_dropped_manifest_is_no_name_collision(tmp_path: Path, frontmatter: str):
    """A same-named skill pi drops is no collision for either side (verified against the pinned
    loader): the dropped row names its own cause and the loaded one is not degraded."""
    root = tmp_path / "proj" / ".pi" / "skills"
    _install(root, "a", name="x")
    dropped = root / "b"
    dropped.mkdir(parents=True)
    (dropped / "SKILL.md").write_text(f"---\n{frontmatter}---\nBody\n", encoding="utf-8")

    rows = _rows(dropped)
    assert rows["pi"]["status"] == "not_loaded"
    assert "PI-NAME-COLLISION" not in _ids(rows, "pi")
    assert _rows(root / "a")["pi"]["status"] == "loads"
    assert "PI-NAME-COLLISION" not in _ids(_rows(root / "a"), "pi")


@pytest.mark.parametrize(
    ("ignore_file", "rule", "manifest"),
    [
        (".gitignore", "sky/\n", "SKILL.md"),
        (".gitignore", "sky/SKILL.md\n", "SKILL.md"),
        ("sky/.ignore", "SKILL.md\n", "SKILL.md"),
        ("sky/.fdignore", "/SKILL.md\n", "SKILL.md"),
        (".gitignore", "sky/\n", "skill.md"),
    ],
    ids=["gitignore-folder", "gitignore-file", "nested-ignore", "nested-fdignore", "gitignore-non-canonical"],
)
def test_pi_ignored_manifest_names_the_ignore_file_and_rule(tmp_path: Path, ignore_file: str, rule: str, manifest: str):
    """An ignore-rule exclusion is its own finding naming the file and the line: pi skips such
    entries silently (verified against the pinned loader), so the manifest-name advice is wrong here."""
    root = tmp_path / "proj" / ".pi" / "skills"
    skill = root / "sky"
    skill.mkdir(parents=True)
    (skill / manifest).write_text("---\nname: sky\ndescription: Use when testing ignores.\n---\nBody\n")
    (root / ignore_file).parent.mkdir(parents=True, exist_ok=True)
    (root / ignore_file).write_text(rule, encoding="utf-8")

    row = _rows(skill)["pi"]
    ids = {f["id"] for f in row["findings"]}
    assert row["status"] == "not_loaded"
    assert "PI-IGNORED" in ids and not ids & {"PI-CASING", "PI-HIDDEN"}
    message = next(f["message"] for f in row["findings"] if f["id"] == "PI-IGNORED")
    assert f".pi/skills/{ignore_file}" in message and rule.strip() in message


@pytest.mark.parametrize(("rule", "ignored"), [("sky/\n!sky/SKILL.md\n", True), ("sky/\n!sky/\n", False)])
def test_pi_ignored_folder_is_not_re_entered_by_a_negated_rule(tmp_path: Path, rule: str, ignored: bool):
    """pi tests a folder before its contents and never re-enters an excluded one (verified against
    the pinned loader): `!sky/SKILL.md` under `sky/` does not bring the file back, `!sky/` does."""
    root = tmp_path / "proj" / ".pi" / "skills"
    skill = _install(root, "sky")
    (root / ".gitignore").write_text(rule, encoding="utf-8")

    row = _rows(skill)["pi"]
    assert row["status"] == ("not_loaded" if ignored else "loads")
    assert ("PI-IGNORED" in {f["id"] for f in row["findings"]}) is ignored


@pytest.mark.parametrize(
    ("manifest", "cause"),
    [
        ("---\nname: tabbed\n\tdescription: x\n---\nBody\n", "PI-FRONTMATTER-TABS"),
        ("---\ndescription: x\ndescription: y\n---\nBody\n", "PI-FRONTMATTER-DUPKEYS"),
        ("Body without frontmatter.\n", "PI-UNPARSEABLE"),
    ],
    ids=["tab-indent", "duplicate-keys", "no-frontmatter"],
)
def test_pi_dropped_manifest_reports_only_its_cause(tmp_path: Path, manifest: str, cause: str):
    """The row of a file pi never loads names its real cause: loads-under-the-folder-name is false for it."""
    skill = _skill(tmp_path)
    (skill / "SKILL.md").write_text(manifest, encoding="utf-8")

    ids = _ids(_rows(skill), "pi")
    assert cause in ids and "PI-NAME-FALLBACK" not in ids


@pytest.mark.parametrize(("description", "chars"), [("yes", 3), ("on", 2), ('"yes"', 3), ('"on"', 2)])
def test_pi_census_counts_descriptions_as_pi_resolves_them(tmp_path: Path, description: str, chars: int):
    """pi's YAML 1.2 core parser lists plain yes and on as strings (verified against the pinned
    loader), so its census counts them; the canonical catalog keeps the YAML 1.1 reading."""
    skill = _skill(tmp_path / "proj" / ".pi" / "skills", desc=description)

    facts = facts_for([skill], load_profiles("pi"))[skill]
    row = _rows(skill)["pi"]
    assert facts["census_desc_chars"] == row["catalog"]["desc_chars"] == chars
    assert row["listed"] == {"mode": "full", "text": description.strip('"')}
    assert facts["catalog_desc_chars"] == (chars if description.startswith('"') else 0)
