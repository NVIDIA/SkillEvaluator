# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the advisory harness check, panel and harness-check command."""

import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.harness import HARNESS_STATUSES, evaluate_harnesses, facts_for, load_profiles, skill_facts
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


def test_copilot_shows_body_for_missing_description_but_drops_non_string(tmp_path: Path):
    skill = tmp_path / "skills" / "no-desc"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: no-desc\n---\nCustom skill\n", encoding="utf-8")
    rows = _rows(skill)
    assert rows["copilot-cli"]["status"] == "degraded"
    assert "CP-DESC-MISSING" in _ids(rows, "copilot-cli") and "CP-STRICT-YAML" not in _ids(rows, "copilot-cli")
    listed = _skill(tmp_path / "list", desc="[a, b]")
    assert "CP-STRICT-YAML" in _ids(_rows(listed), "copilot-cli")


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
    assert "Harness compatibility" in text and "advisory" not in text
    assert "rejected" in text and "[CP-DESC-LIMIT]" not in text
    assert "1 of 6 not loaded" in text and "Details and fixes: -v" in text
    full = _render(harness_panel(block, verbose=True))
    assert "NOT LOADED" in full and "[CP-DESC-LIMIT]" in full


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
    assert "Harness compatibility" in result.output and "· advisory" not in result.output
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
    assert {"CC-PRECEDENCE", "CX-DUP-NAME", "CP-DUP-NAME", "HM-DUP-NAME", "OW-DUP-NAME", "OC-DUP-NAME"} <= {
        f["id"] for r in rows.values() for f in r["findings"]
    }
    assert "(filler-copy)" in next(f["message"] for f in rows["opencode"]["findings"] if f["id"] == "OC-DUP-NAME")


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


def test_codex_implicit_policy_is_codex_only(tmp_path: Path):
    skill = _skill(tmp_path)
    (skill / "agents").mkdir()
    (skill / "agents" / "openai.yaml").write_text("policy:\n  allow_implicit_invocation: false\n")
    rows = _rows(skill)

    flagged = {h for h, r in rows.items() if any(f["id"].endswith("CODEX-POLICY") for f in r["findings"])}
    assert flagged == set(HARNESSES) - {"codex"}


def test_claude_placeholders_reach_other_harnesses_literally(tmp_path: Path):
    rows = _rows(_skill(tmp_path, body=BODY + "Run ${CLAUDE_SKILL_DIR}/run.sh $ARGUMENTS\n"))

    for harness in set(HARNESSES) - {"claude-code"}:
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
        ("openclaw", "OW-READ-BUDGET", "skill_md_chars", 16_000),
        ("openclaw", "OW-READ-REFUSED", "skill_md_chars", 64_000),
        ("codex", "CX-SHELL-READ", "skill_md_bytes", 40_000),
        ("opencode", "OC-BODY-CUT", "skill_md_bytes", 51_200),
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


def test_compact_view_shows_worst_finding_per_harness_and_catalog(tmp_path: Path):
    _catalog(tmp_path, 40)
    block = harness_block(run_validation(tmp_path / ".claude" / "skills" / "filler-000", checks="harness"))
    compact, full = _render(harness_panel(block)), _render(harness_panel(block, verbose=True))

    assert "catalog  40 skills in" in compact and "(12,000 desc" in compact
    assert "not found" in compact and "listing shortened" in compact
    assert "[CX-ROOT]" not in compact and "[CX-ROOT]" in full and "[CX-LIST-BUDGET]" in full
    assert len(compact.splitlines()) < len(full.splitlines())


def test_tier1_summary_gives_harness_its_own_line(tmp_path: Path):
    results = run_validation(_skill(tmp_path, desc=EMOJI_DESC), checks="schema,harness", harness="all")
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
