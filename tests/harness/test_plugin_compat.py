# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for plugin facts and per-harness plugin rules."""

import json
from pathlib import Path

from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.harness import evaluate_plugin, is_plugin_root, load_profiles, plugin_facts

AGENT_V1 = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"


def _plugin(
    root: Path, manifest: str | None, data: dict | None = None, components: tuple[str, ...] = ("skills",)
) -> Path:
    root.mkdir(parents=True)
    if manifest:
        (root / manifest).parent.mkdir(parents=True, exist_ok=True)
        (root / manifest).write_text(json.dumps(data or {"name": "demo", "version": "0.1.0"}))
    if "skills" in components:
        (root / "skills/demo").mkdir(parents=True)
        (root / "skills/demo/SKILL.md").write_text("---\nname: demo\ndescription: Use when testing.\n---\nBody\n")
    if "commands" in components:
        (root / "commands").mkdir()
        (root / "commands/run.md").write_text("---\ndescription: Run\n---\nRun $ARGUMENTS\n")
    if "mcp" in components:
        (root / ".mcp.json").write_text(json.dumps({"mcpServers": {"a": {"command": "x"}, "b": {"command": "y"}}}))
    return root


def _status(root: Path) -> dict[str, tuple[str, list[str]]]:
    facts = plugin_facts(root)
    out = {}
    for profile in load_profiles():
        result = evaluate_plugin(profile, facts)
        out[result.harness] = (result.status, [f["id"] for f in result.findings])
    return out


def test_every_profile_has_plugin_section():
    for profile in load_profiles():
        assert "rules" in profile["plugin"] and "components" in profile["plugin"]


def test_facts_detect_formats_name_and_components(tmp_path: Path):
    root = _plugin(tmp_path / "p", ".claude-plugin/plugin.json", components=("skills", "commands", "mcp"))
    facts = plugin_facts(root)
    assert is_plugin_root(root) and facts["fmt_claude"] and not facts["fmt_codex"]
    assert facts["plugin_name"] == "demo"
    assert facts["components"] == {"skills": 1, "commands": 1, "agents": 0, "hooks": 0, "mcp": 2, "lsp": 0}


def test_claude_format_acceptance_matrix(tmp_path: Path):
    s = _status(_plugin(tmp_path / "p", ".claude-plugin/plugin.json"))
    assert {h: v[0] for h, v in s.items()} == {
        "claude-code": "loads",
        "codex": "loads",
        "copilot-cli": "loads",
        "hermes": "not_loaded",
        "openclaw": "loads",
        "opencode": "not_loaded",
    }
    assert "CX-PLUGIN-NAMESPACE" in s["codex"][1]


def test_codex_only_bundle_is_refused_by_copilot(tmp_path: Path):
    s = _status(_plugin(tmp_path / "p", ".codex-plugin/plugin.json"))
    assert s["copilot-cli"][0] == "not_loaded" and s["codex"][0] == "loads"


def test_agent_v1_needs_schema_for_hermes(tmp_path: Path):
    good = _status(_plugin(tmp_path / "a", "plugin.json", {"$schema": AGENT_V1, "name": "demo", "version": "1"}))
    plain = _status(_plugin(tmp_path / "b", "plugin.json", {"name": "demo"}))
    assert good["hermes"][0] == "loads" and "HM-PLUGIN-DISABLED" in good["hermes"][1]
    assert plain["hermes"][0] == "not_loaded"


def test_undelivered_components_degrade(tmp_path: Path):
    s = _status(_plugin(tmp_path / "p", ".claude-plugin/plugin.json", components=("skills", "commands", "mcp")))
    assert s["codex"][0] == "degraded" and "CX-PLUGIN-COMMANDS" in s["codex"][1]
    assert s["openclaw"][0] == "degraded" and "OW-PLUGIN-MCP" in s["openclaw"][1]
    assert s["claude-code"][0] == "loads" and "CC-PLUGIN-MCP" in s["claude-code"][1]


def test_claude_inline_plugin_needs_default_folders(tmp_path: Path):
    inline = tmp_path / "inline"
    _plugin(inline, None)
    (inline / "plugin.json").write_text(json.dumps({"name": "inline"}))
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "plugin.json").write_text(json.dumps({"name": "empty"}))
    assert plugin_facts(inline)["default_dirs"] and _status(inline)["claude-code"][0] == "loads"
    assert _status(empty)["claude-code"] == ("not_loaded", ["CC-PLUGIN-EMPTY"])


def test_malformed_manifest_is_a_fact_not_a_crash(tmp_path: Path):
    root = _plugin(tmp_path / "p", None)
    (root / ".claude-plugin").mkdir()
    (root / ".claude-plugin/plugin.json").write_text("{not json")
    facts = plugin_facts(root)
    assert facts["manifest_invalid"] and facts["plugin_name"] == "p"


def test_harness_check_prints_plugin_panel(tmp_path: Path):
    root = _plugin(tmp_path / "p", ".claude-plugin/plugin.json")
    result = CliRunner().invoke(cli, ["harness-check", str(root)])
    assert result.exit_code == 0, result.output
    assert "Plugin compatibility" in result.output and "opencode" in result.output


def test_codex_agent_plugins_cap_and_hooks(tmp_path: Path):
    def bundle(name: str, manifest: str, data: dict) -> Path:
        root = _plugin(tmp_path / name, manifest, data)
        (root / "skills/demo/SKILL.md").write_text(
            "---\nname: demo\ndescription: Use when testing.\n---\n" + "x" * 8200
        )
        (root / "hooks").mkdir()
        (root / "hooks/hooks.json").write_text(json.dumps({"hooks": {"Stop": []}}))
        return root

    agent = _status(bundle("a", "plugin.json", {"$schema": AGENT_V1, "name": "demo", "version": "1"}))["codex"]
    codex = _status(bundle("c", ".codex-plugin/plugin.json", {"name": "demo", "version": "1"}))["codex"]
    assert agent[0] == "degraded" and {"CX-PLUGIN-SKILL-CAP", "CX-PLUGIN-AGENT-HOOKS"} <= set(agent[1])
    assert codex[0] == "loads" and "CX-PLUGIN-HOOKS" in codex[1] and "CX-PLUGIN-SKILL-CAP" not in codex[1]


def test_unreadable_or_malformed_manifest_is_not_an_accepted_format(tmp_path: Path):
    from skillevaluator.harness.plugin import plugin_facts

    (tmp_path / ".claude-plugin").mkdir()
    (tmp_path / ".claude-plugin" / "plugin.json").write_text("{not json", encoding="utf-8")
    facts = plugin_facts(tmp_path)
    assert facts["manifest_invalid"] and not facts["fmt_claude"] and not facts["any_manifest"]


def test_manifest_only_plugin_is_flagged_as_adding_nothing(tmp_path: Path):
    (tmp_path / ".claude-plugin").mkdir()
    (tmp_path / ".claude-plugin" / "plugin.json").write_text('{"name": "empty"}', encoding="utf-8")
    facts = plugin_facts(tmp_path)
    claude = next(p for p in load_profiles() if p["harness"] == "claude-code")
    result = evaluate_plugin(claude, facts)
    assert result.status == "loads"
    assert [f["id"] for f in result.findings] == ["CC-PLUGIN-NO-COMPONENTS"]


def test_dangling_skill_link_does_not_abort_plugin_facts(tmp_path: Path):
    (tmp_path / ".claude-plugin").mkdir()
    (tmp_path / ".claude-plugin" / "plugin.json").write_text('{"name": "p"}', encoding="utf-8")
    (tmp_path / "skills" / "gone").mkdir(parents=True)
    (tmp_path / "skills" / "gone" / "SKILL.md").symlink_to(tmp_path / "missing.md")
    assert plugin_facts(tmp_path)["max_skill_bytes"] == 0


def test_unreadable_skill_is_not_counted_as_delivered(tmp_path: Path):
    (tmp_path / ".claude-plugin").mkdir()
    (tmp_path / ".claude-plugin" / "plugin.json").write_text('{"name": "p"}', encoding="utf-8")
    (tmp_path / "skills" / "gone").mkdir(parents=True)
    (tmp_path / "skills" / "gone" / "SKILL.md").symlink_to(tmp_path / "missing.md")
    (tmp_path / "skills" / "loop").mkdir()
    (tmp_path / "skills" / "loop" / "SKILL.md").symlink_to(tmp_path / "skills" / "loop" / "SKILL.md")
    facts = plugin_facts(tmp_path)
    assert facts["components"]["skills"] == 0
    assert facts["max_skill_bytes"] == 0
    claude = next(p for p in load_profiles() if p["harness"] == "claude-code")
    assert [f["id"] for f in evaluate_plugin(claude, facts).findings] == ["CC-PLUGIN-NO-COMPONENTS"]
    (tmp_path / "skills" / "real").mkdir()
    (tmp_path / "skills" / "real" / "SKILL.md").write_text("---\nname: real\n---\n", encoding="utf-8")
    facts = plugin_facts(tmp_path)
    assert facts["components"]["skills"] == 1
    assert facts["max_skill_bytes"] > 0


def test_components_declared_in_the_manifest_are_not_reported_missing(tmp_path: Path):
    (tmp_path / ".claude-plugin").mkdir()
    manifest = {"name": "p", "commands": "./custom/commands"}
    (tmp_path / ".claude-plugin" / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    claude = next(p for p in load_profiles() if p["harness"] == "claude-code")
    assert not evaluate_plugin(claude, plugin_facts(tmp_path)).findings


def test_component_rule_replaces_the_generic_delivery_finding(tmp_path: Path):
    manifest = {"$schema": "https://agent-plugins.org/schemas/v1/plugin.json", "name": "p"}
    (tmp_path / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "hooks").mkdir()
    (tmp_path / "hooks" / "hooks.json").write_text('{"hooks": {"PreToolUse": []}}', encoding="utf-8")
    codex = next(p for p in load_profiles() if p["harness"] == "codex")
    ids = [f["id"] for f in evaluate_plugin(codex, plugin_facts(tmp_path)).findings]
    assert "CX-PLUGIN-AGENT-HOOKS" in ids and "CX-PLUGIN-HOOKS" not in ids


def _quoted_skill(tmp_path: Path) -> Path:
    skill = tmp_path / ".claude" / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        '---\nname: demo\ndescription: Use when testing "quoted" text.\n---\nBody.\n', encoding="utf-8"
    )
    return skill


def test_findings_carry_their_rule_confidence(tmp_path: Path):
    out = tmp_path / "h.json"
    CliRunner().invoke(cli, ["harness-check", str(_quoted_skill(tmp_path)), "--json", str(out)])
    block = json.loads(out.read_text(encoding="utf-8"))["harness_compatibility"]
    findings = [f for rows in block.values() for row in rows for f in row["findings"]]
    assert findings and all(f.get("confidence") for f in findings)


def test_plugin_rows_flag_an_unverified_release(tmp_path: Path):
    (tmp_path / ".codex-plugin").mkdir()
    (tmp_path / ".codex-plugin" / "plugin.json").write_text('{"name": "p"}', encoding="utf-8")
    codex = next(p for p in load_profiles() if p["harness"] == "codex")
    result = evaluate_plugin(codex, plugin_facts(tmp_path), {"harness_versions": {"codex": "9.9.9"}})
    assert "VERSION-UNVERIFIED" in [f["id"] for f in result.findings]


def test_unknown_harness_version_name_is_a_usage_error(tmp_path: Path):
    result = CliRunner().invoke(cli, ["harness-check", str(_quoted_skill(tmp_path)), "--harness-version", "codeex=1.0"])
    assert result.exit_code == 2 and "codeex" in result.output
