# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for plugin facts and per-harness plugin rules."""

import json
import os
import threading
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.constants import PLUGIN_CONFIG_MAX_BYTES
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


@pytest.fixture(autouse=True)
def _empty_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setattr(Path, "home", lambda: home)


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


@pytest.mark.parametrize("harnesses", ["openclaw", "openclaw,copilot-cli"])
def test_harness_check_reports_a_native_only_openclaw_plugin(tmp_path: Path, harnesses: str):
    root = _plugin(
        tmp_path / "demo",
        "openclaw.plugin.json",
        {"id": "demo", "configSchema": {"type": "object", "properties": {}}},
        components=(),
    )
    _manifest(
        root,
        "package.json",
        json.dumps({"name": "demo", "version": "1.0.0", "openclaw": {"extensions": ["./index.js"]}}),
    )
    _manifest(root, "index.js", "module.exports = function register() {};")
    report = tmp_path / "report.json"
    result = CliRunner().invoke(cli, ["harness-check", str(root), "--harness", harnesses, "--json", str(report)])

    assert result.exit_code == 0, result.output
    assert "Plugin compatibility" in result.output
    assert "Tier 1 schema also fails" not in result.output
    block = json.loads(report.read_text(encoding="utf-8"))["harness_compatibility"]
    assert set(block) == {f"{root} (plugin)"}
    rows = {row["harness"]: row for row in block[f"{root} (plugin)"]}
    assert set(rows) == set(harnesses.split(","))
    assert rows["openclaw"]["status"] == "degraded"
    assert [finding["id"] for finding in rows["openclaw"]["findings"]] == ["OW-PLUGIN-NATIVE-UNCHECKED"]


def test_harness_check_reports_schema_warnings_for_real_skills(tmp_path: Path):
    root = _plugin(tmp_path / "demo", ".claude-plugin/plugin.json")
    (root / "skills/demo/SKILL.md").write_text("# Missing frontmatter\n", encoding="utf-8")
    result = CliRunner().invoke(cli, ["harness-check", str(root), "--harness", "openclaw"])

    assert result.exit_code == 0, result.output
    assert "Tier 1 schema also fails for demo:" in result.output
    assert "Tier 1 schema also fails for demo (plugin)" not in result.output


def test_default_only_skill_folder_still_has_no_plugin_panel(tmp_path: Path):
    root = _plugin(tmp_path / "workspace", None)
    result = CliRunner().invoke(cli, ["harness-check", str(root), "--harness", "openclaw"])
    assert result.exit_code == 0, result.output
    assert "Harness compatibility" in result.output
    assert "Plugin compatibility" not in result.output


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


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs named pipes")
def test_named_pipe_skill_is_skipped_without_blocking(tmp_path: Path):
    (tmp_path / ".claude-plugin").mkdir()
    (tmp_path / ".claude-plugin" / "plugin.json").write_text('{"name": "p"}', encoding="utf-8")
    (tmp_path / "skills" / "pipe").mkdir(parents=True)
    fifo = tmp_path / "skills" / "pipe" / "SKILL.md"
    os.mkfifo(fifo)
    out: dict = {}
    worker = threading.Thread(target=lambda: out.update(plugin_facts(tmp_path)), daemon=True)
    worker.start()
    worker.join(5)
    if worker.is_alive():  # still blocked in open(); unblock it so the thread can end
        os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        worker.join(5)
        pytest.fail("plugin_facts blocked on a named-pipe SKILL.md")
    assert out["components"]["skills"] == 0
    assert out["max_skill_bytes"] == 0


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


def test_json_key_count_survives_a_file_nested_past_the_parser_recursion_limit(tmp_path: Path):
    from skillevaluator.harness.plugin import _json_keys

    nested = tmp_path / "hooks.json"
    nested.write_text("[" * 200_000 + "]" * 200_000, encoding="utf-8")

    assert _json_keys(nested) == 1


def test_plugin_json_nested_past_the_parser_recursion_limit_is_an_invalid_manifest(tmp_path: Path):
    (tmp_path / ".claude-plugin").mkdir()
    (tmp_path / ".claude-plugin" / "plugin.json").write_text("[" * 200_000 + "]" * 200_000, encoding="utf-8")

    assert plugin_facts(tmp_path)["manifest_invalid"] is True


def _openclaw(root: Path):
    profile = next(p for p in load_profiles() if p["harness"] == "openclaw")
    return evaluate_plugin(profile, plugin_facts(root))


def test_codex_format_mcp_is_a_note_not_a_degraded_row_on_openclaw(tmp_path: Path):
    result = _openclaw(_plugin(tmp_path / "p", ".codex-plugin/plugin.json", components=("skills", "mcp")))
    mcp = [f for f in result.findings if "MCP" in f["id"]]
    assert result.status == "loads"
    assert [(f["id"], f["effect"]) for f in mcp] == [("OW-PLUGIN-MCP", "advisory")]
    assert "never reached the model" not in mcp[0]["message"]
    # OpenClaw documents MCP tools for every bundle format, so there is no advice to add a Claude manifest.
    assert "fix" not in mcp[0]


def test_openclaw_reports_mcp_the_same_for_codex_and_claude_format(tmp_path: Path):
    codex = _openclaw(_plugin(tmp_path / "codex", ".codex-plugin/plugin.json", components=("skills", "mcp")))
    claude = _openclaw(_plugin(tmp_path / "claude", ".claude-plugin/plugin.json", components=("skills", "mcp")))
    assert codex.status == claude.status == "loads"
    assert codex.findings == claude.findings


def test_mcp_only_codex_bundle_loads_on_openclaw(tmp_path: Path):
    root = _plugin(tmp_path / "p", ".codex-plugin/plugin.json", components=("mcp",))
    result = _openclaw(root)
    assert plugin_facts(root)["components"]["mcp"] == 2
    assert result.status == "loads"
    assert [f["id"] for f in result.findings if "MCP" in f["id"]] == ["OW-PLUGIN-MCP"]


BROKEN = '{"name": "demo", "version": '
VALID = '{"name": "demo", "version": "0.1.0"}'


def _manifest(root: Path, rel: str, text: str) -> Path:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text, encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("rel", "fact"),
    [
        (".claude-plugin/plugin.json", "invalid_claude"),
        (".codex-plugin/plugin.json", "invalid_codex"),
        (".cursor-plugin/plugin.json", "invalid_cursor"),
        (".github/plugin/plugin.json", "invalid_copilot"),
        (".plugin/plugin.json", "invalid_copilot"),
        ("plugin.json", "invalid_root_json"),
    ],
)
def test_manifest_that_is_not_valid_json_sets_only_its_own_invalid_fact(tmp_path: Path, rel: str, fact: str):
    broken = plugin_facts(_manifest(tmp_path / "broken", rel, BROKEN))
    valid = plugin_facts(_manifest(tmp_path / "valid", rel, VALID))
    names = {"invalid_claude", "invalid_codex", "invalid_cursor", "invalid_copilot", "invalid_root_json"}
    assert {key for key in broken if key.startswith("invalid_")} == names
    assert {key for key in names if broken[key]} == {fact} and broken["manifest_invalid"]
    assert not any(valid[key] for key in names) and not valid["manifest_invalid"]


@pytest.mark.parametrize(
    ("rel", "text", "claude_code_loads", "openclaw_installs"),
    [
        (".cursor-plugin/plugin.json", VALID, True, True),
        (".claude-plugin/plugin.json", BROKEN, False, False),
        (".codex-plugin/plugin.json", BROKEN, True, False),
        (".cursor-plugin/plugin.json", BROKEN, True, False),
        ("plugin.json", BROKEN, True, True),
        (".claude-plugin/plugin.json", '{"version": "0.1.0"}', False, True),
    ],
    ids=["cursor-valid", "claude-broken", "codex-broken", "cursor-broken", "root-broken", "claude-nameless"],
)
def test_edge_case_manifest_next_to_default_folders(
    tmp_path: Path, rel: str, text: str, claude_code_loads: bool, openclaw_installs: bool
):
    s = _status(_manifest(_plugin(tmp_path / "p", None, components=("skills", "commands", "mcp")), rel, text))
    assert (s["claude-code"][0] != "not_loaded") is claude_code_loads
    assert (s["openclaw"][0] != "not_loaded") is openclaw_installs


def test_claude_code_names_the_broken_or_nameless_manifest(tmp_path: Path):
    broken = _manifest(_plugin(tmp_path / "broken", None), ".claude-plugin/plugin.json", BROKEN)
    bare = _manifest(tmp_path / "bare", ".claude-plugin/plugin.json", BROKEN)
    nameless = _plugin(tmp_path / "nameless", ".claude-plugin/plugin.json", {"version": "0.1.0"})
    assert _status(broken)["claude-code"] == ("not_loaded", ["CC-PLUGIN-MANIFEST-INVALID"])
    assert _status(bare)["claude-code"] == ("not_loaded", ["CC-PLUGIN-MANIFEST-INVALID"])
    assert plugin_facts(nameless)["nameless_claude"] and not plugin_facts(broken)["nameless_claude"]
    s = _status(nameless)
    assert s["claude-code"] == ("not_loaded", ["CC-PLUGIN-MANIFEST-NAME"])
    assert s["copilot-cli"][0] == "loads" and s["openclaw"][0] == "loads"
    named = _status(_plugin(tmp_path / "named", ".claude-plugin/plugin.json"))["claude-code"]
    assert named[0] == "loads" and not plugin_facts(tmp_path / "named")["nameless_claude"]


@pytest.mark.parametrize(
    "text",
    [
        '{"version": "0.1.0"}',
        '{"name": "", "version": "0.1.0"}',
        '{"name": 42, "version": "0.1.0"}',
        "[]",
        '{"name": "Demo Tools"}',
        '\ufeff{"name": "Demo Tools"}',
        '{"name": "demo/tools"}',
    ],
    ids=["missing", "empty", "not-a-string", "not-an-object", "spaces", "bom-with-spaces", "slash"],
)
def test_claude_code_does_not_load_a_manifest_without_a_usable_name(tmp_path: Path, text: str):
    root = _manifest(_plugin(tmp_path / "p", None), ".claude-plugin/plugin.json", text)
    assert _status(root)["claude-code"] == ("not_loaded", ["CC-PLUGIN-MANIFEST-NAME"])
    assert plugin_facts(root)["nameless_claude"] and not plugin_facts(root)["unparsed_claude"]


def test_claude_code_loads_a_bundle_whose_only_default_content_is_mcp_or_lsp_config(tmp_path: Path):
    mcp = _plugin(tmp_path / "mcp", ".codex-plugin/plugin.json", components=("mcp",))
    servers = json.dumps({"go": {"command": "gopls", "extensionToLanguage": {".go": "go"}}})
    lsp = _manifest(_plugin(tmp_path / "lsp", ".codex-plugin/plugin.json", components=()), ".lsp.json", servers)
    apps = _manifest(_plugin(tmp_path / "apps", ".codex-plugin/plugin.json", components=()), ".app.json", "{}")
    for root in (mcp, lsp):
        status, ids = _status(root)["claude-code"]
        assert status != "not_loaded" and "CC-PLUGIN-EMPTY" not in ids
        facts = plugin_facts(root)
        assert facts["default_layout"] and not facts["default_dirs"]
    assert _status(apps)["claude-code"] == ("not_loaded", ["CC-PLUGIN-EMPTY"])
    assert not plugin_facts(apps)["default_layout"]
    folders = plugin_facts(_plugin(tmp_path / "folders", ".codex-plugin/plugin.json"))
    assert folders["default_layout"] and folders["default_dirs"]


def test_openclaw_installs_a_default_layout_without_a_manifest_it_reads(tmp_path: Path):
    with_skills = _manifest(_plugin(tmp_path / "skills", None), "plugin.yaml", "name: demo\n")
    bare = _manifest(tmp_path / "bare", "plugin.yaml", "name: demo\n")
    status, ids = _status(with_skills)["openclaw"]
    assert status != "not_loaded" and "OW-PLUGIN-FORMAT" not in ids
    assert _status(bare)["openclaw"] == ("not_loaded", ["OW-PLUGIN-FORMAT"])


@pytest.mark.parametrize(
    "rel", [".claude-plugin/plugin.json", ".codex-plugin/plugin.json", ".cursor-plugin/plugin.json"]
)
def test_openclaw_names_the_broken_manifest_with_or_without_default_folders(tmp_path: Path, rel: str):
    with_folders = _manifest(_plugin(tmp_path / "folders", None), rel, BROKEN)
    bare = _manifest(tmp_path / "bare", rel, BROKEN)
    assert _status(with_folders)["openclaw"] == ("not_loaded", ["OW-PLUGIN-MANIFEST-INVALID"])
    assert _status(bare)["openclaw"] == ("not_loaded", ["OW-PLUGIN-MANIFEST-INVALID"])


def test_format_acceptance_rules_carry_medium_confidence():
    labels = {r["id"]: r["confidence"] for p in load_profiles() for r in p["plugin"]["rules"]}
    medium = {
        "CC-PLUGIN-EMPTY",
        "CC-PLUGIN-MANIFEST-INVALID",
        "CC-PLUGIN-MANIFEST-NAME",
        "OW-PLUGIN-FORMAT",
        "OW-PLUGIN-MANIFEST-INVALID",
    }
    assert {rule_id: labels[rule_id] for rule_id in medium} == dict.fromkeys(medium, "medium")


@pytest.mark.parametrize(
    "rel",
    [
        ".codex-plugin/plugin.json",
        ".cursor-plugin/plugin.json",
        ".github/plugin/plugin.json",
        ".plugin/plugin.json",
        "plugin.json",
    ],
)
@pytest.mark.parametrize("text", ['{"version": "0.1.0"}', "[]"], ids=["no-name", "not-an-object"])
def test_only_the_claude_manifest_can_be_nameless_for_claude_code(tmp_path: Path, rel: str, text: str):
    root = _manifest(_plugin(tmp_path / "p", None), rel, text)
    assert not plugin_facts(root)["nameless_claude"]
    assert _status(root)["claude-code"][0] == "loads"


def test_openclaw_default_layout_counts_a_root_mcp_or_lsp_file(tmp_path: Path):
    mcp = _manifest(
        _manifest(tmp_path / "mcp", "plugin.yaml", "name: demo\n"),
        ".mcp.json",
        '{"mcpServers": {"a": {"command": "x"}}}',
    )
    lsp = _manifest(
        _manifest(tmp_path / "lsp", "plugin.yaml", "name: demo\n"), ".lsp.json", '{"go": {"command": "gopls"}}'
    )
    for root in (mcp, lsp):
        status, ids = _status(root)["openclaw"]
        assert status != "not_loaded" and "OW-PLUGIN-FORMAT" not in ids


@pytest.mark.parametrize("valid", [".codex-plugin/plugin.json", ".cursor-plugin/plugin.json"])
def test_openclaw_reads_the_valid_manifest_it_prefers_over_a_broken_claude_one(tmp_path: Path, valid: str):
    # OpenClaw picks .codex-plugin, then .cursor-plugin, before .claude-plugin.
    root = _manifest(_plugin(tmp_path / "p", valid), ".claude-plugin/plugin.json", BROKEN)
    status, ids = _status(root)["openclaw"]
    assert status == "loads" and "OW-PLUGIN-MANIFEST-INVALID" not in ids


@pytest.mark.parametrize(
    ("preferred", "fallback"),
    [
        (".codex-plugin/plugin.json", ".claude-plugin/plugin.json"),
        (".codex-plugin/plugin.json", ".cursor-plugin/plugin.json"),
        (".cursor-plugin/plugin.json", ".claude-plugin/plugin.json"),
        (".claude-plugin/plugin.json", "plugin.json"),
    ],
)
@pytest.mark.parametrize("invalid", [BROKEN, "[]"], ids=["corrupt", "non-object"])
def test_openclaw_does_not_fall_back_after_selecting_a_bundle_manifest(
    tmp_path: Path, preferred: str, fallback: str, invalid: str
):
    root = _plugin(tmp_path / "p", fallback, {"name": "fallback", "$schema": AGENT_V1})
    _manifest(root, preferred, invalid)
    result = _openclaw(root)
    assert result.status == "not_loaded"
    assert [finding["id"] for finding in result.findings] == ["OW-PLUGIN-MANIFEST-INVALID"]
    assert preferred in result.findings[0]["message"]


@pytest.mark.parametrize("content", ['{"name": "demo",}', "// comment\n{name: 'demo',}"])
def test_openclaw_accepts_json5_in_the_selected_manifest(tmp_path: Path, content: str):
    root = _manifest(_plugin(tmp_path / "p", None), ".claude-plugin/plugin.json", BROKEN)
    _manifest(root, ".codex-plugin/plugin.json", content)
    assert _status(root)["openclaw"] == ("loads", [])
    assert plugin_facts(root)["invalid_codex"]


def test_openclaw_selected_manifest_respects_the_loader_byte_limit(tmp_path: Path):
    relative_path = ".codex-plugin/plugin.json"
    root = _manifest(_plugin(tmp_path / "p", None), relative_path, VALID.ljust(PLUGIN_CONFIG_MAX_BYTES))
    assert _status(root)["openclaw"] == ("loads", [])
    _manifest(root, relative_path, VALID.ljust(PLUGIN_CONFIG_MAX_BYTES + 1))
    assert _status(root)["openclaw"] == ("not_loaded", ["OW-PLUGIN-MANIFEST-INVALID"])


def test_openclaw_reports_a_selected_nonregular_manifest(tmp_path: Path):
    root = _plugin(tmp_path / "p", ".claude-plugin/plugin.json")
    (root / ".codex-plugin/plugin.json").mkdir(parents=True)
    assert _status(root)["openclaw"] == ("not_loaded", ["OW-PLUGIN-MANIFEST-INVALID"])


@pytest.mark.parametrize("rel", [".codex-plugin/plugin.json", "plugin.json"])
def test_openclaw_reports_a_manifest_that_reaches_the_local_nesting_limit(
    tmp_path: Path, rel: str, monkeypatch: pytest.MonkeyPatch
):
    manifest = json.dumps({"name": "demo", "$schema": AGENT_V1})
    json_loads = json.loads

    def loads(source, **kwargs):
        # CPython versions differ in where their JSON decoder reaches a nesting limit.
        if source == manifest:
            raise RecursionError("OpenClaw manifest decoder reached its nesting limit")
        return json_loads(source, **kwargs)

    monkeypatch.setattr(json, "loads", loads)
    root = _manifest(tmp_path / "p", rel, manifest)
    result = _openclaw(root)
    assert result.status == "degraded"
    assert [finding["id"] for finding in result.findings] == ["OW-PLUGIN-MANIFEST-UNCHECKED"]
    assert rel in result.findings[0]["message"]


def test_openclaw_portable_manifest_over_the_byte_limit_leaves_default_discovery_available(tmp_path: Path):
    portable = json.dumps({"$schema": AGENT_V1})
    root = _manifest(_plugin(tmp_path / "p", None), "plugin.json", portable.ljust(PLUGIN_CONFIG_MAX_BYTES))
    assert _status(root)["openclaw"] == ("not_loaded", ["OW-PLUGIN-MANIFEST-NAME"])
    _manifest(root, "plugin.json", portable.ljust(PLUGIN_CONFIG_MAX_BYTES + 1))
    assert _status(root)["openclaw"] == ("loads", [])
    assert plugin_facts(root)["fmt_agent_v1"]


def test_openclaw_portable_manifest_needs_strict_json(tmp_path: Path):
    portable = '{"$schema":"' + AGENT_V1 + '","name":"demo","description":NaN}'
    root = _manifest(_plugin(tmp_path / "p", None, components=()), "plugin.json", portable)
    assert _status(root)["openclaw"] == ("not_loaded", ["OW-PLUGIN-FORMAT"])


@pytest.mark.parametrize(
    ("path", "loads"),
    [("hooks/readme.txt", False), ("hooks/hooks.json", True), ("settings.json", True)],
)
def test_openclaw_default_markers_match_bundle_discovery(tmp_path: Path, path: str, loads: bool):
    root = _manifest(_plugin(tmp_path / "p", None, components=()), "plugin.yaml", "name: demo\n")
    _manifest(root, path, "{}")
    status, ids = _status(root)["openclaw"]
    assert status == ("loads" if loads else "not_loaded")
    assert ("OW-PLUGIN-FORMAT" in ids) is not loads
    if path == "settings.json":
        assert "OW-PLUGIN-NO-COMPONENTS" not in ids


@pytest.mark.parametrize("entry", ["index.ts", "index.js", "index.mjs", "index.cjs"])
def test_openclaw_default_entry_suppresses_bundle_markers(tmp_path: Path, entry: str):
    root = _plugin(tmp_path / "p", ".github/plugin/plugin.json")
    _manifest(root, "settings.json", "{}")
    _manifest(root, entry, "// Benign entrypoint")
    assert _status(root)["openclaw"] == ("not_loaded", ["OW-PLUGIN-NATIVE-MANIFEST-MISSING"])


@pytest.mark.parametrize(
    ("client_manifest", "native_manifest", "portable_manifest", "expected"),
    [
        (False, False, False, ("not_loaded", ["OW-PLUGIN-NATIVE-MANIFEST-MISSING"])),
        (True, False, False, ("degraded", ["OW-PLUGIN-COMMANDS"])),
        (False, False, True, ("degraded", ["OW-PLUGIN-COMMANDS"])),
        (True, True, False, ("degraded", ["OW-PLUGIN-COMMANDS"])),
        (False, True, True, ("degraded", ["OW-PLUGIN-NATIVE-UNCHECKED"])),
    ],
)
def test_openclaw_manifest_precedence_with_native_entry(
    tmp_path: Path, client_manifest: bool, native_manifest: bool, portable_manifest: bool, expected
):
    root = _plugin(tmp_path / "p", ".github/plugin/plugin.json", components=("skills", "commands"))
    _manifest(root, "index.js", "module.exports = function register() {};")
    if client_manifest:
        _manifest(root, ".codex-plugin/plugin.json", VALID)
    if native_manifest:
        _manifest(root, "openclaw.plugin.json", '{"id":"demo","configSchema":{"type":"object"}}')
    if portable_manifest:
        _manifest(root, "plugin.json", json.dumps({"name": "demo", "$schema": AGENT_V1}))
    assert _status(root)["openclaw"] == expected


@pytest.mark.parametrize("bundle", [VALID, BROKEN], ids=["valid-bundle", "broken-bundle"])
@pytest.mark.parametrize("harnesses", ["openclaw", "openclaw,codex"])
def test_openclaw_package_extensions_precede_client_bundles(tmp_path: Path, bundle: str, harnesses: str):
    root = _manifest(_plugin(tmp_path / "p", None), ".codex-plugin/plugin.json", bundle)
    _manifest(root, "openclaw.plugin.json", '{"id":"native-demo","configSchema":{"type":"object"}}')
    _manifest(root, "package.json", '{"openclaw":{"extensions":["./lib/start.js"]}}')
    _manifest(root, "lib/start.js", "module.exports = function register() {};")
    report = tmp_path / "report.json"
    result = CliRunner().invoke(cli, ["harness-check", str(root), "--harness", harnesses, "--json", str(report)])

    assert result.exit_code == 0, result.output
    rows = json.loads(report.read_text(encoding="utf-8"))["harness_compatibility"][f"{root} (plugin)"]
    rows_by_harness = {row["harness"]: row for row in rows}
    assert set(rows_by_harness) == set(harnesses.split(","))
    openclaw = rows_by_harness["openclaw"]
    assert openclaw["status"] == "degraded"
    assert [finding["id"] for finding in openclaw["findings"]] == ["OW-PLUGIN-NATIVE-UNCHECKED"]
    assert "package.json#openclaw.extensions" in openclaw["findings"][0]["message"]
    if "codex" in rows_by_harness:
        assert rows_by_harness["codex"]["status"] == ("loads" if bundle == VALID else "not_loaded")


@pytest.mark.parametrize(
    ("extensions", "status", "findings"),
    [
        (None, "loads", []),
        ([], "not_loaded", ["OW-PLUGIN-PACKAGE-INVALID"]),
        (["./missing.js"], "degraded", ["OW-PLUGIN-NATIVE-UNCHECKED"]),
    ],
    ids=["null-declaration", "empty-declaration", "unresolved-entry"],
)
def test_openclaw_package_declaration_controls_bundle_discovery(
    tmp_path: Path, extensions: object, status: str, findings: list[str]
):
    root = _plugin(tmp_path / "p", ".codex-plugin/plugin.json")
    _manifest(root, "openclaw.plugin.json", '{"id":"native-demo","configSchema":{"type":"object"}}')
    _manifest(root, "package.json", json.dumps({"openclaw": {"extensions": extensions}}))
    assert _status(root)["openclaw"] == (status, findings)


@pytest.mark.parametrize("read_error", [False, True], ids=["size-limit", "read-error"])
def test_openclaw_unread_package_leaves_bundle_compatibility_unchecked(
    tmp_path: Path, read_error: bool, monkeypatch: pytest.MonkeyPatch
):
    root = _plugin(tmp_path / "p", ".codex-plugin/plugin.json", components=("commands",))
    package = '{"openclaw":{"extensions":["./lib/start.js"]}}'
    if read_error:

        def fail_read(*args, **kwargs):
            raise OSError("Unable to read package")

        monkeypatch.setattr("skillevaluator.harness.plugin.SecureRoot.read_bytes", fail_read)
    else:
        package = package.ljust(PLUGIN_CONFIG_MAX_BYTES + 1)
    _manifest(root, "package.json", package)

    result = _openclaw(root)
    assert result.status == "degraded"
    assert [finding["id"] for finding in result.findings] == ["OW-PLUGIN-MANIFEST-UNCHECKED"]
    assert result.findings[0]["message"] == (
        "The compatibility check could not inspect package.json. OpenClaw acceptance is unverified."
    )


@pytest.mark.parametrize("entry", ["index.ts", "index.js", "index.mjs", "index.cjs"])
def test_openclaw_attempts_native_index_after_broken_client_bundle(tmp_path: Path, entry: str):
    root = _manifest(_plugin(tmp_path / "p", None), ".codex-plugin/plugin.json", BROKEN)
    _manifest(root, "openclaw.plugin.json", '{"id":"native-demo","configSchema":{"type":"object"}}')
    _manifest(root, entry, "module.exports = function register() {};")

    result = _openclaw(root)
    assert result.status == "degraded"
    assert [finding["id"] for finding in result.findings] == [
        "OW-PLUGIN-MANIFEST-FALLBACK",
        "OW-PLUGIN-NATIVE-UNCHECKED",
    ]
    assert entry in result.findings[1]["message"]


@pytest.mark.parametrize(
    ("bundle", "status", "findings"),
    [
        (BROKEN, "degraded", ["OW-PLUGIN-MANIFEST-FALLBACK", "OW-PLUGIN-NATIVE-UNCHECKED"]),
        (VALID, "loads", []),
    ],
    ids=["broken-bundle", "valid-bundle"],
)
def test_openclaw_native_index_link_respects_bundle_discovery(
    tmp_path: Path, bundle: str, status: str, findings: list[str]
):
    root = _manifest(_plugin(tmp_path / "p", None), ".codex-plugin/plugin.json", bundle)
    _manifest(root, "openclaw.plugin.json", '{"id":"native-demo","configSchema":{"type":"object"}}')
    _manifest(root, "dist/start.js", "module.exports = function register() {};")
    try:
        (root / "index.js").symlink_to("dist/start.js")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"File symlinks unavailable ({exc})")

    assert _status(root)["openclaw"] == (status, findings)


@pytest.mark.parametrize("schema", [None, AGENT_V1, "https://agent-plugins.org/schemas/9.9.0/plugin.schema.json"])
def test_openclaw_accepts_root_manifest_only_with_supported_agent_schema(tmp_path: Path, schema: str | None):
    data = {"name": "root-demo"}
    if schema is not None:
        data["$schema"] = schema
    root = _plugin(tmp_path / "p", "plugin.json", data, components=())
    status, ids = _status(root)["openclaw"]
    assert status == ("loads" if schema == AGENT_V1 else "not_loaded")
    assert ("OW-PLUGIN-FORMAT" in ids) is (schema != AGENT_V1)


@pytest.mark.parametrize("name", [None, "", " \t ", 42], ids=["missing", "empty", "blank", "not-a-string"])
def test_openclaw_selected_agent_manifest_needs_a_name_even_with_default_markers(tmp_path: Path, name: object):
    data = {"$schema": AGENT_V1}
    if name is not None:
        data["name"] = name
    root = _plugin(tmp_path / "p", "plugin.json", data)
    assert _status(root)["openclaw"] == ("not_loaded", ["OW-PLUGIN-MANIFEST-NAME"])
    _manifest(root, ".claude-plugin/plugin.json", VALID)
    assert _status(root)["openclaw"] == ("loads", [])


def test_claude_lenient_manifest_keeps_its_declared_components(tmp_path: Path):
    root = _manifest(tmp_path / "mixed", "plugin.yaml", "name: demo\n")
    _manifest(root, "custom/commands/hello.md", "# Hello\n")
    _manifest(root, ".claude-plugin/plugin.json", '\ufeff{"name": "demo", "commands": "./custom/commands"}')
    assert _status(root)["claude-code"] == ("loads", [])
    assert not plugin_facts(root)["declares_components"]


@pytest.mark.parametrize(
    ("content", "claude_code", "openclaw_loads"),
    [
        (b"\xef\xbb\xbf" + VALID.encode(), ("loads", []), True),
        (b'{"name": "demo", "description": "caf\xe9"}', ("loads", []), True),
        (b"\xef\xbb\xbf" + b'{"version": "0.1.0"}', ("not_loaded", ["CC-PLUGIN-MANIFEST-NAME"]), True),
        (VALID.encode("utf-16"), ("not_loaded", ["CC-PLUGIN-MANIFEST-INVALID"]), False),
        (BROKEN.encode(), ("not_loaded", ["CC-PLUGIN-MANIFEST-INVALID"]), False),
    ],
    ids=["byte-order-mark", "stray-byte", "byte-order-mark-no-name", "utf-16", "truncated"],
)
def test_claude_code_reads_its_manifest_more_leniently_than_strict_json(
    tmp_path: Path, content: bytes, claude_code: tuple[str, list[str]], openclaw_loads: bool
):
    root = _plugin(tmp_path / "p", None)
    (root / ".claude-plugin").mkdir()
    (root / ".claude-plugin/plugin.json").write_bytes(content)
    s = _status(root)
    assert s["claude-code"] == claude_code
    assert plugin_facts(root)["invalid_claude"]
    assert s["openclaw"] == (("loads", []) if openclaw_loads else ("not_loaded", ["OW-PLUGIN-MANIFEST-INVALID"]))


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_claude_code_rejects_nonstandard_constants_that_openclaw_json5_accepts(
    tmp_path: Path, encoding: str, constant: str
):
    root = _plugin(tmp_path / "p", ".claude-plugin/plugin.json")
    (root / ".claude-plugin/plugin.json").write_text('{"name":"demo","x":' + constant + "}", encoding=encoding)
    statuses = _status(root)
    assert statuses["claude-code"] == ("not_loaded", ["CC-PLUGIN-MANIFEST-INVALID"])
    assert statuses["openclaw"] == ("loads", [])


@pytest.mark.skipif(os.name != "posix", reason="needs POSIX directory permissions")
@pytest.mark.parametrize("unreadable", [".codex-plugin/plugin.json", "hooks/hooks.json"])
def test_harness_check_skips_paths_beneath_unsearchable_directories(tmp_path: Path, unreadable: str):
    root = _plugin(tmp_path / "p", ".claude-plugin/plugin.json")
    _manifest(root, unreadable, VALID)
    folder = (root / unreadable).parent
    report = tmp_path / "report.json"
    original_mode = folder.stat().st_mode
    folder.chmod(0)
    try:
        if os.access(folder, os.X_OK):
            pytest.skip("filesystem or account permits access despite directory mode")
        result = CliRunner().invoke(
            cli, ["harness-check", str(root), "--harness", "claude-code,openclaw", "--json", str(report)]
        )
        assert result.exit_code == 0, result.exception
        rows = json.loads(report.read_text())["harness_compatibility"][f"{root} (plugin)"]
        assert {row["harness"]: row["status"] for row in rows} == {"claude-code": "loads", "openclaw": "loads"}
    finally:
        folder.chmod(original_mode)
