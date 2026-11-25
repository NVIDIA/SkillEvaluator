# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for plugin facts and per-harness plugin rules."""

import copy
import json
import os
import threading
from pathlib import Path

import pytest
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


def _manifest(root: Path, rel: str, text: str) -> Path:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text, encoding="utf-8")
    return root


def _component_rows(root: Path) -> dict[str, list[dict]]:
    facts = plugin_facts(root)
    return {p["harness"]: evaluate_plugin(p, facts).findings for p in load_profiles()}


def _bare_manifest(root: Path, manifest: str = ".claude-plugin/plugin.json", data: dict | None = None) -> Path:
    return _plugin(root, manifest, data or {"name": "demo", "version": "0.1.0"}, components=())


def test_flat_mcp_json_is_counted_for_claude_code(tmp_path: Path):
    root = _bare_manifest(tmp_path / "p")
    (root / ".mcp.json").write_text(json.dumps({"github": {"type": "http", "url": "https://example.com/mcp"}}))
    facts = plugin_facts(root)
    assert facts["mcp_flat"] == 1
    assert facts["components"]["mcp"] == 0  # the wrapped-shape count, the shape every harness was tested with
    claude = {f["id"]: f for f in _component_rows(root)["claude-code"]}
    assert claude["CC-PLUGIN-MCP"]["message"].startswith("1 ")
    assert "CC-PLUGIN-MCP-FLAT" not in claude and "CC-PLUGIN-NO-COMPONENTS" not in claude


def test_flat_mcp_json_is_reported_untested_elsewhere_and_drives_no_other_rule(tmp_path: Path):
    root = _bare_manifest(tmp_path / "p")
    agent = _bare_manifest(tmp_path / "a", "plugin.json", {"$schema": AGENT_V1, "name": "demo", "version": "1"})
    for bundle in (root, agent):
        (bundle / ".mcp.json").write_text(json.dumps({"db": {"command": "node", "args": ["server.js"]}}))
    rows = {**_component_rows(root), "hermes": _component_rows(agent)["hermes"]}
    for harness, prefix in (("codex", "CX"), ("copilot-cli", "CP"), ("openclaw", "OW"), ("hermes", "HM")):
        found = {f["id"]: f for f in rows[harness]}
        note = found[f"{prefix}-PLUGIN-MCP-FLAT"]
        assert note["effect"] == "advisory" and "not tested" in note["message"], harness
        # The fix names the wrapped shape as tested only where it was: on Hermes the wrapped servers did not start.
        assert note["fix"].endswith("the shape this harness was tested with.") is (harness != "hermes"), harness
        # The servers are not counted as delivered, registered or named for a harness that was never tested with them.
        assert not {f"{prefix}-PLUGIN-MCP", "CP-PLUGIN-MCP-NAMES"} & set(found), harness
    assert not any(f["id"].endswith("NO-COMPONENTS") for findings in rows.values() for f in findings)


def test_flat_mcp_count_skips_entries_that_are_not_servers(tmp_path: Path):
    flat = _bare_manifest(tmp_path / "flat")
    (flat / ".mcp.json").write_text(
        json.dumps({"$schema": "https://example.com/s.json", "servers": {"a": {}}, "db": {"command": "node"}})
    )
    wrapped = _plugin(tmp_path / "wrapped", ".claude-plugin/plugin.json", components=("mcp",))
    assert plugin_facts(flat)["mcp_flat"] == 1
    assert plugin_facts(wrapped)["mcp_flat"] == 0 and plugin_facts(wrapped)["components"]["mcp"] == 2
    assert "CP-PLUGIN-MCP-NAMES" in _status(wrapped)["copilot-cli"][1]
    assert not any(i.endswith("MCP-FLAT") for _, ids in _status(wrapped).values() for i in ids)


@pytest.mark.parametrize(
    ("declared", "name", "count"),
    [(False, "another", 1), (True, "another", 2), (True, "demo", 1)],
    ids=["other-format", "explicit-distinct-server", "explicit-override"],
)
def test_flat_mcp_count_respects_claude_config_sources(tmp_path: Path, declared: bool, name: str, count: int):
    manifest = {"name": "demo", **({"mcpServers": "./mcp.json"} if declared else {})}
    root = _bare_manifest(tmp_path / "p", data=manifest)
    (root / ".mcp.json").write_text(json.dumps({"demo": {"url": "https://example.invalid/flat"}}))
    (root / "mcp.json").write_text(json.dumps({"mcpServers": {name: {"url": "https://example.invalid/wrapped"}}}))
    rows = _component_rows(root)
    claude = next(f for f in rows["claude-code"] if f["id"] == "CC-PLUGIN-MCP")
    assert claude["message"].startswith(f"{count} mcp:")
    assert "CP-PLUGIN-MCP-NAMES" in {f["id"] for f in rows["copilot-cli"]}


def test_flat_mcp_count_requires_a_nonempty_string_endpoint(tmp_path: Path):
    root = _bare_manifest(tmp_path / "p")
    path = root / ".mcp.json"
    invalid = {"blank": {"command": ""}, "numeric": {"url": 42}, "null": {"command": None}}
    path.write_text(json.dumps(invalid))
    assert _component_rows(root)["claude-code"] == []
    path.write_text(json.dumps({**invalid, "valid": {"command": "", "url": "https://example.invalid/mcp"}}))
    claude = next(f for f in _component_rows(root)["claude-code"] if f["id"] == "CC-PLUGIN-MCP")
    assert claude["message"].startswith("1 mcp:")


@pytest.mark.parametrize(
    ("default_file", "expected"),
    [
        (None, 1),
        ({}, 1),
        ({"$schema": "https://example.invalid/schema.json", "db": {"command": "demo"}}, 2),
        ({"noise": None, "db": {"command": "demo"}}, 2),
        ({"mcpServers": {"db": {"command": "demo"}}}, 2),
    ],
    ids=["manifest-only", "empty-default", "schema-metadata", "invalid-entry", "wrapped-default"],
)
def test_claude_mcp_manifest_merges_with_any_default_shape(tmp_path: Path, default_file: dict | None, expected: int):
    manifest = {"name": "demo", "mcpServers": {"search": {"url": "https://example.invalid/mcp"}}}
    root = _bare_manifest(tmp_path / "p", data=manifest)
    if default_file is not None:
        (root / ".mcp.json").write_text(json.dumps(default_file))
    finding = next((f for f in _component_rows(root)["claude-code"] if f["id"] == "CC-PLUGIN-MCP"), None)
    assert finding is not None
    assert finding["message"].startswith(f"{expected} mcp:")


@pytest.mark.parametrize("filename", [".mcp.json", "servers.json"])
def test_claude_mcp_manifest_can_explicitly_name_a_flat_config_file(tmp_path: Path, filename: str):
    manifest = {
        "name": "demo",
        "mcpServers": [f"./{filename}", {"search": {"url": "https://example.invalid/mcp"}}],
    }
    root = _bare_manifest(tmp_path / "p", data=manifest)
    (root / filename).write_text(
        json.dumps({"$schema": "https://example.invalid/schema.json", "db": {"command": "demo"}})
    )
    finding = next(f for f in _component_rows(root)["claude-code"] if f["id"] == "CC-PLUGIN-MCP")
    assert finding["message"].startswith("2 mcp:")


@pytest.mark.parametrize("declaration", ["servers.json", ["servers.json", {}], False, [7], None])
def test_claude_invalid_mcp_manifest_refuses_plugin_even_with_valid_mcp_files(tmp_path: Path, declaration: object):
    root = _bare_manifest(tmp_path / "p", data={"name": "demo", "mcpServers": declaration})
    servers = {"mcpServers": {"demo": {"command": "demo"}}}
    for filename in (".mcp.json", "servers.json"):
        (root / filename).write_text(json.dumps(servers))
    rows = _status(root)
    assert rows["claude-code"] == ("not_loaded", ["CC-PLUGIN-MCP-MANIFEST"])
    assert rows["copilot-cli"][0] == "loads"


@pytest.mark.parametrize("path", ["./servers.json", "./missing.json"])
def test_claude_valid_mcp_manifest_preserves_default_and_available_config(tmp_path: Path, path: str):
    root = _bare_manifest(tmp_path / "p", data={"name": "demo", "mcpServers": path})
    (root / ".mcp.json").write_text(json.dumps({"default": {"command": "demo"}}))
    (root / "servers.json").write_text(json.dumps({"ref": {"command": "demo"}}))
    rows = _component_rows(root)
    [finding] = rows["claude-code"]
    count = 2 if path == "./servers.json" else 1
    assert (finding["id"], finding["outcome"]) == ("CC-PLUGIN-MCP", "deferred")
    assert finding["message"].startswith(f"{count} mcp:")


def test_invalid_flat_mcp_does_not_count_a_different_formats_file(tmp_path: Path):
    root = _bare_manifest(tmp_path / "p")
    (root / ".mcp.json").write_text(json.dumps({"invalid": {"command": ""}}))
    wrapped = {"mcpServers": {"demo": {"url": "https://example.invalid/mcp"}}}
    (root / "mcp.json").write_text(json.dumps(wrapped))
    rows = _component_rows(root)
    assert "CC-PLUGIN-MCP" not in {f["id"] for f in rows["claude-code"]}
    assert "CP-PLUGIN-MCP-NAMES" in {f["id"] for f in rows["copilot-cli"]}


def test_flat_mcp_note_avoids_wrapper_advice_where_its_delivery_was_unverified(tmp_path: Path):
    root = _bare_manifest(tmp_path / "p", "plugin.json", {"$schema": AGENT_V1, "name": "demo"})
    (root / ".mcp.json").write_text(json.dumps({"demo": {"url": "https://example.invalid/mcp"}}))
    note = next(f for f in _component_rows(root)["hermes"] if f["id"] == "HM-PLUGIN-MCP-FLAT")
    assert "wrapper" not in note["fix"]


def _codex_app(root: Path) -> None:
    _bare_manifest(root, ".codex-plugin/plugin.json", {"name": "demo", "apps": "./.app.json"})
    (root / ".app.json").write_text(json.dumps({"apps": {"demo": {"id": "connector_1"}}}))


def _app_file_only(root: Path) -> None:
    _bare_manifest(root, ".codex-plugin/plugin.json")
    (root / ".app.json").write_text(json.dumps({"apps": {"demo": {"id": "connector_1"}}}))


def _hook_modules(root: Path) -> None:
    _bare_manifest(root)
    (root / "hooks").mkdir()
    (root / "hooks/hooks.json").write_text(json.dumps({"modules": ["./register.tsx"]}))


def _output_style(root: Path) -> None:
    _bare_manifest(root)
    (root / "output-styles").mkdir()
    (root / "output-styles/terse.md").write_text("---\nname: terse\ndescription: Short answers\n---\nBe brief.\n")


def _settings(root: Path) -> None:
    _bare_manifest(root)
    (root / "settings.json").write_text(json.dumps({"agent": "reviewer"}))


def _root_skill(root: Path) -> None:
    _bare_manifest(root)
    (root / "SKILL.md").write_text("---\nname: demo\ndescription: Use when testing.\n---\nBody\n")


def _extensions_key(root: Path) -> None:
    data = {"$schema": AGENT_V1, "name": "demo", "version": "1", "extensions": {"com.example": {"agents": "./x"}}}
    _bare_manifest(root, "plugin.json", data)


def _dependencies_key(root: Path) -> None:
    _bare_manifest(root, data={"name": "demo", "dependencies": ["other-plugin"]})


def _hermes_native(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "plugin.yaml").write_text("name: demo\nversion: 0.1.0\nprovides_hooks:\n  - pre_llm_call\n")
    (root / "__init__.py").write_text("def register(ctx):\n    pass\n")


def _marketplace_entry(root: Path) -> None:
    _bare_manifest(root)
    (root / ".claude-plugin/marketplace.json").write_text(json.dumps({"name": "m", "plugins": []}))


def _linked_skill_folder(root: Path) -> None:
    _bare_manifest(root)
    shared = root.parent / "shared" / "demo"
    shared.mkdir(parents=True)
    (shared / "SKILL.md").write_text("---\nname: demo\ndescription: Use when testing.\n---\nBody\n")
    (root / "skills").mkdir()
    (root / "skills/demo").symlink_to(shared, target_is_directory=True)


def _second_manifest_unreadable(root: Path) -> None:
    _bare_manifest(root, ".codex-plugin/plugin.json")
    (root / ".claude-plugin").mkdir()
    (root / ".claude-plugin/plugin.json").write_text("{not json")


@pytest.mark.parametrize(
    "build",
    [
        _codex_app,
        _app_file_only,
        _hook_modules,
        _output_style,
        _settings,
        _root_skill,
        _extensions_key,
        _dependencies_key,
        _hermes_native,
        _marketplace_entry,
        _linked_skill_folder,
        _second_manifest_unreadable,
    ],
)
def test_adds_nothing_note_stays_silent_when_the_plugin_ships_something_this_check_does_not_model(
    tmp_path: Path, build
):
    root = tmp_path / "p"
    build(root)
    rows = _component_rows(root)
    assert any(not any(f["effect"] == "not_loaded" for f in findings) for findings in rows.values())
    assert not any(f["id"].endswith("NO-COMPONENTS") for findings in rows.values() for f in findings)


def test_adds_nothing_note_fires_for_a_manifest_with_only_documentation_beside_it(tmp_path: Path):
    root = _bare_manifest(
        tmp_path / "p", data={"name": "demo", "version": "0.1.0", "description": "d", "author": {"name": "a"}}
    )
    for name in ("README.md", "LICENSE", "Licence.txt", "NOTICE", "CHANGELOG.rst", ".gitignore"):
        (root / name).write_text("text\n")
    (root / "skills").mkdir()
    (root / ".git").mkdir()
    (root / ".git/HEAD").write_text("ref: refs/heads/main\n")
    loading = {h: f for h, f in _component_rows(root).items() if not any(x["effect"] == "not_loaded" for x in f)}
    assert {"claude-code", "copilot-cli", "openclaw"} <= set(loading)
    for harness, findings in loading.items():
        note = next(f for f in findings if f["id"].endswith("-PLUGIN-NO-COMPONENTS"))
        assert note["effect"] == "advisory", harness
        assert "on its own the plugin adds nothing" in note["message"]
        # The row's status says whether the harness installs the bundle; the note only speaks about content.
        assert "installs" not in note["message"]
        # The fix says what this check leaves unread; it does not say what a harness does with a marketplace entry.
        assert "not read by this check" in note["fix"]


@pytest.mark.parametrize(
    "name", ["readme_loader.py", "notice.sh", "license-check.js", "changelog.json", "license-check"]
)
def test_only_plain_documentation_names_are_set_aside(tmp_path: Path, name: str):
    root = _bare_manifest(tmp_path / "p")
    (root / name).write_text("text\n")
    assert plugin_facts(root)["other_files"] is True
    assert not any(f["id"].endswith("NO-COMPONENTS") for findings in _component_rows(root).values() for f in findings)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs named pipes")
@pytest.mark.parametrize("linked", [False, True])
def test_named_pipe_mcp_json_is_skipped_without_blocking(tmp_path: Path, linked: bool):
    root = _bare_manifest(tmp_path / "p")
    fifo = tmp_path / "pipe" if linked else root / ".mcp.json"
    os.mkfifo(fifo)
    if linked:
        (root / ".mcp.json").symlink_to(fifo)
    out: dict = {}
    worker = threading.Thread(target=lambda: out.update(plugin_facts(root)), daemon=True)
    worker.start()
    worker.join(5)
    if worker.is_alive():  # still blocked in open(); unblock it so the thread can end
        os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        worker.join(5)
        pytest.fail("plugin_facts blocked on a named-pipe .mcp.json")
    assert out["mcp_flat"] == 0 and out["components"]["mcp"] == 0
    assert out["other_files"] is True  # something is there, so the folder is not called empty


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0, reason="needs permission bits that bind the current user"
)
@pytest.mark.parametrize("mode", [0o444, 0o000], ids=["listable_not_searchable", "not_listable"])
def test_adds_nothing_note_stays_silent_about_a_subfolder_it_cannot_read(tmp_path: Path, mode: int):
    root = _bare_manifest(tmp_path / "p")
    (root / "data").mkdir()
    (root / "data/hook.sh").write_text("echo hi\n")
    (root / "data").chmod(mode)
    try:
        rows = _component_rows(root)
    finally:
        (root / "data").chmod(0o755)
    assert not any(f["effect"] == "not_loaded" for f in rows["claude-code"])
    assert not any(f["id"].endswith("NO-COMPONENTS") for findings in rows.values() for f in findings)


@pytest.mark.skipif(
    not hasattr(os, "pathconf") or not {os.open, os.unlink} <= os.supports_dir_fd,
    reason="needs a path length limit and directory descriptors",
)
def test_adds_nothing_note_stays_silent_about_a_file_whose_path_is_too_long_to_examine(tmp_path: Path):
    root = _bare_manifest(tmp_path / "p")
    deep = root / "data"
    while len(str(deep)) < os.pathconf(root, "PC_PATH_MAX") - 150:
        deep = deep / ("d" * 40)
    deep.mkdir(parents=True)
    name = "x" * 200 + ".bin"  # the folder's own path fits, the file's does not
    folder = os.open(deep, os.O_RDONLY)
    try:
        os.close(os.open(name, os.O_WRONLY | os.O_CREAT, 0o644, dir_fd=folder))
        rows = _component_rows(root)
    finally:
        os.unlink(name, dir_fd=folder)
        os.close(folder)
    assert not any(f["effect"] == "not_loaded" for f in rows["claude-code"])
    assert not any(f["id"].endswith("NO-COMPONENTS") for findings in rows.values() for f in findings)


@pytest.mark.parametrize(
    "content",
    [b"[]", b"5", b'{"db": {"command": "node"}', b"\xff\xfe{}", b"[" * 200_000 + b"]" * 200_000],
    ids=["array", "number", "truncated", "not-utf-8", "nested-too-deep"],
)
def test_flat_mcp_count_survives_a_file_it_cannot_use(tmp_path: Path, content: bytes):
    root = _bare_manifest(tmp_path / "p")
    (root / ".mcp.json").write_bytes(content)
    assert plugin_facts(root)["mcp_flat"] == 0


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0), reason="needs file permissions")
def test_flat_mcp_count_survives_a_file_it_cannot_open(tmp_path: Path):
    root = _bare_manifest(tmp_path / "p")
    (root / ".mcp.json").write_text(json.dumps({"db": {"command": "node"}}))
    (root / ".mcp.json").chmod(0)
    try:
        assert plugin_facts(root)["mcp_flat"] == 0
    finally:
        (root / ".mcp.json").chmod(0o644)


@pytest.mark.parametrize("manifest", ["name: demo\n", "name: demo\nprovides_hooks: [pre_llm_call]\n", "name: [\n"])
def test_plugin_yaml_alone_does_not_claim_known_empty_contents(tmp_path: Path, manifest: str):
    bare = _manifest(tmp_path / "bare", "plugin.yaml", manifest)
    (bare / "README.md").write_text("text\n")
    assert "HM-PLUGIN-NO-COMPONENTS" not in _status(bare)["hermes"][1]


def test_native_hermes_files_do_not_trigger_known_empty_note(tmp_path: Path):
    native = tmp_path / "native"
    _hermes_native(native)
    assert _status(native)["hermes"] == ("loads", [])


def test_finder_file_inside_the_manifest_folder_does_not_hide_an_empty_plugin(tmp_path: Path):
    root = _bare_manifest(tmp_path / "p")
    (root / ".claude-plugin/.DS_Store").write_bytes(b"\x00")
    assert _status(root)["claude-code"] == ("loads", ["CC-PLUGIN-NO-COMPONENTS"])


def _profile_copy(harness: str = "claude-code") -> dict:
    return copy.deepcopy(next(p for p in load_profiles() if p["harness"] == harness))


def test_profile_without_skill_rules_still_names_its_plugin_findings(tmp_path: Path):
    profile = _profile_copy("codex")
    profile["rules"] = []
    flat = _bare_manifest(tmp_path / "flat")
    (flat / ".mcp.json").write_text(json.dumps({"db": {"command": "node"}}))
    ids = [f["id"] for f in evaluate_plugin(profile, plugin_facts(flat)).findings]
    assert "CX-PLUGIN-MCP-FLAT" in ids
    del profile["rules"], profile["plugin"]["rules"]
    ids = [f["id"] for f in evaluate_plugin(profile, plugin_facts(_bare_manifest(tmp_path / "bare"))).findings]
    assert ids == ["codex-PLUGIN-NO-COMPONENTS"]
