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
    # OpenClaw reads this bundle as Claude format, where commands were not verified: a note, not a degraded status.
    assert s["openclaw"][0] == "loads" and {"OW-PLUGIN-COMMANDS", "OW-PLUGIN-MCP"} <= set(s["openclaw"][1])
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


BROKEN = '{"name": "demo", "version": '


def _manifest(root: Path, rel: str, text: str) -> Path:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text, encoding="utf-8")
    return root


def _bundle(root: Path, manifests: tuple[str, ...], kinds: tuple[str, ...] = ()) -> Path:
    """A bundle with one skill, the given manifests and one component of each kind in ``kinds``.

    ``agent`` stands for a root plugin.json that carries the Agent Plugins schema.
    """
    _plugin(root, None, components=("skills", "commands") if "commands" in kinds else ("skills",))
    for manifest in manifests:
        path = root / ("plugin.json" if manifest == "agent" else manifest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"$schema": AGENT_V1, "name": "demo"} if manifest == "agent" else {"name": "demo"}))
    if "agents" in kinds:
        (root / "agents").mkdir()
        (root / "agents/helper.md").write_text("---\nname: helper\ndescription: Helps.\n---\nHelp\n")
    if "hooks" in kinds:
        (root / "hooks").mkdir()
        (root / "hooks/hooks.json").write_text(json.dumps({"hooks": {"Stop": []}}))
    if "mcp" in kinds:
        (root / "mcp.json").write_text(json.dumps({"mcpServers": {"a": {"type": "stdio", "command": "x"}}}))
    if "lsp" in kinds:
        (root / ".lsp.json").write_text(json.dumps({"srv": {"command": "x"}}))
    return root


def _plugin_findings(root: Path, harness: str) -> tuple[str, dict[str, dict[str, str]]]:
    profile = next(p for p in load_profiles() if p["harness"] == harness)
    result = evaluate_plugin(profile, plugin_facts(root))
    assert len({f["id"] for f in result.findings}) == len(result.findings)
    return result.status, {f["id"]: f for f in result.findings}


def test_component_tables_separate_tested_from_untested_kinds():
    cells = {p["harness"]: p["plugin"]["components"] for p in load_profiles()}

    def kinds(outcome: str) -> dict[str, list[str]]:
        found = {h: sorted(k for k, v in table.items() if v == outcome) for h, table in cells.items()}
        return {h: k for h, k in found.items() if k}

    assert kinds("not_observed") == {"codex": ["agents", "commands"], "openclaw": ["agents", "hooks"]}
    assert kinds("untested") == {
        "codex": ["lsp"],
        "copilot-cli": ["lsp"],
        "hermes": ["agents", "commands", "hooks", "lsp", "mcp"],
        "openclaw": ["commands"],
    }


def test_every_component_outcome_has_an_effect_and_a_message():
    from skillevaluator.harness.plugin import OUTCOME_EFFECT, OUTCOME_TEXT

    used = {v for p in load_profiles() for v in p["plugin"]["components"].values()}
    assert used <= set(OUTCOME_EFFECT)
    assert {o for o, effect in OUTCOME_EFFECT.items() if effect} == set(OUTCOME_TEXT)
    assert OUTCOME_EFFECT["untested"] == "advisory"
    assert "when tested" not in OUTCOME_TEXT["untested"]
    assert "not registered" not in OUTCOME_TEXT["not_observed"]
    # What one harness's documentation says belongs in that harness's profile, not in a shared outcome text.
    assert not any("documentation" in text for text in OUTCOME_TEXT.values())


def test_untested_component_kind_is_a_note_and_does_not_degrade(tmp_path: Path):
    with_lsp = _bundle(tmp_path / "l", (".claude-plugin/plugin.json",), ("lsp",))
    with_commands = _bundle(tmp_path / "c", (".claude-plugin/plugin.json",), ("commands",))
    agent = _bundle(tmp_path / "a", ("agent",), ("lsp",))
    native = _bundle(tmp_path / "n", ("plugin.yaml",), ("commands", "agents", "hooks", "mcp", "lsp"))
    cases = [
        (with_lsp, "codex", ["CX-PLUGIN-LSP"]),
        (with_commands, "openclaw", ["OW-PLUGIN-COMMANDS"]),
        (agent, "hermes", ["HM-PLUGIN-LSP"]),
        (native, "hermes", [f"HM-PLUGIN-{kind}" for kind in ("COMMANDS", "AGENTS", "HOOKS", "MCP", "LSP")]),
    ]
    for root, harness, ids in cases:
        status, findings = _plugin_findings(root, harness)
        assert status == "loads", harness
        for rule_id in ids:
            finding = findings[rule_id]
            assert finding["effect"] == "advisory" and finding["outcome"] == "untested", rule_id
            assert "not tested on this harness" in finding["message"] and "when tested" not in finding["message"]
            assert finding["fix"]


def test_component_kinds_with_a_negative_test_still_degrade(tmp_path: Path):
    claude = _bundle(tmp_path / "c", (".claude-plugin/plugin.json",), ("agents", "hooks"))
    for harness, ids in (("codex", ["CX-PLUGIN-AGENTS"]), ("openclaw", ["OW-PLUGIN-AGENTS", "OW-PLUGIN-HOOKS"])):
        status, findings = _plugin_findings(claude, harness)
        assert status == "degraded", harness
        for rule_id in ids:
            finding = findings[rule_id]
            assert finding["effect"] == "degraded" and finding["outcome"] == "not observed", rule_id
            assert finding["message"].endswith("did not reach the model or run when tested.")


def test_codex_commands_stay_degraded_and_say_what_was_tested(tmp_path: Path):
    # The helper's command body uses $ARGUMENTS, the one command shape that was tested on Codex.
    manifests = {"c": ".claude-plugin/plugin.json", "x": ".codex-plugin/plugin.json", "u": ".cursor-plugin/plugin.json"}
    bundles = [_bundle(tmp_path / name, (manifest,), ("commands",)) for name, manifest in manifests.items()]
    for root in [*bundles, _bundle(tmp_path / "a", ("agent",), ("commands",))]:
        status, findings = _plugin_findings(root, "codex")
        finding = findings["CX-PLUGIN-COMMANDS"]
        assert status == "degraded", root.name
        assert finding["effect"] == "degraded" and finding["outcome"] == "not observed"
        assert finding["message"] == (
            "Plugin commands did not reach the model when tested. "
            "The tested command used $ARGUMENTS; commands without $ARGUMENTS were not tested."
        )
        assert finding["fix"] and finding["confidence"] == "medium"
    refused = tmp_path / "refused"
    _plugin(refused, "plugin.yaml", components=("commands",))
    assert _status(refused)["codex"] == ("not_loaded", ["CX-PLUGIN-FORMAT"])


def test_openclaw_commands_degrade_unless_the_bundle_is_read_as_claude_format(tmp_path: Path):
    claude, codex, cursor = (f".{name}-plugin/plugin.json" for name in ("claude", "codex", "cursor"))
    cases = {
        "claude": ((claude,), "untested"),
        "claude-agent": ((claude, "agent"), "untested"),
        "plain-json": (("plugin.json",), "untested"),
        "codex": ((codex,), "not observed"),
        "cursor": ((cursor,), "not observed"),
        "agent": (("agent",), "not observed"),
        # .codex-plugin and .cursor-plugin win over .claude-plugin when OpenClaw picks the bundle format.
        "claude-codex": ((claude, codex), "not observed"),
        "claude-cursor": ((claude, cursor), "not observed"),
    }
    for name, (manifests, outcome) in cases.items():
        status, findings = _plugin_findings(_bundle(tmp_path / name, manifests, ("commands",)), "openclaw")
        finding = findings["OW-PLUGIN-COMMANDS"]
        assert finding["outcome"] == outcome, name
        if outcome == "untested":
            assert status == "loads" and finding["effect"] == "advisory", name
            assert finding["message"] == "1 commands: not tested on this harness."
        else:
            assert status == "degraded" and finding["effect"] == "degraded", name
            assert "only for Claude-format bundles" in finding["message"] and finding["fix"]
    refused = _manifest(
        _plugin(tmp_path / "refused", None, components=("commands",)), ".codex-plugin/plugin.json", BROKEN
    )
    status, ids = _status(refused)["openclaw"]
    assert status == "not_loaded" and "OW-PLUGIN-COMMANDS" not in ids


def test_hermes_component_findings_are_scoped_to_portable_packages(tmp_path: Path):
    # Hermes was tested with an Agent Plugins bundle (a portable package) only, so only that shape has a result.
    package = _bundle(tmp_path / "a", ("agent",), ("commands", "agents", "hooks", "mcp"))
    status, findings = _plugin_findings(package, "hermes")
    assert status == "degraded"
    for kind, confidence in (("COMMANDS", "high"), ("AGENTS", "medium"), ("HOOKS", "high")):
        finding = findings[f"HM-PLUGIN-{kind}"]
        assert finding["effect"] == "degraded" and finding["outcome"] == "not observed", kind
        assert "portable package" in finding["message"] and finding["message"].endswith("when tested.")
        assert finding["confidence"] == confidence, kind
    mcp = findings["HM-PLUGIN-MCP"]
    assert (mcp["effect"], mcp["outcome"], mcp["confidence"]) == ("advisory", "disputed", "medium")
    assert mcp["message"] == (
        "MCP servers of a portable package did not start when tested, "
        "although Hermes documents stdio servers from a root mcp.json."
    )
    assert mcp["fix"]
    only_mcp = _bundle(tmp_path / "m", ("agent",), ("mcp",))
    assert _plugin_findings(only_mcp, "hermes")[0] == "loads"


def test_copilot_lsp_note_says_the_file_is_not_a_documented_place(tmp_path: Path):
    root = _bundle(tmp_path / "c", (".claude-plugin/plugin.json",), ("lsp",))
    status, findings = _plugin_findings(root, "copilot-cli")
    finding = findings["CP-PLUGIN-LSP"]
    assert status == "loads"
    assert (finding["effect"], finding["outcome"], finding["confidence"]) == ("advisory", "untested", "medium")
    assert ".lsp.json" in finding["message"] and "when tested" not in finding["message"]
    # The file stays where Claude Code reads it; Copilot's own config is added next to it.
    assert finding["fix"].startswith("Keep .lsp.json") and "plugin reference" in finding["fix"]
    refused = _bundle(tmp_path / "x", (".codex-plugin/plugin.json",), ("lsp",))
    assert "CP-PLUGIN-LSP" not in _plugin_findings(refused, "copilot-cli")[1]


def test_plugin_rule_confidence_labels():
    labels = {r["id"]: r["confidence"] for p in load_profiles() for r in p["plugin"]["rules"]}
    medium = {
        "CX-PLUGIN-NAMESPACE",
        "CX-PLUGIN-COMMANDS",
        "CP-PLUGIN-MCP-NAMES",
        "CP-PLUGIN-LSP",
        "HM-PLUGIN-DISABLED",
        "HM-PLUGIN-AGENTS",
        "HM-PLUGIN-MCP",
        "OW-PLUGIN-COMMANDS",
    }
    high = {
        "CX-PLUGIN-FORMAT",
        "CX-PLUGIN-SKILL-CAP",
        "CX-PLUGIN-AGENT-HOOKS",
        "CP-PLUGIN-FORMAT",
        "CP-PLUGIN-AGENTV1",
        "HM-PLUGIN-FORMAT",
        "HM-PLUGIN-COMMANDS",
        "HM-PLUGIN-HOOKS",
        "OC-PLUGIN-FORMAT",
    }
    assert {rule_id: labels[rule_id] for rule_id in medium} == dict.fromkeys(medium, "medium")
    assert {rule_id: labels[rule_id] for rule_id in high} == dict.fromkeys(high, "high")


def test_unverified_component_note_reaches_the_panel_and_the_json(tmp_path: Path):
    root = _bundle(tmp_path / "p", (".claude-plugin/plugin.json",), ("lsp",))
    out = tmp_path / "h.json"
    args = ["harness-check", str(root), "-v", "--harness", "codex", "--json", str(out)]
    result = CliRunner().invoke(cli, args, env={"COLUMNS": "120"})
    assert result.exit_code == 0, result.output
    printed = " ".join(result.output.split())
    assert "not tested on this harness" in printed and "when tested" not in printed
    block = json.loads(out.read_text(encoding="utf-8"))["harness_compatibility"]
    row = next(rows for key, rows in block.items() if key.endswith("(plugin)"))[0]
    finding = next(f for f in row["findings"] if f["id"] == "CX-PLUGIN-LSP")
    assert row["status"] == "loads"
    assert (finding["effect"], finding["outcome"]) == ("advisory", "untested") and finding["fix"]


def _profile_copy(harness: str = "claude-code") -> dict:
    return copy.deepcopy(next(p for p in load_profiles() if p["harness"] == harness))


def test_profile_that_names_an_unknown_fact_or_outcome_is_a_clear_error(tmp_path: Path):
    facts = plugin_facts(_plugin(tmp_path / "p", ".claude-plugin/plugin.json", components=("skills", "mcp")))
    newer_fact, newer_outcome = _profile_copy(), _profile_copy()
    newer_fact["plugin"]["rules"].append(
        {"id": "CC-PLUGIN-NEW", "when": "not_a_fact", "effect": "advisory", "message": "m"}
    )
    newer_outcome["plugin"]["components"]["mcp"] = "not_an_outcome"
    with pytest.raises(ValueError, match="CC-PLUGIN-NEW reads the fact 'not_a_fact'"):
        evaluate_plugin(newer_fact, facts)
    with pytest.raises(ValueError, match="unknown plugin component outcome 'not_an_outcome'"):
        evaluate_plugin(newer_outcome, facts)


@pytest.mark.parametrize("outcome", ["not_an_outcome", ["delivered"]])
def test_invalid_profile_outcome_is_reported_even_if_plugin_is_not_loaded(tmp_path: Path, outcome):
    facts = plugin_facts(_plugin(tmp_path / "p", ".codex-plugin/plugin.json", components=()))
    profile = _profile_copy()
    assert evaluate_plugin(profile, facts).status == "not_loaded"
    profile["plugin"]["components"]["mcp"] = outcome
    with pytest.raises(ValueError, match="unknown plugin component outcome"):
        evaluate_plugin(profile, facts)


def test_profile_component_map_must_be_a_mapping(tmp_path: Path):
    facts = plugin_facts(_plugin(tmp_path / "p", ".codex-plugin/plugin.json", components=()))
    profile = _profile_copy()
    profile["plugin"]["components"] = ["delivered"]
    with pytest.raises(ValueError, match="plugin components must be a mapping"):
        evaluate_plugin(profile, facts)


def test_plugin_only_profile_can_report_an_untested_component(tmp_path: Path):
    facts = plugin_facts(_plugin(tmp_path / "p", ".claude-plugin/plugin.json", components=("mcp",)))
    profile = _profile_copy()
    profile["rules"] = []
    profile["plugin"]["components"]["mcp"] = "untested"
    result = evaluate_plugin(profile, facts)
    assert result.status == "loads"
    assert any(finding["outcome"] == "untested" for finding in result.findings)


def test_incompatible_profile_comparison_names_the_rule(tmp_path: Path):
    facts = plugin_facts(_plugin(tmp_path / "p", ".claude-plugin/plugin.json", components=("mcp",)))
    profile = _profile_copy()
    profile["plugin"]["rules"].append(
        {"id": "CC-PLUGIN-NEW", "when": "components.not_a_kind > 0", "effect": "advisory", "message": "m"}
    )
    with pytest.raises(ValueError, match="cannot evaluate rule CC-PLUGIN-NEW"):
        evaluate_plugin(profile, facts)


def test_harness_check_reports_a_profile_it_cannot_read_as_a_usage_error(tmp_path: Path):
    root = _plugin(tmp_path / "p", ".claude-plugin/plugin.json", components=("skills", "mcp"))
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    source = Path(evaluate_plugin.__code__.co_filename).parent / "profiles" / "claude-code.yaml"
    text = source.read_text(encoding="utf-8").replace("mcp: deferred", "mcp: not_an_outcome")
    (profiles / "claude-code.yaml").write_text(text, encoding="utf-8")
    args = ["harness-check", str(root), "--harness", "claude-code", "--harness-profiles", str(profiles)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 2 and "unknown plugin component outcome" in result.output
