# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native plugin loading: adapters, staging, hook wrapping, load census, and coverage.

Every test uses fake plugin trees and fake container layouts. Nothing starts
Harbor, Docker, or an agent.
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.tier3.harbor.native_staging import build_native_task_staging, stage_native_bundle
from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package
from skillevaluator.tier3.plugin_native import (
    HARNESS_ADAPTERS,
    HOOK_CENSUS_SCRIPT,
    HOOK_CENSUS_TEMPLATE,
    NATIVE_AGENT_IMPORT_PATHS,
    STAGED_EVIDENCE,
    WRAPPER_COMPONENTS,
    NativeHookSource,
    PluginLoadError,
    apply_load_census,
    component_support_matrix,
    fallback_census,
    finalize_native_provenance,
    hook_id,
    native_agent_import_path,
    normalize_census,
    parse_frontmatter_yaml,
    plugin_load_provenance,
    resolve_plugin_load,
    summarize_censuses,
    wrap_hook_handler,
    wrap_hook_sources,
)

BYPASS_MARKERS = ("dangerously", "bypasspermissions", "--yolo", "skip-permissions")


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _contained_plugin(tmp_path: Path) -> Path:
    plugin = tmp_path / "release-helper"
    manifest = {
        "name": "release-helper",
        "description": "Triage tickets and draft release notes.",
        "hooks": {
            "PostToolUse": [
                {"matcher": "Write", "hooks": [{"type": "command", "command": "echo post | tee /dev/null"}]}
            ]
        },
    }
    _write(plugin / ".claude-plugin" / "plugin.json", json.dumps(manifest))
    _write(
        plugin / "skills" / "ticket-triage" / "SKILL.md",
        "---\nname: ticket-triage\ndescription: Triage tickets.\n---\nTriage body.\n",
    )
    _write(plugin / "skills" / "ticket-triage" / "evals" / "evals.json", '{"secret": "expected output"}')
    _write(
        plugin / "hooks" / "hooks.json",
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {"type": "command", "command": '"${CLAUDE_PLUGIN_ROOT}"/scripts/check.sh'},
                                {"type": "command", "command": "node", "args": ["${CLAUDE_PLUGIN_ROOT}/x.js"]},
                                {"type": "prompt", "prompt": "Review $ARGUMENTS"},
                            ],
                        }
                    ]
                }
            }
        ),
    )
    _write(plugin / "scripts" / "check.sh", "#!/bin/sh\nexit 0\n")
    _write(plugin / "commands" / "review.md", "---\ndescription: Review a change\n---\nReview $ARGUMENTS\n")
    _write(plugin / "agents" / "helper.md", "---\nname: helper\ndescription: Helps\ntools: Read\n---\nYou help.\n")
    _write(plugin / "output-styles" / "terse.md", "---\nname: terse\ndescription: Terse\n---\nBe terse.\n")
    _write(plugin / "rules" / "style.md", "Always cite ticket IDs.\n")
    _write(
        plugin / ".mcp.json",
        json.dumps({"mcpServers": {"tracker": {"command": "npx", "args": ["-y", "@example/tracker-mcp@1.4.2"]}}}),
    )
    _write(plugin / ".lsp.json", json.dumps({"go": {"command": "gopls", "extensionToLanguage": {".go": "go"}}}))
    _write(plugin / ".env", "TOKEN=do-not-copy\n")
    _write(
        plugin / "evals" / "evals.json",
        json.dumps(
            {
                "skill_name": "release-helper",
                "evals": [
                    {
                        "id": "case-1",
                        "prompt": "Triage the tickets.",
                        "expected_output": "A triage summary.",
                        "assertions": ["Triaged"],
                        "expected_skill": "ticket-triage",
                    }
                ],
            }
        ),
    )
    return plugin


@pytest.fixture
def native_source(tmp_path: Path):
    plugin = _contained_plugin(tmp_path)
    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage", plugin_load="native")
    assert package.native_source is not None
    return package.native_source


def _stage(tmp_path: Path, agent: str, source) -> tuple[Path, list[str], Any]:
    staging = build_native_task_staging(agent, HARNESS_ADAPTERS[agent], source)
    env_dir = tmp_path / "tasks" / agent / "environment"
    env_dir.mkdir(parents=True)
    lines = stage_native_bundle(env_dir, staging)
    return env_dir / "skilleval", lines, staging


def _run_setup(bundle: Path, root: Path, env: dict[str, str]) -> dict[str, Any]:
    """Run the generated setup.sh against a fake container layout rooted at ``root``."""
    container = root / "skilleval"
    shutil.copytree(bundle, container)
    logs = root / "logs" / "agent"
    logs.mkdir(parents=True)
    script = (container / "native" / "setup.sh").read_text(encoding="utf-8")
    script = script.replace("/skilleval/", f"{container}/").replace("/logs/agent", str(logs))
    (root / "setup.sh").write_text(script, encoding="utf-8")
    completed = subprocess.run(
        ["/bin/sh", str(root / "setup.sh")],
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    census = json.loads((logs / "skilleval-load-census.json").read_text(encoding="utf-8"))
    normalized = normalize_census(census)
    assert normalized is not None
    return normalized


# --------------------------------------------------------------------------- #
# Plan resolution and provenance                                               #
# --------------------------------------------------------------------------- #
def test_wrapper_is_the_default_and_keeps_todays_component_modes() -> None:
    decisions = resolve_plugin_load("wrapper", ["claude-code", "codex"], env_mode="docker")
    assert {decision.mode for decision in decisions.values()} == {"wrapper"}
    assert decisions["codex"].components == WRAPPER_COMPONENTS
    assert WRAPPER_COMPONENTS["skill"] == "wrapper" and WRAPPER_COMPONENTS["hook"] == "unsupported"


@pytest.mark.parametrize("agent", ["claude-code", "codex", "opencode", "hermes"])
def test_native_resolves_to_each_harness_adapter_in_a_container(agent: str) -> None:
    decision = resolve_plugin_load("native", [agent], env_mode="docker")[agent]
    assert decision.mode == "native"
    assert decision.adapter == HARNESS_ADAPTERS[agent].adapter_id
    assert decision.reason.startswith("native: ")


def test_native_fails_fast_in_local_mode_and_auto_falls_back_with_a_reason() -> None:
    with pytest.raises(PluginLoadError, match="local environment mode"):
        resolve_plugin_load("native", ["codex"], env_mode="local")
    decision = resolve_plugin_load("auto", ["codex"], env_mode="local")["codex"]
    assert decision.mode == "wrapper"
    assert decision.reason.startswith("auto: local environment mode")
    with pytest.raises(PluginLoadError, match="native Harbor task sources"):
        resolve_plugin_load("native", ["codex"], env_mode="docker", task_source="native_harbor")
    with pytest.raises(PluginLoadError, match="no native plugin adapter"):
        resolve_plugin_load("native", ["cursor-cli"], env_mode="docker")


def test_plugin_load_provenance_matches_the_contract_shape() -> None:
    decisions = resolve_plugin_load("auto", ["claude-code", "codex"], env_mode="docker")
    provenance = plugin_load_provenance("auto", decisions)
    assert set(provenance) == {"requested", "by_agent"}
    assert provenance["requested"] == "auto"
    for entry in provenance["by_agent"].values():
        assert set(entry) == {"mode", "reason", "adapter", "components"}
        assert entry["mode"] in {"native", "wrapper"}
        assert set(entry["components"].values()) <= {"native", "wrapper", "unsupported"}
    assert provenance["by_agent"]["claude-code"]["components"]["hook"] == "native"
    assert provenance["by_agent"]["codex"]["components"]["hook"] == "unsupported"


def test_component_support_matrix() -> None:
    matrix = component_support_matrix()
    native = {agent: sorted(k for k, v in modes.items() if v == "native") for agent, modes in matrix.items()}
    # Claude Code also loads a plugin's LSP servers and its settings.json (agent, subagentStatusLine).
    assert native["claude-code"] == [
        "agent",
        "command",
        "hook",
        "lsp",
        "mcp",
        "output_style",
        "rule",
        "settings",
        "skill",
    ]
    assert native["codex"] == ["mcp", "rule", "skill"]
    assert native["opencode"] == ["agent", "command", "mcp", "rule", "skill"]
    # Hermes has no native plugin path yet: its with-plugin task is the wrapper task.
    assert native["hermes"] == []
    assert {matrix["hermes"][kind] for kind in ("skill", "rule", "mcp")} == {"wrapper"}
    assert matrix["claude-code"]["monitor"] == "unsupported"
    for agent in ("codex", "opencode", "hermes"):
        for kind in ("lsp", "monitor", "settings"):
            assert matrix[agent][kind] == "unsupported"


def test_native_agent_import_paths_cover_supported_routes_and_reject_unknown() -> None:
    assert native_agent_import_path("claude-code", None).endswith(":NativeClaudeCode")
    assert native_agent_import_path(
        "codex", "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorGatewayCodex"
    ).endswith(":NativeGatewayCodex")
    with pytest.raises(PluginLoadError):
        native_agent_import_path("codex", "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorLocalCodex")
    assert {agent for agent, _base in NATIVE_AGENT_IMPORT_PATHS} == set(HARNESS_ADAPTERS)


# --------------------------------------------------------------------------- #
# Hook wrapping                                                                #
# --------------------------------------------------------------------------- #
def test_shell_form_hook_passes_the_original_command_as_one_argument() -> None:
    identifier = hook_id("hooks/hooks.json", "PreToolUse", 0, 1)
    assert identifier == "hooks/hooks.json#PreToolUse[0].hooks[1]"
    wrapped = wrap_hook_handler(
        {"type": "command", "command": "echo a && echo b", "timeout": 5, "shell": "bash"},
        hook_id_value=identifier,
        event="PreToolUse",
    )
    assert wrapped["command"] == (
        f"/bin/sh {HOOK_CENSUS_SCRIPT} 'hooks/hooks.json#PreToolUse[0].hooks[1]' PreToolUse -- 'echo a && echo b'"
    )
    assert wrapped["timeout"] == 5
    assert "shell" not in wrapped


def test_exec_form_hook_stays_exec_form_and_other_handlers_are_unchanged() -> None:
    wrapped = wrap_hook_handler(
        {"type": "command", "command": "node", "args": ["${CLAUDE_PLUGIN_ROOT}/x.js", "--fix"]},
        hook_id_value="src#Stop[0].hooks[0]",
        event="Stop",
    )
    assert wrapped["command"] == "/bin/sh"
    assert wrapped["args"] == [
        HOOK_CENSUS_SCRIPT,
        "src#Stop[0].hooks[0]",
        "Stop",
        "--",
        "node",
        "${CLAUDE_PLUGIN_ROOT}/x.js",
        "--fix",
    ]
    prompt = {"type": "prompt", "prompt": "check"}
    assert wrap_hook_handler(prompt, hook_id_value="x", event="Stop") is prompt


def test_wrapped_hooks_merge_sources_with_ids_that_match_the_static_hook_risk_rows() -> None:
    wrapped = wrap_hook_sources(
        [
            NativeHookSource(
                "hooks/hooks.json", "hooks/hooks.json", {"hooks": {"Stop": [{"hooks": [{"command": "a"}]}]}}
            ),
            NativeHookSource("inline", ".claude-plugin/plugin.json", {"Stop": [{"hooks": [{"command": "b"}]}]}),
        ]
    )
    assert [identifier for _source, _event, identifier in wrapped.ids] == [
        "hooks/hooks.json#Stop[0].hooks[0]",
        "inline#Stop[0].hooks[0]",
    ]
    assert len(wrapped.config["hooks"]["Stop"]) == 2


def test_wrapped_hook_runs_through_the_census_script(tmp_path: Path) -> None:
    """The staged template runs a shell-form hook through ``/bin/sh -c`` and keeps its exit code."""
    wrapped = wrap_hook_handler(
        {"type": "command", "command": "printf out; exit 3"}, hook_id_value="h#Stop[0].hooks[0]", event="Stop"
    )
    script = tmp_path / "hook_census.sh"
    census_file = tmp_path / "census.jsonl"
    script.write_text(
        HOOK_CENSUS_TEMPLATE.read_text(encoding="utf-8").replace(
            "/logs/agent/skilleval-hook-census.jsonl", str(census_file)
        ),
        encoding="utf-8",
    )
    command = wrapped["command"].replace(HOOK_CENSUS_SCRIPT, str(script))
    completed = subprocess.run(["/bin/sh", "-c", command], capture_output=True, text=True, check=False, timeout=30)
    assert completed.returncode == 3
    assert completed.stdout == "out"
    row = json.loads(census_file.read_text(encoding="utf-8"))
    assert row["hook_id"] == "h#Stop[0].hooks[0]" and row["exit_code"] == 3


def test_hooks_with_a_permission_bypass_flag_are_refused(tmp_path: Path) -> None:
    plugin = _contained_plugin(tmp_path)
    _write(
        plugin / "hooks" / "hooks.json",
        json.dumps({"hooks": {"Stop": [{"hooks": [{"command": "claude --dangerously-skip-permissions -p x"}]}]}}),
    )
    with pytest.raises(ValueError, match="permission-bypass"):
        prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage", plugin_load="native")


def test_subagents_with_a_bypass_permission_mode_are_refused(tmp_path: Path) -> None:
    plugin = _contained_plugin(tmp_path)
    _write(
        plugin / "agents" / "helper.md",
        "---\nname: helper\ndescription: x\npermissionMode: bypassPermissions\n---\nx\n",
    )
    with pytest.raises(ValueError, match="bypassPermissions"):
        prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage", plugin_load="native")


def test_wrapper_mode_builds_no_native_snapshot(tmp_path: Path) -> None:
    plugin = _contained_plugin(tmp_path)
    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")
    assert package.native_source is None


# --------------------------------------------------------------------------- #
# Per-harness staging                                                          #
# --------------------------------------------------------------------------- #
def _files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def _assert_no_bypass(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file() and path.name != "hook_census.sh":
            text = path.read_text(encoding="utf-8").casefold()
            assert not any(marker in text for marker in BYPASS_MARKERS), path


def test_claude_code_stages_a_plugin_dir_with_wrapped_hooks_and_no_eval_data(tmp_path: Path, native_source) -> None:
    bundle, lines, staging = _stage(tmp_path, "claude-code", native_source)
    assert lines == ["COPY skilleval/ /skilleval/"]
    plugin = bundle / "native" / "claude-code" / "plugin"
    files = _files(plugin)
    assert {
        ".claude-plugin/plugin.json",
        ".mcp.json",
        "hooks/hooks.json",
        "skills/ticket-triage/SKILL.md",
        "commands/review.md",
        "agents/helper.md",
        "output-styles/terse.md",
        "scripts/check.sh",
    } <= files
    # Eval data, secrets, and unsupported components never reach the agent.
    assert not any(name.startswith(("evals/", "skills/ticket-triage/evals")) for name in files)
    assert ".env" not in files
    # LSP servers are staged as a generated .lsp.json (Claude Code loads them for plugins).
    assert json.loads((plugin / ".lsp.json").read_text(encoding="utf-8")) == {
        "go": {"command": "gopls", "extensionToLanguage": {".go": "go"}}
    }
    manifest = json.loads((plugin / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert manifest == {"name": "release-helper", "description": "Triage tickets and draft release notes."}
    assert json.loads((plugin / ".mcp.json").read_text(encoding="utf-8")) == {
        "mcpServers": {"tracker": {"type": "stdio", "command": "npx", "args": ["-y", "@example/tracker-mcp@1.4.2"]}}
    }
    hooks = json.loads((plugin / "hooks" / "hooks.json").read_text(encoding="utf-8"))["hooks"]
    pre = hooks["PreToolUse"][0]["hooks"]
    assert pre[0]["command"].startswith(f"/bin/sh {HOOK_CENSUS_SCRIPT} 'hooks/hooks.json#PreToolUse[0].hooks[0]' ")
    assert pre[1]["args"][:4] == [HOOK_CENSUS_SCRIPT, "hooks/hooks.json#PreToolUse[0].hooks[1]", "PreToolUse", "--"]
    assert pre[2] == {"type": "prompt", "prompt": "Review $ARGUMENTS"}
    assert hooks["PostToolUse"][0]["hooks"][0]["command"].endswith("-- 'echo post | tee /dev/null'")
    assert (bundle / "hook_census.sh").read_bytes() == HOOK_CENSUS_TEMPLATE.read_bytes()
    assert (bundle / "native" / "claude-code" / "rules" / "style.md").read_text() == "Always cite ticket IDs.\n"
    assert staging.stage_member_skills is False and staging.stage_wrapper_skill is False
    assert staging.workspace_skill_aliases(["ticket-triage"]) == ["release-helper:ticket-triage"]
    _assert_no_bypass(bundle)


def test_claude_code_plugin_copy_skips_env_files_in_any_letter_case(tmp_path: Path) -> None:
    plugin = _contained_plugin(tmp_path)
    # One name per directory: a case-insensitive filesystem cannot hold .env and .ENV side by side.
    _write(plugin / "scripts" / ".ENV", "TOKEN=do-not-copy\n")
    _write(plugin / "config" / ".Env.local", "TOKEN=do-not-copy\n")
    _write(plugin / "config" / ".env.example", "TOKEN=\n")
    _write(plugin / "templates" / ".ENV.EXAMPLE", "TOKEN=\n")
    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage", plugin_load="native")
    bundle, _, _ = _stage(tmp_path, "claude-code", package.native_source)
    files = _files(bundle / "native" / "claude-code" / "plugin")
    assert {"scripts/.ENV", "config/.Env.local"}.isdisjoint(files)
    assert {"config/.env.example", "templates/.ENV.EXAMPLE"} <= files


def test_claude_code_plugin_copy_skips_results_generated_output_and_the_evals_source(tmp_path: Path) -> None:
    """Grading data inside the plugin root never reaches the with-plugin image through the native copy."""
    from skillevaluator.tier3.output_provenance import mark_generated_output_root

    plugin = _contained_plugin(tmp_path)
    results_root = plugin / ".skilleval-results"
    entry = {"expected_output": "SECRET-EXPECTED", "skilleval_canary": {"token": "cnry_x"}}
    _write(
        results_root / "release-helper" / "run1" / "_harbor-tasks" / "case-1" / "tests" / "entry.json",
        json.dumps(entry),
    )
    earlier_run = plugin / "old-out" / "run0"
    mark_generated_output_root(earlier_run)
    _write(earlier_run / "results.json", '{"expected_output": "SECRET-EXPECTED"}')
    # An --evals-source kept inside the plugin under a folder not named evals/.
    shutil.copytree(plugin / "evals", plugin / "datasets")
    _write(plugin / "datasets" / "expected.md", "SECRET-EXPECTED\n")
    package = prepare_plugin_eval_package(
        plugin, stage_root=tmp_path / "stage", plugin_load="native", evals_source=plugin / "datasets"
    )
    staging = build_native_task_staging("claude-code", HARNESS_ADAPTERS["claude-code"], package.native_source)
    env_dir = tmp_path / "task" / "environment"
    env_dir.mkdir(parents=True)

    stage_native_bundle(env_dir, staging, excluded_roots=(results_root,))

    files = _files(env_dir / "skilleval")
    assert "native/claude-code/plugin/skills/ticket-triage/SKILL.md" in files
    assert "native/claude-code/plugin/scripts/check.sh" in files
    leaked = [
        name
        for name in files
        if any(part in name for part in ("entry.json", "results.json", "_harbor-tasks", "old-out", "datasets/"))
        or "SECRET-EXPECTED" in (env_dir / "skilleval" / name).read_text(encoding="utf-8", errors="replace")
    ]
    assert leaked == []


def test_claude_code_plugin_copy_skips_datasets_when_the_evals_source_is_the_plugin_root(tmp_path: Path) -> None:
    plugin = _contained_plugin(tmp_path)
    shutil.copy(plugin / "evals" / "evals.json", plugin / "evals.json")
    package = prepare_plugin_eval_package(
        plugin, stage_root=tmp_path / "stage", plugin_load="native", evals_source=plugin
    )

    bundle, _lines, _staging = _stage(tmp_path, "claude-code", package.native_source)

    files = _files(bundle / "native" / "claude-code" / "plugin")
    assert "skills/ticket-triage/SKILL.md" in files
    assert "evals.json" not in files


def test_claude_code_setup_copies_rules_and_census_lists_the_plugin_dir(tmp_path: Path, native_source) -> None:
    bundle, _lines, _staging = _stage(tmp_path, "claude-code", native_source)
    root = tmp_path / "container"
    config_dir = root / "logs-agent-sessions"
    census = _run_setup(bundle, root, {"HOME": str(root / "home"), "CLAUDE_CONFIG_DIR": str(config_dir)})
    rule = config_dir / "rules" / "skilleval-release-helper" / "style.md"
    assert rule.read_text(encoding="utf-8") == "Always cite ticket IDs.\n"
    # A file listing is "listed", never "loaded": only the harness can confirm loading.
    assert census["loaded"] == []
    listed = {(row["type"], row["name"]) for row in census["listed"]}
    assert {("skill", "ticket-triage"), ("mcp", "tracker"), ("hook", "hooks/hooks.json"), ("hook", "inline")} <= listed
    assert {("rule", "style.md"), ("agent", "helper"), ("command", "review"), ("output_style", "terse")} <= listed
    assert census["agent"] == "claude-code" and census["mode"] == "native"
    assert ("lsp", "go") in listed
    assert census["not_loaded"] == []


def test_components_only_an_additional_manifest_declares_are_not_staged_natively(tmp_path: Path) -> None:
    plugin = tmp_path / "demo"
    _write(plugin / ".claude-plugin" / "plugin.json", json.dumps({"name": "demo", "description": "Demo."}))
    _write(plugin / ".codex-plugin" / "plugin.json", json.dumps({"name": "demo", "hooks": "./codex-hooks.json"}))
    codex_hooks = {
        "hooks": {"PostToolUse": [{"matcher": "Write", "hooks": [{"type": "command", "command": "./x.sh"}]}]}
    }
    _write(plugin / "codex-hooks.json", json.dumps(codex_hooks))
    _write(plugin / ".cursor-plugin" / "plugin.json", json.dumps({"name": "demo", "agents": "./cursor-agents"}))
    _write(plugin / "cursor-agents" / "runner.md", "---\nname: runner\ndescription: Runs\n---\nRun.\n")
    _write(plugin / "skills" / "alpha" / "SKILL.md", "---\nname: alpha\ndescription: A.\n---\nBody\n")
    _write(plugin / "evals" / "evals.json", json.dumps([{"id": "c1", "prompt": "p", "expected_output": "o"}]))
    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage", plugin_load="native")
    source = package.native_source

    assert source.hooks == () and source.texts == ()
    bundle, _lines, staging = _stage(tmp_path, "claude-code", source)
    hooks = json.loads((bundle / "native" / "claude-code" / "plugin" / "hooks" / "hooks.json").read_text())
    assert hooks["hooks"] == {}
    assert {row["type"] for row in staging.bundle.declared} == {"skill"}
    coverage = package.provenance()["component_coverage"]
    census = _run_setup(bundle, tmp_path / "container", {"HOME": str(tmp_path / "home")})
    promoted = apply_load_census(coverage, {"claude-code": summarize_censuses("claude-code", "native", [census])})
    states = {(row["type"], row["name"]): row["state"] for row in promoted["components"]}
    assert states[("hook", "codex-hooks.json")] == "unsupported"
    assert states[("agent", "runner")] == "unsupported"


def test_hook_sources_without_staged_handlers_stay_not_loaded(tmp_path: Path, native_source) -> None:
    flat = NativeHookSource("flat-hooks.json", "flat-hooks.json", {"hooks": {"stop": [{"command": "echo hi"}]}})
    prompt_only = NativeHookSource(
        "prompt-hooks.json",
        "prompt-hooks.json",
        {"hooks": {"Stop": [{"hooks": [{"type": "prompt", "prompt": "Summarize the session"}]}]}},
    )
    source = dataclasses.replace(native_source, hooks=(flat, prompt_only))
    bundle, _lines, _staging = _stage(tmp_path, "claude-code", source)
    root = tmp_path / "container"

    census = _run_setup(bundle, root, {"HOME": str(root / "home"), "CLAUDE_CONFIG_DIR": str(root / "cfg")})

    assert ("hook", "flat-hooks.json") in {(row["type"], row["name"]) for row in census["not_loaded"]}
    assert ("hook", "prompt-hooks.json") in {(row["type"], row["name"]) for row in census["listed"]}
    coverage = _coverage(
        {"type": "hook", "name": "flat-hooks.json", "path": "flat-hooks.json", "state": "staged", "reason": "x"},
        {"type": "hook", "name": "prompt-hooks.json", "path": "prompt-hooks.json", "state": "staged", "reason": "x"},
    )
    plugin_load = plugin_load_provenance("native", resolve_plugin_load("native", ["claude-code"], env_mode="docker"))
    promoted = apply_load_census(
        coverage, {"claude-code": summarize_censuses("claude-code", "native", [census])}, plugin_load
    )
    rows = {row["name"]: row for row in promoted["components"]}
    # A listing never proves a hook loaded; the row stays staged and says so.
    assert {name: row["state"] for name, row in rows.items()} == {
        "flat-hooks.json": "staged",
        "prompt-hooks.json": "staged",
    }
    assert "did not confirm" in rows["prompt-hooks.json"]["reason"]
    assert "no hook handlers staged" in rows["flat-hooks.json"]["reason"]


def test_codex_stages_agents_md_and_mcp_tables(tmp_path: Path, native_source) -> None:
    bundle, _lines, staging = _stage(tmp_path, "codex", native_source)
    native = bundle / "native" / "codex"
    assert "## style.md\n\nAlways cite ticket IDs." in (native / "AGENTS.md").read_text(encoding="utf-8")
    tables = tomllib.loads((native / "mcp_servers.toml").read_text(encoding="utf-8"))
    assert tables == {"mcp_servers": {"tracker": {"command": "npx", "args": ["-y", "@example/tracker-mcp@1.4.2"]}}}
    assert staging.stage_member_skills is True and staging.plugin_mcp_via_task is False
    root = tmp_path / "container"
    home = root / "home"
    (home / ".agents" / "skills" / "ticket-triage").mkdir(parents=True)
    (home / ".agents" / "skills" / "ticket-triage" / "SKILL.md").write_text("x")
    codex_home = root / "codex-home"
    codex_home.mkdir(parents=True)
    (codex_home / "config.toml").write_text('model = "m"\n', encoding="utf-8")
    census = _run_setup(bundle, root, {"HOME": str(home), "CODEX_HOME": str(codex_home)})
    merged = tomllib.loads((codex_home / "config.toml").read_text(encoding="utf-8"))
    assert merged["model"] == "m" and merged["mcp_servers"]["tracker"]["args"] == ["-y", "@example/tracker-mcp@1.4.2"]
    assert {(row["type"], row["name"]) for row in census["listed"]} == {
        ("skill", "ticket-triage"),
        ("rule", "style.md"),
        ("mcp", "tracker"),
    }
    not_loaded = {(row["type"], row["name"]) for row in census["not_loaded"]}
    assert ("hook", "hooks/hooks.json") in not_loaded and ("agent", "helper") in not_loaded
    assert all("unsupported by the codex native adapter" in row["reason"] for row in census["not_loaded"])
    _assert_no_bypass(bundle)


def test_codex_setup_refuses_to_write_through_a_link(tmp_path: Path, native_source) -> None:
    bundle, _lines, _staging = _stage(tmp_path, "codex", native_source)
    root = tmp_path / "container"
    codex_home = root / "codex-home"
    codex_home.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("original", encoding="utf-8")
    (codex_home / "AGENTS.md").symlink_to(outside)
    census = _run_setup(bundle, root, {"HOME": str(root / "home"), "CODEX_HOME": str(codex_home)})
    assert outside.read_text(encoding="utf-8") == "original"
    assert "refused-link" in census.get("setup_errors", "")
    assert ("rule", "style.md") in {(row["type"], row["name"]) for row in census["not_loaded"]}


def test_opencode_stages_config_instructions_agents_and_commands(tmp_path: Path, native_source) -> None:
    bundle, _lines, _staging = _stage(tmp_path, "opencode", native_source)
    native = bundle / "native" / "opencode"
    config = json.loads((native / "opencode.json").read_text(encoding="utf-8"))
    assert config == {
        "instructions": ["/skilleval/native/opencode/AGENTS.md"],
        "mcp": {"tracker": {"type": "local", "command": ["npx", "-y", "@example/tracker-mcp@1.4.2"], "enabled": True}},
    }
    agent = (native / "config" / "agents" / "helper.md").read_text(encoding="utf-8")
    # `tools: Read` keeps the subagent read-only in OpenCode.
    assert parse_frontmatter_yaml(agent) == {
        "description": "Helps",
        "mode": "subagent",
        "permission": {"*": "deny", "read": "allow"},
    }
    assert "tools" not in agent
    command = (native / "config" / "commands" / "review.md").read_text(encoding="utf-8")
    assert parse_frontmatter_yaml(command) == {"description": "Review a change"}
    assert HARNESS_ADAPTERS["opencode"].launch_env() == {
        "OPENCODE_CONFIG": "/skilleval/native/opencode/opencode.json",
        "OPENCODE_CONFIG_DIR": "/skilleval/native/opencode/config",
    }
    _assert_no_bypass(bundle)


def test_hermes_keeps_rules_in_the_wrapper_and_mcp_through_harbor(tmp_path: Path, native_source) -> None:
    bundle, _lines, staging = _stage(tmp_path, "hermes", native_source)
    assert staging.stage_wrapper_skill is True and staging.plugin_mcp_via_task is True
    assert _files(bundle) == {"hook_census.sh", "native/setup.sh"}
    root = tmp_path / "container"
    hermes_home = root / "hermes"
    (hermes_home / "skills" / "ticket-triage").mkdir(parents=True)
    (hermes_home / "skills" / "ticket-triage" / "SKILL.md").write_text("x")
    (hermes_home / "config.yaml").write_text("mcp_servers:\n  tracker:\n    command: npx\n", encoding="utf-8")
    census = _run_setup(bundle, root, {"HOME": str(root / "home"), "HERMES_HOME": str(hermes_home)})
    assert {(row["type"], row["name"]) for row in census["listed"]} == {("skill", "ticket-triage"), ("mcp", "tracker")}


def test_census_marks_components_missing_after_setup(tmp_path: Path, native_source) -> None:
    bundle, _lines, _staging = _stage(tmp_path, "hermes", native_source)
    root = tmp_path / "container"
    census = _run_setup(bundle, root, {"HOME": str(root / "home"), "HERMES_HOME": str(root / "hermes")})
    assert census["loaded"] == [] and census["listed"] == []
    assert all(row["reason"].startswith("not found after setup: ") for row in census["not_loaded"][:2])


def test_unsafe_member_skill_name_cannot_inject_into_the_setup_script(tmp_path: Path, native_source) -> None:
    evil = tmp_path / "evil$(touch pwned)"
    _write(evil / "SKILL.md", "---\nname: evil\ndescription: x\n---\nx\n")
    source = type(native_source)(**{**native_source.__dict__, "member_skills": (evil,)})
    bundle, _lines, _staging = _stage(tmp_path, "codex", source)
    root = tmp_path / "container"
    _run_setup(bundle, root, {"HOME": str(root / "home"), "CODEX_HOME": str(root / "codex")})
    assert not (root / "pwned").exists() and not Path("pwned").exists()


# --------------------------------------------------------------------------- #
# Adapter integration                                                          #
# --------------------------------------------------------------------------- #
def _package_tasks(tmp_path: Path, agent: str) -> tuple[Path, Any]:
    from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks

    plugin = _contained_plugin(tmp_path)
    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage", plugin_load="native")
    staging = build_native_task_staging(agent, HARNESS_ADAPTERS[agent], package.native_source)
    tasks = generate_harbor_tasks(
        package.package_path,
        tmp_path / "out" / agent / "with",
        with_skill=True,
        workspace_skill_paths=list(package.include_skills),
        workspace_mode="group",
        native_plugin=staging,
    )
    return tasks[0], package


def test_claude_code_task_drops_the_wrapper_and_member_copies_and_adds_the_bundle(tmp_path: Path) -> None:
    task, package = _package_tasks(tmp_path, "claude-code")
    env_dir = task / "environment"
    assert "COPY skilleval/ /skilleval/" in (env_dir / "Dockerfile").read_text(encoding="utf-8")
    assert not any((env_dir / "skills").iterdir())
    assert (env_dir / "skilleval" / "native" / "claude-code" / "plugin" / ".mcp.json").is_file()
    task_toml = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
    assert "mcp_servers" not in task_toml.get("environment", {})
    entry = json.loads((task / "tests" / "entry.json").read_text(encoding="utf-8"))
    assert entry["workspace_skill_names"] == ["release-helper:ticket-triage", "ticket-triage"]
    assert package.package_path.name in entry["evaluated_skill"]


def test_hermes_task_keeps_the_wrapper_skill_and_plugin_mcp_in_task_toml(tmp_path: Path) -> None:
    task, package = _package_tasks(tmp_path, "hermes")
    env_dir = task / "environment"
    assert (env_dir / "skills" / package.package_path.name / "SKILL.md").is_file()
    assert (env_dir / "skills" / "ticket-triage" / "SKILL.md").is_file()
    task_toml = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
    assert [server["name"] for server in task_toml["environment"]["mcp_servers"]] == ["tracker"]


def test_native_staging_is_rejected_for_baseline_arms(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks

    plugin = _contained_plugin(tmp_path)
    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage", plugin_load="native")
    staging = build_native_task_staging("codex", HARNESS_ADAPTERS["codex"], package.native_source)
    with pytest.raises(ValueError, match="with-plugin arm only"):
        generate_harbor_tasks(package.package_path, tmp_path / "out", with_skill=False, native_plugin=staging)


# --------------------------------------------------------------------------- #
# Census collection and coverage                                               #
# --------------------------------------------------------------------------- #
def test_collector_reads_the_census_and_falls_back_to_staged(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor.collector import _attach_load_census

    job = tmp_path / "job"
    census = {
        "agent": "codex",
        "mode": "native",
        "loaded": [
            {"type": "skill", "name": "alpha", "evidence": "skills listing: /root/.agents/skills/alpha/SKILL.md"}
        ],
        "not_loaded": [{"type": "hook", "name": "hooks/hooks.json", "reason": "unsupported"}],
    }
    _write(job / "trial-1" / "agent" / "skilleval-load-census.json", json.dumps(census))
    (job / "trial-2" / "agent").mkdir(parents=True)
    rewards = [{"_trial_root_name": "trial-1"}, {"_trial_root_name": "trial-2"}]
    plan = {"mode": "native", "declared": [{"type": "skill", "name": "alpha"}]}
    summary = _attach_load_census(rewards, job, plan, agent="codex")
    # A census file can only list: its "loaded" entries are read as "listed".
    assert rewards[0]["plugin_load_census"]["loaded"] == []
    assert rewards[0]["plugin_load_census"]["listed"][0]["evidence"].startswith("skills listing")
    assert rewards[1]["plugin_load_census"]["staged"][0]["evidence"] == STAGED_EVIDENCE
    assert summary["trials"] == 2 and summary["fallback_trials"] == 1
    # Listed in only one of two trials: staged, not listed.
    assert summary["loaded"] == [] and summary["listed"] == []
    assert summary["staged"] == [{"type": "skill", "name": "alpha", "evidence": STAGED_EVIDENCE}]
    assert _attach_load_census(rewards, job, None, agent="codex") is None


def test_malformed_or_linked_census_files_are_ignored(tmp_path: Path) -> None:
    from skillevaluator.tier3.plugin_native import read_census_file

    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    assert read_census_file(bad) is None
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"agent": "codex", "mode": "native", "loaded": [], "not_loaded": []}))
    link = tmp_path / "link.json"
    link.symlink_to(good)
    assert read_census_file(link) is None
    assert read_census_file(good) == {"agent": "codex", "mode": "native", "loaded": [], "listed": [], "not_loaded": []}


def test_missing_or_linked_harness_logs_read_as_none(tmp_path: Path) -> None:
    from skillevaluator.tier3.plugin_native import read_harness_log_prefix

    assert read_harness_log_prefix(tmp_path / "agent" / "claude-code.txt") is None
    log = tmp_path / "claude-code.txt"
    log.write_text('{"type": "system"}\n', encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(log)
    assert read_harness_log_prefix(link) is None
    assert read_harness_log_prefix(log) == '{"type": "system"}\n'


def test_collector_keeps_a_claude_census_when_the_harness_log_is_missing(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor.collector import _attach_load_census

    job = tmp_path / "job"
    census = {"agent": "claude-code", "mode": "native", "loaded": [], "not_loaded": []}
    _write(job / "trial-1" / "agent" / "skilleval-load-census.json", json.dumps(census))
    plan = {
        "mode": "native",
        "declared": [{"type": "skill", "name": "alpha"}],
        "harness": {"kind": "claude-code-init", "plugin": "release-helper"},
    }
    summary = _attach_load_census([{"_trial_root_name": "trial-1"}], job, plan, agent="claude-code")
    assert summary["trials"] == 1 and summary["fallback_trials"] == 0


def _coverage(*rows: dict[str, Any]) -> dict[str, Any]:
    from skillevaluator.plugin_components import summarize_coverage

    return summarize_coverage([dict(row) for row in rows])


def test_load_census_promotes_only_harness_evidence_and_never_downgrades() -> None:
    coverage = _coverage(
        {"type": "skill", "name": "alpha", "path": "skills/alpha", "state": "staged", "reason": "staged"},
        {"type": "hook", "name": "hooks/hooks.json", "path": "hooks/hooks.json", "state": "unsupported", "reason": "x"},
        {"type": "mcp", "name": "tracker", "path": ".mcp.json", "state": "exercised", "reason": "ran"},
        {"type": "rule", "name": "style.md", "path": "rules/style.md", "state": "staged", "reason": "staged"},
        {"type": "agent", "name": "broken", "path": None, "state": "invalid", "reason": "missing"},
    )
    summaries = {
        "claude-code": {
            "mode": "native",
            "loaded": [
                {"type": "skill", "name": "alpha", "evidence": "claude-code system/init event: p:alpha"},
                {"type": "mcp", "name": "tracker", "evidence": "claude-code system/init event: connected"},
                {"type": "agent", "name": "broken", "evidence": "init"},
            ],
            "listed": [{"type": "hook", "name": "hooks/hooks.json", "evidence": "plugin-dir hooks listing: /y"}],
            "staged": [{"type": "rule", "name": "style.md", "evidence": STAGED_EVIDENCE}],
        }
    }
    plugin_load = plugin_load_provenance("native", resolve_plugin_load("native", ["claude-code"], env_mode="docker"))
    promoted = apply_load_census(coverage, summaries, plugin_load)
    states = {row["name"]: row["state"] for row in promoted["components"]}
    assert states == {
        "alpha": "loaded",
        # Listed only: natively staged, not proven loaded.
        "hooks/hooks.json": "staged",
        "tracker": "exercised",
        "style.md": "staged",
        "broken": "invalid",
    }
    assert promoted["not_evaluated"] == 1
    assert promoted["components"][0]["reason"].startswith("staged; loaded natively by claude-code")
    # Without a plugin_load plan nothing is promoted.
    assert apply_load_census(coverage, summaries)["components"][0]["state"] == "staged"


def test_wrapper_censuses_never_promote_coverage() -> None:
    coverage = _coverage({"type": "skill", "name": "alpha", "path": None, "state": "staged", "reason": "x"})
    summary = summarize_censuses(
        "codex", "wrapper", [fallback_census("codex", "wrapper", [{"type": "skill", "name": "alpha"}])]
    )
    assert apply_load_census(coverage, {"codex": summary})["components"][0]["state"] == "staged"


def test_finalize_native_provenance_records_plan_census_and_coverage() -> None:
    provenance: dict[str, Any] = {
        "component_coverage": _coverage(
            {"type": "skill", "name": "alpha", "path": None, "state": "staged", "reason": "x"}
        )
    }
    decisions = resolve_plugin_load("native", ["codex"], env_mode="docker")
    census = summarize_censuses(
        "codex",
        "native",
        [
            {
                "agent": "codex",
                "mode": "native",
                "loaded": [{"type": "skill", "name": "alpha", "evidence": "listing: /a"}],
                "not_loaded": [],
            }
        ],
    )
    engine_result = {
        "run_config": {"plugin_load": plugin_load_provenance("native", decisions)},
        "agents": {"codex": {"plugin_load_census": census}},
    }
    finalize_native_provenance(provenance, engine_result)
    assert provenance["plugin_load"]["requested"] == "native"
    assert provenance["load_census"]["codex"]["loaded"][0]["evidence"] == "listing: /a"
    assert provenance["component_coverage"]["components"][0]["state"] == "loaded"
    untouched: dict[str, Any] = {"plugin_name": "p"}
    finalize_native_provenance(untouched, None)
    assert untouched == {"plugin_name": "p"}


def test_setup_scripts_are_posix_sh(tmp_path: Path, native_source) -> None:
    for agent in HARNESS_ADAPTERS:
        bundle, _lines, _staging = _stage(tmp_path / agent, agent, native_source)
        script = bundle / "native" / "setup.sh"
        completed = subprocess.run(["/bin/sh", "-n", str(script)], capture_output=True, text=True, check=False)
        assert completed.returncode == 0, (agent, completed.stderr)
        assert re.search(r"^exit 0$", script.read_text(encoding="utf-8"), re.MULTILINE)
