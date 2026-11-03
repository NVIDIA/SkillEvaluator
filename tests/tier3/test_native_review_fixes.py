# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regressions for private native staging and harness-reported load evidence."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from skillevaluator.plugin_inventory import plugin_inventory_for_root
from skillevaluator.tier3.harbor.native_staging import build_native_task_staging, stage_native_bundle
from skillevaluator.tier3.plugin_native import ClaudeCodeAdapter, apply_claude_init_evidence
from skillevaluator.tier3.plugin_prepare import prepare_runtime_package, stage_wrapper_skill


def _prepared(root: Path, *, mcp: dict | None = None):
    files = {
        ".claude-plugin/plugin.json": {"name": "demo"},
        "skills/a/SKILL.md": "---\nname: a\ndescription: A.\n---\nUseful instructions.\n",
        "skills/a/eval/dataset.json": [
            {"id": "one", "prompt": "Read the input", "expected_output": "SECRET-ANSWER-42"}
        ],
        "skills/a/eval/files/input.txt": "INPUT-42",
        "skills/a/scripts/run.py": "print('runtime helper')\n",
    }
    if mcp is not None:
        files[".mcp.json"] = {"mcpServers": {"remote": mcp}}
    for rel, value in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
    inventory = plugin_inventory_for_root(root)
    assert inventory is not None
    return prepare_runtime_package(inventory)


def test_legacy_evaluation_data_never_reaches_native_or_wrapper_skill_copies(tmp_path: Path) -> None:
    package = _prepared(tmp_path / "plugin")
    staging = build_native_task_staging("claude-code", ClaudeCodeAdapter(), package.native_source)
    stage_native_bundle(tmp_path / "environment", staging)
    plugin = tmp_path / "environment/skilleval/native/claude-code/plugin"
    assert (plugin / "skills/a/scripts/run.py").is_file()
    assert not (plugin / "skills/a/eval").exists()
    assert all("SECRET-ANSWER-42" not in p.read_text() for p in plugin.rglob("*") if p.is_file())
    stage_wrapper_skill(package, tmp_path / "wrapper")
    assert not (tmp_path / "wrapper/skills/a/eval").exists()
    assert (tmp_path / "wrapper/evals/evals.json").is_file()  # retained for the verifier


@pytest.mark.parametrize("dirname", ["eval", "evals", "EVAL", "EVALS"])
def test_runtime_copy_and_sanitizer_remove_both_evaluator_layouts(tmp_path: Path, dirname: str) -> None:
    from skillevaluator.tier3.harbor.adapter import _runtime_skill_copy_ignore, _sanitize_staged_runtime_skills

    source = tmp_path / "source"
    member = source / "skills/member"
    (member / dirname).mkdir(parents=True)
    (member / "SKILL.md").write_text("instructions")
    (member / dirname / "dataset.json").write_text("SECRET-ANSWER-42")
    (member / "helper.txt").write_text("keep me")
    copied = tmp_path / "copied"
    shutil.copytree(source, copied, ignore=_runtime_skill_copy_ignore(source))
    assert not (copied / "skills/member" / dirname).exists()
    assert (copied / "skills/member/helper.txt").read_text() == "keep me"
    unsanitized = tmp_path / "unsanitized"
    shutil.copytree(source, unsanitized)
    _sanitize_staged_runtime_skills(unsanitized)
    assert not (unsanitized / "skills/member" / dirname).exists()


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ({"type": "sse"}, "sse"),
        ({"transport": "sse"}, "sse"),
        ({"type": "http"}, "http"),
        ({}, "http"),
        ({"type": "sse", "transport": "http"}, "http"),
    ],
)
def test_prepared_remote_transport_survives_real_bundle_staging(
    tmp_path: Path, declaration: dict, expected: str
) -> None:
    package = _prepared(tmp_path / "plugin", mcp={"url": "https://example.com/mcp", **declaration})
    staging = build_native_task_staging("claude-code", ClaudeCodeAdapter(), package.native_source)
    stage_native_bundle(tmp_path / "environment", staging)
    config = json.loads((tmp_path / "environment/skilleval/native/claude-code/plugin/.mcp.json").read_text())
    assert config["mcpServers"]["remote"]["type"] == expected


def _census(*types: str) -> tuple[dict, list[dict]]:
    declared = [{"type": kind, "name": "a", "harness_name": "a"} for kind in types]
    return {"loaded": [], "listed": [{**row, "evidence": "file exists"} for row in declared]}, declared


def test_explicit_empty_claude_skill_list_is_not_overridden_by_slash_commands() -> None:
    census, declared = _census("skill")
    result = apply_claude_init_evidence(
        census,
        declared,
        {"plugins": [{"name": "demo"}], "skills": [], "slash_commands": ["demo:a"]},
        plugin="demo",
    )
    assert result["loaded"] == []
    assert result["not_loaded"][0]["type"] == "skill"


@pytest.mark.parametrize("has_skill_field", [True, False])
def test_claude_shared_slash_name_does_not_confirm_two_distinct_components(has_skill_field: bool) -> None:
    census, declared = _census("skill", "command")
    init = {"plugins": [{"name": "demo"}], "slash_commands": ["demo:a"]}
    if has_skill_field:
        init["skills"] = ["demo:a"]
    result = apply_claude_init_evidence(census, declared, init, plugin="demo")
    assert {(row["type"], row["name"]) for row in result["loaded"]} == ({("skill", "a")} if has_skill_field else set())
    assert any(row["type"] == "command" for row in result["listed"])


def test_older_claude_init_without_skills_still_confirms_unambiguous_skill() -> None:
    census, declared = _census("skill")
    result = apply_claude_init_evidence(
        census, declared, {"plugins": [{"name": "demo"}], "slash_commands": ["demo:a"]}, plugin="demo"
    )
    assert result["loaded"][0]["type"] == "skill"


def test_ambiguous_init_does_not_promote_staging_without_a_file_census() -> None:
    _, declared = _census("skill", "command")
    result = apply_claude_init_evidence(
        {"staged": declared},
        declared,
        {"plugins": [{"name": "demo"}], "slash_commands": ["demo:a"]},
        plugin="demo",
    )
    assert result["loaded"] == result["listed"] == []
    assert {row["type"] for row in result["staged"]} == {"skill", "command"}


def _run_setup(task: Path, home: Path, logs: Path, *, extra_env: dict | None = None) -> dict:
    bundle = task / "environment/skilleval"
    script = (bundle / "native/setup.sh").read_text()
    script = script.replace("/skilleval/", str(bundle) + "/").replace("/logs/agent", str(logs))
    env = {"PATH": os.environ["PATH"], "HOME": str(home), **(extra_env or {})}
    completed = subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr
    return json.loads((logs / "skilleval-load-census.json").read_text())


@pytest.mark.parametrize("agent_name", ["codex", "opencode", "hermes"])
def test_member_projection_matches_real_harbor_register_command_and_generated_census(
    tmp_path: Path, agent_name: str
) -> None:
    from harbor.agents.installed.codex import Codex
    from harbor.agents.installed.hermes import Hermes
    from harbor.agents.installed.opencode import OpenCode

    from skillevaluator.tier3.plugin_execution import NativeRuntimeHooks
    from skillevaluator.tier3.plugin_native import resolve_plugin_load

    package = _prepared(tmp_path / "plugin")
    task = tmp_path / "task"
    environment = task / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (environment / "skills").mkdir()
    stage_wrapper_skill(package, environment / "skills/plugin-demo")
    hooks = NativeRuntimeHooks(package.native_source, resolve_plugin_load("native", [agent_name], env_mode="docker"))
    hooks.stage_with_arm(agent_name, [task])
    assert (environment / "skills/a/SKILL.md").is_file()
    assert not (environment / "skills/a/eval").exists()
    assert (environment / "skills/plugin-demo").exists() == (agent_name == "hermes")

    home = tmp_path / "home"
    home.mkdir()
    hermes_home = home / ".hermes"
    cls = {"codex": Codex, "opencode": OpenCode, "hermes": Hermes}[agent_name]
    agent = object.__new__(cls)
    agent.skills_dir = str(environment / "skills")
    command = agent._build_register_skills_command()
    assert command is not None
    # Harbor pins Hermes state at /tmp/hermes; relocate only that absolute
    # sandbox root for this host process, retaining the real registration code.
    command = command.replace("/tmp/hermes", str(hermes_home))
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "HERMES_HOME": str(hermes_home)}
    subprocess.run(["sh", "-c", command], env=env, check=True, capture_output=True, timeout=15)
    census = _run_setup(task, home, tmp_path / "logs", extra_env={"HERMES_HOME": str(hermes_home)})
    assert {row["name"] for row in census["listed"] if row["type"] == "skill"} == {"a"}
    assert not [row for row in census["not_loaded"] if row["type"] == "skill"]
    assert not census.get("loaded")


def test_real_client_fixture_stream_cannot_confirm_shadowed_command(tmp_path: Path) -> None:
    from skillevaluator.tier3.plugin_execution import NativeRuntimeHooks, collect_native_evidence
    from skillevaluator.tier3.plugin_native import resolve_plugin_load

    command = tmp_path / "plugin/commands/a.md"
    command.parent.mkdir(parents=True)
    command.write_text("---\ndescription: The command.\n---\nDo a command.\n")
    package = _prepared(tmp_path / "plugin")
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "environment/Dockerfile").write_text("FROM python:3.12-slim\n")
    hooks = NativeRuntimeHooks(package.native_source, resolve_plugin_load("native", ["claude-code"], env_mode="docker"))
    hooks.stage_with_arm("claude-code", [task])
    home = tmp_path / "home"
    home.mkdir()
    logs = tmp_path / "jobs/plugin-demo-claude-code-with/trial-1/agent"
    _run_setup(task, home, logs)
    event = {
        "type": "system",
        "subtype": "init",
        "plugins": [{"name": "demo"}],
        "skills": ["demo:a"],
        "slash_commands": ["demo:a"],
    }
    client = tmp_path / "fixture_client.py"
    client.write_text("import json\nprint(json.dumps(" + repr(event) + "))\n")
    completed = subprocess.run([sys.executable, str(client)], capture_output=True, text=True, check=True, timeout=15)
    (logs / "claude-code.txt").write_text(completed.stdout)
    evidence = collect_native_evidence(hooks, tmp_path / "jobs", "plugin-demo")["claude-code"]
    assert {row["type"] for row in evidence["loaded"]} == {"skill"}
    assert any(row["type"] == "command" for row in evidence["listed"])
