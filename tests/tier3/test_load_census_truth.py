# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load census truth, the hook census, and plugin evidence that survives failures.

Each test drives the real pieces end to end where it can: the staged bundle, the
generated ``setup.sh`` run against a fake container layout, the collector, the
provenance merge, runtime evidence, and the report renderers. Nothing starts
Harbor, Docker, or a model.
"""

from __future__ import annotations

import http.client
import json
import shutil
import socket
import ssl
import subprocess
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner

from skillevaluator import cli as cli_module
from skillevaluator.evaluation import EvaluationService
from skillevaluator.models.result import ValidationResult
from skillevaluator.tier3.eval_core.runtime_evidence import (
    HOOK_CENSUS_FILENAME,
    parse_hook_census,
    summarize_hook_census,
)
from skillevaluator.tier3.harbor import runner
from skillevaluator.tier3.harbor.collector import _attach_load_census, collect_harbor_results
from skillevaluator.tier3.harbor.native_staging import build_native_task_staging, stage_native_bundle
from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package
from skillevaluator.tier3.plugin_native import (
    HARNESS_ADAPTERS,
    HOOK_CENSUS_TEMPLATE,
    LOAD_CENSUS_FILENAME,
    NativeHookSource,
    apply_load_census,
    finalize_native_provenance,
    plugin_load_provenance,
    read_census_file,
    resolve_plugin_load,
)
from skillevaluator.tier3.plugin_runtime import apply_runtime_evidence

TEMPLATE = Path(__file__).resolve().parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
SLUG = "release-helper"
OK_HOOK_ID = "hooks/hooks.json#PreToolUse[0].hooks[0]"


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _plugin(tmp_path: Path) -> Path:
    plugin = tmp_path / SLUG
    _write(
        plugin / ".claude-plugin" / "plugin.json",
        json.dumps({"name": SLUG, "description": "Triage tickets and draft release notes."}),
    )
    _write(
        plugin / "skills" / "ticket-triage" / "SKILL.md",
        "---\nname: ticket-triage\ndescription: Triage tickets.\n---\nTriage body.\n",
    )
    hooks = {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "true"}]}]}}
    _write(plugin / "hooks" / "hooks.json", json.dumps(hooks))
    _write(plugin / "commands" / "review.md", "---\ndescription: Review a change\n---\nReview $ARGUMENTS\n")
    _write(plugin / "agents" / "helper.md", "---\nname: helper\ndescription: Helps\n---\nYou help.\n")
    _write(plugin / "output-styles" / "terse.md", "---\nname: terse\ndescription: Terse\n---\nBe terse.\n")
    _write(plugin / "rules" / "style.md", "Always cite ticket IDs.\n")
    servers = {
        "ok-mcp": {"command": "npx", "args": ["-y", "@example/ok-mcp@1.0.0"]},
        "fail-mcp": {"command": "npx", "args": ["-y", "@example/fail-mcp@1.0.0"]},
    }
    _write(plugin / ".mcp.json", json.dumps({"mcpServers": servers}))
    _write(plugin / ".lsp.json", json.dumps({"go": {"command": "gopls", "extensionToLanguage": {".go": "go"}}}))
    case = {"id": "case-1", "prompt": "Triage.", "expected_output": "A summary.", "assertions": ["Triaged"]}
    _write(plugin / "evals" / "evals.json", json.dumps({"skill_name": SLUG, "evals": [case]}))
    return plugin


@pytest.fixture
def package(tmp_path: Path):
    package = prepare_plugin_eval_package(_plugin(tmp_path), stage_root=tmp_path / "stage", plugin_load="native")
    assert package.native_source is not None
    return package


def _stage(tmp_path: Path, agent: str, source: Any) -> tuple[Path, Any]:
    staging = build_native_task_staging(agent, HARNESS_ADAPTERS[agent], source)
    env_dir = tmp_path / "tasks" / agent / "environment"
    env_dir.mkdir(parents=True)
    stage_native_bundle(env_dir, staging)
    return env_dir / "skilleval", staging


def _run_setup(bundle: Path, root: Path, env: dict[str, str]) -> Path:
    """Run the generated setup.sh against a fake container rooted at ``root``; return the census path."""
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
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    return logs / LOAD_CENSUS_FILENAME


def _found(census: dict[str, Any]) -> set[tuple[str, str]]:
    """Every component a census claims was found (listed or loaded, old or new format)."""
    return {(row["type"], row["name"]) for key in ("loaded", "listed") for row in census.get(key) or ()}


def _plan(agent: str, staging: Any, source: Any) -> dict[str, Any]:
    decisions = resolve_plugin_load("native", [agent], env_mode="docker")
    plan = runner._plugin_load_census_plan("native", decisions, {agent: staging}, source)
    assert plan is not None
    return plan[agent]


def _trial(job: Path, name: str, census: Path | None = None, *, init: dict[str, Any] | None = None) -> Path:
    trial = job / name
    (trial / "agent").mkdir(parents=True)
    if census is not None:
        shutil.copyfile(census, trial / "agent" / LOAD_CENSUS_FILENAME)
    if init is not None:
        lines = [json.dumps({"type": "system", "subtype": "hook_started"}), json.dumps(init)]
        lines.append(json.dumps({"type": "assistant", "message": {"content": []}}))
        _write(trial / "agent" / "claude-code.txt", "\n".join(lines) + "\n")
    return trial


def _scored(job: Path) -> list[dict[str, Any]]:
    """One scored reward row per trial directory, the way the collector passes them."""
    return [{"_trial_root_name": trial.name} for trial in sorted(job.iterdir())]


def _init(*, plugin: str = SLUG, mcp: dict[str, str] | None = None) -> dict[str, Any]:
    statuses = mcp if mcp is not None else {"ok-mcp": "connected", "fail-mcp": "failed"}
    return {
        "type": "system",
        "subtype": "init",
        "plugins": [{"name": plugin, "source": f"{plugin}@inline"}, {"name": "agents-md", "path": "builtin"}],
        "skills": ["dataviz", f"{plugin}:ticket-triage"],
        "agents": ["general-purpose", f"{plugin}:helper"],
        "slash_commands": ["clear", f"{plugin}:review", f"{plugin}:ticket-triage"],
        "mcp_servers": [{"name": f"plugin:{plugin}:{name}", "status": status} for name, status in statuses.items()],
        "output_style": "default",
    }


def _provenance(package: Any, agent: str, summary: dict[str, Any], **extra: Any) -> dict[str, Any]:
    provenance = package.provenance()
    decisions = resolve_plugin_load("native", [agent], env_mode="docker")
    engine = {
        "run_config": {"plugin_load": plugin_load_provenance("native", decisions)},
        "agents": {agent: {"plugin_load_census": summary, **extra}},
    }
    finalize_native_provenance(provenance, engine)
    apply_runtime_evidence(provenance, engine)
    return provenance


def _rows(provenance: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["type"], row["name"]): row for row in provenance["component_coverage"]["components"]}


# --------------------------------------------------------------------------- #
# Only harness evidence makes a component "loaded"                            #
# --------------------------------------------------------------------------- #
def test_claude_failed_mcp_server_is_not_loaded_even_though_its_files_are_staged(tmp_path: Path, package) -> None:
    bundle, staging = _stage(tmp_path, "claude-code", package.native_source)
    root = tmp_path / "container"
    census = _run_setup(bundle, root, {"HOME": str(root / "home"), "CLAUDE_CONFIG_DIR": str(root / "cfg")})
    job = tmp_path / "job"
    _trial(job, "case-1__a", census, init=_init())

    summary = _attach_load_census(
        _scored(job), job, _plan("claude-code", staging, package.native_source), agent="claude-code"
    )
    rows = _rows(_provenance(package, "claude-code", summary))

    assert rows[("mcp", "ok-mcp")]["state"] == "loaded"
    assert rows[("skill", "ticket-triage")]["state"] == "loaded"
    assert rows[("agent", "helper")]["state"] == "loaded"
    assert rows[("command", "review")]["state"] == "loaded"
    failed = rows[("mcp", "fail-mcp")]
    assert failed["state"] != "loaded"
    assert "status failed" in failed["reason"]
    # The harness never reports hooks or output styles at startup: listed, not loaded.
    assert rows[("hook", "hooks/hooks.json")]["state"] == "staged"
    assert rows[("output_style", "terse")]["state"] == "staged"
    assert ("mcp", "fail-mcp") in {(row["type"], row["name"]) for row in summary["not_loaded"]}


def test_claude_init_without_the_plugin_marks_its_components_not_loaded_and_the_run_incomplete(
    tmp_path: Path, package
) -> None:
    bundle, staging = _stage(tmp_path, "claude-code", package.native_source)
    root = tmp_path / "container"
    census = _run_setup(bundle, root, {"HOME": str(root / "home"), "CLAUDE_CONFIG_DIR": str(root / "cfg")})
    job = tmp_path / "job"
    _trial(job, "case-1__a", census, init=_init(plugin="someone-else"))

    summary = _attach_load_census(
        _scored(job), job, _plan("claude-code", staging, package.native_source), agent="claude-code"
    )
    provenance = _provenance(package, "claude-code", summary)

    assert not [row for row in provenance["component_coverage"]["components"] if row["state"] == "loaded"]
    reasons = {row["name"]: row["reason"] for row in summary["not_loaded"]}
    assert "did not load plugin release-helper" in reasons["ok-mcp"]
    assert provenance["partial"] is True
    assert "did not load the plugin" in provenance["native_load_unverified"]["claude-code"]


def test_listing_alone_never_promotes_a_row_to_loaded(tmp_path: Path, package) -> None:
    bundle, staging = _stage(tmp_path, "codex", package.native_source)
    root = tmp_path / "container"
    home = root / "home"
    _write(home / ".agents" / "skills" / "ticket-triage" / "SKILL.md", "x")
    codex_home = root / "codex-home"
    _write(codex_home / "config.toml", 'model = "m"\n')
    census = _run_setup(bundle, root, {"HOME": str(home), "CODEX_HOME": str(codex_home)})
    job = tmp_path / "job"
    _trial(job, "case-1__a", census)

    summary = _attach_load_census(_scored(job), job, _plan("codex", staging, package.native_source), agent="codex")
    rows = _rows(_provenance(package, "codex", summary))

    # The files are where Codex reads them, but nothing proves Codex loaded them.
    for key in (("skill", "ticket-triage"), ("rule", "style.md"), ("mcp", "ok-mcp"), ("mcp", "fail-mcp")):
        assert rows[key]["state"] == "staged", key
        assert "did not confirm" in rows[key]["reason"]


def test_census_promotion_appends_to_the_row_reason(tmp_path: Path, package) -> None:
    provenance = package.provenance()
    for row in provenance["component_coverage"]["components"]:
        if row["name"] == "ok-mcp":
            row["reason"] = "its env/headers are not applied by the runtime (run reported INCOMPLETE)"
    decisions = resolve_plugin_load("native", ["claude-code"], env_mode="docker")
    summary = {
        "agent": "claude-code",
        "mode": "native",
        "trials": 1,
        "fallback_trials": 0,
        "loaded": [{"type": "mcp", "name": "ok-mcp", "evidence": "claude-code system/init event: connected"}],
        "listed": [],
        "not_loaded": [],
    }
    engine = {
        "run_config": {"plugin_load": plugin_load_provenance("native", decisions)},
        "agents": {"claude-code": {"plugin_load_census": summary}},
    }

    finalize_native_provenance(provenance, engine)

    row = _rows(provenance)[("mcp", "ok-mcp")]
    assert row["state"] == "loaded"
    assert "(run reported INCOMPLETE)" in row["reason"]
    assert "loaded natively by claude-code" in row["reason"]


def test_opencode_census_requires_the_launch_env_to_point_at_the_bundle(tmp_path: Path, package) -> None:
    bundle, _staging = _stage(tmp_path, "opencode", package.native_source)
    without = read_census_file(_run_setup(bundle, tmp_path / "plain", {"HOME": str(tmp_path / "plain" / "home")}))
    assert without is not None
    config_backed = {("rule", "style.md"), ("mcp", "ok-mcp"), ("agent", "helper"), ("command", "review")}
    # The image always holds the bundle, so without the launch env nothing proves OpenCode reads it.
    assert not config_backed & _found(without)
    reasons = {(row["type"], row["name"]): row["reason"] for row in without["not_loaded"]}
    assert "OPENCODE_CONFIG" in reasons[("mcp", "ok-mcp")]
    assert "OPENCODE_CONFIG_DIR" in reasons[("agent", "helper")]

    root = tmp_path / "launched"
    container = root / "skilleval"
    env = {
        "HOME": str(root / "home"),
        # setup.sh is re-rooted under ``root``; the launch env names the re-rooted bundle the same way.
        "OPENCODE_CONFIG": f"{container}/native/opencode/opencode.json",
        "OPENCODE_CONFIG_DIR": f"{container}/native/opencode/config",
    }
    launched = read_census_file(_run_setup(bundle, root, env))
    assert launched is not None
    assert config_backed <= _found(launched)


@pytest.mark.parametrize(
    ("config", "listed"),
    [
        ("model: m\nmcp_servers:\n  team-ok-mcp:\n    command: npx\n", False),
        ("mcp_servers:\n  other:\n    image: ghcr.io/x/ok-mcp:1.2.3\n", False),
        ("mcp_servers:\n  xok-mcp:\n    command: a\n", False),
        ("agent:\n  ok-mcp: 1\nmcp_servers:\n  other: {}\n", False),
        ("model: m\nmcp_servers:\n  ok-mcp:\n    command: npx\n", True),
        ("model: m\nmcp_servers:\n  'ok-mcp':\n    command: npx\n", True),
    ],
)
def test_hermes_mcp_census_needs_an_exact_key_under_mcp_servers(
    tmp_path: Path, package, config: str, listed: bool
) -> None:
    bundle, _staging = _stage(tmp_path, "hermes", package.native_source)
    root = tmp_path / "container"
    hermes_home = root / "hermes"
    _write(hermes_home / "config.yaml", config)
    census = read_census_file(_run_setup(bundle, root, {"HOME": str(root / "home"), "HERMES_HOME": str(hermes_home)}))
    assert census is not None
    assert (("mcp", "ok-mcp") in _found(census)) is listed


def test_plugin_loading_table_does_not_call_a_file_listing_verified(tmp_path: Path, package) -> None:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    from skillevaluator.reporting.markdown import MarkdownReporter
    from skillevaluator.reporting.plugin_sections import tier3_plugin_view

    bundle, staging = _stage(tmp_path, "codex", package.native_source)
    root = tmp_path / "container"
    home = root / "home"
    _write(home / ".agents" / "skills" / "ticket-triage" / "SKILL.md", "x")
    _write(root / "codex-home" / "config.toml", 'model = "m"\n')
    census = _run_setup(bundle, root, {"HOME": str(home), "CODEX_HOME": str(root / "codex-home")})
    job = tmp_path / "job"
    _trial(job, "case-1__a", census)
    summary = _attach_load_census(_scored(job), job, _plan("codex", staging, package.native_source), agent="codex")
    payload = {"plugin_provenance": _provenance(package, "codex", summary)}

    view = tier3_plugin_view(payload)
    assert view is not None
    lines: list[str] = []
    MarkdownReporter._render_tier3_plugin(view, lines)
    [row] = [line for line in lines if line.startswith("| codex |")]
    assert "verified" not in row
    assert "0 confirmed by harness, 4 listed (files found, not confirmed)" in row

    templates = Path(cli_module.__file__).parent / "reporting" / "templates"
    env = Environment(loader=FileSystemLoader(str(templates)), autoescape=select_autoescape(["html", "j2"]))
    html = str(env.get_template("plugin_sections.html.j2").module.tier3_plugin_load_section(view))
    assert "verified" not in html.split("<tbody>", 1)[1]
    assert "4 listed (files found, not confirmed)" in html


# --------------------------------------------------------------------------- #
# A forged census cannot promote what was never staged                         #
# --------------------------------------------------------------------------- #
def test_forged_load_and_hook_census_cannot_promote_unstaged_components(tmp_path: Path, package) -> None:
    _bundle, staging = _stage(tmp_path, "hermes", package.native_source)
    job = tmp_path / "job"
    trial = _trial(job, "case-1__a")
    forged = {
        "agent": "hermes",
        "mode": "native",
        "loaded": [
            {"type": "hook", "name": "hooks/hooks.json", "evidence": "forged"},
            {"type": "lsp", "name": "go", "evidence": "forged"},
            {"type": "extension", "name": "x", "evidence": "forged"},
            {"type": "skill", "name": "ticket-triage", "evidence": "skills listing: forged"},
        ],
        "not_loaded": [],
    }
    _write(trial / "agent" / LOAD_CENSUS_FILENAME, json.dumps(forged))
    _write(
        trial / "agent" / HOOK_CENSUS_FILENAME,
        json.dumps({"hook_id": OK_HOOK_ID, "event": "PreToolUse", "exit_code": 0}) + "\n",
    )

    summary = _attach_load_census(_scored(job), job, _plan("hermes", staging, package.native_source), agent="hermes")
    hook_census = summarize_hook_census([parse_hook_census((trial / "agent" / HOOK_CENSUS_FILENAME).read_text())])
    provenance = _provenance(
        package, "hermes", summary, plugin_signals_summary={"with_skill": {"hook_census": hook_census}}
    )

    assert _found(summary) <= {("skill", "ticket-triage")}
    ignored = {(row["type"], row["name"]) for row in summary["not_loaded"] if "ignored" in row["reason"]}
    assert {("hook", "hooks/hooks.json"), ("lsp", "go"), ("extension", "x")} <= ignored
    rows = _rows(provenance)
    assert rows[("hook", "hooks/hooks.json")]["state"] == "unsupported"
    assert rows[("lsp", "go")]["state"] == "unsupported"


def test_census_types_the_agent_does_not_load_natively_are_never_promoted(package) -> None:
    # Codex never loads LSP servers natively (Claude Code now does), so a forged entry must not count.
    summary = {
        "agent": "codex",
        "mode": "native",
        "trials": 1,
        "loaded": [{"type": "lsp", "name": "go", "evidence": "forged harness evidence"}],
        "listed": [],
        "not_loaded": [],
    }
    provenance = _provenance(package, "codex", summary)
    assert _rows(provenance)[("lsp", "go")]["state"] == "unsupported"
    # Without a plugin_load plan nothing is trusted at all.
    coverage = apply_load_census(package.provenance()["component_coverage"], {"codex": summary})
    assert {row["name"]: row["state"] for row in coverage["components"]}["go"] == "unsupported"


# --------------------------------------------------------------------------- #
# Only exact, staged hook ids count                                            #
# --------------------------------------------------------------------------- #
def test_only_hook_ids_that_were_wrapped_for_that_source_count(tmp_path: Path, package) -> None:
    import dataclasses

    command = {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "true"}]}]}
    http_only = {"Stop": [{"hooks": [{"type": "http", "url": "https://hooks.example.invalid/stop"}]}]}
    sources = (
        NativeHookSource("hooks/a.json", "hooks/a.json", {"hooks": command}),
        NativeHookSource("hooks/a.json#x.json", "hooks/a.json#x.json", {"hooks": command}),
        NativeHookSource("hooks/http-only.json", "hooks/http-only.json", {"hooks": http_only}),
    )
    source = dataclasses.replace(package.native_source, hooks=sources)
    _bundle, staging = _stage(tmp_path, "claude-code", source)
    job = tmp_path / "job"
    _trial(job, "case-1__a")  # no census file: the staged fallback still carries the staged hook ids
    summary = _attach_load_census([], job, _plan("claude-code", staging, source), agent="claude-code")
    runs = [
        # Forged: the http-only source has no command handler, so nothing was wrapped.
        {"hook_id": "hooks/http-only.json#Stop[0].hooks[0]", "event": "Stop", "exit_code": 0},
        # A real run of the second source only; its id starts with the first source's name.
        {"hook_id": "hooks/a.json#x.json#PreToolUse[0].hooks[0]", "event": "PreToolUse", "exit_code": 0},
    ]
    hook_census = summarize_hook_census([parse_hook_census("\n".join(json.dumps(run) for run in runs))])
    provenance = package.provenance()
    provenance["component_coverage"]["components"] = [
        {"type": "hook", "name": name, "origin": "packaged", "path": name, "state": "unsupported", "reason": "x"}
        for name in ("hooks/a.json", "hooks/a.json#x.json", "hooks/http-only.json")
    ]
    decisions = resolve_plugin_load("native", ["claude-code"], env_mode="docker")
    engine = {
        "run_config": {"plugin_load": plugin_load_provenance("native", decisions)},
        "agents": {
            "claude-code": {
                "plugin_load_census": summary,
                "plugin_signals_summary": {"with_skill": {"hook_census": hook_census}},
            }
        },
    }
    finalize_native_provenance(provenance, engine)
    apply_runtime_evidence(provenance, engine)

    states = {row["name"]: row["state"] for row in provenance["component_coverage"]["components"]}
    assert states == {
        "hooks/a.json": "unsupported",
        "hooks/a.json#x.json": "exercised",
        "hooks/http-only.json": "unsupported",
    }
    exercised = next(row for row in provenance["component_coverage"]["components"] if row["state"] == "exercised")
    assert "self-reported" in exercised["reason"]


# --------------------------------------------------------------------------- #
# Hook census exit codes                                                       #
# --------------------------------------------------------------------------- #
def _hook_runs(tmp_path: Path, command: str, times: int) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    census = tmp_path / "hook-census.jsonl"
    script = tmp_path / "hook_census.sh"
    script.write_text(
        HOOK_CENSUS_TEMPLATE.read_text(encoding="utf-8").replace(
            "/logs/agent/skilleval-hook-census.jsonl", str(census)
        ),
        encoding="utf-8",
    )
    for _ in range(times):
        subprocess.run(
            ["/bin/sh", str(script), OK_HOOK_ID, "PreToolUse", "--", command],
            env={"PATH": "/usr/bin:/bin"},
            capture_output=True,
            check=False,
            timeout=30,
        )
    return census.read_text(encoding="utf-8")


def test_a_blocking_exit_2_is_counted_as_blocked_not_as_a_failure(tmp_path: Path) -> None:
    from skillevaluator.reporting.plugin_sections import _hook_census_rows

    parsed = parse_hook_census(_hook_runs(tmp_path, "exit 2", 3))

    [row] = parsed["hooks"]
    assert (row["runs"], row.get("blocked"), row["failures"]) == (3, 3, 0)
    assert parsed["total_failures"] == 0 and parsed.get("total_blocked") == 3
    [display] = _hook_census_rows(summarize_hook_census([parsed]))
    assert (display.get("blocked"), display["failures"], display["failure_rate"]) == (3, 0, "0%")


def _hook_provenance(package: Any, hook_census: dict[str, Any]) -> dict[str, Any]:
    provenance = package.provenance()
    decisions = resolve_plugin_load("native", ["claude-code"], env_mode="docker")
    summary = {
        "agent": "claude-code",
        "mode": "native",
        "trials": 1,
        "loaded": [],
        "listed": [{"type": "hook", "name": "hooks/hooks.json", "evidence": "plugin-dir hooks listing: /h"}],
        "not_loaded": [],
        "staged_hook_ids": {"hooks/hooks.json": [OK_HOOK_ID]},
    }
    engine = {
        "run_config": {"plugin_load": plugin_load_provenance("native", decisions)},
        "agents": {
            "claude-code": {
                "plugin_load_census": summary,
                "plugin_signals_summary": {"with_skill": {"hook_census": hook_census}},
            }
        },
    }
    finalize_native_provenance(provenance, engine)
    apply_runtime_evidence(provenance, engine)
    return provenance


def test_a_hook_that_never_started_is_not_exercised(tmp_path: Path, package) -> None:
    missing = _hook_runs(tmp_path / "missing", "/nonexistent/skilleval-hook-tool --check", 3)
    parsed = parse_hook_census(missing)

    # Exit 127 every time: the hook command never started, so no plugin code ran.
    never_ran = _hook_provenance(package, summarize_hook_census([parsed]))
    assert _rows(never_ran)[("hook", "hooks/hooks.json")]["state"] == "staged"
    assert parsed["hooks"][0].get("not_started") == 3

    ran = parse_hook_census(missing + _hook_runs(tmp_path / "ok", "true", 1))
    started = _hook_provenance(package, summarize_hook_census([ran]))
    row = _rows(started)[("hook", "hooks/hooks.json")]
    assert row["state"] == "exercised"
    assert "1 started run(s)" in row["reason"]


# --------------------------------------------------------------------------- #
# Evidence survives failed trials and failed runs                             #
# --------------------------------------------------------------------------- #
def test_census_and_hook_census_are_read_from_trials_that_failed(tmp_path: Path, package) -> None:
    from skillevaluator.tier3.eval_core.plugin_signals import build_plugin_signals_context

    bundle, staging = _stage(tmp_path, "claude-code", package.native_source)
    root = tmp_path / "container"
    census = _run_setup(bundle, root, {"HOME": str(root / "home"), "CLAUDE_CONFIG_DIR": str(root / "cfg")})
    job = tmp_path / "jobs" / f"{SLUG}-claude-code-with"
    for name in ("case-1__a", "case-1__b"):
        trial = _trial(job, name, census, init=_init())
        failure = {"exception_type": "NonZeroAgentExitCodeError", "exception_message": "401 invalid x-api-key"}
        _write(trial / "result.json", json.dumps({"exception_info": failure}))
        _write(
            trial / "agent" / HOOK_CENSUS_FILENAME,
            json.dumps({"hook_id": OK_HOOK_ID, "event": "SessionStart", "exit_code": 0}) + "\n",
        )

    results = collect_harbor_results(
        skill_name=SLUG,
        agents=["claude-code"],
        output_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        skip_baseline=True,
        expected_cases=1,
        expected_case_ids=["case-1"],
        plugin_signals=build_plugin_signals_context(member_skills=["ticket-triage"], wrapper_skills=[SLUG]),
        plugin_load_census={"claude-code": _plan("claude-code", staging, package.native_source)},
    )

    agent = results["agents"]["claude-code"]
    assert agent["num_trials_with"] == 0
    load = agent["plugin_load_census"]
    assert load["trials"] == 2 and load["fallback_trials"] == 0
    assert ("mcp", "ok-mcp") in {(row["type"], row["name"]) for row in load["loaded"]}
    hooks = agent["plugin_signals_summary"]["with_skill"]["hook_census"]
    assert hooks["n_trials"] == 2 and hooks["total_runs"] == 2


def test_an_unscored_trial_that_never_launched_does_not_downgrade_the_census(tmp_path: Path, package) -> None:
    bundle, staging = _stage(tmp_path, "claude-code", package.native_source)
    root = tmp_path / "container"
    census = _run_setup(bundle, root, {"HOME": str(root / "home"), "CLAUDE_CONFIG_DIR": str(root / "cfg")})
    job = tmp_path / "job"
    _trial(job, "case-1__a", census, init=_init())
    failed = _trial(job, "case-1__b")  # agent setup timed out: no census, no reward
    _write(failed / "exception.txt", "AgentSetupTimeoutError: setup timed out\n")

    summary = _attach_load_census(
        [{"_trial_root_name": "case-1__a"}],
        job,
        _plan("claude-code", staging, package.native_source),
        agent="claude-code",
    )

    loaded = {(row["type"], row["name"]) for row in summary["loaded"]}
    assert ("mcp", "ok-mcp") in loaded and ("mcp", "fail-mcp") not in loaded
    assert summary["trials"] == 1 and summary["unscored_trials_without_census"] == 1


def test_a_native_arm_with_no_census_in_any_trial_is_incomplete(tmp_path: Path, package) -> None:
    from skillevaluator.evaluation.tier3_report import _incomplete_skip_reason
    from skillevaluator.reporting.markdown import MarkdownReporter
    from skillevaluator.reporting.plugin_sections import tier3_plugin_view

    _bundle, staging = _stage(tmp_path, "claude-code", package.native_source)
    job = tmp_path / "job"
    for name in ("case-1__a", "case-1__b"):
        _trial(job, name)  # setup.sh never ran: no census, so the launch was never rewritten
    summary = _attach_load_census(
        _scored(job), job, _plan("claude-code", staging, package.native_source), agent="claude-code"
    )

    provenance = _provenance(package, "claude-code", summary)

    assert provenance["partial"] is True
    assert "no load census in any of 2" in provenance["native_load_unverified"]["claude-code"]
    assert _incomplete_skip_reason(provenance).startswith("INCOMPLETE: claude-code: no load census")
    view = tier3_plugin_view({"plugin_provenance": provenance})
    assert view is not None and view["partial"]
    lines: list[str] = []
    MarkdownReporter._render_tier3_plugin(view, lines)
    assert any(line.startswith("**INCOMPLETE:** claude-code: no load census") for line in lines)


def _failed_engine(tmp_path: Path, package: Any) -> tuple[dict[str, Any], Path]:
    run_dir = tmp_path / "results" / "2026-09-30_run"
    run_dir.mkdir(parents=True)
    decisions = resolve_plugin_load("native", ["claude-code"], env_mode="docker")
    census = {
        "agent": "claude-code",
        "mode": "native",
        "trials": 1,
        "fallback_trials": 0,
        "loaded": [{"type": "mcp", "name": "ok-mcp", "evidence": "claude-code system/init event: connected"}],
        "listed": [],
        "not_loaded": [],
    }
    engine = {
        "run_dir": str(run_dir),
        "execution_status": "failed",
        "execution_errors": ["claude-code without-skill Harbor run failed: AgentSetupTimeoutError"],
        "run_config": {"plugin_load": plugin_load_provenance("native", decisions)},
        "agents": {"claude-code": {"plugin_load_census": census, "execution_status": "failed"}},
    }
    return engine, run_dir


def _prepared(tmp_path: Path, package: Any) -> SimpleNamespace:
    package_path = tmp_path / "package"
    package_path.mkdir(exist_ok=True)
    return SimpleNamespace(
        skipped=False,
        skip_reason=None,
        package_path=package_path,
        include_skills=(),
        unresolved_skill_refs=(),
        unresolved_rule_refs=(),
        unresolved_mcp_servers=(),
        mcp_probe_targets=(),
        native_source=None,
        integration_evidence_error=lambda: None,
        provenance=package.provenance,
    )


def test_a_failed_plugin_run_keeps_its_provenance_and_evidence(
    tmp_path: Path, package, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    engine, run_dir = _failed_engine(tmp_path, package)
    captured: dict[str, Any] = {}

    def fake_result(*_args: Any, **kwargs: Any) -> ValidationResult:
        captured.update(kwargs)
        result = ValidationResult(validator_name="AGENT_EVAL")
        result.metadata["agent_eval"] = {
            "execution_status": "failed",
            "plugin_provenance": kwargs.get("plugin_provenance"),
        }
        result.add_error("claude-code without-skill Harbor run failed: AgentSetupTimeoutError")
        return result

    monkeypatch.setattr(
        "skillevaluator.tier3.plugin_eval.prepare_plugin_eval_package", lambda *_a, **_k: _prepared(tmp_path, package)
    )
    monkeypatch.setattr(EvaluationService, "evaluate", lambda _self, _options, **_kwargs: engine)
    monkeypatch.setattr("skillevaluator.tier3.results_location.resolve_latest_results", lambda *_a, **_k: run_dir)
    monkeypatch.setattr("skillevaluator.evaluation.tier3_report.agent_eval_result_from_run", fake_result)
    refreshed: list[Path] = []
    monkeypatch.setattr(
        "skillevaluator.evaluation.tier3_report.refresh_plugin_run_report",
        lambda _plugin, path, **_kwargs: refreshed.append(path),
    )

    result = cli_module._run_agent_eval_or_skip(
        plugin,
        agents="claude-code",
        env_mode="docker",
        skip_baseline=False,
        n_concurrent=1,
        max_agents=1,
        kind="plugin",
    )

    sidecar = json.loads((run_dir / "plugin_provenance.json").read_text(encoding="utf-8"))
    assert "claude-code" in sidecar["load_census"]
    assert "did not complete" in sidecar["execution_incomplete"]
    payload = result.metadata["agent_eval"]
    assert payload["plugin_provenance"]["load_census"]["claude-code"]["loaded"][0]["name"] == "ok-mcp"
    assert captured["dataset_source"] == tmp_path / "package"
    assert result.metadata["skip_reason"].startswith("INCOMPLETE: Tier 3 plugin evaluation did not complete")
    assert result.passed is False
    assert refreshed == [run_dir]


def test_evaluate_plugin_writes_provenance_before_failing(
    tmp_path: Path, package, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    engine, run_dir = _failed_engine(tmp_path, package)
    monkeypatch.setattr(
        "skillevaluator.tier3.plugin_eval.prepare_plugin_eval_package", lambda *_a, **_k: _prepared(tmp_path, package)
    )
    monkeypatch.setattr(EvaluationService, "evaluate", lambda _self, _options, **_kwargs: engine)
    monkeypatch.setattr("skillevaluator.tier3.result_display.render_evaluation_result", lambda *_a, **_k: None)
    refreshed: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "skillevaluator.evaluation.tier3_report.refresh_plugin_run_report",
        lambda _plugin, _path, **kwargs: refreshed.append(kwargs["plugin_provenance"]),
    )

    outcome = CliRunner().invoke(cli_module.cli, ["tier3", "evaluate-plugin", str(plugin), "--progress", "off"])

    assert outcome.exit_code != 0
    assert "did not complete" in outcome.output
    sidecar = json.loads((run_dir / "plugin_provenance.json").read_text(encoding="utf-8"))
    assert sidecar["load_census"]["claude-code"]["loaded"][0]["name"] == "ok-mcp"
    assert sidecar["partial"] is True
    assert refreshed and refreshed[0]["execution_incomplete"] == sidecar["execution_incomplete"]


def test_a_crash_in_the_report_only_sum_of_parts_arm_keeps_the_other_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launched: list[str] = []

    def fake_run(**kwargs: Any) -> tuple[bool, str]:
        launched.append(str(kwargs["job_name"]))
        if str(kwargs["job_name"]).endswith("sumofparts"):
            raise RuntimeError("Docker socket disappeared")
        return True, ""

    monkeypatch.setattr(runner, "_run_harbor", fake_run)
    errors = runner._run_agent_pair(
        skill_name="plugin",
        agent="claude-code",
        model="model",
        env_mode="docker",
        with_skill=tmp_path / "with",
        baseline=tmp_path / "without",
        sum_of_parts=tmp_path / "parts",
        jobs_dir=tmp_path / "jobs",
        run_env={},
        n_attempts=1,
        n_concurrent=3,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        expected_trials=1,
    )

    assert errors == []
    assert len(launched) == 3


# --------------------------------------------------------------------------- #
# The verifier judge retries transient failures                               #
# --------------------------------------------------------------------------- #
class _RecordingTime:
    """The real ``time`` module for the verifier, except ``sleep`` records the backoff instead of waiting."""

    def __init__(self, sleeps: list[float]) -> None:
        self._sleeps = sleeps

    def sleep(self, seconds: float) -> None:
        self._sleeps.append(seconds)

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


@pytest.fixture
def verifier(monkeypatch: pytest.MonkeyPatch):
    import importlib.util

    for name in (
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "SKILL_EVAL_LLM_API_KEY",
        "SKILL_EVAL_LLM_BASE_URL",
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "LLM_JUDGE_MODEL",
        "SKILL_EVAL_JUDGE_MODEL",
        "SKILL_EVAL_LLM_MODEL",
        "LLM_JUDGE_FALLBACK_MODELS",
        "SKILL_EVAL_LLM_MAX_RETRIES",
        "SKILL_EVAL_LLM_RETRY_BASE_DELAY",
        "SKILL_EVAL_LLM_RETRY_MAX_DELAY",
        "SKILL_EVAL_LLM_JUDGE_BUDGET_SEC",
    ):
        monkeypatch.delenv(name, raising=False)
    spec = importlib.util.spec_from_file_location("p3_judge_retry_verifier", TEMPLATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sleeps: list[float] = []
    module.time = _RecordingTime(sleeps)
    module.recorded_sleeps = sleeps
    return module


def _judge_server(statuses: list[int]) -> tuple[ThreadingHTTPServer, list[int]]:
    served: list[int] = []
    reply = json.dumps({"choices": [{"message": {"content": "judge ok"}}]}).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            status = statuses[len(served)] if len(served) < len(statuses) else 200
            served.append(status)
            body = reply if status == 200 else b'{"error": "try later"}'
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    return server, served


@pytest.mark.parametrize(
    ("statuses", "content", "attempts"),
    [
        ([503], "judge ok", 2),
        ([502, 429], "judge ok", 3),
        ([408], "judge ok", 2),
        ([529], "judge ok", 2),  # Anthropic "overloaded"
        ([503, 503, 503], None, 3),
        ([501], None, 1),
        ([400], None, 1),
    ],
)
def test_the_verifier_judge_retries_transient_http_errors(
    verifier, monkeypatch: pytest.MonkeyPatch, statuses: list[int], content: str | None, attempts: int
) -> None:
    server, served = _judge_server(statuses)
    try:
        monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai-compatible")
        monkeypatch.setenv("SKILL_EVAL_LLM_API_KEY", "test-judge-key")
        monkeypatch.setenv("SKILL_EVAL_LLM_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}/v1")
        monkeypatch.setenv("SKILL_EVAL_LLM_MAX_RETRIES", "2")
        result, error = verifier.call_public_llm("judge this", model="judge-model", allow_model_fallback=False)
    finally:
        server.shutdown()
    assert result == content
    assert len(served) == attempts
    assert len(verifier.recorded_sleeps) == attempts - 1
    if content is None:
        assert error and "HTTP" in error


def test_the_verifier_judge_retries_a_read_timeout(verifier, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "judge ok"}}]}).encode()

    def flaky(*_args: Any, **_kwargs: Any) -> Response:
        calls.append("call")
        if len(calls) == 1:
            raise TimeoutError("The read operation timed out")
        if len(calls) == 2:
            raise urllib.error.URLError(ConnectionResetError("connection reset by peer"))
        return Response()

    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-judge-key")
    monkeypatch.setattr(verifier.urllib.request, "urlopen", flaky)

    content, error = verifier.call_public_llm("judge this", model="judge-model")

    assert (content, error) == ("judge ok", None)
    assert len(calls) == 3
    # Full-jitter backoff: attempt n sleeps somewhere in [0, base_delay * 2**n].
    assert len(verifier.recorded_sleeps) == 2
    assert 0.0 <= verifier.recorded_sleeps[0] <= verifier._DEFAULT_BASE_DELAY
    assert 0.0 <= verifier.recorded_sleeps[1] <= verifier._DEFAULT_BASE_DELAY * 2


def _raised_from(outer: BaseException, inner: BaseException) -> BaseException:
    """Return ``outer`` raised while handling ``inner``, the way SDKs wrap transport errors."""
    try:
        try:
            raise inner
        except BaseException as caught:
            raise outer from caught
    except BaseException as raised:
        return raised


def _botocore_cert_error(*, chained: bool) -> BaseException:
    """Build botocore's SSLError for a failed certificate check.

    ``chained`` matches what botocore raises over urllib3 (the cert error two
    links down); otherwise the cert error sits directly in ``kwargs["error"]``.
    """
    import botocore.exceptions
    import urllib3.exceptions

    endpoint = "https://bedrock-runtime.invalid"
    cert = ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    if not chained:
        return botocore.exceptions.SSLError(endpoint_url=endpoint, error=cert)
    urllib3_error = _raised_from(urllib3.exceptions.SSLError(cert), cert)
    return _raised_from(botocore.exceptions.SSLError(endpoint_url=endpoint, error=urllib3_error), urllib3_error)


@pytest.mark.parametrize(
    ("error", "transient"),
    [
        pytest.param(socket.timeout("timed out"), True, id="socket-timeout"),  # noqa: UP041 -- Python 3.9 read timeout
        pytest.param(TimeoutError("The read operation timed out"), True, id="read-timeout"),
        pytest.param(ConnectionResetError("reset"), True, id="connection-reset"),
        pytest.param(http.client.IncompleteRead(b"partial"), True, id="incomplete-read"),
        pytest.param(urllib.error.URLError(ConnectionResetError("reset")), True, id="url-connection-reset"),
        pytest.param(urllib.error.URLError("timed out"), True, id="url-timed-out"),
        pytest.param(urllib.error.URLError(ssl.SSLCertVerificationError("bad cert")), False, id="url-cert"),
        pytest.param(ssl.SSLCertVerificationError("bad cert"), False, id="cert"),
        pytest.param(
            _raised_from(ConnectionError("connect failed"), ssl.SSLCertVerificationError("bad cert")),
            False,
            id="chained-cert",
        ),
        pytest.param(urllib.error.URLError("unknown url type: ftp"), False, id="url-not-network"),
        pytest.param(ValueError("bad json"), False, id="value-error"),
    ],
)
def test_the_verifier_judge_classifies_transient_errors(verifier, error: BaseException, transient: bool) -> None:
    assert verifier._is_transient_judge_error(error) is transient


def test_the_verifier_judge_never_retries_once_its_time_budget_is_spent(verifier) -> None:
    exhausted = verifier._JudgeBudgetExhausted("LLM judge time budget exhausted")

    assert verifier._is_transient_judge_error(exhausted) is False
    assert verifier._is_transient_judge_error(urllib.error.URLError(exhausted)) is False
    assert verifier._classify_bedrock_retry_error(exhausted)[0] is False
    # The type decides, not the message: a read timeout is retried whatever it says.
    assert verifier._is_transient_judge_error(TimeoutError("LLM judge time budget exhausted")) is True


@pytest.mark.parametrize(
    ("code", "transient"),
    [
        (408, True),
        (429, True),
        (500, True),
        (502, True),
        (503, True),
        (504, True),
        (520, True),
        (529, True),
        (400, False),
        (401, False),
        (501, False),
        (505, False),
    ],
)
def test_the_verifier_judge_classifies_http_statuses(verifier, code: int, transient: bool) -> None:
    error = urllib.error.HTTPError("https://judge.invalid/v1", code, "status", hdrs=None, fp=None)
    assert verifier._is_transient_judge_error(error) is transient


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(_botocore_cert_error(chained=True), id="botocore-ssl-chained"),
        pytest.param(_botocore_cert_error(chained=False), id="botocore-ssl-kwargs"),
        pytest.param(ssl.SSLCertVerificationError(1, "certificate verify failed"), id="bare-cert"),
    ],
)
def test_the_verifier_bedrock_classifier_never_retries_a_certificate_failure(verifier, error: BaseException) -> None:
    # botocore's SSLError is an OSError, which the Bedrock classifier otherwise treats as a network blip.
    assert isinstance(error, OSError)
    retriable, _label, _retry_after = verifier._classify_bedrock_retry_error(error)
    assert retriable is False


def test_the_verifier_bedrock_classifier_still_retries_a_dropped_connection(verifier) -> None:
    import botocore.exceptions

    error = botocore.exceptions.EndpointConnectionError(endpoint_url="https://bedrock-runtime.invalid")
    assert verifier._classify_bedrock_retry_error(error)[0] is True


def test_the_verifier_judge_never_retries_a_certificate_failure(verifier, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def bad_cert(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("call")
        raise urllib.error.URLError(ssl.SSLCertVerificationError("certificate verify failed"))

    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-judge-key")
    monkeypatch.setattr(verifier.urllib.request, "urlopen", bad_cert)

    content, error = verifier.call_public_llm("judge this", model="judge-model", allow_model_fallback=False)

    assert content is None and error and "certificate verify failed" in error
    assert calls == ["call"] and verifier.recorded_sleeps == []


def test_the_verifier_judge_does_not_retry_past_its_time_budget(verifier, monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    timeouts: list[float] = []

    class Clock(_RecordingTime):
        def monotonic(self) -> float:
            return now[0]

        def sleep(self, seconds: float) -> None:
            super().sleep(seconds)
            now[0] += seconds

    def slow(_request: Any, timeout: float) -> Any:
        timeouts.append(timeout)
        now[0] += timeout  # every attempt waits out its whole read timeout
        raise TimeoutError("The read operation timed out")

    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-judge-key")
    monkeypatch.setattr(verifier, "time", Clock(verifier.recorded_sleeps))
    monkeypatch.setattr(verifier.urllib.request, "urlopen", slow)
    budget = verifier._resolve_judge_wall_time_budget()
    token = verifier._ACTIVE_JUDGE_DEADLINE.set(budget)  # what _call_required_judge sets for a required judge
    try:
        content, error = verifier.call_public_llm("judge this", model="judge-model", allow_model_fallback=False)
    finally:
        verifier._ACTIVE_JUDGE_DEADLINE.reset(token)

    # The retry only gets the time left, and none starts once the budget is spent.
    assert content is None and error and "time budget exhausted" in error
    assert len(timeouts) == 2 and timeouts[0] == 90 and timeouts[1] <= budget - 90
    assert len(verifier.recorded_sleeps) == 1
    assert now[0] <= budget
