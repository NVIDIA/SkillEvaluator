# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 coverage and load reporting (proof L21, L33, and the census-merge half of M14).

Each test replays a small, synthetic run with the shapes of the proof's check 17
cases through the shipped code: the real package step builds the static
coverage rows, the collector reads per-trial load and hook census files and the
Claude Code ``system/init`` line from disk, the CLI's provenance step folds the
census and runtime evidence in, and every reporter renders the result.

- ``p06-cc-plugin-absent`` / ``p05-cc-native-mcp-failed`` / ``e03-cc-mcp-pending``:
  a staged component the harness did not load must not read as covered.
- ``e04-plugin-missing-some-trials``: the reason says in how many trials it loaded.
- ``p01`` / ``n06``: a hook row is "exercised" only when every staged handler
  started, and an exercised hook row is never "not observed".
- ``p03`` / ``p04`` / ``p02`` and the TG auditor's SK1 run: components that
  cannot be staged are not counted as "declared, unverified".
- skeptic ``sk-2agent-both-smoke``: one agent's listing never hides another
  agent's load failure.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from _plugin_fixtures import element_text
from rich.console import Console

from skillevaluator import cli as cli_module
from skillevaluator.evaluation.tier3_report import agent_eval_result_from_directory
from skillevaluator.models import ValidationResult
from skillevaluator.reporting import BenchmarkReporter, HTMLReporter, MarkdownReporter
from skillevaluator.reporting.cli import CLIReporter
from skillevaluator.reporting.plugin_sections import tier3_plugin_view
from skillevaluator.reporting.sarif_reporter import SARIFReporter
from skillevaluator.tier3 import plugin_native as pn
from skillevaluator.tier3.harbor import collector
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS
from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package

PLUGIN = "release-kit"
HOOK_IDS = ["hooks/hooks.json#PreToolUse[0].hooks[0]", "hooks/hooks.json#Stop[0].hooks[0]"]
DECLARED = [
    {"type": "skill", "name": "notes"},
    {"type": "mcp", "name": "tools"},
    {"type": "hook", "name": "hooks/hooks.json"},
    {"type": "agent", "name": "reviewer"},
    {"type": "command", "name": "check"},
]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _plugin(root: Path) -> Path:
    plugin = root / PLUGIN
    _write(
        plugin / ".claude-plugin" / "plugin.json",
        json.dumps({"name": PLUGIN, "version": "1.0.0", "description": "Release helpers."}),
    )
    _write(plugin / "skills" / "notes" / "SKILL.md", "---\nname: notes\ndescription: Release notes.\n---\nWrite.\n")
    _write(plugin / "agents" / "reviewer.md", "---\nname: reviewer\ndescription: Reviews.\n---\nReview.\n")
    _write(plugin / "commands" / "check.md", "---\ndescription: Checks a release.\n---\nCheck.\n")
    hooks = {
        "hooks": {
            "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "echo pre"}]}],
            "Stop": [{"hooks": [{"type": "command", "command": "echo stop"}]}],
        }
    }
    _write(plugin / "hooks" / "hooks.json", json.dumps(hooks))
    _write(plugin / ".mcp.json", json.dumps({"mcpServers": {"tools": {"command": "node", "args": ["server.js"]}}}))
    _write(plugin / "server.js", "console.log('tools')\n")
    _write(
        plugin / "evals" / "evals.json", json.dumps([{"id": "c1", "prompt": "Notes for v1", "expected_output": "x"}])
    )
    return plugin


def _trial(
    job: Path,
    name: str,
    *,
    plugin_in_init: bool = True,
    mcp_status: str = "connected",
    hook_exits: dict[str, int] | None = None,
) -> str:
    """One with-plugin trial: the setup listing, Claude's init line, and hook census lines."""
    agent = job / name / "agent"
    listing = {
        "agent": "claude-code",
        "mode": "native",
        "listed": [{**item, "evidence": f"plugin-dir listing: {item['name']}"} for item in DECLARED],
        "not_loaded": [],
    }
    _write(agent / pn.LOAD_CENSUS_FILENAME, json.dumps(listing))
    init = {
        "type": "system",
        "subtype": "init",
        "plugins": [{"name": PLUGIN}] if plugin_in_init else [],
        "skills": [f"{PLUGIN}:notes"],
        "agents": [f"{PLUGIN}:reviewer"],
        "slash_commands": [f"{PLUGIN}:check"],
        "mcp_servers": [{"name": f"plugin:{PLUGIN}:tools", "status": mcp_status}],
    }
    _write(agent / pn.CLAUDE_CODE_LOG_FILENAME, json.dumps(init) + "\n")
    lines = [
        json.dumps({"hook_id": hook_id, "event": hook_id.split("#")[1].split("[")[0], "exit_code": exit_code})
        for hook_id, exit_code in (hook_exits or {}).items()
    ]
    _write(agent / "skilleval-hook-census.jsonl", "\n".join(lines) + ("\n" if lines else ""))
    return name


def _activation(*, exercised: list[str], declared: list[str] | None = None) -> dict[str, Any]:
    declared = declared or ["skill:notes", "mcp:tools", "subagent:reviewer", "command:check"]
    return {
        "declared": declared,
        "exercised": exercised,
        "unverified": [name for name in declared if name not in exercised],
        "unavailable": [],
        "exercise_rate": {name: (0.5 if name in exercised else 0.0) for name in declared},
    }


def _run(
    tmp_path: Path,
    trials: list[dict[str, Any]],
    *,
    activation: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[ValidationResult]]:
    """Package, collect, fold, and wrap one native Claude Code run."""
    package = prepare_plugin_eval_package(
        _plugin(tmp_path), stage_root=tmp_path / "stage", plugin_load="native", agents="claude-code", env_mode="docker"
    )
    job = tmp_path / "job"
    names = [_trial(job, f"attempt{index:03d}__c1__trial", **trial) for index, trial in enumerate(trials, 1)]
    rewards = [{"_trial_root_name": name, "_trial_name": name, "entry_id": "c1"} for name in names]
    plan = {
        "mode": "native",
        "declared": DECLARED,
        "harness": {"kind": "claude-code-init", "plugin": PLUGIN},
        "hook_ids": {"hooks/hooks.json": HOOK_IDS},
    }
    census = collector._attach_load_census(rewards, job, plan, agent="claude-code")
    signals = {
        "n_trials": len(trials),
        "activation_coverage": activation or _activation(exercised=[]),
        "hook_census": collector._arm_hook_census(job),
    }
    plugin_load = pn.plugin_load_provenance(
        "native", dict(pn.resolve_plugin_load("native", ["claude-code"], env_mode="docker"))
    )
    agent = {"plugin_signals_summary": {"with_skill": signals}, "plugin_load_census": census}
    engine = {"run_config": {"plugin_load": plugin_load}, "agents": {"claude-code": agent}}
    provenance = cli_module._plugin_provenance_with_runtime_evidence(package, engine, None)
    result = _agent_eval("claude-code", agent, engine, {**provenance, "_run_dir": str(tmp_path / "results" / "run")})
    tier1 = ValidationResult(validator_name="PLUGIN_SCHEMA", validator_description="Plugin schema")
    tier1.metadata["plugin"] = {"name": PLUGIN, "manifest_type": "claude", "plugin_mode": "contained"}
    return provenance, [tier1, result]


def _agent_eval(
    agent: str, agent_block: dict[str, Any], engine: dict[str, Any], provenance: dict[str, Any]
) -> ValidationResult:
    """The canonical Tier 3 result for the run, built from a run directory like a real run's.

    The with-plugin arm's ``summary.json`` carries the plugin signals (with the
    hook census); ``run_config.json`` carries the load plan.
    """
    run_dir = Path(provenance["_run_dir"])
    for variant, score in (("with-skill", 0.8), ("without-skill", 0.5)):
        data: dict[str, Any] = {
            "scores": dict.fromkeys(DEFAULT_METRICS, score),
            "metrics": list(DEFAULT_METRICS),
            "num_trials": 1,
            "execution_status": "succeeded",
            "execution_errors": [],
            "expected_attempts": 1,
            "scored_attempts": 1,
        }
        if variant == "with-skill":
            data["plugin_signals_summary"] = agent_block["plugin_signals_summary"]["with_skill"]
        _write(run_dir / agent / variant / "summary.json", json.dumps(data))
    _write(
        run_dir / agent / "with-skill" / "trials" / "c1__attempt1" / "reward.json",
        json.dumps({"entry_id": "c1", "overall": 0.8}),
    )
    run_config = {"eval_target": {"kind": "plugin"}, "agents": {agent: {"model": "test-model"}}, **engine["run_config"]}
    _write(run_dir / "run_config.json", json.dumps(run_config))
    clean = {key: value for key, value in provenance.items() if key != "_run_dir"}
    result = agent_eval_result_from_directory(
        run_dir.parent / PLUGIN, run_dir, engine_result=engine, plugin_provenance=clean, use_llm_judge=False
    )
    assert result is not None
    return result


def _rows(provenance: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in provenance["component_coverage"]["components"]}


def _cli(results: list[ValidationResult]) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=400, color_system=None, emoji=False)
    CLIReporter(console=console)._render_all_results(results, console)
    return buffer.getvalue()


def _all_reports(results: list[ValidationResult]) -> dict[str, str]:
    return {
        "cli": _cli(results),
        "markdown": MarkdownReporter(include_timestamp=False).render_all(results),
        "html": HTMLReporter(include_timestamp=False).render_all(results),
        "benchmark": BenchmarkReporter(include_timestamp=False, content_type="plugin", skill_name=PLUGIN).render_all(
            results
        ),
        "sarif": SARIFReporter(include_timestamp=False).render_all(results),
    }


# ---------------------------------------------------------------------------
# L21: a staged component the harness did not load is not shown as covered
# ---------------------------------------------------------------------------


def test_plugin_missing_from_every_init_marks_its_components_not_loaded(tmp_path: Path) -> None:
    provenance, results = _run(tmp_path, [{"plugin_in_init": False}] * 3)

    rows = _rows(provenance)
    for name in ("notes", "tools", "hooks/hooks.json", "reviewer", "check"):
        assert rows[name]["state"] == "not_loaded", name
        assert "claude-code did not load plugin release-kit" in rows[name]["reason"]
    assert provenance["component_coverage"]["not_evaluated"] == 5
    assert provenance["partial"] is True

    reports = _all_reports(results)
    assert "Component coverage: 0 components not staged, 5 not loaded (of 5; 0 staged" in reports["cli"]
    assert "- skill notes (staged, not loaded): " in reports["cli"]
    assert "**0 components not staged, 5 not loaded** of 5 component(s); 0 staged." in reports["markdown"]
    assert "| skill | notes | Not loaded |" in reports["markdown"]
    assert "| notes | skill | Not loaded |" in reports["benchmark"]
    assert "Staged but not loaded (the harness reported it did not load):" in reports["benchmark"]
    coverage = element_text(reports["html"], "tier3-plugin-coverage") or ""
    assert "0 components not staged, 5 not loaded" in coverage and "Not loaded" in coverage
    plugin = json.loads(reports["sarif"])["runs"][0]["properties"]["plugin"]
    assert plugin["componentsNotStaged"] == 0
    assert plugin["componentsNotLoaded"] == 5


@pytest.mark.parametrize("status", ["failed", "pending"])
def test_mcp_server_that_failed_or_is_pending_is_not_loaded(tmp_path: Path, status: str) -> None:
    provenance, results = _run(tmp_path, [{"mcp_status": status}] * 2)

    rows = _rows(provenance)
    assert rows["tools"]["state"] == "not_loaded"
    assert f"plugin:release-kit:tools with status {status}" in rows["tools"]["reason"]
    assert rows["notes"]["state"] == "loaded"
    view = tier3_plugin_view(results[1].metadata["agent_eval"])
    assert view is not None
    assert view["coverage"]["headline"] == "0 components not staged, 1 not loaded"
    assert [row["name"] for row in view["coverage"]["not_loaded_rows"]] == ["tools"]
    assert view["coverage"]["all_exercised"] is False


def test_partial_load_reason_says_how_many_trials_loaded_the_plugin(tmp_path: Path) -> None:
    trials = [{"plugin_in_init": index >= 2} for index in range(5)]
    provenance, _results = _run(tmp_path, trials)

    reason = _rows(provenance)["notes"]["reason"]
    assert "did not load plugin release-kit (not listed in its init event) in 2 of 5 trial(s)" in reason
    assert "loaded (harness evidence) in 3 of 5" in reason
    assert _rows(provenance)["notes"]["state"] == "not_loaded"


def test_an_exercised_hook_row_is_reported_as_observed(tmp_path: Path) -> None:
    started = {HOOK_IDS[0]: 0, HOOK_IDS[1]: 0}
    provenance, results = _run(
        tmp_path, [{"hook_exits": started}] * 2, activation=_activation(exercised=["skill:notes", "mcp:tools"])
    )

    assert _rows(provenance)["hooks/hooks.json"]["state"] == "exercised"
    view = tier3_plugin_view(results[1].metadata["agent_eval"])
    assert view is not None
    hook = next(row for row in view["coverage"]["rows"] if row["type"] == "hook")
    assert hook["observed"] == "exercised"
    html = element_text(_all_reports(results)["html"], "tier3-plugin-coverage") or ""
    assert "hook hooks/hooks.json hooks/hooks.json packaged Exercised exercised" in html


def test_a_hooks_file_with_a_handler_that_never_started_is_not_exercised(tmp_path: Path) -> None:
    # p01 / n06: the Stop hook's command was not found (exit 127) in every trial.
    exits = {HOOK_IDS[0]: 0, HOOK_IDS[1]: 127}
    provenance, results = _run(
        tmp_path, [{"hook_exits": exits}] * 3, activation=_activation(exercised=["skill:notes", "mcp:tools"])
    )

    row = _rows(provenance)["hooks/hooks.json"]
    assert row["state"] == "staged"
    assert "runtime evidence" not in row["reason"]
    assert "- hook hooks/hooks.json (staged, 1 of 2 hook handlers started)" in _all_reports(results)["cli"]


# ---------------------------------------------------------------------------
# M14 (census-merge half): one agent's listing never hides another's failure
# ---------------------------------------------------------------------------


def _two_agent_coverage() -> dict[str, Any]:
    from skillevaluator.plugin_components import summarize_coverage

    return summarize_coverage(
        [
            {
                "type": "mcp",
                "name": "cfdocs",
                "origin": "packaged",
                "path": ".mcp.json",
                "state": "staged",
                "reason": "runnable MCP server staged for the with-plugin arm only",
            },
            {
                "type": "skill",
                "name": "repo-docs",
                "origin": "packaged",
                "path": "skills/repo-docs",
                "state": "staged",
                "reason": "bundled skill staged as a plugin member skill",
            },
        ]
    )


CLAUDE_FAILED = {
    "agent": "claude-code",
    "mode": "native",
    "trials": 5,
    "loaded": [],
    "listed": [],
    "staged": [],
    "not_loaded": [
        {
            "type": "mcp",
            "name": "cfdocs",
            "reason": "claude-code init reported MCP server plugin:remote-docs:cfdocs with status failed",
        },
        {
            "type": "skill",
            "name": "repo-docs",
            "reason": "claude-code did not load plugin remote-docs (not listed in its init event)",
        },
    ],
}
CODEX_LISTED = {
    "agent": "codex",
    "mode": "native",
    "trials": 5,
    "loaded": [],
    "staged": [],
    "not_loaded": [],
    "listed": [
        {"type": "mcp", "name": "cfdocs", "evidence": "config.toml listing"},
        {"type": "skill", "name": "repo-docs", "evidence": "skills listing"},
    ],
}


@pytest.mark.parametrize("order", [("claude-code", "codex"), ("codex", "claude-code")])
def test_load_census_merge_keeps_one_agents_failure_visible(order: tuple[str, str]) -> None:
    decisions = dict(pn.resolve_plugin_load("native", ["claude-code", "codex"], env_mode="docker"))
    plugin_load = pn.plugin_load_provenance("native", decisions)
    summaries = {"claude-code": CLAUDE_FAILED, "codex": CODEX_LISTED}

    merged = pn.apply_load_census(_two_agent_coverage(), {agent: summaries[agent] for agent in order}, plugin_load)

    for row in merged["components"]:
        assert row["state"] == "not_loaded", row["name"]
        assert "not loaded natively by claude-code" in row["reason"]
        assert "staged natively for codex" in row["reason"]
    assert merged["not_evaluated"] == 2


def test_another_arms_use_still_promotes_a_component_one_agent_did_not_load(tmp_path: Path) -> None:
    """Claude Code native did not load the plugin; the Codex wrapper arm used the skill and the server."""
    package = prepare_plugin_eval_package(
        _plugin(tmp_path), stage_root=tmp_path / "stage", plugin_load="native", agents="claude-code", env_mode="docker"
    )
    claude = pn.summarize_censuses(
        "claude-code",
        "native",
        [
            pn.apply_claude_init_evidence(
                {"agent": "claude-code", "mode": "native", "loaded": [], "listed": [], "not_loaded": []},
                DECLARED,
                {"plugins": []},
                plugin=PLUGIN,
            )
        ],
    )
    decisions = {
        "claude-code": dict(pn.resolve_plugin_load("native", ["claude-code"], env_mode="docker"))["claude-code"],
        "codex": pn.wrapper_decision("codex", "wrapper requested for this agent"),
    }
    codex_signals = {"n_trials": 2, "activation_coverage": _activation(exercised=["skill:notes", "mcp:tools"])}
    engine = {
        "run_config": {"plugin_load": pn.plugin_load_provenance("native", decisions)},
        "agents": {
            "claude-code": {"plugin_load_census": claude},
            "codex": {"plugin_signals_summary": {"with_skill": codex_signals}},
        },
    }

    rows = _rows(cli_module._plugin_provenance_with_runtime_evidence(package, engine, None))

    for name in ("notes", "tools"):
        assert rows[name]["state"] == "exercised", name
        assert "not loaded natively by claude-code" in rows[name]["reason"]
        assert "activated in the codex with-plugin arm" in rows[name]["reason"]
    assert rows["reviewer"]["state"] == "not_loaded"


# ---------------------------------------------------------------------------
# L33: components that cannot be staged are not "declared, unverified"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("agent", "plugin_load"), [("claude-code", "wrapper"), ("codex", "wrapper"), ("codex", "native")]
)
def test_activation_summary_leaves_out_components_that_cannot_be_staged(
    tmp_path: Path, agent: str, plugin_load: str
) -> None:
    package = prepare_plugin_eval_package(
        _plugin(tmp_path), stage_root=tmp_path / "stage", plugin_load=plugin_load, agents=agent, env_mode="docker"
    )
    signals = {"n_trials": 3, "activation_coverage": _activation(exercised=["skill:notes", "mcp:tools"])}
    decisions = dict(pn.resolve_plugin_load(plugin_load, [agent], env_mode="docker"))
    engine = {
        "run_config": {"plugin_load": pn.plugin_load_provenance(plugin_load, decisions)},
        "agents": {agent: {"plugin_signals_summary": {"with_skill": signals}}},
    }
    provenance = cli_module._plugin_provenance_with_runtime_evidence(package, engine, None)
    states = {row["name"]: row["state"] for row in provenance["component_coverage"]["components"]}
    assert states["reviewer"] == states["check"] == "unsupported"
    result = _agent_eval(
        agent, engine["agents"][agent], engine, {**provenance, "_run_dir": str(tmp_path / "results" / "run")}
    )

    cli = _cli([result])
    html = HTMLReporter(include_timestamp=False).render_all([result])

    summary = (
        "2 of 2 declared components were exercised in at least one plugin trial "
        "(2 more declared components could not be staged)"
    )
    assert f"Observed activation (advisory): {summary}" in cli
    assert "4 declared components" not in cli
    assert "- activation 2/2 exercised" in cli
    assert summary in (element_text(html, "tier3-plugin-coverage") or "")
    assert "declared 2, exercised 2, unverified 0, unavailable 0" in (element_text(html, "tier3-plugin-signals") or "")
    view = tier3_plugin_view(result.metadata["agent_eval"])
    assert view is not None
    observed = {row["name"]: row["observed"] for row in view["coverage"]["rows"]}
    assert observed["reviewer"] == observed["check"] == "not staged"
