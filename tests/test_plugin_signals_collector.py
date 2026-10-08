# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin signals wiring: collector attachment, runner context, and dataset validation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.evaluation.tier3_report import _attach_plugin_report_fields
from skillevaluator.tier3.eval_core.plugin_signals import build_plugin_signals_context
from skillevaluator.tier3.evals_spec import validate_tier3_source
from skillevaluator.tier3.harbor import runner
from skillevaluator.tier3.harbor.adapter import _write_entry_json
from skillevaluator.tier3.harbor.collector import collect_harbor_results
from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package

AGENT = "claude-code"
REWARD = {
    "security": 0.9,
    "skill_execution": 0.8,
    "skill_efficiency": 0.7,
    "accuracy": 0.6,
    "goal_accuracy": 0.5,
    "behavior_check": 0.4,
}
CASE = {
    "id": "case-1",
    "prompt": "Triage the open issue",
    "expected_tools": ["Skill:alpha", "mcp__github__*"],
    "tool_arguments": [{"tool": "mcp__github__get_issue", "required": ["number"]}],
    "expected_order": [["Skill:alpha", "mcp:github"]],
    "conflict_probes": [{"id": "no-beta", "must_use": "Skill:alpha", "must_not_use": "Skill:beta"}],
}


def _agent_step(step_id: int, call_id: str, function: str, arguments: dict[str, Any], content: str) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "source": "agent",
        "message": "",
        "tool_calls": [{"tool_call_id": call_id, "function_name": function, "arguments": arguments}],
        "observation": {"results": [{"source_call_id": call_id, "content": content}]},
    }


def _trajectory(*extra_steps: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "ATIF-v1.2",
        "steps": [
            {"step_id": 1, "source": "user", "message": "Triage the open issue"},
            {
                "step_id": 2,
                "source": "agent",
                "message": "",
                "tool_calls": [{"tool_call_id": "c1", "function_name": "Skill", "arguments": {"skill": "alpha"}}],
                "observation": {"results": [{"source_call_id": "c1", "content": "Launching skill: alpha"}]},
            },
            {
                "step_id": 3,
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {"tool_call_id": "c2", "function_name": "mcp__github__get_issue", "arguments": {"number": 7}}
                ],
                "observation": {"results": [{"source_call_id": "c2", "content": "Issue 7: crash on start"}]},
            },
            *extra_steps,
            {"step_id": 9, "source": "agent", "message": "done", "tool_calls": [], "observation": {"results": []}},
        ],
    }


def _write_job(jobs_dir: Path, variant: str, *, trajectory: dict[str, Any] | None) -> str:
    job_dir = jobs_dir / f"demo-{AGENT}-{variant}"
    trial_name = "case-1__AbCd123"
    trial_dir = job_dir / trial_name
    (trial_dir / "verifier").mkdir(parents=True)
    (trial_dir / "verifier" / "reward.json").write_text(json.dumps(REWARD), encoding="utf-8")
    if trajectory is not None:
        (trial_dir / "agent").mkdir()
        (trial_dir / "agent" / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": 1,
                "stats": {
                    "n_trials": 1,
                    "n_errors": 0,
                    "evals": {
                        "agent__model___harbor-tasks": {
                            "n_trials": 1,
                            "n_errors": 0,
                            "reward_stats": {"reward": {"0.65": [trial_name]}},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return trial_name


def _jobs(tmp_path: Path, *, with_trajectory: bool = True, trajectory: dict[str, Any] | None = None) -> Path:
    jobs_dir = tmp_path / "jobs"
    _write_job(jobs_dir, "with", trajectory=(trajectory or _trajectory()) if with_trajectory else None)
    _write_job(jobs_dir, "without", trajectory=_trajectory())
    _write_job(jobs_dir, "sumofparts", trajectory=_trajectory())
    return jobs_dir


def _collect(tmp_path: Path, jobs_dir: Path, output: str, **kwargs: Any) -> dict[str, Any]:
    return collect_harbor_results(
        skill_name="demo",
        agents=[AGENT],
        output_dir=tmp_path / output,
        jobs_dir=jobs_dir,
        sum_of_parts_arm=True,
        expected_cases=1,
        expected_case_ids=["case-1"],
        expected_trials=1,
        **kwargs,
    )


def _context(**overrides: Any):
    values: dict[str, Any] = {
        "member_skills": ["alpha", "beta"],
        "mcp_servers": ["github"],
        "wrapper_skills": ["demo"],
        "entries": [CASE],
    }
    values.update(overrides)
    return build_plugin_signals_context(**values)


def _trial_reward(tmp_path: Path, output: str, condition: str) -> dict[str, Any]:
    path = tmp_path / output / AGENT / condition / "trials" / "case-1__AbCd123" / "reward.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(tmp_path: Path, output: str, condition: str) -> dict[str, Any]:
    return json.loads((tmp_path / output / AGENT / condition / "summary.json").read_text(encoding="utf-8"))


def test_plugin_run_attaches_per_trial_signals_and_per_arm_summaries(tmp_path: Path) -> None:
    jobs_dir = _jobs(tmp_path)

    results = _collect(tmp_path, jobs_dir, "plugin", plugin_signals=_context())

    signals = _trial_reward(tmp_path, "plugin", "with-skill")["plugin_signals"]
    assert [(a["type"], a["name"], a["tool"], a["step_index"], a["succeeded"]) for a in signals["activations"]] == [
        ("skill", "alpha", "Skill", 1, True),
        ("mcp", "github", "mcp__github__get_issue", 2, True),
    ]
    assert signals["tool_selection"]["recall"] == 1.0
    assert signals["arguments"] == {"checked": 1, "passed": 1, "failures": [], "status": "scored"}
    assert signals["order"]["satisfied"] == 1
    assert signals["conflict"]["passed"] == 1
    assert signals["handoff"]["status"] == "not_applicable"
    assert signals["activation_coverage"] == {
        "declared": ["skill:alpha", "skill:beta", "mcp:github"],
        "exercised": ["skill:alpha", "mcp:github"],
        "unverified": ["skill:beta"],
        "unavailable": [],
    }

    # The member-skills arm has no plugin MCP wiring, so MCP is not declared there.
    parts = _trial_reward(tmp_path, "plugin", "sum-of-parts")["plugin_signals"]
    assert parts["activation_coverage"]["declared"] == ["skill:alpha", "skill:beta"]
    # The effectiveness baseline carries no plugin components.
    assert "plugin_signals" not in _trial_reward(tmp_path, "plugin", "without-skill")

    agent = results["agents"][AGENT]
    assert set(agent["plugin_signals_summary"]) == {"with_skill", "sum_of_parts"}
    with_summary = agent["plugin_signals_summary"]["with_skill"]
    assert with_summary["n_trials"] == 1
    assert with_summary["mcp_calls"]["success_rate"] == 1.0
    assert with_summary["activation_coverage"]["unverified"] == ["skill:beta"]
    assert _summary(tmp_path, "plugin", "with-skill")["plugin_signals_summary"] == with_summary
    assert "plugin_signals_summary" in _summary(tmp_path, "plugin", "sum-of-parts")
    assert "plugin_signals_summary" not in _summary(tmp_path, "plugin", "without-skill")


def test_rewards_and_report_payload_never_carry_argument_values_or_non_name_components(tmp_path: Path) -> None:
    url = "https://deploy:S3cr3tP4ss@gitlab.example.com/x.git"
    trajectory = _trajectory(
        _agent_step(4, "c3", "mcp__github__get_issue", {"number": url}, "Issue 8: flaky"),
        _agent_step(5, "c4", "Skill", {"skill": "-----BEGIN RSA PRIVATE KEY-----MIIEowIBAAKCAQEA"}, "Unknown skill"),
    )
    case = {**CASE, "tool_arguments": [{"tool": "mcp__github__get_issue", "equals": {"number": 7}}]}

    results = _collect(
        tmp_path, _jobs(tmp_path, trajectory=trajectory), "plugin", plugin_signals=_context(entries=[case])
    )
    reward = _trial_reward(tmp_path, "plugin", "with-skill")
    raw_agent = {"plugin_signals_summary": results["agents"][AGENT]["plugin_signals_summary"], "rewards": [reward]}
    payload: dict[str, Any] = {"agents": {AGENT: {}}, "best_agent": AGENT}
    _attach_plugin_report_fields(payload, {AGENT: raw_agent})

    (top,) = payload["plugin_signals_summary"]["with_skill"]["arguments"]["top_failures"]
    assert top["detail"] == f"expected 7, got string(len={len(url)})"
    assert "Skill:<non-name>" in reward["plugin_signals"]["routing"]["called"]
    persisted = [json.dumps(reward), json.dumps(payload), json.dumps(_summary(tmp_path, "plugin", "with-skill"))]
    for text in persisted:
        assert "S3cr3tP4ss" not in text
        assert "MIIEowIBAAKCAQEA" not in text


def test_plugin_signals_never_change_scores_or_pass_results(tmp_path: Path) -> None:
    jobs_dir = _jobs(tmp_path)

    plain = _collect(tmp_path, jobs_dir, "plain")
    plugin = _collect(tmp_path, jobs_dir, "plugin", plugin_signals=_context())

    for key in ("with_skill", "without_skill", "sum_of_parts", "lift", "integration_lift", "pass_at_k", "conditions"):
        assert plugin["agents"][AGENT][key] == plain["agents"][AGENT][key], key
    plain_reward = _trial_reward(tmp_path, "plain", "with-skill")
    plugin_reward = _trial_reward(tmp_path, "plugin", "with-skill")
    plugin_reward.pop("plugin_signals")
    assert plugin_reward == plain_reward


def test_skill_run_attaches_no_plugin_signals(tmp_path: Path) -> None:
    jobs_dir = _jobs(tmp_path)

    results = _collect(tmp_path, jobs_dir, "skill")

    assert "plugin_signals_summary" not in results["agents"][AGENT]
    for condition in ("with-skill", "without-skill", "sum-of-parts"):
        assert "plugin_signals" not in _trial_reward(tmp_path, "skill", condition)
        assert "plugin_signals_summary" not in _summary(tmp_path, "skill", condition)


def test_integration_baseline_with_members_gets_signals(tmp_path: Path) -> None:
    jobs_dir = _jobs(tmp_path)

    results = _collect(tmp_path, jobs_dir, "plugin", plugin_signals=_context(baseline_has_members=True))

    assert set(results["agents"][AGENT]["plugin_signals_summary"]) == {"with_skill", "without_skill", "sum_of_parts"}
    baseline = _trial_reward(tmp_path, "plugin", "without-skill")["plugin_signals"]
    assert baseline["activation_coverage"]["declared"] == ["skill:alpha", "skill:beta"]


def test_each_trial_hook_census_is_read_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The scored-trial signals and the with-plugin every-trial summary share one read of a trial's census."""
    from skillevaluator.tier3.harbor import collector

    reads: Counter[str] = Counter()
    read_hook_census = collector.read_hook_census

    def counted(trial_root: Path) -> dict[str, Any]:
        reads[f"{trial_root.parent.name}/{trial_root.name}"] += 1
        return read_hook_census(trial_root)

    monkeypatch.setattr(collector, "read_hook_census", counted)

    results = _collect(tmp_path, _jobs(tmp_path), "plugin", plugin_signals=_context())

    assert results["agents"][AGENT]["plugin_signals_summary"]["with_skill"]["hook_census"]["n_trials"] == 1
    assert reads == {f"demo-{AGENT}-with/case-1__AbCd123": 1, f"demo-{AGENT}-sumofparts/case-1__AbCd123": 1}


def test_missing_trajectory_is_counted_not_fabricated(tmp_path: Path) -> None:
    jobs_dir = _jobs(tmp_path, with_trajectory=False)

    results = _collect(tmp_path, jobs_dir, "plugin", plugin_signals=_context())

    assert "plugin_signals" not in _trial_reward(tmp_path, "plugin", "with-skill")
    summary = results["agents"][AGENT]["plugin_signals_summary"]["with_skill"]
    assert (summary["n_trials"], summary["n_missing_trajectory"]) == (0, 1)


def test_unknown_case_still_reports_activations(tmp_path: Path) -> None:
    jobs_dir = _jobs(tmp_path)

    _collect(tmp_path, jobs_dir, "plugin", plugin_signals=_context(entries=[]))

    signals = _trial_reward(tmp_path, "plugin", "with-skill")["plugin_signals"]
    assert len(signals["activations"]) == 2
    assert signals["tool_selection"]["status"] == "not_applicable"


# ---------------------------------------------------------------------------
# Runner context and dataset plumbing
# ---------------------------------------------------------------------------


def test_runner_context_reads_runnable_mcp_and_staged_case_fields(tmp_path: Path) -> None:
    package = tmp_path / "demo-plugin-eval"
    env_dir = package / "evals" / "environment"
    env_dir.mkdir(parents=True)
    (env_dir / "plugin_mcp_servers.toml").write_text(
        '[[mcp_servers]]\nname = "github"\ncommand = "github-mcp"\n', encoding="utf-8"
    )
    run_dir = tmp_path / "run"
    task_tests = run_dir / "_harbor-tasks" / AGENT / "with" / "case-1" / "tests"
    task_tests.mkdir(parents=True)
    _write_entry_json(task_tests.parent, CASE, True, evaluated_skill="demo-plugin-eval")

    context = runner._plugin_signals_context(
        skill_path=package,
        evaluator_skill_path=package,
        workspace_skills=[tmp_path / "skills" / "alpha", tmp_path / "skills" / "beta"],
        run_dir=run_dir,
        baseline_has_members=False,
    )

    assert context.member_skills == ("alpha", "beta")
    assert context.mcp_servers == ("github",)
    assert context.wrapper_skills == ("demo-plugin-eval", "demo")
    assert context.case_spec("case-1")["expected_tools"] == ["Skill:alpha", "mcp__github__*"]


def test_staged_task_entry_preserves_plugin_signal_fields(tmp_path: Path) -> None:
    task_dir = tmp_path / "case-1"
    _write_entry_json(task_dir, CASE, True, evaluated_skill="demo")

    staged = json.loads((task_dir / "tests" / "entry.json").read_text(encoding="utf-8"))

    for field in ("expected_tools", "tool_arguments", "expected_order", "conflict_probes"):
        assert staged[field] == CASE[field]


def _skill_with_dataset(tmp_path: Path, entry: dict[str, Any]) -> Path:
    skill = tmp_path / "source-skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: source-skill\ndescription: test\n---\n", encoding="utf-8")
    (skill / "evals" / "evals.json").write_text(
        json.dumps({"skill_name": "source-skill", "evals": [{"expected_output": "done", **entry}]}),
        encoding="utf-8",
    )
    return skill


def test_dataset_validation_accepts_plugin_signal_fields(tmp_path: Path) -> None:
    _source, checks = validate_tier3_source(_skill_with_dataset(tmp_path, CASE))
    assert not [check for check in checks if check.status in {"missing", "error"}]


def test_dataset_validation_accepts_huge_numeric_bounds_instead_of_raising(tmp_path: Path) -> None:
    schema = {"properties": {"number": {"minimum": -(10**400), "maximum": 10**400}}}
    entry = {**CASE, "tool_arguments": [{"tool": "mcp__github__get_issue", "schema": schema}]}

    _source, checks = validate_tier3_source(_skill_with_dataset(tmp_path, entry))

    assert not [check for check in checks if check.status in {"missing", "error"}]


def test_dataset_validation_rejects_malformed_plugin_signal_fields(tmp_path: Path) -> None:
    bad = {**CASE, "expected_tools": "Skill:alpha", "handoffs": [{"producer": "a", "consumer": "b"}]}

    _source, checks = validate_tier3_source(_skill_with_dataset(tmp_path, bad))

    errors = [check.message for check in checks if check.status == "error"]
    assert any("expected_tools: must be a list" in message for message in errors)
    assert any("value or an artifact" in message for message in errors)


def _plugin_with_dataset(tmp_path: Path, entry: dict[str, Any]) -> Path:
    plugin = tmp_path / "plugin"
    manifest = plugin / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"name": "public-plugin", "skills": "./skills"}), encoding="utf-8")
    skill = plugin / "skills" / "alpha"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Public test skill\n---\n# alpha\n", encoding="utf-8"
    )
    (plugin / "evals").mkdir()
    (plugin / "evals" / "evals.json").write_text(json.dumps([entry]), encoding="utf-8")
    return plugin


def test_plugin_package_accepts_valid_plugin_signal_fields(tmp_path: Path) -> None:
    package = prepare_plugin_eval_package(_plugin_with_dataset(tmp_path, CASE), stage_root=tmp_path / "stage")
    assert package.dataset_case_count == 1


def test_plugin_package_rejects_malformed_plugin_signal_fields_before_running(tmp_path: Path) -> None:
    bad = {**CASE, "tool_arguments": [{"tool": "mcp__github__x", "schema": {"items": {}}}]}
    with pytest.raises(ValueError, match=r"Invalid plugin signal fields.*unsupported JSON-Schema keyword"):
        prepare_plugin_eval_package(_plugin_with_dataset(tmp_path, bad), stage_root=tmp_path / "stage")


def test_collector_credits_a_renamed_opencode_agent_to_the_declared_agent(tmp_path: Path) -> None:
    # OpenCode stages a plugin agent named "build" as "<plugin>-build"; the runner reads
    # that alias from the package and the collector maps the call back.
    package = tmp_path / "demo-plugin-eval"
    env_dir = package / "evals" / "environment"
    env_dir.mkdir(parents=True)
    (env_dir / "plugin_runtime_components.json").write_text(
        json.dumps({"subagents": ["build"], "commands": [], "subagent_aliases": {"demo-kit-build": "build"}}),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    context = runner._plugin_signals_context(
        skill_path=package,
        evaluator_skill_path=package,
        workspace_skills=[tmp_path / "skills" / "alpha"],
        run_dir=run_dir,
        baseline_has_members=False,
    )
    assert context.subagent_aliases == {"demo-kit-build": "build"}
    task = _agent_step(4, "c4", "task", {"subagent_type": "demo-kit-build", "prompt": "build it"}, "Built.")
    jobs_dir = _jobs(tmp_path, trajectory=_trajectory(task))

    _collect(tmp_path, jobs_dir, "plugin", plugin_signals=context)

    coverage = _trial_reward(tmp_path, "plugin", "with-skill")["plugin_signals"]["activation_coverage"]
    assert "subagent:build" in coverage["exercised"]


def _plugin_file_mcp_plugin(tmp_path: Path, entry: dict[str, Any]) -> Path:
    plugin = tmp_path / "issue-kit"
    manifest = plugin / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"name": "issue-kit"}), encoding="utf-8")
    server = {"command": "python3", "args": ["${CLAUDE_PLUGIN_ROOT}/server/tracker.py"]}
    (plugin / ".mcp.json").write_text(json.dumps({"mcpServers": {"tracker": server}}), encoding="utf-8")
    (plugin / "server").mkdir()
    (plugin / "server" / "tracker.py").write_text("print('tracker')\n", encoding="utf-8")
    (plugin / "evals").mkdir()
    (plugin / "evals" / "evals.json").write_text(json.dumps([entry]), encoding="utf-8")
    return plugin


def test_collector_credits_a_plugin_file_mcp_server_only_in_the_claude_code_arm(tmp_path: Path) -> None:
    # Claude Code's native plugin copy starts a ${CLAUDE_PLUGIN_ROOT} server that the
    # wrapper's runnable MCP file never lists, and Codex cannot stage it at all.
    from skillevaluator.tier3.harbor.native_staging import build_native_task_staging
    from skillevaluator.tier3.plugin_native import adapter_for
    from skillevaluator.tier3.plugin_runtime import apply_runtime_coverage

    entry = {
        "id": "case-1",
        "prompt": "Status of ISSUE-7?",
        "expected_output": "Open.",
        "expected_tools": ["MCP:tracker"],
    }
    package = prepare_plugin_eval_package(
        _plugin_file_mcp_plugin(tmp_path, entry),
        stage_root=tmp_path / "stage",
        plugin_load="native",
        agents=[AGENT, "codex"],
        env_mode="docker",
    )
    stagings = {
        agent: build_native_task_staging(agent, adapter_for(agent), package.native_source) for agent in (AGENT, "codex")
    }
    run_dir = tmp_path / "run"
    task_tests = run_dir / "_harbor-tasks" / AGENT / "with" / "case-1" / "tests"
    task_tests.mkdir(parents=True)
    _write_entry_json(task_tests.parent, entry, True, evaluated_skill=package.package_path.name)
    context = runner._plugin_signals_context(
        skill_path=package.package_path,
        evaluator_skill_path=package.package_path,
        workspace_skills=[],
        run_dir=run_dir,
        baseline_has_members=False,
        agent_mcp_servers=runner._native_plugin_file_mcp_servers(stagings),
    )
    assert context.declared_for("with_skill", AGENT)["mcp"] == ["tracker"]
    assert context.declared_for("with_skill", "codex")["mcp"] == []
    # Every agent's member-skills arm stages the same members only, so each one skips the ref.
    assert [context.declared_for("sum_of_parts", agent).get("unstaged") for agent in (AGENT, "codex")] == [["mcp"]] * 2
    call = _agent_step(2, "c1", "mcp__plugin_issue-kit_tracker__get_issue", {"id": "ISSUE-7"}, "ISSUE-7: open")
    trajectory = {"schema_version": "ATIF-v1.2", "steps": [{"step_id": 1, "source": "user", "message": "x"}, call]}

    results = _collect(tmp_path, _jobs(tmp_path, trajectory=trajectory), "plugin", plugin_signals=context)

    signals = _trial_reward(tmp_path, "plugin", "with-skill")["plugin_signals"]
    assert list(signals["mcp_calls"]["by_server"]) == ["tracker"]
    assert signals["tool_selection"]["recall"] == 1.0
    assert signals["activation_coverage"]["exercised"] == ["mcp:tracker"]
    # The member-skills arm never carries the server, so the ref is skipped there, not missed.
    assert _trial_reward(tmp_path, "plugin", "sum-of-parts")["plugin_signals"]["tool_selection"]["skipped"] == [
        "MCP:tracker"
    ]
    provenance = package.provenance()
    assert apply_runtime_coverage(provenance, results) == 1
    (row,) = [row for row in provenance["component_coverage"]["components"] if row["type"] == "mcp"]
    assert (row["name"], row["state"]) == ("tracker", "exercised")


def test_run_hands_the_collector_the_per_agent_plugin_signals_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Native Claude Code and Codex arms, with Harbor and the collector mocked: the
    # context the run builds must keep what only the Claude Code arm stages.
    from skillevaluator.provider_config import ProviderConfig
    from skillevaluator.tier3.harbor import runtime_preflight

    entry = {"id": "case-1", "prompt": "Status of ISSUE-7?", "expected_output": "Open.", "expected_tools": []}
    agents = [AGENT, "codex"]
    package = prepare_plugin_eval_package(
        _plugin_file_mcp_plugin(tmp_path, entry),
        stage_root=tmp_path / "stage",
        plugin_load="native",
        agents=agents,
        env_mode="docker",
    )
    provider = ProviderConfig(
        provider="nv_build",
        model="test-model",
        api_key="provider-key",
        base_url="https://provider.example/v1",
        litellm_model="openai/test-model",
    )
    collected: dict[str, Any] = {}

    def collect(**kwargs: Any) -> dict[str, Any]:
        collected.update(kwargs)
        return {"execution_status": "succeeded", "execution_errors": [], "metrics": [], "agents": {}}

    monkeypatch.setattr(runner, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setattr(runner, "_run_agent_pair", lambda **_kwargs: [])
    monkeypatch.setattr(runner, "collect_harbor_results", collect)
    monkeypatch.setattr(runner, "render_agent_eval_html_report", lambda *_args, **_kwargs: tmp_path / "report.html")
    monkeypatch.setattr(
        runtime_preflight,
        "probe_model",
        lambda selected: runtime_preflight.ModelProbeResult(True, selected.provider, selected.model, "available"),
    )

    result = runner.run_harbor_eval(
        package.package_path,
        agents,
        output_dir=tmp_path / "results",
        env_mode="docker",
        agent_runtime_preflight=False,
        eval_target_kind="plugin",
        plugin_load="native",
        native_plugin_source=package.native_source,
    )

    assert "error" not in result, result
    context = collected["plugin_signals"]
    assert context.agent_mcp_servers == {AGENT: ("tracker",)}
    assert context.declared_for("with_skill", AGENT)["mcp"] == ["tracker"]
    assert context.declared_for("with_skill", "codex")["mcp"] == []

