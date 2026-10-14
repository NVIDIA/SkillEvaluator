# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The loose-parts arm wired end to end: ``--lift-mode ladder`` from the CLI to the lift ladder block.

Every seam a four-arm run crosses: the CLI option and its fallback table, the
options pass-through, the runner (which agents run the arm, the four task
directories, the fourth Harbor job, the run config), the task emitter, the
content hashes of the emitted tasks, the collector (the fourth arm, the per-trial
load evidence, the dropped cases), the report payload and the run summary. The
staging is the real one, built from the fake ``shipit`` plugin. Harbor, Docker,
and every model are mocked away: nothing here makes a model call.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _f90_lift_fixtures import SKILL, metrics, write_job
from _loose_parts_fixtures import make_plugin, write

from skillevaluator import cli as cli_module
from skillevaluator.evaluation import EvaluationService
from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3.harbor import runner, runtime_preflight
from skillevaluator.tier3.loose_parts import NOT_SUPPORTED_WRAPPER, NOTHING_LOOSE, not_supported_reason
from skillevaluator.tier3.plugin_native import resolve_plugin_load

AGENT = "claude-code"
NONE, SKILLS, LOOSE, FULL = "without_skill", "sum_of_parts", "loose_parts", "with_skill"
CROSS = ["x-1", "x-2", "x-3", "x-4", "x-5", "x-6"]
OTHER = ["o-1", "o-2", "o-3"]
CASES = [*CROSS, *OTHER]
MEMBERS = ["changelog-collect", "version-bump"]


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
def _cross_component_plugin(root: Path) -> Path:
    """``shipit`` with a dataset that has one cross-component case naming both member skills."""
    plugin = make_plugin(root)
    cases = [
        {
            "id": "x-1",
            "prompt": "Collect the changes and bump the version.",
            "expected_output": "Changes and version.",
            "assertions": ["Both done"],
            "cross_component": True,
            "expected_skills": list(MEMBERS),
        },
        {
            "id": "o-1",
            "prompt": "Collect the changes.",
            "expected_output": "Changes.",
            "assertions": ["Listed"],
            "expected_skill": "changelog-collect",
        },
    ]
    write(plugin / "evals" / "evals.json", json.dumps({"skill_name": "shipit", "evals": cases}))
    return plugin


def _package(tmp_path: Path, *, agents: list[str] | None = None, plugin_load: str = "native"):
    from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package

    return prepare_plugin_eval_package(
        _cross_component_plugin(tmp_path / "plugins"),
        stage_root=tmp_path / "stage",
        plugin_load=plugin_load,
        agents=agents or [AGENT],
        env_mode="docker",
    )


def _prepared(tmp_path: Path, *, evidence: str | None = None, loose: dict[str, str | None] | None = None):
    """A prepared-package stand-in: ``loose`` maps an agent to its loose-parts error (default: none)."""
    (tmp_path / "package").mkdir(exist_ok=True)
    (tmp_path / "member").mkdir(exist_ok=True)
    errors = loose or {}
    return SimpleNamespace(
        skipped=False,
        skip_reason=None,
        package_path=tmp_path / "package",
        include_skills=(tmp_path / "member",),
        unresolved_skill_refs=(),
        unresolved_rule_refs=(),
        unresolved_mcp_servers=(),
        native_source="snapshot",
        mcp_probe_targets=(),
        integration_evidence_error=lambda: evidence,
        loose_parts_error=lambda agent="claude-code": (
            errors.get(agent, not_supported_reason(agent)) if agent != AGENT else errors.get(agent)
        ),
        provenance=lambda: {"plugin_name": "plugin", "partial": False},
    )


def _provider() -> ProviderConfig:
    return ProviderConfig(
        provider="nv_build",
        model="test-model",
        api_key="provider-key",
        base_url="https://provider.example/v1",
        litellm_model="openai/test-model",
    )


def _stub_runner_edges(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, captured: dict[str, Any]) -> None:
    """Mock the provider, the host checks, the model probes, Harbor and the collector; keep task staging real."""

    def pair(**kwargs: Any) -> list[str]:
        captured["pair"] = kwargs
        # The task directories exist only during the run: record what each arm staged.
        captured["tasks"] = {
            key: sorted(path.name for path in kwargs[key].iterdir() if path.is_dir())
            for key in ("with_skill", "baseline", "sum_of_parts", "loose_parts")
            if kwargs.get(key) is not None
        }
        loose_dir = kwargs.get("loose_parts")
        if loose_dir is not None:
            task = next(path for path in sorted(loose_dir.iterdir()) if path.is_dir())
            env = task / "environment"
            captured["loose_env"] = sorted(
                path.relative_to(env).as_posix() for path in env.rglob("*") if path.is_file()
            )
            captured["loose_entry"] = json.loads((task / "tests" / "entry.json").read_text(encoding="utf-8"))
        return []

    def collect(**kwargs: Any) -> dict[str, Any]:
        captured["collect"] = kwargs
        return {"execution_status": "succeeded", "execution_errors": [], "metrics": [], "agents": {}}

    monkeypatch.setattr(runner, "resolve_llm_provider", _provider)
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setattr(runner, "_run_agent_pair", pair)
    monkeypatch.setattr(runner, "collect_harbor_results", collect)
    monkeypatch.setattr(runner, "render_agent_eval_html_report", lambda *_args, **_kwargs: tmp_path / "report.html")
    monkeypatch.setattr(
        runtime_preflight,
        "probe_model",
        lambda selected: runtime_preflight.ModelProbeResult(True, selected.provider, selected.model, "available"),
    )


def _run(package, tmp_path: Path, agents: list[str], **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "output_dir": tmp_path / "results",
        "env_mode": "docker",
        "agent_runtime_preflight": False,
        "eval_target_kind": "plugin",
        "plugin_load": "native",
        "native_plugin_source": package.native_source,
        "skill_workspace_mode": "group",
        "include_skills": list(package.include_skills),
        "workspace_skills_baseline": False,
        "sum_of_parts_arm": True,
        "loose_parts_arm": True,
        "lift_mode_requested": "ladder",
        "n_attempts": 1,
    }
    kwargs.update(overrides)
    return runner.run_harbor_eval(package.package_path, agents, **kwargs)


# --------------------------------------------------------------------------- #
# CLI: the option and the fallback table                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("command", [["tier3", "evaluate-plugin", "--help"], ["validate", "--help"]])
def test_ladder_is_a_lift_mode_choice(command: list[str]) -> None:
    result = CliRunner().invoke(cli_module.cli, command)

    assert result.exit_code == 0, result.output
    assert "ladder" in result.output


def test_ladder_falls_back_to_effectiveness_without_cross_component_evidence(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path, evidence="no cross-component case")

    assert cli_module._plugin_lift_mode_for_evidence(prepared, "ladder") == ("effectiveness", "no cross-component case")
    assert cli_module._plugin_integration_error("ladder", False, "no cross-component case") is None


def test_ladder_with_skip_baseline_is_an_error() -> None:
    error = cli_module._plugin_integration_error("ladder", True, None)

    assert error is not None and "requires a baseline" in error and "--skip-baseline" in error


@pytest.mark.parametrize(
    ("agents", "loose", "expected_mode", "expected_reason"),
    [
        ("claude-code", {}, "ladder", None),
        (None, {}, "ladder", None),
        ("claude-code,codex", {}, "ladder", None),
        ("claude-code", {AGENT: NOTHING_LOOSE}, "both", NOTHING_LOOSE),
        ("claude-code", {AGENT: NOT_SUPPORTED_WRAPPER}, "both", NOT_SUPPORTED_WRAPPER),
        (None, {AGENT: NOTHING_LOOSE}, "both", NOTHING_LOOSE),
        ("codex", {}, "both", not_supported_reason("codex")),
        ("claude", {}, "ladder", None),
    ],
)
def test_ladder_fallback_table(
    tmp_path: Path, agents: str | None, loose: dict[str, str], expected_mode: str, expected_reason: str | None
) -> None:
    prepared = _prepared(tmp_path, loose=loose)

    assert cli_module._plugin_ladder_mode(prepared, "ladder", agents) == (expected_mode, expected_reason)


@pytest.mark.parametrize("effective", ["effectiveness", "integration", "both"])
def test_other_lift_modes_never_ask_the_loose_parts_staging(tmp_path: Path, effective: str) -> None:
    prepared = _prepared(tmp_path)
    prepared.loose_parts_error = lambda *_args: pytest.fail("the staging must not be asked outside ladder mode")

    assert cli_module._plugin_ladder_mode(prepared, effective, "claude-code") == (effective, None)


def test_ladder_metadata_records_the_ladder_skip_reason() -> None:
    metadata = cli_module._plugin_lift_metadata("ladder", "both", None, ladder_skip_reason=NOTHING_LOOSE)

    assert metadata == {
        "requested_lift_mode": "ladder",
        "effective_lift_mode": "both",
        "ladder_skip_reason": NOTHING_LOOSE,
    }
    assert "ladder_skip_reason" not in cli_module._plugin_lift_metadata("both", "both", None)


@pytest.mark.parametrize(
    ("effective", "loose_arm", "sum_of_parts_arm"),
    [("ladder", True, True), ("both", False, True), ("effectiveness", False, False), ("integration", False, False)],
)
def test_options_for_each_effective_mode(
    tmp_path: Path, effective: str, loose_arm: bool, sum_of_parts_arm: bool
) -> None:
    options = cli_module._plugin_evaluation_options(
        _prepared(tmp_path),
        plugin_dir=tmp_path,
        lift_mode="ladder",
        effective_lift_mode=effective,
        integration_skip_reason=None,
        ladder_skip_reason=None if effective == "ladder" else "why",
        plugin_load="native",
        results_dir=None,
    )

    assert options.loose_parts_arm is loose_arm
    assert options.sum_of_parts_arm is sum_of_parts_arm
    assert options.workspace_skills_baseline is (effective == "integration")
    assert options.ladder_skip_reason == (None if effective == "ladder" else "why")
    assert options.engine_kwargs()["loose_parts_arm"] is loose_arm


def _invoke_evaluate_plugin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, prepared: Any, arguments: list[str]
) -> tuple[Any, dict[str, Any]]:
    plugin = tmp_path / "plugin"
    plugin.mkdir(exist_ok=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    captured: dict[str, Any] = {}
    monkeypatch.setattr("skillevaluator.tier3.plugin_eval.prepare_plugin_eval_package", lambda *_a, **_k: prepared)
    monkeypatch.setattr(
        EvaluationService,
        "evaluate",
        lambda _self, options, **_kwargs: captured.setdefault("options", options) and {"run_dir": str(run_dir)},
    )
    monkeypatch.setattr(EvaluationService, "failure_reason", staticmethod(lambda _result: None))
    monkeypatch.setattr("skillevaluator.tier3.result_display.render_evaluation_result", lambda *_a, **_k: None)
    monkeypatch.setattr("skillevaluator.evaluation.tier3_report.refresh_plugin_run_report", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "skillevaluator.tier3.plugin_eval.write_plugin_provenance",
        lambda _run_dir, provenance: captured.setdefault("provenance", dict(provenance)),
    )
    result = CliRunner().invoke(
        cli_module.cli, ["tier3", "evaluate-plugin", str(plugin), "--progress", "off", *arguments]
    )
    return result, captured


def test_evaluate_plugin_ladder_requests_the_loose_parts_arm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    arguments = ["-a", AGENT, "--plugin-load", "native", "--lift-mode", "ladder"]
    result, captured = _invoke_evaluate_plugin(monkeypatch, tmp_path, _prepared(tmp_path), arguments)

    assert result.exit_code == 0, result.output
    options = captured["options"]
    assert (options.loose_parts_arm, options.sum_of_parts_arm, options.workspace_skills_baseline) == (True, True, False)
    assert options.lift_mode_requested == "ladder"
    assert captured["provenance"]["requested_lift_mode"] == "ladder"
    assert captured["provenance"]["effective_lift_mode"] == "ladder"
    assert "ladder_skip_reason" not in captured["provenance"]


def test_evaluate_plugin_ladder_on_codex_runs_three_arms_and_says_why(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    arguments = ["-a", "codex", "--plugin-load", "native", "--lift-mode", "ladder"]
    result, captured = _invoke_evaluate_plugin(monkeypatch, tmp_path, _prepared(tmp_path), arguments)

    assert result.exit_code == 0, result.output
    options = captured["options"]
    assert (options.loose_parts_arm, options.sum_of_parts_arm) == (False, True)
    assert captured["provenance"]["effective_lift_mode"] == "both"
    assert captured["provenance"]["ladder_skip_reason"] == not_supported_reason("codex")
    assert "Lift ladder skipped" in result.output


def test_evaluate_plugin_ladder_with_skip_baseline_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    arguments = ["-a", AGENT, "--lift-mode", "ladder", "--skip-baseline"]
    result, captured = _invoke_evaluate_plugin(monkeypatch, tmp_path, _prepared(tmp_path), arguments)

    assert result.exit_code != 0
    assert "requires a baseline" in result.output
    assert "options" not in captured


def test_validate_tier3_forwards_ladder_to_the_plugin_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "skillevaluator.tier3.plugin_eval.prepare_plugin_eval_package", lambda *_a, **_k: _prepared(tmp_path)
    )
    monkeypatch.setattr(
        EvaluationService, "evaluate", lambda _self, options, **_kwargs: captured.setdefault("options", options) or {}
    )
    monkeypatch.setattr(EvaluationService, "failure_reason", staticmethod(lambda _result: None))
    monkeypatch.setattr("skillevaluator.evaluation.tier3_report.agent_eval_result_from_run", lambda *_a, **_k: None)

    cli_module._run_agent_eval_or_skip(
        plugin,
        agents=AGENT,
        env_mode="docker",
        skip_baseline=False,
        n_concurrent=1,
        max_agents=1,
        kind="plugin",
        lift_mode="ladder",
        plugin_load="native",
    )

    assert captured["options"].loose_parts_arm is True
    assert captured["options"].sum_of_parts_arm is True


def test_commands_evaluate_passes_the_loose_parts_arm_to_the_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from skillevaluator.tier3 import commands

    captured: dict[str, Any] = {}
    monkeypatch.setattr(commands, "resolve_llm_provider", _provider)
    monkeypatch.setattr(commands, "run_harbor_eval", lambda **kwargs: captured.update(kwargs) or {})
    skill = tmp_path / "skill"
    skill.mkdir()

    commands.evaluate(
        skill,
        agents=AGENT,
        env_mode="docker",
        skip_baseline=False,
        n_attempts=1,
        pass_threshold=None,
        n_concurrent=1,
        max_agents=1,
        model=None,
        agent_model=(),
        custom_dockerfile_mode=None,
        skill_workspace_mode=None,
        include_skills=(),
        sum_of_parts_arm=True,
        loose_parts_arm=True,
        ladder_skip_reason=None,
        copy_repo=False,
        grading_mode=None,
        results_dir=None,
        harbor_keep_jobs=False,
        timeout_multiplier=None,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
    )

    assert captured["loose_parts_arm"] is True
    assert captured["sum_of_parts_arm"] is True


# --------------------------------------------------------------------------- #
# Runner: the fourth job                                                       #
# --------------------------------------------------------------------------- #
_PAIR_SETTINGS: dict[str, Any] = {
    "skill_name": "plugin",
    "agent": AGENT,
    "model": "model",
    "env_mode": "docker",
    "run_env": {},
    "n_attempts": 3,
    "n_concurrent": 4,
    "timeout_multiplier": 2.5,
    "override_cpus": 2,
    "override_memory_mb": 4096,
    "override_storage_mb": None,
    "expected_trials": 6,
    "agent_import_path": "base:Agent",
    "with_agent_import_path": "native:Agent",
    "trial_retries": 2,
    "environment_kwargs": {"cpus": 2},
    "verifier_env": {"JUDGE": "x"},
}


def test_the_agent_pair_launches_four_jobs_with_the_same_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launched: dict[str, dict[str, Any]] = {}

    def fake_run(**kwargs: Any) -> tuple[bool, str]:
        launched[str(kwargs["job_name"]).rsplit("-", 1)[-1]] = kwargs
        return True, ""

    monkeypatch.setattr(runner, "_run_harbor", fake_run)
    errors = runner._run_agent_pair(
        **_PAIR_SETTINGS,
        with_skill=tmp_path / "with",
        baseline=tmp_path / "without",
        sum_of_parts=tmp_path / "sumofparts",
        loose_parts=tmp_path / "looseparts",
        loose_agent_import_path="loose:Agent",
        jobs_dir=tmp_path / "jobs",
    )

    assert errors == []
    assert sorted(launched) == ["looseparts", "sumofparts", "with", "without"]
    assert {variant: job["agent_import_path"] for variant, job in launched.items()} == {
        "with": "native:Agent",
        "without": "base:Agent",
        "sumofparts": "base:Agent",
        "looseparts": "loose:Agent",
    }
    assert launched["looseparts"]["dataset"] == tmp_path / "looseparts"
    assert launched["looseparts"]["job_name"] == "plugin-claude-code-looseparts"
    for key in (
        "n_attempts",
        "timeout_multiplier",
        "trial_retries",
        "expected_trials",
        "environment_kwargs",
        "verifier_env",
        "override_cpus",
        "override_memory_mb",
    ):
        assert len({json.dumps(job[key], sort_keys=True) for job in launched.values()}) == 1, key
    # The per-agent trial budget is split across the four jobs, never multiplied.
    assert sum(job["n_concurrent"] for job in launched.values()) == 4


def _launch_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, n_concurrent: int, loose: bool = True, **settings: Any
) -> dict[str, int]:
    """Launch one agent's arms with a fake Harbor; return each job's trial slots."""
    slots: dict[str, int] = {}

    def fake_run(**kwargs: Any) -> tuple[bool, str]:
        slots[str(kwargs["job_name"]).rsplit("-", 1)[-1]] = kwargs["n_concurrent"]
        return True, ""

    monkeypatch.setattr(runner, "_run_harbor", fake_run)
    monkeypatch.setattr(
        runner, "_run_stop_on_pass_variant", lambda **kwargs: slots.setdefault(kwargs["variant"], 1) and []
    )
    runner._run_agent_pair(
        **{**_PAIR_SETTINGS, "n_concurrent": n_concurrent, **settings},
        with_skill=tmp_path / "with",
        baseline=tmp_path / "without",
        sum_of_parts=tmp_path / "sumofparts",
        loose_parts=tmp_path / "looseparts" if loose else None,
        loose_agent_import_path="loose:Agent" if loose else None,
        jobs_dir=tmp_path / "jobs",
    )
    return slots


def _concurrency_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if "--n-concurrent" in record.getMessage()]


def test_ladder_below_four_slots_warns_with_the_value_that_runs_every_arm_at_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("WARNING", logger="skillevaluator.tier3.harbor.runner")

    slots = _launch_pair(monkeypatch, tmp_path, n_concurrent=3)

    # Concurrency is not raised behind the user's back: 3 workers, one trial slot each, the fourth arm waits.
    assert sorted(slots) == ["looseparts", "sumofparts", "with", "without"]
    assert set(slots.values()) == {1}
    (warning,) = _concurrency_warnings(caplog)
    assert "4 arms" in warning and "--n-concurrent is 3" in warning
    assert "--n-concurrent 4" in warning
    assert "harbor.n_concurrent" in warning


@pytest.mark.parametrize(
    ("n_concurrent", "loose", "settings"),
    [
        (4, True, {}),  # every arm runs at once
        (8, True, {}),
        (1, False, {}),  # no ladder: three arms below the slots is the user's choice, as before
        (1, True, {"stop_on_pass": True}),  # stop-on-pass runs the arms one after another anyway
    ],
    ids=["ladder-4", "ladder-8", "both-1", "ladder-stop-on-pass"],
)
def test_no_concurrency_warning_when_every_arm_runs_at_once_or_the_ladder_is_off(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    n_concurrent: int,
    loose: bool,
    settings: dict[str, Any],
) -> None:
    caplog.set_level("WARNING", logger="skillevaluator.tier3.harbor.runner")

    _launch_pair(monkeypatch, tmp_path, n_concurrent=n_concurrent, loose=loose, **settings)

    assert _concurrency_warnings(caplog) == []


def test_stop_on_pass_runs_the_loose_job_last_with_the_same_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    order: list[tuple[str, str | None, int]] = []

    def fake_variant(**kwargs: Any) -> list[str]:
        order.append((kwargs["variant"], kwargs["agent_import_path"], kwargs["trial_retries"]))
        return []

    monkeypatch.setattr(runner, "_run_stop_on_pass_variant", fake_variant)
    runner._run_agent_pair(
        **_PAIR_SETTINGS,
        with_skill=tmp_path / "with",
        baseline=tmp_path / "without",
        sum_of_parts=tmp_path / "sumofparts",
        loose_parts=tmp_path / "looseparts",
        loose_agent_import_path="loose:Agent",
        jobs_dir=tmp_path / "jobs",
        stop_on_pass=True,
    )

    assert order == [
        ("with", "native:Agent", 2),
        ("without", "base:Agent", 2),
        ("sumofparts", "base:Agent", 2),
        ("looseparts", "loose:Agent", 2),
    ]


def test_the_loose_job_is_report_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(**kwargs: Any) -> tuple[bool, str]:
        if str(kwargs["job_name"]).endswith("looseparts"):
            raise RuntimeError("loose arm crashed")
        return True, ""

    monkeypatch.setattr(runner, "_run_harbor", fake_run)
    errors = runner._run_agent_pair(
        **_PAIR_SETTINGS,
        with_skill=tmp_path / "with",
        baseline=tmp_path / "without",
        sum_of_parts=tmp_path / "sumofparts",
        loose_parts=tmp_path / "looseparts",
        loose_agent_import_path="loose:Agent",
        jobs_dir=tmp_path / "jobs",
    )

    assert errors == []
    monkeypatch.setattr(runner, "_run_harbor", lambda **_kwargs: (False, "loose launch failed"))
    errors = runner._run_agent_pair(
        **{**_PAIR_SETTINGS, "n_concurrent": 1},
        with_skill=tmp_path / "with",
        baseline=None,
        loose_parts=tmp_path / "looseparts",
        loose_agent_import_path="loose:Agent",
        jobs_dir=tmp_path / "jobs",
    )
    assert errors == ["claude-code with-skill Harbor run failed: loose launch failed"]


def test_a_loose_job_without_its_wrapper_is_refused(tmp_path: Path) -> None:
    """Without the loose wrapper the stock agent would start with no loose parts: never launch that."""
    with pytest.raises(ValueError, match="loose-parts"):
        runner._run_agent_pair(
            **_PAIR_SETTINGS,
            with_skill=tmp_path / "with",
            baseline=tmp_path / "without",
            loose_parts=tmp_path / "looseparts",
            loose_agent_import_path=None,
            jobs_dir=tmp_path / "jobs",
        )


def test_the_loose_wrapper_over_the_stock_agent_runs_on_harbor_native_backends() -> None:
    from skillevaluator.tier3.loose_parts import LOOSE_PARTS_AGENT_IMPORT_PATHS

    stock = LOOSE_PARTS_AGENT_IMPORT_PATHS[(AGENT, None)]
    command = runner.build_harbor_run_command(
        dataset_path="d", agent=AGENT, job_name="j", env_mode="gke", agent_import_path=stock
    )
    assert stock in command
    nvidia = LOOSE_PARTS_AGENT_IMPORT_PATHS[
        (AGENT, "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildClaudeCode")
    ]
    with pytest.raises(ValueError, match="only with --env docker or local"):
        runner.build_harbor_run_command(
            dataset_path="d", agent=AGENT, job_name="j", env_mode="gke", agent_import_path=nvidia
        )


# --------------------------------------------------------------------------- #
# Runner: four task directories, hashes, run config                            #
# --------------------------------------------------------------------------- #
def test_the_runner_stages_four_arms_for_native_claude_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}
    _stub_runner_edges(monkeypatch, tmp_path, captured)
    package = _package(tmp_path)

    result = _run(package, tmp_path, [AGENT])

    assert "error" not in result, result
    pair = captured["pair"]
    assert [pair[key].name for key in ("with_skill", "baseline", "sum_of_parts", "loose_parts")] == [
        "with",
        "without",
        "sumofparts",
        "looseparts",
    ]
    assert pair["loose_agent_import_path"] == "skillevaluator.tier3.harbor.loose_agents:LoosePartsNvidiaBuildClaudeCode"
    # Every arm stages the same cases.
    assert len({tuple(names) for names in captured["tasks"].values()}) == 1
    # The loose arm: member skills and the loose bundle, no wrapper skill and no plugin copy.
    env = captured["loose_env"]
    assert "skills/changelog-collect/SKILL.md" in env and "skills/version-bump/SKILL.md" in env
    assert "skilleval/native/claude-code/loose/mcp.json" in env
    assert "skilleval/native/claude-code/loose/settings.json" in env
    assert "skilleval/native/setup.sh" in env
    assert not [path for path in env if path.startswith("skilleval/native/claude-code/plugin/")]
    assert not [path for path in env if path.startswith(f"skills/{package.package_path.name}/")]
    # Scored like the skills-only arm: no target skill in the arm.
    assert captured["loose_entry"]["has_skill"] is False

    run_config = result["run_config"]
    assert run_config["skill_workspace"]["loose_parts_arm"] is True
    assert run_config["skill_workspace"]["sum_of_parts_arm"] is True
    assert run_config["lift_mode"]["requested"] == "ladder"
    assert run_config["lift_mode"]["effective"] == "ladder"
    assert run_config["loose_parts"]["agents"][AGENT]["measured"] is True

    plan = captured["collect"]["loose_parts"]
    content = plan.arm_for(AGENT).content
    assert content["equal_across_arms"] is True, content["mismatches"]
    assert set(content["content_hashes"]) == {FULL, LOOSE, SKILLS}
    assert content["tasks_hashed"] == len(captured["tasks"]["loose_parts"])
    assert "mcp:reltools" in content["content_hashes"][LOOSE]
    # The plugin signals declare the plugin's MCP server in the loose arm too.
    context = captured["collect"]["plugin_signals"]
    assert context.arm_enabled(LOOSE)
    assert "reltools" in context.declared_for(LOOSE, AGENT)["mcp"]


def test_a_non_claude_code_agent_gets_no_loose_arm_and_a_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}
    pairs: dict[str, dict[str, Any]] = {}
    _stub_runner_edges(monkeypatch, tmp_path, captured)
    monkeypatch.setattr(runner, "_run_agent_pair", lambda **kwargs: pairs.setdefault(kwargs["agent"], kwargs) and [])
    package = _package(tmp_path, agents=[AGENT, "codex"])

    result = _run(package, tmp_path, [AGENT, "codex"])

    assert "error" not in result, result
    assert pairs[AGENT]["loose_parts"] is not None
    assert pairs["codex"].get("loose_parts") is None
    agents = result["run_config"]["loose_parts"]["agents"]
    assert agents["codex"] == {"measured": False, "not_measured_reason": not_supported_reason("codex")}
    assert result["run_config"]["lift_mode"]["effective"] == "ladder"


@pytest.mark.parametrize(
    ("agents", "plugin_load", "reason"),
    [
        (["codex"], "native", not_supported_reason("codex")),
        ([AGENT], "wrapper", NOT_SUPPORTED_WRAPPER),
    ],
)
def test_no_measured_agent_runs_three_arms_and_records_why(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, agents: list[str], plugin_load: str, reason: str
) -> None:
    captured: dict[str, Any] = {}
    _stub_runner_edges(monkeypatch, tmp_path, captured)
    package = _package(tmp_path, agents=agents, plugin_load=plugin_load)

    result = _run(package, tmp_path, agents, plugin_load=plugin_load)

    assert "error" not in result, result
    assert captured["pair"].get("loose_parts") is None
    lift_mode = result["run_config"]["lift_mode"]
    assert (lift_mode["requested"], lift_mode["effective"]) == ("ladder", "both")
    assert reason in lift_mode["ladder_skip_reason"]
    assert result["run_config"]["skill_workspace"]["loose_parts_arm"] is False


def test_a_run_without_the_ladder_has_no_loose_arm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}
    _stub_runner_edges(monkeypatch, tmp_path, captured)
    package = _package(tmp_path)

    result = _run(package, tmp_path, [AGENT], loose_parts_arm=False, lift_mode_requested="both")

    assert captured["pair"].get("loose_parts") is None
    assert result["run_config"]["lift_mode"]["effective"] == "both"
    assert "loose_parts" not in result["run_config"]
    assert result["run_config"]["skill_workspace"]["loose_parts_arm"] is False
    assert captured["collect"].get("loose_parts") is None


# --------------------------------------------------------------------------- #
# Emitter                                                                      #
# --------------------------------------------------------------------------- #
def test_the_emitter_takes_a_loose_staging_in_an_arm_without_the_target_skill(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks
    from skillevaluator.tier3.harbor.native_staging import build_native_task_staging
    from skillevaluator.tier3.loose_parts import build_loose_parts
    from skillevaluator.tier3.plugin_native import ClaudeCodeAdapter

    package = _package(tmp_path)
    loose = build_loose_parts(package.native_source).task_staging()
    tasks = generate_harbor_tasks(
        package.package_path,
        tmp_path / "out" / "loose",
        with_skill=False,
        workspace_skill_paths=list(package.include_skills),
        workspace_mode="group",
        native_plugin=loose,
        arm_suffix="-without-skill",
    )

    env = tasks[0] / "environment"
    assert (env / "skilleval" / "native" / "claude-code" / "loose" / "mcp.json").is_file()
    assert (env / "skills" / "changelog-collect" / "SKILL.md").is_file()
    assert not (env / "skills" / package.package_path.name).exists()
    assert "COPY skilleval/ /skilleval/" in (env / "Dockerfile").read_text(encoding="utf-8")
    plugin_arm = build_native_task_staging(AGENT, ClaudeCodeAdapter(), package.native_source)
    with pytest.raises(ValueError, match="with-plugin arm only"):
        generate_harbor_tasks(
            package.package_path,
            tmp_path / "out" / "bad",
            with_skill=False,
            workspace_skill_paths=list(package.include_skills),
            native_plugin=plugin_arm,
        )


def test_hashes_of_emitted_tasks_catch_a_changed_piece(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor.loose_parts_arm import plan_loose_parts

    package = _package(tmp_path)
    plan = plan_loose_parts(
        requested=True,
        agents=[AGENT],
        decisions=resolve_plugin_load("native", [AGENT], env_mode="docker"),
        source=package.native_source,
        agent_import_paths={},
    )
    arm = plan.arm_for(AGENT)
    assert arm is not None
    envs: dict[str, list[Path]] = {}
    for key in (FULL, LOOSE, SKILLS):
        env = tmp_path / "tasks" / key / "case" / "environment"
        arm.parts.stage_arm(key, env)
        envs[key] = [env.parent]
    plan.hash_tasks(AGENT, envs)
    assert plan.arm_for(AGENT).content["equal_across_arms"] is True

    mcp = envs[LOOSE][0] / "environment" / "skilleval" / "native" / "claude-code" / "loose" / "mcp.json"
    mcp.write_text(mcp.read_text(encoding="utf-8").replace("--strict", "--lenient"), encoding="utf-8")
    plan.hash_tasks(AGENT, envs)

    content = plan.arm_for(AGENT).content
    assert content["equal_across_arms"] is False
    assert content["mismatches"] == ["mcp:reltools"]


# --------------------------------------------------------------------------- #
# Collector: four arms into the lift ladder                                    #
# --------------------------------------------------------------------------- #
def _init_event(arm: str, *, plugin: str = "shipit", **overrides: Any) -> dict[str, Any]:
    """Claude Code's ``system/init`` for one arm, listing every part that arm staged."""
    full = arm == FULL
    prefix = f"{plugin}:" if full else ""
    staged = arm in (FULL, LOOSE)
    event: dict[str, Any] = {
        "type": "system",
        "subtype": "init",
        "skills": [f"{prefix}{name}" for name in MEMBERS],
        "slash_commands": [f"{prefix}{name}" for name in MEMBERS] + ([f"{prefix}release-check"] if staged else []),
        "agents": ["Explore", *([f"{prefix}release-reviewer"] if staged else [])],
        "mcp_servers": (
            [{"name": f"plugin:{plugin}:reltools" if full else "reltools", "status": "connected"}] if staged else []
        ),
        "tools": ["Bash"],
        "plugins": [{"name": plugin, "path": "/skilleval/native/claude-code/plugin"}] if full else [],
    }
    event.update(overrides)
    return event


_VARIANTS = {FULL: "with", NONE: "without", SKILLS: "sumofparts", LOOSE: "looseparts"}
_ARM_SCORES = {NONE: 0.4, SKILLS: 0.6, LOOSE: 0.7, FULL: 0.8}


def _write_four_arm_run(
    tmp_path: Path,
    *,
    arms: tuple[str, ...] = (FULL, NONE, SKILLS, LOOSE),
    init: dict[tuple[str, str], dict[str, Any] | None] | None = None,
    scores: dict[tuple[str, str], dict[str, float | None]] | None = None,
) -> Path:
    """Harbor jobs for every arm (no plugin 0.4, skills 0.6, loose 0.7, full 0.8) with Claude Code init logs.

    ``scores`` replaces the metric scores of one ``(arm, case)`` trial.
    """
    jobs = tmp_path / "jobs"
    overrides = init or {}
    replaced = scores or {}
    for arm in arms:
        variant = _VARIANTS[arm]
        value = _ARM_SCORES[arm]
        write_job(
            jobs,
            AGENT,
            variant,
            {case: [replaced.get((arm, case), metrics(value, skill=arm != NONE))] for case in CASES},
        )
        for case in CASES:
            event = overrides.get((arm, case), _init_event(arm))
            log = jobs / f"{SKILL}-{AGENT}-{variant}" / f"{case}__attempt001" / "agent" / "claude-code.txt"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(json.dumps(event) + "\n" if event is not None else "no init\n", encoding="utf-8")
    return jobs


def _signals_context():
    from skillevaluator.tier3.eval_core.plugin_signals import build_plugin_signals_context

    entries = [
        {"id": case, "cross_component": True, "expected_skills": list(MEMBERS)} if case in CROSS else {"id": case}
        for case in CASES
    ]
    return build_plugin_signals_context(member_skills=MEMBERS, mcp_servers=["reltools"], entries=entries)


def _loose_plan(tmp_path: Path):
    from skillevaluator.tier3.harbor.loose_parts_arm import plan_loose_parts

    package = _package(tmp_path / "pkg")
    return plan_loose_parts(
        requested=True,
        agents=[AGENT],
        decisions=resolve_plugin_load("native", [AGENT], env_mode="docker"),
        source=package.native_source,
        agent_import_paths={},
    )


def _collect(tmp_path: Path, jobs: Path, *, loose_parts: Any = None, sum_of_parts_arm: bool = True) -> dict[str, Any]:
    from skillevaluator.tier3.harbor.collector import collect_harbor_results

    kwargs: dict[str, Any] = {} if loose_parts is None else {"loose_parts": loose_parts}
    return collect_harbor_results(
        skill_name=SKILL,
        agents=[AGENT],
        output_dir=tmp_path / "results",
        jobs_dir=jobs,
        sum_of_parts_arm=sum_of_parts_arm,
        n_attempts=1,
        expected_cases=len(CASES),
        expected_case_ids=list(CASES),
        expected_trials=len(CASES),
        plugin_signals=_signals_context(),
        **kwargs,
    )


def test_the_collector_builds_the_ladder_from_four_arms(tmp_path: Path) -> None:
    plan = _loose_plan(tmp_path)
    jobs = _write_four_arm_run(tmp_path)

    collected = _collect(tmp_path, jobs, loose_parts=plan)

    agent = collected["agents"][AGENT]
    ladder = agent["lift_ladder"]
    assert ladder is not None
    assert (ladder["n_cases"], ladder["excluded_cases"], ladder["failed_arms"]) == (len(CASES), [], [])
    steps = ladder["steps"]
    assert {name: steps[name]["estimate"] for name in steps} == {
        "skill": 0.2,
        "capability": 0.1,
        "integration": 0.1,
        "plugin": 0.4,
    }
    assert ladder["bundle"]["estimate"] == 0.2
    assert ladder["glue"]["n"] == len(CROSS)
    # The fourth arm is collected like the skills-only arm and stays report-only.
    assert agent["loose_parts"] and agent["num_trials_loose_parts"] == len(CASES)
    assert agent["conditions"][LOOSE]["execution_status"] == "succeeded"
    assert agent["execution_status"] == "succeeded"
    assert agent["loose_parts_lift"]
    results = tmp_path / "results" / AGENT
    assert (results / "loose-parts" / "summary.json").is_file()
    assert (results / "loose_parts_lift.json").is_file()
    assert (results / "loose_parts_load_evidence.json").is_file()
    persisted = json.loads((results / "statistics.json").read_text(encoding="utf-8"))
    assert persisted["lift_ladder"]["steps"]["integration"]["estimate"] == 0.1
    assert persisted["integration_completeness"]["loose_parts"]["complete"] is True
    # The run-level loose_parts block: the staging's info with the emitted hashes and the load evidence.
    block = collected["loose_parts"]
    for key in ("content_hashes", "equal_across_arms", "mismatches", "dropped_as_glue", "load_evidence"):
        assert key in block, key
    evidence = block["load_evidence"]
    assert evidence["load_failures_by_case"] == {}
    assert evidence["pieces"][LOOSE]["mcp:reltools"]["loaded"] == len(CASES)
    assert evidence["pieces"][FULL]["agent:release-reviewer"]["loaded"] == len(CASES)
    assert evidence["pieces"][SKILLS]["skill:changelog-collect"]["loaded"] == len(CASES)
    assert block["agents"][AGENT]["measured"] is True


def test_a_part_that_did_not_load_drops_the_case_from_every_arm(tmp_path: Path) -> None:
    plan = _loose_plan(tmp_path)
    jobs = _write_four_arm_run(
        tmp_path,
        init={
            (LOOSE, "x-1"): _init_event(LOOSE, mcp_servers=[{"name": "reltools", "status": "failed"}]),
            (FULL, "x-2"): _init_event(FULL, plugins=[]),
            (SKILLS, "o-1"): None,
        },
    )

    collected = _collect(tmp_path, jobs, loose_parts=plan)

    ladder = collected["agents"][AGENT]["lift_ladder"]
    excluded = {entry["case"]: entry for entry in ladder["excluded_cases"]}
    assert sorted(excluded) == ["o-1", "x-1", "x-2"]
    assert {entry["reason"] for entry in excluded.values()} == {"load_failed"}
    assert set(excluded["x-1"]["load_failures"]) == {LOOSE}
    assert "mcp:reltools" in excluded["x-1"]["load_failures"][LOOSE]
    assert set(excluded["x-2"]["load_failures"]) == {FULL}
    assert set(excluded["o-1"]["load_failures"]) == {SKILLS}
    assert ladder["n_cases"] == len(CASES) - 3
    # Every step uses the same remaining cases, so the four steps still add up per case.
    steps = ladder["steps"]
    assert {steps[name]["n_cases"] for name in steps} == {len(CASES) - 3}
    failures = collected["loose_parts"]["load_evidence"]["load_failures_by_case"]
    assert sorted(failures) == ["o-1", "x-1", "x-2"]


def test_loose_parts_lift_compares_both_arms_on_the_same_cases(tmp_path: Path) -> None:
    # The pilot's shape: on a case with no expected skill the loose arm records the skill metrics as N/A,
    # while the full plugin scores them. The old file averaged the full arm over every case (0.7333)
    # and the loose arm over the rest (0.7): different trial sets.
    plan = _loose_plan(tmp_path)
    jobs = _write_four_arm_run(
        tmp_path,
        scores={
            (FULL, "o-3"): metrics(0.8, skill_execution=0.2, skill_efficiency=0.2),
            (LOOSE, "o-3"): metrics(0.7, skill=False),
        },
    )

    collected = _collect(tmp_path, jobs, loose_parts=plan)

    lift = json.loads((tmp_path / "results" / AGENT / "loose_parts_lift.json").read_text(encoding="utf-8"))
    assert collected["agents"][AGENT]["loose_parts_lift"] == lift
    assert lift["skill_execution"] == {"with_skill": 0.8, "without_skill": 0.7, "delta": 0.1, "direction": "up"}
    assert lift["skill_efficiency"]["delta"] == 0.1
    basis = lift["basis"]
    assert basis["arms"] == [FULL, LOOSE]
    assert basis["metrics"]["skill_execution"] == {"cases": len(CASES) - 1, "trials": {FULL: 8, LOOSE: 8}}
    assert basis["metrics"]["security"] == {"cases": len(CASES), "trials": {FULL: 9, LOOSE: 9}}
    assert basis["excluded_cases"] == [
        {"case": "o-3", "reason": "not_applicable", "arms": [LOOSE], "metrics": ["skill_execution", "skill_efficiency"]}
    ]


def test_loose_parts_lift_leaves_out_the_cases_the_ladder_drops_for_load(tmp_path: Path) -> None:
    plan = _loose_plan(tmp_path)
    jobs = _write_four_arm_run(
        tmp_path,
        init={(LOOSE, "x-1"): _init_event(LOOSE, mcp_servers=[{"name": "reltools", "status": "failed"}])},
        scores={(LOOSE, "x-1"): metrics(0.1)},
    )

    collected = _collect(tmp_path, jobs, loose_parts=plan)

    lift = collected["agents"][AGENT]["loose_parts_lift"]
    assert [entry["case"] for entry in collected["agents"][AGENT]["lift_ladder"]["excluded_cases"]] == ["x-1"]
    # x-1's loose trial ran without its MCP server, so its 0.1 never reaches the comparison.
    assert lift["security"]["without_skill"] == 0.7
    assert lift["basis"]["metrics"]["security"]["cases"] == len(CASES) - 1
    (entry,) = lift["basis"]["excluded_cases"]
    assert entry["case"] == "x-1" and entry["reason"] == "load_failed" and entry["arms"] == [LOOSE]
    assert entry["metrics"] == [
        "security",
        "skill_execution",
        "skill_efficiency",
        "accuracy",
        "goal_accuracy",
        "behavior_check",
    ]


def test_a_slow_mcp_server_is_proven_by_a_successful_call(tmp_path: Path) -> None:
    plan = _loose_plan(tmp_path)
    jobs = _write_four_arm_run(
        tmp_path, init={(LOOSE, "x-1"): _init_event(LOOSE, mcp_servers=[{"name": "reltools", "status": "pending"}])}
    )
    trial = jobs / f"{SKILL}-{AGENT}-looseparts" / "x-1__attempt001" / "agent"
    trajectory = {
        "schema_version": "ATIF-v1.2",
        "session_id": "s",
        "agent": {"name": "claude-code", "version": "1"},
        "steps": [
            {"step_id": 1, "source": "user", "message": "go"},
            {
                "step_id": 2,
                "source": "agent",
                "message": "",
                "tool_calls": [{"tool_call_id": "t1", "function_name": "mcp__reltools__list_changes", "arguments": {}}],
                "observation": {"results": [{"source_call_id": "t1", "content": "[]"}]},
            },
        ],
    }
    (trial / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")

    collected = _collect(tmp_path, jobs, loose_parts=plan)

    assert collected["agents"][AGENT]["lift_ladder"]["excluded_cases"] == []
    pieces = collected["loose_parts"]["load_evidence"]["pieces"][LOOSE]["mcp:reltools"]
    assert pieces["loaded"] == len(CASES)
    assert "mcp-call" in pieces["sources"]


def test_a_missing_arm_computes_no_ladder_step(tmp_path: Path) -> None:
    plan = _loose_plan(tmp_path)
    jobs = _write_four_arm_run(tmp_path, arms=(FULL, NONE, SKILLS))

    collected = _collect(tmp_path, jobs, loose_parts=plan)

    agent = collected["agents"][AGENT]
    ladder = agent["lift_ladder"]
    assert ladder["n_cases"] == 0
    assert LOOSE in ladder["failed_arms"]
    assert all(step["estimate"] is None for step in ladder["steps"].values())
    assert ladder["glue"]["lift"] is None
    # The loose arm is report-only: its missing job never fails the agent.
    assert agent["conditions"][LOOSE]["execution_status"] == "failed"
    assert agent["execution_status"] == "succeeded"


def test_an_old_three_arm_run_is_collected_as_before(tmp_path: Path) -> None:
    jobs = _write_four_arm_run(tmp_path, arms=(FULL, NONE, SKILLS))

    collected = _collect(tmp_path, jobs)

    agent = collected["agents"][AGENT]
    assert agent["lift_ladder"] is None
    assert not [key for key in agent if "loose" in key]
    assert LOOSE not in agent["conditions"]
    assert "loose_parts" not in collected
    results = tmp_path / "results" / AGENT
    assert not (results / "loose-parts").exists()
    assert not (results / "loose_parts_lift.json").exists()
    assert (
        "loose_parts"
        not in json.loads((results / "statistics.json").read_text(encoding="utf-8"))["integration_completeness"]
    )


# --------------------------------------------------------------------------- #
# Report payload, provenance and the run summary                              #
# --------------------------------------------------------------------------- #
def _engine_result(collected: dict[str, Any], *, plugin_load: str = "native") -> dict[str, Any]:
    return {
        **collected,
        "run_config": {
            "eval_target": {"kind": "plugin"},
            "skill_workspace": {"staged_skills": MEMBERS, "sum_of_parts_arm": True, "loose_parts_arm": True},
            "lift_mode": {"requested": "ladder", "effective": "ladder", "integration_skip_reason": None},
            "plugin_load": {"requested": plugin_load, "by_agent": {AGENT: {"mode": plugin_load}}},
        },
    }


def _provenance(engine_result: dict[str, Any]) -> dict[str, Any]:
    from skillevaluator.tier3.plugin_eval import finalize_loose_parts_provenance

    provenance: dict[str, Any] = {
        "plugin_name": "shipit",
        "requested_lift_mode": "ladder",
        "effective_lift_mode": "ladder",
        "plugin_load": engine_result["run_config"]["plugin_load"],
    }
    finalize_loose_parts_provenance(provenance, engine_result)
    return provenance


def _payload(tmp_path: Path, engine_result: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
    from skillevaluator.tier3.harbor.report_data import load_agent_data

    built = build_agent_eval_payload(
        SKILL,
        load_agent_data(tmp_path / "results"),
        run_config=engine_result["run_config"],
        plugin_provenance=provenance,
        use_llm_judge=False,
    )
    assert built is not None
    return built


def test_provenance_carries_the_loose_parts_block(tmp_path: Path) -> None:
    collected = _collect(tmp_path, _write_four_arm_run(tmp_path), loose_parts=_loose_plan(tmp_path))
    engine = _engine_result(collected)

    provenance = _provenance(engine)

    assert provenance["loose_parts"] == collected["loose_parts"]
    assert provenance["loose_parts"]["equal_across_arms"] is True


def test_the_payload_gives_native_claude_code_an_integration_lift(tmp_path: Path) -> None:
    collected = _collect(tmp_path, _write_four_arm_run(tmp_path), loose_parts=_loose_plan(tmp_path))
    engine = _engine_result(collected)

    built = _payload(tmp_path, engine, _provenance(engine))

    block = built["lift_ladder"]
    assert block["measured"] is True
    assert block["integration_label"] == "Integration lift"
    assert block["steps"]["integration"]["estimate"] == 0.1
    assert block["loose_parts"]["equal_across_arms"] is True
    assert block["glue"]["verdict"] is not None
    # The report data reads the fourth arm like the skills-only arm.
    agent = built["agents"][AGENT]
    assert agent["lift_ladder"]["steps"]["capability"]["estimate"] == 0.1


def test_wrapper_mode_reports_a_wrapper_effect_without_a_verdict(tmp_path: Path) -> None:
    collected = _collect(tmp_path, _write_four_arm_run(tmp_path), loose_parts=_loose_plan(tmp_path))
    engine = _engine_result(collected, plugin_load="wrapper")

    built = _payload(tmp_path, engine, _provenance(engine))

    block = built["lift_ladder"]
    assert block["integration_label"].startswith("Wrapper effect")
    assert block["glue"]["verdict"] is None
    assert "Integration lift" not in json.dumps(block["glue"].get("reason"))


def test_a_not_measured_agent_shows_the_runner_reason(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor.loose_parts_arm import plan_loose_parts

    package = _package(tmp_path, agents=["codex"])
    plan = plan_loose_parts(
        requested=True,
        agents=["codex"],
        decisions=resolve_plugin_load("native", ["codex"], env_mode="docker"),
        source=package.native_source,
        agent_import_paths={},
    )
    engine = {
        "agents": {"codex": {"with_skill": {}, "lift_ladder": None}},
        "loose_parts": plan.record({}),
        "run_config": {
            "eval_target": {"kind": "plugin"},
            "lift_mode": {
                "requested": "ladder",
                "effective": "both",
                "ladder_skip_reason": plan.skip_reason,
            },
        },
    }
    from skillevaluator.evaluation.tier3_report import _build_ladder_report

    block = _build_ladder_report(
        engine["agents"]["codex"], engine["run_config"], _provenance_for(engine), agent_name="codex"
    )

    assert block is not None and block["measured"] is False
    assert block["reason"] == not_supported_reason("codex")[:300]


def _provenance_for(engine: dict[str, Any]) -> dict[str, Any]:
    from skillevaluator.tier3.plugin_eval import finalize_loose_parts_provenance

    provenance: dict[str, Any] = {"requested_lift_mode": "ladder"}
    finalize_loose_parts_provenance(provenance, engine)
    return provenance


def test_the_run_summary_shows_the_ladder_with_its_native_label(tmp_path: Path) -> None:
    from skillevaluator.tier3.result_display import render_result

    collected = _collect(tmp_path, _write_four_arm_run(tmp_path), loose_parts=_loose_plan(tmp_path))
    engine = _engine_result(collected)
    shown = {**engine, "plugin_provenance": _provenance(engine), "skill_name": SKILL}

    output = render_result(shown)

    assert "Lift ladder" in output
    assert "Integration lift" in output
    assert "Wrapper effect" not in output


def test_a_failed_evidence_write_never_fails_the_collection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor import loose_parts_arm

    def refuse(_path: Path, _payload: bytes) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("skillevaluator.tier3.output_provenance.write_output_file_atomically", refuse)
    plan = _loose_plan(tmp_path)
    collected = _collect(tmp_path, _write_four_arm_run(tmp_path), loose_parts=plan)

    assert collected["agents"][AGENT]["lift_ladder"]["n_cases"] == len(CASES)
    assert not (tmp_path / "results" / AGENT / loose_parts_arm.LOAD_EVIDENCE_FILENAME).exists()


def test_text_from_the_agent_log_is_redacted_in_the_evidence(tmp_path: Path) -> None:
    secret = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
    plan = _loose_plan(tmp_path)
    jobs = _write_four_arm_run(
        tmp_path, init={(LOOSE, "x-1"): _init_event(LOOSE, mcp_servers=[{"name": "reltools", "status": secret}])}
    )

    collected = _collect(tmp_path, jobs, loose_parts=plan)

    saved = (tmp_path / "results" / AGENT / "loose_parts_load_evidence.json").read_text(encoding="utf-8")
    assert secret not in saved
    assert secret not in json.dumps(collected["loose_parts"])
    # The failure itself is kept: the case is still dropped from every arm.
    assert [entry["case"] for entry in collected["agents"][AGENT]["lift_ladder"]["excluded_cases"]] == ["x-1"]
