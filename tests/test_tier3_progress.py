# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 progress reporting is structured, safe, and terminal-aware."""

from __future__ import annotations

import importlib
import io
import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest


class _TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


class _FakeLive:
    instances: ClassVar[list[_FakeLive]] = []

    def __init__(self, renderable, **kwargs) -> None:
        self.renderables = [renderable]
        self.kwargs = kwargs
        self.start_calls = 0
        self.update_calls = 0
        self.stop_calls = 0
        self.__class__.instances.append(self)

    def start(self, *, refresh: bool = False) -> None:
        self.start_calls += 1
        self.start_refresh = refresh

    def update(self, renderable, *, refresh: bool = False) -> None:
        self.renderables.append(renderable)
        self.update_calls += 1
        self.update_refresh = refresh

    def stop(self) -> None:
        self.stop_calls += 1


def _progress_module() -> Any:
    return importlib.import_module("skillevaluator.tier3.harbor.progress")


def _plan(progress: Any) -> Any:
    return progress.Tier3RunPlan(
        skill_name="demo-skill",
        environment="docker",
        agents=("codex", "opencode"),
        agent_models=(
            ("codex", "gpt-5"),
            ("opencode", "nvidia/llama"),
        ),
        provider="openai",
        task_count=2,
        case_count=2,
        attempts=3,
        baseline=True,
        concurrency=4,
        max_agents=2,
        timeout_multiplier=1.5,
    )


def _invoke_invalid_agent_progress_cli(mode: str) -> Any:
    from click.testing import CliRunner

    from skillevaluator.cli import cli

    skill = Path(__file__).parent / "fixtures" / "skills" / "simple"
    return CliRunner().invoke(
        cli,
        [
            "evaluate",
            str(skill),
            "--agents",
            "unsupported-agent",
            "--skip-baseline",
            "--progress",
            mode,
        ],
    )


def test_auto_reporter_selects_terminal_appropriate_presentation(monkeypatch: pytest.MonkeyPatch) -> None:
    progress = _progress_module()
    monkeypatch.setenv("TERM", "xterm-256color")

    assert isinstance(progress.create_progress_reporter("auto", stream=io.StringIO()), progress.PlainProgressReporter)
    assert isinstance(progress.create_progress_reporter("auto", stream=_TTYBuffer()), progress.RichProgressReporter)
    assert isinstance(progress.create_progress_reporter("plain", stream=_TTYBuffer()), progress.PlainProgressReporter)
    assert isinstance(progress.create_progress_reporter("off", stream=_TTYBuffer()), progress.NullProgressReporter)


@pytest.mark.parametrize("term", ["dumb", "unknown"])
def test_auto_reporter_uses_immediate_plain_output_for_non_interactive_tty(
    monkeypatch: pytest.MonkeyPatch,
    term: str,
) -> None:
    progress = _progress_module()
    monkeypatch.setenv("TERM", term)
    output = _TTYBuffer()

    reporter = progress.create_progress_reporter("auto", stream=output)
    reporter.start(_plan(progress))

    assert type(reporter) is progress.PlainProgressReporter
    assert output.getvalue().startswith("Tier 3 live evaluation: demo-skill\n")
    reporter.close()


def test_rich_reporter_owns_live_table_and_updates_stage_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    import rich.live
    from rich.console import Console
    from rich.table import Table

    progress = _progress_module()
    _FakeLive.instances.clear()
    monkeypatch.setattr(rich.live, "Live", _FakeLive)
    reporter = progress.RichProgressReporter(stream=_TTYBuffer(), refresh_interval=60)

    reporter.start(_plan(progress))
    reporter.emit(progress.ProgressEvent(stage="configuration", state="running", detail="loading"))
    reporter.emit(progress.ProgressEvent(stage="configuration", state="ready", detail="validated"))
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="with-skill + baseline"))
    reporter.heartbeat()

    assert len(_FakeLive.instances) == 1
    live = _FakeLive.instances[0]
    assert live.start_calls == 1
    assert live.start_refresh is True
    assert live.update_calls >= 4
    assert all(isinstance(renderable, Table) for renderable in live.renderables)
    running_table = live.renderables[-1]
    assert "Harbor Run Configuration" in str(running_table.title)
    rendered = io.StringIO()
    Console(file=rendered, force_terminal=False, width=180).print(running_table)
    text = rendered.getvalue()
    assert "Skill-lift evaluation" in text
    assert "Agent codex" in text
    assert "running" in text
    assert "00:00:00" in text

    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="complete"))
    reporter.emit(
        progress.ProgressEvent(
            stage="run-finished",
            state="complete",
            output_dir="/tmp/results/run-1",
            result_path="/tmp/results/run-1/result.json",
            report_path="/tmp/results/run-1/report.html",
        )
    )
    terminal = io.StringIO()
    Console(file=terminal, force_terminal=False, width=180).print(live.renderables[-1])
    terminal_text = terminal.getvalue()
    assert "/tmp/results/run-1/result.json" not in terminal_text
    assert "/tmp/results/run-1/report.html" not in terminal_text

    reporter.close()
    assert live.stop_calls == 1
    assert reporter.is_active is False


def test_rich_reporter_immediately_starts_one_live_box_and_keeps_stage_updates_inside_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rich.live
    from rich.console import Console

    progress = _progress_module()
    _FakeLive.instances.clear()
    monkeypatch.setattr(rich.live, "Live", _FakeLive)
    output = _TTYBuffer()
    reporter = progress.RichProgressReporter(stream=output, refresh_interval=60)

    reporter.start(_plan(progress))

    assert len(_FakeLive.instances) == 1
    live = _FakeLive.instances[0]
    assert live.start_calls == 1
    assert live.start_refresh is True
    initial = io.StringIO()
    Console(file=initial, force_terminal=False, width=180).print(live.renderables[0])
    assert "Harbor Run Configuration" in initial.getvalue()
    assert "demo-skill" in initial.getvalue()
    assert output.getvalue() == ""

    reporter.emit(progress.ProgressEvent(stage="agent-runtime-preflight", state="running"))

    updated = io.StringIO()
    Console(file=updated, force_terminal=False, width=180).print(live.renderables[-1])
    assert "Agent Runtime Preflight" in updated.getvalue()
    assert "running" in updated.getvalue()
    assert output.getvalue() == ""
    reporter.close()


def test_skip_baseline_rich_title_says_agent_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    import rich.live
    from rich.console import Console

    progress = _progress_module()
    _FakeLive.instances.clear()
    monkeypatch.setattr(rich.live, "Live", _FakeLive)
    plan = _plan(progress)
    plan = progress.Tier3RunPlan(
        **{field: getattr(plan, field) for field in plan.__dataclass_fields__ if field != "baseline"},
        baseline=False,
    )
    reporter = progress.RichProgressReporter(stream=_TTYBuffer(), refresh_interval=60)
    reporter.start(plan)
    rendered = io.StringIO()
    Console(file=rendered, force_terminal=False, width=180).print(_FakeLive.instances[0].renderables[-1])

    assert "Agent evaluation" in rendered.getvalue()
    assert "Skill-lift evaluation" not in rendered.getvalue()
    reporter.close()


def test_rich_reporter_stops_live_display_during_exception_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    import rich.live

    progress = _progress_module()
    _FakeLive.instances.clear()
    monkeypatch.setattr(rich.live, "Live", _FakeLive)
    reporter = progress.RichProgressReporter(stream=_TTYBuffer(), refresh_interval=60)

    with pytest.raises(RuntimeError, match="Harbor failed"):
        try:
            reporter.start(_plan(progress))
            reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running"))
            raise RuntimeError("Harbor failed")
        finally:
            reporter.close()

    live = _FakeLive.instances[0]
    assert live.stop_calls == 1
    rendered = io.StringIO()
    from rich.console import Console

    Console(file=rendered, force_terminal=False, width=180).print(live.renderables[-1])
    assert "running" not in rendered.getvalue()
    assert "failed" in rendered.getvalue()
    reporter.close()
    assert live.stop_calls == 1


def test_plain_reporter_emits_durable_plan_events_and_elapsed_heartbeat() -> None:
    progress = _progress_module()
    output = io.StringIO()
    reporter = progress.PlainProgressReporter(stream=output, refresh_interval=60)

    reporter.start(_plan(progress))
    reporter.emit(progress.ProgressEvent(stage="configuration", state="running"))
    reporter.emit(progress.ProgressEvent(stage="configuration", state="ready", detail="eval config loaded"))
    reporter.emit(progress.ProgressEvent(stage="docker-images", state="delegated", detail="build delegated to Harbor"))
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="with-skill + baseline"))
    reporter.heartbeat()
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="complete"))
    reporter.close()

    rendered = output.getvalue()
    assert "Tier 3 live evaluation: demo-skill" in rendered
    assert "environment: docker" in rendered
    assert "codex=gpt-5" in rendered
    assert "provider: openai" in rendered
    assert "tasks=2" in rendered
    assert "cases=2" in rendered
    assert "attempts=3" in rendered
    assert "baseline=yes" in rendered
    assert "concurrency=4" in rendered
    assert "timeout=1.5x" in rendered
    assert "configuration: ready - eval config loaded" in rendered
    assert "docker-images: delegated - build delegated to Harbor" in rendered
    assert "still running" in rendered
    assert "agent:codex" in rendered
    assert "\x1b[" not in rendered
    assert "\r" not in rendered


def test_plain_reporter_leaves_terminal_paths_to_the_final_footer() -> None:
    progress = _progress_module()
    output = io.StringIO()
    reporter = progress.PlainProgressReporter(stream=output, refresh_interval=60)
    reporter.start(_plan(progress))

    reporter.emit(
        progress.ProgressEvent(
            stage="run-finished",
            state="complete",
            output_dir="/tmp/results/run-1",
            result_path="/tmp/results/run-1/result.json",
            report_path="/tmp/results/run-1/report.html",
        )
    )
    reporter.close()

    rendered = output.getvalue()
    assert "run-finished: complete" in rendered
    assert "output_dir=/tmp/results/run-1" not in rendered
    assert "result_path=/tmp/results/run-1/result.json" not in rendered
    assert "report_path=/tmp/results/run-1/report.html" not in rendered
    assert "[00:00:00]" in rendered


def test_plain_reporter_supports_start_close_start() -> None:
    progress = _progress_module()
    output = io.StringIO()
    reporter = progress.PlainProgressReporter(stream=output, refresh_interval=60)

    reporter.start(_plan(progress))
    reporter.close()
    reporter.start(_plan(progress))

    assert reporter.is_active is True
    reporter.emit(progress.ProgressEvent(stage="configuration", state="complete"))
    reporter.close()
    assert output.getvalue().count("Tier 3 live evaluation: demo-skill") == 2


def test_plain_reporter_renders_one_config_header_for_plan_updates() -> None:
    progress = _progress_module()
    output = io.StringIO()
    reporter = progress.PlainProgressReporter(stream=output, refresh_interval=60)
    reporter.start(_plan(progress))
    reporter.start(_plan(progress))
    reporter.close()

    assert output.getvalue().count("Tier 3 live evaluation: demo-skill") == 1
    assert "plan update:" in output.getvalue()


def test_non_tty_auto_reporter_emits_one_line_per_configuration_transition() -> None:
    progress = _progress_module()
    output = io.StringIO()
    reporter = progress.create_progress_reporter("auto", stream=output)

    reporter.start(_plan(progress))
    reporter.start(_plan(progress))
    reporter.emit(progress.ProgressEvent(stage="configuration", state="running", detail="loading"))
    reporter.emit(progress.ProgressEvent(stage="configuration", state="ready", detail="validated"))
    reporter.close()

    rendered = output.getvalue()
    configuration_lines = [line.split("] ", 1)[-1] for line in rendered.splitlines() if "configuration:" in line]
    assert configuration_lines == [
        "configuration: running - loading",
        "configuration: ready - validated",
    ]
    assert rendered.count("Tier 3 live evaluation: demo-skill") == 1
    assert "Harbor Run Configuration" not in rendered


def test_cli_rich_progress_keeps_configuration_transitions_inside_one_live_box() -> None:
    result = _invoke_invalid_agent_progress_cli("rich")

    assert result.exit_code == 1, result.output
    assert result.output.count("Harbor Run Configuration") == 1
    stage_lines = [
        line
        for line in result.output.splitlines()
        if "Configuration" in line and "Harbor Run Configuration" not in line
    ]
    assert len(stage_lines) == 1
    assert "failed" in stage_lines[0]
    assert "→ Configuration" not in result.output
    assert "✗ Configuration" not in result.output
    assert "Tier 3 Evaluation" not in result.output


def test_cli_plain_progress_emits_exact_configuration_sequence() -> None:
    result = _invoke_invalid_agent_progress_cli("plain")

    assert result.exit_code == 1, result.output
    configuration_lines = [line.split("] ", 1)[-1] for line in result.output.splitlines() if "configuration:" in line]
    assert configuration_lines[0] == "configuration: running"
    assert len(configuration_lines) == 2
    assert configuration_lines[1].startswith("configuration: failed - Unknown agent(s): unsupported-agent.")
    assert result.output.count("Tier 3 live evaluation: simple") == 1
    assert "Harbor Run Configuration" not in result.output


def test_rich_reporter_supports_start_close_start(monkeypatch: pytest.MonkeyPatch) -> None:
    import rich.live

    progress = _progress_module()
    _FakeLive.instances.clear()
    monkeypatch.setattr(rich.live, "Live", _FakeLive)
    reporter = progress.RichProgressReporter(stream=_TTYBuffer(), refresh_interval=60)

    reporter.start(_plan(progress))
    reporter.close()
    reporter.start(_plan(progress))
    assert reporter.is_active is True
    reporter.close()

    assert len(_FakeLive.instances) == 2
    assert [live.stop_calls for live in _FakeLive.instances] == [1, 1]


def test_safe_reporter_disables_broken_presentation_without_raising() -> None:
    progress = _progress_module()

    class BrokenReporter:
        is_active = False

        def __init__(self) -> None:
            self.close_calls = 0
            self.start_calls = 0

        def start(self, _plan) -> None:
            self.start_calls += 1
            raise RuntimeError("terminal unavailable")

        def set_secret_values(self, _values) -> None:
            raise RuntimeError("terminal unavailable")

        def emit(self, _event) -> None:
            raise RuntimeError("terminal unavailable")

        def heartbeat(self) -> None:
            raise RuntimeError("terminal unavailable")

        def close(self) -> None:
            self.close_calls += 1

    broken = BrokenReporter()
    reporter = progress.safe_progress_reporter(broken)

    reporter.set_secret_values(["secret"])
    reporter.start(_plan(progress))
    reporter.emit(progress.ProgressEvent(stage="configuration", state="running"))
    reporter.heartbeat()
    reporter.close()

    assert broken.close_calls >= 1
    reporter.start(_plan(progress))
    reporter.close()
    assert broken.start_calls == 1


def test_safe_reporter_tracks_lifecycle_when_delegate_has_no_active_state() -> None:
    progress = _progress_module()
    reporter = progress.safe_progress_reporter(progress.NullProgressReporter())

    reporter.start(_plan(progress))
    assert reporter.is_active is True
    reporter.close()
    assert reporter.is_active is False


def test_reporters_redact_exact_values_and_secret_shaped_details() -> None:
    progress = _progress_module()
    output = io.StringIO()
    reporter = progress.PlainProgressReporter(stream=output, refresh_interval=60)
    reporter.set_secret_values(["literal-super-secret"])

    reporter.start(_plan(progress))
    reporter.emit(
        progress.ProgressEvent(
            stage="credential-validation",
            state="failed",
            detail=("token=literal-super-secret API_KEY=plain-secret-value OPENAI_API_KEY=sk-AbCdEf1234567890"),
        )
    )
    reporter.close()

    rendered = output.getvalue()
    assert "literal-super-secret" not in rendered
    assert "plain-secret-value" not in rendered
    assert "sk-AbCdEf1234567890" not in rendered
    assert "<redacted>" in rendered


def test_progress_detail_strips_osc_title_and_hyperlink_payloads() -> None:
    progress = _progress_module()

    rendered = progress.redact_progress_detail(
        "safe\x1b]0;forged title\x07 text \x1b]8;;https://malicious.example\x1b\\click\x1b]8;;\x1b\\ done"
    )

    assert "forged title" not in rendered
    assert "malicious.example" not in rendered
    assert "\x1b" not in rendered
    assert rendered == "safe text click done"


def test_plain_reporter_redacts_secrets_from_plan_values() -> None:
    progress = _progress_module()
    output = io.StringIO()
    reporter = progress.PlainProgressReporter(stream=output, refresh_interval=60)
    secret = "sk-AbCdEf1234567890"
    reporter.set_secret_values([secret])
    plan = progress.Tier3RunPlan(
        skill_name="demo",
        environment="docker",
        agents=("codex",),
        agent_models=(("codex", f"model-{secret}"),),
        provider="openai",
    )

    reporter.start(plan)
    reporter.close()

    rendered = output.getvalue()
    assert secret not in rendered
    assert "<redacted>" in rendered


def test_plain_reporter_strips_terminal_controls_from_untrusted_plan_text() -> None:
    progress = _progress_module()
    output = io.StringIO()
    reporter = progress.PlainProgressReporter(stream=output, refresh_interval=60)

    reporter.start(
        progress.Tier3RunPlan(
            skill_name="demo\x1b[2JINJECT",
            environment="docker",
            agents=("opencode",),
            agent_models=(("opencode", "safe\x1b[31mRED\x00"),),
        )
    )
    reporter.close()

    rendered = output.getvalue()
    assert "\x1b" not in rendered
    assert "\x00" not in rendered
    assert "demoINJECT" in rendered
    assert "safeRED" in rendered


def test_null_reporter_has_no_output_or_background_activity() -> None:
    progress = _progress_module()
    reporter = progress.NullProgressReporter()

    reporter.start(_plan(progress))
    reporter.set_secret_values(["unused"])
    reporter.emit(progress.ProgressEvent(stage="configuration", state="running"))
    reporter.heartbeat()
    reporter.close()

    assert reporter.is_active is False


class _RecordingReporter:
    def __init__(self, timeline: list[str] | None = None) -> None:
        self.plans: list[Any] = []
        self.events: list[Any] = []
        self.secret_values: set[str] = set()
        self.closed = False
        self.timeline = timeline
        self.callback_threads: list[int] = []

    @property
    def is_active(self) -> bool:
        return bool(self.plans) and not self.closed

    def start(self, plan: Any) -> None:
        self.callback_threads.append(threading.get_ident())
        self.plans.append(plan)

    def set_secret_values(self, values) -> None:
        self.callback_threads.append(threading.get_ident())
        self.secret_values.update(values)

    def emit(self, event: Any) -> None:
        self.callback_threads.append(threading.get_ident())
        self.events.append(event)
        if self.timeline is not None:
            self.timeline.append(f"event:{event.stage}:{event.state}")

    def heartbeat(self) -> None:
        return None

    def close(self) -> None:
        self.callback_threads.append(threading.get_ident())
        self.closed = True


def _stub_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    run_agent=None,
    collect=None,
    html_report=None,
    environment_check=None,
    credential_check=None,
    task_emitter=None,
    skill_name: str = "demo-skill",
    provider_model: str = "gpt-5",
):
    from skillevaluator.tier3.harbor import runner

    skill = tmp_path / skill_name
    (skill / "evals").mkdir(parents=True)
    evals_file = skill / "evals" / "evals.json"
    evals_file.write_text("{}", encoding="utf-8")
    provider = SimpleNamespace(
        provider="openai",
        model=provider_model,
        api_key="exact-provider-secret",
        base_url=None,
    )

    def emit_tasks(_skill, output, **_kwargs):
        output.mkdir(parents=True, exist_ok=True)
        return [output / "case-1"]

    def collect_results(**_kwargs):
        return {"execution_status": "succeeded", "execution_errors": [], "metrics": []}

    def write_html(_skill_path, run_dir, **_kwargs):
        path = run_dir / "report.html"
        path.write_text("<html></html>", encoding="utf-8")
        return path

    monkeypatch.setattr(runner, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(
        runner,
        "load_evals_config",
        lambda _path: (
            {
                "harbor": {
                    "task_source": "evals_json",
                    "n_attempts": 1,
                    "n_concurrent": 1,
                    "max_agents": 1,
                }
            },
            None,
        ),
    )
    monkeypatch.setattr(runner, "_check_prerequisites", environment_check or (lambda **_kwargs: []))
    monkeypatch.setattr(runner, "find_evals_file", lambda _path: evals_file)
    monkeypatch.setattr(runner, "generate_harbor_tasks", task_emitter or emit_tasks)
    monkeypatch.setattr(runner, "_provider_environment", lambda _provider: {"OPENAI_API_KEY": provider.api_key})
    monkeypatch.setattr(runner, "_resolve_runtime_env", lambda _templates: ({}, []))
    monkeypatch.setattr(
        runner,
        "_validate_agent_provider_credentials",
        credential_check or (lambda *_args, **_kwargs: []),
    )
    monkeypatch.setattr(runner, "_harbor_subprocess_environment", lambda **_kwargs: {})
    monkeypatch.setattr(runner, "_run_agent_pair", run_agent or (lambda **_kwargs: []))
    from skillevaluator.tier3.harbor import runtime_preflight

    monkeypatch.setattr(
        runtime_preflight,
        "run_agent_runtime_preflight",
        lambda **kwargs: runtime_preflight.PreflightResult(
            True,
            kwargs["agent"],
            kwargs["model"],
            "agent started",
            f"runtime-preflight-{kwargs['agent']}",
        ),
    )
    monkeypatch.setattr(runner, "collect_harbor_results", collect or collect_results)
    monkeypatch.setattr(runner, "render_agent_eval_html_report", html_report or write_html)
    return runner, skill


def test_default_run_cleans_transient_harbor_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, skill = _stub_runner(monkeypatch, tmp_path)

    result = runner.run_harbor_eval(skill, ["codex"], output_dir=tmp_path / "results")

    run_dir = Path(result["run_dir"])
    assert not (run_dir / "_harbor-jobs").exists()
    assert not (run_dir / "_harbor-tasks").exists()
    assert result["harbor_jobs_retained"] is False
    assert result["harbor_jobs_retention_reason"] == "not_retained"
    persisted = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert persisted["harbor_jobs_retained"] is False
    assert persisted["run_config"]["harbor"]["jobs_retained"] is False


def test_runner_persists_compact_feedback_to_result_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    feedback = {
        "schema_version": "1.0",
        "conclusions": [{"message": "Execution feedback"}],
        "recommendations": [{"message": "Actionable suggestion"}],
        "suggestions": ["Actionable suggestion"],
        "suggestions_v2": [],
    }

    def write_html(_skill_path, run_dir, *, engine_result, **_kwargs):
        engine_result["tier3_feedback"] = feedback
        path = run_dir / "report.html"
        path.write_text("<html></html>", encoding="utf-8")
        return path

    runner, skill = _stub_runner(monkeypatch, tmp_path, html_report=write_html)

    result = runner.run_harbor_eval(skill, ["codex"], output_dir=tmp_path / "results")
    persisted = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))

    assert persisted["tier3_feedback"] == feedback
    assert "agent_eval" not in persisted


def test_keep_flag_retains_harbor_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, skill = _stub_runner(monkeypatch, tmp_path)

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        output_dir=tmp_path / "results",
        keep_harbor_jobs=True,
    )

    run_dir = Path(result["run_dir"])
    assert (run_dir / "_harbor-jobs").is_dir()
    assert (run_dir / "_harbor-tasks").is_dir()
    assert result["harbor_jobs_retained"] is True
    assert result["harbor_jobs_retention_reason"] == "explicit_keep"


def test_default_task_staging_failure_cleans_transient_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_staging(*_args, **_kwargs):
        raise ValueError("invalid task")

    runner, skill = _stub_runner(monkeypatch, tmp_path, task_emitter=fail_staging)

    result = runner.run_harbor_eval(skill, ["codex"], output_dir=tmp_path / "results")

    run_dir = Path(result["run_dir"])
    assert result["error"] == ["invalid task"]
    assert not (run_dir / "_harbor-jobs").exists()
    assert not (run_dir / "_harbor-tasks").exists()
    assert result["harbor_jobs_retained"] is False


def test_default_collection_exception_cleans_transient_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_collection(**_kwargs):
        raise RuntimeError("collection exploded")

    runner, skill = _stub_runner(monkeypatch, tmp_path, collect=fail_collection)
    results_root = tmp_path / "results"

    with pytest.raises(RuntimeError, match="collection exploded"):
        runner.run_harbor_eval(skill, ["codex"], output_dir=results_root)

    run_dirs = [path for path in results_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    assert not (run_dirs[0] / "_harbor-jobs").exists()
    assert not (run_dirs[0] / "_harbor-tasks").exists()


def test_runtime_preflight_running_event_precedes_slow_preflight_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, skill = _stub_runner(monkeypatch, tmp_path)
    reporter = _RecordingReporter()
    from skillevaluator.tier3.harbor import runtime_preflight

    def observed_preflight(**kwargs):
        transitions = [(event.stage, event.state) for event in reporter.events]
        assert ("agent-runtime-preflight", "running") in transitions
        return runtime_preflight.PreflightResult(
            True,
            kwargs["agent"],
            kwargs["model"],
            "agent started",
            f"runtime-preflight-{kwargs['agent']}",
        )

    monkeypatch.setattr(runtime_preflight, "run_agent_runtime_preflight", observed_preflight)

    runner.run_harbor_eval(
        skill,
        ["codex"],
        output_dir=tmp_path / "results",
        keep_harbor_jobs=True,
        progress_reporter=reporter,
    )


def test_cleanup_failure_degrades_terminal_progress_after_retention_finalizes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, skill = _stub_runner(monkeypatch, tmp_path)
    reporter = _RecordingReporter()
    from skillevaluator.tier3.harbor import artifact_retention

    monkeypatch.setattr(
        artifact_retention.shutil,
        "rmtree",
        lambda path: (_ for _ in ()).throw(OSError(f"busy: {path}")),
    )

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        output_dir=tmp_path / "results",
        progress_reporter=reporter,
    )

    assert result["execution_status"] == "succeeded"
    assert result["harbor_jobs_retention_reason"] == "cleanup_failed"
    assert any("Harbor artifact cleanup failed" in warning for warning in result["warnings"])
    assert (reporter.events[-1].stage, reporter.events[-1].state) == ("run-finished", "degraded")


def test_runner_emits_truthful_stages_plan_and_per_agent_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator.tier3.harbor import runner

    operations: list[str] = []
    timeline: list[str] = []
    skill = tmp_path / "demo-skill"
    (skill / "evals").mkdir(parents=True)
    evals_file = skill / "evals" / "evals.json"
    evals_file.write_text("{}", encoding="utf-8")
    reporter = _RecordingReporter(timeline)

    monkeypatch.setattr(
        runner,
        "resolve_llm_provider",
        lambda: SimpleNamespace(provider="openai", model="gpt-5", api_key="sk-test-secret", base_url=None),
    )
    monkeypatch.setattr(
        runner,
        "load_evals_config",
        lambda _path: (
            {
                "harbor": {
                    "task_source": "evals_json",
                    "n_attempts": 2,
                    "n_concurrent": 3,
                    "max_agents": 1,
                    "timeout_multiplier": 1.25,
                }
            },
            None,
        ),
    )

    def _preflight(**_kwargs):
        operations.append("environment-preflight")
        return []

    def _emit_tasks(_skill, output, **kwargs):
        variant = "with" if kwargs["with_skill"] else "baseline"
        operations.append(f"emit-{variant}")
        output.mkdir(parents=True, exist_ok=True)
        return [output / "case-1", output / "case-2"]

    monkeypatch.setattr(runner, "_check_prerequisites", _preflight)
    monkeypatch.setattr(runner, "find_evals_file", lambda _path: evals_file)
    monkeypatch.setattr(runner, "generate_harbor_tasks", _emit_tasks)
    monkeypatch.setattr(runner, "_provider_environment", lambda _provider: {"OPENAI_API_KEY": "sk-test-secret"})
    monkeypatch.setattr(runner, "_resolve_runtime_env", lambda _templates: ({}, []))
    monkeypatch.setattr(runner, "_validate_agent_provider_credentials", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(runner, "_harbor_subprocess_environment", lambda **_kwargs: {})
    from skillevaluator.tier3.harbor import runtime_preflight

    monkeypatch.setattr(
        runtime_preflight,
        "run_agent_runtime_preflight",
        lambda **kwargs: runtime_preflight.PreflightResult(
            True,
            kwargs["agent"],
            kwargs["model"],
            "agent started",
            f"runtime-preflight-{kwargs['agent']}",
        ),
    )

    def _run_agent(**kwargs):
        agent = kwargs["agent"]
        operations.append(f"run-agent:{agent}")
        timeline.append(f"operation:{agent}")
        return []

    monkeypatch.setattr(runner, "_run_agent_pair", _run_agent)
    monkeypatch.setattr(
        runner,
        "collect_harbor_results",
        lambda **_kwargs: (
            operations.append("collect") or {"execution_status": "succeeded", "execution_errors": [], "metrics": []}
        ),
    )

    def _write_report(_skill_path, run_dir, **_kwargs):
        operations.append("report")
        report = run_dir / "report.html"
        report.write_text("<html></html>", encoding="utf-8")
        return report

    monkeypatch.setattr(runner, "render_agent_eval_html_report", _write_report)
    result = runner.run_harbor_eval(
        skill,
        ["codex", "opencode"],
        output_dir=tmp_path / "results",
        keep_harbor_jobs=True,
        progress_reporter=reporter,
    )

    assert result["execution_status"] == "succeeded"
    final_plan = reporter.plans[-1]
    assert final_plan.provider == "openai"
    assert final_plan.environment == "docker"
    assert final_plan.agent_models == (("codex", "gpt-5"), ("opencode", "openai/gpt-5"))
    assert final_plan.task_count == 2
    assert final_plan.case_count == 2
    assert final_plan.attempts == 2
    assert final_plan.baseline is True
    assert final_plan.concurrency == 3
    assert final_plan.timeout_multiplier == 1.25
    assert final_plan.matrix_trials == 16
    assert final_plan.preflight_trials == 2
    assert final_plan.total_containers == 18
    assert "sk-test-secret" in reporter.secret_values

    transitions = [(event.stage, event.state) for event in reporter.events]
    assert transitions == [
        ("configuration", "running"),
        ("configuration", "ready"),
        ("model-resolution", "running"),
        ("model-resolution", "complete"),
        ("environment-preflight", "running"),
        ("environment-preflight", "complete"),
        ("credential-validation", "running"),
        ("credential-validation", "complete"),
        ("with-skill-tasks", "running"),
        ("with-skill-tasks", "ready"),
        ("baseline-tasks", "running"),
        ("baseline-tasks", "ready"),
        ("docker-images", "delegated"),
        ("agent-runtime-preflight", "running"),
        ("agent-runtime-preflight", "complete"),
        ("agent:codex", "running"),
        ("agent:codex", "complete"),
        ("agent:opencode", "running"),
        ("agent:opencode", "complete"),
        ("collection", "running"),
        ("collection", "complete"),
        ("report", "running"),
        ("report", "complete"),
        ("run-finished", "complete"),
    ]
    assert transitions.index(("environment-preflight", "complete")) > transitions.index(
        ("environment-preflight", "running")
    )
    assert operations == [
        "environment-preflight",
        "emit-with",
        "emit-with",
        "emit-baseline",
        "emit-baseline",
        "run-agent:codex",
        "run-agent:opencode",
        "collect",
        "report",
    ]
    assert timeline.index("event:agent:opencode:running") > timeline.index("operation:codex")
    delegated = next(event for event in reporter.events if event.stage == "docker-images")
    assert "delegated to Harbor" in delegated.detail
    finished = reporter.events[-1]
    assert finished.output_dir == result["run_dir"]
    assert finished.result_path == result["result_path"]
    assert finished.report_path == result["report_path"]
    assert Path(finished.result_path).is_file()
    assert Path(finished.report_path).is_file()


def test_runner_reports_known_plan_without_claiming_failed_preflight_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator.tier3.harbor import runner

    reporter = _RecordingReporter()
    monkeypatch.setattr(
        runner,
        "resolve_llm_provider",
        lambda: SimpleNamespace(provider="openai", model="gpt-5", api_key="secret-value", base_url=None),
    )
    monkeypatch.setattr(
        runner,
        "load_evals_config",
        lambda _path: (
            {"harbor": {"n_attempts": 2, "n_concurrent": 3, "max_agents": 1, "timeout_multiplier": 1.25}},
            None,
        ),
    )
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: ["Docker is unavailable"])

    result = runner.run_harbor_eval(tmp_path / "demo", ["codex"], progress_reporter=reporter)

    assert result == {"error": ["Docker is unavailable"]}
    known_plan = reporter.plans[-1]
    assert known_plan.provider == "openai"
    assert known_plan.agent_models == (("codex", "gpt-5"),)
    assert known_plan.attempts == 2
    transitions = [(event.stage, event.state) for event in reporter.events]
    assert ("environment-preflight", "running") in transitions
    assert ("environment-preflight", "failed") in transitions
    assert ("environment-preflight", "complete") not in transitions


def test_runner_does_not_mark_invalid_configuration_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator.tier3.harbor import runner

    reporter = _RecordingReporter()
    monkeypatch.setattr(
        runner,
        "resolve_llm_provider",
        lambda: SimpleNamespace(provider="openai", model="gpt-5", api_key="secret-value", base_url=None),
    )
    monkeypatch.setattr(
        runner,
        "load_evals_config",
        lambda _path: ({"harbor": {"n_attempts": 0}}, None),
    )

    result = runner.run_harbor_eval(tmp_path / "demo", ["codex"], progress_reporter=reporter)

    assert result == {"error": ["n_attempts must be >= 1"]}
    transitions = [(event.stage, event.state) for event in reporter.events]
    assert transitions == [
        ("configuration", "running"),
        ("configuration", "failed"),
        ("run-finished", "failed"),
    ]


def test_runner_invokes_reporter_callbacks_only_on_coordinator_thread(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, skill = _stub_runner(monkeypatch, tmp_path)
    reporter = _RecordingReporter()
    coordinator_thread = threading.get_ident()

    runner.run_harbor_eval(
        skill,
        ["codex"],
        output_dir=tmp_path / "results",
        keep_harbor_jobs=True,
        progress_reporter=reporter,
    )

    assert reporter.callback_threads
    assert set(reporter.callback_threads) == {coordinator_thread}
    assert reporter.closed is True


def test_runner_terminalizes_agent_when_worker_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_worker(**_kwargs):
        raise RuntimeError("worker exploded")

    runner, skill = _stub_runner(monkeypatch, tmp_path, run_agent=fail_worker)
    reporter = _RecordingReporter()

    with pytest.raises(RuntimeError, match="worker exploded"):
        runner.run_harbor_eval(
            skill,
            ["codex"],
            output_dir=tmp_path / "results",
            keep_harbor_jobs=True,
            progress_reporter=reporter,
        )

    transitions = [(event.stage, event.state) for event in reporter.events]
    assert ("agent:codex", "running") in transitions
    assert ("agent:codex", "failed") in transitions
    assert transitions[-1] == ("run-finished", "failed")
    assert transitions.count(("run-finished", "failed")) == 1


@pytest.mark.parametrize(
    ("failure_hook", "expected_stage"),
    [
        ("environment_check", "environment-preflight"),
        ("credential_check", "credential-validation"),
        ("task_emitter", "with-skill-tasks"),
    ],
)
def test_runner_terminalizes_active_stage_for_unexpected_orchestrator_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_hook: str,
    expected_stage: str,
) -> None:
    def fail_unexpectedly(*_args, **_kwargs):
        raise RuntimeError(f"{failure_hook} exploded")

    runner, skill = _stub_runner(monkeypatch, tmp_path, **{failure_hook: fail_unexpectedly})
    reporter = _RecordingReporter()

    with pytest.raises(RuntimeError, match=rf"^{failure_hook} exploded$"):
        runner.run_harbor_eval(
            skill,
            ["codex"],
            output_dir=tmp_path / "results",
            keep_harbor_jobs=True,
            progress_reporter=reporter,
        )

    transitions = [(event.stage, event.state) for event in reporter.events]
    assert (expected_stage, "running") in transitions
    assert (expected_stage, "failed") in transitions
    assert transitions[-1] == ("run-finished", "failed")
    assert transitions.count(("run-finished", "failed")) == 1

    finished = reporter.events[-1]
    if failure_hook == "task_emitter":
        assert finished.output_dir is not None
        assert Path(finished.output_dir).is_dir()
        assert finished.result_path is None


def test_runner_terminalizes_baseline_staging_for_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = 0

    def fail_baseline(_skill, output, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("baseline emitter exploded")
        output.mkdir(parents=True, exist_ok=True)
        return [output / "case-1"]

    runner, skill = _stub_runner(monkeypatch, tmp_path, task_emitter=fail_baseline)
    reporter = _RecordingReporter()

    with pytest.raises(RuntimeError, match=r"^baseline emitter exploded$"):
        runner.run_harbor_eval(
            skill,
            ["codex"],
            output_dir=tmp_path / "results",
            keep_harbor_jobs=True,
            progress_reporter=reporter,
        )

    transitions = [(event.stage, event.state) for event in reporter.events]
    assert ("with-skill-tasks", "ready") in transitions
    assert ("baseline-tasks", "running") in transitions
    assert ("baseline-tasks", "failed") in transitions
    assert transitions[-1] == ("run-finished", "failed")
    assert transitions.count(("run-finished", "failed")) == 1


def test_runner_terminalizes_collection_when_collection_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_collection(**_kwargs):
        raise RuntimeError("collection exploded")

    runner, skill = _stub_runner(monkeypatch, tmp_path, collect=fail_collection)
    reporter = _RecordingReporter()

    with pytest.raises(RuntimeError, match="collection exploded"):
        runner.run_harbor_eval(
            skill,
            ["codex"],
            output_dir=tmp_path / "results",
            keep_harbor_jobs=True,
            progress_reporter=reporter,
        )

    transitions = [(event.stage, event.state) for event in reporter.events]
    assert ("collection", "running") in transitions
    assert ("collection", "failed") in transitions
    assert transitions[-1] == ("run-finished", "failed")
    assert transitions.count(("run-finished", "failed")) == 1


def test_runner_marks_html_report_failure_degraded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_html(*_args, **_kwargs):
        raise RuntimeError("template rendering failed")

    runner, skill = _stub_runner(monkeypatch, tmp_path, html_report=fail_html)
    reporter = _RecordingReporter()

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        output_dir=tmp_path / "results",
        keep_harbor_jobs=True,
        progress_reporter=reporter,
    )

    report_states = [event.state for event in reporter.events if event.stage == "report"]
    assert report_states == ["running", "degraded"]
    finished = reporter.events[-1]
    assert (finished.stage, finished.state) == ("run-finished", "degraded")
    assert finished.result_path == result["result_path"]
    assert finished.report_path is None
    assert Path(finished.result_path).is_file()


def test_runner_terminalizes_report_when_result_write_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, skill = _stub_runner(monkeypatch, tmp_path)
    reporter = _RecordingReporter()
    original_write_text = Path.write_text

    def fail_result_write(path: Path, *args, **kwargs):
        if path.name == "result.json":
            raise OSError("result write failed")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_result_write)

    with pytest.raises(OSError, match="result write failed"):
        runner.run_harbor_eval(
            skill,
            ["codex"],
            output_dir=tmp_path / "results",
            keep_harbor_jobs=True,
            progress_reporter=reporter,
        )

    transitions = [(event.stage, event.state) for event in reporter.events]
    assert ("report", "running") in transitions
    assert ("report", "failed") in transitions
    assert transitions[-1] == ("run-finished", "failed")
    assert transitions.count(("run-finished", "failed")) == 1


def test_runner_continues_when_reporter_callbacks_raise(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, skill = _stub_runner(monkeypatch, tmp_path)

    class BrokenReporter:
        is_active = False

        def start(self, _plan) -> None:
            raise RuntimeError("presentation failed")

        def set_secret_values(self, _values) -> None:
            raise RuntimeError("presentation failed")

        def emit(self, _event) -> None:
            raise RuntimeError("presentation failed")

        def heartbeat(self) -> None:
            raise RuntimeError("presentation failed")

        def close(self) -> None:
            return None

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        output_dir=tmp_path / "results",
        keep_harbor_jobs=True,
        progress_reporter=BrokenReporter(),
    )

    assert result["execution_status"] == "succeeded"
    assert Path(result["result_path"]).is_file()


def test_runner_registers_exact_secrets_before_skill_and_model_render(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret = "literal-secret-model-value"
    runner, skill = _stub_runner(
        monkeypatch,
        tmp_path,
        skill_name=secret,
        provider_model=secret,
    )
    monkeypatch.setattr(runner, "_resolve_runtime_env", lambda _templates: ({"CUSTOM_VALUE": secret}, []))
    output = io.StringIO()
    reporter = _progress_module().PlainProgressReporter(stream=output, refresh_interval=60)

    runner.run_harbor_eval(
        skill,
        ["codex"],
        output_dir=tmp_path / "results",
        keep_harbor_jobs=True,
        progress_reporter=reporter,
    )
    reporter.close()

    assert secret not in output.getvalue()
    assert "<redacted>" in output.getvalue()


def test_harbor_failure_detail_redacts_runtime_secrets_without_streaming(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator.tier3.harbor import runner

    secret = "plain-runtime-secret"
    monkeypatch.setattr(runner, "build_harbor_run_command", lambda **_kwargs: ["harbor", "run"])
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["harbor", "run"],
            17,
            stdout=f"OPENAI_API_KEY={secret}\nraw agent output",
            stderr=f"token={secret}\nraw verifier output",
        ),
    )

    ok, detail = runner._run_harbor(
        dataset=tmp_path / "dataset",
        agent="codex",
        job_name="demo-codex-with",
        env_mode="docker",
        model="gpt-5",
        jobs_dir=tmp_path,
        run_env={"CUSTOM_RUNTIME_VALUE": secret},
        n_attempts=1,
        n_concurrent=1,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
    )

    assert ok is False
    assert secret not in detail
    assert "<redacted>" in detail


def test_command_does_not_relabel_runtime_failure_as_configuration_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator.tier3 import commands

    skill = tmp_path / "demo"
    skill.mkdir()
    reporter = _RecordingReporter()
    monkeypatch.setattr(commands, "resolve_llm_provider", lambda: SimpleNamespace(provider="openai"))

    def _failed_run(**kwargs):
        progress_reporter = kwargs["progress_reporter"]
        progress_reporter.emit(ProgressEvent(stage="configuration", state="ready"))
        progress_reporter.emit(ProgressEvent(stage="agent:codex", state="failed", detail="Harbor job failed"))
        return {"error": ["Harbor job failed"]}

    progress = _progress_module()
    ProgressEvent = progress.ProgressEvent
    monkeypatch.setattr(commands, "run_harbor_eval", _failed_run)

    result = commands.evaluate(
        skill,
        agents="codex",
        env_mode="docker",
        skip_baseline=True,
        n_attempts=None,
        pass_threshold=None,
        n_concurrent=None,
        max_agents=None,
        model=None,
        agent_model=(),
        custom_dockerfile_mode=None,
        skill_workspace_mode=None,
        include_skills=(),
        copy_repo=False,
        grading_mode=None,
        results_dir=tmp_path / "results",
        harbor_keep_jobs=False,
        timeout_multiplier=None,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        progress_reporter=reporter,
    )

    assert result == {"error": ["Harbor job failed"]}

    transitions = [(event.stage, event.state) for event in reporter.events]
    assert ("configuration", "ready") in transitions
    assert ("configuration", "failed") not in transitions


def test_command_protects_reporter_startup_and_still_runs_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator.tier3 import commands

    skill = tmp_path / "demo"
    skill.mkdir()
    engine_calls: list[dict[str, object]] = []

    class StartFailureReporter:
        is_active = False

        def __init__(self) -> None:
            self.close_calls = 0

        def start(self, _plan) -> None:
            raise RuntimeError("terminal startup failed")

        def set_secret_values(self, _values) -> None:
            return None

        def emit(self, _event) -> None:
            return None

        def heartbeat(self) -> None:
            return None

        def close(self) -> None:
            self.close_calls += 1

    reporter = StartFailureReporter()
    monkeypatch.setattr(commands, "resolve_llm_provider", lambda: SimpleNamespace(provider="openai"))
    monkeypatch.setattr(
        commands,
        "run_harbor_eval",
        lambda **kwargs: engine_calls.append(kwargs) or {"execution_status": "succeeded", "execution_errors": []},
    )

    result = commands.evaluate(
        skill,
        agents="codex",
        env_mode="docker",
        skip_baseline=True,
        n_attempts=None,
        pass_threshold=None,
        n_concurrent=None,
        max_agents=None,
        model=None,
        agent_model=(),
        custom_dockerfile_mode=None,
        skill_workspace_mode=None,
        include_skills=(),
        copy_repo=False,
        grading_mode=None,
        results_dir=tmp_path / "results",
        harbor_keep_jobs=False,
        timeout_multiplier=None,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        progress_reporter=reporter,
    )

    assert result["execution_status"] == "succeeded"
    assert len(engine_calls) == 1
    assert reporter.close_calls >= 1


def test_command_keeps_runner_finished_event_terminal_on_engine_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator.tier3 import commands

    skill = tmp_path / "demo"
    skill.mkdir()
    reporter = _RecordingReporter()
    progress = _progress_module()
    monkeypatch.setattr(commands, "resolve_llm_provider", lambda: SimpleNamespace(provider="openai"))

    def _failed_run(**kwargs):
        kwargs["progress_reporter"].emit(
            progress.ProgressEvent(stage="run-finished", state="failed", detail="worker failed")
        )
        raise RuntimeError("worker failed")

    monkeypatch.setattr(commands, "run_harbor_eval", _failed_run)

    with pytest.raises(RuntimeError, match="worker failed"):
        commands.evaluate(
            skill,
            agents="codex",
            env_mode="docker",
            skip_baseline=True,
            n_attempts=None,
            pass_threshold=None,
            n_concurrent=None,
            max_agents=None,
            model=None,
            agent_model=(),
            custom_dockerfile_mode=None,
            skill_workspace_mode=None,
            include_skills=(),
            copy_repo=False,
            grading_mode=None,
            results_dir=tmp_path / "results",
            harbor_keep_jobs=False,
            timeout_multiplier=None,
            override_cpus=None,
            override_memory_mb=None,
            override_storage_mb=None,
            progress_reporter=reporter,
        )

    assert (reporter.events[-1].stage, reporter.events[-1].state) == ("run-finished", "failed")


def test_command_runner_terminalizes_inherited_configuration_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator.tier3 import commands
    from skillevaluator.tier3.harbor import runner

    skill = tmp_path / "demo"
    skill.mkdir()
    reporter = _RecordingReporter()
    monkeypatch.setattr(commands, "resolve_llm_provider", lambda: SimpleNamespace(provider="openai"))

    def fail_runner_provider():
        raise RuntimeError("runner provider exploded")

    monkeypatch.setattr(runner, "resolve_llm_provider", fail_runner_provider)

    with pytest.raises(RuntimeError, match=r"^runner provider exploded$"):
        commands.evaluate(
            skill,
            agents="codex",
            env_mode="docker",
            skip_baseline=True,
            n_attempts=None,
            pass_threshold=None,
            n_concurrent=None,
            max_agents=None,
            model=None,
            agent_model=(),
            custom_dockerfile_mode=None,
            skill_workspace_mode=None,
            include_skills=(),
            copy_repo=False,
            grading_mode=None,
            results_dir=tmp_path / "results",
            harbor_keep_jobs=False,
            timeout_multiplier=None,
            override_cpus=None,
            override_memory_mb=None,
            override_storage_mb=None,
            progress_reporter=reporter,
        )

    transitions = [(event.stage, event.state) for event in reporter.events]
    assert transitions == [
        ("configuration", "running"),
        ("configuration", "failed"),
        ("run-finished", "failed"),
    ]
    assert transitions.count(("run-finished", "failed")) == 1
