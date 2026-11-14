# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 progress reporting is structured, safe, and terminal-aware."""

from __future__ import annotations

import errno
import importlib
import io
import json
import math
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest


@pytest.fixture(autouse=True)
def _rich_detects_interactive_terminals(monkeypatch: pytest.MonkeyPatch) -> None:
    # Rich reads TTY_INTERACTIVE when it builds a console, and
    # TTY_INTERACTIVE=0 turns off the in-place redraws that the tests with a
    # forced terminal drive. Clear it so they draw the same frames wherever
    # they run.
    monkeypatch.delenv("TTY_INTERACTIVE", raising=False)


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


class _RecordingStop:
    """Stop event that records the heartbeat wait and ends the loop at once."""

    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def wait(self, timeout: float) -> bool:
        self.timeouts.append(timeout)
        return True


def test_plain_heartbeat_defaults_to_one_minute_and_rich_to_one_second(monkeypatch: pytest.MonkeyPatch) -> None:
    progress = _progress_module()
    monkeypatch.setenv("TERM", "xterm-256color")
    plain = progress.create_progress_reporter("auto", stream=io.StringIO())
    rich = progress.create_progress_reporter("auto", stream=_TTYBuffer())

    assert type(plain) is progress.PlainProgressReporter
    assert type(rich) is progress.RichProgressReporter
    for reporter, expected in ((plain, 60.0), (rich, 1.0)):
        stop = _RecordingStop()
        reporter._stop = stop
        reporter._heartbeat_loop()
        assert stop.timeouts == [expected]


@pytest.mark.parametrize("mode", ["plain", "rich", "auto"])
def test_progress_interval_overrides_plain_and_rich_defaults(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    progress = _progress_module()
    monkeypatch.setenv("TERM", "xterm-256color")

    reporter = progress.create_progress_reporter(mode, stream=_TTYBuffer(), refresh_interval=12.5)

    assert reporter._refresh_interval == 12.5


@pytest.mark.parametrize("interval", [float("nan"), float("inf"), float("-inf")])
def test_progress_reporter_rejects_non_finite_refresh_interval(interval: float) -> None:
    progress = _progress_module()

    with pytest.raises(ValueError, match="finite number of seconds"):
        progress.create_progress_reporter("plain", stream=io.StringIO(), refresh_interval=interval)


def test_progress_reporter_bounds_refresh_interval_below_thread_timeout_limit() -> None:
    progress = _progress_module()

    fast = progress.PlainProgressReporter(stream=io.StringIO(), refresh_interval=0.0)
    slow = progress.PlainProgressReporter(stream=io.StringIO(), refresh_interval=1e12)

    assert fast._refresh_interval == progress.MIN_REFRESH_INTERVAL_SECONDS
    assert slow._refresh_interval == progress.MAX_REFRESH_INTERVAL_SECONDS
    assert progress.MAX_REFRESH_INTERVAL_SECONDS < threading.TIMEOUT_MAX


def test_plain_plan_labels_the_maximum_scaled_top_level_agent_timeout() -> None:
    progress = _progress_module()
    output = io.StringIO()
    reporter = progress.PlainProgressReporter(stream=output, refresh_interval=60)
    plan = _plan(progress)
    plan = progress.Tier3RunPlan(
        **{field: getattr(plan, field) for field in plan.__dataclass_fields__ if field != "task_timeout_seconds"},
        task_timeout_seconds=300.0,
    )

    reporter.start(plan)
    reporter.close()

    rendered = output.getvalue()
    assert "max-scaled-top-level-agent-timeout=300s" in rendered
    assert "task-timeout" not in rendered


def test_rich_plan_labels_the_maximum_scaled_top_level_agent_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import rich.live
    from rich.console import Console

    progress = _progress_module()
    _FakeLive.instances.clear()
    monkeypatch.setattr(rich.live, "Live", _FakeLive)
    plan = _plan(progress)
    plan = progress.Tier3RunPlan(
        **{field: getattr(plan, field) for field in plan.__dataclass_fields__ if field != "task_timeout_seconds"},
        task_timeout_seconds=300.0,
    )
    reporter = progress.RichProgressReporter(stream=_TTYBuffer(), refresh_interval=60)

    reporter.start(plan)
    rendered = io.StringIO()
    Console(file=rendered, force_terminal=False, width=180).print(_FakeLive.instances[0].renderables[-1])
    reporter.close()

    text = rendered.getvalue()
    assert "Max scaled top-level agent timeout" in text
    assert "Task timeout" not in text


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


def _render(renderable: Any, *, width: int = 80) -> str:
    from rich.console import Console

    stream = io.StringIO()
    Console(file=stream, width=width, color_system=None).print(renderable)
    return stream.getvalue()


def _rendered_live_lines(reporter: Any, *, width: int = 80) -> list[str]:
    return _render(reporter._build_live_table(), width=width).splitlines()


def test_rich_progress_frame_height_stays_fixed_as_plan_and_events_grow() -> None:
    progress = _progress_module()
    reporter = progress.RichProgressReporter(stream=io.StringIO(), refresh_interval=60)
    reporter._live_plan = progress.Tier3RunPlan(
        skill_name="demo",
        environment="docker",
        agents=("codex",),
        agent_models=(("codex", "example/model"),),
    )

    initial_lines = _rendered_live_lines(reporter)
    reporter._live_plan = progress.Tier3RunPlan(
        skill_name="demo",
        environment="docker",
        agents=("codex",),
        agent_models=(("codex", "example/model"),),
        provider="example-provider",
        task_count=2,
        case_count=1,
        attempts=1,
        baseline=True,
        concurrency=4,
        timeout_multiplier=2,
        matrix_trials=2,
        total_containers=2,
    )
    for index in range(10):
        reporter.emit(
            progress.ProgressEvent(
                stage=f"stage-{index}",
                state="complete",
                detail=f"completed stage {index} " + "with a detail too long for one line " * 3,
            )
        )
    populated_lines = _rendered_live_lines(reporter)

    assert len(populated_lines) == len(initial_lines)
    assert "example-provider" in "\n".join(populated_lines)
    assert "…" in "\n".join(populated_lines)


def test_rich_progress_title_does_not_change_frame_height_in_a_narrow_terminal() -> None:
    progress = _progress_module()
    reporter = progress.RichProgressReporter(stream=io.StringIO(), refresh_interval=60)
    reporter._live_plan = progress.Tier3RunPlan(
        skill_name="a-skill-with-a-very-long-name",
        environment="docker",
        agents=tuple(f"agent-{index}" for index in range(8)),
    )

    initial_height = len(_rendered_live_lines(reporter, width=20))
    reporter._live_plan = progress.Tier3RunPlan(
        skill_name="a-skill-with-a-very-long-name",
        environment="docker",
        agents=("agent-0",),
        provider="example-provider",
        task_count=2,
        case_count=1,
        attempts=1,
        baseline=True,
        concurrency=4,
        timeout_multiplier=2,
        total_containers=2,
    )
    for index in range(10):
        reporter.emit(
            progress.ProgressEvent(
                stage=f"stage-{index}",
                state="complete",
                detail=f"completed stage {index}",
            )
        )

    assert len(_rendered_live_lines(reporter, width=20)) == initial_height


def test_rich_progress_frame_does_not_shrink_when_the_agent_set_shrinks() -> None:
    progress = _progress_module()
    reporter = progress.RichProgressReporter(stream=io.StringIO(), refresh_interval=60)

    reporter._live_plan = progress.Tier3RunPlan(
        skill_name="demo",
        environment="docker",
        agents=tuple(f"agent-{index}" for index in range(8)),
    )
    initial_height = len(_rendered_live_lines(reporter))
    reporter._live_plan = progress.Tier3RunPlan(skill_name="demo", environment="docker", agents=("agent-0",))

    assert len(_rendered_live_lines(reporter)) == initial_height


def test_rich_progress_uses_manual_refresh_and_recent_event_window(monkeypatch: pytest.MonkeyPatch) -> None:
    import rich.live

    progress = _progress_module()
    _FakeLive.instances.clear()
    monkeypatch.setattr(rich.live, "Live", _FakeLive)
    reporter = progress.RichProgressReporter(stream=_TTYBuffer(), refresh_interval=60)
    reporter.start(_plan(progress))

    for index in range(7):
        reporter.emit(
            progress.ProgressEvent(
                stage=f"stage-{index}",
                state="complete",
                detail=f"completed stage {index}",
            )
        )
    reporter.emit(progress.ProgressEvent(stage="stage-0", state="running", detail="active again"))

    live = _FakeLive.instances[0]
    text = _render(live.renderables[-1])
    # Rich's own refresh thread would redraw outside the reporter lock.
    assert live.kwargs["auto_refresh"] is False
    assert "refresh_per_second" not in live.kwargs
    # The six most recently updated stages stay visible; the oldest drops out.
    assert "active again" in text
    assert all(f"completed stage {index}" in text for index in range(2, 7))
    assert "completed stage 1" not in text
    reporter.close()


def test_rich_progress_running_spinner_moves_on_with_each_redraw(monkeypatch: pytest.MonkeyPatch) -> None:
    import re

    import rich.live
    from rich.spinner import Spinner

    progress = _progress_module()
    _FakeLive.instances.clear()
    monkeypatch.setattr(rich.live, "Live", _FakeLive)
    reporter = progress.RichProgressReporter(stream=_TTYBuffer(), refresh_interval=60)
    reporter.start(_plan(progress))
    live = _FakeLive.instances[0]
    frames = Spinner("dots").frames

    def latest_frame() -> int:
        rendered = _render(live.renderables[-1], width=120)
        glyph = re.search(r"Agent codex\s*│ (\S) running", rendered)
        assert glyph is not None, rendered
        return frames.index(glyph.group(1))

    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="evaluating"))
    assert latest_frame() == 0
    # Rich no longer redraws on its own timer, so the spinner moves on one
    # frame with each heartbeat or update that redraws the view, however far
    # apart they come.
    reporter.heartbeat()
    assert latest_frame() == 1
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="attempt 2 of 3"))
    assert latest_frame() == 2
    for _ in range(len(frames)):
        reporter.heartbeat()
    assert latest_frame() == 2
    # A stage that runs again after it finished starts from the first frame.
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="complete", detail="attempt 2 of 3"))
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="attempt 3 of 3"))
    assert latest_frame() == 0
    reporter.close()


def test_rich_progress_resets_state_when_live_initialization_fails() -> None:
    progress = _progress_module()
    reporter = progress.RichProgressReporter(stream=io.StringIO(), refresh_interval=60)

    def _fail_live_initialization(*_args, **_kwargs):
        raise RuntimeError("terminal initialization failed")

    reporter._live_factory = _fail_live_initialization
    plan = progress.Tier3RunPlan(
        skill_name="demo",
        environment="docker",
        agents=tuple(f"agent-{index}" for index in range(8)),
    )
    with pytest.raises(RuntimeError, match="terminal initialization failed"):
        reporter.start(plan)
    reporter.close()

    assert reporter._live is None
    assert reporter._live_plan is None
    assert reporter._live_events == {}
    assert reporter._live_event_slots == progress._LIVE_MIN_EVENT_ROWS
    # A stale record of printed plan lines would hide the next run's plan.
    assert reporter._printed_plan_lines == {}


def test_rich_progress_prints_the_full_plan_above_the_fixed_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    import rich.live

    progress = _progress_module()
    _FakeLive.instances.clear()
    monkeypatch.setattr(rich.live, "Live", _FakeLive)
    monkeypatch.setenv("COLUMNS", "80")
    output = _TTYBuffer()
    reporter = progress.RichProgressReporter(stream=output, refresh_interval=60)
    agents = ("claude-code", "codex")
    options = {"attempts": 3, "baseline": True, "with_skill": True, "concurrency": 4, "max_agents": 2}
    reporter.start(
        progress.Tier3RunPlan(
            skill_name="pdf-processing",
            environment="docker",
            agents=agents,
            timeout_multiplier=1.5,
            **options,
        )
    )
    resolved = progress.Tier3RunPlan(
        skill_name="pdf-processing",
        environment="docker",
        agents=agents,
        agent_models=(("claude-code", "anthropic/claude-sonnet-4-5"), ("codex", "openai/gpt-5.1-codex")),
        provider="nv_build",
        task_count=3,
        case_count=12,
        timeout_multiplier=1.5,
        matrix_trials=72,
        preflight_trials=2,
        total_containers=74,
        task_timeout_seconds=1350,
        **options,
    )
    reporter.start(resolved)
    # An unchanged plan update prints nothing again.
    reporter.start(resolved)

    live = _FakeLive.instances[0]
    frame_heights = {len(_render(renderable).splitlines()) for renderable in live.renderables}
    printed = output.getvalue().splitlines()
    reporter.close()

    assert printed == [
        "Agents / models: claude-code, codex",
        "Run plan: Attempts 3 · With skill yes · Baseline yes · Concurrency 4 · Max agents 2 · Timeout 1.5x",
        "Agents / models: claude-code=anthropic/claude-sonnet-4-5, codex=openai/gpt-5.1-codex",
        "Provider: nv_build",
        "Run plan: Tasks 3 · Cases 12 · Attempts 3 · With skill yes · Baseline yes · Concurrency 4 · "
        "Max agents 2 · Timeout 1.5x · Matrix trials 72 · Preflight trials 2 · Containers 74 · "
        "Max scaled top-level agent timeout 1350s",
    ]
    # The frame below keeps one height at 80 columns while the plan resolves.
    assert len(frame_heights) == 1


def test_rich_progress_prints_colon_delimited_plan_values_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    import rich.live

    progress = _progress_module()
    _FakeLive.instances.clear()
    monkeypatch.setattr(rich.live, "Live", _FakeLive)
    output = _TTYBuffer()
    reporter = progress.RichProgressReporter(stream=output, refresh_interval=60)
    # Rich would read ":llama:" and ":rocket:" as emoji codes.
    reporter.start(
        progress.Tier3RunPlan(
            skill_name="demo",
            environment="docker",
            agents=("opencode",),
            agent_models=(("opencode", "ollama:llama:3.1-8b"),),
            provider="custom:rocket:proxy",
        )
    )
    printed = output.getvalue().splitlines()
    reporter.close()

    assert printed == [
        "Agents / models: opencode=ollama:llama:3.1-8b",
        "Provider: custom:rocket:proxy",
    ]


def test_rich_final_frame_marks_a_plan_that_never_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    import rich.live

    progress = _progress_module()
    _FakeLive.instances.clear()
    monkeypatch.setattr(rich.live, "Live", _FakeLive)
    reporter = progress.RichProgressReporter(stream=_TTYBuffer(), refresh_interval=60)
    # The CLI's first plan has no provider or counts; then configuration fails.
    reporter.start(progress.Tier3RunPlan(skill_name="demo", environment="docker", agents=("codex",)))
    reporter.emit(
        progress.ProgressEvent(
            stage="configuration",
            state="failed",
            detail="A public LLM provider is required for live evaluation",
        )
    )
    live = _FakeLive.instances[0]
    running_frame = _render(live.renderables[-1], width=180)
    reporter.close()
    final_frame = _render(live.renderables[-1], width=180)

    assert "resolving" in running_frame
    assert "preparing" in running_frame
    provider_row = next(line for line in final_frame.splitlines() if "Provider" in line)
    plan_row = next(line for line in final_frame.splitlines() if "Run plan" in line)
    assert "not resolved" in provider_row
    assert "not resolved" in plan_row
    assert not any(word in final_frame for word in ("pending", "resolving", "preparing"))
    assert "A public LLM provider is required for live evaluation" in final_frame
    assert live.stop_calls == 1


def test_rich_final_frame_lists_every_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    import rich.live

    progress = _progress_module()
    _FakeLive.instances.clear()
    monkeypatch.setattr(rich.live, "Live", _FakeLive)
    reporter = progress.RichProgressReporter(stream=_TTYBuffer(), refresh_interval=60)
    reporter.start(_plan(progress))
    reporter.emit(progress.ProgressEvent(stage="docker-images", state="skipped", detail="local environment selected"))
    for index in range(7):
        reporter.emit(
            progress.ProgressEvent(
                stage=f"agent:agent-{index}",
                state="complete",
                detail=f"agent {index} finished",
            )
        )
    reporter.emit(progress.ProgressEvent(stage="run-finished", state="complete", detail="Tier 3 evaluation finished"))
    live = _FakeLive.instances[0]
    running_frame = _render(live.renderables[-1], width=180)
    reporter.close()
    final_frame = _render(live.renderables[-1], width=180)

    assert "local environment selected" not in running_frame
    assert "agent 0 finished" not in running_frame
    # Once the run is over the frame stays in the scrollback, so it keeps every outcome.
    assert "local environment selected" in final_frame
    assert all(f"agent {index} finished" in final_frame for index in range(7))
    assert "Tier 3 evaluation finished" in final_frame
    assert live.stop_calls == 1


def _row_detail(frame: str, stage_label: str) -> str:
    """Return a row's Detail cell, joining the lines it wraps onto."""
    detail: list[str] = []
    for line in frame.splitlines():
        cells = [cell.strip() for cell in line.split("│")[1:-1]]
        if len(cells) != 4:
            continue
        # A wrapped row continues on lines whose Stage cell is empty.
        if cells[0] == stage_label or (detail and not cells[0]):
            detail.append(cells[2])
        elif detail:
            break
    return " ".join(part for part in detail if part)


def test_rich_final_frame_wraps_details_the_running_frame_cuts(monkeypatch: pytest.MonkeyPatch) -> None:
    import rich.live

    progress = _progress_module()
    _FakeLive.instances.clear()
    monkeypatch.setattr(rich.live, "Live", _FakeLive)
    reporter = progress.RichProgressReporter(stream=_TTYBuffer(), refresh_interval=60)
    reporter.start(_plan(progress))
    reason = "one or more Harbor jobs failed; inspect the retained artifacts and the trial logs"
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="failed", detail=reason))
    live = _FakeLive.instances[0]
    running_frame = _render(live.renderables[-1])
    reporter.close()
    final_frame = _render(live.renderables[-1])

    # At 80 columns the running frame cuts the reason to keep one line per row.
    assert _row_detail(running_frame, "Agent codex") != reason
    assert "…" in _row_detail(running_frame, "Agent codex")
    # The final frame no longer needs a fixed height, so it shows the reason in full.
    assert _row_detail(final_frame, "Agent codex") == reason


def test_rich_final_frame_keeps_rows_on_one_line_where_the_detail_column_is_narrow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rich.live

    progress = _progress_module()
    _FakeLive.instances.clear()
    monkeypatch.setattr(rich.live, "Live", _FakeLive)
    reporter = progress.RichProgressReporter(stream=_TTYBuffer(), refresh_interval=60)
    reporter.start(_plan(progress))
    reporter.emit(
        progress.ProgressEvent(
            stage="agent-runtime-preflight",
            state="skipped",
            detail="disabled by default; enable with --agent-runtime-preflight",
        )
    )
    reason = "one or more Harbor jobs failed; inspect the retained artifacts and the trial logs"
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="failed", detail=reason))
    live = _FakeLive.instances[0]
    reporter.close()
    final = live.renderables[-1]

    # Stage (23 characters for Agent Runtime Preflight), State and Elapsed leave
    # the Detail column 20 characters wide at 76 columns, so the details wrap.
    wide = _render(final, width=76)
    assert _row_detail(wide, "Agent codex") == reason
    assert max(len(line.split("│")[3]) - 2 for line in wide.splitlines() if line.count("│") == 5) == 20
    # One column narrower, folding every detail into a tall stack of short
    # pieces would leave the frame unreadable, so each row stays on one line.
    for width in (75, 60):
        narrow = _render(final, width=width)
        rows = [line for line in narrow.splitlines() if line.count("│") == 5]
        # The header, four plan rows and two stages.
        assert len(rows) == 7, (width, narrow)
        assert _row_detail(narrow, "Agent codex").endswith("…")
        assert all(len(line) == width for line in narrow.splitlines() if line.strip())


def _stopping_frame_reporter(progress: Any, output: io.StringIO, agents: tuple[str, ...], stopping_detail: str) -> Any:
    """Return a Rich reporter whose stage sequence matches a run stopped by Ctrl+C."""
    reporter = progress.RichProgressReporter(stream=output, refresh_interval=60)
    plan = progress.Tier3RunPlan(
        skill_name="pdf-processing",
        environment="docker",
        agents=agents,
        agent_models=tuple((agent, f"provider/{agent}-large-model") for agent in agents),
        provider="nv_build",
        task_count=3,
        case_count=12,
        attempts=3,
        baseline=True,
        with_skill=True,
        concurrency=4,
        max_agents=2,
        timeout_multiplier=1.5,
        matrix_trials=72,
        preflight_trials=0,
        total_containers=72,
        task_timeout_seconds=1350,
    )
    reporter.start(plan)
    for stage in (
        "configuration",
        "model-resolution",
        "environment-preflight",
        "credential-validation",
        "with-skill-tasks",
        "baseline-tasks",
        "case-selection",
        "docker-images",
    ):
        reporter.emit(progress.ProgressEvent(stage=stage, state="complete", detail="done"))
    # The longest built-in stage label stays in the recent-stage window.
    reporter.emit(
        progress.ProgressEvent(
            stage="agent-runtime-preflight",
            state="skipped",
            detail="disabled by default; enable with --agent-runtime-preflight",
        )
    )
    for agent in agents:
        reporter.emit(progress.ProgressEvent(stage=f"agent:{agent}", state="running", detail="with-skill + baseline"))
    for agent in agents:
        reporter.emit(progress.ProgressEvent(stage=f"agent:{agent}", state="running", detail=stopping_detail))
    return reporter


@pytest.mark.parametrize("agents", [("claude-code", "codex"), ("mini-swe-agent",)])
def test_rich_progress_at_80_columns_shows_the_ctrl_c_hint_and_the_full_plan(
    monkeypatch: pytest.MonkeyPatch,
    agents: tuple[str, ...],
) -> None:
    import rich.live

    from skillevaluator.tier3.harbor import runner

    progress = _progress_module()
    _FakeLive.instances.clear()
    monkeypatch.setattr(rich.live, "Live", _FakeLive)
    monkeypatch.setenv("COLUMNS", "80")
    stream = _TTYBuffer()
    reporter = _stopping_frame_reporter(progress, stream, agents, runner._HARBOR_RUN_STOPPING_DETAIL)
    output = stream.getvalue()
    table = _FakeLive.instances[0].renderables[-1]
    reporter.close()

    # The frame cuts each row to the terminal width; the second Ctrl+C hint
    # comes first in the detail and is short, so it is still readable. It
    # still fits two columns narrower: a legacy Windows console uses one
    # column less, and one is to spare.
    hint = runner._HARBOR_RUN_STOPPING_DETAIL.split(";")[0]
    assert "Ctrl+C" in hint
    for width in (80, 78):
        frame = _render(table, width=width)
        agent_rows = [line for line in frame.splitlines() if line.startswith("│ Agent ") and "running" in line]
        assert len(agent_rows) == len(agents)
        assert all(hint in row for row in agent_rows), (width, agent_rows)
    # Every plan value is printed in full above the frame.
    assert all(f"{agent}=provider/{agent}-large-model" in output for agent in agents)
    for value in (
        "Concurrency 4",
        "Max agents 2",
        "Timeout 1.5x",
        "Matrix trials 72",
        "Containers 72",
        "Max scaled top-level agent timeout 1350s",
    ):
        assert value in output


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
    # Only the full plan lines are printed above the box; no stage line is.
    plan_lines = (
        "Agents / models: codex=gpt-5, opencode=nvidia/llama\n"
        "Provider: openai\n"
        "Run plan: Tasks 2 · Cases 2 · Attempts 3 · Baseline yes · Concurrency 4 · Max agents 2 · Timeout 1.5x\n"
    )
    assert output.getvalue() == plan_lines

    reporter.emit(progress.ProgressEvent(stage="agent-runtime-preflight", state="running"))

    updated = io.StringIO()
    Console(file=updated, force_terminal=False, width=180).print(live.renderables[-1])
    assert "Agent Runtime Preflight" in updated.getvalue()
    assert "running" in updated.getvalue()
    assert output.getvalue() == plan_lines
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


def test_plain_reporter_emits_durable_plan_events_and_elapsed_heartbeat(monkeypatch: pytest.MonkeyPatch) -> None:
    progress = _progress_module()
    clock = [0.0]
    monkeypatch.setattr(progress, "monotonic", lambda: clock[0])
    output = io.StringIO()
    reporter = progress.PlainProgressReporter(stream=output, refresh_interval=60)
    # Drive heartbeat() ourselves; the real worker only waits for cleanup.
    monkeypatch.setattr(reporter, "_heartbeat_loop", reporter._stop.wait)

    try:
        reporter.start(_plan(progress))
        worker = reporter._thread
        reporter.emit(progress.ProgressEvent(stage="configuration", state="running"))
        reporter.emit(progress.ProgressEvent(stage="configuration", state="ready", detail="eval config loaded"))
        reporter.emit(
            progress.ProgressEvent(stage="docker-images", state="delegated", detail="build delegated to Harbor")
        )
        reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="with-skill + baseline"))
        clock[0] = math.nextafter(60.0, 0.0)
        reporter.heartbeat()
        assert "still running" not in output.getvalue()
        clock[0] = 60.0
        reporter.heartbeat()
        reporter.emit(progress.ProgressEvent(stage="agent:codex", state="complete"))
    finally:
        reporter.close()
    assert not worker.is_alive()
    assert not reporter.is_active

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
    assert "[00:01:00] still running: agent:codex" in rendered
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
    # A disabled reporter shows nothing, so starting it must not mark it active.
    assert reporter.is_active is False
    reporter.emit(progress.ProgressEvent(stage="configuration", state="running"))
    reporter.heartbeat()
    reporter.close()

    assert broken.close_calls >= 1
    reporter.start(_plan(progress))
    assert reporter.is_active is False
    reporter.close()
    assert broken.start_calls == 1


def test_safe_reporter_becomes_inactive_after_post_start_failure() -> None:
    progress = _progress_module()

    class FailingReporter:
        active = False

        @property
        def is_active(self) -> bool:
            return self.active

        def start(self, _plan) -> None:
            self.active = True

        def set_secret_values(self, _values) -> None:
            pass

        def emit(self, _event) -> None:
            raise RuntimeError("redraw failed")

        def heartbeat(self) -> None:
            pass

        def close(self) -> None:
            self.active = False

    reporter = progress.safe_progress_reporter(FailingReporter())
    reporter.start(_plan(progress))
    assert reporter.is_active is True

    reporter.emit(progress.ProgressEvent(stage="configuration", state="ready"))

    assert reporter.is_active is False
    reporter.close()


def test_safe_reporter_disables_after_heartbeat_redraw_failure() -> None:
    progress = _progress_module()
    heartbeat_failed = threading.Event()
    failed_on: list[threading.Thread] = []

    class FailingLive:
        def __init__(self, table, **_kwargs) -> None:
            self.renderables = [table]
            self.update_calls = 0
            self.stop_calls = 0

        def start(self, *, refresh: bool = False) -> None:
            assert refresh

        def update(self, table, *, refresh: bool = False) -> None:
            assert refresh
            self.renderables.append(table)
            self.update_calls += 1
            if self.update_calls == 2:
                failed_on.append(threading.current_thread())
                heartbeat_failed.set()
                raise RuntimeError("heartbeat redraw failed")

        def stop(self) -> None:
            self.stop_calls += 1

    output = io.StringIO()
    rich_reporter = progress.RichProgressReporter(stream=output, refresh_interval=0.05)
    rich_reporter._live_factory = FailingLive
    reporter = progress.safe_progress_reporter(rich_reporter)
    reporter.start(_plan(progress))
    heartbeat = rich_reporter._thread
    live = rich_reporter._live
    assert heartbeat is not None and live is not None
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="evaluating"))

    assert heartbeat_failed.wait(timeout=10)
    # The redraw sets the event before it raises, so the heartbeat thread may still
    # be turning the reporter off. That thread ends right after, so wait for it.
    heartbeat.join(timeout=10)
    assert not heartbeat.is_alive()
    assert failed_on == [heartbeat]
    assert reporter.is_active is False
    assert live.stop_calls == 1
    assert rich_reporter._live is None
    # Only the display failed: the agent is never drawn as failed or interrupted,
    # and a line under the frozen frame says the run goes on.
    frames = [_render(table, width=180) for table in live.renderables]
    assert not any("failed" in frame or "interrupted" in frame for frame in frames)
    assert "running" in frames[-1]
    assert "Tier 3 progress display stopped after a drawing error; the run continues." in output.getvalue()
    updates_before = live.update_calls
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="complete"))
    assert live.update_calls == updates_before
    reporter.close()


_STOP_NOTICE = "Tier 3 progress display stopped after a drawing error; the run continues."
_LATE_STOP_NOTICE = "Tier 3 progress display stopped after a drawing error; the run went on without it."


def _assert_cursor_shown_and_one_stop_notice(output: str, notice: str) -> None:
    # Live hid the cursor while it drew; it is shown again after the last hide.
    assert "\x1b[?25l" in output
    assert output.rfind("\x1b[?25h") > output.rfind("\x1b[?25l")
    assert output.count("Tier 3 progress display stopped after a drawing error") == 1
    # The notice is a line of its own below the last frame, even when the
    # terminal refused the stop that ends the frame's last line.
    visible = output.replace("\x1b[?25h", "").replace("\x1b[?25l", "")
    assert f"\n{notice}\n" in visible


@pytest.mark.parametrize(
    "refusal",
    ["once", "always", "every-thread-for-a-while", "notice-only"],
    ids=["refused-once", "refused-always", "every-thread-refused-for-a-while", "notice-refused"],
)
def test_heartbeat_write_that_would_block_turns_progress_off(
    monkeypatch: pytest.MonkeyPatch,
    refusal: str,
) -> None:
    from rich.console import Console

    progress = _progress_module()
    monkeypatch.setenv("TERM", "xterm-256color")
    uncaught: list[BaseException | None] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: uncaught.append(args.exc_value))
    refused = threading.Event()
    heartbeat_writes: list[int] = []
    # The heartbeat thread's first write is the refused redraw, its second
    # Live's stop and its third the notice under the last frame.
    refused_heartbeat_writes = {"once": {1}, "notice-only": {1, 3}}.get(refusal, set())

    class NonBlockingTerminal(_TTYBuffer):
        """A terminal another process made non-blocking: it refuses some writes."""

        # While set, the terminal's buffer is full and it refuses every
        # thread's writes.
        blocked = False

        def write(self, text: str) -> int:
            if self.blocked:
                refused.set()
                raise BlockingIOError(errno.EAGAIN, "write could not complete without blocking", 0)
            if threading.current_thread().name == "tier3-progress-heartbeat":
                heartbeat_writes.append(len(text))
                if refusal == "always" or len(heartbeat_writes) in refused_heartbeat_writes:
                    refused.set()
                    raise BlockingIOError(errno.EAGAIN, "write could not complete without blocking", 0)
            return super().write(text)

    stream = NonBlockingTerminal()
    rich_reporter = progress.RichProgressReporter(stream=stream, refresh_interval=0.05)
    # A real Rich Live, which keeps a refused frame in its buffer and sends it
    # again with the next one.
    rich_reporter._console = Console(
        file=stream,
        force_terminal=True,
        legacy_windows=False,
        width=120,
        color_system=None,
    )
    reporter = progress.safe_progress_reporter(rich_reporter)
    try:
        reporter.start(_plan(progress))
        heartbeat = rich_reporter._thread
        assert heartbeat is not None
        reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="evaluating"))
        stream.blocked = refusal == "every-thread-for-a-while"

        assert refused.wait(timeout=10)
        # A refused write turns progress off, so the heartbeat thread ends
        # instead of resending a growing backlog of frames.
        heartbeat.join(timeout=10)
        assert not heartbeat.is_alive()
        # The terminal takes writes again before the run ends.
        stream.blocked = False
        assert reporter.is_active is False
        assert rich_reporter._live is None
        reporter.emit(progress.ProgressEvent(stage="agent:codex", state="complete", detail="agent finished"))
        output = stream.getvalue()
        assert "agent finished" not in output
        # Only the display failed, so no stage is reported as failed.
        assert "failed" not in output
        assert "interrupted" not in output
        if refusal == "once":
            # The terminal took the stop, so the frame left behind and the line
            # under it say that the agent is still running.
            assert "running" in output
            assert _STOP_NOTICE in output
    finally:
        reporter.close()
    assert uncaught == []
    # A stop or notice the terminal refused on the heartbeat thread is lost
    # with that thread, so closing the run shows the cursor and the notice.
    _assert_cursor_shown_and_one_stop_notice(
        stream.getvalue(),
        _STOP_NOTICE if refusal == "once" else _LATE_STOP_NOTICE,
    )


@pytest.mark.parametrize(
    "refused_writes",
    [{1, 2}, {1, 3}],
    ids=["update-and-stop-refused", "update-and-notice-refused"],
)
def test_callers_write_that_would_block_is_finished_when_progress_closes(
    monkeypatch: pytest.MonkeyPatch,
    refused_writes: set[int],
) -> None:
    from rich.console import Console

    progress = _progress_module()
    monkeypatch.setenv("TERM", "xterm-256color")
    caller = threading.current_thread()
    caller_writes: list[int] = []

    class NonBlockingTerminal(_TTYBuffer):
        """Refuses the caller's update, then its stop of Live or the notice under the last frame."""

        armed = False

        def write(self, text: str) -> int:
            if self.armed and threading.current_thread() is caller:
                caller_writes.append(len(text))
                if len(caller_writes) in refused_writes:
                    raise BlockingIOError(errno.EAGAIN, "write could not complete without blocking", 0)
            return super().write(text)

    stream = NonBlockingTerminal()
    # No heartbeat during the test, so every write comes from this thread.
    rich_reporter = progress.RichProgressReporter(stream=stream, refresh_interval=60)
    rich_reporter._console = Console(
        file=stream,
        force_terminal=True,
        legacy_windows=False,
        width=120,
        color_system=None,
    )
    reporter = progress.safe_progress_reporter(rich_reporter)
    try:
        reporter.start(_plan(progress))
        reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="evaluating"))
        stream.armed = True
        reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="still evaluating"))
        assert reporter.is_active is False
        assert rich_reporter._live is None
        stream.armed = False
        # Once progress is off, other output (a logged warning, say) goes
        # straight to the terminal below the last frame.
        for number in range(1, 4):
            stream.write(f"warning {number} printed after the display stopped\n")
        printed_before_close = len(stream.getvalue())
    finally:
        reporter.close()
    output = stream.getvalue()
    assert "failed" not in output
    assert "interrupted" not in output
    closing_output = output[printed_before_close:]
    # Rich keeps this thread's refused writes and would send them with its
    # next one. They move the cursor up over the frame that was on screen
    # when drawing failed, so close() drops them instead of erasing the
    # warnings, and prints the late notice in place of a refused one.
    assert "\x1b[1A" not in closing_output
    assert "\x1b[2K" not in closing_output
    assert closing_output.endswith(f"{_LATE_STOP_NOTICE}\n")
    if 2 in refused_writes:
        assert "\x1b[?25h" in closing_output
    _assert_cursor_shown_and_one_stop_notice(output, _LATE_STOP_NOTICE)


def test_run_end_close_during_a_refused_heartbeat_stop_shows_the_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    from rich.console import Console

    progress = _progress_module()
    monkeypatch.setenv("TERM", "xterm-256color")
    uncaught: list[BaseException | None] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: uncaught.append(args.exc_value))
    lock = _InterleavingLock()
    closers: list[threading.Thread] = []

    def close_the_run() -> None:
        # The run ends while the heartbeat thread is still turning a failed
        # display off: its stop was refused and it has let go of the lock.
        closer = threading.Thread(target=reporter.close, name="run-end-close")
        closers.append(closer)
        closer.start()
        closer.join(timeout=10)

    class NonBlockingTerminal(_TTYBuffer):
        """Refuses every heartbeat-thread write while ``refusing`` is set."""

        refusing = True

        def write(self, text: str) -> int:
            if self.refusing and threading.current_thread().name == "tier3-progress-heartbeat":
                if "\x1b[?25h" in text:
                    # Live's stop, which shows the cursor again.
                    lock.arm(close_the_run)
                raise BlockingIOError(errno.EAGAIN, "write could not complete without blocking", 0)
            return super().write(text)

    stream = NonBlockingTerminal()
    rich_reporter = progress.RichProgressReporter(stream=stream, refresh_interval=0.05)
    rich_reporter._lock = lock
    rich_reporter._console = Console(
        file=stream,
        force_terminal=True,
        legacy_windows=False,
        width=120,
        color_system=None,
    )
    reporter = progress.safe_progress_reporter(rich_reporter)
    reporter.start(_plan(progress))
    heartbeat = rich_reporter._thread
    assert heartbeat is not None
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="evaluating"))

    heartbeat.join(timeout=10)
    assert not heartbeat.is_alive()
    assert len(closers) == 1
    assert not closers[0].is_alive()
    stream.refusing = False
    first_run = stream.getvalue()
    _assert_cursor_shown_and_one_stop_notice(first_run, _LATE_STOP_NOTICE)
    # Nothing of the failed run is left for the next one.
    reporter.start(_plan(progress))
    reporter.close()
    assert "display stopped" not in stream.getvalue()[len(first_run) :]
    assert uncaught == []


def test_a_new_run_without_close_carries_nothing_from_a_refused_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    from rich.console import Console

    progress = _progress_module()
    monkeypatch.setenv("TERM", "xterm-256color")

    class NonBlockingTerminal(_TTYBuffer):
        """Refuses every write while ``refusing`` is set."""

        refusing = False

        def write(self, text: str) -> int:
            if self.refusing:
                raise BlockingIOError(errno.EAGAIN, "write could not complete without blocking", 0)
            return super().write(text)

    stream = NonBlockingTerminal()
    rich_reporter = progress.RichProgressReporter(stream=stream, refresh_interval=60)
    rich_reporter._console = Console(
        file=stream,
        force_terminal=True,
        legacy_windows=False,
        width=120,
        color_system=None,
    )
    reporter = progress.safe_progress_reporter(rich_reporter)
    reporter.start(_plan(progress))
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="evaluating"))
    stream.refusing = True
    # The update and Live's stop are both refused.
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="still evaluating"))
    assert reporter.is_active is False
    stream.refusing = False
    first_run = stream.getvalue()

    # The reporter is used again without the close() that ends the failed run.
    rich_reporter.start(_plan(progress))
    rich_reporter.emit(progress.ProgressEvent(stage="agent:codex", state="complete", detail="agent finished"))
    rich_reporter.close()

    second_run = stream.getvalue()[len(first_run) :]
    assert "agent finished" in second_run
    assert "still evaluating" not in second_run
    assert "display stopped" not in second_run
    assert second_run.rfind("\x1b[?25h") > second_run.rfind("\x1b[?25l")


@pytest.mark.parametrize("notice_refused", [True, False], ids=["notice-refused", "notice-accepted"])
def test_refused_first_frame_and_stop_leave_the_cursor_shown(
    monkeypatch: pytest.MonkeyPatch,
    notice_refused: bool,
) -> None:
    from rich.console import Console
    from rich.live import Live

    progress = _progress_module()
    monkeypatch.setenv("TERM", "xterm-256color")

    class NonBlockingTerminal(_TTYBuffer):
        """Takes Live's hide-cursor write, then refuses the first frame and Live's own stop."""

        armed = False

        def write(self, text: str) -> int:
            refused = self.armed and "\x1b[?25l" not in text
            if refused and (notice_refused or _STOP_NOTICE not in text):
                raise BlockingIOError(errno.EAGAIN, "write could not complete without blocking", 0)
            return super().write(text)

    stream = NonBlockingTerminal()
    rich_reporter = progress.RichProgressReporter(stream=stream, refresh_interval=60)
    rich_reporter._console = Console(
        file=stream,
        force_terminal=True,
        legacy_windows=False,
        width=120,
        color_system=None,
    )

    def live_on_a_refusing_terminal(*args: Any, **kwargs: Any) -> Any:
        # The plan lines above the frame are already printed.
        stream.armed = True
        return Live(*args, **kwargs)

    rich_reporter._live_factory = live_on_a_refusing_terminal
    reporter = progress.safe_progress_reporter(rich_reporter)
    try:
        # Live.start() stops Live itself when its first frame fails, then
        # raises. The terminal refuses that stop as well, so the cursor Live
        # hid stays hidden, and Live is already stopped when progress turns off.
        reporter.start(_plan(progress))
        assert reporter.is_active is False
        assert rich_reporter._live is None
        stream.armed = False
        printed_before_close = len(stream.getvalue())
    finally:
        reporter.close()
    output = stream.getvalue()
    _assert_cursor_shown_and_one_stop_notice(output, _LATE_STOP_NOTICE if notice_refused else _STOP_NOTICE)
    # Nothing the terminal refused is sent when the run ends.
    assert "\x1b[1A" not in output[printed_before_close:]
    if notice_refused:
        # No frame reached the terminal, so the late notice follows the plan
        # lines without a blank line.
        assert "Harbor Run Configuration" not in output
        assert "\n\n" not in output.replace("\x1b[?25h", "").replace("\x1b[?25l", "")
    else:
        # The notice went out with the frame and the stop the terminal had
        # refused just before, so the frame is on screen above it.
        assert "Harbor Run Configuration" in output


@pytest.mark.parametrize(
    ("refused_writes", "notice"),
    [
        ({1}, _STOP_NOTICE),
        ({2}, _STOP_NOTICE),
        ({3}, _STOP_NOTICE),
        ({1, 2}, _LATE_STOP_NOTICE),
    ],
    ids=["agents-line-refused", "provider-line-refused", "run-plan-line-refused", "agents-line-and-notice-refused"],
)
def test_a_refused_plan_line_before_the_frame_turns_progress_off_with_a_notice(
    monkeypatch: pytest.MonkeyPatch,
    refused_writes: set[int],
    notice: str,
) -> None:
    from rich.console import Console

    progress = _progress_module()
    monkeypatch.setenv("TERM", "xterm-256color")
    writes: list[str] = []

    class NonBlockingTerminal(_TTYBuffer):
        """Refuses some of the first writes, which print the plan lines before Live starts."""

        def write(self, text: str) -> int:
            # Dropping the console's held output also writes an empty string,
            # which carries nothing to refuse.
            if text:
                writes.append(text)
                if len(writes) in refused_writes:
                    raise BlockingIOError(errno.EAGAIN, "write could not complete without blocking", 0)
            return super().write(text)

    stream = NonBlockingTerminal()
    rich_reporter = progress.RichProgressReporter(stream=stream, refresh_interval=60)
    rich_reporter._console = Console(
        file=stream,
        force_terminal=True,
        legacy_windows=False,
        width=120,
        color_system=None,
    )
    reporter = progress.safe_progress_reporter(rich_reporter)
    reporter.start(_plan(progress))
    # The plan line was refused before Live existed, so there is no frame to stop.
    assert reporter.is_active is False
    assert rich_reporter._live is None
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="evaluating"))
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="complete", detail="agent finished"))
    reporter.close()
    first_run = stream.getvalue()

    # Progress is off for the run, and a line of its own says so: on time when
    # the terminal took it, or when the run ends.
    assert "Harbor Run Configuration" not in first_run
    assert "agent finished" not in first_run
    assert first_run.count("Tier 3 progress display stopped after a drawing error") == 1
    assert f"\n{first_run}".endswith(f"\n{notice}\n")
    assert first_run.count("Agents / models:") <= 1

    # The refused plan line is not sent again with the next run's plan.
    reporter.start(_plan(progress))
    reporter.close()
    second_run = stream.getvalue()[len(first_run) :]
    assert second_run.count("Agents / models:") == 1
    assert "display stopped" not in second_run


@pytest.mark.parametrize(
    "refused_from",
    ["plan-line", "update"],
    ids=["refused-from-the-plan-line", "refused-from-an-update"],
)
def test_writes_refused_until_the_run_closes_are_not_sent_with_the_next_run(
    monkeypatch: pytest.MonkeyPatch,
    refused_from: str,
) -> None:
    from rich.console import Console

    progress = _progress_module()
    monkeypatch.setenv("TERM", "xterm-256color")

    class NonBlockingTerminal(_TTYBuffer):
        """Refuses every write while ``refusing`` is set, and keeps nothing it refused."""

        refusing = False

        def write(self, text: str) -> int:
            if self.refusing and text:
                raise BlockingIOError(errno.EAGAIN, "write could not complete without blocking", 0)
            return super().write(text)

    stream = NonBlockingTerminal()
    rich_reporter = progress.RichProgressReporter(stream=stream, refresh_interval=60)
    rich_reporter._console = Console(
        file=stream,
        force_terminal=True,
        legacy_windows=False,
        width=120,
        color_system=None,
    )
    reporter = progress.safe_progress_reporter(rich_reporter)
    stream.refusing = refused_from == "plan-line"
    reporter.start(_plan(progress))
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="evaluating"))
    stream.refusing = True
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="still evaluating"))
    assert reporter.is_active is False
    # The terminal still refuses what close() writes: the late notice, and
    # after a refused frame the write that shows the cursor again.
    reporter.close()
    stream.refusing = False
    first_run = stream.getvalue()

    reporter.start(_plan(progress))
    reporter.close()
    second_run = stream.getvalue()[len(first_run) :]
    # Nothing the terminal refused during the failed run comes out first.
    assert second_run.startswith("Agents / models:")
    assert second_run.count("Agents / models:") == 1
    assert "display stopped" not in second_run
    assert "still evaluating" not in second_run


@pytest.mark.parametrize("stage_state", ["complete", "running"], ids=["stage-finished", "stage-still-running"])
def test_a_final_frame_refused_when_the_run_closes_is_not_sent_with_the_next_run(
    monkeypatch: pytest.MonkeyPatch,
    stage_state: str,
) -> None:
    from rich.console import Console

    progress = _progress_module()
    monkeypatch.setenv("TERM", "xterm-256color")

    class NonBlockingTerminal(_TTYBuffer):
        """Refuses every write while ``refusing`` is set, and keeps nothing it refused."""

        refusing = False

        def write(self, text: str) -> int:
            if self.refusing and text:
                raise BlockingIOError(errno.EAGAIN, "write could not complete without blocking", 0)
            return super().write(text)

    stream = NonBlockingTerminal()
    rich_reporter = progress.RichProgressReporter(stream=stream, refresh_interval=60)
    rich_reporter._console = Console(
        file=stream,
        force_terminal=True,
        legacy_windows=False,
        width=120,
        color_system=None,
    )
    reporter = progress.safe_progress_reporter(rich_reporter)
    reporter.start(_plan(progress))
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="evaluating"))
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state=stage_state, detail="agent finished"))
    # Nothing failed while the run went on. The terminal refuses only what the
    # run's close() writes: the final frame and Live's stop.
    assert reporter.is_active is True
    stream.refusing = True
    reporter.close()
    stream.refusing = False
    # The caller prints its results below the frame.
    stream.write("Result: pass\n")
    stream.write("Report: report.html\n")
    first_run = stream.getvalue()

    reporter.start(_plan(progress))
    reporter.close()
    second_run = stream.getvalue()[len(first_run) :]
    # The refused final frame starts by moving the cursor up over the frame on
    # screen. Sent first in the next run, it would erase the result lines and
    # draw the old frame over them.
    assert second_run.startswith("Agents / models:")
    assert second_run.count("Agents / models:") == 1
    assert "agent finished" not in second_run
    assert "interrupted before completion" not in second_run


class _WouldBlockTerminal(io.RawIOBase):
    """A terminal file another process made non-blocking: some writes would block and take nothing."""

    def __init__(self, refused_writes: set[int]) -> None:
        super().__init__()
        self.refused_writes = refused_writes
        self.calls = 0
        self.received = bytearray()

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return True

    def write(self, data: Any) -> int | None:
        self.calls += 1
        if self.calls in self.refused_writes:
            # What a raw write returns for EAGAIN.
            return None
        self.received += bytes(data)
        return len(data)


@pytest.mark.parametrize(
    ("refused_writes", "plan_line", "notice"),
    [
        ({1}, "Agents / models:", _STOP_NOTICE),
        ({2}, "Provider:", _STOP_NOTICE),
        ({3}, "Run plan:", _STOP_NOTICE),
        # The terminal takes the refused line when it is flushed, then
        # refuses the notice. The stream keeps the notice and sends it when
        # the run ends, in place of the late one.
        ({1, 3}, "Agents / models:", _STOP_NOTICE),
        # The terminal stays full until start() returns: it refuses the plan
        # line and every flush after it, so the notice waits for close().
        ({1, 2, 3}, "Agents / models:", _LATE_STOP_NOTICE),
    ],
    ids=[
        "agents-line-refused",
        "provider-line-refused",
        "run-plan-line-refused",
        "agents-line-and-notice-refused",
        "terminal-full-during-start",
    ],
)
def test_a_plan_line_refused_by_a_buffered_terminal_stream_is_printed_once(
    monkeypatch: pytest.MonkeyPatch,
    refused_writes: set[int],
    plan_line: str,
    notice: str,
) -> None:
    from rich.console import Console

    progress = _progress_module()
    monkeypatch.setenv("TERM", "xterm-256color")
    terminal = _WouldBlockTerminal(refused_writes)
    # Shaped like the standard error stream the CLI passes. Its buffer keeps a
    # line the terminal refused and sends it with the next write.
    stream = io.TextIOWrapper(io.BufferedWriter(terminal), encoding="utf-8", newline="\n", line_buffering=True)
    rich_reporter = progress.RichProgressReporter(stream=stream, refresh_interval=60)
    rich_reporter._console = Console(
        file=stream,
        force_terminal=True,
        legacy_windows=False,
        width=120,
        color_system=None,
    )
    reporter = progress.safe_progress_reporter(rich_reporter)
    reporter.start(_plan(progress))
    assert reporter.is_active is False
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="evaluating"))
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="complete", detail="agent finished"))
    reporter.close()
    # The terminal takes whatever the stream still holds once the run is over.
    terminal.refused_writes = set()
    stream.flush()
    output = terminal.received.decode("utf-8")

    # The refused line reaches the terminal once, from the stream's buffer,
    # and one stop notice ends the run's output. The stream would also keep a
    # notice the terminal refused and send it before the late one, so the
    # notice is not written while the terminal still refuses the line, and a
    # notice the stream kept is not followed by the late one.
    assert output.count(plan_line) == 1
    assert output.count("Tier 3 progress display stopped after a drawing error") == 1
    assert f"\n{output}".endswith(f"\n{notice}\n")
    assert "Harbor Run Configuration" not in output
    assert "agent finished" not in output


@pytest.mark.parametrize(
    "full_at_close", [False, True], ids=["terminal-takes-writes-at-close", "terminal-full-at-close"]
)
def test_a_stop_notice_kept_by_a_buffered_terminal_stream_is_not_repeated_when_the_run_closes(
    monkeypatch: pytest.MonkeyPatch,
    full_at_close: bool,
) -> None:
    from rich.console import Console

    progress = _progress_module()
    monkeypatch.setenv("TERM", "xterm-256color")

    class Terminal(_WouldBlockTerminal):
        """Refuses the first write after it is armed, and the first write that carries the stop notice."""

        armed = False
        full = False
        notice_refused = False

        def write(self, data: Any) -> int | None:
            if self.full:
                return None
            if self.armed:
                self.armed = False
                return None
            if not self.notice_refused and _STOP_NOTICE.encode("utf-8") in bytes(data):
                self.notice_refused = True
                return None
            return super().write(data)

    terminal = Terminal(set())
    # Shaped like standard error: its buffer keeps what the terminal refused.
    stream = io.TextIOWrapper(io.BufferedWriter(terminal), encoding="utf-8", newline="\n", line_buffering=True)
    rich_reporter = progress.RichProgressReporter(stream=stream, refresh_interval=60)
    rich_reporter._console = Console(
        file=stream,
        force_terminal=True,
        legacy_windows=False,
        width=120,
        color_system=None,
    )
    reporter = progress.safe_progress_reporter(rich_reporter)
    reporter.start(_plan(progress))
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="evaluating"))
    # The terminal refuses the update and takes Live's stop, then refuses the
    # notice under the last frame. The stream keeps that notice.
    terminal.armed = True
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="still evaluating"))
    assert reporter.is_active is False
    assert terminal.notice_refused
    terminal.full = full_at_close
    reporter.close()
    # The terminal takes whatever the stream still holds once the run is over.
    terminal.full = False
    stream.flush()
    # How a refused frame splits across the stream's writes depends on the
    # Python version's buffer, so the bytes need not be valid UTF-8.
    output = terminal.received.decode("utf-8", errors="replace")

    # The kept notice reaches the terminal, when the run ends or with the
    # stream's next write, and no late notice repeats it.
    _assert_cursor_shown_and_one_stop_notice(output, _STOP_NOTICE)
    assert output.endswith(f"{_STOP_NOTICE}\n")


def test_a_stop_notice_refused_after_a_refused_first_frame_is_not_taken_as_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rich.console import Console
    from rich.live import Live

    progress = _progress_module()
    monkeypatch.setenv("TERM", "xterm-256color")

    class Terminal(_WouldBlockTerminal):
        """Refuses every write while ``refusing`` is set, and the first write that carries the stop notice."""

        refusing = False
        notice_refused = False

        def write(self, data: Any) -> int | None:
            if self.refusing:
                return None
            if not self.notice_refused and _STOP_NOTICE.encode("utf-8") in bytes(data):
                self.notice_refused = True
                return None
            return super().write(data)

    terminal = Terminal(set())
    # Shaped like standard error, with a buffer smaller than the first frame:
    # it keeps only the start of a refused frame, whatever the Python
    # version's default buffer size.
    stream = io.TextIOWrapper(
        io.BufferedWriter(terminal, buffer_size=1024), encoding="utf-8", newline="\n", line_buffering=True
    )

    class LiveOnAFullTerminal(Live):
        def start(self, refresh: bool = False) -> None:
            try:
                super().start(refresh=refresh)
            finally:
                # The terminal takes writes again once Live gave up.
                terminal.refusing = False

        def refresh(self) -> None:
            # The plan lines and Live's hide-cursor write went out. The
            # terminal refuses the first frame and the stop that Live.start()
            # runs when that frame fails, so Rich's console holds both.
            terminal.refusing = True
            super().refresh()

    rich_reporter = progress.RichProgressReporter(stream=stream, refresh_interval=60)
    rich_reporter._console = Console(
        file=stream,
        force_terminal=True,
        legacy_windows=False,
        width=120,
        color_system=None,
    )
    rich_reporter._live_factory = LiveOnAFullTerminal
    reporter = progress.safe_progress_reporter(rich_reporter)
    reporter.start(_plan(progress))
    # The notice went out after the held frame and stop, in one write larger
    # than the stream's buffer, and the terminal refused it. The stream kept
    # only the start of the frame.
    assert reporter.is_active is False
    assert terminal.notice_refused
    reporter.close()
    stream.flush()
    output = terminal.received.decode("utf-8", errors="replace")

    # The notice was lost with the rest of that write, so the late notice
    # takes its place, after the write that shows the cursor again.
    assert output.count("Tier 3 progress display stopped after a drawing error") == 1
    assert output.endswith(f"{_LATE_STOP_NOTICE}\n")
    assert output.rfind("\x1b[?25h") > output.rfind("\x1b[?25l")


@pytest.mark.parametrize("refused_from", ["first-frame", "update"], ids=["first-frame-refused", "update-refused"])
def test_late_notice_is_written_when_the_terminal_refuses_the_cursor_as_the_run_closes(
    monkeypatch: pytest.MonkeyPatch,
    refused_from: str,
) -> None:
    from rich.console import Console
    from rich.live import Live

    progress = _progress_module()
    monkeypatch.setenv("TERM", "xterm-256color")

    class Terminal(_WouldBlockTerminal):
        """Refuses every write while ``refusing`` is set."""

        refusing = False

        def write(self, data: Any) -> int | None:
            if self.refusing:
                return None
            return super().write(data)

    terminal = Terminal(set())
    # Shaped like standard error: its buffer keeps what the terminal refused.
    stream = io.TextIOWrapper(io.BufferedWriter(terminal), encoding="utf-8", newline="\n", line_buffering=True)

    class LiveOnAFillingTerminal(Live):
        def refresh(self) -> None:
            # The plan lines and Live's hide-cursor write went out; the
            # terminal fills up from the first frame on.
            if refused_from == "first-frame":
                terminal.refusing = True
            super().refresh()

    rich_reporter = progress.RichProgressReporter(stream=stream, refresh_interval=60)
    rich_reporter._console = Console(
        file=stream,
        force_terminal=True,
        legacy_windows=False,
        width=120,
        color_system=None,
    )
    rich_reporter._live_factory = LiveOnAFillingTerminal
    reporter = progress.safe_progress_reporter(rich_reporter)
    reporter.start(_plan(progress))
    if refused_from == "update":
        reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="evaluating"))
        terminal.refusing = True
        reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="still evaluating"))
    assert reporter.is_active is False
    # The terminal takes writes again while the run goes on, and a warning
    # printed then sends what the stream kept.
    terminal.refusing = False
    stream.write("warning printed after the display stopped\n")
    # It is full again when the run ends, so close() cannot show the cursor.
    terminal.refusing = True
    reporter.close()
    terminal.refusing = False
    stream.flush()
    # The stream's buffer can keep only the start of a refused frame and send
    # it with the next write, cut inside a box-drawing character. How much it
    # keeps depends on the Python version's buffer, so the bytes need not be
    # valid UTF-8.
    output = terminal.received.decode("utf-8", errors="replace")

    # The warning emptied the stream's buffer, so it keeps the late notice
    # with the refused show-cursor write and sends both once the terminal
    # takes writes again.
    _assert_cursor_shown_and_one_stop_notice(output, _LATE_STOP_NOTICE)
    assert output.endswith(f"{_LATE_STOP_NOTICE}\n")


def test_late_notice_starts_a_line_when_the_stream_sent_a_cut_refused_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rich.console import Console

    progress = _progress_module()
    monkeypatch.setenv("TERM", "xterm-256color")
    show_cursor = b"\x1b[?25h"

    class Terminal(_WouldBlockTerminal):
        """Refuses every write while ``refusing`` is set, and one show-cursor write on its own once armed."""

        refusing = False
        cursor_armed = False
        sent_before_cursor = b""

        def write(self, data: Any) -> int | None:
            if self.refusing:
                return None
            if self.cursor_armed and bytes(data) == show_cursor:
                self.cursor_armed = False
                self.sent_before_cursor = bytes(self.received)
                return None
            return super().write(data)

    terminal = Terminal(set())
    # Shaped like standard error, with a buffer smaller than a frame: it keeps
    # only the start of a refused frame, whatever the Python version's default
    # buffer size.
    stream = io.TextIOWrapper(
        io.BufferedWriter(terminal, buffer_size=1024), encoding="utf-8", newline="\n", line_buffering=True
    )
    rich_reporter = progress.RichProgressReporter(stream=stream, refresh_interval=60)
    rich_reporter._console = Console(
        file=stream,
        force_terminal=True,
        legacy_windows=False,
        width=120,
        color_system=None,
    )
    reporter = progress.safe_progress_reporter(rich_reporter)
    reporter.start(_plan(progress))
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="evaluating"))
    # The terminal refuses the update and Live's stop. The stream keeps the
    # start of the update frame and has no room left for the stop.
    terminal.refusing = True
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="still evaluating"))
    assert reporter.is_active is False
    # Nothing else is printed before the run ends. The terminal then takes
    # the part of the frame the stream kept and refuses the write that shows
    # the cursor again, so the line that ends the frame is not written.
    terminal.refusing = False
    terminal.cursor_armed = True
    reporter.close()
    stream.flush()
    assert terminal.cursor_armed is False
    # What reached the terminal before that write ends inside the frame.
    assert terminal.sent_before_cursor.endswith(b"\n") is False
    output = terminal.received.decode("utf-8", errors="replace")

    # The late notice starts a line of its own instead of joining the cut
    # frame's last line.
    _assert_cursor_shown_and_one_stop_notice(output, _LATE_STOP_NOTICE)
    assert output.endswith(f"\n{_LATE_STOP_NOTICE}\n")


def test_changelog_says_a_stream_can_still_send_refused_output_later() -> None:
    # close() drops only the view's own copy of what the terminal refused. A
    # stream shaped like standard error keeps its copy and sends it with its
    # next write, and no public API drops it. The entry is found by its
    # opening words so that a release heading above it can change.
    lead = (
        "- When the terminal refuses one of the plan lines printed before the "
        "interactive Tier 3 progress view's table first appears"
    )
    changelog = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text(encoding="utf-8")
    # Read each bullet with its indented continuation lines, so the entry is
    # found wherever it sits in its subsection.
    bullets: list[list[str]] = []
    for line in changelog.splitlines():
        if line.startswith("- "):
            bullets.append([line])
        elif bullets and line.startswith("  ") and line.strip():
            bullets[-1].append(line)
        elif not line.strip() and bullets and bullets[-1]:
            bullets.append([])
    blocks = [" ".join(" ".join(bullet).split()) for bullet in bullets if bullet]
    entries = [block for block in blocks if block.startswith(lead)]

    assert len(entries) == 1
    # One blank line, not two, separates the entry from what follows it.
    raw_entry_start = changelog.index("- When the terminal refuses one of the plan lines printed before the\n")
    raw_entry_end = changelog.index("\n\n", raw_entry_start)
    assert not changelog.startswith("\n\n\n", raw_entry_end)
    entry = entries[0]
    assert "no longer keeps its own copy of output the terminal still refuses when the run ends" in entry
    assert (
        "A stream that keeps what the terminal refused, as standard error does, can still send it later, "
        "and a final table sent then can still erase those lines." in entry
    )
    # A notice line the stream kept goes out when the run ends, in place of
    # the late one, as the buffered-stream tests above check.
    assert (
        "When the terminal takes the writes before that line and refuses only the line itself, a stream that "
        "keeps what the terminal refused, as standard error does, sends that line when the run ends in place of "
        "the later one." in entry
    )


def test_live_evaluation_docs_say_a_stream_can_still_send_refused_output_later() -> None:
    # The guide's description of the rich view states the same limit as the
    # CHANGELOG entry above, instead of saying refused frames can never be
    # drawn over later output. The paragraph is found by its opening words.
    # The view drops its copy when every run ends, so the sentence is not
    # tied to the drawing failure described just before it: after a failure
    # there is no final table.
    lead = "In a terminal, `--progress auto` uses the `rich` view"
    guide = (Path(__file__).resolve().parents[1] / "docs" / "tier3-live-evaluation.mdx").read_text(encoding="utf-8")
    blocks = [" ".join(block.split()) for block in guide.split("\n\n")]
    paragraphs = [block for block in blocks if block.startswith(lead)]

    assert len(paragraphs) == 1
    paragraph = paragraphs[0]
    assert "never drawn over" not in paragraph
    assert "The view then drops" not in paragraph
    assert (
        "When any run ends, whether or not drawing failed, the view drops its own copy of output the terminal "
        "still refuses, such as a refused final table, so it does not draw that output over lines printed since."
        in paragraph
    )
    assert (
        "A stream that keeps what the terminal refused, as standard error does, can still send it later, "
        "and a table sent then can still erase those lines." in paragraph
    )
    # Standard error keeps a refused notice line and sends it when the run
    # ends, so the late line does not follow it, except after a failed first
    # frame.
    assert (
        "If the terminal refused only that line, a stream that keeps what the terminal refused, as standard "
        "error does, sends that line then instead of the later one, although after a failed first frame both "
        "can appear." in paragraph
    )
    # The late line and the cursor write are kept only while the stream's
    # buffer has room: one already full of a refused frame takes neither, and
    # the terminal never gets them.
    assert (
        "If the terminal still refuses writes when the run ends, such a stream sends the later line and the "
        "write that shows the cursor once the terminal takes writes again, but only if its buffer has room for "
        "them. A buffer already full of a refused table can drop that line and leave the cursor hidden." in paragraph
    )


class _InterleavingLock:
    """A reentrant lock that runs a hook once, when the arming thread next releases it fully."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._depth = 0
        self._hook: tuple[threading.Thread, Any] | None = None

    def arm(self, hook: Any) -> None:
        self._hook = (threading.current_thread(), hook)

    def __enter__(self) -> _InterleavingLock:
        self._lock.acquire()
        self._depth += 1
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._depth -= 1
        released = self._depth == 0
        self._lock.release()
        armed = self._hook
        if released and armed is not None and armed[0] is threading.current_thread():
            self._hook = None
            armed[1]()


def test_an_update_that_races_a_drawing_failure_is_not_drawn_as_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    import rich.live

    progress = _progress_module()
    _FakeLive.instances.clear()
    monkeypatch.setattr(rich.live, "Live", _FakeLive)
    rich_reporter = progress.RichProgressReporter(stream=_TTYBuffer(), refresh_interval=60)
    lock = _InterleavingLock()
    rich_reporter._lock = lock
    reporter = progress.safe_progress_reporter(rich_reporter)
    reporter.start(_plan(progress))
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running", detail="attempt 1 of 3"))

    def late_agent_update() -> None:
        # The caller's next agent update passed the wrapper's check before the
        # failure turned progress off, and reaches the reporter while it stops.
        update = threading.Thread(
            target=rich_reporter.emit,
            args=(progress.ProgressEvent(stage="agent:codex", state="running", detail="attempt 2 of 3"),),
        )
        update.start()
        update.join(timeout=10)

    lock.arm(late_agent_update)
    # What the heartbeat thread's failure callback runs.
    reporter._disable()
    live = _FakeLive.instances[0]
    frames = [_render(renderable, width=180) for renderable in live.renderables]
    reporter.close()

    # Only the display failed: the agent is never drawn as failed, and the
    # frame left on screen shows the update as still running.
    assert not any("failed" in frame or "interrupted" in frame for frame in frames)
    assert "attempt 2 of 3" in frames[-1]
    assert live.stop_calls == 1


def test_heartbeat_failure_during_a_plan_update_stops_what_the_update_drew(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rich.live

    progress = _progress_module()
    _FakeLive.instances.clear()
    monkeypatch.setattr(rich.live, "Live", _FakeLive)

    class RacingReporter(progress.RichProgressReporter):
        report_failure: Any = None
        failure_thread: threading.Thread | None = None
        race = False

        def set_failure_callback(self, callback) -> None:
            super().set_failure_callback(callback)
            self.report_failure = callback

        def start(self, plan) -> None:
            if self.race:
                # The heartbeat thread reports a redraw failure while this plan
                # update is in flight.
                self.failure_thread = threading.Thread(target=self.report_failure)
                self.failure_thread.start()
                self.failure_thread.join(timeout=0.2)
            super().start(plan)

    rich_reporter = RacingReporter(stream=_TTYBuffer(), refresh_interval=60)
    reporter = progress.safe_progress_reporter(rich_reporter)
    reporter.start(_plan(progress))
    rich_reporter.race = True
    reporter.start(_plan(progress))
    failure_thread = rich_reporter.failure_thread
    assert failure_thread is not None
    failure_thread.join(timeout=10)

    assert not failure_thread.is_alive()
    # The failure waited for the update, then stopped the frame it drew.
    assert reporter.is_active is False
    assert rich_reporter.is_active is False
    assert [live.stop_calls for live in _FakeLive.instances] == [1]
    reporter.close()


def test_safe_reporter_wraps_a_reporter_whose_callback_registration_fails() -> None:
    progress = _progress_module()

    class RefusingReporter(progress.NullProgressReporter):
        def __init__(self) -> None:
            self.events: list[Any] = []

        def set_failure_callback(self, _callback) -> None:
            raise RuntimeError("failure callbacks are not supported")

        def emit(self, event) -> None:
            self.events.append(event)

    class StrictProxy:
        """Forwards the reporter protocol and refuses any other attribute."""

        def __init__(self) -> None:
            self.target = RefusingReporter()

        def __getattr__(self, name: str) -> Any:
            if name in {"is_active", "start", "set_secret_values", "emit", "heartbeat", "close"}:
                return getattr(self.target, name)
            raise RuntimeError(f"unsupported attribute: {name}")

    refusing = RefusingReporter()
    proxy = StrictProxy()

    for target, events in ((refusing, refusing.events), (proxy, proxy.target.events)):
        reporter = progress.safe_progress_reporter(target)
        reporter.start(_plan(progress))
        reporter.emit(progress.ProgressEvent(stage="configuration", state="running"))
        reporter.close()
        assert [event.stage for event in events] == ["configuration"]


def test_heartbeat_failure_without_a_safety_wrapper_stops_the_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    progress = _progress_module()
    uncaught: list[BaseException | None] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: uncaught.append(args.exc_value))

    class FailingHeartbeatReporter(progress.PlainProgressReporter):
        def heartbeat(self) -> None:
            raise RuntimeError("heartbeat failed")

    reporter = FailingHeartbeatReporter(stream=io.StringIO(), refresh_interval=0.05)
    reporter.start(_plan(progress))
    reporter.emit(progress.ProgressEvent(stage="agent:codex", state="running"))
    heartbeat = reporter._thread
    assert heartbeat is not None

    heartbeat.join(timeout=10)

    assert not heartbeat.is_alive()
    assert reporter.is_active is False
    assert uncaught == []
    reporter.close()


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


@pytest.mark.parametrize(
    "proxy_uri",
    [
        "http://proxy-user:proxy-password@proxy.example:8080",
        "https://proxy-user:password@fragment@proxy.example:8443",
        "https://bearer-token@proxy.example",
        "socks5://user%3Apassword@proxy.example:1080",
    ],
)
def test_progress_redacts_proxy_uri_userinfo_even_before_secret_values_are_registered(proxy_uri: str) -> None:
    progress = _progress_module()

    extracted = progress.secret_values_from_environment({"HTTP_PROXY": proxy_uri})
    rendered = progress.redact_progress_detail(f"Cannot connect to proxy {proxy_uri}", secret_values=set())

    assert proxy_uri in extracted
    assert "proxy-user" not in rendered
    assert "proxy-password" not in rendered
    assert "password@fragment" not in rendered
    assert "bearer-token" not in rendered
    assert "user%3Apassword" not in rendered
    assert "<redacted>@proxy.example" in rendered


@pytest.mark.parametrize(
    ("proxy_uri", "diagnostic", "secrets"),
    [
        (
            "http://proxy-user:proxy-password@proxy.example:8080",
            "proxy auth failed for proxy-user with proxy-password",
            ("proxy-user", "proxy-password"),
        ),
        (
            "http://proxy%2Duser:proxy%2Dpassword@proxy.example:8080",
            "proxy auth failed for proxy-user with proxy-password",
            ("proxy-user", "proxy-password"),
        ),
        (
            "proxy-user:proxy-password@proxy.example:8080",
            "proxy auth failed for proxy-user with proxy-password",
            ("proxy-user", "proxy-password"),
        ),
    ],
)
def test_progress_registers_proxy_userinfo_components(
    proxy_uri: str,
    diagnostic: str,
    secrets: tuple[str, ...],
) -> None:
    progress = _progress_module()

    extracted = progress.secret_values_from_environment({"HTTPS_PROXY": proxy_uri})
    rendered = progress.redact_progress_detail(diagnostic, secret_values=extracted)

    assert all(secret in extracted for secret in secrets)
    assert all(secret not in rendered for secret in secrets)


@pytest.mark.parametrize("short_secret", ["x", "pw", "usr"])
def test_progress_redacts_short_exact_credential_fragments_as_standalone_tokens(short_secret: str) -> None:
    progress = _progress_module()
    rendered = progress.redact_progress_detail(
        f"proxy rejected credential {short_secret}; password remains a normal word",
        secret_values={short_secret},
    )

    assert f" {short_secret};" not in rendered
    assert "<redacted>" in rendered
    assert "password remains" in rendered


def test_progress_detail_strips_osc_title_and_hyperlink_payloads() -> None:
    progress = _progress_module()

    rendered = progress.redact_progress_detail(
        "safe\x1b]0;forged title\x07 text \x1b]8;;https://malicious.example\x1b\\click\x1b]8;;\x1b\\ done"
    )

    assert "forged title" not in rendered
    assert "malicious.example" not in rendered
    assert "\x1b" not in rendered
    assert rendered == "safe text click done"


@pytest.mark.parametrize(
    ("raw_text", "expected_text"),
    [
        (
            "Zero Skill Execution in gcd-001-explicit-auth: The agent solved the task without invoking the skill.",
            "Zero Skill Execution in gcd-001-explicit-auth: The agent solved the task without invoking the skill.",
        ),
        (
            "3. Align Expected Skill for gcd-001-explicit-auth: Update the expected_skill attribute.",
            "3. Align Expected Skill for gcd-001-explicit-auth: Update the expected_skill attribute.",
        ),
        (
            "claude-code with-skill oauth-token: Trial completed with warnings",
            "claude-code with-skill oauth-token: Trial completed with warnings",
        ),
        (
            "auth: super-secret-value",
            "auth: <redacted>",
        ),
        (
            "--api-key: super-secret-value",
            "--api-key: <redacted>",
        ),
        (
            "gcd-001-explicit-auth=super-secret-value",
            "gcd-001-explicit-auth=<redacted>",
        ),
        (
            "ANTHROPIC_AUTH_TOKEN=super-secret-value",
            "ANTHROPIC_AUTH_TOKEN=<redacted>",
        ),
    ],
)
def test_redact_progress_detail_preserves_hyphenated_identifiers_while_redacting_assignments(
    raw_text: str,
    expected_text: str,
) -> None:
    """Preserve prose after hyphenated case IDs while redacting colon and equals secret assignments."""
    progress = _progress_module()
    assert progress.redact_progress_detail(raw_text) == expected_text


def test_progress_redaction_scales_linearly_for_large_non_uri_text() -> None:
    progress = _progress_module()
    payload = "x" * (1024 * 1024)

    rendered = progress.redact_progress_detail(payload)

    assert rendered == payload


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
    model_probe=None,
    task_emitter=None,
    skill_name: str = "demo-skill",
    provider_name: str = "openai",
    provider_model: str = "gpt-5",
    provider_base_url: str | None = None,
):
    from skillevaluator.tier3.harbor import runner

    skill = tmp_path / skill_name
    (skill / "evals").mkdir(parents=True)
    evals_file = skill / "evals" / "evals.json"
    evals_file.write_text("{}", encoding="utf-8")
    resolved_base_url = provider_base_url
    if resolved_base_url is None and provider_name == "openai":
        resolved_base_url = "https://api.openai.com/v1"
    provider = SimpleNamespace(
        provider=provider_name,
        model=provider_model,
        api_key="exact-provider-secret",
        base_url=resolved_base_url,
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
    monkeypatch.setattr(runner, "_resolve_runtime_env", lambda _templates, **_kwargs: ({}, []))
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
        "probe_model",
        model_probe
        or (
            lambda selected_provider: SimpleNamespace(
                ok=True,
                provider=selected_provider.provider,
                model=selected_provider.model,
                detail=f"model {selected_provider.model} is available",
                failure_kind=None,
                http_status=None,
            )
        ),
    )

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


def _dataset_sufficiency_agent_result(
    *,
    score: float = 0.8,
    paired_cases: int = 13,
    skill_activation: dict[str, Any] | None = None,
    skip_with_skill: bool = False,
) -> dict[str, Any]:
    from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS

    scores = dict.fromkeys(DEFAULT_METRICS, score)
    result = {
        "execution_status": "succeeded",
        "execution_errors": [],
        "metrics_with_skill": list(DEFAULT_METRICS),
        "metrics_without_skill": list(DEFAULT_METRICS),
        "with_skill": {} if skip_with_skill else scores,
        "without_skill": scores,
        "conditions": {
            "with_skill": {"execution_status": "skipped" if skip_with_skill else "succeeded"},
            "without_skill": {"execution_status": "succeeded"},
        },
        "pass_at_k": {
            "with_skill": {"rate_interval": {"lower": 0.45, "upper": 0.55}},
            "without_skill": {"rate": 0.8},
            "lift": {"paired_comparison": {"paired_cases": paired_cases}},
        },
    }
    if skill_activation is not None:
        result["skill_activation"] = skill_activation
    return result


def test_runner_rejects_custom_in_skill_results_root_before_creating_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, skill = _stub_runner(monkeypatch, tmp_path)
    results_root = skill / "custom-results"

    result = runner.run_harbor_eval(skill, ["codex"], output_dir=results_root)

    assert result.get("error")
    assert "canonical" in json.dumps(result)
    assert "evals/results" in json.dumps(result)
    assert not results_root.exists()


def test_root_dataset_sufficiency_uses_best_agent_selected_by_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def collect(**_kwargs):
        return {
            "execution_status": "succeeded",
            "execution_errors": [],
            "metrics": [],
            "agents": {
                "codex": _dataset_sufficiency_agent_result(score=0.7, paired_cases=1),
                "opencode": _dataset_sufficiency_agent_result(score=0.9, paired_cases=7),
            },
        }

    runner, skill = _stub_runner(monkeypatch, tmp_path, collect=collect)

    result = runner.run_harbor_eval(
        skill,
        ["codex", "opencode"],
        output_dir=tmp_path / "results",
        agent_runtime_preflight=False,
    )

    assert "error" not in result
    assert result["dataset_sufficiency"] == result["agents"]["opencode"]["dataset_sufficiency"]
    assert result["dataset_sufficiency"]["checks"][2]["observed"]["paired_cases"] == 7


def test_single_never_activated_agent_gets_root_dataset_sufficiency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, skill = _stub_runner(
        monkeypatch,
        tmp_path,
        collect=lambda **_kwargs: {
            "execution_status": "succeeded",
            "execution_errors": [],
            "metrics": [],
            "agents": {
                "codex": _dataset_sufficiency_agent_result(
                    skill_activation={"status": "never_activated"},
                )
            },
        },
    )

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        output_dir=tmp_path / "results",
        agent_runtime_preflight=False,
    )

    from skillevaluator.tier3.result_display import render_result

    assert "dataset_sufficiency" in result
    assert "Dataset size:" in render_result(result)


def test_skipped_with_skill_agent_gets_root_dataset_sufficiency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, skill = _stub_runner(
        monkeypatch,
        tmp_path,
        collect=lambda **_kwargs: {
            "execution_status": "succeeded",
            "execution_errors": [],
            "metrics": [],
            "agents": {"codex": _dataset_sufficiency_agent_result(skip_with_skill=True)},
        },
    )

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        skip_with_skill=True,
        output_dir=tmp_path / "results",
        agent_runtime_preflight=False,
    )

    from skillevaluator.tier3.result_display import render_result

    assert result["agents"]["codex"]["conditions"]["with_skill"]["execution_status"] == "skipped"
    assert "dataset_sufficiency" in result
    assert "Dataset size:" in render_result(result)


def test_skipped_with_skill_agents_rank_root_dataset_sufficiency_by_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, skill = _stub_runner(
        monkeypatch,
        tmp_path,
        collect=lambda **_kwargs: {
            "execution_status": "succeeded",
            "execution_errors": [],
            "metrics": [],
            "agents": {
                "codex": _dataset_sufficiency_agent_result(score=0.6, paired_cases=1, skip_with_skill=True),
                "opencode": _dataset_sufficiency_agent_result(score=0.9, paired_cases=7, skip_with_skill=True),
            },
        },
    )

    result = runner.run_harbor_eval(
        skill,
        ["codex", "opencode"],
        skip_with_skill=True,
        output_dir=tmp_path / "results",
        agent_runtime_preflight=False,
    )

    from skillevaluator.tier3.result_display import render_result

    agents = result["agents"]
    assert agents["codex"]["dataset_sufficiency"] != agents["opencode"]["dataset_sufficiency"]
    assert result["dataset_sufficiency"] == agents["opencode"]["dataset_sufficiency"]
    assert "Dataset size:" in render_result(result)


def test_runner_keeps_recorded_negative_count_when_unclassified_cases_remain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entries = [{"id": f"positive-{index}", "prompt": "Question", "expected_skill": "demo"} for index in range(10)]
    entries.extend({"id": f"negative-{index}", "prompt": "Question", "expected_skill": None} for index in range(2))
    entries.append({"id": "unclassified", "prompt": "Question"})
    runner, skill = _stub_runner(
        monkeypatch,
        tmp_path,
        collect=lambda **_kwargs: {
            "execution_status": "succeeded",
            "execution_errors": [],
            "metrics": [],
            "agents": {"codex": _dataset_sufficiency_agent_result()},
        },
    )
    from skillevaluator.tier3.harbor.report_data import build_dataset_snapshot

    dataset_truth = build_dataset_snapshot(entries, evaluator_version=runner.__version__)
    monkeypatch.setattr(runner, "_persist_dataset_truth", lambda *_args, **_kwargs: dataset_truth)

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        output_dir=tmp_path / "results",
        agent_runtime_preflight=False,
    )

    from skillevaluator.tier3.result_display import render_result

    summary = result["dataset_sufficiency"]["coverage_summary"]
    assert result["dataset_summary"] == {
        "total_tasks": 13,
        "positive_tasks": 10,
        "negative_tasks": 2,
        "unclassified_tasks": 1,
        "source": "dataset",
    }
    assert summary.startswith("Dataset size: sufficient")
    assert "2 negative cases (needs 1)" in summary
    assert summary in render_result(result)


def test_run_result_bounds_multibyte_launch_errors_for_report_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    github_token = "ghp_" + ("A" * 36)
    launch_errors = [f"launch {index}: {github_token}\x1b[2J" + ("😀" * 2_048) for index in range(256)]
    runner, skill = _stub_runner(
        monkeypatch,
        tmp_path,
        run_agent=lambda **_kwargs: launch_errors,
    )

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        output_dir=tmp_path / "results",
        agent_runtime_preflight=False,
    )

    result_path = Path(result["result_path"])
    serialized = result_path.read_bytes()
    persisted = json.loads(serialized)
    assert len(serialized) <= 2 * 1024 * 1024
    assert persisted["execution_status"] == "failed"
    assert persisted["execution_error_details_total"] == len(launch_errors)
    assert persisted["execution_error_details_shown"] == len(persisted["execution_errors"])
    assert persisted["execution_error_details_truncated"] is True
    assert persisted["error"] == persisted["execution_errors"][:1]
    assert github_token not in serialized.decode("utf-8")
    assert "\\u001b" not in serialized.decode("utf-8")

    from skillevaluator.tier3.harbor import report_data

    diagnostics: list[dict[str, Any]] = []
    assert report_data._load_bounded_json(result_path, diagnostics, artifact="result") == persisted
    assert diagnostics == []


def _configure_native_task_source(
    monkeypatch: pytest.MonkeyPatch,
    runner,
    skill: Path,
    *,
    grading_mode: str = "default",
) -> None:
    native_task = skill / "evals" / "harbor" / "case-1"
    native_task.mkdir(parents=True)
    (native_task / "instruction.md").write_text("Run the native case.\n", encoding="utf-8")
    (native_task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-1"\n\n'
        '[metadata]\nentry_id = "case-1"\n\n[verifier.env]\n'
        'LLM_JUDGE_MODEL = "authored-native-judge"\n',
        encoding="utf-8",
    )
    (skill / "evals" / "config.yml").write_text(
        f"schema_version: 1\nharbor:\n  task_source: native_harbor\ngrading:\n  mode: {grading_mode}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "load_evals_config",
        lambda _path: (
            {
                "harbor": {"task_source": "native_harbor"},
                "grading": {"mode": grading_mode},
            },
            None,
        ),
    )

    def emit_native(_skill, output: Path, **_kwargs):
        output.mkdir(parents=True, exist_ok=True)
        return [output / "case-1"]

    monkeypatch.setattr(runner, "stage_native_harbor_tasks", emit_native)


def test_ack_eval_preflight_uses_the_exact_prospective_bedrock_child_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator.provider_config import ProviderConfig
    from skillevaluator.tier3.harbor import runner as runner_module

    real_harbor_subprocess_environment = runner_module._harbor_subprocess_environment
    real_provider_environment = runner_module._provider_environment
    captured: dict[str, Any] = {}
    launched_env: dict[str, str] = {}

    def capture_preflight(**kwargs: Any) -> list[str]:
        captured.update(kwargs)
        return []

    def capture_launch(**kwargs: Any) -> list[str]:
        launched_env.update(kwargs["run_env"])
        return []

    runner, skill = _stub_runner(
        monkeypatch,
        tmp_path,
        environment_check=capture_preflight,
        run_agent=capture_launch,
        provider_name="bedrock",
        provider_model="us.anthropic.claude-test",
    )
    provider = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-test",
        api_key=None,
        base_url=None,
        litellm_model="bedrock/us.anthropic.claude-test",
        region="us-west-2",
    )
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "eks-exec-auth-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "eks-exec-auth-secret")
    monkeypatch.setenv("KUBECONFIG", "/config/eks")
    monkeypatch.setenv("SKILL_EVAL_JUDGE_MODEL", "judge-model")
    monkeypatch.setenv("ALIBABA_CLOUD_ACCESS_KEY_ID", "ambient-parent-only")
    monkeypatch.setattr(runner, "resolve_llm_provider", lambda: provider)
    monkeypatch.setattr(runner, "_provider_environment", real_provider_environment)
    monkeypatch.setattr(runner, "_harbor_subprocess_environment", real_harbor_subprocess_environment)
    monkeypatch.setattr(
        runner,
        "_resolve_runtime_env",
        lambda _templates, **_kwargs: ({"SAFE_RUNTIME_FLAG": "enabled"}, []),
    )

    result = runner.run_harbor_eval(
        skill,
        ["claude-code"],
        env_mode="ack",
        environment_kwargs={"namespace": "skill-evals"},
        output_dir=tmp_path / "results",
        agent_runtime_preflight=False,
    )

    assert result["execution_status"] == "succeeded"
    child_env = captured["subprocess_env"]
    assert isinstance(child_env, dict)
    assert child_env == launched_env
    assert child_env["KUBECONFIG"] == "/config/eks"
    assert child_env["AWS_ACCESS_KEY_ID"] == "eks-exec-auth-key"
    assert child_env["AWS_SECRET_ACCESS_KEY"] == "eks-exec-auth-secret"
    assert child_env["AWS_REGION"] == "us-west-2"
    assert child_env["CLAUDE_CODE_USE_BEDROCK"] == "1"
    assert child_env["SAFE_RUNTIME_FLAG"] == "enabled"
    assert child_env["LLM_JUDGE_MODEL"] == "judge-model"
    assert child_env["SKILL_EVAL_JUDGE_MODEL"] == "judge-model"
    assert "ALIBABA_CLOUD_ACCESS_KEY_ID" not in child_env


@pytest.mark.parametrize("agent_runtime_preflight", [True, False])
def test_credential_validation_401_stops_before_image_and_task_preparation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent_runtime_preflight: bool,
) -> None:
    secret = "exact-provider-secret"
    probe_calls: list[Any] = []
    image_calls: list[Path] = []
    task_calls: list[Path] = []

    def reject_credential(selected_provider):
        probe_calls.append(selected_provider)
        return SimpleNamespace(
            ok=False,
            provider=selected_provider.provider,
            model=selected_provider.model,
            detail=f"model catalog returned HTTP 401; rejected {secret}",
            failure_kind="authentication",
            http_status=401,
        )

    def build_image(skill_path: Path, *_args, **_kwargs) -> str:
        image_calls.append(skill_path)
        return "skillevaluator-test-image"

    def emit_tasks(_skill, output: Path, **_kwargs):
        task_calls.append(output)
        output.mkdir(parents=True, exist_ok=True)
        return [output / "case-1"]

    runner, skill = _stub_runner(
        monkeypatch,
        tmp_path,
        model_probe=reject_credential,
        task_emitter=emit_tasks,
    )
    monkeypatch.setattr(
        runner,
        "load_evals_config",
        lambda _path: (
            {
                "harbor": {
                    "task_source": "evals_json",
                    "base_image_mode": "reuse",
                    "n_attempts": 1,
                    "n_concurrent": 1,
                    "max_agents": 1,
                }
            },
            None,
        ),
    )
    monkeypatch.setattr(runner, "build_eval_base_image", build_image)
    reporter = _RecordingReporter()

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        output_dir=tmp_path / "results",
        agent_runtime_preflight=agent_runtime_preflight,
        progress_reporter=reporter,
    )

    assert probe_calls and probe_calls[0].provider == "openai"
    assert result.get("error")
    assert "HTTP 401" in json.dumps(result)
    assert image_calls == []
    assert task_calls == []
    transitions = [(event.stage, event.state) for event in reporter.events]
    assert ("credential-validation", "failed") in transitions
    assert ("credential-validation", "complete") not in transitions
    rendered = json.dumps(result) + json.dumps([(event.stage, event.state, event.detail) for event in reporter.events])
    assert secret not in rendered


@pytest.mark.parametrize(
    ("provider_name", "provider_base_url", "failure_kind", "http_status", "detail"),
    [
        ("openai", "https://api.openai.com/v1", "unavailable", None, "model catalog request timed out"),
        ("openai", "https://api.openai.com/v1", "authorization", 403, "model catalog returned HTTP 403"),
        ("openai", "https://api.openai.com/v1", None, None, "model requested-model is not listed"),
        (
            "openai-compatible",
            "https://gateway.example/v1",
            "authentication",
            401,
            "model catalog returned HTTP 401",
        ),
        ("openai", "https://gateway.example/v1", "authentication", 401, "model catalog returned HTTP 401"),
        ("anthropic", "https://gateway.example/v1", "authentication", 401, "model catalog returned HTTP 401"),
    ],
)
@pytest.mark.parametrize("agent_runtime_preflight", [True, False])
def test_inconclusive_credential_probe_degrades_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider_name: str,
    provider_base_url: str,
    failure_kind: str | None,
    http_status: int | None,
    detail: str,
    agent_runtime_preflight: bool,
) -> None:
    probe_calls: list[Any] = []
    task_calls: list[Path] = []

    def inconclusive(selected_provider):
        probe_calls.append(selected_provider)
        return SimpleNamespace(
            ok=False,
            provider=selected_provider.provider,
            model=selected_provider.model,
            detail=detail,
            failure_kind=failure_kind,
            http_status=http_status,
        )

    def emit_tasks(_skill, output: Path, **_kwargs):
        task_calls.append(output)
        output.mkdir(parents=True, exist_ok=True)
        return [output / "case-1"]

    runner, skill = _stub_runner(
        monkeypatch,
        tmp_path,
        model_probe=inconclusive,
        task_emitter=emit_tasks,
        provider_name=provider_name,
        provider_model="requested-model",
        provider_base_url=provider_base_url,
    )
    reporter = _RecordingReporter()

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        output_dir=tmp_path / "results",
        agent_runtime_preflight=agent_runtime_preflight,
        progress_reporter=reporter,
    )

    assert "error" not in result
    # Gateway harness defaults are distinct from the evaluator/judge model.
    assert len(probe_calls) == (2 if provider_name in {"anthropic", "openai-compatible"} else 1)
    assert any(call.provider == provider_name for call in probe_calls)
    assert task_calls
    transitions = [(event.stage, event.state) for event in reporter.events]
    assert ("credential-validation", "degraded") in transitions
    assert ("credential-validation", "complete") not in transitions
    assert transitions[-1] == ("run-finished", "complete")
    validation = result["run_config"]["credential_validation"]
    assert validation["status"] == "degraded"
    assert {target["status"] for target in validation["targets"]} == {"degraded"}
    persisted = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert persisted["run_config"]["credential_validation"] == validation


def test_credential_validation_probes_an_identical_agent_route_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probe_calls: list[Any] = []

    def succeed(selected_provider):
        probe_calls.append(selected_provider)
        return SimpleNamespace(
            ok=True,
            provider=selected_provider.provider,
            model=selected_provider.model,
            detail=f"model {selected_provider.model} is available",
            failure_kind=None,
            http_status=None,
        )

    runner, skill = _stub_runner(monkeypatch, tmp_path, model_probe=succeed)

    result = runner.run_harbor_eval(
        skill,
        ["codex", "opencode"],
        output_dir=tmp_path / "results",
        agent_runtime_preflight=False,
    )

    assert "error" not in result
    assert len(probe_calls) == 1
    assert probe_calls[0].provider == "openai"


def test_credential_validation_probes_each_distinct_agent_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probe_calls: list[Any] = []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-runtime-secret")

    def succeed(selected_provider):
        probe_calls.append(selected_provider)
        return SimpleNamespace(
            ok=True,
            provider=selected_provider.provider,
            model=selected_provider.model,
            detail=f"model {selected_provider.model} is available",
            failure_kind=None,
            http_status=None,
        )

    runner, skill = _stub_runner(monkeypatch, tmp_path, model_probe=succeed)

    result = runner.run_harbor_eval(
        skill,
        ["codex", "claude-code"],
        agent_models={"claude-code": "anthropic/claude-test"},
        output_dir=tmp_path / "results",
        agent_runtime_preflight=False,
    )

    assert "error" not in result
    assert {probe.provider for probe in probe_calls} == {"openai", "anthropic"}


def test_credential_validation_includes_distinct_standard_grading_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probe_calls: list[Any] = []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-runtime-secret")

    def succeed(selected_provider):
        probe_calls.append(selected_provider)
        return SimpleNamespace(
            ok=True,
            provider=selected_provider.provider,
            model=selected_provider.model,
            detail=f"model {selected_provider.model} is available",
            failure_kind=None,
            http_status=None,
        )

    runner, skill = _stub_runner(monkeypatch, tmp_path, model_probe=succeed)
    monkeypatch.setattr(
        runner,
        "_provider_environment",
        lambda _provider: {
            "OPENAI_API_KEY": "exact-provider-secret",
            "LLM_JUDGE_MODEL": "judge-gpt",
        },
    )

    result = runner.run_harbor_eval(
        skill,
        ["claude-code"],
        agent_models={"claude-code": "anthropic/claude-test"},
        output_dir=tmp_path / "results",
        agent_runtime_preflight=False,
    )

    assert "error" not in result
    assert {(probe.provider, probe.model) for probe in probe_calls} == {
        ("anthropic", "claude-test"),
        ("openai", "judge-gpt"),
    }


def test_native_task_source_omits_evals_dataset_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, skill = _stub_runner(monkeypatch, tmp_path)
    (skill / "evals" / "evals.json").write_text(
        json.dumps(
            {
                "skill_name": "demo-skill",
                "generation": {"mode": "template_requested", "review_state": "needs_review"},
                "evals": [{"id": "case-1", "prompt": "Q", "expected_output": "A"}],
            }
        ),
        encoding="utf-8",
    )
    _configure_native_task_source(monkeypatch, runner, skill)

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        output_dir=tmp_path / "results",
        agent_runtime_preflight=False,
    )

    assert "error" not in result
    assert result["run_config"]["task_source"] == "native_harbor"
    assert result["dataset_provenance"] is None


def test_missing_native_task_source_stops_before_credential_probes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probe_calls: list[Any] = []

    def record_probe(selected_provider):
        probe_calls.append(selected_provider)
        raise AssertionError("credential probes must not run without the selected task source")

    runner, skill = _stub_runner(monkeypatch, tmp_path, model_probe=record_probe)
    monkeypatch.setattr(
        runner,
        "load_evals_config",
        lambda _path: ({"harbor": {"task_source": "native_harbor"}}, None),
    )
    reporter = _RecordingReporter()

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        output_dir=tmp_path / "results",
        agent_runtime_preflight=False,
        progress_reporter=reporter,
    )

    assert result == {"error": ["No native Harbor task source found at evals/harbor."]}
    assert probe_calls == []
    transitions = [(event.stage, event.state) for event in reporter.events]
    assert ("environment-preflight", "complete") in transitions
    assert ("with-skill-tasks", "failed") in transitions
    assert not any(stage == "credential-validation" for stage, _state in transitions)


@pytest.mark.parametrize("grading_mode", ["default", "default_plus_custom"])
def test_native_standard_grading_without_host_override_marks_judge_probe_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    grading_mode: str,
) -> None:
    probe_calls: list[Any] = []
    provider_fallback = "unused-provider-fallback"

    def probe(selected_provider):
        probe_calls.append(selected_provider)
        if selected_provider.model == provider_fallback:
            return SimpleNamespace(
                ok=False,
                provider=selected_provider.provider,
                model=selected_provider.model,
                detail="model catalog returned HTTP 401",
                failure_kind="authentication",
                http_status=401,
            )
        return SimpleNamespace(
            ok=True,
            provider=selected_provider.provider,
            model=selected_provider.model,
            detail=f"model {selected_provider.model} is available",
            failure_kind=None,
            http_status=None,
        )

    runner, skill = _stub_runner(
        monkeypatch,
        tmp_path,
        model_probe=probe,
        provider_model=provider_fallback,
    )
    _configure_native_task_source(monkeypatch, runner, skill, grading_mode=grading_mode)
    reporter = _RecordingReporter()

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        agent_models={"codex": "actual-agent-model"},
        output_dir=tmp_path / "results",
        agent_runtime_preflight=False,
        progress_reporter=reporter,
    )

    assert "error" not in result
    assert [(call.provider, call.model) for call in probe_calls] == [("openai", "actual-agent-model")]
    assert ("credential-validation", "degraded") in [(event.stage, event.state) for event in reporter.events]
    assert result["run_config"]["task_source"] == "native_harbor"
    assert result["run_config"]["judge"]["catalog_verification"] == "inconclusive"
    assert result["run_config"]["judge"]["effective_model_source"] == "native_harbor_runtime"
    assert result["run_config"]["credential_validation"]["status"] == "degraded"
    judge_target = next(
        target
        for target in result["run_config"]["credential_validation"]["targets"]
        if target["labels"] == ["standard grader"]
    )
    assert judge_target == {
        "labels": ["standard grader"],
        "provider": "openai",
        "model": None,
        "fallback_model": provider_fallback,
        "status": "inconclusive",
        "detail": (
            "native Harbor resolves the effective judge model at runtime; "
            "task or step verifier env may supersede the configured fallback"
        ),
    }


def test_auto_with_both_task_sources_uses_evals_json_and_probes_the_standard_judge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probe_calls: list[Any] = []

    def succeed(selected_provider):
        probe_calls.append(selected_provider)
        return SimpleNamespace(
            ok=True,
            provider=selected_provider.provider,
            model=selected_provider.model,
            detail=f"model {selected_provider.model} is available",
            failure_kind=None,
            http_status=None,
        )

    runner, skill = _stub_runner(
        monkeypatch,
        tmp_path,
        model_probe=succeed,
        provider_model="standard-judge-model",
    )
    native_task = skill / "evals" / "harbor" / "case-1"
    native_task.mkdir(parents=True)
    (native_task / "instruction.md").write_text("Unused native case.\n", encoding="utf-8")
    (native_task / "task.toml").write_text(
        'schema_version = "1.3"\n[task]\nname = "nvidia/case-1"\n[metadata]\nentry_id = "case-1"\n',
        encoding="utf-8",
    )
    (skill / "evals" / "config.yml").write_text(
        "schema_version: 1\nharbor:\n  task_source: auto\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "load_evals_config",
        lambda _path: ({"harbor": {"task_source": "auto"}}, None),
    )

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        agent_models={"codex": "actual-agent-model"},
        output_dir=tmp_path / "results",
        agent_runtime_preflight=False,
    )

    assert "error" not in result
    assert result["run_config"]["task_source"] == "evals_json"
    assert {(call.provider, call.model) for call in probe_calls} == {
        ("openai", "actual-agent-model"),
        ("openai", "standard-judge-model"),
    }
    assert result["run_config"]["judge"]["catalog_verification"] == "verified"


def test_auto_with_native_only_custom_grading_skips_the_standard_judge_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probe_calls: list[Any] = []

    def succeed(selected_provider):
        probe_calls.append(selected_provider)
        return SimpleNamespace(
            ok=True,
            provider=selected_provider.provider,
            model=selected_provider.model,
            detail=f"model {selected_provider.model} is available",
            failure_kind=None,
            http_status=None,
        )

    runner, skill = _stub_runner(monkeypatch, tmp_path, model_probe=succeed)
    (skill / "evals" / "evals.json").unlink()
    native_task = skill / "evals" / "harbor" / "case-1"
    native_task.mkdir(parents=True)
    (native_task / "instruction.md").write_text("Run the native case.\n", encoding="utf-8")
    (native_task / "task.toml").write_text(
        'schema_version = "1.3"\n[task]\nname = "nvidia/case-1"\n[metadata]\nentry_id = "case-1"\n',
        encoding="utf-8",
    )
    (skill / "evals" / "config.yml").write_text(
        "schema_version: 1\nharbor:\n  task_source: auto\ngrading:\n  mode: custom_only\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "load_evals_config",
        lambda _path: (
            {"harbor": {"task_source": "auto"}, "grading": {"mode": "custom_only"}},
            None,
        ),
    )
    monkeypatch.setattr(runner, "find_evals_file", lambda _path: None)

    def emit_native(_skill, output: Path, **_kwargs):
        output.mkdir(parents=True, exist_ok=True)
        return [output / "case-1"]

    monkeypatch.setattr(runner, "stage_native_harbor_tasks", emit_native)

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        output_dir=tmp_path / "results",
        agent_runtime_preflight=False,
    )

    assert "error" not in result
    assert result["run_config"]["task_source"] == "native_harbor"
    assert result["run_config"]["judge"] == {"enabled": False}
    assert len(probe_calls) == 1


@pytest.mark.parametrize("grading_mode", ["default", "default_plus_custom"])
@pytest.mark.parametrize("override_name", ["LLM_JUDGE_MODEL", "SKILL_EVAL_JUDGE_MODEL"])
def test_native_standard_grading_host_override_is_the_exact_judge_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    grading_mode: str,
    override_name: str,
) -> None:
    probe_calls: list[Any] = []

    def succeed(selected_provider):
        probe_calls.append(selected_provider)
        return SimpleNamespace(
            ok=True,
            provider=selected_provider.provider,
            model=selected_provider.model,
            detail=f"model {selected_provider.model} is available",
            failure_kind=None,
            http_status=None,
        )

    runner, skill = _stub_runner(
        monkeypatch,
        tmp_path,
        model_probe=succeed,
        provider_model="provider-fallback",
    )
    _configure_native_task_source(monkeypatch, runner, skill, grading_mode=grading_mode)
    monkeypatch.setattr(
        runner,
        "_provider_environment",
        lambda _provider: {
            "OPENAI_API_KEY": "exact-provider-secret",
            override_name: "host-judge-model",
        },
    )

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        agent_models={"codex": "actual-agent-model"},
        output_dir=tmp_path / "results",
        agent_runtime_preflight=False,
    )

    assert "error" not in result
    assert {(call.provider, call.model) for call in probe_calls} == {
        ("openai", "actual-agent-model"),
        ("openai", "host-judge-model"),
    }
    assert result["run_config"]["judge"] == {
        "enabled": True,
        "provider": "openai",
        "model": "host-judge-model",
        "source": override_name,
        "override_applied": True,
        "reasoning_effort": None,
        "catalog_verification": "verified",
    }


def test_native_inconclusive_judge_probe_does_not_skip_the_same_route_used_by_an_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probe_calls: list[Any] = []
    staged: list[Path] = []

    def reject_agent_credential(selected_provider):
        probe_calls.append(selected_provider)
        return SimpleNamespace(
            ok=False,
            provider=selected_provider.provider,
            model=selected_provider.model,
            detail="model catalog returned HTTP 401",
            failure_kind="authentication",
            http_status=401,
        )

    def emit_native(_skill, output: Path, **_kwargs):
        staged.append(output)
        return []

    runner, skill = _stub_runner(
        monkeypatch,
        tmp_path,
        model_probe=reject_agent_credential,
        provider_model="shared-route-model",
    )
    _configure_native_task_source(monkeypatch, runner, skill)
    monkeypatch.setattr(runner, "stage_native_harbor_tasks", emit_native)

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        agent_models={"codex": "shared-route-model"},
        output_dir=tmp_path / "results",
        agent_runtime_preflight=False,
    )

    assert result.get("error")
    assert [(call.provider, call.model) for call in probe_calls] == [("openai", "shared-route-model")]
    assert staged == []


def test_distinct_standard_grading_probe_failure_is_redacted_and_stops_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evaluator_secret = "evaluator-only-provider-secret"
    task_calls: list[Path] = []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-runtime-secret")

    def probe(selected_provider):
        if selected_provider.provider == "openai":
            return SimpleNamespace(
                ok=False,
                provider=selected_provider.provider,
                model=selected_provider.model,
                detail=f"model catalog returned HTTP 401; rejected {evaluator_secret}",
                failure_kind="authentication",
                http_status=401,
            )
        return SimpleNamespace(
            ok=True,
            provider=selected_provider.provider,
            model=selected_provider.model,
            detail=f"model {selected_provider.model} is available",
            failure_kind=None,
            http_status=None,
        )

    def emit_tasks(_skill, output: Path, **_kwargs):
        task_calls.append(output)
        return []

    runner, skill = _stub_runner(
        monkeypatch,
        tmp_path,
        model_probe=probe,
        task_emitter=emit_tasks,
    )
    monkeypatch.setattr(
        runner,
        "_provider_environment",
        lambda _provider: {"OPENAI_API_KEY": evaluator_secret},
    )
    monkeypatch.setattr(
        runner,
        "resolve_llm_provider",
        lambda: SimpleNamespace(
            provider="openai",
            model="gpt-5",
            api_key=evaluator_secret,
            base_url="https://api.openai.com/v1",
        ),
    )
    reporter = _RecordingReporter()

    result = runner.run_harbor_eval(
        skill,
        ["claude-code"],
        agent_models={"claude-code": "anthropic/claude-test"},
        output_dir=tmp_path / "results",
        agent_runtime_preflight=False,
        progress_reporter=reporter,
    )

    rendered = json.dumps(result) + json.dumps([event.detail for event in reporter.events])
    assert result.get("error")
    assert "HTTP 401" in rendered
    assert evaluator_secret not in rendered
    assert task_calls == []


def test_custom_only_credential_validation_skips_unused_evaluator_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probe_calls: list[Any] = []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-runtime-secret")

    def succeed(selected_provider):
        probe_calls.append(selected_provider)
        return SimpleNamespace(
            ok=True,
            provider=selected_provider.provider,
            model=selected_provider.model,
            detail=f"model {selected_provider.model} is available",
            failure_kind=None,
            http_status=None,
        )

    runner, skill = _stub_runner(monkeypatch, tmp_path, model_probe=succeed)

    result = runner.run_harbor_eval(
        skill,
        ["claude-code"],
        agent_models={"claude-code": "anthropic/claude-test"},
        grading_mode="custom_only",
        output_dir=tmp_path / "results",
        agent_runtime_preflight=False,
    )

    assert "error" not in result
    assert [(probe.provider, probe.model) for probe in probe_calls] == [("anthropic", "claude-test")]


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


def test_runner_returns_error_when_private_evaluator_snapshot_cannot_be_created(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor import runner

    skill = tmp_path / "missing-evals-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# Missing evals\n", encoding="utf-8")

    result = runner.run_harbor_eval(skill, ["codex"])

    assert result["error"] == [f"No evaluator source directory found at {skill / 'evals'}"]


def test_runner_returns_structured_error_and_cleans_partial_snapshot_on_enospc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator.tier3.harbor import adapter, runner

    skill = tmp_path / "snapshot-enospc-skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Snapshot ENOSPC\n", encoding="utf-8")
    dataset = skill / "evals" / "evals.json"
    dataset.write_text(json.dumps([{"id": "case-1", "question": "Use the skill."}]), encoding="utf-8")
    temporary_root = tmp_path / "snapshot-temporary-root"
    temporary_root.mkdir()
    reporter = _RecordingReporter()
    provider_resolution_calls: list[bool] = []

    def fail_snapshot_copy(_source: Path, destination: Path, **_kwargs: Any) -> None:
        destination.mkdir(parents=True)
        (destination / "partial-copy").write_text("partial\n", encoding="utf-8")
        raise OSError(errno.ENOSPC, "No space left on device")

    def unexpected_provider_resolution() -> None:
        provider_resolution_calls.append(True)
        raise AssertionError("provider resolution must not start after snapshot failure")

    monkeypatch.setattr(adapter, "copytree_secure", fail_snapshot_copy)
    monkeypatch.setattr(tempfile, "tempdir", str(temporary_root))
    monkeypatch.setattr(runner, "resolve_llm_provider", unexpected_provider_resolution)
    results_root = tmp_path / "results"

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        output_dir=results_root,
        progress_reporter=reporter,
    )

    assert result["error"] == ["[Errno 28] No space left on device"]
    assert provider_resolution_calls == []
    assert not results_root.exists()
    assert list(temporary_root.iterdir()) == []
    assert json.loads(dataset.read_text(encoding="utf-8")) == [{"id": "case-1", "question": "Use the skill."}]
    transitions = [(event.stage, event.state) for event in reporter.events]
    assert ("configuration", "failed") in transitions
    assert transitions[-1] == ("run-finished", "failed")
    assert transitions.count(("run-finished", "failed")) == 1


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_runner_returns_error_for_unsafe_evals_config(tmp_path: Path, kind: str) -> None:
    from skillevaluator.tier3.harbor import runner

    skill = tmp_path / "unsafe-config-skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "evals.json").write_text("[]\n", encoding="utf-8")
    outside = tmp_path / "outside-config.yml"
    outside.write_text("schema_version: 1\n", encoding="utf-8")
    config = skill / "evals" / "config.yml"
    try:
        if kind == "symlink":
            config.symlink_to(outside)
        else:
            config.hardlink_to(outside)
    except OSError as exc:
        pytest.skip(f"{kind} creation is unavailable: {exc}")

    result = runner.run_harbor_eval(skill, ["codex"])

    assert result["error"] == [f"Eval configuration must be a regular non-linked file: {config}"]


def test_all_agent_and_baseline_arms_share_one_private_evaluator_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshots: list[Path] = []
    observed_questions: list[str] = []
    skill_ref: list[Path] = []

    def emit_tasks(_skill: Path, output: Path, **kwargs: Any) -> list[Path]:
        snapshot = Path(kwargs["evaluator_skill_path"])
        snapshots.append(snapshot)
        entries = json.loads((snapshot / "evals" / "evals.json").read_text(encoding="utf-8"))
        observed_questions.append(entries[0]["question"])
        if len(snapshots) == 1:
            (skill_ref[0] / "evals" / "evals.json").write_text(
                json.dumps([{"id": "case-1", "question": "Mutated after snapshot."}]),
                encoding="utf-8",
            )
        task = output / "case-1"
        task.mkdir(parents=True)
        return [task]

    runner, skill = _stub_runner(monkeypatch, tmp_path, task_emitter=emit_tasks)
    skill_ref.append(skill)
    (skill / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-1", "question": "Use the original fixture."}]),
        encoding="utf-8",
    )

    result = runner.run_harbor_eval(
        skill,
        ["codex", "opencode"],
        output_dir=tmp_path / "results",
        agent_runtime_preflight=False,
    )

    assert "error" not in result
    assert len(snapshots) == 4
    assert len(set(snapshots)) == 1
    assert observed_questions == ["Use the original fixture."] * 4
    assert not snapshots[0].exists()


def test_multi_agent_run_prevalidates_baseline_alias_candidates_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    validation_token = object()
    validation_calls: list[tuple[Path, tuple[Path, ...]]] = []
    baseline_tokens: list[object | None] = []

    def prevalidate(
        skill_path: Path,
        _reference_skills_dir: Path | None,
        _workspace_skill_paths: list[Path] | None,
        *,
        excluded_roots: tuple[Path, ...],
    ) -> object:
        validation_calls.append((skill_path, excluded_roots))
        return validation_token

    def emit_tasks(_skill: Path, output: Path, **kwargs: Any) -> list[Path]:
        if not kwargs["with_skill"]:
            baseline_tokens.append(kwargs.get("_baseline_alias_validation"))
        task = output / "case-1"
        task.mkdir(parents=True)
        return [task]

    runner, skill = _stub_runner(monkeypatch, tmp_path, task_emitter=emit_tasks)
    monkeypatch.setattr(runner, "_prevalidate_baseline_skill_candidates", prevalidate, raising=False)

    result = runner.run_harbor_eval(
        skill,
        ["codex", "opencode"],
        output_dir=tmp_path / "results",
        agent_runtime_preflight=False,
    )

    assert "error" not in result
    assert len(validation_calls) == 1
    assert validation_calls[0][0] == skill
    assert baseline_tokens == [validation_token, validation_token]


def test_runner_rejects_preexisting_run_symlink_before_child_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, skill = _stub_runner(monkeypatch, tmp_path)
    timestamp = "20260804_123456"
    fixed_now = SimpleNamespace(strftime=lambda _format: timestamp)
    monkeypatch.setattr(runner, "datetime", SimpleNamespace(now=lambda _timezone: fixed_now))
    monkeypatch.setattr(runner.os, "getpid", lambda: 12345)
    monkeypatch.setattr(runner, "uuid4", lambda: SimpleNamespace(hex="abcdef012345"))
    run_id = f"{timestamp}_12345_abcdef012345"
    results_root = tmp_path / "results"
    victim = tmp_path / "victim"
    results_root.mkdir()
    victim.mkdir()
    (results_root / run_id).symlink_to(victim, target_is_directory=True)

    result = runner.run_harbor_eval(skill, ["codex"], output_dir=results_root)

    assert "unique Tier 3 run directory" in str(result["error"][0])
    assert not (victim / "_harbor-jobs").exists()


def test_runner_reserves_unique_run_directories_concurrently(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor import runner

    results_root = tmp_path / "results"
    with ThreadPoolExecutor(max_workers=8) as executor:
        run_dirs = list(executor.map(lambda _index: runner._reserve_run_dir(results_root, "20260804_123456"), range(8)))

    assert len(set(run_dirs)) == 8
    assert all((run_dir / ".skillevaluator-generated-output").is_file() for run_dir in run_dirs)


def test_failed_run_reservation_cleanup_refuses_a_substituted_empty_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator.tier3.harbor import runner

    results_root = tmp_path / "results"
    moved_run = tmp_path / "moved-run"
    monkeypatch.setattr(runner.os, "getpid", lambda: 12345)
    monkeypatch.setattr(runner, "uuid4", lambda: SimpleNamespace(hex="abcdef012345"))
    expected_run = results_root / "20260804_123456_12345_abcdef012345"

    def substitute_then_fail(path: Path) -> None:
        path.rename(moved_run)
        path.mkdir()
        raise OSError("injected marker failure")

    monkeypatch.setattr(runner, "mark_generated_output_root", substitute_then_fail)

    with pytest.raises(OSError, match="injected marker failure"):
        runner._reserve_run_dir(results_root, "20260804_123456")

    assert expected_run.is_dir()
    assert moved_run.is_dir()


def test_jobs_directory_creation_failure_returns_structured_error_and_removes_owned_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, skill = _stub_runner(monkeypatch, tmp_path)
    reporter = _RecordingReporter()
    original_mkdir = Path.mkdir

    def fail_jobs_mkdir(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.name == "_harbor-jobs":
            raise OSError(errno.ENOSPC, "No space left on device")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_jobs_mkdir)
    results_root = tmp_path / "results"

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        output_dir=results_root,
        progress_reporter=reporter,
    )

    assert result == {"error": ["[Errno 28] No space left on device"]}
    assert results_root.is_dir()
    assert list(results_root.iterdir()) == []
    transitions = [(event.stage, event.state) for event in reporter.events]
    assert transitions[-1] == ("run-finished", "failed")
    assert transitions.count(("run-finished", "failed")) == 1


def test_jobs_directory_failure_cleans_owned_empty_run_without_descriptor_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator.tier3 import output_provenance

    runner, skill = _stub_runner(monkeypatch, tmp_path)
    original_mkdir = Path.mkdir

    def fail_jobs_mkdir(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.name == "_harbor-jobs":
            raise OSError("injected jobs directory failure")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(output_provenance, "_DESCRIPTOR_BACKEND", False)
    monkeypatch.setattr(Path, "mkdir", fail_jobs_mkdir)
    results_root = tmp_path / "results"

    result = runner.run_harbor_eval(skill, ["codex"], output_dir=results_root)

    assert result == {"error": ["injected jobs directory failure"]}
    assert list(results_root.iterdir()) == []


def test_jobs_directory_creation_failure_preserves_run_after_provenance_is_lost(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, skill = _stub_runner(monkeypatch, tmp_path)
    original_mkdir = Path.mkdir
    reserved_run: list[Path] = []

    def fail_after_tampering(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.name == "_harbor-jobs":
            reserved_run.append(path.parent)
            marker = path.parent / ".skillevaluator-generated-output"
            marker.write_text("not an authentic marker\n", encoding="utf-8")
            (path.parent / "preserve.txt").write_text("unowned\n", encoding="utf-8")
            raise OSError("injected jobs directory failure")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_after_tampering)

    result = runner.run_harbor_eval(skill, ["codex"], output_dir=tmp_path / "results")

    assert result == {"error": ["injected jobs directory failure"]}
    assert len(reserved_run) == 1
    assert (reserved_run[0] / "preserve.txt").read_text(encoding="utf-8") == "unowned\n"


def test_runner_publishes_latest_through_shared_atomic_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, skill = _stub_runner(monkeypatch, tmp_path)
    published: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        runner,
        "publish_latest_results",
        lambda root, run_id: published.append((root, run_id)),
        raising=False,
    )
    results_root = tmp_path / "results"

    result = runner.run_harbor_eval(skill, ["codex"], output_dir=results_root)

    assert published == [(results_root, Path(result["run_dir"]).name)]


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


def _collect_completed_report_input(*, output_dir: Path, **_kwargs: Any) -> dict[str, Any]:
    from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS

    for arm in ("with-skill", "without-skill"):
        directory = output_dir / "codex" / arm
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(
            json.dumps(
                {
                    "scores": dict.fromkeys(DEFAULT_METRICS, 0.8),
                    "metrics": list(DEFAULT_METRICS),
                    "num_trials": 1,
                    "execution_status": "succeeded",
                    "execution_errors": [],
                    "expected_attempts": 1,
                    "scored_attempts": 1,
                }
            ),
            encoding="utf-8",
        )
    return {"execution_status": "succeeded", "execution_errors": [], "metrics": list(DEFAULT_METRICS)}


def test_runner_persists_scoped_benchmark_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator.evaluation.tier3_report import render_agent_eval_html_report

    def render_without_judge(skill_path: Path, run_dir: Path, **kwargs: Any) -> Path:
        return render_agent_eval_html_report(skill_path, run_dir, use_llm_judge=False, **kwargs)

    runner, skill = _stub_runner(
        monkeypatch, tmp_path, collect=_collect_completed_report_input, html_report=render_without_judge
    )
    result = runner.run_harbor_eval(skill, ["codex"], output_dir=tmp_path / "results")
    benchmark = Path(result["run_dir"]) / "tier3-benchmark.md"
    persisted = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))

    assert benchmark.is_file()
    assert result["benchmark_path"] == str(benchmark)
    assert persisted["benchmark_path"] == str(benchmark)
    assert not benchmark.with_name("BENCHMARK.md").exists()


def test_runner_removes_scoped_benchmark_when_identity_rerender_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator.evaluation.tier3_report import render_agent_eval_html_report

    def render_then_change_source(skill_path: Path, run_dir: Path, **kwargs: Any) -> Path:
        if (run_dir / "tier3-benchmark.md").exists():
            raise OSError("injected rerender failure")
        report = render_agent_eval_html_report(skill_path, run_dir, use_llm_judge=False, **kwargs)
        (skill_path / "SKILL.md").write_text("# changed during reporting\n", encoding="utf-8")
        return report

    runner, skill = _stub_runner(
        monkeypatch, tmp_path, collect=_collect_completed_report_input, html_report=render_then_change_source
    )
    (skill / "SKILL.md").write_text("# evaluated source\n", encoding="utf-8")
    result = runner.run_harbor_eval(skill, ["codex"], output_dir=tmp_path / "results")
    run_dir = Path(result["run_dir"])

    assert result["publication_target_conflict"]
    assert "report_path" not in result
    assert "benchmark_path" not in result
    assert not (run_dir / "report.html").exists()
    assert not (run_dir / "tier3-benchmark.md").exists()


def test_runner_persists_run_owned_publication_identity_in_final_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator.publication_identity import publication_target_from_path

    report_identity: dict[str, Any] = {}

    def write_html(_skill_path: Path, run_dir: Path, *, engine_result: dict[str, Any], **_kwargs: Any) -> Path:
        report_identity.update(
            {
                "run_id": engine_result.get("run_id"),
                "publication_target": engine_result.get("publication_target"),
            }
        )
        path = run_dir / "report.html"
        path.write_text("<html></html>", encoding="utf-8")
        return path

    runner, skill = _stub_runner(monkeypatch, tmp_path, html_report=write_html)
    expected_target = publication_target_from_path(skill)
    assert expected_target is not None

    result = runner.run_harbor_eval(skill, ["codex"], output_dir=tmp_path / "results")
    persisted = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))

    assert result["publication_target"] == expected_target
    assert persisted["publication_target"] == expected_target
    assert persisted["run_id"] == result["run_id"] == Path(result["run_dir"]).name
    assert report_identity == {
        "run_id": result["run_id"],
        "publication_target": expected_target,
    }


@pytest.mark.parametrize(
    ("context_mode", "source_is_target_bound"),
    [("linked", False), ("full", False), ("full", True)],
)
def test_runner_fails_publication_closed_for_external_repo_runtime_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    context_mode: str,
    source_is_target_bound: bool,
) -> None:
    from skillevaluator.publication_identity import publication_target_from_path

    external_source = tmp_path / "shared-runtime.txt"
    external_source.write_text("agent-visible external input\n", encoding="utf-8")

    def emit_tasks(skill: Path, output: Path, **_kwargs: Any) -> list[Path]:
        task = output / "case-1"
        environment = task / "environment"
        environment.mkdir(parents=True)
        staged_source = skill / "SKILL.md" if source_is_target_bound else external_source
        (environment / "repo-context.json").write_text(
            json.dumps(
                {
                    "mode": context_mode,
                    "repo_root": str(tmp_path),
                    "files": [
                        {
                            "source": str(staged_source),
                            "container": "/workspace/repo/shared-runtime.txt",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return [task]

    runner, skill = _stub_runner(monkeypatch, tmp_path, task_emitter=emit_tasks)
    (skill / "SKILL.md").write_text("# evaluated target\n", encoding="utf-8")
    expected_target = publication_target_from_path(skill)
    assert expected_target is not None

    result = runner.run_harbor_eval(skill, ["codex"], output_dir=tmp_path / "results")
    persisted = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))

    assert result["publication_target"] is None
    assert result["publication_target_conflict"] == {
        "run_start": expected_target,
        "run_end": None,
        "reason_code": "runtime_context_outside_publication_target",
    }
    assert persisted["publication_target"] is None
    assert persisted["publication_target_conflict"] == result["publication_target_conflict"]


def test_runtime_context_binding_rejects_external_reference_and_workspace_skills(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor.runner import _runtime_context_is_bound_to_publication_target

    skill = tmp_path / "demo"
    skill.mkdir()
    references = tmp_path / "references"
    (references / "reference-skill").mkdir(parents=True)
    workspace = tmp_path / "workspace-skill"
    workspace.mkdir()

    assert not _runtime_context_is_bound_to_publication_target(
        skill,
        (),
        reference_skills_dir=references,
        workspace_skill_paths=(),
    )
    (references / "reference-skill").rmdir()
    assert not _runtime_context_is_bound_to_publication_target(
        skill,
        (),
        reference_skills_dir=references,
        workspace_skill_paths=(),
    )
    assert not _runtime_context_is_bound_to_publication_target(
        skill,
        (),
        reference_skills_dir=None,
        workspace_skill_paths=(workspace,),
    )

    internal_references = skill / "references"
    (internal_references / "reference-skill").mkdir(parents=True)
    internal_workspace = skill / "workspace-skill"
    internal_workspace.mkdir()
    assert _runtime_context_is_bound_to_publication_target(
        skill,
        (),
        reference_skills_dir=internal_references,
        workspace_skill_paths=(internal_workspace,),
    )

    assert not _runtime_context_is_bound_to_publication_target(
        skill,
        (),
        reference_skills_dir=tmp_path / "missing-references",
        workspace_skill_paths=(),
    )

    external_alias = tmp_path / "external-workspace-alias"
    try:
        external_alias.symlink_to(internal_workspace, target_is_directory=True)
    except OSError:
        pass
    else:
        assert not _runtime_context_is_bound_to_publication_target(
            skill,
            (),
            reference_skills_dir=None,
            workspace_skill_paths=(external_alias,),
        )


@pytest.mark.parametrize(
    "excluded_parent",
    [
        Path("results"),
        Path(".results"),
        Path(".venv"),
        Path("node_modules"),
        Path("evals/results"),
    ],
)
def test_runtime_context_binding_rejects_workspace_skill_excluded_from_target_digest(
    tmp_path: Path,
    excluded_parent: Path,
) -> None:
    from skillevaluator.publication_identity import publication_target_from_path
    from skillevaluator.tier3.harbor.runner import _runtime_context_is_bound_to_publication_target

    skill = (tmp_path / "demo").resolve()
    workspace = skill / excluded_parent / "workspace-skill"
    workspace.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# evaluated target\n", encoding="utf-8")
    workspace_manifest = workspace / "SKILL.md"
    workspace_manifest.write_text("# runtime input v1\n", encoding="utf-8")

    target_before = publication_target_from_path(skill)
    assert target_before is not None
    assert not _runtime_context_is_bound_to_publication_target(
        skill,
        (),
        reference_skills_dir=None,
        workspace_skill_paths=(workspace.resolve(),),
    )

    workspace_manifest.write_text("# attacker-controlled runtime input v2\n", encoding="utf-8")

    assert publication_target_from_path(skill) == target_before
    assert not _runtime_context_is_bound_to_publication_target(
        skill,
        (),
        reference_skills_dir=None,
        workspace_skill_paths=(workspace.resolve(),),
    )


def test_runtime_context_binding_rejects_reference_child_excluded_from_target_digest(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor.runner import _runtime_context_is_bound_to_publication_target

    skill = (tmp_path / "demo").resolve()
    references = skill / "evals"
    excluded_reference = references / "results" / "reference-skill"
    excluded_reference.mkdir(parents=True)

    assert not _runtime_context_is_bound_to_publication_target(
        skill,
        (),
        reference_skills_dir=references,
        workspace_skill_paths=(),
    )


def test_runtime_context_binding_fails_closed_for_unstable_relative_or_oversized_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.tier3.harbor import runner

    missing_skill = tmp_path / "missing-skill"
    assert not runner._runtime_context_is_bound_to_publication_target(
        missing_skill,
        (),
        reference_skills_dir=None,
        workspace_skill_paths=(),
    )

    skill = tmp_path / "demo"
    skill.mkdir()
    missing_manifest_task = tmp_path / "missing-manifest-task"
    (missing_manifest_task / "environment" / "repo").mkdir(parents=True)
    assert not runner._runtime_context_is_bound_to_publication_target(
        skill,
        (missing_manifest_task,),
        reference_skills_dir=None,
        workspace_skill_paths=(),
    )

    task = tmp_path / "task"
    environment = task / "environment"
    environment.mkdir(parents=True)
    (environment / "repo-context.json").write_text(
        json.dumps(
            {
                "mode": "linked",
                "repo_root": str(tmp_path),
                "files": [{"source": "demo/SKILL.md", "container": "/workspace/repo/demo/SKILL.md"}],
            }
        ),
        encoding="utf-8",
    )

    assert not runner._runtime_context_is_bound_to_publication_target(
        skill,
        (task,),
        reference_skills_dir=None,
        workspace_skill_paths=(),
    )

    monkeypatch.setattr(runner, "_MAX_REPO_CONTEXT_METADATA_BYTES", 32)
    (environment / "repo-context.json").write_text("{" + " " * 32 + "}", encoding="utf-8")
    assert not runner._runtime_context_is_bound_to_publication_target(
        skill,
        (task,),
        reference_skills_dir=None,
        workspace_skill_paths=(),
    )


def test_runner_rejects_source_change_after_private_snapshot_before_engine_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator.publication_identity import publication_target_from_path

    task_calls: list[Path] = []
    engine_calls: list[dict[str, Any]] = []

    def emit_tasks(_skill: Path, output: Path, **_kwargs: Any) -> list[Path]:
        task_calls.append(output)
        return [output / "case-1"]

    def run_agent(**kwargs: Any) -> list[str]:
        engine_calls.append(kwargs)
        return []

    runner, skill = _stub_runner(
        monkeypatch,
        tmp_path,
        task_emitter=emit_tasks,
        run_agent=run_agent,
    )
    original_snapshot = runner.private_evaluator_skill_snapshot
    initial_target = publication_target_from_path(skill)
    assert initial_target is not None

    @contextmanager
    def mutate_after_copy(path: Path):
        with original_snapshot(path) as snapshot:
            assert (snapshot / "evals" / "evals.json").read_text(encoding="utf-8") == "{}"
            (path / "evals" / "evals.json").write_text('{"changed": true}', encoding="utf-8")
            yield snapshot

    monkeypatch.setattr(runner, "private_evaluator_skill_snapshot", mutate_after_copy)

    result = runner.run_harbor_eval(skill, ["codex"], output_dir=tmp_path / "results")

    assert result == {"error": ["Publication source changed while its private evaluator snapshot was created"]}
    assert publication_target_from_path(skill) != initial_target
    assert task_calls == []
    assert engine_calls == []
    assert not (tmp_path / "results").exists()


def test_runner_marks_publication_identity_conflict_when_source_changes_mid_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator.publication_identity import publication_target_from_path

    skill_ref: list[Path] = []

    def collect_after_mutation(**_kwargs: Any) -> dict[str, Any]:
        (skill_ref[0] / "SKILL.md").write_text("# changed during evaluation\n", encoding="utf-8")
        return {"execution_status": "succeeded", "execution_errors": [], "metrics": []}

    runner, skill = _stub_runner(monkeypatch, tmp_path, collect=collect_after_mutation)
    (skill / "SKILL.md").write_text("# evaluated source\n", encoding="utf-8")
    skill_ref.append(skill)
    run_start_target = publication_target_from_path(skill)
    assert run_start_target is not None

    result = runner.run_harbor_eval(skill, ["codex"], output_dir=tmp_path / "results")
    run_end_target = publication_target_from_path(skill)
    persisted = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))

    assert run_end_target is not None and run_end_target != run_start_target
    assert result["publication_target"] is None
    assert result["publication_target_conflict"] == {
        "run_start": run_start_target,
        "run_end": run_end_target,
    }
    assert persisted["publication_target"] is None
    assert persisted["publication_target_conflict"] == result["publication_target_conflict"]


def test_runner_rechecks_identity_after_report_generation_and_renders_conflict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator.publication_identity import publication_target_from_path

    skill_ref: list[Path] = []
    rendered_targets: list[object] = []

    def write_html(_skill_path: Path, run_dir: Path, *, engine_result: dict[str, Any], **_kwargs: Any) -> Path:
        rendered_targets.append(engine_result.get("publication_target"))
        if len(rendered_targets) == 1:
            (skill_ref[0] / "SKILL.md").write_text("# changed during report\n", encoding="utf-8")
        path = run_dir / "report.html"
        path.write_text("<html></html>", encoding="utf-8")
        return path

    runner, skill = _stub_runner(monkeypatch, tmp_path, html_report=write_html)
    (skill / "SKILL.md").write_text("# evaluated source\n", encoding="utf-8")
    skill_ref.append(skill)
    run_start_target = publication_target_from_path(skill)
    assert run_start_target is not None

    result = runner.run_harbor_eval(skill, ["codex"], output_dir=tmp_path / "results")
    run_end_target = publication_target_from_path(skill)
    persisted = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))

    assert run_end_target is not None and run_end_target != run_start_target
    assert rendered_targets == [run_start_target, None]
    assert result["publication_target"] is None
    assert result["publication_target_conflict"] == {
        "run_start": run_start_target,
        "run_end": run_end_target,
    }
    assert persisted["publication_target"] is None
    assert persisted["publication_target_conflict"] == result["publication_target_conflict"]


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

    runner, skill = _stub_runner(
        monkeypatch,
        tmp_path,
        task_emitter=fail_staging,
        provider_name="openai-compatible",
        provider_base_url="https://gateway.example/v1",
    )

    result = runner.run_harbor_eval(skill, ["codex"], output_dir=tmp_path / "results")

    run_dir = Path(result["run_dir"])
    assert result["error"] == ["invalid task"]
    assert not (run_dir / "_harbor-jobs").exists()
    assert not (run_dir / "_harbor-tasks").exists()
    assert result["harbor_jobs_retained"] is False
    assert result["run_config"]["credential_validation"]["status"] == "degraded"
    assert result["run_config"]["credential_validation"]["targets"] == [
        {
            "labels": ["codex"],
            "provider": "openai-compatible",
            "model": "openai/openai/gpt-6.1-sol",
            "status": "degraded",
            "detail": "model catalog access does not verify runtime credentials for this endpoint",
        },
        {
            "labels": ["standard grader"],
            "provider": "openai-compatible",
            "model": "gpt-5",
            "status": "degraded",
            "detail": "model catalog access does not verify runtime credentials for this endpoint",
        },
    ]
    persisted = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert persisted["run_config"] == result["run_config"]
    run_config_path = run_dir / "run_config.json"
    assert json.loads(run_config_path.read_text(encoding="utf-8")) == result["run_config"]


def test_runner_persists_run_owned_identity_for_pre_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator.publication_identity import publication_target_from_path

    def fail_staging(*_args: Any, **_kwargs: Any) -> list[Path]:
        raise ValueError("invalid task")

    runner, skill = _stub_runner(monkeypatch, tmp_path, task_emitter=fail_staging)
    expected_target = publication_target_from_path(skill)
    assert expected_target is not None

    result = runner.run_harbor_eval(skill, ["codex"], output_dir=tmp_path / "results")
    persisted = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))

    assert result["publication_target"] == expected_target
    assert persisted["publication_target"] == expected_target
    assert persisted["run_id"] == result["run_id"] == Path(result["run_dir"]).name


@pytest.mark.parametrize("failure_arm", ["with-skill", "baseline"])
def test_task_staging_enospc_returns_structured_error_without_launch_and_cleans_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_arm: str,
) -> None:
    launches: list[dict[str, Any]] = []
    evaluator_snapshots: list[Path] = []

    def fail_staging(_skill: Path, output: Path, **kwargs: Any) -> list[Path]:
        evaluator_snapshots.append(Path(kwargs["evaluator_skill_path"]))
        output.mkdir(parents=True)
        (output / "partial-copy").write_text("partial\n", encoding="utf-8")
        should_fail = (failure_arm == "with-skill" and kwargs["with_skill"]) or (
            failure_arm == "baseline" and not kwargs["with_skill"]
        )
        if should_fail:
            raise OSError(errno.ENOSPC, "No space left on device")
        return [output / "case-1"]

    def record_launch(**kwargs: Any) -> list[str]:
        launches.append(kwargs)
        return []

    runner, skill = _stub_runner(
        monkeypatch,
        tmp_path,
        task_emitter=fail_staging,
        run_agent=record_launch,
    )
    reporter = _RecordingReporter()

    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        skip_baseline=False,
        output_dir=tmp_path / "results",
        progress_reporter=reporter,
    )

    run_dir = Path(result["run_dir"])
    assert result["error"] == ["[Errno 28] No space left on device"]
    assert launches == []
    assert len(evaluator_snapshots) == (1 if failure_arm == "with-skill" else 2)
    assert len(set(evaluator_snapshots)) == 1
    assert not evaluator_snapshots[0].exists()
    assert not (run_dir / "_harbor-jobs").exists()
    assert not (run_dir / "_harbor-tasks").exists()
    assert result["harbor_jobs_retained"] is False
    transitions = [(event.stage, event.state) for event in reporter.events]
    expected_stage = "with-skill-tasks" if failure_arm == "with-skill" else "baseline-tasks"
    assert (expected_stage, "running") in transitions
    assert (expected_stage, "failed") in transitions
    if failure_arm == "baseline":
        assert ("with-skill-tasks", "failed") not in transitions
    assert not any(stage.startswith("agent:") and state == "running" for stage, state in transitions)
    assert transitions[-1] == ("run-finished", "failed")
    assert transitions.count(("run-finished", "failed")) == 1


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


class _InterruptingReporter(_RecordingReporter):
    """Deliver Ctrl+C on the coordinator thread once the first agent is running."""

    def __init__(self) -> None:
        super().__init__()
        self.interrupted = False

    def emit(self, event: Any) -> None:
        super().emit(event)
        if not self.interrupted and event.stage.startswith("agent:") and event.state == "running":
            self.interrupted = True
            raise KeyboardInterrupt


def test_interrupt_cancels_running_agent_and_never_starts_queued_agents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launched: list[str] = []
    observed_cancellation: list[bool] = []

    def run_agent(**kwargs: Any) -> list[str]:
        launched.append(kwargs["agent"])
        cancellation = kwargs.get("cancellation")
        # Harbor jobs poll this event; the bounded wait keeps a regression from hanging.
        observed_cancellation.append(cancellation is not None and cancellation.requested.wait(10))
        return []

    # The stub configures max_agents=1, so the second agent waits in the queue.
    runner, skill = _stub_runner(monkeypatch, tmp_path, run_agent=run_agent)
    reporter = _InterruptingReporter()

    with pytest.raises(KeyboardInterrupt):
        runner.run_harbor_eval(
            skill,
            ["codex", "opencode"],
            output_dir=tmp_path / "results",
            progress_reporter=reporter,
        )

    assert launched == ["codex"]
    assert observed_cancellation == [True]
    events = [(event.stage, event.state, event.detail) for event in reporter.events]
    assert ("agent:codex", "failed", "interrupted before completion") in events
    assert ("agent:opencode", "running", "with-skill + baseline") not in events
    assert events[-1] == ("run-finished", "failed", "Tier 3 evaluation interrupted")


class _RepeatedInterruptReporter(_RecordingReporter):
    """Deliver Ctrl+C when the first agent starts and again when it reports stopping."""

    def __init__(self) -> None:
        super().__init__()
        self.interrupts = 0

    def emit(self, event: Any) -> None:
        super().emit(event)
        if self.interrupts < 2 and event.stage.startswith("agent:") and event.state == "running":
            self.interrupts += 1
            raise KeyboardInterrupt


def test_second_interrupt_forces_running_agents_to_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[tuple[bool, bool]] = []

    def run_agent(**kwargs: Any) -> list[str]:
        cancellation = kwargs["cancellation"]
        # A running Harbor job stays in its cleanup grace until it is forced.
        observed.append((cancellation.requested.wait(10), cancellation.forced.wait(10)))
        return []

    runner, skill = _stub_runner(monkeypatch, tmp_path, run_agent=run_agent)
    reporter = _RepeatedInterruptReporter()

    with pytest.raises(KeyboardInterrupt):
        runner.run_harbor_eval(
            skill,
            ["codex"],
            output_dir=tmp_path / "results",
            progress_reporter=reporter,
        )

    assert observed == [(True, True)]
    assert reporter.interrupts == 2
    events = [(event.stage, event.state, event.detail) for event in reporter.events]
    stopping = [detail for stage, state, detail in events if stage == "agent:codex" and state == "running"]
    assert stopping[-1] == runner._HARBOR_RUN_STOPPING_DETAIL
    assert stopping[-1].startswith("2nd Ctrl+C skips wait; stopping Harbor jobs")
    assert stopping[-1].endswith("up to 30s of environment cleanup")
    assert ("agent:codex", "failed", "interrupted before completion") in events
    assert events[-1] == ("run-finished", "failed", "Tier 3 evaluation interrupted")


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
        agent_runtime_preflight=True,
        progress_reporter=reporter,
    )


def test_runtime_preflight_is_skipped_by_default_with_enablement_hint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, skill = _stub_runner(monkeypatch, tmp_path)
    reporter = _RecordingReporter()

    runner.run_harbor_eval(
        skill,
        ["codex"],
        output_dir=tmp_path / "results",
        progress_reporter=reporter,
    )

    event = next(event for event in reporter.events if event.stage == "agent-runtime-preflight")
    assert event.state == "skipped"
    assert event.detail == (
        "disabled by default; enable with --agent-runtime-preflight or harbor.agent_runtime_preflight"
    )


def test_cleanup_failure_degrades_terminal_progress_after_retention_finalizes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, skill = _stub_runner(monkeypatch, tmp_path)
    reporter = _RecordingReporter()
    from skillevaluator.tier3.harbor import artifact_retention

    real_rmtree = artifact_retention.shutil.rmtree

    def fail_harbor_artifact_cleanup(path: str | Path, *args: Any, **kwargs: Any) -> None:
        if Path(path).name.startswith("_harbor-"):
            raise OSError(f"busy: {path}")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(
        artifact_retention.shutil,
        "rmtree",
        fail_harbor_artifact_cleanup,
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
        lambda: SimpleNamespace(
            provider="openai",
            model="gpt-5",
            api_key="sk-test-secret",
            base_url="https://api.openai.com/v1",
        ),
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
    monkeypatch.setattr(runner, "_resolve_runtime_env", lambda _templates, **_kwargs: ({}, []))
    monkeypatch.setattr(runner, "_validate_agent_provider_credentials", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(runner, "_harbor_subprocess_environment", lambda **_kwargs: {})
    from skillevaluator.tier3.harbor import runtime_preflight

    monkeypatch.setattr(
        runtime_preflight,
        "probe_model",
        lambda selected_provider: SimpleNamespace(
            ok=True,
            provider=selected_provider.provider,
            model=selected_provider.model,
            detail=f"model {selected_provider.model} is available",
            failure_kind=None,
            http_status=None,
        ),
    )
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
        agent_runtime_preflight=True,
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
        ("configuration", "complete"),
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
    skill = tmp_path / "demo"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "evals.json").write_text("[]\n", encoding="utf-8")
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

    result = runner.run_harbor_eval(skill, ["codex"], progress_reporter=reporter)

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
    skill = tmp_path / "demo"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "evals.json").write_text("[]\n", encoding="utf-8")
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

    result = runner.run_harbor_eval(skill, ["codex"], progress_reporter=reporter)

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
    original_replace = runner.os.replace

    def fail_result_write(source: str | Path, destination: str | Path, *args, **kwargs):
        if Path(destination).name == "result.json":
            raise OSError("result write failed")
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(runner.os, "replace", fail_result_write)
    results_root = tmp_path / "results"

    with pytest.raises(OSError, match="result write failed"):
        runner.run_harbor_eval(
            skill,
            ["codex"],
            output_dir=results_root,
            keep_harbor_jobs=True,
            progress_reporter=reporter,
        )

    run_dirs = [candidate for candidate in results_root.iterdir() if candidate.is_dir()]
    assert len(run_dirs) == 1
    assert not (run_dirs[0] / "result.json").exists()
    assert not list(run_dirs[0].glob(".result.json.*.tmp"))
    transitions = [(event.stage, event.state) for event in reporter.events]
    assert ("report", "running") in transitions
    assert ("report", "failed") in transitions
    assert transitions[-1] == ("run-finished", "failed")
    assert transitions.count(("run-finished", "failed")) == 1


def test_runner_atomically_publishes_one_complete_final_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, skill = _stub_runner(monkeypatch, tmp_path)
    original_replace = runner.os.replace
    published_payloads: list[dict[str, Any]] = []

    def inspect_result_publication(source: str | Path, destination: str | Path, *args, **kwargs):
        destination_path = Path(destination)
        if destination_path.name == "result.json":
            source_directory_fd = kwargs.get("src_dir_fd")
            destination_directory_fd = kwargs.get("dst_dir_fd")
            if destination_directory_fd is None:
                assert not destination_path.exists()
            else:
                with pytest.raises(FileNotFoundError):
                    os.stat(destination_path.name, dir_fd=destination_directory_fd, follow_symlinks=False)
            if source_directory_fd is None:
                payload = Path(source).read_text(encoding="utf-8")
            else:
                descriptor = os.open(Path(source).name, os.O_RDONLY, dir_fd=source_directory_fd)
                with os.fdopen(descriptor, encoding="utf-8") as handle:
                    payload = handle.read()
            published_payloads.append(json.loads(payload))
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(runner.os, "replace", inspect_result_publication)

    result = runner.run_harbor_eval(skill, ["codex"], output_dir=tmp_path / "results")

    assert len(published_payloads) == 1
    assert published_payloads[0]["execution_status"] == "succeeded"
    assert published_payloads[0]["run_id"] == result["run_id"]
    assert json.loads(Path(result["result_path"]).read_text(encoding="utf-8")) == published_payloads[0]


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
    monkeypatch.setattr(
        runner,
        "_resolve_runtime_env",
        lambda _templates, **_kwargs: ({"CUSTOM_VALUE": secret}, []),
    )
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
        runner,
        "_run_bounded_harbor_process",
        lambda *_args, **_kwargs: runner._BoundedHarborProcessResult(
            returncode=17,
            output_tail=f"OPENAI_API_KEY={secret}\nraw agent output\ntoken={secret}\nraw verifier output",
            output_exceeded=False,
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


def test_harbor_failure_detail_redacts_before_tail_truncation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator.tier3.harbor import runner

    secret = "SYNTHETIC_SECRET_ABCDEF"
    monkeypatch.setattr(runner, "build_harbor_run_command", lambda **_kwargs: ["harbor", "run"])
    monkeypatch.setattr(
        runner,
        "_run_bounded_harbor_process",
        lambda *_args, **_kwargs: runner._BoundedHarborProcessResult(
            returncode=17,
            output_tail="prefix|" + secret + "x" * 1988,
            output_exceeded=False,
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
    assert "CRET_ABCDEF" not in detail
    assert "<redacted>" in detail


def test_command_does_not_relabel_runtime_failure_as_configuration_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillevaluator.tier3 import commands

    skill = tmp_path / "demo"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "evals.json").write_text("[]\n", encoding="utf-8")
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
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "evals.json").write_text("[]\n", encoding="utf-8")
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


def test_short_credential_named_flags_do_not_redact_counts() -> None:
    from skillevaluator.tier3.harbor.progress import redact_progress_detail, secret_values_from_environment

    secrets = secret_values_from_environment(
        {
            "SDK_HAS_HOST_AUTH_REFRESH": "1",
            "FEATURE_TOKEN_ENABLED": "yes",
            "SERVICE_API_KEY": "service-key-123456",
            "HTTPS_PROXY": "http://u:p@proxy.example:8080",
        }
    )

    assert "1" not in secrets
    assert "yes" not in secrets
    assert "service-key-123456" in secrets
    detail = redact_progress_detail(
        "Harbor job did not complete successfully: 1 errored; codex with-skill (0/1 scored) "
        "via http://u:p@proxy.example:8080 using service-key-123456",
        secret_values=secrets,
    )
    assert "1 errored" in detail
    assert "(0/1 scored)" in detail
    assert "service-key-123456" not in detail
    assert "u:p@" not in detail
