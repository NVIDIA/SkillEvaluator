# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Terminal presentation for Tier 3 live-evaluation progress.

The runner emits structured events through :class:`ProgressReporter`; it does
not know whether those events are shown by Rich, written as durable CI lines,
or discarded by an API caller.
"""

from __future__ import annotations

import logging
import re
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from time import monotonic
from typing import Literal, Protocol, TextIO, runtime_checkable

from skillevaluator.tier3.harbor.secret_redaction import redact_secrets_in_log_line

ProgressMode = Literal["auto", "rich", "plain", "off"]
logger = logging.getLogger(__name__)

ProgressState = Literal["running", "ready", "complete", "failed", "degraded", "delegated", "skipped"]

_TERMINAL_STATES = frozenset({"ready", "complete", "failed", "degraded", "delegated", "skipped"})
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?key|auth|bearer|credential|password|secret|token)\b\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_SECRET_ENV_NAME_RE = re.compile(r"(?i)(?:api[_-]?key|access[_-]?key|auth|credential|password|secret|token)")
_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_OSC_ESCAPE_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_TERMINAL_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


@dataclass(frozen=True, slots=True)
class Tier3RunPlan:
    """Safe-to-render summary of a Tier 3 run.

    Counts may be unknown during the command's initial preflight. A later
    ``start`` call renders the updated plan once task staging has resolved the
    effective values.
    """

    skill_name: str
    environment: str
    agents: tuple[str, ...]
    agent_models: tuple[tuple[str, str], ...] = ()
    provider: str | None = None
    task_count: int | None = None
    case_count: int | None = None
    attempts: int | None = None
    baseline: bool | None = None
    concurrency: int | None = None
    max_agents: int | None = None
    timeout_multiplier: float | None = None
    matrix_trials: int | None = None
    preflight_trials: int | None = None
    total_containers: int | None = None
    task_timeout_seconds: float | None = None
    output_dir: str | None = None
    result_path: str | None = None
    report_path: str | None = None


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """A structured progress transition emitted by the Tier 3 engine."""

    stage: str
    state: ProgressState
    detail: str | None = None
    output_dir: str | None = None
    result_path: str | None = None
    report_path: str | None = None


@runtime_checkable
class ProgressReporter(Protocol):
    """Presentation-independent sink for Tier 3 progress events."""

    @property
    def is_active(self) -> bool: ...

    def start(self, plan: Tier3RunPlan) -> None: ...

    def set_secret_values(self, values: list[str] | tuple[str, ...] | set[str]) -> None: ...

    def emit(self, event: ProgressEvent) -> None: ...

    def heartbeat(self) -> None: ...

    def close(self) -> None: ...


def redact_progress_detail(detail: object, *, secret_values: set[str] | None = None) -> str:
    """Return a single-line diagnostic safe enough for a progress surface."""
    text = _OSC_ESCAPE_RE.sub("", str(detail))
    text = _ANSI_ESCAPE_RE.sub("", text)
    text = _TERMINAL_CONTROL_RE.sub("", text)
    text = " ".join(text.split())
    for secret in sorted(secret_values or (), key=len, reverse=True):
        if len(secret) >= 4:
            text = text.replace(secret, "<redacted>")
    text = redact_secrets_in_log_line(text, extra_secret_values=secret_values)
    return _SECRET_ASSIGNMENT_RE.sub(r"\1<redacted>", text)


def secret_values_from_environment(environment: Mapping[str, str]) -> set[str]:
    """Extract exact credential values without treating every env value as secret."""
    return {str(value) for name, value in environment.items() if value and _SECRET_ENV_NAME_RE.search(name)}


class PlainProgressReporter:
    """Durable line-oriented reporter for redirected output and CI logs."""

    def __init__(self, *, stream: TextIO | None = None, refresh_interval: float = 1.0) -> None:
        self._stream = stream or sys.stderr
        self._refresh_interval = max(float(refresh_interval), 0.05)
        self._started_at: float | None = None
        self._active: dict[str, str] = {}
        self._secret_values: set[str] = set()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._plan_rendered = False

    @property
    def is_active(self) -> bool:
        return self._started_at is not None and not self._stop.is_set()

    def start(self, plan: Tier3RunPlan) -> None:
        with self._lock:
            if self._started_at is None:
                self._started_at = monotonic()
                self._stop.clear()
                self._thread = threading.Thread(
                    target=self._heartbeat_loop,
                    name="tier3-progress-heartbeat",
                    daemon=True,
                )
                self._thread.start()
            self._render_plan(plan)

    def set_secret_values(self, values: list[str] | tuple[str, ...] | set[str]) -> None:
        with self._lock:
            self._secret_values.update(value for value in values if value)

    def emit(self, event: ProgressEvent) -> None:
        with self._lock:
            detail = self._event_detail(event)
            if event.state == "running":
                self._active[event.stage] = detail
            elif event.state in _TERMINAL_STATES:
                self._active.pop(event.stage, None)
            suffix = f" - {detail}" if detail else ""
            self._write_line(f"[{self._elapsed()}] {event.stage}: {event.state}{suffix}")

    def heartbeat(self) -> None:
        with self._lock:
            if not self._active:
                return
            stages = ", ".join(sorted(self._active))
            self._write_line(f"[{self._elapsed()}] still running: {stages}")

    def close(self) -> None:
        try:
            with self._lock:
                for stage in tuple(self._active):
                    self.emit(
                        ProgressEvent(
                            stage=stage,
                            state="failed",
                            detail="interrupted before completion",
                        )
                    )
        finally:
            self._stop.set()
            thread = self._thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=min(self._refresh_interval + 0.1, 1.1))
            with self._lock:
                self._active.clear()
                self._thread = None
                self._started_at = None
                self._plan_rendered = False

    def _render_plan(self, plan: Tier3RunPlan) -> None:
        first_plan = not self._plan_rendered
        self._plan_rendered = True
        if first_plan:
            self._write_line(f"Tier 3 live evaluation: {self._safe_text(plan.skill_name)}")
            self._write_line(f"  environment: {self._safe_text(plan.environment)}")
        model_map = dict(plan.agent_models)
        agents = ", ".join(
            f"{self._safe_text(agent)}={self._safe_text(model_map[agent])}"
            if agent in model_map
            else self._safe_text(agent)
            for agent in plan.agents
        )
        if first_plan:
            self._write_line(f"  agents/models: {agents or 'none'}")
        if first_plan and plan.provider:
            self._write_line(f"  provider: {self._safe_text(plan.provider)}")
        values = (
            ("tasks", plan.task_count),
            ("cases", plan.case_count),
            ("attempts", plan.attempts),
            ("baseline", None if plan.baseline is None else "yes" if plan.baseline else "no"),
            ("concurrency", plan.concurrency),
            ("max-agents", plan.max_agents),
            ("timeout", None if plan.timeout_multiplier is None else f"{plan.timeout_multiplier:g}x"),
            ("matrix-trials", plan.matrix_trials),
            ("preflight-trials", plan.preflight_trials),
            ("containers", plan.total_containers),
            (
                "task-timeout",
                None if plan.task_timeout_seconds is None else f"{plan.task_timeout_seconds:g}s",
            ),
        )
        updates = [f"{label}={value}" for label, value in values if value is not None]
        if not first_plan:
            updates.insert(0, f"agents/models={agents or 'none'}")
            if plan.provider:
                updates.insert(1, f"provider={self._safe_text(plan.provider)}")
        if updates:
            label = "plan" if first_plan else "plan update"
            self._write_line(f"  {label}: {', '.join(updates)}")

    def _elapsed(self) -> str:
        seconds = max(0, int(monotonic() - self._started_at)) if self._started_at is not None else 0
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self._refresh_interval):
            self.heartbeat()

    def _safe_text(self, value: object) -> str:
        return redact_progress_detail(value, secret_values=self._secret_values)

    def _event_detail(self, event: ProgressEvent) -> str:
        return self._safe_text(event.detail) if event.detail else ""

    def _write_line(self, line: str) -> None:
        self._stream.write(f"{line}\n")
        self._stream.flush()


class RichProgressReporter(PlainProgressReporter):
    """Interactive Rich Live table selected for terminal output."""

    def __init__(self, *, stream: TextIO | None = None, refresh_interval: float = 1.0) -> None:
        super().__init__(stream=stream, refresh_interval=refresh_interval)
        from rich.console import Console
        from rich.live import Live

        self._console = Console(file=self._stream)
        self._live_factory = Live
        self._live = None
        self._live_plan: Tier3RunPlan | None = None
        self._live_events: dict[str, tuple[ProgressEvent, str | None]] = {}

    def emit(self, event: ProgressEvent) -> None:
        with self._lock:
            detail = self._event_detail(event)
            safe_event = ProgressEvent(
                stage=event.stage,
                state=event.state,
                detail=detail or None,
            )
            if event.state == "running":
                self._active[event.stage] = detail
                finished_at = None
            else:
                self._active.pop(event.stage, None)
                finished_at = self._elapsed()
            self._live_events[event.stage] = (safe_event, finished_at)
            self._refresh_live()

    def heartbeat(self) -> None:
        with self._lock:
            if self._active:
                self._refresh_live()

    def close(self) -> None:
        live = self._live
        try:
            super().close()
        finally:
            with self._lock:
                if live is not None and live is self._live:
                    try:
                        live.stop()
                    finally:
                        self._live = None
                        self._live_plan = None
                        self._live_events.clear()

    def _render_plan(self, plan: Tier3RunPlan) -> None:
        self._live_plan = plan
        table = self._build_live_table()
        if self._live is None:
            self._live = self._live_factory(
                table,
                console=self._console,
                auto_refresh=True,
                refresh_per_second=8,
                transient=False,
            )
            self._live.start(refresh=True)
        else:
            self._live.update(table, refresh=True)

    def _build_live_table(self):
        from rich import box
        from rich.spinner import Spinner
        from rich.table import Table
        from rich.text import Text

        plan = self._live_plan
        if plan is None:
            return Table(title="Harbor Run Configuration")
        mode_label = "Skill-lift evaluation" if plan.baseline is not False else "Agent evaluation"
        title = Text.assemble(
            ("Harbor Run Configuration", "bold cyan"),
            "  ·  ",
            (self._safe_text(plan.skill_name), "bold"),
            "  ·  ",
            (mode_label, "magenta"),
        )
        table = Table(title=title, box=box.ROUNDED, expand=True, show_lines=False)
        table.add_column("Stage", style="bold", no_wrap=True)
        table.add_column("State", width=12, no_wrap=True)
        table.add_column("Detail", ratio=1)
        table.add_column("Elapsed", justify="right", no_wrap=True)

        table.add_row("Environment", Text("configured", style="cyan"), Text(self._safe_text(plan.environment)), "")
        model_map = dict(plan.agent_models)
        agents = ", ".join(
            f"{self._safe_text(agent)}={self._safe_text(model_map[agent])}"
            if agent in model_map
            else self._safe_text(agent)
            for agent in plan.agents
        )
        table.add_row("Agents / models", Text("configured", style="cyan"), Text(agents or "none"), "")
        if plan.provider:
            table.add_row(
                "Provider",
                Text("configured", style="cyan"),
                Text(self._safe_text(plan.provider)),
                "",
            )
        values = (
            ("Tasks", plan.task_count),
            ("Cases", plan.case_count),
            ("Attempts", plan.attempts),
            ("Baseline", None if plan.baseline is None else "yes" if plan.baseline else "no"),
            ("Concurrency", plan.concurrency),
            ("Max agents", plan.max_agents),
            ("Timeout", None if plan.timeout_multiplier is None else f"{plan.timeout_multiplier:g}x"),
            ("Matrix trials", plan.matrix_trials),
            ("Preflight trials", plan.preflight_trials),
            ("Containers", plan.total_containers),
            (
                "Task timeout",
                None if plan.task_timeout_seconds is None else f"{plan.task_timeout_seconds:g}s",
            ),
        )
        known = " · ".join(f"{label} {value}" for label, value in values if value is not None)
        if known:
            table.add_row("Run plan", Text("configured", style="cyan"), Text(known), "")
        state_styles = {
            "ready": "green",
            "complete": "green",
            "failed": "bold red",
            "degraded": "bold yellow",
            "delegated": "yellow",
            "skipped": "dim",
        }
        for stage, (event, finished_at) in self._live_events.items():
            label = (
                f"Agent {stage.removeprefix('agent:')}"
                if stage.startswith("agent:")
                else stage.replace("-", " ").title()
            )
            state = (
                Spinner("dots", text=Text("running", style="cyan"))
                if event.state == "running"
                else Text(event.state, style=state_styles.get(event.state, "white"))
            )
            table.add_row(
                Text(label),
                state,
                Text(event.detail or ""),
                Text(finished_at or self._elapsed()),
            )
        return table

    def _refresh_live(self) -> None:
        if self._live is not None:
            self._live.update(self._build_live_table(), refresh=True)

    def _write_line(self, line: str) -> None:
        self._console.print(line, markup=False, soft_wrap=True)


class NullProgressReporter:
    """No-op reporter used by API/service callers unless they opt in."""

    @property
    def is_active(self) -> bool:
        return False

    def start(self, plan: Tier3RunPlan) -> None:
        del plan

    def set_secret_values(self, values: list[str] | tuple[str, ...] | set[str]) -> None:
        del values

    def emit(self, event: ProgressEvent) -> None:
        del event

    def heartbeat(self) -> None:
        pass

    def close(self) -> None:
        pass


class SafeProgressReporter:
    """Best-effort adapter that prevents presentation failures from gating evaluation."""

    def __init__(self, reporter: ProgressReporter) -> None:
        self._reporter = reporter
        self._disabled = False
        self._started = False

    @property
    def is_active(self) -> bool:
        if self._disabled or self._started:
            return self._started
        try:
            return self._reporter.is_active
        except Exception:
            self._disable()
            return self._started

    def start(self, plan: Tier3RunPlan) -> None:
        self._started = True
        self._call(self._reporter.start, plan)

    def set_secret_values(self, values: list[str] | tuple[str, ...] | set[str]) -> None:
        self._call(self._reporter.set_secret_values, values)

    def emit(self, event: ProgressEvent) -> None:
        self._call(self._reporter.emit, event)

    def heartbeat(self) -> None:
        self._call(self._reporter.heartbeat)

    def close(self) -> None:
        try:
            self._reporter.close()
        except Exception:
            logger.debug("Tier 3 progress reporter cleanup failed", exc_info=True)
        finally:
            self._started = False
            self._disabled = False

    def _call(self, callback, *args: object) -> None:
        if self._disabled:
            return
        try:
            callback(*args)
        except Exception:
            logger.debug("Tier 3 progress reporter disabled after presentation failure", exc_info=True)
            self._disable()

    def _disable(self) -> None:
        if self._disabled:
            return
        self._disabled = True
        try:
            self._reporter.close()
        except Exception:
            logger.debug("Tier 3 progress reporter cleanup failed", exc_info=True)


def safe_progress_reporter(reporter: ProgressReporter | None) -> SafeProgressReporter:
    """Return an idempotent best-effort wrapper around a progress reporter."""
    if isinstance(reporter, SafeProgressReporter):
        return reporter
    return SafeProgressReporter(reporter or NullProgressReporter())


def create_progress_reporter(mode: ProgressMode, *, stream: TextIO | None = None) -> ProgressReporter:
    """Create the requested presentation; ``auto`` follows stream TTY state."""
    if mode not in {"auto", "rich", "plain", "off"}:
        raise ValueError("progress must be one of: auto, rich, plain, off")
    if mode == "off":
        return NullProgressReporter()
    output = stream or sys.stderr
    use_rich = mode == "rich"
    if mode == "auto" and bool(getattr(output, "isatty", lambda: False)()):
        from rich.console import Console

        console = Console(file=output)
        use_rich = console.is_terminal and not console.is_dumb_terminal
    if use_rich:
        return RichProgressReporter(stream=output)
    return PlainProgressReporter(stream=output)


__all__ = [
    "NullProgressReporter",
    "PlainProgressReporter",
    "ProgressEvent",
    "ProgressMode",
    "ProgressReporter",
    "RichProgressReporter",
    "SafeProgressReporter",
    "Tier3RunPlan",
    "create_progress_reporter",
    "redact_progress_detail",
    "safe_progress_reporter",
    "secret_values_from_environment",
]
