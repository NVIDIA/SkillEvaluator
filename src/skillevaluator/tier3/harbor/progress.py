# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Terminal presentation for Tier 3 live-evaluation progress.

The runner emits structured events through :class:`ProgressReporter`; it does
not know whether those events are shown by Rich, written as durable CI lines,
or discarded by an API caller.
"""

from __future__ import annotations

import io
import logging
import math
import re
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import monotonic
from typing import Literal, Protocol, TextIO, runtime_checkable

from skillevaluator.reporting.terminal_styles import AMBER, FAINT, GREEN, MUTED, RED, TEXT
from skillevaluator.tier3.harbor.secret_redaction import redact_secrets_in_log_line
from skillevaluator.utils.redaction import credential_uri_secret_values
from skillevaluator.utils.rich_markup import strip_terminal_controls

ProgressMode = Literal["auto", "rich", "plain", "off"]
logger = logging.getLogger(__name__)

ProgressState = Literal["running", "ready", "complete", "failed", "degraded", "delegated", "skipped"]

_TERMINAL_STATES = frozenset({"ready", "complete", "failed", "degraded", "delegated", "skipped"})
_SECRET_KEY_WORDS = r"(?:api[_-]?key|access[_-]?key|auth|bearer|credential|password|secret|token)"
_SECRET_ASSIGNMENT_RE = re.compile(
    rf"(?i)((?:(?<![A-Za-z0-9_-])(?:--?)?{_SECRET_KEY_WORDS}\b\s*[:=]|"
    rf"(?<![A-Za-z0-9]){_SECRET_KEY_WORDS}\b\s*=)\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_SECRET_ENV_NAME_RE = re.compile(r"(?i)(?:api[_-]?key|access[_-]?key|auth|credential|password|secret|token(?!s))")
# Whole environment values shorter than this are never treated as exact secrets;
# shorter credential fragments come only from URI userinfo.
MIN_EXACT_SECRET_CHARS = 4
_CREDENTIAL_URI_USERINFO_RE = re.compile(r"(?i)(?P<scheme>[a-z][a-z0-9+.-]{0,31}://)(?P<userinfo>[^\s/?#]+@)")
# Plain lines are durable CI and log output, so the still-running heartbeat is
# written once a minute; the interactive Rich frame updates elapsed times every
# second.
_PLAIN_REFRESH_INTERVAL_SECONDS = 60.0
_RICH_REFRESH_INTERVAL_SECONDS = 1.0
MIN_REFRESH_INTERVAL_SECONDS = 0.05
# One day stays below every platform's threading.TIMEOUT_MAX, beyond which the
# heartbeat thread's wait would raise OverflowError.
MAX_REFRESH_INTERVAL_SECONDS = 86400.0
# The Rich frame keeps at least this many stage rows (more when more agents
# run), padding with blank rows, so its height never changes mid-run. A frame
# that grows or shrinks between redraws leaves stale or jumbled lines behind.
_LIVE_MIN_EVENT_ROWS = 6
_LIVE_STATE_COLUMN_WIDTH = 12
# The rounded table's five borders and one space of padding on each side of
# its four columns.
_LIVE_TABLE_CHROME_WIDTH = 5 + 4 * 2
# The final frame wraps long details only when its Detail column is at least
# this wide. In a narrower column every detail would fold into a tall stack of
# short pieces, so there the rows stay cut on one line as while running.
_FINAL_DETAIL_MIN_WIDTH = 20
_DISPLAY_STOPPED_NOTICE = "Tier 3 progress display stopped after a drawing error; the run continues."
# Printed when the run ends instead, when the terminal refused the line above
# or Live's stop.
_DISPLAY_STOPPED_LATE_NOTICE = "Tier 3 progress display stopped after a drawing error; the run went on without it."
# Frames of a running stage's spinner, as in Rich's "dots". The view redraws
# only on progress events and heartbeats, so a running stage moves on one frame
# with each redraw. Rich's own spinner picks its frame from the clock, so it
# would stand still whenever redraws come a whole number of its cycles apart.
_RUNNING_FRAMES = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"


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
    with_skill: bool | None = None
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
class _FailedDisplayStop:
    """What reached the terminal when the live view stopped after a drawing failure."""

    stop_refused: bool
    notice_shown: bool
    # The terminal refused the notice, but the stream kept it to send later,
    # as standard error's buffer does.
    notice_kept: bool = False


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
    text = " ".join(strip_terminal_controls(str(detail)).split())
    if "://" in text:
        text = _CREDENTIAL_URI_USERINFO_RE.sub(r"\g<scheme><redacted>@", text)
    for secret in sorted(secret_values or (), key=len, reverse=True):
        if len(secret) >= MIN_EXACT_SECRET_CHARS:
            text = text.replace(secret, "<redacted>")
        elif secret:
            # Exact credential-derived fragments can legitimately be short
            # (for example proxy userinfo). Redact them only as standalone
            # tokens so a one-character secret cannot erase normal prose.
            text = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(secret)}(?![A-Za-z0-9_])",
                "<redacted>",
                text,
            )
    text = redact_secrets_in_log_line(text, extra_secret_values=secret_values)
    return _SECRET_ASSIGNMENT_RE.sub(r"\1<redacted>", text)


def secret_values_from_environment(environment: Mapping[str, str]) -> set[str]:
    """Extract exact credential values without treating every env value as secret."""
    protected: set[str] = set()
    for name, value in environment.items():
        if not value:
            continue
        rendered = str(value)
        # Short values of credential-named variables are flags such as "1" or
        # "true", not credentials; redacting them would erase counts in prose.
        if _SECRET_ENV_NAME_RE.search(name) and len(rendered) >= MIN_EXACT_SECRET_CHARS:
            protected.add(rendered)
        if name.upper().endswith("_PROXY") or "://" in rendered:
            protected.update(
                credential_uri_secret_values(
                    rendered,
                    allow_schemeless=name.upper().endswith("_PROXY"),
                )
            )
    return protected


class PlainProgressReporter:
    """Durable line-oriented reporter for redirected output and CI logs."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        refresh_interval: float = _PLAIN_REFRESH_INTERVAL_SECONDS,
    ) -> None:
        interval = float(refresh_interval)
        # A NaN wait returns at once, so the heartbeat thread would spin.
        if not math.isfinite(interval):
            raise ValueError("refresh_interval must be a finite number of seconds")
        self._stream = stream or sys.stderr
        self._refresh_interval = min(max(interval, MIN_REFRESH_INTERVAL_SECONDS), MAX_REFRESH_INTERVAL_SECONDS)
        self._started_at: float | None = None
        self._active: dict[str, str] = {}
        self._secret_values: set[str] = set()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._plan_rendered = False
        self._last_progress_at: float | None = None
        self._last_events: dict[str, tuple[str, str]] = {}
        self._failure_callback: Callable[[], None] | None = None

    @property
    def is_active(self) -> bool:
        return self._started_at is not None and not self._stop.is_set()

    def start_progress(self) -> None:
        """Start the shared lifecycle before a plan (for the validation tiers)."""
        with self._lock:
            if self._started_at is None:
                self._started_at = monotonic()
                self._last_progress_at = self._started_at
                self._stop.clear()
                self._thread = threading.Thread(
                    target=self._heartbeat_loop,
                    name="tier3-progress-heartbeat",
                    daemon=True,
                )
                self._thread.start()

    def start(self, plan: Tier3RunPlan) -> None:
        self.start_progress()
        with self._lock:
            self._render_plan(plan)

    def set_secret_values(self, values: list[str] | tuple[str, ...] | set[str]) -> None:
        with self._lock:
            self._secret_values.update(value for value in values if value)

    def set_failure_callback(self, callback: Callable[[], None] | None) -> None:
        """Notify a safety wrapper when the reporter's own heartbeat thread fails."""
        self._failure_callback = callback

    def emit(self, event: ProgressEvent) -> None:
        with self._lock:
            detail = self._event_detail(event)
            stage = self._safe_text(event.stage)
            current = (event.state, detail)
            if self._last_events.get(stage) == current:
                return
            self._last_events[stage] = current
            self._last_progress_at = monotonic()
            if event.state == "running":
                self._active[stage] = detail
            elif event.state in _TERMINAL_STATES:
                self._active.pop(stage, None)
            suffix = f" - {detail}" if detail else ""
            self._write_line(f"[{self._elapsed()}] {stage}: {event.state}{suffix}")

    def heartbeat(self) -> None:
        with self._lock:
            if not self._active or self._stop.is_set():
                return
            if self._last_progress_at is not None and monotonic() - self._last_progress_at < self._refresh_interval:
                return
            self._last_progress_at = monotonic()
            stages = ", ".join(sorted(self._active))
            self._write_line(f"[{self._elapsed()}] still running: {stages}")

    def close(self) -> None:
        self._close(after_failure=False)

    def close_after_failure(self) -> None:
        """Stop after a presentation failure without marking running stages failed.

        Only the display failed. The run goes on, so a stage that is still
        running has not been interrupted.
        """
        self._close(after_failure=True)

    def _close(self, *, after_failure: bool) -> None:
        try:
            with self._lock:
                # Decided under the same lock hold that would mark the stages,
                # so an update that arrives while a failed display stops is
                # never reported as interrupted.
                if not after_failure:
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
                self._last_events.clear()
                self._last_progress_at = None

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
            ("with-skill", None if plan.with_skill is None else "yes" if plan.with_skill else "no"),
            ("baseline", None if plan.baseline is None else "yes" if plan.baseline else "no"),
            ("concurrency", plan.concurrency),
            ("max-agents", plan.max_agents),
            ("timeout", None if plan.timeout_multiplier is None else f"{plan.timeout_multiplier:g}x"),
            ("matrix-trials", plan.matrix_trials),
            ("preflight-trials", plan.preflight_trials),
            ("containers", plan.total_containers),
            (
                "max-scaled-top-level-agent-timeout",
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
        while True:
            delay = self._refresh_interval
            with self._lock:
                if self._active and self._last_progress_at is not None:
                    delay = max(MIN_REFRESH_INTERVAL_SECONDS, delay - (monotonic() - self._last_progress_at))
            if self._stop.wait(delay):
                return
            try:
                self.heartbeat()
            except Exception:
                # A redraw failure here must turn progress off through the
                # safety wrapper, which stops the live display, instead of
                # killing this thread and leaving a half-drawn frame behind.
                # That includes a write a non-blocking terminal refuses: Rich
                # keeps the refused frame and sends it again before the next
                # one, so skipping it would only grow that backlog.
                logger.debug("Tier 3 progress heartbeat failed", exc_info=True)
                if self._stop.is_set():
                    # close() is already stopping this reporter.
                    return
                callback = self._failure_callback
                if callback is None:
                    self._stop.set()
                    return
                try:
                    callback()
                except Exception:
                    logger.debug("Tier 3 progress failure callback failed", exc_info=True)
                return

    def _safe_text(self, value: object) -> str:
        return redact_progress_detail(value, secret_values=self._secret_values)

    def _event_detail(self, event: ProgressEvent) -> str:
        return self._safe_text(event.detail) if event.detail else ""

    def _write_line(self, line: str) -> None:
        if self._stop.is_set():
            return
        try:
            self._stream.write(f"{line}\n")
            self._stream.flush()
        except (OSError, ValueError):
            # A closed pipe must stop the heartbeat without gating evaluation.
            self._stop.set()


class _FinalLiveFrame:
    """The last Rich frame, which stays in the scrollback once Live stops.

    It draws ``wrapped`` when the width it is drawn at leaves the Detail column
    at least ``_FINAL_DETAIL_MIN_WIDTH`` wide, and ``cut`` otherwise.
    """

    def __init__(self, *, wrapped: object, cut: object, min_width: int) -> None:
        self.wrapped = wrapped
        self.cut = cut
        self.min_width = min_width

    def __rich_console__(self, _console, options):  # rich renderable protocol
        yield self.wrapped if options.max_width >= self.min_width else self.cut


class RichProgressReporter(PlainProgressReporter):
    """Interactive Rich Live table selected for terminal output."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        refresh_interval: float = _RICH_REFRESH_INTERVAL_SECONDS,
    ) -> None:
        super().__init__(stream=stream, refresh_interval=refresh_interval)
        from rich.console import Console
        from rich.live import Live

        self._console = Console(file=self._stream)
        self._live_factory = Live
        self._live = None
        self._live_plan: Tier3RunPlan | None = None
        self._live_events: dict[str, tuple[ProgressEvent, str | None]] = {}
        # How many times each running stage has been drawn since it started
        # running, which picks its spinner frame.
        self._live_spinner_steps: dict[str, int] = {}
        self._live_event_slots = _LIVE_MIN_EVENT_ROWS
        self._printed_plan_lines: dict[str, str] = {}
        # Set, under the same lock hold that stops Live, when a drawing failure
        # stopped it. The run's close() finishes what the terminal refused.
        self._failed_stop: _FailedDisplayStop | None = None

    def start_progress(self) -> None:
        with self._lock:
            if self._started_at is None and self._failed_stop is not None:
                # The run whose display failed was never closed. Its refused
                # writes and its notice do not belong to this run.
                self._failed_stop = None
                self._drop_refused_writes()
            super().start_progress()

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
                self._live_spinner_steps.pop(event.stage, None)
                finished_at = self._elapsed()
            # Move the stage to the end so the recent-event window shows it.
            self._live_events.pop(event.stage, None)
            self._live_events[event.stage] = (safe_event, finished_at)
            self._refresh_live()

    def heartbeat(self) -> None:
        with self._lock:
            self._last_progress_at = monotonic()
            if self._active:
                self._refresh_live()

    def _close(self, *, after_failure: bool) -> None:
        live = self._live
        try:
            super()._close(after_failure=after_failure)
        finally:
            with self._lock:
                try:
                    if live is not None and live is self._live:
                        if after_failure:
                            self._stop_live_after_failure(live)
                        else:
                            try:
                                # Once Live stops, its last frame stays in the
                                # scrollback and no longer needs a fixed
                                # height, so it lists every stage in full.
                                live.update(self._build_live_table(final=True), refresh=False)
                            finally:
                                live.stop()
                    elif (
                        after_failure
                        and self._live is None
                        and self._live_plan is not None
                        and self._failed_stop is None
                    ):
                        # Drawing failed before Live started, for example on a
                        # plan line printed above where the frame goes. There
                        # is no frame to stop, but progress is off for the run
                        # all the same, so the notice is due. A stream that
                        # refuses a write can still have taken it, as standard
                        # error's buffer does, and then sends it with the
                        # notice. Rich's copy of the refused line is dropped
                        # first so the line is not printed twice.
                        self._drop_refused_writes()
                        self._stop_live_after_failure(None)
                finally:
                    # Reset even when Live never started, so a failed start
                    # leaves no plan or rows behind for the next run.
                    if live is self._live:
                        self._live = None
                    self._live_plan = None
                    self._live_events.clear()
                    self._live_spinner_steps.clear()
                    self._live_event_slots = _LIVE_MIN_EVENT_ROWS
                    self._printed_plan_lines.clear()
                    if not after_failure:
                        failed_stop, self._failed_stop = self._failed_stop, None
                        if failed_stop is not None:
                            self._finish_failed_stop(failed_stop)
                        # Whatever the terminal still refuses as the run ends,
                        # such as the final frame and Live's stop or the late
                        # notice, belongs to no later run on this reporter.
                        self._drop_refused_writes()

    def _stop_live_after_failure(self, live) -> None:
        """Stop the live view after a drawing failure and say that the run goes on.

        The full final frame could fail the same way, so Live keeps the last
        frame it drew. ``live`` is ``None`` when drawing failed before Live
        started, and then only the notice is printed. Runs under the reporter
        lock and records the outcome in the same lock hold, so the run's
        close() either stops Live itself or finds the record.
        """
        # The console holds no refused output once Live stops here, and the
        # caller drops it before Live started. Live that stopped itself when
        # its first frame failed leaves that frame and stop held when the
        # terminal refused that stop too. They then go out with the notice,
        # and a stream's buffer may keep only part of them. Nothing here
        # tells whether that stop went out, so after a failed first frame
        # the run's close() writes the late notice either way, although a
        # stream that took the stop and kept the notice sends both.
        notice_alone = live is None or bool(getattr(live, "is_started", False))
        if live is not None:
            try:
                live.stop()
            except Exception:
                # A terminal that refused the redraw can refuse Live's stop too,
                # and with it the write that shows the cursor again.
                logger.debug("Tier 3 progress display stop failed", exc_info=True)
                self._failed_stop = _FailedDisplayStop(stop_refused=True, notice_shown=False)
                return
        try:
            # A stream can keep what the terminal refused, as standard error's
            # buffer does, and send it later. It would then also keep a refused
            # notice and send it before the late one, so while the terminal
            # still refuses what the stream holds, the notice waits for the
            # run's close().
            self._console.file.flush()
        except Exception:
            logger.debug("Tier 3 progress stop notice failed", exc_info=True)
            self._failed_stop = _FailedDisplayStop(stop_refused=False, notice_shown=False)
            return
        try:
            # Live keeps the last frame it drew, with its stages still running,
            # so explain why it no longer updates.
            self._write_line(_DISPLAY_STOPPED_NOTICE)
        except Exception as error:
            logger.debug("Tier 3 progress stop notice failed", exc_info=True)
            # The flush above emptied the stream's buffer, so a buffer that
            # keeps refused writes now holds the notice and sends it later.
            # The run's close() then sends it instead of the late one. A
            # buffer smaller than the notice would keep only part of it, but
            # standard error's buffer is many times the notice's size.
            notice_kept = notice_alone and isinstance(error, BlockingIOError) and self._stream_keeps_refused_writes()
            self._failed_stop = _FailedDisplayStop(stop_refused=False, notice_shown=False, notice_kept=notice_kept)
        else:
            self._failed_stop = _FailedDisplayStop(stop_refused=False, notice_shown=True)

    def _stream_keeps_refused_writes(self) -> bool:
        """Whether the console's stream keeps a write the terminal refused, as standard error's buffer does."""
        try:
            return isinstance(getattr(self._console.file, "buffer", None), (io.BufferedWriter, io.BufferedRandom))
        except Exception:
            return False

    def _drop_refused_writes(self) -> str:
        """Discard the output this thread's Rich console holds from refused writes, and return it.

        Rich keeps a refused write for its thread and sends it before that
        thread's next one. A refused frame or stop starts by moving the cursor
        up over the frame that was on screen when drawing failed, so sent any
        later it would erase and overwrite whatever was printed since. A
        thread that ended took its refused writes with it.
        """
        try:
            self._console.begin_capture()
            return self._console.end_capture()
        except Exception:
            logger.debug("Tier 3 progress refused output could not be dropped", exc_info=True)
            return ""

    def _finish_failed_stop(self, failed_stop: _FailedDisplayStop) -> None:
        """Show the cursor and the stop notice when the terminal refused them after a drawing failure.

        Runs on the thread that closes the run, once the terminal may take
        writes again.
        """
        from rich.control import Control

        dropped = self._drop_refused_writes()
        # Whether the frame's last line is ended before the late notice.
        line_ended = True
        try:
            if failed_stop.stop_refused:
                # Live hides the cursor while it draws and shows it again when
                # it stops, then ends the frame's last line.
                line_ended = not self._console.is_terminal
                self._console.show_cursor(True)
                if self._console.is_terminal:
                    # The console keeps a line the terminal refuses and sends
                    # it before its next write, so the line ends before the
                    # notice either way.
                    line_ended = True
                    self._console.line()
            elif str(Control.show_cursor(True)) in dropped:
                # Live.start() stops Live itself when its first frame fails,
                # so the stop above had nothing left to do. The terminal
                # refused that earlier stop too, and its write that shows the
                # cursor was just dropped. No frame reached the terminal, so
                # there is no last line to end.
                self._console.show_cursor(True)
        except Exception:
            # The run's close() then drops what the terminal still refuses.
            logger.debug("Tier 3 progress terminal restore failed", exc_info=True)
        if failed_stop.notice_shown:
            return
        if failed_stop.notice_kept:
            try:
                # The stream kept the notice the terminal refused. It goes out
                # now, or with the stream's next write if the terminal still
                # refuses it, and a late notice would only repeat it.
                self._console.file.flush()
            except Exception:
                logger.debug("Tier 3 progress kept stop notice could not be sent", exc_info=True)
            return
        try:
            # Written even when the terminal refused the cursor just now: a
            # stream that keeps what the terminal refused, as standard error
            # does, can send the notice with it later if its buffer has room.
            # A refused cursor write leaves the line above unwritten, and the
            # stream may have kept only the start of the refused stop, without
            # its newline, so the notice then starts a new line of its own.
            # Where the stream kept the whole stop, a blank line comes first,
            # as it does when the terminal takes the cursor write.
            if line_ended:
                self._write_line(_DISPLAY_STOPPED_LATE_NOTICE)
            else:
                self._write_line(f"\n{_DISPLAY_STOPPED_LATE_NOTICE}")
        except Exception:
            logger.debug("Tier 3 progress late stop notice failed", exc_info=True)

    def _render_plan(self, plan: Tier3RunPlan) -> None:
        self._live_plan = plan
        # The frame cuts long rows to keep its height, so each plan line is
        # also printed in full above it whenever that line changes.
        for label, text in self._plan_lines(plan):
            if self._printed_plan_lines.get(label) != text:
                self._printed_plan_lines[label] = text
                self._write_line(f"{label}: {text}")
        table = self._build_live_table()
        if self._live is None:
            self._live = self._live_factory(
                table,
                console=self._console,
                # Rich never redraws on its own timer. The frame is redrawn when
                # the plan or a stage changes, on each heartbeat, and when other
                # output is printed above it.
                auto_refresh=False,
                transient=False,
            )
            self._live.start(refresh=True)
        else:
            self._live.update(table, refresh=True)

    def _plan_agents_text(self, plan: Tier3RunPlan) -> str:
        model_map = dict(plan.agent_models)
        return ", ".join(
            f"{self._safe_text(agent)}={self._safe_text(model_map[agent])}"
            if agent in model_map
            else self._safe_text(agent)
            for agent in plan.agents
        )

    def _plan_values_text(self, plan: Tier3RunPlan) -> str:
        values = (
            ("Tasks", plan.task_count),
            ("Cases", plan.case_count),
            ("Attempts", plan.attempts),
            ("With skill", None if plan.with_skill is None else "yes" if plan.with_skill else "no"),
            ("Baseline", None if plan.baseline is None else "yes" if plan.baseline else "no"),
            ("Concurrency", plan.concurrency),
            ("Max agents", plan.max_agents),
            ("Timeout", None if plan.timeout_multiplier is None else f"{plan.timeout_multiplier:g}x"),
            ("Matrix trials", plan.matrix_trials),
            ("Preflight trials", plan.preflight_trials),
            ("Containers", plan.total_containers),
            (
                "Max scaled top-level agent timeout",
                None if plan.task_timeout_seconds is None else f"{plan.task_timeout_seconds:g}s",
            ),
        )
        return " · ".join(f"{label} {value}" for label, value in values if value is not None)

    def _plan_lines(self, plan: Tier3RunPlan) -> list[tuple[str, str]]:
        lines = []
        agents = self._plan_agents_text(plan)
        if agents:
            lines.append(("Agents / models", agents))
        if plan.provider:
            lines.append(("Provider", self._safe_text(plan.provider)))
        known = self._plan_values_text(plan)
        if known:
            lines.append(("Run plan", known))
        return lines

    def _build_live_table(self, *, final: bool = False):
        from rich import box
        from rich.cells import cell_len
        from rich.table import Table
        from rich.text import Text

        plan = self._live_plan
        if plan is None:
            return Table(title="Harbor Run Configuration")
        if plan.with_skill is False:
            mode_label = "Baseline-only evaluation"
        else:
            mode_label = "Skill-lift evaluation" if plan.baseline is not False else "Agent evaluation"
        title = Text.assemble(
            ("Harbor Run Configuration", f"bold {TEXT}"),
            "  ·  ",
            (self._safe_text(plan.skill_name), "bold"),
            "  ·  ",
            (mode_label, MUTED),
            overflow="ellipsis",
            no_wrap=True,
        )
        rows = [
            ("Environment", Text("configured", style=TEXT), Text(self._safe_text(plan.environment)), ""),
        ]
        agents = self._plan_agents_text(plan)
        rows.append(("Agents / models", Text("configured", style=TEXT), Text(agents or "none"), ""))

        def placeholder(waiting_detail: str) -> tuple[Text, Text]:
            # The row is always present so a plan update does not add rows. A
            # finished run's frame stays on screen, so there it no longer says
            # the value is still on its way.
            if final:
                return Text("-", style=MUTED), Text("not resolved", style=MUTED)
            return Text("pending", style=MUTED), Text(waiting_detail)

        if plan.provider:
            provider_state, provider_detail = Text("configured", style=TEXT), Text(self._safe_text(plan.provider))
        else:
            provider_state, provider_detail = placeholder("resolving")
        rows.append(("Provider", provider_state, provider_detail, ""))
        known = self._plan_values_text(plan)
        if known:
            plan_state, plan_detail = Text("configured", style=TEXT), Text(known)
        else:
            plan_state, plan_detail = placeholder("preparing")
        rows.append(("Run plan", plan_state, plan_detail, ""))
        state_styles = {
            "ready": GREEN,
            "complete": GREEN,
            "failed": f"bold {RED}",
            "degraded": f"bold {AMBER}",
            "delegated": AMBER,
            "skipped": "dim",
        }
        # Show only the most recently updated stages in a fixed number of rows.
        # The slot count only grows (to fit every agent), so the frame never
        # shrinks either. The final frame lists every stage, without padding.
        self._live_event_slots = max(self._live_event_slots, len(plan.agents))
        event_slots = self._live_event_slots
        events = list(self._live_events.items())
        shown_events = events if final else events[-event_slots:]
        for stage, (event, finished_at) in shown_events:
            label = (
                f"Agent {stage.removeprefix('agent:')}"
                if stage.startswith("agent:")
                else stage.replace("-", " ").title()
            )
            if event.state == "running":
                step = self._live_spinner_steps.get(stage, 0)
                self._live_spinner_steps[stage] = step + 1
                state = Text.assemble(_RUNNING_FRAMES[step % len(_RUNNING_FRAMES)], " ", ("running", TEXT))
            else:
                state = Text(event.state, style=state_styles.get(event.state, "white"))
            rows.append(
                (
                    Text(label),
                    state,
                    Text(event.detail or ""),
                    Text(finished_at or self._elapsed()),
                )
            )
        if not final:
            rows.extend(("", "", "", "") for _ in range(event_slots - len(shown_events)))

        def table(*, wrap_detail: bool) -> Table:
            built = Table(title=title, box=box.ROUNDED, border_style=FAINT, expand=True, show_lines=False)
            # One line per row: long values are cut with an ellipsis rather than
            # wrapped, so the frame height does not depend on the text. Only the
            # final frame may wrap its details.
            built.add_column("Stage", style="bold", no_wrap=True, overflow="ellipsis")
            built.add_column("State", width=_LIVE_STATE_COLUMN_WIDTH, no_wrap=True, overflow="ellipsis")
            if wrap_detail:
                built.add_column("Detail", ratio=1, overflow="fold")
            else:
                built.add_column("Detail", ratio=1, no_wrap=True, overflow="ellipsis")
            built.add_column("Elapsed", justify="right", no_wrap=True, overflow="ellipsis")
            for row in rows:
                built.add_row(*row)
            return built

        if not final:
            return table(wrap_detail=False)
        # The final frame stays in the scrollback, so its details wrap in full
        # where the Detail column, which gets the width the other columns
        # leave, is wide enough.
        stage_width = max(cell_len(str(cell)) for cell in ["Stage", *(row[0] for row in rows)])
        elapsed_width = max(cell_len(str(cell)) for cell in ["Elapsed", *(row[3] for row in rows)])
        return _FinalLiveFrame(
            wrapped=table(wrap_detail=True),
            cut=table(wrap_detail=False),
            min_width=(
                _LIVE_TABLE_CHROME_WIDTH
                + stage_width
                + _LIVE_STATE_COLUMN_WIDTH
                + elapsed_width
                + _FINAL_DETAIL_MIN_WIDTH
            ),
        )

    def _refresh_live(self) -> None:
        if self._live is not None:
            self._live.update(self._build_live_table(), refresh=True)

    def _write_line(self, line: str) -> None:
        # Printed as is: model IDs and providers can contain ":name:", which
        # Rich would otherwise replace with an emoji.
        self._console.print(line, markup=False, highlight=False, emoji=False, soft_wrap=True)


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
        # start() holds this lock while the reporter draws a plan, so a failure
        # reported from the reporter's heartbeat thread waits for that plan
        # update and then stops what it drew, instead of letting it start a new
        # live view after progress was turned off.
        self._lock = threading.RLock()
        # A reporter's own heartbeat thread reports failures here, so a redraw
        # error there turns progress off just like one on the caller's thread.
        try:
            set_failure_callback = getattr(reporter, "set_failure_callback", None)
            if callable(set_failure_callback):
                set_failure_callback(self._disable)
        except Exception:
            logger.debug("Tier 3 progress reporter failure callback registration failed", exc_info=True)

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
        with self._lock:
            if self._disabled:
                return
            self._started = True
            try:
                self._reporter.start(plan)
                return
            except Exception:
                logger.debug("Tier 3 progress reporter disabled after presentation failure", exc_info=True)
        # Closed outside the lock: closing joins the heartbeat thread, which
        # may be waiting for the lock to report its own failure.
        self._disable()

    def set_secret_values(self, values: list[str] | tuple[str, ...] | set[str]) -> None:
        self._call(self._reporter.set_secret_values, values)

    def emit(self, event: ProgressEvent) -> None:
        self._call(self._reporter.emit, event)

    def heartbeat(self) -> None:
        self._call(self._reporter.heartbeat)

    def close(self) -> None:
        # The lock is not held here: closing joins the heartbeat thread, which
        # may be waiting for the lock to report a failure.
        try:
            self._reporter.close()
        except Exception:
            logger.debug("Tier 3 progress reporter cleanup failed", exc_info=True)
        finally:
            with self._lock:
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
        with self._lock:
            if self._disabled:
                return
            self._disabled = True
            # A disabled reporter shows nothing, so it is no longer active.
            self._started = False
        # Only the display failed, so a reporter that can stop without marking
        # its running stages failed does so.
        try:
            close_after_failure = getattr(self._reporter, "close_after_failure", None)
        except Exception:
            close_after_failure = None
        try:
            if callable(close_after_failure):
                close_after_failure()
            else:
                self._reporter.close()
        except Exception:
            logger.debug("Tier 3 progress reporter cleanup failed", exc_info=True)


def safe_progress_reporter(reporter: ProgressReporter | None) -> SafeProgressReporter:
    """Return an idempotent best-effort wrapper around a progress reporter."""
    if isinstance(reporter, SafeProgressReporter):
        return reporter
    return SafeProgressReporter(reporter or NullProgressReporter())


def create_progress_reporter(
    mode: ProgressMode,
    *,
    stream: TextIO | None = None,
    refresh_interval: float | None = None,
) -> ProgressReporter:
    """Create the requested presentation; ``auto`` follows stream TTY state.

    ``refresh_interval`` overrides the heartbeat period, which otherwise
    defaults to one second for Rich and one minute for plain lines.
    """
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
    reporter_options = {} if refresh_interval is None else {"refresh_interval": refresh_interval}
    if use_rich:
        return RichProgressReporter(stream=output, **reporter_options)
    return PlainProgressReporter(stream=output, **reporter_options)


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
