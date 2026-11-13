# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quiet-mode console UI for ``skillevaluator validate``.

Renders the three-tier pipeline as a compact, glanceable display: a header
panel, a pipeline gate line whose nodes fill as tiers complete (and whose
track breaks at a failing gate), one bordered section per tier that carries
its details (check ticker and results for Tier 1, dedup stages for Tier 2,
run configuration and skill-lift bars for Tier 3), then a verdict panel with
the single most actionable fix and a footer pointing at the detailed reports.

Color is NVIDIA-themed and deliberately minimal: green (#76B900) marks
success and identity, red marks failure, everything else stays in quiet
grays. On a live terminal the gate and the active tier animate via
``rich.live``; when stdout is not a terminal (CI, pipes) each element prints
once, statically, as it completes.

``validate --verbose`` bypasses this module entirely and keeps the historical
full-detail stream.

Skill names, finding messages, failure reasons and paths are untrusted. The
view strips terminal control sequences from every string it renders, and its
consoles are created with ``emoji=False`` so ``:name:`` codes print literally.
"""

from __future__ import annotations

import math
import os
import re
import sys
import textwrap
import threading
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from rich import box
from rich.cells import cell_len
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from skillevaluator.constants import CONTENT_DEDUP_LLM_REFUSED_CHECK
from skillevaluator.logging_config import route_logs_to
from skillevaluator.reporting.base import (
    get_skip_reason,
    source_validation_errors,
    source_validation_preview,
)
from skillevaluator.reporting.presentation import result_execution_status
from skillevaluator.reporting.terminal_styles import (
    AMBER,
    BRIGHT,
    FAINT,
    GREEN,
    INK,
    MUTED,
    RED,
    TEXT,
    artifact_text,
    grade_style,
    lift_style,
)
from skillevaluator.skillspector_references import reference_summary_message
from skillevaluator.utils.rich_markup import strip_terminal_controls

if TYPE_CHECKING:
    from collections.abc import Iterable

    from skillevaluator.models.result import ValidationResult

_TIER2_SCAN_FAILURE_CHECKS = frozenset(
    {
        CONTENT_DEDUP_LLM_REFUSED_CHECK,
        "invalid_skill_root",
        "invalid_text_encoding",
        "path_access_error",
        "secure_open_unavailable",
        "unsafe_path",
    }
)

WIDTH = 98
LABEL = 12
DUR = 8
BAR = 10
_SKIP_REASON_MAX_CHARS = 110
# Setup instructions in a skip reason span several short lines (provider
# variables, a docs link, the flag that skips the tier); keep them readable.
_SKIP_REASON_MAX_LINES = 12
_SKIP_REASON_SOURCE_MAX_CHARS = 4096
_TIER3_TEXT_MAX_CHARS = 110
SPIN = "◐◓◑◒"

PENDING = "pending"
RUNNING = "running"
PASS = "pass"
FAIL = "fail"
SKIP = "skip"
WARN = "warn"


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 59.95:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes}m {secs:02d}s"


def engine_feed_rows(lines: list[str]) -> list[TierRow]:
    """Render the engine progress tail as muted rows under the run config."""
    rows: list[TierRow] = []
    for i, line in enumerate(lines):
        rows.append(TierRow("engine" if i == 0 else "", [(line, MUTED)]))
    return rows


class ViewProgressReporter:
    """Adapts Tier 3 engine :class:`ProgressEvent` emissions to the view.

    Keeps a rolling tail of the most recent meaningful stage transitions and
    forwards them (as plain feed lines) to a callback; details pass through
    :func:`redact_progress_detail` so secrets never reach the terminal.
    """

    def __init__(
        self,
        on_tail,
        max_lines: int = 4,
        *,
        on_cases=None,
        pass_threshold: float | None = None,
        poll_seconds: float = 2.0,
        progress_reporter=None,
    ):
        self._on_tail = on_tail
        self._max_lines = max_lines
        self._tail: list[str] = []
        self._secrets: set[str] = set()
        self._on_cases = on_cases
        self._pass_threshold = 0.5 if pass_threshold is None else pass_threshold
        self._poll_seconds = poll_seconds
        self._plan = None
        self._progress_reporter = progress_reporter
        self._last_case_rows = None
        self._last_event = None
        self._stop = threading.Event()
        self._poller: threading.Thread | None = None

    @property
    def is_active(self) -> bool:
        return True

    def start(self, plan) -> None:
        """Track the run plan; once it names the run directory, poll case results."""
        if self._progress_reporter is not None:
            self._progress_reporter.start(plan)
        if self._on_cases is None or not getattr(plan, "output_dir", None):
            return
        self._plan = plan
        self.poll_cases()
        if self._poller is None:
            self._poller = threading.Thread(target=self._poll_loop, name="tier3-case-poller", daemon=True)
            self._poller.start()

    def poll_cases(self) -> None:
        """Push live per-case rows for the current plan to the view."""
        from skillevaluator.tier3.case_summary import live_case_marks

        plan = self._plan
        if plan is None or self._on_cases is None:
            return
        try:
            marks, running = live_case_marks(
                Path(plan.output_dir), list(plan.agents), pass_threshold=self._pass_threshold
            )
        except OSError:
            return
        done = sum(len(m) for conditions in marks.values() for cases in conditions.values() for m in cases.values())
        conditions = ("with_skill",) if plan.baseline is False else ("with_skill", "without_skill")
        rows = tier3_case_rows(
            marks,
            agents=list(plan.agents),
            conditions=conditions,
            k=plan.attempts,
            running=running,
            progress=(done, plan.matrix_trials) if plan.matrix_trials else None,
            show_agent=False,
        )
        if rows and rows != self._last_case_rows:
            self._last_case_rows = rows
            self._on_cases(rows)

    def _poll_loop(self) -> None:
        while not self._stop.wait(self._poll_seconds):
            try:
                self.poll_cases()
            except Exception:  # a display refresh must never break the run
                continue

    def set_secret_values(self, values) -> None:
        self._secrets.update(str(v) for v in values if v)
        if self._progress_reporter is not None:
            self._progress_reporter.set_secret_values(values)

    def emit(self, event) -> None:
        from skillevaluator.tier3.harbor.progress import redact_progress_detail

        if self._progress_reporter is not None:
            self._progress_reporter.emit(event)
        if event == self._last_event:
            return
        self._last_event = event
        state = str(getattr(event.state, "value", event.state))
        glyph = {"complete": "✓", "ready": "✓", "failed": "✗", "degraded": "⚠", "skipped": "·"}.get(state)
        if glyph is None:
            return  # transient running/delegated states: the spinner covers them
        line = f"{glyph} {event.stage}: {state}"
        if event.detail:
            line += f" — {redact_progress_detail(event.detail, secret_values=self._secrets)}"
        self._tail.append(line[:110])
        del self._tail[: -self._max_lines]
        self._on_tail(list(self._tail))

    def heartbeat(self) -> None:
        return None

    def close(self) -> None:
        self._stop.set()
        if self._poller is not None:
            self._poller.join(timeout=5)
            self._poller = None


def make_view_console(file: TextIO | None = None) -> Console:
    """A console for view-level output.

    Resolves ``sys.stdout`` at CALL time: do not call inside the Tier 3
    engine-capture window (``redirect_stdout``) or the output lands in the
    capture buffer -- pass the already-pinned view console's file instead.
    """
    return Console(highlight=False, emoji=False, file=file if file is not None else sys.stdout)


def _pill(label: str, color: str, ink: str = INK) -> Text:
    return Text(f" {label} ", style=f"bold {ink} on {color}")


@dataclass
class TierRow:
    """One detail line inside a tier section."""

    label: str
    segments: list[tuple[str, str]]
    glyph: str = " "
    glyph_style: str = MUTED
    label_style: str = MUTED
    fit: str = ""  # "wrap": word-wrap the last segment to the panel; "rule": a full-width divider
    atomic_groups: list[list[tuple[str, str]]] | None = None


def detail_row(label: str, value: str, note: str = "") -> TierRow:
    """A plain label/value row, with an optional muted note."""
    segments: list[tuple[str, str]] = [(value, TEXT)]
    if note:
        segments.append((f"  {note}", MUTED))
    return TierRow(label, segments)


def lift_row(lift: float, with_score: float, baseline: float) -> TierRow:
    """The skill-lift row with paired horizontal score bars."""

    def bar(score: float, color: str) -> list[tuple[str, str]]:
        filled = max(0, min(BAR, round(score * BAR)))
        return [("█" * filled, color), ("░" * (BAR - filled), FAINT)]

    delta_style = f"bold {lift_style(lift)}"
    segments: list[tuple[str, str]] = [
        (f"{lift:+.2f}", delta_style),
        ("   with-skill ", MUTED),
        (f"{with_score:.2f} ", f"bold {TEXT}"),
        *bar(with_score, GREEN),
        ("   baseline ", MUTED),
        (f"{baseline:.2f} ", f"bold {TEXT}"),
        *bar(baseline, MUTED),
    ]
    return TierRow("lift", segments, atomic_groups=[segments[:1], segments[1:5], segments[5:]])


_CASES_PER_LINE = 8
_FIX_TEXT_LIMIT = 300
_CONDITION_LABELS = {"with_skill": "with skill", "without_skill": "without"}
_CASE_STYLES = {"✓": GREEN, "✗": f"bold {RED}", "⊘": RED, "▸": f"bold {BRIGHT}", "·": FAINT}


def _case_state(marks: list[str], k: int | None, *, running: bool) -> str:
    """Case glyph: ✓ every attempt passed, ✗ one failed, ⊘ one did not run, ▸ running, · waiting."""
    if "✗" in marks:
        return "✗"
    if "⊘" in marks:
        return "⊘"
    finished = "✓" in marks and not running and ("·" in marks or k is None or len(marks) >= k)
    if finished:
        return "✓"
    return "▸" if running or marks else "·"


def _case_prefix(case_ids: list[str]) -> str:
    """The shared ``name-`` prefix of the case ids, shown once in the agent row."""
    if len(case_ids) < 2:
        return ""
    first, last = min(case_ids), max(case_ids)
    common = os.path.commonprefix([first, last])
    cut = common.rfind("-") + 1
    return common[:cut] if all(len(c) > cut for c in case_ids) else ""


def _case_line_rows(label: str, states: dict[str, str], case_ids: list[str], prefix: str) -> list[TierRow]:
    counts = {glyph: list(states.values()).count(glyph) for glyph in _CASE_STYLES}
    tally: list[tuple[str, str]] = [
        ("    ", ""),
        (f"{counts['✓']} passed", GREEN if counts["✓"] else MUTED),
        (" · ", FAINT),
        (f"{counts['✗']} failed", f"bold {RED}" if counts["✗"] else MUTED),
    ]
    if counts["⊘"]:
        tally += [(" · ", FAINT), (f"{counts['⊘']} did not run", RED)]
    if counts["▸"] + counts["·"]:
        tally += [(" · ", FAINT), (f"{counts['▸'] + counts['·']} to go", MUTED)]
    rows: list[TierRow] = []
    for start in range(0, len(case_ids), _CASES_PER_LINE):
        segments: list[tuple[str, str]] = []
        for case_id in case_ids[start : start + _CASES_PER_LINE]:
            glyph = states.get(case_id, "·")
            name = sanitize_tier3_console_text(case_id[len(prefix) :], limit=24)
            segments += [("  " if segments else "", ""), (f"{glyph} {name}", _CASE_STYLES[glyph])]
        rows.append(TierRow(label if not rows else "", segments))
    rows[0].segments += tally
    return rows


def tier3_case_rows(
    marks: dict[str, dict[str, dict[str, list[str]]]],
    *,
    agents: list[str] | None = None,
    conditions: tuple[str, ...] = ("with_skill", "without_skill"),
    k: int | None = None,
    running: set[tuple[str, str, str]] | None = None,
    progress: tuple[int, int] | None = None,
    show_agent: bool = True,
) -> list[TierRow]:
    """Tier 1-style case rows: one ✓/✗ per test case and condition, plus passed/failed counts."""
    running = running or set()
    agents = [a for a in (agents or list(marks)) if a in marks]
    case_ids = sorted({c for a in agents for key in conditions for c in marks[a].get(key, {})})
    if not case_ids:
        return []
    prefix = _case_prefix(case_ids)
    if k is None:
        k = (
            max((len(m) for a in agents for key in conditions for m in marks[a].get(key, {}).values()), default=0)
            or None
        )
    names = sanitize_tier3_console_text(", ".join(agents), limit=64)
    detail = (
        f"  {len(case_ids)} test case{'s' if len(case_ids) != 1 else ''}"
        + (f" ({prefix}…)" if prefix else "")
        + (f" · k={k}" if k else "")
    )
    header = [(names, TEXT), (detail, MUTED)] if show_agent else [(detail.strip(), TEXT)]
    rows = [TierRow("agent" if show_agent else "cases", header)]
    if progress and progress[1] > 0:
        done, total = min(progress[0], progress[1]), progress[1]
        filled = round(BAR * done / total)
        rows.append(
            TierRow(
                "progress",
                [("█" * filled, GREEN), ("░" * (BAR - filled), FAINT), (f"  {done}/{total} attempts", MUTED)],
            )
        )
    for agent in agents:
        indent = ""
        if len(agents) > 1:
            rows.append(TierRow(sanitize_tier3_console_text(agent, limit=LABEL - 1), []))
            indent = " "
        for key in conditions:
            cases = marks[agent].get(key, {})
            states = {c: _case_state(cases.get(c, []), k, running=(agent, key, c) in running) for c in case_ids}
            rows += _case_line_rows(indent + _CONDITION_LABELS[key], states, case_ids, prefix)
    return rows


_FIX_ITEMS_SHOWN = 3


@dataclass
class FixItem:
    """One "what to fix" entry: where it is, what is wrong, and the action(s) that fix it."""

    where: str
    problem: str
    actions: list[str] = field(default_factory=list)


def fix_section_rows(items: list[FixItem]) -> list[TierRow]:
    """The highlighted "what to fix" section closing a tier box; empty when nothing needs fixing."""
    if not items:
        return []
    rows = [
        TierRow("", []),
        TierRow("what to fix", [("─", FAINT)], label_style=f"bold {AMBER}", fit="rule"),
    ]
    for number, item in enumerate(items[:_FIX_ITEMS_SHOWN], start=1):
        rows.append(
            TierRow(
                item.where[: LABEL - 1],
                [(item.problem, TEXT)],
                glyph=str(number),
                glyph_style=f"bold {AMBER}",
                fit="wrap",
            )
        )
        rows += [TierRow("", [("→ ", AMBER), (action, MUTED)], fit="wrap") for action in item.actions]
    if len(items) > _FIX_ITEMS_SHOWN:
        rows.append(TierRow("", [(f"+{len(items) - _FIX_ITEMS_SHOWN} more in detailed reports", FAINT)]))
    return rows


def _case_span(case_ids: list[str]) -> str:
    """Compact case label: ``001-003`` for a consecutive run, else a list, else a count."""
    prefix = _case_prefix(case_ids)
    names = [c[len(prefix) :] for c in case_ids]
    if len(names) > 2 and all(n.isdigit() for n in names):
        numbers = [int(n) for n in names]
        if numbers == list(range(numbers[0], numbers[0] + len(numbers))):
            return f"{names[0]}-{names[-1]}"
    label = ", ".join(names)
    return label if len(label) < LABEL else f"{len(names)} cases"


def tier3_fix_rows(findings: list[dict] | None, *, agents: int = 1) -> list[TierRow]:
    """Tier 3 "How to improve" items as the tier's "what to fix" section."""
    items = []
    for finding in findings or []:
        case_ids = [c.strip() for c in str(finding.get("case", "")).split(",") if c.strip()]
        kind = sanitize_tier3_console_text(str(finding.get("kind", "")), limit=24)
        what = sanitize_tier3_console_text(str(finding.get("what", "")).split(": ", 1)[-1], limit=_FIX_TEXT_LIMIT)
        if agents > 1 and finding.get("scope"):
            what += f" ({sanitize_tier3_console_text(str(finding.get('scope')), limit=32)})"
        action = sanitize_tier3_console_text(str(finding.get("action", "")), limit=_FIX_TEXT_LIMIT)
        items.append(
            FixItem(
                sanitize_tier3_console_text(_case_span(case_ids), limit=LABEL - 1) if case_ids else "cases",
                f"{kind.capitalize()}: {what}" if kind else what,
                [action] if action else [],
            )
        )
    return fix_section_rows(items)


def _fit_row(row: TierRow, room: int) -> list[list[tuple[str, str]]]:
    """Lay a row out in ``room`` columns so nothing wraps under the label column."""
    if row.atomic_groups:
        lines: list[list[tuple[str, str]]] = [[]]
        for group in row.atomic_groups:
            if lines[-1] and cell_len("".join(c for c, _ in [*lines[-1], *group])) > room:
                lines.append([])
                group = [(group[0][0].lstrip(), group[0][1]), *group[1:]]
            lines[-1].extend(group)
        return lines
    if row.fit == "rule":
        return [[("─" * room, row.segments[0][1])]]
    if row.fit != "wrap" or not row.segments:
        return _wrap_segments(row.segments, room)
    *prefix, (text, style) = row.segments
    indent = sum(len(chunk) for chunk, _ in prefix)
    wrapped = textwrap.wrap(text, max(room - indent, 20), break_on_hyphens=False) or [""]
    return [[*prefix, (wrapped[0], style)]] + [[(" " * indent + line, style)] for line in wrapped[1:]]


def _wrap_segments(segments: list[tuple[str, str]], room: int) -> list[list[tuple[str, str]]]:
    """Split a row at segment boundaries so it never wraps under the label column."""
    lines: list[list[tuple[str, str]]] = [[]]
    used = 0
    for chunk, style in segments:
        if used and used + cell_len(chunk) > room:
            lines.append([])
            used = 0
            if chunk.strip() in {"·", ""}:
                continue
        lines[-1].append((chunk, style))
        used += cell_len(chunk)
    return lines


@dataclass
class TierBlock:
    """State of one pipeline tier as the run progresses."""

    number: int
    name: str
    caption: str
    status: str = PENDING
    duration: float | None = None
    rows: list[TierRow] = field(default_factory=list)
    started_at: float | None = None


@dataclass
class Verdict:
    passed: bool
    headline: str
    fix: list[tuple[str, str]] | None = None
    rerun: str | None = None
    incomplete: bool = False
    details: tuple[str, ...] = ()


class ValidateView:
    """Progressive pipeline renderer for validate.

    Drive it from the command: :meth:`start`, then :meth:`tier_start` /
    :meth:`tier_progress` / :meth:`tier_done` (or :meth:`tier_skip`) per
    tier, and :meth:`finish` with the verdict and footer links. All methods
    are no-ops when the view was constructed with ``enabled=False``.
    """

    def __init__(
        self,
        skill: str,
        tiers: list[tuple[int, str, str]],
        *,
        command: str = "validate",
        console: Console | None = None,
        enabled: bool = True,
    ):
        self.skill = skill
        self.command = command
        self.enabled = enabled
        # Pin the output stream at construction: the Tier 3 engine's narration
        # is captured via redirect_stdout, and the view must keep writing to
        # the real stream (or the test runner's buffer) while that happens.
        self.console = console or Console(highlight=False, emoji=False, file=sys.stdout)
        # Design width is 98 columns; degrade gracefully on narrower terminals.
        self.width = max(60, min(WIDTH, self.console.width))
        self.blocks = [TierBlock(number=number, name=name, caption=caption) for number, name, caption in tiers]
        self.verdict: Verdict | None = None
        self.links: list[tuple[str, str]] = []
        self._live: Live | None = None
        self.harness_panel: Panel | None = None
        self.harness_note: str | None = None
        self.progress_reporter = None
        self._progress_stage: dict[int, set[str]] = {}
        self._static_printed = 0  # blocks already flushed in non-tty mode

    # ── lifecycle ────────────────────────────────────────────────

    def _live_capable(self) -> bool:
        """Live redraws need a real, capable TTY, as Harbor progress requires.

        FORCE_COLOR/TTY_COMPATIBLE make Rich treat a CI pipe as a terminal; that
        stream still needs durable stderr progress, not cursor-control frames.
        """
        console = self.console
        return bool(
            console.is_terminal and not console.is_dumb_terminal and getattr(console.file, "isatty", lambda: False)()
        )

    def start(self) -> None:
        if not self.enabled:
            return
        if self._live_capable():
            self.console.print(self._header())
            self.console.print()
            # Only unfinished tiers stay live; finished ones print once above. A live
            # region taller than the window can't be erased, so each redraw would repeat it.
            self._live = Live(get_renderable=self._live_region, console=self.console, refresh_per_second=8)
            self._live.start()
            route_logs_to(self.console)
        else:
            from skillevaluator.tier3.harbor.progress import PlainProgressReporter

            self.progress_reporter = PlainProgressReporter(stream=sys.stderr)
            self.progress_reporter.start_progress()
            self.console.print(self._header())
            self.console.print()

    def tier_start(self, index: int) -> None:
        if not self.enabled:
            return
        block = self.blocks[index]
        block.status = RUNNING
        if block.started_at is None:
            block.started_at = time.time()
        self._progress_event(index, "running")
        self._refresh()

    def tier_progress(self, index: int, rows: list[TierRow]) -> None:
        """Update a running tier's detail rows (check ticker / run config)."""
        if not self.enabled:
            return
        self.blocks[index].rows = rows
        meaningful = [row for row in rows if row.label in {"checks", "status", "stages", "progress"}]
        for row in meaningful:
            detail = "".join(chunk for chunk, _ in row.segments).strip()
            if row.label == "checks":
                detail = next((chunk.removeprefix("▸ ") for chunk, _ in row.segments if chunk.startswith("▸ ")), detail)
            elif row.label == "progress" and row.segments:
                # The bar cells only make sense in Live; durable logs keep the count.
                detail = row.segments[-1][0].strip()
            stage = f"Tier {self.blocks[index].number}/{row.label}"
            self._progress_stage.setdefault(index, set()).add(stage)
            self._progress_event(index, "running", detail, stage=stage)
        self._refresh()

    def tier_done(self, index: int, *, failed: bool = False, rows: list[TierRow] | None = None) -> None:
        if not self.enabled:
            return
        block = self.blocks[index]
        block.status = FAIL if failed else PASS
        block.duration = time.time() - block.started_at if block.started_at else None
        block.rows = rows or []
        self._complete_progress(index, "failed" if failed else "complete")
        self._flush_static()
        self._refresh()

    def tier_skip(self, index: int, reason: str) -> None:
        if not self.enabled:
            return
        block = self.blocks[index]
        block.status = SKIP
        self._complete_progress(index, "skipped", reason)
        # Multi-line reasons (setup instructions) continue on unlabeled rows.
        lines = reason.split("\n") or [""]
        block.rows = [TierRow("skipped" if index == 0 else "", [(line, MUTED)]) for index, line in enumerate(lines)]
        self._flush_static()
        self._refresh()

    def set_harness(self, panel: Panel, note: str | None) -> None:
        """Show the advisory harness panel after the Tier 1 block and *note* in the verdict."""
        self.harness_panel = panel
        self.harness_note = note
        self._refresh()

    def finish(self, verdict: Verdict, links: list[tuple[str, str]]) -> None:
        if not self.enabled:
            return
        self.verdict = verdict
        self.links = links
        self.stop()
        if len(self.blocks) >= 2:
            self.console.print(self._gate())
            self.console.print()
        self.console.print(self._verdict_panel(verdict))
        self.console.print()
        for line in self._footer_lines():
            self.console.print(line)

    def stop(self) -> None:
        """Tear down the live display (safe on errors / KeyboardInterrupt)."""
        if self.progress_reporter is not None:
            self.progress_reporter.close()
        if self._live is not None:
            route_logs_to(None)
            self._live.stop()
            self._live = None

    def _progress_event(self, index, state, detail=None, *, stage=None):
        if self.progress_reporter is not None:
            from skillevaluator.tier3.harbor.progress import ProgressEvent

            self.progress_reporter.emit(ProgressEvent(stage or f"Tier {self.blocks[index].number}", state, detail))

    def _complete_progress(self, index, state, detail=None):
        for stage in sorted(self._progress_stage.pop(index, set())):
            self._progress_event(index, state, stage=stage)
        self._progress_event(index, state, detail)

    # ── rendering ────────────────────────────────────────────────

    def __rich_console__(self, console, options):  # rich renderable protocol
        yield self._header()
        yield Text()
        if len(self.blocks) >= 2:
            yield self._gate()
            yield Text()
        for index, block in enumerate(self.blocks):
            if block.status == PENDING:
                continue
            yield self._block_panel(block)
            if index == 0 and self.harness_panel is not None:
                yield self.harness_panel
        if self.verdict is not None:
            yield self._verdict_panel(self.verdict)
            yield Text()
            yield from self._footer_lines()

    def _header(self) -> Panel:
        inner = self.width - 6
        left = Text.assemble(
            ("skillevaluator", f"bold {TEXT}"),
            (f" {self.command}", MUTED),
        )
        right = _pill(strip_terminal_controls(self.skill), GREEN)
        pad = max(1, inner - left.cell_len - right.cell_len)
        return Panel(
            left + Text(" " * pad) + right,
            box=box.ROUNDED,
            border_style=FAINT,
            width=self.width,
            padding=(0, 2),
        )

    def _gate(self) -> Group:
        indent = 8
        n = len(self.blocks)
        span = min(68 if n >= 3 else 34, self.width - 30)
        seg = span // max(1, n - 1) - 1 if n > 1 else 0
        tick = int(time.time() * 8)
        node_style = {
            PENDING: ("○", FAINT),
            SKIP: ("○", FAINT),
            RUNNING: (SPIN[tick % len(SPIN)], f"bold {BRIGHT}"),
            PASS: ("●", f"bold {GREEN}"),
            FAIL: ("●", f"bold {RED}"),
        }
        g = Text(" " * indent)
        for i, block in enumerate(self.blocks):
            if i > 0:
                prev = self.blocks[i - 1].status
                if prev == FAIL:
                    half = (seg - 3) // 2
                    g.append("═" * half, style=RED)
                    g.append(" ✗ ", style=f"bold {RED}")
                    g.append("╌" * (seg - 3 - half), style=FAINT)
                elif prev == PASS:
                    g.append("═" * seg, style=GREEN)
                else:
                    g.append("╌" * seg, style=FAINT)
            glyph, style = node_style[block.status]
            g.append(glyph, style=style)

        c = Text(" " * indent)
        pos = 0
        for i, block in enumerate(self.blocks):
            offset = i * (seg + 1)
            style = {FAIL: RED, RUNNING: BRIGHT}.get(block.status, MUTED)
            # Clamp every caption but the last to its segment so a long
            # caption on a narrow terminal can neither run into its neighbor
            # nor shift the neighbor off its node.
            caption = block.caption
            if i < len(self.blocks) - 1 and len(caption) > seg:
                caption = caption[: max(1, seg - 1)] + "…"
            c.append(" " * max(0, offset - pos))
            c.append(caption, style=style)
            pos = max(pos, offset) + len(caption)
        return Group(g, c)

    def _block_panel(self, block: TierBlock) -> Panel:
        tick = int(time.time() * 8)
        if block.status == RUNNING:
            status = Text(f"{SPIN[tick % len(SPIN)]} running", f"bold {BRIGHT}")
            duration = time.time() - block.started_at if block.started_at else 0.0
        elif block.status == PASS:
            status = Text("✓ pass", f"bold {GREEN}")
            duration = block.duration
        elif block.status == SKIP:
            status = Text("· skipped", MUTED)
            duration = None
        else:
            status = Text("✗ fail", f"bold {RED}")
            duration = block.duration
        dur = _fmt_duration(duration)
        if dur:
            status = status + Text(f" · {dur}", MUTED)

        title = Text.assemble(
            (f"Tier {block.number}", f"bold {GREEN if block.status == PASS else MUTED}"),
            (" · ", FAINT),
            (block.name, f"bold {TEXT}"),
        )
        lines = []
        room = self.width - 8 - LABEL
        for row in block.rows:
            label = strip_terminal_controls(row.label)
            if cell_len(label) > LABEL:
                lines.append(Text(label, style=row.label_style))
                label = ""
            for i, segments in enumerate(_fit_row(row, room)):
                t = Text()
                t.append(f"{row.glyph if i == 0 else ' '} ", style=row.glyph_style)
                t.append(
                    (label if i == 0 else "") + " " * (LABEL - cell_len(label if i == 0 else "")), style=row.label_style
                )
                for chunk, style in segments:
                    t.append(strip_terminal_controls(chunk), style=style)
                lines.append(t)
        body = Group(*lines) if lines else Text("…", style=FAINT)
        return Panel(
            body,
            box=box.ROUNDED,
            border_style=RED if block.status == FAIL else FAINT,
            width=self.width,
            padding=(0, 2),
            title=title,
            title_align="left",
            subtitle=status,
            subtitle_align="right",
        )

    def _verdict_panel(self, verdict: Verdict) -> Panel:
        headline = strip_terminal_controls(verdict.headline)
        note = (
            Text.assemble(("! harness: ", f"bold {AMBER}"), (strip_terminal_controls(self.harness_note), AMBER))
            if self.harness_note
            else None
        )
        if verdict.passed:
            body = Text.assemble(_pill("✓ PASS", GREEN), ("  ", ""), (headline, f"bold {TEXT}"), ("  ·  exit 0", MUTED))
            details = [Text(strip_terminal_controls(line), style=MUTED) for line in verdict.details]
            return Panel(
                Group(body, *details, *([Text(), note] if note is not None else [])),
                box=box.ROUNDED,
                border_style=GREEN,
                width=self.width,
                padding=(0, 2),
            )
        color = AMBER if verdict.incomplete else RED
        line1 = Text.assemble(
            _pill("INCOMPLETE" if verdict.incomplete else "✗ FAIL", color, ink="#1C0605"),
            ("  ", ""),
            (headline, f"bold {TEXT}"),
            ("  ·  exit 1", MUTED),
        )
        parts: list = [line1, *(Text(strip_terminal_controls(line), style=MUTED) for line in verdict.details)]
        if note is not None:
            parts.extend([Text(), note])
        if verdict.fix:
            fix = Text.assemble(("fix     ", f"bold {MUTED}"))
            for chunk, style in verdict.fix:
                fix.append(strip_terminal_controls(chunk), style=style)
            parts.extend([Text(), fix])
        if verdict.rerun:
            parts.append(Text.assemble(("        ", ""), (strip_terminal_controls(verdict.rerun), GREEN)))
        return Panel(Group(*parts), box=box.ROUNDED, border_style=color, width=self.width, padding=(0, 2))

    def _footer_lines(self):
        for label, target in self.links:
            prefix = Text.assemble(("      ", ""), (f"{strip_terminal_controls(label):<{LABEL}}", MUTED))
            yield artifact_text(target, self.console, prefix=prefix)

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.refresh()

    def _live_region(self) -> Group:
        return Group(*(self._block_panel(b) for b in self.blocks[self._static_printed :] if b.status != PENDING))

    def _flush_static(self) -> None:
        """Print blocks once, in order, as they complete."""
        done = []
        for block in self.blocks[self._static_printed :]:
            if block.status in (PENDING, RUNNING):
                break
            done.append(block)
        # Advance first so the live region drops these blocks as they print above it.
        first = self._static_printed
        self._static_printed += len(done)
        for index, block in enumerate(done, start=first):
            self.console.print(self._block_panel(block))
            if index == 0 and self.harness_panel is not None:
                self.console.print(self.harness_panel)


def check_ticker_row(lineup: list[str], done: list[str], current: str | None) -> TierRow:
    """Live Tier 1 lineup: completed checks tick, the active one is marked.

    The tier header's spinner carries the animation; the active check gets a
    static ``▸`` marker so the row stays legible between refreshes.
    """
    segments: list[tuple[str, str]] = []
    for i, name in enumerate(lineup):
        if i:
            segments.append((" · ", FAINT))
        if name in done:
            segments.append((f"✓ {name}", GREEN))
        elif name == current:
            segments.append((f"▸ {name}", f"bold {BRIGHT}"))
        else:
            segments.append((name, MUTED))
    return TierRow("checks", segments)


def stage_hint_row(label: str, hint: str, *, lead: str = "") -> TierRow:
    """A muted single-line hint shown while a tier is running."""
    segments: list[tuple[str, str]] = []
    if lead:
        segments.append((f"{lead}  ", TEXT))
    segments.append((hint, MUTED))
    return TierRow(label, segments)


# ── summaries: turn ValidationResult lists into tier rows ──────────────


_SCANNER_NAMES = {"skillspector": "SkillSpector", "bandit": "Bandit", "semgrep": "Semgrep", "gitleaks": "Gitleaks"}
_CHECK_SCANNERS = {"Security Scan": "skillspector", "Code Risk Analysis": "bandit", "Secrets Detection": "gitleaks"}


def _scanners_row(results: list[ValidationResult]) -> list[TierRow]:
    """Scanner versions and incomplete evidence, retaining every member's scan state."""
    versions: dict[str, str] = {}
    states: dict[str, set[str]] = {}
    for r in results:
        if _is_user_skipped(r):
            if scanner := _CHECK_SCANNERS.get(r.validator_name):
                states.setdefault(scanner, set()).add("skipped")
            continue
        for scanner, version in r.scanners.items():
            if scanner == "skillspector" and version == "unknown":
                runtime_version = (r.metadata or {}).get("skillspector_version")
                if isinstance(runtime_version, str) and runtime_version:
                    version = runtime_version
            if versions.get(scanner, "unknown") == "unknown":
                versions[scanner] = version
            states.setdefault(scanner, set()).add("ran")
        for scan in r.incomplete_scans:
            if (scanner := scan.removesuffix("-llm")) in _SCANNER_NAMES:
                if scan != scanner and scanner in r.scanners:
                    state = "llm"
                else:
                    state = "missing" if _scanner_setup_issue(r) else "incomplete"
                states.setdefault(scanner, set()).add(state)
    segments: list[tuple[str, str]] = []
    for scanner in [name for name in _SCANNER_NAMES if name in states]:
        version, observed = versions.get(scanner), states[scanner]
        style = f"bold {RED}"
        if "incomplete" in observed or ("missing" in observed and version):
            state = f"{version}, incomplete" if version else "incomplete"
        elif "llm" in observed:
            state = f"{version}, LLM didn't run" if version else "LLM incomplete"
        elif "missing" in observed:
            state = "didn't run"
        else:
            state, style = version or "skipped", MUTED
        segments += [*([(" · ", FAINT)] if segments else []), (_SCANNER_NAMES[scanner], TEXT), (f" {state}", style)]
    return [TierRow("scanners", segments)] if segments else []


def _is_user_skipped(result: ValidationResult) -> bool:
    """True when the user left this check out with ``--skip``."""
    return isinstance(result.metadata, dict) and bool(result.metadata.get("skipped_by_user"))


def _is_not_applicable(result: ValidationResult) -> bool:
    """True when a check was recorded as not applicable to the target instead of run."""
    return not _is_user_skipped(result) and (
        result_execution_status(result) == "not_applicable"
        or (_is_skipped(result) and "not applicable" in get_skip_reason(result).casefold())
    )


def _is_skipped(result: ValidationResult) -> bool:
    """True when a validator was recorded without executing (OSS conventions)."""
    if bool(getattr(result, "skipped", False)):
        return True
    meta = result.metadata if isinstance(result.metadata, dict) else {}
    if meta.get("skipped") or meta.get("execution_status") == "skipped":
        return True
    payload = meta.get("agent_eval")
    if not isinstance(payload, dict):
        return False
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    return payload.get("execution_status") == "skipped" or summary.get("execution_status") == "skipped"


def _bounded_console_line(line: str) -> str:
    """Flatten one line of untrusted text and bound it for the compact view."""
    flattened = " ".join(
        "".join(
            " " if character.isspace() or unicodedata.category(character).startswith("C") else character
            for character in line
        ).split()
    )
    if len(flattened) <= _SKIP_REASON_MAX_CHARS:
        return flattened
    return flattened[: _SKIP_REASON_MAX_CHARS - 1] + "…"


def _raw_skip_reason(result: ValidationResult) -> str:
    """Return the untrusted reason text that ``get_skip_reason`` would select."""
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    payload = metadata.get("agent_eval")
    provenance = payload.get("provenance") if isinstance(payload, dict) else None
    candidates = (
        metadata.get("skip_reason"),
        provenance.get("message") if isinstance(provenance, dict) else None,
        result.warnings[0] if result.warnings else None,
    )
    for candidate in candidates:
        if candidate is None:
            continue
        if not isinstance(candidate, str):
            # Non-text metadata keeps the shared single-line projection.
            return ""
        if _bounded_console_line(strip_terminal_controls(candidate)):
            return candidate
    return ""


def _skip_reason(result: ValidationResult) -> str:
    """Return a bounded, terminal-safe reason that keeps its line structure.

    Each line is flattened and bounded separately, and at most
    ``_SKIP_REASON_MAX_LINES`` lines are kept, so setup instructions stay
    readable while escapes, controls, and oversized text cannot reach the
    terminal.
    """
    raw = _raw_skip_reason(result)
    if not raw:
        return _bounded_console_line(get_skip_reason(result))
    sample = strip_terminal_controls(raw[:_SKIP_REASON_SOURCE_MAX_CHARS])
    lines = [
        bounded
        for line in sample.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if (bounded := _bounded_console_line(line))
    ]
    truncated = len(raw) > _SKIP_REASON_SOURCE_MAX_CHARS or len(lines) > _SKIP_REASON_MAX_LINES
    lines = lines[:_SKIP_REASON_MAX_LINES]
    if truncated and lines and not lines[-1].endswith("…"):
        lines[-1] = lines[-1][: _SKIP_REASON_MAX_CHARS - 1] + "…"
    return "\n".join(lines)


def sanitize_tier3_console_text(value: object, *, limit: int = _TIER3_TEXT_MAX_CHARS) -> str:
    """Project untrusted Tier 3 text into one bounded terminal-safe line."""
    if not isinstance(value, str) or limit <= 0:
        return ""
    sample_limit = max(limit, limit * 4)
    sample = value[:sample_limit]
    flattened = " ".join(
        "".join(
            " " if character.isspace() or unicodedata.category(character).startswith("C") else character
            for character in sample
        ).split()
    )
    truncated = len(value) > len(sample) or len(flattened) > limit
    if not truncated:
        return flattened
    if limit == 1:
        return "…"
    return flattened[: limit - 1] + "…"


def _tier3_string_list(value: object, *, max_items: int = 16) -> list[str]:
    """Return a small list of safe strings from a canonical JSON array."""
    if not isinstance(value, list):
        return []
    projected = [sanitize_tier3_console_text(item, limit=64) for item in value[:max_items]]
    return [item for item in projected if item]


def _tier3_finite_number(value: object) -> float | None:
    """Return a finite display number without overflowing on huge integers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


_VALIDATOR_CHECK_KEYS = (
    ("schema", "schema"),
    ("security scan", "security"),
    ("pii", "pii"),
    ("license", "license"),
    ("code risk", "code-integrity"),
    ("secrets", "code-integrity"),
    ("hygiene", "code-integrity"),
    ("integrity", "code-integrity"),
    ("unicode", "unicode"),
    ("quality", "quality"),
    ("script_lint", "lint"),
    ("lint", "lint"),
    ("dependency", "dependency"),
    ("version", "version"),
    ("harness", "harness"),
)


def _check_key(validator_name: str) -> str | None:
    name = (validator_name or "").lower()
    for needle, key in _VALIDATOR_CHECK_KEYS:
        if needle in name:
            return key
    return None


def final_check_ticker_row(lineup: list[str], results: list[ValidationResult]) -> TierRow:
    """The check lineup with final per-check states: ✓ passed, ✗ failed, dim skipped."""
    status: dict[str, str] = {}
    user_skipped: set[str] = set()
    for result in results:
        key = _check_key(result.validator_name)
        if key is not None and _is_user_skipped(result):
            user_skipped.add(key)
        if key is None or _is_skipped(result):
            continue
        if not result.passed:
            status[key] = FAIL
        elif status.get(key) != FAIL:
            status[key] = WARN if _harness_flagged(result) else PASS
    segments: list[tuple[str, str]] = []
    for i, name in enumerate(lineup):
        if i:
            segments.append((" · ", FAINT))
        state = status.get(name)
        if state == FAIL:
            segments.append((f"✗ {name}", f"bold {RED}"))
        elif state == WARN:
            segments.append((f"! {name}", f"bold {AMBER}"))
        elif state == PASS:
            segments.append((f"✓ {name}", GREEN))
        elif name in user_skipped:
            segments.append((f"\N{EN DASH} {name}", MUTED))
        else:
            segments.append((name, FAINT))
    return TierRow("checks", segments)


def _harness_flagged(result: ValidationResult) -> bool:
    """True when the advisory harness check found a harness that drops or degrades a skill."""
    block = (result.metadata or {}).get("harness_compatibility") or {}
    return any(row.get("status") != "loads" for rows in block.values() for row in rows)


def _short_name(validator_name: str) -> str:
    name = (validator_name or "check").strip()
    aliases = {
        "Schema & Repository Governance": "schema",
        "Security Scan": "security",
        "PII Scan": "pii",
        "License Compliance": "license",
        "Code Risk Analysis": "code",
        "Secrets Detection": "secrets",
        "Code Integrity & Hygiene": "integrity",
        "Unicode Smuggling Detection": "unicode",
        "QUALITY": "quality",
        "SCRIPT_LINT": "lint",
        "Harness Compatibility": "harness",
        "Context Deduplication": "context",
        "Inter-Skill Deduplication": "catalog",
    }
    return aliases.get(name, name.lower())


def _quality_stats(results: list[ValidationResult]) -> tuple[str, float] | None:
    for result in results:
        if (result.validator_name or "").upper() != "QUALITY":
            continue
        scores = (result.metadata or {}).get("quality_scores") or {}
        if "grade" in scores and "overall_score" in scores:
            return str(scores["grade"]), float(scores["overall_score"])
        for detail in result.success_details:
            meta = detail.metadata or {}
            if "grade" in meta and "overall_score" in meta:
                return str(meta["grade"]), float(meta["overall_score"])
    return None


def summarize_tier1(results: list[ValidationResult], lineup: list[str] | None = None) -> tuple[bool, list[TierRow]]:
    """Build the Tier 1 section rows from its validator results.

    When *lineup* is given, the per-check ticker stays in the completed
    section with each check's final state (failed checks in red).
    """
    from skillevaluator.reporting.run_status import Status, result_status

    # A check recorded as not applicable did not run: it is neither a pass nor
    # a failure, so it is listed on its own row and left out of the counts.
    not_applicable = [r for r in results if _is_not_applicable(r)]
    results = [r for r in results if not _is_not_applicable(r)]

    user_skipped = [r for r in results if _is_user_skipped(r)]
    failed = [r for r in results if not r.passed and not _is_skipped(r)]
    # A blocking finding or failed gate outranks an incomplete scan, as on every other surface.
    incomplete = [r for r in failed if result_status(r) is Status.INCOMPLETE]
    broken = [r for r in failed if result_status(r) is not Status.INCOMPLETE]
    # Warnings from passing validators surface as advisories below; only the
    # failing validators' warnings belong on the headline checks row.
    warnings = sum(len(r.warnings) for r in failed)
    advisory: dict[str, int] = {}
    for r in results:
        if r.passed and not _is_skipped(r) and r.findings and _short_name(r.validator_name) != "harness":
            advisory[_short_name(r.validator_name)] = advisory.get(_short_name(r.validator_name), 0) + len(r.findings)

    rows: list[TierRow] = []
    scanners = _scanners_row(results)
    if lineup:
        rows.append(final_check_ticker_row(lineup, results))
        rows += scanners

    # Count validators with the verdict line's own classification (TierPresentation), so the two agree.
    outcome = {id(r): result_execution_status(r) for r in results}
    count = Counter(outcome.values())
    executed = count["pass"] + count["fail"] + count["incomplete"]
    checks_segments: list[tuple[str, str]] = [(f"{count['pass']}/{executed} checks passed", TEXT)]
    if sub_checks := sum(
        r.summary.checks_performed for r in results if outcome[id(r)] in {"pass", "fail", "incomplete"}
    ):
        checks_segments.append((f" ({sub_checks} sub-checks)", MUTED))
    if count["fail"]:
        checks_segments += [("  ·  ", FAINT), (f"{count['fail']} failed", f"bold {RED}")]
    if count["incomplete"]:
        checks_segments += [("  ·  ", FAINT), (f"{count['incomplete']} incomplete", f"bold {RED}")]
    if count["skipped"]:
        checks_segments += [("  ·  ", FAINT), (f"{count['skipped']} skipped", MUTED)]
    if warnings:
        checks_segments += [("  ·  ", FAINT), (f"{warnings} warning{'s' if warnings != 1 else ''}", MUTED)]
    rows.append(TierRow("summary" if lineup else "checks", checks_segments))
    if not lineup:
        rows += scanners
    if user_skipped:
        names = ", ".join(dict.fromkeys(_short_name(r.validator_name) for r in user_skipped))
        rows.append(TierRow("skipped", [(names, MUTED), ("  with --skip, not checked", FAINT)]))
    if not_applicable:
        names = ", ".join(
            dict.fromkeys(_check_key(r.validator_name) or _short_name(r.validator_name) for r in not_applicable)
        )
        first = not_applicable[0]
        reason = _bounded_console_line(
            first.success_details[0].message
            if first.success_details and not _is_skipped(first)
            else get_skip_reason(first)
        )
        rows += [TierRow("n/a", [(names, MUTED)]), TierRow("", [(reason, FAINT)], fit="wrap")]

    if advisory:
        detail = " · ".join(f"{name} {count}" for name, count in sorted(advisory.items(), key=lambda kv: -kv[1]))
        rows.append(TierRow("advisories", [(str(sum(advisory.values())), TEXT), (f"  {detail}", MUTED)]))
    harness_rows = [
        row
        for r in results
        for rows in ((r.metadata or {}).get("harness_compatibility") or {}).values()
        for row in rows
    ]
    counts = [
        f"{sum(row.get('status') == status for row in harness_rows)} {label}"
        for status, label in (("not_loaded", "not loaded"), ("degraded", "degraded"))
        if any(row.get("status") == status for row in harness_rows)
    ]
    if counts:
        rows.append(TierRow("harness", [(" · ".join(counts), TEXT), ("  (see below)", MUTED)]))

    suppressed_security = 0
    for r in results:
        applicability = (r.metadata or {}).get("skillspector_applicability")
        if isinstance(applicability, dict):
            count = applicability.get("suppressed_inapplicable_issues")
            if isinstance(count, int) and not isinstance(count, bool) and count > 0:
                suppressed_security += count
    if suppressed_security:
        noun = "issue" if suppressed_security == 1 else "issues"
        rows.append(
            TierRow(
                "filtered",
                [(f"{suppressed_security} scanner {noun} suppressed outside agent control paths", MUTED)],
            )
        )
    for r in failed:
        if reference_summary := reference_summary_message((r.metadata or {}).get("skillspector_reference_summary")):
            rows.append(TierRow("references", [(reference_summary, MUTED)], fit="wrap"))

    quality = _quality_stats(results)
    if quality:
        grade, score = quality
        quality_result = next((r for r in results if _quality_stats([r])), None)
        status = result_status(quality_result).value if quality_result else "unavailable"
        rows.append(
            TierRow("quality", [(grade, f"bold {grade_style(grade, status=status)}"), (f"  {score:.1f} / 100", TEXT)])
        )
    items = [_tier1_fix(r) for r in broken]
    non_llm = [r for r in incomplete if not _only_llm_stages_missing(r)]
    if partial := [r for r in non_llm if not _scanner_setup_issue(r)]:
        items.append(_incomplete_fix(partial))
    if setup := [r for r in non_llm if _scanner_setup_issue(r)]:
        items.append(_setup_fix(setup))
    if llm := [r for r in incomplete if _only_llm_stages_missing(r)]:
        items.append(_llm_fix(llm))
    return not failed, rows + fix_section_rows(items)


def _tier1_fix(result: ValidationResult) -> FixItem:
    problem = _first_error(result) or f"{result.validator_name} failed"
    action = next((f.suggestion for f in result.findings if f.message == problem and f.suggestion), None)
    if action is None and not result.gate_failures:
        action = first_fix([result])
        action = action[0][0] if action else None
    where = _short_name(result.validator_name)
    return FixItem(where, problem, [action] if action and action != problem else [])


_INSTALL_HINT = re.compile(r"\b[Ii]nstall(?: [\w-]+)*? with: (.+?)\.?$")


_DOWNLOAD_FAILED = re.compile(r"couldn't (?:download|set up) .+? \((?:[^()]|\([^()]*\))*\)")
_MISSING_SCANNER = re.compile(r"\bnot installed\b|\bnot found in PATH\b", re.IGNORECASE)


def _scanner_setup_issue(result: ValidationResult) -> bool:
    """Only explicit installation diagnostics justify a scanner setup explanation."""
    return any(
        _DOWNLOAD_FAILED.search(text) or _INSTALL_HINT.search(text) or _MISSING_SCANNER.search(text)
        for text in map(str, [*result.independent_warnings(), *result.independent_errors()])
    )


def _incomplete_fix(results: list[ValidationResult]) -> FixItem:
    """Keep partial, malformed and failed analyses separate from installation failures."""
    checks = ", ".join(dict.fromkeys(_short_name(r.validator_name) for r in results))
    diagnostics = list(
        dict.fromkeys(
            str(message)
            for result in results
            for message in (
                result.gate_failures
                or result.independent_errors()
                or result.independent_warnings()
                or [f"{', '.join(result.incomplete_scans)} did not produce complete evidence"]
            )
        )
    )
    return FixItem(
        "scan",
        f"{checks} incomplete: {'; '.join(diagnostics)}",
        ["resolve the reported scanner or input issues, then rerun; see detailed reports for full diagnostics"],
    )


def _setup_fix(results: list[ValidationResult]) -> FixItem:
    """One item for checks that could not run because a scanner is missing or failed to download."""
    scanners = list(dict.fromkeys(name for r in results for name in r.incomplete_scans))
    hints: list[str] = []
    failed_downloads: list[str] = []
    for r in results:
        for text in map(str, [*r.independent_warnings(), *r.independent_errors()]):
            if (download := _DOWNLOAD_FAILED.search(text)) and download.group(0) not in failed_downloads:
                failed_downloads.append(download.group(0))
            match = _INSTALL_HINT.search(text)
            if match and match.group(1) not in hints:
                hints.append(match.group(1))
    checks = ", ".join(dict.fromkeys(_short_name(r.validator_name) for r in results))
    skip = ",".join(dict.fromkeys(_skip_name(r.validator_name) for r in results))
    if failed_downloads:
        hosts = "github.com and pypi.org" if any("set up" in d for d in failed_downloads) else "github.com"
        actions = [f"check network access to {hosts}, then rerun"]
        actions += [f"or install it yourself: {hint}" for hint in hints]
        problem = f"{checks} didn't run: {'; '.join(failed_downloads)}"
    else:
        actions = list(hints)
        problem = f"{checks} didn't run: {', '.join(scanners) or 'a required scanner'} not installed"
    return FixItem("setup", problem, [*actions, f"or skip it: --skip {skip}"])


def _only_llm_stages_missing(result: ValidationResult) -> bool:
    """True when the check itself ran and only its ``--llm`` / ``--llm-verify`` stage did not.

    ``skillspector-llm`` is also recorded when SkillSpector is missing or its baseline is
    rejected, before any scan runs; only a recorded SkillSpector version shows its static
    scan completed.
    """
    scans = result.incomplete_scans
    if not scans or not all(s.endswith("-llm") or s.startswith("llm-") for s in scans):
        return False
    return "skillspector-llm" not in scans or "skillspector" in result.scanners


def _llm_fix(results: list[ValidationResult]) -> FixItem:
    """One item for checks whose LLM stage could not run: a missing key or a failed provider call."""
    checks = ", ".join(dict.fromkeys(_short_name(r.validator_name) for r in results))
    scans = {s for r in results for s in r.incomplete_scans}
    flags = " / ".join(
        f for f, on in (("--llm", "skillspector-llm" in scans), ("--llm-verify", "llm-verification" in scans)) if on
    )
    # SkillSpector reads its own provider settings when SKILLSPECTOR_PROVIDER names an adapter, so the
    # public provider matters only for the other stages.
    direct = _skillspector_adapter_fix() if "skillspector-llm" in scans else None
    parts = [direct] if direct else []
    if direct is None or "llm-verification" in scans:
        reason, fixes = _public_provider_fix()
        parts.append((f"--llm-verify: {reason}" if direct else reason, fixes))
    reasons = "; ".join(reason for reason, _ in parts)
    actions = [action for _, fixes in parts for action in fixes]
    return FixItem(
        "setup", f"{checks} LLM review didn't run: {reasons}", [*actions, f"or rerun without {flags or '--llm'}"]
    )


def _public_provider_fix() -> tuple[str, list[str]]:
    """The problem and actions for the provider behind ``SKILL_EVAL_LLM_*``: a missing key or a failed call."""
    from skillevaluator.provider_config import resolve_llm_provider

    try:
        resolve_llm_provider()
    except ValueError as exc:
        reason, *setup = str(exc).splitlines() or [""]
        reason = reason.rstrip(".")
        exports = [line.strip().removeprefix("export ") for line in setup if line.strip().startswith("export ")]
        # Some errors name the accepted keys without an export line ("OPENAI_API_KEY or SKILL_EVAL_LLM_API_KEY ...").
        keys = list(dict.fromkeys(re.findall(r"\b[A-Z][A-Z0-9_]*_KEY\b", reason)))
        if exports:
            return reason, [f"export {' '.join(exports)}"]
        if keys:
            return reason, [f"{'or ' if i else ''}export {key}='your-api-key'" for i, key in enumerate(keys)]
        return reason, ["set up an LLM provider and key, then rerun"]
    return "the LLM provider call failed", [
        "check the key and model with: skillevaluator doctor --tier 2 --verify-models"
    ]


def _skillspector_adapter_fix() -> tuple[str, list[str]] | None:
    """The problem and actions for a failed SkillSpector LLM stage when ``SKILLSPECTOR_PROVIDER`` names its adapter.

    ``None`` when it does not: SkillSpector then runs on the provider bridged from ``SKILL_EVAL_LLM_*``.
    """
    from skillevaluator.validators.security import _SKILLSPECTOR_EXPLICIT_PROVIDER_ENV

    adapter = os.environ.get("SKILLSPECTOR_PROVIDER", "").strip().lower()
    if adapter not in _SKILLSPECTOR_EXPLICIT_PROVIDER_ENV:
        return None
    key = next(
        (name for name in sorted(_SKILLSPECTOR_EXPLICIT_PROVIDER_ENV[adapter]) if name.endswith("_API_KEY")), None
    )
    if key is not None and not os.environ.get(key, "").strip():
        return f"{key} is not set for SKILLSPECTOR_PROVIDER={adapter}", [f"export {key}='your-api-key'"]
    credentials = key or "your AWS credentials"
    return f"SkillSpector's {adapter} provider call failed", [
        f"check {credentials} and SKILLSPECTOR_MODEL for SKILLSPECTOR_PROVIDER={adapter}"
    ]


def _skip_name(validator_name: str) -> str:
    """The ``--skip`` value that leaves out the check producing *validator_name*."""
    if validator_name == "Secrets Detection":
        return "secrets"
    return _check_key(validator_name) or _short_name(validator_name)


def summarize_tier2(results: list[ValidationResult]) -> tuple[bool, bool, list[TierRow], str]:
    """Return (ran, passed, rows, skip_reason) for the dedup results."""
    if not results:
        return False, True, [], "no deduplication results returned"
    skipped = [r for r in results if _is_skipped(r)]
    if skipped and len(skipped) == len(results):
        return False, True, [], _skip_reason(skipped[0])
    failed = [r for r in results if not r.passed and not _is_skipped(r)]
    advisories = sum(len(r.findings) for r in results if r.passed)
    rows: list[TierRow] = []
    if failed:
        details: list[tuple[str, str]] = []
        for result in failed:
            # An advisory LLM refusal never explains why a result failed; the
            # reports still carry it.
            findings = [finding for finding in result.findings if not _tier2_finding_is_advisory_refusal(finding)]
            if findings:
                details.extend(
                    (
                        "error" if _tier2_finding_is_scan_failure(finding) else "duplicate",
                        str(finding.message),
                    )
                    for finding in findings
                )
            else:
                details.extend(("error", str(error)) for error in result.errors)

        # Resource ceilings and provider/traversal failures are represented as
        # structured findings, but they mean duplicate analysis did not finish.
        found_duplicates = any(label == "duplicate" for label, _message in details)
        headline = "duplicates found" if found_duplicates else "scan failed"
        rows.append(TierRow("context", [(headline, f"bold {RED}")]))
        errors = [message for label, message in details if label == "error"]
        for message in errors[:3]:
            rows.append(TierRow("error", [(message[:110], TEXT)], glyph="✗", glyph_style=f"bold {RED}"))
        if len(errors) > 3:
            rows.append(TierRow("", [(f"… {len(errors) - 3} more — see the report", MUTED)]))
        duplicates = [
            finding for result in failed for finding in result.findings if not _tier2_finding_is_scan_failure(finding)
        ]
        fix_rows = fix_section_rows([_tier2_fix(finding) for finding in duplicates])
    else:
        fix_rows = []
        rows.append(TierRow("context", [("clean", TEXT), ("  no duplicate guidance", MUTED)]))
    if advisories:
        rows.append(TierRow("advisories", [(str(advisories), TEXT), ("  catalog", MUTED)]))
    return True, not failed, rows + fix_rows, ""


_DUPLICATE_SECTION = re.compile(r'^(?:vs )?"(?P<title>.+)" in (?P<file>\S+) \(lines (?P<lines>[\d-]+)\)$')


def _tier2_fix(finding) -> FixItem:
    """A duplicate finding as a fix item; the section list collapses to ``"Title" lines a-b`` parts."""
    first, *rest = [line.strip() for line in str(finding.message).splitlines() if line.strip()] or [""]
    sections = [_DUPLICATE_SECTION.match(line) for line in rest]
    if rest and all(sections):
        first = first.rstrip(":") + ": " + " · ".join(f'"{m["title"]}" lines {m["lines"]}' for m in sections)
    elif rest:
        first = " ".join([first, *rest])
    where = Path(str(finding.file_path or "")).name or "context"
    return FixItem(where, first, [str(finding.suggestion)] if finding.suggestion else [])


def _tier2_finding_is_scan_failure(finding: object) -> bool:
    check_name = str(getattr(finding, "check_name", "")).casefold()
    return check_name in _TIER2_SCAN_FAILURE_CHECKS or check_name.endswith("_limit") or check_name.endswith("_error")


def _tier3_source_validation_rows(result: ValidationResult, *, report_formats: tuple[str, ...]) -> list[TierRow]:
    """Count the failed Tier 3 source checks and, unless the CLI report lists them, show them."""
    errors = [error for finding in result.findings for error in source_validation_errors(finding.metadata)]
    if not errors:
        return []
    count = len(errors)
    summary = f"{count} source validation {'error' if count == 1 else 'errors'}"
    if "cli" in report_formats:
        return [detail_row("validation", summary)]
    shown, overflow_note = source_validation_preview(errors, report_formats)
    if len(shown) < count:
        summary += f" (showing first {len(shown)})"
    rows = [detail_row("validation", summary)]
    for error in shown:
        text = f"{error['path']}: {error['message']}" if error["path"] else error["message"]
        rows.append(detail_row("source", sanitize_tier3_console_text(text)))
    if overflow_note:
        rows.append(detail_row("", overflow_note))
    return rows


def _tier3_run_dir(payload: dict) -> Path | None:
    provenance = payload.get("provenance")
    raw = provenance.get("run_dir") if isinstance(provenance, dict) else None
    path = Path(raw).resolve() if isinstance(raw, str) and raw else None
    return path if path is not None and path.is_dir() else None


def _tier3_final_marks(run_dir: Path) -> dict[str, dict[str, dict[str, list[str]]]]:
    from skillevaluator.tier3.case_summary import load_case_marks

    try:
        return load_case_marks(run_dir)
    except OSError:
        return {}


def _tier3_findings(run_dir: Path) -> list[dict] | None:
    from skillevaluator.tier3.case_summary import load_case_findings

    return load_case_findings(run_dir)


def _tier2_finding_is_advisory_refusal(finding: object) -> bool:
    # A policy can raise a refusal to a blocking severity; it then reads as a scan failure.
    if getattr(finding, "check_name", None) != CONTENT_DEDUP_LLM_REFUSED_CHECK:
        return False
    return str(getattr(finding, "severity", "")).casefold() not in {"critical", "high"}


def summarize_tier3(
    result: ValidationResult,
    *,
    report_formats: Iterable[str] = (),
) -> tuple[bool, bool, list[TierRow], str]:
    """Return (ran, passed, rows, skip_reason) for the agent-eval result.

    Failed Tier 3 source checks add a count row. ``report_formats`` are the
    formats the run emits. When they include ``cli``, the full CLI report lists
    the errors under the finding, so the panel keeps only the count. Otherwise
    the panel lists them, and shortens the list only when an HTML or JSON
    report written by the run keeps every error.
    """
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    raw_payload = metadata.get("agent_eval")
    payload = raw_payload if isinstance(raw_payload, dict) else {}
    raw_summary = payload.get("summary")
    has_summary = isinstance(raw_summary, dict) and bool(raw_summary)
    if _is_skipped(result) or (not result.passed and not has_summary):
        return False, True, [], _skip_reason(result)
    summary = raw_summary if has_summary else payload
    raw_agents = summary.get("agents_run")
    if not isinstance(raw_agents, list) or not raw_agents:
        raw_agents = payload.get("agents_run")
    agents = _tier3_string_list(raw_agents)
    agent_label = sanitize_tier3_console_text(", ".join(agents)) or "n/a"
    agent_segments: list[tuple[str, str]] = [(agent_label, TEXT)]
    run_dir = _tier3_run_dir(payload)
    case_rows = tier3_case_rows(_tier3_final_marks(run_dir), agents=agents or None) if run_dir else []
    case_list = payload.get("cases")
    if case_rows:
        rows = case_rows
    else:
        if isinstance(case_list, list) and case_list:
            agent_segments.append((f"  eval cases {len(case_list)}", MUTED))
        rows = [TierRow("agent", agent_segments)]
    if payload.get("verdict") == "inconclusive" or summary.get("verdict") == "inconclusive":
        rows.append(TierRow("verdict", [("INCONCLUSIVE: skill never activated", TEXT)]))
    lift = _tier3_finite_number(summary.get("overall_lift"))
    with_score = _tier3_finite_number(summary.get("overall_score"))
    if lift is not None and -1.0 <= lift <= 1.0 and with_score is not None and 0.0 <= with_score <= 1.0:
        baseline = with_score - lift
        if 0.0 <= baseline <= 1.0:
            rows.append(lift_row(lift, with_score, baseline))
    if case_rows and len(agents) > 1:
        rows = [row for row in rows if row.label != "lift"]
    fix_rows = tier3_fix_rows(_tier3_findings(run_dir), agents=len(agents)) if run_dir else []
    exec_status = payload.get("execution_status") or summary.get("execution_status")
    ok = bool(result.passed) and exec_status in (None, "succeeded")
    if not ok:
        errors = _tier3_string_list(payload.get("execution_errors"), max_items=1)
        if not errors:
            errors = _tier3_string_list(result.errors, max_items=1)
        reason = errors[0] if errors else "execution reported errors"
        rows.append(
            TierRow(
                "error",
                [(reason[:110], RED)],
                glyph="✗",
                glyph_style=f"bold {RED}",
            )
        )
    rows.extend(_tier3_source_validation_rows(result, report_formats=tuple(report_formats)))
    return True, ok, rows + fix_rows, ""


def _first_error(result: ValidationResult) -> str:
    if result.gate_failures:
        return result.gate_failures[0]
    for finding in result.findings:
        severity = getattr(finding.severity, "value", str(finding.severity)).lower()
        if severity in ("critical", "high"):
            return finding.message
    if result.findings:
        return result.findings[0].message
    if result.errors:
        return str(result.errors[0])
    return ""


def first_fix(results: list[ValidationResult]) -> list[tuple[str, str]] | None:
    """The single most actionable fix: the gating finding's suggestion.

    Prefers independent gate failures, then critical/high findings over
    advisory-severity ones, and trims to one terse directive line.
    """
    severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    def rank(finding) -> int:
        severity = getattr(finding.severity, "value", str(finding.severity)).lower()
        return severity_rank.get(severity, 4)

    for result in results:
        if result.passed or _is_skipped(result):
            continue
        for text in [
            *result.gate_failures,
            *(finding.suggestion or finding.message for finding in sorted(result.findings, key=rank)),
        ]:
            if text:
                text = text.split(". ")[0].strip().rstrip(".")
                if len(text) > 150:
                    text = text[:147] + "..."
                return [(text, TEXT)]
    return None
