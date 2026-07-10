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
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TextIO

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    from skillevaluator.models.result import ValidationResult

GREEN = "#76B900"  # NVIDIA green: success + identity moments only
RED = "#F5524C"  # failure highlights only
TEXT = "#D9DEE3"  # body
MUTED = "#8A8F98"  # labels / secondary
FAINT = "#565B66"  # chrome, pending, bar remainders
BRIGHT = "#EDEDED"  # the actively-running element
INK = "#101403"  # pill/badge foreground on green fills

_TIER2_SCAN_FAILURE_CHECKS = frozenset(
    {
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
SPIN = "◐◓◑◒"

PENDING = "pending"
RUNNING = "running"
PASS = "pass"
FAIL = "fail"
SKIP = "skip"


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

    def __init__(self, on_tail, max_lines: int = 4):
        self._on_tail = on_tail
        self._max_lines = max_lines
        self._tail: list[str] = []
        self._secrets: set[str] = set()

    @property
    def is_active(self) -> bool:
        return True

    def start(self, _plan) -> None:  # Tier3RunPlan — the view renders its own config rows
        return None

    def set_secret_values(self, values) -> None:
        self._secrets.update(str(v) for v in values if v)

    def emit(self, event) -> None:
        from skillevaluator.tier3.harbor.progress import redact_progress_detail

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
        return None


def make_view_console(file: TextIO | None = None) -> Console:
    """A console for view-level output.

    Resolves ``sys.stdout`` at CALL time: do not call inside the Tier 3
    engine-capture window (``redirect_stdout``) or the output lands in the
    capture buffer -- pass the already-pinned view console's file instead.
    """
    return Console(highlight=False, file=file if file is not None else sys.stdout)


def _pill(label: str, color: str, ink: str = INK) -> Text:
    return Text(f" {label} ", style=f"bold {ink} on {color}")


@dataclass
class TierRow:
    """One detail line inside a tier section."""

    label: str
    segments: list[tuple[str, str]]
    glyph: str = " "
    glyph_style: str = MUTED


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

    lift_style = f"bold {GREEN}" if lift >= 0 else f"bold {RED}"
    segments: list[tuple[str, str]] = [
        (f"{lift:+.2f}", lift_style),
        ("   with-skill ", MUTED),
        (f"{with_score:.2f} ", f"bold {TEXT}"),
        *bar(with_score, GREEN),
        ("   baseline ", MUTED),
        (f"{baseline:.2f} ", f"bold {TEXT}"),
        *bar(baseline, MUTED),
    ]
    return TierRow("lift", segments)


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
        self.console = console or Console(highlight=False, file=sys.stdout)
        # Design width is 98 columns; degrade gracefully on narrower terminals.
        self.width = max(60, min(WIDTH, self.console.width))
        self.blocks = [TierBlock(number=number, name=name, caption=caption) for number, name, caption in tiers]
        self.verdict: Verdict | None = None
        self.links: list[tuple[str, str]] = []
        self._live: Live | None = None
        self._static_printed = 0  # blocks already flushed in non-tty mode

    # ── lifecycle ────────────────────────────────────────────────

    def start(self) -> None:
        if not self.enabled:
            return
        if self.console.is_terminal:
            self._live = Live(
                self,
                console=self.console,
                refresh_per_second=8,
                vertical_overflow="visible",
            )
            self._live.start()
        else:
            self.console.print(self._header())
            self.console.print()

    def tier_start(self, index: int) -> None:
        if not self.enabled:
            return
        block = self.blocks[index]
        block.status = RUNNING
        block.started_at = time.time()
        self._refresh()

    def tier_progress(self, index: int, rows: list[TierRow]) -> None:
        """Update a running tier's detail rows (check ticker / run config)."""
        if not self.enabled:
            return
        self.blocks[index].rows = rows
        self._refresh()

    def tier_done(self, index: int, *, failed: bool = False, rows: list[TierRow] | None = None) -> None:
        if not self.enabled:
            return
        block = self.blocks[index]
        block.status = FAIL if failed else PASS
        block.duration = time.time() - block.started_at if block.started_at else None
        block.rows = rows or []
        self._refresh()
        self._flush_static()

    def tier_skip(self, index: int, reason: str) -> None:
        if not self.enabled:
            return
        block = self.blocks[index]
        block.status = SKIP
        block.rows = [TierRow("skipped", [(reason, MUTED)])]
        self._refresh()
        self._flush_static()

    def finish(self, verdict: Verdict, links: list[tuple[str, str]]) -> None:
        if not self.enabled:
            return
        self.verdict = verdict
        self.links = links
        if self._live is not None:
            self._live.update(self)
            self._live.stop()
            self._live = None
        else:
            if len(self.blocks) >= 2:
                self.console.print(self._gate())
                self.console.print()
            self.console.print(self._verdict_panel(verdict))
            self.console.print()
            for line in self._footer_lines():
                self.console.print(line)

    def stop(self) -> None:
        """Tear down the live display (safe on errors / KeyboardInterrupt)."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    # ── rendering ────────────────────────────────────────────────

    def __rich_console__(self, console, options):  # rich renderable protocol
        yield self._header()
        yield Text()
        if len(self.blocks) >= 2:
            yield self._gate()
            yield Text()
        for block in self.blocks:
            if block.status == PENDING:
                continue
            yield self._block_panel(block)
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
        right = _pill(self.skill, GREEN)
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
        for row in block.rows:
            t = Text()
            t.append(f"{row.glyph} ", style=row.glyph_style)
            t.append(f"{row.label:<{LABEL}}", style=MUTED)
            for chunk, style in row.segments:
                t.append(chunk, style=style)
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
        if verdict.passed:
            body = _pill("✓ PASS", GREEN) + Text.assemble(
                ("  ", ""), (verdict.headline, f"bold {TEXT}"), ("  ·  exit 0", MUTED)
            )
            return Panel(body, box=box.ROUNDED, border_style=GREEN, width=self.width, padding=(0, 2))
        line1 = _pill("✗ FAIL", RED, ink="#1C0605") + Text.assemble(
            ("  ", ""), (verdict.headline, f"bold {TEXT}"), ("  ·  exit 1", MUTED)
        )
        parts: list = [line1]
        if verdict.fix:
            fix = Text.assemble(("fix     ", f"bold {MUTED}"))
            for chunk, style in verdict.fix:
                fix.append(chunk, style=style)
            parts.extend([Text(), fix])
        if verdict.rerun:
            parts.append(Text.assemble(("        ", ""), (verdict.rerun, GREEN)))
        return Panel(Group(*parts), box=box.ROUNDED, border_style=RED, width=self.width, padding=(0, 2))

    def _footer_lines(self):
        for label, target in self.links:
            yield Text.assemble(("      ", ""), (f"{label:<{LABEL}}", MUTED), (target, MUTED))

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self)

    def _flush_static(self) -> None:
        """Non-tty mode: print blocks once, as they complete."""
        if self._live is not None or self.console.is_terminal:
            return
        for block in self.blocks[self._static_printed :]:
            if block.status in (PENDING, RUNNING):
                break
            self.console.print(self._block_panel(block))
            self._static_printed += 1


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


def _is_skipped(result: ValidationResult) -> bool:
    """True when a validator was recorded without executing (OSS conventions)."""
    if bool(getattr(result, "skipped", False)):
        return True
    meta = result.metadata or {}
    if meta.get("skipped") or meta.get("execution_status") == "skipped":
        return True
    payload = meta.get("agent_eval") or {}
    summary = payload.get("summary") or {}
    return payload.get("execution_status") == "skipped" or summary.get("execution_status") == "skipped"


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
    for result in results:
        key = _check_key(result.validator_name)
        if key is None or _is_skipped(result):
            continue
        if not result.passed:
            status[key] = FAIL
        elif status.get(key) != FAIL:
            status[key] = PASS
    segments: list[tuple[str, str]] = []
    for i, name in enumerate(lineup):
        if i:
            segments.append((" · ", FAINT))
        state = status.get(name)
        if state == FAIL:
            segments.append((f"✗ {name}", f"bold {RED}"))
        elif state == PASS:
            segments.append((f"✓ {name}", GREEN))
        else:
            segments.append((name, FAINT))
    return TierRow("checks", segments)


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
    total = len(results)
    failed = [r for r in results if not r.passed and not _is_skipped(r)]
    # Warnings from passing validators surface as advisories below; only the
    # failing validators' warnings belong on the headline checks row.
    warnings = sum(len(r.warnings) for r in failed)
    advisory: dict[str, int] = {}
    for r in results:
        if r.passed and not _is_skipped(r) and r.findings:
            advisory[_short_name(r.validator_name)] = advisory.get(_short_name(r.validator_name), 0) + len(r.findings)

    rows: list[TierRow] = []
    if lineup:
        rows.append(final_check_ticker_row(lineup, results))
    checks_segments: list[tuple[str, str]] = [(f"{total - len(failed)}/{total} passed", TEXT)]
    if failed:
        checks_segments += [("  ·  ", FAINT), (f"{len(failed)} failed", f"bold {RED}")]
    if warnings:
        checks_segments += [("  ·  ", FAINT), (f"{warnings} warning{'s' if warnings != 1 else ''}", MUTED)]
    rows.append(TierRow("summary" if lineup else "checks", checks_segments))

    for r in failed:
        reason = _first_error(r)
        rows.append(
            TierRow(
                "failed",
                [(r.validator_name, f"bold {TEXT}"), (f" — {reason}" if reason else "", RED)],
                glyph="✗",
                glyph_style=f"bold {RED}",
            )
        )

    if advisory:
        detail = " · ".join(f"{name} {count}" for name, count in sorted(advisory.items(), key=lambda kv: -kv[1]))
        rows.append(TierRow("advisories", [(str(sum(advisory.values())), TEXT), (f"  {detail}", MUTED)]))

    quality = _quality_stats(results)
    if quality:
        grade, score = quality
        rows.append(TierRow("quality", [(grade, f"bold {GREEN}"), (f"  {score:.1f} / 100", TEXT)]))
    return not failed, rows


def summarize_tier2(results: list[ValidationResult]) -> tuple[bool, bool, list[TierRow], str]:
    """Return (ran, passed, rows, skip_reason) for the dedup results."""
    if not results:
        return False, True, [], "no deduplication results returned"
    skipped = [r for r in results if _is_skipped(r)]
    if skipped and len(skipped) == len(results):
        first = skipped[0]
        reason = str(
            (first.metadata or {}).get("skip_reason")
            or (first.warnings[0] if first.warnings else "prerequisite unavailable")
        )
        return False, True, [], reason
    failed = [r for r in results if not r.passed and not _is_skipped(r)]
    advisories = sum(len(r.findings) for r in results if r.passed)
    rows: list[TierRow] = []
    if failed:
        details: list[tuple[str, str]] = []
        for result in failed:
            if result.findings:
                details.extend(
                    (
                        "error" if _tier2_finding_is_scan_failure(finding) else "duplicate",
                        str(finding.message),
                    )
                    for finding in result.findings
                )
            else:
                details.extend(("error", str(error)) for error in result.errors)

        # Resource ceilings and provider/traversal failures are represented as
        # structured findings, but they mean duplicate analysis did not finish.
        found_duplicates = any(label == "duplicate" for label, _message in details)
        headline = "duplicates found" if found_duplicates else "scan failed"
        rows.append(TierRow("context", [(headline, f"bold {RED}")]))
        # Show the actual overlaps, or the errors that stopped the scan.
        shown = 0
        for label, message in details:
            if shown >= 3:
                continue
            rows.append(
                TierRow(
                    label,
                    [(message[:110], TEXT)],
                    glyph="✗",
                    glyph_style=f"bold {RED}",
                )
            )
            shown += 1
        if len(details) > shown:
            rows.append(TierRow("", [(f"… {len(details) - shown} more — see the report", MUTED)]))
    else:
        rows.append(TierRow("context", [("clean", TEXT), ("  no duplicate guidance", MUTED)]))
    if advisories:
        rows.append(TierRow("advisories", [(str(advisories), TEXT), ("  catalog", MUTED)]))
    return True, not failed, rows, ""


def _tier2_finding_is_scan_failure(finding: object) -> bool:
    check_name = str(getattr(finding, "check_name", "")).casefold()
    return check_name in _TIER2_SCAN_FAILURE_CHECKS or check_name.endswith("_limit") or check_name.endswith("_error")


def summarize_tier3(result: ValidationResult) -> tuple[bool, bool, list[TierRow], str]:
    """Return (ran, passed, rows, skip_reason) for the agent-eval result."""
    if _is_skipped(result) or (not result.passed and not (result.metadata or {}).get("agent_eval", {}).get("summary")):
        reason = str(
            (result.metadata or {}).get("skip_reason")
            or (result.warnings[0] if result.warnings else "prerequisite unavailable")
        )
        return False, True, [], reason
    payload = (result.metadata or {}).get("agent_eval") or {}
    summary = payload.get("summary") or payload
    agents = summary.get("agents_run") or payload.get("agents_run") or []
    agent_segments: list[tuple[str, str]] = [(", ".join(agents) or "n/a", TEXT)]
    case_list = payload.get("cases") or []
    if case_list:
        # A plain count: per-case pass/fail lives in the report, and rendering
        # a fabricated N/N ratio here would overstate what we measured.
        agent_segments.append((f"  eval cases {len(case_list)}", MUTED))
    rows = [TierRow("agent", agent_segments)]
    lift = summary.get("overall_lift")
    with_score = summary.get("overall_score")
    if isinstance(lift, (int, float)) and isinstance(with_score, (int, float)):
        baseline = max(0.0, min(1.0, float(with_score) - float(lift)))
        rows.append(lift_row(float(lift), float(with_score), baseline))
    exec_status = payload.get("execution_status") or summary.get("execution_status")
    ok = bool(result.passed) and exec_status in (None, "succeeded")
    if not ok:
        errors = list(payload.get("execution_errors") or []) or list(result.errors)
        reason = str(errors[0]) if errors else "execution reported errors"
        rows.append(
            TierRow(
                "error",
                [(reason[:110], RED)],
                glyph="✗",
                glyph_style=f"bold {RED}",
            )
        )
    return True, ok, rows, ""


def _first_error(result: ValidationResult) -> str:
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

    Prefers critical/high findings (the ones that actually failed the gate)
    over advisory-severity ones, and trims to one terse directive line.
    """
    severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    def rank(finding) -> int:
        severity = getattr(finding.severity, "value", str(finding.severity)).lower()
        return severity_rank.get(severity, 4)

    for result in results:
        if result.passed or _is_skipped(result):
            continue
        for finding in sorted(result.findings, key=rank):
            text = finding.suggestion or finding.message
            if text:
                text = text.split(". ")[0].strip().rstrip(".")
                if len(text) > 150:
                    text = text[:147] + "..."
                return [(text, TEXT)]
    return None
