# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Table rows for the Tier 3 ``token_usage`` payload, shared by every report.

The terminal, the Markdown report, ``BENCHMARK.md`` and the HTML report all
render the same cells from these rows, so the reports agree word for word.
Each renderer escapes the agent and case names for its own format.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from rich.console import Console

#: The explanation printed with every token usage table.
TOKEN_USAGE_NOTE = (
    "Measured agent token usage. It is not scored and changes no score or verdict. "
    "Tokens are prompt plus completion; prompt tokens include cached tokens, which are not added again. "
    "Attempts that failed before scoring are counted. A partial total covers only the attempts with "
    "token counts; nothing is estimated. Change is with skill minus baseline, as a share of the baseline, "
    "and needs complete counts for the same attempts in both arms."
)
#: Header of the coverage column: measured attempts of the expected ones in each arm.
COVERAGE_HEADER = "Measured attempts"
#: Explains the coverage column under every table.
COVERAGE_NOTE = (
    "Measured attempts: attempts with token counts, of those expected, in each arm. "
    "Unscored attempts failed before scoring."
)
#: Where the terminal, which prints agent rows only, sends readers for case rows.
PER_CASE_POINTER = "Token usage per case is in the HTML report and BENCHMARK.md."


@dataclass(frozen=True)
class TokenUsageRow:
    """One row: an agent's case, an agent's ``All cases`` sum, or the all-agent sum (``agent`` is ``None``)."""

    agent: str | None
    case: str
    is_case: bool
    with_skill: str
    baseline: str
    change: str
    coverage: str
    note: str | None = None


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _coverage(arm: object) -> Mapping[str, Any]:
    return _mapping(_mapping(arm).get("coverage"))


def _arm_cell(arm: object) -> str:
    coverage = _coverage(arm)
    if coverage.get("status") == "not_run":
        return "not run"
    total = _count(_mapping(_mapping(arm).get("totals")).get("total_tokens"))
    if total is None or not _count(coverage.get("measured_attempts")):
        return "N/A"
    return f"{total:,}" if coverage.get("status") == "complete" else f"{total:,} (partial)"


def _change_cell(delta: object) -> str:
    delta = _mapping(delta)
    difference = delta.get("total_tokens")
    if delta.get("status") != "available" or type(difference) is not int:
        return "N/A"
    percent = delta.get("total_tokens_change_percent")
    if isinstance(percent, int | float) and not isinstance(percent, bool):
        return f"{difference:+,} ({percent:+.1f}%)"
    return f"{difference:+,}"


def _coverage_part(label: str, arm: object) -> str:
    coverage = _coverage(arm)
    if coverage.get("status") == "not_run":
        return f"{label} not run"
    measured = _count(coverage.get("measured_attempts")) or 0
    expected = _count(coverage.get("expected_attempts")) or 0
    unscored = _count(coverage.get("unscored_attempts")) or 0
    return f"{label} {measured}/{expected}" + (f" ({unscored} unscored)" if unscored else "")


def _row(agent: str | None, case: str | None, item: Mapping[str, Any]) -> TokenUsageRow:
    """Build one row; ``case`` is ``None`` for an ``All cases`` sum, which alone carries a note."""
    delta = _mapping(item.get("delta"))
    reason = delta.get("reason")
    not_run = any(_coverage(item.get(arm)).get("status") == "not_run" for arm in ("with_skill", "baseline"))
    note = reason if case is None and isinstance(reason, str) and reason and not not_run else None
    return TokenUsageRow(
        agent,
        "All cases" if case is None else case,
        case is not None,
        _arm_cell(item.get("with_skill")),
        _arm_cell(item.get("baseline")),
        _change_cell(delta),
        "; ".join((_coverage_part("skill", item.get("with_skill")), _coverage_part("baseline", item.get("baseline")))),
        note,
    )


def token_usage_rows(
    usage: object, *, include_cases: bool = True, max_case_rows: int | None = None
) -> tuple[list[TokenUsageRow], int]:
    """Return the rows of ``usage`` and how many case rows ``max_case_rows`` left out.

    Each agent gets an ``All cases`` row followed by its case rows; the
    all-agent row comes last when more than one agent ran. Only agent and
    all-agent rows carry a note: a case's change is unavailable for the same
    reasons, which its cells already show.
    """
    usage = _mapping(usage)
    agents = _mapping(usage.get("agents"))
    rows: list[TokenUsageRow] = []
    omitted = 0
    case_rows = 0
    for name, agent in agents.items():
        if not isinstance(name, str) or not isinstance(agent, Mapping):
            continue
        rows.append(_row(name, None, agent))
        if not include_cases:
            continue
        cases = agent.get("cases")
        for case in cases if isinstance(cases, list) else []:
            if not isinstance(case, Mapping):
                continue
            if max_case_rows is not None and case_rows >= max_case_rows:
                omitted += 1
                continue
            case_rows += 1
            rows.append(_row(name, str(case.get("entry_id") or "unknown"), case))
    dataset = usage.get("dataset")
    if isinstance(dataset, Mapping) and sum(not row.is_case for row in rows) > 1:
        rows.append(_row(None, None, dataset))
    return rows, omitted


def token_usage_markdown_lines(
    usage: object,
    *,
    heading: str,
    agent_label: Callable[[str | None], str],
    cell: Callable[[str], str],
    max_case_rows: int,
) -> list[str]:
    """Return a Markdown token usage section: agent rows, then case rows in a collapsed table.

    ``agent_label`` gives the display name of an agent (``None`` is all
    agents) and ``cell`` escapes an untrusted case name for a table cell.
    """
    rows, omitted = token_usage_rows(usage, max_case_rows=max_case_rows)
    if not rows:
        return []
    header = f"| Agent | Case | With skill | Baseline | Change | {COVERAGE_HEADER} |"
    separator = "|---|---|---:|---:|---:|---|"

    def line(row: TokenUsageRow) -> str:
        cells = (agent_label(row.agent), cell(row.case), row.with_skill, row.baseline, row.change, row.coverage)
        return "| " + " | ".join(cells) + " |"

    summary = [row for row in rows if not row.is_case]
    cases = [row for row in rows if row.is_case]
    lines = [heading, "", TOKEN_USAGE_NOTE, "", header, separator, *(line(row) for row in summary), ""]
    lines.extend([COVERAGE_NOTE, ""])
    notes = [f"- {agent_label(row.agent)}: {row.note}" for row in summary if row.note]
    if notes:
        lines.extend([*notes, ""])
    if cases:
        lines.extend(["<details>", "<summary>Show token usage per case</summary>", "", header, separator])
        lines.extend(line(row) for row in cases)
        lines.append("")
        if omitted:
            lines.extend([f"{omitted} more case row(s) are in the HTML report and the JSON payload.", ""])
        lines.extend(["</details>", ""])
    return lines


def render_token_usage_table(console: Console, usage: object, *, text: Callable[[str], str] = str) -> bool:
    """Print one row per agent and the all-agent row, then the notes; return whether anything printed.

    ``text`` makes agent names and notes safe for the console, for example by
    redacting secrets. Cells are plain text, never console markup.
    """
    rows, _omitted = token_usage_rows(usage, include_cases=False)
    if not rows:
        return False
    table = Table(title="Token Usage (measured, not scored)", show_header=True, header_style="bold")
    table.add_column("Agent", style="bold")
    for column in ("With skill", "Baseline", "Change"):
        table.add_column(column, justify="right")
    table.add_column(COVERAGE_HEADER)
    for row in rows:
        table.add_row(
            Text(text(row.agent if row.agent is not None else "All agents")),
            Text(row.with_skill),
            Text(row.baseline),
            Text(row.change),
            Text(row.coverage),
        )
    console.print(table)
    for row in rows:
        if row.note:
            console.print(Text(f"  {text(row.agent if row.agent is not None else 'All agents')}: {text(row.note)}"))
    console.print(Text(f"  {TOKEN_USAGE_NOTE} {COVERAGE_NOTE} {PER_CASE_POINTER}", style="dim"))
    return True


def token_usage_table(usage: object) -> dict[str, Any] | None:
    """Return the HTML report's token usage table: every row and the notes, or ``None`` without rows."""
    rows, _omitted = token_usage_rows(usage)
    return {"rows": rows, "note": TOKEN_USAGE_NOTE, "coverage_note": COVERAGE_NOTE} if rows else None
