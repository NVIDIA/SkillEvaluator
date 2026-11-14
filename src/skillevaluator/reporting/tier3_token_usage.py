# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Table rows for the Tier 3 ``token_usage`` payload, shared by every report.

The terminal, the Markdown report, ``BENCHMARK.md`` and the HTML report all
render the same cells from these rows, so the reports agree word for word.
Each renderer escapes the agent and case names for its own format.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
#: Said under the table of a run that reused an earlier run's baseline (``--reuse-baseline``).
REUSED_BASELINE_NOTE = (
    "This run reused the baseline of an earlier run, so the baseline tokens were used in that run, not in this one."
)
#: Where ``validate``'s terminal report, which prints agent rows only, sends readers for case rows.
VALIDATE_CASE_POINTER = "Token usage per case is in the HTML, Markdown and JSON reports (-r html, markdown or json)."
#: Where ``tier3 evaluate``'s terminal summary sends readers for case rows.
EVALUATE_CASE_POINTER = "Token usage per case is in report.html and tier3-benchmark.md in the run directory."
#: Case rows per agent in the HTML table; the JSON payload keeps the rest.
HTML_CASE_ROWS_PER_AGENT = 64
#: Marks a case row whose ID is not one of the evaluation dataset's case IDs.
UNVERIFIED_CASE_MARK = " (unverified case ID)"
#: Explains that mark wherever a table carries one.
UNVERIFIED_CASE_NOTE = (
    "A row marked (unverified case ID) names an attempt whose case could not be matched to any case ID in the "
    "evaluation dataset. Its tokens were measured, but the row does not stand for a dataset case: read it as a "
    "report-integrity problem, not as coverage of that case."
)
#: Replaces that explanation where no marked row is rendered, naming the cases instead.
UNVERIFIED_CASE_IDS_NOTE = (
    "{count} case ID(s) could not be matched to any case ID in the evaluation dataset ({names}); this table shows "
    "no row for them. Their tokens were measured, but they do not stand for dataset cases: read this as a "
    "report-integrity problem, not as coverage of those cases."
)
#: Case IDs named in :data:`UNVERIFIED_CASE_IDS_NOTE` before the rest are summarized.
UNVERIFIED_CASE_IDS_NAMED = 10


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
    if case is not None and item.get("case_id_in_dataset") is False:
        case += UNVERIFIED_CASE_MARK
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


def more_case_rows_note(count: int, *, code: Callable[[str], str] = lambda text: f"`{text}`") -> str | None:
    """Say how many case rows a table left out; ``code`` formats the payload key."""
    return f"{count} more case row(s) are in the JSON payload under {code('token_usage')}." if count else None


def reused_baseline_note(usage: object) -> str | None:
    """Say that the baseline tokens come from an earlier run when this run reused its baseline."""
    return REUSED_BASELINE_NOTE if _mapping(usage).get("baseline_reused") is True else None


def unverified_case_ids_note(
    usage: object,
    rows: Sequence[TokenUsageRow] = (),
    *,
    case_label: Callable[[str], str] = str,
) -> str | None:
    """Explain the ``(unverified case ID)`` mark, or name the cases *rows* leaves out.

    The mark lives on a case row, so its explanation is only useful once a
    rendered row carries one. A table whose row caps or payload pruning
    dropped every such row, and one that renders no case row at all, says
    instead how many case IDs could not be matched and names them, so the
    reader is not sent hunting for a mark that is not in the table.
    ``case_label`` applies the renderer's redaction and escaping to each full
    ID before its display text is bounded.
    """
    unmatched = _mapping(_mapping(usage).get("case_id_validation")).get("unmatched_case_ids")
    names = (
        sorted({case for case in unmatched if isinstance(case, str) and case}) if isinstance(unmatched, list) else []
    )
    if not names:
        return None
    if any(row.is_case and row.case.endswith(UNVERIFIED_CASE_MARK) for row in rows):
        return UNVERIFIED_CASE_NOTE
    listed = []
    for case in names[:UNVERIFIED_CASE_IDS_NAMED]:
        # Exact-secret and private-label filters must see the full ID.
        label = case_label(case)
        listed.append(label[:120] + ("…" if len(label) > 120 else ""))
    if len(names) > len(listed):
        listed.append(f"and {len(names) - len(listed)} more")
    return UNVERIFIED_CASE_IDS_NOTE.format(count=len(names), names=", ".join(listed))


def dropped_case_rows_note(usage: object) -> str | None:
    """Say how many case rows the report payload dropped to stay within the report limits, if any."""
    count = _count(_mapping(usage).get("case_rows_omitted"))
    if not count:
        return None
    return (
        f"{count:,} case row(s) were left out of this report to keep it within the report size limits; "
        "the agent sums cover every attempt."
    )


def token_usage_rows(
    usage: object,
    *,
    include_cases: bool = True,
    max_case_rows: int | None = None,
    max_case_rows_per_agent: int | None = None,
) -> tuple[list[TokenUsageRow], int]:
    """Return the rows of ``usage`` and how many case rows the caps left out.

    Each agent gets an ``All cases`` row followed by its case rows; the
    all-agent row comes last when more than one agent ran. ``max_case_rows``
    caps the case rows of all agents together and ``max_case_rows_per_agent``
    those of each agent. Only agent and all-agent rows carry a note: a case's
    change is unavailable for the same reasons, which its cells already show.
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
        agent_rows = 0
        for case in cases if isinstance(cases, list) else []:
            if not isinstance(case, Mapping):
                continue
            if (max_case_rows is not None and case_rows >= max_case_rows) or (
                max_case_rows_per_agent is not None and agent_rows >= max_case_rows_per_agent
            ):
                omitted += 1
                continue
            case_rows += 1
            agent_rows += 1
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
    if reused := reused_baseline_note(usage):
        lines.extend([reused, ""])
    notes = [f"- {agent_label(row.agent)}: {row.note}" for row in summary if row.note]
    if notes:
        lines.extend([*notes, ""])
    if dropped := dropped_case_rows_note(usage):
        lines.extend([dropped, ""])
    if unverified := unverified_case_ids_note(usage, rows, case_label=cell):
        lines.extend([unverified, ""])
    if cases:
        lines.extend(["<details>", "<summary>Show token usage per case</summary>", "", header, separator])
        lines.extend(line(row) for row in cases)
        lines.append("")
        if more := more_case_rows_note(omitted):
            lines.extend([more, ""])
        lines.extend(["</details>", ""])
    return lines


def render_token_usage_table(
    console: Console, usage: object, *, case_pointer: str, text: Callable[[str], str] = str
) -> bool:
    """Print one row per agent and the all-agent row, then the notes; return whether anything printed.

    ``case_pointer`` names the files of the calling command that hold the
    case rows. ``text`` makes agent names and notes safe for the console, for
    example by redacting secrets. Cells are plain text, never console markup.
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
    if reused := reused_baseline_note(usage):
        console.print(Text(f"  {reused}"))
    for row in rows:
        if row.note:
            console.print(Text(f"  {text(row.agent if row.agent is not None else 'All agents')}: {text(row.note)}"))
    pointer = " ".join(
        part
        for part in (
            case_pointer,
            dropped_case_rows_note(usage),
            unverified_case_ids_note(usage, rows, case_label=text),
        )
        if part
    )
    console.print(Text(f"  {TOKEN_USAGE_NOTE} {COVERAGE_NOTE} {pointer}", style="dim"))
    return True


def token_usage_table(usage: object) -> dict[str, Any] | None:
    """Return the HTML report's token usage table, or ``None`` without rows.

    Each agent shows its first :data:`HTML_CASE_ROWS_PER_AGENT` case rows;
    ``more_cases`` says how many more are in the JSON payload.
    """
    rows, omitted = token_usage_rows(usage, max_case_rows_per_agent=HTML_CASE_ROWS_PER_AGENT)
    if not rows:
        return None
    return {
        "rows": rows,
        "note": TOKEN_USAGE_NOTE,
        "coverage_note": COVERAGE_NOTE,
        "reused_baseline": reused_baseline_note(usage),
        "more_cases": more_case_rows_note(omitted, code=str),
        "dropped_cases": dropped_case_rows_note(usage),
        "unverified_cases": unverified_case_ids_note(usage, rows),
    }
