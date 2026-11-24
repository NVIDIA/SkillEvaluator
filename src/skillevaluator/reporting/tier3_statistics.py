# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Table rows for the advisory Tier 3 run statistics, shared by every report.

The terminal, the Markdown report and the HTML report render the same cells
from these tables. The statistics are the per-agent ``lift_uncertainty`` and
``reliability`` blocks. They are advisory: nothing here changes a score,
verdict or exit code. Every value is read defensively, since a saved run's
``statistics.json`` is untrusted input; a value that is missing or out of range
shows as ``N/A``. Each renderer escapes agent names for its own format.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING, Any

from rich.table import Table
from rich.text import Text

from skillevaluator.constants import LIFT_BOOTSTRAP_CONFIDENCE
from skillevaluator.tier3.harbor.metrics import finite_number

if TYPE_CHECKING:
    from rich.console import Console

#: Title of the terminal table, and caption of the Markdown and HTML tables, of the Skill Lift interval.
LIFT_TITLE = "Lift Uncertainty (advisory)"
#: The same for pass^k beside pass@k.
RELIABILITY_TITLE = "Reliability (advisory)"
#: The explanation printed with the statistics tables.
STATISTICS_NOTE = (
    "Advisory statistics. They are not scored and change no score, verdict or exit code. "
    "Intervals resample whole cases of this run only; they do not establish that the skill generalizes."
)
LIFT_HEADERS = (
    "Agent",
    "Skill Lift (paired case mean)",
    f"{round(LIFT_BOOTSTRAP_CONFIDENCE * 100)}% interval",
    "Precision",
    "Paired cases",
)
RELIABILITY_HEADERS = ("Agent", "Arm", "pass@k", "pass^k", "Attempts (k)")
_ARMS = (("with_skill", "With skill"), ("without_skill", "Baseline"))
_PRECISIONS = frozenset({"insufficient", "low", "adequate"})
_MAX_AGENTS = 64
_NA = "N/A"


@dataclass(frozen=True)
class StatisticsRow:
    """One agent's cells after its name; ``note`` is shown under the table."""

    agent: str
    cells: tuple[str, ...]
    note: str | None = None


@dataclass(frozen=True)
class StatisticsTable:
    """A titled table; the first ``left_columns`` columns, the agent name included, are left-aligned."""

    title: str
    headers: tuple[str, ...]
    rows: list[StatisticsRow]
    left_columns: int


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: object, low: float, high: float) -> float | None:
    """Return *value* as a finite number within ``[low, high]``, else ``None``; booleans are not numbers."""
    number = finite_number(value)
    return number if number is not None and low <= number <= high else None


def _count(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= 1_000_000_000 else None


def _percent(value: object) -> str:
    number = _number(value, 0.0, 1.0)
    return _NA if number is None else f"{number * 100:.0f}%"


def _agent_items(agents: object) -> list[tuple[str, Mapping[str, Any]]]:
    return [
        (name, agent)
        for name, agent in islice(_mapping(agents).items(), _MAX_AGENTS)
        if isinstance(name, str) and isinstance(agent, Mapping)
    ]


def _lift_row(name: str, agent: Mapping[str, Any]) -> StatisticsRow | None:
    interval = _mapping(_mapping(agent.get("lift_uncertainty")).get("effectiveness"))
    estimate = _number(interval.get("estimate"), -1.0, 1.0)
    if estimate is None:
        return None
    low, high = _number(interval.get("ci_low"), -1.0, 1.0), _number(interval.get("ci_high"), -1.0, 1.0)
    text = _NA if low is None or high is None else f"[{low:+.2f}, {high:+.2f}]"
    if text != _NA and interval.get("ci_includes_zero") is True:
        text += " includes zero"
    cases, expected = _count(interval.get("n_cases")), _count(interval.get("expected_cases"))
    note = None
    if interval.get("partial") is True:
        note = f"partial: {cases} of {expected} expected cases" if cases is not None and expected else "partial"
    precision = interval.get("precision")
    cells = (
        f"{estimate:+.2f}",
        text,
        precision if isinstance(precision, str) and precision in _PRECISIONS else _NA,
        _NA if cases is None else str(cases),
    )
    return StatisticsRow(name, cells, note)


def _reliability_rows(name: str, agent: Mapping[str, Any]) -> list[StatisticsRow]:
    """One row per arm that made more than one attempt per case and has a pass^k.

    With a single attempt pass^k equals pass@k, and an early stop leaves pass^k
    unknown, so the row would repeat what the report already shows.
    """
    reliability = _mapping(agent.get("reliability"))
    rows = []
    for arm, label in _ARMS:
        summary = _mapping(reliability.get(arm))
        k = _count(summary.get("k"))
        if not k or k < 2 or _number(summary.get("pass_hat_k"), 0.0, 1.0) is None:
            continue
        cells = (label, _percent(summary.get("pass_at_k")), _percent(summary.get("pass_hat_k")), str(k))
        rows.append(StatisticsRow(name, cells))
    return rows


def statistics_tables(agents: object) -> list[StatisticsTable]:
    """Return the Skill Lift and reliability tables of *agents*; a table with no usable row is left out."""
    lift: list[StatisticsRow] = []
    reliability: list[StatisticsRow] = []
    for name, agent in _agent_items(agents):
        if row := _lift_row(name, agent):
            lift.append(row)
        reliability.extend(_reliability_rows(name, agent))
    return [
        StatisticsTable(title, headers, rows, left_columns)
        for title, headers, rows, left_columns in (
            (LIFT_TITLE, LIFT_HEADERS, lift, 1),
            (RELIABILITY_TITLE, RELIABILITY_HEADERS, reliability, 2),
        )
        if rows
    ]


def render_statistics_tables(console: Console, agents: object, *, text: Callable[[str], str] = str) -> bool:
    """Print the Skill Lift interval and reliability tables and the note.

    Returns whether anything printed. ``text`` makes agent names safe for the
    console, for example by redacting secrets. Cells are plain text, never
    console markup.
    """
    tables = statistics_tables(agents)
    if not tables:
        return False
    for statistics in tables:
        table = Table(title=statistics.title, show_header=True, header_style="bold")
        for index, header in enumerate(statistics.headers):
            table.add_column(
                header,
                justify="left" if index < statistics.left_columns else "right",
                style="bold" if index == 0 else "",
            )
        for row in statistics.rows:
            table.add_row(Text(text(row.agent)), *(Text(cell) for cell in row.cells))
        console.print(table)
        for row in statistics.rows:
            if row.note:
                console.print(Text(f"  {text(row.agent)}: {row.note}", style="dim"))
    console.print(Text(f"  {STATISTICS_NOTE}", style="dim"))
    return True


def statistics_markdown_lines(
    agents: object,
    *,
    heading: str,
    agent_label: Callable[[str], str],
) -> list[str]:
    """Return a Markdown statistics section, or ``[]`` when there is nothing to show.

    ``agent_label`` escapes an agent name for a table cell.
    """
    tables = statistics_tables(agents)
    if not tables:
        return []
    out = [heading, "", STATISTICS_NOTE, ""]
    for table in tables:
        rule = (
            "|" + "|".join("---" if index < table.left_columns else "---:" for index in range(len(table.headers))) + "|"
        )
        out.extend([f"**{table.title}**", "", _row(table.headers), rule])
        out.extend(_row((agent_label(row.agent), *row.cells)) for row in table.rows)
        out.append("")
        notes = [f"- {agent_label(row.agent)}: {row.note}" for row in table.rows if row.note]
        if notes:
            out.extend([*notes, ""])
    return out


def _row(cells: tuple[str, ...]) -> str:
    return "| " + " | ".join(cells) + " |"


def statistics_section(agents: object) -> dict[str, Any] | None:
    """Return the HTML report's statistics tables and note, or ``None`` without anything to show."""
    tables = statistics_tables(agents)
    return {"tables": tables, "note": STATISTICS_NOTE} if tables else None
