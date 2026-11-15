# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared display projections for recorded declarative-check coverage."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from rich.console import Console

DENOMINATOR_NOTE = (
    "Behavior denominator = checked + judged per reward; not checked is excluded. "
    "Totals sum these denominators, not a pooled behavior score. A judge error still yields no score."
)


def declarative_coverage_groups(agents: object) -> list[dict[str, Any]]:
    groups = []
    if not isinstance(agents, Mapping):
        return groups
    for name, agent in agents.items():
        coverage = agent.get("declarative_checks") if isinstance(agent, Mapping) else None
        if not isinstance(coverage, Mapping):
            continue
        for arm in ("with_skill", "without_skill"):
            data = coverage.get(arm)
            if not isinstance(data, dict) or not data.get("results"):
                continue
            groups.append({**data, "agent": str(name), "arm": arm.replace("_", " ")})
    return groups


def coverage_summary(group: dict[str, Any]) -> str:
    def count(key: str) -> str:
        value = group.get(key)
        return str(value) if value is not None else "unavailable"

    omitted = group.get("omitted_checks")
    shown = f"; {omitted} more check rows omitted to fit report limits" if type(omitted) is int and omitted > 0 else ""
    return (
        f"Checked {count('checked_count')} ({count('passed_count')} passed / {count('failed_count')} failed); "
        f"judged {count('judged_count')}; not checked {count('not_checked_count')}; "
        f"behavior denominator {count('score_denominator')}{shown}"
    )


def check_status(item: dict[str, Any]) -> str:
    return "passed" if item.get("passed") is True else "failed" if item.get("passed") is False else "not checked"


def attempt_denominators(group: dict[str, Any]) -> str:
    return "; ".join(
        f"{item['case_id']} [{item['trial_id']}]: "
        f"{item['score_denominator'] if item['score_denominator'] is not None else 'unavailable'} "
        f"({item['score_status']})"
        for item in group.get("attempts", [])
    )


def render_declarative_coverage(console: Console, agents: object, *, text: Callable[[str], str] = str) -> None:
    groups = declarative_coverage_groups(agents)
    if not groups:
        return
    console.print(Text("\nDeclarative-check coverage", style="bold"))
    console.print(Text(DENOMINATOR_NOTE, style="dim"))
    for group in groups:
        console.print(Text(text(f"{group['agent']} / {group['arm']}: {coverage_summary(group)}")))
        console.print(Text(text("Per-reward denominators: " + attempt_denominators(group)), style="dim"))
        table = Table(expand=True)
        for label in ("Case / trial", "Check", "Expected", "Status", "Reason"):
            table.add_column(label, overflow="fold")
        for item in group["results"]:
            table.add_row(
                *(
                    Text(text(str(value)))
                    for value in (
                        f"{item['case_id']} / {item['trial_id']}",
                        item["type"],
                        item["expected"],
                        check_status(item),
                        item["reason"],
                    )
                )
            )
        console.print(table)
