# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the advisory harness-compatibility panel and JSON block."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from skillevaluator.harness.overlay import evaluate_profile, facts_for, load_profiles
from skillevaluator.reporting.console_ui import AMBER, BRIGHT, FAINT, GREEN, MUTED, RED, SPIN, TEXT, WIDTH
from skillevaluator.validators.harness import HARNESS_METADATA_KEY, HarnessValidator

if TYPE_CHECKING:
    from skillevaluator.models.result import ValidationResult

VALIDATE_FOOTER = "Doesn't change the result. Details: skillevaluator harness-check <skill>"
_STATUS = {
    "loads": ("✓", GREEN, "loads", TEXT),
    "degraded": ("!", AMBER, "degraded", f"bold {AMBER}"),
    "not_loaded": ("✗", RED, "NOT LOADED", f"bold {RED}"),
    "pending": ("○", FAINT, "", FAINT),
}
_EFFECT = {
    "not_loaded": ("✗", RED, "NOT LOADED"),
    "degraded": ("!", AMBER, "degraded"),
    "advisory": ("·", MUTED, "advisory"),
}

Block = dict[str, list[dict[str, Any]]]


def harness_block(results: Iterable[ValidationResult]) -> Block:
    """Collect the per-skill harness rows recorded by the ``harness`` check."""
    block: Block = {}
    for result in results:
        block.update((result.metadata or {}).get(HARNESS_METADATA_KEY) or {})
    return block


def harness_note(block: Block) -> str | None:
    """One-line verdict note naming harnesses that drop or degrade a skill."""
    parts = []
    for status, label in (("not_loaded", "won't load in"), ("degraded", "degraded in")):
        names = sorted({row["harness"] for rows in block.values() for row in rows if row["status"] == status})
        if names:
            parts.append(f"{label} {', '.join(names)}")
    return " · ".join(parts) or None


def _glyph(row: dict[str, Any]) -> tuple[str, str]:
    if row["status"] == "checking":
        return SPIN[int(time.time() * 8) % len(SPIN)], f"bold {BRIGHT}"
    glyph, color, _, _ = _STATUS[row["status"]]
    return glyph, f"bold {color}"


def _summary_line(rows: list[dict[str, Any]], prefix: str = "") -> Text:
    line = Text(prefix, style=f"bold {TEXT}")
    for index, row in enumerate(rows):
        glyph, style = _glyph(row)
        line.append("   " if index else "", style="")
        line.append(f"{glyph}\u00a0", style=style)
        line.append(row["harness"], style=TEXT if row["status"] not in ("pending", "checking") else MUTED)
    return line


def _detail_grid(rows: list[dict[str, Any]], verbose: bool) -> Table | None:
    grid = Table.grid(padding=(0, 2), expand=True)
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_column(ratio=1)
    if verbose:
        grid.add_column(no_wrap=True, style=MUTED)
    count = 0
    for row in rows:
        findings = row.get("findings") or []
        if row["status"] in ("pending", "checking") or (not findings and not verbose):
            continue
        version = f"{row.get('verified_version', '')}, {row.get('verified_date', '')}"
        if not findings:
            cells = [Text(f"✓ {row['harness']}", GREEN), Text("loads", TEXT), Text("-", MUTED)]
            grid.add_row(*cells, *([version] if verbose else []))
            count += 1
        for index, finding in enumerate(findings):
            glyph, color, label = _EFFECT[finding["effect"]]
            cells = [
                Text(f"{glyph} {row['harness']}" if index == 0 else "", f"bold {color}"),
                Text(label, f"bold {color}" if finding["effect"] != "advisory" else MUTED),
                Text.assemble(
                    (finding["message"], TEXT),
                    (f"  [{finding['id']}]", MUTED),
                    *([("\n→ ", GREEN), (finding["fix"], MUTED)] if finding.get("fix") else []),
                ),
            ]
            grid.add_row(*cells, *([version if index == 0 else ""] if verbose else []))
            count += 1
    return grid if count else None


_LISTING = {"cut": "cut to {limit} chars", "name_only": "name only", "dropped": "not listed"}


def _listing_grid(rows: list[dict[str, Any]]) -> Table | None:
    grid = Table.grid(padding=(0, 2), expand=True)
    grid.add_column(no_wrap=True, style=TEXT)
    grid.add_column(no_wrap=True, style=MUTED)
    grid.add_column(ratio=1, style=MUTED)
    for row in rows:
        listed = row.get("listed") or {}
        if listed.get("mode", "full") == "full":
            continue
        text = listed.get("text", "")
        shown = f"“{text[:57]}…”" if len(text) > 57 else f"“{text}”" if text else "-"
        grid.add_row(row["harness"], _LISTING[listed["mode"]].format(limit=listed.get("limit", "")), shown)
    return grid if grid.row_count else None


def _subtitle(rows: list[dict[str, Any]]) -> tuple[Text, str]:
    done = [row for row in rows if row["status"] not in ("pending", "checking")]
    if len(done) < len(rows):
        return Text(f"checking {len(done)}/{len(rows)}", MUTED), FAINT
    total = len(rows)
    for status, label, color in (("not_loaded", "not loaded", RED), ("degraded", "degraded", AMBER)):
        count = sum(row["status"] == status for row in rows)
        if count:
            glyph = _STATUS[status][0]
            return Text(f"{glyph} {count} of {total} {label}", f"bold {color}"), color
    return Text(f"✓ {total} of {total} load", f"bold {GREEN}"), GREEN


def harness_panel(
    block: Block,
    *,
    title: str = "Harness compatibility · advisory",
    footer: str | None = VALIDATE_FOOTER,
    verbose: bool = False,
    show_listing: bool = False,
    width: int = WIDTH,
) -> Panel:
    """One panel: a ✓/!/✗ line per skill, then a row per harness finding."""
    parts: list[Any] = []
    many = len(block) > 1
    all_rows = [row for rows in block.values() for row in rows]
    for skill, rows in block.items():
        if parts:
            parts.append(Text())
        parts.append(_summary_line(rows, f"{Path(skill).name}   " if many else ""))
        grid = _detail_grid(rows, verbose)
        if grid is not None:
            parts.extend([Text(), grid])
        listing = _listing_grid(rows) if show_listing or verbose else None
        if listing is not None:
            parts.extend([Text(), Text("Description in the model's skill list (from the profile):", MUTED), listing])
    flagged = any(row.get("findings") for row in all_rows)
    if footer and (flagged or verbose):
        parts.extend([Text(), Text(footer, MUTED)])
    subtitle, border = _subtitle(all_rows)
    return Panel(
        Group(*parts),
        box=box.ROUNDED,
        border_style=border,
        width=width,
        padding=(0, 2),
        title=Text(title, f"bold {TEXT}"),
        title_align="left",
        subtitle=subtitle,
        subtitle_align="right",
    )


def write_json_block(json_report: Path, block: Block) -> None:
    """Add the ``harness_compatibility`` block to an existing JSON report."""
    if json_report.exists():
        data = json.loads(json_report.read_text(encoding="utf-8"))
        data[HARNESS_METADATA_KEY] = block
        json_report.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_harness_check(
    target: Path,
    harness: str,
    profiles_dir: Path | None,
    *,
    console: Console,
    verbose: bool = False,
) -> Block:
    """Evaluate each skill harness by harness, updating a live panel as each finishes."""
    profiles = load_profiles(harness, profiles_dir)
    finder = HarnessValidator()
    skills = [target] if finder._is_skill_directory(target) else finder._find_all_skills(target)
    width = max(60, min(WIDTH, console.width))
    block: Block = {}
    all_facts = facts_for(skills)
    for skill in skills:
        rows: list[dict[str, Any]] = [{"harness": p["harness"], "status": "pending"} for p in profiles]

        def panel(rows: list[dict[str, Any]] = rows, skill: Path = skill) -> Panel:
            return harness_panel(
                {str(skill): rows},
                title=f"Harness compatibility · {skill.name}",
                footer="Advisory: doesn't change any validation result.",
                verbose=verbose,
                show_listing=True,
                width=width,
            )

        with Live(panel(), console=console, refresh_per_second=12) as live:
            facts = all_facts[skill]
            for index, profile in enumerate(profiles):
                rows[index] = {"harness": profile["harness"], "status": "checking"}
                live.update(panel())
                rows[index] = evaluate_profile(profile, facts).to_dict()
                live.update(panel())
        console.print()
        block[str(skill)] = rows
    return block
