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
from skillevaluator.harness.plugin import evaluate_plugin, is_plugin_root, plugin_facts
from skillevaluator.reporting.console_ui import AMBER, BRIGHT, FAINT, GREEN, MUTED, RED, SPIN, TEXT, WIDTH
from skillevaluator.validators.harness import HARNESS_METADATA_KEY, HarnessValidator

if TYPE_CHECKING:
    from skillevaluator.models.result import ValidationResult

FOOTER = "Doesn't change pass/fail."
VERBOSE_HINT = "  Details and fixes: -v"
_STATUS = {
    "loads": ("✓", GREEN, "loads", TEXT),
    "degraded": ("!", AMBER, "degraded", f"bold {AMBER}"),
    "not_loaded": ("✗", RED, "NOT LOADED", f"bold {RED}"),
    "pending": ("○", FAINT, "", FAINT),
}
_EFFECT = {
    "not_loaded": ("✗", RED, "NOT LOADED"),
    "degraded": ("!", AMBER, "degraded"),
    "advisory": ("·", MUTED, "note"),
}
_RANK = {"advisory": 0, "degraded": 1, "not_loaded": 2}

Block = dict[str, list[dict[str, Any]]]


def harness_block(results: Iterable[ValidationResult]) -> Block:
    """Collect the per-skill harness rows recorded by the ``harness`` check."""
    block: Block = {}
    for result in results:
        block.update((result.metadata or {}).get(HARNESS_METADATA_KEY) or {})
    return block


def harness_note(block: Block) -> str | None:
    """One-line verdict note naming the harnesses that drop or degrade the skill(s)."""
    parts = []
    for status, verb in (("not_loaded", "won't load in"), ("degraded", "loads degraded in")):
        skills = [skill for skill, rows in block.items() if any(row["status"] == status for row in rows)]
        names = sorted({row["harness"] for rows in block.values() for row in rows if row["status"] == status})
        if names:
            subject = "skill" if len(skills) == 1 else f"{len(skills)} skills"
            parts.append(f"{subject} {verb} {', '.join(names)}")
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


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'s' if count != 1 else ''}"


def _compact_grid(rows: list[dict[str, Any]]) -> Table | None:
    """One line per flagged harness: its most serious finding, then how many more."""
    grid = Table.grid(padding=(0, 2), expand=True)
    grid.add_column(no_wrap=True)
    grid.add_column(ratio=1, style=TEXT)
    grid.add_column(no_wrap=True, style=MUTED, justify="right")
    for row in rows:
        findings = row.get("findings") or []
        if row["status"] in ("pending", "checking") or not findings:
            continue
        worst = max(findings, key=lambda finding: _RANK[finding["effect"]])
        problems = sum(finding["effect"] != "advisory" for finding in findings)
        notes = len(findings) - problems
        glyph, color, label, _ = _STATUS[row["status"]]
        more = [f"+{problems - 1} more"] if problems > 1 else []
        more += [_plural(notes, "note")] if notes else []
        if worst["effect"] == "advisory":
            grid.add_row(Text(f"{glyph} {row['harness']}", f"bold {color}"), Text(label, TEXT), ", ".join(more))
            continue
        grid.add_row(
            Text(f"{glyph} {row['harness']}", f"bold {color}"),
            Text.assemble((worst.get("outcome") or label, f"bold {color}"), ("  ", ""), (worst["message"], TEXT)),
            ", ".join(more),
        )
    return grid if grid.row_count else None


def _catalog_line(rows: list[dict[str, Any]]) -> Text | None:
    catalog = next((row["catalog"] for row in rows if row.get("catalog")), None)
    if not catalog or catalog["size"] < 2:
        return None
    folder = catalog["folder"].replace(str(Path.home()), "~", 1) if catalog["folder"] else "this folder"
    return Text.assemble(
        ("catalog  ", MUTED), (f"{catalog['size']} skills in {folder} ({catalog['desc_chars']:,} desc chars)", TEXT)
    )


def _detail_grid(rows: list[dict[str, Any]], verbose: bool) -> Table | None:
    """Every finding with its fix, under a line naming the harness, its status and verified version."""
    grid = Table.grid(padding=(0, 2), expand=True)
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_column(ratio=1)
    for row in rows:
        if row["status"] in ("pending", "checking"):
            continue
        glyph, color, label, _ = _STATUS[row["status"]]
        version = f"verified {row.get('verified_version', '')}, {row.get('verified_date', '')}"
        grid.add_row(
            Text(f"{glyph} {row['harness']}", f"bold {color}"), Text(label, f"bold {color}"), Text(version, MUTED)
        )
        for finding in row.get("findings") or []:
            glyph, color, label = _EFFECT[finding["effect"]]
            grid.add_row(
                "",
                Text(f"{glyph} {label}", f"bold {color}" if finding["effect"] != "advisory" else MUTED),
                Text.assemble(
                    (finding["message"], TEXT),
                    (f"  [{finding['id']}]", MUTED),
                    *([("\n→ ", GREEN), (finding["fix"], MUTED)] if finding.get("fix") else []),
                    *([("\n  applies: ", MUTED), (finding["versions"], MUTED)] if finding.get("versions") else []),
                ),
            )
    return grid if grid.row_count else None


_LISTING = {
    "cut": "cut to {limit} chars",
    "name_only": "name only",
    "dropped": "not listed",
    "shortened": "shortened (over list budget)",
}


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
    title: str = "Harness compatibility",
    footer: str | None = FOOTER,
    verbose: bool = False,
    width: int = WIDTH,
) -> Panel:
    """One panel: a ✓/!/✗ line per skill, then its worst finding per harness (every finding with ``verbose``)."""
    parts: list[Any] = []
    many = len(block) > 1
    all_rows = [row for rows in block.values() for row in rows]
    for skill, rows in block.items():
        if parts:
            parts.append(Text())
        catalog = _catalog_line(rows)
        if catalog is not None:
            parts.extend([catalog, Text()])
        parts.append(_summary_line(rows, f"{Path(skill).name}   " if many else ""))
        grid = _detail_grid(rows, verbose) if verbose else _compact_grid(rows)
        if grid is not None:
            parts.extend([Text(), grid])
        listing = _listing_grid(rows) if verbose else None
        if listing is not None:
            parts.extend([Text(), Text("Description in the model's skill list (from the profile):", MUTED), listing])
    flagged = any(row.get("findings") for row in all_rows)
    if footer and (flagged or verbose):
        parts.extend([Text(), Text(footer + ("" if verbose else VERBOSE_HINT), MUTED)])
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
    config: dict[str, Any] | None = None,
) -> Block:
    """Evaluate each skill harness by harness, updating a live panel as each finishes."""
    target = target.parent if target.is_file() else target
    profiles = load_profiles(harness, profiles_dir)
    finder = HarnessValidator()
    skills = [target] if finder._is_skill_directory(target) else finder._find_all_skills(target)
    width = max(60, min(WIDTH, console.width))
    block: Block = {}
    if is_plugin_root(target):
        facts = plugin_facts(target)
        rows = [evaluate_plugin(profile, facts).to_dict() for profile in profiles]
        console.print(
            harness_panel(
                {str(target): rows},
                title=f"Plugin compatibility · {facts['plugin_name']}",
                verbose=verbose,
                width=width,
            )
        )
        console.print()
        block[f"{target} (plugin)"] = rows
    all_facts = facts_for(skills)
    for skill in skills:
        rows: list[dict[str, Any]] = [{"harness": p["harness"], "status": "pending"} for p in profiles]

        def panel(rows: list[dict[str, Any]] = rows, skill: Path = skill) -> Panel:
            return harness_panel(
                {str(skill): rows},
                title=f"Harness compatibility · {skill.name}",
                verbose=verbose,
                width=width,
            )

        with Live(panel(), console=console, refresh_per_second=12) as live:
            facts = all_facts[skill]
            for index, profile in enumerate(profiles):
                rows[index] = {"harness": profile["harness"], "status": "checking"}
                live.update(panel())
                rows[index] = evaluate_profile(profile, facts, config).to_dict()
                live.update(panel())
        console.print()
        block[str(skill)] = rows
    return block
