# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the advisory harness-compatibility table and JSON block."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from skillevaluator.harness.overlay import HarnessResult, evaluate_harnesses
from skillevaluator.validators.schema import SchemaValidator


def _skill_dirs(target: Path) -> list[Path]:
    finder = SchemaValidator()
    return [target] if finder._is_skill_directory(target) else finder._find_all_skills(target)


def emit_harness_compatibility(
    target: Path, harness: str, profiles_dir: Path | None, json_report: Path | None = None
) -> dict[str, list[HarnessResult]]:
    """Print one row per harness for each skill and add ``harness_compatibility`` to the JSON report."""
    try:
        results = {str(skill): evaluate_harnesses(skill, harness, profiles_dir) for skill in _skill_dirs(target)}
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    console = Console()
    for skill, rows in results.items():
        table = Table(
            title=f"Harness compatibility: {Path(skill).name} (advisory, does not change the result)",
            title_justify="left",
            show_edge=False,
            box=None,
            pad_edge=False,
        )
        for column in ("Harness", "Status", "Reason", "Verified"):
            table.add_column(column)
        for row in rows:
            reason = "; ".join(f"{f['message']} [{f['id']}]" for f in row.findings if f["effect"] != "advisory") or "-"
            table.add_row(row.harness, row.status, reason, f"{row.version}, {row.verified_date}")
        console.print(table)
    if json_report is not None and json_report.exists():
        data = json.loads(json_report.read_text(encoding="utf-8"))
        data["harness_compatibility"] = {skill: [r.to_dict() for r in rows] for skill, rows in results.items()}
        json_report.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return results
