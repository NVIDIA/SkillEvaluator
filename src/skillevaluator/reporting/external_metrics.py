# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain reports for imported observations, deliberately separate from live runs."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

from skillevaluator.tier3.external_metrics import ARMS, load_external_metrics, summarize_external_metrics


def _text(value: object) -> str:
    return re.sub(r"([\\`*{}\[\]()#+.!_|>~-])", r"\\\1", html.escape(str(value), quote=False))


def _score(value: float | None) -> str:
    return "NO SCORE" if value is None else f"{value:.4f}"


def render_external_metrics_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Imported custom metrics",
        "",
        "External observations — SkillEvaluator did not execute these tests or verify their source claims.",
        "Only JSON validation, custom-metric aggregation and paired comparison ran locally.",
        "No canonical dimensions, Overall, pass/fail or publication evidence are assigned.",
        "",
        f"Source SHA-256: `{report['source_sha256']}`",
        f"Imported at: {_text(report['imported_at'])}",
        "",
        "## Declared provenance and evaluation context",
        "",
    ]
    for key, value in {**report["provenance"], **report["evaluation"]}.items():
        lines.append(f"- {_text(key)}: {_text(value)}")
    lines.extend(
        [
            "",
            "## Custom scores (0-1)",
            "",
            "Arm scores are means over available observations; n/total shows coverage, not test success.",
            "Missing scores are excluded, never replaced with zero.",
            "",
            "| Custom metric | Version | Unit | Direction | With skill (n/total) | Without skill (n/total) | Paired delta | Improvement | Paired cases/attempts |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for name, definition in sorted(report["metric_definitions"].items()):
        values = []
        for arm in ARMS:
            summary = report["arms"][arm]
            score = _score(summary["custom_metrics"].get(name))
            values.append(f"{score} ({summary['counts'][name]}/{summary['observations']})")
        lift = report["custom_lift"][name]
        cells = [
            _text(name),
            _text(definition["version"]),
            _text(definition["unit"]),
            _text(definition["direction"]),
            *values,
            _score(lift["delta"]),
            _score(lift["improvement"]),
            f"{lift['paired_cases']}/{lift['paired_attempts']}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Paired delta is with-skill minus without-skill for matching case/attempt IDs in this declared context.",
            "Matched attempt deltas are averaged within each case, then equally across cases.",
            "It may differ from subtraction of arm means when coverage differs.",
            "Improvement reverses the delta for lower-is-better metrics; positive means better.",
            "NO SCORE means no compatible scored pairs, not zero lift. No statistical significance is claimed.",
            "",
        ]
    )
    return "\n".join(lines)


def import_external_metrics(source: Path, output_dir: Path) -> dict[str, Any]:
    """Validate before writing to a new directory; never overwrite a saved live run."""
    contract, digest = load_external_metrics(source)
    report = summarize_external_metrics(contract, digest)
    markdown = render_external_metrics_markdown(report)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "external_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "external_metrics.md").write_text(markdown, encoding="utf-8")
    return report
