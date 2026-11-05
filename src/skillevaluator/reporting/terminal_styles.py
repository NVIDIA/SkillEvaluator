# SPDX-License-Identifier: Apache-2.0
"""Lightweight semantic terminal styles shared by report presentations."""

from __future__ import annotations

GREEN = "#76B900"
RED = "#F5524C"
TEXT = "default"
MUTED = "#8A8F98"
FAINT = "#565B66"
BRIGHT = "default"
INK = "#101403"
AMBER = "#E5A33B"


def grade_style(grade: str, *, status: str = "passed") -> str:
    """Keep unavailable or incomplete evidence from looking like a passing grade."""
    if status in {"failed", "fail"}:
        return RED
    if status in {"incomplete", "degraded"}:
        return AMBER
    if status not in {"passed", "pass", "succeeded", "complete"}:
        return MUTED
    return {"A": GREEN, "B": GREEN, "C": AMBER, "D": RED, "F": RED}.get(grade, MUTED)


def lift_style(value: float) -> str:
    """Style the displayed two-decimal lift; rounded zero is neutral."""
    displayed = float(f"{value:.2f}")
    return GREEN if displayed > 0 else RED if displayed < 0 else MUTED


def artifact_text(target: str, console, *, label: str | None = None):
    """Link only on a capable terminal; captured strings remain readable text."""
    from pathlib import Path
    from urllib.parse import urlsplit

    from rich.style import Style
    from rich.text import Text

    from skillevaluator.reporting.harbor_viewer import safe_url
    from skillevaluator.utils.rich_markup import strip_terminal_controls

    clean = strip_terminal_controls(target)
    text = Text(strip_terminal_controls(label) if label is not None else clean, style=MUTED)
    if (
        clean != target
        or not console.is_terminal
        or console.no_color
        or console.is_dumb_terminal
        or not bool(getattr(console.file, "isatty", lambda: False)())
    ):
        return text
    try:
        scheme = urlsplit(clean).scheme
    except ValueError:
        return text
    url = safe_url(clean) if scheme else Path(clean).expanduser().absolute().as_uri()
    if url:
        text.stylize(Style(link=url))
    return text
