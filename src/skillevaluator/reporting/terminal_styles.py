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


def _artifact_uri(target: str, *, windows: bool) -> str | None:
    """Build a terminal-only URI without changing canonical URL normalization."""
    from pathlib import Path, PurePosixPath, PureWindowsPath
    from urllib.parse import quote, unquote, urlsplit

    from skillevaluator.utils.rich_markup import strip_terminal_controls

    def has_controls(value: str) -> bool:
        return any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)

    if strip_terminal_controls(target) != target or has_controls(target):
        return None
    windows_path = PureWindowsPath(target)
    if windows_path.drive:
        # A foreign Windows path must not become a relative POSIX file URI;
        # drive-relative paths and UNC shares are deliberately left as text.
        if windows and len(windows_path.drive) == 2 and windows_path.is_absolute():
            # PurePath.as_uri() is deprecated from Python 3.14; build the same URI.
            drive = windows_path.drive
            return "file:///" + drive + quote(windows_path.as_posix()[len(drive) :])
        return None
    try:
        parts = urlsplit(target)
        if parts.scheme:
            decoded = urlsplit(unquote(target))
            if (
                parts.scheme not in {"http", "https"}
                or not parts.hostname
                or parts.username is not None
                or parts.password is not None
                or decoded.username is not None
                or decoded.password is not None
                or any(char.isspace() for char in target)
                or has_controls(unquote(target))
            ):
                return None
            # Access validates malformed/out-of-range ports before linking.
            _ = parts.port
            return target
    except ValueError:
        return None
    if windows and PurePosixPath(target).is_absolute():
        return None
    return Path(target).expanduser().absolute().as_uri()


def artifact_text(target: str, console, *, label: str | None = None, prefix=None):
    """Defer link styling until render, keeping captured strings inert."""
    import os

    from rich.measure import Measurement
    from rich.style import Style
    from rich.text import Text

    from skillevaluator.utils.rich_markup import strip_terminal_controls

    text = Text()
    if prefix is not None:
        text.append_text(prefix)
    start = len(text)
    text.append(strip_terminal_controls(label if label is not None else target), style=MUTED)
    url = _artifact_uri(target, windows=os.name == "nt")

    def capable(output) -> bool:
        return (
            output.is_terminal
            and not output.no_color
            and not output.is_dumb_terminal
            and bool(getattr(output.file, "isatty", lambda: False)())
        )

    class ArtifactLine:
        def __rich_console__(self, output, options):
            rendered = text.copy()
            # Rich's normal print() owns one buffer context. capture() adds an
            # outer context, even when its underlying stream is a real TTY.
            # Check at render time so prebuilt/footer renderables remain safe.
            if url and capable(console) and capable(output) and getattr(output, "_buffer_index", 0) <= 1:
                rendered.stylize(Style(link=url), start)
            yield from rendered.__rich_console__(output, options)

        def __rich_measure__(self, output, options):
            return Measurement.get(output, options, text)

    return ArtifactLine()
