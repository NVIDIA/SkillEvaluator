# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Colorized Click help rendering for the SkillEvaluator CLI.

Click's stock help is plain text. This module provides drop-in ``click.Group`` /
``click.Command`` subclasses that re-render the same help (usage, description,
options, sub-commands) through :mod:`rich` -- already a base dependency -- so
``skillevaluator --help`` matches the branded, color-coded ``skill-evaluator -h`` look.

Color degrades gracefully: when stdout is not a TTY (pipes, CI, ``CliRunner``)
or ``NO_COLOR`` is set, rich emits plain text, so the help stays greppable and
the existing CLI tests keep passing.
"""

from __future__ import annotations

import inspect
import os
import re
import sys

import click
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Color scheme for the help screen. Kept intentionally small and semantic so the
# palette can be tweaked in one place.
_HELP_THEME = Theme(
    {
        "help.usage": "bold",
        "help.prog": "bold cyan",
        "help.metavar": "cyan",
        "help.title": "bold green",
        "help.heading": "bold yellow",
        "help.option": "cyan",
        "help.command": "bold cyan",
        "help.text": "default",
        "help.comment": "dim",
    }
)

_INDENT = (0, 0, 0, 2)


class GroupedOption(click.Option):
    """``click.Option`` that records the help section it belongs to.

    Pass ``help_group="Tier 3: Live Agent Evaluation"`` (or any label) to place
    the option under its own heading in the rich help output, mirroring the
    grouped sections in ``skill-evaluator validate -h``. Options without a ``help_group``
    fall under the default ``Options:`` heading.
    """

    def __init__(self, *args: object, help_group: str | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.help_group = help_group


def _wants_color(ctx: click.Context | None) -> bool:
    """Decide whether to emit ANSI color, mirroring Click/rich conventions."""
    if ctx is not None and ctx.color is not None:
        return ctx.color
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _usage(command: click.Command, ctx: click.Context) -> Text:
    text = Text()
    text.append("Usage: ", style="help.usage")
    text.append(ctx.command_path, style="help.prog")
    for piece in command.collect_usage_pieces(ctx):
        text.append(" ")
        text.append(piece, style="help.metavar")
    return text


def _description(command: click.Command) -> RenderableType | None:
    raw = command.help or command.short_help
    if not raw:
        return None
    # Match Click: drop the post-``\f`` body and re-wrap paragraphs.
    cleaned = inspect.cleandoc(raw).split("\f", 1)[0].strip()
    if not cleaned:
        return None
    paragraphs = re.split(r"\n\s*\n", cleaned)
    blocks: list[RenderableType] = []
    for index, paragraph in enumerate(paragraphs):
        collapsed = " ".join(paragraph.split())
        if not collapsed:
            continue
        style = "help.title" if index == 0 else "help.text"
        if blocks:
            blocks.append(Text(""))
        blocks.append(Padding(Text(collapsed, style=style), _INDENT, expand=False))
    return Group(*blocks) if blocks else None


def _definition_table(rows: list[tuple[str, str]], *, key_style: str) -> RenderableType:
    table = Table.grid(padding=(0, 2, 0, 0))
    table.add_column(style=key_style, no_wrap=True)
    table.add_column(style="help.text")
    for key, value in rows:
        table.add_row(Text(key), Text(value or ""))
    return Padding(table, _INDENT, expand=False)


def _options(command: click.Command, ctx: click.Context) -> RenderableType | None:
    # Partition options by their (optional) help_group, preserving declaration
    # order. ``None`` is the default, untitled "Options:" section.
    grouped: dict[str | None, list[tuple[str, str]]] = {}
    seen: list[str | None] = []
    for param in command.get_params(ctx):
        record = param.get_help_record(ctx)
        if record is None:
            continue
        name = getattr(param, "help_group", None)
        if name not in grouped:
            grouped[name] = []
            seen.append(name)
        grouped[name].append((record[0], record[1]))
    if not grouped:
        return None

    # Default section first, then named sections in first-seen order.
    ordered: list[str | None] = ([None] if None in grouped else []) + [n for n in seen if n is not None]
    descriptions = getattr(command, "help_group_descriptions", {}) or {}

    blocks: list[RenderableType] = []
    for index, name in enumerate(ordered):
        if index:
            blocks.append(Text(""))
        blocks.append(Text("Options:" if name is None else f"{name}:", style="help.heading"))
        description = descriptions.get(name)
        if description:
            collapsed = " ".join(description.split())
            blocks.append(Padding(Text(collapsed, style="help.text"), _INDENT, expand=False))
            blocks.append(Text(""))
        blocks.append(_definition_table(grouped[name], key_style="help.option"))
    return Group(*blocks)


def _commands(command: click.Command, ctx: click.Context) -> RenderableType | None:
    if not isinstance(command, click.Group):
        return None
    rows: list[tuple[str, str]] = []
    for name in command.list_commands(ctx):
        sub = command.get_command(ctx, name)
        if sub is None or getattr(sub, "hidden", False):
            continue
        # Large limit so rich (not Click) handles wrapping in the help column,
        # avoiding the truncating "..." Click otherwise inserts.
        rows.append((name, sub.get_short_help_str(limit=120)))
    if not rows:
        return None
    return Group(Text("Commands:", style="help.heading"), _definition_table(rows, key_style="help.command"))


def _epilog_line(line: str) -> Text:
    """Color one pre-formatted epilog line (raw layout is preserved verbatim)."""
    stripped = line.rstrip()
    # no_wrap + ignore overflow keeps the author's raw layout intact (argparse
    # RawDescriptionHelpFormatter parity); we keep lines within 80 cols anyway.
    opts = {"no_wrap": True, "overflow": "ignore"}
    if not stripped:
        return Text("")
    # Section headers sit at column 0 and end with a colon.
    if not stripped[0].isspace() and stripped.endswith(":"):
        return Text(stripped, style="help.heading", **opts)
    # Dim a trailing "  # comment" (used in the Examples block).
    marker = stripped.find(" #")
    if marker > 0:
        text = Text(**opts)
        text.append(stripped[: marker + 1], style="help.text")
        text.append(stripped[marker + 1 :], style="help.comment")
        return text
    return Text(stripped, style="help.text", **opts)


def _epilog(command: click.Command) -> RenderableType | None:
    if not command.epilog:
        return None
    # Render the epilog raw (preserve the author's line layout, like argparse's
    # RawDescriptionHelpFormatter that skill-evaluator uses) with light colorization.
    lines = inspect.cleandoc(command.epilog).split("\n")
    return Group(*[_epilog_line(line) for line in lines]) if lines else None


def render_help(command: click.Command, ctx: click.Context, *, width: int | None = None) -> str:
    """Render a command's help screen as a (optionally colorized) string."""
    color = _wants_color(ctx)
    # Use an explicit color system (not "auto") so styles survive ``capture()``,
    # which has no attached terminal to auto-detect against. "standard" (8/16
    # colors) is universally supported and covers our palette.
    console = Console(
        theme=_HELP_THEME,
        width=width or 80,
        highlight=False,
        force_terminal=color,
        color_system="standard" if color else None,
    )

    sections = [
        _usage(command, ctx),
        _description(command),
        _options(command, ctx),
        _commands(command, ctx),
        _epilog(command),
    ]

    body: list[RenderableType] = []
    for section in sections:
        if section is None:
            continue
        if body:
            body.append(Text(""))
        body.append(section)

    with console.capture() as capture:
        console.print(Group(*body))
    # Strip the trailing column padding rich adds to aligned table cells so the
    # output has no trailing whitespace (parity with Click's own help).
    return "\n".join(line.rstrip() for line in capture.get().splitlines()) + "\n"


class RichCommand(click.Command):
    """``click.Command`` whose ``--help`` is rendered with rich."""

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        formatter.write(render_help(self, ctx, width=formatter.width))


class RichGroup(click.Group):
    """``click.Group`` with rich help that propagates to sub-commands/-groups."""

    command_class = RichCommand
    group_class = type  # sub-groups created via @group.group() reuse RichGroup

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        formatter.write(render_help(self, ctx, width=formatter.width))
