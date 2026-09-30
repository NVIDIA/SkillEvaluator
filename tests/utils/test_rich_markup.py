# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``escape_markup`` must make any text render literally inside Rich markup."""

from __future__ import annotations

import io
import itertools
import time

import pytest
from rich.console import Console

from skillevaluator.utils.rich_markup import escape_markup


def _render(markup: str) -> str:
    console = Console(file=io.StringIO(), record=True, width=10_000, highlight=False)
    console.print(markup, soft_wrap=True)
    return console.export_text()[:-1]


@pytest.mark.parametrize(
    "text",
    [
        "[/x]",
        "[bold]evil[/bold]",
        "[link=http://x]y[/link]",
        "[#fff]x",
        "[@click]y",
        # Windows paths: Rich turns a plain-text "\[" into "[", dropping the separator.
        r"\work\skills\[\x]",
        r"C:\Users\runner\Temp\[abc]\report.json",
        r"\\server\share\[a]",
        r"\[/x]",
        r"[a\[b]",
        "[a[b]",
        "[[b]]",
        "[",
        "]",
        "x[",
        "[]",
        "plain",
        # Trailing backslashes must stay literal and must not escape the closing tag.
        "C:\\dir\\",
        "a\\\\",
        "a\\\\\\",
    ],
)
def test_escaped_text_renders_literally_between_tags(text: str) -> None:
    assert _render(f"[green]<[/green]{escape_markup(text)}[green]>[/green]") == f"<{text}>"


def test_escaped_text_renders_literally_between_plain_text() -> None:
    text = r"C:\a\[b]\c [/x] \\[bold]"
    assert _render(f"pre {escape_markup(text)} post") == f"pre {text} post"


def test_every_short_string_renders_literally() -> None:
    alphabet = ["[", "]", "\\", "/", "a", "#", "="]
    for length in range(6):
        for chars in itertools.product(alphabet, repeat=length):
            text = "".join(chars)
            assert _render(f"[green]<[/green]{escape_markup(text)}[green]>[/green]") == f"<{text}>", text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("clear\x1b[2Jscreen", "clearscreen"),
        ("up\x1b[1A\x1b[2Kerase", "uperase"),
        ("\x1b]8;;https://evil.test/\x1b\\see docs\x1b]8;;\x1b\\", "see docs"),
        ("title\x1b]0;forged\x07 set", "title set"),
        ("bad\x1bc thing", "bad thing"),
        ("charset\x1b(B reset", "charset reset"),
        ("eight-bit\x9b2J csi", "eight-bit2J csi"),
        ("nul\x00 bel\x07 del\x7f", "nul bel del"),
        ("crlf\r\nline", "crlf\nline"),
        ("tab\tand\nnewline", "tab\tand\nnewline"),
    ],
)
def test_terminal_controls_are_stripped(text: str, expected: str) -> None:
    assert escape_markup(text) == expected


def test_controls_are_stripped_before_brackets_are_escaped() -> None:
    text = "[\x1b[0m/x]"
    assert _render(f"[green]<[/green]{escape_markup(text)}[green]>[/green]") == "<[/x]>"


@pytest.mark.parametrize("unit", ["\\", "[a", "\\[", "[a]", "\x1b[", "\x1b]8;;"])
def test_long_input_is_escaped_in_linear_time(unit: str) -> None:
    started = time.perf_counter()
    escape_markup(unit * 200_000)
    assert time.perf_counter() - started < 1.0
