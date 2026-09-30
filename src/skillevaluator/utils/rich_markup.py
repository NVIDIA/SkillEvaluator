# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make untrusted text safe to print through Rich."""

from __future__ import annotations

import re

# OSC strings (hyperlinks, window titles) go first so their payload goes with them.
_OSC_ESCAPE_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
# CSI sequences, then every other ECMA-48 escape (such as "ESC c", a full terminal reset).
_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[ -/]*[0-~])")
# C0 controls except tab, line feed and carriage return, then DEL and C1 controls.
_TERMINAL_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Every "[" with the whole backslash run before it (the lookbehind and possessive
# quantifier keep long runs linear). The optional group is set when the bracket
# opens something Rich parses as a tag (same shape as ``rich.markup.RE_TAGS``).
_OPEN_BRACKET = re.compile(r"(?<!\\)(\\*+)\[(?=([a-z#/@][^\[\]]*\])?)")
# Rich halves a backslash run only when a tag follows it, so an escaped trailing
# run is always followed by this empty tag pair.
_EMPTY_TAG = "[bold][/bold]"


def strip_terminal_controls(text: str) -> str:
    """Remove terminal escape sequences and control characters other than tab, LF and CR.

    Rich only strips BEL, BS, VT, FF and CR, so without this untrusted text can
    move the cursor, erase lines, reset the terminal or emit OSC 8 hyperlinks.
    """
    text = _OSC_ESCAPE_RE.sub("", text)
    text = _ANSI_ESCAPE_RE.sub("", text)
    return _TERMINAL_CONTROL_RE.sub("", text)


def _escape_bracket(match: re.Match[str]) -> str:
    backslashes, tag = match.group(1), match.group(2)
    if tag is not None:
        # Tag-shaped: literal backslashes are doubled, then the tag is escaped.
        return f"{backslashes}{backslashes}\\["
    # Rich renders a plain-text "\[" as "[", so one extra backslash keeps the rest.
    return f"{backslashes}\\["


def escape_markup(text: str) -> str:
    """Return ``text`` made safe to interpolate into Rich markup.

    Terminal escape sequences and control characters other than tab and newline
    are removed first (see ``strip_terminal_controls``). Every bracket is then
    escaped: ``rich.markup.escape`` only escapes tag-shaped brackets, but Rich
    also turns every plain-text ``\\[`` into ``[``, so Windows paths such as
    ``C:\\skills\\[x]`` would lose a separator. A trailing backslash run is
    doubled and closed with an empty tag, so it renders literally whatever
    follows it in the markup (a tag, plain text, or nothing).

    Rich also replaces ``:name:`` emoji codes, which markup cannot escape.
    Consoles that print untrusted text must be created with ``emoji=False``, and
    Panel titles and Status text, which Rich parses with emoji enabled whatever
    the console says, must be passed as ``Text.from_markup(..., emoji=False)``.
    """
    text = strip_terminal_controls(text).replace("\r", "")
    escaped = _OPEN_BRACKET.sub(_escape_bracket, text)
    body = escaped.rstrip("\\")
    trailing = len(escaped) - len(body)
    return body + "\\" * (2 * trailing) + _EMPTY_TAG if trailing else escaped
