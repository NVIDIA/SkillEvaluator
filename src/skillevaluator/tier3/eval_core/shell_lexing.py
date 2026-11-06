# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shell lexical rules shared by host checks and the staged verifier."""

from __future__ import annotations

import re
import shlex
from typing import Any

_ATTACHED_DESCRIPTOR_RE = re.compile(r"(\d+)(>>|>\||>&|<&|<>|>|<)(?![<])")


# Words the walk reads as shell syntax when they stand unquoted. Quoted or
# escaped, each is an ordinary word: `'done'` is a command named done, and
# `printf "("` prints a parenthesis. The tokenizer drops quotes, so such a
# word is prefixed with a private-use character before tokenizing and the
# prefix is removed wherever a word is read as a value.
_QUOTED_SYNTAX_MARK = "\ue000"
_SYNTAX_WORDS = frozenset(
    {
        "for",
        "select",
        "while",
        "until",
        "if",
        "then",
        "else",
        "elif",
        "fi",
        "do",
        "done",
        "case",
        "esac",
        "in",
        "!",
        "{",
        "}",
        "(",
        ")",
        "--",
        ";",
        ";;",
        "&",
        "&&",
        "|",
        "||",
        "<",
        ">",
        ">>",
        ">|",
        "<<",
        "<<<",
        "<&",
        ">&",
        "&>",
        "&>>",
        "<>",
    }
)
_SHELL_METACHARS = ";&|()<>"
# A quoted word made of metacharacters (``';|'``, ``'2>'``, ``'<<-'``) would be
# split into operators after tokenizing; it is marked like a listed word.
_SYNTAX_SHAPE_RE = re.compile(r"\d*[;&|<>(){}]+-?|!|--")


def _mark_quoted_syntax(text: str) -> str:
    """Prefix each quoted or escaped word that would otherwise read as syntax.

    Words are delimited as the shell delimits them: at unquoted whitespace
    and unquoted metacharacters. A word that contains a quote or an escape
    and whose unquoted text is in ``_SYNTAX_WORDS`` or shaped like syntax
    gets the mark; every other character is copied through unchanged,
    quotes included, so the tokenizer still sees the original quoting. A
    mark character already present in the text is doubled first, so the
    reader can tell a mark (one, at the start of a word) from data (always
    two), and a file whose name carries that character keeps its identity.
    """
    text = text.replace(_QUOTED_SYNTAX_MARK, _QUOTED_SYNTAX_MARK * 2)
    out = []
    raw = []
    plain = []
    quoted = False
    quote = None
    index = 0

    def flush() -> None:
        nonlocal quoted
        if raw:
            if quoted and ("".join(plain) in _SYNTAX_WORDS or _SYNTAX_SHAPE_RE.fullmatch("".join(plain))):
                out.append(_QUOTED_SYNTAX_MARK)
            out.extend(raw)
            raw.clear()
            plain.clear()
        quoted = False

    while index < len(text):
        char = text[index]
        if quote:
            raw.append(char)
            if char == quote:
                quote = None
            elif char == "\\" and quote == '"' and index + 1 < len(text) and text[index + 1] in '"\\':
                raw.append(text[index + 1])
                plain.append(text[index + 1])
                index += 1
            else:
                plain.append(char)
            index += 1
            continue
        if char in "'\"":
            quote = char
            quoted = True
            raw.append(char)
            index += 1
            continue
        if char == "\\" and index + 1 < len(text):
            quoted = True
            raw.append(char)
            raw.append(text[index + 1])
            plain.append(text[index + 1])
            index += 2
            continue
        if char.isspace() or char in _SHELL_METACHARS:
            flush()
            out.append(char)
            index += 1
            continue
        raw.append(char)
        plain.append(char)
        index += 1
    flush()
    return "".join(out)


def _keep_attached_descriptors(text: str) -> str:
    """Quote ``2>`` so the lexer keeps the descriptor with its operator.

    The lexer splits every ``>`` and ``<`` into its own token, so ``2>/dev/null``
    and ``2 > numeric.out`` arrive as the same tokens. In the shell they are
    not the same command: the first sends standard error away, the second
    passes ``2`` as an argument and redirects standard output. Only a digit
    run written flush against its operator is a descriptor, so that pairing
    is fixed here, before the text is tokenized, by quoting it. Text inside
    quotes is data and is left alone.
    """
    out = []
    quote = None
    index = 0
    at_word_start = True
    while index < len(text):
        char = text[index]
        if quote:
            out.append(char)
            if char == "\\" and quote == '"' and index + 1 < len(text):
                out.append(text[index + 1])
                index += 1
            elif char == quote:
                quote = None
            index += 1
            continue
        if char == "\\" and index + 1 < len(text):
            out.append(text[index : index + 2])
            index += 2
            at_word_start = False
            continue
        if char in "'\"":
            quote = char
            out.append(char)
            index += 1
            at_word_start = False
            continue
        if at_word_start and char.isdigit():
            match = _ATTACHED_DESCRIPTOR_RE.match(text, index)
            if match:
                out.append("'" + match.group(1) + match.group(2) + "'")
                index = match.end()
                # An operand written flush against the operator
                # (``0<run.py``) would concatenate onto the quoted
                # operator as one word, hiding the filename from the
                # walk. A space splits it into its own token, as the
                # spaced form already tokenizes.
                if index < len(text) and not text[index].isspace() and text[index] not in ";&|(){}<>":
                    out.append(" ")
                at_word_start = False
                continue
        out.append(char)
        at_word_start = char.isspace() or char in ";&|(){}"
        index += 1
    return "".join(out)


def _shell_tokens(cmd: Any) -> list[str]:
    normalized = str(cmd).replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ; ")
    normalized = _keep_attached_descriptors(_mark_quoted_syntax(normalized))
    lexer = shlex.shlex(normalized, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        try:
            return shlex.split(str(cmd), posix=True)
        except ValueError:
            return []
