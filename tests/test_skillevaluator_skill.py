# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the repository's SkillEvaluator operator skill in sync with the CLI."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import click
import pytest

from skillevaluator.cli import cli

SKILL_DIR = Path(__file__).resolve().parents[1] / "skills" / "skillevaluator"
SKILL_FILES = sorted(SKILL_DIR.rglob("*.md"))
INLINE_COMMAND = re.compile(r"`(skillevaluator [^`]+)`")
FENCED_BLOCK = re.compile(r"```(?:bash|sh)\n(.*?)```", re.DOTALL)


def _commands(text: str) -> list[str]:
    commands = INLINE_COMMAND.findall(text)
    for block in FENCED_BLOCK.findall(text):
        joined = block.replace("\\\n", " ")
        for line in joined.splitlines():
            line = line.split("#", 1)[0].strip()
            if "skillevaluator " in line:
                commands.append(line[line.index("skillevaluator ") :])
    return commands


def _resolve(command: str) -> tuple[click.Command, list[str]]:
    tokens = shlex.split(command.split("|", 1)[0])[1:]
    node: click.Command = cli
    while tokens and isinstance(node, click.Group):
        token = tokens[0]
        if token.startswith("-"):
            tokens.pop(0)
            continue
        sub = node.get_command(click.Context(node), token)
        if sub is None:
            node = getattr(node, "workflow", None) or node
            break
        node = sub
        tokens.pop(0)
    return node, tokens


def _all_commands() -> list[tuple[str, str]]:
    return [
        (path.relative_to(SKILL_DIR).as_posix(), cmd)
        for path in SKILL_FILES
        for cmd in _commands(path.read_text())
        if "<command>" not in cmd and cmd.split()[1:] != ["--version"]
    ]


def test_skill_files_exist() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()
    assert _all_commands(), "expected documented skillevaluator commands"


@pytest.mark.parametrize(("source", "command"), _all_commands())
def test_documented_commands_and_options_exist(source: str, command: str) -> None:
    node, rest = _resolve(command)
    assert node is not cli, f"{source}: unknown command in {command!r}"
    known = {"--help", "-h"} | {opt for param in node.params for opt in [*param.opts, *param.secondary_opts]}
    for token in rest:
        if token.startswith("--") or re.fullmatch(r"-[a-zA-Z]", token):
            flag = token.split("=", 1)[0]
            assert flag in known, f"{source}: {flag} is not an option of {node.name!r} in {command!r}"


def test_reference_links_resolve() -> None:
    for path in SKILL_FILES:
        for target in re.findall(r"\]\(([^)#]+\.md)(?:#[^)]*)?\)", path.read_text()):
            assert (path.parent / target).is_file(), f"{path.name}: broken link {target}"
