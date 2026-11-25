# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep ``docs/cli-reference.mdx`` in step with the live Click command tree.

The page claims to list every command and flag. These tests walk the real tree,
so a new visible command or long option fails until the page names it. Hidden
commands and hidden options stay undocumented on purpose and are skipped.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import click

from skillevaluator.cli import cli

REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "cli-reference.mdx"
TIER_GROUPS = {"tier1", "tier2", "tier3"}


def _runnable_commands(group: click.Group, prefix: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], click.Command]]:
    """Yield every visible command that does work, with its full path.

    A pure group such as ``ci`` is skipped. A tier group also runs a direct
    ``tierN PATH`` workflow, so that workflow's options count under the tier's name.
    """
    context = click.Context(group)
    workflow = getattr(group, "workflow", None)
    if prefix and workflow is not None and not workflow.hidden:
        yield prefix, workflow
    for name in group.list_commands(context):
        command = group.get_command(context, name)
        if command is None or command.hidden:
            continue
        if isinstance(command, click.Group):
            yield from _runnable_commands(command, (*prefix, name))
        else:
            yield (*prefix, name), command


def _long_options(command: click.Command) -> list[str]:
    """Return every visible long option name, including secondary names such as ``--no-llm``."""
    names: list[str] = []
    for parameter in command.params:
        if not isinstance(parameter, click.Option) or parameter.hidden:
            continue
        names.extend(name for name in (*parameter.opts, *parameter.secondary_opts) if name.startswith("--"))
    return names


def _mentioned(text: str, token: str) -> bool:
    """Match ``token`` as a whole word, so ``--tiers`` is not satisfied by ``--tiers-extra``."""
    return re.search(rf"(?<![\w-]){re.escape(token)}(?![\w-])", text) is not None


def missing_commands(root: click.Group, page: str) -> list[str]:
    """Return visible commands that the page never names."""
    missing = []
    for path, _command in _runnable_commands(root):
        full = " ".join(path)
        leaf = path[-1]
        names = [full] if path[0] not in TIER_GROUPS or len(path) == 1 else [full, leaf]
        if not any(_mentioned(page, name) for name in names):
            missing.append(full)
    return missing


def missing_options(root: click.Group, page: str) -> list[str]:
    """Return ``command --flag`` strings for visible long options the page never names."""
    missing = []
    for path, command in _runnable_commands(root):
        for name in _long_options(command):
            if not _mentioned(page, name):
                missing.append(f"{' '.join(path)} {name}")
    return missing


def test_every_visible_command_is_named_in_the_cli_reference() -> None:
    missing = missing_commands(cli, REFERENCE.read_text(encoding="utf-8"))
    assert not missing, (
        "These commands are visible in `skillevaluator --help` but missing from docs/cli-reference.mdx. "
        f"Add them to the command map and give each a section: {missing}"
    )


def test_every_visible_long_option_is_documented_in_the_cli_reference() -> None:
    missing = missing_options(cli, REFERENCE.read_text(encoding="utf-8"))
    assert not missing, (
        "These visible options are missing from docs/cli-reference.mdx. Edit that page to add a row "
        "for each, or mark the option hidden if it is a compatibility alias: " + ", ".join(missing)
    )


def test_guard_ignores_hidden_commands_and_options() -> None:
    @click.group()
    def root() -> None:
        pass

    @root.command("shown")
    @click.option("--known", is_flag=True)
    @click.option("--flip/--no-flip", default=False)
    @click.option("--legacy", is_flag=True, hidden=True)
    @click.option("-s", "--short-and-long")
    def shown(known: bool, flip: bool, legacy: bool, short_and_long: str) -> None:
        pass

    @root.command("secret", hidden=True)
    @click.option("--never-documented", is_flag=True)
    def secret(never_documented: bool) -> None:
        pass

    @root.group("team")
    def team() -> None:
        pass

    @team.command("sync")
    @click.option("--force", is_flag=True)
    def sync(force: bool) -> None:
        pass

    page = "`shown` `team sync` --known --flip --no-flip --short-and-long --force"
    assert missing_commands(root, page) == []
    assert missing_options(root, page) == []
    assert missing_options(root, "`shown` `team sync` --known --flip --short-and-long --force") == ["shown --no-flip"]
    assert missing_commands(root, "`shown` --known --flip --no-flip --short-and-long --force") == ["team sync"]
    # A flag that only appears as a prefix of another does not count.
    assert missing_options(root, "`shown` `team sync` --known --flip --no-flip --short-and-long --force-all") == [
        "team sync --force"
    ]
