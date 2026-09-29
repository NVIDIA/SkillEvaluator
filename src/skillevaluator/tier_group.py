# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A tier's default path workflow alongside its explicit expert commands."""

from __future__ import annotations

import copy
from pathlib import Path

import click

from skillevaluator.cli_help import RichGroup, render_help


class TierGroup(RichGroup):
    """Dispatch paths to a workflow without inventing a public subcommand."""

    workflow: click.Command | None = None

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if not args and not ctx.resilient_parsing:
            click.echo(ctx.get_help())
            ctx.exit()
        if args and self.workflow is not None:
            first = args[0]
            if first not in ctx.help_option_names and first not in self.commands:
                is_path = first.startswith((".", "~")) or "/" in first or "\\" in first or Path(first).exists()
                if first.startswith("-") or is_path:
                    ctx.meta["direct_tier_args"] = args
                    return args
        return super().parse_args(ctx, args)

    def invoke(self, ctx: click.Context):
        args = ctx.meta.pop("direct_tier_args", None)
        if args is not None and self.workflow is not None:
            with self.workflow.make_context(ctx.info_name, args, parent=ctx.parent) as workflow_ctx:
                return self.workflow.invoke(workflow_ctx)
        return super().invoke(ctx)

    def collect_usage_pieces(self, ctx: click.Context) -> list[str]:  # noqa: ARG002 - Click method interface
        return ["[OPTIONS] PATH", "| COMMAND [ARGS]..."]

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        # Show workflow options and expert commands together without making
        # workflow flags apply to every expert subcommand's parser.
        display = copy.copy(self)
        if self.workflow is not None:
            display.params = list(self.workflow.params)
        formatter.write(render_help(display, ctx, width=formatter.width))
