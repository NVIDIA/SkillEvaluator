# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic introspection of the Click command tree.

Used to freeze the CLI surface (commands, options, choices, defaults) as a
golden baseline so the Phase 1 rename and later refactors can prove they only
change *names* and not the command structure.
"""

from __future__ import annotations

from typing import Any

import click


def _param_to_dict(param: click.Parameter) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": param.name,
        "param_type": param.param_type_name,  # "option" or "argument"
    }
    opts = list(getattr(param, "opts", []) or [])
    secondary = list(getattr(param, "secondary_opts", []) or [])
    if opts:
        data["opts"] = sorted(opts)
    if secondary:
        data["secondary_opts"] = sorted(secondary)
    if getattr(param, "multiple", False):
        data["multiple"] = True
    if getattr(param, "is_flag", False):
        data["is_flag"] = True
    if getattr(param, "required", False):
        data["required"] = True
    default = getattr(param, "default", None)
    if default is not None and default is not getattr(click.core, "UNSET", None):
        data["default"] = str(default)
    param_type = getattr(param, "type", None)
    if isinstance(param_type, click.Choice):
        data["choices"] = list(param_type.choices)
    elif param_type is not None:
        data["type"] = param_type.name
    return data


def _command_to_dict(command: click.Command) -> dict[str, Any]:
    data: dict[str, Any] = {
        "params": [_param_to_dict(p) for p in command.params],
    }
    if isinstance(command, click.Group):
        data["commands"] = {name: _command_to_dict(command.commands[name]) for name in sorted(command.commands)}
    return data


def build_cli_surface(group: click.Group) -> dict[str, Any]:
    """Return a JSON-serializable snapshot of the CLI command tree."""
    return _command_to_dict(group)
