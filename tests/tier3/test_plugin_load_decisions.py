# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private per-agent plugin load decisions, including Hermes compatibility."""

from __future__ import annotations

import pytest

from skillevaluator.tier3.plugin_native import PLUGIN_LOAD_CHOICES, PluginLoadError, resolve_plugin_load

AGENTS = ["claude-code", "codex", "opencode", "hermes"]


def test_wrapper_is_the_default_behavior_for_every_agent() -> None:
    assert PLUGIN_LOAD_CHOICES == ("wrapper", "native", "auto")
    assert {d.mode for d in resolve_plugin_load("wrapper", AGENTS, env_mode="docker").values()} == {"wrapper"}


def test_auto_loads_native_clients_natively_and_keeps_hermes_on_the_wrapper() -> None:
    decisions = resolve_plugin_load("auto", AGENTS, env_mode="docker")
    assert {a: d.mode for a, d in decisions.items()} == {
        "claude-code": "native",
        "codex": "native",
        "opencode": "native",
        "hermes": "wrapper",
    }
    assert "hermes" in decisions["hermes"].reason


def test_native_hermes_keeps_the_wrapper_task_under_its_own_home() -> None:
    decision = resolve_plugin_load("native", ["hermes"], env_mode="docker")["hermes"]
    assert decision.native and "wrapper task" in decision.reason


def test_unknown_load_choice_is_refused() -> None:
    with pytest.raises((PluginLoadError, ValueError)):
        resolve_plugin_load("magic", ["claude-code"], env_mode="docker")
