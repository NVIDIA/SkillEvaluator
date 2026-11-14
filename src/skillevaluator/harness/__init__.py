# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static harness-compatibility overlay for Tier 1.

Each packaged profile describes how one agent harness discovers, parses and
shows skills, as verified against a pinned harness build. Rules map skill facts
to an effect (``not_loaded``, ``degraded`` or ``advisory``). The overlay is
advisory: it never changes Tier 1 findings or the exit code.
"""

from skillevaluator.harness.overlay import (
    HARNESS_STATUSES,
    Catalog,
    HarnessResult,
    evaluate_harnesses,
    evaluate_profile,
    facts_for,
    listed_description,
    load_profiles,
    model_config,
    skill_facts,
)
from skillevaluator.harness.plugin import evaluate_plugin, is_plugin_root, plugin_facts

__all__ = [
    "HARNESS_STATUSES",
    "Catalog",
    "HarnessResult",
    "evaluate_harnesses",
    "evaluate_plugin",
    "evaluate_profile",
    "facts_for",
    "is_plugin_root",
    "listed_description",
    "load_profiles",
    "model_config",
    "plugin_facts",
    "skill_facts",
]
