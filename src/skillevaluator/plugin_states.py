# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Passive dependency-resolution states shared by the private resolver and its consumers."""

from __future__ import annotations

# How a declared bundle-reference dependency resolved (see skillevaluator.plugin_dependencies).
DEPENDENCY_STATES: tuple[str, ...] = ("provided", "referenced", "missing", "external", "unresolved")
