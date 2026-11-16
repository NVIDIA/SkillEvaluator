# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognize an import failure caused by a missing ``tier3`` extra."""

from __future__ import annotations

# Top-level import names of the packages that only the ``tier3`` extra installs
# (``[project.optional-dependencies]`` in pyproject.toml).
TIER3_EXTRA_MODULES = frozenset(
    {"anthropic", "boto3", "dotenv", "harbor", "jwt", "litellm", "mcp", "oauthlib", "openai"}
)


def is_missing_tier3_extra(exc: ImportError) -> bool:
    """Return whether ``exc`` names a package that the ``tier3`` extra provides."""
    return (exc.name or "").partition(".")[0] in TIER3_EXTRA_MODULES
