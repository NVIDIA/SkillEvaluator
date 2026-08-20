# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SkillEvaluator package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("skillevaluator")
except PackageNotFoundError:
    __version__ = "0.2.0"

__all__ = ["__version__"]
