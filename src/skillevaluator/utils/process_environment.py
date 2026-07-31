# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment handling for child processes."""

from __future__ import annotations

import os
from collections.abc import Mapping


def child_process_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a child environment without inherited observability exporters."""
    child_env = dict(os.environ if env is None else env)
    for key in list(child_env):
        if key.startswith(("OTEL_", "DD_", "SKILLEVALUATOR_TELEMETRY_")):
            child_env.pop(key, None)
    return child_env
