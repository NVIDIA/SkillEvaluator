# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in Harbor environment names available in the public Tier 3 surface.

This deliberately lives outside the Tier 3 package so that the base CLI can
show its command help without importing Harbor or any optional dependencies.
"""

from __future__ import annotations

HARBOR_ENVIRONMENTS = (
    "docker",
    "daytona",
    "e2b",
    "modal",
    "runloop",
    "langsmith",
    "gke",
    "novita",
    "apple-container",
    "singularity",
    "islo",
    "tensorlake",
    "cwsandbox",
    "wandb",
    "use-computer",
    # Not a Harbor-native backend: SkillEvaluator's host execution mode, run
    # under an OS sandbox (bubblewrap on Linux, Seatbelt on macOS). Dispatched
    # via --environment-import-path, not Harbor's --env.
    "local",
)
HARBOR_ENV_MODES = frozenset(HARBOR_ENVIRONMENTS)
#: env modes that Harbor accepts natively via ``--env`` (everything except ``local``).
HARBOR_NATIVE_ENV_MODES = frozenset(m for m in HARBOR_ENVIRONMENTS if m != "local")
ENV_MODE_LOCAL = "local"
DEFAULT_ENV_MODE = "docker"
