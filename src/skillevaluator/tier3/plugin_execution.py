# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private plugin evaluation lifecycle over the existing evaluation engine.

The runner consults :func:`active_runtime` at its with-arm seams only. It is
``None`` unless :func:`private_runtime` is active, so ordinary skill runs are
unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Protocol


class PrivateRuntimeHooks(Protocol):
    """The with-arm extensions the runner calls while a private plugin run is active."""

    def with_agent_import_path(self, agent: str, base_import_path: str | None) -> str | None:
        """The Harbor agent for ``agent``'s with arm, or ``None`` to keep the arm's ordinary agent."""

    def stage_with_arm(self, agent: str, task_dirs: Sequence[Path]) -> None:
        """Stage per-agent runtime additions into the with arm's staged tasks."""


_ACTIVE_RUNTIME: ContextVar[PrivateRuntimeHooks | None] = ContextVar("skilleval_private_runtime", default=None)


def active_runtime() -> PrivateRuntimeHooks | None:
    """The private runtime hooks of the current run, or ``None`` for an ordinary run."""
    return _ACTIVE_RUNTIME.get()


@contextmanager
def private_runtime(hooks: PrivateRuntimeHooks) -> Iterator[PrivateRuntimeHooks]:
    """Activate ``hooks`` for the evaluation run inside the block."""
    token = _ACTIVE_RUNTIME.set(hooks)
    try:
        yield hooks
    finally:
        _ACTIVE_RUNTIME.reset(token)
