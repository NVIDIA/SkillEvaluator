# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle for transient OSS Harbor execution artifacts."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

RetentionReason = Literal["explicit_keep", "cleanup_failed", "not_retained"]


@dataclass(frozen=True)
class RetentionOutcome:
    """Filesystem-backed outcome of a Harbor artifact retention decision."""

    retained: bool
    reason: RetentionReason
    warning: str = ""


class HarborArtifactLifecycle:
    """Finalize Harbor jobs/tasks exactly once on every terminal path."""

    def __init__(self, paths: list[Path], *, keep_requested: bool) -> None:
        self.paths = paths
        self.keep_requested = keep_requested
        self.outcome: RetentionOutcome | None = None

    def __enter__(self) -> HarborArtifactLifecycle:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.finalize()

    def finalize(self) -> RetentionOutcome:
        """Apply the requested policy and return actual filesystem truth."""
        if self.outcome is not None:
            return self.outcome

        if self.keep_requested:
            retained = any(path.exists() for path in self.paths)
            self.outcome = RetentionOutcome(
                retained=retained,
                reason="explicit_keep" if retained else "not_retained",
            )
            return self.outcome

        failures: list[str] = []
        for path in self.paths:
            if not path.exists():
                continue
            try:
                shutil.rmtree(path)
            except OSError as exc:
                failures.append(f"{path}: {exc}")

        retained = any(path.exists() for path in self.paths)
        self.outcome = RetentionOutcome(
            retained=retained,
            reason="cleanup_failed" if retained else "not_retained",
            warning="; ".join(failures),
        )
        return self.outcome
