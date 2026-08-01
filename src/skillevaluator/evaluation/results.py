# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public result and error types for dataset generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


class DatasetGenerationError(RuntimeError):
    """Expected dataset-generation failure safe for API callers to handle."""


@dataclass(frozen=True)
class DatasetGenerationResult:
    """Structured outcome shared by CLI and programmatic dataset generation."""

    status: Literal["created", "preview", "unchanged"]
    path: Path
    dataset: dict[str, Any] | None = None
    cases_count: int = 0


__all__ = ["DatasetGenerationError", "DatasetGenerationResult"]
