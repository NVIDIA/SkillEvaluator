# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect the advisory harness-compatibility rows and write them to the JSON report."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skillevaluator.validators.harness import HARNESS_METADATA_KEY

if TYPE_CHECKING:
    from skillevaluator.models.result import ValidationResult

Block = dict[str, list[dict[str, Any]]]


def harness_block(results: Iterable[ValidationResult]) -> Block:
    """Collect the per-skill harness rows recorded by the ``harness`` check."""
    block: Block = {}
    for result in results:
        block.update((result.metadata or {}).get(HARNESS_METADATA_KEY) or {})
    return block


def write_json_block(json_report: Path, block: Block) -> None:
    """Add the ``harness_compatibility`` block to an existing JSON report."""
    if json_report.exists():
        data = json.loads(json_report.read_text(encoding="utf-8"))
        data[HARNESS_METADATA_KEY] = block
        json_report.write_text(json.dumps(data, indent=2), encoding="utf-8")
