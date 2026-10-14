# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared evidence-reference identity."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def evidence_ref_identity(ref: Mapping[str, Any]) -> str:
    """Return the most specific stable location carried by an evidence ref."""
    for key in ("evidence_id", "json_pointer", "path"):
        if value := str(ref.get(key) or "").strip():
            return value
    return ""
