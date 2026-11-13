# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep JSON readable when the output encoding can represent its values."""

from __future__ import annotations

import json


def json_needs_ascii(data: object, encoding: str | None) -> bool:
    """Escape JSON only when strict encoding would otherwise lose characters."""
    try:
        json.dumps(data, ensure_ascii=False).encode(encoding or "utf-8", errors="strict")
    except UnicodeEncodeError:
        return True
    return False
