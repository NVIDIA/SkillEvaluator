# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free TOML string serialization helpers."""

from __future__ import annotations

import json


def toml_quote(value: str) -> str:
    """Serialize *value* as a TOML basic string."""
    if not isinstance(value, str):
        raise TypeError("TOML string value must be a string")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError("TOML strings must not contain surrogate code points")
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007F")
