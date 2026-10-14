# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free TOML string serialization helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping


def toml_quote(value: str) -> str:
    """Serialize *value* as a TOML basic string."""
    if not isinstance(value, str):
        raise TypeError("TOML string value must be a string")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError("TOML strings must not contain surrogate code points")
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007F")


def extract_toml_metadata_entry_id(parsed: object) -> str | None:
    """Extract a non-empty, non-boolean [metadata].entry_id string from parsed TOML data."""
    if not isinstance(parsed, Mapping):
        return None
    metadata = parsed.get("metadata")
    if not isinstance(metadata, Mapping) or "entry_id" not in metadata:
        return None
    raw_entry_id = metadata["entry_id"]
    if raw_entry_id is None or isinstance(raw_entry_id, bool):
        return None
    stripped = str(raw_entry_id).strip()
    return stripped or None
