# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared field validators for Pydantic models.

Provides reusable validators for common field patterns across
Skills, Rules, and Workflows models, eliminating duplication.
"""

from typing import Any


def ensure_string_list(v: Any) -> list[str] | None:
    """Convert various input types to a list of strings.

    Handles the common pattern of accepting either a single string
    or a list for fields like tags, globs, languages, etc.

    Args:
        v: Input value (None, str, or list)

    Returns:
        List of strings, or None if input was None
    """
    if v is None:
        return None
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [str(item) for item in v]
    return v


def parse_nested_model(model_class: type, v: Any, error_message: str | None = None) -> Any:
    """Parse a dict into a nested Pydantic model.

    Common pattern for parsing metadata dicts into their respective models.

    Args:
        model_class: The Pydantic model class to instantiate
        v: Input value (None, dict, or already-instantiated model)
        error_message: Optional error message if None is not allowed

    Returns:
        Instance of model_class, or None/raises based on input

    Raises:
        ValueError: If v is None and error_message is provided
    """
    if v is None:
        if error_message:
            raise ValueError(error_message)
        return None
    if isinstance(v, dict):
        return model_class(**v)
    return v
