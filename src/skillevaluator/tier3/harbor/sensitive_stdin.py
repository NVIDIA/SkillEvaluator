# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-local reader for credentials delivered through standard input."""

from __future__ import annotations

import os
import sys
from threading import Lock

NVIDIA_BUILD_STDIN_SENTINEL = "skillevaluator-stdin-backed-nvidia-key"
NVIDIA_BUILD_KEY_STDIN_ENV = "SKILLEVALUATOR_NVIDIA_API_KEY_STDIN"

_UNSET = object()


class _SecretCache:
    value: str | object = _UNSET


_nvidia_build_key_cache = _SecretCache()
_nvidia_build_key_lock = Lock()


def read_nvidia_build_key_from_stdin() -> str:
    """Read and cache the parent-provided key without storing it on disk."""
    if os.environ.get(NVIDIA_BUILD_KEY_STDIN_ENV) != "1":
        raise RuntimeError(f"{NVIDIA_BUILD_KEY_STDIN_ENV} is required for NVIDIA Build Docker runs")

    with _nvidia_build_key_lock:
        if _nvidia_build_key_cache.value is _UNSET:
            try:
                api_key = sys.stdin.read().strip()
            except (OSError, UnicodeError) as exc:
                raise RuntimeError("NVIDIA Build key stdin handoff is unavailable") from exc
            if not api_key:
                raise RuntimeError("NVIDIA Build key stdin handoff is empty")
            _nvidia_build_key_cache.value = api_key

        assert isinstance(_nvidia_build_key_cache.value, str)
        return _nvidia_build_key_cache.value
