# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI feedback while dataset preparation runs before the evaluation reporter."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TextIO

HEARTBEAT_INTERVAL_SECONDS = 10.0


@contextmanager
def dataset_generation_progress(*, enabled: bool, stream: TextIO | None = None) -> Iterator[None]:
    """Report elapsed waiting time; never imply provider-side progress or a deadline."""
    if not enabled:
        yield
        return

    output = stream if stream is not None else sys.stderr
    started = time.monotonic()
    stopped = threading.Event()

    def announce(message: str) -> None:
        try:
            output.write(message + "\n")
            output.flush()
        except (OSError, ValueError):
            # A closed progress stream must not fail dataset generation.
            stopped.set()

    def heartbeat() -> None:
        while not stopped.wait(HEARTBEAT_INTERVAL_SECONDS):
            elapsed = int(time.monotonic() - started)
            announce(
                f"Autopilot: waiting for dataset generation ({elapsed}s elapsed); live evaluation has not started."
            )

    worker = threading.Thread(target=heartbeat, name="dataset-generation-progress", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join(timeout=1.0)

    announce(f"Autopilot: evaluation source ready ({time.monotonic() - started:.1f}s); continuing to evaluation.")
