# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""History directory discovery should not repeatedly scan every retained run."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from skillevaluator.tier3.history import history_directories


def test_mixed_completed_and_reserved_runs_have_bounded_membership_work(tmp_path, monkeypatch):
    count = 256
    paths = []
    for index in range(count):
        timestamp = datetime(2026, 1, 1) + timedelta(seconds=index)  # noqa: DTZ001 -- run names have no timezone
        path = tmp_path / timestamp.strftime("%Y%m%d_%H%M%S")
        path.mkdir()
        paths.append(path)
        if index % 2 == 0:
            (path / "run_config.json").write_text("{}", encoding="utf-8")
            (path / "result.json").write_text(json.dumps({"run_id": path.name, "agents": {}}), encoding="utf-8")

    # Count equality work instead of measuring filesystem-dependent wall time.
    path_type = type(tmp_path)
    original_eq = path_type.__eq__
    comparisons = 0

    def counted_eq(left, right):
        nonlocal comparisons
        comparisons += 1
        return original_eq(left, right)

    with monkeypatch.context() as scoped:
        scoped.setattr(path_type, "__eq__", counted_eq)
        observed = history_directories(tmp_path)

    assert observed == paths
    assert comparisons < 16 * count, f"{count} runs required {comparisons} path equality checks"
