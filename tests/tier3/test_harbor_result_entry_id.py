# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from skillevaluator.tier3.harbor.collector import _entry_id_from_harbor_result


def test_entry_id_prefers_task_id_path_over_prefixed_task_name() -> None:
    result = {
        "task_name": "nvidia/skillevaluator-apt-pinning",
        "task_id": {"path": "evals/cases/case-014-apt-pinning"},
    }
    assert _entry_id_from_harbor_result(result) == "case-014-apt-pinning"
