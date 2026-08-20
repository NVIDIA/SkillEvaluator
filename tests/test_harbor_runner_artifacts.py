# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import skillevaluator.tier3.harbor.runner as harbor_runner
from skillevaluator import __version__


def test_runner_persists_deduplicated_dataset_truth(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for task, entry in (
        ("first", {"id": "same-task", "expected_skill": "demo"}),
        ("duplicate", {"id": "same-task", "expected_skill": "demo"}),
        ("negative", {"id": "negative", "expected_skill": None}),
    ):
        entry_path = run_dir / "_harbor-tasks" / task / "tests" / "entry.json"
        entry_path.parent.mkdir(parents=True)
        entry_path.write_text(json.dumps(entry), encoding="utf-8")

    snapshot = harbor_runner._persist_dataset_truth(run_dir, fallback_task_ids=[])

    assert snapshot["evaluator_version"] == __version__
    assert snapshot["dataset_summary"] == {
        "total_tasks": 2,
        "positive_tasks": 1,
        "negative_tasks": 1,
        "unclassified_tasks": 0,
        "source": "dataset",
    }
    assert snapshot["dataset_digest"].startswith("sha256:")
    assert json.loads((run_dir / "dataset_snapshot.json").read_text(encoding="utf-8")) == snapshot
