# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Staged file metadata that changes execution must invalidate baseline reuse."""

import os
import subprocess
from pathlib import Path

import pytest

from skillevaluator.tier3.baseline_store import baseline_task_tree_digest, comparison_mismatches
from skillevaluator.tier3.harbor.secure_copy import copy_file_secure


@pytest.mark.skipif(os.name != "posix", reason="POSIX executable permission bits")
def test_staged_executable_permission_change_rejects_the_old_baseline(tmp_path: Path) -> None:
    source = tmp_path / "authored"
    source.mkdir()
    script = source / "check.sh"
    script.write_text("#!/bin/sh\nprintf ready", encoding="utf-8")
    script.chmod(0o755)
    task = tmp_path / "case-001"
    staged = task / "environment" / "repo" / script.name
    staged.parent.mkdir(parents=True)
    copy_file_secure(script, staged, allowed_root=source)
    before = baseline_task_tree_digest({"codex": [task]})
    assert subprocess.run([str(staged)], capture_output=True, text=True, check=True, timeout=5).stdout == "ready"

    script.chmod(0o644)
    copy_file_secure(script, staged, allowed_root=source)
    assert staged.read_bytes() == script.read_bytes()
    with pytest.raises(PermissionError):
        subprocess.run([str(staged)], capture_output=True, text=True, check=True, timeout=5)
    after = baseline_task_tree_digest({"codex": [task]})
    assert comparison_mismatches(
        {"config": {}, "baseline_task_tree_digest": after},
        {"config": {}, "baseline_task_tree_digest": before},
    ) == ["staged baseline task tree changed"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes and symlinks")
def test_regular_file_cannot_impersonate_a_symlinks_digest_payload(tmp_path: Path) -> None:
    task = tmp_path / "case-001"
    task.mkdir()
    entry = task / "entry"
    entry.write_bytes(b"symlink\0target")
    entry.chmod(0o777)
    regular = baseline_task_tree_digest({"codex": [task]})
    entry.unlink()
    entry.symlink_to("target")
    assert baseline_task_tree_digest({"codex": [task]}) != regular


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_file_permissions_matter_but_timestamps_do_not(tmp_path: Path) -> None:
    task = tmp_path / "case-001"
    task.mkdir()
    entry = task / "settings.json"
    entry.write_text("{}", encoding="utf-8")
    entry.chmod(0o644)
    original = baseline_task_tree_digest({"codex": [task]})
    os.utime(entry, (1_000_000_000, 1_000_000_000))
    assert baseline_task_tree_digest({"codex": [task]}) == original
    entry.chmod(0o444)
    assert baseline_task_tree_digest({"codex": [task]}) != original
