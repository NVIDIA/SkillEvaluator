# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Large staged files must preserve baseline identity without full-file copies."""

import tracemalloc
from pathlib import Path

import pytest

from skillevaluator.tier3.baseline_store import baseline_task_tree_digest


def test_large_baseline_file_uses_bounded_memory(tmp_path: Path) -> None:
    task = tmp_path / "case-001"
    payload = task / "repo" / "assets.dat"
    payload.parent.mkdir(parents=True)
    block = bytes(range(256)) * 256
    with payload.open("wb") as stream:
        for _ in range(256):
            stream.write(block)

    tracemalloc.start()
    try:
        observed = baseline_task_tree_digest({"codex": [task]})
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Recorded with the existing persisted identity format for this 16 MiB file.
    assert observed == "sha256:072cda027309a53c3875eeb9a7a93c29e8944cf6e5253246664eb369982097ee"
    assert peak < 2 * 1024 * 1024


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (0, "sha256:111ca93bec4a04f5e5779848d313702b8131c3a356b8ae147e5db9005b9da643"),
        (1, "sha256:a8f0eab499d65e9a88fa69aa44a9b2394dcbfba2e3b5bd0ad26d4bc3f2727533"),
        (65535, "sha256:d0f986b20c400729df580f6880ea54823b1d93c99effe68ab97839b1fe71be47"),
        (65536, "sha256:61d122ed0bb5b8e03bff2c0eafde6c3f0c9e19aed594803fca5b44b07b6ee7af"),
        (65537, "sha256:91e5cceafccc12ac81345994c0ff29f0399b4b8905597a208bfb0a98320f166f"),
    ],
)
def test_streaming_preserves_existing_binary_and_metadata_identity(tmp_path: Path, size: int, expected: str) -> None:
    for arm in ("with", "without"):
        task = tmp_path / f"case-001-{arm}-skill"
        (task / "repo").mkdir(parents=True)
        (task / "environment").mkdir()
        (task / "task.toml").write_text(f'[task]\nname = "case-001-{arm}-skill" # retained comment\n', encoding="utf-8")
        (task / "environment" / "repo-context.json").write_text(f'{{"staged_path":"{arm}"}}', encoding="utf-8")
        (task / "repo" / "repo-context.json").write_text('{"fixture":"authored"}', encoding="utf-8")
        (task / "repo" / "assets.dat").write_bytes((bytes(range(256)) * 257)[:size])

        assert baseline_task_tree_digest({"codex": [task]}) == expected


@pytest.mark.parametrize("change", ["grow", "truncate"])
def test_baseline_file_size_change_cannot_publish_a_partial_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    task = tmp_path / "case-001"
    payload = task / "repo" / "assets.dat"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"original" * 16384)
    original_open = Path.open

    class ChangingReader:
        def __init__(self, stream):
            self.stream = stream
            self.changed = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def read(self, size=-1):
            if not self.changed:
                self.changed = True
                with original_open(payload, "ab" if change == "grow" else "wb") as writer:
                    writer.write(b"changed")
            return self.stream.read(size)

    def changing_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        return ChangingReader(stream) if path == payload and mode == "rb" else stream

    monkeypatch.setattr(Path, "open", changing_open)

    with pytest.raises(ValueError, match="changed while hashing"):
        baseline_task_tree_digest({"codex": [task]})
