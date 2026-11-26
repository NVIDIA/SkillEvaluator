# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The context chunk ceiling must apply before nested shell bodies expand."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skillevaluator.constants import CONTENT_DEDUP_MAX_CHUNK_CHARS, CONTENT_DEDUP_MAX_CHUNKS
from skillevaluator.deduplication.intra_skill import intra_skill_validator
from skillevaluator.deduplication.intra_skill.intra_skill_validator import IntraSkillValidator
from skillevaluator.deduplication.plugin import commands as plugin_commands
from skillevaluator.deduplication.utils.chunker import ChunkCountLimitError, chunk_file
from skillevaluator.deduplication.utils.skill_collector import CollectedFile


class _SliceCountingText(str):
    """Count the source slices used to materialize shell function bodies.

    This pins the overlapping copies, not all character reads or allocations.
    The separate real-file allocation measurement covers the full validator.
    """

    __slots__ = ("copied",)

    def __new__(cls, value):
        instance = super().__new__(cls, value)
        instance.copied = 0
        return instance

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(key, slice):
            self.copied += len(result)
        return result


@pytest.mark.parametrize("functions", [1000, 2000])
def test_nested_shell_expansion_stops_at_the_scan_chunk_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, functions: int
) -> None:
    source = _SliceCountingText(
        "".join(f"f{index}() {{\n" for index in range(functions))
        + 'echo "retained source evidence"\n'
        + "}\n" * functions
    )
    collected = CollectedFile(tmp_path / "nested.sh", "nested.sh", ".sh", source, len(source.splitlines()))
    embedding = Mock(side_effect=AssertionError("a capped scan must not request embeddings"))
    monkeypatch.setattr("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient", embedding)

    result = IntraSkillValidator().validate(tmp_path, _collected_files=[collected])

    assert result.passed is False
    assert result.status == "incomplete"
    assert result.incomplete_scans == ["context-deduplication"]
    assert result.findings == []
    assert len(result.errors) == 1
    assert "more than 1024 content chunks" in result.errors[0]
    assert "nested.sh" in result.errors[0]
    assert source.copied <= CONTENT_DEDUP_MAX_CHUNKS * CONTENT_DEDUP_MAX_CHUNK_CHARS + 2 * len(source)
    assert result.metadata["work_limit"] == {
        "check_name": "chunk_count_limit",
        "file_path": "nested.sh",
        "actual": 1025,
        "limit": 1024,
    }
    embedding.assert_not_called()


@pytest.mark.parametrize("extension", [".md", ".mdc", ".py", ".sh"])
def test_chunk_budget_preserves_exact_complete_output(extension: str) -> None:
    content = "preserve the complete source and line range " * 300
    if extension == ".py":
        content = f'"""{content}"""\n'
    elif extension == ".sh":
        content = f'run() {{\n  echo "{content}"\n}}\n'
    collected = CollectedFile(Path("example" + extension), "example" + extension, extension, content, 3)
    expected = chunk_file(collected)
    assert len(expected) > 1

    assert chunk_file(collected, max_chunks=len(expected)) == expected
    with pytest.raises(ChunkCountLimitError) as error:
        chunk_file(collected, max_chunks=len(expected) - 1)
    assert error.value.actual == len(expected)


def test_zero_remaining_budget_can_still_read_files_without_chunks() -> None:
    collected = CollectedFile(Path("empty.sh"), "empty.sh", ".sh", "#!/bin/sh\ntrue\n", 2)

    assert chunk_file(collected, max_chunks=0) == []


@pytest.mark.parametrize("limit", [3, 4])
def test_chunk_budget_is_shared_across_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: int) -> None:
    files = []
    for filename in ("first.sh", "second.sh"):
        source = "".join(f'f{index}() {{\n echo "' + str(index) * 100 + '"\n}\n' for index in range(2))
        files.append(CollectedFile(tmp_path / filename, filename, ".sh", source, len(source.splitlines())))
    embedding = Mock()
    embedding.return_value.embed.return_value = [[float(i == j) for i in range(4)] for j in range(4)]
    monkeypatch.setattr(intra_skill_validator, "EmbeddingClient", embedding)
    monkeypatch.setattr(intra_skill_validator, "CONTENT_DEDUP_MAX_CHUNKS", limit)

    result = IntraSkillValidator().validate(tmp_path, _collected_files=files)

    if limit == 3:
        assert result.passed is False
        assert result.status == "incomplete"
        assert result.metadata["work_limit"] == {
            "check_name": "chunk_count_limit",
            "file_path": "second.sh",
            "actual": 4,
            "limit": 3,
        }
        embedding.assert_not_called()
    else:
        assert result.passed is True
        assert result.incomplete_scans == []
        assert "work_limit" not in result.metadata
        embedding.return_value.embed.assert_called_once()
        assert len(embedding.return_value.embed.call_args.args[0]) == 4


@pytest.mark.parametrize("functions", [1000, 2000])
def test_plugin_preflight_stops_nested_shell_expansion_before_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, functions: int
) -> None:
    source = _SliceCountingText(
        "".join(f"f{index}() {{\n" for index in range(functions))
        + 'echo "retained source evidence"\n'
        + "}\n" * functions
    )
    collected = CollectedFile(tmp_path / "nested.sh", "nested.sh", ".sh", source, len(source.splitlines()))
    profile = SimpleNamespace(bundled_skills=[SimpleNamespace(path=tmp_path, root_relative="skills/nested")])
    monkeypatch.setattr(plugin_commands, "collect_files", lambda _path: [collected])
    embedding = Mock(side_effect=AssertionError("an over-budget bundle must not request embeddings"))
    monkeypatch.setattr(intra_skill_validator, "EmbeddingClient", embedding)

    results = plugin_commands.run_plugin_skill_context_dedup(profile)

    assert len(results) == 1
    result = results[0]
    assert result.passed is False
    assert result.status == "incomplete"
    assert result.incomplete_scans == ["bundle-context"]
    assert result.findings == []
    assert result.errors == ["Aggregate bundle content limit exceeded before provider work"]
    limit = plugin_commands.MAX_PLUGIN_DEDUP_CHUNKS
    assert source.copied <= limit * CONTENT_DEDUP_MAX_CHUNK_CHARS + 2 * len(source)
    assert result.metadata["work_limit"] == {
        "check_name": "plugin_content_limit",
        "actual_chunks": limit + 1,
        "chunk_limit": limit,
        "actual_text_bytes": len(source.encode("utf-8")),
        "text_byte_limit": plugin_commands.MAX_PLUGIN_DEDUP_TEXT_BYTES,
        "actual_files": 1,
        "file_limit": plugin_commands.MAX_PLUGIN_DEDUP_FILES,
    }
    used = result.metadata["plugin_context_budget"]["consumed"]
    assert used == {
        "chunks_collected": limit + 1,
        "chunks_embedded": 0,
        "scalar_comparisons": 0,
        "llm_clusters_submitted": 0,
        "prompt_chars_submitted": 0,
    }
    embedding.assert_not_called()


@pytest.mark.parametrize("limit", [3, 4])
def test_plugin_chunk_budget_is_shared_across_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    skills = []
    collected = {}
    for name in ("first", "second"):
        path = tmp_path / name
        path.mkdir()
        source = "".join(f'f{index}() {{\n echo "' + str(index) * 100 + '"\n}\n' for index in range(2))
        skills.append(SimpleNamespace(path=path, root_relative=f"skills/{name}"))
        collected[path] = [CollectedFile(path / "run.sh", "run.sh", ".sh", source, len(source.splitlines()))]
    monkeypatch.setattr(plugin_commands, "collect_files", collected.__getitem__)
    embedding = Mock()
    embedding.return_value.embed.return_value = [[1.0, 0.0], [0.0, 1.0]]
    monkeypatch.setattr(intra_skill_validator, "EmbeddingClient", embedding)

    results = plugin_commands.run_plugin_skill_context_dedup(SimpleNamespace(bundled_skills=skills), max_chunks=limit)

    assert results[-1].metadata["plugin_context_budget"]["consumed"]["chunks_collected"] == 4
    if limit == 3:
        assert len(results) == 1
        assert results[0].passed is False
        assert results[0].status == "incomplete"
        assert results[0].metadata["work_limit"]["check_name"] == "plugin_content_limit"
        assert results[0].metadata["work_limit"]["actual_chunks"] == 4
        assert results[0].metadata["work_limit"]["chunk_limit"] == 3
        embedding.assert_not_called()
    else:
        assert len(results) == 2
        assert all(result.passed and not result.is_incomplete for result in results)
        assert [result.metadata["bundled_skill"] for result in results] == ["skills/first", "skills/second"]
        assert embedding.return_value.embed.call_count == 2
        assert all(len(call.args[0]) == 2 for call in embedding.return_value.embed.call_args_list)
