# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep literal fenced-code headings within their parent deduplication section."""

from pathlib import Path

import pytest

from skillevaluator.deduplication.utils.chunker import chunk_file
from skillevaluator.deduplication.utils.skill_collector import CollectedFile, collect_files


def collected(content: str, offset: int = 0) -> CollectedFile:
    return CollectedFile(
        path=Path("SKILL.md"),
        rel_path="SKILL.md",
        extension=".md",
        content=content,
        line_count=len(content.splitlines()) + offset,
        line_offset=offset,
    )


@pytest.mark.parametrize("offset", [0, 5])
@pytest.mark.parametrize(
    "opener, code, closer",
    [
        ("```python", "# A comment\n## Another comment\nprint(1)", "```"),
        ("~~~sh", "# A comment\nprintf done", "~~~"),
        ("````markdown", "```\n# Nested example heading\n```", "````"),
        ("~~~~", "~~~\n# Still code", "~~~~~"),
        ("```", "~~~\n# Different fence marker", "```"),
        ("   ```python", "# Zero-indent content\nprint(1)", "   ```"),
    ],
)
def test_fenced_headings_remain_literal(opener: str, code: str, closer: str, offset: int) -> None:
    section = f"## Usage\n{opener}\n{code}\n{closer}\n"
    content = section + "## Next\nDone\n"
    chunks = chunk_file(collected(content, offset), min_chars=1)
    assert [chunk.heading for chunk in chunks] == ["## Usage", "## Next"]
    assert chunks[0].text == section.strip()
    assert chunks[0].start_line == offset + 1
    assert chunks[0].end_line == offset + len(section.splitlines())
    assert chunks[1].start_line == chunks[0].end_line + 1
    assert chunks[1].end_line == offset + len(content.splitlines())
    assert all(chunk.source_file == "SKILL.md" for chunk in chunks)


@pytest.mark.parametrize("opener", ["```python", "~~~"])
def test_unclosed_fence_keeps_all_remaining_lines(opener: str) -> None:
    content = f"## Usage\n{opener}\n# A comment\n## Not a new section\nprint(1)\n"
    chunks = chunk_file(collected(content), min_chars=1)
    assert len(chunks) == 1
    assert chunks[0].heading == "## Usage"
    assert chunks[0].text == content.strip()
    assert chunks[0].end_line == len(content.splitlines())


@pytest.mark.parametrize("line", ["``not a fence``", "Text with ``` inline backticks"])
def test_inline_delimiters_do_not_hide_real_headings(line: str) -> None:
    content = f"## Usage\n{line}\n## Next\nDone\n"
    chunks = chunk_file(collected(content), min_chars=1)
    assert [chunk.heading for chunk in chunks] == ["## Usage", "## Next"]


def test_code_comment_does_not_drop_a_short_parent_section() -> None:
    content = "## Usage\n```python\n# A long explanatory comment that belongs to Usage\nprint(1)\n```"
    chunks = chunk_file(collected(content), min_chars=40)
    assert len(chunks) == 1
    assert chunks[0].heading == "## Usage"
    assert chunks[0].text == content


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_file_collection_preserves_original_lines_and_fence_content(tmp_path: Path, newline: str) -> None:
    frontmatter = "---\nname: example\ndescription: A simple example\n---\n"
    section = "## Usage\n```python\n# Literal comment\nprint(1)\n```\n"
    content = (frontmatter + section + "## Next\nDone\n").replace("\n", newline)
    (tmp_path / "SKILL.md").write_bytes(content.encode("utf-8"))
    files = collect_files(tmp_path)
    assert len(files) == 1
    chunks = chunk_file(files[0], min_chars=1)
    assert [chunk.heading for chunk in chunks] == ["## Usage", "## Next"]
    original_lines = content.splitlines(keepends=True)
    for chunk in chunks:
        assert "".join(original_lines[chunk.start_line - 1 : chunk.end_line]).strip() == chunk.text
    assert chunks[0].start_line == 5
    assert chunks[0].text == section.replace("\n", newline).strip()
