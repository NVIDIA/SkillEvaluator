# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for skillevaluator.deduplication.utils.chunker."""

from __future__ import annotations

from skillevaluator.deduplication.utils.chunker import (
    ContentChunk,
    chunk_file,
    chunk_markdown,
    chunk_python,
    chunk_shell,
    split_into_paragraphs,
)
from skillevaluator.deduplication.utils.skill_collector import CollectedFile


def _cf(
    content: str,
    ext: str = ".md",
    rel_path: str = "test.md",
    line_offset: int = 0,
) -> CollectedFile:
    """Shorthand to build a CollectedFile from content string."""
    from pathlib import Path

    return CollectedFile(
        path=Path(f"/fake/{rel_path}"),
        rel_path=rel_path,
        extension=ext,
        content=content,
        line_count=len(content.splitlines()),
        line_offset=line_offset,
    )


class TestContentChunk:
    def test_char_count(self) -> None:
        chunk = ContentChunk("f.md", "## H", 1, 3, "hello world", "markdown")
        assert chunk.char_count == 11

    def test_default_embedding_is_empty(self) -> None:
        chunk = ContentChunk("f.md", "## H", 1, 3, "text", "markdown")
        assert chunk.embedding == []


class TestChunkFile:
    def test_routes_md(self, make_collected_file) -> None:
        cf = make_collected_file(extension=".md", content="## Heading\n" + "a" * 100)
        chunks = chunk_file(cf)
        assert len(chunks) >= 1
        assert all(c.source_format == "markdown" for c in chunks)

    def test_routes_mdc(self, make_collected_file) -> None:
        cf = make_collected_file(extension=".mdc", rel_path="rule.mdc", content="## Rule\n" + "a" * 100)
        chunks = chunk_file(cf)
        assert all(c.source_format == "markdown" for c in chunks)

    def test_routes_py(self) -> None:
        cf = _cf(
            '"""A module docstring that is long enough to pass the minimum character filter easily."""\n',
            ".py",
            "script.py",
        )
        chunks = chunk_file(cf)
        assert all(c.source_format == "python" for c in chunks)

    def test_routes_sh(self) -> None:
        body = "setup() {\n" + "  echo line\n" * 20 + "}\n"
        cf = _cf(body, ".sh", "run.sh")
        chunks = chunk_file(cf)
        assert all(c.source_format == "shell" for c in chunks)

    def test_unknown_extension_returns_empty(self) -> None:
        cf = _cf("content", ".txt", "notes.txt")
        assert chunk_file(cf) == []


class TestChunkMarkdown:
    def test_single_heading_section(self) -> None:
        content = "## Overview\n" + "This is the overview section. " * 10 + "\n"
        chunks = chunk_markdown(_cf(content), min_chars=80)
        assert len(chunks) == 1
        assert chunks[0].heading == "## Overview"
        assert chunks[0].source_format == "markdown"

    def test_preamble_before_heading(self) -> None:
        content = "Some preamble text that is long enough.\n" * 5 + "\n## Section\nBody text.\n"
        chunks = chunk_markdown(_cf(content), min_chars=80)
        assert chunks[0].heading == "(preamble)"

    def test_multiple_headings(self) -> None:
        content = "## First\n" + "a" * 100 + "\n## Second\n" + "b" * 100 + "\n"
        chunks = chunk_markdown(_cf(content), min_chars=80)
        assert len(chunks) == 2
        assert chunks[0].heading == "## First"
        assert chunks[1].heading == "## Second"

    def test_filters_short_sections(self) -> None:
        content = "## Short\nHi\n## Long\n" + "a" * 100 + "\n"
        chunks = chunk_markdown(_cf(content), min_chars=80)
        assert len(chunks) == 1
        assert chunks[0].heading == "## Long"

    def test_line_numbers_tracked(self) -> None:
        content = "## First\nLine 2\nLine 3\n## Second\nLine 5\n"
        chunks = chunk_markdown(_cf(content), min_chars=5)
        assert chunks[0].start_line == 1
        assert chunks[1].start_line == 4

    def test_line_numbers_include_stripped_frontmatter_offset(self) -> None:
        content = "## First\nLine 6\nLine 7\n## Second\nLine 9\n"
        chunks = chunk_markdown(_cf(content, line_offset=4), min_chars=5)

        assert chunks[0].start_line == 5
        assert chunks[0].end_line == 7
        assert chunks[1].start_line == 8
        assert chunks[1].end_line == 9

    def test_no_headings_produces_preamble(self) -> None:
        content = "Just plain text without any headings.\n" * 5
        chunks = chunk_markdown(_cf(content), min_chars=80)
        assert len(chunks) == 1
        assert chunks[0].heading == "(preamble)"

    def test_oversized_section_splits_into_paragraphs(self) -> None:
        para = "a" * 200
        content = "## Big\n" + f"{para}\n\n{para}\n\n{para}\n" * 6
        chunks = chunk_markdown(_cf(content), min_chars=80)
        assert len(chunks) > 1


class TestChunkPython:
    def test_module_docstring(self) -> None:
        content = (
            '"""This is a module docstring that is long enough to pass the minimum character filter."""\n\nx = 1\n'
        )
        chunks = chunk_python(_cf(content, ".py", "mod.py"), min_chars=80)
        assert len(chunks) == 1
        assert chunks[0].heading == "(module docstring)"
        assert chunks[0].source_format == "python"

    def test_function_with_docstring(self) -> None:
        content = 'def my_func():\n    """A docstring that is long enough to pass the minimum character filter for testing."""\n    pass\n'
        chunks = chunk_python(_cf(content, ".py", "mod.py"), min_chars=80)
        assert any("my_func()" in c.heading for c in chunks)

    def test_class_heading(self) -> None:
        content = 'class MyClass:\n    """A class docstring that is definitely long enough to pass the minimum character threshold."""\n    pass\n'
        chunks = chunk_python(_cf(content, ".py", "mod.py"), min_chars=80)
        assert any("class MyClass" in c.heading for c in chunks)

    def test_syntax_error_returns_empty(self) -> None:
        content = "def broken(\n"
        chunks = chunk_python(_cf(content, ".py", "bad.py"), min_chars=10)
        assert chunks == []

    def test_short_function_excluded(self) -> None:
        content = "def tiny():\n    pass\n"
        chunks = chunk_python(_cf(content, ".py", "mod.py"), min_chars=80)
        assert chunks == []


class TestChunkShell:
    def test_function_body(self) -> None:
        body = "setup() {\n" + "  echo step\n" * 15 + "}\n"
        chunks = chunk_shell(_cf(body, ".sh", "run.sh"), min_chars=80)
        assert len(chunks) == 1
        assert chunks[0].heading == "setup()"
        assert chunks[0].source_format == "shell"

    def test_comment_block_3_plus_lines(self) -> None:
        content = "# Comment line one about setup\n# Comment line two about config\n# Comment line three about usage\n\necho hello\n"
        chunks = chunk_shell(_cf(content, ".sh", "run.sh"), min_chars=80)
        assert any(c.heading == "(comment)" for c in chunks)

    def test_short_comment_block_excluded(self) -> None:
        content = "# Short\n# Two lines\necho hello\n"
        chunks = chunk_shell(_cf(content, ".sh", "run.sh"), min_chars=10)
        comment_chunks = [c for c in chunks if c.heading == "(comment)"]
        assert len(comment_chunks) == 0

    def test_shebang_not_counted_as_comment(self) -> None:
        content = "#!/bin/bash\n# Real comment one about something\n# Real comment two about something\necho hi\n"
        chunks = chunk_shell(_cf(content, ".sh", "run.sh"), min_chars=10)
        comment_chunks = [c for c in chunks if c.heading == "(comment)"]
        assert len(comment_chunks) == 0


class TestSplitIntoParagraphs:
    def test_splits_on_blank_lines(self) -> None:
        section = ContentChunk("f.md", "## H", 1, 10, "a" * 100 + "\n\n" + "b" * 100, "markdown")
        paras = split_into_paragraphs(section, min_chars=80)
        assert len(paras) == 2

    def test_preserves_blank_line_offsets(self) -> None:
        text = "first line\nsecond line\n\nthird line\nfourth line"
        section = ContentChunk("f.md", "## H", 10, 14, text, "markdown")

        paragraphs = split_into_paragraphs(section, min_chars=5)

        assert [(item.start_line, item.end_line) for item in paragraphs] == [
            (10, 11),
            (13, 14),
        ]

    def test_returns_original_if_no_paragraph_meets_min(self) -> None:
        section = ContentChunk("f.md", "## H", 1, 3, "short\n\nshort", "markdown")
        paras = split_into_paragraphs(section, min_chars=500)
        assert len(paras) == 1
        assert paras[0] is section
