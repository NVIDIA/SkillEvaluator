# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bound embedding inputs without losing source identity or inventing duplicates."""

from pathlib import Path

import pytest

from skillevaluator.deduplication.intra_skill.semantic_clustering import build_clusters
from skillevaluator.deduplication.utils.chunker import ContentChunk, chunk_file
from skillevaluator.deduplication.utils.skill_collector import CollectedFile


def collected(text, extension=".md"):
    return CollectedFile(Path("example" + extension), "example" + extension, extension, text, len(text.splitlines()))


@pytest.mark.parametrize("extension", [".md", ".mdc", ".py", ".sh"])
def test_every_embedding_input_is_bounded(extension):
    body = "measurement " * 5000
    if extension == ".py":
        text = '"""' + body + '"""\n'
    elif extension == ".sh":
        text = 'run() {\n  echo "' + body + '"\n}\n'
    else:
        text = "# Reference\n| Key | Value |\n" + "| entry | " + body + "|\n"
    chunks = chunk_file(collected(text, extension))
    assert len(chunks) > 1
    assert max(len(c.text) for c in chunks) <= 3000
    assert all(c.start_line <= c.end_line for c in chunks)


def test_split_preserves_a_short_meaningful_tail():
    chunks = chunk_file(collected("a" * 3000 + "TAIL"))
    assert "".join(c.text for c in chunks) == "a" * 3000 + "TAIL"
    assert max(len(c.text) for c in chunks) <= 3000


def test_python_module_docstring_uses_actual_ast_lines():
    text = "\n" * 40 + '"""\n' + "documentation " * 20 + '\n"""\n'
    chunks = chunk_file(collected(text, ".py"))
    assert [(c.start_line, c.end_line) for c in chunks] == [(41, 43)]


def test_split_python_signature_and_docstring_keep_the_containing_range():
    text = 'def run(\n    argument,\n):\n    """\n' + "    documentation " * 500 + '\n    """\n    pass\n'
    chunks = chunk_file(collected(text, ".py"))
    assert len(chunks) > 1
    assert all((c.start_line, c.end_line) == (1, 7) for c in chunks)


def test_sibling_pieces_do_not_create_duplicate_clusters():
    pieces = chunk_file(collected("reference " * 1500))
    for piece in pieces:
        piece.embedding = [1.0, 0.0]
    assert len(pieces) > 1
    assert build_clusters(pieces) == []


def test_independent_copy_can_still_link_split_pieces():
    pieces = chunk_file(collected("reference " * 1500))
    for piece in pieces:
        piece.embedding = [1.0, 0.0]
    copy = ContentChunk("copy.md", "reference", 1, 1, pieces[0].text, "markdown", embedding=[0.9, 0.1])
    clusters = build_clusters([*pieces, copy])
    assert len(clusters) == 1
    assert clusters[0].cross_file
    assert clusters[0].max_similarity < 1.0
    assert len(clusters[0].members) == 2
    merged = next(member for member in clusters[0].members if member.split_group is not None)
    assert merged.text.count("reference") == sum(piece.text.count("reference") for piece in pieces)


def test_fenced_headings_are_not_promoted_to_sections_during_split():
    text = "# Reference\n```text\n# example heading\n" + "x" * 7000 + "\n```\n"
    chunks = chunk_file(collected(text))
    assert len(chunks) > 1
    assert all(c.heading == "# Reference" for c in chunks)


def test_markdown_preamble_ranges_follow_leading_blank_lines():
    chunks = chunk_file(collected("\n" * 40 + "instruction " * 700))
    assert len(chunks) > 1
    assert all((c.start_line, c.end_line) == (41, 41) for c in chunks)


def test_oversized_section_preserves_short_final_paragraph():
    chunks = chunk_file(collected("# Guide\n" + "instruction " * 300 + "\n\nIMPORTANT_TAIL: Keep a backup."))
    assert any("IMPORTANT_TAIL: Keep a backup." in c.text for c in chunks)


def test_oversized_section_keeps_short_labels_with_their_examples():
    example = (
        "Example:\n\n```bash\nskill run --input data.csv\n\nskill run --verbose\n```\n\n"
        "This example shows how the command reads its input and reports progress.\n\n"
    )
    chunks = chunk_file(collected("# Usage\n" + example * 25))
    assert len(chunks) > 1
    assert min(len(c.text) for c in chunks) >= 80
    assert max(len(c.text) for c in chunks) <= 3000
    assert len({c.split_group for c in chunks}) == 1
    for chunk in chunks:
        chunk.embedding = [1.0, 0.0]
    assert build_clusters(chunks) == []


def test_packed_short_tail_keeps_source_range_across_blank_lines():
    text = "# Guide\n" + "x" * 2990 + "\n" * 100 + "IMPORTANT_TAIL: Keep a backup."
    chunks = chunk_file(collected(text))
    tail = next(chunk for chunk in chunks if "IMPORTANT_TAIL" in chunk.text)
    assert tail.end_line == len(text.splitlines())
    assert max(len(chunk.text) for chunk in chunks) <= 3000
