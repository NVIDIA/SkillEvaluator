# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python semantic chunks retain complete headers without implementation bodies."""

from pathlib import Path

import pytest

from skillevaluator.deduplication.utils.chunker import chunk_file, chunk_python
from skillevaluator.deduplication.utils.skill_collector import CollectedFile


def _collected(content: str) -> CollectedFile:
    return CollectedFile(Path("script.py"), "script.py", ".py", content, len(content.splitlines()))


def test_long_multiline_signature_survives_minimum_length_filter():
    signature = (
        "def convert(\n"
        "    source_document: str,\n"
        "    destination_directory: str,\n"
        "    preserve_metadata: bool = True,\n"
        ") -> list[str]:"
    )
    content = signature + "\n    return ['implementation detail']\n"

    chunks = chunk_file(_collected(content))

    assert len(chunks) == 1
    assert chunks[0].text == signature
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 6)


@pytest.mark.parametrize(
    ("signature", "body", "doc"),
    [
        (
            "async def convert(\n    value: str,\n    transform=lambda item: {'key': item},\n) -> dict[str, str]:",
            '\n    """Convert the supplied document."""\n    return transform(value)\n',
            "Convert the supplied document.",
        ),
        (
            "class Converter(\n    BaseConverter,\n    metaclass=ConverterMeta,\n):",
            '\n    """Convert source documents."""\n    implementation = 2\n',
            "Convert source documents.",
        ),
        (
            'def café(value: str = "déjà:vu"):',
            ' """Convert Unicode text."""; return "implementation detail"\n',
            "Convert Unicode text.",
        ),
        ("def transform() -> lambda value: value:", ' return "implementation detail"\n', ""),
        ("class Converter:", ' implementation = "detail"\n', ""),
        ("def convert[T: str](value: T) -> T:", " return value\n", ""),
    ],
)
def test_complete_headers_exclude_inline_bodies_and_keep_docstrings(signature, body, doc):
    chunks = chunk_python(_collected(signature + body), min_chars=1)

    assert len(chunks) == 1
    assert chunks[0].text == signature + ("\n" + doc if doc else "")


def test_class_header_stops_before_decorated_first_member():
    content = (
        "class Converter(\n    BaseConverter,\n):  # a comment with another colon:\n"
        "    # More implementation notes: ignored.\n"
        "    @decorate(lambda value: value)\n"
        "    def convert(\n        self,\n        value: str,\n    ) -> str:\n"
        '        return "implementation detail"\n'
    )

    chunks = chunk_python(_collected(content), min_chars=1)

    assert [chunk.text for chunk in chunks] == [
        "class Converter(\n    BaseConverter,\n):",
        "    def convert(\n        self,\n        value: str,\n    ) -> str:",
    ]
    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [(1, 10), (6, 10)]


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_physical_source_lines_follow_python_ast_coordinates(newline):
    # Unicode line separators inside a literal are not Python source newlines.
    content = newline.join(
        [
            'def first(value="before\u2028after\x85end"):',
            "    pass",
            "def second(",
            "    value: int,",
            ") -> int:",
            "    return value",
        ]
    )

    chunks = chunk_python(_collected(content), min_chars=1)

    assert [chunk.text for chunk in chunks] == [
        'def first(value="before\u2028after\x85end"):',
        "def second(\n    value: int,\n) -> int:",
    ]
    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [(1, 2), (3, 6)]
