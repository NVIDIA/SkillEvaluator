# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for structural Markdown helpers used by deterministic validators."""

import pytest

from skillevaluator.validators.markdown import markdown_link_targets


@pytest.mark.parametrize(
    "content",
    [
        "`[example](advanced.md)`",
        "    [example](advanced.md)",
        "<span data-doc='[example](advanced.md)'>text</span>",
        "<span href='not-a-link.md'>text</span>",
        "<!-- <a href='hidden.md'>hidden</a> -->",
        "![diagram](advanced.md)",
        "![diagram][guide]\n\n[guide]: advanced.md",
        "![[navigation](should-not-be-returned.md)](diagram.png)",
        "> ```markdown\n> [example](advanced.md)\n> ```",
    ],
)
def test_non_navigation_markdown_does_not_produce_links(content: str) -> None:
    assert markdown_link_targets(content) == []


@pytest.mark.parametrize(
    ("content", "target"),
    [
        ("[guide](inline.md)", "inline.md"),
        ("[outer [inner]](nested.md)", "nested.md"),
        (r"[escaped \] label](escaped.md)", "escaped.md"),
        ("[reference][guide]\n\n[guide]: reference.md", "reference.md"),
        ("[reference][guide\\]]\n\n[guide\\]]: escaped-ref.md", "escaped-ref.md"),
        ("[guide][]\n\n[guide]: collapsed.md", "collapsed.md"),
        ("[guide]\n\n[guide]: shortcut.md", "shortcut.md"),
        ('[guide](<my file.md> "Title")', "my%20file.md"),
        ("[wrapped\nlabel](multiline.md)", "multiline.md"),
    ],
)
def test_commonmark_links_are_extracted(content: str, target: str) -> None:
    assert markdown_link_targets(content) == [target]


@pytest.mark.parametrize(
    ("content", "target"),
    [
        ('<a href="README.md">docs</a>', "README.md"),
        ("<div>\n<a href='block.md'>docs</a>\n</div>", "block.md"),
    ],
)
def test_html_anchor_links_are_extracted(content: str, target: str) -> None:
    assert markdown_link_targets(content) == [target]


def test_markdown_and_html_links_preserve_source_order() -> None:
    content = '[first](first.md) <a href="second.md">second</a> [third](third.md)'

    assert markdown_link_targets(content) == ["first.md", "second.md", "third.md"]


def test_raw_text_html_suppresses_links_and_then_restores_navigation() -> None:
    content = (
        'prefix <script><a href="hidden.md">[hidden](also-hidden.md)</a></script> '
        '<a href="visible.md">visible</a> [last](last.md)'
    )

    assert markdown_link_targets(content) == ["visible.md", "last.md"]


def test_escaped_space_destination_does_not_produce_link() -> None:
    assert markdown_link_targets(r"[guide](my\ file.md)") == []


def test_yaml_frontmatter_links_are_ignored() -> None:
    content = (
        '---\ndescription: "See [metadata](frontmatter.md) for details."\n---\n\nSee [body](body.md) for details.\n'
    )

    assert markdown_link_targets(content) == ["body.md"]


def test_thematic_breaks_do_not_hide_markdown_links() -> None:
    content = "---\nSee [guide](advanced.md).\n---\n"

    assert markdown_link_targets(content) == ["advanced.md"]


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_frontmatter_does_not_promote_indented_first_body_line(newline: str) -> None:
    content = newline.join(["---", "title: Test", "---", "    [code](false.md)", ""])

    assert markdown_link_targets(content) == []


def test_malformed_80_kib_input_does_not_produce_links() -> None:
    """Malformed link syntax must not manufacture navigation destinations."""
    content = ("[broken](" * 10_000)[: 80 * 1024]

    assert len(content) == 80 * 1024
    assert markdown_link_targets(content) == []
