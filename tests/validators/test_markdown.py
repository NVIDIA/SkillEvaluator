# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for structural Markdown helpers used by deterministic validators."""

import pytest

from skillevaluator.validators import markdown as markdown_validator
from skillevaluator.validators.markdown import markdown_link_targets, normalized_local_path

_CLOSED_NON_NAVIGATION_HTML_TAGS = (
    "script",
    "style",
    "textarea",
    "title",
    "xmp",
    "iframe",
    "noembed",
    "noframes",
    "template",
    "XmP",
)


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
        ("![diagram](inline.png)", "inline.png"),
        ("![diagram][asset]\n\n[asset]: reference.png", "reference.png"),
    ],
)
def test_image_targets_are_opt_in(content: str, target: str) -> None:
    assert markdown_link_targets(content) == []
    assert markdown_link_targets(content, include_images=True) == [target]


def test_opted_in_images_preserve_order_and_escaped_destinations() -> None:
    content = "[first](first.md) ![diagram](a&b.png) [last](last.md)"

    assert markdown_link_targets(content, include_images=True) == ["first.md", "a&b.png", "last.md"]


def test_local_path_normalization_can_preserve_directory_targets() -> None:
    assert normalized_local_path("docs/") is None
    assert normalized_local_path("docs/", allow_directory=True) == "docs"
    assert normalized_local_path("") == "."


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
        ('<A HREF="upper.md">docs</A>', "upper.md"),
        ("<div>\n<a href='block.md'>docs</a>\n</div>", "block.md"),
    ],
)
def test_html_anchor_links_are_extracted(content: str, target: str) -> None:
    assert markdown_link_targets(content) == [target]


def test_cdata_mode_falls_back_to_legacy_stdlib_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def legacy_set_cdata_mode(_parser, element: str) -> None:
        calls.append(element)

    monkeypatch.setattr(markdown_validator.HTMLParser, "set_cdata_mode", legacy_set_cdata_mode)
    monkeypatch.setattr(markdown_validator, "_HTML_PARSER_SUPPORTS_ESCAPABLE_CDATA", False)
    parser = markdown_validator._AnchorHrefParser()

    parser.set_cdata_mode("script")
    parser.set_cdata_mode("textarea", escapable=True)

    assert calls == ["script", "textarea"]


def test_cdata_mode_forwards_escapable_to_new_stdlib_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []

    def modern_set_cdata_mode(_parser, element: str, *, escapable: bool = False) -> None:
        calls.append((element, escapable))

    monkeypatch.setattr(markdown_validator.HTMLParser, "set_cdata_mode", modern_set_cdata_mode)
    monkeypatch.setattr(markdown_validator, "_HTML_PARSER_SUPPORTS_ESCAPABLE_CDATA", True)
    parser = markdown_validator._AnchorHrefParser()

    parser.set_cdata_mode("textarea", escapable=True)

    assert calls == [("textarea", True)]


def test_markdown_and_html_links_preserve_source_order() -> None:
    content = '[first](first.md) <a href="second.md">second</a> [third](third.md)'

    assert markdown_link_targets(content) == ["first.md", "second.md", "third.md"]


@pytest.mark.parametrize("tag", _CLOSED_NON_NAVIGATION_HTML_TAGS)
def test_non_navigation_html_suppresses_links_and_then_restores_navigation(tag: str) -> None:
    content = f'prefix <{tag}><a href="hidden.md">[hidden](also-hidden.md)</a></{tag}> '
    content += '<a href="visible.md">visible</a> [last](last.md)'

    assert markdown_link_targets(content) == ["visible.md", "last.md"]


@pytest.mark.parametrize("tag", _CLOSED_NON_NAVIGATION_HTML_TAGS)
def test_self_closing_syntax_does_not_close_non_void_html_content(tag: str) -> None:
    content = f'prefix <{tag}/><a href="hidden.md">hidden</a></{tag}> <a href="visible.md">visible</a>'

    assert markdown_link_targets(content) == ["visible.md"]


def test_nested_template_content_remains_non_navigation_until_outer_close() -> None:
    content = (
        '<template><template><a href="hidden.md">hidden</a></template>'
        '<a href="still-hidden.md">still hidden</a></template>'
        '<a href="visible.md">visible</a>'
    )

    assert markdown_link_targets(content) == ["visible.md"]


def test_shadowroot_template_without_a_valid_host_remains_non_navigation() -> None:
    content = (
        '<a><template shadowrootmode="open"><a href="hidden.md">hidden</a></template></a>'
        '<a href="visible.md">visible</a>'
    )

    assert markdown_link_targets(content) == ["visible.md"]


@pytest.mark.parametrize("mode", ["open", "CLOSED"])
def test_shadowroot_template_content_is_conservatively_non_navigation(mode: str) -> None:
    content = (
        f'<div><template shadowrootmode="{mode}"><a href="shadow.md">shadow link</a></template></div>'
        '<a href="visible.md">visible</a>'
    )

    assert markdown_link_targets(content) == ["visible.md"]


def test_raw_text_pseudo_start_tag_does_not_require_an_extra_close() -> None:
    content = '<xmp><xmp><a href="hidden.md">hidden</a></xmp><a href="visible.md">visible</a>'

    assert markdown_link_targets(content) == ["visible.md"]


@pytest.mark.parametrize("outer", ["xmp", "iframe", "noembed", "noframes"])
@pytest.mark.parametrize("pseudo_inner", ["script", "style", "textarea", "title"])
def test_text_only_pseudo_start_does_not_hijack_outer_state(outer: str, pseudo_inner: str) -> None:
    content = f'<{outer}><{pseudo_inner}></{outer}><a href="visible.md">visible</a>'

    assert markdown_link_targets(content) == ["visible.md"]


@pytest.mark.parametrize("tag", ["xmp", "iframe", "noembed", "noframes"])
def test_comment_syntax_does_not_hide_raw_text_end_tag(tag: str) -> None:
    content = f'<{tag}><!-- </{tag}> --><a href="visible.md">visible</a>'

    assert markdown_link_targets(content) == ["visible.md"]


@pytest.mark.parametrize("tag", ["xmp", "iframe", "noembed", "noframes"])
def test_self_closing_raw_text_syntax_keeps_native_text_mode(tag: str) -> None:
    content = f'<{tag}/><!-- </{tag}> --><a href="visible.md">visible</a>'

    assert markdown_link_targets(content) == ["visible.md"]


def test_mismatched_raw_text_close_does_not_restore_navigation() -> None:
    content = '<iframe><a href="hidden.md">hidden</a></xmp><a href="still-hidden.md">still hidden</a>'

    assert markdown_link_targets(content) == []


def test_plaintext_html_suppresses_navigation_to_end_of_document() -> None:
    content = (
        '<plaintext><a href="hidden.md">[also hidden](also-hidden.md)</a>'
        '</plaintext> <a href="still-hidden.md">still hidden</a> [last](last.md)'
    )

    assert markdown_link_targets(content) == []


def test_script_double_escaped_content_does_not_produce_links() -> None:
    content = '<script><!--<script></script><a href="hidden.md">hidden</a>--></script><a href="visible.md">visible</a>'

    assert markdown_link_targets(content) == ["visible.md"]


@pytest.mark.parametrize("escape_boundary", ["<!-->", "<!--->"])
def test_script_escaped_dash_dash_greater_than_returns_to_data(escape_boundary: str) -> None:
    content = (
        f'<script>{escape_boundary}<script><a href="hidden.md">hidden</a></script><a href="visible.md">visible</a>'
    )

    assert markdown_link_targets(content) == ["visible.md"]


@pytest.mark.parametrize(
    "content",
    [
        '<script><!--<\u017fcript></script><a href="visible.md">visible</a>',
        '<script></\u017fcript><a href="hidden.md">hidden</a></script><a href="visible.md">visible</a>',
    ],
)
def test_script_end_scanner_uses_ascii_case_insensitive_matching(content: str) -> None:
    assert markdown_link_targets(content) == ["visible.md"]


def test_html_tag_matching_does_not_casefold_non_ascii_characters() -> None:
    content = '<noframe\u017f><a href="inside.md">inside</a></noframe\u017f><a href="outside.md">outside</a>'

    assert markdown_link_targets(content) == ["inside.md", "outside.md"]


def test_many_inline_html_tokens_are_fed_to_html_parser_once(monkeypatch: pytest.MonkeyPatch) -> None:
    feed_calls = 0
    original_feed = markdown_validator._AnchorHrefParser.feed

    def counted_feed(parser: markdown_validator._AnchorHrefParser, content: str) -> None:
        nonlocal feed_calls
        feed_calls += 1
        original_feed(parser, content)

    monkeypatch.setattr(markdown_validator._AnchorHrefParser, "feed", counted_feed)
    content = "prefix <xmp>" + ' <a href="hidden.md">hidden</a>' * 100

    assert markdown_link_targets(content) == []
    assert feed_calls == 1


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
