# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural Markdown helpers for deterministic validators."""

from collections.abc import Iterable, Iterator
from html.parser import HTMLParser

import yaml
from markdown_it import MarkdownIt
from markdown_it.token import Token

from skillevaluator.validators.frontmatter_parser import FRONTMATTER_PATTERN

_MARKDOWN_PARSER = MarkdownIt("commonmark", {"html": True, "linkify": False})


class _AnchorHrefParser(HTMLParser):
    """Collect string href values from HTML anchor start tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.targets: list[str] = []
        self._raw_text_tag: str | None = None

    @property
    def suppresses_navigation(self) -> bool:
        return self._raw_text_tag is not None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag.casefold()
        if normalized_tag in {"script", "style"}:
            self._raw_text_tag = normalized_tag
            return
        if self.suppresses_navigation or normalized_tag != "a":
            return
        for name, value in attrs:
            if name.casefold() == "href" and isinstance(value, str):
                self.targets.append(value)
                break

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == self._raw_text_tag:
            self._raw_text_tag = None

    def feed_targets(self, content: str) -> list[str]:
        start = len(self.targets)
        self.feed(content)
        return self.targets[start:]


def _walk_tokens(tokens: Iterable[Token]) -> Iterator[Token]:
    """Yield tokens and their descendants in document order."""
    for token in tokens:
        yield token
        if token.children and token.type != "image":
            yield from _walk_tokens(token.children)


def markdown_link_targets(content: str) -> list[str]:
    """Return parser-normalized Markdown link destinations in document order."""
    frontmatter = FRONTMATTER_PATTERN.match(content)
    markdown_content = content
    if frontmatter:
        try:
            frontmatter_data = yaml.safe_load(frontmatter.group(1))
        except yaml.YAMLError:
            frontmatter_data = None
        if isinstance(frontmatter_data, dict):
            markdown_content = frontmatter.group(2)

    targets: list[str] = []
    html_parser = _AnchorHrefParser()
    for token in _walk_tokens(_MARKDOWN_PARSER.parse(markdown_content)):
        if token.type == "link_open":
            href = token.attrGet("href")
            if isinstance(href, str) and not html_parser.suppresses_navigation:
                targets.append(href)
        elif token.type in {"html_inline", "html_block"}:
            targets.extend(html_parser.feed_targets(token.content))
    html_parser.close()
    return targets
