# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural Markdown helpers for deterministic validators."""

import posixpath
import re
from collections.abc import Iterable, Iterator
from html import escape
from html.parser import HTMLParser
from inspect import signature
from pathlib import PureWindowsPath
from urllib.parse import unquote, urlsplit

import yaml
from markdown_it import MarkdownIt
from markdown_it.rules_inline import html_inline
from markdown_it.rules_inline.state_inline import StateInline
from markdown_it.token import Token

from skillevaluator.validators.frontmatter_parser import FRONTMATTER_PATTERN

_MARKDOWN_PARSER = MarkdownIt("commonmark", {"html": True, "linkify": False})
_TEXT_ONLY_HTML_TAGS = frozenset(
    {"script", "style", "textarea", "title", "xmp", "iframe", "noembed", "noframes", "plaintext"}
)
_INERT_HTML_TAGS = frozenset({"template"})
_NON_NAVIGATION_HTML_TAGS = _TEXT_ONLY_HTML_TAGS | _INERT_HTML_TAGS
_SCRIPT_START_RE = re.compile(r"<script(?=[\t\n\r\f />])", re.IGNORECASE | re.ASCII)
_SCRIPT_END_RE = re.compile(r"</script(?=[\t\n\r\f />])", re.IGNORECASE | re.ASCII)
_SCRIPT_ESCAPED_STATES = frozenset({"escaped", "escaped-dash", "escaped-dash-dash"})
_SCRIPT_DOUBLE_ESCAPED_STATES = frozenset({"double-escaped", "double-escaped-dash", "double-escaped-dash-dash"})
_URL_EDGE_C0_OR_SPACE_RE = re.compile(r"^[\x00-\x20]+|[\x00-\x20]+$")
_HTML_PARSER_SUPPORTS_ESCAPABLE_CDATA = "escapable" in signature(HTMLParser.set_cdata_mode).parameters


def _guarded_html_inline(state: StateInline, silent: bool) -> bool:
    """Avoid rescanning suffixes for HTML fragments with no possible terminator."""
    terminator = None
    if state.src.startswith("<!--", state.pos) and not state.src.startswith(("<!-->", "<!--->"), state.pos):
        terminator = "-->"
    elif state.src.startswith("<?", state.pos):
        terminator = "?>"
    elif state.src.startswith("<![CDATA[", state.pos):
        terminator = "]]>"
    elif state.src.startswith("<!", state.pos) and state.pos + 2 < len(state.src):
        letter = state.src[state.pos + 2]
        if "A" <= letter <= "Z" or "a" <= letter <= "z":
            terminator = ">"
    if terminator is not None:
        # Each inline source has its own offsets, even within the same document.
        if state.env.get("_html_source") is not state.src:
            state.env["_html_source"] = state.src
            state.env["_html_ends"] = {end: state.src.rfind(end) for end in ("-->", "?>", "]]>", ">")}
        if state.pos > state.env["_html_ends"][terminator]:
            return False
    return html_inline(state, silent)


_MARKDOWN_PARSER.inline.ruler.at("html_inline", _guarded_html_inline)


def normalized_local_path(href: str, *, allow_directory: bool = False, reject_anchors: bool = False) -> str | None:
    """Normalize a local destination, optionally rejecting anchors acquired in normalization.

    Explicit URL schemes and URLs whose raw path begins with ``/`` remain
    outside local checks. Unparseable URLs also return ``None``.
    With ``reject_anchors``, a relative URL that gains an absolute or
    drive-relative path raises ``ValueError`` instead of being skipped.
    """
    href = _URL_EDGE_C0_OR_SPACE_RE.sub("", href)
    try:
        parsed = urlsplit(href)
    except ValueError:
        return None
    if parsed.scheme or parsed.netloc:
        return None

    # Preserve invalid UTF-8 bytes instead of aliasing distinct destinations to
    # the replacement character. Scheme classification belongs to the raw URL.
    path = unquote(parsed.path, errors="surrogateescape").replace("\\", "/")
    if parsed.path.startswith("/"):
        return None
    normalized = posixpath.normpath(path)
    if PureWindowsPath(normalized).anchor:
        if reject_anchors:
            raise ValueError("absolute or drive-relative path")
        return None
    if not allow_directory and path.endswith(("/", "/.", "/..")):
        return None
    return normalized


def _find_script_end(content: str, start: int) -> int | None:
    state = "data"
    position = start
    while position < len(content):
        character = content[position]
        if state == "data":
            if content.startswith("<!--", position):
                state = "escaped-dash-dash"
                position += 4
                continue
            if _SCRIPT_END_RE.match(content, position):
                return position
        elif state in _SCRIPT_ESCAPED_STATES:
            if character == "<":
                if _SCRIPT_END_RE.match(content, position):
                    return position
                if _SCRIPT_START_RE.match(content, position):
                    state = "double-escaped"
                    position += len("<script")
                    continue
                state = "escaped"
            elif state == "escaped":
                if character == "-":
                    state = "escaped-dash"
            elif state == "escaped-dash":
                state = "escaped-dash-dash" if character == "-" else "escaped"
            elif character == ">":
                state = "data"
            elif character != "-":
                state = "escaped"
        elif state in _SCRIPT_DOUBLE_ESCAPED_STATES:
            if character == "<" and _SCRIPT_END_RE.match(content, position):
                state = "escaped"
                position += len("</script")
                continue
            if character == "<":
                state = "double-escaped"
            elif state == "double-escaped":
                if character == "-":
                    state = "double-escaped-dash"
            elif state == "double-escaped-dash":
                state = "double-escaped-dash-dash" if character == "-" else "double-escaped"
            elif character == ">":
                state = "data"
            elif character != "-":
                state = "double-escaped"
        position += 1
    return None


class _ScriptEndSearch:
    def search(self, content: str, start: int = 0) -> re.Match[str] | None:
        position = _find_script_end(content, start)
        return None if position is None else _SCRIPT_END_RE.match(content, position)


_SCRIPT_END_SEARCH = _ScriptEndSearch()


class _AnchorHrefParser(HTMLParser):
    """Collect string href values from HTML anchor start tags."""

    CDATA_CONTENT_ELEMENTS = ("script", "style", "xmp", "iframe", "noembed", "noframes")
    RCDATA_CONTENT_ELEMENTS = ("textarea", "title")

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.targets: list[str] = []
        self._non_navigation_tags: list[str] = []

    def set_cdata_mode(self, elem: str, *, escapable: bool = False) -> None:
        if _HTML_PARSER_SUPPORTS_ESCAPABLE_CDATA:
            super().set_cdata_mode(elem, escapable=escapable)
        else:
            # ``escapable`` was added in CPython 3.12.12 and 3.13.6.
            super().set_cdata_mode(elem)
        if elem == "script":
            self.interesting = _SCRIPT_END_SEARCH

    @property
    def suppresses_navigation(self) -> bool:
        return bool(self._non_navigation_tags)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # HTMLParser has already normalized ASCII tag and attribute names to
        # lowercase. Unicode case folding would incorrectly turn unknown names
        # containing U+017F into reserved HTML names such as ``noframes``.
        normalized_tag = tag
        if self.suppresses_navigation:
            if (
                self._non_navigation_tags[-1] not in _TEXT_ONLY_HTML_TAGS
                and normalized_tag in _NON_NAVIGATION_HTML_TAGS
            ):
                self._non_navigation_tags.append(normalized_tag)
            return
        if normalized_tag in _NON_NAVIGATION_HTML_TAGS:
            self._non_navigation_tags.append(normalized_tag)
            return
        if normalized_tag != "a":
            return
        for name, value in attrs:
            if name == "href" and isinstance(value, str):
                self.targets.append(value)
                break

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag
        if normalized_tag in _NON_NAVIGATION_HTML_TAGS:
            self.handle_starttag(tag, attrs)
            if normalized_tag in self.CDATA_CONTENT_ELEMENTS:
                self.set_cdata_mode(normalized_tag)
            elif normalized_tag in self.RCDATA_CONTENT_ELEMENTS:
                self.set_cdata_mode(normalized_tag, escapable=True)
            return
        super().handle_startendtag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if not self._non_navigation_tags or self._non_navigation_tags[-1] == "plaintext":
            return
        if tag == self._non_navigation_tags[-1]:
            self._non_navigation_tags.pop()


def _walk_tokens(tokens: Iterable[Token]) -> Iterator[Token]:
    """Yield tokens and their descendants in document order."""
    for token in tokens:
        yield token
        if token.children and token.type != "image":
            yield from _walk_tokens(token.children)


def markdown_link_targets(content: str, *, include_images: bool = False) -> list[str]:
    """Return parser-normalized Markdown destinations in document order."""
    frontmatter = FRONTMATTER_PATTERN.match(content)
    markdown_content = content
    if frontmatter:
        try:
            frontmatter_data = yaml.safe_load(frontmatter.group(1))
        except (yaml.YAMLError, ValueError, RecursionError):
            frontmatter_data = None
        if isinstance(frontmatter_data, dict):
            markdown_content = frontmatter.group(2)

    html_parts: list[str] = []
    for token in _walk_tokens(_MARKDOWN_PARSER.parse(markdown_content)):
        if token.type == "link_open":
            href = token.attrGet("href")
            if isinstance(href, str):
                html_parts.append(f'<a href="{escape(href, quote=True)}"></a>')
        elif include_images and token.type == "image":
            source = token.attrGet("src")
            if isinstance(source, str):
                html_parts.append(f'<a href="{escape(source, quote=True)}"></a>')
        elif token.type in {"html_inline", "html_block"}:
            html_parts.append(token.content)

    html_parser = _AnchorHrefParser()
    html_parser.feed("".join(html_parts))
    html_parser.close()
    return html_parser.targets
