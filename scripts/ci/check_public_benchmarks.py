#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lint committed BENCHMARK.md fixtures for structure and known leak patterns.

A clean scan means no configured pattern matched; it does not prove that a card
is safe to publish. Publication still requires the repository's broader review
and security controls.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote as url_unquote

from markdown_it import MarkdownIt

from skillevaluator.publication_text import publication_confusable_skeleton, publication_identity_present

if TYPE_CHECKING:
    from markdown_it.token import Token

REQUIRED_MARKERS = (
    "- Evaluation date:",
    "- Evaluator version:",
    "- Agents:",
    "- Tasks:",
    "- Source digest:",
    "- Dataset digest:",
    "- Tier 3 run ID:",
    "- Attempts per task:",
    "- Environment:",
    "- Tier 3 evidence:",
)

_REQUIRED_HEADINGS = (
    (1, "Skill Benchmark:", True, "# Skill Benchmark:"),
    (2, "Evaluation Metadata", False, "## Evaluation Metadata"),
    (2, "Results at a Glance", False, "## Results at a Glance"),
    (2, "Tier Status", False, "## Tier Status"),
    (2, "Freshness", False, "## Freshness"),
)

LINE_RULES = (
    (
        "retired product identity",
        re.compile(r"\b[a-z]*[\s_-]*skills[\s_-]*eval\b", flags=re.IGNORECASE),
    ),
    (
        "internal environment identity",
        re.compile(
            r"(?:^\s*-\s*Environment:\s*`?astra(?:[^A-Za-z0-9]|$)|\bastra[\s_-]+sandbox\b)",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "validation profile metadata",
        re.compile(
            r"^\s*-\s*(?:Skill\s+Evaluator\s+)?Profile\s*:",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "absolute macOS user path",
        re.compile(r"/Users/[^/\s`]+/"),
    ),
    (
        "absolute Linux home path",
        re.compile(r"/home/[^/\s`]+/"),
    ),
    (
        "absolute Windows user path",
        re.compile(r"[A-Za-z]:\\Users\\[^\\\s`]+\\"),
    ),
    (
        "legacy ambiguous uplift cell",
        re.compile(r"\b\d+%\s+\([+-]\d+%\)"),
    ),
    (
        "legacy Num score column",
        re.compile(r"\|\s*Dimension\s*\|\s*Num\s*\|", flags=re.IGNORECASE),
    ),
)

_AGENT_MODEL_STATE = re.compile(
    r"^(?P<agent>[^,]+) \((?P<model>`[^`]+`|model not recorded)\)$",
    flags=re.IGNORECASE,
)
_OVERALL_VERDICT_FIELD = re.compile(
    r"^\s*(?:(?:✅|❌|⚠\ufe0f?)\s*)?Overall verdict:\s*(?P<value>.*)$",
    flags=re.IGNORECASE,
)
_OVERALL_VERDICT_SKELETON_FIELD = re.compile(
    r"^\s*(?:(?:✅|❌|⚠\ufe0f?)\s*)?overall verdict:\s*(?P<value>.*)$",
)
_INVISIBLE_IDENTITY_CHARACTERS = frozenset(
    {
        "\u115f",
        "\u1160",
        "\u2800",
        "\u3164",
        "\uffa0",
        "\U00013441",
        "\U00013442",
        "\U0001d159",
    }
)
_SECURITY_CONFUSABLES = {"\u0406": "I", "\u0456": "i"}
_METADATA_FIELD_RULES = (
    (
        "Source digest",
        re.compile(
            r"(?:`sha256:[0-9a-f]{64}`\s+\(skill-evaluator-source-tree/2\)|not recorded\b.*)",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "Evaluation date",
        re.compile(r"(?:\d{4}-\d{2}-\d{2}|not recorded\b.*)", flags=re.IGNORECASE),
    ),
    (
        "Evaluator version",
        re.compile(r"(?:`[^`\s][^`]*`|not recorded\b.*)", flags=re.IGNORECASE),
    ),
    (
        "Tasks",
        re.compile(r"(?:[1-9]\d*\s+evaluation tasks?(?:\s+\(.*\))?|not recorded\b.*)", flags=re.IGNORECASE),
    ),
    (
        "Dataset digest",
        re.compile(r"(?:`[^`\s][^`]*`(?:\s+\([^)]*\))?|not recorded\b.*)", flags=re.IGNORECASE),
    ),
    (
        "Tier 3 run ID",
        re.compile(r"(?:`[A-Za-z0-9][A-Za-z0-9._-]{0,159}`|not recorded\b.*)"),
    ),
    (
        "Attempts per task",
        re.compile(r"(?:[1-9]\d*|not recorded\b.*)", flags=re.IGNORECASE),
    ),
    (
        "Environment",
        re.compile(r"(?:`[^`\s][^`]*`|not recorded\b.*)", flags=re.IGNORECASE),
    ),
    (
        "Tier 3 evidence",
        re.compile(r"(?:required for publication|optional by policy)"),
    ),
)
_METADATA_FIELD_NAMES = (
    *(field for field, _pattern in _METADATA_FIELD_RULES),
    "Tier 2 evidence",
    "Agents",
)
_PASS_SOURCE_METADATA_FIELD_RULES = (
    (
        "Source digest",
        re.compile(r"`sha256:[0-9a-f]{64}`\s+\(skill-evaluator-source-tree/2\)"),
    ),
)
_PASS_METADATA_FIELD_RULES = (
    ("Evaluation date", re.compile(r"\d{4}-\d{2}-\d{2}")),
    ("Evaluator version", re.compile(r"`[^`\s][^`]*`")),
    ("Tasks", re.compile(r"[1-9]\d*\s+evaluation tasks?(?:\s+\(.*\))?", flags=re.IGNORECASE)),
    (
        "Dataset digest",
        re.compile(
            r"`sha256:[0-9a-f]{64}`\s+\(skill-evaluator-dataset-snapshot/1\)",
            flags=re.IGNORECASE,
        ),
    ),
    ("Tier 3 run ID", re.compile(r"`[A-Za-z0-9][A-Za-z0-9._-]{0,159}`")),
    ("Attempts per task", re.compile(r"[1-9]\d*")),
    ("Environment", re.compile(r"`[^`\s][^`]*`")),
)
_TIER_COMPLETION_STATUSES = {
    1: frozenset({"PASSED", "PASSED WITH OBSERVATIONS"}),
    2: frozenset({"PASSED", "PASSED WITH OBSERVATIONS"}),
    3: frozenset({"PASS"}),
}
_OPTIONAL_TIER_ABSENCE_STATUSES = frozenset({"NOT RUN", "SKIPPED (ADVISORY)"})
_TIER_POLICY_VALUES = frozenset({"required for publication", "optional by policy"})
_OVERALL_VERDICT_STATUSES = frozenset({"PASS", "FAIL", "NEUTRAL", "INCOMPLETE"})
_KNOWN_TIER_STATUSES = {
    1: frozenset({"PASSED", "PASSED WITH OBSERVATIONS", "FAILED", "INCOMPLETE", "NOT RUN", "SKIPPED (ADVISORY)"}),
    2: frozenset({"PASSED", "PASSED WITH OBSERVATIONS", "FAILED", "INCOMPLETE", "NOT RUN", "SKIPPED (ADVISORY)"}),
    3: frozenset({"PASS", "FAIL", "FAILED", "NEUTRAL", "INCOMPLETE", "NOT RUN", "SKIPPED (ADVISORY)"}),
}
_MAX_BENCHMARK_BYTES = 128 * 1024
_MAX_BENCHMARK_LINE_CHARACTERS = 32 * 1024
_MARKDOWN = MarkdownIt("commonmark").enable("table")
_NON_RENDERED_HTML_TAGS = frozenset({"noscript", "script", "style", "template"})
_STRUCTURAL_HTML_HEADING_TAGS = frozenset({"h1", "h2"})
_VOID_HTML_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
)
_HTML_TEXT_SEPARATOR_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "dd",
        "details",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hgroup",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "summary",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
)
_AMBIGUOUS_HTML_RENDERING_TAGS = _NON_RENDERED_HTML_TAGS | frozenset(
    {
        "audio",
        "base",
        "bdi",
        "bdo",
        "button",
        "canvas",
        "dialog",
        "embed",
        "form",
        "iframe",
        "input",
        "link",
        "math",
        "meta",
        "meter",
        "noembed",
        "noframes",
        "object",
        "option",
        "optgroup",
        "output",
        "picture",
        "plaintext",
        "progress",
        "select",
        "source",
        "svg",
        "table",
        "textarea",
        "video",
        "xmp",
    }
)
_AMBIGUOUS_HTML_RENDERING_ATTRIBUTES = frozenset(
    {
        "align",
        "bgcolor",
        "class",
        "color",
        "dir",
        "face",
        "height",
        "hidden",
        "id",
        "inert",
        "size",
        "style",
        "width",
    }
)


def _html_element_is_visually_hidden(tag: str, attrs: list[tuple[str, str | None]]) -> bool:
    """Return whether an HTML element is removed from visual rendering."""
    if tag.casefold() in _NON_RENDERED_HTML_TAGS:
        return True
    normalized_attrs = {name.casefold(): value for name, value in attrs}
    if "hidden" in normalized_attrs:
        return True
    style = normalized_attrs.get("style")
    if style is None:
        return False
    uncommented_style = re.sub(r"/\*.*?\*/", "", style, flags=re.DOTALL)
    declarations: dict[str, tuple[str, bool]] = {}
    for declaration in uncommented_style.split(";"):
        property_name, separator, value = declaration.partition(":")
        if not separator:
            continue
        normalized_property = property_name.strip().casefold()
        if normalized_property not in {"display", "visibility"}:
            continue
        important = re.search(r"!\s*important\s*$", value, flags=re.IGNORECASE) is not None
        normalized_value = re.sub(r"!\s*important\s*$", "", value, flags=re.IGNORECASE).strip().casefold()
        previous = declarations.get(normalized_property)
        if previous is not None and previous[1] and not important:
            continue
        declarations[normalized_property] = (normalized_value, important)
    display = declarations.get("display", ("", False))[0]
    visibility = declarations.get("visibility", ("", False))[0]
    return display == "none" or visibility in {"hidden", "collapse"}


def _close_html_element_stack(stack: list[tuple[str, bool]], tag: str) -> int | None:
    """Pop one exactly nested element; ambiguous markup stays fail-closed."""
    if not stack or stack[-1][0] != tag:
        return None
    _name, is_hidden = stack.pop()
    return int(is_hidden)


class _VisibleHTMLParser(HTMLParser):
    """Collect rendered text while suppressing comments and control elements."""

    def __init__(self, *, suppress_visually_hidden: bool = True) -> None:
        super().__init__(convert_charrefs=True)
        self._suppress_visually_hidden = suppress_visually_hidden
        self._element_stack: list[tuple[str, bool]] = []
        self._hidden_depth = 0
        self._line_parts: dict[int, list[str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag.casefold()
        is_hidden = normalized_tag in _NON_RENDERED_HTML_TAGS or (
            self._suppress_visually_hidden and _html_element_is_visually_hidden(normalized_tag, attrs)
        )
        if normalized_tag not in _VOID_HTML_TAGS:
            self._element_stack.append((normalized_tag, is_hidden))
        if is_hidden:
            if normalized_tag not in _VOID_HTML_TAGS:
                self._hidden_depth += 1
            return
        if self._hidden_depth:
            return
        if normalized_tag == "br" or normalized_tag in _HTML_TEXT_SEPARATOR_TAGS:
            self._append_visible(" ")
        elif normalized_tag == "img":
            alt = next((value for name, value in attrs if name.casefold() == "alt"), None)
            if alt is not None:
                self._append_visible(f" {alt} ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # HTML slash syntax does not self-close non-void elements in browsers.
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.casefold()
        if normalized_tag == "br":
            if not self._hidden_depth:
                # HTML5 treats the otherwise-invalid </br> token as <br>.
                self._append_visible(" ")
            return
        hidden_before = self._hidden_depth
        hidden_count = _close_html_element_stack(self._element_stack, normalized_tag)
        if hidden_count is None:
            return
        self._hidden_depth = max(0, self._hidden_depth - hidden_count)
        if not hidden_before and normalized_tag in _HTML_TEXT_SEPARATOR_TAGS:
            self._append_visible(" ")

    def handle_data(self, data: str) -> None:
        if self._hidden_depth:
            return
        self._append_visible(data)

    def _append_visible(self, data: str) -> None:
        start_line, _column = self.getpos()
        for offset, part in enumerate(data.split("\n")):
            self._line_parts.setdefault(start_line + offset, []).append(part)

    def visible_lines(self) -> tuple[tuple[int, str], ...]:
        return tuple(
            (line, normalized)
            for line, parts in sorted(self._line_parts.items())
            if (normalized := " ".join("".join(parts).split()))
        )


class _HTMLAttributeParser(HTMLParser):
    """Collect decoded HTML attribute values for leak-pattern scanning."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.values: list[str] = []
        self.image_alts: list[str] = []
        self.events: list[tuple[str, str, bool, str | None, int]] = []
        self.ambiguous_rendering_lines: list[int] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag.casefold()
        self.values.extend(value for _name, value in attrs if value is not None)
        alt = next((value for name, value in attrs if name.casefold() == "alt"), None)
        line = self.getpos()[0]
        self.events.append(("start", normalized_tag, _html_element_is_visually_hidden(normalized_tag, attrs), alt, line))
        attribute_names = {name.casefold() for name, _value in attrs}
        if (
            normalized_tag in _AMBIGUOUS_HTML_RENDERING_TAGS
            or attribute_names & _AMBIGUOUS_HTML_RENDERING_ATTRIBUTES
            or any(name.startswith("on") for name in attribute_names)
        ):
            self.ambiguous_rendering_lines.append(line)
        if normalized_tag == "img":
            self.image_alts.extend(value for name, value in attrs if name.casefold() == "alt" and value is not None)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        self.events.append(("end", tag.casefold(), False, None, self.getpos()[0]))


class _RawStructuralHeadingParser(HTMLParser):
    """Locate raw HTML headings that cannot be trusted as card structure."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[int] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.casefold() in _STRUCTURAL_HTML_HEADING_TAGS:
            self.lines.append(self.getpos()[0])

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # A slash does not self-close h1/h2 in text/html, but the start still
        # creates ambiguous browser-visible structure and must be rejected.
        self.handle_starttag(tag, attrs)


class _HTMLStructureParser(HTMLParser):
    """Collect real tag events/headings without regex-parsing quoted markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.events: list[tuple[str, str]] = []
        self.headings: list[tuple[int, int, str]] = []
        self._heading_level: int | None = None
        self._heading_line = 1
        self._heading_parts: list[str] = []
        self._element_stack: list[tuple[str, bool]] = []
        self._hidden_depth = 0
        self._in_noscript = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag.casefold()
        if self._in_noscript:
            return
        self.events.append(("start", normalized_tag))
        is_hidden = _html_element_is_visually_hidden(normalized_tag, attrs)
        if normalized_tag not in _VOID_HTML_TAGS:
            self._element_stack.append((normalized_tag, is_hidden))
        if normalized_tag == "noscript":
            self._in_noscript = True
        if is_hidden:
            if normalized_tag not in _VOID_HTML_TAGS:
                self._hidden_depth += 1
            return
        if self._hidden_depth:
            return
        if normalized_tag in _STRUCTURAL_HTML_HEADING_TAGS and self._heading_level is None:
            self._heading_level = int(normalized_tag[1])
            self._heading_line = self.getpos()[0]
            self._heading_parts = []
        elif self._heading_level is not None and (
            normalized_tag == "br" or normalized_tag in _HTML_TEXT_SEPARATOR_TAGS
        ):
            self._heading_parts.append(" ")
        elif self._heading_level is not None and normalized_tag == "img":
            alt = next((value for name, value in attrs if name.casefold() == "alt"), None)
            if alt is not None:
                self._heading_parts.extend((" ", alt, " "))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # HTML slash syntax does not self-close non-void elements in browsers.
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.casefold()
        if normalized_tag == "br":
            self.events.append(("end", normalized_tag))
            if not self._in_noscript and not self._hidden_depth and self._heading_level is not None:
                self._heading_parts.append(" ")
            return
        if self._in_noscript:
            if normalized_tag != "noscript":
                return
            self._in_noscript = False
        self.events.append(("end", normalized_tag))
        hidden_before = self._hidden_depth
        hidden_count = _close_html_element_stack(self._element_stack, normalized_tag)
        if hidden_count:
            self._hidden_depth = max(0, self._hidden_depth - hidden_count)
        if self._hidden_depth:
            return
        if (
            not hidden_before
            and self._heading_level is not None
            and normalized_tag in _HTML_TEXT_SEPARATOR_TAGS
            and normalized_tag != f"h{self._heading_level}"
        ):
            self._heading_parts.append(" ")
        if self._heading_level is not None and normalized_tag == f"h{self._heading_level}":
            title = " ".join("".join(self._heading_parts).split())
            self.headings.append((self._heading_level, self._heading_line, title))
            self._heading_level = None
            self._heading_parts = []

    def handle_data(self, data: str) -> None:
        if not self._in_noscript and not self._hidden_depth and self._heading_level is not None:
            self._heading_parts.append(data)


@lru_cache(maxsize=256)
def _html_structure(
    content: str,
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[tuple[int, int, str], ...],
]:
    parser = _HTMLStructureParser()
    try:
        parser.feed(content)
        parser.close()
    except (AssertionError, ValueError):
        return (), ()
    return tuple(parser.events), tuple(parser.headings)


@dataclass(frozen=True)
class Offender:
    path: Path
    line: int
    reason: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.reason}"


def benchmark_files(roots: list[Path]) -> list[Path]:
    """Return unique BENCHMARK.md files below the requested roots."""
    found: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"input path does not exist: {resolved}")
        if resolved.is_file() and resolved.name == "BENCHMARK.md":
            found.add(resolved)
        elif resolved.is_dir():
            found.update(path.resolve() for path in resolved.rglob("BENCHMARK.md") if path.is_file())
    return sorted(found)


def scan_file(path: Path) -> list[Offender]:
    try:
        if path.stat().st_size > _MAX_BENCHMARK_BYTES:
            return [Offender(path, 1, f"BENCHMARK.md exceeds {_MAX_BENCHMARK_BYTES:,} bytes")]
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return [Offender(path, 1, f"unreadable file ({type(error).__name__})")]

    offenders: list[Offender] = []
    for line_number, line in enumerate(text.splitlines(keepends=True), 1):
        # Character references can render the same bidirectional and format
        # controls as literal Unicode. Reject both representations before any
        # semantic normalization removes them.
        encoded_surface = unescape(line.replace("\n", "").replace("\r", "").replace("\t", ""))
        if _contains_disallowed_category_c(line, allow_layout_whitespace=True) or _contains_disallowed_category_c(
            encoded_surface,
            allow_layout_whitespace=False,
        ):
            offenders.append(Offender(path, line_number, "benchmark contains disallowed control character"))
            break

    for line_number, line in enumerate(text.splitlines(), 1):
        if len(line) > _MAX_BENCHMARK_LINE_CHARACTERS:
            return [
                Offender(
                    path,
                    line_number,
                    f"benchmark line exceeds {_MAX_BENCHMARK_LINE_CHARACTERS:,} characters",
                )
            ]

    for marker in REQUIRED_MARKERS:
        if marker not in text:
            offenders.append(Offender(path, 1, f"missing required section: {marker}"))

    literal_code_lines = {
        line_number
        for token in _markdown_tokens(text)
        if token.type in {"fence", "code_block"} and token.map is not None
        for line_number in range(token.map[0] + 1, token.map[1] + 1)
    }
    for line_number, line in enumerate(text.splitlines(), 1):
        # Decode references in rendered text while preserving literal bytes in
        # code spans, matching CommonMark rather than raw HTML unescaping.
        rendered_surfaces = () if line_number in literal_code_lines else _rendered_inline_surfaces(line)
        semantic_lines = tuple(_semantic_text(surface) for surface in (line, *rendered_surfaces))
        for reason, pattern in LINE_RULES:
            candidate_lines = semantic_lines
            if reason in {"retired product identity", "internal environment identity"}:
                candidate_lines = (
                    *candidate_lines,
                    *(publication_confusable_skeleton(semantic_line) for semantic_line in semantic_lines),
                )
            if reason == "internal environment identity":
                for semantic_line in semantic_lines:
                    environment_field = re.match(
                        r"^\s*-\s*Environment:\s*(?P<value>.*)$",
                        semantic_line,
                        flags=re.IGNORECASE,
                    )
                    if environment_field is not None:
                        candidate_lines = (
                            *candidate_lines,
                            "- Environment: "
                            + publication_confusable_skeleton(environment_field.group("value")),
                        )
            if any(pattern.search(candidate_line) for candidate_line in candidate_lines):
                offenders.append(Offender(path, line_number, reason))

    semantic_document = _semantic_text(text)
    for reason, pattern in LINE_RULES:
        if reason not in {"retired product identity", "internal environment identity"}:
            continue
        if (
            pattern.search(semantic_document)
            or pattern.search(publication_confusable_skeleton(semantic_document))
        ) and not any(offender.reason == reason for offender in offenders):
            offenders.append(Offender(path, 1, reason))

    _check_required_headings(path, text, offenders)
    _check_metadata_semantics(path, text, offenders)
    _check_verdict_tier_consistency(path, text, offenders)
    _check_untrusted_structural_headings(path, text, offenders)
    _check_html_rendering_safety(path, text, offenders)
    return offenders


def _check_required_headings(path: Path, text: str, offenders: list[Offender]) -> None:
    """Require canonical headings in rendered Markdown structure."""
    headings = _heading_entries(text)
    for _index, level, line, title, _trusted in headings:
        structural_title = title.partition(":")[0] + ":" if level == 1 and ":" in title else title
        if level in {1, 2} and not structural_title.isascii():
            offenders.append(Offender(path, line, "non-ASCII structural heading"))
    for required_level, required_title, is_prefix, marker in _REQUIRED_HEADINGS:
        required_visual_key = _visual_text_key(required_title)
        required_canonical_key = _canonical_text_key(required_title)
        matching_headings = [
            (line, title, trusted)
            for _index, level, line, title, trusted in headings
            if level == required_level
            and (
                _visual_text_key(title).startswith(required_visual_key)
                if is_prefix
                else _visual_text_key(title) == required_visual_key
            )
        ]
        trusted_headings = [
            heading
            for heading in matching_headings
            if heading[2]
            and (
                _canonical_text_key(heading[1]).startswith(required_canonical_key)
                if is_prefix
                else _canonical_text_key(heading[1]) == required_canonical_key
            )
        ]
        trusted_prefix_identity = (
            trusted_headings[0][1][len(required_title) :]
            if trusted_headings and trusted_headings[0][1].casefold().startswith(required_title.casefold())
            else ""
        )
        if not trusted_headings or (is_prefix and not _identity_present(trusted_prefix_identity)):
            offenders.append(Offender(path, 1, f"missing required section: {marker}"))
        if len(matching_headings) > 1 and required_title not in {"Evaluation Metadata", "Tier Status"}:
            offenders.append(Offender(path, matching_headings[1][0], f"duplicate required section: {marker}"))


def _check_untrusted_structural_headings(path: Path, text: str, offenders: list[Offender]) -> None:
    """Require root h1/h2 structure to use unambiguous plain Markdown text."""
    duplicate_lines = {offender.line for offender in offenders if offender.reason.startswith("duplicate ")}
    for line_number in _untrusted_structural_heading_lines(text):
        if line_number not in duplicate_lines:
            offenders.append(Offender(path, line_number, "ambiguous structural heading markup"))


def _check_html_rendering_safety(path: Path, text: str, offenders: list[Offender]) -> None:
    """Reject raw HTML whose browser presentation cannot be parsed safely."""
    stack: list[tuple[str, int]] = []
    ambiguous_lines: set[int] = set()

    def inspect_fragment(content: str, base_line: int) -> None:
        parser = _HTMLAttributeParser()
        try:
            parser.feed(content)
            parser.close()
        except (AssertionError, ValueError):
            ambiguous_lines.add(base_line)
            return
        ambiguous_lines.update(base_line + relative_line - 1 for relative_line in parser.ambiguous_rendering_lines)
        for event, tag, _visually_hidden, _alt, relative_line in parser.events:
            line = base_line + relative_line - 1
            if event == "start":
                if tag not in _VOID_HTML_TAGS:
                    stack.append((tag, line))
                continue
            if tag == "br":
                # HTML5 defines </br> as a line break rather than a close.
                continue
            if not stack or stack[-1][0] != tag:
                ambiguous_lines.add(line)
                continue
            stack.pop()

    for token in _markdown_tokens(text):
        if token.map is None:
            continue
        if token.type == "html_block":
            inspect_fragment(token.content, token.map[0] + 1)
        elif token.type == "inline":
            for child in token.children or []:
                if child.type == "html_inline":
                    inspect_fragment(child.content, token.map[0] + 1)

    ambiguous_lines.update(line for _tag, line in stack)
    for line in sorted(ambiguous_lines):
        offenders.append(Offender(path, line, "ambiguous HTML rendering semantics"))


def _check_metadata_semantics(path: Path, text: str, offenders: list[Offender]) -> None:
    metadata_heading_lines = _section_heading_lines(text, "Evaluation Metadata")
    metadata_sections = _section_occurrences(text, "Evaluation Metadata")
    if len(metadata_heading_lines) > 1:
        offenders.append(Offender(path, metadata_heading_lines[1], "duplicate Evaluation Metadata section"))
    if not metadata_sections:
        return
    fallback_line, section_tokens = metadata_sections[0]
    metadata_lines = _metadata_list_items(section_tokens)

    for line_number, source in metadata_lines:
        rendered = _rendered_inline_fragment(source)
        rendered_match = re.match(r"^\s*-\s*(?P<label>[^:]+?)\s*:", rendered or "")
        if rendered_match is not None and not rendered_match.group("label").isascii():
            offenders.append(Offender(path, line_number, "non-ASCII metadata field label"))

    for line_number, field in _confusable_metadata_field_aliases(metadata_lines):
        offenders.append(Offender(path, line_number, f"confusable metadata field: - {field}:"))

    for field, pattern in _METADATA_FIELD_RULES:
        matches = _metadata_field_matches(metadata_lines, field)
        marker = f"- {field}:"
        if not matches:
            offenders.append(Offender(path, fallback_line, f"missing metadata field: {marker}"))
            continue
        line_number, value = matches[0]
        valid_value = pattern.fullmatch(value) is not None
        if valid_value and not value.lower().startswith("not recorded"):
            if field == "Evaluation date":
                valid_value = _valid_evaluation_date(value)
            elif field in {"Evaluator version", "Environment"}:
                valid_value = _backticked_identity_present(value)
        if len(matches) > 1 or not valid_value:
            offenders.append(Offender(path, line_number, f"invalid metadata field: {marker}"))

    tier2_matches = _metadata_field_matches(metadata_lines, "Tier 2 evidence")
    if tier2_matches:
        line_number, value = tier2_matches[0]
        if len(tier2_matches) > 1 or value not in _TIER_POLICY_VALUES:
            offenders.append(Offender(path, line_number, "invalid metadata field: - Tier 2 evidence:"))

    agent_matches = _metadata_field_matches(metadata_lines, "Agents")
    if not agent_matches:
        offenders.append(Offender(path, fallback_line, "missing metadata field: - Agents:"))
        return
    line_number, value = agent_matches[0]
    lowered = value.casefold()
    if len(agent_matches) > 1:
        offenders.append(Offender(path, line_number, "agent model identity not recorded"))
    elif lowered == "not recorded (legacy or non-live result)":
        return
    elif lowered.startswith("requested but not run — "):
        requested_states = value[len("requested but not run — ") :].strip()
        if not requested_states or not _valid_unrecorded_agent_models(requested_states):
            offenders.append(Offender(path, line_number, "agent model identity not recorded"))
    elif not _valid_agent_model_states(value):
        offenders.append(Offender(path, line_number, "agent model identity not recorded"))


def _valid_agent_model_states(value: str) -> bool:
    """Validate a comma-delimited agent list without a backtracking list regex."""
    agents = _split_agent_model_states(value)
    if not agents:
        return False
    seen_agents: set[str] = set()
    for agent in agents:
        match = _AGENT_MODEL_STATE.fullmatch(agent)
        rendered_agent = _rendered_identity(match.group("agent")) if match is not None else None
        if rendered_agent is None:
            return False
        normalized_agent = _semantic_text(rendered_agent).casefold()
        if normalized_agent in seen_agents:
            return False
        seen_agents.add(normalized_agent)
        model = match.group("model")
        if model.casefold() != "model not recorded" and not _identity_present(model[1:-1]):
            return False
    return True


def _valid_unrecorded_agent_models(value: str) -> bool:
    agents = _split_agent_model_states(value)
    if not agents:
        return False
    seen_agents: set[str] = set()
    for agent in agents:
        match = _AGENT_MODEL_STATE.fullmatch(agent)
        rendered_agent = _rendered_identity(match.group("agent")) if match is not None else None
        if match is None or rendered_agent is None or match.group("model").casefold() != "model not recorded":
            return False
        normalized_agent = _semantic_text(rendered_agent).casefold()
        if normalized_agent in seen_agents:
            return False
        seen_agents.add(normalized_agent)
    return True


def _split_agent_model_states(value: str) -> list[str]:
    """Split agent records on commas outside model code spans."""
    agents: list[str] = []
    start = 0
    in_code = False
    for index, character in enumerate(value):
        if character == "`":
            in_code = not in_code
        elif character == "," and not in_code:
            agents.append(value[start:index].strip())
            start = index + 1
    agents.append(value[start:].strip())
    return agents


def _metadata_section_lines(text: str) -> list[tuple[int, str]]:
    """Return canonical list items from the first Evaluation Metadata section."""
    sections = _section_occurrences(text, "Evaluation Metadata")
    return _metadata_list_items(sections[0][1]) if sections else []


def _metadata_field_matches(
    metadata_lines: list[tuple[int, str]],
    field: str,
) -> list[tuple[int, str]]:
    pattern = re.compile(rf"^\s*-\s*{re.escape(field)}:\s*(?P<value>.*)$", flags=re.IGNORECASE)
    return [
        (line_number, match.group("value").strip())
        for line_number, line in metadata_lines
        if (match := pattern.fullmatch(line))
    ]


def _confusable_metadata_field_aliases(
    metadata_lines: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    """Return visually canonical metadata labels that are not canonical text."""
    canonical_by_visual_key = {_visual_text_key(field): field for field in _METADATA_FIELD_NAMES}
    aliases: list[tuple[int, str]] = []
    for line_number, source in metadata_lines:
        rendered = _rendered_inline_fragment(source)
        if rendered is None:
            continue
        rendered_match = re.match(r"^\s*-\s*(?P<label>[^:]+?)\s*:", rendered)
        if rendered_match is None:
            continue
        canonical = canonical_by_visual_key.get(_visual_text_key(rendered_match.group("label")))
        if canonical is None:
            continue
        source_match = re.match(r"^\s*-\s*(?P<label>[^:]+?)\s*:", source)
        if source_match is None or _canonical_text_key(source_match.group("label")) != _canonical_text_key(canonical):
            aliases.append((line_number, canonical))
    return aliases


def _metadata_field_value(text: str, field: str) -> str | None:
    matches = _metadata_field_matches(_metadata_section_lines(text), field)
    return matches[0][1] if len(matches) == 1 else None


def _overall_verdict_fields(text: str) -> list[tuple[int, str, str]]:
    """Return canonical and visibly callout-shaped verdict fields."""
    tokens = _markdown_tokens(text)
    fields: list[tuple[int, str, str]] = []
    seen_fields: set[tuple[int, str]] = set()

    def add_field(line_number: int, line: str, *, trusted: bool) -> None:
        semantic_line = _semantic_text(line)
        match = _OVERALL_VERDICT_FIELD.fullmatch(semantic_line)
        matched_confusable = False
        if match is None:
            skeleton_line = publication_confusable_skeleton(semantic_line)
            match = _OVERALL_VERDICT_SKELETON_FIELD.fullmatch(skeleton_line)
            if match is None:
                return
            matched_confusable = True
        value = match.group("value").strip()
        value_skeleton = publication_confusable_skeleton(value)
        methodology_prefixes = tuple(
            publication_confusable_skeleton(prefix)
            for prefix in ("derived from", "pass only when every configured dimension passes")
        )
        if value_skeleton.startswith(methodology_prefixes):
            # This is the generated methodology definition, not a decision
            # field. Exact late PASS/FAIL/NEUTRAL/INCOMPLETE fields are still
            # counted so a second visible verdict cannot hide after metadata.
            return
        field_key = (line_number, publication_confusable_skeleton(semantic_line))
        if field_key in seen_fields:
            return
        seen_fields.add(field_key)
        status_match = re.match(r"(?P<status>[A-Za-z]+)\b", value)
        fields.append(
            (
                line_number,
                status_match.group("status").upper() if trusted and not matched_confusable and status_match else "",
                value,
            )
        )

    raw_container_stack: list[str] = []
    for token in tokens:
        if token.type == "html_block" and token.map is not None:
            for suppress_visually_hidden in (True, False):
                for relative_line, line in _visible_html_lines(
                    token.content,
                    suppress_visually_hidden=suppress_visually_hidden,
                ):
                    add_field(token.map[0] + relative_line, line, trusted=False)
            _update_raw_html_stack(token.content, raw_container_stack)
            continue
        if token.type != "inline" or token.map is None:
            continue
        unsafe_markup = bool(raw_container_stack) or any(
            child.type in {"html_inline", "link_open", "image"} for child in token.children or []
        )
        for suppress_visually_hidden in (True, False) if unsafe_markup else (True,):
            for offset, line in enumerate(
                _inline_visible_lines(
                    token,
                    include_code=False,
                    suppress_visually_hidden=suppress_visually_hidden,
                )
            ):
                add_field(token.map[0] + offset + 1, line, trusted=not unsafe_markup)
    return fields


def _section_occurrences(text: str, title: str) -> list[tuple[int, tuple[Token, ...]]]:
    """Return structurally parsed level-two sections and their token bodies."""
    tokens = _markdown_tokens(text)
    headings = _heading_entries(text)
    sections: list[tuple[int, tuple[Token, ...]]] = []
    for token_index, level, start_line, heading_title, trusted in headings:
        if not trusted or level != 2 or _canonical_text_key(heading_title) != _canonical_text_key(title):
            continue
        end_index = len(tokens)
        # Even an untrusted linked/raw-markup heading ends the current section.
        # Such a heading cannot establish a required section itself, but its
        # body must not be credited to the preceding trusted section.
        for next_index in range(token_index + 3, len(tokens)):
            candidate = tokens[next_index]
            if (
                candidate.type == "heading_open" and candidate.level == 0 and int(candidate.tag.removeprefix("h")) <= 2
            ) or (
                candidate.type == "html_block" and candidate.level == 0 and bool(_html_structure(candidate.content)[1])
            ):
                end_index = next_index
                break
        sections.append((start_line, tokens[token_index + 3 : end_index]))
    return sections


def _section_heading_lines(text: str, title: str) -> list[int]:
    """Return every matching level-two heading, including untrusted variants."""
    title_key = _visual_text_key(title)
    return [
        start_line
        for _token_index, level, start_line, heading_title, _trusted in _heading_entries(text)
        if level == 2 and _visual_text_key(heading_title) == title_key
    ]


@lru_cache(maxsize=128)
def _markdown_tokens(text: str) -> tuple[Token, ...]:
    """Parse Markdown once so structural checks use rendered block semantics."""
    return tuple(_MARKDOWN.parse(text))


@lru_cache(maxsize=128)
def _heading_entries(text: str) -> tuple[tuple[int, int, int, str, bool], ...]:
    """Return token index, level, line, title, and structural trust."""
    tokens = _markdown_tokens(text)
    headings: list[tuple[int, int, int, str, bool]] = []
    raw_container_stack: list[str] = []
    for index, token in enumerate(tokens):
        if token.type == "html_block":
            if token.map is not None:
                headings.extend(
                    (index, level, token.map[0] + relative_line, title, False)
                    for level, relative_line, title in _html_structure(token.content)[1]
                )
            _update_raw_html_stack(token.content, raw_container_stack)
            continue
        if raw_container_stack:
            continue
        if token.type != "heading_open" or token.level != 0 or token.map is None:
            continue
        inline = tokens[index + 1] if index + 1 < len(tokens) else None
        if inline is None or inline.type != "inline":
            continue
        trusted = _is_plain_markdown_heading(inline)
        title = " ".join(_inline_visible_lines(inline, include_code=True)).strip()
        headings.append((index, int(token.tag.removeprefix("h")), token.map[0] + 1, title, trusted))
    return tuple(headings)


@lru_cache(maxsize=256)
def _raw_structural_heading_relative_lines(content: str) -> tuple[int, ...]:
    """Return line numbers for every real raw-HTML h1/h2 start tag."""
    parser = _RawStructuralHeadingParser()
    try:
        parser.feed(content)
        parser.close()
    except (AssertionError, ValueError):
        # Malformed HTML is untrusted, but without a parsed start tag there is
        # no reliable structural-heading location to report here.
        return ()
    return tuple(parser.lines)


@lru_cache(maxsize=128)
def _untrusted_structural_heading_lines(text: str) -> tuple[int, ...]:
    """Return root heading lines whose rendered structure is ambiguous."""
    lines: set[int] = set()
    raw_container_stack: list[str] = []
    tokens = _markdown_tokens(text)
    for index, token in enumerate(tokens):
        if token.map is None:
            continue
        if token.type == "html_block":
            lines.update(
                token.map[0] + relative_line for relative_line in _raw_structural_heading_relative_lines(token.content)
            )
            _update_raw_html_stack(token.content, raw_container_stack)
            continue
        if token.type == "heading_open" and token.tag in _STRUCTURAL_HTML_HEADING_TAGS:
            inline = tokens[index + 1] if index + 1 < len(tokens) else None
            if (
                token.level != 0
                or raw_container_stack
                or inline is None
                or inline.type != "inline"
                or not _is_plain_markdown_heading(inline)
            ):
                lines.add(token.map[0] + 1)
            continue
        if token.type != "inline":
            continue
        for child in token.children or []:
            if child.type == "html_inline":
                lines.update(
                    token.map[0] + relative_line
                    for relative_line in _raw_structural_heading_relative_lines(child.content)
                )
    return tuple(sorted(lines))


def _is_plain_markdown_heading(inline: Token) -> bool:
    """Return whether a heading contains only undecorated rendered text."""
    return bool(inline.children) and all(child.type == "text" for child in inline.children)


def _inline_visible_lines(
    token: Token,
    *,
    include_code: bool,
    suppress_visually_hidden: bool = True,
) -> list[str]:
    """Flatten inline Markdown into visible lines while excluding inert markup."""
    return _inline_children_visible_lines(
        token.children or [],
        include_code=include_code,
        suppress_visually_hidden=suppress_visually_hidden,
    )


def _inline_children_visible_lines(
    children: list[Token],
    *,
    include_code: bool,
    suppress_visually_hidden: bool,
) -> list[str]:
    """Flatten inline child tokens, including recursively rendered image alt text."""
    lines = [""]
    element_stack: list[tuple[str, bool]] = []
    hidden_html_depth = 0

    def append_text(content: str) -> None:
        parts = content.split("\n")
        lines[-1] += parts[0]
        lines.extend(parts[1:])

    for child in children:
        if child.type == "html_inline":
            attribute_parser = _HTMLAttributeParser()
            try:
                attribute_parser.feed(child.content)
                attribute_parser.close()
            except (AssertionError, ValueError):
                continue
            for event, tag, visually_hidden, alt, _line in attribute_parser.events:
                if event == "end":
                    if tag == "br":
                        if not hidden_html_depth:
                            append_text(" ")
                        continue
                    hidden_before = hidden_html_depth
                    hidden_count = _close_html_element_stack(element_stack, tag)
                    if hidden_count:
                        hidden_html_depth = max(0, hidden_html_depth - hidden_count)
                    if not hidden_before and tag in _HTML_TEXT_SEPARATOR_TAGS:
                        append_text(" ")
                    continue
                is_hidden = tag in _NON_RENDERED_HTML_TAGS or (suppress_visually_hidden and visually_hidden)
                if tag not in _VOID_HTML_TAGS:
                    element_stack.append((tag, is_hidden))
                if is_hidden:
                    if tag not in _VOID_HTML_TAGS:
                        hidden_html_depth += 1
                elif not hidden_html_depth and (tag == "br" or tag in _HTML_TEXT_SEPARATOR_TAGS):
                    append_text(" ")
                elif not hidden_html_depth and tag == "img" and alt is not None:
                    append_text(f" {alt} ")
        elif hidden_html_depth:
            continue
        elif child.type in {"text", "text_special"}:
            append_text(child.content)
        elif child.type == "code_inline":
            append_text(child.content if include_code else " ")
        elif child.type == "image":
            alt_text = (
                " ".join(
                    _inline_children_visible_lines(
                        child.children,
                        include_code=True,
                        suppress_visually_hidden=suppress_visually_hidden,
                    )
                )
                if child.children is not None
                else unescape(child.content)
            )
            append_text(f" {alt_text} ")
        elif child.type in {"softbreak", "hardbreak"}:
            lines.append("")
    return [" ".join(line.split()) for line in lines]


def _rendered_inline_surfaces(value: str) -> tuple[str, ...]:
    """Return visible text and decoded link/HTML attribute surfaces."""
    tokens = _MARKDOWN.parseInline(value)
    if len(tokens) != 1 or tokens[0].type != "inline":
        return ()
    inline = tokens[0]
    surfaces = [" ".join(_inline_visible_lines(inline, include_code=True))]
    for child in inline.children or []:
        if child.type in {"link_open", "image"}:
            for attribute in ("href", "title") if child.type == "link_open" else ("src", "title"):
                if destination := child.attrGet(attribute):
                    surfaces.extend((destination, url_unquote(destination)))
        elif child.type == "html_inline":
            parser = _HTMLAttributeParser()
            try:
                parser.feed(child.content)
                parser.close()
            except (AssertionError, ValueError):
                continue
            for attribute_value in parser.values:
                surfaces.extend((attribute_value, url_unquote(attribute_value)))
    return tuple(surfaces)


def _rendered_inline_fragment(value: str) -> str | None:
    """Return visible inline text while keeping code-span contents literal."""
    surfaces = _rendered_inline_surfaces(value)
    return surfaces[0] if surfaces else None


def _rendered_identity(value: str) -> str | None:
    tokens = _MARKDOWN.parseInline(value)
    if len(tokens) != 1 or tokens[0].type != "inline":
        return None
    inline = tokens[0]
    if any(child.type in {"html_inline", "link_open", "image", "code_inline"} for child in inline.children or []):
        return None
    rendered = " ".join(_inline_visible_lines(inline, include_code=False))
    return rendered if _identity_present(rendered) else None


def _rendered_identity_present(value: str) -> bool:
    return _rendered_identity(value) is not None


def _update_raw_html_stack(content: str, stack: list[str]) -> None:
    """Track Markdown nested inside raw HTML containers across block tokens."""
    events, _headings = _html_structure(content)
    for event, tag in events:
        if "noscript" in stack:
            if event == "end" and stack[-1] == tag == "noscript":
                stack.pop()
            continue
        if event == "end":
            # Trust only properly nested closures. A separate safety check
            # rejects ambiguous markup rather than guessing HTML5 insertion
            # modes that differ across raw-text/select/table contexts.
            if stack and stack[-1] == tag:
                stack.pop()
            continue
        # In text/html, a trailing slash does not self-close non-void elements
        # such as <details/> or <div/>. Browsers keep those containers open.
        if tag in _VOID_HTML_TAGS:
            continue
        stack.append(tag)


@lru_cache(maxsize=256)
def _visible_html_lines(
    content: str,
    *,
    suppress_visually_hidden: bool = True,
) -> tuple[tuple[int, str], ...]:
    """Return line-relative text that a raw HTML block would visibly render."""
    parser = _VisibleHTMLParser(suppress_visually_hidden=suppress_visually_hidden)
    try:
        parser.feed(content)
        parser.close()
    except (AssertionError, ValueError):
        # Malformed raw HTML is not trustworthy visible evidence.
        return ()
    return parser.visible_lines()


def _metadata_list_items(section_tokens: tuple[Token, ...]) -> list[tuple[int, str]]:
    """Return one-line items from top-level unordered metadata lists."""
    items: list[tuple[int, str]] = []
    in_root_bullet_list = False
    raw_container_stack: list[str] = []
    for index, token in enumerate(section_tokens):
        if token.type == "html_block":
            _update_raw_html_stack(token.content, raw_container_stack)
            continue
        if raw_container_stack:
            continue
        if token.type == "bullet_list_open" and token.level == 0:
            in_root_bullet_list = True
            continue
        if token.type == "bullet_list_close" and token.level == 0:
            in_root_bullet_list = False
            continue
        if not in_root_bullet_list or token.type != "list_item_open" or token.level != 1:
            continue

        item_end = next(
            (
                candidate
                for candidate in range(index + 1, len(section_tokens))
                if section_tokens[candidate].type == "list_item_close"
                and section_tokens[candidate].level == token.level
            ),
            len(section_tokens),
        )
        inline_tokens = [
            candidate
            for candidate in section_tokens[index + 1 : item_end]
            if candidate.type == "inline" and candidate.level == token.level + 2 and candidate.map is not None
        ]
        if len(inline_tokens) != 1:
            continue
        inline = inline_tokens[0]
        if inline.map is None or inline.map[1] != inline.map[0] + 1:
            continue
        if any(child.type in {"html_inline", "link_open", "image"} for child in inline.children or []):
            continue
        # Preserve encoded commas because literal commas delimit agents. Each
        # captured identity is rendered and decoded before it counts as proof.
        items.append((inline.map[0] + 1, f"- {inline.content}"))
    return items


def _tier_status_rows(section_tokens: tuple[Token, ...], tier: int) -> list[tuple[int, str]]:
    """Return Tier rows from rendered root-level Markdown tables."""
    rows: list[tuple[int, str]] = []
    in_root_table = False
    in_header = False
    in_body = False
    row_line = 1
    cells: list[str] | None = None
    cell_parts: list[str] | None = None
    tier_column: int | None = None
    status_column: int | None = None
    header_is_canonical = False
    cell_has_unsafe_markup = False
    cell_safety: list[bool] | None = None
    raw_container_stack: list[str] = []

    for token in section_tokens:
        if token.type == "html_block":
            _update_raw_html_stack(token.content, raw_container_stack)
            continue
        if raw_container_stack:
            continue
        if token.type == "table_open" and token.level == 0:
            in_root_table = True
            tier_column = None
            status_column = None
            header_is_canonical = False
            continue
        if token.type == "table_close" and token.level == 0:
            in_root_table = False
            in_header = False
            in_body = False
            continue
        if not in_root_table:
            continue
        if token.type == "thead_open":
            in_header = True
            continue
        if token.type == "thead_close":
            in_header = False
            continue
        if token.type == "tbody_open":
            in_body = True
            continue
        if token.type == "tbody_close":
            in_body = False
            continue
        if not (in_header or in_body):
            continue
        if token.type == "tr_open":
            row_line = token.map[0] + 1 if token.map is not None else 1
            cells = []
            cell_safety = []
            continue
        if token.type in {"th_open", "td_open"} and cells is not None:
            cell_parts = []
            cell_has_unsafe_markup = False
            continue
        if token.type == "inline" and cell_parts is not None:
            cell_parts.extend(_inline_visible_lines(token, include_code=True))
            cell_has_unsafe_markup = cell_has_unsafe_markup or any(
                child.type in {"html_inline", "link_open", "image"} for child in token.children or []
            )
            continue
        if token.type in {"th_close", "td_close"} and cells is not None and cell_parts is not None:
            cells.append(" ".join(cell_parts).strip())
            assert cell_safety is not None
            cell_safety.append(not cell_has_unsafe_markup)
            cell_parts = None
            continue
        if token.type != "tr_close" or cells is None:
            continue

        if in_header:
            normalized_headers = [" ".join(cell.split()).casefold() for cell in cells]
            tier_indexes = [
                index for index, header in enumerate(normalized_headers) if _visual_text_key(header) == "tier"
            ]
            status_indexes = [
                index for index, header in enumerate(normalized_headers) if _visual_text_key(header) == "status"
            ]
            if len(tier_indexes) == len(status_indexes) == 1:
                tier_column = tier_indexes[0]
                status_column = status_indexes[0]
                assert cell_safety is not None
                header_is_canonical = bool(
                    cell_safety[tier_column]
                    and cell_safety[status_column]
                    and normalized_headers[tier_column] == "tier"
                    and normalized_headers[status_column] == "status"
                )
        elif in_body and tier_column is not None and status_column is not None:
            if max(tier_column, status_column) < len(cells):
                tier_label = " ".join(cells[tier_column].split())
                if _visual_text_key(tier_label) == _visual_text_key(f"Tier {tier}"):
                    assert cell_safety is not None
                    status = (
                        " ".join(cells[status_column].split()).upper()
                        if header_is_canonical
                        and cell_safety[tier_column]
                        and cell_safety[status_column]
                        and re.fullmatch(rf"Tier\s*{tier}", tier_label, flags=re.IGNORECASE)
                        else ""
                    )
                    rows.append((row_line, status))
        cells = None
        cell_safety = None
        cell_parts = None
    return rows


def _non_ascii_tier_identifier_lines(section_tokens: tuple[Token, ...]) -> list[int]:
    """Return rows whose Tier Status header or tier label is not ASCII.

    Generated cards use fixed ASCII identifiers. Restricting only those
    structural cells prevents Unicode lookalikes outside the deliberately
    small identity-confusables table from creating a second visible table or
    result row while leaving the canonical proof intact.
    """
    lines: list[int] = []
    in_root_table = False
    in_header = False
    in_body = False
    row_line = 1
    cells: list[str] | None = None
    cell_parts: list[str] | None = None
    raw_container_stack: list[str] = []

    for token in section_tokens:
        if token.type == "html_block":
            _update_raw_html_stack(token.content, raw_container_stack)
            continue
        if raw_container_stack:
            continue
        if token.type == "table_open" and token.level == 0:
            in_root_table = True
            continue
        if token.type == "table_close" and token.level == 0:
            in_root_table = False
            in_header = False
            in_body = False
            continue
        if not in_root_table:
            continue
        if token.type == "thead_open":
            in_header = True
            continue
        if token.type == "thead_close":
            in_header = False
            continue
        if token.type == "tbody_open":
            in_body = True
            continue
        if token.type == "tbody_close":
            in_body = False
            continue
        if token.type == "tr_open":
            row_line = token.map[0] + 1 if token.map is not None else 1
            cells = []
            continue
        if token.type in {"th_open", "td_open"} and cells is not None:
            cell_parts = []
            continue
        if token.type == "inline" and cell_parts is not None:
            cell_parts.extend(_inline_visible_lines(token, include_code=True))
            continue
        if token.type in {"th_close", "td_close"} and cells is not None and cell_parts is not None:
            cells.append(" ".join(cell_parts).strip())
            cell_parts = None
            continue
        if token.type != "tr_close" or cells is None:
            continue
        identifiers = cells if in_header else cells[:1] if in_body else []
        if any(not identifier.isascii() for identifier in identifiers):
            lines.append(row_line)
        cells = None
        cell_parts = None
    return lines


_DECISION_ALIAS_SENTINEL = "\x00"


@lru_cache(maxsize=8)
def _decision_alias_pattern(canonical_text: str) -> re.Pattern[str]:
    """Match one decision phrase while failing closed on unmapped lookalikes."""
    canonical = publication_confusable_skeleton(_semantic_text(canonical_text))
    parts: list[str] = []
    for character in canonical:
        if character.isspace():
            parts.append(r"\s+")
        elif character.isascii() and character.isalnum():
            parts.append(f"(?:{re.escape(character)}|{re.escape(_DECISION_ALIAS_SENTINEL)})")
        else:
            parts.append(re.escape(character))
    return re.compile("".join(parts))


def _has_decision_text_alias(value: str, canonical_text: str) -> bool:
    """Return whether visible text contains a canonical or confusable decision phrase.

    The shared publication skeleton is intentionally limited to the alphabet
    needed by reserved identity placeholders. Decision prose has a wider
    alphabet, so any non-ASCII character left unmapped by that skeleton is a
    one-character wildcard for an ASCII letter in the protected phrase. This
    is deliberately fail-closed and applies only to known decision phrases.
    """
    skeleton = publication_confusable_skeleton(_semantic_text(value))
    surface = "".join(
        " " if character.isspace() else character if character.isascii() else _DECISION_ALIAS_SENTINEL
        for character in skeleton
    )
    return _decision_alias_pattern(canonical_text).search(surface) is not None


def _has_publication_recommendation(text: str) -> bool:
    """Return whether rendered content recommends publication."""
    for token in _markdown_tokens(text):
        if token.type == "inline":
            visible = " ".join(_inline_visible_lines(token, include_code=False))
            if _has_decision_text_alias(visible, "recommended for publication"):
                return True
        elif token.type == "html_block":
            visible = " ".join(line for _offset, line in _visible_html_lines(token.content))
            if _has_decision_text_alias(visible, "recommended for publication"):
                return True
    return False


def _semantic_text(value: str) -> str:
    """Normalize compatibility characters and remove invisible spoofing characters."""
    normalized = unicodedata.normalize("NFKD", value)

    def canonical_character(character: str) -> str:
        if unicodedata.category(character) == "Pd" or character in {"\u2043", "\u2212"}:
            return "-"
        if character in {"\u2044", "\u2215", "\u29f8"}:
            return "/"
        if character in {"\u2216", "\u29f5"}:
            return "\\"
        return _SECURITY_CONFUSABLES.get(character, character)

    return "".join(
        canonical_character(character)
        for character in normalized
        if unicodedata.category(character)[0] not in {"C", "M"} and character not in _INVISIBLE_IDENTITY_CHARACTERS
    )


def _canonical_text_key(value: str) -> str:
    """Normalize layout and case without accepting lookalike characters."""
    return " ".join(value.split()).casefold()


def _visual_text_key(value: str) -> str:
    """Return a confusable-aware key used only to reject visual aliases."""
    semantic = " ".join(_semantic_text(value).split())
    return publication_confusable_skeleton(semantic)


def _contains_disallowed_category_c(value: str, *, allow_layout_whitespace: bool) -> bool:
    """Return whether text contains a disallowed control or line separator."""
    return any(
        (not allow_layout_whitespace or character not in {"\n", "\r", "\t"})
        and (
            unicodedata.category(character).startswith("C")
            or unicodedata.category(character) in {"Zl", "Zp"}
        )
        for character in value
    )


def _identity_present(value: str) -> bool:
    """Return whether public identity text contains recorded alphanumeric provenance."""
    return publication_identity_present(value)


def _valid_evaluation_date(value: str) -> bool:
    """Return whether an ISO date exists on the calendar and is not future-dated."""
    try:
        evaluated_on = date.fromisoformat(value)
    except ValueError:
        return False
    return evaluated_on <= (datetime.now(UTC) + timedelta(minutes=5)).date()


def _backticked_identity_present(value: str) -> bool:
    match = re.fullmatch(r"`(?P<identity>[^`]*)`", value)
    return match is not None and _identity_present(match.group("identity"))


def _check_verdict_tier_consistency(path: Path, text: str, offenders: list[Offender]) -> None:
    tier_rows: dict[int, tuple[int, str]] = {}
    tier_heading_lines = _section_heading_lines(text, "Tier Status")
    tier_sections = _section_occurrences(text, "Tier Status")
    if len(tier_heading_lines) > 1:
        offenders.append(Offender(path, tier_heading_lines[1], "duplicate Tier Status section"))
    elif tier_sections:
        section_tokens = tier_sections[0][1]
        for line_number in _non_ascii_tier_identifier_lines(section_tokens):
            offenders.append(Offender(path, line_number, "non-ASCII Tier Status identifier"))
        for tier in _TIER_COMPLETION_STATUSES:
            rows = _tier_status_rows(section_tokens, tier)
            if not rows:
                offenders.append(Offender(path, 1, f"missing Tier {tier} status row"))
            elif len(rows) > 1:
                offenders.append(Offender(path, rows[1][0], f"duplicate Tier {tier} status row"))
            else:
                tier_rows[tier] = rows[0]
    else:
        for tier in _TIER_COMPLETION_STATUSES:
            offenders.append(Offender(path, 1, f"missing Tier {tier} status row"))

    for tier, (line_number, status) in tier_rows.items():
        if status not in _KNOWN_TIER_STATUSES[tier]:
            offenders.append(Offender(path, line_number, f"invalid Tier {tier} status"))

    verdict_fields = _overall_verdict_fields(text)
    if not verdict_fields:
        offenders.append(Offender(path, 1, "missing Overall verdict field"))
        return
    if len(verdict_fields) > 1:
        offenders.append(Offender(path, verdict_fields[1][0], "duplicate Overall verdict field"))
        return
    verdict_line, verdict_status, _verdict_value = verdict_fields[0]
    if verdict_status not in _OVERALL_VERDICT_STATUSES:
        offenders.append(Offender(path, verdict_line, "invalid Overall verdict field"))
        return
    recommendation_heading_lines = _section_heading_lines(text, "Publication Recommendation")
    recommendation_sections = _section_occurrences(text, "Publication Recommendation")
    if len(recommendation_heading_lines) > 1:
        offenders.append(
            Offender(path, recommendation_heading_lines[1], "duplicate Publication Recommendation section")
        )
    if verdict_status != "PASS":
        if _has_publication_recommendation(text) or recommendation_sections:
            recommendation_line = recommendation_sections[0][0] if recommendation_sections else verdict_line
            offenders.append(Offender(path, recommendation_line, "non-PASS verdict recommends publication"))
        return

    _check_pass_source_identity(path, text, offenders)
    for tier, row in tier_rows.items():
        line_number, status = row
        complete = status in _TIER_COMPLETION_STATUSES[tier]
        optional = tier in {2, 3} and _metadata_field_value(text, f"Tier {tier} evidence") == "optional by policy"

        if tier == 3 and complete:
            _check_pass_provenance(path, text, offenders)
        if not complete and not (optional and status in _OPTIONAL_TIER_ABSENCE_STATUSES):
            offenders.append(Offender(path, line_number, f"publication PASS without completed Tier {tier} evidence"))


def _check_pass_source_identity(path: Path, text: str, offenders: list[Offender]) -> None:
    """Require every PASS card to identify the exact evaluated source tree."""
    metadata_lines = _metadata_section_lines(text)
    fallback_line = metadata_lines[0][0] if metadata_lines else 1
    for field, pattern in _PASS_SOURCE_METADATA_FIELD_RULES:
        matches = _metadata_field_matches(metadata_lines, field)
        line_number = matches[0][0] if matches else fallback_line
        if len(matches) != 1 or pattern.fullmatch(matches[0][1]) is None:
            offenders.append(Offender(path, line_number, f"publication PASS without recorded {field.lower()}"))


def _check_pass_provenance(path: Path, text: str, offenders: list[Offender]) -> None:
    """Reject PASS cards that replace required provenance with legacy placeholders."""
    metadata_lines = _metadata_section_lines(text)
    fallback_line = metadata_lines[0][0] if metadata_lines else 1
    for field, pattern in _PASS_METADATA_FIELD_RULES:
        matches = _metadata_field_matches(metadata_lines, field)
        line_number = matches[0][0] if matches else fallback_line
        valid_value = len(matches) == 1 and pattern.fullmatch(matches[0][1]) is not None
        if valid_value:
            if field == "Evaluation date":
                valid_value = _valid_evaluation_date(matches[0][1])
            elif field in {"Evaluator version", "Environment"}:
                valid_value = _backticked_identity_present(matches[0][1])
        if not valid_value:
            offenders.append(Offender(path, line_number, f"publication PASS without recorded {field.lower()}"))

    agent_matches = _metadata_field_matches(metadata_lines, "Agents")
    line_number = agent_matches[0][0] if agent_matches else fallback_line
    if len(agent_matches) != 1 or not _valid_recorded_agent_models(agent_matches[0][1]):
        offenders.append(Offender(path, line_number, "publication PASS without recorded agent model identity"))


def _valid_recorded_agent_models(value: str) -> bool:
    agents = _split_agent_model_states(value)
    if not agents:
        return False
    seen_agents: set[str] = set()
    for agent in agents:
        match = _AGENT_MODEL_STATE.fullmatch(agent)
        rendered_agent = _rendered_identity(match.group("agent")) if match is not None else None
        if (
            match is None
            or rendered_agent is None
            or not match.group("model").startswith("`")
            or not _identity_present(match.group("model")[1:-1])
        ):
            return False
        normalized_agent = _semantic_text(rendered_agent).casefold()
        if normalized_agent in seen_agents:
            return False
        seen_agents.add(normalized_agent)
    return True


def find_offenders(roots: list[Path]) -> tuple[list[Path], list[Offender]]:
    files: set[Path] = set()
    offenders: list[Offender] = []
    for root in roots:
        try:
            files.update(benchmark_files([root]))
        except FileNotFoundError:
            offenders.append(Offender(root.expanduser().resolve(), 1, "input path does not exist"))

    sorted_files = sorted(files)
    offenders.extend(offender for path in sorted_files for offender in scan_file(path))
    return sorted_files, offenders


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path.cwd()],
        help="BENCHMARK.md file or directory tree to scan (default: current directory)",
    )
    parser.add_argument(
        "--require-files",
        action="store_true",
        help="Fail when no BENCHMARK.md files are found",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    files, offenders = find_offenders(args.paths)

    if offenders:
        print("Public benchmark scan FAILED:")
        for offender in offenders:
            print(f"  {offender}")
        return 1
    if args.require_files and not files:
        print("Public benchmark scan FAILED: no BENCHMARK.md files found.")
        return 1

    print(f"Public benchmark scan passed ({len(files)} file(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
