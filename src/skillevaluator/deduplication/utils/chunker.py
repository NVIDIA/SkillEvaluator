# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content chunking with line tracking for deduplication.

Three strategies:
  - Markdown: heading-based sections + paragraph fallback
  - Python: AST extraction (docstrings, functions, comments)
  - Shell: function bodies + comment blocks
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from markdown_it import MarkdownIt

from skillevaluator.constants import CONTENT_DEDUP_MAX_CHUNK_CHARS, CONTENT_DEDUP_MIN_CHUNK_CHARS
from skillevaluator.spdx import is_spdx_only_html_comment

if TYPE_CHECKING:
    from skillevaluator.deduplication.utils.skill_collector import CollectedFile

logger = logging.getLogger(__name__)

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)", re.MULTILINE)
SHELL_FUNC_PATTERN = re.compile(r"^(?:function\s+)?(\w+)\s*\(\)\s*\{", re.MULTILINE)


@dataclass
class ContentChunk:
    """A section or paragraph extracted from a file within a skill."""

    source_file: str  # rel_path from CollectedFile
    heading: str  # section title, function name, or "(preamble)"
    start_line: int  # 1-based
    end_line: int  # 1-based
    text: str
    source_format: str  # "markdown" | "python" | "shell"
    embedding: list[float] = field(default_factory=list)
    # Pieces of one source chunk must not create duplicate findings with each other.
    split_group: tuple[str, int, int] | None = None

    @property
    def char_count(self) -> int:
        return len(self.text)


def chunk_file(
    collected: CollectedFile,
    min_chars: int = CONTENT_DEDUP_MIN_CHUNK_CHARS,
) -> list[ContentChunk]:
    """Route to the appropriate chunker based on file extension."""
    ext = collected.extension
    if ext in {".md", ".mdc"}:
        chunks = chunk_markdown(collected, min_chars)
    elif ext == ".py":
        chunks = chunk_python(collected, min_chars)
    elif ext == ".sh":
        chunks = chunk_shell(collected, min_chars)
    else:
        return []
    return [piece for chunk in chunks for piece in _bounded_pieces(chunk)]


def _bounded_pieces(chunk: ContentChunk) -> list[ContentChunk]:
    """Cap provider inputs, keeping meaningful tails and source containment.

    The minimum-length filter applies to the original chunk. Its final short
    piece still contains evidence and must not disappear during splitting.
    Python text joins signatures and cleaned docstrings rather than contiguous
    source lines, so every piece retains the enclosing AST node's range.
    """
    limit = CONTENT_DEDUP_MAX_CHUNK_CHARS
    if len(chunk.text) <= limit:
        return [chunk]
    pieces: list[ContentChunk] = []
    group = chunk.split_group or (chunk.source_file, chunk.start_line, chunk.end_line)
    buffer = ""
    buffer_start = 0

    def add(text: str, first: int, last: int) -> None:
        if not text.strip():
            return
        start = min(chunk.start_line + first, chunk.end_line)
        end = min(chunk.start_line + last, chunk.end_line)
        if chunk.source_format == "python":
            start, end = chunk.start_line, chunk.end_line
        pieces.append(
            replace(chunk, text=text.strip(), start_line=start, end_line=end, embedding=[], split_group=group)
        )

    def flush(last: int) -> None:
        nonlocal buffer
        if buffer:
            add(buffer, buffer_start, last)
            buffer = ""

    lines = chunk.text.splitlines(keepends=True)
    for offset, line in enumerate(lines):
        if len(line) > limit:
            flush(max(0, offset - 1))
            for index in range(0, len(line), limit):
                add(line[index : index + limit], offset, offset)
        else:
            if len(buffer) + len(line) > limit:
                flush(max(0, offset - 1))
            if not buffer:
                buffer_start = offset
            buffer += line
    flush(len(lines) - 1)
    return pieces


def chunk_markdown(
    collected: CollectedFile,
    min_chars: int,
) -> list[ContentChunk]:
    """Split markdown into heading-level sections with line tracking."""
    lines = collected.content.splitlines(keepends=True)
    # A heading-like line in a fenced example is literal content, not a section.
    fenced_lines = {
        line + collected.line_offset + 1
        for token in MarkdownIt("commonmark").parse(collected.content)
        if token.type == "fence" and token.map is not None
        for line in range(*token.map)
    }
    sections: list[ContentChunk] = []
    current_heading = "(preamble)"
    current_start = collected.line_offset + 1
    current_lines: list[str] = []

    for i, line in enumerate(lines, start=collected.line_offset + 1):
        match = None if i in fenced_lines else HEADING_PATTERN.match(line)
        if match:
            if current_lines:
                raw_text = "".join(current_lines)
                text = raw_text.strip()
                leading_lines = raw_text[: len(raw_text) - len(raw_text.lstrip())].count("\n")
                if len(text) >= min_chars and not (current_heading == "(preamble)" and is_spdx_only_html_comment(text)):
                    sections.append(
                        ContentChunk(
                            source_file=collected.rel_path,
                            heading=current_heading,
                            start_line=current_start + leading_lines,
                            end_line=i - 1,
                            text=text,
                            source_format="markdown",
                        )
                    )
            current_heading = match.group(0).strip()
            current_start = i
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        raw_text = "".join(current_lines)
        text = raw_text.strip()
        leading_lines = raw_text[: len(raw_text) - len(raw_text.lstrip())].count("\n")
        if len(text) >= min_chars and not (current_heading == "(preamble)" and is_spdx_only_html_comment(text)):
            sections.append(
                ContentChunk(
                    source_file=collected.rel_path,
                    heading=current_heading,
                    start_line=current_start + leading_lines,
                    end_line=collected.line_offset + len(lines),
                    text=text,
                    source_format="markdown",
                )
            )

    result: list[ContentChunk] = []
    for section in sections:
        if len(section.text) > CONTENT_DEDUP_MAX_CHUNK_CHARS:
            result.extend(_pack_paragraphs(section, min_chars))
        else:
            result.append(section)

    return result


def _pack_paragraphs(section: ContentChunk, min_chars: int) -> list[ContentChunk]:
    """Pack consecutive paragraphs of an oversized section up to the embedding cap.

    Paragraphs shorter than ``min_chars`` stay attached to a neighbour instead of
    becoming chunks of their own; ``_bounded_pieces`` caps any overflow. Every
    piece shares the section's split group, so pieces never cluster together.
    """
    group = (section.source_file, section.start_line, section.end_line)
    packed: list[ContentChunk] = []
    for paragraph in split_into_paragraphs(section, min_chars=1):
        # Keep the source line gap so the bounding pass can map pieces back to
        # their actual lines even when paragraphs have several blank lines.
        separator = "\n" * max(2, paragraph.start_line - packed[-1].end_line) if packed else ""
        if packed and (
            len(packed[-1].text) + len(separator) + len(paragraph.text) <= CONTENT_DEDUP_MAX_CHUNK_CHARS
            or len(paragraph.text) < min_chars
            or len(packed[-1].text) < min_chars
        ):
            last = packed[-1]
            packed[-1] = replace(last, text=f"{last.text}{separator}{paragraph.text}", end_line=paragraph.end_line)
        else:
            packed.append(replace(paragraph, split_group=group))
    return packed


def split_into_paragraphs(
    section: ContentChunk,
    min_chars: int,
) -> list[ContentChunk]:
    """Split an oversized section into paragraph-level chunks."""
    paragraphs: list[ContentChunk] = []
    separators = list(re.finditer(r"\n\s*\n", section.text))
    spans: list[tuple[int, int]] = []
    start = 0
    for separator in separators:
        spans.append((start, separator.start()))
        start = separator.end()
    spans.append((start, len(section.text)))

    for block_start, block_end in spans:
        block = section.text[block_start:block_end]
        block_stripped = block.strip()
        if len(block_stripped) >= min_chars:
            content_start = block_start + len(block) - len(block.lstrip())
            content_end = block_start + len(block.rstrip())
            start_line = section.start_line + section.text[:content_start].count("\n")
            end_line = section.start_line + section.text[:content_end].count("\n")
            paragraphs.append(
                ContentChunk(
                    source_file=section.source_file,
                    heading=section.heading,
                    start_line=start_line,
                    end_line=end_line,
                    text=block_stripped,
                    source_format="markdown",
                )
            )

    return paragraphs or [section]


def chunk_python(
    collected: CollectedFile,
    min_chars: int,
) -> list[ContentChunk]:
    """Extract docstrings and function/class signatures via AST."""
    try:
        tree = ast.parse(collected.content)
    except SyntaxError:
        logger.debug("Skipping %s: SyntaxError during AST parse", collected.rel_path)
        return []

    chunks: list[ContentChunk] = []

    module_doc = ast.get_docstring(tree)
    if module_doc and len(module_doc) >= min_chars:
        doc_node = tree.body[0]
        chunks.append(
            ContentChunk(
                source_file=collected.rel_path,
                heading="(module docstring)",
                start_line=doc_node.lineno,
                end_line=doc_node.end_lineno or doc_node.lineno,
                text=module_doc,
                source_format="python",
            )
        )

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node) or ""
            name = node.name
            if isinstance(node, ast.ClassDef):
                heading = f"class {name}"
            else:
                heading = f"{name}()"

            sig_line = collected.content.splitlines()[node.lineno - 1]
            text = f"{sig_line}\n{doc}" if doc else sig_line

            if len(text) >= min_chars:
                end = node.end_lineno or node.lineno
                chunks.append(
                    ContentChunk(
                        source_file=collected.rel_path,
                        heading=heading,
                        start_line=node.lineno,
                        end_line=end,
                        text=text,
                        source_format="python",
                    )
                )

    return chunks


def chunk_shell(
    collected: CollectedFile,
    min_chars: int,
) -> list[ContentChunk]:
    """Extract function bodies and comment blocks from shell scripts."""
    lines = collected.content.splitlines()
    chunks: list[ContentChunk] = []

    comment_start = None
    comment_lines: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("#!"):
            if comment_start is None:
                comment_start = i
            comment_lines.append(stripped.lstrip("# "))
        else:
            if comment_start is not None and len(comment_lines) >= 3:
                text = "\n".join(comment_lines)
                if len(text) >= min_chars:
                    chunks.append(
                        ContentChunk(
                            source_file=collected.rel_path,
                            heading="(comment)",
                            start_line=comment_start + 1,
                            end_line=i,
                            text=text,
                            source_format="shell",
                        )
                    )
            comment_start = None
            comment_lines = []

    full_text = collected.content
    for match in SHELL_FUNC_PATTERN.finditer(full_text):
        func_name = match.group(1)
        brace_start = match.end() - 1
        depth = 1
        pos = brace_start + 1
        while pos < len(full_text) and depth > 0:
            if full_text[pos] == "{":
                depth += 1
            elif full_text[pos] == "}":
                depth -= 1
            pos += 1

        if depth == 0:
            func_text = full_text[match.start() : pos]
            start_line = full_text[: match.start()].count("\n") + 1
            end_line = full_text[:pos].count("\n") + 1
            if len(func_text) >= min_chars:
                chunks.append(
                    ContentChunk(
                        source_file=collected.rel_path,
                        heading=f"{func_name}()",
                        start_line=start_line,
                        end_line=end_line,
                        text=func_text,
                        source_format="shell",
                    )
                )

    return chunks
