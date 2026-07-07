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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from skillevaluator.constants import CONTENT_DEDUP_MIN_CHUNK_CHARS

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
        return chunk_markdown(collected, min_chars)
    if ext == ".py":
        return chunk_python(collected, min_chars)
    if ext == ".sh":
        return chunk_shell(collected, min_chars)
    return []


def chunk_markdown(
    collected: CollectedFile,
    min_chars: int,
) -> list[ContentChunk]:
    """Split markdown into heading-level sections with line tracking."""
    lines = collected.content.splitlines(keepends=True)
    sections: list[ContentChunk] = []
    current_heading = "(preamble)"
    current_start = collected.line_offset + 1
    current_lines: list[str] = []

    for i, line in enumerate(lines, start=collected.line_offset + 1):
        match = HEADING_PATTERN.match(line)
        if match:
            if current_lines:
                text = "".join(current_lines).strip()
                if len(text) >= min_chars:
                    sections.append(
                        ContentChunk(
                            source_file=collected.rel_path,
                            heading=current_heading,
                            start_line=current_start,
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
        text = "".join(current_lines).strip()
        if len(text) >= min_chars:
            sections.append(
                ContentChunk(
                    source_file=collected.rel_path,
                    heading=current_heading,
                    start_line=current_start,
                    end_line=collected.line_offset + len(lines),
                    text=text,
                    source_format="markdown",
                )
            )

    result: list[ContentChunk] = []
    for section in sections:
        if len(section.text) > 3000:
            result.extend(split_into_paragraphs(section, min_chars))
        else:
            result.append(section)

    return result


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
        chunks.append(
            ContentChunk(
                source_file=collected.rel_path,
                heading="(module docstring)",
                start_line=1,
                end_line=module_doc.count("\n") + 2,
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
