# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Find CHANGELOG entries by their opening words, for tests that pin them."""

from __future__ import annotations

import re
from html import unescape
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"


def _plain(text: str) -> str:
    """Drop HTML tags, entities and Markdown emphasis, and collapse whitespace."""
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", text)).replace("*", "").split())


def changelog_entry(lead: str) -> tuple[str, str, str]:
    """Return the release heading, subsection heading and plain text of the one entry starting with ``lead``."""
    release = subsection = ""
    found: list[tuple[str, str, str]] = []
    for block in CHANGELOG.read_text(encoding="utf-8").split("\n\n"):
        first = block.strip().splitlines()[0] if block.strip() else ""
        if first.startswith("## "):
            release, subsection = first, ""
        elif first.startswith("### "):
            subsection = first
        elif _plain(block).startswith(lead):
            found.append((release, subsection, _plain(block)))
    assert len(found) == 1
    return found[0]


def newer_than_0_9_0(release: str) -> bool:
    """Whether ``release`` is ``## Unreleased`` or the heading of a release after 0.9.0."""
    if release == "## Unreleased":
        return True
    match = re.match(r"## (\d+)\.(\d+)\.(\d+)\b", release)
    return match is not None and tuple(int(part) for part in match.groups()) > (0, 9, 0)
