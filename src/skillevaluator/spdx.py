# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict recognition of the public repository's SPDX HTML preamble."""

from __future__ import annotations

import re

_HTML_COMMENT_PATTERN = re.compile(r"<!--(?P<body>.*?)-->", re.DOTALL)
_PUBLIC_COPYRIGHT_PATTERN = re.compile(
    r"^SPDX-FileCopyrightText: Copyright \(c\) \d{4}(?:-\d{4})? "
    r"NVIDIA CORPORATION & AFFILIATES\. All rights reserved\.$"
)
_PUBLIC_LICENSE_PATTERN = re.compile(
    r"^SPDX-License-Identifier: (?:Apache-2\.0|CC-BY-4\.0 AND Apache-2\.0)$"
)


def is_spdx_only_html_comment(text: str, *, allow_frontmatter_separator: bool = False) -> bool:
    """Return whether *text* is exactly the repository's two-line SPDX comment."""
    matches = list(_HTML_COMMENT_PATTERN.finditer(text))
    if len(matches) != 1:
        return False

    comment = matches[0]
    surrounding = f"{text[: comment.start()]}{text[comment.end() :]}".strip()
    allowed_surrounding = {"", "---"} if allow_frontmatter_separator else {""}
    if surrounding not in allowed_surrounding:
        return False

    lines = [line.strip() for line in comment.group("body").splitlines() if line.strip()]
    return len(lines) == 2 and bool(
        _PUBLIC_COPYRIGHT_PATTERN.fullmatch(lines[0]) and _PUBLIC_LICENSE_PATTERN.fullmatch(lines[1])
    )
