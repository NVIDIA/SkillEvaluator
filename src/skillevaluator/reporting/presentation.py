# SPDX-License-Identifier: Apache-2.0
"""Shared presentation of explicit evidence that a check was not applicable."""

from __future__ import annotations


def success_is_not_applicable(check: object, message: object, metadata: object = None) -> bool:
    """Use typed producer metadata, with exact legacy evidence for retained reports.

    A message merely mentioning applicability is not proof of a skipped check.
    Keep the compatibility vocabulary in one place for HTML and CI Markdown.
    """
    if isinstance(metadata, dict) and type(metadata.get("not_applicable")) is bool:
        return metadata["not_applicable"]
    evidence = {
        "version_optional": None,
        "lint": "No scripts/ or tools/ directory found",
        "code_file_discovery": "No code files found - code risk analysis is not applicable",
        "dependency_file_discovery": "No dependency files found - vulnerability audit is not applicable",
    }
    if not isinstance(check, str) or check not in evidence:
        return False
    expected = evidence[check]
    return expected is None or message == expected
