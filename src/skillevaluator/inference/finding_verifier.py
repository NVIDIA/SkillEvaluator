# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM-based false-positive verifier for security / PII findings.

Implements ``LLMClient`` (template-method mode) to send each batch of
findings (grouped by file) to an LLM that classifies them as
*true_positive*, *false_positive*, or *uncertain*.  Results are returned
as a dict mapping finding index to a verdict object.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skillevaluator.constants import LLM_VERIFY_MAX_TOKENS
from skillevaluator.inference.client import LLMClient
from skillevaluator.logging_config import get_logger

if TYPE_CHECKING:
    from skillevaluator.validators.base import Finding

logger = get_logger(__name__)

Verdict = dict[str, Any]


class FindingVerifier(LLMClient):
    """Classifies security / PII findings as true or false positives."""

    default_max_tokens: int | None = LLM_VERIFY_MAX_TOKENS

    # -- Prompt definitions -----------------------------------------------

    _SYSTEM_PROMPT = (
        "You are a security code reviewer for agent skills. "
        "Your task is to determine whether flagged security/PII findings are "
        "true positives or false positives, given the surrounding code context.\n\n"
        "For EACH finding, respond with a JSON object on its own line:\n"
        '{"index": <int>, "verdict": "true_positive"|"false_positive"|"uncertain", '
        '"confidence": "high"|"medium"|"low", "reasoning": "<brief explanation>"}\n\n'
        "Guidelines:\n"
        "- Test card numbers (4111...), placeholder data, documentation examples, "
        "and CLI command samples are false positives.\n"
        "- Zero-value coordinates (0.0, 0.0) used as defaults are false positives.\n"
        "- URLs with embedded credentials (://user:pass@host) are not email addresses.\n"
        "- Standard Docker commands (docker run --rm) are not tool misuse.\n"
        "- Trusted package manager installs (pip, npm, brew, uv) are not supply chain attacks.\n"
        "- Real secrets, actual PII, and genuine security issues are true positives.\n"
    )

    def get_system_prompt(self) -> str:
        return self._SYSTEM_PROMPT

    def create_user_prompt(self, **kwargs: Any) -> str:
        """Build a user prompt from *findings* and *skill_path*.

        Expected kwargs:
            findings: list[Finding]
            skill_path: Path
        """
        findings: list[Finding] = kwargs["findings"]
        skill_path: Path = kwargs["skill_path"]
        return self._build_prompt(findings, skill_path)

    def parse_response(self, response_text: str, **_kwargs: Any) -> dict[int, Verdict]:
        """Parse newline-delimited JSON verdicts from the LLM response."""
        return self._parse_verdicts(response_text)

    def get_fallback_response(self, **_kwargs: Any) -> dict[int, Verdict]:
        """When LLM is unavailable, return an empty dict (no verdicts)."""
        return {}

    # -- Prompt construction ----------------------------------------------

    @staticmethod
    def _can_safely_open_under_skill(skill_path: Path, file_path: str) -> bool:
        """Return whether *file_path* resolves to a regular file inside the skill."""
        if not file_path:
            return True
        skill_root = skill_path.resolve()
        for candidate in (skill_path / file_path, Path(file_path)):
            if FindingVerifier._resolved_inside_skill(candidate, skill_root) is not None:
                return True
        return False

    @staticmethod
    def _redacted_prompt_fields() -> tuple[str, str, str]:
        """Message, line content, and line number placeholders for unsafe files."""
        return (
            "(redacted: file outside skill boundary)",
            "(redacted)",
            "?",
        )

    def _build_prompt(self, findings: list[Finding], skill_path: Path) -> str:
        """Build a user prompt grouping findings by file with surrounding code context."""
        groups: dict[str, list[tuple[int, Finding]]] = {}
        for i, f in enumerate(findings):
            groups.setdefault(f.file_path, []).append((i, f))

        parts: list[str] = []
        for file_path, indexed_findings in groups.items():
            file_context = self._read_file_context(skill_path, file_path)
            parts.append(f"=== File: {file_path} ===")
            if file_context:
                parts.append(f"Context (first 100 lines):\n```\n{file_context}\n```")

            for idx, finding in indexed_findings:
                sev = finding.severity.value if hasattr(finding.severity, "value") else str(finding.severity)
                if self._can_safely_open_under_skill(skill_path, finding.file_path or ""):
                    message = finding.message
                    line_content = finding.line_content or "(no content)"
                    line_number = str(finding.line_number or "?")
                else:
                    message, line_content, line_number = self._redacted_prompt_fields()
                parts.append(
                    f"\nFinding #{idx}:\n"
                    f"  Category: {finding.category}\n"
                    f"  Severity: {sev}\n"
                    f"  Check: {finding.check_name}\n"
                    f"  Message: {message}\n"
                    f"  Line {line_number}: {line_content}\n"
                    f"  Suggestion: {finding.suggestion or 'N/A'}"
                )

        parts.append(
            "\nFor each finding above, output one JSON object per line with index, verdict, confidence, and reasoning."
        )
        return "\n".join(parts)

    @staticmethod
    def _resolved_inside_skill(path: Path, skill_root: Path) -> Path | None:
        """Return *path* resolved only if it is a file inside *skill_root*."""
        try:
            resolved = path.resolve()
            if not resolved.is_file() or not resolved.is_relative_to(skill_root):
                return None
            return resolved
        except (OSError, ValueError, RuntimeError):
            return None

    @staticmethod
    def _read_file_context(skill_path: Path, file_path: str) -> str | None:
        """Read up to 100 lines from a file under the skill root for LLM context."""
        if not file_path:
            return None
        skill_root = skill_path.resolve()
        candidates = [skill_path / file_path, Path(file_path)]
        for path in candidates:
            resolved = FindingVerifier._resolved_inside_skill(path, skill_root)
            if resolved is None:
                continue
            try:
                lines = resolved.read_text(encoding="utf-8", errors="ignore").split("\n")[:100]
                return "\n".join(lines)
            except (OSError, ValueError):
                continue
        return None

    # -- Response parsing -------------------------------------------------

    @staticmethod
    def _parse_verdicts(response_text: str) -> dict[int, Verdict]:
        verdicts: dict[int, Verdict] = {}
        for line in response_text.strip().split("\n"):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
                idx = obj.get("index")
                if idx is not None and "verdict" in obj:
                    verdicts[int(idx)] = obj
            except (ValueError, TypeError):
                continue
        return verdicts

    # -- High-level entry point -------------------------------------------

    def verify(self, findings: list[Finding], skill_path: Path) -> dict[int, Verdict]:
        """Convenience wrapper around ``process()``."""
        return self.process(findings=findings, skill_path=skill_path)
