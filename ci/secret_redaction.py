# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001 -- shared standalone CI helper.

"""Collect known credential matches and sanitize private scanner output."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from skillevaluator.reporting import MarkdownReporter, SARIFReporter
from skillevaluator.reporting.retained import retained_results
from skillevaluator.utils.secret_redaction import SENSITIVE_PII_CATEGORIES, Redactor, sensitive_values
from skillevaluator.utils.secure_fs import SecureRoot
from skillevaluator.validators.security import SecurityValidator

MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_FILES = 10000
MAX_TOKENS = 4096
MAX_TOKEN_BYTES = 1024 * 1024
MAX_LINE_CHARS = 8192
# The PII patterns need a quoted value. Also collect NVIDIA API keys and the
# unquoted KEY=value and Bearer forms. Such a value must contain letters and
# digits, so ordinary words and variable names are not collected. Lengths are
# bounded to limit backtracking and to stay within the redaction limits.
NVIDIA_API_KEY = re.compile(r"\bnvapi-[A-Za-z0-9_-]{40,250}")
UNQUOTED_VALUES = (
    re.compile(
        r"(?i)(?:secret|password|passwd|pwd|api[-_]?key|apikey|auth[-_]?token)\w{0,64}\s*[=:]\s*"
        r"([A-Za-z0-9_+-][A-Za-z0-9_+/=-]{15,255})(?=[\s\"'`;&|,)\]}]|$)"
    ),
    re.compile(r"(?i)\bbearer\s+([A-Za-z0-9_+-][A-Za-z0-9_+/=.~-]{15,255})(?=[\s\"'`;&|,)\]}]|$)"),
)


def collect_sensitive_values(skill: Path) -> set[str]:
    """Find credential literals using the same patterns as the PII scanner.

    We scan matches even in comments or configured exceptions: another scanner
    can still quote those values. Never put collected values into a report.
    """
    validator = SecurityValidator(use_llm=False)
    patterns = [item for item in validator._compile_pii_patterns([]) if item[0] in SENSITIVE_PII_CATEGORIES]
    tokens: set[str] = set()
    total = 0
    files = []

    def inaccessible(_error):
        raise ValueError("The credential preflight could not read a directory.")

    for directory, _directories, names in os.walk(skill, followlinks=False, onerror=inaccessible):
        files.extend(Path(directory) / name for name in names)
        if len(files) > MAX_FILES:
            raise ValueError("The credential preflight exceeded its file limit.")
    with SecureRoot(skill) as root:
        for path in files:
            raw, _ = root.read_bytes(path.relative_to(skill), MAX_FILE_BYTES)
            total += len(raw)
            if total > MAX_TOTAL_BYTES:
                raise ValueError("The credential preflight exceeded its size limit.")
            text = (
                raw.decode("utf-16", errors="ignore")
                if raw.startswith((b"\xff\xfe", b"\xfe\xff"))
                else raw.decode("utf-8-sig", errors="ignore")
            )
            # Split lines as the scanners read them, on newlines only.
            # str.splitlines() also splits on characters such as U+2028 and
            # form feed, which can sit inside a quoted credential.
            for line in re.split(r"\r\n?|\n", text):
                # Some PII patterns can backtrack heavily on very long input.
                # Refuse the scan before applying them to unbounded lines.
                if len(line) > MAX_LINE_CHARS:
                    raise ValueError("The credential preflight exceeded its line length limit.")
                for category, pattern, _exceptions, _definition in patterns:
                    for match in pattern.finditer(line):
                        tokens.update(sensitive_values(category, match.group(), line))
                tokens.update(NVIDIA_API_KEY.findall(line))
                for pattern in UNQUOTED_VALUES:
                    tokens.update(
                        value
                        for value in pattern.findall(line)
                        if any(character.isdigit() for character in value)
                        and any(character.isalpha() for character in value)
                    )
            if "PRIVATE KEY-----" in text:
                blocks = re.findall(
                    r"-----BEGIN [^\r\n]*PRIVATE KEY-----[\s\S]*?-----END [^\r\n]*PRIVATE KEY-----", text
                )
                if not blocks:
                    raise ValueError("A private key could not be safely redacted.")
                for block in blocks:
                    tokens.add(block)
                    tokens.update(line.strip() for line in block.splitlines()[1:-1] if line.strip())
            if len(tokens) > MAX_TOKENS or any(len(token.encode("utf-8")) > MAX_TOKEN_BYTES for token in tokens):
                raise ValueError("The credential preflight exceeded its redaction limit.")
    # Replacing a very short secret throughout a report can corrupt schema keys
    # and identities. Fail closed rather than risk emitting it or a false verdict.
    if any(len(token) < 4 for token in tokens):
        raise ValueError("A credential value is too short for safe report redaction.")
    return tokens


def sanitized_outputs(raw: Path, tokens: set[str]) -> dict[str, bytes]:
    """Read and sanitize bounded outputs before writing any shared artifact."""
    allowed = (
        "skillevaluator-tier1.json",
        "skillevaluator-tier1.html",
        "skillevaluator-tier1.md",
        "skillevaluator-tier1.sarif.json",
        "stdout.log",
        "stderr.log",
    )
    sanitized = {}
    total = 0
    redactor = Redactor(tokens)
    document = None
    with SecureRoot(raw) as root:
        for name in allowed:
            path = raw / name
            if not path.exists() and not path.is_symlink():
                continue
            content, _ = root.read_bytes(Path(name), MAX_FILE_BYTES)
            total += len(content)
            if total > MAX_TOTAL_BYTES:
                raise ValueError("Scanner output exceeded its redaction limit.")
            if tokens and name in {"stdout.log", "stderr.log"}:
                sanitized[name] = (
                    b"Raw scanner log withheld while credential redaction is active. See the structured reports.\n"
                )
                continue
            if name in {"skillevaluator-tier1.md", "skillevaluator-tier1.sarif.json"}:
                # These files cut and escape text, so a value can survive a
                # literal search. Render them from the redacted JSON instead.
                if document is None:
                    raise ValueError("The evaluator did not produce JSON evidence.")
                try:
                    results = retained_results(document)
                except ValueError:
                    # The runner rejects this JSON next; keep the other evidence.
                    continue
                reporter = MarkdownReporter() if name.endswith(".md") else SARIFReporter()
                sanitized[name] = reporter.render_all(results).encode("utf-8")
                continue
            text = content.decode("utf-8")
            if name.endswith(".json"):
                document = redactor.object(json.loads(text))
                text = json.dumps(document, ensure_ascii=True, allow_nan=False, indent=2)
            else:
                text = redactor.text(text)
            sanitized[name] = text.encode("utf-8")
    if "skillevaluator-tier1.json" not in sanitized:
        raise ValueError("The evaluator did not produce JSON evidence.")
    return sanitized
