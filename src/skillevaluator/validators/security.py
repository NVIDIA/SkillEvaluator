# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security and PII Scanning Validator.

Detects security vulnerabilities and PII/secrets using:
- skillspector CLI: 15+ vulnerability patterns (prompt injection, data exfil, etc.)
  Invoked via `skillspector scan <path> --format json`; use `--no-llm` for static-only.
- Custom regex patterns: Personal paths, emails, phone numbers, SSNs, IPs
- Optional LLM verification layer to suppress false positives (--llm-verify)
"""

from __future__ import annotations

import bisect
import contextlib
import functools
import getpass
import hashlib
import html
import io
import itertools
import json
import math
import os
import re
import shutil
import stat
import struct
import tempfile
import tokenize
import unicodedata
import zlib
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

import yaml
from yaml.events import ScalarEvent

from skillevaluator import finding_verification_policy
from skillevaluator.config import load_pii_patterns
from skillevaluator.constants import (
    HOME_PATH_SUBMITTER_ENV_VARS,
    SCAN_EXCLUDED_DIRS,
    SCAN_EXCLUDED_FILES,
    SCANNABLE_EXTENSIONS,
    SKILL_MANIFEST_VARIANTS,
)
from skillevaluator.logging_config import get_logger
from skillevaluator.models.skill import SEMVER_RE
from skillevaluator.provider_config import (
    SKILLSPECTOR_DEFAULT_MODELS,
    ProviderConfigurationError,
    resolve_llm_provider,
    skillspector_reasoning_effort,
)
from skillevaluator.skillspector_diagnostics import completeness_record, merge_completeness
from skillevaluator.skillspector_provenance import SKILLSPECTOR_LLM_ANALYZERS, sanitize_skillspector_llm_provenance
from skillevaluator.skillspector_reference_caveats import REFERENCE_COMPLETENESS_FIELDS, reference_caveat_metadata
from skillevaluator.skillspector_references import (
    AE1_PATTERN,
    AE1_RULE_ID,
    REFERENCE_SUMMARY_KEY,
    ReportNormalizer,
    combine_reference_summaries,
    index_legacy_reference_reasons,
    issue_diagnostics,
    reference_suggestion,
    reference_summary,
)
from skillevaluator.spdx import is_spdx_only_html_comment
from skillevaluator.utils.path_security import matches_filesystem_name
from skillevaluator.utils.scanner_install import prepare_skillspector
from skillevaluator.utils.secure_fs import (
    MAX_SECURE_DIRECTORY_DEPTH,
    SecurePathError,
    SecureRoot,
    stat_is_link_or_reparse,
)
from skillevaluator.utils.tool_runner import Tools, parse_json_output
from skillevaluator.validators.base import (
    Finding,
    Severity,
    ValidationResult,
    ValidatorBase,
    iter_scannable_files,
)
from skillevaluator.validators.pii_ownership import ownership_email_spans
from skillevaluator.validators.pii_waivers import PIIWaiverSet, configured_pii_categories, pii_evidence_sha256
from skillevaluator.validators.skillspector_scope import (
    IssueScope,
    scope_skillspector_issues,
    skillspector_report_is_truncated,
    skillspector_report_serializes_every_finding,
)

logger = get_logger(__name__)

# Watchdog for each SkillSpector subprocess. The default stays above
# SkillSpector's own workflow deadline (1,080 seconds in v2.13.3) so that deadline, not a kill,
# normally ends a slow scan. The cap keeps the value inside the range that
# subprocess waits accept on every platform.
_DEFAULT_SKILLSPECTOR_TIMEOUT_SECONDS = 1200
_MAX_SKILLSPECTOR_TIMEOUT_SECONDS = 86400
_SKILLSPECTOR_TIMEOUT_ENV = "SKILLEVALUATOR_SKILLSPECTOR_TIMEOUT_SECONDS"

_AUTHOR_IDENTITY_RE = re.compile(r"^\S[^<>\n]* <(?P<email>[^<>@\s]+@[^<>\s]+)>$")
_SKILLSPECTOR_POLICY_EXIT_CODES = frozenset({0, 1})
# SkillSpector maps an unreadable or invalid --baseline file to exit code 2.
_SKILLSPECTOR_INPUT_ERROR_EXIT_CODE = 2
_SKILLSPECTOR_BASELINE_SCHEMA_VERSION = 2
# SkillSpector's baseline loader limits (v2.12.0 and later); a file beyond
# them is rejected by SkillSpector, so the pre-check stops there too.
_SKILLSPECTOR_BASELINE_MAX_BYTES = 2 * 1024 * 1024
_SKILLSPECTOR_BASELINE_MAX_NODES = 100_000
_SKILLSPECTOR_BASELINE_MAX_DEPTH = 64
_SKILLSPECTOR_BASELINE_MAX_SCALAR_CHARS = 64 * 1024
_SKILLSPECTOR_STATUSLESS_COMPLETENESS_VERSIONS = {(2, 9, 5), (2, 9, 6)}
_SKILLSPECTOR_FINDING_IDENTITY_VERSION = (2, 11, 1)
_SKILLSPECTOR_COMPLETENESS_SCHEMA_VERSION = (2, 10, 0)
_SKILLSPECTOR_DOCS_ONLY_APPLICABILITY_VERSION = (2, 11, 2)
# SkillSpector 2.13.4 reports an analysis whose only gaps are missing references
# as ``complete_with_caveats`` instead of ``partial``; earlier releases never do.
_SKILLSPECTOR_CAVEATS_STATUS_VERSION = (2, 13, 4)
_SKILLSPECTOR_DOCS_ONLY_NOT_APPLICABLE_ANALYZERS = frozenset(
    {
        "behavioral_ast",
        "behavioral_taint_tracking",
        "bundled_execution_surface",
        "mcp_least_privilege",
        "meta_analyzer",
    }
)
# SkillSpector 2.11.1+ classifies each occurrence separately, so benign-context
# tags describe the reported row instead of a compacted sibling.
_SKILLSPECTOR_OCCURRENCE_TRIAGE_VERSION = (2, 11, 1)
_SKILLSPECTOR_BENIGN_CONTEXT_TAGS = frozenset({"contextual-triage", "likely-benign-context"})
_SKILLSPECTOR_PASSWD_MOUNT_SOURCE_LIMIT_BYTES = 1024 * 1024
_PASSWD_PATH_PATTERN = re.compile(r"/etc/passwd", re.IGNORECASE)
# SkillSpector's own bounded grammar for an exact read-only UID-map mount.
_READ_ONLY_PASSWD_VOLUME_PATTERN = re.compile(
    r"\b(?:docker|podman)\s+run\b"
    r"(?:(?:\\\r?\n)|[^\n;&|]){0,1000}?"
    r"(?:-v|--volume)(?:=|\s+)"
    r"(?P<quote>['\"]?)"
    r"(?P<source>/etc/passwd):(?P<target>/etc/passwd):ro"
    r"(?P=quote)(?=$|[\s\\])",
    re.IGNORECASE | re.MULTILINE,
)


def _is_filesystem_safe_path_text(value: str) -> bool:
    """Return whether an untrusted report path can be passed to OS path APIs."""
    try:
        encoded = os.fsencode(value)
    except (ValueError, UnicodeError):
        return False
    return bool(encoded) and b"\x00" not in encoded and not any(
        unicodedata.category(character).startswith("C")
        or unicodedata.category(character) in {"Zl", "Zp"}
        for character in value
    )


_SKILLSPECTOR_SEMANTIC_ANALYZERS = frozenset(
    {
        "semantic_developer_intent",
        "semantic_quality_policy",
        "semantic_security_discovery",
    }
)
_SKILLSPECTOR_OPTIONAL_ANALYZERS = _SKILLSPECTOR_SEMANTIC_ANALYZERS | {"meta_analyzer"}
_SKILLSPECTOR_COMMON_REQUIRED_ANALYZERS = frozenset(
    {
        "behavioral_ast",
        "behavioral_taint_tracking",
        "mcp_least_privilege",
        "mcp_rug_pull",
        "mcp_tool_poisoning",
        "meta_analyzer",
        "static_patterns_agent_snooping",
        "static_patterns_anti_refusal",
        "static_patterns_data_exfiltration",
        "static_patterns_deserialization",
        "static_patterns_excessive_agency",
        "static_patterns_harmful_content",
        "static_patterns_memory_poisoning",
        "static_patterns_output_handling",
        "static_patterns_privilege_escalation",
        "static_patterns_prompt_injection",
        "static_patterns_rogue_agent",
        "static_patterns_ssrf",
        "static_patterns_supply_chain",
        "static_patterns_system_prompt_leakage",
        "static_patterns_tool_misuse",
        "static_yara",
    }
)
_SKILLSPECTOR_2_9_6_REQUIRED_ANALYZERS = (
    _SKILLSPECTOR_COMMON_REQUIRED_ANALYZERS | _SKILLSPECTOR_SEMANTIC_ANALYZERS
)
# SkillSpector 2.10+ can omit semantic analyzers when no provider is available.
_SKILLSPECTOR_2_10_REQUIRED_ANALYZERS = _SKILLSPECTOR_COMMON_REQUIRED_ANALYZERS | {
    "artifact_integrity"
}
_SKILLSPECTOR_COMMON_UNIVERSAL_ANALYZERS = frozenset(
    analyzer_id
    for analyzer_id in _SKILLSPECTOR_COMMON_REQUIRED_ANALYZERS
    if analyzer_id == "static_yara" or analyzer_id.startswith("static_patterns_")
)
_SKILLSPECTOR_DISABLED_ANALYZER_LIMITATION = "Analyzer was disabled by the requested configuration."
_SKILLSPECTOR_RISK_BANDS = ((81, "CRITICAL"), (51, "HIGH"), (21, "MEDIUM"), (0, "LOW"))
_SKILLSPECTOR_RECOMMENDATION_BY_SEVERITY = {
    "CRITICAL": "DO_NOT_INSTALL",
    "HIGH": "DO_NOT_INSTALL",
    "MEDIUM": "CAUTION",
    "LOW": "SAFE",
}
_SKILLSPECTOR_SEVERITY_POINTS = {"CRITICAL": 50, "HIGH": 25, "MEDIUM": 10, "LOW": 5, "INFO": 5}
_SKILLSPECTOR_EXECUTABLE_MULTIPLIER = 1.3
_SKILLSPECTOR_RISK_SCORE_FLOORS_BY_RULE_ID = {"SC8": 51}
_SKILLSPECTOR_DIMINISHING_WEIGHTS = (1.0, 0.5, 0.25)
# Ledger exceptions quoted in an incomplete-analysis error, and the length of each quoted field.
_SKILLSPECTOR_INCOMPLETE_EXAMPLES = 3
_SKILLSPECTOR_REPORT_TEXT_MAX_CHARS = 200
# Evaluation definitions and helpers are authored inputs, including when linked
# from SKILL.md. Keep them and shipped bytecode visible to SkillSpector; generated
# results below these directories still match the existing output exclusions.
_SKILLSPECTOR_SCAN_EXCLUDED_DIRS = SCAN_EXCLUDED_DIRS - {"__pycache__", "evals", ".evals"}
_SKILLSPECTOR_PROVIDER_MAP = {
    "anthropic": "anthropic",
    "bedrock": "bedrock",
    # SkillSpector's nv_build registry holds the Build models' output caps; with
    # no nv_build credential in the child, its requests use the OPENAI_* fallback.
    "nv_build": "nv_build",
    "openai": "openai",
    "openai-compatible": "openai",
}
_SKILLSPECTOR_PROCESS_ENV_NAMES = frozenset(
    {
        "COMSPEC",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "PATHEXT",
        "REQUESTS_CA_BUNDLE",
        # SkillSpector's own workflow deadline, raised together with the watchdog.
        "SKILLSPECTOR_MAX_WORKFLOW_SECONDS",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "WINDIR",
    }
)
_SKILLSPECTOR_AWS_ENV_NAMES = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_CONFIG_FILE",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_DEFAULT_REGION",
        "AWS_PROFILE",
        "AWS_REGION",
        "AWS_ROLE_ARN",
        "AWS_ROLE_SESSION_NAME",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_SDK_LOAD_CONFIG",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    }
)
_SKILLSPECTOR_EXPLICIT_PROVIDER_ENV = {
    "anthropic": frozenset({"ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"}),
    "bedrock": _SKILLSPECTOR_AWS_ENV_NAMES,
    "openai": frozenset({"OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_PROJECT_ID"}),
}

_SKILLSPECTOR_LLM_FAILURE_MARKERS = (
    "authorization failed",
    "authentication failed",
    "llm analysis failed",
    "llm batch failed",
    "llm call failed",
    "llm check failed",
    "llm not configured",
    "llm returned malformed",
    "llm unavailable",
)
_LLM_VERDICTS = frozenset({"true_positive", "false_positive", "uncertain"})
_LLM_CONFIDENCE_LEVELS = frozenset({"high", "medium", "low"})
_HEREDOC_OPEN = re.compile(r"""<<-?(?!<)\s*(?:'([^']+)'|"([^"]+)"|\\?([A-Za-z_][A-Za-z0-9_]*))""")
_FENCE_OPEN = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")
_PYTHON_FENCE_LANGS = frozenset({"python", "py", "python3"})
_YAML_FENCE_LANGS = frozenset({"yaml", "yml"})
_SHELL_FENCE_LANGS = frozenset({"sh", "bash", "shell", "zsh"})


def _leading_hash_or_slash_comment_lines(lines: list[str]) -> frozenset[int]:
    """Lines whose first non-space text is `#` or `//`."""
    return frozenset(index for index, line in enumerate(lines, 1) if line.lstrip().startswith(("#", "//")))


def _shift_line_numbers(line_numbers: frozenset[int], offset: int) -> frozenset[int]:
    return frozenset(number + offset for number in line_numbers)


def _python_comment_line_numbers(source: str) -> frozenset[int]:
    """Line numbers whose first non-space token is a Python comment."""
    skip: set[int] = set()
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            if token.line.lstrip().startswith("#"):
                skip.add(token.start[0])
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return _leading_hash_or_slash_comment_lines(source.split("\n"))
    return frozenset(skip)


def _yaml_scalar_content_lines(source: str) -> frozenset[int] | None:
    """1-indexed lines covered by YAML scalars, or None when the document cannot be parsed."""
    try:
        covered: set[int] = set()
        for event in yaml.parse(source, Loader=yaml.SafeLoader):
            if not isinstance(event, ScalarEvent):
                continue
            start = event.start_mark.line
            end = event.end_mark.line - (1 if event.end_mark.column == 0 else 0)
            for index in range(start, max(start, end) + 1):
                covered.add(index + 1)
        return frozenset(covered)
    except yaml.YAMLError:
        return None


def _yaml_comment_line_numbers(lines: list[str]) -> frozenset[int]:
    """YAML `#` comments only. Scalar content, including block and quoted forms, is scanned."""
    covered = _yaml_scalar_content_lines("\n".join(lines))
    if covered is None:
        return frozenset()
    return frozenset(
        index for index, line in enumerate(lines, 1) if line.lstrip().startswith("#") and index not in covered
    )


def _shell_comment_line_numbers(lines: list[str]) -> frozenset[int]:
    """Shell `#` comments, excluding heredoc payloads."""
    skip: set[int] = set()
    delimiter: str | None = None
    strip_tabs = False
    for index, line in enumerate(lines, 1):
        if delimiter is not None:
            candidate = line.rstrip("\r\n")
            if strip_tabs:
                candidate = candidate.lstrip("\t")
            if candidate == delimiter:
                delimiter = None
            continue
        if line.lstrip().startswith("#"):
            skip.add(index)
            continue
        opened = _HEREDOC_OPEN.search(line)
        if opened:
            delimiter = opened.group(1) or opened.group(2) or opened.group(3)
            strip_tabs = "<<-" in opened.group(0)
    return frozenset(skip)


def _fence_language(info: str) -> str:
    token = info.strip().split()[0] if info.strip() else ""
    return token.strip("{.}").lower()


def _is_closing_fence(line: str, marker: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= len(marker) and set(stripped) <= {marker[0]}


def _comments_for_language(language: str, lines: list[str]) -> frozenset[int]:
    if language in _PYTHON_FENCE_LANGS:
        return _python_comment_line_numbers("\n".join(lines))
    if language in _YAML_FENCE_LANGS:
        return _yaml_comment_line_numbers(lines)
    if language in _SHELL_FENCE_LANGS:
        return _shell_comment_line_numbers(lines)
    return _leading_hash_or_slash_comment_lines(lines)


def _markdown_frontmatter_end(lines: list[str]) -> int | None:
    if not lines or lines[0].strip() != "---":
        return None
    try:
        return next(index for index, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration:
        return None


def _markdown_fence_comment_line_numbers(lines: list[str], start: int) -> frozenset[int]:
    skip: set[int] = set()
    index = start
    while index < len(lines):
        match = _FENCE_OPEN.match(lines[index])
        if match is None:
            index += 1
            continue
        marker = match.group(2)
        language = _fence_language(match.group(3))
        body_start = index + 1
        body_end = body_start
        while body_end < len(lines) and not _is_closing_fence(lines[body_end], marker):
            body_end += 1
        skip.update(_shift_line_numbers(_comments_for_language(language, lines[body_start:body_end]), body_start))
        index = body_end + 1
    return frozenset(skip)


def _markdown_comment_line_numbers(lines: list[str]) -> frozenset[int]:
    """Skip YAML frontmatter comments and fenced-code comments, not ATX headings."""
    skip: set[int] = set()
    body_at = 0
    fm_end = _markdown_frontmatter_end(lines)
    if fm_end is not None:
        skip.update(_shift_line_numbers(_yaml_comment_line_numbers(lines[1:fm_end]), 1))
        body_at = fm_end + 1
    skip.update(_markdown_fence_comment_line_numbers(lines, body_at))
    return frozenset(skip)


def _comment_line_numbers(file_path: Path, lines: list[str]) -> frozenset[int]:
    """Lines to skip as comments for the file's actual comment syntax."""
    suffix = file_path.suffix.lower()
    if suffix == ".py":
        return _python_comment_line_numbers("\n".join(lines))
    if suffix in {".yaml", ".yml"}:
        return _yaml_comment_line_numbers(lines)
    if suffix == ".sh":
        return _shell_comment_line_numbers(lines)
    if suffix in {".md", ".mdc", ".markdown"}:
        return _markdown_comment_line_numbers(lines)
    return _leading_hash_or_slash_comment_lines(lines)


def _skillspector_llm_stderr_failed(stderr: str) -> bool:
    """Detect SkillSpector LLM failures hidden behind exit 0 and clean JSON.

    SkillSpector 2.3.7 can catch provider/auth errors, log them to stderr, and
    still return a static-looking report with ``llm_available=true``. For an
    explicitly requested LLM stage, those warnings mean the evidence is
    incomplete regardless of the process exit code or JSON metadata.
    """
    normalized = (stderr or "").casefold()
    return any(marker in normalized for marker in _SKILLSPECTOR_LLM_FAILURE_MARKERS)


def _skillspector_report_llm_provenance(report: object) -> dict | None:
    """Return the validated ``metadata.llm_provenance`` of a SkillSpector JSON report, if any."""
    if not isinstance(report, Mapping):
        return None
    metadata = report.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    return sanitize_skillspector_llm_provenance(metadata.get("llm_provenance"))


# SkillSpector 2.12.0 and later retries transient provider failures, such as rate
# limits, in its own bounded loop and records a fatal
# ``llm_connection_retries_exhausted`` ledger row when that loop gives up. A
# fatal row makes ``execution_successful`` false, and the scan command prints the
# report before it exits with this code. Releases 2.9.1 to 2.11.2 used the same
# reason only for a connection error left after the provider SDK's own retries,
# so their reports keep the generic error.
_SKILLSPECTOR_EXECUTION_FAILED_EXIT_CODE = 2
_SKILLSPECTOR_PROVIDER_RETRY_VERSION = (2, 12, 0)
_SKILLSPECTOR_PROVIDER_RETRY_REASON = "llm_connection_retries_exhausted"
_SKILLSPECTOR_PROVIDER_RETRY_MAX_ROWS = 4096
_SKILLSPECTOR_PROVIDER_RETRY_MAX_MESSAGE = 4096
# The run-level row SkillSpector adds on SKILL.md when requested semantic
# analysis did not finish. It names no analyzer and is not fatal.
_SKILLSPECTOR_SEMANTIC_RUNTIME_ROW = {
    "outcome": "partial",
    "phase": "semantic_runtime",
    "reason_code": "semantic_runtime_incomplete",
    "message": "Requested semantic analysis did not produce complete per-source runtime telemetry.",
    "path": "SKILL.md",
    "start_line": None,
    "end_line": None,
    "fatal": False,
}
_SKILLSPECTOR_REPORT_FAILURE_MARKERS = ("error", "errors", "exceptions", "failed", "failure", "failures")


def _skillspector_provider_retry_row(row: object) -> bool:
    """Return whether one ledger row is a provider-retry exhaustion on LLM analyzers only."""
    if not isinstance(row, dict):
        return False
    phase = row.get("phase")
    path = row.get("path")
    analyzers = row.get("analyzers")
    message = row.get("message")
    return (
        row.get("reason_code") == _SKILLSPECTOR_PROVIDER_RETRY_REASON
        and row.get("outcome") == "failed"
        and row.get("fatal") is True
        and phase in ("semantic", "meta")
        and isinstance(message, str)
        and 0 < len(message) <= _SKILLSPECTOR_PROVIDER_RETRY_MAX_MESSAGE
        and isinstance(path, str)
        and bool(path.strip())
        and isinstance(analyzers, list)
        and bool(analyzers)
        and all(isinstance(analyzer, str) and analyzer in SKILLSPECTOR_LLM_ANALYZERS for analyzer in analyzers)
        and len(set(analyzers)) == len(analyzers)
        # Only the meta analyzer runs in the meta phase.
        and all((analyzer == "meta_analyzer") == (phase == "meta") for analyzer in analyzers)
    )


def _skillspector_flags_truncation(container: Mapping[str, object]) -> bool:
    """Return whether a report object has a ``*_truncated`` key that is neither ``false`` nor ``null``.

    SkillSpector flags a list it cut short this way, as in
    ``ledger_exceptions_truncated``, and what it left out may be a different gap.
    """
    return any(
        isinstance(key, str) and key.endswith("_truncated") and value is not False and value is not None
        for key, value in container.items()
    )


def _skillspector_provider_retries_exhausted(data: object) -> bool:
    """Return whether a failed LLM-stage report shows only an LLM provider outage.

    Every ledger exception must be a fatal ``llm_connection_retries_exhausted``
    row for LLM analyzers, apart from SkillSpector's run-level
    ``semantic_runtime_incomplete`` row on ``SKILL.md``. Exactly those analyzers
    must report ``failed``, every other analyzer must have finished, and every
    limitation must name one of the failed analyzers. Older,
    truncated, malformed or mixed reports return ``False`` and keep the generic
    error, so a scanner fault cannot pass for a provider outage.
    """
    if not isinstance(data, dict) or data.get("execution_successful") is not False:
        return False
    metadata = data.get("metadata")
    raw_version = metadata.get("skillspector_version") if isinstance(metadata, dict) else None
    if not isinstance(raw_version, str) or SEMVER_RE.fullmatch(raw_version) is None:
        return False
    if tuple(int(part) for part in raw_version.split(".")) < _SKILLSPECTOR_PROVIDER_RETRY_VERSION:
        return False
    completeness = data.get("analysis_completeness")
    if (
        not isinstance(completeness, dict)
        or completeness.get("execution_successful") is not False
        or completeness.get("status") != "failed"
        or completeness.get("is_complete") is not False
    ):
        return False
    for container in (data, completeness):
        if any(container.get(key) for key in _SKILLSPECTOR_REPORT_FAILURE_MARKERS):
            return False
        # A truncated list may hide a different cause.
        if _skillspector_flags_truncation(container):
            return False

    rows = completeness.get("ledger_exceptions")
    if not isinstance(rows, list) or not 0 < len(rows) <= _SKILLSPECTOR_PROVIDER_RETRY_MAX_ROWS:
        return False
    provider_analyzers: set[str] = set()
    provider_messages: set[str] = set()
    for row in rows:
        if isinstance(row, dict) and row == _SKILLSPECTOR_SEMANTIC_RUNTIME_ROW and row["fatal"] is False:
            continue
        if not _skillspector_provider_retry_row(row):
            return False
        provider_analyzers.update(row["analyzers"])
        provider_messages.add(row["message"])
    if not provider_analyzers:
        return False

    statuses = completeness.get("analyzer_statuses")
    limitations = completeness.get("limitations", [])
    if not isinstance(statuses, list) or not isinstance(limitations, list):
        return False
    seen: set[str] = set()
    failed: set[str] = set()
    explained: set[str] = set()
    for status in statuses:
        if not isinstance(status, dict) or not isinstance(status.get("analyzer_id"), str):
            return False
        analyzer = status["analyzer_id"]
        if analyzer in seen:
            return False
        seen.add(analyzer)
        state = status.get("status")
        if state in ("completed", "not_applicable") and analyzer not in provider_analyzers:
            continue
        message = status.get("message")
        if (
            state != "failed"
            or analyzer not in provider_analyzers
            or any(status.get(key) for key in ("partial", "skipped", "unaccounted"))
            or status.get("reason_code") not in (None, _SKILLSPECTOR_PROVIDER_RETRY_REASON)
            or (message is not None and (not isinstance(message, str) or message not in provider_messages))
        ):
            return False
        failed.add(analyzer)
        explained.add(f"Analyzer {analyzer} status: failed.")
        if message is not None:
            explained.add(message)
    # Every analyzer named by a provider row must itself report the failure.
    return failed == provider_analyzers and all(
        isinstance(limitation, str) and limitation in explained for limitation in limitations
    )


def _skillspector_provider_retry_error(stage_name: str) -> str:
    """Describe a provider outage in fixed text; no report field is quoted."""
    return (
        f"{stage_name} did not complete: the LLM provider kept failing after SkillSpector's bounded "
        f"retries ({_SKILLSPECTOR_PROVIDER_RETRY_REASON}). This is a retriable provider failure, not a "
        "security finding, and static findings are kept. Retry the full scan when the provider is available."
    )


def _copy_selected_environment(environ: Mapping[str, str], names: Iterable[str]) -> dict[str, str]:
    return {name: environ[name] for name in names if environ.get(name, "").strip()}


def _skillspector_timeout_seconds(environ: Mapping[str, str] | None = None) -> int:
    """Return the SkillSpector subprocess watchdog configured by the operator."""
    source = os.environ if environ is None else environ
    raw = source.get(_SKILLSPECTOR_TIMEOUT_ENV, "").strip()
    if not raw:
        return _DEFAULT_SKILLSPECTOR_TIMEOUT_SECONDS
    try:
        timeout: int | None = int(raw)
    except ValueError:
        timeout = None
    if timeout is None or timeout <= 0:
        logger.warning(
            "Ignoring %s=%r: expected a positive integer number of seconds; using %ss",
            _SKILLSPECTOR_TIMEOUT_ENV,
            raw,
            _DEFAULT_SKILLSPECTOR_TIMEOUT_SECONDS,
        )
        return _DEFAULT_SKILLSPECTOR_TIMEOUT_SECONDS
    if timeout > _MAX_SKILLSPECTOR_TIMEOUT_SECONDS:
        logger.warning(
            "Capping %s=%r at %ss",
            _SKILLSPECTOR_TIMEOUT_ENV,
            raw,
            _MAX_SKILLSPECTOR_TIMEOUT_SECONDS,
        )
        return _MAX_SKILLSPECTOR_TIMEOUT_SECONDS
    return timeout


def _skillspector_process_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    return _copy_selected_environment(source, _SKILLSPECTOR_PROCESS_ENV_NAMES)


def _skillspector_operator_env(environ: Mapping[str, str]) -> dict[str, str]:
    """Return operator-set SkillSpector slot models, model registry and reasoning effort.

    ``SKILLSPECTOR_TEMPERATURE`` stays out: current reasoning models reject a
    custom temperature, which would fail every LLM request.
    """
    return _copy_selected_environment(
        environ,
        (name for name in environ if name.startswith("SKILLSPECTOR_MODEL_") or name == "SKILLSPECTOR_REASONING_EFFORT"),
    )


def _skillspector_provider_env() -> tuple[dict[str, str], str | None]:
    """Map public provider settings into an invocation-scoped SkillSpector environment.

    Also returns the SkillEvaluator provider the bridge resolved. It is ``None``
    when ``SKILLSPECTOR_PROVIDER`` selects a SkillSpector adapter directly or
    names an unsupported provider, and when no provider resolves.
    ``SKILLSPECTOR_PROVIDER=nv_build`` is bridged and returns ``nv_build``.
    """
    source = os.environ
    child_env = _skillspector_process_env(source)
    child_env.update(_skillspector_operator_env(source))
    skillspector_provider = source.get("SKILLSPECTOR_PROVIDER", "").strip().lower()
    skillspector_model = source.get("SKILLSPECTOR_MODEL", "").strip()

    if skillspector_provider:
        if skillspector_provider in _SKILLSPECTOR_EXPLICIT_PROVIDER_ENV:
            child_env["SKILLSPECTOR_PROVIDER"] = skillspector_provider
            if skillspector_model:
                child_env["SKILLSPECTOR_MODEL"] = skillspector_model
            child_env.update(
                _copy_selected_environment(source, _SKILLSPECTOR_EXPLICIT_PROVIDER_ENV[skillspector_provider])
            )
            return child_env, None
        if skillspector_provider != "nv_build":
            # Unsupported/private providers fail closed through the generic
            # public path without receiving any provider credential.
            child_env["SKILLSPECTOR_PROVIDER"] = "openai"
            if skillspector_model:
                child_env["SKILLSPECTOR_MODEL"] = skillspector_model
            return child_env, None

    try:
        resolution_env = dict(source)
        if skillspector_provider == "nv_build":
            resolution_env["SKILL_EVAL_LLM_PROVIDER"] = "nv_build"
            if skillspector_model:
                resolution_env["SKILL_EVAL_LLM_MODEL"] = skillspector_model
        provider = resolve_llm_provider(resolution_env)
    except ProviderConfigurationError:
        return child_env, None

    mapped_provider = _SKILLSPECTOR_PROVIDER_MAP.get(provider.provider)
    if mapped_provider is None:
        return child_env, None

    child_env["SKILLSPECTOR_PROVIDER"] = mapped_provider
    # An explicit SKILL_EVAL_LLM_MODEL reaches SkillSpector; the chat default does not.
    child_env["SKILLSPECTOR_MODEL"] = skillspector_model or (
        provider.model if "SKILL_EVAL_LLM_MODEL" in resolution_env else SKILLSPECTOR_DEFAULT_MODELS[provider.provider]
    )

    if provider.provider in {"nv_build", "openai-compatible"}:
        if provider.api_key:
            child_env["OPENAI_API_KEY"] = provider.api_key
        if provider.base_url:
            child_env["OPENAI_BASE_URL"] = provider.base_url
    else:
        child_env.update(provider.child_environment())
        if provider.provider == "bedrock":
            child_env.update(_copy_selected_environment(source, _SKILLSPECTOR_AWS_ENV_NAMES))

    return child_env, provider.provider


def _skillspector_default_reasoning_effort(child_env: Mapping[str, str]) -> str | None:
    """Return SkillEvaluator's reasoning effort when every configured SkillSpector model takes it.

    The effort is one global SkillSpector setting, so a slot override naming a
    model without one (for example a Claude model) leaves it unset. Without
    ``SKILLSPECTOR_MODEL`` SkillSpector picks its own model and effort.
    """
    if "SKILLSPECTOR_MODEL" not in child_env:
        return None
    efforts = {
        skillspector_reasoning_effort(model)
        for name, model in child_env.items()
        if name.startswith("SKILLSPECTOR_MODEL") and name != "SKILLSPECTOR_MODEL_REGISTRY"
    }
    return efforts.pop() if len(efforts) == 1 else None


def _skillspector_llm_env() -> tuple[dict[str, str], str | None]:
    """Return the LLM-stage SkillSpector environment and the bridged SkillEvaluator provider."""
    child_env, provider_id = _skillspector_provider_env()
    effort = _skillspector_default_reasoning_effort(child_env)
    if effort is not None:
        child_env.setdefault("SKILLSPECTOR_REASONING_EFFORT", effort)
    return child_env, provider_id


def _skillspector_child_env() -> dict[str, str]:
    """Return the SkillSpector environment plus SkillEvaluator's default reasoning effort."""
    return _skillspector_llm_env()[0]


def _tree_contains_artifact_dirs(root: Path) -> bool:
    """Return True when the SkillSpector scan tree needs staging."""
    return any(
        any(d in _SKILLSPECTOR_SCAN_EXCLUDED_DIRS for d in dirnames)
        for _dirpath, dirnames, _filenames in os.walk(root)
    )


def _ignore_artifact_dirs(dirpath: str, names: list[str]) -> set[str]:
    """``shutil.copytree`` ignore hook dropping artifact directories only."""
    return {n for n in names if n in _SKILLSPECTOR_SCAN_EXCLUDED_DIRS and Path(dirpath, n).is_dir()}


_SKILLSPECTOR_SUPPRESSED_RULE_ID_LIMIT = 50

# Shortest run of literal characters that bounds a rule glob. SkillSpector matches
# message and path globs case-insensitively against free text, so a single
# letter such as ``*e*`` matches almost every finding.
_BASELINE_ID_MIN_LITERAL_RUN = 2
_BASELINE_TEXT_MIN_LITERAL_RUN = 4


def _baseline_glob_has_literal_scope(value: object, *, min_run: int = 1) -> bool:
    """Return whether a SkillSpector suppression glob has a literal run of ``min_run`` characters.

    Wildcards, negated classes, and multi-character classes match too broadly
    to bound a rule and break a run; a one-character class such as ``[d]`` is a
    literal. Only letters and digits count toward a run.
    """
    if not isinstance(value, str):
        return False
    pattern = value.strip()
    run = 0
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character in "*?":
            run = 0
            index += 1
            continue
        if character != "[":
            if character.isalnum():
                run += 1
                if run >= min_run:
                    return True
            index += 1
            continue
        cursor = index + 1
        negated = cursor < len(pattern) and pattern[cursor] in "!^"
        if negated:
            cursor += 1
        if cursor < len(pattern) and pattern[cursor] == "]":
            cursor += 1
        close = pattern.find("]", cursor)
        if close < 0:
            # fnmatch treats an unclosed bracket as literal text.
            return len(pattern) - index + run >= min_run
        class_body = pattern[index + 1 : close]
        if not negated and len(class_body) == 1 and class_body.isalnum():
            run += 1
            if run >= min_run:
                return True
        else:
            run = 0
        index = close + 1
    return False


class _SkillSpectorBaselineLimitError(ValueError):
    """A baseline exceeds the size or YAML expansion limits SkillSpector enforces."""


class _BoundedSkillSpectorBaselineLoader(yaml.SafeLoader):
    """SafeLoader that bounds composition and alias or merge expansion like SkillSpector.

    Plain ``yaml.safe_load`` expands chained merge keys without limit, so a
    small file could take minutes or exhaust memory before any check runs.
    Expansion is counted on the composed node graph, revisiting every alias,
    before any value is constructed.
    """

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._composed_nodes = 0
        self._compose_depth = 0

    def compose_node(self, parent: object, index: object) -> yaml.Node:
        self._composed_nodes += 1
        if self._composed_nodes > _SKILLSPECTOR_BASELINE_MAX_NODES:
            raise _SkillSpectorBaselineLimitError("exceeds the YAML node limit")
        if self._compose_depth >= _SKILLSPECTOR_BASELINE_MAX_DEPTH:
            raise _SkillSpectorBaselineLimitError("exceeds the YAML depth limit")
        self._compose_depth += 1
        try:
            return super().compose_node(parent, index)
        finally:
            self._compose_depth -= 1

    def construct_document(self, node: yaml.Node) -> object:
        counts = {"nodes": 0, "characters": 0}
        active: set[int] = set()

        def visit(item: yaml.Node, depth: int) -> None:
            counts["nodes"] += 1
            if counts["nodes"] > _SKILLSPECTOR_BASELINE_MAX_NODES:
                raise _SkillSpectorBaselineLimitError("exceeds the expanded YAML node limit")
            if depth > _SKILLSPECTOR_BASELINE_MAX_DEPTH:
                raise _SkillSpectorBaselineLimitError("exceeds the expanded YAML depth limit")
            if id(item) in active:
                raise _SkillSpectorBaselineLimitError("contains cyclic YAML aliases")
            if isinstance(item, yaml.ScalarNode):
                if len(item.value) > _SKILLSPECTOR_BASELINE_MAX_SCALAR_CHARS:
                    raise _SkillSpectorBaselineLimitError("exceeds the YAML scalar character limit")
                counts["characters"] += len(item.value)
                if counts["characters"] > _SKILLSPECTOR_BASELINE_MAX_BYTES:
                    raise _SkillSpectorBaselineLimitError("exceeds the expanded YAML character limit")
                return
            active.add(id(item))
            children = (
                (child for pair in item.value for child in pair) if isinstance(item, yaml.MappingNode) else item.value
            )
            for child in children:
                visit(child, depth + 1)
            active.remove(id(item))

        # Depth is bounded, so this recursion stays shallow; every alias is
        # revisited because counting unique nodes misses merge amplification.
        visit(node, 1)
        return super().construct_document(node)


def _is_non_empty_text(value: object) -> bool:
    """Return whether a parsed YAML value is a string with visible text."""
    return isinstance(value, str) and bool(value.strip())


def _inspect_skillspector_baseline(path: Path) -> tuple[dict, str | None]:
    """Return the reportable facts of an operator-selected baseline and any blocking error.

    SkillSpector validates the full schema itself. SkillEvaluator rejects a
    file it cannot read, the cheap schema errors SkillSpector would otherwise
    report only as an opaque exit code, and rules whose globs carry no literal
    text, because SkillSpector accepts a rule such as ``id: "*"`` that silently
    suppresses every finding.

    Only the file name and a content digest are recorded, so reports and
    errors do not leak local paths.
    """
    name = path.name
    state: dict = {
        "file": name,
        "digest": None,
        "loaded": False,
        "status": "not_applied",
        "schema_version": None,
        "baseline_scanner_version": None,
        "runtime_scanner_version": None,
        "rule_count": 0,
        "fingerprint_count": 0,
        "suppressed_findings": None,
        "llm_stage_suppressed_findings": None,
        "active_findings": None,
        "active_blocking_findings": None,
        "suppressed_by_severity": {},
        "suppressed_rule_ids": [],
        "message": None,
    }
    if not path.is_file():
        return state, f"SkillSpector baseline {name} does not exist or is not a file"
    try:
        with path.open("rb") as source:
            content = source.read(_SKILLSPECTOR_BASELINE_MAX_BYTES + 1)
    except OSError as exc:
        return state, f"SkillSpector baseline {name} could not be read ({type(exc).__name__})"
    if len(content) > _SKILLSPECTOR_BASELINE_MAX_BYTES:
        return state, f"SkillSpector baseline {name} exceeds the {_SKILLSPECTOR_BASELINE_MAX_BYTES}-byte limit"
    state["digest"] = f"sha256:{hashlib.sha256(content).hexdigest()}"
    try:
        raw = yaml.load(content.decode("utf-8"), Loader=_BoundedSkillSpectorBaselineLoader)
    except _SkillSpectorBaselineLimitError as exc:
        return state, f"SkillSpector baseline {name} {exc}"
    except (
        RecursionError,
        yaml.YAMLError,
        # SafeLoader scalar constructors raise these for values such as an
        # impossible timestamp (``2026-06-31``) or ``!!int ""``.
        ValueError,
        IndexError,
        KeyError,
        AttributeError,
        TypeError,
        OverflowError,
    ) as exc:
        return state, f"SkillSpector baseline {name} could not be parsed ({type(exc).__name__})"
    if raw is None:
        # Like SkillSpector, an empty or comment-only file is an empty baseline.
        raw = {}
    if not isinstance(raw, dict):
        return state, f"SkillSpector baseline {name} must be a YAML or JSON mapping"

    version = raw.get("version")
    rules = raw.get("rules")
    fingerprints = raw.get("fingerprints")
    scanner_version = raw.get("scanner_version")
    state.update(
        {
            "schema_version": version if isinstance(version, int | str) and not isinstance(version, bool) else None,
            "baseline_scanner_version": scanner_version.strip() if _is_non_empty_text(scanner_version) else None,
            "rule_count": len(rules) if isinstance(rules, list) else 0,
            "fingerprint_count": len(fingerprints) if isinstance(fingerprints, list) else 0,
        }
    )
    for field, value in (("rules", rules), ("fingerprints", fingerprints)):
        if value is not None and not isinstance(value, list):
            return state, f"SkillSpector baseline {name} field '{field}' must be a list"
    rule_list = rules or []
    fingerprint_list = fingerprints or []
    # Shape errors first, so a bare string is not reported as an unbounded
    # rule or a missing reason.
    malformed = [f"rule {index}" for index, rule in enumerate(rule_list) if not isinstance(rule, dict)]
    if malformed:
        return state, f"SkillSpector baseline {', '.join(malformed)} in {name} must be a mapping"
    malformed = [f"fingerprint {index}" for index, entry in enumerate(fingerprint_list) if not isinstance(entry, dict)]
    if malformed:
        return state, (
            f"SkillSpector baseline {', '.join(malformed)} in {name} must be a mapping with 'hash' and 'reason'"
        )
    unbounded = [
        index
        for index, rule in enumerate(rule_list)
        # Field aliases match SkillSpector's rule loader.
        if not any(
            _baseline_glob_has_literal_scope(rule.get(field) or rule.get(alias), min_run=min_run)
            for field, alias, min_run in (
                ("id", "rule_id", _BASELINE_ID_MIN_LITERAL_RUN),
                ("path", "file", _BASELINE_TEXT_MIN_LITERAL_RUN),
                ("message", "message", _BASELINE_TEXT_MIN_LITERAL_RUN),
            )
        )
    ]
    if unbounded:
        noun = "rule" if len(unbounded) == 1 else "rules"
        return state, (
            f"SkillSpector baseline {noun} {', '.join(map(str, unbounded))} in {name} must set an id glob with "
            f"{_BASELINE_ID_MIN_LITERAL_RUN} literal characters in a row, or a path or message glob with "
            f"{_BASELINE_TEXT_MIN_LITERAL_RUN}; a broader rule would suppress nearly every finding"
        )
    if version == _SKILLSPECTOR_BASELINE_SCHEMA_VERSION:
        # SkillSpector rejects a version 2 entry without a reason.
        unexplained = [
            f"rule {index}" for index, rule in enumerate(rule_list) if not _is_non_empty_text(rule.get("reason"))
        ]
        unexplained += [
            f"fingerprint {index}"
            for index, entry in enumerate(fingerprint_list)
            if not _is_non_empty_text(entry.get("reason"))
        ]
        if unexplained:
            return state, (
                f"SkillSpector baseline {name} has no reason for {', '.join(unexplained)}; "
                "a version 2 baseline needs a non-empty reason on every rule and fingerprint"
            )
    return state, None


def _skillspector_baseline_for_scan(original_root: Path, scan_root: Path, baseline: Path | None) -> Path | None:
    """Point SkillSpector at the staged copy of a baseline that lives inside the skill.

    SkillSpector excludes only the exact file passed to ``--baseline`` from its
    content scan. On a staged copy, passing the original path would leave the
    copied baseline in scope, and its reviewed match text would be reported as
    new findings. Baselines outside the skill, or not copied into the staged
    tree, keep their original path.
    """
    if baseline is None or scan_root == original_root:
        return baseline
    try:
        relative = baseline.relative_to(original_root)
    except ValueError:
        return baseline
    staged_baseline = scan_root / relative
    return staged_baseline if staged_baseline.is_file() else baseline


def _rewrite_path_prefix(value, old: str, new: str):
    """Recursively rewrite ``old`` path prefixes in a parsed JSON structure."""
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_rewrite_path_prefix(v, old, new) for v in value]
    if isinstance(value, dict):
        return {k: _rewrite_path_prefix(v, old, new) for k, v in value.items()}
    return value


def _combine_skillspector_applicability(blocks: list[dict]) -> dict:
    """Total per-skill SkillSpector applicability counters for a folder of skills."""
    by_role: Counter[str] = Counter()
    for block in blocks:
        by_role.update(block["suppressed_by_file_role"])
    return {
        "artifact_type": blocks[0]["artifact_type"],
        "kept_issues": sum(block["kept_issues"] for block in blocks),
        "suppressed_inapplicable_issues": sum(block["suppressed_inapplicable_issues"] for block in blocks),
        "suppressed_by_file_role": dict(sorted(by_role.items())),
        "raw_risk_score": max(block["raw_risk_score"] for block in blocks),
        "applicable_risk_score": max(block["applicable_risk_score"] for block in blocks),
    }


def _record_skillspector_reference_summary(result: ValidationResult) -> None:
    """Count the referenced artifacts behind *result*'s AE1 evidence in its metadata.

    The counts come from the findings
    (:func:`~skillevaluator.skillspector_references.reference_summary`), so a
    second call after the ``--llm`` stage is merged replaces the first count.
    They are not added to the result's messages: reporters list a passing
    result's messages as passed checks, and they show the counts only for a
    failed result.
    """
    summary = reference_summary(result.findings)
    if summary is None:
        result.metadata.pop(REFERENCE_SUMMARY_KEY, None)
        return
    result.metadata[REFERENCE_SUMMARY_KEY] = summary


def _issue_field(issue: dict):
    """Return a getter that retrieves a field from *issue*, treating None as empty string."""

    def get(key: str, default: str = "") -> str:
        return issue.get(key) or default

    return get


def _skillspector_scoring_source_scope(item: Mapping[str, object]) -> tuple[str, str]:
    """Return the producer's source identity for score reconciliation."""
    for key in ("source_identity", "source_url", "source_digest"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return key, value
    return "", ""


def _skillspector_scoring_match_identity(
    issue: Mapping[str, object],
    index: int,
    *,
    uses_report_identities: bool,
    uses_finding_identity: bool,
) -> tuple[str, str | int]:
    """Return the producer's finding identity used for report compaction."""
    if uses_finding_identity:
        return "finding_id", str(issue["finding_id"])
    fingerprint = issue.get("match_fingerprint")
    if isinstance(fingerprint, str) and fingerprint:
        return "match_fingerprint", fingerprint
    if uses_report_identities:
        finding_id = issue.get("finding_id")
        if isinstance(finding_id, str) and finding_id:
            return "finding_id", finding_id
        return "row", index
    finding = str(issue.get("finding") or "").strip()[:100]
    return ("finding", finding) if finding else ("row", index)


def _skillspector_report_text(value: object) -> str | None:
    """Return untrusted report text as one bounded line, or ``None`` when it is not text."""
    if not isinstance(value, str):
        return None
    # Control, format and separator characters (newlines included) become spaces.
    text = "".join(
        " " if unicodedata.category(character)[0] in {"C", "Z"} else character
        for character in value[: _SKILLSPECTOR_REPORT_TEXT_MAX_CHARS + 1]
    ).strip()
    if len(text) > _SKILLSPECTOR_REPORT_TEXT_MAX_CHARS:
        text = f"{text[: _SKILLSPECTOR_REPORT_TEXT_MAX_CHARS - 3]}..."
    return text or None


def _skillspector_ledger_exception_label(exception: Mapping[str, object]) -> str:
    """Describe one ledger exception as ``path[:line] [phase; reason; analyzers=...]``."""
    location = _skillspector_report_text(exception.get("path")) or "unknown"
    start_line = exception.get("start_line")
    if isinstance(start_line, int) and not isinstance(start_line, bool):
        location = f"{location}:{start_line}"
    context = [text for key in ("phase", "reason_code") if (text := _skillspector_report_text(exception.get(key)))]
    analyzers = exception.get("analyzers")
    if isinstance(analyzers, list) and (
        analyzer_text := _skillspector_report_text(",".join(name for name in analyzers if isinstance(name, str)))
    ):
        context.append(f"analyzers={analyzer_text}")
    return f"{location} [{'; '.join(context)}]" if context else location


def _skillspector_incomplete_files_by_reason(ledger_exceptions: Iterable[Mapping[str, object]]) -> str:
    """Count distinct files per ledger reason code, most common first."""
    paths_by_reason: dict[str, set[str]] = {}
    for exception in ledger_exceptions:
        reason = _skillspector_report_text(exception.get("reason_code")) or "unknown"
        path = exception.get("path")
        paths_by_reason.setdefault(reason, set()).add(path if isinstance(path, str) else "")
    counts = Counter({reason: len(paths) for reason, paths in paths_by_reason.items()})
    return ", ".join(f"{reason} {count}" for reason, count in counts.most_common())


def _skillspector_runtime_limit_hint(ledger_exceptions: Iterable[Mapping[str, object]]) -> str | None:
    """Explain how to clear ``runtime_limit`` ledger exceptions, or ``None`` when there are none."""
    scan_limited: set[str] = set()
    budget_limited: set[str] = set()
    for exception in ledger_exceptions:
        if exception.get("reason_code") != "runtime_limit":
            continue
        path = exception.get("path")
        analyzers = exception.get("analyzers")
        # A sub-analysis that stops at its own time budget is reported as
        # "<analyzer>_runtime_limit" (for example the SC8 bytecode walk). That
        # budget is independent of the whole-scan deadline.
        if (
            isinstance(analyzers, list)
            and analyzers
            and all(isinstance(name, str) and name.endswith("_runtime_limit") for name in analyzers)
        ):
            budget_limited.add(path if isinstance(path, str) else "")
        else:
            scan_limited.add(path if isinstance(path, str) else "")
    hints: list[str] = []
    if scan_limited:
        # These rows come from SkillSpector's whole-scan deadline or from a shorter
        # stage budget such as bundle discovery, and the report does not say which,
        # so the hint names no specific limit.
        hints.append(
            f"SkillSpector reached a runtime limit before it finished {len(scan_limited)} file(s). "
            "When many files are affected, the skill most likely bundles more content than SkillSpector can "
            "inspect within its time budgets, so a skill that bundles a large corpus (vendored documents, "
            "generated figures, renderer scripts) hits the same limit on every rerun; keep large reference "
            "material outside the skill and point to it. When only a few files are affected on a busy machine, "
            "a plain rerun may complete."
        )
    if budget_limited:
        # Reported alongside the scan-budget advice too, so every runtime_limit file
        # counted by reason is covered by one of the two explanations.
        hints.append(
            f"A SkillSpector sub-analysis stopped at its own time budget on {len(budget_limited)} file(s). "
            "That budget is separate from the whole-scan limit, so reducing bundled content does not change it. "
            "Rerun the scan; if the limit repeats, update SkillSpector or report the completeness exceptions "
            "to the SkillSpector project."
        )
    return " ".join(hints) or None


_SKILLSPECTOR_REFERENCE_REJECTION_HINT = (
    "Missing references require a standalone target with complete, matching reference and scan evidence. "
    "See https://docs.nvidia.com/skills/skillevaluator/tier1-validation."
)


def _skillspector_incomplete_analysis_error(status: str, ledger_exceptions: list[dict]) -> str:
    """Describe a partial SkillSpector analysis with example gaps and, for runtime limits, a remedy."""
    message = f"skillspector JSON field 'analysis_completeness' reports incomplete analysis (status '{status}')"
    if ledger_exceptions:
        shown = ledger_exceptions[:_SKILLSPECTOR_INCOMPLETE_EXAMPLES]
        message = f"{message}: {'; '.join(_skillspector_ledger_exception_label(row) for row in shown)}"
        remaining = len(ledger_exceptions) - len(shown)
        if remaining:
            # A few examples cannot show why hundreds of files were cut short.
            message = (
                f"{message}; and {remaining} more "
                f"(incomplete files by reason: {_skillspector_incomplete_files_by_reason(ledger_exceptions)})"
            )
    message = f"{message}; security scan did not complete"
    hint = _skillspector_runtime_limit_hint(ledger_exceptions)
    if not hint and ledger_exceptions and all(row.get("phase") == "reference_resolution" for row in ledger_exceptions):
        hint = _SKILLSPECTOR_REFERENCE_REJECTION_HINT
    return f"{message}. {hint}" if hint else message


# SkillSpector's static pattern analyzers read text, so each records a referenced
# PNG or JPEG image as a nonfatal, whole-file limitation and the analysis becomes
# ``partial``; see :func:`_skillspector_passive_raster_note`.
_PASSIVE_RASTER_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
_PASSIVE_RASTER_REASONS = frozenset({"binary_content", "opaque_content"})
_PASSIVE_RASTER_ANALYZER_RE = re.compile(r"static_patterns_[a-z0-9]+(?:_[a-z0-9]+)*")
# An image is not a passive asset when it sits in a directory whose files an
# agent runs or loads as instructions (at any depth), or under a top-level
# configuration, nested-skill or agent-settings directory.
_PASSIVE_RASTER_CONTROL_DIRS = frozenset(
    {"agents", "commands", "hooks", "prompts", "rules", "scripts", "team-rules", "tools", "workflows"}
)
_PASSIVE_RASTER_CONTROL_ROOTS = frozenset(
    {".agents", ".claude", ".codex", ".continue", ".cursor", ".gemini", ".mcp", ".windsurf", "config", "skills"}
)
_PASSIVE_RASTER_MAX_FILES = 256
_PASSIVE_RASTER_MAX_BYTES = 128 * 1024 * 1024
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# Bit depths the PNG specification allows for each colour type.
_PNG_BIT_DEPTHS = {0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8}, 4: {8, 16}, 6: {8, 16}}
# JPEG start-of-frame markers SOF0-SOF3, SOF5-SOF7, SOF9-SOF11 and SOF13-SOF15.
_JPEG_FRAME_MARKERS = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})
_JPEG_FILL_BYTES = re.compile(rb"\xff+")
# Where entropy-coded data ends: the first marker, after any fill bytes, other
# than a stuffed 0xFF00 byte or a restart marker (RST0-RST7). The lookbehind
# starts each match at the first 0xFF of a run, which keeps the search linear.
_JPEG_SCAN_END = re.compile(rb"(?<!\xff)\xff+[^\x00\xd0-\xd7\xff]")
# A non-marker byte or a stuffed 0xFF00 is entropy-coded content; fill bytes
# and restart markers by themselves do not supply scan data.
_JPEG_SCAN_DATA = re.compile(rb"(?<!\xff)[^\xff]|\xff\x00")


def _is_png(data: bytes) -> bool:
    """Whether *data* has the chunk structure of a PNG image.

    The chunk stream must be whole, with every CRC valid: IHDR first, with a
    nonzero size and a valid bit depth and colour type; a palette exactly when
    the colour type needs or allows one, before the image data; one contiguous
    run of IDAT chunks that starts a zlib stream; and IEND last, with nothing
    after it. Unknown critical chunks and chunk types with the reserved bit set
    are rejected. This corroborates the file type without decoding the image.
    The compressed IDAT data beyond its two-byte zlib header and ancillary
    chunks are not inspected, so both can still carry arbitrary bytes.
    """
    if not data.startswith(_PNG_SIGNATURE):
        return False
    view = memoryview(data)
    offset = len(_PNG_SIGNATURE)
    depth = colour = palette_entries = 0
    idat_head = b""
    idat_state = 0  # 0: no IDAT yet, 1: inside the IDAT run, 2: after it
    while offset + 12 <= len(data):
        (length,) = struct.unpack_from(">I", data, offset)
        kind = data[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(data) or not kind.isalpha() or kind[2] & 0x20:
            return False
        (crc,) = struct.unpack_from(">I", data, end - 4)
        if zlib.crc32(view[offset + 4 : end - 4]) != crc or (offset == len(_PNG_SIGNATURE)) != (kind == b"IHDR"):
            return False
        if kind == b"IHDR":
            if length != 13:
                return False
            width, height, depth, colour, compression, filtering, interlace = struct.unpack_from(
                ">IIBBBBB", data, offset + 8
            )
            if (
                not 0 < width < 1 << 31
                or not 0 < height < 1 << 31
                or depth not in _PNG_BIT_DEPTHS.get(colour, ())
                or compression != 0
                or filtering != 0
                or interlace not in (0, 1)
            ):
                return False
        elif kind == b"PLTE":
            if palette_entries or idat_state or not 0 < length <= 768 or length % 3 or colour in (0, 4):
                return False
            palette_entries = length // 3
        elif kind == b"IDAT":
            if idat_state == 2:
                return False
            idat_state = 1
            idat_head = (idat_head + data[offset + 8 : offset + 8 + min(length, 2)])[:2]
        elif kind == b"IEND":
            # The zlib header names deflate with a window of at most 32 KiB,
            # its check bits are valid, and it sets no preset dictionary.
            return (
                length == 0
                and end == len(data)
                and len(idat_head) == 2
                and idat_head[0] & 0x0F == 8
                and idat_head[0] >> 4 <= 7
                and int.from_bytes(idat_head, "big") % 31 == 0
                and not idat_head[1] & 0x20
                and (colour != 3 or 0 < palette_entries <= 1 << depth)
            )
        elif not kind[0] & 0x20:
            return False  # An unknown critical chunk.
        elif idat_state:
            idat_state = 2
        offset = end
    return False


def _jpeg_tables_are_valid(marker: int, payload: bytes) -> bool:
    """Whether a DQT (0xDB), DHT (0xC4) or DAC (0xCC) segment payload holds whole, valid tables."""
    if marker == 0xCC:
        return bool(payload) and len(payload) % 2 == 0
    cursor = 0
    while cursor < len(payload):
        table_class, table_id = payload[cursor] >> 4, payload[cursor] & 0x0F
        if marker == 0xDB:
            cursor += 1 + 64 * (table_class + 1)
            valid = table_class <= 1
        else:
            symbols = sum(payload[cursor + 1 : cursor + 17])
            cursor += 17 + symbols
            valid = table_class <= 1 and 0 < symbols <= 256
        if not valid or table_id > 3 or cursor > len(payload):
            return False
    return bool(payload)


def _is_jpeg(data: bytes) -> bool:
    """Whether *data* has the marker structure of a JPEG image.

    The file must run from SOI to EOI, its last bytes, through well-formed
    marker segments: one frame header (SOF); valid quantization (DQT) and
    entropy-coding (DHT or DAC) tables before the first scan; and scans (SOS)
    whose components belong to the frame, each followed by entropy-coded
    data. Reserved markers and markers out of place are rejected. This
    corroborates the file type without decoding the image. Entropy-coded
    data and APPn or COM segments are not inspected and can still carry
    arbitrary bytes.
    """
    if len(data) < 20 or not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        return False
    frame: set[int] = set()
    has_quantization = has_entropy_tables = False
    entropy_bytes = 0
    offset = 2
    while offset < len(data):
        if data[offset] != 0xFF:
            return False
        fill = _JPEG_FILL_BYTES.match(data, offset)
        offset = fill.end() if fill else offset
        if offset >= len(data):
            return False
        marker = data[offset]
        offset += 1
        if marker == 0xD9:
            return offset == len(data) and entropy_bytes > 0
        if marker < 0xC0 or 0xD0 <= marker <= 0xD8 or offset + 2 > len(data):
            return False
        (length,) = struct.unpack_from(">H", data, offset)
        payload = data[offset + 2 : offset + length]
        offset += length
        if length < 2 or offset > len(data):
            return False
        if marker in (0xDB, 0xC4, 0xCC):
            if not _jpeg_tables_are_valid(marker, payload):
                return False
            has_quantization = has_quantization or marker == 0xDB
            has_entropy_tables = has_entropy_tables or marker != 0xDB
        elif marker in _JPEG_FRAME_MARKERS:
            count = payload[5] if len(payload) > 5 else 0
            if frame or not 1 <= count <= 4 or len(payload) != 6 + 3 * count or payload[0] not in (8, 12):
                return False
            if not int.from_bytes(payload[1:3], "big") or not int.from_bytes(payload[3:5], "big"):
                return False
            for component in range(6, len(payload), 3):
                identifier, sampling, table = payload[component : component + 3]
                if identifier in frame or not 1 <= sampling >> 4 <= 4 or not 1 <= sampling & 0x0F <= 4 or table > 3:
                    return False
                frame.add(identifier)
        elif marker == 0xDA:
            count = payload[0] if payload else 0
            if not (frame and has_quantization and has_entropy_tables and 1 <= count <= len(frame)):
                return False
            if len(payload) != 4 + 2 * count:
                return False
            scanned = payload[1 : 1 + 2 * count : 2]
            selectors = payload[2 : 2 + 2 * count : 2]
            start, stop, approximation = payload[-3:]
            if (
                len(set(scanned)) != count
                or not set(scanned) <= frame
                or any(selector >> 4 > 3 or selector & 0x0F > 3 for selector in selectors)
                or not start <= stop <= 63
                or approximation >> 4 > 13
                or approximation & 0x0F > 13
            ):
                return False
            scan_end = _JPEG_SCAN_END.search(data, offset)
            if scan_end is None:
                return False
            if not _JPEG_SCAN_DATA.search(data, offset, scan_end.start()):
                return False
            entropy_bytes += scan_end.start() - offset
            offset = scan_end.start()
        elif marker == 0xDD:
            if length != 4:
                return False
        elif marker != 0xFE and not 0xE0 <= marker <= 0xEF:
            return False  # Reserved extensions and segments outside this single-frame format.
    return False


def _skillspector_passive_raster_note(
    ledger_exceptions: list[dict],
    analyzer_statuses: list[dict],
    counts: Mapping[str, int],
    limitations: list[str],
    scope_exclusions: object,
    components: object,
    scanned_root: Path | None,
) -> str | None:
    """A note for a ``partial`` SkillSpector analysis whose only gaps are referenced images, else ``None``.

    SkillSpector's ``static_patterns_*`` analyzers read text. A PNG or JPEG
    image that a skill references (``![logo](assets/logo.png)``) is recorded as
    one nonfatal, whole-file ``static`` ledger exception (``opaque_content`` or
    ``binary_content``) naming those analyzers, and the analysis is reported as
    ``partial``. Every other analyzer still inspected every file.

    That gap is accepted only when it accounts exactly for the report: each
    named analyzer is ``degraded`` by one ``partial`` work item per image and
    completed the rest, every other analyzer completed, the uninspected-file
    counts and limitations match, no scope exclusion is fatal, and each image
    is a non-executable component outside the directories that agents run,
    load as instructions or read as configuration
    (:data:`_PASSIVE_RASTER_CONTROL_DIRS`, :data:`_PASSIVE_RASTER_CONTROL_ROOTS`).
    Each file under *scanned_root* must also be a regular, single-link,
    non-executable file, reached without following a link, whose bytes have the
    structure of a PNG or JPEG matching its suffix (:func:`_is_png`,
    :func:`_is_jpeg`), within :data:`_PASSIVE_RASTER_MAX_FILES` files and
    :data:`_PASSIVE_RASTER_MAX_BYTES` bytes. A binary payload named
    ``logo.png`` without PNG structure, another reason, a fatal or line-scoped
    exception, or any accounting mismatch keeps the scan INCOMPLETE. Compressed
    image data is not decoded or assessed, including arbitrary payload bytes
    behind an otherwise well-formed header.
    """
    if (
        scanned_root is None
        or not ledger_exceptions
        or not isinstance(components, list)
        or not isinstance(scope_exclusions, list)
        or any(not isinstance(row, dict) or row.get("fatal") is not False for row in scope_exclusions)
    ):
        return None
    named: Counter[str] = Counter()
    rows: set[tuple[str, str]] = set()
    for exception in ledger_exceptions:
        path = exception.get("path")
        reason = exception.get("reason_code")
        analyzers = exception.get("analyzers")
        if (
            exception.get("outcome") != "partial"
            or exception.get("phase") != "static"
            or exception.get("fatal") is not False
            or exception.get("start_line") is not None
            or exception.get("end_line") is not None
            or not isinstance(reason, str)
            or reason not in _PASSIVE_RASTER_REASONS
            or not isinstance(path, str)
            or PurePosixPath(path).suffix.lower() not in _PASSIVE_RASTER_SUFFIXES
            or not isinstance(analyzers, list)
            or not analyzers
        ):
            return None
        for analyzer in analyzers:
            if (
                not isinstance(analyzer, str)
                or not _PASSIVE_RASTER_ANALYZER_RE.fullmatch(analyzer)
                or (path, analyzer) in rows
            ):
                return None
            rows.add((path, analyzer))
            named[analyzer] += 1
    paths = sorted({path for path, _analyzer in rows})
    total = counts["total_components"]
    uninspected = counts["partially_inspected_files"] + counts["entirely_uninspected_files"]
    if (
        len(paths) > _PASSIVE_RASTER_MAX_FILES
        or uninspected != len(paths)
        or counts["findings_before_filtering"] != counts["findings_after_filtering"]
    ):
        return None

    universal = _SKILLSPECTOR_COMMON_UNIVERSAL_ANALYZERS | {"artifact_integrity"} | set(named)
    accounted: set[str] = set()
    for status in analyzer_statuses:
        analyzer_id = status.get("analyzer_id")
        state = status.get("status")
        if analyzer_id not in universal:
            if state not in {"completed", "not_applicable", "disabled"}:
                return None
            continue
        gaps = named.get(analyzer_id, 0)
        if (
            analyzer_id in accounted
            or state != ("degraded" if gaps else "completed")
            or status.get("planned_work") != total
            or status.get("completed") != total - gaps
            or status.get("partial") != gaps
        ):
            return None
        accounted.add(analyzer_id)
    allowed_limitations = {f"Analyzer {analyzer} status: degraded." for analyzer in named}
    if accounted != universal or any(limitation not in allowed_limitations for limitation in limitations):
        return None

    components_by_path: dict[str, list[dict]] = {}
    for component in components:
        if isinstance(component, dict) and isinstance(component.get("path"), str):
            components_by_path.setdefault(component["path"], []).append(component)
    budget = _PASSIVE_RASTER_MAX_BYTES
    try:
        with SecureRoot(scanned_root) as root:
            for path in paths:
                matches = components_by_path.get(path, [])
                parts = path.split("/")
                directories = [part.casefold() for part in parts[:-1]]
                if (
                    len(matches) != 1
                    or matches[0].get("executable") is not False
                    or "\\" in path
                    or not _is_filesystem_safe_path_text(path)
                    or any(part in {"", ".", ".."} or Path(part).anchor for part in parts)
                    or (directories and directories[0] in _PASSIVE_RASTER_CONTROL_ROOTS)
                    or not _PASSIVE_RASTER_CONTROL_DIRS.isdisjoint(directories)
                ):
                    return None
                data, metadata = root.read_bytes(Path(*parts), budget)
                is_image = _is_png(data) if path.lower().endswith(".png") else _is_jpeg(data)
                if metadata.st_mode & 0o111 or not is_image:
                    return None
                budget -= len(data)
    except (OSError, SecurePathError):
        return None

    shown = ", ".join(paths[:5]) + (f" and {len(paths) - 5} more" if len(paths) > 5 else "")
    return (
        f"SkillSpector's text pattern analyzers could not read {len(paths)} referenced image file(s) "
        f"({shown}); each is a regular, non-executable file with PNG or JPEG structure outside agent-control "
        "directories, so the scan is not marked incomplete. The image content itself was not assessed"
    )


# SkillSpector 2.12.0 split ``reference_unresolved``: a path-like reference that
# matches no bundled file became ``reference_missing``, and ``reference_unresolved``
# now means a reference that matches several bundled files.
_SKILLSPECTOR_REFERENCE_MISSING_VERSION = (2, 12, 0)
# SkillSpector inspects at most 1,000,000 bytes of a file, so a larger file
# holding a reference is never fully inspected and is not accepted.
_SKILLSPECTOR_REFERENCE_SOURCE_MAX_BYTES = 1_000_000
# Entries listed under the skill, and characters of a reference's line, beyond
# which a missing reference is no longer accepted.
_SKILLSPECTOR_REFERENCE_TREE_MAX_ENTRIES = 50_000
_SKILLSPECTOR_REFERENCE_LINE_MAX_CHARS = 10_000
# Markdown escapes of ASCII punctuation, and shell-escaped spaces, can hide
# a path's spelling. SkillSpector itself removes only ``\\``, ``\(``, ``\)``,
# ``\<`` and ``\>``.
_SKILLSPECTOR_MARKDOWN_ESCAPE_RE = re.compile(r"\\([ !-/:-@\[-`{-~])")
# Removing Markdown escapes, resolving HTML character references and decoding
# percent-escapes can each reveal another; a reference's line is also read
# after up to this many of them, in any order.
_SKILLSPECTOR_REFERENCE_LINE_DECODINGS = 3
# Characters that continue a file name in SkillSpector's reference grammar.
_SKILLSPECTOR_NAME_RUN_RE = re.compile(r"[\w.-]+")
# A word of a reference's line: text between whitespace and the Markdown or
# shell delimiters around a path.
_SKILLSPECTOR_LINE_WORD_RE = re.compile(r"[^\s`()<>|;&]+")
# Stands for any text where a shell group or range was read in a line
# (:func:`_skillspector_shell_groups`); unlike ``*``, a word made of it is
# never taken for a list marker.
_SKILLSPECTOR_SHELL_ANY = "\x00"
# Follows the one character of a zsh pattern ``x#`` that is a whole path
# part, such as ``C#``, which matches only names made of that character
# (:func:`_skillspector_zsh_pattern`, :func:`_skillspector_glob_matcher`).
_SKILLSPECTOR_ZSH_REPEAT = "\x01"
# Characters with which a shell, cmd.exe or Windows expands, quotes or escapes
# a word, so that it can open a file that the word does not spell out.
_SKILLSPECTOR_SHELL_SYNTAX = frozenset("*?[{$\"'\\^%!#~" + _SKILLSPECTOR_SHELL_ANY)
# A parameter expansion such as ``$name`` or ``$1``, which may stand for any text.
_SKILLSPECTOR_SHELL_PARAMETER_RE = re.compile(r"\$[\w@*#?!$-]*")
_SKILLSPECTOR_SHELL_QUOTE_RE = re.compile(r"[\"'\\]")
# The quotes and the ``^`` escape that cmd.exe or PowerShell remove.
_SKILLSPECTOR_CMD_QUOTE_RE = re.compile(r"[\"'^]")
# cmd.exe expansions: a batch parameter such as ``%1``, ``%*`` or ``%~dp0``, a
# variable such as ``%DIR%`` and, with delayed expansion, ``!DIR!``.
_SKILLSPECTOR_CMD_VARIABLE_RE = re.compile(r"%~[^%/]*?[0-9*]|%[0-9*]|%[^%]*%|![^!]*!")
# A zsh numeric range such as ``<1-10>`` or ``<->``, which matches any number.
_SKILLSPECTOR_SHELL_RANGE_RE = re.compile(r"<\d*-\d*>")
# A character that opens, closes or splits a shell group, or a run of
# characters that end a word.
_SKILLSPECTOR_SHELL_GROUP_TOKEN_RE = re.compile(r"[()|]|[\s/`<>;&]+")
# Characters that make the group after them a bash or ksh extended glob.
_SKILLSPECTOR_EXTGLOB_OPERATORS = frozenset("@!+?*")
# A zsh EXTENDED_GLOB globbing flag group, such as ``(#i)``, ``(#a2)``,
# ``(#c1,2)``, ``(#s)`` or ``(#q.)``, which changes how the rest of its word
# matches; ``(#1)`` or ``(#setup)`` is not one. A run of characters other than
# whitespace that holds one is a word that may stand for any text.
_SKILLSPECTOR_ZSH_FLAGS_RE = re.compile(r"\(#(?:q|[se]\)|c\d*(?:,\d*)?\)|(?:[ilIbBmMuU]|a\d+)+\))")
_SKILLSPECTOR_NON_SPACE_RUN_RE = re.compile(r"\S+")
_SKILLSPECTOR_GLOB_SYNTAX = frozenset("*?[" + _SKILLSPECTOR_ZSH_REPEAT)
_SKILLSPECTOR_PATH_PART_SEPARATOR_RE = re.compile(r"[=:,]")
# Work that checking the lines of one report's missing references may take, in
# units of about one character compared: decoding a line and reading its shell
# groups and words costs 9,000 per character, each search of a decoded line
# for a name 100 plus the line's length, and each pattern 20,000 plus 1,000 per
# character and 300 plus the name's length for each name matched against it.
# Beyond the budget the lines are not accepted, which bounds the time that a
# skill built to slow the check can cost.
_SKILLSPECTOR_REFERENCE_MATCH_BUDGET = 1_000_000_000


# The files and directories that Git itself writes at the top of a ".git"
# directory (gitrepository-layout(5), and the state files of merge, rebase,
# cherry-pick, revert and bisect), with the lock files Git takes on them.
_SKILLSPECTOR_GIT_FILE_RE = re.compile(
    r"(?:HEAD|[A-Z_]+_HEAD|config|config\.worktree|description|index|packed-refs|shallow|commondir|"
    r"COMMIT_EDITMSG|TAG_EDITMSG|SQUASH_MSG|AUTO_MERGE|MERGE_(?:MSG|MODE|RR)|BISECT_[A-Z_]+|"
    r"gitk\.cache|sharedindex\.[0-9a-f]+)(?:\.lock)?"
)
_SKILLSPECTOR_GIT_DIRECTORIES = frozenset(
    {
        "branches",
        "common",
        "info",
        "logs",
        "modules",
        "objects",
        "rebase-apply",
        "rebase-merge",
        "refs",
        "remotes",
        "rr-cache",
        "sequencer",
        "worktrees",
    }
)


def _skillspector_report_path_key(path: object) -> str | None:
    """A SkillSpector report *path* that is relative POSIX text, else ``None``.

    The path must be text that is safe to pass to OS path APIs, with no
    backslash, no empty, ``.`` or ``..`` part and no anchor. Other spellings
    of the same file are not normalized, so they match nothing.
    """
    if (
        not isinstance(path, str)
        or "\\" in path
        or not _is_filesystem_safe_path_text(path)
        or any(part in {"", ".", ".."} or Path(part).anchor for part in path.split("/"))
    ):
        return None
    return path


def _skillspector_bundled_file_bytes(path: object, components: object, scanned_root: Path | None) -> bytes | None:
    """The bytes of the scanned file a SkillSpector report *path* names, else ``None``.

    The path must pass :func:`_skillspector_report_path_key`, and exactly one
    ``components`` entry must carry it. Under a directory *scanned_root* it
    must name a regular, single-link file of at most
    :data:`_SKILLSPECTOR_REFERENCE_SOURCE_MAX_BYTES` bytes, reached without
    following a link at any component (:class:`SecureRoot`); a single-file
    *scanned_root* matches only its own name. Without a root nothing is
    accepted.
    """
    if scanned_root is None or not isinstance(path, str) or not isinstance(components, list):
        return None
    parts = path.split("/")
    if (
        _skillspector_report_path_key(path) is None
        or [component.get("path") for component in components if isinstance(component, dict)].count(path) != 1
    ):
        return None
    try:
        root_metadata = scanned_root.lstat()
        if stat.S_ISREG(root_metadata.st_mode) and not stat_is_link_or_reparse(root_metadata):
            if path != scanned_root.name:
                return None
            directory, relative = scanned_root.parent, Path(path)
        else:
            directory, relative = scanned_root, Path(*parts)
        with SecureRoot(directory) as root:
            data, _metadata = root.read_bytes(relative, _SKILLSPECTOR_REFERENCE_SOURCE_MAX_BYTES)
    except (OSError, SecurePathError):
        return None
    return data


def _skillspector_report_path_is_bundled_file(path: object, components: object, scanned_root: Path | None) -> bool:
    """Whether a SkillSpector report *path* names one scanned component that is a regular file.

    The rules are those of :func:`_skillspector_bundled_file_bytes`.
    """
    return _skillspector_bundled_file_bytes(path, components, scanned_root) is not None


def _skillspector_reference_row_is_advisory(
    row: Mapping[str, object],
    skillspector_version: tuple[int, int, int] | None,
    references: object,
) -> bool:
    """Whether a ledger row only records a reference to a file the skill does not bundle.

    The row must be a nonfatal ``partial`` row in phase ``reference_resolution``
    that names no analyzer and no error class and covers one line. SkillSpector
    2.12.0 and later report such a reference as ``reference_missing``; there
    ``reference_unresolved`` means a reference that matches several bundled
    files, which is not advisory. Earlier releases report both cases as
    ``reference_unresolved``. Either way the report's ``references`` list must
    corroborate the row: it records at least one reference on the row's line,
    and every unresolved reference there is ``missing``, none ``ambiguous``.
    """
    if (
        row.get("fatal") is not False
        or row.get("outcome") != "partial"
        or row.get("phase") != "reference_resolution"
        or row.get("analyzers") not in (None, [])
        or row.get("error_class") is not None
    ):
        return False
    reason = row.get("reason_code")
    if reason != "reference_missing" and (
        reason != "reference_unresolved"
        or skillspector_version is None
        or skillspector_version >= _SKILLSPECTOR_REFERENCE_MISSING_VERSION
    ):
        return False
    line = row.get("start_line")
    end_line = row.get("end_line")
    if (
        not isinstance(references, list)
        or isinstance(line, bool)
        or not isinstance(line, int)
        or isinstance(end_line, bool)
        or end_line != line
    ):
        return False
    statuses = {
        reference.get("status")
        for reference in _skillspector_references_on_line(references, row.get("path"), line)
        if reference.get("status") in {"missing", "ambiguous"}
    }
    return statuses == {"missing"}


def _skillspector_references_on_line(references: list, path: object, line: int) -> list[dict]:
    """The entries of a report's ``references`` list for line *line* of *path*."""
    return [
        reference
        for reference in references
        if isinstance(reference, dict)
        and reference.get("source_path") == path
        and not isinstance(reference.get("line"), bool)
        and reference.get("line") == line
    ]


def _skillspector_unscanned_paths(scanned_root: Path, components: object) -> set[str] | None:
    """Relative paths of entries that SkillSpector did not inspect, else ``None``.

    An entry is uninspected when it is a link, a junction or another
    non-regular file, a regular file that no ``components`` entry lists, or a
    directory that holds no listed file, such as an artifact directory left out
    of a staged scan, with everything inside it. Links are not followed.

    A ``.git`` directory that holds no listed file counts by its own name, by
    every entry outside Git's own layout (:data:`_SKILLSPECTOR_GIT_FILE_RE`,
    :data:`_SKILLSPECTOR_GIT_DIRECTORIES`) with everything inside it, and by
    the entries of its ``hooks`` directory other than regular ``*.sample``
    files, which Git never runs. Git writes its layout, and names such as
    ``config``, ``HEAD`` or a branch name such as ``main`` are common words, so
    those files and what Git's directories hold do not count.

    SkillSpector scans a single-file *scanned_root* on its own, so every other
    entry of the directory that holds it is uninspected.

    ``None`` means the tree could not be listed within
    :data:`_SKILLSPECTOR_REFERENCE_TREE_MAX_ENTRIES` entries and
    :data:`MAX_SECURE_DIRECTORY_DEPTH` levels.
    """
    if not isinstance(components, list):
        return None
    listed = {
        component["path"]
        for component in components
        if isinstance(component, dict) and isinstance(component.get("path"), str)
    }
    try:
        root_metadata = scanned_root.lstat()
    except OSError:
        return None
    if stat.S_ISREG(root_metadata.st_mode) and not stat_is_link_or_reparse(root_metadata):
        listed &= {scanned_root.name}
        scanned_root = scanned_root.parent
    ancestors = {
        "/".join(parts[:index]) for parts in (path.split("/") for path in listed) for index in range(1, len(parts))
    }
    uninspected: set[str] = set()
    # Each pending directory carries its path relative to the root, whether it
    # holds a listed file, and whether it is a left-out ".git" or its "hooks".
    pending: list[tuple[Path, str, bool, str]] = [(scanned_root, "", True, "")]
    seen = 0
    while pending:
        directory, prefix, holds_listed, git_part = pending.pop()
        if prefix.count("/") > MAX_SECURE_DIRECTORY_DEPTH:
            return None
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    seen += 1
                    if seen > _SKILLSPECTOR_REFERENCE_TREE_MAX_ENTRIES:
                        return None
                    relative = prefix + entry.name
                    metadata = entry.stat(follow_symlinks=False)
                    linked = stat_is_link_or_reparse(metadata)
                    is_directory = stat.S_ISDIR(metadata.st_mode) and not linked
                    is_file = stat.S_ISREG(metadata.st_mode) and not linked
                    if git_part == ".git":
                        if entry.name == "hooks" and is_directory:
                            pending.append((Path(entry.path), relative + "/", False, "hooks"))
                            continue
                        if (is_directory and entry.name in _SKILLSPECTOR_GIT_DIRECTORIES) or (
                            is_file and _SKILLSPECTOR_GIT_FILE_RE.fullmatch(entry.name)
                        ):
                            continue
                    if git_part == "hooks" and is_file and entry.name.endswith(".sample"):
                        continue
                    if is_directory:
                        holds = holds_listed and relative in ancestors
                        if not holds:
                            uninspected.add(relative)
                        git_root = not holds and entry.name == ".git"
                        pending.append((Path(entry.path), relative + "/", holds, ".git" if git_root else ""))
                    elif not is_file or relative not in listed:
                        uninspected.add(relative)
        except OSError:
            return None
    return uninspected


def _skillspector_unscanned_names(scanned_root: Path, components: object) -> set[str] | None:
    """Every path part of an uninspected entry, or ``None`` when the bounded walk fails.

    A directory that holds an uninspected entry is named even when it also
    holds listed files. The paths come from :func:`_skillspector_unscanned_paths`.
    """
    paths = _skillspector_unscanned_paths(scanned_root, components)
    return None if paths is None else {part for relative in paths for part in relative.split("/")}


def _skillspector_name_keys(names: Iterable[str]) -> frozenset[str]:
    """Case-folded forms of *names*, as written and in NFC, for :func:`_skillspector_reference_line_avoids`."""
    return frozenset(
        form for name in names for form in (name.casefold(), unicodedata.normalize("NFC", name).casefold())
    )


class _SkillspectorMatchBudget:
    """The work left for checking the lines of one report's missing references."""

    __slots__ = ("remaining",)

    def __init__(self, remaining: int | None = None) -> None:
        self.remaining = _SKILLSPECTOR_REFERENCE_MATCH_BUDGET if remaining is None else remaining

    def spend(self, cost: int) -> bool:
        """Take *cost* from the budget; ``False`` once it is overdrawn."""
        self.remaining -= cost
        return self.remaining >= 0


@functools.lru_cache(maxsize=8)
def _skillspector_key_kinds(keys: frozenset[str]) -> tuple[Mapping[str, int], tuple[str, ...], int, int]:
    """*keys* split into plain name runs and other names, with their total and longest length.

    Each name run is given by its text before any trailing dots, mapped to the
    fewest trailing dots of a key with that text.
    """
    runs: dict[str, int] = {}
    others: list[str] = []
    for key in keys:
        if _SKILLSPECTOR_NAME_RUN_RE.fullmatch(key):
            base = key.rstrip(".")
            dots = len(key) - len(base)
            runs[base] = min(dots, runs.get(base, dots))
        else:
            others.append(key)
    return runs, tuple(sorted(others)), sum(len(key) for key in keys), max(map(len, keys), default=0)


def _skillspector_glob_matcher(pattern: str, longest: int) -> Callable[[str], object] | None:
    """Match the names of at most *longest* characters that the shell glob *pattern* may match, else ``None``.

    ``*`` matches any text and ``?`` one character. A bracket expression
    matches any one character, whatever it lists, and a ``]`` outside one
    matches any text, so a class such as ``[[:alpha:]]`` is covered too; the
    matcher may accept more names than the shell would, never fewer. ``None``
    means that no name that short can match. The work is linear in the length
    of *pattern*, and the regex never backtracks over more than one ``*``.

    A character followed by :data:`_SKILLSPECTOR_ZSH_REPEAT`, the whole
    pattern, matches only names made of that character, as zsh ``C#`` does.
    """
    size = len(pattern)
    if size == 2 and pattern[1] == _SKILLSPECTOR_ZSH_REPEAT:
        return re.compile(f"(?s:{re.escape(pattern[0])}*)\\Z").match
    closes = [size] * (size + 1)
    for index in range(size - 1, -1, -1):
        closes[index] = index if pattern[index] == "]" else closes[index + 1]
    segments: list[list[str]] = [[]]
    index = 0
    while index < size:
        char = pattern[index]
        index += 1
        if char in "*]":
            segments.append([])
            continue
        if char == "[":
            start = index + 1 if index < size and pattern[index] in "!^" else index
            start += 1 if start < size and pattern[start] == "]" else 0
            if closes[start] < size:
                segments[-1].append(".")
                index = closes[start] + 1
                continue
        segments[-1].append("." if char == "?" else re.escape(char))
    if sum(len(segment) for segment in segments) > longest:
        return None
    first, *rest = ("".join(segment) for segment in segments)
    if rest:
        *middle, last = rest
        first += "".join(f"(?>.*?{segment})" for segment in middle if segment) + f".*{last}"
    return re.compile(f"(?s:{first})\\Z").match


def _skillspector_shell_groups(text: str) -> str:
    """*text* with the shell groups and ranges inside its words read as patterns, in time linear in its length.

    zsh reads ``(a)`` inside a word as ``a`` and ``(a|b)`` as either
    alternative, bash and ksh read ``@(...)``, ``!(...)``, ``+(...)``,
    ``?(...)`` and ``*(...)`` as extended globs, and zsh reads ``<1-10>`` as
    any number in that range. A group with ``|`` at its own level, an extended
    glob and a range become :data:`_SKILLSPECTOR_SHELL_ANY`, and so does each
    run of characters other than whitespace that holds a zsh globbing flag
    group (:data:`_SKILLSPECTOR_ZSH_FLAGS_RE`), such as ``(#a2)``, since the
    flag changes how the rest of the word matches, across ``/`` too. Any other
    group loses its parentheses, also after ``]``, so ``[v](e)rsions`` reads
    as ``[v]ersions``; the line as written, with a Markdown link destination in
    its parentheses, is read as well. Parentheses around whitespace, ``/``, a
    backtick, ``<``, ``>``, ``;`` or ``&`` do not form a group and still end a
    word.
    """
    text = _SKILLSPECTOR_SHELL_RANGE_RE.sub(_SKILLSPECTOR_SHELL_ANY, text)
    if "(" not in text:
        return text
    text = _SKILLSPECTOR_NON_SPACE_RUN_RE.sub(
        lambda run: _SKILLSPECTOR_SHELL_ANY if _SKILLSPECTOR_ZSH_FLAGS_RE.search(run.group()) else run.group(), text
    )
    pieces: list[str] = []
    # Each open group: the index of its "(" in pieces, whether "|" splits it,
    # and the character before it.
    groups: list[tuple[int, list[bool], str]] = []
    last = 0
    for match in _SKILLSPECTOR_SHELL_GROUP_TOKEN_RE.finditer(text):
        index = match.start()
        char = match.group()
        pieces.append(text[last:index])
        last = match.end()
        if char == "(":
            groups.append((len(pieces), [False], text[index - 1 : index]))
            pieces.append(char)
        elif char == "|" and groups:
            groups[-1][1][0] = True
            pieces.append(char)
        elif char == ")" and groups:
            start, alternation, before = groups.pop()
            if alternation[0] or before in _SKILLSPECTOR_EXTGLOB_OPERATORS:
                del pieces[start:]
                if before in _SKILLSPECTOR_EXTGLOB_OPERATORS:
                    pieces[-1] = pieces[-1][:-1]
                pieces.append(_SKILLSPECTOR_SHELL_ANY)
            else:
                pieces[start] = ""
        else:
            if char not in "|)":
                groups.clear()
            pieces.append(char)
    pieces.append(text[last:])
    return "".join(pieces)


def _skillspector_shell_expansions(word: str) -> str:
    """*word* with its brace and parameter expansions read as ``*``, in time linear in its length.

    Everything from the first ``${`` to the last ``}`` after it (or to the
    end), everything from the first ``{`` to the last ``}`` when a ``,`` or
    ``..`` lies between them (brace expansion), and each other parameter
    expansion such as ``$name`` becomes ``*``, so nested expansions are covered
    too.
    """
    while (start := word.find("${")) >= 0:
        end = word.rfind("}")
        word = word[:start] + "*" + (word[end + 1 :] if end > start else "")
    start, end = word.find("{"), word.rfind("}")
    if 0 <= start < end and ("," in word[start:end] or ".." in word[start:end]):
        word = word[:start] + "*" + word[end + 1 :]
    return _SKILLSPECTOR_SHELL_PARAMETER_RE.sub("*", word)


def _skillspector_shell_word(word: str) -> tuple[str, str]:
    """*word* as glob patterns for what a POSIX shell, and cmd.exe or PowerShell, may expand it to.

    The work is linear in the length of *word*, and a group or range read in
    the line (:data:`_SKILLSPECTOR_SHELL_ANY`) is ``*`` in both readings.

    - A POSIX shell: quotes and backslashes are removed, then the expansions
      (:func:`_skillspector_shell_expansions`) become ``*``.
    - cmd.exe or PowerShell: ``"``, ``'`` and the ``^`` escape are removed; a
      batch parameter such as ``%1`` or ``%~dp0``, a variable such as
      ``%DIR%`` or ``!DIR!``, and the expansions become ``*``; and ``\\``
      separates directories as ``/`` does.

    ``*``, ``?`` and ``[...]`` keep their meaning, and so does ``~``
    (:func:`_skillspector_tilde_forms`).
    """
    word = word.replace(_SKILLSPECTOR_SHELL_ANY, "*")
    posix = _skillspector_shell_expansions(_SKILLSPECTOR_SHELL_QUOTE_RE.sub("", word))
    windows = _SKILLSPECTOR_CMD_VARIABLE_RE.sub("*", _SKILLSPECTOR_CMD_QUOTE_RE.sub("", word))
    windows = _skillspector_shell_expansions(windows).replace("\\", "/")
    return posix, windows


def _skillspector_tilde_forms(part: str, *, first: bool) -> set[str]:
    """Patterns for what the path part *part* may name when it has a ``~`` after its start.

    *part* itself is one. zsh ``EXTENDED_GLOB`` reads ``a~b`` as what ``a``
    matches except what ``b`` matches, so the text before the first ``~`` is
    another. With text after the ``~``, the part may also be a Windows short
    (8.3) name such as ``NODE_M~1`` or ``CONFIG~1.YAM``, or a name that a Samba
    share mangles, which stand for a longer name. Each keeps the first
    character of that name after the dots and spaces Windows drops, or ``_``
    for one it cannot keep, so ``*``, that character and ``*`` is a pattern
    too, and ``*`` when the character is ``_`` or glob syntax. A ``~`` at the
    start of the first part of a word (*first*) is a home directory and is
    skipped; one at the start of any other part makes the part ``*``.
    """
    tilde = part.find("~", 1 if first and part.startswith("~") else 0)
    if tilde < 0:
        return {part}
    if tilde == 0:
        return {"*"}
    base = part[:tilde]
    forms = {part, base}
    if tilde + 1 < len(part):
        lead = (base[1:] if first and base.startswith("~") else base).lstrip(". ")[:1]
        forms.add(f"*{lead}*" if lead and lead not in "_*?[]\\" else "*")
    return forms


def _skillspector_zsh_pattern(pattern: str) -> str:
    """*pattern* with the zsh ``EXTENDED_GLOB`` operators ``^`` and ``#`` read, in time linear in its length.

    Outside a bracket expression, ``^`` negates the rest of the pattern, which
    then becomes ``*``; that also covers ``^`` as the cmd.exe escape. ``x#``
    and ``x##`` repeat the character or bracket expression ``x`` and become
    ``*``. When they make up the whole pattern, as in ``C#``, they match only
    names made of ``x``: a character becomes ``x`` and
    :data:`_SKILLSPECTOR_ZSH_REPEAT`, and another ``x``, such as a bracket
    expression, ``x*``. A ``#`` that starts the pattern or follows a literal
    ``#``, as in a Markdown heading, is literal.
    """
    if "^" not in pattern and "#" not in pattern:
        return pattern
    size = len(pattern)
    closes = [size] * (size + 1)
    for index in range(size - 1, -1, -1):
        closes[index] = index if pattern[index] == "]" else closes[index + 1]
    elements: list[str] = []
    index = 0
    while index < size:
        char = pattern[index]
        if char == "[":
            start = index + 1
            start += 1 if start < size and pattern[start] in "!^" else 0
            start += 1 if start < size and pattern[start] == "]" else 0
            if closes[start] < size:
                elements.append(pattern[index : closes[start] + 1])
                index = closes[start] + 1
                continue
        elif char == "^":
            elements.append("*")
            break
        elif char == "#" and elements and elements[-1] != "#":
            while index < size and pattern[index] == "#":
                index += 1
            if len(elements) == 1 and index == size:
                plain = len(elements[0]) == 1 and elements[0] not in "*?"
                elements[0] += _SKILLSPECTOR_ZSH_REPEAT if plain else "*"
            else:
                elements[-1] = "*"
            continue
        elements.append(char)
        index += 1
    return "".join(elements)


def _skillspector_shell_patterns(texts: Iterable[str]) -> set[str]:
    """Glob patterns for the path parts of the words in *texts* that a shell or Windows expands or unquotes.

    A word with ``*``, ``?``, ``[``, ``{``, ``$``, ``%``, ``!``, ``^``, ``#``,
    ``~``, a quote, a backslash or a group read in the line is read as a POSIX shell and as
    cmd.exe or PowerShell would (:func:`_skillspector_shell_word`). Each part
    between ``/`` is a pattern, and so is each piece of it between ``=``,
    ``:`` and ``,``, as written and without the dots and punctuation that end
    a sentence after it, with what a ``~`` in it may stand for
    (:func:`_skillspector_tilde_forms`), each read with the zsh
    ``EXTENDED_GLOB`` operators (:func:`_skillspector_zsh_pattern`).
    """
    # Ignore only a leading Markdown bullet. A standalone wildcard elsewhere
    # (for example ``bash *``) can name every uninspected file.
    words = {
        word
        for text in texts
        for word in _SKILLSPECTOR_LINE_WORD_RE.findall(re.sub(r"^[ \t]*\*(?=[ \t])", "", text, count=1))
    }
    patterns: set[str] = set()
    for word in words:
        if _SKILLSPECTOR_SHELL_SYNTAX.isdisjoint(word):
            continue
        for reading in _skillspector_shell_word(word):
            for index, part in enumerate(reading.split("/")):
                for piece in {part, *_SKILLSPECTOR_PATH_PART_SEPARATOR_RE.split(part)}:
                    patterns.update(
                        _skillspector_zsh_pattern(form)
                        for candidate in (piece, piece.rstrip(".,;:!?"))
                        if candidate
                        for form in _skillspector_tilde_forms(candidate, first=index == 0)
                    )
    return patterns


def _skillspector_reference_line_avoids(
    source: bytes,
    row: Mapping[str, object],
    references: list,
    keys: frozenset[str],
    budget: _SkillspectorMatchBudget | None = None,
) -> bool:
    """Whether the line of a missing-reference *row* names no entry whose name key is in *keys*.

    The line is taken from *source* as SkillSpector numbers it: UTF-8 with
    replacement, split at ``\\n``. Every ``missing`` entry of *references* on
    that line must quote part of it as ``evidence``, which confirms the line.
    SkillSpector resolves a reference by its path, or by its file name alone
    when it has no ``/``, so the line must not name an uninspected entry. A
    name counts when no letter, digit, ``_``, ``-`` or ``.`` adjoins it, after
    any dots that follow it, which Windows drops when it opens a file. It is
    compared case-folded (:func:`_skillspector_name_keys`), as written and in
    NFC, in the line as written and as a Markdown renderer or a file path
    reader may read it: with backslash escapes of ASCII punctuation removed,
    HTML character references resolved and percent-escapes decoded, up to
    :data:`_SKILLSPECTOR_REFERENCE_LINE_DECODINGS` of these steps in any order,
    and then also with its shell groups and ranges read
    (:func:`_skillspector_shell_groups`), with and without its backticks. A
    word that a shell, cmd.exe, PowerShell or Windows would expand or unquote
    is read as a pattern (:func:`_skillspector_shell_patterns`), and no name
    may match it.

    Only this line is read, and no command in it is run: a span between
    backticks is read as text, as Markdown shows it, not as the output of a
    command. So a name on another line, or one that a command prints, such as
    ``printf`` in backticks, a pipe into ``xargs`` or ``find -exec``, is not
    detected.

    The check takes work from *budget*, by default a fresh
    :data:`_SKILLSPECTOR_REFERENCE_MATCH_BUDGET`, and names the line as
    uninspected when it would overdraw it.
    """
    line_number = row.get("start_line")
    lines = source.decode("utf-8", errors="replace").split("\n")
    if isinstance(line_number, bool) or not isinstance(line_number, int) or not 1 <= line_number <= len(lines):
        return False
    line = lines[line_number - 1]
    cleaned = " ".join(line.split())
    evidence = [
        reference.get("evidence")
        for reference in _skillspector_references_on_line(references, row.get("path"), line_number)
        if reference.get("status") == "missing"
    ]
    if (
        len(line) > _SKILLSPECTOR_REFERENCE_LINE_MAX_CHARS
        or not evidence
        or any(not isinstance(text, str) or not text or text not in cleaned for text in evidence)
    ):
        return False
    if not keys:
        return True
    budget = budget or _SkillspectorMatchBudget()
    if not budget.spend(len(line) * 9_000):
        return False
    decodings: tuple[Callable[[str], str], ...] = (
        lambda text: _SKILLSPECTOR_MARKDOWN_ESCAPE_RE.sub(r"\1", text),
        html.unescape,
        unquote,
    )
    variants = {line}
    for _step in range(_SKILLSPECTOR_REFERENCE_LINE_DECODINGS):
        variants |= {decode(text) for text in variants for decode in decodings}
    # Shell groups and ranges, and backticks, which PowerShell reads as an
    # escape, may join the text around them.
    variants |= {_skillspector_shell_groups(form) for text in variants for form in {text, text.replace("`", "")}}
    texts = {text.casefold() for text in variants} | {
        unicodedata.normalize("NFC", text).casefold() for text in variants
    }
    run_keys, other_keys, key_chars, longest = _skillspector_key_kinds(keys)
    patterns = _skillspector_shell_patterns(texts)
    globs = [pattern for pattern in patterns if not _SKILLSPECTOR_GLOB_SYNTAX.isdisjoint(pattern)]
    searches = len(texts) * len(other_keys)
    cost = searches * 100 + sum(len(text) for text in texts) * len(other_keys)
    cost += sum(20_000 + 1_000 * len(pattern) + 300 * len(keys) + key_chars for pattern in globs)
    if not budget.spend(cost):
        return False
    # A key made only of name characters matches a whole run of them, less any trailing dots.
    runs: dict[str, int] = {}
    for run in {run for text in texts for run in _SKILLSPECTOR_NAME_RUN_RE.findall(text)}:
        base = run.rstrip(".")
        runs[base] = max(len(run) - len(base), runs.get(base, 0))
    if any(base in run_keys and run_keys[base] <= dots for base, dots in runs.items()):
        return False
    if not patterns.isdisjoint(keys):
        return False
    if any(_skillspector_text_names(text, key) for key in other_keys for text in texts):
        return False
    for pattern in globs:
        matches = _skillspector_glob_matcher(pattern, longest)
        if matches is not None and any(matches(key) for key in keys):
            return False
    return True


def _skillspector_text_names(text: str, name: str) -> bool:
    """Whether *name* appears in *text* with no name character around it, after any trailing dots."""
    start = text.find(name)
    while start >= 0:
        end = start + len(name)
        while end < len(text) and text[end] == ".":
            end += 1
        if (start == 0 or not _SKILLSPECTOR_NAME_RUN_RE.fullmatch(text[start - 1])) and (
            end == len(text) or not _SKILLSPECTOR_NAME_RUN_RE.fullmatch(text[end])
        ):
            return True
        start = text.find(name, start + 1)
    return False


def _skillspector_missing_reference_warning(rows: list[Mapping[str, object]]) -> str:
    """Describe accepted missing-reference ledger rows, naming the file and line of up to three of them."""
    shown = rows[:_SKILLSPECTOR_INCOMPLETE_EXAMPLES]
    labels = "; ".join(_skillspector_ledger_exception_label(row) for row in shown)
    if len(rows) > len(shown):
        labels = f"{labels}; and {len(rows) - len(shown)} more"
    return (
        f"SkillSpector reported {len(rows)} reference(s) to files the skill does not bundle ({labels}); "
        "SkillSpector reported no other gap for them, and those lines name no file left out of the scan, "
        "so they do not mark the scan incomplete. Files not in the scan, including any outside the scanned "
        "path, were not inspected"
    )


def _skillspector_missing_references_have_rows(references: list, rows: list[Mapping[str, object]]) -> bool:
    """Whether *references* holds only objects, none ``ambiguous``, each ``missing`` one on the line of a row in *rows*.

    Only the line of a row is read for the names it holds
    (:func:`_skillspector_reference_line_avoids`), so a ``missing`` entry on
    any other line, which SkillSpector reported without a row, would never be.
    """
    locations = {(row.get("path"), row.get("start_line")) for row in rows}
    for reference in references:
        if not isinstance(reference, dict) or reference.get("status") == "ambiguous":
            return False
        if reference.get("status") == "missing":
            path, line = reference.get("source_path"), reference.get("line")
            if not isinstance(path, str) or type(line) is not int or (path, line) not in locations:
                return False
    return True


def _skillspector_reference_evidence_unmatched(completeness: Mapping[str, object]) -> bool:
    """Whether a report whose ledger has no missing-reference row lists a gap that nothing reads.

    SkillSpector gives every ``missing`` or ``ambiguous`` reference a ledger
    row, so with none, such an entry in the ``references`` of *completeness*
    names a line that no check reads, and a ``*_truncated`` flag means a row
    may have been cut off. The report is then not accepted on the strength of
    its other gaps, such as a referenced image.
    """
    references = completeness.get("references")
    return _skillspector_flags_truncation(completeness) or (
        isinstance(references, list) and not _skillspector_missing_references_have_rows(references, [])
    )


def _skillspector_missing_reference_notes(
    ledger_exceptions: list[dict],
    analyzer_statuses: list[dict],
    counts: Mapping[str, int],
    limitations: list[str],
    scope_exclusions: object,
    components: object,
    scanned_root: Path | None,
    *,
    references: object,
    skillspector_version: tuple[int, int, int] | None,
    reference_root: Path | None,
    analysis_completeness: Mapping[str, object],
) -> list[str] | None:
    """Warnings for a ``partial`` analysis whose gaps are references to files not bundled, else ``None``.

    SkillSpector records a path-like reference in the skill's instructions that
    matches no file it scanned, such as a file the skill writes at run time or
    a repository source, as a nonfatal ``reference_resolution`` ledger row and
    reports the analysis as ``partial``, although every file it scanned was
    inspected.

    That gap is accepted only when every ``reference_resolution`` row passes
    :func:`_skillspector_reference_row_is_advisory` and names a bundled file
    (:func:`_skillspector_bundled_file_bytes`) whose line names no entry of the
    skill that SkillSpector did not inspect (:func:`_skillspector_unscanned_names`,
    :func:`_skillspector_reference_line_avoids`, with one work budget for all
    of the report's lines), such as a link or a file in an artifact directory
    left out of a staged scan, and there is no fatal scope exclusion.
    Every missing reference must have an accepted ledger row at its source
    location, and no reference may be ambiguous. Unknown completeness fields,
    including truncation markers, are not accepted. Nonfatal scope exclusions
    must name actual uninspected paths. Any other ledger rows must be referenced
    images that :func:`_skillspector_passive_raster_note` accepts, which then accounts for
    the counts, analyzers and limitations, and its note follows. Without them,
    no file is partially or entirely uninspected, no finding was filtered out,
    there is no limitation, every analyzer is ``completed``, ``not_applicable``
    or ``disabled``, and each universal analyzer completed one work item per
    component. Anything else keeps the scan INCOMPLETE.

    *reference_root* is the root of the artifact that the scan belongs to,
    and the gap is accepted only when it is *scanned_root* itself. Another
    root, such as a plugin around the bundled skill that was scanned or a
    folder whose skills are scanned one by one, means that instructions may
    name files of the artifact beside the scanned directory, which this scan
    did not inspect. Without a root it is not known what the scan left out.

    SkillSpector 2.13.4 and later report such an analysis as
    ``complete_with_caveats`` instead; :func:`_skillspector_reference_caveat_notes`
    applies the same rules to it.
    """
    # A bounded subset cannot establish that every reference was checked.
    # Keep unknown evidence fields (including truncation markers) outside the
    # same report contract used by the report-only reference-caveat gate.
    if set(analysis_completeness) != REFERENCE_COMPLETENESS_FIELDS:
        return None
    reference_rows = [row for row in ledger_exceptions if row.get("phase") == "reference_resolution"]
    other_rows = [row for row in ledger_exceptions if row.get("phase") != "reference_resolution"]
    if (
        scanned_root is None
        or reference_root != scanned_root
        or not reference_rows
        or not isinstance(scope_exclusions, list)
        or any(not isinstance(row, dict) or row.get("fatal") is not False for row in scope_exclusions)
        or not isinstance(references, list)
    ):
        return None
    for row in reference_rows:
        if not isinstance(row.get("path"), str) or not _skillspector_reference_row_is_advisory(
            row, skillspector_version, references
        ):
            return None

    if not _skillspector_missing_references_have_rows(references, reference_rows):
        return None

    notes = [_skillspector_missing_reference_warning(reference_rows)]
    if other_rows:
        image_note = _skillspector_passive_raster_note(
            other_rows, analyzer_statuses, counts, limitations, scope_exclusions, components, scanned_root
        )
        if image_note is None:
            return None
        notes.append(image_note)
    elif (
        limitations
        or counts["partially_inspected_files"]
        or counts["entirely_uninspected_files"]
        or counts["findings_before_filtering"] != counts["findings_after_filtering"]
        or not _skillspector_universal_analyzers_completed(analyzer_statuses, counts["total_components"])
    ):
        return None
    unscanned_paths = _skillspector_unscanned_paths(scanned_root, components)
    if unscanned_paths is None or any(
        not isinstance(row.get("path"), str) or row["path"] not in unscanned_paths for row in scope_exclusions
    ):
        return None
    keys = _skillspector_name_keys(part for path in unscanned_paths for part in path.split("/"))
    budget = _SkillspectorMatchBudget()
    sources: dict[str, bytes] = {}
    checked: set[tuple[str, object]] = set()
    for row in reference_rows:
        path = row["path"]
        if (path, row.get("start_line")) in checked:
            continue
        if path not in sources:
            source = _skillspector_bundled_file_bytes(path, components, scanned_root)
            if source is None:
                return None
            sources[path] = source
        if not _skillspector_reference_line_avoids(sources[path], row, references, keys, budget):
            return None
        checked.add((path, row.get("start_line")))
    return notes


def _skillspector_universal_analyzers_completed(analyzer_statuses: list[dict], total: int) -> bool:
    """Whether each universal analyzer completed one work item per component and no other analyzer degraded."""
    universal = _SKILLSPECTOR_COMMON_UNIVERSAL_ANALYZERS | {"artifact_integrity"}
    accounted: set[str] = set()
    for status in analyzer_statuses:
        analyzer_id = status.get("analyzer_id")
        state = status.get("status")
        if analyzer_id not in universal:
            if state not in {"completed", "not_applicable", "disabled"}:
                return False
            continue
        if (
            analyzer_id in accounted
            or state != "completed"
            or status.get("planned_work") != total
            or status.get("completed") != total
        ):
            return False
        accounted.add(analyzer_id)
    return accounted == universal


def _skillspector_row_is_reference_caveat(row: Mapping[str, object]) -> bool:
    """Whether a ledger row is a missing reference, the only caveat of a ``complete_with_caveats`` analysis."""
    return row.get("reason_code") == "reference_missing" and row.get("phase") == "reference_resolution"


def _skillspector_rows_are_reference_caveats(ledger_exceptions: list[dict]) -> bool:
    """Whether a ``complete_with_caveats`` ledger has caveats and every one is a missing reference.

    SkillSpector 2.13.4 and later give that status only when every ledger row
    is a nonfatal ``reference_missing`` row, which they record in phase
    ``reference_resolution``; any other row makes the analysis ``partial`` or
    ``failed``. A report that fails this contradicts its own status.
    """
    return bool(ledger_exceptions) and all(_skillspector_row_is_reference_caveat(row) for row in ledger_exceptions)


def _skillspector_reference_caveat_notes(
    ledger_exceptions: list[dict],
    analyzer_statuses: list[dict],
    counts: Mapping[str, int],
    limitations: list[str],
    scope_exclusions: object,
    components: object,
    scanned_root: Path | None,
    *,
    references: object,
    skillspector_version: tuple[int, int, int] | None,
    reference_root: Path | None,
    analysis_completeness: Mapping[str, object],
) -> list[str] | None:
    """Warnings for a ``complete_with_caveats`` analysis, else ``None``.

    SkillSpector 2.13.4 and later report an analysis whose only gaps are
    nonfatal ``reference_missing`` rows as ``complete_with_caveats``, where
    earlier releases report it as ``partial``. Its caveats are accepted only
    when there is at least one, every one is a ``reference_missing`` row in
    phase ``reference_resolution``, and together they pass
    :func:`_skillspector_missing_reference_notes`, the rule for the same rows
    in a ``partial`` report. A caveat of any other kind is not one SkillSpector
    emits with this status, so it keeps the scan INCOMPLETE.
    """
    if not _skillspector_rows_are_reference_caveats(ledger_exceptions):
        return None
    return _skillspector_missing_reference_notes(
        ledger_exceptions,
        analyzer_statuses,
        counts,
        limitations,
        scope_exclusions,
        components,
        scanned_root,
        references=references,
        skillspector_version=skillspector_version,
        reference_root=reference_root,
        analysis_completeness=analysis_completeness,
    )


def _skillspector_reference_caveat_error(ledger_exceptions: list[dict]) -> str:
    """Describe a ``complete_with_caveats`` analysis whose caveats were not accepted, naming up to three.

    When some caveats are not missing references, which contradicts the
    status, only those are named.
    """
    if not ledger_exceptions:
        return (
            "skillspector JSON field 'analysis_completeness' reports status 'complete_with_caveats' "
            "without a caveat; security scan did not complete"
        )
    rows = [row for row in ledger_exceptions if not _skillspector_row_is_reference_caveat(row)]
    only_missing_references = not rows
    description = "caveats other than missing references"
    if not rows:
        rows = ledger_exceptions
        description = "caveats that SkillEvaluator does not accept"
    shown = rows[:_SKILLSPECTOR_INCOMPLETE_EXAMPLES]
    message = (
        f"skillspector JSON field 'analysis_completeness' reports {description} "
        f"(status 'complete_with_caveats'): {'; '.join(_skillspector_ledger_exception_label(row) for row in shown)}"
    )
    if len(rows) > len(shown):
        message = f"{message}; and {len(rows) - len(shown)} more"
    message = f"{message}; security scan did not complete"
    return f"{message}. {_SKILLSPECTOR_REFERENCE_REJECTION_HINT}" if only_missing_references else message


# SkillSpector 2.12.0 and later record a nonfatal ``partial`` ``opaque_content``
# row from ``bundled_execution_surface`` on a hook manifest whenever they cannot
# model what a hook does with hook input (NVIDIA/SkillSpector#573). That is every
# command or HTTP hook unless SkillSpector recognizes each one in the manifest as
# sending hook content to a remote host (BH2). The scan itself completed, but its
# findings cannot show that the hooks are safe, so the manifest gets a Critical
# gate failure instead of an incomplete scan.
_SKILLSPECTOR_HOOK_ANALYZER = "bundled_execution_surface"
_SKILLSPECTOR_HOOK_LIMITATION = f"Analyzer {_SKILLSPECTOR_HOOK_ANALYZER} status: degraded."
_SKILLSPECTOR_HOOK_UNVERIFIED_CHECK = "skillspector_hook_payload_unverified"
# SkillSpector reads at most this many hook, permission and settings records
# from one manifest (``_MAX_DECLARATIONS`` in bundled_execution_surface.py).
_SKILLSPECTOR_HOOK_DECLARATION_LIMIT = 2048
# Findings that are facts about the scan, not judgments an LLM verifier can
# overturn: the hook gate failure, and the reference-caveat note, which reports
# scanner coverage, not a vulnerability. They keep their evidence unchanged.
_LLM_VERIFY_EXEMPT_CHECKS = frozenset({_SKILLSPECTOR_HOOK_UNVERIFIED_CHECK, "skillspector_reference_caveats"})
_SKILLSPECTOR_HOOK_COUNT_FIELDS = (
    "total_components",
    "partially_inspected_files",
    "entirely_uninspected_files",
    "findings_before_filtering",
    "findings_after_filtering",
)


def _is_count(value: object) -> bool:
    """Whether *value* is a JSON integer, not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def _skillspector_unmodeled_hook_manifests(issues: object) -> dict[str, int]:
    """The manifests whose BH1 finding says SkillSpector parsed hooks but could not model their payload.

    Each manifest is the BH1 location's path when it passes
    :func:`_skillspector_report_path_key`, mapped to the BH1
    ``declaration_count``, which must be a positive count. A manifest whose
    BH1 findings give different counts is left out.
    """
    if not isinstance(issues, list):
        return {}
    manifests: dict[str, int] = {}
    conflicting: set[str] = set()
    for issue in issues:
        if not isinstance(issue, dict) or issue.get("id") != "BH1":
            continue
        evidence = issue.get("evidence")
        location = issue.get("location")
        if not (
            isinstance(evidence, dict)
            and evidence.get("payload_analysis_level") == "unmodeled"
            and isinstance(location, dict)
            and (manifest := _skillspector_report_path_key(location.get("file"))) is not None
        ):
            continue
        declarations = evidence.get("declaration_count")
        if (
            not _is_count(declarations)
            or declarations < 1
            or manifests.setdefault(manifest, declarations) != declarations
        ):
            conflicting.add(manifest)
    return {manifest: count for manifest, count in manifests.items() if manifest not in conflicting}


def _skillspector_ledger_exception_is_hook_surface_limitation(
    exception: Mapping[str, object], unmodeled_manifests: Mapping[str, int]
) -> bool:
    """Whether a ledger row is SkillSpector's limitation for hooks it parsed but could not model.

    The row must be a nonfatal, whole-file ``partial`` ``opaque_content`` row
    in phase ``static`` that names only ``bundled_execution_surface`` and no
    error class, on a manifest in *unmodeled_manifests*
    (:func:`_skillspector_unmodeled_hook_manifests`).

    SkillSpector records the same row when it could not parse part of the
    manifest or read only its first 2,048 hook, permission and settings
    records. When the row carries ``observed_records`` and ``limit_records``
    (SkillSpector 2.13.4), ``observed_records`` must equal the BH1
    ``declaration_count`` and not exceed ``limit_records``: a skipped handler,
    a permission or settings record, or a truncated manifest makes them
    differ. Without them (SkillSpector 2.13.3 and 2.14.0), the BH1 must count
    fewer than 2,048 declarations. Other partial parses cannot be told apart
    from unmodeled hooks and get the same Critical.
    """
    manifest = _skillspector_report_path_key(exception.get("path"))
    declarations = unmodeled_manifests.get(manifest) if manifest is not None else None
    if declarations is None:
        return False
    if "observed_records" in exception or "limit_records" in exception:
        observed = exception.get("observed_records")
        limit = exception.get("limit_records")
        counted = _is_count(observed) and _is_count(limit) and observed == declarations and observed <= limit
    else:
        counted = declarations < _SKILLSPECTOR_HOOK_DECLARATION_LIMIT
    return (
        counted
        and exception.get("outcome") == "partial"
        and exception.get("fatal") is False
        and exception.get("phase") == "static"
        and exception.get("reason_code") == "opaque_content"
        and exception.get("analyzers") == [_SKILLSPECTOR_HOOK_ANALYZER]
        and exception.get("error_class") is None
        and exception.get("start_line") is None
        and exception.get("end_line") is None
    )


def _skillspector_without_hook_surface(
    data: Mapping[str, object],
) -> tuple[list[dict], list[dict], list[dict], dict[str, int], list[str]] | None:
    """A ``partial`` report's gaps without the rows for hooks SkillSpector could not model, else ``None``.

    Returns the hook rows (:func:`_skillspector_ledger_exception_is_hook_surface_limitation`)
    and the report's other ledger rows, analyzer statuses, counts and
    limitations as they would be without the hook rows. BH1 findings are read
    from both ``issues`` and ``suppressed``, because a baseline suppresses the
    finding but not the row.

    The hook rows are taken out only when they account exactly for their part
    of the report: one row per manifest; one ``bundled_execution_surface``
    status, ``degraded`` by one ``partial`` work item per hook row with nothing
    failed or skipped, which then counts as completed; its ``Analyzer
    bundled_execution_surface status: degraded.`` limitation, which is
    dropped; and one partially inspected file for each manifest that no other
    row names. Anything else returns ``None``, and the report is judged with
    every row.
    """
    completeness = data.get("analysis_completeness")
    if (
        not isinstance(completeness, dict)
        or completeness.get("status") != "partial"
        or completeness.get("is_complete") is not False
    ):
        return None
    ledger_exceptions = completeness.get("ledger_exceptions")
    analyzer_statuses = completeness.get("analyzer_statuses")
    limitations = completeness.get("limitations")
    counts = {field: completeness.get(field) for field in _SKILLSPECTOR_HOOK_COUNT_FIELDS}
    if (
        not isinstance(ledger_exceptions, list)
        or not all(isinstance(row, dict) for row in ledger_exceptions)
        or not isinstance(analyzer_statuses, list)
        or not all(isinstance(status, dict) for status in analyzer_statuses)
        or not isinstance(limitations, list)
        or not all(isinstance(limitation, str) for limitation in limitations)
        or not all(_is_count(value) for value in counts.values())
    ):
        return None
    issues: list[object] = []
    for field in ("issues", "suppressed"):
        found = data.get(field)
        if isinstance(found, list):
            issues.extend(found)
    manifests = _skillspector_unmodeled_hook_manifests(issues)
    hook_rows: list[dict] = []
    other_rows: list[dict] = []
    for row in ledger_exceptions:
        if _skillspector_ledger_exception_is_hook_surface_limitation(row, manifests):
            hook_rows.append(row)
        else:
            other_rows.append(row)
    hook_paths = {row["path"] for row in hook_rows}
    surface = [status for status in analyzer_statuses if status.get("analyzer_id") == _SKILLSPECTOR_HOOK_ANALYZER]
    if not hook_rows or len(hook_paths) != len(hook_rows) or len(surface) != 1:
        return None
    [status] = surface
    completed = status.get("completed")
    if (
        status.get("status") != "degraded"
        or not _is_count(completed)
        or not _is_count(status.get("partial"))
        or status.get("partial") != len(hook_rows)
        or any(not _is_count(status.get(field)) or status.get(field) != 0 for field in ("skipped", "failed"))
        or _SKILLSPECTOR_HOOK_LIMITATION not in limitations
    ):
        return None
    other_paths = {path for row in other_rows if isinstance(path := row.get("path"), str)}
    partially_inspected = counts["partially_inspected_files"] - len(hook_paths - other_paths)
    if partially_inspected < 0:
        return None
    remaining_limitations = list(limitations)
    remaining_limitations.remove(_SKILLSPECTOR_HOOK_LIMITATION)
    completed_status = {**status, "status": "completed", "completed": completed + len(hook_rows), "partial": 0}
    return (
        hook_rows,
        other_rows,
        [completed_status if item is status else item for item in analyzer_statuses],
        {**counts, "partially_inspected_files": partially_inspected},
        remaining_limitations,
    )


def _skillspector_gaps_without_rows(
    analyzer_statuses: list[dict],
    counts: Mapping[str, int],
    limitations: list[str],
    scope_exclusions: object,
) -> list[str]:
    """Describe the gaps a report without its hook rows (:func:`_skillspector_without_hook_surface`) shows outside its rows.

    A gap is a limitation, a partially or entirely uninspected file, a fatal
    scope exclusion, a finding count that changed in filtering, or analyzer
    work that did not finish: each universal analyzer must complete one work
    item per component, and every other analyzer must complete or be not
    applicable or disabled. No gap gives an empty list.
    """
    gaps = [
        f"limitation '{_skillspector_report_text(limitation) or ''}'"
        for limitation in limitations[:_SKILLSPECTOR_INCOMPLETE_EXAMPLES]
    ]
    if len(limitations) > _SKILLSPECTOR_INCOMPLETE_EXAMPLES:
        gaps.append(f"{len(limitations) - _SKILLSPECTOR_INCOMPLETE_EXAMPLES} more limitation(s)")
    if counts["partially_inspected_files"]:
        gaps.append(f"{counts['partially_inspected_files']} partially inspected file(s)")
    if counts["entirely_uninspected_files"]:
        gaps.append(f"{counts['entirely_uninspected_files']} uninspected file(s)")
    if not isinstance(scope_exclusions, list):
        gaps.append("unreadable scope exclusions")
    elif fatal := sum(not (isinstance(row, dict) and row.get("fatal") is False) for row in scope_exclusions):
        gaps.append(f"{fatal} fatal scope exclusion(s)")
    if counts["findings_before_filtering"] != counts["findings_after_filtering"]:
        gaps.append(
            f"{counts['findings_before_filtering']} finding(s) before filtering and "
            f"{counts['findings_after_filtering']} after"
        )
    if not _skillspector_universal_analyzers_completed(analyzer_statuses, counts["total_components"]):
        unfinished = [
            f"{_skillspector_report_text(status.get('analyzer_id')) or 'unknown'} "
            f"{_skillspector_report_text(status.get('status')) or 'unknown'}"
            for status in analyzer_statuses
            if status.get("status") not in {"completed", "not_applicable", "disabled"}
        ]
        gaps.append(
            f"analyzer status {', '.join(unfinished[:_SKILLSPECTOR_INCOMPLETE_EXAMPLES])}"
            if unfinished
            else "analyzer work that does not cover every component"
        )
    return gaps


def _skillspector_gaps_without_rows_error(status: str, gaps: list[str]) -> str:
    """Describe a partial analysis whose only gaps outside the hook rows have no ledger row."""
    return (
        f"skillspector JSON field 'analysis_completeness' reports incomplete analysis (status '{status}'): "
        f"{'; '.join(gaps)}; security scan did not complete"
    )


class SecurityValidator(ValidatorBase):
    """Scans skills for security vulnerabilities and PII leakage.

    Combines the skillspector CLI (15+ vulnerability patterns) with custom PII
    detection for comprehensive security analysis.

    Lifecycle: construct one instance **per validation job**. The submitter
    identity is resolved from the environment once and cached for the instance's
    lifetime, so a long-lived instance reused across requests/users with
    differing env vars would apply the first caller's identity to all of them.
    A service that validates on behalf of different users should create a fresh
    ``SecurityValidator`` per request (or pass an explicit ``submitter_usernames``).
    """

    # A skill's unmodeled hooks say nothing about the next skill in a folder.
    _FAIL_FAST_EXEMPT_CHECKS = frozenset({_SKILLSPECTOR_HOOK_UNVERIFIED_CHECK})

    def __init__(
        self,
        use_llm: bool = False,
        verify_llm: bool = False,
        submitter_usernames: Iterable[str] | None = None,
        pii_waiver_path: Path | None = None,
        content_type: str | None = None,
        skillspector_reference_root: Path | None = None,
        skillspector_baseline: Path | str | None = None,
        target_root: Path | str | None = None,
    ):
        """Initialize with optional LLM analysis for deeper inspection.

        Args:
            use_llm: Enable LLM analysis in skillspector for deeper detection.
            verify_llm: Enable LLM second-pass verification to suppress false positives.
            submitter_usernames: Explicit submitter identity(ies) used by the
                home-path PII check. When ``None`` (the default) the submitter is
                auto-detected from the environment (see
                :data:`HOME_PATH_SUBMITTER_ENV_VARS`) and the OS login name.
                Pass an explicit value to wire a CI-provided contributor identity
                or to make tests deterministic.
            pii_waiver_path: Optional PII waiver file (see
                :mod:`skillevaluator.validators.pii_waivers`) applied by
                :meth:`validate` and :meth:`validate_pii_only`.
            content_type: Detected agent artifact type (``skill``, ``rules``,
                ``workflows``, or ``plugin``). When set, SkillSpector findings
                outside agent control paths are suppressed (see
                :mod:`skillevaluator.validators.skillspector_scope`). ``None``
                keeps the unrestricted repository scan used by ``security-scan``.
            skillspector_reference_root: Artifact root that contains the
                scanned directory, such as a plugin around its bundled skills.
                References written relative to it still count as agent
                references. Defaults to the scanned directory.
                :func:`_skillspector_missing_reference_notes` accepts missing
                references only in a scan of this root, or without it, of
                *target_root* or else the whole target of :meth:`validate` or
                :meth:`validate_security_only`. A scan outside those methods
                without this root has no reference root, so that rule accepts
                none there; a 2.13.3 report that
                :func:`reference_caveat_metadata` accepts still passes as its
                caveat.
            skillspector_baseline: Audited SkillSpector YAML/JSON suppression
                baseline. Matching findings are removed by SkillSpector before
                scoring. Baselines are only applied when passed explicitly; one
                shipped inside the scanned skill is never picked up on its own.
            target_root: Root of the whole target when this validator is given
                only part of it, such as one skill of a folder that
                ``validate`` checks skill by skill. It replaces the target of
                :meth:`validate` or :meth:`validate_security_only` as the root
                that missing references are judged against, so a skill inside
                it accepts none. Finding scoping is unchanged.
        """
        self._pii_patterns: dict | None = None
        self._submitter_override: set[str] | None = (
            self._normalize_usernames(submitter_usernames) if submitter_usernames is not None else None
        )
        # Cached for the instance lifetime (see class docstring: per-job usage).
        self._submitter_cache: set[str] | None = None
        # Ensures the "home-path check disabled" warning is emitted at most once.
        self._home_check_disabled_warned = False
        self.use_llm = use_llm
        self.verify_llm = verify_llm
        self.pii_waiver_path = Path(pii_waiver_path) if pii_waiver_path is not None else None
        # The waivers of the scan in progress; set only while validate() or validate_pii_only() runs.
        self._pii_waivers: PIIWaiverSet | None = None
        self.content_type = content_type
        self.skillspector_reference_root = (
            Path(skillspector_reference_root).resolve() if skillspector_reference_root is not None else None
        )
        self.target_root = Path(target_root).resolve() if target_root is not None else None
        # The resolved target of the scan in progress, or target_root; set only
        # while _validate_with_applicability() walks it.
        self._scan_target_root: Path | None = None
        self.skillspector_baseline = (
            Path(skillspector_baseline).expanduser().resolve() if skillspector_baseline is not None else None
        )

    @property
    def pii_patterns(self) -> dict:
        """Lazy-load PII detection patterns from config."""
        if self._pii_patterns is None:
            self._pii_patterns = load_pii_patterns()
        return self._pii_patterns

    @staticmethod
    def _normalize_usernames(values: Iterable[str] | None) -> set[str]:
        """Lower-case, strip, and drop empties from an iterable of usernames."""
        if not values:
            return set()
        return {v.strip().lower() for v in values if v and v.strip()}

    @staticmethod
    def _usernames_from_identity(identity: str | None) -> set[str]:
        """Extract candidate Linux usernames from an identity string.

        Handles the common shapes found in skill ``author`` frontmatter and CI
        environment variables:
        - ``Example User <example-user@example.com>`` -> ``{"example-user"}``
          (email local part)
        - ``example-user@example.com`` -> ``{"example-user"}``
        - ``example-user`` (bare token) -> ``{"example-user"}``

        A value containing spaces but no email (a display name like
        ``Example User``) yields nothing, since a display name is not a
        filesystem username and matching it would re-introduce false positives.
        """
        if not identity:
            return set()
        text = str(identity).strip()
        names: set[str] = set()
        email_match = re.search(r"([A-Za-z0-9._%+\-]+)@", text)
        if email_match:
            names.add(email_match.group(1).strip().lower())
        if text and "@" not in text and "<" not in text and ">" not in text and " " not in text:
            names.add(text.lower())
        return {n for n in names if n}

    def _submitter_usernames(self) -> set[str]:
        """Resolve the identity(ies) of whoever is submitting/validating the skill.

        Used (together with the skill's declared author) to decide which
        ``/home/<user>/`` paths are personal. Resolution order:

        1. An explicit override passed to the constructor (``submitter_usernames``),
           e.g. a CI-provided contributor identity. When set, env/OS detection is
           skipped entirely.
        2. Environment variables in :data:`HOME_PATH_SUBMITTER_ENV_VARS`
           (``SKILLEVALUATOR_SUBMITTER``, ``GITHUB_ACTOR``, ``USER``, ``LOGNAME``,
           ``USERNAME``).
        3. The OS login name (:func:`getpass.getuser`).

        Results are cached for the lifetime of the validator instance.
        """
        if self._submitter_override is not None:
            return self._submitter_override
        if self._submitter_cache is not None:
            return self._submitter_cache

        names: set[str] = set()
        for var in HOME_PATH_SUBMITTER_ENV_VARS:
            names |= self._usernames_from_identity(os.environ.get(var))
        # getpass.getuser() can raise if the uid has no passwd entry (some CI sandboxes).
        with contextlib.suppress(Exception):
            names |= self._usernames_from_identity(getpass.getuser())

        self._submitter_cache = names
        return names

    def _author_identity(self, skill_path: Path) -> str | None:
        """Return the skill's declared author identity, if one is parseable."""
        if not skill_path.is_dir():
            return None
        manifest = self._find_skill_manifest(skill_path)
        if manifest is None:
            return None

        from skillevaluator.validators.frontmatter_parser import parse_frontmatter

        parsed, _ = parse_frontmatter(manifest)
        if parsed is None or not isinstance(parsed.yaml_data, dict):
            return None

        data = parsed.yaml_data
        author = data.get("author")
        meta = data.get("metadata")
        if not author and isinstance(meta, dict):
            author = meta.get("author")
        return str(author).strip() if author else None

    def _author_usernames(self, skill_path: Path) -> set[str]:
        """Usernames derived from the skill's declared ``author`` frontmatter.

        Reads the SKILL.md manifest in ``skill_path`` (if any) and extracts the
        author identity from either the top-level ``author`` field or
        ``metadata.author`` (SkillEvaluator nests it under ``metadata``). Returns the
        empty set when there is no manifest or no parseable author.
        """
        return self._usernames_from_identity(self._author_identity(skill_path))

    def _protected_home_usernames(self, skill_path: Path) -> set[str]:
        """Identities whose ``/home/<user>/`` paths should be flagged as PII.

        The union of the submitter identity (see :meth:`_submitter_usernames`)
        and the skill author identity (see :meth:`_author_usernames`). These are
        the only home directories that reliably identify a contributor;
        unrelated ``/home/<root>/`` paths are intentionally left unflagged.
        """
        return self._submitter_usernames() | self._author_usernames(skill_path)

    @property
    def name(self) -> str:
        return "Security & PII Scanning"

    @property
    def description(self) -> str:
        return "Detect security vulnerabilities and PII/secrets"

    def validate(self, skill_path: Path) -> ValidationResult:
        """Run both security and PII scanning on skill(s)."""
        return self._with_pii_waivers(
            skill_path,
            lambda: self._validate_with_applicability(
                skill_path,
                self._validate_single_skill,
                action_description="Scanning",
            ),
        )

    def validate_security_only(self, skill_path: Path) -> ValidationResult:
        """Run only skillspector security scanning (no PII detection)."""
        result = self._validate_with_applicability(
            skill_path,
            self._run_skillspector,
            action_description="Security scanning",
        )
        if self.verify_llm and result.findings:
            self._verify_findings_with_llm(result, skill_path)
        return result

    def validate_pii_only(self, skill_path: Path) -> ValidationResult:
        """Run only PII detection (no skillspector security scan)."""

        def scan() -> ValidationResult:
            result = self._validate_folder_or_skill(
                skill_path,
                self._scan_for_pii,
                action_description="PII scanning",
            )
            if self.verify_llm and result.findings:
                self._verify_findings_with_llm(result, skill_path)
            return result

        return self._with_pii_waivers(skill_path, scan)

    def _with_pii_waivers(self, skill_path: Path, scan: Callable[[], ValidationResult]) -> ValidationResult:
        """Run ``scan`` of ``skill_path`` with the configured PII waivers, then report their outcomes once.

        The outcomes are added after the scan (and any LLM verification) at the
        top level, so a folder run reports them once, not per skill. Only the
        waivers for files the scan covered are reported.
        """
        if self.pii_waiver_path is None:
            return scan()
        waivers = PIIWaiverSet.load(self.pii_waiver_path, configured_pii_categories(self.pii_patterns))
        self._pii_waivers = waivers
        try:
            result = scan()
        finally:
            self._pii_waivers = None
        for finding, is_error in waivers.findings(skill_path):
            result.add_structured_finding(finding, is_error=is_error)
        if waivers.applied:
            result.add_message(f"PII waivers suppressed {len(waivers.applied)} reviewed PII value(s)")
        return result

    def _validate_with_applicability(
        self,
        skill_path: Path,
        single_skill_validator: Callable[[Path], ValidationResult],
        *,
        action_description: str,
    ) -> ValidationResult:
        """Validate each discovered skill and total their SkillSpector applicability counters.

        The folder walker does not merge per-skill metadata, so the counters
        are collected as each skill finishes. The counts of referenced
        artifacts that SkillSpector did not completely inspect are totaled like
        the applicability counters.

        Without a configured ``skillspector_reference_root``, the target is the
        reference root of every scan of the walk
        (:func:`_skillspector_missing_reference_notes`). When *skill_path* is a
        directory but not a skill, each skill found in it is scanned on its
        own, and its instructions may name files elsewhere in the folder that
        its scan does not inspect, so their missing references are not
        accepted. A configured ``target_root`` replaces the target, such as
        the folder of skills that a skill passed here belongs to. In a scan
        run outside this walk, :func:`_skillspector_missing_reference_notes`
        accepts missing references only when the configured
        ``skillspector_reference_root`` is the scanned root. Otherwise, as for
        the per-partition scans of ``tier1/plugin_checks.py``, that rule
        accepts none. Such a scan without a configured root has no known
        target, so a 2.13.3 report that :func:`reference_caveat_metadata`
        accepts still passes as its caveat.
        """
        blocks: list[dict] = []
        reference_summaries: list[object] = []

        def validate_one(path: Path) -> ValidationResult:
            child = single_skill_validator(path)
            block = child.metadata.get("skillspector_applicability")
            if isinstance(block, dict):
                blocks.append(block)
            if REFERENCE_SUMMARY_KEY in child.metadata:
                reference_summaries.append(child.metadata[REFERENCE_SUMMARY_KEY])
            return child

        target_root = self.target_root or skill_path.resolve()
        previous_target_root, self._scan_target_root = self._scan_target_root, target_root
        try:
            result = self._validate_folder_or_skill(skill_path, validate_one, action_description=action_description)
        finally:
            self._scan_target_root = previous_target_root
        if blocks:
            result.metadata["skillspector_applicability"] = _combine_skillspector_applicability(blocks)
        if (references := combine_reference_summaries(reference_summaries)) is not None:
            result.metadata[REFERENCE_SUMMARY_KEY] = references
        return result

    def _validate_single_skill(self, skill_path: Path) -> ValidationResult:
        """Run complete security + PII scan on a single skill directory."""
        result = ValidationResult()
        result.merge(self._run_skillspector(skill_path))
        result.merge(self._scan_for_pii(skill_path))

        if self.verify_llm and result.findings:
            self._verify_findings_with_llm(result, skill_path)

        return result

    def _run_skillspector(self, skill_path: Path) -> ValidationResult:
        """Execute skillspector security scan via CLI on a single directory.

        Runs `skillspector scan <path> --format json`; adds `--no-llm` when
        LLM analysis is disabled for static-only analysis.

        skillspector has no exclude flag and reads every file it finds, so
        when the skill carries generated output dirs (``evals/results/``
        snapshots can reach hundreds of MB after Tier 3 runs) the scan runs
        on a temp copy without those outputs. Shipped bytecode remains for
        artifact-integrity analysis, and reported paths map back to the
        original location.

        An explicitly selected baseline is passed to both stages with
        ``--baseline``. When it lives inside a staged skill, the staged copy is
        passed instead so SkillSpector excludes the file it actually scans.
        """
        result = ValidationResult()

        if not Tools.skillspector.is_available:
            result.add_warning(
                f"skillspector not installed - skipping security scan. {Tools.skillspector.get_install_hint()}"
            )
            result.mark_scan_incomplete("skillspector-llm" if self.use_llm else "skillspector")
            return result

        baseline_state: dict | None = None
        if self.skillspector_baseline is not None:
            baseline_state, baseline_error = _inspect_skillspector_baseline(self.skillspector_baseline)
            if baseline_error is not None:
                baseline_state["message"] = baseline_error
                result.metadata["skillspector_baseline"] = baseline_state
                result.add_error(f"{baseline_error}; security scan did not run")
                result.mark_scan_incomplete("skillspector-llm" if self.use_llm else "skillspector")
                return result

        original_root = skill_path.resolve()
        scan_root = original_root
        staged: tempfile.TemporaryDirectory | None = None
        if skill_path.is_dir() and _tree_contains_artifact_dirs(skill_path):
            staged = tempfile.TemporaryDirectory(prefix="skillspector-scan-")
            try:
                copy_root = Path(staged.name) / original_root.name
                shutil.copytree(original_root, copy_root, symlinks=True, ignore=_ignore_artifact_dirs)
                scan_root = copy_root
            except (OSError, shutil.Error) as exc:
                staged.cleanup()
                staged = None
                result.add_warning(f"Could not stage artifact-free skill copy ({exc}); scanning in place.")

        try:
            baseline = _skillspector_baseline_for_scan(original_root, scan_root, self.skillspector_baseline)
            # The deterministic stage is authoritative and always runs, even
            # when LLM enrichment was requested. Enrichment is additive and
            # cannot erase static findings.
            result = self._run_skillspector_once(
                scan_root,
                original_root=original_root if staged is not None else None,
                use_llm=False,
                baseline=baseline,
            )
            # Merging the enrichment stage overwrites this metadata, so keep
            # each stage's own suppression count for the baseline outcome.
            static_suppressed = result.metadata.get("skillspector_suppressed_count")
            static_summary = result.metadata.get("skillspector_suppressed_summary")
            llm_suppressed = None
            if self.use_llm and not result.is_incomplete:
                enrichment = self._run_skillspector_once(
                    scan_root,
                    original_root=original_root if staged is not None else None,
                    use_llm=True,
                    baseline=baseline,
                )
                if not enrichment.is_incomplete:
                    llm_suppressed = enrichment.metadata.get("skillspector_suppressed_count")
                # Both stages run the same static analyzers, so a note such as
                # an accepted image gap can repeat word for word; keep one copy.
                finding_messages = {finding.to_legacy_string() for finding in enrichment.findings}
                repeated = {
                    warning
                    for warning in enrichment.warnings
                    if warning in result.warnings and warning not in finding_messages
                }
                if repeated:
                    kept = [warning for warning in enrichment.warnings if warning not in repeated]
                    enrichment.summary.warnings -= len(enrichment.warnings) - len(kept)
                    enrichment.warnings = kept
                # The static stage is the authoritative audit of what was
                # suppressed; keep its block and record the LLM stage's apart.
                static_applicability = result.metadata.get("skillspector_applicability")
                llm_applicability = enrichment.metadata.pop("skillspector_applicability", None)
                self._drop_repeated_hook_reports(result, enrichment)
                result.merge(enrichment)
                if static_applicability is not None:
                    result.metadata["skillspector_applicability"] = static_applicability
                if llm_applicability is not None:
                    result.metadata["skillspector_llm_applicability"] = llm_applicability
                if not enrichment.is_incomplete:
                    self._deduplicate_skillspector_findings(result)
                # Count each reference line once across both stages.
                _record_skillspector_reference_summary(result)
            if baseline_state is not None:
                result.metadata["skillspector_baseline"] = baseline_state
                self._report_skillspector_baseline(
                    result,
                    baseline_state,
                    static_suppressed=static_suppressed,
                    llm_suppressed=llm_suppressed,
                    static_summary=static_summary,
                )
            return result
        finally:
            if staged is not None:
                staged.cleanup()

    def _run_skillspector_once(
        self,
        scan_root: Path,
        *,
        original_root: Path | None,
        use_llm: bool,
        baseline: Path | None = None,
    ) -> ValidationResult:
        """Run one SkillSpector stage and validate its process/report contract."""
        result = ValidationResult()
        stage_name = "skillspector-llm" if use_llm else "skillspector"
        args = ["scan", str(scan_root), "--format", "json"]
        if baseline is not None:
            args.extend(["--baseline", str(baseline)])
        if not use_llm:
            args.append("--no-llm")

        logger.info("Running %s on %s", stage_name, scan_root)
        if use_llm:
            child_env, provider_id = _skillspector_llm_env()
            # Requested LLM configuration; None leaves the choice to SkillSpector.
            result.metadata["skillspector_model"] = child_env.get("SKILLSPECTOR_MODEL")
            result.metadata["skillspector_reasoning_effort"] = child_env.get("SKILLSPECTOR_REASONING_EFFORT")
            if provider_id is not None:
                result.metadata["skillspector_provider_id"] = provider_id
        else:
            child_env = _skillspector_process_env()
        # Read from the parent environment: the isolated child environment omits it.
        timeout = _skillspector_timeout_seconds()
        result.metadata["skillspector_timeout_seconds"] = timeout
        tool_result = Tools.skillspector.run(args, timeout=timeout, env=child_env, replace_env=True)
        try:
            data = parse_json_output(tool_result.stdout)
        except (RecursionError, ValueError):
            data = None
        if use_llm:
            # Recorded before the completeness checks below, so a failed or
            # unconfirmed LLM stage still names the provider and models it
            # attempted. Only a record that passes the strict provenance
            # contract is kept, and it never makes the stage complete.
            provenance = _skillspector_report_llm_provenance(data)
            if provenance is not None:
                result.metadata["skillspector_llm_provenance"] = provenance

        if tool_result.exit_code not in _SKILLSPECTOR_POLICY_EXIT_CODES:
            error_message = tool_result.error_message
            if not error_message and Tools.skillspector.resolution_source == "uv tool run":
                error_message = prepare_skillspector(Tools.skillspector.path)
            hint = Tools.skillspector.get_install_hint() if (error_message or "").startswith("couldn't ") else ""
            detail = (
                f"{error_message}. {hint}".rstrip(". ")
                if error_message
                else (
                    f"{stage_name} failed with unexpected exit code {tool_result.exit_code}: "
                    "scanner diagnostics were redacted"
                )
            )
            if (
                use_llm
                and not error_message
                and tool_result.exit_code == _SKILLSPECTOR_EXECUTION_FAILED_EXIT_CODE
                and _skillspector_provider_retries_exhausted(data)
            ):
                # The stage stays incomplete; only the explanation changes.
                detail = _skillspector_provider_retry_error(stage_name)
            elif (
                # SkillSpector uses exit code 2 for both; a provider outage report
                # proves it accepted its input, including the baseline.
                baseline is not None
                and tool_result.error_message is None
                and tool_result.exit_code == _SKILLSPECTOR_INPUT_ERROR_EXIT_CODE
            ):
                detail += (
                    f". SkillSpector exits {_SKILLSPECTOR_INPUT_ERROR_EXIT_CODE} when it rejects its input, "
                    "including an invalid --baseline file; check the baseline with "
                    "`skillspector scan <skill> --baseline <file>`"
                )
            result.add_error(detail)
            result.mark_scan_incomplete(stage_name)
            return result

        if use_llm and _skillspector_llm_stderr_failed(tool_result.stderr):
            result.add_error(
                "skillspector-llm reported failed LLM analysis; provider or model diagnostics were redacted"
            )
            result.mark_scan_incomplete(stage_name)
            return result

        if data is None:
            result.add_error(
                f"{stage_name} did not return valid JSON (exit code {tool_result.exit_code}); "
                "security scan did not complete"
            )
            result.mark_scan_incomplete(stage_name)
            return result
        if not isinstance(data, dict):
            result.add_error(f"{stage_name} JSON output was not an object; security scan did not complete")
            result.mark_scan_incomplete(stage_name)
            return result
        result.record_scanner("skillspector", Tools.skillspector.version)

        if original_root is not None:
            try:
                data = _rewrite_path_prefix(data, str(scan_root), str(original_root))
            except RecursionError:
                result.add_error(f"{stage_name} JSON output was nested too deeply; security scan did not complete")
                result.mark_scan_incomplete(stage_name)
                return result

        if not self._validate_skillspector_report(
            data,
            tool_result.exit_code,
            use_llm,
            stage_name,
            result,
            baseline_requested=baseline is not None,
            scanned_root=original_root if original_root is not None else scan_root,
            reference_root=self.skillspector_reference_root or self._scan_target_root,
        ):
            result.mark_scan_incomplete(stage_name)
            return result

        metadata = data.get("metadata") or {}
        if use_llm and not (metadata.get("llm_requested") is True and metadata.get("llm_available") is True):
            result.add_error(
                "skillspector-llm did not confirm available LLM analysis; provider or model diagnostics were redacted"
            )
            result.mark_scan_incomplete(stage_name)
            return result

        self._process_skillspector_cli_result(data, result, source_root=original_root or scan_root)
        return result

    @staticmethod
    def _deduplicate_skillspector_findings(result: ValidationResult) -> None:
        """Deduplicate findings repeated by static and enriched reports."""
        unique = []
        seen: set[tuple[str, str, int | None, str]] = set()
        for finding in result.findings:
            key = (finding.check_name, finding.file_path, finding.line_number, finding.message)
            if key not in seen:
                seen.add(key)
                unique.append(finding)
        if len(unique) != len(result.findings):
            errors = result.independent_errors()
            warnings = result.independent_warnings()
            result.findings = unique
            result.recalculate_from_findings()
            for error in errors:
                if error not in result.errors:
                    result.add_error(error)
            for warning in warnings:
                if warning not in result.warnings:
                    result.add_warning(warning)

    @staticmethod
    def _drop_repeated_hook_reports(result: ValidationResult, enrichment: ValidationResult) -> None:
        """Drop the hook Criticals and gate failures of *enrichment* that the static stage's *result* already holds.

        Both stages run the same static analyzers, so each reports the same
        manifests (:meth:`_report_unverified_hooks`). They are dropped before
        the merge whether or not the enrichment stage completed:
        :meth:`_deduplicate_skillspector_findings` runs only after a complete
        stage, because its rebuild would drop an incomplete stage's error.
        """
        reported = {
            (finding.file_path, finding.message)
            for finding in result.findings
            if finding.check_name == _SKILLSPECTOR_HOOK_UNVERIFIED_CHECK
        }
        kept: list[Finding] = []
        for finding in enrichment.findings:
            legacy = finding.to_legacy_string()
            if (
                finding.check_name == _SKILLSPECTOR_HOOK_UNVERIFIED_CHECK
                and finding.severity == Severity.CRITICAL
                and (finding.file_path, finding.message) in reported
                and legacy in enrichment.errors
            ):
                enrichment.errors.remove(legacy)
                enrichment.summary.errors -= 1
                enrichment.summary.critical_count -= 1
            else:
                kept.append(finding)
        enrichment.findings = kept
        repeated = [message for message in enrichment.gate_failures if message in result.gate_failures]
        for message in repeated:
            if message in enrichment.errors:
                enrichment.errors.remove(message)
                enrichment.summary.errors -= 1
        if repeated:
            enrichment.metadata["gate_failures"] = [
                message for message in enrichment.gate_failures if message not in repeated
            ]

    @staticmethod
    def _validate_skillspector_report(
        data: dict,
        process_exit_code: int,
        use_llm: bool,
        stage_name: str,
        result: ValidationResult,
        *,
        baseline_requested: bool = False,
        scanned_root: Path | None = None,
        reference_root: Path | None = None,
    ) -> bool:
        """Return whether JSON is a trustworthy SkillSpector findings report.

        Suppressed findings are accepted only when SkillEvaluator passed an
        explicit baseline, and that report must then account for them.

        A ``partial`` analysis makes the scan INCOMPLETE unless its only gaps are
        missing-reference caveats in a fully inspected SkillSpector 2.13.3 or
        later 2.x report (:func:`reference_caveat_metadata`), references to
        files the skill does not bundle
        (:func:`_skillspector_missing_reference_notes`), referenced PNG or JPEG
        images (:func:`_skillspector_passive_raster_note`), or both of the last
        two, each confirmed by the files under *scanned_root*. Each kind of gap
        is then a visible warning, not a claim that referenced targets were
        inspected or that the scanner reported SAFE. *scanned_root* is the
        directory the report's relative paths start from. *reference_root* is
        the root of the artifact that contains it: the validator's
        ``skillspector_reference_root``, or else its ``target_root`` or the
        target of the scan in progress. Missing references are accepted by
        :func:`_skillspector_missing_reference_notes` only when it is
        *scanned_root*. When *reference_root* is given, a caveat that
        :func:`reference_caveat_metadata` accepts must also pass
        :func:`_skillspector_missing_reference_notes`; without it, that rule
        alone decides.

        Rows for hooks SkillSpector parsed but could not model are judged
        apart: when they account exactly for their part of the report
        (:func:`_skillspector_without_hook_surface`), the rest of the report
        decides whether the scan is complete, and
        :meth:`_process_skillspector_cli_result` reports a Critical gate
        failure for each manifest instead. :func:`reference_caveat_metadata`
        reads the whole report, hook rows included, so its report-only
        missing-reference caveat does not apply beside the hooks: without a
        *reference_root*, a missing reference there records the scan as
        incomplete next to the Critical.

        From SkillSpector 2.13.4, an analysis whose only gaps are missing
        references is ``complete_with_caveats`` instead of ``partial``. Its
        caveats get the warning of :func:`_skillspector_missing_reference_notes`
        when they pass its rules (:func:`_skillspector_reference_caveat_notes`)
        and the rest of the report is accepted. Otherwise the scan is
        INCOMPLETE and, as for a ``partial`` report, the report's findings are
        still checked and kept. :func:`reference_caveat_metadata` reads only
        ``partial`` reports, so it never accepts these caveats, even without a
        *reference_root*. A report with that status and no caveat, or with a
        caveat that is not a missing reference, contradicts itself and is not
        trusted, and neither is the status from an earlier release.
        """
        partial_reported = False
        # Rejected caveats mark the scan INCOMPLETE below, but SkillSpector still
        # keeps a LOW complete_with_caveats report SAFE, so the recommendation
        # check reads the state the result had on entry.
        incomplete_before = result.is_incomplete
        # Accepted caveats are reported only once every later check has passed.
        caveat_warnings: list[str] = []
        accepted_reference_caveats = None
        if "error" in data and data["error"] is not None:
            result.add_error("skillspector reported an error; security scan did not complete")
            return False
        if "errors" in data and (not isinstance(data["errors"], list) or data["errors"]):
            result.add_error("skillspector reported errors; security scan did not complete")
            return False
        for field in ("failure", "failed"):
            marker = data.get(field)
            if marker is not None and not isinstance(marker, bool):
                result.add_error(f"skillspector JSON field '{field}' must be a boolean; security scan did not complete")
                return False
            if marker is True:
                result.add_error("skillspector reported a failure; security scan did not complete")
                return False
        if data.get("success") is False:
            result.add_error("skillspector reported success=false; security scan did not complete")
            return False
        if data.get("success") is not None and not isinstance(data["success"], bool):
            result.add_error("skillspector JSON field 'success' must be a boolean; security scan did not complete")
            return False

        report_metadata = data.get("metadata")
        version_error: str | None = None
        skillspector_version: tuple[int, int, int] | None = None
        if isinstance(report_metadata, dict):
            raw_version = report_metadata.get("skillspector_version")
            if not isinstance(raw_version, str) or SEMVER_RE.fullmatch(raw_version) is None:
                version_error = (
                    "skillspector JSON field 'metadata.skillspector_version' must be a semantic version; "
                    "security scan did not complete"
                )
            else:
                major, minor, patch = (int(part) for part in raw_version.split("."))
                skillspector_version = (major, minor, patch)
        elif report_metadata is None:
            version_error = (
                "skillspector JSON report is missing "
                "'metadata.skillspector_version'; security scan did not complete"
            )
        uses_completeness_schema = (
            skillspector_version is not None
            and skillspector_version >= _SKILLSPECTOR_COMPLETENESS_SCHEMA_VERSION
        )
        uses_statusless_completeness_schema = (
            skillspector_version in _SKILLSPECTOR_STATUSLESS_COMPLETENESS_VERSIONS
        )
        uses_finding_identity = (
            skillspector_version is not None
            and skillspector_version >= _SKILLSPECTOR_FINDING_IDENTITY_VERSION
        )
        uses_versioned_completeness = uses_completeness_schema or uses_statusless_completeness_schema
        completeness_contract = "2.10+" if uses_completeness_schema else "2.9.5/2.9.6"

        execution_successful = data.get("execution_successful")
        if uses_versioned_completeness and "execution_successful" not in data:
            result.add_error(
                f"skillspector {completeness_contract} JSON report is missing required "
                "'execution_successful' field; "
                "security scan did not complete"
            )
            return False
        if "execution_successful" in data and not isinstance(execution_successful, bool):
            result.add_error(
                "skillspector JSON field 'execution_successful' must be a boolean; "
                "security scan did not complete"
            )
            return False
        if execution_successful is False:
            if use_llm and _skillspector_provider_retries_exhausted(data):
                result.add_error(_skillspector_provider_retry_error(stage_name))
            else:
                result.add_error("skillspector reported execution_successful=false; security scan did not complete")
            return False

        findings_after_filtering: int | None = None
        universal_analyzer_evidence_valid = True
        analysis_completeness = data.get("analysis_completeness")
        if uses_versioned_completeness and "analysis_completeness" not in data:
            result.add_error(
                f"skillspector {completeness_contract} JSON report is missing required "
                "'analysis_completeness' object; "
                "security scan did not complete"
            )
            return False
        if "analysis_completeness" in data:
            if not isinstance(analysis_completeness, dict):
                result.add_error(
                    "skillspector JSON field 'analysis_completeness' must be an object; "
                    "security scan did not complete"
                )
                return False
            if skillspector_version is not None and not uses_versioned_completeness:
                result.add_error(
                    "skillspector JSON report uses an unsupported pre-2.10 completeness schema; "
                    "security scan did not complete"
                )
                return False
            completeness_execution_successful = analysis_completeness.get("execution_successful")
            if uses_versioned_completeness and "execution_successful" not in analysis_completeness:
                result.add_error(
                    "skillspector JSON field 'analysis_completeness.execution_successful' is required; "
                    "security scan did not complete"
                )
                return False
            if "execution_successful" in analysis_completeness and not isinstance(
                completeness_execution_successful,
                bool,
            ):
                result.add_error(
                    "skillspector JSON field 'analysis_completeness.execution_successful' must be a boolean; "
                    "security scan did not complete"
                )
                return False
            if (
                execution_successful is not None
                and execution_successful is not completeness_execution_successful
            ):
                result.add_error(
                    "skillspector JSON execution_successful fields contradict each other; "
                    "security scan did not complete"
                )
                return False
            if completeness_execution_successful is False:
                result.add_error(
                    "skillspector JSON field 'analysis_completeness.execution_successful' is false; "
                    "security scan did not complete"
                )
                return False

            if uses_versioned_completeness:
                is_complete = analysis_completeness.get("is_complete")
                if not isinstance(is_complete, bool):
                    result.add_error(
                        "skillspector JSON field 'analysis_completeness.is_complete' must be a boolean; "
                        "security scan did not complete"
                    )
                    return False
                if uses_completeness_schema:
                    completeness_status = analysis_completeness.get("status")
                    recognized_statuses = {"complete", "partial", "failed"}
                    if skillspector_version >= _SKILLSPECTOR_CAVEATS_STATUS_VERSION:
                        recognized_statuses.add("complete_with_caveats")
                    if not isinstance(completeness_status, str) or completeness_status not in recognized_statuses:
                        result.add_error(
                            "skillspector JSON field 'analysis_completeness.status' is not recognized; "
                            "security scan did not complete"
                        )
                        return False
                    if completeness_status == "failed":
                        result.add_error(
                            "skillspector JSON field 'analysis_completeness' reports failed analysis; "
                            "security scan did not complete"
                        )
                        return False
                elif "status" in analysis_completeness:
                    result.add_error(
                        "skillspector 2.9.5/2.9.6 JSON field 'analysis_completeness.status' must be absent; "
                        "security scan did not complete"
                    )
                    return False

                count_fields = (
                    "total_components",
                    "scanned_components",
                    "fully_inspected_files",
                    "partially_inspected_files",
                    "entirely_uninspected_files",
                    "findings_before_filtering",
                    "findings_after_filtering",
                )
                counts: dict[str, int] = {}
                for field in count_fields:
                    value = analysis_completeness.get(field)
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        result.add_error(
                            f"skillspector JSON field 'analysis_completeness.{field}' must be a "
                            "non-negative integer; security scan did not complete"
                        )
                        return False
                    counts[field] = value
                findings_before_filtering = counts["findings_before_filtering"]
                findings_after_filtering = counts["findings_after_filtering"]
                if findings_before_filtering < findings_after_filtering or (
                    uses_completeness_schema
                    and (
                        bool(findings_before_filtering) is not bool(findings_after_filtering)
                        or (is_complete and findings_before_filtering != findings_after_filtering)
                    )
                ):
                    result.add_error(
                        "skillspector JSON field 'analysis_completeness' has inconsistent finding counts; "
                        "security scan did not complete"
                    )
                    return False

                coverage_percent = analysis_completeness.get("coverage_percent")
                if (
                    isinstance(coverage_percent, bool)
                    or not isinstance(coverage_percent, (int, float))
                    or not 0 <= coverage_percent <= 100
                ):
                    result.add_error(
                        "skillspector JSON field 'analysis_completeness.coverage_percent' must be a "
                        "finite number from 0 to 100; security scan did not complete"
                    )
                    return False
                ledger_exceptions = analysis_completeness.get("ledger_exceptions")
                limitations = analysis_completeness.get("limitations")
                if (
                    not isinstance(ledger_exceptions, list)
                    or not all(
                        isinstance(exception, dict) and isinstance(exception.get("fatal"), bool)
                        for exception in ledger_exceptions
                    )
                    or not isinstance(limitations, list)
                    or not all(isinstance(limitation, str) for limitation in limitations)
                ):
                    result.add_error(
                        "skillspector JSON field 'analysis_completeness' has invalid detail lists; "
                        "security scan did not complete"
                    )
                    return False
                if any(exception["fatal"] for exception in ledger_exceptions):
                    result.add_error(
                        "skillspector JSON field 'analysis_completeness' has a fatal exception despite "
                        "successful execution; security scan did not complete"
                    )
                    return False

                analyzer_statuses = analysis_completeness.get("analyzer_statuses")
                if not isinstance(analyzer_statuses, list) or not analyzer_statuses:
                    result.add_error(
                        "skillspector JSON field 'analysis_completeness.analyzer_statuses' must be a "
                        "non-empty list; security scan did not complete"
                    )
                    return False
                expected_limitations: list[str] = []
                observed_analyzer_ids: set[str] = set()
                analyzer_evidence: dict[str, list[tuple[str, dict[str, int]]]] = {}
                not_applicable_analyzer_ids: set[str] = set()
                not_applicable_evidence_valid = True
                for index, analyzer_status in enumerate(analyzer_statuses):
                    if not isinstance(analyzer_status, dict):
                        result.add_error(
                            "skillspector JSON field "
                            f"'analysis_completeness.analyzer_statuses[{index}]' must be an object; "
                            "security scan did not complete"
                        )
                        return False
                    analyzer_id = analyzer_status.get("analyzer_id")
                    analyzer_state = analyzer_status.get("status")
                    if (
                        not isinstance(analyzer_id, str)
                        or not analyzer_id
                        or not isinstance(analyzer_state, str)
                    ):
                        result.add_error(
                            "skillspector JSON field 'analysis_completeness.analyzer_statuses' has "
                            "invalid analyzer evidence; security scan did not complete"
                        )
                        return False
                    observed_analyzer_ids.add(analyzer_id)
                    for field in ("reason_code", "message"):
                        value = analyzer_status.get(field)
                        if value is not None and not isinstance(value, str):
                            result.add_error(
                                "skillspector JSON field 'analysis_completeness.analyzer_statuses' has "
                                f"a non-string '{field}'; security scan did not complete"
                            )
                            return False
                    outcome_fields = (
                        ("completed", "partial", "skipped", "failed", "unaccounted")
                        if uses_completeness_schema
                        else ("completed", "skipped", "failed", "unaccounted")
                    )
                    analyzer_counts: dict[str, int] = {}
                    for field in ("planned_work", *outcome_fields):
                        value = analyzer_status.get(field)
                        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                            result.add_error(
                                "skillspector JSON field 'analysis_completeness.analyzer_statuses' has "
                                f"invalid '{field}' accounting; security scan did not complete"
                            )
                            return False
                        analyzer_counts[field] = value
                    analyzer_evidence.setdefault(analyzer_id, []).append(
                        (analyzer_state, analyzer_counts)
                    )
                    if analyzer_counts["planned_work"] != sum(
                        analyzer_counts[field] for field in outcome_fields
                    ):
                        result.add_error(
                            "skillspector JSON field 'analysis_completeness.analyzer_statuses' has "
                            "inconsistent work accounting; security scan did not complete"
                        )
                        return False
                    if analyzer_counts["unaccounted"]:
                        result.add_error(
                            "skillspector JSON field 'analysis_completeness.analyzer_statuses' reports "
                            "unaccounted work despite successful execution; security scan did not complete"
                        )
                        return False
                    if analyzer_counts["planned_work"]:
                        expected_analyzer_state = (
                            "failed"
                            if analyzer_counts["failed"]
                            else "degraded"
                            if any(analyzer_counts.get(field, 0) for field in ("partial", "skipped"))
                            else "completed"
                        )
                        if analyzer_state != expected_analyzer_state:
                            result.add_error(
                                "skillspector JSON field 'analysis_completeness.analyzer_statuses' has "
                                "a status that contradicts its work accounting; security scan did not complete"
                            )
                            return False
                    incomplete_outcomes = outcome_fields[1:]
                    if (is_complete or uses_statusless_completeness_schema) and any(
                        analyzer_counts[field] for field in incomplete_outcomes
                    ):
                        result.add_error(
                            "skillspector JSON field 'analysis_completeness.analyzer_statuses' contradicts "
                            "complete analysis; security scan did not complete"
                        )
                        return False
                    if analyzer_state == "disabled":
                        if (
                            use_llm
                            or analyzer_id not in _SKILLSPECTOR_OPTIONAL_ANALYZERS
                            or analyzer_status.get("reason_code") != "disabled_by_configuration"
                        ):
                            result.add_error(
                                "skillspector JSON field 'analysis_completeness.analyzer_statuses' "
                                "reports an unexpected disabled analyzer; security scan did not complete"
                            )
                            return False
                        if uses_statusless_completeness_schema:
                            expected_limitations.append(_SKILLSPECTOR_DISABLED_ANALYZER_LIMITATION)
                    elif analyzer_state in {"failed", "degraded", "unavailable"}:
                        if uses_statusless_completeness_schema or is_complete:
                            result.add_error(
                                "skillspector JSON field 'analysis_completeness.analyzer_statuses' reports "
                                f"incomplete analyzer '{analyzer_id}'; security scan did not complete"
                            )
                            return False
                        message = analyzer_status.get("message")
                        expected_limitations.append(
                            message
                            if isinstance(message, str) and message
                            else f"Analyzer {analyzer_id} status: {analyzer_state}."
                        )
                    elif analyzer_state == "not_applicable":
                        not_applicable_analyzer_ids.add(analyzer_id)
                        not_applicable_evidence_valid &= (
                            analyzer_status.get("reason_code") == "no_applicable_files"
                            and not any(analyzer_counts.values())
                        )
                    elif analyzer_state not in {"completed", "not_applicable"}:
                        result.add_error(
                            "skillspector JSON field 'analysis_completeness.analyzer_statuses' reports "
                            f"unknown analyzer status '{analyzer_state}'; security scan did not complete"
                        )
                        return False

                required_analyzer_ids = (
                    _SKILLSPECTOR_2_9_6_REQUIRED_ANALYZERS
                    if uses_statusless_completeness_schema
                    else _SKILLSPECTOR_2_10_REQUIRED_ANALYZERS
                )
                if skillspector_version >= (2, 11, 0):
                    required_analyzer_ids |= {"bundled_execution_surface"}
                if (
                    uses_completeness_schema
                    and use_llm
                    and report_metadata.get("llm_available") is True
                ):
                    required_analyzer_ids |= _SKILLSPECTOR_SEMANTIC_ANALYZERS
                if not required_analyzer_ids.issubset(observed_analyzer_ids):
                    result.add_error(
                        "skillspector JSON field 'analysis_completeness.analyzer_statuses' is "
                        "missing required analyzer evidence; security scan did not complete"
                    )
                    return False
                docs_only_analyzer_states_valid = all(
                    all(
                        state
                        == (
                            "not_applicable"
                            if analyzer_id in _SKILLSPECTOR_DOCS_ONLY_NOT_APPLICABLE_ANALYZERS
                            else "completed"
                        )
                        for state, _accounting in analyzer_evidence[analyzer_id]
                    )
                    for analyzer_id in required_analyzer_ids
                )
                complete_by_applicability = (
                    uses_completeness_schema
                    and skillspector_version == _SKILLSPECTOR_DOCS_ONLY_APPLICABILITY_VERSION
                    and not is_complete
                    and completeness_status == "partial"
                    and coverage_percent == 100
                    and counts["partially_inspected_files"] == 0
                    and counts["entirely_uninspected_files"] == 0
                    and not ledger_exceptions
                    and not limitations
                    and report_metadata.get("has_executable_scripts") is False
                    and not_applicable_analyzer_ids
                    == _SKILLSPECTOR_DOCS_ONLY_NOT_APPLICABLE_ANALYZERS
                    and not_applicable_evidence_valid
                    and docs_only_analyzer_states_valid
                )
                accepted_reference_caveats = reference_caveat_metadata(
                    data,
                    scanner_version=skillspector_version,
                    required_analyzers=required_analyzer_ids,
                    universal_analyzers=_SKILLSPECTOR_COMMON_UNIVERSAL_ANALYZERS | {"artifact_integrity"},
                    optional_analyzers=_SKILLSPECTOR_OPTIONAL_ANALYZERS,
                    use_llm=use_llm,
                )
                if (
                    is_complete
                    or uses_statusless_completeness_schema
                    or complete_by_applicability
                    or accepted_reference_caveats is not None
                ) and counts["total_components"]:
                    universal_analyzer_ids = _SKILLSPECTOR_COMMON_UNIVERSAL_ANALYZERS | (
                        {"artifact_integrity"} if uses_completeness_schema else set()
                    )
                    universal_analyzer_evidence_valid = all(
                        all(state == "completed" for state, _item in analyzer_evidence[analyzer_id])
                        and sum(item["planned_work"] for _state, item in analyzer_evidence[analyzer_id])
                        == counts["total_components"]
                        and sum(item["completed"] for _state, item in analyzer_evidence[analyzer_id])
                        == counts["total_components"]
                        for analyzer_id in universal_analyzer_ids
                    )
                actual_limitation_counts = Counter(limitations)
                expected_limitation_counts = Counter(expected_limitations)
                if (
                    uses_statusless_completeness_schema
                    and actual_limitation_counts != expected_limitation_counts
                ) or (
                    uses_completeness_schema
                    and bool(expected_limitation_counts - actual_limitation_counts)
                ):
                    result.add_error(
                        "skillspector JSON field 'analysis_completeness.limitations' contradicts "
                        "analyzer evidence; security scan did not complete"
                    )
                    return False

                if uses_completeness_schema and is_complete is not (
                    completeness_status in {"complete", "complete_with_caveats"}
                ):
                    # SkillSpector rewrites a truncated complete_with_caveats
                    # report to partial, so only "complete" can keep its status.
                    has_report_stage_truncation = any(
                        limitation.startswith("Transitive traversal truncated: ")
                        for limitation in limitations
                    )
                    if not (
                        not is_complete
                        and completeness_status == "complete"
                        and has_report_stage_truncation
                    ):
                        result.add_error(
                            "skillspector JSON field 'analysis_completeness' has contradictory status markers; "
                            "security scan did not complete"
                        )
                        return False

                if (
                    counts["scanned_components"] != counts["fully_inspected_files"]
                    or counts["total_components"]
                    != counts["fully_inspected_files"]
                    + counts["partially_inspected_files"]
                    + counts["entirely_uninspected_files"]
                ):
                    result.add_error(
                        "skillspector JSON field 'analysis_completeness' has inconsistent counters or coverage; "
                        "security scan did not complete"
                    )
                    return False

                expected_coverage = (
                    round(counts["fully_inspected_files"] / counts["total_components"] * 100, 1)
                    if counts["total_components"]
                    else 100.0
                )
                if coverage_percent != expected_coverage:
                    result.add_error(
                        "skillspector JSON field 'analysis_completeness' has inconsistent counters or coverage; "
                        "security scan did not complete"
                    )
                    return False

                if uses_statusless_completeness_schema:
                    if (
                        counts["partially_inspected_files"] != 0
                        or counts["entirely_uninspected_files"] != 0
                        or ledger_exceptions
                        or coverage_percent != 100
                        or is_complete is not (not limitations)
                    ):
                        result.add_error(
                            "skillspector 2.9.5/2.9.6 JSON field 'analysis_completeness' does not describe "
                            "a fully covered scan; security scan did not complete"
                        )
                        return False
                elif is_complete:
                    with_caveats = completeness_status == "complete_with_caveats"
                    if (
                        counts["partially_inspected_files"] != 0
                        or counts["entirely_uninspected_files"] != 0
                        or (ledger_exceptions and not with_caveats)
                        or limitations
                    ):
                        result.add_error(
                            "skillspector JSON field 'analysis_completeness' details contradict complete analysis; "
                            "security scan did not complete"
                        )
                        return False
                    if with_caveats:
                        # Caveats do not make the analysis partial: SkillSpector
                        # keeps a LOW report SAFE, so partial_reported stays False.
                        if not _skillspector_rows_are_reference_caveats(ledger_exceptions):
                            # SkillSpector reports any other gap as partial or failed.
                            result.add_error(_skillspector_reference_caveat_error(ledger_exceptions))
                            return False
                        caveat_notes = _skillspector_reference_caveat_notes(
                            ledger_exceptions,
                            analyzer_statuses,
                            counts,
                            limitations,
                            analysis_completeness.get("scope_exclusions", []),
                            data.get("components"),
                            scanned_root,
                            references=analysis_completeness.get("references"),
                            skillspector_version=skillspector_version,
                            reference_root=reference_root,
                            analysis_completeness=analysis_completeness,
                        )
                        if caveat_notes is None:
                            # As for a partial report's gaps, the findings are
                            # still checked and kept.
                            result.add_error(_skillspector_reference_caveat_error(ledger_exceptions))
                            result.mark_scan_incomplete(stage_name)
                        else:
                            caveat_warnings = caveat_notes
                elif not complete_by_applicability:
                    if not (
                        counts["partially_inspected_files"]
                        or counts["entirely_uninspected_files"]
                        or ledger_exceptions
                        or limitations
                    ):
                        result.add_error(
                            "skillspector JSON field 'analysis_completeness' details contradict partial analysis; "
                            "security scan did not complete"
                        )
                        return False
                    partial_reported = True
                    # Hooks SkillSpector parsed but could not model get a
                    # Critical in _report_unverified_hooks; only the rest of
                    # the report decides whether the scan is complete.
                    without_hooks = _skillspector_without_hook_surface(data)
                    gap_rows, gap_statuses, gap_counts, gap_limitations = (
                        (ledger_exceptions, analyzer_statuses, counts, limitations)
                        if without_hooks is None
                        else without_hooks[1:]
                    )
                    gap_notes = _skillspector_missing_reference_notes(
                        gap_rows,
                        gap_statuses,
                        gap_counts,
                        gap_limitations,
                        analysis_completeness.get("scope_exclusions", []),
                        data.get("components"),
                        scanned_root,
                        references=analysis_completeness.get("references"),
                        skillspector_version=skillspector_version,
                        reference_root=reference_root,
                        analysis_completeness=analysis_completeness,
                    )
                    if accepted_reference_caveats is not None and reference_root is not None and gap_notes is None:
                        # reference_caveat_metadata reads only the report. When
                        # the scan's reference root is known, its missing
                        # references must also meet the conditions above, so a
                        # line that names a file left out of the scan, or a
                        # skill scanned inside a larger artifact, stays
                        # incomplete.
                        accepted_reference_caveats = None
                    if gap_notes is None:
                        raster_note = (
                            None
                            if _skillspector_reference_evidence_unmatched(analysis_completeness)
                            else _skillspector_passive_raster_note(
                                gap_rows,
                                gap_statuses,
                                gap_counts,
                                gap_limitations,
                                analysis_completeness.get("scope_exclusions", []),
                                data.get("components"),
                                scanned_root,
                            )
                        )
                        gap_notes = None if raster_note is None else [raster_note]
                    other_gaps: list[str] = []
                    if gap_notes is None and without_hooks is not None and not gap_rows:
                        other_gaps = _skillspector_gaps_without_rows(
                            gap_statuses,
                            gap_counts,
                            gap_limitations,
                            analysis_completeness.get("scope_exclusions", []),
                        )
                        if not other_gaps:
                            gap_notes = []
                    if accepted_reference_caveats is not None:
                        # Add the warning only after all inventory, finding and
                        # risk-accounting checks below have accepted the report.
                        pass
                    elif gap_notes is not None:
                        for note in gap_notes:
                            result.add_warning(note)
                    else:
                        # The hook surface has its own confirmed Critical; only
                        # residual gaps belong in incomplete-scan diagnostics.
                        diagnostic = completeness_record(
                            {"ledger_exceptions": gap_rows, "limitations": gap_limitations}, stage_name
                        )
                        result.metadata["skillspector_completeness"] = merge_completeness(
                            result.metadata.get("skillspector_completeness"), {"records": [diagnostic]}
                        )
                        message = (
                            _skillspector_gaps_without_rows_error(completeness_status, other_gaps)
                            if other_gaps
                            else _skillspector_incomplete_analysis_error(completeness_status, gap_rows)
                        )
                        if not gap_rows and not other_gaps and diagnostic["limitations"]:
                            # No incomplete file to name: the limitation is the only stated reason.
                            message = f"{message}. Limitation: {diagnostic['limitations'][0]}"
                        result.fail_gate(message)
                        result.mark_scan_incomplete(stage_name)

        status = data.get("status")
        if status is not None and not isinstance(status, str):
            result.add_error("skillspector JSON field 'status' must be a string; security scan did not complete")
            return False
        if isinstance(status, str):
            normalized_status = status.strip().lower()
            if normalized_status in {
                "cancelled",
                "canceled",
                "error",
                "failed",
                "failure",
                "fatal",
                "incomplete",
                "partial",
                "timed_out",
                "timeout",
            }:
                result.add_error(f"skillspector reported failure status '{status}'; security scan did not complete")
                return False
            if normalized_status not in {"complete", "completed", "ok", "success", "successful"}:
                result.add_error("skillspector JSON field 'status' is not recognized; security scan did not complete")
                return False

        if "issues" not in data:
            result.add_error(
                "skillspector JSON report is missing required 'issues' list; security scan did not complete"
            )
            return False
        issues = data["issues"]
        if not isinstance(issues, list):
            result.add_error("skillspector JSON field 'issues' must be a list; security scan did not complete")
            return False
        if not all(isinstance(issue, dict) for issue in issues):
            result.add_error("skillspector JSON 'issues' entries must be objects; security scan did not complete")
            return False

        risk = data.get("risk_assessment")
        if not isinstance(risk, dict):
            result.add_error(
                "skillspector JSON report is missing required 'risk_assessment' object; security scan did not complete"
            )
            return False
        score = risk.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or (isinstance(score, float) and not math.isfinite(score))
            or not 0 <= score <= 100
        ):
            result.add_error(
                "skillspector JSON field 'risk_assessment.score' must be a finite number from 0 to 100; "
                "security scan did not complete"
            )
            return False
        severity = risk.get("severity")
        expected_severity = next(band for threshold, band in _SKILLSPECTOR_RISK_BANDS if score >= threshold)
        if not isinstance(severity, str) or severity != expected_severity:
            result.add_error(
                "skillspector JSON field 'risk_assessment.severity' does not match the risk score; "
                "security scan did not complete"
            )
            return False
        recommendation = risk.get("recommendation")
        expected_recommendation = _SKILLSPECTOR_RECOMMENDATION_BY_SEVERITY[severity]
        if (incomplete_before or partial_reported) and severity == "LOW":
            expected_recommendation = "CAUTION"
        if not isinstance(recommendation, str) or recommendation != expected_recommendation:
            result.add_error(
                "skillspector JSON field 'risk_assessment.recommendation' does not match the risk severity; "
                "security scan did not complete"
            )
            return False

        for index, issue in enumerate(issues):
            if not SecurityValidator._validate_skillspector_issue(
                issue,
                index,
                result,
                require_finding_id=uses_completeness_schema,
                require_location_file=uses_versioned_completeness,
            ):
                return False
        if uses_completeness_schema:
            scoring_fields_by_identity: dict[tuple, tuple] = {}
            for index, issue in enumerate(issues):
                match_identity = _skillspector_scoring_match_identity(
                    issue,
                    index,
                    uses_report_identities=True,
                    uses_finding_identity=uses_finding_identity,
                )
                identity = (
                    _skillspector_scoring_source_scope(issue),
                    issue["id"],
                    match_identity,
                )
                scoring_fields = (
                    issue["severity"],
                    issue["confidence"],
                    issue["finding_id"] if match_identity[0] == "match_fingerprint" else None,
                )
                if uses_finding_identity:
                    # Occurrence rows may vary in location, not classification.
                    identity = match_identity
                    scoring_fields += (
                        _skillspector_scoring_source_scope(issue),
                        issue["id"],
                        issue.get("match_fingerprint"),
                        *(issue.get(field) for field in (
                            "finding", "category", "pattern", "explanation",
                            "remediation", "intent", "tags",
                        )),
                        # JSON classification evidence distinguishes true from 1.
                        json.dumps(issue.get("evidence"), sort_keys=True),
                    )
                previous = scoring_fields_by_identity.setdefault(identity, scoring_fields)
                if previous != scoring_fields:
                    result.add_error(
                        "skillspector JSON compacted identity has inconsistent scoring fields; "
                        "security scan did not complete"
                    )
                    return False
        if not issues and score != 0:
            result.add_error(
                "skillspector JSON reports a nonzero risk score without any issues; security scan did not complete"
            )
            return False

        expected_exit_code = 1 if score > 50 else 0
        if process_exit_code != expected_exit_code:
            result.add_error(
                "skillspector process exit code does not match the report risk score; security scan did not complete"
            )
            return False

        for field in ("skill", "metadata"):
            value = data.get(field)
            if value is not None and not isinstance(value, dict):
                result.add_error(f"skillspector JSON field '{field}' must be an object; security scan did not complete")
                return False
        skill = data.get("skill") or {}
        for field in ("name", "source", "scanned_at"):
            value = skill.get(field)
            if value is not None and not isinstance(value, str):
                result.add_error(
                    f"skillspector JSON field 'skill.{field}' must be a string or null; security scan did not complete"
                )
                return False
        metadata = data.get("metadata") or {}
        for field in ("skillspector_version", "filtering_mode"):
            value = metadata.get(field)
            if value is not None and not isinstance(value, str):
                result.add_error(
                    f"skillspector JSON field 'metadata.{field}' must be a string or null; "
                    "security scan did not complete"
                )
                return False
        for field in ("has_executable_scripts", "llm_requested", "llm_available", "meta_analysis_applied"):
            value = metadata.get(field)
            if value is not None and not isinstance(value, bool):
                result.add_error(
                    f"skillspector JSON field 'metadata.{field}' must be a boolean or null; "
                    "security scan did not complete"
                )
                return False
        if version_error is not None:
            result.add_error(version_error)
            return False
        if uses_versioned_completeness and metadata.get("llm_requested") is not use_llm:
            stage_description = "LLM" if use_llm else "--no-llm"
            result.add_error(
                "skillspector JSON field 'metadata.llm_requested' contradicts the "
                f"{stage_description} stage; "
                "security scan did not complete"
            )
            return False
        # Provider availability is independent of execution: configured
        # credentials do not contradict a --no-llm report.
        if not use_llm and (
            metadata.get("llm_requested") not in {None, False}
            or metadata.get("meta_analysis_applied") not in {None, False}
        ):
            result.add_error(
                "skillspector deterministic report contradicts the --no-llm stage; security scan did not complete"
            )
            return False
        components = data.get("components")
        if uses_versioned_completeness and "components" not in data:
            result.add_error(
                f"skillspector {completeness_contract} JSON report is missing required "
                "'components' list; security scan did not complete"
            )
            return False
        if components is not None and not isinstance(components, list):
            result.add_error("skillspector JSON field 'components' must be a list; security scan did not complete")
            return False
        if isinstance(components, list) and not all(isinstance(component, dict) for component in components):
            result.add_error("skillspector JSON 'components' entries must be objects; security scan did not complete")
            return False
        normalized_components = components or []
        for index, component in enumerate(normalized_components):
            for field in ("path", "source_identity", "source_url", "source_digest"):
                value = component.get(field)
                if uses_versioned_completeness and field == "path" and (
                    not isinstance(value, str) or not value
                ):
                    result.add_error(
                        f"skillspector JSON field 'components[{index}].path' must be a non-empty string; "
                        "security scan did not complete"
                    )
                    return False
                if value is not None and not isinstance(value, str):
                    result.add_error(
                        f"skillspector JSON field 'components[{index}].{field}' must be a string or null; "
                        "security scan did not complete"
                    )
                    return False
            executable = component.get("executable")
            if uses_versioned_completeness and not isinstance(executable, bool):
                result.add_error(
                    f"skillspector JSON field 'components[{index}].executable' must be a boolean; "
                    "security scan did not complete"
                )
                return False
            if executable is not None and not isinstance(executable, bool):
                result.add_error(
                    f"skillspector JSON field 'components[{index}].executable' must be a boolean or null; "
                    "security scan did not complete"
                )
                return False
        component_has_executable = any(component.get("executable") is True for component in normalized_components)
        declared_has_executable = metadata.get("has_executable_scripts")
        if (
            uses_versioned_completeness
            and isinstance(declared_has_executable, bool)
            and declared_has_executable is not component_has_executable
        ) or (component_has_executable and declared_has_executable is False):
            result.add_error(
                "skillspector JSON components contradict metadata.has_executable_scripts; "
                "security scan did not complete"
            )
            return False
        if uses_versioned_completeness and not result.is_incomplete:
            if len(normalized_components) != analysis_completeness["total_components"]:
                result.add_error(
                    "skillspector JSON component inventory contradicts analysis completeness; "
                    "security scan did not complete"
                )
                return False
            component_keys = {
                (_skillspector_scoring_source_scope(component), component["path"])
                for component in normalized_components
            }
            if len(component_keys) != len(normalized_components):
                result.add_error(
                    "skillspector JSON component inventory contains duplicate identities; "
                    "security scan did not complete"
                )
                return False
            if not universal_analyzer_evidence_valid:
                result.add_error(
                    "skillspector JSON universal analyzer evidence contradicts the "
                    "component inventory; security scan did not complete"
                )
                return False
            for issue in issues:
                location = issue.get("location") or {}
                issue_key = (
                    _skillspector_scoring_source_scope(issue),
                    str(location.get("file") or "SKILL.md"),
                )
                if issue.get("id") != "SC8" and issue_key not in component_keys:
                    result.add_error(
                        "skillspector JSON issue has no matching component inventory entry; "
                        "security scan did not complete"
                    )
                    return False
        suppressed_count = data.get("suppressed_count")
        if suppressed_count is not None and (
            isinstance(suppressed_count, bool) or not isinstance(suppressed_count, int) or suppressed_count < 0
        ):
            result.add_error(
                "skillspector JSON field 'suppressed_count' must be a non-negative integer; "
                "security scan did not complete"
            )
            return False
        suppressed = data.get("suppressed")
        if suppressed is not None and not isinstance(suppressed, list):
            result.add_error("skillspector JSON field 'suppressed' must be a list; security scan did not complete")
            return False
        if isinstance(suppressed, list) and not all(isinstance(item, dict) for item in suppressed):
            result.add_error("skillspector JSON 'suppressed' entries must be objects; security scan did not complete")
            return False
        if baseline_requested and (suppressed_count is None or suppressed is None):
            # Without both fields, findings could disappear without a record.
            result.add_error(
                "skillspector JSON report does not account for findings suppressed by the requested baseline; "
                "security scan did not complete"
            )
            return False
        normalized_suppressed_count = suppressed_count or 0
        normalized_suppressed = suppressed or []
        if normalized_suppressed_count != len(normalized_suppressed):
            result.add_error(
                "skillspector JSON suppressed_count does not match the suppressed findings list; "
                "security scan did not complete"
            )
            return False
        if findings_after_filtering is not None:
            serialized_findings = len(issues) + normalized_suppressed_count
            serialized_identity_count = len(
                SecurityValidator._deduplicate_skillspector_issues_for_scoring(
                    issues,
                    uses_report_identities=uses_completeness_schema,
                    uses_finding_identity=uses_finding_identity,
                )[0]
            )
            finding_counts_match = (
                findings_after_filtering == serialized_findings
                if uses_statusless_completeness_schema
                else (
                    (findings_after_filtering == 0) is (serialized_findings == 0)
                    and normalized_suppressed_count <= findings_after_filtering
                    and serialized_identity_count <= findings_after_filtering
                )
            )
            if not finding_counts_match:
                result.add_error(
                    "skillspector JSON analysis completeness finding counts contradict the serialized findings; "
                    "security scan did not complete"
                )
                return False
        if normalized_suppressed_count and not baseline_requested:
            result.add_error(
                "skillspector reported unexpected suppressed findings without a requested baseline; "
                "security scan did not complete"
            )
            return False
        minimum_score = SecurityValidator._minimum_skillspector_risk_score(
            issues,
            normalized_components,
            uses_report_identities=uses_completeness_schema,
            uses_finding_identity=uses_finding_identity,
            # Baseline-suppressed findings do not score; counting them as
            # unseen occurrences would loosen this reconciliation.
            findings_after_filtering=(
                findings_after_filtering - normalized_suppressed_count if findings_after_filtering is not None else None
            ),
            all_findings_serialized=(
                findings_after_filtering is None
                or findings_after_filtering == len(issues) + normalized_suppressed_count
            ),
        )
        if score < minimum_score:
            result.add_error(
                "skillspector JSON risk score understates the reported issues; security scan did not complete"
            )
            return False

        for note in caveat_warnings:
            result.add_warning(note)
        if accepted_reference_caveats is not None:
            result.metadata["skillspector_reference_caveats"] = accepted_reference_caveats
            locations = ", ".join(
                f"{_skillspector_report_text(row['path'])}:{row['start_line']}"
                for row in ledger_exceptions[:_SKILLSPECTOR_INCOMPLETE_EXAMPLES]
            )
            remaining = len(ledger_exceptions) - _SKILLSPECTOR_INCOMPLETE_EXAMPLES
            if remaining > 0:
                locations = f"{locations} (+{remaining} more)"
            result.add_finding(
                Finding(
                    category="SCAN",
                    severity=Severity.INFO,
                    check_name="skillspector_reference_caveats",
                    message=(
                        f"SkillSpector inspected all {accepted_reference_caveats['fully_inspected_files']} bundled file(s). "
                        f"{accepted_reference_caveats['count']} missing-reference caveat(s) remain; "
                        f"review referenced paths at {locations}. "
                        "Referenced targets were not verified; the scanner's original partial status is retained."
                    ),
                    file_path=ledger_exceptions[0]["path"],
                    line_number=ledger_exceptions[0]["start_line"],
                    suggestion="Review the referenced paths; only files bundled in this skill were inspected.",
                    metadata={
                        "scanner_status": "partial",
                        "scanner_recommendation": recommendation,
                        "missing_targets_verified": False,
                    },
                )
            )
        return True

    @staticmethod
    def _minimum_skillspector_risk_score(
        issues: list[dict],
        components: list[dict],
        *,
        uses_report_identities: bool,
        uses_finding_identity: bool,
        findings_after_filtering: int | None = None,
        reported_score: int | float | None = None,
        removed_issues: list[dict] | None = None,
        all_findings_serialized: bool = False,
    ) -> int | float:
        """Return a conservative score floor from the public report fields."""
        file_executable = {
            (_skillspector_scoring_source_scope(component), component["path"]):
            component.get("executable") is True
            for component in components
            if isinstance(component.get("path"), str)
        }

        def base_contribution(issue: dict, *, trust_location: bool = True) -> float:
            location = issue.get("location") or {}
            multiplier = (
                _SKILLSPECTOR_EXECUTABLE_MULTIPLIER
                if trust_location
                and file_executable.get(
                    (_skillspector_scoring_source_scope(issue), location.get("file")),
                    False,
                )
                else 1.0
            )
            return _SKILLSPECTOR_SEVERITY_POINTS[issue["severity"]] * issue["confidence"] * multiplier

        def legacy_score(candidate_issues: list[dict]) -> int:
            by_rule: dict[str, list[dict]] = {}
            for issue in candidate_issues:
                by_rule.setdefault(issue["id"], []).append(issue)

            total = 0.0
            for rule_issues in by_rule.values():
                ordered = sorted(
                    (issue for issue in rule_issues if issue["confidence"] > 0),
                    key=lambda issue: (
                        -_SKILLSPECTOR_SEVERITY_POINTS[issue["severity"]],
                        base_contribution(issue),
                    ),
                )
                for index, issue in enumerate(ordered[: len(_SKILLSPECTOR_DIMINISHING_WEIGHTS)]):
                    total += base_contribution(issue) * _SKILLSPECTOR_DIMINISHING_WEIGHTS[index]
            score_floor = max(
                (
                    _SKILLSPECTOR_RISK_SCORE_FLOORS_BY_RULE_ID.get(issue["id"], 0)
                    for issue in candidate_issues
                    if issue["confidence"] > 0
                ),
                default=0,
            )
            return min(100, max(score_floor, int(total)))

        def compacted_score(
            candidate_issues: list[dict],
            ambiguous_identities: set[tuple],
            unknown_finding_count: int,
        ) -> int:
            by_rule: dict[str, list[float]] = {}
            for index, issue in enumerate(candidate_issues):
                if issue["confidence"] <= 0:
                    continue
                identity = (
                    _skillspector_scoring_source_scope(issue),
                    issue["id"],
                    _skillspector_scoring_match_identity(
                        issue,
                        index,
                        uses_report_identities=True,
                        uses_finding_identity=uses_finding_identity,
                    ),
                )
                by_rule.setdefault(issue["id"], []).append(
                    base_contribution(
                        issue,
                        trust_location=(
                            unknown_finding_count == 0 and identity not in ambiguous_identities
                        ),
                    )
                )

            total = 0.0
            reductions: list[float] = []
            for contributions in by_rule.values():
                ordered = sorted(contributions)
                costs = [
                    sum(
                        contribution * weight
                        for contribution, weight in zip(
                            ordered,
                            _SKILLSPECTOR_DIMINISHING_WEIGHTS[unknowns:],
                            strict=False,
                        )
                    )
                    for unknowns in range(len(_SKILLSPECTOR_DIMINISHING_WEIGHTS) + 1)
                ]
                total += costs[0]
                reductions.extend(
                    costs[index] - costs[index + 1]
                    for index in range(len(_SKILLSPECTOR_DIMINISHING_WEIGHTS))
                )
            total -= sum(sorted(reductions, reverse=True)[:unknown_finding_count])
            score_floor = max(
                (
                    _SKILLSPECTOR_RISK_SCORE_FLOORS_BY_RULE_ID.get(issue["id"], 0)
                    for issue in candidate_issues
                    if issue["confidence"] > 0
                ),
                default=0,
            )
            return min(100, max(score_floor, int(max(0, total))))

        # SkillSpector scores before report compaction, which can discard
        # occurrence-level confidence, executable status, and same-severity order.
        deduplicated, _ambiguous_identities = SecurityValidator._deduplicate_skillspector_issues_for_scoring(
            issues,
            uses_report_identities=uses_report_identities,
            uses_finding_identity=uses_finding_identity,
        )
        if uses_report_identities:
            all_visible_issues = [*issues, *(removed_issues or [])]
            all_deduplicated, all_ambiguous_identities = (
                SecurityValidator._deduplicate_skillspector_issues_for_scoring(
                    all_visible_issues,
                    uses_report_identities=True,
                    uses_finding_identity=uses_finding_identity,
                )
            )
            unknown_finding_count = max(
                0,
                (findings_after_filtering or 0) - len(all_deduplicated),
            )
            minimum_score = compacted_score(
                deduplicated,
                all_ambiguous_identities,
                unknown_finding_count,
            )
        else:
            minimum_score = min(legacy_score(issues), legacy_score(deduplicated))
        if reported_score is None or not removed_issues or not all_findings_serialized:
            return minimum_score
        maximum_removed_loss = sum(
            max(
                _SKILLSPECTOR_SEVERITY_POINTS[issue["severity"]]
                * issue["confidence"]
                * _SKILLSPECTOR_EXECUTABLE_MULTIPLIER,
                _SKILLSPECTOR_RISK_SCORE_FLOORS_BY_RULE_ID.get(issue["id"], 0)
                if issue["confidence"] > 0
                else 0,
            )
            for issue in removed_issues
        )
        reported_floor = max(0, reported_score - math.ceil(maximum_removed_loss))
        return max(minimum_score, reported_floor)

    @staticmethod
    def _retained_skillspector_risk_score(issues: list[dict], components: list[dict]) -> int:
        """Score *issues* with SkillSpector's formula, one finding per serialized record.

        Use only when the report serializes every finding it counted. Each
        occurrence is a finding, scored with its own file's executable
        multiplier. Within a rule, SkillSpector gives the largest weights to
        the most severe findings; within one severity, the lowest
        contributions take the larger weights here, so the result never
        exceeds SkillSpector's score for the same findings.
        """
        file_executable = {
            (_skillspector_scoring_source_scope(component), component["path"]): component.get("executable") is True
            for component in components
            if isinstance(component, dict) and isinstance(component.get("path"), str)
        }
        by_rule: dict[str, list[tuple[float, float]]] = {}
        score_floor = 0
        for issue in issues:
            if issue["confidence"] <= 0:
                continue
            score_floor = max(score_floor, _SKILLSPECTOR_RISK_SCORE_FLOORS_BY_RULE_ID.get(issue["id"], 0))
            occurrences = issue.get("occurrences")
            records = occurrences if isinstance(occurrences, list) and occurrences else [issue.get("location")]
            source_scope = _skillspector_scoring_source_scope(issue)
            for record in records:
                record_file = record.get("file") if isinstance(record, dict) else None
                multiplier = (
                    _SKILLSPECTOR_EXECUTABLE_MULTIPLIER
                    if file_executable.get((source_scope, record_file), False)
                    else 1.0
                )
                severity_points = _SKILLSPECTOR_SEVERITY_POINTS[issue["severity"]]
                by_rule.setdefault(issue["id"], []).append(
                    (-severity_points, severity_points * issue["confidence"] * multiplier)
                )
        total = sum(
            contribution * weight
            for contributions in by_rule.values()
            for (_severity, contribution), weight in zip(
                sorted(contributions), _SKILLSPECTOR_DIMINISHING_WEIGHTS, strict=False
            )
        )
        return min(100, max(score_floor, int(total)))

    @staticmethod
    def _deduplicate_skillspector_issues_for_scoring(
        issues: list[dict],
        *,
        uses_report_identities: bool,
        uses_finding_identity: bool,
    ) -> tuple[list[dict], set[tuple]]:
        """Mirror scanner dedup using serialized source and match identities."""

        same_file_best: dict[tuple, tuple[int, dict]] = {}
        modern_identity_counts: Counter[tuple] = Counter()
        for index, issue in enumerate(issues):
            location = issue.get("location") or {}
            identity = _skillspector_scoring_match_identity(
                issue,
                index,
                uses_report_identities=uses_report_identities,
                uses_finding_identity=uses_finding_identity,
            )
            if uses_report_identities:
                modern_identity_counts[
                    (_skillspector_scoring_source_scope(issue), issue["id"], identity)
                ] += 1
            key = (
                _skillspector_scoring_source_scope(issue),
                issue["id"],
                str(location.get("file") or "SKILL.md"),
                identity,
            )
            existing = same_file_best.get(key)
            if existing is None or issue["confidence"] > existing[1]["confidence"]:
                same_file_best[key] = index, issue

        cross_file_best: dict[tuple, dict] = {}
        for index, issue in same_file_best.values():
            key = (
                _skillspector_scoring_source_scope(issue),
                issue["id"],
                _skillspector_scoring_match_identity(
                    issue,
                    index,
                    uses_report_identities=uses_report_identities,
                    uses_finding_identity=uses_finding_identity,
                ),
            )
            existing = cross_file_best.get(key)
            if existing is None or issue["confidence"] > existing["confidence"]:
                cross_file_best[key] = issue
        ambiguous_identities = {
            identity
            for identity, count in modern_identity_counts.items()
            if count > 1
        }
        return list(cross_file_best.values()), ambiguous_identities

    @staticmethod
    def _validate_skillspector_issue(
        issue: dict,
        index: int,
        result: ValidationResult,
        *,
        require_finding_id: bool,
        require_location_file: bool,
    ) -> bool:
        """Validate every nested issue field consumed by the report converter."""
        prefix = f"skillspector JSON field 'issues[{index}]"
        issue_id = issue.get("id")
        if not isinstance(issue_id, str) or not issue_id.strip():
            result.add_error(f"{prefix}.id' must be a non-empty string; security scan did not complete")
            return False

        severity = issue.get("severity")
        if not isinstance(severity, str) or severity not in {
            "CRITICAL",
            "HIGH",
            "MEDIUM",
            "LOW",
            "INFO",
        }:
            result.add_error(f"{prefix}.severity' must be a recognized severity; security scan did not complete")
            return False

        optional_strings = (
            "finding_id",
            "category",
            "pattern",
            "finding",
            "explanation",
            "remediation",
            "code_snippet",
            "intent",
            "match_fingerprint",
            "source_identity",
            "source_url",
            "source_digest",
        )
        for field in optional_strings:
            value = issue.get(field)
            if value is not None and not isinstance(value, str):
                result.add_error(f"{prefix}.{field}' must be a string or null; security scan did not complete")
                return False
        finding_id = issue.get("finding_id")
        if require_finding_id and (not isinstance(finding_id, str) or not finding_id.strip()):
            result.add_error(
                f"{prefix}.finding_id' must be a non-empty string; security scan did not complete"
            )
            return False
        if not any(
            isinstance(issue.get(field), str) and issue[field].strip()
            for field in ("pattern", "finding", "explanation")
        ):
            result.add_error(
                f"{prefix}' must include a non-empty pattern, finding, or explanation; security scan did not complete"
            )
            return False

        confidence = issue.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or (isinstance(confidence, float) and not math.isfinite(confidence))
            or not 0 <= confidence <= 1
        ):
            result.add_error(
                f"{prefix}.confidence' must be a finite number from 0 to 1; security scan did not complete"
            )
            return False

        location = issue.get("location")
        if location is None:
            if require_location_file:
                result.add_error(
                    f"{prefix}.location.file' must be a non-empty string; security scan did not complete"
                )
                return False
            return True
        if not isinstance(location, dict):
            result.add_error(f"{prefix}.location' must be an object or null; security scan did not complete")
            return False
        file_path = location.get("file")
        if require_location_file and (not isinstance(file_path, str) or not file_path):
            result.add_error(
                f"{prefix}.location.file' must be a non-empty string; security scan did not complete"
            )
            return False
        if file_path is not None and not isinstance(file_path, str):
            result.add_error(f"{prefix}.location.file' must be a string or null; security scan did not complete")
            return False
        if isinstance(file_path, str) and file_path and not _is_filesystem_safe_path_text(file_path):
            result.add_error(
                f"{prefix}.location.file' must be a filesystem-safe string; security scan did not complete"
            )
            return False
        for field in ("start_line", "line", "end_line"):
            line_number = location.get(field)
            if line_number is not None and (
                isinstance(line_number, bool) or not isinstance(line_number, int) or line_number < 1
            ):
                result.add_error(
                    f"{prefix}.location.{field}' must be a positive integer or null; security scan did not complete"
                )
                return False
        return True

    def _process_skillspector_cli_result(
        self,
        data: dict,
        result: ValidationResult,
        *,
        source_root: Path | None = None,
    ) -> None:
        """Convert a validated SkillSpector JSON report into ValidationResult entries.

        With a ``content_type``, issues outside agent control paths are
        suppressed and excluded from the aggregate risk score (see
        :func:`~skillevaluator.validators.skillspector_scope.scope_skillspector_issues`).
        Each manifest whose hooks SkillSpector could not model gets a Critical
        gate failure (:meth:`_report_unverified_hooks`), whatever the scoping.
        """
        self._store_skillspector_metadata(data, result, score_message=False)

        issues = data.get("issues", [])
        report_metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        raw_version = report_metadata.get("skillspector_version")
        report_components = data.get("components") if isinstance(data.get("components"), list) else []
        candidate_issues = []
        removed_issues = []
        skipped_generated = 0
        skipped_spdx_comments = 0
        skipped_passwd_mounts = 0
        passwd_mount_lines: dict[str, frozenset[int]] = {}
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            if self._is_generated_artifact_issue(issue, source_root=source_root):
                removed_issues.append(issue)
                skipped_generated += 1
                continue
            if self._is_spdx_only_hidden_instruction(issue):
                removed_issues.append(issue)
                skipped_spdx_comments += 1
                continue
            if self._is_read_only_passwd_mount_issue(
                issue,
                scanner_version=raw_version,
                source_root=source_root,
                safe_lines_by_file=passwd_mount_lines,
            ):
                removed_issues.append(issue)
                skipped_passwd_mounts += 1
                continue
            candidate_issues.append(issue)

        scopes = self._skillspector_issue_scopes(
            candidate_issues,
            report_components,
            source_root=source_root,
            report_truncated=skillspector_report_is_truncated(data),
        )
        scanned_issues = []
        skipped_inapplicable = 0
        skipped_by_role: Counter[str] = Counter()
        has_critical_or_high = False
        completeness = data.get("analysis_completeness")
        # One normalizer per report: each distinct path and text is redacted once.
        normalizer = ReportNormalizer(source_root)
        legacy_reason_index = index_legacy_reference_reasons(
            candidate_issues,
            normalize_path=normalizer.path,
            completeness=completeness,
            text=normalizer.text,
        )
        for issue, scope in zip(candidate_issues, scopes, strict=True):
            if scope is not None and not scope.applicable:
                removed_issues.append(issue)
                skipped_inapplicable += 1
                skipped_by_role[scope.file_role] += 1
                continue
            scanned_issues.append(issue)
            finding, is_error = self._convert_skillspector_issue(
                issue,
                completeness=completeness,
                legacy_reason_index=legacy_reason_index,
                normalizer=normalizer,
            )
            if scope is not None:
                finding.metadata.update(
                    {
                        "artifact_type": self.content_type,
                        "file_role": scope.file_role,
                        "applicability": scope.applicability,
                    }
                )
            if is_error:
                has_critical_or_high = True
            result.add_structured_finding(finding, is_error=is_error)
        _record_skillspector_reference_summary(result)

        if skipped_generated:
            result.add_message(f"skillspector ignored {skipped_generated} generated artifact issue(s)")
        if skipped_spdx_comments:
            result.add_message(f"skillspector ignored {skipped_spdx_comments} SPDX-only HTML comment issue(s)")
        if skipped_passwd_mounts:
            result.add_message(f"skillspector ignored {skipped_passwd_mounts} read-only /etc/passwd mount issue(s)")
        if skipped_inapplicable:
            roles = ", ".join(f"{role}={count}" for role, count in sorted(skipped_by_role.items()))
            result.add_message(
                f"skillspector suppressed {skipped_inapplicable} issue(s) outside agent control paths ({roles})"
            )

        reported_score = data["risk_assessment"]["score"]
        uses_report_identities = (
            isinstance(raw_version, str)
            and SEMVER_RE.fullmatch(raw_version) is not None
            and tuple(int(part) for part in raw_version.split("."))
            >= _SKILLSPECTOR_COMPLETENESS_SCHEMA_VERSION
        )
        uses_finding_identity = (
            uses_report_identities
            and tuple(int(part) for part in raw_version.split("."))
            >= _SKILLSPECTOR_FINDING_IDENTITY_VERSION
        )
        analysis_completeness = data.get("analysis_completeness")
        findings_after_filtering = (
            analysis_completeness.get("findings_after_filtering")
            if isinstance(analysis_completeness, dict)
            else None
        )
        suppressed_count = len(data["suppressed"]) if isinstance(data.get("suppressed"), list) else 0
        serialized_findings = len(issues) + suppressed_count
        effective_score = (
            self._minimum_skillspector_risk_score(
                scanned_issues,
                report_components,
                uses_report_identities=uses_report_identities,
                uses_finding_identity=uses_finding_identity,
                findings_after_filtering=(
                    findings_after_filtering - suppressed_count if isinstance(findings_after_filtering, int) else None
                ),
                reported_score=reported_score,
                removed_issues=removed_issues,
                all_findings_serialized=findings_after_filtering == serialized_findings,
            )
            if skipped_generated or skipped_spdx_comments or skipped_passwd_mounts or skipped_inapplicable
            else reported_score
        )
        if skipped_inapplicable and uses_report_identities and skillspector_report_serializes_every_finding(data):
            # The reconciliation floor treats a match repeated across files
            # as one finding plus unknown findings, which lowers the score of
            # the kept issues. With every counted finding serialized, score
            # the kept records directly.
            effective_score = max(
                effective_score,
                min(reported_score, self._retained_skillspector_risk_score(scanned_issues, report_components)),
            )
        if reported_score > 50:
            reported_severity = data["risk_assessment"].get("severity", "UNKNOWN")
            if skipped_inapplicable:
                # The gate uses the applicable score, so label both scores.
                result.add_message(
                    f"skillspector risk score: {effective_score}/100 applicable "
                    f"(raw {reported_score}/100 {reported_severity})"
                )
            else:
                result.add_message(f"skillspector risk score: {reported_score}/100 ({reported_severity})")
        if self.content_type is not None:
            result.metadata["skillspector_applicability"] = {
                "artifact_type": self.content_type,
                "kept_issues": len(scanned_issues),
                "suppressed_inapplicable_issues": skipped_inapplicable,
                "suppressed_by_file_role": dict(sorted(skipped_by_role.items())),
                "raw_risk_score": reported_score,
                "applicable_risk_score": effective_score,
            }
        if effective_score > 50 and not has_critical_or_high:
            result.add_structured_finding(
                Finding(
                    category="SECURITY",
                    severity=Severity.HIGH,
                    check_name="skillspector_risk_score",
                    message=f"SkillSpector aggregate risk score {effective_score}/100 exceeds the policy threshold",
                    file_path=str((data.get("skill") or {}).get("source") or "SKILL.md"),
                    suggestion="Review and resolve the contributing security findings",
                ),
                is_error=True,
            )
            has_critical_or_high = True

        without_hooks = _skillspector_without_hook_surface(data)
        unverified_hooks = self._report_unverified_hooks(
            without_hooks[0] if without_hooks is not None else [],
            result,
            report_components,
            source_root,
        )

        if not result.is_incomplete:
            # Each manifest's Critical counts as an issue, so the summary never
            # reports a clean scan beside it.
            self._summarize_skillspector_results(
                [*scanned_issues, *unverified_hooks],
                has_critical_or_high or bool(unverified_hooks),
                result,
                suppressed_inapplicable_issues=skipped_inapplicable,
            )

    @staticmethod
    def _report_unverified_hooks(
        hook_rows: list[dict],
        result: ValidationResult,
        components: object,
        source_root: Path | None,
    ) -> list[str]:
        """Fail the gate for each manifest whose hooks SkillSpector could not model, and return the manifests.

        *hook_rows* come from :func:`_skillspector_without_hook_surface`. The
        scan completed, but the absence of findings does not show that such
        hooks are safe: a hook script that sends its input over a socket gets
        no blocking SkillSpector finding. Each manifest gets a Critical and a
        gate failure, so neither a ``--policy`` severity override nor
        ``--llm-verify`` can clear it; SkillEvaluator has no way to record a
        review of the hooks, so only removing them passes the check. The
        Critical does not stop a folder scan at this skill
        (:attr:`_FAIL_FAST_EXEMPT_CHECKS`). The finding records the manifest's
        SHA-256 when it is a bundled file
        (:func:`_skillspector_bundled_file_bytes`), else ``None``.
        """
        manifests: list[str] = []
        for row in hook_rows:
            manifest = row["path"]
            manifest_bytes = _skillspector_bundled_file_bytes(manifest, components, source_root)
            result.add_structured_finding(
                Finding(
                    category="SECURITY",
                    severity=Severity.CRITICAL,
                    check_name=_SKILLSPECTOR_HOOK_UNVERIFIED_CHECK,
                    message=(
                        f"SkillSpector could not fully model the hooks in {manifest} (analyzer "
                        f"{_SKILLSPECTOR_HOOK_ANALYZER} could not verify what they do with hook input), so this "
                        "check cannot pass while these hooks are present."
                    ),
                    file_path=manifest,
                    suggestion=(
                        f"Review {manifest} and every file its hooks run. Only removing the hooks lets this check "
                        "pass: neither a --policy severity override nor --llm-verify clears it."
                    ),
                    metadata={
                        "ledger_exceptions": [dict(row)],
                        "manifest_sha256": (
                            hashlib.sha256(manifest_bytes).hexdigest() if manifest_bytes is not None else None
                        ),
                    },
                ),
                is_error=True,
            )
            result.fail_gate(
                f"{manifest}: SkillSpector could not model these hooks; this check cannot pass while they are present"
            )
            # The gate is a confirmed failure, so it outranks another gap that
            # records the scan as incomplete, also when a --policy override
            # lowers the Critical. Other gate failures, such as the residual
            # incomplete-analysis error, do not set this.
            result.metadata["confirmed_failure"] = True
            manifests.append(manifest)
        return manifests

    def _skillspector_issue_scopes(
        self,
        issues: list[dict],
        components: list,
        *,
        source_root: Path | None,
        report_truncated: bool,
    ) -> list[IssueScope | None]:
        """Return each issue's applicability, or ``None`` for an unrestricted scan."""
        if self.content_type is None or source_root is None:
            return [None] * len(issues)
        try:
            return scope_skillspector_issues(
                issues,
                components,
                scan_root=source_root,
                reference_root=self.skillspector_reference_root,
                report_truncated=report_truncated,
            )
        except Exception:
            # Scoping only ever removes findings, so an unexpected failure
            # keeps every one of them rather than abort the security check.
            logger.warning("Could not scope SkillSpector findings to agent control paths; keeping all", exc_info=True)
            return [IssueScope("unknown", "fail-closed") for _issue in issues]

    @staticmethod
    def _is_generated_artifact_issue(issue: dict, *, source_root: Path | None = None) -> bool:
        """Return True when a skillspector issue points at generated output."""
        if issue.get("id") == "SC8":
            return False
        file_path, _line_number = SecurityValidator._parse_issue_location(issue)
        if not _is_filesystem_safe_path_text(file_path):
            return False
        path = Path(file_path)

        if source_root is None:
            if path.name in SCAN_EXCLUDED_FILES:
                return True
            return any(part in _SKILLSPECTOR_SCAN_EXCLUDED_DIRS for part in path.parts)

        root = Path(os.path.abspath(source_root))  # noqa: PTH100 - do not follow untrusted path links
        candidate = path if path.is_absolute() else root / path
        candidate = Path(os.path.abspath(candidate))  # noqa: PTH100 - lexical containment is intentional
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            return False

        if matches_filesystem_name(candidate, SCAN_EXCLUDED_FILES):
            return True
        current = root
        for part in relative.parts[:-1]:
            current /= part
            if matches_filesystem_name(current, _SKILLSPECTOR_SCAN_EXCLUDED_DIRS):
                return True
        return False

    @staticmethod
    def _is_spdx_only_hidden_instruction(issue: dict) -> bool:
        """Suppress only the exact public SPDX comment false positive."""
        if issue.get("id") != "P2" or issue.get("pattern") != "Hidden Instructions":
            return False
        return is_spdx_only_html_comment(
            str(issue.get("code_snippet") or ""),
            allow_frontmatter_separator=True,
        )

    @staticmethod
    def _is_read_only_passwd_mount_issue(
        issue: dict,
        *,
        scanner_version: object,
        source_root: Path | None,
        safe_lines_by_file: dict[str, frozenset[int]],
    ) -> bool:
        """Suppress only a benign-tagged PE3 on an exact read-only passwd mount.

        SkillSpector tags ``docker run -v /etc/passwd:/etc/passwd:ro`` (and its
        Podman, ``--volume`` and quoted forms) as likely benign context but
        still reports it as a HIGH Credential Access finding. Trust those tags
        only where each occurrence is classified separately, and re-check the
        reported source line: every ``/etc/passwd`` on it must be the source or
        target of an exact read-only mount. Writable or implicit-mode mounts,
        other paths, same-line reads such as ``cat /etc/passwd``, aggregated
        rows, unsafe or unreadable locations, and older scanners stay reported.
        """
        tags = issue.get("tags")
        if (
            issue.get("id") != "PE3"
            or issue.get("pattern") != "Credential Access"
            or issue.get("finding") != "/etc/passwd"
            or not isinstance(tags, list)
            or not all(isinstance(tag, str) for tag in tags)
            or not _SKILLSPECTOR_BENIGN_CONTEXT_TAGS.issubset(tags)
            or not isinstance(scanner_version, str)
            or SEMVER_RE.fullmatch(scanner_version) is None
            or tuple(int(part) for part in scanner_version.split(".")) < _SKILLSPECTOR_OCCURRENCE_TRIAGE_VERSION
            or source_root is None
        ):
            return False

        location = issue.get("location")
        occurrences = issue.get("occurrences")
        if not isinstance(location, dict) or not isinstance(occurrences, list) or len(occurrences) != 1:
            return False
        occurrence = occurrences[0]
        if not isinstance(occurrence, dict) or any(
            occurrence.get(field) != location.get(field) for field in ("file", "start_line", "end_line")
        ):
            return False
        file_path = location.get("file")
        line_number = location.get("start_line")
        if (
            not isinstance(file_path, str)
            or not _is_filesystem_safe_path_text(file_path)
            or isinstance(line_number, bool)
            or not isinstance(line_number, int)
            or location.get("end_line") not in (None, line_number)
        ):
            return False

        if file_path not in safe_lines_by_file:
            safe_lines_by_file[file_path] = SecurityValidator._read_only_passwd_mount_lines(source_root, file_path)
        return line_number in safe_lines_by_file[file_path]

    @staticmethod
    def _read_only_passwd_mount_lines(source_root: Path, file_path: str) -> frozenset[int]:
        """Return lines whose every ``/etc/passwd`` belongs to an exact read-only mount.

        The file is read once, without following links, from beneath the scan
        root. Unreadable, oversized, non-UTF-8, or out-of-root files yield no
        lines, which keeps their findings reported.
        """
        root = Path(os.path.abspath(source_root))  # noqa: PTH100 - do not follow untrusted path links
        relative = Path(file_path)
        try:
            if relative.is_absolute():
                relative = relative.relative_to(root)
            with SecureRoot(root) as secure_root:
                raw, _metadata = secure_root.read_bytes(relative, _SKILLSPECTOR_PASSWD_MOUNT_SOURCE_LIMIT_BYTES)
            content = raw.decode("utf-8")
        except (OSError, SecurePathError, UnicodeError, ValueError):
            return frozenset()

        mount_path_starts = {
            volume.start(group)
            for volume in _READ_ONLY_PASSWD_VOLUME_PATTERN.finditer(content)
            for group in ("source", "target")
        }
        safe_lines = set()
        offset = 0
        # str.splitlines() uses the same logical line breaks as SkillSpector's
        # reported line numbers.
        for line_number, line in enumerate(content.splitlines(keepends=True), start=1):
            passwd_starts = [offset + match.start() for match in _PASSWD_PATH_PATTERN.finditer(line)]
            if passwd_starts and all(start in mount_path_starts for start in passwd_starts):
                safe_lines.add(line_number)
            offset += len(line)
        return frozenset(safe_lines)

    @staticmethod
    def _store_skillspector_metadata(data: dict, result: ValidationResult, *, score_message: bool = True) -> None:
        """Store the validated SkillSpector report metadata on the result.

        With *score_message* false, the caller reports the risk score once it
        knows the score that applies after scoping.
        """
        skill_info = data.get("skill") or {}
        sp_metadata = data.get("metadata") or {}
        components = data.get("components") or []

        risk = data.get("risk_assessment") or {}
        score = risk.get("score", 0)
        severity = risk.get("severity", "UNKNOWN")
        suppressed_count = data.get("suppressed_count") or 0

        result.metadata.update(
            {
                "skillspector_score": score,
                "skillspector_severity": severity,
                "skillspector_recommendation": risk.get("recommendation", ""),
                "skillspector_version": sp_metadata.get("skillspector_version"),
                "skillspector_skill_name": skill_info.get("name"),
                "skillspector_scanned_at": skill_info.get("scanned_at"),
                "skillspector_components_count": len(components),
                "skillspector_has_executable_scripts": sp_metadata.get("has_executable_scripts", False),
                "skillspector_suppressed_count": suppressed_count,
                "skillspector_suppressed_summary": SecurityValidator._summarize_suppressed(data.get("suppressed")),
            }
        )

        if suppressed_count:
            result.add_message(f"skillspector suppressed {suppressed_count} audited finding(s)")

        if sp_metadata.get("llm_requested") and not sp_metadata.get("llm_available"):
            result.add_warning(
                "LLM analysis was requested but not available; provider or model diagnostics were redacted"
            )

        if score_message and score > 50:
            result.add_message(f"skillspector risk score: {score}/100 ({severity})")

    @staticmethod
    def _summarize_suppressed(suppressed: object) -> dict:
        """Return the severity counts and rule IDs of the findings a baseline suppressed."""
        by_severity: dict[str, int] = {}
        rule_ids: set[str] = set()
        for item in suppressed if isinstance(suppressed, list) else []:
            if not isinstance(item, dict):
                continue
            severity = str(item.get("severity") or "UNKNOWN").upper()
            by_severity[severity] = by_severity.get(severity, 0) + 1
            rule_id = item.get("id")
            if isinstance(rule_id, str) and rule_id:
                rule_ids.add(rule_id[:64])
        return {"by_severity": by_severity, "rule_ids": sorted(rule_ids)[:_SKILLSPECTOR_SUPPRESSED_RULE_ID_LIMIT]}

    @staticmethod
    def _report_skillspector_baseline(
        result: ValidationResult,
        baseline: dict,
        *,
        static_suppressed: object = None,
        llm_suppressed: object = None,
        static_summary: object = None,
    ) -> None:
        """Record whether the requested baseline applied, so an inert one is not silent.

        The diagnostic is added after SkillSpector returns, outside its finding
        stream, so the baseline cannot suppress it. ``static_suppressed`` and
        ``llm_suppressed`` are the counts each stage reported; the LLM count
        is ``None`` when that stage did not run or did not complete.

        Every processed outcome is also a ``skillspector_baseline`` success
        detail, so Markdown and HTML reports show that reviewed exceptions
        were applied, not only the JSON report.
        """
        runtime_version = result.metadata.get("skillspector_version")
        if not isinstance(runtime_version, str):
            # No report was processed; the scan is already incomplete.
            baseline["message"] = (
                "SkillSpector did not return a usable report, so the baseline outcome is unknown; "
                "see the security check errors."
            )
            return
        suppressed = static_suppressed if isinstance(static_suppressed, int) else 0
        summary = static_summary if isinstance(static_summary, dict) else {}
        by_severity = summary.get("by_severity") if isinstance(summary.get("by_severity"), dict) else {}
        baseline["suppressed_by_severity"] = by_severity
        baseline["suppressed_rule_ids"] = summary.get("rule_ids") if isinstance(summary.get("rule_ids"), list) else []
        llm_stage_suppressed = llm_suppressed if isinstance(llm_suppressed, int) else None
        # Coverage advisories remain visible, but are not active scanner issues
        # and cannot be suppressed by a reviewed vulnerability baseline.
        active_findings = [
            finding for finding in result.findings if finding.check_name != "skillspector_reference_caveats"
        ]
        blocking = sum(
            1
            for finding in active_findings
            if (finding.severity if isinstance(finding.severity, Severity) else Severity(str(finding.severity).lower()))
            in (Severity.CRITICAL, Severity.HIGH)
        )
        baseline.update(
            {
                "loaded": True,
                "runtime_scanner_version": runtime_version,
                "suppressed_findings": suppressed,
                "llm_stage_suppressed_findings": llm_stage_suppressed,
                "active_findings": len(active_findings),
                "active_blocking_findings": blocking,
            }
        )
        counts = (
            f"rules={baseline['rule_count']}, fingerprints={baseline['fingerprint_count']}, suppressed={suppressed}"
        )
        if llm_stage_suppressed is not None:
            counts += f", llm_stage_suppressed={llm_stage_suppressed}"
        if by_severity:
            breakdown = ", ".join(f"{name}={count}" for name, count in sorted(by_severity.items()))
            counts += f", suppressed_by_severity: {breakdown}"
        file_label = f"{baseline['file']} ({baseline['digest']})"

        def report(status: str, message: str, *, problem: bool) -> None:
            baseline["status"] = status
            baseline["message"] = message
            # Recorded for every outcome, so reports that hide errors still show it.
            result.add_success(check_name="skillspector_baseline", message=f"{file_label}: {message}")
            if problem and blocking:
                # Explain remaining blockers without turning a passing scan into a failure.
                result.add_error(message)
            elif problem:
                result.add_warning(message)

        # SkillSpector compares scanner versions as exact strings and ignores
        # every fingerprint on a mismatch; rules still apply.
        if baseline["fingerprint_count"] and baseline["baseline_scanner_version"] != runtime_version:
            report(
                "fingerprint_version_mismatch",
                "SkillSpector baseline fingerprints were ignored: they were generated with SkillSpector "
                f"{baseline['baseline_scanner_version'] or 'unknown'}, but this scan used {runtime_version} "
                f"({counts}). Regenerate the fingerprints with SkillSpector {runtime_version}, or replace them "
                "with narrow reviewed rules.",
                problem=True,
            )
            return
        if llm_stage_suppressed is not None and llm_stage_suppressed < suppressed:
            # SkillSpector's LLM meta-analysis rewrites findings before the
            # baseline is applied, so fingerprints of static findings stop
            # matching and those findings come back from the LLM stage.
            report(
                "llm_stage_unmatched",
                f"SkillSpector baseline suppressed {suppressed} finding(s) in the static stage but only "
                f"{llm_stage_suppressed} in the LLM stage ({counts}). LLM analysis rewrites findings before the "
                "baseline applies, so fingerprints do not match them; use narrow id, path, or message rules for "
                "runs with LLM analysis, or scan without --llm.",
                problem=True,
            )
            return
        if suppressed or llm_stage_suppressed:
            report("applied", f"SkillSpector baseline applied ({counts}).", problem=False)
            return
        report("no_matches", f"SkillSpector baseline loaded but matched no findings ({counts}).", problem=False)

    @staticmethod
    def _convert_skillspector_issue(
        issue: dict,
        *,
        completeness: object = None,
        scan_root: Path | None = None,
        legacy_reason_index: Mapping[str, list[dict]] | None = None,
        normalizer: ReportNormalizer | None = None,
    ) -> tuple[Finding, bool]:
        """Convert a single skillspector issue dict into a Finding.

        Returns (Finding, is_error) where is_error is True for CRITICAL/HIGH.

        An ``AE1`` (incomplete referenced artifact analysis) finding is named
        by its rule when SkillSpector leaves ``pattern`` empty, as releases
        before v2.12.0 do. When SkillSpector says which file it did not
        completely inspect and why, its metadata and suggestion carry that
        (:func:`~skillevaluator.skillspector_references.issue_diagnostics`),
        with the reasons of earlier releases read from *completeness* or
        *legacy_reason_index*. Paths count only inside *scan_root*. A
        *normalizer* shared by a report's issues replaces *scan_root*, so each
        distinct path and text is redacted once per report.
        """
        g = _issue_field(issue)
        issue_sev = str(g("severity", "UNKNOWN")).upper()
        explanation = g("explanation")
        remediation = g("remediation")
        suggestion = f"{explanation} {remediation}".strip() or None
        code_snippet = g("code_snippet")
        file_path, line_number = SecurityValidator._parse_issue_location(issue)
        raw_pattern = issue.get("pattern")
        if issue.get("id") == AE1_RULE_ID and not (isinstance(raw_pattern, str) and raw_pattern.strip()):
            pattern = AE1_PATTERN
        else:
            pattern = g("pattern", "Unknown")
        metadata = SecurityValidator._build_issue_metadata(issue)
        normalizer = normalizer or ReportNormalizer(scan_root)
        metadata.update(
            issue_diagnostics(
                issue,
                normalize_path=normalizer.path,
                completeness=completeness,
                legacy_reason_index=legacy_reason_index,
                text=normalizer.text,
            )
        )
        if evidence := metadata.get("skillspector_evidence"):
            suggestion = reference_suggestion(evidence, remediation, text=normalizer.text)

        finding = Finding(
            category="SECURITY",
            severity=issue_sev,
            check_name=f"{pattern} ({g('id', '?')})",
            message=SecurityValidator._build_issue_message(g("category"), g("finding"), explanation, pattern),
            file_path=file_path,
            line_number=line_number,
            line_content=code_snippet[:200] if code_snippet else None,
            suggestion=suggestion,
            metadata=metadata,
        )
        return finding, issue_sev in ("CRITICAL", "HIGH")

    @staticmethod
    def _parse_issue_location(issue: dict) -> tuple[str, int | None]:
        """Extract (file_path, line_number) from a skillspector issue."""
        loc = issue.get("location")
        if not isinstance(loc, dict):
            return "unknown", None
        return (
            loc.get("file") or "unknown",
            loc.get("start_line") or loc.get("line"),
        )

    @staticmethod
    def _build_issue_metadata(issue: dict) -> dict:
        """Extract optional confidence and intent into a metadata dict."""
        metadata: dict = {}
        sp_confidence = issue.get("confidence")
        if sp_confidence is not None:
            metadata["skillspector_confidence"] = sp_confidence
        intent = issue.get("intent")
        if intent:
            metadata["intent"] = intent
        return metadata

    @staticmethod
    def _build_issue_message(category: str, finding_text: str, explanation: str, pattern_name: str) -> str:
        """Build a display message from skillspector issue fields."""
        if finding_text:
            return f"{category}: {finding_text}" if category else finding_text
        if explanation:
            return f"{category}: {explanation[:120]}" if category else explanation[:120]
        return pattern_name

    @staticmethod
    def _summarize_skillspector_results(
        issues: list,
        has_critical_or_high: bool,
        result: ValidationResult,
        *,
        suppressed_inapplicable_issues: int = 0,
    ) -> None:
        """Add a summary success/message entry for the skillspector scan."""
        suppressed_note = ""
        suppressed_metadata: dict[str, int] = {}
        if suppressed_inapplicable_issues:
            noun = "issue" if suppressed_inapplicable_issues == 1 else "issues"
            suppressed_note = (
                f"; {suppressed_inapplicable_issues} scanner {noun} suppressed outside agent control paths"
            )
            suppressed_metadata = {"suppressed_inapplicable_issues": suppressed_inapplicable_issues}
        suppressed = result.metadata.get("skillspector_suppressed_count")
        if not issues and isinstance(suppressed, int) and suppressed > 0:
            # Suppressions require an explicit baseline; never present them as a clean scan.
            result.add_success(
                check_name="skillspector",
                message=(
                    f"No active security findings; {suppressed} finding(s) suppressed by the requested baseline"
                    f"{suppressed_note}"
                ),
                suppressed_count=suppressed,
                **suppressed_metadata,
            )
        elif not issues:
            result.add_success(
                check_name="skillspector",
                message=(
                    f"No applicable security vulnerabilities detected{suppressed_note}"
                    if suppressed_note
                    else "No security vulnerabilities detected (secrets, API keys, credentials)"
                ),
                **suppressed_metadata,
            )
        elif has_critical_or_high:
            result.add_message(f"skillspector found {len(issues)} issue(s) including critical/high severity")
        else:
            result.add_success(
                check_name="skillspector",
                message=(
                    f"Security scan completed - {len(issues)} advisory finding(s) (no critical/high issues)"
                    f"{suppressed_note}"
                ),
                issue_count=len(issues),
                **suppressed_metadata,
            )

    def _scan_for_pii(self, skill_path: Path) -> ValidationResult:
        """Scan files for PII using regex patterns (emails, paths, SSNs, etc.)."""
        result = ValidationResult()
        files = self._get_scannable_files(skill_path)
        waivers = self._pii_waivers
        if waivers is not None:
            waivers.cover(skill_path)

        if not files:
            result.add_warning("No scannable files found for PII scan")
            return result

        result.summary.files_scanned = len(files)
        result.add_success(
            check_name="pii_scan_start",
            message=f"Scanning {len(files)} files for PII",
            file_count=len(files),
        )
        pii_found = False

        # Identities whose /home/<user>/ paths count as PII for this skill.
        # If neither the submitter nor the author resolves, the home-path check
        # is effectively a no-op; warn once per instance so operators can enable
        # it via SKILLEVALUATOR_SUBMITTER (logged here, not per matched line).
        protected_usernames = self._protected_home_usernames(skill_path)
        if not protected_usernames and not self._home_check_disabled_warned:
            self._home_check_disabled_warned = True
            logger.warning(
                "home-path PII check disabled: could not resolve author or submitter "
                "identity. Set SKILLEVALUATOR_SUBMITTER (or add 'author' to SKILL.md) to enable it."
            )

        # The same matched value (address, path, number) often repeats across
        # a doc-heavy skill; report one finding per (check, value) with the
        # occurrence list instead of one near-identical finding per line.
        groups: dict[object, dict] = {}
        for file_path in files:
            # Compute relative path for cleaner output
            try:
                relative_path = str(file_path.relative_to(skill_path))
            except ValueError:
                relative_path = file_path.name

            waive = waivers.matcher(file_path) if waivers is not None else None
            for finding_data in self._scan_file_for_pii(
                file_path, protected_usernames, skill_root=skill_path, waive=waive
            ):
                pii_found = True
                value = finding_data.get("matched_value")
                key: object = (
                    (finding_data["category"], value.casefold())
                    if value
                    else (finding_data["category"], relative_path, finding_data["line"])
                )
                group = groups.setdefault(
                    key, {"first": finding_data, "first_file": relative_path, "occurrences": [], "confidences": []}
                )
                group["occurrences"].append((relative_path, finding_data["line"]))
                group["confidences"].append(finding_data.get("confidence", "high"))

        from skillevaluator.utils.secret_redaction import (
            SENSITIVE_PII_CATEGORIES,
            Redactor,
            sensitive_values,
        )

        secret_values = set()
        sensitive_patterns = [item for item in self._compile_pii_patterns([]) if item[0] in SENSITIVE_PII_CATEGORIES]
        for group in groups.values():
            first = group["first"]
            line = first.get("line_content") or ""
            secret_values.update(sensitive_values(first["category"], first.get("matched_value") or "", line))
            for category, pattern, _exceptions, _definition in sensitive_patterns:
                for match in pattern.finditer(line):
                    secret_values.update(sensitive_values(category, match.group(), line))
        redactor = Redactor(secret_values)
        confidence_rank = {"low": 0, "medium": 1, "high": 2}
        for group in groups.values():
            first = group["first"]
            value = first.get("matched_value")
            occurrences = group["occurrences"]
            severity = first["severity"].upper()

            sensitive = first["category"] in SENSITIVE_PII_CATEGORIES
            display_value = "[REDACTED]" if sensitive and value else value
            message = f"{first['description']}: {display_value}" if display_value else first["description"]
            message = redactor.text(message)
            metadata: dict = {"confidence": max(group["confidences"], key=lambda c: confidence_rank.get(c, 1))}
            if value:
                metadata["matched_value"] = display_value
                metadata["occurrence_count"] = len(occurrences)
                metadata["occurrences"] = [{"file": f, "line": line} for f, line in occurrences]
            if len(occurrences) > 1:
                message += f" — {len(occurrences)} occurrences ({self._format_occurrences(occurrences)})"
                message = redactor.text(message)
            metadata = redactor.object(metadata)
            if value and not sensitive and redactor.text(value) == value:
                # Identifies the matched value for a PII waiver (see validators/pii_waivers.py). Only a
                # value the report shows in full gets one: a digest of a masked credential, or of a value
                # that contains one, would let anyone holding the report test guesses for it offline.
                # It is added after redaction, which would otherwise mask a short secret that happens
                # to occur in the hex digest of another value.
                metadata["evidence_sha256"] = pii_evidence_sha256(first["category"], value)

            finding = Finding(
                category="PII",
                severity=severity,
                check_name=first["category"],
                message=message,
                file_path=redactor.text(group["first_file"]),
                line_number=first["line"],
                line_content=None if sensitive else redactor.text(first.get("line_content") or "") or None,
                suggestion=first.get("suggestion"),
                metadata=metadata,
            )
            result.add_structured_finding(finding, is_error=severity in ("CRITICAL", "HIGH"))

        if not pii_found:
            result.add_success(
                check_name="pii_detection",
                message=f"No PII detected in {len(files)} files (emails, SSNs, phone numbers, paths)",
                files_scanned=len(files),
            )

        return result

    @staticmethod
    def _format_occurrences(occurrences: list[tuple[str, int]], max_files: int = 3, max_lines: int = 10) -> str:
        """Render grouped occurrence locations compactly, e.g. ``a.md lines 3, 9; b.md line 12``."""
        by_file: dict[str, list[int]] = {}
        for rel_path, line in occurrences:
            by_file.setdefault(rel_path, []).append(line)

        parts = []
        for index, (rel_path, lines) in enumerate(by_file.items()):
            if index == max_files:
                parts.append(f"+{len(by_file) - max_files} more file(s)")
                break
            shown = ", ".join(str(n) for n in lines[:max_lines])
            if len(lines) > max_lines:
                shown += ", …"
            label = "line" if len(lines) == 1 else "lines"
            parts.append(f"{rel_path} {label} {shown}")
        return "; ".join(parts)

    # Keywords that suggest a PII match is likely a false positive
    _LOW_CONFIDENCE_HINTS: tuple[str, ...] = (
        # Numeric quantities commonly mistaken for GPS/phone
        "words",
        "characters",
        "chars",
        "lines",
        "pages",
        "items",
        "files",
        "bytes",
        "tokens",
        "total",
        "count",
        "size",
        "length",
        "offset",
        "index",
        "version",
        "Timestamp",
        "timestamp",
        "duration",
        # URL context
        "http://",
        "https://",
        "pageId",
        "url=",
        "href=",
        "src=",
        # Code/doc context
        "```",
        "import ",
        "from ",
        "def ",
        "class ",
        "return ",
    )

    def _estimate_confidence(self, category: str, line: str) -> str:
        """Estimate confidence level of a PII match based on line context.

        Returns 'high', 'medium', or 'low' confidence.
        High-confidence categories (secrets, SSNs, private keys) always return 'high'.
        """
        # These categories are almost never false positives
        high_confidence_categories = {
            "ssn",
            "private_keys",
            "hardcoded_secrets",
            "database_credentials",
            "github_tokens",
            "aws_identifiers",
            "jwt_tokens",
            "webhook_urls",
        }
        if category in high_confidence_categories:
            return "high"

        # Check if the line contains low-confidence hints
        line_lower = line.lower()
        hint_count = sum(1 for hint in self._LOW_CONFIDENCE_HINTS if hint.lower() in line_lower)

        if hint_count >= 2:
            return "low"
        if hint_count == 1:
            return "medium"

        return "high"

    def _is_personal_home_path(self, match: re.Match, protected_usernames: set[str]) -> bool:
        """Return True when a ``/home/<root>/`` match is a personal home directory.

        Implements the identity-based home-path check: a ``/home/<root>/`` path
        is treated as PII only when ``<root>`` matches the skill author or the
        submitter (``protected_usernames``). This avoids organization-specific
        root allowlists while still detecting contributor identity leakage.

        The ``home_paths`` pattern captures the first path component after
        ``/home/`` in group 1; the comparison is anchored to that component and
        case-insensitive.
        """
        if not match.re.groups or match.lastindex is None:
            return False
        root = (match.group(1) or "").strip()
        if not root:
            return False
        if not protected_usernames:
            return False
        return root.lower() in protected_usernames

    _GPS_ZERO_PATTERN = re.compile(r"[-+]?0+\.0+[,\s]+[-+]?0+\.0+")
    # Geospatial cues. A strong cue names a geographic coordinate system, a map
    # link (an RFC 5870 ``geo:`` URI or a Google, Apple, OpenStreetMap, or Bing
    # map URL), or a GeoJSON object: a geometry ``"type"``, a ``Feature``,
    # ``FeatureCollection`` or ``GeometryCollection`` ``"type"``, or a
    # ``"geometry": {`` member, which comes before ``"coordinates"`` when keys
    # are sorted. GeoJSON member names and these object types are
    # case-sensitive, so ``"type": "feature"`` in other JSON is not a cue. A soft
    # cue (location, office, address, degrees, or a degree sign) also labels
    # technical data such as ``calibration_location_offset`` or a rotation in
    # degrees, so it does not override a technical label on the same line.
    # Cue words accept their plurals and a numeric suffix (``locations:``,
    # ``latlngs``, ``lat1, lon1``, ``location2``). The GeoJSON ``type`` and
    # ``geometry`` keys may be double-quoted, single-quoted (a Python dict) or
    # bare (a JS object or YAML); a bare type value must use GeoJSON's own
    # spelling (``Point``, ``MultiPolygon``). Map hosts are matched with bounded
    # TLD forms so long lines stay linear.
    _GPS_STRONG_CONTEXT_PATTERN = re.compile(
        r"(?i)\b(?:gps|geolocations?|geospatial|latitudes?|longitudes?|lats?|lons?|lngs?|latlngs?|latlong?s?"
        r"|lonlats?|lnglats?|wgs[- ]?84)\d*\b"
        r"|\bgeo:"
        r"|\bmaps\.google\.|\bgoogle(?:\.[a-z]{2,3}){1,2}/maps\b|\bmaps\.apple\.com\b|\bopenstreetmap\.org\b"
        r"|\bbing\.com/maps\b"
        r"|\btype[\"']?\s*:\s*(?:[\"'](?:(?:multi)?(?:point|linestring|polygon)"
        r"|(?-i:Feature(?:Collection)?|GeometryCollection))[\"']"
        r"|(?-i:(?:Multi)?(?:Point|LineString|Polygon)|Feature(?:Collection)?|GeometryCollection)\b)"
        r"|\b(?-i:geometry)[\"']?\s*:\s*\{"
    )
    _GPS_SOFT_CONTEXT_PATTERN = re.compile(r"(?i)\b(?:locations?|offices?|address(?:es)?|degrees?)\d*\b|[°º]")
    _TECHNICAL_COORDINATE_CONTEXT_PATTERN = re.compile(
        r"(?i)\b(?:texture|uv|pixel|screen|canvas|image|normalized|cartesian|robot(?:ic(?:s)?)?|local|world|camera"
        r"|pose|joint|model|object|viewport|mesh|vertex|clip|device)[ -]+coordinates?\b"
    )
    _TECHNICAL_VECTOR_CONTEXT_PATTERN = re.compile(
        r"(?i)\b(?:quaternion|pose(?:[ _-]?translation)?|waypoint|vector|calibration|roi|scale)s?\b"
        r"|\bcoordinates?\s*=|(?:^|\s)(?:x\s+y|u\s+v)(?:\s|$)"
    )
    _CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
    # Double- or single-quoted strings (one line at most, so a stray quote in
    # prose cannot run on), bare identifiers (JS object keys), and structural
    # characters, for the GeoJSON ``"coordinates"`` array pass.
    _JSON_TOKEN_PATTERN = re.compile(r'"((?:[^"\\\n]|\\.)*)"|\'((?:[^\'\\\n]|\\.)*)\'|([A-Za-z_$][\w$]*)|([{}\[\]:,])')
    _GEOJSON_GEOMETRY_TYPE = re.compile(r"(?i)(?:multi)?(?:point|linestring|polygon)")
    # A bare (unquoted) type value must use GeoJSON's own spelling.
    _GEOJSON_BARE_GEOMETRY_TYPE = re.compile(r"(?:Multi)?(?:Point|LineString|Polygon)")
    _SCIENTIFIC_LITERAL_PATTERN = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)[eE][-+]?\d+")
    # ``<identifier>_`` directly before a candidate; group 1 is the identifier.
    _UNDERSCORE_JOINED_PREFIX_PATTERN = re.compile(r"(?-i:([A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)_)\Z")
    _PAYMENT_LABEL_WORDS = frozenset(
        {"card", "cards", "cc", "ccn", "pan", "credit", "debit", "payment", "visa", "mastercard", "amex"}
    )
    # A timestamp or digest label that directly precedes a value
    # (``timestamp_microseconds: ``, ``"sha256": "``, ``checksum=``). An
    # underscore separates words, so snake_case prefixes such as
    # ``event_timestamp:`` and ``"created_at_epoch":`` count as labels.
    _NON_PAYMENT_NUMBER_CONTEXT_PATTERN = re.compile(
        r"(?i)(?:(?<![A-Za-z0-9])(?:(?:timestamp|unix(?:[_ -]?time)?|epoch)"
        r"(?:[_ -]?(?:microseconds?|micros?|us|usec|nanoseconds?|nanos?|ns|nsec))?"
        r"|(?:microseconds?|nanoseconds?)(?:[_ -]?timestamp)?)\b"
        r"|(?<![A-Za-z0-9])(?:md5|sha(?:1|224|256|384|512)|hash|digest|checksum)\b)"
        r"[\"']?(?:\s*[:=]\s*[\"']?|\s+[\"']?)\Z"
    )
    # Match version/tag as identifier tokens, including separator and camel-case styles.
    # Plain substrings such as ``conversion`` and ``staging`` are not version labels.
    _VERSION_LABEL_PATTERN = re.compile(
        r"(?:"
        r"(?i:(?<![a-z0-9])(?:[a-z0-9]+[_-])*(?:versions?|tags?)(?:[_-][a-z0-9]+)*(?![a-z0-9]))"
        r"|(?<![A-Za-z0-9])(?:[A-Za-z][a-z0-9]*)*(?:Version|Versions|Tag|Tags)"
        r"(?:[A-Z][A-Za-z0-9]*)*(?![A-Za-z0-9])"
        r"|(?<![A-Za-z0-9])(?:version|versions|tag|tags)"
        r"(?:[A-Z][A-Za-z0-9]*)+(?![A-Za-z0-9])"
        r")"
    )
    _PACKAGE_VERSION_CALL_PATTERN = re.compile(
        r"(?i)\b[a-z_]\w*(?:wheel|archive|artifact|package|conda)[a-z_]*\([^)]*\Z"
    )
    _PACKAGE_ARTIFACT_SUFFIX_PATTERN = (
        r"(?:7z|apk|conda|crate|deb|egg|gem|jar|nupkg|rpm|tgz|tbz2|txz|whl|zip|"
        r"tar\.(?:bz2|gz|xz|zst))"
    )
    _NETWORK_ADDRESS_PATTERN = re.compile(
        r"(?i)(?:^|[^a-z0-9])(?:address|dns|endpoint|gateway|host|hostname|ip|mirror|nameserver|proxy|"
        r"registry|resolver|server|uri|url)(?=$|[^a-z0-9])"
    )
    _URL_AUTHORITY_PREFIX_PATTERN = re.compile(r"(?i)[a-z][a-z0-9+.-]*://[^/\s\"']*\Z")
    _IPV4_LITERAL_PATTERN = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
    # Chromium's reduced User-Agent reports every version as ``<major>.0.0.0``
    # after a product token, as in ``Chrome/140.0.0.0`` (or ``Chrome\/140.0.0.0``
    # in JSON with escaped slashes). A ``.0.0.0`` value is never a host address,
    # and a version is never followed by ``/``, so CIDR blocks stay reported.
    _USER_AGENT_PRODUCT_PREFIX_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]*\\?/\Z")
    _REDUCED_USER_AGENT_VERSION_PATTERN = re.compile(r"\d{1,3}\.0\.0\.0")
    # A canonical 8-4-4-4-12 GUID/UUID, such as a COM interface ID. Its digit
    # groups can form a Luhn-valid 16-digit run without being a card number.
    _UUID_PATTERN = re.compile(r"(?i)(?<!\w)[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}(?!\w)")
    # A payment label that directly precedes a value (``card_number: "``,
    # ``credit_card=``, ``cc_no: ``, ``PAN ``), anchored to the end of the prefix.
    _PAYMENT_LABEL_PATTERN = re.compile(
        r"(?i)(?<![a-z0-9])(?:(?:credit[_ -]?)?card(?:[_ -]?(?:number|num|no))?|cc[_ -]?(?:number|num|no)|pan)"
        r"(?![a-z0-9])[\"']?(?:\s*[:=]\s*[\"']?|\s+[\"']?)\Z"
    )

    # Proton (Steam Play) runs each Windows game in its own prefix whose only
    # Windows profile is ``pfx/drive_c/users/steamuser/``. Proton uses that name
    # for every player, so the path names no person. ``pfx`` must start a path
    # component. A ``..`` segment after the profile could lead anywhere, so a
    # line with one anywhere after the profile stays reported: Windows folder
    # names often contain spaces and quotes, so the path's end is not reliable.
    # A ``..`` counts as a segment when a path separator (plain or encoded),
    # whitespace, a quote, or another delimiter bounds it on both sides, so
    # ``steamuser/" + "../x"``, ``/ ".." /``, ``"..",``, ``cd ../x`` and
    # ``cd ..&& cd x`` count. Shell ``&`` and ``:`` bound a ``..`` but not ``pfx``.
    _PROTON_PREFIX = "pfx/drive_c"
    _PROTON_PREFIX_BOUNDARIES = frozenset("/\\'\"`=(){}[]<>|*,; \t")
    _DOT_DOT_BOUNDARY = (
        "[" + "".join(re.escape(c) for c in sorted(_PROTON_PREFIX_BOUNDARIES | {"&", ":"})) + r"\s]|%2f|%5c"
    )
    _DOT_DOT_SEGMENT_PATTERN = re.compile(
        rf"(?:^|{_DOT_DOT_BOUNDARY})(?:\.|%2e){{2}}(?=$|{_DOT_DOT_BOUNDARY})", re.IGNORECASE
    )

    @classmethod
    def _is_proton_virtual_profile(cls, match: re.Match, line: str) -> bool:
        """Return whether a personal-path match is Proton's ``pfx/drive_c/users/steamuser/`` profile."""
        value = match.group()
        if not value.isascii() or value.lower() != "/users/steamuser/":
            return False
        start = match.start() - len(cls._PROTON_PREFIX)
        prefix = line[max(start, 0) : match.start()]
        if start < 0 or not prefix.isascii() or prefix.lower().replace("\\", "/") != cls._PROTON_PREFIX:
            return False
        if start > 0 and line[start - 1] not in cls._PROTON_PREFIX_BOUNDARIES:
            return False
        return cls._DOT_DOT_SEGMENT_PATTERN.search(line[match.end() :]) is None

    @staticmethod
    def _is_near_zero_gps(line: str) -> bool:
        """Check if a GPS match contains only near-zero coordinates (Null Island)."""
        return bool(SecurityValidator._GPS_ZERO_PATTERN.search(line))

    @classmethod
    def _is_numeric_pii_false_positive(cls, category: str, match: re.Match, line: str) -> bool:
        """Return whether a card or phone candidate on *line* is technical data."""
        if category == "credit_cards":
            # A Luhn-valid timestamp or digest is not payment data. Only a label
            # directly before this number counts, so a timestamp field does not
            # hide a card number in a later field on the same line.
            label_start = max(0, match.start() - 100)
            if cls._NON_PAYMENT_NUMBER_CONTEXT_PATTERN.search(line, label_start, match.start()) is not None:
                return True
            # An underscore joining the number to a letter or digit makes it part
            # of an identifier (``id_1707938736269577``), unless a word in that
            # identifier is a payment label (``card_4532015112830366``).
            joined = cls._UNDERSCORE_JOINED_PREFIX_PATTERN.search(line, label_start, match.start())
            if joined is None:
                return False
            words = cls._CAMEL_CASE_BOUNDARY.sub("_", joined.group(1)).lower().split("_")
            return not any(word in cls._PAYMENT_LABEL_WORDS for word in words)

        if category == "phone_numbers":
            # International phone syntax can begin at the ``+00`` exponent of a
            # scientific-notation value and run into the next column. Only a
            # candidate that starts inside such a literal is dropped, so a real
            # phone number elsewhere on the line is still reported.
            window = (max(0, match.start() - 64), match.start() + 64)
            return any(
                literal.start() <= match.start() < literal.end()
                for literal in cls._SCIENTIFIC_LITERAL_PATTERN.finditer(line, *window)
            )

        return False

    @classmethod
    def _cue_text_variants(cls, text: str) -> tuple[str, ...]:
        """Return *text* with underscores as spaces, unsplit and camelCase-split.

        Keys and file names often use snake_case or camelCase (``homeLocation``,
        ``gpsCoords``). The unsplit form keeps GeoJSON ``"type"`` values such as
        ``FeatureCollection`` intact.
        """
        spaced = text.replace("_", " ")
        split = cls._CAMEL_CASE_BOUNDARY.sub(" ", spaced)
        return (spaced,) if split == spaced else (spaced, split)

    @staticmethod
    def _search_cue(pattern: re.Pattern, variants: tuple[str, ...]) -> bool:
        """Return whether *pattern* matches any form of the cue text."""
        return any(pattern.search(text) for text in variants)

    @classmethod
    def _lacks_geospatial_context(
        cls, line: str, context: str, pair_start: int | None = None, label_start: int = 0
    ) -> bool:
        """Return whether GPS-shaped pairs on *line* have no geospatial cue.

        Precise vectors, poses, quaternions, and waypoints have the same shape
        as coordinates, so a pair needs a cue. *context* is the file name, up to
        two lines before *line*, and *line*. Keys and file names often use
        snake_case or camelCase, so underscores and case changes separate words.
        Vector labels such as ``waypoint`` count only before the pair at
        *pair_start*; a GPS cue after a separator can limit that label segment.
        """
        local = cls._cue_text_variants(line)
        # A technical label labels the pair only when it comes before it, so
        # ``Location: 37.77, -122.41 (map scale 1:5000)`` keeps its location cue.
        label_end = len(line) if pair_start is None else pair_start
        label_segment = line[label_start:label_end]
        last_separator = max(label_segment.rfind(";"), label_segment.rfind("|"))
        tail = label_segment[last_separator + 1 :]
        tail_variants = cls._cue_text_variants(tail)
        if cls._search_cue(cls._GPS_SOFT_CONTEXT_PATTERN, tail_variants) or cls._search_cue(
            cls._GPS_STRONG_CONTEXT_PATTERN, tail_variants
        ):
            label_segment = tail
        label = cls._cue_text_variants(label_segment)
        nearby = cls._cue_text_variants(context)
        if cls._search_cue(cls._GPS_STRONG_CONTEXT_PATTERN, local):
            return False
        inherits_technical_label = False
        if label_start and ";" not in line[label_start:label_end] and "|" not in line[label_start:label_end]:
            previous_separator = max(line.rfind(";", 0, label_start), line.rfind("|", 0, label_start))
            previous_label = cls._cue_text_variants(line[previous_separator + 1 : label_start])
            inherits_technical_label = cls._search_cue(
                cls._TECHNICAL_COORDINATE_CONTEXT_PATTERN, previous_label
            ) or cls._search_cue(cls._TECHNICAL_VECTOR_CONTEXT_PATTERN, previous_label)
        if (
            cls._search_cue(cls._TECHNICAL_COORDINATE_CONTEXT_PATTERN, label)
            or cls._search_cue(cls._TECHNICAL_VECTOR_CONTEXT_PATTERN, label)
            or inherits_technical_label
        ):
            return not cls._search_cue(cls._GPS_STRONG_CONTEXT_PATTERN, nearby)
        if cls._search_cue(cls._GPS_SOFT_CONTEXT_PATTERN, local):
            return False
        if cls._search_cue(cls._GPS_STRONG_CONTEXT_PATTERN, nearby):
            return False
        if cls._search_cue(cls._TECHNICAL_COORDINATE_CONTEXT_PATTERN, nearby):
            return True
        return not cls._search_cue(cls._GPS_SOFT_CONTEXT_PATTERN, nearby)

    @classmethod
    def _geojson_coordinate_lines(cls, lines: list[str]) -> frozenset[int]:
        """Return 1-based numbers of lines inside a GeoJSON geometry's ``"coordinates"`` array.

        A pretty-printed ``Polygon`` or ``Multi*`` geometry opens its rings on
        their own lines, so positions can sit far below the ``"type"`` cue. The
        array counts when its object has a geometry ``"type"`` (before or after
        ``"coordinates"``) or is the value of a ``"geometry"`` member. Keys and
        values may be double-quoted (JSON), single-quoted (a Python dict) or
        bare keys (a JS object). One pass over these tokens finds the arrays,
        and nested arrays are merged before their lines are marked, so the scan
        stays linear however deep the nesting; unbalanced brackets in other
        text are tolerated.
        """
        text = "\n".join(lines)
        if "coordinates" not in text:
            return frozenset()
        line_starts = [0, *itertools.accumulate(len(line) + 1 for line in lines)]
        # Each frame is [kind, is_geojson, coordinate_spans, coordinates_owner, start].
        stack: list[list] = []
        spans: list[tuple[int, int]] = []
        last_string: str | None = None
        key: str | None = None
        for token in cls._JSON_TOKEN_PATTERN.finditer(text):
            bare = token.group(3)
            value = token.group(1) if token.group(1) is not None else token.group(2)
            if value is None:
                value = bare
            punct = token.group(4)
            top = stack[-1] if stack else None
            if value is not None:
                type_pattern = cls._GEOJSON_BARE_GEOMETRY_TYPE if bare is not None else cls._GEOJSON_GEOMETRY_TYPE
                if key == "type" and top is not None and top[0] == "{" and type_pattern.fullmatch(value):
                    top[1] = True
                last_string, key = value, None
            elif punct == ":":
                key, last_string = last_string, None
            elif punct in "{[":
                owner = top if punct == "[" and key == "coordinates" and top is not None and top[0] == "{" else None
                stack.append([punct, punct == "{" and key == "geometry", [], owner, token.start()])
                key = last_string = None
            elif punct in "}]":
                if not stack:
                    continue
                frame = stack.pop()
                if frame[3] is not None:
                    frame[3][2].append((frame[4], token.start()))
                if frame[0] == "{" and frame[1]:
                    spans.extend(frame[2])
                key = last_string = None
            else:
                key = last_string = None
        # Nested arrays yield nested spans; keep only the outermost of each
        # group so every line is marked once.
        numbers: set[int] = set()
        kept_end = -1
        for start, end in sorted(spans):
            if end <= kept_end:
                continue
            first = bisect.bisect_right(line_starts, max(start, kept_end + 1))
            last = bisect.bisect_right(line_starts, end)
            numbers.update(range(first, last + 1))
            kept_end = end
        return frozenset(numbers)

    @classmethod
    def _is_version_literal(cls, match: re.Match, line: str) -> bool:
        """Return whether an IPv4-shaped match is clearly a release version."""
        prefix = line[: match.start()]
        if cls._URL_AUTHORITY_PREFIX_PATTERN.search(prefix):
            return False

        literal = re.escape(match.group())
        artifact_pattern = re.compile(
            rf"(?i)[a-z0-9_./+\\-]*{literal}[a-z0-9_.+\\-]*\."
            rf"{cls._PACKAGE_ARTIFACT_SUFFIX_PATTERN}(?:[?#][^\s\"']*)?"
        )
        if any(
            artifact.start() <= match.start() and artifact.end() >= match.end()
            for artifact in artifact_pattern.finditer(line)
        ):
            return True

        if (
            cls._REDUCED_USER_AGENT_VERSION_PATTERN.fullmatch(match.group())
            and cls._USER_AGENT_PRODUCT_PREFIX_PATTERN.search(prefix)
            and not line.startswith("/", match.end())
        ):
            return True

        quoted_value: str | None = None
        quote_start = max(line.rfind('"', 0, match.start()), line.rfind("'", 0, match.start()))
        if quote_start >= 0:
            quote = line[quote_start]
            quote_end = line.find(quote, match.end())
            if quote_end >= 0:
                quoted_value = line[quote_start + 1 : quote_end]

        context_start = max(0, match.start() - 80)
        context_end = min(len(line), match.end() + 80)
        context = line[context_start:context_end]
        if cls._NETWORK_ADDRESS_PATTERN.search(context):
            return False
        prefix_context = line[context_start : match.start()]
        version_labels = list(cls._VERSION_LABEL_PATTERN.finditer(prefix_context))
        if version_labels:
            nearest_label = version_labels[-1]
            between_label_and_value = prefix_context[nearest_label.end() :]
            assignment_prefix = re.fullmatch(r"\s*(?::|=)\s*[\[({'\"]*\s*", between_label_and_value)
            if assignment_prefix and not cls._IPV4_LITERAL_PATTERN.search(between_label_and_value):
                return True

        return bool(quoted_value == match.group() and cls._PACKAGE_VERSION_CALL_PATTERN.search(prefix))

    @staticmethod
    def _passes_luhn(digits: str) -> bool:
        """Validate a number string using the Luhn algorithm.

        Returns True if the digit string passes the Luhn check, meaning it
        could be a valid credit card number. An all-zero string has a zero
        checksum but is never a card number, so it does not pass.
        """
        total = 0
        for i, ch in enumerate(reversed(digits)):
            n = int(ch)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        return total > 0 and total % 10 == 0

    @classmethod
    def _is_card_number_candidate(cls, match: re.Match, line: str, previous_line: str = "") -> bool:
        """Return whether a card-shaped match on *line* can be a payment card number.

        The digits must pass the Luhn check. Digits inside a canonical GUID/UUID
        (optionally in braces) are an identifier, not a card, unless a payment
        label directly precedes the GUID on its line or ends *previous_line*
        (a YAML block value). Both prefixes are bounded so long lines stay linear.
        """
        if not cls._passes_luhn(re.sub(r"[\s-]", "", match.group())):
            return False
        # A GUID is 36 characters, so one that contains the match lies in this window.
        guid = next(
            (
                found
                for found in cls._UUID_PATTERN.finditer(line, max(0, match.start() - 36), match.end() + 36)
                if found.start() <= match.start() and match.end() <= found.end()
            ),
            None,
        )
        if guid is None:
            return True
        value_start = guid.start()
        if value_start and line[value_start - 1] == "{" and line[guid.end() : guid.end() + 1] == "}":
            value_start -= 1
        prefix = f"{previous_line[-100:]}\n{line[max(0, value_start - 100) : value_start]}"
        return cls._PAYMENT_LABEL_PATTERN.search(prefix) is not None

    @staticmethod
    def _is_frontmatter_delimiter(line: str) -> bool:
        """True when a line is a YAML frontmatter fence, including BOM-prefixed openers."""
        return line.strip().removeprefix("\ufeff").strip() == "---"

    def _scan_file_for_pii(
        self,
        file_path: Path,
        protected_usernames: set[str] | None = None,
        skill_root: Path | None = None,
        *,
        waive: Callable[[str, int, str], bool] | None = None,
    ) -> list[dict]:
        """Scan a single file for PII patterns, yielding findings with full context.

        ``protected_usernames`` is the set of author/submitter identities used by
        the home-path check; when omitted it is resolved from the file's parent
        directory so the method stays usable standalone. ``waive`` returns
        whether a PII waiver covers a candidate (category, line, value); a
        waived candidate is skipped as if it did not match.

        ``skill_root`` is the directory being scanned. Only the manifest sitting
        directly in it may declare ownership contacts; a ``SKILL.md`` nested
        deeper in the tree is ordinary content. It defaults to the file's parent.
        """
        if protected_usernames is None:
            protected_usernames = self._protected_home_usernames(file_path.parent)

        try:
            content = file_path.read_text(encoding="utf-8-sig", errors="ignore")
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return []

        lines = content.split("\n")
        author_emails = self._frontmatter_author_emails(file_path, lines)
        skill_root = file_path.parent if skill_root is None else skill_root
        is_root_manifest = skill_root in (file_path.parent, file_path)
        ownership_spans = ownership_email_spans(file_path, content) if is_root_manifest else {}
        comment_lines = _comment_line_numbers(file_path, lines)
        global_exceptions = self.pii_patterns.get("exceptions", {}).get("allowed_paths", [])
        compiled = self._compile_pii_patterns(global_exceptions)

        findings: list[dict] = []
        for category, regex, exceptions, pattern_def in compiled:
            self._scan_lines_for_pattern(
                lines,
                category,
                regex,
                exceptions,
                pattern_def,
                file_path,
                findings,
                protected_usernames,
                author_emails,
                comment_lines,
                ownership_spans,
                waive=waive,
            )
        return findings

    def _frontmatter_author_emails(self, file_path: Path, lines: list[str]) -> dict[int, str]:
        """Map valid frontmatter author lines to the public contributor email."""
        if file_path.name not in SKILL_MANIFEST_VARIANTS or not lines or not self._is_frontmatter_delimiter(lines[0]):
            return {}
        try:
            frontmatter_end = next(
                index for index, line in enumerate(lines[1:], 1) if self._is_frontmatter_delimiter(line)
            )
        except StopIteration:
            return {}

        identity = self._author_identity(file_path.parent)
        identity_match = _AUTHOR_IDENTITY_RE.fullmatch(identity or "")
        if identity_match is None:
            return {}
        author_email = identity_match.group("email")

        return {
            line_number: author_email
            for line_number, line in enumerate(lines[1:frontmatter_end], 2)
            if re.match(r"^\s*author\s*:", line, flags=re.IGNORECASE) and author_email.casefold() in line.casefold()
        }

    def _compile_pii_patterns(self, global_exceptions: list[str]) -> list[tuple[str, re.Pattern, list[str], dict]]:
        """Pre-compile all PII patterns with their merged exception lists."""
        compiled: list[tuple[str, re.Pattern, list[str], dict]] = []
        for category, patterns in self.pii_patterns.items():
            if category == "exceptions" or not isinstance(patterns, list):
                continue
            for pattern_def in patterns:
                raw = pattern_def.get("pattern")
                if not raw:
                    continue
                try:
                    regex = re.compile(raw, re.IGNORECASE)
                except re.error:
                    continue
                exceptions = global_exceptions + pattern_def.get("exceptions", [])
                compiled.append((category, regex, exceptions, pattern_def))
        return compiled

    def _scan_lines_for_pattern(
        self,
        lines: list[str],
        category: str,
        regex: re.Pattern,
        exceptions: list[str],
        pattern_def: dict,
        file_path: Path,
        findings: list[dict],
        protected_usernames: set[str] | None = None,
        author_emails: dict[int, str] | None = None,
        comment_lines: frozenset[int] | None = None,
        ownership_spans: dict[int, list[tuple[int, int]]] | None = None,
        *,
        waive: Callable[[str, int, str], bool] | None = None,
    ) -> None:
        """Check all lines against a single compiled PII pattern.

        Only the first remaining match on a line is reported. Matches are
        filtered before that choice, so a value that is not reported (a version
        literal, the Proton profile, an unrelated home directory, a number that
        is not a card number, or a waived value) does not hide another value on
        the same line.
        """
        protected_usernames = protected_usernames or set()
        author_emails = author_emails or {}
        ownership_spans = ownership_spans or {}
        comment_lines = comment_lines if comment_lines is not None else _comment_line_numbers(file_path, lines)
        line_exceptions = exceptions
        excepted_card_digits: frozenset[str] = frozenset()
        if category == "credit_cards":
            # Known test card numbers are compared with each candidate, in any
            # grouping, so one on a line does not hide a different card number.
            numeric_exceptions = {exc for exc in exceptions if re.fullmatch(r"[0-9\s-]*[0-9][0-9\s-]*", exc)}
            excepted_card_digits = frozenset(re.sub(r"[\s-]", "", exc) for exc in numeric_exceptions)
            line_exceptions = [exc for exc in exceptions if exc not in numeric_exceptions]
        geojson_lines: frozenset[int] | None = None
        for line_num, line in enumerate(lines, 1):
            if line_num in comment_lines:
                continue

            scan_line = line
            if category == "emails" and (author_email := author_emails.get(line_num)):
                # Keep source offsets intact for adjacent ownership contacts in
                # flow YAML; substituting a different-length address shifts them.
                scan_line = re.sub(
                    re.escape(author_email), lambda match: " " * len(match.group()), line, count=1, flags=re.IGNORECASE
                )

            matches = list(regex.finditer(scan_line))
            if excepted_card_digits:
                matches = [
                    match for match in matches if re.sub(r"[\s-]", "", match.group()) not in excepted_card_digits
                ]
            if not matches or any(exc in line for exc in line_exceptions):
                continue
            if category in ("credit_cards", "phone_numbers"):
                matches = [match for match in matches if not self._is_numeric_pii_false_positive(category, match, line)]
                if not matches:
                    continue
            if category == "gps_coordinates":
                # A position inside a GeoJSON geometry's "coordinates" array is
                # cued however far its rings sit below the "type" line.
                if geojson_lines is None:
                    geojson_lines = self._geojson_coordinate_lines(lines)
                if line_num not in geojson_lines:
                    context = "\n".join((file_path.name, *lines[max(0, line_num - 3) : line_num]))
                    previous_end = 0
                    kept_matches = []
                    for pair in matches:
                        if not self._lacks_geospatial_context(line, context, pair.start(), label_start=previous_end):
                            kept_matches.append(pair)
                        previous_end = pair.end()
                    matches = kept_matches
                    if not matches:
                        continue
            if category == "ip_addresses":
                matches = [match for match in matches if not self._is_version_literal(match, line)]
            elif category == "personal_paths":
                matches = [match for match in matches if not self._is_proton_virtual_profile(match, line)]
            elif category == "home_paths":
                # /home/<root>/ is PII only when <root> is the author/submitter
                # username; unrelated roots are skipped.
                matches = [match for match in matches if self._is_personal_home_path(match, protected_usernames)]
            elif category == "credit_cards":
                # Check every candidate: a GUID or non-Luhn number must not hide a later card.
                previous_line = lines[line_num - 2] if line_num > 1 else ""
                matches = [match for match in matches if self._is_card_number_candidate(match, line, previous_line)]
            # A waiver is consulted only for a value that would be reported, so
            # it never counts as applied for a value the scan already dropped.
            if waive is not None:
                matches = [match for match in matches if not waive(category, line_num, self._reported_value(match))]
            if not matches:
                continue
            match = matches[0]

            confidence = self._estimate_confidence(category, line)
            if category == "gps_coordinates" and self._is_near_zero_gps(line):
                confidence = "low"

            # Preserve the first occurrence of each email class on a line.
            # An ownership contact must not hide an ordinary address beside it.
            candidates = [(category, match)]
            if category == "emails":
                classes = {}
                for candidate in matches:
                    check = (
                        "ownership_emails"
                        if candidate.span() in ownership_spans.get(line_num, [])
                        else "emails"
                    )
                    classes.setdefault(check, candidate)
                candidates = list(classes.items())
            for check, candidate in candidates:
                finding = {
                    "file": file_path.name,
                    "line": line_num,
                    "line_content": line,
                    "severity": pattern_def.get("severity", "medium"),
                    "description": pattern_def.get("description", category),
                    "suggestion": pattern_def.get("suggestion"),
                    "category": check,
                    "confidence": confidence,
                    "matched_value": self._reported_value(candidate),
                }
                if check == "ownership_emails":
                    finding["description"] = "Ownership contact email in skill metadata"
                    finding["suggestion"] = (
                        "Review whether ownership contacts are permitted for this repository. "
                        "If approved, set PII.ownership_emails to info in an explicit --policy file; "
                        "otherwise remove the address or use a placeholder."
                    )
                findings.append(finding)

    @staticmethod
    def _reported_value(match: re.Match) -> str:
        """Return the value a PII finding identifies for ``match``, before any display redaction.

        A finding's ``evidence_sha256`` and the waiver check both use this value,
        so a change to which part of a match a finding reports belongs here.
        """
        return match.group()

    # ------------------------------------------------------------------
    # LLM finding verification (delegates to skillevaluator.inference.FindingVerifier)
    # ------------------------------------------------------------------

    def _verify_findings_with_llm(
        self,
        result: ValidationResult,
        skill_path: Path,
    ) -> None:
        """Run LLM second-pass verification on security findings.

        Findings the LLM classifies as false_positive with high confidence
        are downgraded to INFO severity and annotated in metadata. Findings of
        checks in :data:`_LLM_VERIFY_EXEMPT_CHECKS` are not sent to the
        verifier: they keep their severity and get no verdict.
        """
        findings = [finding for finding in result.findings if finding.check_name not in _LLM_VERIFY_EXEMPT_CHECKS]
        if not findings:
            return

        from skillevaluator.inference import FindingVerifier
        from skillevaluator.inference.diagnostics import safe_llm_labels

        logger.info(f"Running LLM verification on {len(findings)} finding(s)")
        verifier = FindingVerifier()
        verdicts = verifier.verify(findings, skill_path)

        if not verdicts:
            result.mark_scan_incomplete("llm-verification")
            cause = f": {verifier.last_failure}" if verifier.last_failure else ""
            result.add_message(f"LLM finding verification was skipped (no verdicts returned{cause})")
            return

        # The provider and model that gave the verdicts: the configured model,
        # or a server-side fallback model that answered in its place.
        model_labels: dict[str, str] = {}
        if verifier.model_used is not None:
            provider, model = safe_llm_labels(verifier.provider, verifier.model_used)
            model_labels = {"llm_provider": provider, "llm_model_used": model}

        suppressed = 0
        verified = 0
        for idx, finding in enumerate(findings):
            verdict_data = verdicts.get(idx)
            if not isinstance(verdict_data, dict) or not verdict_data:
                continue

            verdict = verdict_data.get("verdict")
            confidence = verdict_data.get("confidence")
            # The prompt asks for "rationale"; replies that still say "reasoning" count too.
            reasoning = verdict_data.get("rationale", verdict_data.get("reasoning"))
            if (
                not isinstance(verdict, str)
                or verdict not in _LLM_VERDICTS
                or not isinstance(confidence, str)
                or confidence not in _LLM_CONFIDENCE_LEVELS
                or not isinstance(reasoning, str)
            ):
                continue
            verified += 1

            finding.metadata["llm_verdict"] = verdict
            finding.metadata["llm_confidence"] = confidence
            finding.metadata["llm_reasoning"] = reasoning
            finding.metadata.update(model_labels)

            if finding_verification_policy.should_downgrade(verdict, confidence):
                finding.severity = finding_verification_policy.DOWNGRADE_SEVERITY
                finding.metadata["downgraded"] = True
                finding.metadata["confidence"] = "low"
                suppressed += 1

        if suppressed:
            result.recalculate_from_findings()
            target_severity = finding_verification_policy.DOWNGRADE_SEVERITY.value.upper()
            verdict_label = finding_verification_policy.DOWNGRADE_VERDICT.replace("_", " ")
            result.add_message(
                f"LLM verification downgraded {suppressed} finding(s) to {target_severity} ({verdict_label}s)"
            )

        unverified = len(findings) - verified
        if unverified:
            result.mark_scan_incomplete("llm-verification")
            noun = "finding" if unverified == 1 else "findings"
            verb = "was" if unverified == 1 else "were"
            cause = f": {verifier.last_failure}" if verifier.last_failure else ""
            result.add_message(
                f"LLM verification returned verdicts for {verified} of {len(findings)} findings; "
                f"{unverified} {noun} {verb} not verified{cause}"
            )
        elif not suppressed:
            confidence_label = finding_verification_policy.DOWNGRADE_CONFIDENCE
            verdict_label = finding_verification_policy.DOWNGRADE_VERDICT.replace("_", " ")
            result.add_message(
                f"LLM verification reviewed all findings; no {confidence_label}-confidence {verdict_label}s identified"
            )

    # ------------------------------------------------------------------
    # File utilities
    # ------------------------------------------------------------------

    def _get_scannable_files(self, skill_path: Path) -> list[Path]:
        """Collect files with scannable extensions from path.

        Skips evaluation-artifact directories (``evals/``, ``results/``,
        ``versions/`` and the dot-prefixed variants) at any depth via
        :func:`iter_scannable_files` -- those trees contain LLM agent
        transcripts and JSON score files whose random digit sequences
        produce false-positive PII matches.
        """
        return iter_scannable_files(skill_path, SCANNABLE_EXTENSIONS)
