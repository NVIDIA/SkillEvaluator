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

import contextlib
import getpass
import io
import math
import os
import re
import shutil
import tempfile
import tokenize
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

import yaml
from yaml.events import ScalarEvent

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
from skillevaluator.provider_config import ProviderConfigurationError, resolve_llm_provider
from skillevaluator.spdx import is_spdx_only_html_comment
from skillevaluator.utils.tool_runner import Tools, parse_json_output
from skillevaluator.validators.base import (
    Finding,
    Severity,
    ValidationResult,
    ValidatorBase,
    iter_scannable_files,
)

logger = get_logger(__name__)

_AUTHOR_IDENTITY_RE = re.compile(r"^\S[^<>\n]* <(?P<email>[^<>@\s]+@[^<>\s]+)>$")
_SKILLSPECTOR_POLICY_EXIT_CODES = frozenset({0, 1})
_SKILLSPECTOR_STATUSLESS_COMPLETENESS_VERSION = (2, 9, 6)
_SKILLSPECTOR_COMPLETENESS_SCHEMA_VERSION = (2, 10, 0)
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
_SKILLSPECTOR_SCAN_EXCLUDED_DIRS = SCAN_EXCLUDED_DIRS - {"__pycache__"}
_SKILLSPECTOR_PROVIDER_MAP = {
    "anthropic": "anthropic",
    "bedrock": "bedrock",
    "nv_build": "openai",
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
    if suffix in {".md", ".markdown"}:
        return _markdown_comment_line_numbers(lines)
    return _leading_hash_or_slash_comment_lines(lines)


def _skillspector_llm_stderr_failed(stderr: str) -> bool:
    """Detect SkillSpector LLM failures hidden behind exit 0 and clean JSON.

    SkillSpector 2.3.7 can catch provider/auth errors, log them to stderr, and
    still return a static-looking report with ``llm_available=true``. For an
    explicitly requested LLM stage, those warnings mean the evidence is
    incomplete regardless of the process exit code or JSON metadata.
    """
    normalized = stderr.casefold()
    return any(marker in normalized for marker in _SKILLSPECTOR_LLM_FAILURE_MARKERS)


def _copy_selected_environment(environ: Mapping[str, str], names: Iterable[str]) -> dict[str, str]:
    return {name: environ[name] for name in names if environ.get(name, "").strip()}


def _skillspector_process_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    return _copy_selected_environment(source, _SKILLSPECTOR_PROCESS_ENV_NAMES)


def _skillspector_child_env() -> dict[str, str]:
    """Map public provider settings into an invocation-scoped SkillSpector environment."""
    source = os.environ
    child_env = _skillspector_process_env(source)
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
            return child_env
        if skillspector_provider != "nv_build":
            # Unsupported/private providers fail closed through the generic
            # public path without receiving any provider credential.
            child_env["SKILLSPECTOR_PROVIDER"] = "openai"
            if skillspector_model:
                child_env["SKILLSPECTOR_MODEL"] = skillspector_model
            return child_env

    try:
        resolution_env = dict(source)
        if skillspector_provider == "nv_build":
            resolution_env["SKILL_EVAL_LLM_PROVIDER"] = "nv_build"
            if skillspector_model:
                resolution_env["SKILL_EVAL_LLM_MODEL"] = skillspector_model
        provider = resolve_llm_provider(resolution_env)
    except ProviderConfigurationError:
        return child_env

    mapped_provider = _SKILLSPECTOR_PROVIDER_MAP.get(provider.provider)
    if mapped_provider is None:
        return child_env

    child_env["SKILLSPECTOR_PROVIDER"] = mapped_provider
    child_env["SKILLSPECTOR_MODEL"] = skillspector_model or provider.model

    if provider.provider in {"nv_build", "openai-compatible"}:
        if provider.api_key:
            child_env["OPENAI_API_KEY"] = provider.api_key
        if provider.base_url:
            child_env["OPENAI_BASE_URL"] = provider.base_url
    else:
        child_env.update(provider.child_environment())
        if provider.provider == "bedrock":
            child_env.update(_copy_selected_environment(source, _SKILLSPECTOR_AWS_ENV_NAMES))

    return child_env


def _tree_contains_artifact_dirs(root: Path) -> bool:
    """Return True when the SkillSpector scan tree needs staging."""
    return any(
        any(d in _SKILLSPECTOR_SCAN_EXCLUDED_DIRS for d in dirnames)
        for _dirpath, dirnames, _filenames in os.walk(root)
    )


def _ignore_artifact_dirs(dirpath: str, names: list[str]) -> set[str]:
    """``shutil.copytree`` ignore hook dropping artifact directories only."""
    return {n for n in names if n in _SKILLSPECTOR_SCAN_EXCLUDED_DIRS and Path(dirpath, n).is_dir()}


def _rewrite_path_prefix(value, old: str, new: str):
    """Recursively rewrite ``old`` path prefixes in a parsed JSON structure."""
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_rewrite_path_prefix(v, old, new) for v in value]
    if isinstance(value, dict):
        return {k: _rewrite_path_prefix(v, old, new) for k, v in value.items()}
    return value


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
) -> tuple[str, str | int]:
    """Return the producer's finding identity used for report compaction."""
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

    def __init__(
        self,
        use_llm: bool = False,
        verify_llm: bool = False,
        submitter_usernames: Iterable[str] | None = None,
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
        return self._validate_folder_or_skill(
            skill_path,
            self._validate_single_skill,
            action_description="Scanning",
        )

    def validate_security_only(self, skill_path: Path) -> ValidationResult:
        """Run only skillspector security scanning (no PII detection)."""
        result = self._validate_folder_or_skill(
            skill_path,
            self._run_skillspector,
            action_description="Security scanning",
        )
        if self.verify_llm and result.findings:
            self._verify_findings_with_llm(result, skill_path)
        return result

    def validate_pii_only(self, skill_path: Path) -> ValidationResult:
        """Run only PII detection (no skillspector security scan)."""
        result = self._validate_folder_or_skill(
            skill_path,
            self._scan_for_pii,
            action_description="PII scanning",
        )
        if self.verify_llm and result.findings:
            self._verify_findings_with_llm(result, skill_path)
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
        """
        result = ValidationResult()

        if not Tools.skillspector.is_available:
            result.add_warning(
                f"skillspector not installed - skipping security scan. {Tools.skillspector.get_install_hint()}"
            )
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
            # The deterministic stage is authoritative and always runs, even
            # when LLM enrichment was requested. Enrichment is additive and
            # cannot erase static findings.
            result = self._run_skillspector_once(
                scan_root,
                original_root=original_root if staged is not None else None,
                use_llm=False,
            )
            if self.use_llm and not result.is_incomplete:
                enrichment = self._run_skillspector_once(
                    scan_root,
                    original_root=original_root if staged is not None else None,
                    use_llm=True,
                )
                result.merge(enrichment)
                if not enrichment.is_incomplete:
                    self._deduplicate_skillspector_findings(result)
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
    ) -> ValidationResult:
        """Run one SkillSpector stage and validate its process/report contract."""
        result = ValidationResult()
        stage_name = "skillspector-llm" if use_llm else "skillspector"
        args = ["scan", str(scan_root), "--format", "json"]
        if not use_llm:
            args.append("--no-llm")

        logger.info("Running %s on %s", stage_name, scan_root)
        child_env = _skillspector_child_env() if use_llm else _skillspector_process_env()
        tool_result = Tools.skillspector.run(args, timeout=300, env=child_env, replace_env=True)

        if tool_result.exit_code not in _SKILLSPECTOR_POLICY_EXIT_CODES:
            detail = tool_result.error_message or (
                f"{stage_name} failed with unexpected exit code {tool_result.exit_code}: "
                "scanner diagnostics were redacted"
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

        try:
            data = parse_json_output(tool_result.stdout)
        except (RecursionError, ValueError):
            data = None
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

        self._process_skillspector_cli_result(data, result)
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
            result.findings = unique
            result.recalculate_from_findings()

    @staticmethod
    def _validate_skillspector_report(
        data: dict,
        process_exit_code: int,
        use_llm: bool,
        stage_name: str,
        result: ValidationResult,
    ) -> bool:
        """Return whether JSON is a trustworthy SkillSpector findings report."""
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
            skillspector_version == _SKILLSPECTOR_STATUSLESS_COMPLETENESS_VERSION
        )
        uses_versioned_completeness = uses_completeness_schema or uses_statusless_completeness_schema
        completeness_contract = "2.10+" if uses_completeness_schema else "2.9.6"

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
                    if not isinstance(completeness_status, str) or completeness_status not in {
                        "complete",
                        "partial",
                        "failed",
                    }:
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
                        "skillspector 2.9.6 JSON field 'analysis_completeness.status' must be absent; "
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
                if (is_complete or uses_statusless_completeness_schema) and counts["total_components"]:
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

                if uses_completeness_schema and is_complete is not (completeness_status == "complete"):
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
                            "skillspector 2.9.6 JSON field 'analysis_completeness' does not describe "
                            "a fully covered scan; security scan did not complete"
                        )
                        return False
                elif is_complete:
                    if (
                        counts["partially_inspected_files"] != 0
                        or counts["entirely_uninspected_files"] != 0
                        or ledger_exceptions
                        or limitations
                    ):
                        result.add_error(
                            "skillspector JSON field 'analysis_completeness' details contradict complete analysis; "
                            "security scan did not complete"
                        )
                        return False
                else:
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
                    result.add_error(
                        "skillspector JSON field 'analysis_completeness' reports incomplete analysis "
                        f"(status '{completeness_status}'); security scan did not complete"
                    )
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
        if result.is_incomplete and severity == "LOW":
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
        if not use_llm and (
            metadata.get("llm_requested") not in {None, False}
            or metadata.get("llm_available") not in {None, False}
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
        if normalized_suppressed_count:
            result.add_error(
                "skillspector reported unexpected suppressed findings without a requested baseline; "
                "security scan did not complete"
            )
            return False
        minimum_score = SecurityValidator._minimum_skillspector_risk_score(
            issues,
            normalized_components,
            uses_report_identities=uses_completeness_schema,
            findings_after_filtering=findings_after_filtering,
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

        return True

    @staticmethod
    def _minimum_skillspector_risk_score(
        issues: list[dict],
        components: list[dict],
        *,
        uses_report_identities: bool,
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
        )
        if uses_report_identities:
            all_visible_issues = [*issues, *(removed_issues or [])]
            all_deduplicated, all_ambiguous_identities = (
                SecurityValidator._deduplicate_skillspector_issues_for_scoring(
                    all_visible_issues,
                    uses_report_identities=True,
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
    def _deduplicate_skillspector_issues_for_scoring(
        issues: list[dict],
        *,
        uses_report_identities: bool,
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
    ) -> None:
        """Convert a validated SkillSpector JSON report into ValidationResult entries."""
        self._store_skillspector_metadata(data, result)

        issues = data.get("issues", [])
        scanned_issues = []
        removed_issues = []
        skipped_generated = 0
        skipped_spdx_comments = 0
        has_critical_or_high = False
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            if self._is_generated_artifact_issue(issue):
                removed_issues.append(issue)
                skipped_generated += 1
                continue
            if self._is_spdx_only_hidden_instruction(issue):
                removed_issues.append(issue)
                skipped_spdx_comments += 1
                continue
            scanned_issues.append(issue)
            finding, is_error = self._convert_skillspector_issue(issue)
            if is_error:
                has_critical_or_high = True
            result.add_structured_finding(finding, is_error=is_error)

        if skipped_generated:
            result.add_message(f"skillspector ignored {skipped_generated} generated artifact issue(s)")
        if skipped_spdx_comments:
            result.add_message(f"skillspector ignored {skipped_spdx_comments} SPDX-only HTML comment issue(s)")

        reported_score = data["risk_assessment"]["score"]
        report_components = data.get("components") if isinstance(data.get("components"), list) else []
        report_metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        raw_version = report_metadata.get("skillspector_version")
        uses_report_identities = (
            isinstance(raw_version, str)
            and SEMVER_RE.fullmatch(raw_version) is not None
            and tuple(int(part) for part in raw_version.split("."))
            >= _SKILLSPECTOR_COMPLETENESS_SCHEMA_VERSION
        )
        analysis_completeness = data.get("analysis_completeness")
        findings_after_filtering = (
            analysis_completeness.get("findings_after_filtering")
            if isinstance(analysis_completeness, dict)
            else None
        )
        suppressed = data.get("suppressed")
        serialized_findings = len(issues) + (len(suppressed) if isinstance(suppressed, list) else 0)
        effective_score = (
            self._minimum_skillspector_risk_score(
                scanned_issues,
                report_components,
                uses_report_identities=uses_report_identities,
                findings_after_filtering=findings_after_filtering,
                reported_score=reported_score,
                removed_issues=removed_issues,
                all_findings_serialized=findings_after_filtering == serialized_findings,
            )
            if skipped_generated or skipped_spdx_comments
            else reported_score
        )
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

        if not result.is_incomplete:
            self._summarize_skillspector_results(scanned_issues, has_critical_or_high, result)

    @staticmethod
    def _is_generated_artifact_issue(issue: dict) -> bool:
        """Return True when a skillspector issue points at generated output."""
        if issue.get("id") == "SC8":
            return False
        file_path, _line_number = SecurityValidator._parse_issue_location(issue)
        path = Path(file_path)
        if path.name.lower() in SCAN_EXCLUDED_FILES:
            return True
        return any(part in SCAN_EXCLUDED_DIRS for part in path.parts)

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
    def _store_skillspector_metadata(data: dict, result: ValidationResult) -> None:
        """Store the validated SkillSpector report metadata on the result."""
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
            }
        )

        if suppressed_count:
            result.add_message(f"skillspector suppressed {suppressed_count} audited finding(s)")

        if sp_metadata.get("llm_requested") and not sp_metadata.get("llm_available"):
            result.add_warning(
                "LLM analysis was requested but not available; provider or model diagnostics were redacted"
            )

        if score > 50:
            result.add_message(f"skillspector risk score: {score}/100 ({severity})")

    @staticmethod
    def _convert_skillspector_issue(issue: dict) -> tuple[Finding, bool]:
        """Convert a single skillspector issue dict into a Finding.

        Returns (Finding, is_error) where is_error is True for CRITICAL/HIGH.
        """
        g = _issue_field(issue)
        issue_sev = str(g("severity", "UNKNOWN")).upper()
        explanation = g("explanation")
        remediation = g("remediation")
        suggestion = f"{explanation} {remediation}".strip() or None
        code_snippet = g("code_snippet")
        file_path, line_number = SecurityValidator._parse_issue_location(issue)

        finding = Finding(
            category="SECURITY",
            severity=issue_sev,
            check_name=f"{g('pattern', 'Unknown')} ({g('id', '?')})",
            message=SecurityValidator._build_issue_message(
                g("category"), g("finding"), explanation, g("pattern", "Unknown")
            ),
            file_path=file_path,
            line_number=line_number,
            line_content=code_snippet[:200] if code_snippet else None,
            suggestion=suggestion,
            metadata=SecurityValidator._build_issue_metadata(issue),
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
    def _summarize_skillspector_results(issues: list, has_critical_or_high: bool, result: ValidationResult) -> None:
        """Add a summary success/message entry for the skillspector scan."""
        if not issues:
            result.add_success(
                check_name="skillspector",
                message="No security vulnerabilities detected (secrets, API keys, credentials)",
            )
        elif has_critical_or_high:
            result.add_message(f"skillspector found {len(issues)} issue(s) including critical/high severity")
        else:
            result.add_success(
                check_name="skillspector",
                message=f"Security scan completed - {len(issues)} advisory finding(s) (no critical/high issues)",
                issue_count=len(issues),
            )

    def _scan_for_pii(self, skill_path: Path) -> ValidationResult:
        """Scan files for PII using regex patterns (emails, paths, SSNs, etc.)."""
        result = ValidationResult()
        files = self._get_scannable_files(skill_path)

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

            for finding_data in self._scan_file_for_pii(file_path, protected_usernames):
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

        confidence_rank = {"low": 0, "medium": 1, "high": 2}
        for group in groups.values():
            first = group["first"]
            value = first.get("matched_value")
            occurrences = group["occurrences"]
            severity = first["severity"].upper()

            message = f"{first['description']}: {value}" if value else first["description"]
            metadata: dict = {"confidence": max(group["confidences"], key=lambda c: confidence_rank.get(c, 1))}
            if value:
                metadata["matched_value"] = value
                metadata["occurrence_count"] = len(occurrences)
                metadata["occurrences"] = [{"file": f, "line": line} for f, line in occurrences]
            if len(occurrences) > 1:
                message += f" — {len(occurrences)} occurrences ({self._format_occurrences(occurrences)})"

            finding = Finding(
                category="PII",
                severity=severity,
                check_name=first["category"],
                message=message,
                file_path=group["first_file"],
                line_number=first["line"],
                line_content=first.get("line_content"),
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

    @staticmethod
    def _is_near_zero_gps(line: str) -> bool:
        """Check if a GPS match contains only near-zero coordinates (Null Island)."""
        return bool(SecurityValidator._GPS_ZERO_PATTERN.search(line))

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
        could be a valid credit card number.
        """
        total = 0
        for i, ch in enumerate(reversed(digits)):
            n = int(ch)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        return total % 10 == 0

    def _scan_file_for_pii(self, file_path: Path, protected_usernames: set[str] | None = None) -> list[dict]:
        """Scan a single file for PII patterns, yielding findings with full context.

        ``protected_usernames`` is the set of author/submitter identities used by
        the home-path check; when omitted it is resolved from the file's parent
        directory so the method stays usable standalone.
        """
        if protected_usernames is None:
            protected_usernames = self._protected_home_usernames(file_path.parent)

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return []

        lines = content.split("\n")
        author_emails = self._frontmatter_author_emails(file_path, lines)
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
            )
        return findings

    def _frontmatter_author_emails(self, file_path: Path, lines: list[str]) -> dict[int, str]:
        """Map valid frontmatter author lines to the public contributor email."""
        if file_path.name not in SKILL_MANIFEST_VARIANTS or not lines or lines[0].strip() != "---":
            return {}
        try:
            frontmatter_end = next(index for index, line in enumerate(lines[1:], 1) if line.strip() == "---")
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
    ) -> None:
        """Check all lines against a single compiled PII pattern."""
        protected_usernames = protected_usernames or set()
        author_emails = author_emails or {}
        comment_lines = comment_lines if comment_lines is not None else _comment_line_numbers(file_path, lines)
        for line_num, line in enumerate(lines, 1):
            if line_num in comment_lines:
                continue

            scan_line = line
            if category == "emails" and (author_email := author_emails.get(line_num)):
                scan_line = re.sub(re.escape(author_email), "author@example.com", line, count=1, flags=re.IGNORECASE)

            matches = list(regex.finditer(scan_line))
            if not matches or any(exc in line for exc in exceptions):
                continue
            if category == "ip_addresses":
                matches = [match for match in matches if not self._is_version_literal(match, line)]
                if not matches:
                    continue
            match = matches[0]

            # /home/<root>/ is PII only when <root> is the author/submitter
            # username; unrelated roots are skipped.
            if category == "home_paths" and not self._is_personal_home_path(match, protected_usernames):
                continue

            if category == "credit_cards":
                digits = re.sub(r"[\s-]", "", match.group())
                if not self._passes_luhn(digits):
                    continue

            confidence = self._estimate_confidence(category, line)
            if category == "gps_coordinates" and self._is_near_zero_gps(line):
                confidence = "low"

            findings.append(
                {
                    "file": file_path.name,
                    "line": line_num,
                    "line_content": line,
                    "severity": pattern_def.get("severity", "medium"),
                    "description": pattern_def.get("description", category),
                    "suggestion": pattern_def.get("suggestion"),
                    "category": category,
                    "confidence": confidence,
                    "matched_value": match.group(),
                }
            )

    # ------------------------------------------------------------------
    # LLM finding verification (delegates to skillevaluator.inference.FindingVerifier)
    # ------------------------------------------------------------------

    def _verify_findings_with_llm(
        self,
        result: ValidationResult,
        skill_path: Path,
    ) -> None:
        """Run LLM second-pass verification on all findings.

        Findings the LLM classifies as false_positive with high confidence
        are downgraded to INFO severity and annotated in metadata.
        """
        if not result.findings:
            return

        from skillevaluator.inference import FindingVerifier

        logger.info(f"Running LLM verification on {len(result.findings)} finding(s)")
        verifier = FindingVerifier()
        verdicts = verifier.verify(result.findings, skill_path)

        if not verdicts:
            result.mark_scan_incomplete("llm-verification")
            result.add_message("LLM finding verification was skipped (no verdicts returned)")
            return

        suppressed = 0
        verified = 0
        for idx, finding in enumerate(result.findings):
            verdict_data = verdicts.get(idx)
            if not isinstance(verdict_data, dict) or not verdict_data:
                continue

            verdict = verdict_data.get("verdict")
            confidence = verdict_data.get("confidence")
            reasoning = verdict_data.get("reasoning")
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

            if verdict == "false_positive" and confidence == "high":
                finding.severity = Severity.INFO
                finding.metadata["downgraded"] = True
                finding.metadata["confidence"] = "low"
                suppressed += 1

        if suppressed:
            result.recalculate_from_findings()
            result.add_message(f"LLM verification downgraded {suppressed} finding(s) to INFO (false positives)")

        unverified = len(result.findings) - verified
        if unverified:
            result.mark_scan_incomplete("llm-verification")
            noun = "finding" if unverified == 1 else "findings"
            verb = "was" if unverified == 1 else "were"
            result.add_message(
                f"LLM verification returned verdicts for {verified} of {len(result.findings)} findings; "
                f"{unverified} {noun} {verb} not verified"
            )
        elif not suppressed:
            result.add_message("LLM verification reviewed all findings; no high-confidence false positives identified")

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
