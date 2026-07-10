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
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

from skillevaluator.config import load_pii_patterns
from skillevaluator.constants import (
    HOME_PATH_SUBMITTER_ENV_VARS,
    SCAN_EXCLUDED_DIRS,
    SCAN_EXCLUDED_FILES,
    SCANNABLE_EXTENSIONS,
    SKILL_MANIFEST_VARIANTS,
)
from skillevaluator.logging_config import get_logger
from skillevaluator.provider_config import ProviderConfigurationError, resolve_llm_provider
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
    """Return True when any :data:`SCAN_EXCLUDED_DIRS` dir exists under root."""
    return any(any(d in SCAN_EXCLUDED_DIRS for d in dirnames) for _dirpath, dirnames, _filenames in os.walk(root))


def _ignore_artifact_dirs(dirpath: str, names: list[str]) -> set[str]:
    """``shutil.copytree`` ignore hook dropping artifact directories only."""
    return {n for n in names if n in SCAN_EXCLUDED_DIRS and Path(dirpath, n).is_dir()}


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
        ``metadata.author`` (Skill Evaluator nests it under ``metadata``). Returns the
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
        when the skill carries Tier 1 artifact dirs (``evals/results/``
        snapshots can reach hundreds of MB after Tier 3 runs) the scan runs
        on a temp copy of the skill without those dirs, and reported paths
        are mapped back onto the original location.
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

        if not tool_result.success or tool_result.error_message:
            result.add_error(tool_result.error_message or f"{stage_name} failed to execute")
            result.mark_scan_incomplete(stage_name)
            return result

        if tool_result.exit_code not in _SKILLSPECTOR_POLICY_EXIT_CODES:
            result.add_error(
                f"{stage_name} failed with unexpected exit code {tool_result.exit_code}: "
                "scanner diagnostics were redacted"
            )
            result.mark_scan_incomplete(stage_name)
            return result

        if use_llm and _skillspector_llm_stderr_failed(tool_result.stderr):
            result.add_error(
                "skillspector-llm reported failed LLM analysis; provider or model diagnostics were redacted"
            )
            result.mark_scan_incomplete(stage_name)
            return result

        data = parse_json_output(tool_result.stdout)
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
            data = _rewrite_path_prefix(data, str(scan_root), str(original_root))

        if not self._validate_skillspector_report(data, result):
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
    def _validate_skillspector_report(data: dict, result: ValidationResult) -> bool:
        """Return whether JSON is a trustworthy SkillSpector findings report."""
        if data.get("error"):
            result.add_error("skillspector reported an error; security scan did not complete")
            return False
        if data.get("errors"):
            result.add_error("skillspector reported errors; security scan did not complete")
            return False
        if data.get("failure") or data.get("failed") is True:
            result.add_error("skillspector reported a failure; security scan did not complete")
            return False
        if data.get("success") is False:
            result.add_error("skillspector reported success=false; security scan did not complete")
            return False

        status = data.get("status")
        if isinstance(status, str) and status.strip().lower() in {"error", "failed", "failure"}:
            result.add_error(f"skillspector reported failure status '{status}'; security scan did not complete")
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

        if not isinstance(data.get("risk_assessment"), dict):
            result.add_error(
                "skillspector JSON report is missing required 'risk_assessment' object; security scan did not complete"
            )
            return False

        for field in ("skill", "metadata"):
            value = data.get(field)
            if value is not None and not isinstance(value, dict):
                result.add_error(f"skillspector JSON field '{field}' must be an object; security scan did not complete")
                return False
        components = data.get("components")
        if components is not None and not isinstance(components, list):
            result.add_error("skillspector JSON field 'components' must be a list; security scan did not complete")
            return False

        return True

    def _process_skillspector_cli_result(self, data: dict, result: ValidationResult) -> None:
        """Convert skillspector CLI JSON output into ValidationResult entries.

        Handles the skillspector 1.0 JSON schema which includes:
        - skill: {name, source, scanned_at}
        - risk_assessment: {score, severity, recommendation}
        - components: [{path, type, lines, executable, size_bytes}]
        - issues[]: {id, category, pattern, severity, confidence, location,
                     finding, explanation, remediation, code_snippet, intent}
        - metadata: {has_executable_scripts, skillspector_version}

        Applies post-processing to downgrade known false-positive patterns
        (e.g. trusted package installers, standard Docker commands).
        """
        self._store_skillspector_metadata(data, result)

        issues = data.get("issues", [])
        scanned_issues = []
        skipped_generated = 0
        has_critical_or_high = False
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            if self._is_generated_artifact_issue(issue):
                skipped_generated += 1
                continue
            scanned_issues.append(issue)
            finding, is_error = self._convert_skillspector_issue(issue)
            if is_error:
                has_critical_or_high = True
            result.add_structured_finding(finding, is_error=is_error)

        if skipped_generated:
            result.add_message(f"skillspector ignored {skipped_generated} generated artifact issue(s)")

        self._summarize_skillspector_results(scanned_issues, has_critical_or_high, result)

    @staticmethod
    def _is_generated_artifact_issue(issue: dict) -> bool:
        """Return True when a skillspector issue points at generated output."""
        file_path, _line_number = SecurityValidator._parse_issue_location(issue)
        path = Path(file_path)
        if path.name.lower() in SCAN_EXCLUDED_FILES:
            return True
        return any(part in SCAN_EXCLUDED_DIRS for part in path.parts)

    @staticmethod
    def _store_skillspector_metadata(data: dict, result: ValidationResult) -> None:
        """Extract and store top-level skillspector 1.0 metadata on the result."""
        skill_info = data.get("skill") or {}
        sp_metadata = data.get("metadata") or {}
        components = data.get("components") or []

        risk = data.get("risk_assessment") or {}
        score = risk.get("score", 0)
        severity = risk.get("severity", "UNKNOWN")

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
            }
        )

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

    @staticmethod
    def _is_near_zero_gps(line: str) -> bool:
        """Check if a GPS match contains only near-zero coordinates (Null Island)."""
        return bool(SecurityValidator._GPS_ZERO_PATTERN.search(line))

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
    ) -> None:
        """Check all lines against a single compiled PII pattern."""
        protected_usernames = protected_usernames or set()
        author_emails = author_emails or {}
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith(("#", "//")):
                continue

            scan_line = line
            if category == "emails" and (author_email := author_emails.get(line_num)):
                scan_line = re.sub(re.escape(author_email), "author@example.com", line, count=1, flags=re.IGNORECASE)

            match = regex.search(scan_line)
            if not match or any(exc in line for exc in exceptions):
                continue

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
