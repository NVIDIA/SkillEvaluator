# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Secrets Detection Validator using Gitleaks.

Detects hardcoded secrets, API keys, tokens, and credentials using
entropy-based detection and 200+ pre-built patterns.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

from skillevaluator.constants import SCAN_EXCLUDED_DIRS
from skillevaluator.utils.tool_runner import (
    GITLEAKS_FINDINGS_EXIT_CODE,
    Severity,
    Tools,
    create_temp_config,
    parse_json_output,
)
from skillevaluator.validators.base import Finding, ValidationResult, ValidatorBase


class SecretsValidator(ValidatorBase):
    """Scan for hardcoded secrets using Gitleaks.

    Provides entropy-based detection, 200+ pre-built patterns,
    optional git history scanning, and low false-positive rate
    via configurable allowlists.
    """

    # Map rule tags/IDs to severity levels
    _SEVERITY_KEYWORDS: ClassVar[dict[str, Severity]] = {
        "key": Severity.CRITICAL,
        "secret": Severity.CRITICAL,
        "token": Severity.CRITICAL,
        "password": Severity.CRITICAL,
        "credential": Severity.CRITICAL,
        "private-key": Severity.CRITICAL,
        "api": Severity.HIGH,
        "auth": Severity.HIGH,
        "generic": Severity.MEDIUM,
    }

    # Public NVIDIA API key detection rules
    _NVIDIA_RULES = """
[[rules]]
id = "nvidia-ngc-api-key"
description = "NVIDIA NGC API Key"
regex = '''\\$oauthtoken:\\S{84}'''
tags = ["nvidia", "ngc", "api-key"]

[[rules]]
id = "nvidia-api-key"
description = "NVIDIA API Key"
regex = '''nvapi-[A-Za-z0-9_-]{40,}'''
tags = ["nvidia", "api-key"]
"""

    def __init__(self, scan_git_history: bool = False):
        """Initialize secrets validator.

        Args:
            scan_git_history: Scan git history for secrets (slower but thorough)
        """
        self.scan_git_history = scan_git_history

    @property
    def name(self) -> str:
        return "Secrets Detection"

    @property
    def description(self) -> str:
        return "Detect hardcoded secrets, API keys, and credentials using Gitleaks"

    def validate(self, skill_path: Path) -> ValidationResult:
        """Run secrets detection on skill(s) at path."""
        return self._validate_folder_or_skill(
            skill_path,
            self._validate_single_skill,
            action_description="Scanning for secrets in",
        )

    def _validate_single_skill(self, skill_path: Path) -> ValidationResult:
        """Run Gitleaks scan on a single skill directory."""
        result = ValidationResult()

        if not Tools.gitleaks.is_available:
            result.add_warning(
                f"gitleaks not installed - secrets scanning skipped. {Tools.gitleaks.get_install_hint()}"
            )
            result.mark_scan_incomplete("gitleaks")
            return result

        config_path = self._create_gitleaks_config()
        try:
            tool_result = Tools.gitleaks.run(
                [
                    "detect",
                    "--source",
                    str(skill_path.resolve()),
                    "--report-format",
                    "json",
                    "--report-path",
                    "-",
                    "--exit-code",
                    str(GITLEAKS_FINDINGS_EXIT_CODE),
                    "--config",
                    str(config_path),
                    *([] if self.scan_git_history else ["--no-git"]),
                ],
                timeout=120,
            )

            if tool_result.error_message:
                result.add_warning(tool_result.error_message)
                result.mark_scan_incomplete("gitleaks")
                return result

            # The custom findings code disambiguates findings from Gitleaks'
            # default exit 1, which is also used for operational failures.
            if tool_result.exit_code == 0:
                if not tool_result.stdout.strip():
                    result.add_message("No secrets detected by Gitleaks")
                else:
                    findings = parse_json_output(tool_result.stdout)
                    if findings == []:
                        result.add_message("No secrets detected by Gitleaks")
                    elif isinstance(findings, list) and findings and all(isinstance(item, dict) for item in findings):
                        self._process_findings(findings, result)
                        result.add_error(
                            "Gitleaks returned findings with clean exit code 0; scanner status is inconsistent"
                        )
                        result.mark_scan_incomplete("gitleaks")
                    else:
                        result.add_error("Gitleaks clean exit did not return a valid empty JSON report")
                        result.mark_scan_incomplete("gitleaks")
            elif tool_result.exit_code == GITLEAKS_FINDINGS_EXIT_CODE:
                findings = parse_json_output(tool_result.stdout)
                if isinstance(findings, list) and findings and all(isinstance(item, dict) for item in findings):
                    self._process_findings(findings, result)
                else:
                    result.add_error("Gitleaks findings exit did not return a valid nonempty JSON report")
                    result.mark_scan_incomplete("gitleaks")
            else:
                error_msg = (
                    tool_result.stderr.strip()[:200] if tool_result.stderr.strip() else "scanner output redacted"
                )
                result.add_error(f"Gitleaks error (exit code {tool_result.exit_code}): {error_msg}")
                result.mark_scan_incomplete("gitleaks")

        finally:
            config_path.unlink(missing_ok=True)

        return result

    def _create_gitleaks_config(self) -> Path:
        """Create Gitleaks config with NVIDIA rules and allowlists.

        Tier 1 artifact dirs (evals/, results/, versions/, ...) hold Tier 3
        output snapshots that routinely carry harvested test credentials;
        gitleaks has to skip them via config since it takes no exclude flag.
        """
        artifact_paths = "\n".join(f"    '''(^|/){re.escape(name)}(/|$)'''," for name in sorted(SCAN_EXCLUDED_DIRS))
        config = f"""
[extend]
useDefault = true

[allowlist]
description = "SkillEvaluator allowlist"
paths = [
    '''(.*)?test(.*)?''',
    '''(.*)?example(.*)?''',
    '''(.*)?fixture(.*)?''',
    '''(.*)?mock(.*)?''',
{artifact_paths}
]
regexes = [
    '''YOUR_.*_HERE''',
    '''INSERT_.*_HERE''',
    '''PLACEHOLDER''',
    '''example\\.com''',
    '''test\\.nvidia\\.com''',
    '''dummy''',
    '''fake''',
]

"""
        return create_temp_config(config + self._NVIDIA_RULES, suffix=".toml")

    def _process_findings(self, findings: list[dict], result: ValidationResult) -> None:
        """Report an already validated, nonempty Gitleaks findings list."""
        result.add_message(f"Gitleaks found {len(findings)} potential secret(s)")

        for finding in findings:
            rule_id = finding.get("RuleID", "unknown")
            description = finding.get("Description", "Secret detected")
            file_path = finding.get("File", "unknown")
            line = finding.get("StartLine", 0)

            severity = self._determine_severity(rule_id, finding.get("Tags", []))
            message = f"{description} in {file_path}:{line} [{rule_id}]"

            result.add_structured_finding(
                Finding(
                    category="SECRET",
                    severity=severity,
                    check_name=rule_id,
                    message=message,
                    file_path=file_path,
                    line_number=line,
                ),
                is_error=severity.is_error(),
            )

    def _determine_severity(self, rule_id: str, tags: list[str]) -> Severity:
        """Determine severity from rule ID and tags."""
        # Check tags first (more specific)
        for tag in tags:
            tag_lower = tag.lower()
            for keyword, severity in self._SEVERITY_KEYWORDS.items():
                if keyword in tag_lower:
                    return severity

        # Fall back to rule ID
        rule_lower = rule_id.lower()
        for keyword, severity in self._SEVERITY_KEYWORDS.items():
            if keyword in rule_lower:
                return severity

        return Severity.HIGH  # Default for unknown secrets
