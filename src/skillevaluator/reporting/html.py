# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTML reporter for standalone web reports.

This reporter produces self-contained HTML files suitable for:
- Email attachments
- CI/CD artifacts
- Compliance audit trails
- Standalone report viewing

Features include:
- Professional MARSFlow-inspired styling
- Dark mode toggle
- JSON export functionality
- Navigation tabs for future Tier 2/3/4 support
- Filtering and collapsible sections
"""

from __future__ import annotations

import base64
import hashlib
import json
import pkgutil
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlparse

from jinja2 import BaseLoader, Environment

from skillevaluator import __version__
from skillevaluator.constants import TIER3_LIFT_FAIL_THRESHOLD, TIER3_LIFT_PASS_THRESHOLD
from skillevaluator.reporting.base import ReporterBase, is_advisory_agent_eval_skip, passes_required_gate
from skillevaluator.reporting.harbor_viewer import normalize_agent_eval_harbor_links

if TYPE_CHECKING:
    from skillevaluator.models import ValidationResult


_TIER2_VALIDATOR_MARKERS = ("similarity", "dedup", "context optimization")

# Tier 3 already enforces a 2 MiB canonical payload limit. HTML needs a
# separate bound because script-safe escaping (``<`` -> ``\u003c``), pretty
# diagnostics, and visible dataset fields can otherwise multiply that payload
# many times over. Large canonical payloads are embedded once as base64 while a
# bounded projection feeds the human-readable panels.
_TIER3_JSON_EMBED_MAX_BYTES = 512 * 1024
_TIER3_HTML_PREVIEW_TRIGGER_BYTES = 256 * 1024
_TIER3_HTML_PREVIEW_CHARS = 128 * 1024
_TIER3_HTML_PREVIEW_STRING_CHARS = 4 * 1024
_TIER3_HTML_PREVIEW_COLLECTION_ITEMS = 64
_TIER3_PREVIEW_MARKER = "... [HTML preview truncated; download the full Tier 3 payload]"


@dataclass
class _Tier3PreviewBudget:
    chars_remaining: int = _TIER3_HTML_PREVIEW_CHARS
    omitted_characters: int = 0
    omitted_items: int = 0


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _script_safe_json(value: object) -> str:
    """Serialize JSON for an HTML raw-text script element without expansion attacks."""
    return (
        _compact_json(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _canonical_tier3_embed(payload: dict[str, Any] | None) -> tuple[str, str]:
    """Return one safe canonical Tier 3 copy and its browser decoding mode."""
    if not payload:
        return "", "json"
    raw = _compact_json(payload)
    safe = _script_safe_json(payload)
    if len(safe.encode("utf-8")) <= _TIER3_JSON_EMBED_MAX_BYTES:
        return safe, "json"
    return base64.b64encode(raw.encode("utf-8")).decode("ascii"), "base64"


def _bounded_tier3_preview_value(value: Any, budget: _Tier3PreviewBudget) -> Any:
    if isinstance(value, str):
        allowed = min(_TIER3_HTML_PREVIEW_STRING_CHARS, max(0, budget.chars_remaining))
        if len(value) <= allowed:
            budget.chars_remaining -= len(value)
            return value
        budget.omitted_characters += len(value) - allowed
        budget.chars_remaining -= allowed
        if allowed <= len(_TIER3_PREVIEW_MARKER):
            return _TIER3_PREVIEW_MARKER[:allowed]
        return value[: allowed - len(_TIER3_PREVIEW_MARKER)] + _TIER3_PREVIEW_MARKER

    if isinstance(value, list):
        kept = min(len(value), _TIER3_HTML_PREVIEW_COLLECTION_ITEMS)
        budget.omitted_items += len(value) - kept
        bounded: list[Any] = []
        for item in value[:kept]:
            bounded.append(_bounded_tier3_preview_value(item, budget))
        return bounded

    if isinstance(value, dict):
        items = list(value.items())
        kept = min(len(items), _TIER3_HTML_PREVIEW_COLLECTION_ITEMS)
        budget.omitted_items += len(items) - kept
        bounded_dict: dict[Any, Any] = {}
        for key, item in items[:kept]:
            budget.chars_remaining = max(0, budget.chars_remaining - len(str(key)))
            bounded_dict[key] = _bounded_tier3_preview_value(item, budget)
        return bounded_dict

    return value


def _bounded_tier3_preview(payload: dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, int]]:
    """Return a presentation-only projection plus visible omission counts."""
    if not payload or len(_compact_json(payload).encode("utf-8")) <= _TIER3_HTML_PREVIEW_TRIGGER_BYTES:
        return payload, {}

    budget = _Tier3PreviewBudget()
    preview = _bounded_tier3_preview_value(payload, budget)
    notice = {
        key: value
        for key, value in {
            "characters": budget.omitted_characters,
            "items": budget.omitted_items,
        }.items()
        if value > 0
    }
    return preview, notice


def is_tier2_validator_name(validator_name: str | None) -> bool:
    """Return whether a validator name belongs to Tier 2 reporting."""
    normalized_name = " ".join((validator_name or "").casefold().replace("_", " ").replace("-", " ").split())
    return any(marker in normalized_name for marker in _TIER2_VALIDATOR_MARKERS)


def _related_paths(finding: object) -> list[str]:
    """Return distinct path-like string values carried in finding metadata."""
    metadata = finding.get("metadata", {}) if isinstance(finding, dict) else getattr(finding, "metadata", {})
    if not isinstance(metadata, dict):
        return []

    paths: list[str] = []
    for key, value in metadata.items():
        normalized_key = str(key).casefold()
        if not (normalized_key == "path" or normalized_key.startswith("path_") or normalized_key.endswith("_path")):
            continue
        if isinstance(value, str) and value and value not in paths:
            paths.append(value)
    return paths


class PackageLoader(BaseLoader):
    """Custom Jinja2 loader that loads templates from package resources."""

    def __init__(self, package: str, path: str) -> None:
        self.package = package
        self.path = path

    def get_source(self, _environment: Environment, template: str) -> tuple[str, str, callable]:
        """Load template source from package resources."""
        template_path = f"{self.path}/{template}"
        try:
            source = resources.files(self.package).joinpath(template_path).read_text(encoding="utf-8")
        except AttributeError as exc:
            encoded = pkgutil.get_data(self.package, template_path)
            if encoded is None:
                raise FileNotFoundError(template_path) from exc
            source = encoded.decode("utf-8")
        return source, template, lambda: True


class HTMLReporter(ReporterBase):
    """HTML report generator using Jinja2 templates."""

    ICONS: ClassVar[dict[str, str]] = {
        "checkmark": '<svg class="success-icon" viewBox="0 0 16 16" fill="currentColor"><path d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"/></svg>',
        "file": '<svg class="meta-icon" viewBox="0 0 16 16" fill="currentColor"><path d="M3.75 1.5a.25.25 0 00-.25.25v11.5c0 .138.112.25.25.25h8.5a.25.25 0 00.25-.25V4.664a.25.25 0 00-.073-.177l-2.914-2.914a.25.25 0 00-.177-.073H3.75z"/><path fill-rule="evenodd" d="M2 1.75C2 .784 2.784 0 3.75 0h5.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v8.586A1.75 1.75 0 0112.25 15h-8.5A1.75 1.75 0 012 13.25V1.75z"/></svg>',
        "search": '<svg class="meta-icon" viewBox="0 0 16 16" fill="currentColor"><path fill-rule="evenodd" d="M11.5 7a4.499 4.499 0 11-8.998 0A4.499 4.499 0 0111.5 7zm-.82 4.74a6 6 0 111.06-1.06l3.04 3.04a.75.75 0 11-1.06 1.06l-3.04-3.04z"/></svg>',
        "arrow_up": '<svg viewBox="0 0 16 16" fill="currentColor" width="20" height="20"><path fill-rule="evenodd" d="M8 12a.75.75 0 01-.75-.75V5.56L5.03 7.78a.75.75 0 01-1.06-1.06l3.5-3.5a.75.75 0 011.06 0l3.5 3.5a.75.75 0 01-1.06 1.06L8.75 5.56v5.69A.75.75 0 018 12z"/></svg>',
    }

    DEFAULT_TABS: ClassVar[list[dict[str, str]]] = [
        {"id": "tier1", "label": "Tier 1: Security and Static Validation"},
    ]

    def __init__(
        self,
        *,
        include_timestamp: bool = True,
        title: str | None = None,
        tabs: list[dict[str, str]] | None = None,
        target_path: str | None = None,
        content_label: str = "Skill",
        profile: str | None = None,
    ) -> None:
        self.include_timestamp = include_timestamp
        self.title = title or "Skill Evaluator Validation Report"
        self.tabs = list(tabs) if tabs is not None else list(self.DEFAULT_TABS)
        self._tabs_explicit = tabs is not None
        self.target_path = target_path
        self.content_label = content_label
        # Active validation profile (e.g. "internal", "external"). Surfaced
        # in the report header so reviewers can tell which audience the
        # validation gate was applied for.
        self.profile = profile
        self._env = self._create_environment()

    def _create_environment(self) -> Environment:
        loader = PackageLoader("skillevaluator.reporting", "templates")
        environment = Environment(loader=loader, autoescape=True)
        environment.filters["related_paths"] = _related_paths
        return environment

    @staticmethod
    def _infer_profile_from_results(results: list[ValidationResult]) -> str | None:
        """Read the active profile name out of result metadata.

        ``commands.validate._stamp_policy`` writes ``result.metadata['policy']``
        for every result; we read it back here so reporters constructed
        without an explicit ``profile=`` argument still surface the profile.
        Returns ``None`` if no result carries policy metadata.
        """
        for r in results:
            policy_meta = (r.metadata or {}).get("policy") if isinstance(r.metadata, dict) else None
            if isinstance(policy_meta, dict) and policy_meta.get("profile"):
                return str(policy_meta["profile"])
        return None

    @property
    def name(self) -> str:
        return "html"

    @property
    def description(self) -> str:
        return "Standalone HTML reports for archiving/sharing"

    def _is_single_skill_mode(self, results: list[ValidationResult]) -> str | None:
        """Detect if results are from a single-skill run (not folder-of-skills).

        In folder mode, findings have ``[skill-name] path`` prefixes from
        ``merge_with_prefix`` and success_details use the skill name as
        ``check_name``.  In single-skill mode neither of these holds; instead
        success_detail check_names are validator-internal IDs like
        ``manifest_exists``.

        Returns the inferred skill name if single-skill mode, else None.
        """
        # Check for [prefix] pattern in any finding — indicates folder mode
        for r in results:
            for f in r.findings:
                if f.file_path.startswith("[") and "]" in f.file_path:
                    return None

        # Try to infer skill name from the first absolute file_path
        for r in results:
            for f in r.findings:
                fp = f.file_path
                if "/" in fp:
                    p = Path(fp)
                    while p.parent != p:
                        if (p / "SKILL.md").exists() or (p / "skill.md").exists():
                            return p.name
                        p = p.parent

        # Fall back: look for quality_scores metadata which always has skill_name
        for r in results:
            qs = r.metadata.get("quality_scores") if r.metadata else None
            if qs and qs.get("skill_name"):
                return qs["skill_name"]

        # Fall back: target_path
        if self.target_path:
            tp = Path(self.target_path)
            if (tp / "SKILL.md").exists() or (tp / "skill.md").exists():
                return tp.name

        return None

    def _reorganize_single_skill(
        self,
        results: list[ValidationResult],
        skill_name: str,
    ) -> dict[str, dict[str, Any]]:
        """Reorganize results for a single-skill run into one entry."""
        skill_data: dict[str, Any] = {"passed": True, "validators": {}, "issue_count": 0}

        for result in results:
            vn = result.validator_name
            vdata: dict[str, Any] = {
                "passed": result.passed,
                "description": result.validator_description,
                "details": [],
                "findings": [],
            }

            for detail in result.success_details:
                vdata["details"].append(
                    {
                        "check_name": detail.check_name,
                        "message": detail.message,
                        "metadata": detail.metadata,
                    }
                )

            for finding in result.findings:
                clean_path = finding.file_path
                if "/" + skill_name + "/" in clean_path:
                    clean_path = clean_path[clean_path.index("/" + skill_name + "/") + len(skill_name) + 2 :]

                confidence = "high"
                if hasattr(finding, "metadata") and isinstance(finding.metadata, dict):
                    confidence = finding.metadata.get("confidence", "high")

                vdata["findings"].append(
                    {
                        "category": finding.category,
                        "severity": finding.severity.value
                        if hasattr(finding.severity, "value")
                        else str(finding.severity),
                        "check_name": finding.check_name,
                        "message": self._normalize_message(finding.message),
                        "file_path": clean_path,
                        "line_number": finding.line_number,
                        "line_content": finding.line_content,
                        "suggestion": finding.suggestion,
                        "location": finding.location,
                        "confidence": confidence,
                        "metadata": finding.metadata if hasattr(finding, "metadata") else {},
                    }
                )

            if not result.passed:
                skill_data["passed"] = False
                vdata["passed"] = False

            skill_data["validators"][vn] = vdata

        # Deduplicate findings per validator
        for vdata in skill_data["validators"].values():
            if vdata["findings"]:
                vdata["findings"] = self._deduplicate_findings(vdata["findings"])

        skill_data["issue_count"] = sum(
            sum(f.get("occurrences", 1) for f in vd["findings"]) for vd in skill_data["validators"].values()
        )

        return {skill_name: skill_data}

    def _reorganize_by_skill(self, results: list[ValidationResult]) -> dict[str, dict[str, Any]]:
        """Reorganize validation results by skill instead of by validator.

        Returns a dict like:
        {
            "skill-name": {
                "passed": True/False,
                "validators": {
                    "SCHEMA": {"passed": True, "details": [...], "findings": [...]},
                    "SECURITY": {"passed": True, "details": [...], "findings": [...]},
                }
            }
        }
        """
        # Detect single-skill mode and use the dedicated path
        single_skill = self._is_single_skill_mode(results)
        if single_skill:
            return self._reorganize_single_skill(results, single_skill)

        skills: dict[str, dict[str, Any]] = {}

        for result in results:
            validator_name = result.validator_name

            # Extract skill names from success_details
            for detail in result.success_details:
                skill_name = detail.check_name
                # Skip discovery/folder-level checks
                if skill_name in (
                    "skill_discovery",
                    "folder_structure",
                    "skills_directory",
                    "team_skills_directory",
                    "pii_scan_start",
                    "pii_detection",
                ):
                    continue

                if skill_name not in skills:
                    skills[skill_name] = {"passed": True, "validators": {}}

                if validator_name not in skills[skill_name]["validators"]:
                    skills[skill_name]["validators"][validator_name] = {
                        "passed": True,
                        "description": result.validator_description,
                        "details": [],
                        "findings": [],
                    }

                skills[skill_name]["validators"][validator_name]["details"].append(
                    {
                        "check_name": detail.check_name,
                        "message": detail.message,
                        "metadata": detail.metadata,
                    }
                )

            # Extract skill names from findings (failures)
            for finding in result.findings:
                # Extract skill name from file_path (e.g., "[skill-name] file.md")
                file_path = finding.file_path
                skill_name = None
                if file_path.startswith("[") and "]" in file_path:
                    skill_name = file_path[1 : file_path.index("]")]
                else:
                    # Try to extract from path
                    parts = file_path.split("/")
                    if len(parts) > 0:
                        skill_name = parts[0]

                if skill_name:
                    if skill_name not in skills:
                        skills[skill_name] = {"passed": False, "validators": {}, "issue_count": 0}

                    if validator_name not in skills[skill_name]["validators"]:
                        skills[skill_name]["validators"][validator_name] = {
                            "passed": False,
                            "description": result.validator_description,
                            "details": [],
                            "findings": [],
                        }

                    skills[skill_name]["validators"][validator_name]["passed"] = False
                    skills[skill_name]["passed"] = False

                    # Clean file_path: strip redundant [skill-name] prefix
                    clean_path = file_path
                    if file_path.startswith("[") and "] " in file_path:
                        clean_path = file_path[file_path.index("] ") + 2 :]

                    # Strip absolute paths -- keep only path relative to skill dir
                    if "/" + skill_name + "/" in clean_path:
                        clean_path = clean_path[clean_path.index("/" + skill_name + "/") + len(skill_name) + 2 :]

                    confidence = "high"
                    if hasattr(finding, "metadata") and isinstance(finding.metadata, dict):
                        confidence = finding.metadata.get("confidence", "high")
                    skills[skill_name]["validators"][validator_name]["findings"].append(
                        {
                            "category": finding.category,
                            "severity": finding.severity.value
                            if hasattr(finding.severity, "value")
                            else str(finding.severity),
                            "check_name": finding.check_name,
                            "message": finding.message,
                            "file_path": clean_path,
                            "line_number": finding.line_number,
                            "line_content": finding.line_content,
                            "suggestion": finding.suggestion,
                            "location": finding.location,
                            "confidence": confidence,
                            "metadata": finding.metadata if hasattr(finding, "metadata") else {},
                        }
                    )
                    skills[skill_name]["issue_count"] = skills[skill_name].get("issue_count", 0) + 1

        # Deduplicate findings within each skill/validator
        for skill_data in skills.values():
            # Recompute issue_count after deduplication (use occurrences)
            deduped_count = 0
            for validator_data in skill_data["validators"].values():
                if validator_data["findings"]:
                    validator_data["findings"] = self._deduplicate_findings(validator_data["findings"])
                    deduped_count += sum(f.get("occurrences", 1) for f in validator_data["findings"])
            if deduped_count > 0:
                skill_data["issue_count"] = deduped_count

        # Ensure all skills have issue_count (including passing ones)
        for skill_data in skills.values():
            if "issue_count" not in skill_data:
                skill_data["issue_count"] = 0

        return skills

    @staticmethod
    def _normalize_message(msg: str) -> str:
        """Normalize a finding message for deduplication.

        Strips trailing whitespace/newlines and collapses internal whitespace
        so that near-identical messages (e.g., "Privilege Escalation: .env\\n"
        vs "Privilege Escalation: .env ") are grouped together.
        """
        # Strip and collapse whitespace
        return " ".join((msg or "").split())

    @staticmethod
    def _deduplicate_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Aggregate duplicate findings that share the same message, file, and suggestion.

        Groups findings by (message, file_path, suggestion) and merges duplicates
        into a single entry with an 'occurrences' count and a 'lines' list.
        This dramatically reduces visual noise when the same PII pattern
        (e.g., "Personal macOS user path") fires on many lines in the same file.
        """
        groups: dict[tuple, dict[str, Any]] = {}

        for finding in findings:
            # Normalize message for deduplication (strip trailing whitespace/newlines)
            norm_msg = HTMLReporter._normalize_message(finding.get("message", ""))
            key = (
                norm_msg,
                finding.get("file_path", ""),
                finding.get("suggestion", ""),
            )

            if key in groups:
                groups[key]["occurrences"] += 1
                line_num = finding.get("line_number")
                if line_num and line_num not in groups[key]["lines"]:
                    groups[key]["lines"].append(line_num)
                # Keep highest severity
                sev_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
                existing_sev = sev_order.get(groups[key]["severity"], 0)
                new_sev = sev_order.get(finding.get("severity", "medium"), 0)
                if new_sev > existing_sev:
                    groups[key]["severity"] = finding["severity"]
                # Preserve the first line_content as representative
                if not groups[key].get("line_content") and finding.get("line_content"):
                    groups[key]["line_content"] = finding["line_content"]
                # Track confidence (keep lowest = most suspicious)
                conf_order = {"low": 0, "medium": 1, "high": 2}
                existing_conf = groups[key].get("confidence", "high")
                new_conf = finding.get("confidence", "high")
                if conf_order.get(new_conf, 2) < conf_order.get(existing_conf, 2):
                    groups[key]["confidence"] = new_conf
            else:
                entry = dict(finding)
                entry["message"] = norm_msg  # Store normalized message
                entry["occurrences"] = 1
                line_num = finding.get("line_number")
                entry["lines"] = [line_num] if line_num else []
                # Extract confidence from metadata if present
                metadata = finding.get("metadata", {})
                if isinstance(metadata, dict) and "confidence" in metadata:
                    entry["confidence"] = metadata["confidence"]
                elif "confidence" not in entry:
                    entry["confidence"] = "high"
                groups[key] = entry

        # Sort lines for display
        for group in groups.values():
            group["lines"].sort()

        return list(groups.values())

    @staticmethod
    def _extract_issue_group_key(message: str, category: str) -> str:
        """Extract a grouping key from a finding message.

        For SECURITY findings from skillspector, messages often include the matched
        code snippet after a colon (e.g., "Privilege Escalation: ~/.ssh/id_ed25519").
        We group these by the prefix so all "Privilege Escalation" variants are
        aggregated into one top-issue row in the executive summary.

        For PII and SCHEMA findings, the full normalized message is used as-is.
        """
        norm = HTMLReporter._normalize_message(message)
        if category == "SECURITY" and ": " in norm:
            # Use the category prefix (e.g., "Privilege Escalation", "Data Exfiltration")
            return norm.split(": ", 1)[0]
        return norm

    @staticmethod
    def _compute_issue_key(category: str, group_key: str) -> str:
        """Stable, JS/CSS-safe identifier shared between top-issue rows and findings.

        The HTML report cross-links the executive-summary row, the per-skill
        affected pill, and the finding card itself. We need an identifier we
        can safely embed in ``data-issue-key`` attributes and inline ``onclick``
        JS strings without escaping concerns. Truncated SHA-1 hex meets that
        bar: hex-only, deterministic, and short enough to read in DevTools.
        16 hex chars (~64 bits) is well below collision risk for the few
        dozen unique issue groups a single report ever shows.
        """
        payload = f"{category}::{group_key}".encode()
        return hashlib.sha1(payload, usedforsecurity=False).hexdigest()[:16]

    @staticmethod
    def _compute_top_issues(
        skills: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compute cross-skill aggregation of top issues for the executive summary.

        For SECURITY findings, groups by category prefix (e.g., all
        "Privilege Escalation: ..." variants are merged into one row).
        For PII/SCHEMA findings, groups by exact (normalized) message.

        As a side effect, each underlying finding is tagged with
        ``issue_key`` (matching the row it rolls up to) so the template can
        deep-link from the executive summary directly to the finding card.

        Returns a list of dicts sorted by total_count descending.
        """
        sev_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
        issue_map: dict[tuple, dict[str, Any]] = {}

        for skill_name, skill_data in skills.items():
            for validator_data in skill_data["validators"].values():
                for finding in validator_data.get("findings", []):
                    msg = finding.get("message", "")
                    category = finding.get("category", "")
                    group_key = HTMLReporter._extract_issue_group_key(msg, category)
                    issue_key = HTMLReporter._compute_issue_key(category, group_key)
                    # Tag the finding so the template can attach a matching
                    # ``data-issue-key`` to its card and JS can locate it.
                    finding["issue_key"] = issue_key
                    key = (group_key, category)

                    occurrences = finding.get("occurrences", 1)

                    if key in issue_map:
                        issue_map[key]["total_count"] += occurrences
                        if skill_name not in issue_map[key]["skills_affected"]:
                            issue_map[key]["skills_affected"].append(skill_name)
                        # Keep highest severity
                        existing = sev_order.get(issue_map[key]["severity"], 0)
                        new = sev_order.get(finding.get("severity", "medium"), 0)
                        if new > existing:
                            issue_map[key]["severity"] = finding["severity"]
                    else:
                        issue_map[key] = {
                            "message": group_key,
                            "severity": finding.get("severity", "medium"),
                            "total_count": occurrences,
                            "skills_affected": [skill_name],
                            "category": category,
                            "suggestion": finding.get("suggestion", ""),
                            "issue_key": issue_key,
                        }

        return sorted(issue_map.values(), key=lambda x: x["total_count"], reverse=True)

    @staticmethod
    def _extract_contributors(
        skills: dict[str, dict[str, Any]],
        results: list[ValidationResult],
    ) -> list[dict[str, Any]]:
        """Extract contributor summary by mapping authors to their content items.

        Searches for author information in:
        1. Success details metadata (for passed skills — author_format check)
        2. Finding metadata (for failed skills — current_author key)

        Returns a list of contributor dicts sorted by issue count descending:
        [
            {
                "author": "John Doe <john@example.com>",
                "items": [{"name": "skill-x", "passed": True, "issue_count": 0}, ...],
                "total_items": 3,
                "passed_count": 2,
                "failed_count": 1,
                "total_issues": 5,
            },
        ]
        """
        # Build skill_name -> author mapping from validation results
        skill_authors: dict[str, str] = {}

        # Track author from unprefixed author_format (single-skill runs)
        single_skill_author: str | None = None

        for result in results:
            # Check success_details for author info (passed skills)
            for detail in result.success_details:
                skill_name = detail.check_name
                if skill_name in skills and skill_name not in skill_authors:
                    # Look in nested checks metadata for author_format
                    checks = detail.metadata.get("checks", [])
                    for check in checks:
                        if check.get("name") == "author_format":
                            msg = check.get("description", "")
                            # Message format: "Valid author format: Name <email>"
                            if ": " in msg:
                                author = msg.split(": ", 1)[1]
                                skill_authors[skill_name] = author

                # Handle prefixed success details from failed skills:
                # check_name format: "[skill-name] author_format"
                if detail.check_name.startswith("[") and "] author_format" in detail.check_name:
                    sname = detail.check_name[1 : detail.check_name.index("]")]
                    if sname in skills and sname not in skill_authors:
                        msg = detail.message
                        if ": " in msg:
                            author = msg.split(": ", 1)[1]
                            skill_authors[sname] = author

                # Handle unprefixed author_format (single-skill runs):
                # check_name is just "author_format" with author in message
                if (
                    detail.check_name == "author_format"
                    and not detail.check_name.startswith("[")
                    and single_skill_author is None
                ):
                    msg = detail.message
                    if ": " in msg:
                        single_skill_author = msg.split(": ", 1)[1]

            # Check findings for author info (failed skills with author metadata)
            for finding in result.findings:
                if finding.check_name == "author_format" and finding.metadata:
                    current_author = finding.metadata.get("current_author")
                    if current_author:
                        # Extract skill name from prefixed file_path "[skill-name] path"
                        fp = finding.file_path
                        if fp.startswith("[") and "]" in fp:
                            sname = fp[1 : fp.index("]")]
                            if sname in skills:
                                skill_authors[sname] = current_author

        # For single-skill runs, apply the unprefixed author to all skills
        # that don't already have an author assigned
        if single_skill_author:
            for skill_name in skills:
                if skill_name not in skill_authors:
                    skill_authors[skill_name] = single_skill_author

        # Group skills by author
        author_map: dict[str, list[dict[str, Any]]] = {}
        for skill_name, skill_data in skills.items():
            author = skill_authors.get(skill_name, "Unknown")
            if author not in author_map:
                author_map[author] = []
            author_map[author].append(
                {
                    "name": skill_name,
                    "passed": skill_data["passed"],
                    "issue_count": skill_data.get("issue_count", 0),
                }
            )

        # Build contributor list
        contributors = []
        for author, items in author_map.items():
            passed_count = sum(1 for i in items if i["passed"])
            total_issues = sum(i["issue_count"] for i in items)
            contributors.append(
                {
                    "author": author,
                    "items": sorted(items, key=lambda x: (x["passed"], x["name"])),
                    "total_items": len(items),
                    "passed_count": passed_count,
                    "failed_count": len(items) - passed_count,
                    "total_issues": total_issues,
                }
            )

        # Sort: most issues first, then by name
        contributors.sort(key=lambda c: (-c["total_issues"], c["author"]))
        return contributors

    @staticmethod
    def _compute_target_display(url: str) -> str:
        """Compute a short display label for a target URL.

        For repository URLs like
        ``https://github.com/example/project/tree/HEAD/skills/my-skill``
        returns ``ai_tools/ai_rules / skills/my-skill``.

        Falls back to the path portion of the URL, or the raw string for
        non-URL values.
        """
        if not url or not url.startswith("https://"):
            return url or ""
        parsed = urlparse(url)
        path = parsed.path.lstrip("/")
        if "/-/tree/" in path:
            repo_part, _, rest = path.partition("/-/tree/")
            branch_and_path = rest.split("/", 1)
            if len(branch_and_path) == 2:
                return f"{repo_part} / {branch_and_path[1]}"
            return repo_part
        return path or url

    @staticmethod
    def _compute_friendly_skill_label(target: str | None) -> str:
        """Strip repo / filesystem prefixes so the hero card shows ``skills/<name>``.

        The header keeps the full clickable path/URL for traceability; the hero
        card only gets a short, audience-friendly label so reviewers don't have
        to skim ``/workspaces/example-project/.../skills/log-analyzer`` to find
        the skill they care about.

        Resolution order — first match wins:

        1. ``team-skills/<team>/<name>`` is preserved verbatim. The team prefix
           carries useful provenance ("which team owns this skill") so we keep
           it instead of collapsing to the bare name.
        2. ``skills/<name>`` is preserved verbatim. The single ``skills/``
           segment makes it obvious the artifact is a skill (not a rule or
           workflow) without leaking the surrounding repo / worktree path.
        3. Falls back to the basename — best effort when no canonical
           ``skills/`` or ``team-skills/`` segment is present (e.g. validating
           an ad-hoc directory outside the standard Skill Evaluator layout).
        4. Empty string when ``target`` is ``None`` / empty so callers can
           defensibly chain ``label or fallback``.
        """
        if not target:
            return ""
        path = target
        if path.startswith("https://"):
            parsed = urlparse(path)
            path = parsed.path.lstrip("/")
            if "/-/tree/" in path:
                _repo, _, rest = path.partition("/-/tree/")
                branch_and_path = rest.split("/", 1)
                if len(branch_and_path) == 2:
                    path = branch_and_path[1]
        for marker in ("team-skills/", "skills/"):
            idx = path.rfind(marker)
            if idx >= 0:
                return path[idx:]
        try:
            return Path(path).name or path
        except (ValueError, OSError):
            return path

    # Validator names produced by ``commands.validate`` for each tier.  Used
    # to bucket combined run results back into per-tier summaries for the
    # hero card so the chip for "Tier 1" reflects only Tier 1 validators
    # instead of leaking Tier 2 / Tier 3 stats from the global totals.
    _TIER3_VALIDATOR_NAMES: ClassVar[frozenset[str]] = frozenset({"AGENT_EVAL"})

    @classmethod
    def _split_results_by_tier(
        cls, results: list[ValidationResult]
    ) -> tuple[list[ValidationResult], list[ValidationResult], list[ValidationResult]]:
        """Bucket a flat results list into ``(tier1, tier2, tier3)`` slices.

        ``commands.validate`` concatenates results in tier order before handing
        them to the reporter, so we restore tier identity here by validator
        name rather than relying on list slicing (which would silently break
        if a tier ever produced zero results).
        """
        tier1: list[ValidationResult] = []
        tier2: list[ValidationResult] = []
        tier3: list[ValidationResult] = []
        for r in results:
            name = getattr(r, "validator_name", None) or ""
            if name in cls._TIER3_VALIDATOR_NAMES:
                tier3.append(r)
            elif is_tier2_validator_name(name):
                tier2.append(r)
            else:
                tier1.append(r)
        return tier1, tier2, tier3

    @staticmethod
    def _compute_tier_summary(results: list[ValidationResult]) -> dict[str, Any]:
        """Compact stats for a single tier — what the hero chip displays.

        Returns ``passed`` (boolean), executed/pass/skip counts,
        ``issue_count`` (sum of findings), and per-severity totals.
        ``total == 0`` lets the template hide the chip entirely without an
        extra "did this tier run?" predicate.
        """
        total = len(results)
        passed_count = sum(1 for r in results if r.passed)
        advisory_skipped_count = sum(1 for r in results if is_advisory_agent_eval_skip(r))
        failed_count = sum(1 for r in results if not passes_required_gate(r))
        issue_count = 0
        critical = high = medium = low = 0
        for r in results:
            issue_count += len(r.findings or [])
            for f in r.findings or []:
                sev = f.severity.value if hasattr(f.severity, "value") else str(f.severity).lower()
                if sev == "critical":
                    critical += 1
                elif sev == "high":
                    high += 1
                elif sev == "medium":
                    medium += 1
                elif sev == "low":
                    low += 1
        return {
            "total": total,
            "passed_count": passed_count,
            "advisory_skipped_count": advisory_skipped_count,
            "failed_count": failed_count,
            "issue_count": issue_count,
            "all_passed": total > 0 and failed_count == 0,
            "incomplete_count": sum(1 for r in results if r.is_incomplete),
            "critical": critical,
            "high": high,
            "medium": medium,
            "low": low,
        }

    def _results_to_dict(self, results: list[ValidationResult]) -> list[dict[str, Any]]:
        output = []
        for result in results:
            result_dict = {
                "validator_name": result.validator_name,
                "validator_description": result.validator_description,
                "passed": result.passed,
                "status": "skipped" if is_advisory_agent_eval_skip(result) else result.status,
                "incomplete_scans": result.incomplete_scans,
                "summary": {
                    "files_scanned": result.summary.files_scanned,
                    "checks_performed": result.summary.checks_performed,
                    "errors": result.summary.errors,
                    "warnings": result.summary.warnings,
                    "critical_count": result.summary.critical_count,
                    "high_count": result.summary.high_count,
                    "medium_count": result.summary.medium_count,
                    "low_count": result.summary.low_count,
                },
                "findings": [],
                "success_details": [],
                "messages": result.messages,
                "errors": result.errors,
                "warnings": result.warnings,
            }
            for finding in result.findings:
                result_dict["findings"].append(
                    {
                        "category": finding.category,
                        "severity": finding.severity.value,
                        "check_name": finding.check_name,
                        "message": finding.message,
                        "file_path": finding.file_path,
                        "line_number": finding.line_number,
                        "line_content": finding.line_content,
                        "suggestion": finding.suggestion,
                        "location": finding.location,
                        "metadata": finding.metadata,
                    }
                )
            for detail in result.success_details:
                result_dict["success_details"].append(
                    {
                        "check_name": detail.check_name,
                        "message": detail.message,
                        "metadata": detail.metadata,
                    }
                )
            output.append(result_dict)
        return output

    @staticmethod
    def _tier3_report_data(results: list[ValidationResult]) -> dict[str, Any] | None:
        """Return canonical Tier 3 data, with a visible fallback for bare results."""
        for result in results:
            payload = result.metadata.get("agent_eval") if result.metadata else None
            if isinstance(payload, dict) and payload:
                return normalize_agent_eval_harbor_links(payload)

        if not results:
            return None

        # A validator failure should remain visible even if normalization
        # failed before canonical ``agent_eval`` metadata could be attached.
        result = results[0]
        verdict = "pass" if result.passed else "fail"
        execution_status = "succeeded" if result.passed else "failed"
        messages = [*result.errors, *result.warnings, *result.messages]
        message = messages[0] if messages else "Tier 3 did not provide canonical evaluation details."
        return {
            "schema_version": "2.0",
            "summary": {
                "schema_version": "2.0",
                "verdict": verdict,
                "execution_status": execution_status,
                "execution_errors": list(result.errors),
            },
            "skill_name": "",
            "verdict": verdict,
            "overall_score": None,
            "overall_lift": None,
            "execution_status": execution_status,
            "execution_errors": list(result.errors),
            "agents_run": [],
            "agents": {},
            "dimensions": [],
            "evaluators": {},
            "evaluator_cards": [],
            "cases": [],
            "suggestions": messages,
            "metric_ids": [],
            "metric_labels": {},
            "dataset": [],
            "provenance": {"source": "validation_result", "message": message},
        }

    def render(self, result: ValidationResult) -> str:
        return self.render_all([result])

    def render_all(self, results: list[ValidationResult]) -> str:
        all_passed = all(passes_required_gate(r) for r in results)
        has_incomplete = any(r.is_incomplete for r in results)
        overall_status = "incomplete" if has_incomplete else "passed" if all_passed else "failed"
        total_errors = sum(r.summary.errors for r in results)
        total_warnings = sum(r.summary.warnings for r in results)
        passed_count = sum(1 for r in results if r.passed)
        advisory_skipped_count = sum(1 for r in results if is_advisory_agent_eval_skip(r))
        failed_count = sum(1 for r in results if not passes_required_gate(r))
        total_issues = total_errors + total_warnings
        total_validators = len(results)
        executed_count = total_validators - advisory_skipped_count
        pass_percentage = round((passed_count / executed_count * 100) if executed_count > 0 else 0, 1)

        timestamp = ""
        if self.include_timestamp:
            timestamp = datetime.now(tz=UTC).strftime("%B %d, %Y at %I:%M %p UTC")

        # Reorganize results by skill for the new view
        skills_by_name = self._reorganize_by_skill(results)

        # Count total skills
        total_skills = len(skills_by_name)
        passed_skills = sum(1 for s in skills_by_name.values() if s["passed"])
        failed_skills = total_skills - passed_skills

        # Compute cross-skill top issues for executive summary
        top_issues = self._compute_top_issues(skills_by_name)

        # Extract contributor summary (author -> skills mapping)
        contributors = self._extract_contributors(skills_by_name, results)

        # Compute severity breakdown from actual findings (summary counts may be
        # incomplete when merge_with_prefix is used, so count from findings directly)
        total_critical = 0
        total_high = 0
        total_medium = 0
        total_low = 0
        for r in results:
            for f in r.findings:
                sev = f.severity.value if hasattr(f.severity, "value") else str(f.severity).lower()
                if sev == "critical":
                    total_critical += 1
                elif sev == "high":
                    total_high += 1
                elif sev == "medium":
                    total_medium += 1
                elif sev == "low":
                    total_low += 1

        target_display = self._compute_target_display(self.target_path or "")
        # Friendly hero label (``skills/<name>`` or ``team-skills/<team>/<name>``)
        # — strips repo / filesystem prefixes so the hero stays readable when
        # ``target_path`` is something like
        # ``/workspaces/example-project/skills/log-analyzer``. The full path
        # remains in the top-right header (``Target: ...``) for traceability.
        friendly_label = self._compute_friendly_skill_label(self.target_path or "")
        # Per-tier summaries so the hero card chips can show "Tier 1: 6/7
        # passed (0 critical)" etc. without leaking Tier 2 / Tier 3 stats
        # from the global aggregate counters.
        tier1_results, tier2_results, tier3_results = self._split_results_by_tier(results)
        tier1_summary = self._compute_tier_summary(tier1_results)
        tier2_summary = self._compute_tier_summary(tier2_results)
        tier3_summary = self._compute_tier_summary(tier3_results)
        tier3_data = self._tier3_report_data(tier3_results)
        tier3_preview, tier3_preview_notice = _bounded_tier3_preview(tier3_data)
        tier3_canonical_data, tier3_canonical_encoding = _canonical_tier3_embed(tier3_data)
        tier3_truncation = tier3_data.get("report_truncation", {}) if isinstance(tier3_data, dict) else {}

        # Keep the Tier 1 dashboard scoped to Tier 1. Tier 2 and Tier 3 have
        # dedicated tabs; including an advisory Tier 3 skip here would make
        # the dashboard report a failure even though it does not gate Tier 1.
        tier1_display_results = tier1_results
        tier1_skills = self._reorganize_by_skill(tier1_display_results)
        tier1_top_issues = self._compute_top_issues(tier1_skills)
        tier1_contributors = self._extract_contributors(tier1_skills, tier1_display_results)
        tier1_display_total = len(tier1_display_results)
        tier1_display_passed = sum(1 for result in tier1_display_results if result.passed)
        tier1_display_total_skills = len(tier1_skills)
        tier1_display_passed_skills = sum(1 for skill in tier1_skills.values() if skill["passed"])
        tier1_display_summary = {
            "total_validators": tier1_display_total,
            "passed_count": tier1_display_passed,
            "failed_count": tier1_display_total - tier1_display_passed,
            "total_issues": sum(result.summary.errors + result.summary.warnings for result in tier1_display_results),
            "pass_percentage": round(
                (tier1_display_passed / tier1_display_total * 100) if tier1_display_total else 0,
                1,
            ),
            "total_skills": tier1_display_total_skills,
            "passed_skills": tier1_display_passed_skills,
            "failed_skills": tier1_display_total_skills - tier1_display_passed_skills,
        }

        # Tier 1 and Tier 2 gate the CLI exit code; Tier 3 remains advisory.
        # Use each result's finalized ``passed`` state for ``would_block`` so
        # policy-remapped findings and validator errors match the actual CLI.
        blocking_results = tier1_results + tier2_results
        blocking_critical = tier1_summary["critical"] + tier2_summary["critical"]
        blocking_high = tier1_summary["high"] + tier2_summary["high"]
        blocking_medium = tier1_summary["medium"] + tier2_summary["medium"]
        blocking_low = tier1_summary["low"] + tier2_summary["low"]
        advisory_critical = tier3_summary["critical"]
        advisory_high = tier3_summary["high"]
        advisory_medium = tier3_summary["medium"]
        advisory_low = tier3_summary["low"]
        gating = {
            "blocking_tiers": ["tier1", "tier2"],
            "advisory_tiers": ["tier3"],
            "blocking": {
                "critical": blocking_critical,
                "high": blocking_high,
                "medium": blocking_medium,
                "low": blocking_low,
            },
            "advisory": {
                "critical": advisory_critical,
                "high": advisory_high,
                "medium": advisory_medium,
                "low": advisory_low,
            },
            "blocking_findings": blocking_critical + blocking_high,
            "would_block": any(not result.passed for result in blocking_results),
        }

        # Extract quality scores from results for per-skill quality display
        quality_scores_by_skill: dict[str, dict] = {}
        for r in results:
            qs = r.metadata.get("quality_scores") if r.metadata else None
            if qs and qs.get("dimensions"):
                sname = qs.get("skill_name", "")
                if sname:
                    quality_scores_by_skill[sname] = qs
            # Also check folder-level aggregation
            qs_all = r.metadata.get("quality_scores_all") if r.metadata else None
            if qs_all:
                for q in qs_all:
                    sname = q.get("skill_name", "")
                    if sname and sname not in quality_scores_by_skill:
                        quality_scores_by_skill[sname] = q

        def _attach_quality_scores(skills: dict[str, dict[str, Any]]) -> None:
            # Attach quality scores to per-skill data for the template.
            # Two strategies: (1) exact name match for folder-of-skills mode,
            # (2) attach to the FIRST entry only for single-skill mode (where
            # _reorganize_by_skill keys are check names, not skill names).
            matched = set()
            for sname, sdata in skills.items():
                if sname in quality_scores_by_skill:
                    sdata["quality"] = quality_scores_by_skill[sname]
                    matched.add(sname)

            # If no exact matches found, the report is for a single skill whose
            # name doesn't appear as a key. Attach to the first entry only to
            # avoid duplicating the quality panel across every check entry.
            if not matched and quality_scores_by_skill and skills:
                single_qs = next(iter(quality_scores_by_skill.values()))
                first_key = next(iter(skills))
                skills[first_key]["quality"] = single_qs

        _attach_quality_scores(skills_by_name)
        _attach_quality_scores(tier1_skills)

        report_data = {
            "title": self.title,
            "timestamp": timestamp,
            "version": __version__,
            "target_path": self.target_path,
            "target_display": target_display,
            "summary": {
                "all_passed": all_passed,
                "status": overall_status,
                "incomplete_scans": list(dict.fromkeys(tool for result in results for tool in result.incomplete_scans)),
                "total_validators": total_validators,
                "passed_count": passed_count,
                "advisory_skipped_count": advisory_skipped_count,
                "failed_count": failed_count,
                "total_issues": total_issues,
                "pass_percentage": pass_percentage,
                "total_skills": total_skills,
                "passed_skills": passed_skills,
                "failed_skills": failed_skills,
            },
            "results": self._results_to_dict(results),
            "skills": skills_by_name,
            "top_issues": top_issues,
            "contributors": contributors,
            "quality_scores": quality_scores_by_skill,
            # Tier 3 is embedded once in ``#tier3-full``. The export helper
            # resolves this reference at download time, avoiding a second full
            # copy inside ``#report-data``.
            "tier3": {"$ref": "#tier3-full"} if tier3_data else None,
            "gating": gating,
        }
        report_json = (
            json.dumps(report_data, indent=2, allow_nan=False)
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("&", "\\u0026")
        )

        # Dynamically add Tier 3 tab when agent eval data is present
        active_tabs = list(self.tabs)
        if not self._tabs_explicit and tier2_results and not any(tab["id"] == "tier2" for tab in active_tabs):
            active_tabs.append({"id": "tier2", "label": "Tier 2: Deduplication"})
        if not self._tabs_explicit and tier2_results and not tier1_display_results:
            active_tabs = [tab for tab in active_tabs if tab["id"] != "tier1"]
        if tier3_data and not any(t["id"] == "tier3" for t in active_tabs):
            active_tabs.append({"id": "tier3", "label": "Tier 3: Live Agent Evaluation"})

        # When the run only produced agent-eval results (no real Tier 1 validators
        # ran), drop the Tier 1 tab entirely so the report opens on the tier
        # that actually has data. This mirrors the user experience for
        # ``skill-evaluator agent-eval`` and ``skill-evaluator agent-eval-report``: there is
        # no "Tier 1: Security and Static Validation" content to show.
        agent_eval_only = bool(results) and all(getattr(r, "validator_name", None) == "AGENT_EVAL" for r in results)
        if agent_eval_only and tier3_data:
            active_tabs = [t for t in active_tabs if t["id"] != "tier1"]

        template = self._env.get_template("report.html.j2")
        cl = self.content_label
        return template.render(
            title=self.title,
            timestamp=timestamp,
            version=__version__,
            target_path=self.target_path,
            target_display=target_display,
            friendly_label=friendly_label,
            profile=self.profile or self._infer_profile_from_results(results),
            all_passed=all_passed,
            has_incomplete=has_incomplete,
            overall_status=overall_status,
            total_validators=total_validators,
            passed_count=passed_count,
            failed_count=failed_count,
            total_issues=total_issues,
            pass_percentage=pass_percentage,
            total_skills=total_skills,
            passed_skills=passed_skills,
            failed_skills=failed_skills,
            results=results,
            skills=skills_by_name,
            top_issues=top_issues,
            total_critical=total_critical,
            total_high=total_high,
            total_medium=total_medium,
            total_low=total_low,
            blocking_critical=blocking_critical,
            blocking_high=blocking_high,
            blocking_medium=blocking_medium,
            blocking_low=blocking_low,
            advisory_critical=advisory_critical,
            advisory_high=advisory_high,
            advisory_medium=advisory_medium,
            advisory_low=advisory_low,
            gating=gating,
            icons=self.ICONS,
            tabs=active_tabs,
            contributors=contributors,
            report_json=report_json,
            content_label=cl,
            content_label_plural=cl + "s",
            quality_scores=quality_scores_by_skill,
            tier3=tier3_preview,
            tier3_canonical_data=tier3_canonical_data,
            tier3_canonical_encoding=tier3_canonical_encoding,
            tier3_truncation=tier3_truncation,
            tier3_preview_notice=tier3_preview_notice,
            tier1_summary=tier1_summary,
            tier2_summary=tier2_summary,
            tier3_summary=tier3_summary,
            tier1_display_results=tier1_display_results,
            tier1_skills=tier1_skills,
            tier1_top_issues=tier1_top_issues,
            tier1_contributors=tier1_contributors,
            tier1_display_summary=tier1_display_summary,
            tier2_results=tier2_results,
            tier3_lift_pass_threshold=TIER3_LIFT_PASS_THRESHOLD,
            tier3_lift_fail_threshold=TIER3_LIFT_FAIL_THRESHOLD,
        )

    def get_file_extension(self) -> str:
        return ".html"
