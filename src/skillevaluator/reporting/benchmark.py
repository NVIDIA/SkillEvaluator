# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publication-ready BENCHMARK.md reporter for skill evaluation cards."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any

from skillevaluator.constants import (
    DIMENSION_HINTS,
    DIMENSION_MAPPING,
    DIMENSION_VERDICT_NEUTRAL_THRESHOLD,
    DIMENSION_VERDICT_PASS_THRESHOLD,
    TIER3_LIFT_FAIL_THRESHOLD,
    TIER3_LIFT_PASS_THRESHOLD,
)
from skillevaluator.publication_evidence import result_has_publication_evidence, result_publication_evidence
from skillevaluator.reporting.base import (
    PublicationTargetIdentity,
    ReporterBase,
    agent_eval_dimension_verdict,
    agent_eval_publication_dataset_provenance,
    agent_eval_publication_evaluated_at,
    agent_eval_publication_evidence_complete,
    agent_eval_publication_run_id,
    assess_publication,
    assess_tier3_evidence,
    get_skip_reason,
    is_advisory_agent_eval_skip,
    is_cleanly_skipped,
    is_tier2_result,
    is_tier3_result,
    publication_identity_present,
    publication_semantic_text,
    publication_target_for_results,
    resolve_benchmark_policy,
    result_has_execution_evidence,
    result_matches_publication_target,
    select_agent_eval_payload,
)
from skillevaluator.tier3_environments import HARBOR_ENV_MODES

if TYPE_CHECKING:
    from skillevaluator.models import Finding, ValidationResult


_SIGNAL_DESCRIPTIONS = {
    "security": "unsafe operations, secret leakage, and unauthorized access",
    "skill_execution": "whether the expected skill was found and executed",
    "skill_efficiency": "routing quality, workspace-aware skill reads, and productive tool use",
    "accuracy": "final-answer correctness against the reference answer",
    "goal_accuracy": "whether the user's goal was achieved",
    "behavior_check": "whether the expected workflow behavior was followed",
    "token_efficiency": "token usage with and without the skill (reported separately; not scored as a dimension)",
}

_RETIRED_PRODUCT_NAME = re.compile(r"\b[a-z]*[\s_-]*skills[\s_-]*eval\b", flags=re.IGNORECASE)
_RETIRED_PRODUCT_TOKEN = re.compile(r"\bskills[\s_-]*eval\b", flags=re.IGNORECASE)
_RETIRED_SANDBOX_REFERENCE = re.compile(
    rf"\b{re.escape(chr(97) + 'stra')}[\s_\-\u2010-\u2015\u2043\u2212]+sandbox\b",
    flags=re.IGNORECASE,
)
_POSIX_PATH_SEPARATORS = "/\u2044\u2215\u29f8"
_WINDOWS_PATH_SEPARATORS = "\\\u2216\u29f5"
_PATH_SEPARATOR_CLASS = re.escape(_POSIX_PATH_SEPARATORS + _WINDOWS_PATH_SEPARATORS)
_PATH_SEPARATOR_TRANSLATION = str.maketrans(
    {**dict.fromkeys(_POSIX_PATH_SEPARATORS[1:], "/"), **dict.fromkeys(_WINDOWS_PATH_SEPARATORS[1:], "\\")}
)
_PATH_START = re.compile(rf"(?<![A-Za-z0-9:/])(?:[A-Za-z]:[{_PATH_SEPARATOR_CLASS}]|[{_PATH_SEPARATOR_CLASS}])")
_QUOTED_ABSOLUTE_PATH = re.compile(
    rf"(?P<quote>['\"])(?P<path>(?:[A-Za-z]:[{_PATH_SEPARATOR_CLASS}]|[{_PATH_SEPARATOR_CLASS}])[^'\"\r\n]+)(?P=quote)"
)
_QUOTED_FILE_URI_PATH = re.compile(
    rf"(?P<quote>['\"])(?:file:)(?://[^/'\"\r\n]*)?"
    rf"(?P<path>(?:[A-Za-z]:[{_PATH_SEPARATOR_CLASS}]|[{_PATH_SEPARATOR_CLASS}])[^'\"\r\n]+)(?P=quote)",
    flags=re.IGNORECASE,
)
_FILE_URI_PATH = re.compile(
    rf"\bfile:(?://[^/\s'\"<>]*)?"
    rf"(?P<path>(?:[A-Za-z]:[{_PATH_SEPARATOR_CLASS}]|[{_PATH_SEPARATOR_CLASS}])[^\s'\"<>]+)",
    flags=re.IGNORECASE,
)
_PUBLIC_USER_PATH = re.compile(
    rf"(?P<path>(?:"
    rf"[{re.escape(_POSIX_PATH_SEPARATORS)}](?:Users|home)[{re.escape(_POSIX_PATH_SEPARATORS)}]"
    rf"|[A-Za-z]:[{_PATH_SEPARATOR_CLASS}]Users[{_PATH_SEPARATOR_CLASS}]"
    rf"|[{re.escape(_WINDOWS_PATH_SEPARATORS)}]Users[{re.escape(_WINDOWS_PATH_SEPARATORS)}]"
    rf")[^\s'\"<>]+)",
    flags=re.IGNORECASE,
)
_MARKDOWN_INLINE_SPECIAL = re.compile(r"([\\*_\[\]~])")
_MARKDOWN_BLOCK_PREFIX = re.compile(r"^(?:#{1,6}|>|[+*-]|\d+[.)])(?=\s|$)")
_MARKDOWN_THEMATIC_BREAK = re.compile(r"^(?:\s*[-*_]){3,}\s*$")
_PUBLICATION_URL_SCHEME = re.compile(r"(?P<scheme>https?|ftp)://", flags=re.IGNORECASE)
_PUBLICATION_WWW_PREFIX = re.compile(r"\bwww\.", flags=re.IGNORECASE)
_TRAILING_PATH_PUNCTUATION = ".,;!?)]}>`'\""
_LEGACY_AMBIGUOUS_UPLIFT = re.compile(r"\b(?P<score>\d+%)\s+\((?P<change>[+-]\d+%)\)")
_LEGACY_NUM_HEADER = re.compile(r"\|\s*Dimension\s*\|\s*Num\s*\|", flags=re.IGNORECASE)


class BenchmarkReporter(ReporterBase):
    """Render a stable, publication-oriented ``BENCHMARK.md`` card."""

    def __init__(
        self,
        *,
        include_timestamp: bool = True,
        max_findings_shown: int = 5,
        skill_name: str | None = None,
    ) -> None:
        self.include_timestamp = include_timestamp
        self.max_findings_shown = max_findings_shown
        self.skill_name = skill_name

    @property
    def name(self) -> str:
        return "benchmark"

    @property
    def description(self) -> str:
        return "Publication-ready BENCHMARK.md skill evaluation card"

    def render(self, result: ValidationResult) -> str:
        return self.render_all([result])

    def render_all(self, results: list[ValidationResult]) -> str:
        ae = _agent_eval_payload(results)
        expected_skill_name = self.skill_name or _skill_name(results, ae)
        skill_name = _publication_safe_skill_name(expected_skill_name)
        private_labels = _private_environment_labels(ae)
        policy = _benchmark_policy(results, ae, expected_skill_name=expected_skill_name)
        status = _overall_status(results, ae, policy, expected_skill_name=expected_skill_name)

        lines: list[str] = [
            f"# Skill Benchmark: {skill_name}",
            "",
            _verdict_callout(status),
            "",
        ]
        if status == "PASS":
            self._render_publication_recommendation(lines, results)
        elif status == "FAIL":
            lines.extend(
                [
                    (
                        "The skill should be reviewed before publication. Address the blocking findings below, "
                        "then rerun SkillEvaluator."
                    ),
                    "",
                ]
            )
        elif status == "INCOMPLETE":
            lines.extend(
                [
                    (
                        "One or more required evaluation tiers did not complete, so this benchmark is not "
                        "publication-complete."
                    ),
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    (
                        "Live evaluation did not show a material gain or regression. Collect more evidence or "
                        "improve the skill before making a publication decision."
                    ),
                    "",
                ]
            )

        self._render_metadata(
            lines,
            results,
            ae,
            skill_name,
            policy,
            private_labels=private_labels,
            expected_skill_name=expected_skill_name,
        )
        self._render_report_purpose(lines)
        self._render_results_at_a_glance(lines, ae, private_labels)
        self._render_tier_status(
            lines,
            results,
            ae,
            policy,
            private_labels,
            expected_skill_name=expected_skill_name,
        )
        self._render_findings(lines, results, private_labels)
        self._render_methodology(lines, ae, private_labels)
        self._render_freshness(lines)

        return "\n".join(lines).rstrip() + "\n"

    def _render_publication_recommendation(
        self,
        lines: list[str],
        results: list[ValidationResult],
    ) -> None:
        lines.extend(["## Publication Recommendation", ""])
        if _advisory_agent_eval_skip_message(results):
            lines.append(
                "Tier 3 live evaluation was skipped and does not block required validation. "
                "Publication suitability in this report is based on the completed required-tier "
                "results; rerun Tier 3 when the live evaluation runtime is available."
            )
        else:
            lines.append("Recommended for publication based on the completed evaluation evidence in this report.")
        lines.append("")

    def _render_metadata(
        self,
        lines: list[str],
        results: list[ValidationResult],
        ae: dict[str, Any] | None,
        skill_name: str,
        benchmark_policy: dict[str, bool],
        *,
        private_labels: tuple[str, ...],
        expected_skill_name: str | None,
    ) -> None:
        lines.extend(["## Evaluation Metadata", "", f"- Skill: `{skill_name}`"])

        publication_target = publication_target_for_results(
            results,
            ae,
            expected_skill_name=expected_skill_name,
        )
        if publication_target is not None:
            lines.append(
                f"- Source digest: `{publication_target.skill_digest}` ({publication_target.skill_digest_algorithm})"
            )
        else:
            lines.append("- Source digest: not recorded (legacy or unbound result)")

        evaluated_at = _evaluated_at(ae)
        lines.append(
            f"- Evaluation date: {_publication_safe_inline(_evaluation_date(evaluated_at))}"
            if evaluated_at
            else "- Evaluation date: not recorded (legacy or non-live result)"
        )

        summary = _mapping((ae or {}).get("summary"))
        version = _first_publication_safe_label(
            ((ae or {}).get("evaluator_version"), summary.get("evaluator_version")),
        )
        lines.append(
            f"- Evaluator version: `{version}`"
            if version
            else "- Evaluator version: not recorded (legacy or non-live result)"
        )

        agents = _agents(ae)
        if agents:
            lines.append(
                "- Agents: " + ", ".join(_agent_label(name, agent, private_labels) for name, agent in agents.items())
            )
        else:
            requested = (ae or {}).get("requested_agents") or []
            if isinstance(requested, list) and requested:
                labels = ", ".join(
                    f"{_human_agent_name(_first_publication_safe_label((agent,)) or 'Agent')} (model not recorded)"
                    for agent in requested
                )
                lines.append("- Agents: requested but not run — " + labels)
            else:
                lines.append("- Agents: not recorded (legacy or non-live result)")

        dataset_summary = _dataset_summary(ae)
        if dataset_summary["total_tasks"] > 0:
            composition = _dataset_composition_label(dataset_summary)
            lines.append(f"- Tasks: {dataset_summary['total_tasks']} evaluation tasks{composition}")
        else:
            lines.append("- Tasks: not recorded (legacy or non-live result)")

        dataset_provenance = agent_eval_publication_dataset_provenance(ae)
        if dataset_provenance is not None:
            digest, digest_algorithm = dataset_provenance
            safe_digest = _publication_safe_inline(digest)
            algorithm_label = f" ({_publication_safe_inline(digest_algorithm)})"
            lines.append(f"- Dataset digest: `{safe_digest}`{algorithm_label}")
        else:
            lines.append("- Dataset digest: not recorded (legacy or non-live result)")

        tier3_run_id = agent_eval_publication_run_id(ae)
        lines.append(
            f"- Tier 3 run ID: `{_publication_safe_inline(tier3_run_id)}`"
            if tier3_run_id
            else "- Tier 3 run ID: not recorded (Tier 3 did not complete)"
        )

        policy = _mapping((ae or {}).get("attempt_policy"))
        attempts = _nonnegative_int(policy.get("max_attempts"))
        if attempts > 0:
            lines.append(f"- Attempts per task: {attempts}")
        else:
            lines.append("- Attempts per task: not recorded (legacy or non-live result)")

        environment = _environment(ae)
        if environment:
            lines.append(f"- Environment: `{_publication_safe_environment(environment)}`")
        else:
            lines.append("- Environment: not recorded (legacy or non-live result)")

        tier2_requirement = "required for publication" if benchmark_policy["tier2_required"] else "optional by policy"
        lines.append(f"- Tier 2 evidence: {tier2_requirement}")
        tier3_requirement = "required for publication" if benchmark_policy["tier3_required"] else "optional by policy"
        lines.append(f"- Tier 3 evidence: {tier3_requirement}")
        if skip_message := _advisory_agent_eval_skip_message(results):
            lines.append(
                f"- Tier 3 live evaluation: SKIPPED — {_publication_safe_inline(skip_message, private_labels)}"
            )

        lines.append("")
        environment_note = _environment_note(environment)
        if environment_note:
            lines.extend([environment_note, ""])

    @staticmethod
    def _render_report_purpose(lines: list[str]) -> None:
        lines.extend(
            [
                "## What This Report Answers",
                "",
                "The three-tier evaluation checks whether the skill:",
                "",
                "- is safe to use;",
                "- produces correct answers;",
                "- is discovered and activated when needed;",
                "- helps the agent complete the user's goal and expected workflow; and",
                "- avoids wasted skill and tool usage.",
                "",
            ]
        )

    @staticmethod
    def _render_results_at_a_glance(
        lines: list[str],
        ae: dict[str, Any] | None,
        private_labels: tuple[str, ...],
    ) -> None:
        lines.extend(["## Results at a Glance", ""])
        agents = _agents(ae)
        if not agents:
            lines.extend(
                [
                    "Tier 3 live-agent scores were not available. See the tier status table for what ran.",
                    "",
                ]
            )
            return

        headers = [
            "Measure",
            *[_agent_table_label(name, agent, private_labels) for name, agent in agents.items()],
        ]
        lines.append("| " + " | ".join(_md_cell(header, private_labels) for header in headers) + " |")
        lines.append("|---|" + "|".join(["---:"] * len(agents)) + "|")

        overall_row = ["Overall"]
        overall_row.extend(
            _score_transition_values(
                _number(agent.get("baseline")),
                _number(agent.get("with_skill", agent.get("overall_score"))),
            )
            for agent in agents.values()
        )
        lines.append("| " + " | ".join(_md_cell(value, private_labels) for value in overall_row) + " |")

        has_partial = False
        for dim_id in DIMENSION_MAPPING:
            row = [dim_id.title()]
            for agent in agents.values():
                dimension = _agent_dimension(agent, dim_id)
                if dimension and dimension.get("partial"):
                    has_partial = True
                row.append(_score_transition(dimension))
            lines.append("| " + " | ".join(_md_cell(value, private_labels) for value in row) + " |")

        lines.extend(
            [
                "",
                (
                    "**How to read this table:** baseline is the same task attempted without the target skill. "
                    "Uplift is `skill score - baseline score`, shown in percentage points."
                ),
                "",
                (
                    "Example: `47% → 92% (+45 points)` means the skill-assisted run scored 92%, "
                    "45 percentage points above its 47% no-skill baseline."
                ),
                "",
            ]
        )
        if has_partial:
            lines.extend(
                [
                    (
                        "A partial dimension was calculated from only the available configured signals; "
                        "review the detailed report before relying on it."
                    ),
                    "",
                ]
            )

    def _render_tier_status(
        self,
        lines: list[str],
        results: list[ValidationResult],
        ae: dict[str, Any] | None,
        benchmark_policy: dict[str, bool],
        private_labels: tuple[str, ...],
        *,
        expected_skill_name: str | None,
    ) -> None:
        publication_target = publication_target_for_results(
            results,
            ae,
            expected_skill_name=expected_skill_name,
        )
        tier_groups = [
            ("Tier 1", "Static validation", _tier1_results(results)),
            ("Tier 2", "Semantic deduplication", _tier2_results(results)),
            ("Tier 3", "Live agent evaluation", _tier3_results(results)),
        ]

        lines.extend(
            [
                "## Tier Status",
                "",
                "| Tier | Purpose | Status | Evidence |",
                "|---|---|---|---|",
            ]
        )
        for tier, purpose, tier_results in tier_groups:
            status, evidence = _tier_status(
                tier,
                tier_results,
                ae,
                benchmark_policy,
                expected_skill_name=expected_skill_name,
                expected_publication_target=publication_target,
            )
            lines.append(f"| {tier} | {purpose} | **{status}** | {_md_cell(evidence, private_labels)} |")
        lines.append("")

        for tier, _purpose, tier_results in tier_groups:
            if tier_results and all(is_cleanly_skipped(result) for result in tier_results):
                lines.append(f"{tier} validation was skipped and executed 0 checks.")
                for reason in _skip_reasons(tier_results):
                    lines.append(f"- {_publication_safe_inline(reason, private_labels)}")
                lines.append("")

    def _render_findings(
        self,
        lines: list[str],
        results: list[ValidationResult],
        private_labels: tuple[str, ...],
    ) -> None:
        findings_with_result = [(finding, result) for result in results for finding in result.findings]
        blocking = [
            finding
            for finding, result in findings_with_result
            if not result.passed or finding.severity.value in {"critical", "high"}
        ]

        if blocking:
            lines.extend(["## Blocking Findings", ""])
            for finding in _top_findings(blocking, limit=self.max_findings_shown):
                lines.append(_finding_line(finding, private_labels))
            lines.append("")

        static_test_limitations = list(
            dict.fromkeys(
                message for result in results if (message := self._static_test_evidence_message(result)) is not None
            )
        )
        if static_test_limitations:
            lines.extend(["Test execution limitations:", ""])
            lines.extend(
                f"- {_publication_safe_inline(message, private_labels)}" for message in static_test_limitations
            )
            lines.append("")

        lines.extend(["## Findings and Observations", ""])
        lines.extend(["<details>", "<summary>Show detailed findings and successful checks</summary>", ""])

        findings = [finding for finding, _result in findings_with_result]
        if findings:
            for finding in _top_findings(findings, limit=self.max_findings_shown):
                lines.append(_finding_line(finding, private_labels))
            remaining = len(findings) - min(len(findings), self.max_findings_shown)
            if remaining > 0:
                lines.append(f"- {remaining} additional finding(s) are available in the full evaluation artifacts.")
        else:
            observations = [
                f"- {_publication_safe_inline(result.validator_name, private_labels)}: "
                f"{_publication_safe_inline(detail.message, private_labels)}"
                for result in results
                for detail in result.success_details[:1]
            ]
            lines.extend(observations or ["- No findings or successful-check details were recorded."])

        lines.extend(["", "</details>", ""])

    @staticmethod
    def _static_test_evidence_message(result: ValidationResult) -> str | None:
        """Return the static-test limitation from direct or aggregated results."""
        for detail in result.success_details:
            if detail.check_name == "test_discovery":
                return detail.message
        for detail in result.success_details:
            checks = detail.metadata.get("checks") if isinstance(detail.metadata, dict) else None
            if not isinstance(checks, list):
                continue
            for check in checks:
                if isinstance(check, dict) and check.get("name") == "test_discovery":
                    return "Target tests were not executed and coverage was not measured for any discovered skill"
        return None

    @staticmethod
    def _render_methodology(
        lines: list[str],
        ae: dict[str, Any] | None,
        private_labels: tuple[str, ...],
    ) -> None:
        policy = _mapping((ae or {}).get("verdict_policy"))
        attempt_policy = _mapping((ae or {}).get("attempt_policy"))
        attempt_threshold = _number(policy.get("attempt_pass_threshold", attempt_policy.get("pass_threshold")))
        dimension_pass = _number(policy.get("dimension_pass_threshold")) or DIMENSION_VERDICT_PASS_THRESHOLD
        dimension_neutral = (
            _number(policy.get("dimension_neutral_threshold"))
            if policy.get("dimension_neutral_threshold") is not None
            else DIMENSION_VERDICT_NEUTRAL_THRESHOLD
        )
        lift_pass = _number(policy.get("lift_pass_threshold"))
        lift_fail = _number(policy.get("lift_fail_threshold"))
        if lift_pass is None:
            lift_pass = TIER3_LIFT_PASS_THRESHOLD
        if lift_fail is None:
            lift_fail = TIER3_LIFT_FAIL_THRESHOLD

        lines.extend(
            [
                "## Scoring Methodology",
                "",
                "<details>",
                "<summary>Show dimension definitions, source signals, and thresholds</summary>",
                "",
                "| Dimension | Question | Scored signals |",
                "|---|---|---|",
            ]
        )
        for dim_id, config in DIMENSION_MAPPING.items():
            signals = _weighted_signals(config)
            question = config.get("question") or DIMENSION_HINTS.get(dim_id, "")
            lines.append(f"| {dim_id.title()} | {_trusted_md_cell(question)} | {_trusted_md_cell(signals)} |")

        lines.extend(
            [
                "",
                (
                    f"- Dimension bands: PASS at {dimension_pass:.0%} or above; NEUTRAL from "
                    f"{dimension_neutral:.0%} to below {dimension_pass:.0%}; FAIL below {dimension_neutral:.0%}."
                ),
                (
                    f"- Overall Tier 3 lift: PASS at +{lift_pass * 100:.0f} points or more; "
                    f"FAIL at {lift_fail * 100:.0f} points or less; values between those bands are NEUTRAL."
                ),
                (
                    "- Overall verdict: PASS only when every configured dimension passes for at least one "
                    "supported agent. Lift is reported as diagnostic evidence and does not override this gate."
                ),
            ]
        )
        if attempt_threshold is not None:
            lines.append(
                f"- The {attempt_threshold:.0%} attempt pass threshold is a separate per-task gate; "
                "it is not the dimension pass threshold."
            )

        lines.extend(
            [
                (
                    "- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and "
                    "expected workflow adherence (`behavior_check`)."
                ),
                (
                    "- Token efficiency is a separate report-only signal. It does not change a dimension "
                    "score or the overall verdict."
                ),
                "",
            ]
        )

        signals = _metric_signals(ae)
        if signals:
            labels = _metric_labels(ae)
            lines.append("Signals present in this run:")
            lines.append("")
            for signal in signals:
                safe_signal = _publication_safe_inline(signal, private_labels)
                label = _publication_safe_label(
                    labels.get(signal, signal.replace("_", " ").title()),
                    private_labels,
                )
                description = _SIGNAL_DESCRIPTIONS.get(signal, "additional evaluator signal")
                lines.append(f"- `{safe_signal}` ({label}): {description}.")
            lines.append("")

        lines.extend(["</details>", ""])

    @staticmethod
    def _render_freshness(lines: list[str]) -> None:
        lines.extend(
            [
                "## Freshness",
                "",
                (
                    "Regenerate this benchmark when the skill, evaluation dataset, target agent/model, "
                    "evaluator version, environment, or scoring policy changes."
                ),
                "",
            ]
        )

    def get_file_extension(self) -> str:
        return ".md"


def _agent_eval_payload(results: list[ValidationResult]) -> dict[str, Any] | None:
    return select_agent_eval_payload(results)


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _benchmark_policy(
    results: list[ValidationResult],
    ae: dict[str, Any] | None,
    *,
    expected_skill_name: str | None = None,
) -> dict[str, bool]:
    """Resolve the persisted publication policy for all configurable tiers."""
    return resolve_benchmark_policy(results, ae, expected_skill_name=expected_skill_name)


def _tier3_evidence_complete(ae: dict[str, Any] | None) -> bool:
    """Require a succeeded run with the minimum publication provenance."""
    return agent_eval_publication_evidence_complete(ae)


def _tier3_dimension_verdict(ae: dict[str, Any] | None) -> str | None:
    """Recompute the canonical verdict from every supported agent's dimensions."""
    return agent_eval_dimension_verdict(ae)


def _advisory_agent_eval_skip_message(results: list[ValidationResult]) -> str | None:
    if assess_tier3_evidence(results).status != "skipped":
        return None
    for result in results:
        if not is_advisory_agent_eval_skip(result):
            continue
        payload = result.metadata.get("agent_eval", {}) if result.metadata else {}
        provenance = payload.get("provenance", {}) if isinstance(payload, dict) else {}
        message = provenance.get("message") if isinstance(provenance, dict) else None
        return _safe_scalar_text(message).strip() or "Live evaluation did not run."
    return None


def _overall_status(
    results: list[ValidationResult],
    ae: dict[str, Any] | None,
    benchmark_policy: dict[str, bool],
    *,
    expected_skill_name: str | None = None,
) -> str:
    # Keep this compatibility wrapper so existing callers continue to use the
    # benchmark's uppercase vocabulary while every reporter shares one
    # publication assessment implementation.
    del benchmark_policy
    return assess_publication(results, ae, expected_skill_name=expected_skill_name).status.upper()


def _is_blocking_publication_skip(
    result: ValidationResult,
    benchmark_policy: dict[str, bool],
) -> bool:
    """Return whether a clean skip blocks this publication policy."""
    return bool(
        is_cleanly_skipped(result)
        and not is_advisory_agent_eval_skip(result)
        and not (_is_tier2(result) and not benchmark_policy["tier2_required"])
        and not (_is_tier3(result) and not benchmark_policy["tier3_required"])
    )


def _verdict_callout(status: str) -> str:
    labels = {
        "PASS": "✅ **Overall verdict: PASS — Recommended for publication**",
        "FAIL": "❌ **Overall verdict: FAIL — Publication blocked**",
        "INCOMPLETE": "⚠️ **Overall verdict: INCOMPLETE — Required evidence is missing**",
        "NEUTRAL": "**Overall verdict: NEUTRAL — One or more dimensions remain below PASS**",
    }
    return f"> {labels.get(status, f'**Overall verdict: {status}**')}"


def _skill_name(results: list[ValidationResult], ae: dict[str, Any] | None) -> str:
    if ae:
        summary = _mapping(ae.get("summary"))
        candidate = ae.get("skill_name") or summary.get("skill_name")
        if candidate_text := _safe_scalar_text(candidate).strip():
            return candidate_text
    for result in results:
        quality = result.metadata.get("quality_scores") if isinstance(result.metadata, dict) else None
        if isinstance(quality, dict) and (candidate := _safe_scalar_text(quality.get("skill_name")).strip()):
            return candidate
    return "skill"


def _safe_scalar_text(value: object) -> str:
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value) if value.bit_length() <= 256 else ""
    if isinstance(value, float) and math.isfinite(value):
        return str(value)
    return ""


def _agents(ae: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    agents = (ae or {}).get("agents")
    if not isinstance(agents, dict):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    normalized_names: set[str] = set()
    for raw_name, agent in agents.items():
        if not publication_identity_present(raw_name):
            return {}
        safe_name = publication_semantic_text(raw_name).strip()
        identity_key = safe_name.casefold()
        if not publication_identity_present(safe_name) or not isinstance(agent, dict):
            return {}
        if identity_key in normalized_names:
            return {}
        normalized_names.add(identity_key)
        normalized[safe_name] = agent
    return normalized


def _agent_label(
    name: str,
    agent: dict[str, Any],
    private_labels: tuple[str, ...] = (),
) -> str:
    display = _agent_display_label(name, agent, private_labels).replace(",", "&#44;")
    safe_model = _first_publication_safe_label(
        (agent.get("model"), agent.get("model_name"), agent.get("llm_model")),
    )
    if safe_model:
        return f"{display} (`{safe_model}`)"
    return f"{display} (model not recorded)"


def _agent_table_label(
    name: str,
    agent: dict[str, Any],
    private_labels: tuple[str, ...] = (),
) -> str:
    display = _agent_display_label(name, agent, private_labels)
    return f"{display} (Baseline → Skill Uplift)"


def _agent_display_label(
    name: str,
    agent: dict[str, Any],
    private_labels: tuple[str, ...] = (),
) -> str:
    label = ""
    for value in (agent.get("display_name"), agent.get("label")):
        if not publication_identity_present(value):
            continue
        candidate = _normalized_publication_text(value)
        contains_private_label = any(
            re.search(
                rf"(?<![^\W_]){re.escape(private_label)}(?![^\W_])",
                candidate,
                flags=re.IGNORECASE,
            )
            for private_label in private_labels
        )
        if contains_private_label:
            continue
        label = _publication_safe_label(candidate)
        if publication_identity_present(label):
            break
        label = ""
    # A display label containing private environment identity is presentation
    # data, not proof. Fall back to the canonical executed-agent key instead
    # of substituting text that would falsify provenance.
    if not label:
        label = _first_publication_safe_label((name,))
    return _human_agent_name(label or "Agent")


def _first_publication_safe_label(
    values: tuple[object, ...],
    private_labels: tuple[str, ...] = (),
) -> str:
    for value in values:
        if not publication_identity_present(value):
            continue
        label = _publication_safe_label(value, private_labels)
        if publication_identity_present(label):
            return label
    return ""


def _human_agent_name(name: str) -> str:
    if name == "claude-code":
        return "Claude Code"
    if re.fullmatch(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*", name):
        humanized = name.replace("_", " ").replace("-", " ").title()
    else:
        humanized = name.title()
    return humanized.replace("Skill" + "evaluator", "SkillEvaluator")


def _evaluated_at(ae: dict[str, Any] | None) -> str | None:
    return agent_eval_publication_evaluated_at(ae)


def _evaluation_date(value: str) -> str:
    candidate = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
        if parsed.tzinfo is not None and parsed.utcoffset() is not None:
            parsed = parsed.astimezone(UTC)
        return parsed.date().isoformat()
    except ValueError:
        return value[:10] if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", value) else value


def _environment(ae: dict[str, Any] | None) -> str | None:
    summary = _mapping((ae or {}).get("summary"))
    for value in (summary.get("environment"), (ae or {}).get("environment")):
        if publication_identity_present(value):
            return _safe_scalar_text(value).strip()
    return None


def _environment_note(environment: str | None) -> str | None:
    if not environment:
        return None
    public_environment = _publication_safe_environment(environment)
    lowered = public_environment.lower().replace("_", "-")
    if public_environment == "Isolated sandbox":
        return "Each task attempt ran in its own isolated sandbox."
    if "k8s" in lowered or "sandbox" in lowered:
        return "Each task attempt ran in its own isolated sandbox pod."
    if "docker" in lowered:
        return "Each task attempt ran in its own isolated Docker container."
    if "local" in lowered:
        return "Tasks ran on the trusted local host; local mode is not sandboxed."
    return None


def _dataset(ae: dict[str, Any] | None) -> list[dict[str, Any]]:
    dataset = (ae or {}).get("dataset")
    return [item for item in dataset if isinstance(item, dict)] if isinstance(dataset, list) else []


def _dataset_summary(ae: dict[str, Any] | None) -> dict[str, int | str]:
    summary = (ae or {}).get("dataset_summary")
    if isinstance(summary, dict):
        return {
            "total_tasks": _nonnegative_int(summary.get("total_tasks")),
            "positive_tasks": _nonnegative_int(summary.get("positive_tasks")),
            "negative_tasks": _nonnegative_int(summary.get("negative_tasks")),
            "unclassified_tasks": _nonnegative_int(summary.get("unclassified_tasks")),
            "source": _safe_scalar_text(summary.get("source")) or "payload",
        }

    dataset = _dataset(ae)
    if dataset:
        composition = _dataset_composition(dataset)
        return {
            "total_tasks": len(dataset),
            "positive_tasks": composition["positive"],
            "negative_tasks": composition["negative"],
            "unclassified_tasks": composition["unlabeled"],
            "source": "dataset",
        }

    task_ids: set[str] = set()
    raw_trials = (ae or {}).get("trials")
    trials = raw_trials if isinstance(raw_trials, list) else []
    for trial in trials:
        if not isinstance(trial, dict):
            continue
        for key in ("entry_id", "case_id", "task_id", "id"):
            value = trial.get(key)
            safe_value = _safe_scalar_text(value).strip()
            if safe_value:
                task_ids.add(safe_value)
                break
    return {
        "total_tasks": len(task_ids),
        "positive_tasks": 0,
        "negative_tasks": 0,
        "unclassified_tasks": len(task_ids),
        "source": "trials" if task_ids else "unavailable",
    }


def _dataset_composition(dataset: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter({"positive": 0, "negative": 0, "unlabeled": 0})
    for entry in dataset:
        if "expected_skill" not in entry:
            counts["unlabeled"] += 1
        elif entry.get("expected_skill") is None:
            counts["negative"] += 1
        else:
            counts["positive"] += 1
    return counts


def _dataset_composition_label(summary: dict[str, int | str]) -> str:
    positive = int(summary["positive_tasks"])
    negative = int(summary["negative_tasks"])
    unclassified = int(summary["unclassified_tasks"])
    parts = []
    if positive:
        parts.append(f"{positive} positive")
    if negative:
        parts.append(f"{negative} negative")
    if unclassified:
        parts.append(f"{unclassified} unclassified")
    return f" ({', '.join(parts)})" if parts else ""


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value if 0 <= value <= 2**63 - 1 else 0


def _number(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _agent_dimension(agent: dict[str, Any], dim_id: str) -> dict[str, Any] | None:
    raw_dimensions = agent.get("dimensions")
    dimensions = raw_dimensions if isinstance(raw_dimensions, list) else []
    for dimension in dimensions:
        if isinstance(dimension, dict) and dimension.get("id") == dim_id:
            return dimension
    return None


def _score_transition(dimension: dict[str, Any] | None) -> str:
    if not dimension:
        return "Not available"
    baseline = _number(dimension.get("baseline"))
    score = _number(dimension.get("with_skill", dimension.get("score")))
    return _score_transition_values(baseline, score)


def _score_transition_values(baseline: float | None, score: float | None) -> str:
    if score is None:
        return "Not available"
    skill_label = f"{score:.0%}"
    if baseline is None:
        return f"{skill_label} — baseline not run; uplift unavailable"
    delta = score - baseline
    return f"{baseline:.0%} → {skill_label} ({_format_points(delta)})"


def _format_points(delta: float) -> str:
    points = delta * 100
    if math.isclose(points, 0.0, abs_tol=0.05):
        return "±0 points"
    sign = "+" if points > 0 else "-"
    return f"{sign}{abs(points):.0f} points"


def _tier_status(
    tier: str,
    results: list[ValidationResult],
    ae: dict[str, Any] | None,
    benchmark_policy: dict[str, bool],
    *,
    expected_skill_name: str | None = None,
    expected_publication_target: PublicationTargetIdentity | None = None,
) -> tuple[str, str]:
    if not results:
        return "NOT RUN", "No result was recorded"
    incomplete = [result for result in results if result.is_incomplete]
    if incomplete:
        tools = list(dict.fromkeys(tool for result in incomplete for tool in result.incomplete_scans))
        return "INCOMPLETE", f"Missing trustworthy evidence from {', '.join(tools)}"
    if all(is_cleanly_skipped(result) for result in results):
        if tier == "Tier 3" and ae:
            tier3 = assess_tier3_evidence(
                results,
                ae,
                expected_skill_name=expected_skill_name,
            )
            if tier3.status != "skipped":
                return tier3.status.upper(), tier3.reason or "Tier 3 skip evidence is incomplete"
        if tier == "Tier 1":
            optional = False
        elif tier == "Tier 2":
            optional = not benchmark_policy["tier2_required"]
        else:
            optional = not benchmark_policy["tier3_required"] or all(
                is_advisory_agent_eval_skip(result) for result in results
            )
        return ("SKIPPED (ADVISORY)" if optional else "INCOMPLETE"), "; ".join(_skip_reasons(results))

    findings = [finding for result in results for finding in result.findings]
    if any(not result.passed and not is_cleanly_skipped(result) for result in results):
        return "FAILED", f"{len(results)} validator(s); {len(findings)} finding(s)"

    publication_results = results
    if tier in {"Tier 1", "Tier 2"}:
        expected_tier = 1 if tier == "Tier 1" else 2
        publication_results = [
            result for result in results if result_has_publication_evidence(result, tier=expected_tier)
        ]
        if not publication_results:
            return "INCOMPLETE", f"No recognized built-in {tier} producer evidence was recorded"

    unbound_results = [
        result
        for result in publication_results
        if not is_cleanly_skipped(result) and not result_matches_publication_target(result, expected_publication_target)
    ]
    if tier in {"Tier 1", "Tier 2"} and unbound_results:
        validator_names = list(
            dict.fromkeys(result.validator_name or f"{tier} validator" for result in unbound_results)
        )
        return "INCOMPLETE", f"Missing canonical source identity from {', '.join(validator_names)}"

    if tier in {"Tier 1", "Tier 2"} and (missing_evidence := _results_without_execution_evidence(publication_results)):
        validator_names = list(
            dict.fromkeys(result.validator_name or f"{tier} validator" for result in missing_evidence)
        )
        return "INCOMPLETE", f"Missing trustworthy execution evidence from {', '.join(validator_names)}"

    skipped = [result for result in publication_results if is_cleanly_skipped(result)]
    if tier in {"Tier 1", "Tier 2"} and skipped:
        reasons = "; ".join(_skip_reasons(skipped))
        completed = len(publication_results) - len(skipped)
        evidence = f"{completed} completed validator(s); {len(findings)} finding(s); {reasons}"
        if any(_is_blocking_publication_skip(result, benchmark_policy) for result in skipped):
            return "INCOMPLETE", evidence
        return "PASSED WITH OBSERVATIONS", evidence

    if tier == "Tier 3":
        if not ae:
            evidence = (
                "Required Tier 3 evidence is missing"
                if benchmark_policy["tier3_required"]
                else "Present Tier 3 result lacks complete evidence"
            )
            return "INCOMPLETE", evidence
        tier3 = assess_tier3_evidence(results, ae, expected_skill_name=expected_skill_name)
        if tier3.status == "fail":
            return "FAIL", f"{len(_agents(ae))} agent(s); {_dataset_summary(ae)['total_tasks']} task(s)"
        if not tier3.evidence_complete:
            evidence = tier3.reason or (
                "Required Tier 3 evidence is missing"
                if benchmark_policy["tier3_required"]
                else "Present Tier 3 result lacks complete evidence"
            )
            return "INCOMPLETE", evidence
        if unbound_results:
            validator_names = list(
                dict.fromkeys(result.validator_name or "Tier 3 validator" for result in unbound_results)
            )
            return "INCOMPLETE", f"Missing canonical source identity from {', '.join(validator_names)}"
        if tier3.status in {"pass", "neutral"}:
            return tier3.status.upper(), (f"{len(_agents(ae))} agent(s); {_dataset_summary(ae)['total_tasks']} task(s)")

    status = "PASSED WITH OBSERVATIONS" if findings else "PASSED"
    return status, f"{len(results)} validator(s); {len(findings)} finding(s)"


def _results_without_execution_evidence(
    results: list[ValidationResult],
) -> list[ValidationResult]:
    """Return non-skipped results that do not prove a validator ran."""
    return [
        result for result in results if not is_cleanly_skipped(result) and not result_has_execution_evidence(result)
    ]


def _tier2_results_without_execution_evidence(
    results: list[ValidationResult],
) -> list[ValidationResult]:
    """Backward-compatible alias for the generalized evidence check."""
    return _results_without_execution_evidence(results)


def _skip_reasons(results: list[ValidationResult]) -> list[str]:
    return list(dict.fromkeys(get_skip_reason(result) for result in results))


def _metric_signals(ae: dict[str, Any] | None) -> list[str]:
    seen: set[str] = set()
    signals: list[str] = []
    for agent in _agents(ae).values():
        evaluators = agent.get("evaluators")
        if not isinstance(evaluators, dict):
            continue
        for name, values in evaluators.items():
            safe_name = _safe_scalar_text(name).strip()
            if not safe_name or safe_name in seen or not isinstance(values, dict):
                continue
            if any(values.get(field) is not None for field in ("with_skill", "baseline", "lift")):
                seen.add(safe_name)
                signals.append(safe_name)
    if signals:
        return signals
    metric_ids = (ae or {}).get("metric_ids")
    return (
        [safe for item in metric_ids if (safe := _safe_scalar_text(item).strip())]
        if isinstance(metric_ids, list)
        else []
    )


def _metric_labels(ae: dict[str, Any] | None) -> dict[str, str]:
    labels = (ae or {}).get("metric_labels")
    return labels if isinstance(labels, dict) else {}


def _weighted_signals(config: dict[str, Any]) -> str:
    evaluators = list(config.get("evaluators") or [])
    weights = list(config.get("weights") or [])
    parts = []
    for evaluator, weight in zip(evaluators, weights, strict=False):
        parts.append(f"`{evaluator}` ({float(weight):.0%})")
    return " + ".join(parts) or "Not configured"


def _tier1_results(results: list[ValidationResult]) -> list[ValidationResult]:
    return [result for result in results if not _is_tier3(result) and not _is_tier2(result)]


def _tier2_results(results: list[ValidationResult]) -> list[ValidationResult]:
    return [result for result in results if _is_tier2(result)]


def _tier3_results(results: list[ValidationResult]) -> list[ValidationResult]:
    return [result for result in results if _is_tier3(result)]


def _is_tier2(result: ValidationResult) -> bool:
    producer = result_publication_evidence(result)
    if producer is not None:
        return producer.tier == 2
    return is_tier2_result(result)


def _is_tier3(result: ValidationResult) -> bool:
    return is_tier3_result(result)


def _top_findings(findings: list[Finding], *, limit: int) -> list[Finding]:
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    return sorted(findings, key=lambda finding: (order.get(finding.severity.value, 99), finding.category))[:limit]


def _finding_line(finding: Finding, private_labels: tuple[str, ...] = ()) -> str:
    location = f" (`{_publication_safe_location(finding)}`)" if finding.file_path else ""
    category = _publication_safe_inline(finding.category, private_labels)
    check_name = _publication_safe_inline(finding.check_name, private_labels)
    message = _publication_safe_inline(finding.message, private_labels)
    return f"- **{finding.severity.value.upper()}** {category}/{check_name}: {message}{location}"


def _md_cell(value: object, private_labels: tuple[str, ...] = ()) -> str:
    return _publication_safe_inline(value, private_labels).replace("|", "\\|")


def _trusted_md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _publication_safe_skill_name(value: object) -> str:
    """Return an exact NFC target identity or a non-injectable fallback."""
    candidate = _exact_publication_text(value)
    if not publication_identity_present(candidate):
        return "skill"
    if _RETIRED_PRODUCT_NAME.search(publication_semantic_text(candidate, strip_marks=True)):
        return "SkillEvaluator"
    return _publication_safe_inline(candidate, preserve_exact_nfc=True)


def _private_environment_labels(ae: dict[str, Any] | None) -> tuple[str, ...]:
    """Return imported non-public environment labels that must not escape in free text."""
    if not ae:
        return ()
    summary = _mapping(ae.get("summary"))
    candidates = [
        summary.get("environment"),
        summary.get("requested_environment"),
        ae.get("environment"),
        ae.get("requested_environment"),
    ]
    labels: list[str] = []
    for value in candidates:
        if not publication_identity_present(value):
            continue
        label = _normalized_publication_text(value)
        if publication_identity_present(label) and label.casefold() not in HARBOR_ENV_MODES and label not in labels:
            labels.append(label)
    return tuple(labels)


def _publication_safe_label(value: object, private_labels: tuple[str, ...] = ()) -> str:
    """Sanitize a classified display label for a public benchmark card."""
    label = _normalized_publication_text(value)
    is_private_label = any(label.casefold() == private_label.casefold() for private_label in private_labels)
    matching_label = publication_semantic_text(label, strip_marks=True)
    if not is_private_label and _RETIRED_PRODUCT_NAME.fullmatch(matching_label):
        label = "SkillEvaluator"
    return _publication_safe_inline(label, private_labels)


def _publication_safe_inline(
    value: object,
    private_labels: tuple[str, ...] = (),
    *,
    preserve_exact_nfc: bool = False,
) -> str:
    """Render untrusted metadata as one publication-safe Markdown line."""
    text = _exact_publication_text(value) if preserve_exact_nfc else _normalized_publication_text(value)
    text = _redact_absolute_paths(text)
    text = _RETIRED_SANDBOX_REFERENCE.sub("isolated sandbox", text)
    for label in sorted(private_labels, key=len, reverse=True):
        text = re.sub(
            rf"(?<![^\W_]){re.escape(label)}(?![^\W_])",
            "Isolated sandbox",
            text,
            flags=re.IGNORECASE,
        )
    text = _replace_semantic_tokens(text, _RETIRED_SANDBOX_REFERENCE, "isolated sandbox")
    text = _replace_retired_product_tokens(text)
    text = _LEGACY_AMBIGUOUS_UPLIFT.sub(r"\g<score> [change \g<change>]", text)
    text = _LEGACY_NUM_HEADER.sub("| Dimension | Count |", text)
    text = text.replace("&", "&amp;").replace("`", "'").replace("<", "&lt;").replace(">", "&gt;")
    text = _PUBLICATION_URL_SCHEME.sub(lambda match: f"{match.group('scheme')}&#58;//", text)
    text = _PUBLICATION_WWW_PREFIX.sub(lambda match: f"{match.group(0)[:-1]}&#46;", text)
    text = text.replace("@", "&#64;")
    text = _MARKDOWN_INLINE_SPECIAL.sub(r"\\\1", text)
    if _MARKDOWN_BLOCK_PREFIX.match(text) or _MARKDOWN_THEMATIC_BREAK.fullmatch(text):
        marker_end = text.find(" ")
        marker_end = len(text) if marker_end < 0 else marker_end
        if text[:marker_end].rstrip(".)").isdigit():
            punctuation_index = marker_end - 1
            return f"{text[:punctuation_index]}\\{text[punctuation_index:]}"
        return f"\\{text}"
    return text


def _normalized_publication_text(value: object) -> str:
    """Canonicalize text and discard invisible controls before publication decisions."""
    return " ".join(publication_semantic_text(value).split())


def _exact_publication_text(value: object) -> str:
    """Preserve a filesystem identity only when its exact NFC text is safe."""
    if not isinstance(value, str):
        return ""
    text = value.encode("utf-8", errors="replace").decode("utf-8")
    if text != value:
        return ""
    text = unicodedata.normalize("NFC", text)
    if publication_semantic_text(text) != unicodedata.normalize("NFKC", text):
        return ""
    return text


def _replace_retired_product_tokens(value: str) -> str:
    """Replace retired identity tokens even when Unicode marks split the spelling."""
    return _replace_semantic_tokens(value, _RETIRED_PRODUCT_TOKEN, "SkillEvaluator")


def _replace_semantic_tokens(value: str, pattern: re.Pattern[str], replacement: str) -> str:
    """Replace match-only normalized tokens with one linear source reconstruction."""
    searchable: list[str] = []
    source_offsets: list[int] = []
    for index, character in enumerate(value):
        for semantic_character in publication_semantic_text(character, strip_marks=True):
            searchable.append(semantic_character)
            source_offsets.append(index)
    matches = list(pattern.finditer("".join(searchable)))
    if not matches:
        return value

    # Reconstruct once. Repeated whole-string slicing here is quadratic for
    # dense untrusted metadata such as thousands of adjacent retired tokens.
    parts: list[str] = []
    cursor = 0
    for match in matches:
        start = source_offsets[match.start()]
        end = source_offsets[match.end() - 1] + 1
        if start < cursor:
            continue
        parts.extend((value[cursor:start], replacement))
        cursor = end
    parts.append(value[cursor:])
    return "".join(parts)


def _redact_absolute_paths(value: str) -> str:
    """Reduce absolute POSIX and Windows paths embedded in free text to basenames."""

    def redact_quoted_file_uri(match: re.Match[str]) -> str:
        basename = _absolute_path_basename(match.group("path"))
        return f"{match.group('quote')}{basename}{match.group('quote')}" if basename else match.group(0)

    def redact_file_uri(match: re.Match[str]) -> str:
        candidate = match.group("path")
        core = candidate.rstrip(_TRAILING_PATH_PUNCTUATION)
        suffix = candidate[len(core) :]
        basename = _absolute_path_basename(core)
        return f"{basename}{suffix}" if basename else match.group(0)

    def redact_quoted(match: re.Match[str]) -> str:
        path = match.group("path")
        basename = _absolute_path_basename(path)
        return f"{match.group('quote')}{basename}{match.group('quote')}" if basename else match.group(0)

    def redact_public_user_path(match: re.Match[str]) -> str:
        candidate = match.group("path")
        core = candidate.rstrip(_TRAILING_PATH_PUNCTUATION)
        suffix = candidate[len(core) :]
        basename = _absolute_path_basename(core)
        if not basename:
            return match.group(0)
        is_drive_path = len(core) >= 2 and core[1] == ":"
        preceding = value[match.start("path") - 1] if match.start("path") > 0 else ""
        preserve_separator = (
            not is_drive_path
            and candidate[0] in _POSIX_PATH_SEPARATORS
            and (preceding.isalnum() or preceding in {":", "/"})
        )
        separator = candidate[0] if preserve_separator else ""
        return f"{separator}{basename}{suffix}"

    text = _QUOTED_FILE_URI_PATH.sub(redact_quoted_file_uri, value)
    text = _FILE_URI_PATH.sub(redact_file_uri, text)
    text = _PUBLIC_USER_PATH.sub(redact_public_user_path, text)
    text = _QUOTED_ABSOLUTE_PATH.sub(redact_quoted, text)
    tokens: list[str] = []
    for token in text.split(" "):
        match = _PATH_START.search(token)
        if not match:
            tokens.append(token)
            continue
        prefix = token[: match.start()]
        candidate = token[match.start() :]
        core = candidate.rstrip(_TRAILING_PATH_PUNCTUATION)
        suffix = candidate[len(core) :]
        basename = _absolute_path_basename(core)
        tokens.append(f"{prefix}{basename}{suffix}" if basename else token)
    return " ".join(tokens)


def _absolute_path_basename(value: str) -> str | None:
    canonical_value = value.translate(_PATH_SEPARATOR_TRANSLATION)
    posix_path = PurePosixPath(canonical_value)
    windows_path = PureWindowsPath(canonical_value)
    if posix_path.is_absolute() and not canonical_value.startswith("//"):
        return posix_path.name or "redacted-path"
    if windows_path.is_absolute() or windows_path.root:
        return windows_path.name or "redacted-path"
    return None


def _publication_safe_environment(value: object) -> str:
    environment = _normalized_publication_text(value)
    return environment if environment.casefold() in HARBOR_ENV_MODES else "Isolated sandbox"


def _publication_safe_location(finding: Finding) -> str:
    file_path = str(finding.file_path)
    posix_path = PurePosixPath(file_path)
    windows_path = PureWindowsPath(file_path)
    if posix_path.is_absolute():
        file_path = posix_path.name
    elif windows_path.is_absolute() or windows_path.root:
        file_path = windows_path.name
    if finding.line_number:
        file_path += f":{finding.line_number}"
    return _publication_safe_inline(file_path)
