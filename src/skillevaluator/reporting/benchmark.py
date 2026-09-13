# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publication-ready BENCHMARK.md reporter for skill evaluation cards."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any

from skillevaluator.constants import (
    DIMENSION_HINTS,
    DIMENSION_MAPPING,
    DIMENSION_VERDICT_NEUTRAL_THRESHOLD,
    DIMENSION_VERDICT_PASS_THRESHOLD,
    KEBAB_CASE_PATTERN,
    TIER3_LIFT_FAIL_THRESHOLD,
    TIER3_LIFT_PASS_THRESHOLD,
)
from skillevaluator.reporting.base import ReporterBase, is_advisory_agent_eval_skip, passes_required_gate
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

_TIER2_VALIDATORS = {
    "context deduplication",
    "intra-skill deduplication",
}

_RETIRED_PRODUCT_NAME = re.compile(r"\b[a-z]*[\s_-]*skills[\s_-]*eval\b", flags=re.IGNORECASE)
_RETIRED_SANDBOX_REFERENCE = re.compile(
    rf"\b{re.escape(chr(97) + 'stra')}[\s_-]+sandbox\b",
    flags=re.IGNORECASE,
)
_PATH_START = re.compile(r"(?<![A-Za-z0-9:/])(?:[A-Za-z]:[\\/]|\\\\|\\|/)")
_QUOTED_ABSOLUTE_PATH = re.compile(r"(?P<quote>['\"])(?P<path>(?:[A-Za-z]:[\\/]|\\\\|\\|/)[^'\"\r\n]+)(?P=quote)")
_QUOTED_FILE_URI_PATH = re.compile(
    r"(?P<quote>['\"])(?:file:)(?://[^/'\"\r\n]*)?(?P<path>/[^'\"\r\n]+)(?P=quote)",
    flags=re.IGNORECASE,
)
_FILE_URI_PATH = re.compile(
    r"\bfile:(?://[^/\s'\"<>]*)?(?P<path>/[^\s'\"<>]+)",
    flags=re.IGNORECASE,
)
_MARKDOWN_INLINE_SPECIAL = re.compile(r"([\\*_\[\]~])")
_MARKDOWN_BLOCK_PREFIX = re.compile(r"^(?:#{1,6}|>|[+*-]|\d+[.)])(?=\s|$)")
_MARKDOWN_THEMATIC_BREAK = re.compile(r"^(?:\s*[-*_]){3,}\s*$")
_PUBLICATION_URL_SCHEME = re.compile(r"(?P<scheme>https?|ftp)://", flags=re.IGNORECASE)
_PUBLICATION_WWW_PREFIX = re.compile(r"\bwww\.", flags=re.IGNORECASE)
_TRAILING_PATH_PUNCTUATION = ".,;!?)]}>`'\""


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
        skill_name = _publication_safe_skill_name(self.skill_name or _skill_name(results, ae))
        private_labels = _private_environment_labels(ae)
        policy = _benchmark_policy(results, ae)
        status = _overall_status(results, ae, policy)

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
        )
        self._render_report_purpose(lines)
        self._render_results_at_a_glance(lines, ae, private_labels)
        self._render_tier_status(lines, results, ae, policy, private_labels)
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
    ) -> None:
        lines.extend(["## Evaluation Metadata", "", f"- Skill: `{skill_name}`"])

        evaluated_at = _evaluated_at(ae)
        lines.append(
            f"- Evaluation date: {_publication_safe_inline(_evaluation_date(evaluated_at), private_labels)}"
            if evaluated_at
            else "- Evaluation date: not recorded (legacy or non-live result)"
        )

        summary = _mapping((ae or {}).get("summary"))
        version = (ae or {}).get("evaluator_version") or summary.get("evaluator_version")
        lines.append(
            f"- Evaluator version: `{_publication_safe_inline(version, private_labels)}`"
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
                    f"{_human_agent_name(_publication_safe_label(agent, private_labels))} (model not recorded)"
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

        digest = (ae or {}).get("dataset_digest") or summary.get("dataset_digest")
        digest_algorithm = (ae or {}).get("dataset_digest_algorithm") or summary.get("dataset_digest_algorithm")
        if digest:
            safe_digest = _publication_safe_inline(digest, private_labels)
            algorithm_label = (
                f" ({_publication_safe_inline(digest_algorithm, private_labels)})" if digest_algorithm else ""
            )
            lines.append(f"- Dataset digest: `{safe_digest}`{algorithm_label}")
        else:
            lines.append("- Dataset digest: not recorded (legacy or non-live result)")

        policy = _mapping((ae or {}).get("attempt_policy"))
        attempts = policy.get("max_attempts")
        if attempts is not None:
            lines.append(f"- Attempts per task: {_publication_safe_inline(attempts, private_labels)}")
        else:
            lines.append("- Attempts per task: not recorded (legacy or non-live result)")

        environment = _environment(ae)
        if environment:
            lines.append(f"- Environment: `{_publication_safe_environment(environment)}`")
        else:
            lines.append("- Environment: not recorded (legacy or non-live result)")

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
    ) -> None:
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
            status, evidence = _tier_status(tier, tier_results, ae, benchmark_policy)
            lines.append(f"| {tier} | {purpose} | **{status}** | {_md_cell(evidence, private_labels)} |")
        lines.append("")

        for tier, _purpose, tier_results in tier_groups:
            if tier_results and all(_result_skipped(result) for result in tier_results):
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
    for result in results:
        payload = result.metadata.get("agent_eval") if isinstance(result.metadata, dict) else None
        if isinstance(payload, dict):
            return payload
    return None


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _benchmark_policy(
    results: list[ValidationResult],
    ae: dict[str, Any] | None,
) -> dict[str, bool]:
    """Resolve the persisted publication policy, defaulting Tier 3 to required."""
    candidates: list[object] = [
        (ae or {}).get("benchmark_policy"),
        _mapping((ae or {}).get("summary")).get("benchmark_policy"),
    ]
    candidates.extend(
        result.metadata.get("benchmark_policy") for result in results if isinstance(result.metadata, dict)
    )
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        required = candidate.get("tier3_required")
        if isinstance(required, bool):
            return {"tier3_required": required}
    return {"tier3_required": True}


def _tier3_evidence_complete(ae: dict[str, Any] | None) -> bool:
    """Require a succeeded run with the minimum publication provenance."""
    if not isinstance(ae, dict):
        return False
    agents = _agents(ae)
    if not agents or any(
        not str(agent.get("model") or agent.get("model_name") or agent.get("llm_model") or "").strip()
        for agent in agents.values()
    ):
        return False
    summary = _mapping(ae.get("summary"))
    verdict = str(ae.get("verdict") or summary.get("verdict") or "").lower()
    if verdict not in {"pass", "neutral", "fail"}:
        return False
    execution_status = str(ae.get("execution_status") or summary.get("execution_status") or "").lower()
    evaluated_at = ae.get("evaluated_at") or summary.get("evaluated_at")
    evaluator_version = ae.get("evaluator_version") or summary.get("evaluator_version")
    dataset_digest = ae.get("dataset_digest") or summary.get("dataset_digest")
    attempt_policy = _mapping(ae.get("attempt_policy"))
    attempts = _nonnegative_int(attempt_policy.get("max_attempts"))
    environment = summary.get("environment") or ae.get("environment")
    return bool(
        execution_status == "succeeded"
        and _tier3_dimension_verdict(ae) is not None
        and str(evaluated_at or "").strip()
        and str(evaluator_version or "").strip()
        and str(dataset_digest or "").strip()
        and _dataset_summary(ae)["total_tasks"] > 0
        and attempts > 0
        and str(environment or "").strip()
    )


def _tier3_dimension_verdict(ae: dict[str, Any] | None) -> str | None:
    """Recompute the canonical verdict from every supported agent's dimensions."""
    supported_agents = [agent for agent in _agents(ae).values() if agent.get("execution_status") == "succeeded"]
    if not supported_agents:
        return None

    agent_verdicts: list[str] = []
    has_partial_evidence = False
    for agent in supported_agents:
        scores = _agent_dimension_scores(agent)
        if scores is None:
            has_partial_evidence = True
            continue
        if any(score < DIMENSION_VERDICT_NEUTRAL_THRESHOLD for score in scores):
            agent_verdicts.append("fail")
        elif any(score < DIMENSION_VERDICT_PASS_THRESHOLD for score in scores):
            agent_verdicts.append("neutral")
        else:
            agent_verdicts.append("pass")

    if "pass" in agent_verdicts:
        return "pass"
    if has_partial_evidence or not agent_verdicts:
        return None
    if "neutral" in agent_verdicts:
        return "neutral"
    return "fail"


def _agent_dimension_scores(agent: dict[str, Any]) -> list[float] | None:
    """Return all configured in-range dimension scores, rejecting partial evidence."""
    raw_dimensions = agent.get("dimensions")
    if not isinstance(raw_dimensions, list):
        return None
    dimensions: dict[str, dict[str, Any]] = {}
    for dimension in raw_dimensions:
        if not isinstance(dimension, dict):
            continue
        dimension_id = str(dimension.get("id") or "")
        if dimension_id in dimensions:
            return None
        dimensions[dimension_id] = dimension

    scores: list[float] = []
    for dimension_id in DIMENSION_MAPPING:
        dimension = dimensions.get(dimension_id)
        if dimension is None:
            return None
        value = dimension.get("with_skill") if "with_skill" in dimension else dimension.get("score")
        score = _number(value)
        if score is None or not 0.0 <= score <= 1.0:
            return None
        scores.append(score)
    return scores


def _advisory_agent_eval_skip_message(results: list[ValidationResult]) -> str | None:
    for result in results:
        if not is_advisory_agent_eval_skip(result):
            continue
        payload = result.metadata.get("agent_eval", {}) if result.metadata else {}
        provenance = payload.get("provenance", {}) if isinstance(payload, dict) else {}
        message = provenance.get("message") if isinstance(provenance, dict) else None
        return str(message or "Live evaluation did not run.")
    return None


def _result_skipped(result: ValidationResult) -> bool:
    """Return whether a result records a skipped validator run."""
    return bool(result.metadata.get("skipped")) or is_advisory_agent_eval_skip(result)


def _overall_status(
    results: list[ValidationResult],
    ae: dict[str, Any] | None,
    benchmark_policy: dict[str, bool],
) -> str:
    if any(result.is_incomplete for result in results):
        return "INCOMPLETE"

    blocking_skips = [
        result
        for result in results
        if _result_skipped(result) and not result.metadata.get("optional") and not is_advisory_agent_eval_skip(result)
    ]
    if blocking_skips:
        return "INCOMPLETE"

    has_failures = not all(passes_required_gate(result) for result in results)
    summary = _mapping((ae or {}).get("summary"))
    verdict = str((ae or {}).get("verdict") or summary.get("verdict") or "").lower()
    dimension_verdict = _tier3_dimension_verdict(ae)
    if has_failures or verdict == "fail" or dimension_verdict == "fail":
        return "FAIL"

    tier3_results = _tier3_results(results)
    has_present_tier3_result = bool(tier3_results) and not all(_result_skipped(result) for result in tier3_results)
    if (benchmark_policy["tier3_required"] or has_present_tier3_result) and not _tier3_evidence_complete(ae):
        return "INCOMPLETE"
    execution_status = str((ae or {}).get("execution_status") or summary.get("execution_status") or "").lower()
    if verdict == "neutral" and execution_status in {"succeeded", ""}:
        return "NEUTRAL"
    if verdict == "pass" and dimension_verdict == "neutral":
        return "NEUTRAL"
    return "PASS"


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
        if candidate:
            return str(candidate)
    for result in results:
        quality = result.metadata.get("quality_scores") if isinstance(result.metadata, dict) else None
        if isinstance(quality, dict) and quality.get("skill_name"):
            return str(quality["skill_name"])
    return "skill"


def _agents(ae: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    agents = (ae or {}).get("agents")
    if not isinstance(agents, dict):
        return {}
    return {str(name): agent for name, agent in agents.items() if isinstance(agent, dict)}


def _agent_label(
    name: str,
    agent: dict[str, Any],
    private_labels: tuple[str, ...] = (),
) -> str:
    display = _human_agent_name(
        _publication_safe_label(agent.get("display_name") or agent.get("label") or name, private_labels)
    )
    model = agent.get("model") or agent.get("model_name") or agent.get("llm_model")
    if model:
        return f"{display} (`{_publication_safe_label(model, private_labels)}`)"
    return f"{display} (model not recorded)"


def _agent_table_label(
    name: str,
    agent: dict[str, Any],
    private_labels: tuple[str, ...] = (),
) -> str:
    display = _human_agent_name(
        _publication_safe_label(agent.get("display_name") or agent.get("label") or name, private_labels)
    )
    return f"{display} (Baseline → Skill Uplift)"


def _human_agent_name(name: str) -> str:
    if name == "claude-code":
        return "Claude Code"
    if re.fullmatch(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*", name):
        return name.replace("_", " ").replace("-", " ").title()
    return name.title()


def _evaluated_at(ae: dict[str, Any] | None) -> str | None:
    value = (ae or {}).get("evaluated_at") or _mapping((ae or {}).get("summary")).get("evaluated_at")
    return str(value).strip() if value else None


def _evaluation_date(value: str) -> str:
    candidate = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate).date().isoformat()
    except ValueError:
        return value[:10] if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", value) else value


def _environment(ae: dict[str, Any] | None) -> str | None:
    summary = _mapping((ae or {}).get("summary"))
    value = summary.get("environment") or (ae or {}).get("environment")
    return str(value) if value else None


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
            "source": str(summary.get("source") or "payload"),
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
    for trial in (ae or {}).get("trials") or []:
        if not isinstance(trial, dict):
            continue
        for key in ("entry_id", "case_id", "task_id", "id"):
            value = trial.get(key)
            if value is not None and str(value).strip():
                task_ids.add(str(value).strip())
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
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _number(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _agent_dimension(agent: dict[str, Any], dim_id: str) -> dict[str, Any] | None:
    for dimension in agent.get("dimensions") or []:
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
) -> tuple[str, str]:
    if not results:
        return "NOT RUN", "No result was recorded"
    incomplete = [result for result in results if result.is_incomplete]
    if incomplete:
        tools = list(dict.fromkeys(tool for result in incomplete for tool in result.incomplete_scans))
        return "INCOMPLETE", f"Missing trustworthy evidence from {', '.join(tools)}"
    if all(_result_skipped(result) for result in results):
        optional = all(result.metadata.get("optional") or is_advisory_agent_eval_skip(result) for result in results)
        return ("SKIPPED (ADVISORY)" if optional else "INCOMPLETE"), "; ".join(_skip_reasons(results))

    findings = [finding for result in results for finding in result.findings]
    if any(not result.passed and not _result_skipped(result) for result in results):
        return "FAILED", f"{len(results)} validator(s); {len(findings)} finding(s)"

    if tier == "Tier 3" and ae:
        summary = _mapping(ae.get("summary"))
        execution = str(ae.get("execution_status") or summary.get("execution_status") or "").lower()
        verdict = str(ae.get("verdict") or summary.get("verdict") or "").lower()
        dimension_verdict = _tier3_dimension_verdict(ae)
        if verdict == "fail" or dimension_verdict == "fail":
            return "FAIL", f"{len(_agents(ae))} agent(s); {_dataset_summary(ae)['total_tasks']} task(s)"
        if execution and execution != "succeeded":
            return "INCOMPLETE", f"Execution status: {execution}"
        if not _tier3_evidence_complete(ae):
            evidence = (
                "Required Tier 3 evidence is missing"
                if benchmark_policy["tier3_required"]
                else "Present Tier 3 result lacks complete evidence"
            )
            return "INCOMPLETE", evidence
        effective_verdict = verdict
        if verdict == "pass" and dimension_verdict == "neutral":
            effective_verdict = "neutral"
        if effective_verdict in {"pass", "neutral"}:
            return effective_verdict.upper(), (
                f"{len(_agents(ae))} agent(s); {_dataset_summary(ae)['total_tasks']} task(s)"
            )

    status = "PASSED WITH OBSERVATIONS" if findings else "PASSED"
    return status, f"{len(results)} validator(s); {len(findings)} finding(s)"


def _skip_reasons(results: list[ValidationResult]) -> list[str]:
    reasons: list[str] = []
    for result in results:
        payload = result.metadata.get("agent_eval", {}) if result.metadata else {}
        provenance = payload.get("provenance", {}) if isinstance(payload, dict) else {}
        advisory_message = provenance.get("message") if isinstance(provenance, dict) else None
        reasons.append(str(result.metadata.get("skip_reason") or advisory_message or "Prerequisite unavailable"))
    return list(dict.fromkeys(reasons))


def _metric_signals(ae: dict[str, Any] | None) -> list[str]:
    seen: set[str] = set()
    signals: list[str] = []
    for agent in _agents(ae).values():
        evaluators = agent.get("evaluators")
        if not isinstance(evaluators, dict):
            continue
        for name, values in evaluators.items():
            if name in seen or not isinstance(values, dict):
                continue
            if any(values.get(field) is not None for field in ("with_skill", "baseline", "lift")):
                seen.add(str(name))
                signals.append(str(name))
    if signals:
        return signals
    metric_ids = (ae or {}).get("metric_ids")
    return [str(item) for item in metric_ids] if isinstance(metric_ids, list) else []


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
    return [result for result in results if not _is_tier2(result) and not _is_tier3(result)]


def _tier2_results(results: list[ValidationResult]) -> list[ValidationResult]:
    return [result for result in results if _is_tier2(result)]


def _tier3_results(results: list[ValidationResult]) -> list[ValidationResult]:
    return [result for result in results if _is_tier3(result)]


def _is_tier2(result: ValidationResult) -> bool:
    name = result.validator_name.lower()
    if name in _TIER2_VALIDATORS or "dedup" in name:
        return True
    return any(finding.category == "CONTENT_DEDUP" for finding in result.findings)


def _is_tier3(result: ValidationResult) -> bool:
    return bool(result.metadata.get("agent_eval")) or result.validator_name == "AGENT_EVAL"


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
    """Return a canonical target identity or a non-injectable public fallback."""
    candidate = " ".join(str(value).split())
    if re.fullmatch(KEBAB_CASE_PATTERN, candidate) is not None:
        return candidate
    if _RETIRED_PRODUCT_NAME.fullmatch(candidate):
        return "SkillEvaluator"
    return "skill"


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
        label = " ".join(str(value or "").split())
        if label and label.casefold() not in HARBOR_ENV_MODES and label not in labels:
            labels.append(label)
    return tuple(labels)


def _publication_safe_label(value: object, private_labels: tuple[str, ...] = ()) -> str:
    """Sanitize a classified display label and normalize only an exact retired product name."""
    label = _publication_safe_inline(value, private_labels)
    if _RETIRED_PRODUCT_NAME.fullmatch(label):
        return "SkillEvaluator"
    return label


def _publication_safe_inline(value: object, private_labels: tuple[str, ...] = ()) -> str:
    """Render untrusted metadata as one publication-safe Markdown line."""
    text = " ".join(str(value).split())
    text = _redact_absolute_paths(text)
    text = _RETIRED_SANDBOX_REFERENCE.sub("isolated sandbox", text)
    for label in sorted(private_labels, key=len, reverse=True):
        text = re.sub(
            re.escape(label),
            "Isolated sandbox",
            text,
            flags=re.IGNORECASE,
        )
    text = text.replace("`", "'").replace("<", "&lt;").replace(">", "&gt;")
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

    text = _QUOTED_FILE_URI_PATH.sub(redact_quoted_file_uri, value)
    text = _FILE_URI_PATH.sub(redact_file_uri, text)
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
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() and not value.startswith("//"):
        return posix_path.name or "redacted-path"
    if windows_path.is_absolute() or windows_path.root:
        return windows_path.name or "redacted-path"
    return None


def _publication_safe_environment(value: object) -> str:
    environment = str(value).strip()
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
