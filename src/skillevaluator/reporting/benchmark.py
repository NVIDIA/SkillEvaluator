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
_INTERNAL_SANDBOX_NAME = re.compile(r"\Aastra(?:[\s_-]+sandbox)?\Z", flags=re.IGNORECASE)
_INTERNAL_SANDBOX_REFERENCE = re.compile(r"\bastra[\s_-]+sandbox\b", flags=re.IGNORECASE)
_WINDOWS_USER_HOME = re.compile(r"\b[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s`]+", flags=re.IGNORECASE)
_POSIX_USER_HOME = re.compile(r"/(?:Users|home)/[^/\s`]+")


class BenchmarkReporter(ReporterBase):
    """Render a stable, publication-oriented ``BENCHMARK.md`` card."""

    def __init__(self, *, include_timestamp: bool = True, max_findings_shown: int = 5) -> None:
        self.include_timestamp = include_timestamp
        self.max_findings_shown = max_findings_shown

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
        skill_name = _skill_name(results, ae)
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
                        "then rerun Skill Evaluator."
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

        self._render_metadata(lines, results, ae, skill_name, policy)
        self._render_report_purpose(lines)
        self._render_results_at_a_glance(lines, ae)
        self._render_tier_status(lines, results, ae)
        self._render_findings(lines, results)
        self._render_methodology(lines, ae)
        self._render_freshness(lines)

        return _publication_safe_text("\n".join(lines).rstrip() + "\n")

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
    ) -> None:
        lines.extend(["## Evaluation Metadata", "", f"- Skill: `{skill_name}`"])

        evaluated_at = _evaluated_at(ae)
        lines.append(
            f"- Evaluation date: {_evaluation_date(evaluated_at)}"
            if evaluated_at
            else "- Evaluation date: not recorded (legacy or non-live result)"
        )

        version = (ae or {}).get("evaluator_version") or ((ae or {}).get("summary") or {}).get("evaluator_version")
        lines.append(
            f"- Evaluator version: `{version}`"
            if version
            else "- Evaluator version: not recorded (legacy or non-live result)"
        )

        agents = _agents(ae)
        if agents:
            lines.append("- Agents: " + ", ".join(_agent_label(name, agent) for name, agent in agents.items()))
        else:
            requested = (ae or {}).get("requested_agents") or []
            if requested:
                labels = ", ".join(f"{agent} (model not recorded)" for agent in map(str, requested))
                lines.append("- Agents: requested but not run — " + labels)
            else:
                lines.append("- Agents: not recorded (legacy or non-live result)")

        dataset_summary = _dataset_summary(ae)
        if dataset_summary["total_tasks"] > 0:
            composition = _dataset_composition_label(dataset_summary)
            lines.append(f"- Tasks: {dataset_summary['total_tasks']} evaluation tasks{composition}")
        else:
            lines.append("- Tasks: not recorded (legacy or non-live result)")

        digest = (ae or {}).get("dataset_digest") or ((ae or {}).get("summary") or {}).get("dataset_digest")
        digest_algorithm = (ae or {}).get("dataset_digest_algorithm") or ((ae or {}).get("summary") or {}).get(
            "dataset_digest_algorithm"
        )
        if digest:
            algorithm_label = f" ({digest_algorithm})" if digest_algorithm else ""
            lines.append(f"- Dataset digest: `{digest}`{algorithm_label}")
        else:
            lines.append("- Dataset digest: not recorded (legacy or non-live result)")

        policy = (ae or {}).get("attempt_policy") or {}
        attempts = policy.get("max_attempts")
        if attempts is not None:
            lines.append(f"- Attempts per task: {attempts}")
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
            lines.append(f"- Tier 3 live evaluation: SKIPPED — {skip_message}")

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
    def _render_results_at_a_glance(lines: list[str], ae: dict[str, Any] | None) -> None:
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

        headers = ["Measure", *[_agent_table_label(name, agent) for name, agent in agents.items()]]
        lines.append("| " + " | ".join(_md_cell(header) for header in headers) + " |")
        lines.append("|---|" + "|".join(["---:"] * len(agents)) + "|")

        overall_row = ["Overall"]
        overall_row.extend(
            _score_transition_values(
                _number(agent.get("baseline")),
                _number(agent.get("with_skill", agent.get("overall_score"))),
            )
            for agent in agents.values()
        )
        lines.append("| " + " | ".join(_md_cell(value) for value in overall_row) + " |")

        has_partial = False
        for dim_id in DIMENSION_MAPPING:
            row = [dim_id.title()]
            for agent in agents.values():
                dimension = _agent_dimension(agent, dim_id)
                if dimension and dimension.get("partial"):
                    has_partial = True
                row.append(_score_transition(dimension))
            lines.append("| " + " | ".join(_md_cell(value) for value in row) + " |")

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
            status, evidence = _tier_status(tier, tier_results, ae)
            lines.append(f"| {tier} | {purpose} | **{status}** | {_md_cell(evidence)} |")
        lines.append("")

        for tier, _purpose, tier_results in tier_groups:
            if tier_results and all(_result_skipped(result) for result in tier_results):
                lines.append(f"{tier} validation was skipped and executed 0 checks.")
                for reason in _skip_reasons(tier_results):
                    lines.append(f"- {reason}")
                lines.append("")

    def _render_findings(self, lines: list[str], results: list[ValidationResult]) -> None:
        findings_with_result = [(finding, result) for result in results for finding in result.findings]
        blocking = [
            finding
            for finding, result in findings_with_result
            if not result.passed or finding.severity.value in {"critical", "high"}
        ]

        if blocking:
            lines.extend(["## Blocking Findings", ""])
            for finding in _top_findings(blocking, limit=self.max_findings_shown):
                lines.append(_finding_line(finding))
            lines.append("")

        static_test_limitations = list(
            dict.fromkeys(
                message for result in results if (message := self._static_test_evidence_message(result)) is not None
            )
        )
        if static_test_limitations:
            lines.extend(["Test execution limitations:", ""])
            lines.extend(f"- {message}" for message in static_test_limitations)
            lines.append("")

        lines.extend(["## Findings and Observations", ""])
        lines.extend(["<details>", "<summary>Show detailed findings and successful checks</summary>", ""])

        findings = [finding for finding, _result in findings_with_result]
        if findings:
            for finding in _top_findings(findings, limit=self.max_findings_shown):
                lines.append(_finding_line(finding))
            remaining = len(findings) - min(len(findings), self.max_findings_shown)
            if remaining > 0:
                lines.append(f"- {remaining} additional finding(s) are available in the full evaluation artifacts.")
        else:
            observations = [
                f"- {result.validator_name}: {detail.message}"
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
    def _render_methodology(lines: list[str], ae: dict[str, Any] | None) -> None:
        policy = (ae or {}).get("verdict_policy") or {}
        attempt_policy = (ae or {}).get("attempt_policy") or {}
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
            lines.append(f"| {dim_id.title()} | {_md_cell(question)} | {_md_cell(signals)} |")

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
                label = labels.get(signal, signal.replace("_", " ").title())
                description = _SIGNAL_DESCRIPTIONS.get(signal, "additional evaluator signal")
                lines.append(f"- `{signal}` ({label}): {description}.")
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


def _benchmark_policy(
    results: list[ValidationResult],
    ae: dict[str, Any] | None,
) -> dict[str, bool]:
    """Resolve the persisted publication policy, defaulting Tier 3 to required."""
    candidates: list[object] = [
        (ae or {}).get("benchmark_policy"),
        ((ae or {}).get("summary") or {}).get("benchmark_policy"),
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
    """Require an actual supported-agent verdict, with legacy compatibility."""
    if not isinstance(ae, dict) or not _agents(ae):
        return False
    verdict = str(ae.get("verdict") or (ae.get("summary") or {}).get("verdict") or "").lower()
    if verdict not in {"pass", "neutral", "fail"}:
        return False
    execution_status = str(
        ae.get("execution_status") or (ae.get("summary") or {}).get("execution_status") or ""
    ).lower()
    # Historical payloads predate execution_status. Their recorded agents and
    # verdict are the strongest available evidence and remain renderable.
    return execution_status in {"", "succeeded"}


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
    verdict = str((ae or {}).get("verdict") or ((ae or {}).get("summary") or {}).get("verdict") or "").lower()
    if has_failures or verdict == "fail":
        return "FAIL"

    if (
        benchmark_policy["tier3_required"]
        and not _tier3_evidence_complete(ae)
        and not _advisory_agent_eval_skip_message(results)
    ):
        return "INCOMPLETE"

    execution_status = str(
        (ae or {}).get("execution_status") or ((ae or {}).get("summary") or {}).get("execution_status") or ""
    ).lower()
    if verdict == "neutral" and execution_status in {"succeeded", ""}:
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
        summary = ae.get("summary") or {}
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


def _agent_label(name: str, agent: dict[str, Any]) -> str:
    display = _human_agent_name(str(agent.get("display_name") or agent.get("label") or name))
    model = agent.get("model") or agent.get("model_name") or agent.get("llm_model")
    return f"{display} (`{model}`)" if model else f"{display} (model not recorded)"


def _agent_table_label(name: str, agent: dict[str, Any]) -> str:
    display = _human_agent_name(str(agent.get("display_name") or agent.get("label") or name))
    return f"{display} (Baseline → Skill Uplift)"


def _human_agent_name(name: str) -> str:
    if name == "claude-code":
        return "Claude Code"
    return name.replace("_", " ").replace("-", " ").title()


def _evaluated_at(ae: dict[str, Any] | None) -> str | None:
    value = (ae or {}).get("evaluated_at") or ((ae or {}).get("summary") or {}).get("evaluated_at")
    return str(value).strip() if value else None


def _evaluation_date(value: str) -> str:
    candidate = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate).date().isoformat()
    except ValueError:
        return value[:10] if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", value) else value


def _environment(ae: dict[str, Any] | None) -> str | None:
    summary = (ae or {}).get("summary") or {}
    value = summary.get("environment") or (ae or {}).get("environment") or (ae or {}).get("requested_environment")
    return str(value) if value else None


def _environment_note(environment: str | None) -> str | None:
    if not environment:
        return None
    lowered = environment.lower().replace("_", "-")
    if _INTERNAL_SANDBOX_NAME.fullmatch(environment):
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
        execution = str(ae.get("execution_status") or (ae.get("summary") or {}).get("execution_status") or "")
        verdict = str(ae.get("verdict") or (ae.get("summary") or {}).get("verdict") or "").upper()
        if execution and execution != "succeeded":
            return "INCOMPLETE", f"Execution status: {execution}"
        if verdict in {"PASS", "NEUTRAL", "FAIL"}:
            return verdict, f"{len(_agents(ae))} agent(s); {_dataset_summary(ae)['total_tasks']} task(s)"

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


def _finding_line(finding: Finding) -> str:
    location = f" (`{_publication_safe_location(finding)}`)" if finding.file_path else ""
    return (
        f"- **{finding.severity.value.upper()}** {finding.category}/{finding.check_name}: {finding.message}{location}"
    )


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _publication_safe_text(value: object) -> str:
    branded = _RETIRED_PRODUCT_NAME.sub("Skill Evaluator", str(value))
    isolated = _INTERNAL_SANDBOX_REFERENCE.sub("isolated sandbox", branded)
    windows_safe = _WINDOWS_USER_HOME.sub("~", isolated)
    return _POSIX_USER_HOME.sub("~", windows_safe)


def _publication_safe_environment(value: object) -> str:
    environment = str(value).strip()
    return environment if environment.casefold() in HARBOR_ENV_MODES else "Isolated sandbox"


def _publication_safe_location(finding: Finding) -> str:
    file_path = str(finding.file_path)
    posix_path = PurePosixPath(file_path)
    windows_path = PureWindowsPath(file_path)
    if posix_path.is_absolute():
        file_path = posix_path.name
    elif windows_path.is_absolute():
        file_path = windows_path.name
    if finding.line_number:
        file_path += f":{finding.line_number}"
    return file_path
