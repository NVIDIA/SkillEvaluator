# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publication-ready BENCHMARK.md reporter for skill and plugin evaluation cards."""

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
from skillevaluator.reporting.plugin_sections import (
    NOT_LOADED_STATE,
    STAGING_CAVEAT,
    number,
    tier3_plugin_view,
    unsupported_type_split,
)
from skillevaluator.source_identity import evaluated_source_revision, recorded_evaluated_source
from skillevaluator.tier3_environments import HARBOR_ENV_MODES
from skillevaluator.utils.rich_markup import strip_terminal_controls

if TYPE_CHECKING:
    from skillevaluator.models import Finding, ValidationResult


# These fields are empty because the orchestration input did not supply them,
# which is a different cause from the legacy/non-live wording used elsewhere.
_SOURCE_UNRECORDED = "not recorded (not supplied by the orchestration input)"

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
        content_type: str = "skill",
    ) -> None:
        self.include_timestamp = include_timestamp
        self.max_findings_shown = max_findings_shown
        self.skill_name = skill_name
        self.content_type = content_type

    @property
    def name(self) -> str:
        return "benchmark"

    @property
    def description(self) -> str:
        return "Publication-ready BENCHMARK.md skill evaluation card"

    def render(self, result: ValidationResult) -> str:
        return self.render_all([result])

    def render_all(self, results: list[ValidationResult]) -> str:
        # Untrusted text can carry lone surrogates (which UTF-8 cannot encode) and
        # terminal escape sequences (which run when someone prints the card).
        return strip_terminal_controls(self._render_card(results))

    def _render_card(self, results: list[ValidationResult]) -> str:
        ae = _agent_eval_payload(results)
        if self.content_type == "plugin":
            return self._render_plugin_card(results, ae)
        skill_name = _publication_safe_skill_name(self.skill_name or _skill_name(results, ae))
        private_labels = _private_environment_labels(ae)
        policy = _benchmark_policy(results, ae)
        status = _overall_status(results, ae, policy)

        advisory = status == "FAIL" and _advisory_tier3_failure(results, ae)
        lines: list[str] = [
            f"# Skill Benchmark: {skill_name}",
            "",
            _verdict_callout(status, advisory=advisory),
            "",
        ]
        if status == "PASS":
            self._render_publication_recommendation(lines, results)
        elif status == "FAIL":
            lines.extend(_fail_lines(results, ae, "skill", advisory=advisory))
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

        lines.extend(_lift_band_lines(ae, "skill"))
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
        subject_label: str = "Skill",
        extra_lines: list[str] | None = None,
    ) -> None:
        lines.extend(["## Evaluation Metadata", "", f"- {subject_label}: `{skill_name}`"])

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

        # Validated by ``normalized_evaluated_source``, so the identity is
        # published verbatim: escaping would rewrite `_` and `@` and corrupt the
        # very value the card exists to record.
        source = _evaluated_source(results)
        repository = source.get("repository", "")
        lines.append(
            f"- Evaluated source: `{repository}`" if repository else "- Evaluated source: " + _SOURCE_UNRECORDED
        )

        revision = evaluated_source_revision(source)
        lines.append(
            f"- Evaluated source revision: `{revision}`"
            if revision
            else "- Evaluated source revision: " + _SOURCE_UNRECORDED
        )

        container_revision = source.get("evaluator_container_revision", "")
        lines.append(
            f"- Evaluator container revision: `{container_revision}`"
            if container_revision
            else "- Evaluator container revision: " + _SOURCE_UNRECORDED
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
        lines.extend(extra_lines or [])

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
        subject: str = "skill",
        *,
        sum_of_parts_baseline: bool = False,
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

        # An agent whose run did not complete is INCOMPLETE: its partial scores are not comparable, so the card
        # does not print them as results (the terminal and HTML summaries leave them out too).
        incomplete = {name: _incomplete_run_note(agent, subject) for name, agent in agents.items()}
        incomplete = {name: note for name, note in incomplete.items() if note is not None}
        if incomplete:
            notes = "; ".join(
                f"{_agent_label(name, agents[name], private_labels)}: {note}" for name, note in incomplete.items()
            )
            lines.extend(
                [
                    (
                        f"**INCOMPLETE run:** Tier 3 did not complete ({notes}), so this card reports no Tier 3 "
                        "score or uplift for it. The detailed report keeps the per-trial evidence."
                    ),
                    "",
                ]
            )
            if len(incomplete) == len(agents):
                return

        headers = [
            "Measure",
            *[_agent_table_label(name, agent, private_labels, subject) for name, agent in agents.items()],
        ]
        lines.append("| " + " | ".join(_md_cell(header, private_labels) for header in headers) + " |")
        lines.append("|---|" + "|".join(["---:"] * len(agents)) + "|")

        # A missing baseline is not always "not run": an INCOMPLETE run withholds a baseline that did run.
        missing = {name: _missing_baseline_text(agent, ae) for name, agent in agents.items()}
        not_reported = "Not reported (INCOMPLETE run)"
        overall_row = ["Overall"]
        overall_row.extend(
            not_reported
            if name in incomplete
            else _overall_lift_transition(
                agent, subject, sum_of_parts_baseline=sum_of_parts_baseline, missing_baseline=missing[name]
            )
            for name, agent in agents.items()
        )
        lines.append("| " + " | ".join(_md_cell(value, private_labels) for value in overall_row) + " |")

        has_partial = False
        for dim_id in DIMENSION_MAPPING:
            row = [dim_id.title()]
            for name, agent in agents.items():
                if name in incomplete:
                    row.append(not_reported)
                    continue
                dimension = _agent_dimension(agent, dim_id)
                if dimension and dimension.get("partial"):
                    has_partial = True
                row.append(_dimension_lift_transition(agent, dimension, subject, missing_baseline=missing[name]))
            lines.append("| " + " | ".join(_md_cell(value, private_labels) for value in row) + " |")
        lines.extend(_shared_basis_note(agents, subject, sum_of_parts_baseline=sum_of_parts_baseline))

        # Lift mode integration runs no no-plugin arm: its baseline is the sum of parts.
        baseline = (
            f"the same task attempted with the {subject}'s member components staged individually "
            f"(sum of parts), not without the {subject}"
            if sum_of_parts_baseline
            else f"the same task attempted without the target {subject}"
        )
        example_baseline = "sum-of-parts" if sum_of_parts_baseline else f"no-{subject}"
        lines.extend(
            [
                "",
                (
                    f"**How to read this table:** baseline is {baseline}. "
                    f"Uplift is `{subject} score - baseline score`, shown in percentage points."
                ),
                "",
                (
                    f"Example: `47% → 92% (+45 points)` means the {subject}-assisted run scored 92%, "
                    f"45 percentage points above its 47% {example_baseline} baseline."
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
        tier3_override: tuple[str, str] | None = None,
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
            if tier == "Tier 3" and tier3_override is not None and tier_results:
                status, evidence = tier3_override
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
        blocking = [finding for finding, result in findings_with_result if _finding_blocks(finding, result)]

        if blocking:
            lines.extend(["## Blocking Findings", ""])
            shown = _top_findings(blocking, limit=self.max_findings_shown)
            for finding in shown:
                lines.append(_finding_line(finding, private_labels))
            if len(blocking) > len(shown):
                # Never cut the list silently: say how many blocking rows are not shown.
                lines.append(
                    f"- {len(blocking) - len(shown)} more blocking finding(s) are in the full evaluation artifacts."
                )
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
        attempt_threshold = number(policy.get("attempt_pass_threshold", attempt_policy.get("pass_threshold")))
        dimension_pass = number(policy.get("dimension_pass_threshold")) or DIMENSION_VERDICT_PASS_THRESHOLD
        dimension_neutral = (
            number(policy.get("dimension_neutral_threshold"))
            if policy.get("dimension_neutral_threshold") is not None
            else DIMENSION_VERDICT_NEUTRAL_THRESHOLD
        )
        lift_pass = number(policy.get("lift_pass_threshold"))
        lift_fail = number(policy.get("lift_fail_threshold"))
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
                    f"- Skill Lift band (with versus without, best agent): PASS at +{lift_pass * 100:.0f} points "
                    f"or more; FAIL at {lift_fail * 100:.0f} points or less; values between those bands are NEUTRAL."
                ),
                (
                    "- Overall verdict: PASS only when every configured dimension passes for at least one "
                    "supported agent. The Skill Lift band never changes the Tier 3 dimension verdict."
                ),
                (
                    "- A Skill Lift in the FAIL band whose paired-case interval lies wholly below zero is a "
                    "confirmed regression: it adds a warning and makes this card FAIL. A FAIL-band lift whose "
                    "interval includes zero, or has no interval, only warns. Integration-only runs have no "
                    "Skill Lift band."
                ),
                (
                    "- Exit code: `validate --block-on-agent-eval` fails on a Tier 3 FAIL verdict, a confirmed "
                    "regression, or a skipped or INCOMPLETE Tier 3 run; a NEUTRAL verdict never fails it. "
                    "Without the flag, Tier 3 is advisory and never changes the exit code."
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
    def _render_freshness(lines: list[str], subject: str = "skill") -> None:
        changed = "plugin or any of its components" if subject == "plugin" else "skill"
        lines.extend(
            [
                "## Freshness",
                "",
                (
                    f"Regenerate this benchmark when the {changed}, evaluation dataset, target agent/model, "
                    "evaluator version, environment, or scoring policy changes."
                ),
                "",
            ]
        )

    # ------------------------------------------------------------------
    # Plugin card
    # ------------------------------------------------------------------

    def _render_plugin_card(self, results: list[ValidationResult], ae: dict[str, Any] | None) -> str:
        """Render the plugin card: the skill card plus coverage, Integration, and exclusions."""
        plugin = self._plugin_block_from_results(results) or {}
        view = tier3_plugin_view(ae)
        name = _publication_safe_target_name(
            (plugin.get("name"), (view or {}).get("plugin_name"), self.skill_name),
            fallback="plugin",
        )
        private_labels = _private_environment_labels(ae)
        policy = _benchmark_policy(results, ae)
        partial = bool(view and view["partial"])
        status = _plugin_overall_status(results, ae, policy, partial=partial)

        advisory = status == "FAIL" and _advisory_tier3_failure(results, ae)
        lines: list[str] = [f"# Plugin Benchmark: {name}", "", _verdict_callout(status, advisory=advisory), ""]
        if status == "PASS":
            lines.extend(["## Publication Recommendation", ""])
            if _advisory_agent_eval_skip_message(results):
                lines.append(
                    "Tier 3 live evaluation was skipped and does not block required validation. "
                    "Publication suitability in this report is based on the completed required-tier "
                    "results; rerun Tier 3 when the live evaluation runtime is available."
                )
            else:
                lines.append(
                    "Recommended for publication based on the completed evaluation evidence in this report. "
                    "The recommendation covers only the components this run staged and the behavior its trials "
                    "exercised, as listed below."
                )
            lines.append("")
        elif status == "FAIL":
            lines.extend(_fail_lines(results, ae, "plugin", advisory=advisory))
        elif status == "INCOMPLETE":
            lines.extend(
                [
                    (
                        "Tier 3 evaluated only part of this plugin, so this benchmark is a partial result and is "
                        "not publication-complete."
                        if partial
                        else (
                            "One or more required evaluation tiers did not complete, so this benchmark is not "
                            "publication-complete."
                        )
                    ),
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    (
                        "Live evaluation did not show a material gain or regression. Collect more evidence or "
                        "improve the plugin before making a publication decision."
                    ),
                    "",
                ]
            )

        lines.extend(_lift_band_lines(ae, "plugin"))
        self._render_metadata(
            lines,
            results,
            ae,
            name,
            policy,
            private_labels=private_labels,
            subject_label="Plugin",
            extra_lines=_plugin_metadata_lines(plugin, view, private_labels),
        )
        self._render_plugin_purpose(lines)
        self._render_results_at_a_glance(
            lines,
            ae,
            private_labels,
            subject="plugin",
            sum_of_parts_baseline=bool(view and view["sum_of_parts_baseline"]),
        )
        self._render_plugin_effectiveness(lines, ae, view, private_labels)
        self._render_plugin_canary(lines, view, private_labels)
        self._render_plugin_coverage(lines, view, private_labels)
        self._render_plugin_provenance(lines, view, private_labels, plugin)
        tier3_override = None
        if view and partial:
            reason = f"Partial plugin run: {view['incomplete_reason']}"
            tier3_override = ("INCOMPLETE", reason)
            if _tier3_fail_reasons(ae):
                # The evaluated parts already failed: FAIL outranks the partial run, as on the card.
                _status, evidence = _tier_status("Tier 3", _tier3_results(results), ae, policy)
                tier3_override = ("FAIL", f"{evidence}; {reason[0].lower()}{reason[1:]}")
        self._render_tier_status(lines, results, ae, policy, private_labels, tier3_override)
        self._render_findings(lines, results, private_labels)
        self._render_methodology(lines, ae, private_labels)
        self._render_freshness(lines, subject="plugin")
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _render_plugin_purpose(lines: list[str]) -> None:
        lines.extend(
            [
                "## What This Report Answers",
                "",
                "The three-tier evaluation checks whether the plugin:",
                "",
                "- is safe to use;",
                "- produces correct answers;",
                "- is discovered and activated when needed;",
                "- helps the agent complete the user's goal and expected workflow;",
                "- avoids wasted skill and tool usage; and",
                "- adds value as a coordinated plugin beyond its individual components (Integration, when measured).",
                "",
                (
                    f"A plugin evaluation demonstrates only what it staged and exercised. {STAGING_CAVEAT} The "
                    "coverage and exclusion sections below state what this run did not evaluate."
                ),
                "",
            ]
        )

    @staticmethod
    def _render_plugin_effectiveness(
        lines: list[str],
        ae: dict[str, Any] | None,
        view: dict[str, Any] | None,
        private_labels: tuple[str, ...],
    ) -> None:
        lines.extend(
            [
                "## Effectiveness and Integration",
                "",
                "| Measure | Result | Uncertainty |",
                "|---|---|---|",
            ]
        )
        statistics = (view or {}).get("statistics")
        lift_ci = {row["kind"]: row for row in statistics["primary"]["lift_ci"]} if statistics else {}
        summary = _mapping((ae or {}).get("summary"))
        overall_lift = number((ae or {}).get("overall_lift", summary.get("overall_lift")))
        sum_of_parts_baseline = bool(view and view["sum_of_parts_baseline"])
        if sum_of_parts_baseline:
            # The only baseline staged the member components individually, so the
            # overall lift and its interval are the Integration comparison.
            effectiveness = "Not measured — lift mode integration compares against sum-of-parts"
            effectiveness_uncertainty = "Not measured"
        else:
            effectiveness = _effectiveness_result(ae, overall_lift, lift_ci.get("effectiveness"))
            effectiveness_uncertainty = _ci_label(lift_ci.get("effectiveness"))
        lines.append(
            f"| Plugin lift (plugin vs. no plugin) | {_md_cell(effectiveness, private_labels)} "
            f"| {_md_cell(effectiveness_uncertainty, private_labels)} |"
        )
        integration = (view or {}).get("integration")
        modes = (view or {}).get("lift_modes")
        # One named row per agent in a multi-agent run; one unnamed row otherwise.
        entries = (integration or {}).get("per_agent") or [integration]
        for entry in entries:
            label = "Integration (plugin vs. its own parts)"
            if entry and entry.get("agent") and len(entries) > 1:
                label += f" — {entry['agent']}"
            if entry and entry["measured"]:
                lift = entry.get("lift_value")
                result = f"{entry['verdict_label']}, {_format_points(lift) if lift is not None else 'lift n/a'}"
                # The view's Integration interval already falls back to the statistics block.
                uncertainty = _ci_label(entry.get("ci"))
            elif entry:
                result = f"INCONCLUSIVE — {entry['reason']}"
                uncertainty = "Not measured"
            elif modes:
                result = f"Not requested (lift mode: {modes['requested']})"
                uncertainty = "Not measured"
            else:
                result = "Not measured"
                uncertainty = "Not measured"
            lines.append(
                f"| {_md_cell(label, private_labels)} | {_md_cell(result, private_labels)} "
                f"| {_md_cell(uncertainty, private_labels)} |"
            )
        lines.extend(
            [
                "",
                (
                    "Plugin lift compares the plugin with the same tasks run without it, on the dimensions both runs "
                    "can score; it is not the Effectiveness dimension (goal and workflow) above. Integration compares "
                    "the coordinated plugin with its member components staged individually; it is advisory and never "
                    "changes the verdict."
                ),
                "",
            ]
        )
        if sum_of_parts_baseline:
            lines.extend(
                [
                    (
                        "This run used lift mode `integration`: its only baseline staged the member components "
                        "individually, so the plugin was never compared with a run without it."
                    ),
                    "",
                ]
            )
        if modes:
            lines.extend(
                [
                    (
                        f"Lift mode: requested `{_md_cell(modes['requested'], private_labels)}`, "
                        f"effective `{_md_cell(modes['effective'], private_labels)}`."
                    ),
                    "",
                ]
            )

    @staticmethod
    def _render_plugin_canary(
        lines: list[str],
        view: dict[str, Any] | None,
        private_labels: tuple[str, ...],
    ) -> None:
        """Surface canary exfiltration results, so a plugin-attributable leak is on the card."""
        canary = (view or {}).get("canary")
        if not canary:
            return
        lines.extend(["## Canary Exfiltration", ""])
        for entry in canary["entries"]:
            scope = _publication_safe_inline(entry["scope"], private_labels)
            marker = "**CRITICAL:** " if entry["plugin_attributable_leak"] else ""
            lines.append(f"- {scope}: {marker}{_publication_safe_inline(entry['verdict'], private_labels)}")
            for label, key, names_key in (
                ("credential reads", "credential_verdict", "credential_names"),
                ("protected writes", "write_verdict", "write_names"),
            ):
                if not entry.get(f"{key}_base"):
                    continue
                marker = "**CRITICAL:** " if entry.get(f"{key}_class") == "fail" else ""
                # Store and file names are canonical entries (~/.ssh, /etc/sudoers.d); anything else stays out.
                names = [
                    name
                    for name in entry.get(names_key) or []
                    if re.fullmatch(r"~(?:/[A-Za-z0-9._-]+)*|/(?:etc|run|var/run)(?:/[A-Za-z0-9._-]+)+", name)
                ]
                suffix = f": {', '.join(f'`{name}`' for name in names)}" if names else ""
                verdict = _publication_safe_inline(entry[f"{key}_base"], private_labels)
                lines.append(f"- {scope}: {label}: {marker}{verdict}{suffix}")
        lines.append("")

    @staticmethod
    def _render_plugin_coverage(
        lines: list[str],
        view: dict[str, Any] | None,
        private_labels: tuple[str, ...],
    ) -> None:
        lines.extend(["## Component Coverage", ""])
        coverage = (view or {}).get("coverage")
        if not coverage:
            lines.extend(
                [
                    (
                        "Per-component coverage was not recorded for this run. Do not assume that every declared "
                        "component was evaluated."
                    ),
                    "",
                ]
            )
            return
        lines.extend(
            [
                (
                    f"**{_md_cell(coverage['headline'], private_labels)}** "
                    f"{_md_cell(coverage['detail'], private_labels)}."
                ),
                "",
                f"{coverage['caveat']} {_md_cell(coverage['note'], private_labels)}",
                "",
                "| Component | Type | State | Reason |",
                "|---|---|---|---|",
            ]
        )
        for row in coverage["rows"]:
            lines.append(
                f"| {_md_cell(row['name'] or 'unnamed', private_labels)} | {_md_cell(row['type'], private_labels)} "
                f"| {_md_cell(row['state_label'], private_labels)} | {_md_cell(row['reason'] or '—', private_labels)} |"
            )
        if coverage["omitted"]:
            lines.append(f"| {coverage['omitted']} more component(s) | | | |")
        lines.append("")
        lines.extend(_not_staged_past_table_lines(coverage, private_labels))
        if coverage.get("not_loaded_rows"):
            lines.extend(["Staged but not loaded (the harness reported it did not load):", ""])
            for row in coverage["not_loaded_rows"]:
                reason = f" — {_publication_safe_inline(row['reason'], private_labels)}" if row["reason"] else ""
                lines.append(
                    f"- {_publication_safe_inline(row['type'], private_labels)} "
                    f"{_publication_safe_inline(row['name'], private_labels)}{reason}"
                )
            lines.append("")
        if coverage["staged_not_observed_rows"]:
            lines.extend(["Staged but not observed in any plugin trial:", ""])
            lines.extend(
                f"- {_publication_safe_inline(row['type'], private_labels)} "
                f"{_publication_safe_inline(row['name'], private_labels)} "
                f"({_publication_safe_inline(row['observed'], private_labels)})"
                for row in coverage["staged_not_observed_rows"]
            )
            lines.append("")

    @staticmethod
    def _render_plugin_provenance(
        lines: list[str],
        view: dict[str, Any] | None,
        private_labels: tuple[str, ...],
        plugin: dict[str, Any],
    ) -> None:
        lines.extend(["## Provenance and Excluded Behavior", ""])
        excluded = list((view or {}).get("excluded") or [])
        unsupported = unsupported_type_split(plugin, (view or {}).get("coverage"))
        if unsupported["static_only"]:
            excluded.append(
                "Runtime behavior of these component types was not evaluated (Tier 3 does not stage them in "
                "wrapper mode); Tier 1 checks them statically: " + ", ".join(unsupported["static_only"])
            )
        if unsupported["unevaluated"]:
            excluded.append(
                "Tier 3 does not stage these component types in wrapper mode, and no check evaluates them: "
                + ", ".join(unsupported["unevaluated"])
            )
        completeness = (view or {}).get("completeness")
        if not completeness:
            lines.append(
                "- Plugin provenance was not recorded for this run; treat component coverage and exclusions as unknown."
            )
            lines.extend(f"- {_publication_safe_inline(statement, private_labels)}" for statement in excluded)
            lines.append("")
            return
        counts = completeness["counts"]
        if view and view["partial"]:
            lines.append(
                f"- Status: **INCOMPLETE** — {_publication_safe_inline(view['incomplete_reason'], private_labels)}. "
                "This is a partial result, not a pass."
            )
        else:
            lines.append("- Status: complete — every declared dependency was resolved for Tier 3.")
        # An unreadable sidecar recorded no counts; zeros would read as "nothing deferred".
        if not completeness.get("sidecar_error"):
            lines.append(
                f"- Resolved: {counts['skills_resolved']} skill(s), {counts['rules_resolved']} rule(s), "
                f"{counts['mcp_runnable']} runnable MCP server(s)"
            )
            lines.append(
                f"- Deferred: {counts['skills_unresolved']} skill ref(s), {counts['rules_unresolved']} rule ref(s), "
                f"{counts['mcp_provider_only']} provider-only MCP server(s), "
                f"{counts['mcp_unsupported_config']} MCP server(s) with unsupported config"
            )
        dataset = (view or {}).get("dataset")
        if dataset:
            cases = dataset["cases"] if dataset["cases"] is not None else "not recorded"
            cross = dataset["cross_component_cases"] if dataset["cross_component_cases"] is not None else "not recorded"
            lines.append(f"- Dataset: {cases} case(s), {cross} cross-component case(s)")
        lines.extend(["", "Excluded behavior:", ""])
        if excluded:
            lines.extend(f"- {_publication_safe_inline(statement, private_labels)}" for statement in excluded)
        else:
            lines.append("- No declared component or measurement was recorded as excluded from this run.")
        lines.append("")

    def get_file_extension(self) -> str:
        return ".md"


def _agent_eval_payload(results: list[ValidationResult]) -> dict[str, Any] | None:
    """Return the payload the score and run-detail lines describe.

    A card reports one live evaluation, so the first payload found is the one
    rendered. That choice is safe for scores, which belong to a single run, but
    not for the evaluated-source identity: see ``_evaluated_source``, which folds
    every carrier instead so a second payload cannot be silently dropped.
    """
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
        score = number(value)
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
    # A real failure outranks missing evidence (the footer and the terminal summary use the same rule):
    # a result that failed the gate on a blocking finding of its own is FAIL even if a scan also did not
    # complete, and the FAIL text says what did not complete.
    real_failure = any(not passes_required_gate(result) and not _missing_evidence_only(result) for result in results)
    if any(result.is_incomplete for result in results) and not real_failure:
        return "INCOMPLETE"

    blocking_skips = [
        result
        for result in results
        if _result_skipped(result) and not result.metadata.get("optional") and not is_advisory_agent_eval_skip(result)
    ]
    if blocking_skips and not real_failure:
        return "INCOMPLETE"

    has_failures = not all(passes_required_gate(result) for result in results)
    if has_failures and not real_failure and not _tier3_fail_reasons(ae):
        # Only missing evidence blocks the gate (a partial Tier 3 run): INCOMPLETE, as in the footer.
        return "INCOMPLETE"
    summary = _mapping((ae or {}).get("summary"))
    verdict = str((ae or {}).get("verdict") or summary.get("verdict") or "").lower()
    dimension_verdict = _tier3_dimension_verdict(ae)
    if has_failures or _tier3_fail_reasons(ae):
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


def _plugin_overall_status(
    results: list[ValidationResult],
    ae: dict[str, Any] | None,
    benchmark_policy: dict[str, bool],
    *,
    partial: bool,
) -> str:
    """Return the card verdict; a partial plugin run is INCOMPLETE unless something failed.

    A Tier 3 FAIL verdict or a confirmed Skill Lift regression fails a partial run
    as it fails a complete one: the legend, the footer and ``--block-on-agent-eval``
    apply the same rule.
    """
    if not partial:
        return _overall_status(results, ae, benchmark_policy)
    non_tier3 = [result for result in results if not _is_tier3(result)]
    if any(not passes_required_gate(result) and not _missing_evidence_only(result) for result in non_tier3):
        return "FAIL"
    if _tier3_fail_reasons(ae):
        return "FAIL"
    return "INCOMPLETE"


def _tier3_gating(results: list[ValidationResult]) -> bool | None:
    """Return whether Tier 3 was in the exit gate, or ``None`` when no gate was recorded."""
    values = [
        gating["blocking"]
        for result in _tier3_results(results)
        if isinstance(gating := (result.metadata or {}).get("gating"), dict)
        and isinstance(gating.get("blocking"), bool)
    ]
    return any(values) if values else None


def _tier3_fail_reasons(ae: dict[str, Any] | None) -> list[str]:
    """Say why a complete Tier 3 run fails its gate, in report words.

    A FAIL verdict, or a confirmed Skill Lift regression (FAIL band, whole
    interval below zero). The same rules gate ``validate --block-on-agent-eval``.
    """
    reasons: list[str] = []
    summary = _mapping((ae or {}).get("summary"))
    verdict = str((ae or {}).get("verdict") or summary.get("verdict") or "").lower()
    if _tier3_dimension_verdict(ae) == "fail":
        reasons.append(
            f"the Tier 3 verdict is FAIL (every scored agent has a dimension below "
            f"{DIMENSION_VERDICT_NEUTRAL_THRESHOLD:.0%})"
        )
    elif verdict == "fail":
        reasons.append("the Tier 3 verdict is FAIL")
    band = _lift_band(ae)
    if band and band.get("regression_confirmed") is True:
        reasons.append(
            f"Skill Lift is a confirmed regression ({_lift_band_phrase(band, ci_separator=', ', ci_end='')})"
        )
    return reasons


def _lift_band(ae: dict[str, Any] | None) -> dict[str, Any]:
    """Return the payload's Skill Lift band (``verdict``, ``lift``, interval, ``regression_confirmed``)."""
    return _mapping(_mapping(ae).get("lift_band"))


def _lift_band_phrase(band: dict[str, Any], *, ci_separator: str = " (", ci_end: str = ")") -> str:
    """``-20 points (95% CI [-20, -19])``; the interval is left out when none was computed."""
    lift = number(band.get("lift"))
    text = _format_points(lift) if lift is not None else "not measured"
    low, high = number(band.get("ci_low")), number(band.get("ci_high"))
    if low is not None and high is not None:
        confidence = number(band.get("confidence")) or 0.95
        text += f"{ci_separator}{confidence:.0%} CI [{low * 100:+.0f}, {high * 100:+.0f}]{ci_end}"
    return text


def _lift_band_lines(ae: dict[str, Any] | None, subject: str) -> list[str]:
    """Warn on the card when the Skill Lift is in the FAIL band; say whether it is confirmed."""
    band = _lift_band(ae)
    if band.get("verdict") != "fail":
        return []
    fail_points = (number(band.get("fail_threshold")) or TIER3_LIFT_FAIL_THRESHOLD) * 100
    band_text = f"in the FAIL band ({fail_points:.0f} points or less)"
    if band.get("regression_confirmed") is True:
        text = (
            f"⚠️ **Skill Lift regression:** {_lift_band_phrase(band)} is {band_text} and the whole interval is "
            f"below zero, so the tasks went worse with the {subject} than without it."
        )
    else:
        text = (
            f"⚠️ **Negative Skill Lift:** {_lift_band_phrase(band)} is {band_text}, but the interval includes "
            "zero or was not computed, so the regression is not confirmed and does not gate."
        )
    return [text, ""]


def _advisory_tier3_failure(results: list[ValidationResult], ae: dict[str, Any] | None) -> bool:
    """Whether FAIL comes only from Tier 3 while Tier 3 was outside the exit gate (exit code 0)."""
    if not _tier3_fail_reasons(ae):
        return False
    if not all(passes_required_gate(result) for result in results):
        return False
    return _tier3_gating(results) is False


def _fail_lines(
    results: list[ValidationResult],
    ae: dict[str, Any] | None,
    subject: str,
    *,
    advisory: bool,
) -> list[str]:
    """Explain a FAIL card: what failed, and whether it changed the exit code."""
    reasons = _tier3_fail_reasons(ae)
    if advisory:
        return [
            (
                f"Tier 3 live evaluation failed: {'; '.join(reasons)}. Tier 3 was advisory in this run "
                "(`--block-on-agent-eval` was not set), so this result did not change the `validate` exit code. "
                f"Review the Tier 3 results before publishing the {subject}; add `--block-on-agent-eval` to make "
                "Tier 3 gate the run."
            ),
            "",
        ]
    gated = " `--block-on-agent-eval` made Tier 3 gate this run." if reasons and _tier3_gating(results) else ""
    others_pass = all(passes_required_gate(result) for result in results if not _is_tier3(result))
    # A gated Tier 3 run that never ran fails the gate, the same as in the footer and the terminal summary.
    not_run = [
        str(_mapping(_mapping(result.metadata.get("agent_eval")).get("provenance")).get("message") or "").strip()
        for result in results
        if _tier3_not_run(result) and not passes_required_gate(result)
    ]
    if reasons and others_pass:
        text = (
            f"Tier 3 live evaluation failed: {'; '.join(reasons)}.{gated} "
            f"Review the Tier 3 results and improve the {subject}, then rerun SkillEvaluator."
        )
    elif not_run and others_pass:
        detail = f" ({_publication_safe_inline(not_run[0])})" if not_run[0] else ""
        text = (
            f"Tier 3 live evaluation did not run{detail}, and `--block-on-agent-eval` made Tier 3 gate this run. "
            "Rerun SkillEvaluator when the live evaluation runtime is available."
        )
    else:
        text = (
            f"The {subject} should be reviewed before publication. Address the blocking findings below, "
            "then rerun SkillEvaluator."
        )
        if reasons:
            text += f" Tier 3 live evaluation failed: {'; '.join(reasons)}.{gated}"
    missing = list(dict.fromkeys(tool for result in results for tool in result.incomplete_scans))
    if missing:
        # FAIL outranks missing evidence, but the card still says what did not complete.
        text += f" Evidence is also incomplete: {_publication_safe_inline(', '.join(missing))} did not complete."
    return [text, ""]


def _plugin_metadata_lines(
    plugin: dict[str, Any],
    view: dict[str, Any] | None,
    private_labels: tuple[str, ...],
) -> list[str]:
    lines: list[str] = []
    manifest_type = plugin.get("manifest_type")
    if manifest_type:
        lines.append(f"- Plugin manifest: {_publication_safe_inline(manifest_type, private_labels)}")
    dataset = (view or {}).get("dataset") or {}
    cross = dataset.get("cross_component_cases")
    if cross is not None:
        lines.append(f"- Cross-component tasks: {cross}")
    coverage = (view or {}).get("coverage")
    if coverage:
        observed = (
            f"; {_publication_safe_inline(coverage['observed_headline'], private_labels)}"
            if coverage["observed_headline"]
            else ""
        )
        lines.append(
            f"- Component coverage: {_publication_safe_inline(coverage['headline'], private_labels)}{observed}"
        )
    if view is not None:
        lines.append(f"- Plugin run: {'INCOMPLETE (partial)' if view['partial'] else 'complete'}")
    return lines


def _not_staged_past_table_lines(coverage: dict[str, Any], private_labels: tuple[str, ...]) -> list[str]:
    """Name, with its reason, each not-staged component past the coverage table's last row.

    The table names the rest, so a plugin with more components than the table
    holds still has every not-staged component named on its card.
    """
    past_table = coverage["not_staged_past_table"]
    if not past_table:
        return []
    lines = [f"Not staged, beyond the {len(coverage['rows'])} rows above:", ""]
    for row in past_table:
        reason = f" — {_publication_safe_inline(row['reason'], private_labels)}" if row["reason"] else ""
        lines.append(
            f"- {_publication_safe_inline(row['type'], private_labels)} "
            f"{_publication_safe_inline(row['name'], private_labels)} "
            f"({_publication_safe_inline(row['state_label'], private_labels)}){reason}"
        )
    # The table's not-staged rows; a not-loaded row was staged, and the card lists it on its own.
    listed = len(past_table) + sum(
        1 for row in coverage["rows"] if not row["staged"] and row["state"] != NOT_LOADED_STATE
    )
    if coverage["not_staged"] > listed:
        lines.append(f"- {coverage['not_staged'] - listed} more component(s) not staged")
    lines.append("")
    return lines


def _ci_label(row: dict[str, Any] | None) -> str:
    """Render a lift interval in percentage points, with its precision and zero warning."""
    if not row:
        return "Not measured"
    low = row.get("low_value")
    high = row.get("high_value")
    if low is None or high is None:
        if row.get("precision") == "insufficient":
            return f"No interval: {row['n_cases']} paired case(s), too few; precision insufficient"
        return f"Interval not recorded; precision {row['precision']}"
    label = f"[{low * 100:+.0f}, {high * 100:+.0f}] points ({row['confidence']}); precision {row['precision']}"
    if row.get("ci_includes_zero"):
        label += "; CI includes zero"
    if row.get("partial_note"):
        label += f"; {row['partial_note']}"
    return label


def _effectiveness_result(ae: dict[str, Any] | None, lift: float | None, row: dict[str, Any] | None) -> str:
    """The plugin-vs-no-plugin result cell: final, partial (not final), or why it is missing."""
    if lift is not None:
        return _format_points(lift)
    if row and row.get("partial") and row.get("estimate_value") is not None:
        return f"Partial run, not final: {_format_points(row['estimate_value'])} on the cases both arms scored"
    agents = _agents(ae)
    if agents and all(_baseline_failed(agent) for agent in agents.values()):
        return "Not available — the no-plugin arm did not complete"
    return "Not available"


def _effectiveness_basis(agent: dict[str, Any], *, sum_of_parts_baseline: bool = False) -> dict[str, Any] | None:
    """The comparable plugin and baseline scores behind an agent's lift (same dimensions, same cases).

    In lift mode integration the only baseline is the member-skills arm, so its
    comparison is the Integration basis.
    """
    bases = _mapping(agent.get("lift_basis"))
    if sum_of_parts_baseline:
        entry = _mapping(bases.get("integration"))
        entry = {**entry, "baseline": entry.get("sum_of_parts")} if entry else {}
    else:
        entry = _mapping(bases.get("effectiveness"))
    if number(entry.get("with_skill")) is None or number(entry.get("baseline")) is None:
        return None
    return entry


def _baseline_failed(agent: dict[str, Any]) -> bool:
    without = _mapping(_mapping(agent.get("conditions")).get("without_skill"))
    return str(without.get("execution_status") or "") in {"failed", "unknown"}


def _missing_baseline_transition(
    agent: dict[str, Any], score: float | None, subject: str, *, missing_baseline: str | None = None
) -> str:
    """A score without a comparable baseline: say whether the baseline arm ran, failed, or was not run."""
    if score is not None and _baseline_failed(agent):
        arm = _mapping(_mapping(agent.get("conditions")).get("without_skill"))
        expected = _nonnegative_int(arm.get("expected_attempts"))
        scored = (
            f" ({_nonnegative_int(arm.get('scored_attempts'))} of {expected} attempts scored, so the run is INCOMPLETE)"
            if expected
            else ""
        )
        return f"{score:.0%} — the no-{subject} baseline did not complete; uplift unavailable{scored}"
    return _score_transition_values(None, score, missing_baseline=missing_baseline)


def _overall_lift_basis_differs(agent: dict[str, Any], basis: dict[str, Any]) -> bool:
    """Whether the lift's basis is not simply the full score (fewer dimensions, or other case weights)."""
    score = number(agent.get("with_skill", agent.get("overall_score")))
    dimensions = [str(dim) for dim in basis.get("dimensions") or []]
    shared = number(basis.get("with_skill"))
    if dimensions and not set(DIMENSION_MAPPING) <= set(dimensions):
        return True
    return score is not None and shared is not None and round(score * 100) != round(shared * 100)


def _overall_lift_transition(
    agent: dict[str, Any],
    subject: str,
    *,
    sum_of_parts_baseline: bool = False,
    missing_baseline: str | None = None,
) -> str:
    """Overall row: the lift's own basis, so it never disagrees with the lift row."""
    score = number(agent.get("with_skill", agent.get("overall_score")))
    basis = _effectiveness_basis(agent, sum_of_parts_baseline=sum_of_parts_baseline)
    if basis is None or score is None:
        # Payloads without a lift basis (older runs) keep their own baseline and score.
        if score is not None and number(agent.get("baseline")) is not None:
            return _score_transition_values(number(agent.get("baseline")), score)
        return _missing_baseline_transition(agent, score, subject, missing_baseline=missing_baseline)
    baseline = float(basis["baseline"])
    shared = float(basis["with_skill"])
    if not _overall_lift_basis_differs(agent, basis):
        return _score_transition_values(baseline, shared)
    return f"{score:.0%}; lift basis {baseline:.0%} → {shared:.0%} ({_format_points(shared - baseline)})"


def _dimension_lift_transition(
    agent: dict[str, Any],
    dimension: dict[str, Any] | None,
    subject: str,
    *,
    missing_baseline: str | None = None,
) -> str:
    if dimension and number(dimension.get("baseline")) is None:
        score = number(dimension.get("with_skill", dimension.get("score")))
        if score is not None and dimension.get("baseline_not_applicable"):
            return f"{score:.0%} — not applicable without the {subject}"
        return _missing_baseline_transition(agent, score, subject, missing_baseline=missing_baseline)
    return _score_transition(dimension, missing_baseline=missing_baseline)


def _shared_basis_note(
    agents: dict[str, dict[str, Any]], subject: str, *, sum_of_parts_baseline: bool = False
) -> list[str]:
    """Explain an Overall row whose uplift uses the lift's basis rather than the full score."""
    for agent in agents.values():
        basis = _effectiveness_basis(agent, sum_of_parts_baseline=sum_of_parts_baseline)
        if basis is not None and _overall_lift_basis_differs(agent, basis):
            break
    else:
        return []
    shared = [str(dim) for dim in basis.get("dimensions") or []]
    names = ", ".join(dim.title() for dim in shared if dim in DIMENSION_MAPPING) or "the overall score"
    note = (
        f"In the Overall row the first value is the full {subject} score. The uplift after it is the lift's own "
        f"basis: each case counts once, and only the dimensions both runs scored ({names}) are compared."
    )
    if shared:
        # Name only the dimensions the run without the plugin cannot score; others
        # (for example Correctness with no ground_truth) are left out for another reason.
        not_applicable = {
            str(dimension.get("id"))
            for dimension in agent.get("dimensions") or []
            if isinstance(dimension, dict) and dimension.get("baseline_not_applicable")
        }
        needs_subject = [dim for dim in DIMENSION_MAPPING if dim not in shared and dim in not_applicable]
        other = [dim for dim in DIMENSION_MAPPING if dim not in shared and dim not in not_applicable]
        if needs_subject:
            plural = len(needs_subject) > 1
            note += (
                f" {_join_names(needs_subject)} {'need' if plural else 'needs'} the {subject} installed, so the run "
                f"without it has no score for {'them' if plural else 'it'}."
            )
        if other:
            plural = len(other) > 1
            note += (
                f" The lift also leaves out {_join_names(other)}, which {'do' if plural else 'does'} not have a score "
                "in both runs."
            )
    return ["", note]


def _join_names(dimensions: list[str]) -> str:
    """``Correctness``, ``Discoverability and Efficiency``, or ``A, B and C``."""
    names = [dim.title() for dim in dimensions]
    return " and ".join([", ".join(names[:-1]), names[-1]] if len(names) > 1 else names)


def _verdict_callout(status: str, *, advisory: bool = False) -> str:
    if status == "FAIL" and advisory:
        # The exit code was 0 (Tier 3 was advisory), so nothing was blocked.
        return "> ❌ **Overall verdict: FAIL — Not recommended for publication (Tier 3 was advisory in this run)**"
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
    subject: str = "skill",
) -> str:
    display = _human_agent_name(
        _publication_safe_label(agent.get("display_name") or agent.get("label") or name, private_labels)
    )
    return f"{display} (Baseline → {subject.title()} Uplift)"


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


def _evaluated_source(results: list[ValidationResult]) -> dict[str, str]:
    """Return the validated evaluated-source identity recorded for this run.

    Every card shape needs a carrier for the identity, not just a completed
    Tier 3 run: a Tier 1-only card and an advisory Tier 3 skip can both publish
    a PASS. The identity is therefore read from every result's metadata and from
    every nested agent-eval payload alike, mirroring how the persisted
    publication policy is resolved.

    The results are handed to ``recorded_evaluated_source`` whole rather than
    alongside the single payload the score lines use, because that payload is
    the first one found: a run carrying two agent-eval results would otherwise
    publish the identity of whichever came first and would name the other after
    the list was reversed. Folding every carrier makes that disagreement raise
    ``EvaluatedSourceConflict`` in either order.

    The value is re-validated by the fold rather than trusted, because a card can
    be rendered from a hand-built or legacy metadata dict that never passed the
    producer. Kept separate from the evaluator/container provenance so a reader
    can tell which source tree was evaluated from the build that evaluated it.
    """
    return recorded_evaluated_source(result.metadata for result in results) or {}


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


def _agent_dimension(agent: dict[str, Any], dim_id: str) -> dict[str, Any] | None:
    for dimension in agent.get("dimensions") or []:
        if isinstance(dimension, dict) and dimension.get("id") == dim_id:
            return dimension
    return None


def _score_transition(dimension: dict[str, Any] | None, *, missing_baseline: str | None = None) -> str:
    if not dimension:
        return "Not available"
    baseline = number(dimension.get("baseline"))
    score = number(dimension.get("with_skill", dimension.get("score")))
    return _score_transition_values(baseline, score, missing_baseline=missing_baseline)


def _score_transition_values(
    baseline: float | None,
    score: float | None,
    *,
    missing_baseline: str | None = None,
) -> str:
    if score is None:
        return "Not available"
    skill_label = f"{score:.0%}"
    if baseline is None:
        return f"{skill_label} — {missing_baseline or 'baseline not run; uplift unavailable'}"
    delta = score - baseline
    return f"{baseline:.0%} → {skill_label} ({_format_points(delta)})"


def _incomplete_run_note(agent: dict[str, Any], subject: str) -> str | None:
    """Say how much of an agent's INCOMPLETE run scored; ``None`` when it completed or recorded no status."""
    status = str(agent.get("execution_status") or "").strip().lower()
    if not status or status == "succeeded":
        return None
    conditions = _mapping(agent.get("conditions"))
    counts: list[str] = []
    for key, label in (("with_skill", f"with-{subject}"), ("without_skill", "baseline")):
        arm = _mapping(conditions.get(key))
        expected = _nonnegative_int(arm.get("expected_attempts"))
        if expected:
            counts.append(f"{label} {_nonnegative_int(arm.get('scored_attempts'))} of {expected} attempts scored")
    note = f"execution status {_publication_safe_inline(status)}"
    return f"{note}; {', '.join(counts)}" if counts else note


def _missing_baseline_text(agent: dict[str, Any], ae: dict[str, Any] | None) -> str | None:
    """Say why an agent has no comparable baseline when its no-plugin arm did run.

    ``None`` keeps the default "baseline not run" wording, which is right only
    when no baseline arm ran (for example ``--skip-baseline``).
    """
    arm = _mapping(_mapping(agent.get("conditions")).get("without_skill"))
    expected = _nonnegative_int(arm.get("expected_attempts"))
    scored = _nonnegative_int(arm.get("scored_attempts"))
    trials = _nonnegative_int(agent.get("num_trials_baseline"))
    if expected:
        ran = f"baseline ran ({scored} of {expected} attempts scored)"
    elif trials:
        ran = f"baseline ran ({trials} trial(s))"
    else:
        return None
    statuses = {
        str(agent.get("execution_status") or "").lower(),
        str(_mapping(ae).get("execution_status") or "").lower(),
        str(arm.get("execution_status") or "").lower(),
    }
    if statuses & {"failed", "unknown", "skipped"}:
        return f"{ran}, but the run is INCOMPLETE; uplift not computed"
    return f"{ran}, but no comparable baseline score was recorded; uplift not computed"


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
        missing = f"missing trustworthy evidence from {', '.join(tools)}"
        if any(not passes_required_gate(result) and not _missing_evidence_only(result) for result in results):
            # A real failure outranks missing evidence, as on the card and in the footer.
            blocking = sum(1 for result in results for finding in result.findings if _finding_blocks(finding, result))
            return "FAILED", f"{blocking} blocking finding(s); {missing}"
        return "INCOMPLETE", missing[0].upper() + missing[1:]
    if all(_result_skipped(result) for result in results):
        optional = all(result.metadata.get("optional") or is_advisory_agent_eval_skip(result) for result in results)
        return ("SKIPPED (ADVISORY)" if optional else "INCOMPLETE"), "; ".join(_skip_reasons(results))

    findings = [finding for result in results for finding in result.findings]
    if tier == "Tier 3" and ae and _tier3_fail_reasons(ae):
        # The verdict (or a confirmed Skill Lift regression) decides the Tier 3 row; say whether it gated.
        gating = _tier3_gating(results)
        gate = "; advisory (not in the exit gate)" if gating is False else "; gates the exit code" if gating else ""
        regression = "; Skill Lift regression" if _lift_band(ae).get("regression_confirmed") is True else ""
        agents_tasks = f"{len(_agents(ae))} agent(s); {_dataset_summary(ae)['total_tasks']} task(s)"
        return "FAIL", f"{agents_tasks}{regression}{gate}"
    not_run = [result for result in results if _tier3_not_run(result) and not passes_required_gate(result)]
    if not_run:
        # --block-on-agent-eval made Tier 3 gate, and it never ran: a failed gate, not missing evidence.
        return "FAILED", "Did not run: " + "; ".join(_skip_reasons(not_run)) + "; gates the exit code"
    if any(not result.passed and not _result_skipped(result) for result in results):
        return "FAILED", f"{len(results)} validator(s); {len(findings)} finding(s)"

    if tier == "Tier 3" and ae:
        summary = _mapping(ae.get("summary"))
        execution = str(ae.get("execution_status") or summary.get("execution_status") or "").lower()
        verdict = str(ae.get("verdict") or summary.get("verdict") or "").lower()
        dimension_verdict = _tier3_dimension_verdict(ae)
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


def _finding_blocks(finding: Finding, result: ValidationResult) -> bool:
    """Whether *finding* blocks: its result fails the required gate and records it as an error.

    CRITICAL and HIGH are errors; a MEDIUM or LOW is one only when its check made
    it fail (``fail_on_medium``). Findings of a result that passes the gate, such
    as advisory Tier 2 or Tier 3, never block.
    """
    if passes_required_gate(result):
        return False
    return _finding_is_error(finding, result)


def _finding_is_error(finding: Finding, result: ValidationResult) -> bool:
    """CRITICAL and HIGH, or a MEDIUM or LOW that *result* records as an error."""
    return finding.severity.value in {"critical", "high"} or finding.to_legacy_string() in result.errors


def has_blocking_finding(result: ValidationResult) -> bool:
    """Whether *result* records a blocking finding of its own, not only missing evidence.

    A scanner that did not complete adds warnings or plain errors, never a
    finding, so this tells a real failure from an INCOMPLETE scan when one
    result has both. The footer, the terminal summary and this card all use it:
    a real failure outranks missing evidence.
    """
    return any(_finding_is_error(finding, result) for finding in result.findings)


def _missing_evidence_only(result: ValidationResult) -> bool:
    """Whether a result that fails the gate is only missing evidence (INCOMPLETE), not failed."""
    if result.is_incomplete:
        return not has_blocking_finding(result)
    return _is_tier3(result) and _tier3_result_incomplete(result)


def _tier3_result_incomplete(result: ValidationResult) -> bool:
    """A partial Tier 3 plugin run that did not fail its gate (no FAIL verdict, no confirmed regression)."""
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if metadata.get("execution_status") != "skipped" or metadata.get("tier3_gate_failures"):
        return False
    payload = _mapping(metadata.get("agent_eval"))
    return str(payload.get("verdict") or "").lower() != "fail" and not _tier3_not_run(result)


def _tier3_not_run(result: ValidationResult) -> bool:
    """Whether *result* is a Tier 3 run that never ran (an advisory skip), gated or not."""
    if not _is_tier3(result):
        return False
    provenance = _mapping(_mapping((result.metadata or {}).get("agent_eval")).get("provenance"))
    return bool(provenance.get("advisory") and provenance.get("reason") == "skipped")


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


def _publication_safe_target_name(candidates: tuple[object, ...], *, fallback: str) -> str:
    """Return the first canonical kebab-case identity among *candidates*, else *fallback*."""
    for value in candidates:
        candidate = " ".join(str(value or "").split())
        if candidate and re.fullmatch(KEBAB_CASE_PATTERN, candidate) is not None:
            return candidate
    return fallback


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
    text = " ".join(strip_terminal_controls(str(value)).split())
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
        match = next(
            (found for found in _PATH_START.finditer(token) if not _is_relative_or_markup_slash(token, found.start())),
            None,
        )
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


_CLOSING_TAG = re.compile(r"/[A-Za-z][A-Za-z0-9-]*>")


def _is_relative_or_markup_slash(token: str, index: int) -> bool:
    """Whether the separator at ``token[index]`` starts no absolute path.

    ``./x``, ``../x`` and ``~/x`` are relative paths, and the ``/`` of a
    closing tag such as ``</script>`` is markup. Redacting them kept only the
    basename (``./missing/reviewer.md`` became ``.reviewer.md``) or changed
    the text (``</script>`` became ``<script>``). A run of three or more dots
    (an ellipsis before a real path) still counts as an absolute path start.
    """
    before = token[:index]
    if before.endswith("<") and _CLOSING_TAG.match(token, index):
        return True
    if before.endswith("~"):
        return len(before) == 1 or not before[-2].isalnum()
    dots = len(before) - len(before.rstrip("."))
    return dots in (1, 2) and (len(before) == dots or not before[-dots - 1].isalnum())


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
