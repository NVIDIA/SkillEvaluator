# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publication-ready BENCHMARK.md reporter for skill and plugin evaluation cards."""

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
    KEBAB_CASE_PATTERN,
    TIER3_LIFT_FAIL_THRESHOLD,
    TIER3_LIFT_PASS_THRESHOLD,
)
from skillevaluator.publication_evidence import result_has_publication_evidence, result_publication_evidence
from skillevaluator.publication_text import publication_confusable_skeleton
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
    is_unfailed_partial_plugin_agent_eval,
    missing_tier1_publication_checks,
    passes_required_gate,
    publication_identity_present,
    publication_semantic_text,
    publication_target_for_results,
    resolve_benchmark_policy,
    result_has_execution_evidence,
    result_matches_publication_target,
    select_agent_eval_payload,
    tier1_unscanned_publication_paths,
    unrecognized_tier2_results,
)
from skillevaluator.reporting.plugin_sections import (
    BUNDLE_LIFT_DEFINITION,
    BUNDLE_LIFT_LABEL,
    NOT_LOADED_STATE,
    STAGING_CAVEAT,
    number,
    tier3_plugin_view,
    unsupported_type_split,
)
from skillevaluator.reporting.run_status import (
    RunStatus,
    Status,
    TierStatus,
    _finding_is_error,
    _publication_skipped,
    _tier3_not_run,
    resolve_run_status,
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
# The public benchmark gate rejects a non-PASS card whose visible text contains
# this decision phrase, matching confusable aliases by treating each non-ASCII
# character as a wildcard for one letter. Untrusted text is screened the same way.
_DECISION_ALIAS_SENTINEL = "\x00"
_PUBLICATION_DECISION_PHRASE = "recommended for publication"
_PLAIN_DECISION_PHRASE = re.compile(r"recommended\s+for\s+publication", flags=re.IGNORECASE)
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
        publication_target = publication_target_for_results(
            results,
            ae,
            expected_skill_name=self.skill_name,
        )
        expected_skill_name = self.skill_name or (
            publication_target.skill_name if publication_target is not None else None
        )
        display_skill_name = self.skill_name or _skill_name(results, ae) or expected_skill_name or "skill"
        skill_name = _publication_safe_skill_name(display_skill_name)
        private_labels = _private_environment_labels(ae)
        policy = _benchmark_policy(results, ae, expected_skill_name=expected_skill_name)
        run_status = resolve_run_status(results)
        # The run status's card verdict, never kinder than the shared publication assessment.
        status = _conservative_verdict(
            run_status.publication.verdict,
            _overall_status(results, ae, policy, expected_skill_name=expected_skill_name),
        )

        advisory = run_status.publication_advisory
        lines: list[str] = [
            f"# Skill Benchmark: {skill_name}",
            "",
            _verdict_callout(status, advisory=advisory),
            "",
        ]
        if status == "PASS":
            self._render_publication_recommendation(lines, results)
        elif status == "FAIL":
            lines.extend(_fail_lines(run_status, ae, "skill", advisory=advisory))
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
            expected_skill_name=expected_skill_name,
        )
        self._render_report_purpose(lines)
        self._render_results_at_a_glance(lines, ae, private_labels)
        self._render_tier_status(
            lines,
            run_status,
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
        subject_label: str = "Skill",
        extra_lines: list[str] | None = None,
        expected_skill_name: str | None = None,
    ) -> None:
        lines.extend(["## Evaluation Metadata", "", f"- {subject_label}: `{skill_name}`"])

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
        # The accessor admits only `[A-Za-z0-9][A-Za-z0-9._-]*`, which is inert in a
        # code span. Markdown escaping would put literal backslashes into runner IDs
        # such as `20261016_081505_4821_3f9c2a1b7d4e`.
        lines.append(
            f"- Tier 3 run ID: `{tier3_run_id}`"
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
        run_status: RunStatus,
        ae: dict[str, Any] | None,
        benchmark_policy: dict[str, bool],
        private_labels: tuple[str, ...],
        tier3_override: tuple[str, str] | None = None,
        *,
        expected_skill_name: str | None = None,
    ) -> None:
        tier_groups = [
            ("Tier 1", "Static validation", run_status.tiers[1]),
            ("Tier 2", "Semantic deduplication", run_status.tiers[2]),
            ("Tier 3", "Live agent evaluation", run_status.tiers[3]),
        ]

        lines.extend(
            [
                "## Tier Status",
                "",
                "| Tier | Purpose | Status | Evidence |",
                "|---|---|---|---|",
            ]
        )
        all_results = [result for tier_status in run_status.tiers.values() for result in tier_status.results]
        publication_target = publication_target_for_results(
            all_results,
            ae,
            expected_skill_name=expected_skill_name,
        )
        for tier, purpose, tier_status in tier_groups:
            status, evidence = _tier_status(tier_status, ae, benchmark_policy)
            evidence_row = _publication_evidence_tier_status(
                tier,
                list(tier_status.results),
                ae,
                benchmark_policy,
                expected_skill_name=expected_skill_name,
                expected_publication_target=publication_target,
            )
            if tier == "Tier 3" and tier3_override is not None and tier_status.results:
                # A partial plugin run says so, unless the evidence contract found something worse.
                override_rank = _ROW_RANK.get(tier3_override[0], _ROW_RANK["INCOMPLETE"])
                evidence_rank = _ROW_RANK.get(evidence_row[0], _ROW_RANK["INCOMPLETE"])
                status, evidence = tier3_override if override_rank >= evidence_rank else evidence_row
            else:
                # The publication evidence contract can only make a row more conservative.
                status, evidence = _conservative_row((status, evidence), evidence_row)
            lines.append(f"| {tier} | {purpose} | **{status}** | {_md_cell(evidence, private_labels)} |")
        lines.append("")

        for tier, _purpose, tier_status in tier_groups:
            tier_results = list(tier_status.results)
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
                    "interval includes zero, rests on too few paired cases, or has no interval, only warns. "
                    "Integration-only runs have no Skill Lift band."
                ),
                (
                    "- Exit code: `validate --block-on-agent-eval` passes only a completed Tier 3 PASS verdict "
                    "with complete attempt coverage; a NEUTRAL or FAIL verdict, a confirmed regression, or a skipped "
                    "or INCOMPLETE Tier 3 run fails it. Without the flag, Tier 3 is advisory and never changes the "
                    "exit code."
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
        # The publication evidence contract binds a plugin card to the plugin root's source snapshot, as it binds a
        # skill card to the skill's: Tier 1 and Tier 2 stamp the plugin root. Plugin Tier 3 runs a staged package
        # that records no plugin-root identity, so its evidence is not publication-complete (fail-closed) until a
        # catalog waives Tier 3.
        publication_target = publication_target_for_results(results, ae, expected_skill_name=self.skill_name)
        expected_target_name = self.skill_name or (
            publication_target.skill_name if publication_target is not None else None
        )
        policy = _benchmark_policy(results, ae, expected_skill_name=expected_target_name)
        partial = bool(view and view["partial"])
        run_status = resolve_run_status(results)
        # The run status's card verdict, never kinder than the shared publication assessment.
        status = _conservative_verdict(
            run_status.publication.verdict,
            _overall_status(results, ae, policy, expected_skill_name=expected_target_name),
        )

        advisory = run_status.publication_advisory
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
            lines.extend(_fail_lines(run_status, ae, "plugin", advisory=advisory))
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
            if run_status.tiers[3].publication is Status.FAILED:
                # The evaluated parts already failed: FAIL outranks the partial run, as on the card.
                _status, evidence = _tier_status(run_status.tiers[3], ae, policy)
                tier3_override = ("FAIL", f"{evidence}; {reason[0].lower()}{reason[1:]}")
        self._render_tier_status(
            lines,
            run_status,
            ae,
            policy,
            private_labels,
            tier3_override,
            expected_skill_name=expected_target_name,
        )
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
                "- adds value beyond its skills alone (Bundle lift, when measured).",
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
        splits: list[str] = []
        for entry in entries:
            label = f"{BUNDLE_LIFT_LABEL} (plugin vs. its skills alone)"
            if entry and entry.get("agent") and len(entries) > 1:
                label += f" — {entry['agent']}"
            if entry and entry["measured"]:
                if entry.get("by_case_type"):
                    # Below the table, the line names its lift (and agent) itself.
                    scope = f" — {entry['agent']}" if entry.get("agent") and len(entries) > 1 else ""
                    split = f"{BUNDLE_LIFT_LABEL} by case type{scope}: {entry['by_case_type']}"
                    splits.extend([_md_cell(split, private_labels), ""])
                lift = entry.get("lift_value")
                result = f"{entry['verdict_label']}, {_format_points(lift) if lift is not None else 'lift n/a'}"
                # The view's Integration interval already falls back to the statistics block.
                uncertainty = _ci_label(entry.get("ci"))
            elif entry and entry.get("requested", True):
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
        ladder_notes = _render_ladder_rows(lines, (view or {}).get("lift_ladder"), private_labels)
        lines.extend(
            [
                "",
                (
                    "Plugin lift compares the plugin with the same tasks run without it, on the dimensions both runs "
                    "can score; it is not the Effectiveness dimension (goal and workflow) above. "
                    f"{BUNDLE_LIFT_DEFINITION} It is advisory and never changes the verdict."
                ),
                "",
                *splits,
                *ladder_notes,
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
            lines.extend(["Staged but not exercised in any plugin trial:", ""])
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

    A card reports one live evaluation, so one payload is rendered: the
    strongest Tier 3 payload, selected independently of result order. That
    choice is safe for scores, which belong to a single run, but not for the
    evaluated-source identity: see ``_evaluated_source``, which folds every
    carrier instead so a second payload cannot be silently dropped.
    """
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
            "zero, rests on too few paired cases, or was not computed, so the regression is not confirmed and "
            "does not gate."
        )
    return [text, ""]


def _fail_lines(
    run_status: RunStatus,
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
    # Say Tier 3 gated the run only when it failed the exit gate, not merely because the flag was set.
    failed_gate = {id(result) for result in run_status.failed}
    tier3_failed_gate = any(id(result) in failed_gate for result in run_status.tiers[3].results)
    gated = " `--block-on-agent-eval` made Tier 3 gate this run." if reasons and tier3_failed_gate else ""
    others_pass = all(passes_required_gate(result) for tier in (1, 2) for result in run_status.tiers[tier].results)
    # A gated Tier 3 run that never ran fails the gate, the same as in the footer and the terminal summary.
    not_run = [
        str(_mapping(_mapping(result.metadata.get("agent_eval")).get("provenance")).get("message") or "").strip()
        for result in run_status.tiers[3].results
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
    if run_status.missing:
        # FAIL outranks missing evidence, but the card still says what did not complete.
        missing = ", ".join(run_status.missing)
        text += f" Evidence is also incomplete: {_publication_safe_inline(missing)} did not complete."
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


_VERDICT_RANK = {"PASS": 0, "NEUTRAL": 1, "INCOMPLETE": 2, "FAIL": 3}
_ROW_RANK = {
    "PASSED": 0,
    "PASSED WITH OBSERVATIONS": 0,
    "PASS": 0,
    "NOT RUN": 1,
    "NEUTRAL": 1,
    "SKIPPED (ADVISORY)": 1,
    "INCOMPLETE": 2,
    "FAILED": 3,
    "FAIL": 3,
}


def _conservative_verdict(*verdicts: str) -> str:
    """The most conservative card verdict; a failure outranks missing evidence, as on every report."""
    return max(verdicts, key=lambda verdict: _VERDICT_RANK.get(verdict, _VERDICT_RANK["INCOMPLETE"]))


def _conservative_row(run_status_row: tuple[str, str], evidence_row: tuple[str, str]) -> tuple[str, str]:
    """The more conservative Tier Status row of the run status's and the publication evidence contract's.

    On a tie the evidence contract's wording is kept (it names the missing evidence), except for a failure, where the
    run status's wording also says whether the tier gated the exit code and whether Skill Lift regressed.
    """
    run_rank = _ROW_RANK.get(run_status_row[0], _ROW_RANK["INCOMPLETE"])
    evidence_rank = _ROW_RANK.get(evidence_row[0], _ROW_RANK["INCOMPLETE"])
    if run_rank != evidence_rank:
        return run_status_row if run_rank > evidence_rank else evidence_row
    return run_status_row if run_rank == _ROW_RANK["FAIL"] else evidence_row


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


def _skill_name(results: list[ValidationResult], ae: dict[str, Any] | None) -> str | None:
    if ae:
        summary = _mapping(ae.get("summary"))
        candidate = ae.get("skill_name") or summary.get("skill_name")
        if candidate_text := _safe_scalar_text(candidate).strip():
            return candidate_text
    for result in results:
        quality = result.metadata.get("quality_scores") if isinstance(result.metadata, dict) else None
        if isinstance(quality, dict) and (candidate := _safe_scalar_text(quality.get("skill_name")).strip()):
            return candidate
    return None


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
    subject: str = "skill",
) -> str:
    display = _agent_display_label(name, agent, private_labels)
    return f"{display} (Baseline → {subject.title()} Uplift)"


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
    if lowered == "kata":
        return "Each task attempt ran in its own Kata Containers microVM."
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


def _agent_dimension(agent: dict[str, Any], dim_id: str) -> dict[str, Any] | None:
    raw_dimensions = agent.get("dimensions")
    dimensions = raw_dimensions if isinstance(raw_dimensions, list) else []
    for dimension in dimensions:
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


_LADDER_GLUE_ROW = "Integration lift (glue: full plugin vs. loose parts)"


def _render_ladder_rows(lines: list[str], ladder: dict[str, Any] | None, private_labels: tuple[str, ...]) -> list[str]:
    """Add one lift-ladder row per agent to the lift table; return the notes printed below it.

    The view (plugin_sections.lift_ladder_view) words everything; this only
    places it. In wrapper mode the row is the Wrapper effect, with no verdict.
    """
    if not ladder:
        return []
    notes: list[str] = []
    for entry in ladder["entries"]:
        scope = f" — {entry['agent']}" if ladder["named"] and entry["agent"] else ""
        label = (entry["integration_label"] if entry["wrapper"] else _LADDER_GLUE_ROW) + scope
        if entry["measured"]:
            glue = entry["glue"]
            lift = glue["lift_value"]
            result = f"{glue['verdict_label']}, {_format_points(lift) if lift is not None else 'lift n/a'}"
            uncertainty = _ci_label(glue["ci"]) if lift is not None else "Not measured"
            for note in (
                f"Lift ladder{scope}: {entry['line']}",
                glue["reason"],
                entry["loose_parts_line"],
                entry["capability_line"],
            ):
                if note:
                    notes.extend([_md_cell(note, private_labels), ""])
        else:
            result = f"Not measured — {entry['reason']}"
            uncertainty = "Not measured"
        lines.append(
            f"| {_md_cell(label, private_labels)} | {_md_cell(result, private_labels)} "
            f"| {_md_cell(uncertainty, private_labels)} |"
        )
    if ladder["basis_note"]:
        notes.extend([_md_cell(ladder["basis_note"], private_labels), ""])
    return notes


def _format_points(delta: float) -> str:
    points = delta * 100
    if math.isclose(points, 0.0, abs_tol=0.05):
        return "±0 points"
    sign = "+" if points > 0 else "-"
    return f"{sign}{abs(points):.0f} points"


def _tier_status(
    tier: TierStatus,
    ae: dict[str, Any] | None,
    benchmark_policy: dict[str, bool],
    *,
    expected_skill_name: str | None = None,
    expected_publication_target: PublicationTargetIdentity | None = None,
) -> tuple[str, str]:
    """Word a Tier Status row; the status itself is the run status's ``publication`` for the tier."""
    results = list(tier.results)
    verdict = tier.publication
    if verdict is Status.NOT_RUN:
        return "NOT RUN", "No result was recorded"
    incomplete = [result for result in results if result.is_incomplete]
    if incomplete:
        tools = list(dict.fromkeys(tool for result in incomplete for tool in result.incomplete_scans))
        missing = f"missing trustworthy evidence from {', '.join(tools)}"
        if verdict is Status.FAILED:
            # A real failure outranks missing evidence, as on the card and in the footer.
            blocking = sum(1 for result in results for finding in result.findings if _finding_blocks(finding, result))
            return "FAILED", f"{blocking} blocking finding(s); {missing}"
        return "INCOMPLETE", missing[0].upper() + missing[1:]
    if all(_publication_skipped(result) for result in results):
        return ("SKIPPED (ADVISORY)" if verdict is Status.SKIPPED else "INCOMPLETE"), "; ".join(_skip_reasons(results))

    findings = [finding for result in results for finding in result.findings]
    if tier.tier == 3 and ae and verdict is Status.FAILED and _tier3_fail_reasons(ae):
        # The verdict (or a confirmed Skill Lift regression) decides the Tier 3 row; say whether it gated.
        gate = (
            "; advisory (not in the exit gate)"
            if tier.blocking is False
            else "; gates the exit code"
            if tier.blocking
            else ""
        )
        regression = "; Skill Lift regression" if _lift_band(ae).get("regression_confirmed") is True else ""
        agents_tasks = f"{len(_agents(ae))} agent(s); {_dataset_summary(ae)['total_tasks']} task(s)"
        return "FAIL", f"{agents_tasks}{regression}{gate}"
    not_run = [result for result in results if _tier3_not_run(result) and not passes_required_gate(result)]
    if not_run:
        # --block-on-agent-eval made Tier 3 gate, and it never ran: a failed gate, not missing evidence.
        return "FAILED", "Did not run: " + "; ".join(_skip_reasons(not_run)) + "; gates the exit code"
    if verdict is Status.FAILED:
        return "FAILED", f"{len(results)} validator(s); {len(findings)} finding(s)"

    if tier.tier == 3 and ae:
        summary = _mapping(ae.get("summary"))
        execution = str(ae.get("execution_status") or summary.get("execution_status") or "").lower()
        if verdict is Status.INCOMPLETE and execution and execution != "succeeded":
            return "INCOMPLETE", f"Execution status: {execution}"
        if verdict is Status.INCOMPLETE:
            evidence = (
                "Required Tier 3 evidence is missing"
                if benchmark_policy["tier3_required"]
                else "Present Tier 3 result lacks complete evidence"
            )
            return "INCOMPLETE", evidence
        if verdict in {Status.PASSED, Status.NEUTRAL}:
            return verdict.verdict, f"{len(_agents(ae))} agent(s); {_dataset_summary(ae)['total_tasks']} task(s)"
    if verdict is Status.INCOMPLETE:
        return "INCOMPLETE", "Required evidence is missing"

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


def _publication_evidence_tier_status(
    tier: str,
    results: list[ValidationResult],
    ae: dict[str, Any] | None,
    benchmark_policy: dict[str, bool],
    *,
    expected_skill_name: str | None = None,
    expected_publication_target: PublicationTargetIdentity | None = None,
) -> tuple[str, str]:
    """Main's publication-evidence Tier Status row: recognized producers, source binding, required checks.

    The run status's row (:func:`_tier_status`) and this one are combined with :func:`_conservative_row`.
    """
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
    if any(
        not result.passed and not is_cleanly_skipped(result) and not is_unfailed_partial_plugin_agent_eval(result)
        for result in results
    ):
        return "FAILED", f"{len(results)} validator(s); {len(findings)} finding(s)"

    publication_results = results
    if tier in {"Tier 1", "Tier 2"}:
        expected_tier = 1 if tier == "Tier 1" else 2
        publication_results = [
            result for result in results if result_has_publication_evidence(result, tier=expected_tier)
        ]
        if not publication_results:
            return "INCOMPLETE", f"No recognized built-in {tier} producer evidence was recorded"
        if tier == "Tier 2" and (unrecognized := unrecognized_tier2_results(results)):
            validator_names = list(
                dict.fromkeys(result.validator_name or "Tier 2 validator" for result in unrecognized)
            )
            return "INCOMPLETE", f"Unrecognized Tier 2 producer evidence from {', '.join(validator_names)}"

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

    if tier == "Tier 1" and (missing_checks := missing_tier1_publication_checks(publication_results)):
        return "INCOMPLETE", f"Required Tier 1 checks did not run: {', '.join(missing_checks)}"

    if tier == "Tier 1" and (unscanned_paths := tier1_unscanned_publication_paths(publication_results)):
        shown = ", ".join(unscanned_paths[:5]) + (", ..." if len(unscanned_paths) > 5 else "")
        return "INCOMPLETE", f"Tier 1 skipped certified paths with generated-artifact names: {shown}"

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

def _finding_blocks(finding: Finding, result: ValidationResult) -> bool:
    """Whether *finding* blocks: its result fails the required gate and records it as an error.

    CRITICAL and HIGH are errors; a MEDIUM or LOW is one only when its check made
    it fail (``fail_on_medium``). Findings of a result that passes the gate, such
    as advisory Tier 2 or Tier 3, never block.
    """
    if passes_required_gate(result):
        return False
    return _finding_is_error(finding, result)


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
    if isinstance(value, str) and not preserve_exact_nfc:
        # Drop whole terminal escape sequences (OSC hyperlinks, CSI styling) before flattening.
        value = strip_terminal_controls(value)
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
    if _contains_publication_decision_phrase(text):
        # Validator output or skill content must not carry the card's decision
        # wording, or a legitimate non-PASS card would fail the public gate.
        # Join the phrase itself first, then the whole value, then withhold it.
        text = _PLAIN_DECISION_PHRASE.sub(lambda match: "-".join(match.group(0).split()), text)
        if _contains_publication_decision_phrase(text):
            text = "-".join(text.split())
        if _contains_publication_decision_phrase(text):
            text = "[text withheld: it contains publication decision wording]"
    if _MARKDOWN_BLOCK_PREFIX.match(text) or _MARKDOWN_THEMATIC_BREAK.fullmatch(text):
        marker_end = text.find(" ")
        marker_end = len(text) if marker_end < 0 else marker_end
        if text[:marker_end].rstrip(".)").isdigit():
            punctuation_index = marker_end - 1
            return f"{text[:punctuation_index]}\\{text[punctuation_index:]}"
        return f"\\{text}"
    return text


def _decision_phrase_surface(value: str) -> str:
    """Project text the way the public gate does before matching decision wording."""
    skeleton = publication_confusable_skeleton(publication_semantic_text(value, strip_marks=True))
    return "".join(
        " " if character.isspace() else character if character.isascii() else _DECISION_ALIAS_SENTINEL
        for character in skeleton
    )


_PUBLICATION_DECISION_PATTERN = re.compile(
    "".join(
        r"\s+"
        if character.isspace()
        else f"(?:{re.escape(character)}|{re.escape(_DECISION_ALIAS_SENTINEL)})"
        if character.isascii() and character.isalnum()
        else re.escape(character)
        for character in publication_confusable_skeleton(
            publication_semantic_text(_PUBLICATION_DECISION_PHRASE, strip_marks=True)
        )
    )
)


def _contains_publication_decision_phrase(value: str) -> bool:
    """Return whether text would read as a publication decision to the public gate."""
    return _PUBLICATION_DECISION_PATTERN.search(_decision_phrase_surface(value)) is not None


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
