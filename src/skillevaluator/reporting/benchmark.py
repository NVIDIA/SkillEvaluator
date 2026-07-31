# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publication-ready BENCHMARK.md reporter for SkillEvaluator cards."""

from __future__ import annotations

import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any

from skillevaluator.constants import DIMENSION_HINTS, DIMENSION_MAPPING, KEBAB_CASE_PATTERN
from skillevaluator.reporting.base import ReporterBase, is_advisory_agent_eval_skip, passes_required_gate
from skillevaluator.tier3_environments import HARBOR_ENV_MODES

if TYPE_CHECKING:
    from skillevaluator.models import Finding, ValidationResult


_DIMENSION_DESCRIPTIONS = {
    "security": (
        "checks whether skill-assisted execution avoids unsafe behavior such as "
        "secret leakage, destructive commands, or unauthorized access."
    ),
    "correctness": ("checks whether the agent follows the expected workflow and produces the correct final output."),
    "discoverability": ("checks whether the agent loads the skill when relevant and avoids using it when irrelevant."),
    "effectiveness": ("checks whether the agent performs measurably better with the skill than without it."),
    "efficiency": "checks whether the agent uses fewer tokens and avoids redundant work.",
}

_SIGNAL_DESCRIPTIONS = {
    "security": "checks for unsafe operations, secret leakage, and unauthorized access.",
    "skill_execution": "verifies that the agent loaded the expected skill and workflow.",
    "skill_efficiency": "checks routing quality, decoy avoidance, and redundant tool usage.",
    "accuracy": "grades final-answer correctness against the reference answer.",
    "goal_accuracy": "checks whether the overall user task completed successfully.",
    "behavior_check": "verifies expected behavior steps, including safety expectations.",
    "token_efficiency": "compares token usage with and without the skill.",
}

_TIER2_VALIDATORS = {
    "context deduplication",
    "intra-skill deduplication",
}

_RETIRED_PRODUCT_NAME = re.compile(r"\b[a-z]*[\s_-]*skills[\s_-]*eval\b", flags=re.IGNORECASE)
_PATH_START = re.compile(r"(?<![A-Za-z0-9:/])(?:[A-Za-z]:[\\/]|\\\\|\\|/)")
_QUOTED_ABSOLUTE_PATH = re.compile(
    r"(?P<quote>['\"])(?P<path>(?:[A-Za-z]:[\\/]|\\\\|\\|/)[^'\"\r\n]+)(?P=quote)"
)
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
    """Render a stable ``BENCHMARK.md`` skill evaluation card."""

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
        return "BENCHMARK.md skill evaluation card"

    def render(self, result: ValidationResult) -> str:
        return self.render_all([result])

    def render_all(self, results: list[ValidationResult]) -> str:
        ae = _agent_eval_payload(results)
        skill_name = _publication_safe_skill_name(self.skill_name or _skill_name(results, ae))
        private_labels = _private_environment_labels(ae)

        lines: list[str] = [
            "# Evaluation Report",
            "",
            (f"Evaluation of the `{skill_name}` skill before publication through SkillEvaluator."),
            "",
            (
                "This benchmark summarizes 3-Tier Evaluation from SkillEvaluator "
                "results for the skill. The goal is to document whether the "
                "skill is safe, discoverable, effective, and useful for agents before "
                "it is published for broader workflow use."
            ),
            "",
        ]

        self._render_evaluation_summary(lines, results, ae, skill_name, private_labels)
        self._render_agents_used(lines, ae, private_labels)
        self._render_metrics_used(lines, ae, private_labels)
        self._render_test_tasks(lines, ae)
        self._render_results(lines, ae, private_labels)
        self._render_tier_summary(
            lines,
            "Tier 1: Static Validation Summary",
            _tier1_results(results),
            private_labels,
        )
        self._render_tier_summary(
            lines,
            "Tier 2: Deduplication Summary",
            _tier2_results(results),
            private_labels,
        )
        self._render_publication_recommendation(lines, results, ae)

        return "\n".join(lines).rstrip() + "\n"

    def _render_evaluation_summary(
        self,
        lines: list[str],
        results: list[ValidationResult],
        ae: dict[str, Any] | None,
        skill_name: str,
        private_labels: tuple[str, ...],
    ) -> None:
        lines.append("## Evaluation Summary")
        lines.append("")
        lines.append(f"- Skill: `{skill_name}`")
        if self.include_timestamp:
            date = datetime.now(tz=UTC).date().isoformat()
            lines.append(f"- Evaluation date: {date}")

        if ae:
            summary_value = ae.get("summary")
            summary = summary_value if isinstance(summary_value, dict) else {}
            environment = summary.get("environment") or ae.get("environment")
            if environment:
                lines.append(f"- Environment: `{_publication_safe_environment(environment)}`")

            dataset_count = _dataset_count(ae)
            if dataset_count is not None:
                lines.append(f"- Dataset: {dataset_count} evaluation tasks")

            attempt_policy_value = ae.get("attempt_policy")
            attempt_policy = attempt_policy_value if isinstance(attempt_policy_value, dict) else {}
            attempts = attempt_policy.get("max_attempts")
            if attempts is not None:
                lines.append(f"- Attempts per task: {_publication_safe_inline(attempts, private_labels)}")

            pass_threshold = attempt_policy.get("pass_threshold")
            if isinstance(pass_threshold, (int, float)):
                lines.append(f"- Pass threshold: {pass_threshold:.0%}")

            verdict = ae.get("verdict") or summary.get("verdict")
            combined = _combined_verdict(results, verdict)
            lines.append(f"- Overall verdict: {combined}")
            if skip_message := _advisory_agent_eval_skip_message(results):
                lines.append(
                    "- Tier 3 live evaluation: SKIPPED — "
                    f"{_publication_safe_inline(skip_message, private_labels)}"
                )
            if combined == "INCOMPLETE":
                _render_incomplete_benchmark_notes(lines, results, private_labels)
            elif combined == "FAIL":
                _render_failed_benchmark_notes(lines)
        else:
            combined = _combined_verdict(results, None)
            lines.append(f"- Overall verdict: {combined}")
            if combined == "INCOMPLETE":
                _render_incomplete_benchmark_notes(lines, results, private_labels)
            elif combined == "FAIL":
                _render_failed_benchmark_notes(lines)
            lines.append("- Tier 3 live agent evaluation: not available in this report")
        lines.append("")

    @staticmethod
    def _render_agents_used(
        lines: list[str],
        ae: dict[str, Any] | None,
        private_labels: tuple[str, ...],
    ) -> None:
        lines.append("## Agents Used")
        lines.append("")
        agents = _agents(ae)
        if not agents:
            lines.append("- Tier 3 agent details were not available in this report.")
        else:
            for name, agent in agents.items():
                lines.append(f"- {_agent_label(name, agent, private_labels)}")
        lines.append("")

    @staticmethod
    def _render_metrics_used(
        lines: list[str],
        ae: dict[str, Any] | None,
        private_labels: tuple[str, ...],
    ) -> None:
        lines.append("## Metrics Used")
        lines.append("")
        lines.append("Reported benchmark dimensions:")
        lines.append("")
        for dim_id in DIMENSION_MAPPING:
            desc = _DIMENSION_DESCRIPTIONS.get(dim_id) or DIMENSION_HINTS.get(dim_id, "")
            lines.append(f"- {dim_id.title()}: {desc}")
        lines.append("")
        lines.append("Underlying evaluation signals used in this run:")
        lines.append("")
        signals = _metric_signals(ae)
        if not signals:
            lines.append("- No Tier 3 evaluation signal details were available in this report.")
        else:
            labels = _metric_labels(ae)
            for signal in signals:
                safe_signal = _publication_safe_inline(signal, private_labels)
                label = _publication_safe_label(labels.get(signal, signal.replace("_", " ").title()), private_labels)
                desc = _SIGNAL_DESCRIPTIONS.get(signal, "captured by the Tier 3 evaluation payload.")
                lines.append(f"- `{safe_signal}` ({label}): {desc}")
        lines.append("")

    @staticmethod
    def _render_test_tasks(lines: list[str], ae: dict[str, Any] | None) -> None:
        lines.append("## Test Tasks")
        lines.append("")
        if not ae:
            lines.append("Tier 3 evaluation task details were not available in this report.")
            lines.append("")
            return

        dataset = _dataset(ae)
        if not dataset:
            trial_count = len(ae.get("trials") or [])
            if trial_count:
                lines.append(
                    f"The benchmark included {trial_count} recorded Tier 3 trials, but "
                    "the source evaluation dataset was not available in this report payload."
                )
            else:
                lines.append("The evaluation dataset was not available in this report payload.")
            lines.append("")
            return

        counts = _dataset_composition(dataset)
        lines.append(f"The benchmark dataset contained {len(dataset)} evaluation tasks:")
        lines.append("")
        lines.append(f"- Positive tasks: {counts['positive']} tasks where the skill was expected to activate.")
        lines.append(f"- Negative tasks: {counts['negative']} tasks where no skill was expected.")
        lines.append(
            f"- Unlabeled tasks: {counts['unlabeled']} tasks where positive/negative intent could not be inferred."
        )
        lines.append("")
        lines.append(
            "Task composition is derived from the evaluation dataset when possible. Entries "
            "with `expected_skill` set are treated as positive skill-activation cases, while "
            "entries with `expected_skill: null` are treated as negative activation cases."
        )
        lines.append("")

    @staticmethod
    def _render_results(
        lines: list[str],
        ae: dict[str, Any] | None,
        private_labels: tuple[str, ...],
    ) -> None:
        lines.append("## Results")
        lines.append("")
        agents = _agents(ae)
        if not agents:
            lines.append("Tier 3 dimension rollup was not available in this report.")
            lines.append("")
            return

        header = [
            "Dimension",
            "Num",
            *[_agent_label(name, agent, private_labels) for name, agent in agents.items()],
        ]
        lines.append("| " + " | ".join(_md_cell(cell, private_labels) for cell in header) + " |")
        lines.append("|---|---:|" + "|".join(["---:"] * len(agents)) + "|")
        for dim_id in DIMENSION_MAPPING:
            row = [dim_id.title(), _dimension_num(ae, dim_id)]
            for agent in agents.values():
                row.append(_score_lift_cell(_agent_dimension(agent, dim_id)))
            lines.append("| " + " | ".join(_md_cell(cell, private_labels) for cell in row) + " |")
        lines.append("")
        lines.append(
            "Score values show skill-assisted performance. Values in parentheses show "
            "uplift versus the no-skill baseline when baseline data is available."
        )
        lines.append("")

    def _render_tier_summary(
        self,
        lines: list[str],
        title: str,
        results: list[ValidationResult],
        private_labels: tuple[str, ...],
    ) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not results:
            lines.append("This tier was not run or did not produce findings in this report.")
            lines.append("")
            return

        incomplete = [r for r in results if r.is_incomplete]
        failures = [r for r in results if not r.passed]
        findings = [finding for result in results for finding in result.findings]
        high_count = _severity_count(findings, "high") + _severity_count(findings, "critical")
        if incomplete:
            tools = list(dict.fromkeys(tool for result in incomplete for tool in result.incomplete_scans))
            safe_tools = [_publication_safe_inline(tool, private_labels) for tool in tools]
            status = f"is incomplete because {', '.join(safe_tools)} did not produce trustworthy evidence"
        elif failures or high_count:
            status = "reported findings"
        elif findings:
            status = "passed with observations"
        else:
            status = "passed"
        lines.append(
            f"{title.split(':', 1)[0]} validation {status}. "
            f"SkillEvaluator ran {len(results)} checks and found "
            f"{len(findings)} total findings."
        )
        lines.append("")

        static_test_limitations = list(
            dict.fromkeys(
                message for result in results if (message := self._static_test_evidence_message(result)) is not None
            )
        )
        if static_test_limitations:
            lines.append("Test execution limitations:")
            lines.append("")
            for message in static_test_limitations:
                lines.append(f"- {_publication_safe_inline(message, private_labels)}")
            lines.append("")

        if not findings:
            lines.append("Notable observations:")
            lines.append("")
            for result in results[: self.max_findings_shown]:
                validator_name = _publication_safe_inline(result.validator_name, private_labels)
                if result.success_details:
                    message = _publication_safe_inline(result.success_details[0].message, private_labels)
                    lines.append(f"- {validator_name}: {message}")
                else:
                    lines.append(f"- {validator_name}: no findings reported.")
            lines.append("")
            return

        lines.append("Top findings:")
        lines.append("")
        for finding in _top_findings(findings, limit=self.max_findings_shown):
            loc = f" (`{_publication_safe_location(finding)}`)" if finding.file_path else ""
            category = _publication_safe_inline(finding.category, private_labels)
            check_name = _publication_safe_inline(finding.check_name, private_labels)
            message = _publication_safe_inline(finding.message, private_labels)
            lines.append(
                f"- {finding.severity.value.upper()} {category}/{check_name}: {message}{loc}"
            )
        lines.append("")

    @staticmethod
    def _static_test_evidence_message(result: ValidationResult) -> str | None:
        """Return the static test limitation from direct or folder-aggregated results."""
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
    def _render_publication_recommendation(
        lines: list[str],
        results: list[ValidationResult],
        ae: dict[str, Any] | None,
    ) -> None:
        verdict = (ae or {}).get("verdict")
        if _combined_verdict(results, verdict) != "PASS":
            return

        lines.append("## Publication Recommendation")
        lines.append("")
        if _advisory_agent_eval_skip_message(results):
            lines.append(
                "Tier 3 live evaluation was skipped and does not block required validation. "
                "Publication suitability in this report is based on the completed required-tier "
                "results; rerun Tier 3 when the live evaluation runtime is available."
            )
        else:
            lines.append(
                "The skill is suitable to proceed toward SkillEvaluator publication "
                "based on this benchmark. Skill owners should keep this file with the "
                "skill and refresh it when the evaluation dataset, skill behavior, or "
                "target agents materially change."
            )
        lines.append("")

    def get_file_extension(self) -> str:
        return ".md"


def _agent_eval_payload(results: list[ValidationResult]) -> dict[str, Any] | None:
    for result in results:
        payload = result.metadata.get("agent_eval") if isinstance(result.metadata, dict) else None
        if isinstance(payload, dict):
            return payload
    return None


def _render_failed_benchmark_notes(lines: list[str]) -> None:
    lines.append(
        "The skill should be reviewed before SkillEvaluator publication. "
        "**Skill owners should address the applicable findings below and rerun "
        "SkillEvaluator to refresh this benchmark.**"
    )


def _advisory_agent_eval_skip_message(results: list[ValidationResult]) -> str | None:
    for result in results:
        if not is_advisory_agent_eval_skip(result):
            continue
        payload = result.metadata.get("agent_eval", {}) if result.metadata else {}
        provenance = payload.get("provenance", {}) if isinstance(payload, dict) else {}
        message = provenance.get("message") if isinstance(provenance, dict) else None
        return str(message or "Live evaluation did not run.")
    return None


def _combined_verdict(results: list[ValidationResult], tier3_verdict: object) -> str:
    """Return one report verdict with incomplete evidence taking precedence."""
    if any(result.is_incomplete for result in results):
        return "INCOMPLETE"
    if not all(passes_required_gate(result) for result in results) or str(tier3_verdict or "pass").lower() == "fail":
        return "FAIL"
    return "PASS"


def _render_incomplete_benchmark_notes(
    lines: list[str],
    results: list[ValidationResult],
    private_labels: tuple[str, ...],
) -> None:
    tools = list(dict.fromkeys(tool for result in results for tool in result.incomplete_scans))
    safe_tools = [_publication_safe_inline(tool, private_labels) for tool in tools]
    lines.append(
        "Required scanner evidence is incomplete "
        f"({', '.join(safe_tools)}). **Do not use this benchmark to recommend publication; "
        "restore the scanners and rerun SkillEvaluator.**"
    )


def _skill_name(results: list[ValidationResult], ae: dict[str, Any] | None) -> str:
    if ae:
        summary_value = ae.get("summary")
        summary = summary_value if isinstance(summary_value, dict) else {}
        candidate = ae.get("skill_name") or summary.get("skill_name")
        if candidate:
            return str(candidate)
    for result in results:
        quality = result.metadata.get("quality_scores") if isinstance(result.metadata, dict) else None
        if isinstance(quality, dict) and quality.get("skill_name"):
            return str(quality["skill_name"])
    return "skill"


def _agents(ae: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not ae:
        return {}
    agents = ae.get("agents")
    if isinstance(agents, dict) and agents:
        return {str(name): value for name, value in agents.items() if isinstance(value, dict)}
    return {}


def _agent_label(name: str, agent: dict[str, Any], private_labels: tuple[str, ...]) -> str:
    display = agent.get("display_name") or agent.get("label") or name
    model = agent.get("model") or agent.get("model_name") or agent.get("llm_model")
    safe_name = _publication_safe_label(name, private_labels)
    safe_display = _publication_safe_label(display, private_labels)
    if model:
        safe_model = _publication_safe_label(model, private_labels)
        return f"{_human_agent_name(safe_display)} (`{safe_model}`)"
    return f"`{safe_name}`" if display == name else _human_agent_name(safe_display)


def _human_agent_name(name: str) -> str:
    if name == "claude-code":
        return "Claude Code"
    if name.startswith("\\"):
        return name.title()
    return name.replace("_", " ").replace("-", " ").title()


def _metric_labels(ae: dict[str, Any] | None) -> dict[str, str]:
    labels = (ae or {}).get("metric_labels")
    return labels if isinstance(labels, dict) else {}


def _metric_signals(ae: dict[str, Any] | None) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for agent in _agents(ae).values():
        evaluators = agent.get("evaluators") if isinstance(agent, dict) else None
        if not isinstance(evaluators, dict):
            continue
        for key, value in evaluators.items():
            if key in seen or not isinstance(value, dict):
                continue
            if any(value.get(field) is not None for field in ("with_skill", "baseline", "lift")):
                seen.add(str(key))
                ordered.append(str(key))
    if ordered:
        return ordered
    metric_ids = (ae or {}).get("metric_ids")
    return [str(item) for item in metric_ids] if isinstance(metric_ids, list) else []


def _dataset(ae: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = ae.get("dataset")
    if isinstance(dataset, list):
        return [item for item in dataset if isinstance(item, dict)]
    return []


def _dataset_count(ae: dict[str, Any]) -> int | None:
    dataset = _dataset(ae)
    if dataset:
        return len(dataset)
    trials = ae.get("trials")
    if isinstance(trials, list) and trials:
        return len(trials)
    return None


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


def _dimension_num(ae: dict[str, Any] | None, dim_id: str) -> str:
    explicit = _explicit_dimension_num(ae, dim_id)
    if explicit is not None:
        return explicit
    evidence_count = _dimension_evidence_count(ae, dim_id)
    if evidence_count is not None:
        return str(evidence_count)
    dataset_size = _dataset_count(ae or {})
    return str(dataset_size) if dataset_size is not None else "N/A"


def _explicit_dimension_num(ae: dict[str, Any] | None, dim_id: str) -> str | None:
    for agent in _agents(ae).values():
        dim = _agent_dimension(agent, dim_id)
        if not dim:
            continue
        for key in ("num", "n", "sample_count", "samples"):
            value = dim.get(key)
            if value is not None:
                return str(value)
    return None


def _dimension_evidence_count(ae: dict[str, Any] | None, dim_id: str) -> int | None:
    counts: list[int] = []
    for agent in _agents(ae).values():
        metric_ids = _dimension_metric_ids(agent, dim_id)
        for card in agent.get("evaluator_cards") or []:
            if not isinstance(card, dict) or card.get("id") not in metric_ids:
                continue
            evidence = card.get("evidence")
            if isinstance(evidence, list) and evidence:
                counts.append(len(evidence))
    return max(counts) if counts else None


def _dimension_metric_ids(agent: dict[str, Any], dim_id: str) -> list[str]:
    dim = _agent_dimension(agent, dim_id)
    if isinstance(dim, dict) and isinstance(dim.get("evaluators"), list) and dim["evaluators"]:
        return [str(item) for item in dim["evaluators"]]
    return [str(item) for item in DIMENSION_MAPPING.get(dim_id, {}).get("evaluators") or []]


def _agent_dimension(agent: dict[str, Any], dim_id: str) -> dict[str, Any] | None:
    for dim in agent.get("dimensions") or []:
        if isinstance(dim, dict) and dim.get("id") == dim_id:
            return dim
    return None


def _score_lift_cell(dim: dict[str, Any] | None) -> str:
    if not dim:
        return "N/A"
    score = dim.get("with_skill", dim.get("score"))
    if not isinstance(score, (int, float)):
        return "N/A"
    value = f"{score:.0%}"
    lift = dim.get("lift")
    if isinstance(lift, (int, float)):
        value += f" ({lift:+.0%})"
    return value


def _tier1_results(results: list[ValidationResult]) -> list[ValidationResult]:
    return [r for r in results if not _is_tier2(r) and not _is_tier3(r)]


def _tier2_results(results: list[ValidationResult]) -> list[ValidationResult]:
    return [r for r in results if _is_tier2(r)]


def _is_tier2(result: ValidationResult) -> bool:
    name = result.validator_name.lower()
    if name in _TIER2_VALIDATORS or "dedup" in name:
        return True
    return any(f.category == "CONTENT_DEDUP" for f in result.findings)


def _is_tier3(result: ValidationResult) -> bool:
    return bool(result.metadata.get("agent_eval")) or result.validator_name == "AGENT_EVAL"


def _severity_count(findings: list[Finding], severity: str) -> int:
    return sum(1 for finding in findings if finding.severity.value == severity)


def _top_findings(findings: list[Finding], *, limit: int) -> list[Finding]:
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    return sorted(findings, key=lambda f: (order.get(f.severity.value, 99), f.category))[:limit]


def _md_cell(value: object, private_labels: tuple[str, ...] = ()) -> str:
    return _publication_safe_inline(value, private_labels).replace("|", "\\|")


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
    summary_value = ae.get("summary")
    summary = summary_value if isinstance(summary_value, dict) else {}
    candidates = [summary.get("environment"), ae.get("environment")]
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
    """Keep public environment names and generalize unknown imported labels."""
    environment = str(value).strip()
    return environment if environment.casefold() in HARBOR_ENV_MODES else "Isolated sandbox"


def _publication_safe_location(finding: Finding) -> str:
    """Render a finding location without publishing an absolute host path."""
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
