# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the generated BENCHMARK.md publication gate."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, tzinfo
from pathlib import Path

import pytest
from scripts.ci import check_public_benchmarks as benchmark_gate

from skillevaluator.evaluation.tier3_report import (
    _validation_result_from_payload,
    advisory_skip_result,
    build_agent_eval_payload,
)
from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.publication_evidence import stamp_publication_evidence
from skillevaluator.reporting import BenchmarkReporter

_PUBLICATION_TARGET = {
    "skill_name": "demo-skill",
    "skill_digest": "sha256:" + "b" * 64,
    "skill_digest_algorithm": "skill-evaluator-source-tree/2",
}
_TIER3_RUN_ID = "run-demo-skill-fixture"


def _freeze_gate_clock(
    monkeypatch: pytest.MonkeyPatch,
    instant: datetime,
) -> None:
    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            del cls
            return instant.replace(tzinfo=None) if tz is None else instant.astimezone(tz)

    monkeypatch.setattr(benchmark_gate, "datetime", _FrozenDateTime)


def _bind_publication_target(result: ValidationResult) -> ValidationResult:
    fixture_producers = {
        "Schema & Repository Governance": (1, "schema"),
        "Similarity Check": (2, "similarity"),
    }
    result.metadata["publication_target"] = dict(_PUBLICATION_TARGET)
    if "publication_evidence" not in result.metadata and result.validator_name in fixture_producers:
        tier, check_id = fixture_producers[result.validator_name]
        stamp_publication_evidence([result], tier=tier, check_id=check_id)
    payload = result.metadata.get("agent_eval")
    if isinstance(payload, dict):
        payload["publication_target"] = dict(_PUBLICATION_TARGET)
        summary = payload.get("summary")
        if isinstance(summary, dict):
            summary["publication_target"] = dict(_PUBLICATION_TARGET)
    return result


def _private_sandbox_name() -> str:
    """Build the retired private label without embedding it in public source."""
    private_name = chr(65) + "stra"
    return f"{private_name} sandbox"


def _valid_benchmark() -> str:
    return """\
# Skill Benchmark: demo

> **Overall verdict: PASS**

## Evaluation Metadata

- Source digest: `sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` (skill-evaluator-source-tree/2)
- Evaluation date: 2026-07-25
- Evaluator version: `0.8.3`
- Agents: Codex (`gpt-codex`)
- Tasks: 4 evaluation tasks
- Dataset digest: `sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef` (skill-evaluator-dataset-snapshot/1)
- Tier 3 run ID: `run-fixture-001`
- Attempts per task: 1
- Environment: `Isolated sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

## Results at a Glance

| Measure | Codex (Baseline → Skill Uplift) |
|---|---:|
| Overall | 47% → 92% (+45 points) |

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED** | Complete |
| Tier 2 | Semantic deduplication | **PASSED** | Complete |
| Tier 3 | Live agent evaluation | **PASS** | Complete |

## Freshness

Regenerate after material inputs change.
"""


def test_gate_accepts_redesigned_public_card(tmp_path: Path) -> None:
    benchmark = tmp_path / "skill" / "BENCHMARK.md"
    benchmark.parent.mkdir()
    benchmark.write_text(_valid_benchmark(), encoding="utf-8")

    files, offenders = benchmark_gate.find_offenders([tmp_path])

    assert files == [benchmark.resolve()]
    assert offenders == []


@pytest.mark.parametrize(
    "evaluation_date",
    ["2026-02-30", "2026-13-01", "0000-01-01", "9999-99-99"],
)
def test_gate_rejects_calendar_impossible_evaluation_dates(
    tmp_path: Path,
    evaluation_date: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace("- Evaluation date: 2026-07-25", f"- Evaluation date: {evaluation_date}"),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {
        "invalid metadata field: - Evaluation date:",
        "publication PASS without recorded evaluation date",
    }


def test_gate_rejects_future_evaluation_date_beyond_utc_skew(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_gate_clock(monkeypatch, datetime(2026, 7, 25, 12, tzinfo=UTC))
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace("- Evaluation date: 2026-07-25", "- Evaluation date: 2026-07-26"),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {
        "invalid metadata field: - Evaluation date:",
        "publication PASS without recorded evaluation date",
    }


def test_gate_accepts_next_utc_date_within_clock_skew(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_gate_clock(monkeypatch, datetime(2026, 7, 25, 23, 58, tzinfo=UTC))
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace("- Evaluation date: 2026-07-25", "- Evaluation date: 2026-07-26"),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert offenders == []


def test_gate_accepts_valid_leap_day_evaluation_date(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace("- Evaluation date: 2026-07-25", "- Evaluation date: 2024-02-29"),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert offenders == []


@pytest.mark.parametrize(
    ("field", "control"),
    [
        ("Source digest", "\x00"),
        ("Tier 3 run ID", "\x1b"),
    ],
    ids=["nul-in-source-digest", "escape-in-run-id"],
)
def test_gate_rejects_control_characters_in_publication_proof_fields(
    tmp_path: Path,
    field: str,
    control: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    text = _valid_benchmark()
    marker = "sha256:" if field == "Source digest" else "run-fixture-001"
    replacement = f"sha{control}256:" if field == "Source digest" else f"run-fixture-{control}001"
    benchmark.write_text(text.replace(marker, replacement, 1), encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "benchmark contains disallowed control character" in {offender.reason for offender in offenders}


def test_gate_rejects_legacy_and_private_output(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark()
        + f"\n- Finding: Observed in {_private_sandbox_name()} during evaluation\n"
        + "- Skill Evaluator profile: external\n"
        + "- Finding: `/Users/private/repo/SKILL.md`\n"
        + "Generated by legacy-skills-eval v1.2\n"
        + "| Dimension | Num | Codex |\n"
        + "| Security | 8 | 92% (+45%) |\n",
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])
    reasons = {offender.reason for offender in offenders}

    assert reasons == {
        "internal environment identity",
        "validation profile metadata",
        "retired product identity",
        "absolute macOS user path",
        "legacy Num score column",
        "legacy ambiguous uplift cell",
    }


def test_gate_rejects_old_card_missing_decision_sections(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text("# Evaluation Report\n", encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    reasons = {offender.reason for offender in offenders}
    assert "missing required section: # Skill Benchmark:" in reasons
    assert "missing required section: ## Evaluation Metadata" in reasons
    assert "missing required section: - Evaluation date:" in reasons
    assert "missing required section: - Evaluator version:" in reasons
    assert "missing required section: - Agents:" in reasons
    assert "missing required section: - Tasks:" in reasons
    assert "missing required section: - Attempts per task:" in reasons
    assert "missing required section: - Environment:" in reasons
    assert "missing Tier 3 status row" in reasons


@pytest.mark.parametrize(
    ("marker", "hidden_marker"),
    [
        ("# Skill Benchmark: demo", "<!-- # Skill Benchmark: demo -->"),
        ("## Evaluation Metadata", "```markdown\n## Evaluation Metadata\n```"),
        ("## Results at a Glance", "<!-- ## Results at a Glance -->"),
        ("## Tier Status", "```markdown\n## Tier Status\n```"),
        ("## Freshness", "<!-- ## Freshness -->"),
    ],
    ids=["title", "metadata", "results", "tier-status", "freshness"],
)
def test_gate_requires_visible_structural_headings(
    tmp_path: Path,
    marker: str,
    hidden_marker: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(_valid_benchmark().replace(marker, hidden_marker), encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    expected_marker = "# Skill Benchmark:" if marker.startswith("# Skill") else marker
    assert f"missing required section: {expected_marker}" in {offender.reason for offender in offenders}


def test_gate_requires_explicit_agent_model_state(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(_valid_benchmark().replace("Agents: Codex (`gpt-codex`)", "Agents: Codex"), encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {
        "agent model identity not recorded",
        "publication PASS without recorded agent model identity",
    }


def test_gate_rejects_pass_when_any_agent_model_is_not_recorded(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace(
            "Agents: Codex (`gpt-codex`)",
            "Agents: Codex (`gpt-codex`), Claude Code (model not recorded)",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {"publication PASS without recorded agent model identity"}


@pytest.mark.parametrize(
    "agent_identity",
    [
        "&#8203;",
        "&ZeroWidthSpace;",
        "&nbsp;",
        "&#x3164;",
        "&#10240;",
        "**\u200b**",
        "__&nbsp;__",
        "*\u200b*",
        "_&nbsp;_",
    ],
    ids=[
        "decimal-zero-width",
        "named-zero-width",
        "nonbreaking-space",
        "hangul-filler",
        "braille-blank",
        "strong-control",
        "strong-entity",
        "em-control",
        "em-entity",
    ],
)
def test_gate_rejects_visually_empty_entity_or_emphasis_agent_identity(
    tmp_path: Path,
    agent_identity: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace(
            "Agents: Codex (`gpt-codex`)",
            f"Agents: {agent_identity} (`gpt-codex`)",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    reasons = {offender.reason for offender in offenders}
    assert "agent model identity not recorded" in reasons
    assert "publication PASS without recorded agent model identity" in reasons


def test_gate_decodes_visible_agent_entities_without_losing_identity(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace(
            "Agents: Codex (`gpt-codex`)",
            "Agents: Caf&eacute; (`gpt-codex`)",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert offenders == []


@pytest.mark.parametrize(
    "agents",
    [
        "requested but not run — Codex (model not recorded)",
        "requested but not run — Alpha, Beta (model not recorded)",
        "requested but not run — Alpha, junk model not recorded",
        "requested but not run — , (model not recorded)",
    ],
)
def test_pass_cannot_use_requested_but_unrecorded_agent_state(
    tmp_path: Path,
    agents: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace("Agents: Codex (`gpt-codex`)", f"Agents: {agents}"),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "publication PASS without recorded agent model identity" in {offender.reason for offender in offenders}


@pytest.mark.parametrize(
    "field",
    [
        "Evaluation date",
        "Evaluator version",
        "Tasks",
        "Dataset digest",
        "Attempts per task",
        "Environment",
        "Tier 3 evidence",
    ],
)
def test_gate_rejects_blank_required_metadata_values(tmp_path: Path, field: str) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    lines = [f"- {field}:" if line.startswith(f"- {field}:") else line for line in _valid_benchmark().splitlines()]
    benchmark.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert f"invalid metadata field: - {field}:" in {offender.reason for offender in offenders}


def test_gate_rejects_blank_present_tier2_policy_metadata(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace(
            "- Tier 2 evidence: required for publication",
            "- Tier 2 evidence:",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {"invalid metadata field: - Tier 2 evidence:"}


@pytest.mark.parametrize("tier", [2, 3])
def test_gate_requires_exact_publication_policy_value_casing(tmp_path: Path, tier: int) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace(
            f"- Tier {tier} evidence: required for publication",
            f"- Tier {tier} evidence: Optional by policy",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {f"invalid metadata field: - Tier {tier} evidence:"}


def test_gate_requires_metadata_fields_inside_metadata_section(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    date_line = "- Evaluation date: 2026-07-25"
    content = _valid_benchmark().replace(f"{date_line}\n", "")
    benchmark.write_text(content + f"\n{date_line}\n", encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "missing metadata field: - Evaluation date:" in {offender.reason for offender in offenders}


@pytest.mark.parametrize(
    "profile_line",
    [
        "- Profile: internal",
        "- Profile: `external`",
        "- Profile: team-strict",
        "- Skill Evaluator profile: external",
        "- skill evaluator Profile: `internal`",
        "- Skill Evaluator Profile: `custom-publication-policy` (generated)",
    ],
)
def test_gate_rejects_all_profile_metadata_shapes(tmp_path: Path, profile_line: str) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(_valid_benchmark() + f"\n{profile_line}\n", encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "validation profile metadata" in {offender.reason for offender in offenders}


def test_gate_rejects_internal_sandbox_phrase_anywhere_but_preserves_astra_db(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark()
        + "\n- Skill: `astra-db`\n"
        + f"- Finding: Observed in {_private_sandbox_name()} during evaluation\n",
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert [offender.reason for offender in offenders] == ["internal environment identity"]


def _tier1_result() -> ValidationResult:
    result = ValidationResult(validator_name="Schema & Repository Governance")
    result.metadata["quality_scores"] = {"skill_name": "demo-skill"}
    result.add_success("schema", "Schema passed")
    return _bind_publication_target(result)


def _tier2_result() -> ValidationResult:
    result = ValidationResult(
        validator_name="Similarity Check",
        validator_description="Detect duplicate content via embedding similarity",
    )
    result.add_success("similarity_scan", "No duplicate content found")
    return _bind_publication_target(result)


def _tier3_result(verdict: str, *, legacy: bool = False) -> ValidationResult:
    result = ValidationResult(validator_name="AGENT_EVAL")
    payload: dict = {
        "skill_name": "demo-skill",
        "publication_target": dict(_PUBLICATION_TARGET),
        "verdict": verdict.lower(),
        "summary": {
            "verdict": verdict.lower(),
            "environment": "docker",
            "publication_target": dict(_PUBLICATION_TARGET),
        },
        "agents": {"codex": {} if legacy else {"model": "gpt-codex"}},
    }
    if not legacy:
        dimension_score = {"pass": 0.9, "neutral": 0.45, "fail": 0.3}[verdict.lower()]
        payload["agents"]["codex"].update(
            {
                "execution_status": "succeeded",
                "expected_attempts": 4,
                "scored_attempts": 4,
                "dimensions": [
                    {"id": dimension, "with_skill": dimension_score}
                    for dimension in ("security", "correctness", "discoverability", "effectiveness", "efficiency")
                ],
            }
        )
        payload.update(
            {
                "execution_status": "succeeded",
                "run_id": _TIER3_RUN_ID,
                "evaluated_at": "2026-07-25T12:00:00+00:00",
                "evaluator_version": "0.8.3",
                "expected_attempts": 4,
                "scored_attempts": 4,
                "dataset_summary": {
                    "total_tasks": 4,
                    "positive_tasks": 3,
                    "negative_tasks": 1,
                    "unclassified_tasks": 0,
                    "source": "dataset",
                },
                "dataset_digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                "dataset_digest_algorithm": "skill-evaluator-dataset-snapshot/1",
                "attempt_policy": {"max_attempts": 1, "pass_threshold": 0.5},
            }
        )
        payload["summary"].update(
            {
                "execution_status": "succeeded",
                "run_id": _TIER3_RUN_ID,
                "expected_attempts": 4,
                "scored_attempts": 4,
            }
        )
    result.metadata["agent_eval"] = payload
    result.add_success("agent_eval", "Live evaluation completed")
    return _bind_publication_target(result)


@pytest.mark.parametrize("verdict", ["PASS", "FAIL", "NEUTRAL"])
def test_real_reporter_cards_pass_publication_gate(tmp_path: Path, verdict: str) -> None:
    rendered = BenchmarkReporter(include_timestamp=True).render_all(
        [_tier1_result(), _tier2_result(), _tier3_result(verdict)]
    )
    benchmark = tmp_path / verdict.lower() / "BENCHMARK.md"
    benchmark.parent.mkdir()
    benchmark.write_text(rendered, encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert f"Overall verdict: {verdict}" in rendered
    assert offenders == []


def test_real_reporter_normalizes_positive_offset_evaluation_date_to_utc(tmp_path: Path) -> None:
    tier3 = _tier3_result("PASS")
    tier3.metadata["agent_eval"]["evaluated_at"] = "2026-07-25T00:02:00+14:00"

    rendered = BenchmarkReporter(include_timestamp=True).render_all([_tier1_result(), _tier2_result(), tier3])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "- Evaluation date: 2026-07-24" in rendered
    assert offenders == []


@pytest.mark.parametrize(
    ("field", "encoded_comma"),
    [("agent-name", True), ("display-name", True), ("model", False)],
)
def test_generated_pass_card_encodes_commas_in_agent_identity(
    tmp_path: Path,
    field: str,
    encoded_comma: bool,
) -> None:
    tier3 = _tier3_result("PASS")
    agent = tier3.metadata["agent_eval"]["agents"]["codex"]
    if field == "agent-name":
        tier3.metadata["agent_eval"]["agents"] = {"codex,primary": agent}
    elif field == "display-name":
        agent["display_name"] = "Codex, Primary"
    else:
        agent["model"] = "gpt-codex,stable"

    rendered = BenchmarkReporter(include_timestamp=True).render_all([_tier1_result(), _tier2_result(), tier3])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "Overall verdict: PASS" in rendered
    assert ("&#44;" in rendered) is encoded_comma
    if field == "model":
        assert "`gpt-codex,stable`" in rendered
    assert offenders == []


@pytest.mark.parametrize(
    ("dimension_score", "expected_verdict"),
    [(0.40, "NEUTRAL"), (0.50, "PASS")],
)
def test_dimension_verdict_boundaries_pass_publication_gate(
    tmp_path: Path,
    dimension_score: float,
    expected_verdict: str,
) -> None:
    tier3 = _tier3_result("PASS")
    for dimension in tier3.metadata["agent_eval"]["agents"]["codex"]["dimensions"]:
        dimension["with_skill"] = dimension_score
    rendered = BenchmarkReporter(include_timestamp=True).render_all([_tier1_result(), _tier2_result(), tier3])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert f"Overall verdict: {expected_verdict}" in rendered
    assert f"| Tier 3 | Live agent evaluation | **{expected_verdict}** |" in rendered
    assert offenders == []


def test_real_incomplete_and_legacy_cards_pass_publication_gate(tmp_path: Path) -> None:
    cards = {
        "incomplete": BenchmarkReporter(include_timestamp=True).render_all([_tier1_result(), _tier2_result()]),
        "legacy": BenchmarkReporter(include_timestamp=True).render_all(
            [_tier1_result(), _tier2_result(), _tier3_result("NEUTRAL", legacy=True)]
        ),
    }
    for name, rendered in cards.items():
        benchmark = tmp_path / name / "BENCHMARK.md"
        benchmark.parent.mkdir()
        benchmark.write_text(rendered, encoding="utf-8")
        _files, offenders = benchmark_gate.find_offenders([benchmark])
        assert offenders == [], f"{name}: {offenders}"


def test_legacy_pass_is_incomplete_without_publication_provenance(tmp_path: Path) -> None:
    rendered = BenchmarkReporter(include_timestamp=True).render_all(
        [_tier1_result(), _tier2_result(), _tier3_result("PASS", legacy=True)]
    )
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "## Publication Recommendation" not in rendered
    assert offenders == []


def test_legacy_neutral_is_incomplete_without_required_publication_provenance(tmp_path: Path) -> None:
    rendered = BenchmarkReporter(include_timestamp=True).render_all(
        [_tier1_result(), _tier2_result(), _tier3_result("NEUTRAL", legacy=True)]
    )
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** |" in rendered
    assert offenders == []


def test_malformed_canonical_pass_payload_is_incomplete_and_lints_cleanly(tmp_path: Path) -> None:
    payload = deepcopy(_tier3_result("PASS").metadata["agent_eval"])
    payload["overall_score"] = 0.9
    payload["agents"]["codex"]["dimensions"] = []
    result = _validation_result_from_payload(payload)
    assert result is not None

    rendered = BenchmarkReporter(include_timestamp=True).render_all([_tier1_result(), _tier2_result(), result])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** |" in rendered
    assert offenders == []


def test_canonical_multi_agent_pass_survives_partial_peer_evidence(tmp_path: Path) -> None:
    passing_scores = {
        "security": 0.9,
        "skill_execution": 0.9,
        "skill_efficiency": 0.9,
        "accuracy": 0.9,
        "goal_accuracy": 0.9,
        "behavior_check": 0.9,
    }
    baseline_scores = dict.fromkeys(passing_scores, 0.5)
    partial_scores = dict(passing_scores)
    partial_scores.pop("accuracy")
    partial_baseline = dict(baseline_scores)
    partial_baseline.pop("accuracy")
    agents = {
        "codex": {
            "model": "gpt-codex",
            "with_skill": passing_scores,
            "without_skill": baseline_scores,
            "execution_status": "succeeded",
            "rewards": [],
            "num_trials": 1,
            "expected_attempts": 1,
            "scored_attempts": 1,
        },
        "claude-code": {
            "model": "claude-sonnet",
            "with_skill": partial_scores,
            "without_skill": partial_baseline,
            "execution_status": "succeeded",
            "rewards": [],
            "num_trials": 1,
            "expected_attempts": 1,
            "scored_attempts": 1,
        },
    }
    payload = build_agent_eval_payload(
        "demo-skill",
        agents,
        dataset=[{"id": "case-1", "expected_skill": "demo-skill"}],
        env_mode="docker",
        evaluated_at="2026-07-25T12:00:00+00:00",
        run_id=_TIER3_RUN_ID,
        publication_target=dict(_PUBLICATION_TARGET),
        use_llm_judge=False,
    )
    assert payload is not None
    assert payload["verdict"] == "pass"
    result = _validation_result_from_payload(payload)
    assert result is not None

    rendered = BenchmarkReporter(include_timestamp=True).render_all([_tier1_result(), _tier2_result(), result])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "Overall verdict: PASS" in rendered
    assert "| Tier 3 | Live agent evaluation | **PASS** |" in rendered
    assert offenders == []


@pytest.mark.parametrize(
    ("tier3_required", "expected_verdict"),
    [(True, "INCOMPLETE"), (False, "PASS")],
)
def test_advisory_tier3_skip_respects_publication_policy(
    tmp_path: Path,
    tier3_required: bool,
    expected_verdict: str,
) -> None:
    tier1 = _tier1_result()
    tier1.metadata["benchmark_policy"] = {"tier3_required": tier3_required}
    rendered = BenchmarkReporter(include_timestamp=True).render_all(
        [
            tier1,
            _tier2_result(),
            advisory_skip_result("Live evaluation runtime unavailable", skill_name="demo-skill"),
        ]
    )
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert f"Overall verdict: {expected_verdict}" in rendered
    assert "| Tier 3 | Live agent evaluation | **SKIPPED (ADVISORY)** |" in rendered
    assert offenders == []


@pytest.mark.parametrize("policy_location", ["tier1", "agent_eval", "summary"])
def test_optional_policy_does_not_certify_malformed_present_tier3(
    tmp_path: Path,
    policy_location: str,
) -> None:
    tier1 = _tier1_result()
    tier3 = _tier3_result("PASS", legacy=True)
    if policy_location == "tier1":
        tier1.metadata["benchmark_policy"] = {"tier3_required": False}
    elif policy_location == "agent_eval":
        tier3.metadata["agent_eval"]["benchmark_policy"] = {"tier3_required": False}
    else:
        tier3.metadata["agent_eval"]["summary"]["benchmark_policy"] = {"tier3_required": False}

    rendered = BenchmarkReporter(include_timestamp=True).render_all([tier1, _tier2_result(), tier3])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "- Tier 3 evidence: optional by policy" in rendered
    assert "Overall verdict: INCOMPLETE" in rendered
    assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** |" in rendered
    assert "Tier 3 truth fields are missing, invalid, or contradictory." in rendered
    assert offenders == []


def test_reporter_without_generation_timestamp_passes_publication_gate(tmp_path: Path) -> None:
    rendered = BenchmarkReporter(include_timestamp=False).render_all([_tier1_result(), _tier2_result()])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "- Evaluation date: not recorded (legacy or non-live result)" in rendered
    assert offenders == []


def test_reporter_sanitizes_finding_before_publication_gate(tmp_path: Path) -> None:
    tier3 = _tier3_result("PASS")
    tier3.add_finding(
        Finding(
            category="AGENT_EVAL",
            severity=Severity.LOW,
            check_name="environment",
            message=f"Observed in {_private_sandbox_name()} during evaluation",
            file_path=None,
        )
    )
    rendered = BenchmarkReporter(include_timestamp=True).render_all([_tier1_result(), _tier2_result(), tier3])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert _private_sandbox_name() not in rendered
    assert "Observed in isolated sandbox during evaluation" in rendered
    assert offenders == []


def test_reporter_sanitizes_requested_environment_from_advisory_payload(tmp_path: Path) -> None:
    private_environment = "secret-cluster"
    result = advisory_skip_result(
        f"Runtime unavailable in {private_environment}",
        skill_name="demo-skill",
    )
    result.metadata["agent_eval"]["requested_environment"] = private_environment
    rendered = BenchmarkReporter(include_timestamp=True).render_all([_tier1_result(), _tier2_result(), result])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert private_environment not in rendered
    assert "Runtime unavailable in Isolated sandbox" in rendered
    assert offenders == []


@pytest.mark.parametrize("requested_environment_location", ["agent_eval", "summary"])
def test_requested_environment_is_not_completed_run_provenance(
    tmp_path: Path,
    requested_environment_location: str,
) -> None:
    private_environment = "secret-cluster"
    tier3 = _tier3_result("PASS")
    payload = tier3.metadata["agent_eval"]
    payload["summary"].pop("environment")
    target = payload if requested_environment_location == "agent_eval" else payload["summary"]
    target["requested_environment"] = private_environment

    rendered = BenchmarkReporter(include_timestamp=True).render_all([_tier1_result(), _tier2_result(), tier3])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert private_environment not in rendered
    assert "- Environment: not recorded (legacy or non-live result)" in rendered
    assert "Overall verdict: INCOMPLETE" in rendered
    assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** |" in rendered
    assert offenders == []


@pytest.mark.parametrize(
    "private_path",
    [
        "/Users/private/catalog/SKILL.md",
        "/home/private/catalog/SKILL.md",
        r"C:\Users\private\catalog\SKILL.md",
    ],
)
def test_reporter_sanitizes_absolute_paths_inside_finding_messages(
    tmp_path: Path,
    private_path: str,
) -> None:
    tier3 = _tier3_result("PASS")
    tier3.add_finding(
        Finding(
            category="AGENT_EVAL",
            severity=Severity.LOW,
            check_name="private_path",
            message=f"Read private artifact at {private_path}",
            file_path=None,
        )
    )
    rendered = BenchmarkReporter(include_timestamp=True).render_all([_tier1_result(), _tier2_result(), tier3])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert private_path not in rendered
    assert offenders == []


def test_real_optional_tier3_policy_card_passes_gate(tmp_path: Path) -> None:
    tier1 = _tier1_result()
    tier1.metadata["benchmark_policy"] = {"tier3_required": False}
    rendered = BenchmarkReporter(include_timestamp=True).render_all([tier1, _tier2_result()])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "Overall verdict: PASS" in rendered
    assert "- Tier 3 evidence: optional by policy" in rendered
    assert offenders == []


@pytest.mark.parametrize(
    ("replacement", "expected_reason"),
    [
        (
            "| Tier 1 | Static validation | **NOT RUN** | No result was recorded |",
            "publication PASS without completed Tier 1 evidence",
        ),
        ("", "missing Tier 1 status row"),
    ],
    ids=["not-run", "missing-row"],
)
def test_gate_rejects_pass_without_completed_tier1(
    tmp_path: Path,
    replacement: str,
    expected_reason: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    completed_row = "| Tier 1 | Static validation | **PASSED** | Complete |"
    replacement_with_newline = f"{replacement}\n" if replacement else ""
    benchmark.write_text(
        _valid_benchmark().replace(
            f"{completed_row}\n",
            replacement_with_newline,
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {expected_reason}


@pytest.mark.parametrize(
    ("replacement", "expected_reason"),
    [
        (
            "| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |",
            "publication PASS without completed Tier 2 evidence",
        ),
        ("", "missing Tier 2 status row"),
    ],
    ids=["not-run", "missing-row"],
)
def test_gate_rejects_pass_without_completed_required_tier2(
    tmp_path: Path,
    replacement: str,
    expected_reason: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    completed_row = "| Tier 2 | Semantic deduplication | **PASSED** | Complete |"
    replacement_with_newline = f"{replacement}\n" if replacement else ""
    benchmark.write_text(
        _valid_benchmark().replace(
            f"{completed_row}\n",
            replacement_with_newline,
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {expected_reason}


def test_gate_rejects_duplicate_tier_rows_inside_status_section(tmp_path: Path) -> None:
    completed_row = "| Tier 2 | Semantic deduplication | **PASSED** | Complete |"
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace(
            completed_row,
            completed_row + "\n| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {"duplicate Tier 2 status row"}


def test_gate_ignores_tier_row_decoy_outside_status_section(tmp_path: Path) -> None:
    completed_row = "| Tier 2 | Semantic deduplication | **PASSED** | Complete |"
    content = _valid_benchmark().replace(
        completed_row,
        "| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |",
    )
    content = content.replace("## Results at a Glance\n", completed_row + "\n\n## Results at a Glance\n")
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(content, encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {"publication PASS without completed Tier 2 evidence"}


@pytest.mark.parametrize(
    ("tier", "purpose", "completed_status"),
    [
        (1, "Static validation", "PASSED"),
        (2, "Semantic deduplication", "PASSED"),
    ],
)
def test_gate_does_not_treat_pass_in_required_tier_evidence_as_completion(
    tmp_path: Path,
    tier: int,
    purpose: str,
    completed_status: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace(
            f"| Tier {tier} | {purpose} | **{completed_status}** | Complete |",
            f"| Tier {tier} | {purpose} | **NOT RUN** | Previous result **PASS** |",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {f"publication PASS without completed Tier {tier} evidence"}


@pytest.mark.parametrize(
    "verdict_line",
    [
        "**Overall verdict: PASS**",
        "- **Overall verdict: PASS**",
        "> **Overall verdict**: PASS",
    ],
)
def test_gate_checks_pass_tiers_even_when_verdict_format_is_noncanonical(
    tmp_path: Path,
    verdict_line: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark()
        .replace("> **Overall verdict: PASS**", verdict_line)
        .replace(
            "| Tier 2 | Semantic deduplication | **PASSED** | Complete |",
            "| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "publication PASS without completed Tier 2 evidence" in {offender.reason for offender in offenders}


def test_gate_rejects_hidden_verdict_decoy_after_bold_label_pass(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark()
        .replace(
            "> **Overall verdict: PASS**",
            "> **Overall verdict**: PASS\n<!--\n> **Overall verdict: INCOMPLETE**\n-->",
        )
        .replace(
            "| Tier 2 | Semantic deduplication | **PASSED** | Complete |",
            "| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {"publication PASS without completed Tier 2 evidence"}


def test_gate_rejects_visible_verdict_decoy_after_an_earlier_section(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark()
        .replace(
            "> **Overall verdict: PASS**",
            "> **Overall verdict: INCOMPLETE**\n\n## Prelude\n\n> **Overall verdict: PASS**",
        )
        .replace(
            "| Tier 2 | Semantic deduplication | **PASSED** | Complete |",
            "| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "duplicate Overall verdict field" in {offender.reason for offender in offenders}


@pytest.mark.parametrize(
    "late_verdict",
    [
        "> **Overall verdict: PASS — Publication approved**",
        "> Overall verdict: PASS",
    ],
    ids=["emphasized", "bare-blockquote"],
)
def test_gate_rejects_visible_verdict_callout_after_metadata(
    tmp_path: Path,
    late_verdict: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark()
        .replace("> **Overall verdict: PASS**", "> **Overall verdict: INCOMPLETE**")
        .replace("## Results at a Glance", f"{late_verdict}\n\n## Results at a Glance")
        .replace(
            "| Tier 2 | Semantic deduplication | **PASSED** | Complete |",
            "| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "duplicate Overall verdict field" in {offender.reason for offender in offenders}


def test_gate_normalizes_bold_tier_name_before_checking_pass_evidence(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace(
            "| Tier 2 | Semantic deduplication | **PASSED** | Complete |",
            "| **Tier 2** | Semantic deduplication | **NOT RUN** | No result was recorded |",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {"publication PASS without completed Tier 2 evidence"}


@pytest.mark.parametrize(
    "hidden_policy",
    [
        "<!--\n- Tier 2 evidence: optional by policy\n-->",
        "```markdown\n- Tier 2 evidence: optional by policy\n```",
        "`\n- Tier 2 evidence: optional by policy\n`",
        "``\n- Tier 2 evidence: optional by policy\n``",
        "<pre>\n- Tier 2 evidence: optional by policy\n</pre>",
        "<script>\n- Tier 2 evidence: optional by policy\n</script>",
        "<template>\n- Tier 2 evidence: optional by policy\n</template>",
        "<div>\n- Tier 2 evidence: optional by policy\n</div>",
        "<![CDATA[\n- Tier 2 evidence: optional by policy\n]]>",
        "<?hidden\n- Tier 2 evidence: optional by policy\n?>",
    ],
    ids=[
        "html-comment",
        "fenced-code",
        "single-backtick-span",
        "double-backtick-span",
        "pre-block",
        "script-block",
        "template-block",
        "div-block",
        "cdata-block",
        "processing-instruction",
    ],
)
def test_gate_ignores_hidden_optional_tier2_policy(tmp_path: Path, hidden_policy: str) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark()
        .replace("- Tier 2 evidence: required for publication", hidden_policy)
        .replace(
            "| Tier 2 | Semantic deduplication | **PASSED** | Complete |",
            "| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "publication PASS without completed Tier 2 evidence" in {offender.reason for offender in offenders}


@pytest.mark.parametrize(
    "boundary",
    [
        "##",
        "# Policy Appendix",
        "Policy Appendix\n===============",
        "Policy Appendix\n---------------",
    ],
    ids=["bare-h2", "atx-h1", "setext-h1", "setext-h2"],
)
def test_gate_stops_metadata_at_same_or_higher_heading(tmp_path: Path, boundary: str) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark()
        .replace("- Tier 2 evidence: required for publication\n", "")
        .replace(
            "- Tier 3 evidence: required for publication",
            f"- Tier 3 evidence: required for publication\n\n{boundary}\n\n- Tier 2 evidence: optional by policy",
        )
        .replace(
            "| Tier 2 | Semantic deduplication | **PASSED** | Complete |",
            "| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "publication PASS without completed Tier 2 evidence" in {offender.reason for offender in offenders}


@pytest.mark.parametrize("indent", ["    ", "   \t"], ids=["four-spaces", "mixed-tab"])
def test_gate_does_not_parse_indented_code_as_tier_rows(tmp_path: Path, indent: str) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    content = _valid_benchmark()
    for tier in (1, 2, 3):
        content = content.replace(f"| Tier {tier} |", f"{indent}| Tier {tier} |")
    benchmark.write_text(content, encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    reasons = {offender.reason for offender in offenders}
    assert {f"missing Tier {tier} status row" for tier in (1, 2, 3)} <= reasons


@pytest.mark.parametrize(
    "marker",
    ["## Evaluation Metadata", "## Tier Status"],
    ids=["metadata", "tier-status"],
)
def test_gate_does_not_treat_blockquoted_headings_as_root_sections(
    tmp_path: Path,
    marker: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(_valid_benchmark().replace(marker, f"> {marker}"), encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert f"missing required section: {marker}" in {offender.reason for offender in offenders}


def test_gate_uses_the_named_status_column_when_tier_columns_are_reordered(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark()
        .replace(
            "| Tier | Purpose | Status | Evidence |",
            "| Tier | Purpose | Evidence | Status |",
        )
        .replace(
            "| Tier 2 | Semantic deduplication | **PASSED** | Complete |",
            "| Tier 2 | Semantic deduplication | **PASSED** | **NOT RUN** |",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "publication PASS without completed Tier 2 evidence" in {offender.reason for offender in offenders}


def test_gate_ignores_mixed_whitespace_indented_policy_code(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark()
        .replace(
            "- Tier 2 evidence: required for publication",
            "   \t- Tier 2 evidence: optional by policy",
        )
        .replace(
            "| Tier 2 | Semantic deduplication | **PASSED** | Complete |",
            "| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "publication PASS without completed Tier 2 evidence" in {offender.reason for offender in offenders}


def test_gate_does_not_treat_verdict_prose_as_a_duplicate_field(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace(
            "> **Overall verdict: PASS**",
            "The overall verdict combines all required tiers.\n\n> **Overall verdict: PASS**",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert offenders == []


def test_gate_rejects_duplicate_overall_verdict_fields(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace(
            "> **Overall verdict: PASS**",
            "> **Overall verdict: PASS**\n> **Overall verdict: INCOMPLETE**",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {"duplicate Overall verdict field"}


def test_gate_rejects_conflicting_duplicate_evaluation_metadata_sections(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    duplicate_metadata = """\
## Evaluation Metadata

- Source digest: `sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` (skill-evaluator-source-tree/2)
- Evaluation date: 2026-07-25
- Evaluator version: `0.8.3`
- Agents: Codex (`gpt-codex`)
- Tasks: 4 evaluation tasks
- Dataset digest: `sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef` (skill-evaluator-dataset-snapshot/1)
- Tier 3 run ID: `run-fixture-001`
- Attempts per task: 1
- Environment: `Isolated sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication
"""
    benchmark.write_text(
        _valid_benchmark()
        .replace("- Tier 2 evidence: required for publication", "- Tier 2 evidence: optional by policy", 1)
        .replace(
            "| Tier 2 | Semantic deduplication | **PASSED** | Complete |",
            "| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |",
        )
        .replace("## Results at a Glance", f"{duplicate_metadata}\n## Results at a Glance"),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {"duplicate Evaluation Metadata section"}


@pytest.mark.parametrize("verdict", ["", "UNKNOWN"], ids=["empty", "unknown"])
def test_gate_rejects_invalid_overall_verdict_field(tmp_path: Path, verdict: str) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    replacement = f"> **Overall verdict: {verdict}**" if verdict else "> **Overall verdict:**"
    benchmark.write_text(
        _valid_benchmark().replace(
            "> **Overall verdict: PASS**",
            replacement,
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {"invalid Overall verdict field"}


@pytest.mark.parametrize("location", ["callout", "section", "body"])
def test_gate_rejects_publication_recommendation_for_nonpass_verdict(
    tmp_path: Path,
    location: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    content = _valid_benchmark().replace(
        "> **Overall verdict: PASS**",
        "> **Overall verdict: INCOMPLETE — Required evidence is missing**",
    )
    if location == "callout":
        content = content.replace(
            "Required evidence is missing",
            "Recommended for publication",
        )
    else:
        insertion = (
            "## Publication Recommendation\n\nRecommended for publication."
            if location == "section"
            else "**Recommended for publication.**"
        )
        content = content.replace("## Freshness", f"{insertion}\n\n## Freshness")
    benchmark.write_text(content, encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {"non-PASS verdict recommends publication"}


@pytest.mark.parametrize(
    "recommendation",
    [
        "**Recommended  for publication.**",
        "**Recommended\tfor publication.**",
        "**Recommended&nbsp;for publication.**",
        "<span>Recommended  for publication.</span>",
    ],
)
def test_gate_normalizes_visible_recommendation_whitespace(
    tmp_path: Path,
    recommendation: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark()
        .replace("> **Overall verdict: PASS**", "> **Overall verdict: INCOMPLETE**")
        .replace("## Freshness", f"{recommendation}\n\n## Freshness"),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {"non-PASS verdict recommends publication"}


@pytest.mark.parametrize(
    "duplicate_verdict",
    [
        "**Overall  verdict: PASS**",
        "**Overall&nbsp;verdict: PASS**",
        "<span>Overall  verdict: PASS</span>",
        "<strong>Overall&nbsp;verdict: PASS</strong>",
    ],
)
def test_gate_normalizes_visible_verdict_whitespace(
    tmp_path: Path,
    duplicate_verdict: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark()
        .replace("> **Overall verdict: PASS**", "> **Overall verdict: INCOMPLETE**")
        .replace("## Freshness", f"{duplicate_verdict}\n\n## Freshness"),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "duplicate Overall verdict field" in {offender.reason for offender in offenders}


def test_gate_ignores_hidden_comment_recommendation_for_nonpass_verdict(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark()
        .replace("> **Overall verdict: PASS**", "> **Overall verdict: INCOMPLETE**")
        .replace("## Freshness", "<!-- Recommended for publication. -->\n\n## Freshness"),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert offenders == []


@pytest.mark.parametrize(
    "hidden_html",
    [
        "<script>Recommended for publication.</script>",
        "<style>Recommended for publication.</style>",
        "<template>Recommended for publication.</template>",
        "<!-- Recommended for publication. -->",
    ],
)
def test_gate_ignores_hidden_html_recommendation_for_nonpass_verdict(
    tmp_path: Path,
    hidden_html: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark()
        .replace("> **Overall verdict: PASS**", "> **Overall verdict: INCOMPLETE**")
        .replace("## Freshness", f"{hidden_html}\n\n## Freshness"),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert offenders == []


@pytest.mark.parametrize(
    "visible_html",
    [
        "<script></script>Recommended for publication.",
        "<style></style> Recommended for publication.",
        "<!-- hidden -->Recommended for publication.",
    ],
)
def test_gate_rejects_visible_recommendation_after_hidden_html_control(
    tmp_path: Path,
    visible_html: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark()
        .replace("> **Overall verdict: PASS**", "> **Overall verdict: INCOMPLETE**")
        .replace("## Freshness", f"{visible_html}\n\n## Freshness"),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {"non-PASS verdict recommends publication"}


@pytest.mark.parametrize(
    "visible_html",
    [
        "<div>Overall verdict: PASS</div>",
        "<strong>Overall verdict: PASS</strong>",
        "<!-- hidden -->Overall verdict: PASS",
    ],
)
def test_gate_rejects_visible_raw_html_verdict_callout(
    tmp_path: Path,
    visible_html: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark()
        .replace("> **Overall verdict: PASS**", "> **Overall verdict: INCOMPLETE**")
        .replace("## Freshness", f"{visible_html}\n\n## Freshness"),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "duplicate Overall verdict field" in {offender.reason for offender in offenders}


@pytest.mark.parametrize(
    "hidden_verdict",
    [
        "<div hidden>Overall verdict: PASS — Recommended for publication</div>",
        '<div aria-hidden="true">Overall verdict: PASS — Recommended for publication</div>',
        '<div style="display:none">Overall verdict: PASS — Recommended for publication</div>',
        '<div style="visibility:hidden">Overall verdict: PASS — Recommended for publication</div>',
    ],
)
def test_gate_rejects_raw_html_as_the_only_verdict_field(
    tmp_path: Path,
    hidden_verdict: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace("> **Overall verdict: PASS**", hidden_verdict),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "invalid Overall verdict field" in {offender.reason for offender in offenders}


def test_gate_rejects_linked_overall_verdict_field(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace(
            "> **Overall verdict: PASS**",
            "> **[Overall verdict: PASS](https://example.invalid/phish)**",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "invalid Overall verdict field" in {offender.reason for offender in offenders}


@pytest.mark.parametrize(
    ("tier", "status"),
    [
        (1, "PASSED"),
        (2, "PASSED"),
        (3, "PASS"),
    ],
)
@pytest.mark.parametrize(
    "wrapped_status",
    [
        "<span hidden>{status}</span>",
        '<span aria-hidden="true">{status}</span>',
        '<span style="display:none">{status}</span>',
        "<s>{status}</s>",
    ],
)
def test_gate_rejects_html_wrapped_tier_completion_status(
    tmp_path: Path,
    tier: int,
    status: str,
    wrapped_status: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    original_row = next(line for line in _valid_benchmark().splitlines() if line.startswith(f"| Tier {tier} |"))
    wrapped_row = original_row.replace(f"**{status}**", wrapped_status.format(status=status))
    benchmark.write_text(
        _valid_benchmark().replace(original_row, wrapped_row),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert f"invalid Tier {tier} status" in {offender.reason for offender in offenders}


def test_gate_rejects_linked_tier_completion_status(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace(
            "**PASSED** | Complete |",
            "**[PASSED](https://example.invalid/phish)** | Complete |",
            1,
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "invalid Tier 1 status" in {offender.reason for offender in offenders}


@pytest.mark.parametrize("section", ["Tier Status", "Evaluation Metadata"])
def test_gate_rejects_critical_markdown_inside_raw_html_container(
    tmp_path: Path,
    section: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    text = _valid_benchmark()
    start = text.index(f"## {section}") + len(f"## {section}")
    next_heading = text.index("\n## ", start)
    text = f"{text[:start]}\n\n<details hidden>\n{text[start:next_heading]}\n</details>\n{text[next_heading:]}"
    benchmark.write_text(text, encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    reasons = {offender.reason for offender in offenders}
    if section == "Tier Status":
        assert "missing Tier 1 status row" in reasons
        assert "missing Tier 2 status row" in reasons
        assert "missing Tier 3 status row" in reasons
    else:
        assert "missing metadata field: - Evaluation date:" in reasons


@pytest.mark.parametrize(
    "boundary_heading",
    [
        "## [Unrelated](https://example.invalid/phish)",
        "## <span>Unrelated</span>",
        "<h2>Unrelated</h2>",
    ],
    ids=["linked-heading", "inline-html-heading", "raw-html-heading"],
)
@pytest.mark.parametrize("section", ["Tier Status", "Evaluation Metadata"])
def test_gate_does_not_attribute_evidence_across_untrusted_section_boundary(
    tmp_path: Path,
    section: str,
    boundary_heading: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    text = _valid_benchmark()
    start = text.index(f"## {section}") + len(f"## {section}")
    next_heading = text.index("\n## ", start)
    original_body = text[start:next_heading]
    text = f"{text[:start]}\n\n{boundary_heading}{original_body}{text[next_heading:]}"
    benchmark.write_text(text, encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    reasons = {offender.reason for offender in offenders}
    if section == "Tier Status":
        assert "missing Tier 1 status row" in reasons
        assert "missing Tier 2 status row" in reasons
        assert "missing Tier 3 status row" in reasons
    else:
        assert "missing metadata field: - Evaluation date:" in reasons


def test_gate_rejects_entire_card_inside_closed_raw_html_container(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(f"<details>\n\n{_valid_benchmark()}\n\n</details>\n", encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    reasons = {offender.reason for offender in offenders}
    assert "missing required section: # Skill Benchmark:" in reasons
    assert "invalid Overall verdict field" in reasons


@pytest.mark.parametrize(
    "opening_html",
    [
        "<details hidden><noscript></details></noscript>",
        "<details hidden><select></details></select>",
        "<details hidden><table></details></table>",
    ],
    ids=["noscript-raw-text", "select-insertion-mode", "table-insertion-mode"],
)
def test_gate_rejects_card_hidden_by_browser_html_parsing_semantics(
    tmp_path: Path,
    opening_html: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        f"{opening_html}\n\n{_valid_benchmark()}\n</details>\n",
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    reasons = {offender.reason for offender in offenders}
    assert "missing required section: # Skill Benchmark:" in reasons
    assert "invalid Overall verdict field" in reasons


def test_gate_treats_nonvoid_slash_tag_as_an_open_html_container(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace(
            "## Evaluation Metadata",
            "<details/>\n\n## Evaluation Metadata",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    reasons = {offender.reason for offender in offenders}
    assert "missing required section: ## Evaluation Metadata" in reasons
    assert "missing Tier 1 status row" in reasons
    assert "missing Tier 2 status row" in reasons
    assert "missing Tier 3 status row" in reasons


@pytest.mark.parametrize("control", ["\u200b", "\u2060", "\u00ad"])
def test_gate_detects_invisibly_split_publication_recommendation(
    tmp_path: Path,
    control: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    text = _valid_benchmark().replace("Overall verdict: PASS", "Overall verdict: FAIL")
    text = text.replace(
        "## Freshness",
        f"Recommended{control} for publication.\n\n## Freshness",
    )
    benchmark.write_text(text, encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "non-PASS verdict recommends publication" in {offender.reason for offender in offenders}


@pytest.mark.parametrize(
    "obfuscated_identity",
    [
        "Skills\u200bEval",
        "Skills\x00Eval",
        "\uff33\uff4b\uff49\uff4c\uff4c\uff53\uff25\uff56\uff41\uff4c",
        "skills\u034feval",
        "Skills\u0301Eval",
        "SkillsE\u0301val",
    ],
    ids=[
        "format-control",
        "control",
        "fullwidth",
        "noncomposing-mark",
        "composing-mark-before-eval",
        "composing-mark-inside-eval",
    ],
)
def test_gate_normalizes_line_rules_before_scanning_retired_identity(
    tmp_path: Path,
    obfuscated_identity: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace("- Evaluator version: `0.8.3`", f"- Evaluator version: `{obfuscated_identity}`"),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "retired product identity" in {offender.reason for offender in offenders}


def test_gate_decodes_html_character_references_in_rendered_text_rules(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(_valid_benchmark() + "\n- Finding: Skills&#69;val\n", encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "retired product identity" in {offender.reason for offender in offenders}


@pytest.mark.parametrize("separator", ["\u2010", "\u2011", "\u2012", "\u2013", "\u2043", "\u2212"])
@pytest.mark.parametrize("identity", ["Skills{separator}Eval", "astra{separator}sandbox"])
def test_gate_normalizes_unicode_dash_confusables_for_private_identity_rules(
    tmp_path: Path,
    separator: str,
    identity: str,
) -> None:
    rendered_identity = identity.format(separator=separator)
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(_valid_benchmark() + f"\n- Finding: {rendered_identity}\n", encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    expected = "retired product identity" if identity.startswith("Skills") else "internal environment identity"
    assert expected in {offender.reason for offender in offenders}


def test_gate_preserves_html_character_references_inside_code_spans(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace(
            "- Evaluator version: `0.8.3`",
            "- Evaluator version: `Skills&#69;val`",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "retired product identity" not in {offender.reason for offender in offenders}


@pytest.mark.parametrize(
    "literal_block",
    [
        "```text\nSkills&#69;val\n```",
        "    Skills&#69;val",
    ],
    ids=["fenced-code", "indented-code"],
)
def test_gate_preserves_character_references_inside_literal_code_blocks(
    tmp_path: Path,
    literal_block: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(_valid_benchmark() + f"\n{literal_block}\n", encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "retired product identity" not in {offender.reason for offender in offenders}


def test_gate_decodes_html_character_references_before_path_rules(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark() + "\n- Finding: /Us&#101;rs/private/repo/SKILL.md\n",
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "absolute macOS user path" in {offender.reason for offender in offenders}


def test_gate_does_not_decode_path_entities_inside_code_spans(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark() + "\n- Finding: `/Us&#101;rs/private/repo/SKILL.md`\n",
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "absolute macOS user path" not in {offender.reason for offender in offenders}


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("\u2215Users\u2215alice\u2215private\u2215secret.txt", "absolute macOS user path"),
        ("\u2044home\u2044alice\u2044private\u2044secret.txt", "absolute Linux home path"),
        ("C:\u29f5Users\u29f5alice\u29f5private\u29f5secret.txt", "absolute Windows user path"),
    ],
)
def test_gate_normalizes_path_separator_confusables(
    tmp_path: Path,
    path: str,
    reason: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(_valid_benchmark() + f"\n- Finding: {path}\n", encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert reason in {offender.reason for offender in offenders}


@pytest.mark.parametrize(
    "markup",
    [
        "[safe](/Us&#101;rs/private/repo)",
        '<a href="/Us&#101;rs/private/repo">safe</a>',
        '<img src="/Us&#101;rs/private/repo.png" alt="safe">',
        "![&#47;Users&#47;alice&#47;secret](safe.png)",
        "![**&#47;Users&#47;alice&#47;secret**](safe.png)",
    ],
    ids=["markdown-destination", "html-href", "html-image-src", "image-alt", "formatted-image-alt"],
)
def test_gate_scans_decoded_link_image_and_html_surfaces(tmp_path: Path, markup: str) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(_valid_benchmark() + f"\n- Finding: {markup}\n", encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "absolute macOS user path" in {offender.reason for offender in offenders}


@pytest.mark.parametrize(
    "markup",
    [
        "SkillsEval [docs](https://example.invalid)",
        "SkillsEval <span>ok</span>",
        "SkillsEval ![alt](image.png)",
        "/Users/alice/private [docs](https://example.invalid)",
    ],
)
def test_gate_markup_cannot_suppress_other_line_rules(tmp_path: Path, markup: str) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(_valid_benchmark() + f"\n- Finding: {markup}\n", encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    reasons = {offender.reason for offender in offenders}
    if "SkillsEval" in markup:
        assert "retired product identity" in reasons
    else:
        assert "absolute macOS user path" in reasons


@pytest.mark.parametrize(
    "identity",
    [
        "\u200b",
        "\x00",
        "\ufe0f",
        "\u115f",
        "\u2800",
        "\U00013441",
        "\U00013442",
        "\U0001d159",
        "not recorded",
        "model not recorded",
        "not recor\u0301ded",
        "not recor\ufe0fded",
        "not recor\u034fded",
        "not recor\u20ddded",
        "-",
        "N/A",
        "unknown",
        "TBD",
        "unkn\u043ewn",
        "unkn\u03bfwn",
        "m\u043edel not recorded",
        "modeI not recorded",
        "mode\u0406 not recorded",
        "unkn0wn",
        "\u039codel not recorded",
        "model not re\u03f2orded",
        "unkn\u0c02wn",
        "unkn\U0001cce4wn",
        "mode\u0140not recorded",
        "model\u0149ot recorded",
        "unknow\u145amodel",
        "u\u0295nknown",
        "u\U0001f40dnknown",
        "u\u6138nknown",
    ],
    ids=[
        "format-control",
        "control",
        "variation-selector",
        "hangul-filler",
        "braille-blank",
        "hieroglyph-full-blank",
        "hieroglyph-half-blank",
        "musical-null-notehead",
        "reserved-placeholder",
        "reserved-model-placeholder",
        "reserved-combining-placeholder",
        "reserved-variation-placeholder",
        "reserved-joiner-placeholder",
        "reserved-enclosing-placeholder",
        "punctuation-only",
        "reserved-na-placeholder",
        "reserved-unknown-placeholder",
        "reserved-tbd-placeholder",
        "reserved-cyrillic-confusable-placeholder",
        "reserved-greek-confusable-placeholder",
        "reserved-confusable-model-placeholder",
        "reserved-ascii-confusable-model-placeholder",
        "reserved-cyrillic-i-confusable-model-placeholder",
        "reserved-digit-confusable-placeholder",
        "reserved-greek-mu-confusable-model-placeholder",
        "reserved-greek-lunate-sigma-placeholder",
        "reserved-telugu-mark-confusable-placeholder",
        "reserved-unicode17-confusable-placeholder",
        "reserved-trailing-separator-confusable-placeholder",
        "reserved-leading-separator-confusable-placeholder",
        "reserved-folded-separator-confusable-placeholder",
        "reserved-vanishing-letter-confusable-placeholder",
        "reserved-vanishing-symbol-confusable-placeholder",
        "reserved-vanishing-cjk-confusable-placeholder",
    ],
)
@pytest.mark.parametrize(
    ("field", "expected_reason"),
    [
        ("heading", "missing required section: # Skill Benchmark:"),
        ("evaluator_version", "invalid metadata field: - Evaluator version:"),
        ("agent", "agent model identity not recorded"),
        ("model", "agent model identity not recorded"),
        ("environment", "invalid metadata field: - Environment:"),
    ],
)
def test_gate_rejects_semantically_empty_public_identities(
    tmp_path: Path,
    identity: str,
    field: str,
    expected_reason: str,
) -> None:
    text = _valid_benchmark()
    if field == "heading":
        text = text.replace("# Skill Benchmark: demo", f"# Skill Benchmark: {identity}")
    elif field == "evaluator_version":
        text = text.replace("- Evaluator version: `0.8.3`", f"- Evaluator version: `{identity}`")
    elif field == "agent":
        text = text.replace("- Agents: Codex (`gpt-codex`)", f"- Agents: {identity} (`gpt-codex`)")
    elif field == "model":
        text = text.replace("- Agents: Codex (`gpt-codex`)", f"- Agents: Codex (`{identity}`)")
    else:
        text = text.replace("- Environment: `Isolated sandbox`", f"- Environment: `{identity}`")
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(text, encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    reason = "benchmark contains disallowed control character" if identity == "\x00" else expected_reason
    assert reason in {offender.reason for offender in offenders}


def test_gate_preserves_visible_decomposed_unicode_identities(tmp_path: Path) -> None:
    identity = "Cafe\u0301"
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark()
        .replace("# Skill Benchmark: demo", f"# Skill Benchmark: {identity}")
        .replace("- Evaluator version: `0.8.3`", f"- Evaluator version: `{identity}`")
        .replace("- Agents: Codex (`gpt-codex`)", f"- Agents: {identity} (`{identity}`)")
        .replace("- Environment: `Isolated sandbox`", f"- Environment: `{identity}`"),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert offenders == []


def test_gate_rejects_second_plain_overall_verdict_after_metadata(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace(
            "## Freshness",
            "Overall verdict: FAIL — Publication blocked\n\n## Freshness",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "duplicate Overall verdict field" in {offender.reason for offender in offenders}


@pytest.mark.parametrize(
    ("heading", "reason"),
    [
        ("# Skill Benchmark: demo-skill", "duplicate required section: # Skill Benchmark:"),
        ("## Results at a Glance", "duplicate required section: ## Results at a Glance"),
        ("## Freshness", "duplicate required section: ## Freshness"),
        ("## Publication Recommendation", "duplicate Publication Recommendation section"),
    ],
)
def test_gate_rejects_duplicate_decision_headings(
    tmp_path: Path,
    heading: str,
    reason: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    text = _valid_benchmark()
    if heading == "## Publication Recommendation":
        text = text.replace("## Freshness", f"{heading}\n\nRecommended for publication.\n\n{heading}\n\n## Freshness")
    else:
        text = f"{text}\n\n{heading}\n"
    benchmark.write_text(text, encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert reason in {offender.reason for offender in offenders}


def test_gate_rejects_unknown_tier_status_for_nonpass_card(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark()
        .replace("> **Overall verdict: PASS**", "> **Overall verdict: FAIL**")
        .replace(
            "| Tier 2 | Semantic deduplication | **PASSED** | Complete |",
            "| Tier 2 | Semantic deduplication | **BANANA** | Malformed status |",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {"invalid Tier 2 status"}


@pytest.mark.parametrize("status", ["NOT RUN", "SKIPPED (ADVISORY)"])
def test_gate_accepts_clean_optional_tier2_without_completed_evidence(
    tmp_path: Path,
    status: str,
) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark()
        .replace("- Tier 2 evidence: required for publication", "- Tier 2 evidence: optional by policy")
        .replace(
            "| Tier 2 | Semantic deduplication | **PASSED** | Complete |",
            f"| Tier 2 | Semantic deduplication | **{status}** | No result was recorded |",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert offenders == []


def test_gate_rejects_optional_but_incomplete_tier2_evidence(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark()
        .replace("- Tier 2 evidence: required for publication", "- Tier 2 evidence: optional by policy")
        .replace(
            "| Tier 2 | Semantic deduplication | **PASSED** | Complete |",
            "| Tier 2 | Semantic deduplication | **INCOMPLETE** | Present result lacks complete evidence |",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {"publication PASS without completed Tier 2 evidence"}


def test_gate_missing_tier2_policy_metadata_defaults_to_required(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark()
        .replace("- Tier 2 evidence: required for publication\n", "")
        .replace(
            "| Tier 2 | Semantic deduplication | **PASSED** | Complete |",
            "| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {
        "publication PASS without completed Tier 2 evidence",
    }


def test_gate_accepts_completed_tier2_without_policy_metadata(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace("- Tier 2 evidence: required for publication\n", ""),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert offenders == []


def test_gate_rejects_pass_with_missing_required_tier3(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace(
            "| Tier 3 | Live agent evaluation | **PASS** | Complete |",
            "| Tier 3 | Live agent evaluation | **NOT RUN** | No result was recorded |",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {"publication PASS without completed Tier 3 evidence"}


def test_gate_does_not_treat_pass_in_tier3_evidence_as_completion(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace(
            "| Tier 3 | Live agent evaluation | **PASS** | Complete |",
            "| Tier 3 | Live agent evaluation | **NOT RUN** | Previous result **PASS** |",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {"publication PASS without completed Tier 3 evidence"}


def test_gate_rejects_pass_with_placeholder_provenance(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    content = _valid_benchmark()
    replacements = {
        "- Source digest: `sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` "
        "(skill-evaluator-source-tree/2)": "- Source digest: not recorded (legacy or unbound result)",
        "- Evaluation date: 2026-07-25": "- Evaluation date: not recorded (legacy or non-live result)",
        "- Evaluator version: `0.8.3`": "- Evaluator version: not recorded (legacy or non-live result)",
        "- Agents: Codex (`gpt-codex`)": "- Agents: Codex (model not recorded)",
        "- Tasks: 4 evaluation tasks": "- Tasks: not recorded (legacy or non-live result)",
        "- Dataset digest: `sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef` "
        "(skill-evaluator-dataset-snapshot/1)": ("- Dataset digest: not recorded (legacy or non-live result)"),
        "- Tier 3 run ID: `run-fixture-001`": "- Tier 3 run ID: not recorded (Tier 3 did not complete)",
        "- Attempts per task: 1": "- Attempts per task: not recorded (legacy or non-live result)",
        "- Environment: `Isolated sandbox`": "- Environment: not recorded (legacy or non-live result)",
    }
    for original, replacement in replacements.items():
        content = content.replace(original, replacement)
    benchmark.write_text(content, encoding="utf-8")

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {
        "publication PASS without recorded source digest",
        "publication PASS without recorded evaluation date",
        "publication PASS without recorded evaluator version",
        "publication PASS without recorded agent model identity",
        "publication PASS without recorded tasks",
        "publication PASS without recorded dataset digest",
        "publication PASS without recorded tier 3 run id",
        "publication PASS without recorded attempts per task",
        "publication PASS without recorded environment",
    }


def test_gate_requires_provenance_when_optional_tier3_row_claims_pass(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark()
        .replace("- Tier 3 evidence: required for publication", "- Tier 3 evidence: optional by policy")
        .replace("- Environment: `Isolated sandbox`", "- Environment: not recorded (legacy or non-live result)"),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert {offender.reason for offender in offenders} == {"publication PASS without recorded environment"}


def test_gate_rejects_pass_with_malformed_dataset_digest(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(
        _valid_benchmark().replace(
            "`sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef` "
            "(skill-evaluator-dataset-snapshot/1)",
            "`md5:0123456789abcdef` (legacy)",
        ),
        encoding="utf-8",
    )

    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "publication PASS without recorded dataset digest" in {offender.reason for offender in offenders}


def test_require_files_fails_empty_tree(tmp_path: Path, capsys) -> None:
    assert benchmark_gate.main(["--require-files", str(tmp_path)]) == 1
    assert "no BENCHMARK.md files found" in capsys.readouterr().out


def test_gate_reports_invalid_utf8_as_unreadable(tmp_path: Path) -> None:
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_bytes(b"\xff\xfe\x00")

    files, offenders = benchmark_gate.find_offenders([benchmark])

    assert files == [benchmark.resolve()]
    assert offenders == [benchmark_gate.Offender(benchmark.resolve(), 1, "unreadable file (UnicodeDecodeError)")]


def test_gate_reports_nonexistent_input_path(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing"

    files, offenders = benchmark_gate.find_offenders([missing])

    assert files == []
    assert offenders == [benchmark_gate.Offender(missing.resolve(), 1, "input path does not exist")]
    assert benchmark_gate.main(["--require-files", str(missing)]) == 1
    output = capsys.readouterr().out
    assert str(missing.resolve()) in output
    assert "input path does not exist" in output
