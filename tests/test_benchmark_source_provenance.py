# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluated-source provenance in the published benchmark card.

A published card has to say which source tree was evaluated, separately from the
evaluator build that evaluated it. Without that separation two skills from
different repositories can publish cards whose only recorded revision is the
shared evaluator container tag.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.ci import check_public_benchmarks as benchmark_gate

from skillevaluator import source_identity
from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
from skillevaluator.models import ValidationResult
from skillevaluator.reporting import BenchmarkReporter
from skillevaluator.source_identity import (
    EvaluatedSourceConflict,
    normalized_evaluated_source,
    resolve_evaluated_source,
)

_COMMIT_A = "2263a2ebdab903e87f7e7c0a001d22c3a926a9cf"
_COMMIT_B = "134c829305918a3e9a84e819b42b54fafd125186"
_CONTAINER = "ghcr.io/nvidia/skillevaluator@sha256:" + "0117bc2e" * 8
_DIGEST = "sha256:" + "ab" * 32
_UNRECORDED = "not recorded (not supplied by the orchestration input)"

_SCORES = {
    "security": 0.9,
    "skill_execution": 0.9,
    "skill_efficiency": 0.9,
    "accuracy": 0.9,
    "goal_accuracy": 0.9,
    "behavior_check": 0.9,
}
_AGENTS = {
    "codex": {
        "model": "gpt-codex",
        "with_skill": _SCORES,
        "without_skill": dict.fromkeys(_SCORES, 0.5),
        "execution_status": "succeeded",
        "rewards": [],
        "num_trials": 1,
    }
}


def _card(evaluated_source: dict[str, str] | None) -> str:
    result = ValidationResult(
        validator_name="AGENT_EVAL",
        validator_description="Run live agent evaluation",
    )
    result.metadata["agent_eval"] = {
        "skill_name": "demo-skill",
        "evaluator_version": "0.8.2",
        "evaluated_source": evaluated_source,
        "summary": {"environment": "Isolated sandbox"},
        "agents": {"codex": {"model": "gpt-codex"}},
    }
    return BenchmarkReporter().render_all([result])


def _value(card: str, field: str) -> str | None:
    return benchmark_gate._metadata_field_value(card, field)


def _payload(**kwargs: object) -> dict[str, object]:
    payload = build_agent_eval_payload("demo-skill", _AGENTS, use_llm_judge=False, **kwargs)
    assert payload is not None
    return payload


class TestNormalization:
    """The identity comes from orchestration input, so it is validated, not trusted."""

    def test_canonical_identity_is_preserved(self) -> None:
        assert normalized_evaluated_source({"repository": "NVIDIA/NVFlare", "commit": _COMMIT_A}) == {
            "repository": "NVIDIA/NVFlare",
            "commit": _COMMIT_A,
        }

    @pytest.mark.parametrize("field", ["commit", "content_digest"])
    def test_hex_revisions_are_case_folded(self, field: str) -> None:
        value = _COMMIT_A if field == "commit" else _DIGEST
        assert normalized_evaluated_source({field: value.upper()}) == {field: value}

    @pytest.mark.parametrize(
        "value",
        [
            None,
            {},
            "NVIDIA/NVFlare",
            {"repository": "not a repository"},
            {"repository": "NVIDIA/NVFlare`` injected"},
            {"repository": "-leading/dash"},
            {"repository": "a" * 40 + "/repo"},
            {"commit": "zzzz"},
            {"commit": "abc"},
            {"content_digest": "sha256:nothex"},
            {"content_digest": "totally-fake:" + "0" * 64},
            {"evaluator_container_revision": "has space"},
            {"repository": 42, "commit": ["list"]},
        ],
    )
    def test_malformed_identity_is_dropped(self, value: object) -> None:
        assert normalized_evaluated_source(value) is None

    def test_valid_fields_survive_an_invalid_sibling(self) -> None:
        assert normalized_evaluated_source({"repository": "NVIDIA/NVFlare", "commit": "nope"}) == {
            "repository": "NVIDIA/NVFlare"
        }

    @pytest.mark.parametrize(
        "value",
        [
            {"commit": _COMMIT_A[:7]},
            {"commit": _COMMIT_A[:12]},
            {"commit": "a" * 39},
            {"commit": "a" * 41},
            {"content_digest": "sha512:" + "ab" * 16},
            {"content_digest": "sha256:" + "ab" * 48},
            {"content_digest": "sha384:" + "ab" * 32},
        ],
    )
    def test_ambiguous_or_mismatched_revisions_are_dropped(self, value: object) -> None:
        """A short prefix can grow a second match, and a digest must match its own algorithm."""
        assert normalized_evaluated_source(value) is None

    @pytest.mark.parametrize("commit", ["a" * 40, "b" * 64])
    def test_full_git_object_ids_are_accepted(self, commit: str) -> None:
        """SHA-1 and SHA-256 object ids are both canonical Git revisions."""
        assert normalized_evaluated_source({"commit": commit}) == {"commit": commit}

    @pytest.mark.parametrize(("algorithm", "width"), [("sha256", 64), ("sha384", 96), ("sha512", 128)])
    def test_each_digest_algorithm_requires_its_own_width(self, algorithm: str, width: int) -> None:
        digest = f"{algorithm}:" + "a" * width
        assert normalized_evaluated_source({"content_digest": digest}) == {"content_digest": digest}

    @pytest.mark.parametrize(
        "repository",
        ["holgerroth/nvflare_examples", "NVIDIA/Megatron_LM", "some-org/repo.name"],
    )
    def test_real_world_repository_names_are_accepted(self, repository: str) -> None:
        """Underscores and dots are ordinary in forge names and must survive."""
        assert normalized_evaluated_source({"repository": repository}) == {"repository": repository}

    def test_digest_pinned_container_reference_is_accepted(self) -> None:
        """``@sha256:`` is the only immutable way to pin the evaluator build."""
        assert normalized_evaluated_source({"evaluator_container_revision": _CONTAINER}) == {
            "evaluator_container_revision": _CONTAINER
        }

    def test_a_long_repository_pinned_by_digest_is_accepted(self) -> None:
        """A cap over the whole reference charged the digest to the name and dropped ordinary references."""
        reference = "ghcr.io/nvidia/" + "e" * 53 + "@sha256:" + "0117bc2e" * 8
        assert len(reference.partition("@")[0]) == 68
        assert len(reference) == 140
        assert normalized_evaluated_source({"evaluator_container_revision": reference}) == {
            "evaluator_container_revision": reference
        }

    @pytest.mark.parametrize(("length", "accepted"), [(255, True), (256, False)])
    def test_the_path_is_bounded_at_the_oci_maximum(self, length: int, accepted: bool) -> None:
        """255 is RepositoryNameTotalLengthMax, and it bounds the path rather than the whole reference."""
        reference = "n" * length + "@sha256:" + "0117bc2e" * 8
        assert bool(normalized_evaluated_source({"evaluator_container_revision": reference})) is accepted

    @pytest.mark.parametrize(("length", "accepted"), [(255, True), (256, False)])
    def test_a_registry_host_does_not_spend_the_path_budget(self, length: int, accepted: bool) -> None:
        """The reference grammar bounds the path once the host is split off, not the two together."""
        reference = "ghcr.io/" + "n" * length + "@sha256:" + "0117bc2e" * 8
        assert bool(normalized_evaluated_source({"evaluator_container_revision": reference})) is accepted

    @pytest.mark.parametrize(
        "revision",
        [
            "NVIDIA/skillevaluator@sha256:" + "0117bc2e" * 8,
            "ghcr.io/NVIDIA/skillevaluator@sha256:" + "0117bc2e" * 8,
        ],
    )
    def test_an_uppercase_path_is_refused_with_and_without_a_registry(self, revision: str) -> None:
        """Reading any first label as a host took the bare form and refused the same name under a registry."""
        assert normalized_evaluated_source({"evaluator_container_revision": revision}) is None

    @pytest.mark.parametrize(
        "revision",
        [
            "localhost/team/image@sha256:" + "0117bc2e" * 8,
            "Registry.Example.COM/team/image@sha256:" + "0117bc2e" * 8,
            "myregistry:5000/team/image@sha256:" + "0117bc2e" * 8,
        ],
    )
    def test_a_registry_host_is_localhost_dotted_or_ported(self, revision: str) -> None:
        """Those are the three shapes the grammar reads as a host, and a host name may use any case."""
        assert normalized_evaluated_source({"evaluator_container_revision": revision}) == {
            "evaluator_container_revision": revision
        }

    @pytest.mark.parametrize(
        "revision",
        [
            "localhost:5000/team/image:1.2.3@sha256:" + "0117bc2e" * 8,
            "nvcr.io/nvidia/skillevaluator:1.3.2",
            _COMMIT_A,
        ],
    )
    def test_references_that_name_a_revision_are_accepted(self, revision: str) -> None:
        """A registry port, a tag and an implementation revision all name the build that ran."""
        assert normalized_evaluated_source({"evaluator_container_revision": revision}) == {
            "evaluator_container_revision": revision
        }

    @pytest.mark.parametrize(
        "revision",
        [
            "ghcr.io/nvidia/skillevaluator",
            "ghcr.io/nvidia/skillevaluator@sha512:" + "ab" * 16,
            "ghcr.io/nvidia/skillevaluator@md5:" + "ab" * 16,
            "ghcr.io/nvidia/skillevaluator:" + "t" * 129,
            "has space",
            "ghcr.io/nvidia/skillevaluator`` injected",
        ],
    )
    def test_references_that_name_no_build_are_dropped(self, revision: str) -> None:
        """A bare name is a repository, not a revision, and the rest are unpinnable or not inert in a code span."""
        assert normalized_evaluated_source({"evaluator_container_revision": revision}) is None


class TestPayloadContract:
    """The identity travels through the result contract rather than being re-derived."""

    def test_run_config_supplies_the_identity(self) -> None:
        payload = _payload(run_config={"evaluated_source": {"repository": "NVIDIA/NVFlare", "commit": _COMMIT_A}})
        assert payload["evaluated_source"] == {"repository": "NVIDIA/NVFlare", "commit": _COMMIT_A}
        assert payload["summary"]["evaluated_source"] == payload["evaluated_source"]

    def test_the_two_channels_combine(self) -> None:
        payload = _payload(
            run_config={"evaluated_source": {"commit": _COMMIT_A}},
            evaluated_source={"repository": "NVIDIA/NVFlare"},
        )
        assert payload["evaluated_source"] == {"repository": "NVIDIA/NVFlare", "commit": _COMMIT_A}

    def test_absent_identity_is_recorded_as_none(self) -> None:
        assert _payload()["evaluated_source"] is None

    def test_malformed_identity_does_not_reach_the_payload(self) -> None:
        assert _payload(evaluated_source={"repository": "not a repository"})["evaluated_source"] is None


class TestRendering:
    """Source provenance and evaluator provenance get separate, unambiguous labels."""

    def test_card_records_the_evaluated_source(self) -> None:
        card = _card({"repository": "NVIDIA/NVFlare", "commit": _COMMIT_A, "evaluator_container_revision": _CONTAINER})
        assert _value(card, "Evaluated source") == "`NVIDIA/NVFlare`"
        assert _value(card, "Evaluated source revision") == f"`{_COMMIT_A}`"
        assert _value(card, "Evaluator container revision") == f"`{_CONTAINER}`"

    def test_evaluator_version_is_not_the_source_revision(self) -> None:
        card = _card({"repository": "NVIDIA/NVFlare", "commit": _COMMIT_A})
        assert _value(card, "Evaluator version") == "`0.8.2`"
        assert _value(card, "Evaluated source revision") == f"`{_COMMIT_A}`"

    def test_content_digest_stands_in_for_a_missing_commit(self) -> None:
        card = _card({"repository": "NVIDIA/NVFlare", "content_digest": _DIGEST})
        assert _value(card, "Evaluated source revision") == f"`{_DIGEST}`"

    def test_commit_wins_over_a_content_digest(self) -> None:
        card = _card({"repository": "NVIDIA/NVFlare", "commit": _COMMIT_A, "content_digest": _DIGEST})
        assert _value(card, "Evaluated source revision") == f"`{_COMMIT_A}`"

    def test_missing_identity_names_the_orchestration_input(self) -> None:
        card = _card(None)
        for field in ("Evaluated source", "Evaluated source revision", "Evaluator container revision"):
            assert _value(card, field) == _UNRECORDED

    @pytest.mark.parametrize("repository", ["holgerroth/nvflare_examples", "NVIDIA/Megatron_LM"])
    def test_underscored_repository_is_published_verbatim(self, repository: str) -> None:
        """Markdown escaping would rewrite ``_`` and publish a repository that does not exist."""
        card = _card({"repository": repository, "commit": _COMMIT_A})
        assert _value(card, "Evaluated source") == f"`{repository}`"
        assert "\\_" not in card

    def test_digest_pinned_container_is_published_verbatim(self) -> None:
        """Escaping ``@`` to ``&#64;`` would publish an unresolvable image reference."""
        card = _card({"repository": "NVIDIA/NVFlare", "evaluator_container_revision": _CONTAINER})
        assert _value(card, "Evaluator container revision") == f"`{_CONTAINER}`"
        assert "&#64;" not in card

    @pytest.mark.parametrize(
        "hostile",
        [
            {"repository": "A/B` INJECTED **bold** `x"},
            {"repository": "A/B\n- Evaluated source: `spoofed/repo`"},
            {"evaluator_container_revision": "x` <script>alert(1)</script> `"},
            {"commit": "`evil`"},
        ],
    )
    def test_hostile_payload_is_dropped_not_escaped(self, hostile: dict[str, str]) -> None:
        """A card can be rendered from a metadata dict that never passed the producer."""
        card = _card(hostile)
        for field in ("Evaluated source", "Evaluated source revision", "Evaluator container revision"):
            assert _value(card, field) == _UNRECORDED
        assert "INJECTED" not in card
        assert "spoofed/repo" not in card
        assert "<script>" not in card

    def test_two_repositories_sharing_an_evaluator_keep_distinct_source_revisions(self) -> None:
        """The regression this feature exists to prevent.

        Both skills are evaluated by the same evaluator container. Before the
        evaluated-source contract the only revision on either card was that
        shared container tag, so the two cards were indistinguishable.
        """
        first = _card({"repository": "NVIDIA/NVFlare", "commit": _COMMIT_A, "evaluator_container_revision": _CONTAINER})
        second = _card(
            {"repository": "NVIDIA/NeMo-Fabric", "commit": _COMMIT_B, "evaluator_container_revision": _CONTAINER}
        )

        assert _value(first, "Evaluated source revision") == f"`{_COMMIT_A}`"
        assert _value(second, "Evaluated source revision") == f"`{_COMMIT_B}`"
        assert _value(first, "Evaluated source") != _value(second, "Evaluated source")

        # The shared evaluator provenance stays identical, and is never mistaken
        # for the source revision -- the precise confusion issue #72 reported.
        shared = _value(first, "Evaluator container revision")
        assert shared == _value(second, "Evaluator container revision") == f"`{_CONTAINER}`"
        assert _value(first, "Evaluated source revision") != shared
        assert _value(second, "Evaluated source revision") != shared

    def test_a_run_without_a_commit_does_not_borrow_the_container_revision(self) -> None:
        """With no source revision the card says so, rather than showing the evaluator's."""
        card = _card({"repository": "NVIDIA/NVFlare", "evaluator_container_revision": _CONTAINER})
        assert _value(card, "Evaluated source revision") == _UNRECORDED
        assert _value(card, "Evaluator container revision") == f"`{_CONTAINER}`"


_PASS_CARD = """\
# Skill Benchmark: demo

> **Overall verdict: PASS**

## Evaluation Metadata

- Evaluation date: 2026-07-25
- Evaluator version: `0.8.3`
{source_lines}- Agents: Codex (`gpt-codex`)
- Tasks: 4 evaluation tasks
- Dataset digest: `sha256:{digest}` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `Isolated sandbox`
- Tier 3 evidence: {tier3_evidence}

## Results at a Glance

| Measure | Codex (Baseline → Skill Uplift) |
|---|---:|
| Overall | 47% → 92% (+45 points) |

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED** | Complete |
| Tier 2 | Semantic deduplication | **PASSED** | Complete |
| Tier 3 | Live agent evaluation | **{tier3_status}** | Complete |

## Freshness

Regenerate after material inputs change.
"""

_RECORDED = (
    "- Evaluated source: `NVIDIA/demo_skills`\n"
    f"- Evaluated source revision: `{_COMMIT_A}`\n"
    f"- Evaluator container revision: `{_CONTAINER}`\n"
)
_NOT_RECORDED = (
    f"- Evaluated source: {_UNRECORDED}\n"
    f"- Evaluated source revision: {_UNRECORDED}\n"
    f"- Evaluator container revision: {_UNRECORDED}\n"
)
_ABSENT = ""


def _pass_card(
    source_lines: str,
    *,
    tier3_status: str = "PASS",
    tier3_evidence: str = "required for publication",
) -> str:
    return _PASS_CARD.format(
        source_lines=source_lines,
        digest="0123456789abcdef" * 4,
        tier3_status=tier3_status,
        tier3_evidence=tier3_evidence,
    )


def _scan(tmp_path: Path, card: str, *, require: bool) -> list[str]:
    path = tmp_path / "BENCHMARK.md"
    path.write_text(card, encoding="utf-8")
    return [offender.reason for offender in benchmark_gate.scan_file(path, require_source_provenance=require)]


class TestPublicationGateIsOptIn:
    """The default scan stays byte-compatible with pre-contract behaviour."""

    def test_card_without_the_fields_passes_by_default(self, tmp_path: Path) -> None:
        assert _scan(tmp_path, _pass_card(_ABSENT), require=False) == []

    def test_card_without_the_fields_fails_when_required(self, tmp_path: Path) -> None:
        reasons = _scan(tmp_path, _pass_card(_ABSENT), require=True)
        assert "missing required section: - Evaluated source:" in reasons
        assert "missing required section: - Evaluated source revision:" in reasons

    def test_placeholder_provenance_is_tolerated_by_default(self, tmp_path: Path) -> None:
        assert _scan(tmp_path, _pass_card(_NOT_RECORDED), require=False) == []


class TestPublicationGateFailsClosed:
    """``--require-source-provenance`` makes a published PASS name its source."""

    def test_recorded_provenance_passes(self, tmp_path: Path) -> None:
        assert _scan(tmp_path, _pass_card(_RECORDED), require=True) == []

    def test_missing_provenance_fails_a_pass_card(self, tmp_path: Path) -> None:
        reasons = _scan(tmp_path, _pass_card(_NOT_RECORDED), require=True)
        assert "publication PASS without recorded evaluated source" in reasons
        assert "publication PASS without recorded evaluated source revision" in reasons

    def test_optional_tier3_policy_cannot_dodge_the_check(self, tmp_path: Path) -> None:
        """A PASS published under an optional-Tier-3 policy still needs the identity."""
        card = _pass_card(_NOT_RECORDED, tier3_status="SKIPPED", tier3_evidence="optional by policy")
        reasons = _scan(tmp_path, card, require=True)
        assert "publication PASS without recorded evaluated source" in reasons

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("Evaluated source", "`not a repository`"),
            ("Evaluated source", "`-leading/dash`"),
            ("Evaluated source revision", "`zzzz`"),
            ("Evaluator container revision", "`has space`"),
        ],
    )
    def test_malformed_values_are_rejected(self, tmp_path: Path, field: str, value: str) -> None:
        lines = []
        for line in _RECORDED.strip().splitlines():
            name = line.split(":", 1)[0][2:]
            lines.append(f"- {field}: {value}" if name == field else line)
        reasons = _scan(tmp_path, _pass_card("\n".join(lines) + "\n"), require=True)
        assert f"invalid metadata field: - {field}:" in reasons


class TestGateEntryPoints:
    """The flag has to survive main() and find_offenders(), not just scan_file()."""

    def _write(self, tmp_path: Path, card: str) -> Path:
        (tmp_path / "BENCHMARK.md").write_text(card, encoding="utf-8")
        return tmp_path

    def test_main_accepts_a_recorded_card(self, tmp_path: Path) -> None:
        root = self._write(tmp_path, _pass_card(_RECORDED))
        assert benchmark_gate.main(["--require-source-provenance", str(root)]) == 0

    def test_main_rejects_a_card_without_provenance(self, tmp_path: Path) -> None:
        root = self._write(tmp_path, _pass_card(_NOT_RECORDED))
        assert benchmark_gate.main(["--require-source-provenance", str(root)]) == 1

    def test_main_tolerates_the_same_card_by_default(self, tmp_path: Path) -> None:
        root = self._write(tmp_path, _pass_card(_NOT_RECORDED))
        assert benchmark_gate.main([str(root)]) == 0

    def test_find_offenders_propagates_the_flag(self, tmp_path: Path) -> None:
        root = self._write(tmp_path, _pass_card(_NOT_RECORDED))
        _files, offenders = benchmark_gate.find_offenders([root], require_source_provenance=True)
        assert offenders


class TestReporterGateRoundTrip:
    """Whatever the reporter can emit, the gate must accept."""

    @pytest.mark.parametrize(
        "source",
        [
            {"repository": "NVIDIA/NVFlare", "commit": _COMMIT_A, "evaluator_container_revision": _CONTAINER},
            {
                "repository": "holgerroth/nvflare_examples",
                "commit": _COMMIT_A,
                "evaluator_container_revision": _CONTAINER,
            },
            {"repository": "NVIDIA/Megatron_LM", "commit": _COMMIT_A, "evaluator_container_revision": _CONTAINER},
            {"repository": "some-org/repo.name", "content_digest": _DIGEST, "evaluator_container_revision": _CONTAINER},
        ],
    )
    def test_rendered_card_satisfies_the_strict_gate(self, tmp_path: Path, source: dict[str, str]) -> None:
        result = ValidationResult(
            validator_name="AGENT_EVAL",
            validator_description="Run live agent evaluation",
        )
        result.metadata["agent_eval"] = {
            "skill_name": "demo-skill",
            "evaluated_source": source,
            "summary": {"environment": "Isolated sandbox"},
            "agents": {"codex": {"model": "gpt-codex"}},
        }
        path = tmp_path / "BENCHMARK.md"
        path.write_text(BenchmarkReporter().render_all([result]), encoding="utf-8")
        assert [o.reason for o in benchmark_gate.scan_file(path, require_source_provenance=True)] == []


class TestIdentityResolution:
    """Two orchestration channels, resolved per field and never silently reconciled."""

    def test_fields_merge_across_channels(self) -> None:
        merged = resolve_evaluated_source({"repository": "NVIDIA/NVFlare"}, {"commit": _COMMIT_A})
        assert merged == {"repository": "NVIDIA/NVFlare", "commit": _COMMIT_A}

    def test_an_invalid_field_does_not_discard_the_other_channel(self) -> None:
        merged = resolve_evaluated_source(
            {"repository": "not a repository"},
            {"repository": "NVIDIA/NVFlare", "commit": _COMMIT_A},
        )
        assert merged == {"repository": "NVIDIA/NVFlare", "commit": _COMMIT_A}

    def test_conflicting_channels_fail_closed(self) -> None:
        with pytest.raises(EvaluatedSourceConflict, match="commit"):
            resolve_evaluated_source({"commit": _COMMIT_A}, {"commit": _COMMIT_B})

    def test_agreeing_channels_do_not_conflict(self) -> None:
        assert resolve_evaluated_source({"commit": _COMMIT_A}, {"commit": _COMMIT_A}) == {"commit": _COMMIT_A}

    def test_conflict_propagates_out_of_the_payload_builder(self) -> None:
        with pytest.raises(EvaluatedSourceConflict):
            _payload(
                run_config={"evaluated_source": {"repository": "NVIDIA/stale"}},
                evaluated_source={"repository": "NVIDIA/NVFlare"},
            )


class TestNonTier3CardsCanRecordTheIdentity:
    """A PASS can be published without a completed Tier 3 run, so those cards need a carrier."""

    def _card_from(self, *results: ValidationResult) -> str:
        return BenchmarkReporter().render_all(list(results))

    def test_result_metadata_carries_the_identity(self) -> None:
        tier1 = ValidationResult(validator_name="Schema", validator_description="d")
        tier1.add_success(check_name="ok", message="fine")
        tier1.metadata["evaluated_source"] = {"repository": "NVIDIA/NVFlare", "commit": _COMMIT_A}
        card = self._card_from(tier1)
        assert _value(card, "Evaluated source") == "`NVIDIA/NVFlare`"
        assert _value(card, "Evaluated source revision") == f"`{_COMMIT_A}`"

    def test_carriers_that_agree_are_merged(self) -> None:
        tier1 = ValidationResult(validator_name="Schema", validator_description="d")
        tier1.metadata["evaluated_source"] = {"repository": "NVIDIA/NVFlare"}
        tier3 = ValidationResult(validator_name="AGENT_EVAL", validator_description="d")
        tier3.metadata["agent_eval"] = {
            "skill_name": "demo-skill",
            "evaluated_source": {"repository": "NVIDIA/NVFlare", "commit": _COMMIT_A},
            "summary": {"environment": "Isolated sandbox"},
            "agents": {"codex": {"model": "gpt-codex"}},
        }
        card = self._card_from(tier3, tier1)
        assert _value(card, "Evaluated source") == "`NVIDIA/NVFlare`"
        assert _value(card, "Evaluated source revision") == f"`{_COMMIT_A}`"

    @pytest.mark.parametrize("reversed_order", [False, True])
    def test_contradictory_carriers_fail_closed(self, reversed_order: bool) -> None:
        """Result ordering must not decide which source tree a published card names."""
        tier1 = ValidationResult(validator_name="Schema", validator_description="d")
        tier1.metadata["evaluated_source"] = {"repository": "NVIDIA/stale", "commit": _COMMIT_B}
        tier3 = ValidationResult(validator_name="AGENT_EVAL", validator_description="d")
        tier3.metadata["agent_eval"] = {
            "skill_name": "demo-skill",
            "evaluated_source": {"repository": "NVIDIA/NVFlare", "commit": _COMMIT_A},
            "summary": {"environment": "Isolated sandbox"},
            "agents": {"codex": {"model": "gpt-codex"}},
        }
        results = [tier1, tier3] if reversed_order else [tier3, tier1]
        with pytest.raises(EvaluatedSourceConflict):
            self._card_from(*results)

    def test_payload_conflicting_with_its_own_summary_fails_closed(self) -> None:
        tier3 = ValidationResult(validator_name="AGENT_EVAL", validator_description="d")
        tier3.metadata["agent_eval"] = {
            "skill_name": "demo-skill",
            "evaluated_source": {"repository": "NVIDIA/NVFlare"},
            "summary": {"environment": "Isolated sandbox", "evaluated_source": {"repository": "NVIDIA/stale"}},
            "agents": {"codex": {"model": "gpt-codex"}},
        }
        with pytest.raises(EvaluatedSourceConflict):
            self._card_from(tier3)

    def test_two_results_disagreeing_fail_closed(self) -> None:
        first = ValidationResult(validator_name="Schema", validator_description="d")
        first.metadata["evaluated_source"] = {"commit": _COMMIT_A}
        second = ValidationResult(validator_name="Lint", validator_description="d")
        second.metadata["evaluated_source"] = {"commit": _COMMIT_B}
        with pytest.raises(EvaluatedSourceConflict):
            self._card_from(first, second)

    def test_hostile_result_metadata_is_dropped(self) -> None:
        tier1 = ValidationResult(validator_name="Schema", validator_description="d")
        tier1.metadata["evaluated_source"] = {"repository": "A/B` INJECTED `x"}
        card = self._card_from(tier1)
        assert _value(card, "Evaluated source") == _UNRECORDED
        assert "INJECTED" not in card


class TestPassDetection:
    """The verdict line is honoured whether or not the card blockquotes it."""

    _POLICY_BULLET = (
        "- Overall verdict: PASS only when every configured dimension passes for at least one supported agent."
    )
    _UNRECORDED_REASONS = (
        "publication PASS without recorded evaluated source",
        "publication PASS without recorded evaluated source revision",
        "publication PASS without recorded evaluator container revision",
    )

    @staticmethod
    def _unrecorded(reasons: list[str]) -> tuple[str, ...]:
        return tuple(reason for reason in reasons if reason.startswith("publication PASS without recorded"))

    @pytest.mark.parametrize(
        "verdict_line",
        [
            "> **Overall verdict: PASS**",
            "**Overall verdict: PASS**",
            # The callout the reporter renders, written with the em dash escaped.
            "> \u2705 **Overall verdict: PASS \u2014 Recommended for publication**",
        ],
    )
    def test_published_pass_needs_the_identity(self, tmp_path: Path, verdict_line: str) -> None:
        card = _pass_card(_NOT_RECORDED).replace("> **Overall verdict: PASS**", verdict_line, 1)
        reasons = _scan(tmp_path, card, require=True)
        assert "publication PASS without recorded evaluated source" in reasons

    @pytest.mark.parametrize("verdict", ["INCOMPLETE", "FAIL"])
    def test_the_policy_bullet_is_not_a_published_verdict(self, tmp_path: Path, verdict: str) -> None:
        """The policy section restates the rule as a bullet under every verdict.

        A card that publishes no PASS claims no source identity, so reading that
        bullet as the verdict charges it with three offences it never committed.
        """
        card = _pass_card(_ABSENT).replace("> **Overall verdict: PASS**", f"> **Overall verdict: {verdict}**", 1)
        card = f"{card}\n## Scoring Policy\n\n{self._POLICY_BULLET}\n"
        assert self._unrecorded(_scan(tmp_path, card, require=True)) == ()

    def test_the_published_card_carries_that_bullet_and_is_still_detected(self, tmp_path: Path) -> None:
        """Excluding bullets must not blind the check on the card the reporter writes."""
        card = (Path(__file__).parent / "golden" / "benchmark_pass" / "BENCHMARK.md").read_text(encoding="utf-8")
        assert self._POLICY_BULLET in card
        assert _scan(tmp_path, card, require=True) == []

        stripped = "\n".join(
            line for line in card.splitlines() if not line.startswith(benchmark_gate._SOURCE_PROVENANCE_MARKERS)
        )
        assert self._unrecorded(_scan(tmp_path, stripped, require=True)) == self._UNRECORDED_REASONS


class TestStrictGateRevisionSyntax:
    """The stdlib-only gate mirrors the library rules, so neither side can drift."""

    _STRICT = "publication PASS without recorded evaluator container revision"
    _ADVISORY = "invalid metadata field: - Evaluator container revision:"

    @staticmethod
    def _container_reasons(tmp_path: Path, revision: str) -> list[str]:
        card = _pass_card(
            f"- Evaluated source: `NVIDIA/NVFlare`\n"
            f"- Evaluated source revision: `{_COMMIT_A}`\n"
            f"- Evaluator container revision: `{revision}`\n"
        )
        return _scan(tmp_path, card, require=True)

    def test_the_gate_mirrors_the_library_reference_grammar(self) -> None:
        """Two copies of one rule stay one rule only while their source text is identical."""
        assert benchmark_gate._CONTAINER_REFERENCE_PATTERN == source_identity._CONTAINER_REFERENCE_PATTERN
        assert benchmark_gate._CONTAINER_PATH_MAX == source_identity._CONTAINER_PATH_MAX
        # The gate's hand copy of the revision pattern went uncompared, so it
        # could drift from the library while the reference grammar still matched.
        assert benchmark_gate._GIT_OBJECT_ID.pattern == source_identity._SOURCE_COMMIT.pattern

    def test_a_long_repository_pinned_by_digest_satisfies_a_pass(self, tmp_path: Path) -> None:
        """The reference is 140 characters, which the old whole-reference cap discarded."""
        reference = "ghcr.io/nvidia/" + "e" * 53 + "@sha256:" + "0117bc2e" * 8
        assert len(reference) == 140
        assert self._container_reasons(tmp_path, reference) == []

    @pytest.mark.parametrize(("length", "accepted"), [(255, True), (256, False)])
    def test_the_path_is_bounded_at_the_oci_maximum(self, tmp_path: Path, length: int, accepted: bool) -> None:
        reference = "n" * length + "@sha256:" + "0117bc2e" * 8
        reasons = self._container_reasons(tmp_path, reference)
        assert (self._STRICT not in reasons) is accepted

    @pytest.mark.parametrize(("length", "accepted"), [(255, True), (256, False)])
    def test_a_registry_host_does_not_spend_the_path_budget(self, tmp_path: Path, length: int, accepted: bool) -> None:
        """The bound measures the path once the host is split off, as the reference grammar does."""
        reference = "ghcr.io/" + "n" * length + "@sha256:" + "0117bc2e" * 8
        reasons = self._container_reasons(tmp_path, reference)
        assert (self._STRICT not in reasons) is accepted

    @pytest.mark.parametrize(
        "revision",
        [
            "NVIDIA/skillevaluator@sha256:" + "0117bc2e" * 8,
            "ghcr.io/NVIDIA/skillevaluator@sha256:" + "0117bc2e" * 8,
        ],
    )
    def test_an_uppercase_path_satisfies_no_pass_either_way(self, tmp_path: Path, revision: str) -> None:
        """One rule for both spellings: a first label is a host only where the grammar says so."""
        assert self._STRICT in self._container_reasons(tmp_path, revision)

    @pytest.mark.parametrize(
        "revision",
        [
            "localhost/team/image@sha256:" + "0117bc2e" * 8,
            "Registry.Example.COM/team/image@sha256:" + "0117bc2e" * 8,
            "myregistry:5000/team/image@sha256:" + "0117bc2e" * 8,
        ],
    )
    def test_a_registry_host_is_localhost_dotted_or_ported(self, tmp_path: Path, revision: str) -> None:
        """Those are the three shapes the grammar reads as a host, and a host name may use any case."""
        assert self._container_reasons(tmp_path, revision) == []

    def test_a_registry_port_and_tag_still_pin_by_digest(self, tmp_path: Path) -> None:
        reference = "localhost:5000/team/image:1.2.3@sha256:" + "0117bc2e" * 8
        assert self._container_reasons(tmp_path, reference) == []

    def test_an_implementation_revision_satisfies_a_pass(self, tmp_path: Path) -> None:
        """An evaluator built from a checkout has no image digest to record."""
        assert self._container_reasons(tmp_path, _COMMIT_A) == []

    def test_a_tagged_reference_is_recorded_but_not_publishable(self, tmp_path: Path) -> None:
        """Normalization keeps the tag the operator supplied; only publication demands the digest."""
        reasons = self._container_reasons(tmp_path, "nvcr.io/nvidia/skillevaluator:1.3.2")
        assert self._ADVISORY not in reasons
        assert self._STRICT in reasons

    @pytest.mark.parametrize(
        "revision",
        [
            "ghcr.io/nvidia/skillevaluator",
            "ghcr.io/nvidia/skillevaluator@sha512:" + "ab" * 16,
            "ghcr.io/nvidia/skillevaluator@md5:" + "ab" * 16,
            "ghcr.io/nvidia/skillevaluator:" + "t" * 129,
            "has space",
            "ghcr.io/nvidia/skillevaluator`` injected",
        ],
    )
    def test_references_that_name_no_build_do_not_satisfy_a_pass(self, tmp_path: Path, revision: str) -> None:
        assert self._STRICT in self._container_reasons(tmp_path, revision)

    @pytest.mark.parametrize("revision", [_COMMIT_A[:7], "sha512:" + "ab" * 16])
    def test_ambiguous_revision_does_not_satisfy_a_pass(self, tmp_path: Path, revision: str) -> None:
        card = _pass_card(
            f"- Evaluated source: `NVIDIA/NVFlare`\n"
            f"- Evaluated source revision: `{revision}`\n"
            f"- Evaluator container revision: `{_CONTAINER}`\n"
        )
        reasons = _scan(tmp_path, card, require=True)
        assert "publication PASS without recorded evaluated source revision" in reasons

    def test_pass_without_a_container_revision_is_flagged(self, tmp_path: Path) -> None:
        """#72 was a card whose only revision was the evaluator image, so strict mode needs it too."""
        card = _pass_card(
            f"- Evaluated source: `NVIDIA/NVFlare`\n"
            f"- Evaluated source revision: `{_COMMIT_A}`\n"
            f"- Evaluator container revision: {_UNRECORDED}\n"
        )
        reasons = _scan(tmp_path, card, require=True)
        assert "publication PASS without recorded evaluator container revision" in reasons

    def test_mutable_container_tag_does_not_satisfy_a_pass(self, tmp_path: Path) -> None:
        """A tag can be repointed after publication, so it cannot pin the build that ran."""
        card = _pass_card(
            f"- Evaluated source: `NVIDIA/NVFlare`\n"
            f"- Evaluated source revision: `{_COMMIT_A}`\n"
            f"- Evaluator container revision: `ghcr.io/nvidia/skillevaluator:latest`\n"
        )
        reasons = _scan(tmp_path, card, require=True)
        assert "publication PASS without recorded evaluator container revision" in reasons

    def test_digest_pinned_container_satisfies_a_pass(self, tmp_path: Path) -> None:
        card = _pass_card(
            f"- Evaluated source: `NVIDIA/NVFlare`\n"
            f"- Evaluated source revision: `{_COMMIT_A}`\n"
            f"- Evaluator container revision: `{_CONTAINER}`\n"
        )
        reasons = _scan(tmp_path, card, require=True)
        assert "publication PASS without recorded evaluator container revision" not in reasons


class TestCliSurfacesTheConflict:
    """Failing closed must still read as a CLI error, not an unhandled traceback."""

    def test_validate_reports_a_conflicting_identity_cleanly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from click.testing import CliRunner

        from skillevaluator.cli import cli
        from skillevaluator.reporting import BenchmarkReporter
        from skillevaluator.validators.code_risk import CodeRiskValidator
        from skillevaluator.validators.secrets import SecretsValidator

        skill = tmp_path / "conflicting-source"
        skill.mkdir()
        (skill / "SKILL.md").write_text(
            "---\n"
            "name: conflicting-source\n"
            "description: A skill whose run records two different evaluated sources.\n"
            "---\n"
            "\n"
            "# Conflicting source\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(CodeRiskValidator, "validate", lambda _self, _path: ValidationResult())
        monkeypatch.setattr(SecretsValidator, "validate", lambda _self, _path: ValidationResult())

        def _conflict(*_args: object, **_kwargs: object) -> None:
            raise EvaluatedSourceConflict("conflicting evaluated source identity (repository: 'A/b' vs 'C/d')")

        monkeypatch.setattr(BenchmarkReporter, "save", _conflict)

        result = CliRunner().invoke(
            cli,
            ["validate", "--no-tier3", str(skill), "--no-dedup", "-r", "cli", "-o", str(tmp_path / "out")],
        )

        assert result.exit_code != 0
        assert "evaluated source" in result.output
        assert "Traceback" not in result.output


class TestCliSuppliesTheIdentity:
    """A normal validate run records the identity, which is what #72 asked for."""

    def _skill(self, tmp_path: Path) -> Path:
        skill = tmp_path / "demo-skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text(
            "---\n"
            "name: demo-skill\n"
            "description: A minimal skill used to check the recorded source identity.\n"
            "---\n"
            "\n"
            "# Demo skill\n",
            encoding="utf-8",
        )
        return skill

    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *args: str):
        from click.testing import CliRunner

        from skillevaluator.cli import cli
        from skillevaluator.validators.code_risk import CodeRiskValidator
        from skillevaluator.validators.secrets import SecretsValidator

        # Exercise the public CLI without the independent scanner integrations.
        monkeypatch.setattr(CodeRiskValidator, "validate", lambda _self, _path: ValidationResult())
        monkeypatch.setattr(SecretsValidator, "validate", lambda _self, _path: ValidationResult())
        output_dir = tmp_path / "out"
        invocation = CliRunner().invoke(
            cli,
            [
                "validate",
                "--no-tier3",
                str(self._skill(tmp_path)),
                "--no-dedup",
                "-r",
                "cli",
                "-o",
                str(output_dir),
                *args,
            ],
        )
        return invocation, output_dir / "BENCHMARK.md"

    def test_supplied_identity_reaches_the_card(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _invocation, card_path = self._run(
            tmp_path,
            monkeypatch,
            "--evaluated-source-repository",
            "NVIDIA/NVFlare",
            "--evaluated-source-revision",
            _COMMIT_A,
            "--evaluator-container-revision",
            _CONTAINER,
        )
        card = card_path.read_text(encoding="utf-8")
        assert _value(card, "Evaluated source") == "`NVIDIA/NVFlare`"
        assert _value(card, "Evaluated source revision") == f"`{_COMMIT_A}`"
        assert _value(card, "Evaluator container revision") == f"`{_CONTAINER}`"

    def test_a_content_digest_is_accepted_as_the_revision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run with no upstream commit still records an immutable revision."""
        _invocation, card_path = self._run(tmp_path, monkeypatch, "--evaluated-source-revision", _DIGEST)
        assert _value(card_path.read_text(encoding="utf-8"), "Evaluated source revision") == f"`{_DIGEST}`"

    def test_without_the_options_the_card_says_not_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default behaviour is unchanged for a run with no orchestration input."""
        _invocation, card_path = self._run(tmp_path, monkeypatch)
        card = card_path.read_text(encoding="utf-8")
        assert _value(card, "Evaluated source") == _UNRECORDED
        assert _value(card, "Evaluated source revision") == _UNRECORDED

    @pytest.mark.parametrize(
        ("option", "value"),
        [
            ("--evaluated-source-revision", _COMMIT_A[:7]),
            ("--evaluated-source-revision", "sha512:" + "ab" * 16),
            ("--evaluated-source-repository", "not a repository"),
            ("--evaluator-container-revision", "has space"),
        ],
    )
    def test_non_canonical_values_are_refused_at_the_boundary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option: str, value: str
    ) -> None:
        """Rejecting here beats rendering `not recorded` for a field the operator supplied."""
        invocation, card_path = self._run(tmp_path, monkeypatch, option, value)
        assert invocation.exit_code != 0
        assert "expected" in invocation.output
        assert not card_path.exists()


class TestIdentityReachesTheRunArtifact:
    """The CLI identity has to survive every hop down to `run_config.json`."""

    def test_evaluation_options_forward_the_identity(self) -> None:
        from skillevaluator.evaluation import EvaluationOptions

        identity = {"repository": "NVIDIA/NVFlare", "commit": _COMMIT_A}
        options = EvaluationOptions(skill_path=Path(), evaluated_source=identity)
        assert options.engine_kwargs()["evaluated_source"] == identity

    @pytest.mark.parametrize(
        ("module_path", "function_name"),
        [
            ("skillevaluator.cli", "_run_agent_eval_or_skip"),
            ("skillevaluator.tier3.commands", "evaluate"),
            ("skillevaluator.tier3.harbor.runner", "_run_harbor_eval_impl"),
        ],
    )
    def test_each_hop_accepts_the_identity(self, module_path: str, function_name: str) -> None:
        import importlib
        import inspect

        module = importlib.import_module(module_path)
        signature = inspect.signature(getattr(module, function_name))
        assert "evaluated_source" in signature.parameters

    def test_a_standalone_run_persists_the_identity_in_run_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A signature alone does not prove the value is written to disk.

        The standalone ``tier3 evaluate`` producer owns its own run directory, so
        the file a card is later rendered from has to carry the identity rather
        than the ``null`` a dropped option leaves behind.
        """
        import json

        from skillevaluator.provider_config import ProviderConfig
        from skillevaluator.tier3 import commands as tier3_commands
        from skillevaluator.tier3.harbor import runner, runtime_preflight

        skill = tmp_path / "demo"
        (skill / "evals").mkdir(parents=True)
        (skill / "evals" / "evals.json").write_text("[]\n", encoding="utf-8")
        provider = ProviderConfig(
            provider="nv_build",
            model="meta/llama-3.1-8b-instruct",
            api_key="nvapi-test",
            base_url="https://integrate.api.nvidia.com/v1",
            litellm_model="openai/meta/llama-3.1-8b-instruct",
        )

        def emit(_skill, target, **_kwargs):
            task = target / "case-001"
            task.mkdir(parents=True)
            return [task]

        monkeypatch.setattr(tier3_commands, "resolve_llm_provider", lambda: provider)
        monkeypatch.setattr(runner, "resolve_llm_provider", lambda: provider)
        monkeypatch.setattr(
            runner,
            "load_evals_config",
            lambda _path: ({"harbor": {"task_source": "evals_json"}}, None),
        )
        monkeypatch.setattr(runner, "find_evals_file", lambda _path: skill / "evals" / "evals.json")
        monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])
        monkeypatch.setattr(runner, "generate_harbor_tasks", emit)
        monkeypatch.setattr(
            runtime_preflight,
            "probe_model",
            lambda selected_provider: runtime_preflight.ModelProbeResult(
                True,
                selected_provider.provider,
                selected_provider.model,
                f"model {selected_provider.model} is available",
            ),
        )
        # The preflight is failed deliberately: the run stops before any agent
        # work, and `run_config.json` is still written, which is the artifact
        # under test.
        monkeypatch.setattr(
            runtime_preflight,
            "run_agent_runtime_preflight",
            lambda **_kwargs: runtime_preflight.PreflightResult(
                False,
                "opencode",
                provider.model,
                "401 Unauthorized",
                "runtime-preflight-opencode",
            ),
        )

        identity = {"repository": "NVIDIA/NVFlare", "commit": _COMMIT_A}
        result = tier3_commands.evaluate(
            skill,
            agents="opencode",
            env_mode="docker",
            skip_baseline=True,
            n_attempts=None,
            pass_threshold=None,
            n_concurrent=None,
            max_agents=None,
            model=None,
            agent_model=(),
            custom_dockerfile_mode=None,
            skill_workspace_mode=None,
            include_skills=(),
            copy_repo=False,
            grading_mode=None,
            results_dir=tmp_path / "results",
            harbor_keep_jobs=False,
            agent_runtime_preflight=True,
            timeout_multiplier=None,
            override_cpus=None,
            override_memory_mb=None,
            override_storage_mb=None,
            evaluated_source=identity,
        )

        run_config_path = Path(result["run_dir"]) / "run_config.json"
        persisted = json.loads(run_config_path.read_text(encoding="utf-8"))
        assert persisted["evaluated_source"] == identity


def _agent_eval_carrier(repository: str) -> dict[str, object]:
    """A complete agent-eval payload naming one repository in both of its carriers."""
    return {
        "skill_name": "demo-skill",
        "evaluated_source": {"repository": repository},
        "summary": {"environment": "Isolated sandbox", "evaluated_source": {"repository": repository}},
        "agents": {"codex": {"model": "gpt-codex"}},
    }


def _agent_eval_result(payload: dict[str, object]) -> ValidationResult:
    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Run live agent evaluation")
    result.metadata["agent_eval"] = payload
    return result


class TestEveryNestedAgentEvalCarrierIsFolded:
    """Selecting one payload let result ordering decide which source tree a card named."""

    @pytest.mark.parametrize("reversed_order", [False, True])
    def test_two_payloads_naming_different_repositories_fail_closed(self, reversed_order: bool) -> None:
        first = _agent_eval_result(_agent_eval_carrier("NVIDIA/NVFlare"))
        second = _agent_eval_result(_agent_eval_carrier("NVIDIA/stale"))
        results = [second, first] if reversed_order else [first, second]
        with pytest.raises(EvaluatedSourceConflict, match="repository"):
            BenchmarkReporter().render_all(results)

    @pytest.mark.parametrize("reversed_order", [False, True])
    def test_two_payloads_that_agree_still_render_the_identity(self, reversed_order: bool) -> None:
        """Folding must not turn a repeated identity into a conflict."""
        first = _agent_eval_result(_agent_eval_carrier("NVIDIA/NVFlare"))
        second = _agent_eval_result(_agent_eval_carrier("NVIDIA/NVFlare"))
        results = [second, first] if reversed_order else [first, second]
        assert _value(BenchmarkReporter().render_all(results), "Evaluated source") == "`NVIDIA/NVFlare`"

    @pytest.mark.parametrize("reversed_order", [False, True])
    def test_a_summary_only_carrier_on_a_second_result_still_conflicts(self, reversed_order: bool) -> None:
        """The deepest carrier of the second payload counts as much as the first payload's."""
        payload = _agent_eval_result(_agent_eval_carrier("NVIDIA/NVFlare"))
        summary_only = _agent_eval_result(
            {
                "skill_name": "demo-skill",
                "summary": {
                    "environment": "Isolated sandbox",
                    "evaluated_source": {"repository": "NVIDIA/stale"},
                },
                "agents": {"codex": {"model": "gpt-codex"}},
            }
        )
        results = [summary_only, payload] if reversed_order else [payload, summary_only]
        with pytest.raises(EvaluatedSourceConflict, match="repository"):
            BenchmarkReporter().render_all(results)


class TestJsonReportRecordsTheIdentity:
    """CI parses the JSON report, so the provenance contract has to hold there too."""

    def _report(self, *results: ValidationResult) -> dict[str, object]:
        import json

        from skillevaluator.reporting.json_reporter import JSONReporter

        return json.loads(JSONReporter(include_timestamp=False).render_all(list(results)))

    def test_result_metadata_reaches_the_top_level(self) -> None:
        tier1 = ValidationResult(validator_name="Schema", validator_description="d")
        tier1.metadata["evaluated_source"] = {"repository": "NVIDIA/NVFlare", "commit": _COMMIT_A}
        assert self._report(tier1)["evaluated_source"] == {
            "repository": "NVIDIA/NVFlare",
            "commit": _COMMIT_A,
        }

    def test_a_nested_payload_carrier_reaches_the_top_level(self) -> None:
        """A Tier 3 run records the identity on the payload, not on the result."""
        report = self._report(_agent_eval_result(_agent_eval_carrier("NVIDIA/NVFlare")))
        assert report["evaluated_source"] == {"repository": "NVIDIA/NVFlare"}

    def test_a_run_with_no_identity_records_null(self) -> None:
        """The key is always present, so null means none was supplied, not an older reporter."""
        report = self._report(ValidationResult(validator_name="Schema", validator_description="d"))
        assert "evaluated_source" in report
        assert report["evaluated_source"] is None

    def test_conflicting_carriers_fail_closed(self) -> None:
        first = ValidationResult(validator_name="Schema", validator_description="d")
        first.metadata["evaluated_source"] = {"commit": _COMMIT_A}
        second = ValidationResult(validator_name="Lint", validator_description="d")
        second.metadata["evaluated_source"] = {"commit": _COMMIT_B}
        with pytest.raises(EvaluatedSourceConflict, match="commit"):
            self._report(first, second)


def _minimal_skill(tmp_path: Path, name: str, description: str) -> Path:
    skill = tmp_path / name
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n",
        encoding="utf-8",
    )
    return skill


class TestTier1OnlyReportsCarryTheIdentity:
    """A Tier 1-only run has no Tier 3 payload, and its JSON report still has to say what ran."""

    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *args: str) -> Path:
        from click.testing import CliRunner

        from skillevaluator.cli import cli
        from skillevaluator.validators.code_risk import CodeRiskValidator
        from skillevaluator.validators.secrets import SecretsValidator

        # Exercise the public CLI without the independent scanner integrations.
        monkeypatch.setattr(CodeRiskValidator, "validate", lambda _self, _path: ValidationResult())
        monkeypatch.setattr(SecretsValidator, "validate", lambda _self, _path: ValidationResult())
        skill = _minimal_skill(tmp_path, "demo-skill", "A minimal skill used to check the recorded identity.")
        output_dir = tmp_path / "out"
        invocation = CliRunner().invoke(
            cli,
            ["validate", "--no-tier3", str(skill), "--no-dedup", "-r", "json", "-o", str(output_dir), *args],
        )
        assert "evaluated source" not in invocation.output
        return output_dir

    def test_the_json_report_and_the_card_record_the_same_identity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json

        output_dir = self._run(
            tmp_path,
            monkeypatch,
            "--evaluated-source-repository",
            "NVIDIA/NVFlare",
            "--evaluated-source-revision",
            _COMMIT_A,
            "--evaluator-container-revision",
            _CONTAINER,
        )

        report = json.loads(next(output_dir.glob("*.json")).read_text(encoding="utf-8"))
        assert report["evaluated_source"] == {
            "repository": "NVIDIA/NVFlare",
            "commit": _COMMIT_A,
            "evaluator_container_revision": _CONTAINER,
        }

        card = (output_dir / "BENCHMARK.md").read_text(encoding="utf-8")
        assert _value(card, "Evaluated source") == "`NVIDIA/NVFlare`"
        assert _value(card, "Evaluated source revision") == f"`{_COMMIT_A}`"
        assert _value(card, "Evaluator container revision") == f"`{_CONTAINER}`"


class TestConflictAbortsBeforeAnyReportIsWritten:
    """Failing closed after emit left JSON and HTML on disk carrying one of two identities."""

    def test_no_report_file_survives_a_contradictory_identity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from click.testing import CliRunner

        from skillevaluator import cli as cli_module
        from skillevaluator.cli import cli

        skill = _minimal_skill(tmp_path, "conflicting-source", "A skill whose run records two sources.")
        tier1 = ValidationResult(validator_name="SCHEMA")
        tier1.add_success("schema", "ok")
        tier1.metadata["evaluated_source"] = {"repository": "C/d"}
        monkeypatch.setattr(cli_module, "run_validation", lambda *_args, **_kwargs: [tier1])

        output_dir = tmp_path / "out"
        invocation = CliRunner().invoke(
            cli,
            [
                "validate",
                "--no-tier3",
                str(skill),
                "--no-dedup",
                "-r",
                "json",
                "-r",
                "html",
                "-o",
                str(output_dir),
                "--evaluated-source-repository",
                "A/b",
            ],
        )

        assert invocation.exit_code != 0
        assert "evaluated source" in invocation.output
        assert "Traceback" not in invocation.output
        written = sorted(path.name for path in output_dir.rglob("*")) if output_dir.exists() else []
        assert [name for name in written if name.endswith((".json", ".html")) or name == "BENCHMARK.md"] == []


class TestPartialRecordedIdentityMergesWithTheInput:
    """A producer that recorded one field must not shadow the fields the operator supplied."""

    def test_a_partial_carrier_and_the_options_publish_one_union(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Attaching the supplied identity only where none was recorded dropped the rest of it."""
        import json

        from click.testing import CliRunner

        from skillevaluator import cli as cli_module
        from skillevaluator.cli import cli

        skill = _minimal_skill(tmp_path, "partial-source", "A skill whose run records only the repository.")
        tier1 = ValidationResult(validator_name="SCHEMA")
        tier1.add_success("schema", "ok")
        # The documented result-metadata seam, carrying one field and no more.
        tier1.metadata["evaluated_source"] = {"repository": "NVIDIA/NVFlare"}
        monkeypatch.setattr(cli_module, "run_validation", lambda *_args, **_kwargs: [tier1])

        output_dir = tmp_path / "out"
        invocation = CliRunner().invoke(
            cli,
            [
                "validate",
                "--no-tier3",
                str(skill),
                "--no-dedup",
                "-r",
                "json",
                "-o",
                str(output_dir),
                "--evaluated-source-revision",
                _COMMIT_A,
                "--evaluator-container-revision",
                _CONTAINER,
            ],
        )

        assert "evaluated source" not in invocation.output
        report = json.loads(next(output_dir.glob("*.json")).read_text(encoding="utf-8"))
        assert report["evaluated_source"] == {
            "repository": "NVIDIA/NVFlare",
            "commit": _COMMIT_A,
            "evaluator_container_revision": _CONTAINER,
        }

        card = (output_dir / "BENCHMARK.md").read_text(encoding="utf-8")
        assert _value(card, "Evaluated source") == "`NVIDIA/NVFlare`"
        assert _value(card, "Evaluated source revision") == f"`{_COMMIT_A}`"
        assert _value(card, "Evaluator container revision") == f"`{_CONTAINER}`"
