# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SkillSpector LLM provenance report boundary."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from skillevaluator.skillspector_provenance import (
    SKILLSPECTOR_LLM_ANALYZERS,
    SKILLSPECTOR_PROVIDER_IDS,
    sanitize_skillspector_llm_provenance,
    sanitize_skillspector_provenance_metadata,
    skillspector_provenance_records,
)
from skillevaluator.validators.security import _SKILLSPECTOR_PROVIDER_MAP

# ``metadata.llm_provenance`` as SkillSpector v2.12.0 reports it for an
# ``openai`` scan that forwarded SKILLSPECTOR_REASONING_EFFORT=medium.
_FIXTURE = Path(__file__).parent / "fixtures" / "skillspector-2.12.0-llm-provenance.json"


def _provenance() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _set_model(record: dict, model: object) -> None:
    for analyzer in record["analyzers"]:
        analyzer["model"] = model


def test_skillspector_record_round_trips_unchanged() -> None:
    record = _provenance()

    assert sanitize_skillspector_llm_provenance(record) == record


def test_canonical_record_drops_unknown_fields_and_is_a_copy() -> None:
    record = _provenance()
    record["prompt"] = "do-not-export"
    record["provider"]["endpoint"] = "https://gateway.example.test/v1"
    record["analyzers"][0]["raw_response"] = "do-not-export"
    record["sampling"]["seed"]["extra"] = True

    canonical = sanitize_skillspector_llm_provenance(record)
    record["analyzers"][0]["model"] = "changed"

    assert canonical is not None
    rendered = json.dumps(canonical)
    assert "do-not-export" not in rendered
    assert "gateway.example.test" not in rendered
    assert "extra" not in canonical["sampling"]["seed"]
    assert canonical["analyzers"][0]["model"] == "gpt-6.1-sol"


def test_record_without_source_revision_is_accepted() -> None:
    record = _provenance()
    for analyzer in record["analyzers"]:
        del analyzer["analyzer_revision"]["source_revision"]

    canonical = sanitize_skillspector_llm_provenance(record)

    assert canonical is not None
    assert canonical["analyzers"][0]["analyzer_revision"] == {"value": "2.12.0", "source": "skillspector_package"}


def test_supported_control_the_adapter_omitted_is_partially_forwarded() -> None:
    record = _provenance()
    record["sampling"]["reasoning_effort"]["forwarded_to_client"] = None
    record["determinism"]["control_status"] = "controls_partially_forwarded"

    assert sanitize_skillspector_llm_provenance(record) is not None


def test_responses_without_recorded_controls_are_not_observed() -> None:
    record = _provenance()
    record["determinism"]["control_status"] = "controls_not_observed"

    assert sanitize_skillspector_llm_provenance(record) is not None


def test_llm_stage_without_executed_analyzers_is_not_applicable() -> None:
    record = _provenance()
    record["provider"].update({"effective_adapter": "not_applicable", "effective_adapters": []})
    record["sampling"]["reasoning_effort"]["forwarded_to_client"] = None
    record["determinism"].update({"classification": "not_applicable", "control_status": "not_applied"})

    canonical = sanitize_skillspector_llm_provenance(record)

    assert canonical is not None
    assert canonical["determinism"]["reason"] == "LLM analysis was not executed for this scan."


@pytest.mark.parametrize(
    ("control", "update"),
    [
        # A client default the provider request carried although nothing was requested.
        pytest.param(
            "temperature",
            {"requested": None, "source": "provider_default", "forwarded_to_client": 1.0},
            id="client-injected-default",
        ),
        # A client override of the requested value.
        pytest.param(
            "temperature",
            {"requested": 0.0, "source": "environment", "forwarded_to_client": 1.0},
            id="client-overrode-request",
        ),
        pytest.param("reasoning_effort", {"forwarded_to_client": "high"}, id="forwarded-differs-from-request"),
    ],
)
def test_controls_the_client_forwarded_on_its_own_are_kept(control: str, update: dict) -> None:
    record = _provenance()
    record["sampling"][control].update(update)

    canonical = sanitize_skillspector_llm_provenance(record)

    assert canonical is not None
    assert canonical["sampling"][control] == record["sampling"][control]


def _anthropic_record() -> dict:
    record = _provenance()
    record["provider"].update(
        configured_adapter="anthropic",
        resolved_adapter="anthropic",
        effective_adapter="anthropic",
        effective_adapters=["anthropic"],
    )
    record["sampling"]["seed"]["adapter_support"] = False
    return record


def test_adapter_without_seed_support_is_accepted() -> None:
    record = _anthropic_record()

    assert sanitize_skillspector_llm_provenance(record) == record


def _forward_seed_on_anthropic(record: dict) -> None:
    record.clear()
    record.update(_anthropic_record())
    record["sampling"]["seed"]["forwarded_to_client"] = 7


def _set_schema_version(value: object) -> Callable[[dict], None]:
    return lambda record: record.update(schema_version=value)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_set_schema_version(2), id="future-schema"),
        pytest.param(_set_schema_version("1"), id="string-schema"),
        pytest.param(_set_schema_version(True), id="boolean-schema"),
        pytest.param(lambda record: record["analyzers"].pop(), id="missing-analyzer"),
        pytest.param(lambda record: record["analyzers"].append(record["analyzers"][0]), id="duplicate-analyzer"),
        pytest.param(lambda record: record["analyzers"][0].update(analyzer_id="other"), id="unknown-analyzer"),
        pytest.param(lambda record: _set_model(record, "https://gateway.example.test/v1"), id="url-model"),
        pytest.param(lambda record: _set_model(record, "user@example.test"), id="email-model"),
        pytest.param(lambda record: _set_model(record, "sk-" + "a" * 24), id="api-key-model"),
        pytest.param(lambda record: _set_model(record, "AI" + "za" + "A" * 35), id="google-key-model"),
        pytest.param(lambda record: _set_model(record, "hf" + "_" + "a" * 34), id="hugging-face-token-model"),
        pytest.param(lambda record: _set_model(record, "npm" + "_" + "a" * 36), id="npm-token-model"),
        pytest.param(lambda record: _set_model(record, "pypi" + "-" + "A" * 90), id="pypi-token-model"),
        pytest.param(lambda record: _set_model(record, "model <b>"), id="markup-model"),
        pytest.param(lambda record: _set_model(record, None), id="missing-model"),
        pytest.param(
            lambda record: record["analyzers"][0]["analyzer_revision"]["source_revision"].update(value="not-hex"),
            id="bad-source-revision",
        ),
        pytest.param(lambda record: record["provider"].update(service="api.example.test"), id="service-named"),
        pytest.param(
            lambda record: record["provider"]["routing"].update(deployment_override="deploy"),
            id="routing-on-non-azure",
        ),
        pytest.param(lambda record: record["provider"].update(effective_adapter="anthropic"), id="wrong-effective"),
        pytest.param(
            lambda record: record["provider"].update(effective_adapters=["openai", "anthropic"]),
            id="unsorted-adapters",
        ),
        pytest.param(
            lambda record: record["sampling"]["seed"].update(adapter_support=False),
            id="adapter-support-mismatch",
        ),
        pytest.param(_forward_seed_on_anthropic, id="forwarded-without-adapter-support"),
        pytest.param(
            lambda record: record["sampling"]["temperature"].update(forwarded_to_client=2.0),
            id="forwarded-temperature-out-of-range",
        ),
        pytest.param(
            lambda record: record["sampling"]["seed"].update(forwarded_to_client=True),
            id="forwarded-boolean-seed",
        ),
        pytest.param(
            lambda record: record["sampling"]["reasoning_effort"].update(forwarded_to_client="sk-" + "a" * 24),
            id="forwarded-credential",
        ),
        pytest.param(
            lambda record: record["sampling"]["seed"].update(requested=7),
            id="requested-without-environment",
        ),
        pytest.param(
            lambda record: record["sampling"]["temperature"].update(requested=2.0, source="environment"),
            id="temperature-out-of-range",
        ),
        pytest.param(lambda record: record["determinism"].update(provider_guarantee=True), id="guarantee"),
        pytest.param(
            lambda record: record["determinism"].update(control_status="provider_defaults"),
            id="wrong-control-status",
        ),
        pytest.param(
            lambda record: record["determinism"].update(classification="deterministic"),
            id="unknown-classification",
        ),
    ],
)
def test_invalid_or_unsafe_records_are_omitted(mutate: Callable[[dict], object]) -> None:
    record = _provenance()
    mutate(record)

    assert sanitize_skillspector_llm_provenance(record) is None


def test_metadata_projection_keeps_only_canonical_fields() -> None:
    metadata = {
        "skillspector_llm_provenance": _provenance(),
        "skillspector_provider_id": "nv_build",
        "skillspector_model": "gpt-6.1-sol",
    }

    projected = sanitize_skillspector_provenance_metadata(metadata)

    assert set(projected) == {"skillspector_llm_provenance", "skillspector_provider_id"}
    assert projected["skillspector_provider_id"] == "nv_build"
    assert skillspector_provenance_records(metadata) == [
        {"provider_id": "nv_build", "provenance": projected["skillspector_llm_provenance"]}
    ]


@pytest.mark.parametrize("provider_id", ["private-gateway", "OPENAI", "", None, 1])
def test_unknown_provider_ids_are_omitted(provider_id: object) -> None:
    metadata = {"skillspector_provider_id": provider_id}

    assert sanitize_skillspector_provenance_metadata(metadata) == {}
    assert skillspector_provenance_records(metadata) == []


def test_provider_record_without_provenance_is_reported() -> None:
    unsafe = _provenance()
    _set_model(unsafe, "sk-" + "a" * 24)
    metadata = {"skillspector_provider_id": "openai", "skillspector_llm_provenance": unsafe}

    assert skillspector_provenance_records(metadata) == [{"provider_id": "openai", "provenance": None}]


def test_provider_ids_match_the_skillspector_bridge() -> None:
    assert frozenset(_SKILLSPECTOR_PROVIDER_MAP) == SKILLSPECTOR_PROVIDER_IDS


def test_fixture_covers_every_llm_analyzer() -> None:
    analyzers = [item["analyzer_id"] for item in _provenance()["analyzers"]]

    assert analyzers == list(SKILLSPECTOR_LLM_ANALYZERS)
