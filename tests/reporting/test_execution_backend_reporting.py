# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Tier 3 report payload names the execution backend a run used."""

from __future__ import annotations

import json

import pytest

from skillevaluator.evaluation import tier3_report
from skillevaluator.models import ValidationResult
from skillevaluator.reporting import HTMLReporter
from skillevaluator.tier3.harbor import report_data
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS


def _agents():
    scores = dict.fromkeys(DEFAULT_METRICS, 1.0)
    baseline = dict.fromkeys(DEFAULT_METRICS, 0.5)
    return {
        "agent": {
            "execution_status": "succeeded",
            "with_skill": scores,
            "without_skill": baseline,
            "rewards": [{**scores, "entry_id": "case", "trial_id": "case__attempt1"}],
            "rewards_baseline": [{**baseline, "entry_id": "case", "trial_id": "case__attempt1"}],
            "num_trials": 1,
            "num_trials_baseline": 1,
        }
    }


def _result(monkeypatch, run_dir, **kwargs):
    monkeypatch.setattr(report_data, "load_agent_data", lambda *_args, **_kwargs: _agents())
    result = tier3_report.agent_eval_result_from_directory(
        run_dir / "demo", run_dir, env_mode="docker", use_llm_judge=False, **kwargs
    )
    assert result is not None
    return result


def _payload(monkeypatch, run_dir, **kwargs):
    return _result(monkeypatch, run_dir, **kwargs).metadata["agent_eval"]


@pytest.mark.parametrize(
    ("result_json", "expected"),
    [
        ({}, "harbor"),
        ({"execution_backend": {"name": "nemo-gym", "version": "0.7.0rc0", "sandbox": "docker"}}, "nemo-gym"),
        ({"execution_backend": {"name": "unknown"}}, "harbor"),
    ],
)
def test_payload_records_the_backend_from_result_json(monkeypatch, tmp_path, result_json, expected):
    (tmp_path / "result.json").write_text(json.dumps(result_json))

    payload = _payload(monkeypatch, tmp_path)

    assert payload["summary"]["execution_backend"] == expected
    assert payload["summary"]["environment"] == "docker"


def test_live_engine_result_and_run_config_name_the_backend(monkeypatch, tmp_path):
    gym = {"execution_backend": {"name": "nemo-gym", "version": "0.7.0rc0", "sandbox": "docker"}}
    assert _payload(monkeypatch, tmp_path, engine_result=gym)["summary"]["execution_backend"] == "nemo-gym"

    (tmp_path / "run_config.json").write_text(json.dumps(gym))
    assert _payload(monkeypatch, tmp_path)["summary"]["execution_backend"] == "nemo-gym"


def test_payload_built_without_a_run_leaves_the_backend_unset():
    payload = tier3_report.build_agent_eval_payload("demo", _agents(), use_llm_judge=False)

    assert payload is not None
    assert payload["summary"]["execution_backend"] is None


@pytest.mark.parametrize(
    ("result_json", "label"),
    [({}, "Harbor"), ({"execution_backend": {"name": "nemo-gym", "sandbox": "docker"}}, "NeMo Gym")],
)
def test_html_report_shows_the_backend_beside_the_environment(monkeypatch, tmp_path, result_json, label):
    (tmp_path / "result.json").write_text(json.dumps(result_json))

    html = HTMLReporter(report_scope="tier3", include_timestamp=False).render_all([_result(monkeypatch, tmp_path)])

    assert f"<div><strong>Backend:</strong> {label}</div>" in html
    assert "<div><strong>Environment:</strong> <code>docker</code></div>" in html


def _dataset_result(backend):
    """A Tier 3 result with a dataset, so the report renders the dataset header too."""
    agent = {
        "execution_status": "succeeded",
        "execution_errors": [],
        "expected_attempts": 1,
        "scored_attempts": 1,
        "with_skill": {"security": 1.0, "goal_accuracy": 1.0},
        "rewards": [{"entry_id": "case", "security": 1.0, "goal_accuracy": 1.0}],
    }
    payload = tier3_report.build_agent_eval_payload(
        "demo",
        {"codex": agent},
        dataset=[{"id": "case", "prompt": "Add 2 and 3."}],
        env_mode="docker",
        execution_backend=backend,
        use_llm_judge=False,
    )
    return tier3_report._validation_result_from_payload(payload)


@pytest.mark.parametrize(("backend", "label"), [("harbor", "Harbor"), ("nemo-gym", "NeMo Gym")])
def test_dataset_header_shows_the_backend_beside_the_environment(backend, label):
    html = HTMLReporter(report_scope="tier3", include_timestamp=False).render_all([_dataset_result(backend)])

    assert f'<span class="t3-ds-chip">Backend: <b>{label}</b></span>' in html
    assert '<span class="t3-ds-chip">Environment: <b>docker</b></span>' in html


def test_html_report_omits_an_unrecorded_backend():
    payload = tier3_report.build_agent_eval_payload("demo", _agents(), use_llm_judge=False)
    result = tier3_report._validation_result_from_payload(payload)

    html = HTMLReporter(report_scope="tier3", include_timestamp=False).render_all([result])

    assert "Backend:" not in html


# The report payload is untyped metadata, so the backend can arrive as anything.
_NOT_A_BACKEND_NAME = [
    pytest.param({"name": "nemo-gym", "sandbox": "docker"}, id="dict"),
    pytest.param(["nemo-gym"], id="list"),
    pytest.param(7, id="int"),
    pytest.param(None, id="none"),
    pytest.param("", id="empty"),
    pytest.param("  ", id="whitespace"),
    pytest.param("some-other-backend", id="unknown-name"),
]


@pytest.mark.parametrize("backend", _NOT_A_BACKEND_NAME)
def test_html_report_omits_a_backend_that_is_not_a_known_name(backend):
    html = HTMLReporter(report_scope="tier3", include_timestamp=False).render_all([_dataset_result(backend)])

    assert "Backend:" not in html
    assert "<div><strong>Environment:</strong> <code>docker</code></div>" in html
    assert '<span class="t3-ds-chip">Environment: <b>docker</b></span>' in html


def _untrusted_result(backend):
    # A success claim with no agent evidence must be rebuilt as incomplete.
    return ValidationResult(
        validator_name="AGENT_EVAL",
        metadata={
            "agent_eval": {
                "skill_name": "demo",
                "execution_status": "succeeded",
                "verdict": "pass",
                "summary": {"environment": "docker", "execution_backend": backend},
                "agents": {},
            }
        },
    )


@pytest.mark.parametrize(("backend", "label"), [("harbor", "Harbor"), ("nemo-gym", "NeMo Gym")])
def test_html_report_keeps_a_recorded_backend_when_rebuilding_untrusted_evidence(backend, label):
    html = HTMLReporter(report_scope="tier3", include_timestamp=False).render_all([_untrusted_result(backend)])

    assert "Tier 3 quality: Incomplete" in html
    assert f"<div><strong>Backend:</strong> {label}</div>" in html
    assert "<div><strong>Environment:</strong> <code>docker</code></div>" in html


@pytest.mark.parametrize("backend", _NOT_A_BACKEND_NAME)
def test_html_report_omits_a_malformed_backend_on_an_untrusted_payload(backend):
    result = _untrusted_result(backend)

    html = HTMLReporter(report_scope="tier3", include_timestamp=False).render_all([result])

    assert "Tier 3 quality: Incomplete" in html
    assert "Backend:" not in html


def test_html_report_still_shows_a_string_backend_next_to_malformed_ones():
    reporter = HTMLReporter(report_scope="tier3", include_timestamp=False)

    html = reporter.render_all([_dataset_result({"name": "nemo-gym"})])
    assert "Backend:" not in html

    html = reporter.render_all([_dataset_result("nemo-gym")])
    assert "<div><strong>Backend:</strong> NeMo Gym</div>" in html
    assert '<span class="t3-ds-chip">Backend: <b>NeMo Gym</b></span>' in html
