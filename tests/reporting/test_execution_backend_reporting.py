# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Tier 3 report payload names the execution backend a run used."""

from __future__ import annotations

import json

import pytest

from skillevaluator.evaluation import tier3_report
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


def _payload(monkeypatch, run_dir, **kwargs):
    monkeypatch.setattr(report_data, "load_agent_data", lambda *_args, **_kwargs: _agents())
    result = tier3_report.agent_eval_result_from_directory(
        run_dir / "demo", run_dir, env_mode="docker", use_llm_judge=False, **kwargs
    )
    assert result is not None
    return result.metadata["agent_eval"]


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
