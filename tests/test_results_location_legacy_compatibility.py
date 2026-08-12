# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility and race contracts for pre-result Tier 3 run directories."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from skillevaluator.tier3 import output_provenance, results_location
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET
from skillevaluator.tier3.output_provenance import GENERATED_OUTPUT_MARKER, mark_generated_output_root


def _run(root: Path, name: str = "20260709_120000") -> Path:
    candidate = root / name
    candidate.mkdir(parents=True)
    return candidate


def _write_legacy_run_config(run: Path, *agents: str) -> None:
    (run / "run_config.json").write_text(
        json.dumps(
            {
                "harbor": {
                    "environment": {"value": "docker", "source": "CLI"},
                    "n_attempts": 1,
                    "stop_on_pass": False,
                    "n_concurrent": 1,
                    "timeout_multiplier": 1.0,
                    "base_image_mode": "auto",
                    "jobs_retained": False,
                },
                "provider": {"name": "nvidia", "model": "test-model"},
                "task_source": "evals_json",
                "grading": {"mode": "default"},
                "agents": {
                    agent: {
                        "agent": agent,
                        "model": "test-model",
                        "source": "test default",
                    }
                    for agent in agents
                },
            }
        ),
        encoding="utf-8",
    )


def _pass_at_k_payload() -> dict[str, object]:
    return {
        "k": 1,
        "pass_threshold": 0.5,
        "stop_on_pass": False,
        "passed_cases": 1,
        "failed_cases": 0,
        "total_cases": 1,
        "rate": 1.0,
        "attempts_used": 1,
        "max_attempts_possible": 1,
        "avg_attempts_used": 1.0,
        "extra_cases": [],
        "cases": {
            "case": {
                "passed": True,
                "first_pass_attempt": 1,
                "attempts_used": 1,
                "attempts_skipped": 0,
                "attempts_missing": 0,
                "best_score": 0.8,
                "attempts": [{"attempt": 1, "trial": "case__1", "score": 0.8, "passed": True}],
            }
        },
    }


def _successful_summary_payload(agent: str = "opencode") -> dict[str, object]:
    return {
        "agent": agent,
        "model": "test-model",
        "model_source": "test default",
        "scores": {"security": 0.8},
        "custom_scores": {},
        "metric_set": DEFAULT_METRIC_SET,
        "metrics": ["security"],
        "dimensions": {"safety": {"score": 0.8, "sources": {"security": 1.0}}},
        "num_trials": 1,
        "pass_at_k": _pass_at_k_payload(),
        "execution_status": "succeeded",
        "execution_errors": [],
        "expected_attempts": 1,
        "scored_attempts": 1,
        "job_failure": "",
        "trial_failures": [],
    }


def _write_successful_summary(run: Path, *, agent: str = "opencode") -> Path:
    summary = run / agent / "with-skill" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps(_successful_summary_payload(agent)), encoding="utf-8")
    return summary


def _write_current_run(root: Path, name: str = "20260709_120000_1_aaaaaaaaaaaa") -> Path:
    run = _run(root, name)
    mark_generated_output_root(run)
    run_config: dict[str, object] = {
        "config_file": "none",
        "harbor": {
            "environment": {"value": "local", "source": "test"},
            "n_attempts": 1,
            "stop_on_pass": False,
            "n_concurrent": 1,
            "timeout_multiplier": 1.0,
            "base_image_mode": "disabled",
            "jobs_retained": False,
        },
        "provider": {"name": "nvidia", "model": "test-model"},
        "task_source": "evals_json",
        "grading": {"mode": "default"},
        "agents": {"opencode": {"agent": "opencode", "model": "test-model", "source": "test default"}},
    }
    agent_result = {
        "model": "test-model",
        "model_source": "test default",
        "model_resolution": {"model": "test-model", "source": "test default"},
        "with_skill": {"security": 0.8},
        "without_skill": {},
        "custom_with_skill": {},
        "custom_without_skill": {},
        "dimensions_with_skill": {},
        "dimensions_without_skill": {},
        "lift": {},
        "custom_lift": {},
        "pass_at_k": {"with_skill": {}, "without_skill": {}, "lift": {}},
        "security_attribution": {},
        "agent_runtime_failures": {"with_skill": [], "without_skill": []},
        "trial_failures": {"with_skill": [], "without_skill": []},
        "job_failures": {"with_skill": "", "without_skill": ""},
        "conditions": {"with_skill": {}, "without_skill": {}},
        "execution_status": "succeeded",
        "execution_errors": [],
        "expected_attempts": 1,
        "scored_attempts": 1,
        "num_trials_with": 1,
        "num_trials_without": 0,
        "output_dir": str((run / "opencode").resolve()),
    }
    (run / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    (run / "result.json").write_text(
        json.dumps(
            {
                "skill_name": "demo",
                "run_id": name,
                "run_dir": str(run),
                "result_path": str((run / "result.json").resolve()),
                "run_config": run_config,
                "agents": {"opencode": agent_result},
                "attempt_policy": {
                    "max_attempts": 1,
                    "pass_threshold": 0.5,
                    "stop_on_pass": False,
                    "score_definition": "test",
                },
                "execution_status": "succeeded",
                "execution_errors": [],
                "report_status": "complete",
                "duration_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    return run


def _is_completed(run: Path) -> bool:
    return results_location.run_directory_sort_key(run, require_completed_result=True) is not None


def test_current_run_discovery_accepts_windows_crt_descriptor_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows path and CRT descriptor stat identities are not comparable."""
    run = _write_current_run(tmp_path)
    original_read_bytes = results_location.SecureRoot.read_bytes

    def read_with_windows_crt_identity(
        secure_root: results_location.SecureRoot,
        relative_path: Path,
        max_bytes: int,
        *,
        expected: os.stat_result | None = None,
    ) -> tuple[bytes, object]:
        raw, opened = original_read_bytes(secure_root, relative_path, max_bytes, expected=expected)
        return raw, SimpleNamespace(
            st_dev=opened.st_dev + 10_000,
            st_ino=opened.st_ino + 10_000,
            st_mode=opened.st_mode,
            st_nlink=opened.st_nlink,
            st_size=opened.st_size,
            st_uid=opened.st_uid,
            st_mtime_ns=opened.st_mtime_ns,
            st_ctime_ns=opened.st_ctime_ns,
        )

    monkeypatch.setattr(results_location, "_PATH_DESCRIPTOR_IDENTITIES_COMPARABLE", False, raising=False)
    monkeypatch.setattr(output_provenance, "_PATH_DESCRIPTOR_IDENTITIES_COMPARABLE", False)
    monkeypatch.setattr(results_location.SecureRoot, "read_bytes", read_with_windows_crt_identity)

    assert _is_completed(run)


def test_rejects_current_run_with_list_valued_run_config(tmp_path: Path) -> None:
    run = _write_current_run(tmp_path)
    (run / "run_config.json").write_text("[]", encoding="utf-8")

    assert not _is_completed(run)


def test_accepts_pre_result_run_with_public_runner_config_shape(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    _write_successful_summary(run)

    assert _is_completed(run)


@pytest.mark.parametrize("sidecar_name", ["harbor-run-logs", "astra-cleanup"])
def test_accepts_opaque_pre_result_runtime_sidecar(tmp_path: Path, sidecar_name: str) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    _write_successful_summary(run)
    sidecar = run / sidecar_name
    sidecar.mkdir()
    (sidecar / "unexpected.log").write_text("opaque runtime data", encoding="utf-8")
    nested = sidecar / "unexpected-directory"
    nested.mkdir()
    (nested / "payload.bin").write_bytes(b"opaque")

    assert _is_completed(run)


@pytest.mark.parametrize("sidecar_name", ["harbor-run-logs", "astra-cleanup"])
def test_accepts_link_inside_opaque_pre_result_runtime_sidecar(tmp_path: Path, sidecar_name: str) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    _write_successful_summary(run)
    sidecar = run / sidecar_name
    sidecar.mkdir()
    outside = tmp_path / f"outside-{sidecar_name}.log"
    outside.write_text("opaque runtime data", encoding="utf-8")
    try:
        (sidecar / "unexpected-link").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    assert _is_completed(run)


@pytest.mark.parametrize("sibling_name", ["staged", "unknown-runtime"])
def test_rejects_unknown_pre_result_sibling_directory(tmp_path: Path, sibling_name: str) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    _write_successful_summary(run)
    (run / sibling_name).mkdir()

    assert not _is_completed(run)


def test_accepts_pre_result_run_with_valid_optional_occurrence(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    config_path = run / "run_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["agents"]["opencode"]["occurrence"] = "2"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_successful_summary(run)

    assert _is_completed(run)


@pytest.mark.parametrize(
    "occurrence",
    [
        None,
        "",
        "0",
        "not-a-number",
        "\N{SUPERSCRIPT TWO}",
        "\N{FULLWIDTH DIGIT ONE}",
        "\N{ARABIC-INDIC DIGIT TWO}",
        1,
    ],
)
def test_rejects_pre_result_run_with_invalid_optional_occurrence(
    tmp_path: Path,
    occurrence: object,
) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    config_path = run / "run_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["agents"]["opencode"]["occurrence"] = occurrence
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_successful_summary(run)

    assert not _is_completed(run)


def test_rejects_oversized_ascii_occurrence_without_raising(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    config_path = run / "run_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["agents"]["opencode"]["occurrence"] = "9" * 5000

    assert results_location._legacy_run_agents(payload) is None

    config_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_successful_summary(run)
    assert not _is_completed(run)


def test_accepts_summary_from_before_truthful_status_fields(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    summary = run / "opencode" / "with-skill" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "agent": "opencode",
                "model": "test-model",
                "model_source": "test default",
                "scores": {"security": 0.8},
                "custom_scores": {},
                "metric_set": "aces-default-v2",
                "metrics": ["security"],
                "dimensions": {"safety": {"score": 0.8, "sources": {"security": 1.0}}},
                "num_trials": 1,
                "pass_at_k": _pass_at_k_payload(),
            }
        ),
        encoding="utf-8",
    )

    assert _is_completed(run)


@pytest.mark.parametrize(
    "run_config_payload",
    [
        "not-json",
        "[]",
        "null",
        "1",
        "{}",
        '{"agents": {}}',
        json.dumps(
            {
                "harbor": {"n_attempts": {"value": 1, "source": "ACES default"}},
                "task_source": "generated",
                "agents": {
                    "opencode": {
                        "agent": "opencode",
                        "model": "test-model",
                        "source": "test default",
                        "occurrence": "1",
                    }
                },
            }
        ),
        json.dumps(
            {
                "harbor": {"n_attempts": {"value": 1, "source": "ACES default"}},
                "task_source": "evals_json",
                "agents": {"opencode": {}},
            }
        ),
    ],
)
def test_rejects_legacy_run_with_invalid_run_config(tmp_path: Path, run_config_payload: str) -> None:
    run = _run(tmp_path)
    (run / "run_config.json").write_text(run_config_payload, encoding="utf-8")
    _write_successful_summary(run)

    assert not _is_completed(run)


def test_requires_summary_for_every_configured_agent(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode", "claude")
    _write_successful_summary(run, agent="opencode")

    assert not _is_completed(run)


def test_rejects_malformed_present_baseline_summary(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    _write_successful_summary(run)
    baseline = run / "opencode" / "without-skill" / "summary.json"
    baseline.parent.mkdir()
    baseline.write_text("not-json", encoding="utf-8")

    assert not _is_completed(run)


def test_rejects_ambiguous_flat_and_nested_summaries(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    _write_successful_summary(run)
    (run / "opencode" / "summary.json").write_text(
        json.dumps(_successful_summary_payload()),
        encoding="utf-8",
    )

    assert not _is_completed(run)


def test_accepts_authentic_pre_status_custom_only_summary(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    payload = _successful_summary_payload()
    for field in (
        "execution_status",
        "execution_errors",
        "expected_attempts",
        "scored_attempts",
        "job_failure",
        "trial_failures",
    ):
        payload.pop(field)
    payload.update(
        {
            "scores": {},
            "custom_scores": {},
            "metric_set": "custom-only",
            "metrics": [],
            "dimensions": {},
        }
    )
    summary = run / "opencode" / "with-skill" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps(payload), encoding="utf-8")

    assert _is_completed(run)


def test_accepts_truthful_status_era_failed_summary(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    payload = _successful_summary_payload()
    payload.update(
        {
            "scores": {},
            "dimensions": {},
            "num_trials": 0,
            "pass_at_k": {
                "k": 1,
                "pass_threshold": 0.5,
                "stop_on_pass": False,
                "passed_cases": 0,
                "failed_cases": 1,
                "total_cases": 1,
                "rate": 0.0,
                "attempts_used": 0,
                "max_attempts_possible": 1,
                "avg_attempts_used": 0.0,
                "extra_cases": [],
                "cases": {
                    "case": {
                        "passed": False,
                        "first_pass_attempt": None,
                        "attempts_used": 0,
                        "attempts_skipped": 0,
                        "attempts_missing": 1,
                        "best_score": None,
                        "attempts": [],
                    }
                },
            },
            "execution_status": "failed",
            "execution_errors": ["Harbor job directory was not created"],
            "expected_attempts": 1,
            "scored_attempts": 0,
            "job_failure": "Harbor job directory was not created",
            "trial_failures": [],
        }
    )
    summary = run / "opencode" / "with-skill" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps(payload), encoding="utf-8")

    assert _is_completed(run)


def test_rejects_status_summary_with_incoherent_scored_attempts(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    payload = _successful_summary_payload()
    payload.update(
        {
            "execution_status": "failed",
            "execution_errors": ["Harbor reported a failure"],
            "scored_attempts": 0,
            "job_failure": "Harbor reported a failure",
        }
    )
    summary = _write_successful_summary(run)
    summary.write_text(json.dumps(payload), encoding="utf-8")

    assert not _is_completed(run)


def test_accepts_truthful_failed_excess_attempt_coverage(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    payload = _successful_summary_payload()
    pass_at_k = _pass_at_k_payload()
    attempts = [
        {"attempt": 1, "trial": "case__attempt1", "score": 0.8, "passed": True},
        {"attempt": 2, "trial": "case__attempt2", "score": 0.7, "passed": True},
    ]
    pass_at_k.update({"attempts_used": 2, "avg_attempts_used": 2.0})
    pass_at_k["cases"] = {
        "case": {
            "passed": True,
            "first_pass_attempt": 1,
            "attempts_used": 2,
            "attempts_skipped": 0,
            "attempts_missing": 0,
            "best_score": 0.8,
            "attempts": attempts,
        }
    }
    payload.update(
        {
            "num_trials": 2,
            "pass_at_k": pass_at_k,
            "execution_status": "failed",
            "execution_errors": ["Excess scored attempts for cases: case", "Scored attempt coverage is 2/1"],
            "expected_attempts": 1,
            "scored_attempts": 2,
            "job_failure": "",
        }
    )
    summary = _write_successful_summary(run)
    summary.write_text(json.dumps(payload), encoding="utf-8")

    assert _is_completed(run)


def test_rejects_legacy_summary_without_authentic_pass_at_k(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    payload = _successful_summary_payload()
    payload["pass_at_k"] = {}
    summary = run / "opencode" / "with-skill" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps(payload), encoding="utf-8")

    assert not _is_completed(run)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("passed_cases", 0),
        ("attempts_used", 0),
        ("rate", 0.0),
    ],
)
def test_rejects_incoherent_legacy_pass_at_k_totals(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    payload = _successful_summary_payload()
    payload["pass_at_k"] = {**_pass_at_k_payload(), field: value}
    summary = _write_successful_summary(run)
    summary.write_text(json.dumps(payload), encoding="utf-8")

    assert not _is_completed(run)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("passed", False),
        ("first_pass_attempt", None),
        ("best_score", 0.7),
        ("attempts_missing", 1),
    ],
)
def test_rejects_incoherent_legacy_pass_at_k_case(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    payload = _successful_summary_payload()
    pass_at_k = _pass_at_k_payload()
    pass_at_k["cases"] = {
        "case": {
            **pass_at_k["cases"]["case"],
            field: value,
        }
    }
    payload["pass_at_k"] = pass_at_k
    summary = _write_successful_summary(run)
    summary.write_text(json.dumps(payload), encoding="utf-8")

    assert not _is_completed(run)


def test_rejects_legacy_summary_for_wrong_agent(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    summary = _write_successful_summary(run)
    summary.write_text(json.dumps(_successful_summary_payload("claude")), encoding="utf-8")

    assert not _is_completed(run)


def test_rejects_legacy_run_with_malformed_sibling_summary(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode", "claude")
    _write_successful_summary(run)
    malformed = run / "claude" / "with-skill" / "summary.json"
    malformed.parent.mkdir(parents=True)
    malformed.write_text("not-json", encoding="utf-8")

    assert not _is_completed(run)


@pytest.mark.parametrize(
    "summary_payload",
    [
        {**_successful_summary_payload(), "scores": {"security": 0.8, "malformed": "not-a-number"}},
        {**_successful_summary_payload(), "scores": {"security": 0.8, "non_finite": float("nan")}},
        {**_successful_summary_payload(), "scores": {"security": 0.8, "overflowing": 10**1000}},
        {**_successful_summary_payload(), "custom_scores": {"quality": 0.6, "malformed": None}},
        {key: value for key, value in _successful_summary_payload().items() if key != "scores"},
        {key: value for key, value in _successful_summary_payload().items() if key != "custom_scores"},
        {key: value for key, value in _successful_summary_payload().items() if key != "model"},
        {key: value for key, value in _successful_summary_payload().items() if key != "model_source"},
        {key: value for key, value in _successful_summary_payload().items() if key != "metric_set"},
        {key: value for key, value in _successful_summary_payload().items() if key != "metrics"},
        {key: value for key, value in _successful_summary_payload().items() if key != "dimensions"},
        {key: value for key, value in _successful_summary_payload().items() if key != "pass_at_k"},
        {key: value for key, value in _successful_summary_payload().items() if key != "job_failure"},
        {key: value for key, value in _successful_summary_payload().items() if key != "trial_failures"},
        {**_successful_summary_payload(), "scores": []},
    ],
)
def test_rejects_legacy_summary_with_incomplete_score_maps(
    tmp_path: Path,
    summary_payload: dict[str, object],
) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    summary = _write_successful_summary(run)
    summary.write_text(json.dumps(summary_payload), encoding="utf-8")

    assert not _is_completed(run)


@pytest.mark.parametrize(
    "optional_fields",
    [
        {"execution_status": "succeeded"},
        {"execution_errors": []},
        {"execution_status": "succeeded", "execution_errors": []},
        {"expected_attempts": 1, "scored_attempts": 1},
        {"execution_status": "succeeded", "execution_errors": [], "expected_attempts": 1},
    ],
)
def test_rejects_legacy_summary_with_partial_completion_fields(
    tmp_path: Path,
    optional_fields: dict[str, object],
) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    payload = _successful_summary_payload()
    for field in (
        "execution_status",
        "execution_errors",
        "expected_attempts",
        "scored_attempts",
        "job_failure",
        "trial_failures",
    ):
        payload.pop(field)
    payload.update(optional_fields)
    summary = _write_successful_summary(run)
    summary.write_text(json.dumps(payload), encoding="utf-8")

    assert not _is_completed(run)


@pytest.mark.parametrize(
    "summary_payload",
    [
        "not-json",
        json.dumps([]),
        json.dumps(
            {
                "agent": "opencode",
                "scores": {"security": 0.8},
                "custom_scores": {},
                "num_trials": 1,
                "execution_status": "failed",
                "execution_errors": ["trial failed"],
                "expected_attempts": 1,
                "scored_attempts": 0,
            }
        ),
        json.dumps(
            {
                "agent": "opencode",
                "scores": {"security": 0.8},
                "custom_scores": {},
                "num_trials": 1,
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 2,
                "scored_attempts": 1,
            }
        ),
    ],
)
def test_rejects_legacy_run_without_valid_completion_summary(tmp_path: Path, summary_payload: str) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    summary = run / "opencode" / "with-skill" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(summary_payload, encoding="utf-8")

    assert not _is_completed(run)


def test_rejects_authenticated_current_partial_with_legacy_summary(tmp_path: Path) -> None:
    run = _run(tmp_path)
    mark_generated_output_root(run)
    _write_legacy_run_config(run, "opencode")
    _write_successful_summary(run)

    assert not _is_completed(run)


def test_rejects_invalid_current_marker_downgrade_to_legacy(tmp_path: Path) -> None:
    run = _run(tmp_path)
    (run / GENERATED_OUTPUT_MARKER).write_text("tampered", encoding="utf-8")
    _write_legacy_run_config(run, "opencode")
    _write_successful_summary(run)

    assert not _is_completed(run)


def test_rejects_legacy_run_config_replaced_before_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = _run(tmp_path)
    run_config = run / "run_config.json"
    _write_legacy_run_config(run, "opencode")
    run_config_payload = run_config.read_text(encoding="utf-8")
    _write_successful_summary(run)
    real_read_bytes = results_location.SecureRoot.read_bytes
    replaced = False

    def replace_before_open(
        secure_root: results_location.SecureRoot,
        relative_path: Path,
        max_bytes: int,
        *,
        expected: os.stat_result | None = None,
    ) -> tuple[bytes, os.stat_result]:
        nonlocal replaced
        if not replaced and relative_path == Path("run_config.json"):
            replacement = run_config.with_suffix(".replacement")
            replacement.write_text(run_config_payload, encoding="utf-8")
            replacement.replace(run_config)
            replaced = True
        return real_read_bytes(secure_root, relative_path, max_bytes, expected=expected)

    monkeypatch.setattr(results_location.SecureRoot, "read_bytes", replace_before_open)

    assert not _is_completed(run)


def test_rejects_same_inode_summary_rewrite_with_restored_mtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    summary = _write_successful_summary(run)
    summary_metadata = summary.stat()
    summary_inode = summary_metadata.st_ino
    original = summary.read_text(encoding="utf-8")
    rewritten = original.replace("0.8", "0.9")
    assert len(rewritten) == len(original)
    real_read = os.read
    rewritten_once = False

    def rewrite_during_read(fd: int, length: int) -> bytes:
        nonlocal rewritten_once
        if not rewritten_once and os.fstat(fd).st_ino == summary_inode:
            summary.write_text(rewritten, encoding="utf-8")
            os.utime(summary, ns=(summary_metadata.st_atime_ns, summary_metadata.st_mtime_ns))
            rewritten_once = True
        return real_read(fd, length)

    monkeypatch.setattr(results_location.os, "read", rewrite_during_read)

    assert not _is_completed(run)


def test_rejects_summary_parent_swapped_to_link_before_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = _run(tmp_path)
    _write_legacy_run_config(run, "opencode")
    summary = _write_successful_summary(run)
    condition_dir = summary.parent
    moved = run / "opencode" / "moved-with-skill"
    real_read_bytes = results_location.SecureRoot.read_bytes
    swapped = False

    def swap_summary_parent(
        secure_root: results_location.SecureRoot,
        relative_path: Path,
        max_bytes: int,
        *,
        expected: os.stat_result | None = None,
    ) -> tuple[bytes, os.stat_result]:
        nonlocal swapped
        if not swapped and relative_path == Path("opencode/with-skill/summary.json"):
            swapped = True
            condition_dir.rename(moved)
            condition_dir.symlink_to(moved, target_is_directory=True)
        return real_read_bytes(secure_root, relative_path, max_bytes, expected=expected)

    monkeypatch.setattr(results_location.SecureRoot, "read_bytes", swap_summary_parent)

    assert not _is_completed(run)


def test_rejects_current_result_replaced_during_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = _write_current_run(tmp_path)
    result_path = run / "result.json"
    result_inode = result_path.stat().st_ino
    result_payload = result_path.read_text(encoding="utf-8")
    real_read = os.read
    replaced = False

    def replace_after_read(fd: int, length: int) -> bytes:
        nonlocal replaced
        content = real_read(fd, length)
        if not replaced and os.fstat(fd).st_ino == result_inode:
            replacement = result_path.with_suffix(".replacement")
            replacement.write_text(result_payload, encoding="utf-8")
            replacement.replace(result_path)
            replaced = True
        return content

    monkeypatch.setattr(results_location.os, "read", replace_after_read)

    assert not _is_completed(run)


def test_rejects_run_config_changed_during_result_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = _write_current_run(tmp_path)
    result_path = run / "result.json"
    result_inode = result_path.stat().st_ino
    run_config = run / "run_config.json"
    config_metadata = run_config.stat()
    real_read = os.read
    changed = False

    def change_config_after_result_read(fd: int, length: int) -> bytes:
        nonlocal changed
        content = real_read(fd, length)
        if not changed and os.fstat(fd).st_ino == result_inode:
            run_config.write_text("[]", encoding="utf-8")
            os.utime(run_config, ns=(config_metadata.st_atime_ns, config_metadata.st_mtime_ns))
            changed = True
        return content

    monkeypatch.setattr(results_location.os, "read", change_config_after_result_read)

    assert not _is_completed(run)


@pytest.mark.parametrize("newest_first", [False, True], ids=("oldest-first", "newest-first"))
def test_rejects_candidate_tree_swap_between_artifact_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    newest_first: bool,
) -> None:
    completed = _write_current_run(tmp_path, "20260709_110000_111_aaaaaaaaaaaa")
    unstable = _write_current_run(tmp_path, "20260709_120000_222_bbbbbbbbbbbb")
    moved = tmp_path / "moved-run"
    real_iterdir = results_location.Path.iterdir
    real_read_bytes = results_location.SecureRoot.read_bytes
    swapped = False

    candidates = (unstable, completed) if newest_first else (completed, unstable)

    def stable_candidate_order(path: Path) -> Iterator[Path]:
        if path == tmp_path:
            return iter(candidates)
        return real_iterdir(path)

    def swap_candidate(
        secure_root: results_location.SecureRoot,
        relative_path: Path,
        max_bytes: int,
        *,
        expected: os.stat_result | None = None,
    ) -> tuple[bytes, os.stat_result]:
        nonlocal swapped
        if not swapped and secure_root.root == unstable.absolute() and relative_path == Path("result.json"):
            swapped = True
            unstable.rename(moved)
            _write_current_run(tmp_path, unstable.name)
        return real_read_bytes(secure_root, relative_path, max_bytes, expected=expected)

    monkeypatch.setattr(results_location.Path, "iterdir", stable_candidate_order)
    monkeypatch.setattr(results_location.SecureRoot, "read_bytes", swap_candidate)

    assert results_location._newest_run_dir(tmp_path) == completed
    assert swapped
