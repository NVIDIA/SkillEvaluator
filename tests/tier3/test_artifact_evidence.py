# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original/delivered evidence exposes bytes, never a semantic acceptance verdict."""
import asyncio
import json
import logging

import pytest

from skillevaluator.tier3 import artifact_evidence as evidence
from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks

PAIR = {"id": "tests", "original": "test original.py", "delivered": "/workspace/output/project/test.py"}


def staged(tmp_path, original=b"def test_value():\n    assert value() == 3\n"):
    task = tmp_path / "task"
    (task / "environment/input").mkdir(parents=True)
    (task / "environment/input/test original.py").write_bytes(original)
    (task / "task.toml").write_text('[environment]\nos = "linux"\n')
    evidence.stage_artifact_pairs(task, {"id": "case", "artifact_pairs": [PAIR]})
    trial = tmp_path / "native"
    delivered = trial / "artifacts/skillevaluator-evidence/tests/delivered"
    delivered.parent.mkdir(parents=True)
    delivered.write_bytes(original)
    manifest = [{"source": PAIR["delivered"], "destination": "artifacts/skillevaluator-evidence/tests/delivered",
                 "status": "ok", "type": "file", "service": None, "exclude": []}]
    (trial / "artifacts/manifest.json").write_text(json.dumps(manifest))
    return task, trial, delivered


def collect(tmp_path, task, trial, trial_id="attempt-1"):
    out = tmp_path / "run/agent/with-skill/trials" / trial_id
    out.mkdir(parents=True, exist_ok=True)
    evidence.collect_artifact_evidence(task, trial, out, entry_id="case", trial_id=trial_id,
                                       agent="agent", variant="with_skill")
    return json.loads((out / "artifact_evidence.json").read_text())


def test_adapter_declares_capture_and_freezes_actual_original(tmp_path):
    skill = tmp_path / "skill"
    (skill / "evals/files").mkdir(parents=True)
    (skill / "SKILL.md").write_text("# skill")
    (skill / "evals/files/test original.py").write_text("assert True\n")
    (skill / "evals/evals.json").write_text(json.dumps([{"id": "case", "question": "work",
        "artifact_pairs": [PAIR]}]))
    task, = generate_harbor_tasks(skill, tmp_path / "tasks")
    import tomllib
    config = tomllib.loads((task / "task.toml").read_text())
    assert config["artifacts"] == [{"source": PAIR["delivered"],
                                    "destination": "skillevaluator-evidence/tests/delivered"}]
    assert json.loads((task / "artifact_pairs.json").read_text())["pairs"][0]["original_sha256"]


@pytest.mark.parametrize("delivered,equal", [
    (b"def test_value():\n    assert value() == 3\n", True),
    (b"def test_value():\n    assert value() == 3\n\ndef test_extra():\n    assert True\n", False),
    (b"def test_value():\n    assert True\n", False),
])
def test_actual_original_and_delivered_expose_same_name_weakening(tmp_path, delivered, equal):
    task, trial, path = staged(tmp_path)
    path.write_bytes(delivered)
    pair = collect(tmp_path, task, trial)["pairs"][0]
    assert pair["status"] == "complete"
    assert pair["bytes_equal"] is equal
    assert pair["original"]["text"] == "def test_value():\n    assert value() == 3\n"
    assert pair["delivered"]["text"] == delivered.decode()
    assert "semantic_equal" not in pair and "tests_preserved" not in pair


@pytest.mark.parametrize("delivered", [b'{"value": 1}\n', b'{"value": 2}', b"{broken"])
def test_json_bytes_are_not_repaired_or_semantically_scored(tmp_path, delivered):
    task, trial, path = staged(tmp_path, b'{"value": 1}')
    path.write_bytes(delivered)
    pair = collect(tmp_path, task, trial)["pairs"][0]
    assert pair["bytes_equal"] is False
    assert pair["delivered"]["text"] == delivered.decode()


@pytest.mark.parametrize("change", [
    {"original": "../secret"}, {"original": "/secret"}, {"original": "a//b"},
    {"delivered": "/workspace/output/../secret"}, {"delivered": "/tmp/x"},
    {"id": "../x"}, {"extra": "unknown"},
])
def test_invalid_declarations_fail_before_capture(change):
    with pytest.raises(ValueError):
        evidence.validate_artifact_pairs({"artifact_pairs": [{**PAIR, **change}]})


def test_duplicate_declarations_and_missing_original_fail(tmp_path):
    with pytest.raises(ValueError):
        evidence.validate_artifact_pairs({"artifact_pairs": [PAIR, PAIR]})
    (tmp_path / "environment/input").mkdir(parents=True)
    with pytest.raises(ValueError):
        evidence.stage_artifact_pairs(tmp_path, {"id": "case", "artifact_pairs": [PAIR]})


@pytest.mark.parametrize("problem", ["missing", "duplicate", "symlink", "ancestor", "oversized", "binary", "changed_original"])
def test_unavailable_inputs_do_not_become_complete(tmp_path, problem):
    task, trial, path = staged(tmp_path)
    if problem == "missing":
        path.unlink()
    elif problem == "duplicate":
        manifest = trial / "artifacts/manifest.json"
        manifest.write_text(json.dumps(json.loads(manifest.read_text()) * 2))
    elif problem == "symlink":
        path.unlink()
        path.symlink_to(task / "environment/input/test original.py")
    elif problem == "ancestor":
        path.unlink()
        path.parent.rmdir()
        path.parent.symlink_to(task / "environment/input", target_is_directory=True)
    elif problem == "oversized":
        path.write_bytes(b"x" * (evidence.MAX_FILE_BYTES + 1))
    elif problem == "binary":
        path.write_bytes(b"\xff\x00")
    else:
        (task / "environment/input/test original.py").write_text("changed")
    pair = collect(tmp_path, task, trial)["pairs"][0]
    assert pair["status"] != "complete"
    assert "bytes_equal" not in pair


def test_redaction_precedes_clipping_and_raw_comparison(tmp_path, monkeypatch):
    secret = "unique-private-api-key-abcdefghijklmnopqrstuvwxyz"
    monkeypatch.setenv("NVIDIA_API_KEY", secret)
    task, trial, path = staged(tmp_path, ("a" * (evidence.MAX_TEXT_CHARS - 10) + secret).encode())
    path.write_text("a" * (evidence.MAX_TEXT_CHARS - 10) + "another-value")
    record = collect(tmp_path, task, trial)
    assert secret[:15] not in json.dumps(record)
    pair = record["pairs"][0]
    assert pair["bytes_equal"] is False
    assert pair["original"]["redacted"] is True


def test_consumer_binds_repeated_attempt_identity_and_declaration(tmp_path):
    task, trial, _ = staged(tmp_path)
    collect(tmp_path, task, trial, "attempt-1")
    collect(tmp_path, task, trial, "attempt-2")
    root = tmp_path / "run"
    def read(trial_id="attempt-1", case="case", declarations=None):
        return evidence.load_artifact_evidence(root, "agent", trial_id, entry_id=case,
                                              declarations=[PAIR] if declarations is None else declarations)
    assert read()["pairs"][0]["bytes_equal"] is True
    assert read("attempt-2")["trial_id"] == "attempt-2"
    assert read(case="other")["status"] == "unavailable"
    assert read(declarations=[{**PAIR, "original": "other.py"}])["status"] == "unavailable"
    assert read("../attempt-1")["status"] == "unavailable"


@pytest.mark.parametrize("invalid", [[], {"pairs": [None]}])
def test_consumer_rejects_malformed_record_without_crashing(tmp_path, invalid):
    task, trial, _ = staged(tmp_path)
    record = collect(tmp_path, task, trial)
    if isinstance(invalid, dict):
        record.update(invalid)
    else:
        record = invalid
    path = tmp_path / "run/agent/with-skill/trials/attempt-1/artifact_evidence.json"
    path.write_text(json.dumps(record))
    assert evidence.load_artifact_evidence(tmp_path / "run", "agent", "attempt-1",
        entry_id="case", declarations=[PAIR])["status"] == "unavailable"


def test_consumer_refreshes_redaction_metadata_for_newly_configured_secret(tmp_path, monkeypatch):
    secret = "previously-unconfigured-private-credential-abcdef"
    task, trial, _ = staged(tmp_path, ("value=" + secret).encode())
    record = collect(tmp_path, task, trial)
    assert record["pairs"][0]["original"]["redacted"] is False
    monkeypatch.setenv("NVIDIA_API_KEY", secret)
    loaded = evidence.load_artifact_evidence(tmp_path / "run", "agent", "attempt-1",
        entry_id="case", declarations=[PAIR])
    assert secret not in json.dumps(loaded)
    for side in ("original", "delivered"):
        value = loaded["pairs"][0][side]
        assert value["redacted"] is True
        assert value["text_sha256"] == evidence._digest(value["text"].encode())
        assert value["sha256"] == record["pairs"][0][side]["sha256"]


@pytest.mark.parametrize("manifest", [None, "a" * 64, "opaqueConfiguredCredentialOnly", False])
def test_partial_pair_validates_present_manifest_hash(tmp_path, monkeypatch, manifest):
    monkeypatch.setenv("NVIDIA_API_KEY", "opaqueConfiguredCredentialOnly")
    task, trial, _ = staged(tmp_path)
    record = collect(tmp_path, task, trial)
    record.update(status="partial", pairs=[{"id": PAIR["id"], "status": "delivered_not_collected"}])
    record.pop("native_manifest_sha256")
    if manifest is not None:
        record["native_manifest_sha256"] = manifest
    (tmp_path / "run/agent/with-skill/trials/attempt-1/artifact_evidence.json").write_text(json.dumps(record))
    loaded = evidence.load_artifact_evidence(tmp_path / "run", "agent", "attempt-1",
        entry_id="case", declarations=[PAIR])
    assert loaded["status"] == ("partial" if manifest in (None, "a" * 64) else "unavailable")
    assert "opaqueConfiguredCredentialOnly" not in json.dumps(loaded)


@pytest.mark.parametrize("problem", ["false_equality", "invalid_hash", "oversized_diff", "text_raw_mismatch", "text_size_mismatch", "equal_size_mismatch"])
def test_consumer_rejects_inconsistent_or_unbounded_pair_claims(tmp_path, problem):
    task, trial, _ = staged(tmp_path)
    record = collect(tmp_path, task, trial)
    pair = record["pairs"][0]
    if problem == "false_equality":
        pair["delivered"]["sha256"] = "a" * 64
    elif problem == "invalid_hash":
        pair["original"]["sha256"] = "not a digest"
    elif problem == "text_raw_mismatch":
        pair["original"]["sha256"] = pair["delivered"]["sha256"] = "a" * 64
    elif problem == "text_size_mismatch":
        pair["original"]["size_bytes"] += 1
    elif problem == "equal_size_mismatch":
        pair["original"]["truncated"] = pair["delivered"]["truncated"] = True
        pair["original"]["size_bytes"] += 1
    else:
        pair["diff"] = "a" * (evidence.MAX_TEXT_CHARS + 1)
    path = tmp_path / "run/agent/with-skill/trials/attempt-1/artifact_evidence.json"
    path.write_text(json.dumps(record))
    assert evidence.load_artifact_evidence(tmp_path / "run", "agent", "attempt-1",
        entry_id="case", declarations=[PAIR])["status"] == "unavailable"


def test_directory_cannot_bypass_reserved_safe_file_capture(tmp_path, monkeypatch):
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    from skillevaluator.tier3.harbor.secure_docker_environment import SkillEvaluatorSecureDockerEnvironment
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text("FROM scratch\n")
    env = SkillEvaluatorSecureDockerEnvironment(
        environment_dir=environment_dir, environment_name="pair-test", session_id="pair-test",
        trial_paths=TrialPaths(tmp_path / "trial"), task_env_config=EnvironmentConfig(),
        logger=logging.getLogger("pair-test"),
    )
    async def unexpected(*args, **kwargs):
        pytest.fail("Unsafe generic directory download was dispatched")
    monkeypatch.setattr(env, "_run_docker_compose_command", unexpected)
    target = tmp_path / "artifacts/skillevaluator-evidence/tests/delivered"
    with pytest.raises(RuntimeError, match="regular file"):
        asyncio.run(env.service_download_dir("/workspace/output/linked-directory", target))


@pytest.mark.parametrize("backend", ["local", "gke", "kata", "mosaic"])
def test_unsupported_backend_rejected_before_dispatch(tmp_path, backend):
    from skillevaluator.tier3.harbor.runner import _staged_task_environment_error
    task, _, _ = staged(tmp_path)
    assert "unsupported" in _staged_task_environment_error(backend, [task])
    assert _staged_task_environment_error("docker", [task]) is None


@pytest.mark.parametrize("split", ["discovery", "selection", "holdout", "heldout", "held-out", "test", "validation", "val"])
def test_public_consumers_include_only_discovery_with_skill_pairs(tmp_path, split):
    from skillevaluator.tier3.generate_dataset import _discover_trajectories
    from skillevaluator.tier3.harbor.report_data import build_dataset_snapshot
    from skillevaluator.tier3.improvement_signals import load_signals

    task, trial, _ = staged(tmp_path, b"DECLARED FILE CONTEXT\n")
    collect(tmp_path, task, trial)
    run = tmp_path / "run"
    entry = {"id": "case", "split": split, "artifact_pairs": [PAIR]}
    documents = {
        "result.json": {"skill_name": "skill", "execution_status": "succeeded"},
        "run_config.json": {"agents": {"agent": {"model": "fixed"}}},
        "dataset_snapshot.json": build_dataset_snapshot([entry], evaluator_version="test"),
    }
    for arm in ("with-skill", "without-skill"):
        documents[f"agent/{arm}/summary.json"] = {"scores": {}, "execution_status": "succeeded",
            "num_trials": 1, "expected_attempts": 1, "scored_attempts": 1, "metric_set": "custom-only"}
        documents[f"agent/{arm}/trials/attempt-1/reward.json"] = {"entry_id": "case", "trial_id": "attempt-1",
            "metric_set": "custom-only", "score_policy": "custom-overall-v1", "overall": 1}
        documents[f"agent/{arm}/trials/attempt-1/result.json"] = {"finished_at": "2026-10-28T00:00:00Z"}
    for relative, value in documents.items():
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    case = load_signals(run)["agents"]["agent"]["cases"][0]
    observations = _discover_trajectories(tmp_path / "skill", from_results=run)["case"]
    if split == "discovery":
        assert case["attempts"][0]["artifact_evidence"]["pairs"][0]["original"]["text"] == "DECLARED FILE CONTEXT\n"
        assert observations[0]["artifact_evidence"]["pairs"][0]["bytes_equal"] is True
    else:
        assert "DECLARED FILE CONTEXT" not in json.dumps(case)
        assert all("artifact_evidence" not in item for item in observations)
    assert all("artifact_evidence" not in attempt for attempt in case["baseline_attempts"])
    assert all("artifact_evidence" not in item for item in observations if item["arm"] == "without-skill")


@pytest.mark.parametrize("field,value", [("capture", "legacy_verifier_capture"), ("task_contract_sha256", "invalid"), ("native_manifest_sha256", "invalid"), ("status", "unavailable")])
def test_consumer_rejects_contradictory_capture_provenance(tmp_path, field, value):
    task, trial, _ = staged(tmp_path)
    record = collect(tmp_path, task, trial)
    record[field] = value
    (tmp_path / "run/agent/with-skill/trials/attempt-1/artifact_evidence.json").write_text(json.dumps(record))
    loaded = evidence.load_artifact_evidence(tmp_path / "run", "agent", "attempt-1",
        entry_id="case", declarations=[PAIR])
    assert loaded["status"] == "unavailable" and "pairs" not in loaded


def test_collector_rejects_unknown_staged_contract_version(tmp_path):
    task, trial, _ = staged(tmp_path)
    path = task / "artifact_pairs.json"
    contract = json.loads(path.read_text())
    contract["schema_version"] = 999
    path.write_text(json.dumps(contract))
    record = collect(tmp_path, task, trial)
    assert record["status"] == "unavailable" and record["reason"] == "invalid_staged_contract"


@pytest.mark.parametrize("location", ["top_level", "complete_side", "partial_reason", "limits"])
def test_consumer_never_exposes_secret_in_unrecognized_record_fields(tmp_path, monkeypatch, location):
    secret = 'synthetic-private-credential-with-"quote\\slash-abcdefgh'
    monkeypatch.setenv("NVIDIA_API_KEY", secret)
    task, trial, _ = staged(tmp_path)
    record = collect(tmp_path, task, trial)
    if location == "top_level":
        record["extra"] = {"nested": secret}
    elif location == "complete_side":
        record["pairs"][0]["original"]["extra"] = {"nested": secret}
    elif location == "partial_reason":
        record["status"] = "partial"
        record["pairs"] = [{"id": PAIR["id"], "status": "collection_failed", "reason": secret}]
    else:
        record["limits"]["extra"] = secret
    path = tmp_path / "run/agent/with-skill/trials/attempt-1/artifact_evidence.json"
    path.write_text(json.dumps(record))
    loaded = evidence.load_artifact_evidence(tmp_path / "run", "agent", "attempt-1",
        entry_id="case", declarations=[PAIR])
    assert loaded["status"] == "unavailable" and "pairs" not in loaded
    assert "synthetic-private-credential" not in json.dumps(loaded)
