# SPDX-License-Identifier: Apache-2.0
"""Durable comparison evidence retains every attempt within the artifact bounds."""

import json
from types import SimpleNamespace

import pytest

from skillevaluator.tier3.harbor import collector, loose_parts_arm
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS
from skillevaluator.tier3.plugin_comparisons import ComparisonOptions, ComparisonRuntime
from skillevaluator.tier3.plugin_evidence import LogicalTrialEvidence


@pytest.mark.parametrize(
    "cases,note_size,limit", [(100, 80, "node count"), (30, 1024, "encoded JSON"), (1000, 80, "node count")]
)
def test_indexed_evidence_retains_all_attempts_beyond_uncompressed_artifact_limits(tmp_path, cases, note_size, limit):
    evidence = {
        arm: {
            f"case-{case}": [
                {
                    f"skill:piece-{piece}": {
                        "loaded": (True, False, None)[(case + attempt + piece) % 3],
                        "source": "system/init",
                        "evidence": f"piece-{piece}: " + "x" * note_size,
                        "status": "connected",
                        "tools": ["fixture"],
                    }
                    for piece in range(15)
                }
                for attempt in range(3)
            ]
            for case in range(cases)
        }
        for arm in ("with_skill", "sum_of_parts", "loose_parts")
    }
    with pytest.raises(ValueError, match=limit):
        collector._write_generated_root_json(tmp_path / "raw.json", tmp_path, evidence)

    compact = loose_parts_arm.compact_load_evidence(evidence)
    path = tmp_path / loose_parts_arm.LOAD_EVIDENCE_FILENAME
    collector._write_generated_root_json(path, tmp_path, compact)
    saved = json.loads(path.read_text())
    # Decode using the documented tables, independently of the encoder. Missing
    # attempts, piece names, failures, or unknown load states must all be visible.
    restored = {
        arm: {
            case: [
                {piece: saved["results"][row] for piece, row in saved["attempts"][index].items()} for index in attempts
            ]
            for case, attempts in rows.items()
        }
        for arm, rows in saved["arms"].items()
    }
    assert restored == evidence
    assert saved["format"] == "indexed-v1"


def _runtime(tmp_path, *, case_count=1, n_attempts=1, n_pieces=0):
    agents = ("claude-code", "codex")
    jobs_dir = tmp_path / "_harbor-jobs"
    jobs_dir.mkdir()
    cases = tuple(f"case-{index}" for index in range(case_count))
    names = [f"piece-{index}" for index in range(n_pieces)]
    records = [
        LogicalTrialEvidence(
            trial_id=f"{case}__attempt-{ordinal}",
            case_id=case,
            reward={
                "entry_id": case,
                "_attempt_ordinal": ordinal,
                "metric_set": DEFAULT_METRIC_SET,
                **dict.fromkeys(DEFAULT_METRICS, 1.0),
            },
            signals=None,
            native_census=None,
            usage=None,
            first_turn_usage=None,
            evidence_status="recorded",
            init_event={"skills": names + [f"fixture:{name}" for name in names], "plugins": [{"name": "fixture"}]},
        )
        for case in cases
        for ordinal in range(1, n_attempts + 1)
    ]
    arms = (
        ("with_skill", "without_skill", "sum_of_parts", "loose_parts") if n_pieces else ("with_skill", "without_skill")
    )
    observer = SimpleNamespace(
        logical_trials={
            agent: dict.fromkeys(arms if agent == "claude-code" else arms[:2], records) for agent in agents
        },
        context=SimpleNamespace(case_spec=lambda _case: {"cross_component": True}),
    )
    native = SimpleNamespace(observer=observer, decisions={}, collect_evidence=lambda *_args: None)
    package = SimpleNamespace(
        native_source=SimpleNamespace(member_skills=(tmp_path / "member",) if n_pieces else ()),
        identity=SimpleNamespace(),
    )
    runtime = ComparisonRuntime(native, package, ComparisonOptions(loose_parts=True), agents, baseline=True)
    runtime.expected_case_ids = cases
    runtime.n_attempts = n_attempts
    for agent in agents:
        runtime.jobs[agent] = {}
        for arm in observer.logical_trials[agent]:
            job = jobs_dir / f"{agent}-{arm}"
            job.mkdir()
            (job / "result.json").write_text(
                json.dumps(
                    {
                        "n_total_trials": len(records),
                        "stats": {
                            "n_trials": len(records),
                            "n_errors": 0,
                            "evals": {
                                "fixture": {
                                    "n_trials": len(records),
                                    "n_errors": 0,
                                    "reward_stats": {"score": {"1.0": [record.trial_id for record in records]}},
                                }
                            },
                        },
                    }
                )
            )
            runtime.jobs[agent][arm] = job
    # Only the first agent has a loose arm. Its artifact failure must not prevent
    # statistics for the later agent, whose native route cannot stage loose parts.
    placements = [SimpleNamespace(id=f"skill:{name}", kind="skill", name=name) for name in names]
    runtime.loose_plan.arms["claude-code"] = SimpleNamespace(
        parts=SimpleNamespace(_placements=lambda _arm: placements, source=SimpleNamespace(plugin_name="fixture")),
        content={},
        info=lambda evidence: {"measured": True, "load_evidence": loose_parts_arm.load_evidence_summary(evidence)},
    )
    result = {"agents": {agent: {} for agent in agents}}
    return runtime, jobs_dir, result


def test_large_runtime_collection_saves_complete_evidence_and_statistics(tmp_path):
    runtime, jobs_dir, result = _runtime(tmp_path, case_count=100, n_attempts=3, n_pieces=15)

    runtime.collect_evidence(jobs_dir, result)

    saved = json.loads((tmp_path / "private_comparisons.json").read_text())
    assert saved["status"] == "recorded" and saved["errors"] == []
    agent = saved["agents"]["claude-code"]
    assert agent["lift_ladder"]["n_cases"] == 100
    assert agent["loose_parts"]["load_evidence"]["trials"] == {
        "with_skill": 300,
        "sum_of_parts": 300,
        "loose_parts": 300,
    }
    assert agent["loose_parts"]["load_evidence"]["pieces"]["loose_parts"]["skill:piece-14"]["loaded"] == 300
    evidence = json.loads((tmp_path / "claude-code" / loose_parts_arm.LOAD_EVIDENCE_FILENAME).read_text())
    assert evidence["format"] == "indexed-v1"
    assert {arm: len(cases) for arm, cases in evidence["arms"].items()} == {
        "with_skill": 100,
        "sum_of_parts": 100,
        "loose_parts": 100,
    }
    assert all(len(attempts) == 3 for cases in evidence["arms"].values() for attempts in cases.values())
    assert all(len(attempt) == 15 for attempt in evidence["attempts"])
    assert (tmp_path / "codex/statistics.json").is_file()


@pytest.mark.parametrize("artifact", ["statistics.json", loose_parts_arm.LOAD_EVIDENCE_FILENAME])
def test_one_artifact_write_failure_keeps_other_artifacts_and_later_agents(tmp_path, monkeypatch, artifact):
    runtime, jobs_dir, result = _runtime(tmp_path)
    real_write = collector._write_generated_root_json

    def write(path, output, payload):
        if path.parent.name == "claude-code" and path.name == artifact:
            raise ValueError("encoded JSON exceeds limit")
        return real_write(path, output, payload)

    monkeypatch.setattr(collector, "_write_generated_root_json", write)
    runtime.collect_evidence(jobs_dir, result)

    saved = json.loads((tmp_path / "private_comparisons.json").read_text())
    assert saved["status"] == "incomplete"
    assert set(saved["agents"]) == {"claude-code", "codex"}
    assert len(saved["errors"]) == 1 and f"claude-code/{artifact}" in saved["errors"][0]
    assert result["private_comparisons"] == saved
    assert result["agents"]["codex"]["cost"]
    assert json.loads((tmp_path / "codex/statistics.json").read_text())["reliability"]
    other = (
        "statistics.json"
        if artifact == loose_parts_arm.LOAD_EVIDENCE_FILENAME
        else loose_parts_arm.LOAD_EVIDENCE_FILENAME
    )
    assert (tmp_path / "claude-code" / other).is_file()


def test_standalone_write_failure_is_reported_inline_after_agent_artifacts_are_saved(tmp_path, monkeypatch):
    runtime, jobs_dir, result = _runtime(tmp_path)
    real_write = collector._write_generated_root_json

    def write(path, output, payload):
        if path.name == "private_comparisons.json":
            raise OSError("fixture write failure")
        return real_write(path, output, payload)

    monkeypatch.setattr(collector, "_write_generated_root_json", write)
    runtime.collect_evidence(jobs_dir, result)

    assert result["private_comparisons"]["status"] == "incomplete"
    assert len(result["private_comparisons"]["errors"]) == 1
    assert "private_comparisons.json" in result["private_comparisons"]["errors"][0]
    assert all((tmp_path / agent / "statistics.json").is_file() for agent in ("claude-code", "codex"))
    assert (tmp_path / "claude-code" / loose_parts_arm.LOAD_EVIDENCE_FILENAME).is_file()


def test_comparison_attempt_reset_preserves_previous_results_and_resets_native_hooks(tmp_path):
    runtime, _jobs_dir, _result = _runtime(tmp_path)
    runtime.native.begin_attempt = lambda: setattr(runtime.native, "evidence", {})
    runtime.native.evidence = {"previous": True}
    runtime.data = {"status": "recorded", "agents": {"previous": {}}, "errors": []}
    previous = runtime.provenance()

    runtime.begin_attempt()

    assert runtime.native.evidence == {}
    assert runtime.data == {"status": "not_collected", "agents": {}, "errors": []}
    assert runtime.jobs == {} and runtime.expected_case_ids == ()
    assert runtime.loose_plan.arms == {}
    assert previous["agents"] == {"previous": {}}
