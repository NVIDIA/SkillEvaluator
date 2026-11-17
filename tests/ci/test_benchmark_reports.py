# SPDX-License-Identifier: Apache-2.0
"""Combined cards use retained evidence and exact manifest skill identities."""

import copy
import json
from pathlib import Path

import pytest

from ci import benchmark_reports
from skillevaluator.models import ValidationResult
from skillevaluator.publication_evidence import stamp_publication_evidence, tier1_required_checks
from skillevaluator.reporting import JSONReporter
from skillevaluator.reporting.retained import retained_results


def results(tier, name="demo", *, digest="a", verdict="pass"):
    target = {
        "skill_name": name,
        "skill_digest": "sha256:" + digest * 64,
        "skill_digest_algorithm": "skill-evaluator-source-tree/2",
    }
    if tier == 1:
        checks = tier1_required_checks(skill_like=True)
        rows = []
        for check in checks:
            row = ValidationResult(validator_name=check.upper())
            row.add_success(check, "Completed")
            stamp_publication_evidence([row], tier=1, check_id=check)
            row.metadata["tier1_required_checks"] = checks
            rows.append(row)
    elif tier == 2:
        rows = [ValidationResult(validator_name="Similarity Check")]
        rows[0].add_success("similarity", "Completed")
        stamp_publication_evidence(rows, tier=2, check_id="similarity")
    else:
        score = 0.9 if verdict == "pass" else 0.2
        coverage = {
            "execution_status": "succeeded",
            "execution_errors": [],
            "expected_attempts": 2,
            "scored_attempts": 2,
        }
        summary = {
            **coverage,
            "skill_name": name,
            "verdict": verdict,
            "run_id": "retained-live-run",
            "publication_target": target,
            "environment": "docker",
            "agents_run": ["codex"],
            "best_agent": "codex",
            "overall_score": score,
            "overall_lift": score - 0.4,
        }
        agent = {
            **coverage,
            "model": "gpt-codex",
            "baseline": 0.4,
            "with_skill": score,
            "lift": score - 0.4,
            "conditions": {
                condition: {**coverage, "expected_attempts": 1, "scored_attempts": 1}
                for condition in ("with_skill", "without_skill")
            },
            "dimensions": [
                {"id": dimension, "with_skill": score, "baseline": 0.4, "lift": score - 0.4}
                for dimension in ("security", "correctness", "discoverability", "effectiveness", "efficiency")
            ],
        }
        row = ValidationResult(validator_name="AGENT_EVAL")
        row.add_success("agent_eval", "Completed")
        row.metadata["agent_eval"] = {
            **summary,
            "summary": summary,
            "evaluated_at": "2026-08-25T12:00:00+00:00",
            "evaluator_version": "0.9.0",
            "agents": {"codex": agent},
            "dimensions": agent["dimensions"],
            "dataset_summary": {"total_tasks": 1},
            "dataset_digest": "sha256:" + "b" * 64,
            "dataset_digest_algorithm": "skill-evaluator-dataset-snapshot/1",
            "attempt_policy": {"max_attempts": 1, "pass_threshold": 0.5, "stop_on_pass": False},
        }
        rows = [row]
    for row in rows:
        row.metadata["publication_target"] = copy.deepcopy(target)
        row.metadata["gating"] = {"tier": tier, "blocking": tier == 1}
    return rows


def retain(reports, *, names=("demo",), tiers=(1, 2, 3), verdict="pass"):
    manifest = {"tiers": []}
    for tier in tiers:
        record = {"tier": tier, "required": tier == 1, "status": "passed", "skills": []}
        for index, name in enumerate(names, 1):
            relative = f"tier{tier}/skill-{index:03}/skillevaluator-tier{tier}.json"
            destination = reports / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                JSONReporter(include_timestamp=False).render_all(results(tier, name, verdict=verdict))
            )
            record["skills"].append(
                {
                    "path": f"skills/{name}",
                    "status": "failed" if tier == 3 and verdict == "fail" else "passed",
                    "report_json": relative,
                }
            )
        manifest["tiers"].append(record)
    return manifest


def read_combined(reports, row):
    assert row["combined_json"].startswith("benchmarks/skill-")
    assert row["benchmark_md"].endswith("/BENCHMARK.md")
    combined = json.loads((reports / row["combined_json"]).read_text())
    card = (reports / row["benchmark_md"]).read_text()
    assert row["publication_status"] == combined["publication_status"]
    assert row["eligible"] == combined["publication"]["eligible"]
    return combined, card


def test_complete_retained_reports_preserve_run_owned_evidence(tmp_path):
    manifest = retain(tmp_path)
    original = json.loads((tmp_path / manifest["tiers"][2]["skills"][0]["report_json"]).read_text())
    rows = benchmark_reports.write_benchmark_reports(tmp_path, manifest)
    combined, card = read_combined(tmp_path, rows[0])
    assert rows[0]["path"] == "skills/demo"
    assert combined["publication_status"] == "pass"
    assert "Overall verdict: PASS" in card
    assert combined["tier3"] == original["tier3"]
    assert combined["results"][-1]["publication_target"] == original["results"][0]["publication_target"]


def test_reordered_rows_compose_by_exact_skill_path(tmp_path):
    manifest = retain(tmp_path, names=("second", "demo"))
    manifest["tiers"][1]["skills"].reverse()
    rows = benchmark_reports.write_benchmark_reports(tmp_path, manifest)
    assert [row["path"] for row in rows] == ["skills/demo", "skills/second"]
    for row in rows:
        combined, _ = read_combined(tmp_path, row)
        assert combined["publication_status"] == "pass"
        assert {result["publication_target"]["skill_name"] for result in combined["results"]} == {
            Path(row["path"]).name
        }


@pytest.mark.parametrize("tiers", [(1,), (2, 3), (1, 3)])
def test_missing_tiers_stay_publication_required(tmp_path, tiers):
    manifest = retain(tmp_path, tiers=tiers)
    combined, card = read_combined(tmp_path, benchmark_reports.write_benchmark_reports(tmp_path, manifest)[0])
    assert combined["publication_status"] == "incomplete"
    assert combined["benchmark_policy"] == {"tier2_required": True, "tier3_required": True}
    assert "Overall verdict: INCOMPLETE" in card


@pytest.mark.parametrize("change", ["digest", "name", "case"])
def test_foreign_source_cannot_certify_card(tmp_path, change):
    manifest = retain(tmp_path)
    path = tmp_path / manifest["tiers"][1]["skills"][0]["report_json"]
    name = {"name": "foreign", "case": "Demo"}.get(change, "demo")
    path.write_text(
        JSONReporter(include_timestamp=False).render_all(results(2, name, digest="c" if change == "digest" else "a"))
    )
    combined, card = read_combined(tmp_path, benchmark_reports.write_benchmark_reports(tmp_path, manifest)[0])
    assert combined["publication_status"] == "incomplete"
    assert "Overall verdict: INCOMPLETE" in card


def test_native_advisory_quality_failure_does_not_block_ci(tmp_path):
    manifest = retain(tmp_path, verdict="fail")
    combined, card = read_combined(tmp_path, benchmark_reports.write_benchmark_reports(tmp_path, manifest)[0])
    assert combined["overall_passed"] is True
    assert combined["publication_status"] == "fail"
    assert "Overall verdict: FAIL" in card


def test_incomplete_manifest_cannot_reuse_pass_report(tmp_path):
    manifest = retain(tmp_path)
    manifest["tiers"][2]["skills"][0]["status"] = "incomplete"
    combined, card = read_combined(tmp_path, benchmark_reports.write_benchmark_reports(tmp_path, manifest)[0])
    assert combined["overall_passed"] is True
    assert combined["publication_status"] == "incomplete"
    assert "Overall verdict: INCOMPLETE" in card


@pytest.mark.parametrize("unsafe", ["../outside.json", "/tmp/outside.json", "tier2/link/report.json"])
def test_report_paths_cannot_escape_retained_root(tmp_path, unsafe):
    manifest = retain(tmp_path)
    outside = tmp_path.parent / "benchmark-outside"
    outside.mkdir(exist_ok=True)
    (tmp_path / "tier2/link").symlink_to(outside, target_is_directory=True)
    manifest["tiers"][1]["skills"][0]["report_json"] = unsafe
    combined, _ = read_combined(tmp_path, benchmark_reports.write_benchmark_reports(tmp_path, manifest)[0])
    assert combined["publication_status"] == "incomplete"


def test_failed_manifest_cannot_publish_a_passed_native_report(tmp_path):
    manifest = retain(tmp_path)
    manifest["tiers"][1]["skills"][0]["status"] = "failed"
    combined, _ = read_combined(tmp_path, benchmark_reports.write_benchmark_reports(tmp_path, manifest)[0])
    assert combined["publication_status"] != "pass"


def test_explicit_optional_policy_survives_without_deriving_it_from_ci_gate(tmp_path):
    manifest = retain(tmp_path, tiers=(1,))
    path = tmp_path / manifest["tiers"][0]["skills"][0]["report_json"]
    native = results(1)
    for result in native:
        result.metadata["benchmark_policy"] = {"tier2_required": False, "tier3_required": False}
    path.write_text(JSONReporter(include_timestamp=False).render_all(native))
    combined, _ = read_combined(tmp_path, benchmark_reports.write_benchmark_reports(tmp_path, manifest)[0])
    assert combined["benchmark_policy"] == {"tier2_required": False, "tier3_required": False}
    assert combined["publication_status"] == "pass"


def test_selected_missing_row_is_incomplete_despite_optional_policy(tmp_path):
    manifest = retain(tmp_path, tiers=(1,))
    path = tmp_path / manifest["tiers"][0]["skills"][0]["report_json"]
    native = results(1)
    for result in native:
        result.metadata["benchmark_policy"] = {"tier2_required": False, "tier3_required": False}
    path.write_text(JSONReporter(include_timestamp=False).render_all(native))
    manifest["tiers"].append(
        {"tier": 2, "status": "incomplete", "required": False, "skills": [], "selected": ["skills/demo"]}
    )
    combined, _ = read_combined(tmp_path, benchmark_reports.write_benchmark_reports(tmp_path, manifest)[0])
    assert combined["publication_status"] == "incomplete"


def test_legacy_tier1_without_required_lineup_stays_incomplete(tmp_path):
    manifest = retain(tmp_path)
    path = tmp_path / manifest["tiers"][0]["skills"][0]["report_json"]
    payload = json.loads(path.read_text())
    for result in payload["results"]:
        result.pop("tier1_required_checks", None)
    path.write_text(json.dumps(payload))
    combined, _ = read_combined(tmp_path, benchmark_reports.write_benchmark_reports(tmp_path, manifest)[0])
    assert combined["publication_status"] == "incomplete"
    assert all("tier1_required_checks" not in row for row in combined["results"])


def test_corrupted_skipped_report_does_not_create_optional_waiver(tmp_path):
    manifest = retain(tmp_path, tiers=(1, 2))
    path = tmp_path / manifest["tiers"][0]["skills"][0]["report_json"]
    native = results(1)
    for result in native:
        result.metadata["benchmark_policy"] = {"tier2_required": False, "tier3_required": False}
    path.write_text(JSONReporter(include_timestamp=False).render_all(native))
    row = manifest["tiers"][1]["skills"][0]
    row["status"] = "skipped"
    (tmp_path / row["report_json"]).write_text("{broken")
    combined, _ = read_combined(tmp_path, benchmark_reports.write_benchmark_reports(tmp_path, manifest)[0])
    assert combined["publication_status"] == "incomplete"


def test_native_tier2_failure_is_advisory_in_combined_ci_gate(tmp_path):
    manifest = retain(tmp_path)
    path = tmp_path / manifest["tiers"][1]["skills"][0]["report_json"]
    native = results(2)
    native[0].add_error("Similarity threshold exceeded")
    native[0].metadata["gating"]["blocking"] = True
    path.write_text(JSONReporter(include_timestamp=False).render_all(native))
    manifest["tiers"][1]["skills"][0]["status"] = "failed"
    combined, _ = read_combined(tmp_path, benchmark_reports.write_benchmark_reports(tmp_path, manifest)[0])
    assert combined["overall_passed"] is True
    assert combined["publication_status"] == "fail"
    assert combined["benchmark_policy"]["tier2_required"] is True


def in_memory_reports(reports, manifest):
    return {
        row["report_json"]: json.loads((reports / row["report_json"]).read_text())
        for tier in manifest["tiers"]
        for row in tier["skills"]
        if row.get("report_json")
    }


def test_in_memory_documents_match_disk_composition_without_mutating_inputs(tmp_path):
    manifest = retain(tmp_path)
    manifest.update(repository="acme/skills", commit="c" * 40, runtime={"tools_ref": "d" * 40})
    native = in_memory_reports(tmp_path, manifest)
    original = copy.deepcopy(native)
    documents = benchmark_reports.benchmark_documents(manifest, native)
    assert native == original
    assert len(documents) == 1
    document = documents[0]
    combined = json.loads(document["combined_content"])
    assert combined["publication_status"] == "pass"
    assert combined["evaluated_source"] == {
        "repository": "acme/skills",
        "commit": "c" * 40,
        "evaluator_container_revision": "d" * 40,
    }
    assert "- Evaluated source: `acme/skills`" in document["benchmark_content"]
    assert "- Evaluated source revision: `" + "c" * 40 + "`" in document["benchmark_content"]
    assert "- Evaluator container revision: `" + "d" * 40 + "`" in document["benchmark_content"]
    rows = benchmark_reports.write_benchmark_reports(tmp_path, manifest)
    assert rows == [{key: value for key, value in document.items() if not key.endswith("_content")}]
    assert (tmp_path / rows[0]["combined_json"]).read_text() == document["combined_content"]
    assert (tmp_path / rows[0]["benchmark_md"]).read_text() == document["benchmark_content"]


@pytest.mark.parametrize("tools_ref", [None, "main", "0123456"])
def test_missing_or_mutable_tools_ref_never_invents_evaluator_revision(tmp_path, tools_ref):
    manifest = retain(tmp_path)
    manifest.update(repository="acme/skills", commit="c" * 40, runtime={"tools_ref": tools_ref})
    document = benchmark_reports.benchmark_documents(manifest, in_memory_reports(tmp_path, manifest))[0]
    assert "- Evaluator container revision: not recorded" in document["benchmark_content"]
    assert json.loads(document["combined_content"])["evaluated_source"] == {
        "repository": "acme/skills",
        "commit": "c" * 40,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("repository", "foreign/skills"),
        ("commit", "e" * 40),
        ("commit", "malformed"),
        ("evaluator_container_revision", "f" * 40),
    ],
)
def test_conflicting_native_source_is_retained_and_marked_incomplete(tmp_path, field, value):
    manifest = retain(tmp_path)
    manifest.update(repository="acme/skills", commit="c" * 40, runtime={"tools_ref": "d" * 40})
    native = in_memory_reports(tmp_path, manifest)
    relative = manifest["tiers"][2]["skills"][0]["report_json"]
    foreign = {field: value}
    native[relative]["results"][0]["tier3"]["summary"]["evaluated_source"] = foreign
    original = copy.deepcopy(native)
    document = benchmark_reports.benchmark_documents(manifest, native)[0]
    combined = json.loads(document["combined_content"])
    assert native == original
    assert combined["publication_status"] == "incomplete"
    assert foreign in combined["evaluated_source_conflicts"]
    rehydrated = retained_results(combined)
    roundtrip = json.loads(JSONReporter(include_timestamp=False, expected_skill_name="demo").render_all(rehydrated))
    assert roundtrip["publication_status"] == "incomplete"
    assert roundtrip["publication"]["eligible"] is False
    assert "Overall verdict: INCOMPLETE" in document["benchmark_content"]
    assert "- Evaluated source: `acme/skills`" in document["benchmark_content"]


@pytest.mark.parametrize("status", ["neutral", "skipped"])
def test_manifest_nonpass_status_cannot_publish_passed_native_evidence(tmp_path, status):
    manifest = retain(tmp_path)
    manifest["tiers"][2]["skills"][0]["status"] = status
    combined, card = read_combined(tmp_path, benchmark_reports.write_benchmark_reports(tmp_path, manifest)[0])
    assert combined["publication_status"] == "incomplete"
    assert "Overall verdict: INCOMPLETE" in card


def test_consistent_native_neutral_status_is_not_replaced_by_incomplete(tmp_path):
    manifest = retain(tmp_path)
    native = in_memory_reports(tmp_path, manifest)
    row = manifest["tiers"][2]["skills"][0]
    row["status"] = "neutral"
    payload = native[row["report_json"]]
    payload["results"][0]["tier3"]["verdict"] = "neutral"
    payload["results"][0]["tier3"]["summary"]["verdict"] = "neutral"
    document = benchmark_reports.benchmark_documents(manifest, native)[0]
    combined = json.loads(document["combined_content"])
    assert combined["publication_status"] == "neutral"
    assert not any(result.get("incomplete_scans") for result in combined["results"])


def test_consistent_optional_native_skip_stays_disclosed(tmp_path):
    manifest = retain(tmp_path, tiers=(1, 2))
    native = in_memory_reports(tmp_path, manifest)
    for result in native[manifest["tiers"][0]["skills"][0]["report_json"]]["results"]:
        result["benchmark_policy"] = {"tier2_required": False, "tier3_required": False}
    row = manifest["tiers"][1]["skills"][0]
    row["status"] = "skipped"
    skipped = results(2)[0]
    skipped.success_details.clear()
    skipped.summary.checks_performed = 0
    skipped.metadata.update(skipped=True, skip_reason="Comparison intentionally skipped by policy")
    native[row["report_json"]] = json.loads(JSONReporter(include_timestamp=False).render_all([skipped]))
    document = benchmark_reports.benchmark_documents(manifest, native)[0]
    combined = json.loads(document["combined_content"])
    assert combined["publication_status"] == "pass"
    assert "SKIPPED" in document["benchmark_content"]
    assert not any(result.get("incomplete_scans") for result in combined["results"])
