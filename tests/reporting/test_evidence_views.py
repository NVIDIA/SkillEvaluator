# SPDX-License-Identifier: Apache-2.0
"""Recorded Tier 2 evidence stays separate from missing comparison coverage."""

from copy import deepcopy

import pytest

from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.reporting.html import HTMLReporter


def _within():
    result = ValidationResult(validator_name="Context Deduplication")
    result.add_success("context_dedup", "Complete")
    result.metadata["workflow"] = {
        "tier": 2,
        "stages": {
            "intra_skill": {"status": "passed"},
            "catalog_comparison": {"status": "skipped", "reason": "No comparison catalog supplied."},
        },
    }
    return result


def test_scope_projection_distinguishes_clean_zero_from_skipped():
    from skillevaluator.reporting.tier2_views import tier2_scope_views

    result = _within()
    original = deepcopy(result)
    within, across = tier2_scope_views([result])
    assert (within.label, within.completed, within.duplicates, within.affected) == ("Within this skill", 1, 0, 0)
    assert (across.label, across.completed, across.duplicates, across.catalog_entries) == (
        "Across skills",
        0,
        None,
        None,
    )
    assert "No comparison catalog supplied." in across.notes
    assert result == original


def test_catalog_counts_self_exclusion_and_pairwise_is_not_catalog():
    from skillevaluator.reporting.tier2_views import tier2_scope_views

    catalog = ValidationResult(validator_name="Similarity Check")
    catalog.add_success("catalog_compared", "Compared", catalog_entry_count=5, self_matches_excluded=1)
    across = tier2_scope_views([catalog])[1]
    assert (across.completed, across.duplicates, across.catalog_entries) == (1, 0, 4)
    assert any("target against" in note for note in across.notes)
    pairwise = ValidationResult(validator_name="Similarity Check")
    pairwise.add_success("index_built", "Indexed", entry_count=3, content_type="skill")
    across = tier2_scope_views([pairwise])[1]
    assert across.catalog_entries is None
    assert "Pairwise comparison of 3 collection entries; no catalog used." in across.notes


def test_partial_and_unknown_evidence_does_not_claim_complete_or_zero():
    from skillevaluator.reporting.tier2_views import tier2_scope_views

    unknown = ValidationResult(validator_name="Context Deduplication")
    assert tier2_scope_views([unknown])[0].completed is None
    assert tier2_scope_views([unknown])[0].duplicates is None
    unknown.metadata["workflow"] = {"tier": 2, "stages": {"intra_skill": {"status": "failed"}}}
    unknown.add_error("Provider failed")
    assert tier2_scope_views([unknown])[0].completed is None
    unknown.mark_scan_incomplete("embedding-provider")
    assert tier2_scope_views([unknown])[0].duplicates is None


def test_refused_and_safety_findings_are_not_duplicates_and_paths_are_distinct():
    from skillevaluator.reporting.tier2_views import tier2_scope_views

    result = _within()
    for category, check in (("CONTENT_DEDUP", "llm_refused"), ("SAFETY", "unsafe_file"), ("DUPLICATE", "duplicate")):
        result.add_finding(
            Finding(
                category=category,
                check_name=check,
                severity=Severity.MEDIUM,
                message="Recorded evidence",
                file_path="SKILL.md",
                metadata={"source_files": ["SKILL.md", "ref.md", "ref.md"]},
            )
        )
    within = tier2_scope_views([result])[0]
    assert (within.duplicates, within.affected) == (1, 2)


def test_html_scopes_keep_kpis_and_findings_separate():
    output = HTMLReporter(include_timestamp=False).render_all([_within()])
    assert 'id="tier2-within"' in output
    assert 'id="tier2-across"' in output
    assert "Within this skill" in output and "Across skills" in output
    assert "Checks completed" in output and "Duplicate findings" in output
    assert "Affected files" in output and "Affected skills" in output
    assert "Catalog entries compared" in output
    assert "No comparison catalog supplied." in output
    assert "Unavailable" in output


def test_insufficient_content_does_not_claim_zero_match_comparison():
    from skillevaluator.reporting.tier2_views import tier2_scope_views

    result = _within()
    result.success_details.clear()
    result.add_success("chunking", "Extracted", chunk_count=1)
    result.add_success("context_dedup", "Not enough content to compare")
    view = tier2_scope_views([result])[0]
    assert view.completed == 0
    assert view.duplicates is None
    assert any("Not enough content" in note for note in view.notes)


def test_completed_failing_duplicate_check_is_still_completed():
    from skillevaluator.reporting.tier2_views import tier2_scope_views

    result = ValidationResult(validator_name="Similarity Check")
    result.add_success("catalog_compared", "Compared", catalog_entry_count=3, self_matches_excluded=0)
    result.add_finding(
        Finding(
            category="SIMILARITY",
            check_name="EXACT_DUPLICATE",
            severity=Severity.HIGH,
            message="Match",
            file_path="a",
            metadata={"path_a": "a", "path_b": "b"},
        )
    )
    view = tier2_scope_views([result])[1]
    assert result.passed is False
    assert (view.completed, view.duplicates, view.affected, view.catalog_entries) == (1, 1, 2, 3)


def test_harbor_projection_preserves_unmatched_legacy_analysis_and_deduplicates():
    from skillevaluator.reporting.harbor_viewer import harbor_job_links

    payload = {
        "summary": {
            "harbor_viewer": {
                "analysis_url": "https://example.test/legacy",
                "jobs": [
                    {"url": "https://example.test/a", "name": "a"},
                    {"url": "https://example.test/a", "name": "copy"},
                    {"url": "https://example.test/b", "analysis_url": "https://example.test/b/analysis"},
                    {"url": "javascript:alert(1)"},
                ],
            }
        }
    }
    original = deepcopy(payload)
    jobs = harbor_job_links(payload)
    assert [job["name"] for job in jobs] == ["a", "Job 2", "Run analysis"]
    assert jobs[2]["analysis_url"] == "https://example.test/legacy"
    assert jobs[1]["analysis_url"] == "https://example.test/b/analysis"
    assert payload == original
    assert harbor_job_links({"harbor_viewer": {"analysis_url": "https://example.test/analysis"}}) == [
        {"name": "Run analysis", "analysis_url": "https://example.test/analysis"}
    ]
    assert harbor_job_links({}) == []


def test_refused_workflow_does_not_claim_a_complete_zero_match_scan():
    from skillevaluator.reporting.tier2_views import tier2_scope_views

    result = _within()
    result.success_details.clear()
    result.metadata["llm_analysis"] = {
        "clusters_total": 1,
        "clusters_completed": 0,
        "clusters_failed": 0,
        "clusters_refused": 1,
    }
    view = tier2_scope_views([result])[0]
    assert view.completed is None
    assert view.duplicates is None


def test_harbor_shared_analysis_is_linked_once_without_losing_results():
    from skillevaluator.reporting.harbor_viewer import harbor_job_links

    shared = "https://example.test/shared-analysis"
    jobs = harbor_job_links(
        {
            "harbor_viewer": {
                "jobs": [
                    {"url": "https://example.test/a", "analysis_url": shared},
                    {"url": "https://example.test/b", "analysis_url": shared},
                ]
            }
        }
    )
    assert len(jobs) == 2
    assert [job["url"] for job in jobs] == ["https://example.test/a", "https://example.test/b"]
    assert sum(job.get("analysis_url") == shared for job in jobs) == 1


def test_compact_validate_summary_keeps_all_harbor_evidence_reachable(tmp_path):
    from io import StringIO

    from rich.console import Console

    from skillevaluator.cli import _finish_pipeline_view
    from skillevaluator.reporting.console_ui import ValidateView

    result = ValidationResult(validator_name="AGENT_EVAL")
    result.metadata["agent_eval"] = {
        "summary": {"execution_status": "succeeded"},
        "harbor_viewer": {
            "jobs": [
                {"url": "https://example.test/a", "analysis_url": "https://example.test/analysis", "name": "[bad]"},
                {"url": "https://example.test/b"},
            ]
        },
    }
    original = deepcopy(result)
    output = StringIO()
    view = ValidateView("test", [], console=Console(file=output, width=1000, color_system=None))
    _finish_pipeline_view(
        view,
        tier_gate_results=[],
        tier3_result=result,
        gate_failed=False,
        output_dir=tmp_path,
        basename="report",
        report_formats=("html",),
        target_path=tmp_path,
        agent_eval=True,
    )
    text = output.getvalue()
    assert "https://example.test/a" in text
    assert "https://example.test/b" in text
    assert "https://example.test/analysis" in text
    assert "Job 2" in text
    assert result == original


def test_catalog_kpi_marks_non_catalog_scopes_not_applicable():
    result = ValidationResult(validator_name="Similarity Check")
    result.add_success("index_built", "Indexed", entry_count=3, content_type="skill")
    output = HTMLReporter(include_timestamp=False).render_all([result])
    across = output.split('id="tier2-across"', 1)[1].split("</section>", 1)[0]
    catalog_card = across.split("Catalog entries compared</dt>", 1)[1].split("</div>", 1)[0]
    assert "Not applicable" in catalog_card
    # similarity-check never runs the within-skill scope: say so, without Unavailable KPIs.
    within = output.split('id="tier2-within"', 1)[1].split("</section>", 1)[0]
    assert "Not run by this command." in within
    assert "Unavailable" not in within and "dashboard-card" not in within


def test_passing_workflow_stages_alone_do_not_prove_comparison_completion():
    from skillevaluator.reporting.tier2_views import tier2_scope_views

    for name, stage, index in (
        ("Context Deduplication", "intra_skill", 0),
        ("Similarity Check", "catalog_comparison", 1),
    ):
        result = ValidationResult(validator_name=name)
        result.metadata["workflow"] = {"tier": 2, "stages": {stage: {"status": "passed"}}}
        original = deepcopy(result)
        view = tier2_scope_views([result])[index]
        assert (view.completed, view.duplicates, view.affected) == (None, None, None)
        output = HTMLReporter(include_timestamp=False).render_all([result])
        assert "No semantic overlaps detected." not in output
        assert result == original


def test_single_entry_collection_index_is_not_a_completed_comparison():
    from skillevaluator.reporting.tier2_views import tier2_scope_views

    single = ValidationResult(validator_name="Similarity Check")
    single.add_success("index_built", "Indexed", entry_count=1, content_type="skill")
    across = tier2_scope_views([single])[1]
    assert across.completed == 0
    assert across.duplicates is None
    assert not any("Pairwise comparison" in note for note in across.notes)
    assert any(note.startswith("Comparison not applicable") for note in across.notes)


def test_insufficient_content_note_is_not_contradicted_by_incomplete_coverage():
    from skillevaluator.reporting.tier2_views import tier2_scope_views

    result = ValidationResult(validator_name="Context Deduplication")
    result.add_success("context_dedup", "Not enough content to compare")
    within = tier2_scope_views([result])[0]
    assert "Comparison not applicable: Not enough content to compare" in within.notes
    assert not any("incomplete" in note for note in within.notes)


def test_across_scope_counts_similarity_matches_not_duplicates():
    result = ValidationResult(validator_name="Similarity Check")
    result.add_success("catalog_compared", "Compared", catalog_entry_count=5, self_matches_excluded=1)
    output = HTMLReporter(include_timestamp=False).render_all([result])
    across = output.split('id="tier2-across"', 1)[1].split("</section>", 1)[0]
    assert "Similarity matches</dt>" in across and "Duplicate findings" not in across


def test_harbor_legacy_pair_stays_one_job_and_bad_shapes_do_not_crash():
    from skillevaluator.reporting.harbor_viewer import harbor_job_links, normalize_harbor_viewer_for_display

    viewer = {
        "job_url": "https://example.test/legacy",
        "analysis_url": "https://example.test/legacy/analysis",
        "jobs": [{"url": "https://example.test/a", "name": "a"}],
    }
    jobs = harbor_job_links({"harbor_viewer": viewer})
    assert [(job["name"], job.get("analysis_url")) for job in jobs] == [
        ("a", None),
        ("Job 2", "https://example.test/legacy/analysis"),
    ]
    assert normalize_harbor_viewer_for_display({"harbor_viewer": {"jobs": 1, "evidence_links": "x"}}) == {}


def test_terminal_lists_harbor_jobs_from_engine_uploads():
    from skillevaluator.tier3.result_display import render_result

    engine = {
        "execution_status": "succeeded",
        "harbor_viewer": {
            "uploads": [
                {"viewer_url": "https://hv.example.test/jobs/with", "uploaded_job_name": "with-skill"},
                {"viewer_url": "https://hv.example.test/jobs/base", "uploaded_job_name": "baseline"},
            ]
        },
    }
    output = render_result(engine)
    assert "hv.example.test/jobs/with" in output and "hv.example.test/jobs/base" in output


@pytest.mark.parametrize("uploads", [1, True, {}, "malformed", None, []])
def test_terminal_bad_or_empty_uploads_preserve_normalized_jobs_and_run_analysis(uploads):
    from skillevaluator.tier3.result_display import render_result

    result = {
        "execution_status": "succeeded",
        "harbor_viewer": {
            "uploads": uploads,
            "jobs": [
                {
                    "url": "https://example.test/results/retained",
                    "name": "Retained job",
                    "analysis_url": "https://example.test/analysis/retained",
                }
            ],
            "analysis_url": "https://example.test/run-analysis",
        },
    }
    before = deepcopy(result)
    output = render_result(result)
    assert "Retained job" in output
    for url in (
        "https://example.test/results/retained",
        "https://example.test/analysis/retained",
        "https://example.test/run-analysis",
    ):
        assert output.count(url) == 1
    assert "Run analysis" in output
    assert result == before


def test_terminal_upload_projection_merges_with_canonical_jobs_without_reattribution():
    from skillevaluator.tier3.result_display import render_result

    result = {
        "execution_status": "succeeded",
        "harbor_viewer": {
            "jobs": [
                {
                    "url": "https://example.test/results/retained",
                    "name": "Retained job",
                    "analysis_url": "https://example.test/analysis/retained",
                }
            ],
            "analysis_url": "https://example.test/run-analysis",
            "uploads": [
                {
                    "viewer_url": "https://example.test/results/uploaded",
                    "uploaded_job_name": "Uploaded job",
                    "analysis_url": "https://example.test/analysis/uploaded",
                },
                {"viewer_url": "https://example.test/results/uploaded", "uploaded_job_name": "Duplicate upload"},
                {
                    "viewer_url": "https://example.test/results/retained",
                    "uploaded_job_name": "Renamed retained job",
                    "analysis_url": "https://example.test/analysis/retained",
                },
                {"viewer_url": "javascript:alert(1)"},
            ],
        },
    }
    before = deepcopy(result)
    output = render_result(result)
    assert "Retained job" in output and "Uploaded job" in output
    assert "Renamed retained job" not in output and "Duplicate upload" not in output
    for url in (
        "https://example.test/results/retained",
        "https://example.test/analysis/retained",
        "https://example.test/results/uploaded",
        "https://example.test/analysis/uploaded",
        "https://example.test/run-analysis",
    ):
        assert output.count(url) == 1
    assert "Run analysis" in output
    assert "javascript:" not in output
    assert result == before


@pytest.mark.parametrize("existing_analysis", [None, "https://example.test/analysis/canonical"])
@pytest.mark.parametrize("padded_url", [False, True])
def test_terminal_matching_upload_fills_only_missing_canonical_analysis(existing_analysis, padded_url):
    from skillevaluator.tier3.result_display import render_result

    url = "https://example.test/results/retained"
    job = {"url": f" {url} " if padded_url else url, "name": "Retained job"}
    if existing_analysis:
        job["analysis_url"] = existing_analysis
    uploaded_analysis = "https://example.test/analysis/uploaded"
    result = {
        "execution_status": "succeeded",
        "harbor_viewer": {
            "jobs": [job],
            "uploads": [{"viewer_url": url, "analysis_url": uploaded_analysis, "uploaded_job_name": "Upload name"}],
        },
    }
    before = deepcopy(result)
    output = render_result(result)
    assert "Retained job" in output and "Upload name" not in output
    assert output.count(url) == 1
    assert output.count(existing_analysis or uploaded_analysis) == 1
    if existing_analysis:
        assert uploaded_analysis not in output
    assert result == before


def test_terminal_upload_analysis_never_fills_a_different_canonical_job():
    from skillevaluator.tier3.result_display import render_result

    result = {
        "execution_status": "succeeded",
        "harbor_viewer": {
            "jobs": [{"url": "https://example.test/results/retained", "name": "Retained job"}],
            "uploads": [
                {
                    "viewer_url": "https://example.test/results/uploaded",
                    "uploaded_job_name": "Uploaded job",
                    "analysis_url": "https://example.test/analysis/uploaded",
                }
            ],
        },
    }
    before = deepcopy(result)
    output = render_result(result)
    assert output.index("Retained job") < output.index("Uploaded job")
    assert output.index("Uploaded job") < output.index("https://example.test/analysis/uploaded")
    assert output.count("https://example.test/analysis/uploaded") == 1
    assert result == before
