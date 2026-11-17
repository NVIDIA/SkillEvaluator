# SPDX-License-Identifier: Apache-2.0
"""Benchmark artifact links remain bounded, source-scoped and explicit."""

import pytest

from ci import tier_manifest


def manifest():
    return {
        "schema_version": 1,
        "report_contract": "skillevaluator-ci-v1",
        "commit": "a" * 40,
        "base_commit": None,
        "repository": "acme/skills",
        "platform": "github",
        "run_id": "123",
        "run_attempt": "1",
        "environment": "docker",
        "tiers": [{"tier": 1, "status": "skipped", "required": True, "skills": [], "reason": "No work"}],
        "benchmarks": [],
    }


def card():
    return {
        "path": "skills/demo",
        "combined_json": "benchmarks/skill-001/skillevaluator.json",
        "benchmark_md": "benchmarks/skill-001/BENCHMARK.md",
        "publication_status": "incomplete",
        "eligible": False,
    }


def with_card():
    payload = manifest()
    payload["tiers"][0].update(
        status="passed",
        skills=[
            {
                "path": "skills/demo",
                "status": "passed",
                "report_json": "tier1/skill-001/skillevaluator-tier1.json",
                "report_html": "tier1/skill-001/skillevaluator-tier1.html",
            }
        ],
    )
    payload["benchmarks"] = [card()]
    return payload


def test_cards_link_from_both_aggregate_surfaces_without_claiming_a_pass():
    payload = with_card()
    assert tier_manifest.validate_manifest(payload)["overall_passed"] is True
    markdown = tier_manifest.render_markdown(payload)
    html = tier_manifest._render_html(tier_manifest.validate_manifest(payload))
    assert "benchmarks/skill-001/BENCHMARK.md" in markdown
    assert "benchmarks/skill-001/BENCHMARK.md" in html
    assert "Publication" in markdown and "incomplete" in markdown


@pytest.mark.parametrize(
    "change",
    [
        {"benchmark_md": "../BENCHMARK.md"},
        {"benchmark_md": "https://other.example/card"},
        {"benchmark_md": "benchmarks/skill-002/BENCHMARK.md"},
        {"path": "skills/foreign"},
        {"eligible": True},
        {"publication_status": "successful"},
    ],
)
def test_benchmark_links_reject_unsafe_foreign_or_contradictory_entries(change):
    payload = with_card()
    payload["benchmarks"][0].update(change)
    with pytest.raises(ValueError):
        tier_manifest.validate_manifest(payload)


def test_duplicate_card_targets_are_rejected():
    payload = with_card()
    payload["benchmarks"].append(card())
    with pytest.raises(ValueError):
        tier_manifest.validate_manifest(payload)
