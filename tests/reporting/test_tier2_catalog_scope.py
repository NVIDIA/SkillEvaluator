# SPDX-License-Identifier: Apache-2.0
"""Native plugin catalog producers retain their completed scope in reports."""

from copy import deepcopy
from dataclasses import replace

import pytest

from skillevaluator.deduplication.plugin.catalog import LocalCatalog, PluginRegistryEntry
from skillevaluator.deduplication.plugin.catalog_checks import check_catalog_plugins, check_catalog_skills
from skillevaluator.deduplication.plugin.profile import BundledSkill, PluginProfile
from skillevaluator.embedding.extractor import ContentEntry
from skillevaluator.embedding.registry import RegistryEntry
from skillevaluator.reporting.html import HTMLReporter
from skillevaluator.reporting.tier2_views import tier2_scope_views


class FixtureEmbedding:
    """Deterministic vectors; the real catalog scorer and producers still run."""

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def _catalog_results(tmp_path, *, matches):
    root = tmp_path / "target"
    member = ContentEntry("member", "Target member", str(root / "skills/member"), "skill", "Run the task.")
    profile = PluginProfile(
        root,
        "target",
        "Target plugin",
        ".claude-plugin/plugin.json",
        "a" * 64,
        ("member",),
        (BundledSkill("skills/member", root / "skills/member", member),),
    )
    vector = [1.0, 0.0] if matches else [0.0, 1.0]
    catalog = LocalCatalog(
        skills=(
            RegistryEntry(
                "member",
                "Target member",
                "source/skills/member",
                "skill",
                [1.0, 0.0],
                "skill:source/skills/member",
                "b" * 64,
            ),
            RegistryEntry(
                "external-member",
                "External member",
                "catalog/skills/external",
                "skill",
                vector,
                "skill:catalog/skills/external",
                "c" * 64,
            ),
        ),
        plugins=(
            PluginRegistryEntry(
                "plugin:source",
                "target",
                "Target plugin",
                "source",
                ".claude-plugin/plugin.json",
                "a" * 64,
                ("member",),
                [1.0, 0.0],
            ),
            PluginRegistryEntry(
                "plugin:catalog/external",
                "external",
                "External plugin",
                "catalog/external",
                ".claude-plugin/plugin.json",
                "d" * 64,
                ("external-member",),
                vector,
            ),
        ),
        vector_dimension=2,
        client=FixtureEmbedding(),
        max_scalar_comparisons=100,
    )
    return profile, catalog


@pytest.mark.parametrize("matches", [False, True])
def test_real_catalog_comparisons_report_completion_and_match_counts(tmp_path, matches):
    profile, catalog = _catalog_results(tmp_path, matches=matches)
    results = [
        check_catalog_skills(profile, catalog, threshold=0.9),
        check_catalog_plugins(profile, catalog, threshold=0.9),
    ]
    before = deepcopy(results)

    across = tier2_scope_views(results)[1]

    assert (across.completed, across.duplicates, across.affected, across.catalog_entries) == (
        2,
        2 if matches else 0,
        None if matches else 0,
        2,
    )
    assert not across.catalog_not_applicable
    assert not any("incomplete" in note for note in across.notes)
    output = HTMLReporter(include_timestamp=False).render_all(results)
    section = output.split('id="tier2-across"', 1)[1].split("</section>", 1)[0]
    assert "Checks completed</dt><dd>2</dd>" in section
    assert f"Similarity matches</dt><dd>{2 if matches else 0}</dd>" in section
    assert f"Affected entries</dt><dd>{'Unavailable' if matches else 0}</dd>" in section
    assert "Catalog entries compared</dt><dd>2</dd>" in section
    assert ("No semantic overlaps detected." in section) is not matches
    assert results == before


def test_real_catalog_findings_remain_recorded_when_a_comparison_is_incomplete(tmp_path):
    profile, catalog = _catalog_results(tmp_path, matches=True)
    results = [
        check_catalog_skills(profile, catalog, threshold=0.9),
        check_catalog_plugins(profile, catalog, threshold=0.9),
    ]
    results[0].mark_scan_incomplete("embedding-provider")

    across = tier2_scope_views(results)[1]

    assert (across.completed, across.duplicates, across.affected, across.catalog_entries) == (None, 2, None, None)
    assert any("incomplete" in note for note in across.notes)


@pytest.mark.parametrize(
    "skill_category,plugin_category,duplicates",
    [
        ("inter_skill", "inter_plugin", 2),
        ("InTeR_sKiLl", "InTeR_pLuGiN", 2),
        ("INTER_\u017fKILL", "INTER_PLUGIN", 2),
        ("INTER_SKILL_extra", "INTER_PLUGIN_extra", 0),
        ("INTER_SK\u0131LL", "INTER_PLUG\u0131N", 0),
    ],
)
def test_catalog_category_matching_preserves_casefold_and_unknown_categories(
    tmp_path, skill_category, plugin_category, duplicates
):
    profile, catalog = _catalog_results(tmp_path, matches=True)
    results = [
        check_catalog_skills(profile, catalog, threshold=0.9),
        check_catalog_plugins(profile, catalog, threshold=0.9),
    ]
    results[0].findings[0].category = skill_category
    results[1].findings[0].category = plugin_category
    before = deepcopy(results)

    across = tier2_scope_views(results)[1]

    assert (across.completed, across.duplicates) == (2, duplicates)
    assert results == before


def test_catalog_and_target_entries_with_the_same_relative_path_remain_distinct(tmp_path):
    profile, catalog = _catalog_results(tmp_path, matches=True)
    external = replace(catalog.skills[1], path="skills/member", entry_id="skill:skills/member")
    catalog = replace(catalog, skills=(external,))
    result = check_catalog_skills(profile, catalog, threshold=0.9)

    across = tier2_scope_views([result])[1]

    assert (across.completed, across.duplicates, across.affected, across.catalog_entries) == (1, 1, 2, 1)


def test_optional_plugin_verdict_status_does_not_hide_completed_catalog_comparison(tmp_path):
    profile, catalog = _catalog_results(tmp_path, matches=True)
    result = check_catalog_plugins(profile, catalog, threshold=0.9)
    # This native producer's optional verdict metadata uses candidates, rather
    # than the mandatory clusters emitted by within-skill content analysis.
    result.metadata["llm_analysis"] = {"candidates_total": 1, "candidates_completed": 0, "candidates_failed": 1}

    across = tier2_scope_views([result])[1]

    assert (across.completed, across.duplicates, across.affected, across.catalog_entries) == (1, 1, None, 1)


@pytest.mark.parametrize(
    "field,value",
    [("catalog_entries", True), ("catalog_entries", -1), ("catalog_entries", 1 << 63), ("self_matches_excluded", 3)],
)
def test_native_catalog_invalid_counts_remain_unknown(tmp_path, field, value):
    profile, catalog = _catalog_results(tmp_path, matches=False)
    result = check_catalog_skills(profile, catalog, threshold=0.9)
    result.success_details[0].metadata[field] = value

    across = tier2_scope_views([result])[1]

    assert (across.completed, across.duplicates, across.affected, across.catalog_entries) == (1, 0, 0, None)


def test_empty_native_catalog_comparisons_remain_optional_skips(tmp_path):
    profile, catalog = _catalog_results(tmp_path, matches=False)
    catalog = replace(catalog, skills=(), plugins=())
    results = [
        check_catalog_skills(profile, catalog, threshold=0.9),
        check_catalog_plugins(profile, catalog, threshold=0.9),
    ]

    across = tier2_scope_views(results)[1]

    assert (across.completed, across.duplicates, across.affected, across.catalog_entries) == (0, None, None, None)
    assert len(across.skips) == 2
