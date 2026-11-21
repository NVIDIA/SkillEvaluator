# SPDX-License-Identifier: Apache-2.0
"""Public plugin dispatch preserves catalog compatibility and provider bounds."""

import json
import shutil
from types import SimpleNamespace

import pytest

from skillevaluator.constants import SIMILARITY_DEFAULT_MAX_ENTRIES, SIMILARITY_DEFAULT_MAX_SCALAR_COMPARISONS
from skillevaluator.models.result import ValidationResult
from skillevaluator.tier2 import commands, plugin_commands


class FakeEmbedding:
    model = "fixture-embedding"

    def __init__(self):
        self.requests = []

    def _resolved_config(self):
        return SimpleNamespace(provider="openai", base_url="https://embedding.example.test/v1")

    def embed(self, texts):
        self.requests.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


@pytest.fixture
def embedding(monkeypatch):
    client = FakeEmbedding()
    monkeypatch.setattr("skillevaluator.embedding.client.EmbeddingClient", lambda **_kwargs: client)
    return client


def make_plugin(root, name="demo", *, description="Reusable project workflow"):
    manifest = root / ".claude-plugin/plugin.json"
    manifest.parent.mkdir(parents=True)
    data = {"name": name} if description is None else {"name": name, "description": description}
    manifest.write_text(json.dumps(data), encoding="utf-8")
    member = root / "skills" / f"{name}-member" / "SKILL.md"
    member.parent.mkdir(parents=True)
    member.write_text(
        f"---\nname: {name}-member\ndescription: Complete the project task\n---\nExecute the task.\n", encoding="utf-8"
    )
    return root


def similarity(target, **options):
    """Call the plugin catalog engine with the Tier 2 similarity defaults."""
    defaults = {
        "threshold": 0.75,
        "full_body": False,
        "model": None,
        "catalog": None,
        "save_catalog": None,
        "max_entries": SIMILARITY_DEFAULT_MAX_ENTRIES,
        "max_scalar_comparisons": SIMILARITY_DEFAULT_MAX_SCALAR_COMPARISONS,
    }
    return plugin_commands.plugin_similarity_results(target, **{**defaults, **options})


def test_plugin_dispatch_preserves_every_result(tmp_path, monkeypatch):
    root = make_plugin(tmp_path / "plugin")
    expected = [ValidationResult(validator_name="Plugin refs"), ValidationResult(validator_name="Member overlap")]
    seen = []

    def run(path, **kwargs):
        seen.append((path, kwargs))
        return expected

    monkeypatch.setattr("skillevaluator.tier2.plugin_commands.plugin_dedup_results", run)
    result = plugin_commands.run_public_plugin_dedup(
        root / ".claude-plugin/plugin.json", threshold=0.7, model="embed", llm_model="judge"
    )
    assert result == expected
    assert seen[0][0] == root / ".claude-plugin/plugin.json"
    assert seen[0][1]["threshold"] == 0.7
    assert all(r.metadata["publication_evidence"]["check_id"] == "context-optimization" for r in result)


def test_ordinary_skills_keep_the_existing_validator(tmp_path, monkeypatch):
    root = tmp_path / "skill"
    root.mkdir()
    (root / "SKILL.md").write_text("---\nname: skill\ndescription: Test skill\n---\nRun the task.\n", encoding="utf-8")
    expected = ValidationResult(validator_name="Context Deduplication")
    seen = []

    class Validator:
        def __init__(self, **kwargs):
            seen.append(kwargs)

        def validate(self, path):
            assert path == root
            return expected

    monkeypatch.setattr(commands, "IntraSkillValidator", Validator)
    assert commands.run_dedup_scan(root, threshold=0.73, model="embed", llm_model="judge") == [expected]
    assert seen == [{"threshold": 0.73, "embedding_model": "embed", "llm_model": "judge"}]


def test_collection_v2_catalog_reuses_fresh_vectors(tmp_path, embedding):
    from skillevaluator.reporting.tier2_views import tier2_scope_views

    root = tmp_path / "collection"
    make_plugin(root / "one", "one")
    make_plugin(root / "two", "two")
    output = tmp_path / "catalog.json"
    result = similarity(root, save_catalog=output)
    assert all(r.passed for r in result), [r.errors for r in result]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2 and len(payload["plugins"]) == 2
    assert len(embedding.requests) == 1
    assert sum(r.validator_name == "Inter-Plugin Deduplication" for r in result) == 2
    assert any(f.category == "INTER_PLUGIN" for r in result for f in r.findings)
    across = tier2_scope_views(result)[1]
    # The catalog setup record counts with the four comparisons; it adds no catalog entries.
    assert (across.completed, across.duplicates, across.affected, across.catalog_entries) == (5, 4, 4, 4)


def test_existing_v1_skill_catalog_can_be_used_for_a_plugin(tmp_path, embedding):
    from skillevaluator.embedding.registry import EmbeddingRegistry

    collection = tmp_path / "skills-catalog"
    skill = collection / "external"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: external\ndescription: Existing reusable task\n---\nExecute the task.\n", encoding="utf-8"
    )
    registry = EmbeddingRegistry(embedding)
    assert registry.build_from_directory(collection, "skill", minimum_entries=1, for_pairwise_scan=True) == 1
    catalog = tmp_path / "v1.json"
    registry.save_catalog(catalog)
    assert json.loads(catalog.read_text(encoding="utf-8"))["schema_version"] == 1
    result = similarity(make_plugin(tmp_path / "plugin"), catalog=catalog)
    assert all(r.passed for r in result), [r.errors for r in result]
    assert any(f.category == "INTER_SKILL" for r in result for f in r.findings)


def test_full_body_is_refused_without_embedding(tmp_path, embedding):
    result = similarity(make_plugin(tmp_path / "plugin"), full_body=True)
    assert any("--full-body" in error for r in result for error in r.errors)
    assert embedding.requests == []


@pytest.mark.parametrize("explicit_manifest", [False, True])
def test_source_swapped_during_embedding_is_blocking_and_cannot_save_catalog(tmp_path, embedding, explicit_manifest):
    root = make_plugin(tmp_path / "plugin")
    member = root / "skills/demo-member/SKILL.md"
    outside = tmp_path / "outside.md"
    outside.write_text(member.read_text(encoding="utf-8"), encoding="utf-8")
    output = tmp_path / "catalog.json"
    original = embedding.embed

    def replace_source(texts):
        vectors = original(texts)
        member.unlink()
        try:
            member.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("Symbolic links unavailable")
        return vectors

    embedding.embed = replace_source
    target = root / ".claude-plugin/plugin.json" if explicit_manifest else root
    results = similarity(target, save_catalog=output)
    assert any(
        result.metadata.get("security_failure") and not result.metadata.get("advisory_tier2") for result in results
    )
    assert not output.exists()


@pytest.mark.parametrize("explicit_manifest", [False, True])
def test_unsafe_source_blocks_before_embedding(tmp_path, embedding, explicit_manifest):
    root = make_plugin(tmp_path / "plugin")
    outside = tmp_path / "outside.md"
    outside.write_text("Outside source", encoding="utf-8")
    try:
        (root / "unsafe.md").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symbolic links unavailable")
    target = root / ".claude-plugin/plugin.json" if explicit_manifest else root
    result = similarity(target)
    assert any(r.metadata.get("security_failure") and not r.metadata.get("advisory_tier2") for r in result)
    assert embedding.requests == []


def make_dual_profile_plugin(root):
    for vendor in ("claude", "codex"):
        manifest = root / f".{vendor}-plugin/plugin.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps(
                {
                    "name": f"{vendor}-demo",
                    "description": f"{vendor.title()} workflow",
                    "version": "1.0.0",
                    "skills": f"./{vendor}-skills",
                }
            ),
            encoding="utf-8",
        )
        member = root / f"{vendor}-skills/member/SKILL.md"
        member.parent.mkdir(parents=True)
        member.write_text(
            f"---\nname: {vendor}-member\ndescription: {vendor.title()} task\n---\nExecute the task.\n",
            encoding="utf-8",
        )
    return root / ".codex-plugin/plugin.json"


def test_local_catalog_build_preserves_explicit_manifest_selection(tmp_path, embedding):
    from skillevaluator.deduplication.plugin.catalog import LocalCatalog

    selected = make_dual_profile_plugin(tmp_path / "plugin")
    catalog = LocalCatalog.build(selected, client=embedding)
    assert [(entry.name, entry.path, entry.manifest) for entry in catalog.plugins] == [
        ("codex-demo", ".", ".codex-plugin/plugin.json")
    ]
    assert [(entry.name, entry.path) for entry in catalog.skills] == [("codex-member", "codex-skills/member")]
    assert embedding.requests == [["codex-member: Codex task", "codex-demo: Codex workflow"]]


def test_dedup_without_catalog_reports_comparisons_as_skipped(tmp_path, monkeypatch):
    from skillevaluator.reporting.json_reporter import JSONReporter
    from skillevaluator.reporting.run_status import result_status

    def forbidden(**_kwargs):
        raise AssertionError("This small plugin must not need a live provider")

    monkeypatch.setattr("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient", forbidden)
    results = plugin_commands.run_public_plugin_dedup(make_plugin(tmp_path / "plugin"))

    assert [result_status(result).value for result in results] == ["passed", "passed", "skipped", "skipped"]
    payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    assert payload["total_skipped"] == 2
    for row in payload["results"][2:]:
        assert row["status"] == "skipped"
        assert row["skipped"] is True
        assert row["optional_skip"] is True
        assert row["skip_reason"] == "No local catalog supplied"
        assert row["summary"]["checks_performed"] == 0
        assert row["success_details"] == []


def test_dedup_with_unavailable_catalog_keeps_comparisons_required(tmp_path, monkeypatch):
    from skillevaluator.reporting.run_status import result_status

    monkeypatch.setattr(
        "skillevaluator.deduplication.plugin.commands.EmbeddingClient", lambda **_kwargs: FakeEmbedding()
    )
    results = plugin_commands.run_public_plugin_dedup(
        make_plugin(tmp_path / "plugin"), catalog=tmp_path / "missing.json"
    )

    assert all(result.metadata.get("optional_skip") is False for result in results[-2:])
    assert all(result_status(result).value == "skipped" for result in results[-2:])
    assert all("Local catalog comparison unavailable" in result.warnings[0] for result in results[-2:])


def test_dedup_budget_stop_is_incomplete_in_public_json(tmp_path, monkeypatch):
    from skillevaluator.reporting.json_reporter import JSONReporter
    from skillevaluator.reporting.retained import retained_results
    from skillevaluator.reporting.run_status import result_status

    root = make_plugin(tmp_path / "plugin")
    (root / "skills/demo-member/SKILL.md").write_text(
        "---\nname: demo-member\ndescription: Project workflow\n---\n" + "Useful content " * 30
    )
    monkeypatch.setattr("skillevaluator.deduplication.plugin.commands.MAX_PLUGIN_DEDUP_CHUNKS", 0)
    results = plugin_commands.run_public_plugin_dedup(root)
    payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    context = payload["results"][1]

    assert context["status"] == "incomplete"
    assert context["passed"] is False
    assert context["skipped"] is False
    assert context["optional_skip"] is False
    assert context["summary"]["checks_performed"] == 0
    assert context["work_limit"]["actual_chunks"] > context["work_limit"]["chunk_limit"] == 0
    assert result_status(retained_results(payload)[1]).value == "incomplete"


def _comparisons(results, name):
    return {r.metadata["plugin_catalog_target"]["path"]: r for r in results if r.validator_name == name}


@pytest.mark.parametrize(("other_description", "schema_version"), [("Other workflow", 2), (None, 1)])
def test_collection_without_bundle_description_excludes_only_its_own_entries(
    tmp_path, embedding, other_description, schema_version
):
    root = tmp_path / "collection"
    make_plugin(root / "one", "one", description=other_description)
    make_plugin(root / "two", "two", description=None)
    output = tmp_path / "catalog.json"

    results = similarity(root, save_catalog=output)

    assert all(r.passed for r in results), [r.errors for r in results]
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == schema_version
    skills = _comparisons(results, "Inter-Skill Deduplication")
    assert sorted(skills) == ["one", "two"]
    for target, result in skills.items():
        own = f"{target}/skills/{target}-member"
        comparison = next(d.metadata for d in result.success_details if d.check_name == "skill_catalog_comparison")
        assert (comparison["self_matches_excluded"], comparison["excluded_catalog_paths"]) == (1, [own])
        assert [f.metadata["catalog_path"] for f in result.findings] == [
            f"{'two' if target == 'one' else 'one'}/skills/{'two' if target == 'one' else 'one'}-member"
        ]


def test_collection_reports_an_identical_bundle_copy(tmp_path, embedding):
    root = tmp_path / "collection"
    make_plugin(root / "one", "one")
    shutil.copytree(root / "one", root / "one-fork")
    make_plugin(root / "other", "other")

    results = similarity(root)

    assert all(r.passed for r in results), [r.errors for r in results]
    for target, copy in (("one", "one-fork"), ("one-fork", "one")):
        skill = _comparisons(results, "Inter-Skill Deduplication")[target]
        bundle = _comparisons(results, "Inter-Plugin Deduplication")[target]
        assert f"{copy}/skills/one-member" in {f.metadata["catalog_path"] for f in skill.findings}
        assert f"{target}/skills/one-member" not in {f.metadata["catalog_path"] for f in skill.findings}
        copies = [f for f in bundle.findings if f.metadata["catalog_path"] == copy]
        assert len(copies) == 1 and copies[0].category == "INTER_PLUGIN"
        assert copies[0].message.endswith("; same name and identical manifest")
        comparison = next(d.metadata for d in bundle.success_details if d.check_name == "plugin_catalog_comparison")
        assert comparison["excluded_catalog_paths"] == [target]


def test_optional_verdicts_share_one_allowance_across_a_collection(tmp_path, embedding, monkeypatch):
    from skillevaluator import inference
    from skillevaluator.deduplication.plugin.llm_analyzer import MAX_RUN_PROMPT_CHARS, MAX_RUN_VERDICT_CALLS

    requests = []

    class Verdicts:
        def __init__(self, **_kwargs):
            pass

        def extract_json_from_response(self, system, prompt):
            requests.append(len(system) + len(prompt))
            return {"verdict": "PARTIAL_OVERLAP", "confidence": 0.5, "reasoning": "Shared", "suggestion": "Review"}

    monkeypatch.setattr(inference, "LLMClient", Verdicts)
    root = tmp_path / "collection"
    for index in range(15):
        make_plugin(root / f"b{index:02d}", f"b{index:02d}")

    results = similarity(root, llm_verdict=True)

    assert 0 < len(requests) <= MAX_RUN_VERDICT_CALLS
    assert sum(requests) <= MAX_RUN_PROMPT_CHARS
    bundles = [r for r in results if r.validator_name == "Inter-Plugin Deduplication"]
    refused = [r for r in bundles if r.metadata["llm_analysis"].get("status") == "skipped"]
    assert len(refused) == 15 - MAX_RUN_VERDICT_CALLS // 10
    assert all(any("allowance is used up" in w for w in r.warnings) for r in refused)
    assert all(r.findings for r in refused)


@pytest.mark.parametrize(
    "options",
    [
        {"max_chunks": 0},
        {"max_chunks": True},
        {"max_chunks": 1.5},
        {"max_scalar_comparisons": 0},
        {"max_chunks": 0, "run_context": False},
    ],
)
def test_invalid_context_limits_are_caller_errors(tmp_path, options):
    with pytest.raises(ValueError, match="must be an integer within"):
        plugin_commands.run_public_plugin_dedup(make_plugin(tmp_path / "plugin"), **options)


@pytest.mark.parametrize("provider", ["available", "failed"])
def test_catalog_summary_is_tier2_similarity_evidence(tmp_path, embedding, provider):
    from skillevaluator.embedding.client import SimilarityConfigError
    from skillevaluator.reporting.html import HTMLReporter
    from skillevaluator.reporting.run_status import result_tier
    from skillevaluator.reporting.tier2_views import tier2_scope_views
    from skillevaluator.tier2.results import guarded_results

    root = tmp_path / "collection"
    make_plugin(root / "one", "one")
    make_plugin(root / "two", "two")
    if provider == "failed":

        def unavailable(_texts):
            raise SimilarityConfigError("Provider unavailable")

        embedding.embed = unavailable

    results = guarded_results("Similarity Check", root, lambda: similarity(root), check_id="similarity")

    summary = results[0]
    assert summary.validator_name == plugin_commands.CATALOG_SUMMARY_NAME
    assert all(result_tier(r) == 2 for r in results)
    views = tier2_scope_views(results)
    assert [view.id for view in views] == ["within", "across"]
    across = views[1]
    assert summary in across.results and not across.not_run
    section = HTMLReporter(include_timestamp=False).render_all(results).split('id="tier2-across"', 1)[1]
    section = section.split("</section>", 1)[0]
    if provider == "failed":
        assert results == [summary] and summary.is_incomplete
        assert (across.completed, across.catalog_entries) == (None, None)
        assert "Embedding provider error" in section
    else:
        assert (across.completed, across.duplicates, across.affected, across.catalog_entries) == (5, 4, 4, 4)
        assert not any("Pairwise" in note or "not applicable" in note for note in across.notes)


def test_loaded_catalog_summary_counts_as_completed_similarity_evidence(tmp_path, embedding):
    from skillevaluator.reporting.tier2_views import tier2_scope_views

    catalog = tmp_path / "catalog.json"
    similarity(make_plugin(tmp_path / "source", "source"), save_catalog=catalog)

    results = similarity(make_plugin(tmp_path / "target", "target"), catalog=catalog)

    assert all(r.passed for r in results), [r.errors for r in results]
    across = tier2_scope_views(results)[1]
    assert across.results[0].validator_name == plugin_commands.CATALOG_SUMMARY_NAME
    assert (across.completed, across.duplicates, across.catalog_entries) == (3, 2, 2)
