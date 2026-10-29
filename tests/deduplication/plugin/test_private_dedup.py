# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private bundle comparisons use local fixtures and deterministic providers."""

from __future__ import annotations

import hashlib
import importlib
import json
from types import SimpleNamespace

import pytest

from skillevaluator.embedding.registry import EmbeddingRegistry
from skillevaluator.models.result import Severity


@pytest.fixture(autouse=True)
def forbid_live_providers(monkeypatch):
    from skillevaluator.embedding.client import EmbeddingClient
    from skillevaluator.inference.client import LLMClient

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Live provider calls are forbidden in these tests")

    monkeypatch.setattr(EmbeddingClient, "embed", forbidden)
    monkeypatch.setattr(EmbeddingClient, "embed_chunked", forbidden)
    monkeypatch.setattr(LLMClient, "extract_json_from_response", forbidden)


class FakeEmbedding:
    model = "fixture-model"

    def __init__(self):
        self.calls = []

    def _resolved_config(self):
        return SimpleNamespace(provider="openai", base_url=None)

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    def embed_single(self, text):
        return self.embed([text])[0]


@pytest.fixture
def modules():
    return SimpleNamespace(
        profile=importlib.import_module("skillevaluator.deduplication.plugin.profile"),
        checks=importlib.import_module("skillevaluator.deduplication.plugin.catalog_checks"),
        catalog=importlib.import_module("skillevaluator.deduplication.plugin.catalog"),
        commands=importlib.import_module("skillevaluator.deduplication.plugin.commands"),
    )


def inventory(root, *, skill_paths=("custom/a",), name="bundle", description="A useful bundle", complete=True):
    manifest = root / "agent_plugin.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        manifest.write_text(f"name: {name}\ndescription: {description}\n")
    skills = []
    for rel in skill_paths:
        folder = root / rel
        folder.mkdir(parents=True, exist_ok=True)
        text = "---\nname: same-skill\ndescription: Useful skill\n---\n\n# Guide\n" + "useful words " * 20
        (folder / "SKILL.md").write_text(text)
        skills.append(
            SimpleNamespace(
                type="skill",
                name="same-skill",
                path=rel,
                problem=None,
                effective=True,
                text=text,
                frontmatter={"name": "same-skill", "description": "Useful skill"},
            )
        )
    from skillevaluator.plugin_manifest import locate_plugin_manifest

    selected = locate_plugin_manifest(root)
    return SimpleNamespace(
        root=root,
        selected_manifest=selected,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        selected_skills=skills,
        selected_components=skills,
        components=skills,
        complete=complete,
        tree_complete=complete,
        discovery=SimpleNamespace(findings=[]),
        source_files=(),
    )


def catalog_payload(client, *, version=2, skills=(), plugins=()):
    from skillevaluator.embedding.registry import _client_endpoint_fingerprint

    result = {
        "schema_version": version,
        "provider": "openai",
        "model": client.model,
        "mode": "description",
        "endpoint_fingerprint": _client_endpoint_fingerprint(client),
        "vector_dimension": 2,
        "created_at": "2026-01-01T00:00:00+00:00",
        "entries": list(skills),
    }
    if version == 2:
        result["plugins"] = list(plugins)
    return result


def skill_entry(path, *, fingerprint=None, content_type="skill"):
    return {
        "id": f"{content_type}:{path}",
        "name": "same-skill",
        "description": "Useful skill",
        "path": path,
        "content_type": content_type,
        "content_fingerprint": fingerprint or "b" * 64,
        "embedding": [1.0, 0.0],
    }


def bundle_entry(path, *, name="bundle", fingerprint="c" * 64, members=("same-skill",)):
    return {
        "id": f"plugin:{path}",
        "name": name,
        "description": "A useful bundle",
        "path": path,
        "manifest": "agent_plugin.yaml",
        "source_fingerprint": fingerprint,
        "members": list(members),
        "embedding": [1.0, 0.0],
    }


def load_catalog(modules, tmp_path, *, client=None, version=2, skills=(), plugins=(), **kwargs):
    client = client or FakeEmbedding()
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog_payload(client, version=version, skills=skills, plugins=plugins)))
    return modules.catalog.LocalCatalog.load(path, client=client, **kwargs), client, path


def test_duplicate_refs_keep_occurrences_and_do_not_fetch(modules, tmp_path):
    from skillevaluator.deduplication.plugin.intra_plugin_validator import IntraPluginValidator

    root = tmp_path / "bundle"
    root.mkdir()
    (root / "agent_plugin.yaml").write_text(
        "name: bundle\nskills:\n  refs:\n    - github::owner/repo::skills::one\n"
        "    - {source: GITHUB, repo: owner/repo.git, path: skills/one}\n"
    )
    result = IntraPluginValidator().validate(root)
    assert result.passed
    assert [(f.check_name, f.severity, f.metadata["occurrences"]) for f in result.findings] == [
        ("duplicate_skill_ref", Severity.MEDIUM, 2)
    ]


def test_profile_uses_custom_and_root_skill_paths(modules, tmp_path, monkeypatch):
    inv = inventory(tmp_path / "bundle", skill_paths=("custom/a", "."))
    monkeypatch.setattr(modules.profile, "plugin_inventory_for_root", lambda _path: inv)
    profile = modules.profile.load_plugin_profile(inv.root)
    assert [s.root_relative for s in profile.bundled_skills] == ["custom/a", "."]
    assert profile.members == ("same-skill",)


def test_private_v2_acceptance_does_not_change_public_v1_loader(modules, tmp_path):
    private, client, path = load_catalog(modules, tmp_path, plugins=[bundle_entry("other")])
    assert len(private.plugins) == 1
    with pytest.raises(ValueError):
        EmbeddingRegistry(client).load_catalog(path)


@pytest.mark.parametrize("version", [1, 2])
def test_catalog_projection_filters_non_skills(modules, tmp_path, version):
    catalog, _, _ = load_catalog(
        modules,
        tmp_path,
        version=version,
        skills=[skill_entry("rules/a", content_type="rules"), skill_entry("skills/a")],
    )
    assert [entry.path for entry in catalog.skills] == ["skills/a"]


@pytest.mark.parametrize("field,value", [("model", "wrong"), ("vector_dimension", True), ("mode", "full-body")])
def test_private_catalog_rejects_incompatible_metadata_before_provider(modules, tmp_path, field, value):
    client = FakeEmbedding()
    payload = catalog_payload(client, skills=[skill_entry("a")])
    payload[field] = value
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        modules.catalog.LocalCatalog.load(path, client=client)
    assert client.calls == []


def test_inter_skill_excludes_only_identified_self_paths(modules, tmp_path, monkeypatch):
    inv = inventory(tmp_path / "bundle")
    monkeypatch.setattr(modules.profile, "plugin_inventory_for_root", lambda _path: inv)
    profile = modules.profile.load_plugin_profile(inv.root)
    cat, _, _ = load_catalog(
        modules,
        tmp_path,
        skills=[skill_entry("self/custom/a"), skill_entry("other/custom/a")],
        plugins=[bundle_entry("self", fingerprint=inv.manifest_sha256)],
    )
    result = modules.checks.check_catalog_skills(profile, cat, threshold=0.9)
    assert [f.metadata["catalog_path"] for f in result.findings] == ["other/custom/a"]
    assert result.success_details[0].metadata["excluded_catalog_paths"] == ["self/custom/a"]


def test_v1_self_exclusion_requires_expected_path_and_fingerprint(modules, tmp_path, monkeypatch):
    inv = inventory(tmp_path / "bundle", skill_paths=("skills/a",))
    monkeypatch.setattr(modules.profile, "plugin_inventory_for_root", lambda _path: inv)
    profile = modules.profile.load_plugin_profile(inv.root)
    fingerprint = hashlib.sha256(b"same-skill: Useful skill").hexdigest()
    cat, _, _ = load_catalog(
        modules,
        tmp_path,
        version=1,
        skills=[
            skill_entry("a", fingerprint=fingerprint),
            skill_entry("elsewhere/a", fingerprint=fingerprint),
            skill_entry("skills/a"),
        ],
    )
    result = modules.checks.check_catalog_skills(profile, cat, threshold=0.9)
    assert {f.metadata["catalog_path"] for f in result.findings} == {"elsewhere/a", "skills/a"}


def test_inter_skill_work_budget_counts_all_targets_before_embedding(modules, tmp_path, monkeypatch):
    inv = inventory(tmp_path / "bundle", skill_paths=("skills/a", "skills/b"))
    monkeypatch.setattr(modules.profile, "plugin_inventory_for_root", lambda _path: inv)
    profile = modules.profile.load_plugin_profile(inv.root)
    cat, client, _ = load_catalog(modules, tmp_path, skills=[skill_entry("other")], max_scalar_comparisons=3)
    with pytest.raises(ValueError, match="comparison"):
        modules.checks.check_catalog_skills(profile, cat, threshold=0.9)
    assert client.calls == []


def test_same_name_different_manifest_remains_collision_after_unique_verdict(modules, tmp_path, monkeypatch):
    inv = inventory(tmp_path / "bundle")
    monkeypatch.setattr(modules.profile, "plugin_inventory_for_root", lambda _path: inv)
    profile = modules.profile.load_plugin_profile(inv.root)
    cat, _, _ = load_catalog(
        modules, tmp_path, plugins=[bundle_entry("self", fingerprint=inv.manifest_sha256), bundle_entry("other")]
    )
    from skillevaluator.inference import LLMVerdict

    monkeypatch.setattr(
        modules.checks, "_llm_verdicts", lambda *_args, **_kwargs: [LLMVerdict("UNIQUE", 0.9, "distinct", "")]
    )
    result = modules.checks.check_catalog_plugins(profile, cat, threshold=0.9, llm_verdict=True)
    assert [(f.check_name, f.severity, f.metadata["catalog_path"]) for f in result.findings] == [
        ("NAME_COLLISION", Severity.LOW, "other")
    ]


def test_scan_refreshes_inventory_after_context_work(modules, tmp_path, monkeypatch):
    inv = inventory(tmp_path / "bundle")
    calls = []

    def discover(path):
        calls.append(path)
        return inv

    monkeypatch.setattr(modules.profile, "plugin_inventory_for_root", discover)
    monkeypatch.setattr(modules.commands, "run_plugin_skill_context_dedup", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(modules.commands, "EmbeddingClient", lambda **_kwargs: FakeEmbedding())
    _, _, path = load_catalog(modules, tmp_path, skills=[skill_entry("other")])
    results = modules.commands.run_plugin_dedup_scan(inv.root, catalog=path)
    assert len(calls) == 2
    assert any(f.metadata.get("catalog_path") == "other" for r in results for f in r.findings)


def test_context_skill_limit_refuses_work_before_provider(modules, tmp_path, monkeypatch):
    inv = inventory(tmp_path / "bundle", skill_paths=("a", "b"))
    monkeypatch.setattr(modules.profile, "plugin_inventory_for_root", lambda _path: inv)
    profile = modules.profile.load_plugin_profile(inv.root)
    monkeypatch.setattr(modules.commands, "MAX_PLUGIN_DEDUP_SKILLS", 1)
    result = modules.commands.run_plugin_skill_context_dedup(profile)[0]
    assert result.metadata["work_limit_exceeded"]
    assert result.metadata["scan_status"] == "skipped"


def test_incomplete_inventory_never_reports_clean_comparison(modules, tmp_path, monkeypatch):
    inv = inventory(tmp_path / "bundle", complete=False)
    monkeypatch.setattr(modules.profile, "plugin_inventory_for_root", lambda _path: inv)
    with pytest.raises(ValueError, match="incomplete"):
        modules.profile.load_plugin_profile(inv.root)


def test_optional_verdict_failure_keeps_similarity_finding(modules, tmp_path, monkeypatch):
    inv = inventory(tmp_path / "bundle")
    monkeypatch.setattr(modules.profile, "plugin_inventory_for_root", lambda _path: inv)
    profile = modules.profile.load_plugin_profile(inv.root)
    catalog, _, _ = load_catalog(modules, tmp_path, plugins=[bundle_entry("other", name="other")])
    from skillevaluator import inference

    class Unavailable:
        def __init__(self, **_kwargs):
            raise inference.LLMClientError("No key")

    monkeypatch.setattr(inference, "LLMClient", Unavailable)
    result = modules.checks.check_catalog_plugins(profile, catalog, threshold=0.9, llm_verdict=True)
    assert len(result.findings) == 1
    assert result.metadata["llm_analysis"]["status"] == "skipped"


def test_verdict_prompt_accepts_bounded_long_names_and_newlines(modules, tmp_path, monkeypatch):
    inv = inventory(tmp_path / "bundle", name="long-" + "x" * 100, description="line one\nline two")
    # Write the multiline scalar as YAML rather than the fixture's plain scalar.
    (inv.root / "agent_plugin.yaml").write_text(
        "name: " + "long-" + "x" * 100 + "\ndescription: |\n  line one\n  line two\n"
    )
    from skillevaluator.plugin_manifest import locate_plugin_manifest

    inv.selected_manifest = locate_plugin_manifest(inv.root)
    inv.manifest_sha256 = hashlib.sha256((inv.root / "agent_plugin.yaml").read_bytes()).hexdigest()
    monkeypatch.setattr(modules.profile, "plugin_inventory_for_root", lambda _path: inv)
    profile = modules.profile.load_plugin_profile(inv.root)
    cat, _, _ = load_catalog(modules, tmp_path, plugins=[bundle_entry("other")])
    analyzer = importlib.import_module("skillevaluator.deduplication.plugin.llm_analyzer")
    prompt = analyzer.build_inter_plugin_prompt(
        profile=profile, existing=cat.plugins[0], similarity_score=1.0, member_overlap=1.0
    )
    assert "line one\\nline two" in prompt
    assert "long-" + "x" * 100 in prompt


@pytest.mark.parametrize("confidence", [True, float("nan"), -1, 2, "1"])
def test_optional_verdict_rejects_invalid_confidence(modules, confidence):
    analyzer = importlib.import_module("skillevaluator.deduplication.plugin.llm_analyzer")

    class Client:
        def extract_json_from_response(self, *_args, **_kwargs):
            return {"verdict": "UNIQUE", "confidence": confidence, "reasoning": "distinct", "suggestion": ""}

    with pytest.raises(Exception, match="confidence"):
        analyzer.analyze_inter_plugin(Client(), "Compare bounded bundles")


def test_context_scalar_limit_stops_after_first_provider_batch(tmp_path, monkeypatch):
    from skillevaluator.deduplication.intra_skill import intra_skill_validator as core

    root = tmp_path / "skill"
    root.mkdir()
    (root / "SKILL.md").write_text("\n".join(f"## Section {i}\n" + "words " * 40 for i in range(4)))
    client = FakeEmbedding()
    monkeypatch.setattr(core, "EmbeddingClient", lambda **_kwargs: client)
    monkeypatch.setattr(core, "CONTENT_DEDUP_EMBEDDING_BATCH_SIZE", 2)
    monkeypatch.setattr(core, "CONTENT_DEDUP_MAX_SCALAR_COMPARISONS", 1)
    result = core.IntraSkillValidator().validate(root)
    assert result.findings[0].check_name == "scalar_comparison_limit"
    assert len(client.calls) == 1


def test_context_batch_cardinality_is_validated_before_later_requests(tmp_path, monkeypatch):
    from skillevaluator.deduplication.intra_skill import intra_skill_validator as core

    root = tmp_path / "skill"
    root.mkdir()
    (root / "SKILL.md").write_text("\n".join(f"## Section {i}\n" + "words " * 40 for i in range(4)))
    client = FakeEmbedding()

    def wrong_count(texts):
        client.calls.append(texts)
        return [[1.0, 0.0]]

    client.embed = wrong_count
    monkeypatch.setattr(core, "EmbeddingClient", lambda **_kwargs: client)
    monkeypatch.setattr(core, "CONTENT_DEDUP_EMBEDDING_BATCH_SIZE", 2)
    result = core.IntraSkillValidator().validate(root)
    assert result.status == "incomplete"
    assert len(client.calls) == 1


def test_private_cluster_allowance_keeps_default_skill_behavior(tmp_path, monkeypatch):
    from skillevaluator.deduplication.intra_skill import intra_skill_validator as core
    from skillevaluator.deduplication.intra_skill.semantic_clustering import ContentCluster
    from skillevaluator.deduplication.utils.chunker import ContentChunk

    root = tmp_path / "skill"
    root.mkdir()
    (root / "SKILL.md").write_text("## One\n" + "a" * 200 + "\n## Two\n" + "b" * 200)
    client = FakeEmbedding()
    monkeypatch.setattr(core, "EmbeddingClient", lambda **_kwargs: client)
    chunks = [
        ContentChunk("SKILL.md", "one", 1, 2, "a" * 100, "markdown"),
        ContentChunk("SKILL.md", "two", 3, 4, "b" * 100, "markdown"),
    ]
    monkeypatch.setattr(core, "build_clusters", lambda *_args: [ContentCluster(chunks, 1.0, 1.0, False)] * 2)

    class LLM:
        def __init__(self, **_kwargs):
            pass

        def _resolved_config(self):
            return SimpleNamespace(provider="fixture", model="fixture")

        def extract_json_from_response(self, *_args, **_kwargs):
            return {"verdict": "RELATED_BUT_DISTINCT", "confidence": 0.9, "rationale": "distinct", "suggestion": ""}

    monkeypatch.setattr(core, "LLMClient", LLM)
    default = core.IntraSkillValidator().validate(root)
    constrained = core.IntraSkillValidator(max_llm_clusters=1).validate(root)
    assert default.passed
    assert constrained.findings[0].check_name == "llm_cluster_count_limit"


def test_context_prompt_limit_refuses_llm_before_external_work(tmp_path, monkeypatch):
    from skillevaluator.deduplication.intra_skill import intra_skill_validator as core

    root = tmp_path / "skill"
    root.mkdir()
    (root / "SKILL.md").write_text("## One\n" + "a" * 200 + "\n## Two\n" + "a" * 200)
    monkeypatch.setattr(core, "EmbeddingClient", lambda **_kwargs: FakeEmbedding())
    monkeypatch.setattr(core, "CONTENT_DEDUP_MAX_LLM_PROMPT_CHARS", 100, raising=False)

    def forbidden(**_kwargs):
        raise AssertionError("LLM must not start above the prompt bound")

    monkeypatch.setattr(core, "LLMClient", forbidden)
    result = core.IntraSkillValidator().validate(root)
    assert result.findings[0].check_name == "llm_prompt_size_limit"


def test_clustering_normalizes_once_and_checks_aggregate_scalar_work(monkeypatch):
    from skillevaluator.deduplication.intra_skill import semantic_clustering as core
    from skillevaluator.deduplication.utils.chunker import ContentChunk

    chunks = [ContentChunk("SKILL.md", str(i), 1, 2, "a" * 100, "markdown", [1.0, 0.0]) for i in range(3)]
    monkeypatch.setattr(core, "CONTENT_DEDUP_MAX_SCALAR_COMPARISONS", 5, raising=False)
    with pytest.raises(ValueError, match="scalar"):
        core.build_clusters(chunks)


def test_context_aggregate_chunk_budget_refuses_all_provider_work(modules, tmp_path, monkeypatch):
    inv = inventory(tmp_path / "bundle", skill_paths=("skills/a", "skills/b"))
    monkeypatch.setattr(modules.profile, "plugin_inventory_for_root", lambda _path: inv)
    profile = modules.profile.load_plugin_profile(inv.root)
    for skill in profile.bundled_skills:
        (skill.path / "SKILL.md").write_text("\n".join(f"## Section {i}\n" + "words " * 40 for i in range(2)))
    monkeypatch.setattr(modules.commands, "MAX_PLUGIN_DEDUP_CHUNKS", 3, raising=False)
    from skillevaluator.deduplication.intra_skill import intra_skill_validator as core

    monkeypatch.setattr(core, "EmbeddingClient", lambda **_kwargs: FakeEmbedding())
    result = modules.commands.run_plugin_skill_context_dedup(profile)[0]
    assert result.metadata["work_limit_exceeded"]
    assert result.metadata["actual_chunks"] == 4


def test_unsafe_context_content_remains_blocking(modules, tmp_path, monkeypatch):
    inv = inventory(tmp_path / "bundle")
    monkeypatch.setattr(modules.profile, "plugin_inventory_for_root", lambda _path: inv)
    profile = modules.profile.load_plugin_profile(inv.root)
    from skillevaluator.deduplication.intra_skill import intra_skill_validator as core

    monkeypatch.setattr(core, "EmbeddingClient", lambda **_kwargs: FakeEmbedding())
    external = tmp_path / "outside.md"
    external.write_text("outside")
    (profile.bundled_skills[0].path / "outside.md").symlink_to(external)
    result = modules.commands.run_plugin_skill_context_dedup(profile)[0]
    assert result.metadata["security_failure"]
    assert not result.passed
    assert result.findings[0].file_path == "custom/a/outside.md"


def test_manifest_swap_during_check_a_is_blocking(modules, tmp_path, monkeypatch):
    import skillevaluator.deduplication.plugin.intra_plugin_validator as module
    from skillevaluator.plugin_manifest import locate_plugin_manifest

    inv = inventory(tmp_path / "bundle")

    def swapped(root):
        location = locate_plugin_manifest(root)
        manifest = root / "agent_plugin.yaml"
        manifest.unlink()
        manifest.symlink_to(tmp_path / "external.yaml")
        return location

    (tmp_path / "external.yaml").write_text("name: outside\n")
    monkeypatch.setattr(module, "locate_plugin_manifest", swapped)
    result = module.IntraPluginValidator().validate(inv.root)
    assert result.metadata["security_failure"]
    assert not result.passed


def test_real_inventory_projects_declared_custom_skills(modules, tmp_path):
    root = tmp_path / "bundle"
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin/plugin.json").write_text(
        json.dumps({"name": "bundle", "description": "Bundle", "skills": ["./custom"]})
    )
    skill = root / "custom"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: custom\ndescription: A custom skill\n---\n\n# Use\n")
    profile = modules.profile.load_plugin_profile(root)
    assert [s.root_relative for s in profile.bundled_skills] == ["custom"]


def test_optional_verdict_returns_bounded_parsed_result(modules):
    analyzer = importlib.import_module("skillevaluator.deduplication.plugin.llm_analyzer")

    class Client:
        def extract_json_from_response(self, *_args, **_kwargs):
            return {
                "verdict": "PARTIAL_OVERLAP",
                "confidence": 0.8,
                "reasoning": "Shared work",
                "suggestion": "Consolidate",
            }

    result = analyzer.analyze_inter_plugin(Client(), "Compare bundles")
    assert (result.verdict, result.confidence, result.suggestion) == ("PARTIAL_OVERLAP", 0.8, "Consolidate")


def test_private_catalog_bounds_total_vector_scalars(modules, tmp_path, monkeypatch):
    monkeypatch.setattr(modules.catalog, "MAX_CATALOG_VECTOR_VALUES", 3, raising=False)
    with pytest.raises(ValueError, match="scalar"):
        load_catalog(modules, tmp_path, skills=[skill_entry("a"), skill_entry("b")])


def test_incomplete_inventory_has_a_failed_context_result(modules, tmp_path, monkeypatch):
    inv = inventory(tmp_path / "bundle", complete=False)
    monkeypatch.setattr(modules.profile, "plugin_inventory_for_root", lambda _path: inv)
    results = modules.commands.run_plugin_dedup_scan(inv.root)
    assert any(result.validator_name == "Context Deduplication" and result.is_incomplete for result in results)


def test_catalog_provider_failure_is_an_explicit_skip(modules, tmp_path, monkeypatch):
    from skillevaluator.embedding.client import SimilarityConfigError

    inv = inventory(tmp_path / "bundle")
    monkeypatch.setattr(modules.profile, "plugin_inventory_for_root", lambda _path: inv)

    class Unavailable(FakeEmbedding):
        def embed(self, _texts):
            raise SimilarityConfigError("Fixture provider unavailable")

    monkeypatch.setattr(modules.commands, "EmbeddingClient", lambda **_kwargs: Unavailable())
    _, _, path = load_catalog(modules, tmp_path, skills=[skill_entry("other")])
    results = modules.commands.run_plugin_dedup_scan(inv.root, run_context=False, catalog=path)
    summaries = [r.metadata.get("plugin", {}).get("catalog_skill_similarity") for r in results]
    assert any(summary and summary["status"] == "skipped" for summary in summaries)


def test_private_catalog_builder_saves_v2_bundle_and_skill_entries(modules, tmp_path, monkeypatch):
    collection = tmp_path / "collection"
    first = inventory(collection / "a", skill_paths=("custom/a",), name="same")
    second = inventory(collection / "b", skill_paths=("skills/b",), name="same")
    snapshots = {first.root: first, second.root: second}
    monkeypatch.setattr(modules.profile, "plugin_inventory_for_root", lambda path: snapshots[path])
    client = FakeEmbedding()
    catalog = modules.catalog.LocalCatalog.build(collection, client=client)
    path = tmp_path / "built.json"
    catalog.save(path)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 2
    assert [entry["path"] for entry in payload["plugins"]] == ["a", "b"]
    assert [entry["name"] for entry in payload["plugins"]] == ["same", "same"]
    assert [entry["path"] for entry in payload["entries"]] == ["a/custom/a", "b/skills/b"]
    loaded = modules.catalog.LocalCatalog.load(path, client=client)
    assert len(loaded.plugins) == len(loaded.skills) == 2


def test_private_catalog_build_refuses_entry_overflow_before_embedding(modules, tmp_path, monkeypatch):
    inv = inventory(tmp_path / "bundle", skill_paths=("skills/a", "skills/b"))
    monkeypatch.setattr(modules.profile, "plugin_inventory_for_root", lambda _path: inv)
    client = FakeEmbedding()
    with pytest.raises(ValueError, match="entry limit"):
        modules.catalog.LocalCatalog.build(inv.root, client=client, max_entries=2)
    assert client.calls == []


def test_private_catalog_builder_refuses_vector_budget_before_later_batches(modules, tmp_path, monkeypatch):
    inv = inventory(tmp_path / "bundle", skill_paths=("skills/a", "skills/b"))
    monkeypatch.setattr(modules.profile, "plugin_inventory_for_root", lambda _path: inv)
    monkeypatch.setattr(modules.catalog, "MAX_CATALOG_VECTOR_VALUES", 4)
    monkeypatch.setattr(modules.catalog.core, "EMBEDDING_BATCH_SIZE", 1)
    client = FakeEmbedding()
    with pytest.raises(ValueError, match="scalar"):
        modules.catalog.LocalCatalog.build(inv.root, client=client)
    assert len(client.calls) == 1


def test_private_catalog_builder_bounds_source_bytes_before_embedding(modules, tmp_path, monkeypatch):
    inv = inventory(tmp_path / "bundle")
    monkeypatch.setattr(modules.profile, "plugin_inventory_for_root", lambda _path: inv)
    monkeypatch.setattr(modules.catalog, "MAX_CATALOG_SOURCE_BYTES", 1, raising=False)
    client = FakeEmbedding()
    with pytest.raises(ValueError, match="byte"):
        modules.catalog.LocalCatalog.build(inv.root, client=client)
    assert client.calls == []


def test_private_catalog_save_refuses_unsafe_output_and_invalid_vectors(modules, tmp_path):
    catalog, _, path = load_catalog(modules, tmp_path, skills=[skill_entry("a")], plugins=[bundle_entry("bundle")])
    outside = tmp_path / "outside.json"
    outside.write_text("untouched")
    linked = tmp_path / "linked.json"
    linked.symlink_to(outside)
    with pytest.raises(ValueError):
        catalog.save(linked)
    assert outside.read_text() == "untouched"
    original = path.read_bytes()
    catalog.skills[0].embedding[0] = float("nan")
    with pytest.raises(ValueError):
        catalog.save(path)
    assert path.read_bytes() == original


def test_private_save_keeps_v1_rules_and_skill_entries_compatible(modules, tmp_path):
    catalog, client, _ = load_catalog(
        modules, tmp_path, version=1, skills=[skill_entry("rules/a", content_type="rules"), skill_entry("skills/a")]
    )
    path = tmp_path / "saved-v1.json"
    catalog.save(path)
    registry = EmbeddingRegistry(client)
    registry.load_catalog(path)
    assert registry.size == 2


def test_private_catalog_build_validates_metadata_before_embedding(modules, tmp_path, monkeypatch):
    inv = inventory(tmp_path / "bundle")
    inv.selected_skills[0].frontmatter["name"] = "bad\nname"
    monkeypatch.setattr(modules.profile, "plugin_inventory_for_root", lambda _path: inv)
    client = FakeEmbedding()
    with pytest.raises(ValueError):
        modules.catalog.LocalCatalog.build(inv.root, client=client)
    assert client.calls == []


def test_disappearing_inventory_after_safe_manifest_is_a_secure_refusal(modules, tmp_path, monkeypatch):
    from skillevaluator.utils.secure_fs import SecurePathError

    inv = inventory(tmp_path / "bundle")
    monkeypatch.setattr(modules.profile, "plugin_inventory_for_root", lambda _path: None)
    with pytest.raises(SecurePathError):
        modules.profile.load_plugin_profile(inv.root)


def test_catalog_builder_does_not_skip_an_unsafe_source_tree(modules, tmp_path, monkeypatch):
    from skillevaluator.utils.secure_fs import SecurePathError

    inv = inventory(tmp_path / "bundle", complete=False)
    monkeypatch.setattr(modules.profile, "plugin_inventory_for_root", lambda _path: inv)
    client = FakeEmbedding()
    with pytest.raises(SecurePathError):
        modules.catalog.LocalCatalog.build(inv.root, client=client)
    assert client.calls == []


def test_post_context_source_swap_blocks_catalog_provider_work(modules, tmp_path, monkeypatch):
    inv = inventory(tmp_path / "bundle", skill_paths=("skills/a",))
    client = FakeEmbedding()
    outside = tmp_path / "outside.md"
    outside.write_text("outside")

    def provider_then_swap(_profile, **_kwargs):
        client.embed(["simulated context chunk"])
        (inv.root / "skills/a/link.md").symlink_to(outside)
        return []

    monkeypatch.setattr(modules.commands, "run_plugin_skill_context_dedup", provider_then_swap)
    monkeypatch.setattr(modules.commands, "EmbeddingClient", lambda **_kwargs: client)
    _, _, path = load_catalog(modules, tmp_path, client=client, skills=[skill_entry("other")])
    results = modules.commands.run_plugin_dedup_scan(inv.root, catalog=path)
    blocked = [result for result in results if result.metadata.get("security_failure")]
    assert blocked and all(not result.passed for result in blocked)
    assert len(client.calls) == 1
