# SPDX-License-Identifier: Apache-2.0
"""Public plugin dispatch preserves catalog compatibility and provider bounds."""

import json
from types import SimpleNamespace

import pytest

from skillevaluator.models.result import ValidationResult
from skillevaluator.tier2 import commands


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


def make_plugin(root, name="demo"):
    manifest = root / ".claude-plugin/plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"name": name, "description": "Reusable project workflow"}))
    member = root / "skills" / f"{name}-member" / "SKILL.md"
    member.parent.mkdir(parents=True)
    member.write_text(f"---\nname: {name}-member\ndescription: Complete the project task\n---\nExecute the task.\n")
    return root


def test_plugin_dispatch_preserves_every_result(tmp_path, monkeypatch):
    root = make_plugin(tmp_path / "plugin")
    expected = [ValidationResult(validator_name="Plugin refs"), ValidationResult(validator_name="Member overlap")]
    seen = []

    def run(path, **kwargs):
        seen.append((path, kwargs))
        return expected

    monkeypatch.setattr("skillevaluator.tier2.plugin_commands.plugin_dedup_results", run)
    result = commands.run_dedup_scan(
        root / ".claude-plugin/plugin.json", threshold=0.7, model="embed", llm_model="judge"
    )
    assert result == expected
    assert seen[0][0] == root
    assert seen[0][1]["threshold"] == 0.7
    assert all(r.metadata["publication_evidence"]["check_id"] == "context-optimization" for r in result)


def test_ordinary_skills_keep_the_existing_validator(tmp_path, monkeypatch):
    root = tmp_path / "skill"
    root.mkdir()
    (root / "SKILL.md").write_text("---\nname: skill\ndescription: Test skill\n---\nRun the task.\n")
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
    root = tmp_path / "collection"
    make_plugin(root / "one", "one")
    make_plugin(root / "two", "two")
    output = tmp_path / "catalog.json"
    result = commands.run_similarity_check(root, content_type="plugin", save_catalog=output)
    assert all(r.passed for r in result), [r.errors for r in result]
    payload = json.loads(output.read_text())
    assert payload["schema_version"] == 2 and len(payload["plugins"]) == 2
    assert len(embedding.requests) == 1
    assert sum(r.validator_name == "Inter-Plugin Deduplication" for r in result) == 2
    assert any(f.category == "INTER_PLUGIN" for r in result for f in r.findings)


def test_existing_v1_skill_catalog_can_be_used_for_a_plugin(tmp_path, embedding):
    from skillevaluator.embedding.registry import EmbeddingRegistry

    collection = tmp_path / "skills-catalog"
    skill = collection / "external"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: external\ndescription: Existing reusable task\n---\nExecute the task.\n"
    )
    registry = EmbeddingRegistry(embedding)
    assert registry.build_from_directory(collection, "skill", minimum_entries=1, for_pairwise_scan=True) == 1
    catalog = tmp_path / "v1.json"
    registry.save_catalog(catalog)
    assert json.loads(catalog.read_text())["schema_version"] == 1
    result = commands.run_similarity_check(make_plugin(tmp_path / "plugin"), catalog=catalog)
    assert all(r.passed for r in result), [r.errors for r in result]
    assert any(f.category == "INTER_SKILL" for r in result for f in r.findings)


def test_full_body_is_refused_without_embedding(tmp_path, embedding):
    result = commands.run_similarity_check(make_plugin(tmp_path / "plugin"), full_body=True)
    assert any("--full-body" in error for r in result for error in r.errors)
    assert embedding.requests == []


def test_source_swapped_during_embedding_is_blocking_and_cannot_save_catalog(tmp_path, embedding):
    root = make_plugin(tmp_path / "plugin")
    member = root / "skills/demo-member/SKILL.md"
    outside = tmp_path / "outside.md"
    outside.write_text(member.read_text())
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
    results = commands.run_similarity_check(root, save_catalog=output)
    assert any(
        result.metadata.get("security_failure") and not result.metadata.get("advisory_tier2") for result in results
    )
    assert not output.exists()


def test_unsafe_source_blocks_before_embedding(tmp_path, embedding):
    root = make_plugin(tmp_path / "plugin")
    outside = tmp_path / "outside.md"
    outside.write_text("Outside source")
    try:
        (root / "unsafe.md").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symbolic links unavailable")
    result = commands.run_similarity_check(root)
    assert any(r.metadata.get("security_failure") and not r.metadata.get("advisory_tier2") for r in result)
    assert embedding.requests == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"catalog": "one.json", "cache": "two.json"},
        {"save_catalog": "one.json", "save_cache": "two.json"},
        {"catalog": "one.json", "save_catalog": "two.json"},
    ],
)
def test_catalog_alias_conflicts_precede_provider_work(tmp_path, embedding, kwargs):
    result = commands.run_similarity_check(
        make_plugin(tmp_path / "plugin"), **{name: tmp_path / value for name, value in kwargs.items()}
    )
    assert any(not r.passed for r in result)
    assert embedding.requests == []
