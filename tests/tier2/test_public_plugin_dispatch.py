# SPDX-License-Identifier: Apache-2.0
"""Public plugin dispatch preserves catalog compatibility and provider bounds."""

import json
from types import SimpleNamespace

import pytest

from skillevaluator.models.result import ValidationResult
from skillevaluator.publication_identity import publication_target_from_path
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
    manifest.write_text(json.dumps({"name": name, "description": "Reusable project workflow"}), encoding="utf-8")
    member = root / "skills" / f"{name}-member" / "SKILL.md"
    member.parent.mkdir(parents=True)
    member.write_text(
        f"---\nname: {name}-member\ndescription: Complete the project task\n---\nExecute the task.\n", encoding="utf-8"
    )
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
    root = tmp_path / "collection"
    make_plugin(root / "one", "one")
    make_plugin(root / "two", "two")
    output = tmp_path / "catalog.json"
    result = commands.run_similarity_check(root, content_type="plugin", save_catalog=output)
    assert all(r.passed for r in result), [r.errors for r in result]
    payload = json.loads(output.read_text(encoding="utf-8"))
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
        "---\nname: external\ndescription: Existing reusable task\n---\nExecute the task.\n", encoding="utf-8"
    )
    registry = EmbeddingRegistry(embedding)
    assert registry.build_from_directory(collection, "skill", minimum_entries=1, for_pairwise_scan=True) == 1
    catalog = tmp_path / "v1.json"
    registry.save_catalog(catalog)
    assert json.loads(catalog.read_text(encoding="utf-8"))["schema_version"] == 1
    result = commands.run_similarity_check(make_plugin(tmp_path / "plugin"), catalog=catalog)
    assert all(r.passed for r in result), [r.errors for r in result]
    assert any(f.category == "INTER_SKILL" for r in result for f in r.findings)


def test_full_body_is_refused_without_embedding(tmp_path, embedding):
    result = commands.run_similarity_check(make_plugin(tmp_path / "plugin"), full_body=True)
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
    results = commands.run_similarity_check(target, save_catalog=output)
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
    result = commands.run_similarity_check(target)
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


@pytest.mark.parametrize("save_option", ["save_catalog", "save_cache"])
def test_explicit_manifest_catalog_uses_selected_embeddings_and_fresh_profile(tmp_path, embedding, save_option):
    selected = make_dual_profile_plugin(tmp_path / "plugin")
    output = tmp_path / "catalog.json"
    results = commands.run_similarity_check(selected, **{save_option: output})
    assert all(result.passed for result in results), [result.errors for result in results]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert [(entry["name"], entry["manifest"], entry["path"]) for entry in payload["plugins"]] == [
        ("codex-demo", ".codex-plugin/plugin.json", ".")
    ]
    assert [(entry["name"], entry["path"]) for entry in payload["entries"]] == [("codex-member", "codex-skills/member")]
    assert embedding.requests == [["codex-member: Codex task", "codex-demo: Codex workflow"]]
    targets = [
        result.metadata["plugin_catalog_target"] for result in results if "plugin_catalog_target" in result.metadata
    ]
    assert targets == [{"name": "codex-demo", "path": "."}, {"name": "codex-demo", "path": "."}]
    expected_identity = publication_target_from_path(selected.parent.parent)
    assert expected_identity is not None
    assert all(result.metadata.get("publication_target") == expected_identity for result in results)


def test_explicit_manifest_catalog_query_keeps_selected_profile(tmp_path, embedding):
    catalog = tmp_path / "catalog.json"
    commands.run_similarity_check(make_plugin(tmp_path / "external", "external"), save_catalog=catalog)
    embedding.requests.clear()
    selected = make_dual_profile_plugin(tmp_path / "plugin")
    results = commands.run_similarity_check(selected, catalog=catalog)
    assert all(result.passed for result in results), [result.errors for result in results]
    texts = [text for request in embedding.requests for text in request]
    assert "codex-demo: Codex workflow" in texts
    assert "codex-member: Codex task" in texts
    assert all("claude" not in text.casefold() for text in texts)


def test_explicit_manifest_dedup_uses_selected_members(tmp_path, monkeypatch):
    selected = make_dual_profile_plugin(tmp_path / "plugin")
    monkeypatch.setattr(
        "skillevaluator.deduplication.plugin.commands.IntraSkillValidator.validate",
        lambda *_args, **_kwargs: ValidationResult(validator_name="Context Deduplication"),
    )
    results = commands.run_dedup_scan(selected)
    assert [result.metadata["bundled_skill"] for result in results if "bundled_skill" in result.metadata] == [
        "codex-skills/member"
    ]
    expected_identity = publication_target_from_path(selected.parent.parent)
    assert expected_identity is not None
    assert all(result.metadata.get("publication_target") == expected_identity for result in results)


def test_dedup_without_catalog_reports_comparisons_as_skipped(tmp_path, monkeypatch):
    from skillevaluator.reporting.json_reporter import JSONReporter
    from skillevaluator.reporting.run_status import result_status

    def forbidden(**_kwargs):
        raise AssertionError("This small plugin must not need a live provider")

    monkeypatch.setattr("skillevaluator.deduplication.intra_skill.intra_skill_validator.EmbeddingClient", forbidden)
    results = commands.run_dedup_scan(make_plugin(tmp_path / "plugin"))

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
    results = commands.run_dedup_scan(make_plugin(tmp_path / "plugin"), catalog=tmp_path / "missing.json")

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
    results = commands.run_dedup_scan(root)
    payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    context = payload["results"][1]

    assert context["status"] == "incomplete"
    assert context["passed"] is False
    assert context["skipped"] is False
    assert context["optional_skip"] is False
    assert context["summary"]["checks_performed"] == 0
    assert context["work_limit"]["actual_chunks"] > context["work_limit"]["chunk_limit"] == 0
    assert result_status(retained_results(payload)[1]).value == "incomplete"


def test_combined_validate_keeps_explicit_manifest_for_dedup(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from skillevaluator.cli import cli

    selected = make_dual_profile_plugin(tmp_path / "plugin")
    seen = []

    def dedup(path):
        seen.append(path)
        return [ValidationResult(validator_name="Context Deduplication")]

    monkeypatch.setattr("skillevaluator.cli._run_dedup_or_skip", dedup)
    CliRunner().invoke(
        cli,
        [
            "validate",
            str(selected),
            "--tiers",
            "1,2",
            "--checks",
            "schema",
            "-r",
            "json",
            "-o",
            str(tmp_path / "reports"),
        ],
    )
    assert seen == [selected]


def _skill(root):
    root.mkdir()
    (root / "SKILL.md").write_text("---\nname: skill\ndescription: Test skill\n---\nRun the task.\n", encoding="utf-8")
    return root


@pytest.mark.parametrize(
    "command",
    [
        ["dedup-scan", "{target}", "--catalog", "{catalog}"],
        ["dedup-scan", "{target}", "--llm"],
        ["similarity-check", "{target}", "--llm"],
        ["similarity-check", "{target}", "--llm-model", "judge"],
    ],
)
def test_plugin_only_tier2_options_are_refused_for_skills(tmp_path, monkeypatch, command):
    from click.testing import CliRunner

    from skillevaluator.cli import cli

    skill = _skill(tmp_path / "skill")
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")

    def forbidden(**_kwargs):
        raise AssertionError("A refused option must not start a Tier 2 check")

    monkeypatch.setattr(commands, "IntraSkillValidator", forbidden)
    monkeypatch.setattr(commands, "SimilarityValidator", forbidden)
    args = [part.format(target=skill, catalog=catalog) for part in command]
    result = CliRunner().invoke(cli, ["tier2", *args])
    assert result.exit_code == 2, result.output
    assert "require a plugin target" in result.output


def test_plugin_only_tier2_options_still_reach_plugin_dedup(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from skillevaluator.cli import cli

    root = make_plugin(tmp_path / "plugin")
    seen = []

    def run(path, **kwargs):
        seen.append(kwargs)
        return [ValidationResult(validator_name="Plugin refs")]

    monkeypatch.setattr("skillevaluator.tier2.plugin_commands.plugin_dedup_results", run)
    result = CliRunner().invoke(cli, ["tier2", "dedup-scan", str(root), "--llm", "-o", str(tmp_path / "reports")])
    assert result.exit_code == 0, result.output
    assert seen[0]["llm_verdict"] is True


def test_combined_validate_keeps_the_tier2_setup_guard_for_plugins(tmp_path, monkeypatch):
    from skillevaluator import cli as cli_module
    from skillevaluator.provider_config import ProviderConfigurationError

    root = make_plugin(tmp_path / "plugin")
    seen = []

    def run(path, **kwargs):
        seen.append(path)
        return [ValidationResult(validator_name="Plugin refs")]

    def unconfigured():
        raise ProviderConfigurationError("No provider is configured.")

    monkeypatch.setattr("skillevaluator.tier2.plugin_commands.plugin_dedup_results", run)
    monkeypatch.setattr("skillevaluator.provider_config.resolve_embedding_provider", unconfigured)
    skipped = cli_module._run_dedup_or_skip(root)
    assert seen == []
    assert [result.validator_name for result in skipped] == ["Tier 2 Deduplication"]
    assert skipped[0].metadata["skipped"] is True
    assert "embedding setup is required" in skipped[0].warnings[0]

    monkeypatch.setattr("skillevaluator.provider_config.resolve_embedding_provider", lambda: None)
    manifest = root / ".claude-plugin/plugin.json"
    assert [result.validator_name for result in cli_module._run_dedup_or_skip(manifest)] == ["Plugin refs"]
    assert seen == [manifest]
