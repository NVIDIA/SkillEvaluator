# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local-catalog plugin Tier 2: Check C-inter and Check B (advisory, offline catalog)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

import skillevaluator.embedding.registry as registry_module
from skillevaluator.embedding.client import EmbeddingClient
from skillevaluator.embedding.registry import EmbeddingRegistry
from skillevaluator.models.result import Severity
from skillevaluator.tier2.commands import run_plugin_catalog_checks, run_plugin_dedup_scan
from skillevaluator.validators.similarity import SimilarityValidator

_VOCAB = ("deploy", "kubernetes", "cluster", "docs", "markdown", "confluence", "review", "code", "gpu", "audit")
_METADATA_KEYS = {"status", "catalog_entries", "matches", "reason"}


def _vector(text: str) -> list[float]:
    words = text.lower().replace(":", " ").replace(",", " ").split()
    return [float(sum(1 for word in words if word.startswith(term))) for term in _VOCAB] + [0.05]


@pytest.fixture(autouse=True)
def embed_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Deterministic local bag-of-words embeddings; no provider request leaves the test."""
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "SKILL_EVAL_EMBEDDING_PROVIDER", "SKILL_EVAL_EMBEDDING_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-credential")
    calls: list[list[str]] = []

    def embed(_self: EmbeddingClient, texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [_vector(text) for text in texts]

    def embed_single(_self: EmbeddingClient, text: str) -> list[float]:
        calls.append([text])
        return _vector(text)

    monkeypatch.setattr(EmbeddingClient, "embed", embed)
    monkeypatch.setattr(EmbeddingClient, "embed_single", embed_single)
    return calls


def _plugin(
    root: Path,
    name: str,
    description: str | None,
    skills: list[tuple[str, str]] = (),  # type: ignore[assignment]
    refs: list[str] = (),  # type: ignore[assignment]
) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    lines = [f"name: {name}", "author: {email: dev@example.com}"]
    if description is not None:
        lines.insert(1, f"description: {description}")
    if refs:
        lines.append("skills:\n  refs:\n" + "".join(f"    - {ref}\n" for ref in refs))
    (directory / "agent_plugin.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for skill_name, skill_description in skills:
        skill_dir = directory / "skills" / skill_name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {skill_name}\ndescription: {skill_description}\n---\n# {skill_name}\n",
            encoding="utf-8",
        )
    return directory


@pytest.fixture
def plugins(tmp_path: Path) -> Path:
    root = tmp_path / "plugins"
    _plugin(
        root,
        "alpha",
        "Deploy kubernetes cluster workloads",
        [("deploy-app", "Deploy apps to a kubernetes cluster"), ("gpu-check", "Check gpu nodes")],
    )
    _plugin(
        root,
        "beta",
        "Deploy kubernetes cluster services",
        [("deploy-svc", "Deploy services to a kubernetes cluster")],
        refs=["github::example/repo::skills::gpu-check"],
    )
    _plugin(root, "gamma", "Convert confluence docs to markdown", [("conf2md", "Convert confluence docs to markdown")])
    # Same members as alpha but an unrelated description: a member-overlap-only match.
    _plugin(
        root,
        "delta",
        "Audit code review",
        refs=["github::example/repo::skills::deploy-app", "github::example/repo::skills::gpu-check"],
    )
    return root


def _save_catalog(root: Path, destination: Path, content_type: str = "plugin") -> dict:
    result = SimilarityValidator(content_type=content_type, save_catalog_path=destination).validate(root)
    assert result.passed, result.errors
    return json.loads(destination.read_text(encoding="utf-8"))


def _by_name(results):
    return {result.validator_name: result for result in results}


class TestSavePluginCatalog:
    def test_directory_of_plugins_writes_versioned_plugin_and_skill_entries(self, plugins: Path) -> None:
        data = _save_catalog(plugins, plugins.parent / "catalog.json")

        assert data["schema_version"] == 2
        assert data["mode"] == "description"
        by_path = {plugin["path"]: plugin for plugin in data["plugins"]}
        assert set(by_path) == {"alpha", "beta", "gamma", "delta"}
        alpha = by_path["alpha"]
        assert set(alpha) == {
            "id",
            "name",
            "description",
            "path",
            "manifest",
            "source_fingerprint",
            "members",
            "embedding",
        }
        assert alpha["id"] == "plugin:alpha"
        assert alpha["manifest"] == "agent_plugin.yaml"
        manifest_bytes = (plugins / "alpha" / "agent_plugin.yaml").read_bytes()
        assert alpha["source_fingerprint"] == hashlib.sha256(manifest_bytes).hexdigest()
        assert alpha["members"] == ["deploy-app", "gpu-check"]
        # Referenced skill leaves and bundled skills both become members.
        assert by_path["beta"]["members"] == ["deploy-svc", "gpu-check"]
        assert {entry["path"] for entry in data["entries"]} == {
            "alpha/skills/deploy-app",
            "alpha/skills/gpu-check",
            "beta/skills/deploy-svc",
            "gamma/skills/conf2md",
        }
        assert all(entry["content_type"] == "skill" for entry in data["entries"])
        assert all(not Path(item["path"]).is_absolute() for item in [*data["entries"], *data["plugins"]])

    def test_auto_detected_plugin_root_saves_itself(self, plugins: Path) -> None:
        catalog = plugins.parent / "alpha.json"
        result = SimilarityValidator(save_catalog_path=catalog).validate(plugins / "alpha")

        assert result.passed, result.errors
        data = json.loads(catalog.read_text(encoding="utf-8"))
        assert [plugin["path"] for plugin in data["plugins"]] == ["."]
        assert {entry["path"] for entry in data["entries"]} == {"skills/deploy-app", "skills/gpu-check"}

    def test_nested_contained_plugin_is_discovered(self, tmp_path: Path) -> None:
        plugin = tmp_path / "plugins" / "team" / "contained"
        (plugin / ".claude-plugin").mkdir(parents=True)
        (plugin / ".claude-plugin" / "plugin.json").write_text(
            json.dumps({"name": "contained", "description": "Convert confluence docs"}), encoding="utf-8"
        )
        (plugin / "skills" / "conf2md").mkdir(parents=True)
        (plugin / "skills" / "conf2md" / "SKILL.md").write_text(
            "---\nname: conf2md\ndescription: Convert confluence docs to markdown\n---\n", encoding="utf-8"
        )

        data = _save_catalog(tmp_path / "plugins", tmp_path / "catalog.json")

        [entry] = data["plugins"]
        assert (entry["path"], entry["manifest"], entry["members"]) == (
            "team/contained",
            ".claude-plugin/plugin.json",
            ["conf2md"],
        )
        assert [item["path"] for item in data["entries"]] == ["team/contained/skills/conf2md"]

    def test_plugin_without_description_keeps_bundled_skills_only(self, tmp_path: Path) -> None:
        root = tmp_path / "plugins"
        _plugin(root, "nodesc", None, [("deploy-app", "Deploy apps to a kubernetes cluster")])
        _plugin(root, "gamma", "Convert confluence docs to markdown")
        catalog = tmp_path / "catalog.json"

        result = SimilarityValidator(content_type="plugin", save_catalog_path=catalog).validate(root)

        assert result.passed
        assert any("nodesc" in warning and "no description" in warning for warning in result.warnings)
        data = json.loads(catalog.read_text(encoding="utf-8"))
        assert [plugin["name"] for plugin in data["plugins"]] == ["gamma"]
        assert [entry["path"] for entry in data["entries"]] == ["nodesc/skills/deploy-app"]

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({}, "--save-catalog"),
            ({"catalog_path": Path("catalog.json")}, "tier2 PLUGIN --catalog"),
            ({"save_catalog_path": Path("x.json"), "full_body": True}, "full-body"),
        ],
    )
    def test_plugin_similarity_only_builds_description_catalogs(
        self, plugins: Path, embed_calls, kwargs, message: str
    ) -> None:
        result = SimilarityValidator(content_type="plugin", **kwargs).validate(plugins)

        assert not result.passed
        assert any(message in error for error in result.errors)
        assert not embed_calls

    def test_entry_budget_fails_before_embedding(self, plugins: Path, embed_calls) -> None:
        catalog = plugins.parent / "catalog.json"
        result = SimilarityValidator(content_type="plugin", save_catalog_path=catalog, max_entries=3).validate(plugins)

        assert not result.passed
        assert any("entry limit" in error for error in result.errors)
        assert not embed_calls
        assert not catalog.exists()

    def test_symlinked_bundled_skill_manifest_is_refused(self, tmp_path: Path, embed_calls) -> None:
        root = tmp_path / "plugins"
        plugin = _plugin(root, "alpha", "Deploy kubernetes cluster workloads")
        outside = tmp_path / "outside.md"
        outside.write_text("---\nname: leak\ndescription: Outside the plugin\n---\n", encoding="utf-8")
        (plugin / "skills" / "leak").mkdir(parents=True)
        try:
            (plugin / "skills" / "leak" / "SKILL.md").symlink_to(outside)
        except OSError:
            pytest.skip("symlinks are unavailable")
        catalog = tmp_path / "catalog.json"

        result = SimilarityValidator(content_type="plugin", save_catalog_path=catalog).validate(root)

        assert not result.passed
        assert not embed_calls
        assert not catalog.exists()


class TestCatalogSchemaCompatibility:
    def test_skill_only_catalog_is_still_version_one(self, plugins: Path) -> None:
        data = _save_catalog(plugins / "beta" / "skills", plugins.parent / "skills.json", content_type="skill")

        assert data["schema_version"] == 1
        assert "plugins" not in data

    def test_old_skill_only_catalog_still_loads(self, plugins: Path) -> None:
        path = plugins.parent / "skills.json"
        data = _save_catalog(plugins / "beta" / "skills", path, content_type="skill")
        data["created_at"] = "2026-07-06T00:00:00+00:00"
        path.write_text(json.dumps(data), encoding="utf-8")

        registry = EmbeddingRegistry(EmbeddingClient())
        registry.load_catalog(path)

        assert registry.size == 1
        assert registry.plugin_size == 0

    def test_version_two_catalog_roundtrips(self, plugins: Path) -> None:
        path = plugins.parent / "catalog.json"
        _save_catalog(plugins, path)

        registry = EmbeddingRegistry(EmbeddingClient())
        registry.load_catalog(path)

        assert registry.plugin_size == 4
        assert {entry.path for entry in registry.skill_entries} >= {"alpha/skills/deploy-app"}

    @pytest.mark.parametrize(
        ("mutation", "message"),
        [
            (lambda data: data.__setitem__("schema_version", 1), r"unknown.*plugins"),
            (lambda data: data.pop("plugins"), r"missing.*plugins"),
            (lambda data: data.__setitem__("schema_version", 3), "schema version"),
            (lambda data: data["plugins"][0].update(id="plugin:other"), "match its relative path"),
            (lambda data: data["plugins"][0].update(path="../escape", id="plugin:../escape"), "relative"),
            (lambda data: data["plugins"][0].update(manifest="plugin.toml"), "manifest"),
            (lambda data: data["plugins"][0].update(source_fingerprint="nope"), "source fingerprint"),
            (lambda data: data["plugins"][0].update(members=["b", "a"]), "sorted"),
            (lambda data: data["plugins"][0].update(members=["Upper"]), "casefolded"),
            (lambda data: data["plugins"][0].update(members="a"), "list"),
            (lambda data: data["plugins"][0].update(members=[f"m{i:03d}" for i in range(257)]), "limit"),
            (lambda data: data["plugins"][0].update(embedding=[0.0] * len(data["plugins"][0]["embedding"])), "zero"),
            (lambda data: data["plugins"][0].update(secret="x"), r"unknown.*secret"),
            (lambda data: data["plugins"].append(copy.deepcopy(data["plugins"][0])), r"duplicate plugin id"),
            (lambda data: data.update(entries=[], plugins=[]), "at least one"),
        ],
    )
    def test_malformed_plugin_catalogs_are_rejected(self, plugins: Path, mutation, message) -> None:
        path = plugins.parent / "catalog.json"
        data = _save_catalog(plugins, path)
        mutation(data)
        path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ValueError, match=message):
            EmbeddingRegistry(EmbeddingClient()).load_catalog(path)

    def test_plugin_only_version_two_catalog_loads(self, plugins: Path) -> None:
        path = plugins.parent / "catalog.json"
        data = _save_catalog(plugins, path)
        data["entries"] = []
        path.write_text(json.dumps(data), encoding="utf-8")

        registry = EmbeddingRegistry(EmbeddingClient())
        registry.load_catalog(path)

        assert registry.size == 0
        assert registry.plugin_size == 4


class TestInterSkillCheck:
    def test_without_catalog_both_checks_are_recorded_as_skipped(self, plugins: Path, embed_calls) -> None:
        results = _by_name(run_plugin_catalog_checks(plugins / "alpha", catalog=None))

        inter_skill = results["Inter-Skill Deduplication"]
        inter_plugin = results["Inter-Plugin Deduplication"]
        for result, key in ((inter_skill, "catalog_skill_similarity"), (inter_plugin, "inter_plugin_similarity")):
            block = result.metadata["plugin"][key]
            assert set(block) == _METADATA_KEYS
            assert block["status"] == "skipped"
            assert "--catalog" in block["reason"]
            assert result.passed
            assert not result.is_incomplete
            assert result.metadata["optional"] is True
        assert not embed_calls

    def test_bundled_skills_are_compared_and_attributed(self, plugins: Path) -> None:
        catalog = plugins.parent / "catalog.json"
        _save_catalog(plugins, catalog)

        result = _by_name(run_plugin_catalog_checks(plugins / "alpha", catalog=catalog))["Inter-Skill Deduplication"]

        block = result.metadata["plugin"]["catalog_skill_similarity"]
        assert block["status"] == "compared"
        assert block["catalog_entries"] == 4
        assert block["reason"] is None
        assert block["matches"] == [{"skill": "skills/deploy-app", "match": "deploy-svc", "similarity": 1.0}]
        assert all(set(match) == {"skill", "match", "similarity"} for match in block["matches"])
        [finding] = result.findings
        assert finding.file_path == "skills/deploy-app"
        assert finding.category == "INTER_SKILL"
        assert finding.severity == Severity.MEDIUM  # EXACT_DUPLICATE capped: advisory
        assert finding.metadata["native_severity"] == "critical"
        assert result.passed
        assert result.metadata["advisory_tier2"] is True

    def test_self_matches_are_excluded_by_canonical_path(self, plugins: Path) -> None:
        catalog = plugins.parent / "catalog.json"
        _save_catalog(plugins, catalog)

        result = _by_name(run_plugin_catalog_checks(plugins / "alpha", catalog=catalog))["Inter-Skill Deduplication"]

        matched = {finding.metadata["catalog_path"] for finding in result.findings}
        assert not matched & {"alpha/skills/deploy-app", "alpha/skills/gpu-check"}
        compared = next(d for d in result.success_details if d.check_name == "skill_catalog_comparison")
        assert compared.metadata["self_matches_excluded"] == 2

    def test_self_matches_are_excluded_in_a_catalog_built_from_the_skills_directory(self, plugins: Path) -> None:
        catalog = plugins.parent / "alpha-skills.json"
        _save_catalog(plugins / "alpha" / "skills", catalog, content_type="skill")

        result = _by_name(run_plugin_catalog_checks(plugins / "alpha", catalog=catalog))["Inter-Skill Deduplication"]

        assert result.metadata["plugin"]["catalog_skill_similarity"]["matches"] == []
        assert not result.findings

    def test_plugin_without_bundled_skills_skips_inter_skill(self, plugins: Path) -> None:
        catalog = plugins.parent / "catalog.json"
        _save_catalog(plugins, catalog)

        result = _by_name(run_plugin_catalog_checks(plugins / "delta", catalog=catalog))["Inter-Skill Deduplication"]

        block = result.metadata["plugin"]["catalog_skill_similarity"]
        assert block["status"] == "skipped"
        assert "no skills" in block["reason"]


class TestInterPluginCheck:
    def test_description_and_member_overlap_matches_are_reported(self, plugins: Path) -> None:
        catalog = plugins.parent / "catalog.json"
        _save_catalog(plugins, catalog)

        result = _by_name(run_plugin_catalog_checks(plugins / "alpha", catalog=catalog))["Inter-Plugin Deduplication"]

        block = result.metadata["plugin"]["inter_plugin_similarity"]
        assert block["status"] == "compared"
        assert block["catalog_entries"] == 4
        assert [match["name"] for match in block["matches"]] == ["beta", "delta"]
        assert all(set(match) == {"name", "similarity", "member_overlap", "verdict"} for match in block["matches"])
        beta, delta = block["matches"]
        assert beta["similarity"] >= 0.75
        assert beta["member_overlap"] == pytest.approx(1 / 3, abs=1e-4)
        assert delta["similarity"] < 0.75
        assert delta["member_overlap"] == 1.0
        assert all(match["verdict"] is None for match in block["matches"])
        checks = {finding.metadata["catalog_plugin"]: finding for finding in result.findings}
        assert checks["delta"].check_name == "MEMBER_OVERLAP"
        assert checks["delta"].severity == Severity.LOW
        assert checks["beta"].severity == Severity.MEDIUM
        assert result.passed

    def test_plugin_is_excluded_by_name_and_source_identity(self, plugins: Path) -> None:
        catalog = plugins.parent / "catalog.json"
        data = _save_catalog(plugins, catalog)
        # A renamed catalog copy with alpha's exact manifest bytes is the same source.
        alpha = next(plugin for plugin in data["plugins"] if plugin["name"] == "alpha")
        renamed = copy.deepcopy(alpha)
        renamed.update(name="alpha-renamed", path="moved/alpha", id="plugin:moved/alpha")
        data["plugins"].append(renamed)
        catalog.write_text(json.dumps(data), encoding="utf-8")

        result = _by_name(run_plugin_catalog_checks(plugins / "alpha", catalog=catalog))["Inter-Plugin Deduplication"]

        names = [match["name"] for match in result.metadata["plugin"]["inter_plugin_similarity"]["matches"]]
        assert "alpha" not in names
        assert "alpha-renamed" not in names
        detail = next(d for d in result.success_details if d.check_name == "plugin_catalog_comparison")
        assert detail.metadata["self_matches_excluded"] == 2

    def test_old_skill_only_catalog_skips_check_b_but_runs_c_inter(self, plugins: Path) -> None:
        catalog = plugins.parent / "skills.json"
        _save_catalog(plugins / "beta" / "skills", catalog, content_type="skill")

        results = _by_name(run_plugin_catalog_checks(plugins / "alpha", catalog=catalog))

        assert results["Inter-Skill Deduplication"].metadata["plugin"]["catalog_skill_similarity"]["status"] == "compared"
        block = results["Inter-Plugin Deduplication"].metadata["plugin"]["inter_plugin_similarity"]
        assert block["status"] == "skipped"
        assert "no plugin entries" in block["reason"]
        assert all(result.passed for result in results.values())


class _FakeLLM:
    instances: ClassVar[list[_FakeLLM]] = []
    responses: ClassVar[dict[str, object]] = {}

    def __init__(self, **_kwargs: object) -> None:
        self.prompts: list[str] = []
        _FakeLLM.instances.append(self)

    def _resolved_config(self) -> SimpleNamespace:
        return SimpleNamespace(provider="nv_build", model="fake-judge")

    def extract_json_from_response(self, _system: str, user: str) -> dict:
        self.prompts.append(user)
        for name, response in _FakeLLM.responses.items():
            if f'EXISTING PLUGIN: "{name}"' in user:
                if isinstance(response, Exception):
                    raise response
                return response  # type: ignore[return-value]
        raise AssertionError("unexpected prompt")


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> type[_FakeLLM]:
    _FakeLLM.instances = []
    _FakeLLM.responses = {}
    monkeypatch.setattr("skillevaluator.inference.LLMClient", _FakeLLM)
    return _FakeLLM


class TestInterPluginLLMVerdict:
    def test_llm_verdict_is_off_by_default(self, plugins: Path, fake_llm) -> None:
        catalog = plugins.parent / "catalog.json"
        _save_catalog(plugins, catalog)

        result = _by_name(run_plugin_catalog_checks(plugins / "alpha", catalog=catalog))["Inter-Plugin Deduplication"]

        assert not fake_llm.instances
        assert "llm_analysis" not in result.metadata

    def test_mocked_llm_verdict_refines_matches(self, plugins: Path, fake_llm) -> None:
        catalog = plugins.parent / "catalog.json"
        _save_catalog(plugins, catalog)
        fake_llm.responses = {
            "beta": {
                "verdict": "WHOLE_DUPLICATE",
                "confidence": 0.9,
                "reasoning": "Same deployment bundle.",
                "recommendation": "UPDATE_EXISTING",
                "suggestion": "Extend beta instead.",
            },
            "delta": {"verdict": "UNIQUE", "confidence": 0.8, "reasoning": "Different purpose.", "suggestion": ""},
        }

        result = _by_name(run_plugin_catalog_checks(plugins / "alpha", catalog=catalog, llm_verdict=True))[
            "Inter-Plugin Deduplication"
        ]

        matches = result.metadata["plugin"]["inter_plugin_similarity"]["matches"]
        assert {match["name"]: match["verdict"] for match in matches} == {
            "beta": "WHOLE_DUPLICATE",
            "delta": "UNIQUE",
        }
        [finding] = result.findings  # UNIQUE matches stay recorded but raise no finding
        assert finding.check_name == "WHOLE_DUPLICATE"
        assert finding.severity == Severity.MEDIUM
        assert finding.suggestion == "Extend beta instead."
        assert result.metadata["llm_analysis"]["candidates_completed"] == 2
        [client] = fake_llm.instances
        assert all("Member skills:" in prompt for prompt in client.prompts)
        assert result.passed

    def test_llm_failures_keep_similarity_findings(self, plugins: Path, fake_llm) -> None:
        from skillevaluator.inference import LLMClientError

        catalog = plugins.parent / "catalog.json"
        _save_catalog(plugins, catalog)
        fake_llm.responses = {"beta": LLMClientError("boom"), "delta": {"verdict": "MAYBE"}}

        result = _by_name(run_plugin_catalog_checks(plugins / "alpha", catalog=catalog, llm_verdict=True))[
            "Inter-Plugin Deduplication"
        ]

        assert [match["verdict"] for match in result.metadata["plugin"]["inter_plugin_similarity"]["matches"]] == [
            None,
            None,
        ]
        assert len(result.findings) == 2
        assert result.metadata["llm_analysis"]["candidates_failed"] == 2
        assert any("LLM verdict unavailable" in warning for warning in result.warnings)
        assert result.passed
        assert not result.is_incomplete


class TestUnsafeOrOversizedInputs:
    def test_malformed_catalog_is_an_advisory_skip(self, plugins: Path, embed_calls) -> None:
        catalog = plugins.parent / "bad.json"
        catalog.write_text("[]", encoding="utf-8")

        results = run_plugin_catalog_checks(plugins / "alpha", catalog=catalog)

        for result in results:
            block = next(iter(result.metadata["plugin"].values()))
            assert block["status"] == "skipped"
            assert "could not be loaded" in block["reason"]
            assert str(plugins.parent) not in block["reason"]
            assert result.passed
        assert not embed_calls

    def test_oversize_catalog_is_rejected_before_parsing(
        self, plugins: Path, embed_calls, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        catalog = plugins.parent / "catalog.json"
        _save_catalog(plugins, catalog)
        embed_calls.clear()
        monkeypatch.setattr(registry_module, "MAX_CATALOG_BYTES", 64)

        results = run_plugin_catalog_checks(plugins / "alpha", catalog=catalog)

        assert all("size limit" in next(iter(r.metadata["plugin"].values()))["reason"] for r in results)
        assert not embed_calls

    def test_symlinked_catalog_is_refused(self, plugins: Path, embed_calls) -> None:
        catalog = plugins.parent / "catalog.json"
        _save_catalog(plugins, catalog)
        link = plugins.parent / "link.json"
        try:
            link.symlink_to(catalog)
        except OSError:
            pytest.skip("symlinks are unavailable")
        embed_calls.clear()

        results = run_plugin_catalog_checks(plugins / "alpha", catalog=link)

        assert all("symlink" in next(iter(r.metadata["plugin"].values()))["reason"] for r in results)
        assert not embed_calls

    def test_scalar_work_limit_is_respected(self, plugins: Path) -> None:
        catalog = plugins.parent / "catalog.json"
        _save_catalog(plugins, catalog)

        results = _by_name(run_plugin_catalog_checks(plugins / "alpha", catalog=catalog, max_scalar_comparisons=1))

        for result in results.values():
            block = next(iter(result.metadata["plugin"].values()))
            assert block["status"] == "skipped"
            assert "Scalar comparison work limit" in block["reason"]
            assert result.passed

    def test_bundled_skill_limit_skips_before_provider_work(
        self, plugins: Path, embed_calls, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        catalog = plugins.parent / "catalog.json"
        _save_catalog(plugins, catalog)
        embed_calls.clear()
        monkeypatch.setattr("skillevaluator.tier2.commands.MAX_PLUGIN_DEDUP_SKILLS", 1)

        results = run_plugin_catalog_checks(plugins / "alpha", catalog=catalog)

        assert all(result.metadata["work_limit_exceeded"] for result in results)
        assert all(result.metadata["execution_status"] == "skipped" for result in results)
        assert not embed_calls

    def test_symlinked_bundled_skill_is_a_blocking_security_failure(self, tmp_path: Path, embed_calls) -> None:
        plugin = _plugin(tmp_path, "alpha", "Deploy kubernetes cluster workloads")
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "SKILL.md").write_text("---\nname: x\ndescription: y\n---\n", encoding="utf-8")
        (plugin / "skills").mkdir()
        try:
            (plugin / "skills" / "x").symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks are unavailable")
        catalog = tmp_path / "catalog.json"
        catalog.write_text("{}", encoding="utf-8")

        [result] = run_plugin_catalog_checks(plugin, catalog=catalog)

        assert not result.passed
        assert result.metadata["security_failure"] is True
        assert not embed_calls


def test_plugin_dedup_scan_records_catalog_checks_without_a_catalog(plugins: Path) -> None:
    results = run_plugin_dedup_scan(plugins / "alpha", run_context=False)

    names = [result.validator_name for result in results]
    assert names == [
        "Plugin Dependency Deduplication",
        "Context Deduplication",
        "Inter-Skill Deduplication",
        "Inter-Plugin Deduplication",
    ]
    assert all(result.passed for result in results)
    assert results[2].metadata["plugin"]["catalog_skill_similarity"]["status"] == "skipped"
    assert results[3].metadata["plugin"]["inter_plugin_similarity"]["status"] == "skipped"
