# SPDX-License-Identifier: Apache-2.0
"""Generated plugin packages keep their verified authored source context."""

import json
import shutil
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_public_plugin_workflow import _default_results_location, _plugin, evaluate_plugin  # noqa: F401

from skillevaluator.evaluation import EvaluationOptions, EvaluationService


@pytest.mark.parametrize("absolute", [False, True], ids=["sibling", "absolute"])
def test_workspace_includes_keep_authored_source_context(tmp_path, monkeypatch, absolute):
    import yaml

    from skillevaluator.tier3.evals_config import load_evals_config
    from skillevaluator.tier3.harbor.runner import _workspace_skills

    root = _plugin(tmp_path / "plugin")
    helper = tmp_path / "helper"
    helper.mkdir()
    (helper / "SKILL.md").write_text(
        "---\nname: helper\ndescription: Separate helper.\n---\nHelper body.\n", encoding="utf-8"
    )
    (root / "evals").mkdir()
    (root / "evals/config.yml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "skill_workspace": {"mode": "group", "include": [str(helper) if absolute else "helper"]},
            }
        ),
        encoding="utf-8",
    )
    seen = []

    def engine(self, options, *, progress_reporter=None):
        config, _ = load_evals_config(options.skill_path)
        seen.extend(_workspace_skills(options.skill_path, config["skill_workspace"]["include"]))
        return {"execution_status": "succeeded"}

    monkeypatch.setattr(EvaluationService, "evaluate", engine)
    evaluate_plugin(EvaluationOptions(root, agents="claude-code"))
    assert seen == [helper]


@pytest.mark.parametrize("absolute", [False, True], ids=["relative", "absolute"])
def test_price_file_and_decoy_manifest_keep_authored_source_context(tmp_path, monkeypatch, absolute):
    import yaml

    from skillevaluator.tier3 import decoys
    from skillevaluator.tier3.evals_config import load_evals_config
    from skillevaluator.tier3.harbor.runner import _decoy_workspace_skills, _resolve_harbor_price_file

    root = _plugin(tmp_path / "plugin")
    (root / "evals").mkdir()
    price_book = root / "evals/prices.json"
    price_book.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "decoys.json"
    manifest.write_text("{}", encoding="utf-8")
    (root / "evals/config.yml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "harbor": {
                    "max_cost_usd": 1.0,
                    "price_file": str(price_book) if absolute else "evals/prices.json",
                },
                "skill_workspace": {"mode": "group", "decoy_manifest": str(manifest) if absolute else "decoys.json"},
            }
        ),
        encoding="utf-8",
    )
    seen = {}

    def load_manifest(target, manifest_path):
        seen["manifest"] = manifest_path
        return [], {"selected": []}

    def engine(self, options, *, progress_reporter=None):
        config, _ = load_evals_config(options.skill_path)
        seen["price_file"] = _resolve_harbor_price_file(None, config["harbor"], options.skill_path)
        _decoy_workspace_skills(options.skill_path, config["skill_workspace"], "group")
        return {"execution_status": "succeeded"}

    monkeypatch.setattr(decoys, "load_decoy_manifest", load_manifest)
    monkeypatch.setattr(EvaluationService, "evaluate", engine)
    evaluate_plugin(EvaluationOptions(root, agents="claude-code"))
    # The price book is read from the plugin, and the manifest from the plugin's parent, as for a skill.
    assert seen["price_file"] == price_book.resolve() and seen["price_file"].is_file()
    assert seen["manifest"].resolve() == manifest.resolve()


def test_missing_workspace_include_remains_an_error(tmp_path, monkeypatch):
    from skillevaluator.tier3.evals_config import load_evals_config
    from skillevaluator.tier3.harbor.runner import _workspace_skills

    root = _plugin(tmp_path / "plugin")
    (root / "evals").mkdir()
    (root / "evals/config.yml").write_text(
        "schema_version: 1\nskill_workspace:\n  mode: group\n  include: [missing-helper]\n", encoding="utf-8"
    )

    def engine(self, options, *, progress_reporter=None):
        config, _ = load_evals_config(options.skill_path)
        _workspace_skills(options.skill_path, config["skill_workspace"]["include"])
        return {"execution_status": "succeeded"}

    monkeypatch.setattr(EvaluationService, "evaluate", engine)
    with pytest.raises(ValueError, match="Included skill path is not a skill"):
        evaluate_plugin(EvaluationOptions(root, agents="claude-code"))


def test_custom_environment_assets_survive_staging(tmp_path, monkeypatch):
    root = _plugin(tmp_path / "plugin")
    environment = root / "evals/environment"
    (environment / "sidecar").mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM python:3.12-slim\nCOPY sidecar /sidecar\n", encoding="utf-8")
    (environment / "sidecar/config.txt").write_text("fixture", encoding="utf-8")
    seen = {}

    def engine(self, options, *, progress_reporter=None):
        target = options.skill_path / "evals/environment"
        seen["dockerfile"] = (target / "Dockerfile").read_text(encoding="utf-8")
        seen["asset"] = (target / "sidecar/config.txt").read_text(encoding="utf-8")
        return {"execution_status": "succeeded"}

    monkeypatch.setattr(EvaluationService, "evaluate", engine)
    evaluate_plugin(EvaluationOptions(root, agents="claude-code"))
    assert seen == {"dockerfile": "FROM python:3.12-slim\nCOPY sidecar /sidecar\n", "asset": "fixture"}


@pytest.mark.parametrize("comparison", [False, True], ids=["native-hooks", "comparison-hooks"])
def test_copy_repo_uses_original_source_and_excludes_verifier_inputs(tmp_path, monkeypatch, comparison):
    from skillevaluator.plugin_inventory import plugin_inventory_for_root
    from skillevaluator.tier3.harbor import adapter
    from skillevaluator.tier3.plugin_comparisons import ComparisonOptions, ComparisonRuntime
    from skillevaluator.tier3.plugin_execution import NativeRuntimeHooks, private_runtime, runtime_source_path
    from skillevaluator.tier3.plugin_prepare import prepare_runtime_package

    root = _plugin(tmp_path / "repo/plugin")
    (root.parent / "README.md").write_text("surrounding repository context", encoding="utf-8")
    (root.parent / ".env").write_text("PRIVATE_FIXTURE=do-not-copy", encoding="utf-8")
    (root / "evals").mkdir()
    (root / "evals/grader.py").write_text("evaluator-only fixture", encoding="utf-8")
    package = prepare_runtime_package(plugin_inventory_for_root(root))
    hooks = NativeRuntimeHooks(package.native_source, {}, package=package)
    active = (
        ComparisonRuntime(hooks, package, ComparisonOptions(), ["claude-code"], baseline=True) if comparison else hooks
    )
    wrapper = tmp_path / "work/temporary-wrapper"
    wrapper.mkdir(parents=True)
    (wrapper / "SKILL.md").write_text("generated wrapper", encoding="utf-8")

    def repository(source):
        assert source == root
        return root.parent

    monkeypatch.setattr(adapter, "resolve_repo_context_root", repository)
    with private_runtime(active):
        assert runtime_source_path(wrapper) == root
        for baseline in (False, True):
            env = tmp_path / f"env-{baseline}"
            env.mkdir()
            metadata = adapter._stage_repo_context(
                env, source_skill_path=wrapper, mode="full", exclude_source_skill=baseline
            )
            assert (env / "repo/README.md").read_text(encoding="utf-8") == "surrounding repository context"
            assert not (env / "repo/.env").exists()
            assert not (env / "repo/plugin/evals").exists()
            assert not (env / "repo/plugin/skills/a/evals").exists()
            assert (env / "repo/plugin/.claude-plugin/plugin.json").exists() is (not baseline)
            assert metadata["repo_root"] == str(root.parent)
    assert runtime_source_path(wrapper) == wrapper


def test_copy_repo_excludes_member_datasets_in_the_legacy_layout(tmp_path, monkeypatch):
    from skillevaluator.plugin_inventory import plugin_inventory_for_root
    from skillevaluator.tier3.harbor import adapter
    from skillevaluator.tier3.plugin_execution import NativeRuntimeHooks, private_runtime
    from skillevaluator.tier3.plugin_prepare import prepare_runtime_package, stage_wrapper_skill

    root = _plugin(tmp_path / "repo/plugin")
    legacy = root / "skills/b"
    (legacy / "eval").mkdir(parents=True)
    (legacy / "SKILL.md").write_text("---\nname: b\ndescription: Legacy member.\n---\nBody.\n", encoding="utf-8")
    (legacy / "eval/dataset.json").write_text(
        json.dumps([{"id": "legacy", "prompt": "Use b.", "expected_output": "FIXTURE-ANSWER"}]), encoding="utf-8"
    )
    (legacy / "eval/notes.md").write_text("verifier-only notes", encoding="utf-8")
    # A directory named eval without a dataset is ordinary skill content, as in the wrapper.
    (root / "skills/a/eval").mkdir()
    (root / "skills/a/eval/helper.md").write_text("runtime helper", encoding="utf-8")
    package = prepare_runtime_package(plugin_inventory_for_root(root))
    (tmp_path / "work").mkdir()
    wrapper = stage_wrapper_skill(package, tmp_path / "work/wrapper")
    # The preparer reads the legacy member dataset, and the wrapper leaves it out.
    cases = json.loads((wrapper.path / "evals/evals.json").read_text(encoding="utf-8"))
    assert "b-legacy" in [case["id"] for case in cases]
    assert not (wrapper.path / "skills/b/eval").exists()
    monkeypatch.setattr(adapter, "resolve_repo_context_root", lambda _source: root.parent)

    with private_runtime(NativeRuntimeHooks(package.native_source, {}, package=package)):
        env = tmp_path / "env"
        env.mkdir()
        metadata = adapter._stage_repo_context(env, source_skill_path=wrapper.path, mode="full")

    staged = {entry["container"] for entry in metadata["files"]}
    assert "/workspace/repo/plugin/skills/b/SKILL.md" in staged
    assert not (env / "repo/plugin/skills/b/eval").exists()
    assert not any("FIXTURE-ANSWER" in path.read_text(encoding="utf-8") for path in env.rglob("*") if path.is_file())
    assert (env / "repo/plugin/skills/a/eval/helper.md").read_text(encoding="utf-8") == "runtime helper"


def test_relative_source_spelling_resolves_paths_from_its_real_parent(tmp_path, monkeypatch):
    import yaml

    from skillevaluator.tier3 import decoys
    from skillevaluator.tier3.evals_config import load_evals_config
    from skillevaluator.tier3.harbor.runner import (
        _decoy_workspace_skills,
        _resolve_harbor_price_file,
        _workspace_skills,
    )

    root = _plugin(tmp_path / "plugin")
    helper = tmp_path / "helper"
    helper.mkdir()
    (helper / "SKILL.md").write_text("---\nname: helper\ndescription: Separate helper.\n---\nBody.\n", encoding="utf-8")
    (tmp_path / "decoys.json").write_text("{}", encoding="utf-8")
    (root / "evals").mkdir()
    (root / "evals/prices.json").write_text("{}", encoding="utf-8")
    (root / "evals/config.yml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "harbor": {"max_cost_usd": 1.0, "price_file": "evals/prices.json"},
                "skill_workspace": {"mode": "group", "include": ["helper"], "decoy_manifest": "decoys.json"},
            }
        ),
        encoding="utf-8",
    )
    seen = {}

    def load_manifest(target, manifest_path):
        seen["manifest"] = manifest_path
        return [], {"selected": []}

    def engine(self, options, *, progress_reporter=None):
        config, _ = load_evals_config(options.skill_path)
        seen["include"] = _workspace_skills(options.skill_path, config["skill_workspace"]["include"])
        seen["price_file"] = _resolve_harbor_price_file(None, config["harbor"], options.skill_path)
        _decoy_workspace_skills(options.skill_path, config["skill_workspace"], "group")
        return {"execution_status": "succeeded"}

    monkeypatch.setattr(decoys, "load_decoy_manifest", load_manifest)
    monkeypatch.setattr(EvaluationService, "evaluate", engine)
    # Run from inside the source, which names it ``.``: its lexical parent is ``.`` too.
    monkeypatch.chdir(root)
    evaluate_plugin(EvaluationOptions(Path(), agents="claude-code"))
    assert seen["include"] == [helper.resolve()]
    assert Path(seen["manifest"]).resolve() == (tmp_path / "decoys.json").resolve()
    assert seen["price_file"] == (root / "evals/prices.json").resolve()


def _case_alias(path):
    alias = path.with_name(path.name.upper())
    if not alias.exists():
        pytest.skip("case-sensitive filesystem")
    return alias


@pytest.mark.parametrize("hooks", [True, False], ids=["prepared-package", "no-hooks"])
@pytest.mark.parametrize("alias", [False, True], ids=["plain", "case-alias"])
def test_engine_publishes_a_prepared_package_as_its_authored_source(tmp_path, monkeypatch, hooks, alias):
    from skillevaluator.plugin_inventory import plugin_inventory_for_root
    from skillevaluator.publication_identity import publication_target_from_path
    from skillevaluator.tier3 import commands
    from skillevaluator.tier3.plugin_execution import NativeRuntimeHooks, private_runtime
    from skillevaluator.tier3.plugin_prepare import prepare_runtime_package, stage_wrapper_skill, wrapper_skill_name
    from skillevaluator.tier3.results_location import skill_results_name
    from test_tier3_progress import _stub_runner

    root = _plugin(tmp_path / "plugin")
    source = _case_alias(root) if alias and hooks else root
    package = prepare_runtime_package(plugin_inventory_for_root(source))
    _, wrapper = _stub_runner(monkeypatch, tmp_path / "work", skill_name=wrapper_skill_name(package.plugin_name))
    shutil.rmtree(wrapper)
    stage_wrapper_skill(package, wrapper)
    typed = _case_alias(wrapper) if alias and not hooks else wrapper
    monkeypatch.setattr(commands, "resolve_llm_provider", lambda: SimpleNamespace(provider="openai"))
    results = tmp_path / "results"
    # With hooks, include a member skill of the authored source: the runtime-context check
    # binds it only when it checks the source, not the wrapper.
    extra = {"skill_workspace_mode": "group", "include_skills": (source / "skills/a",)} if hooks else {}
    options = EvaluationOptions(typed, agents="claude-code", results_dir=results, **extra)
    runtime = (
        private_runtime(NativeRuntimeHooks(package.native_source, {}, package=package)) if hooks else nullcontext()
    )
    with runtime:
        result = EvaluationService().evaluate(options)
    # With hooks, the authored source's recorded name (its entry name, also for an alias);
    # without, the typed skill path's name, as for any skill.
    published = source if hooks else typed
    name = publication_target_from_path(source)["skill_name"] if hooks else typed.resolve().name
    assert name == (root.name if hooks else typed.name)

    assert result["execution_status"] == "succeeded", result
    assert "publication_target_conflict" not in result, result["publication_target_conflict"]
    assert result["publication_target"] == publication_target_from_path(published)
    saved = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert result["skill_name"] == saved["skill_name"] == name
    assert Path(result["run_dir"]).parent == results / skill_results_name(published)
