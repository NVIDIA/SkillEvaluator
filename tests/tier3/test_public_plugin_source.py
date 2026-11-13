# SPDX-License-Identifier: Apache-2.0
"""Generated plugin packages keep their verified authored source context."""

import pytest
from test_public_plugin_workflow import _plugin

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
    EvaluationService().evaluate_plugin(EvaluationOptions(root, agents="claude-code"))
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
    EvaluationService().evaluate_plugin(EvaluationOptions(root, agents="claude-code"))
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
        EvaluationService().evaluate_plugin(EvaluationOptions(root, agents="claude-code"))


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
    EvaluationService().evaluate_plugin(EvaluationOptions(root, agents="claude-code"))
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
