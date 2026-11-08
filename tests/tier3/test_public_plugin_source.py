# SPDX-License-Identifier: Apache-2.0
"""Generated plugin packages keep their verified authored source context."""

import pytest
from test_public_plugin_workflow import _plugin

from skillevaluator.evaluation import EvaluationOptions, EvaluationService


def test_custom_environment_assets_survive_staging(tmp_path, monkeypatch):
    root = _plugin(tmp_path / "plugin")
    environment = root / "evals/environment"
    (environment / "sidecar").mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM python:3.12-slim\nCOPY sidecar /sidecar\n")
    (environment / "sidecar/config.txt").write_text("fixture")
    seen = {}

    def engine(self, options, *, progress_reporter=None):
        target = options.skill_path / "evals/environment"
        seen["dockerfile"] = (target / "Dockerfile").read_text()
        seen["asset"] = (target / "sidecar/config.txt").read_text()
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
    (root.parent / "README.md").write_text("surrounding repository context")
    (root.parent / ".env").write_text("PRIVATE_FIXTURE=do-not-copy")
    (root / "evals").mkdir()
    (root / "evals/grader.py").write_text("evaluator-only fixture")
    package = prepare_runtime_package(plugin_inventory_for_root(root))
    hooks = NativeRuntimeHooks(package.native_source, {}, package=package)
    active = (
        ComparisonRuntime(hooks, package, ComparisonOptions(), ["claude-code"], baseline=True) if comparison else hooks
    )
    wrapper = tmp_path / "work/temporary-wrapper"
    wrapper.mkdir(parents=True)
    (wrapper / "SKILL.md").write_text("generated wrapper")

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
            assert (env / "repo/README.md").read_text() == "surrounding repository context"
            assert not (env / "repo/.env").exists()
            assert not (env / "repo/plugin/evals").exists()
            assert not (env / "repo/plugin/skills/a/evals").exists()
            assert (env / "repo/plugin/.claude-plugin/plugin.json").exists() is (not baseline)
            assert metadata["repo_root"] == str(root.parent)
    assert runtime_source_path(wrapper) == wrapper
