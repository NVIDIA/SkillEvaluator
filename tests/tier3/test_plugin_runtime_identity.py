# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-owned results must not mutate the authored plugin identity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.plugin_inventory import plugin_inventory_for_root
from skillevaluator.publication_identity import publication_source_digest
from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks
from skillevaluator.tier3.harbor.runner import _reserve_run_dir
from skillevaluator.tier3.plugin_execution import NativeRuntimeHooks
from skillevaluator.tier3.plugin_native import resolve_plugin_load
from skillevaluator.tier3.plugin_prepare import (
    PreparationError,
    bind_runtime_results,
    prepare_runtime_package,
    stage_wrapper_skill,
    verify_source_identity,
)


def _source(root: Path) -> Path:
    files = {
        ".claude-plugin/plugin.json": json.dumps({"name": "demo"}),
        "skills/a/SKILL.md": "---\nname: a\ndescription: An example.\n---\nBody.\n",
        "skills/a/evals/evals.json": json.dumps([{"id": "one", "prompt": "Do it."}]),
    }
    for name, content in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


def _prepare(root: Path):
    inventory = plugin_inventory_for_root(root)
    assert inventory is not None
    return prepare_runtime_package(inventory)


@pytest.mark.parametrize("relative", ["evals/results", "run-data/nested/reports"])
@pytest.mark.parametrize("agent", ["claude-code", "codex", "opencode"])
def test_real_reservation_and_native_staging_preserve_missing_results_ancestors(tmp_path, relative, agent):
    root = _source(tmp_path / "plugin")
    original = _prepare(root)
    output = root / relative
    package = bind_runtime_results(original, output)
    assert package.identity == original.identity
    assert not output.exists()  # Binding remains passive.
    assert output in package.native_source.excluded_paths
    wrapper = stage_wrapper_skill(package, tmp_path / "wrapper")
    run = _reserve_run_dir(output, "run")
    (run / "result.json").write_text('{"private_report": "RESULTS-MUST-NOT-REACH-AGENT"}')
    tasks = generate_harbor_tasks(wrapper.path, tmp_path / "tasks")
    hooks = NativeRuntimeHooks(
        package.native_source, resolve_plugin_load("native", [agent], env_mode="docker"), package=package
    )
    hooks.stage_with_arm(agent, tasks)
    verify_source_identity(package)
    visible = tasks[0] / "environment"
    assert not any(
        b"RESULTS-MUST-NOT-REACH-AGENT" in path.read_bytes() for path in visible.rglob("*") if path.is_file()
    )
    # The global publication recipe still includes new ancestors. Only the
    # private run guard omits them, without rewriting the analyzed identity.
    assert publication_source_digest(root) != original.identity.source_tree_sha256


def test_authored_root_evals_content_stays_guarded(tmp_path):
    root = _source(tmp_path / "plugin")
    authored = root / "evals/config.yml"
    authored.parent.mkdir()
    authored.write_text("schema_version: 1\n")
    package = bind_runtime_results(_prepare(root), root / "evals/results")
    _reserve_run_dir(root / "evals/results", "run")
    verify_source_identity(package)
    authored.write_text("schema_version: 2\n")
    with pytest.raises(PreparationError, match="changed"):
        verify_source_identity(package)


@pytest.mark.parametrize("changed", ["skills/a/SKILL.md", "skills/a/evals/evals.json", "run-data/README.md"])
def test_runtime_outputs_do_not_mask_authored_source_or_new_sibling_changes(tmp_path, changed):
    root = _source(tmp_path / "plugin")
    package = bind_runtime_results(_prepare(root), root / "run-data/nested/reports")
    _reserve_run_dir(root / "run-data/nested/reports", "run")
    verify_source_identity(package)
    target = root / changed
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("authored mutation")
    with pytest.raises(PreparationError, match="changed"):
        verify_source_identity(package)


@pytest.mark.parametrize("bound", [False, True])
def test_authored_empty_directory_changes_remain_guarded(tmp_path, bound):
    root = _source(tmp_path / "plugin")
    package = _prepare(root)
    assert package.identity.source_tree_sha256 == publication_source_digest(root)
    if bound:
        output = root / "run-data/nested/reports"
        package = bind_runtime_results(package, output)
        _reserve_run_dir(output, "run")
    verify_source_identity(package)
    authored = root / ("run-data/authored-empty" if bound else "authored-empty")
    authored.mkdir()
    with pytest.raises(PreparationError, match="changed"):
        verify_source_identity(package)


def test_custom_results_root_with_authenticated_prior_runs_is_usable(tmp_path):
    root = _source(tmp_path / "plugin")
    output = root / "report-output"
    prior = _reserve_run_dir(output, "prior")
    (prior / "result.json").write_text("{}")
    package = bind_runtime_results(_prepare(root), output)
    current = _reserve_run_dir(output, "current")
    (current / "result.json").write_text("{}")
    verify_source_identity(package)


@pytest.mark.parametrize("agent", ["claude-code", "codex", "opencode"])
def test_member_nested_results_stay_out_of_repeated_wrapper_and_native_copies(tmp_path, agent):
    from skillevaluator.tier3.results_location import publish_latest_results

    root = _source(tmp_path / "plugin")
    helper = root / "skills/a/eval/helper.py"
    helper.parent.mkdir()
    helper.write_text("print('authored runtime helper')\n")
    output = root / "skills/a/run-data/reports"
    package = bind_runtime_results(_prepare(root), output)
    hooks = NativeRuntimeHooks(
        package.native_source, resolve_plugin_load("native", [agent], env_mode="docker"), package=package
    )
    for attempt in range(2):
        run = _reserve_run_dir(output, "run")
        (run / "result.json").write_text('{"private_report": "NESTED-REPORT-MUST-NOT-REACH-AGENT"}')
        assert publish_latest_results(output, run.name)
        wrapper = stage_wrapper_skill(package, tmp_path / f"wrapper-{attempt}")
        assert not (wrapper.path / "skills/a/run-data/reports").exists()
        assert (wrapper.path / "skills/a/eval/helper.py").read_text() == helper.read_text()
        tasks = generate_harbor_tasks(wrapper.path, tmp_path / f"tasks-{attempt}")
        hooks.stage_with_arm(agent, tasks)
        assert not any(
            b"NESTED-REPORT-MUST-NOT-REACH-AGENT" in path.read_bytes()
            for path in (tasks[0] / "environment").rglob("*")
            if path.is_file()
        )
        verify_source_identity(package)
    helper.write_text("authored mutation\n")
    with pytest.raises(PreparationError, match="changed"):
        verify_source_identity(package)


def test_existing_custom_results_content_is_not_removed_from_authored_guard(tmp_path):
    root = _source(tmp_path / "plugin")
    output = root / "report-output"
    output.mkdir()
    authored = output / "helper.py"
    authored.write_text("print('runtime helper')\n")
    with pytest.raises(PreparationError, match="authored or unowned"):
        bind_runtime_results(_prepare(root), output)
    assert authored.read_text() == "print('runtime helper')\n"


def test_source_mutation_before_binding_is_refused(tmp_path):
    root = _source(tmp_path / "plugin")
    package = _prepare(root)
    (root / "skills/a/SKILL.md").write_text("changed")
    with pytest.raises(PreparationError, match="changed"):
        bind_runtime_results(package, root / "reports")


@pytest.mark.parametrize("target", ["source", "ancestor", "linked"])
def test_unsafe_results_roots_are_refused_without_creation(tmp_path, target):
    root = _source(tmp_path / "plugin")
    package = _prepare(root)
    if target == "linked":
        output = root / "output-link"
        output.symlink_to(tmp_path / "outside", target_is_directory=True)
        # Adding the link also changes source identity, so either guard may
        # refuse it; neither is allowed to follow the destination.
    else:
        output = root if target == "source" else tmp_path
    with pytest.raises(ValueError):
        bind_runtime_results(package, output)
    assert not (tmp_path / "outside").exists()


def test_external_results_need_no_source_exclusion(tmp_path):
    root = _source(tmp_path / "plugin")
    package = _prepare(root)
    bound = bind_runtime_results(package, tmp_path / "outside")
    assert bound is package
    _reserve_run_dir(tmp_path / "outside", "run")
    verify_source_identity(bound)


def test_replaced_results_parent_does_not_bypass_no_follow_validation(tmp_path):
    root = _source(tmp_path / "plugin")
    package = bind_runtime_results(_prepare(root), root / "report-output/results")
    (root / "report-output").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(PreparationError):
        verify_source_identity(package)


@pytest.mark.parametrize("relative", [None, "reports/nested/output"])
def test_private_entrypoint_binds_effective_results_before_engine_reservation(tmp_path, relative):
    from skillevaluator.evaluation.options import EvaluationOptions
    from skillevaluator.tier3.plugin_execution import active_runtime, evaluate_prepared_plugin
    from skillevaluator.tier3.results_location import publish_latest_results

    root = _source(tmp_path / "plugin")
    options = EvaluationOptions(
        skill_path=root,
        agents="claude-code",
        results_dir=root / relative if relative else None,
    )
    expected = root / (relative or "evals/results")
    assert not expected.parent.exists()

    def evaluate(prepared_options):
        assert prepared_options.results_dir == expected
        run = _reserve_run_dir(prepared_options.results_dir, "run")
        (run / "result.json").write_text("{}")
        tasks = generate_harbor_tasks(prepared_options.skill_path, tmp_path / "tasks")
        hooks = active_runtime()
        assert hooks is not None
        hooks.stage_with_arm("claude-code", tasks)
        assert publish_latest_results(prepared_options.results_dir, run.name)
        return {"execution_status": "succeeded", "run_dir": str(run)}

    original = _prepare(root)
    package = bind_runtime_results(original, expected)
    for _ in range(2):
        outcome = evaluate_prepared_plugin(package, options, work_dir=tmp_path, load="native", evaluate=evaluate)
        assert outcome.status == "succeeded"
        assert outcome.provenance["source"]["source_tree_sha256"] == original.identity.source_tree_sha256
    # Reusing the immutable original must never adopt an unverified source
    # snapshot. Repeated evaluations use the bound package or fresh preparation.
    with pytest.raises(PreparationError, match=r"changed|linked"):
        evaluate_prepared_plugin(original, options, work_dir=tmp_path, load="native", evaluate=evaluate)


def test_existing_case_aliases_do_not_break_runtime_identity_or_native_copy(tmp_path):
    root = _source(tmp_path / "Plugin")
    (root / "Run-Data").mkdir()
    alias = root.with_name("plugin")
    if not alias.exists():
        pytest.skip("requires a case-insensitive filesystem")
    output = alias / "run-data/nested/reports"
    package = bind_runtime_results(_prepare(root), output)
    run = _reserve_run_dir(output, "run")
    (run / "report.json").write_text("REPORT-CASE-ALIAS")
    verify_source_identity(package)
    wrapper = stage_wrapper_skill(package, tmp_path / "wrapper")
    tasks = generate_harbor_tasks(wrapper.path, tmp_path / "tasks")
    NativeRuntimeHooks(
        package.native_source,
        resolve_plugin_load("native", ["claude-code"], env_mode="docker"),
        package=package,
    ).stage_with_arm("claude-code", tasks)
    assert not any(
        b"REPORT-CASE-ALIAS" in path.read_bytes() for path in (tasks[0] / "environment").rglob("*") if path.is_file()
    )
    (root / "Run-Data/nested/authored.txt").write_text("changed")
    with pytest.raises(PreparationError, match="changed"):
        verify_source_identity(package)


def test_case_alias_of_source_root_is_not_a_results_destination(tmp_path):
    root = _source(tmp_path / "Plugin")
    alias = root.with_name("plugin")
    if not alias.exists():
        pytest.skip("requires a case-insensitive filesystem")
    with pytest.raises(PreparationError, match="must not contain"):
        bind_runtime_results(_prepare(root), alias)


def test_differently_cased_authored_directory_is_not_a_runtime_exclusion(tmp_path):
    root = _source(tmp_path / "plugin")
    authored = root / "Reports"
    authored.mkdir()
    output = root / "reports"
    if output.exists():
        pytest.skip("requires a case-sensitive filesystem")
    package = bind_runtime_results(_prepare(root), output)
    _reserve_run_dir(output, "run")
    verify_source_identity(package)
    (authored / "helper.py").write_text("authored mutation")
    with pytest.raises(PreparationError, match="changed"):
        verify_source_identity(package)
