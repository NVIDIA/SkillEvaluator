# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PR243 F2: a root SKILL.md is inventory, not a portable copy of the plugin."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.evaluation import EvaluationService
from skillevaluator.tier3.harbor.adapter import _copy_skill_dirs, validate_results_root_location
from skillevaluator.tier3.harbor.native_staging import build_native_task_staging, stage_native_bundle
from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package
from skillevaluator.tier3.plugin_native import HARNESS_ADAPTERS

_ENV_SENTINEL = "ROOT-ENV-MUST-NOT-REACH-AGENT"
_ANSWER_SENTINEL = "ROOT-DATASET-ANSWER-MUST-NOT-REACH-AGENT"
_ROOT_ONLY_FILES = {".env", ".env.local", "openclaw.plugin.json", "package.json", "index.js", "tests/plugin-evals.json"}


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _skill(path: Path, name: str) -> None:
    _write(
        path / "SKILL.md",
        f"---\nname: {name}\ndescription: Offline root staging regression.\nmetadata:\n"
        "  author: Example <example@example.com>\n---\n# Instructions\nUse this guidance.\n",
    )


def _plugin(root: Path, declared_root: str, *, portable: bool = True) -> Path:
    _skill(root, "root-guidance")
    skills = [declared_root]
    if portable:
        _skill(root / "portable/alpha", "alpha")
        skills.append("portable")
    _write(
        root / "openclaw.plugin.json",
        json.dumps(
            {
                "id": "root-regression",
                "configSchema": {"type": "object", "default": {"hostOnly": "RAW-METADATA"}},
                "skills": skills,
            }
        ),
    )
    _write(root / "package.json", json.dumps({"openclaw": {"extensions": ["./index.js"]}}))
    _write(root / "index.js", "throw new Error('runtime plugin code must never execute in this test');\n")
    _write(root / ".env", f"TOKEN={_ENV_SENTINEL}\n")
    _write(root / ".env.local", f"LOCAL_TOKEN={_ENV_SENTINEL}\n")
    dataset = json.dumps([{"id": "one", "prompt": "Use alpha", "expected_output": _ANSWER_SENTINEL}])
    _write(root / "tests/plugin-evals.json", dataset)
    _write(root / "evals/evals.json", dataset)
    _write(root / "evals/results/old-run/results.json", dataset)
    return root


def _files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def _root_skill_row(package) -> dict:
    return next(
        row
        for row in package.provenance()["component_coverage"]["components"]
        if row["type"] == "skill" and row["path"] == "."
    )


@pytest.mark.parametrize("declared_root", [".", "./"])
def test_f2_wrapper_stages_only_nonroot_members_and_no_root_private_files(tmp_path: Path, declared_root: str) -> None:
    root = _plugin(tmp_path / "demo-plugin", declared_root)
    package = prepare_plugin_eval_package(
        root,
        stage_root=tmp_path / "stage",
        evals_source=root / "tests/plugin-evals.json",
    )
    assert not package.skipped
    env = tmp_path / "task/environment"
    _copy_skill_dirs(
        env_dir=env,
        skill_path=package.package_path,
        reference_skills_dir=None,
        workspace_skill_paths=list(package.include_skills),
        has_skill=True,
        exclude_skill_name=None,
    )
    skills = env / "skills"
    files = _files(skills)
    # Inspect actual secure runtime copies, not just the planner's member list.
    leaked = {name for name in files if any(name.endswith("/" + raw) for raw in _ROOT_ONLY_FILES)}
    assert leaked == set()
    assert not (skills / root.name).exists()
    assert (skills / "alpha/SKILL.md").is_file()
    assert package.include_skills == ((root / "portable/alpha").resolve(),)
    row = _root_skill_row(package)
    assert row["state"] == "not_staged" and "inventoried only" in row["reason"]
    for file in skills.rglob("*"):
        if file.is_file():
            text = file.read_text(encoding="utf-8")
            assert _ENV_SENTINEL not in text and _ANSWER_SENTINEL not in text
            assert "RAW-METADATA" not in text


@pytest.mark.parametrize("declared_root", [".", "./"])
def test_f2_native_claude_filtered_plugin_tree_does_not_recopy_root_as_skill(
    tmp_path: Path, declared_root: str
) -> None:
    root = _plugin(tmp_path / "demo-plugin", declared_root)
    package = prepare_plugin_eval_package(
        root,
        stage_root=tmp_path / "stage",
        evals_source=root / "tests/plugin-evals.json",
        plugin_load="native",
        agents="claude-code",
        env_mode="docker",
    )
    assert package.native_source is not None
    staging = build_native_task_staging("claude-code", HARNESS_ADAPTERS["claude-code"], package.native_source)
    env = tmp_path / "task/environment"
    env.mkdir(parents=True)
    stage_native_bundle(env, staging, excluded_roots=(root / "evals/results",))
    copied = env / "skilleval/native/claude-code/plugin"
    files = _files(copied)
    # The primary whole-plugin copy is intentional in native mode, including
    # runtime code/manifests. Its established env/evaluator exclusions must hold.
    assert {"openclaw.plugin.json", "package.json", "index.js", "SKILL.md"} <= files
    assert (copied / "portable/alpha/SKILL.md").is_file()
    assert not (copied / "skills" / root.name).exists()
    assert all(Path(name).name not in {".env", ".env.local"} for name in files)
    assert all(not name.endswith("tests/plugin-evals.json") for name in files)
    assert all("evals/" not in name for name in files)
    for name in files:
        text = (copied / name).read_text(encoding="utf-8")
        assert _ENV_SENTINEL not in text and _ANSWER_SENTINEL not in text
    assert package.include_skills == ((root / "portable/alpha").resolve(),)
    assert root.resolve() not in package.native_source.member_skills
    assert all(source.resolve() != root.resolve() for source, _dest in staging.bundle.trees)


@pytest.mark.parametrize("declared_root", [".", "./"])
@pytest.mark.parametrize("plugin_load", ["wrapper", "native"])
def test_f2_default_source_results_root_is_not_a_runtime_skill_source(
    tmp_path: Path, declared_root: str, plugin_load: str
) -> None:
    root = _plugin(tmp_path / "demo-plugin", declared_root)
    package = prepare_plugin_eval_package(
        root,
        stage_root=tmp_path / "stage",
        evals_source=root / "tests/plugin-evals.json",
        plugin_load=plugin_load,
        agents="claude-code",
        env_mode="docker",
    )
    validate_results_root_location(
        package.package_path,
        root / "evals/results",
        workspace_skill_paths=package.include_skills,
    )
    # The guard itself still rejects results written into the real member.
    with pytest.raises(ValueError, match="runtime skill source"):
        validate_results_root_location(
            package.package_path,
            root / "portable/alpha/results",
            workspace_skill_paths=package.include_skills,
        )


@pytest.mark.parametrize("alias", ["direct", "dot", "parent", "symlink"])
def test_f2_include_skills_cannot_reintroduce_the_plugin_root(tmp_path: Path, alias: str) -> None:
    root = _plugin(tmp_path / "demo-plugin", ".")
    aliases = {"direct": root, "dot": root / ".", "parent": root / "portable/.."}
    if alias == "symlink":
        link = tmp_path / "root-alias"
        link.symlink_to(root, target_is_directory=True)
        aliases[alias] = link
    with pytest.raises(ValueError, match=r"plugin root.*member skill"):
        prepare_plugin_eval_package(
            root,
            stage_root=tmp_path / "stage",
            evals_source=root / "tests/plugin-evals.json",
            include_skills=(aliases[alias],),
        )
    assert not (tmp_path / "stage").exists()


@pytest.mark.parametrize("declared_root", [".", "./"])
def test_f2_root_only_package_is_an_honest_incomplete_skip(tmp_path: Path, declared_root: str) -> None:
    root = _plugin(tmp_path / "demo-plugin", declared_root, portable=False)
    package = prepare_plugin_eval_package(
        root,
        stage_root=tmp_path / "stage",
        evals_source=root / "tests/plugin-evals.json",
    )
    assert package.skipped and package.incomplete_skip
    assert package.package_path is None and package.include_skills == ()
    assert package.provenance()["partial"]
    row = _root_skill_row(package)
    assert row["state"] == "not_staged" and "inventoried only" in row["reason"]


def test_f2_root_only_cli_reports_incomplete_without_launching_an_agent(tmp_path: Path, monkeypatch) -> None:
    root = _plugin(tmp_path / "demo-plugin", ".", portable=False)

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("An inventory-only root skill cannot run a portable evaluation")

    monkeypatch.setattr(EvaluationService, "evaluate", must_not_run)
    reports = tmp_path / "reports"
    invoked = CliRunner().invoke(cli, ["tier3", "evaluate-plugin", str(root), "-r", "json", "-o", str(reports)])
    assert invoked.exit_code == 1, invoked.output
    assert "INCOMPLETE" in invoked.output
    report = json.loads(next(reports.glob("*.json")).read_text())
    assert report["overall_status"] == "incomplete"
    assert "inventoried only" in json.dumps(report)
