# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for per-entry eval input staging.

Ports Skill Evaluator 0.7.22 ``0d17f5e`` ("upload staged eval inputs to standard sandboxes")
into the in-process Tier 3 engine (``tier3/harbor/adapter.py``). Entries that
declare ``files`` stage only those refs; entries that omit ``files`` retain the
legacy shared ``evals/files/`` behavior. All refs retain traversal protection.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

import skillevaluator.tier3.harbor.adapter as adapter_module
from skillevaluator.tier3.harbor.adapter import (
    _entry_file_refs,
    _load_mcp_servers,
    _resolve_entry_file_ref,
    _stage_task_inputs,
    generate_harbor_tasks,
)


def _make_skill(tmp_path: Path) -> tuple[Path, Path, Path]:
    skill = tmp_path / "myskill"
    (skill / "SKILL.md").parent.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Test skill\n")
    evals = skill / "evals"
    files = evals / "files"
    files.mkdir(parents=True)
    (files / "global.txt").write_text("global")
    (files / "unrelated.txt").write_text("unrelated")
    data = evals / "data"
    data.mkdir()
    (data / "case1.txt").write_text("case1")
    env_dir = tmp_path / "task" / "environment"
    env_dir.mkdir(parents=True)
    return skill, evals, env_dir


class TestEntryFileRefs:
    def test_none_returns_empty(self):
        assert _entry_file_refs({"id": "t"}) == []

    def test_string_is_wrapped(self):
        assert _entry_file_refs({"files": "data/case1.txt"}) == ["data/case1.txt"]

    def test_explicit_null_returns_empty(self):
        assert _entry_file_refs({"files": None}) == []

    def test_list_is_passed_through(self):
        assert _entry_file_refs({"files": ["a/b.txt", " c/d.txt "]}) == ["a/b.txt", "c/d.txt"]

    def test_non_string_entry_rejected(self):
        with pytest.raises(ValueError, match="must be a string"):
            _entry_file_refs({"id": "t", "files": [123]})


class TestStageTaskInputs:
    def test_explicit_files_stage_only_declared_refs(self, tmp_path: Path):
        skill, evals, env_dir = _make_skill(tmp_path)
        entry = {"id": "t1", "files": ["data/case1.txt"]}
        staged = _stage_task_inputs(
            env_dir, input_files_dir=evals / "files", entry=entry, source_skill_path=skill, evals_dir=evals
        )
        assert staged is True
        paths = sorted(
            p.relative_to(env_dir / "input").as_posix() for p in (env_dir / "input").rglob("*") if p.is_file()
        )
        assert paths == ["data/case1.txt"]

    def test_explicit_file_under_shared_directory_stages_only_that_file(self, tmp_path: Path):
        skill, evals, env_dir = _make_skill(tmp_path)
        staged = _stage_task_inputs(
            env_dir,
            input_files_dir=evals / "files",
            entry={"id": "t1", "files": "files/global.txt"},
            source_skill_path=skill,
            evals_dir=evals,
        )
        assert staged is True
        paths = sorted(
            p.relative_to(env_dir / "input").as_posix() for p in (env_dir / "input").rglob("*") if p.is_file()
        )
        assert paths == ["global.txt"]

    def test_omitted_files_stages_entire_shared_directory(self, tmp_path: Path):
        skill, evals, env_dir = _make_skill(tmp_path)
        staged = _stage_task_inputs(
            env_dir,
            input_files_dir=evals / "files",
            entry={"id": "t1"},
            source_skill_path=skill,
            evals_dir=evals,
        )
        assert staged is True
        paths = sorted(
            p.relative_to(env_dir / "input").as_posix() for p in (env_dir / "input").rglob("*") if p.is_file()
        )
        assert paths == ["global.txt", "unrelated.txt"]

    @pytest.mark.parametrize("files", [[], None, "", "   "])
    def test_explicit_empty_files_stages_nothing_and_cleans_stale_input(self, tmp_path: Path, files: object):
        skill, evals, env_dir = _make_skill(tmp_path)
        input_dir = env_dir / "input"
        input_dir.mkdir()
        (input_dir / "stale.txt").write_text("stale")

        staged = _stage_task_inputs(
            env_dir,
            input_files_dir=evals / "files",
            entry={"id": "t1", "files": files},
            source_skill_path=skill,
            evals_dir=evals,
        )

        assert staged is False
        assert not input_dir.exists()

    @pytest.mark.parametrize("stale_kind", ["file", "symlink", "fifo"])
    def test_explicit_empty_files_safely_cleans_non_directory_input(self, tmp_path: Path, stale_kind: str):
        skill, evals, env_dir = _make_skill(tmp_path)
        input_path = env_dir / "input"
        symlink_target = tmp_path / "outside.txt"
        symlink_target.write_text("keep")
        if stale_kind == "file":
            input_path.write_text("stale")
        elif stale_kind == "symlink":
            input_path.symlink_to(symlink_target)
        else:
            if not hasattr(os, "mkfifo"):
                pytest.skip("FIFOs are unavailable on this platform")
            os.mkfifo(input_path)

        staged = _stage_task_inputs(
            env_dir,
            input_files_dir=evals / "files",
            entry={"id": "t1", "files": []},
            source_skill_path=skill,
            evals_dir=evals,
        )

        assert staged is False
        assert not os.path.lexists(input_path)
        assert symlink_target.read_text() == "keep"

    def test_explicit_refs_replace_stale_input(self, tmp_path: Path):
        skill, evals, env_dir = _make_skill(tmp_path)
        input_dir = env_dir / "input"
        input_dir.mkdir()
        (input_dir / "stale.txt").write_text("stale")

        staged = _stage_task_inputs(
            env_dir,
            input_files_dir=evals / "files",
            entry={"id": "t1", "files": ["data/case1.txt"]},
            source_skill_path=skill,
            evals_dir=evals,
        )

        assert staged is True
        paths = sorted(p.relative_to(input_dir).as_posix() for p in input_dir.rglob("*") if p.is_file())
        assert paths == ["data/case1.txt"]

    def test_no_inputs_returns_false(self, tmp_path: Path):
        skill = tmp_path / "myskill"
        evals = skill / "evals"
        evals.mkdir(parents=True)
        env_dir = tmp_path / "task" / "environment"
        env_dir.mkdir(parents=True)
        staged = _stage_task_inputs(
            env_dir, input_files_dir=None, entry={"id": "t"}, source_skill_path=skill, evals_dir=evals
        )
        assert staged is False

    def test_declared_file_symlink_to_undeclared_fixture_is_rejected(self, tmp_path: Path):
        skill, evals, env_dir = _make_skill(tmp_path)
        alias = evals / "files" / "alias.txt"
        try:
            alias.symlink_to("unrelated.txt")
        except OSError as exc:  # pragma: no cover - host policy, primarily native Windows
            pytest.skip(f"symlinks unavailable on this host: {exc}")

        with pytest.raises(ValueError, match="symlink"):
            _stage_task_inputs(
                env_dir,
                input_files_dir=evals / "files",
                entry={"id": "t1", "files": ["files/alias.txt"]},
                source_skill_path=skill,
                evals_dir=evals,
            )

    def test_declared_directory_with_nested_symlink_is_rejected(self, tmp_path: Path):
        skill, evals, env_dir = _make_skill(tmp_path)
        declared = evals / "data" / "declared"
        declared.mkdir()
        alias = declared / "alias.txt"
        try:
            alias.symlink_to("../../files/unrelated.txt")
        except OSError as exc:  # pragma: no cover - host policy, primarily native Windows
            pytest.skip(f"symlinks unavailable on this host: {exc}")

        with pytest.raises(ValueError, match="symlink"):
            _stage_task_inputs(
                env_dir,
                input_files_dir=evals / "files",
                entry={"id": "t1", "files": ["data/declared"]},
                source_skill_path=skill,
                evals_dir=evals,
            )

    def test_legacy_shared_directory_with_nested_symlink_is_rejected(self, tmp_path: Path):
        skill, evals, env_dir = _make_skill(tmp_path)
        alias = evals / "files" / "alias.txt"
        try:
            alias.symlink_to("unrelated.txt")
        except OSError as exc:  # pragma: no cover - host policy, primarily native Windows
            pytest.skip(f"symlinks unavailable on this host: {exc}")

        with pytest.raises(ValueError, match="symlink"):
            _stage_task_inputs(
                env_dir,
                input_files_dir=evals / "files",
                entry={"id": "t1"},
                source_skill_path=skill,
                evals_dir=evals,
            )

    def test_declared_hardlink_is_rejected(self, tmp_path: Path):
        skill, evals, env_dir = _make_skill(tmp_path)
        alias = evals / "data" / "alias.txt"
        try:
            os.link(evals / "files" / "unrelated.txt", alias)
        except OSError as exc:  # pragma: no cover - filesystem policy
            pytest.skip(f"hardlinks unavailable on this host: {exc}")

        with pytest.raises(ValueError, match=r"hardlink|hard link"):
            _stage_task_inputs(
                env_dir,
                input_files_dir=evals / "files",
                entry={"id": "t1", "files": ["data/alias.txt"]},
                source_skill_path=skill,
                evals_dir=evals,
            )


def test_generated_tasks_apply_per_entry_input_isolation(tmp_path: Path):
    skill, evals, _ = _make_skill(tmp_path)
    entries = [
        {"id": "legacy", "question": "legacy"},
        {"id": "selected", "question": "selected", "files": ["data/case1.txt"]},
        {"id": "empty", "question": "empty", "files": []},
        {"id": "null", "question": "null", "files": None},
    ]
    (evals / "evals.json").write_text(json.dumps(entries))

    task_dirs = generate_harbor_tasks(skill, tmp_path / "generated")
    tasks = {task.name: task for task in task_dirs}

    def staged_paths(case_id: str) -> list[str]:
        input_dir = tasks[case_id] / "environment" / "input"
        if not input_dir.exists():
            return []
        return sorted(path.relative_to(input_dir).as_posix() for path in input_dir.rglob("*") if path.is_file())

    assert staged_paths("legacy") == ["global.txt", "unrelated.txt"]
    assert staged_paths("selected") == ["data/case1.txt"]
    assert staged_paths("empty") == []
    assert staged_paths("null") == []

    for case_id in ("legacy", "selected"):
        dockerfile = (tasks[case_id] / "environment" / "Dockerfile").read_text()
        assert "COPY input/ /workspace/input/" in dockerfile
    for case_id in ("empty", "null"):
        dockerfile = (tasks[case_id] / "environment" / "Dockerfile").read_text()
        assert "COPY input/ /workspace/input/" not in dockerfile


def test_generation_uses_one_private_evals_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    skill, evals, _ = _make_skill(tmp_path)
    (skill / "SKILL.md").write_text("# Snapshot test\n", encoding="utf-8")
    (evals / "evals.json").write_text(
        json.dumps([{"id": "case", "question": "dataset-A"}]),
        encoding="utf-8",
    )
    (evals / "files" / "global.txt").write_text("fixture-A", encoding="utf-8")
    moved = tmp_path / "moved-evals"
    real_load = adapter_module._load_evals
    swapped = False

    def swap_original_evals_after_dataset_read(path: Path) -> list[dict[str, object]]:
        nonlocal swapped
        entries = real_load(path)
        if not swapped:
            swapped = True
            evals.rename(moved)
            replacement_files = evals / "files"
            replacement_files.mkdir(parents=True)
            (evals / "evals.json").write_text(
                json.dumps([{"id": "case", "question": "dataset-B"}]),
                encoding="utf-8",
            )
            (replacement_files / "global.txt").write_text("fixture-B", encoding="utf-8")
        return entries

    monkeypatch.setattr(adapter_module, "_load_evals", swap_original_evals_after_dataset_read)

    task = generate_harbor_tasks(skill, tmp_path / "generated", with_skill=False)[0]

    assert (task / "instruction.md").read_text(encoding="utf-8") == "dataset-A\n"
    assert (task / "environment" / "input" / "global.txt").read_text(encoding="utf-8") == "fixture-A"


@pytest.mark.parametrize("results_name", ("results", "Results"))
def test_private_evals_snapshot_ignores_results_through_platform_root_aliases(
    tmp_path: Path,
    results_name: str,
) -> None:
    skill, evals, _ = _make_skill(tmp_path)
    (skill / "SKILL.md").write_text("# Snapshot test\n", encoding="utf-8")
    (evals / "evals.json").write_text(
        json.dumps([{"id": "case", "question": "evaluate"}]),
        encoding="utf-8",
    )
    results_dir = evals / results_name
    run_dir = results_dir / "run-1"
    run_dir.mkdir(parents=True)
    try:
        (results_dir / "current").symlink_to(run_dir.name, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - host policy, primarily native Windows
        pytest.skip(f"symlinks unavailable on this host: {exc}")

    task = generate_harbor_tasks(skill, tmp_path / "generated", with_skill=False)[0]

    assert task.name == "case"


@pytest.mark.parametrize("base_image", ["", "example.invalid/eval-base:latest"])
def test_generated_task_copies_explicit_ref_without_shared_files_directory(tmp_path: Path, base_image: str):
    skill, evals, _ = _make_skill(tmp_path)
    shutil.rmtree(evals / "files")
    entries = [{"id": "selected", "question": "selected", "files": "data/case1.txt"}]
    (evals / "evals.json").write_text(json.dumps(entries))

    task = generate_harbor_tasks(skill, tmp_path / "generated", base_image=base_image)[0]

    input_dir = task / "environment" / "input"
    assert [path.relative_to(input_dir).as_posix() for path in input_dir.rglob("*") if path.is_file()] == [
        "data/case1.txt"
    ]
    dockerfile = (task / "environment" / "Dockerfile").read_text()
    assert "COPY input/ /workspace/input/" in dockerfile


class TestResolveEntryFileRef:
    def test_traversal_outside_evals_blocked(self, tmp_path: Path):
        skill, evals, _ = _make_skill(tmp_path)
        with pytest.raises((ValueError, FileNotFoundError)):
            _resolve_entry_file_ref(
                "../../etc/passwd", skill_path=skill, evals_dir=evals, input_files_dir=evals / "files"
            )

    def test_absolute_path_rejected(self, tmp_path: Path):
        skill, evals, _ = _make_skill(tmp_path)
        absolute_ref = str(Path(tmp_path.anchor) / "outside.txt")
        with pytest.raises(ValueError, match="relative to evals/"):
            _resolve_entry_file_ref(absolute_ref, skill_path=skill, evals_dir=evals, input_files_dir=None)

    def test_uri_scheme_rejected(self, tmp_path: Path):
        skill, evals, _ = _make_skill(tmp_path)
        with pytest.raises(ValueError, match="unsupported URI scheme"):
            _resolve_entry_file_ref("https://example.com/x", skill_path=skill, evals_dir=evals, input_files_dir=None)

    @pytest.mark.parametrize(
        "relative_path,is_directory",
        [
            ("evals.json", False),
            ("EVALS.JSON", False),
            ("dataset.yaml", False),
            ("config.yml", False),
            ("grader.py", False),
            ("grader.sh", False),
            ("EVAL.md", False),
            ("benchmark_" + "conversion_report.md", False),
            (".skillevaluator-generated-output", False),
            ("tests", True),
            ("harbor", True),
            ("environment", True),
            ("results", True),
        ],
    )
    def test_evaluator_only_assets_cannot_be_declared_as_task_input(
        self,
        tmp_path: Path,
        relative_path: str,
        is_directory: bool,
    ) -> None:
        skill, evals, env_dir = _make_skill(tmp_path)
        source = evals / relative_path
        if is_directory:
            source.mkdir()
            (source / "secret.txt").write_text("evaluator-only", encoding="utf-8")
        else:
            source.write_text("evaluator-only", encoding="utf-8")

        with pytest.raises(ValueError, match="evaluator-only"):
            _stage_task_inputs(
                env_dir,
                input_files_dir=evals / "files",
                entry={"id": "case", "files": [f"evals/{relative_path}"]},
                source_skill_path=skill,
                evals_dir=evals,
            )

        assert not (env_dir / "input").exists()


class TestLoadMcpServersSafety:
    def test_linked_mcp_configuration_is_rejected(self, tmp_path: Path) -> None:
        skill = tmp_path / "skill"
        environment = skill / "evals" / "environment"
        environment.mkdir(parents=True)
        outside = tmp_path / "outside-mcp_servers.toml"
        outside.write_text('[[mcp_servers]]\nname = "outside"\ncommand = "/bin/sh"\n', encoding="utf-8")
        link = environment / "mcp_servers.toml"
        try:
            link.symlink_to(outside)
        except OSError as exc:  # pragma: no cover - host policy, primarily native Windows
            pytest.skip(f"symlinks unavailable on this host: {exc}")

        with pytest.raises(ValueError, match="regular non-linked file"):
            _load_mcp_servers(skill)

    def test_hardlinked_mcp_configuration_is_rejected(self, tmp_path: Path) -> None:
        skill = tmp_path / "skill"
        environment = skill / "evals" / "environment"
        environment.mkdir(parents=True)
        outside = tmp_path / "outside-mcp.toml"
        outside.write_text('[[mcp_servers]]\nname = "outside"\ncommand = "/bin/sh"\n', encoding="utf-8")
        try:
            os.link(outside, environment / "mcp_servers.toml")
        except OSError as exc:  # pragma: no cover - filesystem policy
            pytest.skip(f"hardlinks unavailable on this host: {exc}")

        with pytest.raises(ValueError, match="regular non-linked file"):
            _load_mcp_servers(skill)

    def test_linked_mcp_parent_directory_is_rejected(self, tmp_path: Path) -> None:
        skill = tmp_path / "skill"
        evals = skill / "evals"
        evals.mkdir(parents=True)
        outside = tmp_path / "outside-environment"
        outside.mkdir()
        (outside / "mcp_servers.toml").write_text(
            '[[mcp_servers]]\nname = "outside"\ncommand = "/bin/sh"\n',
            encoding="utf-8",
        )
        try:
            (evals / "environment").symlink_to(outside, target_is_directory=True)
        except OSError as exc:  # pragma: no cover - host policy, primarily native Windows
            pytest.skip(f"symlinks unavailable on this host: {exc}")

        with pytest.raises(ValueError, match=r"symlink|reparse"):
            _load_mcp_servers(skill)

    def test_linked_mcp_parent_is_rejected_even_when_config_is_absent(self, tmp_path: Path) -> None:
        skill = tmp_path / "skill"
        evals = skill / "evals"
        evals.mkdir(parents=True)
        outside = tmp_path / "outside-environment"
        outside.mkdir()
        try:
            (evals / "environment").symlink_to(outside, target_is_directory=True)
        except OSError as exc:  # pragma: no cover - host policy, primarily native Windows
            pytest.skip(f"symlinks unavailable on this host: {exc}")

        with pytest.raises(ValueError, match=r"symlink|reparse"):
            _load_mcp_servers(skill)

    def test_mcp_parent_change_is_rejected_before_missing_config_return(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        skill = tmp_path / "skill"
        environment = skill / "evals" / "environment"
        environment.mkdir(parents=True)
        config = environment / "mcp_servers.toml"
        config.write_text('[[mcp_servers]]\nname = "original"\ncommand = "/bin/sh"\n', encoding="utf-8")
        moved = tmp_path / "moved-environment"
        real_lexists = adapter_module.os.path.lexists
        swapped = False

        def swap_before_missing_check(path: os.PathLike[str] | str) -> bool:
            nonlocal swapped
            if not swapped and Path(path) == config:
                swapped = True
                environment.rename(moved)
                environment.mkdir()
                return False
            return real_lexists(path)

        monkeypatch.setattr(adapter_module.os.path, "lexists", swap_before_missing_check)

        with pytest.raises(ValueError, match="changed"):
            _load_mcp_servers(skill)

    def test_mcp_parent_replaced_before_open_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        skill = tmp_path / "skill"
        environment = skill / "evals" / "environment"
        environment.mkdir(parents=True)
        config = environment / "mcp_servers.toml"
        config.write_text('[[mcp_servers]]\nname = "original"\ncommand = "/bin/sh"\n', encoding="utf-8")
        moved = tmp_path / "moved-environment"
        original_read_bytes = adapter_module.SecureRoot.read_bytes
        swapped = False

        def swap_parent(
            secure_root: adapter_module.SecureRoot,
            relative_path: Path,
            max_bytes: int,
            *,
            expected: os.stat_result | None = None,
        ) -> tuple[bytes, os.stat_result]:
            nonlocal swapped
            if not swapped and relative_path == Path("environment/mcp_servers.toml"):
                swapped = True
                environment.rename(moved)
                environment.symlink_to(moved, target_is_directory=True)
            return original_read_bytes(secure_root, relative_path, max_bytes, expected=expected)

        monkeypatch.setattr(adapter_module.SecureRoot, "read_bytes", swap_parent)

        with pytest.raises(ValueError, match=r"changed|symlink|reparse"):
            _load_mcp_servers(skill)

    def test_same_inode_mcp_rewrite_with_restored_mtime_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        skill = tmp_path / "skill"
        environment = skill / "evals" / "environment"
        environment.mkdir(parents=True)
        config = environment / "mcp_servers.toml"
        original_payload = '[[mcp_servers]]\nname = "alpha"\ncommand = "/bin/sh"\n'
        replacement_payload = '[[mcp_servers]]\nname = "bravo"\ncommand = "/bin/sh"\n'
        assert len(original_payload) == len(replacement_payload)
        config.write_text(original_payload, encoding="utf-8")
        original_metadata = config.stat()
        original_read = adapter_module.os.read
        mutated = False

        def rewrite_named_source(descriptor: int, size: int) -> bytes:
            nonlocal mutated
            payload = original_read(descriptor, size)
            if not mutated:
                mutated = True
                config.write_text(replacement_payload, encoding="utf-8")
                os.utime(
                    config,
                    ns=(original_metadata.st_atime_ns, original_metadata.st_mtime_ns),
                )
                mutated_metadata = config.stat()
                assert mutated_metadata.st_ino == original_metadata.st_ino
                assert mutated_metadata.st_size == original_metadata.st_size
                assert mutated_metadata.st_mtime_ns == original_metadata.st_mtime_ns
            return payload

        monkeypatch.setattr(adapter_module.os, "read", rewrite_named_source)

        with pytest.raises(ValueError, match="changed while it was read"):
            _load_mcp_servers(skill)

    def test_mcp_configuration_replaced_during_read_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        skill = tmp_path / "skill"
        environment = skill / "evals" / "environment"
        environment.mkdir(parents=True)
        config = environment / "mcp_servers.toml"
        config.write_text('[[mcp_servers]]\nname = "original"\ncommand = "/bin/sh"\n', encoding="utf-8")
        replacement = tmp_path / "replacement.toml"
        replacement.write_text('[[mcp_servers]]\nname = "replaced"\ncommand = "/bin/sh"\n', encoding="utf-8")
        original_read = adapter_module.os.read
        swapped = False

        def swap_named_source(descriptor: int, size: int) -> bytes:
            nonlocal swapped
            payload = original_read(descriptor, size)
            if not swapped:
                swapped = True
                config.rename(environment / "original-opened.toml")
                replacement.rename(config)
            return payload

        monkeypatch.setattr(adapter_module.os, "read", swap_named_source)

        with pytest.raises(ValueError, match="changed while it was read"):
            _load_mcp_servers(skill)

    def test_task_generation_rejects_linked_mcp_before_output_mutation(self, tmp_path: Path) -> None:
        skill = tmp_path / "skill"
        evals = skill / "evals"
        environment = evals / "environment"
        environment.mkdir(parents=True)
        (skill / "SKILL.md").write_text("# Test skill\n", encoding="utf-8")
        (evals / "evals.json").write_text(
            json.dumps([{"id": "case", "question": "Run the case"}]),
            encoding="utf-8",
        )
        outside = tmp_path / "outside-mcp.toml"
        outside.write_text('[[mcp_servers]]\nname = "outside"\ncommand = "/bin/sh"\n', encoding="utf-8")
        try:
            (environment / "mcp_servers.toml").symlink_to(outside)
        except OSError as exc:  # pragma: no cover - host policy, primarily native Windows
            pytest.skip(f"symlinks unavailable on this host: {exc}")
        output = tmp_path / "generated"

        with pytest.raises(ValueError, match=r"regular non-linked|symlink|reparse"):
            generate_harbor_tasks(skill, output, with_skill=False)

        assert not output.exists()
