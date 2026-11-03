# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local entrypoint dependencies survive staging and participate in parity."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from private_loose_fixtures import make_skills_only_plugin, native_source, write

from skillevaluator.tier3.loose_parts import LOOSE_BUNDLE_DIR, LOOSE_ROOT, build_loose_parts
from skillevaluator.tier3.plugin_native import CLAUDE_PLUGIN_ROOT_VAR

SUPPORT_FILES = {
    "servers/dependency.js": "module.exports = require('./lib/child.js');\n",
    "servers/lib/child.js": "module.exports = require('../../shared/value.json').value;\n",
    "shared/value.json": '{"value":"relative-dependency-ready"}\n',
}
STAGED_ROOTS = {
    "with_skill": Path("skilleval/native/claude-code/plugin"),
    "loose_parts": Path(f"skilleval/{LOOSE_BUNDLE_DIR}/root"),
}


@pytest.fixture
def dependency_tasks(tmp_path: Path):
    plugin = make_skills_only_plugin(tmp_path / "plugins")
    write(
        plugin / ".mcp.json",
        json.dumps(
            {"mcpServers": {"reltools": {"command": "node", "args": ["${CLAUDE_PLUGIN_ROOT}/servers/reltools.js"]}}}
        ),
    )
    write(plugin / "servers/reltools.js", "console.log(require('./dependency.js'));\n", mode=0o755)
    for relative, content in SUPPORT_FILES.items():
        write(plugin / relative, content)
    parts = build_loose_parts(native_source(plugin, tmp_path / "stage").native_source)
    assert parts.error is None, parts.error
    tasks = {}
    for arm in ("with_skill", "loose_parts", "sum_of_parts"):
        task = tmp_path / "tasks" / arm / "case-1"
        parts.stage_arm(arm, task / "environment")
        tasks[arm] = [task]
    return parts, tasks


def test_relative_entrypoint_dependency_closure_is_present_in_both_emitted_arms(dependency_tasks) -> None:
    parts, tasks = dependency_tasks
    for arm, relative_root in STAGED_ROOTS.items():
        root = tasks[arm][0] / "environment" / relative_root
        for relative, content in SUPPORT_FILES.items():
            target = root / relative
            assert target.is_file(), f"{arm} omitted {relative} needed by servers/reltools.js"
            assert target.read_text(encoding="utf-8") == content
    report = parts.content_report_for_tasks(tasks)
    assert report["equal_across_arms"] is True, report["mismatches"]


@pytest.mark.parametrize("arm", ["with_skill", "loose_parts"])
@pytest.mark.parametrize("change", ["modify", "remove", "chmod"])
def test_emitted_dependency_damage_makes_parity_fail(dependency_tasks, arm: str, change: str) -> None:
    parts, tasks = dependency_tasks
    baseline = parts.content_report_for_tasks(tasks)
    assert baseline["equal_across_arms"] is True, baseline
    target = tasks[arm][0] / "environment" / STAGED_ROOTS[arm] / "shared/value.json"
    assert target.is_file(), f"{arm} omitted the dependency before the mutation"
    if change == "modify":
        target.write_text('{"value":"wrong-result"}\n', encoding="utf-8")
    elif change == "remove":
        target.unlink()
    else:
        target.chmod(target.stat().st_mode ^ 0o100)

    report = parts.content_report_for_tasks(tasks)
    assert report["equal_across_arms"] is False, report
    assert report["mismatches"], report


def test_both_emitted_mcp_entrypoints_execute_with_relative_dependencies(dependency_tasks) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("A local Node executable is needed only for the entrypoint runtime check")
    _, tasks = dependency_tasks
    for arm, relative_root in STAGED_ROOTS.items():
        environment = tasks[arm][0] / "environment"
        root = environment / relative_root
        config = (
            root / ".mcp.json" if arm == "with_skill" else environment / "skilleval" / LOOSE_BUNDLE_DIR / "mcp.json"
        )
        server = json.loads(config.read_text(encoding="utf-8"))["mcpServers"]["reltools"]
        assert server["command"] == "node"
        args = [arg.replace(CLAUDE_PLUGIN_ROOT_VAR, str(root)).replace(LOOSE_ROOT, str(root)) for arg in server["args"]]
        # Translate only the emitted container root to this test's local root.
        assert args == [str(root / "servers/reltools.js")]
        environment_variables = {
            key: value for key, value in os.environ.items() if key not in {"NODE_OPTIONS", "NODE_PATH"}
        }
        completed = subprocess.run(
            [node, *args],
            cwd=environment,
            env=environment_variables,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert completed.returncode == 0, f"{arm}: {completed.stderr}"
        assert completed.stdout == "relative-dependency-ready\n", arm
