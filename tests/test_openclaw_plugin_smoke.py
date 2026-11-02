# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The repeatable real-plugin smoke keeps staging and runtime claims separate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.ci import smoke_openclaw_plugins as smoke


def _plugin(root: Path, *, skills: bool) -> Path:
    root.mkdir()
    manifest = {
        "id": "portable-demo" if skills else "runtime-only-demo",
        "name": "Display Name Is Not Identity",
        "configSchema": {"type": "object"},
        "contracts": {"tools": ["demo_tool"]},
    }
    if skills:
        manifest["skills"] = ["portable"]
        skill = root / "portable/group/alpha"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: alpha\ndescription: Offline test guidance.\nmetadata:\n  author: Example <example@example.com>\n---\n# Alpha\nUse this guidance.\n",
            encoding="utf-8",
        )
    (root / "openclaw.plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "package.json").write_text(json.dumps({"openclaw": {"extensions": ["./index.js"]}}), encoding="utf-8")
    (root / "index.js").write_text("throw new Error('must not execute native code');", encoding="utf-8")
    return root


def test_real_plugin_smoke_covers_portable_and_runtime_only_boundaries(tmp_path: Path) -> None:
    plugins = {
        "portable": _plugin(tmp_path / "portable", skills=True),
        "runtime": _plugin(tmp_path / "runtime", skills=False),
    }
    result = smoke.run_smoke(plugins, iterations=2, trace_memory=True)
    assert result["observations_passed"] == 4
    assert result["live_agent_executed"] is False
    assert result["native_runtime_executed"] is False
    portable = result["plugins"]["portable"]
    assert portable["plugin_id"] == "portable-demo"
    assert portable["declared_skill_paths"] == ["portable/group/alpha"]
    assert portable["tier2_profile_members"] == ["alpha"]
    assert not portable["portable_tier3_skipped"]
    assert result["plugins"]["runtime"]["portable_tier3_skipped"]
    assert portable["runtime_components_unsupported"] == 2
    assert str(tmp_path) not in json.dumps(result)


def test_smoke_keeps_existing_schema_findings_visible(tmp_path: Path) -> None:
    root = _plugin(tmp_path / "plugin", skills=True)
    manifest = root / "portable/group/alpha/SKILL.md"
    manifest.write_text(
        "---\nname: alpha\ndescription: Missing evaluator author metadata.\n---\n# Alpha\n", encoding="utf-8"
    )
    result = smoke.inspect_plugin(root)
    assert not result["tier1_schema_passed"]
    assert result["tier1_findings"]["author_missing"] == 1
    assert result["declared_skill_paths"] == ["portable/group/alpha"]


@pytest.mark.parametrize("iterations", [0, 201])
def test_smoke_bounds_repetition(iterations: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 200"):
        smoke.run_smoke({}, iterations=iterations)


def test_smoke_detects_non_deterministic_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = iter([{"plugin_id": "first"}, {"plugin_id": "changed"}])
    monkeypatch.setattr(smoke, "inspect_plugin", lambda _root: next(calls))
    with pytest.raises(AssertionError, match="Non-deterministic"):
        smoke.run_smoke({"demo": tmp_path}, iterations=2)
