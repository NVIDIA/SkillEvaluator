# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The repeatable OpenClaw compatibility smoke over the private plugin helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci" / "smoke_openclaw_plugins.py"


def _load():
    spec = importlib.util.spec_from_file_location("smoke_openclaw_plugins", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load()


def _skill(name: str) -> str:
    return f"---\nname: {name}\ndescription: {name} guidance.\n---\nBody.\n"


def _leaks_host_path(text: str, path: Path) -> bool:
    # JSON escapes Windows separators, so check the raw, POSIX and JSON-escaped spellings.
    return any(form in text for form in {str(path), path.as_posix(), json.dumps(str(path))[1:-1]})


def _plugin(root: Path, files: dict[str, Any] | None = None, **fields: Any) -> Path:
    manifest = {"id": root.name, "configSchema": {"type": "object"}, **fields}
    for rel, value in {"openclaw.plugin.json": manifest, **(files or {})}.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
    return root


def test_smoke_covers_portable_root_and_runtime_only_boundaries(tmp_path: Path) -> None:
    portable = _plugin(
        tmp_path / "portable",
        {"SKILL.md": _skill("root"), "skills/alpha/SKILL.md": _skill("alpha")},
        skills=[".", "skills"],
    )
    runtime_only = _plugin(
        tmp_path / "runtime-only", {"package.json": {"openclaw": {"extensions": ["./index.js"]}}, "index.js": "x"}
    )
    result = smoke.run_smoke({"portable": portable, "runtime": runtime_only}, iterations=2)
    assert result["observations_passed"] == 4
    assert not result["native_runtime_executed"] and not result["live_agent_executed"]
    first, second = result["plugins"]["portable"], result["plugins"]["runtime"]
    assert first["declared_skill_paths"] == [".", "skills/alpha"]
    assert first["portable_tier3_skill_paths"] == ["skills/alpha"]
    assert first["portable_tier3_runnable"] and first["runtime_components_unsupported"] == 1
    assert not second["portable_tier3_runnable"]
    assert any("OpenClaw runtime" in reason for reason in second["portable_tier3_incomplete_reasons"])
    assert second["runtime_components_unsupported"] == 2
    assert not _leaks_host_path(json.dumps(result), tmp_path)


def test_smoke_keeps_existing_schema_findings_visible(tmp_path: Path) -> None:
    files = {"skills/alpha/SKILL.md": _skill("alpha"), ".env": "TOKEN=x"}
    result = smoke.run_smoke({"p": _plugin(tmp_path / "p", files, skills=["skills"])}, iterations=1)
    assert result["plugins"]["p"]["tier1_findings"]


@pytest.mark.parametrize("iterations", [0, smoke.MAX_ITERATIONS + 1])
def test_smoke_bounds_repetition(tmp_path: Path, iterations: int) -> None:
    with pytest.raises(ValueError, match="iterations"):
        smoke.run_smoke({"p": _plugin(tmp_path / "p")}, iterations=iterations)


def test_smoke_detects_non_deterministic_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observations = iter([{"value": 1}, {"value": 2}])
    monkeypatch.setattr(smoke, "inspect_plugin", lambda _root: next(observations))
    with pytest.raises(AssertionError, match="Non-deterministic"):
        smoke.run_smoke({"p": tmp_path}, iterations=2)


def test_smoke_rejects_other_manifest_formats(tmp_path: Path) -> None:
    (tmp_path / ".claude-plugin").mkdir()
    (tmp_path / ".claude-plugin" / "plugin.json").write_text('{"name": "demo"}', encoding="utf-8")
    with pytest.raises(ValueError, match="OpenClaw"):
        smoke.inspect_plugin(tmp_path)


def test_smoke_cli_writes_relative_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plugin = _plugin(tmp_path / "p", {"skills/alpha/SKILL.md": _skill("alpha")}, skills=["skills"])
    output = tmp_path / "out" / "smoke.json"
    smoke.main(["--plugin", f"p={plugin}", "--iterations", "1", "--output", str(output)])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["plugins"]["p"]["portable_tier3_skill_paths"] == ["skills/alpha"]
    assert not _leaks_host_path(output.read_text(encoding="utf-8"), tmp_path)
    assert "Passed 1 pipeline observations across 1 plugins" in capsys.readouterr().out
