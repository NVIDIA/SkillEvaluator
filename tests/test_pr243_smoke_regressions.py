# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PR243: portable smoke must reject successful-looking native skips."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from scripts.ci import smoke_openclaw_plugins as smoke


def _plugin(root: Path, servers: dict) -> Path:
    root.mkdir()
    (root / "openclaw.plugin.json").write_text(
        json.dumps(
            {
                "id": "runtime-review",
                "configSchema": {},
                "mcpServers": servers,
            }
        ),
        encoding="utf-8",
    )
    (root / "package.json").write_text(
        json.dumps(
            {
                "openclaw": {"extensions": ["./index.js"]},
            }
        ),
        encoding="utf-8",
    )
    (root / "index.js").write_text('throw new Error("must not execute");', encoding="utf-8")
    return root


@pytest.mark.parametrize(
    "servers",
    [
        {},
        {"srv": {"command": "uvx", "args": ["example-mcp==1.2.3"]}},
        {"srv": {"url": "https://mcp.example.com/mcp", "enabled": False}},
    ],
)
def test_smoke_rejects_optional_skips_even_with_mcp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    servers: dict,
) -> None:
    original = smoke.prepare_plugin_eval_package

    def optional_skip(*args, **kwargs):
        return dataclasses.replace(
            original(*args, **kwargs),
            skipped=True,
            native_runtime_unavailable=False,
            unresolved_skill_refs=(),
            unresolved_rule_refs=(),
            unresolved_mcp_servers=(),
            mcp_unsupported_config=(),
            dependency_status_counts=(),
        )

    monkeypatch.setattr(smoke, "prepare_plugin_eval_package", optional_skip)
    with pytest.raises(AssertionError, match="INCOMPLETE"):
        smoke.inspect_plugin(_plugin(tmp_path / "plugin", servers))


def test_smoke_records_incomplete_skip_in_observation(tmp_path: Path) -> None:
    observed = smoke.inspect_plugin(_plugin(tmp_path / "plugin", {}))
    assert observed["portable_tier3_skipped"] is True
    assert observed["portable_tier3_incomplete_skip"] is True


@pytest.mark.parametrize("declared_root", [".", "./"])
@pytest.mark.parametrize("portable", [False, True])
def test_smoke_distinguishes_inventory_only_root_from_portable_members(
    tmp_path: Path,
    declared_root: str,
    portable: bool,
) -> None:
    root = _plugin(tmp_path / "plugin", {})
    skill = (
        "---\nname: root-guidance\ndescription: Offline guidance.\nmetadata:\n"
        "  author: Example <example@example.com>\n---\n# Instructions\nRead guidance.\n"
    )
    (root / "SKILL.md").write_text(skill, encoding="utf-8")
    manifest = root / "openclaw.plugin.json"
    data = json.loads(manifest.read_text())
    data["skills"] = [declared_root]
    if portable:
        member = root / "portable/alpha"
        member.mkdir(parents=True)
        (member / "SKILL.md").write_text(skill.replace("root-guidance", "alpha"), encoding="utf-8")
        data["skills"].append("portable")
    manifest.write_text(json.dumps(data), encoding="utf-8")
    observed = smoke.inspect_plugin(root)
    expected = ["portable/alpha"] if portable else []
    assert observed["declared_skill_paths"] == [".", *expected]
    assert observed["portable_tier3_skill_paths"] == expected
    assert observed["portable_tier3_skipped"] is (not portable)
    assert observed["portable_tier3_incomplete_skip"] is (not portable)
    assert observed["native_runtime_executed"] is False
