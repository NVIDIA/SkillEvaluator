# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static plugin facts and per-harness plugin rules.

A profile's ``plugin`` section has ``rules`` (format acceptance, same expression
language as skill rules). Like the skill overlay, this is advisory and never
changes Tier 1 results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skillevaluator.harness.overlay import _EFFECT_RANK, HarnessResult, _fill, _Rule

MANIFESTS = {
    "claude": ".claude-plugin/plugin.json",
    "codex": ".codex-plugin/plugin.json",
    "cursor": ".cursor-plugin/plugin.json",
    "copilot": ".github/plugin/plugin.json",
    "copilot_alt": ".plugin/plugin.json",
    "root_json": "plugin.json",
    "hermes_native": "plugin.yaml",
}
AGENT_PLUGINS_SCHEMA = "https://agent-plugins.org/schemas/"
# Folders a harness can pick up without any manifest (Claude Code inline plugins).
DEFAULT_DIRS = ("skills", "commands", "agents", "hooks")


def is_plugin_root(path: Path) -> bool:
    return any((path / rel).is_file() for rel in MANIFESTS.values())


def plugin_facts(root: Path) -> dict[str, Any]:
    """Facts about a plugin folder: manifest formats, manifest validity and name."""
    facts: dict[str, Any] = {"plugin_name": root.name, "manifest_invalid": False}
    for fmt, rel in MANIFESTS.items():
        path = root / rel
        facts[f"fmt_{fmt}"] = path.is_file()
        if not path.is_file() or path.suffix != ".json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            facts["manifest_invalid"] = True
            continue
        if isinstance(data, dict) and isinstance(data.get("name"), str) and data["name"]:
            facts["plugin_name"] = data["name"]
        if fmt == "root_json":
            facts["fmt_agent_v1"] = isinstance(data, dict) and str(data.get("$schema", "")).startswith(
                AGENT_PLUGINS_SCHEMA
            )
    facts["fmt_copilot"] = facts["fmt_copilot"] or facts.pop("fmt_copilot_alt")
    facts.setdefault("fmt_agent_v1", False)
    facts["any_manifest"] = any(v for k, v in facts.items() if k.startswith("fmt_"))
    facts["default_dirs"] = any((root / d).is_dir() for d in DEFAULT_DIRS)
    return facts


def evaluate_plugin(profile: dict[str, Any], facts: dict[str, Any]) -> HarnessResult:
    """Apply a profile's plugin rules."""
    section = profile.get("plugin") or {}
    findings: list[dict[str, str]] = []
    worst = -1
    for rule in section.get("rules", []):
        if not _Rule.evaluate(str(rule["when"]), {**facts, "always": True}):
            continue
        finding = {"id": rule["id"], "effect": rule["effect"], "message": _fill(rule["message"], facts)}
        for key in ("fix", "outcome"):
            if rule.get(key):
                finding[key] = _fill(rule[key], facts)
        findings.append(finding)
        worst = max(worst, _EFFECT_RANK[rule["effect"]])
    verified = profile.get("verified", {})
    return HarnessResult(
        harness=profile["harness"],
        display_name=profile.get("display_name", profile["harness"]),
        status="not_loaded" if worst == 2 else "degraded" if worst == 1 else "loads",
        version=str(verified.get("version", "")),
        verified_date=str(verified.get("date", "")),
        findings=findings,
    )
