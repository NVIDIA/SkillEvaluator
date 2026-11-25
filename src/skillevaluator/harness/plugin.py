# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static plugin facts and per-harness plugin rules.

A profile's ``plugin`` section has ``rules`` (format acceptance, same expression
language as skill rules) and ``components`` (what the pinned harness version did
with each component kind in canary probes, or that no valid probe covered it). Like
the skill overlay, this is advisory and never changes Tier 1 results.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

from skillevaluator.constants import PLUGIN_CONFIG_MAX_BYTES
from skillevaluator.harness.overlay import _EFFECT_RANK, HarnessResult, _fill, _Rule, version_finding
from skillevaluator.utils.structured_data import StructuredDataLimitError, StructuredDataSyntaxError, load_bounded_yaml

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
COMPONENTS = {
    "skills": lambda r: len(_skill_sizes(r)),
    "commands": lambda r: len(list((r / "commands").rglob("*.md"))) if (r / "commands").is_dir() else 0,
    "agents": lambda r: len(list((r / "agents").rglob("*.md"))) if (r / "agents").is_dir() else 0,
    "hooks": lambda r: _json_keys(r / "hooks" / "hooks.json", "hooks") or _json_keys(r / "hooks.json", "hooks"),
    "mcp": lambda r: _json_keys(r / ".mcp.json", "mcpServers") or _json_keys(r / "mcp.json", "mcpServers"),
    "lsp": lambda r: _json_keys(r / ".lsp.json"),
}
# Folders a harness can pick up without any manifest (Claude Code inline plugins).
DEFAULT_DIRS = ("skills", "commands", "agents", "hooks")
DECLARED_COMPONENT_KEYS = ("skills", "commands", "agents", "hooks", "mcpServers", "lspServers")
# What a component outcome means for the plugin's status. Outcomes come from canary probes.
# Only a negative result from a valid probe degrades the plugin; ``untested`` is a note for a kind no
# valid probe covered on that harness. Where a result holds for one bundle format or one shape of
# component only, a profile rule whose id ends in the kind (HM-PLUGIN-MCP) replaces the generic finding
# and says what was tested.
OUTCOME_EFFECT = {
    "delivered": None,
    "listed": None,
    "runs": None,
    "deferred": "advisory",
    "trust_gated": "advisory",
    "untested": "advisory",
    "registered": "degraded",
    "not_observed": "degraded",
}
OUTCOME_TEXT = {
    "trust_gated": "{n} {kind}: run only after the user trusts the plugin's current content (or with a per-run trust bypass).",
    "deferred": "{n} {kind}: registered; tool schemas reach the model only when it searches for or asks to use them.",
    "untested": "{n} {kind}: not tested on this harness.",
    "registered": "{n} {kind}: started or registered, but nothing reached the model when tested.",
    "not_observed": "{n} {kind}: did not reach the model or run when tested.",
}
OUTCOME_FIX = {"untested": "Try them on this harness before you rely on them."}


def _skill_sizes(root: Path) -> dict[Path, int]:
    """Size of each SKILL.md under skills/ that is a readable regular file.

    Pipes, sockets, devices, dangling or looping links and unreadable files are skipped. Only a
    regular file is ever opened: opening a named pipe with no writer would block forever.
    """
    sizes: dict[Path, int] = {}
    for path in (root / "skills").rglob("SKILL.md") if (root / "skills").is_dir() else []:
        try:
            info = path.stat()
            if not stat.S_ISREG(info.st_mode):
                continue
            with path.open("rb"):
                sizes[path] = info.st_size
        except OSError:
            continue
    return sizes


def _json_keys(path: Path, key: str | None = None) -> int:
    if not path.is_file():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError, RecursionError):
        return 1
    section = data.get(key, {}) if key and isinstance(data, dict) else data
    return len(section) if isinstance(section, (dict, list)) else 1


def _hermes_manifest_status(path: Path) -> str:
    """Check the native manifest without treating local parser limits as a loader rejection."""
    try:
        with path.open("rb") as handle:
            raw = handle.read(PLUGIN_CONFIG_MAX_BYTES + 1)
        if len(raw) > PLUGIN_CONFIG_MAX_BYTES:
            return "unchecked"
        data = load_bounded_yaml(raw.decode("utf-8-sig"))
    except (OSError, StructuredDataLimitError):
        return "unchecked"
    except (UnicodeDecodeError, StructuredDataSyntaxError):
        return "invalid"
    # Hermes treats an empty document as a mapping with default fields.
    return "valid" if isinstance(data or {}, dict) else "invalid"


def is_plugin_root(path: Path) -> bool:
    return any((path / rel).is_file() for rel in MANIFESTS.values())


def plugin_facts(root: Path) -> dict[str, Any]:
    """Facts about a plugin folder: manifest formats, manifest validity, name and component counts."""
    facts: dict[str, Any] = {"plugin_name": root.name, "manifest_invalid": False, "declares_components": False}
    for fmt, rel in MANIFESTS.items():
        path = root / rel
        facts[f"fmt_{fmt}"] = path.is_file()
        if not path.is_file() or path.suffix != ".json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError, RecursionError):
            facts["manifest_invalid"] = True
            facts[f"fmt_{fmt}"] = False
            continue
        if isinstance(data, dict) and isinstance(data.get("name"), str) and data["name"]:
            facts["plugin_name"] = data["name"]
        if isinstance(data, dict) and any(data.get(key) for key in DECLARED_COMPONENT_KEYS):
            facts["declares_components"] = True
        if fmt == "root_json":
            facts["fmt_agent_v1"] = isinstance(data, dict) and str(data.get("$schema", "")).startswith(
                AGENT_PLUGINS_SCHEMA
            )
    facts["fmt_copilot"] = facts["fmt_copilot"] or facts.pop("fmt_copilot_alt")
    facts.setdefault("fmt_agent_v1", False)
    facts["hermes_manifest_status"] = (
        _hermes_manifest_status(root / MANIFESTS["hermes_native"]) if facts["fmt_hermes_native"] else "absent"
    )
    facts["any_manifest"] = any(v for k, v in facts.items() if k.startswith("fmt_"))
    facts["default_dirs"] = any((root / d).is_dir() for d in DEFAULT_DIRS)
    facts["components"] = {kind: count(root) for kind, count in COMPONENTS.items()}
    facts["max_skill_bytes"] = max(_skill_sizes(root).values(), default=0)
    return facts


def evaluate_plugin(
    profile: dict[str, Any], facts: dict[str, Any], config: dict[str, Any] | None = None
) -> HarnessResult:
    """Apply a profile's plugin rules, then add a finding for each component kind the harness does not
    deliver or was not tested with.

    Raises ``ValueError`` for an unknown component outcome or a missing fact read during rule evaluation.
    """
    section = profile.get("plugin") or {}
    components = section.get("components")
    if components is None:
        components = {}
    elif not isinstance(components, dict):
        raise ValueError(f"{profile['harness']} profile: plugin components must be a mapping")
    for outcome in components.values():
        if not isinstance(outcome, str) or outcome not in OUTCOME_EFFECT:
            raise ValueError(f"{profile['harness']} profile: unknown plugin component outcome {outcome!r}")
    findings: list[dict[str, str]] = []
    worst = -1
    for rule in section.get("rules", []):
        try:
            fires = _Rule.evaluate(str(rule["when"]), {**facts, "always": True})
        except KeyError as exc:
            raise ValueError(
                f"{profile['harness']} profile: rule {rule['id']} reads the fact {exc}, which this version does not provide"
            ) from exc
        except (AttributeError, TypeError) as exc:
            raise ValueError(f"{profile['harness']} profile: cannot evaluate rule {rule['id']}: {exc}") from exc
        if not fires:
            continue
        finding = {"id": rule["id"], "effect": rule["effect"], "message": _fill(rule["message"], facts)}
        for key in ("fix", "versions", "outcome", "confidence"):
            if rule.get(key):
                finding[key] = _fill(rule[key], facts)
        findings.append(finding)
        worst = max(worst, _EFFECT_RANK[rule["effect"]])
    if worst < 2:
        first = (profile.get("rules") or section.get("rules") or [{"id": profile["harness"]}])[0]
        prefix = str(first["id"]).split("-")[0]
        for kind, outcome in components.items():
            n = facts["components"].get(kind, 0)
            effect = OUTCOME_EFFECT[outcome]
            if not n or effect is None or any(f["id"].endswith(f"-{kind.upper()}") for f in findings):
                continue
            finding = {
                "id": f"{prefix}-PLUGIN-{kind.upper()}",
                "effect": effect,
                "outcome": outcome.replace("_", " "),
                "message": OUTCOME_TEXT[outcome].format(n=n, kind=kind),
            }
            if outcome in OUTCOME_FIX:
                finding["fix"] = OUTCOME_FIX[outcome]
            findings.append(finding)
            worst = max(worst, _EFFECT_RANK[effect])
        if facts.get("any_manifest") and not facts["declares_components"] and not any(facts["components"].values()):
            findings.append(
                {
                    "id": f"{prefix}-PLUGIN-NO-COMPONENTS",
                    "effect": "advisory",
                    "message": "No skills, commands, agents, hooks, MCP or LSP servers in the default places, "
                    "so the plugin installs but adds nothing.",
                    "fix": "Add components in the default folders, or check that the manifest points to them.",
                }
            )
            worst = max(worst, _EFFECT_RANK["advisory"])
    declared = str(((config or {}).get("harness_versions") or {}).get(profile["harness"], ""))
    if drift := version_finding(profile, declared):
        findings.append(drift)
    verified = profile.get("verified", {})
    return HarnessResult(
        harness=profile["harness"],
        display_name=profile.get("display_name", profile["harness"]),
        status="not_loaded" if worst == 2 else "degraded" if worst == 1 else "loads",
        version=str(verified.get("version", "")),
        verified_date=str(verified.get("date", "")),
        findings=findings,
    )
