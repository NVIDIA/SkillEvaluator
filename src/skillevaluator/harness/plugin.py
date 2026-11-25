# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static plugin facts and per-harness plugin rules.

A profile's ``plugin`` section has ``rules`` (format acceptance, same expression
language as skill rules) and ``components`` (what the pinned harness version did
with each component kind in canary probes). Like the skill overlay, this is
advisory and never changes Tier 1 results.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

from skillevaluator.harness.overlay import _EFFECT_RANK, HarnessResult, _fill, _Rule, version_finding

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
# Root files that, with the folders above, make up the default Claude layout. Claude Code and OpenClaw
# both document these places for a bundle that has no manifest they read.
DEFAULT_FILES = (".mcp.json", ".lsp.json")
DECLARED_COMPONENT_KEYS = ("skills", "commands", "agents", "hooks", "mcpServers", "lspServers")
# What a component outcome means for the plugin's status. Outcomes come from canary probes.
OUTCOME_EFFECT = {
    "delivered": None,
    "listed": None,
    "runs": None,
    "deferred": "advisory",
    "trust_gated": "advisory",
    "registered": "degraded",
    "not_observed": "degraded",
}
OUTCOME_TEXT = {
    "trust_gated": "{n} {kind}: run only after the user trusts the plugin's current content (or with a per-run trust bypass).",
    "deferred": "{n} {kind}: registered; tool schemas reach the model only when it searches for or asks to use them.",
    "registered": "{n} {kind}: started or registered, but nothing reached the model when tested.",
    "not_observed": "{n} {kind}: not registered or delivered when tested.",
}


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


def _is_file(path: Path) -> bool:
    """``path.is_file()``, with a path that cannot be examined (no permission, name too long) read as absent."""
    try:
        return path.is_file()
    except OSError:
        return False


def _json_keys(path: Path, key: str | None = None) -> int:
    if not path.is_file():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError, RecursionError):
        return 1
    section = data.get(key, {}) if key and isinstance(data, dict) else data
    return len(section) if isinstance(section, (dict, list)) else 1


_UNPARSED = object()


def _name(data: Any) -> str:
    """The manifest's ``name`` when it is a non-empty string, else ``""``."""
    name = data.get("name") if isinstance(data, dict) else None
    return name if isinstance(name, str) else ""


def _claude_lenient(path: Path) -> Any:
    """A Claude manifest the strict read rejected, parsed the way Claude Code reads it, or ``_UNPARSED``.

    Claude Code drops a byte order mark and does not stop at bytes that are not UTF-8, so only a
    syntax error or a file that cannot be opened is a manifest it cannot read.
    """
    try:
        return json.loads(path.read_bytes().decode("utf-8", "replace").removeprefix("\ufeff"))
    except (OSError, ValueError, RecursionError):
        return _UNPARSED


def is_plugin_root(path: Path) -> bool:
    return any((path / rel).is_file() for rel in MANIFESTS.values())


def plugin_facts(root: Path) -> dict[str, Any]:
    """Facts about a plugin folder: manifest formats, manifest validity, name and component counts.

    ``invalid_<fmt>`` says that this format's JSON manifest is there but could not be read or parsed as
    strict UTF-8 JSON: a syntax error, a byte order mark, bytes that are not UTF-8, or a file that cannot be
    opened. Such a manifest file also clears its own ``fmt_<fmt>``. ``unparsed_claude`` is narrower: the
    Claude manifest also fails the more lenient read Claude Code does. ``nameless_claude`` says the Claude
    manifest parses but gives no ``name`` that is a non-empty string (a manifest that is not an object counts).
    ``default_dirs`` is any default component folder; ``default_layout`` also counts a root ``.mcp.json``
    or ``.lsp.json``.
    """
    facts: dict[str, Any] = {"plugin_name": root.name, "manifest_invalid": False, "declares_components": False}
    facts.update({f"invalid_{fmt}": False for fmt, rel in MANIFESTS.items() if rel.endswith(".json")})
    facts["nameless_claude"] = facts["unparsed_claude"] = False
    for fmt, rel in MANIFESTS.items():
        path = root / rel
        facts[f"fmt_{fmt}"] = path.is_file()
        if not path.is_file() or path.suffix != ".json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError, RecursionError):
            facts["manifest_invalid"] = True
            facts[f"invalid_{fmt}"] = True
            facts[f"fmt_{fmt}"] = False
            if fmt == "claude":
                lenient = _claude_lenient(path)
                facts["unparsed_claude"] = lenient is _UNPARSED
                facts["nameless_claude"] = lenient is not _UNPARSED and not _name(lenient)
            continue
        if name := _name(data):
            facts["plugin_name"] = name
        elif fmt == "claude":
            facts["nameless_claude"] = True
        if isinstance(data, dict) and any(data.get(key) for key in DECLARED_COMPONENT_KEYS):
            facts["declares_components"] = True
        if fmt == "root_json":
            facts["fmt_agent_v1"] = isinstance(data, dict) and str(data.get("$schema", "")).startswith(
                AGENT_PLUGINS_SCHEMA
            )
    facts["fmt_copilot"] = facts["fmt_copilot"] or facts.pop("fmt_copilot_alt")
    facts["invalid_copilot"] = facts.pop("invalid_copilot_alt") or facts["invalid_copilot"]
    facts.setdefault("fmt_agent_v1", False)
    facts["any_manifest"] = any(v for k, v in facts.items() if k.startswith("fmt_"))
    facts["default_dirs"] = any((root / d).is_dir() for d in DEFAULT_DIRS)
    facts["default_layout"] = facts["default_dirs"] or any(_is_file(root / f) for f in DEFAULT_FILES)
    facts["components"] = {kind: count(root) for kind, count in COMPONENTS.items()}
    facts["max_skill_bytes"] = max(_skill_sizes(root).values(), default=0)
    return facts


def evaluate_plugin(
    profile: dict[str, Any], facts: dict[str, Any], config: dict[str, Any] | None = None
) -> HarnessResult:
    """Apply a profile's plugin rules; components the harness does not deliver become findings."""
    section = profile.get("plugin") or {}
    findings: list[dict[str, str]] = []
    worst = -1
    for rule in section.get("rules", []):
        if not _Rule.evaluate(str(rule["when"]), {**facts, "always": True}):
            continue
        finding = {"id": rule["id"], "effect": rule["effect"], "message": _fill(rule["message"], facts)}
        for key in ("fix", "versions", "outcome", "confidence"):
            if rule.get(key):
                finding[key] = _fill(rule[key], facts)
        findings.append(finding)
        worst = max(worst, _EFFECT_RANK[rule["effect"]])
    if worst < 2:
        for kind, outcome in (section.get("components") or {}).items():
            n = facts["components"].get(kind, 0)
            effect = OUTCOME_EFFECT[outcome]
            if not n or effect is None or any(f["id"].endswith(f"-{kind.upper()}") for f in findings):
                continue
            prefix = str(profile["rules"][0]["id"]).split("-")[0]
            findings.append(
                {
                    "id": f"{prefix}-PLUGIN-{kind.upper()}",
                    "effect": effect,
                    "outcome": outcome.replace("_", " "),
                    "message": OUTCOME_TEXT[outcome].format(n=n, kind=kind),
                }
            )
            worst = max(worst, _EFFECT_RANK[effect])
        if facts.get("any_manifest") and not facts["declares_components"] and not any(facts["components"].values()):
            prefix = str(profile["rules"][0]["id"]).split("-")[0]
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
