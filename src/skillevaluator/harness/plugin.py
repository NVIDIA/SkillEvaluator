# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static plugin facts and per-harness plugin rules.

A profile's ``plugin`` section has ``rules`` (format acceptance, same expression
language as skill rules) and ``components`` (what the pinned harness version did
with each component kind in canary probes). It may
also set ``mcp_flat`` when the harness reads a ``.mcp.json`` that names its servers without
the ``mcpServers`` wrapper. Like the skill overlay, this is advisory and never changes Tier 1 results.
"""

from __future__ import annotations

import errno
import json
import os
import stat
from pathlib import Path
from typing import Any

from skillevaluator.harness.overlay import _EFFECT_RANK, HarnessResult, _fill, _Rule, version_finding
from skillevaluator.plugin_mcp import collect_mcp_declarations, flat_mcp_endpoints, has_mcp_endpoint
from skillevaluator.plugin_paths import PluginRootReader

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
# Manifest keys that only describe the plugin. Any other key may declare or configure something a harness loads.
METADATA_KEYS = (
    "$schema",
    "name",
    "displayName",
    "version",
    "description",
    "keywords",
    "license",
    "author",
    "homepage",
    "repository",
)
# Top-level documentation and housekeeping entries, which add nothing to a plugin. A documentation file
# has exactly one of these names and no extension or a plain-text one; any other name may be code.
DOC_NAMES = ("readme", "license", "licence", "notice", "changelog")
DOC_SUFFIXES = ("", ".md", ".txt", ".rst")
HOUSEKEEPING = (".git", ".gitignore", ".gitattributes", ".DS_Store")
# Manifest rejection checks from the MCP collector, distinct from broken config files the client can skip.
MCP_MANIFEST_REJECTION_CHECKS = {"plugin_component_path_style", "mcp_servers_not_object", "mcp_servers_entry_invalid"}
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


def _json_keys(path: Path, key: str | None = None) -> int:
    if not path.is_file():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError, RecursionError):
        return 1
    section = data.get(key, {}) if key and isinstance(data, dict) else data
    return len(section) if isinstance(section, (dict, list)) else 1


def _default_mcp_counts(root: Path) -> tuple[int, int]:
    """Count usable endpoints in the flat layout and in either layout of Claude's default file."""
    path = root / ".mcp.json"
    try:
        if not path.is_file():  # only a regular file is ever opened: a named pipe with no writer would block
            return 0, 0
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError, RecursionError):
        return 0, 0
    if not isinstance(data, dict):
        return 0, 0
    if "mcpServers" in data:
        servers = data["mcpServers"]
        return 0, sum(has_mcp_endpoint(config) for config in servers.values()) if isinstance(servers, dict) else 0
    count = len(flat_mcp_endpoints(data))
    return count, count


def _is_doc(name: str) -> bool:
    lowered = Path(name.lower())
    return lowered.stem in DOC_NAMES and lowered.suffix in DOC_SUFFIXES


def _present(path: Path, link: bool = False) -> bool:
    """Whether something is there (with ``link``: whether it is a link).

    A dangling or looping link is not there. An entry that cannot be examined counts as there, so a
    folder is never called empty because of something this check could not look at.
    """
    try:
        info = path.lstat() if link else path.stat()
    except OSError as exc:
        return exc.errno not in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP)
    return stat.S_ISLNK(info.st_mode) if link else True


def _has_other_files(root: Path) -> bool:
    """Whether the folder holds anything besides its manifests, plain documentation and housekeeping files.

    Any other file may be a component this check does not model, or one in a place or a shape it does
    not read, so only a folder without any can be called empty. Broken links do not count. No file is
    opened, and a subfolder that cannot be listed counts as holding something.
    """
    manifests = {root / rel for rel in MANIFESTS.values()}
    unlisted: list[OSError] = []
    for folder, dirnames, filenames in os.walk(root, onerror=unlisted.append):
        here = Path(folder)
        filenames = [name for name in filenames if name not in HOUSEKEEPING]
        if here == root:
            dirnames[:] = [name for name in dirnames if name not in HOUSEKEEPING]
            filenames = [name for name in filenames if not _is_doc(name)]
        linked_folders = [name for name in dirnames if _present(here / name, link=True)]
        if any((here / name) not in manifests and _present(here / name) for name in filenames + linked_folders):
            return True
    return bool(unlisted)


def is_plugin_root(path: Path) -> bool:
    return any((path / rel).is_file() for rel in MANIFESTS.values())


def plugin_facts(root: Path) -> dict[str, Any]:
    """Facts about a plugin folder: manifest formats, manifest validity, name and component counts.

    ``declares_components`` says a manifest has a key beyond the descriptive ones, or is not an object.
    ``mcp_flat`` counts servers a ``.mcp.json`` names without the ``mcpServers`` wrapper, and
    ``mcp_claude`` counts usable default servers and any effective declarations from the Claude manifest.
    ``other_files`` says the folder holds something besides manifests and documentation.
    """
    facts: dict[str, Any] = {"plugin_name": root.name, "manifest_invalid": False, "declares_components": False}
    claude_manifest = None
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
        if not isinstance(data, dict) or set(data) - set(METADATA_KEYS):
            facts["declares_components"] = True
        if fmt == "claude" and isinstance(data, dict):
            claude_manifest = data
        if fmt == "root_json":
            facts["fmt_agent_v1"] = isinstance(data, dict) and str(data.get("$schema", "")).startswith(
                AGENT_PLUGINS_SCHEMA
            )
    facts["fmt_copilot"] = facts["fmt_copilot"] or facts.pop("fmt_copilot_alt")
    facts.setdefault("fmt_agent_v1", False)
    facts["any_manifest"] = any(v for k, v in facts.items() if k.startswith("fmt_"))
    facts["default_dirs"] = any((root / d).is_dir() for d in DEFAULT_DIRS)
    facts["components"] = {kind: count(root) for kind, count in COMPONENTS.items()}
    facts["mcp_flat"], facts["mcp_claude"] = _default_mcp_counts(root)
    facts["mcp_claude_manifest_invalid"] = False
    # Empty collections contribute no servers and leave default discovery unchanged.
    if claude_manifest and claude_manifest.get("mcpServers", {}) not in ({}, []):
        # Claude Code merges declared configurations with its default .mcp.json by server name.
        reader = PluginRootReader(root)
        manifest_rel = MANIFESTS["claude"]
        mcp = collect_mcp_declarations(
            reader,
            claude_manifest,
            contained=True,
            manifest_rel=manifest_rel,
            validate_servers=False,
            allow_partial_flat=True,
        )
        # A missing field is optional, while the Claude manifest schema refuses an explicit null.
        rejects_manifest = claude_manifest["mcpServers"] is None or any(
            finding.check_name in MCP_MANIFEST_REJECTION_CHECKS and finding.file_path == reader.display(manifest_rel)
            for finding in mcp.blocking_source_findings
        )
        facts["mcp_claude_manifest_invalid"] = rejects_manifest
        facts["mcp_claude"] = (
            0 if rejects_manifest else sum(has_mcp_endpoint(declaration.config) for declaration in mcp.effective)
        )
    facts["other_files"] = _has_other_files(root)
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
        first = (profile.get("rules") or section.get("rules") or [{"id": profile["harness"]}])[0]
        prefix = str(first["id"]).split("-")[0]
        for kind, outcome in (section.get("components") or {}).items():
            n = facts["components"].get(kind, 0)
            if kind == "mcp" and section.get("mcp_flat"):
                n = facts.get("mcp_flat") or n
                # Keep an explicit zero when Claude's own sources have no usable servers.
                if profile["harness"] == "claude-code":
                    n = facts.get("mcp_claude", n)
            effect = OUTCOME_EFFECT[outcome]
            if not n or effect is None or any(f["id"].endswith(f"-{kind.upper()}") for f in findings):
                continue
            findings.append(
                {
                    "id": f"{prefix}-PLUGIN-{kind.upper()}",
                    "effect": effect,
                    "outcome": outcome.replace("_", " "),
                    "message": OUTCOME_TEXT[outcome].format(n=n, kind=kind),
                }
            )
            worst = max(worst, _EFFECT_RANK[effect])
        if facts.get("mcp_flat") and not section.get("mcp_flat"):
            tested = (section.get("components") or {}).get("mcp") in ("delivered", "deferred")
            findings.append(
                {
                    "id": f"{prefix}-PLUGIN-MCP-FLAT",
                    "effect": "advisory",
                    "message": ".mcp.json names its servers at the top level, without the mcpServers wrapper. "
                    "Claude Code documents that shape; it was not tested on this harness.",
                    "fix": (
                        "Use the mcpServers wrapper, the shape this harness was tested with."
                        if tested
                        else "Check the harness's plugin documentation and try the configuration on that harness."
                    ),
                }
            )
            worst = max(worst, _EFFECT_RANK["advisory"])
        # Only a folder that holds nothing but a descriptive manifest is called empty: anything else may be
        # a component this check does not model. Components supplied from outside the folder are not read.
        if (
            facts.get("any_manifest")
            and not facts.get("manifest_invalid")
            and not facts.get("fmt_hermes_native")  # This check leaves the native YAML manifest contents unread.
            and not facts["declares_components"]
            and not facts.get("other_files")
            and not any(facts["components"].values())
        ):
            findings.append(
                {
                    "id": f"{prefix}-PLUGIN-NO-COMPONENTS",
                    "effect": "advisory",
                    "message": "The folder holds nothing but the manifest (README, license and changelog files "
                    "aside), and the manifest only describes the plugin: on its own the plugin adds nothing.",
                    "fix": "Add components, for example a skill at skills/<name>/SKILL.md. Components supplied "
                    "from outside the folder, such as by a marketplace entry, are not read by this check.",
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
