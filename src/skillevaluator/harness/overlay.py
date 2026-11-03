# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load harness profiles, compute skill facts and evaluate profile rules."""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from skillevaluator.constants import DESCRIPTION_TRIGGER_PATTERN, SCAN_EXCLUDED_DIRS, SKILL_MANIFEST_VARIANTS
from skillevaluator.validators.frontmatter_parser import FRONTMATTER_PATTERN
from skillevaluator.validators.schema import _duplicate_frontmatter_keys

PROFILES_DIR = Path(__file__).parent / "profiles"
HARNESS_STATUSES = ("loads", "degraded", "not_loaded")
_EFFECT_RANK = {"advisory": 0, "degraded": 1, "not_loaded": 2}
_TRIGGER_RE = re.compile(DESCRIPTION_TRIGGER_PATTERN, re.IGNORECASE)
_CLAUDE_ONLY_FIELDS = (
    "argument-hint",
    "when_to_use",
    "context",
    "agent",
    "model",
    "hooks",
)
_KNOWN_FIELDS = frozenset(
    (
        "name",
        "description",
        "license",
        "compatibility",
        "metadata",
        "allowed-tools",
        "disable-model-invocation",
        "user-invocable",
        "platforms",
        "version",
        "author",
        "tags",
        *_CLAUDE_ONLY_FIELDS,
    )
)
_FACT_RE = re.compile(r"\{([a-z_]+)\}")
_KNOWN_ROOTS = (
    ".agents/skills",
    ".claude/skills",
    ".codex/skills",
    ".github/skills",
    ".opencode/skills",
    ".opencode/skill",
    ".hermes/skills",
    ".copilot/skills",
    ".openclaw/skills",
    ".config/opencode/skills",
)


@dataclass
class HarnessResult:
    harness: str
    display_name: str
    status: str
    version: str
    verified_date: str
    findings: list[dict[str, str]] = field(default_factory=list)
    listed: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "listed": self.listed,
            "harness": self.harness,
            "display_name": self.display_name,
            "status": self.status,
            "verified_version": self.version,
            "verified_date": self.verified_date,
            "findings": self.findings,
        }


def load_profiles(names: str | list[str] = "all", profiles_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load profiles by name (comma list or ``all``); unknown names raise ``ValueError``."""
    directory = profiles_dir or PROFILES_DIR
    available = {p.stem: p for p in sorted(directory.glob("*.yaml"))}
    requested = names.split(",") if isinstance(names, str) else list(names)
    requested = [n.strip() for n in requested if n.strip()]
    if not requested or requested == ["all"]:
        requested = list(available)
    unknown = [n for n in requested if n not in available]
    if unknown:
        raise ValueError(f"Unknown harness: {', '.join(unknown)}. Supported: {', '.join(available)}")
    return [yaml.safe_load(available[n].read_text(encoding="utf-8")) for n in requested]


def _install_root(skill_path: Path) -> tuple[str | None, int]:
    parts = skill_path.resolve().parts
    for root in sorted(_KNOWN_ROOTS, key=lambda r: -len(r.split("/"))):
        size = len(root.split("/"))
        for index in range(len(parts) - size - 1, -1, -1):
            if "/".join(parts[index : index + size]) == root:
                return root, len(parts) - (index + size) - 1
    return None, 0


def _read_manifest(skill_path: Path) -> tuple[Path, bytes, str, str, dict[str, Any]]:
    manifest = next(
        (skill_path / n for n in SKILL_MANIFEST_VARIANTS if (skill_path / n).exists()), skill_path / "SKILL.md"
    )
    raw = manifest.read_bytes() if manifest.is_file() else b""
    text = raw.decode("utf-8-sig", errors="replace")
    match = FRONTMATTER_PATTERN.match(text)
    fm_yaml = match.group(1) if match else ""
    try:
        data = yaml.safe_load(fm_yaml) if match else None
    except yaml.YAMLError:
        data = None
    return manifest, raw, text, fm_yaml, data if isinstance(data, dict) else {}


def _skill_name(skill_path: Path, data: dict[str, Any]) -> str:
    name = data.get("name")
    return name.strip() if isinstance(name, str) and name.strip() else skill_path.name


def _is_true(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def _bundle_files(skill_path: Path, manifest: Path) -> list[Path]:
    files = []
    for path in sorted(skill_path.rglob("*")):
        parts = path.relative_to(skill_path).parts
        if path == manifest or any(part in SCAN_EXCLUDED_DIRS for part in parts):
            continue
        if path.is_symlink() or path.is_file():
            files.append(path)
    return files


def _bundle_facts(skill_path: Path, manifest: Path) -> dict[str, Any]:
    base = skill_path.resolve()
    links, escapes, largest, largest_bytes, largest_lines = [], [], "", 0, 0
    for path in _bundle_files(skill_path, manifest):
        rel = path.relative_to(skill_path).as_posix()
        if path.is_symlink():
            links.append(rel)
            target = path.resolve()
            if target != base and base not in target.parents:
                escapes.append(rel)
            continue
        size = path.stat().st_size
        if size > largest_bytes:
            with path.open("rb") as handle:
                largest_lines = sum(1 for _ in handle)
            largest, largest_bytes = rel, size
    return {
        "resource_symlinks": len(links),
        "resource_symlink_paths": ", ".join(links[:3]),
        "resource_symlink_escapes": len(escapes),
        "resource_symlink_escape_paths": ", ".join(escapes[:3]),
        "largest_resource": largest,
        "largest_resource_bytes": largest_bytes,
        "largest_resource_lines": largest_lines,
    }


def _codex_implicit_disabled(skill_path: Path) -> bool:
    policy_file = skill_path / "agents" / "openai.yaml"
    try:
        data = yaml.safe_load(policy_file.read_text(encoding="utf-8")) if policy_file.is_file() else None
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return False
    policy = data.get("policy") if isinstance(data, dict) else None
    return isinstance(policy, dict) and policy.get("allow_implicit_invocation") is False


def _openclaw_gates(data: dict[str, Any]) -> str:
    metadata = data.get("metadata")
    claw = metadata.get("openclaw") if isinstance(metadata, dict) else None
    if not isinstance(claw, dict):
        return ""
    requires = claw.get("requires") if isinstance(claw.get("requires"), dict) else {}
    gates = [
        f"{key} {', '.join(map(str, value)) if isinstance(value, list) else value}"
        for key in ("bins", "anyBins", "env", "config")
        if (value := requires.get(key))
    ]
    if claw.get("os"):
        gates.append(f"os {', '.join(map(str, claw['os'])) if isinstance(claw['os'], list) else claw['os']}")
    return "; ".join(gates)


def skill_facts(skill_path: Path, catalog: Catalog | None = None) -> dict[str, Any]:
    """Compute the facts profile rules refer to (see profiles README).

    ``catalog`` describes the other skills installed alongside this one; without it the
    catalog facts are zero and catalog rules are skipped.
    """
    manifest, raw, text, fm_yaml, data = _read_manifest(skill_path)
    description = data.get("description")
    tools = data.get("allowed-tools")
    trigger = _TRIGGER_RE.search(description) if isinstance(description, str) else None
    root, depth = _install_root(skill_path)
    target = "none"
    if skill_path.is_symlink():
        resolved = skill_path.resolve()
        base = skill_path.parent.resolve()
        target = "inside_root" if base in resolved.parents else "outside_root"
    names = {p.name for p in skill_path.iterdir()} if skill_path.is_dir() else set()
    match = FRONTMATTER_PATTERN.match(text)
    body = match.group(2) if match else text
    custom = {k: v for k, v in data.items() if str(k) not in _KNOWN_FIELDS}
    platforms = data.get("platforms")
    name = _skill_name(skill_path, data)
    duplicates = catalog.duplicates(skill_path, name) if catalog else []
    return {
        "skill_name": name,
        "description_text": description if isinstance(description, str) else "",
        "install_root": root,
        "grouping_depth": depth,
        "manifest_name_exact": "SKILL.md" in names,
        "nested_skill_md": any(
            p.parent != skill_path and not any(part in SCAN_EXCLUDED_DIRS for part in p.relative_to(skill_path).parts)
            for n in SKILL_MANIFEST_VARIANTS
            for p in skill_path.rglob(n)
        ),
        "skill_md_symlink": manifest.is_symlink(),
        "skill_dir_symlink_target": target,
        "skill_md_bytes": len(raw),
        "skill_md_lines": len(text.splitlines()),
        "description_missing": "description" not in data,
        "description_empty": isinstance(description, str) and not description.strip(),
        "description_type": type(description).__name__ if description is not None else "none",
        "description_codepoints": len(description) if isinstance(description, str) else 0,
        "description_utf16_units": len(description.encode("utf-16-le")) // 2 if isinstance(description, str) else 0,
        "trigger_offset": trigger.start() if trigger else None,
        "name_missing": "name" not in data,
        "frontmatter_bom": raw.startswith(b"\xef\xbb\xbf"),
        "frontmatter_duplicate_keys": bool(_duplicate_frontmatter_keys(fm_yaml)),
        "frontmatter_tab_indent": any(line.startswith("\t") for line in fm_yaml.splitlines()),
        "yaml_lenient_only": False,
        "allowed_tools_shape": "none"
        if tools is None
        else {str: "string", list: "list", dict: "map"}.get(type(tools), "other"),
        "disable_model_invocation": _is_true(data.get("disable-model-invocation")),
        "codex_implicit_disabled": _codex_implicit_disabled(skill_path),
        "claude_only_fields": ", ".join(f for f in _CLAUDE_ONLY_FIELDS if f in data),
        "custom_fields": ", ".join(map(str, custom)),
        "custom_field_chars": sum(len(v) if isinstance(v, str) else len(yaml.safe_dump(v)) for v in custom.values()),
        "platforms": ", ".join(map(str, platforms)) if isinstance(platforms, list) else str(platforms or ""),
        "openclaw_gates": _openclaw_gates(data),
        "uses_arguments_placeholder": "$ARGUMENTS" in body,
        "uses_claude_skill_dir": "${CLAUDE_SKILL_DIR}" in body,
        **_bundle_facts(skill_path, manifest),
        "catalog_size": catalog.size if catalog else 0,
        "catalog_desc_chars": catalog.desc_chars if catalog else 0,
        "duplicate_name": bool(duplicates),
        "duplicate_paths": ", ".join(duplicates[:3]),
    }


@dataclass
class Catalog:
    """The skills installed together in one folder: what list budgets and name clashes depend on."""

    names: dict[Path, str]
    size: int
    desc_chars: int
    base: Path | None = None

    @classmethod
    def of(cls, skills: list[Path], base: Path | None = None) -> Catalog:
        names, chars = {}, 0
        for skill in skills:
            _, _, _, _, data = _read_manifest(skill)
            names[skill.resolve()] = _skill_name(skill, data)
            description = data.get("description")
            chars += len(description) if isinstance(description, str) else 0
        return cls(names=names, size=len(skills), desc_chars=chars, base=base)

    def duplicates(self, skill: Path, name: str) -> list[str]:
        own = skill.resolve()
        return [self._display(other) for other, other_name in self.names.items() if other != own and other_name == name]

    def _display(self, path: Path) -> str:
        base = (self.base or path.parent).resolve()
        return path.relative_to(base).as_posix() if base in path.parents else str(path)


def _find_skills(folder: Path) -> list[Path]:
    found = {
        p.parent
        for n in SKILL_MANIFEST_VARIANTS
        for p in folder.rglob(n)
        if not any(part in SCAN_EXCLUDED_DIRS for part in p.relative_to(folder).parts)
    }
    return sorted(s for s in found if not any(parent in found for parent in s.parents))


def facts_for(skills: list[Path]) -> dict[Path, dict[str, Any]]:
    """Facts per skill, with the catalog they are installed in.

    Several skills form their own catalog. A single skill under a known skills folder
    (e.g. ``.claude/skills``) is measured against everything installed in that folder.
    """
    members, base = skills, None
    if len(skills) == 1:
        root, depth = _install_root(skills[0])
        if root is not None:
            base = skills[0].resolve().parents[depth]
            members = _find_skills(base) or skills
    elif skills:
        base = Path(os.path.commonpath([s.resolve() for s in skills]))
    catalog = Catalog.of(members, base)
    return {skill: skill_facts(skill, catalog) for skill in skills}


def listed_description(profile: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    """What the profile says this harness puts in the model's skill list for the description."""
    listing = profile.get("listing") or {}
    text = facts["description_text"]
    budget = listing.get("budget") or {}
    over_budget = facts["catalog_size"] and (
        facts["catalog_desc_chars"] > budget.get("desc_chars", float("inf"))
        or facts["catalog_size"] > budget.get("skills", float("inf"))
    )
    limit = listing.get("drop_over_utf16")
    if limit and facts["description_utf16_units"] > limit:
        return {"mode": "dropped", "text": ""}
    if not text or (listing.get("name_only_over") and len(text) > listing["name_only_over"]):
        return {"mode": "name_only", "text": ""}
    cut = budget.get("cut") if over_budget else listing.get("cut")
    if over_budget and cut == 0:
        return {"mode": "name_only", "text": ""}
    if cut and len(text) > cut:
        return {"mode": "cut", "text": text[:cut], "limit": cut}
    return {"mode": "full", "text": text}


class _Rule:
    _ALLOWED = (
        ast.Expression,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.Not,
        ast.Compare,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Attribute,
        ast.List,
        ast.Tuple,
        ast.Eq,
        ast.NotEq,
        ast.Gt,
        ast.GtE,
        ast.Lt,
        ast.LtE,
        ast.In,
        ast.NotIn,
        ast.Is,
        ast.IsNot,
    )

    @classmethod
    def evaluate(cls, expression: str, names: dict[str, Any]) -> bool:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, cls._ALLOWED):
                raise ValueError(f"Unsupported rule expression: {expression}")
        return bool(cls._eval(tree.body, names))

    @classmethod
    def _eval(cls, node: ast.AST, names: dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return names[node.id]
        if isinstance(node, ast.Attribute):
            return (cls._eval(node.value, names) or {}).get(node.attr, [])
        if isinstance(node, (ast.List, ast.Tuple)):
            return [cls._eval(e, names) for e in node.elts]
        if isinstance(node, ast.UnaryOp):
            return not cls._eval(node.operand, names)
        if isinstance(node, ast.BoolOp):
            values = (cls._eval(v, names) for v in node.values)
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare):
            left = cls._eval(node.left, names)
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                right = cls._eval(comparator, names)
                if left is None and not isinstance(op, (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)):
                    return False
                ok = {
                    ast.Eq: lambda a, b: a == b,
                    ast.NotEq: lambda a, b: a != b,
                    ast.Gt: lambda a, b: a > b,
                    ast.GtE: lambda a, b: a >= b,
                    ast.Lt: lambda a, b: a < b,
                    ast.LtE: lambda a, b: a <= b,
                    ast.In: lambda a, b: a in b,
                    ast.NotIn: lambda a, b: a not in b,
                    ast.Is: lambda a, b: a is b,
                    ast.IsNot: lambda a, b: a is not b,
                }[type(op)](left, right)
                if not ok:
                    return False
                left = right
            return True
        raise ValueError("Unsupported rule expression")


def evaluate_profile(profile: dict[str, Any], facts: dict[str, Any]) -> HarnessResult:
    """Evaluate one profile's rules against precomputed skill facts (stricter effect wins)."""
    roots = profile.get("roots", {})
    all_roots = [*roots.get("project", []), *roots.get("user", [])]
    names = {**facts, "always": True, "roots": _RootList(all_roots, roots)}
    findings: list[dict[str, str]] = []
    worst = -1
    for rule in profile.get("rules", []):
        expression = str(rule["when"])
        if "install_root" in expression and facts["install_root"] is None:
            continue
        if "catalog_" in expression and not facts["catalog_size"]:
            continue
        if expression != "always" and not _Rule.evaluate(expression, names):
            continue
        if expression == "always" and rule["effect"] == "advisory":
            continue
        finding = {"id": rule["id"], "effect": rule["effect"], "message": _fill(rule["message"], facts)}
        if rule.get("fix"):
            finding["fix"] = _fill(rule["fix"], facts)
        findings.append(finding)
        worst = max(worst, _EFFECT_RANK[rule["effect"]])
    status = "not_loaded" if worst == 2 else "degraded" if worst == 1 else "loads"
    verified = profile.get("verified", {})
    return HarnessResult(
        harness=profile["harness"],
        display_name=profile.get("display_name", profile["harness"]),
        status=status,
        version=str(verified.get("version", "")),
        verified_date=str(verified.get("date", "")),
        findings=findings,
        listed=listed_description(profile, facts),
    )


def _fill(text: str, facts: dict[str, Any]) -> str:
    """Substitute ``{fact}`` placeholders; numbers get thousands separators."""

    def value(match: re.Match[str]) -> str:
        fact = facts.get(match.group(1), match.group(0))
        return f"{fact:,}" if isinstance(fact, int) and not isinstance(fact, bool) else str(fact)

    return _FACT_RE.sub(value, str(text))


class _RootList(list):
    def __init__(self, items: list[str], groups: dict[str, Any]) -> None:
        super().__init__(items)
        self._groups = groups

    def get(self, key: str, default: Any = None) -> Any:
        return self._groups.get(key, default)


def evaluate_harnesses(
    skill_path: Path, harnesses: str | list[str] = "all", profiles_dir: Path | None = None
) -> list[HarnessResult]:
    """Evaluate one skill (and the catalog it is installed in) against the selected profiles."""
    facts = facts_for([skill_path])[skill_path]
    return [evaluate_profile(profile, facts) for profile in load_profiles(harnesses, profiles_dir)]
