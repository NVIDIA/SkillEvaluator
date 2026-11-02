# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load harness profiles, compute skill facts and evaluate profile rules."""

from __future__ import annotations

import ast
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

    def to_dict(self) -> dict[str, Any]:
        return {
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


def skill_facts(skill_path: Path) -> dict[str, Any]:
    """Compute the facts profile rules refer to (see profiles README)."""
    manifest = next(
        (skill_path / n for n in SKILL_MANIFEST_VARIANTS if (skill_path / n).exists()), skill_path / "SKILL.md"
    )
    raw = manifest.read_bytes() if manifest.exists() else b""
    text = raw.decode("utf-8-sig", errors="replace")
    match = FRONTMATTER_PATTERN.match(text)
    fm_yaml = match.group(1) if match else ""
    try:
        data = yaml.safe_load(fm_yaml) if match else None
    except yaml.YAMLError:
        data = None
    data = data if isinstance(data, dict) else {}
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
    return {
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
        "catalog_size": 0,
    }


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


def _evaluate_profile(profile: dict[str, Any], facts: dict[str, Any]) -> HarnessResult:
    roots = profile.get("roots", {})
    all_roots = [*roots.get("project", []), *roots.get("user", [])]
    names = {**facts, "always": True, "roots": _RootList(all_roots, roots)}
    findings: list[dict[str, str]] = []
    worst = -1
    for rule in profile.get("rules", []):
        expression = str(rule["when"])
        if "install_root" in expression and facts["install_root"] is None:
            continue
        if "catalog_size" in expression and not facts["catalog_size"]:
            continue
        if expression != "always" and not _Rule.evaluate(expression, names):
            continue
        if expression == "always" and rule["effect"] == "advisory":
            continue
        findings.append({"id": rule["id"], "effect": rule["effect"], "message": rule["message"]})
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
    )


class _RootList(list):
    def __init__(self, items: list[str], groups: dict[str, Any]) -> None:
        super().__init__(items)
        self._groups = groups

    def get(self, key: str, default: Any = None) -> Any:
        return self._groups.get(key, default)


def evaluate_harnesses(
    skill_path: Path, harnesses: str | list[str] = "all", profiles_dir: Path | None = None
) -> list[HarnessResult]:
    """Evaluate one skill against the selected harness profiles."""
    facts = skill_facts(skill_path)
    return [_evaluate_profile(profile, facts) for profile in load_profiles(harnesses, profiles_dir)]
