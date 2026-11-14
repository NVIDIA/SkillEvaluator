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

from skillevaluator.constants import SKILL_MANIFEST_VARIANTS
from skillevaluator.validators.frontmatter_parser import FRONTMATTER_PATTERN
from skillevaluator.validators.schema import _duplicate_frontmatter_keys

PROFILES_DIR = Path(__file__).parent / "profiles"
HARNESS_STATUSES = ("loads", "degraded", "not_loaded")
_EFFECT_RANK = {"advisory": 0, "degraded": 1, "not_loaded": 2}
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
_FACT_RE = re.compile(r"\{([a-z0-9_]+)\}")
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


def _link_location_parts(path: Path) -> tuple[str, ...]:
    # Like os.path.normpath, but ".." after a symlink steps to the link target's parent, as the filesystem does.
    parts: list[str] = []
    for part in path.absolute().parts:
        if part != "..":
            parts.append(part)
            continue
        here = Path(*parts)
        if here.is_symlink():
            parts = list(here.resolve().parent.parts)
        elif len(parts) > 1:
            parts.pop()
    return tuple(parts)


def _install_root(skill_path: Path) -> tuple[str | None, int]:
    # Harnesses scan the folder a skill is installed in, so a symlinked skill counts where the link is.
    parts = _link_location_parts(skill_path)
    try:
        home = Path.home()
        homes = {_link_location_parts(home), home.resolve().parts}
    except (KeyError, RuntimeError):
        homes = set()
    for root in sorted(_KNOWN_ROOTS, key=lambda r: -len(r.split("/"))):
        size = len(root.split("/"))
        for index in range(len(parts) - size - 1, -1, -1):
            if "/".join(parts[index : index + size]) == root:
                user = parts[:index] in homes
                return f"~/{root}" if user else root, len(parts) - (index + size) - 1
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


def _tab_indent_breaks_yaml(fm_yaml: str) -> bool:
    """A tab in a line's indentation that makes the frontmatter invalid YAML (a tab inside a block scalar is fine)."""
    if not any("\t" in line[: len(line) - len(line.lstrip())] for line in fm_yaml.splitlines()):
        return False
    try:
        yaml.safe_load(fm_yaml)
    except yaml.YAMLError:
        return True
    return False


def skill_facts(skill_path: Path) -> dict[str, Any]:
    """Compute the facts profile rules refer to."""
    manifest, raw, text, fm_yaml, data = _read_manifest(skill_path)
    description = data.get("description")
    root, depth = _install_root(skill_path)
    custom = {k: v for k, v in data.items() if str(k) not in _KNOWN_FIELDS}
    name = _skill_name(skill_path, data)
    return {
        "skill_name": name,
        "install_root": root,
        "grouping_depth": depth,
        "frontmatter_missing": manifest.is_file() and FRONTMATTER_PATTERN.match(text) is None,
        "description_type": type(description).__name__ if description is not None else "none",
        "frontmatter_bom": raw.startswith(b"\xef\xbb\xbf"),
        "frontmatter_duplicate_keys": bool(_duplicate_frontmatter_keys(fm_yaml)),
        "frontmatter_tab_indent": _tab_indent_breaks_yaml(fm_yaml),
        "custom_fields": ", ".join(map(str, custom)),
        "custom_field_chars": sum(len(v) if isinstance(v, str) else len(yaml.safe_dump(v)) for v in custom.values()),
    }


def facts_for(skills: list[Path]) -> dict[Path, dict[str, Any]]:
    """Facts per skill."""
    return {skill: skill_facts(skill) for skill in skills}


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


def model_config(profile: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve the user's setup for one profile: ``context_tokens`` (0 = unknown), ``custom_provider`` and
    ``explicit_invocation`` (skills are activated by an explicit mention rather than opened by the model) and
    ``headless`` (a non-interactive run, e.g. ``claude -p``, with default permissions) and ``native_root``
    (evaluate as if installed in each harness's own skills root, so root rules are skipped).

    An unknown window falls back to the profile's ``config_defaults`` for a custom provider when one is set.
    """
    config = config or {}
    custom = bool(config.get("custom_provider", False))
    tokens = int(config.get("context_tokens") or 0)
    if not tokens and custom:
        tokens = int(profile.get("config_defaults", {}).get("custom_provider_context_tokens", 0))
    explicit = bool(config.get("explicit_invocation", False))
    headless = bool(config.get("headless", False))
    native_root = bool(config.get("native_root", False))
    return {
        "context_tokens": tokens,
        "custom_provider": custom,
        "explicit_invocation": explicit,
        "headless": headless,
        "native_root": native_root,
    }


def evaluate_profile(
    profile: dict[str, Any], facts: dict[str, Any], config: dict[str, Any] | None = None
) -> HarnessResult:
    """Evaluate one profile's rules against precomputed skill facts (stricter effect wins).

    ``config`` describes the user's model setup (see :func:`model_config`); rules read it as
    ``context_tokens``, ``custom_provider``, ``explicit_invocation`` and ``headless``. Without it, rules assume an
    unknown context, a native provider, model-initiated activation and an interactive session.
    """
    roots = profile.get("roots", {})
    all_roots = [*roots.get("project", []), *roots.get("user", [])]
    setup = model_config(profile, config)
    names = {**facts, **setup, "always": True, "roots": _RootList(all_roots, roots)}
    findings: list[dict[str, str]] = []
    worst = -1
    for rule in profile.get("rules", []):
        expression = str(rule["when"])
        if "install_root" in expression and (facts["install_root"] is None or names["native_root"]):
            continue
        if expression != "always" and not _Rule.evaluate(expression, names):
            continue
        if expression == "always" and rule["effect"] == "advisory":
            continue
        finding = {"id": rule["id"], "effect": rule["effect"], "message": _fill(rule["message"], facts)}
        for key in ("fix", "outcome"):
            if rule.get(key):
                finding[key] = _fill(rule[key], facts)
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
    skill_path: Path,
    harnesses: str | list[str] = "all",
    profiles_dir: Path | None = None,
    config: dict[str, Any] | None = None,
) -> list[HarnessResult]:
    """Evaluate one skill (and the catalog it is installed in) against the selected profiles."""
    facts = facts_for([skill_path])[skill_path]
    return [evaluate_profile(profile, facts, config) for profile in load_profiles(harnesses, profiles_dir)]
