# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load harness profiles, compute skill facts and evaluate profile rules."""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from skillevaluator.constants import (
    DESCRIPTION_TRIGGER_PATTERN,
    SCAN_EXCLUDED_DIRS,
    SKILL_MANIFEST_VARIANTS,
    SKILL_NOT_SHIPPED_DIRS,
)
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
_FACT_RE = re.compile(r"\{([a-z0-9_]+)\}")
_DOLLAR_N_RE = re.compile(r"(?<!\\)\$(\d+)(?!\w)")
# Always a price: cents or thousands ($4.00), "per"/"each"/USD ($5 per month), a billing unit ($5/month),
# or a spaced slash before another amount ($4 / $20).
_PRICE_TAIL_RE = re.compile(
    r"[.,]\d|\s+(?:per|each|usd)\b|\s*/\s*(?:mo(?:nth)?|y(?:ea)?r|day|week|h(?:ou)?r|user|seat|m?tok)\b(?!/)|\s+/\s+[$\d]",
    re.IGNORECASE,
)
_PRICE_HEAD_RE = re.compile(r"\$\d[\d.,]*\s+/\s+$")
# A bare $N past this index ($5, $20) is an amount, not an argument, unless a path or a closing quote/brace follows.
_MAX_ARG_INDEX = 3
_ARG_TAIL_RE = re.compile(r"\s*(?:/[^\s$\d]|~)|[\"'`}\]]")
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
    catalog: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "listed": self.listed,
            "harness": self.harness,
            "display_name": self.display_name,
            "status": self.status,
            "verified_version": self.version,
            "verified_date": self.verified_date,
            "findings": self.findings,
            "catalog": self.catalog,
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


# OpenClaw 2026.10.9 cjk-chars.ts: CJK code points cost 4-16 chars, every other UTF-16 unit 1.
_OPENCLAW_CJK = (
    (3, ((0xB7, 0xB7), (0x3000, 0x319F), (0x4E00, 0x9FA5), (0xAC00, 0xD7AF), (0xFF01, 0xFF60))),
    (
        11,
        (
            (0x1100, 0x11FF),
            (0x2E80, 0x2FFF),
            (0x31A0, 0x4DFF),
            (0x9FA6, 0x9FFF),
            (0xA000, 0xA4FF),
            (0xA700, 0xA707),
            (0xA960, 0xA97F),
            (0xD7B0, 0xD7FF),
            (0xF900, 0xFAFF),
        ),
    ),
    (
        7,
        (
            (0x2C7, 0x2C7),
            (0x2C9, 0x2CB),
            (0x2D9, 0x2D9),
            (0x2EA, 0x2EB),
            (0x305, 0x305),
            (0x323, 0x323),
            (0xFE10, 0xFE4F),
            (0xFF61, 0xFFDC),
            (0xFFE0, 0xFFE6),
        ),
    ),
    (10, ((0x1D360, 0x1D371),)),
    (
        14,
        (
            (0x16FE0, 0x16FFF),
            (0x1AFF0, 0x1AFFF),
            (0x1B000, 0x1B16F),
            (0x1F200, 0x1F2FF),
            (0x20000, 0x2FA1F),
            (0x30000, 0x3347F),
        ),
    ),
)
# Room OpenClaw keeps for its truncation notice; the skills_read path adds a JSON wrapper and cuts ~1% sooner.
_OPENCLAW_READ_OVERHEAD = 73


def _openclaw_read_chars(text: str) -> int:
    """SKILL.md size as OpenClaw's tool-result cap measures it."""
    extra = 0
    for ch in text:
        cp = ord(ch)
        if cp >= 0x80:
            extra += next((w for w, rs in _OPENCLAW_CJK if any(a <= cp <= b for a, b in rs)), 0)
    return _OPENCLAW_READ_OVERHEAD + len(text.encode("utf-16-le", "surrogatepass")) // 2 + extra


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


def _is_true(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def _shipped_paths(skill_path: Path) -> list[Path]:
    """Files and symlinks a skill ships, without descending into VCS or cache folders."""
    found = []
    for folder, dirs, files in os.walk(skill_path):
        dirs[:] = [d for d in dirs if d not in SKILL_NOT_SHIPPED_DIRS]
        here = Path(folder)
        found.extend(here / name for name in files)
        found.extend(here / name for name in dirs if (here / name).is_symlink())
    return sorted(found)


def _bundle_files(skill_path: Path, manifest: Path) -> list[Path]:
    return [path for path in _shipped_paths(skill_path) if path != manifest]


_HERMES_LINKED_DIRS = ("references", "templates", "assets", "scripts")


def _paths(items: list[str]) -> str:
    return ", ".join(items[:3]) + (f" +{len(items) - 3} more" if len(items) > 3 else "")


def _read_text(path: Path) -> tuple[int, list[str]] | None:
    """Size and lines of a text file a harness could read; None for binary data and files that can't be read.

    The check is advisory, so anything odd is skipped instead of raised: binary data (a NUL byte in the
    first 8 KiB), a pipe or device that would block, a broken or looping link, a file without permission.
    """
    try:
        if not path.is_file():
            return None
        with path.open("rb") as handle:
            head = handle.read(8192)
            if b"\0" in head:
                return None
            data = head + handle.read()
    except OSError:
        return None
    return len(data), data.decode("utf-8", "replace").splitlines()


def _bundle_facts(skill_path: Path, manifest: Path) -> dict[str, Any]:
    base = skill_path.resolve()
    links, escapes, sizes, long_lines, unlisted = [], [], [], [], []
    for path in _bundle_files(skill_path, manifest):
        rel = path.relative_to(skill_path).as_posix()
        if path.is_symlink():
            links.append(rel)
            target = Path(os.path.realpath(path))  # unlike Path.resolve(), no error on a symlink loop
            if target == base or base in target.parents:
                continue  # counted under its own path
            escapes.append(rel)  # OpenCode follows it, so what it points to is read (and cut) like any file
        elif rel.split("/", 1)[0] not in _HERMES_LINKED_DIRS:
            unlisted.append(rel)
        text = _read_text(path)
        if text is not None:
            size, lines = text
            sizes.append((rel, size, len(lines)))
            long_lines += [rel] * sum(len(line) > 2000 for line in lines)
    largest = max(sizes, key=lambda item: item[1], default=("", 0, 0))
    paged = [rel for rel, size, count in sizes if size > 51200 or count > 2000]
    # Only OpenCode is known to follow symlinks to files outside the skill; OpenClaw's reads are not tested there.
    multi_read = [rel for rel, size, _ in sizes if size > 64000 and rel not in escapes]
    return {
        "resource_symlinks": len(links),
        "resource_symlink_paths": _paths(links),
        "resource_symlink_escapes": len(escapes),
        "resource_symlink_escape_paths": _paths(escapes),
        "largest_resource": largest[0],
        "largest_resource_bytes": largest[1],
        "largest_resource_lines": largest[2],
        "files_over_51200": len(paged),
        "files_over_51200_paths": _paths(paged),
        "files_over_64000": len(multi_read),
        "files_over_64000_paths": _paths(multi_read),
        "long_line_count": len(long_lines),
        "long_line_files": _paths(list(dict.fromkeys(long_lines))),
        "hermes_unlisted_files": len(unlisted),
        "hermes_unlisted_dirs": _paths(list(dict.fromkeys(rel.split("/", 1)[0] for rel in unlisted))),
    }


def _is_price(line: str, match: re.Match[str]) -> bool:
    """A price ($4.00, $5 per month, $20 on its own) rather than an argument ($1, $1 /path/to/project)."""
    if _PRICE_TAIL_RE.match(line, match.end()) or _PRICE_HEAD_RE.search(line, 0, match.start()):
        return True
    return int(match.group(1)) > _MAX_ARG_INDEX and not _ARG_TAIL_RE.match(line, match.end())


def _dollar_n(body: str, first_line: int) -> dict[str, Any]:
    """Count ``$<number>`` tokens: price-like ones ($4.00, $5, $4 / $20 per MTok) versus placeholders ($1, $1 /path)."""
    prices, placeholders, lines = 0, 0, []
    for number, line in enumerate(body.splitlines(), first_line):
        for match in _DOLLAR_N_RE.finditer(line):
            if _is_price(line, match):
                prices += 1
                if number not in lines:
                    lines.append(number)
            else:
                placeholders += 1
    shown = ", ".join(map(str, lines[:5])) + (f" +{len(lines) - 5} more" if len(lines) > 5 else "")
    return {"dollar_literal_count": prices, "dollar_literal_lines": shown, "dollar_placeholder_count": placeholders}


def _codex_implicit_disabled(skill_path: Path) -> bool:
    policy_file = skill_path / "agents" / "openai.yaml"
    try:
        data = yaml.safe_load(policy_file.read_text(encoding="utf-8")) if policy_file.is_file() else None
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return False
    policy = data.get("policy") if isinstance(data, dict) else None
    return isinstance(policy, dict) and policy.get("allow_implicit_invocation") is False


_HERMES_OS = {"linux": "linux", "macos": "darwin", "windows": "win32"}


def _excluded_os(platforms: Any) -> str:
    """Operating systems Hermes hides the skill on (it maps names, then prefix-matches sys.platform)."""
    if not platforms:
        return ""
    wanted = [str(p).lower().strip() for p in (platforms if isinstance(platforms, list) else [platforms])]
    mapped = [_HERMES_OS.get(p, p) for p in wanted]
    return ", ".join(os for os, sys_name in _HERMES_OS.items() if not any(sys_name.startswith(m) for m in mapped))


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
        base = (Path(*_link_location_parts(skill_path)).parents[depth] if root else skill_path.parent).resolve()
        target = "inside_root" if base in resolved.parents else "outside_root"
    names = {p.name for p in skill_path.iterdir()} if skill_path.is_dir() else set()
    nested = (
        [p for p in _shipped_paths(skill_path) if p.name in SKILL_MANIFEST_VARIANTS and p.parent != skill_path]
        if skill_path.is_dir()
        else []
    )
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
        "nested_skill_md": bool(nested),
        "nested_skill_md_exact": any(p.name == "SKILL.md" and not p.is_symlink() for p in nested),
        "skill_md_symlink": manifest.is_symlink(),
        "skill_dir_symlink_target": target,
        "skill_md_bytes": len(raw),
        "skill_md_lines": len(text.splitlines()),
        "openclaw_read_chars": _openclaw_read_chars(text),
        # OpenCode strips front matter and wraps the body before its tool-output cut.
        "opencode_body_output_bytes": len(f'<skill_content name="{name}">\n# Skill: {name}\n\n{body.strip()}'.encode()),
        "opencode_body_output_lines": 3 + len(body.strip().splitlines()),
        "frontmatter_missing": manifest.is_file() and FRONTMATTER_PATTERN.match(text) is None,
        "description_missing": "description" not in data,
        "description_empty": isinstance(description, str) and not description.strip(),
        "description_type": type(description).__name__ if description is not None else "none",
        "description_codepoints": len(description) if isinstance(description, str) else 0,
        "description_has_quote": isinstance(description, str) and '"' in description,
        "description_utf16_units": _utf16_units(description),
        "trigger_end": trigger.end() if trigger else None,
        "name_missing": "name" not in data,
        "frontmatter_bom": raw.startswith(b"\xef\xbb\xbf"),
        "frontmatter_duplicate_keys": bool(_duplicate_frontmatter_keys(fm_yaml)),
        "frontmatter_tab_indent": _tab_indent_breaks_yaml(fm_yaml),
        "allowed_tools_shape": "none"
        if tools is None
        else {str: "string", list: "list", dict: "map"}.get(type(tools), "other"),
        "disable_model_invocation": _is_true(data.get("disable-model-invocation")),
        "codex_implicit_disabled": _codex_implicit_disabled(skill_path),
        "claude_only_fields": ", ".join(f for f in _CLAUDE_ONLY_FIELDS if f in data),
        "custom_fields": ", ".join(map(str, custom)),
        "custom_field_chars": sum(len(v) if isinstance(v, str) else len(yaml.safe_dump(v)) for v in custom.values()),
        "platforms": ", ".join(map(str, platforms)) if isinstance(platforms, list) else str(platforms or ""),
        "platforms_excluded": _excluded_os(platforms),
        "openclaw_gates": _openclaw_gates(data),
        "skill_md_chars": len(text),
        "skill_md_tokens": len(raw) // 4,
        "description_over": max(0, (len(description) if isinstance(description, str) else 0) - 1024),
        "utf16_over": max(0, (_utf16_units(description)) - 1024),
        # Hermes skill_view returns JSON (ensure_ascii=False, budget in code points): SKILL.md + description + ~400 chars.
        "hermes_view_chars": len(json.dumps(text, ensure_ascii=False))
        + len(json.dumps(description if isinstance(description, str) else "", ensure_ascii=False))
        + 400,
        **_dollar_n(body, text[: match.start(2)].count("\n") + 1 if match else 1),
        "uses_arguments_placeholder": "$ARGUMENTS" in body,
        "uses_claude_skill_dir": "${CLAUDE_SKILL_DIR}" in body,
        **_bundle_facts(skill_path, manifest),
        "catalog_size": catalog.size if catalog else 0,
        "catalog_desc_chars": catalog.desc_chars if catalog else 0,
        "catalog_base": str(catalog.base) if catalog and catalog.base else "",
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
    """Skill folders under ``folder``, symlinked ones included (Claude Code and Codex load those)."""
    found: set[Path] = set()
    for here, dirs, files in os.walk(folder, followlinks=True):
        path = Path(here).absolute()
        if path.is_symlink() and path.resolve() in {parent.resolve() for parent in path.parents}:
            dirs.clear()  # a link back to one of its own parent folders
            continue
        dirs[:] = [d for d in dirs if d not in SCAN_EXCLUDED_DIRS]
        if any(n in files for n in SKILL_MANIFEST_VARIANTS):
            found.add(Path(here))
    return sorted(s for s in found if not any(parent in found for parent in s.parents))


def _catalog_base(skill: Path) -> Path | None:
    """The skills folder a skill is installed in: a harness root, or a workspace ``skills/`` folder (OpenClaw)."""
    root, depth = _install_root(skill)
    if root is not None:
        return skill.absolute().parents[depth]
    parent = skill.absolute().parent
    return parent if parent.name == "skills" else None


def _catalog_of(group: list[Path]) -> Catalog:
    base = _catalog_base(group[0])
    if base is not None:
        return Catalog.of(_find_skills(base) or group, base)
    return Catalog.of(group, Path(os.path.commonpath([s.resolve() for s in group])) if len(group) > 1 else None)


def facts_for(skills: list[Path]) -> dict[Path, dict[str, Any]]:
    """Facts per skill, with the catalog they are installed in.

    A skill installed in a skills folder (e.g. ``.claude/skills``, or an OpenClaw workspace ``skills/``) is
    measured against everything installed in that folder, and skills in different folders are separate
    installations. Skills outside any skills folder that are given together form their own catalog.
    """
    groups: dict[Path | None, list[Path]] = {}
    for skill in skills:
        base = _catalog_base(skill)
        groups.setdefault(base.resolve() if base else None, []).append(skill)
    catalogs: dict[Path, Catalog] = {}
    for group in groups.values():
        catalogs.update(dict.fromkeys(group, _catalog_of(group)))
    return {skill: skill_facts(skill, catalogs[skill]) for skill in skills}


def _utf16_units(value: Any) -> int:
    return len(value.encode("utf-16-le", "surrogatepass")) // 2 if isinstance(value, str) else 0


def listed_description(profile: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    """What the profile says this harness puts in the model's skill list for the description."""
    listing = profile.get("listing") or {}
    text = facts["description_text"]
    budget = listing.get("budget") or {}
    over_budget = facts["catalog_size"] and (
        facts["catalog_desc_chars"] > (facts.get("list_budget_chars") or budget.get("desc_chars", float("inf")))
        or facts["catalog_size"] > budget.get("skills", float("inf"))
    )
    limit = listing.get("drop_over_utf16")
    if limit and facts["description_utf16_units"] > limit:
        return {"mode": "dropped", "text": ""}
    if not text or (listing.get("name_only_over") and len(text) > listing["name_only_over"]):
        return {"mode": "name_only", "text": ""}
    cut = budget.get("cut") if over_budget else listing.get("cut")
    if over_budget and not cut:
        return {"mode": "shortened", "text": ""}
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


def model_config(profile: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve the user's setup for one profile: ``context_tokens`` (0 = unknown), ``custom_provider`` and
    ``explicit_invocation`` (skills are activated by an explicit mention rather than opened by the model) and
    ``headless`` (a non-interactive run, e.g. ``claude -p``, with default permissions) and ``native_root``
    (evaluate as if installed in each harness's own skills root, so root rules are skipped). ``harness_versions``
    maps a harness to the release the user runs; a release other than the profile's verified one adds an advisory.

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
    versions = config.get("harness_versions") or {}
    return {
        "harness_version": str(versions.get(profile["harness"], "")),
        "context_tokens": tokens,
        "custom_provider": custom,
        "explicit_invocation": explicit,
        "headless": headless,
        "native_root": native_root,
    }


def listing_budget_chars(profile: dict[str, Any], context_tokens: int) -> int:
    """Characters the harness reserves for the skill list: a share of the context window when the profile
    gives ``chars_per_context_token`` and the window is known, else the fixed ``desc_chars`` (0 = no budget)."""
    budget = (profile.get("listing") or {}).get("budget") or {}
    per_token = budget.get("chars_per_context_token")
    if per_token and context_tokens:
        return int(context_tokens * per_token)
    return int(budget.get("desc_chars", 0))


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
    facts = {**facts, "list_budget_chars": listing_budget_chars(profile, setup["context_tokens"])}
    names = {**facts, **setup, "always": True, "roots": _RootList(all_roots, roots)}
    findings: list[dict[str, str]] = []
    worst = -1
    for rule in profile.get("rules", []):
        expression = str(rule["when"])
        if "install_root" in expression and (facts["install_root"] is None or names["native_root"]):
            continue
        if "catalog_" in expression and not facts["catalog_size"]:
            continue
        if expression != "always" and not _Rule.evaluate(expression, names):
            continue
        finding = {"id": rule["id"], "effect": rule["effect"], "message": _fill(rule["message"], facts)}
        for key in ("fix", "versions", "outcome", "confidence"):
            if rule.get(key):
                finding[key] = _fill(rule[key], facts)
        findings.append(finding)
        worst = max(worst, _EFFECT_RANK[rule["effect"]])
    status = "not_loaded" if worst == 2 else "degraded" if worst == 1 else "loads"
    verified = profile.get("verified", {})
    declared, checked = setup["harness_version"], str(verified.get("version", ""))
    if declared and declared != checked and not checked.startswith(f"{declared} "):
        findings.append(
            {
                "id": "VERSION-UNVERIFIED",
                "effect": "advisory",
                "message": f"Profile verified on {checked}; you run {declared}, so its rules may not hold.",
                "fix": "Check the findings on that release, or run the version the profile was verified on.",
            }
        )
    return HarnessResult(
        harness=profile["harness"],
        display_name=profile.get("display_name", profile["harness"]),
        status=status,
        version=str(verified.get("version", "")),
        verified_date=str(verified.get("date", "")),
        findings=findings,
        listed=listed_description(profile, facts),
        catalog={
            "size": facts["catalog_size"],
            "desc_chars": facts["catalog_desc_chars"],
            "folder": facts["catalog_base"],
        },
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
