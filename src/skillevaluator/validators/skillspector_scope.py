# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scope SkillSpector findings to the files an agent reads or runs.

SkillSpector rules describe attacks on an agent: prompt injection, tool
abuse, data exfiltration. A lexical match is an agent threat only when the
agent consumes the matched file. Tier 1 validation classifies each reported
location by its role in the artifact and keeps findings on agent control
paths (instructions, references, executables, runtime configuration, MCP and
plugin manifests) and on any file those control files reference, directly or
through imports. Matches in ordinary source, UI code, tests, documentation,
notebooks, images, and lockfiles are counted and suppressed.

SkillSpector can compact identical matches from several files into one issue
whose ``location`` names a single representative file and whose
``occurrences`` list every matched file. An issue is suppressed only when
every one of those files is outside agent control paths.

Classification fails closed when it cannot place a finding or finish the
walk. A location that is missing, outside the scanned artifact, or reached
through an escaping symlink is ``unknown`` and keeps its finding, and so does
an issue with malformed occurrences. A report that serializes fewer
occurrence records than the findings it counted, or a reference walk that
cannot finish inside its budgets, keeps every finding. Supply-chain findings,
YARA malware signatures, and artifact-integrity findings for a misleading
extension or embedded NUL bytes describe the shipped files themselves rather
than text an agent reads, so they apply on every path. A raster image whose
content is not an image keeps its findings too, because its extension cannot
establish its role.
"""

from __future__ import annotations

import ast
import json
import os
import re
from bisect import bisect_left
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path, PureWindowsPath

import yaml

from skillevaluator.constants import SCAN_EXCLUDED_DIRS, SCAN_EXCLUDED_FILES, SCANNABLE_EXTENSIONS
from skillevaluator.validators.base import iter_scannable_files

_AGENT_INSTRUCTION_FILES = frozenset(
    {
        "agents.md",
        ".aider.conf.yml",
        ".clinerules",
        ".cursorrules",
        ".windsurfrules",
        "claude.md",
        "gemini.md",
        "skill.md",
        "copilot-instructions.md",
        "workflow-rules.mdc",
    }
)
# ``output-styles/`` is a Claude Code plugin default: its Markdown is added to
# the system prompt without any manifest entry.
_AGENT_INSTRUCTION_DIRS = frozenset(
    {"agents", "commands", "output-styles", "prompts", "rules", "team-rules", "workflows"}
)
_AGENT_REFERENCE_DIRS = frozenset({"design-system", "references", "resources", "templates"})
_AGENT_EXECUTABLE_DIRS = frozenset({"hooks", "scripts", "tools"})
_AGENT_CONFIG_DIR = "config"
_HIDDEN_AGENT_CONFIG_DIRS = frozenset({".agents", ".claude", ".codex", ".continue", ".cursor", ".gemini", ".windsurf"})
_AGENT_ASSET_DIR = "assets"
_REPOSITORY_METADATA_FILES = frozenset(
    {
        ".dockerignore",
        ".ds_store",
        ".editorconfig",
        ".eslintignore",
        ".gitattributes",
        ".gitignore",
        ".gitmodules",
        ".npmignore",
        ".prettierignore",
    }
)
_GENERATED_FILE_NAMES = frozenset(name.lower() for name in SCAN_EXCLUDED_FILES)
_LOCKFILE_NAMES = frozenset(
    {"cargo.lock", "package-lock.json", "pnpm-lock.yaml", "poetry.lock", "uv.lock", "yarn.lock"}
)
_IMAGE_EXTENSIONS = frozenset({".avif", ".gif", ".ico", ".jpeg", ".jpg", ".png", ".svg", ".webp"})
# A file with one of these extensions and a matching image or font signature
# has no path text to follow. SVG is XML text and is read like any other file.
_RASTER_IMAGE_EXTENSIONS = _IMAGE_EXTENSIONS - {".svg"}
_FONT_EXTENSIONS = frozenset({".otf", ".ttc", ".ttf", ".woff", ".woff2"})
_SOURCE_EXTENSIONS = frozenset({".c", ".cc", ".cpp", ".cu", ".go", ".java", ".js", ".jsx", ".py", ".rs", ".ts", ".tsx"})
_UI_SOURCE_EXTENSIONS = frozenset({".css", ".html", ".js", ".jsx", ".scss", ".svelte", ".ts", ".tsx", ".vue"})
_UI_SOURCE_DIRS = frozenset({"client", "frontend", "public", "ui", "web"})
# Plugin default component files that launch commands or connectors without
# any manifest entry: Claude Code's ``.lsp.json`` (language servers) and
# ``monitors/monitors.json`` (monitor commands), and Codex's ``.app.json``.
_PLUGIN_DEFAULT_EXECUTABLE_FILES = frozenset({".lsp.json"})
_PLUGIN_DEFAULT_MONITORS_FILE = ("monitors", "monitors.json")
_PLUGIN_DEFAULT_CONFIG_FILES = frozenset({".app.json"})
_PLUGIN_MANIFEST_DIRS = frozenset({".claude-plugin", ".codex-plugin", ".cursor-plugin"})
_MCP_CONTROL_FILES = frozenset(
    {".mcp.json", "agent_plugin.yaml", "agent_plugin.yml", "mcp.json", "mcp.yaml", "mcp.yml"}
)
AGENT_CONTROL_PATH_ROLES = frozenset(
    {"agent_config", "agent_instruction", "agent_reference", "agent_executable", "mcp_control", "plugin_manifest"}
)
# Supply-chain rules (SC*) and the YARA malware, webshell, and cryptominer
# signatures (YR1-YR3) flag what the artifact ships: vulnerable or
# typosquatted dependencies, bytecode, concealed executables, malicious
# payloads. The artifact-integrity rules AE2 (content does not match the
# extension) and AE3 (a text file with embedded NUL bytes) flag a file
# disguised from classification, including the file-role classification here.
# They stay blocking on every path. YR4 also carries lexical prompt-injection
# rules, so it is scoped like other agent-threat rules.
_ARTIFACT_WIDE_RULE_ID = re.compile(r"SC\d+|YR[123]|AE[23]")
_ARTIFACT_WIDE_CATEGORIES = frozenset({"supply chain"})
# A raster image extension on a file whose content is not an image.
UNVERIFIED_IMAGE_ROLE = "unverified_image"

_REFERENCE_TEXT_EXTENSIONS_LOWER = (
    SCANNABLE_EXTENSIONS
    | _SOURCE_EXTENSIONS
    | {
        "",
        ".bash",
        ".cfg",
        ".cjs",
        ".cts",
        ".fish",
        ".ini",
        ".lua",
        ".mdc",
        ".mjs",
        ".mts",
        ".php",
        ".pl",
        ".ps1",
        ".r",
        ".rb",
        ".toml",
        ".xml",
        ".zsh",
    }
)
_REFERENCE_TEXT_EXTENSIONS = _REFERENCE_TEXT_EXTENSIONS_LOWER | {
    extension.upper() for extension in _REFERENCE_TEXT_EXTENSIONS_LOWER if extension
}
_REFERENCE_FILE_LIMIT = 10_000
_REFERENCE_FILE_SIZE_LIMIT = 1_000_000
# Source and candidate pairs the walk may consider. Most pairs are rejected
# by a word check; the comparison limit counts the pairs that pass it and
# need their reference variants built.
_REFERENCE_PAIR_LIMIT = 1_000_000
_REFERENCE_MATCH_COMPARISON_LIMIT = 20_000
_REFERENCE_MATCH_CHARACTER_LIMIT = 5_000_000
_REFERENCE_SOURCE_BYTE_LIMIT = 5_000_000
_REFERENCE_SOURCE_FILE_LIMIT = 1_000
_REFERENCE_MANIFEST_NODE_LIMIT = 10_000
# A reference variant matches only between a leading boundary and a trailing
# boundary, so every match lies inside one maximal run of these characters.
# A variant without ``/`` (a bare name) must start a run. A variant with ``/``
# may also start right after a ``/`` inside a run, so a skill-relative path
# still matches behind an install path or a placeholder the walk does not
# expand, such as ``~/.claude/skills/demo/docs/guide.md``.
_REFERENCE_TOKEN_PATTERN = re.compile(r"[\w./-]+")
_REFERENCE_LEADING_BOUNDARY = r"(?<![\w./-])"
_REFERENCE_PATH_LEADING_BOUNDARY = r"(?<![\w.-])"
_REFERENCE_WORD_PATTERN = re.compile(r"\w+")
# A reference may end a sentence ("Read docs/guide.md."), so trailing dots
# followed by a non-path character still end the reference.
_REFERENCE_TRAILING_BOUNDARY = r"(?=\.*(?![\w./-]))"
_REFERENCE_DIRECTORY_TRAILING_BOUNDARY = r"(?=\.*(?![\w.-]))"
_PLUGIN_ROOT_MARKER = "__skillspector_claude_plugin_root__"
_SKILL_ROOT_MARKER = "__skillspector_skill_root__"


@dataclass(frozen=True)
class IssueScope:
    """Where a SkillSpector issue sits and whether Tier 1 keeps it.

    ``applicability`` is ``agent-control-path`` for control and referenced
    files, ``artifact-wide`` for supply-chain and malware rules elsewhere,
    ``fail-closed`` when the location, the file's role, or the reference graph
    could not be established, and ``outside-agent-control-path`` for
    suppressed issues.
    """

    file_role: str
    applicability: str

    @property
    def applicable(self) -> bool:
        return self.applicability != "outside-agent-control-path"


def _sorted_has_prefix(sorted_values: list[str], prefix: str, *, exact: bool = False) -> bool:
    index = bisect_left(sorted_values, prefix)
    if index >= len(sorted_values):
        return False
    return sorted_values[index] == prefix if exact else sorted_values[index].startswith(prefix)


def _reference_index(text: str) -> tuple[list[str], list[str]]:
    """Index *text* for :func:`_reference_variant_may_match`.

    Returns the sorted path tokens and the sorted pairs of adjacent
    ``/``-separated segments within each token (``a/b`` and ``b/c`` for
    ``a/b/c``). Each pair is at most two segments long, so the index stays
    linear in the size of the text.
    """
    tokens = set(_REFERENCE_TOKEN_PATTERN.findall(text))
    segment_pairs: set[str] = set()
    for token in tokens:
        segments = token.split("/")
        segment_pairs.update(f"{left}/{right}" for left, right in pairwise(segments))
    return sorted(tokens), sorted(segment_pairs)


def _reference_variant_may_match(variant: str, reference_index: tuple[list[str], list[str]]) -> bool:
    """Return False only when *variant* cannot match the indexed reference text.

    A bare variant made of token characters can match only at the start of a
    token it prefixes. A variant with ``/`` starts at a segment boundary, so
    its first two segments are an indexed segment pair, or, when the variant
    has only two segments, begin one. Other variants are always searched.
    """
    if _REFERENCE_TOKEN_PATTERN.fullmatch(variant) is None:
        return True
    sorted_tokens, sorted_segment_pairs = reference_index
    if "/" not in variant:
        return _sorted_has_prefix(sorted_tokens, variant)
    head, _separator, rest = variant.partition("/")
    second, more_segments, _tail = rest.partition("/")
    return _sorted_has_prefix(sorted_segment_pairs, f"{head}/{second}", exact=bool(more_segments))


def normalize_report_path(file_path: object, *, scan_root: Path) -> str:
    """Return a scanner location relative to *scan_root*, or ``unknown``.

    The location must name an existing regular file inside the scanned
    artifact once symlinks are resolved. Control characters, Windows drive or
    absolute paths, parent escapes, and the root itself are ``unknown``. The
    returned path keeps the lexical (unresolved) form so a contained symlink
    alias is classified by its own name.
    """
    if not isinstance(file_path, str) or not file_path.strip():
        return "unknown"
    if any(ord(character) < 32 or ord(character) == 127 for character in file_path):
        return "unknown"
    slash_path = file_path.replace("\\", "/")
    windows_path = PureWindowsPath(slash_path)
    if windows_path.drive or windows_path.is_absolute():
        return "unknown"
    normalized = os.path.normpath(slash_path.removeprefix("./"))
    path = Path(normalized)
    if normalized in {"", "."} or (path.parts and path.parts[0] == ".."):
        return "unknown"
    try:
        resolved_root = scan_root.resolve(strict=False)
        root_directory = resolved_root.parent if scan_root.is_file() else resolved_root
        lexical_path = path.relative_to(root_directory) if path.is_absolute() else path
        resolved_path = (root_directory / lexical_path).resolve(strict=False)
        if scan_root.is_file():
            return scan_root.name if resolved_path == resolved_root else "unknown"
        resolved_path.relative_to(resolved_root)
        if not resolved_path.is_file():
            return "unknown"
    except (OSError, RuntimeError, ValueError):
        return "unknown"
    result = lexical_path.as_posix()
    return "unknown" if result in {"", "."} else result


def executable_report_paths(components: object, *, scan_root: Path) -> frozenset[str]:
    """Return the scan-relative paths of components SkillSpector marks executable.

    A path that does not resolve inside the artifact is returned as
    ``unknown`` so the caller can fail closed.
    """
    if not isinstance(components, list):
        return frozenset()
    paths: set[str] = set()
    for component in components:
        if not isinstance(component, dict) or component.get("executable") is not True:
            continue
        paths.add(normalize_report_path(component.get("path"), scan_root=scan_root))
    return frozenset(paths)


def skillspector_file_role(file_path: str, *, executable: bool = False) -> str:
    """Classify a scan-relative location by its role in an agent artifact."""
    if not isinstance(file_path, str) or not file_path.strip() or file_path == "unknown":
        return "unknown"
    path = Path(file_path.replace("\\", "/"))
    name = path.name.lower()
    suffix = path.suffix.lower()
    ordered_parts = tuple(part.lower() for part in path.parts)
    parts = set(ordered_parts)

    if executable:
        return "agent_executable"
    if ordered_parts[:2] == (".github", "workflows"):
        return "repository_metadata"
    if ordered_parts[:2] == (".github", "instructions") and name.endswith(".instructions.md"):
        return "agent_instruction"
    if name in _REPOSITORY_METADATA_FILES:
        return "repository_metadata"
    if ordered_parts and ordered_parts[0] in _HIDDEN_AGENT_CONFIG_DIRS:
        return "agent_config"
    # Agent control paths take precedence over name-based test conventions;
    # ``scripts/test_inject.sh`` is still agent-executed and
    # ``prompts/test_override.md`` is still an agent instruction.
    if parts & _AGENT_EXECUTABLE_DIRS:
        return "agent_executable"
    if parts & _AGENT_INSTRUCTION_DIRS:
        return "agent_instruction"
    if name in {"agent_plugin.yaml", "agent_plugin.yml"} or (name == "plugin.json" and parts & _PLUGIN_MANIFEST_DIRS):
        return "plugin_manifest"
    if name in _PLUGIN_DEFAULT_EXECUTABLE_FILES or ordered_parts[-2:] == _PLUGIN_DEFAULT_MONITORS_FILE:
        return "agent_executable"
    if name in _PLUGIN_DEFAULT_CONFIG_FILES:
        return "agent_config"
    if name in _MCP_CONTROL_FILES or (ordered_parts and ordered_parts[0] == ".mcp"):
        return "mcp_control"
    if name in _AGENT_INSTRUCTION_FILES or suffix == ".mdc":
        return "agent_instruction"
    if len(ordered_parts) == 1 and name.startswith("."):
        return "repository_metadata"
    if parts & _AGENT_REFERENCE_DIRS:
        return "agent_reference"
    if "tests" in parts or "test" in parts or name.startswith(("test_", "test-")):
        return "test_fixture"
    # Non-image assets can be templates or runtime policies that SKILL.md
    # tells the agent to read. Images stay suppressible unless they live in an
    # instruction or executable directory.
    if ordered_parts and ordered_parts[0] == _AGENT_ASSET_DIR:
        return "image_asset" if suffix in _IMAGE_EXTENSIONS else "agent_reference"
    # Top-level ``config/`` holds runtime values the skill loads. Nested
    # ``tests/config/`` fixtures stay out of scope because only the
    # scan-relative top-level directory counts.
    if ordered_parts and ordered_parts[0] == _AGENT_CONFIG_DIR:
        return "agent_config"
    if len(ordered_parts) == 1 and name in _GENERATED_FILE_NAMES:
        return "generated_artifact"
    if name in _LOCKFILE_NAMES:
        return "generated_lockfile"
    if suffix in _IMAGE_EXTENSIONS:
        return "image_asset"
    if suffix == ".ipynb":
        return "notebook"
    if suffix in _UI_SOURCE_EXTENSIONS and parts & _UI_SOURCE_DIRS:
        return "ui_source"
    if "docs" in parts or name.startswith("readme") or suffix == ".rst":
        return "documentation"
    # A data or runnable file at the artifact root is ambiguous unless a
    # stronger generated or documentation role above proves it irrelevant.
    if len(ordered_parts) == 1:
        return "agent_reference"
    if suffix in _SOURCE_EXTENSIONS:
        return "source_code"
    return "ordinary_data"


def is_artifact_wide_issue(issue: Mapping[str, object]) -> bool:
    """Return whether a SkillSpector rule applies to every shipped file."""
    rule_id = issue.get("id")
    category = issue.get("category")
    return (isinstance(rule_id, str) and _ARTIFACT_WIDE_RULE_ID.fullmatch(rule_id.strip()) is not None) or (
        isinstance(category, str) and category.strip().lower() in _ARTIFACT_WIDE_CATEGORIES
    )


def issue_report_files(issue: Mapping[str, object]) -> list[object] | None:
    """Return the location file and every occurrence file of a SkillSpector issue.

    The location file comes first. A report without ``occurrences`` names only
    its location. Returns None when ``occurrences`` is malformed (not a list,
    or an entry that is not an object with a string ``file``), so the caller
    keeps the issue.
    """
    files: list[object] = [_issue_file(issue)]
    occurrences = issue.get("occurrences")
    if occurrences is None:
        return files
    if not isinstance(occurrences, list):
        return None
    for occurrence in occurrences:
        if not isinstance(occurrence, dict) or not isinstance(occurrence.get("file"), str):
            return None
        files.append(occurrence["file"])
    return files


def skillspector_report_is_truncated(report: Mapping[str, object]) -> bool:
    """Return whether a report serializes fewer occurrence records than findings it counted.

    SkillSpector bounds its output by occurrence record, so findings past the
    bound have no serialized location. ``findings_after_filtering`` is the
    count before that bound. A report without the count is not truncated.
    """
    completeness = report.get("analysis_completeness")
    expected = completeness.get("findings_after_filtering") if isinstance(completeness, dict) else None
    if expected is None:
        return False
    if isinstance(expected, bool) or not isinstance(expected, int):
        return True
    issues = report.get("issues")
    suppressed = report.get("suppressed")
    records = sum(_occurrence_record_count(issue) for issue in issues) if isinstance(issues, list) else 0
    records += len(suppressed) if isinstance(suppressed, list) else 0
    return records < expected


def _occurrence_record_count(issue: object) -> int:
    occurrences = issue.get("occurrences") if isinstance(issue, dict) else None
    return len(occurrences) if isinstance(occurrences, list) and occurrences else 1


def scope_skillspector_issues(
    issues: list[dict],
    components: object,
    *,
    scan_root: Path,
    reference_root: Path | None = None,
    report_truncated: bool = False,
) -> list[IssueScope]:
    """Classify each issue in *issues*, in order, for Tier 1 applicability.

    An issue is classified by its location file and every occurrence file.
    It is suppressed only when all of them are outside agent control paths;
    otherwise it takes the role of the first file that keeps it.

    *reference_root* is the artifact root when *scan_root* is one bundled
    skill inside it (a plugin), so references written relative to the plugin
    root still reach files in the skill. *report_truncated* marks a report
    whose serialized occurrences do not cover every counted finding (see
    :func:`skillspector_report_is_truncated`); every issue then fails closed.
    """
    if not scan_root.exists():
        return [IssueScope("unknown", "fail-closed") for _issue in issues]
    located = [_located_issue(issue, scan_root=scan_root) for issue in issues]
    if scan_root.is_file():
        # A single-file target (such as one rules file) is the artifact the
        # agent loads, so every finding in it applies whatever its name.
        scopes = []
        for location, report_paths in located:
            role = skillspector_file_role(location)
            if report_truncated or report_paths is None or "unknown" in report_paths:
                scopes.append(IssueScope(role, "fail-closed"))
            else:
                scopes.append(
                    IssueScope(role if role in AGENT_CONTROL_PATH_ROLES else "agent_instruction", "agent-control-path")
                )
        return scopes
    executable_paths = executable_report_paths(components, scan_root=scan_root)
    if report_truncated or "unknown" in executable_paths:
        # Findings past the report bound have no location, and an executable
        # outside the artifact (or missing) cannot seed the reference walk,
        # so no ordinary file can be proven unreachable.
        return [
            IssueScope(skillspector_file_role(location, executable=location in executable_paths), "fail-closed")
            for location, _report_paths in located
        ]

    roles: dict[str, str] = {}

    def _role(file_path: str) -> str:
        if file_path not in roles:
            role = skillspector_file_role(file_path, executable=file_path in executable_paths)
            if role == "image_asset" and not _is_verified_image(scan_root / file_path):
                # The extension that gives the image role is the one an
                # analysis-evasion finding says is misleading.
                role = UNVERIFIED_IMAGE_ROLE
            roles[file_path] = role
        return roles[file_path]

    candidates = {
        file_path
        for (_location, report_paths), issue in zip(located, issues, strict=True)
        if report_paths is not None and not is_artifact_wide_issue(issue)
        for file_path in report_paths
        if _role(file_path) != "unknown" and _role(file_path) not in AGENT_CONTROL_PATH_ROLES
    }
    referenced = referenced_report_paths(
        candidates,
        scan_root=scan_root,
        reference_root=reference_root,
        executable_paths=executable_paths,
    )

    def _path_scope(file_path: str, *, artifact_wide: bool) -> IssueScope:
        role = _role(file_path)
        if role == "unknown" or (referenced is None and file_path in candidates):
            return IssueScope(role, "fail-closed")
        if role in AGENT_CONTROL_PATH_ROLES:
            return IssueScope(role, "agent-control-path")
        if referenced is not None and file_path in referenced:
            return IssueScope("agent_reference", "agent-control-path")
        if role == UNVERIFIED_IMAGE_ROLE:
            return IssueScope(role, "fail-closed")
        if artifact_wide:
            return IssueScope(role, "artifact-wide")
        return IssueScope(role, "outside-agent-control-path")

    scopes: list[IssueScope] = []
    for (location, report_paths), issue in zip(located, issues, strict=True):
        if report_paths is None:
            scopes.append(IssueScope(_role(location), "fail-closed"))
            continue
        path_scopes = [
            _path_scope(file_path, artifact_wide=is_artifact_wide_issue(issue)) for file_path in report_paths
        ]
        scopes.append(
            next(
                (
                    scope
                    for applicability in ("agent-control-path", "fail-closed", "artifact-wide")
                    for scope in path_scopes
                    if scope.applicability == applicability
                ),
                path_scopes[0],
            )
        )
    return scopes


def _located_issue(issue: Mapping[str, object], *, scan_root: Path) -> tuple[str, tuple[str, ...] | None]:
    """Return an issue's normalized location and all its normalized report paths.

    The report paths are None when the issue's occurrences are malformed.
    """
    location = normalize_report_path(_issue_file(issue), scan_root=scan_root)
    report_files = issue_report_files(issue)
    if report_files is None:
        return location, None
    report_paths = (location, *(normalize_report_path(file, scan_root=scan_root) for file in report_files[1:]))
    return location, tuple(dict.fromkeys(report_paths))


def _issue_file(issue: Mapping[str, object]) -> object:
    location = issue.get("location")
    return location.get("file") if isinstance(location, dict) else None


def _path_variants(
    file_path: str,
    *,
    include_parents: bool = True,
    include_source_root: bool = True,
    include_import_aliases: bool = True,
) -> set[str]:
    """Return textual forms a control file may use to reference *file_path*."""
    path = Path(file_path)
    variants = {file_path, f"./{file_path}"}
    if path.suffix:
        without_suffix = path.with_suffix("").as_posix()
        variants.add(without_suffix)
        variants.add(f"./{without_suffix}")
        if include_import_aliases and without_suffix.startswith("./"):
            variants.add(f".{without_suffix[2:].replace('/', '.')}")
        elif include_import_aliases and without_suffix.startswith("../"):
            parent_count = 0
            module_path = without_suffix
            while module_path.startswith("../"):
                parent_count += 1
                module_path = module_path[3:]
            variants.add(f"{'.' * (parent_count + 1)}{module_path.replace('/', '.')}")
        elif include_import_aliases:
            dotted_path = without_suffix.replace("/", ".")
            variants.add(dotted_path)
            if len(path.parts) == 1:
                variants.add(f".{dotted_path}")
    if include_parents:
        for parent in path.parents:
            if parent == Path():
                continue
            parent_path = parent.as_posix()
            variants.update({parent_path, f"./{parent_path}", f"{parent_path}/", f"./{parent_path}/"})
            if include_import_aliases:
                variants.add(parent_path.replace("/", "."))
    if include_import_aliases and len(path.parts) > 1 and path.name.lower() in {"__init__.py", "__main__.py"}:
        package = path.parent.as_posix()
        dotted_package = package.replace("/", ".")
        variants.update({package, f"./{package}", dotted_package, f"{dotted_package}."})
    if include_source_root and len(path.parts) > 1 and path.parts[0].lower() == "src":
        variants.update(
            _path_variants(
                Path(*path.parts[1:]).as_posix(),
                include_parents=include_parents,
                include_source_root=False,
                include_import_aliases=include_import_aliases,
            )
        )
    return variants


def _bare_name_variants(file_path: str) -> set[str]:
    """Return the bare file name of a nested file, such as ``helper.md`` for ``lib/helper.md``.

    Guidance often names a helper by file name alone. A name without a suffix
    reads as an ordinary word, so it is left out.
    """
    path = Path(file_path)
    if len(path.parts) > 1 and path.suffix and _REFERENCE_WORD_PATTERN.search(path.stem):
        return {path.name}
    return set()


def _reference_anchor_words(file_path: str) -> frozenset[str] | None:
    """Return words, one of which appears in any text that references *file_path*.

    Every reference variant contains the file's stem or a whole directory name
    from its path, and a variant matches only on path-token or ``/`` segment
    boundaries, so each word of a matched variant is a whole word of the
    source text. Words are lowercase because references match without regard
    to case. None means a path component has no word characters, and the file
    is always compared.
    """
    path = Path(file_path)
    words: set[str] = set()
    for component in (*path.parent.parts, path.stem):
        component_words = _REFERENCE_WORD_PATTERN.findall(component.lower())
        if not component_words:
            return None
        words.update(component_words)
    return frozenset(words)


def _directory_prefix_path_variants(file_path: str) -> set[str]:
    return {
        f"{variant.rstrip('/')}/"
        for variant in _path_variants(
            file_path,
            include_parents=False,
            include_source_root=False,
            include_import_aliases=False,
        )
    }


def _manifest_reference_text(raw_text: str) -> str | None:
    """Keep manifest runtime references while excluding descriptive metadata."""
    # A manifest that parses but cannot be loaded (an integer past Python's
    # digit limit, an invalid YAML date) fails the walk closed.
    try:
        manifest = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        try:
            manifest = yaml.safe_load(raw_text)
        except (RecursionError, TypeError, ValueError):
            return None
        except yaml.YAMLError:
            return raw_text
    except (RecursionError, ValueError):
        return None
    if not isinstance(manifest, (dict, list)):
        return raw_text

    references: list[str] = []
    active_containers: set[int] = set()
    visited_containers: set[tuple[int, bool]] = set()
    visited_nodes = 0
    metadata_keys = {"author", "description", "homepage", "license", "name", "repository", "skills", "version"}

    def _collect(value: object, key: str | None = None, depth: int = 0) -> None:
        nonlocal visited_nodes
        visited_nodes += 1
        if visited_nodes > _REFERENCE_MANIFEST_NODE_LIMIT:
            raise ValueError("manifest traversal budget exceeded")
        if key in metadata_keys:
            return
        if isinstance(value, (dict, list)):
            container_id = id(value)
            visit_key = (container_id, key == "args")
            if depth > 100 or container_id in active_containers:
                raise ValueError("cyclic or excessively deep manifest")
            if visit_key in visited_containers:
                return
            active_containers.add(container_id)
            try:
                if isinstance(value, dict):
                    for child_key, child_value in value.items():
                        _collect(child_value, str(child_key).lower(), depth + 1)
                else:
                    string_items = [item for item in value if isinstance(item, str)]
                    if key == "args" and string_items:
                        references.append(" ".join(string_items))
                    for item in value:
                        _collect(item, key, depth + 1)
            finally:
                active_containers.remove(container_id)
            visited_containers.add(visit_key)
            return
        if not isinstance(value, str):
            return
        path_like = value.startswith(".") or "/" in value or "\\" in value or bool(Path(value).suffix)
        if key == "args" or path_like:
            references.append(value)

    try:
        _collect(manifest)
    except (RecursionError, ValueError):
        return None
    return "\n".join(references)


def _has_raster_signature(path: Path) -> bool:
    """Return whether *path* starts with a PNG, JPEG, GIF, WebP, ICO, or AVIF signature."""
    with path.open("rb") as handle:
        header = handle.read(16)
    return (
        header.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"\x00\x00\x01\x00"))
        or (header[:4] == b"RIFF" and header[8:12] == b"WEBP")
        or header[4:12] in {b"ftypavif", b"ftypavis"}
    )


def _has_font_signature(path: Path) -> bool:
    """Return whether *path* starts with a TrueType, OpenType, collection, or WOFF signature."""
    with path.open("rb") as handle:
        header = handle.read(4)
    return header in {b"\x00\x01\x00\x00", b"OTTO", b"true", b"ttcf", b"wOFF", b"wOF2"}


def _is_verified_image(path: Path) -> bool:
    """Return whether an ``image_asset`` file may keep that role.

    SVG is text by format, so its extension is enough. A raster image needs a
    raster signature; an unreadable file is not verified.
    """
    if path.suffix.lower() not in _RASTER_IMAGE_EXTENSIONS:
        return True
    try:
        return _has_raster_signature(path)
    except OSError:
        return False


def _is_binary_leaf(path: Path, suffix: str) -> bool:
    """Return whether *path* is a verified image or font, which has no path text to follow."""
    if suffix in _RASTER_IMAGE_EXTENSIONS:
        return _has_raster_signature(path)
    if suffix in _FONT_EXTENSIONS:
        return _has_font_signature(path)
    return False


def _root_variable_pattern(variable_name: str, *, placeholder: str | None = None) -> re.Pattern:
    """Match ``$NAME/``, ``${NAME}/``, ``$env:NAME/``, ``%NAME%/``, and an optional placeholder form."""
    placeholder_form = f"|{placeholder}" if placeholder else ""
    return re.compile(
        rf"(?P<quote>[\"']?)(?:\$(?:\{{)?(?:env:)?{variable_name}(?:\}})?|%{variable_name}%{placeholder_form})"
        rf"(?P=quote)/",
        re.IGNORECASE,
    )


_PLUGIN_ROOT_PATTERN = _root_variable_pattern("CLAUDE_PLUGIN_ROOT")
# Claude Code substitutes ``${CLAUDE_SKILL_DIR}`` with the skill directory.
# ``SKILL_DIR`` and ``SKILL_ROOT`` are the generic spellings, and guidance
# also writes the directory as a placeholder such as ``{baseDir}``,
# ``{{skill_dir}}``, or ``<skill-dir>``. The placeholder name is bounded so a
# long run of word characters cannot make the search quadratic.
_PLACEHOLDER_NAME = r"[\w-]{0,32}(?:base|skill)[\w-]{0,32}"
_SKILL_ROOT_PATTERN = _root_variable_pattern(
    "(?:CLAUDE_SKILL_DIR|SKILL_DIR|SKILL_ROOT)",
    placeholder=rf"\{{\{{?{_PLACEHOLDER_NAME}\}}?\}}|<{_PLACEHOLDER_NAME}>",
)


# Whitespace before ``from`` may not include a newline. With ``\s*`` an
# attempt at every line start of a blank-line run scanned to the end of the
# run, which made long whitespace runs quadratic.
_FROM_IMPORT_PATTERN = re.compile(r"(?m)^[^\S\n]*from\s+(\.*[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*|\.+)\s+import\s+([^\n;]+)")


def _expanded_python_imports(raw_text: str, text: str, *, is_python: bool) -> list[str] | None:
    """Return ``module.name`` forms for ``from module import name`` statements.

    Returns None when Python source is too deeply nested to parse.
    """
    expanded_imports: list[str] = []
    parsed_python: ast.AST | None = None
    if is_python:
        try:
            parsed_python = ast.parse(raw_text)
        except RecursionError:
            return None
        except (SyntaxError, ValueError):
            pass
    if parsed_python is not None:
        for import_node in ast.walk(parsed_python):
            if not isinstance(import_node, ast.ImportFrom):
                continue
            module_name = f"{'.' * import_node.level}{import_node.module or ''}"
            separator = "" if module_name.endswith(".") else "."
            expanded_imports.extend(
                f"{module_name}{separator}{imported_name.name}"
                for imported_name in import_node.names
                if imported_name.name != "*"
            )
        return expanded_imports
    for import_match in _FROM_IMPORT_PATTERN.finditer(text):
        module_name = import_match.group(1)
        for imported_name in import_match.group(2).strip("() ").split(","):
            imported_name = imported_name.strip().split()[0] if imported_name.strip() else ""
            if re.fullmatch(r"[A-Za-z_]\w*", imported_name):
                separator = "" if module_name.endswith(".") else "."
                expanded_imports.append(f"{module_name}{separator}{imported_name}")
    return expanded_imports


def referenced_report_paths(
    candidates: Iterable[str],
    *,
    scan_root: Path,
    reference_root: Path | None = None,
    executable_paths: frozenset[str] = frozenset(),
) -> frozenset[str] | None:
    """Return the *candidates* reachable from agent control files.

    Control files (by role, plus executable components) seed a walk that
    follows textual path references, module imports, and
    ``$CLAUDE_PLUGIN_ROOT``/``${CLAUDE_SKILL_DIR}``/``$SKILL_ROOT`` forms.
    References match without regard to case, and a path with ``/`` also
    matches behind another path prefix, such as an install location or the
    artifact's own directory name. Reachability is only a
    suppression filter: on a symlinked directory, an unreadable or oversized
    file, an escaping path, or an exhausted budget, the walk returns None so
    the caller keeps every finding rather than trust partial evidence.
    """
    issue_candidates = set(candidates)
    if not issue_candidates:
        return frozenset()
    reference_root = reference_root or scan_root
    try:
        reference_prefix = scan_root.resolve(strict=True).relative_to(reference_root.resolve(strict=True))
    except (OSError, RuntimeError, ValueError):
        reference_root = scan_root
        reference_prefix = Path()
    try:
        resolved_root = reference_root.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    resolved_base = resolved_root if resolved_root.is_dir() else resolved_root.parent
    lexical_base = reference_root if reference_root.is_dir() else reference_root.parent
    # An absolute install path such as ``~/.claude/skills/<name>/README.md``
    # names a top-level file behind the artifact's directory name. A name
    # with non-path characters would defeat the prefilter for every
    # candidate, so it is left out.
    root_name_markers = tuple(
        dict.fromkeys(
            name
            for name in (lexical_base.name, resolved_base.name)
            if name not in {".", ".."} and _REFERENCE_TOKEN_PATTERN.fullmatch(name)
        )
    )

    def _prefixed(path: str) -> str:
        return (reference_prefix / path).as_posix() if reference_prefix != Path() else path

    unlisted_control_files: list[Path] = []
    if reference_root.is_symlink():
        return None
    if reference_root.is_dir():
        traversed_directories = 0
        control_directories = (
            _AGENT_EXECUTABLE_DIRS
            | _AGENT_INSTRUCTION_DIRS
            | _AGENT_REFERENCE_DIRS
            | _HIDDEN_AGENT_CONFIG_DIRS
            | _PLUGIN_MANIFEST_DIRS
            | {_AGENT_ASSET_DIR, _AGENT_CONFIG_DIR, ".mcp", _PLUGIN_DEFAULT_MONITORS_FILE[0]}
        )
        try:
            for current_root, directory_names, file_names in os.walk(reference_root, followlinks=False):
                directory_names[:] = [name for name in directory_names if name not in SCAN_EXCLUDED_DIRS]
                traversed_directories += len(directory_names)
                if traversed_directories > _REFERENCE_FILE_LIMIT:
                    return None
                current_path = Path(current_root)
                if any((current_path / name).is_symlink() for name in directory_names):
                    return None
                is_skill_root = any(file_name.lower() == "skill.md" for file_name in file_names)
                for file_name in file_names:
                    local_file = current_path / file_name
                    relative_parts = local_file.relative_to(reference_root).parts
                    if len(relative_parts) == 1 or is_skill_root or set(relative_parts[:-1]) & control_directories:
                        unlisted_control_files.append(local_file)
        except (OSError, RuntimeError):
            return None

    referenceable_files = iter_scannable_files(reference_root, _REFERENCE_TEXT_EXTENSIONS, excluded_files=())
    referenceable_files.extend(unlisted_control_files)
    # A plugin's bundled skills can reference one another through the plugin
    # root even when a walker scope leaves ``skills/`` out of the root view.
    bundled_skills_root = reference_root / "skills"
    if bundled_skills_root.is_dir():
        referenceable_files.extend(
            iter_scannable_files(bundled_skills_root, _REFERENCE_TEXT_EXTENSIONS, excluded_files=())
        )
    for executable_path in executable_paths:
        if executable_path != "unknown":
            referenceable_files.append(lexical_base / _prefixed(executable_path))
    referenceable_files = list(dict.fromkeys(referenceable_files))
    if len(referenceable_files) > _REFERENCE_FILE_LIMIT:
        return None

    local_files: dict[str, Path] = {}
    for local_file in referenceable_files:
        try:
            local_file.resolve(strict=True).relative_to(resolved_base)
            lexical_relative = Path(os.path.relpath(local_file, start=lexical_base))
        except (OSError, RuntimeError, ValueError):
            return None
        normalized = normalize_report_path(str(lexical_relative), scan_root=reference_root)
        if normalized == "unknown":
            return None
        local_files[normalized] = local_file

    issue_reference_paths = {candidate: _prefixed(candidate) for candidate in issue_candidates}
    for reference_path in issue_reference_paths.values():
        local_files.setdefault(reference_path, lexical_base / reference_path)

    skill_roots = {Path(path).parent for path in local_files if Path(path).name.lower() == "skill.md"}

    def _enclosing_skill_root(file_path: str) -> Path | None:
        path = Path(file_path)
        for candidate in (path.parent, *path.parents):
            if candidate in skill_roots:
                return candidate
        return None

    def _reference_file_role(file_path: str) -> str:
        skill_root = _enclosing_skill_root(file_path)
        if skill_root is None:
            return skillspector_file_role(file_path)
        return skillspector_file_role(Path(file_path).relative_to(skill_root).as_posix())

    roles = {path: _reference_file_role(path) for path in local_files}
    enclosing_skill_roots = {path: _enclosing_skill_root(path) for path in local_files}
    is_directory = {path: target.is_dir() for path, target in local_files.items()}
    include_parents_by_path = {path: role != "repository_metadata" for path, role in roles.items()}
    variants_by_path = {
        path: _path_variants(path, include_parents=include_parents_by_path[path]) | _bare_name_variants(path)
        for path in local_files
    }
    directory_prefix_variants: dict[str, set[str]] = {}
    for path in local_files:
        if is_directory[path]:
            directory_prefix_variants[path] = _directory_prefix_path_variants(path)
            variants_by_path[path].update(directory_prefix_variants[path])
    # Package files are also reached through relative package forms such as
    # ``from .. import name``, which contain no word of their path.
    anchor_words = {
        path: None
        if is_directory[path] or Path(path).name.lower() in {"__init__.py", "__main__.py"}
        else _reference_anchor_words(path)
        for path in local_files
    }

    def _marked(marker: str, variants: Iterable[str]) -> set[str]:
        return {f"{marker}/{variant.removeprefix('./')}" for variant in variants}

    def _lowercase(variants: set[str], unbounded: set[str]) -> tuple[frozenset[str], frozenset[str]]:
        return frozenset(variant.lower() for variant in variants), frozenset(variant.lower() for variant in unbounded)

    # Variant sets that do not depend on the source text are built once per
    # candidate, per enclosing skill root of the source, or per source
    # directory, as (variants, unbounded directory variants).
    shared_variant_cache: dict[tuple[str, bool], tuple[frozenset[str], frozenset[str]]] = {}
    skill_variant_cache: dict[tuple[str, Path], tuple[frozenset[str], frozenset[str]]] = {}
    relative_variant_cache: dict[tuple[str, Path], tuple[frozenset[str], frozenset[str]]] = {}

    def _shared_variants(candidate: str, *, full_namespace: bool) -> tuple[frozenset[str], frozenset[str]]:
        key = (candidate, full_namespace)
        if key not in shared_variant_cache:
            variants = (
                set(variants_by_path[candidate])
                if full_namespace
                else _path_variants(
                    candidate,
                    include_parents=False,
                    include_source_root=False,
                    include_import_aliases=False,
                )
            )
            unbounded = set(directory_prefix_variants.get(candidate, set()))
            root_path_variants = _path_variants(
                candidate,
                include_parents=include_parents_by_path[candidate],
                include_import_aliases=False,
            )
            root_directory_variants = _directory_prefix_path_variants(candidate) if is_directory[candidate] else set()
            for marker in (_PLUGIN_ROOT_MARKER, *root_name_markers):
                variants.update(_marked(marker, root_path_variants))
                marked_directory_variants = _marked(marker, root_directory_variants)
                variants.update(marked_directory_variants)
                unbounded.update(marked_directory_variants)
            shared_variant_cache[key] = _lowercase(variants, unbounded)
        return shared_variant_cache[key]

    def _skill_root_variants(candidate: str, skill_root: Path) -> tuple[frozenset[str], frozenset[str]]:
        key = (candidate, skill_root)
        if key not in skill_variant_cache:
            variants: set[str] = set()
            unbounded: set[str] = set()
            try:
                skill_relative = Path(candidate).relative_to(skill_root).as_posix()
            except ValueError:
                pass
            else:
                variants.update(
                    _marked(
                        _SKILL_ROOT_MARKER,
                        _path_variants(
                            skill_relative,
                            include_parents=include_parents_by_path[candidate],
                            include_import_aliases=False,
                        ),
                    )
                )
                if is_directory[candidate]:
                    unbounded = _marked(_SKILL_ROOT_MARKER, _directory_prefix_path_variants(skill_relative))
                    variants.update(unbounded)
            skill_variant_cache[key] = _lowercase(variants, unbounded)
        return skill_variant_cache[key]

    def _relative_variants(candidate: str, source_directory: Path) -> tuple[frozenset[str], frozenset[str]]:
        key = (candidate, source_directory)
        if key not in relative_variant_cache:
            variants: set[str] = set()
            unbounded: set[str] = set()
            try:
                relative_path = Path(os.path.relpath(local_files[candidate], start=source_directory)).as_posix()
            except ValueError:
                relative_path = ""
            if relative_path:
                variants = _path_variants(relative_path, include_parents=False)
                if is_directory[candidate]:
                    unbounded = _directory_prefix_path_variants(relative_path)
                    variants.update(unbounded)
                try:
                    relative_without_suffix = Path(relative_path).with_suffix("").as_posix()
                except ValueError:
                    pass
                else:
                    if not relative_without_suffix.startswith("."):
                        variants.add(f".{relative_without_suffix.replace('/', '.')}")
            relative_variant_cache[key] = _lowercase(variants, unbounded)
        return relative_variant_cache[key]

    prefixed_executable_paths = {_prefixed(path) for path in executable_paths if path != "unknown"}
    reachable = {
        path for path in local_files if roles[path] in AGENT_CONTROL_PATH_ROLES or path in prefixed_executable_paths
    }
    pending = list(reachable)
    inspected: set[str] = set()
    reference_pairs = 0
    reference_comparisons = 0
    reference_match_characters = 0
    source_bytes_read = 0
    source_files_read = 0
    while pending:
        source_path = pending.pop()
        if source_path in inspected:
            continue
        inspected.add(source_path)
        control_file = local_files[source_path]
        try:
            resolved_file = control_file.resolve(strict=True)
            resolved_file.relative_to(resolved_base)
            if not resolved_file.is_file():
                return None
            source_files_read += 1
            if source_files_read > _REFERENCE_SOURCE_FILE_LIMIT:
                return None
            if _is_binary_leaf(resolved_file, Path(source_path).suffix.lower()):
                # A referenced image or font keeps its own findings as
                # agent_reference but has no path text to follow; decoding it
                # as text would fail the whole walk closed.
                continue
            file_size = resolved_file.stat().st_size
            if file_size > _REFERENCE_FILE_SIZE_LIMIT:
                return None
            source_bytes_read += file_size
            if source_bytes_read > _REFERENCE_SOURCE_BYTE_LIMIT:
                return None
            raw_text = resolved_file.read_text(encoding="utf-8")
        except (OSError, RuntimeError, UnicodeError, ValueError):
            return None
        text = raw_text.replace("\\", "/")
        source_is_plugin_manifest = roles[source_path] == "plugin_manifest"
        if source_is_plugin_manifest:
            manifest_references = _manifest_reference_text(raw_text)
            if manifest_references is None:
                return None
            text = manifest_references.replace("\\", "/")
        expanded_imports = _expanded_python_imports(raw_text, text, is_python=resolved_file.suffix.lower() == ".py")
        if expanded_imports is None:
            return None
        if expanded_imports:
            text = f"{text}\n{' '.join(expanded_imports)}"
        text = _PLUGIN_ROOT_PATTERN.sub(f"{_PLUGIN_ROOT_MARKER}/", text)
        text = _SKILL_ROOT_PATTERN.sub(f"{_SKILL_ROOT_MARKER}/", text)
        # References match without regard to case: an agent on a
        # case-insensitive file system reads ``DOCS/Guide.md`` as docs/guide.md.
        text = text.lower()
        # Index the source once and charge the shared budget only for variant
        # searches that can match. Charging every file and variant pair
        # exhausted the budget on ordinary skills.
        reference_match_characters += len(text)
        if reference_match_characters > _REFERENCE_MATCH_CHARACTER_LIMIT:
            return None
        reference_index = _reference_index(text)
        reference_words = set(_REFERENCE_WORD_PATTERN.findall(text))
        source_skill_root = enclosing_skill_roots[source_path]
        for candidate, target_file in local_files.items():
            if candidate in reachable:
                continue
            reference_pairs += 1
            if reference_pairs > _REFERENCE_PAIR_LIMIT:
                return None
            candidate_anchor_words = anchor_words[candidate]
            if candidate_anchor_words is not None and candidate_anchor_words.isdisjoint(reference_words):
                # No variant of the candidate can match without one of its
                # path words appearing whole in the source text.
                continue
            reference_comparisons += 1
            if reference_comparisons > _REFERENCE_MATCH_COMPARISON_LIMIT:
                # Retain every scanner issue rather than spend unbounded CPU or
                # trust a partial traversal of an adversarially large graph.
                return None
            same_namespace = source_skill_root == enclosing_skill_roots[candidate]
            variants, unbounded_directory_variants = _shared_variants(
                candidate, full_namespace=source_is_plugin_manifest or same_namespace
            )
            if source_skill_root is not None:
                skill_variants, skill_unbounded = _skill_root_variants(candidate, source_skill_root)
                variants = variants | skill_variants
                unbounded_directory_variants = unbounded_directory_variants | skill_unbounded
            relative_variants, relative_unbounded = _relative_variants(candidate, control_file.parent)
            variants = variants | relative_variants
            unbounded_directory_variants = unbounded_directory_variants | relative_unbounded
            searchable_variants = [
                variant for variant in variants if _reference_variant_may_match(variant, reference_index)
            ]
            reference_match_characters += len(text) * len(searchable_variants)
            if reference_match_characters > _REFERENCE_MATCH_CHARACTER_LIMIT:
                return None
            matched = False
            for variant in searchable_variants:
                if variant.endswith(".") or variant in unbounded_directory_variants:
                    trailing_boundary = ""
                elif variant.endswith("/"):
                    trailing_boundary = _REFERENCE_DIRECTORY_TRAILING_BOUNDARY
                else:
                    trailing_boundary = _REFERENCE_TRAILING_BOUNDARY
                leading_boundary = _REFERENCE_PATH_LEADING_BOUNDARY if "/" in variant else _REFERENCE_LEADING_BOUNDARY
                if re.search(rf"{leading_boundary}{re.escape(variant)}{trailing_boundary}", text):
                    matched = True
                    break
            if matched:
                reachable.add(candidate)
                if target_file.is_file():
                    pending.append(candidate)
    return frozenset(
        candidate for candidate, reference_path in issue_reference_paths.items() if reference_path in reachable
    )
