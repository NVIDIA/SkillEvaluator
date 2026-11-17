# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quality Score Validator — 4-dimension skill quality analysis.

Ported from SkillEvaluator SkillQualityAnalyzer (quality_analyzer.py). Evaluates
SKILL.md files across four weighted dimensions:
  - Correctness  (0.35): structure, frontmatter, type-specific rules
  - Discoverability (0.25): description quality, naming, purpose clarity
  - Reliability  (0.25): error handling, prerequisites, troubleshooting
  - Efficiency   (0.15): token budget, repetition, instruction clarity

Produces a composite 0-100 score with A-F letter grades.

Sources: Anthropic Agent Skills best practices, Anthropic Complete Guide
gap analysis (H1-H4, M1-M5, L1, L4), OpenAI Skill Evals.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

from skillevaluator.constants import (
    EXECUTABLE_SCRIPT_PATTERNS,
    EXECUTABLE_SKILL_DIRS,
    NODE_SCRIPT_PATTERNS,
    PYTHON_SCRIPT_PATTERNS,
    QUALITY_CHARS_PER_TOKEN,
    QUALITY_DEFAULT_MIN_SCORE,
    QUALITY_EXCLUDED_DIRS,
    QUALITY_RECOMMENDED_MAX_TOKENS,
    QUALITY_RESERVED_NAMES,
    QUALITY_RESOURCE_DIRS,
)
from skillevaluator.logging_config import get_logger
from skillevaluator.models.quality import QualityScoreResult
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.models.skill import XML_TAG_RE
from skillevaluator.quality_config import CodeErrorHandlingRule, load_quality_config_with_provenance
from skillevaluator.validators.base import ValidatorBase
from skillevaluator.validators.frontmatter_parser import FRONTMATTER_PATTERN
from skillevaluator.validators.markdown import markdown_link_targets

logger = get_logger(__name__)

_WORD_CHAR = r"A-Za-z0-9_"
# Role and control tags that stay reportable in ``argument-hint`` even when they
# look like a CLI placeholder, including nested forms (``<system<payload>>``) and a
# role word with a message-part suffix (``<system_prompt>``, ``<toolCall>``,
# ``<user_message>``, ``<tool_results>``). Other suffixes name an ordinary value
# (``<tool_name>``, ``<user_id>``, ``<prompt_file>``, ``<toolchain>``,
# ``<systemd-unit>``) and may be masked. ``<script>`` takes no suffix.
_ARGUMENT_CONTROL_TAG_RE = re.compile(
    r"<(?:(?:system|developer|assistant|user|tool|context|instructions?|prompt|override|rules|policy)"
    r"(?:[_-]?(?:prompt|message|call|use|result|instruction)s?)?|script)(?=[\s<>:])",
    re.IGNORECASE,
)
# The text before a nested placeholder in its outer group: a plain name such as the
# ``R`` in ``<R<NNN>>``. Anything else (``<curl;sh<x>>``, ``<$(curl x|sh)<x>>``) is not
# a placeholder.
_ARGUMENT_NESTED_PREFIX_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
# A complete group that may be a CLI placeholder such as ``<file>``, ``<plan|apply>`` or
# ``<codec e.g. av1|h264>``: no attribute, line break, nested tag or self-closing slash.
# ``_is_argument_placeholder`` then bounds the body before the group is masked.
_ARGUMENT_PLACEHOLDER_RE = re.compile(r"<(?![^<>]*/\s*>)([A-Za-z][^<>=\r\n]*)>")
# Placeholders name one value; a longer group reads as free text and stays visible.
_ARGUMENT_PLACEHOLDER_MAX_WORDS = 4
# Shell command syntax that never appears in a placeholder: ``;``, ``&``, backticks,
# ``$(...)`` and ``${...}``.
_ARGUMENT_COMMAND_SYNTAX_RE = re.compile(r"[;&`]|\$[({]")
# Optional label before the first choice of a ``|`` list: ``stage: a | b``, ``codec e.g. av1|h264``.
_ARGUMENT_CHOICE_LABEL_RE = re.compile(r"^(?:[^|:]+:|\S+\s+(?:e\.g\.|i\.e\.),?)\s+", re.IGNORECASE)
# One choice in a ``|`` list: a single word or a quoted literal.
_ARGUMENT_CHOICE_RE = re.compile(r"\S+|\"[^\"]*\"|'[^']*'")
# A ``>`` between groups that is an arrow or a comparison, not a redirect: a
# space-delimited ``->``, ``=>`` or ``>=`` (``<old> -> <new>``, ``<n> >= <m>``), or ``>``
# or ``>=`` before a number after a space or bracket (``<n> > 0``, ``<count> (>0)``).
_ARGUMENT_ARROW_OR_COMPARISON_RE = re.compile(r"(?<!\S)(?:->|=>|>=)(?!\S)|(?<![^\s(\[])>=? ?\d+(?:\.\d+)?(?![^\s)\],])")
_URL_EDGE_C0_OR_SPACE_RE = re.compile(r"^[\x00-\x20]+|[\x00-\x20]+$")


def _is_argument_placeholder(body: str) -> bool:
    """Return whether an ``argument-hint`` group body is a short CLI placeholder.

    A placeholder has at most four words (tokens with a letter or digit, so ``|``
    and ``...`` do not count), no shell command syntax, and no space before the
    closing ``>`` (``<cmd > file>`` is a redirect). ``|`` separates choices, and
    each choice is one word or a quoted literal, so ``<plan|apply>`` and
    ``<issue_id | symptom>`` pass while a pipeline such as ``<curl x | sh>`` does not.
    """
    if body[-1].isspace() or _ARGUMENT_COMMAND_SYNTAX_RE.search(body):
        return False
    words = [token for token in body.split() if any(char.isalnum() for char in token)]
    if len(words) > _ARGUMENT_PLACEHOLDER_MAX_WORDS:
        return False
    if "|" not in body:
        return True
    choices = body.split("|")
    choices[0] = _ARGUMENT_CHOICE_LABEL_RE.sub("", choices[0].strip(), count=1)
    return all(_ARGUMENT_CHOICE_RE.fullmatch(choice.strip()) for choice in choices)


def _mask_argument_placeholder(match: re.Match[str]) -> str:
    """Mask a short CLI placeholder; leave free text and command syntax for the tag check."""
    # "[]" cannot form a new tag when masking nested syntax such as ``<R<NNN>>``.
    return "[]" if _is_argument_placeholder(match.group(1)) else match.group(0)


def _masking_hides_text(masked: str) -> bool:
    """Return whether masking left a ``>`` that could hide text inside a group.

    A ``>`` that closes no group (``<cmd>file>``) reads as a redirect. Inside a group,
    a masked placeholder must be followed directly by the group's own closing ``>``
    (``<R<NNN>>`` masks to ``<R[]>``), and the group must open with a plain name
    before it. Any other text after it (``<<cmd>file>``, ``<<x><y>>``,
    ``<<x>[y] free text>``) or before it (``<curl;sh<x>>``, ``<$(curl x|sh)<x>>``)
    means the outer group is not a placeholder. An arrow or comparison between
    groups (``<old> -> <new>``, ``<n> >= 1``) is not a redirect.
    """
    operators = {
        match.start() + match.group().index(">") for match in _ARGUMENT_ARROW_OR_COMPARISON_RE.finditer(masked)
    }
    opened: list[int] = []
    after_mask = False
    index = 0
    while index < len(masked):
        char = masked[index]
        if after_mask and char != ">":
            return True
        after_mask = False
        if char == "<":
            opened.append(index)
        elif char == ">":
            if not opened:
                if index not in operators:
                    return True
            else:
                opened.pop()
        elif opened and masked.startswith("[]", index):
            if not _ARGUMENT_NESTED_PREFIX_RE.fullmatch(masked, opened[-1] + 1, index):
                return True
            after_mask = True
            index += 1
        index += 1
    return after_mask


def _mask_argument_hint(hint: str) -> str:
    """Mask the short CLI placeholders in a string ``argument-hint`` for the XML tag check.

    Control tags are checked on the raw value first: masking an inner placeholder
    could otherwise erase an enclosing tag such as ``<system<payload>>``. A ``>``
    left after masking reads as a redirect when it closes no group (``<cmd>file>``,
    ``<cmd 2>file>``, ``<curl x> /tmp/y>``), and a group that holds anything after a
    masked placeholder other than its own closing ``>`` (``<<cmd>file>``,
    ``<<x><y>>``) is not a placeholder. The raw hint is then returned, so every
    group in it stays visible to the tag check.
    """
    if _ARGUMENT_CONTROL_TAG_RE.search(hint):
        return hint
    masked = _ARGUMENT_PLACEHOLDER_RE.sub(_mask_argument_placeholder, hint)
    if _masking_hides_text(masked):
        return hint
    return masked


def _contains_term(text: str, term: str) -> bool:
    """Match a word or phrase without accepting it inside another word."""
    normalized = term.strip()
    return bool(
        re.search(
            rf"(?<![{_WORD_CHAR}]){re.escape(normalized)}(?![{_WORD_CHAR}])",
            text,
            re.IGNORECASE,
        )
    )


def _contains_any_term(text: str, terms: Iterable[str]) -> bool:
    return any(_contains_term(text, term) for term in terms)


def _contains_any_word(text: str, words: Iterable[str]) -> bool:
    # Whitespace inside a multi-word term matches any run of whitespace, so wrapped text still matches.
    pattern = "|".join(r"\s+".join(map(re.escape, word.split())) for word in words)
    return bool(re.search(rf"(?<!\w)(?:{pattern})(?!\w)", text, re.IGNORECASE))


def _has_api_documentation(content: str) -> bool:
    api_patterns = (
        r"\bimport\s+[A-Za-z_][A-Za-z0-9_.]*",
        r"\bfrom\s+[A-Za-z_][A-Za-z0-9_.]*\s+import\b",
        r"\bapi\b",
        r"\bmodules?\b",
        r"\blibrar(?:y|ies)\b",
        r"\bpackages?\b",
        r"\bclasses?\b",
        r"\bfunctions?\b",
    )
    return any(re.search(pattern, content, re.IGNORECASE) for pattern in api_patterns)


def _normalized_quality_local_path(href: str) -> str | None:
    """Normalize a local path using Quality Score's legacy classification order.

    This intentionally remains separate from Hygiene's stricter path policy:
    issue #108 requires this PR to leave existing Quality scores unchanged.
    """
    href = _URL_EDGE_C0_OR_SPACE_RE.sub("", href)
    try:
        parsed = urlsplit(href)
    except ValueError:
        return None
    if parsed.scheme or parsed.netloc:
        return None

    path = unquote(parsed.path).replace("\\", "/")
    if path.startswith("/"):
        return None
    try:
        if urlsplit(path).scheme:
            return None
    except ValueError:
        return None
    if path.endswith(("/", "/.", "/..")):
        return None
    return posixpath.normpath(path)


def _has_nested_markdown_reference(content: str) -> bool:
    """Return whether a reference document links to another local Markdown document."""
    for target in markdown_link_targets(content):
        path = _normalized_quality_local_path(target)
        if path is None or not path.casefold().endswith(".md"):
            continue
        if path.casefold() == "../skill.md":
            continue
        return True
    return False


class QualityScoreValidator(ValidatorBase):
    """Evaluates skill quality across 4 weighted dimensions.

    Produces a 0-100 composite score with A-F letter grades. Designed to
    be compatible with SkillEvaluator Tier 1 scoring while composing with
    existing SkillEvaluator SchemaValidator checks.
    """

    def __init__(self, min_score: int = QUALITY_DEFAULT_MIN_SCORE, quality_config: Path | None = None) -> None:
        self.min_score = min_score
        self.quality_config, self.quality_config_provenance = load_quality_config_with_provenance(quality_config)

    @property
    def name(self) -> str:
        return "Quality Score (4-Dimension Analysis)"

    @property
    def description(self) -> str:
        return (
            "Skill quality scoring across Correctness (35%), Discoverability (25%),"
            " Reliability (25%), and Efficiency (15%)"
        )

    # -----------------------------------------------------------------
    # Skill type detection
    # -----------------------------------------------------------------

    @staticmethod
    def _executable_files(skill_path: Path, patterns: tuple[str, ...] = EXECUTABLE_SCRIPT_PATTERNS) -> list[Path]:
        files: list[Path] = []
        for dirname in EXECUTABLE_SKILL_DIRS:
            directory = skill_path / dirname
            if directory.is_dir():
                for pattern in patterns:
                    files.extend(sorted(directory.glob(pattern)))
        return files

    @staticmethod
    def _runs_only_node_scripts(script_files: list[Path]) -> bool:
        """Whether every bundled script is a Node script.

        ``run_script`` is the call the script checks look for in the instructions. It names a
        runner this skill does not use when all its scripts are Node scripts, so such a skill is
        not asked for it. Skills that also bundle Python or shell scripts keep the check.
        """
        return bool(script_files) and all(
            any(script.match(pattern) for pattern in NODE_SCRIPT_PATTERNS) for script in script_files
        )

    @staticmethod
    def _has_executable_directory(skill_path: Path) -> bool:
        return any((skill_path / dirname).is_dir() for dirname in EXECUTABLE_SKILL_DIRS)

    @staticmethod
    def detect_skill_type(skill_path: Path) -> str:
        """Auto-detect skill type from directory structure.

        Returns one of: script-based, lib-based, resource-based, guide-only, hybrid.
        """
        has_scripts = bool(QualityScoreValidator._executable_files(skill_path))

        has_lib = False
        for d in skill_path.iterdir():
            if (
                d.is_dir()
                and d.name not in QUALITY_EXCLUDED_DIRS
                and not d.name.startswith(".")
                and (d / "__init__.py").exists()
            ):
                has_lib = True
                break

        has_resources = any((skill_path / d).exists() for d in QUALITY_RESOURCE_DIRS)

        if has_scripts and (has_lib or has_resources):
            return "hybrid"
        if has_scripts:
            return "script-based"
        if has_lib:
            return "lib-based"
        if has_resources:
            return "resource-based"
        return "guide-only"

    @staticmethod
    def find_lib_module(skill_path: Path) -> Path | None:
        """Find the Python library module directory within a skill."""
        for d in skill_path.iterdir():
            if (
                d.is_dir()
                and d.name not in QUALITY_EXCLUDED_DIRS
                and not d.name.startswith(".")
                and (d / "__init__.py").exists()
            ):
                return d
        return None

    # -----------------------------------------------------------------
    # Main validate entry point
    # -----------------------------------------------------------------

    def validate(self, skill_path: Path) -> ValidationResult:
        """Validate a skill directory and produce quality scores."""
        if self._is_skill_directory(skill_path):
            result = self._validate_single_skill(skill_path)
        else:
            result = self._validate_folder(skill_path)
        # min_score sits beside the digest rather than in it: the digest names the rules, min_score the pass bar.
        result.metadata["quality_config"] = {**self.quality_config_provenance, "min_score": self.min_score}
        return result

    def _validate_folder(self, root: Path) -> ValidationResult:
        """Validate all skills in a folder, aggregating quality results."""
        skill_dirs = self._find_all_skills(root)
        if not skill_dirs:
            result = ValidationResult(validator_name="QUALITY")
            result.add_finding(
                Finding(
                    category="QUALITY",
                    severity=Severity.HIGH,
                    check_name="skill_discovery",
                    message="No skills found in target directory",
                    file_path=str(root),
                )
            )
            return result

        result = ValidationResult(validator_name="QUALITY", validator_description=self.description)
        all_quality: list[dict] = []
        for skill_dir in skill_dirs:
            sub = self._validate_single_skill(skill_dir)
            result.merge_with_prefix(sub, skill_dir.name)
            if sub.metadata.get("quality_scores"):
                all_quality.append(sub.metadata["quality_scores"])

        if all_quality:
            result.metadata["quality_scores_all"] = all_quality
            avg_score = sum(q["overall_score"] for q in all_quality) / len(all_quality)
            dim_names = ["correctness", "discoverability", "reliability", "efficiency"]
            avg_dims: dict[str, dict] = {}
            for dname in dim_names:
                dim_scores = [q["dimensions"][dname] for q in all_quality if "dimensions" in q]
                if dim_scores:
                    avg_dims[dname] = {
                        "score": round(sum(d["score"] for d in dim_scores) / len(dim_scores), 1),
                        "weight": dim_scores[0]["weight"],
                        "issues_count": sum(d["issues_count"] for d in dim_scores),
                    }
            result.metadata["quality_scores"] = {
                "overall_score": round(avg_score, 1),
                "grade": _score_to_grade(avg_score),
                "skill_count": len(all_quality),
                "dimensions": avg_dims,
            }
        return result

    def _validate_single_skill(self, skill_path: Path) -> ValidationResult:
        """Run the full 4-dimension quality analysis on one skill."""
        result = ValidationResult(
            validator_name="QUALITY",
            validator_description=self.description,
        )
        qs = QualityScoreResult(skill_name=skill_path.name)

        manifest = self._find_skill_manifest(skill_path)
        if not manifest:
            result.add_finding(
                Finding(
                    category="QUALITY",
                    severity=Severity.HIGH,
                    check_name="skill_manifest",
                    message="SKILL.md not found",
                    file_path=str(skill_path),
                )
            )
            result.metadata["quality_scores"] = qs.to_dict()
            return result

        content = manifest.read_text(encoding="utf-8-sig")
        lines = content.split("\n")

        frontmatter_data = self._parse_frontmatter(content)
        if frontmatter_data:
            qs.has_frontmatter = True

        qs.skill_type = self.detect_skill_type(skill_path)
        logger.debug(f"Detected skill type for '{qs.skill_name}': {qs.skill_type}")

        self._check_correctness(qs, content, skill_path, frontmatter_data)
        self._check_discoverability(qs, content, frontmatter_data)
        self._check_reliability(qs, content, skill_path)
        self._check_efficiency(qs, content, lines, skill_path, frontmatter_data)
        self._check_spec_fields(qs, frontmatter_data)

        # Convert quality issues into SkillEvaluator Findings
        for qi in qs.all_issues:
            sev_map = {"error": Severity.HIGH, "warning": Severity.MEDIUM, "info": Severity.LOW}
            result.add_finding(
                Finding(
                    category="QUALITY",
                    severity=sev_map.get(qi.severity, Severity.LOW),
                    check_name=f"quality_{qi.dimension}",
                    message=qi.message,
                    file_path=str(manifest),
                    suggestion=qi.suggestion,
                    metadata={"dimension": qi.dimension, "deduction": qi.deduction},
                )
            )

        # Determine pass/fail based on min_score
        if qs.overall_score < self.min_score:
            result.fail_gate(
                f"Quality score {qs.overall_score:.1f}/100 is below the required {self.min_score}/100 in {manifest}"
            )

        result.metadata["quality_scores"] = qs.to_dict()

        result.add_success(
            check_name="quality_score",
            message=f"Score: {qs.overall_score:.1f}/100 (Grade: {qs.grade})",
            overall_score=round(qs.overall_score, 1),
            grade=qs.grade,
            skill_type=qs.skill_type,
        )
        return result

    # -----------------------------------------------------------------
    # Frontmatter parsing
    # -----------------------------------------------------------------

    @staticmethod
    def _parse_frontmatter(content: str) -> dict | None:
        fm_match = FRONTMATTER_PATTERN.match(content)
        if not fm_match:
            return None
        try:
            data = yaml.safe_load(fm_match.group(1))
            return data if isinstance(data, dict) else None
        except yaml.YAMLError:
            return None

    # -----------------------------------------------------------------
    # Correctness dimension (weight: 0.35)
    # -----------------------------------------------------------------

    def _check_correctness(
        self,
        qs: QualityScoreResult,
        content: str,
        skill_path: Path,
        frontmatter: dict | None,
    ) -> None:
        dim = qs.correctness

        if frontmatter:
            self._check_frontmatter_correctness(dim, qs, frontmatter)

        # Instructions — presence is enforced by SchemaValidator; quality only
        # tracks the flag for downstream heuristics (action verbs, list format).
        qs.has_instructions = "## Instructions" in content or "## Usage" in content

        # Type-specific checks
        self._check_type_specific(dim, qs, content, skill_path)

        # Examples — presence of ## Examples heading is enforced by SchemaValidator
        # at MEDIUM. Quality checks for actual example *content* (code fences, etc.).
        if "```" in content or "Example:" in content or "**Example:**" in content:
            qs.has_examples = True
        elif "## Examples" not in content:
            dim.deduct(5, "info", "No examples provided", "Add example usage with code blocks")

        # Windows-style paths
        if re.search(r"(?:scripts|references|assets)\\[\w\\]+", content):
            dim.deduct(
                10,
                "warning",
                "Windows-style paths detected",
                "Use forward slashes for cross-platform compatibility",
            )

        # README.md is an allowed, human-facing supporting file per SkillEvaluator
        # HOW_TO_CONTRIBUTE_SKILLS.md ("Optional Supporting Directories") and is
        # listed as a valid optional file in docs/TIER1.md. Under progressive
        # disclosure, agents only load a supporting file when SKILL.md references
        # it, so an unreferenced README.md costs zero agent context and must not
        # be penalized. The only genuine risk (Anthropic H2 / Codex skill-creator
        # guidance) is when SKILL.md links to README.md, which pulls human-facing
        # docs into the agent context window — flag just that case.
        if (skill_path / "README.md").exists() and self._references_readme(content):
            dim.deduct(
                5,
                "warning",
                "SKILL.md references README.md (pulls human-facing docs into agent context)",
                "Keep README.md human-facing and unreferenced; move any agent-facing "
                "content into SKILL.md or a references/ file",
            )

    @staticmethod
    def _references_readme(content: str) -> bool:
        """Return whether SKILL.md contains a parsed Markdown link to README.md."""
        for link_target in markdown_link_targets(content):
            path = _normalized_quality_local_path(link_target)
            if path is not None and path.casefold() == "readme.md":
                return True
        return False

    def _check_frontmatter_correctness(
        self,
        dim,
        _qs: QualityScoreResult,
        fm: dict,
    ) -> None:
        """Validate frontmatter fields that go beyond basic SchemaValidator checks."""
        # XML tags in non-name/description fields (Anthropic H3)
        for key, val in fm.items():
            if key in ("name", "description"):
                continue
            values = [str(val)]
            # A string argument-hint documents CLI placeholders (``<file> [--strict]``).
            if key == "argument-hint" and isinstance(val, str):
                values = [_mask_argument_hint(val)]
            elif key == "argument-hint" and isinstance(val, list) and val and all(isinstance(i, str) for i in val):
                # YAML reads an unquoted ``[<file>]`` hint as a list. Check every item and
                # the joined hint on its own, so a group split across items stays visible
                # and no tag forms across the boundary between two checked values.
                values = [*map(_mask_argument_hint, val), _mask_argument_hint(" ".join(val))]
            if any(XML_TAG_RE.search(checked) for checked in values):
                dim.deduct(
                    15,
                    "error",
                    f"XML tags in frontmatter field '{key}' (potential prompt injection)",
                    "Remove XML angle brackets from all frontmatter fields",
                )
                break

        name = str(fm.get("name", "")).strip()
        desc = str(fm.get("description", "")).strip()

        if name:
            if not re.match(r"^[a-z0-9-]+$", name):
                dim.deduct(
                    15,
                    "error",
                    f"Invalid name format: '{name}' (lowercase/numbers/hyphens only)",
                    "Use only lowercase letters, numbers, and hyphens",
                )
            if _contains_any_term(name, QUALITY_RESERVED_NAMES):
                dim.deduct(
                    15,
                    "error",
                    "Name contains reserved word (anthropic, claude)",
                    "Remove reserved words from skill name",
                )
            if XML_TAG_RE.search(name):
                dim.deduct(15, "error", "Name contains XML tags", "Remove XML tags from skill name")

        if desc and XML_TAG_RE.search(desc):
            dim.deduct(15, "error", "Description contains XML tags", "Remove XML tags from description")

    def _check_type_specific(
        self,
        dim,
        qs: QualityScoreResult,
        content: str,
        skill_path: Path,
    ) -> None:
        """Apply type-specific correctness checks."""
        skill_type = qs.skill_type

        if skill_type in ("script-based", "hybrid"):
            script_files = self._executable_files(skill_path)
            if self._has_executable_directory(skill_path):
                qs.has_scripts = True
                qs.script_count = len(script_files)
                if qs.script_count == 0:
                    dim.deduct(
                        10, "warning", "scripts/ or tools/ exists but contains no .py, .sh, .mjs, .cjs or .js files"
                    )
            else:
                dim.deduct(
                    25,
                    "error",
                    "No scripts/ or tools/ directory found (detected as script-based skill)",
                    "Create scripts/ or tools/ with at least one executable script",
                )

            if "## Available Scripts" not in content and "| Script |" not in content:
                dim.deduct(
                    10,
                    "warning",
                    "No documented scripts in table format",
                    "Add '## Available Scripts' with table: | Script | Purpose | Arguments |",
                )
            if "run_script" not in content and not self._runs_only_node_scripts(script_files):
                dim.deduct(
                    10,
                    "warning",
                    "Instructions don't mention 'run_script'",
                    "Add explicit run_script() call examples",
                )

        elif skill_type == "lib-based":
            lib_dir = self.find_lib_module(skill_path)
            if lib_dir:
                qs.has_lib_module = True
                if not (skill_path / "pyproject.toml").exists():
                    dim.deduct(
                        10,
                        "warning",
                        "Lib-based skill missing pyproject.toml",
                        "Add pyproject.toml with package metadata and dependencies",
                    )
                if not _has_api_documentation(content):
                    dim.deduct(
                        10,
                        "warning",
                        "Lib-based skill lacks API/import documentation",
                        "Document how to import and use the library",
                    )
            else:
                dim.deduct(
                    15,
                    "warning",
                    "Detected as lib-based but no Python module found",
                    "Ensure module directory has __init__.py",
                )

            present_res = [d for d in QUALITY_RESOURCE_DIRS if (skill_path / d).exists()]
            if present_res:
                res_kw = ["template", "asset", "design", "style", "css", "html", "resource"]
                if not _contains_any_term(content, res_kw):
                    dim.deduct(
                        5,
                        "info",
                        f"Lib-based skill has resource directories ({', '.join(present_res)}) "
                        "but SKILL.md lacks resource documentation",
                    )

        elif skill_type == "resource-based":
            res_kw = ["template", "asset", "design", "style", "css", "html", "resource"]
            if not _contains_any_term(content, res_kw):
                dim.deduct(
                    10,
                    "warning",
                    "Resource-based skill lacks documentation of available resources",
                    "Document available templates, assets, and design resources in SKILL.md",
                )

        elif skill_type == "guide-only":
            body_lines = len([ln for ln in content.split("\n") if ln.strip()])
            if body_lines < 20:
                dim.deduct(
                    15,
                    "warning",
                    f"Guide-only skill has very little content ({body_lines} lines)",
                    "Guide skills should have detailed instructions since they have no code",
                )

    # -----------------------------------------------------------------
    # Discoverability dimension (weight: 0.25)
    # -----------------------------------------------------------------

    def _check_discoverability(
        self,
        qs: QualityScoreResult,
        content: str,
        frontmatter: dict | None,
    ) -> None:
        dim = qs.discoverability

        desc = ""
        if frontmatter:
            desc = str(frontmatter.get("description", "")).strip()

        if desc:
            if len(desc) < 20:
                dim.deduct(
                    20,
                    "warning",
                    f"Description too short ({len(desc)} chars, recommend 50-150)",
                    "Add more context: what tasks does this skill handle?",
                )
            elif len(desc) > 200:
                dim.deduct(
                    5,
                    "info",
                    f"Description very long ({len(desc)} chars, recommend 50-150)",
                    "Keep descriptions concise for progressive disclosure",
                )

            trigger_words = ["use", "when", "for", "helps", "allows"]
            if not _contains_any_term(desc, trigger_words):
                dim.deduct(
                    10,
                    "info",
                    "Description doesn't mention WHEN to use this skill",
                    "Add trigger context: 'Use for...', 'When you need to...'",
                )

            vague_words = ["something", "things", "stuff", "various", "general"]
            if _contains_any_term(desc, vague_words):
                dim.deduct(
                    15,
                    "warning",
                    "Description contains vague words",
                    "Be specific about what this skill does",
                )

            person_phrases = ["i can", "i will", "you can", "you should", "your", "my", "we can"]
            if _contains_any_term(desc, person_phrases):
                dim.deduct(
                    15,
                    "warning",
                    "Description uses first/second person",
                    "Use third person: 'Processes files' not 'I can process'",
                )

            # Broad description without negative triggers (M1)
            generic = ["data", "files", "documents", "project", "manage", "handle", "process"]
            negatives = ["not for", "do not use", "instead use", "except when", "not when"]
            if len(desc) > 100 and _contains_any_term(desc, generic) and not _contains_any_term(desc, negatives):
                dim.deduct(
                    5,
                    "info",
                    "Broad description without negative triggers may cause over-triggering",
                    "Add boundary phrases like 'Do NOT use for...'",
                )

        if "## Purpose" not in content:
            dim.deduct(5, "info", "No '## Purpose' section", "Add purpose section to clarify use cases")

        sn = qs.skill_name
        if len(sn) < 5:
            dim.deduct(
                10,
                "warning",
                f"Skill name very short: '{sn}'",
                "Use descriptive names like 'crypto-utils' not 'crypto'",
            )
        if not re.match(r"^[a-z][a-z0-9_-]*$", sn):
            dim.deduct(
                10,
                "warning",
                f"Skill name not following convention: '{sn}'",
                "Use lowercase with hyphens: my-skill-name",
            )

    # -----------------------------------------------------------------
    # Reliability dimension (weight: 0.25)
    # -----------------------------------------------------------------

    def _check_reliability(
        self,
        qs: QualityScoreResult,
        content: str,
        skill_path: Path,
    ) -> None:
        dim = qs.reliability
        error_handling = self.quality_config["error_handling"]

        mentioned = error_handling["mentioned"]
        qs.has_error_handling = _contains_any_word(content, mentioned["terms"])
        if not qs.has_error_handling and mentioned["points"]:
            dim.deduct(mentioned["points"], "info", mentioned["message"], mentioned["suggestion"])

        # Type-aware code quality
        if qs.skill_type in ("script-based", "hybrid"):
            scripts = self._executable_files(skill_path, patterns=PYTHON_SCRIPT_PATTERNS)
            self._check_code_error_handling(dim, scripts, error_handling["scripts"])
        elif qs.skill_type == "lib-based":
            lib_dir = QualityScoreValidator.find_lib_module(skill_path)
            if lib_dir:
                modules = [py for py in lib_dir.rglob("*.py") if py.name != "__init__.py"]
                self._check_code_error_handling(dim, modules, error_handling["lib"])

        for expected in self.quality_config["expected_headers"]:
            accepted_headings = [expected["heading"], *expected["alternatives"]]
            if not any(heading in content for heading in accepted_headings):
                dim.deduct(expected["points"], "info", expected["message"], expected["suggestion"])

    @staticmethod
    def _check_code_error_handling(dim, files: list[Path], rule: CodeErrorHandlingRule) -> None:
        if not rule["points"]:
            return
        unhandled = []
        for path in files:
            try:
                code = path.read_text(encoding="utf-8")
            except Exception:
                continue
            has_try = "try:" in code and "except" in code
            if not (has_try or any(marker in code for marker in rule["markers"])):
                unhandled.append(path.name)
        if unhandled:
            dim.deduct(rule["points"], "info", f"{rule['message']}: {', '.join(unhandled[:3])}", rule["suggestion"])

    # -----------------------------------------------------------------
    # Efficiency dimension (weight: 0.15)
    # -----------------------------------------------------------------

    def _check_efficiency(
        self,
        qs: QualityScoreResult,
        content: str,
        lines: list[str],
        skill_path: Path,
        _frontmatter: dict | None,
    ) -> None:
        dim = qs.efficiency

        # Token estimates (characters / QUALITY_CHARS_PER_TOKEN); not a tokenizer count
        qs.total_tokens = len(content) // QUALITY_CHARS_PER_TOKEN
        fm_match = FRONTMATTER_PATTERN.match(content)
        if fm_match and _frontmatter:
            qs.frontmatter_tokens = len(fm_match.group(1)) // QUALITY_CHARS_PER_TOKEN
            qs.instructions_tokens = len(fm_match.group(2)) // QUALITY_CHARS_PER_TOKEN

        # The token budget is a recommendation: exceeding it is advisory (MEDIUM),
        # and the deduction still counts toward the --min-score gate.
        if qs.total_tokens > QUALITY_RECOMMENDED_MAX_TOKENS:
            dim.deduct(
                15,
                "warning",
                (
                    f"Large skill (~{qs.total_tokens} estimated tokens at {QUALITY_CHARS_PER_TOKEN} characters per token, "
                    f"recommended max <{QUALITY_RECOMMENDED_MAX_TOKENS}; a real tokenizer may count more). "
                    f"Per agentskills.io, SKILL.md should be concise (~500 lines) — "
                    f"large skill bodies increase token cost after invocation; long or unfocused "
                    f"top-level descriptions can degrade agent routing accuracy"
                ),
                "Keep required sections concise; move detailed examples, reference material, "
                "and supporting docs to the references/ directory",
            )

        # Repetition (compare against non-empty lines to avoid false positives from blank lines)
        non_empty = [ln for ln in lines if ln.strip()]
        stripped = {ln.strip() for ln in non_empty}
        if stripped and len(stripped) < len(non_empty) * 0.7:
            dim.deduct(
                10,
                "info",
                "High line repetition detected",
                "Remove duplicate instructions or use references",
            )

        # Instruction clarity
        for heading in ("## Instructions", "## Usage"):
            if heading not in content:
                continue
            parts = content.split(f"\n{heading}\n", 1)
            section = parts[1].split("\n## ", 1)[0] if len(parts) > 1 else ""
            break
        else:
            section = ""
        if section:
            action_words = ["use", "call", "run", "execute", "pass", "set", "add", "mark", "update", "keep"]
            if not _contains_any_term(section, action_words):
                dim.deduct(
                    15,
                    "warning",
                    "Instructions lack clear action verbs",
                    "Use imperative: 'Use run_script with...', 'Call activate_skill...'",
                )
            if not ("- " in section or "1." in section or "* " in section):
                dim.deduct(
                    5,
                    "info",
                    "Instructions not in list format",
                    "Use bullet points or numbered steps for clarity",
                )

        # Corporate buzzwords
        complex_words = ["utilize", "facilitate", "leverage", "paradigm", "synergy"]
        if _contains_any_term(content, complex_words):
            dim.deduct(
                5,
                "info",
                "Uses complex/corporate language",
                "Use simple, direct language: 'use' not 'utilize'",
            )

        # Reference file naming
        refs_dir = skill_path / "references"
        if refs_dir.exists():
            vague = ["doc", "file", "data", "info", "misc", "temp"]
            for ref in refs_dir.glob("*.md"):
                stem = ref.stem.lower()
                if stem.isdigit() or stem in vague or len(stem) < 4:
                    dim.deduct(
                        5,
                        "info",
                        f"Non-descriptive filename: {ref.name}",
                        "Use descriptive names: 'form_validation_rules.md' not 'doc2.md'",
                    )
                    break

            # Deeply nested references
            for ref in refs_dir.glob("*.md"):
                try:
                    rc = ref.read_text(encoding="utf-8")
                    if _has_nested_markdown_reference(rc):
                        dim.deduct(
                            10,
                            "warning",
                            f"Deeply nested references in {ref.name}",
                            "Keep references one level deep from SKILL.md",
                        )
                        break
                except Exception:
                    pass

            # Non-doc files in references/
            non_doc = {".py", ".sh", ".json", ".csv", ".yaml", ".yml", ".toml"}
            for ref in refs_dir.iterdir():
                if ref.is_file() and ref.suffix in non_doc:
                    dim.deduct(
                        5,
                        "info",
                        f"Non-doc file in references/: {ref.name}",
                        "Code belongs in scripts/, data in assets/. references/ is for .md docs.",
                    )
                    break

    # -----------------------------------------------------------------
    # SKILL_SPEC field checks (adds to correctness)
    # -----------------------------------------------------------------

    def _check_spec_fields(self, qs: QualityScoreResult, frontmatter: dict | None) -> None:
        if not frontmatter:
            return
        metadata = frontmatter.get("metadata")
        if not isinstance(metadata, dict):
            # A list, string, or scalar is invalid metadata per the schema; score it
            # as absent metadata rather than indexing into it.
            metadata = {}

        nested_fields = {
            "author": "Author name or team (under metadata:)",
            "tags": "Categorization tags (under metadata:, list of 1-5 items)",
        }
        for field_name, desc in nested_fields.items():
            if field_name not in metadata or metadata[field_name] is None:
                qs.correctness.deduct(
                    5,
                    "warning",
                    f"SKILL_SPEC recommended field missing: 'metadata.{field_name}'",
                    f"Add '{field_name}' under metadata: — {desc}",
                )


def _score_to_grade(score: float) -> str:
    """Convert numeric score to letter grade (module-level helper for folder aggregation)."""
    from skillevaluator.models.quality import score_to_grade

    return score_to_grade(score)
