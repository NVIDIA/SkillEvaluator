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
    QUALITY_EXCLUDED_DIRS,
    QUALITY_RECOMMENDED_MAX_TOKENS,
    QUALITY_RESERVED_NAMES,
    QUALITY_RESOURCE_DIRS,
)
from skillevaluator.logging_config import get_logger
from skillevaluator.models.quality import QualityScoreResult
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.models.skill import XML_TAG_RE
from skillevaluator.validators.base import ValidatorBase
from skillevaluator.validators.markdown import markdown_link_targets

logger = get_logger(__name__)

_WORD_CHAR = r"A-Za-z0-9_"
_URL_EDGE_C0_OR_SPACE_RE = re.compile(r"^[\x00-\x20]+|[\x00-\x20]+$")
_ERROR_HANDLING_RE = re.compile(
    r"\b(?:errors?|exceptions?|invalid|fail(?:s|ed|ure|ures|ing)?|"
    r"validat(?:e|es|ed|ing|ion|ions))\b",
    re.IGNORECASE,
)


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


def _normalized_local_path(href: str) -> str | None:
    """Return a once-decoded, normalized local file path from a link destination."""
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
        path = _normalized_local_path(target)
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

    def __init__(self, min_score: int = 70) -> None:
        self.min_score = min_score

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
    def detect_skill_type(skill_path: Path) -> str:
        """Auto-detect skill type from directory structure.

        Returns one of: script-based, lib-based, resource-based, guide-only, hybrid.
        """
        scripts_dir = skill_path / "scripts"
        has_scripts = scripts_dir.is_dir() and bool(list(scripts_dir.glob("*.py")) + list(scripts_dir.glob("*.sh")))

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
            return self._validate_single_skill(skill_path)
        return self._validate_folder(skill_path)

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

        content = manifest.read_text(encoding="utf-8")
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
            result.passed = False

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
        fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
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
            path = _normalized_local_path(link_target)
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
            if XML_TAG_RE.search(str(val)):
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
            scripts_dir = skill_path / "scripts"
            if scripts_dir.exists():
                qs.has_scripts = True
                py_sh = list(scripts_dir.glob("*.py")) + list(scripts_dir.glob("*.sh"))
                qs.script_count = len(py_sh)
                if qs.script_count == 0:
                    dim.deduct(10, "warning", "scripts/ directory exists but contains no .py or .sh files")
            else:
                dim.deduct(
                    25,
                    "error",
                    "No scripts/ directory found (detected as script-based skill)",
                    "Create scripts/ directory with at least one executable script",
                )

            if "## Available Scripts" not in content and "| Script |" not in content:
                dim.deduct(
                    10,
                    "warning",
                    "No documented scripts in table format",
                    "Add '## Available Scripts' with table: | Script | Purpose | Arguments |",
                )
            if "run_script" not in content:
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

        if _ERROR_HANDLING_RE.search(content):
            qs.has_error_handling = True
        else:
            dim.deduct(
                10,
                "info",
                "No mention of error handling or validation",
                "Document expected errors and how to handle them",
            )

        # Type-aware code quality
        if qs.skill_type in ("script-based", "hybrid"):
            self._check_script_reliability(dim, skill_path)
        elif qs.skill_type == "lib-based":
            self._check_lib_reliability(dim, skill_path)

        # Universal checks
        if "## Prerequisites" not in content and "## Requirements" not in content:
            dim.deduct(
                5,
                "info",
                "No prerequisites/requirements documented",
                "Document dependencies, API keys, or setup needed",
            )
        if "## Limitations" not in content:
            dim.deduct(
                5,
                "info",
                "No limitations documented",
                "Add '## Limitations' section with known issues/constraints",
            )
        if not any(s in content for s in ["## Troubleshooting", "## Common Issues", "## FAQ"]):
            dim.deduct(
                5,
                "info",
                "No troubleshooting section documented",
                "Add '## Troubleshooting' with Error/Cause/Solution patterns",
            )

    def _check_script_reliability(self, dim, skill_path: Path) -> None:
        scripts_dir = skill_path / "scripts"
        if not scripts_dir.exists():
            return
        no_error_handling = []
        for script in scripts_dir.glob("*.py"):
            try:
                sc = script.read_text(encoding="utf-8")
            except Exception:
                continue
            has_try = "try:" in sc and "except" in sc
            has_err = any(
                p in sc
                for p in [
                    "if not ",
                    "if error",
                    "raise",
                    "assert",
                    "ValueError",
                    "FileNotFoundError",
                ]
            )
            if not (has_try or has_err):
                no_error_handling.append(script.name)

        if no_error_handling:
            dim.deduct(
                5,
                "info",
                f"Scripts may lack error handling: {', '.join(no_error_handling[:3])}",
                "Scripts should handle errors explicitly",
            )

    def _check_lib_reliability(self, dim, skill_path: Path) -> None:
        lib_dir = QualityScoreValidator.find_lib_module(skill_path)
        if not lib_dir:
            return
        no_err = []
        for py in lib_dir.rglob("*.py"):
            if py.name == "__init__.py":
                continue
            try:
                pc = py.read_text(encoding="utf-8")
            except Exception:
                continue
            has_try = "try:" in pc and "except" in pc
            has_err = any(p in pc for p in ["raise", "assert", "ValueError", "TypeError", "RuntimeError"])
            if not (has_try or has_err):
                no_err.append(py.name)
        if no_err:
            dim.deduct(
                5,
                "info",
                f"Lib modules may lack error handling: {', '.join(no_err[:3])}",
                "Library code should handle errors explicitly with try/except or raise",
            )

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

        # Token estimates
        qs.total_tokens = len(content) // 4
        fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if fm_match:
            qs.frontmatter_tokens = len(fm_match.group(1)) // 4
        inst_start = content.find("---", 3) + 3
        if inst_start > 3:
            qs.instructions_tokens = len(content[inst_start:]) // 4

        if qs.total_tokens > QUALITY_RECOMMENDED_MAX_TOKENS:
            dim.deduct(
                15,
                "error",
                (
                    f"Large skill ({qs.total_tokens} tokens, recommended max <{QUALITY_RECOMMENDED_MAX_TOKENS}). "
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
        metadata = frontmatter.get("metadata") or {}

        top_level_fields = {
            "version": 'Semantic version (e.g., "1.0.0")',
        }
        for field_name, desc in top_level_fields.items():
            if field_name not in frontmatter or frontmatter[field_name] is None:
                qs.correctness.deduct(
                    5,
                    "warning",
                    f"SKILL_SPEC recommended field missing: '{field_name}'",
                    f"Add '{field_name}' to frontmatter — {desc}",
                )

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
