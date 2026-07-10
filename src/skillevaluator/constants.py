# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants and configuration for SkillEvaluator.

Based on Skill Evaluator HOW_TO_CONTRIBUTE_SKILLS.md, HOW_TO_CONTRIBUTE_WORKFLOW_RULES.md specifications.
"""

# =============================================================================
# SKILLS CONSTANTS
# =============================================================================

# Skill structure constants
SKILL_MANIFEST_FILE = "SKILL.md"
SKILL_FOLDER_PREFIX = "skills/"
TEAM_SKILL_FOLDER_PREFIX = "team-skills/"

# Required fields per Skill Evaluator specification
REQUIRED_SKILL_FIELDS = ["name", "description"]

# Forbidden fields - these should NOT be present in skills
FORBIDDEN_SKILL_FIELDS = ["alwaysApply", "always_apply", "globs"]

# Reserved words that must NOT appear in skill names (agentskills.io + Skill Evaluator)
RESERVED_SKILL_NAMES = {"anthropic", "claude"}

# Required body sections (HIGH severity if missing).
# Per agentskills.io spec, the SKILL.md body has no format restrictions, so we
# do not enforce any hard-required headings. Kept as a list for future
# extensibility if a hard requirement is ever reinstated.
REQUIRED_BODY_SECTIONS: list[str] = []

# Recommended (not required) body sections — reported at MEDIUM, not HIGH.
# These are convention nudges to help authors structure skills consistently and
# to give the quality scorer a stable anchor for instruction-quality heuristics.
RECOMMENDED_BODY_SECTIONS = ["## Instructions", "## Examples"]

# Skill-root subdirectories the schema validator recognises; anything else
# triggers a LOW "unexpected in skill root" finding. ``tools/`` is the
# agentskills.io name for executable helpers (the skill_evaluator onboarding renames
# ``scripts/`` -> ``tools/``); ``config/`` holds data-driven runtime config the
# skill loads.
DEFAULT_ALLOWED_SKILL_DIRS = frozenset({"references", "scripts", "assets", "evals", "tools", "config"})

# Env var that lets consumers EXTEND the allowed skill-root directory set per
# repo (comma- or whitespace-separated) without editing bundled config — e.g.
# ``SKILLEVALUATOR_SCHEMA_ALLOWED_DIRS="data,fixtures"``. Names are added to,
# never replace, ``DEFAULT_ALLOWED_SKILL_DIRS``.
SCHEMA_ALLOWED_DIRS_ENV = "SKILLEVALUATOR_SCHEMA_ALLOWED_DIRS"

# =============================================================================
# RULES CONSTANTS
# =============================================================================

# Rules structure constants
RULES_FILE_EXTENSION = ".mdc"
RULES_FOLDER_PREFIX = "team-rules/"

# Required fields for Rules (Cursor-standard top-level fields)
# NOTE: alwaysApply and globs are REQUIRED for Rules but FORBIDDEN for Skills
REQUIRED_RULES_FIELDS = ["alwaysApply", "title", "description"]

# =============================================================================
# WORKFLOWS CONSTANTS
# =============================================================================

# Workflows structure constants
WORKFLOWS_MANIFEST_FILE = "workflow-rules.mdc"
WORKFLOWS_README_FILE = "README.md"
WORKFLOWS_REFERENCES_DIR = "references"
WORKFLOWS_SCRIPTS_DIR = "scripts"
WORKFLOWS_FOLDER_PREFIX = "workflows/"
TEAM_WORKFLOWS_FOLDER_PREFIX = "team-workflows/"

# Required fields for Workflows main file (workflow-rules.mdc)
REQUIRED_WORKFLOWS_FIELDS = ["alwaysApply", "title", "description"]

# Required metadata fields for Workflows (more strict than Rules)
REQUIRED_WORKFLOWS_METADATA_FIELDS = ["author"]

# Optional metadata fields for Workflows
OPTIONAL_WORKFLOWS_METADATA_FIELDS = [
    "tags",
    "language",
    "framework",
    "library",
    "version",
    "project",
    "team",
    "domain",
]

# =============================================================================
# SHARED CONSTANTS
# =============================================================================

# Field length constraints per Skill Evaluator specification
NAME_MIN_LENGTH = 1
NAME_MAX_LENGTH = 64
TITLE_MIN_LENGTH = 1
TITLE_MAX_LENGTH = 256
DESCRIPTION_MIN_LENGTH = 1
DESCRIPTION_MAX_LENGTH = 1024
COMPATIBILITY_MAX_LENGTH = 500

# Maximum recommended line counts
MAX_SKILL_MD_LINES = 500
MAX_RULES_MDC_LINES = 500
MAX_WORKFLOWS_MDC_LINES = 1000  # Workflows can be longer

# Naming conventions
# Name must be: lowercase, alphanumeric + hyphens, no leading/trailing/consecutive hyphens
KEBAB_CASE_PATTERN = r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$"

# Content types for auto-detection
CONTENT_TYPE_SKILL = "skill"
CONTENT_TYPE_RULES = "rules"
CONTENT_TYPE_WORKFLOWS = "workflows"
CONTENT_TYPE_PLUGIN = "plugin"
CONTENT_TYPE_UNKNOWN = "unknown"

# =============================================================================
# PLUGIN CONSTANTS (Tier 1 plugins)
# =============================================================================

# Manifest file name variants that root a bundle-reference plugin.
PLUGIN_MANIFEST_FILES = ("agent_plugin.yaml", "agent_plugin.yml")
PLUGIN_CONTAINED_MANIFEST_DIR = ".claude-plugin"
PLUGIN_CONTAINED_MANIFEST_FILE = "plugin.json"

# Reporting alignment tokens surfaced in validator metadata.
PLUGIN_MANIFEST_TYPE = "agent_bundle_yaml"
PLUGIN_MODE = "bundle_reference"
PLUGIN_CONTAINED_MANIFEST_TYPE = "claude_plugin_json"
PLUGIN_CONTAINED_MODE = "contained"

# Note: allowed MCP providers, selector sources, and top-level fields are
# enforced directly by the Pydantic model in ``skillevaluator.models.plugin``
# (the single source of truth), so no duplicate constant sets are kept here.

# Skill manifest file name variants (case-insensitive search)
SKILL_MANIFEST_VARIANTS = ("SKILL.md", "skill.md")

# Banned/deprecated packages
BANNED_PACKAGES = [
    "subprocess32",  # Use subprocess from stdlib
    "pycrypto",  # Use pycryptodome instead
]

# File extensions to scan
SCANNABLE_EXTENSIONS = {".py", ".sh", ".yaml", ".yml", ".json", ".md", ".txt"}

# Directories to skip at any depth during Tier 1 file-tree walks (scan artifact roots, version snapshots, build caches).
SCAN_EXCLUDED_DIRS = frozenset(
    {
        "evals",
        ".evals",
        "results",
        ".results",
        "versions",
        ".versions",
        "__pycache__",
        ".git",
        ".venv",
        "node_modules",
    }
)

# Generated publishing/signing artifacts that may live in a skill root after
# NVSkills CI runs. They are derived from the author-owned skill content, so
# Tier 1 should not scan them as independent source files.
SCAN_EXCLUDED_FILES = frozenset({"skill-card.md", "benchmark.md", "skill.oms.sig"})

# Environment variables consulted (in order) to identify who is submitting a
# skill, for the home-path PII check. A ``/home/<user>/`` path is only flagged
# when ``<user>`` matches this submitter identity or the skill's declared
# author -- the only home directories that realistically leak a contributor's
# identity. ``SKILLEVALUATOR_SUBMITTER`` is an explicit override; the remaining
# variables cover local shells and common public CI environments.
HOME_PATH_SUBMITTER_ENV_VARS = (
    "SKILLEVALUATOR_SUBMITTER",
    "GITHUB_ACTOR",
    "USER",
    "LOGNAME",
    "USERNAME",
)


# =============================================================================
# LICENSE VALIDATION CONSTANTS
# =============================================================================

# Standard license file names to search for
LICENSE_FILE_NAMES = [
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "LICENSE.rst",
    "LICENCE",  # British spelling
    "LICENCE.txt",
    "COPYING",
    "COPYING.txt",
    "NOTICE",
    "NOTICE.txt",
]

# Number of lines to scan for SPDX headers in source files
LICENSE_HEADER_SCAN_LINES = 50

# File extensions to scan for license headers
LICENSE_HEADER_EXTENSIONS = {".py", ".sh", ".js", ".ts", ".yaml", ".yml", ".md", ".txt", ".json"}

# SPDX header pattern for source files
SPDX_LICENSE_PATTERN = r"SPDX-License-Identifier:\s*([A-Za-z0-9.\-+]+)"


# =============================================================================
# SIMILARITY DETECTION CONSTANTS
# =============================================================================

# Four-tier similarity thresholds
SIMILARITY_CRITICAL_THRESHOLD = 0.95  # EXACT_DUPLICATE
SIMILARITY_HIGH_THRESHOLD = 0.90  # HIGH_SIMILARITY
SIMILARITY_MEDIUM_THRESHOLD = 0.75  # SIMILAR
SIMILARITY_LOW_THRESHOLD = 0.50  # LOOSELY_RELATED

# Default --threshold flag: reports CRITICAL + HIGH + MEDIUM by default
SIMILARITY_DEFAULT_THRESHOLD = 0.75

# Embedding model configuration
SIMILARITY_DEFAULT_MODEL = "nvidia/nv-embed-v1"
SIMILARITY_CHUNK_SIZE = 512  # tokens per chunk for full-body mode
SIMILARITY_CHUNK_OVERLAP = 64  # token overlap between chunks


# =============================================================================
# CONTEXT DEDUPLICATION CONSTANTS (Phase 1)
# =============================================================================

CONTENT_DEDUP_SIMILARITY_THRESHOLD = 0.80

CONTENT_DEDUP_MIN_CHUNK_CHARS = 80

# Upper bound (in characters) for treating a repeated chunk as a "trivial"
# duplicate. Short, single-file blocks dominated by comment/config-style lines
# (e.g. a recurring `# default pts-tolerance is 60 ms.` config snippet) are
# advisory at most and are capped at LOW severity. Larger duplicated blocks are
# still reported as genuine context bloat.
CONTENT_DEDUP_TRIVIAL_DUP_MAX_CHARS = 400

CONTENT_DEDUP_EMBEDDING_BATCH_SIZE = 64

# Resource ceilings for untrusted skill content. Tier 2 sends collected text
# to external embedding/LLM providers and performs pairwise comparisons, so
# every stage must have an explicit upper bound.
CONTENT_DEDUP_MAX_FILES = 256
CONTENT_DEDUP_MAX_DISCOVERED_PATHS = 4096
CONTENT_DEDUP_MAX_FILE_BYTES = 1 * 1024 * 1024
CONTENT_DEDUP_MAX_TOTAL_BYTES = 8 * 1024 * 1024
CONTENT_DEDUP_MAX_CHUNKS = 512
CONTENT_DEDUP_MAX_LLM_CLUSTERS = 50

CONTENT_DEDUP_LLM_DEFAULT_MODEL = "azure/anthropic/claude-opus-4-8"
CONTENT_DEDUP_LLM_TEMPERATURE = 0.1

CONTENT_DEDUP_SCANNABLE_EXTENSIONS = frozenset({".md", ".mdc", ".py", ".sh"})
CONTENT_DEDUP_BINARY_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".ico",
        ".zip",
        ".tar",
        ".gz",
        ".potx",
        ".pptx",
        ".lock",
    }
)

# Skipped at any path depth by Tier 2 dedup: evaluation output and version
# snapshots mirror the live skill and would dominate reports with self-matches.
# Tier 1 has its own (broader) :data:`SCAN_EXCLUDED_DIRS`; keep these two in
# sync for the artifact-dir entries.
CONTENT_DEDUP_EXCLUDED_DIRS = frozenset({"evals", ".evals", "results", ".results", "versions", ".versions"})

# Skipped by basename by Tier 2 dedup: generated reports/metadata are not
# author-owned context, so comparing them against SKILL.md produces structural
# self-matches.
CONTENT_DEDUP_EXCLUDED_FILES = frozenset({"skill-card.md", "benchmark.md", "skill.oms.sig"})


# =============================================================================
# LLM FINDING VERIFICATION CONSTANTS
# =============================================================================

LLM_VERIFY_MODEL = "azure/anthropic/claude-opus-4-8"
LLM_VERIFY_MAX_TOKENS = 512
LLM_VERIFY_TEMPERATURE = 0.0


# =============================================================================
# UNICODE SMUGGLING DETECTION CONSTANTS
# =============================================================================

UNICODE_SCAN_EXTENSIONS = {
    ".md",
    ".mdc",
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".json",
    ".txt",
    ".js",
    ".ts",
    ".html",
}
BINARY_CHECK_CHUNK_SIZE = 8192


# =============================================================================
# QUALITY SCORING CONSTANTS (ported from SkillEvaluator)
# =============================================================================

QUALITY_SCORE_WEIGHTS = {
    "correctness": 0.35,
    "discoverability": 0.25,
    "reliability": 0.25,
    "efficiency": 0.15,
}

QUALITY_GRADE_THRESHOLDS = [(90, "A"), (80, "B"), (70, "C"), (60, "D"), (0, "F")]
QUALITY_DEFAULT_MIN_SCORE = 70
QUALITY_RECOMMENDED_MAX_TOKENS = 5000
QUALITY_MAX_BODY_LINES = 500

QUALITY_RESERVED_NAMES = ["anthropic", "claude"]

QUALITY_EXCLUDED_DIRS = frozenset(
    {
        "scripts",
        "references",
        "assets",
        "eval",
        "evals",
        ".evals",
        "results",
        ".results",
        "versions",
        ".versions",
        "tests",
        "examples",
        "docs",
        "resources",
        "templates",
        "design-system",
        "__pycache__",
        ".git",
        ".venv",
        "node_modules",
    }
)

QUALITY_RESOURCE_DIRS = ("assets", "templates", "design-system", "resources")


# =============================================================================
# LLM RUBRIC EVALUATION CONSTANTS (ported from SkillEvaluator Tier 2)
# =============================================================================

RUBRIC_MAX_TOKENS = 4096
RUBRIC_MIN_SCORE = 60

RUBRIC_CRITERIA = [
    {
        "id": "description_clarity",
        "criterion": "Description is clear, specific, and explains WHEN to use the skill",
        "importance": "high",
    },
    {
        "id": "instruction_clarity",
        "criterion": "Instructions are easy to follow with clear action steps",
        "importance": "high",
    },
    {
        "id": "example_quality",
        "criterion": (
            "Examples are helpful, relevant, show proper usage, AND cover sufficient "
            "query variations for robust skill detection"
        ),
        "importance": "high",
    },
    {
        "id": "documentation_completeness",
        "criterion": "All necessary information is present (purpose, scripts, parameters)",
        "importance": "medium",
    },
    {
        "id": "scope_definition",
        "criterion": "Skill scope is well-defined (clear boundaries, not too broad/narrow)",
        "importance": "medium",
    },
    {
        "id": "professional_tone",
        "criterion": "Documentation uses professional, consistent tone and formatting",
        "importance": "low",
    },
    {
        "id": "trigger_simulation",
        "criterion": (
            "Mentally test the skill description against 5 plausible user queries that "
            "SHOULD trigger it and 3 that should NOT. Does the description enable correct "
            "routing without false positives or false negatives?"
        ),
        "importance": "high",
    },
    {
        "id": "workflow_completeness",
        "criterion": (
            "Do the instructions cover a complete end-to-end workflow? Identify any steps "
            "where the user or agent would need to figure out what to do next without guidance."
        ),
        "importance": "high",
    },
    {
        "id": "error_handling_quality",
        "criterion": (
            "Are the documented error scenarios realistic and actionable? Would the solutions "
            "actually help resolve the issues, or are they generic boilerplate?"
        ),
        "importance": "medium",
    },
]


# =============================================================================
# SCRIPT LINT CONSTANTS
# =============================================================================

SCRIPT_LINT_MAX_NESTING = 6
SCRIPT_LINT_SAFE_CONSTANTS = frozenset({0, 1, -1, 2, -1.0})


# =============================================================================
# TIER 3: AGENT EVALUATION CONSTANTS (SkillEvaluator Integration)
# =============================================================================

AGENT_EVAL_DEFAULT_TIMEOUT = 600
AGENT_EVAL_CASE_TIMEOUT = 120
AGENT_EVAL_DEFAULT_PARALLEL = 1
# Unsupported agent integrations are excluded. The default Harbor agent is
# codex; unsupported agent names are rejected by validation.
AGENT_EVAL_DEFAULT_AGENTS = "codex"

AGENT_EVAL_DIMENSIONS = [
    "security",
    "correctness",
    "discoverability",
    "effectiveness",
    "efficiency",
]

AGENT_EVAL_EVALUATORS = [
    "security",
    "skill_execution",
    "skill_efficiency",
    "accuracy",
    "goal_accuracy",
    "behavior_check",
]

AGENT_EVAL_SCORE_DEFINITION = (
    "overall = mean(Security, Correctness, Discoverability, Effectiveness, Efficiency) "
    "dimensions; Security maps to security evaluator (fallback: behavior_check)"
)

AGENT_EVAL_VERDICT_PASS = "pass"
AGENT_EVAL_VERDICT_NEUTRAL = "neutral"
AGENT_EVAL_VERDICT_FAIL = "fail"

# Tier 3 live-agent lift verdict policy. Small deltas stay neutral because
# live agent runs are noisy, especially with low attempt counts. Consumed by
# the HTML reporter to color the Tier 3 lift verdict bands.
TIER3_LIFT_PASS_THRESHOLD = 0.05
TIER3_LIFT_FAIL_THRESHOLD = -0.10

AGENT_EVAL_DATASET_REQUIRED_FIELDS = [
    "id",
    "question",
    "expected_skill",
    "ground_truth",
    "expected_behavior",
]

AGENT_EVAL_DATASET_FORMATS = {".json", ".jsonl", ".yaml", ".yml"}


# =============================================================================
# TIER 3 DIMENSION MAPPING (BENCHMARK.md / reporting)
# =============================================================================

# Dimension mapping: evaluator -> human-readable dimension (from SADD Section 5.2).
# Current payloads include a dedicated ``security`` evaluator. Older payloads
# did not, so the Security dimension retains a behavior_check fallback for
# historical reports.
DIMENSION_MAPPING: dict[str, dict] = {
    "security": {
        "evaluators": ["security"],
        "weights": [1.0],
        "fallback_evaluators": ["behavior_check"],
        "fallback_weights": [1.0],
        "question": "Is it safe to use?",
    },
    "correctness": {
        "evaluators": ["skill_execution", "accuracy"],
        "weights": [0.5, 0.5],
        "question": "Does it do what it's supposed to?",
    },
    "discoverability": {
        "evaluators": ["skill_execution", "skill_efficiency"],
        "weights": [0.5, 0.5],
        "question": "Is it loaded when it should be?",
    },
    "effectiveness": {
        "evaluators": ["goal_accuracy", "behavior_check", "accuracy"],
        "weights": [0.4, 0.3, 0.3],
        "question": "Is it better with the skill than without?",
    },
    "efficiency": {
        "evaluators": ["skill_efficiency", "token_efficiency"],
        "weights": [0.7, 0.3],
        "question": "Does it use fewer tool calls and tokens?",
    },
}

# Short, user-facing one-liner describing each Tier 3 dimension.
DIMENSION_HINTS: dict[str, str] = {
    "security": "Avoids unsafe operations, secret leakage, and unauthorized access.",
    "correctness": "Produces accurate output and follows the prescribed skill workflow.",
    "discoverability": "Activates the right skill when relevant; ignores unrelated skills.",
    "effectiveness": "Reaches the user's goal \u2014 measurably better with the skill than without.",
    "efficiency": "Completes the task with fewer tool calls and tokens.",
}

# Token-efficiency evaluator: derived locally from ``trial.tokens`` (the
# per-trial trajectory token counters). Per-trial sub-score uses a
# soft asymptote ``1 / (1 + total / TOKEN_EFFICIENCY_HALF_LIFE)`` over the
# sum of prompt + completion tokens (cached tokens are excluded since they
# are effectively pre-paid).  The half-life is the token count at which a
# trial scores 0.50; runs well under it score near 1.0, and runs many
# multiples above it tend toward 0.  Tunable in one place if cohorts of
# agents shift their typical usage envelope.
TOKEN_EFFICIENCY_HALF_LIFE: int = 200_000
TOKEN_EFFICIENCY_EVALUATOR_NAME: str = "token_efficiency"

# =============================================================================
# LLM-AS-JUDGE: TIER 3 DIMENSION + INSIGHTS
# =============================================================================

DIMENSION_JUDGE_MODEL = "gpt-5.4-mini"
DIMENSION_JUDGE_MAX_TOKENS = 2048
DIMENSION_JUDGE_TEMPERATURE: float | None = 0.0
DIMENSION_VERDICT_PASS_THRESHOLD = 0.7
DIMENSION_VERDICT_NEUTRAL_THRESHOLD = 0.4

# LLM-as-Judge for the Insights tab (additional Conclusions and Recommendations
# on top of the deterministic ones produced by tier3_normalizer).
INSIGHTS_JUDGE_MODEL = DIMENSION_JUDGE_MODEL
INSIGHTS_JUDGE_MAX_TOKENS = 2048
INSIGHTS_JUDGE_TEMPERATURE: float | None = None
INSIGHTS_JUDGE_MAX_CONCLUSIONS = 5
INSIGHTS_JUDGE_MAX_RECOMMENDATIONS = 5


class ExitCode:
    """Exit codes for CLI."""

    SUCCESS = 0
    VALIDATION_FAILED = 1
    CONFIG_ERROR = 2
    RUNTIME_ERROR = 3
