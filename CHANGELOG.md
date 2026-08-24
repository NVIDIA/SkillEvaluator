# Changelog

All notable changes to SkillEvaluator are documented in this file.

## Unreleased

### Added

- Added a public benchmark publication gate, regression coverage, and a
  documented rollout plan for generated `BENCHMARK.md` cards.

### Changed

- Unified Tier 3 scoring around the canonical five dimensions, persisted an
  immutable dataset-truth snapshot with provenance metadata, and redesigned
  `BENCHMARK.md` as a decision-first publication card.

### Fixed

- Tier 3 eval-dataset generation now parses `SKILL.md` frontmatter as YAML.
  The previous line-based scan captured block-scalar indicators verbatim, so a
  `description: >-` became the literal string `>-` in every generated prompt,
  and multi-line quoted scalars were silently truncated to their first line.

- GitHub Actions pull request reports now link source targets to the checked-out
  repository revision instead of the synthetic `<number>/merge` ref, preventing
  broken or cross-repository links.

## 0.2.0 - 2026-08-18

### Security

- Secure Docker exec redaction now ignores environment values shorter than eight
  characters, matching the exact secret length floor used elsewhere. Short
  flags such as `CLAUDE_CODE_DISABLE_POLICY_SKILLS=1` no longer rewrite digits
  in `docker exec` output, which had broken NVIDIA Build bridge loopback
  origins during Tier 3 preflight.

### Changed

- Updated public OpenAI / Anthropic / Bedrock chat defaults to pinned frontier
  models (`gpt-5.6-sol`, `claude-opus-5`, `us.anthropic.claude-opus-5`),
  centralized in `provider_config`, and documented `gpt-5.4-mini` as the
  lower-cost OpenAI `SKILL_EVAL_LLM_MODEL` alternative. Raised
  dimension/insights judge token budgets to 4096 and widened the gpt-5\*
  temperature guard to bare model IDs.
- Added explicit `--block-on-dedup` / `--no-block-on-dedup` and
  `--block-on-agent-eval` / `--no-block-on-agent-eval` controls with
  backward-compatible defaults, Tier 3 source preflight, and consistent gating
  metadata across CLI, JSON, Markdown, and HTML reports.
- Reduced pull-request runner use for changes confined to `docs/**` and
  `fern/**`: DCO, Gitleaks, and pinned Fern validation still run, while mixed
  and non-docs changes retain the complete Linux, macOS, Windows, packaging,
  and security matrix. Superseded pull-request CI and security runs are
  cancelled so they do not consume runners after a newer commit is pushed.
  Path classification executes from the pull request base revision so a
  change cannot weaken its own CI routing.

### Fixed

- Tier 3 LLM insights now receive explicit labels and bounded expected-behavior
  context for `expected_skill: null` negative controls, and the judge is
  instructed not to flag unrelated successes without invocation or
  failed-routing evidence.
- Fixed Anthropic API-root normalization across evaluator and Claude Code
  paths, and made required Tier 3 judge failures fail closed instead of
  appearing as numeric zero scores or publishing misleading quality results
  ([#55](https://github.com/NVIDIA/SkillEvaluator/issues/55)).
- Tier 3 now normalizes host-configured `LLM_JUDGE_MODEL` and
  `SKILL_EVAL_JUDGE_MODEL` overrides in Harbor's parent process and forwards
  the selected value through its verifier-only job layer for standard grading.
  This lets native separate-verifier placeholders resolve without injecting
  either name into the evaluated agent's initial environment. Skill-authored
  `runtime_env` and native task `[environment.env]` tables cannot set or alias
  either operator-controlled override.
  Native verifier declarations remain compatible, while the job-level value
  takes precedence during standard grading. Tier 3 results now record the
  configured judge provider, model, source, and whether a dedicated job-wide
  override was applied, separately from agent models. A provider fallback may
  still use a different model for an individual judge call.
- Quality scoring now uses boundary-aware lexical matching and CommonMark-parsed
  structural links instead of hand-written Markdown parsing or regex inference
  of author intent. Deterministic checks no longer infer MCP negation, temporal
  intent, README guidance, or exclusivity from prose;
  use `rubric-eval` for semantic documentation judgments and Tier 3 for
  observed agent behavior.
- Tier 3 generated tasks now stage only an entry's declared `files`, preventing
  undeclared fixtures from the shared `evals/files/` directory from appearing
  in that task's `/workspace/input/`, while preserving copy-all behavior for
  legacy entries that omit the field. Agent-visible target, reference, and
  workspace skill projections now omit evaluator-owned `evals/` directories
  from every staged skill package, including sanitized `--copy-repo` contexts,
  while graders, native tasks, custom
  environments, and declared inputs continue to load from the source dataset.
  Authenticated historical result trees are also excluded after output rotation,
  invalid markers fail closed, and late Codex, Cline, Goose, and Qwen
  skill-discovery roots are reset before agent execution. Pre-upgrade custom
  result roots outside `evals/` have no authenticity marker and cannot be
  distinguished safely from authored runtime content. Move or delete that old
  content before `--copy-repo` or other full-context evaluation, then rerun with
  this version if replacement evidence is needed. Explicit task inputs cannot
  select evaluator-owned datasets, configuration, graders, tests, native tasks,
  environments, or results. Every agent and baseline arm now reads from one
  private, selective evaluator snapshot containing the active control files,
  task-source data, consumed fixtures and grader, and the complete authored
  custom environment. Legacy omitted-file entries retain the full shared files
  corpus. Unrelated evaluator subtrees and generated results stay outside the
  snapshot. MCP configuration and completed-run artifacts are
  read through bounded descriptor-anchored roots; on Windows, selected file
  handles deny concurrent writes and deletes while live. Historical unmarked
  runs created before canonical run-level `result.json` remain discoverable only
  when their stable configuration and summaries satisfy the complete historical
  schema. Pre-status scored summaries remain consumable, coherent status-era
  failures remain visible without contributing scores, and marked current
  partial runs continue to fail closed.
- Tier 2 scans now validate but do not follow the exact contained
  `CLAUDE.md -> AGENTS.md` compatibility alias, scanning the exactly named,
  independently discovered, single-link regular target once while continuing
  to reject hard-linked selected files, linked manifests, directories, and all
  other file redirects.

## 0.1.0 - 2026-08-05

### Added

- CI DCO check that fails pull requests whose commits lack a `Signed-off-by`
  trailer, matching the sign-off requirement in `CONTRIBUTING.md`.
- Initial public release candidate.
- Enabled optional semantic-version validation in the default Tier 1 pipeline,
  including a public `--previous-version` monotonic-bump bound.

- Added NVIDIA Build live-agent paths: direct OpenCode support plus Docker
  compatibility bridges for Codex and experimental Claude Code, including
  multi-turn tool-call continuation.
- Fern documentation site configured for `docs.nvidia.com/skills/skillevaluator`,
  building the `docs/` guides (installation, configuration, and the three
  evaluation tiers) as MDX pages.
- Expanded the documentation site to fifteen pages — quickstart, eval
  datasets, agents and sandboxes, custom graders, reports, CI integration,
  CLI reference, and environment variables — under a task-oriented
  navigation, with every command verified against the current CLI.

### Security

- Isolated NVIDIA Build bridge credentials from vendor CLI processes using a
  transient, root-managed, container-only key handoff with cleanup on failure.
- Removed NVIDIA Build secrets from Harbor and Docker exec arguments using a
  host-only key file, a non-secret subprocess sentinel, and per-exec container
  handoffs; provider-secret aliases in `runtime_env` are rejected.
- Hardened compatibility-bridge startup with a dynamic loopback port and
  authenticated, process-bound readiness instead of a fixed health endpoint.
- Tightened local macOS Seatbelt policy so nested workspaces can traverse home
  directory metadata without gaining directory-listing or sibling-file access.
- Removed implicit host-side pytest execution from default Tier 1
  code-integrity validation. Test evidence is now collected with contained,
  filename-only discovery that does not import or execute target-controlled
  Python code.

### Changed

- Simplified the repository README into a concise documentation landing page,
  retained a compact keyless `validate` quickstart, LLM-provider setup, and a
  one-command `validate --full` path through all three tiers, broadened the
  project description to agent artifacts starting with agent skills, and moved
  detailed guidance to `docs.nvidia.com/skills/skillevaluator`.
- Added Tier 3 cost-planning guidance, including trial-volume multipliers,
  cost-saving flags, and the cost and isolation tradeoffs of local mode.
- Standardized the product name as `SkillEvaluator` across documentation,
  repository metadata, CLI output, and generated report artifacts.
- Removed the optional OpenTelemetry integration, the
  `skillevaluator[telemetry]` extra, and the `skillevaluator.telemetry` Python
  module from the public distribution. Imports of that module now fail rather
  than providing the former telemetry and safety helpers. Redaction and
  child-process environment filtering remain available from
  `skillevaluator.utils.redaction` and
  `skillevaluator.utils.process_environment`; direct Protobuf and OpenTelemetry
  dependencies are no longer installed.
- Changed the public OpenAI default to `gpt-5.4-mini` and the NVIDIA Build
  default to `nvidia/nemotron-3-nano-30b-a3b`; OpenCode, Codex, and experimental
  Claude Code now resolve that Build default without redundant model flags.
- Tier 3 now streams staging, arm submission, completion, failure, collection,
  and report-writing progress instead of appearing idle during Harbor startup.
- Tier 3 now reports structured agent/provider failures such as NVIDIA Build
  capacity exhaustion instead of scoring a no-trajectory fallback or emitting
  a generated Harbor task-name mismatch.
- Replaced provisional `test_coverage`, `tests`, and `coverage_percent` output
  with one `test_discovery` detail. Reports now include `test_count`, supported
  filename patterns, `execution_performed=false`, and
  `coverage_measured=false`; projects must run tests and measure coverage in a
  trusted environment or explicit sandbox.

### Fixed

- Public benchmark cards now omit policy profiles, redact absolute host paths,
  and normalize imported internal or retired metadata before publication.
- Previous-version validation now rejects catalog-wide scalar reuse and removal
  of an already bounded `metadata.version` label.
- Tier 1 and Tier 2 now ignore only the exact public SPDX metadata preamble,
  distinguish package versions from network addresses, recognize canonical
  `agents/` and `tests/` support directories, and keep Ruff on the validated
  0.15 release line.
- Accepted structurally complete SkillSpector finding reports on policy exit 1
  and hardened validation of the external scanner's untrusted JSON contract;
  SkillSpector remains separately installed and unpinned by this distribution.
- Programmatic dataset generation now returns explicit created, preview, and
  unchanged outcomes, preserves actionable failures, and no longer mutates
  process-wide command-line arguments.
- Security and full-feature installs now work on RHEL 8 and other glibc 2.28
  Linux systems by keeping Semgrep and SkillSpector in separate tool
  environments while retaining compatible bundled Python dependencies.
- Tier 2 content collection now prunes configured evaluation and version
  artifact directories before enforcing the discovered-path limit, so excluded
  generated results cannot cause false path-count failures.
