# Changelog

All notable changes to SkillEvaluator are documented in this file.

## Unreleased

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
