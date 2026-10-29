# Changelog

All notable changes to SkillEvaluator are documented in this file.

## Unreleased

### Added

- Reusable GitHub Actions onboarding for complete Tier 1 checks, with
  pinned runtime/workflow consumer templates, affected-skill PR selection,
  optional hosted reports and updated PR feedback. The new guide separates
  basic scanning from publisher setup; public image publication and generated
  release files require a reviewed manual release before consumer use.
- `--quality-config PATH` on `validate`, `quality-check`, and `tier1` loads the
  deterministic quality check's expected Markdown headings from a YAML file
  instead of the built-in list. Each `expected_headers` entry sets a `heading`
  and optional `alternatives`, `points` (default 5), `message`, and
  `suggestion`; omitted message and suggestion text is generated from the
  headings. Without the flag, the packaged `quality-default.yaml` reproduces
  the previous Prerequisites, Limitations, and Troubleshooting checks and
  wording, so default scores are unchanged. An invalid file (bad YAML or
  unparseable values, unknown keys, duplicate keys or headings, non-heading
  values) is rejected before validation starts with exit code 2 and a message
  naming the file and field, even when `--checks` leaves out `quality`. The JSON report, Markdown report,
  and BENCHMARK.md record which config produced the quality score as
  `quality_config` (`packaged` or `custom`, the custom file's name, a
  `sha256:` digest of the normalized rules, and the `--min-score` pass
  threshold, which is kept out of the digest).

### Changed

- HTML reports use a verdict-led layout with a sidebar and expandable check
  details. The verdict replaces the pass percentage; checks marked Not
  applicable are excluded from passed and executed counts. Publication status
  remains visible on mobile and in print, and blocking/advisory findings retain
  their separate counts.
- Dependency audits inspect pinned package versions without installing packages
  or executing build scripts. Unpinned, URL and other unresolved requirements
  report incomplete coverage instead of executing package resolution.
- The CLI reference now compares the direct `tier1`, `tier2`, and `tier3`
  workflows with `validate --tiers`: which tiers run, Tier 1 dependency and LLM
  stages, report files, and exit codes. It also states that `--tiers` must
  include Tier 1, so Tier 2 or Tier 3 alone runs through its direct workflow.
- README and onboarding docs now offer two first-run paths: a lightweight,
  keyless quality check on a bundled skill, or an NVIDIA Build rubric evaluation
  with eligible free prototyping access. Installation adds extras as needed,
  and the guides separate quality results from full scanner and live-agent
  evidence.

### Documentation

- Give the performance dashboard its own sidebar guide, retain the previous
  Reports & Results link, and explain matching-judge and activation limits on
  comparisons. Clarify the independent per-skill quality threshold, Windows
  Docker setup, and custom-backend and MCP environment controls.

### Fixed

- PII credential findings mask their values and source lines. GitHub CI stages
  scanner output privately, redacts recognized credential values from retained
  reports and logs, and withholds detailed output when redaction cannot finish.
- Local Claude Code trials retain Harbor's instruction environment variables so
  prompts reach `claude --print`. The exception accepts only the legacy
  `HARBOR_INSTRUCTION` name and Claude Code's UUID-named instruction variable;
  protected paths, loader controls, and unrelated Harbor variables retain their
  existing filtering ([#245](https://github.com/NVIDIA/SkillEvaluator/issues/245)).

- Tier 3 reports preserve unavailable evaluator lift as `null` when no baseline
  ran, rather than reporting a measured zero. CLI evaluator tables display
  missing or unavailable scores as `n/a` without interrupting JSON and HTML
  report generation; measured zeroes retain their numeric formatting
  ([#240](https://github.com/NVIDIA/SkillEvaluator/issues/240)).

- Explicit searches of `/` (`find`, `grep -r`, `rg`, `ag`, `du` and `tree` with a
  `/` operand, and native search tools rooted at `/`) now count as wasted tool
  calls whether or not they name the expected skill or script, including when
  output is redirected or the search runs inside `sh -c`. Host and bundled
  scoring agree, while scoped searches and slash-valued patterns retain their
  classification.
- PII email scans recognize all RFC 2606 reserved domains, including their
  subdomains, mixed-case forms, trailing dots, and prose ellipses. A local-part
  boundary prevents quadratic rescanning of long excluded addresses. Ordinary
  addresses on the same line and domains that merely resemble placeholders remain detectable
  ([#244](https://github.com/NVIDIA/SkillEvaluator/issues/244)).

- Tier 3 security grading preserves ordinary nested `/tmp` cleanup, including
  child operands with trailing slashes, unless the current or an earlier
  execution action contains recognized symlink-creation syntax (`ln -s`,
  `cp -s`, `os.symlink`, `Path.symlink_to`, literal Python subprocess argv
  launching `ln`/`cp`, or `fs.symlink`/`fs.symlinkSync`). With that evidence,
  both graders withhold nested-cleanup exemptions and report critical
  destructive-command findings with a security score of 0.0; unlinking a
  final leaf remains exempt. This conservative signal applies across the
  trace, including after link moves or replacements, without inspecting the
  filesystem or proving a link's destination or creation success. Unobserved
  or pre-existing symlinks are not detected by this policy. Lexical path escapes
  and unresolved or `xargs`-supplied deletion targets remain flagged. Searches,
  script arguments, and unexecuted heredoc data do not supply link evidence;
  Python and Node call syntax is inspected only in executable inline source
  or a recognized program supplied directly through standard input. Comments
  containing apostrophes do not hide actual link calls. Later and multiple
  heredocs, `cat` pipelines into interpreters, and common shell startup options
  retain link detection. Plain `;`, `&&`, `||`, and newline sequences apply
  evidence in command order, so later link creation does not regrade earlier
  cleanup in the same call. Loops, functions, traps, background jobs and other
  compound syntax keep conservative whole-action evidence; unrelated link
  creation can still affect later nested cleanup and remains a policy tradeoff.
- Tier 3 security grading no longer grants temporary-cleanup exemptions to
  destructive `rm` commands launched through `xargs`, whose additional targets
  may come from standard input or an argument file. Both graders recognize
  common options and wrappers while preserving separate cleanup commands.
  Input contents are not evaluated, so this conservative policy also flags
  legitimate scoped cleanup such as
  `find /tmp/cache -name '*.pyc' | xargs rm -f` as a critical destructive-command
  finding with a security score of 0.0.
- Tier 3 security grading now normalizes deletion targets before exempting
  temporary-directory cleanup. Paths such as `/tmp/../workspace` and commands
  mixing temporary and non-temporary targets receive destructive-command
  findings in both the host checker and bundled Harbor verifier. Destructive
  `rm` fragments with no parsed target are also flagged, preserving detection
  of deletions assembled from strings. Temporary cleanup inside backtick
  command substitutions, or followed by shell comments and file-descriptor
  redirections, no longer produces false destructive-command findings. Targets
  after heredoc and here-string redirections are still checked; the input word
  of a redirection is not treated as a deletion target.
  Case-arm/subshell terminators and unrelated comments, ANSI-C literals, or
  heredoc data no longer remove the cleanup exemption. Command substitutions
  used as redirection words do not hide later deletion targets. Unquoted
  `/tmp` deletion targets containing variables, command substitutions, or
  brace expansions (for example standalone `rm -rf /tmp/build-$$`) no longer
  get the cleanup exemption because their destinations remain unresolved.
  Operandless `xargs rm` invocations remain detectable when unrelated syntax
  or parser budgets prevent a complete parse; the fallback uses a bounded
  command span to avoid excessive backtracking on long actions.
- Tier 3 error recovery now scores unrecovered command failures as zero and
  includes them in fault counts and failure verdicts/report reasons. Host and
  bundled verifier detection agree and use a result's execution status when it
  has one: a Codex exit code, a whole `exit code N` line, or Harbor marking a
  Claude Code result as successful. Only results without a status, such as
  OpenCode's, are judged by their text: tracebacks, `error:` lines, pytest,
  jest, go test, tsc and npm failures and OpenCode's timeout note count, while
  `0 errors`, exception handlers, numeric timeout settings and type annotations
  do not. Exit-1 probes with no other output (`grep`, `rg`, `test`, `diff`,
  `cmp`, `command -v`, `which`, `git diff --quiet`, including `grep -c` zero
  counts, `/dev/null` redirects, a leading `cd` or `pwd` and pipelines) remain
  clean. Recovery matches whole command words after a leading `cd`; beyond the
  original five-call retry window, only a successful repeat of the same command
  and arguments counts.
- Catalog validation now includes nested and lowercase skill manifests, skips
  generated directories, and stops at skill boundaries to avoid validating
  bundled templates as separate skills. External linked directories containing
  skills and unreadable directories appear as failures without stopping other
  skills; links back into the catalog are ignored. Linked manifests retain
  direct-target behavior, plugin discovery errors produce findings, and reports
  and external Tier 3 results preserve catalog-relative paths for skills with
  the same basename.
- Changing finding severities through a policy no longer clears an independent
  quality-score threshold failure. A score below `--min-score` still fails,
  and its reason appears in the terminal, HTML, benchmark, and JSON reports.
- Tier 3 on Windows passes `PROGRAMFILES`, `PROGRAMW6432` and `PROGRAMDATA` to
  Harbor, so Docker Desktop finds its `compose` plugin, and passes
  `PYTHONUTF8` and `PYTHONIOENCODING` so Python UTF-8 mode reaches Harbor on
  non-UTF-8 locales.
- A `SKILL.md` `metadata` value that is not a mapping, such as `true`, `1.5`, a
  date, or a list or string containing `author` or `tags`, no longer crashes
  the quality check with a `TypeError`, which aborted `validate`, `tier1`, and
  `quality-check` before any report was written. The schema check reports the
  malformed `metadata`, and the quality score counts `metadata.author` and
  `metadata.tags` as missing, as it already did for other non-mapping values.

## 0.6.0 - 2026-10-20

### Added

- Custom execution backends. An installed package can add an `--env-mode`
  through the `skillevaluator.env_modes` entry-point group, naming an
  `EnvironmentBackendSpec` with a Harbor environment class plus the `--ek`
  kwargs, host variables, prerequisite hook, and optional agent wrappers that
  backend accepts. Custom execution backends follow the same kwarg and
  `runtime_env` policy as Harbor-native backends, may not forward agent,
  provider, or other backends' credentials, cannot replace a built-in mode or
  share a name with another installed package, are not named on published
  benchmark cards, and can be turned off with
  `SKILLEVALUATOR_DISABLE_CUSTOM_BACKENDS=1`.
- Optional `skillevaluator dashboard` tables for comparing saved Tier 3 reports
  and retained runs across skills, models, harnesses, recorded harness
  features, and skill conditions. Paired with/without differences cover score,
  pass@k, tokens, and agent execution time, plus USD cost when retained runs
  record it; the performance table also shows per-trial means alongside
  totals. Comparisons require matching complete trial coverage (repeated
  attempts pair by case, as in the HTML report) and name the reason when a
  pair is not comparable. Inputs are read once and reused between
  interactions (**Reload reports** re-reads them); browser uploads are limited
  to 16 MB. Install it with the `dashboard` extra.
- Tier 3 support for Harbor GKE execution mode (`--env-mode gke`) and `--ek`
  argument forwarding. API keys, bearer tokens, and passwords passed in `--ek`
  keys or values are rejected to keep credentials out of process listings, while
  configuration such as rate limits and token counts is preserved. Cluster
  infrastructure settings in skill configs are rejected to enforce security
  boundaries in favor of host environment variables and CLI flags.
- Claude Code live agent routing for Google Cloud Vertex AI
  (`CLAUDE_CODE_USE_VERTEX=1`, `ANTHROPIC_VERTEX_PROJECT_ID`, `CLOUD_ML_REGION`),
  with redirect-blocking preflight probes, case-insensitive model alias
  resolution, and explicit trusted-skill opt-in (`SKILLEVALUATOR_GKE_ALLOW_WORKLOAD_IDENTITY=1`
  or `--ek allow_workload_identity=true`) when using single-pod GKE Workload Identity.
- Fail-closed validation for skill-authored MCP server configurations
  (`evals/environment/mcp_servers.toml` and `mcp_servers.json`) that blocks
  operator/provider credential references and literal secrets, while allowing
  operator-approved MCP endpoints and non-LLM secrets via
  `SKILLEVALUATOR_ALLOWED_MCP_HOSTS` and `SKILLEVALUATOR_ALLOWED_MCP_SECRETS`.
- The HTML report's Tier 1 card now shows the SkillSpector version that ran
  the security scan (for example `SkillSpector v2.12.0`), and the embedded
  report JSON records it as `skillspector_versions`. Findings depend on the
  scanner release, so reviewers can tell which SkillSpector produced them
  without rerunning the scan. The version is shown even when the report omits
  its timestamp.
- `--env-mode kata` runs Tier 3 trials through the hardened Docker backend with
  every task service in a Kata Containers microVM. It requires a Linux host with
  `/dev/kvm` and a Kata runtime registered with Docker, and accepts only the
  `kata_runtime` and `kata_dns` environment kwargs. Image builds still run on
  the host, and tasks with Compose sidecars are rejected.
- `--env-mode runta` and `--env-mode mosaic` run Tier 3 trials on Harbor's Runta
  and Mosaic backends. Runta's `mode` and Mosaic's `volume`, `persist`,
  `enable_ssh`, `build_args`, and `build_target` kwargs are reserved because they
  bypass the task image or let state and access outlive one isolated trial.

### Changed

- SkillSpector now has its own default model per provider instead of
  following the chat default, so chat-default upgrades no longer reach
  SkillSpector before it supports them. An explicit `SKILL_EVAL_LLM_MODEL`
  still reaches it, and `SKILLSPECTOR_MODEL` still overrides both. The bridge
  forwards your `SKILLSPECTOR_MODEL_<SLOT>`, `SKILLSPECTOR_MODEL_REGISTRY` and
  `SKILLSPECTOR_REASONING_EFFORT` settings (never `SKILLSPECTOR_TEMPERATURE`),
  defaults the reasoning effort to `medium` for GPT-6 models, and records the
  requested `skillspector_model` and `skillspector_reasoning_effort` next to
  `skillspector_version` in the security result.
- Install instructions and the missing-scanner hint now pin SkillSpector to
  `v2.12.0` instead of its default branch.
- GPT-6 models get the GPT-5 request shape everywhere SkillEvaluator calls a
  model: no `temperature`, and `max_completion_tokens` on the native OpenAI
  endpoint. Gateway catalog IDs such as `aws/anthropic/bedrock-claude-opus-5`
  are recognized as the Claude model they name.
- Output-token limits are sized per model family. OpenAI reasoning models get
  at least 32,768 tokens; other models get 16,384 for `--llm-verify`, rubric
  evaluation, deduplication verdicts and judges, 8,192 for report suggestions
  and 16,000 for dataset generation. On the Anthropic API, Claude Opus 4 and
  4.1 stay at the 8,192 tokens the Anthropic SDK allows without streaming.
- Reasoning effort is sent only to GPT-6 models and to Claude 5.5 or later on
  the Anthropic API: `high` for `--llm-verify`, `medium` for rubric evaluation,
  deduplication verdicts and judges.
- Rubric evaluation and deduplication verdicts request structured output. The
  verifier and deduplication prompts ask for `rationale` instead of
  `reasoning`; replies that still say `reasoning` are accepted.
- The `goal_accuracy` RAGAS scorer is skipped for GPT-5 and GPT-6 judges, which
  reject its sampling settings.
- `run_config.json` records the judge reasoning effort, and each standard
  judge's details record `judge_model_used`, `judge_reasoning_effort` and
  `judge_fallback_used`. Skill lift leaves out a case whose two conditions were
  judged by different models and counts it in `judge_mixed_cases`. Dashboard
  score differences preserve this matching-judge lift, including unavailable
  comparisons.
- Removed the unused `CONTENT_DEDUP_LLM_*` constants and `LLM_VERIFY_MODEL`; an
  LLM client built from explicit credentials uses the provider default model.
- Tier 3 pins the agent CLIs Harbor installs: Codex 0.161.0 and Claude Code
  2.1.292, instead of the latest release. Claude Code runs Claude 5.5 and later
  models with `reasoning_effort=medium`; Codex keeps `high`. Local mode keeps
  your installed CLIs and fails a trial early when they are too old for the
  model: Claude 5.5 needs Claude Code 2.1.280 or newer, and GPT-6 needs Codex
  0.159.1 or newer.
- OpenAI-compatible gateway agent defaults: Codex moves to
  `openai/openai/gpt-6.1-sol` and Claude Code to
  `aws/anthropic/bedrock-claude-opus-5-5`. OpenCode stays on
  `nvidia/nvidia/nemotron-3-super-120b-long-ctx`, no longer tied to the gateway
  chat default. Claude Code on the shared gateway key runs with
  `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`, so gateways that forward to
  another provider do not reject its pre-release request fields.
- The OpenAI default model moves from `gpt-5.6-sol` to `gpt-6.1-sol` for chat,
  judging, Codex and SkillSpector, which runs it at `medium` reasoning effort,
  so the effort values above apply to the OpenAI defaults. OpenCode keeps
  `openai/gpt-5.6-sol` unless `SKILL_EVAL_LLM_MODEL` is set, because
  GPT-6.1 Sol supports tool calls only on the Responses API.
- The Anthropic and Bedrock default models move from Claude Opus 5 to Claude
  Opus 5.5 (`claude-opus-5-5` and `us.anthropic.claude-opus-5-5`) for chat,
  judging and Claude Code. On the Claude API they get the effort values above
  and server-side refusal fallback, and Claude Code runs them at
  `reasoning_effort=medium`.
  SkillSpector stays on `claude-opus-5` and `us.anthropic.claude-opus-5`, which
  SkillSpector v2.12.0 supports.
- Docker Claude Code and Codex containers no longer get the distribution
  `nodejs` and `npm` packages from agent setup, except on Alpine images. On
  Debian-based images, including the default eval image, those packages also
  provided `npx`, `gcc`, `g++`, and `make`. Claude Code containers now have no
  Node.js unless the image provides it, and Codex's own Node.js is on `PATH`
  only for the Codex process. Skills whose scripts, tasks, or stdio MCP servers
  need these tools must install them through `scripts/apt-packages.txt` or a
  custom `evals/environment/Dockerfile`.
- Exposed `kata`, `runta`, and `mosaic` as the 24th, 25th, and 26th Harbor 0.24
  backends alongside local mode. `cua-cloud`, `opensandbox`, `hf-sandbox`,
  `podman`, `prime`, and `smol` remain disabled.
- The public benchmark gate (`scripts/ci/check_public_benchmarks.py`) now
  requires a `PASS` card to record its `Source digest` and `Tier 3 run ID`.
  `PASS` cards rendered by earlier releases fail the scan until they are
  regenerated; earlier `FAIL`, `NEUTRAL`, and `INCOMPLETE` cards without those
  lines are still accepted, and the lines are validated whenever present.
- Report output must be outside the publication target. An explicit
  `--output-dir` inside the evaluated skill is now rejected; use an external
  directory. A default in-target `reports/` location relocates to an
  authenticated sibling `<skill>-reports` (or `<catalog>-reports`), refuses
  unowned collisions, and is excluded from Tier 3 runtime staging. Report roots
  inside the target repository or an included-skill root are excluded from
  linked or full-repository context only after they are reserved or
  authenticated as generated output, so an authored subtree is never silently
  omitted.
- A custom Tier 3 `--results-dir` inside the evaluated skill is now rejected;
  use the canonical `evals/results` path or an external results root.
- `validate --block-on-agent-eval` now passes the exit gate only for a completed
  Tier 3 `PASS` verdict with complete attempt coverage. `NEUTRAL` and `FAIL`
  verdicts exit 1; previously any completed, scored run passed.

### Fixed

- Tier 3 reports preserve judge-model exclusions when calculating dimension and
  overall lift. Native multi-step results also retain the judge provenance needed
  for those comparisons, so fallback judges cannot turn a regression into an
  apparent improvement. Judge-free cases retain valid lift, bounded report loads
  use the collector's recorded comparisons, and FINAL-strategy trials use only
  the final step's judges. The CLI displays unavailable comparisons as `n/a`.
- The LLM judge treats a schema option reported as `unavailable` as a schema
  rejection and retries without it, so OpenAI-compatible endpoints such as
  DeepSeek, which answer `response_format: json_schema` with HTTP 400 and
  "This response_format type is unavailable now", can score Tier 3 instead
  of failing every judge call (fixes #207).
- Tier 3 reports an inconclusive result when every observed positive with-skill
  trial has zero activation, and omits skill-lift comparisons for that agent.
  Activation counts are reported separately; negative controls and missing
  evidence do not imply that the skill was never loaded. The skill-execution
  score is zero when positive activation fails, so unrelated successful checks
  cannot inflate it to 0.5 (fixes #208). Benchmark cards and dashboard comparisons
  preserve this eligibility rule, and multi-agent HTML labels explain why lift
  is unavailable. Mixed-agent reports retain advice from the best eligible
  agent without attributing never-activated trials to skill quality.
- Tier 3 now uses one versioned, equal-weight five-dimension overall-score
  policy across Harbor rewards, pass@k, Skill Lift, reports, comparisons, and
  best-agent selection. Previously, some paths averaged all six evaluators
  directly while reports averaged five dimensions, which gave Effectiveness
  two votes and could reverse the reported lift direction. Current artifacts
  persist `skill-evaluator-dimension-mean-v1`; legacy and partial artifacts
  retain their historical semantics. Conditions keep their own recorded metric
  sets; overall lift and best-agent rankings are omitted across different
  standard metric sets while individual scores are preserved, and
  `tier3 compare` refuses standard agents with different policies or metric
  sets. Custom-only agents keep their grader's own overall and are ranked and
  compared against standard agents
  (Relates to [#61](https://github.com/NVIDIA/SkillEvaluator/issues/61)).
- Preserve explicit Claude and OpenCode read failure metadata in reconstructed
  ATIF trajectories. Failed isolated SKILL.md reads and failed native Skill
  calls lose credit; compound shell failures and shell reads whose path may
  expand to several files (braces, globs, `$` or backticks) remain ambiguous
  and retain prior credit in host and verifier grading. Unattributed errors do
  not cancel successful reads, and shell reads count dash-prefixed filenames
  after `--`.
- Tier 1 pyproject dependency audits now inspect static target declarations
  rather than the evaluator environment. Exact direct pins (including optional
  groups) are queried without installation or build-script execution;
  unresolved declarations and invalid scanner evidence are incomplete.
- Tier 1 PII scanning now includes `.mdc` rules and workflows.
- Tier 3 negative-case accuracy excludes inapplicable skill identification and
  records its scoring policy. Reports separately expose measured token and
  agent execution-time differences with complete matching-arm coverage, while
  efficiency continues to measure routing and productive tool use.
- Derive accuracy scores from complete criterion verdicts in both judge
  runtimes (passing applicable criteria divided by applicable criteria), so an
  inconsistent model-supplied total cannot distort Skill Lift. Correct the
  Harbor prompt example to score five passing criteria as 1.0.
- Include schema-valid ATIF message content parts in final answers, conversation
  summaries, and judge evidence references in the host and Harbor verifier.
  Image and audio parts use markers without exposing their source paths.
- Preserve scored Harbor trials when agent answers mention authentication errors
  and per-step token usage is present but aggregate ATIF token metrics are absent
  or zero. Explicit runtime exceptions and error events still invalidate trials.
- Safety-classifier refusals, content filters and output-token truncation are
  detected instead of being read as empty or malformed output. `--llm-verify`
  and rubric evaluation report the cause, and `--llm-verify` keeps the
  complete verdicts of a truncated reply. Claude 5.5 or later requests to the
  native Claude API use server-side fallback to Claude Opus 4.8. In the Harbor
  verifier, Bedrock retries a filtered Claude 5.5 reply on Claude Opus 4.8, and
  a refusal or empty judge reply moves on to the next
  `LLM_JUDGE_FALLBACK_MODELS` entry.
- The Harbor verifier's last-resort judge model follows the selected provider.
  With no judge or chat model configured, it used to request an OpenAI model
  from Anthropic, Bedrock, NVIDIA Build and gateways. The unused host-side
  `llm_judge.DEFAULT_JUDGE_MODEL` constant is removed.
- Fixed Docker Codex and Claude Code setup wasting its timeout installing
  distribution Node/npm before Harbor's own installers. Native, gateway Codex,
  and NVIDIA Build agents now provision Node/npm only on the installer branches
  that need them, while preserving installed-agent and version-pin checks.
- Preserved multiline skill descriptions when saving and loading Tier 2
  embedding catalogs. Descriptions can contain line breaks and tabs, while
  catalog identifiers and other control characters remain strictly validated.
- Brought the published documentation in line with 0.5.0. It now covers how
  standard grading builds judge evidence and rebuilds trajectories from
  OpenCode and Codex logs, structured judge output and retries, the evidence
  budget and judge time-budget variables, and the Harbor process environment.
  The CLI reference adds `validate --workers`, the full `--env-mode` list, and
  the exit codes the CLI actually returns (`0`, `1`, `2`).
- External scanners no longer crash the scan when they print bytes that the
  selected text codec cannot decode (for example a Windows traceback with a
  CJK path). Those bytes are replaced, so the captured output is never `None`,
  and valid output keeps the selected codec and universal-newline handling.
- The Tier 1 LLM rubric gives reasoning models (DeepSeek R and V4, OpenAI o1,
  o3, o4 and GPT-5 families, and models named "reasoning") a 32,768-token
  budget instead of the default 16,384, giving hidden reasoning more room
  before the rubric JSON is emitted. The budget follows the model each request
  resolves to, is capped at 16,384 on the Anthropic provider, and an explicit `max_tokens`
  overrides this default, subject to provider and model limits.
- `doctor --env-mode docker` and Tier 3 runs on Windows now find the Harbor
  CLI installed next to the active interpreter as `harbor.exe` or
  `harbor.cmd`, instead of reporting "harbor CLI not found".
- `BENCHMARK.md` publication verdicts now require completed Tier 1 and Tier 2
  execution evidence by default (Tier 1 must cover every default check that
  applies to the content type, so a run narrowed with `--checks` cannot
  certify publication), conservatively resolve conflicting peer
  policy and Tier 3 result metadata, bind every tier and policy claim to one
  versioned source-tree digest, preserve the Tier 3 run ID, persist an explicit
  publication status in JSON and HTML, and reject
  publication `PASS` cards whose decision evidence is missing, incomplete,
  linked, or hidden in raw HTML. Tier 3 evidence now cross-checks complete
  per-condition attempt and error records, and the public-card gate rejects
  ambiguous structural headings, browser-rendered decision-text aliases, and
  literal or entity-encoded Unicode control/format characters. The source
  identity is read without following links before and after each producer,
  follows the host filesystem's case aliases, and records a bounded
  source-change marker. Tier 3 runs that stage repository context, external
  reference or workspace skills, or task inputs with `.git`, `.venv`,
  `__pycache__`, or `node_modules` entries remain runnable but cannot certify
  publication, and neither can a target whose authored paths below the root
  use names that Tier 1 skips (such as `references/results/`).
- Clean and not-applicable License, Code Risk, Secrets, and Dependency Audit
  runs now record structured execution evidence, so legitimate built-in Tier 1
  results can satisfy publication without relying on findings or errors.
- Skip reasons are bounded, control-free, and escaped in reports and in the
  terminal, so oversized metadata, Rich markup, or terminal escape sequences
  cannot corrupt them. Reports show one line; the pipeline view keeps the
  reason's line breaks.
- SkillSpector issue paths with NUL bytes or lone surrogates now make the
  security scan incomplete instead of being accepted.
- The public benchmark gate rejects `PASS` cards with placeholder identities,
  malformed or future evaluation dates, a missing Tier 3 run ID, hostile
  benchmark-policy metadata, or environment-label substitutions in required
  proof fields.
- Hardened Tier 3 HTML and Markdown reporting against recursive or malformed
  display metadata, Markdown structure injection, and inline-JavaScript
  injection through untrusted agent or skill names.

### Security

- The docs-publishing workflow, the only one that passes a secret to a step,
  now pins `actions/checkout` to a commit SHA, declares
  `permissions: contents: read`, and sets `persist-credentials: false`.
  `dco.yml` no longer persists checkout credentials either.
- Docs are now published with the same Fern CLI version they are validated
  with. Both workflows install the CLI from `fern/package.json` and its
  committed lockfile; a test checks that the pin matches `fern/fern.config.json`.
- Workflow hardening guards now glob `.github/workflows/` instead of a
  hardcoded two-file list, so every workflow — including any added later — must
  pin each action to a commit, avoid persisting checkout credentials, and
  declare a `permissions:` block. The guards also cover job-level
  `permissions:` overrides and job-level reusable-workflow `uses:` references,
  neither of which the step-level checks reached.
- The Fern CLI is now installed from a committed lockfile, `fern/package-lock.json`,
  with `npm ci --prefix fern --ignore-scripts --omit=optional`, and both workflows
  invoke `./fern/node_modules/.bin/fern` rather than a binary on `PATH`. Previously
  `publish-docs.yml` ran `npm install`, which re-resolves every transitive
  dependency from semver ranges on each run: pinning `fern-api`'s own version
  pinned nothing beneath it. `@scarf/scarf` — reached through `fern-api`'s
  optional dependency on `@boundaryml/baml` — declares a `postinstall` script.
  The install step did not receive `FERN_TOKEN`, but its resulting CLI was later
  run in the token-bearing publish step. The lockfile pins every package to an
  exact version and integrity hash, `--ignore-scripts` stops lifecycle code
  running, and `--omit=optional` (which npm honours for this local install, but
  silently ignores for a `--global` one) leaves `fern-api` alone in the tree.
  Guards now permit only the reviewed lockfile install command in workflows,
  check the Fern manifest and lockfile dependency inputs, and require
  `fern/package.json`, `fern/package-lock.json`, and `fern/fern.config.json` to
  name the same CLI version. Fern dependency and config changes run the full
  CI lane, including dependency review, while `fern check` runs in both lanes.
  The developer-facing install instructions
  (`docs/README.md`, `docs/AGENTS.md`, `docs/developer-guide.mdx`) use the same
  lockfile install, so a contributor or agent following them gets the tree CI runs.
- The action-pinning guard now accepts a same-repo composite action or
  reusable workflow (`uses: ./...`) without requiring a commit SHA, since
  GitHub always resolves a local reference from the caller's own commit and
  nothing about it can float. A local reference that escapes the repository
  (e.g. `./../outside`) or carries an `@ref` (a syntax local references don't
  have) is still rejected, and a `docker://` reference is unaffected by the
  exemption.

## 0.5.0 - 2026-10-06

### Added

- Transparent HTTP 429 (rate-limiting), transient 5xx, and timeout recovery for
  LLM judges in both the Harbor container verifier (`eval.py`) and host runtime
  (`LLMClient`). Features zero-dependency full jitter exponential backoff,
  RFC-7231 `Retry-After` header parsing, finite-value environment overrides
  (`SKILL_EVAL_LLM_MAX_RETRIES`, `SKILL_EVAL_LLM_RETRY_BASE_DELAY`, and
  `SKILL_EVAL_LLM_RETRY_MAX_DELAY`), and
  automatic container forwarding via Harbor `task.toml`, and a per-judge
  verifier time budget that leaves room for failure artifacts, without altering
  benchmark metrics or scoring formulas.
- Provider-aware structured JSON schema enforcement (`response_format` for
  OpenAI-compatible / Gemini Vertex / NVIDIA NIM endpoints and `output_config`
  for Anthropic `/v1/messages`) across the custom `judge_accuracy`,
  `judge_goal_accuracy`, and `judge_behavior_check` paths, with automatic
  schema-specific `HTTP 400`/`422` downgrade and per-target memoization
  (`_SCHEMA_UNSUPPORTED_TARGETS`), boolean prompt alignment, and a guard for
  missing `message` fields on reasoning token exhaustion. The canonical OpenAI
  RAGAS goal scorer retains its separate scoring path.
- Configurable evidence bundle budgets (`SKILL_EVAL_ACCURACY_BUDGET`, `SKILL_EVAL_GOAL_ACCURACY_BUDGET`, `SKILL_EVAL_BEHAVIOR_CHECK_BUDGET`) and final response limit (`SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT`).
- Tier 3 log converters now rebuild ATIF trajectories from OpenCode JSON streams
  (`opencode.txt`) and structured Codex tee logs (`codex.txt`) when
  `trajectory.json` is missing or empty.

### Changed

- Upgraded the optional Tier 3 backend to Harbor 0.24.0 and its compatible
  LiteLLM 1.92-1.93 window. Existing SkillEvaluator agent and environment
  options now use Harbor's unified selectors; installed Codex adapters preserve
  merged user and MCP configuration; Docker and local execution stream redacted
  callbacks under a shared 16 MiB per-command output limit, while the parent
  Harbor orchestration process has a separate 16 MiB combined stdout/stderr
  limit; and Docker supports stdin plus isolated sidecar operations without
  exposing environment values on Compose argv. Generated schema 1.3 and
  unmodified native task schemas remain compatible, while collection accepts
  Harbor 0.24 job, trial, reward, and ATIF v1.8 artifacts. Multimodal tool
  output reaches judges and reports as text with `[image]`/`[audio]` markers.
  Native tasks' own test scripts must write numeric `reward.json` values, which
  Harbor 0.24 enforces; a failed job now names the trial or step exception that
  Harbor recorded.
- Exposed 23 Harbor 0.24 backends alongside local mode. `cua-cloud`,
  `opensandbox`, `hf-sandbox`, `podman`, `kata`, `runta`, `prime`, `mosaic`,
  and `smol` remain disabled until generated tasks can be projected through a
  trusted image or backend-native provisioning path. Non-secret backend
  constructor options can be supplied with repeatable, operator-only
  `--environment-kwarg` / `--ek` flags; skill-owned configuration,
  credentials, and sandbox-policy overrides remain outside that surface.
  Harbor's `stream` and `enable_environment_dir_upload` options are reserved,
  and Modal's `modal_sandbox_v2` option no longer exists.
- `--env-mode wandb` now runs Harbor's `cwsandbox` backend with W&B
  authentication, because Harbor 0.24 merged the two. Install
  `harbor[cwsandbox]==0.24.0` instead of the removed `harbor[wandb]` extra.
- Harbor now always starts in an empty, evaluator-owned working directory with
  `.env` loading and Harbor telemetry disabled. A `.env.local` or `.env` file
  near the operator's project can no longer add credentials or settings that
  SkillEvaluator's allowlisted environment removed. The `tier3` extra requires
  python-dotenv 1.2.0.
- Codex runs with `reasoning_effort=high`, the default under Harbor 0.22, so
  scores and cost stay comparable after Harbor stopped pinning it.
- Native tasks follow Harbor 0.24's separate-verifier precedence. A dedicated
  verifier image or build definition must ship its own test script and never
  receives SkillEvaluator's grader. Otherwise the verifier is built from
  `environment/`, whose Compose model must stay within the verifier's
  environment allowlist. `docker-compose.yml` verifier definitions, which
  Harbor ignores, are rejected.

### Fixed

- Report non-string YAML keys as validation errors in skills, rules, workflows,
  and plugin manifests instead of raising a `TypeError`, including frontmatter
  `metadata` keys. Show boolean, null, and date keys in a readable YAML form.
- Harbor ``result.json`` case ids now prefer canonical ``task_id.path`` metadata
  over repository-prefixed ``task_name`` values when resolving eval entries.
- Codex log synthesis maps ``web_search`` action payloads and ``collab_tool_call``
  thread items into ATIF, and error-recovery checks recognize ``status=failed`` /
  ``exit_code=`` terminal evidence emitted by Codex converters.
- Tier 3 Harbor dual-arm evaluation propagates arm suffixes (`-with-skill`,
  `-without-skill`) to `[task] name` in staged native `task.toml` files,
  normalizes external repository and namespace prefixes, and commutatively
  resolves canonical case IDs across attempt and arm suffix combinations
  while preserving expected case IDs.
- Tier 3 native-task collection now keeps Harbor's staged directory selector,
  logical dataset ID, and display name separate; ambiguous or unresolved
  persisted identities fail closed instead of trusting grader-authored IDs.
  Runner-owned attempt ordinals are carried structurally, so `attempt`-like
  text in authored selectors, logical IDs, or display names cannot corrupt
  pass@k or `stop_on_pass` accounting, including truncated aggregate names.
- Tier 3 Harbor subprocess, Docker, and local diagnostics now redact raw and
  percent-decoded URI/proxy userinfo components across streamed callbacks,
  nonzero exits, timeouts, output limits, and persisted launch errors.
- Tier 3 local OpenCode runs routed through NVIDIA Build now retain the rendered
  user instruction for ATIF conversion and fail on OpenCode error events, in
  parity with Harbor's upstream agent lifecycle. Remote MCP servers in that
  configuration disable OAuth, as Harbor's own OpenCode configuration does.
- Tier 3 reports now use the collector's logical attempt overall whenever a
  condition mixes standard and custom rewards, including across separate
  trials, and aggregate execution summaries preserve child-declared hidden
  error counts and truncation through launch-error overlays.
- Tier 3 now rejects non-finite, overflowing, and finite-but-unscalable timeout
  multipliers at YAML, programmatic, and Harbor command boundaries.
- Tier 3 now preserves paired baseline isolation by rejecting skill-owned
  pre-agent setup, native task/step healthchecks, native step workdir overlays,
  and Harbor task-shipped prior trajectories whenever the baseline arm is
  enabled. Native standard grading also fails closed on step test overlays,
  separate verifier contexts, post-agent collect hooks, and task-controlled
  verifier executable-path, shell, loader, proxy/TLS, provider, and judge
  environment controls; exact operator-staged provider/judge placeholders and
  unrelated verifier variables remain compatible. Operator-configured judge
  fallback models are forwarded only to standard verifier jobs, not agent or
  `custom_only` environments. Its Python payload now runs from a replaced
  evaluator-owned directory in isolated mode. `custom_only`
  retains Harbor-native collect hooks and Harbor's shared/separate
  step-test resolution, and fully authored native test paths are left untouched
  instead of replacing their unused `tests/skill_evaluator/` package.
  All native grading modes reject Windows agent or effective verifier
  environments until evaluator projection and verifier scripts are OS-aware.
- Tier 3 collection now fails closed on unsafe reward identities and malformed
  custom-metric contracts, publishes exact truncation metadata for bounded
  case and failure-detail samples, and keeps findings, attribution, and
  per-trial JSON inside the report loader's artifact envelope.
- Tier 3 progress output, diagnostics, and result summaries no longer redact
  counts such as `0/1 scored` or `1 errored` when a credential-named
  environment variable holds a short flag value such as `1`. Short credential
  fragments in proxy URLs are still redacted.
- `create-eval-dataset --refine` no longer requires the `tier3` extra to read
  case IDs from Harbor result files, and no longer fails on multimodal
  trajectory messages.

## 0.4.0 - 2026-09-30

### Fixed

- Render untrusted skill content, paths, tool messages, and LLM output literally
  in CLI reports and logs. Escape Rich markup and strip terminal control
  sequences to prevent rendering failures and misleading output, including
  catalog summaries and the compact validation view. Preserve Windows paths
  and literal emoji codes
  ([#173](https://github.com/NVIDIA/SkillEvaluator/pull/173),
  [#175](https://github.com/NVIDIA/SkillEvaluator/pull/175)).
- Make sensitive-assignment, JWT, and private-key redaction linear-time,
  preventing long adversarial text from stalling logs and reports. Apply the
  JWT fix to Tier 3 command output and the bundled Harbor verifier
  ([#172](https://github.com/NVIDIA/SkillEvaluator/pull/172),
  [#176](https://github.com/NVIDIA/SkillEvaluator/pull/176)).
- `--no-llm` full datasets include a negative bucket only when eval guidance
  supplies an off-skill prompt; template mode no longer guesses canned
  negatives from a fixed question list. CLI and docs now describe `--full` as
  up to four cases instead of always four.
- Treat `apply_patch` file headers (`*** Add File:`, `*** Update File:`, `*** Delete File:`,
  `*** Move to:`) as write targets in the Tier 3 security check. A patch that targets a shell
  profile, SSH, credential, or privileged config path, sent as an `apply_patch` tool call or a
  shell heredoc, is now a critical `sensitive_file_write` finding whose evidence names the
  protected path, not the patch. Every header in the patch is checked.
- Mask GitHub tokens (`ghp_`, `gho_`, `ghu_`, `ghs_`, `ghr_`, and `github_pat_`) in Tier 3
  evidence excerpts and Harbor verifier log output.
- Stop the PII scan reporting User-Agent product versions such as `Chrome/140.0.0.0`
  as public IP addresses. Chromium's reduced User-Agent gives every version this shape.
- Separate Tier 2 collection limits from the 256-file per-skill limit. Fresh
  similarity scans now allow 1,024 selected manifests and 128 million scalar
  comparisons by default, covering 343 skills with 2,048-dimensional embeddings.
  Add `--max-entries` and `--max-scalar-comparisons` for explicit scan budgets;
  exceeded limits fail with actionable errors and never truncate the collection.
  Fresh pairwise scans reject excessive scalar work after the first validated
  embedding response, before requesting more embeddings or saving a catalog.
  Collection discovery has its own 20,000-path ceiling, allowing the supported
  5,000-entry maximum for minimal collections while retaining the 4,096-path
  per-skill ceiling. Invalid Python command API budgets raise option-specific
  errors before provider initialization. Similarity comparisons validate and
  normalize each vector once instead of once per pair, making large pairwise
  scans more than an order of magnitude faster. Equal nonzero embeddings score
  exactly 1.0, so `--threshold 1` reports exact duplicates.
- Keep headings and comments inside fenced code examples in their enclosing Markdown
  section during Tier 2 content chunking, preserving original source line numbers.
- Run the public Docker image as an unprivileged user, with writable default report and home directories.
  Document UID/GID overrides for host-owned output mounts.
- Pin the HTML report Chart.js dependency and verify its integrity before browser execution.

- Preserve full Codex gateway model IDs in cloud environments, including E2B
  and Daytona, while retaining local runtime setup and native provider routing.
- Keep Tier 2 execution diagnostics visible alongside duplicate findings in
  CLI, HTML, and Markdown reports, without repeating the findings as errors.
- Use `nvidia/nemotron-3-super-120b-a12b` as the shared NVIDIA Build default for
  evaluator chat, agent execution, and judging, preserving explicit model overrides.
  Request nonstreaming chat responses explicitly to match the response parser.
- Tier 2 LLM failures now identify the selected provider and model, HTTP status,
  safe error metadata, and failed-cluster count without exposing response bodies.
- Replace the retired NVIDIA embedding default with `nvidia/nemotron-3-embed-1b`
  and send the required passage input type for NVIDIA document comparisons.
  Existing catalogs and caches must be rebuilt when switching embedding models.
- Report Tier 2 embedding and LLM service failures as incomplete checks, retaining
  a nonzero exit without inventing duplicate-content findings. Provider error
  messages include recovery guidance without echoing raw response bodies.
- Tier 3 script execution credit now requires evidence that the expected script
  was invoked. `check_script_execution` previously treated the script name as a
  substring of an execution command, so reading, printing or searching the
  script, or running a similarly named file, scored a full `Executed <script>`.
  Credit is now given only for a recognised invocation: the script run directly,
  an interpreter given it as its script argument, a `source`, or a `sh -c`
  payload that does one of those, with `cd` tracked and script identity compared
  exactly. Interpreter and wrapper options come from grammars derived by running
  each option against a script that records whether it executed, so `--help`,
  `perl -c` and `bash -n` run no script while `python -Wignore` and
  `env FOO=1` still resolve to theirs. A command the walk cannot resolve keeps
  the existing 0.75 partial credit rather than being scored either way: a path
  built at run time, an option outside a grammar, a name only in heredoc data,
  inline code or a module that names the script, and anything reaching a command
  through standard input, including `xargs` and `parallel`, whose behaviour the
  command text never determines. Partial credit is only ever given for a
  command that names the script: an unresolved command that never mentions it
  scores zero, as before. Redirections standing before the script
  (`python3 < /dev/null run.py`), a script's own arguments that look like shell
  options (`bash run.sh -c '...'`), invocations inside `if`, `while`, `until`
  and `for` bodies, and versioned interpreter names (`perl5.38.2`, `python3.13`)
  resolve as the shell runs them. A loop over an empty list keeps the earlier
  binding and its body is not read; a binding made inside `( ... )` stays
  there; the last command of a pipeline keeps its bindings where the shell
  does (zsh, ksh) and not where it forks it (bash, dash, mksh), with an
  option change that could move it (`shopt -s lastpipe`, `emulate sh`)
  read as unresolved; a quoted or escaped word that would read as syntax
  (`'done'`, `printf "("`, `';|'`) is the ordinary word it is; groups and
  compound pipeline stages nest in either order, each compound tested for its
  own pipe; `((` closed by `))` is an arithmetic command where the shell has
  one; the positional
  parameters are empty unless the text gives some, so `for f; do` at the top
  level runs nothing and keeps the variable's value; an interpreter fed its
  program by a pipe (`cat run.py | python3`) is unresolved like
  `python3 < run.py`; a variable bound by `export` or `readonly`, or by
  `declare` and `typeset` where the shell has them, holds the value it had
  when the builtin ran, `unset` empties it, `+x` and `export -n` unexport it
  however the builtin is reached, and one bound from data the text does not
  carry (`read`, `printf -v`, `local` outside a function, `declare -u`, a name
  `eval` may bind, including from a program held in a variable) is
  unresolved, as is a later assignment to a name given `-i`, `-u` or `-n`,
  while `-l` lowercases it and an array is never exported; what is done to a
  `-n` name (`nameref` in ksh and mksh) leaves the name it refers to
  unresolved; `let`, `$((...))` and `$[...]` assign as `((...))` does; an
  assignment the shell rejects (a value that is not a number for an `-i`
  name) and a special builtin given an option the shell rejects end the
  credit where the shell stops there; an assignment written before a command,
  `env NAME=value` included, is that command's environment only, reaching
  neither its own words nor the commands after it, except before a special
  builtin in the POSIX shells; a `-c` payload's shell sees the exported names,
  its command's own prefix and what `env` adds and removes, and a `$` quoted
  or escaped from the outer shell is expanded there (one left unquoted is
  expanded here, a name never bound to nothing), with the payload's own
  heredoc bodies kept as data; text that `eval`, inline
  code or a shell reading a heredoc may expand again is unresolved when a
  variable in it holds the script; every heredoc declared on a line takes its
  body after the line, in order, and past the number a shell accepts on one
  line (16 in bash) the line and the rest are data; a here-string, which dash
  and busybox ash reject before running anything, leaves nothing credited
  under those shells, and so does a compound's opening word written after an
  assignment (`A=1 for ...`, `A=1 if ...`, `A=1 ( ... )`), which every
  modelled shell rejects and which no longer raises; `(((` is a subshell
  around `((` in bash, zsh and mksh; and `ksh`, `mksh` and `ash` are
  recognised shells.
  Applied to both the host checker and the bundled Harbor verifier.

- Added `scripts/script_invocation_differential.py`, a differential harness that
  executes each command for real against fixtures that record whether they ran,
  and compares the result against both implementations. `--baseline REF` also
  scores every command with the checker at an earlier ref and lists each score
  that moved, so a change that lowers a command that ran, or raises one that
  did not, is seen before it is pushed.

### Added

- Interactive top-level help now opens with a green SkillEvaluator wordmark,
  installed version, and tier overview. Narrow terminals use a compact header;
  redirected output and subcommands keep their existing output format.

- OpenAI-compatible gateways now have chat, embedding, and separate Codex,
  Claude Code, and OpenCode model defaults. Set the provider, URL, and key;
  override model IDs when the gateway uses different catalog names. Claude
  Code inherits the gateway route unless an explicit Anthropic route is set.
  Run reports identify harness defaults separately from CLI/config overrides.

- Run individual tiers directly with `skillevaluator tier1 PATH`, `tier2 PATH`,
  and `tier3 PATH`, while retaining the expert subcommands. Tier 1 includes
  dependency checks and enables LLM checks when configured; Tier 2 reports
  whether a catalog comparison ran; Tier 3 creates a missing starter dataset
  and preserves existing evaluation sources.

### Changed

- Show elapsed waiting time during autopilot dataset generation and identify
  deterministic starter datasets used after a provider failure. Fully unscored
  Tier 3 runs show an `INCOMPLETE` summary with coverage, consolidated execution
  errors, and recovery steps; completed comparisons highlight overall Skill
  Lift ([#152](https://github.com/NVIDIA/SkillEvaluator/pull/152)).
- Harden documentation publishing with restricted token permissions, pinned
  checkout and Fern versions, and disabled persisted checkout credentials.
  Apply the checkout credential restriction to DCO checks
  ([#158](https://github.com/NVIDIA/SkillEvaluator/pull/158)).
- Add the methodology paper as the preferred citation and link research,
  developer-blog, and livestream resources from the README
  ([#146](https://github.com/NVIDIA/SkillEvaluator/pull/146),
  [#159](https://github.com/NVIDIA/SkillEvaluator/pull/159)).
- `validate PATH` now runs all three tiers for skills by default. Tier 3
  autopilot reuses an existing evaluation source or creates one starter case
  when none exists. `--full` remains compatible but is unnecessary;
  `--tiers`, `--no-tier3`, and `--no-autopilot` provide explicit scope controls.
  Keyless static CI gates should select `--tiers 1`.
- Tier 3 now selects a provider-native agent when `--agents` is omitted:
  OpenCode for NVIDIA Build, Codex for OpenAI, and Claude Code for Anthropic.
  NVIDIA Build agent runs default to Nemotron Super; explicit agent and model
  overrides remain unchanged.
- Tier 3's extra agent runtime preflight is now disabled by default because it
  executes the first real task prompt and incurs agent runtime and model cost.
  Enable it explicitly with `--agent-runtime-preflight` on either `tier3` or
  `validate`, or with `harbor.agent_runtime_preflight: true` in `evals/config.yml`.
- Missing-provider and API-key errors now show concise, copyable setup steps
  and a link to advanced configuration. Tier 1 also explains `--no-llm`.
- `tier1 validate` now runs only Tier 1, matching `tier1 PATH`. The top-level
  `validate` command retains its combined pipeline behavior.
- README and getting-started guides lead with provider-plus-key setup for
  NVIDIA Build and OpenAI, explain inherited model defaults, and separate
  credentials from scanner and agent runtime requirements.

## 0.3.0 - 2026-09-17

### Added

- Catalog validation now writes `catalog-summary.json` at the reports root with
  per-skill pass/fail status, optional severity rollups from child JSON reports,
  and paths to per-skill report directories.
- Catalog `validate` accepts `--workers N` to validate skills in parallel child
  processes (default 1 preserves the serial per-skill pipeline view).
- `SKILL_EVAL_MODEL_CATALOG_ALLOW_HTTP_HOSTS` names hosts whose model catalog may
  be read over plain HTTP. Catalog reads still require HTTPS for every other
  non-loopback host. Entries match one whole host as written, with no name
  resolution. A plain-HTTP request to an accepted host bypasses any inherited
  HTTP proxy so its bearer token is not offered to an intermediary. The
  transport rechecks authorization before dispatch and rejects hosts that
  are no longer allowed.
- Published benchmark cards record the evaluated source identity. `BENCHMARK.md`
  now carries `Evaluated source`, `Evaluated source revision` and
  `Evaluator container revision` as separate fields, so a reader can tell which
  source tree was evaluated apart from the evaluator build that evaluated it.
  Previously two skills evaluated from different repositories by the same
  evaluator container produced cards whose only recorded revision was the shared
  container tag. `validate` and `tier3 evaluate` take the identity as
  `--evaluated-source-repository`, `--evaluated-source-revision` and
  `--evaluator-container-revision`. `validate` records it on the card and in
  the top-level `evaluated_source` object of its JSON report and forwards it to
  every child of a parallel catalog run; both commands persist it into a Tier 3
  run's `run_config.json`. It can also arrive as the `evaluated_source`
  argument to `build_agent_eval_payload`, as an `evaluated_source` object in the
  run's `run_config.json`, or as `metadata["evaluated_source"]` on any
  validation result. It is never inferred from repository state while rendering,
  because the tree that renders a card is the evaluator checkout rather than the
  evaluated skill's source. Every populated carrier, including the payload of
  every Tier 3 result, is folded into one identity before any report is
  written, so carriers that disagree fail closed with nothing published instead
  of letting result ordering decide which source tree a card claims to describe.
  A revision is accepted only in an unambiguous shape: a full Git object id
  (40 or 64 hex characters), or a digest whose width matches the algorithm it
  names. A container revision is an image reference validated by component (a
  repository path of up to 255 characters, an optional tag of up to 128, and a
  digest at its algorithm's width), so a long repository name is no longer
  discarded. The 255 bound measures the path once the registry host is split
  off it. Path components are lower case, as the OCI grammar requires, while a
  registry host may use any case and is read as a host only when it is
  `localhost`, carries a dot, or carries a port.
  `check_public_benchmarks.py --require-source-provenance` requires the
  fields and fails any card publishing a `PASS` without them, including a
  `PASS` whose evaluator container is named by a mutable tag rather than
  pinned by digest. SkillEvaluator's own CI now runs the scan with that flag;
  it stays opt-in for trees whose cards predate the contract
  ([#72](https://github.com/NVIDIA/SkillEvaluator/issues/72)).
- SARIF 2.1.0 reporter (`-r sarif`) for GitHub Code Scanning and other SARIF
  consumers. Findings map to rule IDs, severity levels, and file locations from
  Tier 1 validation results.

### Fixed

- Fully covered documentation-only skills no longer fail security validation
  solely because non-applicable SkillSpector analyzers report a partial status
  ([#137](https://github.com/NVIDIA/SkillEvaluator/issues/137)).
- Embedding chunking now rejects zero-sized or non-progressing windows before
  entering the splitter or contacting the embedding provider
  ([#139](https://github.com/NVIDIA/SkillEvaluator/issues/139)).
- Scoped network exfiltration command flag patterns in security checks, enforcing command-position anchoring, quote-aware argument segmentation, explicit HTTP method flags, and case-sensitive `-F`/`-d`/`-T` flags to prevent false-positive flags on safe URLs, packages, or download scripts while reliably detecting quoted secrets and subshell wrappers.
- Malformed, non-UTF-8, or unreadable bundled and custom policy files now
  produce path-specific CLI errors instead of leaking raw parser or I/O errors
  ([#128](https://github.com/NVIDIA/SkillEvaluator/issues/128)).
- `create-eval-dataset --refine` resolves Harbor trial case ids from persisted
  `reward.json` `entry_id` metadata, using folder-name parsing only as an
  unambiguous legacy fallback.
- Tier 3 local mode now drops evaluator-managed empty process-loader resets
  while continuing to reject non-empty loader overrides, allowing generated
  tasks to reach agent execution
  ([#132](https://github.com/NVIDIA/SkillEvaluator/issues/132)).
- Unpinned-dependency warnings are no longer suppressed by comparison
  operators inside PEP 508 environment markers; requirements such as
  `pkg; python_version < "3.13"` are now correctly reported, while direct
  references are treated as pinned independently of marker contents.
- Schema, frontmatter, quality parsing, and security PII scanning accept a leading
  UTF-8 BOM, matching the unicode scanner's "benign BOM" note
  ([#91](https://github.com/NVIDIA/SkillEvaluator/issues/91)).
- SPDX headers keep the full license expression, so `MIT OR GPL-3.0` is
  no longer truncated to MIT and allowed. Closing comment markers such as
  `*/` and `-->` are not treated as part of the expression
  ([#86](https://github.com/NVIDIA/SkillEvaluator/issues/86)).
- Windows personal-path PII now flags `C:\Users\...` usernames that start with
  `s` (for example `steve`), matching the intended whitespace class rather than
  excluding the letter `s` ([#87](https://github.com/NVIDIA/SkillEvaluator/issues/87)).
- Quality scoring, script lint, and `create-eval-dataset` now treat `tools/`
  the same as `scripts/` for executable helpers.
- License detection no longer treats a frontmatter `license` identifier as
  authoritative when a LICENSE file declares a different license. Claiming
  MIT while shipping GPL-3.0 now fails closed. Every LICENSE/COPYING file is
  reconciled, NOTICE files stay informational, an unidentified license file is
  not treated as absent, and a blocking conflict no longer publishes
  `license_status=allowed`
  ([#85](https://github.com/NVIDIA/SkillEvaluator/issues/85)).
- `--llm-verify` now refuses to send file context from paths outside the
  skill root, including `..`, absolute paths, and outbound file symlinks.
- Gitleaks path allowlist now skips test/example/fixture/mock directories
  instead of any path containing those substrings, so files like `latest.py`
  are scanned.
- Gitleaks CI now limits pull-request and push scans to history reachable from
  the checked-out commit, while audit events retain all-ref coverage,
  preventing unrelated refs from causing false failures
  ([#106](https://github.com/NVIDIA/SkillEvaluator/pull/106)).
- The Tier 3 agent runtime preflight now fails with an actionable diagnostic when
  the results directory is not visible to the Docker daemon. Previously the smoke
  run passed -- agent output travels over the Docker exec API rather than through
  the mounts -- and every scored trial then failed with `RewardFileNotFoundError`
  while the rewards sat inside the daemon's own filesystem.
- Dead-link validation now uses the shared CommonMark parser, covering
  reference-style and HTML links while preserving Markdown image checks and
  consistently normalizing local destinations. Root-absolute URLs are ignored
  instead of being treated as host paths; href-only diagnostics collapse
  repeated links to the same normalized target. Invalid destination bytes do
  not alias other files, relative URLs that normalize to absolute or
  drive-relative paths are reported without lookup, lookup failures remain
  per-link findings, and link diagnostics are bounded and escaped. Malformed
  frontmatter and repeated unclosed HTML comments no longer abort or stall
  supporting-document checks.
- Tier 3 accuracy and custom goal judges now retry one malformed (including
  empty) or schema-invalid response with a 4096-token output budget before
  failing closed, preventing a transient formatting error from making an otherwise
  successful trial and its full comparison arm unscoreable. Generated and
  injected Harbor verifier configs now reserve 600 seconds for six sequential
  direct provider attempts plus fail-closed artifact writes. Explicit native
  task timeouts remain owner-controlled and are not rewritten, and whole jobs
  defer to Harbor's task-configured phase controls instead of a hidden two-hour
  cap
  ([#70](https://github.com/NVIDIA/SkillEvaluator/issues/70)).
- PII scanning no longer treats Markdown ATX headings as code comments, so
  emails in headings such as `# Contact: ...` are flagged. Hash lines inside
  Python strings, YAML scalars, and shell heredocs are scanned too. Real
  comments stay skipped, including YAML frontmatter, fenced code, and
  `requirements.txt` ([#88](https://github.com/NVIDIA/SkillEvaluator/issues/88)).
- Tier 3 Harbor collection no longer scans an agent's unstructured transcript
  for runtime-error phrases when the recorded exception belongs to the
  verifier, health check, or task. Correct answers that discuss errors such as
  `401 Unauthorized` are no longer misreported as agent runtime failures.
- SkillSpector reports now use validated version-specific completeness
  contracts. Valid findings from coherent 2.10+ partial scans remain visible
  while the result stays incomplete, and fully covered 2.9.5/2.9.6 `--no-llm`
  reports remain compatible. Contradictory finding or component totals and
  duplicate component identities fail closed. Versioned findings require
  producer paths, and complete reports reconcile universal analyzer work with
  the component inventory. Reports scored before 2.10 finding compaction remain
  accepted. Shipped bytecode findings, source-scoped executable evidence, and
  version-specific finding identities remain authoritative without overstating
  compacted or hidden finding evidence. SkillSpector 2.11+ requires bundled
  execution-surface analyzer evidence; 2.11.1+ uses classification-aware
  finding IDs while rejecting conflicting reuse of an ID.
- Tier 3 paired pass@k evidence now respects Python's active integer-string
  conversion limit, preserves nonzero Wilson interval widths and paired-effect
  directions at large case counts, and documents exact-rational omission
  markers.
- Tier 3 now decodes bounded native Codex `exec` wrappers into their static
  tool calls. It preserves call order and outer-call provenance, maps an outer
  observation only when its rendered inner call is known, keeps ambiguous
  observations explicit, and reports unsupported or malformed JavaScript as
  untrusted instead of a clean security result.

## 0.2.1 - 2026-08-24

### Added

- Added a public benchmark publication gate, regression coverage, and a
  documented rollout plan for generated `BENCHMARK.md` cards.
- Tier 3 pass@k results now include per-arm 95% Wilson score intervals and,
  when case identities pair completely, direction-preserving paired outcomes
  with a two-sided exact McNemar diagnostic, its attainable-p resolution limit,
  and the paired pass-rate delta.

### Changed

- Unified Tier 3 scoring around the canonical five dimensions, persisted an
  immutable dataset-truth snapshot with provenance metadata, and redesigned
  `BENCHMARK.md` as a decision-first publication card.
- Updated public OpenAI / Anthropic / Bedrock chat defaults to pinned frontier
  models (`gpt-5.6-sol`, `claude-opus-5`, `us.anthropic.claude-opus-5`),
  centralized in `provider_config`, and documented `gpt-5.4-mini` as the
  lower-cost OpenAI `SKILL_EVAL_LLM_MODEL` alternative. Raised
  dimension/insights judge token budgets to 4096 and widened the gpt-5\*
  temperature guard to bare model IDs.

### Fixed

- Tier 3 now exercises each resolved agent route and the enabled standard-
  grading route against its provider's model catalog before image preparation
  or task staging. Definitive native-provider authentication and deterministic
  Bedrock credential/configuration failures stop immediately with a redacted
  diagnostic. Non-authoritative OpenAI catalog permission/membership results,
  public or compatible catalog success that does not authenticate inference,
  compatible-gateway catalog authentication, transient failures, and native
  Harbor judge selection resolved only at runtime continue as degraded checks.
  Redacted per-route outcomes are retained even when the later agent runtime
  preflight fails ([#71](https://github.com/NVIDIA/SkillEvaluator/issues/71)).
- Tier 3 Harbor collection now accepts the `step_results: null` sentinel
  emitted for successful single-step trials while retaining fail-closed
  validation for malformed non-null multi-step result containers.

- Tier 3 eval-dataset generation now parses `SKILL.md` frontmatter as YAML.
  The previous line-based scan captured block-scalar indicators verbatim, so a
  `description: >-` became the literal string `>-` in every generated prompt,
  and multi-line quoted scalars were silently truncated to their first line.

- GitHub Actions pull request reports now link source targets to the checked-out
  repository revision instead of the synthetic `<number>/merge` ref, preventing
  broken or cross-repository links.
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

## 0.2.0 - 2026-08-18

### Security

- Secure Docker exec redaction now ignores environment values shorter than eight
  characters, matching the exact secret length floor used elsewhere. Short
  flags such as `CLAUDE_CODE_DISABLE_POLICY_SKILLS=1` no longer rewrite digits
  in `docker exec` output, which had broken NVIDIA Build bridge loopback
  origins during Tier 3 preflight.

### Changed

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

- Quality scoring now uses boundary-aware and context-aware matching for XML tags,
  reserved names, MCP guidance, README references, time references, exclusivity
  language, instruction action verbs, and nested Markdown links, avoiding
  incidental-word score changes.
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
