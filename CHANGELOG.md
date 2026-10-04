# Changelog

All notable changes to SkillEvaluator are documented in this file.

## Unreleased

### Added

- Added public plugin evaluation across all tiers: static schema and MCP checks,
  advisory offline dependency/context deduplication, and Harbor-backed live
  evaluation with effectiveness and optional sum-of-parts Integration arms.
- Tier 3 reports paired case-bootstrap 95% intervals for Skill and Integration
  lift, pass^k, cost per passed case, token efficiency, and measured context
  cost as report-only statistics. Integration becomes inconclusive when its
  interval includes zero, too few cases pair, or per-case coverage is incomplete.
- Plugin Tier 3 runs record report-only per-trial and per-arm component signals
  (activations, tool selection, arguments, MCP outcomes, order, handoffs,
  conflicts, coverage), driven by optional advisory dataset fields.
- Plugin Tier 2 compares bundled skills and the plugin with a local catalog
  (`tier2 PLUGIN --catalog FILE`; optional `--llm` verdict); save plugin catalogs
  with `similarity-check PLUGINS --type plugin --save-catalog FILE`.
- Tier 1 classifies each plugin skill/rule reference offline as provided,
  referenced, missing, external, or unresolved, and blocks on a missing
  same-repository dependency; `validate --repo-root` sets the repository root.
- Tier 1 runs the version, quality, and lint checks on each skill bundled in a
  plugin, attributing findings to that skill.
- Plugin MCP servers are accepted in every `mcpServers` form (inline map, `.json`
  path, or array) and from the root `.mcp.json`, in Tier 1 and Tier 3 staging.
- Plugin validation reports a component inventory, MCP pinning, and a static
  context-cost estimate; Tier 3 provenance adds report-only component coverage.
- Validation policies accept `mcp.allowed_private_hosts` for intended private
  MCP endpoints.
- Plugin reports: JSON, Markdown, SARIF, and HTML render plugin dependencies,
  components, MCP pinning, and static context cost; Tier 3 reports add component
  coverage, Integration (including INCONCLUSIVE), lift intervals, and advisory
  plugin signals.
- `validate` writes a plugin `BENCHMARK.md` card with component coverage,
  Effectiveness and Integration results, and the behavior the run excluded.
- Added a Plugin Evaluation docs page covering all three tiers, the plugin
  dataset fields, verdicts, and current limitations.
- Review plugin subagent and command privileges in Tier 1: unrestricted `Bash` and `bypassPermissions` are HIGH; wildcard tools, `acceptEdits`, and subagents that inherit every tool next to write-capable MCP servers are MEDIUM.
- Add a static hook risk model for plugins: per-handler event, matcher, handler type, and risk flags, with findings for broad auto-approval, remote code execution, HTTP endpoints (`hooks.allowed_urls` policy allowlist, compared as parsed URLs), context injection, and files outside the plugin root. Matchers are classified without a backtracking regex engine, flat hooks (Cursor and Agent Plugins client extensions) get the same checks, hook scripts are read from their first 64 KiB even when larger or not UTF-8, and a script that cannot be analyzed (`plugin_hook_script_unanalyzed`) or a hooks source past the scan limits (`plugin_hook_scan_truncated`) is a HIGH finding.
- Extend the opt-in `dependency` check for plugins to npm lockfiles and `package.json` and to container images (MCP `docker`/`podman` commands from every manifest the plugin ships, and Dockerfiles), with OSV-Scanner, `npm audit --package-lock-only`, Grype, or Trivy. The plugin audit never passes silently: no scanner, a failed scanner run (pip-audit included), or a manifest, lockfile, or Dockerfile that cannot be read within its bounds makes the check `INCOMPLETE`.
- Add opt-in `validate --resolve-endpoints` (policy `endpoints.resolve`) DNS and single-`HEAD` redirect checks for MCP (including servers that only an additional manifest declares) and HTTP hook URLs that reach private, link-local, or metadata addresses. Hosts with an unexpanded `${VAR}` are skipped.
- Add the opt-in `claude-validate` Tier 1 check, which compares SkillEvaluator's plugin verdict with `claude plugin validate --strict` when the Claude Code CLI is installed.
- Render plugin privileges, hook risk, the CVE audit summary, validator parity, and endpoint checks in the JSON, Markdown, HTML, and CLI reports.
- Plugin evaluation supports Codex (`.codex-plugin/plugin.json`), Cursor (`.cursor-plugin/plugin.json`),
  and Agent Plugins v1 (root `plugin.json`) manifests in every tier, with deterministic manifest precedence,
  per-format required-field checks, component mapping (Codex apps and Agent Plugins extensions are
  inventoried as unsupported), and MEDIUM `plugin_manifest_conflict` findings when a plugin's manifests
  disagree on name or version. Codex `http_headers` are checked together with `headers`, a root
  `plugin.json` that is not UTF-8 or exceeds the 1 MiB manifest limit no longer fails discovery, and an
  additional manifest with such content is a MEDIUM `plugin_manifest_additional_invalid`.
- Add `--plugin-load {wrapper,native,auto}` to `tier3 evaluate-plugin` and `validate --tier3` (default `wrapper`); `native` stages the with-plugin arm the way Claude Code, Codex, OpenCode, or Hermes load plugins, while the member-skills and no-plugin arms stay unchanged.
- Record `plugin_load` (requested mode and per-agent mode, reason, adapter, and component modes) in plugin provenance, and render a Plugin Loading section in the HTML, Markdown, and CLI reports.
- Wrap natively staged plugin hooks with the hook census logger (`/skilleval/hook_census.sh`) using the Tier 1 `hook_risk` ids.
- Write a per-trial load census (`/logs/agent/skilleval-load-census.json`) under native loading, collect it per agent, and mark verified components `loaded` in component coverage.
- Support `hermes` as an experimental Tier 3 agent in Docker and cloud environments with the Anthropic provider. The OpenAI and OpenAI-compatible providers are refused for Hermes, because Hermes routes `openai/MODEL` to OpenRouter and would send the key there.
- Plugin Tier 3 collects a hook execution census (`hook_census.sh`) into
  `plugin_signals.hook_census` and marks natively staged hooks that ran as
  `exercised` in component coverage.
- Plugin subagents and commands observed at runtime count toward activation
  coverage and are marked `exercised` in component coverage when available.
- Plugin Tier 3 plants a random canary credential in every arm's workspace and
  scores its exfiltration as a critical `canary_exfiltration` security finding,
  with per-arm canary results. Shell commands are checked per statement,
  including `sh -c`/`bash -lc`/`eval` payloads, heredocs, and argv-list
  commands, with taint tracking across statements; URLs count only for
  network, MCP, and command tools.
- `tier3 evaluate-plugin` and `validate` accept `--probe-mcp` to probe
  author-supplied URL MCP servers from the host (advisory `mcp_proof`). The
  probe sends only literal declared headers; `--probe-mcp-env NAME` opts one
  host variable in to `${NAME}` header expansion.
- Plugin reports render the hook census, canary results, and MCP proof.
- `harbor.plugin_canary: false` in `evals/config.yml` turns off the canary decoy for a plugin run, and the run config records the choice.
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

### Changed

- Tier 3 `accuracy`, `goal_accuracy`, and `behavior_check` are now not
  applicable (N/A) instead of a fabricated 1.0 when an eval case has no
  `ground_truth` or `expected_behavior`; N/A metrics are left out of overall
  scores, averages, and lift, and render as N/A.
- `plugin_agent_unrestricted_bash` is now LOW (advisory) and also covers subagents
  that omit `tools`: a subagent's tools list limits it but never pre-approves Bash.
- Tier 3 judges now see what the agent wrote. Write calls (write and edit tools,
  `apply_patch` with OpenCode's `patchText` or Codex's `input`, Hermes `patch`,
  and shell writes, including Hermes `terminal` and `execute_code`) keep their
  body in judge evidence instead of the first 200 characters. Each write body
  gets an even share of the room left, at least 1,800 characters, and any cut
  keeps the start and end with a marker that says how much was cut. File
  changes leave room for the newest tool results and for a short tool history,
  which lists each write as one line, so skill calls and test runs stay in
  view. An over-budget history drops its middle first, and its copy of the
  final answer, which FINAL RESPONSE already shows, shrinks before any tool
  call drops. Judge evidence is secret-redacted; placeholder keys such as
  `sk-your-key-here` stay as written. Scores for cases that write files can
  change.

### Fixed

- Plugin evaluation review fixes. Tier 1 plugin checks fail closed on unreadable
  hook, LSP, monitor, or settings config, on component lists past the 256-entry
  cap, on permission-bypass scans that hit their bound, and on remote MCP bundles
  over `http://`; they catch option-form permission bypasses
  (`--permission-mode bypassPermissions`, `--approval-mode yolo`,
  `--sandbox danger-full-access`) and percent-encoded endpoint hosts, and URL
  findings no longer echo userinfo or query credentials.
- Tier 2 plugin catalogs treat an entry as the plugin under test only when its
  manifest fingerprint matches, so same-name plugins and verbatim skill copies
  are compared and reported. A malformed plugin is skipped by name when building
  a catalog, and an unsafe plugin profile fails both catalog checks.
- Tier 3 plugin runs no longer count MCP servers launched from plugin files as
  runnable, block only on MCP sources they stage, and bound ref-listed rules by
  the aggregate byte limit.
- Tier 3 security checks match protected writes by their redirect, `tee`, or
  `sed -i` target (so `2>/dev/null` reads are no longer critical), normalize
  `//`, `/./`, and `..` path spellings, and store the matched protected entry
  rather than the command as path-finding evidence. Reference-less judges stay
  N/A on no-trajectory trials, and a partially salvaged behavior verdict is left
  unscored.
- Plugin signals never persist tool-argument values or non-name component
  names, run dataset regex patterns under a deadline, scan every MCP result
  block for failure markers, and judge falsy but defined ground truth.
- Plugin reports fail closed on an unreadable provenance sidecar, say how many
  components were not staged or staged but not observed, and label a
  sum-of-parts lift as integration rather than effectiveness. `regex` is now a
  direct runtime dependency.
- Kept Tier 3's interactive progress frame at a stable height, bounded visible
  stage history, serialized terminal redraws, and safely disabled a reporter
  when initialization or background refresh fails.
- Tier 3 preserves completed rewards from partially errored jobs only when each
  aggregate error maps to a concrete failed trial; explicit failed statuses and
  non-zero aggregate exit codes still suppress ambiguous scores.
- Plugin manifest discovery is now root-bounded across all tiers, and Integration
  evaluation requires explicit cross-component dataset evidence instead of
  reporting unsupported composition claims.
- The dependency audit now checks the dependencies declared in `pyproject.toml`
  instead of SkillEvaluator's own environment, and reports unpinned
  declarations as INFO `dependency-version-unverified`.
- Re-rendered plugin Tier 3 reports (`view`, standalone renders, and the report
  delivered after a plugin run) keep plugin provenance and INCOMPLETE status by
  reading the bounded, no-follow `plugin_provenance.json` sidecar.
- Tier 3 JWT log redaction, on the host and in the Harbor verifier, keeps memory flat on long token runs instead of using about 75 bytes per character.
- The Harbor verifier files parse again on task images whose `python3` is older than 3.12, and CI now compiles them on Python 3.9 and 3.11.
- The Tier 1 PII scan stays linear on very long lines: the email, JWT, database URL, and connection-string patterns no longer take minutes on one crafted line. An email local part over 64 characters is reported by its last 64.
- Local mode restores SIGPIPE before it runs a command, so a pipeline such as `yes | head` ends instead of hanging until the timeout.
- `--checks claude-validate` compares only Claude Code (`.claude-plugin`) plugins and reports other formats as not applicable. It parses the messages of Claude Code's text report, leaves output with no verdict INCOMPLETE, and names files relative to the plugin root.
- The Tier 1 PII scan still reports a database URL whose password holds a raw `/`, and no longer reports a host and port followed by a path or text (such as `redis://cache:6379/0?owner=ops@corp.io`) as a credential.
- `--checks claude-validate` still finds a `claude` installed through Volta, asdf, or nvm under the throwaway HOME, does not count Claude Code's notes as warnings in the text report, and leaves a JSON report that names no manifest and no files INCOMPLETE.
- Native plugin loading rewrites only Harbor's real launch command (its launcher line, at a shell-command boundary), so task or plugin MCP text with the launch words no longer gets the setup or `--plugin-dir`, and a run whose launch is missing or repeated fails instead of running without the plugin.
- Codex native MCP config and the wrapper `plugin_mcp_servers.toml` are written as TOML strings (an emoji or DEL no longer breaks Codex's `config.toml`), the Codex TOML is parsed at staging time, and a broken `plugin_mcp_servers.toml` fails the run instead of silently dropping every plugin MCP server.
- OpenCode native loading stages a plugin agent named like an OpenCode built-in (`build`, `plan`, `general`, ...) as `<plugin>-<name>` instead of replacing the built-in, and turns subagent `tools`/`disallowedTools` into OpenCode `permission` rules instead of granting every tool.
- Hermes native loading labels its components `wrapper`, and `--plugin-load auto` uses the wrapper for Hermes, because its with-plugin task is the wrapper task plus the load census.
- Codex and OpenCode native rules drop Cursor rule frontmatter and stage only always-on rules; glob-scoped, agent-requested, and manual Cursor rules are reported as not loaded.
- Tier 3 treats an MCP server that launches through the manifest format's own plugin-root placeholder (`${PLUGIN_ROOT}`, `${CURSOR_PLUGIN_ROOT}`, braced or bare) or a relative `cwd` as a plugin-file launch: it is no longer staged as runnable for wrapper, Codex, OpenCode, or Hermes arms, and the run is reported INCOMPLETE.
- `--plugin-load native|auto` stages `${CLAUDE_PLUGIN_ROOT}` MCP servers in the native Claude Code `.mcp.json` (the plugin tree is copied there), and a plugin whose only component is such a server is evaluated instead of skipped.
- `--plugin-load` resolves the per-agent plan before staging: the skip decision and the INCOMPLETE rule follow what each with-plugin arm stages, agent- or command-only plugins run where an adapter loads them, and the plugin is snapshotted only when some agent loads it natively.
- A permission-bypass flag in a plugin hook, subagent, settings file, or LSP server now blocks only the adapters that would stage it: `auto` falls back to the wrapper for that agent with the reason recorded, and `native` fails only for those agents.
- Native Claude Code staging translates Cursor hook events to Claude Code events and roots relative hook commands at `${CLAUDE_PLUGIN_ROOT}`; Cursor events with no equivalent get a `not_loaded` census row.
- Native Claude Code staging keeps `userConfig`, the plugin `settings.json` keys Claude Code applies (`agent`, `subagentStatusLine`), LSP servers, and MCP `env`/`headers` (refusing literal secrets), and lists `bin/` and Codex apps in the load census.
- Native Claude Code staging copies member skills that Claude Code would not read into `skills/<name>` (failing on a name clash), and the load census and routing aliases now cover every skill the staged plugin loads, including declared skill directories.
- Tier 3 reports a plugin run INCOMPLETE when an MCP server uses a `${user_config.*}` value that a with-plugin arm cannot fill in: the wrapper never does, and native Claude Code applies only `userConfig` defaults.
- Native Claude Code user rules keep their full file names, so `style.md` and `style.mdc` no longer overwrite each other.
- `--plugin-load native` with an environment that cannot load natively (such as `--env-mode local`) now fails with that reason before the environment preflight instead of a missing local CLI error, and `auto` without a native snapshot falls back to the wrapper.
- `--plugin-load` plans with the `harbor.task_source` pinned in the evals `config.yml`, as the run does, so a plugin pinned to native Harbor tasks is no longer planned (and reported complete) as a native Claude Code run.
- Native Claude Code staging rewrites a Codex or Agent Plugins `${PLUGIN_ROOT}` in hook commands to `${CLAUDE_PLUGIN_ROOT}`, so those hook scripts run.
- Native Claude Code staging roots any relative word in a Cursor hook command that names a plugin file (`sh scripts/x.sh`), and translates more Cursor hook events (`afterShellExecution`, `afterMCPExecution`, `beforeReadFile`, `sessionEnd`, `subagentStart`, `subagentStop`, `preCompact`).
- Native Claude Code staging fails when a copied member skill has the same name as a skill in a declared skill directory, since Claude Code would load both under one name.
- The coverage reason of a `${CLAUDE_PLUGIN_ROOT}` MCP server now names its own gap, such as an unfilled `${user_config.*}` key, instead of always blaming the other with-plugin arms.
- The plugin hook census counts exit 2 as a blocked run (the hook's deny
  decision), not a failure, and exit 126/127 as not started; reports show both.
- Native plugin loading marks a component `loaded` only when the harness reports
  it (Claude Code's startup event; a failed or pending MCP server is not
  loaded). A file listing is now `listed` and keeps the row `staged`, the Plugin
  Loading table no longer calls it verified, census notes are appended to the
  row reason, and a native run with no load census in any trial is INCOMPLETE.
- The load census ignores entries for components that were never staged and
  only promotes types the agent loads natively; the OpenCode checks require its
  launch config variables, and the Hermes MCP check needs an exact
  `mcp_servers` key.
- A plugin hook counts as exercised only from runs of the exact handler ids that
  were natively staged for that agent, and not when every run failed to start;
  the evidence is labeled self-reported.
- A plugin run with failed trials or a failed arm keeps its with-plugin
  evidence: the load and hook census read every trial, `plugin_provenance.json`
  is written before the error, and Tier 3 reports an INCOMPLETE result instead
  of a skip; a crash in the report-only sum-of-parts arm no longer ends the run.
- The Tier 3 verifier's judge retry also covers HTTP 408, every 5xx except 501
  and 505 (so Anthropic's 529 "overloaded" and proxy 520-524 errors), and a
  response body cut short (`IncompleteRead`), and never retries a failed TLS
  certificate check or a URL error that is not a network failure. The host
  `LLMClient` retries the same HTTP statuses.
- Plugin signals and in-agent MCP proof map native MCP tool names (Claude Code `mcp__plugin_<plugin>_<server>__<tool>`, OpenCode `<server>_<tool>`, Hermes `mcp_<server>_<tool>`, Codex bare names via its session log) to the declared server, and never credit one server's calls to a similarly named one.
- Plugin signals count Claude Code `Skill(<plugin>:<command>)` calls as command activations, credit every SKILL.md in a chained shell read, and no longer mark a skill read as failed because the skill text says something is "not available".
- The Harbor verifier matches Claude Code native `<plugin>:<skill>` names exactly and keeps `<plugin>:<command>` calls out of skill activation and routing grades.
- Plugin signals still mark a shell read of a SKILL.md as failed when the error line comes after Codex status lines or after other output.
- `--probe-mcp` counts the DNS lookup against its 20-second deadline and stops at once, without client tracebacks, on an oversized response or an off-origin SSE endpoint.
- The canary no longer scores common benign commands as a critical leak: a whole environment handed to a child process, loopback-only calls, `if`/`while` tests, `echo` text, commit messages, search and exclude patterns, bare variable names, symlinks, and the literal `$NAME` in tool arguments.
- The canary summary counts the decoys the verifier found (`planted_file`, `decoy_missing`), compares arms by leak rate, names sum-of-parts leaks, and says "Canary not confirmed" instead of a green pass when the decoy was missing.
- Tier 1 hook risk now covers skill and command frontmatter hooks, plugin
  monitors, and Cursor and GitHub Copilot hooks (their approval events,
  `permission: allow` output, and `bash`/`powershell` commands), and flags plugin
  skill `allowed-tools` like a command's (`plugin_skill_unrestricted_bash`).
- Tier 1 remote-code checks no longer flag fetch-then-parse hooks
  (`curl … | python3 -m json.tool`, `jq . out.json`), and now catch `| $SHELL`,
  `| busybox sh`, `source /dev/stdin`, evaluated download variables,
  `exec(urlopen(…))`, git or URL package runners, unpacked archives, and a file
  one hook downloads and another runs.
- Tier 1 scans hook scripts whole (up to 1 MiB) and one script level deeper (also
  after `cd ${CLAUDE_PLUGIN_ROOT}`), reads hard-linked scripts instead of failing
  them, and reports an approval hook whose plugin script cannot be read.
- Tier 1 adds MEDIUM checks for auto-approving write, fetch, or MCP tools, unpinned
  hook package runners, hooks that run `${CLAUDE_PLUGIN_DATA}` code, the `auto` and
  `acceptEdits` permission modes, and LSP server command forms.
- Tier 1 classifies Bash grants the way Claude Code does (`Bash()`, `Bash(**)`,
  `Bash(python3:*)`, `Bash(sh -c *)`) for commands, skills, and shipped settings,
  and flags Codex bypass flags (`--dangerously-bypass-hook-trust`, `-a never`,
  `-c approval_policy=never`).
- Tier 1 again flags a download piped into `python3 -c`, `node -e`, `perl -e`,
  `ruby -e`, or `sh -c` when that program evaluates its input
  (`exec(sys.stdin.read())`, `eval "$(cat)"`, `bash -c "$(cat)"`).
- Tier 1 follows a script named inside a hook script only where it is run, not
  where it is echoed, tested with `[ -f … ]`, or mentioned in a comment.
- Tier 1 matches hook downloads and runs in linear time, so a 1 MiB hook script
  or many hooks sharing scripts no longer stall it; a hook that runs more than
  2048 distinct files while the plugin downloads something is reported unanalyzed.
- Tier 1 expands simple script variables in download and run paths
  (`D="$CLAUDE_PLUGIN_DATA/bin"; curl -o "$D/tool"`) and counts a relative
  command such as `d/run` as running an unpacked download.
- Tier 1 treats a server-scoped MCP matcher (`mcp__github__.*`) as covering MCP
  tools for auto-approve, and counts Codex's `-a never` and `-c approval_policy=never`
  only after a `codex` command (`grep -a never file` is no longer HIGH).
- Plugin dependency audits no longer pass on missing evidence: `npm audit` runs with `--no-offline`, and an npm manifest with more than 5,000 packages or a missing pip-audit makes a plugin audit INCOMPLETE.
- `--resolve-endpoints` resolves every name before any request, classifies every DNS answer (up to 64), gives each `HEAD` a wall-clock deadline, no longer delays exit on a hanging DNS lookup, and is INCOMPLETE when the endpoint cap, the time budget, or DNS leaves an endpoint unchecked.
- The plugin dependency audit also CVE-audits the packages that `npx`, `bunx`, `pnpm dlx`, `uvx`, and `pipx run` MCP servers install.
- Declared skill folders (for example Cursor `"skills": "./my-skills/"`) now get skill schema checks; skills only in `skills/` are advisory when the format's declared `skills` replace it.
- Codex skills are now inventoried at any depth below `skills/` or a declared skills folder, as Codex discovers them.
- Agent Plugins 1.1.0 is now a MEDIUM unrecognized version and a `$schema` with surrounding whitespace is HIGH, because Codex and Hermes accept only the exact 1.0.0 identifiers.
- A `.codex-plugin/plugin.json` overlay beside a root Agent Plugins manifest is validated as the documented overlay instead of being reported invalid.
- Missing Codex `version`, `description`, or `author` is now MEDIUM, because the Codex runtime loads the plugin without them.
- Markdown and terminal reports show N/A instead of crashing when a baseline score is missing, the lift stays unknown instead of +0.00, and one failing report format no longer stops the others from being written.
- Plugin reports no longer say hooks, subagents and commands cannot be evaluated: they say Tier 3 does not stage them in wrapper mode and Tier 1 checks them statically, and `BENCHMARK.md` drops a type from the excluded list when Tier 3 staged, loaded or exercised it.
- The Markdown report says how many endpoints it left out past 200 and how many `claude plugin validate` errors and warnings it left out past 20, and the docs no longer say the canary token reaches only the verifier.
- HTML reports pass skill and agent names to click handlers through `data-*` attributes, so a crafted skill directory name can no longer run script, and the report sets a Content Security Policy that blocks outside scripts (except the pinned Chart.js file) and outbound requests.
- A lone surrogate in plugin text (for example `"\ud83d"` in a subagent name) no longer crashes the terminal output or the Markdown, HTML and BENCHMARK reports, and terminal output shows bidi and zero-width characters as visible `\uXXXX` escapes.
- Markdown and `BENCHMARK.md` reports keep plugin text inert: terminal escape sequences are removed, link and emphasis syntax is escaped, and plugin values sit in `<code>` elements instead of backtick code spans that showed escapes literally.
- SARIF gives bundled-skill findings repository-relative URIs with `uriBaseId` instead of an encoded `[skill] /absolute/path`, marks a failed Tier 3 run `executionSuccessful: false` with a notification, and reports a plugin-attributable canary leak as an error result; `BENCHMARK.md` shows canary results too.
- The terminal report prints the Tier 3 plugin block (verdict, coverage, loading, canary, census, Integration) for every plugin run, not only when AGENT_EVAL fails or has findings.
- `tier3 evaluate-plugin` no longer says Integration recorded no sum-of-parts comparison next to a measured Integration lift: its summary builds the Integration block the same way the reports do.
- The Markdown Integration line prints the lift once, followed by its interval, instead of repeating the estimate.
- Plugin cost tables show "not priced" with a short note when a run recorded tokens but no USD cost (common for gateway model ids), and the docs explain where USD comes from.
- `validate` and the single-check commands exit 1 when a requested report could not be written; the other reports and `BENCHMARK.md` are still written, and the quiet footer no longer links the missing file.
- SARIF no longer marks an advisory Tier 3 run with no task source as a failed run, and it adds a warning notification when a Tier 3 run was skipped (for example an engine crash or timeout) or is INCOMPLETE.
- A lone surrogate in the plugin provenance sidecar (for example in the plugin name) no longer crashes Tier 3 result building, `view` or re-rendered reports.
- The Markdown report escapes the reason shown for a skipped Tier 3 run and puts the requested plugin load mode in a `<code>` element, and the JSON report counts a plugin-attributable canary leak as critical in `severity_counts`.
- Report-only activation coverage credits an OpenCode plugin agent staged under a new name (`<plugin>-build` for an agent named `build`) to the declared agent, instead of leaving it never exercised.
- Skill-scoped checks (version, quality, lint, and folder schema walks) no longer skip every skill of a plugin or folder that lives under a directory named `results`, `versions`, or `evals`, and they treat a first-level `skills/evals/` or `skills/versions/` folder as a skill, like plugin skill discovery.
- Native Claude Code staging no longer loads a glob-scoped Cursor rule on every task: `globs` (or `paths`) become the user rule's `paths` frontmatter, always-on rules lose their Cursor frontmatter, and agent-requested and manual rules are reported as not loaded.
- The Harbor verifier, custom grader runner, and dataset metric also run on Python 3.9 task images, not only parse there: the canary check no longer calls `zip(strict=)`, the judge retry and reward helpers use tuples instead of `A | B` in `isinstance`, and a 3.9 read timeout (`socket.timeout`) is retried. CI now runs a smoke script on Python 3.9 and 3.11 after compiling.
- Runs without a sum-of-parts arm (the default effectiveness lift, or `--skip-baseline`) no longer print "Integration completeness issues" in the terminal and HTML reports: such runs record `complete: null`, and reports re-rendered from older runs that saved `complete: false` with a skipped sum-of-parts arm stay quiet too.
- The HTML Plugin Signals "Activation coverage" row shows the share of declared components exercised (for example 67% for 2 of 3) instead of "n/a exercised", because it no longer reads the per-component rate map as one number.
- Tier 3 coverage reasons follow the resolved plugin-load plan: a native arm's rule row says "staged as a native rule for <agent>" instead of the wrapper `SKILL.md`, hooks, subagents, and commands a native arm stages are "staged natively for <agent>" before any census arrives, and other rows say why each arm does not stage them (the generated wrapper, or "unsupported by the <agent> native adapter").
- The `tier3 evaluate-plugin` run summary now prints the Component coverage block, like `validate`: it is printed after the plugin provenance is built instead of from the raw engine result.
- A plugin run that is INCOMPLETE only because it did not complete (for example every trial failed on a bad key) or because a native load was never confirmed now says so in the report conclusion ("Evaluation INCOMPLETE - the run did not complete") and in the Dependency Completeness hint, instead of blaming unresolved dependencies when nothing was deferred.
- The Codex, OpenCode, and Hermes load censuses list each MCP server that launches from plugin files as not loaded ("launches from plugin files; not staged by the <agent> native adapter"), so their Plugin Loading rows no longer under-count what did not load.
- Load-census evidence longer than the census text limit is cut in the middle, so a long container or temp path keeps its file name (`.../skills/<name>/SKILL.md`).

### Security

- Hardened plugin input handling with descriptor-anchored, no-follow discovery
  and reads so linked, hard-linked, reparse-point, escaping, and special files
  are rejected before provider calls or sandbox staging.
- Tier 3 security scoring now normalizes home-directory spellings (`/home/<user>`,
  `/Users/<user>`, `/root`, `$HOME`, `${HOME}`, `~user`) and covers more credential
  stores and protected shell, SSH, sudoers, and agent-control files.
- The dependency audit no longer lets pip-audit install or build the audited
  requirements: it audits only exact pins with `--no-deps --disable-pip` and
  ignores pip options in the audited files.
- Plugin static checks flag unpinned MCP package runners, agent permission-bypass
  flags, library-preload and traffic-redirect environment overrides, auto-approve
  keys, metadata and private MCP endpoints, bypass settings, shipped `.env` files,
  and missing or escaping declared component paths.
- Tier 1 security, PII, license, code-integrity, Unicode, and dependency scans of
  a plugin that bundles skills now also cover the plugin's root content
  (`scripts/`, `hooks/`, `.mcp.json`, ...), scanning each file once; a linked,
  hard-linked, or special entry anywhere in the plugin tree fails closed.
- `--checks claude-validate` runs `claude plugin validate` with an allowlisted environment (no credentials; proxy passwords removed) and a throwaway HOME and Claude config directory.
- Fixed an exponential-time Hermes launch pattern (CodeQL py/redos) that a plugin MCP URL could reach to hang a native Hermes run.
- Hermes is refused with the OpenAI and OpenAI-compatible providers, because Harbor's Hermes agent lets Hermes route `openai/MODEL` to OpenRouter and would send the evaluator's OpenAI key there.
- `--probe-mcp-env` accepts `NAME=HOST` and `NAME@SERVER` to send a variable to one host or server, prints which variables may go to which host before probing, and records what was sent; probed tool names and details drop terminal control characters.
- The canary and the other verifier security checks now cover Hermes tools, OpenCode MCP tools and `filePath` writes, Codex `workdir`, Claude Code subagent tool calls, whole-workspace uploads, DNS, cloud, forge, and container clients, `/dev/tcp` descriptors, scripts piped into a shell, and files tainted in an earlier tool call.
- Canary evidence now gets the shared credential redaction, and the outside-file read-back is deduplicated, reads likely files first, and records when it hits its cap.
- The canary treats a call as loopback-only only when every target is a literal loopback address with no proxy, and it now catches sending a symbolic link to the decoy, `awk`/`jq`/`yq` reads of the canary variable, `echo` output captured as a path, escaped `printf`/`echo -e` writes, and package uploads (`npm`/`yarn`/`pnpm`/`cargo publish`, `gem push`).
- Hook and MCP URLs are read the way WHATWG clients (Node, the MCP SDKs) read them, so a backslash or a missing `//` no longer slips past `hooks.allowed_urls` or the metadata checks; such URLs are HIGH findings, and every HTTP hook header is checked for inline secrets.
- The endpoint policy treats documentation, reserved, broadcast, and multicast addresses as non-public and adds the Oracle, ECS task, and EKS Pod Identity credential addresses to the metadata set; `--resolve-endpoints` classifies a redirect `Location` where WHATWG clients follow it.
- Container image audits check each registry host against the endpoint policy (metadata addresses are always refused, private registries need `mcp.allowed_private_hosts`) and run Grype and Trivy with an empty Docker config.
- `--resolve-endpoints` classifies a redirect `Location` that starts with three or more slashes or backslashes by the host a client follows, and decodes a percent-encoded host name before the DNS lookup.
- An HTTP hook URL with a password before a backslash is an inline secret again, an MCP URL such as `https:user:password@host` is too, and MCP URL findings no longer print that password.
- Grype and Trivy scans of a plugin-chosen registry also drop the scanners' registry login variables, `DOCKER_AUTH_CONFIG`, and Podman's auth file.
- Plugin checks now cover what other clients load from the same folder: Claude Code `--plugin-dir` defaults in a folder without `.claude-plugin/plugin.json`, Codex reading a Claude Code or Cursor manifest with its own rules, and hooks and apps in an Agent Plugins `extensions["com.openai"]` object.
- An additional client manifest that is over 1 MiB or not UTF-8 is now HIGH `plugin_manifest_additional_unreadable` and is read leniently so its hooks and MCP servers are still checked.
- A root Agent Plugins `plugin.json` over 1 MiB is now parsed whole (up to 8 MiB; a larger one always counts), so padding before `$schema` or escaped slashes no longer let the folder pass as a skill or a decoy manifest win.
- A Codex manifest path that Codex drops (no `./`, `./`, `..`, or a root variable) no longer turns off the checks on the default `hooks/hooks.json`, `.mcp.json`, `skills/`, `commands/`, or `.app.json`.
- Root variables such as `${CLAUDE_PLUGIN_ROOT}` in Claude Code and Codex manifest component paths are now HIGH `plugin_component_path_invalid`.
- Case variants of client manifests, such as `.Codex-Plugin/plugin.json`, are now found and checked, with a HIGH `manifest_case_variant`.
- Skills in `skills/evals/`, `skills/results/`, and `skills/versions/` are now discovered and scanned; a `SKILL.md` deeper in such a folder, or a manifest path into one, is HIGH.
- A skill in an `evals/`, `results/`, or `versions/` folder inside a declared skills folder (for example `my-skills/evals/`) is now inventoried, schema-checked, and HIGH `plugin_skill_in_unscanned_folder`, because clients load it but Tier 1 scans skip it.
- The Tier 1 MCP inline-secret check stays linear on a long JWT-like value (`eyJeyJ...`), which took about a second per 64 KB value and added up across many MCP args, env values, and headers.
- MCP and HTTP hook URLs with an invisible format character (such as a zero-width space in the host) or Unicode whitespace at either end are now HIGH, like other whitespace and control characters; only leading or trailing ASCII whitespace is still allowed.
- Tier 3 coverage and the dependency (CVE) audit now read an additional client manifest that is over 1 MiB or not UTF-8 leniently, like Tier 1, so its hooks and MCP servers no longer drop out of coverage or the audit; a component only another client loads names that client in its Tier 3 coverage reason.
- Tier 1 whole-tree scans now scan a skill that a declared skills folder loads from an `evals/`, `results/`, or `versions/` folder (such as `my-skills/evals/`, or a declared `./evals/`) as its own skill unit, after a no-follow check, instead of only failing it HIGH; a `SKILL.md` deep inside a skill's own artifact folder stays HIGH `plugin_skill_in_unscanned_folder`.
- A hook that pipes a download into a shell or interpreter is CRITICAL remote code again when a comment or a redirection follows the shell (`curl … | sh # install`, `| sh > /dev/null 2>&1`, `| python3 # c`); only a real script path makes the download plain data.
- Decoding `\uXXXX` and `\xXX` escapes in hook scripts is linear: a hook script that is a long run of backslashes no longer stalls every Tier 1 check that builds the plugin inventory (a 1 MiB script took hours).
- The canary check stays linear on code padded with blank lines: a plugin could steer the agent into one large code or shell call that made the verifier spend minutes on the environment-copy pattern and hit its timeout, losing the trial's canary result.

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
