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

### Fixed

- Brought the published documentation in line with 0.5.0. It now covers how
  standard grading builds judge evidence and rebuilds trajectories from
  OpenCode and Codex logs, structured judge output and retries, the evidence
  budget and judge time-budget variables, and the Harbor process environment.
  The CLI reference adds `validate --workers`, the full `--env-mode` list, and
  the exit codes the CLI actually returns (`0`, `1`, `2`).
- Tier 3 no longer treats flag values of credential-named variables (for example `XDG_SESSION_ID=1` or `FOO_AUTH_ENABLED=true`) as secrets, so progress counts are not redacted and the Docker sidecar no longer refuses values such as `127.0.0.1` or `python:3.13-slim`.

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
  `LLMClient` retries the same HTTP statuses. No judge path retries a failed
  certificate check: not the HTTP providers, not Bedrock, and not the host
  `LLMClient`, even when an SDK wraps it in a connection error.
- A judge HTTP 400 or 422 keeps the provider's error body after the
  structured-output check reads it. `LLM_JUDGE_FALLBACK_MODELS` again switches
  models on a 400/422 "invalid model" error, and the judge error shows the
  provider's message instead of only `HTTP 400: Bad Request`.
- The Harbor verifier honors a `Retry-After` HTTP date on Python 3.9 and 3.10
  task images. It used to fall back to the 1-second base delay there, and a date
  past `SKILL_EVAL_LLM_RETRY_MAX_DELAY` did not fail fast.
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
- Plugin component paths follow the clients: in a Claude Code manifest, a path without `./`, an `agents` path that is not a `.md` file (a folder included), and a `hooks`, `lspServers`, or monitors path that is not a `.json` file are now HIGH, because Claude Code rejects the whole manifest. A commands-map `source` that names a folder is accepted, as in Claude Code. A Codex path that Codex drops is no longer listed or staged at Tier 3. A Cursor root `mcp.json` is checked next to a declared `mcpServers` (LOW `mcp_root_config_also_checked`), and a Cursor component path that starts with a root variable is MEDIUM `plugin_component_path_root_variable`.
- Components in folders the Tier 1 scans skip no longer pass silently: a declared component, or MCP server code, in `node_modules/`, `.venv/`, `.git/`, or `__pycache__/` is HIGH `plugin_component_path_unscanned`; a declared skills folder there is scanned as its own skill unit; agents, commands, and output styles that Claude Code loads from `agents/evals/` and similar subfolders are inventoried, privilege-checked, and HIGH; a link out of the plugin inside a skipped folder is HIGH `plugin_unscanned_folder_link`; and a shipped `node_modules/`, `.venv/`, or nested `.git/` gets a LOW `plugin_unscanned_folders` note.
- Plugin frontmatter that strict YAML rejects is read the way Claude Code reads it, so `allowed-tools`, `tools`, and `permissionMode` after a value such as `description: Deploy: runs it` are checked, and such descriptions count toward the context-cost estimate.
- A folder without a manifest that ships Claude Code plugin components (`agents/`, `commands/`, `output-styles/`, `hooks/hooks.json`, `monitors/monitors.json`, or `.lsp.json`) is now validated as a plugin with a MEDIUM `manifest_missing`, instead of as a skill collection.
- The component inventory keeps each broken component's `problem` and a `broken` count, reports label such rows "Broken" instead of "Evaluated", bundle refs show their dependency state, and findings are counted by the normalized declared path and by the resolved plugin root, so broken MCP files, invalid server names, renamed agents, and plugins behind a symlinked folder no longer show 0 findings.
- More than 100 plugin component findings no longer fail Tier 1 when only MEDIUM or LOW ones are left out (LOW `schema_findings_truncated`); blocking findings are always reported first.
- A plugin skill with deeply nested YAML frontmatter no longer crashes the run with no reports, and a `SKILL.md` that is not UTF-8 is HIGH `bundled_skill_not_utf8` instead of an unsafe path that stopped every other check.
- The plugin `quality` check scores the skills the client loads (declared skill folders too), not a `skills/` folder that the declared folders replace.
- A skill that a client loads from `node_modules/`, `.venv/`, `.git/`, or `__pycache__/` inside `skills/` or a declared skills folder is now HIGH `plugin_skill_in_unscanned_folder`. Claude Code loads `skills/node_modules/SKILL.md` and Codex loads `skills/node_modules/pkg/x/SKILL.md`, but skill discovery skipped them, so such a skill passed Tier 1 with only a LOW note.
- MCP server code in a folder the scans skip is now also found through `..` in the path, a `--flag=./path` argument, and a server `cwd` inside the plugin (Codex resolves a relative `cwd` against the plugin folder).
- A chain of links inside a skipped folder that leads out of the plugin, such as `node_modules/up -> ..` with `node_modules/evil -> up/../outside.md`, is now HIGH `plugin_unscanned_folder_link`; a link loop counts as leading outside.
- A commands-map `source` folder no longer makes Tier 3 native staging refuse the whole run. Each `.md` file directly in the folder is listed as its own command named after the file, as Claude Code loads it; a folder without one is a broken row with a MEDIUM `plugin_command_folder_empty`.
- A standalone skill with deeply nested YAML frontmatter is reported as a HIGH `yaml_syntax` finding instead of crashing the schema check.

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
- Plugin Tier 3 runs no longer lose agent data in Harbor's trajectory conversion: Codex agent steps carry their per-step token counts, so the measured first-turn context cost works for Codex; Claude Code subagent transcripts (`<session>/subagents/*.jsonl`) are converted, so the steps and MCP calls a subagent makes count for handoff, conflict, MCP, and security checks; and parallel Codex tool calls from one model request stay in one step, so check 18's same-step guard holds. A Codex run whose agent calls `spawn_agent` is converted from the main thread's rollout: Harbor converts only the newest rollout, which is the subagent's, so the main agent's calls, final answer, and first-turn tokens were lost and the judge graded the subagent's last message. Every Codex run (plain `codex` included) now goes through SkillEvaluator's Harbor Codex wrapper, which picks the main thread and adds each subagent's own steps as sidechain steps (`extra.is_sidechain`, `extra.agent_id`), never after the main agent's final answer.
- Native Claude Code plugin loading recognizes Harbor's current launch, which pipes the prompt in from an environment variable instead of passing it after `--`; without this, every native Claude Code run failed with "did not find the Harbor launch command".
- Non-secret Harbor backend constructor options can be passed with the repeatable, operator-only `--environment-kwarg` / `--ek` flag on `validate` (skills and plugins), `evaluate`, `doctor`, and `health-check`; skill-owned configuration, credentials, and sandbox-policy fields stay outside that surface.
- A Claude Code `.claude-plugin/plugin.json` is now checked against the manifest schema Claude Code applies, not only its name: wrong field types, a padded or spaced name, a bad `dependencies` value, an invalid inline `lspServers` or monitor entry, and other schema errors are HIGH (Claude Code refuses the whole plugin), unknown top-level fields are MEDIUM, and a long name is no longer HIGH (Claude Code has no length limit). Empty, missing, and non-string names get their own messages.
- A Claude Code plugin's `dependencies` are now resolved offline against the plugin's own `.claude-plugin/marketplace.json`, like `agent_plugin.yaml` refs: a dependency the marketplace does not list is HIGH `plugin_dependency_missing` (Claude Code cannot install it and refuses to load the plugin), and one in another marketplace or with no marketplace in sight is MEDIUM `plugin_dependency_unverified`.
- Codex manifest severities now follow what Codex does: a wrong-typed `interface` field (`capabilities`, `displayName`, `shortDescription`, `category`, `websiteURL`, ...) or a blank or unsafe `version` is HIGH because Codex refuses the manifest, while a 65-character or `-abc` name and a wrong-typed `homepage`, `extensions`, `skills`, `commands`, `mcpServers`, or `hooks` value are MEDIUM because Codex still installs the plugin. Unknown top-level and `interface` fields are now MEDIUM, with a did-you-mean hint.
- The opt-in `claude-validate` parity check now runs `claude plugin validate` without `--strict` (recording what `--strict` would decide as `strict_verdict`), so a clean minimal manifest no longer reports a disagreement, and it compares the manifest fields each side fails, so two different failures no longer count as agreement.
- Check 1 details: a selected manifest that is not UTF-8 or over 1 MiB is HIGH `manifest_unreadable` and is read leniently instead of stopping every other check as a security failure (parity still runs), except when nothing can be parsed even leniently, or for `agent_plugin.yaml`; then it fails closed; a linked manifest is named `manifest_linked` or `manifest_hardlinked` when it stays inside the root; a YAML `version: 1.10` keeps the text `1.10` with a LOW hint to quote it; a policy that raises a manifest finding to HIGH also removes the "passed" row; the Codex overlay row is labelled; and naming a manifest file that is not the selected one gives an INFO `manifest_named_not_selected`.
- A number or boolean where a component path belongs in an additional manifest (such as `"apps": 5` in a Codex overlay) is now reported once, as the component path finding, instead of also as a MEDIUM `plugin_manifest_additional_invalid` with a different severity.
- `declared_dependencies` in plugin reports now counts only a Claude Code manifest's `dependencies`, as `plugins`; `keywords` and component lists such as `skills`, `mcpServers`, and `hooks` are no longer counted as dependencies.
- A Codex manifest is now HIGH when `interface.logo`, `logoDark`, or `composerIcon` is not a string, when `interface.screenshots` is not an array of strings, or when `keywords`, `interface.capabilities`, or `interface.screenshots` is an explicit `null`, because Codex refuses those manifests; this applies to a Codex overlay too.
- A Claude Code `homepage` that Claude Code's URL check refuses (a space, `<`, or `|` in the host, a bad percent escape, an unclosed IPv6 bracket, an invalid IPv4 host, or a port over 65535) is now HIGH `schema:homepage:invalid_url`; a wrong-typed `experimental.themes`, `outputStyles`, or `evals` is reported as `schema:experimental.<key>:type`, and an explicit `null` there is HIGH too.
- A number or boolean where a component path belongs in the selected manifest is now reported once, as the component path finding, instead of also as `schema:<field>:type`; an additional `agent_plugin.yml` keeps a YAML `version: 1.10` as `1.10`, so it no longer conflicts with an identical `agent_plugin.yaml`.
- The `claude-validate` parity check no longer reports a disagreement when both sides fail the same field: a missing `commands['x'].source` path, an MCP server declared inline in `plugin.json`, and an `experimental.<key>` error now count for the field Claude Code names.
- MCP pinning reads a moving tag only where the runner reads the version (after the package name's `@`, or the image's `:tag`), so exact pins such as `@nextui-org/mcp@1.0.0`, an entry point `mod.cli:main`, an e-mail, or a digest-pinned `:latest@sha256:...` image are no longer a blocking HIGH. Every moving tag (`beta`, `rc`, `master`, `nightly`, `:1-latest`, ...) is HIGH `mcp_command_floating_version`.
- MCP pinning looks through launch wrappers (`cmd /c`, `env X=1`, `timeout`, `sudo`, `sh -c`) and reads `bun x`, `uv run --with`, `pnpm --package=... dlx`, `npm --yes exec`, `go run mod@version`, `dnx`, every `uvx --with` requirement, npx value flags, and a Windows launcher path with a space. Command hooks use the same rules: a moving tag is HIGH `plugin_hook_command_floating_version`, and `docker run` is classified.
- MCP pinning reads a moving tag only where the runner reads the version (after the package name's `@`, or the image's `:tag`), so exact pins such as `@nextui-org/mcp@1.0.0`, an entry point `mod.cli:main`, an e-mail, or a digest-pinned `:latest@sha256:...` image are no longer a blocking HIGH. Every moving tag (`beta`, `rc`, `master`, `nightly`, `:1-latest`, ...) is HIGH `mcp_command_floating_version`, while an image tag that starts with a `major.minor` version names one release (`:1.0.0-beta`, `:3.0-dev`) unless it says `latest`.
- MCP pinning looks through launch wrappers (`cmd /c`, `env X=1`, `timeout`, `sudo`, `sh -c`) and reads `bun x`, `uv run --with`, `pnpm --package=... dlx`, `npm --yes exec`, `go run mod@version`, `dnx`, every `uvx --with` requirement (the worst of the tool and its requirements decides), npx value flags, and a Windows launcher path with a space. Claude Code's `${NAME:-default}` in `command` and `args` is read at its default, as in URLs. Command hooks use the same rules: a moving tag is HIGH `plugin_hook_command_floating_version`, and `docker run` is classified.
- MCP checks follow each client's environment expansion: Claude Code's documented `"${API_BASE_URL:-https://api.example.com}/mcp"` is checked as its default instead of CRITICAL `mcp_url_dangerous_scheme`; a Codex plugin URL with `${...}` is HIGH `mcp_url_env_not_expanded`; `${VAR:-}` and Cursor's `${env:NAME}` are references, not inline secrets.
- MCP, endpoint, and PII findings no longer copy a secret into any report format: the inline-secret argument message names the argument, echoed commands and package specs are redacted, URL user information and queries are hidden anywhere (also inside `${VAR:-https://user:pw@host}` and a path default such as `${MCP_PATH:-/mcp?token=...}`), and the PII scan reports a credential as its public prefix and length.
- Plaintext MCP URLs to a loopback host are no longer HIGH (as for HTTP hooks), and `oauth.authServerMetadataUrl` gets the MCP URL policy and `--resolve-endpoints`.
- MCP secret heuristics: `PASSWORD_POLICY=strict`, `DB_PASSWORD_FILE=...`, a path such as `/run/secrets/db`, an argv `foo|bar`, and a PEP 440 range are no longer CRITICAL, but a token that only starts like a path (`/Xk9f...`, `./Secr3t...`) still is; short keys such as `--key`, `--pass`, `--sig`, and an env `KEY` count when their value looks like key material; a random-looking value under a neutral key is MEDIUM `mcp_possible_inline_secret`; `--auth <value>` and `oauth.clientSecret` are found; Codex auth fields get a Tier 1 `mcp_field_ignored` advisory.
- Endpoint checks: a failed `HEAD` or redirect lookup makes `--resolve-endpoints` INCOMPLETE; `key=`, `auth=`, and `sig=` are credentials; a space in the path is not malformed; backslash URLs name the host clients reach; a missing scheme is HIGH `mcp_url_scheme_missing`; the Tencent and Azure metadata IPs are metadata endpoints.
- An MCP or LSP server that runs a shell program through a wrapper or a combined flag (`env sh -c`, `timeout 30 bash -c`, `sudo sh -c`, `bash -lc`, `sh -ec`, `bash -o pipefail -c`) is CRITICAL `mcp_command_dangerous_form`, as a direct `sh -c` is. The program of a shell's `-c` and inline interpreter code (`python -c`, `node -e`) get the shell-metacharacter check, while other arguments are argv text.
- A plugin hooks file with a harmless handler, or a skill that names a file it does not ship (such as `Read CHANGELOG.md`), no longer makes the default Tier 1 run INCOMPLETE: SkillSpector's `opaque_content` on a hooks file the plugin's own hook risk check parses, and `reference_missing`, are reported as notes. Any other partial SkillSpector reason still makes the scan INCOMPLETE.
- Codex plugin hooks are graded with Codex's own names: an allow for `apply_patch` (Codex's file-edit tool) is MEDIUM `plugin_hook_auto_approve_scoped`, code run from `$PLUGIN_DATA` is MEDIUM `plugin_hook_runs_unshipped_code`, `Interrupt` is a known event, and `SubagentStart` output is a LOW context injection. Claude Code's `PowerShell` matcher now counts as a shell matcher, so an allow hook on it is HIGH.
- Hook risk details: an allow string in a script comment is no longer a HIGH, a base64-encoded allow is decoded, a handler's `if` condition narrows the matcher (`if: Bash(git status)` is no longer HIGH), JSON outputs that reach the model (`additionalContext`, a replaced tool output, a Stop block reason) and `UserPromptExpansion`/`PostModelSwitch` output are LOW context injections, an HTTP handler on `SessionStart` (which Claude Code never runs) is not, a relative script path such as `./scripts/x.sh` is MEDIUM `plugin_hook_outside_root` because the client runs it in the user's project, the private-endpoint message is right with `--resolve-endpoints`, flag labels match between summaries and tables, and the CLI says when it cuts its hook and privilege lists at 10 rows.
- A Codex plugin's `allowed-tools` Bash or wildcard grant is now MEDIUM (it does not fail the plugin): Codex ignores `allowed-tools` and has no Bash tool, so the grant matters only if Claude Code loads the same file, and the message now names both clients. Grant findings for a component only another client's view loads (Claude Code `--plugin-dir` in a Codex or Cursor folder) are capped at MEDIUM and start with `loaded only by <client>`.
- Privilege checks follow Claude Code more closely: `disallowedTools: bash` (lowercase) no longer counts as denying Bash while `Bash(*)` does, `disable-model-invocation: "true"` marks the command user-only, `allowed-tools: "*"` is LOW because it pre-approves nothing, `permissionMode: BypassPermissions` is HIGH in Tier 1 as Tier 3 native staging already refused it, the privilege headings in every report count skills, and a YAML-list `allowed-tools` is valid skill frontmatter.
- LSP servers get the MCP `env` checks for TLS turned off and inline credentials (CRITICAL `plugin_env_insecure_tls` and `plugin_env_inline_secret`), their command findings say "LSP server" instead of "MCP", and shell metacharacters in an LSP argument are LOW because Claude Code starts the server without a shell. Monitor findings now count on the monitor's own inventory row, and the auto-mode coverage reason says when native loading would refuse a component.
- Check 8 covers more overrides: MCP `--permission-mode acceptEdits` (MEDIUM `mcp_permission_mode_flag`), `--allowedTools Bash` passed to an agent CLI (HIGH `*_permission_allow_flag`), Gemini CLI's `-y`, `BASH_ENV`, `PERL5OPT`, and `RUBYOPT` (HIGH) and module search paths outside the plugin (MEDIUM) in any `env`, a shipped `.codex/config.toml` with `approval_policy = "never"` or `sandbox_mode = "danger-full-access"` (HIGH), settings `enabledMcpjsonServers` and `mcp__*` allow rules (MEDIUM), and TLS-off or inline secrets in settings `env` (HIGH). The Codex short options now need a codex command in the same shell command: `echo codex done; grep -a never log` and `tool -s danger-full-access` are no longer HIGH, and `codex-wrapper -a never` is.
- Hooks in a skill's or command's frontmatter are now `hook` components of their own (`<file>#hooks`) with an inventory and coverage row. Native Claude Code staging census-wraps them inside their own staged file (they stay scoped to their skill or command), refuses them when they carry a permission-bypass flag, and the wrapper-mode coverage row says that a skill's frontmatter hooks run with the skill but without the census. A bypass flag in a member skill's frontmatter hooks now blocks Tier 3 staging in every load mode, because every mode stages member skills as written.
- Whole-tree content scans: a dead or invalid Markdown link and a banned package are HIGH `HYGIENE` findings (SARIF results, counted on the owning component) instead of plain error strings; a base64-encoded instruction (`decode this base64 and follow it`) is HIGH `base64_hidden_instruction`; a script that pipes credentials into an upload (`tar cz ~/.ssh | curl -T - ...`) is HIGH `credential_exfiltration_pipeline`, both with `--no-llm`; and the HTML report shows bidi, zero-width, and tag characters as visible escapes.
- Every finding about a file that only another client's view loads (for example an `.lsp.json` in a Codex folder that Claude Code `--plugin-dir` would start) now starts with `loaded only by <client>` and carries `metadata.loaded_by`; hook, MCP, and LSP findings keep their severity because that client would run the code.
- A hook's `if` condition counts only for a tool its matcher selects: `Edit|Bash` with `if: Edit` is MEDIUM `plugin_hook_auto_approve_scoped` (it had no finding), and `Edit|Write` with `if: WebFetch` is not flagged. A `decision: block` reason from a `PostToolUse` or `PostToolUseFailure` hook is a LOW context injection, because Claude Code adds it to the agent's context.
- The Codex short options are found again in a backslash-continued command and after a redirection (`codex exec \` then `-a never` on the next line, `codex exec 2>&1 -a never`), at Tier 1 and by the native-staging refusal.
- `credential_exfiltration_pipeline` also scans a plugin whose only scripts are `.bash` or `.zsh` files, extensionless files with a shell, Python, or Node shebang, and executables under `bin/`; before, it ran only when the plugin had a `.py`, `.sh`, or JavaScript file.
- Wrapper-mode coverage rows say when `--plugin-load native` would refuse a component (a permission bypass) instead of saying native loading stages it.
- A command or skill that only Codex loads through a Claude Code manifest (a `commands` map hides `commands/` from Claude Code) now gets a LOW finding that says its `allowed-tools` grant has no effect in either client, instead of a MEDIUM that warned about Claude Code; its privilege row gets `inert_grant` and is not counted as flagged.
- Python dependency advisories are now `python-vulnerability` findings. A policy override on any other dependency finding no longer erases them (it turned exit 1 into exit 0), and they now reach SARIF, `BENCHMARK.md`, HTML and `severity_counts`. Their severity comes from the public advisory record (the GitHub advisory severity, then the CVSS v3 score) instead of "every advisory is HIGH"; when the record cannot be read the finding stays HIGH and says the severity is unknown (`SKILLEVALUATOR_OSV_API_URL=off` turns the lookup off).
- The plugin dependency audit no longer passes silently: an unreadable or unparseable `requirements*.txt` or `pyproject.toml` makes the Python audit INCOMPLETE, a pin pip-audit cannot audit (not on PyPI) is a MEDIUM `dependency-not-audited` finding and is not counted as audited, and a `yarn.lock`, `pnpm-lock.yaml` or Bun lockfile with no npm lockfile beside it makes the npm audit INCOMPLETE. Safety 3.x results are read, MCP container image findings point at the MCP file, and bundled-skill npm findings no longer repeat the skill folder.
- Bundled skills keep their MEDIUM, LOW and INFO findings when they pass, a CRITICAL finding in one bundled skill no longer stops the checks of the next ones, and bundled-skill findings now count in `severity_counts`. `apply_policy` keeps plain-string errors that have no finding behind them.
- Plugin dependency refs: more than 256 refs no longer turn the missing-dependency gate off (every ref is classified, and a MEDIUM `plugin_dependency_limit` finding names the limit); a ref the gate could not check is a MEDIUM `plugin_dependency_unverified` finding with advice for its actual cause instead of an `[OK]` row; an `http://`, `git://` or SSH `origin` proves identity; the result no longer depends on filesystem letter case; and a ref listed twice is counted once.
- Tier 3 resolves dependency refs exactly as Tier 1 does, from the same repository root: a same-named bundled skill no longer covers a missing or external ref (the run is INCOMPLETE), a ref Tier 1 calls `referenced` is staged, and a plugin whose refs are all external or whose only component is a provider-only MCP server is INCOMPLETE (`evaluate-plugin` exits 1) instead of skipped with exit 0.
- Tier 3 rule refs use the Tier 1 classifier too: a rule ref whose name differs only in letter case is no longer staged on a case-insensitive filesystem, and a rule ref Tier 1 calls `provided` (inside the plugin) is staged instead of left unresolved. A plugin whose only component is an MCP server launched from plugin files (such as `node ${CLAUDE_PLUGIN_ROOT}/server/index.js`) is INCOMPLETE in the default wrapper mode, and the message now gives advice only for the components the plugin really has (for that server: run the `claude-code` agent with `--plugin-load native` or `auto`), not skill-ref advice.
- Plugin skill and rule refs accept `source: gitlab`, as a selector or a canonical `gitlab::<group>/<repo>::<kind>::<name>` ref, including GitLab subgroup repositories such as `example-group/tools/agent-catalog`. They resolve against the `origin` remote exactly like `github` and `git` refs, at Tier 1 and Tier 3; any other source is still rejected.
- The dependency audit's OSV severity lookup stops at the first network failure and has a 30-second budget per run, so a blocked advisory service no longer stalls `--checks dependency` for minutes. An advisory is reported for each pin it affects when a file pins a package twice (marker-split pins), for pip-audit and Safety.
- An exact npm pin the public registry does not have (a private package, or a version that was never published) is no longer counted as audited and clean: the audit asks the registry first for pins with no evidence they came from it (bounded, `SKILLEVALUATOR_NPM_REGISTRY_URL=off` turns it off), and such a pin, or a lockfile entry from a git or file source, is a MEDIUM `dependency-not-audited` finding. Findings for packages an MCP runner installs name the server in `metadata.mcp_server`, and an unreadable Dockerfile or npm manifest no longer says "Tier 2" in the Tier 1 audit.
- Tier 2 plugin context deduplication no longer counts each bundled-skill severity twice in `severity_counts`.

### Fixed

- The plugin context-cost estimate now models forced output styles like Claude Code: `force-for-plugin` set to `true`, `"true"`, `yes`, or `1` forces the style, only the first forced style counts (Claude Code applies one), and it replaces the default coding instructions unless `keep-coding-instructions` is set, so its net cost can be negative.
- The static context-cost estimate is now given per harness and load mode (`harness`, `load_mode`, `by_harness`: Claude Code and Codex, native and wrapper). Rules count always-on under native loading, Codex leaves out agents, commands, and styles, Claude Code leaves out `disable-model-invocation` skills and commands, `SessionStart` and `UserPromptSubmit` hook output read from a plugin file is counted, and MCP tool schemas and other hook output are marked `not_counted` with `lower_bound: true` instead of 0. Tier 3 provenance uses the run's own harness and load mode, and the wrapper note appears only for wrapper runs.
- The estimate counts CJK and other full-width characters as 1 token each (`chars_div_4_cjk`), counts Cursor `alwaysApply` rules always-on without their frontmatter, and counts a non-UTF-8 agent, command, or style the way Claude Code lists it.
- The measured Tier 3 context cost leaves out Codex trials whose first model call ran a hosted web search, keeps every trial with a count when an arm failed elsewhere (`partial`, with the arm named in the reason), and marks fewer than 3 paired cases `insufficient`. Reports say "paired cases", no longer print a double period, and name the member-skills baseline in `--lift-mode integration`.
- The static estimate counts a rule always-on only when the native adapter stages it that way. Agent-requested and manual rules (`alwaysApply` set but not true, or a `.mdc` rule with frontmatter and no `alwaysApply`) count 0, a `paths` or `globs` rule counts 0 under Codex native loading, and `alwaysApply` is on only for a YAML true or `"true"`. A `SessionStart` hook whose matcher does not select `startup` (for example `compact`) is on-demand, not first-turn cost. SARIF `contextCost` now carries `harness`, `loadMode`, `lowerBound`, and `notCounted`.
- The measured context-cost reason says when no usage was collected for a trial, instead of saying the trial has no first-turn token count.
- Tier 3 runtime security matches paths on the agent's own HOME and config directories (`SKILLEVAL_AGENT_HOME`, `SKILLEVAL_AGENT_CLAUDE_CONFIG_DIR`, `SKILLEVAL_AGENT_CODEX_HOME` in `[verifier.env]`, else the verifier's own `HOME`, `CLAUDE_CONFIG_DIR`, `CODEX_HOME`), so a write to `/tmp/agent-home/.claude/settings.json` or a local-mode `<trial>/local-environment/home` file is caught; paths match whole components (`~/.sshrc`, `/workspace/root/.bashrc` and `~/.profile_backup` no longer count), and text that only mentions a protected file (a doc heredoc, a commit message) is not a write.
- Runtime security reads every credential store a call touches, relative reads after `cd ~` (Claude Code's Bash keeps its directory), Glob patterns, `find -name id_rsa`, writes through `cp`/`mv`/`ln`/`dd`/`sed -i` and a variable, Codex `write_stdin` input, and Codex child-agent rollouts; protected-write evidence is the same entry for every tool and the message names the file.
- The trial security reason names the critical finding first with counts; harness-written user steps (skill bodies, `AGENTS.md`) are not prompt injections; tool output that instructs the agent is an `indirect_prompt_injection`; key placeholders are not secrets; `git push --force`/`--delete` and `curl | sh` are critical; a quoted `/tmp` target of `rm` is scratch space.
- The canary catches Codex bare-name MCP calls, `web_search_call` and `write_stdin` input, and no longer scores read-only `git tag`, `curl -o` onto the decoy, a `-d` value that only names its path, or `tar -C / etc/hostname` as leaks; an undecodable Codex exec wrapper is read for its string literals.
- `security_attribution.json` merges every attempt of a case, credits the skill only after a real activation, matches the baseline by behavior, follows the canary headline for canary leaks, and reports the harness's own model key in an environment dump as a `harness_credential_exposure` environment warning that is never charged to the plugin.
- Credential reads and protected writes get their own headline, per-arm columns, a BENCHMARK.md CRITICAL line, and SARIF results (`AGENT_EVAL/credential_path_read`, `AGENT_EVAL/protected_path_write`), naming the stores and files.
- In a Claude Code run, a bare tool name (`TaskCreate`, `CronList`) is never a canary MCP sink; Claude Code always names MCP tools `mcp__<server>__<tool>`.
- Relative paths resolve in the directory the agent's shell really used: Claude Code's Bash keeps its `cd` until Claude Code resets it (`Shell cwd was reset to ...`), each Codex `exec_command` starts in the default directory, Codex `write_stdin` text typed into a command that is not a shell or interpreter (`cat > INSTALL.md`) is data, not a command, and a `cd` typed into a shell session stays for that session's next input.
- Runtime security no longer scores a file listing as a credential read: `find -name '*.json' | head`, `rg --files -g SKILL.md ~` and `grep -rl` read nothing; a `find` reads a store only when its expression (`-name`, `-path`, `!`, `-o`) can select the store's file and the files are read (`-exec cat`, `| xargs cat`, `$(find ...)`), and a content search counts only when it prints lines that can hold a secret (`rg` and `ag` skip hidden files unless told not to).
- The `git push --force` and `curl | sh` rules are linear, and so are one long `git push -fff...` flag run and a long `grep -r '[[[...'` pattern: a plugin could steer the agent into one long command line that made the verifier spend minutes there and hit its timeout.
- Credential reads and protected writes reach every report even when no canary was planted (a native Harbor task source); the canary headline then says no canary was planted.
- Plugin MCP calls that fail now count as failed: Claude Code's own error flag (Harbor's `extra.tool_result_is_error` and `extra.tool_result_metadata.is_error`), an `is_error` written as a string such as `"true"`, and a Codex `"status": "failed"` in `codex.txt` all mark the call failed, and a Codex `completed` status marks it succeeded. Codex `codex.txt` items are paired with calls by server, tool, and arguments, not by tool name and order.
- Failure words in a good MCP answer no longer mark the call failed: without a structured flag, failure text is read only on a result's first and last lines, and a number from 400 to 599 counts as an HTTP status only when it opens the line or follows `status code`, so `AT-412: Fix race` and `400-500 rps` are not failures.
- The plugin MCP proof no longer counts a server as proven reachable when the agent could not use it: a server the harness failed to load or never listed (Claude Code's init report) is `not-loaded-in-agent`, a server the agent called without one success is `called-no-success` (a warning, not a green pill; older runs' `reachable-in-agent` reads the same), and a failed call never turns an `unreachable` server into anything else. The headline counts only `reachable-host` and `used-successfully`.
- Plugin argument checks no longer pass calls that did not work or rules whose tool was never called: a call the harness refused or the server rejected never passes (`call_failed`), a rule that matched no call counts as a failed check (`not_called`), and with `--probe-mcp` every with-plugin call is also checked against the server's own tool `inputSchema` (`input_schema`).
- Plugin argument report details: the top argument failures count every failure row (the report stopped at 32 of up to 50 per trial) and stay on the arm summary when a report drops an arm's per-trial rewards, the five-errors-per-call schema cap also covers `required`, a `ghp_`-style token in a dataset's expected value is redacted, every bad `tool_arguments` rule is reported instead of only the first, and `contains` on an object argument matches its string values, never its key names.
- Tier 3 plugin staging applies the same MCP policy as Tier 1: the policy's `severity_overrides` and `mcp.allowed_private_hosts` now reach the static checks that gate staging (from `validate --policy`, and from the new `tier3 evaluate-plugin --policy`), an MCP server that Tier 1 blocks but Tier 3 does not stage (such as a root `.mcp.json` that Claude Code loads from a Codex plugin's folder) refuses the run too, and an `agent_plugin.yaml` `mcp` entry must be a `name` and `provider` pair, as the manifest schema says.
- A Tier 3 refusal for an unsafe MCP declaration no longer repeats the credential it found: inline-secret findings name only the check, and other messages lose URL user information and token shapes.
- Plugin MCP call reports show what failed: the CLI and Markdown print the failed and unknown counts next to the success rate (the CLI used to print `100% succeeded (1/3)` with two unknown calls), each server and tool gets its own counts (`by_tool`, and `success_rate` per server in every trial), and the Markdown report gains an MCP Calls table.
- A server's own error text in a good MCP answer no longer fails the call when the harness gave no error flag, so Claude Code and Codex agree on it: `Error: ENOENT means ...`, `TypeError: is raised when ...`, a log search hit such as `ERROR: disk full ...`, and `403: Forbidden is returned when ...` now count as succeeded, in plain text and in an MCP text block. An HTTP status counts only when its reason phrase follows and ends the clause (`404 Not Found`, `401 - Unauthorized: missing token`), and `status code 503` later in a line is no longer a failure; `Request failed with status code 404` still is.
- Probed MCP tool schemas (`--probe-mcp`) no longer push the package's runtime components file over its 256 KiB limit: the file is written as compact JSON and the schemas get only the room left after the declared subagents, commands, and staged names. A plugin whose servers listed many large schemas used to lose its subagent and command signals and every schema check.
- One flaky trial no longer marks a plugin MCP server `not-loaded-in-agent` in the MCP proof: that status now needs the load census to say the server did not load in any scored with-plugin trial. A server that loaded in 11 of 12 trials keeps `reachable-host` (or becomes `called-no-success` when the agent called it) with a "not loaded in 1 of 12 with-plugin trial(s)" note.
- The report keeps the run's exact top argument failure counts: it recounted them from at most 256 trial rewards and overwrote the collector's counts, so an arm with more trials undercounted again. It now recounts only for older runs whose summary has no top failures.
- Plugin value handoffs (check 18) count only the real task prompt as prompt: the skill-load step Claude Code adds after a `Skill` call (it repeats the Skill arguments) and a subagent's prompt no longer make a value that really flowed from the producer fail as "in the task prompt", and both harnesses use the same rule.
- Plugin activation evidence (checks 15, 17, 19) needs a real activation: another plugin's same-named skill (`acme-tools:release-notes`) no longer counts as this plugin's skill, a `SKILL.md` read that failed or never ran (after a failed `&&` part) is not exercised, `cp` sources and `<` input to a non-reader are not skill loads, one skill named with and without the plugin namespace is one identity, and a component whose every activation failed is `unavailable`, not `exercised`, in the trial, the arm summary, the exercise rate, and the reports.
- Plugin component routing (check 15) and tool selection (check 22) are now separate numbers (`routing` and `tool_selection`), so MCP calls no longer dilute skill routing and skill loads no longer dilute MCP tool choice. Codex hosted web search counts as a `WebSearch`/`WebFetch` decoy, unprefixed tool refs mean the same tool on both harnesses (`Bash`/`exec_command`, `Write`/`apply_patch`), an untyped bare MCP tool name such as `list_changes` matches, Claude Code's server spelling works with a tool (`MCP:docs_v2/search`), a call to a tool that does not exist is never a correct choice, a glob never credits the generated wrapper skill, precision is `null` (not 0) when nothing was called, refs to component types the arm cannot carry are skipped, a decoy ref that overlaps an expected one is rejected, and reports call the decoy rate a share of trials.
- Plugin order edges (check 16) now say why they failed (`never_called`, `before_never_called`, `after_never_called`, `same_call`, `same_step`, or `reversed`): `SKILL.md` files read by one shell command and parallel calls in one step are unordered, not reversals; failed calls are not uses; an early look at one `after` alternative no longer flips an edge done in order; edges for component types the arm cannot carry are skipped; every violated edge is listed; edges that can never pass and typo prefixes such as `Skil:` are rejected at validation; and the CLI and HTML say the unit is edges, show the trials fully in order, and list the edges not in order with their reasons. The docs explain when to set `expected_skill: null`.
- Plugin handoffs (check 18) compare values as parsed values, so values with quotes or backslashes (also inside MCP JSON results) pass and `AT-48` no longer matches `AT-4821`; one read of several `SKILL.md` files no longer opens a shared window (the failure says the producer was only read together with others); a non-member or failed skill load, or a look at the consumer's instructions, no longer cuts the producer's window; a subagent's calls keep their own window and count for an `Agent:` side; artifact paths respect the working directory, `cd`, globs, `sed -i`, and inline interpreter code; an MCP write or delete and a subagent prompt that only names the path are not reads; a read must come in a later step than the write and not fail; and self-handoffs are rejected.
- Plugin conflict probes (check 19): a failed call or a missing tool no longer satisfies `must_use`; an MCP tool called as JSON-RPC through the shell and a member skill's script run directly now count; one read of several `SKILL.md` files is not a use for `must_not_use`; probes the arm cannot satisfy are skipped; probe ids that differ only by spaces, probes that can never pass, and `rule:`/`hook:` refs are rejected at validation instead of silently dropping or never matching; and the CLI and HTML list the probes that failed.
- Plugin signals (checks 15 to 19) count a member skill named after its plugin (a `csv-tidy` skill in a `csv-tidy` plugin) again. Only the generated `<plugin>-plugin-eval` package always means the wrapper; the plugin's own name means it only when no member skill or command has that name.
- Plugin handoffs (check 18) see files that interpreter code fed as a here-document opens (`python3 - <<'PY'` ... `PY`, also through a name such as `out = Path(...)`), and a redirect after a here-document (`cat <<'EOF' > out.json`). A subagent's own calls belong to the `Agent` call whose result names its `agentId`, so a plugin agent started together with another subagent keeps its writes.
- Unprefixed `Write`, `Edit`, and `MultiEdit` refs also match a `Bash` or `exec_command` call that writes a file, so they mean the same on Codex (which writes most files through the shell) as on Claude Code. `apply_patch` still means only the dedicated file tools.
- Tier 3 lift no longer rewards skill activation alone: an arm without the skill or plugin records `skill_execution` and `skill_efficiency` as not applicable, and every lift compares its two arms case by case on the dimensions both scored. The headline lift is the case-weighted paired mean, the same estimate as its interval, so the Integration verdict can no longer contradict its own interval.
- One failed trial (a judge error or a timeout) no longer removes a lift interval or the Integration verdict: the interval pairs the cases both arms scored and says "partial: n of m cases", Integration ignores a failed no-plugin trial, and the reason names the arm that really failed.
- Lift intervals are expanded percentile bootstraps, which no longer undercover at small case counts, and "adequate" precision needs at least 10 paired cases; a 1-case run reports no interval instead of a final-looking zero-width one.
- In `--lift-mode integration` the plugin-vs-member-skills interval is filed under `integration`, and no report calls it Skill Lift.
- Multi-agent plugin runs show one named Integration result per agent.
- The Integration verdict uses its whole interval: real when it clears +0.05, negative below -0.05, cosmetic when it lies inside the band; the readiness gate counts only member skills, a plugin without member skills is told up front, and a refused Integration request says "was not run".
- `evaluate-plugin` shows one lift on the reports' basis and no final-looking headline for an INCOMPLETE run; BENCHMARK's lift row is "Plugin lift", so it is no longer a second "Effectiveness" number.
- The pass@k lift also stops rewarding skill activation alone: `pass_at_k_lift.json`, the paired pass counts and the McNemar test score both arms on the metrics both scored (`basis: shared_metrics`, with `excluded_metrics`), and the Reliability table says when the arms' own pass@k rates use different metrics.
- `tier3 compare` shows N/A lift for the metrics the no-skill arm cannot score and computes its Overall lift on the metrics both arms scored, instead of counting them as 0.0.
- The `validate` Tier 3 panel shows the baseline on the lift's own basis, not the full with-skill score minus the lift.
- A dimension the baseline cannot score says "Not applicable without the skill or plugin" instead of "No baseline run available"; the HTML says "baseline did not complete" for a baseline that ran and failed, and an INCOMPLETE plugin run marks its Skill Lift row "partial run, not final".
- A multi-agent Markdown report names the agent of every lift interval, and BENCHMARK's lift-basis note names only the dimensions that need the plugin installed.
- SARIF `properties.pluginComponent` now names the component a finding belongs to (its MCP server, its tagged LSP server, monitor or hook, or its declared ref) for relative and absolute target paths alike, instead of the first component in the same file or nothing; a path that only a manifest or a shared file gives attributes to no component. Artifact URIs no longer repeat the target or bundled-skill folder, a symlink finding points at the link, and `message.markdown` escapes HTML and Markdown from scanned text.
- Plugin `BENCHMARK.md` path redaction keeps relative paths and closing tags: `./missing-agents/reviewer.md` no longer prints as `.reviewer.md`, "does not start with './'" no longer prints `.redacted-path`, and `</script>` is no longer rewritten to `<script>`, in findings and in the Component Coverage table; absolute local paths are still reduced to their file names.
- Plugin reports no longer say LSP servers, monitors, settings and output styles are "only listed" or that "no check evaluates them": Tier 1 checks them statically, and the Markdown, HTML and `BENCHMARK.md` notes now say so; only types no check reads (such as apps and extensions) are called listed-only.
- Tier 3 coverage has a `not_loaded` state: a component staged natively that the harness reported as not loaded (plugin missing from Claude Code's init event, MCP server `failed` or `pending`) no longer shows as a green "Staged" row; reports headline it ("0 components not staged, 7 not loaded"), list it as staged but not loaded, and SARIF adds `componentsNotLoaded`. In multi-agent runs one agent's listing or load no longer hides another agent's load failure, and a partial load says in how many trials the component was missing and loaded. A hooks file counts as exercised only when every staged handler started, and an exercised hook row no longer reads "not observed". The "Observed activation" summary leaves out subagents and commands that could not be staged (wrapper mode, Codex) instead of counting them as declared and unverified.
- `validate --block-on-agent-eval` now fails on a Tier 3 FAIL verdict (exit 1), as its help says; before, only a skipped or INCOMPLETE Tier 3 run failed and a FAIL verdict exited 0. A NEUTRAL verdict never fails it. Without the flag Tier 3 stays advisory, and the reports now say so: `BENCHMARK.md` reads "FAIL — Not recommended for publication (Tier 3 was advisory in this run)" instead of "Publication blocked", and the terminal footer and summary say the Tier 3 FAIL was advisory instead of "all tiers passed".
- The `validate` footer and terminal summary say INCOMPLETE, not FAIL, when a required check did not complete (such as a `claude plugin validate` crash, timeout, or output with no verdict) or a Tier 3 plugin run was partial. A real failure still shows FAIL and counts what did not complete.
- Tier 3 now applies the Skill Lift band it documents. A lift of -0.10 or less adds a warning, and when its paired-case 95% interval also lies wholly below zero it is a confirmed regression: the run gets a "Skill Lift regression" conclusion, `BENCHMARK.md` says FAIL, and `validate --block-on-agent-eval` exits 1. Before, a lift of -0.20 with the interval [-0.20, -0.19] still passed with no warning. The band (`lift_band` in the JSON) is shown in the terminal verdict line, the HTML overview and BENCHMARK.md; it never changes the dimension verdict, and Integration-only runs have none.
- `BENCHMARK.md` "Blocking Findings" lists only findings that block: CRITICAL and HIGH findings, plus a MEDIUM or LOW that its check made blocking, from results that fail the required gate. MEDIUM and LOW rows of a failed validator and findings of advisory tiers are no longer listed there, and the list says how many blocking findings it does not show instead of stopping at five silently.
- Tier 3 evaluator cards use the dimension verdict's thresholds (pass at 0.50, warn at 0.40), so a 0.56 security card no longer says `fail` beside a PASS security dimension; the HTML gauge color follows the card status. `BENCHMARK.md` no longer says "baseline not run" for an INCOMPLETE run whose baseline arm ran: it says how many baseline attempts scored and that the run is INCOMPLETE.
- A real failure outranks missing evidence in the `validate` footer, the terminal summary and `BENCHMARK.md`: when a check records a blocking finding (such as a HIGH prompt injection) and its scanner also did not complete, they say FAIL and name what did not complete, instead of INCOMPLETE. A run where nothing failed but evidence is missing still says INCOMPLETE. The JSON `overall_status` still reports `incomplete` whenever a required scanner produced no evidence.
- With `--block-on-agent-eval`, a Tier 3 run that never ran (for example, Docker is missing) says FAIL and the reason in the footer, as the JSON, the terminal summary and `BENCHMARK.md` already did; the terminal row and the `BENCHMARK.md` Tier 3 row say it did not run.
- A partial Tier 3 plugin run whose verdict is FAIL, or whose Skill Lift is a confirmed regression, is FAIL on every surface, advisory without `--block-on-agent-eval`: the footer no longer says "all tiers passed", and the card no longer says INCOMPLETE beside a confirmed regression. Without the flag, the footer also says when an INCOMPLETE Tier 3 run was advisory.
- Tier 3 dimension scores use the dimension verdict thresholds (0.50 and 0.40) in the HTML dimension table, the per-agent dimension cells and pills, and the standalone terminal display, so a 0.56 score gets the pass color next to its PASS verdict. `BENCHMARK.md` prints no Tier 3 score or uplift for an agent whose run did not complete; it says how many attempts of each arm scored.
- A wrong-typed `skills`, `commands`, `hooks`, or `mcpServers` value in a Codex manifest is one MEDIUM finding: Codex drops the value, loads the default location, and still installs the plugin. The component path and MCP checks no longer add a HIGH for the same defect. A wrong-typed `apps` value and every Claude Code component value stay HIGH.
- A plugin subagent with no frontmatter gets the same findings as one that omits `tools`: Claude Code loads it under its file name, and it inherits every tool.
- Tier 3 checks a Codex plugin's MCP servers with Codex's own `${VAR}` rules, so a `${VAR:-default}` URL that Codex does not expand is refused at staging, as Tier 1 reports it, with a message about the staged server.
- The dependency audit reads the packages that MCP runners install behind launch wrappers (`cmd /c`, `env X=1`, `timeout`, `sh -c`) and through `bun x`, `uv run --with`, and `pnpm --package=... dlx`, the same way the pinning check does.
- Plugin names longer than 64 characters, which Tier 1 accepts, also work in Tier 2 and Tier 3. The generated wrapper skill name is cut to 64 characters with a short hash of the full name; reports keep the full name.
- SkillSpector findings mask credentials in their code snippet, message, and suggestion, so a token in a plugin script no longer reaches the JSON, SARIF, or HTML report.
- The paired lift statistics and the pass@k lift score a multi-step attempt whose steps mix standard and custom-only metric contracts by its logical overall, the score that pass@k and the report headline use.
- LSP server arguments follow the MCP argv rule: an operator inside one argument is not flagged, and an argument that is only an operator or carries command substitution is a LOW note.
- The MCP and hook shell check also catches `+c` (which bash, sh, and zsh run like `-c`) and `-oe pipefail -c`, where the `o` is not the last letter of the option cluster.
- A floating tag written after an image name taken from the environment (`docker run ${IMAGE}:main`) is HIGH `mcp_command_floating_version`, and each package of `npx -p a@latest,b` is classified.
- An MCP status line after a request line (`GET /repos/x: 404 - Not Found`) or after an error clause (`Error calling tool: 404: not found`) counts the call as failed; the status still needs its reason phrase, so a server's own `Error: 500 - boom` is an answer. An HTTP response status line (`HTTP/1.1 404 - x`, `HTTP/2 500`) fails without one.
- An MCP server whose `args` is not a list no longer crashes the Cursor `${env:NAME}` check; it gets HIGH `mcp_args_not_list`.
- A git spec on a moving branch or tag is HIGH `mcp_command_floating_version` again, as it was before the pinning rewrite: `uvx --from git+https://github.com/o/r@main`, `@latest`, `@next`, and npm's `github:o/r#main`. A fixed tag or no ref stays MEDIUM `mcp_unpinned_package`.
- `validate <relative folder>` lists each blocking finding once again. Making finding paths relative to the content root left the matching `errors` and `warnings` strings on the old path, so CLI and Markdown reports printed them a second time in an extra `Errors:` list, and the JSON `errors` kept the old path. Those strings now move with their finding.

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
