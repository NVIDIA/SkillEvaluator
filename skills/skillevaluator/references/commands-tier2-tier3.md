# Tier 2, Tier 3 and setup command reference

`skillevaluator <command> --help` is authoritative.

## Tier 2 commands

| Command | Key flags |
| --- | --- |
| `context-optimization-check PATH` (alias `dedup-scan`) | `--threshold` (0.8), `--model`, `--llm-model` |
| `similarity-check PATH` | `--threshold` (0.75), `--full-body`, `--catalog`, `--save-catalog`, `--max-entries` |

## Tier 3 commands

| Command | Purpose |
| --- | --- |
| `tier3 evaluate PATH` | Live A/B run; add `--autopilot` to generate a missing case |
| `tier3 validate PATH` | Check `<skill-dir>/evals/` (`--strict`, `--json`, `--harbor-contract`) |
| `create-eval-dataset PATH` | Write `<skill-dir>/evals/evals.json` (`--grounding auto` requires substantive author guidance or usable matching retained traces; existing datasets need `--force` or `--refine`; opt in to `--grounding skill` for skill-only generation; `--full`, `--no-llm`, `--dry-run`, `--force`, `--prompt`, `--refine`) |
| `init-custom-grader PATH` | BYOG starter (`--mode`, `--language`) |
| `init-harbor-task PATH` | BYOT starter (`--case-id`, `--with-config`) |
| `tier3 select-decoys PATH` | Save neighbor skills for group workspaces |
| `view PATH`, `compare PATH` | Open latest report; compare agents |
| `tier3 feedback RUN_DIR` | Chat about one completed run (`--skill`) |
| `harbor-view JOBS_DIR` | Browse jobs kept with `--harbor-keep-jobs` |
| `dashboard [PATHS]` | Tables over saved reports (`dashboard` extra) |

Key `tier3 evaluate` flags: `-a, --agents` (`opencode`, `codex`,
`claude-code`), `--env-mode` (`docker` default, `local`, `kata`, `gke`,
`daytona`, `e2b`, `modal`, ...), `--n-attempts`, `--pass-threshold`,
`--stop-on-pass`, `--n-concurrent`, `--max-agents`, `--max-cases`, `--model`,
`--agent-model AGENT=MODEL`, `--skip-baseline`, `--skip-with-skill`,
`--grading-mode default|default_plus_custom|custom_only`, `--results-dir`,
`--include-skills`, `--skill-workspace-mode isolated|group`, `--copy-repo`,
`--timeout-multiplier`, `--harbor-keep-jobs`, `--agent-runtime-preflight`,
`--override-cpus`, `--override-memory-mb`, `--progress`, `--verbose`.

## Readiness and setup

| Command | Purpose |
| --- | --- |
| `health-check [--tier 1\|2\|3\|all]` | Offline readiness; `--checks` for Tier 1 scanners |
| `doctor [--tier ...] [--env-mode M] [-a AGENTS] [--verify-models]` | Deeper readiness; `--verify-models` makes live catalog probes |
| `models [--limit N] [--json]` | List provider catalog models |
| `ci init`, `ci status`, `ci demo` | Guided GitHub CI setup |
