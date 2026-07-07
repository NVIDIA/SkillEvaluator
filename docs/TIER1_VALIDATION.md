# Tier 1: Static and Security Validation

Tier 1 is the deterministic quality gate: schema, quality, security, PII,
license, secrets, code-integrity, Unicode-safety, and script checks that run
offline after the scanner binaries are installed. No API key is needed for the
deterministic stages. The only LLM-backed
pieces of Tier 1 are `rubric-eval` (LLM required) and the optional
`--llm`/`--llm-verify` flags — those need a configured provider
(see [CONFIGURATION.md](CONFIGURATION.md)).

## Commands

```bash
skillevaluator validate ./my-skill            # full Tier 1 + Tier 2 dedup
skillevaluator validate ./my-skill --no-dedup # Tier 1 only, no key needed
skillevaluator quality-check ./my-skill       # quality score only
skillevaluator security-scan ./my-skill       # security checks only
skillevaluator pii-scan ./my-skill            # PII checks only
skillevaluator lint-scripts ./my-skill        # advisory script lint

# LLM-backed (needs a provider key):
skillevaluator rubric-eval ./my-skill         # LLM-as-judge rubric scoring
skillevaluator validate ./my-skill --llm      # + LLM security analysis
```

`validate` is the umbrella command: Tier 1 checks gate the exit code
(non-zero on failure or incomplete required scanner evidence — safe to use as
a CI gate), Tier 2 dedup runs by
default and degrades gracefully without embedding access, and Tier 3 can be
attached with `--agent-eval` as an advisory pass.

## Selecting checks

`--checks` takes a comma-separated subset of the Tier 1 checks:

| Check | What it covers |
| --- | --- |
| `schema` | SKILL.md frontmatter, repository governance, naming |
| `security` | SkillSpector static scan; `--llm` adds enrichment without replacing static findings |
| `pii` | Personally identifiable information |
| `license` | License compliance |
| `code-integrity` | Bandit, packaged-rule Semgrep, external Gitleaks, and dead-link/dependency-file/static-test-discovery hygiene |
| `unicode` | Unicode smuggling detection |
| `quality` | Quality score (skill-only; threshold via `--min-score`, default 70) |
| `lint` | Script linting (skill-only) |
| `version` | Opt-in: version checks (not run by default) |
| `dependency` | Opt-in: dependency vulnerability audit (not run by default) |

Default Tier 1 does not import or execute target-controlled Python code. The
`test_discovery` result is filename evidence only: it counts regular, in-tree,
non-symlink `test_*.py` and `*_test.py` candidates without parsing or importing
them. It reports `execution_performed=false` and `coverage_measured=false`;
test success and code coverage must be established separately in a trusted
project environment or an explicit sandbox.

The default external profile treats a missing, timed-out, crashed, or malformed
Bandit, Semgrep, Gitleaks, or SkillSpector run as `INCOMPLETE`: terminal and
saved reports are non-green and `validate` exits non-zero. Install
`skillevaluator[security]` for the bundled Python scanners and install Gitleaks
separately (`brew install gitleaks` or
`go install github.com/gitleaks/gitleaks/v8@latest`).

Semgrep uses `skillevaluator/config/semgrep_rules.yaml` from the installed
package, with metrics and version checks disabled; Tier 1 never fetches a
Semgrep registry policy at runtime. Bundled scanners resolve next to the
SkillEvaluator interpreter before PATH. Intentional replacements require an
auditable absolute executable path in `SKILLEVALUATOR_BANDIT_PATH`,
`SKILLEVALUATOR_SEMGREP_PATH`, or `SKILLEVALUATOR_SKILLSPECTOR_PATH`; relative,
missing, and non-executable overrides fail closed.

`rubric-eval` derives each criterion verdict locally: scores of 7/10 or higher
pass, and lower scores fail regardless of the model-provided pass flag. The
overall score is an importance-weighted mean, and the rubric passes only when
that score meets `--min-score` and every criterion passes.

Other useful flags: `--fail-fast` stops at the first failing check,
`--continue-on-failure` records everything without stopping, `--llm` enables
LLM-backed security analysis (requires a provider; add `--llm-verify` for a
false-positive suppression pass), and `--profile external` (the default)
validates for public publication. A custom policy YAML can be overlaid with
`--policy`.

## Content types

`validate` auto-detects what it is looking at; force it with `--type`:

| Type | Detected from |
| --- | --- |
| `skill` | `SKILL.md` in `skills/` or `team-skills/` |
| `rules` | `.mdc` files in `team-rules/` |
| `workflows` | `workflow-rules.mdc` in a workflow directory |
| `plugin` | `agent_plugin.yaml`/`.yml` or `.claude-plugin/plugin.json` manifest |

Quality, lint, and version checks are skill-only and are skipped for the
other types.

## Reports

`-r`/`--report` selects one or more formats; `-o`/`--output-dir` sets the
destination (default `reports/`):

| Format | Output |
| --- | --- |
| `cli` | Rich terminal table (default) |
| `json` | `skillevaluator-output-<timestamp>.json` |
| `html` | Standalone `skillevaluator-output-<timestamp>.html` |
| `markdown` | `skillevaluator-output-<timestamp>.md`, ready for PR comments |

For skill targets, `validate` also writes `BENCHMARK.md` regardless of the `-r`
selection. It records `INCOMPLETE` and never recommends publication when
required scanner evidence is unavailable.

```bash
skillevaluator validate ./my-skill -r cli,json,html -o reports/
```

## Using validate in CI

`validate` returns a non-zero exit code when a Tier 1 gate fails, and the
markdown report drops straight into a PR comment:

```yaml
- name: Validate skill
  run: |
    skillevaluator validate ./my-skill --no-llm --no-dedup -r cli,markdown -o reports
```

Run it with `--no-llm --no-dedup` for a hermetic, key-free gate, or configure
a provider secret and drop those flags for full coverage.

## More commands

- `skillevaluator tier1|tier2|tier3 <command>` — expert alias groups. The
  Tier 3-only members (`tier3 validate`, `tier3 harbor-view`) are covered in
  the [live evaluation guide](TIER3_LIVE_EVALUATION.md).
- `skillevaluator health-check` and `doctor` probe Tier 3 agent/backend
  readiness; they are not Tier 1 validation commands.
