# Tier 1 command reference

Report options and Tier 1 commands. `skillevaluator <command> --help` is authoritative for the installed version.

Global options go before the command: `skillevaluator -v validate <skill-dir>`.

## Reports (Tier 1 and Tier 2 commands)

| Flag | Default | Effect |
| --- | --- | --- |
| `-r, --report` | `cli,json,html` for `tier1 PATH`, `tier2 PATH`, `validate`; `cli` for focused commands | `cli`, `json`, `html`, `markdown`, `sarif`; comma-separated or repeated |
| `-o, --output-dir` | `reports` | Report directory |

Passing `-r` is honored exactly: `-r cli` writes no files. Basenames:
`validate` writes `skillevaluator-output-<timestamp>.*`; `tier1`/`tier2` write
`skillevaluator-tier1.*`/`skillevaluator-tier2.*`; focused commands write
`skillevaluator-<kind>.*` (for example `skillevaluator-quality.json`) and
overwrite earlier runs.

## Direct tier workflows

A bare `tier1`, `tier2`, or `tier3` prints help. Prefix a path with `./` when
its name matches a subcommand (`tier1 <path>/validate`).

`skillevaluator tier1 PATH`: all static checks including `dependency`, plus
rubric, LLM security and finding verification when a provider is configured.

| Flag | Default | Effect |
| --- | --- | --- |
| `--checks TEXT` | all applicable | Subset: `schema`, `version`, `security`, `pii`, `license`, `code-integrity`, `unicode`, `quality`, `lint`, `dependency` |
| `--skip CHECKS` | none | Leave checks out (also `secrets` for Gitleaks); the run cannot certify Tier 1 |
| `--llm / --no-llm` | on when a provider is configured | Explicit `--llm` requires a valid provider |
| `--min-score INTEGER` | `70` | Quality and rubric minimum |
| `--quality-config PATH` | packaged | Replace expected-heading rules |
| `--pii-waivers PATH` | none | Reviewed, expiring PII waivers |
| `--profile TEXT` | `default` | Validation policy |

`skillevaluator tier2 PATH`: intra-skill duplicate analysis of one skill.
`--catalog FILE` adds inter-skill comparison (build it with
`skillevaluator similarity-check <skills-dir> --save-catalog <catalog.json>`);
`--threshold` (0.80), `--similarity-threshold` (0.75), `--full-body`.

`skillevaluator tier3 PATH`: generates one starter case when no source
exists, then runs the provider-native agent in Docker with and without the
skill. Takes every `tier3 evaluate` option. Does not accept `-r`/`-o`.

## Focused Tier 1 commands

| Command | Key flags |
| --- | --- |
| `quality-check PATH` | `--min-score` (70), `--quality-config` |
| `rubric-eval PATH` | `--min-score` (70); every criterion must reach 7/10 |
| `security-scan PATH` | `--llm`, `--llm-verify`, `--skillspector-baseline` |
| `pii-scan PATH` | `--llm-verify`, `--pii-waivers` |
| `lint-scripts PATH` | report options only; advisory |
