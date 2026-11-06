# validate command reference

The combined pipeline. `skillevaluator validate --help` is authoritative.

## validate

`skillevaluator validate PATH` runs Tier 1, Tier 2 and Tier 3 (with
autopilot) for skills. Tier 1 always gates; Tier 2 gates unless
`--no-block-on-dedup`; Tier 3 gates only with `--block-on-agent-eval`. A
folder without a root `SKILL.md` validates each skill inside it.

| Flag | Effect |
| --- | --- |
| `--tiers 1`, `--tiers 1,2`, `--tiers 1,3` | Scope; Tier 1 always included |
| `--type skill\|rules\|workflows\|plugin\|auto` | Force the content type |
| `--checks`, `--skip` | Tier 1 selection (`dependency` is opt-in here; `--checks` replaces the list) |
| `--llm`, `--llm-verify` | LLM security analysis and false-positive suppression |
| `--min-score`, `--quality-config`, `--pii-waivers`, `--skillspector-baseline FILE` | Tier 1 tuning |
| `--previous-version VERSION` | Require a greater `metadata.version` (single skill only) |
| `--fail-fast`, `-c, --continue-on-failure` | Stop early, or record every issue (recommended in CI) |
| `--no-dedup`, `--no-block-on-dedup` | Skip Tier 2, or make it advisory |
| `--no-tier3`, `--no-autopilot`, `--block-on-agent-eval` | Tier 3 scope and gating |
| `-a`, `--env-mode`, `--n-attempts`, `--max-cases`, `--agent-model`, `--harbor-keep-jobs`, ... | Forwarded to Tier 3 |
| `--workers N` | Validate N skills of a folder in parallel |
| `--profile`, `--policy FILE` | Policy profile and overlay |
| `--evaluated-source-repository`, `--evaluated-source-revision` | Provenance for `BENCHMARK.md` |
