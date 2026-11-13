# Running and scoring Tier 3

Run policy, run sequence, scoring and cost control.

## config.yml

```yaml
schema_version: 1
harbor:
  n_attempts: 3
  pass_threshold: 0.5
  n_concurrent: 2
  timeout_multiplier: 1.5
  agents:
    opencode:
      model: your-model-id
skill_workspace:
  mode: isolated   # or group, with include: [...]
grading:
  mode: default    # default | default_plus_custom | custom_only
```

Unknown keys fail the load. CLI flags override config values.
`pre_agent_setup` runs only with `--skip-baseline`. Never put credentials in
`runtime_env`.

## Run sequence

Write task requests and observable outcomes in `<skill-dir>/evals/EVAL.md`
before generation; a file containing only headings does not qualify. For an
existing dataset, use `--refine` with matching retained results to refine it,
or `--force` to replace it after reviewing a `--dry-run` preview.

```bash
skillevaluator doctor --env-mode docker --verify-models
skillevaluator create-eval-dataset <skill-dir> --full --prompt <skill-dir>/evals/EVAL.md
skillevaluator tier3 validate <skill-dir> --strict
skillevaluator tier3 evaluate <skill-dir> -a opencode --max-cases 1   # smoke
skillevaluator tier3 evaluate <skill-dir> -a opencode --n-attempts 3
skillevaluator view <skill-dir>
```

Results land in `<skill-dir>/evals/results/<timestamp>_<pid>_<id>/` with `result.json`,
`run_config.json`, `report.html` and per-agent `lift.json`,
`with-skill/`, `without-skill/`. Use `--results-dir` to write elsewhere.

## Scoring

Standard grading scores five dimensions on 0.0-1.0 and averages them:

| Dimension | Source |
| --- | --- |
| Security | Deterministic trace checks (destructive commands, uploads) |
| Correctness | LLM judge of the answer (`accuracy`) |
| Discoverability | Whether the skill was loaded when expected (`skill_execution`) |
| Effectiveness | LLM judges `goal_accuracy` and `behavior_check` |
| Efficiency | Wasted tool or skill calls (`skill_efficiency`) |

Bands: PASS >= 0.50, NEUTRAL 0.40-0.50, FAIL < 0.40. Verdict PASS needs one
agent with every dimension passing. Lift band: PASS >= +0.05, FAIL <= -0.10.
pass@k counts a case passed when any attempt reaches `--pass-threshold`.

## Cost control

Trials = cases x attempts x agents x 2 arms. Use `--max-cases`,
`--skip-baseline` (no lift), `--stop-on-pass`, and `--n-concurrent` to
control spend. Runs that skip cases are never publication evidence.
