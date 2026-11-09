# Reports and CI

## JSON fields to read

From `skillevaluator-output-<timestamp>.json` (or `skillevaluator-tier1.json`):

| Field | Meaning |
| --- | --- |
| `overall_passed`, `overall_status` | Command gate: `passed`, `failed`, `incomplete` |
| `publication_status` | `pass`, `neutral`, `fail`, `incomplete`; use for publishing |
| `quality_summary[].overall_score`, `.grade` | Quality 0-100 and A-F |
| `severity_counts` | `critical`, `high`, `medium`, `low` totals |
| `finding_groups[]` | One entry per unique finding (`finding_id` `SEF-...`) |
| `tier3.run_id` | Run identity for the Tier 3 section |

`overall_passed` can be true while an advisory tier did not run; require
`publication_status == "pass"` before claiming a skill is publishable.

Example gate on a stricter score:

```bash
report=$(ls -t <report-dir>/skillevaluator-output-*.json | head -1)
jq -e '([.quality_summary[]?.overall_score] | min // 100) >= 80' "$report"
```

`validate` on a skill also writes `BENCHMARK.md`: verdict first, the three
tier states, and `Baseline -> Skill Uplift` columns.

## CI

Guided GitHub setup from a checkout: `skillevaluator ci init` (then
`ci status`). Manual: copy the GitHub or GitLab template from the
repository's `ci/bundles/<version>/` folder.

Recommended keyless gate:

```bash
skillevaluator validate <skills-dir> --profile default --tiers 1 -c \
  -r json,markdown,sarif -o reports --min-score 70
```

Adopt progressively: run with `|| true` (advisory), then gate on Tier 1,
raise `--min-score`, add `--llm --tiers 1,2` with a masked provider secret,
and finally an advisory Tier 3 job (`--block-on-agent-eval` when ready).
Keep keys in CI secrets exposed through the job environment, never in the
repository. Pin policy with `--policy <policy.yaml>` so severity changes are
reviewed like code.
