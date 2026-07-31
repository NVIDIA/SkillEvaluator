---
name: create-custom-grader
description: Use when converting an existing benchmark, rubric, verifier, task YAML/JSON, or domain check into SkillEvaluator BYOG/BYOT custom evaluation.
metadata:
  author: SkillEvaluator Maintainers <maintainers@example.com>
---

# Create Custom Grader

Convert team-owned benchmark definitions into runnable SkillEvaluator custom graders
and, when needed, native Harbor tasks.

## Purpose

Help an agent author valid SkillEvaluator BYOG/BYOT files from a user's
benchmark instead of leaving the user with empty grader templates.

## When To Use

Use this skill when the user wants to:

- bring an existing benchmark into SkillEvaluator
- turn a rubric into `evals/grader.py` or `evals/grader.sh`
- add custom metrics beside the default evaluator metrics
- convert task files such as `task.yaml`, `task.json`, pytest checks, or shell
  verifiers into BYOG or BYOT
- prove a team can run its own benchmark through SkillEvaluator

Do not use this skill for ordinary `evals/evals.json` authoring when no custom
grading logic is needed. Use the normal dataset authoring workflow for that.

## Instructions

1. Read the target skill, existing `evals/`, benchmark prompts, fixtures, and any verifier code.
2. Choose `default_plus_custom` when custom metrics should complement default evaluator scoring.
3. Choose `custom_only` only when the user wants the custom grader to own pass/fail semantics.
4. Write or update `evals/grader.py` or `evals/grader.sh`, then validate the Harbor contract.

## Examples

```bash
skillevaluator init-custom-grader <skill-dir> --language python --mode default_plus_custom
skillevaluator tier3 validate <skill-dir>
```

## Prerequisites

- The target skill directory should contain `SKILL.md`.
- The SkillEvaluator CLI should be available as `skillevaluator`.
- Full E2E evaluation may need agent credentials, sandbox access, GPU access, or
  service credentials depending on the benchmark.

## Core Choice

Choose one path before writing files:

| User need | Evaluator shape |
| --- | --- |
| Existing `evals.json` task plus extra domain checks | Top-level BYOG: `evals/grader.py` or `evals/grader.sh` |
| Existing benchmark prompt/rubric that can run in the generated workspace | Top-level BYOG plus `evals/evals.json` and `evals/files/` |
| Benchmark owns task layout, setup, service lifecycle, or verifier harness | Native BYOT/BYOG: `evals/harbor/<case>/...` |
| User wants only custom reward/pass criteria | `grading.mode: custom_only` |
| User wants default evaluator dimensions plus custom metrics | `grading.mode: default_plus_custom` |

Default to `default_plus_custom` unless the user explicitly wants the custom
grader to replace the default evaluator metrics.

## Workflow

1. Resolve the target skill and benchmark source.
   Read the target `SKILL.md`, existing `evals/`, benchmark prompts, fixtures,
   rubric, reference solution, tags, and any expected trigger/non-trigger
   metadata.

2. Map benchmark fields into evaluator inputs.
   Use benchmark prompts or prompt variants as `question` entries. Use the
   target skill as `expected_skill`. Put required starter files under
   `evals/files/`. Preserve benchmark-specific rubric text in the entry only
   when the grader needs to read it.

3. Scaffold the evaluator contract.
   For generated tasks:
   ```bash
   skillevaluator init-custom-grader <skill-dir> --language python --mode default_plus_custom
   ```
   For shell checks:
   ```bash
   skillevaluator init-custom-grader <skill-dir> --language shell --mode default_plus_custom
   ```
   For native Harbor tasks:
   ```bash
   skillevaluator init-harbor-task <skill-dir> --case-id <case-id> --with-config
   ```

4. Replace scaffold placeholders.
   The custom grader is real executable logic, not metadata. It must read
   available evidence, compute numeric scores, and write the evaluator reward
   contract.

5. Validate before running.
   ```bash
   skillevaluator validate <skill-dir> --harbor-contract
   ```
   Fix missing files, invalid Python, missing reward output, and native Harbor
   ID mismatches before evaluation.

6. Run the deepest practical proof.
   Prefer a real with-skill/baseline run. If services, credentials, GPU, or
   cost block full E2E, state exactly what was validated and what was not.

## Grader Contract

Python and shell graders run inside the Harbor verifier context. They may read:

- `/logs/agent/trajectory.json` for agent actions and final answer evidence
- `/tests/entry.json` for the eval case metadata
- `/workspace/input/` for committed fixture files from `evals/files/`
- `/solution/` or other task outputs only when the task environment produces
  them

They must write:

- `/logs/verifier/reward.json`
- `/logs/verifier/reward.txt` with a numeric score from `0.0` to `1.0`

Use this reward shape:

```json
{
  "overall": 0.92,
  "custom_metrics": {
    "domain_repair": 1.0,
    "domain_verification": 0.8
  },
  "details": {
    "domain_repair": {
      "score": 1.0,
      "reason": "The solution repaired the required files."
    }
  }
}
```

In `default_plus_custom`, default evaluator scoring keeps its `overall`
authoritative and adds the grader's `custom_metrics` into reports. In
`custom_only`, the grader's `overall` is the pass/fail reward.

Never emit custom metric names that collide with reserved evaluator fields:
`security`, `skill_execution`, `skill_efficiency`, `accuracy`,
`goal_accuracy`, `behavior_check`, `overall`, `details`, `metrics`,
`metric_set`, or `entry_id`.

## Translation Rules

- Convert each rubric item into a deterministic check when possible.
- If a rubric item requires judgment, encode observable proxies and explain the
  limits in `details`.
- Keep metrics stable across baseline and with-skill runs.
- Score only the generated task workspace. Do not accidentally score copied
  skill source files, reference fixtures, or grader templates.
- Keep custom metric values clamped to `0.0` through `1.0`.
- Preserve benchmark prompt variants as separate eval entries only when they
  exercise meaningfully different behavior.
- Convert expected trigger/non-trigger metadata into `expected_skill`,
  `expected_behavior`, negative cases, or custom metrics that inspect
  trajectory evidence.

## RAPIDS-Style Example

For a benchmark task with `task.yaml`, `code/`, prompt variants, coverage, and a
rubric:

1. Copy `code/` into `evals/files/<case-id>/`.
2. Create one or more `evals/evals.json` entries from the prompt variants.
3. Set `expected_skill` to the benchmark's target skill.
4. Implement `evals/grader.py` to inspect the agent trajectory and changed
   workspace files.
5. Emit custom metrics for each rubric criterion, for example
   `rapids_diagnosis`, `rapids_requirements_repair`,
   `rapids_repair_safety`, and `rapids_verification`.
6. Validate and run SkillEvaluator with and without the target skill, then
   report both default evaluator metrics and custom metric deltas.

## Limitations

- The skill can design and implement deterministic checks, but ambiguous rubric
  judgment still needs explicit observable proxies or a human-approved scoring
  policy.
- `init-custom-grader` creates scaffolding only; the agent must replace the
  placeholder scoring logic.
- Local validation proves file contracts, not live agent behavior. Do not call
  the benchmark proven until an evaluation run has produced real rewards.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `evals/evals.json` missing | Create entries from the benchmark prompt or run `init-custom-grader` to seed one. |
| Custom metrics do not appear | Ensure `reward.json` has numeric values under `custom_metrics` and no reserved-name collisions. |
| `custom_only` fails | Write numeric `overall` in `reward.json` or numeric `reward.txt`. |
| Grader scores copied fixtures | Restrict file searches to generated workspace/output paths, not the skill package or grader source. |

## Final Response

When finished, report:

- files created or changed
- exact validation and evaluation commands
- default evaluator metric results
- custom metric results
- whether the proof was full E2E or only static/local validation
- any benchmark rubric criteria that remain partly judgment-based
