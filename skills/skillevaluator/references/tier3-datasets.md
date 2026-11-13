# Tier 3 evaluation datasets

Tier 3 runs a real agent on each case with the skill installed and without it (baseline). This file covers the `<skill-dir>/evals/` folder and dataset format.

## The <skill-dir>/evals/ directory

| Path | Purpose |
| --- | --- |
| `<skill-dir>/evals/evals.json` | Required dataset (`.jsonl`, `.yaml`, `.yml` also accepted) |
| `<skill-dir>/evals/config.yml` | Run policy |
| `<skill-dir>/evals/EVAL.md` | Guidance for `create-eval-dataset` (`## Questions`, `## Behaviors`, `## Negative Cases`, `## Notes`) |
| `<skill-dir>/evals/files/` | Fixtures staged into `/workspace/input/` |
| `<skill-dir>/evals/environment/` | `Dockerfile`, `docker-compose.yaml` sidecars, `mcp_servers.toml` |
| `<skill-dir>/evals/harbor/` | Native Harbor tasks (BYOT) |
| `<skill-dir>/evals/grader.py` or `grader.sh` | Custom grader (BYOG) |
| `<skill-dir>/evals/results/` | Output only; never edit or commit |

## Dataset format

```json
{
  "skill_name": "my-skill",
  "evals": [
    {
      "id": "my-skill-001",
      "prompt": "Convert report.docx in /workspace/input to PDF",
      "expected_output": "A PDF of the document saved next to the input",
      "assertions": ["Saves report.pdf next to report.docx", "Keeps the document's headings in the PDF"],
      "expected_skill": "my-skill",
      "files": ["files/case-001"]
    },
    {
      "id": "my-skill-002-negative",
      "prompt": "What is the capital of France?",
      "expected_output": "Answers directly without loading my-skill",
      "expected_skill": null
    }
  ]
}
```

Required: `skill_name`, `evals`, and per case `id` (letters, digits, `.`,
`_`, `-`; at most 128 chars, unique), `prompt`, `expected_output`. Optional:
`assertions`, `expected_skill` (`null` for negatives), `expected_script`,
`acceptable_skills`, `files` (paths relative to `<skill-dir>/evals/`; omit it and the
whole `<skill-dir>/evals/files/` is staged). The legacy flat array with `question`,
`ground_truth`, `expected_behavior` still runs with a deprecation warning.

Good datasets cover four buckets: explicit (names the skill), implicit
(describes the task), contextual (task inside a project), negative (should
not trigger). Write `expected_output` as an observable outcome and one
behavior per assertion. The baseline is graded on the same prompt,
`expected_output` and assertions without the skill, so keep them about the task
outcome: never point the prompt at the skill's files or assert that the agent
read `SKILL.md`, ran the skill's scripts or used the skill (`skill_execution`
scores that). `tier3 validate` warns about common wordings of these; it is a
heuristic, so review the cases too.

## Guided generation

Standalone `create-eval-dataset` defaults to `--grounding auto`: prefer author
guidance in `--prompt` / `<skill-dir>/evals/EVAL.md` with real content beyond
headings, then usable matching retained trajectories from this skill's completed
runs. Write task requests and observable outcomes in the guidance file first:

```bash
skillevaluator create-eval-dataset <skill-dir> --full --prompt <skill-dir>/evals/EVAL.md
```

If neither guidance nor matching retained traces exists, generation fails before
provider access or writes. `--refine` and `--no-llm` do not bypass that preflight.
For the prior skill-only starter workflow, explicitly opt in:

```bash
skillevaluator create-eval-dataset <skill-dir> --full --no-llm --grounding skill
```

Auto leaves an existing dataset unchanged unless you pass `--force` or
`--refine`, even when matching traces exist. To refine from completed results:

```bash
skillevaluator create-eval-dataset <skill-dir> --refine --from-results <run-dir>
```

Auto trace refinement preserves existing authored case IDs and expectations;
unmatched cases and observed answer proposals need review. When auto selects
traces, proposals that describe skill use leave `expected_output` empty for an
independent authored outcome. Existing authored expectations are preserved. Explicit
`--grounding trace` still opts into in-place refinement without `--refine`.
Generation provenance records the effective `intent` or `trace` grounding and input hashes. Explicit
`intent`, `trace`, and `skill` modes remain available. Unattended autopilot retains
its existing skill-grounded one-case starter contract.
