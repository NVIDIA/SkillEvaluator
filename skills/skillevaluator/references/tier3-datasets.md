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
      "assertions": ["Read my-skill/SKILL.md", "Ran scripts/convert.py"],
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
behavior per assertion.
