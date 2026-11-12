# Guided dataset grounding default

The approved standalone `create-eval-dataset` default becomes `auto`: prefer nonblank author guidance (`--prompt`, then `evals/EVAL.md`), otherwise require usable retained trajectories matching existing authored case IDs or deterministic template seed IDs. Fail before provider access, trajectory collection, or writes when neither is available; explain how to supply guidance, retained results, or explicit `--grounding skill`.

## Approaches

Changing every generator including unattended autopilot would broaden the migration. Changing only the standalone command, its service options, and generator entry point preserves unattended behavior and is the approved scope. Retain the existing explicit `skill`, `intent`, and `trace` behavior and `--refine` collection semantics after successful auto preflight. Auto itself never collects evidence to satisfy its input requirement.

Auto trace selects only completed results scoped to this skill, with meaningful matching steps and retained source files. Preserve existing authored cases and expectations; deterministic seeds keep discovery independent of provider output. Flag observed expectations and unmatched cases for review. Record the effective `intent` or `trace` grounding and source hashes through existing provenance. No schema, scoring, or acceptance policy changes. `generate_one_case` and unattended autopilot keep recorded skill grounding.
