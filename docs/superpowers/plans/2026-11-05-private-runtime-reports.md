# Private Runtime Reports Implementation Plan

> **For agentic workers:** Use the existing execution workflow, focused helpers and an independent final review. Track completed checks in the task ledger.

**Goal:** Complete item 24 with consistent private runtime views and durable offline replay.

**Architecture:** Reduce existing results/provenance once into an immutable sanitized display projection. Retain it before cleanup and render it through the existing six private report adapters.

**Tech Stack:** Python 3.11+, dataclasses, shared secure filesystem/JSON readers, Jinja and existing reporters; no added dependency.

**Spec:** `docs/superpowers/specs/2026-11-05-private-runtime-reports-design.md`

## Global Constraints

- Public CLI/configuration/detection/exports remain disabled until item 25.
- Keep unknown values distinct from zero; staging never proves loading or execution.
- Reports are passive and advisory and do not alter scores, gates or source results.
- Redact before truncating, escape each output format and bound input/output work.
- Use existing statistics without recomputing lifts, eligibility, confidence intervals or costs.

## Review Focus

- Missing trajectories/startup evidence must remain incomplete despite successful execution.
- Missing or unsupported arms must retain their measurement/exclusion reasons.
- Hostile strings, malformed values and report budget exhaustion must remain safe and explicit.
- Saved snapshots must replay after source/jobs removal and refuse unsafe or malformed files.
- Publication failure must preserve callbacks, evaluation status and cleanup.

### Task 1: Projection and retained snapshot

Files: new `reporting/plugin_runtime.py`, `reporting/plugin_runtime_io.py`;
modify `tier3/plugin_execution.py`; new focused reporting/lifecycle tests.

Interfaces: `project_plugin_runtime(result, *, provenance=None) -> RuntimePluginReport | None`;
`RuntimePluginReport.payload() -> dict`; `save_plugin_runtime_report(report, run_dir)`;
`load_plugin_runtime_report(run_dir) -> RuntimePluginReport`.

- [x] Write failing tests for run identity, load states, coverage/signals/MCP, comparisons, measured costs, unknowns and explicit omissions.
- [x] Implement fixed-schema bounded projection using existing sanitization and section budgets.
- [x] Test strict no-follow, bounded and version/schema-checked snapshot round trips, including malformed/oversized/missing/link inputs.
- [x] Retain the snapshot in the private publication callback, keeping failures advisory and caller publication independent.
- [x] Run focused projection, replay and lifecycle regressions.

### Task 2: Six report formats

Files: new `reporting/plugin_runtime_render.py`, runtime HTML template and tests;
minimal reuse/refactor of `reporting/plugin_static_render.py`.

Interface: `render_plugin_runtime_reports(results, *, report, reporters) -> dict[str, str]`.

- [x] Write failing cross-format tests for identical payloads, escaping, missing evidence, combined static/runtime output and ordinary-output compatibility.
- [x] Render the same projection through all six existing reporters, with no new collection or public exports.
- [x] Run runtime/static reporting tests and check a generated HTML report visually.

### Task 3: Assembled acceptance and delivery

Files: new `tests/tier3/test_private_runtime_report_acceptance.py`, private usage notes and release note.

- [x] Exercise real discovery, deterministic validation, preparation, service/runner and collection with local execution fixtures: five cross-component cases, two attempts, four arms.
- [x] Verify saved replay after source/tasks/jobs removal, shared comparison estimates, missing evidence and ordinary-run isolation.
- [ ] Run `make lint`, `make test`, `make build`; verify installed base-only replay and private CLI boundaries.
- [ ] Independently review, fix actionable findings, sign off commits, push and open/attach a focused PR with exact validation limits.
