# Report Clarity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** consolidate duplicate report information and quietly retain complete native features/evidence, then open a structured before/after PR on the existing branch.

**Architecture:** keep the existing renderer and canonical/display boundary. Independent bounded telemetry and plugin presentation helpers feed the single native template; a concise summary projection supplies Overview without duplicating full Insights. Routine metadata lives in the closed footer.

**Tech Stack:** Python 3.12+, Jinja2, embedded Chart.js 4.5.1, development-only Playwright/Chromium.

**Spec:** `docs/superpowers/specs/2026-11-14-report-clarity-design.md` (user approved).

## Global Constraints

- One renderer, unchanged scoring/gates/CLI/exports and canonical Tier 3 object; preserve all existing output safety and bounds.
- Native typography/palette, 16px body and Chart.js 4.5.1 remain. All tiers, six subtabs, evidence and controls stay accessible.
- At most three primary Overview metric tiles: Score including agent, Lift, Coverage; all-missing quality uses one unavailable state plus coverage/action.
- One full Insights owner. No arbitrary prose splitting/paraphrasing, lost trace associations or provider calls during rendering.
- Passive plugin sections default closed, preserve stable IDs and snapshots, distinguish omitted rows/display shortening/pre-retention clipping.
- Keep validation, quality, execution, publication and completed-agent scopes distinct, with visible blockers and comparison limitations.
- Preserve deep links/search/keyboard/print/no-JS/offline/mobile behavior, CI direct canonical artifact links and installed-wheel packaging.
- Synthetic committed fixtures only; private reports/artifacts remain ignored. No fresh evaluations or rescoring.
- Same branch `codex/report-gitlab-parity`; sign off commits; push feature branch and create PR only after local review/verification. No merge.

## Review Focus

- Mixed completed/failed agents, source-preflight and skipped scopes must not become a whole-run pass or fabricated quality score.
- Folded empty/plugin/evaluator evidence must remain searchable and reachable by existing deep links, including static-only reports.
- Legacy native and current telemetry schemas must preserve partial/unknown versus zero, cached accounting and unavailable deltas.
- Fewer visible tiles must retain every metric family, full authored text and per-agent/arm/case association.
- Metadata compression must not hide comparability, provisional data, source rejection or bounded-evidence qualifications.

---

### Task 1: Bounded recorded telemetry presentation

**Files:** create `src/skillevaluator/reporting/telemetry_view.py`, `templates/telemetry.html.j2`, `tests/reporting/test_telemetry_view.py`.

**Interfaces:** produce `build_telemetry_view(payload: object) -> dict` with `token_sections`, `latency_sections`, `token_chart` (labels and three bounded datasets, or null), `has_tokens`, `has_latency`. Produce Jinja macro `render_telemetry(view)`. Task 3 supplies the helper with full assessed display data before the existing preview drops token detail; full retained raw data may be read only through validated bounded projections. Do not edit `html.py` or the main template.

- [ ] Write failing tests for native dataset/agent/case token breakdowns and latency, both arms, zero, unknown, partial coverage, unavailable delta reasons, malicious strings/nonfinite values and bounds.
- [ ] Run focused tests with a fresh explicit basetemp; capture the expected failure.
- [ ] Implement the pure read-only helper and native disclosure/table partial. Preserve recorded values, coverage and reason codes; no collection or recomputed timings. Support long-key native telemetry while retaining the existing public resources path.
- [ ] Expose IDs `tier3-token-usage`, `tier3-latency-metrics`; a legacy token canvas uses `tier3-recorded-token-chart`, leaving the current chart path intact. Chart data consists of labels for agent/condition groups and uncached/cached/completion datasets; no cached double count.
- [ ] Verify focused tests, no input mutation and bounded payloads; write a report. Controller reviews and commits explicit owned paths.

### Task 2: Quiet plugin evidence with stable ownership

**Files:** modify `reporting/plugin_evidence.py`, `templates/plugin_evidence.html.j2`, `tests/reporting/test_plugin_evidence_report.py`.

**Interfaces:** preserve `build_plugin_evidence()` canonical `static`/`runtime` values, slots and macro `render_plugin_evidence(evidence, slot)`. Add display-only summaries/default-open metadata if useful; all existing section IDs remain.

- [ ] Add failing tests for closed passive/empty panels, compact owner/source qualification and retained section anchors, plus exact canonical exports/omission distinctions.
- [ ] Capture red, implement compact closed sections/grouped empty states, verify green with explicit basetemp.
- [ ] Keep status/limitation cues visible and complete notes reachable. Remove repeated run ID and generic prose ahead of each section; routine metadata stays in accessible details.
- [ ] Verify static-only/runtime-only, omission and ownership fixtures without source reads/provider calls. Write report; controller reviews/commits owned paths.

### Task 3: Single-owner report hierarchy and summary integration

**Files:** `reporting/html.py`, optional focused `summary_view.py`, `templates/report.html.j2`, `report_ui.css.j2`, `report_ui.js.j2`, and presentation/unit tests. Do not edit Task 1/2 owned files or browser suite (controller owns browser acceptance).

**Interfaces:** consume `build_telemetry_view()` and `render_telemetry()` plus unchanged plugin macro. Template context `tier3_telemetry`; token-chart series enter the existing chart lifecycle with safe JSON serialization, theme/print refresh and accessible fallback. Keep all current canonical/publication assessment boundaries.

- [ ] Add failing ownership/shape tests for one full Insights instance, <=3 primary metrics, one empty quality state, one tier navigation, closed Run Details and native telemetry inclusion. Cover complete/incomplete/mixed/custom/baseline-only/skipped/source-rejected states.
- [ ] Capture red, implement header/hero/navigation simplification, one theme control retaining System/Light/Dark, compact publication reasons, concise Tier1/2 summaries.
- [ ] Put result/coverage and one structured recorded action first in Overview; remove full card copies and scorecard echoes. Preserve authored text/action/check/evidence in Insights. Secondary outcome families use a compact table when recorded.
- [ ] Replace shared paragraph coverage preamble with a short scoped cue; full agent/arm coverage belongs in Agent Details and full errors in Diagnostics. Keep important qualifiers at affected results.
- [ ] Move plugin inventories below useful native content; consolidate routine metadata in closed Run Details; human-readable Agent/Condition artifact labels keep exact paths/IDs in Diagnostics.
- [ ] Collapse method prose and nested case attempt diagnostics, preserving native Dataset cards. Preserve IDs, search and deep-link ancestor reveal, print restoration and no-JS.
- [ ] Integrate Task 1 helper/partial/chart while keeping current public resource and chart behavior. Verify focused report/CI regressions and write report; controller reviews/commits.

### Task 4: Browser acceptance, retained replay, packaging and PR

**Files:** existing `tests/reporting/test_report_browser.py` and fixtures, reporting docs/CHANGELOG, local ignored audit scripts/results, PR body.

- [ ] Extend synthetic browser coverage for no duplicate cards/navigation, folded evidence/deep links/search, <=96px scrolled mobile header and first-viewport result/action, legacy telemetry and chart fallback.
- [ ] Run report/security/retention and current browser checks once relevant changes are integrated; preserve meaningful old assertions while adapting obsolete layout selectors.
- [ ] Run lint, OSS source/wheel/sdist scan, package build and installed-wheel Chromium outside checkout; validate docs with pinned Fern.
- [ ] Replay retained actual/native samples offline with no rescoring; preserve rows, canonical/projections and 28/36,12/20. Capture new before/after desktop/mobile views and word/disclosure metrics against the approved audit baseline.
- [ ] Final independent review; one scoped fix wave if needed. Record final verification and any pre-existing full-suite limitations honestly.
- [ ] Verify sign-offs/clean tracked tree, push feature branch and open a PR with repository-template verification, BEFORE/AFTER table, safety/data preservation and test evidence. Do not include private data/internal repository URLs or local file links in the public PR. Do not merge.
