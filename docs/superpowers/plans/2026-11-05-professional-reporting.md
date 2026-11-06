# Professional NVIDIA Reporting Implementation Plan

> For agentic workers: implement task by task using subagent-driven development or executing-plans. Each task requires self-review, an independent review, and fixes for confirmed findings before its PR is ready.

**Goal:** Make GitHub SkillEvaluator's CLI and standalone HTML reports consistent, understandable, accessible, and professional with NVIDIA styling.

**Architecture:** Extend existing public status, presentation, provenance, resource, and Harbor-link abstractions. Keep canonical evaluation data and decision policies unchanged; render shared presentation projections into HTML and terminal views.

**Tech stack:** Python, Rich, Jinja2, vanilla browser JavaScript/CSS, packaged Chart.js 4.5.1, pytest, Ruff, Fern.

**Spec:** The user-approved Professional NVIDIA Reporting plan, reproduced as executable tasks below. GitHub repository only; no internal source or integrations are imported.

## Global Constraints

- Preserve scoring formulas, thresholds, publication policy, exit codes, CLI flags/defaults, artifact filenames, canonical JSON schemas, redaction, and report-size limits.
- Use latest GitHub main in isolated worktrees; preserve the existing checkout and its untracked duplicate directories.
- Extend existing shared status logic; no duplicate verdict resolver, fabricated measurements, inferred missing baselines, or mutated canonical payloads.
- Keep report_scope's existing decision semantics separate from requested/displayed tier context. Optional keyword-only presentation arguments must preserve existing callers and emit_reports()'s boolean return.
- Keep private integrations and internal plugin-specific code out of the public changes. Use only existing public recorded evidence and public upstream assets.
- Each PR includes focused behavioral tests, documentation/changelog, signed-off commits, root self-review, independent review, and fixes for confirmed findings.
- Publish five focused PRs; keep drafts during verification, mark ready when verified, and leave them open without merging main.

## Review Focus

- Confirmed failures plus missing evidence: command status must agree across terminal and HTML while retaining stricter publication assessment separately.
- Missing/N/A/skipped/baseline-only/retained/truncated evidence: unknown is not zero, N/A is not execution, and preview caps cannot change totals.
- Long hostile or Unicode names/paths: escape markup/HTML/control characters, preserve safe links, and fit narrow displays without losing numeric values.
- Offline and installed-wheel reports: all chart assets/licenses are packaged; generated reports make no runtime network requests for charts.
- Non-TTY interruption/closed streams: durable progress must stop cleanly without affecting evaluation, duplicating events, or leaking workers.

### Task 1: Consistent status, scope, and execution accounting

Branch: naren/reporting-status-accounting, based on current main.

- [ ] Extend existing run_status and presentation helpers with an immutable final-render context: command outcome, separate publication assessment, tier status/counts, score scope, and per-condition coverage. Build once per final render and share across compact CLI, full CLI, HTML header/hero/tiles.
- [ ] Preserve failure precedence and advisory gating. Use primary labels Passed required checks / Changes required / Incomplete. Keep publication eligibility separate; standalone Tier 3 publication remains Not assessed.
- [ ] Label agent scores explicitly (Best agent score); they are not whole-validation approval.
- [ ] Count each tier's own results; distinguish passed, failed, incomplete, skipped, not run and N/A, excluding N/A from executed denominators.
- [ ] Prefer recorded per-condition scored/expected counters before previews. Display N scored trials: X with skill, Y baseline. Missing/malformed remains unavailable, explicit skip stays skipped, legacy row counts are observed/retained rows, not complete totals. Handle retries, early stopping, skill-only/baseline-only, partial/failed arms and >64 preview rows.
- [ ] Add optional keyword-only requested/displayed presentation context where necessary in emit_reports and HTMLReporter, preserving defaults, decision report_scope, return values and JSON schemas. Do not synthesize ValidationResult evidence for missing tiles.
- [ ] Add meaningful failing tests first, implement, run focused reporting/status/CLI tests; prove payload immutability and status/count parity. Update user docs and changelog; signed-off commit; self-review and independent review.

### Task 2: Truly self-contained HTML charts

Branch: naren/reporting-offline-charts, independently based on current main.

- [ ] Vendor public Chart.js 4.5.1 UMD from public upstream; verify its original bytes using the existing SRI, record source/version/digests, retain applicable Chart.js/bundled dependency licenses in resources, notices and generated reports.
- [ ] Load fixed package resources with a cached loader and embed only when Tier 3 requires charts. Remove CDN/source-map dependencies. No new npm/Python runtime dependencies or lockfile churn.
- [ ] Preserve chart data, theme changes, print redraw and number formatting. Add accessible descriptions/table references and visible initialization-failure fallback instead of blank space.
- [ ] Package assets/licenses in wheel and sdist. Replace CDN tests with embedded-runtime/version/digest/license tests; confirm Tier 1 avoids asset bytes and installed-wheel render works without network.
- [ ] Update docs/changelog, run focused tests/build/package checks; signed-off commit; self-review and independent review.

### Task 3: Persistent HTML navigation and accessible NVIDIA styling

Branch: naren/reporting-workspace-navigation, based on reviewed Task 1.

- [ ] Retain existing logo, public dark sidebar and restrained surfaces. Compact hero to skill, overall decision/reason, labeled agent result; move environment/metrics/provenance details into existing disclosures.
- [ ] Comprehensive validate displays all 3 tier tiles with truthful absent/unrequested states; targeted reports keep focused scope. Sidebar has Summary, Tier 1/2/3 and Run details; mobile has compact sticky tier navigation.
- [ ] One activation function synchronizes sidebar, tiles, active panel, selected state and focus. Place evidence destination outside hidden tier panels; preserve exact finding links, clearing filters and highlighting/focus. Current sidebar follows real section navigation.
- [ ] Retain Overview/Insights/Agent Details/Trials/Dataset/Diagnostics. Associate tabs/panels and aria state; roving tabindex and Left/Right/Home/End/Enter/Space support for subtabs and agent tabs, including unusual names.
- [ ] Remove expanded height caps that hide findings or tables. One main page scroll, horizontal overflow only for wide tables, bounded nested vertical scrolling only for raw logs/JSON. Native theme-aware scrollbars, stable gutters, forced-colors fallback; no simulated scrollbars.
- [ ] Add System/Light/Dark control; persist explicit preferences, follow system by default. Respect reduced motion and print/no-JS evidence access and expansion restoration. Offset sticky mobile/local navigation without overlap.
- [ ] Use lime #76B900 for accents, dark green #3E6500 for light-theme text/actions, charcoal navigation, restrained borders/shadows, red/amber semantic states. Normal text 14px, secondary labels >=11px, WCAG AA text/controls.
- [ ] Add behavioral tests plus browser checks at 1440/760/390px, both themes, keyboard, long reports, reduced motion, no JS, and print. Update docs/changelog; signed-off commit; self-review and independent review.

### Task 4: Clearer Tier 2 summaries and complete evidence navigation

Branch: naren/reporting-evidence-views, based on reviewed Task 3.

- [ ] Split Tier 2 into Within this skill / Across skills using existing typed workflow and similarity metadata. KPIs: Checks completed, Duplicate findings, Affected files/skills, Catalog entries compared; explicit scope and unavailable values.
- [ ] Completed zero matches differs from skipped/unavailable comparison. Explain target-vs-catalog vs pairwise collections; retain self-exclusion and duplicate accounting. Context/token savings only if recorded.
- [ ] Expose every existing normalized Harbor job in HTML, CLI and Markdown, with Results and available Analysis. Preserve names or Job N fallback, legacy single-job/analysis-only, deduplication, URL safety and escaping; no inferred attribution.
- [ ] Surface existing sanitized scanner/model/evaluator/dataset/source provenance in appropriate disclosures. Retain both-arm resource comparisons, nulls, omission counts and full downloads. Explicit with-skill-only chart labels; no cached-token double counting.
- [ ] Test missing/skipped/zero-match workflows, 0/1/multiple/duplicate/unsafe job links, retained reports, nullable telemetry, limits and immutability. Update docs/changelog; signed-off commit; self-review and independent review.

### Task 5: Consistent responsive CLI and durable CI progress

Branch: naren/reporting-cli-polish, based on reviewed Task 1.

- [ ] Share lightweight semantic Rich styles across compact/full/direct Tier 3 reports. Preserve green/gray pipeline and amber numbered fixes; remove inconsistent decorative cyan/magenta in report views.
- [ ] Reuse grade mapping A/B green, C amber, D/F red with failed/incomplete/unavailable overrides. Displayed zero lift is neutral; preserve unavailability/inconclusive labels and all scoring thresholds.
- [ ] Keep each condition label/number/complete bar atomic; stack at narrow widths and preserve all numbers. Test 98/80/60 columns and long/wide Unicode labels.
- [ ] Add concise successful quality/lift evidence to existing full-report Details cells; do not expand every successful panel.
- [ ] Use actual producer-emitted paths/basenames, not hardcoded report.html. Safe hyperlinks only for capable interactive output; readable paths with no newly added OSC8 in plain/no-color/redirected/string-render APIs. Preserve escaping and control stripping.
- [ ] Final panels stdout; durable noninteractive progress stderr: tier/check starts, stage changes, measured completion milestones, completion. Deduplicate unchanged events; heartbeat after 60 seconds without progress. Reuse progress lifecycle, no invented percentages, clean interruption/closed-stream handling and no leaked workers.
- [ ] Fake-clock progress tests, artifact/escaping/color/layout/status parity tests, unchanged CLI surface and JSON/exit behavior. Update docs/changelog; signed-off commit; self-review and independent review.

## Validation and delivery

- Focused regressions per task, then repository-required make lint / make test / make build, Fern validation, OSS source/archive boundary, packaging and relevant compatibility tests. Record baseline environment failures separately from regressions; never claim unrun checks passed.
- Review identical synthetic/retained fixtures at desktop/mobile and terminal widths in both themes, no-color/pipes, offline, keyboard/reduced-motion/no-JS/print. No live provider evaluation needed for reporting changes.
- Optimize classify-once projections, immutable asset cache and repeated traversals only where justified; avoid broad rewrites and unnecessary polling/workers.
- PR dependencies: 1 first; 2 independent; 3 and 5 follow 1; 4 follows 3. Test an isolated assembled integration checkout with all five changes. Do not merge main.
- Deliver five PR links, CI statuses, before/after previews, and concise review/fix records. All commits signed off; PR template and changelog/docs updated.
