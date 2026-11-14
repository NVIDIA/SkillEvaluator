# Report presentation parity implementation plan

Approved specification: the user's “Bring GitHub SkillEvaluator reports to GitLab report parity” plan in this chat. This file records its implementation boundaries and task interfaces.

**Goal:** reuse the verified native report presentation, with all three tiers and all six Tier 3 tabs, while preserving public result, gate, safety, and retained-evidence contracts.

**Base:** public main f5bf6c1f3666699fd145612a2e9162e9721db3f9. Worktree is isolated from the user's dirty checkout. The presentation reference and private reports remain outside committed fixtures; source provenance belongs in the local audit manifest.

## Global constraints

- One HTMLReporter and one authoritative presentation; no alternate renderer or competing layout stylesheet.
- Preserve public scores, gates, applicability, custom-only neutrality, failed-arm suppression, source identity, bounded previews, redaction, escaped JSON, safe URLs, surrogate handling and atomic outputs.
- Rendering saved evidence makes zero provider calls, performs no rescoring, and does not reopen evaluated source paths.
- Keep existing Chart.js 4.5.1 and its licenses. Use copied chart configuration without downgrading assets.
- Keep existing call sites and CLI contracts working; public plugin CLI dispatch remains on its independent PR track.
- No private benchmark material, private URLs, credentials, or reference report payloads in committed fixtures or public-facing documentation.

## Task 1: concise, evidence-backed Insights

Own evaluation/tier3_report.py, a new reporting/insights_view.py module and focused tests. Do not edit reporting/html.py or templates, which Task 2 owns.

Implement and export `build_insights_view(payload: dict) -> dict` returning `conclusions`, `recommendations`, `next_steps`, and `execution_errors`. Conclusion/recommendation items retain the template's established title/message/severity/source/category/metric/evidence fields. Preserve healthy persisted feedback verbatim. For failed retained runs, consume structured condition coverage and next_steps to produce concise execution summaries and actions, never split arbitrary semicolon prose. Deduplicate only identified copies of the same failure, retaining case/agent/evidence association. Full original errors remain accessible separately. Both Overview and Insights will consume this projection.

For new failed evaluations replace error concatenation with concise coverage/conclusion/action messages, retaining complete execution_errors. Preserve all scoring/guard semantics. Tests must catch duplicate blobs, unavailable coverage, separate case/agent errors, retained-payload mutation, healthy text rewriting, and provider calls during rendering.

## Task 2: exact native presentation and canonical exports

Own reporting/html.py, templates/report.html.j2, report_ui.css.j2/report_ui.js.j2, and presentation-focused tests. Reuse the pinned source template's header, hero, fonts, palette, tables, cards, badges, disclosure layout, chart configuration and footer; adapt fields at the context boundary. Replace the sidebar and layout override system. Preserve search/filters, a11y, deep links, mobile, print, no-JS, CI metadata and safe rendering with narrow enhancements. Default to dark with Tier3 and light otherwise, respecting explicit saved themes.

Integrate Task 1's `build_insights_view`. Render the same cards in Overview and Insights, and retain full errors in Diagnostics. Use native Dataset case cards with per-case disclosures and reference shared run errors once. Make all Tier1/2/3 summaries truthful; include Similarity Check in Tier2 mapping. Comprehensive validation shows all3 tiers including explicit not-run states; standalone Tier3 stays Tier3-only. Measurements absent in retained evidence remain unavailable, never zeros or invented quality data.

Add optional keyword-only `plugin_static` / `plugin_runtime` inputs to HTMLReporter's render_all, default None, accepting existing typed report projections or validated snapshots as agreed with Task3. Project them through Task3's view helper and pass the resulting `plugin_evidence` to template slots at tier1, tier2, overview, agents, trials, provenance. Each detailed section appears once.

Keep validated canonical Tier3 separate from display shaping, and embed/export the retained object unchanged (including neutral vs forced display-fail, null and empty fields). Preserve base64/script-safe encoding and preview limits. Keep existing HTML report_data envelope, adding supplied plugin projections; do not replace it with JSONReporter's schema. Keep CI canonical_json_url direct-link behavior and embedding suppression.

## Task 3: integrate passive plugin projections and retained metadata

Own reporting/plugin_static_render.py, plugin_runtime_render.py, a new reporting/plugin_evidence.py, plugin templates, reporting/retained.py, json_reporter.py and focused plugin/round-trip tests. Do not edit html.py or main template.

Implement `build_plugin_evidence(static: object = None, runtime: object = None) -> dict` with keys tier1, tier2, overview, agents, trials, provenance, each a list of existing bounded section dictionaries. Accept existing StaticPluginReport/RuntimePluginReport via payload() and validated dictionaries; never inspect source trees or invoke a runtime. Preserve snapshots exactly for export (return payload snapshots separately under static/runtime or supply a helper).

Static inventories/dependencies/MCP/hooks/privileges go Tier1; catalog similarity goes Tier2. Runtime eligibility/arm coverage goes Overview; loading/activation/behavior/MCP goes Agent Details; paired comparisons/context/cost goes Trials; source/limitations/exclusions goes Diagnostics. Include source labels/ownership and truncation notices. Distinguish omitted rows and shortened text; do not promise to recover already-clipped values.

Replace post-render HTML string insertion with pre-render context through existing private composition functions; retain non-HTML behavior. Coordinate the render_all keyword interface with Task2; no automatic public plugin dispatch. Preserve optional_skip typed metadata in retained reader and JSON round-trip without changing gate policy.

## Task 4: regression, browser and distribution acceptance

Root owns dev-only Playwright/Chromium integration, CI job, package checks, documentation/changelog and local regeneration deliverables. Use synthetic sanitized fixtures shaped like the two successful two-agent references, both incomplete plugin reports, mixed-agent/custom-only/missing baseline and telemetry/skipped Tier3, single-tier and complete reports, and nonzero collection similarity.

Exercise all tiers/subtabs/agent/dedup views, themes, desktop/mobile, keyboard, filters, details, copy/downloads, chart fallback/offline behavior, no-JS and print with no page-wide overflow or console errors. Compare actual style/geometry using same browser/data/viewport/theme; keep before/after screenshots outside tracked code. Add installed-wheel render smoke outside the checkout.

Regenerate the two supplied reports from retained artifacts outside the repository. Record source hashes, renderer revision, raw Tier3 equality, result/projection preservation, optional_skip metadata, coverage 28/36 and12/20, no provider calls/rescoring, screenshots and test results. Audit manifest records source pin, copied fragments and intentional adaptations without placing private source references in distributable docs.

## Review and completion

Review each independent task against the approved requirements and code quality, then review the integrated branch. Run focused regressions, project lint/build and relevant existing report suites. Resolve material failures; report unrelated suite failures precisely. Deliver isolated branch/worktree plus regenerated reports, screenshots, provenance and verification results. No push, merge or publishing is part of this implementation request.
