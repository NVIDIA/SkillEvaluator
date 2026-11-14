# Report clarity and complete recorded telemetry

Approved design: retain the native three-tier report presentation and its features, consolidate repetitive tiles and prose, and keep routine Run Details and provenance quietly available. This extends the report-parity work on the same branch.

## Content ownership

| Surface | Visible first | Detail owner |
| --- | --- | --- |
| Header / hero | Brand, target name, one required-validation outcome, short reason, separate publication state | Routine target path/profile/version/configuration in closed Run Details; publication reasons in a disclosure |
| Tier navigation | One set of clickable native tier cards with status and one useful headline | Preserve old anchors, keyboard navigation and explicit not-run scopes |
| Tier 1 | Completion, severity hierarchy, actionable findings and native views | Passive inventory and scanner provenance in disclosures |
| Tier 2 | Native Intra/Inter tabs, completion and findings | Catalog metadata and distinct skipped/incomplete records; no unavailable-value tile grid |
| Overview | Short scoped status, Score including agent, Lift, Coverage, one recorded action | Full conclusions and recommendations in Insights; comparisons and agent evidence remain reachable |
| Insights | Complete native cards, source/severity, action, verification and evidence | Long recorded text remains intact; explanatory legend can collapse |
| Agent Details | Selected-agent coverage, dimensions and evaluator evidence | Passive loading/component/MCP details in disclosures |
| Trials | Trial/reliability results and recorded resources | Full token accounting, latency, cost, charts and methods in native disclosures |
| Dataset | Native filterable case cards, prompts and expectations | Per-case attempt diagnostics closed; shared failures referenced once |
| Diagnostics | Compact diagnostic navigation and downloads | Full errors, paths/copy controls, raw data, source evidence and limitations |
| Run Details | One closed footer disclosure | Routine identity/environment/models/digests/CI/gate metadata |

Keep verdict, score, best-agent and lift capabilities: verdict becomes a scoped status, score and best agent share one tile. At most three primary Overview metric tiles: Score, Lift, Coverage. If all quality measurements are absent, use one explicit unavailable state plus coverage/action rather than empty Score/Lift tiles. Secondary goals/checks/security outcomes remain available as a compact comparison table when recorded.

Passive plugin sections keep their ownership, content and anchors. Default closed; empty observations may share a compact group, with each retained section addressable. Keep a visible limitation cue for omitted rows, shortened display text and unrecoverable pre-retention clipping; exports do not recover previously missing text.

## Recorded telemetry compatibility

Canonical native-shaped `token_usage` and `latency_metrics` remain retained but are not fully presented by the current totals-only/native-trial adapters. Restore readable dataset/agent/case and with-skill/baseline tables, prompt/cached/uncached/completion/total/efficiency tokens, deltas/reasons/coverage, timing sources, LLM/tool seconds, ms/token and arm runtime. Preserve the newer public `agents[*].resources` path.

Use a bounded read-only display adapter; no collector calls, source reopening, rescoring, canonical mutation, computed replacement timing or relaxed comparability gates. Native token charts compare both conditions using uncached prompt + cached prompt + completion without double counting cached reads. Unknown/partial counters remain unknown/partial. Recorded unavailable comparison reasons remain visible next to their values.

## Acceptance

- One full Insights card per recorded insight; Overview has a concise summary/action and native links.
- No default-open empty plugin panels, duplicate tier navigation, repeated hero score block or empty quality tile wall.
- Required validation, live quality, completed-agent scope, publication and execution status remain distinct. Keep source rejection, incomplete arms, never-activated skills, provisional evidence, historical baselines and invalid comparisons visible at the affected result.
- Opening Tier 3 Overview at 390 x 1000 exposes status, scored/planned coverage and recorded next action before routine metadata. Target a persistent mobile header no taller than 96px after scrolling; retain the 16px native body scale.
- Target at least 40% fewer default Overview explanatory words on the audited incomplete and successful shapes. Never clip authored evidence or hide a necessary qualification to meet the target.
- Routine Run Details stays closed; methods, run IDs, digests and paths do not precede primary results. A short, approximately 15-word shared attention cue links to complete scope-specific reasons.
- All native features, filters, disclosures, links/copy/download actions and six Tier 3 tabs remain usable. Deep links reveal owning tabs and closed ancestors; search exposes folded matches. Preserve print state restoration, no-JS, offline charts, both themes and mobile overflow handling.
- Preserve scores, gates, canonical payload/download equality, source rows/projections, retained optional-skip metadata, CI canonical artifact links, bounds, escaping, redaction, safe URLs, surrogate handling and atomic output protections.
- Replay retained reports with zero provider calls/rescored trials; preserve coverage 28/36 and 12/20. Commit synthetic fixtures only. Verify installed-wheel behavior outside the checkout and report unrelated baseline failures honestly.

## Chosen approach

Use one main home for each fact with concise summaries and native disclosures. Wording-only changes leave structural duplication intact; a separate compact/detailed mode adds redundant UI and maintenance. Keep one renderer, pinned Chart.js 4.5.1 and the native typography/palette, with narrow accessibility/layout adaptations.
