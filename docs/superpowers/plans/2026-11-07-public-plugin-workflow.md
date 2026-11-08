# Public Plugin Workflow Implementation Plan

> **For agentic workers:** Use the subagent-driven-development workflow. Each task includes focused regression checks and independent review.

**Goal:** Complete split-plan Item 25 by exposing the assembled plugin workflow through installed public entrypoints.

**Architecture:** Reuse the bounded discovery, static composition, deduplication, preparation, execution, evidence and report implementations merged in Items 1–24. Keep the existing skill engine and its defaults intact; add a public coordinator and shared options at the boundary.

**Tech Stack:** Python 3.12+, Click, Pydantic, existing Harbor adapters, Jinja and Fern.

**Spec:** The user-approved 25-item split plan, Item 25: CLI commands, detection, dispatch, exported options, configuration, reporting, documentation and installed-entrypoint acceptance.

## Global Constraints

- Preserve existing skill, rules, workflow and basic bundle functionality.
- Keep network audits and MCP host probes opt-in; bound input, output and filesystem access.
- Keep staged, listed, loaded and exercised evidence distinct. Missing evidence remains explicit.
- Retain common-score comparisons and advisory signal semantics.
- Use the user's requested terminology in all new prose.
- Preserve other agents' work and concurrent upstream commits; never force-push.

## Review Focus

- Public parsing and detection cover all five formats and reject unsafe inputs.
- CLI and API use the same resolved configuration and lifecycle without recursive dispatch.
- Ordinary skill calls retain their original engine arguments and output.
- Saved report replay works after source removal on a base installation without executing code or making network calls.
- Public reports preserve evidence limits and safe output paths in all six formats.

## Tasks

- [x] Connect public type detection and Tier 1 composition, preserving check selection, policy and publication identity.
- [x] Connect Tier 2 plugin deduplication/catalog dispatch and compatibility.
- [x] Export shared runtime options; apply configuration precedence and expose the existing lifecycle through the service.
- [x] Add public runtime and saved-report commands, reusing existing command options and report adapters.
- [x] Publish consistent static/runtime evidence through report hooks and document supported behavior and limitations.
- [ ] Run focused regression tests, independent review, package/base-install and installed public acceptance checks.
- [ ] Push the signed-off branch and open an accurate Item 25 PR; close superseded PR #28 without a comment.
