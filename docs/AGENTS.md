<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

# AGENTS.md — `docs/`

Guidance for AI agents editing SkillEvaluator's long-form documentation. This
folder holds MDX pages published to <https://docs.nvidia.com/skills/skillevaluator> via
[Fern](https://buildwithfern.com/). Human-facing notes live in
[`README.md`](README.md); this file is the machine-facing companion.

## What lives here

- `index.mdx` — landing page and three-tier overview (Fern slug `/`)
- `quickstart.mdx` — check a bundled example with the installed CLI
- `starter-demo.mdx` — reproducible public terminal starter, output checks and adaptation
- `installation.mdx` — shared pip, uv, and GitHub installation paths and verification
- `installation-options.mdx` — upgrades, Docker, release differences, extras, and scanner setup
- `configuration.mdx` — providers, embeddings, and Tier 3 credential roles
- `agent-skill.mdx` — installing and using the `skills/skillevaluator` agent skill
- `tier1-validation.mdx` — first static run, report, and next steps
- `tier1-reference.mdx` — detailed checks, policies, scanners, and exit semantics
- `tier2-deduplication.mdx` — choose a scope, run deduplication, and read findings
- `tier2-reference.mdx` — chunking, catalogs, thresholds, algorithms, and limits
- `tier3-live-evaluation.mdx` — first Tier 3 comparison; keep this task-focused
- `prepare-dataset.mdx` — author, generate, review, and validate cases before trials
- `tier3-run-options.mdx` — choose an agent and attempt count for reviewed cases
- `tier3-run-reference.mdx` — staging contracts, run settings, budgets, and repeat comparisons
- `tier3-runtime.mdx` — backend compatibility, runtime contracts, and troubleshooting
- `tier3-grading.mdx` — judges, evidence budgets, retries, and fallback policies
- `tier3-results.mdx` — scores, applicability, verdicts, and incomplete evidence
- `tier3-experiments.mdx` — skill interaction and activation experiments
- `grader-validation.mdx` — independent-label qualification of the shipped verifier
- `eval-datasets.mdx` — the `evals/` contract, dataset generation, run policy
- `agents-and-sandboxes.mdx` — agents, environment backends, local mode, readiness
- `custom-graders.mdx` — bring-your-own graders and native Harbor tasks
- `reports.mdx` — report formats, scores, and the results-on-disk contract
- `metrics-reference.mdx` — every metric's formula, range, units, evidence, and validation status
- `performance-dashboard.mdx` — compare saved Tier 3 scores and resource use
- `benchmark-rollout.mdx` — regenerating and promoting `BENCHMARK.md` cards
- `github-actions.mdx` — skill check workflow setup and optional hosted PR feedback
- `cli-onboarding.mdx` — released GitHub setup wizard
- `repository-integration.mdx` — choose GitHub or GitLab for skill checks; pre-commit hooks
- `gitlab-ci.mdx` — reusable skill checks and downloadable GitLab reports
- `advanced-ci.mdx` — secondary opt-in prerequisites and all-tier reference
- `ci-integration.mdx` — exit codes, GitHub Actions recipe, progressive adoption
- `cli-reference.mdx` — every command and flag
- `environment-variables.mdx` — variable-by-variable inventory
- `developer-guide.mdx` — contributor setup, tests, and docs workflow
- `assets/` — images referenced by the pages
- `README.md` (human authoring guide) and this `AGENTS.md` are **not** published
  pages — do not add them to the navigation

Navigation order, page titles, and slugs are defined in
[`../fern/docs.yml`](../fern/docs.yml). The Fern CLI version is pinned in
[`../fern/fern.config.json`](../fern/fern.config.json).

## Rules for editing

Keep entry pages focused on the reader's next action: setup, a runnable command,
the expected output, and a next step. Put advanced contracts and edge cases in
the linked reference page. Use Steps for ordered actions, Tabs for exclusive
choices, and Accordions for optional detail; do not put a reference chapter
ahead of the first runnable example.

Keep pip, uv, uv bootstrap, and direct repository installation in
`installation.mdx`; keep upgrades and advanced setup in
`installation-options.mdx`. Default to the lightweight `skillevaluator` install;
keep `skillevaluator[all]` as an explicit opt-in for all tiers. Distinguish the published PyPI release from
unreleased GitHub features and dependencies. Tier guides and the
quickstart link there and use the installed `skillevaluator` command. Avoid
copying package-manager recipes into each tier or mixing `uv run`, `uv tool
install`, and pip in a single first-run workflow. Contributor checkout commands
belong in the development guide.

1. **Every page needs YAML frontmatter** with `title` and `description`. The
   `title` renders as the page H1 — do **not** also add a `# Heading` in the
   body, or the title appears twice.

   ```mdx
   ---
   title: "Configuration"
   description: "One-sentence summary used for SEO and the nav preview."
   ---
   ```

2. **File names use URL best practices**: lowercase, hyphen-separated, no spaces
   or underscores (`tier1-validation.mdx`, not `TIER1_VALIDATION.mdx`). Keep the
   file name equal to the page `slug` in `docs.yml`.

3. **Adding a page**: create the `.mdx` file, then register it under
   `navigation` in [`../fern/docs.yml`](../fern/docs.yml). A page not listed
   there will not appear on the site. Rename via `git mv` to preserve history,
   and update every reference (see rule 5).

4. **This is MDX, not Markdown.** A bare `<` starts a JSX tag and `{` starts an
   expression. Keep angle brackets and braces (e.g. `<placeholder>`,
   `${VAR}`) inside backtick code spans or fenced code blocks, where MDX leaves
   them literal. Do not use `<https://...>` autolink syntax — write a normal
   `[text](url)` link.

5. **Links**:
   - In published MDX, use site paths such as
     `/skills/skillevaluator/configuration` or
     `/skills/skillevaluator/tier1-validation#anchor`. The pinned Fern 5.121.0
     preview returns HTTP 400 for file-relative `.mdx` page URLs; it does not
     resolve them to published slugs.
   - For repository-only files outside `docs/`, use full GitHub source URLs in
     published MDX. Keep repository Markdown, images and YAML `path:` entries
     relative to their source files.
   - When renaming a page, grep the whole repo and fix references in the sibling
     pages, the root [`../README.md`](../README.md), [`README.md`](README.md),
     and [`../fern/docs.yml`](../fern/docs.yml).

## Verify before committing

Requires **Node.js 22+**. Install the pinned CLI from the committed lockfile with
`npm ci --prefix fern --ignore-scripts --omit=optional`.

```bash
./fern/node_modules/.bin/fern check       # validate docs.yml config and all links — must pass
./fern/node_modules/.bin/fern docs dev    # optional live preview at http://localhost:3000
```

`fern check` must pass; it is the same gate the site build relies on. Publishing
is automatic from `main` via [`../.github/workflows/publish-docs.yml`](../.github/workflows/publish-docs.yml)
— do not run `fern generate` locally.

## Repo conventions that still apply

- Sign off commits (`git commit -s`); the DCO check rejects unsigned commits
  (see [`../CONTRIBUTING.md`](../CONTRIBUTING.md)).
- Update [`../CHANGELOG.md`](../CHANGELOG.md) when a docs change is user-visible.
