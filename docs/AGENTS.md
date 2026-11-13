<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

# AGENTS.md — `docs/`

Guidance for AI agents editing SkillEvaluator's long-form documentation. This
folder holds MDX pages published to <https://docs.nvidia.com/skills/skillevaluator> via
[Fern](https://buildwithfern.com/). Human-facing notes live in
[`README.md`](README.md); this file is the machine-facing companion.

## What lives here

- `index.mdx` — landing page and three-tier overview (Fern slug `/`)
- `quickstart.mdx` — install, one provider and key, and first runs
- `starter-demo.mdx` — reproducible public terminal starter, output checks and adaptation
- `installation.mdx` — extras, uv, pip, Docker, requirements
- `configuration.mdx` — providers, embeddings, and Tier 3 credential roles
- `agent-skill.mdx` — installing and using the `skills/skillevaluator` agent skill
- `tier1-validation.mdx` — Tier 1 checks, profiles, scanner setup, exit codes
- `tier2-deduplication.mdx` — Tier 2 dedup commands, catalogs, thresholds, limits
- `tier3-live-evaluation.mdx` — Tier 3 live agent evaluation workflow
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
- `repository-integration.mdx` — choose GitHub or GitLab for skill checks
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
