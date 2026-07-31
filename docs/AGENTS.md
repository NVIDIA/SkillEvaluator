<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

# AGENTS.md — `docs/`

Guidance for AI agents editing SkillEvaluator's long-form documentation. This
folder holds MDX pages published to <https://docs.nvidia.com/skills/skillevaluator> via
[Fern](https://buildwithfern.com/). Human-facing notes live in
[`README.md`](README.md); this file is the machine-facing companion.

## What lives here

- `index.mdx` — landing page and three-tier overview (Fern slug `/`)
- `installation.mdx`, `configuration.mdx` — setup and credential reference
- `tier1-validation.mdx`, `tier2-deduplication.mdx`, `tier3-live-evaluation.mdx`
  — the three evaluation tiers
- `developer-guide.mdx` — contributor setup
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
   - Between pages in this folder, link with the relative `.mdx` file path
     (`configuration.mdx`, `tier1-validation.mdx#anchor`). Fern resolves these
     to the published slug, and they also work when browsing the source on
     GitHub — a bare slug (`configuration`) 404s on GitHub.
   - To files outside `docs/`, use a relative repo path kept as `.md`
     (`../README.md`, `../THIRD_PARTY_NOTICES.md`) — those are not Fern pages.
   - When renaming a page, grep the whole repo and fix references in the sibling
     pages, the root [`../README.md`](../README.md), [`README.md`](README.md),
     and [`../fern/docs.yml`](../fern/docs.yml).

## Verify before committing

Requires **Node.js 22+** and the Fern CLI (`npm install -g fern-api`).

```bash
fern check       # validate docs.yml config and all links — must pass
fern docs dev    # optional live preview at http://localhost:3000
```

`fern check` must pass; it is the same gate the site build relies on. Publishing
is automatic from `main` via [`../.github/workflows/publish-docs.yml`](../.github/workflows/publish-docs.yml)
— do not run `fern generate` locally.

## Repo conventions that still apply

- Sign off commits (`git commit -s`); the DCO check rejects unsigned commits
  (see [`../CONTRIBUTING.md`](../CONTRIBUTING.md)).
- Update [`../CHANGELOG.md`](../CHANGELOG.md) when a docs change is user-visible.
