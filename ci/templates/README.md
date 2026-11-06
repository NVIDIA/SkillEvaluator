# Copy a CI template

Bring SkillEvaluator to your repository by copying a configuration file. You do
not need to install SkillEvaluator locally, use an onboarding CLI, or copy
evaluation code into your project. GitHub and GitLab templates are maintained
together in this public NVIDIA/SkillEvaluator repository.

| Platform and result | Source template | Where to put it |
| --- | --- | --- |
| GitHub: checks and downloadable reports | [skill-checks.yml](skill-checks.yml) | `.github/workflows/skill-checks.yml` |
| GitHub: also host reports and update a PR comment | [publish-skill-reports.yml](publish-skill-reports.yml) | `.github/workflows/publish-skill-reports.yml`, alongside the checks file |
| GitHub: optional trusted manual LLM checks | [skill-checks-llm.yml](skill-checks-llm.yml) | `.github/workflows/skill-checks-llm.yml` |
| GitLab: checks and downloadable reports | [gitlab-skill-checks.yml](gitlab-skill-checks.yml) | `.gitlab-ci.yml`, or merge into your existing root configuration |

Open a source file to inspect it; GitHub's **Raw** view lets you copy its full
contents. See [example results](../examples/README.md) without needing access to
a separate demo repository.

## Get a version you can run

**These source templates contain placeholders.** Copy the
[ready-to-use configurations](../bundles/0.8.0/README.md) instead; their reviewed
workflow commits and container image digest are already filled in.

Open a file in that folder and use GitHub's **Raw** view to copy its complete
contents. The generated files and their checksums are kept in this public
repository. Future releases attach the same files to the single SkillEvaluator
product release, alongside its Python packages.

If you customize a source template instead, use the immutable workflow/image
identities in the generated bundle's `release.json`. Do not leave placeholders
or substitute `main` or `latest` for runtime pins.

Maintainers can release a reviewed publisher update while keeping the verified
scanner workflow and image unchanged. Pass its full commit SHA as
`--publisher-ref` to `ci/render_templates.py`; otherwise it uses `--workflow-ref`.
The release receipt records both revisions. This changes the publisher workflow
and tools only; the scanner pin and approved evaluator identity stay unchanged.

## Customize and commit

- **GitHub:** set `paths` and optionally `min-score` (default `70`). Change
  `branches: [main]` if needed, then commit the workflow to your default branch.
  Existing workflows remain separate. Follow the
  [GitHub guide](../../docs/github-actions.mdx).
- **GitLab:** set `SKILL_EVAL_PATHS` and optionally `SKILL_EVAL_MIN_SCORE`
  (default `70`). For an existing pipeline, merge the include, variables and job
  rules; preserve your other jobs and `workflow: rules`. The job uses `test` by
  default, so override its stage if your pipeline does not define it. Follow the
  [GitLab guide](../../docs/gitlab-ci.mdx).
- **Optional GitHub comments:** the second file needs the
  [one-time hosting and trust setup](../../docs/github-actions.mdx#enable-pr-comments).
  Uncomment `id-token: write` in the scan file for signed execution receipts,
  then approve that file's new blob. Copying the second file alone does not
  enable comments.
- **Optional LLM checks:** configure a provider key in the separate manual
  workflow. The default checks need no model key.

Open a pull request or merge request that changes a skill to see the checks.
Enable the platform's CI service, allow the referenced workflows or includes,
and provide a compatible runner. The [platform guides](../../docs/repository-integration.mdx)
cover results and merge-gating settings.


## Candidate all-tier templates

The unreleased all-tier setup uses two thin consumer files per platform:

| Platform | Evaluation | Trusted feedback |
| --- | --- | --- |
| GitHub | [skill-checks-all-tiers.yml](all-tiers/skill-checks-all-tiers.yml) | [feedback-all-tiers.yml](all-tiers/feedback-all-tiers.yml) |
| GitLab | [gitlab-all-tiers.yml](all-tiers/gitlab-all-tiers.yml) | [gitlab-feedback-all-tiers.yml](all-tiers/gitlab-feedback-all-tiers.yml) |

These files contain placeholders. Use a reviewed schema-version 2 release
with the platform's all-tier capability and immutable workflow, tools and
image pins. Existing 0.7.2 downloads remain Tier 1 only. The all-tier CLI
renders these references and approves the exact GitHub caller blob. See
[the onboarding guide](../../docs/cli-onboarding.mdx#opt-in-to-three-tier-ci).
Merge the trusted feedback entrypoint onto the default branch before the
first demo PR/MR. On GitLab, merging into an existing root `.gitlab-ci.yml`
means replacing the Tier 1 include with exactly the two all-tier includes and
starting every other job's rules with
`- if: '$SKILL_EVAL_FEEDBACK == "true"'` / `when: never`, because trusted
feedback runs as a default-branch pipeline; see the
[GitLab guide](../../docs/gitlab-ci.mdx#merge-into-an-existing-root-configuration).
Only Tier 1 gates CI; Tier 2 and Tier 3 remain advisory, including failed or
incomplete outcomes, with their native evidence retained.
The CLI's `--tier3-required`, GitHub's `tier3-required` input and GitLab's
`SKILL_EVAL_TIER3_REQUIRED` variable are deprecated and ignored; they cannot
make Tier 3 blocking. Reports expire after 14 days, Docker is the default, and
native local mode is also supported.
