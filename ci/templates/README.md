# Copy a CI template

Bring SkillEvaluator to your repository by copying a configuration file. You do
not need to install SkillEvaluator locally, use an onboarding CLI, or copy
evaluation code into your project. GitHub and GitLab templates are maintained
together in this public NVIDIA/SkillEvaluator repository.

| Platform and result | Source template | Where to put it |
| --- | --- | --- |
| GitHub: checks and downloadable reports | [skill-checks.yml](skill-checks.yml) | `.github/workflows/skill-checks.yml` |
| GitHub: also update the standard PR comment, without hosting | [comment-skill-checks.yml](comment-skill-checks.yml) | `.github/workflows/comment-skill-checks.yml`, alongside the checks file |
| GitHub: also host reports and update a PR comment | [publish-skill-reports.yml](publish-skill-reports.yml) | `.github/workflows/publish-skill-reports.yml`, alongside the checks file |
| GitHub: optional trusted manual LLM checks | [skill-checks-llm.yml](skill-checks-llm.yml) | `.github/workflows/skill-checks-llm.yml` |
| GitLab: checks and downloadable reports | [gitlab-skill-checks.yml](gitlab-skill-checks.yml) | `.gitlab-ci.yml`, or merge into your existing root configuration |

Open a source file to inspect it; GitHub's **Raw** view lets you copy its full
contents. See [example results](../examples/README.md) without needing access to
a separate demo repository.

## Get a version you can run

**These source templates contain placeholders.** Copy the
[ready-to-use configurations](../bundles/0.9.0/README.md) instead; their reviewed
workflow commits and container image digest are already filled in.
The comments-only workflow is a source capability for a future bundle; the
archived 0.9.0 bundle and released onboarding CLI still offer the hosted publisher.
For comments only, use a reviewed revision containing the new reusable workflow
and follow the [comment setup](../README.md#pr-comments-without-hosted-reports).

Open a file in that folder and use GitHub's **Raw** view to copy its complete
contents. The generated files and their checksums are kept in this public
repository. Future releases attach the same files to the single SkillEvaluator
product release, alongside its Python packages.

If you customize a source template instead, use the immutable workflow/image
identities in the generated bundle's `release.json`. Do not leave placeholders
or substitute `main` or `latest` for runtime pins.

Maintainers can release a reviewed feedback workflow update while keeping the verified
scanner workflow and image unchanged. Pass its full commit SHA as
`--publisher-ref` to `ci/render_templates.py`; otherwise it uses `--workflow-ref`.
That revision must contain both `comment-skill-checks.yml` and
`publish-skill-reports.yml`. The release receipt records both revisions. This
changes the feedback workflows and tools only; the scanner pin and approved
evaluator identity stay unchanged.

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
- **Optional GitHub comments:** choose the
  [comments-only setup](../README.md#pr-comments-without-hosted-reports) or the
  [hosted publisher setup](../README.md#optional-hosted-reports-and-pr-feedback).
  Uncomment `id-token: write` in the scan file for signed execution receipts,
  then approve that file's new blob. Copying the second file alone does not
  enable comments.
- **Optional LLM checks:** configure a provider key in the separate manual
  workflow. The default checks need no model key.

Open a pull request or merge request that changes a skill to see the checks.
Enable the platform's CI service, allow the referenced workflows or includes,
and provide a compatible runner. The [platform guides](../../docs/repository-integration.mdx)
cover results and merge-gating settings.

## Advanced CI

The unreleased, explicitly enabled all-tier integration has separate
[setup and maintainer instructions](all-tiers/README.md). Released templates
above provide quality and security checks without a model key.
