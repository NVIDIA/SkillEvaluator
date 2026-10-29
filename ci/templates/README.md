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

**These source templates contain release placeholders.** Before running them,
use the generated copies attached to an official CI image release on
[NVIDIA/SkillEvaluator Releases](https://github.com/NVIDIA/SkillEvaluator/releases).
Choose a release that explicitly includes the CI onboarding bundle; a regular
Python package release may not include it. If no such release is available, the
official CI integration is not yet ready to install from these templates.

The generated files already contain the reviewed workflow commit and container
image digest. If you copy a source template instead, replace
`RELEASE_WORKFLOW_SHA` and `RELEASE_IMAGE_DIGEST` with the values recorded in that
CI release. Do not leave placeholders or substitute `main` or `latest`.

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
