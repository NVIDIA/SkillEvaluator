# Ready-to-use CI configurations

Copy a configuration file into your repository, set your skill folders and
quality threshold, and commit it. No local SkillEvaluator installation is
needed for manual setup.

| Result | Current configuration | Copy destination |
| --- | --- | --- |
| GitHub checks | [skill-checks.yml](1.0.1/skill-checks.yml) | `.github/workflows/skill-checks.yml` |
| GitHub PR comments without hosting | [comment-skill-checks.yml](1.0.1/comment-skill-checks.yml) | Second workflow; follow the [comment setup](1.0.1/README.md#pr-comment-setup) |
| GitHub hosted reports and PR comments | [publish-skill-reports.yml](1.0.1/publish-skill-reports.yml) | Second workflow; follow the [setup instructions](1.0.1/README.md) |
| GitLab checks | [gitlab-skill-checks.yml](1.0.1/gitlab-skill-checks.yml) | `.gitlab-ci.yml`, or merge into your existing pipeline |
| Optional manual LLM checks | [skill-checks-llm.yml](1.0.1/skill-checks-llm.yml) | `.github/workflows/skill-checks-llm.yml` |

Open a file and use GitHub's **Raw** view to copy its full contents. Workflow
commits and the container image digest are already pinned. [Full setup](1.0.1/README.md).

## Download and verification

[Download the current ZIP](1.0.1/skillevaluator-ci-1.0.1.zip). Each version folder
contains the exact generated files, checksums, source/image identities and
scanner verification records. The historical [0.7.0 files](0.7.0/), [0.7.1 files](0.7.1/) [0.7.2 files](0.7.2/) and [0.8.0 files](0.8.0/) remain
available for existing configurations, alongside [0.9.0](0.9.0/); use 1.0.1 for current onboarding.

These manual configurations run Tier 1. The all-tier CLI was already included in
0.9.0; 1.0.1 refreshes its separately packaged schema-version 2 descriptor.
Use `skillevaluator ci init --tiers 1,2,3` for that opt-in, with the matching
release installed. Existing committed workflow pins do not change when the CLI
is upgraded.

## Release convention

Publish one SkillEvaluator product release per version. Attach Python packages,
the verified CI bundle and checksums to the same draft before publishing, and
include the versioned PyPI package link in its release notes. Separate CI release
entries are unnecessary. Source templates for maintainers live in
[ci/templates](../templates/README.md).
