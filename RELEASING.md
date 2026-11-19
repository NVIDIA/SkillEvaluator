# Releasing SkillEvaluator

Publish one product release containing the Python packages and verified CI
configuration. Users normally install the package and run `skillevaluator ci
init`; image builds and template generation are maintainer work.

## Prepare and validate

1. Select the reviewed source revision and update the package version, import
   fallback, editable lock entry, citation and documentation citation example.
   Move the complete Unreleased changelog into the dated release section.
2. Run the locked full test suite, lint, package build, strict distribution
   metadata checks, OSS source/archive checks, benchmark provenance checks and
   documentation validation. Install the built wheel and source distribution
   in clean supported Python environments and exercise actual CLI/report output.
3. Review edge cases, option interactions, output correctness and regressions.
   Resolve required current-head CI failures and review threads before merging.

## Build the CI release before publishing Python packages

For a release that changes the installed evaluator, runtime dependencies or
scanners, build and verify a fresh official CI image. Workflow/publisher-only
or documentation-only releases can reuse a compatible verified image; record
the reuse and its source identity explicitly. Review the current template
catalog and packaged CLI descriptors for every product release.

1. Merge the validated release source to `main`. The publishing workflow accepts
   only upstream `main`; ordinary PR and main-push builds verify without publishing.
2. Run **Build and publish Tier 1 image** with `publish=true`:

   ```bash
   gh workflow run publish-tier1-image.yml \
     --repo NVIDIA/SkillEvaluator --ref main -f publish=true
   ```

   Record the run's exact source SHA. Wait for image verification, publication
   and the onboarding job's anonymous pull of the resulting immutable digest.
3. Download `github-actions-onboarding` and the image verification evidence.
   The onboarding job renders the templates with the reviewed workflow SHA and
   published image digest. For separate workflow/publisher revisions, use:

   ```bash
   python3 ci/render_templates.py --workflow-ref REVIEWED_FULL_SHA \
     --publisher-ref REVIEWED_PUBLISHER_FULL_SHA \
     --image ghcr.io/nvidia/skillevaluator-tier1@sha256:VERIFIED_DIGEST \
     --output /tmp/skillevaluator-onboarding
   ```

   Use a new output directory. The renderer writes configurations, README and
   `release.json`; prepare the versioned ZIP, checksums, scanner inventory and
   verification receipt separately.
4. Commit the verified files under `ci/bundles/VERSION/` through a follow-up PR.
   Update `ci/bundles/README.md`, current onboarding links and the packaged
   `src/skillevaluator/ci-release.json` to the same verified image/workflow pins.
   Preserve historical version folders. If releasing all-tier onboarding, review
   its separate schema-version 2 descriptor, workflow/tools refs, capabilities,
   evaluator version and runtime-lock hash together.
5. Test the exact generated files in clean consumers, including pass/failure,
   changed/all/no-change selection, missing prerequisites and retained reports.
   Test optional trusted publishing and provider-backed tiers separately; record
   unavailable integration checks explicitly.
6. Merge those pin/bundle changes, freeze the final source revision, then rebuild
   and validate the Python packages so `ci init` contains the final descriptors.
   Keep the image's source SHA distinct from the later bundle/metadata commit.

See [CI maintainer instructions](ci/README.md#maintainer-release-procedure) for
image publication, scanner verification and consumer qualification details.

## Publish the product release

1. Create the version tag at the final reviewed source revision. Stage one draft
   GitHub release with the reviewed notes, wheel, source distribution, CI ZIP,
   artifact manifest and checksums. Download and verify every asset before
   publishing; immutable release assets cannot be replaced afterward.
2. Publish the completed GitHub release. Verify anonymous artifact downloads and
   the exact image link/digest. Include direct versioned YAML links and the
   versioned PyPI package URL in the notes, marking PyPI pending until verified.
3. Use the approved Kitmaker project to dry-run both exact Python artifact URLs.
   After both checks complete, submit those same URLs for managed publication.
   Retain submission receipts; inspect status before retrying an ambiguous request.
4. Verify index-hosted hashes and fresh pip/uv installations, dependencies, CLI
   version and real reports. Then update the editable release notes from pending
   to published and verified, with any validation limits stated.

## Promote the stable channel

Generate the GitHub entry points and GitLab stable include from the verified
Tier 1 bundle as part of the release-bundle PR.

```bash
python3 ci/stable_channel.py ci/bundles/VERSION
python3 ci/stable_channel.py ci/bundles/VERSION --check
python scripts/check_oss_boundary.py --root . --allowlist config/oss_boundary_allowlist.json
```

The source-boundary check scans tracked text files across the repository,
including the generated workflows, promotion scripts and documentation. CI runs
it for both code and docs-only changes.

Qualify the exact generated references in a disposable GitHub repository.
Check a credentials-free scan, a failing same-repository PR scan with signed
evidence and a successful private Pages deployment that updates **Open full
report**. Test a publisher from the candidate release against retained evidence
from the previous stable scanner. The scan and its feedback can resolve the
channel at different times. Include queued scans, reruns and the full artifact
retention period when reviewing a publisher compatibility change.

Publishing a stable release signals **Promote stable GitHub workflows**. The
promotion runs from `main`, verifies the latest immutable release and checks
its CI archive and runtime identities against the published artifact manifest.
It verifies the GitHub and GitLab stable configurations against the templates
at the release commit and moves the single `stable` tag to that commit. Before
activation, install a dedicated GitHub App on this repository with Contents
write access. Allow it to bypass only the tag ruleset for `stable`. In the
`stable-channel` Actions environment, restrict deployments to `main` and set the
`STABLE_CHANNEL_APP_CLIENT_ID` variable and `STABLE_CHANNEL_APP_PRIVATE_KEY`
secret. Reserve the name `stable` for the tag and protect the version tags
separately.

Run the same workflow on `main` to retry promotion after a release created by an
automation token. A retry verifies the published release even if `main` already
has the next version's wrappers. Qualify this case with the release App on GitHub.
If GitHub refuses to update `stable` to the older workflow tree, keep the exact
release commit at the tip of a protected release branch and retry.

The immutable `v1.0.1` release predates the stable entry points. If it is still the
latest release at first activation, merge its generated configurations into
`main` and dispatch promotion with the full reviewed commit SHA as
`bootstrap-sha`. The selected commit must contain the GitLab stable include
and the exact published CI bundle. Later retries of `v1.0.1` verify the existing
`stable` target. Subsequent releases include the generated configurations in
their version tags. The workflow rejects downgrades.

## Qualify the GitLab Tier 1 stable include

The release generator writes `ci/gitlab/stable-tier1.yml` from the immutable
evaluator commit and scanner image digest in the same Tier 1 bundle used for
GitHub. Include that file and its source template in the release commit. The
consumer's root `.gitlab-ci.yml` retains the merge-request rules.
[GitLab CI setup](docs/gitlab-ci.mdx#follow-the-latest-stable-release)

Before promotion, run GitLab CI Lint with a consumer root configuration and
verify changed-skill selection in a detached merge request, HTML and
`skill-evaluator-ci.json` artifact retention after an intentional scan failure,
and a full scan on the default branch. Check that the metadata records the
release, evaluator commit, image digest, consumer commit, pipeline ID, and job
ID. Verify that a new pipeline can pick up the promotion while a job retry keeps
the include snapshot from its original pipeline.
[GitLab include behavior](https://docs.gitlab.com/ci/yaml/#include)

If `stable` already serves a GitHub-only release, leave that release's target in
place. GitLab starts with the next qualifying release. Once published through
the channel, later stable promotions must retain the GitLab include. Consumers
can pin a previous verified bundle while a corrected release is prepared.

Upgrade all-tier setups with trusted merge-request feedback through reviewed
changes to the immutable configuration, tools, and image pins.
[All-tier GitLab setup](ci/templates/all-tiers/README.md#gitlab-checks-and-feedback)

## User-facing instructions

Keep normal GitHub onboarding short:

```bash
python -m pip install --upgrade skillevaluator
skillevaluator ci init
```

The CLI generates the workflow from its packaged descriptor; the consumer's CI
pulls the prebuilt image. Manual alternatives are in the
[current template catalog](ci/bundles/README.md). Generated files keep immutable
pins; existing workflows are not silently upgraded by installing a newer CLI.
