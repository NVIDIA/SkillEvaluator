# Releasing SkillEvaluator

Publish one product release containing the Python packages and verified CI
configuration. Users normally install the package and run `skillevaluator ci
init`; image builds and template generation are maintainer work.

## Configure release access

Use **Actions → SkillEvaluator Release → Run workflow** and select `main`.
The workflow accepts a bare `X.Y.Z` version, a full source commit SHA, and a
phase. `Validate release` is the default. PRs, pushes, and tags never trigger
the product release workflow.

Before merging the release workflows, create the `release-image`,
`release-github`, `release-kitmaker-dry-run`, and `release-kitmaker`
GitHub Actions environments. For each environment, require approval from
the repository's release maintainers, enable **Prevent self-review**, disable
administrator bypass, and allow deployments only from the `main` branch.

Before the first release, enable immutable releases and set
`RELEASE_DEVZONE_INDEX_URL` to the approved public HTTPS NVIDIA DevZone
listing for `/skillevaluator/`.

- Grant this repository's Actions token write access to the official public
  GHCR package. The workflow creates the version tag and writes the GitHub
  release using its scoped job token.
- Set the approved `KITMAKER_TELEPORT_VERSION`, `KITMAKER_FERRY_PROXY`, and
  `KITMAKER_FERRY_JOIN_TOKEN` variables in both Kitmaker environments. Enroll
  the upstream release workflow in Charon Ferry. The submission endpoint
  validates the wheel and source distribution during a dry run and publishes
  them during an approved upload. The status endpoint reports each result.
- Configure `KITMAKER_VAULT_URL`, `KITMAKER_VAULT_AUTH_MOUNT`,
  `KITMAKER_VAULT_ROLE`, and `KITMAKER_VAULT_AUDIENCE` on each Kitmaker
  environment. Bind each Vault JWT role to the approved repository and
  environment, `refs/heads/main`, caller workflow `release.yml`, and reusable
  workflow `release-kitmaker.yml`. Confirm the approved Vault URL is reachable
  from the GitHub-hosted runner. Set `KITMAKER_VAULT_NAMESPACE` and
  `KITMAKER_VAULT_CA_PEM` where required by the Vault deployment.
- Both roles read the KV v2 data API path in `KITMAKER_VAULT_CONFIG_PATH`.
  The record holds `project_id`, `owner_email`, and the authorized Kitmaker
  service `token`. Grant both roles read and compare-and-set write access
  below the separate KV v2 data API prefix in `KITMAKER_VAULT_RECEIPT_PATH`.
  Each version has one receipt key. Confirm the project ID and owner in
  Kitmaker and run the protected dry-run job successfully before approving
  the first upload. The checked-in dry-run job sends `upload: false`.
  Apply that restriction in Ferry policy as well if the tenant supports it.

## Prepare the CI release

1. In a release-preparation PR, update the package version, its import fallback,
   the editable lock entry, the citation and its documentation example, and any
   version-specific starter expectations. Move the complete Unreleased changelog
   into a dated version section. Review the changes and pass required CI before
   merging.
2. In the release workflow UI, choose **Build CI image and bundle** and enter
   the version-preparation merge SHA. Approve the `release-image` job. The
   workflow verifies the official Linux amd64 runtime and anonymously pulls and
   retests the exact published digest. It retains the scanner inventory and
   verification receipt.
3. Download the `ci-bundle-VERSION` artifact. Commit its versioned files under
   `ci/bundles/VERSION/`, both descriptors under `src/skillevaluator/`, and the
   generated workflow wrappers under `.github/workflows/` and GitLab stable
   include under `ci/gitlab/` through a follow-up PR. Update the bundle catalog
   and current onboarding links. Test the exact
   generated files in clean GitHub and GitLab consumers for pass/fail,
   all/changed/no-change selection, missing prerequisites, and retained reports.
   Qualify optional trusted publishing and provider-backed tiers when affected.
4. Merge the pin PR and freeze its merge SHA as the product release source.
   If candidate runtime, scanner, dependency, or reusable-workflow inputs
   changed after the image build, qualify a new image source and refresh the
   bundle first.

See [CI maintainer instructions](ci/README.md#maintainer-release-procedure) for
image and consumer qualification details.

## Publish the product release

1. Open the workflow in the GitHub UI again, choose **Validate release**, and
   enter the version and product release SHA from the bundle PR. Review the
   source, wheel, sdist, and CI bundle checks. On the successful run's
   **Summary** page, copy `validation_run` (the run ID and attempt, such as
   `123-1`) and `validation_manifest_sha256` from **Publish release inputs**.
2. Choose **Publish release** with the same version and product SHA, enter the
   two values from the summary, and approve the `release-github` job. The
   publish run verifies the selected validation attempt and uses its exact
   wheel, sdist, CI ZIP, manifest, and checksums. If the retained validation
   artifact has expired or is missing before the first draft is staged, run
   **Validate release** again and use both values from the new summary.
   Before creating `vVERSION` at the reviewed product SHA, the workflow
   checks the release notes, the official GHCR digest, and both package
   indexes for version conflicts. It verifies the staged bytes before
   publishing and verifies anonymous downloads afterward. The notes show that
   package publication is pending. Release runs execute one at a time across
   versions and phases. GitHub retains up to 100 pending dispatches. A
   backfilled version stays off **Latest** when a newer release already holds
   it.
3. Approve `release-kitmaker-dry-run` to validate both anonymous Python asset
   URLs. Review the results and approve `release-kitmaker` after both pass. The job submits those URLs
   for both public PyPI and NVIDIA DevZone and records its request UUID in the
   private Vault receipt. The workflow verifies the exact index-hosted files,
   fresh pip and uv installations, a Tier 1 report, and the installed CI pins
   before recording verification in the GitHub release notes.

On interruption, use **Re-run failed jobs** or open **Run workflow** with the
same version, product SHA, and **Publish release** phase. Supply the validation
inputs if a GitHub draft has yet to be created. For an existing draft, the
approved job restores the bytes retained by the original Publish release run.
Preserve that artifact until completion. An immutable GitHub release uses its
published bytes and resumes any recorded Kitmaker submission. Both recovery
paths accept blank validation inputs. If GitHub rejects the workflow token
when creating the version tag, an authorized maintainer can create that tag at
the exact reviewed product SHA and resume the same run. The workflow checks
the tag target on every attempt.

For a Kitmaker timeout or partial publication, inspect the private receipt,
Kitmaker status and both public indexes. The workflow polls a saved submission
and records the terminal result for each artifact. A recorded success is
excluded from retries while the indexes converge.

For a receipt stuck at `submitting`, reconcile the exact project, URLs, and
upload mode against Portal history. With authorized compare-and-set access,
retain the existing `urls` and `upload` values, add the matched Portal `uuid`,
and set `status` to `submitted`. Leave `post_response_uuid` absent so the
workflow validates the mode and URLs echoed by Portal. If Portal confirms
there was no accepted request and the indexes agree, use `status: failed`
with an empty `job_statuses` map and no UUID.

Retry a failed artifact only when both indexes lack it and Kitmaker has
reported no prior success for it. Escalate a hash conflict or a file present
on only one index after the submission finishes. The GitHub release stays
pending while verification is incomplete.

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

After the manual product release finishes verification, open **Promote stable
GitHub workflows** in Actions and select **Run workflow** on `main`. The
product workflow's GitHub Actions token does not trigger the release relay.
A retry verifies the published release even if `main` already has the next
version's wrappers. Qualify this case with the release App on GitHub.
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

Qualify the release commit before publishing, since publication triggers
promotion. For the first `v1.0.1` promotion, qualify the bootstrap commit before
dispatching the workflow. In a test GitLab project, use the consumer root
configuration from the GitLab CI guide and replace `refs/tags/stable` in its
include URL with the full commit SHA you are qualifying.

Run GitLab CI Lint and verify changed-skill selection in a detached merge
request, HTML and `skill-evaluator-ci.json` artifact retention after an
intentional scan failure, and a full scan on the default branch. Check that the
metadata records the release, evaluator commit, image digest, consumer commit,
pipeline ID, and job ID. Run a new pipeline after switching the test include to
a different qualified release. A retry of the original job should keep its
original release and configuration.
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
