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
the repository's release maintainers, leave **Prevent self-review** disabled,
disable administrator bypass, and allow deployments only from the `main` branch.
A listed maintainer may approve a release they started; each environment still
pauses for an explicit approval. Review the final release notes and all five
validated assets before approving GitHub publication.

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
- For **Full release**, install a dedicated bundle PR GitHub App on SkillEvaluator
  with repository Contents, Pull requests, and Workflows write access. Keep the
  App subject to the normal `main` rules. In a `release-bundle-pr` Actions
  environment restricted to `main`, add the
  `RELEASE_BUNDLE_APP_CLIENT_ID` variable and `RELEASE_BUNDLE_APP_PRIVATE_KEY`
  secret. The workflow checks both are present before it builds an image.
  The App writes generated files to a release branch and opens the PR.

## Release without a GitHub App

Merge the version-preparation PR and run **Build CI image and bundle**. Open
the bundle PR yourself using the generated artifact, then run **Validate and
publish** against its merge SHA. Follow the [CI preparation](#prepare-the-ci-release)
and [publication](#publish-the-product-release) checks and approvals. Leave
**promote_stable** unchecked and follow [manual promotion](#promote-stable-manually)
after the release passes verification. Only the version and bundle changes
need PRs.

## Start an automated release from GitHub Actions

1. Merge the version-preparation PR described below. Open **SkillEvaluator
   Release → Run workflow** on `main`, choose **Full release**, and enter the
   version and the version-preparation merge SHA.
2. Approve the image job. The workflow builds or reuses the immutable image,
   verifies the public digest, generates the bundle and pins, and opens or
   reuses the bundle PR. Its link appears in the run summary.
3. Review the PR and qualify the generated GitHub and GitLab configurations.
   Update the bundle catalog and current onboarding links in the PR if needed.
   Merge the PR after the required checks pass. The run waits up to two hours
   for that merge and verifies the exact squash-merge commit on `main`. If the
   wait expires, merge the PR and use **Re-run failed jobs** on the original run.
   Before the version tag exists, a fresh dispatch can reuse an open or
   merged bundle PR for the same candidate. If intervening changes require a
   new image candidate, close the old bundle PR if it is still open and
   dispatch **Full release** with the new reviewed SHA. The workflow generates
   a separate branch and PR for that candidate.
4. Review the validated bundle merge SHA, artifact ID, and manifest checksum
   in the run summary. Follow the GitHub release and Kitmaker approvals from
   the same run. Publication uses the exact validated artifacts, verifies PyPI
   and DevZone, runs fresh install smoke tests, and finalizes the release.

If the bundle PR is already merged, choose **Validate and publish** with the
version and the bundle merge SHA. This starts at package validation and skips
image building and bundle PR creation.

Select **promote_stable** with **Full release**, **Validate and publish**, or
**Publish release**. After both package indexes pass verification, the workflow
promotes the exact version if it is GitHub Latest and the stable writer App is
configured. Use the standalone promotion workflow for retries and the first
`v1.0.1` bootstrap. Docs publish through **Publish Docs** on pushes to `main`,
including the bundle PR merge.

The individual image, validation and publication phases are also available for
manual qualification and recovery.

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
   source, wheel, sdist, and CI bundle checks. Validation checks the selected
   commit for a dated changelog entry and matches the live public GHCR digest
   against the bundle. On the successful run's **Summary** page, copy both
   values from **Publish release inputs**. `validation_run` includes the run ID
   and attempt (such as `123-1`), and `validation_manifest_sha256` is the
   manifest hash.
2. Choose **Publish release** with the same version and product SHA, enter the
   two values from the summary, and approve the `release-github` job. The
   publish run verifies the selected validation attempt and uses its exact
   wheel, sdist, CI ZIP, manifest, and checksums. If the retained validation
   artifact has expired or is missing before the first draft is staged, run
   **Validate release** again and use both values from the new summary.
   Before creating `vVERSION` at the reviewed product SHA, the workflow
   rechecks the release notes and official GHCR digest and checks both package
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

On interruption, use **Re-run failed jobs** on the original run first. To
start a fresh run, enter the same version and reviewed bundle merge SHA.

- If the tag exists and the GitHub draft was never created, use **Validate and
  publish**. The standalone **Publish release** phase can also continue with
  the receipt from a successful standalone **Validate release** run.
- For an existing draft, choose **Publish release** with blank validation
  inputs. The approved job restores the exact bytes retained by the original
  publishing run. Preserve that artifact until completion.
- For an immutable GitHub release, choose **Publish release** with blank
  validation inputs. The workflow uses the published bytes and resumes any
  recorded Kitmaker submission.

If GitHub rejects the workflow token when creating the version tag, an
authorized maintainer can create that tag at the exact reviewed product SHA
and resume the same run. The workflow checks the tag target on every attempt.

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

Promotion verifies the latest immutable release and checks its CI archive and
runtime identities against the published artifact manifest.
It verifies the GitHub and GitLab stable configurations against the templates
at the release commit. Before moving the `stable` tag, it verifies that public
PyPI and NVIDIA DevZone serve the unyanked wheel and source distribution with
the exact filenames, SHA-256 digests, and sizes in that same manifest.

### Promote stable manually

For the first promotion, a repository administrator must create an active tag
ruleset targeting exactly `stable` that restricts creation, updates, and
deletion. Grant bypass to the designated release maintainers' team. Keep
version-tag protections separate and reserve the name `stable` for the tag.

After **Validate and publish** finishes, use a clean HTTPS checkout of
`NVIDIA/SkillEvaluator` at current `main`, Python 3.11 or later, and a GitHub
CLI login with permission to update the tag. Use the approved
`RELEASE_DEVZONE_INDEX_URL` from the repository Actions variable.

```bash
RELEASE_DEVZONE_INDEX_URL="$(gh variable get RELEASE_DEVZONE_INDEX_URL --repo NVIDIA/SkillEvaluator)" \
  bash ci/promote_stable.sh --manual v1.2.1
```

Substitute the exact reviewed release tag on each promotion. Use
`--verify-only` in place of `--manual` to run the same checks without updating
the tag. The update fails if GitHub Latest differs, either package index lacks
the verified artifacts, or another maintainer moves the tag after verification
starts. GitHub and GitLab consumers following the stable channel pick up the
approved workflow and image on subsequent runs.

For the historical `v1.0.1` bootstrap, if it is still GitHub Latest and
`stable` does not exist, set `STABLE_BOOTSTRAP_SHA` to the full reviewed
`main` commit containing the generated workflows and GitLab include, then
run the command above with `v1.0.1`. The published CI bundle must match
the bootstrap commit.

### Automate stable promotion

Publishing a stable release can signal **Promote stable GitHub workflows**.
Before using App-based promotion, a repository administrator must configure
automation access.

- Set the repository Actions variable `RELEASE_DEVZONE_INDEX_URL` to the approved
  public HTTPS NVIDIA DevZone listing for `/skillevaluator/`.
- Install a dedicated GitHub App on this repository with Contents write access.
  Allow it to bypass only the tag ruleset for `stable`. Reserve the name `stable`
  for the tag and protect the version tags separately.
- Create the `stable-channel` Actions environment and restrict deployments to
  `main`. Add the `STABLE_CHANNEL_APP_CLIENT_ID` variable and the
  `STABLE_CHANNEL_APP_PRIVATE_KEY` secret to that environment.

If the DevZone URL is unset or invalid, package verification stops promotion
before the `stable` tag can move.

For a standalone promotion after the product release finishes verification,
open **Promote stable GitHub workflows** in Actions and select **Run workflow** on `main`. Set
`expected-release` to the reviewed `vX.Y.Z` tag shown in the release workflow
summary. Promotion stops if GitHub Latest differs from that tag. The
product workflow's GitHub Actions token does not trigger the release relay.
If a promotion ran before both indexes served the verified files, rerun it
after package verification finishes. A retry verifies the published release
even if `main` already has the next version's wrappers. Qualify this case with
the release App on GitHub.
For either promotion path, if GitHub refuses to update `stable` to the older
workflow tree, keep the exact release commit at the tip of a protected release
branch and retry.

The immutable `v1.0.1` release predates the stable entry points. If it is still the
latest release at first activation, merge its generated configurations into
`main` and dispatch promotion with `expected-release` set to `v1.0.1` and
`bootstrap-sha` set to the full reviewed commit SHA. The selected commit must
contain the GitLab stable include and the exact published CI bundle. Later
retries of `v1.0.1` verify the existing `stable` target. Subsequent releases
include the generated configurations in their version tags. The workflow
rejects downgrades.

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
