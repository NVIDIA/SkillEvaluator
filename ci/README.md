# Reusable repository CI

This directory supplies the shared CI runtime for quality and security checks,
plus consumer onboarding templates. Start with
[repository integration](../docs/repository-integration.mdx) for GitHub Actions
or GitLab CI/CD. The GitLab adapter is in `ci/gitlab/`;
the image and report generation are shared by both platforms.
The upstream workflow release and public package must be reviewed and published
before consumers can use this path. Source templates are not runnable while
`RELEASE_WORKFLOW_SHA` or `RELEASE_IMAGE_DIGEST` remains unresolved.

For manual setup, start with the [copyable template catalog](templates/README.md).
The [report and feedback examples](examples/README.md) are kept here in the public
repository. Neither requires a local onboarding CLI or personal-demo access.

## Consumer contract

Copy the release-generated `skill-checks.yml` to
`.github/workflows/skill-checks.yml`, then edit `paths`. The shared
`NVIDIA/SkillEvaluator/.github/workflows/evaluate-skills.yml` requires an explicit
immutable `image`, such as
`ghcr.io/nvidia/skillevaluator-tier1@sha256:RELEASE_IMAGE_DIGEST`; the consumer
also pins the workflow with `uses: ...@RELEASE_WORKFLOW_SHA`. Do not substitute a
mutable image tag or an unreviewed branch reference.

The image selects one bundled `default` validation policy for public and private
repositories. Its rules live in `src/skillevaluator/config/profiles/default.yaml`;
there is no `internal`/`external` CI input. Older released images label this same
policy `external`. Upgrading the workflow alone does not change a pinned image's
policy label. Profile names do not control network access or report visibility.

| Input | Basic configuration |
| --- | --- |
| `paths` | `skills`; up to 100 newline-separated repository-relative skill directories or roots |
| `scan-mode` | `auto`: affected skills on PRs, all configured skills on pushes/manual runs; `all` overrides selection |
| `llm` | `false`; static checks require no model key |
| `artifact-name` | Keep `skill-evaluator-tier1` for PR comments or the optional hosted publisher |
| `report-base-url` | Optional own Pages URL, only after hosted publication is configured |
| `scan-workers` | `4`; skills scanned at once, from 1 to 16. `1` scans one skill at a time |
| `enforcement-mode` | `strict` (default) blocks on failed verdicts; `advisory` keeps complete policy/score failures non-blocking but still blocks runtime/evidence/incomplete failures |
| `security-baselines` | `false`; explicitly apply reviewed per-skill SkillSpector baselines as described below |

The runtime discovers exact-case `SKILL.md` files, rejects unsafe paths and
symlinks, deduplicates overlapping roots and bounds each invocation to 100
skills. This is a selection cap, not a runtime guarantee: scans have a bounded
time budget and retain completed results with explicit incomplete entries for
unprocessed skills. Do not interpret a changed-skill selection as a new
full-catalog pass.
Up to `scan-workers` skills are scanned at once, each with its own private home
and temporary directory. Logs, annotations and report rows keep skill order. When
the time budget runs out, running scans stop and skills not yet started are
reported as not evaluated. Ctrl-C also stops running scans at once, including
the scanner tools they started.
With no affected skills, the check succeeds with **No skills changed** and
records that validation did not run. Fully removed skills are listed as
deleted/skipped. A removed manifest with tracked files left in its skill
folder is an error. Selection uses changed files inside discovered skills;
shared files outside the configured roots are not inferred as dependencies.
Use `scan-mode: all` when changes to those shared resources require evaluation.
Changed-skill selection checks out full history without old file contents (a
blobless clone); full scans fetch only the scanned commit.
Enabling reviewed security baselines fetches full, unfiltered history so target
policy files are available without network access inside the scanner image.
Static scanning is network connected: vulnerability databases may receive
package metadata. Dependency auditing never installs packages or runs build
scripts: requirements must provide exact package/version pins. Unpinned, URL,
local-path and included requirements that cannot be audited safely are
reported as incomplete; transitive coverage requires an explicitly pinned
file. Tier 2/3 and plugin evaluation are outside this workflow.
Scanner installation and source evaluation happen in the prebuilt Linux image;
reports are written outside the read-only consumer checkout. Missing scanners,
incomplete scans and missing or contradictory report evidence fail the job.

The short `summary.md` is the Actions landing view. `details.md`, original
per-skill JSON, native HTML, SARIF and scanner logs retain detailed evidence in
the scan artifact, named `skill-evaluator-tier1-attempt-N` for attempt `N`.
A second, small artifact, `skill-evaluator-tier1-evidence-attempt-N`, holds
only `summary.json` and each skill's `skillevaluator-tier1.json`. The comment and
hosted publishing workflows download only that one. Artifacts expire after the configured retention period;
hosted history is an optional separate surface. Quality scores are not scanner
pass percentages or proof of useful live-agent behavior.

Before scanning, CI collects values matched by the bundled credential patterns,
NVIDIA API keys, and unquoted `KEY=value` and `Bearer` credentials across the
selected skills. Raw scanner files stay in private temporary storage. Recognized
values, parts of them cut or escaped by a scanner, and database URL passwords
are removed from retained reports and annotations. Retained Markdown and SARIF
are rendered from the redacted JSON. When a credential or an LLM key is present,
raw scanner logs are withheld; the structured reports remain available. The
preflight is bounded to 10,000 files, 32 MiB per file, 128 MiB per skill and
8,192 characters per line, with bounded redaction values. If it cannot safely
inspect or redact the input, the run reports an error without publishing
detailed scan output. This is not a guarantee that every credential or
transformed value is detected. Remove and rotate exposed credentials; reports
and Git history from previous runs are not rewritten automatically.

## Reviewed security baselines

This source capability requires a supporting workflow and image release; it is
not added retroactively to already published onboarding bundles. Enable
`security-baselines: true` on the GitHub reusable workflow, or set GitLab's
`SKILL_EVAL_SECURITY_BASELINES: "true"`. Both default to false. This option
applies SkillSpector exceptions only; it does not disable PII or security checks,
change the quality gate, or forward the separate CLI `--pii-waivers` option.

If a central component wraps the GitHub workflow, update its approved workflow
and image pins together and forward the `security-baselines` boolean to the
scan job. Each consumer keeps its own reviewed mapping and baseline files.

Commit the fixed manifest `.skillevaluator/skillspector-baselines.yaml`:

```yaml
version: 1
baselines:
  legacy/sample-expert: config/security-baselines/sample-expert.json
```

Keys are exact repository-relative skill directories, and values are
repository-relative native SkillSpector YAML or JSON baseline files. There are
no globs. Unmapped skills receive no baseline. Paths must be canonical, without
absolute paths, parent traversal, control characters, or links; the manifest
rejects duplicate keys, YAML anchors and aliases, and unknown top-level keys.
The manifest is limited to 64 KiB and 1,000 mappings, each baseline to 2 MiB,
and the combined policy files to 16 MiB. The credential preflight also applies
its existing line and redaction limits to the policy files.

Both the mapping and baseline bytes come from the same approved Git commit:
the pull request's target commit on GitHub, or GitLab's target-branch commit.
For detached same-project GitLab merge requests without a target SHA, the job
fetches the actual target branch before scanning; it never substitutes the
merge base or candidate head. Fork GitLab merge requests require an exact
platform-provided target SHA already available in the checkout and fail closed
otherwise. Push/manual runs use their current commit. A pull request proposing
the first manifest must merge through review before this option can use it;
the candidate's new or edited approvals never apply to that same pull request.

Missing policy commits, manifests, files, or invalid mappings fail configuration.
Selected baseline bytes are materialized read-only in private temporary storage
outside the skill tree and passed with `--skillspector-baseline` only to the
mapped skill. The scanner retains its normal validation of the baseline format.
Exact fingerprints still bind the source content and scanner version; regenerate
and review them after either changes, or use deliberate narrow reviewed rules.
Baselines cannot clear incomplete analysis.

The retained `summary.json` records the policy commit, manifest path, digest, and
`mapped_skill_count` in `execution.security_baselines`, and each mapped skill's baseline file and
digest in `skills[].security_baseline`. The skill's native report records whether
the baseline applied and what it suppressed. Policy files receive the same
credential preflight and report redaction as skill content.

## PR comments without hosted reports

Use the canonical
[`comment-skill-checks.yml`](../.github/workflows/comment-skill-checks.yml) reusable
workflow to update the existing scored Tier 1 PR comment without deploying a
report site. It uses the same signed execution receipt, caller and evaluator
approvals, artifact validation, report renderer and bot comment updater as the
hosted publisher. The comment shows counts, status, quality score and grade, and
the first action for each affected skill. The workflow run links to downloadable
reports. No Pages settings, site history branch or report URL are needed.

This is a source capability for a future release bundle. Existing archived
bundles and the released `skillevaluator ci init` still configure hosted
publication. Use a reviewed immutable revision containing the new workflow and
helpers; do not add its filename to an older release pin.

1. Keep `.github/workflows/skill-checks.yml` named **Skill checks**, with one
   reusable scan job and the default **skill-evaluator-tier1** artifact name.
   Enable `id-token: write` alongside `contents: read` for signed execution
   receipts. Keep the reviewed evaluator workflow and image pins.
2. Copy [`comment-skill-checks.yml`](templates/comment-skill-checks.yml) into
   `.github/workflows/`. Pin its `uses:` and `tools-ref` to the reviewed comment
   workflow revision. Set `trusted-evaluator-shas` to the approved scanner
   revision; this may differ from the comment tools revision.
3. After all scan-file edits, run
   `git hash-object .github/workflows/skill-checks.yml` and set
   `trusted-caller-blobs` to that exact value. Review and refresh the blob after
   every later scan-file edit. For shared wrappers, keep the
   [same wrapper approvals and delegation](#calling-through-a-shared-workflow).
4. Merge both files into the default branch and open a PR changing a skill.
   The comment workflow grants only `contents: read`, `actions: read` and
   `pull-requests: write`. Organization policy must allow these permissions and
   the referenced actions. Choose this workflow or the hosted publisher for PR
   feedback; both update the same bot comment.

The second workflow runs after **Skill checks** completes, including failed
scans. Its manual trigger requires the completed `run-id` and `run-attempt`.
It comments only on an open, current-head PR from the same repository. Failed
scans with valid retained native evidence receive the standard failure comment;
failures before evidence upload do not produce a synthetic result. No affected
skills uses the existing **No skills changed** result. Fork PRs, superseded
results, stale PR heads, expired artifacts and unverifiable evidence do not
receive a comment. The signed token's expiry alone does not invalidate a
historical receipt; its signing key and the attempt's artifacts must still be
available.

## Optional hosted reports and PR feedback

This is additional setup, not a `comment: true` switch in the basic scan file.
This publisher couples comments to successful hosted deployment. To use the
same comment without hosting, follow the [comments-only setup](#pr-comments-without-hosted-reports).

1. Decide where reports can live. **Pages sites are public unless explicitly
   configured as private, even when the repository is private.** Private Pages
   requires GitHub Enterprise Cloud; see [Pages access control](https://docs.github.com/en/pages/getting-started-with-github-pages/changing-the-visibility-of-your-github-pages-site).
   Reports include paths, findings and source excerpts from unmerged PRs. Use
   the scan artifact when public hosting is inappropriate and private Pages is
   unavailable. This publisher owns the report site's root
   `index.html`, `history.json`, `runs/`, `main/latest/` and `pr/` routes and
   deploys the entire site. Coordinate an existing repository Pages site before
   enabling it; changing the URL alone does not namespace its output.
2. Enable **Settings → Pages → Source: GitHub Actions**. Create a `gh-pages`
   history branch with `.nojekyll` and `history.json` containing `[]`. Do not
   initialize over an existing site/history checkout.
3. Add the generated `publish-skill-reports.yml` to the consumer's **default
   branch**. Set `report-base-url` to your own Pages URL. Optionally set the same
   URL on the scan job to expose its pending report link; publication follows
   scan completion.
4. Keep the scan filename `.github/workflows/skill-checks.yml`, workflow name
   **Skill checks**, and artifact name **skill-evaluator-tier1**. The dedicated
   publisher validates this caller and accepts exactly one approved referenced
   evaluator workflow, plus explicitly approved wrappers as described below.
   A differently named manual LLM workflow is not accepted
   automatically. The publisher reads only the attempt's
   `skill-evaluator-tier1-evidence-attempt-N` artifact. A scan from an
   evaluator revision that does not upload it cannot be published. Re-running
   that scan reuses its commit and pinned evaluator. Instead, update the scan
   caller to a release that uploads the artifact, approve that evaluator SHA
   and the new caller blob (steps 5 and 6), and start a new scan.
5. Set `trusted-evaluator-shas` to the reviewed evaluator workflow commit. Set
   `tools-ref` to the reviewed immutable commit containing the publisher helpers;
   generated release files supply both. `tools-repository` defaults to
   `NVIDIA/SkillEvaluator`; an explicitly reviewed fork override is available
   for integration testing, not needed for standard onboarding.
6. Generated release files include the caller blob for their unedited scan
   template. For hosting, first uncomment `id-token: write` under `permissions`
   in `.github/workflows/skill-checks.yml`. This lets GitHub sign the workflow's
   execution identity; no personal token or cloud account is needed.
   After **any edits** to that consumer file, compute its exact Git blob:
   `git hash-object .github/workflows/skill-checks.yml`. Put that value in
   `trusted-caller-blobs`. Every later scan-file edit needs review and a new
   approved blob. Preserve older reviewed SHA/blob entries only when needed for
   historical backfill. Both allowlists accept comma- or newline-separated
   entries; compute trust hashes from your own reviewed configuration. The
   publisher verifies a separate GitHub-signed execution receipt and checks the
   caller at that receipt's immutable workflow revision. It never reconstructs
   the executed workflow from a pull request's current base or head.
7. The publisher caller must grant `contents: write`, `actions: read`,
   `pages: write`, `id-token: write` and `pull-requests: write`. Reusable jobs
   narrow these to history preparation, Pages deployment and PR feedback. The
   scan uses `contents: read`, plus `id-token: write` when PR comments or hosting are enabled.
   The automatic GitHub token is sufficient;
   no personal access token or model credential is required for publication.

The publisher rebuilds native HTML from bounded, validated JSON using pinned
trusted tools. It does not execute consumer PR code or trust uploaded HTML as
site code. Its jobs use sparse checkouts of the tools repository: preparation
gets the root files, `ci/` and `src/`; the comment job gets only
`ci/comment_report.py` and `ci/github_api.py`, which use only the standard library. Each completion event binds publication to its recorded run attempt;
manual backfills must specify that attempt. It retains the newest 100 runs by
default (`retention-runs`), within a 512 MiB site budget, plus current
`main/latest/` and open-PR reports. Older unprotected runs are pruned first;
publication stops if protected reports alone exceed the budget. Closed PR
aliases are retired. Retained `runs/<run-id>/<attempt>/` paths are immutable; delayed
runs cannot replace newer aliases. Current default-branch reports use the
`main/latest/` route even if the actual default branch has a different name.
A bot comment is created once and updated in place only for an open,
current-head, same-repository PR after that publication's own deployment. Human
comments are untouched. When an automatic publication commits nothing new to
`gh-pages` because another publisher run already pushed the same run attempt,
its Pages deploy and PR comment are skipped. The run that pushed it deploys and
then comments. Manual runs and re-runs always deploy, so re-running after a
failed deploy still publishes and comments.
A publication skipped because its PR closed or moved after the scan neither
deploys nor comments.
Fork PR scans can run without provider secrets; fork PR bot comments and current
PR aliases are not enabled. Hosting alongside an arbitrary existing Pages
deployment needs further integration work.

### Calling through a shared workflow

The comments-only workflow accepts the same `trusted-wrapper-workflows` and
`delegated-evaluator-wrapper` inputs below, with the same signed receipts and
caller blob approvals. Its wrapper continues to forward scan settings and pins;
the standard comment renderer stays in SkillEvaluator.

Direct consumers need no wrapper configuration. If the scan calls a shared
workflow that then calls SkillEvaluator, use a reviewed publisher revision that
exposes `trusted-wrapper-workflows`. The archived 0.7.1 publisher pin predates
this support; adding the input to that older workflow does not enable it.
Update both the publisher job's `uses:` revision and its `tools-ref` to that
reviewed publisher commit. The evaluator workflow and image can keep their
existing approved pins.

In the consumer's publisher workflow, approve each additional workflow by its
full repository path and immutable commit:

```yaml
with:
  trusted-wrapper-workflows: |-
    your-org/shared-ci/.github/workflows/skill-checks.yml@FULL_COMMIT_SHA
```

Replace `FULL_COMMIT_SHA` with the reviewed 40-character commit. Multiple wrappers
can be separated by newlines or commas. Each referenced wrapper must be approved;
branch references are rejected. Continue approving the actual evaluator
in `trusted-evaluator-shas` and the consumer scan file in `trusted-caller-blobs`.
Review any wrapper update and refresh its pin in both the calling workflow and
publisher approval list; refresh the consumer blob after editing its scan file.

For a central component maintained at a version tag such as `@v3`, the consumer
can explicitly delegate evaluator version selection to that component:

```yaml
with:
  trusted-wrapper-workflows: your-org/shared-ci/.github/workflows/skill-checks.yml@v3
  delegated-evaluator-wrapper: your-org/shared-ci/.github/workflows/skill-checks.yml@v3
  trusted-caller-blobs: YOUR_REVIEWED_SCAN_FILE_BLOB
```

This mode supports one consumer scan job calling one shared component, which
calls one SHA-pinned evaluator in the configured evaluator repository. Omit
`trusted-evaluator-shas` when delegating. The publisher verifies GitHub's
historical tag resolution and the signed evaluator receipt; it never resolves
today's tag to validate an earlier run. Each report records the resolved wrapper
commit. Component maintainers can update evaluator pins behind the approved tag
without changing each consumer; edits to the consumer scan file still need a
new approved blob. Tag delegation grants that component control over evaluator
versions and common policy, so enable it only for a component your team trusts.
Multiple wrapper levels and matrix callers require the explicit SHA allowlists.

`skillevaluator ci init` detects this existing caller, keeps its configuration,
and prepares the publisher and caller approval automatically. The CLI checks
the shared component's identity permission before changing GitHub settings.
For private reports, an administrator first selects **Private** and **GitHub
Actions** in Pages settings. The CLI verifies the assigned private URL and
configures report history and the setup PR. It never creates a public site for
a private repository. Private Pages requires an Enterprise Cloud organization.

The consumer scan job **and every wrapper job calling the next workflow** need:

```yaml
permissions:
  contents: read
  id-token: write
```

An intermediate job that grants only `contents: read` removes the identity
permission, so the scan cannot produce its signed publication receipt. Start a
new scan after configuring the chain. Keep the scan workflow name, filename and
artifact name listed above. Wrappers forward `paths`, thresholds and other scan
choices normally. They do not need PR-write or Pages-write permissions.

The publisher verifies the signed identity of the evaluator job and the exact
consumer workflow. It records approved wrapper identities in the publication
receipt. Private repositories still need appropriately restricted report hosting.

Basic scans do not request identity credentials. If GitHub withholds the signing
permission (including under fork policies), the scan and downloadable reports
remain available, but hosting is skipped with a setup message. Historical runs
without a signed receipt cannot be backfilled. Commit the updated configuration
and trigger a new scan from that revision; rerunning an old job keeps its old
workflow. Receipts are separate,
attempt-specific artifacts and are never copied to the report site. The publisher
checks their signature, repository, attempt, workflow revisions and issuance
time; token expiration does not invalidate historical evidence. If GitHub has
retired the signing key, that evidence can no longer be verified and publication
stops. The receipt establishes execution identity; the approved complete caller
workflow and GitHub artifact metadata also remain required.

## Optional LLM checks

The generated `skill-checks-llm.yml` is a separate trusted manual run. Set
`paths`, choose `llm-provider` (`openai`, `anthropic` or `nv_build`), optionally
set `llm-model`, and explicitly map the repository secret `LLM_API_KEY`. Missing
credentials fail explicitly. When `llm-model` is empty the provider's default model
is used, and that model may not be callable with every key (for example, an NVIDIA
Build key can list a model that still answers HTTP 404). If the LLM checks report
"Model or endpoint unavailable", set `llm-model` to a model your key can call. Avoid passing secrets to untrusted PR content;
the basic PR workflow remains keyless. Provider billing and content transmission
apply. A configured LLM run is not proof of Tier 2 or Tier 3 evaluation.

## Maintainer release procedure

Start with the [product release checklist](../RELEASING.md), which covers when
to rebuild the image, refresh templates and CLI pins, and publish through Kitmaker.

Release publication is deliberate and manual. Consumer files become usable
after the official image is published and verified; source templates contain
placeholders until then.

1. Review and merge source, wrapper, scanner pins, renderer and reusable
   workflows. Run focused selection/report/trust tests plus practical E2E
   checks using the candidate revision. The [public examples](examples/README.md)
   illustrate the report and feedback surfaces; they are not proof that a new
   release has passed end-to-end verification.
2. From a reviewed commit on upstream `main`, manually run **Build and publish
   Tier 1 image** with `publish: true`. Publishing is allowed only from `main`
   in `NVIDIA/SkillEvaluator`; PR runs build and smoke-test without uploading.
   The default `publish: false` builds and verifies without publication.
   A release run builds the image once, smoke-tests that local image and pushes
   the same image. Pushes to `main` also build and smoke-test the image; they
   write the GitHub Actions layer cache that later builds read. Pull request
   runs read that cache but never write it. Release builds do not reuse cached
   layers, so `git` and `ca-certificates` come from the Debian packages current
   at release time.
   Build the shared CI runtime with its required external scanners. The base
   image digest, uv bootstrap, scanner runtime dependencies and SkillSpector
   build dependencies are pinned; scanner installs require package hashes.
   Confirm the resulting digest's source revision and the installed scanner
   versions; do not promote an arbitrary feature-branch build as an upstream
   release.
3. Ensure the GHCR package `ghcr.io/nvidia/skillevaluator-tier1` is **Public**.
   Repository visibility does not make a new container package public; see
   [GitHub's container registry documentation](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).
   Prove an anonymous pull using a clean Docker configuration; an authenticated
   maintainer pull is insufficient evidence for consumer onboarding. Anonymous
   verification and template generation run in a separate `onboarding` job. If
   the first package was private, set its visibility and **Re-run failed jobs**
   to verify the same published digest. A fresh dispatch reuses the existing
   revision tag after checking its source label; it never overwrites that tag.
   Registry authentication or network failures stop publication rather than
   being treated as an absent image.
4. Select the reviewed immutable reusable-workflow/tools revision and the
   verified image digest. Use `ci/render_templates.py` to generate release
   copies of the files in `ci/templates/`, replacing
   `RELEASE_WORKFLOW_SHA` and `RELEASE_IMAGE_DIGEST` with those identities.
   Follow the helper's current `--help` for its CLI. Reject unresolved release
   placeholders or mutable references in generated files. The consumer's Pages
   URL remains their own configuration; the generated caller blob must be
   recalculated after any scan-file edits.
5. Download the workflow's `github-actions-onboarding` artifact (retained for
   90 days). Commit the verified generated files under `ci/bundles/VERSION/`.
   Use **one SkillEvaluator product release per version**: stage the Python
   wheel, source archive, CI bundle and checksums together in that release's
   draft, include the versioned PyPI package link in its notes, then publish
   after all assets are ready. Do not create separate `ci-*` release entries.
   Test those exact generated files in a clean consumer repository. Verify a
   passing collection, an intentional failure, configuration and missing-tool
   failures, affected-skill PR selection and `scan-mode: all`, and report
   preservation. For the optional publisher, test deployment, matching caller
   provenance, one updated comment, current-head/race handling and negative
   trust cases with the consumer's real configuration.
6. Inspect actual Actions summaries, downloaded JSON/HTML/Markdown and, where
   configured, hosted reports and the bot comment. Keep build/source evidence,
   local tests, real GitHub runtime and publication proof separate. Only then
   present the release files as ready for external consumers.

For a separately verified image and reviewed workflow revision, the template
renderer can also be invoked directly (substitute the recorded release values):

```bash
python3 ci/render_templates.py --workflow-ref RELEASE_WORKFLOW_SHA \
  --image ghcr.io/nvidia/skillevaluator-tier1@sha256:RELEASE_IMAGE_DIGEST \
  --output /tmp/github-actions-onboarding
```

Changing the report helper requires both a rebuilt runtime image and a reviewed
`tools-ref` for the publisher. Merely repointing a publisher does not update the
scan's baked code. Existing immutable run paths are not re-rendered on backfill;
use fresh runs to verify a renderer release.

## Real image smoke verification

After building or pulling the image locally, run:

```bash
python3 ci/smoke_image.py --image skillevaluator-tier1:smoke --output /tmp/skill-ci-evidence
```

Use a new or empty output directory. The harness creates a temporary committed
Git repository containing two public synthetic skills and runs the actual image
with the workflow's read-only checkout/root filesystem, unprivileged user and
dropped capabilities. It checks a full two-skill pass, an intentional invalid-name failure, one changed
skill out of two, no affected skills and an absent LLM-key failure. No provider
key is passed and no paid model call is made. It continues after a failing case
so diagnostics for the remaining cases are retained.

Each case retains raw container logs, its actual exit code and a `reports/`
folder. `smoke-receipt.json` records image identity, Git commit IDs, outcomes
and summary hashes. A successful process exit alone is insufficient: the harness
also requires matching JSON counts/verdicts, completeness evidence and HTML,
Markdown, SARIF and scanner logs. Failures in the harness's own acceptance checks
are unit tested separately; only a run against a real image establishes runtime
proof. The script does not publish an image, modify a remote repository or
trigger a GitHub workflow.

## Image layout

`ci/Dockerfile.tier1` is a multi-stage build. The build stages install the
locked environments with uv. The runtime stage copies only the three virtual
environments, Gitleaks, `ci/` and the license files; uv, its cache and the
source tree are not in the published image. Scanner layers come before any
SkillEvaluator source, so a source edit rebuilds only the small project layer.
The runtime filesystem is read-only, so Python bytecode is compiled at build
time, including the standard library. The `.pyc` files are hash-based and do
not embed source timestamps. Installed sources never change inside the image,
so Python does not re-check them on import. Do not bind-mount replacement
Python files over single installed modules; rebuild the image instead.

## Updating scanner locks

`ci/requirements/*.in` records direct scanner inputs; the corresponding `.txt`
files pin every resolved package and its hashes for Linux amd64/Python 3.12.
SkillSpector uses an immutable source archive with a content hash and a separate
locked build environment. Regenerate locks deliberately, review the diff, then
build and run the real image verification:

```bash
for input in ci/requirements/*.in; do
  uv pip compile "$input" --python-version 3.12 \
    --python-platform x86_64-unknown-linux-gnu --generate-hashes \
    --no-annotate --no-header --output-file "${input%.in}.txt"
done
```

The Docker base digest is updated separately. These pins control reviewed
inputs; they do not promise byte-for-byte Docker rebuilds. The exact image
digest must pass verification before release.
