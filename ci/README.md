# Reusable GitHub Actions CI

This directory supplies the public GitHub CI runtime and consumer
onboarding templates. Start with the [short GitHub Actions guide](../docs/github-actions.mdx).
The upstream workflow release and public package must be reviewed and published
before consumers can use this path. Source templates are not runnable while
`RELEASE_WORKFLOW_SHA` or `RELEASE_IMAGE_DIGEST` remains unresolved.

## Consumer contract

Copy the release-generated `skill-checks.yml` to
`.github/workflows/skill-checks.yml`, then edit `paths`. The shared
`NVIDIA/SkillEvaluator/.github/workflows/evaluate-skills.yml` requires an explicit
immutable `image`, such as
`ghcr.io/nvidia/skillevaluator-tier1@sha256:RELEASE_IMAGE_DIGEST`; the consumer
also pins the workflow with `uses: ...@RELEASE_WORKFLOW_SHA`. Do not substitute a
mutable image tag or an unreviewed branch reference.

| Input | Basic configuration |
| --- | --- |
| `paths` | `skills`; newline-separated repository-relative skill directories or roots |
| `scan-mode` | `auto`: affected skills on PRs, all configured skills on pushes/manual runs; `all` overrides selection |
| `llm` | `false`; static Tier 1 requires no model key |
| `artifact-name` | Keep `skill-evaluator-tier1` for the optional publisher |
| `report-base-url` | Optional own Pages URL, only after hosted publication is configured |

The runtime discovers exact-case `SKILL.md` files, rejects unsafe paths and
symlinks, deduplicates overlapping roots and bounds each invocation to 100
skills. This is a selection cap, not a runtime guarantee: scans have a bounded
time budget and retain completed results with explicit incomplete entries for
unprocessed skills. Do not interpret a changed-skill selection as a new
full-catalog pass.
With no affected skills, the check succeeds with **No skills changed** and
records that validation did not run. Fully removed skills are listed as
deleted/skipped. A removed manifest with tracked files left in its skill
folder is an error. Selection uses changed files inside discovered skills;
shared files outside the configured roots are not inferred as dependencies.
Use `scan-mode: all` when changes to those shared resources require evaluation.
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
Artifacts expire after the configured retention period;
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

## Optional hosted reports and PR feedback

This is additional setup, not a `comment: true` switch in the basic scan file.
The current implementation couples comments to successful hosted deployment;
Pages is not an inherent requirement for GitHub PR comments.

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
   evaluator workflow. A differently named manual LLM workflow is not accepted
   automatically.
5. Set `trusted-evaluator-shas` to the reviewed evaluator workflow commit. Set
   `tools-ref` to the reviewed immutable commit containing the publisher helpers;
   generated release files supply both. `tools-repository` defaults to
   `NVIDIA/SkillEvaluator`; an explicitly reviewed fork override is available
   for integration testing, not needed for standard onboarding.
6. Generated release files include the caller blob for their unedited scan
   template. After **any edits** to that consumer file, compute its exact Git blob:
   `git hash-object .github/workflows/skill-checks.yml`. Put that value in
   `trusted-caller-blobs`. Every later scan-file edit needs review and a new
   approved blob. Preserve older reviewed SHA/blob entries only when needed for
   historical backfill. Both allowlists accept comma- or newline-separated
   entries; do not copy the personal demo's trust hashes. The publisher verifies
   the caller from recorded PR base/head revisions. Conflicting workflow edits
   or unavailable historical metadata are rejected: update/rebase the consumer
   branch and rerun its scan.
7. The publisher caller must grant `contents: write`, `actions: read`,
   `pages: write`, `id-token: write` and `pull-requests: write`. Reusable jobs
   narrow these to history preparation, Pages deployment and PR feedback. The
   scan remains `contents: read`. The automatic GitHub token is sufficient;
   no personal access token or model credential is required for publication.

The publisher rebuilds native HTML from bounded, validated JSON using pinned
trusted tools. It does not execute consumer PR code or trust uploaded HTML as
site code. Each completion event binds publication to its recorded run attempt;
manual backfills must specify that attempt. It retains the newest 100 runs by
default (`retention-runs`), within a 512 MiB site budget, plus current
`main/latest/` and open-PR reports. Older unprotected runs are pruned first;
publication stops if protected reports alone exceed the budget. Closed PR
aliases are retired. Retained `runs/<run-id>/<attempt>/` paths are immutable; delayed
runs cannot replace newer aliases. Current default-branch reports use the
`main/latest/` route even if the actual default branch has a different name.
A bot comment is created once and updated in place only for an open,
current-head, same-repository PR after deployment. Human comments are untouched.
Fork PR scans can run without provider secrets; fork PR bot comments and current
PR aliases are not enabled. Comment-only operation and hosting alongside an
arbitrary existing Pages deployment need further integration work.

## Optional LLM checks

The generated `skill-checks-llm.yml` is a separate trusted manual run. Set
`paths`, choose `llm-provider` (`openai`, `anthropic` or `nv_build`), optionally
set `llm-model`, and explicitly map the repository secret `LLM_API_KEY`. Missing
credentials fail explicitly. Avoid passing secrets to untrusted PR content;
the basic PR workflow remains keyless. Provider billing and content transmission
apply. A configured LLM run is not proof of Tier 2 or Tier 3 evaluation.

## Maintainer release procedure

Release publication is deliberate and manual. Consumer files become usable
after the official image release; source templates contain placeholders until
then.

1. Review and merge source, wrapper, scanner pins, renderer and reusable
   workflows. Run focused selection/report/trust tests plus practical E2E
   checks. A personal-fork demo is prior evidence, not proof of this new upstream
   revision. Its [consumer](https://github.com/chrisknvidia/skill-evaluator-consumer-demo)
   illustrates the earlier full-path scan and hosted-feedback flow; it does not
   establish changed-skill selection for a newer release.
2. From a reviewed commit on upstream `main`, manually run **Build and publish
   Tier 1 image** with `publish: true`. Publishing is allowed only from `main`
   in `NVIDIA/SkillEvaluator`; PR runs build and smoke-test without uploading.
   The default `publish: false` builds and verifies without publication.
   Build the complete Tier 1 runtime with required external scanners. The base
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
   copies of the three files in `ci/templates/`, replacing
   `RELEASE_WORKFLOW_SHA` and `RELEASE_IMAGE_DIGEST` with those identities.
   Follow the helper's current `--help` for its CLI. Reject unresolved release
   placeholders or mutable references in generated files. The consumer's Pages
   URL remains their own configuration; the generated caller blob must be
   recalculated after any scan-file edits.
5. Download the workflow's `github-actions-onboarding` artifact (retained for
   90 days) and attach that generated bundle and image/source identity to the
   official release for a durable public download.
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
