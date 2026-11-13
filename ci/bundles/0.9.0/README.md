# Repository CI onboarding

## GitHub: choose your templates

- Checks and downloadable reports: copy `skill-checks.yml` to `.github/workflows/skill-checks.yml`. Set `paths` to your skill roots and choose `min-score` (default 70).
- Hosted reports and PR comments: also copy `publish-skill-reports.yml` to `.github/workflows/` and complete the one-time setup below.

The workflow and image pins are already filled in. Both templates use GitHub's automatic token; no personal token or model key is needed for the default checks. Your organization must allow the referenced actions and the provided permissions.

### One-time publisher setup

1. Enable Settings → Pages → Source: GitHub Actions. Reports can include source excerpts; a private repository does not automatically make its Pages site private. Use scan artifacts if hosting is unsuitable. Coordinate any existing Pages site first.
2. Initialize a new `gh-pages` history branch with `.nojekyll` and `history.json` containing `[]`. Preserve existing site content and history.
3. Set `YOUR_REPORT_BASE_URL` in the publisher template to your Pages URL.
4. Uncomment `id-token: write` under `permissions` in `skill-checks.yml` for signed execution receipts. After customizing the scan file, run `git hash-object .github/workflows/skill-checks.yml` and put the result in `trusted-caller-blobs`. Refresh this value after every scan-file edit.
5. Merge both files into your default branch, keeping the supplied scan filename, workflow name and artifact name. The publisher already declares its required permissions.

New eligible PR scans trigger publication automatically. After successful deployment, the bot creates or updates one comment on the current, open, same-repository PR. Fork PR comments and comment-only operation are not supported by this publisher.

Without the signing permission, ordinary scans and downloadable reports still run; hosting is skipped. Commit the setup and trigger a new scan; rerunning an old job keeps its original workflow.

[Full publisher setup](https://github.com/NVIDIA/SkillEvaluator/blob/ba7779ad617c6444c98b6241a75bce92067e9d56/ci/README.md#optional-hosted-reports-and-pr-feedback).

## GitLab

For GitLab, merge gitlab-skill-checks.yml into your root .gitlab-ci.yml and set SKILL_EVAL_PATHS. Its rules must remain in the root file to enable merge-request pipelines.
