# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001 -- standalone release helper.
"""Produce copyable consumer workflows from an actual released image digest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    import github_api
except ModuleNotFoundError:
    from ci import github_api

TEMPLATES = Path(__file__).parent / "templates"
IMAGE_PLACEHOLDER = "ghcr.io/nvidia/skillevaluator-tier1@sha256:RELEASE_IMAGE_DIGEST"


def render_templates(output: Path, workflow_ref: str, image: str, repository: str = "NVIDIA/SkillEvaluator") -> dict:
    if not github_api.COMMIT_SHA.fullmatch(workflow_ref):
        raise ValueError("A full reviewed workflow commit SHA is required.")
    if not github_api.GHCR_DIGEST.fullmatch(image):
        raise ValueError("A published GHCR image digest is required.")
    if not github_api.REPOSITORY.fullmatch(repository):
        raise ValueError("A GitHub owner/repository is required.")
    if output.exists():
        raise ValueError("Use a new output directory to preserve previous release files.")
    rendered = {}
    for source in sorted(TEMPLATES.glob("*.yml")):
        rendered[source.name] = (
            source.read_text(encoding="utf-8")
            .replace("RELEASE_WORKFLOW_SHA", workflow_ref)
            .replace(IMAGE_PLACEHOLDER, image)
            .replace("RELEASE_IMAGE_DIGEST", image.rsplit(":", 1)[1])
            .replace("NVIDIA/SkillEvaluator", repository)
        )
    scan = rendered["skill-checks.yml"].encode()
    # Equivalent to git hash-object without needing a local Git checkout.
    blob = hashlib.sha1(b"blob " + str(len(scan)).encode() + b"\0" + scan, usedforsecurity=False).hexdigest()
    rendered = {name: text.replace("YOUR_APPROVED_CALLER_BLOB", blob) for name, text in rendered.items()}
    if any("RELEASE_WORKFLOW_SHA" in text or "RELEASE_IMAGE_DIGEST" in text for text in rendered.values()):
        raise ValueError("A template contains an unresolved release pin.")
    output.mkdir(parents=True)
    for name, text in rendered.items():
        (output / name).write_text(text, encoding="utf-8")
    receipt = {"repository": repository, "workflow_ref": workflow_ref, "image": image, "caller_blob": blob}
    (output / "release.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Repository CI onboarding\n\n"
        "## GitHub: choose your templates\n\n"
        "- Checks and downloadable reports: copy `skill-checks.yml` to `.github/workflows/skill-checks.yml`. "
        "Set `paths` to your skill roots and choose `min-score` (default 70).\n"
        "- Hosted reports and PR comments: also copy `publish-skill-reports.yml` to `.github/workflows/` "
        "and complete the one-time setup below.\n\n"
        "The workflow and image pins are already filled in. Both templates use GitHub's automatic token; "
        "no personal token or model key is needed for the default checks. "
        "Your organization must allow the referenced actions and the provided permissions.\n\n"
        "### One-time publisher setup\n\n"
        "1. Enable Settings → Pages → Source: GitHub Actions. Reports can include source excerpts; "
        "a private repository does not automatically make its Pages site private. "
        "Use scan artifacts if hosting is unsuitable. Coordinate any existing Pages site first.\n"
        "2. Initialize a new `gh-pages` history branch with `.nojekyll` and `history.json` containing `[]`. "
        "Preserve existing site content and history.\n"
        "3. Set `YOUR_REPORT_BASE_URL` in the publisher template to your Pages URL.\n"
        "4. Uncomment `id-token: write` under `permissions` in `skill-checks.yml` for signed execution receipts. "
        "After customizing the scan file, run `git hash-object .github/workflows/skill-checks.yml` "
        "and put the result in `trusted-caller-blobs`. Refresh this value after every scan-file edit.\n"
        "5. Merge both files into your default branch, keeping the supplied scan filename, "
        "workflow name and artifact name. The publisher already declares its required permissions.\n\n"
        "New eligible PR scans trigger publication automatically. After successful deployment, the bot "
        "creates or updates one comment on the current, open, same-repository PR. Fork PR comments "
        "and comment-only operation are not supported by this publisher.\n\n"
        "Without the signing permission, ordinary scans and downloadable reports still run; hosting is skipped. "
        "Commit the setup and trigger a new scan; rerunning an old job keeps its original workflow.\n\n"
        f"[Full publisher setup](https://github.com/{repository}/blob/{workflow_ref}/ci/README.md"
        "#optional-hosted-reports-and-pr-feedback).\n\n"
        "## GitLab\n\n"
        "For GitLab, merge gitlab-skill-checks.yml into your root .gitlab-ci.yml and set "
        "SKILL_EVAL_PATHS. Its rules must remain in the root file to enable merge-request pipelines.\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-ref", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--repository", default="NVIDIA/SkillEvaluator")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_templates(args.output, args.workflow_ref, args.image, args.repository)


if __name__ == "__main__":
    main()
