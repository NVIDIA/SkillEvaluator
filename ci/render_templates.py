# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001 -- standalone release helper.
"""Produce copyable consumer workflows from an actual released image digest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

TEMPLATES = Path(__file__).parent / "templates"
IMAGE_PLACEHOLDER = "ghcr.io/nvidia/skillevaluator-tier1@sha256:RELEASE_IMAGE_DIGEST"


def render_templates(output: Path, workflow_ref: str, image: str, repository: str = "NVIDIA/SkillEvaluator") -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", workflow_ref):
        raise ValueError("A full reviewed workflow commit SHA is required.")
    if not re.fullmatch(r"ghcr\.io/[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}", image):
        raise ValueError("A published GHCR image digest is required.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
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
        "# GitHub Actions onboarding\n\n"
        "Copy skill-checks.yml to .github/workflows/skill-checks.yml and set paths to your skill roots. "
        "The workflow and image pins are already filled in. No API key is needed.\n\n"
        "For optional hosted reports and PR comments, follow ci/README.md in the evaluator repository. "
        "Set YOUR_REPORT_BASE_URL in the publisher template. If you edit skill-checks.yml, update "
        "trusted-caller-blobs using git hash-object .github/workflows/skill-checks.yml.\n",
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
