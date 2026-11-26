# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001 -- standalone release helper.

"""Render the stable GitHub and GitLab configurations from a verified Tier 1 release bundle."""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import subprocess
from pathlib import Path
from zipfile import BadZipFile, ZipFile

try:
    import github_api
except ModuleNotFoundError:
    from ci import github_api

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "ci/templates/stable"
GITHUB_OUTPUT = ROOT / ".github/workflows"
GITLAB_INCLUDE = "stable-tier1.yml"
COMMENTS_WORKFLOW = "comment-skill-checks-stable.yml"
WORKFLOWS = ("skill-checks-stable.yml", "publish-skill-reports-stable.yml", COMMENTS_WORKFLOW)
RELEASE_HEADER = "# SkillEvaluator release: "
VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")
OFFICIAL_IMAGE = re.compile(r"ghcr\.io/nvidia/skillevaluator-tier1@sha256:[a-f0-9]{64}")


def render(
    bundle: Path, *, templates: Path = TEMPLATES, include_gitlab: bool = True, include_comments: bool = True
) -> dict[str, str]:
    if not VERSION.fullmatch(bundle.name):
        raise ValueError("A stable channel bundle must have a numeric major.minor.patch version.")
    release = github_api.strict_json((bundle / "release.json").read_bytes())
    receipt = github_api.strict_json((bundle / "verification-receipt.json").read_bytes())
    if not isinstance(release, dict) or release.get("repository") != "NVIDIA/SkillEvaluator":
        raise ValueError("An official NVIDIA/SkillEvaluator bundle is required.")
    for key in ("workflow_ref", "publisher_ref"):
        if not github_api.COMMIT_SHA.fullmatch(str(release.get(key))):
            raise ValueError(f"The release {key} must be a full commit SHA.")
    image = release.get("image")
    if not isinstance(image, str) or not OFFICIAL_IMAGE.fullmatch(image):
        raise ValueError("The release image must use the official Tier 1 GHCR repository and an immutable digest.")
    # The image can come from an earlier verified release when only workflows change.
    if (
        not isinstance(receipt, dict)
        or receipt.get("all_verified") is not True
        or receipt.get("public_manifest_matches_smoke_tested_image") is not True
        or receipt.get("anonymous_pull_verified") is not True
        or receipt.get("image") != image
        or not github_api.COMMIT_SHA.fullmatch(str(receipt.get("source_revision")))
    ):
        raise ValueError("The image must match the release's public verification receipt.")

    names = tuple(name for name in WORKFLOWS if include_comments or name != COMMENTS_WORKFLOW)
    if include_gitlab:
        names = (*names, GITLAB_INCLUDE)
    return {
        name: (
            (templates / name)
            .read_text(encoding="utf-8")
            .replace("RELEASE_TAG", f"v{bundle.name}")
            .replace("RELEASE_WORKFLOW_SHA", release["workflow_ref"])
            .replace("RELEASE_PUBLISHER_SHA", release["publisher_ref"])
            .replace("RELEASE_IMAGE", image)
        )
        for name in names
    }


def verify_publication(
    bundle: Path,
    manifest_path: Path,
    release_commit: str,
    digest: str,
    size: int,
    zip_digest: str,
    zip_size: int,
    *,
    include_gitlab: bool = False,
    include_comments: bool = False,
) -> None:
    """Tie the verified CI bundle and its runtime refs to the immutable GitHub release."""
    raw = manifest_path.read_bytes()
    if len(raw) != size or f"sha256:{hashlib.sha256(raw).hexdigest()}" != digest:
        raise ValueError("The published artifact manifest does not match its GitHub size and digest.")
    manifest = github_api.strict_json(raw)
    release = github_api.strict_json((bundle / "release.json").read_bytes())
    receipt = github_api.strict_json((bundle / "verification-receipt.json").read_bytes())
    if not github_api.COMMIT_SHA.fullmatch(release_commit):
        raise ValueError("The published release must resolve to a full commit SHA.")

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.PIPE).strip()

    if not isinstance(manifest, dict) or any(
        manifest.get(key) != value
        for key, value in {
            "version": bundle.name,
            "source_commit": release_commit,
            "source_tree": git("rev-parse", f"{release_commit}^{{tree}}"),
            "image_source_commit": receipt["source_revision"],
            "image": release["image"],
        }.items()
    ):
        raise ValueError("The published artifact manifest does not match the tagged CI release.")
    filename = f"skillevaluator-ci-{bundle.name}.zip"
    archive = (bundle / filename).read_bytes()
    checksum = hashlib.sha256(archive).hexdigest()
    if len(archive) != zip_size or f"sha256:{checksum}" != zip_digest:
        raise ValueError("The checked-in CI archive does not match the published GitHub asset.")
    if manifest.get("ci_bundle") != {"filename": filename, "size": zip_size, "sha256": checksum}:
        raise ValueError("The published artifact manifest does not match the checked-in CI archive.")
    with ZipFile(io.BytesIO(archive)) as zip_file:
        names = zip_file.namelist()
        for name in ("release.json", "verification-receipt.json"):
            expected = (bundle / name).read_bytes()
            if (
                names.count(name) != 1
                or zip_file.getinfo(name).file_size != len(expected)
                or zip_file.read(name) != expected
            ):
                raise ValueError(f"The published CI archive's {name} differs from the tagged CI bundle.")

    sources = [
        (release["workflow_ref"], ".github/workflows/evaluate-skills.yml"),
        (release["publisher_ref"], ".github/workflows/publish-skill-reports.yml"),
        (receipt["source_revision"], "ci/Dockerfile.tier1"),
    ]
    if include_gitlab:
        sources.append((release["workflow_ref"], "ci/gitlab/skill-evaluation.yml"))
    if include_comments:
        sources.append((release["publisher_ref"], ".github/workflows/comment-skill-checks.yml"))
    for ref, path in sources:
        try:
            git("merge-base", "--is-ancestor", ref, release_commit)
            git("cat-file", "-e", f"{ref}:{path}")
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"The published release does not contain its pinned source for {path}.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Versioned CI bundle directory")
    parser.add_argument("--output", type=Path, default=GITHUB_OUTPUT, help="GitHub workflow output directory")
    parser.add_argument("--templates", type=Path, default=TEMPLATES, help="Directory containing stable templates")
    parser.add_argument(
        "--gitlab-output", type=Path, help="GitLab include path (defaults to ci/gitlab or the custom output directory)"
    )
    parser.add_argument("--check", action="store_true", help="Verify the outputs without changing any files")
    parser.add_argument(
        "--publication", nargs=6, metavar=("MANIFEST", "COMMIT", "DIGEST", "SIZE", "ZIP_DIGEST", "ZIP_SIZE")
    )
    args = parser.parse_args()
    gitlab_dir = ROOT / "ci/gitlab" if args.output == GITHUB_OUTPUT else args.output
    gitlab_output = args.gitlab_output or gitlab_dir / GITLAB_INCLUDE
    # A historical release can omit an entry point it predates. A partially
    # present entry point is invalid, and new bundles must include both files.
    historical = args.check and args.publication is not None
    gitlab_files = ((args.templates / GITLAB_INCLUDE).is_file(), gitlab_output.is_file())
    comment_files = (
        (args.templates / COMMENTS_WORKFLOW).is_file(),
        (args.output / COMMENTS_WORKFLOW).is_file(),
    )
    include_gitlab = not historical or any(gitlab_files)
    include_comments = not historical or any(comment_files)
    try:
        if args.check and include_gitlab and not all(gitlab_files):
            raise ValueError("The GitLab stable template and generated include are both required.")
        if args.check and include_comments and not all(comment_files):
            raise ValueError("The stable comments template and generated workflow are both required.")
        workflows = render(
            args.bundle, templates=args.templates, include_gitlab=include_gitlab, include_comments=include_comments
        )
        if args.publication:
            manifest, commit, digest, size, zip_digest, zip_size = args.publication
            verify_publication(
                args.bundle,
                Path(manifest),
                commit,
                digest,
                int(size),
                zip_digest,
                int(zip_size),
                include_gitlab=include_gitlab,
                include_comments=include_comments,
            )
        for name, content in workflows.items():
            target = gitlab_output if name == GITLAB_INCLUDE else args.output / name
            if args.check:
                if target.read_bytes() != content.encode("utf-8"):
                    raise ValueError(f"{name} differs from the verified release bundle.")
            else:
                target.write_bytes(content.encode("utf-8"))
    except (OSError, ValueError, BadZipFile, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
