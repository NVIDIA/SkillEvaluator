# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001, T201

"""Stage, verify, and resume the manually approved SkillEvaluator GitHub release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.request import urlopen

try:
    import product_release
    import release_image
except ModuleNotFoundError:
    from ci import product_release, release_image

REPOSITORY = "NVIDIA/SkillEvaluator"
API_ROOT = f"repos/{REPOSITORY}"
GITHUB = f"https://github.com/{REPOSITORY}"
GITHUB_RAW = f"https://raw.githubusercontent.com/{REPOSITORY}"
STAGING_MARKER = re.compile(r"<!-- skillevaluator-stage: ([1-9][0-9]*)-([1-9][0-9]*) -->")
PENDING = "**Package publication:** Pending PyPI and NVIDIA DevZone verification."
VERIFIED = "**Package publication:** Verified on PyPI and NVIDIA DevZone."


def gh(*args: str, data: str | None = None) -> str:
    return subprocess.run(["gh", *args], text=True, input=data, stdout=subprocess.PIPE, check=True).stdout


def api(path: str, method: str = "GET", data: dict | None = None):
    args = ["api", "--method", method, f"{API_ROOT}/{path}"]
    if data is not None:
        args.extend(("--input", "-"))
    response = gh(*args, data=json.dumps(data) if data is not None else None)
    return json.loads(response) if response else None


def get_release(tag: str) -> dict | None:
    # The release-by-tag API can exclude drafts. List with authenticated access
    # so an interrupted upload resumes the same draft.
    pages = json.loads(gh("api", "--paginate", "--slurp", f"{API_ROOT}/releases?per_page=100"))
    matches = [release for page in pages for release in page if release["tag_name"] == tag]
    if len(matches) > 1:
        raise ValueError(f"Multiple GitHub releases for {tag}")
    return matches[0] if matches else None


def tag_target(tag: str) -> str | None:
    ref = f"refs/tags/{tag}"
    lines = subprocess.check_output(["git", "ls-remote", "--tags", GITHUB, ref, f"{ref}^{{}}"], text=True).splitlines()
    refs = {git_ref: sha for sha, git_ref in (line.split("\t", 1) for line in lines)}
    return refs.get(f"{ref}^{{}}", refs.get(ref))


def check_tag(tag: str, source_sha: str) -> str | None:
    target = tag_target(tag)
    if target is not None and target != source_sha:
        raise ValueError(f"GitHub tag {tag} points to a different commit: {target}")
    return target


def preflight(version: str, source_sha: str, main_sha: str, phase: str) -> None:
    product_release.names_for(version)
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("Enter a full lowercase 40-character source SHA.")
    history = subprocess.check_output(["git", "rev-list", "--first-parent", main_sha], text=True)
    if source_sha not in history.splitlines():
        raise ValueError("Select a reviewed commit on upstream main's first-parent history.")
    contents = subprocess.check_output(["git", "show", f"{source_sha}:pyproject.toml"], text=True)
    if tomllib.loads(contents)["project"]["version"] != version:
        raise ValueError("Selected commit's package version does not match the workflow input.")
    tag = f"v{version}"
    if phase == "Build CI image and bundle" and check_tag(tag, source_sha) is not None:
        raise ValueError(f"{tag} already exists. Use the reviewed product SHA to validate or resume publication.")
    if phase != "Build CI image and bundle":
        check_tag(tag, source_sha)


def expected_filenames(version: str) -> set[str]:
    return {*product_release.names_for(version), product_release.MANIFEST, product_release.CHECKSUMS}


def release_assets(release: dict, version: str) -> dict[str, dict]:
    assets = {asset["name"]: asset for asset in release["assets"]}
    if len(assets) != len(release["assets"]) or set(assets) - expected_filenames(version):
        raise ValueError("GitHub release has unexpected or duplicate assets.")
    return assets


def download_asset(asset: dict, output: Path) -> None:
    with output.open("xb") as stream:
        subprocess.run(
            ["gh", "api", "-H", "Accept: application/octet-stream", f"{API_ROOT}/releases/assets/{asset['id']}"],
            stdout=stream,
            check=True,
        )


def restore(version: str, source_sha: str, output: Path) -> bool:
    tag = f"v{version}"
    target = check_tag(tag, source_sha)
    release = get_release(tag)
    if release is None:
        return False
    if target is None:
        raise ValueError(f"Existing GitHub release is missing tag {tag}.")
    assets = release_assets(release, version)
    if not release["draft"]:
        if (
            not release.get("immutable")
            or set(assets) != expected_filenames(version)
            or any(asset["state"] != "uploaded" for asset in assets.values())
        ):
            raise ValueError("Expected an immutable GitHub release with all required assets.")
        output.mkdir()
        for name, asset in assets.items():
            download_asset(asset, output / name)
        return True

    match = STAGING_MARKER.search(release["body"] or "")
    if match is None:
        raise ValueError("Draft lacks its original run artifact. Reconcile the draft before retrying.")
    run, attempt = match.groups()
    origin = api(f"actions/runs/{run}/attempts/{attempt}")
    if (
        origin["id"] != int(run)
        or origin["run_attempt"] != int(attempt)
        or origin["event"] != "workflow_dispatch"
        or origin["path"] != ".github/workflows/release.yml"
        or origin["head_branch"] != "main"
        or (origin["head_repository"] or {}).get("full_name") != REPOSITORY
    ):
        raise ValueError("The draft artifact must originate from an upstream manual release run.")
    gh("run", "download", run, "--name", f"product-release-{version}-{run}-{attempt}", "--dir", str(output))
    return True


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_local(version: str, source_sha: str, assets: Path) -> dict:
    product_release.verify_assets(version, source_sha, assets)
    manifest = json.loads((assets / product_release.MANIFEST).read_text(encoding="utf-8"))
    image = manifest["image"]
    published = release_image.resolve_digest(image.partition("@")[0], manifest["image_source_commit"])
    if published != image:
        raise ValueError("The official public image revision does not match the reviewed release digest.")
    return {asset["filename"]: asset for asset in manifest["artifacts"]}


def release_notes(version: str, assets: Path) -> str:
    changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")
    title = re.search(rf"^## {re.escape(version)} - \d{{4}}-\d{{2}}-\d{{2}}\s*$", changelog, re.MULTILINE)
    if title is None:
        raise ValueError(f"CHANGELOG.md has no dated entry for {version}.")
    changes = re.split(r"^## ", changelog[title.end() :], maxsplit=1, flags=re.MULTILINE)[0].strip()
    tag = f"v{version}"
    manifest = json.loads((assets / "artifact-manifest.json").read_text(encoding="utf-8"))
    bundle = f"ci/bundles/{version}"
    return (
        f"{PENDING}\n\n"
        f"[PyPI](https://pypi.org/project/skillevaluator/{version}/) · "
        f"[GitHub YAML]({GITHUB_RAW}/{tag}/{bundle}/skill-checks.yml) · "
        f"[GitLab YAML]({GITHUB_RAW}/{tag}/{bundle}/gitlab-skill-checks.yml) · "
        f"[CI bundle]({GITHUB}/releases/download/{tag}/skillevaluator-ci-{version}.zip)\n\n"
        f"CI image: `{manifest['image']}`\n\n{changes}\n"
    )


def empty_upload(asset: dict) -> bool:
    return asset["state"] == "starter" and asset["size"] == 0


def verify_staged(release: dict, version: str, assets: Path, *, complete: bool) -> set[str]:
    staged = release_assets(release, version)
    if complete and set(staged) != expected_filenames(version):
        raise ValueError("GitHub release is missing staged assets.")
    uploaded = set()
    for name, asset in staged.items():
        if not complete and empty_upload(asset):
            continue
        if asset["state"] != "uploaded":
            raise ValueError(f"GitHub asset {name} is in state {asset['state']!r}. Inspect the draft before retrying.")
        result = subprocess.run(
            ["gh", "api", "-H", "Accept: application/octet-stream", f"{API_ROOT}/releases/assets/{asset['id']}"],
            stdout=subprocess.PIPE,
            check=True,
        )
        if hashlib.sha256(result.stdout).hexdigest() != digest(assets / name):
            raise ValueError(f"GitHub asset {name} differs from the verified release bytes.")
        uploaded.add(name)
    return uploaded


def stage(version: str, source_sha: str, assets: Path, run: str) -> None:
    tag = f"v{version}"
    target = check_tag(tag, source_sha)
    packages = verify_local(version, source_sha, assets)
    release = get_release(tag)
    if release is not None and target is None:
        raise ValueError(f"Existing GitHub release is missing tag {tag}.")
    if release is None or release["draft"]:
        product_release.inspect_public_indexes(version, packages)
        notes = release_notes(version, assets)
    if release is None:
        if target is None:
            api("git/refs", "POST", {"ref": f"refs/tags/{tag}", "sha": source_sha})
        if check_tag(tag, source_sha) != source_sha:
            raise ValueError(f"Could not verify GitHub tag {tag}.")
        release = api(
            "releases",
            "POST",
            {
                "tag_name": tag,
                "name": f"SkillEvaluator {tag}",
                "draft": True,
                "body": f"{notes}\n<!-- skillevaluator-stage: {run} -->",
            },
        )
    present = verify_staged(release, version, assets, complete=not release["draft"])
    if release["draft"]:
        # A failed GitHub upload can leave a zero-byte starter. Uploaded
        # assets have already matched and are never replaced.
        for asset in release_assets(release, version).values():
            if empty_upload(asset):
                api(f"releases/assets/{asset['id']}", "DELETE")
        for name in sorted(expected_filenames(version) - present):
            gh("release", "upload", tag, str(assets / name), "--repo", REPOSITORY)
        draft = api(f"releases/{release['id']}")
        verify_staged(draft, version, assets, complete=True)
        if check_tag(tag, source_sha) != source_sha:
            raise ValueError(f"Could not verify GitHub tag {tag}.")
        release = api(f"releases/{release['id']}", "PATCH", {"draft": False, "body": notes})
    if release["draft"] or not release.get("immutable"):
        raise ValueError("The GitHub release is not immutable. Verify repository release settings.")


def verify_public(version: str, source_sha: str, assets: Path) -> str:
    tag = f"v{version}"
    release = get_release(tag)
    if check_tag(tag, source_sha) != source_sha or not release or release["draft"] or not release.get("immutable"):
        raise ValueError("Expected an immutable, published GitHub release at the reviewed source SHA.")
    staged = release_assets(release, version)
    if set(staged) != expected_filenames(version):
        raise ValueError("Missing published GitHub release assets.")
    for name, asset in staged.items():
        url = f"{GITHUB}/releases/download/{tag}/{name}"
        if asset["browser_download_url"] != url:
            raise ValueError(f"Unexpected public asset URL for {name}.")
        with urlopen(url, timeout=60) as response:
            received = hashlib.file_digest(response, "sha256").hexdigest()
        if received != digest(assets / name):
            raise ValueError(f"Public GitHub asset {name} differs from the verified release bytes.")
    return digest(assets / "artifact-manifest.json")


def finalize(version: str, source_sha: str, assets: Path) -> None:
    verify_public(version, source_sha, assets)
    release = get_release(f"v{version}")
    body = release["body"] or ""
    if VERIFIED in body:
        return
    if body.count(PENDING) != 1:
        raise ValueError("Expected a single pending publication status in the GitHub release notes.")
    api(f"releases/{release['id']}", "PATCH", {"body": body.replace(PENDING, VERIFIED, 1)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "restore", "stage", "verify-public", "finalize"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--phase")
    parser.add_argument("--main-sha")
    parser.add_argument("--run")
    args = parser.parse_args()
    commands = {
        "restore": restore,
        "verify-public": verify_public,
        "finalize": finalize,
    }
    try:
        if args.command == "preflight":
            preflight(args.version, args.source_sha, args.main_sha, args.phase)
        elif args.command == "stage":
            stage(args.version, args.source_sha, args.assets, args.run)
        else:
            result = commands[args.command](args.version, args.source_sha, args.assets)
            if result is not None:
                print(str(result).lower() if isinstance(result, bool) else result)
    except (OSError, ValueError, subprocess.CalledProcessError, release_image.ReleaseImageError) as exc:
        print(f"Release check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
