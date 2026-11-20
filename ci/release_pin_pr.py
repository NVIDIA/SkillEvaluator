# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001, T201

"""Open or resume the bundle PR and identify its reviewed squash merge."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path
from urllib.parse import quote, urlencode
from zipfile import BadZipFile

try:
    import github_api
    import product_release
    import stable_channel
    from github_release import api, gh
except ModuleNotFoundError:
    from ci.github_release import api, gh

    from ci import github_api, product_release, stable_channel

UPSTREAM = "https://github.com/NVIDIA/SkillEvaluator"
MAIN = "refs/remotes/origin/main"
APP_SLUG = re.compile(r"[a-z0-9][a-z0-9-]*")
GIT_CREDENTIALS = ("-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential")
MAX_WAIT_MINUTES = 120


def git(*args: str, env: dict | None = None, data: bytes | None = None) -> bytes:
    if env is None and data is None:
        return product_release.git(*args)
    result = subprocess.run(["git", *args], env=env, input=data, capture_output=True, check=False)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip() or f"git {args[0]} failed.")
    return result.stdout


def branch_for(version: str) -> str:
    return f"release/pin-v{version}"


def bundle_paths(version: str) -> set[str]:
    prefix = f"ci/bundles/{version}"
    names = {file.name for file in product_release.render_templates.TEMPLATES.glob("*.yml")}
    names.update(
        {
            "README.md",
            "release.json",
            product_release.VERIFICATION,
            product_release.SCANNERS,
            product_release.CHECKSUMS,
            product_release.names_for(version)[2],
        }
    )
    return (
        {f"{prefix}/{name}" for name in names}
        | {f"{product_release.PACKAGE}/{name}" for name in product_release.DESCRIPTORS}
        | set(product_release.STABLE_PATHS.values())
    )


def fetch_main() -> set[str]:
    remote = git("remote", "get-url", "origin").decode().strip()
    if remote not in (UPSTREAM, f"{UPSTREAM}.git"):
        raise ValueError("Check out the upstream SkillEvaluator repository before preparing a release.")
    git(*GIT_CREDENTIALS, "fetch", "--no-tags", "origin", f"+refs/heads/main:{MAIN}")
    return set(git("rev-list", "--first-parent", MAIN).decode().splitlines())


def verify_source(version: str, candidate: str, image: str) -> None:
    product_release.names_for(version)
    if not github_api.COMMIT_SHA.fullmatch(candidate) or not stable_channel.OFFICIAL_IMAGE.fullmatch(image):
        raise ValueError("A full candidate SHA and official Tier 1 GHCR digest are required.")


def require_reviewed_candidate(version: str, candidate: str, main: set[str]) -> None:
    if candidate not in main:
        raise ValueError("The image candidate must be on upstream main's first-parent history.")
    project = tomllib.loads(product_release.source_file(candidate, "pyproject.toml").decode())
    if project["project"]["version"] != version:
        raise ValueError("The reviewed image candidate does not match the release version.")


def verify_bundle_commit(version: str, candidate: str, image: str, source: str) -> None:
    paths = bundle_paths(version)
    entries = git("ls-tree", "-r", "-z", "--format=%(objectmode) %(path)", source, "--", *sorted(paths))
    if set(entries.decode().split("\0")) != {"", *(f"100644 {path}" for path in paths)}:
        raise ValueError("The release bundle commit must contain regular generated files at the expected paths.")

    descriptors = {
        name: product_release.source_file(source, f"{product_release.PACKAGE}/{name}")
        for name in product_release.DESCRIPTORS
    }
    pins, _ = product_release.verify_bundle(version, source, product_release.names_for(version)[2], descriptors)
    if pins != {"image_source_commit": candidate, "image": image}:
        raise ValueError("The release bundle does not pin the selected image candidate and digest.")


def checked_bundle(version: str, directory: Path) -> dict[str, bytes]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("The CI bundle must be a regular artifact directory.")
    files = {}
    for entry in directory.rglob("*"):
        path = entry.relative_to(directory).as_posix()
        if entry.is_symlink() or not (entry.is_file() or entry.is_dir()):
            raise ValueError(f"Unexpected non-regular CI bundle artifact entry {path}.")
        if entry.is_file():
            files[path] = entry
    if set(files) != bundle_paths(version):
        raise ValueError("The CI bundle artifact does not contain exactly the expected generated files.")
    return {path: file.read_bytes() for path, file in files.items()}


def app_identity() -> tuple[str, str]:
    slug = os.environ.get("RELEASE_BUNDLE_APP_SLUG", "")
    if not APP_SLUG.fullmatch(slug):
        raise ValueError("The release bundle writer App must provide its GitHub slug.")
    login = f"{slug}[bot]"
    user = json.loads(gh("api", f"users/{quote(login, safe='')}"))
    user_id = user.get("id")
    if user.get("login") != login or type(user_id) is not int or user_id < 1:
        raise ValueError("GitHub did not return the configured release bundle App identity.")
    return login, f"{user_id}+{login}@users.noreply.github.com"


def bundle_commit(version: str, candidate: str, files: dict[str, bytes]) -> str:
    name, email = app_identity()
    with tempfile.TemporaryDirectory(prefix="release-pin-") as temp:
        env = {
            **os.environ,
            "GIT_INDEX_FILE": str(Path(temp) / "index"),
            "GIT_AUTHOR_NAME": name,
            "GIT_AUTHOR_EMAIL": email,
            "GIT_COMMITTER_NAME": name,
            "GIT_COMMITTER_EMAIL": email,
        }
        git("read-tree", candidate, env=env)
        for path, content in sorted(files.items()):
            blob = git("hash-object", "-w", "--stdin", data=content).decode().strip()
            git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", env=env)
        tree = git("write-tree", env=env).decode().strip()
        message = f"chore(ci): pin SkillEvaluator {version} release bundle\n\nSigned-off-by: {name} <{email}>"
        return git("commit-tree", tree, "-p", candidate, "-m", message, env=env).decode().strip()


def check_pull(pull: dict, branch: str, *, number: int | None = None) -> None:
    repo = product_release.REPOSITORY
    if (
        not isinstance(pull, dict)
        or type(pull.get("number")) is not int
        or pull["number"] < 1
        or (number is not None and pull["number"] != number)
        or pull["base"]["ref"] != "main"
        or pull["base"]["repo"]["full_name"] != repo
        or pull["head"]["ref"] != branch
        or (pull["head"]["repo"] or {}).get("full_name") != repo
        or not github_api.COMMIT_SHA.fullmatch(str(pull["head"]["sha"]))
    ):
        raise ValueError("The bundle PR does not use the expected upstream branches.")


def existing_pull(branch: str) -> dict | None:
    query = urlencode({"state": "all", "head": f"NVIDIA:{branch}", "base": "main"})
    matches = api(f"pulls?{query}")
    if not isinstance(matches, list) or len(matches) > 1:
        raise ValueError("Cannot identify a unique bundle PR for this release branch.")
    if not matches:
        return None
    pull = matches[0]
    check_pull(pull, branch)
    return pull


def verified_merge(version: str, candidate: str, image: str, pull: dict, main: set[str]) -> str:
    source = pull.get("merge_commit_sha")
    if pull.get("state") != "closed" or not pull.get("merged_at"):
        raise ValueError("The release bundle PR was closed without a merge.")
    if not github_api.COMMIT_SHA.fullmatch(str(source)) or source not in main:
        raise ValueError("The release bundle merge SHA is not on upstream main's first-parent history.")
    verify_bundle_commit(version, candidate, image, source)
    return source


def read_remote_branch(branch: str) -> str:
    ref = f"refs/heads/{branch}"
    remote = git(*GIT_CREDENTIALS, "ls-remote", "--refs", "origin", ref)
    if not remote:
        return ""
    git(*GIT_CREDENTIALS, "fetch", "--no-tags", "origin", f"+{ref}:refs/remotes/origin/{branch}")
    return git("rev-parse", f"refs/remotes/origin/{branch}").decode().strip()


def open_pull(version: str, candidate: str, image: str, bundle: Path) -> int:
    verify_source(version, candidate, image)
    main = fetch_main()
    require_reviewed_candidate(version, candidate, main)
    branch = branch_for(version)
    pull = existing_pull(branch)
    if pull is not None and pull.get("state") != "open":
        verified_merge(version, candidate, image, pull, main)
        return pull["number"]

    product_release.check_candidate_diff(candidate, MAIN, f"ci/bundles/{version}/")
    source = read_remote_branch(branch)
    if source:
        verify_bundle_commit(version, candidate, image, source)
        if pull is not None:
            if pull["head"]["sha"] != source:
                raise ValueError("The bundle PR head changed during inspection. Retry the release.")
            return pull["number"]
    else:
        if pull is not None:
            raise ValueError("The open bundle PR is missing its release branch.")
        source = bundle_commit(version, candidate, checked_bundle(version, bundle))
        verify_bundle_commit(version, candidate, image, source)
        git(*GIT_CREDENTIALS, "push", "origin", f"{source}:refs/heads/{branch}")

    payload = {
        "title": f"chore(ci): pin SkillEvaluator {version} release bundle",
        "head": branch,
        "base": "main",
        "body": (
            f"Pin the generated bundle and consumer configs for SkillEvaluator `{version}`.\n\n"
            f"Image source `{candidate}`\n\nVerified image `{image}`\n\n"
            "Review the generated files and qualify the GitHub and GitLab consumer configs before merging."
        ),
    }
    pull = api("pulls", "POST", payload)
    check_pull(pull, branch)
    if pull.get("state") != "open" or pull["head"]["sha"] != source:
        raise ValueError("GitHub did not open the expected release bundle PR.")
    return pull["number"]


def wait_for_merge(version: str, candidate: str, image: str, number: int, timeout_minutes: int) -> str:
    verify_source(version, candidate, image)
    if number < 1 or not 1 <= timeout_minutes <= MAX_WAIT_MINUTES:
        raise ValueError(f"Specify a pull request number and a timeout up to {MAX_WAIT_MINUTES} minutes.")
    branch = branch_for(version)
    deadline = time.monotonic() + timeout_minutes * 60
    print(f"Waiting for review and merge of {UPSTREAM}/pull/{number}", file=sys.stderr)
    while True:
        pull = api(f"pulls/{number}")
        check_pull(pull, branch, number=number)
        if pull.get("state") != "open":
            main = fetch_main()
            require_reviewed_candidate(version, candidate, main)
            return verified_merge(version, candidate, image, pull, main)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(f"Timed out waiting for pull request #{number}. Rerun the failed job after it merges.")
        time.sleep(min(30, remaining))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("open")
    prepare.add_argument("--bundle", type=Path, required=True)
    wait = commands.add_parser("wait")
    wait.add_argument("--pr", type=int, required=True)
    wait.add_argument("--timeout-minutes", type=int, default=MAX_WAIT_MINUTES)
    for command in (prepare, wait):
        command.add_argument("--version", required=True)
        command.add_argument("--source-sha", required=True)
        command.add_argument("--image", required=True)
    args = parser.parse_args()
    try:
        if args.command == "open":
            print(open_pull(args.version, args.source_sha, args.image, args.bundle))
        else:
            print(wait_for_merge(args.version, args.source_sha, args.image, args.pr, args.timeout_minutes))
    except (BadZipFile, KeyError, OSError, TypeError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"CI bundle PR failed: {error}\n")


if __name__ == "__main__":
    main()
