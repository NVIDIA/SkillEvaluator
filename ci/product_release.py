# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001, T201 -- standalone release CLI.

"""Assemble and verify SkillEvaluator release assets and public package indexes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

try:
    import github_api
    import render_templates
    import stable_channel
except ModuleNotFoundError:
    from ci import github_api, render_templates, stable_channel

REPOSITORY = "NVIDIA/SkillEvaluator"
DOCUMENTATION = "https://docs.nvidia.com/skills/skillevaluator/"
PYPI = "https://pypi.org/pypi/skillevaluator/{version}/json"
DEVZONE_INDEX_VARIABLE = "RELEASE_DEVZONE_INDEX_URL"
MANIFEST = "artifact-manifest.json"
CHECKSUMS = "SHA256SUMS.txt"
DESCRIPTORS = ("ci-release.json", "ci-release-all-tiers.json")
PACKAGE = "src/skillevaluator"
VERIFICATION = "verification-receipt.json"
SCANNERS = "scanner-inventory.json"
SHA256 = re.compile(r"[0-9a-f]{64}")
RUN_URL = re.compile(r"https://github\.com/NVIDIA/SkillEvaluator/actions/runs/[1-9][0-9]*")
MAX_METADATA_BYTES = 2 * 1024 * 1024
REQUEST_TIMEOUT = 15
SMOKE_CASES = {
    "all-pass": (0, "passed"),
    "invalid-name": (1, "failed"),
    "changed-one": (0, "passed"),
    "no-change": (0, "skipped"),
    "missing-llm-key": (1, "error"),
}
SCANNER_NAMES = ("bandit", "gitleaks", "pip-audit", "semgrep", "skillevaluator", "skillspector")
STABLE_PATHS = {
    **{name: f".github/workflows/{name}" for name in stable_channel.WORKFLOWS},
    "stable-tier1.yml": "ci/gitlab/stable-tier1.yml",
}


class IndexPending(ValueError):
    """An index response may change as the release propagates."""


class IndexMissing(IndexPending):
    """The requested URL is absent from the index (HTTP 404)."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_object(data: bytes, label: str) -> dict:
    value = github_api.strict_json(data, message=f"{label} is not valid JSON.")
    require(isinstance(value, dict), f"{label} must contain a JSON object.")
    return value


def names_for(version: str) -> tuple[str, str, str]:
    require(bool(stable_channel.VERSION.fullmatch(version)), "Expected a numeric major.minor.patch version.")
    return (
        f"skillevaluator-{version}-py3-none-any.whl",
        f"skillevaluator-{version}.tar.gz",
        f"skillevaluator-ci-{version}.zip",
    )


def git(*args: str) -> bytes:
    result = subprocess.run(["git", *args], capture_output=True, check=False)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip() or f"git {args[0]} failed.")
    return result.stdout


def source_file(commit: str, path: str) -> bytes:
    return git("cat-file", "blob", f"{commit}:{path}")


def check_ancestry(ancestor: str, source: str) -> None:
    try:
        git("merge-base", "--is-ancestor", ancestor, source)
    except ValueError as exc:
        raise ValueError("The image source must be an ancestor of the final release checkout.") from exc


def check_candidate_diff(candidate: str, source: str, bundle: str) -> None:
    check_ancestry(candidate, source)
    allowed = {
        "README.md",
        "CHANGELOG.md",
        "ci/README.md",
        "ci/bundles/README.md",
        "ci/templates/README.md",
        "ci/templates/all-tiers/README.md",
        *(f"{PACKAGE}/{name}" for name in DESCRIPTORS),
        *STABLE_PATHS.values(),
    }
    changed = git("diff", "--no-renames", "--name-only", "-z", candidate, source).decode().split("\0")
    unexpected = [path for path in changed if path and path not in allowed and not path.startswith(("docs/", bundle))]
    require(not unexpected, f"The final release changed image-candidate inputs: {', '.join(unexpected)}.")


def verify_checksum_file(checksums: bytes, contents: dict[str, str], label: str) -> None:
    recorded = {}
    for line in checksums.decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9_.-]*)", line)
        require(bool(match), f"{label} contains an invalid checksum line.")
        digest, name = match.groups()
        require(name not in recorded, f"{label} contains a duplicate checksum for {name}.")
        recorded[name] = digest
    require(recorded == contents, f"{label} does not match the exact contents.")


def read_zip_member(archive: ZipFile, name: str, limit: int) -> bytes:
    require(archive.namelist().count(name) == 1, f"Archive must contain exactly one {name}.")
    entry = archive.getinfo(name)
    require(entry.file_size <= limit, f"Archive entry {name} exceeds its size limit.")
    return archive.read(entry)


def verify_bundle(
    version: str, source: str, zip_name: str, descriptors: dict[str, bytes]
) -> tuple[dict[str, str], bytes]:
    prefix = f"ci/bundles/{version}/"
    content = {}
    for path in git("ls-tree", "-r", "--name-only", "-z", source, "--", prefix).decode().split("\0"):
        if path:
            content[path.removeprefix(prefix)] = source_file(source, path)
    require(zip_name in content, f"{zip_name} is missing from the final release commit.")
    zip_data = content.pop(zip_name)
    require(
        "release.json" in content and VERIFICATION in content and SCANNERS in content, "Missing bundle verification."
    )
    pins = json_object(content["release.json"], "Bundle release pins")
    receipt = json_object(content[VERIFICATION], "Image verification receipt")
    image = pins.get("image")
    candidate = receipt.get("source_revision")
    require(
        pins.get("repository") == REPOSITORY
        and isinstance(image, str)
        and bool(stable_channel.OFFICIAL_IMAGE.fullmatch(image))
        and isinstance(candidate, str)
        and bool(github_api.COMMIT_SHA.fullmatch(candidate))
        and pins.get("workflow_ref") == pins.get("publisher_ref") == candidate,
        "The bundle must pin the official image and reviewed candidate workflows.",
    )
    check_candidate_diff(candidate, source, prefix)
    require(
        receipt.get("all_verified") is True
        and receipt.get("anonymous_pull_verified") is True
        and receipt.get("public_manifest_matches_smoke_tested_image") is True
        and receipt.get("image") == image
        and receipt.get("public_manifest_digest") == image.split("@")[1]
        and receipt.get("platform") == {"os": "linux", "architecture": "amd64"}
        and bool(RUN_URL.fullmatch(str(receipt.get("official_build_run"))))
        and bool(SHA256.fullmatch(str(receipt.get("runtime_observation_sha256")))),
        "The image verification receipt does not prove the official public digest.",
    )
    cases = receipt.get("image_smoke_cases")
    require(
        isinstance(cases, list)
        and all(isinstance(case, dict) and case.get("verified") is True for case in cases)
        and len(cases) == len(SMOKE_CASES)
        and {str(case.get("case")): (case.get("exit_code"), case.get("overall_status")) for case in cases}
        == SMOKE_CASES,
        "The image verification receipt must contain all five expected smoke outcomes.",
    )
    scanner_data = json_object(content[SCANNERS], "Scanner inventory")
    scanner_verification = scanner_data.get("verification")
    require(
        isinstance(scanner_verification, dict)
        and all(
            scanner_verification.get(key) == value
            for key, value in {
                "image": image,
                "source_revision": candidate,
                "build_run": receipt["official_build_run"],
                "runtime_observation_sha256": receipt["runtime_observation_sha256"],
            }.items()
        )
        and all(
            isinstance(scanner_data.get(name), dict)
            and isinstance(scanner_data[name].get("command"), str)
            and bool(scanner_data[name]["command"])
            and isinstance(scanner_data[name].get("version"), str)
            and bool(scanner_data[name]["version"])
            for name in SCANNER_NAMES
        )
        and scanner_data["skillevaluator"]["version"] == version,
        "Scanner inventory must report the observed versions for the verified release image.",
    )
    with tempfile.TemporaryDirectory() as scratch:
        generated = Path(scratch) / version
        render_templates.render_templates(generated, candidate, image)
        expected = {path.name: path.read_bytes() for path in generated.iterdir()}
        require(
            set(content) == set(expected) | {CHECKSUMS, SCANNERS, VERIFICATION}
            and all(content[name] == data for name, data in expected.items()),
            "The checked-in bundle differs from the rendered immutable onboarding templates.",
        )
        (generated / VERIFICATION).write_bytes(content[VERIFICATION])
        workflows = stable_channel.render(generated)
    for name, text in workflows.items():
        path = STABLE_PATHS[name]
        require(source_file(source, path) == text.encode(), f"{path} differs from the verified release bundle.")

    verify_checksum_file(
        content[CHECKSUMS],
        {name: sha256(data) for name, data in content.items() if name != CHECKSUMS},
        "Bundle checksums",
    )
    with ZipFile(io.BytesIO(zip_data)) as archive:
        require(
            set(archive.namelist()) == set(content), "The CI archive must contain the exact committed bundle files."
        )
        for name, expected_data in content.items():
            require(
                read_zip_member(archive, name, len(expected_data)) == expected_data,
                f"The CI archive contains a different {name}.",
            )

    common = {key: pins[key] for key in ("repository", "workflow_ref", "publisher_ref", "image")}
    tier1 = json_object(descriptors[DESCRIPTORS[0]], DESCRIPTORS[0])
    require(tier1 == {**common, "schema_version": 1}, "Tier 1 descriptor must match the verified bundle pins.")
    all_tiers = json_object(descriptors[DESCRIPTORS[1]], DESCRIPTORS[1])
    candidate_descriptor = json_object(
        source_file(candidate, f"{PACKAGE}/{DESCRIPTORS[1]}"), "Candidate all-tier descriptor"
    )
    expected_all_tiers = {
        **candidate_descriptor,
        **common,
        "schema_version": 2,
        "feedback_ref": candidate,
        "gitlab_ref": candidate,
        "tools_ref": candidate,
        "evaluator_version": version,
        "runtime_lock_sha256": sha256(source_file(candidate, "uv.lock")),
        "report_contract": "skillevaluator-ci-v1",
    }
    capabilities = all_tiers.get("capabilities")
    require(
        all(all_tiers.get(key) == value for key, value in expected_all_tiers.items())
        and isinstance(capabilities, list)
        and all(isinstance(name, str) and name for name in capabilities)
        and {"all-tiers-github", "all-tiers-gitlab", "ci-model-routing-v1", "ci-benchmark-pr-v1"}.issubset(
            capabilities
        ),
        "All-tier descriptor must bind its tools, workflow, feedback, capabilities, version and lock to the release.",
    )
    return {"image_source_commit": candidate, "image": image}, zip_data


def verify_metadata(payload: bytes, version: str, artifact: str) -> None:
    metadata = BytesParser().parsebytes(payload, headersonly=True)
    require(metadata.get_all("Name") == ["skillevaluator"], f"{artifact} has an incorrect distribution name.")
    require(metadata.get_all("Version") == [version], f"{artifact} has an incorrect distribution version.")
    python_versions = metadata.get_all("Requires-Python", [])
    require(
        len(python_versions) == 1 and sorted(python_versions[0].replace(" ", "").split(",")) == ["<3.15", ">=3.12"],
        f"{artifact} has incorrect Requires-Python metadata.",
    )
    project_urls = [re.split(r",\s*", entry, maxsplit=1) for entry in metadata.get_all("Project-URL", [])]
    require(
        [entry for entry in project_urls if entry[0].casefold() == "documentation"]
        == [["Documentation", DOCUMENTATION]],
        f"{artifact} has an incorrect Documentation URL.",
    )


def read_tar_member(archive: tarfile.TarFile, name: str, limit: int) -> bytes:
    matches = [entry for entry in archive.getmembers() if entry.name == name]
    require(len(matches) == 1 and matches[0].isfile(), f"Source archive must contain exactly one regular {name}.")
    require(matches[0].size <= limit, f"Source archive entry {name} exceeds its size limit.")
    with archive.extractfile(matches[0]) as handle:
        return handle.read()


def verify_distributions(directory: Path, version: str, descriptors: dict[str, bytes]) -> None:
    wheel, sdist, _ = names_for(version)
    with ZipFile(directory / wheel) as archive:
        verify_metadata(
            read_zip_member(archive, f"skillevaluator-{version}.dist-info/METADATA", MAX_METADATA_BYTES), version, wheel
        )
        for name, expected in descriptors.items():
            require(
                read_zip_member(archive, f"skillevaluator/{name}", len(expected)) == expected,
                f"Wheel {name} differs from the final source.",
            )
    with tarfile.open(directory / sdist, "r:gz") as archive:
        prefix = f"skillevaluator-{version}"
        verify_metadata(read_tar_member(archive, f"{prefix}/PKG-INFO", MAX_METADATA_BYTES), version, sdist)
        for name, expected in descriptors.items():
            require(
                read_tar_member(archive, f"{prefix}/{PACKAGE}/{name}", len(expected)) == expected,
                f"Source distribution {name} differs from the final source.",
            )


def asset_data(path: Path) -> dict:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"filename": path.name, "size": path.stat().st_size, "sha256": digest}


def verify_directory(directory: Path, names: set[str]) -> None:
    entries = list(directory.iterdir())
    require(
        {entry.name for entry in entries} == names
        and all(entry.is_file() and not entry.is_symlink() for entry in entries),
        f"{directory} must contain exactly these regular files: {', '.join(sorted(names))}.",
    )


def source_data(version: str, source: str) -> tuple[dict, dict[str, bytes], bytes]:
    _, _, zip_name = names_for(version)
    require(bool(github_api.COMMIT_SHA.fullmatch(source)), "The final source must be a full commit SHA.")
    require(git("rev-parse", "HEAD").decode().strip() == source, "HEAD must match the final release source SHA.")
    project = tomllib.loads(source_file(source, "pyproject.toml").decode())
    require(
        project.get("project", {}).get("version") == version, "The committed package version differs from the release."
    )
    descriptors = {name: source_file(source, f"{PACKAGE}/{name}") for name in DESCRIPTORS}
    image, bundle = verify_bundle(version, source, zip_name, descriptors)
    identities = {
        "version": version,
        "source_commit": source,
        "source_tree": git("rev-parse", f"{source}^{{tree}}").decode().strip(),
        **image,
    }
    return identities, descriptors, bundle


def assemble(version: str, source: str, dist: Path, output: Path) -> None:
    wheel, sdist, zip_name = names_for(version)
    verify_directory(dist, {wheel, sdist})
    identities, descriptors, bundle = source_data(version, source)
    verify_distributions(dist, version, descriptors)
    output.mkdir()
    for name in (wheel, sdist):
        shutil.copyfile(dist / name, output / name)
    (output / zip_name).write_bytes(bundle)
    manifest = {
        **identities,
        "artifacts": [asset_data(output / wheel), asset_data(output / sdist)],
        "ci_bundle": asset_data(output / zip_name),
    }
    (output / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (output / CHECKSUMS).open("x", encoding="utf-8") as sums:
        for name in (MANIFEST, wheel, sdist, zip_name):
            sums.write(f"{asset_data(output / name)['sha256']}  {name}\n")


def checked_manifest(version: str, assets: Path) -> tuple[dict, str]:
    wheel, sdist, zip_name = names_for(version)
    verify_directory(assets, {wheel, sdist, zip_name, MANIFEST, CHECKSUMS})
    manifest_bytes = (assets / MANIFEST).read_bytes()
    manifest = json_object(manifest_bytes, "Artifact manifest")
    files = {name: asset_data(assets / name) for name in (wheel, sdist, zip_name)}
    require(
        manifest.get("version") == version
        and manifest.get("artifacts") == [files[wheel], files[sdist]]
        and manifest.get("ci_bundle") == files[zip_name],
        "Artifact manifest does not match the exact release assets.",
    )
    manifest_digest = sha256(manifest_bytes)
    verify_checksum_file(
        (assets / CHECKSUMS).read_bytes(),
        {MANIFEST: manifest_digest, **{name: data["sha256"] for name, data in files.items()}},
        "Product checksums",
    )
    return manifest, manifest_digest


def verify_assets(version: str, source: str, assets: Path) -> str:
    manifest, digest = checked_manifest(version, assets)
    identities, descriptors, bundle = source_data(version, source)
    require(
        all(manifest.get(key) == value for key, value in identities.items()),
        "Artifact manifest does not match the final release source and image identities.",
    )
    require((assets / names_for(version)[2]).read_bytes() == bundle, "The staged bundle differs from the final source.")
    verify_distributions(assets, version, descriptors)
    return digest


class DevZoneLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            attributes = dict(attrs)
            if attributes.get("href"):
                self.links.append((attributes["href"], "data-yanked" in attributes))


def get_url(url: str, limit: int, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "SkillEvaluator-Release", "Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            require(response.geturl() == url, f"Unexpected redirect while reading {url}.")
            return github_api.bounded_read(
                response, limit, message=f"The response for {url} exceeds its expected size."
            )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise IndexMissing(f"{url} is not visible yet (HTTP 404).") from exc
        if exc.code in (408, 429) or 500 <= exc.code < 600:
            raise IndexPending(f"{url} is not available yet (HTTP {exc.code}).") from exc
        raise ValueError(f"Unable to verify {url} (HTTP {exc.code}).") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise IndexPending(f"{url} is temporarily unavailable.") from exc


def file_url(url: str, host: str, filename: str) -> str:
    address = urllib.parse.urlsplit(url)
    require(
        address.scheme == "https"
        and address.netloc == host
        and not address.query
        and PurePosixPath(urllib.parse.unquote(address.path)).name == filename,
        f"Unexpected public download URL for {filename}.",
    )
    return urllib.parse.urlunsplit(address._replace(fragment=""))


def pypi_entries(version: str, timeout: float) -> list[tuple]:
    data = json_object(get_url(PYPI.format(version=version), MAX_METADATA_BYTES, timeout), "PyPI version")
    require(
        isinstance(data.get("info"), dict)
        and data["info"].get("version") == version
        and isinstance(data.get("urls"), list),
        "PyPI returned an unexpected project version.",
    )
    entries = []
    for release_file in data["urls"]:
        require(
            isinstance(release_file, dict)
            and isinstance(release_file.get("filename"), str)
            and isinstance(release_file.get("url"), str)
            and isinstance(release_file.get("digests"), dict)
            and isinstance(release_file.get("yanked"), bool)
            and type(release_file.get("size")) is int,
            "PyPI returned invalid file metadata.",
        )
        name = release_file["filename"]
        url = file_url(release_file["url"], "files.pythonhosted.org", name)
        entries.append((name, url, release_file["digests"].get("sha256"), release_file["yanked"], release_file["size"]))
    return entries


def distribution_version(name: str) -> str | None:
    if not name.lower().startswith("skillevaluator-"):
        return None
    rest = name.split("-", 1)[1]
    if rest.endswith(".whl"):
        return rest.split("-", 1)[0]
    for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".tar", ".tgz", ".zip"):
        if rest.endswith(suffix):
            return rest.removesuffix(suffix)
    return None


def devzone_entries(version: str, timeout: float) -> list[tuple]:
    index_url = os.environ.get(DEVZONE_INDEX_VARIABLE, "")
    address = urllib.parse.urlsplit(index_url)
    require(
        bool(re.fullmatch(r"[A-Za-z0-9.-]+(?::[0-9]+)?", address.netloc))
        and index_url == f"https://{address.netloc}/skillevaluator/",
        f"Configure {DEVZONE_INDEX_VARIABLE} with the HTTPS /skillevaluator/ project listing.",
    )
    origin = address.netloc
    page = DevZoneLinks()
    page.feed(get_url(index_url, MAX_METADATA_BYTES, timeout).decode("utf-8"))
    entries = []
    for href, yanked in page.links:
        absolute = urllib.parse.urljoin(index_url, href)
        address = urllib.parse.urlsplit(absolute)
        name = PurePosixPath(urllib.parse.unquote(address.path)).name
        if distribution_version(name) == version:
            entries.append(
                (
                    name,
                    file_url(absolute, origin, name),
                    address.fragment.removeprefix("sha256=") if address.fragment.startswith("sha256=") else "",
                    yanked,
                    None,
                )
            )
    return entries


def verify_index(
    entries: list[tuple], expected: dict, label: str, timeout: float, *, allow_missing: bool = False
) -> set[str]:
    """Validate all listed files against the expected distribution bytes."""
    seen = set()
    for name, _url, digest, yanked, size in entries:
        require(name in expected and name not in seen, f"{label} lists an unexpected or duplicate artifact: {name}.")
        seen.add(name)
        require(not yanked, f"{label} lists a yanked artifact: {name}.")
        require(digest == expected[name]["sha256"], f"{label} lists the wrong hash for {name}.")
        require(size is None or size == expected[name]["size"], f"{label} lists the wrong size for {name}.")
    if not allow_missing and seen != expected.keys():
        raise IndexPending(f"{label} artifacts are not visible yet: {', '.join(sorted(expected.keys() - seen))}.")
    for name, url, _, _, _ in entries:
        target = expected[name]
        data = get_url(url, target["size"] + 1, timeout)
        require(
            len(data) == target["size"] and sha256(data) == target["sha256"],
            f"{label} download has different bytes for {name}.",
        )
    return seen


def inspect_public_indexes(version: str, expected: dict) -> set[str]:
    """Identify matching existing artifacts before tagging or submitting a release."""
    try:
        pypi = pypi_entries(version, REQUEST_TIMEOUT)
    except IndexMissing:
        # PyPI's version-specific endpoint may legitimately be absent. The
        # DevZone project listing and all listed downloads must remain available.
        pypi = []
    devzone = devzone_entries(version, REQUEST_TIMEOUT)
    py_files = verify_index(pypi, expected, "PyPI", REQUEST_TIMEOUT, allow_missing=True)
    nv_files = verify_index(devzone, expected, "DevZone", REQUEST_TIMEOUT, allow_missing=True)
    require(
        py_files == nv_files, "An artifact is present on only one index. Reconcile it with Kitmaker before retrying."
    )
    return py_files


def verify_indexes(version: str, assets: Path, wait_seconds: int = 0) -> None:
    require(wait_seconds >= 0, "The index wait must be non-negative.")
    manifest, _ = checked_manifest(version, assets)
    expected = {entry["filename"]: entry for entry in manifest["artifacts"]}
    deadline = time.monotonic() + wait_seconds
    remaining = {"PyPI": pypi_entries, "DevZone": devzone_entries}
    while remaining:
        pending = []
        for label, fetch in list(remaining.items()):
            timeout = min(REQUEST_TIMEOUT, max(1.0, deadline - time.monotonic())) if wait_seconds else REQUEST_TIMEOUT
            try:
                verify_index(fetch(version, timeout), expected, label, timeout)
            except IndexPending as exc:
                pending.append(str(exc))
            else:
                del remaining[label]
        if remaining:
            delay = deadline - time.monotonic()
            if delay <= 0:
                raise IndexPending(" ".join(pending))
            time.sleep(min(10, delay))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("assemble", help="Build the exact five staged release assets.")
    build.add_argument("--version", required=True)
    build.add_argument("--source-sha", required=True)
    build.add_argument("--dist", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify-assets", help="Verify staged release assets and print the manifest digest.")
    verify.add_argument("--version", required=True)
    verify.add_argument("--source-sha", required=True)
    verify.add_argument("--assets", type=Path, required=True)

    inspect = commands.add_parser("inspect-indexes", help="Check for conflicting public artifacts before publication.")
    inspect.add_argument("--version", required=True)
    inspect.add_argument("--assets", type=Path, required=True)

    indexes = commands.add_parser("verify-indexes", help="Verify package artifacts served by PyPI and NVIDIA DevZone.")
    indexes.add_argument("--version", required=True)
    indexes.add_argument("--assets", type=Path, required=True)
    indexes.add_argument("--wait-seconds", type=int, default=0)

    args = parser.parse_args()
    try:
        if args.command == "assemble":
            assemble(args.version, args.source_sha, args.dist, args.output)
        elif args.command == "verify-assets":
            print(verify_assets(args.version, args.source_sha, args.assets))
        elif args.command == "inspect-indexes":
            manifest, _ = checked_manifest(args.version, args.assets)
            expected = {asset["filename"]: asset for asset in manifest["artifacts"]}
            existing = inspect_public_indexes(args.version, expected)
            print(f"No public package conflicts for {args.version}. Matching existing files: {len(existing)}.")
        else:
            verify_indexes(args.version, args.assets, args.wait_seconds)
            print(f"Verified SkillEvaluator {args.version} wheel and source distribution on PyPI and DevZone.")
    except (OSError, ValueError, BadZipFile, tarfile.TarError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
