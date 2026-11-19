# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from tarfile import TarFile, TarInfo
from types import SimpleNamespace
from urllib.error import HTTPError
from zipfile import ZipFile

import pytest

from ci import render_templates, stable_channel

VERSION = "1.0.2"
WHEEL = f"skillevaluator-{VERSION}-py3-none-any.whl"
SDIST = f"skillevaluator-{VERSION}.tar.gz"
BUNDLE = f"skillevaluator-ci-{VERSION}.zip"
IMAGE = "ghcr.io/nvidia/skillevaluator-tier1@sha256:" + "a" * 64
RUN = "https://github.com/NVIDIA/SkillEvaluator/actions/runs/123"
DESCRIPTORS = ("ci-release.json", "ci-release-all-tiers.json")
CAPABILITIES = ["all-tiers-github", "all-tiers-gitlab", "ci-model-routing-v1", "ci-benchmark-pr-v1"]
SCRIPT = Path(__file__).resolve().parents[2] / "ci/product_release.py"
DEVZONE_URL = "https://packages.example.com/skillevaluator/"


@pytest.fixture(autouse=True)
def configure_devzone(monkeypatch):
    monkeypatch.setenv("RELEASE_DEVZONE_INDEX_URL", DEVZONE_URL)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def commit(repo):
    git(repo, "add", ".")
    git(repo, "-c", "user.name=Release tests", "-c", "user.email=test@example.com", "commit", "-qm", "Fixture")
    return git(repo, "rev-parse", "HEAD")


def write_json(path, contents):
    path.write_text(json.dumps(contents) + "\n")


def checksums(files):
    return "".join(f"{digest(path.read_bytes())}  {path.name}\n" for path in sorted(files))


def build_dist(dist, package, *, documentation="https://docs.nvidia.com/skills/skillevaluator/"):
    metadata = (
        f"Metadata-Version: 2.4\nName: skillevaluator\nVersion: {VERSION}\n"
        f"Requires-Python: <3.15,>=3.12\nProject-URL: Documentation, {documentation}\n"
    ).encode()
    with ZipFile(dist / WHEEL, "w") as wheel:
        wheel.writestr(f"skillevaluator-{VERSION}.dist-info/METADATA", metadata)
        for name in DESCRIPTORS:
            wheel.writestr(f"skillevaluator/{name}", (package / name).read_bytes())
    with TarFile.open(dist / SDIST, "w:gz") as sdist:
        files = {"PKG-INFO": metadata}
        files.update({f"src/skillevaluator/{name}": (package / name).read_bytes() for name in DESCRIPTORS})
        for name, data in files.items():
            member = TarInfo(f"skillevaluator-{VERSION}/{name}")
            member.size = len(data)
            sdist.addfile(member, io.BytesIO(data))


@pytest.fixture
def release(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    package = repo / "src/skillevaluator"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(f"__version__ = {VERSION!r}\n")
    (repo / "pyproject.toml").write_text(f'[project]\nversion = "{VERSION}"\n')
    lock = b"version = 1\n"
    (repo / "uv.lock").write_bytes(lock)
    write_json(package / DESCRIPTORS[1], {"capabilities": CAPABILITIES, "report_contract": "skillevaluator-ci-v1"})
    candidate = commit(repo)
    bundle = repo / "ci/bundles" / VERSION
    pins = render_templates.render_templates(bundle, candidate, IMAGE)
    observation = "b" * 64
    write_json(
        bundle / "verification-receipt.json",
        {
            "all_verified": True,
            "image": IMAGE,
            "source_revision": candidate,
            "platform": {"os": "linux", "architecture": "amd64"},
            "public_manifest_matches_smoke_tested_image": True,
            "anonymous_pull_verified": True,
            "official_build_run": RUN,
            "runtime_observation_sha256": observation,
            "public_manifest_digest": IMAGE.split("@")[1],
            "image_smoke_cases": [
                {"case": case, "verified": True, "exit_code": code, "overall_status": status}
                for case, code, status in (
                    ("all-pass", 0, "passed"),
                    ("invalid-name", 1, "failed"),
                    ("changed-one", 0, "passed"),
                    ("no-change", 0, "skipped"),
                    ("missing-llm-key", 1, "error"),
                )
            ],
        },
    )
    inventory = {
        tool: {"version": "2.0.0", "command": f"{tool} --version"}
        for tool in ("bandit", "pip-audit", "skillspector", "semgrep", "gitleaks")
    }
    inventory["skillevaluator"] = {"version": VERSION, "command": "skillevaluator --version"}
    inventory["verification"] = {
        "image": IMAGE,
        "source_revision": candidate,
        "build_run": RUN,
        "runtime_observation_sha256": observation,
    }
    write_json(bundle / "scanner-inventory.json", inventory)
    (bundle / "SHA256SUMS.txt").write_text(checksums(bundle.iterdir()))
    files = sorted(bundle.iterdir())
    with ZipFile(bundle / BUNDLE, "w") as archive:
        for path in files:
            archive.write(path, arcname=path.name)
    pins.pop("caller_blob")
    write_json(package / DESCRIPTORS[0], {**pins, "schema_version": 1})
    write_json(
        package / DESCRIPTORS[1],
        {
            **pins,
            "schema_version": 2,
            "capabilities": CAPABILITIES,
            "feedback_ref": candidate,
            "gitlab_ref": candidate,
            "tools_ref": candidate,
            "report_contract": "skillevaluator-ci-v1",
            "evaluator_version": VERSION,
            "runtime_lock_sha256": digest(lock),
        },
    )
    for name, text in stable_channel.render(bundle).items():
        output_directory = "ci/gitlab" if name == "stable-tier1.yml" else ".github/workflows"
        output = repo / output_directory / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    source = commit(repo)
    dist = tmp_path / "dist"
    dist.mkdir()
    build_dist(dist, package)
    return repo, source, dist, tmp_path / "assets", candidate


def release_command(repo, source, action, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), action, "--version", VERSION, "--source-sha", source, *map(str, args)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def test_assembles_manifest_and_verifies_exact_assets(release):
    repo, source, dist, assets, candidate = release
    result = release_command(repo, source, "assemble", "--dist", dist, "--output", assets)
    assert result.returncode == 0, result.stderr
    assert {p.name for p in assets.iterdir()} == {WHEEL, SDIST, BUNDLE, "artifact-manifest.json", "SHA256SUMS.txt"}
    manifest_file = assets / "artifact-manifest.json"
    manifest = json.loads(manifest_file.read_bytes())
    assert manifest["source_commit"] == source
    assert manifest["source_tree"] == git(repo, "rev-parse", "HEAD^{tree}")
    assert manifest["image_source_commit"] == candidate
    assert manifest["image"] == IMAGE
    assert manifest["artifacts"] == [
        {"filename": name, "size": (dist / name).stat().st_size, "sha256": digest((dist / name).read_bytes())}
        for name in (WHEEL, SDIST)
    ]
    assert (assets / BUNDLE).read_bytes() == (repo / "ci/bundles" / VERSION / BUNDLE).read_bytes()
    verified = release_command(repo, source, "verify-assets", "--assets", assets)
    assert verified.returncode == 0, verified.stderr
    assert verified.stdout.strip() == digest(manifest_file.read_bytes())
    (assets / WHEEL).write_bytes((assets / WHEEL).read_bytes() + b"tampered")
    verified = release_command(repo, source, "verify-assets", "--assets", assets)
    assert verified.returncode == 1
    assert "manifest" in verified.stderr


def test_rejects_installed_package_from_before_bundle_pin(release):
    repo, source, dist, assets, _candidate = release
    package = repo / "src/skillevaluator"
    (package / DESCRIPTORS[0]).write_text("{}\n")
    build_dist(dist, package)
    result = release_command(repo, source, "assemble", "--dist", dist, "--output", assets)
    assert result.returncode == 1
    assert "ci-release.json" in result.stderr
    assert not assets.exists()


def test_rejects_stale_distribution_documentation_link(release):
    repo, source, dist, assets, _candidate = release
    build_dist(dist, repo / "src/skillevaluator", documentation="https://github.com/NVIDIA/SkillEvaluator#readme")
    result = release_command(repo, source, "assemble", "--dist", dist, "--output", assets)
    assert result.returncode == 1
    assert "Documentation" in result.stderr


def test_rejects_candidate_to_final_package_drift(release):
    repo, _source, dist, assets, _candidate = release
    (repo / "src/skillevaluator/__init__.py").write_text("unaudited_change = True\n")
    new_source = commit(repo)
    result = release_command(repo, new_source, "assemble", "--dist", dist, "--output", assets)
    assert result.returncode == 1
    assert "src/skillevaluator/__init__.py" in result.stderr


def test_rejects_unreviewed_all_tier_capabilities(release):
    repo, _source, dist, assets, _candidate = release
    package = repo / "src/skillevaluator"
    descriptor = package / DESCRIPTORS[1]
    payload = json.loads(descriptor.read_bytes())
    payload["capabilities"].append("unreviewed-feature")
    write_json(descriptor, payload)
    build_dist(dist, package)
    source = commit(repo)
    result = release_command(repo, source, "assemble", "--dist", dist, "--output", assets)
    assert result.returncode == 1
    assert "capabilities" in result.stderr


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/skill-checks-stable.yml",
        pytest.param(
            "ci/gitlab/stable-tier1.yml",
            marks=pytest.mark.skipif(
                not hasattr(stable_channel, "GITLAB_INCLUDE"), reason="Pre-GitLab stable renderer"
            ),
        ),
    ],
)
def test_rejects_stable_configuration_that_differs_from_bundle(release, path):
    repo, _source, dist, assets, _candidate = release
    configuration = repo / path
    configuration.write_text(configuration.read_text() + "# tampered\n")
    source = commit(repo)
    result = release_command(repo, source, "assemble", "--dist", dist, "--output", assets)
    assert result.returncode == 1
    assert f"{path} differs from the verified release bundle" in result.stderr


def test_rejects_unchecked_out_source(release):
    repo, _source, dist, assets, candidate = release
    result = release_command(repo, candidate, "assemble", "--dist", dist, "--output", assets)
    assert result.returncode == 1
    assert "HEAD" in result.stderr


def test_rejects_archived_bundle_that_disagrees_with_commit(release):
    repo, _source, dist, assets, _candidate = release
    bundle = repo / "ci/bundles" / VERSION
    with ZipFile(bundle / BUNDLE, "w") as archive:
        for path in sorted(bundle.iterdir()):
            if path.name != BUNDLE:
                archive.writestr(path.name, b"tampered" if path.name == "README.md" else path.read_bytes())
    source = commit(repo)
    result = release_command(repo, source, "assemble", "--dist", dist, "--output", assets)
    assert result.returncode == 1
    assert "archive" in result.stderr


class Response(io.BytesIO):
    def __init__(self, url, data):
        super().__init__(data)
        self.url = url
        self.headers = {}

    def geturl(self):
        return self.url


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        ("yanked", "yanked"),
        ("index-hash", "hash"),
        ("download", "bytes"),
        ("extra", "unexpected"),
        ("pending", "not visible"),
        ("propagating", None),
        (None, None),
    ],
)
def test_index_verification_checks_both_listings_and_downloads(release, monkeypatch, failure, match):
    from ci import product_release

    repo, source, dist, assets, _candidate = release
    result = release_command(repo, source, "assemble", "--dist", dist, "--output", assets)
    assert result.returncode == 0, result.stderr
    downloads = {name: (assets / name).read_bytes() for name in (WHEEL, SDIST)}
    listing = []
    devzone = []
    clock = 0
    for name, data in downloads.items():
        file_hash = digest(data)
        listing.append(
            {
                "filename": name,
                "size": len(data),
                "url": f"https://files.pythonhosted.org/packages/test/{name}",
                "digests": {"sha256": file_hash},
                "yanked": failure == "yanked",
            }
        )
        if failure == "pending" and name == SDIST:
            continue
        if failure == "index-hash":
            file_hash = "f" * 64
        devzone.append(f'<a href="{name}#sha256={file_hash}">{name}</a>')
    if failure == "extra":
        devzone.append(f'<a href="skillevaluator-{VERSION}-py3-none-manylinux_x86_64.whl">extra</a>')

    def urlopen(request, *, timeout):
        assert timeout > 0
        url = request.full_url
        if url == f"https://pypi.org/pypi/skillevaluator/{VERSION}/json":
            return Response(url, json.dumps({"info": {"version": VERSION}, "urls": listing}).encode())
        if url == DEVZONE_URL:
            entries = devzone[:-1] if failure == "propagating" and clock == 0 else devzone
            return Response(url, "\n".join(entries).encode())
        name = url.rsplit("/", 1)[-1]
        if name in downloads and url.startswith(("https://files.pythonhosted.org/", DEVZONE_URL)):
            data = b"corrupt" if failure == "download" and url.startswith(DEVZONE_URL) else downloads[name]
            return Response(url, data)
        raise HTTPError(url, 404, "missing", {}, None)

    def advance_clock(delay):
        nonlocal clock
        clock += delay

    monkeypatch.setattr(product_release.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(product_release, "time", SimpleNamespace(monotonic=lambda: clock, sleep=advance_clock))
    if failure == "propagating":
        product_release.verify_indexes(VERSION, assets, wait_seconds=20)
        assert 0 < clock <= 20
    elif match is None:
        product_release.verify_indexes(VERSION, assets)
    else:
        with pytest.raises(ValueError, match=match):
            product_release.verify_indexes(VERSION, assets)


def test_pre_upload_inspection_checks_partial_index_bytes(monkeypatch):
    from ci import product_release

    url = f"https://files.pythonhosted.org/packages/test/{WHEEL}"
    entries = [(WHEEL, url, digest(b"wheel"), False, 5)]
    expected = {
        WHEEL: {"size": 5, "sha256": digest(b"wheel")},
        SDIST: {"size": 5, "sha256": digest(b"sdist")},
    }

    def urlopen(request, *, timeout):
        return Response(request.full_url, b"wheel")

    monkeypatch.setattr(product_release.urllib.request, "urlopen", urlopen)
    assert product_release.verify_index(entries, expected, "PyPI", 1, allow_missing=True) == {WHEEL}
    with pytest.raises(product_release.IndexPending, match="not visible"):
        product_release.verify_index(entries, expected, "PyPI", 1)


@pytest.mark.parametrize("status", [404, 503])
def test_absent_listing_is_distinguishable_from_an_index_outage(monkeypatch, status):
    from ci import product_release

    def urlopen(request, *, timeout):
        raise HTTPError(request.full_url, status, "unavailable", {}, None)

    monkeypatch.setattr(product_release.urllib.request, "urlopen", urlopen)
    error = product_release.IndexMissing if status == 404 else product_release.IndexPending
    with pytest.raises(error) as raised:
        product_release.pypi_entries(VERSION, 1)
    assert type(raised.value) is error


@pytest.mark.parametrize("state", ["absent", "matching", "one-index", "devzone-outage"])
def test_prepublication_index_inspection_detects_conflicts(monkeypatch, state):
    from ci import product_release

    expected = {WHEEL: {"sha256": digest(b"wheel"), "size": 5}}
    pypi_url = f"https://files.pythonhosted.org/packages/test/{WHEEL}"

    def urlopen(request, *, timeout):
        url = request.full_url
        if url == f"https://pypi.org/pypi/skillevaluator/{VERSION}/json":
            if state in {"absent", "devzone-outage"}:
                raise HTTPError(url, 404, "absent", {}, None)
            entry = {
                "filename": WHEEL,
                "size": 5,
                "url": pypi_url,
                "digests": {"sha256": digest(b"wheel")},
                "yanked": False,
            }
            return Response(url, json.dumps({"info": {"version": VERSION}, "urls": [entry]}).encode())
        if url == DEVZONE_URL:
            if state == "devzone-outage":
                raise HTTPError(url, 404, "outage", {}, None)
            body = f'<a href="{WHEEL}#sha256={digest(b"wheel")}">{WHEEL}</a>' if state == "matching" else ""
            return Response(url, body.encode())
        assert url in {pypi_url, f"{DEVZONE_URL}{WHEEL}"}
        return Response(url, b"wheel")

    monkeypatch.setattr(product_release.urllib.request, "urlopen", urlopen)
    if state in {"one-index", "devzone-outage"}:
        with pytest.raises(ValueError, match="only one index" if state == "one-index" else "404"):
            product_release.inspect_public_indexes(VERSION, expected)
    else:
        assert product_release.inspect_public_indexes(VERSION, expected) == ({WHEEL} if state == "matching" else set())


@pytest.mark.parametrize(
    "url",
    [
        None,
        "http://packages.example.com/skillevaluator/",
        "https://login@packages.example.com/skillevaluator/",
        "https://packages.example.com/skillevaluator/?query=1",
        "https://packages.example.com/skillevaluator/#section",
        "https://packages.example.com/simple/skillevaluator/",
    ],
)
def test_devzone_configuration_requires_an_https_project_listing(monkeypatch, url):
    from ci import product_release

    if url is None:
        monkeypatch.delenv("RELEASE_DEVZONE_INDEX_URL")
    else:
        monkeypatch.setenv("RELEASE_DEVZONE_INDEX_URL", url)
    with pytest.raises(ValueError, match="RELEASE_DEVZONE_INDEX_URL"):
        product_release.devzone_entries(VERSION, 1)


def test_devzone_rejects_a_link_to_a_different_origin(monkeypatch):
    from ci import product_release

    html = f'<a href="https://different.example.com/skillevaluator/{WHEEL}#sha256={digest(b"wheel")}">{WHEEL}</a>'

    def urlopen(request, *, timeout):
        assert request.full_url == DEVZONE_URL
        return Response(request.full_url, html.encode())

    monkeypatch.setattr(product_release.urllib.request, "urlopen", urlopen)
    with pytest.raises(ValueError, match="Unexpected public download URL"):
        product_release.devzone_entries(VERSION, 1)
