# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import io
import json
import shlex
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tarfile import TarFile, TarInfo
from types import SimpleNamespace
from urllib.error import HTTPError
from zipfile import ZipFile

import pytest
import yaml

from ci import github_release, release_bundle, stable_channel

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
def release(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    package = repo / "src/skillevaluator"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(f"__version__ = {VERSION!r}\n")
    (repo / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools>=77", "wheel"]\nbuild-backend = "setuptools.build_meta"\n'
        f'[project]\nname = "skillevaluator"\nversion = "{VERSION}"\nrequires-python = ">=3.12,<3.15"\n'
        '[project.urls]\nDocumentation = "https://docs.nvidia.com/skills/skillevaluator/"\n'
        '[tool.setuptools.package-data]\nskillevaluator = ["ci-release.json", "ci-release-all-tiers.json"]\n',
        encoding="utf-8",
    )
    (repo / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    write_json(
        package / DESCRIPTORS[1],
        {"schema_version": 2, "capabilities": CAPABILITIES, "report_contract": "skillevaluator-ci-v1"},
    )
    candidate = commit(repo)
    smoke = tmp_path / "smoke-receipt.json"
    write_json(
        smoke,
        {
            "all_verified": True,
            "image": IMAGE,
            "runtime": {
                "id": "sha256:" + "b" * 64,
                "digests": [IMAGE],
                "os": "linux",
                "architecture": "amd64",
                "source_revision": candidate,
            },
            "cases": [
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

    def scanner_version(command, **_):
        assert command[:2] == ["docker", "run"]
        scanner = command[command.index("--entrypoint") + 1]
        return VERSION if scanner == "skillevaluator" else "2.0.0"

    generated = tmp_path / "generated"
    with monkeypatch.context() as patch:
        patch.setattr(release_bundle, "ROOT", repo)
        patch.setattr(release_bundle.subprocess, "check_output", scanner_version)
        release_bundle.build_bundle(generated, VERSION, candidate, IMAGE, smoke, RUN)
    shutil.copytree(generated, repo, dirs_exist_ok=True)
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


def test_release_workflow_builds_only_distribution_artifacts():
    path = SCRIPT.parents[1] / ".github/workflows/release.yml"
    workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assemble = next(
        step["run"]
        for step in workflow["jobs"]["prepare"]["steps"]
        if "product_release.py assemble" in step.get("run", "")
    )
    command = shlex.split(
        next(line for line in assemble.replace("\\\n", " ").splitlines() if line.startswith("uv build "))
    )
    assert "--no-create-gitignore" in command, "uv build adds .gitignore to the strict distribution directory"


def test_assembles_generated_bundle_and_verifies_exact_assets(release):
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


@pytest.mark.parametrize(
    ("invalid", "error"),
    [
        (None, ""),
        ("changelog", "no dated entry"),
        ("worktree-only-heading", "no dated entry"),
        ("removed-worktree-heading", ""),
        ("digest", "does not match the reviewed release digest"),
        ("missing-image", "does not match the reviewed release digest"),
        ("source", "does not match the reviewed source"),
    ],
)
def test_validate_checks_the_reviewed_changelog_and_live_public_image(release, monkeypatch, capsys, invalid, error):
    repo, _, dist, assets, candidate = release
    changelog = repo / "CHANGELOG.md"
    dated = f"# Changelog\n\n## {VERSION} - 2026-11-20\n\nPublished changes.\n"
    undated = f"# Changelog\n\n## {VERSION}\n\nPublished changes.\n"
    changelog.write_text(undated if invalid in ("changelog", "worktree-only-heading") else dated, encoding="utf-8")
    source = commit(repo)
    result = release_command(repo, source, "assemble", "--dist", dist, "--output", assets)
    assert result.returncode == 0, result.stderr
    if invalid == "worktree-only-heading":
        changelog.write_text(dated, encoding="utf-8")
    if invalid == "removed-worktree-heading":
        changelog.write_text(undated, encoding="utf-8")
    repository = "ghcr.io/nvidia/skillevaluator-tier1"
    published = {"digest": f"{repository}@sha256:{'b' * 64}", "missing-image": None}.get(invalid, IMAGE)
    lookups = []

    def resolve_image(repo_name, source_revision):
        lookups.append((repo_name, source_revision))
        return published

    def unexpected_write(*_, **__):
        pytest.fail("Release validation must avoid GitHub writes")

    monkeypatch.chdir(repo)
    monkeypatch.setattr(github_release.release_image, "resolve_digest", resolve_image)
    monkeypatch.setattr(github_release, "api", unexpected_write)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "github_release.py",
            "verify-release-metadata",
            "--version",
            VERSION,
            "--source-sha",
            candidate if invalid == "source" else source,
            "--assets",
            str(assets),
        ],
    )
    assert github_release.main() == (1 if error else 0)
    stderr = capsys.readouterr().err
    if error:
        assert error in stderr
    else:
        assert not stderr
    if invalid in (None, "removed-worktree-heading", "digest", "missing-image"):
        assert lookups == [(repository, candidate)]


@pytest.mark.parametrize(
    "artifact_change",
    [
        None,
        "manifest-digest",
        "source-sha",
        "wheel-bytes",
        "extra-file",
        "wrong-run",
        "duplicate-artifact",
        "archive-digest",
    ],
)
def test_publish_acquires_only_the_exact_validated_release(release, monkeypatch, artifact_change):
    repo, source, dist, validated, candidate = release
    result = release_command(repo, source, "assemble", "--dist", dist, "--output", validated)
    assert result.returncode == 0, result.stderr
    original = {path.name: path.read_bytes() for path in validated.iterdir()}
    manifest_sha256 = digest(original["artifact-manifest.json"])
    archive = io.BytesIO()
    with ZipFile(archive, "w") as zip_file:
        for name, data in original.items():
            zip_file.writestr(name, b"changed" if artifact_change == "wheel-bytes" and name == WHEEL else data)
        if artifact_change == "extra-file":
            zip_file.writestr("other.zip", b"unexpected")
    zipped = archive.getvalue()

    def api(path):
        if path == "actions/runs/123/attempts/2":
            return {
                "id": 123,
                "run_attempt": 2,
                "head_branch": "main",
                "head_sha": "a" * 40,
                "event": "workflow_dispatch",
                "path": ".github/workflows/release.yml",
                "head_repository": {"id": 42, "full_name": "NVIDIA/SkillEvaluator"},
                "repository": {"id": 42, "full_name": "NVIDIA/SkillEvaluator"},
                "display_title": f"Validate release · SkillEvaluator {VERSION}",
                "status": "completed",
                "conclusion": "success",
            }

        assert path == f"actions/runs/123/artifacts?name=validated-release-{VERSION}-123-2"
        artifact = {
            "id": 789,
            "name": f"validated-release-{VERSION}-123-2",
            "expired": False,
            "size_in_bytes": len(zipped),
            "digest": "sha256:" + ("b" * 64 if artifact_change == "archive-digest" else digest(zipped)),
            "workflow_run": {
                "id": 456 if artifact_change == "wrong-run" else 123,
                "repository_id": 42,
                "head_repository_id": 42,
                "head_branch": "main",
                "head_sha": "a" * 40,
            },
        }
        artifacts = [artifact, {**artifact, "id": 790}] if artifact_change == "duplicate-artifact" else [artifact]
        return {"total_count": len(artifacts), "artifacts": artifacts}

    def download(args, *, stdout, check):
        assert args == ["gh", "api", "repos/NVIDIA/SkillEvaluator/actions/artifacts/789/zip"]
        assert check
        assert artifact_change not in ("wrong-run", "duplicate-artifact")
        stdout.write(zipped)

    monkeypatch.chdir(repo)
    monkeypatch.setattr(github_release, "tag_target", lambda _: None)
    monkeypatch.setattr(github_release, "get_release", lambda _: None)
    monkeypatch.setattr(github_release, "api", api)
    monkeypatch.setattr(github_release.subprocess, "run", download)

    if artifact_change == "manifest-digest":
        manifest_sha256 = "a" * 64
    if artifact_change == "source-sha":
        source = candidate
    output = validated.with_name("publish")
    if artifact_change:
        with pytest.raises(ValueError, match=r"manifest|exactly the five release files|validation artifact"):
            github_release.acquire_publication(VERSION, source, output, "123-2", manifest_sha256)
    else:
        assert github_release.acquire_publication(VERSION, source, output, "123-2", manifest_sha256) is True
        assert {path.name: path.read_bytes() for path in output.iterdir()} == original


def test_real_uv_build_output_assembles(release, tmp_path):
    uv = shutil.which("uv")
    if uv is None or any(importlib.util.find_spec(module) is None for module in ("setuptools", "wheel")):
        pytest.skip("The offline package build requires uv, setuptools, and wheel")
    repo, source, _fixture_dist, assets, _candidate = release
    dist = tmp_path / "uv-build"
    build = subprocess.run(
        [
            uv,
            "build",
            "--offline",
            "--no-build-isolation",
            "--python",
            sys.executable,
            "--no-sources",
            "--no-create-gitignore",
            "--out-dir",
            str(dist),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    assert {path.name for path in dist.iterdir()} == {WHEEL, SDIST}
    result = release_command(repo, source, "assemble", "--dist", dist, "--output", assets)
    assert result.returncode == 0, result.stderr
    assert release_command(repo, source, "verify-assets", "--assets", assets).returncode == 0


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
        ("yanked-after-lag", "yanked"),
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
    downloads_served = []
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
            entries = (
                [{**entry, "yanked": True} for entry in listing] if failure == "yanked-after-lag" and clock else listing
            )
            return Response(url, json.dumps({"info": {"version": VERSION}, "urls": entries}).encode())
        if url == DEVZONE_URL:
            entries = devzone[:-1] if failure in {"propagating", "yanked-after-lag"} and clock == 0 else devzone
            return Response(url, "\n".join(entries).encode())
        name = url.rsplit("/", 1)[-1]
        if name in downloads and url.startswith(("https://files.pythonhosted.org/", DEVZONE_URL)):
            downloads_served.append(url)
            data = b"corrupt" if failure == "download" and url.startswith(DEVZONE_URL) else downloads[name]
            return Response(url, data)
        raise HTTPError(url, 404, "missing", {}, None)

    def advance_clock(delay):
        nonlocal clock
        clock += delay

    monkeypatch.setattr(product_release.INDEX_OPENER, "open", urlopen)
    monkeypatch.setattr(product_release, "time", SimpleNamespace(monotonic=lambda: clock, sleep=advance_clock))
    if failure == "propagating":
        product_release.verify_indexes(VERSION, assets, wait_seconds=20)
        assert 0 < clock <= 20
        assert len(downloads_served) == len(set(downloads_served)) == 4
    elif match is None:
        product_release.verify_indexes(VERSION, assets)
        assert set(downloads_served) == {
            *(f"https://files.pythonhosted.org/packages/test/{name}" for name in downloads),
            *(f"{DEVZONE_URL}{name}" for name in downloads),
        }
    else:
        with pytest.raises(ValueError, match=match):
            wait = 20 if failure == "yanked-after-lag" else 0
            product_release.verify_indexes(VERSION, assets, wait_seconds=wait)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (None, None),
        ("propagating", None),
        ("wrong-bytes", "different bytes"),
        ("wrong-hash", "wrong hash"),
        ("wrong-host", "Unexpected public download URL"),
        ("yanked", "yanked artifact"),
        ("missing", "not visible yet"),
    ],
)
def test_testpypi_cli_saves_only_verified_public_downloads(release, monkeypatch, capsys, tmp_path, failure, message):
    from ci import product_release

    repo, source, dist, assets, _candidate = release
    result = release_command(repo, source, "assemble", "--dist", dist, "--output", assets)
    assert result.returncode == 0, result.stderr
    downloads = {name: (assets / name).read_bytes() for name in (WHEEL, SDIST)}
    output = tmp_path / "verified-downloads"
    testpypi_url = f"https://test.pypi.org/pypi/skillevaluator/{VERSION}/json"
    file_base = "https://test-files.pythonhosted.org/packages/test/"
    steps = []
    listing = []
    for name, data in downloads.items():
        if name == SDIST and failure == "missing":
            continue
        listing.append(
            {
                "filename": name,
                "size": len(data),
                "url": f"{file_base}{name}" if failure != "wrong-host" else f"https://files.pythonhosted.org/{name}",
                "digests": {"sha256": digest(data) if failure != "wrong-hash" else "0" * 64},
                "yanked": failure == "yanked",
            }
        )

    def urlopen(request, *, timeout):
        assert timeout > 0
        assert not output.exists()
        url = request.full_url
        steps.append(url)
        if url == testpypi_url:
            return Response(url, json.dumps({"info": {"version": VERSION}, "urls": listing}).encode())
        assert url in {f"{file_base}{name}" for name in downloads}
        name = url.rsplit("/", 1)[-1]
        data = downloads[name]
        if name == SDIST and failure == "wrong-bytes":
            data = bytes([data[0] ^ 1]) + data[1:]
        if name == SDIST and failure == "propagating" and steps.count(url) == 1:
            raise HTTPError(url, 404, "missing", {}, None)
        return Response(url, data)

    clock = 0

    def advance_clock(delay):
        nonlocal clock
        clock += delay

    monkeypatch.delenv("RELEASE_DEVZONE_INDEX_URL")
    monkeypatch.setattr(product_release.INDEX_OPENER, "open", urlopen)
    monkeypatch.setattr(product_release, "time", SimpleNamespace(monotonic=lambda: clock, sleep=advance_clock))
    wait = "20" if failure == "propagating" else "0"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "product_release.py",
            "verify-testpypi",
            "--version",
            VERSION,
            "--assets",
            str(assets),
            "--wait-seconds",
            wait,
            "--output",
            str(output),
        ],
    )
    if message is None:
        product_release.main()
        assert {path.name: path.read_bytes() for path in output.iterdir()} == downloads
        assert steps == [testpypi_url, f"{file_base}{WHEEL}", f"{file_base}{SDIST}"] * (2 if failure else 1)
    else:
        with pytest.raises(SystemExit) as raised:
            product_release.main()
        assert raised.value.code == 1
        assert message in capsys.readouterr().err
        assert not output.exists()


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

    monkeypatch.setattr(product_release.INDEX_OPENER, "open", urlopen)
    assert product_release.verify_index(entries, expected, "PyPI", 1, allow_missing=True) == {WHEEL}
    with pytest.raises(product_release.IndexPending, match="not visible"):
        product_release.verify_index(entries, expected, "PyPI", 1)


@pytest.mark.parametrize("status", [404, 503])
def test_absent_listing_is_distinguishable_from_an_index_outage(monkeypatch, status):
    from ci import product_release

    def urlopen(request, *, timeout):
        raise HTTPError(request.full_url, status, "unavailable", {}, None)

    monkeypatch.setattr(product_release.INDEX_OPENER, "open", urlopen)
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

    monkeypatch.setattr(product_release.INDEX_OPENER, "open", urlopen)
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

    monkeypatch.setattr(product_release.INDEX_OPENER, "open", urlopen)
    with pytest.raises(ValueError, match="Unexpected public download URL"):
        product_release.devzone_entries(VERSION, 1)


def test_index_redirect_is_not_followed_or_treated_as_missing():
    from ci import product_release

    paths = []

    class RedirectToMissing(BaseHTTPRequestHandler):
        def do_GET(self):
            paths.append(self.path)
            if self.path == "/index":
                self.send_response(302)
                self.send_header("Location", "/missing")
            else:
                self.send_response(404)
            self.end_headers()

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), RedirectToMissing) as server:
        listener = threading.Thread(target=server.serve_forever, daemon=True)
        listener.start()
        try:
            with pytest.raises(ValueError, match="redirect"):
                product_release.get_url(f"http://127.0.0.1:{server.server_port}/index", 1024, 5)
        finally:
            server.shutdown()
            listener.join()
    assert paths == ["/index"]
