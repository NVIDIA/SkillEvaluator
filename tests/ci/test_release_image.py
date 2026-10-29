# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import io
import os
import subprocess
import urllib.error
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location("release_image", ROOT / "ci/release_image.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
IMAGE = "ghcr.io/owner/scanner"
REVISION = "a" * 40


class Response(io.BytesIO):
    def __init__(self, body, headers=None):
        super().__init__(body)
        self.headers = headers or {}


def registry(monkeypatch, outcome):
    calls = []

    def open_url(request, timeout):
        calls.append(request)
        if "/token?" in request.full_url:
            return Response(b'{"token":"test-bearer"}')
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(release.urllib.request, "urlopen", open_url)
    return calls


def test_resolves_existing_tag_and_checks_manifest_content(monkeypatch):
    body = b'{"schemaVersion":2}'
    digest = "sha256:" + hashlib.sha256(body).hexdigest()
    calls = registry(monkeypatch, Response(body, {"Docker-Content-Digest": digest}))
    assert release.resolve_digest(IMAGE, REVISION, token="secret", actor="owner") == f"{IMAGE}@{digest}"
    assert calls[0].get_header("Authorization").startswith("Basic ")
    assert calls[1].get_header("Authorization") == "Bearer test-bearer"
    assert calls[1].full_url.endswith("/manifests/sha-" + REVISION)


def test_only_explicit_registry_not_found_permits_new_build(monkeypatch):
    error = urllib.error.HTTPError(
        "https://ghcr.io", 404, "missing", {}, io.BytesIO(b'{"errors":[{"code":"MANIFEST_UNKNOWN"}]}')
    )
    registry(monkeypatch, error)
    assert release.resolve_digest(IMAGE, REVISION) is None


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_authentication_and_transient_failures_do_not_allow_overwrite(monkeypatch, status):
    error = urllib.error.HTTPError("https://ghcr.io", status, "failed", {}, io.BytesIO(b""))
    registry(monkeypatch, error)
    with pytest.raises(release.ReleaseImageError, match="refusing to overwrite"):
        release.resolve_digest(IMAGE, REVISION)


def test_unrecognized_not_found_and_digest_mismatch_fail_closed(monkeypatch):
    error = urllib.error.HTTPError("https://ghcr.io", 404, "missing", {}, io.BytesIO(b'{"errors":[{"code":"DENIED"}]}'))
    registry(monkeypatch, error)
    with pytest.raises(release.ReleaseImageError, match="missing manifest"):
        release.resolve_digest(IMAGE, REVISION)
    registry(monkeypatch, Response(b"changed content", {"Docker-Content-Digest": "sha256:" + "0" * 64}))
    with pytest.raises(release.ReleaseImageError, match="digest"):
        release.resolve_digest(IMAGE, REVISION)


@pytest.mark.parametrize("scenario", ["existing", "new", "race", "auth-error", "wrong-revision"])
def test_release_shell_never_pushes_an_existing_or_unverified_tag(tmp_path, scenario):
    workflow = yaml.load((ROOT / ".github/workflows/publish-tier1-image.yml").read_text(), Loader=yaml.BaseLoader)
    publish = workflow["jobs"]["publish"]
    script = next(step["run"] for step in publish["steps"] if step.get("id") == "release")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    docker = binaries / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$COMMAND_LOG"\n'
        'if [ "$1" = inspect ]; then\n'
        '  if [ "$SCENARIO" = wrong-revision ]; then printf "bad-revision\\n"; else printf "%s\\n" "$GITHUB_SHA"; fi\n'
        "fi\n"
    )
    docker.chmod(0o755)
    python = binaries / "python3"
    python.write_text(
        "#!/bin/sh\n"
        'case "$*" in *--require*) exit 0 ;; esac\n'
        'case "$SCENARIO" in auth-error) exit 1 ;; race) exists=true ;; *) exists=false ;; esac\n'
        'printf "exists=%s\\n" "$exists" > "$RUNNER_TEMP/pre-push-check"\n'
    )
    python.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
        "RUNNER_TEMP": str(tmp_path),
        "COMMAND_LOG": str(tmp_path / "commands"),
        "SCENARIO": scenario,
        "IMAGE": IMAGE,
        "GITHUB_SHA": REVISION,
        "GHCR_TOKEN": "dummy",
        "GITHUB_ACTOR": "owner",
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "EXISTING_IMAGE": IMAGE + "@sha256:" + "b" * 64 if scenario in {"existing", "wrong-revision"} else "",
    }
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script], env=environment, capture_output=True, text=True
    )
    commands = (tmp_path / "commands").read_text()
    assert ("push " in commands) is (scenario == "new")
    assert (result.returncode == 0) is (scenario in {"existing", "new"})
    if scenario == "existing":
        assert "pull " + environment["EXISTING_IMAGE"] in commands


def test_anonymous_verification_retry_has_no_build_or_write_permission():
    workflow = yaml.load((ROOT / ".github/workflows/publish-tier1-image.yml").read_text(), Loader=yaml.BaseLoader)
    onboarding = workflow["jobs"]["onboarding"]
    assert onboarding["needs"] == "publish"
    assert onboarding["env"]["IMAGE_DIGEST"] == "${{ needs.publish.outputs.image_digest }}"
    assert onboarding["permissions"] == {"contents": "read"}
    commands = "\n".join(step.get("run", "") for step in onboarding["steps"])
    assert "docker build" not in commands and "docker push" not in commands and "docker login" not in commands
    assert 'pull "$published_image"' in commands


def test_scanner_locks_cover_every_requirement_with_hashes():
    requirements = ROOT / "ci/requirements"
    for source in requirements.glob("*.in"):
        locked = source.with_suffix(".txt").read_text()
        entries = locked.replace("\\\n", " ").splitlines()
        assert entries
        for entry in entries:
            if not entry.strip() or entry.startswith("#"):
                continue
            assert "==" in entry or " @ https://github.com/NVIDIA/SkillSpector/archive/" in entry
            assert "--hash=sha256:" in entry
    dockerfile = (ROOT / "ci/Dockerfile.tier1").read_text()
    assert "python:3.12-slim-bookworm@sha256:" in dockerfile
    assert dockerfile.count("--require-hashes") == 4
    assert "--no-build-isolation" in dockerfile
