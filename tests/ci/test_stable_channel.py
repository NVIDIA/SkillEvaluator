# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "ghcr.io/nvidia/skillevaluator-tier1@sha256:" + "a" * 64
SCAN_REF = "b" * 40
PUBLISHER_REF = "c" * 40
STABLE_SCAN = "NVIDIA/SkillEvaluator/.github/workflows/skill-checks-stable.yml@stable"
DEVZONE_URL = "https://packages.example.com/skillevaluator/"


@pytest.fixture
def bundle(tmp_path):
    path = tmp_path / "1.2.3"
    path.mkdir()
    manifest = {
        "repository": "NVIDIA/SkillEvaluator",
        "workflow_ref": SCAN_REF,
        "publisher_ref": PUBLISHER_REF,
        "image": IMAGE,
    }
    receipt = {
        "all_verified": True,
        "image": IMAGE,
        "source_revision": "d" * 40,
        "public_manifest_matches_smoke_tested_image": True,
        "anonymous_pull_verified": True,
    }
    for name, content in {"release.json": manifest, "verification-receipt.json": receipt}.items():
        (path / name).write_text(json.dumps(content), encoding="utf-8")
    return path


def test_stable_workflows_pair_the_release_with_delegated_provenance(bundle):
    from ci.stable_channel import render

    files = render(bundle)
    scan = yaml.load(files["skill-checks-stable.yml"], Loader=yaml.BaseLoader)
    assert list(scan["jobs"]) == ["tier1"]
    job = scan["jobs"]["tier1"]
    assert "permissions" not in scan and "permissions" not in job
    assert job["uses"] == f"NVIDIA/SkillEvaluator/.github/workflows/evaluate-skills.yml@{SCAN_REF}"
    assert job["with"]["image"] == IMAGE
    assert "image" not in scan["on"]["workflow_call"]["inputs"]
    publisher = yaml.load(files["publish-skill-reports-stable.yml"], Loader=yaml.BaseLoader)
    assert publisher["permissions"] == {
        "actions": "read",
        "contents": "write",
        "id-token": "write",
        "pages": "write",
        "pull-requests": "write",
    }
    job = publisher["jobs"]["publish"]
    assert job["uses"] == f"NVIDIA/SkillEvaluator/.github/workflows/publish-skill-reports.yml@{PUBLISHER_REF}"
    assert job["with"]["tools-ref"] == PUBLISHER_REF
    assert job["with"]["trusted-wrapper-workflows"] == job["with"]["delegated-evaluator-wrapper"] == STABLE_SCAN
    assert job["with"]["trusted-caller-blobs"] == "${{ inputs.trusted-caller-blobs }}"
    assert "trusted-evaluator-shas" not in job["with"]

    comments = yaml.load(files["comment-skill-checks-stable.yml"], Loader=yaml.BaseLoader)
    job = comments["jobs"]["comment"]
    assert job["permissions"] == {"contents": "read", "actions": "read", "pull-requests": "write"}
    assert job["uses"] == f"NVIDIA/SkillEvaluator/.github/workflows/comment-skill-checks.yml@{PUBLISHER_REF}"
    assert job["with"]["tools-ref"] == PUBLISHER_REF
    assert job["with"]["trusted-wrapper-workflows"] == job["with"]["delegated-evaluator-wrapper"] == STABLE_SCAN
    assert set(comments["on"]["workflow_call"]["inputs"]) == {"run-id", "run-attempt", "trusted-caller-blobs"}


def test_manual_stable_comment_template_matches_the_callable_workflow(bundle):
    from ci.stable_channel import COMMENTS_WORKFLOW, render

    template = ROOT / "ci/templates/manual" / COMMENTS_WORKFLOW
    manual = yaml.load(template.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    callable_workflow = yaml.load(render(bundle)[COMMENTS_WORKFLOW], Loader=yaml.BaseLoader)

    assert set(manual["on"]) == {"workflow_run", "workflow_dispatch"}
    assert manual["on"]["workflow_run"] == {"workflows": ["Skill checks"], "types": ["completed"]}
    dispatch_inputs = manual["on"]["workflow_dispatch"]["inputs"]
    assert set(dispatch_inputs) == {"run-id", "run-attempt"}
    for field in dispatch_inputs.values():
        assert field["type"] == "string"
        assert field["required"] == "true"

    assert manual["permissions"] == {}
    assert set(manual["jobs"]) == {"comment"}
    job = manual["jobs"]["comment"]
    assert job["uses"] == f"NVIDIA/SkillEvaluator/.github/workflows/{COMMENTS_WORKFLOW}@stable"
    assert job["permissions"] == callable_workflow["jobs"]["comment"]["permissions"]
    assert set(job["with"]) == set(callable_workflow["on"]["workflow_call"]["inputs"])
    assert job["with"] == {
        "run-id": "${{ format('{0}', github.event.workflow_run.id || inputs.run-id) }}",
        "run-attempt": "${{ format('{0}', github.event.workflow_run.run_attempt || inputs.run-attempt) }}",
        "trusted-caller-blobs": "APPROVED_SCAN_FILE_BLOB",
    }


@pytest.mark.parametrize(
    ("filename", "key", "value", "match"),
    [
        ("release.json", "repository", "another/repository", "official"),
        ("release.json", "workflow_ref", "main", "workflow_ref"),
        ("release.json", "publisher_ref", "stable", "publisher_ref"),
        ("release.json", "image", "ghcr.io/nvidia/skillevaluator-tier1:latest", "image"),
        ("verification-receipt.json", "all_verified", False, "verification"),
        ("verification-receipt.json", "image", None, "verification"),
        ("verification-receipt.json", "source_revision", "main", "verification"),
    ],
)
def test_rejects_unverified_or_mutable_release(bundle, filename, key, value, match):
    from ci.stable_channel import render

    path = bundle / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**data, key: value}), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        render(bundle)


def test_checked_in_workflows_match_the_named_release_bundle():
    from ci.stable_channel import RELEASE_HEADER, render

    workflows = ROOT / ".github/workflows"
    version = (workflows / "skill-checks-stable.yml").read_text(encoding="utf-8").splitlines()[1]
    assert version.startswith(RELEASE_HEADER)
    bundle = ROOT / "ci/bundles" / version.removeprefix(RELEASE_HEADER).removeprefix("v")
    rendered = render(bundle)
    assert "stable-tier1.yml" in rendered
    for name, expected in rendered.items():
        directory = ROOT / "ci/gitlab" if name == "stable-tier1.yml" else workflows
        assert (directory / name).read_bytes() == expected.encode("utf-8")


def test_published_manifest_digest_is_verified_before_use(bundle, tmp_path):
    from ci.stable_channel import verify_publication

    manifest = tmp_path / "artifact-manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="published artifact manifest"):
        verify_publication(bundle, manifest, SCAN_REF, "sha256:" + "e" * 64, 2, "sha256:" + "f" * 64, 8)


def mock_index_transport(bin_dir):
    (bin_dir / "python3").symlink_to(sys.executable)
    (bin_dir / "sitecustomize.py").write_text(
        """\
import io
import json
import os
import urllib.request
from pathlib import Path

def open_fixture(self, request, *, timeout):
    url = request.full_url
    paths = json.loads(os.environ["TEST_INDEX_RESPONSES"])
    response = io.BytesIO(Path(paths[url]).read_bytes())
    response.geturl = lambda: url
    response.headers = {}
    return response

urllib.request.OpenerDirector.open = open_fixture
""",
        encoding="utf-8",
    )


def index_responses(directory, version, artifacts, problem):
    downloads = {}
    pypi = []
    devzone = []
    for artifact in artifacts:
        name = artifact["filename"]
        pypi_url = f"https://files.pythonhosted.org/packages/test/{name}"
        pypi.append(
            {
                "filename": name,
                "url": pypi_url,
                "size": artifact["size"],
                "digests": {"sha256": artifact["sha256"]},
                "yanked": False,
            }
        )
        path = directory / name
        if problem == "wrong-bytes" and name.endswith(".whl"):
            path = directory / "corrupt-wheel"
            path.write_bytes(b"x" * artifact["size"])
        downloads[pypi_url] = str(path)
        if problem == "missing-sdist" and name.endswith(".tar.gz"):
            continue
        devzone.append(f'<a href="{name}#sha256={artifact["sha256"]}">{name}</a>')
        downloads[f"{DEVZONE_URL}{name}"] = str(directory / name)
    pypi_listing = directory / "pypi.json"
    pypi_listing.write_text(json.dumps({"info": {"version": version}, "urls": pypi}), encoding="utf-8")
    devzone_listing = directory / "devzone.html"
    devzone_listing.write_text("\n".join(devzone), encoding="utf-8")
    return json.dumps(
        {
            **downloads,
            f"https://pypi.org/pypi/skillevaluator/{version}/json": str(pypi_listing),
            DEVZONE_URL: str(devzone_listing),
        }
    )


@pytest.mark.skipif(os.name != "posix", reason="The release promotion workflow runs Bash on Linux.")
@pytest.mark.parametrize("existing_github_only", [False, True])
def test_promotion_bootstraps_or_preserves_an_existing_release(tmp_path, bundle, existing_github_only):
    repo = tmp_path / "repo"
    repo.mkdir()
    upstream = tmp_path / "upstream.git"

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.PIPE).strip()

    git("init", "--bare", str(upstream))
    git("init", "--initial-branch=main")
    git("config", "user.name", "Release Test")
    git("config", "user.email", "release-test@example.com")
    git("config", "commit.gpgsign", "false")
    git("remote", "add", "origin", str(upstream))
    workflows = repo / ".github/workflows"
    workflows.mkdir(parents=True)
    (repo / "ci/gitlab").mkdir(parents=True)
    (repo / "ci/gitlab/skill-evaluation.yml").write_text("skill-evaluation: {}\n", encoding="utf-8")
    for name in ("evaluate-skills.yml", "publish-skill-reports.yml", "comment-skill-checks.yml"):
        (workflows / name).write_text("name: Released workflow\n", encoding="utf-8")
    (repo / "ci/Dockerfile.tier1").write_text("FROM scratch\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "Reviewed workflow and image source")
    source = git("rev-parse", "HEAD")

    def prepare_bundle(version, publisher_ref):
        dest = repo / "ci/bundles" / version
        shutil.copytree(bundle, dest)
        pins = dest / "release.json"
        pins.write_text(
            json.dumps({**json.loads(pins.read_text()), "workflow_ref": source, "publisher_ref": publisher_ref}),
            encoding="utf-8",
        )
        receipt = dest / "verification-receipt.json"
        receipt.write_text(json.dumps({**json.loads(receipt.read_text()), "source_revision": source}), encoding="utf-8")
        with ZipFile(dest / f"skillevaluator-ci-{version}.zip", "w") as archive:
            for path in (pins, receipt):
                archive.write(path, arcname=path.name)
        return dest

    initial = prepare_bundle("1.0.1", source)
    git("add", "ci/bundles")
    git("commit", "-m", "Immutable published release")
    git("tag", "v1.0.1")

    for name in (
        "github_api.py",
        "stable_channel.py",
        "product_release.py",
        "render_templates.py",
        "promote_stable.sh",
    ):
        shutil.copyfile(ROOT / "ci" / name, repo / "ci" / name)
    shutil.copytree(ROOT / "ci/templates/stable", repo / "ci/templates/stable")

    gitlab_include = repo / "ci/gitlab/stable-tier1.yml"
    gitlab_template = repo / "ci/templates/stable/stable-tier1.yml"

    comments_workflow = workflows / "comment-skill-checks-stable.yml"
    comments_template = repo / "ci/templates/stable" / comments_workflow.name

    def prepare_release(release_bundle, *, include_gitlab=True, include_comments=True):
        if not gitlab_template.is_file():
            shutil.copyfile(ROOT / "ci/templates/stable/stable-tier1.yml", gitlab_template)
        if not comments_template.is_file():
            shutil.copyfile(ROOT / "ci/templates/stable" / comments_template.name, comments_template)
        subprocess.run(
            [sys.executable, "ci/stable_channel.py", str(release_bundle)],
            cwd=repo,
            check=True,
        )
        if not include_gitlab:
            gitlab_template.unlink()
            gitlab_include.unlink()
        if not include_comments:
            comments_template.unlink()
            comments_workflow.unlink()
        git("add", ".")
        git("commit", "-m", f"Prepare stable workflows for {release_bundle.name}")
        return git("rev-parse", "HEAD")

    def published_ref():
        return git("ls-remote", "origin", "refs/tags/stable").split()[0]

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    mock_index_transport(bin_dir)
    fake_gh = bin_dir / "gh"
    fake_gh.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'case "$1" in\n'
        '    api) printf "%s\\n" "$TEST_RELEASE_RECORD" ;;\n'
        "    release)\n"
        "        shift 5\n"
        "        files=()\n"
        "        while [[ $1 == --pattern ]]; do\n"
        '            files+=("$2")\n'
        "            shift 2\n"
        "        done\n"
        '        for file in "${files[@]}"; do cp "$TEST_ASSETS/$file" "$2/$file"; done\n'
        "        ;;\n"
        "    *) exit 1 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)

    # Exercise real fetches and pushes against the bare remote while simulating
    # the production URL allowlist for the manual entry point.
    git_wrapper = bin_dir / "git"
    git_wrapper.write_text(
        "#!/bin/bash\n"
        "if [[ ${1:-} == remote && ${2:-} == get-url && ${TEST_APPROVED_ORIGIN:-} == true ]]; then\n"
        "    echo https://github.com/NVIDIA/SkillEvaluator.git\n"
        "    exit 0\n"
        "fi\n"
        'exec "$TEST_REAL_GIT" "$@"\n',
        encoding="utf-8",
    )
    git_wrapper.chmod(0o755)

    assets = tmp_path / "assets"
    assets.mkdir()

    def promote(
        tag,
        *,
        source_tree=None,
        bootstrap_sha="",
        event=None,
        index_problem=None,
        expected_release=None,
        mode=None,
        approved_origin=True,
    ):
        if event is None:
            event = "" if mode else "workflow_dispatch"
        if expected_release is None:
            expected_release = tag if mode or event == "workflow_dispatch" else ""
        version = tag.removeprefix("v")
        commit = git("rev-parse", f"refs/tags/{tag}^{{commit}}")
        archive = repo / "ci/bundles" / version / f"skillevaluator-ci-{version}.zip"
        archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        archive_size = archive.stat().st_size
        artifacts = []
        for name, payload in (
            (f"skillevaluator-{version}-py3-none-any.whl", b"verified wheel"),
            (f"skillevaluator-{version}.tar.gz", b"verified source distribution"),
        ):
            (assets / name).write_bytes(payload)
            artifacts.append({"filename": name, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
        data = {
            "version": version,
            "source_commit": commit,
            "source_tree": source_tree or git("rev-parse", f"{commit}^{{tree}}"),
            "image_source_commit": source,
            "image": IMAGE,
            "artifacts": artifacts,
            "ci_bundle": {
                "filename": archive.name,
                "size": archive_size,
                "sha256": archive_digest,
            },
        }
        body = json.dumps(data).encode()
        manifest_digest = hashlib.sha256(body).hexdigest()
        (assets / "artifact-manifest.json").write_bytes(body)
        (assets / "SHA256SUMS.txt").write_text(
            f"{manifest_digest}  artifact-manifest.json\n"
            + "".join(f"{artifact['sha256']}  {artifact['filename']}\n" for artifact in artifacts)
            + f"{archive_digest}  {archive.name}\n",
            encoding="utf-8",
        )
        return subprocess.run(
            ["bash", "ci/promote_stable.sh", *([mode, expected_release] if mode else [])],
            cwd=repo,
            env={
                **os.environ,
                "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(bin_dir),
                "TEST_RELEASE_RECORD": (
                    f"{tag}\tsha256:{manifest_digest}\t{len(body)}\tsha256:{archive_digest}\t{archive_size}"
                ),
                "TEST_ASSETS": str(assets),
                "TEST_INDEX_RESPONSES": index_responses(assets, version, artifacts, index_problem),
                "RELEASE_DEVZONE_INDEX_URL": DEVZONE_URL,
                "STABLE_PUSH_TOKEN": "" if mode else "synthetic-test-token",
                "STABLE_BOOTSTRAP_SHA": bootstrap_sha,
                "STABLE_EXPECTED_RELEASE": expected_release,
                "GITHUB_EVENT_NAME": event,
                "TEST_APPROVED_ORIGIN": str(approved_origin).lower(),
                "TEST_REAL_GIT": shutil.which("git"),
            },
            check=False,
            text=True,
            capture_output=True,
        )

    first = prepare_release(initial, include_gitlab=not existing_github_only, include_comments=False)
    next_bundle = prepare_bundle("1.0.2", first)
    second = prepare_release(next_bundle)
    git("push", "origin", "main", "v1.0.1")
    for blocked in (promote("v1.0.1"), promote("v1.0.1", bootstrap_sha=first, event="workflow_run")):
        assert blocked.returncode != 0
        assert "bootstrap-sha" in blocked.stderr
        assert not git("ls-remote", "origin", "refs/tags/stable")
    manual_bootstrap = promote("v1.0.1", mode="--manual")
    assert manual_bootstrap.returncode != 0
    assert "STABLE_BOOTSTRAP_SHA" in manual_bootstrap.stderr
    if not existing_github_only:
        pending = promote("v1.0.1", bootstrap_sha=first, index_problem="missing-sdist")
        assert pending.returncode != 0
        assert "DevZone artifacts are not visible yet" in pending.stderr
        assert not git("ls-remote", "origin", "refs/tags/stable")
        manual_promoted = promote("v1.0.1", bootstrap_sha=first, mode="--manual")
        assert manual_promoted.returncode == 0, manual_promoted.stderr
        assert published_ref() == first
        # Restore the unbootstrapped remote to test the Actions entry point too.
        git("push", "origin", "--delete", "stable")
    promoted = promote("v1.0.1", bootstrap_sha=first)
    if existing_github_only:
        assert promoted.returncode != 0
        assert "reviewed bootstrap commit must contain the GitLab stable include" in promoted.stderr
        assert not git("ls-remote", "origin", "refs/tags/stable")
        # An earlier release may have been bootstrapped by the GitHub-only promoter.
        git("push", "origin", f"{first}:refs/tags/stable")
    else:
        assert promoted.returncode == 0, promoted.stderr
    assert published_ref() == first
    early_retry = promote("v1.0.1", index_problem="missing-sdist")
    assert early_retry.returncode == 0, early_retry.stderr
    assert "already serves v1.0.1" in early_retry.stdout
    assert published_ref() == first
    git("tag", "v1.0.2")
    git("push", "origin", "main", "v1.0.2")

    for template in (repo / "ci/templates/stable/skill-checks-stable.yml", gitlab_template):
        template.write_text(template.read_text(encoding="utf-8") + "# Unreleased template edit\n", encoding="utf-8")
    prepare_release(prepare_bundle("1.0.3", second))
    git("push", "origin", "main")
    mismatch = promote("v1.0.2", expected_release="v1.0.1")
    assert mismatch.returncode != 0 and published_ref() == first
    assert "GitHub Latest is v1.0.2, but the expected release is v1.0.1." in mismatch.stderr
    for invalid_tag in ("", "1.0.2", "v01.0.2", "v1.0.2-rc1"):
        invalid = promote("v1.0.2", expected_release=invalid_tag)
        assert invalid.returncode != 0 and published_ref() == first
        assert "Manual promotion requires expected-release to be a canonical vX.Y.Z tag." in invalid.stderr
    rejected = promote("v1.0.2", source_tree="f" * 40)
    assert rejected.returncode != 0 and published_ref() == first
    assert "does not match the tagged CI release" in rejected.stderr
    corrupt = promote("v1.0.2", index_problem="wrong-bytes")
    assert corrupt.returncode != 0 and published_ref() == first
    assert "PyPI download has different bytes" in corrupt.stderr
    wrong_remote = promote("v1.0.2", mode="--manual", approved_origin=False)
    assert wrong_remote.returncode != 0 and published_ref() == first
    assert "requires the official HTTPS SkillEvaluator origin" in wrong_remote.stderr
    pending = promote("v1.0.2", mode="--manual", index_problem="missing-sdist")
    assert pending.returncode != 0 and published_ref() == first
    assert "DevZone artifacts are not visible yet" in pending.stderr
    checked = promote("v1.0.2", mode="--verify-only")
    assert checked.returncode == 0 and published_ref() == first, checked.stderr
    assert f"Verified v1.0.2 at {second} for stable promotion." in checked.stdout
    if existing_github_only:
        promoted = promote("v1.0.2", mode="--manual")
    else:
        promoted = promote("v1.0.2", event="workflow_run")
    assert promoted.returncode == 0, promoted.stderr
    assert "Verified SkillEvaluator 1.0.2 wheel and source distribution on PyPI and DevZone" in promoted.stdout
    assert published_ref() == second
    assert "# SkillEvaluator release: v1.0.2" in git("show", f"{second}:ci/gitlab/stable-tier1.yml")
    assert "# SkillEvaluator release: v1.0.2" in git("show", f"{second}:.github/workflows/{comments_workflow.name}")
    assert "already serves v1.0.2" in promote("v1.0.2").stdout
    missing_package = promote("v1.0.2", mode="--verify-only", index_problem="missing-sdist")
    assert missing_package.returncode != 0 and published_ref() == second
    assert "DevZone artifacts are not visible yet" in missing_package.stderr
    rechecked = promote("v1.0.2", mode="--verify-only")
    assert rechecked.returncode == 0 and published_ref() == second, rechecked.stderr
    assert "Verified SkillEvaluator 1.0.2 wheel and source distribution on PyPI and DevZone" in rechecked.stdout

    git("checkout", "--detach", first)
    stale = promote("v1.0.1")
    assert stale.returncode != 0
    assert "already serves a newer release" in stale.stderr
    assert published_ref() == second

    git("checkout", "main")
    git("rm", "ci/templates/stable/stable-tier1.yml")
    git("commit", "-m", "Incomplete GitLab stable configuration")
    git("tag", "v1.0.3")
    git("push", "origin", "main", "v1.0.3")
    incomplete = promote("v1.0.3")
    assert incomplete.returncode != 0
    assert "The GitLab stable template and generated include are both required" in incomplete.stderr
    assert published_ref() == second

    prepare_release(prepare_bundle("1.0.4", second), include_gitlab=False)
    git("tag", "v1.0.4")
    git("push", "origin", "main", "v1.0.4")
    dropped = promote("v1.0.4")
    assert dropped.returncode != 0
    assert "The next stable release must retain the GitLab stable include" in dropped.stderr
    assert published_ref() == second

    prepare_release(prepare_bundle("1.0.5", second))
    git("rm", str(comments_template))
    git("commit", "-m", "Incomplete stable comments configuration")
    git("tag", "v1.0.5")
    git("push", "origin", "main", "v1.0.5")
    incomplete = promote("v1.0.5")
    assert incomplete.returncode != 0
    assert "The stable comments template and generated workflow are both required" in incomplete.stderr
    assert published_ref() == second

    prepare_release(prepare_bundle("1.0.6", second), include_comments=False)
    git("tag", "v1.0.6")
    git("push", "origin", "main", "v1.0.6")
    dropped = promote("v1.0.6")
    assert dropped.returncode != 0
    assert "The next stable release must retain the comments-only workflow" in dropped.stderr
    assert published_ref() == second
