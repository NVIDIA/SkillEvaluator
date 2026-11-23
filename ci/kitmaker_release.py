# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001 -- release workflow entry point.

"""Submit verified GitHub release assets to Kitmaker and resume by receipt."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from ci.github_api import bounded_read, strict_json

from ci import product_release

REPOSITORY = "NVIDIA/SkillEvaluator"
PROJECT = "skillevaluator"
VERSION = re.compile(r"(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){2}")
SHA1 = re.compile(r"[a-f0-9]{40}")
SHA256 = re.compile(r"[a-f0-9]{64}")
RELEASE_UUID = re.compile(r"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}")
MAX_JSON = 4 * 1024 * 1024
MAX_ASSET = 1024**3 - 1
FERRY_URL = "http://127.0.0.1:8888/kitmaker-portal/api/v0"
SUCCESSFUL_JOB_STATES = frozenset({"completed", "success"})


class ReleaseError(RuntimeError):
    """A release operation could not be verified safely."""


class RedirectPolicy(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_hosts: tuple[str, ...] = ()):
        self.allowed_hosts = allowed_hosts

    def redirect_request(self, request, fp, code, message, headers, new_url):
        parsed = urllib.parse.urlsplit(new_url)
        if parsed.scheme == "https" and parsed.netloc in self.allowed_hosts:
            return super().redirect_request(request, fp, code, message, headers, new_url)
        return None


def open_request(url, *, label, headers=None, data=None, context=None, missing=False, redirects=()):
    """Open a bounded, explicit trust boundary without exposing request details on error."""
    handlers = [RedirectPolicy(redirects)]
    if context is not None:
        handlers.append(urllib.request.HTTPSHandler(context=context))
    if url.startswith("http://127.0.0.1:8888/"):
        handlers.append(urllib.request.ProxyHandler({}))
    request_headers = {"User-Agent": "SkillEvaluator-Release", **(headers or {})}
    encoded = None
    if data is not None:
        encoded = json.dumps(data, allow_nan=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, headers=request_headers, data=encoded)
    try:
        return urllib.request.build_opener(*handlers).open(request, timeout=45)
    except urllib.error.HTTPError as exc:
        exc.close()
        if missing and exc.code == 404:
            return None
        raise ReleaseError(f"{label} returned HTTP {exc.code}.") from None
    except (OSError, ValueError):
        raise ReleaseError(f"{label} request could not be completed.") from None


def request_bytes(url, **kwargs):
    response = open_request(url, **kwargs)
    if response is None:
        return None
    try:
        with response:
            return bounded_read(response, MAX_JSON, ReleaseError)
    except (OSError, http.client.HTTPException):
        raise ReleaseError("Could not read the release service response.") from None


def request_json(url, **kwargs):
    raw = request_bytes(url, **kwargs)
    if raw is None:
        return None
    return as_object(strict_json(raw, ReleaseError), "service response")


def as_object(value, label):
    if not isinstance(value, dict):
        raise ReleaseError(f"The {label} must be a JSON object.")
    return value


def required_env(key):
    value = os.getenv(key)
    if not value:
        raise ReleaseError(f"Configure {key} in the release environment.")
    return value


def oidc_token(audience):
    url = required_env("ACTIONS_ID_TOKEN_REQUEST_URL")
    if not url.startswith("https://"):
        raise ReleaseError("GitHub Actions provided an invalid OIDC endpoint.")
    suffix = urllib.parse.urlencode({"audience": audience})
    response = request_json(
        f"{url}&{suffix}" if "?" in url else f"{url}?{suffix}",
        label="GitHub OIDC",
        headers={"Authorization": f"Bearer {required_env('ACTIONS_ID_TOKEN_REQUEST_TOKEN')}"},
    )
    token = response.get("value")
    if not isinstance(token, str) or not token:
        raise ReleaseError("GitHub OIDC did not return an identity token.")
    return token


@dataclass(frozen=True)
class Artifact:
    sha256: str
    size: int


@dataclass(frozen=True)
class Release:
    version: str
    source_sha: str
    manifest_sha256: str
    artifacts: dict[str, Artifact]
    asset_tag: str = ""

    @property
    def asset_base(self) -> str:
        tag = checked_asset_tag(self.version, self.asset_tag)
        return f"https://github.com/{REPOSITORY}/releases/download/{tag}/"

    def urls(self, names) -> tuple[str, ...]:
        return tuple(self.asset_base + name for name in self.artifacts if name in names)


def checked_asset_tag(version: str, asset_tag: str = "") -> str:
    tag = asset_tag or f"v{version}"
    if not VERSION.fullmatch(version) or tag not in (f"v{version}", f"v{version}-assets"):
        raise ReleaseError("Asset tag must be vVERSION or vVERSION-assets for the same package version.")
    return tag


def parse_manifest(version: str, source_sha: str, manifest_sha256: str, raw: bytes, asset_tag: str = "") -> Release:
    tag = checked_asset_tag(version, asset_tag)
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ReleaseError("GitHub release manifest digest did not match the approved digest.")
    manifest = as_object(strict_json(raw, ReleaseError), "release manifest")
    if manifest.get("version") != version or manifest.get("source_commit") != source_sha:
        raise ReleaseError("GitHub release manifest did not match the approved version and source.")
    if tag != f"v{version}" and manifest.get("asset_tag") != tag:
        raise ReleaseError("The recovery manifest must bind the approved asset tag.")
    expected = (f"{PROJECT}-{version}-py3-none-any.whl", f"{PROJECT}-{version}.tar.gz")
    items = manifest.get("artifacts")
    if not isinstance(items, list) or len(items) != len(expected):
        raise ReleaseError("The manifest must specify exactly one wheel and one source distribution.")
    artifacts = {}
    for item in items:
        item = as_object(item, "release artifact")
        filename, digest, size = item.get("filename"), item.get("sha256"), item.get("size")
        if (
            filename not in expected
            or filename in artifacts
            or not isinstance(digest, str)
            or not SHA256.fullmatch(digest)
            or type(size) is not int
            or not 0 < size <= MAX_ASSET
        ):
            raise ReleaseError("The release manifest contains invalid or duplicate distribution metadata.")
        artifacts[filename] = Artifact(digest, size)
    return Release(version, source_sha, manifest_sha256, {name: artifacts[name] for name in expected}, asset_tag)


def verify_download(url: str, artifact: Artifact):
    digest = hashlib.sha256()
    received = 0
    try:
        with open_request(
            url,
            label="GitHub release asset",
            redirects=(
                "release-assets.githubusercontent.com",
                "objects.githubusercontent.com",
            ),
        ) as response:
            while chunk := response.read(min(1024**2, artifact.size - received + 1)):
                received += len(chunk)
                if received > artifact.size:
                    raise ReleaseError("A GitHub release asset exceeds its approved size.")
                digest.update(chunk)
    except (OSError, http.client.HTTPException):
        raise ReleaseError("Could not read the GitHub release asset.") from None
    if received != artifact.size or digest.hexdigest() != artifact.sha256:
        raise ReleaseError("A GitHub release asset does not match its approved size and digest.")


def verify_recovery_assets(published: dict, release: Release, raw: bytes) -> None:
    """Bind all five recovery assets to the reviewed manifest, without changing package pins."""
    manifest = as_object(strict_json(raw, ReleaseError), "release manifest")
    bundle = as_object(manifest.get("ci_bundle"), "CI bundle")
    name, digest, size = bundle.get("filename"), bundle.get("sha256"), bundle.get("size")
    if (
        name != f"{PROJECT}-ci-{release.version}.zip"
        or not isinstance(digest, str)
        or not SHA256.fullmatch(digest)
        or type(size) is not int
        or not 0 < size <= MAX_ASSET
    ):
        raise ReleaseError("The recovery manifest must identify the exact CI bundle.")
    files = {**release.artifacts, name: Artifact(digest, size)}
    expected = {*files, product_release.MANIFEST, product_release.CHECKSUMS}
    assets = published.get("assets")
    if (
        not isinstance(assets, list)
        or len(assets) != len(expected)
        or any(not isinstance(asset, dict) or asset.get("state") != "uploaded" for asset in assets)
        or {asset.get("name") for asset in assets} != expected
    ):
        raise ReleaseError("Recovery publication requires all five uploaded release assets.")
    verify_download(release.asset_base + name, files[name])
    sums = request_bytes(
        release.asset_base + product_release.CHECKSUMS,
        label="GitHub recovery checksums",
        redirects=("release-assets.githubusercontent.com", "objects.githubusercontent.com"),
    )
    try:
        product_release.verify_checksum_file(
            sums,
            {product_release.MANIFEST: release.manifest_sha256, **{name: item.sha256 for name, item in files.items()}},
            "Recovery checksums",
        )
    except ValueError as exc:
        raise ReleaseError(str(exc)) from None


def verify_github_release(version: str, source_sha: str, manifest_sha256: str, asset_tag: str = "") -> Release:
    if not (VERSION.fullmatch(version) and SHA1.fullmatch(source_sha) and SHA256.fullmatch(manifest_sha256)):
        raise ReleaseError("Expected a bare stable version, full source SHA and bare manifest SHA256.")
    base_api = f"https://api.github.com/repos/{REPOSITORY}"
    github_headers = {
        "Authorization": f"Bearer {required_env('GH_TOKEN')}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    def github(path):
        return request_json(base_api + path, label="GitHub release identity", headers=github_headers)

    tag = checked_asset_tag(version, asset_tag)
    # A companion download release must identify the same source as the locked product release.
    for identity_tag in dict.fromkeys((f"v{version}", tag)):
        published = github(f"/releases/tags/{identity_tag}")
        if (
            published.get("tag_name") != identity_tag
            or published.get("immutable") is not True
            or published.get("draft") is not False
            or published.get("prerelease") is not False
            or not published.get("published_at")
        ):
            raise ReleaseError("Kitmaker requires the published, immutable GitHub release for the approved tag.")
        reference = github(f"/git/ref/tags/{identity_tag}")
        if reference.get("ref") != f"refs/tags/{identity_tag}":
            raise ReleaseError("GitHub did not resolve the approved release tag.")
        tagged = as_object(reference.get("object"), "Git tag object")
        for _ in range(5):
            if tagged.get("type") != "tag":
                break
            annotated_sha = tagged.get("sha")
            if not isinstance(annotated_sha, str) or not SHA1.fullmatch(annotated_sha):
                raise ReleaseError("GitHub returned an invalid annotated tag.")
            tagged = as_object(github(f"/git/tags/{annotated_sha}").get("object"), "annotated Git tag object")
        if tagged.get("type") != "commit" or tagged.get("sha") != source_sha:
            raise ReleaseError("The release tag does not refer to the approved source commit.")

    asset_base = f"https://github.com/{REPOSITORY}/releases/download/{tag}/"
    raw = request_bytes(
        asset_base + "artifact-manifest.json",
        label="GitHub release manifest",
        redirects=("release-assets.githubusercontent.com", "objects.githubusercontent.com"),
    )
    release = parse_manifest(version, source_sha, manifest_sha256, raw, asset_tag)
    if tag != f"v{version}":
        verify_recovery_assets(published, release, raw)
    for name, artifact in release.artifacts.items():
        verify_download(asset_base + name, artifact)
    return release


def check_indexes(release: Release) -> set[str]:
    expected = {
        name: {"sha256": artifact.sha256, "size": artifact.size} for name, artifact in release.artifacts.items()
    }
    try:
        return product_release.inspect_public_indexes(release.version, expected)
    except (ValueError, OSError, http.client.HTTPException):
        raise ReleaseError(
            "The exact artifacts on PyPI and DevZone could not be verified safely. "
            "Check for an index outage, a digest conflict, or an artifact present on only one index."
        ) from None


class Vault:
    def __init__(self, url, token, namespace, context):
        self.url = url
        self.headers = {"X-Vault-Token": token}
        if namespace:
            self.headers["X-Vault-Namespace"] = namespace
        self.context = context

    @classmethod
    def connect(cls):
        url = required_env("KITMAKER_VAULT_URL").rstrip("/")
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ReleaseError("Configure KITMAKER_VAULT_URL as an approved HTTPS Vault origin.")
        mount = required_env("KITMAKER_VAULT_AUTH_MOUNT")
        if not re.fullmatch(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*", mount):
            raise ReleaseError("Configure a valid Vault JWT authentication mount.")
        namespace = os.getenv("KITMAKER_VAULT_NAMESPACE")
        ca_pem = os.getenv("KITMAKER_VAULT_CA_PEM")
        context = ssl.create_default_context()
        if ca_pem:
            context.load_verify_locations(cadata=ca_pem)
        response = request_json(
            f"{url}/v1/auth/{mount}/login",
            label="Vault authentication",
            context=context,
            headers={"X-Vault-Namespace": namespace} if namespace else None,
            data={
                "role": required_env("KITMAKER_VAULT_ROLE"),
                "jwt": oidc_token(required_env("KITMAKER_VAULT_AUDIENCE")),
            },
        )
        token = as_object(response.get("auth"), "Vault authentication").get("client_token")
        if not isinstance(token, str) or not token:
            raise ReleaseError("Vault did not issue an authorized workflow token.")
        return cls(url, token, namespace, context)

    def read(self, path, *, missing=False):
        response = request_json(
            f"{self.url}/v1/{path}",
            label="Vault KV read",
            headers=self.headers,
            context=self.context,
            missing=missing,
        )
        if response is None:
            return None, 0
        envelope = as_object(response.get("data"), "Vault KV envelope")
        version = as_object(envelope.get("metadata"), "Vault KV metadata").get("version")
        if type(version) is not int or version < 1:
            raise ReleaseError("Vault did not return a valid receipt revision.")
        return as_object(envelope.get("data"), "Vault KV data"), version

    def write(self, path, data, revision):
        response = request_json(
            f"{self.url}/v1/{path}",
            label="Vault KV compare-and-set",
            headers=self.headers,
            context=self.context,
            data={"options": {"cas": revision}, "data": data},
        )
        next_revision = as_object(response.get("data"), "Vault KV update").get("version")
        if next_revision != revision + 1:
            raise ReleaseError("Vault could not confirm the next receipt revision.")
        return next_revision


@dataclass(frozen=True)
class VaultReceipts:
    vault: Vault
    path: str

    def read(self):
        return self.vault.read(self.path, missing=True)

    def write(self, data, revision):
        return self.vault.write(self.path, data, revision)


def vault_path(key):
    value = required_env(key).rstrip("/")
    if not re.fullmatch(r"[A-Za-z0-9_-]+/data/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+", value) or any(
        part in {".", ".."} for part in value.split("/")
    ):
        raise ReleaseError(f"Configure {key} as a Vault KV v2 data path.")
    return value


class PortalClient:
    def __init__(self, project_id, owner_email, token):
        self.project_id = project_id
        self.owner_email = owner_email
        self.token = token

    def call(self, path, *, data=None):
        return request_json(
            FERRY_URL + path,
            label="Kitmaker Portal",
            headers={
                "Authorization": f"Bearer {self.token}",
                "X-Charon-GHA-Token": oidc_token("charon.nvidia.com"),
            },
            data=data,
        )

    def submit(self, urls, *, upload):
        payload = [
            {
                "pic": self.owner_email,
                "job_type": "wheel-release-job",
                "url": url,
                "upload": upload,
                "mirroring_strategy": "upload-both",
            }
            for url in urls
        ]
        receipt = self.call(f"/projects/{self.project_id}/releases", data={"project_name": PROJECT, "payload": payload})
        release_uuid = receipt.get("release_uuid")
        if not isinstance(release_uuid, str) or not RELEASE_UUID.fullmatch(release_uuid):
            raise ReleaseError("Kitmaker did not return a valid submission receipt.")
        return release_uuid

    def wait(self, receipt) -> tuple[str, dict[str, str]]:
        release_uuid = receipt.get("uuid")
        if not isinstance(release_uuid, str) or not RELEASE_UUID.fullmatch(release_uuid):
            raise ReleaseError("The stored Kitmaker submission receipt is invalid.")
        post_response_uuid = receipt.get("post_response_uuid")
        if post_response_uuid is not None and post_response_uuid != release_uuid:
            raise ReleaseError("The saved Kitmaker UUID differs from its recorded POST response.")
        if type(receipt.get("upload")) is not bool:
            raise ReleaseError("The stored Kitmaker submission has no valid upload mode.")
        mode_verified = post_response_uuid is not None
        deadline = time.monotonic() + 45 * 60
        while time.monotonic() < deadline:
            status = self.call(f"/status/{release_uuid}")
            if status.get("uuid") != release_uuid or status.get("project_id") != self.project_id:
                raise ReleaseError("Kitmaker returned a different project or submission receipt.")
            result = status.get("status")
            if not isinstance(result, str) or not result:
                raise ReleaseError("Kitmaker returned an invalid release status.")
            reported_payload = status.get("payload")
            if reported_payload is not None:
                self.check_echoed_request(reported_payload, receipt)
                mode_verified = True
            if result in {"failed", "completed"}:
                if not mode_verified:
                    raise ReleaseError("Kitmaker could not prove the upload mode for this manually reconciled UUID.")
                return result, self.check_terminal_jobs(status, receipt, result)
            time.sleep(15)
        raise ReleaseError("Kitmaker is still pending. Resume this submission from its stored receipt.")

    @staticmethod
    def check_echoed_request(reported_payload, receipt):
        message = "The Kitmaker request does not match the expected upload mode and artifact URLs."
        entries = as_object(reported_payload, "Kitmaker request payload").get("payload")
        if not isinstance(entries, list) or len(entries) != len(receipt["urls"]):
            raise ReleaseError(message)
        reported_urls = []
        for entry in entries:
            entry = as_object(entry, "Kitmaker artifact request")
            if entry.get("upload") is not receipt["upload"] or not isinstance(entry.get("url"), str):
                raise ReleaseError(message)
            reported_urls.append(entry["url"])
        if sorted(reported_urls) != sorted(receipt["urls"]):
            raise ReleaseError(message)

    @staticmethod
    def check_terminal_jobs(status, receipt, result) -> dict[str, str]:
        allowed_states = SUCCESSFUL_JOB_STATES
        message = "Kitmaker completion did not cover every submitted artifact."
        if result == "failed":
            allowed_states = SUCCESSFUL_JOB_STATES | {"canceled", "cancelled", "failed", "skipped"}
            message = (
                "Kitmaker reported failure before confirming all artifact jobs are terminal. Reconcile the receipt."
            )
        jobs = status.get("jobs")
        if not isinstance(jobs, list) or len(jobs) != len(receipt["urls"]):
            raise ReleaseError(message)
        outcomes = {}
        for job in jobs:
            job = as_object(job, "Kitmaker artifact status")
            url = job.get("wheel_url")
            job_status = job.get("status")
            if job_status not in allowed_states or not isinstance(url, str) or url in outcomes:
                raise ReleaseError(message)
            outcomes[url] = job_status
        if outcomes.keys() != set(receipt["urls"]):
            raise ReleaseError(message)
        return outcomes


def finish_attempt(state, revision, attempts, urls, portal, receipts, *, upload, dry_run_uuids=frozenset()):
    attempt = attempts[-1] if attempts else None
    if attempt is None or attempt["status"] == "failed":
        attempt = {"urls": list(urls), "upload": upload, "status": "submitting"}
        attempts.append(attempt)
        revision = receipts.write(state, revision)
        # A lost response can still mean the POST reached Kitmaker. Keep the checkpoint and never replay it.
        release_uuid = portal.submit(urls, upload=upload)
        if release_uuid in dry_run_uuids:
            raise ReleaseError("Kitmaker returned a dry-run UUID for publication. Reconcile the private receipt.")
        attempt.update(uuid=release_uuid, post_response_uuid=release_uuid, status="submitted")
        revision = receipts.write(state, revision)
    if attempt["status"] == "submitting":
        raise ReleaseError("A previous Kitmaker POST is ambiguous. Reconcile the private receipt before retrying.")
    if attempt["status"] != "submitted" or attempt.get("upload") is not upload:
        raise ReleaseError("The private Kitmaker receipt contains an invalid submission state or upload mode.")
    status, job_statuses = portal.wait(attempt)
    attempt.update(status=status, job_statuses=job_statuses)
    receipts.write(state, revision)
    if status != "completed":
        raise ReleaseError("A release request failed in Kitmaker. Inspect the private receipt before retrying.")


def release_to_kitmaker(mode, release, portal, receipts, inventory):
    binding = {
        "version": release.version,
        "source_sha": release.source_sha,
        "manifest_sha256": release.manifest_sha256,
        "project_id": portal.project_id,
    }
    if checked_asset_tag(release.version, release.asset_tag) != f"v{release.version}":
        binding["asset_tag"] = release.asset_tag
    state, revision = receipts.read()
    if state is None:
        state = {"binding": binding, "dry_runs": [], "publications": []}
    if state.get("binding") != binding:
        raise ReleaseError("The private Kitmaker receipt is bound to a different release or project.")
    urls = release.urls(release.artifacts)
    dry_runs = state["dry_runs"]
    dry = dry_runs[-1] if dry_runs else None
    if dry is not None and (dry.get("urls") != list(urls) or dry.get("upload") is not False):
        raise ReleaseError("The saved Kitmaker dry run does not cover the approved assets and upload mode.")
    dry_completed = dry is not None and dry.get("status") == "completed"
    if mode == "dry-run":
        if not dry_completed:
            finish_attempt(state, revision, dry_runs, urls, portal, receipts, upload=False)
        return

    if not dry_completed:
        raise ReleaseError("Kitmaker publication requires a completed dry run of the approved assets.")
    publications = state["publications"]
    dry_run_uuids = {attempt["uuid"] for attempt in dry_runs if "uuid" in attempt}
    for attempt in publications:
        if attempt.get("uuid") in dry_run_uuids:
            raise ReleaseError("The publication receipt uses a dry-run UUID. Reconcile the private receipt.")
        if attempt.get("upload") is not True or not attempt.get("urls") or not set(attempt["urls"]) <= set(urls):
            raise ReleaseError("A saved Kitmaker publication does not match the approved assets or upload mode.")
    previous = publications[-1] if publications else None
    if previous and previous["status"] == "completed":
        # The caller separately verifies index convergence and clean installs. Never replay a completed submission.
        return
    if previous and previous["status"] == "submitted":
        finish_attempt(state, revision, publications, (), portal, receipts, upload=True)
        return

    present = inventory(release)
    if present == set(release.artifacts):
        return
    already_uploaded = set()
    for attempt in publications:
        if attempt["status"] == "failed":
            outcomes = attempt.get("job_statuses")
            if not isinstance(outcomes, dict):
                raise ReleaseError(
                    "A failed Kitmaker receipt lacks per-artifact outcomes. Reconcile the private receipt."
                )
            already_uploaded.update(url for url, status in outcomes.items() if status in SUCCESSFUL_JOB_STATES)
    # Both public listings can lag the Portal. Keep every confirmed success out of future requests.
    missing_urls = tuple(url for url in release.urls(set(release.artifacts) - present) if url not in already_uploaded)
    if missing_urls:
        finish_attempt(
            state, revision, publications, missing_urls, portal, receipts, upload=True, dry_run_uuids=dry_run_uuids
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry-run", "publish", "verify"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--asset-tag", default="", help="Optional vVERSION-assets recovery download tag.")
    args = parser.parse_args()
    try:
        checkout = verify_github_release(args.version, args.source_sha, args.manifest_sha256, args.asset_tag)
        if args.mode == "verify":
            if check_indexes(checkout) != set(checkout.artifacts):
                raise ReleaseError("Both public indexes must serve the exact wheel and source archive.")
            sys.stdout.write("Verified the exact Python distributions on PyPI and NVIDIA DevZone.\n")
            return 0
        config_path = vault_path("KITMAKER_VAULT_CONFIG_PATH")
        receipt_prefix = vault_path("KITMAKER_VAULT_RECEIPT_PATH")
        receipt_path = f"{receipt_prefix}/{args.version}"
        if config_path == receipt_prefix or config_path.startswith(receipt_prefix + "/"):
            raise ReleaseError("Vault must use separate release credential and receipt paths.")
        vault = Vault.connect()
        config, _ = vault.read(config_path)
        project_id = config.get("project_id")
        if not re.fullmatch(r"[1-9][0-9]{0,12}", str(project_id)):
            raise ReleaseError("Vault must provide the approved numeric Kitmaker project ID.")
        owner, token = config.get("owner_email"), config.get("token")
        if (
            not isinstance(owner, str)
            or not re.fullmatch(r"[^@\s]+@[^@\s]+", owner)
            or not isinstance(token, str)
            or not re.fullmatch(r"kmp_service_[A-Za-z0-9_.-]+", token)
        ):
            raise ReleaseError("Vault must provide the Kitmaker project owner's email and service-account token.")
        portal = PortalClient(int(project_id), owner, token)
        release_to_kitmaker(args.mode, checkout, portal, VaultReceipts(vault, receipt_path), check_indexes)
    except ReleaseError as exc:
        sys.stderr.write(f"Kitmaker release stopped. {exc}\n")
        return 1
    except (OSError, ValueError, KeyError, TypeError):
        sys.stderr.write("Kitmaker release stopped. The release configuration or service response was invalid.\n")
        return 1
    sys.stdout.write(f"Kitmaker {args.mode} finished. The receipt is stored in the configured private Vault path.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
