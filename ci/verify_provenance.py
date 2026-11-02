# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001 -- standalone trusted publisher helper.
"""Verify GitHub-signed execution identity without trusting report content.

The receipt is historical evidence, not a credential used for authentication.
Its signature and issuance during the selected run attempt remain mandatory
after expiration. A removed signing key makes old receipts unverifiable.
"""

from __future__ import annotations

import io
import json
import re
import stat
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime

ISSUER = "https://token.actions.githubusercontent.com"
JWKS_URL = ISSUER + "/.well-known/jwks"
AUDIENCE = "urn:skillevaluator:report-provenance:v1"
WORKFLOW_PATH = ".github/workflows/skill-checks.yml"
MAX_TOKEN_BYTES = 32 * 1024
MAX_RECEIPT_BYTES = 64 * 1024
MAX_JWKS_BYTES = 256 * 1024
CLOCK_SKEW_SECONDS = 30
MAX_TOKEN_LIFETIME_SECONDS = 900
SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


class ProvenanceError(ValueError):
    """A fixed diagnostic that never embeds receipt or provider response data."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, _request, _file, _code, _message, _headers, _url):
        return None


def fetch_signing_keys() -> dict:
    """Fetch only GitHub's fixed public key endpoint, without credentials."""
    request = urllib.request.Request(
        JWKS_URL,
        headers={"Accept": "application/json", "User-Agent": "SkillEvaluator-report-publisher"},
    )
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=30) as response:
            declared = response.headers.get("Content-Length")
            if declared and (not declared.isdecimal() or int(declared) > MAX_JWKS_BYTES):
                raise ProvenanceError("GitHub signing keys exceed the response limit.")
            payload = response.read(MAX_JWKS_BYTES + 1)
            if len(payload) > MAX_JWKS_BYTES:
                raise ProvenanceError("GitHub signing keys exceed the response limit.")
        result = json.loads(payload)
    except ProvenanceError:
        raise
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise ProvenanceError("GitHub signing keys are unavailable; retry report publication.") from exc
    if not isinstance(result, dict) or not isinstance(result.get("keys"), list) or not 1 <= len(result["keys"]) <= 32:
        raise ProvenanceError("GitHub returned an unsupported signing key document.")
    return result


def read_receipt(payload: bytes) -> str:
    """Read exactly one bounded, regular execution.jwt member without extraction."""
    if len(payload) > MAX_RECEIPT_BYTES:
        raise ProvenanceError("The execution receipt archive exceeds its size limit.")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if len(entries) != 1:
                raise ProvenanceError("The execution receipt must contain exactly one JWT file.")
            entry = entries[0]
            kind = stat.S_IFMT(entry.external_attr >> 16)
            if (
                entry.filename != "execution.jwt"
                or entry.is_dir()
                or entry.flag_bits & 1
                or kind not in {0, stat.S_IFREG}
                or not 1 <= entry.file_size <= MAX_TOKEN_BYTES
            ):
                raise ProvenanceError("The execution receipt contains an unsupported member.")
            with archive.open(entry) as handle:
                raw = handle.read(MAX_TOKEN_BYTES + 1)
            if len(raw) > MAX_TOKEN_BYTES:
                raise ProvenanceError("The execution receipt exceeds its size limit.")
            token = raw.decode("ascii").strip()
    except (OSError, ValueError, UnicodeError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ProvenanceError("The execution receipt archive is invalid.") from exc
    if re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", token) is None:
        raise ProvenanceError("The execution receipt is not a signed JWT.")
    return token


def _timestamp(value) -> float:
    try:
        result = datetime.fromisoformat(value)
        if result.tzinfo is None:
            raise ValueError
        return result.timestamp()
    except (ValueError, TypeError, AttributeError) as exc:
        raise ProvenanceError("The run attempt lacks a valid time interval.") from exc


def verify_receipt(
    payload: bytes,
    *,
    repository: str,
    run: dict,
    evaluator_workflow: str,
    evaluator_sha: str,
) -> dict:
    """Return signed identity claims only after signature and context validation."""
    # This optional publisher dependency is never imported by static scans.
    import jwt

    token = read_receipt(payload)
    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        if (
            header.get("alg") != "RS256"
            or header.get("typ") != "JWT"
            or not isinstance(kid, str)
            or not 1 <= len(kid) <= 256
        ):
            raise ProvenanceError("The execution receipt uses an unsupported signing header.")
        keys = fetch_signing_keys()["keys"]
        matches = [key for key in keys if isinstance(key, dict) and key.get("kid") == kid]
        if len(matches) != 1 or matches[0].get("kty") != "RSA" or matches[0].get("alg") != "RS256":
            raise ProvenanceError("The execution receipt signing key is unavailable; rerun the scan.")
        if matches[0].get("use") != "sig":
            raise ProvenanceError("The execution receipt signing key is not a signature key.")
        key = jwt.PyJWK.from_dict(matches[0], algorithm="RS256").key
        if not 2048 <= key.key_size <= 8192:
            raise ProvenanceError("The execution receipt signing key has an unsupported size.")
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=AUDIENCE,
            issuer=ISSUER,
            options={
                "strict_aud": True,
                "verify_exp": False,
                "verify_nbf": False,
                "verify_iat": False,
                "require": ["iss", "aud", "iat", "nbf", "exp"],
            },
        )
    except ProvenanceError:
        raise
    except (jwt.PyJWTError, TypeError, ValueError, KeyError) as exc:
        raise ProvenanceError("The execution receipt signature or audience is invalid.") from exc

    repository_id = run.get("repository", {}).get("id")
    if type(repository_id) is not int or repository_id <= 0:
        raise ProvenanceError("The workflow run lacks an immutable repository identity.")
    expected = {
        "repository": repository,
        "repository_id": str(repository_id),
        "run_id": str(run["id"]),
        "run_attempt": str(run["run_attempt"]),
        "event_name": run["event"],
        "job_workflow_ref": f"{evaluator_workflow}@{evaluator_sha}",
        "job_workflow_sha": evaluator_sha,
    }
    if any(type(claims.get(name)) is not str or claims[name] != value for name, value in expected.items()):
        raise ProvenanceError("The execution receipt does not match the selected run and approved evaluator.")
    workflow_sha = claims.get("workflow_sha")
    ref = claims.get("ref")
    if (
        not isinstance(workflow_sha, str)
        or SHA_PATTERN.fullmatch(workflow_sha) is None
        or claims.get("sha") != workflow_sha
        or not isinstance(ref, str)
        or len(ref) > 1024
        or claims.get("workflow_ref") != f"{repository}/{WORKFLOW_PATH}@{ref}"
    ):
        raise ProvenanceError("The execution receipt lacks a valid caller workflow identity.")
    if run["event"] == "pull_request":
        if re.fullmatch(r"refs/pull/[1-9][0-9]*/merge", ref) is None:
            raise ProvenanceError("The execution receipt lacks a pull-request merge identity.")
    elif re.fullmatch(r"refs/(?:heads|tags)/[^\x00-\x20\x7f]+", ref) is None or workflow_sha != run["head_sha"]:
        raise ProvenanceError("The execution receipt does not match the scanned revision.")

    iat, nbf, exp = (claims.get(name) for name in ("iat", "nbf", "exp"))
    started, finished = (_timestamp(run.get(name)) for name in ("run_started_at", "updated_at"))
    if (
        any(type(value) is not int for value in (iat, nbf, exp))
        or finished < started
        or not started - CLOCK_SKEW_SECONDS <= iat <= finished + CLOCK_SKEW_SECONDS
        or iat > time.time() + CLOCK_SKEW_SECONDS
        or not 0 < exp - iat <= MAX_TOKEN_LIFETIME_SECONDS
        or not 0 <= iat - nbf <= MAX_TOKEN_LIFETIME_SECONDS
    ):
        raise ProvenanceError("The execution receipt was not issued during the selected run attempt.")
    return {
        name: claims[name] for name in ("workflow_sha", "job_workflow_sha", "repository_id", "run_id", "run_attempt")
    }
