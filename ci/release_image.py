#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve a GHCR release tag without confusing auth failures with missing tags."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class ReleaseImageError(RuntimeError):
    """A registry response cannot establish the release identity."""


def resolve_digest(image: str, revision: str, *, token: str = "", actor: str = "") -> str | None:
    if not re.fullmatch(r"ghcr\.io/[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)+", image):
        raise ReleaseImageError("Expected a lowercase GHCR repository")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ReleaseImageError("Expected a full source revision")
    repository = image.removeprefix("ghcr.io/")
    scope = urllib.parse.urlencode({"service": "ghcr.io", "scope": f"repository:{repository}:pull"})
    headers = {}
    if token:
        credentials = base64.b64encode(f"{actor}:{token}".encode()).decode()
        headers["Authorization"] = f"Basic {credentials}"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(f"https://ghcr.io/token?{scope}", headers=headers), timeout=30
        ) as response:
            authorization = json.loads(response.read(65536)).get("token")
        if not isinstance(authorization, str) or not authorization:
            raise ReleaseImageError("Registry did not return a pull token")
        request = urllib.request.Request(
            f"https://ghcr.io/v2/{repository}/manifests/sha-{revision}",
            headers={
                "Authorization": f"Bearer {authorization}",
                "Accept": (
                    "application/vnd.oci.image.index.v1+json, "
                    "application/vnd.oci.image.manifest.v1+json, "
                    "application/vnd.docker.distribution.manifest.list.v2+json, "
                    "application/vnd.docker.distribution.manifest.v2+json"
                ),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                manifest = response.read(4 * 1024 * 1024 + 1)
                digest = response.headers.get("Docker-Content-Digest", "")
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            payload = json.loads(error.read(65536))
            codes = {entry.get("code") for entry in payload.get("errors", []) if isinstance(entry, dict)}
            if codes and codes <= {"MANIFEST_UNKNOWN", "NAME_UNKNOWN"}:
                return None
            raise ReleaseImageError("Registry 404 did not identify a missing manifest") from error
        if len(manifest) > 4 * 1024 * 1024 or digest != "sha256:" + hashlib.sha256(manifest).hexdigest():
            raise ReleaseImageError("Registry manifest digest did not match its content")
        return f"{image}@{digest}"
    except urllib.error.HTTPError as error:
        raise ReleaseImageError(
            f"Registry request failed with HTTP {error.code}; refusing to overwrite a tag"
        ) from error
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        raise ReleaseImageError("Registry identity could not be verified; refusing to overwrite a tag") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require", action="store_true")
    args = parser.parse_args()
    try:
        digest = resolve_digest(
            args.image, args.revision, token=os.environ.get("GHCR_TOKEN", ""), actor=os.environ.get("GITHUB_ACTOR", "")
        )
        if args.require and not digest:
            raise ReleaseImageError("The published revision tag was not found")
        with args.output.open("a") as output:
            output.write(f"exists={'true' if digest else 'false'}\nimage_digest={digest or ''}\n")
    except (ReleaseImageError, OSError) as error:
        sys.stderr.write(f"Release image verification failed: {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
