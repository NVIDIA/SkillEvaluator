# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001 -- shared trusted CI helper.
"""Read per-skill security approvals from an explicit, immutable Git revision.

Both the mapping and baseline bytes come from Git objects, never the candidate
working tree. The caller supplies the platform's target commit for a PR/MR and
the scanned commit for a push. This helper performs no network operations.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

import yaml

from skillevaluator.utils.structured_data import load_bounded_yaml
from skillevaluator.validators.security import _BoundedSkillSpectorBaselineLoader

try:
    from secret_redaction import collect_sensitive_bytes
    from select_changed_skills import SelectionError, _git, has_control_character
except ModuleNotFoundError:
    from ci.secret_redaction import collect_sensitive_bytes
    from ci.select_changed_skills import SelectionError, _git, has_control_character

MANIFEST = ".skillevaluator/skillspector-baselines.yaml"
MAX_MANIFEST_BYTES = 64 * 1024
MAX_BASELINE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
MAX_BASELINES = 1000
SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


class BaselineConfigurationError(ValueError):
    """Reviewed policy could not be loaded; no scan should start."""


def _path(value: object, *, skill: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise BaselineConfigurationError("Security baseline paths must be bounded relative paths.")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError as exc:
        raise BaselineConfigurationError("Security baseline paths must contain valid UTF-8 text.") from exc
    if size > 4096:
        raise BaselineConfigurationError("Security baseline paths must be bounded relative paths.")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or ":" in value
        or has_control_character(value)
        or value != path.as_posix()
        or (value == "." and not skill)
    ):
        raise BaselineConfigurationError("Security baseline paths must be canonical relative paths without traversal.")
    return value


def _blob(workspace: Path, revision: str, path: str, limit: int) -> bytes:
    listing = _git(workspace, "ls-tree", "-l", "-z", revision, "--", path)
    records = listing.split(b"\0")
    if len(records) != 2 or records[1] or b"\t" not in records[0]:
        raise BaselineConfigurationError("A reviewed security baseline file is missing from the policy commit.")
    header, name = records[0].split(b"\t", 1)
    fields = header.split()
    if len(fields) != 4 or fields[0] not in (b"100644", b"100755") or fields[1] != b"blob" or name != path.encode():
        raise BaselineConfigurationError("Reviewed security baseline files must be regular Git blobs, not links.")
    if not fields[3].isdigit() or int(fields[3]) > limit:
        raise BaselineConfigurationError("A reviewed security baseline file exceeds its size limit.")
    content = _git(workspace, "cat-file", "blob", fields[2].decode("ascii"))
    if len(content) != int(fields[3]):
        raise BaselineConfigurationError("A reviewed security baseline Git blob could not be read completely.")
    return content


def _decoded_policy_bytes(raw: bytes) -> bytes:
    """Also inspect decoded strings that scanner reports can echo after YAML escapes."""
    parsed = yaml.load(raw.decode("utf-8"), Loader=_BoundedSkillSpectorBaselineLoader)
    return json.dumps(parsed, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def load_security_baselines(workspace: Path, environ: dict[str, str]) -> tuple[dict[str, dict], dict | None, set[str]]:
    """Return immutable baseline bytes keyed by exact skill path and audit facts."""
    enabled = environ.get("SECURITY_BASELINES", "false")
    if enabled not in ("true", "false"):
        raise BaselineConfigurationError("SECURITY_BASELINES must be true or false.")
    if enabled == "false":
        return {}, None, set()
    revision = environ.get("POLICY_SOURCE_SHA", "")
    if not SHA.fullmatch(revision):
        raise BaselineConfigurationError("Security baselines require an exact POLICY_SOURCE_SHA commit.")
    try:
        resolved = _git(workspace, "rev-parse", "--show-toplevel", revision + "^{commit}", "--").decode().splitlines()
        if len(resolved) != 3 or Path(resolved[0]).resolve() != workspace or resolved[1:] != [revision, "--"]:
            raise BaselineConfigurationError("The security policy revision must identify a commit in this checkout.")
        raw = _blob(workspace, revision, MANIFEST, MAX_MANIFEST_BYTES)
        try:
            text = raw.decode("utf-8")
            if any(getattr(event, "anchor", None) is not None for event in yaml.parse(text)):
                raise ValueError("Aliases and anchors are not allowed in a security approval manifest.")
            manifest = load_bounded_yaml(text)
        except (ValueError, UnicodeError, RecursionError, yaml.YAMLError) as exc:
            raise BaselineConfigurationError("The reviewed security baseline manifest is invalid YAML.") from exc
        if (
            not isinstance(manifest, dict)
            or set(manifest) != {"version", "baselines"}
            or type(manifest["version"]) is not int
            or manifest["version"] != 1
            or not isinstance(manifest["baselines"], dict)
            or len(manifest["baselines"]) > MAX_BASELINES
        ):
            raise BaselineConfigurationError(
                "The reviewed manifest requires version: 1 and a bounded baselines mapping."
            )
        baselines, files, total = {}, {}, len(raw)
        for skill_path, file_path in manifest["baselines"].items():
            skill_path, file_path = _path(skill_path, skill=True), _path(file_path)
            if file_path not in files:
                content = _blob(workspace, revision, file_path, MAX_BASELINE_BYTES)
                total += len(content)
                if total > MAX_TOTAL_BYTES:
                    raise BaselineConfigurationError("Reviewed security baselines exceed the total size limit.")
                files[file_path] = content
            content = files[file_path]
            baselines[skill_path] = {
                "file": file_path,
                "digest": "sha256:" + hashlib.sha256(content).hexdigest(),
                "content": content,
            }
        try:
            inputs = [raw, *files.values()]
            tokens = collect_sensitive_bytes([*inputs, *(_decoded_policy_bytes(content) for content in inputs)])
        except (
            ValueError,
            UnicodeError,
            yaml.YAMLError,
            TypeError,
            RecursionError,
            IndexError,
            AttributeError,
            KeyError,
            OverflowError,
        ) as exc:
            raise BaselineConfigurationError(
                "Credential preflight could not safely inspect reviewed security policy."
            ) from exc
        return (
            baselines,
            {
                "source_commit": revision,
                "manifest": MANIFEST,
                "manifest_digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
                "mapped_skill_count": len(baselines),
            },
            tokens,
        )
    except SelectionError as exc:
        raise BaselineConfigurationError(
            "The reviewed security policy commit or its files are unavailable; verify checkout history."
        ) from exc
