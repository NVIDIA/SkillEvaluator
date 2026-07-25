# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation policy / profile system.

Implements audience-aware policy for SkillEvaluator validators. A
:class:`ValidationPolicy` carries the severity overrides and identity
constraints (e.g. acceptable author email domains) that a validator should
apply for a given repository or organization.

Validators emit findings with stable ``(category, check_name)`` identifiers
and a ``default`` severity; the active policy decides the final severity
through :meth:`ValidationPolicy.severity_for`. This keeps validators
policy-blind and lets CI pipelines override severities without a code release.

Profiles are shipped as YAML in ``skillevaluator/config/profiles/`` and selected
at runtime via the ``--profile`` CLI flag, the ``--policy PATH`` flag (custom
overlay), or the ``SKILLEVALUATOR_PROFILE`` environment variable.

"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from skillevaluator.logging_config import get_logger
from skillevaluator.models.result import Severity, ValidationResult

logger = get_logger(__name__)

PROFILES_DIR = Path(__file__).parent.parent / "config" / "profiles"

DEFAULT_PROFILE_NAME = "external"

#: Environment variable that selects the default profile when no CLI flag is set.
PROFILE_ENV_VAR = "SKILLEVALUATOR_PROFILE"

# Stable, profile-agnostic email-shape pattern: "Name <local@domain>".
_AUTHOR_SHAPE_PATTERN = re.compile(r"\S[^<>\n]* <[^<>@\s]+@[^<>\s]+>")


@dataclass(frozen=True)
class ValidationPolicy:
    """Audience-aware policy bundle consumed by validators.

    Attributes:
        profile: Human-readable profile name or a custom name supplied by a
            user policy file.
        author_email_regex: Compiled regex applied to ``metadata.author``
            values to enforce a required email domain. ``None`` disables the
            domain check (the shape check still runs).
        severity_overrides: Mapping ``"CATEGORY.check_name"`` or
            ``"CATEGORY.*"`` -> :class:`Severity`. Overrides the default
            severity that the validator would otherwise emit.
        source: Path to the YAML file the policy was loaded from (for
            diagnostics). ``None`` for programmatically constructed policies.
    """

    profile: str = DEFAULT_PROFILE_NAME
    audience: str = "external"
    author_email_regex: re.Pattern[str] | None = None
    severity_overrides: dict[str, Severity] = field(default_factory=dict)
    source: Path | None = None

    @property
    def digest(self) -> str:
        """Return a stable digest for the effective, source-independent policy."""
        payload = {
            "profile": self.profile,
            "audience": self.audience,
            "author_email_regex": self.author_email_regex.pattern if self.author_email_regex else None,
            "severity_overrides": {key: value.value for key, value in sorted(self.severity_overrides.items())},
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return f"sha256:{hashlib.sha256(canonical).hexdigest()}"

    @property
    def author_shape_regex(self) -> re.Pattern[str]:
        """Profile-agnostic ``Name <email>`` shape pattern."""
        return _AUTHOR_SHAPE_PATTERN

    def severity_for(
        self,
        category: str,
        check_name: str,
        default: Severity,
    ) -> Severity:
        """Return severity for ``CATEGORY.check_name`` or ``CATEGORY.*``."""
        exact_key = f"{category}.{check_name}"
        wildcard_key = f"{category}.*"
        return self.severity_overrides.get(
            exact_key,
            self.severity_overrides.get(wildcard_key, default),
        )

    def is_author_email_acceptable(self, author: str) -> bool:
        """Return True iff *author* satisfies the policy's domain regex.

        Always True when no domain regex is configured. Independent of the
        shape check, which is enforced separately.
        """
        if self.author_email_regex is None:
            return True
        return self.author_email_regex.search(author) is not None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for inclusion in reports (read-only summary)."""
        return {
            "profile": self.profile,
            "audience": self.audience,
            "digest": self.digest,
            "author_email_regex": (self.author_email_regex.pattern if self.author_email_regex else None),
            "severity_overrides": {k: v.value for k, v in self.severity_overrides.items()},
            "source": str(self.source) if self.source else None,
        }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _coerce_severity(value: Any, key: str) -> Severity | None:
    if value is None:
        return None
    try:
        return Severity(str(value).lower())
    except ValueError:
        logger.warning(
            "Ignoring invalid severity %r for override %r (expected one of %s).",
            value,
            key,
            ", ".join(s.value for s in Severity),
        )
        return None


def _coerce_email_regex(value: Any, source: str) -> re.Pattern[str] | None:
    if value is None:
        return None
    if not isinstance(value, str):
        logger.warning("Ignoring non-string author_email_regex in %s (got %r).", source, type(value).__name__)
        return None
    try:
        return re.compile(value)
    except re.error as exc:
        logger.warning("Ignoring invalid author_email_regex in %s: %s", source, exc)
        return None


_KNOWN_TOP_LEVEL_KEYS = {"profile", "identity", "severity_overrides"}
_KNOWN_IDENTITY_KEYS = {"author_email_regex"}


def _warn_unknown_keys(data: dict[str, Any], known: set[str], context: str, source: str) -> None:
    unknown = set(data) - known
    if unknown:
        logger.warning("Ignoring unknown %s key(s) %s in %s.", context, sorted(unknown), source)


def _policy_from_data(
    data: dict[str, Any],
    *,
    fallback_profile: str,
    source: Path | None,
) -> ValidationPolicy:
    source_str = str(source) if source else "<inline>"
    if not isinstance(data, dict):
        raise ValueError(f"Profile YAML at {source_str} must be a mapping at the top level")

    _warn_unknown_keys(data, _KNOWN_TOP_LEVEL_KEYS, "top-level", source_str)

    profile_name = str(data.get("profile") or fallback_profile)

    identity = data.get("identity") or {}
    if not isinstance(identity, dict):
        logger.warning("Ignoring non-mapping 'identity' block in %s.", source_str)
        identity = {}
    else:
        _warn_unknown_keys(identity, _KNOWN_IDENTITY_KEYS, "identity", source_str)

    author_email_regex = _coerce_email_regex(identity.get("author_email_regex"), source_str)

    overrides_raw = data.get("severity_overrides") or {}
    severity_overrides: dict[str, Severity] = {}
    if not isinstance(overrides_raw, dict):
        logger.warning("Ignoring non-mapping 'severity_overrides' block in %s.", source_str)
    else:
        for key, value in overrides_raw.items():
            if not isinstance(key, str) or "." not in key:
                logger.warning(
                    "Ignoring severity override %r in %s (expected 'CATEGORY.check_name').",
                    key,
                    source_str,
                )
                continue
            sev = _coerce_severity(value, key)
            if sev is not None:
                severity_overrides[key] = sev

    return ValidationPolicy(
        profile=profile_name,
        audience="external",
        author_email_regex=author_email_regex,
        severity_overrides=severity_overrides,
        source=source,
    )


def load_profile(name: str = DEFAULT_PROFILE_NAME) -> ValidationPolicy:
    """Load a bundled profile by name.

    Raises:
        FileNotFoundError: when no YAML exists for the requested profile.
        ValueError: when the YAML is malformed.
    """
    path = PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in PROFILES_DIR.glob("*.yaml")) if PROFILES_DIR.exists() else []
        raise FileNotFoundError(f"Unknown profile {name!r}. Available bundled profiles: {available or '(none)'}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return _policy_from_data(data, fallback_profile=name, source=path)


def load_policy_file(
    path: Path,
    *,
    base_profile: str = DEFAULT_PROFILE_NAME,
) -> ValidationPolicy:
    """Load a custom policy YAML, layered on top of *base_profile*.

    Raises:
        FileNotFoundError: when *path* does not exist.
        ValueError: when the YAML is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Custom policy file not found: {path}")

    base = load_profile(base_profile)
    with path.open(encoding="utf-8") as fh:
        custom_data = yaml.safe_load(fh) or {}
    custom = _policy_from_data(custom_data, fallback_profile=base.profile, source=path)

    merged_overrides = dict(base.severity_overrides)
    merged_overrides.update(custom.severity_overrides)

    # The overlay only takes precedence on author_email_regex when the *key*
    # itself is present (including an explicit ``null`` to disable the domain
    # check). A bare ``identity:`` block must NOT silently null the base regex.
    identity_block = custom_data.get("identity")
    if not isinstance(identity_block, dict):
        identity_block = {}
    overlay_sets_email_regex = "author_email_regex" in identity_block

    return ValidationPolicy(
        profile=custom.profile,
        audience=base.audience,
        author_email_regex=(custom.author_email_regex if overlay_sets_email_regex else base.author_email_regex),
        severity_overrides=merged_overrides,
        source=path,
    )


def default_policy() -> ValidationPolicy:
    """Return the default public policy."""
    try:
        return load_profile(DEFAULT_PROFILE_NAME)
    except FileNotFoundError:
        logger.warning("Default profile YAML not found; falling back to an in-memory public policy.")
        return ValidationPolicy(
            profile=DEFAULT_PROFILE_NAME,
            author_email_regex=None,
            severity_overrides={
                "SCHEMA.author_missing": Severity.HIGH,
                "SCHEMA.author_format": Severity.HIGH,
                "LICENSE.*": Severity.CRITICAL,
                "LICENSE.frontmatter_license_mismatch": Severity.CRITICAL,
            },
        )


def resolve_policy(
    *,
    profile: str | None = None,
    policy_path: Path | None = None,
) -> ValidationPolicy:
    """Resolve the active policy from CLI/env inputs.

    Precedence (highest to lowest):
    1. ``policy_path`` overlaid on top of ``profile`` (or the default profile)
    2. ``profile`` named profile
    3. ``SKILLEVALUATOR_PROFILE`` env var
    4. Default public profile
    """
    base = profile or os.environ.get(PROFILE_ENV_VAR) or DEFAULT_PROFILE_NAME
    if policy_path is not None:
        return load_policy_file(policy_path, base_profile=base)
    return load_profile(base)


# ---------------------------------------------------------------------------
# Central application
# ---------------------------------------------------------------------------


def apply_policy(
    results: Iterable[ValidationResult],
    policy: ValidationPolicy | None,
) -> list[ValidationResult]:
    """Remap finding severities per the active policy and recompute pass/fail.

    Validators emit findings with a default severity; this central pass applies
    the policy's ``severity_overrides`` keyed on ``(category, check_name)``,
    rebuilds each result's pass/fail and summary counts, and stamps the active
    profile onto ``result.metadata['policy']`` for reporters.

    Passing ``None`` leaves results untouched (used when no policy applies).
    """
    results = list(results)
    if policy is None:
        return results
    for result in results:
        changed = False
        for finding in result.findings:
            current = (
                finding.severity if isinstance(finding.severity, Severity) else Severity(str(finding.severity).lower())
            )
            new_severity = policy.severity_for(finding.category, finding.check_name, current)
            if new_severity != current:
                finding.severity = new_severity
                changed = True
        if changed:
            result.recalculate_from_findings()
        if isinstance(result.metadata, dict):
            result.metadata["policy"] = policy.to_dict()
    return results
