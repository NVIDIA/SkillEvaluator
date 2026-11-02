# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence-bound PII waivers.

A waiver accepts one reviewed PII value at one place: a PII category, a file,
a line, and a SHA-256 digest of the value (``evidence_sha256`` in the PII
finding's metadata), with a reason, an owner, and a review window. It stops
applying as soon as any of those no longer match, so it never covers a new
value, a moved line, or a different file.

The waiver file is passed by whoever runs the scan (``--pii-waivers PATH``); a
scanned skill cannot supply its own. Its ``file`` entries are relative to the
directory that holds the waiver file, so one file serves single-skill, folder,
and catalog runs alike; each run reports only the waivers whose file lies in a
skill or folder it scanned. A file that does not load applies no waivers and
fails the PII scan with a HIGH finding.

Credential checks (:data:`~skillevaluator.utils.secret_redaction.SENSITIVE_PII_CATEGORIES`)
cannot be waived. Reports mask their values, so their findings carry no digest:
a digest of a short password would let anyone holding the report test guesses
for it offline. A credential in a skill should be removed or rotated.

Dates are UTC calendar dates. ``created`` may be one day ahead of the UTC date,
because a reviewer east of UTC can already be on the next day.

Example::

    version: 1
    waivers:
      - id: PII-001
        category: emails
        file: skills/support/SKILL.md
        line: 12
        evidence_sha256: <64 lowercase hex characters from the PII finding>
        reason: Public support address of the documented service
        owner: docs-team
        created: 2026-10-01
        expires: 2027-01-01
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path, PurePosixPath

import yaml

from skillevaluator.models.result import Finding, Severity
from skillevaluator.quality_config import _UniqueKeySafeLoader, _yaml_error_detail
from skillevaluator.utils.secret_redaction import SENSITIVE_PII_CATEGORIES

PII_WAIVER_CATEGORY = "PII_WAIVER"
_WAIVER_FIELDS = ("id", "category", "file", "line", "evidence_sha256", "reason", "owner", "created", "expires")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MIN_REASON_LENGTH = 8
# The local date can be at most one day ahead of the UTC date (UTC+14 at most).
_CREATED_TOLERANCE = timedelta(days=1)


class PIIWaiverError(ValueError):
    """Raised when a PII waiver file cannot be applied."""


def pii_evidence_sha256(category: str, value: str) -> str:
    """Return the digest that identifies a PII value in a waiver without storing the value.

    The value is case-folded because PII findings group values case-insensitively.
    A digest is an identifier, not a secret: a short value such as an SSN can be
    recovered from it by trying every candidate. Findings therefore report it only
    for values they show unmasked.
    """
    return hashlib.sha256(f"{category}\x00{value.casefold()}".encode()).hexdigest()


def configured_pii_categories(pii_patterns: Mapping[str, object]) -> frozenset[str]:
    """Return the PII check names in the PII pattern config; the loader also refuses credential checks."""
    return frozenset(
        name for name, patterns in pii_patterns.items() if name != "exceptions" and isinstance(patterns, list)
    )


def today_utc() -> date:
    """Return today's date in UTC, the calendar waiver review windows use."""
    return datetime.now(UTC).date()


@dataclass(frozen=True)
class PIIWaiver:
    """One reviewed PII value at one file and line."""

    waiver_id: str
    category: str
    file: str
    line: int
    evidence_sha256: str
    reason: str
    owner: str
    created: date
    expires: date


def _text_field(item: dict, name: str, location: str) -> str:
    value = item[name]
    if not isinstance(value, str) or not value.strip():
        raise PIIWaiverError(f"{location}.{name} must be a non-empty string")
    return value.strip()


def _date_field(item: dict, name: str, location: str) -> date:
    value = item[name]
    if isinstance(value, datetime) or not isinstance(value, date | str):
        raise PIIWaiverError(f"{location}.{name} must be a date (YYYY-MM-DD)")
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise PIIWaiverError(f"{location}.{name} must be a date (YYYY-MM-DD)") from exc
    return value


def _file_field(item: dict, location: str) -> str:
    value = item["file"]
    if not isinstance(value, str) or not value or "\\" in value:
        raise PIIWaiverError(f"{location}.file must be a relative path with forward slashes")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or value == "."
        or ".." in path.parts
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise PIIWaiverError(f"{location}.file must be a normalized relative path inside the waiver file's directory")
    return value


def load_pii_waivers(path: Path, categories: Collection[str], *, today: date) -> list[PIIWaiver]:
    """Load a version-1 PII waiver file, rejecting any entry that is not exactly valid.

    Expired waivers load (they are reported, not applied); a waiver created more
    than a day after ``today`` (UTC) does not. Messages name entries by position
    and id, never the file's location, so they are safe to put in reports.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PIIWaiverError("file is not valid UTF-8") from exc
    except OSError as exc:
        raise PIIWaiverError(f"cannot read file: {exc.strerror or type(exc).__name__}") from exc
    try:
        raw = yaml.load(text, Loader=_UniqueKeySafeLoader)  # nosec B506
    except yaml.YAMLError as exc:
        raise PIIWaiverError(f"invalid YAML: {_yaml_error_detail(exc)}") from exc
    except RecursionError as exc:
        raise PIIWaiverError("invalid YAML: nesting is too deep") from exc
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise PIIWaiverError(f"invalid YAML: cannot parse a value: {exc}") from exc

    if not isinstance(raw, dict) or set(raw) != {"version", "waivers"}:
        raise PIIWaiverError("the top level must contain only 'version: 1' and a 'waivers' list")
    if type(raw["version"]) is not int or raw["version"] != 1:
        raise PIIWaiverError("version must be 1")
    if not isinstance(raw["waivers"], list):
        raise PIIWaiverError("waivers must be a list")

    waivers: list[PIIWaiver] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw["waivers"]):
        location = f"waivers[{index}]"
        if not isinstance(item, dict):
            raise PIIWaiverError(f"{location} must be a mapping")
        unknown = sorted(str(key) for key in item if key not in _WAIVER_FIELDS)
        if unknown:
            raise PIIWaiverError(f"{location} has unknown field(s): {', '.join(unknown)}")
        missing = [name for name in _WAIVER_FIELDS if name not in item]
        if missing:
            raise PIIWaiverError(f"{location} is missing: {', '.join(missing)}")

        waiver_id = _text_field(item, "id", location)
        if waiver_id in seen_ids:
            raise PIIWaiverError(f"{location}.id {waiver_id!r} is used more than once")
        seen_ids.add(waiver_id)
        location = f"waiver {waiver_id!r}"

        category = item["category"]
        if isinstance(category, str) and category in SENSITIVE_PII_CATEGORIES:
            raise PIIWaiverError(
                f"{location}.category {category!r} is a credential check and cannot be waived; "
                "remove or rotate the credential"
            )
        if not isinstance(category, str) or category not in categories:
            raise PIIWaiverError(f"{location}.category must be a PII check name such as 'emails'")
        line = item["line"]
        if type(line) is not int or line < 1:
            raise PIIWaiverError(f"{location}.line must be a positive integer")
        digest = item["evidence_sha256"]
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest.lower()):
            raise PIIWaiverError(f"{location}.evidence_sha256 must be 64 hexadecimal characters")
        reason = _text_field(item, "reason", location)
        if len(reason) < _MIN_REASON_LENGTH:
            raise PIIWaiverError(
                f"{location}.reason must explain the review (at least {_MIN_REASON_LENGTH} characters)"
            )
        created = _date_field(item, "created", location)
        expires = _date_field(item, "expires", location)
        if created > today + _CREATED_TOLERANCE:
            raise PIIWaiverError(f"{location}.created is in the future")
        if expires < created:
            raise PIIWaiverError(f"{location}.expires is before created")

        waivers.append(
            PIIWaiver(
                waiver_id=waiver_id,
                category=category,
                file=_file_field(item, location),
                line=line,
                evidence_sha256=digest.lower(),
                reason=reason,
                owner=_text_field(item, "owner", location),
                created=created,
                expires=expires,
            )
        )
    return waivers


@dataclass
class PIIWaiverSet:
    """The waivers of one scan, which of them matched evidence, and what the scan covered."""

    path: Path
    waivers: list[PIIWaiver] = field(default_factory=list)
    error: str | None = None
    today: date = field(default_factory=today_utc)
    applied: set[str] = field(default_factory=set)
    # Each directory or file the PII scan covered, as passed and resolved.
    covered: list[tuple[Path, Path]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path, categories: Collection[str], *, today: date | None = None) -> PIIWaiverSet:
        """Load ``path``; a file that does not load is kept as an error and applies no waivers."""
        waiver_set = cls(path=path, today=today or today_utc())
        try:
            waiver_set.waivers = load_pii_waivers(path, categories, today=waiver_set.today)
        except PIIWaiverError as exc:
            waiver_set.error = str(exc)
        return waiver_set

    def _base(self) -> Path:
        # The directory the waiver file is in, even when the file itself is a symlink.
        return self.path.absolute().parent.resolve()

    def _locate(self, relative: str) -> Path | None:
        located = self._base() / relative
        try:
            return located.parent.resolve() / located.name
        except (OSError, RuntimeError, ValueError):
            return None

    def cover(self, unit: Path) -> None:
        """Record that the PII scan covered ``unit``: one skill, a folder scanned directly, or a file.

        Only waivers whose file lies in a covered unit are reported, so a run on
        one skill, or one child of a catalog run, does not report the waivers of
        skills it never scanned as stale.
        """
        try:
            self.covered.append((unit, unit.absolute().resolve()))
        except (OSError, RuntimeError):
            return

    def _report_path(self, waiver: PIIWaiver, target: Path) -> str | None:
        """Return the waiver's file as the PII findings of a scan of ``target`` name it.

        That is relative to the covered unit, with the ``[skill] `` prefix of a
        folder run, or ``None`` when no covered unit holds the file.
        """
        located = self._locate(waiver.file)
        if located is None:
            return None
        units = [(unit, resolved) for unit, resolved in self.covered if located.is_relative_to(resolved)]
        if not units:
            return None
        unit, resolved = max(units, key=lambda item: len(item[1].parts))
        relative = str(located.relative_to(resolved))
        try:
            is_target = resolved == target.absolute().resolve()
        except (OSError, RuntimeError):
            is_target = False
        return relative if is_target else f"[{unit.name}] {relative}"

    def matcher(self, file_path: Path) -> Callable[[str, int, str], bool] | None:
        """Return a check for PII candidates in ``file_path``, or ``None`` when no waiver names the file.

        The check takes a candidate's category, line, and value, records the
        waiver that covers it, and returns whether one does.
        """
        try:
            located = file_path.absolute().parent.resolve() / file_path.name
            relative = located.relative_to(self._base()).as_posix()
        except (OSError, RuntimeError, ValueError):
            return None
        # ``created`` was checked when the file loaded; only the expiry date can end a waiver.
        candidates = [waiver for waiver in self.waivers if waiver.file == relative and self.today <= waiver.expires]
        if not candidates:
            return None

        def waive(category: str, line: int, value: str) -> bool:
            digest = pii_evidence_sha256(category, value)
            for waiver in candidates:
                if waiver.category == category and waiver.line == line and waiver.evidence_sha256 == digest:
                    self.applied.add(waiver.waiver_id)
                    return True
            return False

        return waive

    def findings(self, target: Path) -> list[tuple[Finding, bool]]:
        """Report the waiver file's outcome for a scan of ``target``.

        An invalid file is one HIGH finding. Otherwise each waiver whose file the
        scan covered is reported as applied, expired, or stale, at the path the
        PII findings of the same report use; ``metadata["waiver_file"]`` keeps
        the waiver's own ``file``. Waivers for files outside the scan are not
        judged. Returns ``(finding, is_error)`` pairs; only an invalid file is
        an error.
        """
        if self.error is not None:
            return [
                (
                    Finding(
                        category=PII_WAIVER_CATEGORY,
                        severity=Severity.HIGH,
                        check_name="invalid-pii-waiver-file",
                        message=f"PII waiver file {self.path.name} is invalid, so no waiver was applied: {self.error}",
                        file_path=self.path.name,
                        suggestion="Fix the waiver file; see the PII waiver format in the Tier 1 documentation.",
                    ),
                    True,
                )
            ]

        reported: list[tuple[Finding, bool]] = []
        for waiver in self.waivers:
            applied = waiver.waiver_id in self.applied
            file_path = self._report_path(waiver, target)
            if file_path is None:
                if not applied:
                    continue
                file_path = waiver.file
            metadata = {
                "waiver_id": waiver.waiver_id,
                "waiver_file": waiver.file,
                "pii_category": waiver.category,
                "evidence_sha256": waiver.evidence_sha256,
                "owner": waiver.owner,
                "reason": waiver.reason,
                "created": waiver.created.isoformat(),
                "expires": waiver.expires.isoformat(),
            }
            if applied:
                finding = Finding(
                    category=PII_WAIVER_CATEGORY,
                    severity=Severity.INFO,
                    check_name="applied-pii-waiver",
                    message=f"PII waiver {waiver.waiver_id} accepted the reviewed value for the {waiver.category} check",
                    file_path=file_path,
                    line_number=waiver.line,
                    suggestion=(
                        f"Review again before {waiver.expires.isoformat()}; remove the waiver once the value is gone."
                    ),
                    metadata=metadata,
                )
            elif waiver.expires < self.today:
                finding = Finding(
                    category=PII_WAIVER_CATEGORY,
                    severity=Severity.MEDIUM,
                    check_name="expired-pii-waiver",
                    message=f"PII waiver {waiver.waiver_id} expired on {waiver.expires.isoformat()} and was not applied",
                    file_path=file_path,
                    line_number=waiver.line,
                    suggestion="Review the value again and extend the expiry date, or remove the waiver.",
                    metadata=metadata,
                )
            else:
                finding = Finding(
                    category=PII_WAIVER_CATEGORY,
                    severity=Severity.LOW,
                    check_name="stale-pii-waiver",
                    message=f"PII waiver {waiver.waiver_id} matched no {waiver.category} finding at its file and line",
                    file_path=file_path,
                    line_number=waiver.line,
                    suggestion="Remove the waiver, or update it from the current PII finding after review.",
                    metadata=metadata,
                )
            reported.append((finding, False))
        return reported
