# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, attributable original/delivered files; never semantic acceptance."""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from skillevaluator.tier3.eval_core.llm_judge import _redact_configured_credentials
from skillevaluator.tier3.output_provenance import write_output_file_atomically
from skillevaluator.utils.redaction import redact_sensitive_text
from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot

MAX_PAIRS = 8
MAX_FILE_BYTES = 128 * 1024
MAX_TEXT_CHARS = 4096
MAX_RECORD_BYTES = 512 * 1024
PAIR_NAMESPACE = "skillevaluator-evidence"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_LIMITS = {"file_bytes": MAX_FILE_BYTES, "text_chars": MAX_TEXT_CHARS, "max_pairs": MAX_PAIRS}
_INTERPRETATION = "Raw byte equality for declared files only, not whole-project or semantic preservation. Redacted or truncated text is incomplete."
_UNAVAILABLE_PAIR_STATUSES = {"ambiguous_capture", "delivered_not_collected", "collection_failed",
    "original_changed_after_staging", "non_utf8_or_binary", "too_large", "unsafe_or_missing_file"}


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _path(value: Any, *, delivered: bool = False) -> str:
    if not isinstance(value, str) or not value or len(value) > 512 or any(ord(c) < 32 for c in value):
        raise ValueError("artifact pair paths must be bounded strings")
    path = PurePosixPath(value)
    if "\\" in value or any(p in {".", "..", ""} for p in value.lstrip("/").split("/")):
        raise ValueError("artifact pair paths must be canonical and contain no traversal")
    if str(path) != value or (delivered and not value.startswith(("/workspace/output/", "/workspace/work/"))):
        raise ValueError("delivered artifact paths must be below /workspace/output or /workspace/work")
    if not delivered and path.is_absolute():
        raise ValueError("original artifact paths must be relative to staged input")
    return value


def validate_artifact_pairs(entry: dict) -> list[dict[str, str]]:
    pairs = entry.get("artifact_pairs", [])
    if not isinstance(pairs, list) or len(pairs) > MAX_PAIRS:
        raise ValueError(f"artifact_pairs must be a list of at most {MAX_PAIRS} file pairs")
    ids, originals, delivered_paths = set(), set(), set()
    result = []
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {"id", "original", "delivered"}:
            raise ValueError("each artifact pair must contain exactly id, original, delivered")
        if not isinstance(pair["id"], str) or not _ID.fullmatch(pair["id"]):
            raise ValueError("artifact pair id must be a portable identifier of at most 64 characters")
        original, delivered = _path(pair["original"]), _path(pair["delivered"], delivered=True)
        if pair["id"].casefold() in ids or original.casefold() in originals or delivered.casefold() in delivered_paths:
            raise ValueError("artifact pair declarations must be unambiguous")
        for prior in delivered_paths:
            if delivered.casefold().startswith(prior + "/") or prior.startswith(delivered.casefold() + "/"):
                raise ValueError("artifact file destinations cannot overlap")
        ids.add(pair["id"].casefold())
        originals.add(original.casefold())
        delivered_paths.add(delivered.casefold())
        result.append({"id": pair["id"], "original": original, "delivered": delivered})
    return result


def _read(root: Path, relative: str, limit: int = MAX_FILE_BYTES) -> bytes:
    with SecureRoot(root) as secure:
        return secure.read_bytes(Path(relative), limit)[0]


def stage_artifact_pairs(task_dir: Path, entry: dict) -> None:
    pairs = validate_artifact_pairs(entry)
    if not pairs:
        return
    frozen = []
    for pair in pairs:
        try:
            raw = _read(task_dir, "environment/input/" + pair["original"])
        except (OSError, ValueError) as exc:
            raise ValueError(f"artifact pair {pair['id']!r}: original is missing, unsafe or exceeds the size limit") from exc
        frozen.append({**pair, "original_sha256": _digest(raw), "original_size_bytes": len(raw)})
    record = {"schema_version": 1, "entry_id": entry["id"], "pairs": frozen,
              "declarations_sha256": _digest(_json_bytes(pairs))}
    (task_dir / "artifact_pairs.json").write_bytes(_json_bytes(record))


def _redact(text: str) -> str:
    # Whole strings are redacted before any clipping, including configured keys.
    return _redact_configured_credentials(redact_sensitive_text(text))


def _text_record(raw: bytes, source: str) -> tuple[dict, str]:
    text = raw.decode("utf-8", errors="strict")
    if "\x00" in text:
        raise UnicodeError("binary file")
    safe = _redact(text)
    clipped = safe[:MAX_TEXT_CHARS]
    return {"source": source, "sha256": _digest(raw), "size_bytes": len(raw),
            "text": clipped, "text_sha256": _digest(clipped.encode()),
            "redacted": safe != text, "truncated": len(safe) > len(clipped)}, safe


def collect_artifact_evidence(task_dir: Path, trial_src: Path, trial_out: Path, *,
                              entry_id: str, trial_id: str, agent: str, variant: str) -> None:
    """Read only the exact staged contract and native pre-verifier capture."""
    try:
        contract_raw = _read(task_dir, "artifact_pairs.json", MAX_RECORD_BYTES)
    except (OSError, ValueError):
        return
    record: dict[str, Any] = {"schema_version": 1, "entry_id": entry_id, "trial_id": trial_id,
                              "agent": agent, "variant": variant, "status": "unavailable", "pairs": []}
    try:
        contract = json.loads(contract_raw)
        if not isinstance(contract, dict) or contract.get("schema_version") != 1:
            raise ValueError("invalid contract version")
        pairs = validate_artifact_pairs({"artifact_pairs": [
            {k: p[k] for k in ("id", "original", "delivered")} for p in contract["pairs"]]})
        declarations_sha = _digest(_json_bytes(pairs))
        if contract.get("entry_id") != entry_id or contract.get("declarations_sha256") != declarations_sha:
            raise ValueError("staged identity mismatch")
        record.update(task_contract_sha256=_digest(contract_raw), declarations_sha256=declarations_sha,
                      capture="secure-docker-file-before-verifier", task_selector=task_dir.name)
        try:
            manifest_raw = _read(trial_src, "artifacts/manifest.json", MAX_RECORD_BYTES)
            manifest = json.loads(manifest_raw)
            if not isinstance(manifest, list):
                raise ValueError("invalid manifest")
            record["native_manifest_sha256"] = _digest(manifest_raw)
        except (OSError, ValueError):
            manifest = []
        for pair, frozen in zip(pairs, contract["pairs"], strict=True):
            item: dict[str, Any] = {"id": pair["id"], "status": "unavailable"}
            record["pairs"].append(item)
            destination = f"artifacts/{PAIR_NAMESPACE}/{pair['id']}/delivered"
            matching = [m for m in manifest if isinstance(m, dict) and
                        (m.get("destination") == destination or m.get("source") == pair["delivered"])]
            if len(matching) != 1:
                item["status"] = "ambiguous_capture" if matching else "delivered_not_collected"
                continue
            capture = matching[0]
            if (capture.get("destination") != destination or capture.get("source") != pair["delivered"]
                    or capture.get("service") is not None or capture.get("exclude", []) != []
                    or capture.get("type") != "file" or capture.get("status") != "ok"):
                item["status"] = "collection_failed"
                continue
            try:
                original = _read(task_dir, "environment/input/" + pair["original"])
                if (_digest(original) != frozen["original_sha256"]
                        or len(original) != frozen["original_size_bytes"]):
                    item["status"] = "original_changed_after_staging"
                    continue
                delivered = _read(trial_src, destination)
                original_info, original_text = _text_record(original, "environment/input/" + pair["original"])
                delivered_info, delivered_text = _text_record(delivered, destination)
                # Diff complete redacted strings; cap the emitted diff independently.
                diff = "".join(difflib.unified_diff(original_text.splitlines(keepends=True),
                              delivered_text.splitlines(keepends=True), fromfile="original", tofile="delivered"))
                item.update(status="complete", original=original_info, delivered=delivered_info,
                            bytes_equal=original == delivered, diff=diff[:MAX_TEXT_CHARS],
                            diff_truncated=len(diff) > MAX_TEXT_CHARS)
            except UnicodeError:
                item["status"] = "non_utf8_or_binary"
            except SecurePathError as exc:
                item["status"] = "too_large" if exc.code == "file_size_limit" else "unsafe_or_missing_file"
            except (OSError, ValueError, KeyError):
                item["status"] = "unsafe_or_missing_file"
        record["status"] = "available" if all(p["status"] == "complete" for p in record["pairs"]) else "partial"
        record["limits"] = _LIMITS
        record["interpretation"] = _INTERPRETATION
    except (ValueError, KeyError, TypeError):
        record["reason"] = "invalid_staged_contract"
    write_output_file_atomically(trial_out / "artifact_evidence.json", _json_bytes(record))


def load_artifact_evidence(run_dir: Path, agent: str, trial_id: str, *, entry_id: str,
                           declarations: list | None) -> dict:
    """Shared improvement/refinement reader; callers exclude reserved splits first."""
    unavailable = {"status": "unavailable", "reason": "retained_pair_identity_or_content_unavailable"}
    if not declarations:
        return {"status": "not_declared", "pairs": []}
    if any(not isinstance(s, str) or not s or s in {".", ".."} or Path(s).name != s
           for s in (agent, trial_id)):
        return unavailable
    try:
        pairs = validate_artifact_pairs({"artifact_pairs": declarations})
        relative = f"{agent}/with-skill/trials/{trial_id}/artifact_evidence.json"
        raw = _read(run_dir, relative, MAX_RECORD_BYTES)
        record = json.loads(raw)
        if not isinstance(record, dict) or any(record.get(k) != v for k, v in {
            "schema_version": 1, "agent": agent, "trial_id": trial_id, "entry_id": entry_id,
            "variant": "with_skill", "declarations_sha256": _digest(_json_bytes(pairs)),
            "capture": "secure-docker-file-before-verifier",
        }.items()):
            return unavailable
        allowed = {"schema_version", "agent", "trial_id", "entry_id", "variant", "status", "pairs",
            "declarations_sha256", "task_contract_sha256", "native_manifest_sha256", "capture", "task_selector",
            "limits", "interpretation"}
        if (set(record) - allowed or record.get("limits") != _LIMITS
                or record.get("interpretation") != _INTERPRETATION
                or not isinstance(record.get("task_selector"), str) or not 0 < len(record["task_selector"]) <= 512
                or any(_redact(record[key]) != record[key] for key in ("agent", "trial_id", "entry_id", "task_selector"))):
            return unavailable
        if not isinstance(record.get("pairs"), list) or len(record["pairs"]) != len(pairs):
            return unavailable
        complete = all(isinstance(p, dict) and p.get("status") == "complete" for p in record["pairs"])
        if record.get("status") != ("available" if complete else "partial"):
            return unavailable
        hashes = [record.get("task_contract_sha256")]
        if "native_manifest_sha256" in record or any(
            isinstance(p, dict) and p.get("status") == "complete" for p in record["pairs"]
        ):
            hashes.append(record.get("native_manifest_sha256"))
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes):
            return unavailable
        for item, declared in zip(record["pairs"], pairs, strict=True):
            if not isinstance(item, dict) or item.get("id") != declared["id"] or _redact(declared["id"]) != declared["id"]:
                return unavailable
            if item.get("status") != "complete":
                if set(item) != {"id", "status"} or item.get("status") not in _UNAVAILABLE_PAIR_STATUSES:
                    return unavailable
                continue
            if set(item) != {"id", "status", "original", "delivered", "bytes_equal", "diff", "diff_truncated"}:
                return unavailable
            if item.get("status") == "complete":
                if (type(item["bytes_equal"]) is not bool or not isinstance(item["diff"], str)
                        or len(item["diff"]) > MAX_TEXT_CHARS or type(item["diff_truncated"]) is not bool):
                    return unavailable
                for side, source in (("original", "environment/input/" + declared["original"]),
                                     ("delivered", f"artifacts/{PAIR_NAMESPACE}/{declared['id']}/delivered")):
                    value = item[side]
                    if (not isinstance(value, dict)
                            or set(value) != {"source", "sha256", "size_bytes", "text", "text_sha256", "redacted", "truncated"}
                            or value["source"] != source or _redact(source) != source or not isinstance(value["text"], str)
                            or len(value["text"]) > MAX_TEXT_CHARS
                            or not isinstance(value["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
                            or type(value["size_bytes"]) is not int or not 0 <= value["size_bytes"] <= MAX_FILE_BYTES
                            or any(type(value[k]) is not bool for k in ("redacted", "truncated"))
                            or value["text_sha256"] != _digest(value["text"].encode())):
                        return unavailable
                    if not value["redacted"] and not value["truncated"]:
                        raw_text = value["text"].encode()
                        if _digest(raw_text) != value["sha256"] or len(raw_text) != value["size_bytes"]:
                            return unavailable
                    safe = _redact(value["text"])
                    value.update(redacted=value["redacted"] or safe != value["text"],
                                 truncated=value["truncated"] or len(safe) > MAX_TEXT_CHARS,
                                 text=safe[:MAX_TEXT_CHARS], text_sha256=_digest(safe[:MAX_TEXT_CHARS].encode()))
                if (item["bytes_equal"] != (item["original"]["sha256"] == item["delivered"]["sha256"])
                        or (item["bytes_equal"] and item["original"]["size_bytes"] != item["delivered"]["size_bytes"])):
                    return unavailable
                safe_diff = _redact(item["diff"])
                item.update(diff=safe_diff[:MAX_TEXT_CHARS],
                            diff_truncated=item["diff_truncated"] or len(safe_diff) > MAX_TEXT_CHARS)
        return {**record, "source": relative, "sha256": _digest(raw)}
    except (OSError, ValueError, TypeError, KeyError, UnicodeError):
        return unavailable


# Executed only by the reserved secure-Docker download route. Every component is
# opened relative to a pinned directory descriptor; nonblocking avoids FIFO hangs.
CONTAINER_READ_SCRIPT = r"""
import base64, os, stat, sys
p = sys.argv[1]
if not p.startswith(('/workspace/output/', '/workspace/work/')) or any(x in ('', '.', '..') for x in p[1:].split('/')):
    raise SystemExit(2)
fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
try:
    parts = p[1:].split('/')
    for part in parts[:-1]:
        child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
        os.close(fd)
        fd = child
    leaf = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    try:
        before = os.fstat(leaf)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > 131072:
            raise SystemExit(3)
        data = bytearray()
        while len(data) <= 131072:
            chunk = os.read(leaf, min(65536, 131073 - len(data)))
            if not chunk: break
            data.extend(chunk)
        after = os.fstat(leaf)
        if len(data) > 131072 or len(data) != before.st_size or any(
            getattr(before, k) != getattr(after, k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_nlink')):
            raise SystemExit(4)
        sys.stdout.write(base64.b64encode(data).decode('ascii'))
    finally: os.close(leaf)
finally: os.close(fd)
"""
