# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import production trace exports as redacted, explicitly reviewed candidates."""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path, PureWindowsPath
from typing import Any

from skillevaluator.config import load_pii_patterns
from skillevaluator.tier3.dataset_utils import find_eval_file
from skillevaluator.tier3.eval_core import log_converters
from skillevaluator.tier3.eval_core.codex_tool_call_normalizer import atif_content_text
from skillevaluator.tier3.evals_spec import _validate_dataset_contents, validate_tier3_source
from skillevaluator.tier3.generate_dataset import _parse_skill
from skillevaluator.utils.redaction import redact_sensitive_data, redact_sensitive_text

_CONVERTERS = {
    "claude-code": log_converters.synthetic_trajectory_from_claude_stream_jsonl,
    "cursor-cli": log_converters.synthetic_trajectory_from_cursor_cli,
    "cline": log_converters.synthetic_trajectory_from_cline_cli,
    "opencode": log_converters.synthetic_trajectory_from_opencode_json,
    "codex": log_converters.synthetic_trajectory_from_codex_json,
}
_FIELDS = {
    "session_id",
    "prompt",
    "outcome",
    "skill",
    "trajectory",
    "agent",
    "agent_log",
    "final_output",
    "files",
    "source",
    "started_at",
    "tags",
}
_SCIENTIFIC_LITERAL_PATTERN = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)[eE][-+]?\d+")
_SIGNED_DECIMAL_PATTERN = re.compile(r"(?<![\w.])[+-]\d+\.\d+(?![\w.])")
_HHMM_OFFSET_PATTERN = re.compile(r"\b\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?P<offset>[+-]\d{4})\b")


@lru_cache(maxsize=1)
def _pii_patterns() -> tuple[tuple[str, re.Pattern[str], tuple[str, ...]], ...]:
    configured = load_pii_patterns()
    patterns = [
        (category, re.compile(spec["pattern"]), tuple(spec.get("exceptions", [])))
        for category in ("personal_paths", "home_paths", "emails", "phone_numbers", "ssn")
        for spec in configured.get(category, [])
    ]
    patterns.extend(
        (
            ("home_paths", re.compile(r"/home/(?!<user>)[^/\s\"'<>]+/?"), ()),
            ("personal_paths", re.compile(r"/Users/(?!(?:<user>|\{[^/{}]+\}))[^/\s\"'<>]+/?"), ()),
        )
    )
    return tuple(patterns)


def _redaction_markers(value: Any) -> int:
    if isinstance(value, str):
        return value.count("<redacted>")
    if isinstance(value, list):
        return sum(_redaction_markers(item) for item in value)
    if isinstance(value, dict):
        return sum(_redaction_markers(key) + _redaction_markers(item) for key, item in value.items())
    return 0


def redact(value: Any, *, structured: bool = False) -> tuple[Any, dict[str, int]]:
    """Redact string leaves, counting introduced secret markers and PII matches."""
    counts = {"secrets": 0, "pii": 0}
    patterns = _pii_patterns()
    if structured:
        sanitized = redact_sensitive_data(value)
        counts["secrets"] += max(0, _redaction_markers(sanitized) - _redaction_markers(value))
        value = sanitized

    def clean(item: Any) -> Any:
        if isinstance(item, list):
            return [clean(v) for v in item]
        if isinstance(item, dict):
            cleaned = {}
            for key, value in item.items():
                cleaned_key = clean(key)
                if cleaned_key in cleaned:
                    base = str(cleaned_key)
                    suffix = 2
                    while f"{base}-{suffix}" in cleaned:
                        suffix += 1
                    cleaned_key = f"{base}-{suffix}"
                cleaned[cleaned_key] = clean(value)
            return cleaned
        if not isinstance(item, str):
            return item
        text = redact_sensitive_text(item)
        counts["secrets"] += max(0, text.count("<redacted>") - item.count("<redacted>"))
        for category, pattern, exceptions in patterns:

            def replace(
                match: re.Match[str],
                category: str = category,
                exceptions: tuple[str, ...] = exceptions,
                text: str = text,
            ) -> str:
                found = match.group()
                if category == "home_paths":
                    replacement = "/home/<user>" + ("/" if found.endswith("/") else "")
                elif category == "personal_paths":
                    replacement = (
                        "/Users/<user>" + ("/" if found.endswith("/") else "")
                        if found.startswith("/")
                        else "C:\\Users\\<user>"
                    )
                else:
                    replacement = "[redacted-pii]"
                if found == replacement or any(e in found for e in exceptions):
                    return found
                if category == "phone_numbers":
                    if "\n" in found or "\r" in found:
                        return found
                    window = (max(0, match.start() - 64), match.start() + 64)
                    if any(
                        literal.start() <= match.start() < literal.end()
                        for literal in _SCIENTIFIC_LITERAL_PATTERN.finditer(text, *window)
                    ):
                        return found
                    if any(
                        number.start() <= match.start() and match.end() <= number.end()
                        for number in _SIGNED_DECIMAL_PATTERN.finditer(text)
                    ):
                        return found
                    if any(
                        offset.start("offset") <= match.start() and match.end() <= offset.end("offset")
                        for offset in _HHMM_OFFSET_PATTERN.finditer(text)
                    ):
                        return found
                counts["pii"] += 1
                return replacement

            text = pattern.sub(replace, text)
        return text

    return clean(value), counts


def parse_export(raw: bytes) -> list[tuple[int, Any]]:
    """Parse a v1 wrapper or JSONL; malformed lines become unsupported records."""
    text = raw.decode("utf-8-sig")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict) and "session_id" not in obj and ("format" in obj or "sessions" in obj):
        if (
            obj.get("format") != "skillevaluator.trace-export"
            or type(obj.get("version")) is not int
            or obj["version"] != 1
        ):
            raise ValueError("Expected skillevaluator.trace-export version 1")
        if not isinstance(obj.get("sessions"), list):
            raise ValueError("Export sessions must be a list")
        return list(enumerate(obj["sessions"], 1))
    records = []
    for number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            records.append((number, json.loads(line)))
        except json.JSONDecodeError:
            records.append((number, None))
    return records


def safe_relative(value: Any) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError("Fixture path must be a relative POSIX file path")
    path = Path(value)
    if path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts or not path.parts:
        raise ValueError("Fixture path must not be absolute or traverse parents")
    return path


def contained(root: Path, relative: str) -> Path:
    path = root / safe_relative(relative)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Fixture path resolves outside its directory")
    return path


def _missing_replay_path(call: dict, fixtures: dict[str, str], skill_name: str, skill_path: Path | None) -> str | None:
    if str(call.get("function_name", "")).casefold() not in ("read", "read_file", "open", "open_file"):
        return None
    args = call.get("arguments", {})
    if not isinstance(args, dict):
        return None
    path = args.get("path", args.get("file_path"))
    if not isinstance(path, str) or not path:
        return None
    relative = Path(path).as_posix()
    if relative in fixtures:
        return None
    target = Path(path)
    if target.is_absolute():
        if (
            skill_path is not None and target.resolve().is_relative_to(skill_path.resolve())
        ) or _is_installed_skill_path(target.parts, skill_name):
            return None
    else:
        parts = target.parts
        if parts and parts[0] == skill_name:
            parts = parts[1:]
        if ".." not in parts and parts and parts[0] in ("SKILL.md", "scripts", "references", "assets", "tools"):
            return None
        if (
            skill_path is not None
            and (skill_path / target).is_file()
            and (skill_path / target).resolve().is_relative_to(skill_path.resolve())
        ):
            return None
    return path


def _is_installed_skill_path(parts: tuple[str, ...], skill_name: str) -> bool:
    normalized = [part.casefold() for part in parts]
    skill = skill_name.casefold()
    allowed = {"skill.md", "scripts", "references", "assets", "tools"}
    return any(
        normalized[index] == "skills"
        and normalized[index + 1] == skill
        and index + 2 < len(parts)
        and normalized[index + 2] in allowed
        for index in range(len(parts) - 2)
    )


def build_candidates(
    raw: bytes,
    export_name: str,
    skill_name: str,
    *,
    skill_path: Path | None = None,
    skill_directory_name: str | None = None,
) -> tuple[dict, dict[str, str]]:
    """Return only redacted review metadata and fixture contents, without writing."""
    candidates: list[dict] = []
    skipped: list[dict] = []
    fixtures: dict[str, str] = {}
    fingerprints: dict[str, str] = {}
    ids: set[str] = set()
    for record, session in parse_export(raw):
        sid = session.get("session_id") if isinstance(session, dict) else None

        def skip(reason: str, sid: Any = sid, record: int = record, **extra: Any) -> None:
            skipped.append(
                {"session_id": sid if isinstance(sid, str) else None, "record": record, "reason": reason, **extra}
            )

        if not isinstance(session, dict):
            skip("unsupported")
            continue
        try:
            json.dumps(session, ensure_ascii=False).encode("utf-8")
        except UnicodeEncodeError:
            skip("unsupported")
            continue
        accepted_skill_names = (skill_name, skill_directory_name)
        if "skill" in session and session["skill"] not in accepted_skill_names:
            skip("other skill")
            continue
        outcome = session.get("outcome", "unknown")
        if outcome not in ("success", "failure", "unknown"):
            skip("unsupported")
            continue
        warnings = [f"unsupported field: {k}" for k in session if k not in _FIELDS]
        trajectory = session.get("trajectory")
        source = "trajectory" if isinstance(trajectory, dict) else None
        if source is None and "agent_log" in session:
            agent = session.get("agent")
            converter = _CONVERTERS.get(agent) if isinstance(agent, str) else None
            if converter and isinstance(session["agent_log"], str):
                try:
                    trajectory = converter(session["agent_log"])
                    if not trajectory and agent == "codex":
                        trajectory = log_converters.synthetic_trajectory_from_codex_txt(session["agent_log"])
                except (ValueError, TypeError, AttributeError):
                    trajectory = None
                source = f"{agent} log" if trajectory else None
            if source is None:
                warnings.append("unsupported agent log format")
        steps = trajectory.get("steps", []) if isinstance(trajectory, dict) else []
        steps = [s for s in steps if isinstance(s, dict)] if isinstance(steps, list) else []
        prompt = session.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            prompt = next((atif_content_text(s.get("message")) for s in steps if s.get("source") == "user"), "")
        if not isinstance(sid, str) or not sid.strip() or not prompt.strip():
            skip("incomplete")
            continue
        final = session.get("final_output")
        if not isinstance(final, str):
            final = next(
                (atif_content_text(s.get("message")) for s in reversed(steps) if s.get("source") == "agent"), ""
            )
        local_files = {}
        inputs = session.get("files", [])
        if not isinstance(inputs, list):
            warnings.append("incomplete: invalid files list")
            inputs = []
        for item in inputs:
            try:
                if not isinstance(item, dict) or not isinstance(item.get("content"), str):
                    raise ValueError("invalid fixture content")
                path = safe_relative(item.get("path")).as_posix()
                local_files[path] = item["content"]
            except ValueError:
                warnings.append("incomplete: unsafe or invalid fixture dropped")
        redacted_prompt, prompt_counts = redact(prompt)
        redacted_files, fixture_counts = redact(local_files)
        fingerprint = hashlib.sha256(
            json.dumps([" ".join(redacted_prompt.split()), sorted(redacted_files.items())], ensure_ascii=False).encode(
                "utf-8"
            )
        ).hexdigest()
        if fingerprint in fingerprints:
            skip("duplicate", duplicate_of=fingerprints[fingerprint])
            continue
        cid = "trace-" + hashlib.sha256(sid.encode()).hexdigest()[:10]
        if cid in ids:
            raise ValueError("Repeated session_id with different replay inputs")
        ids.add(cid)
        fingerprints[fingerprint] = cid
        tools = []
        suggestions = []
        missing_paths: set[str] = set()
        structured_counts = {"secrets": 0, "pii": 0}
        for step in steps:
            calls = step.get("tool_calls", [])
            for call in calls if isinstance(calls, list) else []:
                if not isinstance(call, dict):
                    continue
                missing = _missing_replay_path(call, local_files, skill_name, skill_path)
                if missing is not None and missing not in missing_paths and len(missing_paths) < 20:
                    warnings.append(f"missing replay input: {missing}")
                    missing_paths.add(missing)
                call, call_counts = redact(call, structured=True)
                for category in structured_counts:
                    structured_counts[category] += call_counts[category]
                summary = f"{call.get('function_name', 'unknown')}: {json.dumps(call.get('arguments', {}), ensure_ascii=False)}"
                tools.append(summary)
                if "SKILL.md" in summary or "scripts/" in summary:
                    suggestions.append(f"Review whether the replay should use: {summary}")
        provenance, provenance_counts = redact(
            {
                "session_id": sid,
                "source": session.get("source"),
                "started_at": session.get("started_at"),
                "tags": session.get("tags"),
                "record": record,
            },
            structured=True,
        )
        candidate = {
            "id": cid,
            "prompt": redacted_prompt,
            "files": [],
            "observed": {"outcome": outcome, "final_output": final, "tool_calls": tools, "trajectory_source": source},
            "provenance": provenance,
            "warnings": warnings,
            "review": {
                "decision": "pending",
                "expected_output": "",
                "assertions": [],
                "suggested_assertions": suggestions,
                "notes": "",
            },
        }
        clean, counts = redact({"candidate": candidate, "fixtures": redacted_files})
        for category in counts:
            counts[category] += (
                prompt_counts[category]
                + fixture_counts[category]
                + structured_counts[category]
                + provenance_counts[category]
            )
        candidate = clean["candidate"]
        candidate["redactions"] = counts
        for path, content in clean["fixtures"].items():
            ref = f"fixtures/{cid}/{safe_relative(path).as_posix()}"
            fixtures[ref] = content
            candidate["files"].append(ref)
        candidate["observed"]["tool_calls"] = [s[:250] for s in candidate["observed"]["tool_calls"]]
        candidates.append(candidate)
    document = {
        "format": "skillevaluator.trace-candidates",
        "version": 1,
        "skill_name": skill_name,
        "export": {"name": export_name, "sha256": hashlib.sha256(raw).hexdigest()},
        "candidates": candidates,
        "skipped": skipped,
    }
    return redact(document)[0], fixtures


def import_traces(skill: Path, export: Path, output: Path, *, force: bool = False, dry_run: bool = False) -> dict:
    if output.resolve().is_relative_to(skill.resolve()):
        raise ValueError("Review output must be outside the skill directory")
    target = output / "candidates.json"
    if target.exists() and not force:
        raise ValueError("candidates.json already exists; use --force to overwrite")
    skill_data = _parse_skill(skill)
    document, fixtures = build_candidates(
        export.read_bytes(),
        export.name,
        skill_data["name"],
        skill_path=skill,
        skill_directory_name=skill.name,
    )
    destinations = [(contained(output, ref), content) for ref, content in fixtures.items()]
    contained(output, "candidates.json")
    fixture_bytes = [(path, content.encode("utf-8")) for path, content in destinations]
    document_bytes = (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)
        for path, content in fixture_bytes:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        target.write_bytes(document_bytes)
    return document


def accept_traces(skill: Path, review_dir: Path, *, dry_run: bool = False, allow_native_harbor: bool = False) -> dict:
    document = json.loads((review_dir / "candidates.json").read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or document.get("format") != "skillevaluator.trace-candidates"
        or type(document.get("version")) is not int
        or document["version"] != 1
    ):
        raise ValueError("Expected skillevaluator.trace-candidates version 1")
    name = _parse_skill(skill)["name"]
    accepted_skill_names = (name, skill.name)
    if document.get("skill_name") not in accepted_skill_names:
        raise ValueError("Review skill_name does not match skill")
    source, _ = validate_tier3_source(skill)
    if source == "native_harbor" and not allow_native_harbor:
        raise ValueError(
            "Native Harbor tasks are selected for this skill; pass --allow-native-harbor to explicitly accept traces"
        )
    target = skill / "evals" / "evals.json"
    existing = find_eval_file(skill)
    if existing and existing != target:
        raise ValueError("Existing dataset uses another location or extension; migrate to evals/evals.json first")
    dataset = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {"skill_name": name, "evals": []}
    if (
        not isinstance(dataset, dict)
        or not isinstance(dataset.get("skill_name"), str)
        or not isinstance(dataset.get("evals"), list)
    ):
        raise ValueError("Existing dataset must use agentskills shape, not a legacy flat list")
    dataset_name = dataset["skill_name"]
    if dataset_name not in accepted_skill_names:
        raise ValueError("Existing dataset skill_name must match the frontmatter or directory name")
    errors = []
    additions = []
    copies = []
    present = []
    skipped = []
    by_id = {e.get("id"): e for e in dataset["evals"]}
    for candidate in document["candidates"]:
        cid = candidate["id"]
        safe_relative(cid)
        if not re.fullmatch(r"trace-[0-9a-f]{10}", cid):
            raise ValueError("Invalid candidate id")
        review = candidate["review"]
        decision = review.get("decision")
        if decision in ("pending", "reject"):
            skipped.append(cid)
            continue
        expected = review.get("expected_output", "")
        observed = candidate["observed"]
        if decision == "accept_observed":
            expected = observed.get("final_output", "")
            if observed.get("outcome") != "success" or not isinstance(expected, str) or not expected.strip():
                errors.append(
                    f"{cid}: write an independent expectation; accept_observed requires success and non-empty output"
                )
                continue
        elif decision != "accept":
            errors.append(f"{cid}: unknown review decision")
            continue
        if not isinstance(expected, str) or not expected.strip():
            errors.append(f"{cid}: accept requires non-empty reviewer-written expected_output")
            continue
        trace_source = {
            "session_id": candidate["provenance"]["session_id"],
            "export_sha256": document["export"]["sha256"],
            "observed_outcome": observed["outcome"],
            "decision": decision,
        }
        if cid in by_id:
            if by_id[cid].get("trace_source", {}).get("session_id") == trace_source["session_id"]:
                present.append(cid)
            else:
                errors.append(f"{cid}: id collision with a different session")
            continue
        entry = {
            "id": cid,
            "prompt": candidate["prompt"],
            "expected_output": expected,
            "files": [],
            "expected_skill": dataset_name,
            "trace_source": trace_source,
        }
        assertions = review.get("assertions", [])
        if not isinstance(assertions, list) or any(not isinstance(a, str) for a in assertions):
            errors.append(f"{cid}: assertions must be a list of strings")
            continue
        if assertions:
            entry["assertions"] = assertions
        contents = []
        for ref in candidate.get("files", []):
            prefix = f"fixtures/{cid}/"
            if not isinstance(ref, str) or not ref.startswith(prefix):
                raise ValueError(f"{cid}: invalid fixture reference")
            source = contained(review_dir, ref)
            rel = f"trace-files/{cid}/{safe_relative(ref[len(prefix) :]).as_posix()}"
            dest = contained(skill / "evals", rel)
            content = source.read_bytes().decode("utf-8")
            contents.append(content)
            copies.append((dest, content))
            entry["files"].append(rel)
        payload = [entry, contents]
        if redact(payload)[0] != payload:
            errors.append(f"{cid}: secret/PII redaction required; sanitize the review before accepting")
            continue
        additions.append(entry)
        by_id[cid] = entry
    if errors:
        raise ValueError("\n".join(errors))
    validation = []
    if additions:
        contained(skill, "evals/evals.json")
        updated_dataset = {**dataset, "evals": [*dataset["evals"], *additions]}
        copy_bytes = [(dest, content.encode("utf-8")) for dest, content in copies]
        dataset_bytes = (json.dumps(updated_dataset, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if not dry_run and additions:
        for dest, content in copy_bytes:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(dataset_bytes)
        _validate_dataset_contents(target, validation)
    return {
        "added": [e["id"] for e in additions],
        "already present": present,
        "skipped": skipped,
        "validation": validation,
    }
