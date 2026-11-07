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
from skillevaluator.tier3.evals_spec import _validate_dataset_contents
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


@lru_cache(maxsize=1)
def _pii_patterns() -> tuple[tuple[str, re.Pattern[str], tuple[str, ...]], ...]:
    configured = load_pii_patterns()
    return tuple(
        (category, re.compile(spec["pattern"]), tuple(spec.get("exceptions", [])))
        for category in ("personal_paths", "home_paths", "emails", "phone_numbers", "ssn")
        for spec in configured.get(category, [])
    )


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
            return {clean(k): clean(v) for k, v in item.items()}
        if not isinstance(item, str):
            return item
        text = redact_sensitive_text(item)
        counts["secrets"] += max(0, text.count("<redacted>") - item.count("<redacted>"))
        for category, pattern, exceptions in patterns:

            def replace(
                match: re.Match[str], category: str = category, exceptions: tuple[str, ...] = exceptions
            ) -> str:
                found = match.group()
                if category == "home_paths":
                    replacement = "/home/<user>/"
                elif category == "personal_paths":
                    replacement = "/Users/<user>/" if found.startswith("/") else "C:\\Users\\<user>"
                else:
                    replacement = "[redacted-pii]"
                if found == replacement or any(e in found for e in exceptions):
                    return found
                counts["pii"] += 1
                return replacement

            text = pattern.sub(replace, text)
        return text

    return clean(value), counts


def parse_export(raw: bytes) -> list[tuple[int, Any]]:
    """Parse a v1 wrapper or JSONL; malformed lines become unsupported records."""
    text = raw.decode("utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict) and any(k in obj for k in ("format", "version", "sessions")):
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
    for number, line in enumerate(text.splitlines(), 1):
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
    if relative in fixtures or relative.removeprefix("/workspace/input/") in fixtures:
        return None
    target = Path(path)
    if target.is_absolute():
        if skill_path is not None and target.resolve().is_relative_to(skill_path.resolve()):
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


def build_candidates(
    raw: bytes, export_name: str, skill_name: str, *, skill_path: Path | None = None
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
        if "skill" in session and session["skill"] != skill_name:
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
        fingerprint = hashlib.sha256(
            json.dumps([" ".join(prompt.split()), sorted(local_files.items())]).encode()
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
            "prompt": prompt,
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
        clean, counts = redact({"candidate": candidate, "fixtures": local_files})
        for category in counts:
            counts[category] += structured_counts[category] + provenance_counts[category]
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
    document, fixtures = build_candidates(
        export.read_bytes(), export.name, _parse_skill(skill)["name"], skill_path=skill
    )
    destinations = [(contained(output, ref), content) for ref, content in fixtures.items()]
    contained(output, "candidates.json")
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)
        for path, content in destinations:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        target.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return document


def accept_traces(skill: Path, review_dir: Path, *, dry_run: bool = False) -> dict:
    document = json.loads((review_dir / "candidates.json").read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or document.get("format") != "skillevaluator.trace-candidates"
        or type(document.get("version")) is not int
        or document["version"] != 1
    ):
        raise ValueError("Expected skillevaluator.trace-candidates version 1")
    name = _parse_skill(skill)["name"]
    if document.get("skill_name") != name:
        raise ValueError("Review skill_name does not match skill")
    target = skill / "evals" / "evals.json"
    existing = find_eval_file(skill)
    if existing and existing != target:
        raise ValueError("Existing dataset uses another location or extension; migrate to evals/evals.json first")
    dataset = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {"skill_name": name, "evals": []}
    if not isinstance(dataset, dict) or dataset.get("skill_name") != name or not isinstance(dataset.get("evals"), list):
        raise ValueError("Existing dataset must use agentskills shape, not a legacy flat list")
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
            "expected_skill": name,
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
            rel = f"files/trace/{cid}/{safe_relative(ref[len(prefix) :]).as_posix()}"
            dest = contained(skill / "evals", rel)
            content = source.read_text(encoding="utf-8")
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
    if not dry_run and additions:
        contained(skill, "evals/evals.json")
        for dest, content in copies:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        dataset["evals"].extend(additions)
        target.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _validate_dataset_contents(target, validation)
    return {
        "added": [e["id"] for e in additions],
        "already present": present,
        "skipped": skipped,
        "validation": validation,
    }
