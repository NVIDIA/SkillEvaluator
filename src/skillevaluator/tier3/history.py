# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summary-only, read-only Skill Lift history and exports."""

from __future__ import annotations

import csv
import html
import io
import json
import math
import stat
from contextlib import suppress
from pathlib import Path
from typing import Any

from rich.box import SIMPLE
from rich.console import Console
from rich.table import Table
from rich.text import Text

from skillevaluator.tier3.baseline_store import comparison_identity, comparison_mismatches
from skillevaluator.tier3.results_location import (
    _CURRENT_RUN_SUFFIX,
    _run_timestamp_parts,
    iter_candidate_results_roots,
    ordered_run_directories,
)
from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot


def history_directories(root: Path) -> list[Path]:
    """Keep shared ordering, including reservations without completion artifacts."""
    directories = ordered_run_directories(root, completed_only=False)
    try:
        for candidate in root.iterdir():
            parsed = _run_timestamp_parts(candidate.name)
            if parsed is None or (parsed[1] and not _CURRENT_RUN_SUFFIX.fullmatch(parsed[1])):
                continue
            if candidate not in directories and stat.S_ISDIR(candidate.lstat().st_mode) and not candidate.is_symlink():
                directories.append(candidate)
    except OSError:
        pass

    def key(path: Path):
        parsed = _run_timestamp_parts(path.name)
        try:
            mtime = (path / "result.json").lstat().st_mtime_ns
        except OSError:
            mtime = 0
        return (parsed[0].isoformat() if parsed else "", mtime, path.name)

    return sorted(directories, key=key)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value: {value}")


def _read(root: SecureRoot, path: str) -> tuple[dict[str, Any], str | None]:
    try:
        data, _ = root.read_bytes(Path(path), 32 * 1024 * 1024)
        result = json.loads(data, parse_constant=_invalid_constant)
        if not isinstance(result, dict):
            raise ValueError("expected an object")
        return result, None
    except (OSError, ValueError, SecurePathError) as exc:
        return {}, f"missing or corrupt {path}: {str(exc)[:160]}"


def _number(value: Any, minimum: float = 0, maximum: float = 1) -> float | None:
    if type(value) not in (int, float):
        return None
    try:
        return float(value) if math.isfinite(value) and minimum <= value <= maximum else None
    except OverflowError:
        return None


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _identity(config: dict, result: dict) -> dict:
    recorded = _mapping(result.get("comparison_identity"))
    facts = {**recorded, **result}
    policy = _mapping(facts.get("attempt_policy"))
    facts["attempt_policy"] = {
        key: value for key, value in policy.items() if key not in {"score_policy", "score_definition"}
    }
    cases = facts.get("expected_case_ids")
    facts["expected_case_ids"] = (
        cases if isinstance(cases, list) and all(isinstance(case, str) for case in cases) else []
    )
    identity = comparison_identity(_mapping(recorded.get("config")) or config, facts)
    skill = _mapping(recorded.get("skill"))
    for key in ("evaluated_source", "publication_target"):
        if identity["skill"].get(key) is None:
            identity["skill"][key] = skill.get(key, result.get(key))
    return identity


def _known_identity(identity: dict, agent: str | None) -> bool:
    model = _mapping(_mapping(identity["config"].get("agents")).get(agent)).get("model")
    return bool(
        isinstance(model, str)
        and model
        and isinstance(identity.get("dataset_digest"), str)
        and identity["dataset_digest"]
        and identity.get("expected_case_ids")
        and identity.get("attempt_policy")
        and isinstance(identity.get("score_policy"), str)
        and identity["score_policy"]
        and isinstance(identity.get("evaluator_version"), str)
        and identity["evaluator_version"]
    )


def _known_mismatches(current: dict, previous: dict) -> list[str]:
    if not current.get("config") or not previous.get("config"):
        return []
    current, previous = dict(current), dict(previous)
    for field in ("dataset_digest", "expected_case_ids", "attempt_policy", "score_policy", "evaluator_version"):
        if not current.get(field) or not previous.get(field):
            current[field] = previous[field] = [] if field == "expected_case_ids" else None
    for identity in (current, previous):
        config = dict(_mapping(identity.get("config")))
        config["agents"] = _mapping(config.get("agents"))
        identity["config"] = config
    if not current["config"].get("agents") or not previous["config"].get("agents"):
        current["config"]["agents"] = previous["config"]["agents"] = {}
    reasons = comparison_mismatches(current, previous)
    if set(current["config"]["agents"]) != set(previous["config"]["agents"]):
        reasons.append("agent selection changed")
    return reasons


def _run_rows(run_dir: Path, selected_agent: str | None) -> list[tuple[dict, dict]]:
    with SecureRoot(run_dir) as root:
        config, config_error = _read(root, "run_config.json")
        result, result_error = _read(root, "result.json")
        config = config or _mapping(result.get("run_config"))
        identity = _identity(config, result)
        agents = set(_mapping(config.get("agents"))) | set(_mapping(result.get("agents")))
        agents.update(_mapping(identity["config"].get("agents")))
        with suppress(OSError):
            agents.update(
                p.name
                for p in run_dir.iterdir()
                if p.is_dir()
                and not p.is_symlink()
                and any(
                    (p / name).exists()
                    for name in ("summary.json", "with-skill/summary.json", "without-skill/summary.json")
                )
            )
        names = sorted(
            name for name in agents if isinstance(name, str) and Path(name).name == name and name not in {".", ".."}
        )
        if selected_agent is not None:
            names = [name for name in names if name == selected_agent]
        elif not names:
            names = [None]
        rows = []
        for agent in names:
            payload = _mapping(_mapping(result.get("agents")).get(agent))
            conditions = _mapping(payload.get("conditions"))
            summaries = {}
            for arm in ("with_skill", "without_skill"):
                summaries[arm], _ = (
                    _read(root, f"{agent}/{arm.replace('_', '-')}/summary.json") if agent else ({}, None)
                )
            if agent and not summaries["with_skill"]:
                summaries["with_skill"], _ = _read(root, f"{agent}/summary.json")
            with_condition = {**summaries["with_skill"], **_mapping(conditions.get("with_skill"))}
            without_condition = {**summaries["without_skill"], **_mapping(conditions.get("without_skill"))}
            coverage = {
                arm: {
                    "scored_cases": _count(condition.get("scored_cases")),
                    "expected_cases": _count(condition.get("expected_cases")),
                }
                for arm, condition in (("with_skill", with_condition), ("without_skill", without_condition))
            }
            baseline = _mapping(result.get("baseline", config.get("baseline")))
            status = result.get("execution_status") or (
                "corrupt" if result_error and (run_dir / "result.json").exists() else "incomplete"
            )
            reasons = [error for error in (config_error, result_error) if error]
            if not isinstance(status, str):
                status = "corrupt"
                reasons.append("invalid run execution status")
            if result.get("run_id", run_dir.name) != run_dir.name:
                reasons.append("result run id disagrees with directory")
                status = "corrupt"
            if status != "succeeded":
                reasons.append(f"run status is {status}")
            if baseline.get("mode") == "skipped" or without_condition.get("execution_status") == "skipped":
                reasons.append("without-skill baseline was skipped")
            if with_condition.get("execution_status") == "skipped":
                reasons.append("with-skill arm was skipped (baseline-only run)")
            for arm, condition in (("with_skill", with_condition), ("without_skill", without_condition)):
                scored, expected = coverage[arm]["scored_cases"], coverage[arm]["expected_cases"]
                if condition.get("execution_status") != "succeeded":
                    reasons.append(f"{arm}: condition did not succeed")
                if expected is None or expected == 0 or scored != expected:
                    reasons.append(f"{arm}: incomplete or unknown case coverage")
                attempts, expected_attempts = (
                    _count(condition.get("scored_attempts")),
                    _count(condition.get("expected_attempts")),
                )
                if (
                    expected_attempts is not None
                    and (attempts is None or attempts < expected_attempts)
                    and not _mapping(identity.get("attempt_policy")).get("stop_on_pass")
                ):
                    reasons.append(f"{arm}: incomplete attempt coverage")
            with_policy = payload.get("score_policy_with_skill", with_condition.get("score_policy"))
            without_policy = payload.get("score_policy_without_skill", without_condition.get("score_policy"))
            row_identity = {**identity, "score_policy": with_policy or identity.get("score_policy")}
            if with_policy and without_policy and with_policy != without_policy:
                reasons.append("score policy differs between arms")
            lift = _mapping(_mapping(payload.get("lift")).get("overall")) or _mapping(
                _mapping(payload.get("custom_lift")).get("overall")
            )
            if not lift and agent:
                retained_lift, _ = _read(root, f"{agent}/lift.json")
                lift = _mapping(retained_lift.get("overall"))
            delta = _number(lift.get("delta"), -1, 1)
            with_score = _number(payload.get("overall_with_skill", summaries["with_skill"].get("overall_score")))
            without_score = _number(
                payload.get("overall_without_skill", summaries["without_skill"].get("overall_score"))
            )
            if with_score is None or without_score is None:
                reasons.append("one or both condition overall scores are unscored or unknown")
            if delta is None:
                reasons.append("paired overall lift was not recorded")
            elif (
                any(_number(lift.get(key)) is None for key in ("with_skill", "without_skill"))
                or abs(lift["with_skill"] - lift["without_skill"] - delta) > 0.0002
            ):
                reasons.append("paired overall lift is corrupt")
            skill = identity["skill"]
            revision = _mapping(skill.get("evaluated_source")).get("revision")
            digest = _mapping(skill.get("publication_target")).get("skill_digest")
            model = (
                payload.get("model")
                or _mapping(_mapping(config.get("agents")).get(agent)).get("model")
                or summaries["with_skill"].get("model")
            )
            row = {
                "run_id": run_dir.name,
                "evaluated_at": result.get("evaluated_at"),
                "agent": agent,
                "skill_identity": skill,
                "skill_revision": revision,
                "skill_digest": digest,
                "model": model,
                "dataset_digest": identity.get("dataset_digest"),
                "case_count": len(identity["expected_case_ids"])
                if identity["expected_case_ids"]
                else coverage["with_skill"]["expected_cases"],
                "with_skill_score": with_score,
                "without_skill_score": without_score,
                "lift_delta": None if reasons else delta,
                "lift_status": "unavailable" if reasons else "available",
                "lift_reasons": reasons,
                "coverage": coverage,
                "baseline_mode": baseline.get("mode"),
                "status": status,
                "score_policy": row_identity.get("score_policy"),
            }
            rows.append((row, row_identity))
        return rows


def collect_history(skill_path: Path, results_dir: Path | None = None, agent: str | None = None) -> dict:
    """Read the first populated results root, with the usual legacy fallback."""
    candidates = iter_candidate_results_roots(skill_path, results_dir)
    root, directories = candidates[0], []
    for candidate in candidates:
        if found := history_directories(candidate):
            root, directories = candidate, found
            break
    rows, states = [], {}
    for directory in directories:
        try:
            observed = _run_rows(directory, agent)
        except (OSError, ValueError, SecurePathError) as exc:
            observed = [
                (
                    {
                        "run_id": directory.name,
                        "agent": agent,
                        "status": "corrupt",
                        "lift_delta": None,
                        "lift_status": "unavailable",
                        "lift_reasons": [str(exc)],
                    },
                    {},
                )
            ]
        for row, identity in observed:
            previous = states.get(row["agent"])
            known = bool(identity) and _known_identity(identity, row["agent"])
            mismatches = []
            if previous and identity and previous["identity"]:
                mismatches = _known_mismatches(identity, previous["identity"])
            segment = (previous["segment"] + bool(mismatches)) if previous else 1
            prior_lift = previous["lift"] if previous and not mismatches else None
            row.update(
                segment=segment,
                config_changed=bool(mismatches),
                identity_status="known" if known else "unknown",
                comparison_mismatches=mismatches,
                lift_change=round(row["lift_delta"] - prior_lift, 4)
                if known and row["lift_delta"] is not None and prior_lift is not None
                else None,
            )
            states[row["agent"]] = {
                "segment": segment,
                "identity": identity if known or mismatches or not previous else previous["identity"],
                "lift": row["lift_delta"]
                if row["lift_status"] == "available" and known
                else (prior_lift if not mismatches else None),
            }
            rows.append({key: row.get(key) for key in _ROW_FIELDS})
    return {"schema_version": "1.0", "results_root": str(root), "rows": rows}


def _coverage(row: dict) -> str:
    def arm(name):
        value = _mapping(_mapping(row.get("coverage")).get(name))
        return f"{value.get('scored_cases') if value.get('scored_cases') is not None else '?'}/{value.get('expected_cases') if value.get('expected_cases') is not None else '?'}"

    return f"{arm('with_skill')} / {arm('without_skill')}"


def _display(value: Any) -> str:
    return "unknown" if value is None else str(value)


def _cells(row: dict) -> list[str]:
    return [
        str(row["segment"]) + (" * BREAK" if row["config_changed"] else ""),
        row["run_id"],
        _display(row.get("evaluated_at")),
        _display(row.get("agent")),
        _display(row.get("skill_revision") or row.get("skill_digest")),
        _display(row.get("model")),
        f"{_display(row.get('dataset_digest'))} ({_display(row.get('case_count'))})",
        *[
            "unavailable"
            if key == "lift_delta" and row.get(key) is None
            else f"{row[key]:+.4f}"
            if key == "lift_change" and row.get(key) is not None
            else _display(row.get(key))
            for key in ("with_skill_score", "without_skill_score", "lift_delta", "lift_change")
        ],
        _coverage(row),
        _display(row.get("baseline_mode")),
        _display(row.get("status")),
        "; ".join(row.get("lift_reasons", [])),
    ]


_HEADERS = (
    "Segment",
    "Run",
    "Evaluated at",
    "Agent",
    "Revision / content",
    "Model",
    "Dataset (cases)",
    "With",
    "Without",
    "Lift",
    "Change",
    "Cases W / WO",
    "Baseline",
    "Status",
    "Lift reason",
)

_ROW_FIELDS = (
    "run_id",
    "evaluated_at",
    "agent",
    "skill_identity",
    "skill_revision",
    "skill_digest",
    "model",
    "dataset_digest",
    "case_count",
    "with_skill_score",
    "without_skill_score",
    "lift_delta",
    "lift_status",
    "lift_reasons",
    "coverage",
    "baseline_mode",
    "status",
    "score_policy",
    "segment",
    "config_changed",
    "comparison_mismatches",
    "identity_status",
    "lift_change",
)


def print_history(history: dict, console: Console) -> None:
    headers = (_HEADERS[0], "Run / evaluated at", *_HEADERS[3:-1])
    table = Table(*headers, box=SIMPLE, min_width=220)
    for row in history["rows"]:
        if row["config_changed"]:
            table.add_section()
        cells = _cells(row)
        cells[4] = cells[4][:22]
        cells[6] = f"{_display(row.get('dataset_digest'))[:22]} ({_display(row.get('case_count'))})"
        if row["lift_delta"] is None:
            cells[9] = "unavailable"
        table.add_row(*(Text(cell) for cell in [cells[0], cells[1] + "\n" + cells[2], *cells[3:-1]]))
    wide_console = Console(
        file=console.file,
        width=max(console.width, 240),
        color_system=console.color_system,
        force_terminal=console.is_terminal,
    )
    wide_console.print(table)
    if not history["rows"]:
        console.print("No retained runs match this history request.")
    for row in history["rows"]:
        if row.get("lift_reasons"):
            console.print(Text(f"{row['run_id']} ({row.get('agent')}): " + "; ".join(row["lift_reasons"])))
        if row["config_changed"]:
            console.print(
                Text(
                    f"BREAK: segment {row['segment']} ({row.get('agent')}): " + "; ".join(row["comparison_mismatches"])
                )
            )


def export_history(history: dict, suffix: str) -> str:
    if suffix == ".json":
        return json.dumps(history, indent=2, allow_nan=False) + "\n"
    if suffix == ".csv":
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=_ROW_FIELDS)
        writer.writeheader()
        for row in history["rows"]:
            writer.writerow(
                {key: json.dumps(value) if isinstance(value, dict | list) else value for key, value in row.items()}
            )
        return output.getvalue()
    if suffix != ".md":
        raise ValueError("History output must end in .json, .csv or .md")

    def safe(value):
        return html.escape(str(value)).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    header = "| " + " | ".join(_HEADERS) + " |"
    divider = "| " + " | ".join("---" for _ in _HEADERS) + " |"
    lines = ["# Skill Lift history", "", header, divider]
    for row in history["rows"]:
        if row["config_changed"]:
            lines.extend(
                [
                    "",
                    f"**Segment {row['segment']} break ({safe(row.get('agent'))}): {safe('; '.join(row['comparison_mismatches']))}**",
                    "",
                    header,
                    divider,
                ]
            )
        lines.append("| " + " | ".join(safe(cell) for cell in _cells(row)) + " |")
    return "\n".join(lines) + "\n"
