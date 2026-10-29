# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Case-first Tier 3 summary: per-case attempt grid, verdict, and a short "How to improve" list."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping
from difflib import SequenceMatcher
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.box import SIMPLE
from rich.table import Table
from rich.text import Text

from skillevaluator.tier3.harbor.progress import redact_progress_detail, secret_values_from_environment
from skillevaluator.utils.redaction import redact_sensitive_text
from skillevaluator.utils.secret_redaction import Redactor
from skillevaluator.utils.secure_fs import SecurePathError, SecureRoot

if TYPE_CHECKING:
    from rich.console import Console

logger = logging.getLogger(__name__)

PASS, FAIL, ERROR, SKIPPED = "✓", "✗", "⊘", "·"
_VARIANTS = (("with_skill", "with-skill"), ("without_skill", "without-skill"))
_CLEAR_EFFECT_PP = 10
_CLEAR_SCORE_DELTA = 0.1
_MAX_FIXES = 3
_ROOT_ERROR = re.compile(r"(?m)^\s*(?:error|Error|\w+Error):\s*(.+)$")
_MAX_REASON_CHARS = 220
_MAX_SKILL_MD_CHARS = 6000
_MAX_TRIAL_JSON_BYTES = 8 * 1024 * 1024
_MAX_LOG_TAIL_BYTES = 64 * 1024
_MAX_TRIAL_PATHS_SCANNED = 4096
_STDOUT = re.compile(r"\bstdout:\s*(.+?)\s*(?:\bstderr:|$)", re.DOTALL)
_RESTATED_CHARS = 60
_ASKS_FOR_INPUT = re.compile(
    r"(?i)(workspace (?:is|appears to be|looks|seems)[^.\n]{0,20}empty|"
    r"(?:there's|there is) no [^.\n]{0,60}(?:here|in (?:the|your|it|this))|doesn't (?:have|contain) any|"
    r"(?:could|can) you (?:share|provide|upload)|please (?:share|provide|upload)|"
    r"what's the file path|I(?:'ll| will)? need (?:a few (?:quick )?(?:things|details)|a bit more|more (?:context|information))|"
    r"send me (?:these|the following|a few|the details))"
)
_STOPPED_FOR_INPUT = re.compile(
    r"(?i)\b(?:asked|request(?:ed|s)|clarification|does not exist|did not exist|not found|no [^.]{0,40} (?:was |were )?found)\b"
)
_SAFE_TRIAL = re.compile(r"^[A-Za-z0-9._-]+$")
_ENV_ERROR = re.compile(
    r"(ModuleNotFoundError: No module named '?[\w.]+'?|No module named '?[\w.]+'?|"
    r"[\w./-]+: command not found|command not found: [\w.-]+|"
    r"(?:is not installed|not installed in)[^\n\"]{0,60}|"
    r"No such file or directory: '[^']+'|Could not find a version that satisfies[^\n\"]{0,80})"
)
_JUDGED_METRICS = ("goal_accuracy", "accuracy", "behavior_check")


def _never_activated(agent: Mapping[str, Any]) -> bool:
    activation = agent.get("skill_activation")
    return isinstance(activation, Mapping) and activation.get("status") == "never_activated"


def _details_truncated(agent: Mapping[str, Any]) -> bool:
    branches = agent.get("pass_at_k")
    for key, _ in _VARIANTS:
        branch = branches.get(key) if isinstance(branches, Mapping) else None
        if not isinstance(branch, Mapping):
            continue
        if branch.get("case_details_truncated"):
            return True
        cases = branch.get("cases")
        if isinstance(cases, Mapping) and any(
            isinstance(case, Mapping) and case.get("attempt_details_truncated") for case in cases.values()
        ):
            return True
    return False


def _contract_mismatch(agent: Mapping[str, Any]) -> bool:
    with_metrics, without_metrics = agent.get("metrics_with_skill"), agent.get("metrics_without_skill")
    if (
        isinstance(with_metrics, list | tuple)
        and isinstance(without_metrics, list | tuple)
        and set(with_metrics) != set(without_metrics)
    ):
        return True
    with_policy, without_policy = agent.get("score_policy_with_skill"), agent.get("score_policy_without_skill")
    return bool(with_policy and without_policy and with_policy != without_policy)


def _execution_unavailable(agent: Mapping[str, Any]) -> bool:
    if agent.get("execution_status") not in {None, "succeeded", "complete"}:
        return True
    conditions = agent.get("conditions")
    return isinstance(conditions, Mapping) and any(
        isinstance(condition := conditions.get(key), Mapping)
        and condition.get("execution_status") not in {None, "succeeded", "complete"}
        for key, _ in _VARIANTS
    )


def _comparable(agent: Mapping[str, Any]) -> bool:
    return not (
        _never_activated(agent)
        or _details_truncated(agent)
        or _contract_mismatch(agent)
        or _case_mismatch(agent)
        or _execution_unavailable(agent)
    ) and all(ERROR not in _marks(case) for key, _ in _VARIANTS for case in _cases(agent, key).values())


def _case_mismatch(agent: Mapping[str, Any]) -> bool:
    case_sets = [
        {
            case_id
            for case_id, case in _cases(agent, key).items()
            if not isinstance(case, Mapping) or not case.get("extra_case")
        }
        for key, _ in _VARIANTS
    ]
    return bool(all(case_sets) and case_sets[0] != case_sets[1])


def _case_comparable(agent: Mapping[str, Any], with_case: Any, without_case: Any) -> bool:
    return not (_never_activated(agent) or _contract_mismatch(agent) or _execution_unavailable(agent)) and all(
        isinstance(case, Mapping) and not case.get("attempt_details_truncated") and ERROR not in _marks(case)
        for case in (with_case, without_case)
    )


# ---------------------------------------------------------------- per-case data


def _cases(agent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    pass_at_k = agent.get("pass_at_k")
    variant = pass_at_k.get(key) if isinstance(pass_at_k, Mapping) else None
    cases = variant.get("cases") if isinstance(variant, Mapping) else None
    if isinstance(cases, Mapping) and cases:
        return cases
    # A failed condition persists no pass@k cases; rebuild them from its trial rewards and errors.
    failures = agent.get("trial_failures")
    failed = failures.get(key) if isinstance(failures, Mapping) else None
    errored = Counter(
        str(f["trial"]).split("__", 1)[0] for f in failed or [] if isinstance(f, Mapping) and f.get("trial")
    )
    rebuilt = _scored_trials(agent, key)
    for case_id, n in errored.items():
        rebuilt.setdefault(case_id, {"attempts": []})["attempts_missing"] = n
    return rebuilt


def _scored_trials(agent: Mapping[str, Any], key: str) -> dict[str, dict[str, Any]]:
    output_dir = agent.get("output_dir")
    trials = Path(str(output_dir)) / dict(_VARIANTS)[key] / "trials" if output_dir else None
    if trials is None or not trials.is_dir():
        return {}
    pass_at_k = agent.get("pass_at_k") if isinstance(agent.get("pass_at_k"), Mapping) else {}
    threshold = next(
        (
            float(v["pass_threshold"])
            for v in pass_at_k.values()
            if isinstance(v, Mapping) and isinstance(v.get("pass_threshold"), int | float)
        ),
        0.5,
    )
    cases: dict[str, dict[str, Any]] = {}
    for trial in sorted(trials.iterdir()):
        reward = _load_json(trial / "reward.json", root=Path(str(output_dir)))
        score = reward.get("overall") if isinstance(reward, Mapping) else None
        if isinstance(score, int | float) and not isinstance(score, bool):
            attempts = cases.setdefault(trial.name.split("__", 1)[0], {"attempts": []})["attempts"]
            attempts.append(
                {"attempt": len(attempts) + 1, "trial": trial.name, "score": score, "passed": score >= threshold}
            )
    return cases


def _marks(case: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(case, Mapping):
        return []
    attempts = case.get("attempts") if isinstance(case.get("attempts"), list) else []
    marks = [PASS if a.get("passed") is True else FAIL for a in attempts if isinstance(a, Mapping)]
    marks += [ERROR] * int(case.get("attempts_missing") or 0)
    marks += [SKIPPED] * int(case.get("attempts_skipped") or 0)
    return marks


def _attempt_rate(marks: list[str]) -> tuple[int, int]:
    scored = [m for m in marks if m in (PASS, FAIL)]
    return scored.count(PASS), len(scored)


def _case_ids(agents: Mapping[str, Any]) -> list[str]:
    ids: dict[str, None] = {}
    for agent in agents.values():
        if isinstance(agent, Mapping):
            for key, _ in _VARIANTS:
                ids.update(dict.fromkeys(str(case_id) for case_id in _cases(agent, key)))
    return sorted(ids)


def _agent_rates(agent: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    rates = {}
    for key, _ in _VARIANTS:
        passed = total = 0
        for case in _cases(agent, key).values():
            p, n = _attempt_rate(_marks(case))
            passed, total = passed + p, total + n
        rates[key] = (passed, total)
    return rates


def _pct(passed: int, total: int) -> float | None:
    return 100.0 * passed / total if total else None


def verdict(agents: Mapping[str, Any]) -> tuple[str, str]:
    """Return (label, per-agent detail) using the raw per-attempt success rate."""
    deltas: dict[str, float] = {}
    not_run: dict[str, int] = {}
    unavailable: list[str] = []
    for name, agent in agents.items():
        if not isinstance(agent, Mapping):
            continue
        if _never_activated(agent):
            unavailable.append(f"{name}: skill never activated; scores are diagnostic only")
            continue
        if _details_truncated(agent):
            unavailable.append(f"{name}: case/attempt details are truncated")
            continue
        if _contract_mismatch(agent):
            unavailable.append(f"{name}: condition score contracts differ")
            continue
        if _case_mismatch(agent):
            unavailable.append(f"{name}: conditions cover different cases")
            continue
        if _execution_unavailable(agent):
            unavailable.append(f"{name}: condition execution is incomplete")
            continue
        missing = sum(_marks(case).count(ERROR) for key, _ in _VARIANTS for case in _cases(agent, key).values())
        if missing:
            # Rates over different subsets of attempts are not comparable.
            not_run[str(name)] = missing
            continue
        rates = _agent_rates(agent)
        with_pct, without_pct = _pct(*rates["with_skill"]), _pct(*rates["without_skill"])
        if with_pct is not None and without_pct is not None:
            deltas[str(name)] = with_pct - without_pct
    detail = " · ".join(
        [f"{name} {delta:+.0f} pp" for name, delta in deltas.items()]
        + [f"{name}: {n} attempt(s) did not run" for name, n in not_run.items()]
        + unavailable
    )
    if not deltas:
        return ("inconclusive", detail) if not_run or unavailable else ("", "")
    if all(delta >= _CLEAR_EFFECT_PP for delta in deltas.values()):
        label = "skill helps"
    elif all(delta <= -_CLEAR_EFFECT_PP for delta in deltas.values()):
        label = "skill hurts"
    elif any(delta <= -_CLEAR_EFFECT_PP for delta in deltas.values()):
        label = "mixed: skill hurts on some agents"
    elif any(delta >= _CLEAR_EFFECT_PP for delta in deltas.values()):
        label = "mixed: skill helps on some agents"
    else:
        label = "no clear effect"
    return label, detail


# ---------------------------------------------------------------- rendering


def _marks_text(marks: list[str]) -> Text:
    text = Text()
    style = {PASS: "green", FAIL: "red", ERROR: "yellow", SKIPPED: "dim"}
    for mark in marks:
        text.append(mark, style=style[mark])
    return text if marks else Text("-", style="dim")


def _mean_score(case: Mapping[str, Any] | None) -> float | None:
    attempts = case.get("attempts") if isinstance(case, Mapping) and isinstance(case.get("attempts"), list) else []
    scores = [s for a in attempts if isinstance(a, Mapping) and (s := _score(a.get("score"))) is not None]
    return sum(scores) / len(scores) if scores else None


def _condition_cell(case: Mapping[str, Any] | None) -> Text:
    marks = _marks(case)
    p, n = _attempt_rate(marks)
    cell = Text(f"{p}/{n} " if n else "", style="bold")
    cell.append_text(_marks_text(marks))
    if (mean := _mean_score(case)) is not None:
        cell.append(f" ({mean:.2f})", style="dim")
    return cell


def _effect(
    with_case: Mapping[str, Any] | None, without_case: Mapping[str, Any] | None, *, comparable: bool = True
) -> Text:
    """One word per case on what the skill changed: pass rate first, then mean score."""
    with_marks, without_marks = _marks(with_case), _marks(without_case)
    (wp, wn), (op, on) = _attempt_rate(with_marks), _attempt_rate(without_marks)
    if not comparable or not wn or not on or ERROR in with_marks + without_marks:
        return Text("inconclusive", style="yellow")
    if wp * on != op * wn:
        return Text("helps", style="green") if wp * on > op * wn else Text("hurts", style="bold red")
    if not wp:
        return Text("both fail", style="red")
    delta = (_mean_score(with_case) or 0.0) - (_mean_score(without_case) or 0.0)
    if delta >= _CLEAR_SCORE_DELTA:
        return Text(f"better score ({delta:+.2f})", style="green")
    if delta <= -_CLEAR_SCORE_DELTA:
        return Text(f"worse score ({delta:+.2f})", style="yellow")
    return Text("no change", style="dim")


def _summary_cells(
    with_value: float | None,
    without_value: float | None,
    fmt: str,
    delta_fmt: str,
    weak: float | None = None,
    *,
    comparable: bool = True,
) -> list[Text]:
    def one(value: float | None) -> Text:
        if value is None:
            return Text("-", style="dim")
        return Text(fmt.format(value), style="yellow" if weak is not None and value < weak else "")

    delta = (
        Text(delta_fmt.format(with_value - without_value), style="dim")
        if comparable and with_value is not None and without_value is not None
        else Text("")
    )
    return [one(with_value), one(without_value), delta]


def _score(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _dimension_score(agent: Mapping[str, Any], key: str, name: str) -> float | None:
    dims = agent.get(f"dimensions_{key}")
    entry = dims.get(name) if isinstance(dims, Mapping) else None
    return _score(entry.get("score") if isinstance(entry, Mapping) else entry)


def _quality(agent: Mapping[str, Any], key: str) -> float | None:
    overall = agent.get("lift", {}).get("overall") if isinstance(agent.get("lift"), Mapping) else None
    if isinstance(overall, Mapping) and _score(overall.get(key)) is not None:
        return _score(overall.get(key))
    return _score(agent.get(f"overall_{key}"))


def _table(*headers: str | Text) -> Table:
    table = Table(box=SIMPLE, show_edge=False, pad_edge=False)
    table.add_column(headers[0], overflow="fold")
    for header in headers[1:-1]:
        table.add_column(header, no_wrap=True)
    table.add_column(headers[-1])
    return table


def _condition_grid(
    agents: Mapping[str, Any], case_ids: list[str], dims: list[str], title: str, safe: Callable[[object], str]
) -> Table:
    """One agent: a row per case with both conditions side by side and the effect."""
    [agent] = agents.values()
    comparable = _comparable(agent)
    table = _table(title, "with skill", "without skill", "effect")
    for case_id in case_ids:
        with_case, without_case = _cases(agent, "with_skill").get(case_id), _cases(agent, "without_skill").get(case_id)
        table.add_row(
            Text(safe(case_id)),
            _condition_cell(with_case),
            _condition_cell(without_case),
            _effect(with_case, without_case, comparable=_case_comparable(agent, with_case, without_case)),
        )
    table.add_section()
    rates = _agent_rates(agent)
    table.add_row(
        Text("success rate (shown)" if _details_truncated(agent) else "success rate", style="bold"),
        *_summary_cells(
            _pct(*rates["with_skill"]), _pct(*rates["without_skill"]), "{:.0f}%", "{:+.0f} pp", comparable=comparable
        ),
    )
    table.add_row(
        Text("quality score (0-1)", style="bold"),
        *_summary_cells(
            _quality(agent, "with_skill"), _quality(agent, "without_skill"), "{:.2f}", "{:+.2f}", comparable=comparable
        ),
    )
    if dims:
        table.add_section()
    for dim in dims:
        table.add_row(
            Text(f"  {safe(dim)}", style="dim"),
            *_summary_cells(
                _dimension_score(agent, "with_skill", dim),
                _dimension_score(agent, "without_skill", dim),
                "{:.2f}",
                "{:+.2f}",
                weak=0.6,
                comparable=comparable,
            ),
        )
    return table


def _change(text: str, value: float | None) -> Text:
    if value is None:
        return Text("")
    return Text(f"  {text}", style="green" if value > 0 else "red" if value < 0 else "dim")


def _agent_cell(
    with_case: Mapping[str, Any] | None, without_case: Mapping[str, Any] | None, *, comparable: bool = True
) -> Text:
    with_marks, without_marks = _marks(with_case), _marks(without_case)
    cell = _marks_text(without_marks)
    cell.append(" → ", style="dim")
    cell.append_text(_marks_text(with_marks))
    (wp, wn), (op, on) = _attempt_rate(with_marks), _attempt_rate(without_marks)
    if not comparable or not wn or not on or ERROR in with_marks + without_marks:
        return cell.append("  ?", style="yellow")
    delta = 100.0 * (wp / wn - op / on)
    return cell.append_text(_change(f"{delta:+.0f}" if delta else "·", delta))


def _arrow_cell(
    with_value: float | None, without_value: float | None, fmt: str, delta_fmt: str, *, comparable: bool = True
) -> Text:
    cell = Text(" → ".join(fmt.format(v) if v is not None else "-" for v in (without_value, with_value)))
    if not comparable or with_value is None or without_value is None:
        return cell
    return cell.append_text(_change(delta_fmt.format(with_value - without_value), with_value - without_value))


def _agent_grid(
    agents: Mapping[str, Any], case_ids: list[str], dims: list[str], title: str, safe: Callable[[object], str]
) -> Table:
    """Several agents: a row per case, a column per agent, each cell without skill → with skill."""
    table = _table(title, *(Text.assemble(f"{safe(name)}\n", ("(without → with skill)", "dim")) for name in agents))
    for case_id in case_ids:
        table.add_row(
            Text(safe(case_id)),
            *(
                _agent_cell(
                    _cases(a, "with_skill").get(case_id),
                    _cases(a, "without_skill").get(case_id),
                    comparable=_case_comparable(
                        a, _cases(a, "with_skill").get(case_id), _cases(a, "without_skill").get(case_id)
                    ),
                )
                for a in agents.values()
            ),
        )
    table.add_section()
    rates = []
    for a in agents.values():
        r = _agent_rates(a)
        cell = _arrow_cell(_pct(*r["with_skill"]), _pct(*r["without_skill"]), "{:.0f}%", "{:+.0f} pp")
        complete = _comparable(a)
        rates.append(cell if complete else Text(cell.plain.split("  ")[0]).append("  ?", style="yellow"))
    table.add_row(
        Text(
            "success rate (shown)" if any(_details_truncated(a) for a in agents.values()) else "success rate",
            style="bold",
        ),
        *rates,
    )
    table.add_row(
        Text("quality score (0-1)", style="bold"),
        *(
            _arrow_cell(
                _quality(a, "with_skill"),
                _quality(a, "without_skill"),
                "{:.2f}",
                "{:+.2f}",
                comparable=_comparable(a),
            )
            for a in agents.values()
        ),
    )
    if dims:
        table.add_section()
        table.add_row(Text("dimensions", style="dim"))
    for dim in dims:
        table.add_row(
            Text(f"  {safe(dim)}", style="dim"),
            *(
                _arrow_cell(
                    _dimension_score(a, "with_skill", dim),
                    _dimension_score(a, "without_skill", dim),
                    "{:.2f}",
                    "{:+.2f}",
                    comparable=_comparable(a),
                )
                for a in agents.values()
            ),
        )
    return table


def render_case_summary(
    result: Mapping[str, Any],
    *,
    console: Console,
    fixes: list[dict[str, Any]] | None,
    safe: Callable[[object], str] | None = None,
) -> bool:
    """Print the case grid and fixes; False when the result has no per-case rows to show."""
    if safe is None:
        secrets = secret_values_from_environment(os.environ)

        def safe(value: object) -> str:
            return redact_progress_detail(value, secret_values=secrets)

    agents = {k: v for k, v in (result.get("agents") or {}).items() if isinstance(v, Mapping)}
    case_ids = _case_ids(agents)
    if not case_ids:
        return False
    label, detail = verdict(agents)
    if label:
        style = "bold green" if label == "skill helps" else "bold red" if label == "skill hurts" else "bold yellow"
        line = Text(f"{'✔' if label == 'skill helps' else '✖' if label == 'skill hurts' else '•'} {label}", style=style)
        line.append(f"   {safe(detail)}", style="dim")
        console.print(line)

    k = next(
        (
            v["k"]
            for a in agents.values()
            if isinstance(p := a.get("pass_at_k"), Mapping)
            for v in p.values()
            if isinstance(v, Mapping) and v.get("k")
        ),
        None,
    )
    names = sorted(
        {
            d
            for a in agents.values()
            for d in (a.get("dimensions_with_skill") or {})
            if isinstance(a.get("dimensions_with_skill"), Mapping)
        },
        key=lambda d: (
            min((s for a in agents.values() if (s := _dimension_score(a, "with_skill", d)) is not None), default=1.0),
            d,
        ),
    )
    grid = _agent_grid if len(agents) > 1 else _condition_grid
    table = grid(agents, case_ids, names, f"test case (k={k})" if k else "test case", safe)
    console.print(table)
    legend = f"{PASS} pass  {FAIL} fail  {ERROR} did not run  {SKIPPED} skipped after a pass"
    if len(agents) > 1:
        legend += "  ? incomplete\n+50 = change in pass rate, in percentage points"
    else:
        legend += "  (0.82) mean score 0-1"
    console.print(Text(legend, style="dim"))
    if fixes:
        console.print()
        heading = Text("How to improve", style="bold")
        heading.append(" (consider these fixes)", style="dim")
        console.print(heading)
        for i, fix in enumerate(fixes, 1):
            head = Text(f" {i}. ", style="bold")
            head.append(safe(fix["case"]), style="bold cyan")
            lost = f" · lost {fix['attempts_lost']} attempt(s)" if fix["attempts_lost"] else ""
            head.append(f"  [{safe(fix['kind'])}] · {safe(fix['scope'])}{lost}", style="dim")
            console.print(head)
            console.print(Text(f"    what happened: {safe(fix['what'])}"))
            if fix.get("action"):
                console.print(Text(f"    fix: {safe(fix['action'])}", style="green"))
            if fix.get("evidence"):
                console.print(Text(f"    evidence: {safe(fix['evidence'])}", style="dim"))
    elif fixes is not None and any(_details_truncated(a) for a in agents.values()):
        console.print(
            Text("How to improve: case/attempt details are incomplete; review the retained artifacts.", style="yellow")
        )
    elif fixes is not None:
        console.print()
        line = Text("How to improve: ", style="bold")
        line.append(
            "nothing to fix in the available case evidence.",
            style="green",
        )
        console.print(line)
    console.print()
    return True


def _load_json(path: Path, *, root: Path | None = None) -> Any:
    try:
        anchor = root if root is not None else path.parent
        with SecureRoot(anchor) as reader:
            raw, _ = reader.read_bytes(path.relative_to(anchor), _MAX_TRIAL_JSON_BYTES)
        return json.loads(raw)
    except (OSError, ValueError, SecurePathError):
        return None


def load_case_findings(run_dir: Path) -> list[dict[str, Any]] | None:
    """Read the optional advisory cache without following artifact links."""
    value = _load_json(run_dir / "what_to_fix.json", root=run_dir)
    required = {"case", "scope", "kind", "what", "attempts_lost"}
    if not isinstance(value, list) or any(not isinstance(item, dict) or not required <= item.keys() for item in value):
        return None
    return value


def save_case_findings(run_dir: Path, fixes: list[dict[str, Any]]) -> None:
    """Replace the advisory cache atomically, including an existing symlink itself."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=run_dir, prefix=".what_to_fix.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(fixes, handle, indent=2)
        temporary.replace(run_dir / "what_to_fix.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _trial_dir(run_dir: Path, agent: str, variant_dir: str, trial: object) -> Path | None:
    if (
        not isinstance(trial, str)
        or not _SAFE_TRIAL.fullmatch(trial)
        or not _SAFE_TRIAL.fullmatch(agent)
        or trial in {".", ".."}
        or agent in {".", ".."}
    ):
        return None
    path = run_dir / agent / variant_dir / "trials" / trial
    if any((run_dir / Path(*path.relative_to(run_dir).parts[:i])).is_symlink() for i in range(1, 5)):
        return None
    return path if path.is_dir() else None


def _trial_dirs(run_dir: Path, agent: str, trial: object) -> list[Path]:
    """Resolve one logical attempt to its persisted physical or native-step artifacts."""
    direct = _trial_dir(run_dir, agent, "with-skill", trial)
    if direct is not None:
        return [direct]
    if (
        not isinstance(trial, str)
        or not _SAFE_TRIAL.fullmatch(trial)
        or trial in {".", ".."}
        or not _SAFE_TRIAL.fullmatch(agent)
        or agent in {".", ".."}
    ):
        return []
    trials = run_dir / agent / "with-skill" / "trials"
    if not trials.is_dir():
        return []
    matched = []
    for candidate in sorted(islice(trials.iterdir(), _MAX_TRIAL_PATHS_SCANNED)):
        path = _trial_dir(run_dir, agent, "with-skill", candidate.name)
        if path is None:
            continue
        reward = _load_json(path / "reward.json", root=run_dir)
        if isinstance(reward, Mapping) and reward.get("trial_id") == trial:
            matched.append(path)
    return matched


def _short(text: object, limit: int = _MAX_REASON_CHARS) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _activated(reward: Mapping[str, Any]) -> bool | None:
    details = reward.get("details")
    execution = details.get("skill_execution") if isinstance(details, Mapping) else None
    activation = execution.get("activation") if isinstance(execution, Mapping) else None
    passed = activation.get("passed") if isinstance(activation, Mapping) else None
    return passed if isinstance(passed, bool) else None


def _failed_reasons(reward: Mapping[str, Any]) -> list[str]:
    details = reward.get("details") if isinstance(reward.get("details"), Mapping) else {}
    reasons: list[str] = []
    behavior = details.get("behavior_check")
    for step in behavior.get("results", []) if isinstance(behavior, Mapping) else []:
        if isinstance(step, Mapping) and step.get("passed") is False and step.get("reason"):
            reasons.append(_short(step["reason"]))
    scored = [
        (score, metric)
        for metric in _JUDGED_METRICS
        if (score := _score(reward.get(metric))) is not None and isinstance(details.get(metric), Mapping)
    ]
    for score, metric in sorted(scored):
        if score < 0.8 and details[metric].get("reason"):
            reasons.append(_short(details[metric]["reason"]))
    return reasons


def _env_errors(trial_dir: Path, run_dir: Path) -> list[str]:
    trajectory = _load_json(trial_dir / "trajectory.json", root=run_dir)
    steps = trajectory.get("steps") if isinstance(trajectory, Mapping) else None
    found: list[str] = []
    for step in steps if isinstance(steps, list) else []:
        if isinstance(step, Mapping) and step.get("source") != "user":
            found += [m.group(0) for m in _ENV_ERROR.finditer(json.dumps(step.get("observation") or "")[:200_000])]
    return [_short(e, 120) for e in dict.fromkeys(found)]


def _final_message(trial_dir: Path, run_dir: Path) -> str:
    trajectory = _load_json(trial_dir / "trajectory.json", root=run_dir)
    steps = trajectory.get("steps") if isinstance(trajectory, Mapping) else None
    for step in reversed(steps if isinstance(steps, list) else []):
        if isinstance(step, Mapping) and step.get("source") == "agent" and str(step.get("message") or "").strip():
            return _short(step["message"], 600)
    return ""


def _root_error_in_logs(trial_dir: Path, run_dir: Path) -> re.Match[str] | None:
    """Find the agent CLI's own error line in its log (failure.json truncates long messages)."""
    for log in sorted(trial_dir.glob("*.txt")):
        try:
            with SecureRoot(run_dir) as reader:
                raw, _ = reader.read_bytes(log.relative_to(run_dir), _MAX_TRIAL_JSON_BYTES)
            text = raw[-_MAX_LOG_TAIL_BYTES:].decode("utf-8", "replace")
        except (OSError, SecurePathError):
            continue
        plain = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("{"))
        if match := _ROOT_ERROR.search(plain):
            return match
    return None


def _run_error(trial_dir: Path, run_dir: Path) -> str:
    failure = _load_json(trial_dir / "failure.json", root=run_dir)
    error = failure.get("error") if isinstance(failure, Mapping) else None
    if not isinstance(error, Mapping):
        return ""
    message = str(error.get("message") or "")
    # Agent exit errors embed the whole command line; the root cause is the CLI's own error line.
    root = _root_error_in_logs(trial_dir, run_dir) or _ROOT_ERROR.search(message)
    detail = root.group(1) if root else message.split("\nstdout:", 1)[0]
    # Setup-command failures embed the script; its stdout carries the actual cause (e.g. a curl error).
    if detail.startswith("Command failed") and (stdout := _STDOUT.search(message)):
        detail = stdout.group(1)
    return _short(f"{error.get('type', 'error')}: {detail}", 160)


def _diagnose_agent_case(
    run_dir: Path, agent_name: str, agent: Mapping[str, Any], case_id: str
) -> dict[str, Any] | None:
    with_case = _cases(agent, "with_skill").get(case_id)
    with_marks, without_marks = _marks(with_case), _marks(_cases(agent, "without_skill").get(case_id))
    wp, wn = _attempt_rate(with_marks)
    bp, bn = _attempt_rate(without_marks)
    errors = with_marks.count(ERROR)
    not_used = _never_activated(agent)
    failed_trials = []
    for attempt in with_case.get("attempts", []) if isinstance(with_case, Mapping) else []:
        if isinstance(attempt, Mapping) and attempt.get("passed") is not True:
            failed_trials.extend(
                (attempt.get("score"), d) for d in _trial_dirs(run_dir, agent_name, attempt.get("trial"))
            )
    errored_trials = []
    with_trials = run_dir / agent_name / "with-skill" / "trials"
    if errors and with_trials.is_dir() and _SAFE_TRIAL.match(agent_name):
        errored_trials = [
            d
            for d in sorted(with_trials.iterdir())
            if d.name.split("__", 1)[0] == case_id
            and _trial_dir(run_dir, agent_name, "with-skill", d.name) is not None
            and (d / "failure.json").is_file()
        ]

    lost = (wn - wp) + errors
    hurts = (
        _case_comparable(agent, with_case, _cases(agent, "without_skill").get(case_id))
        and bn
        and wn
        and bp / bn > wp / wn
    )
    if not lost and not hurts and not not_used:
        return _goal_not_met(run_dir, agent_name, case_id, with_case)
    rewards = [
        (r, d) for _, d in failed_trials if isinstance(r := _load_json(d / "reward.json", root=run_dir), Mapping)
    ]
    reasons = Counter(reason for reward, _ in rewards for reason in _failed_reasons(reward))
    env = list(dict.fromkeys(e for _, d in failed_trials for e in _env_errors(d, run_dir)))
    activations = [_activated(reward) for reward, _ in rewards]
    finals = [_final_message(d, run_dir) for _, d in failed_trials]
    evidence_dir = (failed_trials or [(None, d) for d in errored_trials] or [(None, None)])[0][1]

    if not_used:
        kind, what = "skill not used", "the skill never activated; scores are diagnostic only"
        lost = max(lost, wn)
    elif hurts and wp < wn:
        kind, what = "skill hurts", f"passes {bp}/{bn} without the skill but {wp}/{wn} with it"
        if reasons:
            what += f"; {reasons.most_common(1)[0][0]}"
        lost = max(lost, round(bp / bn * wn) - wp)
    elif errors and not failed_trials:
        detail = next((e for d in errored_trials if (e := _run_error(d, run_dir))), "")
        kind, what = "run error", f"{errors} attempt(s) never produced a score" + (f": {detail}" if detail else "")
    elif finals and all(_ASKS_FOR_INPUT.search(f) for f in finals):
        kind = "missing input"
        what = f'the agent stopped to ask for input the case does not provide: "{_short(finals[0], 140)}"'
    elif activations and all(a is False for a in activations):
        kind, what = "skill not used", "the agent never loaded the skill in the failed attempts"
        if reasons:
            what += f"; {reasons.most_common(1)[0][0]}"
    elif env:
        kind, what = "environment", f"tool/dependency errors in the sandbox: {'; '.join(env[:2])}"
    else:
        kind = "skill"
        what = "; ".join(r for r, _ in reasons.most_common(2)) or "attempts scored below the pass threshold"
    return {
        "case": case_id,
        "agent": agent_name,
        "kind": kind,
        "what": _short(what, 320),
        "attempts_lost": lost,
        "reasons": [r for r, _ in reasons.most_common(6)],
        "env_errors": env[:4],
        "final_message": _final_message(failed_trials[0][1], run_dir) if failed_trials else "",
        "evidence": _evidence(run_dir, evidence_dir, run_error=kind == "run error"),
    }


def _goal_not_met(
    run_dir: Path, agent_name: str, case_id: str, with_case: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Passing attempts whose judge scored goal accuracy 0: other metrics carried the score, the task wasn't done."""
    attempts = with_case.get("attempts") if isinstance(with_case, Mapping) else None
    passed = [a for a in attempts or [] if isinstance(a, Mapping) and a.get("passed") is True]
    misses = []
    for attempt in passed:
        rows = [
            (reward, d)
            for d in _trial_dirs(run_dir, agent_name, attempt.get("trial"))
            if isinstance(reward := _load_json(d / "reward.json", root=run_dir), Mapping)
        ]
        goals = [_score(reward.get("goal_accuracy")) for reward, _ in rows]
        # Canonical reports average complete step metrics into one logical trial.
        if goals and all(goal is not None and 0 <= goal <= 1 for goal in goals) and sum(goals) == 0:
            misses.append(rows[0])
    if not misses:
        return None
    details = misses[0][0].get("details")
    goal = details.get("goal_accuracy") if isinstance(details, Mapping) else None
    goal = goal if isinstance(goal, Mapping) else {}
    outcome = goal.get("end_state") or goal.get("reason") or ""
    finals = [_final_message(d, run_dir) for _, d in misses]
    what = "passed on the other metrics but scored goal accuracy 0.00"
    return {
        "case": case_id,
        "agent": agent_name,
        "kind": "goal not met",
        "what": _short(what + (f": {outcome}" if outcome else ""), 320),
        "attempts_lost": len(misses),
        "reasons": [_short(goal["reason"])] if goal.get("reason") else [],
        "env_errors": [],
        "final_message": finals[0],
        "asks_for_input": all(_ASKS_FOR_INPUT.search(f) for f in finals) or bool(_STOPPED_FOR_INPUT.search(outcome)),
        "evidence": _evidence(run_dir, misses[0][1], run_error=False),
    }


def goal_not_met_count(result: Mapping[str, Any], *, run_dir: Path) -> int:
    """Count every observed passing attempt with goal accuracy zero, independently of the short fix list."""
    agents = result.get("agents")
    if not isinstance(agents, Mapping):
        return 0
    return sum(
        issue["attempts_lost"]
        for name, agent in agents.items()
        if isinstance(agent, Mapping)
        for case_id, case in _cases(agent, "with_skill").items()
        if (issue := _goal_not_met(run_dir, str(name), str(case_id), case)) is not None
    )


def _evidence(run_dir: Path, trial_dir: Path | None, *, run_error: bool) -> str:
    if trial_dir is None:
        return ""
    names = ("exception.txt", "failure.json") if run_error else ("trajectory.json",)
    name = next((n for n in names if (trial_dir / n).is_file()), names[-1])
    return (trial_dir.relative_to(run_dir) / name).as_posix()


def _fallback_action(issue: Mapping[str, Any]) -> str:
    kind = issue["kind"]
    if kind == "skill not used":
        request = f' such as "{_short(issue["prompt"], 120)}"' if issue.get("prompt") else ""
        return (
            f"Widen the SKILL.md frontmatter description to cover requests{request}; "
            "the agent decides from the description alone whether to load the skill."
        )
    if kind == "environment":
        return "Install the missing tool or package in evals/environment/Dockerfile, or document the install step in SKILL.md."
    if kind == "run error":
        return "Not a skill problem: re-run this case; if it repeats, check the agent/gateway logs."
    if kind == "missing input":
        return (
            "Not a skill problem: add the files or app the prompt refers to (evals/files or "
            "evals/environment/Dockerfile), or make the prompt self-contained."
        )
    if kind == "eval case":
        return (
            "Not a skill problem: rewrite this prompt in evals/evals.json as a concrete user request "
            "(the input and the output wanted), so the case actually exercises the skill."
        )
    if kind == "goal not met" and issue.get("asks_for_input"):
        return (
            "The agent stopped to ask for input instead of producing the output: add the files the prompt refers to "
            "(evals/files) or put the needed details in the prompt, or have SKILL.md proceed with marked placeholders."
        )
    if kind == "skill hurts":
        return "Check whether SKILL.md steers the agent away from an approach that already works for this request."
    return f"Address in SKILL.md: {issue['reasons'][0]}" if issue.get("reasons") else ""


def _llm_actions(
    issues: list[dict[str, Any]], skill_path: Path | None, dataset: Mapping[str, Mapping[str, Any]]
) -> list[str] | None:
    skill_md = ""
    files: list[str] = []
    if skill_path and (skill_path / "SKILL.md").is_file():
        skill_md = (skill_path / "SKILL.md").read_text(encoding="utf-8", errors="replace")[:_MAX_SKILL_MD_CHARS]
        files = sorted(
            str(p.relative_to(skill_path)) for p in skill_path.rglob("*") if p.is_file() and ".git" not in p.parts
        )[:80]
    blocks = []
    for i, issue in enumerate(issues, 1):
        entry = dataset.get(issue["case"], {})
        expected = entry.get("expected_behavior") or entry.get("assertions") or []
        blocks.append(
            f"ISSUE {i} [{issue['kind']}] case {issue['case']} ({issue['scope']})\n"
            f"prompt: {_short(entry.get('question') or entry.get('prompt'), 500)}\n"
            f"expected: {_short(entry.get('ground_truth') or entry.get('expected_output'), 400)}\n"
            f"expected steps: {_short('; '.join(map(str, expected)), 500)}\n"
            f"judge reasons for failure: {' | '.join(issue['reasons'][:5]) or '(none)'}\n"
            f"sandbox errors: {' | '.join(issue['env_errors']) or '(none)'}\n"
            f"agent final answer: {issue['final_message'] or '(none)'}"
        )
    prompt = (
        "You help a skill author fix an agent skill after a live evaluation. For each issue give ONE concrete fix, "
        "at most 35 words, naming the file and what to change (SKILL.md section, a script, "
        "evals/environment/Dockerfile, or the eval case in evals/evals.json if the case expects something the skill "
        "does not provide). For [skill not used] issues the fix belongs in the SKILL.md frontmatter description, "
        "which is all the agent sees when deciding whether to load the skill. "
        "Write fixes that cover the whole class of requests; do not copy names or details from this one case. "
        "Base it only on the evidence; do not invent features.\n\n"
        f"SKILL FILES: {', '.join(files) or '(unknown)'}\n\nSKILL.md:\n{skill_md or '(unavailable)'}\n\n"
        + "\n\n".join(blocks)
        + "\n\nRespond with ONLY a JSON array of strings, one fix per issue, in order."
    )
    try:
        from skillevaluator.tier3.eval_core.llm_judge import _extract_json, call_public_llm

        redactor = Redactor(secret_values_from_environment(os.environ))
        prompt = redact_sensitive_text(redactor.text(prompt))
        content, error = call_public_llm(prompt, max_tokens=4096, timeout=120)
        parsed = _extract_json(content) if content and not error else None
    except Exception:  # advisory: fall back to rule-based actions
        logger.debug("What-to-fix generation failed", exc_info=True)
        return None
    if isinstance(parsed, list) and len(parsed) == len(issues) and all(isinstance(p, str) for p in parsed):
        return [_short(p, 400) for p in parsed]
    return None


def _dataset_entries(run_dir: Path) -> dict[str, Mapping[str, Any]]:
    snapshot = _load_json(run_dir / "dataset_snapshot.json", root=run_dir)
    entries = snapshot.get("dataset") if isinstance(snapshot, Mapping) else None
    return {str(e["id"]): e for e in entries or [] if isinstance(e, Mapping) and e.get("id") is not None}


_MERGED_KINDS = ("run error", "skill not used", "eval case", "missing input", "goal not met")


def _merge_across_cases(issues: list[dict[str, Any]], n_cases: int) -> list[dict[str, Any]]:
    """Collapse one cause repeated across cases (same error type, or same kind and scope) into one item."""
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    rest = []
    for issue in issues:
        if issue["kind"] not in _MERGED_KINDS:
            rest.append(issue)
            continue
        cause = issue["what"].split(": ", 1)[-1]
        key = (issue["kind"], cause.split(":", 1)[0] if issue["kind"] == "run error" else issue["scope"])
        if (seen := merged.get(key)) is None:
            merged[key] = dict(issue, cases=[issue["case"]], reasons=list(issue["reasons"]))
        else:
            seen["cases"].append(issue["case"])
            seen["attempts_lost"] += issue["attempts_lost"]
            seen["reasons"] += [r for r in issue["reasons"] if r not in seen["reasons"]]
    for issue in merged.values():
        if len(issue["cases"]) > 1:
            issue["case"] = f"all {n_cases} cases" if len(set(issue["cases"])) == n_cases else ", ".join(issue["cases"])
            if issue["kind"] == "run error":
                cause = issue["what"].split(": ", 1)[-1]
                issue["what"] = f"{issue['attempts_lost']} attempt(s) never produced a score: {cause}"
    return rest + list(merged.values())


def _skill_description(skill_path: Path | None) -> str:
    skill_md = skill_path / "SKILL.md" if skill_path else None
    if skill_md is None or not skill_md.is_file():
        return ""
    match = re.match(r"---\s*\n(.*?)\n---", skill_md.read_text(encoding="utf-8", errors="replace"), re.DOTALL)
    try:
        import yaml

        meta = yaml.safe_load(match.group(1)) if match else None
    except Exception:  # malformed frontmatter: Tier 1 reports it
        return ""
    return str(meta.get("description") or "") if isinstance(meta, Mapping) else ""


def _restates(prompt: str, description: str) -> bool:
    """True when the eval prompt pastes in a long run of the skill description instead of posing a task."""
    a, b = prompt.lower(), description.lower()
    if len(a) < _RESTATED_CHARS or len(b) < _RESTATED_CHARS:
        return False
    return SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b)).size >= _RESTATED_CHARS


def what_to_fix(
    result: Mapping[str, Any], *, run_dir: Path, skill_path: Path | None = None, use_llm: bool = True
) -> list[dict[str, Any]]:
    """Rank failed test cases across agents by attempts lost and attach one fix each."""
    agents = {str(k): v for k, v in (result.get("agents") or {}).items() if isinstance(v, Mapping)}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for case_id in _case_ids(agents):
        for name, agent in agents.items():
            if issue := _diagnose_agent_case(run_dir, name, agent, case_id):
                grouped.setdefault((case_id, issue["kind"]), []).append(issue)
    issues = []
    for items in grouped.values():
        names = [item["agent"] for item in items]
        merged = dict(items[0])
        merged.update(
            scope=", ".join(names) + " only"
            if len(agents) > 1 and len(names) < len(agents)
            else "all agents"
            if len(agents) > 1
            else names[0],
            attempts_lost=sum(item["attempts_lost"] for item in items),
            reasons=list(dict.fromkeys(r for item in items for r in item["reasons"])),
            env_errors=list(dict.fromkeys(e for item in items for e in item["env_errors"])),
        )
        issues.append(merged)
    dataset = _dataset_entries(run_dir)
    description = _skill_description(skill_path)
    for issue in issues:
        entry = dataset.get(issue["case"], {})
        prompt = str(entry.get("question") or entry.get("prompt") or "")
        issue["prompt"] = prompt
        if issue["kind"] != "run error" and _restates(prompt, description):
            issue["kind"] = "eval case"
            issue["what"] = "the test prompt repeats the skill description instead of asking for a concrete task"
    issues = _merge_across_cases(issues, len(_case_ids(agents)))
    if saturated := _baseline_saturated(agents):
        issues.append(saturated)
    issues.sort(key=lambda issue: (issue["kind"] == "run error", -issue["attempts_lost"], issue["case"]))
    issues = issues[:_MAX_FIXES]
    # These kinds have a fixed remedy; the LLM only writes fixes that depend on the skill's content.
    actionable = [
        issue
        for issue in issues
        if issue["kind"] not in ("run error", "eval case", "missing input", "skill not used")
        and not issue.get("asks_for_input")
    ]
    actions = _llm_actions(actionable, skill_path, dataset) if use_llm and actionable else None
    by_id = dict(zip((id(i) for i in actionable), actions or [], strict=False))
    for issue in issues:
        issue["action"] = issue.get("action") or by_id.get(id(issue)) or _fallback_action(issue)
    secrets = secret_values_from_environment(os.environ)
    return [
        {
            key: redact_progress_detail(value, secret_values=secrets)
            if isinstance(value, str)
            else [redact_progress_detail(item, secret_values=secrets) for item in value]
            if isinstance(value, list)
            else value
            for key, value in issue.items()
        }
        for issue in issues
    ]


def _baseline_saturated(agents: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """Flag agents that pass every case without the skill: the eval cannot show what the skill adds."""
    names = []
    for name, agent in agents.items():
        if _details_truncated(agent) or _case_mismatch(agent) or _execution_unavailable(agent):
            continue
        cases = _cases(agent, "without_skill").values()
        passed, total = _agent_rates(agent)["without_skill"]
        if total and passed == total and not any(ERROR in _marks(case) for case in cases):
            names.append(name)
    if not names:
        return None
    return {
        "case": "all cases",
        "agent": names[0],
        "kind": "eval case",
        "scope": "all agents" if len(agents) > 1 and len(names) == len(agents) else ", ".join(names),
        "attempts_lost": 0,
        "what": "every case also passes without the skill, so this eval cannot show what the skill adds",
        "reasons": [],
        "env_errors": [],
        "evidence": "",
        "action": (
            "Add cases the agent fails without the skill (tasks that need the skill's scripts, "
            "conventions or domain rules) so the eval measures the skill's value."
        ),
    }
