# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only joins between retained discovery, proposed edits and validation runs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from skillevaluator.reporting.dashboard_data import _read_native_json, _safe_native_path
from skillevaluator.tier3.improve import _digest, replay_proposal, retained_task_roots, snapshot_tree


def _runtime_digest(path: Path) -> str:
    from skillevaluator.tier3.harbor.adapter import _runtime_skill_copy_ignore
    from skillevaluator.tier3.harbor.secure_copy import tree_content_fingerprint_secure

    return tree_content_fingerprint_secure(path, allowed_root=path, ignore=_runtime_skill_copy_ignore(path))


def _source_matches(run: Path, skill: str, expected: str, expected_cases: set[tuple[str, str]]) -> bool:
    """Require complete agent/case coverage before checking every staged source."""
    tasks = [path for path in retained_task_roots(run) if path.parent.name == "with"]
    identities = []
    for task in tasks:
        entry = _read_native_json(task / "tests" / "entry.json", run)
        case_id = next(
            (entry[key] for key in ("id", "entry_id", "case_id", "task_id") if entry.get(key) is not None), ""
        )
        identities.append((task.parent.parent.name, str(case_id)))
    return (
        bool(expected and expected_cases)
        and len(identities) == len(expected_cases)
        and set(identities) == expected_cases
        and _read_native_json(run / "result.json", run).get("skill_name") == skill
        and all(
            _safe_native_path(source := task / "environment" / "skills" / skill, run)
            and source.is_dir()
            and _runtime_digest(source) == expected
            for task in tasks
        )
    )


def load_skill_change(
    proposal_dir: Path, before: Path | None = None, after: Path | None = None, *, outcome_metrics: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Only user-selected roots are read; embedded artifact paths are never followed."""
    from skillevaluator.tier3.improvement_signals import compare_signals, load_signals

    root = proposal_dir.expanduser().resolve()
    receipt = _read_native_json(root / "receipt.json", root)
    packet = (
        _read_native_json(root / "evidence.json", root)
        if (root / "evidence.json").exists()
        else {"evaluation": {"agents": {}}, "skill_files": {}}
    )
    proposal = _read_native_json(root / "edits.json", root) if (root / "edits.json").exists() else {"edits": []}
    findings = {
        finding["id"]: {"agent": agent_name, "case": case["entry_id"], "trial": attempt["trial_id"], **finding}
        for agent_name, agent in packet["evaluation"]["agents"].items()
        for case in agent["cases"]
        for attempt in case["attempts"]
        for finding in [*attempt["findings"], *attempt.get("agent_observations", [])]
    }
    issues = []
    if receipt.get("evaluated_source_verified") is not True or packet.get("source_verified") is not True:
        issues.append("The proposal does not record a verified discovery source.")
    edits = proposal["edits"]
    if not isinstance(edits, list):
        raise ValueError("Proposal must contain an edits list")
    if edits and not (
        receipt.get("source_digest")
        and receipt["source_digest"] == packet.get("source_digest") == proposal.get("source_digest")
    ):
        raise ValueError("Proposal and discovery source identities differ")
    pending, diff = replay_proposal(
        packet["skill_files"], proposal, source_digest=packet["source_digest"], evidence_ids=findings,
    ) if edits else ({}, "")
    changed = sorted(pending)
    skill = receipt.get("evaluated_skill_name")
    if not isinstance(skill, str) or not skill or Path(skill).name != skill or skill in {".", ".."}:
        raise ValueError("Receipt has no valid evaluated skill identity")
    candidate = root / "candidate" / skill
    candidate_digest = None
    try:
        if not _safe_native_path(candidate, root) or not candidate.is_dir():
            raise ValueError("Retained candidate is unavailable")
        snapshot = snapshot_tree(candidate, include_generated=True)
        if any(item["kind"] not in {"file", "dir"} for item in snapshot.values()):
            raise ValueError("Retained candidate contains links or special files")
        if _digest(snapshot) != receipt.get("candidate_digest"):
            raise ValueError("Retained candidate differs from its proposal receipt")
        for name in changed:
            if snapshot.get(name, {}).get("digest") != hashlib.sha256(pending[name].encode()).hexdigest():
                raise ValueError("Retained candidate differs from the proposed replacements")
        if changed != receipt.get("changed_files"):
            raise ValueError("Changed files differ from the proposal receipt")
        candidate_digest = _runtime_digest(candidate)
    except (OSError, ValueError, RuntimeError) as exc:
        issues.append(str(exc))
    data = {
        "receipt": receipt,
        "edits": edits,
        "findings": findings,
        "diff": diff,
        "issues": issues,
        "association": "Not evaluated",
        "comparison": None,
        "runs": {},
    }
    if bool(before) != bool(after):
        raise ValueError("Select both original and candidate validation runs")
    if before and after:
        runs = {"before": before.expanduser().resolve(), "after": after.expanduser().resolve()}
        data["runs"] = {label: load_signals(run) for label, run in runs.items()}
        for label, run in runs.items():
            expected = receipt.get("runtime_digest") if label == "before" else candidate_digest
            expected_cases = {
                (agent, case["entry_id"])
                for agent, result in data["runs"][label]["agents"].items()
                for case in result["cases"]
            }
            try:
                if not expected or not _source_matches(run, skill, expected, expected_cases):
                    issues.append(
                        f"{label}: retained runtime source is missing or differs from this proposal. "
                        "Identity includes file contents and executable permissions, while excluding evaluation files."
                    )
            except (OSError, ValueError, RuntimeError) as exc:
                issues.append(f"{label}: source association unavailable: {exc}")
        data["association"] = "Verified runtime source match" if not issues else "Unverified source association"
        data["comparison"] = compare_signals(
            data["runs"]["before"], data["runs"]["after"], outcome_metrics=outcome_metrics
        )
    return data


def show_skill_changes() -> None:
    import streamlit as st

    st.caption(
        "Follow discovery evidence → proposed changes → validation results. Saved files only; no agents run here."
    )
    with st.form("skill_change_inputs"):
        proposal = st.text_input("Proposal directory", key="change_proposal")
        left, right = st.columns(2)
        before = left.text_input("Original validation run (optional)", key="change_before")
        after = right.text_input("Candidate validation run (optional)", key="change_after")
        metrics = st.text_input(
            "Outcome metrics (comma-separated, optional)",
            key="change_metrics",
            help="Same as CLI --outcome-metric. Labels outcomes; it does not change the strict verdict.",
        )
        if st.form_submit_button("Load skill changes"):
            st.session_state.pop("skill_change_data", None)
            try:
                if not proposal.strip():
                    raise ValueError("Choose a proposal directory")
                st.session_state["skill_change_data"] = load_skill_change(
                    Path(proposal),
                    Path(before) if before.strip() else None,
                    Path(after) if after.strip() else None,
                    outcome_metrics=tuple(dict.fromkeys(item.strip() for item in metrics.split(",") if item.strip())),
                )
            except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError, RecursionError) as exc:
                st.error("Cannot load this evidence.")
                st.text(str(exc))
    data = st.session_state.get("skill_change_data")
    if not data:
        return
    receipt = data["receipt"]
    st.subheader("Proposal")
    st.text(f"Status: {receipt.get('status', 'unknown')} · Editor invoked: {receipt.get('editor_invoked', 'unknown')}")
    st.text(receipt.get("proposal_summary") or "No proposal explanation recorded.")
    if receipt.get("proposal_summary_truncated"):
        st.caption("Receipt explanation is shortened; the full proposal remains in edits.json.")
    st.text(receipt.get("error") or receipt.get("next_step") or "")
    st.caption(
        "Loaded snapshot. Load again to refresh. Source matching does not prove the changed helper executed or caused a gain."
    )
    for issue in data["issues"]:
        st.warning("Source association needs review.")
        st.text(issue)
    for index, edit in enumerate(data["edits"]):
        with st.expander(f"Change {index + 1}", expanded=True):
            st.text(edit["path"])
            st.text(edit.get("reason", ""))
            for ref in edit["evidence_ids"]:
                finding = data["findings"][ref]
                st.text(f"Discovery evidence: {ref}")
                st.json(finding, expanded=False)
    if data["diff"]:
        st.code(data["diff"], language="diff")
    comparison = data["comparison"]
    if not comparison:
        st.caption("No validation pair selected. A proposal alone is not an improvement.")
        return
    st.subheader("Validation")
    st.text(data["association"])
    st.text(f"Native verdict: {comparison['verdict']}")
    for issue in comparison["issues"]:
        st.warning("Comparison is incomplete or incompatible.")
        st.text(issue)
    st.caption(comparison["interpretation"])
    st.caption(
        "Outcome gains and all metric declines remain visible. Metric labels do not establish guardrail compliance."
    )
    with st.expander("Run summaries, lift and activation"):
        for agent, result in comparison["agents"].items():
            original_model = data["runs"]["before"]["agents"][agent].get("model") or "Not recorded"
            candidate_model = data["runs"]["after"]["agents"][agent].get("model") or "Not recorded"
            st.text(f"{agent} · Original model: {original_model} · Candidate model: {candidate_model}")
            st.dataframe(
                [{"Metric": metric, **values} for metric, values in result["aggregate"].items()], hide_index=True
            )
            st.json(result["skill_activation"], expanded=False)
    cases = [(agent, case) for agent, result in comparison["agents"].items() for case in result["cases"]]
    counts = {
        (label, agent, case["entry_id"]): len(case["attempts"])
        for label, run in data["runs"].items()
        for agent, result in run["agents"].items()
        for case in result["cases"]
    }
    st.dataframe(
        [
            {
                "Agent": agent,
                "Case": case["entry_id"],
                "Original attempts": counts["before", agent, case["entry_id"]],
                "Candidate attempts": counts["after", agent, case["entry_id"]],
                "Signal status": case["status"],
                "Outcome gains": ", ".join(case["outcome_gains"]) or "None"
                if case["outcome_gains"] is not None
                else "Not assessed",
                "Outcome regressions": ", ".join(case["outcome_regressions"]) or "None"
                if case["outcome_regressions"] is not None
                else "Not assessed",
                "Other declines": ", ".join(case["other_metric_declines"]),
            }
            for agent, case in cases
        ],
        hide_index=True,
        width="stretch",
    )
    if cases:
        identities = [(agent, case["entry_id"]) for agent, case in cases]
        selected = st.selectbox(
            "Validation case",
            identities,
            format_func=lambda item: f"{item[0]} / {item[1]}",
            key="change_case",
        )
        agent, case = cases[identities.index(selected)]
        st.dataframe([{"Metric": metric, **values} for metric, values in case["metrics"].items()], hide_index=True)
        for label, run in data["runs"].items():
            matched = next(item for item in run["agents"][agent]["cases"] if item["entry_id"] == case["entry_id"])
            with st.expander(f"{label.capitalize()} attempt evidence"):
                st.json(matched, expanded=False)
        st.caption(
            "Activation means the skill was loaded, not proof that its changed helper was executed. Findings are observations, not established causes."
        )
