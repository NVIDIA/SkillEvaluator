# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Describe Tier 3 agent arms that ran but did not complete.

Each Tier 3 agent runs a with-skill arm and a no-skill baseline arm. When one
arm fails, for example because an attempt hit the agent timeout, the other arm
keeps its scores and the failed arm has none. Reports must not read that gap
as an arm that never ran. These helpers read the per-condition counters the
collector records (``conditions.<arm>.execution_status``, ``scored_attempts``
and ``expected_attempts``), so the JSON payload, the HTML report, the terminal
and ``BENCHMARK.md`` describe the gap in the same words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_ARM_LABELS = {"with_skill": "with-skill run", "without_skill": "baseline"}
_NOT_REPORTED = {
    "with_skill": "With-skill scores and uplift are not reported for this agent.",
    "without_skill": "Baseline and uplift are not reported for this agent.",
}
# The code names the arm whose recorded status is ``failed``; ``state`` says how
# many of its attempts scored.
_NOTICE_CODES = {"with_skill": "with_skill_failed", "without_skill": "baseline_failed"}
# Harbor records a failed trial as ``ExceptionType: message`` after the trial
# label. A job whose statistics miss a trial names the trial or step exception
# Harbor recorded as ``trial <label> recorded ExceptionType: message``. Only the
# type name is kept, as plain words, so it is safe to publish.
_EXCEPTION_TYPE = re.compile(
    r"(?:^|: | recorded )(?:[A-Za-z_]\w{0,63}\.){0,8}([A-Z][A-Za-z0-9]{0,63}(?:Error|Exception))(?=:|$)"
)
# An acronym keeps its trailing digits: ``S3UploadFailedError`` gives ``s3 upload
# failed error`` and ``HTTP2Error`` gives ``http2 error``.
_CAMEL_WORD = re.compile(r"[A-Z]+[0-9]*(?![a-z])|[A-Z][a-z0-9]*")
# The collector's case-coverage diagnostics list case IDs after the colon, and a
# case ID can look like an exception type. They never name a failure.
_CASE_LIST_DIAGNOSTICS = (
    "Missing scored attempts for cases",
    "Excess scored attempts for cases",
    "Unexpected scored cases",
)
_MAX_COUNT = 2**63 - 1


@dataclass(frozen=True)
class IncompleteArm:
    """One agent arm that ran but failed, with how many of its attempts scored.

    ``str()`` gives the short form used in report cells, such as
    ``baseline incomplete (1 of 2 attempts scored)``.
    """

    condition: str
    scored: int | None
    expected: int | None
    failure: str | None = None
    agent_failed: bool = False

    @property
    def counts_known(self) -> bool:
        """Whether the counters are usable: attempts were expected and the scored count does not exceed them."""
        if self.scored is None or self.expected is None:
            return False
        return self.expected > 0 and self.scored <= self.expected

    @property
    def state(self) -> str:
        """``incomplete`` (some attempts scored), ``unavailable`` (none scored) or ``failed``.

        ``failed`` means no attempt is missing, or the counters are unusable.
        """
        if not self.counts_known or self.scored == self.expected:
            return "failed"
        return "incomplete" if self.scored else "unavailable"

    def __str__(self) -> str:
        counts = f" ({self.scored} of {self.expected} attempts scored)" if self.counts_known else ""
        return f"{_ARM_LABELS[self.condition]} {self.state}{counts}"

    def describe(self) -> str:
        """Return the arm, its state, its counts and the first recorded failure, then what is not reported.

        When the agent's recorded status is ``failed``, the text also says that
        the agent and the Tier 3 run are marked failed: any failed agent fails
        the run.
        """
        details = [f"{self.scored} of {self.expected} attempts scored"] if self.counts_known else []
        if self.failure:
            details.append(f"first failure: {self.failure}")
        detail_text = f" ({'; '.join(details)})" if details else ""
        outcome = " The agent and the Tier 3 run are marked failed." if self.agent_failed else ""
        return f"{_ARM_LABELS[self.condition]} {self.state}{detail_text}.{outcome} {_NOT_REPORTED[self.condition]}"


def incomplete_arms(agent: object) -> list[IncompleteArm]:
    """Return the agent's arms whose recorded ``execution_status`` is ``failed``.

    A skipped arm (``--skip-baseline`` or ``--skip-with-skill``) did not run,
    so it is not incomplete. Missing or malformed condition records give no
    arms rather than a guess.
    """
    if not isinstance(agent, dict) or not isinstance(conditions := agent.get("conditions"), dict):
        return []
    agent_failed = agent.get("execution_status") == "failed"
    arms = []
    for condition in _ARM_LABELS:
        record = conditions.get(condition)
        if not isinstance(record, dict) or record.get("execution_status") != "failed":
            continue
        arms.append(
            IncompleteArm(
                condition,
                _count(record.get("scored_attempts")),
                _count(record.get("expected_attempts")),
                _first_failure(record.get("execution_errors")),
                agent_failed,
            )
        )
    return arms


def coverage_notices(agents: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one notice per incomplete arm, in agent order, for the canonical payload."""
    notices = []
    for name, agent in agents.items():
        for arm in incomplete_arms(agent):
            notices.append(
                {
                    "code": _NOTICE_CODES[arm.condition],
                    "agent": name,
                    "condition": arm.condition,
                    "state": arm.state,
                    "scored_attempts": arm.scored if arm.counts_known else None,
                    "expected_attempts": arm.expected if arm.counts_known else None,
                    "failure": arm.failure,
                    "message": f"{name} {arm.describe()}",
                }
            )
    return notices


def _count(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= _MAX_COUNT else None


def _first_failure(errors: object) -> str | None:
    """Name the first Harbor exception type in an arm's errors, in words (``agent timeout error``)."""
    if not isinstance(errors, list):
        return None
    for error in errors:
        if not isinstance(error, str) or error.startswith(_CASE_LIST_DIAGNOSTICS):
            continue
        if match := _EXCEPTION_TYPE.search(error):
            return " ".join(word.lower() for word in _CAMEL_WORD.findall(match.group(1)))
    return None
