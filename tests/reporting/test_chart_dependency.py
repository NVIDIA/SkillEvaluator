# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated reports must not execute a mutable, unchecked CDN dependency."""

from html.parser import HTMLParser

from skillevaluator.evaluation.tier3_report import build_agent_eval_payload
from skillevaluator.models import ValidationResult
from skillevaluator.reporting import HTMLReporter


class _Scripts(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.external: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "script" and attributes.get("src"):
            self.external.append(attributes)


def test_generated_tier3_report_pins_and_checks_chart_dependency() -> None:
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.metadata["agent_eval"] = build_agent_eval_payload(
        "example",
        {
            "agent": {
                "execution_status": "succeeded",
                "expected_attempts": 1,
                "scored_attempts": 1,
                "with_skill": {"security": 1.0},
                "rewards": [{"entry_id": "case-1", "security": 1.0}],
            }
        },
        use_llm_judge=False,
    )
    parser = _Scripts()
    parser.feed(HTMLReporter(include_timestamp=False).render_all([result]))
    chart_scripts = [script for script in parser.external if "chart.js" in (script["src"] or "")]
    assert len(chart_scripts) == 1
    script = chart_scripts[0]
    assert script["src"] == "https://cdn.jsdelivr.net/npm/chart.js@4.5.1/dist/chart.umd.min.js"
    assert script.get("integrity") == ("sha384-jb8JQMbMoBUzgWatfe6COACi2ljcDdZQ2OxczGA3bGNeWe+6DChMTBJemed7ZnvJ")
    assert script.get("crossorigin") == "anonymous"
