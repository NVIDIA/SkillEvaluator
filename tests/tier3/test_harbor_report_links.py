# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from skillevaluator.tier3.harbor.html_report import (
    _build_agent_findings,
    _build_dataset_html,
    _build_suggestions_html,
    generate_html_report,
)
from skillevaluator.tier3.harbor.report import add_evidence_links_to_suggestions


def test_add_evidence_links_to_suggestions_uses_step_link_when_available() -> None:
    suggestions = ["Add a safety guardrail for destructive cleanup."]
    rewards = [
        {
            "harbor_viewer": {
                "trial_url": "https://viewer/jobs/job/tasks/case/trials/trial",
                "evidence_urls": [
                    {
                        "label": "security",
                        "url": "https://viewer/jobs/job/tasks/case/trials/trial?step=4",
                    }
                ],
            }
        }
    ]

    linked = add_evidence_links_to_suggestions(suggestions, rewards)

    assert linked == [
        "Add a safety guardrail for destructive cleanup. Evidence: "
        "https://viewer/jobs/job/tasks/case/trials/trial?step=4"
    ]


def test_add_evidence_links_to_suggestions_falls_back_to_trial_link() -> None:
    suggestions = ["Review the failing behavior check."]
    rewards = [
        {
            "harbor_viewer": {
                "trial_url": "https://viewer/jobs/job/tasks/case/trials/trial",
                "evidence_urls": [],
            }
        }
    ]

    linked = add_evidence_links_to_suggestions(suggestions, rewards)

    assert linked == ["Review the failing behavior check. Evidence: https://viewer/jobs/job/tasks/case/trials/trial"]


def test_add_evidence_links_to_suggestions_prioritizes_failing_rewards() -> None:
    suggestions = ["Review the failing behavior check."]
    rewards = [
        {
            "behavior_check": 1.0,
            "harbor_viewer": {
                "trial_url": "https://viewer/jobs/job/tasks/passing/trials/trial-1",
                "evidence_urls": [],
            },
        },
        {
            "behavior_check": 0.3333,
            "harbor_viewer": {
                "trial_url": "https://viewer/jobs/job/tasks/failing/trials/trial-2",
                "evidence_urls": [],
            },
        },
    ]

    linked = add_evidence_links_to_suggestions(suggestions, rewards)

    assert linked == [
        "Review the failing behavior check. Evidence: https://viewer/jobs/job/tasks/failing/trials/trial-2"
    ]


def test_html_suggestions_use_evidence_links_when_trial_dips_but_aggregate_passes() -> None:
    html = _build_suggestions_html(
        "hld-documents",
        {
            "codex": {
                "rewards": [
                    {
                        "entry_id": "case-1",
                        "security": 1.0,
                        "skill_execution": 1.0,
                        "skill_efficiency": 1.0,
                        "accuracy": 1.0,
                        "goal_accuracy": 0.7,
                        "behavior_check": 1.0,
                        "details": {
                            "goal_accuracy": {
                                "reason": "Register details were partially incomplete.",
                            },
                        },
                        "harbor_viewer": {
                            "trial_url": "https://viewer/jobs/job/tasks/task/trials/case-1",
                            "evidence_urls": [
                                {
                                    "metric": "goal_accuracy",
                                    "step": 5,
                                    "url": "https://viewer/jobs/job/tasks/task/trials/case-1?step=5",
                                }
                            ],
                        },
                    },
                    {
                        "entry_id": "case-2",
                        "security": 1.0,
                        "skill_execution": 1.0,
                        "skill_efficiency": 1.0,
                        "accuracy": 1.0,
                        "goal_accuracy": 0.9,
                        "behavior_check": 1.0,
                        "details": {
                            "goal_accuracy": {
                                "reason": "The expected HLD deliverable was complete.",
                            },
                        },
                    },
                ]
            }
        },
    )

    assert "Next Steps" in html
    assert "https://viewer/jobs/job/tasks/task/trials/case-1?step=5" in html


def test_html_suggestions_render_compact_evidence_links() -> None:
    html = _build_suggestions_html(
        "hld-documents",
        {
            "codex": {
                "rewards": [
                    {
                        "entry_id": "case-1",
                        "security": 1.0,
                        "skill_execution": 1.0,
                        "skill_efficiency": 0.7,
                        "accuracy": 1.0,
                        "goal_accuracy": 1.0,
                        "behavior_check": 1.0,
                        "details": {
                            "skill_efficiency": {
                                "reason": "The agent spent extra turns reading unrelated files.",
                            },
                        },
                        "harbor_viewer": {
                            "trial_url": "https://viewer/jobs/job/tasks/task/trials/case-1",
                            "evidence_urls": [
                                {
                                    "metric": "skill_efficiency",
                                    "step": 5,
                                    "url": "https://viewer/jobs/job/tasks/task/trials/case-1?step=5",
                                },
                                {
                                    "metric": "goal_accuracy",
                                    "step": 6,
                                    "url": "https://viewer/jobs/job/tasks/task/trials/case-1?step=6",
                                },
                                {
                                    "metric": "accuracy",
                                    "step": 7,
                                    "url": "https://viewer/jobs/job/tasks/task/trials/case-1?step=7",
                                },
                                {
                                    "metric": "behavior_check",
                                    "step": 8,
                                    "url": "https://viewer/jobs/job/tasks/task/trials/case-1?step=8",
                                },
                            ],
                        },
                    },
                    {
                        "entry_id": "case-2",
                        "security": 1.0,
                        "skill_execution": 1.0,
                        "skill_efficiency": 0.9,
                        "accuracy": 1.0,
                        "goal_accuracy": 1.0,
                        "behavior_check": 1.0,
                        "details": {
                            "skill_efficiency": {
                                "reason": "The run used a direct workflow.",
                            },
                        },
                    },
                ]
            }
        },
    )

    assert 'class="evidence-link"' in html
    assert 'href="https://viewer/jobs/job/tasks/task/trials/case-1?step=5"' in html
    assert 'target="_blank"' in html
    assert "Step 5" in html
    assert "View evidence" not in html
    assert ">https://viewer/jobs/job/tasks/task/trials/case-1?step=5</a>" not in html


def test_html_suggestions_fall_back_to_evidence_label_without_step() -> None:
    html = _build_suggestions_html(
        "hld-documents",
        {
            "codex": {
                "rewards": [
                    {
                        "entry_id": "case-1",
                        "security": 1.0,
                        "skill_execution": 1.0,
                        "skill_efficiency": 0.7,
                        "accuracy": 1.0,
                        "goal_accuracy": 1.0,
                        "behavior_check": 1.0,
                        "details": {
                            "skill_efficiency": {
                                "reason": "The agent spent extra turns reading unrelated files.",
                            },
                        },
                        "harbor_viewer": {
                            "trial_url": "https://viewer/jobs/job/tasks/task/trials/case-1",
                            "evidence_urls": [
                                {
                                    "metric": "skill_efficiency",
                                    "url": "https://viewer/jobs/job/tasks/task/trials/case-1",
                                }
                            ],
                        },
                    },
                    {
                        "entry_id": "case-2",
                        "security": 1.0,
                        "skill_execution": 1.0,
                        "skill_efficiency": 0.9,
                        "accuracy": 1.0,
                        "goal_accuracy": 1.0,
                        "behavior_check": 1.0,
                        "details": {
                            "skill_efficiency": {
                                "reason": "The run used a direct workflow.",
                            },
                        },
                    },
                ]
            }
        },
    )

    assert "View evidence" in html
    assert "Step " not in html


def test_agent_findings_link_metric_reasons_to_matching_trajectory_steps() -> None:
    html = _build_agent_findings(
        {
            "metrics_with_skill": ["accuracy", "goal_accuracy", "behavior_check"],
            "with_skill": {
                "accuracy": 0.9,
                "goal_accuracy": 0.8,
                "behavior_check": 0.7,
            },
            "rewards": [
                {
                    "accuracy": 0.9,
                    "goal_accuracy": 0.8,
                    "behavior_check": 0.7,
                    "details": {
                        "accuracy": {
                            "criteria": {"TASK_ADDRESSED": True},
                            "reason": "The response addressed the HLD request.",
                        },
                        "goal_accuracy": {
                            "reason": "The generated document covered the requested architecture.",
                        },
                        "behavior_check": {
                            "results": [
                                {
                                    "passed": False,
                                    "reason": "The agent did not verify register paths.",
                                }
                            ]
                        },
                    },
                    "harbor_viewer": {
                        "evidence_urls": [
                            {
                                "label": "accuracy",
                                "url": "https://viewer/jobs/job/tasks/task/trials/case-1?step=9",
                            },
                            {
                                "label": "goal_accuracy",
                                "url": "https://viewer/jobs/job/tasks/task/trials/case-1?step=12",
                            },
                            {
                                "label": "behavior_check",
                                "url": "https://viewer/jobs/job/tasks/task/trials/case-1?step=14",
                            },
                        ]
                    },
                }
            ],
        }
    )

    assert html.count('class="evidence-link"') == 4
    assert 'href="https://viewer/jobs/job/tasks/task/trials/case-1?step=9"' in html
    assert 'href="https://viewer/jobs/job/tasks/task/trials/case-1?step=12"' in html
    assert 'href="https://viewer/jobs/job/tasks/task/trials/case-1?step=14"' in html
    assert "Step 9" in html
    assert "Step 12" in html
    assert "Step 14" in html
    assert ">https://viewer/jobs/job/tasks/task/trials/case-1?step=9</a>" not in html


def test_dataset_html_uses_agentskills_field_names() -> None:
    html = _build_dataset_html(
        [
            {
                "id": "case-1",
                "prompt": "Use hld-documents for this design.",
                "question": "legacy fallback should not be shown",
                "expected_output": "A complete HLD document is produced.",
                "ground_truth": "legacy fallback should not be shown",
                "assertions": [
                    "The agent reads SKILL.md.",
                    "The agent writes the HLD sections.",
                ],
                "expected_behavior": ["legacy fallback should not be shown"],
                "expected_skill": "hld-documents",
                "expected_script": None,
            }
        ]
    )

    assert "1 AgentSkills eval case(s) in dataset" in html
    assert '<span class="ds-label">Prompt</span>' in html
    assert '<span class="ds-label">Expected Output</span>' in html
    assert '<span class="ds-label">Assertions</span>' in html
    assert "Use hld-documents for this design." in html
    assert "A complete HLD document is produced." in html
    assert "The agent reads SKILL.md." in html
    assert '<span class="ds-label">Question</span>' not in html
    assert '<span class="ds-label">Ground Truth</span>' not in html
    assert '<span class="ds-label">Expected Behavior</span>' not in html


def test_dataset_html_renders_legacy_entries_with_agentskills_labels() -> None:
    html = _build_dataset_html(
        [
            {
                "id": "case-1",
                "question": "Legacy prompt text.",
                "ground_truth": "Legacy expected output text.",
                "expected_behavior": ["Legacy assertion text."],
            }
        ]
    )

    assert '<span class="ds-label">Prompt</span>' in html
    assert '<span class="ds-label">Expected Output</span>' in html
    assert '<span class="ds-label">Assertions</span>' in html
    assert "Legacy prompt text." in html
    assert "Legacy expected output text." in html
    assert "Legacy assertion text." in html


def test_html_report_uses_agentskills_dataset_heading(tmp_path) -> None:
    summary_dir = tmp_path / "codex" / "with-skill"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(
            {
                "num_trials": 1,
                "scores": {
                    "security": 1.0,
                    "skill_execution": 1.0,
                    "skill_efficiency": 1.0,
                    "accuracy": 1.0,
                    "goal_accuracy": 1.0,
                    "behavior_check": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )

    report = generate_html_report("hld-documents", tmp_path)
    html = report.read_text(encoding="utf-8")

    assert "<h2>AgentSkills Dataset</h2>" in html
    assert "<h2>Evaluation Dataset</h2>" not in html


def test_html_report_loads_dataset_from_staged_harbor_entries_without_skill_path(tmp_path) -> None:
    summary_dir = tmp_path / "codex" / "with-skill"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(
            {
                "num_trials": 1,
                "scores": {
                    "security": 1.0,
                    "skill_execution": 1.0,
                    "skill_efficiency": 1.0,
                    "accuracy": 1.0,
                    "goal_accuracy": 1.0,
                    "behavior_check": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    entry_dir = tmp_path / "_harbor-tasks" / "hld-documents-001" / "tests"
    entry_dir.mkdir(parents=True)
    (entry_dir / "entry.json").write_text(
        json.dumps(
            {
                "id": "hld-documents-001",
                "question": "Create an HLD for packet reordering.",
                "ground_truth": "A complete HLD document is produced.",
                "expected_behavior": [
                    "The agent reads the hld-documents skill.",
                    "The agent includes hardware register tables.",
                ],
                "expected_skill": "hld-documents",
            }
        ),
        encoding="utf-8",
    )

    report = generate_html_report("hld-documents", tmp_path)
    html = report.read_text(encoding="utf-8")

    assert "1 AgentSkills eval case(s) in dataset" in html
    assert "Create an HLD for packet reordering." in html
    assert "A complete HLD document is produced." in html
    assert "The agent includes hardware register tables." in html
    assert "No dataset found" not in html


def test_html_report_links_uploaded_harbor_analysis(tmp_path) -> None:
    summary_dir = tmp_path / "codex" / "with-skill"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(
            {
                "num_trials": 1,
                "scores": {
                    "security": 1.0,
                    "skill_execution": 1.0,
                    "skill_efficiency": 1.0,
                    "accuracy": 1.0,
                    "goal_accuracy": 1.0,
                    "behavior_check": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    trial_dir = summary_dir / "trials" / "case-1"
    trial_dir.mkdir(parents=True)
    (trial_dir / "reward.json").write_text(
        json.dumps(
            {
                "entry_id": "case-1",
                "security": 1.0,
                "skill_execution": 1.0,
                "skill_efficiency": 1.0,
                "accuracy": 1.0,
                "goal_accuracy": 1.0,
                "behavior_check": 1.0,
                "harbor_viewer": {
                    "job_url": "https://harbor-viewer.prd.astra.nvidia.com/jobs/hld-documents-codex-with--run123",
                    "analysis_url": (
                        "https://harbor-viewer.prd.astra.nvidia.com/jobs/"
                        "hld-documents-codex-with--run123?tab=analysis"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    report = generate_html_report("hld-documents", tmp_path)
    html = report.read_text(encoding="utf-8")

    assert "Harbor Analysis" in html
    assert (
        'href="https://harbor-viewer.prd.astra.nvidia.com/jobs/'
        'hld-documents-codex-with--run123?tab=analysis"'
    ) in html
    assert 'target="_blank"' in html
