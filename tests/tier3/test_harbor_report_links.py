# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
