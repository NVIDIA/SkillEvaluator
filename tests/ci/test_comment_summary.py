# SPDX-License-Identifier: Apache-2.0
"""Compact PR/MR comments retain truthful, skill-scoped native evidence."""

import copy
import html
import re

import pytest

ARTIFACT_URL = "https://github.com/acme/skills/actions/runs/123/artifacts/456"


def manifest():
    return {
        "schema_version": 1,
        "report_contract": "skillevaluator-ci-v1",
        "commit": "a" * 40,
        "repository": "acme/skills",
        "platform": "github",
        "run_id": "123",
        "run_attempt": "1",
        "environment": "docker",
        "tiers": [
            {
                "tier": tier,
                "required": tier == 1,
                "status": "passed",
                "skills": [
                    {
                        "path": "skills/demo",
                        "status": "passed",
                        "report_json": f"tier{tier}/skill-001/skillevaluator-tier{tier}.json",
                        "report_html": f"tier{tier}/skill-001/skillevaluator-tier{tier}.html",
                        "metrics": {
                            "issue_count": 0,
                            "issue_count_known": True,
                            "critical_count": 0,
                            "high_count": 0,
                            "top_issue": None,
                        },
                    }
                ],
            }
            for tier in (1, 2, 3)
        ],
    }


def render(payload):
    from ci.comment_summary import render_comment_summary

    return render_comment_summary(payload, ARTIFACT_URL)


def data_rows(body):
    return [line for line in body.splitlines() if line.startswith("| ")][2:]


def test_one_skill_row_combines_all_three_tiers_and_one_download_link():
    payload = manifest()
    payload["tiers"][1]["status"] = "failed"
    skill = payload["tiers"][1]["skills"][0]
    skill.update(
        status="failed",
        metrics={
            "issue_count": 3,
            "issue_count_known": True,
            "critical_count": 1,
            "high_count": 1,
            "top_issue": "Repeated deployment instructions",
            "private_metric": "must not appear",
        },
    )
    before = copy.deepcopy(payload)

    body = render(payload)

    assert "Required gate: **Passed**" in body
    assert "Only Tier 1 blocks CI; Tier 2 and Tier 3 are advisory." in body
    assert "| Skill | Tier 1: Static and Security Validation | Tier 2: Deduplication |" in body
    assert "Tier 3: Live Agent Evaluation | Report |" in body
    assert len(data_rows(body)) == 1
    assert data_rows(body)[0].count("PASSED") == 2
    assert "FAILED · 3 issues (1 critical, 1 high) · Repeated deployment instructions" in body
    assert body.count(f"[Report]({ARTIFACT_URL})") == 1
    assert "Native HTML" not in body and "## Tier" not in body and "must not appear" not in body
    assert ".html" not in body and ".json" not in body
    assert payload == before


def test_missing_native_metrics_cannot_become_zero_issues():
    payload = manifest()
    skill = payload["tiers"][2]["skills"][0]
    skill.update(status="incomplete", report_json=None, report_html=None, error="Configure the live agent key")
    skill.pop("metrics")
    payload["tiers"][2]["status"] = "incomplete"

    tier3_cell = data_rows(render(payload))[0].split("|")[4]

    assert "INCOMPLETE" in tier3_cell and "issues unknown" in tier3_cell
    assert "Configure the live agent key" in tier3_cell and "0 issues" not in tier3_cell


@pytest.mark.parametrize(
    "count,critical,high,expected",
    [
        (0, 0, 0, "PASSED · 0 issues"),
        (2, 0, 1, "PASSED · 2 issues (1 high)"),
        (2, 1, 0, "PASSED · 2 issues (1 critical)"),
    ],
)
def test_zero_severity_counts_do_not_distract_from_the_issue_summary(count, critical, high, expected):
    payload = manifest()
    payload["tiers"][0]["skills"][0]["metrics"].update(issue_count=count, critical_count=critical, high_count=high)

    cell = data_rows(render(payload))[0].split("|")[2].strip()

    assert cell == expected


def test_unknown_skill_evidence_uses_tier_provider_action_when_no_own_detail_exists():
    payload = manifest()
    tier = payload["tiers"][2]
    tier.update(status="needs_action", reason="Missing provider credential: configure the live agent key")
    skill = tier["skills"][0]
    skill.update(status="needs_action", report_json=None, report_html=None)
    skill.pop("metrics")

    cell = data_rows(render(payload))[0].split("|")[4]

    assert "NEEDS_ACTION · issues unknown · Missing provider credential: configure the live agent key" in cell
    assert "Native evidence unavailable" not in cell


def test_complete_clean_skill_does_not_borrow_a_tier_reason():
    payload = manifest()
    payload["tiers"][0]["reason"] = "Missing provider credential for another selected skill"

    cell = data_rows(render(payload))[0].split("|")[2].strip()

    assert cell == "PASSED · 0 issues"


def test_highest_native_finding_takes_priority_over_a_generic_execution_error():
    payload = manifest()
    tier = payload["tiers"][0]
    tier["status"] = "failed"
    skill = tier["skills"][0]
    skill.update(status="failed", error="Validation process exited with code 1")
    skill["metrics"].update(issue_count=1, critical_count=1, top_issue="Critical command injection in deploy.sh")

    cell = data_rows(render(payload))[0].split("|")[2]

    assert "Critical command injection in deploy.sh" in cell
    assert "Validation process exited" not in cell


def test_incomplete_native_evidence_reports_observed_findings_as_a_lower_bound():
    payload = manifest()
    payload["tiers"][0]["status"] = "incomplete"
    skill = payload["tiers"][0]["skills"][0]
    skill.update(status="incomplete")
    skill["metrics"].update(issue_count=2, issue_count_known=False, critical_count=1, top_issue="Unsafe shell")

    body = render(payload)

    assert "Required gate: **Not passed**" in body
    assert "INCOMPLETE · at least 2 issues" in body and "Unsafe shell" in body


def test_missing_skill_does_not_reuse_sibling_finding_or_counts():
    payload = manifest()
    sibling = payload["tiers"][1]["skills"][0]
    sibling.update(path="skills/other", status="failed")
    sibling["metrics"].update(issue_count=9, critical_count=9, top_issue="Sibling only issue")
    payload["tiers"][1]["status"] = "failed"

    rows = data_rows(render(payload))
    demo = next(row for row in rows if row.startswith("| skills/demo |"))
    other = next(row for row in rows if row.startswith("| skills/other |"))

    assert "INCOMPLETE · issues unknown · No retained result for this skill" in demo
    assert "Sibling only issue" not in demo and "9 issues" not in demo
    assert "FAILED · 9 issues" in other and "Sibling only issue" in other
    assert "INCOMPLETE · issues unknown" in other


def test_empty_tier_preflight_failure_keeps_its_declared_reason():
    payload = manifest()
    payload["tiers"][1].update(status="failed", skills=[], reason="Duplication check failed")

    cell = data_rows(render(payload))[0].split("|")[3]

    assert "FAILED · issues unknown · Duplication check failed" in cell


def test_missing_whole_tier_and_unrequested_tier_are_distinct():
    payload = manifest()
    payload["tiers"] = payload["tiers"][:1]
    body = render(payload)
    assert body.count("NOT_RUN · issues unknown · No retained result") == 2
    payload["expected_tiers"] = [1]
    assert render(payload).count("NOT_RUN · issues unknown · Not requested") == 2


@pytest.mark.parametrize("required_paid", [[], [2], [3], [2, 3]])
def test_policy_matches_historical_required_flags(required_paid):
    payload = manifest()
    for tier in required_paid:
        record = payload["tiers"][tier - 1]
        record.update(required=True, status="failed")
        record["skills"][0]["status"] = "failed"

    body = render(payload)

    if required_paid:
        assert "Required gate: **Not passed**" in body
        assert "Only Tier 1 blocks CI" not in body
        required = ", ".join(f"Tier {tier}" for tier in [1, *required_paid])
        advisory = ", ".join(f"Tier {tier}" for tier in (2, 3) if tier not in required_paid) or "none"
        assert f"Required tiers: {required}. Advisory tiers: {advisory}." in body
    else:
        assert "Required gate: **Passed**" in body
        assert "Only Tier 1 blocks CI" in body


def test_no_work_keeps_each_tier_state_and_a_single_report_link():
    payload = manifest()
    for record in payload["tiers"]:
        record.update(status="skipped", skills=[], reason="No changed skills")
    body = render(payload)
    assert "Required gate: **Skipped**" in body
    assert len(data_rows(body)) == 1
    assert "No skills evaluated" in body
    assert body.count("SKIPPED · issues unknown · No changed skills") == 3
    assert body.count(f"[Report]({ARTIFACT_URL})") == 1


def test_cell_content_cannot_create_markup_mentions_or_extra_rows():
    payload = manifest()
    for tier in payload["tiers"]:
        skill = tier["skills"][0]
        skill["path"] = "skills/@everyone[demo]|<b>*"
        skill["metrics"].update(
            issue_count=1, high_count=1, top_issue="@team <script> [link](https://bad.test) | **bold**\n" + "x" * 500
        )

    body = render(payload)
    row = data_rows(body)[0]

    assert len(data_rows(body)) == 1 and row.count("|") == 6
    assert re.search(r"(@|&#64;)\w", body) is None
    assert "<script>" not in body and "<b>" not in body
    assert "[link](https://bad.test)" not in body and "**bold**" not in body
    assert "x" * 150 not in body and "…" in body


@pytest.mark.parametrize("handle", ["@user", "@all", "@org/team", "&#64;user", "&commat;user", "\\@user"])
def test_untrusted_text_never_renders_user_or_team_mentions(handle):
    from ci.summary_metrics import _plain_text

    payload = manifest()
    path = f"skills/{handle}"
    tier1, tier2, tier3 = payload["tiers"]
    tier1.update(status="failed")
    tier1["skills"][0].update(path=path, status="failed")
    tier1["skills"][0]["metrics"].update(issue_count=1, high_count=1, top_issue=f"{handle} raw top issue")
    other = copy.deepcopy(tier1["skills"][0])
    other.update(path="skills/other")
    other["metrics"].update(top_issue=_plain_text(f"{handle} summarized issue"))
    tier1["skills"].append(other)
    # A tier without per-skill evidence explains itself with its own error text.
    tier2.update(status="incomplete", skills=[], error=f"Tier error from {handle}")
    tier3.update(status="failed")
    tier3["skills"][0].update(path=path, status="failed", error=f"{handle} skill error")
    tier3["skills"][0]["metrics"].update(issue_count_known=False)

    body = render(payload)

    # CommonMark decodes "&#64;" to "@" before GitHub and GitLab resolve mentions.
    assert re.search(r"(@|&#64;)\w", body) is None
    assert set(re.findall(r"&[^\s&;]*;", body)) <= {"&amp;", "&lt;", "&gt;", "&#124;"}
    rendered_text = html.unescape(re.sub(r"\\(.)", r"\1", body))
    assert re.search(r"@\w", rendered_text) is None
    assert "skills/" + handle.replace("@", "@\u200b") in rendered_text
    for text in ("skills/other", "raw top issue", "summarized issue", "Tier error from", "skill error"):
        assert text in rendered_text


def test_large_skill_set_is_bounded_and_reports_the_omitted_count():
    payload = manifest()
    for tier in payload["tiers"]:
        original = tier["skills"][0]
        tier["skills"] = [dict(original, path=f"skills/demo-{index:04d}") for index in range(1000)]

    body = render(payload)
    rows = data_rows(body)
    omitted = re.search(r"([0-9]+) additional skills omitted", body)

    assert len(body.encode("utf-8")) < 55000
    assert 0 < len(rows) < 1000 and omitted is not None
    assert len(rows) + int(omitted.group(1)) == 1000
    assert body.count(f"[Report]({ARTIFACT_URL})") == len(rows)


def test_rejects_contradictory_gate_instead_of_rendering_it():
    payload = manifest()
    payload["overall_passed"] = False
    with pytest.raises(ValueError, match="contradicts"):
        render(payload)


@pytest.mark.parametrize(
    "url", ["javascript:alert(1)", "http://example.test/archive.zip", "https://a.test/x\n| forged |"]
)
def test_rejects_unsafe_artifact_links(url):
    from ci.comment_summary import render_comment_summary

    with pytest.raises(ValueError, match="artifact"):
        render_comment_summary(manifest(), url)
