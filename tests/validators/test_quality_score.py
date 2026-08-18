# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for QualityScoreValidator -- 4-dimension skill quality analysis."""

from pathlib import Path

import pytest

import skillevaluator.validators.quality_score as quality_score_module
from skillevaluator.models.quality import QualityScoreResult, score_to_grade
from skillevaluator.validators.quality_score import QualityScoreValidator


@pytest.fixture
def quality_skill(tmp_path: Path) -> Path:
    """High-quality skill that should score well across all dimensions."""
    skill_dir = tmp_path / "quality-skill"
    skill_dir.mkdir()
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    refs_dir = skill_dir / "references"
    refs_dir.mkdir()

    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: quality-skill\n"
        "description: A well-structured skill for data processing. Use when you need to process CSV files.\n"
        "metadata:\n"
        "  author: Test User <test@nvidia.com>\n"
        "  tags:\n"
        "    - data\n"
        "    - csv\n"
        "allowed-tools: Shell Read\n"
        "---\n\n"
        "# Quality Skill\n\n"
        "## Purpose\n\nThis skill processes CSV data into structured JSON output.\n\n"
        "## Prerequisites\n\n- Python 3.10+\n- pandas library\n\n"
        "## Instructions\n\n"
        "1. Use `run_script` with the `convert.py` script\n"
        "2. Pass the input CSV path as the first argument\n"
        "3. Set the output path with `--output`\n\n"
        "## Available Scripts\n\n"
        "| Script | Purpose | Arguments |\n"
        "|--------|---------|----------|\n"
        "| convert.py | Convert CSV to JSON | input_path, --output |\n\n"
        "## Examples\n\n"
        "```bash\nrun_script('scripts/convert.py', 'data.csv')\n```\n\n"
        "## Limitations\n\n- Maximum file size: 100MB\n\n"
        "## Troubleshooting\n\n"
        "| Error | Cause | Solution |\n"
        "|-------|-------|----------|\n"
        "| FileNotFoundError | Input path missing | Check file path |\n"
    )

    (scripts_dir / "convert.py").write_text(
        "#!/usr/bin/env python3\n"
        "import argparse\nimport json\nimport sys\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser(description='Convert CSV to JSON')\n"
        "    parser.add_argument('input_path', help='Input CSV file')\n"
        "    args = parser.parse_args()\n"
        "    try:\n"
        "        with open(args.input_path) as f:\n"
        "            data = f.read()\n"
        "    except FileNotFoundError:\n"
        "        raise ValueError(f'Input file not found: {args.input_path}')\n\n"
        "if __name__ == '__main__':\n    main()\n"
    )

    (refs_dir / "csv-format-guide.md").write_text("# CSV Format Guide\n\nDetailed guide.\n")
    return skill_dir


@pytest.fixture
def minimal_skill(tmp_path: Path) -> Path:
    """Minimal guide-only skill that should score lower."""
    skill_dir = tmp_path / "minimal-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: minimal-skill\ndescription: A minimal skill\n---\n\n# Minimal Skill\n\nShort content.\n"
    )
    return skill_dir


@pytest.fixture
def bad_skill(tmp_path: Path) -> Path:
    """Skill with many issues."""
    skill_dir = tmp_path / "bad-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: Bad_Skill\ndescription: stuff\n---\n\nNot much here.\n")
    return skill_dir


def _write_issue_skill(
    tmp_path: Path,
    *,
    name: str = "compile-time-probe",
    description: str = "Use when mapping kernels to modules and finding redundant builds.",
    instructions: str = "- Run the probe.\n- Read the report.",
    troubleshooting: str = "Read diagnostic output when the probe reports an error.",
    extra_body: str = "",
) -> Path:
    """Write a complete guide skill for issue #30 scoring regressions."""
    skill_dir = tmp_path / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "version: 0.1.0\n"
        "metadata:\n"
        "  author: Example\n"
        "  tags:\n"
        "    - performance\n"
        "---\n\n"
        "## Purpose\n\nMeasure compile time.\n\n"
        "## Instructions\n\n"
        f"{instructions}\n\n"
        "## Examples\n\n```bash\nprobe measure -- python app.py\n```\n\n"
        "## Prerequisites\n\nNone.\n\n"
        "## Limitations\n\nNone known.\n\n"
        "## Troubleshooting\n\n"
        f"{troubleshooting}\n"
        f"{extra_body}"
    )
    return skill_dir


def _finding_messages(skill_dir: Path) -> list[str]:
    result = QualityScoreValidator(min_score=0).validate(skill_dir)
    return [finding.message for finding in result.findings]


class TestScoreToGrade:
    def test_grade_a(self):
        assert score_to_grade(95.0) == "A"
        assert score_to_grade(90.0) == "A"

    def test_grade_b(self):
        assert score_to_grade(85.0) == "B"
        assert score_to_grade(80.0) == "B"

    def test_grade_c(self):
        assert score_to_grade(75.0) == "C"
        assert score_to_grade(70.0) == "C"

    def test_grade_d(self):
        assert score_to_grade(65.0) == "D"
        assert score_to_grade(60.0) == "D"

    def test_grade_f(self):
        assert score_to_grade(55.0) == "F"
        assert score_to_grade(0.0) == "F"


class TestQualityScoreResult:
    def test_default_scores(self):
        qs = QualityScoreResult(skill_name="test")
        assert qs.correctness.score == 100.0
        assert qs.overall_score == 100.0
        assert qs.grade == "A"

    def test_deduction_affects_score(self):
        qs = QualityScoreResult(skill_name="test")
        qs.correctness.deduct(50, "error", "bad thing")
        assert qs.correctness.score == 50.0
        assert qs.overall_score < 100.0

    def test_to_dict(self):
        qs = QualityScoreResult(skill_name="test", skill_type="guide-only")
        d = qs.to_dict()
        assert d["skill_name"] == "test"
        assert d["skill_type"] == "guide-only"
        assert "correctness" in d["dimensions"]
        assert d["dimensions"]["correctness"]["weight"] == 0.35

    def test_weighted_formula(self):
        qs = QualityScoreResult(skill_name="test")
        qs.correctness.score = 80.0
        qs.discoverability.score = 60.0
        qs.reliability.score = 90.0
        qs.efficiency.score = 70.0
        expected = 80 * 0.35 + 60 * 0.25 + 90 * 0.25 + 70 * 0.15
        assert abs(qs.overall_score - expected) < 0.01


class TestSkillTypeDetection:
    def test_guide_only(self, tmp_path):
        d = tmp_path / "guide"
        d.mkdir()
        (d / "SKILL.md").write_text("guide")
        assert QualityScoreValidator.detect_skill_type(d) == "guide-only"

    def test_script_based(self, tmp_path):
        d = tmp_path / "scripted"
        d.mkdir()
        sd = d / "scripts"
        sd.mkdir()
        (sd / "run.py").write_text("print('hi')")
        assert QualityScoreValidator.detect_skill_type(d) == "script-based"

    def test_lib_based(self, tmp_path):
        d = tmp_path / "lib"
        d.mkdir()
        mod = d / "mymodule"
        mod.mkdir()
        (mod / "__init__.py").write_text("")
        assert QualityScoreValidator.detect_skill_type(d) == "lib-based"

    def test_resource_based(self, tmp_path):
        d = tmp_path / "res"
        d.mkdir()
        (d / "assets").mkdir()
        assert QualityScoreValidator.detect_skill_type(d) == "resource-based"

    def test_hybrid(self, tmp_path):
        d = tmp_path / "hyb"
        d.mkdir()
        sd = d / "scripts"
        sd.mkdir()
        (sd / "run.sh").write_text("echo hi")
        (d / "assets").mkdir()
        assert QualityScoreValidator.detect_skill_type(d) == "hybrid"


class TestQualityScoreValidator:
    def test_high_quality_skill(self, quality_skill):
        v = QualityScoreValidator(min_score=70)
        result = v.validate(quality_skill)
        qs = result.metadata.get("quality_scores")
        assert qs is not None
        assert qs["overall_score"] >= 70
        assert qs["grade"] in ("A", "B", "C")
        assert qs["skill_type"] == "script-based"

    def test_minimal_skill_lower_score(self, minimal_skill):
        v = QualityScoreValidator(min_score=0)
        result = v.validate(minimal_skill)
        qs = result.metadata.get("quality_scores")
        assert qs is not None
        assert qs["overall_score"] < 90
        assert qs["skill_type"] == "guide-only"

    def test_bad_skill_fails_min_score(self, bad_skill):
        v = QualityScoreValidator(min_score=70)
        result = v.validate(bad_skill)
        assert not result.passed
        qs = result.metadata.get("quality_scores")
        assert qs is not None
        assert qs["overall_score"] < 70

    def test_missing_skill_md(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        v = QualityScoreValidator()
        result = v.validate(d)
        assert result.findings

    def test_folder_validation(self, tmp_path):
        skills = tmp_path / "skills"
        skills.mkdir()
        for name in ["skill-a", "skill-b"]:
            sd = skills / name
            sd.mkdir()
            (sd / "SKILL.md").write_text(
                f"---\nname: {name}\n"
                f"description: Skill {name} for testing. Use when you need to test.\n"
                "---\n\n"
                f"# {name}\n\n## Instructions\n\n1. Run the skill\n2. Check results\n"
            )
        v = QualityScoreValidator(min_score=0)
        result = v.validate(skills)
        assert result.metadata.get("quality_scores") is not None

    def test_missing_metadata_author_remains_quality_warning(self, tmp_path):
        """Contributor metadata behavior stays unchanged while optional tags are ignored."""
        skill_dir = tmp_path / "missing-author"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: missing-author\n"
            "description: A focused skill. Use when testing missing contributor metadata.\n"
            "allowed-tools: Read\n"
            "---\n\n"
            "# Missing Author\n\n"
            "## Purpose\n\nThis skill verifies contributor metadata warnings.\n\n"
            "## Instructions\n\n1. Check quality findings.\n\n"
        )

        result = QualityScoreValidator(min_score=0).validate(skill_dir)

        assert any("metadata.author" in finding.message for finding in result.findings)

    def test_xml_tags_in_description_remain_quality_error(self, tmp_path):
        """Real XML/HTML-like tags in descriptions are still flagged."""
        skill_dir = tmp_path / "xml-desc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: xml-desc\n"
            "description: \"A skill <script>alert('xss')</script> with injected tags\"\n"
            "metadata:\n"
            "  author: Test User <test@nvidia.com>\n"
            "---\n\n"
            "# XML Description\n\n"
            "## Instructions\n\n1. Inspect frontmatter quality findings.\n\n"
            "## Examples\n\n"
            "```text\n"
            "Validate the skill.\n"
            "```\n"
        )

        result = QualityScoreValidator(min_score=0).validate(skill_dir)

        assert any("Description contains XML tags" in finding.message for finding in result.findings)

    def test_unclosed_xml_tag_in_description_remains_quality_error(self, tmp_path):
        """Unclosed tag-like descriptions remain covered by XML-tag detection."""
        skill_dir = tmp_path / "unclosed-xml-desc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: unclosed-xml-desc\n"
            'description: "A skill with an unclosed <script foo tag in the description"\n'
            "metadata:\n"
            "  author: Test User <test@nvidia.com>\n"
            "---\n\n"
            "# Unclosed XML Description\n\n"
            "## Instructions\n\n1. Inspect frontmatter quality findings.\n\n"
            "## Examples\n\n"
            "```text\n"
            "Validate the skill.\n"
            "```\n"
        )

        result = QualityScoreValidator(min_score=0).validate(skill_dir)

        assert any("Description contains XML tags" in finding.message for finding in result.findings)

    def test_closing_xml_tag_in_description_remains_quality_error(self, tmp_path):
        """Closing tags use the same XML definition as schema validation."""
        skill_dir = tmp_path / "closing-xml-desc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: closing-xml-desc\n"
            'description: "Use when validating a closing </tool> tag."\n'
            "metadata:\n"
            "  author: Test User <test@nvidia.com>\n"
            "---\n\n"
            "# Closing XML Description\n\n"
            "## Instructions\n\n1. Inspect frontmatter quality findings.\n\n"
            "## Examples\n\nValidate the skill.\n"
        )

        result = QualityScoreValidator(min_score=0).validate(skill_dir)

        assert any("Description contains XML tags" in finding.message for finding in result.findings)

    def test_self_closing_xml_tag_in_description_remains_quality_error(self, tmp_path: Path):
        """Self-closing tags use the same XML definition as schema validation."""
        skill_dir = _write_issue_skill(
            tmp_path,
            description="Use when validating an injected <script/> tag.",
        )

        result = QualityScoreValidator(min_score=0).validate(skill_dir)

        assert any("Description contains XML tags" in finding.message for finding in result.findings)

    def test_readme_supporting_file_is_allowed(self, quality_skill):
        # An unreferenced README.md is a permitted human-facing supporting file
        # (SkillEvaluator HOW_TO_CONTRIBUTE_SKILLS.md). The quality_skill SKILL.md does
        # not link to it, so under progressive disclosure it costs no agent
        # context and must not be penalized.
        (quality_skill / "README.md").write_text("# Human-facing skill notes\n")

        v = QualityScoreValidator(min_score=0)
        result = v.validate(quality_skill)

        assert all("README.md found inside skill folder" not in finding.message for finding in result.findings)
        correctness_issues = result.metadata["quality_scores"]["dimensions"]["correctness"]["issues"]
        assert all("README.md" not in issue["message"] for issue in correctness_issues)

    def test_readme_referenced_by_skill_is_flagged(self, quality_skill):
        # Referencing README.md from SKILL.md pulls human-facing docs into the
        # agent context window under progressive disclosure, which the quality
        # scorer should flag (Anthropic H2 / Codex skill-creator guidance).
        (quality_skill / "README.md").write_text("# Human-facing skill notes\n")
        skill_md = quality_skill / "SKILL.md"
        skill_md.write_text(skill_md.read_text() + "\n## More\n\nSee [the overview](README.md) for background.\n")

        v = QualityScoreValidator(min_score=0)
        result = v.validate(quality_skill)

        readme_findings = [
            finding
            for finding in result.findings
            if "README.md" in finding.message and "references" in finding.message.lower()
        ]
        assert readme_findings, "Expected a finding when SKILL.md references README.md"

    def test_reference_style_readme_link_is_flagged(self, quality_skill: Path):
        (quality_skill / "README.md").write_text("# Human-facing skill notes\n")
        skill_md = quality_skill / "SKILL.md"
        skill_md.write_text(
            skill_md.read_text() + "\n## More\n\nSee [the overview][readme] for background.\n\n[readme]: README.md\n"
        )

        result = QualityScoreValidator(min_score=0).validate(quality_skill)

        assert any(
            "README.md" in finding.message and "references" in finding.message.lower() for finding in result.findings
        )

    def test_large_skill_is_high_severity(self, tmp_path):
        """Large skills (>5000 tokens) should produce HIGH severity findings."""
        from skillevaluator.models.result import Severity

        skill_dir = tmp_path / "large-skill"
        skill_dir.mkdir()
        # ~5500 tokens (22000 chars / 4)
        filler = "This is filler content for testing token limits.\n" * 440
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: large-skill\n"
            "description: A skill that exceeds the token limit. Use for testing.\n"
            "---\n\n"
            "# Large Skill\n\n"
            "## Instructions\n\n1. Run the skill\n\n"
            "## Examples\n\n```\nexample\n```\n\n" + filler
        )
        v = QualityScoreValidator(min_score=0)
        result = v.validate(skill_dir)
        large_findings = [
            f for f in result.findings if "Large skill" in f.message or "large skill" in f.message.lower()
        ]
        assert len(large_findings) >= 1, "Expected a finding about large skill size"
        assert large_findings[0].severity == Severity.HIGH
        assert "recommended max <5000" in large_findings[0].message
        assert "long or unfocused top-level descriptions" in large_findings[0].message
        assert "Keep required sections concise" in large_findings[0].suggestion
        assert "references/" in large_findings[0].suggestion

    def test_above_6000_tokens_uses_same_5000_recommendation(self, tmp_path):
        """Skills above 6000 tokens should still use the single >5000 recommendation."""
        from skillevaluator.models.result import Severity

        skill_dir = tmp_path / "above-6000-token-skill"
        skill_dir.mkdir()
        # ~6500 tokens (26000 chars / 4)
        filler = "This is filler content for testing token limits.\n" * 520
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: above-6000-token-skill\n"
            "description: A skill above 6000 tokens. Use for testing token limit wording.\n"
            "---\n\n"
            "# Above 6000 Token Skill\n\n"
            "## Instructions\n\n1. Run the skill\n\n"
            "## Examples\n\n```\nexample\n```\n\n" + filler
        )
        v = QualityScoreValidator(min_score=0)
        result = v.validate(skill_dir)
        large_findings = [f for f in result.findings if "large skill" in f.message.lower()]
        assert len(large_findings) >= 1, "Expected a finding about large skill size"
        assert large_findings[0].severity == Severity.HIGH
        assert "recommended max <5000" in large_findings[0].message
        assert "recommend <6000" not in large_findings[0].message
        assert "Very large skill" not in large_findings[0].message

        efficiency_issues = result.metadata["quality_scores"]["dimensions"]["efficiency"]["issues"]
        large_issue = next(issue for issue in efficiency_issues if "Large skill" in issue["message"])
        assert large_issue["deduction"] == 15

    def test_validator_name_and_description(self):
        v = QualityScoreValidator()
        assert "Quality" in v.name
        assert "Correctness" in v.description


class TestQualityScoreKeywordBoundaries:
    @pytest.mark.parametrize(
        "description",
        [
            "Use when mapping kernel -> module relationships during builds.",
            "Use when cutting rebuild cost by >30% for large modules.",
            "Use when targeting compile steps that should finish in <1s.",
        ],
    )
    def test_comparison_syntax_is_not_reported_as_xml(self, tmp_path: Path, description: str):
        messages = _finding_messages(_write_issue_skill(tmp_path, description=description))

        assert "Description contains XML tags" not in messages

    def test_reserved_names_match_complete_name_segments(self, tmp_path: Path):
        messages = _finding_messages(_write_issue_skill(tmp_path, name="philanthropic-grant-matcher"))

        assert "Name contains reserved word (anthropic, claude)" not in messages

    def test_reserved_name_segment_is_still_reported(self, tmp_path: Path):
        messages = _finding_messages(_write_issue_skill(tmp_path, name="anthropic-grant-matcher"))

        assert "Name contains reserved word (anthropic, claude)" in messages

    def test_person_phrases_do_not_match_inside_words(self, tmp_path: Path):
        messages = _finding_messages(
            _write_issue_skill(
                tmp_path,
                description="Use when generating dummy data fixtures for compile probes.",
            )
        )

        assert "Description uses first/second person" not in messages

    def test_when_to_use_trigger_does_not_match_inside_performance(self, tmp_path: Path):
        messages = _finding_messages(
            _write_issue_skill(
                tmp_path,
                description="Maps performance characteristics across compilation workloads.",
            )
        )

        assert "Description doesn't mention WHEN to use this skill" in messages

    @pytest.mark.parametrize("incidental_word", ["important", "rapid"])
    def test_lib_documentation_requires_complete_api_terms(self, tmp_path: Path, incidental_word: str):
        skill_dir = _write_issue_skill(
            tmp_path,
            description="Use when analyzing compile speed across build targets.",
            extra_body=f"\n## Notes\n\nThis is {incidental_word} for compile analysis.\n",
        )
        module = skill_dir / "probe"
        module.mkdir()
        (module / "__init__.py").write_text("")
        (skill_dir / "pyproject.toml").write_text("[project]\nname = 'probe'\nversion = '0.1.0'\n")

        messages = _finding_messages(skill_dir)

        assert "Lib-based skill lacks API/import documentation" in messages

    def test_check_by_itself_does_not_claim_error_handling(self, tmp_path: Path):
        skill_dir = _write_issue_skill(
            tmp_path,
            troubleshooting="Check the output directory.",
        )

        result = QualityScoreValidator(min_score=0).validate(skill_dir)

        assert result.metadata["quality_scores"]["metrics"]["has_error_handling"] is False
        assert "No mention of error handling or validation" in [finding.message for finding in result.findings]

    def test_error_terms_still_record_error_handling(self, tmp_path: Path):
        skill_dir = _write_issue_skill(
            tmp_path,
            troubleshooting="If validation fails, report the error and retry the probe.",
        )

        result = QualityScoreValidator(min_score=0).validate(skill_dir)

        assert result.metadata["quality_scores"]["metrics"]["has_error_handling"] is True
        assert "No mention of error handling or validation" not in [finding.message for finding in result.findings]

    def test_negated_mcp_mention_does_not_classify_mcp_skill(self, tmp_path: Path):
        messages = _finding_messages(_write_issue_skill(tmp_path, extra_body="\nThis skill does not use MCP.\n"))

        assert "MCP skill lacks connection/error guidance" not in messages

    def test_postposed_negated_mcp_mention_does_not_classify_mcp_skill(self, tmp_path: Path):
        messages = _finding_messages(_write_issue_skill(tmp_path, extra_body="\nMCP is not used by this skill.\n"))

        assert "MCP skill lacks connection/error guidance" not in messages

    @pytest.mark.parametrize(
        "statement",
        [
            "This skill is not using MCP.",
            "This workflow isn't currently relying on MCP.",
            "MCP usage is not supported.",
            "MCP access isn't available.",
        ],
    )
    def test_copular_negated_mcp_mention_does_not_classify_mcp_skill(self, tmp_path: Path, statement: str):
        messages = _finding_messages(_write_issue_skill(tmp_path, extra_body=f"\n{statement}\n"))

        assert "MCP skill lacks connection/error guidance" not in messages

    @pytest.mark.parametrize(
        "statement",
        [
            "This skill should not use MCP.",
            "MCP should not be used by this skill.",
        ],
    )
    def test_modal_negated_mcp_mention_does_not_classify_mcp_skill(self, tmp_path: Path, statement: str):
        messages = _finding_messages(_write_issue_skill(tmp_path, extra_body=f"\n{statement}\n"))

        assert "MCP skill lacks connection/error guidance" not in messages

    @pytest.mark.parametrize(
        "statement",
        [
            "This skill avoids MCP.",
            "This skill avoids using MCP.",
            "This workflow excludes the MCP integration.",
            "MCP is disabled for this skill.",
            "MCP support remains unavailable.",
            "This workflow is MCP-free.",
        ],
    )
    def test_negative_mcp_capability_does_not_classify_mcp_skill(self, tmp_path: Path, statement: str):
        messages = _finding_messages(_write_issue_skill(tmp_path, extra_body=f"\n{statement}\n"))

        assert "MCP skill lacks connection/error guidance" not in messages

    def test_avoiding_mcp_failures_remains_mcp_usage(self, tmp_path: Path):
        messages = _finding_messages(
            _write_issue_skill(tmp_path, extra_body="\nAvoid MCP failures during validation.\n")
        )

        assert "MCP skill lacks connection/error guidance" in messages

    def test_fenced_mcp_example_does_not_classify_mcp_skill(self, tmp_path: Path):
        messages = _finding_messages(_write_issue_skill(tmp_path, extra_body="\n```text\nMCP example output\n```\n"))

        assert "MCP skill lacks connection/error guidance" not in messages

    def test_other_cli_mcp_comparison_does_not_classify_current_skill(self, tmp_path: Path):
        messages = _finding_messages(
            _write_issue_skill(
                tmp_path,
                name="authenticating-entra-device-code",
                extra_body=(
                    "\nNote: sharepoint-cli, gdrive-cli, and glean-cli use MaaS MCP auth "
                    "(via nv-discovery-cli), not Entra device code.\n"
                ),
            )
        )

        assert "MCP skill lacks connection/error guidance" not in messages

    def test_current_cli_mcp_comparison_remains_mcp_usage(self, tmp_path: Path):
        messages = _finding_messages(
            _write_issue_skill(
                tmp_path,
                name="sharepoint-cli",
                extra_body="\nsharepoint-cli uses MaaS MCP auth, not Entra device code.\n",
            )
        )

        assert "MCP skill lacks connection/error guidance" in messages

    def test_mcp_mention_in_html_comment_does_not_classify_skill(self, tmp_path: Path):
        messages = _finding_messages(
            _write_issue_skill(
                tmp_path,
                extra_body=(
                    "\n<!-- Related skills:\n- managing-sharepoint: SharePoint sites and lists (MaaS MCP)\n-->\n"
                ),
            )
        )

        assert "MCP skill lacks connection/error guidance" not in messages

    def test_mcp_guidance_in_html_comment_does_not_satisfy_active_usage(self, tmp_path: Path):
        messages = _finding_messages(
            _write_issue_skill(
                tmp_path,
                extra_body=(
                    "\nUse the MCP server to enumerate tools.\n\n"
                    "<!--\n## MCP Troubleshooting\n\n"
                    "Reconnect the server and retry after a timeout.\n-->\n"
                ),
            )
        )

        assert "MCP skill lacks connection/error guidance" in messages

    def test_interconnect_does_not_satisfy_mcp_connection_guidance(self, tmp_path: Path):
        messages = _finding_messages(
            _write_issue_skill(tmp_path, extra_body="\nUse the MCP server for GPU interconnect analysis.\n")
        )

        assert "MCP skill lacks connection/error guidance" in messages

    def test_unrelated_connection_text_does_not_satisfy_mcp_guidance(self, tmp_path: Path):
        messages = _finding_messages(
            _write_issue_skill(
                tmp_path,
                extra_body=(
                    "\nUse the MCP server to enumerate tools.\n\n"
                    "## Compilation Cache\n\nConnect to the compilation cache before measuring.\n"
                ),
            )
        )

        assert "MCP skill lacks connection/error guidance" in messages

    def test_unrelated_guidance_in_same_paragraph_does_not_satisfy_mcp_guidance(self, tmp_path: Path):
        messages = _finding_messages(
            _write_issue_skill(
                tmp_path,
                extra_body=(
                    "\n## Usage\n\n"
                    "- Use the MCP server to enumerate tools.\n"
                    "- Connect to the compilation cache before measuring.\n"
                ),
            )
        )

        assert "MCP skill lacks connection/error guidance" in messages

    def test_mcp_reconnect_guidance_is_still_accepted(self, tmp_path: Path):
        messages = _finding_messages(
            _write_issue_skill(
                tmp_path,
                extra_body="\nUse the MCP server. Reconnect the server if the session expires.\n",
            )
        )

        assert "MCP skill lacks connection/error guidance" not in messages

    def test_later_explicit_mcp_sentence_can_supply_guidance(self, tmp_path: Path):
        messages = _finding_messages(
            _write_issue_skill(
                tmp_path,
                extra_body=(
                    "\n## Usage\n\n"
                    "Use the MCP server to enumerate tools. "
                    "If the MCP server disconnects, reconnect it.\n"
                ),
            )
        )

        assert "MCP skill lacks connection/error guidance" not in messages

    def test_mcp_guidance_in_troubleshooting_section_is_accepted(self, tmp_path: Path):
        messages = _finding_messages(
            _write_issue_skill(
                tmp_path,
                extra_body=(
                    "\n## MCP Usage\n\nUse the MCP server to enumerate tools.\n\n"
                    "## Troubleshooting\n\nIf the connection drops, reconnect the server and retry.\n"
                ),
            )
        )

        assert "MCP skill lacks connection/error guidance" not in messages

    def test_mcp_guidance_in_explicitly_titled_support_section_is_accepted(self, tmp_path: Path):
        messages = _finding_messages(
            _write_issue_skill(
                tmp_path,
                extra_body=(
                    "\n## MCP Usage\n\nUse the MCP server to enumerate tools.\n\n"
                    "## MCP Troubleshooting\n\nReconnect the server and retry after a timeout.\n"
                ),
            )
        )

        assert "MCP skill lacks connection/error guidance" not in messages

    def test_unrelated_troubleshooting_does_not_satisfy_mcp_guidance(self, tmp_path: Path):
        messages = _finding_messages(
            _write_issue_skill(
                tmp_path,
                extra_body=(
                    "\n## MCP Usage\n\nUse the MCP server to enumerate tools.\n\n"
                    "## Troubleshooting\n\nRetry compilation if the local cache is unavailable.\n"
                ),
            )
        )

        assert "MCP skill lacks connection/error guidance" in messages

    def test_unrelated_build_server_troubleshooting_does_not_satisfy_mcp_guidance(self, tmp_path: Path):
        messages = _finding_messages(
            _write_issue_skill(
                tmp_path,
                extra_body=(
                    "\n## MCP Usage\n\nUse the MCP server to enumerate tools.\n\n"
                    "## Troubleshooting\n\nIf the build server fails, retry compilation.\n"
                ),
            )
        )

        assert "MCP skill lacks connection/error guidance" in messages

    @pytest.mark.parametrize(
        "statement",
        [
            "Unrolling stops after 2048 iterations.",
            "Stop after 2048 files.",
            "Retry after 2025 requests.",
            "Fail after 2000 data points.",
        ],
    )
    def test_four_digit_count_is_not_time_sensitive_information(self, tmp_path: Path, statement: str):
        messages = _finding_messages(_write_issue_skill(tmp_path, extra_body=f"\n{statement}\n"))

        assert "Time-sensitive information detected" not in messages

    def test_actual_year_reference_remains_time_sensitive(self, tmp_path: Path):
        messages = _finding_messages(
            _write_issue_skill(tmp_path, extra_body="\nUse this compatibility path after 2025.\n")
        )

        assert "Time-sensitive information detected" in messages

    def test_explicit_year_reference_with_following_plural_remains_time_sensitive(self, tmp_path: Path):
        messages = _finding_messages(
            _write_issue_skill(tmp_path, extra_body="\nAfter the year 2025, releases use this compatibility path.\n")
        )

        assert "Time-sensitive information detected" in messages

    def test_future_year_reference_remains_time_sensitive(self, tmp_path: Path):
        messages = _finding_messages(
            _write_issue_skill(tmp_path, extra_body="\nUse this compatibility path after 2101.\n")
        )

        assert "Time-sensitive information detected" in messages

    def test_replacing_deprecated_calls_is_not_exclusivity(self, tmp_path: Path):
        messages = _finding_messages(
            _write_issue_skill(tmp_path, extra_body="\nThis release replaces all deprecated launch calls.\n")
        )

        assert "Skill uses exclusivity language that conflicts with composability" not in messages

    @pytest.mark.parametrize(
        "statement",
        [
            "This release replaces all deprecated tool calls.",
            "This migration replaces all legacy build tools.",
            "This migration replaces all other deprecated tool calls.",
        ],
    )
    def test_replacing_obsolete_implementation_details_is_not_exclusivity(self, tmp_path: Path, statement: str):
        messages = _finding_messages(_write_issue_skill(tmp_path, extra_body=f"\n{statement}\n"))

        assert "Skill uses exclusivity language that conflicts with composability" not in messages

    def test_replacing_all_other_tools_remains_exclusivity(self, tmp_path: Path):
        messages = _finding_messages(
            _write_issue_skill(tmp_path, extra_body="\nThis skill replaces all other build tools.\n")
        )

        assert "Skill uses exclusivity language that conflicts with composability" in messages

    @pytest.mark.parametrize(
        "statement",
        [
            "This skill replaces all tools.",
            "This skill replaces all competing build tools.",
            "This skill replaces all alternative skills.",
            "This skill replaces all competing third-party tools.",
            "This skill replaces all currently available competing build tools.",
        ],
    )
    def test_replacing_all_tooling_remains_exclusivity(self, tmp_path: Path, statement: str):
        messages = _finding_messages(_write_issue_skill(tmp_path, extra_body=f"\n{statement}\n"))

        assert "Skill uses exclusivity language that conflicts with composability" in messages

    def test_readme_negative_guidance_is_not_a_reference(self, tmp_path: Path):
        skill_dir = _write_issue_skill(
            tmp_path,
            extra_body="\nREADME.md is human-facing; do not load it.\n",
        )
        (skill_dir / "README.md").write_text("# Human documentation\n")

        messages = _finding_messages(skill_dir)

        assert all("SKILL.md references README.md" not in message for message in messages)

    def test_readme_load_instruction_is_still_a_reference(self, tmp_path: Path):
        skill_dir = _write_issue_skill(tmp_path, extra_body="\nRead README.md before publishing.\n")
        (skill_dir / "README.md").write_text("# Human documentation\n")

        messages = _finding_messages(skill_dir)

        assert any("SKILL.md references README.md" in message for message in messages)

    def test_wrapped_readme_negative_guidance_is_not_a_reference(self, tmp_path: Path):
        skill_dir = _write_issue_skill(tmp_path, extra_body="\nDo not\nread README.md before publishing.\n")
        (skill_dir / "README.md").write_text("# Human documentation\n")

        messages = _finding_messages(skill_dir)

        assert all("SKILL.md references README.md" not in message for message in messages)

    def test_passive_readme_instruction_is_still_a_reference(self, tmp_path: Path):
        skill_dir = _write_issue_skill(tmp_path, extra_body="\nREADME.md should be read before publishing.\n")
        (skill_dir / "README.md").write_text("# Human documentation\n")

        messages = _finding_messages(skill_dir)

        assert any("SKILL.md references README.md" in message for message in messages)

    def test_passive_negative_readme_guidance_is_not_a_reference(self, tmp_path: Path):
        skill_dir = _write_issue_skill(tmp_path, extra_body="\nREADME.md should not be read by agents.\n")
        (skill_dir / "README.md").write_text("# Human documentation\n")

        messages = _finding_messages(skill_dir)

        assert all("SKILL.md references README.md" not in message for message in messages)

    @pytest.mark.parametrize(
        "statement",
        [
            "Agents should not read README.md.",
            "Agents must not load README.md.",
            "Agents shouldn't consult README.md.",
            "Agents should not ever read README.md.",
            "Agents should not refer to README.md.",
            "Agents don't need to read README.md.",
            "Agents are not required to read README.md.",
            "Agents are not allowed to load README.md.",
        ],
    )
    def test_modal_negative_readme_guidance_is_not_a_reference(self, tmp_path: Path, statement: str):
        skill_dir = _write_issue_skill(tmp_path, extra_body=f"\n{statement}\n")
        (skill_dir / "README.md").write_text("# Human documentation\n")

        messages = _finding_messages(skill_dir)

        assert all("SKILL.md references README.md" not in message for message in messages)

    def test_not_only_readme_instruction_remains_a_reference(self, tmp_path: Path):
        skill_dir = _write_issue_skill(
            tmp_path,
            extra_body="\nAgents should not only read README.md; they should also inspect the release notes.\n",
        )
        (skill_dir / "README.md").write_text("# Human documentation\n")

        messages = _finding_messages(skill_dir)

        assert any("SKILL.md references README.md" in message for message in messages)

    def test_affirmative_readme_instruction_after_negative_one_remains_a_reference(self, tmp_path: Path):
        skill_dir = _write_issue_skill(
            tmp_path,
            extra_body="\nDo not read README.md for setup; review README.md before publishing.\n",
        )
        (skill_dir / "README.md").write_text("# Human documentation\n")

        messages = _finding_messages(skill_dir)

        assert any("SKILL.md references README.md" in message for message in messages)

    def test_markdown_link_to_readme_suffix_is_not_a_readme_reference(self, tmp_path: Path):
        skill_dir = _write_issue_skill(tmp_path, extra_body="\nSee [release notes](NOTREADME.md).\n")
        (skill_dir / "README.md").write_text("# Human documentation\n")
        (skill_dir / "NOTREADME.md").write_text("# Release notes\n")

        messages = _finding_messages(skill_dir)

        assert all("SKILL.md references README.md" not in message for message in messages)

    def test_fenced_readme_link_example_is_not_a_readme_reference(self, tmp_path: Path):
        skill_dir = _write_issue_skill(
            tmp_path,
            extra_body="\n```markdown\nSee [human documentation](README.md).\n```\n",
        )
        (skill_dir / "README.md").write_text("# Human documentation\n")

        messages = _finding_messages(skill_dir)

        assert all("SKILL.md references README.md" not in message for message in messages)

    def test_commented_readme_link_is_not_a_readme_reference(self, tmp_path: Path):
        skill_dir = _write_issue_skill(
            tmp_path,
            extra_body="\n<!-- See [human documentation](README.md). -->\n",
        )
        (skill_dir / "README.md").write_text("# Human documentation\n")

        messages = _finding_messages(skill_dir)

        assert all("SKILL.md references README.md" not in message for message in messages)

    def test_markdown_link_parser_scans_malformed_nested_input_linearly(self):
        read_count = 0

        class CountingText(str):
            __slots__ = ()

            def __getitem__(self, key):
                nonlocal read_count
                if isinstance(key, int):
                    read_count += 1
                return super().__getitem__(key)

        content = CountingText("[x](" * 200)

        assert quality_score_module._markdown_link_targets(content) == []
        assert read_count <= len(content)

    def test_inline_backtick_span_at_line_start_does_not_mask_following_links(self):
        content = "```literal```\n[details](advanced.md)\n"

        assert quality_score_module._markdown_link_targets(content) == ["advanced.md"]

    def test_shortcut_reference_style_link_resolves_definition(self):
        content = "See [readme] for details.\n\n[readme]: README.md\n"

        assert quality_score_module._markdown_link_targets(content) == ["README.md"]

    @pytest.mark.parametrize(
        "target",
        [
            "https://agentskills.io/specification.md",
            "../SKILL.md",
        ],
    )
    def test_external_and_parent_markdown_links_are_not_nested_references(self, tmp_path: Path, target: str):
        skill_dir = _write_issue_skill(tmp_path)
        references = skill_dir / "references"
        references.mkdir()
        (references / "mechanisms.md").write_text(f"See [the specification]({target}).\n")

        messages = _finding_messages(skill_dir)

        assert all("Deeply nested references" not in message for message in messages)

    def test_local_markdown_link_remains_a_nested_reference(self, tmp_path: Path):
        skill_dir = _write_issue_skill(tmp_path)
        references = skill_dir / "references"
        references.mkdir()
        (references / "mechanisms.md").write_text("See [more details](advanced.md).\n")

        messages = _finding_messages(skill_dir)

        assert "Deeply nested references in mechanisms.md" in messages

    @pytest.mark.parametrize(
        "target",
        [
            "../other.md",
            "nested/SKILL.md",
            "guide(v2).md",
        ],
    )
    def test_other_local_markdown_paths_remain_nested_references(self, tmp_path: Path, target: str):
        skill_dir = _write_issue_skill(tmp_path)
        references = skill_dir / "references"
        references.mkdir()
        (references / "mechanisms.md").write_text(f"See [more details]({target}).\n")

        messages = _finding_messages(skill_dir)

        assert "Deeply nested references in mechanisms.md" in messages

    def test_external_url_with_balanced_parentheses_is_not_nested(self, tmp_path: Path):
        skill_dir = _write_issue_skill(tmp_path)
        references = skill_dir / "references"
        references.mkdir()
        (references / "mechanisms.md").write_text("See [the specification](https://example.com/spec(v2).md).\n")

        messages = _finding_messages(skill_dir)

        assert all("Deeply nested references" not in message for message in messages)

    def test_fenced_markdown_link_example_is_not_nested(self, tmp_path: Path):
        skill_dir = _write_issue_skill(tmp_path)
        references = skill_dir / "references"
        references.mkdir()
        (references / "mechanisms.md").write_text("```markdown\n[example](advanced.md)\n```\n")

        messages = _finding_messages(skill_dir)

        assert all("Deeply nested references" not in message for message in messages)

    def test_commented_markdown_link_is_not_nested(self, tmp_path: Path):
        skill_dir = _write_issue_skill(tmp_path)
        references = skill_dir / "references"
        references.mkdir()
        (references / "mechanisms.md").write_text("<!-- [example](advanced.md) -->\n")

        messages = _finding_messages(skill_dir)

        assert all("Deeply nested references" not in message for message in messages)

    def test_reference_style_local_markdown_link_is_nested(self, tmp_path: Path):
        skill_dir = _write_issue_skill(tmp_path)
        references = skill_dir / "references"
        references.mkdir()
        (references / "mechanisms.md").write_text("See [more details][advanced].\n\n[advanced]: advanced.md\n")

        messages = _finding_messages(skill_dir)

        assert "Deeply nested references in mechanisms.md" in messages

    def test_instruction_action_verbs_do_not_match_inside_words(self, tmp_path: Path):
        messages = _finding_messages(
            _write_issue_skill(
                tmp_path,
                instructions="- Because the offset is a runtime property, results vary.",
            )
        )

        assert "Instructions lack clear action verbs" in messages

    @pytest.mark.parametrize("action", ["Add", "Mark", "Update", "Keep"])
    def test_task_list_imperatives_are_clear_action_verbs(self, tmp_path: Path, action: str):
        messages = _finding_messages(
            _write_issue_skill(
                tmp_path,
                instructions=f"- {action} task status as each step completes.",
            )
        )

        assert "Instructions lack clear action verbs" not in messages
