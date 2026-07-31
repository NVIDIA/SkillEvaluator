# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for QualityScoreValidator -- 4-dimension skill quality analysis."""

from pathlib import Path

import pytest

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
