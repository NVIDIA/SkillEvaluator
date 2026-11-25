# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: RUF001 -- the fixtures are written in Russian and Chinese on purpose (issue #865).

"""Tests for the language-neutral word lists of the quality score (issue #865)."""

from pathlib import Path

import pytest
from tests.validators.test_quality_score import _finding_messages, _validate_with, _write_issue_skill

from skillevaluator.validators.quality_score import _contains_any_term


def _write_localized_skill(tmp_path: Path, *, name: str, description: str, body: str) -> Path:
    """Write a complete guide skill whose prose and headings are not in English."""
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
        "    - build\n"
        "---\n\n"
        f"{body}",
        encoding="utf-8",
    )
    return skill_dir


RUSSIAN_BODY = (
    "## Назначение\n\nИзмеряет время сборки.\n\n"
    "## Инструкции\n\n- Запусти пробу.\n- Прочитай отчёт.\n\n"
    "## Примеры\n\n```bash\nprobe measure -- python app.py\n```\n\n"
    "## Требования\n\nНет.\n\n"
    "## Ограничения\n\nНеизвестны.\n\n"
    "## Устранение неполадок\n\nЕсли возникает ошибка, прочитай вывод диагностики.\n"
)
CHINESE_BODY = (
    "## 目的\n\n测量构建时间。\n\n"
    "## 使用说明\n\n- 运行探针。\n- 查看报告。\n\n"
    "## 示例\n\n```bash\nprobe measure -- python app.py\n```\n\n"
    "## 前提条件\n\n无。\n\n"
    "## 限制\n\n暂无已知限制。\n\n"
    "## 故障排除\n\n如果出现错误，请查看诊断输出。\n"
)
LANGUAGE_SENSITIVE_MESSAGES = (
    "Description doesn't mention WHEN to use this skill",
    "No mention of error handling or validation",
    "No troubleshooting section documented",
    "No prerequisites/requirements documented",
    "No limitations documented",
    "No '## Purpose' section",
)


def _dimension_scores(result) -> dict[str, float]:
    return {area: scores["score"] for area, scores in result.metadata["quality_scores"]["dimensions"].items()}


class TestLanguageNeutralWordLists:
    """Issue #865: well-formed Russian and Chinese skills lost points to English-only word lists."""

    def test_russian_skill_keeps_trigger_error_handling_and_heading_points(self, tmp_path: Path):
        skill_dir = _write_localized_skill(
            tmp_path,
            name="proba-sborki",
            description="Используй для анализа журналов сборки и поиска лишних шагов.",
            body=RUSSIAN_BODY,
        )

        result = _validate_with(skill_dir, tmp_path)

        messages = [finding.message for finding in result.findings]
        assert not any(message.startswith(LANGUAGE_SENSITIVE_MESSAGES) for message in messages), messages
        assert result.metadata["quality_scores"]["metrics"]["has_error_handling"] is True
        assert _dimension_scores(result)["discoverability"] == 100.0
        assert _dimension_scores(result)["reliability"] == 100.0

    def test_chinese_skill_keeps_trigger_error_handling_and_heading_points(self, tmp_path: Path):
        skill_dir = _write_localized_skill(
            tmp_path,
            name="build-probe-zh",
            description="当需要分析构建日志时使用此技能，用于查找多余的步骤。",
            body=CHINESE_BODY,
        )

        result = _validate_with(skill_dir, tmp_path)

        messages = [finding.message for finding in result.findings]
        assert not any(message.startswith(LANGUAGE_SENSITIVE_MESSAGES) for message in messages), messages
        assert result.metadata["quality_scores"]["metrics"]["has_error_handling"] is True
        assert _dimension_scores(result)["discoverability"] == 100.0
        assert _dimension_scores(result)["reliability"] == 100.0

    def test_chinese_terms_match_inside_a_run_of_characters(self, tmp_path: Path):
        """Chinese has no spaces between words, so a term matches without a word boundary."""
        skill_dir = _write_issue_skill(tmp_path, troubleshooting="探针会报告错误信息并停止。")

        result = _validate_with(skill_dir, tmp_path)

        assert result.metadata["quality_scores"]["metrics"]["has_error_handling"] is True

    @pytest.mark.parametrize(
        ("text", "term"),
        [
            ("报告错误信息并处理异常", "错误"),
            ("Используй для анализа.", "для"),
            ("не\nудалось запустить", "не удалось"),
            ("built for speed", "for"),
        ],
    )
    def test_terms_match_across_scripts(self, text: str, term: str):
        assert _contains_any_term(text, [term])

    @pytest.mark.parametrize(
        ("text", "term"),
        [
            ("платформа", "для"),  # Cyrillic word boundaries are honoured
            ("некогда", "когда"),
            ("performance", "for"),  # English stays whole-word
            ("forêt", "for"),
            ("处理error时", "error"),  # an English term does not match inside Chinese text
        ],
    )
    def test_terms_do_not_match_inside_other_words(self, text: str, term: str):
        assert not _contains_any_term(text, [term])

    def test_russian_vague_words_are_still_reported(self, tmp_path: Path):
        skill_dir = _write_issue_skill(tmp_path, description="Делает разное и что-то общее для проекта.")

        assert "Description contains vague words" in _finding_messages(skill_dir)

    def test_english_skill_scores_exactly_as_before(self, tmp_path: Path):
        """The former hard-coded lists are the front of each packaged list, so English scores do not move."""
        clean = _validate_with(_write_issue_skill(tmp_path, name="probe-clean"), tmp_path)
        weak = _validate_with(
            _write_issue_skill(
                tmp_path,
                name="probe-weak",
                description=(
                    "Handles various data things across the whole project tree, "
                    "so you can manage documents and files in one place."
                ),
                instructions="- Because the offset is a runtime property, results vary.",
                troubleshooting="Leverage the diagnostic output when the probe reports a problem.",
            ),
            tmp_path,
        )

        assert clean.metadata["quality_scores"]["overall_score"] == 100.0
        assert clean.findings == []
        assert weak.metadata["quality_scores"]["overall_score"] == 83.2
        assert _dimension_scores(weak) == {
            "correctness": 100.0,
            "discoverability": 55.0,
            "reliability": 90.0,
            "efficiency": 80.0,
        }
        assert sorted(finding.message for finding in weak.findings) == sorted(
            [
                "Description doesn't mention WHEN to use this skill",
                "Description contains vague words",
                "Description uses first/second person",
                "Broad description without negative triggers may cause over-triggering",
                "No mention of error handling or validation",
                "Instructions lack clear action verbs",
                "Uses complex/corporate language",
            ]
        )

    def test_custom_trigger_words_replace_the_packaged_list(self, tmp_path: Path):
        config = "discoverability:\n  trigger_words:\n    terms: [whenever]\n    message: No trigger phrase\n"
        packaged_wording = _write_issue_skill(tmp_path, name="probe-a", description="Use when mapping kernels.")
        custom_wording = _write_issue_skill(tmp_path, name="probe-b", description="Whenever kernels need mapping.")

        packaged_result = _validate_with(packaged_wording, tmp_path, config)
        custom_result = _validate_with(custom_wording, tmp_path, config)

        assert "No trigger phrase" in [finding.message for finding in packaged_result.findings]
        assert "No trigger phrase" not in [finding.message for finding in custom_result.findings]
        assert packaged_result.metadata["quality_config"]["source"] == "custom"

    def test_word_list_rule_points_zero_turns_the_rule_off(self, tmp_path: Path):
        skill_dir = _write_issue_skill(tmp_path, description="Maps kernels to modules and finds redundant builds.")

        packaged = _validate_with(skill_dir, tmp_path)
        disabled = _validate_with(skill_dir, tmp_path, "discoverability: {trigger_words: {points: 0}}\n")

        assert "Description doesn't mention WHEN to use this skill" in [f.message for f in packaged.findings]
        assert "Description doesn't mention WHEN to use this skill" not in [f.message for f in disabled.findings]
        assert _dimension_scores(disabled)["discoverability"] == _dimension_scores(packaged)["discoverability"] + 10

    def test_custom_purpose_headings_replace_the_packaged_list(self, tmp_path: Path):
        skill_dir = _write_issue_skill(tmp_path)
        config = 'discoverability:\n  purpose_heading:\n    headings: ["## Why"]\n    message: No why section\n'

        result = _validate_with(skill_dir, tmp_path, config)

        assert "No why section" in [finding.message for finding in result.findings]
        assert "No '## Purpose' section" not in [finding.message for finding in result.findings]


@pytest.mark.parametrize(
    ("path_name", "desc", "body", "example", "scripts_heading", "scripts_header"),
    [
        (
            "russian-scripts",
            "Используй для подготовки отчётов по сборке.",
            RUSSIAN_BODY,
            "Пример: отчёт содержит время сборки.",
            "## Доступные скрипты",
            "Скрипт",
        ),
        (
            "chinese-scripts",
            "当需要分析构建日志时使用此技能，用于查找多余的步骤。",
            CHINESE_BODY,
            "示例：报告包含构建时间。",
            "## 可用脚本",
            "脚本",
        ),
    ],
)
def test_localized_examples_script_documentation_and_instruction_options(
    tmp_path, path_name, desc, body, example, scripts_heading, scripts_header
):
    body = body.replace("```bash\nprobe measure -- python app.py\n```", example)
    body += f"\n{scripts_heading}\n\n| {scripts_header} | Purpose |\n| --- | --- |\n| audit.py | Report |\n"
    body += "\nUse run_script for audit.py.\n"
    skill = _write_localized_skill(tmp_path, name=path_name, description=desc, body=body)
    (skill / "scripts").mkdir()
    (skill / "scripts" / "audit.py").write_text("print('report')\n", encoding="utf-8")
    result = _validate_with(skill, tmp_path, "efficiency: {action_verbs: {terms: [verify]}}\n")
    messages = [finding.message for finding in result.findings]

    assert result.metadata["quality_scores"]["metrics"]["has_instructions"] is True
    assert result.metadata["quality_scores"]["metrics"]["has_examples"] is True
    assert "No examples provided" not in messages
    assert "No documented scripts in table format" not in messages
    assert "Instructions lack clear action verbs" in messages


@pytest.mark.parametrize(
    ("description", "expected", "excluded"),
    [
        ("当前版本生成按阶段汇总的详细构建耗时报告。", "Description doesn't mention WHEN to use this skill", None),
        ("当排查东西向网络流量丢包时，按租户生成连接追踪报告。", None, "Description contains vague words"),
        ("Use when You  can map kernels for the build.", None, "Description uses first/second person"),
    ],
)
def test_multilingual_trigger_and_description_boundaries(tmp_path, description, expected, excluded):
    result = _validate_with(_write_issue_skill(tmp_path, description=description), tmp_path)
    messages = {finding.message for finding in result.findings}
    if expected:
        assert expected in messages
    if excluded:
        assert excluded not in messages


def test_instruction_rules_use_real_markdown_sections_and_keep_legacy_example_credit(tmp_path):
    body = (
        "```markdown\n## Instructions\n- verify\n```\n\n"
        "##  Instructions ##\n\n- Read the summary carefully.\n\n"
        "## Examples ##\n\nA worked report explains the build order.\n"
    )
    skill = _write_localized_skill(
        tmp_path, name="marked-sections", description="Use when generating build reports.", body=body
    )
    result = _validate_with(skill, tmp_path, "efficiency: {action_verbs: {terms: [verify]}}\n")

    assert result.metadata["quality_scores"]["metrics"]["has_instructions"] is True
    assert "Instructions lack clear action verbs" in [finding.message for finding in result.findings]
    assert "No examples provided" not in [finding.message for finding in result.findings]


def test_empty_final_instructions_section_has_no_efficiency_deduction(tmp_path):
    skill = _write_localized_skill(
        tmp_path,
        name="empty-final-instructions",
        description="Use when checking build reports.",
        body="## Examples\n\nA worked report.\n\n## Instructions\n",
    )
    result = _validate_with(skill, tmp_path)
    messages = [finding.message for finding in result.findings]

    assert "Instructions lack clear action verbs" not in messages
    assert "Instructions not in list format" not in messages


def test_markdown_section_offsets_keep_unicode_line_separators_in_prose(tmp_path):
    skill = _write_localized_skill(
        tmp_path,
        name="unicode-sections",
        description="Use when checking build reports.",
        body=(
            "Overview.\u2028Diagnostic notes.\u2029Report summary.\n\n"
            "## Instructions\n\n- Run the probe.\n\n## Examples\n\nA worked report.\n"
        ),
    )
    result = _validate_with(skill, tmp_path)
    messages = [finding.message for finding in result.findings]

    assert result.metadata["quality_scores"]["metrics"]["has_examples"] is True
    assert "Instructions lack clear action verbs" not in messages
    assert "Instructions not in list format" not in messages


@pytest.mark.parametrize("examples", ["## Examples ##\n\nA worked report.", "## Examples\n\n## 範例\n\n報表範例。"])
def test_plain_prose_examples_count_under_recognized_headings(tmp_path, examples):
    skill = _write_localized_skill(
        tmp_path,
        name="prose-examples",
        description="Use when writing build reports.",
        body=examples,
    )
    result = _validate_with(skill, tmp_path)

    assert result.metadata["quality_scores"]["metrics"]["has_examples"] is True
    assert "No examples provided" not in [finding.message for finding in result.findings]
