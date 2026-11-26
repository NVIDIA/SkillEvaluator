# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: RUF001 -- the fixtures are written in Russian and Chinese on purpose (issue #865).

"""Tests for the language-neutral word lists of the quality score (issue #865)."""

from pathlib import Path
from unicodedata import normalize

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

    @pytest.mark.parametrize(
        ("headings", "description"),
        [
            (
                ("概述", "使用方法", "使用示例", "前置条件", "注意事项", "常见问题"),
                "当需要分析构建日志时使用此技能，用于查找多余的步骤。",
            ),
            (
                ("概述", "使用方法", "使用範例", "前置條件", "注意事項", "常見問題"),
                "當需要分析建置日誌時使用此技能，用於找出多餘的步驟。",
            ),
        ],
    )
    def test_common_chinese_section_headings_and_prose_examples(
        self, tmp_path: Path, headings: tuple[str, ...], description: str
    ):
        purpose, instructions, examples, prerequisites, limitations, troubleshooting = headings
        body = (
            f"## {purpose}\n\n測量建置時間。\n\n"
            f"## {instructions}\n\n- 執行探針。\n- 查看報告。\n\n"
            f"## {examples}\n\n報告顯示每個步驟所花的時間。\n\n"
            f"## {prerequisites}\n\n需要先取得建置記錄。\n\n"
            f"## {limitations}\n\n無法分析尚未完成的建置。\n\n"
            f"## {troubleshooting}\n\n遇到錯誤時檢查診斷記錄。\n"
        )
        result = _validate_with(
            _write_localized_skill(tmp_path, name="build-probe-zh", description=description, body=body),
            tmp_path,
        )

        assert result.findings == []
        metrics = result.metadata["quality_scores"]["metrics"]
        assert metrics["has_instructions"] and metrics["has_examples"]
        assert result.metadata["quality_scores"]["overall_score"] == 100.0

    @pytest.mark.parametrize(
        ("action", "troubleshooting"),
        [
            ("Запустить пробу.", "Скрипт может вернуть ошибку."),
            ("Использовать форматтер.", "При ошибках смотреть журнал."),
            ("Откройте журнал.", "Повторить запрос после сбоя."),
            ("Запускай пробу.", "Сохранить сведения об исключении."),
            ("Открыть отчёт.", "Изучить файл с исключениями."),
        ],
    )
    def test_russian_procedures_and_inflected_error_terms(self, tmp_path: Path, action: str, troubleshooting: str):
        body = RUSSIAN_BODY.replace("- Запусти пробу.", f"- {action}").replace(
            "Если возникает ошибка, прочитай вывод диагностики.", troubleshooting
        )
        result = _validate_with(
            _write_localized_skill(
                tmp_path,
                name="proba-sborki",
                description="Используй для анализа журналов сборки и поиска лишних шагов.",
                body=body,
            ),
            tmp_path,
        )

        assert result.findings == []
        assert result.metadata["quality_scores"]["metrics"]["has_error_handling"] is True
        assert _dimension_scores(result)["efficiency"] == 100.0

    def test_chinese_terms_match_inside_a_run_of_characters(self, tmp_path: Path):
        """Chinese has no spaces between words, so a term matches without a word boundary."""
        skill_dir = _write_issue_skill(tmp_path, troubleshooting="探针会报告错误信息并停止。")

        result = _validate_with(skill_dir, tmp_path)

        assert result.metadata["quality_scores"]["metrics"]["has_error_handling"] is True

    @pytest.mark.parametrize(
        ("text", "term"),
        [
            ("报告错误信息并处理异常", "错误"),
            ("处理error时", "error"),
            ("从目录中RUN探针。", "run"),
            ("利用APIします", "api"),
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
            ("在forêt中", "for"),
            ("在платформа中", "форма"),
            ("处理error_code时", "error"),
            ("处理error2时", "error"),
        ],
    )
    def test_terms_do_not_match_inside_other_words(self, text: str, term: str):
        assert not _contains_any_term(text, [term])

    def test_english_terms_in_chinese_error_and_instruction_text(self, tmp_path: Path):
        body = CHINESE_BODY.replace("运行探针", "从目录中RUN探针").replace("出现错误", "出现error")
        skill_dir = _write_localized_skill(
            tmp_path,
            name="build-probe-zh",
            description="当需要分析构建日志时使用此技能，用于查找多余的步骤。",
            body=body,
        )

        result = _validate_with(skill_dir, tmp_path)

        assert result.findings == []
        assert result.metadata["quality_scores"]["metrics"]["has_error_handling"] is True
        assert _dimension_scores(result)["efficiency"] == 100.0

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
        "```markdown\n## Steps\n- verify\n```\n\n"
        "##  Steps ##\n\n- Read the summary carefully.\n\n"
        "## Examples ##\n\nA worked report explains the build order.\n"
    )
    skill = _write_localized_skill(
        tmp_path, name="marked-sections", description="Use when generating build reports.", body=body
    )
    result = _validate_with(skill, tmp_path, 'efficiency: {action_verbs: {headings: ["## Steps"], terms: [verify]}}\n')

    assert result.metadata["quality_scores"]["metrics"]["has_instructions"] is True
    assert "Instructions lack clear action verbs" in [finding.message for finding in result.findings]
    assert "No examples provided" not in [finding.message for finding in result.findings]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("## Instructions ##\n\nAn overview appears here.\n", set()),
        ("## Usage ###\n\nAn overview appears here.\n", set()),
        ("## Instructions ##\n\nAn overview appears here.\n\n## Usage\n\nAn overview appears here.\n", set()),
        (
            "## Instructions ##\n\n- Run the report.\n\n## Instructions\n\nAn overview appears here.\n",
            {"Instructions lack clear action verbs", "Instructions not in list format"},
        ),
    ],
)
def test_legacy_english_instruction_scoring_uses_literal_heading_lines(tmp_path, body, expected):
    skill = _write_localized_skill(
        tmp_path, name="legacy-english-heading", description="Use when checking build reports.", body=body
    )

    result = _validate_with(skill, tmp_path)

    assert result.metadata["quality_scores"]["metrics"]["has_instructions"] is True
    assert {finding.message for finding in result.findings if finding.check_name == "quality_efficiency"} == expected


@pytest.mark.parametrize("form", ["NFC", "NFD"])
@pytest.mark.parametrize("decomposed_config", [False, True])
def test_localized_action_verb_matches_canonical_equivalent_text(tmp_path, form, decomposed_config):
    body = "## Инструкции\n\n- " + normalize(form, "Используй отчёт.") + "\n"
    skill = _write_localized_skill(
        tmp_path, name="localized-action", description="Use when checking build reports.", body=body
    )
    term = normalize("NFD", "используй")
    custom = f"efficiency: {{action_verbs: {{terms: [{term}]}}}}\n" if decomposed_config else None

    result = _validate_with(skill, tmp_path, custom)

    assert "Instructions lack clear action verbs" not in [finding.message for finding in result.findings]


@pytest.mark.parametrize("form", ["NFC", "NFD"])
def test_localized_error_handling_matches_canonical_equivalent_text(tmp_path, form):
    skill = _write_localized_skill(
        tmp_path,
        name="localized-format",
        description="Use when checking build reports.",
        body="## Устранение неполадок\n\n" + normalize(form, "Формат невалидный.") + "\n",
    )

    result = _validate_with(skill, tmp_path)

    assert result.metadata["quality_scores"]["metrics"]["has_error_handling"] is True
    assert "No mention of error handling or validation" not in [finding.message for finding in result.findings]


@pytest.mark.parametrize("heading", ["Instructions", "Инструкции", "使用说明"])
def test_action_verb_heading_override_keeps_list_format_check(tmp_path, heading):
    skill = _write_localized_skill(
        tmp_path,
        name="independent-instruction-checks",
        description="Use when checking build reports.",
        body=f"## {heading}\n\nAn overview appears here.\n\n## Steps\n\n- Run the report.\n",
    )
    result = _validate_with(skill, tmp_path, 'efficiency: {action_verbs: {headings: ["## Steps"]}}\n')
    messages = {finding.message for finding in result.findings}

    assert "Instructions not in list format" in messages
    assert "Instructions lack clear action verbs" not in messages


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
