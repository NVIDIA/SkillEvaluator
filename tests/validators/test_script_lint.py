# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ScriptLintValidator -- AST-based advisory lint checks."""

from pathlib import Path

import pytest

from skillevaluator.models.result import Severity, ValidationResult
from skillevaluator.validators.script_lint import ScriptLintValidator


@pytest.fixture
def skill_with_scripts(tmp_path: Path) -> Path:
    """Skill with scripts/ directory containing test Python files."""
    skill_dir = tmp_path / "scripted-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: scripted-skill\ndescription: test\n---\n\n# Test\n")
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    return skill_dir


class TestScriptLintValidator:
    def test_no_scripts_dir(self, tmp_path):
        skill_dir = tmp_path / "no-scripts"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: no-scripts\ndescription: test\n---\n")
        v = ScriptLintValidator()
        result = v.validate(skill_dir)
        assert result.passed
        assert not result.findings

    def test_node_scripts_reported_as_not_linted(self, skill_with_scripts):
        scripts = skill_with_scripts / "scripts"
        (scripts / "a.mjs").write_text("console.log('a')\n")
        (scripts / "b.js").write_text("console.log('b')\n")
        result = ScriptLintValidator().validate(skill_with_scripts)
        assert result.passed
        assert not result.findings
        messages = [d.message for d in result.success_details]
        assert any("2 Node script(s) not linted" in m for m in messages), messages

    def test_node_scripts_reported_alongside_python_lint_findings(self, skill_with_scripts):
        scripts = skill_with_scripts / "scripts"
        (scripts / "flat.py").write_text("#!/usr/bin/env python3\nimport sys\nprint(sys.argv)\n")
        (scripts / "a.js").write_text("console.log('a')\n")
        result = ScriptLintValidator().validate(skill_with_scripts)
        checks = [f.check_name for f in result.findings]
        assert "flat_script" in checks
        (note,) = [f for f in result.findings if f.check_name == "node_scripts_not_linted"]
        assert note.severity == Severity.INFO
        assert "1 Node script(s) not linted" in note.message

    def test_node_note_survives_folder_aggregation(self, skill_with_scripts):
        scripts = skill_with_scripts / "scripts"
        (scripts / "flat.py").write_text("#!/usr/bin/env python3\nimport sys\nprint(sys.argv)\n")
        (scripts / "a.js").write_text("console.log('a')\n")
        single = ScriptLintValidator().validate(skill_with_scripts)
        folder = ValidationResult()
        folder.merge_with_prefix(single, "skill-a")
        assert any(
            f.check_name == "node_scripts_not_linted" and f.file_path.startswith("[skill-a]") for f in folder.findings
        )

    def test_node_scripts_do_not_add_a_second_completed_check(self, skill_with_scripts):
        scripts = skill_with_scripts / "scripts"
        (scripts / "a.mjs").write_text("console.log('a')\n")
        with_node = ScriptLintValidator().validate(skill_with_scripts)
        (scripts / "a.mjs").unlink()
        without_node = ScriptLintValidator().validate(skill_with_scripts)
        assert with_node.summary.checks_performed == without_node.summary.checks_performed
        assert len(with_node.success_details) == len(without_node.success_details)

    def test_node_scripts_in_tools_dir_and_clean_python_are_both_reported(self, tmp_path):
        skill_dir = tmp_path / "tools-skill"
        tools = skill_dir / "tools"
        tools.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: tools-skill\ndescription: test\n---\n\n# Test\n")
        (tools / "ok.py").write_text(
            "#!/usr/bin/env python3\n"
            "import argparse\n\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('input')\n"
            "    args = parser.parse_args()\n"
            "    print(args.input)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        (tools / "a.mjs").write_text("export {}\n")
        (tools / "b.cjs").write_text("module.exports = {}\n")
        result = ScriptLintValidator().validate(skill_dir)
        messages = [d.message for d in result.success_details]
        assert any("passed lint checks" in m for m in messages), messages
        assert any("2 Node script(s) not linted" in m for m in messages), messages

    def test_flat_script_finding(self, skill_with_scripts):
        scripts = skill_with_scripts / "scripts"
        (scripts / "flat.py").write_text("#!/usr/bin/env python3\nimport sys\nprint(sys.argv)\n")
        v = ScriptLintValidator()
        result = v.validate(skill_with_scripts)
        checks = [f.check_name for f in result.findings]
        assert "flat_script" in checks

    def test_flat_script_finding_in_tools_dir(self, tmp_path):
        skill_dir = tmp_path / "tools-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: tools-skill\ndescription: test\n---\n\n# Test\n")
        tools = skill_dir / "tools"
        tools.mkdir()
        (tools / "flat.py").write_text("#!/usr/bin/env python3\nimport sys\nprint(sys.argv)\n")
        v = ScriptLintValidator()
        result = v.validate(skill_dir)
        checks = [f.check_name for f in result.findings]
        assert "flat_script" in checks

    def test_deep_nesting_finding(self, skill_with_scripts):
        scripts = skill_with_scripts / "scripts"
        nested_code = (
            "#!/usr/bin/env python3\n"
            "import argparse\n"
            "def main():\n"
            "    for i in range(10):\n"
            "        for j in range(10):\n"
            "            for k in range(10):\n"
            "                if i > 0:\n"
            "                    if j > 0:\n"
            "                        if k > 0:\n"
            "                            while True:\n"
            "                                print(i, j, k)\n"
            "                                break\n"
        )
        (scripts / "nested.py").write_text(nested_code)
        v = ScriptLintValidator()
        result = v.validate(skill_with_scripts)
        checks = [f.check_name for f in result.findings]
        assert "deep_nesting" in checks

    def test_magic_numbers_finding(self, skill_with_scripts):
        scripts = skill_with_scripts / "scripts"
        (scripts / "magic.py").write_text(
            "#!/usr/bin/env python3\nimport argparse\ndef main():\n    x = 42\n    y = 3.14159\n"
        )
        v = ScriptLintValidator()
        result = v.validate(skill_with_scripts)
        checks = [f.check_name for f in result.findings]
        assert "magic_numbers" in checks

    def test_missing_shebang_finding(self, skill_with_scripts):
        scripts = skill_with_scripts / "scripts"
        (scripts / "noshebang.py").write_text("import argparse\ndef main():\n    pass\n")
        v = ScriptLintValidator()
        result = v.validate(skill_with_scripts)
        checks = [f.check_name for f in result.findings]
        assert "missing_shebang" in checks

    def test_no_input_validation_finding(self, skill_with_scripts):
        scripts = skill_with_scripts / "scripts"
        (scripts / "noval.py").write_text("#!/usr/bin/env python3\ndef main():\n    print('hi')\n")
        v = ScriptLintValidator()
        result = v.validate(skill_with_scripts)
        checks = [f.check_name for f in result.findings]
        assert "no_input_validation" in checks

    def test_clean_script_no_findings(self, skill_with_scripts):
        scripts = skill_with_scripts / "scripts"
        (scripts / "clean.py").write_text(
            "#!/usr/bin/env python3\n"
            "import argparse\n\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('input')\n"
            "    args = parser.parse_args()\n"
            "    print(args.input)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        v = ScriptLintValidator()
        result = v.validate(skill_with_scripts)
        assert not result.findings

    def test_syntax_error_skipped(self, skill_with_scripts):
        scripts = skill_with_scripts / "scripts"
        (scripts / "broken.py").write_text("def broken(:\n    pass\n")
        v = ScriptLintValidator()
        result = v.validate(skill_with_scripts)
        # Syntax errors should be silently skipped
        assert "read_error" not in [f.check_name for f in result.findings]

    def test_validator_name(self):
        v = ScriptLintValidator()
        assert "Lint" in v.name
        assert "Advisory" in v.name
