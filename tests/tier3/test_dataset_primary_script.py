# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dataset generator never names an import-only helper module as the expected script."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.tier3 import generate_dataset

API_CALLER = Path(generate_dataset.__file__).parent / "reference_skills" / "api-caller"
ENTRY_POINTS = {"call_api.py", "parse_openapi.py"}

HELPER = '"""Shared checks."""\n\nfrom urllib.parse import urlparse\n\n\ndef check(url):\n    return urlparse(url)\n'
GUARDED_ENTRY = "import sys\n\n\ndef main():\n    return 0\n\n\nif __name__ == '__main__':\n    sys.exit(main())\n"


def _write_skill(tmp_path: Path, scripts: dict[str, str], skill_md_body: str = "# Skill\n") -> Path:
    skill = tmp_path / "demo-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        f"---\nname: demo-skill\ndescription: Demo skill.\n---\n\n{skill_md_body}",
        encoding="utf-8",
    )
    for relative, source in scripts.items():
        path = skill / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return skill


def _parse_in_order(skill_path: Path, order: tuple[str, ...]) -> dict[str, Any]:
    """Parse a skill, then list its scripts in `order`, because the directory listing order is not defined."""
    parsed = generate_dataset._parse_skill(skill_path)
    assert sorted(parsed["scripts"]) == sorted(order)
    parsed["scripts"] = list(order)
    return parsed


def test_api_caller_reference_skill_never_expects_the_url_helper() -> None:
    """Whatever order the file system lists scripts in, url_safety.py is not the expected script."""
    parsed = generate_dataset._parse_skill(API_CALLER)

    assert "url_safety.py" in parsed["scripts"]
    assert generate_dataset._pick_primary_script(parsed) in ENTRY_POINTS


@pytest.mark.parametrize("order", list(itertools.permutations(["call_api.py", "parse_openapi.py", "url_safety.py"])))
def test_api_caller_helper_listed_first_is_skipped(order: tuple[str, ...]) -> None:
    parsed = _parse_in_order(API_CALLER, order)

    assert generate_dataset._pick_primary_script(parsed) == next(name for name in order if name in ENTRY_POINTS)


@pytest.mark.parametrize("generate", [generate_dataset._generate_simple, generate_dataset._generate_full])
def test_generated_cases_expect_the_entry_point_not_the_helper(generate: Any) -> None:
    parsed = _parse_in_order(API_CALLER, ("url_safety.py", "parse_openapi.py", "call_api.py"))

    scripts = {case["expected_script"] for case in generate(parsed) if case["expected_skill"]}

    assert scripts == {"parse_openapi.py"}


@pytest.mark.parametrize(
    "helper",
    [
        pytest.param(HELPER, id="functions-and-docstring"),
        pytest.param("", id="empty-init"),
        pytest.param("LIMIT = 10\nNAMES: list[str] = []\nNAMES += ['a']\n", id="constants-only"),
        pytest.param(
            "import logging\n\nlogger = logging.getLogger(__name__)\n\n\nclass Config:\n    retries = 3\n",
            id="logger-and-class",
        ),
        pytest.param(
            "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from pathlib import Path\n\n\n"
            "def name(path: 'Path') -> str:\n    return path.name\n",
            id="type-checking-block",
        ),
        pytest.param(
            "try:\n    import tomllib\nexcept ImportError:\n    tomllib = None\n\n\ndef load():\n    return tomllib\n",
            id="import-fallback",
        ),
        pytest.param(
            "import argparse\nimport sys\n\n\ndef parse():\n    return argparse.ArgumentParser().parse_args(sys.argv[1:])\n",
            id="argument-parsing-only-inside-a-function",
        ),
    ],
)
def test_import_only_modules_are_skipped_in_any_position(tmp_path: Path, helper: str) -> None:
    skill = _write_skill(tmp_path, {"scripts/helper.py": helper, "scripts/run.py": GUARDED_ENTRY})

    for order in itertools.permutations(["helper.py", "run.py"]):
        assert generate_dataset._pick_primary_script(_parse_in_order(skill, order)) == "run.py"


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(GUARDED_ENTRY, id="main-guard"),
        pytest.param("import sys\n\nif sys.argv[1:] and __name__ == '__main__':\n    print('x')\n", id="guard-in-and"),
        pytest.param("def main():\n    pass\n\n\nif '__main__' == __name__:\n    main()\n", id="reversed-guard"),
        pytest.param("def main():\n    print('hi')\n\n\nmain()\n", id="bare-top-level-call"),
        pytest.param("import sys\n\n\ndef main():\n    return 0\n\n\nsys.exit(main())\n", id="exit-main"),
        pytest.param("def main():\n    return 0\n\n\nresult = main()\n", id="assigned-main-call"),
        pytest.param("import sys\n\nname = sys.argv[1]\n", id="reads-argv-at-top-level"),
        pytest.param(
            "import argparse\n\nparser = argparse.ArgumentParser()\nargs = parser.parse_args()\n",
            id="top-level-argument-parsing",
        ),
        pytest.param("import json\n\nfor line in open('x'):\n    print(json.loads(line))\n", id="top-level-loop"),
        pytest.param("print('hello')\n", id="print"),
        pytest.param("def broken(:\n", id="syntax-error-is-not-assumed-a-helper"),
    ],
)
def test_entry_points_are_kept_even_when_listed_after_a_helper(tmp_path: Path, entry: str) -> None:
    skill = _write_skill(tmp_path, {"scripts/helper.py": HELPER, "scripts/run.py": entry})

    assert generate_dataset._pick_primary_script(_parse_in_order(skill, ("helper.py", "run.py"))) == "run.py"


def test_scripts_that_are_all_entry_points_keep_listing_order(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path, {"scripts/a.py": GUARDED_ENTRY, "scripts/b.py": GUARDED_ENTRY})

    for order in (("a.py", "b.py"), ("b.py", "a.py")):
        assert generate_dataset._pick_primary_script(_parse_in_order(skill, order)) == order[0]


def test_single_import_only_script_is_still_the_expected_script(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path, {"scripts/helper.py": HELPER})

    assert generate_dataset._pick_primary_script(generate_dataset._parse_skill(skill)) == "helper.py"


def test_all_import_only_scripts_fall_back_to_the_first(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path, {"scripts/one.py": HELPER, "scripts/two.py": HELPER})

    for order in (("one.py", "two.py"), ("two.py", "one.py")):
        assert generate_dataset._pick_primary_script(_parse_in_order(skill, order)) == order[0]


def test_helper_in_scripts_and_entry_point_in_tools(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path, {"scripts/helper.py": HELPER, "tools/run.py": GUARDED_ENTRY})

    assert generate_dataset._pick_primary_script(_parse_in_order(skill, ("helper.py", "run.py"))) == "run.py"


def test_interactive_entry_point_still_loses_to_a_non_interactive_one(tmp_path: Path) -> None:
    skill = _write_skill(
        tmp_path,
        {"scripts/helper.py": HELPER, "scripts/login.py": GUARDED_ENTRY, "scripts/fetch.py": GUARDED_ENTRY},
        skill_md_body="login.py is interactive and opens a browser.\n",
    )

    parsed = _parse_in_order(skill, ("helper.py", "login.py", "fetch.py"))

    assert generate_dataset._pick_primary_script(parsed) == "fetch.py"


def test_interactive_entry_point_beats_a_non_interactive_helper(tmp_path: Path) -> None:
    """Only entry point is interactive: it still beats the helper, which the agent never runs."""
    skill = _write_skill(
        tmp_path,
        {"scripts/helper.py": HELPER, "scripts/login.py": GUARDED_ENTRY},
        skill_md_body="login.py is interactive and opens a browser.\n",
    )

    parsed = _parse_in_order(skill, ("helper.py", "login.py"))

    assert generate_dataset._pick_primary_script(parsed) == "login.py"


def test_unreadable_script_is_not_assumed_to_be_a_helper(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path, {"scripts/helper.py": HELPER})
    (skill / "scripts" / "binary.py").write_bytes(b"\xff\xfe\x00\x00 not utf-8 \x80")

    parsed = _parse_in_order(skill, ("helper.py", "binary.py"))

    assert generate_dataset._pick_primary_script(parsed) == "binary.py"
