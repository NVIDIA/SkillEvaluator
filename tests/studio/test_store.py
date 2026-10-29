# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
import json
import stat
import warnings
import zipfile
from pathlib import Path

import pytest

from skillevaluator.studio import store
from skillevaluator.studio.store import Workspace, copy_source, safe_file, source_hash

DOCUMENT = "---\nname: local-counter\ndescription: Count supplied words.\n---\n\nCount words accurately.\n"


def archive_payload(entries: list[tuple[str | zipfile.ZipInfo, str | bytes]]) -> dict:
    buffer = io.BytesIO()
    with warnings.catch_warnings(), zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        warnings.simplefilter("ignore", UserWarning)
        for name, content in entries:
            archive.writestr(name, content)
    return {"filename": "skill.zip", "archive_base64": base64.b64encode(buffer.getvalue()).decode()}


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path / "studio")


def test_import_export_and_snapshot_share_the_same_nonsecret_files(workspace: Workspace, tmp_path: Path) -> None:
    skill = workspace.create(
        archive_payload(
            [
                ("package/SKILL.md", DOCUMENT),
                ("package/scripts/count.py", "print(3)\n"),
                ("package/evals/files/input.txt", "red blue green\n"),
                ("package/.env", "SECRET=must-not-import"),
                ("package/auth.json", '{"api_key":"must-not-import"}'),
                ("package/node_modules/unused.js", "unused"),
                ("README.md", "Unrelated repository file"),
            ]
        )
    )
    assert skill["files"] == ["SKILL.md", "evals/files/input.txt", "scripts/count.py"]
    assert set(skill["ignored_files"]) == {
        "package/.env",
        "package/auth.json",
        "package/node_modules/unused.js",
        "README.md",
    }
    folder = workspace.directory(skill["id"])
    (folder / ".env.local").write_text("LOCAL=also-excluded")
    (folder / "credentials.json").write_text('{"token":"also-excluded"}')
    (folder / "scripts").chmod(0o750)
    (folder / "empty").mkdir(mode=0o710)
    before = source_hash(folder)
    with zipfile.ZipFile(io.BytesIO(workspace.export(skill["id"]))) as exported:
        assert sorted(exported.namelist()) == skill["files"]
        assert exported.read("scripts/count.py") == b"print(3)\n"
    destination = tmp_path / "snapshot"
    copy_source(folder, destination)
    assert source_hash(destination) == before
    assert not (destination / ".env.local").exists()
    assert not (destination / "credentials.json").exists()
    for relative in (".", "scripts", "empty"):
        assert stat.S_IMODE((destination / relative).stat().st_mode) == stat.S_IMODE((folder / relative).stat().st_mode)


@pytest.mark.parametrize(
    "bad_path", ["../outside.txt", "/outside.txt", "package/../../outside.txt", r"package\outside.txt"]
)
def test_import_rejects_traversal_even_outside_the_skill_subtree(workspace: Workspace, bad_path: str) -> None:
    with pytest.raises(ValueError, match="workspace"):
        workspace.create(archive_payload([("package/SKILL.md", DOCUMENT), (bad_path, "bad")]))
    assert list(workspace.skills.iterdir()) == []


@pytest.mark.parametrize("mode", [stat.S_IFLNK | 0o777, stat.S_IFIFO | 0o600])
def test_import_rejects_archive_links_and_special_files(workspace: Workspace, mode: int) -> None:
    entry = zipfile.ZipInfo("package/scripts/tool")
    entry.create_system = 3
    entry.external_attr = mode << 16
    with pytest.raises(ValueError, match="links and special"):
        workspace.create(archive_payload([("package/SKILL.md", DOCUMENT), (entry, "../../private")]))
    assert list(workspace.skills.iterdir()) == []


@pytest.mark.parametrize("second", ["script.py", "SCRIPT.py"])
def test_import_rejects_duplicate_or_case_conflicting_files(workspace: Workspace, second: str) -> None:
    with pytest.raises(ValueError, match=r"duplicate|case-conflicting"):
        workspace.create(
            archive_payload(
                [
                    ("package/SKILL.md", DOCUMENT),
                    ("package/script.py", "first"),
                    (f"package/{second}", "second"),
                ]
            )
        )
    assert list(workspace.skills.iterdir()) == []


def test_import_rejects_multiple_skills_and_cleans_partial_workspace(workspace: Workspace) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        workspace.create(archive_payload([("one/SKILL.md", DOCUMENT), ("two/SKILL.md", DOCUMENT)]))
    assert list(workspace.skills.iterdir()) == []


@pytest.mark.parametrize("limit,value", [("MAX_FILES", 1), ("MAX_EXPANDED", 10), ("MAX_UPLOAD", 10)])
def test_upload_limits_reject_before_retaining_partial_skill(
    workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
    limit: str,
    value: int,
) -> None:
    monkeypatch.setattr(store, limit, value)
    with pytest.raises(ValueError, match=r"Upload|upload"):
        workspace.create(archive_payload([("SKILL.md", DOCUMENT), ("input.txt", "payload")]))
    assert list(workspace.skills.iterdir()) == []


def test_native_prompt_cases_roundtrip_and_change_dataset_fingerprint(workspace: Workspace) -> None:
    skill = workspace.create({"content": DOCUMENT})
    cases = [
        {
            "id": "count-zero",
            "prompt": "Count the words in an empty file.",
            "expected_skill": "local-counter",
            "expected_script": None,
            "ground_truth": '{"words":0}',
            "expected_behavior": ["Returns integer zero"],
            "files": [],
            "custom_assertions": {"strict_json": True},
        }
    ]
    saved = workspace.save_cases(skill["id"], cases)
    assert len(saved["cases"]) == 1
    assert all(saved["cases"][0][key] == value for key, value in cases[0].items())
    assert saved["cases"][0]["question"] == cases[0]["prompt"]
    assert saved["dataset_hash"] and saved["source_hash"] != skill["source_hash"]
    assert workspace.describe(skill["id"])["cases"][0]["custom_assertions"] == {"strict_json": True}
    with zipfile.ZipFile(io.BytesIO(workspace.export(skill["id"]))) as archive:
        from skillevaluator.tier3.dataset_utils import normalize_dataset_entries

        contents = json.loads(archive.read("evals/evals.json"))
        assert normalize_dataset_entries(contents) == saved["cases"]


def test_candidate_keeps_original_dataset_assets_and_independent_source(workspace: Workspace) -> None:
    original = workspace.sample()
    original_hash = original["source_hash"]
    candidate = workspace.candidate(original["id"], original["content"] + "\nVerify the final JSON shape.\n")
    assert candidate["parent_id"] == original["id"]
    assert candidate["source_hash"] != original_hash
    assert candidate["dataset_hash"] == original["dataset_hash"]
    assert candidate["cases"] == original["cases"]
    assert candidate["files"] == original["files"]
    assert workspace.describe(original["id"])["source_hash"] == original_hash
    workspace.update(candidate["id"], candidate["content"] + "\nSecond edit.\n")
    assert workspace.describe(original["id"])["content"] == original["content"]


def test_symlink_added_after_import_cannot_be_exported_or_snapshotted(workspace: Workspace, tmp_path: Path) -> None:
    skill = workspace.create({"content": DOCUMENT})
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    folder = workspace.directory(skill["id"])
    (folder / "linked.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic"):
        workspace.export(skill["id"])
    with pytest.raises(ValueError, match="symbolic"):
        copy_source(folder, tmp_path / "copy")
    with pytest.raises(ValueError, match="Symbolic"):
        safe_file(folder, "linked.txt")


def test_invalid_dataset_save_does_not_replace_previously_saved_cases(workspace: Workspace) -> None:
    skill = workspace.sample()
    folder = workspace.directory(skill["id"])
    prior = (folder / "evals/evals.json").read_bytes()
    duplicate = [skill["cases"][0], skill["cases"][0]]
    with pytest.raises(ValueError, match="Duplicate case"):
        workspace.save_cases(skill["id"], duplicate)
    assert (folder / "evals/evals.json").read_bytes() == prior


@pytest.mark.parametrize(
    "filename,contents",
    [
        ("evals/evals.jsonl", '{"id":"case-one","question":"Count red blue","ground_truth":"2"}\n'),
        ("evals/evals.yaml", "- id: case-one\n  question: Count red blue\n  ground_truth: '2'\n"),
    ],
)
def test_imported_native_formats_preserve_cases_and_can_be_saved_then_cleared(
    workspace: Workspace,
    filename: str,
    contents: str,
) -> None:
    imported = workspace.create(archive_payload([("SKILL.md", DOCUMENT), (filename, contents)]))
    assert imported["dataset_error"] is None
    assert imported["cases"][0]["question"] == "Count red blue"
    saved = workspace.save_cases(imported["id"], imported["cases"])
    assert saved["cases"][0]["ground_truth"] == "2"
    assert saved["dataset_hash"] != imported["dataset_hash"]
    cleared = workspace.save_cases(imported["id"], [])
    assert cleared["cases"] == [] and cleared["dataset_error"] is None
    assert cleared["dataset_hash"] != saved["dataset_hash"]
    assert workspace.describe(imported["id"])["cases"] == []
    roundtrip = workspace.create(
        {"filename": "cleared.zip", "archive_base64": base64.b64encode(workspace.export(imported["id"])).decode()}
    )
    assert roundtrip["cases"] == [] and roundtrip["dataset_error"] is None


def test_malformed_yaml_is_a_dataset_issue_and_does_not_make_skill_unreadable(workspace: Workspace) -> None:
    imported = workspace.create(
        archive_payload(
            [
                ("SKILL.md", DOCUMENT),
                ("evals/evals.yaml", "cases: [broken: ["),
            ]
        )
    )
    assert imported["cases"] == []
    assert imported["dataset_error"]
    assert imported["content"] == DOCUMENT
    assert workspace.list()[0]["id"] == imported["id"]
