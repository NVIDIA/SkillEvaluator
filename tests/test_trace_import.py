# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Production trace export import and human review regression tests."""

import json
from time import perf_counter

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.tier3.dataset_utils import load_dataset_entries_with_format
from skillevaluator.tier3.harbor.adapter import _entry_file_refs, _resolve_entry_file_ref
from skillevaluator.tier3.trace_import import (
    _pii_patterns,
    accept_traces,
    build_candidates,
    import_traces,
    parse_export,
    redact,
)


@pytest.fixture
def workspace(tmp_path):
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: sample\ndescription: Example\n---\nInstructions\n")
    return skill, tmp_path / "export.jsonl", tmp_path / "review"


def session(**kwargs):
    return {"session_id": "session-one", "prompt": "Analyze the input", **kwargs}


def write_export(path, sessions):
    path.write_text("\n".join(json.dumps(s) for s in sessions))


def reviewed(workspace, decision="accept", **kwargs):
    skill, export, review = workspace
    write_export(export, [session(**kwargs)])
    doc = import_traces(skill, export, review)
    doc["candidates"][0]["review"].update(decision=decision, expected_output="Independent expectation")
    save_review(review, doc)
    return doc


def save_review(review, doc):
    (review / "candidates.json").write_text(json.dumps(doc))


def test_parse_jsonl_and_wrapper():
    records = [session(), session(session_id="two")]
    assert [s for _, s in parse_export("\n".join(map(json.dumps, records)).encode())] == records
    assert [
        s
        for _, s in parse_export(
            json.dumps({"format": "skillevaluator.trace-export", "version": 1, "sessions": records}).encode()
        )
    ] == records


@pytest.mark.parametrize(
    ("fmt", "version"), [("other", 1), ("skillevaluator.trace-export", 2), ("skillevaluator.trace-export", True)]
)
def test_invalid_contract(fmt, version):
    with pytest.raises(ValueError, match="version 1"):
        parse_export(json.dumps({"format": fmt, "version": version, "sessions": []}).encode())


def test_classifications_and_warnings():
    records = [
        session(extra="ignored", files=[{"path": "../escape", "content": "x"}]),
        session(session_id="duplicate", prompt=" Analyze   the input "),
        {"prompt": "Missing identity"},
        session(session_id="missing", prompt=""),
        session(outcome="invalid"),
        session(skill="different"),
        4,
    ]
    raw = ("\n".join(map(json.dumps, records)) + "\ninvalid json").encode()
    doc, fixtures = build_candidates(raw, "export.jsonl", "sample")
    assert len(doc["candidates"]) == 1
    assert not fixtures
    assert [s["reason"] for s in doc["skipped"]] == [
        "duplicate",
        "incomplete",
        "incomplete",
        "unsupported",
        "other skill",
        "unsupported",
        "unsupported",
    ]
    assert doc["skipped"][0]["duplicate_of"] == doc["candidates"][0]["id"]
    assert doc["skipped"][-1]["record"] == 8
    assert "unsupported field: extra" in doc["candidates"][0]["warnings"]
    assert any("incomplete" in w for w in doc["candidates"][0]["warnings"])


def test_redaction_every_written_file(workspace):
    skill, export, review = workspace
    sensitive = "sk-abcdefghijklmno person@real-domain.com /home/privateperson/input"
    write_export(
        export,
        [
            session(
                prompt=sensitive,
                final_output=sensitive,
                source=sensitive,
                files=[{"path": "input.txt", "content": sensitive}],
                trajectory={
                    "steps": [{"tool_calls": [{"function_name": "shell", "arguments": {"command": sensitive}}]}]
                },
            )
        ],
    )
    doc = import_traces(skill, export, review)
    for path in review.rglob("*"):
        if path.is_file():
            content = path.read_text()
            for raw in ("sk-abcdefghijklmno", "person@real-domain.com", "privateperson"):
                assert raw not in content
    counts = doc["candidates"][0]["redactions"]
    assert counts["secrets"] >= 4
    assert counts["pii"] >= 8
    assert "/home/<user>/" in doc["candidates"][0]["prompt"]
    assert redact(doc)[0] == doc


@pytest.mark.parametrize(
    "path", ["/absolute", "../escape", "nested/../../escape", "C:\\Users\\someone\\file", "ok\x00bad"]
)
def test_bad_fixture_paths(path):
    doc, fixtures = build_candidates(
        json.dumps(session(files=[{"path": path, "content": "text"}])).encode(), "export", "sample"
    )
    assert not fixtures
    assert "incomplete" in doc["candidates"][0]["warnings"][0]


def test_reconstruct_log_and_unknown_agent():
    log = json.dumps(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Finished"},
                    {"type": "tool_use", "id": "call", "name": "Read", "input": {"file_path": "SKILL.md"}},
                ]
            },
        }
    )
    doc, _ = build_candidates(json.dumps(session(agent="claude-code", agent_log=log)).encode(), "export", "sample")
    c = doc["candidates"][0]
    assert c["observed"]["trajectory_source"] == "claude-code log"
    assert c["observed"]["final_output"] == "Finished"
    assert c["review"]["suggested_assertions"]
    assert not c["review"]["assertions"]
    doc, _ = build_candidates(json.dumps(session(agent="unrecognized", agent_log="raw")).encode(), "export", "sample")
    assert "unsupported agent log format" in doc["candidates"][0]["warnings"]
    assert doc["candidates"][0]["observed"]["tool_calls"] == []


def test_prompt_recovered_from_trajectory():
    doc, _ = build_candidates(
        json.dumps(
            {
                "session_id": "recovered",
                "trajectory": {
                    "steps": [
                        {"source": "user", "message": "Original request"},
                        {"source": "agent", "message": "Response"},
                    ]
                },
            }
        ).encode(),
        "export",
        "sample",
    )
    assert doc["candidates"][0]["prompt"] == "Original request"


def test_output_refusal_force_and_dry_run(workspace):
    skill, export, review = workspace
    write_export(export, [session()])
    with pytest.raises(ValueError, match="outside"):
        import_traces(skill, export, skill / "review")
    import_traces(skill, export, review, dry_run=True)
    assert not review.exists()
    import_traces(skill, export, review)
    with pytest.raises(ValueError, match="already exists"):
        import_traces(skill, export, review)
    import_traces(skill, export, review, force=True)


@pytest.mark.parametrize("decision", ["pending", "reject"])
def test_pending_and_reject(workspace, decision):
    reviewed(workspace, decision)
    result = accept_traces(workspace[0], workspace[2])
    assert len(result["skipped"]) == 1
    assert not (workspace[0] / "evals").exists()


@pytest.mark.parametrize("decision", ["accept", "not-a-decision"])
def test_invalid_review(workspace, decision):
    doc = reviewed(workspace, decision)
    doc["candidates"][0]["review"]["expected_output"] = ""
    save_review(workspace[2], doc)
    with pytest.raises(ValueError, match=doc["candidates"][0]["id"]):
        accept_traces(workspace[0], workspace[2])
    assert not (workspace[0] / "evals").exists()


@pytest.mark.parametrize("outcome", ["failure", "unknown"])
def test_refuse_observed_failure(workspace, outcome):
    reviewed(workspace, "accept_observed", outcome=outcome, final_output="Observed response")
    with pytest.raises(ValueError, match="independent expectation"):
        accept_traces(workspace[0], workspace[2])


def test_allow_observed_success(workspace):
    reviewed(workspace, "accept_observed", outcome="success", final_output="Observed response")
    result = accept_traces(workspace[0], workspace[2])
    assert result["added"]
    entry = json.loads((workspace[0] / "evals/evals.json").read_text())["evals"][0]
    assert entry["expected_output"] == "Observed response"
    assert entry["files"] == []


@pytest.mark.parametrize("field", ["expected_output", "assertions"])
def test_reviewer_pasted_secret(workspace, field):
    doc = reviewed(workspace)
    value = "sk-abcdefghijklmno"
    doc["candidates"][0]["review"][field] = [value] if field == "assertions" else value
    save_review(workspace[2], doc)
    with pytest.raises(ValueError, match="redaction required"):
        accept_traces(workspace[0], workspace[2])
    assert not (workspace[0] / "evals").exists()


@pytest.mark.parametrize("surface", ["prompt", "fixture", "expected_output"])
def test_reviewer_pasted_pii(workspace, surface):
    skill, _, review = workspace
    doc = reviewed(workspace, files=[{"path": "input.txt", "content": "Safe fixture"}])
    candidate = doc["candidates"][0]
    pasted = "Contact person@real-domain.com at /home/privateperson/input"
    if surface == "prompt":
        candidate["prompt"] = pasted
    elif surface == "fixture":
        (review / candidate["files"][0]).write_text(pasted)
    else:
        candidate["review"]["expected_output"] = pasted
    save_review(review, doc)
    with pytest.raises(ValueError, match=candidate["id"]):
        accept_traces(skill, review)
    assert not (skill / "evals").exists()


def test_errors_abort_all_candidates_before_copying(workspace):
    skill, export, review = workspace
    write_export(
        export,
        [
            session(files=[{"path": "input.txt", "content": "Safe fixture"}]),
            session(session_id="second", prompt="Another request"),
        ],
    )
    doc = import_traces(skill, export, review)
    first, second = doc["candidates"]
    first["review"].update(decision="accept", expected_output="Independent expectation")
    second["review"]["decision"] = "accept"
    save_review(review, doc)
    with pytest.raises(ValueError, match=second["id"]):
        accept_traces(skill, review)
    assert not (skill / "evals").exists()


def test_invalid_fixture_content_dropped():
    doc, fixtures = build_candidates(
        json.dumps(session(files=[{"path": "input.txt", "content": 123}])).encode(), "export", "sample"
    )
    assert not fixtures
    assert "incomplete" in doc["candidates"][0]["warnings"][0]


def test_append_fixtures_validation_adapter_and_idempotency(workspace):
    skill, _, review = workspace
    reviewed(workspace, files=[{"path": "nested/input.txt", "content": "Replay fixture"}])
    (skill / "evals").mkdir()
    target = skill / "evals/evals.json"
    original = {"id": "original", "prompt": "Original", "expected_output": "Expected"}
    target.write_text(json.dumps({"skill_name": "sample", "evals": [original]}))
    assert accept_traces(skill, review, dry_run=True)["added"]
    assert not (skill / "evals/files").exists()
    result = accept_traces(skill, review)
    assert result["added"]
    assert all(c.status != "error" for c in result["validation"])
    entries, fmt = load_dataset_entries_with_format(target)
    assert fmt == "agentskills"
    assert json.loads(target.read_text())["evals"][0] == original
    entry = entries[1]
    assert entry["trace_source"]["session_id"] == "session-one"
    refs = _entry_file_refs(entry)
    assert refs == [f"files/trace/{entry['id']}/nested/input.txt"]
    source, _ = _resolve_entry_file_ref(
        refs[0], skill_path=skill, evals_dir=skill / "evals", input_files_dir=skill / "evals/files"
    )
    assert source.read_text() == "Replay fixture"
    assert accept_traces(skill, review)["already present"] == [entry["id"]]
    assert len(json.loads(target.read_text())["evals"]) == 2
    entries[1]["trace_source"]["session_id"] = "different"
    target.write_text(json.dumps({"skill_name": "sample", "evals": entries}))
    with pytest.raises(ValueError, match="collision"):
        accept_traces(skill, review)


@pytest.mark.parametrize("extension", ["json", "yaml"])
def test_legacy_dataset_refused(workspace, extension):
    skill, _, review = workspace
    reviewed(workspace)
    (skill / "evals").mkdir()
    (skill / "evals" / f"evals.{extension}").write_text("[]")
    with pytest.raises(ValueError, match=r"agentskills|extension"):
        accept_traces(skill, review)


def test_cli_smoke(workspace):
    skill, export, review = workspace
    write_export(export, [session()])
    runner = CliRunner()
    result = runner.invoke(cli, ["tier3", "import-traces", str(skill), str(export), "-o", str(review)])
    assert result.exit_code == 0, result.output
    assert "candidate: 1" in result.output
    assert "accept-traces" in result.output
    result = runner.invoke(cli, ["tier3", "accept-traces", str(skill), str(review), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "would add: 0" in result.output


def test_empty_export_and_bad_review_contract(workspace):
    skill, export, review = workspace
    export.write_text("")
    assert import_traces(skill, export, review)["candidates"] == []
    save_review(review, {"format": "wrong", "version": 1})
    with pytest.raises(ValueError, match="trace-candidates"):
        accept_traces(skill, review)


def test_pii_categories_and_stability():
    original = (
        "email: real@domain.io /Users/someone/file C:\\Users\\someone\\file /home/root/file 212-345-6789 123-45-6789"
    )
    clean, counts = redact(original)
    assert counts["pii"] == 6
    assert "someone" not in clean and "/home/root/" not in clean
    assert redact(clean)[0] == clean
    assert redact("user@example.com /Users/{id}/")[0] == "user@example.com /Users/{id}/"


def test_structured_tool_and_provenance_secrets():
    raw = json.dumps(
        session(
            source={"password": "opaque-value"},
            trajectory={
                "steps": [
                    {"tool_calls": [{"function_name": "request", "arguments": {"password": "another-opaque-value"}}]}
                ]
            },
        )
    ).encode()
    doc, _ = build_candidates(raw, "export", "sample")
    assert "opaque-value" not in json.dumps(doc)
    assert doc["candidates"][0]["redactions"]["secrets"] >= 2


def test_large_text_redaction_is_fast():
    chunk = "Observed tool output with ordinary words and whitespace. "
    text = (
        chunk * (2 * 1024 * 1024 // len(chunk)) + " sk-abcdefghijklmno person@real-domain.com /home/privateperson/input"
    )
    start = perf_counter()
    clean, counts = redact(text)
    elapsed = perf_counter() - start
    assert elapsed < 5, f"Redacting 2 MiB took {elapsed:.2f}s"
    assert "sk-abcdefghijklmno" not in clean
    assert "person@real-domain.com" not in clean
    assert "privateperson" not in clean
    assert counts == {"secrets": 1, "pii": 2}


def test_pii_patterns_are_cached():
    assert _pii_patterns() is _pii_patterns()
    assert all(hasattr(pattern, "sub") for _, pattern, _ in _pii_patterns())


def test_marker_counts_ignore_existing_markers():
    assert redact("<redacted> sk-abcdefghijklmno")[1]["secrets"] == 1
    assert redact({"password": "opaque", "message": "<redacted>"}, structured=True)[1]["secrets"] == 1


def test_missing_replay_inputs_redacted_deduplicated_and_capped(workspace):
    skill, _, _ = workspace
    paths = [
        "input.txt",
        "/workspace/input/input.txt",
        "SKILL.md",
        "scripts/run.py",
        "references/guide.md",
        str(skill / "SKILL.md"),
        "missing.csv",
        "missing.csv",
        "/home/privateperson/report.csv",
    ]
    paths += [f"missing-{i}.csv" for i in range(30)]
    calls = [{"function_name": "Read", "arguments": {"file_path": path}} for path in paths]
    doc, _ = build_candidates(
        json.dumps(
            session(files=[{"path": "input.txt", "content": "fixture"}], trajectory={"steps": [{"tool_calls": calls}]})
        ).encode(),
        "export",
        "sample",
        skill_path=skill,
    )
    warnings = doc["candidates"][0]["warnings"]
    assert len(warnings) == 20
    assert warnings.count("missing replay input: missing.csv") == 1
    assert "missing replay input: /home/<user>/report.csv" in warnings
    assert "privateperson" not in json.dumps(doc)
    assert not any("SKILL.md" in w or "scripts/" in w or "references/" in w for w in warnings)


@pytest.mark.parametrize("tool", ["read", "read_file", "open", "open_file"])
def test_read_and_open_tools_report_missing_inputs(tool):
    doc, _ = build_candidates(
        json.dumps(
            session(
                trajectory={"steps": [{"tool_calls": [{"function_name": tool, "arguments": {"path": "missing.txt"}}]}]}
            )
        ).encode(),
        "export",
        "sample",
    )
    assert "missing replay input: missing.txt" in doc["candidates"][0]["warnings"]
