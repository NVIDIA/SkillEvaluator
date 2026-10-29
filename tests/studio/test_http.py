# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import gzip
import io
import json
import threading
import time
import zipfile
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from skillevaluator.studio import app as http_app
from skillevaluator.studio import jobs
from skillevaluator.studio.jobs import PROVIDER_KEYS, Studio
from skillevaluator.studio.store import source_hash, write_json

LOCAL = "http://127.0.0.1:7865"
DOCUMENT = "---\nname: local-counter\ndescription: Count supplied words.\n---\n\nCount words accurately.\n"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for variable in (*PROVIDER_KEYS.values(), "SKILL_EVAL_LLM_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    # No capability probe should inspect a developer's running Docker service.
    monkeypatch.setattr(
        Studio,
        "capabilities",
        lambda _self: {
            "tier3": True,
            "docker": {"available": False, "reason": "Test has no Docker service."},
            "local": {"available": False, "reason": "Test has no agent sandbox."},
            "agents": [],
            "optimizer": False,
        },
    )
    application = http_app.create_app(tmp_path / "workspace")
    with TestClient(application, base_url=LOCAL, raise_server_exceptions=False) as connection:
        yield connection


def mutation_headers(client: TestClient) -> dict:
    return {"X-Studio-Token": client.app.state.studio.token, "Origin": LOCAL}


def wait_job(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200, response.text
        value = response.json()
        if value["status"] in jobs.TERMINAL:
            return value
        time.sleep(0.05)
    pytest.fail("The local native worker did not complete within 45 seconds.")


def test_state_is_empty_and_response_protects_local_artifacts(client: TestClient) -> None:
    response = client.get("/api/state")
    assert response.status_code == 200
    assert response.json()["skills"] == []
    assert response.json()["jobs"] == []
    assert response.json()["provider"]["configured"] is False
    assert response.json()["capabilities"]["optimizer"] is False
    assert response.headers["cache-control"] == "no-store"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"


def test_gzip_assets_decode_exactly_and_keep_boundary_headers(client: TestClient) -> None:
    for path in ("/", "/app.js", "/style.css", "/logo.svg", "/wordmark.svg"):
        identity = client.get(path, headers={"Accept-Encoding": "identity"})
        assert identity.status_code == 200
        assert "content-encoding" not in identity.headers
        with client.stream("GET", path, headers={"Accept-Encoding": "gzip"}) as compressed:
            wire = b"".join(compressed.iter_raw())
            assert compressed.status_code == 200
            assert compressed.headers["content-encoding"] == "gzip"
            assert "Accept-Encoding" in compressed.headers["vary"]
            if "content-length" in compressed.headers:
                assert int(compressed.headers["content-length"]) == len(wire)
            assert gzip.decompress(wire) == identity.content
            assert compressed.headers["cache-control"] == "no-store"
            assert compressed.headers["content-security-policy"] == identity.headers["content-security-policy"]


def test_rotated_credentials_remain_redacted_without_mutating_active_iteration(client: TestClient) -> None:
    service = client.app.state.studio
    service.configure({"name": "nv_build", "api_key": "synthetic-old-one"})
    service.configure({"name": "nv_build", "api_key": "synthetic-old-two"})
    replacing = threading.Event()
    release = threading.Event()
    configured = threading.Event()
    errors = []
    result = []

    class PausedLog(str):
        __slots__ = ()

        def replace(self, old, new, *args):
            replacing.set()
            assert release.wait(5)
            return super().replace(old, new, *args)

    def scrub_log():
        try:
            result.append(service.scrub(PausedLog("synthetic-old-one / synthetic-old-two")))
        except Exception as exc:
            errors.append(exc)

    def rotate():
        try:
            service.configure({"name": "nv_build", "api_key": "synthetic-new-three"})
        finally:
            configured.set()

    scrubber = threading.Thread(target=scrub_log)
    scrubber.start()
    assert replacing.wait(5)
    rotator = threading.Thread(target=rotate)
    rotator.start()
    # A lock-based implementation may correctly defer configuration until scrub completes.
    configured.wait(0.1)
    release.set()
    scrubber.join(5)
    rotator.join(5)
    assert not scrubber.is_alive() and not rotator.is_alive()
    assert not errors, errors
    assert result == ["[redacted] / [redacted]"]
    assert (
        service.scrub("synthetic-old-one synthetic-old-two synthetic-new-three") == "[redacted] [redacted] [redacted]"
    )


def test_rotating_a_partial_key_to_its_full_value_does_not_leave_the_secret_suffix(client: TestClient) -> None:
    service = client.app.state.studio
    full_values = []
    for index in range(20):
        partial = f"synthetic-prefix-{index}"
        full = partial + "-complete-credential"
        service.configure({"name": "nv_build", "api_key": partial})
        service.configure({"name": "nv_build", "api_key": full})
        full_values.append(full)
    assert all(service.scrub(value) == "[redacted]" for value in full_values)


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "attacker.example:7865"},
        {"Host": "127.0.0.1:7866"},
        {"Host": "user@127.0.0.1:7865"},
        {"Origin": "https://attacker.example"},
        {"Origin": "http://localhost:7865"},
        {"Sec-Fetch-Site": "cross-site"},
    ],
)
def test_host_port_and_origin_boundaries_reject_foreign_requests(client: TestClient, headers: dict) -> None:
    assert client.get("/api/state", headers=headers).status_code == 403


def test_mutations_require_the_current_server_token_and_json_object(client: TestClient) -> None:
    assert client.post("/api/skills", json={"content": DOCUMENT}).status_code == 403
    assert (
        client.post("/api/skills", json={"content": DOCUMENT}, headers={"X-Studio-Token": "wrong"}).status_code == 403
    )
    assert client.post("/api/skills", content="not-json", headers=mutation_headers(client)).status_code == 400
    assert client.post("/api/skills", json=[], headers=mutation_headers(client)).status_code == 400
    assert client.get("/api/state").json()["skills"] == []


def test_oversized_body_is_rejected_without_partial_skill(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http_app, "MAX_BODY", 30)
    response = client.post("/api/skills", json={"content": "x" * 100}, headers=mutation_headers(client))
    assert response.status_code == 400
    assert client.get("/api/state").json()["skills"] == []


@pytest.mark.parametrize(
    "route,payload",
    [
        ("/api/config", {"name": []}),
        ("/api/config", {"name": {}}),
        ("/api/jobs", {"action": [], "skill_id": "abc123abc123"}),
        ("/api/jobs", {"action": "validate", "skill_id": None}),
    ],
)
def test_wrong_field_types_are_request_errors_and_leave_workspace_usable(
    client: TestClient,
    route: str,
    payload: dict,
) -> None:
    response = client.post(route, json=payload, headers=mutation_headers(client))
    assert response.status_code == 400, response.text
    assert response.json()["error"]
    assert client.get("/api/state").status_code == 200


def test_corrupted_archive_returns_actionable_400_and_retains_no_partial_skill(client: TestClient) -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("SKILL.md", DOCUMENT)
    raw = output.getvalue()
    position = raw.index(DOCUMENT.encode())
    corrupted = raw[:position] + b"X" + raw[position + 1 :]
    response = client.post(
        "/api/skills",
        headers=mutation_headers(client),
        json={
            "filename": "damaged.zip",
            "archive_base64": base64.b64encode(corrupted).decode(),
        },
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]
    assert client.get("/api/state").json()["skills"] == []


def test_real_template_preview_can_be_reviewed_saved_and_validated(client: TestClient) -> None:
    skill = client.post("/api/sample", headers=mutation_headers(client)).json()
    workspace = client.app.state.studio.workspace
    source = workspace.directory(skill["id"])
    original_dataset = (source / "evals/evals.json").read_bytes()
    response = client.post(
        "/api/jobs",
        headers=mutation_headers(client),
        json={
            "skill_id": skill["id"],
            "action": "draft",
            "options": {"no_llm": True},
            "guidance": {
                "task": "Count the words red blue",
                "success": "Return exactly 2",
                "avoid": "Do not change the input",
            },
        },
    )
    assert response.status_code == 202, response.text
    draft = wait_job(client, response.json()["id"])
    assert draft["status"] == "succeeded", draft
    assert draft["result"]["generation_mode"] == "template"
    assert draft["result"]["review_required"] is True
    assert client.get(f"/api/skills/{skill['id']}").json()["dataset_hash"] == skill["dataset_hash"]
    cases = draft["result"]["cases"]
    assert 1 <= len(cases) <= 4
    assert all(case["id"] and case["prompt"] for case in cases)
    assert "Return exactly 2" in json.dumps(cases)
    request = json.loads((workspace.jobs / draft["id"] / "request.json").read_text())
    snapshot = Path(request["skill_path"])
    assert source_hash(snapshot) == draft["source_hash"] == skill["source_hash"]
    assert (snapshot / "SKILL.md").read_text() == skill["content"]
    assert (snapshot / "evals/evals.json").read_bytes() == original_dataset
    assert (source / "evals/evals.json").read_bytes() == original_dataset
    saved = client.put(f"/api/skills/{skill['id']}/dataset", headers=mutation_headers(client), json={"cases": cases})
    assert saved.status_code == 200, saved.text
    assert len(saved.json()["cases"]) == len(cases)
    for original, normalized in zip(cases, saved.json()["cases"], strict=True):
        assert all(normalized[key] == value for key, value in original.items())
        assert normalized["question"] == original["prompt"]
    validation = client.post(
        "/api/jobs", headers=mutation_headers(client), json={"skill_id": skill["id"], "action": "validate"}
    )
    assert validation.status_code == 202, validation.text
    validated = wait_job(client, validation.json()["id"])
    assert validated["status"] == "succeeded"
    assert validated["result"]["valid"] is True, validated["result"]
    cleared = client.put(f"/api/skills/{skill['id']}/dataset", headers=mutation_headers(client), json={"cases": []})
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["cases"] == []
    assert client.get(f"/api/skills/{skill['id']}").json()["cases"] == []
    assert (snapshot / "evals/evals.json").read_bytes() == original_dataset
    assert wait_job(client, draft["id"])["result"]["cases"] == cases


def test_quality_failure_is_not_misreported_as_process_failure_and_snapshot_stays_fixed(client: TestClient) -> None:
    skill = client.post("/api/skills", json={"content": DOCUMENT}, headers=mutation_headers(client)).json()
    response = client.post(
        "/api/jobs", headers=mutation_headers(client), json={"skill_id": skill["id"], "action": "validate"}
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    edited = client.put(
        f"/api/skills/{skill['id']}",
        headers=mutation_headers(client),
        json={"content": DOCUMENT + "\nEdited after launch."},
    )
    assert edited.status_code == 200
    result = wait_job(client, job_id)
    assert result["status"] == "succeeded"
    assert result["result"]["valid"] is False
    assert result["source_hash"] == skill["source_hash"] != edited.json()["source_hash"]
    request_file = client.app.state.studio.workspace.jobs / job_id / "request.json"
    snapshot = Path(json.loads(request_file.read_text())["skill_path"])
    assert source_hash(snapshot) == result["source_hash"]
    assert (snapshot / "SKILL.md").read_text() == DOCUMENT
    assert client.delete(f"/api/jobs/{job_id}", headers=mutation_headers(client)).json()["status"] == "succeeded"


def test_job_fingerprint_describes_the_snapshot_even_if_source_changes_during_launch(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = client.post("/api/skills", json={"content": DOCUMENT}, headers=mutation_headers(client)).json()
    real_copy = jobs.copy_source

    def copy_after_edit(source: Path, destination: Path) -> None:
        # Reproduce an edit between initial metadata lookup and source copying.
        (source / "SKILL.md").write_text(DOCUMENT + "\nChanged before the snapshot copy.\n")
        real_copy(source, destination)

    monkeypatch.setattr(jobs, "copy_source", copy_after_edit)
    response = client.post(
        "/api/jobs", headers=mutation_headers(client), json={"skill_id": skill["id"], "action": "validate"}
    )
    assert response.status_code == 202, response.text
    result = wait_job(client, response.json()["id"])
    request_file = client.app.state.studio.workspace.jobs / result["id"] / "request.json"
    snapshot = Path(json.loads(request_file.read_text())["skill_path"])
    assert result["source_hash"] == source_hash(snapshot)
    assert result["source_hash"] != skill["source_hash"]


def test_real_subprocess_control_blocks_overlapping_work_and_cancellation_is_terminal(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_popen = jobs.subprocess.Popen

    def start_control_process(command, **kwargs):
        # A real sleeping child exercises supervision without agents or model calls.
        program = "from pathlib import Path; import sys,time; Path(sys.argv[1], 'control-ready').write_text('ready'); time.sleep(30)"
        return real_popen([command[0], "-c", program, command[-1]], **kwargs)

    monkeypatch.setattr(jobs.subprocess, "Popen", start_control_process)
    skill = client.post("/api/skills", json={"content": DOCUMENT}, headers=mutation_headers(client)).json()
    payload = {"skill_id": skill["id"], "action": "check", "options": {"tier1": True}}
    response = client.post("/api/jobs", headers=mutation_headers(client), json=payload)
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    marker = client.app.state.studio.workspace.jobs / job_id / "control-ready"
    deadline = time.monotonic() + 10
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert marker.exists()
    assert client.post("/api/jobs", headers=mutation_headers(client), json=payload).status_code == 409
    assert client.delete(f"/api/jobs/{job_id}", headers=mutation_headers(client)).status_code == 200
    result = wait_job(client, job_id)
    assert result["status"] == "cancelled"
    assert result["result"] is None
    assert result["finished_at"]
    assert client.delete(f"/api/jobs/{job_id}", headers=mutation_headers(client)).json()["status"] == "cancelled"


def test_provider_secret_stays_in_process_environment_and_public_endpoint_isolation(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_key = "studio-selected-synthetic-credential"
    monkeypatch.setenv("OPENAI_API_KEY", "studio-unselected-openai-credential")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "studio-unselected-anthropic-credential")
    for variable in (
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "SKILL_EVAL_LLM_BASE_URL",
        "SKILL_EVAL_EMBEDDING_PROVIDER",
        "SKILL_EVAL_EMBEDDING_MODEL",
        "SKILL_EVAL_EMBEDDING_BASE_URL",
        "SKILL_EVAL_EMBEDDING_API_KEY",
        "SKILLSPECTOR_PROVIDER",
        "SKILLSPECTOR_MODEL",
        "SKILLSPECTOR_MODEL_SECURITY",
    ):
        monkeypatch.setenv(variable, "https://unselected.example/v1")
    monkeypatch.setenv("SKILL_EVAL_LLM_API_KEY", "studio-unselected-generic-credential")
    configured = client.post(
        "/api/config",
        headers=mutation_headers(client),
        json={
            "name": "nv_build",
            "model": "nvidia/test-model",
            "api_key": selected_key,
        },
    )
    assert configured.status_code == 200
    assert selected_key not in configured.text
    assert configured.json()["key_source"] == "session"
    captured = {}
    real_popen = jobs.subprocess.Popen

    def record_environment(*args, **kwargs):
        captured.update(kwargs["env"])
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(jobs.subprocess, "Popen", record_environment)
    skill = client.post("/api/skills", json={"content": DOCUMENT}, headers=mutation_headers(client)).json()
    response = client.post(
        "/api/jobs", headers=mutation_headers(client), json={"skill_id": skill["id"], "action": "validate"}
    )
    assert response.status_code == 202, response.text
    result = wait_job(client, response.json()["id"])
    assert result["status"] == "succeeded"
    assert captured["NVIDIA_API_KEY"] == selected_key
    for variable in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "SKILL_EVAL_LLM_BASE_URL",
        "SKILL_EVAL_LLM_API_KEY",
        "SKILL_EVAL_EMBEDDING_PROVIDER",
        "SKILL_EVAL_EMBEDDING_MODEL",
        "SKILL_EVAL_EMBEDDING_BASE_URL",
        "SKILL_EVAL_EMBEDDING_API_KEY",
        "SKILLSPECTOR_PROVIDER",
        "SKILLSPECTOR_MODEL",
        "SKILLSPECTOR_MODEL_SECURITY",
    ):
        assert variable not in captured
    assert captured["PYTHON_DOTENV_DISABLED"] == "1"
    workspace = client.app.state.studio.workspace
    for path in workspace.root.rglob("*"):
        if path.is_file():
            assert selected_key.encode() not in path.read_bytes(), path
    assert selected_key not in client.get("/api/state").text


def test_job_provider_model_and_key_are_one_frozen_selection(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = client.app.state.studio
    service.configure({"name": "nv_build", "model": "nvidia/original-model", "api_key": "synthetic-original-key"})
    skill = client.post("/api/skills", json={"content": DOCUMENT}, headers=mutation_headers(client)).json()
    original_copy = jobs.copy_source
    original_popen = jobs.subprocess.Popen
    captured = {}

    def copy_after_provider_change(source: Path, destination: Path) -> None:
        service.configure({"name": "openai", "model": "different-model", "api_key": "synthetic-later-key"})
        original_copy(source, destination)

    def capture_environment(*args, **kwargs):
        captured.update(kwargs["env"])
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(jobs, "copy_source", copy_after_provider_change)
    monkeypatch.setattr(jobs.subprocess, "Popen", capture_environment)
    response = client.post(
        "/api/jobs", headers=mutation_headers(client), json={"skill_id": skill["id"], "action": "validate"}
    )
    assert response.status_code == 202, response.text
    result = wait_job(client, response.json()["id"])
    assert result["status"] == "succeeded"
    assert result["provider"] == "nv_build"
    assert result["options"]["model"] == "nvidia/original-model"
    assert captured["SKILL_EVAL_LLM_PROVIDER"] == "nv_build"
    assert captured["SKILL_EVAL_LLM_MODEL"] == "nvidia/original-model"
    assert captured["NVIDIA_API_KEY"] == "synthetic-original-key"
    assert "OPENAI_API_KEY" not in captured
    assert service.provider()["name"] == "openai"


@pytest.mark.parametrize(
    "native_pack,grading,allowed",
    [
        (True, "custom_only", True),
        (True, "default", False),
        (False, "custom_only", False),
    ],
)
def test_no_dataset_exception_requires_the_native_pack_and_explicit_custom_only_mode(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    native_pack: bool,
    grading: str,
    allowed: bool,
) -> None:
    service = client.app.state.studio
    service.configure({"name": "nv_build", "api_key": "synthetic-never-used-key"})
    monkeypatch.setattr(
        Studio,
        "capabilities",
        lambda _self: {
            "tier3": True,
            "docker": {"available": True},
            "local": {"available": False},
            "agents": [{"id": "opencode", "available": True}],
            "optimizer": False,
        },
    )
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("SKILL.md", DOCUMENT)
        if native_pack:
            archive.writestr(
                "evals/config.yml",
                "schema_version: 1\nharbor:\n  task_source: native_harbor\ngrading:\n  mode: custom_only\n",
            )
            archive.writestr(
                "evals/harbor/example/instruction.md", "An authored native task, not executed by this test.\n"
            )
    skill_response = client.post(
        "/api/skills",
        headers=mutation_headers(client),
        json={
            "filename": "native.zip",
            "archive_base64": base64.b64encode(archive_bytes.getvalue()).decode(),
        },
    )
    assert skill_response.status_code == 201, skill_response.text
    skill = skill_response.json()
    assert skill["cases"] == []
    assert skill["native_custom_only"] is native_pack
    launches = []

    def prevent_actual_agent_launch(*args, **kwargs):
        launches.append((args, kwargs))
        raise OSError("Agent launch deliberately prevented in a request-boundary regression.")

    monkeypatch.setattr(jobs.subprocess, "Popen", prevent_actual_agent_launch)
    response = client.post(
        "/api/jobs",
        headers=mutation_headers(client),
        json={
            "skill_id": skill["id"],
            "action": "evaluate",
            "options": {"grading_mode": grading, "env_mode": "docker"},
        },
    )
    assert response.status_code == 400
    if allowed:
        assert len(launches) == 1
        assert "job could not start" in response.json()["error"]
        assert service.list_jobs()[0]["status"] == "failed"
    else:
        assert launches == []
        assert "test cases" in response.json()["error"]
        assert service.list_jobs() == []


def test_downloads_are_run_scoped_redacted_and_never_expose_source_or_request(
    client: TestClient, tmp_path: Path
) -> None:
    service = client.app.state.studio
    service.configure({"name": "nv_build", "api_key": "synthetic-artifact-secret"})
    job_id = "abc123abc123"
    folder = service.workspace.jobs / job_id
    folder.mkdir()
    write_json(folder / "metadata.json", {"id": job_id, "status": "succeeded", "started_at": "2026-10-24"})
    write_json(
        folder / "result.json",
        {"api_key": "synthetic-artifact-secret", "message": "value synthetic-artifact-secret", "accuracy": None},
    )
    (folder / "report.html").write_text("<script>unsafe()</script> synthetic-artifact-secret")
    (folder / "request.json").write_text('{"configuration":"private"}')
    (folder / "source").mkdir()
    (folder / "source/SKILL.md").write_text(DOCUMENT)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (folder / "linked.txt").symlink_to(outside)
    response = client.get(f"/api/jobs/{job_id}/download")
    assert response.status_code == 200
    assert "synthetic-artifact-secret" not in response.text
    assert json.loads(response.content)["accuracy"] is None
    html = client.get(f"/api/jobs/{job_id}/download", params={"file": "report.html"})
    assert html.headers["content-type"] == "application/octet-stream"
    assert html.headers["content-disposition"].startswith("attachment;")
    assert "synthetic-artifact-secret" not in html.text
    for relative in ("request.json", "metadata.json", "source/SKILL.md", "../outside.txt", "linked.txt", "."):
        blocked = client.get(f"/api/jobs/{job_id}/download", params={"file": relative})
        assert blocked.status_code == 400, (relative, blocked.status_code, blocked.text)


def test_restart_marks_only_unfinished_jobs_interrupted_and_drops_session_credentials(tmp_path: Path) -> None:
    service = Studio(tmp_path / "restart")
    service.configure({"name": "nv_build", "api_key": "never-persist-session-key"})
    initial_token = service.token
    for job_id, status in (("aaa111aaa111", "running"), ("bbb222bbb222", "stopping"), ("ccc333ccc333", "succeeded")):
        folder = service.workspace.jobs / job_id
        folder.mkdir()
        write_json(
            folder / "metadata.json", {"id": job_id, "status": status, "started_at": "2026-10-24", "finished_at": None}
        )
    restarted = Studio(service.workspace.root)
    assert restarted.token != initial_token
    assert restarted.keys == {}
    assert restarted.job("aaa111aaa111")["status"] == "interrupted"
    assert restarted.job("bbb222bbb222")["status"] == "interrupted"
    assert restarted.job("ccc333ccc333")["status"] == "succeeded"
    assert restarted.job("aaa111aaa111")["finished_at"]


def test_static_scan_needs_no_dataset_agent_or_provider_and_retains_incomplete_evidence(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        Studio,
        "capabilities",
        lambda _self: {"tier3": False, "local": {"available": False}, "docker": {"available": False}, "agents": []},
    )
    # Real native validators must report unavailable scanners, not a clean scan.
    for tool in ("SKILLSPECTOR", "BANDIT", "SEMGREP"):
        monkeypatch.setenv(f"SKILLEVALUATOR_{tool}_PATH", str(tmp_path / "missing-scanner"))
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("SKILL.md", DOCUMENT)
        archive.writestr("scripts/count.py", "def count_words(text):\n    return len(text.split())\n")
    skill = client.post(
        "/api/skills",
        headers=mutation_headers(client),
        json={"filename": "skill.zip", "archive_base64": base64.b64encode(archive_bytes.getvalue()).decode()},
    ).json()
    assert skill["cases"] == []
    response = client.post(
        "/api/jobs",
        headers=mutation_headers(client),
        json={"skill_id": skill["id"], "action": "check", "options": {"tier1": True}},
    )
    assert response.status_code == 202, response.text
    job = wait_job(client, response.json()["id"])
    assert job["status"] == "succeeded", job
    assert job["result"]["kind"] == "check"
    assert job["result"]["passed"] is False
    packet = job["result"]["optional_checks"]
    assert packet["status"] == "incomplete"
    assert packet["selection"] == {"tier1": True, "tier1_llm": False, "dedup": False}
    assert len(packet["results"]) >= 9
    assert all(item["status"] in {"passed", "failed", "incomplete"} for item in packet["results"])
    missing = {tool for item in packet["results"] for tool in item["incomplete_scans"]}
    assert {"skillspector", "bandit", "semgrep"} <= missing
    assert all(not item["passed"] for item in packet["results"] if item["status"] == "incomplete")
    artifact = client.get(f"/api/jobs/{job['id']}/download", params={"file": packet["artifact"]})
    assert artifact.status_code == 200, artifact.text
    assert artifact.json() == packet
    assert client.get(f"/api/skills/{skill['id']}").json()["source_hash"] == skill["source_hash"]
    assert client.get(f"/api/skills/{skill['id']}").json()["cases"] == []


@pytest.mark.parametrize(
    "options,provider",
    [
        ({"tier1": "yes"}, None),
        ({"tier1": True, "tier1_llm": 1}, None),
        ({"dedup": []}, None),
        ({"tier1_llm": True}, None),
        ({"tier1": True, "tier1_llm": True}, None),
        ({"dedup": True}, None),
        ({"tier1": True, "dedup": True}, None),
        ({"dedup": True}, "anthropic"),
    ],
)
def test_scan_option_and_provider_preflight_reject_before_launch(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, options: dict, provider: str | None
) -> None:
    def must_not_launch(*_args, **_kwargs):
        pytest.fail("Invalid or unconfigured selected checks must not launch a worker.")

    monkeypatch.setattr(jobs.subprocess, "Popen", must_not_launch)
    if provider:
        client.app.state.studio.configure({"name": provider, "api_key": "synthetic-unsupported-embedding-key"})
    skill = client.post("/api/skills", headers=mutation_headers(client), json={"content": DOCUMENT}).json()
    response = client.post(
        "/api/jobs",
        headers=mutation_headers(client),
        json={"skill_id": skill["id"], "action": "check", "options": options},
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]
    assert client.get("/api/state").json()["jobs"] == []


def test_selected_scan_options_are_frozen_before_model_worker_launch(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = client.app.state.studio
    service.configure({"name": "nv_build", "api_key": "synthetic-scan-key-never-used"})
    for name, value in {
        "SKILL_EVAL_EMBEDDING_PROVIDER": "openai-compatible",
        "SKILL_EVAL_EMBEDDING_MODEL": "unselected-model",
        "SKILL_EVAL_EMBEDDING_API_KEY": "unselected-synthetic-key",
        "SKILL_EVAL_EMBEDDING_BASE_URL": "https://unselected.example/v1",
    }.items():
        monkeypatch.setenv(name, value)
    skill = client.post("/api/skills", headers=mutation_headers(client), json={"content": DOCUMENT}).json()
    captured = {}

    def prevent_model_launch(command, **kwargs):
        folder = Path(command[-1])
        captured.update(request=json.loads((folder / "request.json").read_text()), environment=kwargs["env"])
        raise OSError("Model worker deliberately not started in this request-boundary test.")

    monkeypatch.setattr(jobs.subprocess, "Popen", prevent_model_launch)
    selection = {"tier1": True, "tier1_llm": True, "dedup": True}
    response = client.post(
        "/api/jobs",
        headers=mutation_headers(client),
        json={"skill_id": skill["id"], "action": "check", "options": selection},
    )
    assert response.status_code == 400, response.text
    record = service.list_jobs()[0]
    assert record["status"] == "failed"
    assert record["action"] == captured["request"]["action"] == "check"
    assert all(record["options"][name] is value for name, value in selection.items())
    assert all(captured["request"]["options"][name] is value for name, value in selection.items())
    assert source_hash(Path(captured["request"]["skill_path"])) == record["source_hash"] == skill["source_hash"]
    assert captured["environment"]["NVIDIA_API_KEY"] == "synthetic-scan-key-never-used"
    assert captured["environment"]["SKILL_EVAL_EMBEDDING_PROVIDER"] == "nv_build"
    assert not any(
        name in captured["environment"]
        for name in ("SKILL_EVAL_EMBEDDING_MODEL", "SKILL_EVAL_EMBEDDING_API_KEY", "SKILL_EVAL_EMBEDDING_BASE_URL")
    )
    assert "synthetic-scan-key" not in response.text
