# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001, T201 -- standalone image verification CLI.
"""Verify a local Tier 2 image using a loopback provider in an offline Docker namespace."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if __package__:
    from ci.smoke_image import SmokeError, _commit, _git, _load, _require
else:
    from smoke_image import SmokeError, _commit, _git, _load, _require

PROVIDER_PORT = 8080
# Use the shape of an opaque key. Report redaction also matches key fragments.
FIXTURE_KEY = "sk-06276fca39b945cd9800dc7b19323fde"
JOB_HARDENING = (
    "--pull=never",
    "--read-only",
    "--cap-drop=ALL",
    "--security-opt=no-new-privileges",
)


class FixtureProvider(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible embeddings and chat responses for synthetic skills."""

    def do_POST(self):
        if self.headers.get("Authorization") != f"Bearer {FIXTURE_KEY}":
            self.send_error(401)
            return
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/embeddings":
            inputs = request["input"]
            embeddings = [
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": [0.0, 1.0] if "unrelated workflow" in content else [1.0, 0.0],
                }
                for index, content in enumerate(inputs)
            ]
            response = {"object": "list", "model": request["model"], "data": embeddings}
            print(f"embeddings {len(embeddings)}", flush=True)
        elif self.path == "/v1/chat/completions":
            print("chat", flush=True)
            decision = {
                "verdict": "DUPLICATE",
                "confidence": 0.99,
                "reasoning": "The fixture intentionally repeats the same instructions.",
                "suggestion": "Consolidate them into a single section.",
            }
            response = {
                "id": "fixture-completion",
                "object": "chat.completion",
                "created": 0,
                "model": request["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(decision)},
                    }
                ],
            }
        else:
            self.send_error(404)
            return
        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check, timeout=60)


def _prepare_workspace(workspace: Path) -> tuple[str, str]:
    workspace.chmod(0o755)  # The default image user differs from the Docker host user.
    _git(workspace, "init", "--quiet", "--template=")
    paragraph = "Read the supplied passage and identify the statement that directly supports each answer. " * 4
    alpha = workspace / "skills/alpha/SKILL.md"
    beta = workspace / "skills/beta/SKILL.md"
    for path in (alpha, beta):
        path.parent.mkdir(parents=True)
    alpha.write_text(
        "---\nname: alpha\ndescription: Use this skill to extract evidence from a passage.\n---\n"
        f"# Alpha\n\n## Section A\n{paragraph}\n\n## Section B\n{paragraph}\n",
        encoding="utf-8",
    )
    beta.write_text(
        "---\nname: beta\ndescription: Use this skill for an unrelated workflow.\n---\n"
        "# Beta\n\n## Workflow\nSort the supplied tasks alphabetically.\n",
        encoding="utf-8",
    )
    base = _commit(workspace, "Add two synthetic skills")
    alpha.write_text(alpha.read_text(encoding="utf-8") + "\nUse the source wording.\n", encoding="utf-8")
    return base, _commit(workspace, "Change alpha only")


def _start_provider(image: str, container: str) -> None:
    fixtures = Path(__file__).resolve().parent
    _require("," not in str(fixtures), "The fixture directory cannot contain a comma in a Docker mount path")
    _docker(
        "run",
        "--detach",
        "--name",
        container,
        "--network=none",
        *JOB_HARDENING,
        "--mount",
        f"type=bind,src={fixtures},dst=/fixture,readonly",
        "--entrypoint",
        "python",
        image,
        "-Bu",
        "/fixture/smoke_tier2_image.py",
        "--serve-fixture",
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if "fixture-ready" in _docker("logs", container).stdout.splitlines():
            return
        time.sleep(0.25)
    raise SmokeError("The offline provider fixture did not start.")


def _evaluate(
    image: str, workspace: Path, output: Path, provider: str, base: str, head: str, revision: str, *, with_key: bool
) -> int:
    output.mkdir()
    reports = output / "reports"
    reports.mkdir()
    reports.chmod(0o777)  # Only this fresh artifact mount is writable by the image user.
    name = "skillevaluator-tier2-worker-" + uuid.uuid4().hex
    environment = {
        "DIFF_BASE_SHA": base,
        "DIFF_HEAD_SHA": head,
        "NO_PROXY": "127.0.0.1",
        "REPORT_COMMIT": head,
        "SKILLEVALUATOR_TOOLS_REF": revision,
        "SKILL_EVAL_PATHS": "skills",
        "SKILL_EVAL_PROVIDER": "openai-compatible",
        "SKILL_EVAL_MODEL": "fixture-chat",
        "SKILL_EVAL_BASE_URL": f"http://127.0.0.1:{PROVIDER_PORT}/v1",
        "SKILL_EVAL_EMBEDDING_MODEL": "fixture-embedding",
        "SKILL_EVAL_LLM_API_KEY": FIXTURE_KEY if with_key else "",
        "SKILL_EVAL_TIMEOUT_SECONDS": "180",
        "no_proxy": "127.0.0.1",
    }
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        f"--network=container:{provider}",
        *JOB_HARDENING,
        "--tmpfs",
        "/tmp:rw,nosuid,noexec,mode=1777,size=128m",
        "--mount",
        f"type=bind,src={workspace},dst=/workspace,readonly",
        "--mount",
        f"type=bind,src={reports},dst=/reports",
    ]
    for key, value in environment.items():
        command.extend(("--env", f"{key}={value}"))
    command.extend((image, "--workspace", "/workspace", "--reports", "/reports"))
    try:
        with (output / "container.log").open("wb") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=210)
    finally:
        _docker("rm", "--force", name, check=False)
    return result.returncode


def _verify_reports(output: Path, base: str, head: str, revision: str, exit_code: int) -> dict:
    _require(exit_code == 0, "Advisory Tier 2 returned a blocking exit code")
    reports = output / "reports"
    manifest = _load(reports / "tier2/ci-tier.json")
    (tier,) = manifest["tiers"]
    (row,) = tier["skills"]
    _require(manifest["selection"]["selected"] == ["skills/alpha"], "The scan did not select only the changed skill")
    _require(manifest["commit"] == head and manifest["base_commit"] == base, "Report commits differ from the fixture")
    _require(manifest["runtime"]["tools_ref"] == revision, "Report is missing the image revision")
    _require(
        tier["tier"] == 2 and tier["status"] == "failed" and not tier["required"] and manifest["overall_passed"],
        "Expected a completed advisory duplicate finding",
    )
    _require(row["status"] == "failed", "The changed skill's finding was not recorded")
    canonical = reports / row["report_json"]
    report = _load(canonical)
    results = {result["validator"]: result for result in report["results"]}
    _require(results["Similarity Check"]["status"] == "passed", "The changed skill did not complete its catalog query")
    _require(
        all(result["gating"]["blocking"] is False for result in results.values()), "Published evidence must be advisory"
    )
    _require(
        any(finding["check_name"] == "duplicate" for finding in results["Context Deduplication"]["findings"]),
        "The chat-confirmed duplicate is missing from the evidence",
    )
    _require((reports / row["report_html"]).is_file(), "Missing retained HTML report")
    _require(canonical.with_suffix(".md").is_file(), "Missing retained Markdown report")
    _require(FIXTURE_KEY not in canonical.read_text(encoding="utf-8"), "Credential leaked into the canonical report")
    return {"case": "changed-skill", "verified": True, "exit_code": exit_code, "status": tier["status"]}


def smoke(image: str, output: Path) -> bool:
    _require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]{0,511}", image)), "Invalid local image reference")
    _require(not output.is_symlink(), "Evidence output must not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve(strict=True)
    _require(not any(output.iterdir()) and "," not in str(output), "Use an empty output directory without commas")
    receipt = {"image": image, "all_verified": False, "cases": []}
    provider = "skillevaluator-tier2-provider-" + uuid.uuid4().hex
    try:
        (details,) = json.loads(_docker("image", "inspect", image).stdout)
        config = details["Config"]
        revision = (config.get("Labels") or {}).get("org.opencontainers.image.revision", "")
        _require(bool(re.fullmatch(r"[0-9a-f]{40}", revision)), "Image lacks a full source revision label")
        _require(config["User"] == "evaluator", "Image must default to its nonroot evaluator user")
        marker = _docker(
            "run",
            "--rm",
            "--network=none",
            *JOB_HARDENING,
            "--entrypoint",
            "/bin/sh",
            image,
            "-c",
            'test -x /bin/bash && test "$(id -u)" -eq 10001 && cat /opt/skillevaluator/revision',
        ).stdout.strip()
        _require(marker == revision, "Installed image revision differs from its label")
        receipt["runtime"] = {
            "id": details["Id"],
            "digests": details["RepoDigests"],
            "os": details["Os"],
            "architecture": details["Architecture"],
            "source_revision": revision,
        }
        with tempfile.TemporaryDirectory(prefix="skillevaluator-tier2-smoke-") as temporary:
            workspace = Path(temporary).resolve()
            _require("," not in str(workspace), "The temporary directory cannot contain a comma")
            base, head = _prepare_workspace(workspace)
            _start_provider(image, provider)
            changed = output / "changed-skill"
            code = _evaluate(image, workspace, changed, provider, base, head, revision, with_key=True)
            requests = _docker("logs", provider).stdout.splitlines()
            _require(
                requests[1:2] == ["embeddings 2"] and requests.count("chat") == 1,
                "Expected a full two-skill catalog and a chat verdict for the selected skill",
            )
            receipt["cases"].append(_verify_reports(changed, base, head, revision, code))
            missing = output / "missing-provider-key"
            code = _evaluate(image, workspace, missing, provider, base, head, revision, with_key=False)
            manifest = _load(missing / "reports/tier2/ci-tier.json")
            (tier,) = manifest["tiers"]
            _require(
                code == 0 and tier["status"] == "incomplete" and "credential is missing" in tier["reason"],
                "Missing credentials must produce explicit advisory incomplete evidence",
            )
            _require(
                _docker("logs", provider).stdout.splitlines() == requests, "A missing credential triggered inference"
            )
            receipt["cases"].append({"case": "missing-provider-key", "verified": True, "exit_code": code})
        receipt["all_verified"] = True
        print("Tier 2 image smoke verified changed-skill and missing-credential evidence.")
    except (OSError, ValueError, KeyError, TypeError, SmokeError, subprocess.SubprocessError) as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        print(f"Tier 2 image smoke failed: {error}", file=sys.stderr)
    finally:
        try:
            logs = _docker("logs", provider, check=False)
            if logs.returncode == 0:
                (output / "provider.log").write_text(logs.stdout + logs.stderr, encoding="utf-8")
            _docker("rm", "--force", provider, check=False)
        finally:
            (output / "smoke-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt["all_verified"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Already built or pulled local Docker image")
    parser.add_argument("--output", required=True, type=Path, help="New or empty evidence directory")
    if sys.argv[1:] == ["--serve-fixture"]:
        with ThreadingHTTPServer(("127.0.0.1", PROVIDER_PORT), FixtureProvider) as server:
            print("fixture-ready", flush=True)
            server.serve_forever()
        return 0
    args = parser.parse_args()
    try:
        return 0 if smoke(args.image, args.output) else 1
    except (OSError, SmokeError, subprocess.SubprocessError) as error:
        print(f"Tier 2 image smoke could not run: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
