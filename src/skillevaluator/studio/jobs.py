# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded subprocess jobs for the localhost workbench."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from skillevaluator.provider_config import CHAT_DEFAULT_MODELS
from skillevaluator.studio.store import IDENTIFIER, Workspace, copy_source, new_id, now, source_hash, write_json

PROVIDER_KEYS = {"nv_build": "NVIDIA_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
# Public free prototyping endpoint; catalog checked 2026-10-25.
DEFAULT_MODELS = CHAT_DEFAULT_MODELS | {"nv_build": "nvidia/nemotron-3.5-lightning-30b-a3b"}
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SENSITIVE = re.compile(
    r"(?:api[_-]?key|authorization|password|access[_-]?token|refresh[_-]?token|client[_-]?secret)", re.IGNORECASE
)


class BusyError(ValueError):
    """A second process would exceed the local job budget."""


class Studio:
    def __init__(self, root: Path):
        self.workspace = Workspace(root)
        self.token = __import__("secrets").token_urlsafe(32)
        self.provider_name = "nv_build"
        self.model = DEFAULT_MODELS[self.provider_name]
        self.keys: dict[str, str] = {}
        self._secret_values = {value for name in PROVIDER_KEYS.values() if (value := os.environ.get(name))}
        self.processes: dict[str, subprocess.Popen] = {}
        self.lock = threading.RLock()
        self._capabilities = None
        self._capabilities_at = 0.0
        # A previous process's PID is not trusted after restart.
        for path in self.workspace.jobs.glob("*/metadata.json"):
            try:
                record = json.loads(path.read_text())
                if record.get("status") not in TERMINAL:
                    record.update(
                        status="interrupted",
                        finished_at=now(),
                        error="The UI stopped before this job completed. Start a new run to continue.",
                    )
                    write_json(path, record)
            except (OSError, ValueError):
                continue

    def scrub(self, value: object) -> object:
        if isinstance(value, dict):
            return {k: "[redacted]" if SENSITIVE.search(str(k)) else self.scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.scrub(v) for v in value]
        if isinstance(value, str):
            with self.lock:
                secrets = tuple(sorted(self._secret_values, key=len, reverse=True))
            for secret in secrets:
                if secret:
                    value = value.replace(secret, "[redacted]")
            return ANSI.sub("", value)
        return value

    def provider(self) -> dict:
        configured = bool(self.keys.get(self.provider_name) or os.environ.get(PROVIDER_KEYS[self.provider_name]))
        return {
            "name": self.provider_name,
            "model": self.model,
            "configured": configured,
            "key_source": "session" if self.keys.get(self.provider_name) else "environment" if configured else None,
        }

    def configure(self, payload: dict) -> dict:
        name = payload.get("name", self.provider_name)
        if not isinstance(name, str) or name not in PROVIDER_KEYS:
            raise ValueError("Choose NVIDIA Build, OpenAI or Anthropic.")
        model = payload.get("model") or DEFAULT_MODELS[name]
        if not isinstance(model, str) or not model.strip() or len(model) > 200 or any(ord(c) < 32 for c in model):
            raise ValueError("Enter a valid model name.")
        key = payload.get("api_key")
        if key is not None and (not isinstance(key, str) or len(key) > 4096 or any(ord(c) < 32 for c in key)):
            raise ValueError("The API key must be a single line.")
        with self.lock:
            if key is not None and key.strip():
                self.keys[name] = key.strip()
                self._secret_values.add(key.strip())
            self.provider_name, self.model = name, model.strip()
            return self.provider()

    def capabilities(self) -> dict:
        if self._capabilities is not None and time.monotonic() - self._capabilities_at < 30:
            return self._capabilities
        docker_installed = bool(shutil.which("docker"))
        docker_ready = False
        if docker_installed:
            try:
                result = subprocess.run(
                    ["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, timeout=3, check=False
                )
                docker_ready = result.returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                pass
        agents = [
            {"id": agent, "available": bool(shutil.which(binary))}
            for agent, binary in [("opencode", "opencode"), ("claude-code", "claude"), ("codex", "codex")]
        ]
        sandbox = "Seatbelt" if sys.platform == "darwin" else "bubblewrap" if sys.platform.startswith("linux") else None
        sandbox_available = (
            bool(shutil.which("sandbox-exec" if sys.platform == "darwin" else "bwrap")) if sandbox else False
        )
        tier3 = importlib.util.find_spec("harbor") is not None
        self._capabilities = {
            "tier3": tier3,
            "docker": {
                "available": docker_ready,
                "installed": docker_installed,
                "reason": "Docker is ready."
                if docker_ready
                else "Start Docker Desktop or install Docker to use this mode.",
            },
            "local": {
                "available": tier3 and sandbox_available and any(a["available"] for a in agents),
                "sandbox": sandbox,
                "reason": "Uses the evaluator's required OS sandbox and an installed agent."
                if sandbox_available
                else "Local mode needs Seatbelt on macOS or bubblewrap on Linux.",
            },
            "agents": agents,
            "optimizer": True,
        }
        self._capabilities_at = time.monotonic()
        return self._capabilities

    def state(self) -> dict:
        from skillevaluator import __version__

        return {
            "version": __version__,
            "workspace": str(self.workspace.root),
            "token": self.token,
            "provider": self.provider(),
            "capabilities": self.capabilities(),
            "skills": self.workspace.list(),
            "jobs": self.list_jobs(),
        }

    def _job_path(self, job_id: str) -> Path:
        if not isinstance(job_id, str) or not IDENTIFIER.fullmatch(job_id):
            raise ValueError("Unknown run.")
        path = self.workspace.jobs / job_id
        if not (path / "metadata.json").is_file() or path.is_symlink():
            raise ValueError("Unknown run.")
        return path

    def job(self, job_id: str) -> dict:
        folder = self._job_path(job_id)
        record = json.loads((folder / "metadata.json").read_text())
        log = folder / "output.log"
        text = ""
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 96 * 1024))
                text = stream.read().decode("utf-8", errors="replace")
        events = []
        event_file = folder / "events.jsonl"
        if event_file.exists():
            with event_file.open("rb") as stream:
                stream.seek(max(0, event_file.stat().st_size - 256 * 1024))
                for line in stream.read().decode("utf-8", errors="replace").splitlines():
                    try:
                        events.append(json.loads(line))
                    except ValueError:
                        continue
        result = None
        result_file = folder / "result.json"
        if result_file.exists():
            with contextlib.suppress(ValueError):
                result = json.loads(result_file.read_text())
        record.update(logs=text.splitlines()[-500:], events=events[-100:], result=result)
        return self.scrub(record)

    def list_jobs(self) -> list[dict]:
        records = []
        for folder in self.workspace.jobs.iterdir():
            if IDENTIFIER.fullmatch(folder.name):
                try:
                    record = json.loads((folder / "metadata.json").read_text())
                    record.update(logs=[], events=[], result=None)
                    records.append(self.scrub(record))
                except (OSError, ValueError):
                    continue
        return sorted(records, key=lambda x: x["started_at"], reverse=True)[:30]

    @staticmethod
    def _options(payload: dict, default_model: str) -> dict:
        value = payload.get("options") or {}
        if not isinstance(value, dict):
            raise ValueError("Invalid run settings.")
        agent = value.get("agent", "opencode")
        mode = value.get("env_mode", "docker")
        grading = value.get("grading_mode", "default")
        if (
            not isinstance(agent, str)
            or not isinstance(mode, str)
            or agent not in {"opencode", "claude-code", "codex"}
            or mode not in {"local", "docker"}
        ):
            raise ValueError("Choose a supported agent and local or Docker execution.")
        if not isinstance(grading, str) or grading not in {"default", "default_plus_custom", "custom_only"}:
            raise ValueError("Choose a supported grading mode.")
        for key in ("skip_baseline", "no_llm", "tier1", "tier1_llm", "dedup"):
            if key in value and not isinstance(value[key], bool):
                raise ValueError(f"{key} must be true or false.")
        if value.get("model") is not None and not isinstance(value["model"], str):
            raise ValueError("Enter a model name as text.")
        options = {
            "agent": agent,
            "env_mode": mode,
            "grading_mode": grading,
            "model": str(value.get("model") or default_model),
            "skip_baseline": bool(value.get("skip_baseline", False)),
            "no_llm": bool(value.get("no_llm", False)),
            "tier1": value.get("tier1", False),
            "tier1_llm": value.get("tier1_llm", False),
            "dedup": value.get("dedup", False),
        }
        if options["tier1_llm"] and not options["tier1"]:
            raise ValueError("Select Scan skill before adding AI security review.")
        for key, minimum, maximum, default in [
            ("n_attempts", 1, 5, 1),
            ("n_concurrent", 1, 4, 1),
            ("timeout_multiplier", 1, 5, 3),
        ]:
            number = value.get(key, default)
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or number < minimum
                or number > maximum
                or int(number) != number
            ):
                raise ValueError(f"{key} must be a whole number from {minimum} to {maximum}.")
            options[key] = int(number)
        if len(options["model"]) > 200 or any(ord(c) < 32 for c in options["model"]):
            raise ValueError("Invalid model name.")
        return options

    def start(self, payload: dict) -> dict:
        from skillevaluator.studio import improvements

        action = payload.get("action")
        if not isinstance(action, str) or action not in {
            "draft",
            "validate",
            "check",
            "evaluate",
            "improve",
            "compare_improvement",
        }:
            raise ValueError("Choose a supported Studio action.")
        extra = {}
        if action == "compare_improvement":
            _root, proposal_metadata, _result = improvements.proposal(self.workspace, payload.get("proposal_job_id"))
            if not _result.get("candidate_id"):
                raise ValueError("This proposal has no changed candidate to evaluate.")
            payload = {**payload, "skill_id": proposal_metadata["skill_id"]}
            extra = {key: payload.get(key) for key in ("proposal_job_id", "before_job_id", "after_job_id")}
            extra["outcome_metrics"] = improvements.outcome_metrics(payload.get("outcome_metrics", []))
            improvements.evaluation_run(self.workspace, extra["before_job_id"], skill_id=proposal_metadata["skill_id"])
            improvements.evaluation_run(self.workspace, extra["after_job_id"], skill_id=_result.get("candidate_id"))
        skill_id = payload.get("skill_id", "")
        skill = self.workspace.describe(skill_id)
        with self.lock:
            provider_name, default_model = self.provider_name, self.model
            provider_key = self.keys.get(provider_name) or os.environ.get(PROVIDER_KEYS[provider_name])
        options = self._options(payload, default_model)
        reference = None
        if action == "improve":
            configuration = payload.get("improvement", {})
            if not isinstance(configuration, dict) or set(configuration) - {
                "mode",
                "include_agent_observations",
                "edits",
            }:
                raise ValueError("Choose an improvement mode and supported proposal settings.")
            configuration = {"mode": "llm", "include_agent_observations": False, **configuration}
            if (
                not isinstance(configuration["mode"], str)
                or configuration["mode"] not in {"export", "llm", "deterministic"}
                or not isinstance(configuration["include_agent_observations"], bool)
            ):
                raise ValueError("Choose export, LLM or reviewed deterministic edits.")
            if ("edits" in configuration) != (configuration["mode"] == "deterministic"):
                raise ValueError("Reviewed proposal JSON is required only for deterministic edits.")
            if len(json.dumps(configuration).encode()) > 1024 * 1024:
                raise ValueError("The reviewed proposal exceeds 1 MiB.")
            discovery_id = payload.get("discovery_job_id")
            improvements.evaluation_run(self.workspace, discovery_id, skill_id=skill_id)
            ready = improvements.readiness(self.workspace, discovery_id, skill_id)
            if configuration["mode"] != "export" and not ready["eligible"]:
                raise ValueError(" ".join(ready["reasons"]))
            extra = {"skill_id": skill_id, "discovery_job_id": discovery_id, "improvement": configuration}
        if action == "evaluate" and payload.get("validation_of"):
            before_id, before, reference = improvements.validation_reference(
                self.workspace, payload["validation_of"], skill_id, payload.get("before_job_id")
            )
            if provider_name != before.get("provider"):
                raise ValueError(
                    "Reconnect the original run's provider before testing this proposal. Its model and evaluation settings will be replayed."
                )
            options = self._options({"options": before["options"]}, default_model)
            extra = {
                "validation_of": payload["validation_of"],
                "before_job_id": before_id,
                "outcome_metrics": improvements.outcome_metrics(payload.get("outcome_metrics", [])),
            }
        selected_checks = options["tier1"] or options["dedup"]
        if selected_checks and action not in {"check", "evaluate"}:
            raise ValueError("Extra checks are available with skill checks or agent evaluation.")
        if options["dedup"] and provider_name == "anthropic":
            raise ValueError(
                "Repeated-content checks need embeddings. Choose NVIDIA API Catalog or OpenAI, "
                "which support embeddings and model review with one provider key."
            )
        needs_provider = (
            (action == "draft" and not options["no_llm"])
            or action == "evaluate"
            or (action == "check" and (options["tier1_llm"] or options["dedup"]))
            or (action == "improve" and extra["improvement"]["mode"] == "llm")
        )
        if needs_provider and not provider_key:
            raise ValueError("Set a provider key first. Starter drafts and static scans work without one.")
        if action == "evaluate":
            from skillevaluator.studio.adapter import is_native_custom_only

            if not skill["cases"] and not (
                options["grading_mode"] == "custom_only" and is_native_custom_only(self.workspace.directory(skill_id))
            ):
                raise ValueError("Save and validate your test cases before evaluating.")
            capabilities = self.capabilities()
            if not capabilities["tier3"]:
                raise ValueError('Install live evaluation support: pip install "skillevaluator[ui,tier3]".')
            if not capabilities[options["env_mode"]]["available"]:
                raise ValueError(capabilities[options["env_mode"]]["reason"])
            if options["env_mode"] == "local" and not next(
                a["available"] for a in capabilities["agents"] if a["id"] == options["agent"]
            ):
                raise ValueError(f"Install the {options['agent']} runtime before running locally.")
        guidance = payload.get("guidance") or {}
        if not isinstance(guidance, dict) or any(not isinstance(v, str) or len(v) > 12_000 for v in guidance.values()):
            raise ValueError("Keep each guidance answer under 12,000 characters.")
        with self.lock:
            if any(p.poll() is None for p in self.processes.values()):
                raise BusyError(
                    "Another task is running. You can keep editing; wait for it to finish "
                    "or stop it in Run history before starting another task."
                )
            job_id = new_id()
            folder = self.workspace.jobs / job_id
            folder.mkdir(mode=0o700)
            source_name = re.sub(r"[^a-z0-9-]+", "-", skill["name"].lower()).strip("-")[:64] or "uploaded-skill"
            if reference is not None:
                source_name = reference.name
            source = folder / "source" / source_name
            if action == "improve" or reference is not None or skill.get("proposal_job_id"):
                improvements.copy_package(self.workspace.directory(skill_id), source)
            else:
                copy_source(self.workspace.directory(skill_id), source)
            if reference is not None:
                improvements.freeze_evaluation(reference, source)
            from skillevaluator.tier3.dataset_utils import find_eval_file

            frozen_dataset = find_eval_file(source)
            frozen_source_hash = source_hash(source)
            frozen_dataset_hash = hashlib.sha256(frozen_dataset.read_bytes()).hexdigest() if frozen_dataset else None
            write_json(
                folder / "request.json",
                {
                    "action": action,
                    "skill_path": str(source),
                    "results_dir": str(folder / "results"),
                    "guidance": guidance,
                    "options": options,
                    **extra,
                },
            )
            metadata = {
                "id": job_id,
                "skill_id": skill_id,
                "action": action,
                "status": "running",
                "started_at": now(),
                "finished_at": None,
                "error": None,
                "source_hash": frozen_source_hash,
                "dataset_hash": frozen_dataset_hash,
                "provider": provider_name,
                "options": options,
                **{key: value for key, value in extra.items() if key != "improvement"},
            }
            write_json(folder / "metadata.json", metadata)
            env = os.environ.copy()
            for credential_name in PROVIDER_KEYS.values():
                env.pop(credential_name, None)
            if provider_key:
                env[PROVIDER_KEYS[provider_name]] = provider_key
            env.update(
                SKILL_EVAL_LLM_PROVIDER=provider_name,
                SKILL_EVAL_LLM_MODEL=options["model"],
                SKILL_EVAL_TELEMETRY_ENABLED="false",
                SKILL_EVAL_DISABLE_TELEMETRY="1",
                PYTHONUNBUFFERED="1",
                PYTHON_DOTENV_DISABLED="1",
                SKILLEVALUATOR_MAX_HARBOR_SUBPROCESSES="1",
            )
            # One selected public provider; inherited endpoint overrides must not send its key elsewhere.
            for name in [
                "OPENAI_BASE_URL",
                "ANTHROPIC_BASE_URL",
                "SKILL_EVAL_LLM_BASE_URL",
                "SKILL_EVAL_LLM_API_KEY",
                "SKILL_EVAL_EMBEDDING_PROVIDER",
                "SKILL_EVAL_EMBEDDING_MODEL",
                "SKILL_EVAL_EMBEDDING_API_KEY",
                "SKILL_EVAL_EMBEDDING_BASE_URL",
                "SKILLSPECTOR_PROVIDER",
                "SKILLSPECTOR_MODEL",
            ]:
                env.pop(name, None)
            for name in list(env):
                if name.startswith("SKILLSPECTOR_MODEL_"):
                    env.pop(name)
            if options["dedup"]:
                env["SKILL_EVAL_EMBEDDING_PROVIDER"] = provider_name
            try:
                process = subprocess.Popen(
                    [sys.executable, "-m", "skillevaluator.studio.worker", str(folder)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                    start_new_session=os.name != "nt",
                )
            except OSError as exc:
                metadata.update(status="failed", finished_at=now(), error=str(exc))
                write_json(folder / "metadata.json", metadata)
                raise ValueError("The job could not start. Check your Python installation.") from exc
            self.processes[job_id] = process
            threading.Thread(target=self._watch, args=(job_id, process), daemon=True).start()
        return self.job(job_id)

    def _watch(self, job_id: str, process: subprocess.Popen) -> None:
        folder = self.workspace.jobs / job_id
        with (folder / "output.log").open("w", encoding="utf-8") as stream:
            stream.write("Starting the existing SkillEvaluator engine in an isolated job process.\n")
            stream.flush()
            for line in process.stdout:
                if stream.tell() < 2 * 1024 * 1024:
                    stream.write(str(self.scrub(line)))
                    stream.flush()
            code = process.wait()
        with self.lock:
            record = json.loads((folder / "metadata.json").read_text())
            result_exists = (folder / "result.json").exists()
            stopped = record["status"] == "stopping"
            record.update(
                status="cancelled" if stopped else "succeeded" if code == 0 and result_exists else "failed",
                finished_at=now(),
            )
            if record["status"] == "failed":
                record["error"] = f"Evaluation process exited with status {code}. Review the saved result and log."
            write_json(folder / "metadata.json", record)
            self.processes.pop(job_id, None)

    def cancel(self, job_id: str) -> dict:
        folder = self._job_path(job_id)
        with self.lock:
            process = self.processes.get(job_id)
            if process is None or process.poll() is not None:
                return self.job(job_id)
            record = json.loads((folder / "metadata.json").read_text())
            record["status"] = "stopping"
            write_json(folder / "metadata.json", record)
            try:
                if os.name == "nt":
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            threading.Thread(target=self._force_stop, args=(process,), daemon=True).start()
        return self.job(job_id)

    @staticmethod
    def _force_stop(process: subprocess.Popen) -> None:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def close(self) -> None:
        with self.lock:
            active = list(self.processes)
        for job_id in active:
            self.cancel(job_id)
        for process in list(self.processes.values()):
            self._force_stop(process)
