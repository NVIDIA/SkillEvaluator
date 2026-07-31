# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Harbor Adapter -- converts evals/evals.json to Harbor task directories.

Generates two Harbor datasets from a single evals.json:
  - harbor-tasks/        (with skill installed)
  - harbor-tasks-baseline/ (without skill, reference skills only)

Each dataset entry becomes one Harbor task directory with:
  instruction.md, task.toml, environment/Dockerfile or environment/skills,
  tests/eval.py, tests/entry.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from skillevaluator.tier3.case_ids import safe_child, validate_case_ids
from skillevaluator.tier3.harbor.secure_copy import copytree_secure
from skillevaluator.tier3.toml_utils import toml_quote
from skillevaluator.utils.process_environment import child_process_env

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
_EVAL_CORE_DIR = Path(__file__).resolve().parent.parent / "eval_core"
_BASE_IMAGE_PREFIX = "skillevaluator-base"
_MAX_REPO_CONTEXT_FILE_BYTES = 10 * 1024 * 1024
_MAX_REPO_CONTEXT_TOTAL_BYTES = 200 * 1024 * 1024
_REPO_CONTEXT_IGNORE_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
    "target",
    ".env",
    ".env.local",
    ".envrc",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials",
    "credentials.json",
    "secrets.json",
}
_REPO_CONTEXT_IGNORE_PARTS = {("evals", "results")}
_REPO_CONTEXT_IGNORE_SUFFIXES = (".pem", ".key", ".p12", ".pfx")
_REPO_CONTEXT_PUBLIC_ENV_SUFFIXES = (".dist", ".example", ".sample", ".template")
_REPO_CONTEXT_SENSITIVE_NAMES = {
    ".git-credentials",
    ".gitcredentials",
    ".npmrc",
    ".pypirc",
    ".terraformrc",
    ".yarnrc",
    ".yarnrc.yml",
    "_netrc",
    "access_tokens.db",
    "application_default_credentials.json",
    "credentials.db",
    "credentials.tfrc.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "terraform.rc",
}
_REPO_CONTEXT_SENSITIVE_COMPONENTS = {
    ".azure",
}
_REPO_CONTEXT_SENSITIVE_PARTS = {
    (".aws", "config"),
    (".aws", "credentials"),
    (".config", "doctl", "config.yaml"),
    (".config", "gcloud"),
    (".config", "gh", "hosts.yml"),
    (".config", "pypoetry", "auth.toml"),
    (".docker", "config.json"),
    (".kube", "config"),
    (".pulumi", "credentials.json"),
    (".terraform.d", "credentials.tfrc.json"),
}
_MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
_PLAIN_RELATIVE_PATH_RE = re.compile(r"(?<![\w/])(?:\.\.?/)+(?:[^\s\])<>\"']+)")
_LOCAL_LINK_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
_COMPOSE_ENV_RE = re.compile(r"(?<!\$)(?:\$\$)*\$(?:\{([A-Za-z_][A-Za-z0-9_]*)|([A-Za-z_][A-Za-z0-9_]*))")
_COMPOSE_NAMED_VOLUME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_COMPOSE_SSH_PATH_RE = re.compile(r"^(?:[^/\\\s@:]+@)?[^/\\\s:]+:.+$")
_COMPOSE_WINDOWS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_COMPOSE_ALLOWED_TOP_LEVEL_KEYS = frozenset({"networks", "services", "version", "volumes"})
_COMPOSE_SIDECAR_ALLOWED_KEYS = frozenset(
    {
        "build",
        "cap_drop",
        "command",
        "depends_on",
        "entrypoint",
        "environment",
        "expose",
        "healthcheck",
        "image",
        "init",
        "networks",
        "platform",
        "ports",
        "pull_policy",
        "read_only",
        "stop_grace_period",
        "stop_signal",
        "tmpfs",
        "user",
        "volumes",
        "working_dir",
    }
)
_COMPOSE_MAIN_ALLOWED_KEYS = frozenset({"depends_on"})
_COMPOSE_ALLOWED_BUILD_KEYS = frozenset(
    {
        "additional_contexts",
        "args",
        "context",
        "dockerfile",
        "dockerfile_inline",
        "labels",
        "no_cache",
        "pull",
        "target",
    }
)
_COMPOSE_ALLOWED_NETWORK_KEYS = frozenset({"attachable", "enable_ipv4", "enable_ipv6", "internal", "labels"})
_COMPOSE_ALLOWED_VOLUME_KEYS = frozenset({"labels"})
_VERIFIER_PROVIDER_ENV_VARS = frozenset(
    {
        "SKILL_EVAL_LLM_PROVIDER",
        "SKILL_EVAL_LLM_MODEL",
        "SKILL_EVAL_LLM_API_KEY",
        "SKILL_EVAL_LLM_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "NVIDIA_API_KEY",
        "AWS_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_CONFIG_FILE",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_DEFAULT_REGION",
        "AWS_PROFILE",
        "AWS_ROLE_ARN",
        "AWS_ROLE_SESSION_NAME",
        "AWS_SDK_LOAD_CONFIG",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    }
)


def _verifier_env_vars(runtime_env: dict[str, str] | None = None) -> tuple[str, ...]:
    """Return public provider variables explicitly staged for the verifier."""
    return tuple(sorted(set(runtime_env or {}).intersection(_VERIFIER_PROVIDER_ENV_VARS)))


def _verifier_env_block(runtime_env: dict[str, str] | None = None, indent: str = "") -> str:
    return "\n".join(f'{indent}{name} = "${{{name}}}"' for name in _verifier_env_vars(runtime_env))


def _find_repo_root(path: Path) -> Path | None:
    """Return the git repo root for *path*, falling back to parent .git search."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return Path(result.stdout.strip()).resolve()
    except Exception:
        pass

    current = path.resolve()
    if current.is_file():
        current = current.parent
    for parent in (current, *current.parents):
        if (parent / ".git").exists():
            return parent
    return None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _strip_link_target(raw_target: str) -> str:
    target = raw_target.strip()
    if not target:
        return ""
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    elif any(ch.isspace() for ch in target):
        # Markdown allows optional titles: [x](../file.md "title").
        target = target.split(None, 1)[0]
    target = target.split("#", 1)[0].split("?", 1)[0]
    target = unquote(target)
    return target.strip().strip("'\"")


def _is_local_link_target(target: str) -> bool:
    if not target or target.startswith("#"):
        return False
    return not _LOCAL_LINK_SCHEME_RE.match(target)


def _discover_skill_link_targets(skill_md: Path) -> list[str]:
    try:
        text = skill_md.read_text(encoding="utf-8")
    except OSError:
        return []

    targets: list[str] = []
    for match in _MARKDOWN_LINK_RE.finditer(text):
        target = _strip_link_target(match.group(1))
        if _is_local_link_target(target):
            targets.append(target)

    for match in _PLAIN_RELATIVE_PATH_RE.finditer(text):
        target = _strip_link_target(match.group(0).rstrip(".,;:"))
        if _is_local_link_target(target):
            targets.append(target)

    seen: set[str] = set()
    unique: list[str] = []
    for target in targets:
        if target not in seen:
            seen.add(target)
            unique.append(target)
    return unique


def _safe_repo_context_rel(path: Path, root: Path) -> Path | None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    try:
        rel = path.absolute().relative_to(root.absolute())
    except ValueError:
        return None
    if any(part in {"", ".", ".."} for part in rel.parts):
        return None
    return rel


def _staged_relative_link_path(skill_name: str, raw_target: str) -> Path | None:
    target = _strip_link_target(raw_target)
    if not target or Path(target).is_absolute():
        return None
    normalized = posixpath.normpath(posixpath.join("skills", skill_name, target.replace("\\", "/")))
    if normalized in {"", "."} or normalized.startswith("../"):
        return None
    rel = Path(normalized)
    # Do not let linked repo docs stage sibling skills as discoverable skills.
    if rel.parts[:1] == ("skills",) and (len(rel.parts) < 2 or rel.parts[1] != skill_name):
        return None
    return rel


def _repo_context_ignore_file(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return True
    try:
        rel = path.absolute().relative_to(root.absolute())
    except ValueError:
        return True
    parts = tuple(part.casefold() for part in rel.parts)
    name = parts[-1]
    if any(part in _REPO_CONTEXT_IGNORE_NAMES for part in parts):
        return True
    if any(part in _REPO_CONTEXT_SENSITIVE_COMPONENTS for part in parts):
        return True
    if name in {".env", ".envrc"}:
        return True
    if name.startswith(".env.") and not name.endswith(_REPO_CONTEXT_PUBLIC_ENV_SUFFIXES):
        return True
    if name in _REPO_CONTEXT_SENSITIVE_NAMES:
        return True
    if name.endswith(_REPO_CONTEXT_IGNORE_SUFFIXES):
        return True
    for ignored in (*_REPO_CONTEXT_IGNORE_PARTS, *_REPO_CONTEXT_SENSITIVE_PARTS):
        if any(parts[index : index + len(ignored)] == ignored for index in range(len(parts) - len(ignored) + 1)):
            return True
    return False


def _copy_repo_context_file(src: Path, dest_root: Path, rel: Path, *, total_bytes: list[int]) -> bool:
    if not src.is_file():
        return False
    try:
        size = src.stat().st_size
    except OSError:
        return False
    if size > _MAX_REPO_CONTEXT_FILE_BYTES:
        logger.warning("Skipping linked repo file over size limit: %s", src)
        return False
    if total_bytes[0] + size > _MAX_REPO_CONTEXT_TOTAL_BYTES:
        logger.warning("Skipping repo context after total size limit: %s", src)
        return False
    dest = dest_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    total_bytes[0] += size
    return True


def _git_context_files(root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-co", "--exclude-standard"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    return [root / line for line in result.stdout.splitlines() if line.strip()]


def _iter_repo_context_files(root: Path) -> list[Path]:
    git_files = _git_context_files(root)
    if git_files:
        return sorted(path for path in git_files if path.is_file())
    return sorted(path for path in root.rglob("*") if path.is_file() and not _repo_context_ignore_file(path, root))


def _stage_repo_context(
    env_dir: Path,
    *,
    source_skill_path: Path | None,
    mode: str,
    exclude_source_skill: bool = False,
) -> dict[str, Any]:
    """Stage repo files referenced by SKILL.md, or the full repo when requested."""
    if not source_skill_path:
        return {"mode": "none", "files": []}

    source_skill_path = source_skill_path.resolve()
    skill_md = source_skill_path / "SKILL.md"
    repo_root = _find_repo_root(source_skill_path)
    if repo_root is None:
        repo_root = source_skill_path.parent.resolve()

    repo_dest = env_dir / "repo"
    linked_root_dest = env_dir / "repo-linked-root"
    total_bytes = [0]
    staged: list[dict[str, str]] = []

    if mode == "full":
        for src in _iter_repo_context_files(repo_root):
            if _repo_context_ignore_file(src, repo_root):
                continue
            if exclude_source_skill and _is_relative_to(src, source_skill_path):
                continue
            rel = _safe_repo_context_rel(src, repo_root)
            if rel is None:
                continue
            if _copy_repo_context_file(src, repo_dest, rel, total_bytes=total_bytes):
                staged.append({"source": str(src), "container": f"/workspace/repo/{rel.as_posix()}"})
        metadata = {"mode": "full", "repo_root": str(repo_root), "files": staged}
        if staged:
            (env_dir / "repo-context.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    if mode != "linked" or not skill_md.exists():
        return {"mode": mode, "repo_root": str(repo_root), "files": []}

    for raw_target in _discover_skill_link_targets(skill_md):
        target = _strip_link_target(raw_target)
        target_path = (source_skill_path / target).resolve() if not Path(target).is_absolute() else Path(target)
        if not target_path.is_file():
            continue
        if not _is_relative_to(target_path, repo_root):
            logger.warning("Skipping SKILL.md link outside repo root: %s", raw_target)
            continue
        if _repo_context_ignore_file(target_path, repo_root):
            logger.warning("Skipping ignored SKILL.md repo link: %s", raw_target)
            continue
        if _is_relative_to(target_path, source_skill_path):
            continue
        rel = _safe_repo_context_rel(target_path, repo_root)
        if rel is None:
            continue
        copied = _copy_repo_context_file(target_path, repo_dest, rel, total_bytes=total_bytes)
        compat_rel = _staged_relative_link_path(source_skill_path.name, raw_target)
        if compat_rel is not None:
            _copy_repo_context_file(target_path, linked_root_dest, compat_rel, total_bytes=total_bytes)
        if copied:
            staged.append(
                {
                    "source": str(target_path),
                    "container": f"/workspace/repo/{rel.as_posix()}",
                    "link": raw_target,
                }
            )

    metadata = {"mode": "linked", "repo_root": str(repo_root), "files": staged}
    if staged:
        (env_dir / "repo-context.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _collect_all_skill_deps(
    skill_path: Path | None,
    reference_skills_dir: Path | None,
    workspace_skill_paths: list[Path] | None = None,
) -> tuple[list[str], list[str]]:
    """Collect pip and apt deps from target skill + staged workspace skills."""
    all_pip: set[str] = set()
    all_apt: set[str] = set()

    skill_dirs: list[Path] = []
    if skill_path and skill_path.exists():
        skill_dirs.append(skill_path)
    if reference_skills_dir and reference_skills_dir.exists():
        for ref in reference_skills_dir.iterdir():
            if ref.is_dir() and not ref.name.startswith("."):
                skill_dirs.append(ref)
    for workspace_skill in workspace_skill_paths or []:
        if workspace_skill.exists():
            skill_dirs.append(workspace_skill)

    for sd in skill_dirs:
        for req_file in sd.rglob("requirements.txt"):
            for line in req_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    all_pip.add(line)
        for apt_file in sd.rglob("apt-packages.txt"):
            for line in apt_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    all_apt.add(line)

    return sorted(all_pip), sorted(all_apt)


def build_eval_base_image(
    skill_path: Path,
    reference_skills_dir: Path | None = None,
    *,
    workspace_skill_paths: list[Path] | None = None,
    force_rebuild: bool = False,
    action_out: list[str] | None = None,
) -> str:
    """Pre-build a Docker base image with verifier and public-provider dependencies.

    Builds once and tags with a content hash so rebuilds only happen when
    dependencies change.  Subsequent calls return instantly when the image exists
    unless ``force_rebuild`` is true.

    When the skill provides a custom ``evals/environment/Dockerfile``, skip
    collecting deps from the skill itself — the custom Dockerfile handles
    those.  We still collect deps from reference skills.

    Returns the image tag (e.g. ``skillevaluator-base:a1b2c3d4e5f6``) or ``""``
    on failure (callers fall back to full per-task Dockerfiles).
    """
    has_custom_dockerfile = (
        skill_path
        and (skill_path / "evals" / "environment" / "Dockerfile").is_file()
        and _validate_custom_dockerfile(skill_path / "evals" / "environment" / "Dockerfile") is None
    )

    if has_custom_dockerfile:
        extra_pip, extra_apt = _collect_all_skill_deps(None, reference_skills_dir, workspace_skill_paths)
    else:
        extra_pip, extra_apt = _collect_all_skill_deps(skill_path, reference_skills_dir, workspace_skill_paths)

    lines = [
        "FROM python:3.12-slim",
        "",
        "RUN apt-get -o Acquire::Retries=3 update && \\",
        "    apt-get -o Acquire::Retries=3 install -y --no-install-recommends \\",
        "    bash curl git jq ripgrep \\",
    ]
    if extra_apt:
        lines[-1] += " \\"
        lines.append("    " + " ".join(extra_apt) + " \\")
    lines.append("    && rm -rf /var/lib/apt/lists/*")

    lines.extend(
        [
            "",
            "RUN pip install --no-cache-dir \\",
            "    ragas~=0.4.0 \\",
            "    openai>=1.0 \\",
            "    anthropic>=0.40 \\",
            "    boto3>=1.34",
        ]
    )

    if extra_pip:
        escaped = " ".join(f'"{r}"' for r in extra_pip)
        lines.append(f"RUN pip install --no-cache-dir {escaped}")

    lines.extend(
        [
            "",
            "RUN mkdir -p /workspace/skills /workspace/input /workspace/output \\",
            "    /logs/verifier /logs/agent",
            "",
            "WORKDIR /workspace",
        ]
    )

    content = "\n".join(lines) + "\n"
    tag_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
    image_tag = f"{_BASE_IMAGE_PREFIX}:{tag_hash}"

    try:
        check = subprocess.run(
            ["docker", "image", "inspect", image_tag],
            capture_output=True,
            timeout=10,
            env=child_process_env(),
        )
        if check.returncode == 0 and not force_rebuild:
            logger.debug("Base image %s already exists, skipping build", image_tag)
            if action_out is not None:
                action_out.append("reused")
            return image_tag
    except (subprocess.TimeoutExpired, OSError):
        pass

    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "Dockerfile").write_text(content, encoding="utf-8")

        logger.debug("Building eval base image %s ...", image_tag)
        try:
            cmd = ["docker", "build"]
            if force_rebuild:
                cmd.extend(["--pull", "--no-cache"])
            cmd.extend(["-t", image_tag, tmpdir])
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=child_process_env(),
            )
        except subprocess.TimeoutExpired:
            logger.error("Base image build timed out after 600s")
            return ""

        if result.returncode != 0:
            stderr_tail = (result.stderr or "")[-500:]
            logger.error("Base image build failed: %s", stderr_tail)
            if action_out is not None:
                action_out.append("failed")
            return ""

        logger.debug("Built eval base image: %s", image_tag)
    if action_out is not None:
        action_out.append("rebuilt" if force_rebuild else "built")
    return image_tag


def _set_task_docker_image(task_dir: Path, image_tag: str) -> None:
    """Inject ``docker_image`` into a task's ``task.toml`` so Harbor skips building."""
    toml_path = task_dir / "task.toml"
    if not toml_path.exists():
        return
    content = toml_path.read_text(encoding="utf-8")
    if "docker_image" in content:
        return
    import re

    replaced = re.sub(
        r"(\[environment\]\s*\n)",
        rf'\1docker_image = "{image_tag}"\n',
        content,
        count=1,
    )
    if replaced == content:
        logger.warning("Could not inject docker_image into %s — [environment] section not found", toml_path)
        return
    toml_path.write_text(replaced, encoding="utf-8")


def prebuild_task_environments(dataset_dirs: list[Path]) -> int:
    """Pre-build Docker images for generated task environments.

    After ``generate_harbor_tasks`` has written all task directories, this
    function builds each unique environment once, tags it, and sets
    ``docker_image`` in every task's ``task.toml`` so Harbor uses the prebuilt
    image directly (skipping ``docker compose build`` entirely).

    Returns the number of tasks configured to use prebuilt images.
    """
    # Harbor currently tears down trial environments with ``docker compose down
    # --rmi all``.  When multiple tasks share one injected ``docker_image``,
    # the first completed trial can remove the shared prebuilt image while
    # sibling trials still need it, causing later trials to fail with Docker
    # pull errors.  Keep this optimization opt-in until Harbor exposes a safe
    # way to retain prebuilt images during cleanup.
    if os.environ.get("SKILL_EVAL_HARBOR_PREBUILD_TASK_ENVS") != "1":
        logger.debug("Skipping Harbor task environment pre-build; set SKILL_EVAL_HARBOR_PREBUILD_TASK_ENVS=1 to opt in")
        return 0

    rewritten = 0

    for dataset_dir in dataset_dirs:
        if not dataset_dir or not dataset_dir.exists():
            continue

        task_dirs = [
            d
            for d in sorted(dataset_dir.iterdir())
            if d.is_dir() and not d.name.startswith((".", "_")) and (d / "environment" / "Dockerfile").exists()
        ]
        if not task_dirs:
            continue

        first_env = task_dirs[0] / "environment"

        ctx_hash = hashlib.sha256()
        for f in sorted(first_env.rglob("*")):
            if f.is_file():
                ctx_hash.update(str(f.relative_to(first_env)).encode())
                ctx_hash.update(f.read_bytes())
        tag = f"skillevaluator-env:{ctx_hash.hexdigest()[:12]}"

        try:
            check = subprocess.run(
                ["docker", "image", "inspect", tag],
                capture_output=True,
                timeout=10,
                env=child_process_env(),
            )
            if check.returncode == 0:
                logger.debug("Pre-built env %s exists, reusing for %d tasks", tag, len(task_dirs))
                for td in task_dirs:
                    _set_task_docker_image(td, tag)
                rewritten += len(task_dirs)
                continue
        except (subprocess.TimeoutExpired, OSError):
            pass

        logger.debug("Pre-building task environment as %s ...", tag)
        try:
            result = subprocess.run(
                ["docker", "build", "-t", tag, str(first_env)],
                capture_output=True,
                text=True,
                timeout=600,
                env=child_process_env(),
            )
        except subprocess.TimeoutExpired:
            logger.warning("Environment pre-build timed out for %s", tag)
            continue

        if result.returncode != 0:
            stderr_tail = (result.stderr or "")[-500:]
            logger.warning("Environment pre-build failed for %s: %s", tag, stderr_tail)
            continue

        for td in task_dirs:
            _set_task_docker_image(td, tag)
        rewritten += len(task_dirs)
        logger.debug("Pre-built env %s, configured %d tasks", tag, len(task_dirs))

    return rewritten


def _load_evals(evals_path: Path) -> list[dict[str, Any]]:
    """Load normalized dataset entries from evals.json/jsonl/yaml."""
    from skillevaluator.tier3.dataset_utils import load_dataset_entries

    return load_dataset_entries(evals_path)


def find_evals_file(skill_path: Path) -> Path | None:
    """Return the first supported SkillEvaluator eval dataset for a skill, if present."""
    evals_dir = skill_path / "evals"
    for name in ("evals.json", "evals.jsonl", "evals.yaml", "evals.yml", "dataset.json", "dataset.jsonl"):
        candidate = evals_dir / name
        if candidate.exists():
            return candidate
    return None


def _preflight_generated_tasks(entries: list[dict[str, Any]], output_dir: Path) -> list[tuple[dict[str, Any], str]]:
    case_ids = validate_case_ids(entry.get("id") for entry in entries)
    prepared = [({**entry, "id": case_id}, case_id) for entry, case_id in zip(entries, case_ids, strict=True)]
    for _entry, case_id in prepared:
        safe_child(output_dir, case_id)
    return prepared


def _write_instruction(task_dir: Path, question: str) -> None:
    (task_dir / "instruction.md").write_text(question + "\n", encoding="utf-8")


def _load_mcp_servers(skill_path: Path) -> list[dict[str, Any]]:
    """Load MCP server declarations from evals/environment/mcp_servers.toml."""
    mcp_file = skill_path / "evals" / "environment" / "mcp_servers.toml"
    if not mcp_file.exists():
        return []
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    try:
        data = tomllib.loads(mcp_file.read_text(encoding="utf-8"))
        servers = data.get("mcp_servers", [])
        if not isinstance(servers, list):
            logger.warning("mcp_servers.toml: expected [[mcp_servers]] array, got %s", type(servers).__name__)
            return []
        valid = []
        for s in servers:
            if not isinstance(s, dict) or "name" not in s:
                logger.warning("mcp_servers.toml: skipping entry missing 'name': %s", s)
                continue
            if "url" not in s and "command" not in s:
                logger.warning("mcp_servers.toml: entry '%s' needs 'url' or 'command'", s.get("name"))
                continue
            if "command" in s and "transport" not in s:
                s = {**s, "transport": "stdio"}
                logger.debug("mcp_servers.toml: inferred transport=stdio for '%s'", s["name"])
            valid.append(s)
        if valid:
            logger.debug("Loaded %d MCP server(s) from %s", len(valid), mcp_file)
        return valid
    except Exception as e:
        logger.warning("Failed to parse %s: %s", mcp_file, e)
        return []


def _task_resource_value(resources: dict[str, int] | None, key: str, default: int) -> int:
    if not resources or key not in resources:
        return default
    return int(resources[key])


def _write_task_toml(
    task_dir: Path,
    entry: dict[str, Any],
    has_skill: bool,
    mcp_servers: list[dict[str, Any]] | None = None,
    docker_image: str = "",
    runtime_env: dict[str, str] | None = None,
    verifier_env: dict[str, str] | None = None,
    pre_agent_setup: list[str] | None = None,
    task_resources: dict[str, int] | None = None,
    agent_workdir: str | None = None,
) -> None:
    entry_id = entry.get("id", "unknown")
    expected_skill = entry.get("expected_skill") or "none"
    if not isinstance(entry_id, str):
        raise TypeError("entry id must be a string before Harbor TOML serialization")
    if not isinstance(expected_skill, str):
        raise TypeError("expected_skill must be a string before Harbor TOML serialization")
    if not isinstance(docker_image, str):
        raise TypeError("docker_image must be a string before Harbor TOML serialization")
    docker_image_line = f"docker_image = {_toml_quote(docker_image)}\n" if docker_image else ""
    cpus = _task_resource_value(task_resources, "cpus", 2)
    memory_mb = _task_resource_value(task_resources, "memory_mb", 4096)
    storage_mb = _task_resource_value(task_resources, "storage_mb", 2048)
    workdir_line = f"workdir = {_toml_quote(agent_workdir)}\n" if agent_workdir else ""

    content = f"""schema_version = "1.3"

[task]
name = {_toml_quote(f"nvidia/skillevaluator-{entry_id}")}
description = {_toml_quote(f"Skill evaluation task for {expected_skill}")}

[metadata]
skill = {_toml_quote(expected_skill)}
entry_id = {_toml_quote(entry_id)}
has_skill = {str(has_skill).lower()}

[agent]
timeout_sec = 300.0

[verifier]
timeout_sec = 180.0

[verifier.env]
{_verifier_env_block(verifier_env if verifier_env is not None else runtime_env)}

[environment]
{docker_image_line}cpus = {cpus}
memory_mb = {memory_mb}
storage_mb = {storage_mb}
{workdir_line}\
network_mode = "public"
skills_dir = "/workspace/skills"
"""

    content += _runtime_env_toml_block(runtime_env)
    content += _pre_agent_setup_healthcheck_toml_block(pre_agent_setup)

    if mcp_servers:
        for srv in mcp_servers:
            content += "\n[[environment.mcp_servers]]\n"
            for key, val in srv.items():
                if not isinstance(key, str):
                    raise TypeError("MCP TOML keys must be strings")
                content += f"{_toml_quote(key)} = {_toml_value(val)}\n"

    tomllib.loads(content)
    (task_dir / "task.toml").write_text(content, encoding="utf-8")


def _toml_quote(value: str) -> str:
    """Return a TOML-compatible quoted string."""
    return toml_quote(value)


def _toml_value(value: Any) -> str:
    """Serialize the documented MCP TOML scalar and string-list values."""

    if isinstance(value, str):
        return _toml_quote(value)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "[" + ", ".join(_toml_quote(item) for item in value) + "]"
    raise TypeError("MCP TOML values must be strings or lists of strings")


def _runtime_env_toml_block(runtime_env: dict[str, str] | None) -> str:
    if not runtime_env:
        return ""
    lines = ["", "[environment.env]"]
    for key in sorted(runtime_env):
        lines.append(f"{_toml_quote(key)} = {_toml_quote(runtime_env[key])}")
    return "\n".join(lines) + "\n"


def _pre_agent_setup_command(pre_agent_setup: list[str] | None) -> str:
    commands = [cmd.strip() for cmd in (pre_agent_setup or []) if cmd and cmd.strip()]
    if not commands:
        return ""
    script = "set -euo pipefail\n" + "\n".join(commands)
    return "bash -lc " + shlex.quote(script)


def _pre_agent_setup_healthcheck_toml_block(pre_agent_setup: list[str] | None) -> str:
    command = _pre_agent_setup_command(pre_agent_setup)
    if not command:
        return ""
    return (
        "\n[environment.healthcheck]\n"
        f"command = {_toml_quote(command)}\n"
        "interval_sec = 5.0\n"
        "timeout_sec = 120.0\n"
        "retries = 1\n"
    )


def _write_entry_json(
    task_dir: Path,
    entry: dict[str, Any],
    has_skill: bool,
    *,
    workspace_mode: str = "isolated",
    workspace_skill_names: list[str] | None = None,
    grading_mode: str = "default",
    custom_grader: bool = False,
) -> None:
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    entry_with_flag = {
        **entry,
        "has_skill": has_skill,
        "skill_workspace_mode": workspace_mode,
        "workspace_skill_names": workspace_skill_names or [],
        "grading_mode": grading_mode,
        "custom_grader": custom_grader,
    }
    (tests_dir / "entry.json").write_text(json.dumps(entry_with_flag, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_test_sh(task_dir: Path, *, grading_mode: str, custom_grader: bool) -> None:
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    prefix = '#!/bin/bash\nset -euo pipefail\ntests_dir="${HARBOR_TESTS_DIR:-/tests}"\n'
    if grading_mode == "custom_only":
        script = prefix + 'python3 "${tests_dir}/custom_grader_runner.py" --mode custom_only\n'
    elif grading_mode == "default_plus_custom" and custom_grader:
        script = (
            prefix
            + 'python3 "${tests_dir}/eval.py"\n'
            + 'python3 "${tests_dir}/custom_grader_runner.py" --mode default_plus_custom\n'
        )
    else:
        script = prefix + 'python3 "${tests_dir}/eval.py"\n'
    test_sh = tests_dir / "test.sh"
    test_sh.write_text(script, encoding="utf-8")
    test_sh.chmod(0o755)


def _copy_verifier(task_dir: Path) -> None:
    """Copy the standalone eval.py verifier into the task's tests/ directory."""
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    src = TEMPLATES_DIR / "eval.py"
    if src.exists():
        shutil.copy2(src, tests_dir / "eval.py")
    else:
        logger.warning("Verifier template not found at %s", src)
    lc = _EVAL_CORE_DIR / "log_converters.py"
    if lc.exists():
        shutil.copy2(lc, tests_dir / "log_converters.py")
    else:
        logger.warning("log_converters helper not found at %s", lc)


def _has_symlink_component(path: Path, root: Path) -> bool:
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _copy_custom_grader(task_dir: Path, skill_path: Path, grading_mode: str) -> bool:
    """Copy user custom grader support into a generated task, if configured."""
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    runner_src = TEMPLATES_DIR / "custom_grader_runner.py"
    if runner_src.exists():
        shutil.copy2(runner_src, tests_dir / "custom_grader_runner.py")

    grader_candidates = [
        (skill_path / "evals" / "grader.py", tests_dir / "grader.py"),
        (skill_path / "evals" / "grader.sh", tests_dir / "grader.sh"),
        (skill_path / "evals" / "tests" / "grader.py", tests_dir / "grader.py"),
        (skill_path / "evals" / "tests" / "grader.sh", tests_dir / "grader.sh"),
    ]
    for grader, destination in grader_candidates:
        if _has_symlink_component(grader, skill_path):
            raise ValueError(f"custom grader must be a non-symlinked regular file contained under evals/: {grader}")
        if not grader.exists():
            continue
        if not grader.is_file():
            raise ValueError(f"custom grader must be a non-symlinked regular file contained under evals/: {grader}")

        resolved_skill = skill_path.resolve(strict=True)
        resolved_evals = (skill_path / "evals").resolve(strict=True)
        resolved_grader = grader.resolve(strict=True)
        if not resolved_evals.is_relative_to(resolved_skill) or not resolved_grader.is_relative_to(resolved_evals):
            raise ValueError(f"custom grader must be a non-symlinked regular file contained under evals/: {grader}")

        shutil.copy2(resolved_grader, destination, follow_symlinks=False)
        if destination.suffix == ".sh":
            destination.chmod(0o755)
        return True

    if grading_mode == "custom_only":
        raise FileNotFoundError("grading.mode=custom_only requires evals/grader.py or evals/grader.sh")
    return False


def _collect_txt_deps(skills_dir: Path, filename: str) -> list[str]:
    """Collect non-comment lines from all instances of ``filename`` inside skill dirs."""
    deps: list[str] = []
    for dep_file in skills_dir.rglob(filename):
        for line in dep_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                deps.append(line)
    return sorted(set(deps))


def _validate_custom_dockerfile(path: Path) -> str | None:
    """Basic validation. Returns error message or None if OK."""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        return f"Cannot read Dockerfile: {e}"
    if path.stat().st_size > 20_000:
        return "Dockerfile exceeds 20KB limit"
    lines = [ln.strip() for ln in content.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines or not lines[0].upper().startswith("FROM"):
        return "Dockerfile must start with a FROM instruction"
    return None


def _compose_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _compose_strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in _compose_strings(child)]
    return []


def _validate_compose_interpolation(content: dict[str, Any], allowed_env: set[str]) -> None:
    referenced = {
        match.group(1) or match.group(2)
        for value in _compose_strings(content)
        for match in _COMPOSE_ENV_RE.finditer(value)
    }
    undeclared = sorted(referenced - allowed_env)
    if undeclared:
        raise ValueError(
            f"Custom Docker Compose uses undeclared interpolation variables: {', '.join(undeclared)}; "
            "declare each variable in harbor.runtime_env"
        )


def _is_relative_compose_path(value: object, root: Path) -> bool:
    if not isinstance(value, str) or not value or "$" in value:
        return False
    if (
        Path(value).is_absolute()
        or _COMPOSE_WINDOWS_PATH_RE.match(value)
        or _COMPOSE_SSH_PATH_RE.match(value)
        or value.startswith(("\\", "~"))
        or "://" in value
    ):
        return False
    parts = Path(value.replace("\\", "/")).parts
    if ".." in parts:
        return False
    return _is_relative_to(root / value, root)


def _validate_compose_host_passthrough(value: object, *, allowed_env: set[str], field: str) -> None:
    if value is None:
        return
    if isinstance(value, dict):
        passthrough = {str(name) for name, configured in value.items() if configured is None}
    elif isinstance(value, list):
        if any(not isinstance(item, str) for item in value):
            raise ValueError(f"Custom Docker Compose {field} list entries must be strings")
        passthrough = {item for item in value if "=" not in item}
    else:
        raise ValueError(f"Custom Docker Compose {field} must be a mapping or list")

    undeclared = sorted(passthrough - allowed_env)
    if undeclared:
        raise ValueError(
            f"Custom Docker Compose {field} passes undeclared host variables: {', '.join(undeclared)}; "
            "declare each variable in harbor.runtime_env"
        )


def _validate_compose_build(
    service_name: str,
    build: object,
    environment_dir: Path,
    *,
    allowed_env: set[str],
) -> None:
    if isinstance(build, str):
        if not _is_relative_compose_path(build, environment_dir):
            raise ValueError(f"Custom Docker Compose service '{service_name}' has an unsafe build context")
        return
    if not isinstance(build, dict):
        raise ValueError(f"Custom Docker Compose service '{service_name}' build must be a path or mapping")

    unsupported_keys = sorted(set(build) - _COMPOSE_ALLOWED_BUILD_KEYS)
    if unsupported_keys:
        raise ValueError(
            f"Custom Docker Compose service '{service_name}' build cannot set: {', '.join(unsupported_keys)}"
        )

    context = build.get("context", ".")
    if not _is_relative_compose_path(context, environment_dir):
        raise ValueError(f"Custom Docker Compose service '{service_name}' has an unsafe build context")
    context_dir = environment_dir / str(context)

    dockerfile = build.get("dockerfile")
    if dockerfile is not None and not _is_relative_compose_path(dockerfile, context_dir):
        raise ValueError(f"Custom Docker Compose service '{service_name}' has an unsafe build dockerfile")

    additional_contexts = build.get("additional_contexts", {})
    if isinstance(additional_contexts, list):
        values = [
            item.split("=", 1)[1] if isinstance(item, str) and "=" in item else None for item in additional_contexts
        ]
    elif isinstance(additional_contexts, dict):
        values = list(additional_contexts.values())
    else:
        raise ValueError(
            f"Custom Docker Compose service '{service_name}' build additional_contexts must be a mapping or list"
        )
    if any(not _is_relative_compose_path(value, environment_dir) for value in values):
        raise ValueError(f"Custom Docker Compose service '{service_name}' has unsafe build additional_contexts")
    _validate_compose_host_passthrough(
        build.get("args"),
        allowed_env=allowed_env,
        field=f"service '{service_name}' build.args",
    )


def _is_host_bind_volume(volume: object) -> bool:
    if isinstance(volume, dict):
        if any("$" in value for value in _compose_strings(volume)):
            return True
        volume_type = volume.get("type")
        if volume_type == "bind":
            return True
        if volume_type == "tmpfs":
            return "source" in volume
        if volume_type != "volume":
            return True
        source = volume.get("source")
        return source is not None and (not isinstance(source, str) or not _COMPOSE_NAMED_VOLUME_RE.fullmatch(source))
    if not isinstance(volume, str):
        return True
    if "$" in volume:
        return True
    if ":" not in volume:
        return False
    if _COMPOSE_WINDOWS_PATH_RE.match(volume):
        return True
    source = volume.split(":", 1)[0]
    return not bool(_COMPOSE_NAMED_VOLUME_RE.fullmatch(source))


def _validate_and_sanitize_custom_compose(
    compose_path: Path,
    *,
    allowed_env: set[str],
) -> None:
    """Reject Docker-host escape features and remove sidecar host ports.

    When Harbor runs multiple trials concurrently, each gets its own compose
    project.  Fixed host port mappings (e.g. ``"5432:5432"``) cause all but the
    first trial to fail with a port-already-in-use error.  Sidecar services
    don't need host ports — the main container reaches them via the compose
    network hostname (e.g. ``postgres:5432``).
    """
    import yaml

    try:
        content = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Custom Docker Compose file cannot be read safely: {exc}") from exc
    if not isinstance(content, dict):
        raise ValueError("Custom Docker Compose file must contain a top-level mapping")

    unsupported_top_level = sorted(set(content) - _COMPOSE_ALLOWED_TOP_LEVEL_KEYS)
    if unsupported_top_level:
        raise ValueError(f"Custom Docker Compose top-level cannot set: {', '.join(unsupported_top_level)}")

    services = content.get("services")
    if not isinstance(services, dict) or not services:
        raise ValueError("Custom Docker Compose services must be a non-empty mapping")

    changed = False
    for svc_name, service in services.items():
        if not isinstance(svc_name, str) or not isinstance(service, dict):
            raise ValueError("Custom Docker Compose services must map names to service mappings")

        volumes = service.get("volumes", [])
        if not isinstance(volumes, list):
            raise ValueError(f"Custom Docker Compose service '{svc_name}' volumes must be a list")
        if any(_is_host_bind_volume(volume) for volume in volumes):
            raise ValueError(f"Custom Docker Compose service '{svc_name}' cannot use a host bind mount")

        allowed_service_keys = _COMPOSE_MAIN_ALLOWED_KEYS if svc_name == "main" else _COMPOSE_SIDECAR_ALLOWED_KEYS
        unsupported_keys = sorted(set(service) - allowed_service_keys)
        if unsupported_keys:
            if svc_name == "main":
                raise ValueError(
                    "Custom Docker Compose service 'main' may set only depends_on; "
                    f"unsupported: {', '.join(unsupported_keys)}"
                )
            raise ValueError(f"Custom Docker Compose service '{svc_name}' cannot set: {', '.join(unsupported_keys)}")
        if "build" in service:
            _validate_compose_build(
                svc_name,
                service["build"],
                compose_path.parent,
                allowed_env=allowed_env,
            )
        _validate_compose_host_passthrough(
            service.get("environment"),
            allowed_env=allowed_env,
            field=f"service '{svc_name}' environment",
        )

        if svc_name != "main" and "ports" in service:
            del service["ports"]
            changed = True
            logger.debug("Stripped host port mapping from sidecar service '%s'", svc_name)

    volumes = content.get("volumes", {})
    if not isinstance(volumes, dict):
        raise ValueError("Custom Docker Compose top-level volumes must be a mapping")
    for volume_name, volume in volumes.items():
        if volume is None:
            continue
        if not isinstance(volume, dict):
            raise ValueError(f"Custom Docker Compose volume '{volume_name}' must be a mapping")
        unsupported_keys = sorted(set(volume) - _COMPOSE_ALLOWED_VOLUME_KEYS)
        if unsupported_keys:
            raise ValueError(f"Custom Docker Compose volume '{volume_name}' cannot set: {', '.join(unsupported_keys)}")

    networks = content.get("networks", {})
    if not isinstance(networks, dict):
        raise ValueError("Custom Docker Compose top-level networks must be a mapping")
    for network_name, network in networks.items():
        if network is None:
            continue
        if not isinstance(network, dict):
            raise ValueError(f"Custom Docker Compose network '{network_name}' must be a mapping")
        unsupported_keys = sorted(set(network) - _COMPOSE_ALLOWED_NETWORK_KEYS)
        if unsupported_keys:
            raise ValueError(
                f"Custom Docker Compose network '{network_name}' cannot set: {', '.join(unsupported_keys)}"
            )

    _validate_compose_interpolation(content, allowed_env)

    if changed:
        compose_path.write_text(yaml.dump(content, default_flow_style=False), encoding="utf-8")


_VERIFIER_DEPS = "ragas~=0.4.0 openai>=1.0 anthropic>=0.40 boto3>=1.34"

_WORKSPACE_SKILL_PATH = "COPY skills/ /workspace/skills/"
_AGENT_SKILL_PATHS = [
    "COPY skills/ /root/.claude/skills/",
    "COPY skills/ /root/.agents/skills/",
    "COPY skills/ /root/.config/opencode/skills/",
]


def _runtime_copy_lines(
    agent_config_lines: list[str],
    include_input: bool,
    include_repo: bool = False,
    include_repo_linked_root: bool = False,
) -> list[str]:
    lines = [_WORKSPACE_SKILL_PATH, *_AGENT_SKILL_PATHS, *agent_config_lines]
    if include_input:
        lines.append("COPY input/ /workspace/input/")
    if include_repo:
        lines.append("COPY repo/ /workspace/repo/")
    if include_repo_linked_root:
        lines.append("COPY repo-linked-root/ /workspace/")
    return lines


def _append_missing_lines(content: str, lines: list[str]) -> str:
    additions = ""
    for line in lines:
        if not line:
            if additions and not additions.endswith("\n\n"):
                additions += "\n"
            continue
        if line not in content:
            additions += f"{line}\n"
    if not additions:
        return content
    separator = "" if content.endswith("\n") else "\n"
    return content + separator + additions


def _write_agent_configs(env_dir: Path) -> list[str]:
    """Use Harbor's agent integrations and provider-native environment variables."""
    _ = env_dir
    return []


def _rebase_custom_dockerfile_content(
    content: str,
    base_image: str,
    *,
    agent_config_lines: list[str],
    include_input: bool,
    include_repo: bool = False,
    include_repo_linked_root: bool = False,
) -> tuple[str, str] | None:
    """Return custom Dockerfile content layered on top of the eval base image.

    The base image already contains: python:3.12-slim, system packages
    (bash/curl/git/jq), verifier dependencies, and the
    standard directory structure (/workspace/skills, /logs, etc.).

    The custom Dockerfile's FROM line is replaced so the skill author's
    additions (extra apt/pip packages, COPY, RUN) layer on top.  Multi-agent
    skill discovery paths are appended if not already present.
    Returns the rebased content and original ``FROM`` instruction, or ``None``
    when the content has no ``FROM`` instruction.
    """
    lines = content.splitlines(keepends=True)
    rebased: list[str] = []
    from_replaced = False
    original_from = ""
    for line in lines:
        stripped = line.strip()
        if not from_replaced and stripped.upper().startswith("FROM "):
            rebased.append(f"FROM {base_image}\n")
            rebased.append(f"# SkillEvaluator: original base was {stripped}\n")
            from_replaced = True
            original_from = stripped
        else:
            rebased.append(line)

    if not from_replaced:
        return None

    rebased_content = _append_missing_lines(
        "".join(rebased),
        _runtime_copy_lines(
            agent_config_lines,
            include_input,
            include_repo=include_repo,
            include_repo_linked_root=include_repo_linked_root,
        ),
    )
    return rebased_content, original_from


def _ensure_verifier_deps(
    dockerfile_path: Path,
    *,
    agent_config_lines: list[str],
    include_input: bool,
    include_repo: bool = False,
    include_repo_linked_root: bool = False,
) -> None:
    """Append verifier deps and runtime COPY lines to a custom Dockerfile."""
    content = dockerfile_path.read_text(encoding="utf-8")
    additions = ""
    if not ("ragas" in content and "anthropic" in content):
        additions += f"\nRUN pip install --no-cache-dir {_VERIFIER_DEPS}\n"
    updated = _append_missing_lines(
        content + additions,
        _runtime_copy_lines(
            agent_config_lines,
            include_input,
            include_repo=include_repo,
            include_repo_linked_root=include_repo_linked_root,
        ),
    )
    if updated == content:
        return
    dockerfile_path.write_text(updated, encoding="utf-8")
    logger.debug("Appended verifier deps + agent paths to %s", dockerfile_path)


def _entry_file_refs(entry: dict[str, Any]) -> list[str]:
    raw_files = entry.get("files")
    if raw_files is None:
        return []
    if isinstance(raw_files, str):
        raw_items = [raw_files]
    elif isinstance(raw_files, list):
        raw_items = raw_files
    else:
        raise ValueError(f"evals.json entry '{entry.get('id', '<unknown>')}' files must be a string or list of strings")

    refs: list[str] = []
    for idx, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, str):
            raise ValueError(f"evals.json entry '{entry.get('id', '<unknown>')}' files[{idx}] must be a string")
        ref = raw_item.strip()
        if ref:
            refs.append(ref)
    return refs


def _resolve_entry_file_ref(
    ref: str,
    *,
    skill_path: Path,
    evals_dir: Path,
    input_files_dir: Path | None,
) -> tuple[Path, Path]:
    if "\x00" in ref:
        raise ValueError("evals.json files entries cannot contain NUL bytes")

    ref_path = Path(ref)
    if ref_path.is_absolute():
        raise ValueError(f"evals.json files entry must be relative to evals/: {ref}")
    if _LOCAL_LINK_SCHEME_RE.match(ref):
        raise ValueError(f"evals.json files entry uses unsupported URI scheme: {ref}")

    if ref_path.parts and ref_path.parts[0] == "evals":
        candidates = [skill_path / ref_path]
    else:
        candidates = [evals_dir / ref_path, skill_path / ref_path]

    source = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
    if source is None:
        raise FileNotFoundError(f"evals.json files entry does not exist: {ref}")

    resolved_evals_dir = evals_dir.resolve()
    if source == resolved_evals_dir or not _is_relative_to(source, resolved_evals_dir):
        raise ValueError(f"evals.json files entry resolves outside evals/: {ref}")

    resolved_input_files_dir = input_files_dir.resolve() if input_files_dir and input_files_dir.exists() else None
    if resolved_input_files_dir and _is_relative_to(source, resolved_input_files_dir):
        rel = source.relative_to(resolved_input_files_dir)
    else:
        rel = source.relative_to(resolved_evals_dir)

    return source, rel


def _copy_input_ref(source: Path, input_dir: Path, rel: Path) -> None:
    dest = input_dir if rel == Path() else input_dir / rel
    if source.is_dir():
        if dest.exists() and dest.is_file():
            dest.unlink()
        copytree_secure(source, dest, dirs_exist_ok=True)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def _stage_task_inputs(
    env_dir: Path,
    *,
    input_files_dir: Path | None,
    entry: dict[str, Any],
    source_skill_path: Path,
    evals_dir: Path,
) -> bool:
    input_dir = env_dir / "input"
    if input_files_dir and input_files_dir.exists():
        if input_dir.exists():
            shutil.rmtree(input_dir)
        copytree_secure(input_files_dir, input_dir, dirs_exist_ok=True)

    for ref in _entry_file_refs(entry):
        source, rel = _resolve_entry_file_ref(
            ref,
            skill_path=source_skill_path,
            evals_dir=evals_dir,
            input_files_dir=input_files_dir,
        )
        input_dir.mkdir(parents=True, exist_ok=True)
        _copy_input_ref(source, input_dir, rel)

    return input_dir.exists()


def _write_dockerfile(
    task_dir: Path,
    skill_path: Path | None,
    reference_skills_dir: Path | None,
    workspace_skill_paths: list[Path] | None,
    has_skill: bool,
    input_files_dir: Path | None = None,
    entry: dict[str, Any] | None = None,
    evals_dir: Path | None = None,
    exclude_skill_name: str | None = None,
    base_image: str = "",
    custom_dockerfile_mode: str = "rebase",
    repo_context_skill_path: Path | None = None,
    repo_context_mode: str = "linked",
    compose_env_names: set[str] | None = None,
) -> None:
    """Generate a Dockerfile that installs skills into the container.

    Environment resolution order:
      1. ``skill_path/evals/environment/Dockerfile`` -- developer's custom Dockerfile
      2. Pre-built base image (when *base_image* is set) -- only COPY layers
      3. ``scripts/requirements.txt`` + ``scripts/apt-packages.txt`` -- auto-detected deps
      4. Default generic Dockerfile
    """
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)

    skills_dir = env_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    def _ignore_results(directory, contents):
        return [c for c in contents if c in ("results", "__pycache__", ".git")]

    if has_skill and skill_path and skill_path.exists():
        dest = skills_dir / skill_path.name
        if dest.exists():
            shutil.rmtree(dest)
        copytree_secure(skill_path, dest, dirs_exist_ok=True, ignore=_ignore_results)

    if reference_skills_dir and reference_skills_dir.exists():
        for ref_skill in reference_skills_dir.iterdir():
            if ref_skill.is_dir() and not ref_skill.name.startswith("."):
                if not has_skill and exclude_skill_name and ref_skill.name == exclude_skill_name:
                    continue
                dest = skills_dir / ref_skill.name
                if not dest.exists():
                    copytree_secure(ref_skill, dest, dirs_exist_ok=True, ignore=_ignore_results)

    for workspace_skill in workspace_skill_paths or []:
        if not workspace_skill.exists() or not workspace_skill.is_dir():
            continue
        if not has_skill and exclude_skill_name and workspace_skill.name == exclude_skill_name:
            continue
        dest = skills_dir / workspace_skill.name
        if dest.exists():
            continue
        copytree_secure(workspace_skill, dest, dirs_exist_ok=True, ignore=_ignore_results)

    include_input = False
    if entry is not None and skill_path is not None and evals_dir is not None:
        include_input = _stage_task_inputs(
            env_dir,
            input_files_dir=input_files_dir,
            entry=entry,
            source_skill_path=skill_path,
            evals_dir=evals_dir,
        )

    effective_repo_context_mode = "full" if repo_context_mode == "full" else ("linked" if has_skill else "none")
    _stage_repo_context(
        env_dir,
        source_skill_path=repo_context_skill_path,
        mode=effective_repo_context_mode,
        exclude_source_skill=repo_context_mode == "full" and not has_skill,
    )

    agent_config_lines = _write_agent_configs(env_dir)
    include_repo = (env_dir / "repo").exists()
    include_repo_linked_root = (env_dir / "repo-linked-root").exists()

    custom_env_dir = skill_path / "evals" / "environment" if skill_path else None

    if custom_env_dir and custom_env_dir.exists():
        staged_custom_env = env_dir / ".skillevaluator-custom-environment"
        copytree_secure(custom_env_dir, staged_custom_env, allowed_root=skill_path)
        try:
            if not has_skill and skill_path:
                _check_custom_environment_does_not_stage_target(staged_custom_env, skill_path.name)

            custom_dockerfile = staged_custom_env / "Dockerfile"
            custom_dockerfile_accepted = False
            if custom_dockerfile.exists():
                err = _validate_custom_dockerfile(custom_dockerfile)
                if err:
                    logger.warning("Custom Dockerfile rejected (%s): %s", custom_env_dir / "Dockerfile", err)
                else:
                    shutil.copy2(custom_dockerfile, env_dir / "Dockerfile")
                    custom_dockerfile_accepted = True
                    logger.debug("Using custom Dockerfile from %s", custom_env_dir / "Dockerfile")

            compose_file = staged_custom_env / "docker-compose.yaml"
            if not compose_file.exists():
                compose_file = staged_custom_env / "docker-compose.yml"
            if compose_file.exists():
                shutil.copy2(compose_file, env_dir / "docker-compose.yaml")
                _validate_and_sanitize_custom_compose(
                    env_dir / "docker-compose.yaml",
                    allowed_env=compose_env_names or set(),
                )
                logger.debug("Copied docker-compose.yaml from %s", custom_env_dir / compose_file.name)

            for subdir in staged_custom_env.iterdir():
                if subdir.is_dir() and subdir.name not in ("__pycache__", ".git"):
                    dest = env_dir / subdir.name
                    if not dest.exists():
                        copytree_secure(subdir, dest, allowed_root=staged_custom_env)
                        logger.debug("Copied sidecar dir %s", subdir.name)

            if custom_dockerfile_accepted:
                if base_image and custom_dockerfile_mode == "rebase":
                    dockerfile_path = env_dir / "Dockerfile"
                    rebased = _rebase_custom_dockerfile_content(
                        dockerfile_path.read_text(encoding="utf-8"),
                        base_image,
                        agent_config_lines=agent_config_lines,
                        include_input=include_input,
                        include_repo=include_repo,
                        include_repo_linked_root=include_repo_linked_root,
                    )
                    if rebased is not None:
                        rebased_content, original_from = rebased
                        dockerfile_path.write_text(rebased_content, encoding="utf-8")
                        logger.warning(
                            "Rebased custom Dockerfile from '%s' onto '%s'",
                            original_from,
                            base_image,
                        )
                else:
                    _ensure_verifier_deps(
                        env_dir / "Dockerfile",
                        agent_config_lines=agent_config_lines,
                        include_input=include_input,
                        include_repo=include_repo,
                        include_repo_linked_root=include_repo_linked_root,
                    )
                return
        finally:
            shutil.rmtree(staged_custom_env, ignore_errors=True)

    if base_image:
        dockerfile_lines = [
            f"FROM {base_image}",
            "",
            "COPY skills/ /workspace/skills/",
            "",
            "COPY skills/ /root/.claude/skills/",
            "COPY skills/ /root/.agents/skills/",
            "COPY skills/ /root/.config/opencode/skills/",
        ]
        dockerfile_lines.extend(agent_config_lines)

        if include_input:
            dockerfile_lines.append("COPY input/ /workspace/input/")
        if include_repo:
            dockerfile_lines.append("COPY repo/ /workspace/repo/")
        if include_repo_linked_root:
            dockerfile_lines.append("COPY repo-linked-root/ /workspace/")

        (env_dir / "Dockerfile").write_text("\n".join(dockerfile_lines) + "\n", encoding="utf-8")
        return

    pip_reqs = _collect_txt_deps(skills_dir, "requirements.txt")
    apt_pkgs = _collect_txt_deps(skills_dir, "apt-packages.txt")

    dockerfile_lines = [
        "FROM python:3.12-slim",
        "",
        "RUN apt-get -o Acquire::Retries=3 update && \\",
        "    apt-get -o Acquire::Retries=3 install -y --no-install-recommends \\",
        "    bash curl git jq ripgrep \\",
    ]
    if apt_pkgs:
        dockerfile_lines[-1] += " \\"
        dockerfile_lines.append("    " + " ".join(apt_pkgs) + " \\")
    dockerfile_lines.append("    && rm -rf /var/lib/apt/lists/*")

    dockerfile_lines.extend(
        [
            "",
            "RUN pip install --no-cache-dir \\",
            "    ragas~=0.4.0 \\",
            "    openai>=1.0 \\",
            "    anthropic>=0.40 \\",
            "    boto3>=1.34",
        ]
    )

    if pip_reqs:
        escaped = " ".join(f'"{r}"' for r in pip_reqs)
        dockerfile_lines.append(f"RUN pip install --no-cache-dir {escaped}")

    dockerfile_lines.extend(
        [
            "",
            "RUN mkdir -p /workspace/skills /workspace/input /workspace/output \\",
            "    /logs/verifier /logs/agent",
            "",
            "COPY skills/ /workspace/skills/",
            "",
            "# Multi-agent skill discovery paths",
            "COPY skills/ /root/.claude/skills/",
            "COPY skills/ /root/.agents/skills/",
            "COPY skills/ /root/.config/opencode/skills/",
        ]
    )
    dockerfile_lines.extend(agent_config_lines)

    if input_files_dir and input_files_dir.exists():
        dockerfile_lines.append("COPY input/ /workspace/input/")
    if include_repo:
        dockerfile_lines.append("COPY repo/ /workspace/repo/")
    if include_repo_linked_root:
        dockerfile_lines.append("COPY repo-linked-root/ /workspace/")

    dockerfile_lines.extend(
        [
            "",
            "WORKDIR /workspace",
        ]
    )

    (env_dir / "Dockerfile").write_text("\n".join(dockerfile_lines) + "\n", encoding="utf-8")


def _copy_skill_dirs(
    *,
    env_dir: Path,
    skill_path: Path | None,
    reference_skills_dir: Path | None,
    workspace_skill_paths: list[Path] | None,
    has_skill: bool,
    exclude_skill_name: str | None,
) -> None:
    """Stage target/workspace skills into a task environment."""
    skills_dir = env_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    def _ignore_results(directory, contents):
        return [c for c in contents if c in ("results", "__pycache__", ".git")]

    if has_skill and skill_path and skill_path.exists():
        dest = skills_dir / skill_path.name
        if dest.exists():
            shutil.rmtree(dest)
        copytree_secure(skill_path, dest, dirs_exist_ok=True, ignore=_ignore_results)

    if reference_skills_dir and reference_skills_dir.exists():
        for ref_skill in reference_skills_dir.iterdir():
            if ref_skill.is_dir() and not ref_skill.name.startswith("."):
                if not has_skill and exclude_skill_name and ref_skill.name == exclude_skill_name:
                    continue
                dest = skills_dir / ref_skill.name
                if not dest.exists():
                    copytree_secure(ref_skill, dest, dirs_exist_ok=True, ignore=_ignore_results)

    for workspace_skill in workspace_skill_paths or []:
        if not workspace_skill.exists() or not workspace_skill.is_dir():
            continue
        if not has_skill and exclude_skill_name and workspace_skill.name == exclude_skill_name:
            continue
        dest = skills_dir / workspace_skill.name
        if dest.exists():
            continue
        copytree_secure(workspace_skill, dest, dirs_exist_ok=True, ignore=_ignore_results)


def _write_default_environment_dockerfile(
    env_dir: Path,
    *,
    base_image: str,
    agent_config_lines: list[str],
    include_input: bool,
    include_repo: bool = False,
    include_repo_linked_root: bool = False,
    include_verifier_deps: bool = True,
) -> None:
    if base_image:
        dockerfile_lines = [
            f"FROM {base_image}",
            "",
            "COPY skills/ /workspace/skills/",
            "",
            "COPY skills/ /root/.claude/skills/",
            "COPY skills/ /root/.agents/skills/",
            "COPY skills/ /root/.config/opencode/skills/",
        ]
        dockerfile_lines.extend(agent_config_lines)
        if include_input:
            dockerfile_lines.append("COPY input/ /workspace/input/")
        if include_repo:
            dockerfile_lines.append("COPY repo/ /workspace/repo/")
        if include_repo_linked_root:
            dockerfile_lines.append("COPY repo-linked-root/ /workspace/")
        (env_dir / "Dockerfile").write_text("\n".join(dockerfile_lines) + "\n", encoding="utf-8")
        return

    pip_reqs = _collect_txt_deps(env_dir / "skills", "requirements.txt")
    apt_pkgs = _collect_txt_deps(env_dir / "skills", "apt-packages.txt")

    dockerfile_lines = [
        "FROM python:3.12-slim",
        "",
        "RUN apt-get -o Acquire::Retries=3 update && \\",
        "    apt-get -o Acquire::Retries=3 install -y --no-install-recommends \\",
        "    bash curl git jq ripgrep \\",
    ]
    if apt_pkgs:
        dockerfile_lines[-1] += " \\"
        dockerfile_lines.append("    " + " ".join(apt_pkgs) + " \\")
    dockerfile_lines.append("    && rm -rf /var/lib/apt/lists/*")
    if include_verifier_deps:
        dockerfile_lines.extend(
            [
                "",
                "RUN pip install --no-cache-dir \\",
                "    ragas~=0.4.0 \\",
                "    openai>=1.0 \\",
                "    anthropic>=0.40 \\",
                "    boto3>=1.34",
            ]
        )
    if pip_reqs:
        escaped = " ".join(f'"{r}"' for r in pip_reqs)
        dockerfile_lines.append(f"RUN pip install --no-cache-dir {escaped}")
    dockerfile_lines.extend(
        [
            "",
            "RUN mkdir -p /workspace/skills /workspace/input /workspace/output \\",
            "    /logs/verifier /logs/agent",
            "",
            "COPY skills/ /workspace/skills/",
            "",
            "# Multi-agent skill discovery paths",
            "COPY skills/ /root/.claude/skills/",
            "COPY skills/ /root/.agents/skills/",
            "COPY skills/ /root/.config/opencode/skills/",
        ]
    )
    dockerfile_lines.extend(agent_config_lines)
    if include_input:
        dockerfile_lines.append("COPY input/ /workspace/input/")
    if include_repo:
        dockerfile_lines.append("COPY repo/ /workspace/repo/")
    if include_repo_linked_root:
        dockerfile_lines.append("COPY repo-linked-root/ /workspace/")
    dockerfile_lines.extend(["", "WORKDIR /workspace"])
    (env_dir / "Dockerfile").write_text("\n".join(dockerfile_lines) + "\n", encoding="utf-8")


def _prepare_native_environment(
    task_dir: Path,
    *,
    skill_path: Path,
    reference_skills_dir: Path | None,
    workspace_skill_paths: list[Path] | None,
    has_skill: bool,
    base_image: str,
    custom_dockerfile_mode: str,
    grading_mode: str,
    repo_context_mode: str = "linked",
    compose_env_names: set[str] | None = None,
) -> None:
    """Stage SkillEvaluator runtime additions into a copied native Harbor task."""
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    _copy_skill_dirs(
        env_dir=env_dir,
        skill_path=skill_path if has_skill else None,
        reference_skills_dir=reference_skills_dir,
        workspace_skill_paths=workspace_skill_paths,
        has_skill=has_skill,
        exclude_skill_name=skill_path.name,
    )
    agent_config_lines = _write_agent_configs(env_dir)
    effective_repo_context_mode = "full" if repo_context_mode == "full" else ("linked" if has_skill else "none")
    _stage_repo_context(
        env_dir,
        source_skill_path=skill_path,
        mode=effective_repo_context_mode,
        exclude_source_skill=repo_context_mode == "full" and not has_skill,
    )
    include_input = (env_dir / "input").exists()
    include_repo = (env_dir / "repo").exists()
    include_repo_linked_root = (env_dir / "repo-linked-root").exists()
    dockerfile_path = env_dir / "Dockerfile"
    compose_path = env_dir / "docker-compose.yaml"
    if compose_path.exists():
        _validate_and_sanitize_custom_compose(compose_path, allowed_env=compose_env_names or set())

    if dockerfile_path.exists():
        err = _validate_custom_dockerfile(dockerfile_path)
        if err:
            raise ValueError(f"{dockerfile_path}: {err}")
        if base_image and custom_dockerfile_mode == "rebase":
            rebased = _rebase_custom_dockerfile_content(
                dockerfile_path.read_text(encoding="utf-8"),
                base_image,
                agent_config_lines=agent_config_lines,
                include_input=include_input,
                include_repo=include_repo,
                include_repo_linked_root=include_repo_linked_root,
            )
            if rebased is not None:
                rebased_content, original_from = rebased
                dockerfile_path.write_text(rebased_content, encoding="utf-8")
                logger.warning(
                    "Rebased custom Dockerfile from '%s' onto '%s'",
                    original_from,
                    base_image,
                )
        elif grading_mode == "custom_only":
            content = dockerfile_path.read_text(encoding="utf-8")
            updated = _append_missing_lines(
                content,
                _runtime_copy_lines(
                    agent_config_lines,
                    include_input,
                    include_repo=include_repo,
                    include_repo_linked_root=include_repo_linked_root,
                ),
            )
            if updated != content:
                dockerfile_path.write_text(updated, encoding="utf-8")
        else:
            _ensure_verifier_deps(
                dockerfile_path,
                agent_config_lines=agent_config_lines,
                include_input=include_input,
                include_repo=include_repo,
                include_repo_linked_root=include_repo_linked_root,
            )
        return

    _write_default_environment_dockerfile(
        env_dir,
        base_image=base_image,
        agent_config_lines=agent_config_lines,
        include_input=include_input,
        include_repo=include_repo,
        include_repo_linked_root=include_repo_linked_root,
        include_verifier_deps=grading_mode != "custom_only",
    )


def _native_task_dirs(dataset_dir: Path) -> list[Path]:
    """Return native Harbor task directories from a copied dataset."""
    return [
        p
        for p in sorted(dataset_dir.iterdir())
        if p.is_dir() and not p.name.startswith((".", "_")) and (p / "task.toml").exists()
    ]


def _native_entry_id(task_dir: Path) -> str:
    task_toml = task_dir / "task.toml"
    try:
        import tomllib

        data = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    except Exception:
        return task_dir.name
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    if isinstance(metadata, dict) and metadata.get("entry_id"):
        return str(metadata["entry_id"])
    return task_dir.name


def _ensure_skill_evaluator_verifier_env(task_dir: Path, *, verifier_env: dict[str, str] | None) -> None:
    """Ensure staged native tasks forward configured public provider variables."""
    task_toml = task_dir / "task.toml"
    content = task_toml.read_text(encoding="utf-8")
    env_lines = [f'{name} = "${{{name}}}"' for name in _verifier_env_vars(verifier_env)]
    if all(line in content for line in env_lines):
        return

    lines = content.splitlines()
    if "[verifier.env]" in lines:
        idx = lines.index("[verifier.env]") + 1
        existing = set(lines)
        for line in reversed(env_lines):
            if line not in existing:
                lines.insert(idx, line)
        task_toml.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    env_block = ["[verifier.env]", *env_lines]
    if "[verifier]" in lines:
        start = lines.index("[verifier]") + 1
        insert_at = len(lines)
        for idx in range(start, len(lines)):
            line = lines[idx].strip()
            if line.startswith("[") and line.endswith("]"):
                insert_at = idx
                break
        lines[insert_at:insert_at] = ["", *env_block]
        task_toml.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    insert_at = lines.index("[environment]") if "[environment]" in lines else len(lines)
    lines[insert_at:insert_at] = ["[verifier]", "timeout_sec = 180.0", "", *env_block, ""]
    task_toml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _insert_table_block(lines: list[str], anchor: str, block: list[str]) -> None:
    """Insert a TOML table block after *anchor* and before the next table."""
    if anchor not in lines:
        lines.extend(["", anchor])
    start = lines.index(anchor) + 1
    insert_at = len(lines)
    for idx in range(start, len(lines)):
        stripped = lines[idx].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            insert_at = idx
            break
    prefix = [] if insert_at == 0 or (insert_at > 0 and lines[insert_at - 1] == "") else [""]
    lines[insert_at:insert_at] = [*prefix, *block]


def _ensure_environment_env(task_dir: Path, runtime_env: dict[str, str]) -> None:
    if not runtime_env:
        return

    task_toml = task_dir / "task.toml"
    lines = task_toml.read_text(encoding="utf-8").splitlines()
    header = "[environment.env]"
    rendered = {key: f"{_toml_quote(key)} = {_toml_quote(runtime_env[key])}" for key in sorted(runtime_env)}
    env_lines = list(rendered.values())
    if header in lines:
        idx = lines.index(header) + 1
        env_end = len(lines)
        for end_idx in range(idx, len(lines)):
            stripped = lines[end_idx].strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                env_end = end_idx
                break
        seen_keys: set[str] = set()
        updated_section: list[str] = []
        for line in lines[idx:env_end]:
            assignment = line.split("=", 1)[0].strip() if line.strip() and "=" in line else ""
            matching_key = next(
                (key for key in runtime_env if assignment in {key, _toml_quote(key)}),
                None,
            )
            if matching_key is not None:
                if matching_key in seen_keys:
                    continue
                updated_section.append(rendered[matching_key])
                seen_keys.add(matching_key)
            else:
                updated_section.append(line)
        for key, line in rendered.items():
            if key not in seen_keys:
                updated_section.append(line)
        lines[idx:env_end] = updated_section
        task_toml.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    block = [header, *env_lines]
    _insert_table_block(lines, "[environment]", block)
    task_toml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_pre_agent_setup_healthcheck(task_dir: Path, pre_agent_setup: list[str]) -> None:
    command = _pre_agent_setup_command(pre_agent_setup)
    if not command:
        return

    task_toml = task_dir / "task.toml"
    lines = task_toml.read_text(encoding="utf-8").splitlines()
    header = "[environment.healthcheck]"
    if header in lines:
        raise ValueError(
            f"{task_toml}: harbor.pre_agent_setup cannot be injected because "
            "the native Harbor task already defines [environment.healthcheck]"
        )

    block = [
        header,
        f"command = {_toml_quote(command)}",
        "interval_sec = 5.0",
        "timeout_sec = 120.0",
        "retries = 1",
    ]
    _insert_table_block(lines, "[environment.env]" if "[environment.env]" in lines else "[environment]", block)
    task_toml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_runtime_env_and_pre_agent_setup(
    task_dir: Path,
    *,
    runtime_env: dict[str, str] | None,
    pre_agent_setup: list[str] | None,
) -> None:
    _ensure_environment_env(task_dir, runtime_env or {})
    _ensure_pre_agent_setup_healthcheck(task_dir, pre_agent_setup or [])


def _load_entries_by_id(skill_path: Path) -> dict[str, dict[str, Any]]:
    evals_file = find_evals_file(skill_path)
    if not evals_file:
        return {}
    entries = _load_evals(evals_file)
    case_ids = validate_case_ids(entry.get("id") for entry in entries)
    return {case_id: {**entry, "id": case_id} for entry, case_id in zip(entries, case_ids, strict=True)}


def _dockerfile_copy_or_add_mentions_skill(line: str, target_skill_name: str) -> bool:
    stripped = line.strip()
    if not stripped.upper().startswith(("COPY ", "ADD ")):
        return False
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        tokens = stripped.split()
    path_tokens = [token for token in tokens[1:] if not token.startswith("--")]
    for token in path_tokens:
        normalized = re.sub(r"[\[\]\",:]+", "/", token).replace("\\", "/")
        if target_skill_name in [segment for segment in normalized.split("/") if segment]:
            return True
    return False


def _check_custom_environment_does_not_stage_target(custom_env_dir: Path, target_skill_name: str) -> None:
    for skill_md in custom_env_dir.rglob("SKILL.md"):
        if skill_md.parent.name == target_skill_name:
            raise ValueError(
                f"Custom eval environment already stages target skill '{target_skill_name}' at {skill_md}. "
                "SkillEvaluator must control with-skill and baseline staging so Skill Lift stays uncontaminated."
            )
    for dockerfile in custom_env_dir.rglob("Dockerfile"):
        try:
            lines = dockerfile.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if _dockerfile_copy_or_add_mentions_skill(line, target_skill_name):
                raise ValueError(
                    f"Custom eval Dockerfile appears to copy target skill '{target_skill_name}' in {dockerfile}. "
                    "Use SkillEvaluator skill staging instead so baseline stays uncontaminated."
                )


def _check_native_source_does_not_stage_target(native_dir: Path, target_skill_name: str) -> None:
    for skill_md in native_dir.rglob("SKILL.md"):
        if skill_md.parent.name == target_skill_name and "environment" in skill_md.parts:
            raise ValueError(
                f"BYOT source already stages target skill '{target_skill_name}' at {skill_md}. "
                "SkillEvaluator must control with-skill and baseline staging."
            )
    for dockerfile in native_dir.rglob("Dockerfile"):
        try:
            lines = dockerfile.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if _dockerfile_copy_or_add_mentions_skill(line, target_skill_name):
                raise ValueError(
                    f"BYOT Dockerfile appears to copy target skill '{target_skill_name}' in {dockerfile}. "
                    "Use SkillEvaluator workspace staging instead so baseline stays uncontaminated."
                )


def _stage_native_harbor_tasks_into(
    skill_path: Path,
    output_dir: Path,
    *,
    with_skill: bool = True,
    reference_skills_dir: Path | None = None,
    workspace_skill_paths: list[Path] | None = None,
    workspace_mode: str = "isolated",
    grading_mode: str = "default",
    base_image: str = "",
    custom_dockerfile_mode: str = "rebase",
    copy_repo: bool = False,
    runtime_env: dict[str, str] | None = None,
    verifier_env: dict[str, str] | None = None,
    pre_agent_setup: list[str] | None = None,
    task_resources: dict[str, int] | None = None,
    agent_workdir: str | None = None,
) -> list[Path]:
    """Build native Harbor tasks inside a private, caller-owned directory.

    The source tree is copied first and all SkillEvaluator injections happen only in the
    staged result directory.
    """
    native_dir = skill_path / "evals" / "harbor"
    if not native_dir.exists():
        raise FileNotFoundError(f"No native Harbor task source found at {native_dir}")
    _check_native_source_does_not_stage_target(native_dir, skill_path.name)
    _ = task_resources
    _ = agent_workdir
    entries_by_id = _load_entries_by_id(skill_path)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    copytree_secure(native_dir, output_dir, ignore=shutil.ignore_patterns("results", "__pycache__", ".git"))
    task_dirs = _native_task_dirs(output_dir)
    if not task_dirs:
        raise ValueError(f"No Harbor task directories with task.toml found in {native_dir}")

    if grading_mode in ("default", "default_plus_custom") and not entries_by_id:
        raise FileNotFoundError(
            "Native Harbor tasks with SkillEvaluator default grading require evals/evals.json metadata matching task IDs"
        )

    workspace_skill_names = sorted({p.name for p in workspace_skill_paths or []})
    for task_dir in task_dirs:
        entry_id = _native_entry_id(task_dir)
        entry = entries_by_id.get(entry_id)
        if grading_mode in ("default", "default_plus_custom") and entry is None:
            raise ValueError(f"Native Harbor task '{entry_id}' has no matching entry in evals/evals.json")

        _ensure_runtime_env_and_pre_agent_setup(
            task_dir,
            runtime_env=runtime_env,
            pre_agent_setup=pre_agent_setup,
        )

        tests_dir = task_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        _copy_verifier(task_dir)
        shutil.copy2(TEMPLATES_DIR / "custom_grader_runner.py", tests_dir / "custom_grader_runner.py")

        custom_grader = (tests_dir / "grader.py").exists() or (tests_dir / "grader.sh").exists()
        if (not custom_grader and grading_mode != "custom_only") or (
            not custom_grader and not (tests_dir / "test.sh").exists()
        ):
            custom_grader = _copy_custom_grader(task_dir, skill_path, grading_mode)

        if grading_mode in ("default", "default_plus_custom"):
            _ensure_skill_evaluator_verifier_env(
                task_dir,
                verifier_env=verifier_env if verifier_env is not None else runtime_env,
            )
            _write_entry_json(
                task_dir,
                entry or {"id": entry_id},
                with_skill,
                workspace_mode=workspace_mode,
                workspace_skill_names=workspace_skill_names,
                grading_mode=grading_mode,
                custom_grader=custom_grader,
            )
            _write_test_sh(task_dir, grading_mode=grading_mode, custom_grader=custom_grader)
        elif custom_grader:
            _write_entry_json(
                task_dir,
                entry or {"id": entry_id},
                with_skill,
                workspace_mode=workspace_mode,
                workspace_skill_names=workspace_skill_names,
                grading_mode=grading_mode,
                custom_grader=True,
            )
            _write_test_sh(task_dir, grading_mode=grading_mode, custom_grader=True)
        elif not (tests_dir / "test.sh").exists():
            raise FileNotFoundError(
                f"custom_only native Harbor task '{entry_id}' requires tests/grader.py or tests/test.sh"
            )

        _prepare_native_environment(
            task_dir,
            skill_path=skill_path,
            reference_skills_dir=reference_skills_dir,
            workspace_skill_paths=workspace_skill_paths,
            has_skill=with_skill,
            base_image=base_image,
            custom_dockerfile_mode=custom_dockerfile_mode,
            grading_mode=grading_mode,
            repo_context_mode="full" if copy_repo else "linked",
            compose_env_names=set(runtime_env or {}),
        )

    if not (output_dir / "dataset.toml").exists():
        _write_dataset_toml(output_dir, [p.name for p in task_dirs])
    _copy_metric_py(output_dir)
    return task_dirs


def stage_native_harbor_tasks(
    skill_path: Path,
    output_dir: Path,
    *,
    with_skill: bool = True,
    reference_skills_dir: Path | None = None,
    workspace_skill_paths: list[Path] | None = None,
    workspace_mode: str = "isolated",
    grading_mode: str = "default",
    base_image: str = "",
    custom_dockerfile_mode: str = "rebase",
    copy_repo: bool = False,
    runtime_env: dict[str, str] | None = None,
    verifier_env: dict[str, str] | None = None,
    pre_agent_setup: list[str] | None = None,
    task_resources: dict[str, int] | None = None,
    agent_workdir: str | None = None,
) -> list[Path]:
    """Stage native tasks privately, then publish one exact output snapshot."""

    with tempfile.TemporaryDirectory(prefix="skillevaluator-native-tasks-") as temporary:
        private_root = Path(temporary).resolve(strict=True)
        private_output = private_root / "dataset"
        private_tasks = _stage_native_harbor_tasks_into(
            skill_path,
            private_output,
            with_skill=with_skill,
            reference_skills_dir=reference_skills_dir,
            workspace_skill_paths=workspace_skill_paths,
            workspace_mode=workspace_mode,
            grading_mode=grading_mode,
            base_image=base_image,
            custom_dockerfile_mode=custom_dockerfile_mode,
            copy_repo=copy_repo,
            runtime_env=runtime_env,
            verifier_env=verifier_env,
            pre_agent_setup=pre_agent_setup,
            task_resources=task_resources,
            agent_workdir=agent_workdir,
        )
        relative_tasks = [task.relative_to(private_output) for task in private_tasks]
        copytree_secure(
            private_output,
            output_dir,
            replace_existing=True,
            allowed_root=private_root,
        )
    return [output_dir / relative for relative in relative_tasks]


def _write_dataset_toml(output_dir: Path, task_dirs: list[str]) -> None:
    """Generate a minimal dataset.toml for the Harbor dataset."""
    lines = [
        "[dataset]",
        'name = "nvidia/skillevaluator"',
        'description = "SkillEvaluator skill evaluation dataset"',
        "",
    ]
    for task_name in sorted(task_dirs):
        lines.append("[[tasks]]")
        lines.append(f"name = {_toml_quote(f'nvidia/{task_name}')}")
        lines.append("")

    (output_dir / "dataset.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_metric_py(output_dir: Path) -> None:
    """Copy the custom metric.py for Harbor dataset-level aggregation."""
    src = TEMPLATES_DIR / "metric.py"
    if src.exists():
        shutil.copy2(src, output_dir / "metric.py")


def _generate_harbor_tasks_into(
    skill_path: Path,
    output_dir: Path,
    *,
    with_skill: bool = True,
    reference_skills_dir: Path | None = None,
    workspace_skill_paths: list[Path] | None = None,
    workspace_mode: str = "isolated",
    grading_mode: str = "default",
    base_image: str = "",
    custom_dockerfile_mode: str = "rebase",
    copy_repo: bool = False,
    runtime_env: dict[str, str] | None = None,
    verifier_env: dict[str, str] | None = None,
    pre_agent_setup: list[str] | None = None,
    task_resources: dict[str, int] | None = None,
    agent_workdir: str | None = None,
) -> list[Path]:
    """Generate Harbor task directories inside a private output directory.

    Args:
        skill_path: Path to the skill directory (contains SKILL.md, scripts/, evals/)
        output_dir: Where to write the Harbor dataset
        with_skill: If True, install the skill in the container. If False, baseline run.
        reference_skills_dir: Optional parent directory of reference/decoy
            skills to stage. Prefer ``workspace_skill_paths`` for new code.
        workspace_skill_paths: Explicit sibling/custom skills to stage into
            the agent workspace for this dataset.
        workspace_mode: ``isolated`` or ``group``; stored in entry metadata for
            workspace-aware scoring.
        grading_mode: ``default``, ``default_plus_custom``, or ``custom_only``.
        base_image: Pre-built Docker image tag to use as base (skips heavy pip
            installs in per-task Dockerfiles).  Empty string = full build.
        custom_dockerfile_mode: ``rebase`` replaces a valid custom Dockerfile's
            FROM with the eval base image. ``preserve`` keeps the custom FROM and
            appends verifier/runtime dependencies.
        runtime_env: Runtime environment variables to expose inside Harbor task
            environments using Harbor's native ``[environment.env]`` template.
        pre_agent_setup: Commands to run as a Harbor environment healthcheck
            before the agent starts.
        task_resources: Optional Harbor task ``[environment]`` resource values
            (``cpus``, ``memory_mb``, ``storage_mb``). Missing keys use SkillEvaluator
            defaults.
        agent_workdir: Optional default working directory for agent command
            execution inside the Harbor task environment.

    Returns:
        List of generated task directory paths.
    """
    evals_dir = skill_path / "evals"
    evals_file = find_evals_file(skill_path)

    if not evals_file:
        raise FileNotFoundError(f"No evals dataset found in {evals_dir}")

    entries = _load_evals(evals_file)
    if not entries:
        raise ValueError(f"Empty dataset: {evals_file}")
    workspace_skill_paths = workspace_skill_paths or []
    workspace_skill_names = sorted({p.name for p in workspace_skill_paths})

    input_files_dir = evals_dir / "files"
    if not input_files_dir.exists():
        input_files_dir = None

    mcp_servers = _load_mcp_servers(skill_path)
    prepared_entries = _preflight_generated_tasks(entries, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_dirs: list[str] = []
    task_paths: list[Path] = []

    for normalized_entry, case_id in prepared_entries:
        task_dir = safe_child(output_dir, case_id)

        if os.path.lexists(task_dir):
            if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
                raise ValueError(
                    f"cannot safely replace generated task {case_id!r}: "
                    "this platform does not provide symlink-attack-resistant recursive deletion"
                )
            shutil.rmtree(task_dir)
        task_dir.mkdir(parents=True)

        _write_instruction(task_dir, normalized_entry.get("question", ""))
        _write_task_toml(
            task_dir,
            normalized_entry,
            with_skill,
            mcp_servers=mcp_servers,
            runtime_env=runtime_env,
            verifier_env=verifier_env,
            pre_agent_setup=pre_agent_setup,
            task_resources=task_resources,
            agent_workdir=agent_workdir,
        )
        _copy_verifier(task_dir)
        custom_grader = _copy_custom_grader(task_dir, skill_path, grading_mode)
        _write_entry_json(
            task_dir,
            normalized_entry,
            with_skill,
            workspace_mode=workspace_mode,
            workspace_skill_names=workspace_skill_names,
            grading_mode=grading_mode,
            custom_grader=custom_grader,
        )
        _write_test_sh(task_dir, grading_mode=grading_mode, custom_grader=custom_grader)

        _write_dockerfile(
            task_dir,
            skill_path=skill_path,
            reference_skills_dir=reference_skills_dir,
            workspace_skill_paths=workspace_skill_paths,
            has_skill=with_skill,
            input_files_dir=input_files_dir,
            entry=normalized_entry,
            evals_dir=evals_dir,
            exclude_skill_name=skill_path.name,
            base_image=base_image,
            custom_dockerfile_mode=custom_dockerfile_mode,
            repo_context_skill_path=skill_path,
            repo_context_mode="full" if copy_repo else "linked",
            compose_env_names=set(runtime_env or {}),
        )

        task_dirs.append(case_id)
        task_paths.append(task_dir)
        logger.debug("Generated task: %s (has_skill=%s)", case_id, with_skill)

    _write_dataset_toml(output_dir, task_dirs)
    _copy_metric_py(output_dir)

    logger.debug(
        "Generated %d Harbor tasks in %s (with_skill=%s)",
        len(task_paths),
        output_dir,
        with_skill,
    )
    return task_paths


def generate_harbor_tasks(
    skill_path: Path,
    output_dir: Path,
    *,
    with_skill: bool = True,
    reference_skills_dir: Path | None = None,
    workspace_skill_paths: list[Path] | None = None,
    workspace_mode: str = "isolated",
    grading_mode: str = "default",
    base_image: str = "",
    custom_dockerfile_mode: str = "rebase",
    copy_repo: bool = False,
    runtime_env: dict[str, str] | None = None,
    verifier_env: dict[str, str] | None = None,
    pre_agent_setup: list[str] | None = None,
    task_resources: dict[str, int] | None = None,
    agent_workdir: str | None = None,
) -> list[Path]:
    """Generate tasks privately, then publish one exact output snapshot."""

    with tempfile.TemporaryDirectory(prefix="skillevaluator-generated-tasks-") as temporary:
        private_root = Path(temporary).resolve(strict=True)
        private_output = private_root / "dataset"
        private_tasks = _generate_harbor_tasks_into(
            skill_path,
            private_output,
            with_skill=with_skill,
            reference_skills_dir=reference_skills_dir,
            workspace_skill_paths=workspace_skill_paths,
            workspace_mode=workspace_mode,
            grading_mode=grading_mode,
            base_image=base_image,
            custom_dockerfile_mode=custom_dockerfile_mode,
            copy_repo=copy_repo,
            runtime_env=runtime_env,
            verifier_env=verifier_env,
            pre_agent_setup=pre_agent_setup,
            task_resources=task_resources,
            agent_workdir=agent_workdir,
        )
        relative_tasks = [task.relative_to(private_output) for task in private_tasks]
        copytree_secure(
            private_output,
            output_dir,
            replace_existing=True,
            allowed_root=private_root,
        )
    return [output_dir / relative for relative in relative_tasks]
