# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public-package regression checks for the OSS edition."""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

from click.testing import CliRunner
from packaging.requirements import Requirement
from packaging.version import Version

from skillevaluator.cli import cli

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "GOVERNANCE.md",
    "SUPPORT.md",
    "CHANGELOG.md",
    "CITATION.cff",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/workflows/ci.yml",
    ".github/workflows/security.yml",
)
PUBLIC_TEXT_SUFFIXES = {"", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yml", ".yaml", ".j2"}
SOURCE_SCAN_EXCLUDED_DIRS = {
    ".git",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".smoke-venv",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "log",
    "node_modules",
    "reports",
    "results",
}


def _project() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as project_file:
        return tomllib.load(project_file)


def _lock() -> dict:
    with (REPO_ROOT / "uv.lock").open("rb") as lock_file:
        return tomllib.load(lock_file)


def _public_source_files(repo_root: Path) -> list[Path]:
    tracked: list[Path] | None = None
    if (repo_root / ".git").exists():
        try:
            result = subprocess.run(
                ["git", "ls-files", "-z"],
                cwd=repo_root,
                check=False,
                capture_output=True,
            )
        except OSError:
            result = None
        if result is not None and result.returncode == 0:
            tracked = [repo_root / relative for relative in result.stdout.decode("utf-8").split("\0") if relative]

    candidates = tracked if tracked is not None else list(repo_root.rglob("*"))
    return sorted(
        path
        for path in candidates
        if path.is_file()
        and path.suffix in PUBLIC_TEXT_SUFFIXES
        and not any(
            part in SOURCE_SCAN_EXCLUDED_DIRS or part.endswith(".egg-info")
            for part in path.relative_to(repo_root).parts
        )
    )


def test_public_package_has_no_private_package_indexes() -> None:
    project = _project()

    assert "extra-index-url" not in project.get("tool", {}).get("uv", {})


def test_public_extras_use_public_dependency_sources() -> None:
    project = _project()
    extras = project["project"]["optional-dependencies"]
    requirements = [Requirement(value) for values in extras.values() for value in values]

    assert "internal" not in extras
    assert all(
        requirement.url is None or requirement.url.startswith("git+https://github.com/") for requirement in requirements
    )


def test_public_extras_exclude_internal_runtime_dependencies() -> None:
    project = _project()
    extras = project["project"]["optional-dependencies"]
    dependency_text = "\n".join(requirement.lower() for requirements in extras.values() for requirement in requirements)

    assert "py" + "mil" + "vus" not in dependency_text
    assert "sandbox" + "-k8s" not in dependency_text
    assert "ipp" + "bot" not in dependency_text


def test_public_sources_exclude_retired_internal_runtime_paths() -> None:
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in (REPO_ROOT / "src").rglob("*.py"))
    retired_terms = (
        "NVI" + "DIA" + "_INFERENCE_KEY",
        "as" + "tra_sandbox",
        "inter" + "_skill",
        "py" + "mil" + "vus",
    )

    for term in retired_terms:
        assert term not in source_text


def test_public_docs_explain_the_single_nvidia_credential_skillspector_path() -> None:
    configuration = (REPO_ROOT / "docs" / "CONFIGURATION.md").read_text(encoding="utf-8")

    assert "SkillSpector's OpenAI-compatible provider path" in configuration
    assert "does not create a second NVIDIA credential name" in configuration
    assert "Only the selected provider settings and basic process environment" in configuration


def test_security_extra_uses_pip_audit_without_bundling_safety() -> None:
    project = _project()
    security = project["project"]["optional-dependencies"]["security"]
    lock_names = {package["name"] for package in _lock()["package"]}

    assert any(requirement.startswith("pip-audit") for requirement in security)
    assert not any(requirement.startswith("safety") for requirement in security)
    assert "safety" not in lock_names
    assert "safety-schemas" not in lock_names
    assert "nltk" not in lock_names


def test_third_party_notices_do_not_list_removed_safety_dependency() -> None:
    notices = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    assert "Safety (MIT)" not in notices


def test_release_lock_avoids_accidental_prereleases_and_known_fixed_versions() -> None:
    project = _project()
    lock = _lock()
    versions = {package["name"]: Version(package["version"]) for package in lock["package"]}

    assert "prerelease" not in project.get("tool", {}).get("uv", {})
    for package in ("numpy", "pydantic", "wrapt"):
        assert versions[package].is_prerelease is False
    assert versions["cryptography"] >= Version("48.0.1")
    assert versions["msgpack"] >= Version("1.2.1")
    assert versions["pydantic-settings"] >= Version("2.14.2")


def test_public_docs_declare_support_and_security_sections() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    support = (REPO_ROOT / "SUPPORT.md").read_text(encoding="utf-8")

    assert "\n## Support\n" in readme
    assert "\n## Security\n" in readme
    assert "Support level: **Experimental**" in readme
    assert "Support level: **Experimental**" in support


def test_public_quickstart_is_one_install_command_without_a_fixture_clone() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    quickstart = readme.split("\n## Quickstart\n", 1)[1].split("\n## Tier 1:", 1)[0]

    install = 'uv tool install --python 3.13 "skillevaluator[all] @ git+https://github.com/NVIDIA/SkillEvaluator.git"'
    assert install in quickstart
    assert "skillevaluator quality-check ./my-skill" in quickstart
    assert "git clone" not in quickstart


def test_release_metadata_is_public_facing_and_version_consistent() -> None:
    notices = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    version = _project()["project"]["version"]

    assert "must be reconciled" not in notices.lower()
    assert f'version: "{version}"' in citation


def test_public_sources_use_the_public_nvidia_build_contract() -> None:
    provider_config = (REPO_ROOT / "src" / "skillevaluator" / "provider_config.py").read_text(encoding="utf-8")

    assert '"NVIDIA_API_KEY"' in provider_config
    assert "https://integrate.api.nvidia.com/v1" in provider_config


def test_removed_benchmark_authoring_surface_stays_absent() -> None:
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in _public_source_files(REPO_ROOT)).lower()
    forbidden = (
        "convert" + "-benchmark",
        "convert" + "_benchmark",
        "benchmark" + "-conversion",
        "benchmark" + "_conversion",
        "benchmark" + "_staging",
        "benchmark" + "_conversion_report",
    )

    for term in forbidden:
        assert term not in source_text
    assert not (REPO_ROOT / "src/skillevaluator/tier3" / ("benchmark" + "_conversion.py")).exists()
    assert not (REPO_ROOT / "src/skillevaluator/tier3" / ("benchmark" + "_staging.py")).exists()


def test_public_release_includes_plc_template_work_products() -> None:
    missing = [path for path in PUBLIC_REQUIRED_FILES if not (REPO_ROOT / path).is_file()]

    assert not missing, f"missing public release work products: {', '.join(missing)}"


def test_public_docker_image_uses_only_public_dependencies() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    private_terms = (
        "NV_" + "SHARED_PIP_INDEX_URL",
        "IPP" + "BOT_SDK_PIP_INDEX_URL",
        ".[" + "internal]",
        "SKILLEVALUATOR_" + "EDITION",
    )

    for term in private_terms:
        assert term not in dockerfile
    assert '".[all]"' in dockerfile


def test_public_slim_docker_image_can_install_pinned_public_skillspector() -> None:
    project = _project()
    extras = project["project"]["optional-dependencies"]
    skillspector_requirement = next(
        requirement for requirement in extras["security"] if requirement.startswith("skillspector @ ")
    )
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert re.fullmatch(
        r"skillspector @ git\+https://github\.com/NVIDIA/SkillSpector\.git@[0-9a-f]{40}",
        skillspector_requirement,
    )
    assert "skillevaluator[tier2,tier3,telemetry,security]" in extras["all"]
    assert re.search(r"^FROM python:3\.12-slim$", dockerfile, flags=re.MULTILINE)

    git_install = "apt-get install --yes --no-install-recommends git"
    public_install = 'python -m pip install --no-cache-dir ".[all]"'
    install_run = next(
        run
        for run in re.findall(r"^RUN\s+(.*?)(?=^[A-Z]+\s|\Z)", dockerfile, flags=re.MULTILINE | re.DOTALL)
        if public_install in run
    )
    assert git_install in install_run
    assert install_run.index(git_install) < install_run.index(public_install)


def test_public_source_files_fall_back_without_git_metadata(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "src" / "package.py"
    source.parent.mkdir()
    source.write_text("public source", encoding="utf-8")
    ignored = tmp_path / ".venv" / "private.py"
    ignored.parent.mkdir()
    ignored.write_text("generated dependency", encoding="utf-8")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **_kwargs: subprocess.CompletedProcess(args[0], 128, stdout=b"", stderr=b"fatal"),
    )

    assert _public_source_files(tmp_path) == [source]


def test_public_release_has_no_internal_repository_metadata() -> None:
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in _public_source_files(REPO_ROOT))
    forbidden = (
        "P4" + "USER",
        "NV_" + "SHARED_PIP_INDEX_URL",
        "IPP" + "BOT_SDK_PIP_INDEX_URL",
    )

    for term in forbidden:
        assert term not in source_text
    assert not (REPO_ROOT / ".nspect-allowlist.toml").exists()
    assert not (REPO_ROOT / (".git" + "lab-ci.yml")).exists()
    assert not (REPO_ROOT / ".p4config").exists()


def test_public_tree_has_no_legacy_root_version_module() -> None:
    assert not (REPO_ROOT / "version.py").exists()


def test_public_docs_have_no_personal_staging_ownership() -> None:
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in _public_source_files(REPO_ROOT))
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert ("@chris" + "knvidia") not in source_text
    assert ("MAINTAINERS" + ".md") not in readme
    assert ("[CODE" + "OWNERS]") not in readme


def test_github_actions_are_pinned_to_commit_shas() -> None:
    workflows_dir = REPO_ROOT / ".github" / "workflows"
    workflow_paths = sorted((*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml")))
    action_refs: list[tuple[Path, str]] = []

    for workflow_path in workflow_paths:
        workflow = workflow_path.read_text(encoding="utf-8")
        refs = re.findall(r"^\s*-\s+uses:\s+[^@\s]+@([^\s#]+)", workflow, flags=re.MULTILINE)
        action_refs.extend((workflow_path, ref) for ref in refs)

    assert workflow_paths
    assert action_refs
    for workflow_path, ref in action_refs:
        assert re.fullmatch(r"[0-9a-f]{40}", ref), f"{workflow_path.relative_to(REPO_ROOT)}: unpinned ref {ref}"


def test_ci_scans_source_and_built_distributions_for_oss_boundary_violations() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    scanner_command = "python scripts/check_oss_boundary.py"
    source_scan = (
        f"{scanner_command} --root . --allowlist config/oss_boundary_allowlist.json"
    )
    artifact_scan = (
        f"{source_scan} --archive dist/*.whl --archive dist/*.tar.gz"
    )

    assert source_scan in workflow
    assert artifact_scan in workflow
    assert workflow.index("uv build --python 3.13 --no-sources") < workflow.index(artifact_scan)


def test_retired_private_upload_artifact_is_not_part_of_public_gitignore() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    retired_artifact = "." + "harbor" + "-viewer-upload/"

    assert retired_artifact not in gitignore


def test_public_package_metadata_has_no_personal_email_addresses() -> None:
    project = _project()

    assert all("email" not in author for author in project["project"].get("authors", []))


def test_public_cli_exposes_both_tier_two_workflows_without_a_service() -> None:
    runner = CliRunner()

    root_help = runner.invoke(cli, ["--help"])
    similarity_help = runner.invoke(cli, ["similarity-check", "--help"])
    dedup_help = runner.invoke(cli, ["dedup-scan", "--help"])

    assert root_help.exit_code == 0
    assert similarity_help.exit_code == 0
    assert dedup_help.exit_code == 0
    assert "inter-skill-check" not in root_help.output
    assert "--save-catalog" in similarity_help.output
    assert "--catalog" in similarity_help.output
    assert "--catalog" not in dedup_help.output


def test_public_docs_show_tier_two_collection_and_catalog_workflows() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    guide = (REPO_ROOT / "docs" / "TIER2_DEDUPLICATION.md").read_text(encoding="utf-8")
    public_docs = f"{readme}\n{guide}"

    assert "similarity-check ./skills" in public_docs
    assert "--save-catalog" in public_docs
    assert "--catalog" in public_docs
    assert "dedup-scan` is an alias" in public_docs
    assert "No external vector database or catalog service" in public_docs
    assert "sends skill names and descriptions" in public_docs
    assert "sends each discovered `SKILL.md` in full" in public_docs
    assert "Only candidate clusters found by the embedding stage are" in public_docs
    assert "sent to the configured chat LLM for classification" in public_docs
    assert "NVI" + "DIA" + "_INFERENCE_KEY" not in public_docs
