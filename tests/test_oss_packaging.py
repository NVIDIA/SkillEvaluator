# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public-package regression checks for the OSS edition."""

from __future__ import annotations

import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

from click.testing import CliRunner
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
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
PUBLIC_TEXT_SUFFIXES = {"", ".json", ".md", ".mdx", ".py", ".sh", ".toml", ".txt", ".yml", ".yaml", ".j2"}
PACKAGED_NVIDIA_BUILD_RUNTIME_FILES = {
    "skillevaluator/tier3/harbor/local_agents.py",
    "skillevaluator/tier3/harbor/nvidia_build_bridge.py",
    "skillevaluator/tier3/harbor/secure_docker_environment.py",
}
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
    configuration = (REPO_ROOT / "docs" / "configuration.mdx").read_text(encoding="utf-8")

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


def test_security_extra_does_not_bundle_external_or_unused_scanner_dependencies() -> None:
    security = [Requirement(raw) for raw in _project()["project"]["optional-dependencies"]["security"]]
    lock_names = {package["name"] for package in _lock()["package"]}

    for external_scanner in ("semgrep", "skillspector"):
        assert not any(canonicalize_name(requirement.name) == external_scanner for requirement in security)
        assert external_scanner not in lock_names
    for unused_dependency in ("langchain-core", "langsmith"):
        assert not any(canonicalize_name(requirement.name) == unused_dependency for requirement in security)
        assert unused_dependency not in lock_names
    assert {
        frozenset(str(specifier) for specifier in requirement.specifier)
        for requirement in security
        if canonicalize_name(requirement.name) == canonicalize_name("pip-audit")
    } == {frozenset({">=2.10.0"})}


def test_third_party_notices_do_not_list_removed_safety_dependency() -> None:
    notices = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    assert "Safety (MIT)" not in notices


def test_release_lock_avoids_accidental_prereleases_and_known_fixed_versions() -> None:
    project = _project()
    lock = _lock()
    versions = {package["name"]: Version(package["version"]) for package in lock["package"]}

    assert "prerelease" not in project.get("tool", {}).get("uv", {})
    prerelease_guarded = {"cyclonedx-python-lib", "numpy", "pydantic", "wrapt"}
    for package in lock["package"]:
        if package["name"] in prerelease_guarded:
            assert Version(package["version"]).is_prerelease is False
    assert versions["cryptography"] >= Version("48.0.1")
    assert versions["msgpack"] >= Version("1.2.1")
    assert versions["pydantic-settings"] >= Version("2.14.2")


def test_release_lock_enforces_nspect_remediation_floors_without_removed_telemetry_stack() -> None:
    project = _project()
    extras = project["project"]["optional-dependencies"]
    tier3 = extras["tier3"]
    all_lock_versions: dict[str, list[Version]] = {}
    for package in _lock()["package"]:
        all_lock_versions.setdefault(package["name"], []).append(Version(package["version"]))

    assert "mcp>=1.28.1,<2" in tier3
    assert "pyjwt[crypto]>=2.13.0" in tier3
    assert "telemetry" not in extras
    assert all(
        "protobuf" not in requirement.lower() for requirements in extras.values() for requirement in requirements
    )
    assert all(
        "opentelemetry" not in requirement.lower() for requirements in extras.values() for requirement in requirements
    )
    assert all(version >= Version("1.28.1") for version in all_lock_versions["mcp"])
    assert all(version >= Version("2.13.0") for version in all_lock_versions["pyjwt"])
    assert "protobuf" not in all_lock_versions
    assert not any(name.startswith("opentelemetry") for name in all_lock_versions)


def test_public_docs_declare_support_and_security_sections() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    support = (REPO_ROOT / "SUPPORT.md").read_text(encoding="utf-8")

    assert "\n## Support\n" in readme
    assert "\n## Security\n" in readme
    assert "Support level: **Experimental**" in readme
    assert "Support level: **Experimental**" in support


def test_public_readme_is_a_concise_docs_landing_page() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    quickstart = readme.partition("\n## Quickstart\n")[2].partition("\n## LLM provider setup\n")[0]
    deeper_evaluations = readme.partition("\n## Run deeper evaluations\n")[2].partition("\n## Documentation\n")[0]
    normalized = " ".join(readme.split())

    positioning = (
        "SkillEvaluator is an open-source, multi-tier framework for evaluating AI agent artifacts, "
        "starting with agent skills: deterministic quality gates, semantic overlap detection, "
        "synthetic eval dataset generation, and live agent evaluation."
    )
    assert positioning in normalized
    assert "https://docs.nvidia.com/skills/skillevaluator/" in readme
    assert "https://github.com/NVIDIA/skills" in readme
    assert "https://github.com/NVIDIA/SkillSpector" in readme
    assert "\n## Three-tier overview\n" in readme
    assert "docs/assets/three-tier-overview.svg" in readme
    assert "\n## Quickstart\n" in readme
    assert "skillevaluator[all] @ git+https://github.com/NVIDIA/SkillEvaluator.git" in readme
    assert (
        "skillevaluator validate ./my-skill \\\n  --checks schema,pii,license,quality,unicode,lint \\\n  --no-dedup"
    ) in quickstart
    assert "skillevaluator quality-check ./my-skill" not in quickstart
    assert "SKILL_EVAL_LLM_PROVIDER=nv_build" in readme
    assert "NVIDIA_API_KEY='nvapi-...'" in readme
    assert "skillevaluator context-optimization-check ./my-skill" in readme
    assert (
        "skillevaluator validate ./my-skill \\\n  --full \\\n  --agents codex \\\n  --env-mode docker"
    ) in deeper_evaluations
    assert "Semgrep, SkillSpector, and Gitleaks" in deeper_evaluations
    assert "enables autopilot" in deeper_evaluations
    assert "evals/evals.json" in deeper_evaluations
    assert "only for trusted skills and workspaces" in deeper_evaluations
    assert "skillevaluator create-eval-dataset ./my-skill --full" in deeper_evaluations
    assert "skillevaluator tier3 evaluate ./my-skill" not in deeper_evaluations
    assert "--n-attempts 1" not in deeper_evaluations
    assert "tier3-live-evaluation#plan-for-cost" in readme
    assert "\n## Tier 1:" not in readme
    assert "Skill Evaluator" not in readme
    assert "Skillevaluator" not in readme
    assert len(readme.split()) <= 1100


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


def test_public_distributions_include_nvidia_build_runtime_bridges(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(tmp_path)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"distribution build failed:\n{result.stdout}\n{result.stderr}"

    wheels = list(tmp_path.glob("*.whl"))
    sdists = list(tmp_path.glob("*.tar.gz"))
    assert len(wheels) == 1
    assert len(sdists) == 1

    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_members = set(archive.namelist())
        metadata_member = next(member for member in wheel_members if member.endswith(".dist-info/METADATA"))
        wheel_metadata = archive.read(metadata_member).decode("utf-8")
    missing_from_wheel = PACKAGED_NVIDIA_BUILD_RUNTIME_FILES - wheel_members
    assert not missing_from_wheel, f"wheel is missing runtime bridge files: {sorted(missing_from_wheel)}"
    assert not any(member.startswith("skillevaluator/telemetry/") for member in wheel_members)
    assert "Provides-Extra: telemetry" not in wheel_metadata
    assert "Requires-Dist: protobuf" not in wheel_metadata
    assert "Requires-Dist: opentelemetry-" not in wheel_metadata

    with tarfile.open(sdists[0], "r:gz") as archive:
        sdist_members = {member.name.partition("/")[2] for member in archive.getmembers()}
    expected_sdist_members = {f"src/{path}" for path in PACKAGED_NVIDIA_BUILD_RUNTIME_FILES}
    missing_from_sdist = expected_sdist_members - sdist_members
    assert not missing_from_sdist, f"sdist is missing runtime bridge files: {sorted(missing_from_sdist)}"
    assert not any(member.startswith("src/skillevaluator/telemetry/") for member in sdist_members)


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


def test_public_slim_docker_image_uses_only_distribution_dependencies() -> None:
    project = _project()
    extras = project["project"]["optional-dependencies"]
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "skillevaluator[tier2,tier3,security]" in extras["all"]
    assert re.search(r"^FROM python:3\.12-slim$", dockerfile, flags=re.MULTILINE)

    public_install = 'python -m pip install --no-cache-dir ".[all]"'
    install_run = next(
        run
        for run in re.findall(r"^RUN\s+(.*?)(?=^[A-Z]+\s|\Z)", dockerfile, flags=re.MULTILINE | re.DOTALL)
        if public_install in run
    )
    assert "apt-get" not in install_run
    assert "git+" not in install_run


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
    source_files = set(_public_source_files(REPO_ROOT)) - {REPO_ROOT / ".github" / "CODEOWNERS"}
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in source_files)
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
    source_scan = f"{scanner_command} --root . --allowlist config/oss_boundary_allowlist.json"
    artifact_scan = f"{source_scan} --archive dist/*.whl --archive dist/*.tar.gz"

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
    guide = (REPO_ROOT / "docs" / "tier2-deduplication.mdx").read_text(encoding="utf-8")
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


def test_public_docs_show_external_nvidia_build_harness_paths_only() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    configuration = (REPO_ROOT / "docs" / "configuration.mdx").read_text(encoding="utf-8")
    tier3 = (REPO_ROOT / "docs" / "tier3-live-evaluation.mdx").read_text(encoding="utf-8")
    public_docs = f"{readme}\n{configuration}\n{tier3}"

    assert "gpt-5.4-mini" in public_docs
    assert "nvidia/nemotron-3-nano-30b-a3b" in public_docs
    assert "nvidia/nvidia/nemotron-3-nano-30b-a3b" in public_docs
    assert "Nemotron Super" in public_docs
    assert "meta/llama-3.1-8b-instruct" in public_docs
    assert "--agent-model opencode=nvidia/nvidia/nemotron-3-super-120b-a12b" in public_docs
    assert "--agent-model codex=nvidia/nemotron-3-super-120b-a12b" in public_docs
    assert "--agent-model claude-code=nvidia/nemotron-3-super-120b-a12b" in public_docs
    assert "--agent-model opencode=nvidia/meta/llama-3.1-8b-instruct" in public_docs
    assert "skillevaluator tier3 evaluate ./my-skill --agents opencode --env-mode docker\n" in tier3
    assert "skillevaluator tier3 evaluate ./my-skill --agents codex --env-mode docker\n" in tier3
    assert "skillevaluator tier3 evaluate ./my-skill --agents claude-code --env-mode docker\n" in tier3
    assert "never changes models silently" in public_docs
    assert "direct OpenCode" in public_docs
    assert "Docker or local compatibility bridge" in public_docs
    assert "experimental Claude Code" in public_docs


def test_ci_installs_the_security_wheel_on_rhel8() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    rhel8_job = workflow.split("  rhel8-security-install:\n", 1)[1].split("\n  package:\n", 1)[0]

    assert "container: rockylinux/rockylinux:8.10" in rhel8_job
    assert 'getconf GNU_LIBC_VERSION)" = "glibc 2.28"' in rhel8_job
    assert "uv build --wheel --python 3.12 --no-sources" in rhel8_job
    assert '"${wheel}[security]"' in rhel8_job
    assert 'Version(version("pip-audit")) >= Version("2.10.0")' in rhel8_job
    assert ".rhel8-security-venv/bin/bandit --version" in rhel8_job
    assert ".rhel8-security-venv/bin/semgrep --version" not in rhel8_job


def test_fern_docs_use_the_verified_skills_basepath_and_launch_positioning() -> None:
    fern = (REPO_ROOT / "fern" / "docs.yml").read_text(encoding="utf-8")
    overview = (REPO_ROOT / "docs" / "index.mdx").read_text(encoding="utf-8")
    normalized_overview = " ".join(overview.split())

    positioning = (
        "SkillEvaluator is an open-source, multi-tier framework for evaluating AI agent artifacts, "
        "starting with agent skills: deterministic quality gates, semantic overlap detection, "
        "synthetic eval dataset generation, and live agent evaluation."
    )
    assert "url: nvidia-skillevaluator.docs.buildwithfern.com/skills/skillevaluator" in fern
    assert "custom-domain: docs.nvidia.com/skills/skillevaluator" in fern
    assert "docs.nvidia.com/skillevaluator" not in fern
    assert positioning in normalized_overview
    assert "https://github.com/NVIDIA/skills" in overview
    assert "https://github.com/NVIDIA/SkillSpector" in overview


def test_tier3_docs_explain_cost_controls_and_local_mode_tradeoffs() -> None:
    tier3 = (REPO_ROOT / "docs" / "tier3-live-evaluation.mdx").read_text(encoding="utf-8")
    normalized = " ".join(tier3.split())

    assert "eval cases \u00d7 agents \u00d7 attempts \u00d7 arms" in normalized
    assert "--skip-baseline" in tier3
    assert "cannot produce Skill Lift" in normalized
    assert "--n-concurrent" in tier3
    assert "--max-agents" in tier3
    assert "do not reduce the total planned trials" in tier3
    assert "--env-mode local" in tier3
    assert "does **not** automatically eliminate model charges" in tier3
    assert "weaker isolation than Docker" in tier3


def test_launch_docs_address_scanner_and_naming_ambiguities() -> None:
    quickstart = (REPO_ROOT / "docs" / "quickstart.mdx").read_text(encoding="utf-8")
    ci = (REPO_ROOT / "docs" / "ci-integration.mdx").read_text(encoding="utf-8")
    environment = (REPO_ROOT / "docs" / "environment-variables.mdx").read_text(encoding="utf-8")
    normalized_quickstart = " ".join(quickstart.split())

    assert "brew install semgrep gitleaks" in quickstart
    assert "uv tool install git+https://github.com/NVIDIA/SkillSpector.git" in quickstart
    assert "Semgrep, SkillSpector, and Gitleaks" in quickstart
    assert "missing scanner evidence leaves the result `INCOMPLETE` and exits `1`" in normalized_quickstart
    assert "most often Gitleaks" not in ci
    assert "Semgrep, SkillSpector, and Gitleaks are all separate executables" in " ".join(ci.split())
    assert "`SKILL_EVAL_*` covers provider and model configuration" in " ".join(environment.split())
    assert "`SKILLEVALUATOR_*` covers product-level validation" in " ".join(environment.split())
    assert "are not interchangeable" in environment


def test_harbor_atif_and_agent_eval_alias_are_defined() -> None:
    harbor_url = "https://github.com/harbor-framework/harbor"
    harbor_pages = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "agents-and-sandboxes.mdx",
        REPO_ROOT / "docs" / "cli-reference.mdx",
        REPO_ROOT / "docs" / "configuration.mdx",
        REPO_ROOT / "docs" / "custom-graders.mdx",
        REPO_ROOT / "docs" / "developer-guide.mdx",
        REPO_ROOT / "docs" / "environment-variables.mdx",
        REPO_ROOT / "docs" / "eval-datasets.mdx",
        REPO_ROOT / "docs" / "installation.mdx",
        REPO_ROOT / "docs" / "reports.mdx",
        REPO_ROOT / "docs" / "tier3-live-evaluation.mdx",
    ]

    for path in harbor_pages:
        content = path.read_text(encoding="utf-8")
        assert f"[Harbor]({harbor_url})" in content, path
        assert "open-source agent evaluation framework" in " ".join(content.split()), path

    for name in ("custom-graders.mdx", "environment-variables.mdx"):
        content = (REPO_ROOT / "docs" / name).read_text(encoding="utf-8")
        assert "Agent Trajectory Interchange Format (ATIF)" in content

    for name in ("tier1-validation.mdx", "ci-integration.mdx", "cli-reference.mdx", "tier3-live-evaluation.mdx"):
        content = (REPO_ROOT / "docs" / name).read_text(encoding="utf-8")
        assert "--agent-eval" in content
        assert "not currently deprecated" in " ".join(content.split())


def test_public_product_branding_uses_the_canonical_name() -> None:
    paths = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "CHANGELOG.md",
        REPO_ROOT / "CITATION.cff",
        REPO_ROOT / "CODE_OF_CONDUCT.md",
        REPO_ROOT / "SECURITY.md",
        REPO_ROOT / "SUPPORT.md",
        REPO_ROOT / "Dockerfile",
        REPO_ROOT / "sonar-project.properties",
        *(REPO_ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.yml"),
        *(REPO_ROOT / "docs").rglob("*.mdx"),
        *(REPO_ROOT / "src" / "skillevaluator").rglob("*.py"),
        *(REPO_ROOT / "src" / "skillevaluator").rglob("*.j2"),
        *(REPO_ROOT / "src" / "skillevaluator").rglob("SKILL.md"),
    ]

    for path in paths:
        content = path.read_text(encoding="utf-8")
        assert "Skill Evaluator" not in content, path
        assert "Skillevaluator" not in content, path
