# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001 -- standalone release helper.

"""Generate CI onboarding files from an anonymously smoke-tested Tier 1 image."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

try:
    import github_api
    import render_templates
    import stable_channel
except ModuleNotFoundError:
    from ci import github_api, render_templates, stable_channel

ROOT = Path(__file__).resolve().parents[1]
PLATFORM = {"os": "linux", "architecture": "amd64"}
SMOKE_CASES = ("all-pass", "invalid-name", "changed-one", "no-change", "missing-llm-key")
RUN_URL = re.compile(r"https://github\.com/NVIDIA/SkillEvaluator/actions/runs/[0-9]+")
DIGEST = re.compile(r"sha256:[a-f0-9]{64}")
SCANNER_COMMANDS = {
    "bandit": "--version",
    "pip-audit": "--version",
    "skillspector": "--version",
    "skillevaluator": "--version",
    "semgrep": "--version",
    "gitleaks": "version",
}


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _verify_receipt(smoke_path: Path, image: str, source_sha: str, build_run: str) -> dict:
    raw = smoke_path.read_bytes()
    smoke = github_api.strict_json(raw)
    runtime = smoke.get("runtime") if isinstance(smoke, dict) else None
    if (
        not isinstance(runtime, dict)
        or smoke.get("image") != image
        or not isinstance(runtime.get("digests"), list)
        or image not in runtime["digests"]
        or runtime.get("source_revision") != source_sha
        or not DIGEST.fullmatch(str(runtime.get("id")))
        or any(runtime.get(key) != value for key, value in PLATFORM.items())
    ):
        raise ValueError("The smoke test did not observe the requested public linux/amd64 image and source revision.")

    cases = smoke.get("cases")
    if (
        smoke.get("all_verified") is not True
        or not isinstance(cases, list)
        or not all(isinstance(case, dict) and case.get("verified") is True for case in cases)
        or tuple(case.get("case") for case in cases) != SMOKE_CASES
    ):
        raise ValueError("The public image must pass all five Tier 1 smoke cases.")

    return {
        "all_verified": True,
        "image": image,
        "source_revision": source_sha,
        "platform": PLATFORM,
        "public_manifest_matches_smoke_tested_image": True,
        "anonymous_pull_verified": True,
        "official_build_run": build_run,
        "public_manifest_digest": image.split("@", 1)[1],
        "smoke_runtime_id": runtime["id"],
        "image_smoke_cases": cases,
        "runtime_observation_sha256": _sha256(raw),
    }


def _observe_scanners(image: str, version: str, source_sha: str, build_run: str, observation_sha: str) -> dict:
    inventory = {}
    for command, argument in SCANNER_COMMANDS.items():
        observed = subprocess.check_output(
            ["docker", "run", "--rm", "--pull=never", "--network=none", "--entrypoint", command, image, argument],
            text=True,
            timeout=60,
        ).splitlines()
        match = re.search(r"(?<![\w.-])v?([0-9]+\.[0-9]+\.[0-9]+)(?![\w.-])", observed[0]) if observed else None
        if match is None:
            raise ValueError(f"Could not read the installed {command} version from the public image.")
        inventory[command] = {"command": f"{command} {argument}", "version": match[1]}
    if inventory["skillevaluator"]["version"] != version:
        raise ValueError("The installed image version does not match the requested release version.")
    inventory["verification"] = {
        "method": "Version commands executed in the anonymously pulled image used in all five smoke cases.",
        "image": image,
        "source_revision": source_sha,
        "build_run": build_run,
        "runtime_observation_sha256": observation_sha,
    }
    return inventory


def build_bundle(output: Path, version: str, source_sha: str, image: str, smoke_path: Path, build_run: str) -> None:
    if not stable_channel.VERSION.fullmatch(version) or not github_api.COMMIT_SHA.fullmatch(source_sha):
        raise ValueError("A numeric release version and full candidate commit SHA are required.")
    if not stable_channel.OFFICIAL_IMAGE.fullmatch(image) or not RUN_URL.fullmatch(build_run):
        raise ValueError("An official Tier 1 image digest and SkillEvaluator workflow run are required.")
    if output.exists() or output.is_symlink():
        raise ValueError("Use a new output directory to preserve previous release files.")
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    if metadata["project"]["version"] != version:
        raise ValueError("The candidate package version does not match the requested release version.")

    receipt = _verify_receipt(smoke_path, image, source_sha, build_run)
    inventory = _observe_scanners(image, version, source_sha, build_run, receipt["runtime_observation_sha256"])
    bundle = output / "ci/bundles" / version
    rendered = render_templates.render_templates(bundle, source_sha, image)
    _write_json(bundle / "verification-receipt.json", receipt)
    _write_json(bundle / "scanner-inventory.json", inventory)
    sums = bundle / "SHA256SUMS.txt"
    sums.write_text(
        "".join(f"{_sha256(path.read_bytes())}  {path.name}\n" for path in sorted(bundle.iterdir())),
        encoding="utf-8",
    )
    members = sorted(bundle.iterdir())
    with ZipFile(bundle / f"skillevaluator-ci-{version}.zip", "x", compression=ZIP_DEFLATED) as archive:
        for path in members:
            member = ZipInfo(path.name, date_time=(1980, 1, 1, 0, 0, 0))
            member.create_system = 3  # Unix
            member.external_attr = 0o100644 << 16
            archive.writestr(member, path.read_bytes(), compress_type=ZIP_DEFLATED)

    installation = output / "src/skillevaluator"
    installation.mkdir(parents=True)
    refs = {key: rendered[key] for key in ("repository", "workflow_ref", "publisher_ref", "image")}
    _write_json(installation / "ci-release.json", {**refs, "schema_version": 1})
    # Preserve semantic capabilities and the report contract offered by the candidate.
    all_tiers = github_api.strict_json((ROOT / "src/skillevaluator/ci-release-all-tiers.json").read_bytes())
    all_tiers.update(
        refs,
        feedback_ref=source_sha,
        gitlab_ref=source_sha,
        tools_ref=source_sha,
        evaluator_version=version,
        runtime_lock_sha256=_sha256((ROOT / "uv.lock").read_bytes()),
    )
    _write_json(installation / "ci-release-all-tiers.json", all_tiers)

    for name, content in stable_channel.render(bundle).items():
        directory = "ci/gitlab" if name == "stable-tier1.yml" else ".github/workflows"
        target = output / directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--smoke-receipt", type=Path, required=True)
    parser.add_argument("--build-run", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        build_bundle(args.output, args.version, args.source_sha, args.image, args.smoke_receipt, args.build_run)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"CI release bundle generation failed: {error}\n")


if __name__ == "__main__":
    main()
