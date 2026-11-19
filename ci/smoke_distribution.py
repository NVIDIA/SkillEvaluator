# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001

"""Exercise an installed SkillEvaluator distribution outside the source checkout."""

import argparse
import json
import subprocess
import sys
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory


def smoke(expected_version: str) -> None:
    if version("skillevaluator") != expected_version:
        raise ValueError("Installed package does not have the requested version.")
    command = str(Path(sys.executable).with_name("skillevaluator"))
    resources = files("skillevaluator")
    descriptors = [
        json.loads(resources.joinpath(name).read_text(encoding="utf-8"))
        for name in ("ci-release.json", "ci-release-all-tiers.json")
    ]
    if descriptors[1]["evaluator_version"] != expected_version:
        raise ValueError("All-tier CI descriptor has a different evaluator version.")

    with TemporaryDirectory(prefix="skillevaluator-release-") as directory:
        repo = Path(directory)
        skill = repo / "skills" / "smoke-check"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: smoke-check\ndescription: Check the installed release.\n---\n"
            "# Smoke check\n\nRead the input and summarize its key points.\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-q", "-b", "main", directory], check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/example/skills.git"], cwd=repo, check=True
        )
        subprocess.run(["git", "add", "skills"], cwd=repo, check=True)
        observed = subprocess.check_output([command, "--version"], cwd=repo, text=True).strip()
        if observed.split()[-1] != expected_version:
            raise ValueError(f"Unexpected CLI version: {observed}")
        subprocess.run(
            [
                command,
                "validate",
                str(skill),
                "--tiers",
                "1",
                "--checks",
                "schema,harness",
                "-r",
                "json",
                "-o",
                "reports",
            ],
            cwd=repo,
            check=True,
        )
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in (repo / "reports").rglob("*.json")]
        if not any("smoke-check" in json.dumps(report) for report in reports):
            raise ValueError("The installed CLI did not write a report for the smoke skill.")

        base = [
            command,
            "ci",
            "init",
            "--repo",
            directory,
            "--default-branch",
            "main",
            "--mode",
            "preview",
            "--no-input",
            "--show-files",
        ]
        for options, filename, descriptor in (
            (["--checks-only"], "skill-checks.yml", descriptors[0]),
            (["--tiers", "1,2,3", "--provider", "openai"], "skill-checks-all-tiers.yml", descriptors[1]),
        ):
            preview = subprocess.check_output(base + options, cwd=repo, text=True)
            for expected in (filename, descriptor["workflow_ref"], descriptor["image"]):
                if expected not in preview:
                    raise ValueError(f"{filename} preview is missing {expected}.")
        if (repo / ".github").exists():
            raise ValueError("CI onboarding preview changed the scratch checkout.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    smoke(parser.parse_args().version)
