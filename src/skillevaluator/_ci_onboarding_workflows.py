# SPDX-License-Identifier: Apache-2.0
"""Conservative onboarding of an existing, reviewable Skill checks caller."""

from __future__ import annotations

import re
from dataclasses import dataclass

import yaml

from skillevaluator._ci_onboarding_git import SetupError, safe_relative

EVALUATOR = "NVIDIA/SkillEvaluator/.github/workflows/evaluate-skills.yml"
WORKFLOW = re.compile(
    r"(?P<repository>[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*)/"
    r"(?P<path>\.github/workflows/[A-Za-z0-9_-][A-Za-z0-9_.-]*\.ya?ml)@"
    r"(?P<ref>[0-9a-f]{40}|v[0-9]+(?:\.[0-9]+)*)"
)


class WorkflowLoader(yaml.SafeLoader):
    """Read workflow keys such as 'on' without YAML 1.1 boolean coercion."""


WorkflowLoader.yaml_implicit_resolvers = {
    key: [(tag, pattern) for tag, pattern in values if tag != "tag:yaml.org,2002:bool"]
    for key, values in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
WorkflowLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|false)$", re.IGNORECASE), list("tTfF")
)


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result or key == "<<":
            raise SetupError("Workflow YAML must use unique string keys without merge aliases.")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


WorkflowLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def parse_workflow(content: str) -> dict:
    try:
        if len(content.encode()) > 128 * 1024:
            raise SetupError("Workflow exceeds the onboarding size limit.")
        if any(
            isinstance(event, yaml.AliasEvent) or getattr(event, "anchor", None) or getattr(event, "tag", None)
            for event in yaml.parse(content)
        ):
            raise SetupError(
                "Workflow aliases or explicit YAML tags require manual review before automatic onboarding."
            )
        result = yaml.load(content, Loader=WorkflowLoader)
    except (yaml.YAMLError, RecursionError) as exc:
        raise SetupError("The existing scan workflow is not valid, bounded YAML.") from exc
    if not isinstance(result, dict):
        raise SetupError("The existing scan workflow must be a YAML mapping.")
    return result


def identity_allowed(workflow: dict, job: dict) -> bool:
    permission = job.get("permissions", workflow.get("permissions", {}))
    return permission == "write-all" or (isinstance(permission, dict) and permission.get("id-token") == "write")


@dataclass(frozen=True)
class ExistingScan:
    text: str
    data: dict
    job: str
    uses: str
    roots: list[str]
    score: int
    scan_mode: str
    evaluator_sha: str | None
    delegated_wrapper: str | None

    def with_identity_permission(self) -> str:
        job = self.data["jobs"][self.job]
        if identity_allowed(self.data, job):
            return self.text
        permissions = job.get("permissions", self.data.get("permissions", {"contents": "read"}))
        if not isinstance(permissions, dict):
            raise SetupError("Use an explicit permissions mapping before adding reporting to this workflow.")
        # The only semantic change is the permission required for a signed scan
        # receipt. Preserve all other text, including triggers and team comments.
        tree = yaml.compose(self.text, Loader=WorkflowLoader)
        jobs = next(value for key, value in tree.value if key.value == "jobs")
        node = next(value for key, value in jobs.value if key.value == self.job)
        permission_node = next((value for key, value in node.value if key.value == "permissions"), None)
        updated = {**permissions, "id-token": "write"}
        if permission_node is None:
            if node.flow_style:
                raise SetupError("Expand the existing scan job to block YAML before adding its identity permission.")
            at = node.start_mark.index
            indent = node.start_mark.column
            lines = ["permissions:"] + ["  " + line for line in yaml.safe_dump(updated, sort_keys=False).splitlines()]
            insertion = ("\n" + " " * indent).join(lines) + "\n" + " " * indent
            return self.text[:at] + insertion + self.text[at:]
        # A compact mapping avoids changing indentation or eating the next key.
        replacement = yaml.safe_dump(updated, default_flow_style=True, sort_keys=False, width=1000).strip()
        start, end = permission_node.start_mark.index, permission_node.end_mark.index
        if not permission_node.flow_style:
            replacement += "\n" + " " * permission_node.end_mark.column
        return self.text[:start] + replacement + self.text[end:]


def existing_scan(content: str) -> ExistingScan:
    data = parse_workflow(content)
    jobs = data.get("jobs")
    if data.get("name") != "Skill checks" or not isinstance(jobs, dict) or len(jobs) != 1:
        raise SetupError("Automatic reporting expects a 'Skill checks' workflow with one reusable scan job.")
    job_id, job = next(iter(jobs.items()))
    if not isinstance(job, dict) or "strategy" in job or "secrets" in job:
        raise SetupError("Matrix or secret-passing scan callers require manual reporting setup.")
    uses = job.get("uses")
    match = WORKFLOW.fullmatch(uses) if isinstance(uses, str) else None
    if not match:
        raise SetupError("The existing scan must call a full-SHA workflow or an approved version tag such as @v3.")
    inputs = job.get("with", {})
    if not isinstance(inputs, dict) or not isinstance(inputs.get("paths"), str) or "${{" in inputs["paths"]:
        raise SetupError("The existing scan must specify literal repository-relative skill folders in 'paths'.")
    roots = list(dict.fromkeys(safe_relative(path.strip()) for path in inputs["paths"].splitlines() if path.strip()))
    score, mode = inputs.get("min-score", 70), inputs.get("scan-mode", "auto")
    if not roots or type(score) is not int or not 0 <= score <= 100 or mode not in {"auto", "all"}:
        raise SetupError("The existing scan has unsupported paths, minimum score, or scan mode.")
    direct = uses.split("@")[0] == EVALUATOR
    if direct and not re.fullmatch(r"[0-9a-f]{40}", match["ref"]):
        raise SetupError("Direct evaluator workflows must retain an immutable full commit SHA.")
    return ExistingScan(
        content, data, job_id, uses, roots, score, mode, match["ref"] if direct else None, None if direct else uses
    )
