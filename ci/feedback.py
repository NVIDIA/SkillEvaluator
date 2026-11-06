# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001, T201 -- trusted CI entrypoint with fixed diagnostics.
"""Comment on current PR/MR evidence without executing artifacts or source code.

GitHub authenticates the producer through its signed reusable-job receipt.
GitLab authenticates this protected downstream job through its own ID token,
then binds its source job, producer configuration and exact aggregate job.
All artifact processing is bounded in memory; HTML is never interpreted.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import http.client
import io
import json
import os
import re
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

# Isolated Python omits the script directory. Trust only this pinned helper
# checkout for sibling imports, never the consumer working directory.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import github_api
    import publish_reports
    from comment_summary import render_comment_summary
    from summary_metrics import summarize_results
    from tier_manifest import tier3_execution_complete, validate_manifest
    from verify_provenance import ProvenanceError, verify_receipt
except ModuleNotFoundError:
    from ci.comment_summary import render_comment_summary
    from ci.summary_metrics import summarize_results
    from ci.tier_manifest import tier3_execution_complete, validate_manifest
    from ci.verify_provenance import ProvenanceError, verify_receipt

    from ci import github_api, publish_reports

MARKER = "<!-- skillevaluator-all-tiers-report -->"
RUN_MARKER = re.compile(r"<!-- skillevaluator-all-tiers-run:([1-9][0-9]*):([1-9][0-9]*) -->")
CALLER_PATH = ".github/workflows/skill-checks-all-tiers.yml"
MAX_BYTES = 64 * 1024 * 1024
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_BODY_BYTES = 60000
MAX_FILES = 1000
MAX_PAGES = 10
GL_AUDIENCE = "urn:skillevaluator:feedback:v1"


class FeedbackError(ValueError):
    """Fixed diagnostic containing no credentials or untrusted response text."""


def _require(condition, message):
    if not condition:
        raise FeedbackError(message)


def _positive(value):
    _require(type(value) is int and value > 0, "A positive API identity is required.")
    return value


def _json(raw):
    return github_api.strict_json(raw, FeedbackError, "Feedback evidence is invalid JSON.")


def _time(value):
    try:
        parsed = datetime.fromisoformat(value)
        _require(parsed.tzinfo is not None, "An API timestamp lacks its timezone.")
        return parsed
    except (ValueError, TypeError, AttributeError) as exc:
        raise FeedbackError("An API timestamp is invalid.") from exc


def _safe_path(value):
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= 4096
        and not any(c in value for c in "\\:%?#")
        and not value.startswith("/")
        and not any(ord(c) < 32 or ord(c) == 127 for c in value)
        and all(p not in {"", ".", ".."} for p in value.split("/"))
    )


def read_results(raw: bytes, *, prefix: str = "") -> dict:
    """Validate exact retained native JSON; never extract or interpret HTML."""
    _require(len(raw) <= MAX_BYTES, "The aggregate archive exceeds its size limit.")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = {}
            total = 0
            for entry in archive.infolist():
                name = entry.filename
                kind = stat.S_IFMT(entry.external_attr >> 16)
                _require(
                    not entry.flag_bits & 1 and kind in {0, stat.S_IFREG, stat.S_IFDIR},
                    "The aggregate archive contains a linked or encrypted member.",
                )
                _require(
                    _safe_path(name.rstrip("/") if entry.is_dir() else name),
                    "The aggregate archive contains an unsafe member path.",
                )
                _require(name not in members, "The aggregate archive contains duplicate members.")
                members[name] = entry
                total += entry.file_size
                _require(
                    len(members) <= MAX_FILES and 0 <= entry.file_size <= MAX_FILE_BYTES and total <= MAX_BYTES,
                    "The aggregate archive exceeds its member or expansion limits.",
                )

            def read(name):
                _require(name in members and not members[name].is_dir(), "Required retained evidence is missing.")
                with archive.open(members[name]) as source:
                    result = source.read(MAX_FILE_BYTES + 1)
                _require(len(result) <= MAX_FILE_BYTES, "Retained evidence exceeds its size limit.")
                return result

            result = validate_manifest(_json(read(prefix + "ci-results.json")))
            from skillevaluator.reporting.retained import retained_results
            from skillevaluator.reporting.run_status import resolve_run_status

            seen = set()
            for tier in result["tiers"]:
                # Comment counters come from validated native findings, never
                # from arbitrary producer-supplied manifest metrics.
                tier.pop("metrics", None)
                for skill in tier["skills"]:
                    skill.pop("metrics", None)
                    json_path, html_path = skill["report_json"], skill["report_html"]
                    if json_path is None:
                        _require(skill["status"] not in {"passed", "failed"}, "A verdict lacks its retained evidence.")
                        _require(html_path is None, "An HTML report lacks its retained JSON.")
                        continue
                    pattern = rf"tier{tier['tier']}/skill-[0-9]{{3}}/skillevaluator-tier{tier['tier']}\.json"
                    _require(
                        re.fullmatch(pattern, json_path) is not None and json_path not in seen,
                        "Native evidence does not have a unique canonical tier path.",
                    )
                    seen.add(json_path)
                    _require(
                        html_path == json_path.removesuffix(".json") + ".html",
                        "Native HTML and JSON evidence paths disagree.",
                    )
                    _require(
                        prefix + html_path in members and not members[prefix + html_path].is_dir(),
                        "The retained native HTML report is missing.",
                    )
                    native = _json(read(prefix + json_path))
                    retained = retained_results(native, copy_metadata=False, allowed_tiers={tier["tier"]})
                    status = resolve_run_status(retained).tiers[tier["tier"]].status.value
                    if tier["tier"] == 3 and status == "passed" and native.get("tier3", {}).get("verdict") == "neutral":
                        status = "neutral"
                    # A producer may downgrade a clean result after an execution
                    # error; it may never promote failed/incomplete evidence.
                    _require(
                        skill["status"] == status
                        or (skill["status"] == "incomplete" and status in {"passed", "not_run"}),
                        "The aggregate verdict contradicts retained native evidence.",
                    )
                    if tier["tier"] == 3 and skill["status"] == "passed":
                        _require(
                            tier3_execution_complete(native),
                            "Tier 3 passing evidence lacks complete paired execution.",
                        )
                    skill["metrics"] = summarize_results(retained, tier=tier["tier"], native=native)
                    if skill["status"] in {"incomplete", "skipped", "not_run", "needs_action"}:
                        skill["metrics"]["issue_count_known"] = False
                    if tier["tier"] == 3 and not tier3_execution_complete(native):
                        skill["metrics"]["issue_count_known"] = False
                        skill["metrics"].pop("score", None)
                        skill["metrics"].pop("skill_lift", None)
            return result
    except FeedbackError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise FeedbackError("The aggregate archive or retained evidence is invalid.") from exc


def _body(manifest, run_url, artifact_url, receipt):
    # Every skill links once to the combined report archive. Its members are
    # navigated through index.html after download, never as fabricated web URLs.
    rendered = render_comment_summary(manifest, artifact_url)
    body = (
        f"{MARKER}\n<!-- skillevaluator-all-tiers-run:{receipt[0]}:{receipt[1]} -->\n"
        f"[Evaluation run]({run_url})\n\n{rendered}\n\n"
        f"Commit: `{manifest['commit']}` · Environment: {manifest['environment']}\n"
    )
    _require(len(body.encode("utf-8")) <= MAX_BODY_BYTES, "The feedback comment exceeds its size limit.")
    return body


def _identity(comment):
    matches = RUN_MARKER.findall(comment["body"])
    _require(len(matches) == 1, "An owned comment lacks a unique producer receipt.")
    return tuple(int(n) for n in matches[0])


def _listing(api, path, *, key=None):
    rows = []
    for page in range(1, MAX_PAGES + 1):
        value = api.get(f"{path}{'&' if '?' in path else '?'}per_page=100&page={page}")
        items = value.get(key) if key and isinstance(value, dict) else value
        _require(
            isinstance(items, list) and len(items) <= 100 and all(isinstance(i, dict) for i in items),
            "An API listing has an unsupported shape.",
        )
        rows.extend(items)
        if len(items) < 100:
            return rows
    raise FeedbackError("An API listing exceeds its pagination limit.")


def _upsert(api, *, collection, member, owned, body, receipt, current, method="patch"):
    comments = [c for c in _listing(api, collection) if owned(c)]
    _require(len(comments) <= 1, "Multiple owned report comments require manual reconciliation.")
    if any(_identity(c) > receipt for c in comments):
        return {"comment_action": "skipped", "comment_reason": "A newer report comment already exists."}
    if not current():
        return {"comment_action": "skipped", "comment_reason": "The source head or producer attempt changed."}
    canonical = comments[0] if comments else None
    if canonical:
        identifier = _positive(canonical.get("id"))
        # Re-read ownership immediately before updating an existing note.
        check = api.get(member.format(id=identifier))
        _require(
            owned(check) and check.get("id") == identifier and check.get("body") == canonical["body"],
            "The report comment changed ownership or content before update.",
        )
        if _identity(check) > receipt:
            return {"comment_action": "skipped", "comment_reason": "A newer report comment already exists."}
        if canonical["body"] == body:
            action = "unchanged"
        else:
            response = getattr(api, method)(member.format(id=identifier), {"body": body})
            _require(response.get("id") == identifier, "The API did not acknowledge the expected comment update.")
            action = "updated"
    else:
        response = api.post(collection, {"body": body})
        identifier = _positive(response.get("id"))
        action = "created"
    check = api.get(member.format(id=identifier))
    _require(
        isinstance(check, dict) and check.get("id") == identifier and owned(check) and check.get("body") == body,
        "The comment readback does not match the expected bot-owned content.",
    )
    return {"comment_action": action, "comment_id": str(identifier)}


def _github_run(run, repository, run_id, attempt):
    _require(
        isinstance(run, dict)
        and run.get("id") == run_id
        and run.get("run_attempt") == attempt
        and run.get("repository", {}).get("full_name") == repository,
        "The run identity or latest attempt does not match the selected producer.",
    )
    _require(
        run.get("event") == "pull_request"
        and run.get("status") == "completed"
        and run.get("path") == CALLER_PATH
        and isinstance(run.get("head_sha"), str)
        and github_api.COMMIT_SHA.fullmatch(run["head_sha"]),
        "Only completed approved pull-request workflow runs can receive feedback.",
    )


def _github_artifacts(api, root, run):
    listing = api.get(f"{root}/actions/runs/{run['id']}/artifacts?per_page=100")
    _require(
        isinstance(listing, dict)
        and isinstance(listing.get("artifacts"), list)
        and type(listing.get("total_count")) is int
        and listing["total_count"] <= 100,
        "The aggregate artifact listing exceeds its limit.",
    )
    started, finished = _time(run.get("run_started_at")), _time(run.get("updated_at"))
    _require(started <= finished, "The producer attempt has an invalid time interval.")
    selected = []
    for prefix in ("skill-evaluator-results", "skill-evaluator-results-provenance"):
        matches = [
            a
            for a in listing["artifacts"]
            if isinstance(a, dict)
            and a.get("name") == f"{prefix}-attempt-{run['run_attempt']}"
            and a.get("expired") is False
        ]
        _require(len(matches) == 1, "The exact aggregate or provenance artifact is missing or ambiguous.")
        artifact = matches[0]
        _positive(artifact.get("id"))
        _require(
            artifact.get("workflow_run", {}).get("id") == run["id"]
            and artifact["workflow_run"].get("head_sha") == run["head_sha"]
            and started <= _time(artifact.get("created_at")) <= finished,
            "An artifact does not belong to the selected producer attempt and revision.",
        )
        _require(
            type(artifact.get("size_in_bytes")) is int
            and 0 < artifact["size_in_bytes"] <= (64 * 1024 if prefix.endswith("provenance") else MAX_BYTES),
            "An artifact exceeds its size limit.",
        )
        selected.append(artifact)
    return selected


def github_feedback(api, repository: str, run_id: int, attempt: int) -> dict:
    """Authenticate immutable producer identity and upsert current same-repo PR feedback."""
    _require(isinstance(repository, str) and github_api.REPOSITORY.fullmatch(repository), "Invalid GitHub repository.")
    _positive(run_id)
    _positive(attempt)
    root = f"/repos/{repository}"
    run = api.get(f"{root}/actions/runs/{run_id}")
    _github_run(run, repository, run_id, attempt)
    reference = os.environ.get("SKILL_EVAL_TRUSTED_WORKFLOW_REF", "")
    match = re.fullmatch(
        r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/\.github/workflows/[A-Za-z0-9_.-]+\.yml)@([0-9a-f]{40})", reference
    )
    _require(match is not None, "An immutable trusted reusable workflow reference is required.")
    workflow, sha = match.groups()
    references = run.get("referenced_workflows")
    _require(
        isinstance(references, list)
        and len(references) <= 32
        and sum(r.get("path") == reference and r.get("sha") == sha for r in references if isinstance(r, dict)) == 1,
        "The run did not use exactly the approved immutable reusable workflow.",
    )
    raw_pins = os.environ.get("SKILL_EVAL_TRUSTED_CALLER_BLOBS", "")
    pins = [s.strip() for s in re.split(r"[,\n]", raw_pins)]
    _require(
        0 < len(pins) <= 32 and all(github_api.COMMIT_SHA.fullmatch(p) for p in pins),
        "Reviewed caller Git blob identities are required.",
    )
    artifact, provenance = _github_artifacts(api, root, run)
    try:
        identity = verify_receipt(
            api.download_artifact(provenance["id"]),
            repository=repository,
            run=run,
            evaluator_workflow=workflow,
            evaluator_sha=sha,
            caller_workflow_path=CALLER_PATH,
        )
    except ProvenanceError as exc:
        raise FeedbackError("The producer execution receipt is invalid.") from exc
    caller = api.get(f"{root}/contents/{CALLER_PATH}?ref={identity['workflow_sha']}")
    _require(
        isinstance(caller, dict)
        and caller.get("type") == "file"
        and caller.get("path") == CALLER_PATH
        and caller.get("sha") in pins,
        "The executed caller workflow blob is not approved.",
    )
    manifest = read_results(api.download_artifact(artifact["id"]))
    _require(
        all(
            manifest[k] == v
            for k, v in {
                "repository": repository,
                "platform": "github",
                "commit": run["head_sha"],
                "run_id": str(run_id),
                "run_attempt": str(attempt),
            }.items()
        ),
        "The report manifest does not match the exact producer attempt and revision.",
    )
    candidates = run.get("pull_requests") or api.get(f"{root}/commits/{run['head_sha']}/pulls")
    _require(
        isinstance(candidates, list) and len(candidates) == 1 and isinstance(candidates[0], dict),
        "The producer must identify exactly one pull request.",
    )
    number = _positive(candidates[0].get("number"))

    def current():
        pr = api.get(f"{root}/pulls/{number}")
        latest = api.get(f"{root}/actions/runs/{run_id}")
        return (
            pr.get("state") == "open"
            and pr.get("head", {}).get("sha") == run["head_sha"]
            and all(pr.get(part, {}).get("repo", {}).get("full_name") == repository for part in ("head", "base"))
            and latest.get("run_attempt") == attempt
            and latest.get("head_sha") == run["head_sha"]
        )

    if not current():
        return {"comment_action": "skipped", "comment_reason": "The pull request head or producer attempt changed."}
    run_url = f"https://github.com/{repository}/actions/runs/{run_id}"
    body = _body(manifest, run_url, f"{run_url}/artifacts/{artifact['id']}", (run_id, attempt))
    bot = os.environ.get("SKILL_EVAL_BOT_LOGIN", "github-actions[bot]")
    _require(re.fullmatch(r"[A-Za-z0-9_-]+\[bot\]", bot), "The configured comment owner is not a bot login.")

    def owned(comment):
        return (
            isinstance(comment, dict)
            and comment.get("user", {}).get("login") == bot
            and comment["user"].get("type") == "Bot"
            and isinstance(comment.get("body"), str)
            and comment["body"].startswith(MARKER + "\n")
        )

    result = _upsert(
        api,
        collection=f"{root}/issues/{number}/comments",
        member=f"{root}/issues/comments/{{id}}",
        owned=owned,
        body=body,
        receipt=(run_id, attempt),
        current=current,
    )
    if "comment_id" in result:
        result["comment_url"] = f"https://github.com/{repository}/pull/{number}#issuecomment-{result['comment_id']}"
    return result


def _gitlab_origin(api_url):
    try:
        parsed = urllib.parse.urlsplit(api_url)
        _require(
            parsed.scheme == "https"
            and bool(parsed.hostname)
            and not parsed.username
            and not parsed.password
            and re.fullmatch(r"(?:/[A-Za-z0-9._~-]+)*/api/v4", parsed.path)
            and all(part not in {".", ".."} for part in parsed.path.split("/"))
            and not parsed.query
            and not parsed.fragment
            and "\\" not in api_url
            and not any(ord(c) < 33 for c in api_url),
            "GitLab requires an exact HTTPS API origin ending in /api/v4.",
        )
        _require(parsed.port is None or 1 <= parsed.port <= 65535, "Invalid GitLab API port.")
        return f"https://{parsed.netloc}" + parsed.path.removesuffix("/api/v4")
    except (ValueError, TypeError) as exc:
        raise FeedbackError("The GitLab API URL is invalid.") from exc


class GitLabAPI:
    """Bounded private-token API client; no redirects and no credential forwarding."""

    def __init__(self, api_url: str, project: str, token: str):
        self.origin = _gitlab_origin(api_url)
        self.api_url, self.project, self.token = api_url, project, token
        self.opener = urllib.request.build_opener(github_api._NoRedirect())

    def _call(self, method, path, payload=None, *, binary=False):
        # Resolve a human namespace once. All later operations use its numeric ID.
        allowed = path == "/user" or re.fullmatch(r"/projects/[A-Za-z0-9_.%/-]+(?:\?[^\s]*)?", path)
        _require(
            allowed
            and ".." not in path
            and "\\" not in path
            and not path.startswith("//")
            and not any(ord(c) < 33 for c in path),
            "Only relative GitLab project API paths are supported.",
        )
        headers = {
            "PRIVATE-TOKEN": self.token,
            "User-Agent": "SkillEvaluator-trusted-feedback",
            "Accept": "application/octet-stream" if binary else "application/json",
        }
        data = None
        if payload is not None:
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.api_url + path, headers=headers, data=data, method=method)
        with self.opener.open(request, timeout=github_api.TIMEOUT_SECONDS) as response:
            raw = github_api.bounded_read(
                response,
                MAX_BYTES if binary else github_api.MAX_API_BYTES,
                FeedbackError,
                "The GitLab API response exceeds its size limit.",
            )
        return raw if binary else _json(raw)

    def get(self, path):
        return self._call("GET", path)

    def post(self, path, payload):
        return self._call("POST", path, payload)

    def put(self, path, payload):
        return self._call("PUT", path, payload)

    def graphql(self, query: str, variables: dict) -> dict:
        """Read server-owned relationships at this instance's fixed GraphQL endpoint."""
        _require(
            isinstance(query, str) and 0 < len(query) <= 16384 and isinstance(variables, dict),
            "The GitLab relationship request is invalid.",
        )
        try:
            data = json.dumps({"query": query, "variables": variables}, allow_nan=False).encode("utf-8")
            _require(len(data) <= 64 * 1024, "The GitLab relationship request exceeds its size limit.")
            request = urllib.request.Request(
                self.origin + "/api/graphql",
                method="POST",
                data=data,
                headers={
                    "PRIVATE-TOKEN": self.token,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "SkillEvaluator-trusted-feedback",
                },
            )
            with self.opener.open(request, timeout=github_api.TIMEOUT_SECONDS) as response:
                _require(response.getcode() == 200, "The GitLab relationship response has an invalid HTTP status.")
                raw = github_api.bounded_read(
                    response,
                    github_api.MAX_API_BYTES,
                    FeedbackError,
                    "The GitLab relationship response exceeds its size limit.",
                )
            payload = _json(raw)
        except FeedbackError:
            raise
        except (http.client.HTTPException, OSError, TypeError, ValueError, RecursionError):
            raise FeedbackError("The GitLab relationship request could not be verified.") from None
        _require(
            isinstance(payload, dict)
            and ("errors" not in payload or payload["errors"] == [])
            and isinstance(payload.get("data"), dict),
            "The GitLab relationship response is incomplete or invalid.",
        )
        return payload["data"]

    def download_artifact(self, job_id):
        _positive(job_id)
        return self._call("GET", f"/projects/{self.project}/jobs/{job_id}/artifacts", binary=True)


def _gitlab_signing_keys(origin):
    # GitLab's documented fixed JWKS endpoint; never follow provider-controlled
    # discovery URLs or attach the private API token to a key fetch.
    request = urllib.request.Request(
        origin + "/oauth/discovery/keys",
        headers={"Accept": "application/json", "User-Agent": "SkillEvaluator-trusted-feedback"},
    )
    with urllib.request.build_opener(github_api._NoRedirect()).open(request, timeout=30) as response:
        keys = _json(github_api.bounded_read(response, 256 * 1024, FeedbackError))
    _require(
        isinstance(keys, dict) and isinstance(keys.get("keys"), list) and 1 <= len(keys["keys"]) <= 32,
        "GitLab returned an invalid signing key set.",
    )
    return keys["keys"]


def verify_gitlab_job(api_url: str, token: str) -> dict:
    """Authenticate the actual protected job independently of overrideable CI variables."""
    import jwt

    origin = _gitlab_origin(api_url)
    _require(isinstance(token, str) and 0 < len(token) <= 32768, "A bounded GitLab feedback job ID token is required.")
    try:
        header = jwt.get_unverified_header(token)
        _require(
            header.get("alg") == "RS256"
            and header.get("typ") == "JWT"
            and isinstance(header.get("kid"), str)
            and 0 < len(header["kid"]) <= 256,
            "The GitLab job token uses an unsupported signing header.",
        )
        matches = [k for k in _gitlab_signing_keys(origin) if isinstance(k, dict) and k.get("kid") == header["kid"]]
        _require(
            len(matches) == 1
            and matches[0].get("kty") == "RSA"
            and matches[0].get("alg", "RS256") == "RS256"
            and matches[0].get("use", "sig") == "sig",
            "The GitLab job signing key is unavailable.",
        )
        key = jwt.PyJWK.from_dict(matches[0], algorithm="RS256").key
        _require(2048 <= key.key_size <= 8192, "The GitLab job signing key has an unsupported size.")
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=GL_AUDIENCE,
            issuer=origin,
            options={"strict_aud": True, "require": ["exp", "iat", "nbf", "iss", "aud"]},
        )
        for field in ("project_id", "pipeline_id", "job_id"):
            _require(
                isinstance(claims.get(field), str) and re.fullmatch(r"[1-9][0-9]{0,19}", claims[field]),
                "The GitLab job token lacks an immutable execution identity.",
            )
        _require(
            claims.get("ref_protected") in (True, "true")
            and claims.get("ref_type") == "branch"
            and claims.get("pipeline_source") == "pipeline"
            and isinstance(claims.get("sha"), str)
            and github_api.COMMIT_SHA.fullmatch(claims["sha"])
            and isinstance(claims.get("ref"), str)
            and bool(claims["ref"]),
            "Only a protected downstream branch job may publish feedback.",
        )
    except FeedbackError:
        raise
    except (jwt.PyJWTError, KeyError, TypeError, ValueError, OSError, urllib.error.URLError) as exc:
        raise FeedbackError("The GitLab feedback job identity could not be authenticated.") from exc
    return {name: claims[name] for name in ("project_id", "pipeline_id", "job_id", "ref", "sha")}


def validate_gitlab_config(document):
    """Allow the installed pinned producer and the downstream-only local include.

    Matching only the root blob would otherwise leave transitive local includes
    mutable on the MR branch. The generated configuration has exactly these two
    includes; supporting other topologies requires an explicit trust contract.
    """
    import yaml

    if isinstance(document, bytes):
        _require(len(document) <= 1024 * 1024, "The producer CI configuration exceeds its size limit.")
        try:
            document = yaml.safe_load(document)
        except (ValueError, yaml.YAMLError, UnicodeError) as exc:
            raise FeedbackError("The producer CI configuration is invalid.") from exc
    _require(isinstance(document, dict), "The producer CI configuration must be an object.")
    includes = document.get("include")
    _require(
        isinstance(includes, list) and len(includes) == 2, "The producer config has unsupported transitive includes."
    )
    remote, local = [], []
    for item in includes:
        _require(isinstance(item, dict), "The producer config has an unsupported include.")
        if set(item) == {"remote"}:
            _require(
                isinstance(item["remote"], str)
                and re.fullmatch(
                    r"https://raw\.githubusercontent\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[0-9a-f]{40}/ci/gitlab/all-tiers\.yml",
                    item["remote"],
                ),
                "The producer config must include an immutable evaluator revision.",
            )
            remote.append(item)
        else:
            expected = '$CI_PIPELINE_SOURCE == "pipeline" && $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH && $SKILL_EVAL_FEEDBACK == "true"'
            _require(
                item == {"local": ".gitlab/skill-feedback.yml", "rules": [{"if": expected}]},
                "Local CI includes must be restricted to the trusted downstream feedback pipeline.",
            )
            local.append(item)
    _require(
        len(remote) == len(local) == 1, "The producer config must have one pinned evaluator and one feedback include."
    )


def gitlab_feedback(api, api_url: str, project: str, pipeline_id: int) -> dict:
    """Bind the trusted downstream job to one current same-project MR producer."""
    _positive(pipeline_id)
    origin = _gitlab_origin(api_url)
    _require(
        isinstance(project, str)
        and re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", project)
        and ".." not in project,
        "A valid GitLab project ID or namespace path is required.",
    )
    info = api.get("/projects/" + urllib.parse.quote(project, safe=""))
    project_id = _positive(info.get("id"))
    root = f"/projects/{project_id}"
    if hasattr(api, "project") and isinstance(api.project, str):
        api.project = str(project_id)
    identity = verify_gitlab_job(api_url, os.environ.get("SKILL_EVAL_GITLAB_ID_TOKEN", ""))
    _require(identity["project_id"] == str(project_id), "The authenticated job belongs to a different project.")
    default = info.get("default_branch")
    _require(
        isinstance(default, str)
        and identity["ref"] == default
        and info.get("ci_config_path") in (None, "", ".gitlab-ci.yml"),
        "The feedback job must use the protected default branch and default CI config path.",
    )
    publisher = api.get(f"{root}/pipelines/{identity['pipeline_id']}")
    _require(
        publisher.get("id") == int(identity["pipeline_id"])
        and publisher.get("sha") == identity["sha"]
        and publisher.get("ref") == default
        and publisher.get("source") == "pipeline",
        "The downstream pipeline does not match its authenticated protected job.",
    )
    branch = api.get(f"{root}/repository/branches/{urllib.parse.quote(default, safe='')}")
    _require(
        branch.get("protected") is True and branch.get("commit", {}).get("id") == identity["sha"],
        "The feedback pipeline is no longer the current protected default branch revision.",
    )
    job = api.get(f"{root}/jobs/{identity['job_id']}")
    _require(
        job.get("id") == int(identity["job_id"])
        and job.get("name") == "skill-evaluation-feedback"
        and job.get("pipeline", {}).get("id") == publisher["id"]
        and job.get("ref") == default
        and job.get("commit", {}).get("id") == identity["sha"]
        and job.get("status") == "running",
        "The authenticated job is not the exact running trusted feedback job.",
    )
    variables = api.get(f"{root}/pipelines/{publisher['id']}/variables")
    expected_variables = {"SKILL_EVAL_FEEDBACK": "true", "SKILL_EVAL_SOURCE_PIPELINE_ID": str(pipeline_id)}
    # GitLab adds an opaque file variable for API trigger parameters. It is
    # never parsed, executed or published; only the server-bound relation below
    # authenticates the source job and pipeline.
    _require(
        isinstance(variables, list)
        and len(variables) in {2, 3}
        and all(isinstance(v, dict) and isinstance(v.get("key"), str) for v in variables)
        and len({v["key"] for v in variables}) == len(variables)
        and expected_variables.keys() <= {v["key"] for v in variables}
        and all(
            (
                v["key"] in expected_variables
                and v.get("variable_type") == "env_var"
                and v.get("value") == expected_variables[v["key"]]
            )
            or (
                v["key"] == "TRIGGER_PAYLOAD"
                and v.get("variable_type") == "file"
                and isinstance(v.get("value"), str)
                and len(v["value"]) <= 64 * 1024
            )
            for v in variables
        ),
        "The downstream pipeline has unsupported forwarded variables.",
    )
    source = api.get(f"{root}/pipelines/{pipeline_id}")
    _require(
        source.get("id") == pipeline_id
        and source.get("source") == "merge_request_event"
        and isinstance(source.get("sha"), str)
        and github_api.COMMIT_SHA.fullmatch(source["sha"])
        and source.get("status") in {"running", "success", "failed", "canceled", "skipped"},
        "Only active or completed same-project merge-request producers may receive feedback.",
    )
    _require(
        api.get(f"{root}/pipelines/{pipeline_id}/variables") == [],
        "Source merge-request pipelines may not override trusted producer configuration through pipeline variables.",
    )
    namespace = info.get("path_with_namespace")
    _require(
        isinstance(namespace, str) and re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+", namespace),
        "The authenticated GitLab project has no valid namespace.",
    )
    relationship = api.graphql(
        "query($project:ID!,$pipeline:CiPipelineID!){project(fullPath:$project){fullPath "
        "pipeline(id:$pipeline){id project{fullPath} sourceJob{id} "
        "upstream{id project{fullPath} mergeRequest{iid sourceProjectId targetProjectId}}}}}",
        {"project": namespace, "pipeline": f"gid://gitlab/Ci::Pipeline/{publisher['id']}"},
    )
    relation_project = relationship.get("project") if isinstance(relationship, dict) else None
    relation = relation_project.get("pipeline") if isinstance(relation_project, dict) else None
    _require(
        isinstance(relation_project, dict)
        and relation_project.get("fullPath") == namespace
        and isinstance(relation, dict)
        and relation.get("id") == f"gid://gitlab/Ci::Pipeline/{publisher['id']}"
        and isinstance(relation.get("project"), dict)
        and relation["project"].get("fullPath") == namespace
        and isinstance(relation.get("upstream"), dict)
        and relation["upstream"].get("id") == f"gid://gitlab/Ci::Pipeline/{pipeline_id}"
        and isinstance(relation["upstream"].get("project"), dict)
        and relation["upstream"]["project"].get("fullPath") == namespace
        and isinstance(relation.get("sourceJob"), dict),
        "The publisher lacks a server-bound same-project source pipeline and job.",
    )
    source_job_gid = relation["sourceJob"].get("id")
    source_job_match = (
        re.fullmatch(r"gid://gitlab/Ci::Build/([1-9][0-9]{0,29})", source_job_gid)
        if isinstance(source_job_gid, str)
        else None
    )
    _require(source_job_match is not None, "The publisher source must be an ordinary authenticated request job.")
    request_job_id = int(source_job_match[1])

    # Read both configurations at immutable API revisions. A pipeline variable
    # allowlist is not a trust anchor because upstream variables can override it.
    def config(sha):
        value = api.get(f"{root}/repository/files/.gitlab-ci.yml?ref={sha}")
        _require(
            value.get("file_path") == ".gitlab-ci.yml"
            and isinstance(value.get("blob_id"), str)
            and github_api.COMMIT_SHA.fullmatch(value["blob_id"]),
            "The producer config lacks a Git blob identity.",
        )
        _require(
            value.get("encoding") == "base64"
            and isinstance(value.get("content"), str)
            and len(value["content"]) <= 2 * 1024 * 1024,
            "The CI config lacks bounded source content.",
        )
        try:
            validate_gitlab_config(base64.b64decode(value["content"].replace("\n", ""), validate=True))
        except (ValueError, binascii.Error) as exc:
            raise FeedbackError("The CI config source is invalid.") from exc
        return value["blob_id"]

    _require(
        config(source["sha"]) == config(identity["sha"]),
        "The producer CI config differs from the trusted default-branch config.",
    )
    linked_mr = relation["upstream"].get("mergeRequest")
    linked_iid = linked_mr.get("iid") if isinstance(linked_mr, dict) else None
    _require(
        isinstance(linked_mr, dict)
        and isinstance(linked_iid, str)
        and re.fullmatch(r"[1-9][0-9]{0,18}", linked_iid)
        and int(linked_iid) <= 2**63 - 1
        and type(linked_mr.get("sourceProjectId")) is int
        and linked_mr["sourceProjectId"] == project_id
        and type(linked_mr.get("targetProjectId")) is int
        and linked_mr["targetProjectId"] == project_id,
        "The producer must have one linked same-project merge request.",
    )
    number = int(linked_iid)

    def aggregate_job():
        jobs = _listing(api, f"{root}/pipelines/{pipeline_id}/jobs?include_retried=false")
        requests = [
            j for j in jobs if j.get("name") == "skill-evaluation-request-feedback" and j.get("retried") is not True
        ]
        request_job = api.get(f"{root}/jobs/{request_job_id}")
        _require(
            len(requests) == 1
            and requests[0].get("id") == request_job_id
            and request_job.get("id") == request_job_id
            and request_job.get("name") == "skill-evaluation-request-feedback"
            and request_job.get("retried") is not True
            and request_job.get("pipeline", {}).get("id") == pipeline_id
            and request_job.get("commit", {}).get("id") == source["sha"]
            and request_job.get("ref") == source.get("ref")
            and request_job.get("status") in {"running", "success", "failed", "canceled"},
            "The publisher source request job is not the exact current producer job.",
        )
        matches = [j for j in jobs if j.get("name") == "skill-evaluation-aggregate" and j.get("retried") is not True]
        _require(len(matches) == 1, "The producer has no unique current aggregate job.")
        value = matches[0]
        identifier = _positive(value.get("id"))
        _require(
            value.get("pipeline", {}).get("id") == pipeline_id
            and value.get("commit", {}).get("id") == source["sha"]
            and value.get("ref") == source.get("ref")
            and value.get("status") in {"success", "failed"}
            and value.get("artifacts_file", {}).get("filename")
            == f"skill-evaluator-results-{pipeline_id}-{identifier}.zip"
            and type(value.get("artifacts_file", {}).get("size")) is int
            and 0 < value["artifacts_file"]["size"] <= MAX_BYTES,
            "The exact aggregate job artifact identity is invalid.",
        )
        return value

    aggregate = aggregate_job()

    def current():
        mr = api.get(f"{root}/merge_requests/{number}")
        latest = aggregate_job()
        return (
            mr.get("state") == "opened"
            and mr.get("sha") == source["sha"]
            and mr.get("source_project_id") == project_id
            and mr.get("target_project_id") == project_id
            and isinstance(mr.get("source_branch"), str)
            and bool(mr["source_branch"])
            # Detached MR pipelines use the internal head ref on GitLab 17.8.
            and source["ref"] in (mr["source_branch"], f"refs/merge-requests/{number}/head")
            and mr.get("head_pipeline", {}).get("id") == pipeline_id
            and latest["id"] == aggregate["id"]
        )

    if not current():
        return {"comment_action": "skipped", "comment_reason": "The merge request head or producer job changed."}
    manifest = read_results(api.download_artifact(aggregate["id"]), prefix="skill-evaluator-results/")
    _require(
        all(
            manifest[k] == v
            for k, v in {
                "repository": info.get("path_with_namespace"),
                "platform": "gitlab",
                "commit": source["sha"],
                "run_id": str(pipeline_id),
                "run_attempt": "1",
            }.items()
        ),
        "The report manifest does not match the exact producer pipeline and revision.",
    )
    user = api.get("/user")
    bot_id = _positive(user.get("id"))
    _require(user.get("bot") is True, "The scoped feedback token must belong to a bot account.")
    web = info.get("web_url")
    _require(
        isinstance(web, str) and web == origin + "/" + info.get("path_with_namespace", ""),
        "The GitLab project web URL does not match the authenticated API origin.",
    )
    # GitLab removes the renderer's terminal LF when storing a note.
    body = _body(
        manifest,
        f"{web}/-/pipelines/{pipeline_id}",
        f"{web}/-/jobs/{aggregate['id']}/artifacts/download",
        (pipeline_id, aggregate["id"]),
    ).removesuffix("\n")

    def owned(note):
        return (
            isinstance(note, dict)
            and note.get("author", {}).get("id") == bot_id
            and isinstance(note.get("body"), str)
            and note["body"].startswith(MARKER + "\n")
        )

    result = _upsert(
        api,
        collection=f"{root}/merge_requests/{number}/notes",
        member=f"{root}/merge_requests/{number}/notes/{{id}}",
        owned=owned,
        body=body,
        receipt=(pipeline_id, aggregate["id"]),
        current=current,
        method="put",
    )
    if "comment_id" in result:
        result["comment_url"] = f"{web}/-/merge_requests/{number}#note_{result['comment_id']}"
    return result


def _failure_diagnostic(error: BaseException) -> str:
    """Expose only trusted code locations and a finite set of failure labels."""
    line = main.__code__.co_firstlineno
    trace = error.__traceback__
    for _ in range(128):
        if trace is None:
            break
        frame = trace.tb_frame
        if frame.f_globals is globals() and frame.f_code.co_filename == __file__ and frame.f_code.co_name != "_require":
            line = trace.tb_lineno
        trace = trace.tb_next

    if isinstance(error, urllib.error.HTTPError):
        category = "http"
    elif isinstance(error, (FeedbackError, publish_reports.PublicationError)):
        category = "validation"
    elif isinstance(error, (OSError, http.client.HTTPException)):
        category = "transport"
    else:
        category = "response"

    # JWT verification already imports this trusted dependency. Match actual
    # classes, never an exception's printable name, message, notes or values.
    jwt = sys.modules.get("jwt")
    cause = error
    for _ in range(8):
        if isinstance(cause, urllib.error.HTTPError):
            codes = {401: "http_401", 403: "http_403", 404: "http_404", 429: "http_429"}
            category = codes.get(cause.code, "http") if type(cause.code) is int else "http"
            return f"check=feedback.py:{line}; category={category}"
        if jwt is not None:
            for kind, label in (
                (jwt.ExpiredSignatureError, "auth_expired"),
                (jwt.InvalidAudienceError, "auth_audience"),
                (jwt.InvalidSignatureError, "auth_signature"),
                (jwt.InvalidIssuerError, "auth_issuer"),
                (jwt.ImmatureSignatureError, "auth_immature"),
                (jwt.MissingRequiredClaimError, "auth_missing_claim"),
            ):
                if isinstance(cause, kind):
                    return f"check=feedback.py:{line}; category={label}"
            if isinstance(cause, jwt.PyJWTError):
                category = "auth_invalid"
        cause = cause.__cause__
        if cause is None:
            break
    return f"check=feedback.py:{line}; category={category}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("github", "gitlab"), required=True)
    parser.add_argument("--repository")
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--run-attempt", type=int)
    parser.add_argument("--api-url")
    parser.add_argument("--project-id")
    parser.add_argument("--pipeline-id", type=int)
    args = parser.parse_args(argv)
    try:
        if args.platform == "github":
            token = os.environ.get("GITHUB_TOKEN", "").strip()
            _require(bool(token), "GITHUB_TOKEN is required for trusted PR feedback.")
            result = github_feedback(
                publish_reports.GitHubAPI(args.repository, token), args.repository, args.run_id, args.run_attempt
            )
        else:
            token = os.environ.get("SKILLEVALUATOR_GITLAB_TOKEN", "").strip()
            _require(bool(token), "A scoped SKILLEVALUATOR_GITLAB_TOKEN is required; CI_JOB_TOKEN cannot post notes.")
            result = gitlab_feedback(
                GitLabAPI(args.api_url, args.project_id, token), args.api_url, args.project_id, args.pipeline_id
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (
        FeedbackError,
        publish_reports.PublicationError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        urllib.error.URLError,
        http.client.HTTPException,
    ) as error:
        # Do not print provider errors, request URLs, tokens or artifact content.
        print(
            "Trusted feedback could not validate its producer, current head, or bot-owned comment. "
            f"[{_failure_diagnostic(error)}]",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
