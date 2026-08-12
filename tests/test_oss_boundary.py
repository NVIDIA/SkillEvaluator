# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mutation-style contract tests for the public-release boundary scanner."""

from __future__ import annotations

import importlib.util
import json
import shutil
import stat
import struct
import subprocess
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNER_PATH = REPO_ROOT / "scripts" / "check_oss_boundary.py"
ALLOWLIST_PATH = REPO_ROOT / "config" / "oss_boundary_allowlist.json"


def _scanner() -> ModuleType:
    assert SCANNER_PATH.is_file(), "the OSS boundary scanner must be checked in"
    spec = importlib.util.spec_from_file_location("check_oss_boundary", SCANNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _joined(*parts: str) -> str:
    return "".join(parts)


MUTATIONS = (
    ("private-repository-host", _joined("https://gitlab", "-master.nvidia.com/group/repo")),
    ("private-package-host", _joined("https://urm.", "nvidia.com/artifactory/api/pypi/private/simple")),
    ("private-package-host", _joined("https://artifactory.", "nvidia.net/api/pypi/team/simple")),
    ("private-package-host", _joined("https://pypi.", "nvidia.com/simple")),
    ("private-package-host", _joined("nv-shared", "-pypi")),
    ("internal-nvidia-credential", _joined("NVI", "DIA_INFERENCE_KEY=secret")),
    ("internal-nvidia-credential", _joined('"NVIDIA"', ' + "_INFERENCE_KEY"')),
    ("internal-nvidia-credential", _joined("nvidia_", "inference_key=secret")),
    ("inference-service-integration", _joined("https://inference-api.", "nvidia.com/v1")),
    ("inference-service-integration", _joined("https://inference", "-hub.nvidia.com/v1")),
    ("inference-service-integration", _joined("Use Inference", " Hub for execution")),
    ("inference-service-integration", _joined("SKILLSPECTOR_PROVIDER=nv_", "inference")),
    ("execution-service-integration", _joined("ASTRA", "_API_KEY=secret")),
    ("execution-service-integration", _joined("astra", "-skill-eval")),
    ("execution-service-integration", _joined("Astra Harbor", " Hub")),
    ("deployment-integration", _joined("OPENSHIFT", "_NAMESPACE=private-project")),
    ("deployment-integration", _joined("CORPORATE", "_DEPLOYMENT_HOOK=https://deploy.invalid")),
    ("active-internal-token", _joined("OPENSHIFT", "_TOKEN=private-token")),
    ("active-internal-token", _joined("HARBOR_VIEWER", "_TOKEN=private-token")),
    ("internal-observability", _joined("INTERNAL", "_TELEMETRY_ENDPOINT=https://metrics.invalid")),
    ("internal-runtime-dependency", _joined("py", "mil", "vus==2.6.0")),
    ("internal-runtime-dependency", _joined("mil", "vus==2.6.0")),
    ("internal-runtime-dependency", _joined("sandbox", "-k8s>=0.1")),
    ("internal-product-name", _joined("NVC", "ARPS")),
    ("internal-product-name", _joined("NV", "-ACES")),
    ("internal-product-name", _joined("NV", "-BASE")),
    ("internal-product-name", _joined("IPP", "Bot")),
    ("harbor-upload-integration", _joined("harbor", "-viewer-upload==1.2.3")),
    ("harbor-upload-integration", _joined("https://harbor", "-viewer.corp.invalid/jobs/1")),
)


@pytest.mark.parametrize(("rule", "mutation"), MUTATIONS)
def test_every_denied_boundary_mutation_produces_a_path_rule_and_line(rule: str, mutation: str) -> None:
    scanner = _scanner()

    findings = scanner.scan_text("src/mutated.py", f"public line\n{mutation}\n")

    assert [(finding.path, finding.rule, finding.line) for finding in findings] == [("src/mutated.py", rule, 2)]


def test_mutations_cover_every_scanner_rule() -> None:
    scanner = _scanner()

    assert {rule for rule, _mutation in MUTATIONS} == {rule.rule_id for rule in scanner.DENY_RULES}


def test_scanner_source_does_not_match_its_constructed_deny_expressions() -> None:
    scanner = _scanner()

    assert scanner.scan_text("scripts/check_oss_boundary.py", SCANNER_PATH.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "public_text",
    (
        "NVIDIA_API_KEY=nvapi-public-placeholder",
        "https://integrate.api.nvidia.com/v1",
        "https://github.com/NVIDIA/SkillSpector.git",
        'harbor_viewer = {"job_url": "https://viewer.example/jobs/1"}',
        "opentelemetry-api>=1.37.0",
        "OTEL_EXPORTER_OTLP_ENDPOINT=https://collector.example.test",
        "internal implementation detail",
        "Astra is a generic project name without an execution integration",
        _joined("re.compile(r'(OPENSHIFT", "_TOKEN=)[^ ]+')"),
        _joined("OPENSHIFT", "_TOKEN_RE = re.compile('sha256')"),
        _joined("HARBOR_VIEWER", "_TOKEN_RE = re.compile('token')"),
    ),
)
def test_generic_public_and_defensive_text_is_not_flagged(public_text: str) -> None:
    assert _scanner().scan_text("src/public.py", public_text) == []


def test_repository_scan_uses_tracked_scopes_and_stable_allowlist_anchor(tmp_path: Path) -> None:
    scanner = _scanner()
    source = tmp_path / "tests" / "test_negative_fixture.py"
    source.parent.mkdir()
    denied_line = _joined("NVI", "DIA_INFERENCE_KEY=negative-fixture")
    source.write_text(f"{denied_line}  # oss-boundary-anchor: fixture-credential\n", encoding="utf-8")
    ignored = tmp_path / "scratch" / "ignored.txt"
    ignored.parent.mkdir()
    ignored.write_text(_joined("NVI", "DIA_INFERENCE_KEY=out-of-scope\n"), encoding="utf-8")
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "version": 2,
                "allowlist": [
                    {
                        "path": "tests/test_negative_fixture.py",
                        "rule": "internal-nvidia-credential",
                        "anchor": "fixture-credential",
                        "reason": "mutation fixture proves the deny rule fires",
                        "expires": "2099-12-31",
                        "review_owner": "oss-release-reviewers",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "tests/test_negative_fixture.py"], cwd=tmp_path, check=True)

    assert scanner.scan_repository(tmp_path, allowlist_path=allowlist) == []

    source.write_text(
        f"unrelated_public_line = True\n{denied_line}  # oss-boundary-anchor: fixture-credential\n",
        encoding="utf-8",
    )

    assert scanner.scan_repository(tmp_path, allowlist_path=allowlist) == []


def _write_anchor_allowlist(
    path: Path,
    *,
    allow_path: str = "tests/test_negative.py",
    rule: str = "internal-nvidia-credential",
    anchor: str = "fixture-credential",
) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "allowlist": [
                    {
                        "path": allow_path,
                        "rule": rule,
                        "anchor": anchor,
                        "reason": "negative fixture proves the deny rule fires",
                        "expires": "2099-12-31",
                        "review_owner": "oss-release-reviewers",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_repository_allowlist_requires_a_real_python_comment_anchor(tmp_path: Path) -> None:
    scanner = _scanner()
    source = tmp_path / "tests" / "test_negative.py"
    source.parent.mkdir()
    source.write_text(
        'marker = "# oss-boundary-anchor: fixture-credential"\n'
        + _joined("NVI", "DIA_INFERENCE_KEY=negative-fixture\n"),
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "tests/test_negative.py"], cwd=tmp_path, check=True)

    with pytest.raises(ValueError, match=r"allowlist anchor .* is missing"):
        scanner.scan_repository(tmp_path, allowlist_path=_write_anchor_allowlist(tmp_path / "allowlist.json"))


def test_repository_allowlist_rejects_duplicate_comment_anchors(tmp_path: Path) -> None:
    scanner = _scanner()
    denied_line = _joined("NVI", "DIA_INFERENCE_KEY=negative-fixture")
    source = tmp_path / "tests" / "test_negative.py"
    source.parent.mkdir()
    source.write_text(
        f"{denied_line}  # oss-boundary-anchor: fixture-credential\n"
        f"{denied_line}  # oss-boundary-anchor: fixture-credential\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "tests/test_negative.py"], cwd=tmp_path, check=True)

    with pytest.raises(ValueError, match="duplicate OSS boundary anchor"):
        scanner.scan_repository(tmp_path, allowlist_path=_write_anchor_allowlist(tmp_path / "allowlist.json"))


def test_repository_allowlist_anchor_must_match_exactly_one_configured_rule(tmp_path: Path) -> None:
    scanner = _scanner()
    source = tmp_path / "tests" / "test_negative.py"
    source.parent.mkdir()
    source.write_text(
        _joined("NVI", "DIA_INFERENCE_KEY=negative-fixture") + "  # oss-boundary-anchor: fixture-credential\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "tests/test_negative.py"], cwd=tmp_path, check=True)

    with pytest.raises(ValueError, match=r"exactly one internal-product-name finding.*found 0 matching"):
        scanner.scan_repository(
            tmp_path,
            allowlist_path=_write_anchor_allowlist(
                tmp_path / "allowlist.json",
                rule="internal-product-name",
            ),
        )


def test_repository_allowlist_rejects_two_same_rule_matches_on_the_anchor_line(tmp_path: Path) -> None:
    scanner = _scanner()
    credential = _joined("NVI", "DIA_INFERENCE_KEY")
    source = tmp_path / "tests" / "test_negative.py"
    source.parent.mkdir()
    source.write_text(
        f"first = '{credential}'; second = '{credential}'  # oss-boundary-anchor: fixture-credential\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "tests/test_negative.py"], cwd=tmp_path, check=True)

    with pytest.raises(ValueError, match=r"exactly one internal-nvidia-credential finding.*found 2"):
        scanner.scan_repository(tmp_path, allowlist_path=_write_anchor_allowlist(tmp_path / "allowlist.json"))


def test_repository_allowlist_counts_separately_reconstructed_matches_on_one_line(tmp_path: Path) -> None:
    scanner = _scanner()
    credential_expression = _joined("'NVI'", " + 'DIA'", " + '_INFERENCE_KEY'")
    source = tmp_path / "tests" / "test_negative.py"
    source.parent.mkdir()
    source.write_text(
        f"first = {credential_expression}; second = {credential_expression}  "
        "# oss-boundary-anchor: fixture-credential\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "tests/test_negative.py"], cwd=tmp_path, check=True)

    with pytest.raises(ValueError, match=r"exactly one internal-nvidia-credential finding.*found 2"):
        scanner.scan_repository(tmp_path, allowlist_path=_write_anchor_allowlist(tmp_path / "allowlist.json"))


def test_repository_allowlist_rejects_other_rule_matches_on_the_anchor_line(tmp_path: Path) -> None:
    scanner = _scanner()
    credential = _joined("NVI", "DIA_INFERENCE_KEY")
    dependency = _joined("py", "mil", "vus")
    source = tmp_path / "tests" / "test_negative.py"
    source.parent.mkdir()
    source.write_text(
        f"first = '{credential}'; second = '{dependency}'  # oss-boundary-anchor: fixture-credential\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "tests/test_negative.py"], cwd=tmp_path, check=True)

    with pytest.raises(ValueError, match=r"exactly one internal-nvidia-credential finding.*2 total"):
        scanner.scan_repository(tmp_path, allowlist_path=_write_anchor_allowlist(tmp_path / "allowlist.json"))


def test_repository_allowlist_anchor_must_share_the_finding_line(tmp_path: Path) -> None:
    scanner = _scanner()
    source = tmp_path / "tests" / "test_negative.py"
    source.parent.mkdir()
    source.write_text(
        "public_value = True  # oss-boundary-anchor: fixture-credential\n"
        + _joined("NVI", "DIA_INFERENCE_KEY=negative-fixture\n"),
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "tests/test_negative.py"], cwd=tmp_path, check=True)

    with pytest.raises(ValueError, match=r"exactly one internal-nvidia-credential finding.*found 0 matching"):
        scanner.scan_repository(tmp_path, allowlist_path=_write_anchor_allowlist(tmp_path / "allowlist.json"))


def test_repository_allowlist_anchor_must_follow_code_on_the_same_line(tmp_path: Path) -> None:
    scanner = _scanner()
    source = tmp_path / "tests" / "test_negative.py"
    source.parent.mkdir()
    source.write_text("# oss-boundary-anchor: fixture-credential\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "tests/test_negative.py"], cwd=tmp_path, check=True)

    with pytest.raises(ValueError, match="must follow code"):
        scanner.scan_repository(tmp_path, allowlist_path=_write_anchor_allowlist(tmp_path / "allowlist.json"))


def test_repository_scan_rejects_a_malformed_anchor_comment(tmp_path: Path) -> None:
    scanner = _scanner()
    source = tmp_path / "tests" / "test_negative.py"
    source.parent.mkdir()
    source.write_text(
        _joined("NVI", "DIA_INFERENCE_KEY=negative-fixture") + "  # oss-boundary-anchor: Fixture_Credential\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "tests/test_negative.py"], cwd=tmp_path, check=True)

    with pytest.raises(ValueError, match="malformed OSS boundary anchor"):
        scanner.scan_repository(tmp_path, allowlist_path=_write_anchor_allowlist(tmp_path / "allowlist.json"))


def test_repository_scan_fails_closed_when_anchor_tokenization_fails(tmp_path: Path) -> None:
    scanner = _scanner()
    source = tmp_path / "tests" / "test_negative.py"
    source.parent.mkdir()
    source.write_text(
        _joined("credential = 'NVI'", " + 'DIA'", " + '_INFERENCE_KEY'")
        + "  # oss-boundary-anchor: fixture-credential\n"
        + "unterminated = '''\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "tests/test_negative.py"], cwd=tmp_path, check=True)

    with pytest.raises(ValueError, match="could not tokenize Python source"):
        scanner.scan_repository(tmp_path, allowlist_path=_write_anchor_allowlist(tmp_path / "allowlist.json"))


def test_repository_scan_fails_closed_when_anchored_python_cannot_be_parsed(tmp_path: Path) -> None:
    scanner = _scanner()
    credential = _joined("NVI", "DIA_INFERENCE_KEY")
    source = tmp_path / "tests" / "test_negative.py"
    source.parent.mkdir()
    source.write_text(
        f"first = '{credential}'; second = 'NVI' + 'DIA' + '_INFERENCE_KEY'  "
        "# oss-boundary-anchor: fixture-credential\n"
        "broken =\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "tests/test_negative.py"], cwd=tmp_path, check=True)

    with pytest.raises(ValueError, match="could not parse anchored Python source"):
        scanner.scan_repository(tmp_path, allowlist_path=_write_anchor_allowlist(tmp_path / "allowlist.json"))


def test_repository_scan_rejects_duplicate_anchor_ids_across_files(tmp_path: Path) -> None:
    scanner = _scanner()
    tests = tmp_path / "tests"
    tests.mkdir()
    denied_line = _joined("NVI", "DIA_INFERENCE_KEY=negative-fixture")
    anchored_line = f"{denied_line}  # oss-boundary-anchor: fixture-credential\n"
    (tests / "test_negative.py").write_text(anchored_line, encoding="utf-8")
    (tests / "test_other.py").write_text(anchored_line, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "tests/test_negative.py", "tests/test_other.py"], cwd=tmp_path, check=True)

    with pytest.raises(ValueError, match="duplicate OSS boundary anchor"):
        scanner.scan_repository(tmp_path, allowlist_path=_write_anchor_allowlist(tmp_path / "allowlist.json"))


def test_allowlist_rejects_legacy_line_number_schema(tmp_path: Path) -> None:
    scanner = _scanner()
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "version": 1,
                "allowlist": [
                    {
                        "path": "tests/test_negative.py",
                        "rule": "internal-nvidia-credential",
                        "line": 1,
                        "reason": "legacy line-number exception",
                        "expires": "2099-12-31",
                        "review_owner": "oss-release-reviewers",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="version=2"):
        scanner.load_allowlist(allowlist)


def test_allowlist_rejects_anchor_ids_that_match_a_denied_rule(tmp_path: Path) -> None:
    scanner = _scanner()
    denied_anchor = _joined("py", "mil", "vus")
    allowlist = _write_anchor_allowlist(tmp_path / "allowlist.json", anchor=denied_anchor)

    with pytest.raises(ValueError, match="anchor ID matches denied OSS boundary rule"):
        scanner.load_allowlist(allowlist)


def test_allowlist_rejects_missing_reason_and_unknown_metadata(tmp_path: Path) -> None:
    scanner = _scanner()
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "version": 2,
                "allowlist": [
                    {
                        "path": "tests/test_negative.py",
                        "rule": "internal-nvidia-credential",
                        "anchor": "fixture-credential",
                        "ticket": "not-an-approved-field",
                        "expires": "2099-12-31",
                        "review_owner": "oss-release-reviewers",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="path, rule, anchor, reason, expires, and review_owner"):
        scanner.load_allowlist(allowlist)


def test_allowlist_rejects_duplicate_anchor_ids_across_paths(tmp_path: Path) -> None:
    scanner = _scanner()
    allowlist = tmp_path / "allowlist.json"
    common = {
        "rule": "internal-nvidia-credential",
        "anchor": "fixture-credential",
        "reason": "negative fixture proves the deny rule fires",
        "expires": "2099-12-31",
        "review_owner": "oss-release-reviewers",
    }
    allowlist.write_text(
        json.dumps(
            {
                "version": 2,
                "allowlist": [
                    {"path": "tests/test_first.py", **common},
                    {"path": "tests/test_second.py", **common},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicates anchor"):
        scanner.load_allowlist(allowlist)


@pytest.mark.parametrize(
    ("allow_path", "anchor"),
    (
        ("tests/negative.py", "fixture-credential"),
        ("tests/./test_negative.py", "fixture-credential"),
        ("tests/test_negative.py", "Fixture_Credential"),
        ("tests/test_negative.py", "fixture.credential"),
    ),
)
def test_allowlist_rejects_noncanonical_test_paths_and_anchor_ids(
    tmp_path: Path,
    allow_path: str,
    anchor: str,
) -> None:
    scanner = _scanner()
    allowlist = _write_anchor_allowlist(
        tmp_path / "allowlist.json",
        allow_path=allow_path,
        anchor=anchor,
    )

    with pytest.raises(ValueError, match=r"valid path|negative test paths|anchor"):
        scanner.load_allowlist(allowlist)


def test_allowlist_rejects_exceptions_outside_negative_tests(tmp_path: Path) -> None:
    scanner = _scanner()
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "version": 2,
                "allowlist": [
                    {
                        "path": "src/runtime.py",
                        "rule": "internal-nvidia-credential",
                        "anchor": "fixture-credential",
                        "reason": "runtime exceptions must never be allowed",
                        "expires": "2099-12-31",
                        "review_owner": "oss-release-reviewers",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="negative test paths"):
        scanner.load_allowlist(allowlist)


def test_allowlist_rejects_expired_negative_test_exception(tmp_path: Path) -> None:
    scanner = _scanner()
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "version": 2,
                "allowlist": [
                    {
                        "path": "tests/test_negative.py",
                        "rule": "internal-nvidia-credential",
                        "anchor": "fixture-credential",
                        "reason": "expired mutation fixture",
                        "expires": "2026-07-07",
                        "review_owner": "oss-release-reviewers",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expired"):
        scanner.load_allowlist(allowlist)


def test_findings_include_a_safely_redacted_excerpt() -> None:
    scanner = _scanner()
    secret = "must-not-appear"
    credential = _joined("NVI", "DIA_INFERENCE_KEY")

    finding = scanner.scan_text("src/config.py", f"{credential}={secret}")[0]

    assert secret not in finding.excerpt
    assert finding.excerpt == credential


def test_finding_excerpt_redacts_a_quoted_secret_with_spaces() -> None:
    scanner = _scanner()
    secret = "highly sensitive passphrase"
    credential = _joined("NVI", "DIA_INFERENCE_KEY")

    finding = scanner.scan_text("src/config.py", f'{credential}="{secret}"')[0]

    assert secret not in finding.excerpt
    assert "sensitive" not in finding.excerpt
    assert finding.excerpt == credential


def test_finding_excerpt_redacts_authorization_header() -> None:
    scanner = _scanner()
    endpoint = _joined("https://inference-api.", "nvidia.com/v1")
    secret = "sensitive-bearer-token"

    finding = scanner.scan_text("src/client.py", f"{endpoint} Authorization: Bearer {secret}")[0]

    assert secret not in finding.excerpt
    assert finding.excerpt == _joined("inference-api", ".nvidia.com")


@pytest.mark.parametrize(
    "sensitive_context",
    (
        'Authorization = "Bearer {secret}"',
        'headers = {"Authorization": "Bearer {secret}"}',
        'auth_header = "Basic {secret}"',
        'cookie = "session={secret}"',
        'session = "{secret}"',
    ),
)
def test_finding_diagnostic_is_derived_only_from_the_denied_match_span(sensitive_context: str) -> None:
    scanner = _scanner()
    secret = "must-never-enter-scanner-output"
    endpoint = _joined("https://gitlab", "-master.nvidia.com/group/repo")

    rendered_context = sensitive_context.replace("{secret}", secret)
    finding = scanner.scan_text("src/client.py", f"{rendered_context}; {endpoint}")[0]

    assert secret not in finding.excerpt
    assert finding.excerpt == _joined("gitlab", "-master.nvidia.com")


@pytest.mark.parametrize("parameter", ("cookie", "session"))
def test_denied_url_match_redacts_cookie_and_session_query_values(parameter: str) -> None:
    scanner = _scanner()
    secret = "must-never-enter-scanner-output"
    endpoint = _joined("https://harbor", f"-viewer.example?{parameter}={secret}")

    finding = scanner.scan_text("config/release.conf", endpoint)[0]

    assert secret not in finding.excerpt
    assert finding.excerpt.endswith(f"?{parameter}=<redacted>")


def test_repository_scan_covers_all_tracked_github_configuration(tmp_path: Path) -> None:
    scanner = _scanner()
    github_config = tmp_path / ".github" / "ISSUE_TEMPLATE" / "config.yml"
    github_config.parent.mkdir(parents=True)
    github_config.write_text(_joined("index: nv-shared", "-pypi\n"), encoding="utf-8")

    findings = scanner.scan_repository(tmp_path)

    assert [(finding.path, finding.rule, finding.line) for finding in findings] == [
        (".github/ISSUE_TEMPLATE/config.yml", "private-package-host", 1)
    ]


@pytest.mark.parametrize("root_name", (".gitignore", "Makefile", "CITATION.cff"))
def test_repository_scan_covers_tracked_root_release_and_config_files(tmp_path: Path, root_name: str) -> None:
    scanner = _scanner()
    (tmp_path / root_name).write_text(_joined("index=nv-shared", "-pypi\n"), encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(_joined("nv-shared", "-pypi").encode())
    (tmp_path / ".coverage").write_bytes(_joined("nv-shared", "-pypi").encode())

    findings = scanner.scan_repository(tmp_path)

    assert [(finding.path, finding.rule, finding.line) for finding in findings] == [
        (root_name, "private-package-host", 1)
    ]


def test_repository_scan_covers_every_tracked_non_binary_file(tmp_path: Path) -> None:
    scanner = _scanner()
    mutation = _joined("index=nv-shared", "-pypi\n")
    tracked = (
        "build/release.custom",
        "dist/settings.conf",
        "examples/provider.settings",
    )
    for relative in tracked:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(mutation, encoding="utf-8")
    ignored_binary = tmp_path / "examples" / "logo.png"
    ignored_binary.write_bytes(mutation.encode())
    untracked = tmp_path / "scratch" / "untracked.conf"
    untracked.parent.mkdir()
    untracked.write_text(mutation, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", *tracked, "examples/logo.png"], cwd=tmp_path, check=True)

    findings = scanner.scan_repository(tmp_path)

    assert [(finding.path, finding.rule, finding.line) for finding in findings] == [
        (relative, "private-package-host", 1) for relative in tracked
    ]


def test_repository_scan_fails_closed_for_undecodable_tracked_unknown_file(tmp_path: Path) -> None:
    scanner = _scanner()
    unknown = tmp_path / "examples" / "payload.settings"
    unknown.parent.mkdir()
    unknown.write_bytes(b"\xff")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "examples/payload.settings"], cwd=tmp_path, check=True)

    with pytest.raises(ValueError, match="not UTF-8 text"):
        scanner.scan_repository(tmp_path)


def test_scanner_constant_folds_arbitrary_python_string_concatenation() -> None:
    scanner = _scanner()
    source = _joined("credential = 'NVI'", " + 'DIA'", " + '_INFERENCE_KEY'")

    findings = scanner.scan_text("src/config.py", source)

    assert [(finding.path, finding.rule, finding.line) for finding in findings] == [
        ("src/config.py", "internal-nvidia-credential", 1)
    ]


def test_scanner_constant_folds_adjacent_python_string_literals() -> None:
    scanner = _scanner()
    source = _joined("credential = 'NVI' 'DIA", "_INFERENCE_KEY'")

    findings = scanner.scan_text("src/config.py", source)

    assert [(finding.path, finding.rule, finding.line) for finding in findings] == [
        ("src/config.py", "internal-nvidia-credential", 1)
    ]


@pytest.mark.parametrize(
    "source",
    (
        _joined("credential = f\"{'NVI'}DIA", '_INFERENCE_KEY"'),
        _joined("credential = f\"{'NVI':s}DIA", '_INFERENCE_KEY"'),
        _joined("credential = f\"{'NVI'!s:s}DIA", '_INFERENCE_KEY"'),
        _joined("credential = ''.join(('NVI', 'DIA", "_INFERENCE_KEY'))"),
        _joined("credential = '{}{}'.format('NVI', 'DIA", "_INFERENCE_KEY')"),
        _joined("credential = '{:s}{}'.format('NVI', 'DIA", "_INFERENCE_KEY')"),
        _joined("credential = '{!s:s}{}'.format('NVI', 'DIA", "_INFERENCE_KEY')"),
    ),
)
def test_scanner_reconstructs_safe_constant_python_string_forms(source: str) -> None:
    scanner = _scanner()

    findings = scanner.scan_text("src/config.py", source)

    assert [(finding.path, finding.rule, finding.line) for finding in findings] == [
        ("src/config.py", "internal-nvidia-credential", 1)
    ]


@pytest.mark.parametrize(
    ("path", "source", "expected_rule"),
    (
        (
            "scripts/release.sh",
            _joined('credential=NVI"DIA', '_INFERENCE_KEY"'),
            "internal-nvidia-credential",
        ),
        (
            "scripts/release",
            _joined('endpoint=https://gitlab"', '-master.nvidia.com"/group/repo'),
            "private-repository-host",
        ),
    ),
)
def test_scanner_reconstructs_shell_adjacent_quotes(path: str, source: str, expected_rule: str) -> None:
    scanner = _scanner()

    findings = scanner.scan_text(path, source)

    assert [(finding.path, finding.rule, finding.line) for finding in findings] == [(path, expected_rule, 1)]


@pytest.mark.parametrize("quote", ("'", '"'))
def test_scanner_reconstructs_bash_dollar_prefixed_static_quotes(quote: str) -> None:
    scanner = _scanner()
    source = _joined(f"credential=NVI${quote}DIA", f"_INFERENCE_KEY{quote}")

    findings = scanner.scan_text("scripts/release.sh", source)

    assert [(finding.path, finding.rule, finding.line) for finding in findings] == [
        ("scripts/release.sh", "internal-nvidia-credential", 1)
    ]


@pytest.mark.skipif(shutil.which("bash") is None, reason="Bash runtime is required for ANSI-C parity proof")
@pytest.mark.parametrize(
    ("assignment", "variable", "expected", "expected_rule"),
    (
        (
            _joined("credential=NVI$", "'\\x44IA", "_INFERENCE_KEY'"),
            "credential",
            _joined("NVI", "DIA_INFERENCE_KEY"),
            "internal-nvidia-credential",
        ),
        (
            _joined("credential=NVI$", "'\\u0044IA", "_INFERENCE_KEY'"),
            "credential",
            _joined("NVI", "DIA_INFERENCE_KEY"),
            "internal-nvidia-credential",
        ),
        (
            _joined("credential=NVI$", "'\\104IA", "_INFERENCE_KEY'"),
            "credential",
            _joined("NVI", "DIA_INFERENCE_KEY"),
            "internal-nvidia-credential",
        ),
        (
            _joined("endpoint=https://gitlab$", "'\\x2d", "master.nvidia.com'/repo"),
            "endpoint",
            _joined("https://gitlab", "-master.nvidia.com/repo"),
            "private-repository-host",
        ),
    ),
)
def test_bash_ansi_c_mutations_match_runtime_and_are_detected(
    assignment: str,
    variable: str,
    expected: str,
    expected_rule: str,
) -> None:
    scanner = _scanner()
    runtime = subprocess.run(
        ["bash", "-c", f'{assignment}; printf %s "${variable}"'],
        check=True,
        capture_output=True,
        text=True,
    )

    if "\\u" not in assignment or runtime.stdout == expected:
        assert runtime.stdout == expected
    else:
        bash_major = int(
            subprocess.run(
                ["bash", "-c", "printf %s $BASH_VERSINFO"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        assert bash_major < 4, "Modern Bash must decode the Unicode ANSI-C mutation"
    findings = scanner.scan_text("scripts/release.sh", assignment)
    assert [(finding.path, finding.rule, finding.line) for finding in findings] == [
        ("scripts/release.sh", expected_rule, 1)
    ]


@pytest.mark.skipif(shutil.which("bash") is None, reason="Bash runtime is required for ANSI-C parity proof")
def test_bash_ansi_c_common_escapes_match_runtime() -> None:
    scanner = _scanner()
    literal = r"$'line\ncolumn\tbackslash:\\ quote:\''"
    runtime = subprocess.run(
        ["bash", "-c", f"printf %s {literal}"],
        check=True,
        capture_output=True,
    )

    assert scanner._shell_normalized_value(literal).encode() == runtime.stdout


@pytest.mark.parametrize(
    "source",
    (
        _joined("value=$", "'unterminated"),
        _joined("value=$", r"'\x'"),
        _joined("value=$", r"'\q'"),
    ),
)
def test_scanner_fails_closed_for_malformed_or_unsupported_bash_ansi_c(source: str) -> None:
    scanner = _scanner()

    findings = scanner.scan_text("scripts/release.sh", source)

    assert [(finding.path, finding.rule, finding.line, finding.excerpt) for finding in findings] == [
        ("scripts/release.sh", "scanner-shell-decode-error", 1, "<shell reconstruction error>")
    ]


def test_scanner_fails_closed_when_bash_ansi_c_decode_limit_is_exceeded(monkeypatch) -> None:
    scanner = _scanner()
    monkeypatch.setattr(scanner, "_MAX_SHELL_DECODE_CHARS", 3)

    findings = scanner.scan_text("src/config.py", "command = \"value=$'four'\"")

    assert [(finding.path, finding.rule, finding.line, finding.excerpt) for finding in findings] == [
        ("src/config.py", "scanner-resource-limit", 1, "<shell reconstruction limit>")
    ]


@pytest.mark.parametrize(
    ("source", "expected_rule"),
    (
        (
            _joined("command = 'credential=NVI\"DIA", "_INFERENCE_KEY\"'"),
            "internal-nvidia-credential",
        ),
        (
            _joined("command = 'endpoint=https://gitlab\"", "-master.nvidia.com\"/repo'"),
            "private-repository-host",
        ),
        (
            _joined('command = "credential=NVI$', r"'\\x44IA", "_INFERENCE_KEY'\""),
            "internal-nvidia-credential",
        ),
    ),
)
def test_scanner_shell_normalizes_reconstructed_python_constants(source: str, expected_rule: str) -> None:
    scanner = _scanner()

    findings = scanner.scan_text("src/config.py", source)

    assert [(finding.path, finding.rule, finding.line) for finding in findings] == [("src/config.py", expected_rule, 1)]


@pytest.mark.parametrize(
    ("limit_name", "limit"),
    (
        ("_MAX_STATIC_STRING_DEPTH", 1),
        ("_MAX_STATIC_STRING_CHARS", 16),
    ),
)
def test_scanner_fails_closed_when_static_reconstruction_limit_is_exceeded(
    monkeypatch,
    limit_name: str,
    limit: int,
) -> None:
    scanner = _scanner()
    monkeypatch.setattr(scanner, limit_name, limit)
    source = _joined("credential = 'NVI' + ('DIA' + '", "_INFERENCE_KEY')")

    findings = scanner.scan_text("src/config.py", source)

    assert [(finding.path, finding.rule, finding.line, finding.excerpt) for finding in findings] == [
        ("src/config.py", "scanner-resource-limit", 1, "<static reconstruction limit>")
    ]


@pytest.mark.parametrize(
    "source",
    (
        _joined("OPENSHIFT", "_TOKEN = re.compile('anything')"),
        _joined("HARBOR_VIEWER", "_TOKEN: object = re.compile('anything')"),
    ),
)
def test_active_token_assignment_is_not_exempted_by_an_unrelated_regex(source: str) -> None:
    scanner = _scanner()

    findings = scanner.scan_text("src/config.py", source)

    assert [(finding.path, finding.rule, finding.line) for finding in findings] == [
        ("src/config.py", "active-internal-token", 1)
    ]


def test_multiline_defensive_regex_exempts_only_its_pattern_source_range() -> None:
    scanner = _scanner()
    source = _joined(
        "TOKEN_PATTERN = re.compile(\n",
        "    r'(OPENSHIFT",
        "_TOKEN=)[^ ]+',\n",
        "    re.IGNORECASE,\n",
        ")\n",
    )

    assert scanner.scan_text("src/patterns.py", source) == []


def test_wheel_and_sdist_members_are_scanned_without_extraction(tmp_path: Path) -> None:
    scanner = _scanner()
    mutation = _joined("pym", "ilvus==2.6.0")
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("package/module.py", f"public\n{mutation}\n")

    sdist = tmp_path / "package-1.0.tar.gz"
    data = f"public\n{mutation}\n".encode()
    info = tarfile.TarInfo("package-1.0/pyproject.toml")
    info.size = len(data)
    with tarfile.open(sdist, "w:gz") as archive:
        archive.addfile(info, BytesIO(data))

    wheel_findings = scanner.scan_archive(wheel)
    sdist_findings = scanner.scan_archive(sdist)

    assert [(finding.path, finding.rule, finding.line) for finding in wheel_findings] == [
        (f"{wheel}!package/module.py", "internal-runtime-dependency", 2)
    ]
    assert [(finding.path, finding.rule, finding.line) for finding in sdist_findings] == [
        (f"{sdist}!package-1.0/pyproject.toml", "internal-runtime-dependency", 2)
    ]


def test_wheel_and_sdist_scan_unknown_text_settings_files(tmp_path: Path) -> None:
    scanner = _scanner()
    mutation = _joined("index=nv-shared", "-pypi\n")
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("package/settings.conf", mutation)
    sdist = tmp_path / "package-1.0.tar.gz"
    data = mutation.encode()
    info = tarfile.TarInfo("package-1.0/settings.conf")
    info.size = len(data)
    with tarfile.open(sdist, "w:gz") as archive:
        archive.addfile(info, BytesIO(data))

    wheel_findings = scanner.scan_archive(wheel)
    sdist_findings = scanner.scan_archive(sdist)

    assert [(finding.path, finding.rule, finding.line) for finding in wheel_findings] == [
        (f"{wheel}!package/settings.conf", "private-package-host", 1)
    ]
    assert [(finding.path, finding.rule, finding.line) for finding in sdist_findings] == [
        (f"{sdist}!package-1.0/settings.conf", "private-package-host", 1)
    ]


def test_archive_scan_fails_closed_for_undecodable_unknown_member(tmp_path: Path) -> None:
    scanner = _scanner()
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("package/settings.conf", b"\xff")

    with pytest.raises(ValueError, match="not UTF-8 text"):
        scanner.scan_archive(wheel)


def test_sdist_negative_test_uses_exact_source_allowlist_path(tmp_path: Path) -> None:
    scanner = _scanner()
    source = (
        _joined("credential = 'NVI'", " + 'DIA'", " + '_INFERENCE_KEY'")
        + "  # oss-boundary-anchor: fixture-credential\n"
    ).encode()
    sdist = tmp_path / "package-1.0.tar.gz"
    info = tarfile.TarInfo("package-1.0/tests/test_negative.py")
    info.size = len(source)
    with tarfile.open(sdist, "w:gz") as archive:
        archive.addfile(info, BytesIO(source))
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "version": 2,
                "allowlist": [
                    {
                        "path": "tests/test_negative.py",
                        "rule": "internal-nvidia-credential",
                        "anchor": "fixture-credential",
                        "reason": "negative archive fixture",
                        "expires": "2099-12-31",
                        "review_owner": "oss-release-reviewers",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert scanner.scan_archive(sdist, allowlist_path=allowlist) == []


def test_sdist_validates_anchor_when_allowlisted_source_path_is_present(tmp_path: Path) -> None:
    scanner = _scanner()
    source = b"public fixture without the configured anchor\n"
    sdist = tmp_path / "package-1.0.tar.gz"
    info = tarfile.TarInfo("package-1.0/tests/test_negative.py")
    info.size = len(source)
    with tarfile.open(sdist, "w:gz") as archive:
        archive.addfile(info, BytesIO(source))

    with pytest.raises(ValueError, match=r"allowlist anchor .* is missing"):
        scanner.scan_archive(sdist, allowlist_path=_write_anchor_allowlist(tmp_path / "allowlist.json"))


def test_wheel_does_not_require_allowance_for_an_absent_source_path(tmp_path: Path) -> None:
    scanner = _scanner()
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("package/module.py", "public_value = True\n")

    assert scanner.scan_archive(wheel, allowlist_path=_write_anchor_allowlist(tmp_path / "allowlist.json")) == []


def test_wheel_uses_the_exact_allowlisted_source_path_and_anchor(tmp_path: Path) -> None:
    scanner = _scanner()
    source = (
        _joined("credential = 'NVI'", " + 'DIA'", " + '_INFERENCE_KEY'")
        + "  # oss-boundary-anchor: fixture-credential\n"
    )
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("tests/test_negative.py", source)

    assert scanner.scan_archive(wheel, allowlist_path=_write_anchor_allowlist(tmp_path / "allowlist.json")) == []


def test_wheel_requires_only_allowances_for_source_paths_present_in_the_artifact(tmp_path: Path) -> None:
    scanner = _scanner()
    allowlist = tmp_path / "allowlist.json"
    common = {
        "rule": "internal-nvidia-credential",
        "reason": "negative fixture proves the deny rule fires",
        "expires": "2099-12-31",
        "review_owner": "oss-release-reviewers",
    }
    allowlist.write_text(
        json.dumps(
            {
                "version": 2,
                "allowlist": [
                    {"path": "tests/test_negative.py", "anchor": "fixture-credential", **common},
                    {"path": "tests/test_absent.py", "anchor": "absent-credential", **common},
                ],
            }
        ),
        encoding="utf-8",
    )
    source = (
        _joined("credential = 'NVI'", " + 'DIA'", " + '_INFERENCE_KEY'")
        + "  # oss-boundary-anchor: fixture-credential\n"
    )
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("tests/test_negative.py", source)

    assert scanner.scan_archive(wheel, allowlist_path=allowlist) == []


def test_wheel_allowlist_rejects_two_same_rule_matches_on_the_anchor_line(tmp_path: Path) -> None:
    scanner = _scanner()
    credential = _joined("NVI", "DIA_INFERENCE_KEY")
    source = f"first = '{credential}'; second = '{credential}'  # oss-boundary-anchor: fixture-credential\n"
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("tests/test_negative.py", source)

    with pytest.raises(ValueError, match=r"exactly one internal-nvidia-credential finding.*found 2"):
        scanner.scan_archive(wheel, allowlist_path=_write_anchor_allowlist(tmp_path / "allowlist.json"))


def test_wheel_allowlist_counts_separately_reconstructed_matches_on_one_line(tmp_path: Path) -> None:
    scanner = _scanner()
    credential_expression = _joined("'NVI'", " + 'DIA'", " + '_INFERENCE_KEY'")
    source = (
        f"first = {credential_expression}; second = {credential_expression}  "
        "# oss-boundary-anchor: fixture-credential\n"
    )
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("tests/test_negative.py", source)

    with pytest.raises(ValueError, match=r"exactly one internal-nvidia-credential finding.*found 2"):
        scanner.scan_archive(wheel, allowlist_path=_write_anchor_allowlist(tmp_path / "allowlist.json"))


def test_wheel_rejects_the_wrong_anchor_at_an_allowlisted_source_path(tmp_path: Path) -> None:
    scanner = _scanner()
    source = (
        _joined("credential = 'NVI'", " + 'DIA'", " + '_INFERENCE_KEY'")
        + "  # oss-boundary-anchor: wrong-credential\n"
    )
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("tests/test_negative.py", source)

    with pytest.raises(ValueError, match="unregistered OSS boundary anchor 'wrong-credential'"):
        scanner.scan_archive(wheel, allowlist_path=_write_anchor_allowlist(tmp_path / "allowlist.json"))


def test_wheel_rejects_an_unregistered_anchor_on_an_unconfigured_path(tmp_path: Path) -> None:
    scanner = _scanner()
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("package/module.py", "public_value = True  # oss-boundary-anchor: rogue-marker\n")

    with pytest.raises(ValueError, match="unregistered OSS boundary anchor"):
        scanner.scan_archive(wheel, allowlist_path=_write_anchor_allowlist(tmp_path / "allowlist.json"))


def test_archive_scan_fails_closed_when_anchor_tokenization_fails(tmp_path: Path) -> None:
    scanner = _scanner()
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "tests/test_negative.py",
            "public_value = True  # oss-boundary-anchor: fixture-credential\nunterminated = '''\n",
        )

    with pytest.raises(ValueError, match="could not tokenize Python source"):
        scanner.scan_archive(wheel, allowlist_path=_write_anchor_allowlist(tmp_path / "allowlist.json"))


def _negative_test_allowlist(path: Path) -> Path:
    allowlist = path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "version": 2,
                "allowlist": [
                    {
                        "path": "tests/test_negative.py",
                        "rule": "internal-nvidia-credential",
                        "anchor": "fixture-credential",
                        "reason": "negative archive fixture",
                        "expires": "2099-12-31",
                        "review_owner": "oss-release-reviewers",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return allowlist


def test_wheel_allowlist_does_not_strip_an_arbitrary_member_prefix(tmp_path: Path) -> None:
    scanner = _scanner()
    source = (
        _joined("credential = 'NVI'", " + 'DIA'", " + '_INFERENCE_KEY'")
        + "  # oss-boundary-anchor: fixture-credential\n"
    )
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("evil/tests/test_negative.py", source)

    with pytest.raises(ValueError, match=r"unregistered OSS boundary anchor.*evil/tests/test_negative.py"):
        scanner.scan_archive(wheel, allowlist_path=_negative_test_allowlist(tmp_path))


def test_wheel_rejects_duplicate_members_before_associating_anchors(tmp_path: Path) -> None:
    scanner = _scanner()
    wheel = tmp_path / "package.whl"
    denied = _joined("credential = 'NVI'", " + 'DIA'", " + '_INFERENCE_KEY'\n")
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("tests/test_negative.py", denied)
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr(
                "tests/test_negative.py",
                "public_value = True  # oss-boundary-anchor: fixture-credential\n",
            )

    with pytest.raises(ValueError, match="duplicate archive member path"):
        scanner.scan_archive(wheel, allowlist_path=_negative_test_allowlist(tmp_path))


def test_sdist_rejects_duplicate_members_before_associating_anchors(tmp_path: Path) -> None:
    scanner = _scanner()
    sdist = tmp_path / "package-1.0.tar.gz"
    denied = _joined("credential = 'NVI'", " + 'DIA'", " + '_INFERENCE_KEY'\n").encode()
    anchored_public = b"public_value = True  # oss-boundary-anchor: fixture-credential\n"
    with tarfile.open(sdist, "w:gz") as archive:
        for source in (denied, anchored_public):
            info = tarfile.TarInfo("package-1.0/tests/test_negative.py")
            info.size = len(source)
            archive.addfile(info, BytesIO(source))

    with pytest.raises(ValueError, match="duplicate archive member path"):
        scanner.scan_archive(sdist, allowlist_path=_negative_test_allowlist(tmp_path))


@pytest.mark.parametrize("member_type", (tarfile.SYMTYPE, tarfile.LNKTYPE))
def test_sdist_rejects_links_before_allowlisting(tmp_path: Path, member_type: bytes) -> None:
    scanner = _scanner()
    sdist = tmp_path / "package-1.0.tar.gz"
    allowed_source = (
        _joined("credential = 'NVI'", " + 'DIA'", " + '_INFERENCE_KEY'")
        + "  # oss-boundary-anchor: fixture-credential\n"
    ).encode()
    with tarfile.open(sdist, "w:gz") as archive:
        allowed = tarfile.TarInfo("package-1.0/tests/test_negative.py")
        allowed.size = len(allowed_source)
        archive.addfile(allowed, BytesIO(allowed_source))
        link = tarfile.TarInfo("package-1.0/src/runtime.py")
        link.type = member_type
        link.linkname = "../tests/test_negative.py"
        archive.addfile(link)

    with pytest.raises(ValueError, match="unsupported TAR member"):
        scanner.scan_archive(sdist, allowlist_path=_negative_test_allowlist(tmp_path))


def test_wheel_rejects_a_unix_symlink_entry(tmp_path: Path) -> None:
    scanner = _scanner()
    wheel = tmp_path / "package.whl"
    symlink = zipfile.ZipInfo("src/runtime.py")
    symlink.create_system = 3
    symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(symlink, "../tests/test_negative.py")

    with pytest.raises(ValueError, match="unsupported ZIP member"):
        scanner.scan_archive(wheel)


def test_wheel_scan_rejects_member_path_traversal_before_allowlisting(tmp_path: Path) -> None:
    scanner = _scanner()
    source = (
        _joined("credential = 'NVI'", " + 'DIA'", " + '_INFERENCE_KEY'")
        + "  # oss-boundary-anchor: fixture-credential\n"
    )
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("../tests/test_negative.py", source)

    with pytest.raises(ValueError, match="unsafe archive member path"):
        scanner.scan_archive(wheel, allowlist_path=_negative_test_allowlist(tmp_path))


def test_sdist_allowlist_only_normalizes_the_expected_distribution_root(tmp_path: Path) -> None:
    scanner = _scanner()
    source = (
        _joined("credential = 'NVI'", " + 'DIA'", " + '_INFERENCE_KEY'")
        + "  # oss-boundary-anchor: fixture-credential\n"
    ).encode()
    sdist = tmp_path / "package-1.0.tar.gz"
    info = tarfile.TarInfo("evil/tests/test_negative.py")
    info.size = len(source)
    with tarfile.open(sdist, "w:gz") as archive:
        archive.addfile(info, BytesIO(source))

    with pytest.raises(ValueError, match="expected source distribution root"):
        scanner.scan_archive(sdist, allowlist_path=_negative_test_allowlist(tmp_path))


def test_repository_scan_fails_closed_for_a_missing_root(tmp_path: Path) -> None:
    scanner = _scanner()

    with pytest.raises(ValueError, match="repository root"):
        scanner.scan_repository(tmp_path / "missing")


def test_archive_scan_fails_closed_for_oversized_text_member(tmp_path: Path, monkeypatch) -> None:
    scanner = _scanner()
    monkeypatch.setattr(scanner, "_MAX_MEMBER_BYTES", 4)
    wheel = tmp_path / "oversized.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("package/module.py", "five!")

    with pytest.raises(ValueError, match="exceeds the scan limit"):
        scanner.scan_archive(wheel)


def test_archive_scan_fails_closed_for_non_utf8_text_member(tmp_path: Path) -> None:
    scanner = _scanner()
    wheel = tmp_path / "invalid-text.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("package/module.py", b"\xff")

    with pytest.raises(ValueError, match="not UTF-8 text"):
        scanner.scan_archive(wheel)


def test_zip_scan_enforces_global_member_count_limit(tmp_path: Path, monkeypatch) -> None:
    scanner = _scanner()
    monkeypatch.setattr(scanner, "_MAX_ARCHIVE_MEMBERS", 2)
    wheel = tmp_path / "too-many.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for index in range(3):
            archive.writestr(f"package/{index}.py", "public")

    with pytest.raises(ValueError, match="member count limit"):
        scanner.scan_archive(wheel)


def test_sdist_scan_enforces_aggregate_decompressed_byte_limit(tmp_path: Path, monkeypatch) -> None:
    scanner = _scanner()
    monkeypatch.setattr(scanner, "_MAX_ARCHIVE_BYTES", 5)
    sdist = tmp_path / "package.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for index in range(2):
            data = b"abc"
            info = tarfile.TarInfo(f"package/{index}.py")
            info.size = len(data)
            archive.addfile(info, BytesIO(data))

    with pytest.raises(ValueError, match="aggregate decompressed size limit"):
        scanner.scan_archive(sdist)


def _patch_zip_flags(path: Path, *, encrypted: bool = False, compression: int | None = None) -> None:
    payload = bytearray(path.read_bytes())
    offset = 0
    while True:
        offset = payload.find(b"PK\x03\x04", offset)
        if offset < 0:
            break
        if encrypted:
            flags = struct.unpack_from("<H", payload, offset + 6)[0] | 0x1
            struct.pack_into("<H", payload, offset + 6, flags)
        if compression is not None:
            struct.pack_into("<H", payload, offset + 8, compression)
        offset += 4
    offset = 0
    while True:
        offset = payload.find(b"PK\x01\x02", offset)
        if offset < 0:
            break
        if encrypted:
            flags = struct.unpack_from("<H", payload, offset + 8)[0] | 0x1
            struct.pack_into("<H", payload, offset + 8, flags)
        if compression is not None:
            struct.pack_into("<H", payload, offset + 10, compression)
        offset += 4
    path.write_bytes(payload)


def test_zip_scan_fails_closed_cleanly_on_encrypted_member(tmp_path: Path) -> None:
    scanner = _scanner()
    wheel = tmp_path / "encrypted.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("package/module.py", "public")
    _patch_zip_flags(wheel, encrypted=True)

    with pytest.raises(ValueError, match="encrypted ZIP member"):
        scanner.scan_archive(wheel)


def test_zip_scan_fails_closed_cleanly_on_unsupported_member(tmp_path: Path) -> None:
    scanner = _scanner()
    wheel = tmp_path / "unsupported.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("package/module.py", "public")
    _patch_zip_flags(wheel, compression=99)

    with pytest.raises(ValueError, match="unsupported ZIP member"):
        scanner.scan_archive(wheel)


def test_checked_in_source_and_allowlist_satisfy_the_public_boundary() -> None:
    scanner = _scanner()

    assert ALLOWLIST_PATH.is_file()
    assert scanner.scan_repository(REPO_ROOT, allowlist_path=ALLOWLIST_PATH) == []
