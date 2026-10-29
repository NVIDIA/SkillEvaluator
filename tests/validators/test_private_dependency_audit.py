# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in dependency audits never install target packages or use real scanners."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from skillevaluator.plugin_mcp import McpDeclaration
from skillevaluator.plugin_paths import PluginRootReader
from skillevaluator.utils.tool_runner import ToolResult

MODULE = "skillevaluator.plugin_dependency_audit"


def test_private_audit_api_exists():
    assert importlib.util.find_spec(MODULE) is not None


@pytest.fixture
def audit():
    return importlib.import_module(MODULE)


class FakeTool:
    def __init__(self, output=None, exit_code=0, available=True):
        self.output, self.exit_code, self.is_available = output, exit_code, available
        self.calls = []

    def get_install_hint(self):
        return "fixture scanner missing"

    def run(self, args, **kwargs):
        files = {p.name: p.read_text() for p in Path(kwargs["cwd"]).iterdir() if p.is_file()}
        self.calls.append((args, kwargs, files))
        output = self.output(args, files) if callable(self.output) else self.output
        return ToolResult(True, json.dumps(output), "", self.exit_code)


def source(tmp_path, files):
    for name, text in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return PluginRootReader(tmp_path), tuple(PurePosixPath(name) for name in files)


def toolset(**overrides):
    tools = {name: FakeTool(available=False) for name in ("pip_audit", "npm", "grype", "trivy")}
    return {**tools, **overrides}


def python_report(args, files):
    pins = files["requirements.txt"].splitlines()
    return {"dependencies": [{"name": pin.split("==")[0], "version": pin.split("==")[1], "vulns": []} for pin in pins]}


def npm_clean():
    return {
        "auditReportVersion": 2,
        "vulnerabilities": {},
        "metadata": {
            "vulnerabilities": {"info": 0, "low": 0, "moderate": 0, "high": 0, "critical": 0, "total": 0},
            "dependencies": {"prod": 1, "dev": 0, "optional": 0, "peer": 0, "peerOptional": 0, "total": 1},
        },
    }


def test_collection_is_offline_and_python_never_installs_untrusted_sources(audit, tmp_path, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "https://user:password@proxy.invalid")
    monkeypatch.setenv("PIP_INDEX_URL", "https://user:password@index.invalid")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "secret")
    reader, paths = source(
        tmp_path, {"requirements.txt": "requests==2.0.0\n-r other.txt\nhttps://secret.invalid/pkg\n"}
    )
    inputs = audit.collect_audit_inputs(reader, paths)
    scanner = FakeTool(python_report)
    result = audit.run_dependency_audits(reader, inputs, tools=toolset(pip_audit=scanner))
    assert result.is_incomplete
    args, kwargs, files = scanner.calls[0]
    assert files["requirements.txt"] == "requests==2.0.0\n"
    assert "--no-deps" in args and "--disable-pip" in args
    assert kwargs["replace_env"] is True
    assert not any(
        key.upper().endswith("PROXY") or key in {"PIP_INDEX_URL", "AWS_ACCESS_KEY_ID"} for key in kwargs["env"]
    )
    assert kwargs["env"]["HOME"] != str(Path.home())


def test_pyproject_optional_groups_and_mcp_pins_are_audited(audit, tmp_path):
    reader, paths = source(
        tmp_path,
        {
            "pyproject.toml": '[project]\nname="fixture"\ndependencies=["foo==1.0"]\n[project.optional-dependencies]\ntest=["foo==2.0; extra == \'test\'"]\n'
        },
    )
    declarations = [McpDeclaration("runner", {"command": "uvx", "args": ["bar@3.0"]}, "inline", "manifest.json")]
    scanner = FakeTool(python_report)
    inputs = audit.collect_audit_inputs(reader, paths, mcp=declarations)
    result = audit.run_dependency_audits(reader, inputs, tools=toolset(pip_audit=scanner))
    assert not result.is_incomplete
    contents = "".join(call[2]["requirements.txt"] for call in scanner.calls)
    assert all(pin in contents for pin in ("foo==1.0", "foo==2.0", "bar==3.0"))
    assert len(scanner.calls) >= 2


@pytest.mark.parametrize(
    "report,code",
    [({}, 0), ({"dependencies": []}, 0), ({"dependencies": [{"name": "foo", "version": "1.0", "vulns": []}]}, 1)],
)
def test_missing_or_contradictory_python_report_is_incomplete(audit, tmp_path, report, code):
    reader, paths = source(tmp_path, {"requirements.txt": "foo==1.0\n"})
    result = audit.run_dependency_audits(
        reader, audit.collect_audit_inputs(reader, paths), tools=toolset(pip_audit=FakeTool(report, code))
    )
    assert result.is_incomplete


def test_lockfile_precedence_synthesizes_only_exact_npm_pins(audit, tmp_path):
    reader, paths = source(
        tmp_path,
        {
            "package.json": '{"dependencies":{"ignored":"8.0.0"}}',
            "package-lock.json": json.dumps(
                {
                    "lockfileVersion": 3,
                    "packages": {
                        "": {},
                        "node_modules/foo": {
                            "version": "1.2.3",
                            "resolved": "https://registry.npmjs.org/foo/-/foo-1.2.3.tgz",
                        },
                    },
                }
            ),
        },
    )
    scanner = FakeTool(npm_clean())
    result = audit.run_dependency_audits(reader, audit.collect_audit_inputs(reader, paths), tools=toolset(npm=scanner))
    assert not result.is_incomplete
    args, kwargs, files = scanner.calls[0]
    assert "audit" in args and "--ignore-scripts" in args and "--package-lock-only" in args
    assert "--registry=https://registry.npmjs.org" in args
    assert "ignored" not in str(files) and "resolved" not in str(files)
    assert json.loads(files["package.json"])["dependencies"] == {"foo": "1.2.3"}
    assert kwargs["replace_env"] is True
    assert kwargs["env"]["NPM_CONFIG_USERCONFIG"].startswith(kwargs["env"]["HOME"])


@pytest.mark.parametrize(
    "report,code",
    [({}, 0), ({"metadata": {}}, 0), (npm_clean(), 2), ({**npm_clean(), "vulnerabilities": []}, 0), (npm_clean(), 1)],
)
def test_untrustworthy_npm_report_never_counts_as_clean(audit, tmp_path, report, code):
    reader, paths = source(tmp_path, {"package.json": '{"dependencies":{"foo":"1.2.3"}}'})
    result = audit.run_dependency_audits(
        reader, audit.collect_audit_inputs(reader, paths), tools=toolset(npm=FakeTool(report, code))
    )
    assert result.is_incomplete


def test_npm_advisory_is_attributed_without_echoing_scanner_secrets(audit, tmp_path):
    reader, paths = source(tmp_path, {"package.json": '{"dependencies":{"foo":"1.2.3"}}'})
    report = npm_clean()
    report["vulnerabilities"] = {
        "foo": {
            "via": [
                {
                    "name": "foo",
                    "source": 123,
                    "title": "unsafe",
                    "severity": "high",
                    "url": "https://github.com/advisories/GHSA-fixture",
                }
            ]
        }
    }
    report["metadata"]["vulnerabilities"].update(high=1, total=1)
    result = audit.run_dependency_audits(
        reader, audit.collect_audit_inputs(reader, paths), tools=toolset(npm=FakeTool(report, 1))
    )
    assert not result.is_incomplete
    assert any(f.check_name == "npm-vulnerability" and f.file_path.endswith("package.json") for f in result.findings)


@pytest.mark.parametrize(
    "name,text",
    [
        ("yarn.lock", "unsupported"),
        ("package.json", "{bad"),
        ("package-lock.json", '{"lockfileVersion":99,"packages":{}}'),
        ("pyproject.toml", '[project]\ndynamic=["dependencies"]'),
        ("Dockerfile", "FROM ${BASE}\n"),
    ],
)
def test_unsupported_unreadable_or_unpinned_sources_are_incomplete(audit, tmp_path, name, text):
    reader, paths = source(tmp_path, {name: text})
    result = audit.run_dependency_audits(reader, audit.collect_audit_inputs(reader, paths), tools=toolset())
    assert result.is_incomplete


def test_symlink_source_is_incomplete_without_reading_target(audit, tmp_path):
    (tmp_path / "secret").write_text("sensitive content")
    (tmp_path / "requirements.txt").symlink_to(tmp_path / "secret")
    reader = PluginRootReader(tmp_path)
    result = audit.run_dependency_audits(
        reader, audit.collect_audit_inputs(reader, [PurePosixPath("requirements.txt")]), tools=toolset()
    )
    assert result.is_incomplete
    assert "sensitive content" not in str(result)


def test_image_collection_covers_multistage_dockerfiles_and_mcp(audit, tmp_path):
    reader, paths = source(tmp_path, {"Dockerfile": "FROM alpine:3.20.1 AS build\nFROM build AS final\nFROM scratch\n"})
    declarations = [
        McpDeclaration("image", {"command": "docker", "args": ["run", "redis:7.2.3"]}, "inline", "manifest.json")
    ]
    inputs = audit.collect_audit_inputs(reader, paths, mcp=declarations)
    scanner = FakeTool({"matches": []})
    result = audit.run_dependency_audits(reader, inputs, tools=toolset(grype=scanner))
    assert not result.is_incomplete
    args = [call[0] for call in scanner.calls]
    assert len(args) == 2
    assert any("registry:alpine:3.20.1" in call for call in args)
    assert any("registry:redis:7.2.3" in call for call in args)
    for _, kwargs, _ in scanner.calls:
        env = kwargs["env"]
        assert kwargs["replace_env"] is True
        assert env["DOCKER_CONFIG"].startswith(env["HOME"])
        assert env["REGISTRY_AUTH_FILE"].startswith(env["HOME"])
        assert "DOCKER_HOST" not in env and "DOCKER_CONTEXT" not in env


@pytest.mark.parametrize("image", ["169.254.169.254/x:1.2.3", "127.0.0.1/x:1.2.3", "example.com/x:latest", "--bad"])
def test_unsafe_or_floating_container_images_do_not_reach_scanners(audit, tmp_path, image):
    reader, paths = source(tmp_path, {"Dockerfile": f"FROM {image}\n"})
    scanner = FakeTool({"matches": []})
    result = audit.run_dependency_audits(
        reader, audit.collect_audit_inputs(reader, paths), tools=toolset(grype=scanner)
    )
    assert result.is_incomplete
    assert not scanner.calls


@pytest.mark.parametrize("report", [{}, {"Results": None}, {"matches": [None]}])
def test_malformed_container_reports_are_incomplete(audit, tmp_path, report):
    reader, paths = source(tmp_path, {"Dockerfile": "FROM alpine:3.20.1\n"})
    result = audit.run_dependency_audits(
        reader, audit.collect_audit_inputs(reader, paths), tools=toolset(grype=FakeTool(report))
    )
    assert result.is_incomplete


def test_trivy_remote_fallback_reports_vulnerability(audit, tmp_path):
    reader, paths = source(tmp_path, {"Dockerfile": "FROM alpine:3.20.1\n"})
    report = {
        "Results": [
            {
                "Vulnerabilities": [
                    {"VulnerabilityID": "CVE-fixture", "PkgName": "foo", "InstalledVersion": "1.0", "Severity": "HIGH"}
                ]
            }
        ]
    }
    scanner = FakeTool(report)
    result = audit.run_dependency_audits(
        reader, audit.collect_audit_inputs(reader, paths), tools=toolset(trivy=scanner)
    )
    assert not result.is_incomplete
    assert any(f.check_name == "container-vulnerability" for f in result.findings)
    assert "--image-src" in scanner.calls[0][0] and "remote" in scanner.calls[0][0]


def test_disabled_ecosystems_do_not_run_scanners(audit, tmp_path):
    reader, paths = source(
        tmp_path, {"requirements.txt": "foo==1.0\n", "package.json": '{"dependencies":{"foo":"1.2.3"}}'}
    )
    python, npm = FakeTool(python_report), FakeTool(npm_clean())
    result = audit.run_dependency_audits(
        reader,
        audit.collect_audit_inputs(reader, paths),
        python=False,
        npm=False,
        container=False,
        tools=toolset(pip_audit=python, npm=npm),
    )
    assert not python.calls and not npm.calls
    assert not result.is_incomplete


def test_endpoint_collection_covers_mcp_oauth_and_http_hook_attribution(audit):
    mcp = [
        McpDeclaration(
            "api",
            {"url": "https://api.example/", "oauth": {"authServerMetadataUrl": "https://auth.example/"}},
            "inline",
            "manifest.json",
        )
    ]
    hook = SimpleNamespace(url="https://hook.example/", id="hook-1", file="hooks.json", source="plugin")
    targets = audit.collect_endpoint_targets(mcp, [hook], allowed_hosts=("corp.example",))
    assert {t.url for t in targets} == {"https://api.example/", "https://auth.example/", "https://hook.example/"}
    assert all(t.allowed_hosts == ("corp.example",) for t in targets)
    assert any(t.kind == "hook" and t.name == "hook-1" and t.file_path == "hooks.json" for t in targets)


def test_noninteger_lockfile_version_and_nonstring_resolved_are_incomplete(audit, tmp_path):
    reader, paths = source(tmp_path, {"package-lock.json": json.dumps({"lockfileVersion": [], "packages": {}})})
    inputs = audit.collect_audit_inputs(reader, paths)
    assert not inputs.complete
    reader, paths = source(
        tmp_path,
        {
            "package-lock.json": json.dumps(
                {
                    "lockfileVersion": 3,
                    "packages": {"node_modules/foo": {"version": "1.2.3", "resolved": {"unexpected": True}}},
                }
            )
        },
    )
    assert not audit.collect_audit_inputs(reader, paths).complete


def test_python_runner_arbitrary_equality_pin_is_audited(audit, tmp_path):
    reader = PluginRootReader(tmp_path)
    declarations = [McpDeclaration("python", {"command": "uvx", "args": ["foo===1.0"]}, "inline", "manifest.json")]
    inputs = audit.collect_audit_inputs(reader, [], mcp=declarations)
    assert inputs.complete
    assert inputs.python_sources[0].pins == (("foo", "1.0"),)


def test_declared_pep508_markers_are_evaluated_in_each_optional_group(audit, tmp_path):
    reader, paths = source(
        tmp_path,
        {
            "pyproject.toml": '[project]\nname="fixture"\n[project.optional-dependencies]\ntest=["foo==1.0; extra == \'test\'", "ignored==2.0; extra == \'other\'"]'
        },
    )
    inputs = audit.collect_audit_inputs(reader, paths)
    assert inputs.complete
    assert inputs.python_sources[0].pins == (("foo", "1.0"),)


def test_direct_audit_inputs_cannot_inject_requirements_or_image_flags(audit, tmp_path):
    reader = PluginRootReader(tmp_path)
    inputs = audit.AuditInputs(
        python_sources=[audit.AuditSource("python", "manifest.json", (("-r /secret", "1.0"),))],
        images=[audit.ImageDeclaration("--config=private", "manifest.json")],
    )
    python, grype = FakeTool(python_report), FakeTool({"matches": []})
    result = audit.run_dependency_audits(reader, inputs, tools=toolset(pip_audit=python, grype=grype))
    assert result.is_incomplete
    assert not python.calls and not grype.calls


def test_file_package_and_image_caps_are_explicitly_incomplete(audit, tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "MAX_DECLARATIONS", 1)
    reader, paths = source(tmp_path, {"package.json": '{"dependencies":{"foo":"1.2.3","bar":"2.0.0"}}'})
    assert not audit.collect_audit_inputs(reader, paths).complete
    monkeypatch.setattr(audit, "MAX_SOURCE_FILES", 1)
    reader, paths = source(tmp_path, {"requirements.txt": "foo==1.0\n", "Dockerfile": "FROM alpine:3.20.1\n"})
    assert not audit.collect_audit_inputs(reader, paths).complete
    monkeypatch.setattr(audit, "MAX_SOURCE_FILES", 64)
    monkeypatch.setattr(audit, "MAX_IMAGES", 1)
    reader, paths = source(tmp_path, {"Dockerfile": "FROM alpine:3.20.1\nFROM redis:7.2.3\n"})
    assert not audit.collect_audit_inputs(reader, paths).complete


def test_npm_legacy_v1_lockfile_nested_pins_and_aliases(audit, tmp_path):
    reader, paths = source(
        tmp_path,
        {
            "package-lock.json": json.dumps(
                {
                    "lockfileVersion": 1,
                    "dependencies": {"foo": {"version": "1.2.3", "dependencies": {"nested": {"version": "2.3.4"}}}},
                }
            )
        },
    )
    inputs = audit.collect_audit_inputs(reader, paths)
    assert inputs.complete
    assert set(inputs.npm_sources[0].pins) == {("foo", "1.2.3"), ("nested", "2.3.4")}
    reader, paths = source(tmp_path, {"package.json": '{"dependencies":{"alias":"npm:@scope/real@1.2.3"}}'})
    assert audit.collect_audit_inputs(reader, paths).npm_sources[0].pins == (("@scope/real", "1.2.3"),)


def test_container_malformed_registry_port_is_never_contacted(audit, tmp_path):
    reader, paths = source(tmp_path, {"Dockerfile": "FROM registry.example:bad/image:1.2.3\n"})
    scanner = FakeTool({"matches": []})
    result = audit.run_dependency_audits(
        reader, audit.collect_audit_inputs(reader, paths), tools=toolset(grype=scanner)
    )
    assert result.is_incomplete
    assert not scanner.calls
