# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Follow-ups to the linear-time JWT redaction change.

* JWT redaction keeps memory flat on long token runs (host and verifier copies).
* The Harbor verifier files still parse on a task image's older python3.
* The Tier 1 PII scan stays linear on one very long line.
* Local mode restores SIGPIPE before it runs a command, so pipelines end.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import tokenize
import tracemalloc
from pathlib import Path
from unittest import mock

import pytest
import yaml

import skillevaluator
from skillevaluator.tier3.eval_core import checks as eval_core_checks
from skillevaluator.tier3.eval_core.secret_redaction import redact_secrets_in_log_line
from skillevaluator.tier3.harbor import adapter as harbor_adapter
from skillevaluator.tier3.harbor import local_sandbox
from skillevaluator.tier3.harbor.local_environment import SkillEvaluatorLocalEnvironment
from skillevaluator.validators.security import SecurityValidator

# Resolved through the imported package, so the tests follow whichever source tree is on sys.path.
_PACKAGE = Path(skillevaluator.__file__).resolve().parent
_TEMPLATES = _PACKAGE / "tier3" / "harbor" / "templates"


def _staged_verifier_sources() -> tuple[tuple[Path, ...], tuple[str, ...]]:
    """What ``_copy_verifier`` really stages: the source of each copy, and the staged file names."""
    real_copy2 = shutil.copy2
    sources: list[Path] = []

    def record(src, dst, *args, **kwargs):
        sources.append(Path(src).resolve())
        return real_copy2(src, dst, *args, **kwargs)

    with tempfile.TemporaryDirectory() as scratch, mock.patch.object(harbor_adapter.shutil, "copy2", record):
        harbor_adapter._copy_verifier(Path(scratch))
        staged = tuple(sorted(path.name for path in (Path(scratch) / "tests").iterdir()))
    return tuple(sources), staged


_STAGED_SOURCES, _STAGED_NAMES = _staged_verifier_sources()
# Everything the Harbor verifier runs with the task image's python3: the templates
# and every helper ``_copy_verifier`` stages beside eval.py (read from what it
# copies, so a newly staged helper is checked here and must be compiled in CI).
_VERIFIER_FILES = tuple(dict.fromkeys((*sorted(_TEMPLATES.glob("*.py")), *_STAGED_SOURCES)))


def _load_template_module():
    spec = importlib.util.spec_from_file_location("harbor_template_eval_pr176_followups", _TEMPLATES / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_template = _load_template_module()

REDACTORS = pytest.mark.parametrize(
    "redact",
    [redact_secrets_in_log_line, eval_template.redact_secrets_in_log_line],
    ids=["host", "template"],
)


# --- JWT redaction memory -----------------------------------------------------


@REDACTORS
@pytest.mark.parametrize(
    "run",
    [
        pytest.param("a" * (1 << 20), id="one-char-run"),
        pytest.param("ab-" * ((1 << 20) // 3), id="dashed-run"),
    ],
)
def test_jwt_redaction_memory_stays_flat_on_a_long_token_run(redact, run: str) -> None:
    """A 1 MB run of JWT characters used to cost about 75 MB of regex backtracking state."""
    # "eyJ" elsewhere on the line skips the fast path, so the JWT regex walks the whole run.
    line = f"note: header eyJhbGciOi follows {run} end"
    tracemalloc.start()
    try:
        redacted = redact(line)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert redacted == line
    # The output copy alone is about 1 MB; the old pattern peaked near 78 MB here.
    assert peak < 8 * (1 << 20), f"peak {peak / (1 << 20):.1f} MB"


# --- Verifier files parse on older Python -------------------------------------


def _python_version(executable: str) -> tuple[int, int] | None:
    try:
        output = subprocess.run(
            [executable, "-I", "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    major, minor = output.split()
    return int(major), int(minor)


def _older_python() -> str | None:
    """An installed interpreter older than 3.12 (task images may still ship one), if any."""
    candidates = [found for name in ("python3.9", "python3.10", "python3.11") if (found := shutil.which(name))]
    uv_root = Path(os.environ.get("UV_PYTHON_INSTALL_DIR") or Path.home() / ".local" / "share" / "uv" / "python")
    for minor in ("9", "10", "11"):
        candidates.extend(str(path) for path in sorted(uv_root.glob(f"cpython-3.{minor}.*/bin/python3.{minor}")))
    candidates.append("/usr/bin/python3")
    for candidate in candidates:
        version = _python_version(candidate) if Path(candidate).exists() else None
        if version is not None and version < (3, 12):
            return candidate
    return None


def test_verifier_files_compile_on_an_older_python() -> None:
    python = _older_python()
    if python is None:
        pytest.skip("no Python older than 3.12 is installed; the tokenizer check below still runs")
    compile_all = (
        "import sys\n"
        "for path in sys.argv[1:]:\n"
        "    with open(path, encoding='utf-8') as handle:\n"
        "        compile(handle.read(), path, 'exec')\n"
    )
    proc = subprocess.run(
        [python, "-I", "-c", compile_all, *map(str, _VERIFIER_FILES)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr


def _fstring_syntax_needing_python_312(source: str) -> list[str]:
    """f-string expressions that only parse on Python 3.12+ (PEP 701).

    Before 3.12 an expression inside ``{...}`` could not hold a backslash, a
    comment, the f-string's own quote, or (in a one-quote f-string) a line break.
    """
    issues: list[str] = []
    frames: list[list] = []  # [closing quote, brace depth] per open f-string
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if frames and frames[-1][1] > 0 and token.type != tokenize.FSTRING_MIDDLE:
            quote = frames[-1][0]
            where = f"line {token.start[0]}"
            if token.type == tokenize.COMMENT:
                issues.append(f"{where}: comment inside an f-string expression")
            elif token.type in (tokenize.NL, tokenize.NEWLINE) and len(quote) == 1:
                issues.append(f"{where}: line break inside a one-quote f-string expression")
            elif "\\" in token.string:
                issues.append(f"{where}: backslash inside an f-string expression")
            elif token.type in (tokenize.STRING, tokenize.FSTRING_START) and quote in token.string:
                issues.append(f"{where}: the f-string's quote is reused inside its expression")
        if token.type == tokenize.FSTRING_START:
            frames.append([token.string.lstrip("rRfFbBuU"), 0])
        elif token.type == tokenize.FSTRING_END:
            frames.pop()
        elif frames and token.type == tokenize.OP and token.string in ("{", "}"):
            frames[-1][1] += 1 if token.string == "{" else -1
    return issues


@pytest.mark.parametrize(
    ("source", "issue"),
    [
        pytest.param("a = f\"{base.rstrip('/\\\\')}/{value}\"\n", "backslash", id="backslash"),
        pytest.param('b = f"{items["key"]}"\n', "quote is reused", id="same-quote"),
        pytest.param('c = f"{value  # note\n}"\n', "comment", id="comment"),
    ],
)
def test_fstring_syntax_checker_flags_python_312_forms(source: str, issue: str) -> None:
    issues = _fstring_syntax_needing_python_312(source)
    assert issues
    assert issue in issues[0]


def test_fstring_syntax_checker_accepts_portable_forms() -> None:
    portable = textwrap.dedent(
        """\
        base = current.rstrip("/\\\\")
        a = f"{base}/{value!r:>10}"
        b = f'{items["key"]}'
        c = f\"\"\"{items["key"]}\"\"\"
        d = f"{x:{width}}" + f"{{literal}}"
        """
    )
    assert _fstring_syntax_needing_python_312(portable) == []


@pytest.mark.parametrize("path", _VERIFIER_FILES, ids=lambda path: path.name)
def test_verifier_files_use_no_python_312_only_fstring_syntax(path: Path) -> None:
    """Runs everywhere: the compile check above needs an older interpreter."""
    assert _fstring_syntax_needing_python_312(path.read_text(encoding="utf-8")) == []


def test_verifier_file_list_covers_everything_copy_verifier_stages() -> None:
    """Every staged file was seen through ``shutil.copy2``, so none is missing from the checks."""
    assert sorted(path.name for path in _STAGED_SOURCES) == list(_STAGED_NAMES)
    assert "eval.py" in _STAGED_NAMES
    assert set(_STAGED_SOURCES) <= set(_VERIFIER_FILES)


def _ci_compile_step_run() -> str:
    workflow = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
    steps = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]["test-python-312"]["steps"]
    [step] = [step for step in steps if step.get("name", "").startswith("Compile Harbor verifier files")]
    return step["run"]


def _missing_from_ci_compile_step(run: str, paths) -> list[str]:
    """Verifier files that the CI compile step neither names nor covers with its templates glob."""
    missing = []
    for path in paths:
        relative = path.relative_to(_PACKAGE.parents[1]).as_posix()
        if path.parent != _TEMPLATES and relative not in run.split():
            missing.append(relative)
    return missing


def test_ci_compiles_the_verifier_files_on_older_python() -> None:
    run = _ci_compile_step_run()
    assert "for version in 3.9 3.11" in run
    assert "src/skillevaluator/tier3/harbor/templates/*.py" in run.split()
    assert _missing_from_ci_compile_step(run, _VERIFIER_FILES) == []


def test_ci_compile_check_catches_a_newly_staged_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """A helper that ``_copy_verifier`` starts staging must also be added to the CI compile step."""
    new_helper = _PACKAGE / "tier3" / "eval_core" / "secret_redaction.py"
    real_copy_verifier = harbor_adapter._copy_verifier

    def copy_verifier_with_new_helper(task_dir: Path) -> None:
        real_copy_verifier(task_dir)
        shutil.copy2(new_helper, task_dir / "tests" / new_helper.name)

    monkeypatch.setattr(harbor_adapter, "_copy_verifier", copy_verifier_with_new_helper)
    sources, _staged = _staged_verifier_sources()

    assert _missing_from_ci_compile_step(_ci_compile_step_run(), sources) == [
        "src/skillevaluator/tier3/eval_core/secret_redaction.py"
    ]


# --- Verifier files also run on older Python --------------------------------------

_SMOKE = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "smoke_harbor_verifier.py"


def _python_310_runtime_calls(source: str) -> list[str]:
    """Calls that parse on older Python but fail when they run: ``zip(strict=)`` and ``isinstance(x, A | B)``."""
    issues: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id == "zip" and any(keyword.arg == "strict" for keyword in node.keywords):
            issues.append(f"line {node.lineno}: zip(strict=) needs Python 3.10")
        union = len(node.args) == 2 and isinstance(node.args[1], ast.BinOp) and isinstance(node.args[1].op, ast.BitOr)
        if node.func.id in {"isinstance", "issubclass"} and union:
            issues.append(f"line {node.lineno}: {node.func.id}() with a union type needs Python 3.10")
    return issues


def test_runtime_call_checker_flags_python_310_calls() -> None:
    source = (
        "zip(a, b, strict=True)\nisinstance(x, int | float)\nissubclass(t, A | B)\nzip(a, b)\nisinstance(x, (A, B))\n"
    )
    assert [issue.split(":")[0] for issue in _python_310_runtime_calls(source)] == ["line 1", "line 2", "line 3"]


@pytest.mark.parametrize("path", _VERIFIER_FILES, ids=lambda path: path.name)
def test_verifier_files_make_no_python_310_only_calls(path: Path) -> None:
    """Runs everywhere: compiling on an older Python cannot see these."""
    assert _python_310_runtime_calls(path.read_text(encoding="utf-8")) == []


def test_verifier_smoke_passes_on_this_python() -> None:
    proc = subprocess.run([sys.executable, "-I", str(_SMOKE)], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_staged_verifier_runs_its_canary_and_retry_code_on_an_older_python(tmp_path: Path) -> None:
    python = _older_python()
    if python is None:
        pytest.skip("no Python older than 3.12 is installed; the call checker above still runs")
    harbor_adapter._copy_verifier(tmp_path)
    tests_dir = tmp_path / "tests"
    for helper in ("custom_grader_runner.py", "metric.py"):  # staged for custom graders and Harbor's metric
        shutil.copy2(_TEMPLATES / helper, tests_dir / helper)
    if subprocess.run([python, "-I", "-c", "import idna"], capture_output=True, timeout=30).returncode != 0:
        # The verifier image installs idna; these code paths never call it, so a stand-in is enough.
        (tests_dir / "idna.py").write_text(
            "class IDNAError(Exception):\n    pass\n\n\ndef encode(value):\n    return value.encode('ascii')\n"
        )
    proc = subprocess.run([python, "-I", str(_SMOKE), str(tests_dir)], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_ci_runs_the_verifier_smoke_on_older_python() -> None:
    run = _ci_compile_step_run()
    assert "for version in 3.9 3.11" in run
    assert "scripts/ci/smoke_harbor_verifier.py" in run.split()


@pytest.mark.parametrize(
    "path_with_shell_cwd",
    [eval_core_checks._path_with_shell_cwd, eval_template._path_with_shell_cwd],
    ids=["host", "template"],
)
def test_shell_cwd_join_keeps_its_behavior(path_with_shell_cwd) -> None:
    assert path_with_shell_cwd("notes.md", "/work/dir/") == "/work/dir/notes.md"
    assert path_with_shell_cwd("notes.md", "C:\\work\\") == "C:\\work/notes.md"
    assert path_with_shell_cwd("/abs/notes.md", "/work") == "/abs/notes.md"
    assert path_with_shell_cwd("notes.md", None) == "notes.md"


# --- Tier 1 PII scan on one long line -------------------------------------------

_LINE_BYTES = 256 * 1024
_PII_SCAN = """
import sys, time
from pathlib import Path
from skillevaluator.validators.security import SecurityValidator
started = time.perf_counter()
SecurityValidator(use_llm=False).validate_pii_only(Path(sys.argv[1]))
print(f"{time.perf_counter() - started:.3f}")
"""


def _repeat(unit: str, size: int = _LINE_BYTES) -> str:
    return (unit * (size // len(unit) + 1))[:size]


def _probe_skill(root: Path, line: str) -> Path:
    skill = root / "pii-probe"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: pii-probe\ndescription: Probe skill.\n---\n# Probe\n")
    (skill / "scripts" / "data.txt").write_text(line + "\n")
    return skill


@pytest.mark.parametrize(
    "line",
    [
        pytest.param(_repeat("eyJ-"), id="jwt-like-run"),
        pytest.param(_repeat("abc-"), id="dashed-words"),
        pytest.param(_repeat("0123456789abcdef"), id="hex"),
        pytest.param("x" * 64 + "@" + _repeat("a.")[: _LINE_BYTES - 65], id="long-domain"),
        # 1 MB here: each old per-start scan was cheap, so 256 KB took only about 4 s.
        pytest.param(_repeat("redis://a:", 4 * _LINE_BYTES), id="repeated-db-url"),
        pytest.param(_repeat("postgres://a:b/c:d", 4 * _LINE_BYTES), id="repeated-db-url-slash-password"),
        pytest.param("mysql://u:" + _repeat("x/:", 4 * _LINE_BYTES), id="one-long-db-password"),
        pytest.param(_repeat("connection-string "), id="repeated-connection-string"),
    ],
)
def test_pii_scan_is_linear_on_one_long_line(tmp_path: Path, line: str) -> None:
    """Each of these lines used to take from seconds to many minutes."""
    skill = _probe_skill(tmp_path, line)
    try:
        proc = subprocess.run([sys.executable, "-c", _PII_SCAN, str(skill)], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        pytest.fail("PII scan of one long line did not finish in 30 s")
    assert proc.returncode == 0, proc.stderr
    elapsed = float(proc.stdout.strip().splitlines()[-1])
    assert elapsed < 3.0, f"PII scan took {elapsed:.2f} s"


def _pii_findings(tmp_path: Path, text: str) -> dict[str, list[str]]:
    skill = _probe_skill(tmp_path, text)
    result = SecurityValidator(use_llm=False).validate_pii_only(skill)
    found: dict[str, list[str]] = {}
    for finding in result.findings:
        found.setdefault(finding.check_name, []).append(finding.metadata.get("matched_value"))
    return found


def _fixture_secret(*parts: str) -> str:
    """Build committed fake secrets from pieces so static scanners do not flag them."""
    return "".join(parts)


_JWT = _fixture_secret(
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", ".", "eyJzdWIiOiIxMjM0NTY3ODkwIn0", ".", "c3ludGhldGljLXNpZ25hdHVyZQ"
)


@pytest.mark.parametrize(
    ("text", "check", "value"),
    [
        pytest.param(f"token {_JWT}", "jwt_tokens", _JWT, id="jwt"),
        pytest.param(f"bearer-{_JWT}", "jwt_tokens", _JWT, id="jwt-glued-to-dash-reports-only-the-token"),
        pytest.param(f"BEARER-{_JWT.upper()}", "jwt_tokens", _JWT.upper(), id="jwt-any-case"),
        pytest.param(
            "connectionString=Server=db;password=x connection_string=y",
            "database_credentials",
            "connectionString=Server=db;password=",
            id="connection-string-from-first-mention",
        ),
        pytest.param(
            _fixture_secret("postgresql://admin:", "pw123", "@db.internal:5432/prod"),
            "database_credentials",
            _fixture_secret("postgresql://admin:", "pw123", "@db.internal:5432"),
            id="db-url",
        ),
        pytest.param("mail " + "r" * 70 + "@corp.io", "emails", "r" * 64 + "@corp.io", id="email-long-local-part"),
    ],
)
def test_linear_pii_patterns_still_report_the_same_values(tmp_path: Path, text: str, check: str, value: str) -> None:
    assert _pii_findings(tmp_path, text).get(check) == [value]


@pytest.mark.parametrize(
    "url",
    [
        pytest.param(_fixture_secret("postgres://admin:", "wJalrXUtnFEMI/K7MDENG", "@db.prod:5432"), id="postgres"),
        pytest.param(_fixture_secret("mongodb://root:", "abc/123", "@mongo:27017"), id="mongodb"),
        pytest.param(_fixture_secret("postgresql://svc:", "S3cr3t+/x=", "@10.0.0.5"), id="base64-like"),
    ],
)
def test_db_url_password_with_a_slash_is_still_reported(tmp_path: Path, url: str) -> None:
    """Generated passwords often hold a raw "/"; the linear pattern must still report them."""
    assert _pii_findings(tmp_path, f"DATABASE_URL={url}/app").get("database_credentials") == [url]


@pytest.mark.parametrize(
    "line",
    [
        pytest.param("cache: redis://cache:6379/0?owner=ops@corp.io", id="port-then-path"),
        pytest.param("see redis://localhost:6379 or mail ops@corp.io", id="port-then-text"),
    ],
)
def test_db_url_port_is_not_a_password(tmp_path: Path, line: str) -> None:
    """A host:port followed by a path or text, then an "@" later on, is not a credential."""
    assert "database_credentials" not in _pii_findings(tmp_path, line)


# --- Local mode SIGPIPE -----------------------------------------------------------


def test_local_bootstrap_resets_only_real_signals() -> None:
    """Every signal the local bootstrap restores exists on POSIX; a misspelled name would be a silent no-op."""
    import re
    import signal

    from skillevaluator.tier3.harbor import local_environment

    names = re.findall(r'"(SIG[A-Z0-9]+)"', local_environment._INNER_ENV_BOOTSTRAP)
    assert names == ["SIGPIPE", "SIGXFSZ"]
    if os.name == "posix":
        assert all(hasattr(signal, name) for name in names)


def _local_environment(tmp_path: Path) -> SkillEvaluatorLocalEnvironment:
    environment = object.__new__(SkillEvaluatorLocalEnvironment)
    environment._runtime_root = tmp_path / "runtime"
    environment._runtime_agent = "opencode"
    environment._root = tmp_path / "run"
    environment._workspace = environment._root / "workspace"
    environment._tests = environment._root / "tests"
    environment._solution = environment._root / "solution"
    environment._installed_agent = environment._root / "installed-agent"
    environment._tmp = environment._root / "tmp"
    environment._home = environment._root / "home"
    environment._sandbox_mode = "off"
    environment._allow_net = False
    environment._inherit_agent_keys = False
    environment._strict_reads = False
    environment._active_processes = {}
    environment._persistent_env = {}
    environment._sandbox = local_sandbox.Sandbox(local_sandbox.SandboxPlan("none", "advisory-only", "test"))
    trial = tmp_path / "trial"
    environment.trial_paths = type(
        "TrialPaths",
        (),
        {
            "trial_dir": trial,
            "agent_dir": trial / "agent",
            "verifier_dir": trial / "verifier",
            "artifacts_dir": trial / "artifacts",
            "reward_json_path": trial / "verifier" / "reward.json",
            "reward_text_path": trial / "verifier" / "reward.txt",
        },
    )()
    environment.logger = logging.getLogger("test-pr176-followups")
    for path in (
        environment._workspace,
        environment._tests,
        environment._solution,
        environment._installed_agent,
        environment._tmp,
        environment._home,
        environment.trial_paths.agent_dir,
        environment.trial_paths.verifier_dir,
        environment.trial_paths.artifacts_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return environment


@pytest.mark.skipif(os.name == "nt", reason="local mode is POSIX-only")
def test_local_mode_pipeline_ends_when_its_reader_exits(tmp_path: Path) -> None:
    """With SIGPIPE ignored, `yes` reports EPIPE (GNU) or `tr` spins forever (macOS)."""
    environment = _local_environment(tmp_path)

    result = asyncio.run(environment.exec("yes x | tr -d '\\n' | head -c 5", timeout_sec=15))

    assert result.return_code == 0, result.stderr
    assert result.stdout == "xxxxx"
    assert "Broken pipe" not in (result.stderr or "")
