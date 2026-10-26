# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, file-backed local workspace. Uploaded skills never run on import."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import shutil
import stat
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import yaml

MAX_UPLOAD = 12 * 1024 * 1024
MAX_EXPANDED = 24 * 1024 * 1024
MAX_FILES = 300
IDENTIFIER = re.compile(r"^[a-f0-9]{12}$")
CASE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


def now() -> str:
    return datetime.now(UTC).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def safe_file(root: Path, relative: str) -> Path:
    """Refuse traversal and symbolic links, including links in parent paths."""
    part = PurePosixPath(relative)
    if not relative or "\\" in relative or part.is_absolute() or any(p in {"..", "."} for p in part.parts):
        raise ValueError("Choose a file inside this workspace.")
    path = root
    for component in part.parts:
        path = path / component
        if path.is_symlink():
            raise ValueError("Symbolic links are not supported in a skill workspace.")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Choose a file inside this workspace.")
    return path


def source_files(path: Path) -> list[Path]:
    files = []
    for p in sorted(path.rglob("*")):
        relative = p.relative_to(path)
        if any(part in {".git", "__pycache__", ".venv", "node_modules"} for part in relative.parts):
            continue
        if p.name.startswith(".env") or p.name in {"credentials.json", "auth.json"}:
            continue
        if p.is_symlink():
            raise ValueError("Remove symbolic links before using this skill.")
        if p.is_file():
            files.append(p)
    return files


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for p in source_files(path):
        digest.update(p.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(p.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def copy_source(source: Path, destination: Path) -> None:
    """Copy only the same validated, non-secret files we hash and export."""
    paths = source_files(source)
    destination.mkdir(parents=True)
    for path in paths:
        relative = path.relative_to(source).as_posix()
        target = safe_file(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target, follow_symlinks=False)


class Workspace:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.skills = self.root / "skills"
        self.jobs = self.root / "jobs"
        self.skills.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.jobs.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root.chmod(0o700)

    def directory(self, skill_id: str) -> Path:
        if not isinstance(skill_id, str) or not IDENTIFIER.fullmatch(skill_id):
            raise ValueError("Unknown skill.")
        result = self.skills / skill_id / "skill"
        if not result.is_dir() or result.is_symlink() or result.parent.is_symlink():
            raise ValueError("Unknown skill.")
        return result

    def describe(self, skill_id: str) -> dict:
        folder = self.directory(skill_id)
        metadata = json.loads((folder.parent / "metadata.json").read_text())
        content = safe_file(folder, "SKILL.md").read_text(encoding="utf-8")
        header = {}
        if content.startswith("---"):
            pieces = content.split("---", 2)
            if len(pieces) == 3:
                try:
                    loaded = yaml.safe_load(pieces[1])
                    header = loaded if isinstance(loaded, dict) else {}
                except yaml.YAMLError:
                    pass
        from skillevaluator.studio.adapter import is_native_custom_only
        from skillevaluator.tier3.dataset_utils import find_eval_file, load_dataset_entries

        found = find_eval_file(folder)
        dataset_path = safe_file(folder, found.relative_to(folder).as_posix()) if found else None
        cases = []
        dataset_error = None
        if dataset_path is not None:
            try:
                cases = load_dataset_entries(dataset_path)
            except (ValueError, OSError, yaml.YAMLError) as exc:
                cases, dataset_error = [], str(exc)
        return {
            **metadata,
            "id": skill_id,
            "name": str(header.get("name") or metadata.get("name") or "Untitled skill"),
            "description": str(header.get("description") or ""),
            "content": content,
            "files": [p.relative_to(folder).as_posix() for p in source_files(folder)],
            "cases": cases,
            "dataset_error": dataset_error,
            "native_custom_only": is_native_custom_only(folder),
            "source_hash": source_hash(folder),
            "dataset_hash": hashlib.sha256(dataset_path.read_bytes()).hexdigest() if dataset_path is not None else None,
        }

    def list(self) -> list[dict]:
        result = []
        for folder in sorted(self.skills.iterdir(), reverse=True):
            if IDENTIFIER.fullmatch(folder.name):
                try:
                    result.append(self.describe(folder.name))
                except (ValueError, OSError, json.JSONDecodeError):
                    continue
        return sorted(result, key=lambda x: x["created_at"], reverse=True)

    def create(self, payload: dict) -> dict:
        skill_id = new_id()
        root = self.skills / skill_id
        folder = root / "skill"
        folder.mkdir(mode=0o700, parents=True)
        ignored = []
        try:
            if "archive_base64" in payload:
                try:
                    data = base64.b64decode(payload["archive_base64"], validate=True)
                except (ValueError, TypeError) as exc:
                    raise ValueError("The uploaded file could not be read.") from exc
                if len(data) > MAX_UPLOAD:
                    raise ValueError("Upload a file smaller than 12 MB.")
                filename = str(payload.get("filename", ""))
                if filename.lower().endswith(".zip"):
                    ignored = self._unpack(data, folder)
                elif filename.lower().endswith(".md"):
                    (folder / "SKILL.md").write_text(data.decode("utf-8"), encoding="utf-8")
                else:
                    raise ValueError("Upload SKILL.md or a zip containing one skill.")
            else:
                content = payload.get("content", "")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Add your skill document first.")
                if len(content.encode()) > MAX_UPLOAD:
                    raise ValueError("The skill document is too large.")
                (folder / "SKILL.md").write_text(content, encoding="utf-8")
            write_json(
                root / "metadata.json",
                {
                    "name": str(payload.get("name") or "Untitled skill")[:120],
                    "created_at": now(),
                    "parent_id": None,
                    "ignored_files": ignored,
                },
            )
            return self.describe(skill_id)
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise

    @staticmethod
    def _unpack(data: bytes, destination: Path) -> list[str]:
        ignored = []
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise ValueError("This file is not a readable zip archive.") from exc
        with archive:
            entries = archive.infolist()
            if len(entries) > MAX_FILES or sum(e.file_size for e in entries) > MAX_EXPANDED:
                raise ValueError("Upload one skill with at most 300 files and 24 MB expanded content.")
            # Validate every archive entry, including entries outside the skill subtree.
            for entry in entries:
                safe_file(destination, entry.filename.rstrip("/"))
                mode = entry.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}):
                    raise ValueError("Archive links and special files are not supported.")
                if entry.flag_bits & 1:
                    raise ValueError("Upload an unencrypted zip archive.")
            skills = [e for e in entries if not e.is_dir() and PurePosixPath(e.filename).name == "SKILL.md"]
            if len(skills) != 1:
                raise ValueError("The zip must contain exactly one SKILL.md. Export a single skill folder.")
            prefix = PurePosixPath(skills[0].filename).parent
            seen = set()
            for entry in entries:
                if entry.is_dir():
                    continue
                path = PurePosixPath(entry.filename)
                if not path.is_relative_to(prefix):
                    ignored.append(entry.filename)
                    continue
                relative = path.relative_to(prefix)
                if (
                    any(p in {".git", "__pycache__", ".venv", "node_modules"} for p in relative.parts)
                    or relative.name.startswith(".env")
                    or relative.name in {"credentials.json", "auth.json"}
                ):
                    ignored.append(entry.filename)
                    continue
                key = relative.as_posix()
                if key.casefold() in seen:
                    raise ValueError("The archive contains duplicate or case-conflicting file names.")
                seen.add(key.casefold())
                target = safe_file(destination, key)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as stream:
                    content = stream.read(MAX_EXPANDED + 1)
                if len(content) > MAX_EXPANDED:
                    raise ValueError("The expanded file is too large.")
                target.write_bytes(content)
                target.chmod(0o700 if entry.external_attr >> 16 & 0o111 else 0o600)
        return ignored

    def update(self, skill_id: str, content: str) -> dict:
        if not isinstance(content, str) or not content.strip() or len(content.encode()) > MAX_UPLOAD:
            raise ValueError("Add a nonempty skill document smaller than 12 MB.")
        safe_file(self.directory(skill_id), "SKILL.md").write_text(content, encoding="utf-8")
        return self.describe(skill_id)

    def save_cases(self, skill_id: str, cases: list) -> dict:
        if not isinstance(cases, list) or len(cases) > 100:
            raise ValueError("Save at most 100 test cases.")
        if any(not isinstance(case, dict) for case in cases):
            raise ValueError("Every case must be a JSON object.")
        from skillevaluator.tier3.dataset_utils import normalize_dataset_entries

        cases = normalize_dataset_entries(cases)
        skill_name = self.describe(skill_id)["name"]
        seen = set()
        for case in cases:
            if (
                not isinstance(case, dict)
                or not isinstance(case.get("id"), str)
                or not CASE_IDENTIFIER.fullmatch(case["id"])
            ):
                raise ValueError("Each case needs a unique ID using letters, numbers, hyphens, underscores or dots.")
            if case["id"] in seen:
                raise ValueError(f"Duplicate case ID: {case['id']}")
            seen.add(case["id"])
            if not isinstance(case.get("question"), str) or not case["question"].strip():
                raise ValueError(f"Add a question for {case['id']}.")
            if "expected_skill" not in case:
                case["expected_skill"] = None if case.get("should_trigger") is False else skill_name
        target = safe_file(self.directory(skill_id), "evals/evals.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        authored = []
        for case in cases:
            item = dict(case)
            item["prompt"] = item.pop("question")
            item["expected_output"] = item.pop("ground_truth", item.get("expected_output", ""))
            item["assertions"] = item.pop("expected_behavior", item.get("assertions", []))
            authored.append(item)
        write_json(target, {"skill_name": skill_name, "evals": authored})
        return self.describe(skill_id)

    def candidate(self, skill_id: str, content: str) -> dict:
        if not isinstance(content, str) or not content.strip() or len(content.encode()) > MAX_UPLOAD:
            raise ValueError("Add a nonempty candidate document smaller than 12 MB.")
        parent = self.describe(skill_id)
        candidate_id = new_id()
        root = self.skills / candidate_id
        copy_source(self.directory(skill_id), root / "skill")
        write_json(
            root / "metadata.json",
            {
                "name": parent["name"],
                "created_at": now(),
                "parent_id": skill_id,
                "ignored_files": [],
            },
        )
        return self.update(candidate_id, content)

    def export(self, skill_id: str) -> bytes:
        folder = self.directory(skill_id)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in source_files(folder):
                archive.write(path, path.relative_to(folder).as_posix())
        return buffer.getvalue()

    def sample(self) -> dict:
        skill = self.create({"name": "word-counter", "content": SAMPLE_SKILL})
        folder = self.directory(skill["id"])
        (folder / "scripts").mkdir()
        (folder / "scripts/count_words.py").write_text(SAMPLE_SCRIPT)
        (folder / "evals/files").mkdir(parents=True)
        (folder / "evals/files/input.txt").write_text("red blue red\n")
        return self.save_cases(skill["id"], SAMPLE_CASES)


SAMPLE_SKILL = """---
name: word-counter
description: Count whitespace-separated words in a local text file and return exact JSON. Use when someone asks for a word count; preserve the input file.
license: Apache-2.0
---

# Word counter

Use this skill when someone requests a word count for a local text file.

## Workflow

1. Identify the file requested by the user.
2. Run `python3 scripts/count_words.py FILE`, using this skill's script path.
3. Return the JSON object produced by the script. If the user requests JSON only,
   do not add Markdown fences or explanatory prose.
4. Never change the input file. If it is missing, report that fact; do not invent a count.

## Output

The output contains `words` as a nonnegative JSON integer. Empty files have zero words.
No network, credentials, packages, or external services are needed by the script.
"""

SAMPLE_SCRIPT = """# SPDX-License-Identifier: Apache-2.0
# Count whitespace-separated words without changing the input.
import json
import sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit("Usage: count_words.py FILE")
print(json.dumps({"words": len(Path(sys.argv[1]).read_text(encoding="utf-8").split())}))
"""

SAMPLE_CASES = [
    {
        "id": "count-three-words",
        "question": "Count the words in /workspace/input/input.txt. Return only a JSON object with the words key. Do not modify the input file.",
        "expected_skill": "word-counter",
        "expected_script": "count_words.py",
        "should_trigger": True,
        "ground_truth": '{"words": 3}',
        "expected_behavior": ["Reports exactly 3 words as a JSON integer.", "Leaves the input file unchanged."],
        "files": ["files/input.txt"],
    },
    {
        "id": "unrelated-greeting",
        "question": "Say hello in one short sentence. This is not a request to inspect or count a file.",
        "expected_skill": None,
        "expected_script": None,
        "should_trigger": False,
        "ground_truth": "A short greeting without invoking the word-counter skill or reading files.",
        "expected_behavior": ["Returns a short greeting.", "Does not run a word-count script or inspect files."],
        "files": [],
    },
]
