#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Task List Script for Task List Skill

This script provides task management for tracking multi-step operations.
State is stored in a markdown file so it persists across runs.

Usage (CLI - for testing):
    python tasks.py <operation> [--task_id ID] [--content TEXT]
"""

import argparse
import os
import re
import shlex
import sys
from pathlib import Path


def _state_path() -> Path:
    """Resolve the task list state file path.

    Priority:
    1. TASK_LIST_STATE_PATH env var (explicit override)
    2. WORKSPACE_MOUNT_PATH env var + /tasks/todo.md
    3. AGENT_WORKSPACE_PATH env var + /tasks/todo.md (legacy)
    4. ~/agent-workspace/tasks/todo.md (safe local fallback, never source code)

    Note: Same priority as resolve_workspace_root() in workspace.py,
    but inlined here because tasks.py is a standalone script that may
    run outside the skills_agent package (e.g., via subprocess in sandbox).
    """
    env_path = os.environ.get("TASK_LIST_STATE_PATH")
    if env_path:
        return Path(env_path)
    ws = os.environ.get("WORKSPACE_MOUNT_PATH") or os.environ.get("AGENT_WORKSPACE_PATH")
    if ws:
        return Path(ws) / "tasks" / "todo.md"
    return Path.home() / "agent-workspace" / "tasks" / "todo.md"


def _load_state(state_path: Path) -> tuple[dict, str | None]:
    """Load state from markdown file, returning (state, error)."""
    if not state_path.exists():
        return {"tasks": {}, "counter": 0}, None

    try:
        content = state_path.read_text(encoding="utf-8")
    except Exception as exc:
        return {"tasks": {}, "counter": 0}, f"Failed to read state file: {exc}"

    # Parse markdown checkbox format
    # - [ ] task_1: Description (pending)
    # - [→] task_2: Description (in_progress)
    # - [x] task_3: Description (completed)
    # - [~] task_4: Description (cancelled)

    tasks = {}
    counter = 0

    # Pattern: - [X] task_ID: content (status)
    pattern = r"^- \[(.)\] (task_\d+): (.+?) \((pending|in_progress|completed|cancelled)\)"

    for line in content.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            _checkbox, task_id, task_content, status = match.groups()
            tasks[task_id] = {"id": task_id, "content": task_content, "status": status}
            # Extract counter from task_id (task_5 -> 5)
            task_num = int(task_id.split("_")[1])
            counter = max(counter, task_num)

    return {"tasks": tasks, "counter": counter}, None


def _save_state(state_path: Path, state: dict) -> str | None:
    """Persist state to disk in markdown format, returning error if any."""
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)

        # Group tasks by status
        by_status = {"pending": [], "in_progress": [], "completed": [], "cancelled": []}

        for task in state["tasks"].values():
            status = task.get("status", "pending")
            if status in by_status:
                by_status[status].append(task)

        # Build markdown content
        lines = ["# Tasks\n"]

        status_config = {
            "pending": ("Pending", "[ ]"),
            "in_progress": ("In Progress", "[→]"),
            "completed": ("Completed", "[x]"),
            "cancelled": ("Cancelled", "[~]"),
        }

        for status_key, (section_name, checkbox) in status_config.items():
            lines.append(f"## {section_name}")
            if by_status[status_key]:
                for task in by_status[status_key]:
                    lines.append(f"- {checkbox} {task['id']}: {task['content']} ({status_key})")
            else:
                lines.append("<!-- No tasks -->")
            lines.append("")

        content = "\n".join(lines)
        state_path.write_text(content, encoding="utf-8")
        return None
    except Exception as exc:
        return f"Failed to write state file: {exc}"


class TaskManager:
    """Manages tasks with file-backed state."""

    def __init__(self, state: dict):
        """
        Initialize TaskManager with file-backed state.

        Args:
            state: State dict {"tasks": {}, "counter": 0} - modified in place
        """
        self._state = state
        # Ensure state has required keys
        if "tasks" not in self._state:
            self._state["tasks"] = {}
        if "counter" not in self._state:
            self._state["counter"] = 0

    @property
    def tasks(self) -> dict:
        return self._state["tasks"]

    @property
    def counter(self) -> int:
        return self._state["counter"]

    @counter.setter
    def counter(self, value: int):
        self._state["counter"] = value

    def add_task(self, content: str) -> dict:
        """Add a single task."""
        if not content:
            return {"error": "Content is required"}

        self.counter += 1
        task_id = f"task_{self.counter}"
        self.tasks[task_id] = {
            "id": task_id,
            "content": content,
            "status": "pending",
        }
        return {
            "operation": "add",
            "task": self.tasks[task_id],
            "total_tasks": len(self.tasks),
        }

    def add_multiple(self, content: str) -> dict:
        """Add multiple tasks (semicolon or newline separated). Clears existing tasks first."""
        if not content:
            return {"error": "Content is required"}

        items = re.split(r"[;\n]", content)
        items = [t.strip() for t in items if t.strip()]

        if not items:
            return {"error": "No valid tasks found"}

        # Clear existing tasks - each request starts fresh
        self.tasks.clear()
        self.counter = 0

        added = []
        for item in items:
            self.counter += 1
            task_id = f"task_{self.counter}"
            self.tasks[task_id] = {
                "id": task_id,
                "content": item,
                "status": "pending",
            }
            added.append(self.tasks[task_id])

        return {
            "operation": "add_multiple",
            "tasks_added": added,
            "count": len(added),
            "total_tasks": len(self.tasks),
        }

    def start_task(self, task_id: str) -> dict:
        """Mark a task as in_progress."""
        if not task_id:
            return {"error": "task_id is required"}
        if task_id not in self.tasks:
            return {"error": f"Task '{task_id}' not found", "available": list(self.tasks.keys())}

        self.tasks[task_id]["status"] = "in_progress"
        return {
            "operation": "start",
            "task": self.tasks[task_id],
        }

    def complete_task(self, task_id: str) -> dict:
        """Mark a task as completed."""
        if not task_id:
            return {"error": "task_id is required"}
        if task_id not in self.tasks:
            return {"error": f"Task '{task_id}' not found", "available": list(self.tasks.keys())}

        self.tasks[task_id]["status"] = "completed"

        completed = sum(1 for t in self.tasks.values() if t["status"] == "completed")
        remaining = [t for t in self.tasks.values() if t["status"] in ["pending", "in_progress"]]

        return {
            "operation": "complete",
            "task": self.tasks[task_id],
            "progress": f"{completed}/{len(self.tasks)} completed",
            "remaining": remaining,
        }

    def cancel_task(self, task_id: str) -> dict:
        """Cancel a task."""
        if not task_id:
            return {"error": "task_id is required"}
        if task_id not in self.tasks:
            return {"error": f"Task '{task_id}' not found", "available": list(self.tasks.keys())}

        self.tasks[task_id]["status"] = "cancelled"
        return {
            "operation": "cancel",
            "task": self.tasks[task_id],
        }

    def list_tasks(self) -> dict:
        """List all tasks grouped by status."""
        by_status = {"in_progress": [], "pending": [], "completed": [], "cancelled": []}

        for task in self.tasks.values():
            status = task.get("status", "pending")
            if status in by_status:
                by_status[status].append(task)

        completed = len(by_status["completed"])
        total = len(self.tasks)

        return {
            "operation": "list",
            "in_progress": by_status["in_progress"],
            "pending": by_status["pending"],
            "completed": by_status["completed"],
            "cancelled": by_status["cancelled"],
            "progress": f"{completed}/{total} completed" if total else "0/0 completed",
        }

    def next_task(self) -> dict:
        """Get the next pending or in_progress task."""
        for task in self.tasks.values():
            if task["status"] == "in_progress":
                return {"operation": "next", "task": task, "message": "Current task in progress"}

        for task in self.tasks.values():
            if task["status"] == "pending":
                return {"operation": "next", "task": task, "message": "Next pending task"}

        return {"operation": "next", "task": None, "message": "No pending tasks"}

    def clear_tasks(self) -> dict:
        """Clear all tasks."""
        count = len(self.tasks)
        self._state["tasks"] = {}
        self._state["counter"] = 0
        return {"operation": "clear", "cleared": count}


def format_result(data: dict) -> str:
    """Format result for human-readable output."""
    if "error" in data:
        return f"Error: {data['error']}"

    op = data.get("operation", "")

    if op == "add":
        t = data["task"]
        return f"Added [{t['id']}]: {t['content']}"

    if op == "add_multiple":
        lines = [f"Added {data['count']} tasks:"]
        for t in data["tasks_added"]:
            lines.append(f"  - [{t['id']}] {t['content']}")
        return "\n".join(lines)

    if op == "start":
        t = data["task"]
        return f"Started [{t['id']}]: {t['content']}"

    if op == "complete":
        t = data["task"]
        lines = [f"Completed [{t['id']}]: {t['content']}", f"Progress: {data['progress']}"]
        if data["remaining"]:
            lines.append("Remaining:")
            for r in data["remaining"]:
                lines.append(f"  - [{r['id']}] {r['content']} ({r['status']})")
        return "\n".join(lines)

    if op == "cancel":
        t = data["task"]
        return f"Cancelled [{t['id']}]: {t['content']}"

    if op == "list":
        has_tasks = any(
            [data.get("in_progress", []), data.get("pending", []), data.get("completed", []), data.get("cancelled", [])]
        )
        if not has_tasks:
            return "No tasks"
        lines = [f"Task List ({data.get('progress', '0/0')})"]
        if data.get("in_progress"):
            lines.append("\nIn Progress:")
            for t in data["in_progress"]:
                lines.append(f"  - [{t['id']}] {t['content']}")
        if data.get("pending"):
            lines.append("\nPending:")
            for t in data["pending"]:
                lines.append(f"  - [{t['id']}] {t['content']}")
        if data.get("completed"):
            lines.append("\nCompleted:")
            for t in data["completed"]:
                lines.append(f"  - [{t['id']}] {t['content']}")
        return "\n".join(lines)

    if op == "next":
        if data["task"]:
            t = data["task"]
            return f"Next: [{t['id']}] {t['content']}"
        return "No pending tasks remaining"

    if op == "clear":
        return f"Cleared {data['cleared']} tasks"

    # Fallback: format unknown result as plain text
    lines = []
    for k, v in data.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def _parse_args(args: str) -> tuple[str | None, str | None, str | None, str | None]:
    """Parse raw args into (operation, task_id, content, error)."""
    try:
        parts = shlex.split(args) if args else []
    except ValueError:
        parts = args.split() if args else []

    if not parts:
        valid_ops = "list, add, add_multiple, start, complete, cancel, next, clear"
        return None, None, None, f"Error: No operation specified. Use: {valid_ops}"

    operation = parts[0]
    task_id = None
    content = None

    i = 1
    while i < len(parts):
        if parts[i] in ("--task_id", "-t") and i + 1 < len(parts):
            task_id = parts[i + 1]
            i += 2
        elif parts[i] in ("--content", "-c") and i + 1 < len(parts):
            content = parts[i + 1]
            i += 2
        else:
            i += 1

    return operation, task_id, content, None


def _execute_operation(operation: str, task_id: str | None, content: str | None, state_path: Path) -> dict:
    """Execute an operation with persisted state."""
    state, load_error = _load_state(state_path)
    if load_error and operation != "clear":
        return {"error": load_error}

    manager = TaskManager(state=state)
    operations = {
        "add": lambda: manager.add_task(content or ""),
        "add_multiple": lambda: manager.add_multiple(content or ""),
        "start": lambda: manager.start_task(task_id or ""),
        "complete": lambda: manager.complete_task(task_id or ""),
        "cancel": lambda: manager.cancel_task(task_id or ""),
        "list": manager.list_tasks,
        "next": manager.next_task,
        "clear": manager.clear_tasks,
    }

    if operation not in operations:
        valid_ops = "list, add, add_multiple, start, complete, cancel, next, clear"
        return {"error": f"Unknown operation '{operation}'. Use: {valid_ops}"}

    result = operations[operation]()
    if "error" not in result:
        save_error = _save_state(state_path, state)
        if save_error:
            return {"error": save_error}

    return result


def handle_operation(args: str) -> str:
    """
    Handle a task operation with persisted state.

    This is the main entry point for module usage.

    Args:
        args: Command line style arguments (e.g., "add --content 'My task'")

    Returns:
        Human-readable result string
    """
    operation, task_id, content, error = _parse_args(args)
    if error:
        return error

    result = _execute_operation(operation, task_id, content, _state_path())
    return format_result(result)


def main():
    """CLI entry point for testing (file-backed)."""
    parser = argparse.ArgumentParser(description="Manage task list")
    parser.add_argument(
        "operation",
        choices=["add", "add_multiple", "start", "complete", "cancel", "list", "next", "clear"],
        help="Operation to perform",
    )
    parser.add_argument("--task_id", "-t", help="Task ID (for start/complete/cancel)")
    parser.add_argument("--content", "-c", help="Task content (for add/add_multiple)")

    args = parser.parse_args()

    # Build args string and use handle_operation
    arg_parts = [args.operation]
    if args.task_id:
        arg_parts.extend(["--task_id", args.task_id])
    if args.content:
        arg_parts.extend(["--content", args.content])

    raw_args = shlex.join(arg_parts)
    result = handle_operation(raw_args)
    if result.startswith("Error:"):
        print(result, file=sys.stderr)
        sys.exit(1)

    print(result)


if __name__ == "__main__":
    main()
