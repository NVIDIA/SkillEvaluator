---
name: task-list
description: Required for 4+ step requests; add tasks at start and update status after each step.
metadata:
  author: SkillEvaluator Maintainers <maintainers@example.com>
---

# Task List Skill

Track multi-step tasks to ensure nothing is forgotten.

## When to use

- The request requires 4 or more distinct steps
- You need to perform several operations in sequence

## Instructions

1. Add tasks before starting multi-step work.
2. Mark exactly one task in progress while working.
3. Update task status as each step completes.
4. Keep task text concise and action-oriented.

## How to use

Use the `tasks` tool directly. Do NOT use run_code, execute, or write_file for task management.

### 1. Add tasks at the start
```
tasks(operation="add_multiple", content="Step one; Step two; Step three")
```
This clears any previous tasks. Each request starts fresh.

## Examples

```python
tasks(operation="add_multiple", content="Inspect repo; Run tests; Summarize findings")
tasks(operation="start", task_id="task_1")
tasks(operation="complete", task_id="task_1")
```

### 2. Start a task before working on it
```
tasks(operation="start", task_id="task_1")
```

### 3. Complete a task when done
```
tasks(operation="complete", task_id="task_1")
```

### 4. Check what's next
```
tasks(operation="next")
```

## All operations

| Operation | Args | Description |
|-----------|------|-------------|
| `add_multiple` | `content="t1; t2; t3"` | Add tasks (clears previous) |
| `start` | `task_id="task_1"` | Mark in_progress |
| `complete` | `task_id="task_1"` | Mark completed |
| `cancel` | `task_id="task_1"` | Cancel a task |
| `list` | (none) | Show all tasks |
| `next` | (none) | Get next pending task |

## Rules

- One task in_progress at a time
- Start before working, complete when done
- Do not leave tasks hanging
