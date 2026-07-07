# Task List Patterns

## Pattern 1: Sequential Operations

For tasks that must be done in order:

```
tasks.py add_multiple --content "1. Backup config; 2. Update config; 3. Validate; 4. Restart"
```

Work through sequentially, completing each before starting the next.

## Pattern 2: Research Then Act

For tasks requiring information gathering first:

```
tasks.py add_multiple --content "Research: Find info; Plan: Determine approach; Execute: Implement; Verify: Confirm"
```

## Pattern 3: Multi-File Changes

For changes spanning multiple files:

```
tasks.py add_multiple --content "Update models.py; Update views.py; Update tests.py; Update docs"
```

## Pattern 4: Conditional Tasks

Add tasks as you discover them:

```
# Initial
tasks.py add --content "Investigate the issue"

# After investigation
tasks.py add_multiple --content "Fix parser.py; Add unit test; Update error message"
```

## Anti-Patterns to Avoid

### Too Granular
Don't create a task for every line of code.

### Too Vague
Don't create tasks that aren't actionable like "Fix everything".

### Just Right
Create specific, actionable tasks like "Refactor auth module to use JWT".
