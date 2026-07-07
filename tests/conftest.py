# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration and fixtures for Skill Evaluator tests.

Based on Skill Evaluator HOW_TO_CONTRIBUTE_SKILLS.md and HOW_TO_CONTRIBUTE_WORKFLOW_RULES.md specifications.
"""

from pathlib import Path

import pytest

# =============================================================================
# SKILLS FIXTURES
# =============================================================================


@pytest.fixture
def sample_skill_dir(tmp_path: Path) -> Path:
    """Create a sample valid skill directory per Skill Evaluator spec.

    Structure: skills/sample-skill/SKILL.md
    Required fields: name, description
    Metadata: author, tags, languages, frameworks, domain
    """
    # Create proper structure: skills/<skill-name>/
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    skill_dir = skills_dir / "sample-skill"
    skill_dir.mkdir()

    # Create valid SKILL.md per Skill Evaluator spec
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("""---
name: sample-skill
description: A sample skill for testing Skill Evaluator validation. Use when demonstrating validation workflows.
metadata:
  author: Test User <testuser@example.com>
  tags:
    - testing
    - sample
  languages:
    - python
  frameworks:
    - pytest
  domain: testing
---

# Sample Skill

This is a sample skill for testing purposes.

## Instructions

1. Run the validator
2. Check the results
3. Fix any reported issues

## Examples

```bash
skillevaluator validate skills/sample-skill/
```
""")

    return skill_dir


@pytest.fixture
def invalid_skill_dir(tmp_path: Path) -> Path:
    """Create a skill directory with invalid SKILL.md (name format violation)."""
    skill_dir = tmp_path / "invalid-skill"
    skill_dir.mkdir()

    # Create invalid SKILL.md - name uses underscore (not kebab-case)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("""---
name: Invalid_Skill_Name
description: A skill with invalid name format for testing validation
---

# Invalid Skill
""")

    return skill_dir


@pytest.fixture
def skill_with_forbidden_fields(tmp_path: Path) -> Path:
    """Create a skill directory with forbidden fields (alwaysApply, globs)."""
    skill_dir = tmp_path / "forbidden-fields"
    skill_dir.mkdir()

    # Create SKILL.md with forbidden fields
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("""---
name: forbidden-fields
description: A skill that uses forbidden fields per Skill Evaluator spec
alwaysApply: true
globs:
  - "*.py"
---

# Forbidden Fields Skill
""")

    return skill_dir


@pytest.fixture
def skill_with_name_mismatch(tmp_path: Path) -> Path:
    """Create a skill where directory name doesn't match frontmatter name."""
    skill_dir = tmp_path / "directory-name"
    skill_dir.mkdir()

    # Create SKILL.md where name doesn't match directory
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("""---
name: different-name
description: A skill where directory name and frontmatter name don't match
---

# Name Mismatch Skill
""")

    return skill_dir


@pytest.fixture
def skill_with_pii(tmp_path: Path) -> Path:
    """Create a skill directory with PII violations."""
    skill_dir = tmp_path / "pii-skill"
    skill_dir.mkdir()

    # Create SKILL.md with PII
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("""---
name: pii-skill
description: A skill that contains PII for testing detection
---

# PII Test Skill

Contact: john.doe@personal-email.com
Path: /Users/johndoe/projects/secret/
Phone: 555-123-4567
""")

    return skill_dir


@pytest.fixture
def skill_with_security_issues(tmp_path: Path) -> Path:
    """Create a skill directory with security vulnerabilities."""
    skill_dir = tmp_path / "insecure-skill"
    skill_dir.mkdir()

    # Create SKILL.md
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("""---
name: insecure-skill
description: A skill with security issues for testing detection
---

# Insecure Skill

This skill has security issues.
""")

    # Create Python file with vulnerabilities
    script_py = skill_dir / "script.py"
    api_key = "sk-" + "1234567890abcdef1234567890abcdef12345678"
    script_py.write_text(f"""
import os
import subprocess

# Hardcoded credentials
api_key = "{api_key}"
password = "MySecretPassword123"

# Dangerous code patterns
def run_command(cmd):
    os.system(cmd)  # Command injection risk
    subprocess.call(cmd, shell=True)  # shell=True risk
    eval(cmd)  # Code injection risk
""")

    return skill_dir


@pytest.fixture
def team_skill_dir(tmp_path: Path) -> Path:
    """Create a valid team-specific skill directory.

    Structure: team-skills/<team-name>/<skill-name>/SKILL.md
    """
    team_skills_dir = tmp_path / "team-skills"
    team_skills_dir.mkdir()
    team_dir = team_skills_dir / "example-team"
    team_dir.mkdir()
    skill_dir = team_dir / "langraph-debugging"
    skill_dir.mkdir()

    # Create valid team skill SKILL.md
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("""---
name: langraph-debugging
description: Debugging workflows for LangGraph applications in Example Team projects. Use when troubleshooting LangGraph mini-graphs.
metadata:
  author: Example Team <team@example.com>
  tags:
    - langraph
    - debugging
    - example-team
  languages:
    - python
  frameworks:
    - langraph
  domain: ai-ml
---

# LangGraph Debugging Workflow

## Overview

This skill provides debugging workflows for LangGraph applications.

## Instructions

1. Identify the failing LangGraph node
2. Enable debug logging
3. Inspect the state transitions

## Examples

```python
graph.invoke({"input": "test"}, config={"debug": True})
```
""")

    return skill_dir


# =============================================================================
# RULES FIXTURES
# =============================================================================


@pytest.fixture
def sample_rule_file(tmp_path: Path) -> Path:
    """Create a sample valid .mdc rule file per Skill Evaluator spec.

    Structure: team-rules/<team-name>/<rule-name>.mdc
    Required fields: alwaysApply, title, description
    Metadata: tags, language, team, domain
    """
    team_rules_dir = tmp_path / "team-rules" / "sample-team"
    team_rules_dir.mkdir(parents=True)

    rule_file = team_rules_dir / "python-standards.mdc"
    rule_file.write_text("""---
alwaysApply: false
title: "Python Coding Standards"
description: "Best practices for Python development in sample-team projects"
globs:
  - '*.py'
metadata:
  tags:
    - python
    - best-practices
    - sample-team
  language: python
  team: sample-team
  domain: backend
---

# Python Coding Standards

## Overview

This rule defines Python coding standards.

## Guidelines

### Do This

- Use type hints
- Write docstrings
- Follow PEP 8

### Don't Do This

- Ignore linter warnings
- Skip tests
""")

    return rule_file


@pytest.fixture
def invalid_rule_file(tmp_path: Path) -> Path:
    """Create a rule file missing required fields."""
    rule_file = tmp_path / "invalid-rule.mdc"
    rule_file.write_text("""---
title: "Missing Required Fields"
description: "This rule is missing alwaysApply"
---

# Invalid Rule
""")

    return rule_file


@pytest.fixture
def team_rule_dir(tmp_path: Path) -> Path:
    """Create a team-rules directory with multiple rule files."""
    team_rules_dir = tmp_path / "team-rules" / "example-team"
    team_rules_dir.mkdir(parents=True)

    # Create multiple rule files
    for name, title in [("error-handling", "Error Handling"), ("testing", "Testing Standards")]:
        rule_file = team_rules_dir / f"{name}.mdc"
        rule_file.write_text(f"""---
alwaysApply: false
title: "{title}"
description: "Guidelines for {name.replace("-", " ")}"
globs:
  - '*.py'
metadata:
  tags:
    - {name}
    - example-team
  language: python
  team: example-team
---

# {title}

## Guidelines

Follow these {name.replace("-", " ")} guidelines.
""")

    return team_rules_dir


# =============================================================================
# WORKFLOWS FIXTURES
# =============================================================================


@pytest.fixture
def sample_workflow_dir(tmp_path: Path) -> Path:
    """Create a sample valid workflow directory per Skill Evaluator spec.

    Structure: workflows/<workflow-name>/
    Required files: README.md, workflow-rules.mdc
    Required dirs: references/ with at least one .mdc file
    """
    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()
    workflow_dir = workflows_dir / "sample-workflow"
    workflow_dir.mkdir()

    # Create README.md
    readme = workflow_dir / "README.md"
    readme.write_text("""# Sample Workflow

## Overview

A sample workflow for testing Skill Evaluator validation.

## Prerequisites

- Python 3.11+

## Quick Start

1. Run validation
2. Check results
""")

    # Create workflow-rules.mdc
    manifest = workflow_dir / "workflow-rules.mdc"
    manifest.write_text("""---
alwaysApply: false
title: "Sample Workflow"
description: "A sample workflow for testing purposes"
globs:
  - '*.py'
metadata:
  author: "Test User <testuser@example.com>"
  tags:
    - testing
    - sample
  language: python
  domain: testing
---

# Sample Workflow

## Overview

This is a sample workflow.

## Step 1: Setup

Configure the environment.

## Step 2: Run

Execute the workflow.
""")

    # Create references directory
    refs_dir = workflow_dir / "references"
    refs_dir.mkdir()

    # Create reference file
    ref_file = refs_dir / "setup-guide.mdc"
    ref_file.write_text("""---
alwaysApply: false
title: "Setup Guide"
description: "Detailed setup instructions"
parent_workflow: "sample-workflow"
---

# Setup Guide

## Installation

Follow these installation steps.
""")

    return workflow_dir


@pytest.fixture
def invalid_workflow_dir(tmp_path: Path) -> Path:
    """Create a workflow directory missing required components."""
    workflow_dir = tmp_path / "invalid-workflow"
    workflow_dir.mkdir()

    # Only create workflow-rules.mdc, missing README.md and references/
    manifest = workflow_dir / "workflow-rules.mdc"
    manifest.write_text("""---
alwaysApply: false
title: "Invalid Workflow"
description: "Missing required files"
metadata:
  author: "Test <test@example.com>"
---

# Invalid Workflow
""")

    return workflow_dir


@pytest.fixture
def team_workflow_dir(tmp_path: Path) -> Path:
    """Create a valid team-specific workflow directory.

    Structure: team-workflows/<team-name>/<workflow-name>/
    """
    team_workflows_dir = tmp_path / "team-workflows" / "example-team"
    team_workflows_dir.mkdir(parents=True)
    workflow_dir = team_workflows_dir / "langraph-debugging"
    workflow_dir.mkdir()

    # Create README.md
    readme = workflow_dir / "README.md"
    readme.write_text("""# LangGraph Debugging Workflow

## Overview

Debugging workflows for LangGraph applications in Example Team.
""")

    # Create workflow-rules.mdc
    manifest = workflow_dir / "workflow-rules.mdc"
    manifest.write_text("""---
alwaysApply: false
title: "LangGraph Debugging"
description: "Debugging workflows for LangGraph applications"
metadata:
  author: "Example Team <team@example.com>"
  tags:
    - langraph
    - debugging
    - example-team
  language: python
  framework: langraph
  team: example-team
  project: skill_evaluator
---

# LangGraph Debugging

## Overview

Debug LangGraph applications effectively.
""")

    # Create references directory
    refs_dir = workflow_dir / "references"
    refs_dir.mkdir()

    ref_file = refs_dir / "common-errors.mdc"
    ref_file.write_text("""---
alwaysApply: false
title: "Common Errors"
description: "Common LangGraph errors and solutions"
parent_workflow: "langraph-debugging"
---

# Common Errors

## State Issues

Handle state management problems.
""")

    return workflow_dir
