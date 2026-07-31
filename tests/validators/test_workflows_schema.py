# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Workflows schema validation.

Based on SkillEvaluator HOW_TO_CONTRIBUTE_WORKFLOW_RULES.md specification.
"""

from pathlib import Path

import pytest

from skillevaluator.validators.workflows_schema import WorkflowsSchemaValidator, parse_workflows_manifest


class TestWorkflowsSchemaValidator:
    """Test suite for WorkflowsSchemaValidator."""

    @pytest.fixture
    def validator(self) -> WorkflowsSchemaValidator:
        """Create a WorkflowsSchemaValidator instance."""
        return WorkflowsSchemaValidator()

    @pytest.fixture
    def valid_workflow_dir(self, tmp_path: Path) -> Path:
        """Create a valid workflow directory per SkillEvaluator spec."""
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        workflow_dir = workflows_dir / "fastapi-setup"
        workflow_dir.mkdir()

        # Create README.md
        readme = workflow_dir / "README.md"
        readme.write_text("""# FastAPI Setup Workflow

## Overview

This workflow guides you through setting up a FastAPI service.

## Prerequisites

- Python 3.11+
- Docker

## Quick Start

1. Run setup script
2. Configure environment
3. Start development
""")

        # Create workflow-rules.mdc
        manifest = workflow_dir / "workflow-rules.mdc"
        manifest.write_text("""---
alwaysApply: false
title: "FastAPI Service Setup"
description: "Complete guide for setting up FastAPI microservices"
globs:
  - '*.py'
  - 'app/**/*.py'
metadata:
  author: "Test User <testuser@example.com>"
  tags:
    - fastapi
    - python
    - microservices
  language: python
  framework: fastapi
  team: platform-infra
  domain: backend
---

# FastAPI Service Setup Workflow

## Overview

This workflow provides step-by-step instructions for FastAPI setup.

## Step 1: Project Setup

Create the project structure.

## Step 2: Configuration

Configure the service.
""")

        # Create references directory
        refs_dir = workflow_dir / "references"
        refs_dir.mkdir()

        # Create reference file
        ref_file = refs_dir / "api-design.mdc"
        ref_file.write_text("""---
alwaysApply: false
title: "API Design Reference"
description: "API design guidelines for FastAPI services"
parent_workflow: "fastapi-setup"
---

# API Design Reference

## Endpoint Naming

Use RESTful conventions.
""")

        return workflow_dir

    @pytest.fixture
    def incomplete_workflow_dir(self, tmp_path: Path) -> Path:
        """Create a workflow directory missing required files."""
        workflow_dir = tmp_path / "incomplete-workflow"
        workflow_dir.mkdir()

        # Only create workflow-rules.mdc, missing README.md and references/
        manifest = workflow_dir / "workflow-rules.mdc"
        manifest.write_text("""---
alwaysApply: false
title: "Incomplete Workflow"
description: "Missing required files"
metadata:
  author: "Test User <testuser@example.com>"
---

# Incomplete Workflow
""")

        return workflow_dir

    @pytest.fixture
    def workflow_missing_author(self, tmp_path: Path) -> Path:
        """Create a workflow directory with missing author."""
        workflow_dir = tmp_path / "no-author-workflow"
        workflow_dir.mkdir()

        # Create minimal structure
        (workflow_dir / "README.md").write_text("# Workflow")
        refs_dir = workflow_dir / "references"
        refs_dir.mkdir()
        (refs_dir / "ref.mdc").write_text("""---
alwaysApply: false
title: "Reference"
description: "A reference"
---

# Reference
""")

        # Create manifest without author
        manifest = workflow_dir / "workflow-rules.mdc"
        manifest.write_text("""---
alwaysApply: false
title: "No Author Workflow"
description: "Missing author in metadata"
metadata:
  tags:
    - test
---

# No Author
""")

        return workflow_dir

    def test_validate_valid_workflow(self, validator: WorkflowsSchemaValidator, valid_workflow_dir: Path):
        """Test validation passes for valid workflow directory."""
        result = validator.validate(valid_workflow_dir)
        assert result.passed, f"Expected validation to pass. Errors: {result.errors}"
        assert any("fastapi-setup" in msg.lower() for msg in result.messages)

    def test_validate_missing_readme(self, validator: WorkflowsSchemaValidator, incomplete_workflow_dir: Path):
        """Test validation fails when README.md is missing."""
        result = validator.validate(incomplete_workflow_dir)
        assert not result.passed
        assert any("README.md" in err for err in result.errors)

    def test_validate_missing_references(self, validator: WorkflowsSchemaValidator, incomplete_workflow_dir: Path):
        """Test validation fails when references/ is missing."""
        result = validator.validate(incomplete_workflow_dir)
        assert not result.passed
        assert any("references" in err.lower() for err in result.errors)

    def test_validate_missing_author(self, validator: WorkflowsSchemaValidator, workflow_missing_author: Path):
        """Test validation fails when author is missing in metadata."""
        result = validator.validate(workflow_missing_author)
        assert not result.passed
        assert any("author" in err.lower() for err in result.errors)

    def test_validate_folder_structure(self, validator: WorkflowsSchemaValidator, valid_workflow_dir: Path):
        """Test folder structure validation for workflows."""
        result = validator.validate(valid_workflow_dir)
        assert result.passed
        assert any("workflows" in msg.lower() for msg in result.messages)

    def test_validate_naming_conventions(self, validator: WorkflowsSchemaValidator, tmp_path: Path):
        """Test naming convention validation for workflow directory."""
        # Create workflow with invalid name but complete structure
        bad_workflow_dir = tmp_path / "BadName_Workflow"
        bad_workflow_dir.mkdir()

        # Create README.md
        (bad_workflow_dir / "README.md").write_text("# Bad Name Workflow")

        # Create references directory with at least one .mdc
        refs_dir = bad_workflow_dir / "references"
        refs_dir.mkdir()
        (refs_dir / "ref.mdc").write_text("""---
alwaysApply: false
title: "Reference"
description: "A reference"
---

# Reference
""")

        # Create manifest
        manifest = bad_workflow_dir / "workflow-rules.mdc"
        manifest.write_text("""---
alwaysApply: false
title: "Bad Name"
description: "Workflow with non-kebab-case name"
metadata:
  author: "Test <test@example.com>"
---

# Bad Name
""")

        result = validator.validate(bad_workflow_dir)
        assert any("kebab-case" in err.lower() for err in result.errors)

    def test_validate_reference_files(self, validator: WorkflowsSchemaValidator, valid_workflow_dir: Path):
        """Test that reference files are validated."""
        result = validator.validate(valid_workflow_dir)
        assert result.passed
        assert any("reference" in msg.lower() for msg in result.messages)

    def test_validate_empty_references_dir(self, validator: WorkflowsSchemaValidator, tmp_path: Path):
        """Test validation fails when references/ has no .mdc files."""
        workflow_dir = tmp_path / "empty-refs-workflow"
        workflow_dir.mkdir()

        (workflow_dir / "README.md").write_text("# Workflow")
        (workflow_dir / "references").mkdir()  # Empty references dir

        manifest = workflow_dir / "workflow-rules.mdc"
        manifest.write_text("""---
alwaysApply: false
title: "Empty Refs"
description: "Workflow with empty references"
metadata:
  author: "Test <test@example.com>"
---

# Empty Refs
""")

        result = validator.validate(workflow_dir)
        assert not result.passed
        assert any("at least one .mdc" in err.lower() for err in result.errors)

    def test_validate_multiple_workflows(self, validator: WorkflowsSchemaValidator, tmp_path: Path):
        """Test validation of folder containing multiple workflows."""
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()

        # Create two valid workflows
        for name in ["workflow-one", "workflow-two"]:
            wf_dir = workflows_dir / name
            wf_dir.mkdir()
            (wf_dir / "README.md").write_text(f"# {name}")
            refs_dir = wf_dir / "references"
            refs_dir.mkdir()
            (refs_dir / "ref.mdc").write_text(f"""---
alwaysApply: false
title: "Ref for {name}"
description: "Reference file"
---

# Reference
""")
            (wf_dir / "workflow-rules.mdc").write_text(f"""---
alwaysApply: false
title: "{name.replace("-", " ").title()}"
description: "Description for {name}"
metadata:
  author: "Test <test@example.com>"
---

# {name}
""")

        result = validator.validate(workflows_dir)
        assert result.passed
        assert any("2 workflow(s)" in msg for msg in result.messages)

    def test_parse_workflows_manifest(self, valid_workflow_dir: Path):
        """Test parse_workflows_manifest utility function."""
        manifest = parse_workflows_manifest(valid_workflow_dir)
        assert manifest is not None
        assert manifest.title == "FastAPI Service Setup"
        assert manifest.author == "Test User <testuser@example.com>"
        assert manifest.has_readme is True
        assert manifest.has_references is True
        assert len(manifest.reference_files) > 0


class TestWorkflowsRequiredFields:
    """Test that Workflows require specific fields."""

    @pytest.fixture
    def validator(self) -> WorkflowsSchemaValidator:
        return WorkflowsSchemaValidator()

    def test_always_apply_is_required(self, validator: WorkflowsSchemaValidator, tmp_path: Path):
        """Test that alwaysApply field is required for Workflows."""
        workflow_dir = self._create_minimal_workflow(
            tmp_path,
            """---
title: "Test Workflow"
description: "Missing alwaysApply"
metadata:
  author: "Test <test@example.com>"
---

# Test
""",
        )
        result = validator.validate(workflow_dir)
        assert not result.passed
        assert any("alwaysApply" in err for err in result.errors)

    def test_title_is_required(self, validator: WorkflowsSchemaValidator, tmp_path: Path):
        """Test that title field is required for Workflows."""
        workflow_dir = self._create_minimal_workflow(
            tmp_path,
            """---
alwaysApply: false
description: "Missing title"
metadata:
  author: "Test <test@example.com>"
---

# Test
""",
        )
        result = validator.validate(workflow_dir)
        assert not result.passed
        assert any("title" in err for err in result.errors)

    def test_metadata_author_is_required(self, validator: WorkflowsSchemaValidator, tmp_path: Path):
        """Test that metadata.author field is required for Workflows."""
        workflow_dir = self._create_minimal_workflow(
            tmp_path,
            """---
alwaysApply: false
title: "No Author"
description: "Missing author in metadata"
metadata:
  tags:
    - test
---

# Test
""",
        )
        result = validator.validate(workflow_dir)
        assert not result.passed
        assert any("author" in err.lower() for err in result.errors)

    def _create_minimal_workflow(self, tmp_path: Path, manifest_content: str) -> Path:
        """Helper to create minimal workflow structure."""
        workflow_dir = tmp_path / "test-workflow"
        workflow_dir.mkdir()
        (workflow_dir / "README.md").write_text("# Test")
        refs_dir = workflow_dir / "references"
        refs_dir.mkdir()
        (refs_dir / "ref.mdc").write_text("""---
alwaysApply: false
title: "Reference"
description: "A reference"
---

# Reference
""")
        (workflow_dir / "workflow-rules.mdc").write_text(manifest_content)
        return workflow_dir


class TestTeamWorkflows:
    """Test team-specific workflows validation."""

    @pytest.fixture
    def validator(self) -> WorkflowsSchemaValidator:
        return WorkflowsSchemaValidator()

    @pytest.fixture
    def valid_team_workflow(self, tmp_path: Path) -> Path:
        """Create a valid team-specific workflow."""
        team_workflows_dir = tmp_path / "team-workflows" / "example-team"
        team_workflows_dir.mkdir(parents=True)
        workflow_dir = team_workflows_dir / "langraph-patterns"
        workflow_dir.mkdir()

        (workflow_dir / "README.md").write_text("# LangGraph Patterns")
        refs_dir = workflow_dir / "references"
        refs_dir.mkdir()
        (refs_dir / "debugging.mdc").write_text("""---
alwaysApply: false
title: "Debugging Guide"
description: "LangGraph debugging patterns"
parent_workflow: "langraph-patterns"
---

# Debugging
""")

        (workflow_dir / "workflow-rules.mdc").write_text("""---
alwaysApply: false
title: "LangGraph Patterns"
description: "Common patterns for LangGraph in IPP"
metadata:
  author: "Example Team <team@example.com>"
  tags:
    - langraph
    - example-team
  team: example-team
  project: skill_evaluator
---

# LangGraph Patterns
""")

        return workflow_dir

    def test_validate_team_workflow_structure(self, validator: WorkflowsSchemaValidator, valid_team_workflow: Path):
        """Test validation of team-workflows structure."""
        result = validator.validate(valid_team_workflow)
        assert result.passed
        assert any("team-workflows" in msg.lower() for msg in result.messages)

    def test_parent_workflow_mismatch_warning(self, validator: WorkflowsSchemaValidator, tmp_path: Path):
        """Test warning when parent_workflow doesn't match workflow name."""
        workflow_dir = tmp_path / "my-workflow"
        workflow_dir.mkdir()

        (workflow_dir / "README.md").write_text("# My Workflow")
        refs_dir = workflow_dir / "references"
        refs_dir.mkdir()
        (refs_dir / "ref.mdc").write_text("""---
alwaysApply: false
title: "Reference"
description: "A reference with wrong parent"
parent_workflow: "different-workflow"
---

# Reference
""")

        (workflow_dir / "workflow-rules.mdc").write_text("""---
alwaysApply: false
title: "My Workflow"
description: "Workflow with mismatched parent reference"
metadata:
  author: "Test <test@example.com>"
---

# My Workflow
""")

        result = validator.validate(workflow_dir)
        # Should warn about parent_workflow mismatch
        assert any("parent_workflow" in warn.lower() for warn in result.warnings)


class TestWorkflowsScripts:
    """Tests for workflows scripts validation."""

    def test_validate_scripts_with_valid_shell_script(self, tmp_path: Path):
        """Test validation of valid shell script with shebang."""
        validator = WorkflowsSchemaValidator()
        workflow_dir = tmp_path / "workflows" / "script-workflow"
        workflow_dir.mkdir(parents=True)

        (workflow_dir / "README.md").write_text("# Script Workflow")
        refs_dir = workflow_dir / "references"
        refs_dir.mkdir()
        (refs_dir / "ref.mdc").write_text("""---
alwaysApply: false
title: "Reference"
description: "A reference"
---

# Reference
""")

        scripts_dir = refs_dir / "scripts"
        scripts_dir.mkdir()
        script_file = scripts_dir / "setup.sh"
        script_file.write_text("""#!/bin/bash
# Description: Setup script for workflow
# Usage: ./setup.sh

echo "Setting up..."
""")

        (workflow_dir / "workflow-rules.mdc").write_text("""---
alwaysApply: false
title: "Script Workflow"
description: "Workflow with scripts"
metadata:
  author: "Test <test@example.com>"
---

# Script Workflow
""")

        result = validator.validate(workflow_dir)
        assert result.passed
        assert any("script" in msg.lower() for msg in result.messages)

    def test_validate_scripts_missing_shebang(self, tmp_path: Path):
        """Test validation warns about shell script without shebang."""
        validator = WorkflowsSchemaValidator()
        workflow_dir = tmp_path / "workflows" / "no-shebang-workflow"
        workflow_dir.mkdir(parents=True)

        (workflow_dir / "README.md").write_text("# No Shebang Workflow")
        refs_dir = workflow_dir / "references"
        refs_dir.mkdir()
        (refs_dir / "ref.mdc").write_text("""---
alwaysApply: false
title: "Reference"
description: "A reference"
---

# Reference
""")

        scripts_dir = refs_dir / "scripts"
        scripts_dir.mkdir()
        script_file = scripts_dir / "bad-script.sh"
        script_file.write_text("""echo "No shebang!"
""")

        (workflow_dir / "workflow-rules.mdc").write_text("""---
alwaysApply: false
title: "No Shebang Workflow"
description: "Workflow with bad scripts"
metadata:
  author: "Test <test@example.com>"
---

# No Shebang Workflow
""")

        result = validator.validate(workflow_dir)
        # Should warn about missing shebang
        assert any("shebang" in warn.lower() for warn in result.warnings)

    def test_validate_scripts_missing_documentation(self, tmp_path: Path):
        """Test validation warns about script without documentation."""
        validator = WorkflowsSchemaValidator()
        workflow_dir = tmp_path / "workflows" / "no-docs-workflow"
        workflow_dir.mkdir(parents=True)

        (workflow_dir / "README.md").write_text("# No Docs Workflow")
        refs_dir = workflow_dir / "references"
        refs_dir.mkdir()
        (refs_dir / "ref.mdc").write_text("""---
alwaysApply: false
title: "Reference"
description: "A reference"
---

# Reference
""")

        scripts_dir = refs_dir / "scripts"
        scripts_dir.mkdir()
        script_file = scripts_dir / "no-docs.sh"
        script_file.write_text("""#!/bin/bash
echo "No documentation!"
""")

        (workflow_dir / "workflow-rules.mdc").write_text("""---
alwaysApply: false
title: "No Docs Workflow"
description: "Workflow with undocumented scripts"
metadata:
  author: "Test <test@example.com>"
---

# No Docs Workflow
""")

        result = validator.validate(workflow_dir)
        # Should warn about missing documentation
        assert any("description" in warn.lower() or "usage" in warn.lower() for warn in result.warnings)

    def test_validate_empty_scripts_directory(self, tmp_path: Path):
        """Test validation of empty scripts directory."""
        validator = WorkflowsSchemaValidator()
        workflow_dir = tmp_path / "workflows" / "empty-scripts-workflow"
        workflow_dir.mkdir(parents=True)

        (workflow_dir / "README.md").write_text("# Empty Scripts Workflow")
        refs_dir = workflow_dir / "references"
        refs_dir.mkdir()
        (refs_dir / "ref.mdc").write_text("""---
alwaysApply: false
title: "Reference"
description: "A reference"
---

# Reference
""")

        scripts_dir = refs_dir / "scripts"
        scripts_dir.mkdir()
        # Empty scripts directory

        (workflow_dir / "workflow-rules.mdc").write_text("""---
alwaysApply: false
title: "Empty Scripts Workflow"
description: "Workflow with empty scripts dir"
metadata:
  author: "Test <test@example.com>"
---

# Empty Scripts Workflow
""")

        result = validator.validate(workflow_dir)
        assert result.passed  # Empty scripts dir is valid


class TestWorkflowsFolderCompliance:
    """Tests for workflows folder compliance."""

    def test_validate_folder_with_workflows_subdir(self, tmp_path: Path):
        """Test validation of folder containing workflows/ subdirectory."""
        validator = WorkflowsSchemaValidator()

        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()

        workflow = workflows_dir / "my-workflow"
        workflow.mkdir()

        (workflow / "README.md").write_text("# My Workflow")
        refs_dir = workflow / "references"
        refs_dir.mkdir()
        (refs_dir / "ref.mdc").write_text("""---
alwaysApply: false
title: "Reference"
description: "A reference"
---

# Reference
""")

        (workflow / "workflow-rules.mdc").write_text("""---
alwaysApply: false
title: "My Workflow"
description: "Test workflow"
metadata:
  author: "Test <test@example.com>"
---

# My Workflow
""")

        result = validator.validate(tmp_path)
        assert any("workflow" in msg.lower() for msg in result.messages)

    def test_validate_folder_with_team_workflows_subdir(self, tmp_path: Path):
        """Test validation of folder containing team-workflows/ subdirectory."""
        validator = WorkflowsSchemaValidator()

        team_workflows_dir = tmp_path / "team-workflows" / "my-team"
        team_workflows_dir.mkdir(parents=True)

        workflow = team_workflows_dir / "my-workflow"
        workflow.mkdir()

        (workflow / "README.md").write_text("# My Workflow")
        refs_dir = workflow / "references"
        refs_dir.mkdir()
        (refs_dir / "ref.mdc").write_text("""---
alwaysApply: false
title: "Reference"
description: "A reference"
---

# Reference
""")

        (workflow / "workflow-rules.mdc").write_text("""---
alwaysApply: false
title: "My Workflow"
description: "Test workflow"
metadata:
  author: "Test <test@example.com>"
---

# My Workflow
""")

        result = validator.validate(tmp_path)
        assert any("team-workflows" in msg.lower() for msg in result.messages)

    def test_validate_folder_not_skill_evaluator_compliant(self, tmp_path: Path):
        """Test validation warns for non-SkillEvaluator compliant folder."""
        validator = WorkflowsSchemaValidator()

        # Create a kebab-case named workflow directory
        workflow_dir = tmp_path / "direct-workflow"
        workflow_dir.mkdir()

        # Create a workflow directly without workflows/ or team-workflows/ parent
        (workflow_dir / "README.md").write_text("# Direct Workflow")
        refs_dir = workflow_dir / "references"
        refs_dir.mkdir()
        (refs_dir / "ref.mdc").write_text("""---
alwaysApply: false
title: "Reference"
description: "A reference"
---

# Reference
""")

        (workflow_dir / "workflow-rules.mdc").write_text("""---
alwaysApply: false
title: "Direct Workflow"
description: "Workflow not in standard folder"
metadata:
  author: "Test <test@example.com>"
---

# Direct Workflow
""")

        result = validator.validate(workflow_dir)
        # Should pass but warn about non-standard structure
        assert result.passed
        # Should have warning about not being in standard location
        assert any("standard" in warn.lower() or "skill_evaluator" in warn.lower() for warn in result.warnings)
