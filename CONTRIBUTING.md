# Contributing to ADE Insight

Thank you for contributing to **ADE Insight**.

This document describes the development workflow, coding conventions, and release process used in this repository.

---

# Development Setup

Clone the repository:

```
git clone <repository-url>
cd ade-insight
```

Create a virtual environment:

```
python -m venv .venv
```

Activate it:

Linux / macOS

```
source .venv/bin/activate
```

Windows (PowerShell)

```
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```
pip install -e ".[dev,gui]"
```

---

# Project Structure

```
src/            Python source code
tests/          Automated tests
build/          Installer and packaging configuration
scripts/        Helper scripts for development and builds
docs/           Documentation
CHANGELOG.md    Release history
```

The project uses a **src-layout** Python package structure.

---

# Branch Workflow

The repository follows a simple branching strategy.

### Main Branch

`main` always contains **stable code** that can be released.

### Feature Branches

All development should occur in feature branches.

Examples:

```
feat/add-energy-summary
fix/gui-crash-on-load
chore/refactor-imports
```

Create a branch:

```
git checkout main
git pull
git checkout -b feat/my-feature
```

---

# Commit Guidelines

Use clear, descriptive commit messages.

Recommended style:

```
type: short description
```

Examples:

```
feat: add energy summary table
fix: correct timestamp parsing bug
chore: refactor plotting module
docs: update installation guide
```

Common commit types:

| Type  | Purpose                 |
| ----- | ----------------------- |
| feat  | New feature             |
| fix   | Bug fix                 |
| docs  | Documentation           |
| chore | Maintenance/refactoring |
| test  | Tests                   |
| build | Build/installer changes |

---

# Code Style

The project uses **Ruff** for linting.

Run:

```
ruff check .
```

Formatting guidelines:

* Line length: 100 characters
* Follow standard Python naming conventions
* Use descriptive variable names

---

# Running Tests

Run tests with:

```
pytest
```

Test data is located in:

```
tests/data
```

---

# Packaging

The project is packaged using **setuptools** with `pyproject.toml`.

Build distributions with:

```
python -m build
```

---

# Building the Application

## Build EXE

```
pyinstaller build/pyinstaller/adeinsight.spec
```

## Build MSI Installer

```
build/installer/wix3/build_msi.ps1
```

---

# Release Process

Releases follow **Semantic Versioning**.

```
MAJOR.MINOR.PATCH
```

Examples:

```
1.0.0
1.1.0
1.1.1
```

Meaning:

| Version | Meaning |
| ------- | ------- |
| MAJOR   |         |

