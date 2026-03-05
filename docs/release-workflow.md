# Release Workflow (ADE Insight)

This document describes the development and release workflow used for **ADE Insight**.

It follows a simple Git workflow with **semantic versioning** and **tagged releases**.

---

# Branch Strategy

## Main Branch

`main` should always represent **stable, releasable code**.

All development is merged into `main` via feature branches.

---

## Feature / Fix Branches

Create a branch for all development work.

Examples:

```
feat/add-energy-summary
fix/gui-plot-crash
chore/refactor-imports
```

Create a branch from main:

```bash
git checkout main
git pull
git checkout -b feat/my-feature
```

Work normally and commit changes.

Example commit:

```bash
git add -A
git commit -m "Add energy summary output table"
```

---

## Merge Back to Main

When the feature is complete:

```bash
git checkout main
git pull
git merge feat/my-feature
git push
```

Then delete the branch:

```bash
git branch -d feat/my-feature
```

---

# Versioning

ADE Insight uses **Semantic Versioning**.

Format:

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

| Version Part | Meaning          |
| ------------ | ---------------- |
| MAJOR        | Breaking changes |
| MINOR        | New features     |
| PATCH        | Bug fixes        |

---

# Pre-1.0 Development

Before the first stable release the project may use versions such as:

```
0.2.0
0.3.0
0.3.1
```

Example workflow:

```
0.3.0  First internal tester release
0.3.1  Bug fixes
0.3.2  Bug fixes
```

When the software is considered stable and ready for production use:

```
1.0.0
```

---

# Release Process

When preparing a release:

## 1. Update Version

Edit:

```
pyproject.toml
```

Example:

```
version = "0.3.0"
```

---

## 2. Update CHANGELOG

Add a section describing the release.

Example:

```
## [0.3.0] - 2026-03-05
### Added
- GUI energy summary table
- Export CSV improvements

### Fixed
- Incorrect timestamp parsing
```

---

## 3. Commit Release Changes

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "Release v0.3.0"
git push
```

---

## 4. Create Git Tag

Tags permanently mark releases.

```bash
git tag v0.3.0
git push origin v0.3.0
```

---

## 5. Build Release Artifacts

Always build from the tag to ensure a clean build.

```bash
git checkout v0.3.0
```

Build:

* Windows EXE (PyInstaller)
* MSI installer (WiX)

Example outputs:

```
ADE-Insight_v0.3.0_setup.msi
ADE-Insight_v0.3.0_setup.msi.sha256
ADE-Insight_v0.3.0_windows_x64.zip
ADE-Insight_v0.3.0_windows_x64.zip.sha256
```

---

## 6. Publish Release

Upload artifacts internally with:

* installer / zip
* checksum file
* release notes

---

## 7. Return to Development

```bash
git checkout main
```

Development then continues for the next version.

---

# Hotfix Releases

If a bug is found in a release:

```
1.0.0 → 1.0.1
```

Steps:

```
git checkout main
fix bug
update version
tag release
```

---

# Example Timeline

Example project history:

```
0.3.0  First tester build
0.3.1  Bug fixes
0.3.2  Bug fixes

1.0.0  First stable production release

1.1.0  New feature release
1.1.1  Bug fix release
```

---

# Artifact Naming Convention

Use consistent names for releases:

```
ADE-Insight_v1.0.0_setup.msi
ADE-Insight_v1.0.0_windows_x64.zip
ADE-Insight_v1.0.0_windows_x64.zip.sha256
```

---

# Notes

* Always build releases from **Git tags**
* Never release from uncommitted code
* Always update the **CHANGELOG**

This keeps releases reproducible and traceable.

# Git Safety Practices

These safeguards help prevent accidental releases, broken builds, or pushing unfinished code.

---

# Pre-Push Protection for `main`

Direct pushes to `main` should be avoided. All work should normally go through feature branches.

A Git **pre-push hook** can prevent accidental pushes.

Create the hook:

```bash
nvim .git/hooks/pre-push
```

Add the following script:

```bash
#!/bin/sh

branch=$(git rev-parse --abbrev-ref HEAD)

if [ "$branch" = "main" ]; then
  echo ""
  echo "⚠️  Direct push to 'main' detected."
  echo "Use a feature branch and merge instead."
  echo ""
  echo "If this push is intentional, run:"
  echo "    git push origin main --no-verify"
  echo ""
  exit 1
fi
```

Make it executable:

```bash
chmod +x .git/hooks/pre-push
```

This prevents accidental pushes to `main`.

If a direct push is truly required:

```bash
git push origin main --no-verify
```

---

# Optional Pre-Commit Checks

Additional checks can be added before committing.

Create:

```bash
nvim .git/hooks/pre-commit
```

Example basic check:

```bash
#!/bin/sh

echo "Running basic repository checks..."
```

This hook can later be expanded to run:

* linting (`ruff`)
* formatting checks
* test runs
* version validation

---

# Release Safety Checklist

Before creating any release:

```bash
git status
git log --oneline -5
```

Verify:

* Working tree is clean
* Recent commits look correct
* Version has been updated
* Changelog has been updated

---

# Always Build Releases From Tags

Releases should always be built from a **Git tag**.

Example:

```bash
git checkout v1.0.0
```

Never build a release directly from `main`.

Building from a tag guarantees:

* reproducible builds
* traceable release history
* exact match between source and binary

---

# Typical Release Workflow

1. Ensure repository is clean

```bash
git status
```

2. Update version in `pyproject.toml`

3. Update `CHANGELOG.md`

4. Commit release changes

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "Release vX.Y.Z"
git push
```

5. Create tag

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

6. Build release from the tag

```bash
git checkout vX.Y.Z
```

7. Build binaries

* PyInstaller executable
* WiX MSI installer

8. Generate SHA256 checksums

9. Publish artifacts internally

---

# Why This Matters

These practices ensure:

* releases are reproducible
* binaries match the source code exactly
* mistakes can be traced and fixed easily

This workflow greatly reduces the risk of broken or inconsistent releases.
