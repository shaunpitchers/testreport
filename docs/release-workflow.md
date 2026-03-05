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
