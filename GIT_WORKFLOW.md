# Git Workflow: Creating Main and Parallel_PSO Branches

## Overview

This guide walks through creating a GitHub repository with two branches:
- **main**: Baseline code (as of Dec 31, 2024, before parallel PSO)
- **parallel_pso**: Full code including parallel PSO feature

## Prerequisites

- GitHub account
- GitHub repository URL (you'll provide this)
- Git installed locally
- SSH key configured for GitHub (or use HTTPS)

---

## Step-by-Step Guide

### Phase 1: Backup Current Code (Parallel PSO Version)

First, save the current complete codebase to a temporary location.

```bash
# Navigate to parent directory
cd /Users/gujiazhen/Documents/projects/SimInfer/msit-master-msserviceprofiler-msserviceprofiler-modelevalstate/msserviceprofiler/msserviceprofiler/

# Create backup of current state (parallel PSO version)
cp -r modelevalstate modelevalstate_parallel_pso_backup

# Verify backup
ls -la modelevalstate_parallel_pso_backup/
```

**Checkpoint**: You should now have a complete backup.

---

### Phase 2: Restore Baseline (Dec 31, 2024 Version)

Now restore the working directory to the pre-parallel-PSO state.

```bash
# Navigate to the working directory
cd modelevalstate

# 1. Delete new files/directories
rm -rf optimizer/parallel/
rm -f tests/test_parallel_*.py
rm -f tests/__init__.py
rm -f exceptions.py
rm -f .pre-commit-config.yaml
rm -f pyproject.toml

# 2. Rename performance_tuner back to performance_tunner (restore typo)
mv optimizer/performance_tuner.py optimizer/performance_tunner.py

# 3. Verify deletions
echo "Checking deleted items..."
test ! -d optimizer/parallel && echo "✓ optimizer/parallel deleted" || echo "✗ optimizer/parallel still exists"
test ! -f exceptions.py && echo "✓ exceptions.py deleted" || echo "✗ exceptions.py still exists"
test -f optimizer/performance_tunner.py && echo "✓ performance_tunner.py exists" || echo "✗ performance_tunner.py missing"
```

**Next**: Manually edit modified files (see below).

---

### Phase 3: Manual File Edits for Baseline

Edit the following files to remove parallel PSO changes:

#### 3.1. `optimizer/optimizer.py`

**Find** (around line 15):
```python
from msserviceprofiler.modelevalstate.optimizer.performance_tuner import PerformanceTuner
```

**Replace with**:
```python
from msserviceprofiler.modelevalstate.optimizer.performance_tunner import PerformanceTuner
```

#### 3.2. `__init__.py`

**Remove** the entire exception hierarchy import block (typically at the end):
```python
# Delete this entire section:
from msserviceprofiler.modelevalstate.exceptions import (
    ModelEvalStateError,
    ConfigurationError,
    ConfigPathError,
    # ... all exception imports
    IPCCommandError,
)
```

#### 3.3. `config/config.py`

This file has many changes. Create a detailed edit:

**Remove imports** (top of file):
```python
# DELETE these lines:
from msserviceprofiler.modelevalstate.exceptions import ConfigPathError, ConfigValidationError
import re

# Also delete this constant:
CONFIG_POSITION_PATTERN = ...  # (around line 20)
```

**In `OptimizerConfigField` class**:
- **Delete** the `VALID_DTYPES` class variable (around line 74)
- **Delete** the `validate_config_position` method (lines ~76-92)
- **Delete** the `validate_dtype` method (lines ~94-104)
- **Simplify** `update_constant` method (lines ~106-120):

  Find:
  ```python
  @model_validator(mode="after")
  def update_constant(self):
      if self.min > self.max:
          raise ConfigValidationError(
              "min/max",
              f"min={self.min}, max={self.max}",
              f"min ({self.min}) cannot be greater than max ({self.max})"
          )
  ```

  Replace with:
  ```python
  @model_validator(mode="after")
  def update_constant(self):
      if self.min > self.max:
          raise ValueError(f"min ({self.min}) cannot be greater than max ({self.max})")
  ```

**Fix typo in default config** (around line 230):
```python
# Find:
OptimizerConfigField(name="max_queue_delay_microseconds", ...)

# Change to (restore typo):
OptimizerConfigField(name="max_queue_deloy_mircroseconds", ...)
```

#### 3.4. `config.toml`

**Fix typo** (around line 50):
```toml
# Find:
name = "max_queue_delay_microseconds"

# Change to (restore typo):
name = "max_queue_deloy_mircroseconds"
```

#### 3.5. `optimizer/interfaces/simulator.py`

**Remove enhanced logging in `health()` method** (around lines 60-80):

Find the health method with detailed logging like:
```python
logger.debug(f"Checking simulator health at {self.base_url}")
logger.error(error_msg)
logger.warning(f"Simulator process health check failed: {process_res.info}")
```

Replace with simpler version (if you have it), or just remove the excessive logging calls.

#### 3.6. `CLAUDE.md`

**Remove** the entire "Parallel PSO Evaluation" section (search for "## Parallel PSO" and delete that section).

#### 3.7. `optimizer/plugins/plugin.md`

**Find**:
```python
raise ValueError("Settings is invalid.")
```

**Change to** (restore typo):
```python
raise ValueError("Settings is invalidator.")
```

---

### Phase 4: Verify Baseline

```bash
# Check all critical changes
echo "=== Verification ==="

# Should NOT exist:
test ! -d optimizer/parallel && echo "✓ No parallel directory" || echo "✗ FAIL: parallel directory exists"
test ! -f exceptions.py && echo "✓ No exceptions.py" || echo "✗ FAIL: exceptions.py exists"

# Should exist with typo:
test -f optimizer/performance_tunner.py && echo "✓ performance_tunner.py exists (with typo)" || echo "✗ FAIL: performance_tunner.py missing"

# Check import
grep -q "performance_tunner" optimizer/optimizer.py && echo "✓ Import uses performance_tunner" || echo "✗ FAIL: Import wrong"

# Check typo in config
grep -q "max_queue_deloy_mircroseconds" config.toml && echo "✓ Typo present in config.toml" || echo "✗ FAIL: Typo fixed"

echo "=== Done ==="
```

---

### Phase 5: Initialize Git and Create Main Branch

```bash
# Initialize git repository
git init

# Configure git (if needed)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.coverage
*.egg-info/
dist/
build/
.DS_Store
*.log
EOF

# Stage all files
git add .

# Create initial commit (baseline)
git commit -m "chore: initial commit - baseline before parallel PSO

This is the codebase state as of December 31, 2024.
Includes all core optimizer functionality without parallel evaluation."

# Rename branch to main
git branch -M main
```

---

### Phase 6: Add Remote and Push Main Branch

**You will provide your GitHub repository URL here.**

```bash
# Add remote (replace URL with your repo)
git remote add origin <YOUR_GITHUB_REPO_URL>

# Example:
# git remote add origin git@github.com:yourusername/modelevalstate.git
# OR
# git remote add origin https://github.com/yourusername/modelevalstate.git

# Push main branch
git push -u origin main
```

**Checkpoint**: Main branch is now on GitHub with baseline code.

---

### Phase 7: Create Parallel_PSO Branch

```bash
# Create new branch
git checkout -b parallel_pso

# Copy back the parallel PSO version from backup
# First, remove current content (keeping .git)
find . -mindepth 1 -maxdepth 1 ! -name '.git' ! -name '.gitignore' -exec rm -rf {} +

# Copy everything from backup
cp -r ../modelevalstate_parallel_pso_backup/* .

# Stage all changes
git add .

# Commit parallel PSO changes
git commit -m "feat: add parallel PSO evaluation with multi-node support

Major Features:
- Parallel particle evaluation across distributed service groups
- Multi-node service management via SSH
- Health monitoring and fault tolerance (circuit breaker, retry policies)
- Custom exception hierarchy for better error handling
- Config validation with Pydantic validators
- Code quality improvements (pre-commit hooks, fixed typos)

Implementation:
- optimizer/parallel/: Core parallel evaluation modules (~2,839 lines)
  - config.py: Node and service group configuration
  - remote.py: SSH-based remote execution
  - service_group.py: Multi-node service lifecycle management
  - pool.py: Service group pool management
  - dispatcher.py: Particle dispatch and result collection
  - monitoring.py: Health monitoring, circuit breaker, retry policies

- exceptions.py: Custom exception hierarchy (17 exception classes)

- tests/: Comprehensive test suite (~2,397 lines, 155 tests, 77% coverage)
  - test_parallel_config.py (27 tests)
  - test_parallel_dispatcher.py (15 tests)
  - test_parallel_monitoring.py (38 tests)
  - test_parallel_remote.py (32 tests)
  - test_parallel_service_group.py (43 tests)

Code Quality:
- Renamed performance_tunner → performance_tuner (fixed typo)
- Fixed config parameter: max_queue_deloy_mircroseconds → max_queue_delay_microseconds
- Added pre-commit hooks (black, isort, flake8)
- Added pyproject.toml for tool configuration

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Phase 8: Push Parallel_PSO Branch

```bash
# Push parallel_pso branch
git push -u origin parallel_pso
```

---

### Phase 9: Create Pull Request (Optional)

If you want to create a PR from `parallel_pso` → `main`:

1. Go to GitHub repository web interface
2. Click "Pull requests" → "New pull request"
3. Set base: `main`, compare: `parallel_pso`
4. Fill in PR description (you can use `PULL_REQUEST.md` template)
5. Create pull request

---

## Verification

After completing all steps:

```bash
# Check branches
git branch -a

# Should show:
# * parallel_pso
#   main
#   remotes/origin/main
#   remotes/origin/parallel_pso

# Check remote
git remote -v

# Check current branch status
git log --oneline --graph --all --decorate -n 10
```

---

## Quick Reference Commands

### View differences between branches
```bash
# See files changed between main and parallel_pso
git diff main..parallel_pso --name-status

# See detailed diff
git diff main..parallel_pso
```

### Switch between branches
```bash
# Switch to main
git checkout main

# Switch to parallel_pso
git checkout parallel_pso
```

### Update after changes
```bash
# If you make more changes on parallel_pso
git add .
git commit -m "fix: your commit message"
git push origin parallel_pso
```

---

## Troubleshooting

### Issue: Git push fails with authentication error

**Solution**: Set up SSH key or use personal access token for HTTPS.

### Issue: Merge conflicts when creating PR

**Solution**: This shouldn't happen if you followed the steps correctly, as main and parallel_pso have completely different states.

### Issue: File permissions errors

**Solution**: Check file permissions and ownership:
```bash
chmod -R u+rw .
```

---

## Next Steps

1. ✅ Backup current code
2. ✅ Restore baseline
3. ✅ Edit modified files
4. ✅ Verify baseline
5. ✅ Init git and commit main
6. ✅ Push main to GitHub
7. ✅ Create parallel_pso branch
8. ✅ Commit parallel PSO changes
9. ✅ Push parallel_pso to GitHub
10. 📋 (Optional) Create Pull Request
11. 📋 (Optional) Add README.md, documentation, CI/CD

---

## Summary

After completion, you will have:
- **GitHub Repository** with two branches
- **main branch**: Clean baseline (Dec 31, 2024)
- **parallel_pso branch**: Full implementation with all improvements
- **Pull Request (optional)**: For code review and discussion
- **Complete history**: All changes documented and tracked
