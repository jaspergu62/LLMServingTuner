# GitHub Repository Setup Guide

## Overview

This guide helps you create a GitHub repository for the ModelEvalState project with two branches:
- **main**: Baseline code (Dec 31, 2024 - before parallel PSO)
- **parallel_pso**: Full implementation with parallel PSO feature

---

## Quick Start

Follow these steps in order:

### 1. Read Documentation (5 minutes)

- **FILE_MANIFEST.md** - Understand what changed
- **GIT_WORKFLOW.md** - Complete step-by-step guide
- **PULL_REQUEST_TEMPLATE.md** - PR description template

### 2. Backup Current Code (1 minute)

```bash
cd /Users/gujiazhen/Documents/projects/SimInfer/msit-master-msserviceprofiler-msserviceprofiler-modelevalstate/msserviceprofiler/msserviceprofiler/
cp -r modelevalstate modelevalstate_parallel_pso_backup
```

### 3. Run Automated Restoration (2 minutes)

```bash
cd modelevalstate
./restore_baseline.sh
```

This automatically:
- Deletes parallel PSO files
- Renames files to restore typos
- Fixes import paths
- Restores config typos

### 4. Manual Edits (10 minutes)

Complete the manual edits listed in `FILE_MANIFEST.md`:
- Edit `config/config.py` (remove validators)
- Edit `__init__.py` (remove exception imports)
- Edit `optimizer/interfaces/simulator.py` (simplify logging)
- Edit `CLAUDE.md` (remove parallel PSO section)

See `FILE_MANIFEST.md` for detailed instructions.

### 5. Create GitHub Repository

1. Go to https://github.com/new
2. Create a new repository (e.g., `modelevalstate`)
3. Copy the repository URL (SSH or HTTPS)

### 6. Initialize Git and Push (5 minutes)

```bash
# Initialize and create main branch
git init
git add .
git commit -m "chore: initial commit - baseline before parallel PSO"
git branch -M main

# Add remote (replace with your URL)
git remote add origin <YOUR_GITHUB_REPO_URL>

# Push main branch
git push -u origin main
```

### 7. Create parallel_pso Branch (2 minutes)

```bash
# Create branch
git checkout -b parallel_pso

# Restore parallel PSO code
find . -mindepth 1 -maxdepth 1 ! -name '.git' ! -name '.gitignore' -exec rm -rf {} +
cp -r ../modelevalstate_parallel_pso_backup/* .

# Commit and push
git add .
git commit -m "feat: add parallel PSO evaluation with multi-node support

[Use content from PULL_REQUEST_TEMPLATE.md as commit message]"

git push -u origin parallel_pso
```

### 8. Create Pull Request (Optional, 2 minutes)

1. Go to your GitHub repository
2. Click "Pull requests" → "New pull request"
3. Base: `main`, Compare: `parallel_pso`
4. Copy content from `PULL_REQUEST_TEMPLATE.md` as PR description
5. Create pull request

---

## Document Reference

### FILE_MANIFEST.md
**What**: Complete list of all file changes
**When to read**: Before starting restoration
**Key info**:
- 14 new files to delete
- 7 modified files to edit
- 1 renamed file
- Detailed edit instructions

### GIT_WORKFLOW.md
**What**: Comprehensive step-by-step workflow
**When to read**: Primary guide to follow
**Key info**:
- 9 phases from backup to PR creation
- Verification commands
- Troubleshooting tips

### restore_baseline.sh
**What**: Automated restoration script
**When to run**: After backing up current code
**What it does**:
- Deletes new files/directories
- Renames files
- Fixes import paths
- Restores typos in config
**What it doesn't do**:
- Manual file edits (config.py, __init__.py, etc.)

### PULL_REQUEST_TEMPLATE.md
**What**: PR description template
**When to use**: Creating PR on GitHub
**Key info**:
- Feature summary
- Implementation details
- Test coverage
- Usage examples

---

## File Change Summary

### New Files (Parallel PSO Feature)

```
✨ Production Code (~2,839 lines):
   optimizer/parallel/__init__.py
   optimizer/parallel/config.py
   optimizer/parallel/dispatcher.py
   optimizer/parallel/monitoring.py
   optimizer/parallel/pool.py
   optimizer/parallel/remote.py
   optimizer/parallel/service_group.py

✨ Tests (~2,397 lines, 155 tests, 77% coverage):
   tests/test_parallel_config.py
   tests/test_parallel_dispatcher.py
   tests/test_parallel_monitoring.py
   tests/test_parallel_remote.py
   tests/test_parallel_service_group.py
   tests/__init__.py

✨ Exception Hierarchy (207 lines):
   exceptions.py

✨ Code Quality Tools:
   .pre-commit-config.yaml
   pyproject.toml
```

### Modified Files (Code Quality Improvements)

```
🔧 optimizer/optimizer.py          - Import path fix
🔧 config/config.py                - Validators + typo fix
🔧 config.toml                     - Typo fix
🔧 __init__.py                     - Exception exports
🔧 optimizer/interfaces/simulator.py - Enhanced logging
🔧 CLAUDE.md                       - Parallel PSO docs
🔧 optimizer/plugins/plugin.md     - Typo fix
```

### Renamed Files

```
📝 optimizer/performance_tunner.py → performance_tuner.py
```

---

## Verification Checklist

### After Automated Restoration

- [ ] `optimizer/parallel/` deleted
- [ ] `tests/test_parallel_*.py` deleted
- [ ] `exceptions.py` deleted
- [ ] `optimizer/performance_tunner.py` exists (with typo)
- [ ] Import in `optimizer/optimizer.py` uses `performance_tunner`
- [ ] Typo `max_queue_deloy_mircroseconds` in config.toml
- [ ] Typo `max_queue_deloy_mircroseconds` in config/config.py

### After Manual Edits

- [ ] No `@field_validator` for config_position in config/config.py
- [ ] No `@field_validator` for dtype in config/config.py
- [ ] No exception imports in config/config.py
- [ ] No exception imports in __init__.py
- [ ] Simplified logging in optimizer/interfaces/simulator.py
- [ ] No parallel PSO section in CLAUDE.md

### After Git Setup

- [ ] Git initialized
- [ ] main branch created and pushed
- [ ] parallel_pso branch created and pushed
- [ ] Both branches visible on GitHub
- [ ] (Optional) Pull request created

---

## Time Estimate

| Task | Time |
|------|------|
| Read documentation | 5 min |
| Backup code | 1 min |
| Run automated script | 2 min |
| Manual edits | 10 min |
| Create GitHub repo | 2 min |
| Git init and push main | 5 min |
| Create parallel_pso branch | 2 min |
| Create PR (optional) | 2 min |
| **Total** | **~30 minutes** |

---

## Troubleshooting

### Script fails with "Permission denied"

```bash
chmod +x restore_baseline.sh
./restore_baseline.sh
```

### Git push fails

**Authentication error**: Set up SSH key or use personal access token

```bash
# Check remote URL
git remote -v

# Change to SSH (if you have SSH key)
git remote set-url origin git@github.com:username/repo.git

# Or use HTTPS with token
git remote set-url origin https://github.com/username/repo.git
```

### Files still exist after script

The script only deletes what it can safely delete. Some files may need manual deletion:
```bash
rm -f <file_that_should_not_exist>
```

### Import errors after restoration

Make sure you renamed and fixed imports:
```bash
# Check if file exists
ls -la optimizer/performance_tunner.py

# Check import
grep "performance_tunner" optimizer/optimizer.py
```

---

## Next Steps After Setup

### Update GitHub Repository Settings

1. Add description: "LLM inference service optimization framework with PSO"
2. Add topics: `python`, `optimization`, `llm`, `inference`, `pso`
3. Configure branch protection for `main`
4. Enable GitHub Actions (optional, for CI/CD)

### Add README.md to Repository

Create a README.md for your repository:
- Project overview
- Installation instructions
- Usage examples
- Documentation links

### Set Up CI/CD (Optional)

Create `.github/workflows/test.yml`:
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/test_parallel_*.py --cov
```

---

## Support

If you encounter issues not covered in troubleshooting:

1. Check `FILE_MANIFEST.md` for detailed file changes
2. Review `GIT_WORKFLOW.md` for complete workflow
3. Verify using the verification checklist above
4. Review git status: `git status`
5. Check git log: `git log --oneline --graph --all -n 10`

---

## Summary

This setup process creates a clean separation between your baseline code and the parallel PSO feature, making it easy to:
- Review changes via pull request
- Test the parallel PSO feature independently
- Merge to main when ready
- Maintain a clear project history

**Estimated total time**: ~30 minutes

**Result**: Professional GitHub repository with documented feature PR ready for review.
