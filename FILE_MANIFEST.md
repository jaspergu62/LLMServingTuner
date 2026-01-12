# File Manifest: Parallel PSO Changes

## Overview

This document lists all files that were created or modified as part of the Parallel PSO feature implementation (Jan 7-9, 2026).

## Summary

- **New Files**: 14 files (~5,236 lines)
- **Modified Files**: 7 files
- **Renamed Files**: 1 file
- **Documentation**: 7 HTML files + 1 plan

---

## New Files to Delete (for baseline)

### Core Implementation (optimizer/parallel/)

All files in `optimizer/parallel/` directory should be deleted:

```
optimizer/parallel/__init__.py          (91 lines)
optimizer/parallel/config.py            (338 lines)
optimizer/parallel/dispatcher.py        (335 lines)
optimizer/parallel/monitoring.py        (551 lines)
optimizer/parallel/pool.py              (371 lines)
optimizer/parallel/remote.py            (564 lines)
optimizer/parallel/service_group.py     (589 lines)
```

**Action**: `rm -rf optimizer/parallel/`

### Test Files (tests/)

All parallel test files should be deleted:

```
tests/test_parallel_config.py           (328 lines)
tests/test_parallel_dispatcher.py       (357 lines)
tests/test_parallel_monitoring.py       (682 lines)
tests/test_parallel_remote.py           (595 lines)
tests/test_parallel_service_group.py    (435 lines)
tests/__init__.py                       (664 bytes - created Jan 8)
```

**Action**:
```bash
rm tests/test_parallel_*.py
rm tests/__init__.py
```

### Exception Handling

```
exceptions.py                           (207 lines - entire file)
```

**Action**: `rm exceptions.py`

### Code Quality Tools

```
.pre-commit-config.yaml
pyproject.toml
```

**Action**:
```bash
rm .pre-commit-config.yaml
rm pyproject.toml
```

---

## Modified Files to Restore

### 1. `optimizer/optimizer.py`

**Changes**: Import path updated for performance_tuner

**Modified Line ~15**:
```python
# CURRENT (Jan 7):
from msserviceprofiler.modelevalstate.optimizer.performance_tuner import PerformanceTuner

# SHOULD BE (Dec 31):
from msserviceprofiler.modelevalstate.optimizer.performance_tunner import PerformanceTuner
```

**Action**: Change import path back to `performance_tunner`

### 2. `optimizer/performance_tuner.py` → `performance_tunner.py`

**Changes**: File was renamed (typo fix)

**Action**:
```bash
mv optimizer/performance_tuner.py optimizer/performance_tunner.py
```

### 3. `config/config.py`

**Changes**:
1. Added field validators (@field_validator, @model_validator)
2. Added VALID_DTYPES constant
3. Added exception imports
4. Fixed typo in parameter name

**Lines with validators** (from grep output):
- Lines 76-92: `validate_config_position`
- Lines 94-104: `validate_dtype`
- Lines 106-120: `update_constant` (modified for better validation)
- Plus other validators at lines 381, 399, 613, 619, 643, 656

**Specific Changes to Revert**:

**Line ~3-4**: Remove exception imports
```python
# REMOVE THESE:
from msserviceprofiler.modelevalstate.exceptions import ConfigPathError, ConfigValidationError
import re
```

**Line ~74**: Remove VALID_DTYPES constant
```python
# REMOVE THIS:
VALID_DTYPES: ClassVar[Set[str]] = {"int", "float", "bool", "enum", "ratio", "range", "factories", "env"}
```

**Lines 76-104**: Remove @field_validator decorators
- Remove `validate_config_position` method
- Remove `validate_dtype` method

**Line ~106-120**: Restore simple `update_constant` method
```python
# RESTORE TO SIMPLE VERSION (without ConfigValidationError):
@model_validator(mode="after")
def update_constant(self):
    if self.min > self.max:
        raise ValueError(f"min ({self.min}) cannot be greater than max ({self.max})")
    # ... rest unchanged
```

**Line ~230** (in default config): Fix typo in parameter name
```python
# CURRENT (Jan 7):
OptimizerConfigField(name="max_queue_delay_microseconds", ...)

# SHOULD BE (Dec 31):
OptimizerConfigField(name="max_queue_deloy_mircroseconds", ...)
```

### 4. `config.toml`

**Changes**: Fixed typo in parameter name

**Line ~50**:
```toml
# CURRENT (Jan 7):
name = "max_queue_delay_microseconds"

# SHOULD BE (Dec 31):
name = "max_queue_deloy_mircroseconds"
```

### 5. `__init__.py` (root)

**Changes**: Added exception exports

**Remove these lines** (likely at end of file):
```python
# Exception hierarchy
from msserviceprofiler.modelevalstate.exceptions import (
    ModelEvalStateError,
    ConfigurationError,
    ConfigPathError,
    ConfigValidationError,
    OptimizationError,
    PSOConvergenceError,
    FitnessCalculationError,
    SimulatorError,
    SimulatorStartError,
    SimulatorStopError,
    SimulatorHealthCheckError,
    BenchmarkError,
    BenchmarkExecutionError,
    BenchmarkParseError,
    CommunicationError,
    IPCTimeoutError,
    IPCCommandError,
)
```

### 6. `optimizer/interfaces/simulator.py`

**Changes**: Added enhanced logging in health() method

**Find and restore simpler version** (around lines 60-80):
- Remove detailed logger.debug(), logger.warning(), logger.error() calls
- Restore simpler error handling without detailed logging

### 7. `CLAUDE.md`

**Changes**: Added parallel PSO documentation

**Find and remove** the entire "Parallel PSO Evaluation" section (likely lines ~140-200+)

### 8. `optimizer/plugins/plugin.md`

**Changes**: Fixed typo in error message

**Find**:
```python
raise ValueError("Settings is invalid.")
```

**Change back to**:
```python
raise ValueError("Settings is invalidator.")
```

---

## Documentation Files (Optional - Keep or Delete)

These are HTML reports and can be kept for reference or deleted:

```
docs/parallel-pso-test-report.html
docs/parallel-pso-code-review.html
docs/parallel-pso-guide.html
docs/improvements-2-4-7-changelog.html
docs/plans/2026-01-07-parallel-pso-design.md
docs/coverage_report/ (directory)
```

**Recommendation**: Keep in docs/ for historical reference, but don't commit to main branch.

---

## Verification Checklist

After restoring baseline:

- [ ] `optimizer/parallel/` directory does not exist
- [ ] `tests/test_parallel_*.py` files do not exist
- [ ] `exceptions.py` does not exist
- [ ] `.pre-commit-config.yaml` does not exist
- [ ] `pyproject.toml` does not exist
- [ ] `optimizer/performance_tunner.py` exists (note: double 'n')
- [ ] `optimizer/performance_tuner.py` does NOT exist
- [ ] Import in `optimizer/optimizer.py` references `performance_tunner`
- [ ] No @field_validator in `config/config.py` (except simple pre-existing ones)
- [ ] Typo `max_queue_deloy_mircroseconds` present in config.toml and config.py
- [ ] No exception imports in `__init__.py`

---

## Quick Summary for Git Workflow

**Baseline (main branch)**:
- All files as of Dec 31, 2024
- Contains the typos (these are "expected" in baseline)
- No parallel PSO code
- No exception hierarchy
- No validators in config

**Parallel PSO (parallel_pso branch)**:
- Current state with all improvements
- Includes parallel PSO modules
- Includes exception hierarchy
- Includes config validators
- Typos fixed
- performance_tuner renamed
