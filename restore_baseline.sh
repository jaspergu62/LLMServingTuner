#!/bin/bash
#
# restore_baseline.sh
#
# This script restores the codebase to the Dec 31, 2024 baseline
# by removing parallel PSO files and reverting modifications.
#
# Usage: ./restore_baseline.sh
#
# IMPORTANT: Run this from the modelevalstate directory!

set -e  # Exit on error

echo "================================================"
echo "Restore Baseline Script"
echo "Restoring codebase to Dec 31, 2024 state"
echo "================================================"
echo ""

# Verify we're in the right directory
if [ ! -f "config.toml" ] || [ ! -d "optimizer" ]; then
    echo "ERROR: This script must be run from the modelevalstate directory!"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Confirm with user
read -p "This will delete parallel PSO files and modify several files. Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Step 1: Deleting new files and directories..."
echo "--------------------------------------------"

# Delete parallel PSO directories and files
if [ -d "optimizer/parallel" ]; then
    rm -rf optimizer/parallel/
    echo "✓ Deleted optimizer/parallel/"
else
    echo "  (optimizer/parallel/ already removed)"
fi

# Delete test files
for file in tests/test_parallel_*.py; do
    if [ -f "$file" ]; then
        rm -f "$file"
        echo "✓ Deleted $file"
    fi
done

if [ -f "tests/__init__.py" ]; then
    rm -f tests/__init__.py
    echo "✓ Deleted tests/__init__.py"
fi

# Delete exceptions.py
if [ -f "exceptions.py" ]; then
    rm -f exceptions.py
    echo "✓ Deleted exceptions.py"
else
    echo "  (exceptions.py already removed)"
fi

# Delete code quality tool configs
if [ -f ".pre-commit-config.yaml" ]; then
    rm -f .pre-commit-config.yaml
    echo "✓ Deleted .pre-commit-config.yaml"
fi

if [ -f "pyproject.toml" ]; then
    rm -f pyproject.toml
    echo "✓ Deleted pyproject.toml"
fi

echo ""
echo "Step 2: Renaming files (restore typos)..."
echo "--------------------------------------------"

# Rename performance_tuner back to performance_tunner
if [ -f "optimizer/performance_tuner.py" ]; then
    mv optimizer/performance_tuner.py optimizer/performance_tunner.py
    echo "✓ Renamed performance_tuner.py → performance_tunner.py"
elif [ -f "optimizer/performance_tunner.py" ]; then
    echo "  (performance_tunner.py already exists)"
else
    echo "ERROR: Neither performance_tuner.py nor performance_tunner.py found!"
    exit 1
fi

echo ""
echo "Step 3: Fixing import paths..."
echo "--------------------------------------------"

# Fix import in optimizer.py
if grep -q "performance_tuner" optimizer/optimizer.py; then
    # macOS compatible sed (use -i '' for in-place edit)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' 's/performance_tuner/performance_tunner/g' optimizer/optimizer.py
    else
        sed -i 's/performance_tuner/performance_tunner/g' optimizer/optimizer.py
    fi
    echo "✓ Fixed import in optimizer/optimizer.py"
else
    echo "  (Import already correct in optimizer/optimizer.py)"
fi

echo ""
echo "Step 4: Restoring typos in config files..."
echo "--------------------------------------------"

# Restore typo in config.toml
if grep -q "max_queue_delay_microseconds" config.toml; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' 's/max_queue_delay_microseconds/max_queue_deloy_mircroseconds/g' config.toml
    else
        sed -i 's/max_queue_delay_microseconds/max_queue_deloy_mircroseconds/g' config.toml
    fi
    echo "✓ Restored typo in config.toml"
else
    echo "  (Typo already present in config.toml)"
fi

# Restore typo in config/config.py
if grep -q "max_queue_delay_microseconds" config/config.py; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' 's/max_queue_delay_microseconds/max_queue_deloy_mircroseconds/g' config/config.py
    else
        sed -i 's/max_queue_delay_microseconds/max_queue_deloy_mircroseconds/g' config/config.py
    fi
    echo "✓ Restored typo in config/config.py"
else
    echo "  (Typo already present in config/config.py)"
fi

# Restore typo in plugin.md
if [ -f "optimizer/plugins/plugin.md" ] && grep -q "Settings is invalid\." optimizer/plugins/plugin.md; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' 's/Settings is invalid\./Settings is invalidator./g' optimizer/plugins/plugin.md
    else
        sed -i 's/Settings is invalid\./Settings is invalidator./g' optimizer/plugins/plugin.md
    fi
    echo "✓ Restored typo in optimizer/plugins/plugin.md"
elif [ -f "optimizer/plugins/plugin.md" ]; then
    echo "  (Typo already present in plugin.md)"
fi

echo ""
echo "================================================"
echo "Automated restoration complete!"
echo "================================================"
echo ""
echo "⚠️  MANUAL STEPS STILL REQUIRED:"
echo ""
echo "1. Edit config/config.py:"
echo "   - Remove exception imports (ConfigPathError, ConfigValidationError)"
echo "   - Remove CONFIG_POSITION_PATTERN constant"
echo "   - Remove VALID_DTYPES class variable from OptimizerConfigField"
echo "   - Remove @field_validator methods (validate_config_position, validate_dtype)"
echo "   - Simplify update_constant() to use ValueError instead of ConfigValidationError"
echo ""
echo "2. Edit __init__.py:"
echo "   - Remove all exception imports (ModelEvalStateError, etc.)"
echo ""
echo "3. Edit optimizer/interfaces/simulator.py:"
echo "   - Simplify health() method logging (remove detailed logger.debug/error/warning)"
echo ""
echo "4. Edit CLAUDE.md:"
echo "   - Remove 'Parallel PSO Evaluation' section"
echo ""
echo "See FILE_MANIFEST.md for detailed instructions on manual edits."
echo ""

# Verification
echo "Verification:"
echo "--------------------------------------------"

# Should NOT exist
if [ ! -d "optimizer/parallel" ]; then
    echo "✓ optimizer/parallel/ does not exist"
else
    echo "✗ FAIL: optimizer/parallel/ still exists"
fi

if [ ! -f "exceptions.py" ]; then
    echo "✓ exceptions.py does not exist"
else
    echo "✗ FAIL: exceptions.py still exists"
fi

# Should exist with typo
if [ -f "optimizer/performance_tunner.py" ]; then
    echo "✓ optimizer/performance_tunner.py exists (with typo)"
else
    echo "✗ FAIL: optimizer/performance_tunner.py does not exist"
fi

# Check import
if grep -q "performance_tunner" optimizer/optimizer.py; then
    echo "✓ Import uses performance_tunner in optimizer.py"
else
    echo "✗ FAIL: Import incorrect in optimizer.py"
fi

# Check typo in config
if grep -q "max_queue_deloy_mircroseconds" config.toml; then
    echo "✓ Typo present in config.toml"
else
    echo "✗ FAIL: Typo not present in config.toml"
fi

echo ""
echo "Next steps:"
echo "1. Complete manual edits (see FILE_MANIFEST.md)"
echo "2. Run verification checks (see GIT_WORKFLOW.md Phase 4)"
echo "3. Initialize git repository and create main branch"
echo ""
