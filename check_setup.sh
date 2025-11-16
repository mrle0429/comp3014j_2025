#!/bin/bash
################################################################################
# Environment Check Script for Part C
# Verifies all dependencies and files are ready
################################################################################

echo "=========================================="
echo "Part C Environment Verification"
echo "=========================================="
echo ""

ERROR_COUNT=0

# Check NS2
echo "Checking NS2..."
if command -v ns &> /dev/null; then
    NS_VERSION=$(ns --version 2>&1 | head -1 || echo "unknown")
    echo "  ✓ NS2 installed: $(which ns)"
    echo "    Version: $NS_VERSION"
else
    echo "  ✗ NS2 not found"
    echo "    Install: sudo apt-get install ns2"
    ((ERROR_COUNT++))
fi
echo ""

# Check Python
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "  ✓ Python3 installed: $PYTHON_VERSION"
    
    # Check version >= 3.6
    PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
    PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
    
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 6 ]; then
        echo "    Version OK (>= 3.6)"
    else
        echo "    ✗ Version too old (need >= 3.6)"
        ((ERROR_COUNT++))
    fi
else
    echo "  ✗ Python3 not found"
    echo "    Install: sudo apt-get install python3 python3-pip"
    ((ERROR_COUNT++))
fi
echo ""

# Check Python packages
echo "Checking Python packages..."
PACKAGES=("numpy" "scipy" "pandas" "matplotlib")
for pkg in "${PACKAGES[@]}"; do
    if python3 -c "import $pkg" 2>/dev/null; then
        VERSION=$(python3 -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "unknown")
        echo "  ✓ $pkg ($VERSION)"
    else
        echo "  ✗ $pkg missing"
        echo "    Install: pip3 install $pkg"
        ((ERROR_COUNT++))
    fi
done
echo ""

# Check required files
echo "Checking required files..."
FILES=("yeahCode_partC.tcl" "analyser_partC.py" "run_partC.sh")
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file exists"
        
        # Check if executable (for .sh files)
        if [[ "$file" == *.sh ]]; then
            if [ -x "$file" ]; then
                echo "    ✓ Executable"
            else
                echo "    ✗ Not executable (run: chmod +x $file)"
                ((ERROR_COUNT++))
            fi
        fi
        
        # Check Python syntax
        if [[ "$file" == *.py ]]; then
            if python3 -m py_compile "$file" 2>/dev/null; then
                echo "    ✓ Syntax OK"
            else
                echo "    ✗ Syntax errors found"
                ((ERROR_COUNT++))
            fi
        fi
    else
        echo "  ✗ $file missing"
        ((ERROR_COUNT++))
    fi
done
echo ""

# Check plots directory
echo "Checking directories..."
if [ -d "plots" ]; then
    echo "  ✓ plots/ directory exists"
else
    echo "  ! plots/ directory missing (will be created)"
    mkdir -p plots
    echo "    ✓ Created plots/ directory"
fi
echo ""

# Check disk space
echo "Checking disk space..."
AVAILABLE=$(df -BM . | tail -1 | awk '{print $4}' | sed 's/M//')
if [ "$AVAILABLE" -gt 100 ]; then
    echo "  ✓ Sufficient disk space (${AVAILABLE}MB available)"
else
    echo "  ! Low disk space (${AVAILABLE}MB available)"
    echo "    Warning: Need at least 100MB for trace files"
fi
echo ""

# System information
echo "System Information:"
echo "  OS: $(uname -s)"
echo "  Kernel: $(uname -r)"
echo "  Arch: $(uname -m)"
echo "  Shell: $SHELL"
echo ""

# Summary
echo "=========================================="
if [ $ERROR_COUNT -eq 0 ]; then
    echo "✓ All checks passed!"
    echo "=========================================="
    echo ""
    echo "Ready to run Part C experiment:"
    echo "  $ ./run_partC.sh"
    echo ""
    exit 0
else
    echo "✗ Found $ERROR_COUNT error(s)"
    echo "=========================================="
    echo ""
    echo "Please fix the errors above before running."
    echo "See SETUP_LINUX.md for installation instructions."
    echo ""
    exit 1
fi
