#!/bin/bash
################################################################################
# Part C: Automated Reproducibility Testing Script
# TCP Yeah with RED Queue Management
# Platform: Linux (Ubuntu/Debian compatible)
################################################################################

set -e  # Exit on error
set -u  # Error on undefined variables

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[✓]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }

# Check dependencies
command -v ns &> /dev/null || { print_error "ns2 not found"; exit 1; }
command -v python3 &> /dev/null || { print_error "python3 not found"; exit 1; }

echo "=========================================="
echo "  Part C: Reproducibility Testing"
echo "  TCP Yeah + RED Queue (5 runs)"
echo "=========================================="

# Define random seeds
SEEDS=(1000 2000 3000 4000 5000)

# Define random seeds
SEEDS=(1000 2000 3000 4000 5000)

# Clean up old files
print_info "Cleaning old trace files..."
rm -f yeahTrace_run*.tr yeah_run*.nam
print_success "Cleanup complete"

# Run 5 simulations
echo ""
print_info "Running 5 simulations with different seeds..."
echo "=========================================="

for i in $(seq 1 5); do
    seed=${SEEDS[$((i-1))]}
    echo ""
    print_info "[$i/5] Running: seed=$seed, suffix=$i"
    
    if ns yeahCode_partC.tcl $seed $i; then
        if [ -f "yeahTrace_run$i.tr" ]; then
            print_success "yeahTrace_run$i.tr created"
        else
            print_error "Failed to generate yeahTrace_run$i.tr"
            exit 1
        fi
    else
        print_error "Simulation $i failed"
        exit 1
    fi
done

echo ""
print_success "All 5 simulations completed!"

# Run analysis
echo ""
print_info "Running statistical analysis..."
if python3 analyser_partC.py; then
    print_success "Analysis completed"
else
    print_error "Analysis failed"
    exit 1
fi

# Check outputs
echo ""
print_info "Checking outputs..."
all_found=true
for file in partC_statistics.csv partC_per_run_results.csv partC_reproducibility_analysis.png; do
    if [ -f "$file" ]; then
        print_success "$file"
    else
        print_error "$file not found"
        all_found=false
    fi
done

# Summary
echo ""
echo "=========================================="
if [ "$all_found" = true ]; then
    print_success "Part C Analysis Complete!"
    echo "=========================================="
    echo ""
    echo "Files generated:"
    echo "  • Trace files:  yeahTrace_run{1..5}.tr"
    echo "  • Statistics:   partC_statistics.csv"
    echo "  • Raw data:     partC_per_run_results.csv"
    echo "  • Plot:         partC_reproducibility_analysis.png"
    echo ""
    echo "Seeds used: ${SEEDS[*]}"
else
    print_error "Some files missing!"
    exit 1
fi
echo "=========================================="
