#!/bin/bash
################################################################################
# Part C: Automated Reproducibility Testing Script
# TCP Yeah with RED Queue Management
#
# This script:
#   1. Runs 5 simulations with different random seeds
#   2. Analyzes results using analyser_partC.py
#   3. Generates CSV files and plots
#
# Usage: bash run_partC.sh
################################################################################

set -e  # Exit on error

echo "=================================="
echo "Part C: Reproducibility Testing"
echo "Scenario: TCP Yeah + RED Queue"
echo "=================================="

# Define random seeds for reproducibility
SEEDS=(22207256 22207264 22207247 10000000 2222222)

# Clean up old results (optional - comment out to keep previous runs)
echo ""
echo "Cleaning up old trace files..."
rm -f yeahTrace_run*.tr yeah_run*.nam

# Run 5 simulations with different seeds
echo ""
echo "Running 5 simulations with different random seeds..."
echo "======================================================="

for i in {1..5}; do
    seed=${SEEDS[$i-1]}
    echo ""
    echo "[$i/5] Running simulation with seed=$seed, output suffix=$i"
    ns yeahCode_partC.tcl $seed $i
    
    if [ -f "yeahTrace_run$i.tr" ]; then
        size=$(wc -c < "yeahTrace_run$i.tr")
        echo "      ✓ Generated yeahTrace_run$i.tr (${size} bytes)"
    else
        echo "      ✗ Error: Failed to generate yeahTrace_run$i.tr"
        exit 1
    fi
done

echo ""
echo "======================================================="
echo "✓ All 5 simulations completed successfully!"
echo ""

# Check if trace files exist
echo "Verifying trace files..."
for i in {1..5}; do
    if [ ! -f "yeahTrace_run$i.tr" ]; then
        echo "✗ Error: yeahTrace_run$i.tr not found!"
        exit 1
    fi
done
echo "✓ All 5 trace files verified"

# Run the analyser script
echo ""
echo "======================================================="
echo "Running statistical analysis (analyser_partC.py)..."
echo "======================================================="
python3 analyser_partC.py

# Check if output files were generated
echo ""
echo "======================================================="
echo "Verifying generated outputs..."
echo "======================================================="

OUTPUT_FILES=(
    "partC_statistics.csv"
    "partC_per_run_results.csv"
    "partC_reproducibility_analysis.png"
    "partC_run_comparison.png"
)

all_found=true
for file in "${OUTPUT_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (not found)"
        all_found=false
    fi
done

echo ""
if [ "$all_found" = true ]; then
    echo "=================================="
    echo "✓ Part C Analysis Complete!"
    echo "=================================="
    echo ""
    echo "Generated files:"
    echo "  Trace files:  yeahTrace_run1.tr ~ yeahTrace_run5.tr"
    echo "  Statistics:   partC_statistics.csv"
    echo "  Raw data:     partC_per_run_results.csv"
    echo "  Plots:        partC_reproducibility_analysis.png"
    echo "                partC_run_comparison.png"
    echo ""
    echo "Random seeds used: ${SEEDS[*]}"
    echo ""
    echo "Next steps:"
    echo "  1. Review the CSV files for statistical results"
    echo "  2. Check the PNG plots for visualization"
    echo "  3. Write your Part C report based on these results"
else
    echo "⚠️  Warning: Some output files are missing!"
    echo "Please check the analyser_partC.py output for errors."
    exit 1
fi
