#!/bin/bash
################################################################################
# Part C: Automated Reproducibility Testing
# Runs 5 simulations with different seeds and generates statistical analysis
################################################################################

set -e

# Random seeds for 5 independent runs
SEEDS=(1000 2000 3000 4000 5000)

echo "Part C: Running 5 reproducibility tests..."

# Clean old files
rm -f yeahTrace_run*.tr yeah_run*.nam

# Run simulations
for i in $(seq 1 5); do
    echo "Run $i/5 (seed=${SEEDS[$((i-1))]})"
    ns yeahCode_partC.tcl ${SEEDS[$((i-1))]} $i
done

# Statistical analysis
echo "Generating statistical analysis..."
python3 analyser_partC.py

echo "Complete! Outputs: partC_statistics.csv, partC_reproducibility_analysis.png"
