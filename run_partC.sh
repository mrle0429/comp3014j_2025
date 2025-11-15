#!/bin/bash
################################################################################
# Part C: Reproducibility Experiment - Automated Execution Script
# 
# This script automates the entire Part C workflow:
# (i)   Runs 5 simulations with different random seeds
# (ii)  Runs the analyser to compute statistics and confidence intervals
# (iii) Generates CSV files and plots
#
# Scenario: TCP Yeah with DropTail queue
# Reason: Lowest packet loss rate (0.86%) and most balanced performance
################################################################################

set -e  # Exit on error

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================="
echo "Part C: Reproducibility Experiments"
echo "TCP Yeah with DropTail Queue"
echo -e "==========================================${NC}\n"

# ============ Configuration ============
SCENARIO="yeah"
NUM_RUNS=5
SEEDS=(12345 23456 34567 45678 56789)
TCL_FILE="yeahCode_partC.tcl"
ANALYSER="analyser_partC.py"

# ============ Step 1: Run Simulations ============
echo -e "${YELLOW}[Step 1/3] Running NS2 simulations with different seeds...${NC}\n"

for i in $(seq 0 $((NUM_RUNS-1))); do
    seed=${SEEDS[$i]}
    run_num=$i
    
    echo -e "${GREEN}Run $((i+1))/$NUM_RUNS${NC} - Seed: $seed"
    
    # Run NS2 simulation
    if ns $TCL_FILE $seed $run_num > /dev/null 2>&1; then
        echo "  ✓ Simulation completed successfully"
        
        # Verify trace file was created
        if [ -f "yeahTrace_run${run_num}.tr" ]; then
            file_size=$(stat -c%s "yeahTrace_run${run_num}.tr" 2>/dev/null || stat -f%z "yeahTrace_run${run_num}.tr" 2>/dev/null)
            echo "  ✓ Trace file generated (${file_size} bytes)"
        else
            echo -e "  ${RED}✗ Warning: Trace file not found${NC}"
        fi
    else
        echo -e "  ${RED}✗ Error: Simulation failed${NC}"
        exit 1
    fi
    
    echo ""
done

echo -e "${GREEN}✓ All simulations completed!${NC}\n"

# ============ Step 2: Run Analyser ============
echo -e "${YELLOW}[Step 2/3] Running analyser to process results...${NC}\n"

if python3 $ANALYSER $NUM_RUNS; then
    echo -e "\n${GREEN}✓ Analysis completed successfully!${NC}\n"
else
    echo -e "\n${RED}✗ Error: Analysis failed${NC}"
    echo "Make sure you have required Python packages installed:"
    echo "  pip3 install numpy scipy pandas matplotlib"
    exit 1
fi

# ============ Step 3: Summary ============
echo -e "${YELLOW}[Step 3/3] Generating summary...${NC}\n"

# Check generated files
echo "Generated files:"
echo "=================="

if [ -f "results_partC.csv" ]; then
    echo -e "  ${GREEN}✓${NC} CSV file: results_partC.csv"
    echo "    $(wc -l < results_partC.csv) rows"
else
    echo -e "  ${RED}✗${NC} CSV file not found"
fi

if [ -f "plots/partC_confidence_intervals.png" ]; then
    echo -e "  ${GREEN}✓${NC} Plot: plots/partC_confidence_intervals.png"
    file_size=$(stat -c%s "plots/partC_confidence_intervals.png" 2>/dev/null || stat -f%z "plots/partC_confidence_intervals.png" 2>/dev/null)
    echo "    Size: ${file_size} bytes"
else
    echo -e "  ${RED}✗${NC} Plot file not found"
fi

echo ""
echo "Trace files:"
echo "============"
for i in $(seq 0 $((NUM_RUNS-1))); do
    trace_file="yeahTrace_run${i}.tr"
    if [ -f "$trace_file" ]; then
        file_size=$(stat -c%s "$trace_file" 2>/dev/null || stat -f%z "$trace_file" 2>/dev/null)
        echo -e "  ${GREEN}✓${NC} $trace_file (${file_size} bytes)"
    fi
done

echo ""
echo -e "${BLUE}=========================================="
echo "Part C Execution Completed Successfully!"
echo -e "==========================================${NC}\n"

echo "Next steps:"
echo "  1. Check results_partC.csv for statistical data"
echo "  2. View plots/partC_confidence_intervals.png for visualizations"
echo "  3. Include these results in your Part C report"
echo ""

# ============ Optional: Display CSV preview ============
if command -v column &> /dev/null && [ -f "results_partC.csv" ]; then
    echo -e "${YELLOW}CSV Preview:${NC}"
    head -n 5 results_partC.csv | column -t -s,
    echo ""
fi
