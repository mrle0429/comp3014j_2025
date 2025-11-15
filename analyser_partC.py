#!/usr/bin/env python3
"""
Part C: Reproducibility Analysis for TCP Yeah
This script analyzes multiple runs of TCP Yeah simulation to verify consistency
and calculate statistical metrics including mean and 95% confidence intervals.
"""

import sys
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def parse_trace(trace_file, flow_id='1'):
    """
    Parse NS2 trace file and extract performance metrics for a specific flow.
    
    Args:
        trace_file: Path to the trace file
        flow_id: Flow identifier ('1' or '2')
    
    Returns:
        Dictionary containing goodput, packet loss rate, and other metrics
    """
    sent_packets = 0
    received_packets = 0
    dropped_packets = 0
    total_bytes = 0
    first_time = None
    last_time = None
    
    try:
        with open(trace_file, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 12:
                    continue
                
                event = parts[0]
                time = float(parts[1])
                packet_type = parts[4]
                size = int(parts[5])
                flow = parts[-2]  # Flow ID is usually at position -2
                
                # Only process TCP packets for the specified flow
                if packet_type != 'tcp':
                    continue
                
                if flow != flow_id:
                    continue
                
                if event == '+':  # Packet enqueued
                    sent_packets += 1
                elif event == 'r':  # Packet received
                    received_packets += 1
                    total_bytes += size
                    if first_time is None:
                        first_time = time
                    last_time = time
                elif event == 'd':  # Packet dropped
                    dropped_packets += 1
        
        # Calculate metrics
        duration = (last_time - first_time) if (last_time and first_time) else 1.0
        goodput_mbps = (total_bytes * 8 / duration) / 1e6 if duration > 0 else 0
        loss_rate = (dropped_packets / sent_packets * 100) if sent_packets > 0 else 0
        
        return {
            'goodput': goodput_mbps,
            'loss_rate': loss_rate,
            'sent': sent_packets,
            'received': received_packets,
            'dropped': dropped_packets,
            'duration': duration
        }
    
    except FileNotFoundError:
        print(f"Warning: File {trace_file} not found")
        return None


def calculate_statistics(data_list):
    """
    Calculate mean, std, and 95% confidence interval.
    
    Args:
        data_list: List of numeric values
    
    Returns:
        Dictionary with mean, std, ci_lower, ci_upper
    """
    if not data_list or len(data_list) == 0:
        return None
    
    data = np.array(data_list)
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Sample standard deviation
    
    # 95% confidence interval using t-distribution
    confidence = 0.95
    t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_error = t_critical * (std / np.sqrt(n))
    
    return {
        'mean': mean,
        'std': std,
        'ci_lower': mean - margin_error,
        'ci_upper': mean + margin_error,
        'margin_error': margin_error,
        'raw_data': data_list
    }


def analyze_multiple_runs(num_runs=5):
    """
    Analyze multiple simulation runs and calculate statistics.
    
    Args:
        num_runs: Number of simulation runs to analyze
    
    Returns:
        Dictionary containing statistics for each metric
    """
    print(f"\n{'='*60}")
    print(f"Part C: Analyzing {num_runs} runs of TCP Yeah")
    print(f"{'='*60}\n")
    
    # Storage for metrics from all runs
    flow1_goodputs = []
    flow1_loss_rates = []
    flow2_goodputs = []
    flow2_loss_rates = []
    total_goodputs = []
    avg_loss_rates = []
    
    # Parse each run
    for i in range(num_runs):
        trace_file = f"yeahTrace_run{i}.tr"
        print(f"Processing Run {i+1}: {trace_file}")
        
        # Parse Flow 1
        flow1_data = parse_trace(trace_file, flow_id='1')
        if flow1_data:
            flow1_goodputs.append(flow1_data['goodput'])
            flow1_loss_rates.append(flow1_data['loss_rate'])
            print(f"  Flow 1 - Goodput: {flow1_data['goodput']:.4f} Mbps, PLR: {flow1_data['loss_rate']:.2f}%")
        
        # Parse Flow 2
        flow2_data = parse_trace(trace_file, flow_id='2')
        if flow2_data:
            flow2_goodputs.append(flow2_data['goodput'])
            flow2_loss_rates.append(flow2_data['loss_rate'])
            print(f"  Flow 2 - Goodput: {flow2_data['goodput']:.4f} Mbps, PLR: {flow2_data['loss_rate']:.2f}%")
        
        # Calculate aggregate metrics
        if flow1_data and flow2_data:
            total_goodput = flow1_data['goodput'] + flow2_data['goodput']
            avg_loss = (flow1_data['loss_rate'] + flow2_data['loss_rate']) / 2
            total_goodputs.append(total_goodput)
            avg_loss_rates.append(avg_loss)
            print(f"  Total Goodput: {total_goodput:.4f} Mbps, Avg PLR: {avg_loss:.2f}%")
        print()
    
    # Calculate statistics
    results = {
        'flow1_goodput': calculate_statistics(flow1_goodputs),
        'flow1_loss_rate': calculate_statistics(flow1_loss_rates),
        'flow2_goodput': calculate_statistics(flow2_goodputs),
        'flow2_loss_rate': calculate_statistics(flow2_loss_rates),
        'total_goodput': calculate_statistics(total_goodputs),
        'avg_loss_rate': calculate_statistics(avg_loss_rates)
    }
    
    return results


def print_statistics(results):
    """Print statistical results in a formatted table."""
    print(f"\n{'='*60}")
    print("Statistical Analysis Results (n=5, 95% CI)")
    print(f"{'='*60}\n")
    
    metrics = [
        ('Flow 1 Goodput (Mbps)', 'flow1_goodput'),
        ('Flow 1 PLR (%)', 'flow1_loss_rate'),
        ('Flow 2 Goodput (Mbps)', 'flow2_goodput'),
        ('Flow 2 PLR (%)', 'flow2_loss_rate'),
        ('Total Goodput (Mbps)', 'total_goodput'),
        ('Average PLR (%)', 'avg_loss_rate')
    ]
    
    for label, key in metrics:
        stats_data = results[key]
        if stats_data:
            print(f"{label}:")
            print(f"  Mean:  {stats_data['mean']:.4f}")
            print(f"  Std:   {stats_data['std']:.4f}")
            print(f"  95% CI: [{stats_data['ci_lower']:.4f}, {stats_data['ci_upper']:.4f}]")
            print()


def save_to_csv(results, output_file='results_partC.csv'):
    """Save results to CSV file."""
    rows = []
    
    metrics = [
        ('Flow 1 Goodput (Mbps)', 'flow1_goodput'),
        ('Flow 1 PLR (%)', 'flow1_loss_rate'),
        ('Flow 2 Goodput (Mbps)', 'flow2_goodput'),
        ('Flow 2 PLR (%)', 'flow2_loss_rate'),
        ('Total Goodput (Mbps)', 'total_goodput'),
        ('Average PLR (%)', 'avg_loss_rate')
    ]
    
    for label, key in metrics:
        stats_data = results[key]
        if stats_data:
            rows.append({
                'Metric': label,
                'Mean': stats_data['mean'],
                'Std Dev': stats_data['std'],
                'CI Lower (95%)': stats_data['ci_lower'],
                'CI Upper (95%)': stats_data['ci_upper'],
                'Margin of Error': stats_data['margin_error']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"✓ Results saved to {output_file}")


def plot_confidence_intervals(results, output_file='plots/partC_confidence_intervals.png'):
    """Generate visualization with confidence intervals."""
    
    # Create plots directory if it doesn't exist
    Path('plots').mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('TCP Yeah Reproducibility Analysis (5 runs, 95% CI)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Flow Goodput
    ax1 = axes[0, 0]
    flow_labels = ['Flow 1', 'Flow 2', 'Total']
    flow_keys = ['flow1_goodput', 'flow2_goodput', 'total_goodput']
    means = [results[k]['mean'] for k in flow_keys]
    errors = [results[k]['margin_error'] for k in flow_keys]
    
    bars1 = ax1.bar(flow_labels, means, yerr=errors, capsize=10, 
                    color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Goodput (Mbps)', fontweight='bold')
    ax1.set_title('Goodput Comparison (with 95% CI)', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Packet Loss Rate
    ax2 = axes[0, 1]
    loss_labels = ['Flow 1', 'Flow 2', 'Average']
    loss_keys = ['flow1_loss_rate', 'flow2_loss_rate', 'avg_loss_rate']
    loss_means = [results[k]['mean'] for k in loss_keys]
    loss_errors = [results[k]['margin_error'] for k in loss_keys]
    
    bars2 = ax2.bar(loss_labels, loss_means, yerr=loss_errors, capsize=10,
                    color=['#3498db', '#e74c3c', '#f39c12'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Packet Loss Rate (%)', fontweight='bold')
    ax2.set_title('Packet Loss Rate (with 95% CI)', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Box plot for Goodput
    ax3 = axes[1, 0]
    goodput_data = [
        results['flow1_goodput']['raw_data'],
        results['flow2_goodput']['raw_data'],
        results['total_goodput']['raw_data']
    ]
    bp1 = ax3.boxplot(goodput_data, labels=flow_labels, patch_artist=True)
    for patch, color in zip(bp1['boxes'], ['#3498db', '#e74c3c', '#2ecc71']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.set_ylabel('Goodput (Mbps)', fontweight='bold')
    ax3.set_title('Goodput Distribution Across 5 Runs', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Plot 4: Box plot for Loss Rate
    ax4 = axes[1, 1]
    loss_data = [
        results['flow1_loss_rate']['raw_data'],
        results['flow2_loss_rate']['raw_data'],
        results['avg_loss_rate']['raw_data']
    ]
    bp2 = ax4.boxplot(loss_data, labels=loss_labels, patch_artist=True)
    for patch, color in zip(bp2['boxes'], ['#3498db', '#e74c3c', '#f39c12']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax4.set_ylabel('Packet Loss Rate (%)', fontweight='bold')
    ax4.set_title('PLR Distribution Across 5 Runs', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {output_file}")
    plt.show()


def main():
    """Main function to run Part C analysis."""
    
    # Default number of runs
    num_runs = 5
    
    # Check command line arguments
    if len(sys.argv) > 1:
        try:
            num_runs = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of runs: {sys.argv[1]}, using default: 5")
    
    # Analyze multiple runs
    results = analyze_multiple_runs(num_runs)
    
    # Print statistics
    print_statistics(results)
    
    # Save to CSV
    save_to_csv(results)
    
    # Generate plots
    plot_confidence_intervals(results)
    
    print(f"\n{'='*60}")
    print("Part C Analysis Completed Successfully!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
