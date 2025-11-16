#!/usr/bin/env python3
"""
Part C: Reproducibility Analysis for TCP Yeah with RED Queue
Analyzes 5 simulation runs with different random seeds and computes:
- Mean and 95% confidence intervals for key metrics
- Statistical visualization with error bars
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os


def parse_trace(file_path, flow_identifier):
    """
    Parse trace file and extract goodput and packet loss rate for a specific flow.
    Returns: (goodput_mbps, loss_rate_pct, dataframe)
    """
    data = []
    send_count = 0
    recv_count = 0
    flow_src, flow_dst_final = flow_identifier
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                fields = line.strip().split()
                if len(fields) < 6:
                    continue
                
                event = fields[0]
                src = fields[2]
                dst = fields[3]
                proto = fields[4]
                
                if event not in ['+', 'r'] or proto.lower() != 'tcp':
                    continue
                
                try:
                    size = int(fields[5])
                except ValueError:
                    continue
                
                if size < 1000:  # Filter out ACK packets
                    continue
                
                try:
                    time = float(fields[1])
                except ValueError:
                    time = 0.0
                
                if event == '+' and src == flow_src:
                    send_count += 1
                    data.append({
                        'event': event,
                        'time': time,
                        'size': size,
                        'src': src,
                        'dst': dst
                    })
                elif event == 'r' and dst == flow_dst_final:
                    recv_count += 1
                    data.append({
                        'event': event,
                        'time': time,
                        'size': size,
                        'src': src,
                        'dst': dst
                    })
        
        df = pd.DataFrame(data) if data else pd.DataFrame(columns=['event', 'time', 'size', 'src', 'dst'])
        
        # Calculate goodput
        recv_df = df[df['event'] == 'r']
        if not recv_df.empty:
            total_bits = recv_df['size'].sum() * 8
            total_time = recv_df['time'].max() - recv_df['time'].min()
            goodput = total_bits / total_time * 1e-6 if total_time > 0 else 0.0
        else:
            goodput = 0.0
        
        # Calculate loss rate
        loss_rate = ((send_count - recv_count) / send_count) * 100 if send_count > 0 else 0.0
        
        return goodput, loss_rate, df
    
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return 0.0, 0.0, pd.DataFrame()
    except Exception as e:
        print(f"Error parsing {file_path}: {str(e)}")
        return 0.0, 0.0, pd.DataFrame()


def calculate_jain_fairness(df1, df2):
    """
    Calculate Jain's fairness index over last 1/3 of simulation time.
    """
    flow_throughputs = []
    
    for df in (df1, df2):
        if df.empty:
            flow_throughputs.append(0.0)
            continue
        
        T = df['time'].max()
        if T <= 0:
            flow_throughputs.append(0.0)
            continue
        
        start = 2 * T / 3
        late_df = df[(df['event'] == 'r') & (df['time'] >= start)]
        total_bits = late_df['size'].sum() * 8
        late_goodput = total_bits / (T / 3) * 1e-6 if T > 0 else 0.0
        flow_throughputs.append(late_goodput)
    
    if sum(x * x for x in flow_throughputs) > 0:
        numerator = (sum(flow_throughputs)) ** 2
        denominator = len(flow_throughputs) * sum(x**2 for x in flow_throughputs)
        jain = numerator / denominator
    else:
        jain = 0.0
    
    return jain


def calculate_throughput_cov(df):
    """
    Calculate Coefficient of Variation (CoV) for throughput stability.
    """
    if df.empty:
        return 0.0
    
    df = df.copy()
    df['time_second'] = df['time'].astype(int)
    recv_per_sec = df[df['event'] == 'r'].groupby('time_second')['size'].sum() * 8 / 1e6
    
    if recv_per_sec.empty:
        return 0.0
    
    mean = recv_per_sec.mean()
    std = recv_per_sec.std()
    
    return std / mean if mean != 0 else 0.0


def analyze_single_run(run_number):
    """
    Analyze a single simulation run and return all metrics.
    Returns dict with: goodput, plr, jain_index, cov
    """
    trace_file = f'yeahTrace_run{run_number}.tr'
    
    if not os.path.exists(trace_file):
        print(f"Warning: {trace_file} not found!")
        return None
    
    print(f"Analyzing run {run_number}: {trace_file}")
    
    # Flow identifiers: n1->n5 (0->4), n2->n6 (1->5)
    flow_id_1 = ('0', '4')
    flow_id_2 = ('1', '5')
    
    # Parse both flows
    g1, l1, df1 = parse_trace(trace_file, flow_id_1)
    g2, l2, df2 = parse_trace(trace_file, flow_id_2)
    
    # Calculate metrics
    avg_goodput = (g1 + g2) / 2.0
    avg_plr = (l1 + l2) / 2.0
    jain_index = calculate_jain_fairness(df1, df2)
    cov1 = calculate_throughput_cov(df1)
    cov2 = calculate_throughput_cov(df2)
    avg_cov = (cov1 + cov2) / 2.0
    
    return {
        'run': run_number,
        'goodput': avg_goodput,
        'plr': avg_plr,
        'jain_index': jain_index,
        'cov': avg_cov,
        'flow1_goodput': g1,
        'flow2_goodput': g2,
        'flow1_plr': l1,
        'flow2_plr': l2
    }


def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate mean and 95% confidence interval using t-distribution.
    Returns: (mean, lower_bound, upper_bound, margin_of_error)
    """
    n = len(data)
    if n < 2:
        return np.mean(data), np.mean(data), np.mean(data), 0.0
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Sample standard deviation
    se = std / np.sqrt(n)  # Standard error
    
    # t-value for 95% CI with n-1 degrees of freedom
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_value * se
    
    return mean, mean - margin, mean + margin, margin


def generate_statistics_table(results):
    """
    Generate comprehensive statistics table with mean and 95% CI.
    """
    # Extract metrics from all runs
    goodputs = [r['goodput'] for r in results]
    plrs = [r['plr'] for r in results]
    jains = [r['jain_index'] for r in results]
    covs = [r['cov'] for r in results]
    
    # Calculate statistics
    metrics = {
        'Goodput (Mbps)': goodputs,
        'PLR (%)': plrs,
        'Jain Index': jains,
        'CoV': covs
    }
    
    print("\n" + "="*80)
    print("PART C: REPRODUCIBILITY ANALYSIS - TCP Yeah with RED Queue")
    print("="*80)
    print(f"\nNumber of runs: {len(results)}")
    print(f"Random seeds used: {[r['run'] for r in results]}")
    
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY (Mean ± 95% Confidence Interval)")
    print("="*80)
    
    stats_data = []
    for metric_name, values in metrics.items():
        mean, lower, upper, margin = calculate_confidence_interval(values)
        stats_data.append({
            'Metric': metric_name,
            'Mean': f'{mean:.4f}',
            'Std Dev': f'{np.std(values, ddof=1):.4f}',
            '95% CI Lower': f'{lower:.4f}',
            '95% CI Upper': f'{upper:.4f}',
            'Margin': f'±{margin:.4f}'
        })
    
    df_stats = pd.DataFrame(stats_data)
    print(df_stats.to_string(index=False))
    
    # Save to CSV
    df_stats.to_csv('partC_statistics.csv', index=False)
    print("\n✓ Statistics saved to: partC_statistics.csv")
    
    # Per-run details
    print("\n" + "="*80)
    print("PER-RUN DETAILED RESULTS")
    print("="*80)
    df_runs = pd.DataFrame(results)
    print(df_runs[['run', 'goodput', 'plr', 'jain_index', 'cov']].to_string(index=False))
    df_runs.to_csv('partC_per_run_results.csv', index=False)
    print("\n✓ Per-run results saved to: partC_per_run_results.csv")
    
    return metrics, stats_data


def plot_results_with_ci(metrics):
    """
    Generate publication-quality plots showing each run's value with 95% CI overlay.
    Each subplot shows 5 bars (one per run) with values labeled and CI band.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('Part C: Reproducibility Analysis - TCP Yeah with RED Queue', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    metric_configs = [
        ('Goodput (Mbps)', 'Goodput (Mbps)', 'green', axes[0, 0]),
        ('PLR (%)', 'Packet Loss Rate (%)', 'red', axes[0, 1]),
        ('Jain Index', 'Jain\'s Fairness Index', 'blue', axes[1, 0]),
        ('CoV', 'Coefficient of Variation', 'orange', axes[1, 1])
    ]
    
    for metric_name, ylabel, color, ax in metric_configs:
        values = metrics[metric_name]
        mean, lower, upper, margin = calculate_confidence_interval(values)
        
        # Plot individual runs as bars
        runs = ['Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5']
        x_pos = np.arange(len(runs))
        
        bars = ax.bar(x_pos, values, color=color, alpha=0.7, edgecolor='black', linewidth=1.5, width=0.6)
        
        # Adjust y-axis limits for better granularity
        data_range = max(values) - min(values)
        if data_range > 0:
            y_margin = data_range * 0.3  # Add 30% margin
            ax.set_ylim(min(values) - y_margin, max(values) + y_margin)
        else:
            # If all values are the same, center around the value
            y_center = values[0]
            ax.set_ylim(y_center * 0.95, y_center * 1.05)
        
        # Add value labels on each bar
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Draw horizontal lines for mean and CI bounds
        ax.axhline(y=mean, color='black', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean:.4f}', alpha=0.8, zorder=3)
        ax.axhline(y=lower, color='darkgray', linestyle=':', linewidth=1.5, 
                  label=f'95% CI: [{lower:.4f}, {upper:.4f}]', alpha=0.6, zorder=3)
        ax.axhline(y=upper, color='darkgray', linestyle=':', linewidth=1.5, alpha=0.6, zorder=3)
        
        # Fill the confidence interval region
        ax.fill_between([-0.5, len(runs)-0.5], lower, upper, alpha=0.15, color=color, 
                       label=f'CI Width: ±{margin:.4f}', zorder=1)
        
        # Formatting
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.set_xlabel('Run Number', fontsize=9)
        ax.set_title(f'{ylabel}', fontsize=11, fontweight='bold', pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(runs, fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
        ax.legend(fontsize=7, loc='best', framealpha=0.95)
        
        # Add statistics box
        std_val = np.std(values, ddof=1)
        cv_val = (std_val/mean*100) if mean != 0 else 0
        stats_text = f'Mean: {mean:.4f}\nStd: {std_val:.4f}\nCV: {cv_val:.2f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=7, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('partC_reproducibility_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved to: partC_reproducibility_analysis.png")
    plt.show()


def plot_run_comparison(results):
    """
    Generate comparison plot showing all 5 runs side by side.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Part C: Run-by-Run Comparison - TCP Yeah with RED Queue', 
                 fontsize=14, fontweight='bold')
    
    runs = [r['run'] for r in results]
    
    # Goodput comparison
    ax = axes[0, 0]
    ax.plot(runs, [r['goodput'] for r in results], 'o-', color='green', linewidth=2, markersize=8)
    ax.set_xlabel('Run Number')
    ax.set_ylabel('Goodput (Mbps)')
    ax.set_title('Goodput Across Runs')
    ax.grid(True, alpha=0.3)
    
    # PLR comparison
    ax = axes[0, 1]
    ax.plot(runs, [r['plr'] for r in results], 'o-', color='red', linewidth=2, markersize=8)
    ax.set_xlabel('Run Number')
    ax.set_ylabel('Packet Loss Rate (%)')
    ax.set_title('PLR Across Runs')
    ax.grid(True, alpha=0.3)
    
    # Jain Index comparison
    ax = axes[1, 0]
    ax.plot(runs, [r['jain_index'] for r in results], 'o-', color='blue', linewidth=2, markersize=8)
    ax.set_xlabel('Run Number')
    ax.set_ylabel('Jain Index')
    ax.set_title('Fairness Across Runs')
    ax.grid(True, alpha=0.3)
    
    # CoV comparison
    ax = axes[1, 1]
    ax.plot(runs, [r['cov'] for r in results], 'o-', color='orange', linewidth=2, markersize=8)
    ax.set_xlabel('Run Number')
    ax.set_ylabel('CoV')
    ax.set_title('Stability Across Runs')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('partC_run_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Comparison plot saved to: partC_run_comparison.png")
    plt.show()


def main():
    """
    Main analysis pipeline for Part C.
    """
    print("="*80)
    print("PART C: REPRODUCIBILITY ANALYSIS")
    print("Scenario: TCP Yeah with RED Queue Management")
    print("="*80)
    
    # Analyze all 5 runs
    results = []
    for run_num in range(1, 6):
        result = analyze_single_run(run_num)
        if result:
            results.append(result)
    
    if not results:
        print("\nError: No trace files found!")
        print("Please run the simulations first using:")
        print("  bash run_partC.sh")
        return
    
    if len(results) < 5:
        print(f"\nWarning: Only {len(results)} out of 5 runs found!")
    
    # Generate statistics
    metrics, stats_data = generate_statistics_table(results)
    
    # Generate plots
    plot_results_with_ci(metrics)
    plot_run_comparison(results)
    
    # Summary conclusion
    print("\n" + "="*80)
    print("REPRODUCIBILITY ASSESSMENT")
    print("="*80)
    
    for stat in stats_data:
        metric = stat['Metric']
        mean = float(stat['Mean'])
        margin = float(stat['Margin'].replace('±', ''))
        relative_margin = (margin / mean * 100) if mean != 0 else 0
        
        print(f"\n{metric}:")
        print(f"  Mean: {mean:.4f}")
        print(f"  95% CI: [{float(stat['95% CI Lower']):.4f}, {float(stat['95% CI Upper']):.4f}]")
        print(f"  Relative uncertainty: {relative_margin:.2f}%")
    
    print("\n" + "="*80)
    print("✓ Analysis complete! Generated files:")
    print("  - partC_statistics.csv")
    print("  - partC_per_run_results.csv")
    print("  - partC_reproducibility_analysis.png")
    print("  - partC_run_comparison.png")
    print("="*80)


if __name__ == "__main__":
    main()
