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
        
        if late_df.empty:
            flow_throughputs.append(0.0)
            continue
        
        total_bits = late_df['size'].sum() * 8
        actual_duration = late_df['time'].max() - late_df['time'].min()
        late_goodput = total_bits / actual_duration * 1e-6 if actual_duration > 0 else 0.0
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
    恢复四宫格（2x2）布局的可视化：
    - 每个指标一个子图
    - 显示 Mean ± 95% CI
    - 柱子上显示 4 位小数
    - CI 用 shaded band + annotation box
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 使用统一风格
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_goodput, ax_plr, ax_jain, ax_cov = axes.flatten()

    # 四个指标
    metric_order = [
        ('Goodput (Mbps)', ax_goodput, 'Goodput (Mbps)'),
        ('PLR (%)', ax_plr, 'Packet Loss Rate (%)'),
        ('Jain Index', ax_jain, "Jain's Fairness Index"),
        ('CoV', ax_cov, 'Coefficient of Variation')
    ]

    runs = ['1', '2', '3', '4', '5']
    x_pos = np.arange(len(runs))

    for metric_name, ax, ylabel in metric_order:
        values = metrics[metric_name]

        # 计算统计区间
        mean, lower, upper, margin = calculate_confidence_interval(values)

        # 绘制柱状图
        bars = ax.bar(
            x_pos, values,
            color="#1f77b4",
            alpha=0.85,
            edgecolor="white",
            linewidth=1.5,
            width=0.65
        )

        # 设置坐标轴
        ax.set_xticks(x_pos)
        ax.set_xticklabels(runs)
        ax.set_xlabel("Run", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight="600", pad=8)

        # 注：下界必须 >= 0（PLR/Goodput/CoV/Jain 均不能为负）
        y_min = max(0, min(values + [lower]))
        y_max = max(values + [upper])
        if y_max == y_min:
            y_max = y_min + 0.01

        ax.set_ylim(y_min - (y_max - y_min) * 0.1,
                    y_max + (y_max - y_min) * 0.15)

        # 在柱子上显示数值（4位小数）
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{val:.4f}',   # ← 4位小数
                ha='center',
                va='bottom',
                fontsize=8
            )

        # 绘制 mean 水平线
        ax.axhline(mean, color='black', linewidth=1.4, linestyle='--')

        # 绘制 95% CI 区间阴影
        ax.fill_between(
            [-0.5, len(runs) - 0.5],
            lower,
            upper,
            color="#1f77b4",
            alpha=0.12
        )

        # 右下角显示 CI 注释框
        annotation = f"Mean={mean:.4f}\n95% CI=[{lower:.4f}, {upper:.4f}]"
        ax.text(
            0.98, 0.02, annotation,
            transform=ax.transAxes,
            fontsize=8,
            ha='right',
            va='bottom',
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor='white',
                edgecolor='gray',
                alpha=0.7
            )
        )

    plt.tight_layout()
    plt.savefig("partC_reproducibility_analysis.png", dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved to: partC_reproducibility_analysis.png")
    plt.show()
    plt.close()







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
    print(" Analysis complete! Generated files:")
    print("  - partC_statistics.csv")
    print("  - partC_per_run_results.csv")
    print("  - partC_reproducibility_analysis.png")
    print("="*80)


if __name__ == "__main__":
    main()
