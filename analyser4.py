import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def parse_trace(file_path, flow_identifier):
    """
    Parse the trace file and extract the throughput, packet loss rate and time series data of the specified stream.
    """
    data = []
    send_count = 0
    recv_count = 0
    flow_src, flow_dst_final = flow_identifier
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
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
                if size < 1000:
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
        recv_df = df[df['event'] == 'r']
        if not recv_df.empty:
            total_bits = recv_df['size'].sum() * 8
            total_time = recv_df['time'].max() - recv_df['time'].min()
            goodput = total_bits / total_time * 1e-6 if total_time > 0 else 0.0
        else:
            goodput = 0.0
        loss_rate = ((send_count - recv_count) / send_count) * 100 if send_count > 0 else 0.0
        return goodput, loss_rate, df
    except FileNotFoundError:
        print(f"no file: {file_path}")
        return 0.0, 0.0, pd.DataFrame()
    except Exception as e:
        print(f"Error occurred while parsing {file_path}：{str(e)}")
        return 0.0, 0.0, pd.DataFrame()


def calculate_jain_fairness(file_path):
    """
    Calculate Jain's fairness index for the two flows in the given trace file
    """
    flow_id_1 = ('0', '4')
    flow_id_2 = ('1', '5')

    _, _, df1 = parse_trace(file_path, flow_id_1)
    _, _, df2 = parse_trace(file_path, flow_id_2)

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
        denominator = len(flow_throughputs) * sum(x ** 2 for x in flow_throughputs)
        jain = numerator / denominator
    else:
        jain = 0.0

    return jain, flow_throughputs


def calculate_throughput_cov(file_path):
    """
    Calculate Coefficient of Variation for throughput stability
    """
    flow_id_1 = ('0', '4')
    flow_id_2 = ('1', '5')

    def cov_from_df(df):
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

    _, _, df1 = parse_trace(file_path, flow_id_1)
    _, _, df2 = parse_trace(file_path, flow_id_2)

    cov1 = cov_from_df(df1)
    cov2 = cov_from_df(df2)
    cov_avg = (cov1 + cov2) / 2.0

    return cov_avg


def normalize_to_1_5(value, min_val, max_val, higher_is_better=True):
    """
    Normalize a value to the range 1-5
    For metrics where higher is better (goodput, fairness): 5 is best, 1 is worst
    For metrics where lower is better (loss rate, CoV): 1 is best, 5 is worst
    """
    if max_val == min_val:
        return 3.0  # Middle value if all values are the same

    if higher_is_better:
        # Higher values get higher scores (5 is best)
        normalized = 1 + 4 * (value - min_val) / (max_val - min_val)
    else:
        # Lower values get higher scores (1 is best)
        normalized = 1 + 4 * (max_val - value) / (max_val - min_val)

    return max(1.0, min(5.0, normalized))  # Clamp to 1-5 range


def analyze_cubic_environments():
    """
    Analyze Cubic TCP algorithm performance in different environments
    """
    # Define the three test scenarios
    scenarios = {
        'Cubic_2Gb_DropTail': 'cubicTrace_2Gb.tr',
        'Cubic_1Gb_DropTail': 'cubicTrace_DropTail.tr',
        'Cubic_1Gb_RED': 'cubicTrace_RED.tr'
    }

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    flow_id_1 = ('0', '4')  # Flow 1: n1 -> n5
    flow_id_2 = ('1', '5')  # Flow 2: n2 -> n6

    # Store results
    results = {}
    raw_metrics = {}

    print("=== Cubic TCP Algorithm Performance in Different Environments ===")

    for scenario, trace_file in scenarios.items():
        if not os.path.exists(trace_file):
            print(f"Warning: File {trace_file} not found, skipping {scenario}")
            continue

        print(f"\nAnalyzing {scenario}...")

        # Parse goodput and loss rate for each flow
        g1, l1, _ = parse_trace(trace_file, flow_id_1)
        g2, l2, _ = parse_trace(trace_file, flow_id_2)

        # Calculate average metrics
        avg_goodput = (g1 + g2) / 2.0
        avg_loss = (l1 + l2) / 2.0

        # Calculate Jain's fairness index
        jain, flow_throughputs = calculate_jain_fairness(trace_file)

        # Calculate throughput stability (CoV)
        cov = calculate_throughput_cov(trace_file)

        # Store raw results
        raw_metrics[scenario] = {
            'Flow1_Goodput': g1,
            'Flow1_LossRate': l1,
            'Flow2_Goodput': g2,
            'Flow2_LossRate': l2,
            'Avg_Goodput': avg_goodput,
            'Avg_LossRate': avg_loss,
            'Jain_Fairness': jain,
            'Stability_CoV': cov,
            'Flow_Throughputs': flow_throughputs
        }

        print(f"  Flow1: Goodput={g1:.4f} Mbps, LossRate={l1:.4f}%")
        print(f"  Flow2: Goodput={g2:.4f} Mbps, LossRate={l2:.4f}%")
        print(f"  Average: Goodput={avg_goodput:.4f} Mbps, LossRate={avg_loss:.4f}%")
        print(f"  Jain Fairness Index: {jain:.4f}")
        print(f"  Throughput Stability (CoV): {cov:.4f}")

    # Calculate min and max values for normalization
    if raw_metrics:
        goodputs = [m['Avg_Goodput'] for m in raw_metrics.values()]
        losses = [m['Avg_LossRate'] for m in raw_metrics.values()]
        fairness = [m['Jain_Fairness'] for m in raw_metrics.values()]
        covs = [m['Stability_CoV'] for m in raw_metrics.values()]

        min_goodput, max_goodput = min(goodputs), max(goodputs)
        min_loss, max_loss = min(losses), max(losses)
        min_fairness, max_fairness = min(fairness), max(fairness)
        min_cov, max_cov = min(covs), max(covs)

        print(f"\nNormalization ranges:")
        print(f"Goodput: {min_goodput:.4f} - {max_goodput:.4f} Mbps")
        print(f"Loss Rate: {min_loss:.4f} - {max_loss:.4f} %")
        print(f"Fairness: {min_fairness:.4f} - {max_fairness:.4f}")
        print(f"Stability CoV: {min_cov:.4f} - {max_cov:.4f}")

        # Normalize all metrics to 1-5 range
        for scenario, metrics in raw_metrics.items():
            normalized_metrics = {
                'Scenario': scenario,
                'Flow1_Goodput': metrics['Flow1_Goodput'],
                'Flow1_LossRate': metrics['Flow1_LossRate'],
                'Flow2_Goodput': metrics['Flow2_Goodput'],
                'Flow2_LossRate': metrics['Flow2_LossRate'],
                'Avg_Goodput': metrics['Avg_Goodput'],
                'Avg_LossRate': metrics['Avg_LossRate'],
                'Jain_Fairness': metrics['Jain_Fairness'],
                'Stability_CoV': metrics['Stability_CoV'],
                'Flow_Throughputs': metrics['Flow_Throughputs'],

                # Normalized scores (1-5)
                'Goodput_Score': normalize_to_1_5(metrics['Avg_Goodput'], min_goodput, max_goodput,
                                                  higher_is_better=True),
                'LossRate_Score': normalize_to_1_5(metrics['Avg_LossRate'], min_loss, max_loss, higher_is_better=False),
                'Fairness_Score': normalize_to_1_5(metrics['Jain_Fairness'], min_fairness, max_fairness,
                                                   higher_is_better=True),
                'Stability_Score': normalize_to_1_5(metrics['Stability_CoV'], min_cov, max_cov, higher_is_better=False),

                # Per-flow normalized scores
                'Flow1_Goodput_Score': normalize_to_1_5(metrics['Flow1_Goodput'], min(goodputs), max(goodputs),
                                                        higher_is_better=True),
                'Flow2_Goodput_Score': normalize_to_1_5(metrics['Flow2_Goodput'], min(goodputs), max(goodputs),
                                                        higher_is_better=True),
                'Flow1_LossRate_Score': normalize_to_1_5(metrics['Flow1_LossRate'], min(losses), max(losses),
                                                         higher_is_better=False),
                'Flow2_LossRate_Score': normalize_to_1_5(metrics['Flow2_LossRate'], min(losses), max(losses),
                                                         higher_is_better=False),
            }
            results[scenario] = normalized_metrics

    return results, colors, raw_metrics


def generate_comparison_table(results):
    """
    Generate performance comparison table and save to CSV
    """
    table_data = []

    for scenario, metrics in results.items():
        row = {
            'Scenario': scenario,
            'Flow1_Goodput_Mbps': metrics['Flow1_Goodput'],
            'Flow1_LossRate_Pct': metrics['Flow1_LossRate'],
            'Flow2_Goodput_Mbps': metrics['Flow2_Goodput'],
            'Flow2_LossRate_Pct': metrics['Flow2_LossRate'],
            'Avg_Goodput_Mbps': metrics['Avg_Goodput'],
            'Avg_LossRate_Pct': metrics['Avg_LossRate'],
            'Jain_Fairness_Index': metrics['Jain_Fairness'],
            'Stability_CoV': metrics['Stability_CoV'],
            'Goodput_Score': metrics['Goodput_Score'],
            'LossRate_Score': metrics['LossRate_Score'],
            'Fairness_Score': metrics['Fairness_Score'],
            'Stability_Score': metrics['Stability_Score'],
            'Flow1_Goodput_Score': metrics['Flow1_Goodput_Score'],
            'Flow2_Goodput_Score': metrics['Flow2_Goodput_Score'],
            'Flow1_LossRate_Score': metrics['Flow1_LossRate_Score'],
            'Flow2_LossRate_Score': metrics['Flow2_LossRate_Score'],
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)
    print("\n=== Performance Comparison Table (with Normalized Scores 1-5) ===")
    print(df.to_string(index=False))

    # Save to CSV
    df.to_csv('cubic_performance_comparison_normalized.csv', index=False)
    print("\nTable saved to 'cubic_performance_comparison_normalized.csv'")

    return df


def generate_comparison_plots(results, colors):
    """
    Generate comparison plots with normalized scores (1-5)
    """
    scenarios = list(results.keys())

    # Prepare normalized data for plotting (1-5 scale)
    metrics_data = {
        'Fairness': [results[s]['Fairness_Score'] for s in scenarios],
        'Stability': [results[s]['Stability_Score'] for s in scenarios],
        'Goodput': [results[s]['Goodput_Score'] for s in scenarios],
        'Packet Loss Rate': [results[s]['LossRate_Score'] for s in scenarios]
    }

    # Create the main comparison figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Subplot 1: Main metrics comparison (normalized 1-5)
    x_pos = np.arange(len(metrics_data.keys()))
    bar_width = 0.25

    for i, scenario in enumerate(scenarios):
        values = [metrics_data[metric][i] for metric in metrics_data.keys()]
        ax1.bar(x_pos + i * bar_width, values, bar_width,
                label=scenario, color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Performance Metrics', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized Score (1-5)', fontsize=12, fontweight='bold')
    ax1.set_title('Cubic TCP Performance - Normalized Comparison (1-5 Scale)\nHigher scores are better for all metrics',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x_pos + bar_width)
    ax1.set_xticklabels(metrics_data.keys(), fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 5.5)

    # Add value labels on bars and original values in parentheses
    for i, scenario in enumerate(scenarios):
        for j, metric in enumerate(metrics_data.keys()):
            norm_score = metrics_data[metric][i]
            original_key = 'Jain_Fairness' if metric == 'Fairness' else \
                'Stability_CoV' if metric == 'Stability' else \
                    'Avg_Goodput' if metric == 'Goodput' else 'Avg_LossRate'
            original_val = results[scenario][original_key]

            if metric == 'Goodput':
                original_text = f'{original_val:.2f}Mbps'
            elif metric == 'Packet Loss Rate':
                original_text = f'{original_val:.2f}%'
            elif metric == 'Fairness':
                original_text = f'{original_val:.3f}'
            else:  # Stability
                original_text = f'{original_val:.3f}'

            ax1.text(j + i * bar_width, norm_score + 0.1,
                     f'{norm_score:.2f}\n({original_text})',
                     ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Subplot 2: Per-flow goodput and loss rate (normalized)
    flow_metrics = ['Per-flow Goodput', 'Per-flow Loss Rate']
    x_pos_flow = np.arange(len(flow_metrics))

    # Prepare per-flow normalized data
    per_flow_data = {
        'Per-flow Goodput': [],
        'Per-flow Loss Rate': []
    }

    for scenario in scenarios:
        # Average of two flows' normalized scores
        avg_goodput_score = (results[scenario]['Flow1_Goodput_Score'] + results[scenario]['Flow2_Goodput_Score']) / 2
        avg_loss_score = (results[scenario]['Flow1_LossRate_Score'] + results[scenario]['Flow2_LossRate_Score']) / 2

        per_flow_data['Per-flow Goodput'].append(avg_goodput_score)
        per_flow_data['Per-flow Loss Rate'].append(avg_loss_score)

    for i, scenario in enumerate(scenarios):
        values = [per_flow_data[metric][i] for metric in flow_metrics]
        ax2.bar(x_pos_flow + i * bar_width, values, bar_width,
                label=scenario, color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Per-flow Metrics', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Normalized Score (1-5)', fontsize=12, fontweight='bold')
    ax2.set_title(
        'Per-flow Goodput and Packet Loss Rate - Normalized Comparison (1-5 Scale)\nHigher scores are better for all metrics',
        fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(x_pos_flow + bar_width)
    ax2.set_xticklabels(['Goodput Score', 'Loss Rate Score'], fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 5.5)

    # Add value labels for per-flow metrics
    for i, scenario in enumerate(scenarios):
        for j, metric in enumerate(flow_metrics):
            norm_score = per_flow_data[metric][i]

            # Get original values for display
            if metric == 'Per-flow Goodput':
                orig1 = results[scenario]['Flow1_Goodput']
                orig2 = results[scenario]['Flow2_Goodput']
                original_text = f'F1:{orig1:.1f}\nF2:{orig2:.1f}'
            else:  # Per-flow Loss Rate
                orig1 = results[scenario]['Flow1_LossRate']
                orig2 = results[scenario]['Flow2_LossRate']
                original_text = f'F1:{orig1:.1f}%\nF2:{orig2:.1f}%'

            ax2.text(j + i * bar_width, norm_score + 0.1,
                     f'{norm_score:.2f}\n({original_text})',
                     ha='center', va='bottom', fontsize=7, fontweight='bold')

    plt.tight_layout()
    plt.savefig('cubic_environment_comparison_normalized.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_radar_chart(results, colors):
    """
    Generate a radar chart for comprehensive comparison
    """
    scenarios = list(results.keys())
    metrics = ['Goodput', 'Fairness', 'Stability', 'Loss Rate']

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)

    # Angles for each metric
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    for i, scenario in enumerate(scenarios):
        values = [
            results[scenario]['Goodput_Score'],
            results[scenario]['Fairness_Score'],
            results[scenario]['Stability_Score'],
            results[scenario]['LossRate_Score']
        ]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=scenario, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    # Add metric labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'])
    ax.grid(True)
    ax.set_title('Cubic TCP Performance - Radar Chart (Normalized 1-5)\nHigher scores are better',
                 size=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig('cubic_radar_chart_normalized.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_queue_comparison_analysis(results, raw_metrics):
    """
    Specialized analysis for DropTail vs RED queue algorithm comparison
    """
    print("\n" + "=" * 60)
    print("DROPTAIL vs RED QUEUE ALGORITHM COMPARISON")
    print("=" * 60)

    # Extract 1Gb scenarios for DropTail and RED
    dt_1gb = None
    red_1gb = None

    for scenario, metrics in results.items():
        if '1Gb_DropTail' in scenario:
            dt_1gb = (scenario, metrics, raw_metrics[scenario])
        elif '1Gb_RED' in scenario:
            red_1gb = (scenario, metrics, raw_metrics[scenario])

    if dt_1gb and red_1gb:
        dt_name, dt_norm, dt_raw = dt_1gb
        red_name, red_norm, red_raw = red_1gb

        print(f"\nComparison at 1Gb/s bottleneck:")
        print(f"{'Metric':<20} {'DropTail':<12} {'RED':<12} {'Difference':<15} {'Winner':<10}")
        print("-" * 70)

        metrics_comparison = [
            ('Goodput (Mbps)', dt_raw['Avg_Goodput'], red_raw['Avg_Goodput'], 'higher'),
            ('Loss Rate (%)', dt_raw['Avg_LossRate'], red_raw['Avg_LossRate'], 'lower'),
            ('Fairness Index', dt_raw['Jain_Fairness'], red_raw['Jain_Fairness'], 'higher'),
            ('Stability CoV', dt_raw['Stability_CoV'], red_raw['Stability_CoV'], 'lower')
        ]

        for name, dt_val, red_val, better in metrics_comparison:
            diff = red_val - dt_val
            abs_diff = abs(diff)
            if better == 'higher':
                winner = 'RED' if diff > 0 else 'DropTail'
                diff_symbol = '+' if diff > 0 else ''
            else:
                winner = 'RED' if diff < 0 else 'DropTail'
                diff_symbol = '+' if diff < 0 else ''

            print(f"{name:<20} {dt_val:<12.4f} {red_val:<12.4f} {diff_symbol}{diff:<14.4f} {winner:<10}")

        # Calculate overall scores
        dt_overall = (dt_norm['Goodput_Score'] + dt_norm['LossRate_Score'] +
                      dt_norm['Fairness_Score'] + dt_norm['Stability_Score']) / 4
        red_overall = (red_norm['Goodput_Score'] + red_norm['LossRate_Score'] +
                       red_norm['Fairness_Score'] + red_norm['Stability_Score']) / 4

        print("-" * 70)
        print(
            f"{'Overall Score':<20} {dt_overall:<12.2f} {red_overall:<12.2f} {red_overall - dt_overall:<15.2f} {'RED' if red_overall > dt_overall else 'DropTail':<10}")

        # Interpretation
        print(f"\nINTERPRETATION:")
        print(f"• RED shows {'better' if red_overall > dt_overall else 'worse'} overall performance")
        print(f"• Key differences:")

        if abs(red_raw['Avg_Goodput'] - dt_raw['Avg_Goodput']) > 10:
            print(f"  - Significant goodput difference: {abs(red_raw['Avg_Goodput'] - dt_raw['Avg_Goodput']):.2f} Mbps")
        if abs(red_raw['Avg_LossRate'] - dt_raw['Avg_LossRate']) > 1:
            print(f"  - Significant loss rate difference: {abs(red_raw['Avg_LossRate'] - dt_raw['Avg_LossRate']):.2f}%")
        if abs(red_raw['Jain_Fairness'] - dt_raw['Jain_Fairness']) > 0.1:
            print(f"  - Noticeable fairness difference: {abs(red_raw['Jain_Fairness'] - dt_raw['Jain_Fairness']):.3f}")

    return dt_1gb, red_1gb


def analyze_capacity_sensitivity(results):
    """
    Analyze performance sensitivity to different link capacities
    """
    print("\n" + "=" * 50)
    print("CAPACITY SENSITIVITY ANALYSIS")
    print("=" * 50)

    # Compare 1Gb vs 2Gb in DropTail
    dt_1gb = None
    dt_2gb = None

    for scenario, metrics in results.items():
        if '1Gb_DropTail' in scenario:
            dt_1gb = metrics
        elif '2Gb_DropTail' in scenario:
            dt_2gb = metrics

    if dt_1gb and dt_2gb:
        print("DropTail queue behavior at different capacities:")
        print(f"• 1Gb/s: Goodput = {dt_1gb['Avg_Goodput']:.2f} Mbps, Loss = {dt_1gb['Avg_LossRate']:.2f}%")
        print(f"• 2Gb/s: Goodput = {dt_2gb['Avg_Goodput']:.2f} Mbps, Loss = {dt_2gb['Avg_LossRate']:.2f}%")

        goodput_improvement = ((dt_2gb['Avg_Goodput'] - dt_1gb['Avg_Goodput']) / dt_1gb['Avg_Goodput']) * 100
        loss_improvement = ((dt_1gb['Avg_LossRate'] - dt_2gb['Avg_LossRate']) / dt_1gb['Avg_LossRate']) * 100

        print(f"\nPerformance changes from 1Gb/s to 2Gb/s:")
        print(f"• Goodput improvement: {goodput_improvement:+.1f}%")
        print(f"• Loss rate improvement: {loss_improvement:+.1f}%")

        # Interpretation
        print(f"\nSENSITIVITY INTERPRETATION:")
        if goodput_improvement > 50:
            print("• HIGH sensitivity: Doubling capacity significantly improves goodput")
        elif goodput_improvement > 20:
            print("• MEDIUM sensitivity: Noticeable goodput improvement with capacity increase")
        else:
            print("• LOW sensitivity: Limited goodput gains from capacity increase")

        if loss_improvement > 50:
            print("• HIGH loss reduction: Significant congestion relief with higher capacity")
        elif loss_improvement > 20:
            print("• MEDIUM loss reduction: Noticeable congestion improvement")
        else:
            print("• LOW loss reduction: Limited congestion relief from capacity increase")

    return dt_1gb, dt_2gb


def generate_queue_algorithm_plot(results, colors):
    """
    Generate specialized plot focusing on queue algorithm comparison
    """
    # Filter only 1Gb scenarios for queue algorithm comparison
    scenarios_1gb = [s for s in results.keys() if '1Gb' in s]

    if len(scenarios_1gb) < 2:
        print("Not enough 1Gb scenarios for queue algorithm comparison")
        return

    metrics = ['Goodput', 'Loss Rate', 'Fairness', 'Stability']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Raw metric comparison
    metric_data_raw = {
        'Goodput (Mbps)': [results[s]['Avg_Goodput'] for s in scenarios_1gb],
        'Loss Rate (%)': [results[s]['Avg_LossRate'] for s in scenarios_1gb],
        'Fairness': [results[s]['Jain_Fairness'] for s in scenarios_1gb],
        'Stability CoV': [results[s]['Stability_CoV'] for s in scenarios_1gb]
    }

    x_pos = np.arange(len(metrics))
    bar_width = 0.35

    for i, scenario in enumerate(scenarios_1gb):
        values = [metric_data_raw[metric][i] for metric in metric_data_raw.keys()]
        ax1.bar(x_pos + i * bar_width, values, bar_width,
                label=scenario, color=colors[i], alpha=0.8, edgecolor='black')

    ax1.set_xlabel('Performance Metrics', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Raw Values', fontsize=12, fontweight='bold')
    ax1.set_title('Queue Algorithm Comparison - Raw Metrics\n(1Gb/s bottleneck)',
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos + bar_width / 2)
    ax1.set_xticklabels(['Goodput\n(Mbps)', 'Loss Rate\n(%)', 'Fairness\nIndex', 'Stability\nCoV'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Normalized scores comparison
    metric_data_norm = {
        'Goodput': [results[s]['Goodput_Score'] for s in scenarios_1gb],
        'Loss Rate': [results[s]['LossRate_Score'] for s in scenarios_1gb],
        'Fairness': [results[s]['Fairness_Score'] for s in scenarios_1gb],
        'Stability': [results[s]['Stability_Score'] for s in scenarios_1gb]
    }

    for i, scenario in enumerate(scenarios_1gb):
        values = [metric_data_norm[metric][i] for metric in metric_data_norm.keys()]
        ax2.bar(x_pos + i * bar_width, values, bar_width,
                label=scenario, color=colors[i], alpha=0.8, edgecolor='black')

    ax2.set_xlabel('Performance Metrics', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Normalized Score (1-5)', fontsize=12, fontweight='bold')
    ax2.set_title('Queue Algorithm Comparison - Normalized Scores\nHigher scores are better',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos + bar_width / 2)
    ax2.set_xticklabels(metric_data_norm.keys())
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 5.5)

    plt.tight_layout()
    plt.savefig('queue_algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_analysis_report(results, raw_metrics):
    """
    Generate detailed analysis report
    """
    print("\n" + "=" * 80)
    print("CUBIC TCP ALGORITHM - COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Find best performing scenario for each metric
    scenarios = list(results.keys())

    best_goodput = max(scenarios, key=lambda x: results[x]['Goodput_Score'])
    best_loss = max(scenarios, key=lambda x: results[x]['LossRate_Score'])
    best_fairness = max(scenarios, key=lambda x: results[x]['Fairness_Score'])
    best_stability = max(scenarios, key=lambda x: results[x]['Stability_Score'])

    print("\nBEST PERFORMING SCENARIOS (Normalized Scores):")
    print(f"• Highest Goodput: {best_goodput} (Score: {results[best_goodput]['Goodput_Score']:.2f})")
    print(f"• Best Loss Rate: {best_loss} (Score: {results[best_loss]['LossRate_Score']:.2f})")
    print(f"• Best Fairness: {best_fairness} (Score: {results[best_fairness]['Fairness_Score']:.2f})")
    print(f"• Best Stability: {best_stability} (Score: {results[best_stability]['Stability_Score']:.2f})")

    # Overall best scenario
    overall_scores = {}
    for scenario in scenarios:
        overall = (results[scenario]['Goodput_Score'] + results[scenario]['LossRate_Score'] +
                   results[scenario]['Fairness_Score'] + results[scenario]['Stability_Score']) / 4
        overall_scores[scenario] = overall

    best_overall = max(overall_scores, key=overall_scores.get)

    print(f"\n• OVERALL BEST: {best_overall} (Average Score: {overall_scores[best_overall]:.2f})")

    print("\nSCORING EXPLANATION:")
    print("• All metrics normalized to 1-5 scale (1=worst, 5=best)")
    print("• For Goodput/Fairness: Higher raw values → Higher scores")
    print("• For Loss Rate/CoV: Lower raw values → Higher scores")
    print("• Scores allow direct comparison across different measurement units")


if __name__ == "__main__":
    # Analyze Cubic algorithm in different environments
    results, colors, raw_metrics = analyze_cubic_environments()

    if results:
        # Generate comparison table
        df = generate_comparison_table(results)

        # Generate comparison plots with normalized scores
        generate_comparison_plots(results, colors)

        # Generate radar chart
        generate_radar_chart(results, colors)

        # Generate queue algorithm comparison plot
        generate_queue_algorithm_plot(results, colors)

        # New analysis functions
        generate_queue_comparison_analysis(results, raw_metrics)
        analyze_capacity_sensitivity(results)
        generate_analysis_report(results, raw_metrics)

        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE")
        print("=" * 50)
        print("Generated files:")
        print("• cubic_performance_comparison_normalized.csv - Performance data with normalized scores")
        print("• cubic_environment_comparison_normalized.png - Main normalized comparison chart")
        print("• cubic_radar_chart_normalized.png - Radar chart visualization")
        print("• queue_algorithm_comparison.png - Specialized queue algorithm comparison")
    else:
        print("No valid trace files found for analysis.")