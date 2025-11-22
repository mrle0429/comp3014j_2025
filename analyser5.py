#!/usr/bin/env python3
"""
analyser_fixed.py

Robust analyser for NS-2 TCP trace files.
- Parses ns-2 style traces (events like +, r, d) and extracts per-flow metrics.
- Computes per-run: average goodput (Mbps), packet loss rate (%), Jain fairness (late window),
  throughput stability (CoV using 0.1s bins).
- Supports multiple TCP algorithms and 3 environments: 2Gb DropTail, 1Gb DropTail, 1Gb RED.
- Produces normalized comparison plot and CSV summary.

Usage:
    python3 analyser_fixed.py

Make sure trace files are present with the names configured in tcp_algorithms below.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Configuration: modify names if your trace filenames differ
# ---------------------------
tcp_algorithms = {
    # If your trace files are named differently, edit the filenames here
    'Cubic': {
        'runs': ['cubicTrace_2Gb.tr', 'cubicTrace_DropTail.tr', 'cubicTrace_RED.tr']
    },
    'Reno': {
        'runs': ['renoTrace_2Gb.tr', 'renoTrace_DropTail.tr', 'renoTrace_RED.tr']
    },
    # If you actually have "yeahTrace" use name "Yeah"
    'Yeah': {
        'runs': ['yeahTrace_2Gb.tr', 'yeahTrace_DropTail.tr', 'yeahTrace_RED.tr']
    },
    'Vegas': {
        'runs': ['vegasTrace_2Gb.tr', 'vegasTrace_DropTail.tr', 'vegasTrace_RED.tr']
    }
}

# Flow identifiers in your trace: (sender_node_id, receiver_node_id)
# These are the values used in the user's original script: ('0','4') and ('1','5').
FLOW_IDS = [('0', '4'), ('1', '5')]

# Time bin resolution for throughput/COV (seconds)
TIME_BIN = 0.1

# Late-window fraction for fairness (use last third by default)
LATE_WINDOW_FRACTION = 1.0 / 3.0

# ---------------------------
# Trace parsing utilities
# ---------------------------

def try_parse_float(s, default=0.0):
    try:
        return float(s)
    except:
        return default

def parse_trace_for_flow(file_path, flow_src, flow_dst_final):
    """
    Parse a single trace file and return a dataframe of relevant packet events for that flow.
    The trace is assumed to have typical ns-2 format fields where:
      fields[0] = event (e.g., '+', 'r', 'd', ...)
      fields[1] = time (float)
      fields[2] = from_node
      fields[3] = to_node
      fields[4] = pkt_type (e.g., TCP)
      fields[5] = pkt_size (bytes)
      ... there may be more columns
    We return:
      - df: DataFrame with columns ['event','time','from','to','type','size']
      - counters: dict with send_count, recv_count, drop_count
    """
    data = []
    send_count = 0
    recv_count = 0
    drop_count = 0

    try:
        with open(file_path, 'r') as f:
            for line in f:
                fields = line.strip().split()
                if len(fields) < 6:
                    continue
                event = fields[0]
                time = try_parse_float(fields[1], default=None)
                fr = fields[2]
                to = fields[3]
                ptype = fields[4]
                # size may not be integer in some traces; try-except
                try:
                    size = int(fields[5])
                except:
                    size = None

                # filter by flow src/dst (simple node id matching)
                # we count events where packet flows from flow_src or reaches flow_dst_final
                # it is intentionally tolerant to field variations
                if (fr == flow_src) or (to == flow_dst_final) or (fr == flow_src and to == flow_dst_final):
                    # we keep non-empty size rows; but we still record event even if size missing
                    rec = {
                        'event': event,
                        'time': time if time is not None else 0.0,
                        'from': fr,
                        'to': to,
                        'type': ptype,
                        'size': size if size is not None else 0
                    }
                    data.append(rec)
                    # count logically: '+' = send, 'r' = receive, 'd' = drop
                    if event == '+':
                        send_count += 1
                    elif event == 'r':
                        recv_count += 1
                    elif event == 'd':
                        # drop may be in different node; some drops will appear with from==node where queue overflows
                        drop_count += 1
    except FileNotFoundError:
        return pd.DataFrame(columns=['event','time','from','to','type','size']), {'send':0,'recv':0,'drop':0}
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return pd.DataFrame(columns=['event','time','from','to','type','size']), {'send':0,'recv':0,'drop':0}

    df = pd.DataFrame(data, columns=['event','time','from','to','type','size'])
    counters = {'send': send_count, 'recv': recv_count, 'drop': drop_count}
    return df, counters

# ---------------------------
# Metrics computation
# ---------------------------

def compute_goodput_mbps_from_recv_df(recv_df):
    """
    Compute goodput in Mbps using received packets dataframe (rows where event == 'r').
    goodput = total bits received / (time_span)
    If time_span <= 0 -> return 0.0
    """
    if recv_df.empty:
        return 0.0
    total_bits = (recv_df['size'].sum() * 8.0)
    tmin = recv_df['time'].min()
    tmax = recv_df['time'].max()
    time_span = tmax - tmin
    if time_span <= 0:
        return 0.0
    goodput_mbps = total_bits / time_span / 1e6
    return goodput_mbps

def compute_loss_rate(counters):
    """
    Compute loss rate in percent.
    Prefer explicit drop count if present; otherwise fall back to (send - recv)/send.
    """
    sends = counters.get('send', 0)
    recvs = counters.get('recv', 0)
    drops = counters.get('drop', 0)

    if sends <= 0:
        return 0.0

    if drops > 0:
        loss_pct = drops / sends * 100.0
    else:
        loss_pct = max(0.0, (sends - recvs) / sends * 100.0)
    return loss_pct

def compute_throughput_timeseries_bits(df, bin_size=TIME_BIN):
    """
    Aggregate received bytes per bin (bits) using bin_size resolution (seconds).
    Returns:
      - times (bin centers)
      - bits_per_bin (bits)
    Uses only receive ('r') events.
    """
    if df.empty:
        return np.array([]), np.array([])

    recv_df = df[df['event'] == 'r'].copy()
    if recv_df.empty:
        return np.array([]), np.array([])

    times = recv_df['time'].values
    sizes = recv_df['size'].values  # bytes

    # compute bin indices
    min_t = times.min()
    max_t = times.max()
    if max_t <= min_t:
        return np.array([]), np.array([])

    # bins from min_t to max_t inclusive
    nbins = int(math.ceil((max_t - min_t) / bin_size))
    bins = np.linspace(min_t, min_t + nbins * bin_size, nbins + 1)
    inds = np.digitize(times, bins) - 1  # bin index
    bits_per_bin = np.zeros(nbins, dtype=float)
    for idx, sz in zip(inds, sizes):
        if 0 <= idx < nbins:
            bits_per_bin[idx] += sz * 8.0
    # bin centers
    centers = bins[:-1] + bin_size / 2.0
    return centers, bits_per_bin

def compute_cov_from_bits_series(bits_per_bin):
    """
    Compute coefficient of variation (std/mean) for bits-per-bin converted to Mbps.
    If mean == 0 -> return 0.0
    """
    if len(bits_per_bin) == 0:
        return 0.0
    mbps = bits_per_bin / (TIME_BIN * 1e6)  # convert bits per bin to Mbps
    mean = mbps.mean()
    std = mbps.std(ddof=0)
    if mean == 0:
        return 0.0
    return std / mean

def compute_jain_fairness_from_goodputs(glist):
    """
    Compute Jain's fairness index given list of throughputs (non-negative numbers).
    """
    x = np.array(glist, dtype=float)
    if np.all(x == 0):
        return 0.0
    numerator = (x.sum()) ** 2
    denominator = len(x) * (x ** 2).sum()
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)

# ---------------------------
# High level runner
# ---------------------------

def get_environment_info_from_filename(filename):
    """Map filename token to environment label used in plots."""
    fname = filename.lower()
    if '2gb' in fname:
        return '2Gb\nDropTail'
    if 'red' in fname:
        return '1Gb\nRED'
    if 'droptail' in fname:
        return '1Gb\nDropTail'
    return 'Unknown'

def analyze_multiple_tcp_algorithms(tcp_algorithms_map):
    """
    Main analysis driver.
    Returns structured results.
    """
    results = {}

    for algo_name, algo_info in tcp_algorithms_map.items():
        runs = algo_info.get('runs', [])
        algo_runs = []
        print(f"\n=== Analyzing {algo_name} ===")
        for run_file in runs:
            if not os.path.exists(run_file):
                print(f"  [WARN] missing file: {run_file}, skipping")
                continue
            env_label = get_environment_info_from_filename(run_file)
            # compute per-flow metrics
            per_flow_metrics = []
            flow_goodputs = []
            # each flow processed with parse_trace_for_flow
            for (src, dst) in FLOW_IDS:
                df_flow, counters = parse_trace_for_flow(run_file, src, dst)
                # compute goodput using only recv events
                recv_df = df_flow[df_flow['event'] == 'r']
                goodput = compute_goodput_mbps_from_recv_df(recv_df)
                loss = compute_loss_rate(counters)
                # throughput timeseries & cov
                _, bits_per_bin = compute_throughput_timeseries_bits(df_flow, bin_size=TIME_BIN)
                cov = compute_cov_from_bits_series(bits_per_bin)
                per_flow_metrics.append({
                    'src': src, 'dst': dst,
                    'goodput_mbps': goodput,
                    'loss_pct': loss,
                    'cov': cov,
                    'counters': counters
                })
                flow_goodputs.append(goodput)

            # average metrics across flows
            avg_goodput = np.mean([m['goodput_mbps'] for m in per_flow_metrics]) if per_flow_metrics else 0.0
            avg_loss = np.mean([m['loss_pct'] for m in per_flow_metrics]) if per_flow_metrics else 0.0
            avg_cov = np.mean([m['cov'] for m in per_flow_metrics]) if per_flow_metrics else 0.0

            # fairness: compute using last fraction of global simulation time (using received events across both flows)
            # build combined recv times to find T_max for this run
            combined_recv_times = []
            for (src, dst) in FLOW_IDS:
                df_flow, _ = parse_trace_for_flow(run_file, src, dst)
                rtimes = df_flow[df_flow['event'] == 'r']['time'].values
                if len(rtimes):
                    combined_recv_times.extend(rtimes.tolist())
            if len(combined_recv_times) == 0:
                fairness = 0.0
            else:
                T_max = max(combined_recv_times)
                start_window = T_max * (1.0 - LATE_WINDOW_FRACTION)  # e.g., last 1/3
                late_goodputs = []
                for (src, dst) in FLOW_IDS:
                    df_flow, _ = parse_trace_for_flow(run_file, src, dst)
                    late_recv = df_flow[(df_flow['event'] == 'r') & (df_flow['time'] >= start_window)]
                    # compute goodput over that late window (duration = T_max - start_window)
                    duration = T_max - start_window
                    if late_recv.empty or duration <= 0:
                        late_goodputs.append(0.0)
                    else:
                        bits = late_recv['size'].sum() * 8.0
                        late_goodputs.append(bits / duration / 1e6)
                fairness = compute_jain_fairness_from_goodputs(late_goodputs)

            run_summary = {
                'Run_File': run_file,
                'Environment': env_label,
                'Avg_Goodput': avg_goodput,
                'Avg_LossRate': avg_loss,
                'Jain_Fairness': fairness,
                'Stability_CoV': avg_cov,
                'Per_Flow': per_flow_metrics
            }
            algo_runs.append(run_summary)
            print(f"  {run_file} | Env: {env_label} | Goodput: {avg_goodput:.3f} Mbps | Loss: {avg_loss:.3f}% | Fairness: {fairness:.3f} | CoV: {avg_cov:.3f}")

        results[algo_name] = {
            'runs': algo_runs
        }

    return results

# ---------------------------
# Normalization and plotting
# ---------------------------

def normalize_metrics_across_runs(results):
    """
    For plotting, compute normalized scores with improved scaling to avoid 0/1 extremes.
    Returns:
        original_data: dict of lists for each metric
        normalized_data: dict of lists for each metric
        labels: list of labels corresponding to positions
    """
    metrics = ['Goodput', 'LossRate', 'Fairness', 'Stability']
    original = {m: [] for m in metrics}
    labels = []
    envs = []

    for algo, info in results.items():
        for run in info['runs']:
            original['Goodput'].append(run['Avg_Goodput'])
            original['LossRate'].append(run['Avg_LossRate'])
            original['Fairness'].append(run['Jain_Fairness'])
            original['Stability'].append(run['Stability_CoV'])
            labels.append(f"{algo}\n{run['Environment']}")
            envs.append(run['Environment'])

    # Improved normalization to avoid 0 and 1 extremes
    normalized = {}

    # Method 1: Add padding to range (recommended)
    padding_factor = 0.1  # 10% padding on both sides

    for m in ['Goodput', 'Fairness']:
        vals = original[m]
        if len(vals) == 0:
            normalized[m] = []
            continue
        vmin = min(vals)
        vmax = max(vals)
        range_val = vmax - vmin

        if range_val == 0:
            normalized[m] = [0.5 for _ in vals]
        else:
            # Add padding: extend range by padding_factor on both sides
            padded_min = vmin - padding_factor * range_val
            padded_max = vmax + padding_factor * range_val
            padded_range = padded_max - padded_min
            normalized[m] = [(v - padded_min) / padded_range for v in vals]

    for m in ['LossRate', 'Stability']:
        vals = original[m]
        if len(vals) == 0:
            normalized[m] = []
            continue
        vmin = min(vals)
        vmax = max(vals)
        range_val = vmax - vmin

        if range_val == 0:
            normalized[m] = [0.5 for _ in vals]
        else:
            # Add padding and invert for lower-better metrics
            padded_min = vmin - padding_factor * range_val
            padded_max = vmax + padding_factor * range_val
            padded_range = padded_max - padded_min
            normalized[m] = [1.0 - (v - padded_min) / padded_range for v in vals]

    return original, normalized, labels, envs


def plot_normalized_comparison(original, normalized, labels, envs,
                               out_png='tcp_algorithms_normalized_comparison_fixed.png'):
    """
    Fixed version with horizontal bars and group separators
    """
    metrics = ['Goodput', 'LossRate', 'Fairness', 'Stability']
    N = len(labels)
    if N == 0:
        print("[WARN] No runs to plot.")
        return

    y = np.arange(N)
    height = 0.18  # bar height for horizontal bars

    # Create horizontal figure (better for document insertion)
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 12))  # Wider, shorter figure

    # Horizontal bars: offsets for different metrics
    offsets = np.linspace(-1.5 * height, 1.5 * height, len(metrics))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Distinct colors

    # Plot horizontal bars
    for i, m in enumerate(metrics):
        vals = normalized[m]
        ax1.barh(y + offsets[i], vals, height, label=m, color=colors[i], alpha=0.7)

    # Set y-axis labels (algorithm + environment)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel('Normalized score (0..1)')
    ax1.set_title('Normalized comparison across TCP algorithms and environments')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_xlim(0, 1.1)  # Add some space for annotations

    # Add group separators (dashed lines between algorithm groups)
    # Each algorithm has 3 runs (3 environments)
    group_boundaries = [2.5, 5.5, 8.5]  # Boundaries between Cubic|Reno|Yeah|Vegas
    for boundary in group_boundaries:
        ax1.axhline(y=boundary, color='gray', linestyle='--', alpha=0.7, linewidth=1)

    # FIXED: Proper annotations with correct formatting
    for i, m in enumerate(metrics):
        for j, val in enumerate(normalized[m]):
            orig = original[m][j]

            # Format the annotation text based on metric type
            if m == 'Goodput':
                text = f"{orig:.3f} Mbps"
                ha = 'left'
                x_offset = 0.02
            elif m == 'LossRate':
                text = f"{orig:.2f}%"
                ha = 'left'
                x_offset = 0.02
            elif m == 'Fairness':
                text = f"{orig:.3f}"
                ha = 'left'
                x_offset = 0.02
            else:  # Stability (CoV)
                text = f"{orig:.3f}"
                ha = 'left'
                x_offset = 0.02

            # Only annotate if there's space
            if val + x_offset < 1.0:
                ax1.text(val + x_offset, j + offsets[i], text,
                         ha=ha, va='center', fontsize=7, rotation=0)
            else:
                # Place inside the bar if no space to the right
                ax1.text(val - 0.05, j + offsets[i], text,
                         ha='right', va='center', fontsize=7, rotation=0,
                         color='white', weight='bold')

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved fixed plot to {out_png}")
    plt.show()

# ---------------------------
# Summary and CSV saving
# ---------------------------

def save_results_to_csv(results, fname='tcp_algorithms_detailed_results.csv'):
    rows = []
    for algo, info in results.items():
        for run in info['runs']:
            row = {
                'Algorithm': algo,
                'Run_File': run['Run_File'],
                'Environment': run['Environment'],
                'Avg_Goodput_Mbps': run['Avg_Goodput'],
                'Avg_LossRate_pct': run['Avg_LossRate'],
                'Jain_Fairness': run['Jain_Fairness'],
                'Stability_CoV': run['Stability_CoV']
            }
            # per-flow details appended
            for i, pf in enumerate(run['Per_Flow'], start=1):
                row[f'Flow{i}_src'] = pf.get('src')
                row[f'Flow{i}_dst'] = pf.get('dst')
                row[f'Flow{i}_Goodput_Mbps'] = pf.get('goodput_mbps')
                row[f'Flow{i}_Loss_pct'] = pf.get('loss_pct')
                row[f'Flow{i}_CoV'] = pf.get('cov')
            rows.append(row)
    if len(rows) == 0:
        print("[WARN] No data to save.")
        return None
    df = pd.DataFrame(rows)
    df.to_csv(fname, index=False)
    print(f"[INFO] Saved CSV to {fname}")
    return df

def print_summary_statistics(results):
    print("\n" + "="*60)
    print("SUMMARY STATISTICS PER ALGORITHM")
    print("="*60)
    metrics = ['Avg_Goodput', 'Avg_LossRate', 'Jain_Fairness', 'Stability_CoV']
    for algo, info in results.items():
        runs = info['runs']
        if not runs:
            continue
        print(f"\nAlgorithm: {algo}")
        for m in metrics:
            vals = [r[m] for r in runs]
            mean = np.mean(vals) if len(vals) else 0.0
            std = np.std(vals, ddof=0) if len(vals) else 0.0
            unit = 'Mbps' if m == 'Avg_Goodput' else '%' if m == 'Avg_LossRate' else ''
            print(f"  {m}: {mean:.3f} ± {std:.3f} {unit}")

# ---------------------------
# Main
# ---------------------------

def main():
    results = analyze_multiple_tcp_algorithms(tcp_algorithms)
    # if no results, exit
    any_runs = any(len(info['runs']) > 0 for info in results.values())
    if not any_runs:
        print("[ERROR] No runs found. Please check trace filenames and directory.")
        return

    original, normalized, labels, envs = normalize_metrics_across_runs(results)
    plot_normalized_comparison(original, normalized, labels, envs)
    save_results_to_csv(results)
    print_summary_statistics(results)
    print("\n[INFO] Analysis complete.")

if __name__ == '__main__':
    main()
