import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def parse_trace(file_path, flow_identifier):
    """
    Parse the trace file and extract the throughput, packet loss rate and time series data of the specified stream. 
    """
    data = []
    # Counters for sent and received packets
    send_count = 0
    recv_count = 0
    # Unpack the source and destination node IDs for the target flow
    flow_src, flow_dst_final = flow_identifier
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                fields = line.strip().split()
                if len(fields) < 6:
                    continue
                # Extract key information from the trace line
                event = fields[0]
                src = fields[2]
                dst = fields[3]
                proto = fields[4]
                # Filter for TCP packets and only 'send' or 'receive' events
                if event not in ['+', 'r'] or proto.lower() != 'tcp':
                    continue
                try:
                    size = int(fields[5])
                except ValueError:
                    continue
                # Filter out small packets
                if size < 1000:
                    continue
                # Extract timestamp, defaulting to 0 if invalid
                try:
                    time = float(fields[1])
                except ValueError:
                    time = 0.0
                # Record send events if they match the flow's source node
                if event == '+' and src == flow_src:
                    send_count += 1
                    data.append({
                        'event': event,
                        'time': time,
                        'size': size,
                        'src': src,
                        'dst': dst
                    })
                # Record receive events if they match the flow's destination node
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
        # Calculate goodput (Mbps) if there are receive events
        if not recv_df.empty:
            total_bits = recv_df['size'].sum() * 8
            total_time = recv_df['time'].max() - recv_df['time'].min()
            goodput = total_bits / total_time * 1e-6 if total_time > 0 else 0.0
        else:
            goodput = 0.0
        # Calculate packet loss rate (%), avoiding division by zero
        loss_rate = ((send_count - recv_count) / send_count) * 100 if send_count > 0 else 0.0
        return goodput, loss_rate, df
    except FileNotFoundError:
        print(f"no file: {file_path}")
        return 0.0, 0.0, pd.DataFrame()
    except Exception as e:
        print(f"Error occurred while parsing {file_path}：{str(e)}")
        return 0.0, 0.0, pd.DataFrame()


def generate_table_and_plot():
    tcp_algos = ['cubic', 'reno', 'yeah', 'vegas']
    # per-flow：n1->n5 (0->4), n2->n6 (1->5)
    flow_id_1 = ('0', '4')  # Flow 1: n1 -> n5
    flow_id_2 = ('1', '5')  # Flow 2: n2 -> n6
    algo_list = []
    f1_goodputs, f1_losses = [], []
    f2_goodputs, f2_losses = [], []
    avg_goodputs, avg_losses = [], []
    for algo in tcp_algos:
        file = f'{algo}Trace.tr'
        g1, l1, _ = parse_trace(file, flow_id_1)
        g2, l2, _ = parse_trace(file, flow_id_2)
        # Calculate average metrics across the two flows
        avg_g = (g1 + g2) / 2.0
        avg_l = (l1 + l2) / 2.0
        algo_list.append(algo)
        f1_goodputs.append(g1)
        f1_losses.append(l1)
        f2_goodputs.append(g2)
        f2_losses.append(l2)
        avg_goodputs.append(avg_g)
        avg_losses.append(avg_l)
    table_data = {
        'TCP Algorithm': algo_list,
        'Flow1 Goodput (Mbps)': f1_goodputs,
        'Flow1 Packet Loss Rate (%)': f1_losses,
        'Flow2 Goodput (Mbps)': f2_goodputs,
        'Flow2 Packet Loss Rate (%)': f2_losses,
        'Avg Goodput (Mbps)': avg_goodputs,
        'Avg Packet Loss Rate (%)': avg_losses,
    }
    # Create and print the summary table
    df_table = pd.DataFrame(table_data)
    print("=== Throughput and Packet Loss Rate Table (per-flow & average) ===")
    print(df_table)
    df_table.to_csv('goodput_loss_table.csv', index=False)
    # Create and save plots for average goodput and loss rate
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    ax1.bar(tcp_algos, avg_goodputs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_ylabel('Avg Goodput (Mbps)')
    ax1.set_title('TCP Algorithm vs Average Goodput (2 flows)')
    ax2.bar(tcp_algos, avg_losses, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_xlabel('TCP Algorithm')
    ax2.set_ylabel('Avg Packet Loss Rate (%)')
    ax2.set_title('TCP Algorithm vs Average Packet Loss Rate (2 flows)')
    plt.tight_layout()
    plt.savefig('goodput_loss_comparison.png')
    plt.show()
    # Create and save plots for per-flow goodput and loss rate
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    x = np.arange(len(tcp_algos))
    width = 0.35
    ax1.bar(x - width/2, f1_goodputs, width, label='Flow1 (n1→n5)', color='#1f77b4')
    ax1.bar(x + width/2, f2_goodputs, width, label='Flow2 (n2→n6)', color='#ff7f0e')
    ax1.set_ylabel('Goodput (Mbps)')
    ax1.set_title('TCP Algorithm vs Per-flow Goodput')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tcp_algos)
    ax1.legend()
    ax2.bar(x - width/2, f1_losses, width, label='Flow1 (n1→n5)', color='#1f77b4')
    ax2.bar(x + width/2, f2_losses, width, label='Flow2 (n2→n6)', color='#ff7f0e')
    ax2.set_xlabel('TCP Algorithm')
    ax2.set_ylabel('Packet Loss Rate (%)')
    ax2.set_title('TCP Algorithm vs Per-flow Packet Loss Rate')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tcp_algos)
    ax2.legend()
    plt.tight_layout()
    plt.savefig('per_flow_goodput_loss_comparison.png')
    plt.show()

    return avg_goodputs, avg_losses


def calculate_jain_fairness():
    tcp_algos = ['cubic', 'reno', 'yeah', 'vegas']
    flow_id_1 = ('0', '4')  
    flow_id_2 = ('1', '5')
    jain_indices = []      
    per_algo_flows = []
    print("\n=== Jain Fairness Index over last 1/3 (per algorithm, 2 flows) ===")
    for algo in tcp_algos:
        file = f'{algo}Trace.tr'
        _, _, df1 = parse_trace(file, flow_id_1)
        _, _, df2 = parse_trace(file, flow_id_2)
        flow_throughputs = []
        # Calculate throughput for each flow in the last third of the simulation
        for df in (df1, df2):
            if df.empty:
                flow_throughputs.append(0.0)
                continue
            # Determine the total simulation time
            T = df['time'].max()
            if T <= 0:
                flow_throughputs.append(0.0)
                continue
            start = 2 * T / 3
            # Filter events to those received in the last third
            late_df = df[(df['event'] == 'r') & (df['time'] >= start)]
            total_bits = late_df['size'].sum() * 8
            late_goodput = total_bits / (T / 3) * 1e-6 if T > 0 else 0.0
            flow_throughputs.append(late_goodput)
        # Calculate Jain's Fairness Index
        if sum(x * x for x in flow_throughputs) > 0:
            numerator = (sum(flow_throughputs)) ** 2
            denominator = len(flow_throughputs) * sum(x**2 for x in flow_throughputs)
            jain = numerator / denominator
        else:
            jain = 0.0
        jain_indices.append(jain)
        per_algo_flows.append(flow_throughputs)

        print(f"{algo}: Jain = {jain:.4f}, flows last-1/3 throughputs = {[f'{x:.4f}' for x in flow_throughputs]} Mbps")

    fairest_idx = int(np.argmax(jain_indices)) if jain_indices else 0
    fairest_algo = tcp_algos[fairest_idx]
    print(f"Fairest algorithm overall: {fairest_algo} (Jain = {jain_indices[fairest_idx]:.4f})")

    
    return jain_indices, per_algo_flows, fairest_algo

def calculate_throughput_cov():
    tcp_algos = ['cubic', 'reno', 'yeah', 'vegas']
    flow_id_1 = ('0', '4')
    flow_id_2 = ('1', '5')
    covs = []
    def cov_from_df(df):
        if df.empty:
            return 0.0
        df = df.copy()
        # Group events by the second they occurred
        df['time_second'] = df['time'].astype(int)
        # Calculate throughput (in Mbps) for each second
        recv_per_sec = df[df['event'] == 'r'].groupby('time_second')['size'].sum() * 8 / 1e6
        if recv_per_sec.empty:
            return 0.0
        # Calculate mean and standard deviation of the per-second throughput
        mean = recv_per_sec.mean()
        std = recv_per_sec.std()
        # CoV is std_dev / mean. Avoid division by zero.
        return std / mean if mean != 0 else 0.0
    # Calculate CoV for each algorithm
    for algo in tcp_algos:
        file = f'{algo}Trace.tr'
        _, _, df1 = parse_trace(file, flow_id_1)
        _, _, df2 = parse_trace(file, flow_id_2)
        # Calculate CoV for each flow
        cov1 = cov_from_df(df1)
        cov2 = cov_from_df(df2)
        # Average the CoV across both flows
        cov_avg = (cov1 + cov2) / 2.0
        covs.append(cov_avg)
    # Determine the most stable algorithm
    min_cov_idx = int(np.argmin(covs)) if covs else 0
    most_stable_algo = tcp_algos[min_cov_idx]
    print("\n=== Throughput stability (CoV, averaged over 2 flows) ===")
    print(f"CoV per algorithm: {dict(zip(tcp_algos, [round(x, 4) for x in covs]))}")
    print(f"Most stable algorithm: {most_stable_algo} (CoV={covs[min_cov_idx]:.4f})")
    return covs, most_stable_algo

def get_best_algorithm(goodputs, loss_rates, jain_indices, covs):
    tcp_algos = ['cubic', 'reno', 'yeah', 'vegas']
    scores = []
    for i in range(len(tcp_algos)):
        # Calculate a weighted score.
        score = (
            goodputs[i] * 0.3
            + (100 - loss_rates[i]) * 0.2
            + jain_indices[i] * 0.3
            + (1 - covs[i]) * 0.2
        )
        scores.append(score)
    # Find the algorithm with the highest score
    best_idx = int(np.argmax(scores))
    best_algo = tcp_algos[best_idx]
    best_score = scores[best_idx]
    return best_algo, best_score, best_idx



def generate_detailed_analysis(goodputs, loss_rates, jain_indices, covs, per_algo_flows, fairest_algo, most_stable_algo):
    """Generate detailed performance analysis report in English"""
    tcp_algos = ['cubic', 'reno', 'yeah', 'vegas']
    
    print("\n" + "="*80)
    print("PART A DETAILED ANALYSIS REPORT")
    print("="*80)
    # 1. Basic performance table (averaged over 2 flows)
    print("\nBASIC PERFORMANCE METRICS (averaged over 2 flows):")
    basic_df = pd.DataFrame({
        'Algorithm': tcp_algos,
        'Goodput(Mbps)': [f"{x:.4f}" for x in goodputs],
        'LossRate(%)': [f"{x:.4f}" for x in loss_rates],
        'CoV': [f"{x:.4f}" for x in covs],
        'JainIndex': [f"{x:.4f}" for x in jain_indices],
    })
    print(basic_df.to_string(index=False))
    # 2. Jain's Fairness Index Analysis
    print(f"\n️JAIN'S FAIRNESS INDEX ANALYSIS (Last 1/3 Duration, fairness between 2 flows per algorithm):")
    for algo, jain, flows in zip(tcp_algos, jain_indices, per_algo_flows):
        print(f"  {algo}: Jain = {jain:.4f}, flow throughputs = {[f'{x:.4f}' for x in flows]} Mbps")
    best_jain = jain_indices[tcp_algos.index(fairest_algo)]
    print(f"\n  Fairest Algorithm: {fairest_algo}")
    print(f"  Explanation: A higher Jain index (closer to 1.0) means more equal bandwidth sharing between the two flows.")
    print(f"               {fairest_algo} achieves the highest Jain index ({best_jain:.4f}), so it provides the most balanced sharing.")
    # 3. Stability Analysis
    print(f"\nTHROUGHPUT STABILITY ANALYSIS (Coefficient of Variation, averaged over 2 flows):")
    print(f"  Most Stable Algorithm: {most_stable_algo} (CoV = {covs[tcp_algos.index(most_stable_algo)]:.4f})")
    print(f"\n  Stability Mechanism Explanation:")
    print(f"  • A smaller CoV means the per-second throughput fluctuates less, i.e., the algorithm sends more smoothly over time.")
    print(f"  • The most stable variant reacts more gently to congestion signals, avoiding large oscillations in sending rate.")
    print(f"  • Variants with larger CoV tend to have more aggressive window growth/backoff, which leads to visible throughput oscillations.")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    # Fairness graph: Jain index for each algorithm
    ax1.bar(tcp_algos, jain_indices, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_ylabel('Jain Index')
    ax1.set_title("TCP Algorithm vs Jain's Fairness Index")
    ax2.bar(tcp_algos, covs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_xlabel('TCP Algorithm')
    ax2.set_ylabel('Coefficient of Variation (CoV)')
    ax2.set_title('TCP Algorithm vs Throughput Stability (CoV)')
    
    plt.tight_layout()
    plt.savefig('fairness_stability_comparison.png')
    plt.show()


if __name__ == "__main__":
    # Step 1: Generate table and plot
    goodputs, loss_rates = generate_table_and_plot()
    
    # Step 2: Calculate Jain fairness index
    jain_indices, per_algo_flows, fairest_algo = calculate_jain_fairness()
    # Step 3: Calculate throughput stability (CoV)
    covs, most_stable_algo = calculate_throughput_cov()
    # Step 4: Get the best algorithm based on the calculated metrics
    best_algo, best_score, best_idx = get_best_algorithm(goodputs, loss_rates, jain_indices, covs)
    
    # Step 5: Generate detailed analysis report
    generate_detailed_analysis(goodputs, loss_rates, jain_indices, covs, per_algo_flows, fairest_algo, most_stable_algo) 
# Final conclusion
    print("\n" + "="*50)
    print("FINAL CONCLUSION")
    print("="*50)
    print(f"Under the current network topology, {best_algo.upper()} is the best TCP algorithm (total score: {best_score:.4f}).")
    throughput_rank = sorted(range(len(goodputs)), key=lambda i: goodputs[i], reverse=True).index(best_idx) + 1
    loss_rank = sorted(range(len(loss_rates)), key=lambda i: loss_rates[i]).index(best_idx) + 1
    print(f"\nDetailed Justification:")
    print(f"1. Average Goodput: {goodputs[best_idx]:.4f} Mbps (Rank: {throughput_rank}/4)")
    print(f"2. Average Packet Loss: {loss_rates[best_idx]:.4f}% (Rank: {loss_rank}/4)")
    print(f"3. Fairness: Jain index for this variant = {jain_indices[best_idx]:.4f} (higher means more equal sharing between its two flows).")
    print(f"4. Stability: CoV = {covs[best_idx]:.4f} — lower CoV implies smoother throughput over time.")
    print(f"\nRecommended Scenarios:")
    print(f"  • Real-time apps (video conferencing, VoIP) needing low loss and reasonably high, stable throughput.")
    print(f"  • Mixed TCP traffic environments requiring fair bandwidth sharing between concurrent flows.")
    print(f"  • Latency-sensitive scenarios where avoiding large oscillations in sending rate is important.")
