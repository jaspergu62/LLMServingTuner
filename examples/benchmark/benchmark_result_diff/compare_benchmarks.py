# -*- coding: utf-8 -*-
"""
Benchmark Result Comparison Tool

Compare performance metrics between multiple benchmark results:
- TTFT (Time to First Token)
- TPOT (Time per Output Token): Mean, P90, P99
- Throughput (TPS - Tokens per Second)
- Request completion progress over time

Supports multiple vLLM and Vidur results for comparison.
vLLM results can be JSON or CSV files.

Usage:
    # Compare using CSV files
    python compare_benchmarks.py \
        --result vllm:/path/to/vllm_requests.csv:vLLM-Real \
        --result vidur:/path/to/vidur_output/:Vidur-Sim \
        --output-dir ./comparison_output

    # Compare multiple results (JSON or CSV)
    python compare_benchmarks.py \
        --result vllm:/path/to/result1.json:vLLM-QPS1 \
        --result vllm:/path/to/result2_requests.csv:vLLM-QPS2 \
        --result vidur:/path/to/vidur_qps1/:Vidur-QPS1 \
        --result vidur:/path/to/vidur_qps2/:Vidur-QPS2 \
        --output-dir ./comparison_output
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Color palette for multiple lines
COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
]

LINE_STYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""
    name: str
    source_type: str  # 'vllm' or 'vidur'
    ttft_mean: float
    ttft_p90: float
    ttft_p99: float
    tpot_mean: float
    tpot_p90: float
    tpot_p99: float
    throughput_tps: float  # tokens per second
    request_throughput: float  # requests per second
    total_requests: int
    total_tokens: int

    # Per-request data for plotting
    request_submit_times: Optional[List[float]] = None
    request_finish_times: Optional[List[float]] = None


def load_vllm_benchmark(file_path: str, name: str, csv_path: Optional[str] = None) -> BenchmarkMetrics:
    """
    Load vLLM benchmark results from JSON or CSV file.

    Args:
        file_path: Path to the vLLM benchmark result file (JSON or CSV)
        name: Display name for this benchmark
        csv_path: Optional path to the per-request CSV file (only used with JSON)

    Returns:
        BenchmarkMetrics object with vLLM results
    """
    # Detect file type
    is_csv = file_path.lower().endswith('.csv')

    if is_csv:
        # Load directly from CSV (per-request data)
        return _load_vllm_from_csv(file_path, name)
    else:
        # Load from JSON
        return _load_vllm_from_json(file_path, name, csv_path)


def _load_vllm_from_csv(csv_path: str, name: str) -> BenchmarkMetrics:
    """
    Load vLLM benchmark results from per-request CSV file.

    Expected columns: request_id, prompt_len, output_len, ttft, tpot, latency, submit_time, finish_time
    """
    df = pd.read_csv(csv_path)

    # Calculate metrics from per-request data
    # TTFT in milliseconds (CSV stores in seconds, convert to ms)
    if 'ttft' in df.columns:
        ttft_values = df['ttft'].values * 1000  # s -> ms
        ttft_mean = np.mean(ttft_values)
        ttft_p90 = np.percentile(ttft_values, 90)
        ttft_p99 = np.percentile(ttft_values, 99)
    else:
        ttft_mean = ttft_p90 = ttft_p99 = 0

    # TPOT in milliseconds
    if 'tpot' in df.columns:
        tpot_values = df['tpot'].values * 1000  # s -> ms
        tpot_mean = np.mean(tpot_values)
        tpot_p90 = np.percentile(tpot_values, 90)
        tpot_p99 = np.percentile(tpot_values, 99)
    else:
        tpot_mean = tpot_p90 = tpot_p99 = 0

    # Calculate totals
    total_requests = len(df)
    total_input = df['prompt_len'].sum() if 'prompt_len' in df.columns else 0
    total_output = df['output_len'].sum() if 'output_len' in df.columns else 0
    total_tokens = total_input + total_output

    # Per-request timing data
    submit_times = df['submit_time'].tolist() if 'submit_time' in df.columns else None
    finish_times = df['finish_time'].tolist() if 'finish_time' in df.columns else None

    # Calculate throughput
    if submit_times and finish_times:
        total_time = max(finish_times) - min(submit_times)
        request_throughput = total_requests / total_time if total_time > 0 else 0
        throughput_tps = total_output / total_time if total_time > 0 else 0
    else:
        request_throughput = 0
        throughput_tps = 0

    return BenchmarkMetrics(
        name=name,
        source_type='vllm',
        ttft_mean=ttft_mean,
        ttft_p90=ttft_p90,
        ttft_p99=ttft_p99,
        tpot_mean=tpot_mean,
        tpot_p90=tpot_p90,
        tpot_p99=tpot_p99,
        throughput_tps=throughput_tps,
        request_throughput=request_throughput,
        total_requests=total_requests,
        total_tokens=int(total_tokens),
        request_submit_times=submit_times,
        request_finish_times=finish_times,
    )


def _load_vllm_from_json(json_path: str, name: str, csv_path: Optional[str] = None) -> BenchmarkMetrics:
    """
    Load vLLM benchmark results from JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract metrics from JSON
    # TTFT is in milliseconds in the JSON
    ttft_mean = data.get('mean_ttft_ms', data.get('ttft_mean', 0))
    ttft_p90 = data.get('p90_ttft_ms', data.get('ttft_p90', 0))
    ttft_p99 = data.get('p99_ttft_ms', data.get('ttft_p99', 0))

    # TPOT (inter-token latency) in milliseconds
    tpot_mean = data.get('mean_tpot_ms', data.get('tpot_mean', 0))
    tpot_p90 = data.get('p90_tpot_ms', data.get('tpot_p90', 0))
    tpot_p99 = data.get('p99_tpot_ms', data.get('tpot_p99', 0))

    # Throughput
    throughput_tps = data.get('output_throughput', data.get('throughput', 0))
    request_throughput = data.get('request_throughput', 0)

    # Totals
    total_requests = data.get('completed', data.get('num_prompts', 0))
    total_output = data.get('total_output', data.get('total_output_tokens', 0))
    total_input = data.get('total_input', data.get('total_input_tokens', 0))
    total_tokens = total_output + total_input

    # Per-request data
    submit_times = None
    finish_times = None

    # Try to load from JSON first
    if 'submit_times' in data and 'finish_times' in data:
        submit_times = data['submit_times']
        finish_times = data['finish_times']

    # If CSV path provided, load from there
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'submit_time' in df.columns and 'finish_time' in df.columns:
            submit_times = df['submit_time'].tolist()
            finish_times = df['finish_time'].tolist()
    else:
        # Try to find CSV next to JSON
        potential_csv = json_path.replace('.json', '_requests.csv')
        if os.path.exists(potential_csv):
            df = pd.read_csv(potential_csv)
            if 'submit_time' in df.columns and 'finish_time' in df.columns:
                submit_times = df['submit_time'].tolist()
                finish_times = df['finish_time'].tolist()

    return BenchmarkMetrics(
        name=name,
        source_type='vllm',
        ttft_mean=ttft_mean,
        ttft_p90=ttft_p90,
        ttft_p99=ttft_p99,
        tpot_mean=tpot_mean,
        tpot_p90=tpot_p90,
        tpot_p99=tpot_p99,
        throughput_tps=throughput_tps,
        request_throughput=request_throughput,
        total_requests=total_requests,
        total_tokens=total_tokens,
        request_submit_times=submit_times,
        request_finish_times=finish_times,
    )


def load_vidur_benchmark(output_dir: str, name: str) -> BenchmarkMetrics:
    """
    Load Vidur simulation results from output directory.

    Args:
        output_dir: Path to Vidur's simulator_output/<timestamp>/ directory
        name: Display name for this benchmark

    Returns:
        BenchmarkMetrics object with Vidur results
    """
    plots_dir = Path(output_dir) / "plots"

    # Load TTFT (prefill e2e time) - in seconds
    ttft_path = plots_dir / "prefill_e2e_time.csv"
    if ttft_path.exists():
        ttft_df = pd.read_csv(ttft_path)
        ttft_col = 'prefill_e2e_time' if 'prefill_e2e_time' in ttft_df.columns else ttft_df.columns[1]
        ttft_values = ttft_df[ttft_col].values * 1000  # Convert to ms
        ttft_mean = np.mean(ttft_values)
        ttft_p90 = np.percentile(ttft_values, 90)
        ttft_p99 = np.percentile(ttft_values, 99)
    else:
        ttft_mean = ttft_p90 = ttft_p99 = 0

    # Load TPOT (decode time normalized per token) - in seconds
    tpot_path = plots_dir / "decode_time_execution_plus_preemption_normalized.csv"
    if tpot_path.exists():
        tpot_df = pd.read_csv(tpot_path)
        tpot_col = [c for c in tpot_df.columns if 'decode' in c.lower()][0] if any('decode' in c.lower() for c in tpot_df.columns) else tpot_df.columns[1]
        tpot_values = tpot_df[tpot_col].values * 1000  # Convert to ms
        tpot_mean = np.mean(tpot_values)
        tpot_p90 = np.percentile(tpot_values, 90)
        tpot_p99 = np.percentile(tpot_values, 99)
    else:
        tpot_mean = tpot_p90 = tpot_p99 = 0

    # Load request timestamps for throughput and plotting
    arrived_path = plots_dir / "request_arrived_at.csv"
    completed_path = plots_dir / "request_completed_at.csv"

    submit_times = None
    finish_times = None
    total_requests = 0

    if arrived_path.exists() and completed_path.exists():
        arrived_df = pd.read_csv(arrived_path)
        completed_df = pd.read_csv(completed_path)

        arrived_col = [c for c in arrived_df.columns if 'arrived' in c.lower()][0] if any('arrived' in c.lower() for c in arrived_df.columns) else arrived_df.columns[1]
        completed_col = [c for c in completed_df.columns if 'completed' in c.lower()][0] if any('completed' in c.lower() for c in completed_df.columns) else completed_df.columns[1]

        submit_times = arrived_df[arrived_col].tolist()
        finish_times = completed_df[completed_col].tolist()
        total_requests = len(finish_times)

    # Load request token counts
    prefill_tokens_path = plots_dir / "request_num_prefill_tokens.csv"
    decode_tokens_path = plots_dir / "request_num_decode_tokens.csv"

    total_tokens = 0
    if prefill_tokens_path.exists() and decode_tokens_path.exists():
        prefill_df = pd.read_csv(prefill_tokens_path)
        decode_df = pd.read_csv(decode_tokens_path)
        prefill_col = prefill_df.columns[1]
        decode_col = decode_df.columns[1]
        total_tokens = prefill_df[prefill_col].sum() + decode_df[decode_col].sum()

    # Calculate throughput
    if submit_times and finish_times:
        total_time = max(finish_times) - min(submit_times)
        request_throughput = total_requests / total_time if total_time > 0 else 0
        throughput_tps = total_tokens / total_time if total_time > 0 else 0
    else:
        request_throughput = 0
        throughput_tps = 0

    return BenchmarkMetrics(
        name=name,
        source_type='vidur',
        ttft_mean=ttft_mean,
        ttft_p90=ttft_p90,
        ttft_p99=ttft_p99,
        tpot_mean=tpot_mean,
        tpot_p90=tpot_p90,
        tpot_p99=tpot_p99,
        throughput_tps=throughput_tps,
        request_throughput=request_throughput,
        total_requests=total_requests,
        total_tokens=int(total_tokens),
        request_submit_times=submit_times,
        request_finish_times=finish_times,
    )


def parse_result_spec(spec: str) -> Tuple[str, str, str]:
    """
    Parse result specification string.

    Format: type:path:name
    - type: 'vllm' or 'vidur'
    - path: path to result file/directory
    - name: display name (optional, defaults to type + index)

    Returns:
        Tuple of (type, path, name)
    """
    parts = spec.split(':')

    if len(parts) < 2:
        raise ValueError(f"Invalid result spec: {spec}. Expected format: type:path[:name]")

    result_type = parts[0].lower()
    if result_type not in ('vllm', 'vidur'):
        raise ValueError(f"Invalid result type: {result_type}. Expected 'vllm' or 'vidur'")

    # Handle paths with colons (e.g., Windows paths or URLs)
    if len(parts) == 2:
        path = parts[1]
        name = None
    elif len(parts) == 3:
        path = parts[1]
        name = parts[2]
    else:
        # Path might contain colons, take last part as name
        path = ':'.join(parts[1:-1])
        name = parts[-1]

    return result_type, path, name


def print_comparison_table(benchmarks: List[BenchmarkMetrics]):
    """Print a formatted comparison table for multiple benchmarks."""

    if len(benchmarks) == 0:
        print("No benchmark results to compare.")
        return

    # Calculate column width based on names
    name_width = max(15, max(len(b.name) for b in benchmarks) + 2)

    print("\n" + "=" * (45 + name_width * len(benchmarks)))
    print("BENCHMARK COMPARISON")
    print("=" * (45 + name_width * len(benchmarks)))

    # Header
    header = f"{'Metric':<40}"
    for b in benchmarks:
        header += f" {b.name:>{name_width}}"
    print(header)
    print("-" * (45 + name_width * len(benchmarks)))

    metric_names = [
        ("TTFT Mean (ms)", lambda b: b.ttft_mean),
        ("TTFT P90 (ms)", lambda b: b.ttft_p90),
        ("TTFT P99 (ms)", lambda b: b.ttft_p99),
        ("TPOT Mean (ms)", lambda b: b.tpot_mean),
        ("TPOT P90 (ms)", lambda b: b.tpot_p90),
        ("TPOT P99 (ms)", lambda b: b.tpot_p99),
        ("Throughput (tokens/s)", lambda b: b.throughput_tps),
        ("Request Throughput (req/s)", lambda b: b.request_throughput),
        ("Total Requests", lambda b: b.total_requests),
        ("Total Tokens", lambda b: b.total_tokens),
    ]

    for metric_name, getter in metric_names:
        row = f"{metric_name:<40}"
        for b in benchmarks:
            value = getter(b)
            row += f" {value:>{name_width}.2f}"
        print(row)

    print("=" * (45 + name_width * len(benchmarks)))


def plot_request_progress(
    benchmarks: List[BenchmarkMetrics],
    output_path: str,
    align_start: bool = True,
):
    """
    Plot request completion progress over time for multiple benchmarks.

    X-axis: Time (seconds)
    Y-axis: Number of completed requests (i-th request)

    Args:
        benchmarks: List of benchmark results to plot
        output_path: Path to save the plot
        align_start: If True, align all benchmarks to start at time 0
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, benchmark in enumerate(benchmarks):
        if not benchmark.request_finish_times:
            continue

        times = sorted(benchmark.request_finish_times)

        if align_start:
            start_time = min(benchmark.request_submit_times) if benchmark.request_submit_times else min(times)
            times_normalized = [t - start_time for t in times]
        else:
            times_normalized = times

        request_counts = list(range(1, len(times) + 1))

        color = COLORS[i % len(COLORS)]
        linestyle = LINE_STYLES[i % len(LINE_STYLES)]

        ax.step(times_normalized, request_counts, where='post',
                label=benchmark.name, color=color, linewidth=2, linestyle=linestyle)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Completed Requests', fontsize=12)
    ax.set_title('Request Completion Progress', fontsize=14)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Progress plot saved to: {output_path}")


def plot_metrics_comparison(
    benchmarks: List[BenchmarkMetrics],
    output_path: str,
):
    """
    Plot bar chart comparison of key metrics for multiple benchmarks.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    n_benchmarks = len(benchmarks)
    bar_width = 0.8 / n_benchmarks

    # TTFT comparison
    ax1 = axes[0]
    metrics = ['Mean', 'P90', 'P99']
    x = np.arange(len(metrics))

    for i, b in enumerate(benchmarks):
        ttft_values = [b.ttft_mean, b.ttft_p90, b.ttft_p99]
        offset = (i - n_benchmarks / 2 + 0.5) * bar_width
        color = COLORS[i % len(COLORS)]
        ax1.bar(x + offset, ttft_values, bar_width, label=b.name, color=color, alpha=0.8)

    ax1.set_xlabel('Metric')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('TTFT Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')

    # TPOT comparison
    ax2 = axes[1]
    for i, b in enumerate(benchmarks):
        tpot_values = [b.tpot_mean, b.tpot_p90, b.tpot_p99]
        offset = (i - n_benchmarks / 2 + 0.5) * bar_width
        color = COLORS[i % len(COLORS)]
        ax2.bar(x + offset, tpot_values, bar_width, label=b.name, color=color, alpha=0.8)

    ax2.set_xlabel('Metric')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('TPOT Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    # Throughput comparison
    ax3 = axes[2]
    throughputs = ['Token TPS', 'Request RPS']
    x = np.arange(len(throughputs))
    bar_width_tp = 0.8 / n_benchmarks

    for i, b in enumerate(benchmarks):
        tp_values = [b.throughput_tps, b.request_throughput]
        offset = (i - n_benchmarks / 2 + 0.5) * bar_width_tp
        color = COLORS[i % len(COLORS)]
        ax3.bar(x + offset, tp_values, bar_width_tp, label=b.name, color=color, alpha=0.8)

    ax3.set_xlabel('Metric')
    ax3.set_ylabel('Rate')
    ax3.set_title('Throughput Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(throughputs)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics comparison plot saved to: {output_path}")


def save_comparison_csv(
    benchmarks: List[BenchmarkMetrics],
    output_path: str,
):
    """Save comparison results to CSV."""

    metric_names = [
        'TTFT Mean (ms)', 'TTFT P90 (ms)', 'TTFT P99 (ms)',
        'TPOT Mean (ms)', 'TPOT P90 (ms)', 'TPOT P99 (ms)',
        'Throughput (tokens/s)', 'Request Throughput (req/s)',
        'Total Requests', 'Total Tokens'
    ]

    data = {'Metric': metric_names}

    for b in benchmarks:
        data[b.name] = [
            b.ttft_mean, b.ttft_p90, b.ttft_p99,
            b.tpot_mean, b.tpot_p90, b.tpot_p99,
            b.throughput_tps, b.request_throughput,
            b.total_requests, b.total_tokens
        ]

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Comparison CSV saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare multiple benchmark results (vLLM and/or Vidur)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two results
  python compare_benchmarks.py \\
      --result vllm:/path/to/vllm_result.json:vLLM-Real \\
      --result vidur:/path/to/vidur_output/:Vidur-Sim

  # Compare multiple configurations
  python compare_benchmarks.py \\
      --result vllm:./results/qps1.json:vLLM-QPS1 \\
      --result vllm:./results/qps2.json:vLLM-QPS2 \\
      --result vidur:./vidur_out/qps1/:Vidur-QPS1 \\
      --result vidur:./vidur_out/qps2/:Vidur-QPS2

  # Legacy mode (two results only)
  python compare_benchmarks.py \\
      --vllm-result /path/to/vllm.json \\
      --vidur-result /path/to/vidur_output/
"""
    )

    # New multi-result argument
    parser.add_argument(
        '--result', '-r',
        type=str,
        action='append',
        default=[],
        help='Result specification: type:path:name (type=vllm|vidur, name is optional)'
    )

    # Legacy arguments for backward compatibility
    parser.add_argument(
        '--vllm-result', '-v',
        type=str,
        default=None,
        help='[Legacy] Path to vLLM benchmark result JSON file'
    )
    parser.add_argument(
        '--vllm-csv',
        type=str,
        default=None,
        help='[Legacy] Path to vLLM per-request CSV file'
    )
    parser.add_argument(
        '--vidur-result', '-d',
        type=str,
        default=None,
        help='[Legacy] Path to Vidur simulator_output/<timestamp>/ directory'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./comparison_output',
        help='Output directory for comparison results'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip generating plots'
    )
    parser.add_argument(
        '--no-align',
        action='store_true',
        help='Do not align start times in progress plot'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    benchmarks: List[BenchmarkMetrics] = []

    # Load from new --result arguments
    for i, spec in enumerate(args.result):
        result_type, path, name = parse_result_spec(spec)

        if name is None:
            name = f"{result_type.upper()}-{i+1}"

        print(f"Loading {result_type} benchmark: {path} as '{name}'...")

        if result_type == 'vllm':
            metrics = load_vllm_benchmark(path, name)
        else:
            metrics = load_vidur_benchmark(path, name)

        benchmarks.append(metrics)

    # Legacy mode: load from --vllm-result and --vidur-result
    if args.vllm_result:
        print(f"Loading vLLM benchmark (legacy): {args.vllm_result}...")
        metrics = load_vllm_benchmark(args.vllm_result, "vLLM", args.vllm_csv)
        benchmarks.append(metrics)

    if args.vidur_result:
        print(f"Loading Vidur benchmark (legacy): {args.vidur_result}...")
        metrics = load_vidur_benchmark(args.vidur_result, "Vidur")
        benchmarks.append(metrics)

    if len(benchmarks) == 0:
        print("Error: No benchmark results specified.")
        print("Use --result type:path:name or legacy --vllm-result/--vidur-result")
        return

    # Print comparison table
    print_comparison_table(benchmarks)

    # Save comparison CSV
    csv_path = os.path.join(args.output_dir, 'comparison_summary.csv')
    save_comparison_csv(benchmarks, csv_path)

    if not args.no_plot:
        # Plot request progress
        progress_path = os.path.join(args.output_dir, 'request_progress.png')
        plot_request_progress(benchmarks, progress_path, align_start=not args.no_align)

        # Plot metrics comparison
        metrics_path = os.path.join(args.output_dir, 'metrics_comparison.png')
        plot_metrics_comparison(benchmarks, metrics_path)

    print(f"\nComparison complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
