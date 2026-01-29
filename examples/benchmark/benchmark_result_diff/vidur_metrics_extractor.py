#!/usr/bin/env python3
"""
Extract metrics (TTFT, TPOT, TPS) from Vidur simulation results.

Usage:
    python vidur_metrics_extractor.py --results-dir ./simulator_output/<timestamp>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def find_request_metrics_file(results_dir: Path) -> Path:
    """Find the request metrics CSV file in Vidur output."""
    # Common locations for request metrics
    candidates = [
        results_dir / "plots" / "request_metrics.csv",
        results_dir / "request_metrics.csv",
        results_dir / "metrics.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Search for any CSV with 'request' in name
    for csv_file in results_dir.rglob("*.csv"):
        if 'request' in csv_file.name.lower():
            return csv_file

    # Return first CSV found
    csv_files = list(results_dir.rglob("*.csv"))
    if csv_files:
        return csv_files[0]

    return None


def load_metrics_json(results_dir: Path) -> dict:
    """Try to load metrics from JSON file if available."""
    json_candidates = [
        results_dir / "metrics.json",
        results_dir / "simulation_metrics.json",
        results_dir / "results.json",
    ]

    for path in json_candidates:
        if path.exists():
            with open(path) as f:
                return json.load(f)

    return None


def calculate_metrics_from_csv(csv_path: Path) -> dict:
    """Calculate TTFT, TPOT, TPS from request-level CSV."""
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} requests from {csv_path}")
    print(f"Columns: {list(df.columns)}")

    metrics = {}

    # --- TTFT (Time to First Token) ---
    # Try different column names that might represent TTFT
    ttft_columns = [
        'request_time_to_first_token',
        'time_to_first_token',
        'ttft',
        'request_scheduling_delay',
        'scheduling_delay',
        'prefill_complete_time',
    ]

    ttft_col = None
    for col in ttft_columns:
        if col in df.columns:
            ttft_col = col
            break

    if ttft_col:
        ttft = df[ttft_col].values
        # Convert to ms if values seem to be in seconds (< 100)
        if np.mean(ttft) < 100:
            ttft = ttft * 1000
        metrics['ttft_mean_ms'] = np.mean(ttft)
        metrics['ttft_p90_ms'] = np.percentile(ttft, 90)
        metrics['ttft_p99_ms'] = np.percentile(ttft, 99)
        print(f"  TTFT from column: {ttft_col}")
    else:
        # Try to compute from arrival and first token time
        if 'request_arrival_time' in df.columns:
            arrival_col = 'request_arrival_time'
        elif 'arrival_time' in df.columns:
            arrival_col = 'arrival_time'
        else:
            arrival_col = None

        if arrival_col and 'request_e2e_time' in df.columns:
            # Approximate TTFT as a fraction of e2e time
            print("  Warning: TTFT not directly available, using approximation")

    # --- TPOT (Time per Output Token) ---
    tpot_columns = [
        'request_time_per_output_token',
        'time_per_output_token',
        'tpot',
        'request_execution_time_per_token',
    ]

    tpot_col = None
    for col in tpot_columns:
        if col in df.columns:
            tpot_col = col
            break

    if tpot_col:
        tpot = df[tpot_col].values
        if np.mean(tpot) < 1:  # Likely in seconds
            tpot = tpot * 1000
        metrics['tpot_mean_ms'] = np.mean(tpot)
        metrics['tpot_p90_ms'] = np.percentile(tpot, 90)
        metrics['tpot_p99_ms'] = np.percentile(tpot, 99)
        print(f"  TPOT from column: {tpot_col}")
    else:
        # Calculate TPOT = execution_time / num_decode_tokens
        exec_cols = ['request_execution_time', 'execution_time', 'decode_time']
        decode_cols = ['request_num_decode_tokens', 'num_decode_tokens', 'output_tokens', 'output_length']

        exec_col = next((c for c in exec_cols if c in df.columns), None)
        decode_col = next((c for c in decode_cols if c in df.columns), None)

        if exec_col and decode_col:
            valid_mask = df[decode_col] > 0
            tpot = df.loc[valid_mask, exec_col] / df.loc[valid_mask, decode_col]
            if np.mean(tpot) < 1:
                tpot = tpot * 1000
            metrics['tpot_mean_ms'] = np.mean(tpot)
            metrics['tpot_p90_ms'] = np.percentile(tpot, 90)
            metrics['tpot_p99_ms'] = np.percentile(tpot, 99)
            print(f"  TPOT calculated from: {exec_col} / {decode_col}")

    # --- Throughput (TPS) ---
    # Total tokens / total time
    prefill_cols = ['request_num_prefill_tokens', 'num_prefill_tokens', 'input_tokens', 'input_length']
    decode_cols = ['request_num_decode_tokens', 'num_decode_tokens', 'output_tokens', 'output_length']
    e2e_cols = ['request_e2e_time', 'e2e_time', 'end_time', 'completion_time']

    prefill_col = next((c for c in prefill_cols if c in df.columns), None)
    decode_col = next((c for c in decode_cols if c in df.columns), None)
    e2e_col = next((c for c in e2e_cols if c in df.columns), None)

    if prefill_col and decode_col and e2e_col:
        total_tokens = df[prefill_col].sum() + df[decode_col].sum()
        total_time = df[e2e_col].max()
        if total_time > 0:
            metrics['throughput_tps'] = total_tokens / total_time
            metrics['total_tokens'] = int(total_tokens)
            metrics['total_time_s'] = total_time
            print(f"  Throughput from: ({prefill_col} + {decode_col}) / max({e2e_col})")

    # Also calculate request throughput
    if e2e_col:
        total_time = df[e2e_col].max()
        if total_time > 0:
            metrics['request_throughput_rps'] = len(df) / total_time

    return metrics


def print_metrics(metrics: dict):
    """Pretty print metrics."""
    print("\n" + "=" * 60)
    print("VIDUR SIMULATION METRICS")
    print("=" * 60)

    if 'ttft_mean_ms' in metrics:
        print(f"\nTTFT (Time to First Token):")
        print(f"  Mean:  {metrics['ttft_mean_ms']:.2f} ms")
        print(f"  P90:   {metrics.get('ttft_p90_ms', 'N/A'):.2f} ms" if 'ttft_p90_ms' in metrics else "  P90:   N/A")
        print(f"  P99:   {metrics.get('ttft_p99_ms', 'N/A'):.2f} ms" if 'ttft_p99_ms' in metrics else "  P99:   N/A")

    if 'tpot_mean_ms' in metrics:
        print(f"\nTPOT (Time per Output Token):")
        print(f"  Mean:  {metrics['tpot_mean_ms']:.2f} ms")
        print(f"  P90:   {metrics.get('tpot_p90_ms', 'N/A'):.2f} ms" if 'tpot_p90_ms' in metrics else "  P90:   N/A")
        print(f"  P99:   {metrics.get('tpot_p99_ms', 'N/A'):.2f} ms" if 'tpot_p99_ms' in metrics else "  P99:   N/A")

    if 'throughput_tps' in metrics:
        print(f"\nThroughput:")
        print(f"  TPS:   {metrics['throughput_tps']:.2f} tokens/s")
        if 'request_throughput_rps' in metrics:
            print(f"  RPS:   {metrics['request_throughput_rps']:.2f} requests/s")
        if 'total_tokens' in metrics:
            print(f"  Total: {metrics['total_tokens']} tokens in {metrics.get('total_time_s', 0):.2f}s")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract metrics from Vidur simulation results"
    )
    parser.add_argument(
        "--results-dir", "-d",
        type=Path,
        required=True,
        help="Path to Vidur simulator_output/<timestamp> directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Optional: Save metrics to JSON file"
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        return 1

    print(f"Analyzing Vidur results from: {args.results_dir}")

    # Try JSON first
    json_metrics = load_metrics_json(args.results_dir)
    if json_metrics:
        print("Found metrics.json")
        print_metrics(json_metrics)
        return 0

    # Find and process CSV
    csv_path = find_request_metrics_file(args.results_dir)
    if not csv_path:
        print(f"Error: No metrics CSV found in {args.results_dir}")
        print("Available files:")
        for f in args.results_dir.rglob("*"):
            if f.is_file():
                print(f"  {f.relative_to(args.results_dir)}")
        return 1

    metrics = calculate_metrics_from_csv(csv_path)

    if not metrics:
        print("Error: Could not extract any metrics")
        return 1

    print_metrics(metrics)

    # Save to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
