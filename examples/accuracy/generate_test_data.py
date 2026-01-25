#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate test data for predictor accuracy evaluation from vLLM profiling data.

Usage:
    python generate_test_data.py \
        --profile-csv /path/to/vllm_profile.csv \
        --output test_data.csv \
        --sample-size 1000
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def generate_test_data(
    profile_csv: Path,
    output_csv: Path,
    sample_size: int = None,
    random_seed: int = 42
) -> None:
    """
    Generate test data from vLLM profiling data.

    Args:
        profile_csv: Path to vLLM profile CSV
        output_csv: Output path for test data
        sample_size: Number of samples (None for all)
        random_seed: Random seed for sampling
    """
    print(f"Loading profile data from {profile_csv}")
    df = pd.read_csv(profile_csv)

    print(f"Original data shape: {df.shape}")

    # Identify and rename columns
    column_mapping = {
        'num_prefill_tokens': 'input_length',
        'num_decode_tokens': 'output_length',
        'latency_ms': 'model_execute_time',
        'latency': 'model_execute_time',
        'execution_time_ms': 'model_execute_time',
    }

    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    # Required columns
    required_cols = ['batch_stage', 'model_execute_time']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Select relevant columns
    output_cols = [
        'batch_stage',
        'batch_size',
        'input_length',
        'output_length',
        'max_seq_len',
        'model_execute_time',
    ]

    # Add default values for missing optional columns
    if 'batch_size' not in df.columns:
        df['batch_size'] = 1
    if 'input_length' not in df.columns:
        df['input_length'] = 128
    if 'output_length' not in df.columns:
        df['output_length'] = 1
    if 'max_seq_len' not in df.columns:
        df['max_seq_len'] = df['input_length'] + df['output_length']

    # Keep only valid columns that exist
    output_cols = [c for c in output_cols if c in df.columns]

    # Filter invalid rows
    df = df[df['model_execute_time'] > 0]
    df = df[df['batch_stage'].isin(['prefill', 'decode'])]

    print(f"After filtering: {len(df)} samples")
    print(f"  Prefill: {len(df[df['batch_stage'] == 'prefill'])}")
    print(f"  Decode: {len(df[df['batch_stage'] == 'decode'])}")

    # Sample if requested
    if sample_size and sample_size < len(df):
        np.random.seed(random_seed)

        # Stratified sampling by batch_stage
        prefill_df = df[df['batch_stage'] == 'prefill']
        decode_df = df[df['batch_stage'] == 'decode']

        prefill_ratio = len(prefill_df) / len(df)
        prefill_samples = int(sample_size * prefill_ratio)
        decode_samples = sample_size - prefill_samples

        sampled_prefill = prefill_df.sample(
            n=min(prefill_samples, len(prefill_df)),
            random_state=random_seed
        )
        sampled_decode = decode_df.sample(
            n=min(decode_samples, len(decode_df)),
            random_state=random_seed
        )

        df = pd.concat([sampled_prefill, sampled_decode])
        print(f"Sampled to {len(df)} samples")

    # Select output columns
    output_df = df[output_cols].copy()

    # Save
    output_df.to_csv(output_csv, index=False)
    print(f"Test data saved to {output_csv}")

    # Print statistics
    print("\nData Statistics:")
    print(f"  Total samples: {len(output_df)}")
    for stage in ['prefill', 'decode']:
        stage_df = output_df[output_df['batch_stage'] == stage]
        if len(stage_df) > 0:
            print(f"\n  {stage.capitalize()}:")
            print(f"    Count: {len(stage_df)}")
            print(f"    Latency mean: {stage_df['model_execute_time'].mean():.2f} ms")
            print(f"    Latency std:  {stage_df['model_execute_time'].std():.2f} ms")
            print(f"    Latency range: [{stage_df['model_execute_time'].min():.2f}, {stage_df['model_execute_time'].max():.2f}] ms")
            if 'input_length' in stage_df.columns:
                print(f"    Input length range: [{stage_df['input_length'].min()}, {stage_df['input_length'].max()}]")


def generate_synthetic_test_data(
    output_csv: Path,
    num_samples: int = 1000,
    random_seed: int = 42
) -> None:
    """
    Generate synthetic test data for testing the evaluation script.

    This creates random test data with realistic distributions.
    """
    np.random.seed(random_seed)

    records = []

    # Generate prefill samples
    n_prefill = num_samples // 2
    for _ in range(n_prefill):
        batch_size = np.random.choice([1, 2, 4, 8, 16, 32])
        input_length = np.random.randint(32, 2048)
        output_length = np.random.randint(1, 512)

        # Synthetic latency model (rough approximation)
        base_latency = 5.0  # base overhead in ms
        per_token_latency = 0.02  # ms per token
        batch_overhead = 0.5 * np.log2(batch_size + 1)

        latency = base_latency + per_token_latency * input_length * batch_size + batch_overhead
        latency *= np.random.uniform(0.9, 1.1)  # Add noise

        records.append({
            'batch_stage': 'prefill',
            'batch_size': batch_size,
            'input_length': input_length,
            'output_length': output_length,
            'max_seq_len': input_length + output_length,
            'model_execute_time': latency,
        })

    # Generate decode samples
    n_decode = num_samples - n_prefill
    for _ in range(n_decode):
        batch_size = np.random.choice([1, 4, 8, 16, 32, 64, 128])
        input_length = np.random.randint(32, 2048)
        output_length = np.random.randint(1, 512)

        # Synthetic decode latency
        base_latency = 2.0
        per_request_latency = 0.5
        kv_cache_overhead = 0.001 * (input_length + output_length)

        latency = base_latency + per_request_latency * batch_size + kv_cache_overhead * batch_size
        latency *= np.random.uniform(0.9, 1.1)

        records.append({
            'batch_stage': 'decode',
            'batch_size': batch_size,
            'input_length': input_length,
            'output_length': output_length,
            'max_seq_len': input_length + output_length,
            'model_execute_time': latency,
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    df.to_csv(output_csv, index=False)
    print(f"Synthetic test data saved to {output_csv}")
    print(f"  Total samples: {len(df)}")
    print(f"  Prefill: {len(df[df['batch_stage'] == 'prefill'])}")
    print(f"  Decode: {len(df[df['batch_stage'] == 'decode'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate test data for predictor evaluation"
    )
    parser.add_argument(
        "--profile-csv",
        type=Path,
        help="Path to vLLM profile CSV (optional if using --synthetic)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("test_data.csv"),
        help="Output CSV path"
    )
    parser.add_argument(
        "--sample-size", "-n",
        type=int,
        default=None,
        help="Number of samples to include (None for all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic test data instead of using profile"
    )

    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_test_data(
            args.output,
            num_samples=args.sample_size or 1000,
            random_seed=args.seed
        )
    elif args.profile_csv:
        generate_test_data(
            args.profile_csv,
            args.output,
            sample_size=args.sample_size,
            random_seed=args.seed
        )
    else:
        parser.error("Either --profile-csv or --synthetic must be specified")


if __name__ == "__main__":
    main()
