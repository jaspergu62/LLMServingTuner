#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate test data for predictor accuracy evaluation from vLLM profiling data.

The vLLM profile CSV has two columns with tuple values:
- Column 1: "('batch_stage', 'batch_size', 'total_need_blocks', 'total_prefill_token', 'max_seq_len', 'model_execute_time')"
- Column 2: "('input_length', 'need_blocks', 'output_length')"

Usage:
    python generate_test_data.py \
        --profile-csv /path/to/vllm_profile.csv \
        --output test_data.csv \
        --sample-size 1000
"""

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd
import numpy as np


# Column names in the vLLM profile CSV
BATCH_COLUMN = "('batch_stage', 'batch_size', 'total_need_blocks', 'total_prefill_token', 'max_seq_len', 'model_execute_time')"
REQUEST_COLUMN = "('input_length', 'need_blocks', 'output_length')"


def parse_tuple_value(value):
    """Parse a tuple string like "(1, 2, 3)" into an actual tuple."""
    if isinstance(value, tuple):
        return value
    if pd.isna(value):
        return None
    try:
        return ast.literal_eval(str(value))
    except (ValueError, SyntaxError) as e:
        print(f"Warning: Failed to parse tuple value: {value}, error: {e}")
        return None


def generate_test_data(
    profile_csv: Path,
    output_csv: Path,
    sample_size: int = None,
    random_seed: int = 42
) -> None:
    """
    Generate test data from vLLM profiling data.

    The vLLM profile CSV has two columns:
    1. Batch info tuple: (batch_stage, batch_size, total_need_blocks, total_prefill_token, max_seq_len, model_execute_time)
    2. Request info tuple of tuples: ((input_length, need_blocks, output_length), ...)

    Args:
        profile_csv: Path to vLLM profile CSV
        output_csv: Output path for test data
        sample_size: Number of samples (None for all)
        random_seed: Random seed for sampling
    """
    print(f"Loading profile data from {profile_csv}")
    df = pd.read_csv(profile_csv)

    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Verify expected columns
    if BATCH_COLUMN not in df.columns:
        print(f"Error: Expected column '{BATCH_COLUMN}' not found")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    if REQUEST_COLUMN not in df.columns:
        print(f"Error: Expected column '{REQUEST_COLUMN}' not found")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Parse the tuple columns
    records = []
    for idx, row in df.iterrows():
        batch_tuple = parse_tuple_value(row[BATCH_COLUMN])
        request_tuple = parse_tuple_value(row[REQUEST_COLUMN])

        if batch_tuple is None:
            print(f"Warning: Skipping row {idx} due to invalid batch tuple")
            continue

        # Unpack batch tuple
        # (batch_stage, batch_size, total_need_blocks, total_prefill_token, max_seq_len, model_execute_time)
        batch_stage, batch_size, total_need_blocks, total_prefill_token, max_seq_len, model_execute_time = batch_tuple

        # Clean batch_stage (remove quotes if present)
        if isinstance(batch_stage, str):
            batch_stage = batch_stage.strip("'\"").lower()
            # Normalize batch stage names
            if 'prefill' in batch_stage.lower():
                batch_stage = 'prefill'
            elif 'decode' in batch_stage.lower():
                batch_stage = 'decode'

        # Calculate aggregated input/output lengths from request tuples
        total_input_length = 0
        total_output_length = 0
        total_need_blocks_from_req = 0

        if request_tuple:
            # request_tuple is a tuple of tuples: ((input_length, need_blocks, output_length), ...)
            for req in request_tuple:
                if len(req) >= 3:
                    input_len, need_blocks, output_len = req[0], req[1], req[2]
                    total_input_length += input_len
                    total_need_blocks_from_req += need_blocks
                    total_output_length += output_len

        # Convert model_execute_time to float
        try:
            model_execute_time = float(model_execute_time)
        except (ValueError, TypeError):
            print(f"Warning: Invalid model_execute_time at row {idx}: {model_execute_time}")
            continue

        # Skip invalid rows
        if model_execute_time <= 0:
            continue

        records.append({
            'batch_stage': batch_stage,
            'batch_size': int(batch_size),
            'total_need_blocks': int(total_need_blocks),
            'total_prefill_token': int(total_prefill_token),
            'max_seq_len': int(max_seq_len),
            'model_execute_time': model_execute_time,
            # Aggregated from request tuples
            'input_length': total_input_length,
            'output_length': total_output_length,
        })

    output_df = pd.DataFrame(records)

    print(f"Parsed {len(output_df)} valid samples")

    # Filter by batch_stage
    output_df = output_df[output_df['batch_stage'].isin(['prefill', 'decode'])]
    print(f"After filtering by batch_stage: {len(output_df)} samples")
    print(f"  Prefill: {len(output_df[output_df['batch_stage'] == 'prefill'])}")
    print(f"  Decode: {len(output_df[output_df['batch_stage'] == 'decode'])}")

    # Sample if requested
    if sample_size and sample_size < len(output_df):
        np.random.seed(random_seed)

        # Stratified sampling by batch_stage
        prefill_df = output_df[output_df['batch_stage'] == 'prefill']
        decode_df = output_df[output_df['batch_stage'] == 'decode']

        prefill_ratio = len(prefill_df) / len(output_df) if len(output_df) > 0 else 0.5
        prefill_samples = int(sample_size * prefill_ratio)
        decode_samples = sample_size - prefill_samples

        sampled_prefill = prefill_df.sample(
            n=min(prefill_samples, len(prefill_df)),
            random_state=random_seed
        ) if len(prefill_df) > 0 else pd.DataFrame()

        sampled_decode = decode_df.sample(
            n=min(decode_samples, len(decode_df)),
            random_state=random_seed
        ) if len(decode_df) > 0 else pd.DataFrame()

        output_df = pd.concat([sampled_prefill, sampled_decode])
        print(f"Sampled to {len(output_df)} samples")

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
            print(f"    Batch size range: [{stage_df['batch_size'].min()}, {stage_df['batch_size'].max()}]")
            print(f"    Max seq len range: [{stage_df['max_seq_len'].min()}, {stage_df['max_seq_len'].max()}]")


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
        max_seq_len = input_length + output_length

        # Synthetic latency model (rough approximation)
        base_latency = 5.0  # base overhead in ms
        per_token_latency = 0.02  # ms per token
        batch_overhead = 0.5 * np.log2(batch_size + 1)

        latency = base_latency + per_token_latency * input_length * batch_size + batch_overhead
        latency *= np.random.uniform(0.9, 1.1)  # Add noise

        records.append({
            'batch_stage': 'prefill',
            'batch_size': batch_size,
            'total_need_blocks': (input_length + output_length) // 16 * batch_size,
            'total_prefill_token': input_length * batch_size,
            'max_seq_len': max_seq_len,
            'model_execute_time': latency,
            'input_length': input_length * batch_size,
            'output_length': output_length * batch_size,
        })

    # Generate decode samples
    n_decode = num_samples - n_prefill
    for _ in range(n_decode):
        batch_size = np.random.choice([1, 4, 8, 16, 32, 64, 128])
        input_length = np.random.randint(32, 2048)
        output_length = np.random.randint(1, 512)
        max_seq_len = input_length + output_length

        # Synthetic decode latency
        base_latency = 2.0
        per_request_latency = 0.5
        kv_cache_overhead = 0.001 * (input_length + output_length)

        latency = base_latency + per_request_latency * batch_size + kv_cache_overhead * batch_size
        latency *= np.random.uniform(0.9, 1.1)

        records.append({
            'batch_stage': 'decode',
            'batch_size': batch_size,
            'total_need_blocks': (input_length + output_length) // 16 * batch_size,
            'total_prefill_token': 0,
            'max_seq_len': max_seq_len,
            'model_execute_time': latency,
            'input_length': input_length * batch_size,
            'output_length': output_length * batch_size,
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
