#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export benchmark dataset to Vidur trace CSV format.

This script extracts request information from various benchmark datasets
and exports them to a CSV file compatible with Vidur's trace_replay mode.

Usage:
    # Export random dataset
    python export_trace_for_vidur.py \
        --dataset-name random \
        --num-prompts 1000 \
        --request-rate 20 \
        --random-input-len 512 \
        --random-output-len 128 \
        --output my_trace.csv

    # Export sharegpt dataset
    python export_trace_for_vidur.py \
        --dataset-name sharegpt \
        --dataset-path /path/to/sharegpt.json \
        --num-prompts 1000 \
        --request-rate 10 \
        --output sharegpt_trace.csv

    # Export with trace arrival times (for vidur dataset)
    python export_trace_for_vidur.py \
        --dataset-name vidur \
        --dataset-path /path/to/trace.csv \
        --num-prompts 500 \
        --use-trace-arrival-times \
        --output vidur_subset.csv

Then run Vidur simulation:
    cd /path/to/vidur
    python -m vidur.main \
        --replica_config_model_name meta-llama/Llama-2-7b-hf \
        --replica_config_device a100 \
        --replica_scheduler_config_type vllm \
        --request_generator_config_type trace_replay \
        --trace_request_generator_config_trace_file /path/to/my_trace.csv
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_dataset import (
    RandomDataset,
    ShareGPTDataset,
    VidurTraceDataset,
    SonnetDataset,
    SampleRequest,
)

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from transformers import AutoTokenizer
    def get_tokenizer(name, **kwargs):
        return AutoTokenizer.from_pretrained(name, **kwargs)


def generate_arrival_times(
    num_requests: int,
    request_rate: float,
    burstiness: float = 1.0,
) -> list[float]:
    """
    Generate arrival times using the same logic as vllm's get_request().

    This matches vllm/benchmarks/serve.py exactly:
    - When burstiness == 1.0: Exponential distribution (Poisson process)
    - Otherwise: Gamma distribution with shape=burstiness, scale=1/(rate*burstiness)
    - Delays are normalized to match the expected total duration

    Note: Random seed should be set before calling this function.

    Args:
        num_requests: Number of requests
        request_rate: Requests per second (inf = all at time 0)
        burstiness: Burstiness factor (1.0 = Poisson, <1 = bursty, >1 = uniform)

    Returns:
        List of arrival times in seconds
    """
    if request_rate == float('inf'):
        return [0.0] * num_requests

    # Generate inter-arrival delays (same as vllm)
    delays = []
    for _ in range(num_requests):
        if burstiness == float('inf'):
            # Constant delay
            delay = 1.0 / request_rate
        else:
            # Gamma distribution (when burstiness=1.0, this is exponential)
            shape = burstiness
            scale = 1.0 / (request_rate * burstiness)
            delay = np.random.gamma(shape, scale)
        delays.append(delay)

    # Normalize delays to match expected total duration (same as vllm)
    # This ensures the actual throughput matches the requested rate
    total_delay = sum(delays)
    expected_total = num_requests / request_rate
    if total_delay > 0:
        scale_factor = expected_total / total_delay
        delays = [d * scale_factor for d in delays]

    # Convert delays to arrival times
    arrival_times = []
    current_time = 0.0
    for delay in delays:
        current_time += delay
        arrival_times.append(current_time)

    return arrival_times


def export_to_vidur_trace(
    requests: list[SampleRequest],
    output_path: Path,
    request_rate: float = float('inf'),
    burstiness: float = 1.0,
    use_trace_arrival_times: bool = False,
) -> pd.DataFrame:
    """
    Export SampleRequest list to Vidur trace CSV format.

    Args:
        requests: List of SampleRequest from benchmark dataset
        output_path: Output CSV file path
        request_rate: Requests per second for generating arrival times
        burstiness: Burstiness factor for arrival time generation
        use_trace_arrival_times: If True, use arrival_time from requests

    Returns:
        DataFrame with exported data
    """
    records = []

    # Generate or extract arrival times
    if use_trace_arrival_times:
        # Use existing arrival times from requests
        arrival_times = []
        for req in requests:
            if req.arrival_time is not None:
                arrival_times.append(req.arrival_time)
            else:
                # Fallback: use index-based time if no arrival_time
                arrival_times.append(len(arrival_times) * 0.1)
    else:
        # Generate new arrival times
        arrival_times = generate_arrival_times(
            num_requests=len(requests),
            request_rate=request_rate,
            burstiness=burstiness,
        )

    for req, arrived_at in zip(requests, arrival_times):
        records.append({
            'arrived_at': arrived_at,
            'num_prefill_tokens': req.prompt_len,
            'num_decode_tokens': req.expected_output_len,
        })

    df = pd.DataFrame(records)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)

    return df


def print_trace_stats(df: pd.DataFrame, output_path: Path):
    """Print statistics about the exported trace."""
    print("\n" + "=" * 60)
    print(" Trace Export Summary ".center(60, "="))
    print("=" * 60)

    print(f"\n{'Output file:':<30} {output_path}")
    print(f"{'Total requests:':<30} {len(df)}")

    # Token statistics
    total_prefill = df['num_prefill_tokens'].sum()
    total_decode = df['num_decode_tokens'].sum()
    print(f"\n{'Total prefill tokens:':<30} {total_prefill:,}")
    print(f"{'Total decode tokens:':<30} {total_decode:,}")
    print(f"{'Avg prefill tokens/request:':<30} {df['num_prefill_tokens'].mean():.1f}")
    print(f"{'Avg decode tokens/request:':<30} {df['num_decode_tokens'].mean():.1f}")

    # Arrival time statistics
    if len(df) > 1:
        duration = df['arrived_at'].max() - df['arrived_at'].min()
        if duration > 0:
            rps = len(df) / duration
            tps = (total_prefill + total_decode) / duration
            print(f"\n{'Trace duration (s):':<30} {duration:.2f}")
            print(f"{'Average RPS:':<30} {rps:.2f}")
            print(f"{'Average TPS:':<30} {tps:.1f}")

    # Token distribution
    print(f"\n{'Prefill tokens distribution:':<30}")
    print(f"  {'Min:':<26} {df['num_prefill_tokens'].min()}")
    print(f"  {'Max:':<26} {df['num_prefill_tokens'].max()}")
    print(f"  {'P50:':<26} {df['num_prefill_tokens'].quantile(0.5):.0f}")
    print(f"  {'P90:':<26} {df['num_prefill_tokens'].quantile(0.9):.0f}")
    print(f"  {'P99:':<26} {df['num_prefill_tokens'].quantile(0.99):.0f}")

    print(f"\n{'Decode tokens distribution:':<30}")
    print(f"  {'Min:':<26} {df['num_decode_tokens'].min()}")
    print(f"  {'Max:':<26} {df['num_decode_tokens'].max()}")
    print(f"  {'P50:':<26} {df['num_decode_tokens'].quantile(0.5):.0f}")
    print(f"  {'P90:':<26} {df['num_decode_tokens'].quantile(0.9):.0f}")
    print(f"  {'P99:':<26} {df['num_decode_tokens'].quantile(0.99):.0f}")

    print("\n" + "=" * 60)

    # Print vidur command hint
    print("\nTo run Vidur simulation with this trace:")
    print("-" * 60)
    print(f"""
python -m vidur.main \\
    --replica_config_model_name meta-llama/Llama-2-7b-hf \\
    --replica_config_device a100 \\
    --replica_scheduler_config_type vllm \\
    --request_generator_config_type trace_replay \\
    --trace_request_generator_config_trace_file {output_path.absolute()}
""")


def load_dataset(args, tokenizer) -> list[SampleRequest]:
    """Load dataset based on arguments."""

    if args.dataset_name == 'random':
        dataset = RandomDataset(dataset_path=args.dataset_path)
        requests = dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.num_prompts,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            range_ratio=args.random_range_ratio,
            prefix_len=args.random_prefix_len,
        )

    elif args.dataset_name == 'sharegpt':
        if args.dataset_path is None:
            raise ValueError("--dataset-path is required for sharegpt dataset")
        dataset = ShareGPTDataset(
            random_seed=args.seed,
            dataset_path=args.dataset_path,
        )
        requests = dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.num_prompts,
            output_len=args.sharegpt_output_len,
        )

    elif args.dataset_name == 'vidur':
        if args.dataset_path is None:
            raise ValueError("--dataset-path is required for vidur dataset")
        dataset = VidurTraceDataset(
            dataset_path=args.dataset_path,
            random_seed=args.seed,
        )
        requests = dataset.sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            prefill_scale_factor=args.vidur_prefill_scale,
            decode_scale_factor=args.vidur_decode_scale,
            time_scale_factor=args.vidur_time_scale,
            max_total_tokens=args.vidur_max_total_tokens,
        )

    elif args.dataset_name == 'sonnet':
        dataset = SonnetDataset(dataset_path=args.dataset_path)
        requests = dataset.sample(
            num_requests=args.num_prompts,
            input_len=args.sonnet_input_len,
            output_len=args.sonnet_output_len,
            prefix_len=args.sonnet_prefix_len,
            tokenizer=tokenizer,
        )

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    return requests


def create_argument_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Export benchmark dataset to Vidur trace CSV format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='benchmark_trace.csv',
        help='Output CSV file path (default: benchmark_trace.csv)',
    )

    # Dataset options
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='random',
        choices=['random', 'sharegpt', 'vidur', 'sonnet'],
        help='Name of the dataset to export (default: random)',
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='Path to the dataset file (required for sharegpt/vidur)',
    )
    parser.add_argument(
        '--num-prompts',
        type=int,
        default=1000,
        help='Number of prompts to export (default: 1000)',
    )

    # Arrival time options
    parser.add_argument(
        '--request-rate',
        type=float,
        default=float('inf'),
        help='Requests per second for generating arrival times. '
             'Use "inf" to set all arrivals at time 0 (default: inf)',
    )
    parser.add_argument(
        '--burstiness',
        type=float,
        default=1.0,
        help='Burstiness factor: 1.0=Poisson, <1=bursty, >1=uniform (default: 1.0)',
    )
    parser.add_argument(
        '--use-trace-arrival-times',
        action='store_true',
        help='Use arrival times from the trace dataset (for vidur dataset)',
    )

    # Tokenizer options
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='meta-llama/Llama-2-7b-hf',
        help='Tokenizer name or path (default: meta-llama/Llama-2-7b-hf)',
    )
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        help='Trust remote code from HuggingFace',
    )

    # Random seed
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)',
    )

    # Random dataset options
    random_group = parser.add_argument_group('random dataset options')
    random_group.add_argument(
        '--random-input-len',
        type=int,
        default=512,
        help='Input length for random dataset (default: 512)',
    )
    random_group.add_argument(
        '--random-output-len',
        type=int,
        default=128,
        help='Output length for random dataset (default: 128)',
    )
    random_group.add_argument(
        '--random-range-ratio',
        type=float,
        default=0.0,
        help='Range ratio for random sampling (default: 0.0)',
    )
    random_group.add_argument(
        '--random-prefix-len',
        type=int,
        default=0,
        help='Prefix length for random dataset (default: 0)',
    )

    # ShareGPT dataset options
    sharegpt_group = parser.add_argument_group('sharegpt dataset options')
    sharegpt_group.add_argument(
        '--sharegpt-output-len',
        type=int,
        default=None,
        help='Override output length for sharegpt dataset',
    )

    # Vidur dataset options
    vidur_group = parser.add_argument_group('vidur dataset options')
    vidur_group.add_argument(
        '--vidur-prefill-scale',
        type=float,
        default=1.0,
        help='Prefill token scaling factor (default: 1.0)',
    )
    vidur_group.add_argument(
        '--vidur-decode-scale',
        type=float,
        default=1.0,
        help='Decode token scaling factor (default: 1.0)',
    )
    vidur_group.add_argument(
        '--vidur-time-scale',
        type=float,
        default=1.0,
        help='Time scaling factor (default: 1.0)',
    )
    vidur_group.add_argument(
        '--vidur-max-total-tokens',
        type=int,
        default=None,
        help='Max total tokens per request',
    )

    # Sonnet dataset options
    sonnet_group = parser.add_argument_group('sonnet dataset options')
    sonnet_group.add_argument(
        '--sonnet-input-len',
        type=int,
        default=550,
        help='Input length for sonnet dataset (default: 550)',
    )
    sonnet_group.add_argument(
        '--sonnet-output-len',
        type=int,
        default=150,
        help='Output length for sonnet dataset (default: 150)',
    )
    sonnet_group.add_argument(
        '--sonnet-prefix-len',
        type=int,
        default=200,
        help='Prefix length for sonnet dataset (default: 200)',
    )

    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    # Set random seeds (same as benchmark_serving.py)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = get_tokenizer(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"Loading dataset: {args.dataset_name}")
    requests = load_dataset(args, tokenizer)
    print(f"Loaded {len(requests)} requests")

    # Export to CSV
    output_path = Path(args.output)
    print(f"Exporting to: {output_path}")

    df = export_to_vidur_trace(
        requests=requests,
        output_path=output_path,
        request_rate=args.request_rate,
        burstiness=args.burstiness,
        use_trace_arrival_times=args.use_trace_arrival_times,
    )

    # Print statistics
    print_trace_stats(df, output_path)


if __name__ == '__main__':
    main()
