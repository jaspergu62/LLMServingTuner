#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample one batch per batch_size (bs) from vLLM profile data, run predictor(s),
save a data file, and plot accuracy (x=bs, y=time).

Usage:
    python plot_bs_accuracy.py \
        --profile-csv /path/to/vllm_profile.csv \
        --predictor vidur \
        --config /path/to/config.toml \
        --output-dir ./bs_accuracy_results

    python plot_bs_accuracy.py \
        --profile-csv /path/to/vllm_profile.csv \
        --predictor xgboost \
        --xgb-model /path/to/xgb_model.ubj
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# vLLM profile CSV columns
BATCH_COLUMN = "('batch_stage', 'batch_size', 'total_need_blocks', 'total_prefill_token', 'max_seq_len', 'model_execute_time')"
REQUEST_COLUMN = "('input_length', 'need_blocks', 'output_length')"


# Import predictor helpers from evaluate_predictors.py
sys.path.insert(0, str(Path(__file__).parent))
try:
    from evaluate_predictors import predict_with_xgboost, predict_with_vidur
except Exception as e:
    print(f"Error importing predictor helpers: {e}")
    raise


def parse_tuple_value(value):
    """Parse a tuple string like "(1, 2, 3)" into a tuple."""
    if isinstance(value, tuple):
        return value
    if pd.isna(value):
        return None
    try:
        return ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return None


def load_profile_data(profile_csv: Path) -> pd.DataFrame:
    """Load and parse vLLM profile CSV into normalized columns."""
    df = pd.read_csv(profile_csv)

    if BATCH_COLUMN not in df.columns or REQUEST_COLUMN not in df.columns:
        raise ValueError(
            "Profile CSV does not contain expected columns. "
            "Please check the vLLM profile data format."
        )

    records = []
    for idx, row in df.iterrows():
        batch_tuple = parse_tuple_value(row[BATCH_COLUMN])
        request_tuple = parse_tuple_value(row[REQUEST_COLUMN])

        if batch_tuple is None:
            continue

        # (batch_stage, batch_size, total_need_blocks, total_prefill_token, max_seq_len, model_execute_time)
        batch_stage, batch_size, total_need_blocks, total_prefill_token, max_seq_len, model_execute_time = batch_tuple

        if isinstance(batch_stage, str):
            batch_stage = batch_stage.strip("'\"").lower()
            if 'prefill' in batch_stage:
                batch_stage = 'prefill'
            elif 'decode' in batch_stage:
                batch_stage = 'decode'

        total_input_length = 0
        total_output_length = 0

        if request_tuple:
            for req in request_tuple:
                if len(req) >= 3:
                    input_len, _, output_len = req[0], req[1], req[2]
                    total_input_length += int(input_len)
                    total_output_length += int(output_len)

        try:
            model_execute_time = float(model_execute_time)
        except (ValueError, TypeError):
            continue

        if model_execute_time <= 0:
            continue

        records.append({
            'batch_stage': batch_stage,
            'batch_size': int(batch_size),
            'total_need_blocks': int(total_need_blocks),
            'total_prefill_token': int(total_prefill_token),
            'max_seq_len': int(max_seq_len),
            'model_execute_time': model_execute_time,
            'input_length': total_input_length,
            'output_length': total_output_length,
        })

    out_df = pd.DataFrame(records)
    out_df = out_df[out_df['batch_stage'].isin(['prefill', 'decode'])]
    return out_df


def sample_one_per_bs(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Randomly sample one row per batch_size, deterministically by seed."""
    rng = np.random.RandomState(seed)

    def _sample(group: pd.DataFrame) -> pd.DataFrame:
        rs = int(rng.randint(0, 2**32 - 1))
        return group.sample(n=1, random_state=rs)

    sampled = df.groupby('batch_size', group_keys=False).apply(_sample)
    sampled = sampled.reset_index(drop=True)
    return sampled


def ensure_path(base_dir: Path, maybe_path: str) -> Path:
    path = Path(maybe_path)
    if path.is_absolute():
        return path
    return base_dir / path


def plot_results(
    df: pd.DataFrame,
    output_path: Path,
    predictor: str,
    stage_label: str,
    ymax: Optional[float]
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    x = df['batch_size'].values
    y_truth = df['model_execute_time'].values

    plt.figure(figsize=(10, 5))
    plt.plot(x, y_truth, marker='o', label='Ground Truth', linewidth=1.5)

    if predictor in ['xgboost', 'both'] and 'xgboost_prediction' in df.columns:
        plt.plot(x, df['xgboost_prediction'].values, marker='s', label='XGBoost', linewidth=1.5)

    if predictor in ['vidur', 'both'] and 'vidur_prediction' in df.columns:
        plt.plot(x, df['vidur_prediction'].values, marker='^', label='Vidur', linewidth=1.5)

    plt.xlabel('Batch size (bs)')
    plt.ylabel('Time (ms)')
    plt.title(f'Predictor Accuracy by Batch Size ({stage_label})')
    if ymax is not None and ymax > 0:
        plt.ylim(bottom=0, top=ymax)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Plot saved to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample one batch per batch size and plot predictor accuracy"
    )
    parser.add_argument(
        "--profile-csv",
        type=Path,
        required=False,
        help="Path to vLLM profile CSV (required unless --use-cache is set)"
    )
    parser.add_argument(
        "--predictor",
        choices=["xgboost", "vidur", "both"],
        default="vidur",
        help="Which predictor(s) to use"
    )
    parser.add_argument(
        "--xgb-model",
        type=Path,
        help="Path to XGBoost model file (.ubj)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config.toml for Vidur"
    )
    parser.add_argument(
        "--stage",
        choices=["prefill", "decode", "all"],
        default="all",
        help="Filter by batch stage before sampling (prefill will be excluded)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./bs_accuracy_results"),
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--output-csv",
        default="bs_accuracy_samples.csv",
        help="Output CSV filename (or absolute path)"
    )
    parser.add_argument(
        "--output-plot",
        default="bs_accuracy_plot.pdf",
        help="Output plot filename (or absolute path)"
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=50.0,
        help="Y-axis upper limit for the plot (ms)"
    )
    parser.add_argument(
        "--figure-ymax",
        type=float,
        default=None,
        help="Alias of --ymax"
    )
    parser.add_argument(
        "--filter-max-x",
        type=int,
        default=40,
        help="Filter: keep rows with batch_size <= this value"
    )
    parser.add_argument(
        "--filter-max-y",
        type=float,
        default=100.0,
        help="Filter: keep rows with model_execute_time <= this value (ms)"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="If set and output CSV exists, reuse it and only plot"
    )

    args = parser.parse_args()
    if args.figure_ymax is not None:
        args.ymax = args.figure_ymax

    if not args.use_cache and args.profile_csv is None:
        print("Error: --profile-csv is required unless --use-cache is set")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = ensure_path(args.output_dir, args.output_csv)
    output_plot = ensure_path(args.output_dir, args.output_plot)

    if args.use_cache:
        if not output_csv.exists():
            print(f"Error: cached CSV not found at {output_csv}")
            return 1
        sampled = pd.read_csv(output_csv)
        if 'batch_stage' in sampled.columns:
            sampled = sampled[sampled['batch_stage'] != 'prefill'].copy()
        if args.stage != 'all' and 'batch_stage' in sampled.columns:
            sampled = sampled[sampled['batch_stage'] == args.stage].copy()
        if 'batch_size' in sampled.columns:
            sampled = sampled[sampled['batch_size'] <= args.filter_max_x].copy()
        if 'model_execute_time' in sampled.columns:
            sampled = sampled[sampled['model_execute_time'] <= args.filter_max_y].copy()
        if sampled.empty:
            print("Cached CSV has no data after filtering.")
            return 1
        print(f"Loaded cached data from {output_csv}")
    else:
        df = load_profile_data(args.profile_csv)

        # Always exclude prefill batches
        df = df[df['batch_stage'] != 'prefill']

        if args.stage != 'all':
            df = df[df['batch_stage'] == args.stage]
        df = df[df['batch_size'] <= args.filter_max_x]
        df = df[df['model_execute_time'] <= args.filter_max_y]

        if df.empty:
            print("No data after filtering. Check stage/profile data.")
            return 1

        sampled = sample_one_per_bs(df, args.seed)
        sampled = sampled.sort_values('batch_size').reset_index(drop=True)

        # Predict
        if args.predictor in ["xgboost", "both"]:
            if args.xgb_model is None:
                print("Error: --xgb-model is required for xgboost predictor")
                return 1
            xgb_pred = predict_with_xgboost(sampled, args.xgb_model, args.config)
            sampled['xgboost_prediction'] = xgb_pred
            sampled['xgboost_abs_error'] = np.abs(sampled['xgboost_prediction'] - sampled['model_execute_time'])

        if args.predictor in ["vidur", "both"]:
            vidur_pred = predict_with_vidur(sampled, args.config)
            sampled['vidur_prediction'] = vidur_pred
            sampled['vidur_abs_error'] = np.abs(sampled['vidur_prediction'] - sampled['model_execute_time'])

        # Save data
        sampled.to_csv(output_csv, index=False)
        print(f"Data saved to {output_csv}")

    # Plot
    stage_label = args.stage
    plot_results(sampled, output_plot, args.predictor, stage_label, args.ymax)

    return 0


if __name__ == "__main__":
    sys.exit(main())
