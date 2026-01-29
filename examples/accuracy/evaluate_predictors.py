#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predictor Accuracy Evaluation Script

This script evaluates and compares the prediction accuracy of XGBoost and Vidur
execution time predictors against ground truth measurements.

Usage:
    python evaluate_predictors.py \
        --test-data /path/to/test_data.csv \
        --xgb-model /path/to/xgb_model.ubj \
        --config /path/to/config.toml \
        --output-dir ./results

    # Or evaluate only one predictor:
    python evaluate_predictors.py \
        --test-data /path/to/test_data.csv \
        --predictor vidur \
        --config /path/to/config.toml
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class PredictionResult:
    """Container for prediction results."""
    predictor_name: str
    predictions: np.ndarray
    ground_truth: np.ndarray
    batch_stages: np.ndarray  # "prefill" or "decode"


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for a predictor."""
    predictor_name: str
    stage: str  # "all", "prefill", or "decode"
    num_samples: int
    mae: float  # Mean Absolute Error
    mape: float  # Mean Absolute Percentage Error (%)
    rmse: float  # Root Mean Squared Error
    r2: float  # R-squared
    median_error: float  # Median Absolute Error
    p90_error: float  # 90th percentile error
    p99_error: float  # 99th percentile error
    max_error: float  # Maximum error
    mean_pred: float  # Mean prediction
    mean_truth: float  # Mean ground truth


def calculate_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    predictor_name: str,
    stage: str = "all"
) -> AccuracyMetrics:
    """Calculate accuracy metrics for predictions."""
    if len(predictions) == 0:
        return AccuracyMetrics(
            predictor_name=predictor_name,
            stage=stage,
            num_samples=0,
            mae=float('nan'),
            mape=float('nan'),
            rmse=float('nan'),
            r2=float('nan'),
            median_error=float('nan'),
            p90_error=float('nan'),
            p99_error=float('nan'),
            max_error=float('nan'),
            mean_pred=float('nan'),
            mean_truth=float('nan'),
        )

    errors = np.abs(predictions - ground_truth)

    # MAE
    mae = np.mean(errors)

    # MAPE (avoid division by zero)
    nonzero_mask = ground_truth != 0
    if np.any(nonzero_mask):
        mape = np.mean(errors[nonzero_mask] / np.abs(ground_truth[nonzero_mask])) * 100
    else:
        mape = float('nan')

    # RMSE
    rmse = np.sqrt(np.mean(errors ** 2))

    # R-squared
    ss_res = np.sum((ground_truth - predictions) ** 2)
    ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')

    # Percentile errors
    median_error = np.median(errors)
    p90_error = np.percentile(errors, 90)
    p99_error = np.percentile(errors, 99)
    max_error = np.max(errors)

    return AccuracyMetrics(
        predictor_name=predictor_name,
        stage=stage,
        num_samples=len(predictions),
        mae=mae,
        mape=mape,
        rmse=rmse,
        r2=r2,
        median_error=median_error,
        p90_error=p90_error,
        p99_error=p99_error,
        max_error=max_error,
        mean_pred=np.mean(predictions),
        mean_truth=np.mean(ground_truth),
    )


def load_test_data(csv_path: Path) -> pd.DataFrame:
    """
    Load test data from CSV file.

    Expected columns:
        - batch_stage: "prefill" or "decode"
        - batch_size: number of requests in batch
        - input_length / num_prefill_tokens: input token count
        - output_length / num_decode_tokens: output token count
        - model_execute_time / latency_ms: ground truth latency in ms
    """
    df = pd.read_csv(csv_path)

    # Normalize column names
    column_mapping = {
        'num_prefill_tokens': 'input_length',
        'num_decode_tokens': 'output_length',
        'latency_ms': 'model_execute_time',
        'latency': 'model_execute_time',
    }

    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    # Ensure required columns exist
    required_cols = ['batch_stage', 'model_execute_time']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"  Prefill samples: {len(df[df['batch_stage'] == 'prefill'])}")
    print(f"  Decode samples: {len(df[df['batch_stage'] == 'decode'])}")

    return df


def predict_with_xgboost(
    df: pd.DataFrame,
    model_path: Path,
    config_path: Optional[Path] = None
) -> np.ndarray:
    """
    Make predictions using XGBoost model.

    The function expects the test data to have the same feature columns
    as the training data. It will automatically match features and
    apply label encoding.

    Returns predictions in milliseconds.
    """
    try:
        import xgboost
    except ImportError as e:
        print(f"Error importing XGBoost: {e}")
        raise

    # Load model
    model = xgboost.Booster()
    model.load_model(str(model_path))

    # Get feature columns from model
    feature_names = model.feature_names
    print(f"XGBoost model expects {len(feature_names)} features")

    # Try to import label encoder
    try:
        from llmservingtuner.data_feature.dataset import CustomLabelEncoder, preset_category_data
        custom_encoder = CustomLabelEncoder(preset_category_data)
        custom_encoder.fit()
        has_encoder = True
    except ImportError:
        print("Warning: CustomLabelEncoder not available, skipping encoding")
        has_encoder = False

    df_encoded = df.copy()

    # Apply label encoding if available
    if has_encoder:
        try:
            df_encoded = custom_encoder.transformer(df_encoded)
        except Exception as e:
            print(f"Warning: Label encoding failed: {e}")

    # Check which features are available
    available_features = [f for f in feature_names if f in df_encoded.columns]
    missing_features = set(feature_names) - set(available_features)

    if missing_features:
        print(f"Warning: {len(missing_features)} features missing from test data")
        if len(missing_features) <= 10:
            print(f"  Missing: {missing_features}")
        # Fill missing features with 0
        for f in missing_features:
            df_encoded[f] = 0

    # Prepare feature matrix
    X = df_encoded[feature_names].values
    dmatrix = xgboost.DMatrix(X, feature_names=feature_names)

    predictions = model.predict(dmatrix)
    print(f"XGBoost made {len(predictions)} predictions")

    return predictions


def predict_with_vidur(
    df: pd.DataFrame,
    config_path: Optional[Path] = None
) -> np.ndarray:
    """
    Make predictions using Vidur predictor.

    Returns predictions in milliseconds.
    """
    try:
        from llmservingtuner.model.vidur_model import VidurStateEvaluate, VidurPredictorConfig
        from llmservingtuner.inference.dataset import InputData
        from llmservingtuner.inference.data_format_v1 import BatchField, RequestField
    except ImportError as e:
        print(f"Error importing Vidur dependencies: {e}")
        raise

    # Load config if provided
    if config_path and config_path.exists():
        try:
            import toml
            config_data = toml.load(config_path)
            latency_config = config_data.get('latency_model', {})
        except ImportError:
            print("Warning: toml not installed, using default config")
            latency_config = {}
    else:
        latency_config = {}

    # Create Vidur config
    vidur_config = VidurPredictorConfig(
        model_name=latency_config.get('vidur_model_name', 'meta-llama/Llama-3.1-8B'),
        device=latency_config.get('vidur_device', 'h20'),
        network_device=latency_config.get('vidur_network_device', 'h20_pairwise_nvlink'),
        tensor_parallel_size=latency_config.get('vidur_tensor_parallel_size', 1),
        num_pipeline_stages=latency_config.get('vidur_num_pipeline_stages', 1),
        block_size=latency_config.get('vidur_block_size', 16),
        predictor_type=latency_config.get('vidur_predictor_type', 'random_forest'),
        cache_dir=latency_config.get('vidur_cache_dir', 'cache'),
        prediction_max_batch_size=latency_config.get('vidur_prediction_max_batch_size', 128),
        prediction_max_tokens_per_request=latency_config.get('vidur_prediction_max_tokens_per_request', 4096),
        # Profiling data files (required for accurate predictions)
        compute_input_file=latency_config.get('vidur_compute_input_file'),
        attention_input_file=latency_config.get('vidur_attention_input_file'),
        all_reduce_input_file=latency_config.get('vidur_all_reduce_input_file'),
        send_recv_input_file=latency_config.get('vidur_send_recv_input_file'),
        cpu_overhead_input_file=latency_config.get('vidur_cpu_overhead_input_file'),
    )

    print(f"Vidur config: model={vidur_config.model_name}, device={vidur_config.device}")
    print(f"  Profiling files:")
    print(f"    compute: {vidur_config.compute_input_file}")
    print(f"    attention: {vidur_config.attention_input_file}")

    # Reset singleton for fresh initialization
    VidurStateEvaluate._instance = None
    VidurStateEvaluate._initialized = False

    evaluator = VidurStateEvaluate(vidur_config)

    predictions = []
    errors = 0

    for idx, row in df.iterrows():
        try:
            batch_stage = row['batch_stage']
            batch_size = int(row.get('batch_size', 1))
            input_length = int(row.get('input_length', row.get('num_prefill_tokens', 128)))
            output_length = int(row.get('output_length', row.get('num_decode_tokens', 1)))

            # Estimate need_blocks (assuming block_size=16)
            block_size = 16
            need_blocks_per_req = (input_length + output_length + block_size - 1) // block_size

            # Build request fields
            # RequestField = (input_length, need_blocks, output_length)
            request_fields = []
            for _ in range(batch_size):
                req = RequestField(
                    input_length=input_length,
                    need_blocks=need_blocks_per_req,
                    output_length=output_length,
                )
                request_fields.append(req)

            # Calculate batch totals
            total_need_blocks = need_blocks_per_req * batch_size
            total_prefill_token = input_length * batch_size if batch_stage == 'prefill' else 0

            # Build batch field
            # BatchField = (batch_stage, batch_size, total_need_blocks, total_prefill_token, max_seq_len)
            batch_field = BatchField(
                batch_stage=batch_stage,
                batch_size=batch_size,
                total_need_blocks=total_need_blocks,
                total_prefill_token=total_prefill_token,
                max_seq_len=int(row.get('max_seq_len', 2048)),
            )

            # Create input data
            input_data = InputData(
                batch_field=batch_field,
                request_field=tuple(request_fields),
            )

            # Predict
            up, ud = evaluator.predict(input_data)

            # Select prediction based on stage
            if batch_stage == 'prefill':
                pred = up
            else:
                pred = ud

            predictions.append(pred)

        except Exception as e:
            if errors < 5:
                print(f"Warning: Prediction failed for row {idx}: {e}")
                import traceback
                traceback.print_exc()
            errors += 1
            predictions.append(0.0)

    if errors > 0:
        print(f"Vidur: {errors}/{len(df)} predictions failed")

    print(f"Vidur made {len(predictions)} predictions")
    return np.array(predictions)


def generate_report(
    metrics_list: List[AccuracyMetrics],
    output_dir: Path
) -> None:
    """Generate evaluation report."""

    # Print summary table
    print("\n" + "=" * 80)
    print("PREDICTOR ACCURACY EVALUATION REPORT")
    print("=" * 80)

    for metrics in metrics_list:
        print(f"\n{metrics.predictor_name} - {metrics.stage.upper()}")
        print("-" * 40)
        print(f"  Samples:      {metrics.num_samples}")
        print(f"  MAE:          {metrics.mae:.4f} ms")
        print(f"  MAPE:         {metrics.mape:.2f}%")
        print(f"  RMSE:         {metrics.rmse:.4f} ms")
        print(f"  R²:           {metrics.r2:.4f}")
        print(f"  Median Error: {metrics.median_error:.4f} ms")
        print(f"  P90 Error:    {metrics.p90_error:.4f} ms")
        print(f"  P99 Error:    {metrics.p99_error:.4f} ms")
        print(f"  Max Error:    {metrics.max_error:.4f} ms")
        print(f"  Mean Pred:    {metrics.mean_pred:.4f} ms")
        print(f"  Mean Truth:   {metrics.mean_truth:.4f} ms")

    # Save to JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / "accuracy_metrics.json"

    metrics_data = [asdict(m) for m in metrics_list]
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)

    print(f"\nMetrics saved to {metrics_file}")

    # Save to CSV for easy comparison
    csv_file = output_dir / "accuracy_metrics.csv"
    df = pd.DataFrame(metrics_data)
    df.to_csv(csv_file, index=False)
    print(f"Metrics CSV saved to {csv_file}")


def plot_comparison(
    results: List[PredictionResult],
    output_dir: Path
) -> None:
    """Generate comparison plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        # Scatter plot: prediction vs ground truth
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        stages = ['all', 'prefill', 'decode']
        for ax, stage in zip(axes, stages):
            if stage == 'all':
                mask = np.ones(len(result.predictions), dtype=bool)
            else:
                mask = result.batch_stages == stage

            if not np.any(mask):
                ax.set_title(f'{stage.capitalize()} (No data)')
                continue

            pred = result.predictions[mask]
            truth = result.ground_truth[mask]

            ax.scatter(truth, pred, alpha=0.5, s=10)

            # Perfect prediction line
            min_val = min(truth.min(), pred.min())
            max_val = max(truth.max(), pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')

            ax.set_xlabel('Ground Truth (ms)')
            ax.set_ylabel('Prediction (ms)')
            ax.set_title(f'{result.predictor_name} - {stage.capitalize()}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = output_dir / f"{result.predictor_name.lower()}_scatter.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"Scatter plot saved to {plot_file}")

    # Combined comparison plot
    if len(results) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, stage in zip(axes, ['prefill', 'decode']):
            for result in results:
                if stage == 'all':
                    mask = np.ones(len(result.predictions), dtype=bool)
                else:
                    mask = result.batch_stages == stage

                if not np.any(mask):
                    continue

                errors = np.abs(result.predictions[mask] - result.ground_truth[mask])
                ax.hist(errors, bins=50, alpha=0.5, label=result.predictor_name)

            ax.set_xlabel('Absolute Error (ms)')
            ax.set_ylabel('Count')
            ax.set_title(f'Error Distribution - {stage.capitalize()}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = output_dir / "comparison_error_hist.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"Comparison plot saved to {plot_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predictor accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--test-data", "-d",
        type=Path,
        required=True,
        help="Path to test data CSV"
    )
    parser.add_argument(
        "--predictor", "-p",
        choices=["xgboost", "vidur", "both"],
        default="both",
        help="Which predictor(s) to evaluate"
    )
    parser.add_argument(
        "--xgb-model",
        type=Path,
        help="Path to XGBoost model file (.ubj)"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to config.toml for Vidur settings"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./predictor_accuracy_results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )

    args = parser.parse_args()

    # Load test data
    df = load_test_data(args.test_data)
    ground_truth = df['model_execute_time'].values
    batch_stages = df['batch_stage'].values

    results: List[PredictionResult] = []
    all_metrics: List[AccuracyMetrics] = []

    # Evaluate XGBoost
    if args.predictor in ["xgboost", "both"]:
        if args.xgb_model is None:
            print("Warning: --xgb-model not specified, skipping XGBoost evaluation")
        else:
            print("\nEvaluating XGBoost predictor...")
            try:
                xgb_predictions = predict_with_xgboost(df, args.xgb_model, args.config)

                result = PredictionResult(
                    predictor_name="XGBoost",
                    predictions=xgb_predictions,
                    ground_truth=ground_truth,
                    batch_stages=batch_stages,
                )
                results.append(result)

                # Calculate metrics
                all_metrics.append(calculate_metrics(xgb_predictions, ground_truth, "XGBoost", "all"))

                prefill_mask = batch_stages == 'prefill'
                decode_mask = batch_stages == 'decode'

                if np.any(prefill_mask):
                    all_metrics.append(calculate_metrics(
                        xgb_predictions[prefill_mask],
                        ground_truth[prefill_mask],
                        "XGBoost", "prefill"
                    ))
                if np.any(decode_mask):
                    all_metrics.append(calculate_metrics(
                        xgb_predictions[decode_mask],
                        ground_truth[decode_mask],
                        "XGBoost", "decode"
                    ))

            except Exception as e:
                print(f"Error evaluating XGBoost: {e}")
                import traceback
                traceback.print_exc()

    # Evaluate Vidur
    if args.predictor in ["vidur", "both"]:
        print("\nEvaluating Vidur predictor...")
        try:
            vidur_predictions = predict_with_vidur(df, args.config)

            result = PredictionResult(
                predictor_name="Vidur",
                predictions=vidur_predictions,
                ground_truth=ground_truth,
                batch_stages=batch_stages,
            )
            results.append(result)

            # Calculate metrics
            all_metrics.append(calculate_metrics(vidur_predictions, ground_truth, "Vidur", "all"))

            prefill_mask = batch_stages == 'prefill'
            decode_mask = batch_stages == 'decode'

            if np.any(prefill_mask):
                all_metrics.append(calculate_metrics(
                    vidur_predictions[prefill_mask],
                    ground_truth[prefill_mask],
                    "Vidur", "prefill"
                ))
            if np.any(decode_mask):
                all_metrics.append(calculate_metrics(
                    vidur_predictions[decode_mask],
                    ground_truth[decode_mask],
                    "Vidur", "decode"
                ))

        except Exception as e:
            print(f"Error evaluating Vidur: {e}")
            import traceback
            traceback.print_exc()

    if not all_metrics:
        print("No predictions were made. Check errors above.")
        return 1

    # Generate report
    generate_report(all_metrics, args.output_dir)

    # Generate plots
    if not args.no_plots and results:
        plot_comparison(results, args.output_dir)

    # Save detailed predictions
    if results:
        details_file = args.output_dir / "detailed_predictions.csv"
        detail_data = {
            'ground_truth': ground_truth,
            'batch_stage': batch_stages,
        }
        for result in results:
            detail_data[f'{result.predictor_name.lower()}_prediction'] = result.predictions
            detail_data[f'{result.predictor_name.lower()}_error'] = np.abs(
                result.predictions - ground_truth
            )

        pd.DataFrame(detail_data).to_csv(details_file, index=False)
        print(f"Detailed predictions saved to {details_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
