#!/usr/bin/env python3
"""
Minimal test script for Vidur predictor.
Run this to verify Vidur integration is working correctly.

Usage:
    python test_vidur_minimal.py --config /path/to/config.toml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_vidur_direct():
    """Test Vidur predictor directly without going through VidurStateEvaluate."""
    print("=== Testing Vidur Direct Import ===")

    try:
        from vidur.config import (
            ReplicaConfig,
            MetricsConfig,
            RandomForrestExecutionTimePredictorConfig,
            VllmSchedulerConfig,
        )
        from vidur.execution_time_predictor.execution_time_predictor_registry import (
            ExecutionTimePredictorRegistry,
        )
        print("  Vidur imports: OK")
    except ImportError as e:
        print(f"  Vidur imports: FAILED - {e}")
        return False

    # Create minimal config - using h20 to match user's setup
    try:
        replica_config = ReplicaConfig(
            model_name="meta-llama/Llama-3.1-8B",
            device="h20",
            network_device="h20_pairwise_nvlink",
            tensor_parallel_size=1,
            num_pipeline_stages=1,
        )
        print(f"  ReplicaConfig: OK")
    except Exception as e:
        print(f"  ReplicaConfig: FAILED - {e}")
        return False

    try:
        scheduler_config = VllmSchedulerConfig(
            block_size=16,
            batch_size_cap=128,
        )
        print(f"  VllmSchedulerConfig: OK")
    except Exception as e:
        print(f"  VllmSchedulerConfig: FAILED - {e}")
        return False

    try:
        metrics_config = MetricsConfig(
            cache_dir="cache",
            write_metrics=False,
        )
        print(f"  MetricsConfig: OK")
    except Exception as e:
        print(f"  MetricsConfig: FAILED - {e}")
        return False

    try:
        predictor_config = RandomForrestExecutionTimePredictorConfig()
        print(f"  PredictorConfig: OK")
        print(f"    compute_input_file: {getattr(predictor_config, 'compute_input_file', 'N/A')}")
        print(f"    attention_input_file: {getattr(predictor_config, 'attention_input_file', 'N/A')}")
    except Exception as e:
        print(f"  PredictorConfig: FAILED - {e}")
        return False

    # Create predictor
    try:
        predictor = ExecutionTimePredictorRegistry.get(
            predictor_config.get_type(),
            predictor_config=predictor_config,
            replica_config=replica_config,
            replica_scheduler_config=scheduler_config,
            metrics_config=metrics_config,
        )
        print(f"  Predictor creation: OK")
        print(f"    Type: {type(predictor).__name__}")
    except Exception as e:
        print(f"  Predictor creation: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

    return predictor


def test_vidur_prediction(predictor):
    """Test a simple prediction with Vidur."""
    print("\n=== Testing Vidur Prediction ===")

    # Create a simple batch using Vidur's own entities
    try:
        from vidur.entities import Request, Batch
        print("  Using Vidur native Request/Batch")

        # Create a request
        request = Request(
            arrived_at=0.0,
            num_prefill_tokens=128,
            num_decode_tokens=32,
        )
        print(f"    Request created: prefill={request.num_prefill_tokens}, decode={request.num_decode_tokens}")

        # Create a batch
        batch = Batch(
            replica_id=0,
            requests=[request],
            num_tokens=[128],  # prefill tokens
        )
        print(f"    Batch created: size={batch.size}, total_tokens={batch.total_num_tokens}")

    except Exception as e:
        print(f"  Creating Vidur entities: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

    # Make prediction
    try:
        execution_time = predictor.get_execution_time(
            batch=batch,
            pipeline_stage=0,
        )
        print(f"  Prediction: OK")
        print(f"    total_time: {execution_time.total_time} seconds")
        print(f"    total_time: {execution_time.total_time * 1000} ms")

        if execution_time.total_time == 0:
            print("  WARNING: Prediction returned 0!")

    except Exception as e:
        print(f"  Prediction: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_simple_batch_adapter():
    """Test using SimpleBatch/SimpleRequest adapters."""
    print("\n=== Testing SimpleBatch Adapter ===")

    try:
        from llmservingtuner.model.vidur_model import (
            VidurStateEvaluate,
            VidurPredictorConfig,
            SimpleBatch,
            SimpleRequest,
        )
        from llmservingtuner.inference.dataset import InputData
        from llmservingtuner.inference.data_format_v1 import BatchField, RequestField
    except ImportError as e:
        print(f"  Import: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

    print("  Imports: OK")

    # Reset singleton
    VidurStateEvaluate._instance = None
    VidurStateEvaluate._initialized = False

    # Create config - using h20 to match user's setup
    config = VidurPredictorConfig(
        model_name="meta-llama/Llama-3.1-8B",
        device="h20",
        network_device="h20_pairwise_nvlink",
        tensor_parallel_size=1,
        predictor_type="random_forest",
    )

    try:
        evaluator = VidurStateEvaluate(config)
        print("  VidurStateEvaluate: OK")
    except Exception as e:
        print(f"  VidurStateEvaluate: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

    # Create test input
    # BatchField = (batch_stage, batch_size, total_need_blocks, total_prefill_token, max_seq_len)
    batch_field = BatchField(
        batch_stage="prefill",
        batch_size=1,
        total_need_blocks=8,  # Estimated blocks needed
        total_prefill_token=128,
        max_seq_len=2048,
    )
    # RequestField = (input_length, need_blocks, output_length)
    request_field = RequestField(
        input_length=128,
        need_blocks=8,
        output_length=0,
    )
    input_data = InputData(
        batch_field=batch_field,
        request_field=(request_field,),
    )

    try:
        up, ud = evaluator.predict(input_data)
        print(f"  Prediction: OK")
        print(f"    Prefill time (Up): {up} ms")
        print(f"    Decode time (Ud): {ud} ms")

        if up == 0 or up == -1:
            print("  WARNING: Prefill prediction may be invalid!")

    except Exception as e:
        print(f"  Prediction: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Test Vidur predictor")
    parser.add_argument("--config", type=Path, help="Path to config.toml")
    args = parser.parse_args()

    print("Vidur Predictor Minimal Test")
    print("=" * 50)

    # Test 1: Direct Vidur import and predictor creation
    predictor = test_vidur_direct()
    if not predictor:
        print("\nDirect Vidur test failed. Check Vidur installation.")
        return 1

    # Test 2: Make a prediction using Vidur's native entities
    if not test_vidur_prediction(predictor):
        print("\nVidur native prediction failed.")
        return 1

    # Test 3: Test the SimpleBatch adapter
    if not test_simple_batch_adapter():
        print("\nSimpleBatch adapter test failed.")
        return 1

    print("\n" + "=" * 50)
    print("All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
