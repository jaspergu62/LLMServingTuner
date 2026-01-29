# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Vidur execution time predictor adapter for LLMServingTuner.

This module provides a wrapper around Vidur's execution time predictor
to be compatible with the LLMServingTuner prediction interface.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from llmservingtuner.inference.dataset import InputData


class SimpleRequest:
    """
    A lightweight Request adapter for Vidur's Batch construction.

    This class mimics the essential properties of vidur.entities.Request
    that are needed for execution time prediction.
    """

    _id_counter = 0

    def __init__(
        self,
        prefill_tokens: int,
        decode_tokens: int,
        is_prefill_complete: bool = False,
    ):
        SimpleRequest._id_counter += 1
        self._id = SimpleRequest._id_counter
        self._prefill_tokens = prefill_tokens
        self._decode_tokens = decode_tokens
        self._is_prefill_complete = is_prefill_complete
        self._num_processed_tokens = decode_tokens if is_prefill_complete else 0

    @property
    def id(self) -> int:
        return self._id

    @property
    def prefill_tokens(self) -> int:
        return self._prefill_tokens

    # Vidur may use num_prefill_tokens instead of prefill_tokens
    @property
    def num_prefill_tokens(self) -> int:
        return self._prefill_tokens

    @property
    def decode_tokens(self) -> int:
        return self._decode_tokens

    # Vidur may use num_decode_tokens instead of decode_tokens
    @property
    def num_decode_tokens(self) -> int:
        return self._decode_tokens

    @property
    def is_prefill_complete(self) -> bool:
        return self._is_prefill_complete

    @property
    def num_processed_tokens(self) -> int:
        return self._num_processed_tokens

    @property
    def total_tokens(self) -> int:
        """Total tokens = prefill + decode tokens."""
        return self._prefill_tokens + self._decode_tokens

    @property
    def completed(self) -> bool:
        return False

    @property
    def preempted(self) -> bool:
        return False

    def on_batch_schedule(self, _time: float) -> None:
        """Interface compatibility - called when batch is scheduled."""
        pass

    def on_batch_end(self, _time: float, _num_tokens: int) -> None:
        """Interface compatibility - called when batch ends."""
        pass


class SimpleBatch:
    """
    A lightweight Batch adapter for Vidur's execution time predictor.

    This class mimics the essential properties of vidur.entities.Batch
    that are needed for execution time prediction.
    """

    _id_counter = 0

    def __init__(
        self,
        requests: List[SimpleRequest],
        num_tokens: List[int],
        replica_id: int = 0,
    ):
        SimpleBatch._id_counter += 1
        self._id = SimpleBatch._id_counter
        self._replica_id = replica_id
        self._requests = requests
        self._num_tokens = num_tokens
        self._total_num_tokens = sum(num_tokens)
        self._num_prefill_tokens = sum(
            t for r, t in zip(requests, num_tokens) if not r.is_prefill_complete
        )
        self._num_decode_tokens = self._total_num_tokens - self._num_prefill_tokens
        self._total_num_tokens_rounded = (self._total_num_tokens + 7) // 8 * 8
        self._scheduled = False
        self._completed = False

        # Compute prefill and decode request lists
        self._prefill_requests = [r for r in requests if not r.is_prefill_complete]
        self._decode_requests = [r for r in requests if r.is_prefill_complete]

    @property
    def id(self) -> int:
        return self._id

    @property
    def replica_id(self) -> int:
        return self._replica_id

    @property
    def requests(self) -> List[SimpleRequest]:
        return self._requests

    # Vidur may use all_requests
    @property
    def all_requests(self) -> List[SimpleRequest]:
        return self._requests

    @property
    def num_tokens(self) -> List[int]:
        return self._num_tokens

    @property
    def total_num_tokens(self) -> int:
        return self._total_num_tokens

    @property
    def num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens

    @property
    def num_decode_tokens(self) -> int:
        return self._num_decode_tokens

    @property
    def size(self) -> int:
        return len(self._requests)

    @property
    def request_ids(self) -> List[int]:
        return [r.id for r in self._requests]

    @property
    def prefill_requests(self) -> List[SimpleRequest]:
        return self._prefill_requests

    @property
    def decode_requests(self) -> List[SimpleRequest]:
        return self._decode_requests

    @property
    def num_prefill_requests(self) -> int:
        return len(self._prefill_requests)

    @property
    def num_decode_requests(self) -> int:
        return len(self._decode_requests)


@dataclass
class VidurPredictorConfig:
    """Configuration for Vidur execution time predictor."""

    # Model configuration
    model_name: str = "meta-llama/Llama-2-7b-hf"

    # Device configuration
    device: str = "a100"
    network_device: str = "a100_pairwise_nvlink"

    # Parallelism configuration
    tensor_parallel_size: int = 1
    num_pipeline_stages: int = 1

    # Scheduler configuration
    block_size: int = 16
    batch_size_cap: int = 128

    # Predictor configuration
    predictor_type: str = "random_forest"  # "random_forest" or "linear_regression"

    # Profiling data paths (optional, uses defaults if not specified)
    compute_input_file: Optional[str] = None
    attention_input_file: Optional[str] = None
    all_reduce_input_file: Optional[str] = None
    send_recv_input_file: Optional[str] = None
    cpu_overhead_input_file: Optional[str] = None

    # Cache configuration
    cache_dir: str = "cache"

    # Prediction limits
    prediction_max_batch_size: int = 128
    prediction_max_tokens_per_request: int = 4096
    prediction_max_prefill_chunk_size: int = 4096


class VidurStateEvaluate:
    """
    Vidur execution time predictor adapter.

    This class wraps Vidur's execution time predictor to provide the same
    interface as XGBStateEvaluate, enabling seamless integration with
    LLMServingTuner's prediction pipeline.

    Usage:
        evaluator = VidurStateEvaluate(config)
        up, ud = evaluator.predict(input_data)
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VidurStateEvaluate, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: VidurPredictorConfig):
        if VidurStateEvaluate._initialized:
            return

        self.config = config
        self.prefill_type = "prefill"
        self.decode_type = "decode"

        try:
            self.predictor = self._create_predictor()
            VidurStateEvaluate._initialized = True
            logger.info("VidurStateEvaluate initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VidurStateEvaluate: {e}")
            raise

    def _create_predictor(self):
        """Create and initialize the Vidur execution time predictor."""
        try:
            from vidur.config import (
                ReplicaConfig,
                MetricsConfig,
                RandomForrestExecutionTimePredictorConfig,
                LinearRegressionExecutionTimePredictorConfig,
                VllmSchedulerConfig,
            )
            from vidur.execution_time_predictor.execution_time_predictor_registry import (
                ExecutionTimePredictorRegistry,
            )
        except ImportError as e:
            logger.error(f"Failed to import vidur modules: {e}")
            logger.error("Please ensure vidur is installed: pip install -e /path/to/vidur")
            raise ImportError(
                "vidur package not found. Please install it first."
            ) from e

        # Create replica config
        replica_config = ReplicaConfig(
            model_name=self.config.model_name,
            device=self.config.device,
            network_device=self.config.network_device,
            tensor_parallel_size=self.config.tensor_parallel_size,
            num_pipeline_stages=self.config.num_pipeline_stages,
        )

        # Create scheduler config
        scheduler_config = VllmSchedulerConfig(
            block_size=self.config.block_size,
            batch_size_cap=self.config.batch_size_cap,
        )

        # Create metrics config
        metrics_config = MetricsConfig(
            cache_dir=self.config.cache_dir,
            write_metrics=False,
        )

        # Create predictor config based on type
        if self.config.predictor_type == "linear_regression":
            predictor_config = LinearRegressionExecutionTimePredictorConfig()
        else:
            predictor_config = RandomForrestExecutionTimePredictorConfig()

        # Override profiling data paths if specified
        if self.config.compute_input_file:
            predictor_config.compute_input_file = self.config.compute_input_file
        if self.config.attention_input_file:
            predictor_config.attention_input_file = self.config.attention_input_file
        if self.config.all_reduce_input_file:
            predictor_config.all_reduce_input_file = self.config.all_reduce_input_file
        if self.config.send_recv_input_file:
            predictor_config.send_recv_input_file = self.config.send_recv_input_file
        if self.config.cpu_overhead_input_file:
            predictor_config.cpu_overhead_input_file = self.config.cpu_overhead_input_file

        # Set prediction limits
        predictor_config.prediction_max_batch_size = self.config.prediction_max_batch_size
        predictor_config.prediction_max_tokens_per_request = self.config.prediction_max_tokens_per_request
        predictor_config.prediction_max_prefill_chunk_size = self.config.prediction_max_prefill_chunk_size

        # Create predictor using registry
        predictor = ExecutionTimePredictorRegistry.get(
            predictor_config.get_type(),
            predictor_config=predictor_config,
            replica_config=replica_config,
            replica_scheduler_config=scheduler_config,
            metrics_config=metrics_config,
        )

        logger.info(f"Created Vidur predictor: {self.config.predictor_type}")
        logger.info(f"Model: {self.config.model_name}, Device: {self.config.device}")
        logger.info(f"Profiling files - compute: {self.config.compute_input_file}, "
                    f"attention: {self.config.attention_input_file}")

        return predictor

    def _convert_to_batch(self, input_data: "InputData"):
        """
        Convert LLMServingTuner InputData to Vidur Batch for prediction.

        Tries to use Vidur's native Request/Batch classes first, falls back
        to SimpleBatch/SimpleRequest if import fails.

        Args:
            input_data: InputData from LLMServingTuner containing batch and request info

        Returns:
            Batch compatible with Vidur's execution time predictor
        """
        batch_field = input_data.batch_field
        request_fields = input_data.request_field

        is_prefill = batch_field.batch_stage.lower() == self.prefill_type

        # Try to use Vidur's native entities
        try:
            from vidur.entities import Request as VidurRequest, Batch as VidurBatch

            requests = []
            num_tokens = []

            for req_field in request_fields:
                prefill_tokens = req_field.input_length
                decode_tokens = req_field.output_length

                # Create Vidur native Request
                request = VidurRequest(
                    arrived_at=0.0,
                    num_prefill_tokens=prefill_tokens,
                    num_decode_tokens=decode_tokens,
                )

                # If decode stage, mark prefill as complete
                if not is_prefill:
                    request._is_prefill_complete = True
                    request._num_processed_tokens = decode_tokens

                requests.append(request)

                # num_tokens is the number of tokens to process in this batch
                if is_prefill:
                    num_tokens.append(prefill_tokens)
                else:
                    num_tokens.append(1)

            return VidurBatch(
                replica_id=0,
                requests=requests,
                num_tokens=num_tokens,
            )

        except ImportError:
            logger.warning("Could not import Vidur native entities, using SimpleBatch adapter")

        # Fallback to SimpleBatch/SimpleRequest
        requests = []
        num_tokens = []

        for req_field in request_fields:
            prefill_tokens = req_field.input_length
            decode_tokens = req_field.output_length
            is_prefill_complete = not is_prefill

            request = SimpleRequest(
                prefill_tokens=prefill_tokens,
                decode_tokens=decode_tokens,
                is_prefill_complete=is_prefill_complete,
            )
            requests.append(request)

            if is_prefill:
                num_tokens.append(prefill_tokens)
            else:
                num_tokens.append(1)

        return SimpleBatch(requests=requests, num_tokens=num_tokens)

    def predict(self, input_data: "InputData") -> Tuple[float, float]:
        """
        Predict execution time for the given input data.

        Args:
            input_data: InputData containing batch and request information

        Returns:
            Tuple (Up, Ud) where:
                - Up: prefill time in milliseconds (-1 if decode stage)
                - Ud: decode time in milliseconds (-1 if prefill stage)
        """
        stage = input_data.batch_field.batch_stage.lower()

        # Convert input data to batch
        batch = self._convert_to_batch(input_data)

        # Get execution time from Vidur predictor
        # pipeline_stage=0 for single-stage or first stage prediction
        try:
            execution_time = self.predictor.get_execution_time(
                batch=batch,
                pipeline_stage=0,
            )

            # Convert total_time from seconds to milliseconds
            time_ms = execution_time.total_time * 1000

            # Warn if prediction is 0 (likely missing profiling data)
            if time_ms == 0:
                logger.warning(
                    f"Vidur returned 0ms for {stage} stage with "
                    f"{len(batch.requests)} requests, {batch.total_num_tokens} tokens. "
                    "This may indicate missing profiling data files."
                )

        except Exception as e:
            import traceback
            logger.error(f"Vidur prediction failed: {e}")
            logger.error(f"Batch info: stage={stage}, size={batch.size}, "
                        f"total_tokens={batch.total_num_tokens}, "
                        f"prefill_tokens={batch.num_prefill_tokens}")
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            raise  # Re-raise to let caller handle it

        # Return in (Up, Ud) format based on stage
        if stage == self.prefill_type:
            return (time_ms, -1)
        elif stage == self.decode_type:
            return (-1, time_ms)
        else:
            raise ValueError(
                f"Invalid batch stage: {stage}. "
                f"Expected '{self.prefill_type}' or '{self.decode_type}'"
            )

    @classmethod
    def reset(cls):
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None
        cls._initialized = False


def create_vidur_evaluator_from_settings(settings) -> VidurStateEvaluate:
    """
    Factory function to create VidurStateEvaluate from LLMServingTuner settings.

    Args:
        settings: Settings object from llmservingtuner.config.config

    Returns:
        Configured VidurStateEvaluate instance
    """
    latency_config = settings.latency_model

    config = VidurPredictorConfig(
        model_name=getattr(latency_config, 'vidur_model_name', 'meta-llama/Llama-2-7b-hf'),
        device=getattr(latency_config, 'vidur_device', 'a100'),
        network_device=getattr(latency_config, 'vidur_network_device', 'a100_pairwise_nvlink'),
        tensor_parallel_size=getattr(latency_config, 'vidur_tensor_parallel_size', 1),
        num_pipeline_stages=getattr(latency_config, 'vidur_num_pipeline_stages', 1),
        block_size=getattr(latency_config, 'vidur_block_size', 16),
        predictor_type=getattr(latency_config, 'vidur_predictor_type', 'random_forest'),
        compute_input_file=getattr(latency_config, 'vidur_compute_input_file', None),
        attention_input_file=getattr(latency_config, 'vidur_attention_input_file', None),
        all_reduce_input_file=getattr(latency_config, 'vidur_all_reduce_input_file', None),
        send_recv_input_file=getattr(latency_config, 'vidur_send_recv_input_file', None),
        cpu_overhead_input_file=getattr(latency_config, 'vidur_cpu_overhead_input_file', None),
        cache_dir=getattr(latency_config, 'vidur_cache_dir', 'cache'),
        prediction_max_batch_size=getattr(latency_config, 'vidur_prediction_max_batch_size', 128),
        prediction_max_tokens_per_request=getattr(latency_config, 'vidur_prediction_max_tokens_per_request', 4096),
        prediction_max_prefill_chunk_size=getattr(latency_config, 'vidur_prediction_max_prefill_chunk_size', 4096),
    )

    return VidurStateEvaluate(config)
