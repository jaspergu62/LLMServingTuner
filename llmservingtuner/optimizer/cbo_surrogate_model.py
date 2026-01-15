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
Surrogate model using Constrained Bayesian Optimization (CBO) with botorch.
This module provides a staged CBO approach to reduce the number of real evaluations
by using a Gaussian Process surrogate model to predict candidate feasibility.
"""

from typing import Optional, List, Tuple, Dict

from loguru import logger

# Lazy imports for optional dependencies
_BOTORCH_AVAILABLE = None


def _check_botorch():
    """Check if botorch is available."""
    global _BOTORCH_AVAILABLE
    if _BOTORCH_AVAILABLE is None:
        try:
            import torch
            import botorch
            import gpytorch
            _BOTORCH_AVAILABLE = True
        except ImportError:
            _BOTORCH_AVAILABLE = False
    return _BOTORCH_AVAILABLE


class StagedCBO:
    """
    Staged Constrained Bayesian Optimization for candidate filtering.

    Uses a Gaussian Process surrogate model to predict which candidates
    are likely to satisfy constraints, reducing unnecessary real evaluations.
    """

    def __init__(
            self,
            param_dim: int,
            bounds: Tuple[Tuple, Tuple],
            objective_index: int = 0,
            constraint_thresholds: Dict[str, float] = None,
            initial_points: int = 5,
            strategy: str = "conservative",
            feasibility_threshold: float = -1.2,
    ):
        """
        Initialize the CBO surrogate model.

        Args:
            param_dim: Number of parameters to optimize
            bounds: Parameter bounds as ((min1, min2, ...), (max1, max2, ...))
            objective_index: Index of the objective in output array (default: 0 for generate_speed)
            constraint_thresholds: Dict with 'ttft' and/or 'tpot' thresholds
            initial_points: Number of real evaluations before using surrogate
            strategy: Selection strategy ("conservative" or "aggressive")
            feasibility_threshold: Log probability threshold for feasibility
        """
        if not _check_botorch():
            raise ImportError(
                "CBO requires botorch. Install with: pip install llmservingtuner[cbo]"
            )

        import torch

        self.param_dim = param_dim
        self.constraint_thresholds = constraint_thresholds or {}
        self.initial_points = initial_points
        self.strategy = strategy
        self.feasibility_threshold = feasibility_threshold

        self.device = torch.device("cpu")
        self.dtype = torch.float64
        self.bounds = torch.stack([
            torch.tensor(_item, device=self.device, dtype=self.dtype)
            for _item in bounds
        ])

        self.objective_index = objective_index
        self.constraint_indices = []
        self.constraint_values = []
        self.token_ratio = None

        if 'ttft' in constraint_thresholds:
            self.constraint_indices.append(1)
            self.constraint_values.append(constraint_thresholds['ttft'])
        if 'tpot' in constraint_thresholds:
            self.constraint_indices.append(2)
            self.constraint_values.append(constraint_thresholds['tpot'])

        self.constraint_values = torch.tensor(self.constraint_values, dtype=self.dtype)
        self.num_outputs = 3

        # State
        self.train_X_real = torch.empty((0, param_dim), device=self.device, dtype=self.dtype)
        self.train_Y_real = torch.empty((0, self.num_outputs), device=self.device, dtype=self.dtype)
        self.throughput_real = torch.empty((0, 1), device=self.device, dtype=self.dtype)
        self.model = None
        self.is_model_ready = False
        self.num_feasible_points = 0
        self.num_real_evaluations = 0

    def add_real_observation(self, x_new, y_new):
        """
        Add a new real observation to the training data.

        Args:
            x_new: Parameter values tensor (1, param_dim)
            y_new: Metrics tensor (1, 3+) - [generate_speed, ttft, tpot, ...]
        """
        import torch

        x_new_d = x_new.to(self.device, self.dtype)
        y_new_d = y_new.to(self.device, self.dtype)

        self.train_X_real = torch.cat([self.train_X_real, x_new_d])
        self.train_Y_real = torch.cat([self.train_Y_real, y_new_d[:, :3]])
        self.throughput_real = torch.cat([self.throughput_real, y_new_d[:, -1:]])

        self.num_real_evaluations += 1

        # Count feasible points
        is_feasible = True
        for i, constraint_idx in enumerate(self.constraint_indices):
            if y_new_d[0, constraint_idx] > self.constraint_values[i]:
                is_feasible = False
                break

        if is_feasible:
            self.num_feasible_points += 1

    def update_surrogate_model(self):
        """Update the Gaussian Process surrogate model with current observations."""
        import torch
        import gpytorch
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP, ModelListGP
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize
        from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

        if self.num_real_evaluations < self.initial_points:
            logger.warning(
                f"Not enough data to build model yet. "
                f"{self.num_real_evaluations}/{self.initial_points}"
            )
            return

        self.token_ratio = self._calculate_token_ratio()
        logger.debug(f"Updating surrogate model with {len(self.train_X_real)} points...")

        models = []
        train_obj = 1.0 / (self.train_Y_real[:, self.objective_index] + 1e-9)

        all_ys = [train_obj]  # Output 0: 1/generate_speed
        all_ys.append(self.train_Y_real[:, 1])  # Output 1: ttft
        all_ys.append(self.train_Y_real[:, 2])  # Output 2: tpot

        for i, train_y in enumerate(all_ys):
            model_i = SingleTaskGP(
                self.train_X_real,
                train_y.unsqueeze(-1),
                input_transform=Normalize(d=self.param_dim, bounds=self.bounds),
                outcome_transform=Standardize(m=1),
            )
            mll_i = ExactMarginalLogLikelihood(model_i.likelihood, model_i)
            try:
                with gpytorch.settings.cholesky_jitter(1e-4):
                    fit_gpytorch_mll(mll_i, max_retries=3)
            except Exception as e:
                logger.error(f"Model fitting for output {i} failed: {e}. Using prior.")

            models.append(model_i)

        self.model = ModelListGP(*models)
        self.is_model_ready = True
        logger.debug("Surrogate model updated successfully.")

    def select_candidates(self, x_candidates) -> Tuple[List[int], Optional[tuple]]:
        """
        Select candidates likely to satisfy constraints.

        Args:
            x_candidates: Tensor of candidate parameter sets

        Returns:
            Tuple of (selected_indices, predictions or None)
        """
        import torch
        from botorch.acquisition.analytic import LogProbabilityOfFeasibility

        num_candidates = x_candidates.shape[0]

        if not self.is_model_ready or self.model is None:
            logger.debug("Surrogate model not ready; keeping all candidates")
            return list(range(num_candidates)), None

        x_candidates_d = x_candidates.to(self.device, self.dtype)

        constraints_dict = {}
        if 'ttft' in self.constraint_thresholds:
            constraints_dict[1] = [None, self.constraint_thresholds['ttft']]
        if 'tpot' in self.constraint_thresholds:
            constraints_dict[2] = [None, self.constraint_thresholds['tpot']]

        try:
            acqf = LogProbabilityOfFeasibility(model=self.model, constraints=constraints_dict)
            acqf_scores = acqf(x_candidates_d.unsqueeze(1)).squeeze()
            feasible_mask = acqf_scores > self.feasibility_threshold

            if feasible_mask.sum() == 0:
                logger.warning("No feasible candidates; selecting top scores")
                num_to_select = max(1, int(0.6 * num_candidates))
                sorted_scores, sorted_indices = torch.sort(acqf_scores, descending=True)
                selected_indices = sorted_indices[:num_to_select].tolist()
                logger.info(f"Selected top {num_to_select}/{num_candidates} candidates (60% fallback)")
            else:
                selected_indices = torch.where(feasible_mask)[0].tolist()
                logger.debug(
                    f"Feasible candidates: {len(selected_indices)}/{num_candidates} "
                    f"(threshold={self.feasibility_threshold})"
                )
        except Exception as e:
            logger.error(f"Error in candidate selection: {e}. Selecting all candidates.")
            selected_indices = list(range(num_candidates))

        # Get predictions
        predictions = self._get_predictions(x_candidates_d, num_candidates)
        return selected_indices, predictions

    def _get_predictions(self, x_candidates_d, num_candidates):
        """Get predictions from the surrogate model."""
        import torch

        try:
            with torch.no_grad():
                posterior = self.model.posterior(x_candidates_d)
                pred_mean = posterior.mean
                pred_var = posterior.variance
                pred_std = pred_var.sqrt()

            pred_y_mean_orig = torch.zeros(num_candidates, 3, dtype=self.dtype)
            pred_y_std_orig = torch.zeros(num_candidates, 3, dtype=self.dtype)

            # Convert from 1/speed back to speed using Delta method
            mu_inv_speed = pred_mean[:, 0]
            var_inv_speed = pred_var[:, 0]
            pred_y_mean_orig[:, 0] = 1.0 / (mu_inv_speed + 1e-9)
            pred_y_std_orig[:, 0] = torch.sqrt(var_inv_speed) / (mu_inv_speed.pow(2) + 1e-9)

            # Constraints
            pred_y_mean_orig[:, 1] = pred_mean[:, 1]  # ttft
            pred_y_mean_orig[:, 2] = pred_mean[:, 2]  # tpot
            pred_y_std_orig[:, 1] = pred_std[:, 1]
            pred_y_std_orig[:, 2] = pred_std[:, 2]

            return (pred_y_mean_orig, pred_y_std_orig)
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return None

    def _calculate_token_ratio(self):
        """Calculate average token ratio from observations."""
        import torch

        try:
            if len(self.train_Y_real) == 0 or len(self.throughput_real) == 0:
                logger.warning("No data. Returning default value.")
                return torch.tensor(1.0, device=self.device, dtype=self.dtype)

            generate_speed_mean = self.train_Y_real[:, self.objective_index].mean()
            throughput_mean = self.throughput_real.mean()

            if torch.isclose(throughput_mean, torch.tensor(0.0, dtype=throughput_mean.dtype)):
                logger.warning("Throughput mean is zero")
                return torch.tensor(1.0, device=self.device, dtype=self.dtype)

            ratio = generate_speed_mean / throughput_mean
            return ratio.item()
        except Exception as e:
            logger.warning(f"Error calculating token ratio: {e}")
            return None

    def get_token_ratio(self):
        """Get the calculated token ratio."""
        return self.token_ratio

    def get_model_info(self) -> Dict:
        """Get information about the current model state."""
        return {
            'num_real_evaluations': self.num_real_evaluations,
            'num_feasible_points': self.num_feasible_points,
            'is_model_ready': self.is_model_ready,
            'constraint_thresholds': self.constraint_thresholds,
            'constraint_indices': self.constraint_indices,
            'feasibility_threshold': self.feasibility_threshold,
            'train_X_shape': self.train_X_real.shape if len(self.train_X_real) > 0 else (0, self.param_dim),
            'train_Y_shape': self.train_Y_real.shape if len(self.train_Y_real) > 0 else (0, self.num_outputs),
        }
