from __future__ import annotations
from typing import Callable

import numpy as np
from scipy.optimize import Bounds

from density_estimation.common import Array1D, OFFSET
from density_estimation.core import ModelSpec, FitData, Distribution, SkewedDistribution
from density_estimation.distributions import Normal


class HarRV(ModelSpec):
    """Heterogeneous Autoregressive (HAR) model for Realized Volatility.

    Models the conditional volatility using Daily, Weekly, and Monthly
    realized volatility components.

    Attributes:
        bounds (scipy.optimize.Bounds): Parameter bounds for optimization.
        constraints (list[dict]): Optimization constraints.
    """

    def __init__(self, error_dist: Distribution = Normal):
        """Initializes the HAR-RV model specification.

        Args:
            error_dist (Distribution, optional): The error distribution class to use.
                Defaults to Normal.
        """
        super().__init__(error_dist)
        self.bounds = self._make_bounds()
        self.constraints = self._make_constraints()

    def __call__(self, data: np.ndarray, x: Array1D) -> np.ndarray:
        """Calculates conditional volatility given data and parameters.

        Args:
            data (np.ndarray): Input data array where columns represent the Daily,
                Weekly, and Monthly realized volatility components.
            x (Array1D): Model parameters array [intercept, beta_daily, beta_weekly,
                beta_monthly, ...distribution_parameters].

        Returns:
            np.ndarray: The calculated conditional volatility series.
        """
        return x[0] + (x[1:4] * data).sum(axis=1)

    def _make_bounds(self) -> Bounds:
        return Bounds(
            lb=np.array([1e-8, 0, 0, 0, *self.error_dist.bounds.lb]),
            ub=np.array([np.inf, np.inf, np.inf, np.inf, *self.error_dist.bounds.ub]),
        )

    def _stationarity(self, x: Array1D) -> float:
        return 1 - OFFSET - sum(x[1:4])

    def _make_constraints(self) -> list[dict[str, Callable]]:
        return [{"type": "ineq", "fun": self._stationarity}]

    def _get_shape_params(self, x: Array1D) -> dict[str, float | None]:
        if self.error_dist.n_params == 0:
            return {"xi": None, "nu": None}
        if self.error_dist.n_params == 1:
            if issubclass(self.error_dist, SkewedDistribution):
                return {"xi": x[-1], "nu": None}
            return {"xi": None, "nu": x[-1]}
        return {"xi": x[-2], "nu": x[-1]}

    def make_initial_guess(self, data: np.ndarray) -> np.ndarray:
        """Generates an initial guess for the model parameters based on data.

        Args:
            data (np.ndarray): The data used for estimation. The first column
                is expected to be returns.

        Returns:
            np.ndarray: Initial parameter guess array.
        """
        sample_var = np.var(data[:, 0], ddof=1)
        return np.array([sample_var, 0.36, 0.28, 0.28, *self.error_dist.initial_guess])

    @property
    def base_step(self) -> np.ndarray:
        """Returns the base step size for numerical differentiation during optimization.

        Returns:
            np.ndarray: Array of step sizes corresponding to model parameters.
        """
        return np.array([5 * OFFSET, 0.1, 0.1, 0.1, *self.error_dist.base_step])

    def make_fit_data(self, data: np.ndarray, x: Array1D) -> FitData:
        """Constructs a FitData object for model evaluation.

        Splits the data into returns and realized volatility components, calculates
        the conditional volatility, and extracts distribution shape parameters.

        Args:
            data (np.ndarray): Input data matrix.
            x (Array1D): Model parameters.

        Returns:
            FitData: Data object ready for likelihood computation.
        """
        returns, rv_dmw = data[1:, 0], data[:-1, 1:]
        sigma = self(rv_dmw, x)
        shape_params = self._get_shape_params(x)
        return FitData(returns, returns, sigma, **shape_params)
