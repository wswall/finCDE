from __future__ import annotations
from typing import Callable

import numpy as np
from scipy.optimize import Bounds

from density_estimation.common import Array1D, OFFSET
from density_estimation.base import ModelSpec, FitData
from density_estimation.dist import SkewedDistribution, Normal, jsu_constraint


class HarRV(ModelSpec):

    def __init__(self, error_dist=Normal):
        super().__init__(error_dist)
        self.bounds = self._make_bounds()
        self.constraints = self._make_constraints()

    def __call__(self, data: np.ndarray, x: Array1D) -> np.ndarray:
        return x[0] + (x[1:4] * data).sum(axis=1)

    def _make_bounds(self) -> Bounds:
        return Bounds(
            lb=np.array([1e-8, 0, 0, 0, *self.error_dist.bounds.lb]),
            ub=np.array([np.inf, np.inf, np.inf, np.inf, *self.error_dist.bounds.ub]),
        )

    def _stationarity(self, x) -> float:
        return 1 - OFFSET - sum(x[1:4])

    def _make_constraints(self) -> list[dict[str, Callable]]:
        constraints = [{"type": "ineq", "fun": self._stationarity}]
        if self.error_dist.__name__ == "JohnsonSU":
            constraints.append({"type": "ineq", "fun": jsu_constraint})
        return constraints

    def _get_shape_params(self, x: Array1D):
        if self.error_dist.n_params == 0:
            return {"xi": None, "nu": None}
        if self.error_dist.n_params == 1:
            if issubclass(self.error_dist, SkewedDistribution):
                return {"xi": x[-1], "nu": None}
            return {"xi": None, "nu": x[-1]}
        return {"xi": x[-2], "nu": x[-1]}

    def make_initial_guess(self, data):
        sample_var = np.var(data[:, 0], ddof=1)
        return np.array([sample_var, 0.36, 0.28, 0.28, *self.error_dist.initial_guess])

    @property
    def base_step(self):
        return np.array([5 * OFFSET, 0.1, 0.1, 0.1, *self.error_dist.base_step])

    def make_fit_data(self, data: np.ndarray, x: Array1D) -> FitData:
        returns, rv_dmw = data[1:, 0], data[:-1, 1:]
        sigma = self(rv_dmw, x)
        shape_params = self._get_shape_params(x)
        return FitData(returns, returns, sigma, **shape_params)
