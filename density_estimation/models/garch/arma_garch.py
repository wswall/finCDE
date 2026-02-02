from __future__ import annotations

from typing import Literal, Callable

import numpy as np
from numpy.polynomial import polynomial as poly
from scipy.optimize import Bounds
from statsmodels.regression.linear_model import yule_walker

from density_estimation.base import ModelSpec, FitData
from density_estimation.common import OFFSET, Array1D
from density_estimation.dist import (
    Distribution,
    SkewedDistribution,
    Normal,
    jsu_constraint,
)
from density_estimation.models.garch import functions as gfunc


class ArmaGarch(ModelSpec):

    def __init__(
        self,
        arma_order: tuple[int, int] = (1, 1),
        garch_order: tuple[int, int] = (1, 1),
        error_dist: Distribution = Normal,
    ):
        super().__init__(error_dist)
        self.arma_order = arma_order
        self.garch_order = garch_order
        if np.any(self.order < 0):
            raise ValueError("All ARMA and GARCH order terms must be nonnegative")
        self.bounds = self._make_bounds()
        self.constraints = self._make_constraints()
        # Slices to recover params from vector during optimization
        self._vec_slices = self._make_slice_dict()
        self._arma_eq = self._get_arma_eq()
        self._garch_eq = self._get_garch_eq()

    def __call__(self, data: Array1D, x: Array1D):
        params = self.param_vec_to_dict(x)
        residuals = self._arma_eq(
            data, data.mean(), params["mu"][0], params["phi"], params["theta"]
        )
        variance = self._garch_eq(
            residuals,
            data.var(ddof=1),
            params["omega"][0],
            params["alpha"],
            params["beta"],
        )
        return np.column_stack((residuals, variance))

    def _get_arma_eq(self):
        if self.arma_order == (1, 0):
            return gfunc.calc_ar_1
        if self.arma_order[0] > 1 and self.arma_order[1] == 0:
            return gfunc.calc_ar_m
        if self.arma_order == (1, 1):
            return gfunc.calc_arma_11
        return gfunc.calc_arma_mn

    def _get_garch_eq(self):
        if self.garch_order == (1, 1):
            return gfunc.calc_garch_11
        return gfunc.calc_garch_pq

    def make_initial_guess(self, data):
        ar_initial = yule_walker(data, order=self.arma_order[0])[0]
        return np.array(
            [
                (1 - ar_initial.sum()) * np.mean(data),
                *ar_initial,
                *np.repeat(-0.1, self.arma_order[1]),
                0.05 * np.var(data, ddof=1),
                *np.repeat(0.05 / self.garch_order[0], self.garch_order[0]),
                *np.repeat(0.9 / self.garch_order[1], self.garch_order[1]),
                *self.error_dist.initial_guess,
            ]
        )

    @property
    def base_step(self):
        return np.array(
            [
                0.2,
                *np.repeat(0.1, self.arma_order[0]),
                *np.repeat(0.1, self.arma_order[1]),
                5 * OFFSET,
                *np.repeat(0.1, self.garch_order[0]),
                *np.repeat(0.1, self.garch_order[1]),
                *self.error_dist.base_step,
            ]
        )

    @property
    def order(self) -> np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.int_]]:
        """Return the order of the ARMA and GARCH parts."""
        return np.array([self.arma_order, self.garch_order])

    def _make_slice_dict(self):
        order = self.order
        n_arma_terms = 1 + order[0].sum()
        n_model_terms = 2 + order.sum()
        return {
            "mu": slice(0, 1),
            "phi": slice(1, 1 + order[0, 0]),
            "theta": slice(1 + order[0, 0], n_arma_terms),
            "omega": slice(n_arma_terms, 1 + n_arma_terms),
            "alpha": slice(1 + n_arma_terms, 1 + n_arma_terms + order[1, 0]),
            "beta": slice(1 + n_arma_terms + order[1, 0], n_model_terms),
        }

    def param_vec_to_dict(self, x: Array1D) -> dict[str, np.ndarray]:
        return {param: x[slc] for param, slc in self._vec_slices.items()}

    def _make_bounds(self) -> Bounds:
        arma_bound = np.repeat(np.inf, self.order[0].sum())
        garch_low = np.repeat(0.0, self.order[1].sum())
        garch_high = np.repeat(np.inf, self.order[1].sum())
        return Bounds(
            lb=[-np.inf, *-arma_bound, OFFSET, *garch_low, *self.error_dist.bounds.lb],
            ub=[np.inf, *arma_bound, np.inf, *garch_high, *self.error_dist.bounds.ub],
        )

    def _ar_stationarity(self, x: Array1D):
        phi = x[self._vec_slices["phi"]]
        if phi.size == 1:
            return 1 - OFFSET - np.abs(phi[0])
        if phi.size == 2:
            return np.min(
                [1 - OFFSET - np.abs(phi[0] + phi[1]), 1 - OFFSET - np.abs(phi[1])]
            )
        char_poly = poly.Polynomial(-np.array([-1, *phi]))
        abs_roots = np.abs(char_poly.roots())
        if abs_roots.size == 0:
            return -1.0
        return np.min(abs_roots) - 1.0 - OFFSET

    def _garch_stationarity(self, x: Array1D):
        alpha, beta = x[self._vec_slices["alpha"]], x[self._vec_slices["beta"]]
        if alpha.size == 1 and beta.size == 1:
            return 1 - OFFSET - np.abs(alpha[0] + beta[0])
        char_poly = poly.Polynomial(-np.array([-1, *alpha])) - poly.Polynomial(
            np.array([0, *beta])
        )
        abs_roots = np.abs(char_poly.roots())
        if abs_roots.size == 0:
            return -1.0
        return np.min(abs_roots) - 1.0 - OFFSET

    def _make_constraints(self) -> list[dict[str, Callable]]:
        constraints = [
            {"type": "ineq", "fun": self._ar_stationarity},
            {"type": "ineq", "fun": self._garch_stationarity},
        ]
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

    def make_fit_data(self, data, x):
        pred_vals = self(data, x)
        sigma = np.sqrt(np.maximum(1e-8, pred_vals[:, 1]))
        shape_params = self._get_shape_params(x)
        return FitData(data, pred_vals[:, 0], sigma, **shape_params)
