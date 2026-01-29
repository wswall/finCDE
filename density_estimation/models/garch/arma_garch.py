from __future__ import annotations

from typing import Literal, Callable

from numba import njit, prange, f8, typed, types
import numpy as np
from numpy.polynomial import polynomial as poly
from scipy.optimize import Bounds

from density_estimation.base import ModelSpec, FitData
from density_estimation.common import OFFSET, Array1D
from density_estimation.dist import Distribution, SkewedDistribution, Normal


param_dict_sig = types.DictType(types.unicode_type, types.float64[:])


@njit(f8[:, :](f8[:], param_dict_sig), fastmath=True)
def calc_armagarch(returns, params):
    mu, phi, theta = params["mu"][0], params["phi"], params["theta"]
    omega, alpha, beta = params["omega"][0], params["alpha"], params["beta"]
    order = np.array([[phi.size, theta.size], [alpha.size, beta.size]])
    output = np.empty((returns.shape[0], 2), dtype=np.float64)
    y_init = mu / (1.0 - phi.sum() - theta.sum())
    output[0, 0] = returns[0] - mu - phi.sum() * y_init
    for i in range(1, order.max()):
        y_pred = mu
        for n in prange(phi.size):
            if n < i:
                y_pred += phi[n] * returns[i - 1 - n]
            else:
                y_pred += phi[n] * y_init
        for n in prange(theta.size):
            if n < i:
                y_pred += theta[n] * output[i - 1 - n, 0]
        output[i, 0] = returns[i] - y_pred
    output[: order.max(), 1] = omega / (1.0 - alpha.sum() - beta.sum())
    for i in range(order.max(), returns.shape[0]):
        y_pred = mu
        var_pred = omega
        for n in prange(phi.size):
            y_pred += phi[n] * returns[i - 1 - n]
        for n in prange(theta.size):
            y_pred += theta[n] * output[i - 1 - n, 0]
        for n in prange(alpha.size):
            var_pred += alpha[n] * output[i - 1 - n, 0] ** 2
        for n in prange(beta.size):
            var_pred += beta[n] * output[i - 1 - n, 1]
        output[i, 0] = returns[i] - y_pred
        output[i, 1] = var_pred
    return output


@njit(f8[:, :](f8[:], param_dict_sig), fastmath=True)
def calc_1111(returns, params):
    mu, phi, theta = params["mu"][0], params["phi"][0], params["theta"][0]
    omega, alpha, beta = params["omega"][0], params["alpha"][0], params["beta"][0]
    output = np.zeros((returns.shape[0], 2), dtype=np.float64)
    output[0, 0] = returns[0] - mu - phi * mu / (1.0 - phi - theta)
    output[0, 1] = omega / (1.0 - alpha - beta)
    for i in range(1, returns.shape[0]):
        output[i, 0] = returns[i] - mu - phi * returns[i - 1] - theta * output[i - 1, 0]
        output[i, 1] = omega + alpha * output[i - 1, 0] ** 2 + beta * output[i - 1, 1]
    return output


@njit(f8[:, :](f8[:], param_dict_sig), fastmath=True)
def calc_1011(returns, params):
    mu, phi = params["mu"][0], params["phi"][0]
    omega, alpha, beta = params["omega"][0], params["alpha"][0], params["beta"][0]
    output = np.zeros((returns.shape[0], 2), dtype=np.float64)
    output[0, 0] = returns[0] - mu - phi * mu / (1.0 - phi)
    output[0, 1] = omega / (1.0 - alpha - beta)
    for i in range(1, returns.shape[0]):
        output[i, 0] = returns[i] - mu - phi * returns[i - 1]
        output[i, 1] = omega + alpha * output[i - 1, 0] ** 2 + beta * output[i - 1, 1]
    return output


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
        self._set_call_func()

    def __call__(self, data: Array1D, x: Array1D):
        return calc_armagarch(data, self.vec_to_parameters(x))

    @property
    def initial_guess(self):
        return np.array(
            [
                1e-3,
                *np.repeat(OFFSET, self.arma_order[0]),
                *np.repeat(OFFSET, self.arma_order[1]),
                1e-5,
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

    def _set_call_func(self):
        if self.arma_order == (1, 0) and self.garch_order == (1, 1):
            self._call_func = calc_1011
        if self.arma_order == (1, 1) and self.garch_order == (1, 1):
            self._call_func = calc_1111
        else:
            self._call_func = calc_armagarch

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

    def vec_to_parameters(self, x: Array1D) -> types.DictType:
        d = typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])
        d.update({param: x[slc] for param, slc in self._vec_slices.items()})
        return d

    def _make_bounds(self) -> Bounds:
        arma_bound = np.repeat(np.inf, self.order[0].sum())
        garch_low = np.repeat(0.0, self.order[1].sum())
        garch_high = np.repeat(np.inf, self.order[1].sum())
        return Bounds(
            lb=[-np.inf, *-arma_bound, OFFSET, *garch_low, *self.error_dist.bounds.lb],
            ub=[np.inf, *arma_bound, np.inf, *garch_high, *self.error_dist.bounds.ub],
        )

    def _arma_moment(self, x: Array1D):
        phi, theta = x[self._vec_slices["phi"]], x[self._vec_slices["theta"]]
        return np.abs(1 - (phi.sum() + theta.sum()))

    def _ar_stationarity(self, x: Array1D):
        phi = x[self._vec_slices["phi"]]
        if phi.size == 1:
            return 1 - OFFSET - np.abs(phi[0])
        if phi.size == 2:
            return np.array(
                [1 - OFFSET - np.abs(phi[0] + phi[1]), 1 - OFFSET - np.abs(phi[1])]
            )
        if np.all(phi == 0):
            return np.array([-1.0])
        ar_char_poly = poly.Polynomial(-np.array([-1, *phi]))
        return np.abs(ar_char_poly.roots()) - 1.0 - OFFSET

    def _garch_stationarity(self, x: Array1D):
        alpha, beta = x[self._vec_slices["alpha"]], x[self._vec_slices["beta"]]
        if alpha.size == 1 and beta.size == 1:
            return 1 - OFFSET - np.abs(alpha + beta)
        alpha_poly = poly.Polynomial(-np.array([-1, *alpha]))
        beta_poly = poly.Polynomial(np.array([0, *beta]))
        char_poly = alpha_poly - beta_poly
        return np.abs(char_poly.roots()) - 1.0 - OFFSET

    def _garch_fourth_moment(self, x: Array1D) -> np.ndarray:
        alpha, beta = x[self._vec_slices["alpha"]], x[self._vec_slices["beta"]]
        return 1.0 - OFFSET - (3 * alpha**2 + 2 * (alpha + beta) + beta**2.0)

    def _make_constraints(self) -> list[dict[str, Callable]]:
        constraints = [
            {"type": "ineq", "fun": self._ar_stationarity},
            {"type": "ineq", "fun": self._arma_moment},
            {"type": "ineq", "fun": self._garch_stationarity},
        ]
        if np.all(self.order[1] == 1):
            constraints.append({"type": "ineq", "fun": self._garch_fourth_moment})
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
        residuals, sigma = pred_vals[:, 0], np.sqrt(pred_vals[:, 1])
        shape_params = self._get_shape_params(x)
        return FitData(data, residuals, sigma, **shape_params)
