from __future__ import annotations

from typing import Literal, Callable

import numpy as np
from numpy.polynomial import polynomial as poly
from scipy.optimize import Bounds
from statsmodels.regression.linear_model import yule_walker

from density_estimation.common import OFFSET, Array1D
from density_estimation.core import ModelSpec, FitData, Distribution, SkewedDistribution
from density_estimation.distributions import Normal
from density_estimation.models.garch import functions as gfunc


class ArmaGarch(ModelSpec):
    """ARMA-GARCH model specification.

    Models the conditional mean using an ARMA(p,q) process and the conditional
    variance using a GARCH(p,q) process.

    Attributes:
        arma_order (tuple[int, int]): Order (p, q) of the ARMA process.
        garch_order (tuple[int, int]): Order (p, q) of the GARCH process.
        bounds (scipy.optimize.Bounds): Parameter bounds for optimization.
        constraints (list[dict]): Optimization constraints.
    """

    def __init__(
        self,
        arma_order: tuple[int, int] = (1, 1),
        garch_order: tuple[int, int] = (1, 1),
        error_dist: Distribution = Normal,
    ):
        """Initializes the ARMA-GARCH model specification.

        Args:
            arma_order (tuple[int, int], optional): Order of AR and MA components.
                Defaults to (1, 1).
            garch_order (tuple[int, int], optional): Order of GARCH components.
                Defaults to (1, 1).
            error_dist (Distribution, optional): The error distribution class.
                Defaults to Normal.
        """
        super().__init__(error_dist)
        self.arma_order = arma_order
        self.garch_order = garch_order
        if np.any(self.order < 0):
            raise ValueError("All ARMA and GARCH order terms must be nonnegative")
        # Dict of slices to recover params from vectors
        self._vec_slices = self._make_slice_dict()
        self._arma_eq = self._get_arma_eq()
        self._garch_eq = self._get_garch_eq()
        self.bounds = self._make_bounds()
        self.constraints = self._make_constraints()

    def __call__(self, data: Array1D, x: Array1D) -> np.ndarray:
        """Calculates residuals and conditional variance.

        Args:
            data (Array1D): Input time series data.
            x (Array1D): Model parameters.

        Returns:
            np.ndarray: Array with columns [residuals, variance].
        """
        params = self.make_param_dict(x)
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

    def __getstate__(self):
        # Constraint callables are lambdas, can't be serialized for multiprocessing
        state = self.__dict__.copy()
        if "constraints" in state:
            del state["constraints"]
        return state

    def _get_arma_eq(self) -> Callable:
        # Select ARMA calculation function based on the specified order
        if self.arma_order == (1, 0):
            return gfunc.calc_ar_1
        if self.arma_order[0] > 1 and self.arma_order[1] == 0:
            return gfunc.calc_ar_m
        if self.arma_order == (1, 1):
            return gfunc.calc_arma_11
        return gfunc.calc_arma_mn

    def _get_garch_eq(self) -> Callable:
        # Select GARCH calculation function based on the specified order
        if self.garch_order == (1, 1):
            return gfunc.calc_garch_11
        return gfunc.calc_garch_pq

    def make_initial_guess(self, data) -> np.ndarray:
        """Generates an initial guess for model parameters.

        Uses Yule-Walker equations to initialize the AR parameters and
        heuristics for others.

        Args:
            data (np.ndarray): The data used for estimation.

        Returns:
            np.ndarray: Initial parameter guess array.
        """
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
    def base_step(self) -> np.ndarray:
        """Base step size for numerical differentiation."""
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

    def _make_slice_dict(self) -> dict[str, slice]:
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

    def make_param_dict(self, x: Array1D) -> dict[str, np.ndarray]:
        """Converts parameter array to a dictionary of parameters.

        Args:
            x (Array1D): Flat array of model parameters.

        Returns:
            dict[str, np.ndarray]: Dictionary with keys 'mu', 'phi', 'theta',
                'omega', 'alpha', 'beta'.
        """
        return {param: x[slc] for param, slc in self._vec_slices.items()}

    def _phi_bound(self) -> tuple[np.ndarray, np.ndarray]:
        if self.arma_order[0] == 1:
            bound = np.array([1 - OFFSET])
        elif self.arma_order[0] == 2:
            bound = np.array([np.inf, 1 - OFFSET])
        else:
            bound = np.repeat(np.inf, self.order[0, 0])
        return -bound, bound

    def _theta_bound(self) -> tuple[np.ndarray, np.ndarray]:
        if self.arma_order[1] == 1:
            bound = np.array([1 - OFFSET])
        elif self.arma_order[1] == 2:
            bound = np.array([np.inf, 1 - OFFSET])
        else:
            bound = np.repeat(np.inf, self.order[0, 1])
        return -bound, bound

    def _make_bounds(self) -> Bounds:
        phi_lb, phi_ub = self._phi_bound()
        theta_lb, theta_ub = self._theta_bound()
        garch_low = np.repeat(0.0, self.order[1].sum())
        garch_high = np.repeat(np.inf, self.order[1].sum())
        return Bounds(
            lb=[
                -np.inf,
                *phi_lb,
                *theta_lb,
                OFFSET,
                *garch_low,
                *self.error_dist.bounds.lb,
            ],
            ub=[
                np.inf,
                *phi_ub,
                *theta_ub,
                np.inf,
                *garch_high,
                *self.error_dist.bounds.ub,
            ],
        )

    def _arma_root(self, coeffs: np.ndarray) -> np.ndarray:
        char_poly = poly.Polynomial(-np.array([-1, *coeffs]))
        roots = np.abs(char_poly.roots())
        if roots.size == 0:
            return np.array([0.0])
        return np.array([np.min(roots)])

    def _ar_stationarity(self) -> Callable:
        phi_slc = self._vec_slices["phi"]
        if self.arma_order[0] == 2:
            return lambda x: 1 - OFFSET - np.abs(x[phi_slc][0] + x[phi_slc][1])
        return lambda x: self._arma_root(x[phi_slc]) - 1 - OFFSET

    def _ma_invertibility(self) -> Callable:
        t_slc = self._vec_slices["theta"]
        if self.arma_order[1] == 2:
            return lambda x: 1 - OFFSET - np.abs(x[t_slc][0] + x[t_slc][1])
        return lambda x: self._arma_root(x[t_slc]) - 1 - OFFSET

    def _garch_roots(self, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        alpha_poly = poly.Polynomial(-np.array([-1, *alpha]))
        beta_poly = poly.Polynomial(np.array([0, *beta]))
        char_poly = alpha_poly - beta_poly
        roots = np.abs(char_poly.roots())
        if roots.size == 0:
            return np.array([0.0])
        return np.array([np.min(roots)])

    def _garch_stationarity(self) -> Callable:
        a_slc, b_slc = self._vec_slices["alpha"], self._vec_slices["beta"]
        if self.garch_order == (1, 1):
            return lambda x: 1 - OFFSET - np.abs(x[a_slc] + x[b_slc])
        return lambda x: self._garch_roots(x[a_slc], x[b_slc]) - 1 - OFFSET

    def _make_constraints(self) -> list[dict[str, Callable]]:
        constraints = [{"type": "ineq", "fun": self._garch_stationarity()}]
        if self.arma_order[0] > 1:
            constraints.append({"type": "ineq", "fun": self._ar_stationarity()})
        if self.arma_order[1] > 1:
            constraints.append({"type": "ineq", "fun": self._ma_invertibility()})
        return constraints

    def _get_shape_params(self, x: Array1D):
        if self.error_dist.n_params == 0:
            return {"xi": None, "nu": None}
        if self.error_dist.n_params == 1:
            if issubclass(self.error_dist, SkewedDistribution):
                return {"xi": x[-1], "nu": None}
            return {"xi": None, "nu": x[-1]}
        return {"xi": x[-2], "nu": x[-1]}

    def make_fit_data(self, data: np.ndarray, x: Array1D) -> FitData:
        """Constructs a FitData object for model evaluation.

        Args:
            data (np.ndarray): Input data array.
            x (Array1D): Model parameters.

        Returns:
            FitData: Data object ready for likelihood computation.
        """
        pred_vals = self(data, x)
        sigma = np.sqrt(np.maximum(1e-8, pred_vals[:, 1]))
        shape_params = self._get_shape_params(x)
        return FitData(data, pred_vals[:, 0], sigma, **shape_params)
