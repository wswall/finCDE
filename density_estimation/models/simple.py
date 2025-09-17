from __future__ import annotations

from numba import njit, f8
import numpy as np
from scipy.optimize import Bounds, minimize, OptimizeResult

from density_estimation.common import OFFSET, LLH_SCALING, sumjit
from density_estimation.dist import ConditionalDistribution


@njit(f8[:, :](f8[:, :], f8[:]))
def argarch(out, params):
    """Compute AR(1)-GARCH(1,1) recursion for mean and variance.

    Args:
        out (np.ndarray): Output array to store results.
        params (np.ndarray): Model parameters [mu, phi, omega, alpha, beta].

    Returns:
        np.ndarray: Updated output array with mean and variance columns.
    """
    mu, phi, omega, alpha, beta = params
    returns = out[:, 0].copy()
    out[0, 0] -= mu - phi * mu / (1.0 - phi)
    out[0, 1] = omega / (1.0 - alpha - beta)
    for i in range(1, len(returns)):
        out[i, 0] -= mu + phi * returns[i - 1]
        out[i, 1] += alpha * out[i - 1, 0] ** 2.0 + beta * out[i - 1, 1]
    return out


class ArGarch:
    """AR(1)-GARCH(1,1) model with various error distribution

    Args:
        error_dist (ConditionalDistribution): The distribution of
            standardized residuals.

    Attributes:
        error_dist (ConditionalDistribution): The distribution of
            standardized residuals.
        params (Mapping[str, float]): The parameters of the model.
        bounds (Bounds): The bounds for parameters during optimization.
    """

    _param_lb = [-np.inf, OFFSET - 1.0, OFFSET, 0.0, 0.0]
    _param_ub = [np.inf, 1.0 - OFFSET, np.inf, 1.0 - OFFSET, 1.0 - OFFSET]

    def __init__(self, error_dist: ConditionalDistribution) -> None:
        self.params = {"mu": 0.0, "phi": 0.0, "omega": 0.0, "alpha": 0.0, "beta": 0.0}
        self.error_dist = error_dist
        self.bounds = Bounds(
            lb=np.array([*self._param_lb, *error_dist.bounds.lb], dtype=np.float64),
            ub=np.array([*self._param_ub, *error_dist.bounds.ub], dtype=np.float64),
        )

    @property
    def param_vec(self) -> np.ndarray[np.float64]:
        """Get array of the model's parameter values"""
        return np.array(list(self.params.values()), dtype=np.float64)

    @property
    def _constraints(self) -> list[dict[str, str | callable]]:
        return [
            {"type": "ineq", "fun": lambda x: 1.0 - OFFSET - x[3] - x[4]},
            {
                "type": "ineq",
                "fun": lambda x: 1.0
                - OFFSET
                - 2.0 * x[3] ** 2.0
                - (x[3] + x[4]) ** 2.0,
            },
        ]

    def __call__(self, returns: np.ndarray[float]) -> np.ndarray[float]:
        out = np.array([returns, np.repeat(self.params["omega"], returns.size)]).T
        return argarch(out, self.param_vec)

    def _params_from_vec(self, x):
        return {"mu": x[0], "phi": x[1], "omega": x[2], "alpha": x[3], "beta": x[4]}

    def set_params(self, params: dict[str, float]) -> None:
        self.params.update(params)

    def _init_dist(self, x):
        if self.error_dist.n_params:
            return self.error_dist(*x[len(self.params) :])
        return self.error_dist()

    def _fitness(self, x, *args):
        returns = args[0]
        out = np.array([returns, np.repeat(x[2], returns.size)]).T
        fit_data = argarch(out, x[:5])
        dist = self._init_dist(x)
        sigma = np.sqrt(fit_data[:, 1])
        llh = dist.llh(fit_data[:, 0] / sigma) - sumjit(np.log(sigma))
        return -llh * LLH_SCALING

    def fit(
        self,
        returns: np.ndarray[float],
        initial_guess: np.ndarray[float],
        display: bool = True,
        ftol: float = OFFSET,
        maxiter: int = 100,
    ) -> OptimizeResult:
        """
        Fits the model to the given returns data using optimization.

        The optimization is performed using the Sequential Least Squares
        Programming (SLSQP) method. If the optimization is successful,
        the model parameters are updated, and the error distribution is
        initialized.

        Args:
            returns (np.ndarray[float]): The observed returns data to fit
                the model to.
            initial_guess (np.ndarray[float]): Initial guess for the
                optimization parameters.
            display (bool, optional): Whether to display optimization
                progress. Defaults to True.
            ftol (float, optional): The tolerance for termination by the
                change in the function value. Defaults to OFFSET.
            maxiter (int, optional): The maximum number of iterations for
                the optimizer. Defaults to 100.

        Returns:
            OptimizeResult: The result of the optimization process.
        """

        res = minimize(
            self._fitness,
            initial_guess,
            args=(returns,),
            method="SLSQP",
            bounds=self.bounds,
            constraints=self._constraints,
            options={"disp": display, "ftol": ftol, "maxiter": maxiter},
        )
        if res.success:
            best_params = self._params_from_vec(res.x)
            self.set_params(best_params)
            self.error_dist = self._init_dist(res.x)
        return res
