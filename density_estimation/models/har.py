from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize, Bounds, OptimizeResult

from density_estimation.common import sumjit, OFFSET, LLH_SCALING
from density_estimation.dist import ConditionalDistribution


class HarRV:
    """Heterogeneous Autoregressive Realized Volatility model."""

    def __init__(self, error_dist: ConditionalDistribution):
        self.params = {"c": 0.0, "beta": np.zeros(3)}
        self.error_dist = error_dist

    def _init_dist(self, x: Sequence[float]) -> ConditionalDistribution:
        if self.error_dist.n_params:
            return self.error_dist(*x[4:])
        return self.error_dist()

    def set_params(self, x: Sequence[float]) -> None:
        """Set HAR model parameters from a sequence.

        Args:
            x (Sequence[float]): Flat array of parameters consisting of
                constant, RV coefficients, and distribution parameters.
        """
        self.params["c"] = x[0]
        self.params["beta"][:] = x[1:4]

    def calc_var(self, rv_data: np.ndarray) -> np.ndarray:
        """Calculate the conditional variance from input data.

        Args:
            rv_data (np.ndarray): Array of T x 3 containing daily, 
                weekly, and monthly realized volatilities.

        Returns:
            np.ndarray: The calculated conditional variance.
        """
        beta_d, beta_w, beta_m = self.params["beta"]
        return (
            self.params["c"]
            + beta_d * rv_data[:, 0]
            + beta_w * rv_data[:, 1]
            + beta_m * rv_data[:, 2]
        )

    def _fitness(self, x: Sequence[float], *args: Any) -> float:
        data = args[0]
        returns, rv_dmw = data[1:, 0], data[:-1, 1:]
        sigma = self.calc_var(rv_dmw)
        dist = self._init_dist(x)
        llh = dist.llh(returns / sigma) - sumjit(np.log(sigma))
        return -llh * LLH_SCALING

    def fit(
        self,
        data: np.ndarray,
        initial_guess: np.ndarray,
        display: bool = True,
        ftol: float = OFFSET,
        maxiter: int = 100,
    ):
        """Fit the model to the historic data using MLE.

        Args:
            data (np.ndarray): Array of shape T X 4 containing the log
                returns and daily, weekly, and monthly realized
                volatilities.
            initial_guess (np.ndarray): Initial guesses for each of the
                model and distribution parameters.
            display (bool, optional): Whether to display optimization
                output. Defaults to True.
            ftol (float, optional): Tolerance for termination. Defaults
                to OFFSET in common.py.
            maxiter (int, optional): Maximum number of iterations.
                Defaults to 100.

        Returns:
            OptimizeResult: Scipy optimization result object containing
                fitted parameters and optimization diagnostics.
        """
        bounds = Bounds(
            lb=np.array([1e-8, -np.inf, -np.inf, -np.inf]), ub=np.repeat(np.inf, 4)
        )
        constraints = [{"type": "ineq", "fun": lambda x: sum(x[1:4])}]
        res = minimize(
            self._fitness,
            initial_guess,
            args=(data,),
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"disp": display},
        )
        if res.success:
            self.set_params(res.x)
            self.error_dist = self._init_dist(res.x)
        return res
