from collections.abc import Callable
from functools import cached_property

import numpy as np
from scipy.stats import norm
from scipy.optimize import OptimizeResult
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera

from density_estimation.common import LLH_SCALING, Array1D


class ModelFit:
    """Class to hold and analyze the results of a model fit.

    Stores optimization results, residuals, and provides methods for
    post-estimation diagnostics and statistics.

    Attributes:
        result (OptimizeResult): The result object from scipy.optimize.
        residuals (Array1D): Array of model residuals.
        jacobian (Callable | np.ndarray): Function to compute Jacobian or the computed Jacobian.
        hessian (Callable | np.ndarray): Function to compute Hessian or the computed Hessian.
    """

    def __init__(
        self,
        result: OptimizeResult,
        residuals: Array1D,
        jacobian_func: Callable,
        hess_func: Callable,
    ):
        """Initialize the ModelFit object.

        Args:
            result (OptimizeResult): Optimization result.
            residuals (Array1D): Model residuals.
            jacobian_func (Callable): Function that calculates the Jacobian matrix.
            hess_func (Callable): Function that calculates the Hessian matrix.
        """
        self.result = result
        self.residuals = residuals
        self.jacobian = jacobian_func
        self.hessian = hess_func

    def __getstate__(self):
        # Jacobian & Hessian callables are lambdas, can't be serialized for multiproc
        state = self.__dict__.copy()
        if isinstance(self.jacobian, Callable):
            del state["jacobian"]
        if isinstance(self.hessian, Callable):
            del state["hessian"]
        return state

    def compute_jacobian(self) -> None:
        """Compute the Jacobian derived from the score function.

        Evaluates the Jacobian with the fitted parameters and updates
        self.jacobian with the calculated array.
        """
        self.jacobian = self.jacobian(self.result.x) / LLH_SCALING

    def compute_hessian(self) -> None:
        """Compute the Hessian derived from the likelihood function.

        Evaluates the Hessian with the fitted parameters and updates
        self.hessian with the calculated array.
        """
        self.hessian = self.hessian(self.result.x) / LLH_SCALING

    @cached_property
    def log_likelihood(self) -> float:
        """Log-likelihood of the fitted model."""
        return -(self.result.fun / LLH_SCALING)

    def aic(self) -> float:
        """Calculate the Akaike Information Criterion (AIC)."""
        n = len(self.residuals)
        k = len(self.result.x)
        return (2 * k - 2 * self.log_likelihood) / n

    def bic(self) -> float:
        """Calculate the Bayesian Information Criterion (BIC)."""
        n = len(self.residuals)
        k = len(self.result.x)
        return (np.log(n) * k - 2 * self.log_likelihood) / n

    def calc_standard_errors(self) -> np.ndarray:
        """Calculate standard errors of the fitted parameters."""
        if isinstance(self.jacobian, Callable):
            self.compute_jacobian()
        if isinstance(self.hessian, Callable):
            self.compute_hessian()
        B = np.linalg.inv(self.hessian)
        M = np.cov(self.jacobian.T)
        return np.sqrt(np.diag(B @ M @ B))

    def significance_test(self) -> np.ndarray:
        """Perform significance tests on the fitted parameters."""
        se = self.calc_standard_errors()
        t_val = self.result.x / se
        return 2 * (1 - norm.cdf(np.abs(t_val)))

    def ljung_box(self, lags: int) -> np.ndarray:
        """Perform the Ljung-Box test for autocorrelation."""
        results = acorr_ljungbox(self.residuals, lags=lags, model_df=self.result.x.size)
        return results.values

    def arch_lm(self, lags: int) -> tuple[float, float]:
        """Perform the ARCH-LM test for conditional heteroskedasticity."""
        results = het_arch(self.residuals)
        return results[0], results[1]

    def jarque_bera(self) -> tuple[float, float]:
        """Perform the Jarque-Bera test for normality on residuals."""
        results = jarque_bera(self.residuals)
        return results[0], results[1]
