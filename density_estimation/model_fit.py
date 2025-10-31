from functools import cached_property

import numpy as np
from scipy.stats import norm
from scipy.optimize import OptimizeResult
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera

from density_estimation.common import LLH_SCALING


class ModelFit:

    def __init__(self, result, residuals, hess_func):
        self.result = result
        self.residuals = residuals
        self.hess_func = hess_func

    @cached_property
    def log_likelihood(self) -> float:
        """Log-likelihood of the fitted model."""
        return -self.result.fun / LLH_SCALING

    @cached_property
    def hessian(self) -> float:
        """Evaluate the Hessian of the fitted model."""
        return self.hess_func(self.result.x) * LLH_SCALING

    @cached_property
    def jacobian(self) -> np.ndarray[float]:
        return self.jacobian_func(self.result.x) * LLH_SCALING

    def calc_standard_errors(self) -> np.ndarray[float]:
        """Calculate standard errors of the fitted parameters."""
        B = np.linalg.inv(self.hessian)
        M = np.cov(self.jacobian.T)
        return np.sqrt(np.diag(B @ M @ B))

    def significance_test(self) -> dict[str, float]:
        """Perform significance tests on the fitted parameters."""
        se = self.calc_standard_errors()
        t_val = self.result.x / se
        return 2 * (1 - norm.cdf(np.abs(t_val)))

    def ljung_box(self, lags: int) -> float:
        """Perform the Ljung-Box test for autocorrelation."""
        results = acorr_ljungbox(self.residuals, lags=lags)
        return results.values

    def arch_lm(self, lags: int) -> float:
        """Perform the ARCH-LM test for conditional heteroskedasticity."""
        results = het_arch(self.residuals, lags=lags)
        return results[0], results[1]

    def jarque_bera(self) -> float:
        """Perform the Jarque-Bera test for normality."""
        results = jarque_bera(self.residuals)
        return results[0], results[1]

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
