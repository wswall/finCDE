from __future__ import annotations

import numdifftools as nd
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import tanhsinh
from scipy.optimize import minimize, OptimizeResult

from density_estimation.common import LLH_SCALING, OFFSET
from density_estimation.core.spec import ModelSpec
from density_estimation.core.model_fit import ModelFit


class Model:
    """Fitted model container with parameters and diagnostics.

    Stores the fitted parameters, derived fit data, and diagnostics
    (e.g., log-likelihood, AIC/BIC) produced during optimization.

    Args:
        model_spec (ModelSpec): Model specification used for fitting.
        fit_data (FitData): Fit data derived from the input series.
        parameters (np.ndarray): Estimated parameter vector.
        model_fit (ModelFit): Diagnostics and test utilities for the fit.

    Attributes:
        spec (ModelSpec): Model specification used for fitting.
        data (FitData): Fit data derived from the input series.
        parameters (np.ndarray): Estimated parameter vector.
        evaluate (ModelFit): Diagnostics and test utilities for the fit.
    """

    def __init__(self, model_spec, fit_data, parameters, model_fit):
        self.spec = model_spec
        self.data = fit_data
        self.parameters = parameters
        self.evaluate = model_fit

    @classmethod
    def fit(
        cls,
        data: NDArray,
        model_spec: ModelSpec,
        compute_derivs=False,
        display: bool = True,
        ftol: float = OFFSET,
        maxiter: int = 100,
    ) -> Model | OptimizeResult:
        """Fit the model to data using maximum likelihood.

        The optimization is performed using the Sequential Least Squares
        Programming (SLSQP) method. If the optimization is successful,
        the model parameters are updated and diagnostics are computed.

        Args:
            data (np.ndarray[float]): Observed data to fit the model to.
            model_spec (ModelSpec): Specification of the model to be fitted.
            compute_derivs (bool, optional): Whether to compute the
                Jacobian and Hessian matrices before returning the
                object. Note that the multiprocessing module cannot
                serialize the Jacobian and Hessian callables, so they
                removed when fitting in parallel if they are not computed.
                Defaults to False.
            display (bool, optional): Whether to display optimization
                progress. Defaults to True.
            ftol (float, optional): The tolerance for termination by the
                change in the function value. Defaults to OFFSET.
            maxiter (int, optional): The maximum number of iterations for
                the optimizer. Defaults to 100.

        Returns:
            Model | OptimizeResult: A fitted ``Model`` when optimization
                succeeds; otherwise the raw optimizer result.
        """
        result = minimize(
            model_spec.fitness,
            model_spec.make_initial_guess(data),
            args=(data,),
            method="SLSQP",
            bounds=model_spec.bounds,
            constraints=model_spec.constraints,
            options={"disp": display, "ftol": ftol, "maxiter": maxiter},
        )
        if result.success:
            fit_data = model_spec.make_fit_data(data, result.x)
            jacobian = nd.Jacobian(
                lambda x: model_spec.score(x, data),
                base_step=model_spec.base_step,
            )
            hessian = nd.Hessian(
                lambda x: model_spec.fitness(x, data) / LLH_SCALING,
                base_step=model_spec.base_step,
            )
            model_fit = ModelFit(result, fit_data.e, jacobian, hessian)
            if compute_derivs:
                model_fit.compute_jacobian()
                model_fit.compute_hessian()
            return cls(model_spec, fit_data, result.x, model_fit)
        return result

    def calc_log_score(self, data: NDArray) -> NDArray:
        """Calculate the log scores for out-of-sample data"""
        return self.spec.score(self.parameters, data)

    @staticmethod
    def _make_q_score(error_dist):
        d = error_dist

        def q_score(alpha, y, mu, sigma):
            quantile = mu + sigma * d.ppf(alpha)
            return ((y <= quantile) - alpha) * (quantile - y)

        return q_score

    def calc_crps(
        self,
        data: NDArray,
        tol: float = 1e-10,
        maxlevel: int = 12,
    ) -> float:
        """Compute the Continuous Ranked Probability Score

        Given array of observations, computes the Continuous Ranked
        Probability Score (CRPS) using the model's fitted parameters and
        distributional assumption. This method uses the tanh-sinh
        quadrature method for numerical integration of the quantile score
        over the interval (0, 1).

        Args:
            data (np.ndarray[float]): Out of sample observations
            tol (float, optional): Absolute tolerance for numerical
                integration. Default is 1e-10.
            maxlevel (int, optional): Maximum level of refinement for
                numerical integration. Default is 12.

        Returns:
            float: CRPS values for each observation.
        """
        fit_data = self.spec.make_fit_data(data, self.parameters)
        error_dist = self.spec.error_dist(fit_data)
        q_score = self._make_q_score(error_dist)
        t = len(fit_data.y)
        mu = fit_data.y - fit_data.e
        results = np.zeros(t)
        for i in range(t):
            result = tanhsinh(
                q_score,
                0.0,
                1.0,
                args=(fit_data.y[i], mu[i], fit_data.sigma[i]),
                atol=tol,
                maxlevel=maxlevel
            )
            results[i] = 2.0 * result.integral
        return results

