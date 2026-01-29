from __future__ import annotations
from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Sequence
from functools import cached_property
from multiprocessing import Pool

import numdifftools as nd
import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.integrate import tanhsinh
from scipy.stats import norm
from scipy.optimize import Bounds, minimize, OptimizeResult
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera

from density_estimation.common import LLH_SCALING, OFFSET, Array1D


class FitData:

    def __init__(
        self,
        y: Array1D,
        e: Array1D,
        sigma: Array1D,
        z: Array1D | None = None,
        xi: ArrayLike | None = None,
        nu: ArrayLike | None = None,
    ):
        self.y = y
        self.e = e
        self.sigma = sigma
        self.z = z if z is not None else self.e / self.sigma
        self.shape_params = {"xi": xi, "nu": nu}
        self._validate()

    def _validate(self):
        if self.e.shape != self.sigma.shape:
            raise ValueError("Residuals and volatility must be same shape.")
        for param in [self.nu, self.xi]:
            if isinstance(param, np.ndarray) and param.shape != self.e.shape:
                raise ValueError(
                    "Shape parameter arrays must be same shape as residuals."
                )
        return True

    @property
    def nu(self):
        return self.shape_params["nu"]

    @property
    def xi(self):
        return self.shape_params["xi"]

    def set_shape_param(self, param, value):
        value = np.array(value) if isinstance(value, Sequence) else value
        self.shape_params[param] = value

    @property
    def conditional_tail(self):
        return (
            isinstance(self.shape_params["nu"], np.ndarray)
            and len(self.shape_params["nu"]) > 1
        )

    @property
    def conditional_skew(self):
        return (
            isinstance(self.shape_params["xi"], np.ndarray)
            and len(self.shape_params["xi"]) > 1
        )


class ModelSpec(metaclass=ABCMeta):

    def __init__(self, dist_class):
        self.error_dist = dist_class
        self.bounds = None
        self.constraints = None

    @abstractmethod
    def __call__(self, data: np.ndarray, x: Array1D) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def initial_guess(self) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def base_step(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def make_fit_data(self, data: np.ndarray, x: Array1D) -> FitData:
        raise NotImplementedError

    def score(self, x: Array1D, *args) -> Array1D:
        data = args[0]
        fit_data = self.make_fit_data(data, x)
        error_dist = self.error_dist(fit_data)
        return -(np.log(error_dist.pdf()) - np.log(fit_data.sigma))

    def fitness(self, x: Array1D, *args) -> float:
        data = args[0]
        fit_data = self.make_fit_data(data, x)
        error_dist = self.error_dist(fit_data)
        llh = error_dist.llh() - np.sum(np.log(fit_data.sigma))
        return -llh * LLH_SCALING


class ModelFit:

    def __init__(
        self,
        result: OptimizeResult,
        fit_data: FitData,
        error_dist,
        jacobian: Callable,
        hessian: Callable,
    ):
        self.result = result
        self.fit_data = fit_data
        self.error_dist = error_dist
        self.jacobian_func = jacobian
        self.hess_func = hessian

    @cached_property
    def log_likelihood(self) -> float:
        """Log-likelihood of the fitted model."""
        return -(self.result.fun / LLH_SCALING)

    @cached_property
    def hessian(self) -> np.ndarray:
        """Evaluate the Hessian of the fitted model."""
        return self.hess_func(self.result.x) / LLH_SCALING

    @cached_property
    def jacobian(self) -> np.ndarray:
        return self.jacobian_func(self.result.x) / LLH_SCALING

    def aic(self) -> float:
        """Calculate the Akaike Information Criterion (AIC)."""
        n = len(self.fit_data.e)
        k = len(self.result.x)
        return (2 * k - 2 * self.log_likelihood) / n

    def bic(self) -> float:
        """Calculate the Bayesian Information Criterion (BIC)."""
        n = len(self.fit_data.e)
        k = len(self.result.x)
        return (np.log(n) * k - 2 * self.log_likelihood) / n

    def calc_standard_errors(self) -> np.ndarray:
        """Calculate standard errors of the fitted parameters."""
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
        results = acorr_ljungbox(
            self.fit_data.e, lags=lags, model_df=self.result.x.size
        )
        return results.values

    def arch_lm(self, lags: int) -> tuple[float, float]:
        """Perform the ARCH-LM test for conditional heteroskedasticity."""
        results = het_arch(self.fit_data.e)
        return results[0], results[1]

    def jarque_bera(self) -> tuple[float, float]:
        """Perform the Jarque-Bera test for normality."""
        results = jarque_bera(self.fit_data.e)
        return results[0], results[1]

    def log_score(self):
        return -self.log_likelihood / len(self.fit_data.e)

    def _q_score(self, alpha, y, mu, sigma):
        quantile = mu + sigma * self.error_dist.ppf(alpha)
        return ((y <= quantile) - alpha) * (quantile - y)

    def crps(
        self,
        tol: float = 1e-10,
        maxlevel: int = 12,
    ) -> float:
        """Compute the Continuous Ranked Probability Score

        Given arrays of observations and corresponding residuals and
        standard deviations, computes the Continuous Ranked Probability
        Score (CRPS). This method uses the tanh-sinh quadrature method
        for numerical integration of the quantile score over the
        interval (0, 1).

        Args:
            y (np.ndarray[float]): Observed values at each time t.
            residuals (np.ndarray[float]): Residuals at each time t.
            sigma (np.ndarray[float]): Standard deviations at each time t.
            tol (float, optional): Absolute tolerance for numerical
                integration. Default is 1e-10.
            maxlevel (int, optional): Maximum level of refinement for
                numerical integration. Default is 12.

        Returns:
            float: Mean of CRPS values for each observation.
        """
        t = len(self.fit_data.y)
        mu = self.fit_data.y + self.fit_data.e
        result = tanhsinh(
            self._q_score,
            np.zeros(t),
            np.ones(t),
            args=(self.fit_data.y, mu, self.fit_data.sigma),
            atol=tol,
            maxlevel=maxlevel,
        )
        return np.mean(2.0 * result.integral)


class Model:

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
        display: bool = True,
        ftol: float = OFFSET,
        maxiter: int = 100,
    ):
        """Fits the model to the given returns data using maximum likelihood.

        The optimization is performed using the Sequential Least Squares
        Programming (SLSQP) method. If the optimization is successful,
        the model parameters are updated, and the error distribution is
        initialized.

        Args:
            data (np.ndarray[float]): The observed returns data to fit
                the model to.
            initial_guess (np.ndarray[float]): Initial guess for the
                optimization parameters.
            model_spec (ModelSpec): The specification of the model to be fitted.
            display (bool, optional): Whether to display optimization
                progress. Defaults to True.
            ftol (float, optional): The tolerance for termination by the
                change in the function value. Defaults to OFFSET.
            maxiter (int, optional): The maximum number of iterations for
                the optimizer. Defaults to 100.

        Returns:
            OptimizeResult: The result of the optimization process.
        """
        result = minimize(
            model_spec.fitness,
            model_spec.initial_guess,
            args=(data,),
            method="SLSQP",
            bounds=model_spec.bounds,
            constraints=model_spec.constraints,
            options={"disp": display, "ftol": ftol, "maxiter": maxiter},
        )
        if result.success:
            fit_data = model_spec.make_fit_data(data, result.x)
            error_dist = model_spec.error_dist(fit_data)
            jacobian = nd.Jacobian(
                lambda x: model_spec.score(x, data),
                base_step=model_spec.base_step,
            )
            hessian = nd.Hessian(
                lambda x: model_spec.fitness(x, data) / LLH_SCALING,
                base_step=model_spec.base_step,
            )
            model_fit = ModelFit(result, fit_data, error_dist, jacobian, hessian)
            return cls(model_spec, fit_data, result.x, model_fit)
        return result


class ModelFactory:

    def __init__(self, model_class, **kwargs):
        self.model_class = model_class
        self.defaults = kwargs or {}

    def _init_spec(self, **kwargs):
        kwargs.update(self.defaults)
        return self.model_class(**kwargs)

    @staticmethod
    def _extract_kw_args(kw_list, **kwargs):
        extracted = {}
        for key in kw_list:
            if key in kwargs:
                extracted[key] = kwargs.pop(key)
        return kwargs, extracted

    def build(self, data, **kwargs):
        spec_args, fit_args = self._extract_kw_args(
            ["display", "ftol", "maxiter"], **kwargs
        )
        model_spec = self._init_spec(**spec_args)
        return Model.fit(data, model_spec, **fit_args)

    def _worker(self, config):
        data = config.pop("data")
        return self.build(data, **config)

    def build_many(self, config_list, proc_count=1):
        # Method to build multiple models in parallel
        if proc_count > 1:
            with Pool(proc_count) as p:
                results = p.map(self._worker, config_list)
        else:
            results = [self._worker(config) for config in config_list]
        return results
