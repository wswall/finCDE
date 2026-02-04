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
from scipy.optimize import minimize, OptimizeResult
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
    """Abstract base class for model specifications.

    Implementations define the conditional mean/volatility dynamics and
    how to construct fit-time data for a given error distribution.

    Attributes:
        error_dist (type): Distribution class used to model standardized errors.
        bounds (Sequence[tuple[float, float]] | None): Optional parameter bounds
            used by optimizers.
        constraints (Sequence | None): Optional optimizer constraints.
    """

    def __init__(self, dist_class):
        """Initialize the model specification.

        Args:
            dist_class (type): Distribution class that will be instantiated
                with a ``FitData`` object during fitting/scoring.
        """
        self.error_dist = dist_class
        self.bounds = None
        self.constraints = None

    @abstractmethod
    def __call__(self, data: np.ndarray, x: Array1D) -> np.ndarray:
        """Compute the model-implied series for the given parameters.

        Args:
            data (np.ndarray): Input data used by the model.
            x (Array1D): Parameter vector.

        Returns:
            np.ndarray: Model-implied series (e.g., mean/volatility output).
        """
        raise NotImplementedError

    @abstractmethod
    def make_initial_guess(self, data) -> np.ndarray:
        """Provide a default parameter initialization.

        Returns:
            np.ndarray: Initial parameter vector.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def base_step(self) -> np.ndarray:
        """Provide base step sizes for numerical differentiation.

        Returns:
            np.ndarray: Step sizes aligned with the parameter vector.
        """
        raise NotImplementedError

    @abstractmethod
    def make_fit_data(self, data: np.ndarray, x: Array1D) -> FitData:
        """Construct fit data required by the error distribution.

        Args:
            data (np.ndarray): Input data used by the model.
            x (Array1D): Parameter vector.

        Returns:
            FitData: Residuals, volatility, and optional shape parameters.
        """
        raise NotImplementedError

    def score(self, x: Array1D, *args) -> Array1D:
        """Compute per-observation negative log-density scores.

        Args:
            x (Array1D): Parameter vector.
            *args: Positional arguments where the first item is the data
                array used by ``make_fit_data``.

        Returns:
            Array1D: Negative log-density contributions for each observation.
        """
        data = args[0]
        fit_data = self.make_fit_data(data, x)
        error_dist = self.error_dist(fit_data)
        return -(np.log(error_dist.pdf()) - np.log(fit_data.sigma))

    def fitness(self, x: Array1D, *args) -> float:
        """Compute the scaled negative log-likelihood objective.

        Args:
            x (Array1D): Parameter vector.
            *args: Positional arguments where the first item is the data
                array used by ``make_fit_data``.

        Returns:
            float: Scaled negative log-likelihood for optimization.
        """
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
        jacobian_func: Callable,
        hess_func: Callable
    ):
        self.result = result
        self.fit_data = fit_data
        self.error_dist = error_dist
        self.jacobian = jacobian_func
        self.hessian = hess_func

    def __getstate__(self):
        # Jacobian and Hessian callables are lambdas and not picklable for multiprocessing
        state = self.__dict__.copy()
        if isinstance(self.jacobian, Callable):
            del state["jacobian"]
        if isinstance(self.hessian, Callable):
            del state["hessian"]
        return state

    def compute_jacobian(self):
        self.jacobian = self.jacobian(self.result.x) / LLH_SCALING

    def compute_hessian(self):
        self.hessian = self.hessian(self.result.x) / LLH_SCALING

    @cached_property
    def log_likelihood(self) -> float:
        """Log-likelihood of the fitted model."""
        return -(self.result.fun / LLH_SCALING)

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
        return self.log_likelihood / len(self.fit_data.e)

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
    ):
        """Fit the model to data using maximum likelihood.

        The optimization is performed using the Sequential Least Squares
        Programming (SLSQP) method. If the optimization is successful,
        the model parameters are updated and diagnostics are computed.

        Args:
            data (np.ndarray[float]): Observed data to fit the model to.
            model_spec (ModelSpec): Specification of the model to be fitted.
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
            if compute_derivs:
                model_fit.compute_jacobian()
                model_fit.compute_hessian()
            return cls(model_spec, fit_data, result.x, model_fit)
        return result


class ModelFactory:
    """Factory for building models from a specification class.

    Handles spec initialization defaults and supports building many
    models in parallel from configuration dictionaries.

    Attributes:
        model_class (type): Model specification class to instantiate.
        defaults (dict): Default keyword arguments for spec construction.
    """

    def __init__(self, model_class, **kwargs):
        """Initialize the factory.

        Args:
            model_class (type): Model specification class to instantiate.
            **kwargs: Default keyword arguments for spec construction.
        """
        self.model_class = model_class
        self.defaults = kwargs or {}

    def _init_spec(self, **kwargs):
        kwargs.update(self.defaults)
        return self.model_class(**kwargs)

    @staticmethod
    def _extract_kw_args(kw_list, **kwargs):
        """Split kwargs into spec args and fit args.

        Args:
            kw_list (Sequence[str]): Keys to extract into a separate dict.
            **kwargs: Input keyword arguments.

        Returns:
            tuple[dict, dict]: Remaining kwargs and extracted kwargs.
        """
        extracted = {}
        for key in kw_list:
            if key in kwargs:
                extracted[key] = kwargs.pop(key)
        return kwargs, extracted

    def build(self, data, **kwargs):
        """Build and fit a model from data and configuration kwargs.

        Args:
            data (np.ndarray): Input data to fit the model to.
            **kwargs: Spec configuration plus optional fit args
                (display, ftol, maxiter).

        Returns:
            Model | OptimizeResult: Fitted model or raw optimizer result.
        """
        spec_args, fit_args = self._extract_kw_args(
            ["display", "ftol", "maxiter", 'compute_derivs'], **kwargs
        )
        model_spec = self._init_spec(**spec_args)
        return Model.fit(data, model_spec, **fit_args)

    def _worker(self, config):
        data = config.pop("data")
        return self.build(data, **config)

    def build_many(self, config_list, proc_count=1):
        """Build and fit multiple models from a list of configs.

        Args:
            config_list (Sequence[dict]): Each dict must include ``data`` and
                any spec/fit args for ``build``.
            proc_count (int, optional): Number of processes to use. Defaults
                to 1 (no multiprocessing).

        Returns:
            list[Model | OptimizeResult]: List of fitted models or optimizer
                results corresponding to the input configs.
        """
        # Method to build multiple models in parallel
        if proc_count > 1:
            with Pool(proc_count) as p:
                results = p.map(self._worker, config_list)
        else:
            results = [self._worker(config) for config in config_list]
        return results
