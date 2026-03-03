from __future__ import annotations
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import Bounds
from scipy.stats import rv_continuous

from density_estimation.common import sumjit, Array1D


class FitData:
    """Container for data used in model fitting and evaluation.

    Holds the target series, residuals, conditional volatility, and standardized
    residuals, as well as shape parameters for the distribution.

    Attributes:
        y (Array1D): Target time series (e.g., returns).
        e (Array1D): Residuals (y - mean).
        sigma (Array1D): Conditional volatility.
        z (Array1D): Standardized residuals (e / sigma).
        shape_params (dict): Dictionary of shape parameters 'xi' and 'nu'.
    """

    def __init__(
        self,
        y: Array1D,
        e: Array1D,
        sigma: Array1D,
        z: Array1D | None = None,
        xi: ArrayLike | None = None,
        nu: ArrayLike | None = None,
    ):
        """Initializes fit data.

        Args:
            y (Array1D): Target time series.
            e (Array1D): Residuals.
            sigma (Array1D): Conditional volatility.
            z (Array1D | None, optional): Standardized residuals. Computed if None.
            xi (ArrayLike | None, optional): Skewness parameter.
            nu (ArrayLike | None, optional): Shape parameter (e.g. degrees of freedom).
        """
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
    def nu(self) -> ArrayLike | None:
        """Shape parameter (e.g. degrees of freedom)."""
        return self.shape_params["nu"]

    @property
    def xi(self) -> ArrayLike | None:
        """Skewness parameter."""
        return self.shape_params["xi"]

    def set_shape_param(self, param: str, value: float | ArrayLike) -> None:
        """Update a shape parameter.

        Args:
            param (str): Parameter name ('xi' or 'nu').
            value (float | ArrayLike): Parameter value.
        """
        value = np.array(value) if isinstance(value, Sequence) else value
        self.shape_params[param] = value

    @property
    def conditional_tail(self) -> bool:
        """Whether the tail parameter is time-varying."""
        return (
            isinstance(self.shape_params["nu"], np.ndarray)
            and len(self.shape_params["nu"]) > 1
        )

    @property
    def conditional_skew(self) -> bool:
        """Whether the skewness parameter is time-varying."""
        return (
            isinstance(self.shape_params["xi"], np.ndarray)
            and len(self.shape_params["xi"]) > 1
        )


class Distribution(metaclass=ABCMeta):
    """Base class for conditional distributions"""

    n_params = 0
    bounds = Bounds(lb=[], ub=[])
    initial_guess = np.array([])
    base_step = np.array([])

    def __init__(self, params: FitData):
        """Initializes the distribution.

        Args:
            params (FitData): Data and parameters for evaluation.
        """
        self.params = params

    @abstractmethod
    def ppf(self, alpha: ArrayLike, scale: ArrayLike = 1.0) -> ArrayLike:
        """Percent point function (inverse of cdf).

        Args:
            alpha (ArrayLike): Probability.
            scale (ArrayLike, optional): Scale parameter. Defaults to 1.0.
        """
        raise NotImplementedError

    @abstractmethod
    def pdf(self) -> ArrayLike:
        """Probability density function evaluated at standardized residuals.

        Returns:
            ArrayLike: PDF values.
        """
        raise NotImplementedError

    @abstractmethod
    def llh(self) -> float:
        """Log-likelihood of the data given the distribution.

        Returns:
            float: Log-likelihood value.
        """
        raise NotImplementedError


class SymmetricDistribution(Distribution):
    """Base class for symmetric distributions"""

    def __init__(self, params: FitData, base_dist: rv_continuous):
        """Initializes a symmetric distribution.

        Args:
            params (FitData): Data and parameters.
            base_dist (rv_continuous): Underlying scipy.stats distribution.
        """
        super().__init__(params)
        self.base_dist = base_dist

    def ppf(self, alpha: ArrayLike, scale: ArrayLike = 1.0) -> ArrayLike:
        """Percent point function."""
        return self.base_dist.ppf(alpha, scale=scale)

    def pdf(self) -> ArrayLike:
        """Probability density function."""
        return self.base_dist.pdf(self.params.z)

    def llh(self) -> float:
        """Log-likelihood."""
        return sumjit(np.log(self.pdf()))


class SkewedDistribution(Distribution):
    """Base class for skewed distributions

    Uses the Wurtz et al. reparametrization of Fernandez and Steel to
    skew a symmetric distribution. The skewness parameter must be > 0,
    with a value of 1 yielding the symmetric distribution. Density,
    quantile and likelihood are calculated as transformations of their
    outputs from the underlying symmetric distribution.
    """

    n_params = 1
    bounds = Bounds(lb=[1e-5], ub=[np.inf])
    initial_guess = np.array([1])
    base_step = np.array([1e-6])

    def __init__(self, params: FitData, base_dist: SymmetricDistribution, m1: float):
        """Initializes a skewed distribution.

        Args:
            params (FitData): Data and parameters.
            base_dist (SymmetricDistribution): Base symmetric distribution to skew.
            m1 (float): First absolute moment of the base distribution.
        """
        super().__init__(params)
        self._validate_xi()
        self.base_dist = base_dist
        self.g = 2.0 / (self.params.xi + 1.0 / self.params.xi)
        self.mu_xi = self._calc_mu_xi(m1)
        self.sigma_xi = self._calc_sigma_xi(m1)
        z_xi = self.params.z * self.sigma_xi + self.mu_xi
        self.params.z = z_xi / self.params.xi ** np.sign(z_xi)

    def _validate_xi(self) -> bool:
        if self.params.xi is None:
            raise ValueError(
                "Skewness parameter 'xi' must be provided for any skewed distribution."
            )
        elif isinstance(self.params.xi, (float, int)) and self.params.xi <= 0.0:
            raise ValueError("Skewness parameter 'xi' must be greater than 0.")
        elif isinstance(self.params.xi, np.ndarray) and np.any(self.params.xi <= 0.0):
            raise ValueError(
                "All values of skewness parameter 'xi' must be greater than 0."
            )
        return True

    def _calc_mu_xi(self, m1: float) -> ArrayLike:
        return m1 * (self.params.xi - 1.0 / self.params.xi)

    def _calc_sigma_xi(self, m1: float) -> ArrayLike:
        xi_sq = self.params.xi**2.0
        m1_sq = m1**2.0
        return np.sqrt((1.0 - m1_sq) * (xi_sq + 1.0 / xi_sq) + 2.0 * m1_sq - 1.0)

    def ppf(self, alpha: ArrayLike, scale: float = 1.0) -> ArrayLike:
        """Percent point function."""
        z = alpha - (1.0 / (1.0 + self.params.xi**2.0))
        big_xi = self.params.xi ** np.sign(z)
        alpha_skew = (np.heaviside(z, 0.0) - np.sign(z) * alpha) / (self.g * big_xi)
        ppf = self.base_dist.ppf(alpha_skew, scale=big_xi)
        return (-np.sign(z) * ppf - self.mu_xi) / self.sigma_xi

    def pdf(self) -> ArrayLike:
        """Probability density function."""
        return self.g * self.sigma_xi * self.base_dist.pdf()

    def llh(self) -> float:
        """Log-likelihood."""
        if not self.params.conditional_skew:
            return (
                self.params.z.size * np.log(self.g * self.sigma_xi)
                + self.base_dist.llh()
            )
        return (
            np.log(2) * len(self.params.z)
            + np.sum(
                np.log(self.sigma_xi) - np.log(self.params.xi - 1 / self.params.xi)
            )
            + self.base_dist.llh()
        )
