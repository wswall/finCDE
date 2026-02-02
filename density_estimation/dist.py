from __future__ import annotations
from abc import ABCMeta, abstractmethod
from math import gamma, lgamma

from numba import njit, f8, prange
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import Bounds
from scipy.stats import laplace, norm, t, rv_continuous

from density_estimation.common import sumjit, OFFSET
from density_estimation.base import FitData


__all__ = [
    "Distribution",
    "SkewedDistribution",
    "Normal",
    "Laplace",
    "StudentT",
    "SkewNorm",
    "SkewLaplace",
    "SkewT",
    "JohnsonSU",
    "jsu_constraint",
]


@njit(f8(f8, f8[:]), fastmath=True)
def _norm_llh(c, z_std):
    """Numba-accelerated log-likelihood for normal distribution."""
    acc = 0.0
    for i in prange(z_std.size):
        acc += z_std[i] ** 2
    return -0.5 * (c + acc)


@njit(f8(f8, f8[:]), fastmath=True)
def _laplace_llh(c, z_std):
    """Numba-accelerated log-likelihood for Laplace distribution."""
    acc = 0.0
    for i in prange(z_std.size):
        acc += np.abs(z_std[i])
    return -c - acc


@njit(f8(f8, f8, f8[:]), fastmath=True)
def _t_llh(c, nu, z_std):
    """Numba-accelerated log-likelihood for t distribution."""
    acc = 0.0
    div = nu - 2.0
    for i in prange(z_std.size):
        acc += np.log(1.0 + (z_std[i] ** 2) / div)
    return c - 0.5 * (nu + 1.0) * acc


@njit(f8(f8[:], f8[:]), fastmath=True)
def _cond_t_llh(nu, z_std):
    """Numba-accelerated log-likelihood for t distribution w/ conditional DoF."""
    acc = 0.0
    for i in prange(z_std.size):
        acc += (
            lgamma((nu[i] + 1.0) / 2.0)
            - lgamma(nu[i] / 2.0)
            - np.log(np.pi * (nu[i] - 2.0)) / 2.0
            - 0.5 * (nu[i] + 1.0) * np.log(1.0 + (z_std[i] ** 2) / (nu[i] - 2.0))
        )
    return acc


class Distribution(metaclass=ABCMeta):
    """Base class for conditional distributions"""

    n_params = 0
    bounds = Bounds(lb=[], ub=[])
    initial_guess = np.array([])
    base_step = np.array([])

    def __init__(self, params: FitData):
        self.params = params

    @abstractmethod
    def ppf(self, alpha: ArrayLike, scale: ArrayLike = 1.0):
        raise NotImplementedError

    @abstractmethod
    def pdf(self) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def llh(self) -> float:
        raise NotImplementedError


class SymmetricDistribution(Distribution):
    """Base class for symmetric distributions"""

    def __init__(self, params: FitData, base_dist: rv_continuous):
        super().__init__(params)
        self.base_dist = base_dist

    def ppf(self, alpha: ArrayLike, scale: ArrayLike = 1.0) -> ArrayLike:
        return self.base_dist.ppf(alpha, scale=scale)

    def pdf(self) -> ArrayLike:
        return self.base_dist.pdf(self.params.z)

    def llh(self) -> float:
        return sumjit(np.log(self.pdf()))


class Normal(SymmetricDistribution):
    """Normal distribution for use with standardized residuals"""

    name = "normal"

    def __init__(self, params: FitData):
        super().__init__(params, norm)

    def llh(self) -> float:
        c = self.params.z.size * np.log(2.0 * np.pi)
        return _norm_llh(c, self.params.z)


class Laplace(SymmetricDistribution):
    """Laplace distribution for use with standardized residuals"""

    name = "laplace"

    def __init__(self, params):
        super().__init__(params, laplace)

    def llh(self) -> float:
        c = self.params.z.size * np.log(2.0)
        return _laplace_llh(c, self.params.z)


class StudentT(SymmetricDistribution):
    """Student's t distribution for use with standardized residuals"""

    name = "t"
    n_params = 1
    bounds = Bounds(lb=[2.0 + OFFSET], ub=[60.0])
    initial_guess = np.array([4.0 + OFFSET])
    base_step = np.ones(1)

    def __init__(self, params: FitData):
        if params.nu is None:
            raise ValueError(
                "Parameter 'nu' must be provided in params object for T distributions."
            )
        super().__init__(params, t)

    def ppf(self, alpha: ArrayLike, scale: float = 1.0) -> ArrayLike:
        return self.base_dist.ppf(alpha, df=self.params.nu, scale=scale)

    def pdf(self) -> ArrayLike:
        return self.base_dist.pdf(self.params.z, df=self.params.nu)

    def _calc_llh_constant(self) -> float:
        return self.params.z.size * (
            lgamma((self.params.nu + 1.0) / 2.0)
            - lgamma(self.params.nu / 2.0)
            - np.log(np.pi * (self.params.nu - 2.0)) / 2.0
        )

    def llh(self) -> float:
        if not self.params.conditional_tail:
            c = self._calc_llh_constant()
            return _t_llh(c, self.params.nu, self.params.z)
        return _cond_t_llh(self.params.nu, self.params.z)


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
        super().__init__(params)
        self._validate_xi()
        self.base_dist = base_dist
        self.g = 2.0 / (self.params.xi + 1.0 / self.params.xi)
        self.mu_xi = self._calc_mu_xi(m1)
        self.sigma_xi = self._calc_sigma_xi(m1)
        z_xi = self.params.z * self.sigma_xi + self.mu_xi
        self.skew_z = z_xi / self.params.xi ** np.sign(z_xi)

    def _validate_xi(self):
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
        z = alpha - (1.0 / (1.0 + self.params.xi**2.0))
        big_xi = self.params.xi ** np.sign(z)
        alpha_skew = (np.heaviside(z, 0.0) - np.sign(z) * alpha) / (self.g * big_xi)
        ppf = self.base_dist.ppf(alpha_skew, scale=big_xi)
        return (-np.sign(z) * ppf - self.mu_xi) / self.sigma_xi

    def pdf(self) -> ArrayLike:
        return self.g * self.base_dist.pdf() * self.sigma_xi

    def llh(self) -> float:
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


class SkewNorm(SkewedDistribution):
    """Skewed normal distribution for use with standardized residuals"""

    name = "skewnorm"

    def __init__(self, params):
        m1 = 2.0 / np.sqrt(2.0 * np.pi)
        super().__init__(params, Normal(params), m1)


class SkewLaplace(SkewedDistribution):
    """Skewed Laplace distribution for use with standardized residuals"""

    name = "skewlap"

    def __init__(self, params):
        m1 = np.sqrt(0.5)
        super().__init__(params, Laplace(params), m1)


class SkewT(SkewedDistribution):
    """Skewed t distribution for use with standardized residuals"""

    name = "skewt"
    n_params = 2
    bounds = Bounds(lb=[1e-5, 2 + 1e-8], ub=[np.inf, 60.0])
    initial_guess = np.array([0.8, 4 + OFFSET])
    base_step = np.array([1e-6, 1.0])

    def __init__(self, params):
        betafn = (gamma(0.5) / gamma(0.5 + params.nu / 2.0)) * gamma(params.nu / 2.0)
        m1 = 2.0 * np.sqrt(params.nu - 2.0) / (params.nu - 1.0) / betafn
        super().__init__(params, StudentT(params), m1)


def jsu_constraint(x: ArrayLike) -> float:
    return np.abs(x[-2]) - OFFSET


class JohnsonSU(Distribution):
    """Johnson's SU distribution for use with standardized residuals"""

    name = "jsu"
    n_params = 2
    bounds = Bounds(lb=[-20, 0.25], ub=[20, 10])
    initial_guess = np.array([1.0, 1.0])
    base_step = np.array([0.5, 0.5])

    def __init__(self, params):
        if params.nu is None:
            raise ValueError(
                "Parameter 'nu' must be provided in params object for JSU distribution."
            )
        if params.xi is None:
            raise ValueError(
                "Parameter 'xi' must be provided in params object for JSU distribution."
            )
        super().__init__(params)
        w = np.exp(1.0 / params.nu**2.0)
        big_w = params.xi / params.nu
        self.scale = np.sqrt(2.0 / ((w - 1.0) * (w * np.cosh(2.0 * big_w) + 1.0)))
        self.loc = self.scale * np.sqrt(w) * np.sinh(big_w)

    def pdf(self) -> ArrayLike:
        z = (self.params.z - self.loc) / self.scale
        big_z = self.params.xi + self.params.nu * np.log(z + np.sqrt(z**2.0 + 1.0))
        num = np.exp(-0.5 * big_z**2.0) * self.params.nu
        denom = self.scale * np.sqrt(2.0 * np.pi * (1.0 + z**2.0))
        return num / denom

    def ppf(self, alpha: ArrayLike) -> ArrayLike:
        z = np.sinh((1.0 / self.params.nu) * (norm.ppf(alpha) + self.params.xi))
        return self.loc + self.scale * z

    def llh(self) -> float:
        return sumjit(np.log(self.pdf()))
