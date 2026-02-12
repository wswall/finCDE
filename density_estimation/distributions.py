from __future__ import annotations
from math import gamma, lgamma

from numba import njit, f8, prange
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import Bounds
from scipy.stats import laplace, norm, t

from density_estimation.common import sumjit, OFFSET
from density_estimation.core import (
    FitData,
    Distribution,
    SymmetricDistribution,
    SkewedDistribution,
)


__all__ = [
    "Normal",
    "Laplace",
    "StudentT",
    "SkewNorm",
    "SkewLaplace",
    "SkewT",
    "JohnsonSU",
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


class JohnsonSU(Distribution):
    """Johnson's SU distribution for use with standardized residuals"""

    name = "jsu"
    n_params = 2
    bounds = Bounds(lb=[-10, 0.5], ub=[10, 50])
    initial_guess = np.array([0.0, 2.0])
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
        big_z = self.params.xi + self.params.nu * np.asinh(z)
        num = np.exp(-0.5 * big_z**2.0) * self.params.nu
        denom = self.scale * np.sqrt(2.0 * np.pi * (1.0 + z**2.0))
        return num / denom

    def ppf(self, alpha: ArrayLike) -> ArrayLike:
        z = np.sinh((1.0 / self.params.nu) * (norm.ppf(alpha) + self.params.xi))
        return self.loc + self.scale * z

    def llh(self) -> float:
        return sumjit(np.log(self.pdf()))
